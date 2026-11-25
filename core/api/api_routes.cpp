// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Route handlers for the REST API — see docs/api-rest.md section 6.
 *
 * Each handler collects a snapshot (api_model.h), builds JSON from it and
 * returns an HTTP status. Handlers never touch the socket.
 *
 * Routes this build does not serve are registered available = false, so they
 * answer 501 and the capability list stays truthful.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "miner.h"
#include "algos.h"
#include "api_metrics.h"
#include "api_model.h"
#include "api_control.h"
#include "api_routes.h"

extern struct stratum_ctx stratum;

/* ---------------------------------------------------------------- helpers */

/* Every reply carries the miner envelope (docs/api-rest.md section 5). */
static json_t *envelope(void)
{
	json_t *root = json_object();
	if (!root) return NULL;
	json_t *m = api_build_miner_json();
	if (!m) { json_decref(root); return NULL; }
	json_object_set_new(root, "miner", m);
	return root;
}

static int oom(char *errmsg, size_t errlen)
{
	snprintf(errmsg, errlen, "out of memory");
	return 500;
}

/* ?id=N / ?index=N, or the trailing path segment of a prefix route. */
static bool query_int(const char *query, const char *key, long *out)
{
	size_t klen = strlen(key);
	for (const char *p = query; p && *p; ) {
		if (strncmp(p, key, klen) == 0 && p[klen] == '=') {
			char *end = NULL;
			long v = strtol(p + klen + 1, &end, 10);
			if (end == p + klen + 1) return false;
			*out = v;
			return true;
		}
		p = strchr(p, '&');
		if (p) p++;
	}
	return false;
}

static bool trailing_int(const char *path, long *out)
{
	const char *slash = strrchr(path, '/');
	if (!slash || !slash[1]) return false;
	char *end = NULL;
	long v = strtol(slash + 1, &end, 10);
	if (end == slash + 1 || *end) return false;
	*out = v;
	return true;
}

/* ---------------------------------------------------------------- handlers */

static int h_index(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);

	json_t *index = json_object();
	json_t *caps = json_array();
	json_t *links = json_object();
	if (!index || !caps || !links) {
		json_decref(root);
		if (index) json_decref(index);
		if (caps) json_decref(caps);
		if (links) json_decref(links);
		return oom(e, n);
	}

	/* Derived from the table, so it cannot claim a capability that is not
	 * routed, nor omit one that is. */
	size_t count = 0;
	const api_route *routes = api_routes_get(&count);
	for (size_t i = 0; i < count; i++) {
		if (!routes[i].available)
			continue;
		const char *p = routes[i].path;
		if (strcmp(p, "/metrics") == 0) {
			json_array_append_new(caps, json_string("metrics"));
			continue;
		}
		if (strncmp(p, "/api/v1/", 8) != 0)
			continue;
		const char *name = p + 8;
		if (!*name)
			continue;                       /* the index itself */
		char buf[64];
		snprintf(buf, sizeof(buf), "%s", name);
		size_t l = strlen(buf);
		if (l && buf[l-1] == '/') buf[l-1] = '\0';   /* prefix route */
		for (char *s = buf; *s; s++) if (*s == '/') *s = '.';  /* pools/switch -> pools.switch */
		/* de-duplicate: /pools and /pools/ both map to "pools" */
		bool seen = false;
		size_t na = json_array_size(caps);
		for (size_t k = 0; k < na; k++) {
			const char *ex = json_string_value(json_array_get(caps, k));
			if (ex && strcmp(ex, buf) == 0) { seen = true; break; }
		}
		if (!seen)
			json_array_append_new(caps, json_string(buf));
	}

	json_object_set_new(links, "summary", json_string("/api/v1/summary"));
	json_object_set_new(links, "devices", json_string("/api/v1/devices"));
	json_object_set_new(index, "capabilities", caps);
	json_object_set_new(index, "links", links);
	json_object_set_new(root, "index", index);
	*out = root;
	return 200;
}

static int h_summary(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	struct api_summary_snapshot snap;
	json_t *root = envelope();
	if (!root) return oom(e, n);

	api_collect_summary(&snap);
	json_t *o = api_build_summary_json(&snap);
	if (!o) { json_decref(root); return oom(e, n); }
	json_object_set_new(root, "summary", o);
	*out = root;
	return 200;
}

static int h_threads(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	long want = -1;
	bool filtered = query_int(req->query, "id", &want);

	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	bool matched = false;
	for (int i = 0; i < opt_n_threads; i++) {
		if (filtered && i != (int) want) continue;
		struct api_thread_snapshot snap;
		api_collect_thread(i, &snap);
		if (!snap.valid) continue;
		matched = true;
		json_t *t = api_build_thread_json(&snap);
		if (t) json_array_append_new(arr, t);
	}
	if (filtered && !matched) {
		json_decref(root);
		json_decref(arr);
		snprintf(e, n, "no such thread");
		return 404;
	}
	json_object_set_new(root, "threads", arr);
	*out = root;
	return 200;
}

/* /devices and /devices/{id} share this: the hardware snapshot plus the
 * hashrate from the thread bound to that device. */
static bool device_json(int gpu_id, json_t **dst)
{
	struct api_gpuhw_snapshot hw;
	struct api_thread_snapshot th;
	api_collect_gpuhw(gpu_id, &hw);
	if (!hw.valid)
		return false;

	memset(&th, 0, sizeof(th));
	for (int i = 0; i < opt_n_threads; i++) {
		struct api_thread_snapshot t;
		api_collect_thread(i, &t);
		if (t.valid && t.gpu_id == gpu_id) { th = t; break; }
	}
	*dst = api_build_device_json(&hw, th.valid ? &th : NULL);
	return *dst != NULL;
}

static int h_devices(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	for (int i = 0; i < cuda_num_devices(); i++) {
		json_t *d = NULL;
		if (device_json(i, &d))
			json_array_append_new(arr, d);
	}
	json_object_set_new(root, "devices", arr);
	*out = root;
	return 200;
}

static int h_device(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	long id = 0;
	if (!trailing_int(req->path, &id) || id < 0 || id >= cuda_num_devices()) {
		snprintf(e, n, "no such device");
		return 404;
	}
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *d = NULL;
	if (!device_json((int) id, &d)) {
		json_decref(root);
		snprintf(e, n, "no such device");
		return 404;
	}
	json_object_set_new(root, "device", d);
	*out = root;
	return 200;
}

static int h_system(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	struct api_system_snapshot snap;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	api_collect_system(&snap);
	json_t *o = api_build_system_json(&snap);
	if (!o) { json_decref(root); return oom(e, n); }
	json_object_set_new(root, "system", o);
	*out = root;
	return 200;
}

static int h_pools(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	long want = -1;
	bool filtered = query_int(req->query, "index", &want);

	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	for (int i = 0; i < num_pools; i++) {
		if (filtered && i != (int) want) continue;
		struct api_pool_snapshot snap;
		api_collect_pool(i, &snap);
		json_t *p = api_build_pool_json(i, i == cur_pooln, &snap);
		if (p) json_array_append_new(arr, p);
	}
	if (filtered && json_array_size(arr) == 0) {
		json_decref(root);
		json_decref(arr);
		snprintf(e, n, "no such pool");
		return 404;
	}
	json_object_set_new(root, "pools", arr);
	*out = root;
	return 200;
}

static int h_pool(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	long idx = 0;
	if (!trailing_int(req->path, &idx) || idx < 0 || idx >= num_pools) {
		snprintf(e, n, "no such pool");
		return 404;
	}
	struct api_pool_snapshot snap;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	api_collect_pool((int) idx, &snap);
	json_t *p = api_build_pool_json((int) idx, (int) idx == cur_pooln, &snap);
	if (!p) { json_decref(root); return oom(e, n); }
	json_object_set_new(root, "pool", p);
	*out = root;
	return 200;
}

static int h_health(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *h = json_object();
	json_t *reasons = json_array();
	if (!h || !reasons) {
		json_decref(root);
		if (h) json_decref(h);
		if (reasons) json_decref(reasons);
		return oom(e, n);
	}

	const bool stratum_pool = (pools[cur_pooln].type & POOL_STRATUM) != 0;
	const bool connected = stratum_pool ? (stratum.curl != NULL) : true;

	/* A stopped miner is disconnected by request, not by fault. */
	const ctl_state_t ctl_state = api_ctl_enabled() ? api_ctl_get_state() : CTL_RUNNING;
	const bool idle_by_request = (ctl_state == CTL_STOPPED || ctl_state == CTL_PAUSED);
	const bool mining = opt_n_threads > 0 && !abort_flag && !idle_by_request;

	int devices_ok = 0;
	for (int i = 0; i < opt_n_threads; i++) {
		struct api_thread_snapshot t;
		api_collect_thread(i, &t);
		if (t.valid) devices_ok++;
	}

	if (!connected && !idle_by_request)
		json_array_append_new(reasons, json_string("pool_disconnected"));

	/* A deliberately stopped miner is healthy: a manager-initiated stop is not
	 * a fault (docs/api-rest.md section 6.6). */
	const bool degraded = !connected && !idle_by_request;

	json_object_set_new(h, "status", json_string(degraded ? "degraded" : "ok"));
	json_object_set_new(h, "mining", json_boolean(mining));
	json_object_set_new(h, "pool_connected", stratum_pool ? json_boolean(connected) : json_null());
	json_object_set_new(h, "devices_ok", json_integer(devices_ok));
	if (json_array_size(reasons))
		json_object_set_new(h, "reasons", reasons);
	else
		json_decref(reasons);

	json_object_set_new(root, "health", h);
	*out = root;
	return degraded ? 503 : 200;
}

/* Wire names for the parameter types advertised by /algos (section 9). */
static const char *param_type_name(ctl_param_type_t t)
{
	switch (t) {
	case CTLP_INT:        return "int";
	case CTLP_STRING:     return "string";
	case CTLP_INT_LIST:   return "int_list";
	case CTLP_FLOAT_LIST: return "float_list";
	}
	return "string";
}

static int h_algos(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	/* The accepted parameter set per algo, so a manager needs no hardcoded
	 * table and cannot send a parameter this build would reject. */
	for (int i = 0; i < ALGO_COUNT; i++) {
		if (!algo_names[i] || !*algo_names[i])
			continue;
		json_t *a = json_object();
		if (!a) continue;
		json_object_set_new(a, "name", json_string(algo_names[i]));

		size_t pcount = 0;
		const char **pnames = api_ctl_algo_params(i, &pcount);
		json_t *params = json_array();
		for (size_t k = 0; params && k < pcount; k++) {
			const ctl_param_def_t *d = api_ctl_param_find(pnames[k]);
			if (!d) continue;
			json_t *pd = json_object();
			if (!pd) continue;
			json_object_set_new(pd, "name", json_string(d->name));
			json_object_set_new(pd, "type", json_string(param_type_name(d->type)));
			/* "slow" means the client must be ready for a 202 and a poll. */
			json_object_set_new(pd, "tier", json_string(d->slow ? "slow" : "fast"));
			json_array_append_new(params, pd);
		}
		json_object_set_new(a, "params", params ? params : json_array());
		json_array_append_new(arr, a);
	}
	json_object_set_new(root, "algos", arr);
	*out = root;
	return 200;
}

static int h_config(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *c = json_object();
	if (!c) { json_decref(root); return oom(e, n); }

	/* Credentials are masked: never return user or pass (section 6.8). */
	char algo[64] = { 0 };
	get_currentalgo(algo, sizeof(algo));
	json_object_set_new(c, "algo", json_string(algo));
	json_object_set_new(c, "threads", json_integer(opt_n_threads));
	json_object_set_new(c, "devices", json_integer(active_gpus));
	/* Intensity is per device, not a scalar: report what was configured for
	 * each, and null where it is left on auto. */
	json_t *inten = json_array();
	if (inten) {
		for (int i = 0; i < active_gpus && i < MAX_GPUS; i++)
			json_array_append_new(inten, gpus_intensity[i]
				? json_real(throughput2intensity(gpus_intensity[i]))
				: json_null());
		json_object_set_new(c, "intensity", inten);
	}
	json_object_set_new(c, "benchmark", json_boolean(opt_benchmark));
	json_object_set_new(c, "debug", json_boolean(opt_debug));
	json_object_set_new(c, "pools", json_integer(num_pools));
	json_object_set_new(root, "config", c);
	*out = root;
	return 200;
}

/* ------------------------------------------------- scan statistics / logs */

static int h_history(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	struct stats_data data[50];
	long thread = -1, limit = (long) ARRAY_SIZE(data);

	query_int(req->query, "thread", &thread);
	if (query_int(req->query, "limit", &limit)) {
		if (limit < 1 || limit > (long) ARRAY_SIZE(data))
			limit = (long) ARRAY_SIZE(data);      /* documented cap, not an error */
	}

	int records = stats_get_history((int) thread, data, (int) limit);

	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	for (int i = 0; i < records; i++) {
		json_t *r = json_object();
		if (!r) continue;
		json_object_set_new(r, "thread_id", json_integer(data[i].thr_id));
		json_object_set_new(r, "device_id", json_integer(data[i].gpu_id));
		json_object_set_new(r, "height", json_integer(data[i].height));
		/* stats_data.hashrate is already H/s — stats.cpp compares it directly
		 * against global_hashrate. The binary API prints this same value under
		 * the key KHS, which mislabels it; do not copy that here. */
		json_object_set_new(r, "hashrate_hs", json_real(data[i].hashrate));
		json_object_set_new(r, "difficulty", json_real(data[i].difficulty));
		json_object_set_new(r, "hashcount", json_integer(data[i].hashcount));
		json_object_set_new(r, "found", json_boolean(data[i].hashfound != 0));
		json_object_set_new(r, "id", json_integer(data[i].uid));
		json_object_set_new(r, "timestamp", json_integer(data[i].tm_stat));
		json_array_append_new(arr, r);
	}
	json_object_set_new(root, "history", arr);
	*out = root;
	return 200;
}

static int h_scanlog(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	struct hashlog_data data[50];
	int records = hashlog_get_history(data, (int) ARRAY_SIZE(data));

	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	for (int i = 0; i < records; i++) {
		json_t *r = json_object();
		if (!r) continue;
		json_object_set_new(r, "height", json_integer(data[i].height));
		json_object_set_new(r, "pool_index", json_integer(data[i].npool));
		json_object_set_new(r, "job_id", json_integer(data[i].njobid));
		json_object_set_new(r, "nonce_id", json_integer(data[i].job_nonce_id));
		json_object_set_new(r, "share_diff", json_real(data[i].sharediff));
		json_object_set_new(r, "nonce", json_integer(data[i].nonce));
		json_object_set_new(r, "scanned_from", json_integer(data[i].scanned_from));
		json_object_set_new(r, "scanned_to", json_integer(data[i].scanned_to));
		json_object_set_new(r, "scanned_count",
			json_integer((json_int_t) data[i].scanned_to - (json_int_t) data[i].scanned_from));
		json_object_set_new(r, "submitted", json_boolean(data[i].tm_sent != 0));
		json_object_set_new(r, "timestamp", json_integer(data[i].tm_upd));
		json_array_append_new(arr, r);
	}
	json_object_set_new(root, "scanlog", arr);
	*out = root;
	return 200;
}

static int h_meminfo(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	uint64_t smem = 0, hmem = 0;
	uint32_t srec = 0, hrec = 0;

	stats_getmeminfo(&smem, &srec);
	hashlog_getmeminfo(&hmem, &hrec);

	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *m = json_object();
	if (!m) { json_decref(root); return oom(e, n); }

	json_object_set_new(m, "stats_records", json_integer(srec));
	json_object_set_new(m, "stats_bytes", json_integer((json_int_t) smem));
	json_object_set_new(m, "hashlog_records", json_integer(hrec));
	json_object_set_new(m, "hashlog_bytes", json_integer((json_int_t) hmem));
	json_object_set_new(m, "total_bytes", json_integer((json_int_t)(smem + hmem)));
	json_object_set_new(root, "meminfo", m);
	*out = root;
	return 200;
}

/* ------------------------------------------------------------ write routes */

/* Body parsing is explicit about types: a wrong type is 400, never a crash and
 * never a silent coercion. An absent body is treated as an empty object so a
 * verb that needs no arguments works with or without one. */
static json_t *parse_body(const api_request *req, char *e, size_t n, int *status)
{
	if (!req->body_len) {
		*status = 200;
		return json_object();
	}
	json_error_t err;
	json_t *root = json_loads(req->body, 0, &err);
	if (!root) {
		snprintf(e, n, "invalid JSON: %s", err.text);
		*status = 400;
		return NULL;
	}
	if (!json_is_object(root)) {
		json_decref(root);
		snprintf(e, n, "body must be a JSON object");
		*status = 400;
		return NULL;
	}
	*status = 200;
	return root;
}

static json_t *result_ok(void)
{
	json_t *root = envelope();
	if (!root) return NULL;
	json_t *res = json_object();
	if (!res) { json_decref(root); return NULL; }
	json_object_set_new(res, "ok", json_true());
	json_object_set_new(root, "result", res);
	return root;
}

static int h_pools_switch(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	int status = 200;
	json_t *body = parse_body(req, e, n, &status);
	if (!body) return status;

	json_t *jnext = json_object_get(body, "next");
	json_t *jindex = json_object_get(body, "index");

	if (jnext && jindex) {
		json_decref(body);
		snprintf(e, n, "send either index or next, not both");
		return 400;
	}
	if (!jnext && !jindex) {
		json_decref(body);
		snprintf(e, n, "index or next is required");
		return 400;
	}

	bool ok;
	if (jnext) {
		if (!json_is_boolean(jnext)) {
			json_decref(body);
			snprintf(e, n, "next must be a boolean");
			return 400;
		}
		if (!json_is_true(jnext)) {
			json_decref(body);
			snprintf(e, n, "next must be true");
			return 400;
		}
		ok = pool_switch_next(-1);
	} else {
		if (!json_is_integer(jindex)) {
			json_decref(body);
			snprintf(e, n, "index must be an integer");
			return 400;
		}
		json_int_t idx = json_integer_value(jindex);
		/* The binary command accepts a negative index and passes it straight
		 * through; the REST contract says an out-of-range index is 404. */
		if (idx < 0 || idx >= num_pools) {
			json_decref(body);
			snprintf(e, n, "no such pool");
			return 404;
		}
		ok = (idx == cur_pooln) ? true : pool_switch(-1, (int) idx);
	}
	json_decref(body);

	if (!ok) {
		snprintf(e, n, "pool switch failed");
		return 409;
	}

	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *res = json_object();
	if (!res) { json_decref(root); return oom(e, n); }
	json_object_set_new(res, "ok", json_true());
	json_object_set_new(res, "active", json_integer(cur_pooln));
	json_object_set_new(root, "result", res);
	*out = root;
	return 200;
}

/* parse_arg('o') accepts exactly these, and calls show_usage_and_exit() on
 * anything else — which would terminate the miner from an API request. The
 * scheme is therefore validated here, before the value can reach it. */
static bool scheme_supported(const char *url)
{
	static const char *ok[] = {
		"http://", "https://", "stratum+tcp://", "stratum+ssl://", "stratum+tcps://"
	};
	for (size_t i = 0; i < sizeof(ok) / sizeof(ok[0]); i++) {
		size_t l = strlen(ok[i]);
		if (strlen(url) > l && strncasecmp(url, ok[i], l) == 0)
			return true;
	}
	return false;
}

static int h_pools_url(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	int status = 200;
	json_t *body = parse_body(req, e, n, &status);
	if (!body) return status;

	json_t *jurl = json_object_get(body, "url");
	json_t *juser = json_object_get(body, "user");
	json_t *jpass = json_object_get(body, "pass");

	if (!jurl || !json_is_string(jurl)) {
		json_decref(body);
		snprintf(e, n, "url is required and must be a string");
		return 400;
	}
	if ((juser && !json_is_string(juser)) || (jpass && !json_is_string(jpass))) {
		json_decref(body);
		snprintf(e, n, "user and pass must be strings");
		return 400;
	}

	const char *url = json_string_value(jurl);
	const char *user = juser ? json_string_value(juser) : NULL;
	const char *pass = jpass ? json_string_value(jpass) : NULL;

	if (!scheme_supported(url)) {
		json_decref(body);
		snprintf(e, n, "unsupported url scheme");
		return 400;
	}

	/* Assemble the scheme://user:pass@host form parse_arg('o') expects, so a
	 * client never has to build a packed string (docs/api-rest.md 6.10). The
	 * credentials are stripped back out of the stored url by the parser, which
	 * is why /pools can never return them. */
	char packed[1280];
	int wrote;
	if (user && *user) {
		const char *sep = strstr(url, "://");
		size_t schemelen = (size_t) (sep - url) + 3;
		if (schemelen >= sizeof(packed)) {
			json_decref(body);
			snprintf(e, n, "url too long");
			return 400;
		}
		wrote = snprintf(packed, sizeof(packed), "%.*s%s:%s@%s",
			(int) schemelen, url, user, pass ? pass : "x", url + schemelen);
	} else {
		wrote = snprintf(packed, sizeof(packed), "%s", url);
	}
	json_decref(body);

	if (wrote < 0 || (size_t) wrote >= sizeof(packed)) {
		snprintf(e, n, "url too long");
		return 400;
	}

	if (!pool_switch_url(packed)) {
		snprintf(e, n, "pool url change failed");
		return 409;
	}

	json_t *root = result_ok();
	if (!root) return oom(e, n);
	*out = root;
	return 200;
}

static int h_quit(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req;
	json_t *root = result_ok();
	if (!root) return oom(e, n);

	/* Only flag it. The caller shuts down after this response has been sent,
	 * so the client always sees its 200. */
	if (ctx)
		((api_ctx *) ctx)->quit_requested = true;

	*out = root;
	return 200;
}


/* ------------------------------------------------------- runtime control */

/* docs/api-rest.md section 7. Both 200 and 202 are success. */
static int ctl_status(ctl_result_t rc)
{
	switch (rc) {
	case CTL_OK:        return 200;
	case CTL_ACCEPTED:  return 202;
	case CTL_BUSY:      return 409;
	case CTL_TIMEOUT:   return 409;
	case CTL_THROTTLED: return 429;
	case CTL_DISABLED:  return 403;
	}
	return 500;
}

/* Params in effect. Every accepted key is present; unset reads null. */
static json_t *ctl_params_json(int algo)
{
	json_t *p = json_object();
	if (!p) return NULL;
	size_t count = 0;
	const char **names = api_ctl_algo_params(algo, &count);
	for (size_t i = 0; i < count; i++) {
		char buf[CTL_PARAM_VALUE_MAX];
		const char *v = api_ctl_param_get(algo, names[i], buf, sizeof(buf));
		json_object_set_new(p, names[i], v ? json_string(v) : json_null());
	}
	return p;
}

static json_t *ctl_state_object(void)
{
	json_t *c = json_object();
	if (!c) return NULL;

	const time_t now = time(NULL);
	const ctl_state_t st = api_ctl_get_state();
	const char *err = api_ctl_last_error();

	json_object_set_new(c, "state", json_string(api_ctl_state_name(st)));
	json_object_set_new(c, "epoch", json_integer(api_ctl_epoch()));
	json_object_set_new(c, "since_s",
		json_integer((json_int_t) difftime(now, api_ctl_state_since())));
	json_object_set_new(c, "algo",
		json_string(algo_names[opt_algo] ? algo_names[opt_algo] : ""));
	json_object_set_new(c, "params", ctl_params_json(opt_algo));

	struct api_pool_snapshot ps;
	api_collect_pool(cur_pooln, &ps);
	json_t *pool = json_object();
	if (pool) {
		json_object_set_new(pool, "index", json_integer(cur_pooln));
		json_object_set_new(pool, "url", json_string(ps.url));
		json_object_set_new(pool, "user", json_string(ps.user));
		json_object_set_new(c, "pool", pool);
	}
	/* null, not false, for getwork/GBT: no connection to report on. */
	json_object_set_new(c, "pool_connected",
		ps.stratum ? json_boolean(ps.connected) : json_null());

	json_object_set_new(c, "threads_total", json_integer(api_ctl_threads_total()));
	json_object_set_new(c, "threads_parked", json_integer(api_ctl_parked()));
	json_object_set_new(c, "switch_count", json_integer(api_ctl_switch_count()));
	json_object_set_new(c, "last_switch_age_s",
		api_ctl_last_switch()
			? json_integer((json_int_t) difftime(now, api_ctl_last_switch()))
			: json_null());
	json_object_set_new(c, "min_interval_s", json_integer(api_ctl_min_interval()));
	json_object_set_new(c, "ready_for_switch", json_boolean(api_ctl_ready_for_switch()));
	json_object_set_new(c, "last_error", err ? json_string(err) : json_null());
	return c;
}

static int h_control_state(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *c = ctl_state_object();
	if (!c) { json_decref(root); return oom(e, n); }
	json_object_set_new(root, "control", c);
	*out = root;
	return 200;
}

/* Optional {"wait_ms": N}. 0 is legal and means "answer 202, I will poll". */
static bool ctl_wait_ms(json_t *body, int *wait_ms, char *e, size_t n)
{
	json_t *j = json_object_get(body, "wait_ms");
	if (!j) return true;
	if (!json_is_integer(j) || json_integer_value(j) < 0) {
		snprintf(e, n, "wait_ms must be a non-negative integer");
		return false;
	}
	*wait_ms = (int) json_integer_value(j);
	return true;
}

/* full = the /control/state body (section 7.4); otherwise the short result
 * object the verbs return (section 7.3). */
static int ctl_reply(ctl_result_t rc, bool full, json_t **out, char *e, size_t n)
{
	const int status = ctl_status(rc);
	if (status >= 400) {
		const char *err = api_ctl_last_error();
		if (rc == CTL_THROTTLED)
			snprintf(e, n, "too frequent, retry after %ds", api_ctl_retry_after_s());
		else
			snprintf(e, n, "%s", err ? err : "another control request is in flight");
		return status;
	}

	json_t *root = envelope();
	if (!root) return oom(e, n);

	json_t *body;
	if (full) {
		body = ctl_state_object();
	} else {
		body = json_object();
		if (body) {
			json_object_set_new(body, "state",
				json_string(api_ctl_state_name(api_ctl_get_state())));
			json_object_set_new(body, "epoch", json_integer(api_ctl_epoch()));
			json_object_set_new(body, "parked", json_integer(api_ctl_parked()));
		}
	}
	if (!body) { json_decref(root); return oom(e, n); }

	/* 202 carries the poll target, so a client never has to construct it. */
	if (status == 202)
		json_object_set_new(body, "poll", json_string("/api/v1/control/state"));

	json_object_set_new(root, full ? "control" : "result", body);
	*out = root;
	return status;
}

static int ctl_verb(const api_request *req, ctl_state_t target,
                    json_t **out, char *e, size_t n)
{
	int status = 200;
	json_t *body = parse_body(req, e, n, &status);
	if (!body) return status;

	int wait_ms = 10000;
	if (!ctl_wait_ms(body, &wait_ms, e, n)) { json_decref(body); return 400; }
	json_decref(body);

	return ctl_reply(api_ctl_set_state(target, wait_ms), false, out, e, n);
}

static int h_control_start(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	return ctl_verb(req, CTL_RUNNING, out, e, n);
}

static int h_control_pause(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	return ctl_verb(req, CTL_PAUSED, out, e, n);
}

static int h_control_stop(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	return ctl_verb(req, CTL_STOPPED, out, e, n);
}

/* --- profile ------------------------------------------------------------ */

static bool profile_run_target(json_t *jrun, ctl_state_t *out, char *e, size_t n)
{
	if (json_is_true(jrun))  { *out = CTL_RUNNING; return true; }
	if (json_is_false(jrun)) { *out = CTL_PAUSED;  return true; }
	if (json_is_string(jrun)) {
		const char *s = json_string_value(jrun);
		if (!strcmp(s, "running") || !strcmp(s, "true"))  { *out = CTL_RUNNING; return true; }
		if (!strcmp(s, "paused"))  { *out = CTL_PAUSED;  return true; }
		if (!strcmp(s, "stopped")) { *out = CTL_STOPPED; return true; }
	}
	snprintf(e, n, "run must be true, false, \"paused\" or \"stopped\"");
	return false;
}

/* {"n": 2048} means the same as {"n": "2048"}; refusing that teaches nothing. */
static bool param_to_string(json_t *v, char *buf, size_t buflen, char *e, size_t n,
                            const char *key)
{
	if (json_is_string(v)) {
		snprintf(buf, buflen, "%s", json_string_value(v));
		return true;
	}
	if (json_is_integer(v)) {
		snprintf(buf, buflen, "%lld", (long long) json_integer_value(v));
		return true;
	}
	if (json_is_real(v)) {
		snprintf(buf, buflen, "%g", json_real_value(v));
		return true;
	}
	if (json_is_boolean(v)) {
		snprintf(buf, buflen, "%d", json_is_true(v) ? 1 : 0);
		return true;
	}
	snprintf(e, n, "parameter '%s' must be a string, number or null", key);
	return false;
}

static int h_control_profile(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	int status = 200;
	json_t *body = parse_body(req, e, n, &status);
	if (!body) return status;

	ctl_profile_req p;
	memset(&p, 0, sizeof(p));

	json_t *jalgo   = json_object_get(body, "algo");
	json_t *jpool   = json_object_get(body, "pool");
	json_t *jparams = json_object_get(body, "params");
	json_t *jrun    = json_object_get(body, "run");

	int wait_ms = 10000;
	if (!ctl_wait_ms(body, &wait_ms, e, n)) { json_decref(body); return 400; }

	if (!jalgo && !jpool && !jparams && !jrun) {
		json_decref(body);
		snprintf(e, n, "body must contain at least one of algo, pool, params, run");
		return 400;
	}

	if (jalgo) {
		if (!json_is_string(jalgo)) {
			json_decref(body);
			snprintf(e, n, "algo must be a string");
			return 400;
		}
		const int a = algo_to_int((char *) json_string_value(jalgo));
		if (a < 0) {
			json_decref(body);
			snprintf(e, n, "unknown algo '%s'", json_string_value(jalgo));
			return 400;
		}
		p.has_algo = true;
		p.algo = a;
		snprintf(p.algo_name, sizeof(p.algo_name), "%s",
		         json_string_value(jalgo));
	}

	/* Structural: algo X against a pool expecting Y yields 100% rejects while
	 * every other metric looks healthy. */
	if (p.has_algo && !jpool) {
		json_decref(body);
		snprintf(e, n, "algo requires pool: send both or neither");
		return 400;
	}

	if (jpool) {
		if (!json_is_object(jpool)) {
			json_decref(body);
			snprintf(e, n, "pool must be an object");
			return 400;
		}
		json_t *jurl = json_object_get(jpool, "url");
		json_t *jidx = json_object_get(jpool, "index");
		json_t *juser = json_object_get(jpool, "user");
		json_t *jpass = json_object_get(jpool, "pass");

		if (jurl && jidx) {
			json_decref(body);
			snprintf(e, n, "send pool.url or pool.index, not both");
			return 400;
		}
		if (jidx) {
			if (!json_is_integer(jidx) || json_integer_value(jidx) < 0) {
				json_decref(body);
				snprintf(e, n, "pool.index must be a non-negative integer");
				return 400;
			}
			p.has_pool_index = true;
			p.pool_index = (int) json_integer_value(jidx);
		} else if (jurl) {
			if (!json_is_string(jurl)) {
				json_decref(body);
				snprintf(e, n, "pool.url must be a string");
				return 400;
			}
			if (!scheme_supported(json_string_value(jurl))) {
				json_decref(body);
				snprintf(e, n, "unsupported url scheme");
				return 400;
			}
			if ((juser && !json_is_string(juser)) || (jpass && !json_is_string(jpass))) {
				json_decref(body);
				snprintf(e, n, "pool.user and pool.pass must be strings");
				return 400;
			}
			p.has_pool = true;
			snprintf(p.pool_url, sizeof(p.pool_url), "%s", json_string_value(jurl));
			if (juser) snprintf(p.pool_user, sizeof(p.pool_user), "%s", json_string_value(juser));
			if (jpass) snprintf(p.pool_pass, sizeof(p.pool_pass), "%s", json_string_value(jpass));
		} else {
			json_decref(body);
			snprintf(e, n, "pool needs url or index");
			return 400;
		}
	}

	if (jparams) {
		if (!json_is_object(jparams)) {
			json_decref(body);
			snprintf(e, n, "params must be an object");
			return 400;
		}
		/* Validated against the algo this request lands on, not the current one. */
		const int target_algo = p.has_algo ? p.algo : (int) opt_algo;
		const char *key;
		json_t *val;
		json_object_foreach(jparams, key, val) {
			if (p.nparams >= CTL_MAX_PARAMS) {
				json_decref(body);
				snprintf(e, n, "too many parameters");
				return 400;
			}
			if (!api_ctl_param_find(key)) {
				json_decref(body);
				snprintf(e, n, "unknown parameter '%s'", key);
				return 400;
			}
			if (!api_ctl_algo_accepts(target_algo, key)) {
				json_decref(body);
				snprintf(e, n, "algo %s does not accept parameter '%s'",
					algo_names[target_algo] ? algo_names[target_algo] : "?", key);
				return 400;
			}
			const int i = p.nparams++;
			snprintf(p.pname[i], CTL_PARAM_NAME_MAX, "%s", key);
			if (json_is_null(val)) {
				p.pnull[i] = true;       /* explicit reset to the startup default */
			} else if (!param_to_string(val, p.pvalue[i], CTL_PARAM_VALUE_MAX, e, n, key)) {
				json_decref(body);
				return 400;
			}
			/* Range-checked here, not at the barrier: a bad value is a 400. */
			if (!api_ctl_param_check(target_algo, key,
			                         p.pnull[i] ? NULL : p.pvalue[i], e, n)) {
				json_decref(body);
				return 400;
			}
		}
	}

	if (jrun) {
		if (!profile_run_target(jrun, &p.run, e, n)) {
			json_decref(body);
			return 400;
		}
		p.has_run = true;
	}

	json_decref(body);

	char err[192] = { 0 };
	const ctl_result_t rc = api_ctl_profile(&p, wait_ms, err, sizeof(err));
	if (rc == CTL_BUSY && err[0]) {
		snprintf(e, n, "%s", err);
		return 409;
	}
	return ctl_reply(rc, true, out, e, n);
}

/* ------------------------------------------------------------------ metrics */

/* Prometheus exposition (docs/api-rest.md section 11). Text, not JSON, so it
 * takes the transport's text path. --api-token applies here too. */
static int h_metrics(const api_request *req, void *ctx, char **out, char *e, size_t n)
{
	(void) req; (void) ctx;

	api_metrics_input in;
	api_collect_metrics(&in);

	/* Sized for the worst case; the renderer reports overflow rather than
	 * emitting half a family. */
	const size_t cap = 8192 + (size_t) in.ndevices * 512 + (size_t) in.npools * 512;
	char *buf = (char *) malloc(cap);
	if (!buf) {
		snprintf(e, n, "out of memory");
		return 500;
	}
	const size_t len = api_metrics_render(&in, buf, cap);
	if (!len) {
		free(buf);
		snprintf(e, n, "metrics buffer too small");
		return 500;
	}
	*out = buf;
	return 200;
}

/* Registered so the route exists and answers 501 rather than 404. */
static int h_unavailable(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx; (void) out;
	snprintf(e, n, "not implemented yet");
	return 501;
}

/* ------------------------------------------------------------------- table */

static const api_route g_routes[] = {
	{ API_M_GET,  "/api/v1",              API_PRIV_READ,    true,  h_index, NULL, NULL },
	{ API_M_GET,  "/api/v1/summary",      API_PRIV_READ,    true,  h_summary, NULL, NULL },
	{ API_M_GET,  "/api/v1/threads",      API_PRIV_READ,    true,  h_threads, NULL, NULL },
	{ API_M_GET,  "/api/v1/devices",      API_PRIV_READ,    true,  h_devices, NULL, NULL },
	{ API_M_GET,  "/api/v1/devices/",     API_PRIV_READ,    true,  h_device, NULL, NULL },
	{ API_M_GET,  "/api/v1/system",       API_PRIV_READ,    true,  h_system, NULL, NULL },
	{ API_M_GET,  "/api/v1/pools",        API_PRIV_READ,    true,  h_pools, NULL, NULL },
	{ API_M_GET,  "/api/v1/pools/",       API_PRIV_READ,    true,  h_pool, NULL, NULL },
	{ API_M_GET,  "/api/v1/health",       API_PRIV_READ,    true,  h_health, NULL, NULL },
	{ API_M_GET,  "/api/v1/config",       API_PRIV_READ,    true,  h_config, NULL, NULL },
	{ API_M_GET,  "/api/v1/algos",        API_PRIV_READ,    true,  h_algos, NULL, NULL },

	{ API_M_GET,  "/api/v1/history",      API_PRIV_READ,    true,  h_history, NULL, NULL },
	{ API_M_GET,  "/api/v1/scanlog",      API_PRIV_READ,    true,  h_scanlog, NULL, NULL },
	{ API_M_GET,  "/api/v1/meminfo",      API_PRIV_READ,    true,  h_meminfo, NULL, NULL },

	/* write routes */
	{ API_M_POST, "/api/v1/pools/switch", API_PRIV_WRITE,   true,  h_pools_switch, NULL, NULL },
	{ API_M_POST, "/api/v1/pools/url",    API_PRIV_WRITE,   true,  h_pools_url, NULL, NULL },
	{ API_M_POST, "/api/v1/quit",         API_PRIV_WRITE,   true,  h_quit, NULL, NULL },

	/* runtime control; the mutating verbs also need --api-control */
	{ API_M_GET,  "/api/v1/control/state",   API_PRIV_READ,    true,  h_control_state, NULL, NULL },
	{ API_M_POST, "/api/v1/control/start",   API_PRIV_CONTROL, true,  h_control_start, NULL, NULL },
	{ API_M_POST, "/api/v1/control/pause",   API_PRIV_CONTROL, true,  h_control_pause, NULL, NULL },
	{ API_M_POST, "/api/v1/control/stop",    API_PRIV_CONTROL, true,  h_control_stop, NULL, NULL },
	{ API_M_POST, "/api/v1/control/profile", API_PRIV_CONTROL, true,  h_control_profile, NULL, NULL },

	/* Prometheus exposition. Outside /api/v1 by design: not versioned JSON,
	 * and every scraper defaults to this path. */
	{ API_M_GET,  "/metrics",               API_PRIV_READ,    true,  NULL, h_metrics, API_METRICS_CONTENT_TYPE },
};

const api_route *api_routes_get(size_t *count)
{
	*count = sizeof(g_routes) / sizeof(g_routes[0]);
	return g_routes;
}

char *api_routes_miner_json_str(void)
{
	json_t *m = api_build_miner_json();
	if (!m) return NULL;
	char *s = json_dumps(m, JSON_COMPACT);
	json_decref(m);
	return s;
}
