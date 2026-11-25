// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Collectors and binary renderers for the API reports — see api_model.h.
 *
 * The format strings and the expressions feeding them are copied verbatim from
 * the handlers that used to live in api.cpp. Any edit here changes a
 * compatibility surface; run api/tests/golden.py before and after.
 */

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "miner.h"
#include "algos.h"
#include "core/gpu/nvml.h"
#include "api_control.h"
#include "api_model.h"

#ifdef WIN32
#include "compat.h"
#endif

/* stratum.job is freed and replaced by stratum_notify() under this lock.
 * Declared locally, not in miner.h — see the linkage note in api_model.h. */
extern pthread_mutex_t stratum_work_lock;
extern struct stratum_ctx stratum;

extern int num_cpus;
extern float cpu_temp(int);
extern uint32_t cpu_clock(int);
extern char driver_version[32];

extern int active_gpus;
extern char *device_name[MAX_GPUS];
extern short device_map[MAX_GPUS];
extern uint32_t device_plimit[MAX_GPUS];

/* startup time, owned by api.cpp */
extern time_t api_startup_time;
extern time_t api_stats_since;

static void copy_str(char *dst, size_t dstlen, const char *src)
{
	if (!dstlen) return;
	if (!src) { dst[0] = '\0'; return; }
	snprintf(dst, dstlen, "%s", src);
}

/* ------------------------------------------------------------------ summary */

/* A parked miner is not hashing, so every rate reads 0 rather than its last
 * value. Past rates stay in the stats table, which /history serves. */
static bool mining_is_parked(void)
{
	if (!api_ctl_enabled())
		return false;
	const ctl_state_t st = api_ctl_get_state();
	return st == CTL_PAUSED || st == CTL_STOPPED;
}

void api_collect_summary(struct api_summary_snapshot *s)
{
	time_t ts = time(NULL);
	double uptime = difftime(ts, api_startup_time);
	uint32_t wait_time = 0, solved_count = 0;
	uint32_t accepted_count = 0, rejected_count = 0;

	memset(s, 0, sizeof(*s));

	for (int p = 0; p < num_pools; p++) {
		wait_time += pools[p].wait_time;
		accepted_count += pools[p].accepted_count;
		rejected_count += pools[p].rejected_count;
		solved_count += pools[p].solved_count;
	}

	s->name = PACKAGE_NAME;
	s->version = PACKAGE_VERSION;
	s->api_version = APIVERSION;
	get_currentalgo(s->algo, sizeof(s->algo));
	s->gpus = active_gpus;
	s->khs = mining_is_parked() ? 0. : (double) global_hashrate / 1000.;
	s->solved = solved_count;
	s->accepted = accepted_count;
	s->rejected = rejected_count;
	/* Over the counter window, not process uptime, which an algo switch would dilute. */
	{
		double window = difftime(ts, api_stats_since ? api_stats_since : api_startup_time);
		s->accps = (60.0 * accepted_count) / (window > 0. ? window : 1.0);
	}
	s->diff = net_diff > 1e-6 ? net_diff : stratum_diff;
	s->netkhs = (double) net_hashrate / 1000.;
	s->pools = (uint32_t) num_pools;
	s->wait_time = wait_time;
	s->uptime = uptime;
	s->ts = (uint32_t) ts;
}

int api_format_summary_binary(const struct api_summary_snapshot *s, char *out, size_t outlen)
{
	return snprintf(out, outlen, "NAME=%s;VER=%s;API=%s;"
		"ALGO=%s;GPUS=%d;KHS=%.2f;SOLV=%d;ACC=%d;REJ=%d;"
		"ACCMN=%.3f;DIFF=%.6f;NETKHS=%.0f;"
		"POOLS=%u;WAIT=%u;UPTIME=%.0f;TS=%u|",
		s->name, s->version, s->api_version,
		s->algo, s->gpus, s->khs,
		s->solved, s->accepted, s->rejected,
		s->accps, s->diff, s->netkhs,
		s->pools, s->wait_time, s->uptime, s->ts);
}

/* ------------------------------------------------- per-thread (GPU) status */

void api_collect_thread(int thr_id, struct api_thread_snapshot *s)
{
	memset(s, 0, sizeof(*s));

	if (thr_id < 0 || thr_id >= opt_n_threads)
		return;

	struct cgpu_info *cgpu = &thr_info[thr_id].gpu;
	double khashes_per_watt = 0;
	int gpuid = cgpu->gpu_id;

	/* These also refresh thr_info[].gpu, which other readers rely on —
	 * keep the writes, then snapshot. */
	cuda_gpu_info(cgpu);
	cgpu->gpu_plimit = device_plimit[cgpu->gpu_id];

#ifdef USE_WRAPNVML
	cgpu->has_monitoring = true;
	cgpu->gpu_bus = gpu_busid(cgpu);
	cgpu->gpu_temp = gpu_temp(cgpu);
	cgpu->gpu_fan = (uint16_t) gpu_fanpercent(cgpu);
	cgpu->gpu_fan_rpm = (uint16_t) gpu_fanrpm(cgpu);
	cgpu->gpu_power = gpu_power(cgpu); // mWatts
	cgpu->gpu_plimit = gpu_plimit(cgpu); // mW or %
#endif
	cgpu->khashes = stats_get_speed(thr_id, 0.0) / 1000.0;
	if (cgpu->monitor.gpu_power) {
		cgpu->gpu_power = cgpu->monitor.gpu_power;
		khashes_per_watt = (double) cgpu->khashes / cgpu->monitor.gpu_power;
		khashes_per_watt *= 1000; // power in mW
	}

	s->valid = true;
	s->id = thr_id;
	s->gpu_id = gpuid;
	s->bus = cgpu->gpu_bus;
	copy_str(s->card, sizeof(s->card), device_name[gpuid]);
	s->temp = cgpu->gpu_temp;
	s->power = cgpu->gpu_power;
	s->fan = cgpu->gpu_fan;
	s->rpm = cgpu->gpu_fan_rpm;
	s->freq = cgpu->gpu_clock / 1000;        // base freqs in MHz
	s->memfreq = cgpu->gpu_memclock / 1000;
	s->gpuf = cgpu->monitor.gpu_clock;       // current
	s->memf = cgpu->monitor.gpu_memclock;
	s->khs = mining_is_parked() ? 0. : cgpu->khashes;
	s->khw = khashes_per_watt;
	s->plim = cgpu->gpu_plimit;
	s->acc = cgpu->accepted;
	s->rej = (unsigned) cgpu->rejected;
	s->hwf = (unsigned) cgpu->hw_errors;
	s->intensity = cgpu->intensity;
	s->throughput = cgpu->throughput;
}

int api_format_thread_binary(const struct api_thread_snapshot *s, char *out, size_t outlen)
{
	if (!s->valid) { if (outlen) out[0] = '\0'; return 0; }
	return snprintf(out, outlen, "GPU=%d;BUS=%hd;CARD=%s;TEMP=%.1f;"
		"POWER=%u;FAN=%hu;RPM=%hu;"
		"FREQ=%u;MEMFREQ=%u;GPUF=%u;MEMF=%u;"
		"KHS=%.2f;KHW=%.5f;PLIM=%u;"
		"ACC=%u;REJ=%u;HWF=%u;I=%.1f;THR=%u|",
		s->gpu_id, s->bus, s->card, s->temp,
		s->power, s->fan, s->rpm,
		s->freq, s->memfreq, s->gpuf, s->memf,
		s->khs, s->khw, s->plim,
		s->acc, s->rej, s->hwf,
		s->intensity, s->throughput);
}

/* --------------------------------------------------------------------- pool */

void api_collect_pool(int pooln, struct api_pool_snapshot *s)
{
	struct pool_infos *p = &pools[pooln];

	memset(s, 0, sizeof(*s));

	if (p->last_share_time)
		s->last_share_age = (uint32_t) (time(NULL) - p->last_share_time);

	/* stratum_notify() frees and replaces job_id and re-points xnonce2 into a
	 * reallocated coinbase while holding this lock. */
	pthread_mutex_lock(&stratum_work_lock);
	if (stratum.job.job_id)
		strncpy(s->job_id, stratum.job.job_id, sizeof(s->job_id) - 1);
	if (stratum.job.xnonce2) {
		/* used temporary to be sure all is ok */
		sprintf(s->xnonce2, "0x");
		if (p->algo == ALGO_DECRED) {
			char compat[32] = { 0 };
			cbin2hex(&s->xnonce2[2], (const char*) stratum.xnonce1, min(36, stratum.xnonce2_size));
			cbin2hex(compat, (const char*) stratum.job.xnonce2, 4);
			memcpy(&s->xnonce2[2], compat, 8); // compat extranonce
		} else {
			cbin2hex(&s->xnonce2[2], (const char*) stratum.job.xnonce2, stratum.xnonce2_size);
		}
	}
	s->height = stratum.job.height;
	s->xnonce2_size = (int) stratum.xnonce2_size;
	s->ping_ms = stratum.answer_msec;
	pthread_mutex_unlock(&stratum_work_lock);

	copy_str(s->name, sizeof(s->name), strlen(p->name) ? p->name : p->short_url);
	s->algo = algo_names[p->algo];
	copy_str(s->url, sizeof(s->url), p->url);
	copy_str(s->user, sizeof(s->user), p->type & POOL_STRATUM ? p->user : "");
	s->solved = p->solved_count;
	s->accepted = p->accepted_count;
	s->rejected = p->rejected_count;
	s->stales = p->stales_count;
	s->diff = stratum_diff;
	s->best_share = p->best_share;
	s->disconnects = p->disconnects;
	s->wait_time = p->wait_time;
	s->work_time = p->work_time;
	s->stratum = (p->type & POOL_STRATUM) != 0;
	s->connected = s->stratum ? (stratum.curl != NULL) : true;
}

int api_format_pool_binary(const struct api_pool_snapshot *s, char *out, size_t outlen)
{
	return snprintf(out, outlen,
		"POOL=%s;ALGO=%s;URL=%s;USER=%s;SOLV=%d;ACC=%d;REJ=%d;STALE=%u;H=%u;JOB=%s;DIFF=%.6f;"
		"BEST=%.6f;N2SZ=%d;N2=%s;PING=%u;DISCO=%u;WAIT=%u;UPTIME=%u;LAST=%u|",
		s->name, s->algo, s->url, s->user,
		s->solved, s->accepted, s->rejected, s->stales,
		s->height, s->job_id, s->diff, s->best_share,
		s->xnonce2_size, s->xnonce2, s->ping_ms,
		s->disconnects, s->wait_time, s->work_time, s->last_share_age);
}

/* ---------------------------------------------------------- GPU hardware */

void api_collect_gpuhw(int gpu_id, struct api_gpuhw_snapshot *s)
{
	struct cgpu_info *cgpu = NULL;

	memset(s, 0, sizeof(*s));

	for (int g = 0; g < opt_n_threads; g++) {
		if (device_map[g] == gpu_id) {
			cgpu = &thr_info[g].gpu;
			break;
		}
	}
	if (cgpu == NULL)
		return;

	cuda_gpu_info(cgpu);
	cgpu->gpu_plimit = device_plimit[cgpu->gpu_id];

#ifdef USE_WRAPNVML
	cgpu->has_monitoring = true;
	cgpu->gpu_bus = gpu_busid(cgpu);
	cgpu->gpu_temp = gpu_temp(cgpu);
	cgpu->gpu_fan = (uint16_t) gpu_fanpercent(cgpu);
	cgpu->gpu_fan_rpm = (uint16_t) gpu_fanrpm(cgpu);
	cgpu->gpu_pstate = (int16_t) gpu_pstate(cgpu);
	cgpu->gpu_power = gpu_power(cgpu);
	cgpu->gpu_plimit = gpu_plimit(cgpu);
	gpu_info(cgpu);
#ifdef WIN32
	if (opt_debug) nvapi_pstateinfo(cgpu->gpu_id);
#endif
#endif

	if (cgpu->gpu_pstate != -1)
		snprintf(s->pstate, sizeof(s->pstate), "P%d", (int) cgpu->gpu_pstate);

	s->valid = true;
	s->gpu_id = gpu_id;
	s->bus = cgpu->gpu_bus;
	copy_str(s->card, sizeof(s->card), device_name[gpu_id]);
	s->arch = cgpu->gpu_arch;
	s->mem = (uint32_t) cgpu->gpu_mem;
	s->temp = cgpu->gpu_temp;
	s->fan = cgpu->gpu_fan;
	s->rpm = cgpu->gpu_fan_rpm;
	s->freq = cgpu->gpu_clock / 1000U;       // base clocks
	s->memfreq = cgpu->gpu_memclock / 1000U;
	s->gpuf = cgpu->monitor.gpu_clock;       // current
	s->memf = cgpu->monitor.gpu_memclock;
	s->power = cgpu->gpu_power;
	s->plim = cgpu->gpu_plimit;
	s->vid = cgpu->gpu_vid;
	s->pid = cgpu->gpu_pid;
	s->nvml_id = cgpu->nvml_id;
	s->nvapi_id = cgpu->nvapi_id;
	copy_str(s->sn, sizeof(s->sn), cgpu->gpu_sn);
	copy_str(s->bios, sizeof(s->bios), cgpu->gpu_desc);
}

int api_format_gpuhw_binary(const struct api_gpuhw_snapshot *s, char *out, size_t outlen)
{
	if (!s->valid) { if (outlen) out[0] = '\0'; return 0; }
	return snprintf(out, outlen, "GPU=%d;BUS=%hd;CARD=%s;SM=%hu;MEM=%u;"
		"TEMP=%.1f;FAN=%hu;RPM=%hu;FREQ=%u;MEMFREQ=%u;GPUF=%u;MEMF=%u;"
		"PST=%s;POWER=%u;PLIM=%u;"
		"VID=%hx;PID=%hx;NVML=%d;NVAPI=%d;SN=%s;BIOS=%s|",
		s->gpu_id, s->bus, s->card, s->arch, s->mem,
		s->temp, s->fan, s->rpm,
		s->freq, s->memfreq, s->gpuf, s->memf,
		s->pstate, s->power, s->plim,
		s->vid, s->pid, s->nvml_id, s->nvapi_id,
		s->sn, s->bios);
}

/* ------------------------------------------------------- system hardware */

#ifndef WIN32
static char os_version[64] = "linux ";
#endif

static const char* api_os_name()
{
#ifdef WIN32
	return "windows";
#else
	FILE *fd = fopen("/proc/version", "r");
	if (!fd)
		return "linux";
	if (!fscanf(fd, "Linux version %48s", &os_version[6])) {
		fclose(fd);
		return "linux";
	}
	fclose(fd);
	os_version[48] = '\0';
	return (const char*) os_version;
#endif
}

void api_collect_system(struct api_system_snapshot *s)
{
	memset(s, 0, sizeof(*s));
	copy_str(s->os, sizeof(s->os), api_os_name());
	copy_str(s->driver, sizeof(s->driver), driver_version);
	s->cpus = num_cpus;
	s->cpu_temp_c = (int) cpu_temp(0);
	s->cpu_clock_mhz = cpu_clock(0) / 1000;
	/* cpu_fanpercent() is a stub returning 0 on every platform here; 0 would
	 * mean *measured* zero, so report it as unavailable instead. */
	s->cpu_fan_pct = -1;
}

int api_format_system_binary(const struct api_system_snapshot *s, char *out, size_t outlen)
{
	return snprintf(out, outlen, "OS=%s;NVDRIVER=%s;CPUS=%d;CPUTEMP=%d;CPUFREQ=%d|",
		s->os, s->driver, s->cpus, s->cpu_temp_c, s->cpu_clock_mhz);
}

/* ---- JSON renderers ------------------------------------------------------
 * Nothing here reads a global the collector did not capture. Unit change from
 * the binary API (docs/api-rest.md): kH/s there, H/s here, converted once. */

/* Unavailable values must be null, never 0 — 0 means measured zero. */
static json_t *jnum_or_null(double v, bool available)
{
	return available ? json_real(v) : json_null();
}

static json_t *jint_or_null(json_int_t v, bool available)
{
	return available ? json_integer(v) : json_null();
}

json_t *api_build_miner_json(void)
{
	json_t *m = json_object();
	if (!m) return NULL;
	json_object_set_new(m, "name", json_string(PACKAGE_NAME));
	json_object_set_new(m, "version", json_string(PACKAGE_VERSION));
	json_object_set_new(m, "api_version", json_string("1.0"));  /* contract revision */
	json_object_set_new(m, "kind", json_string("gpu"));
	return m;
}

json_t *api_build_summary_json(const struct api_summary_snapshot *s)
{
	json_t *o = json_object(), *shares = json_object();
	json_t *diff = json_object(), *net = json_object(), *pools_o = json_object();
	if (!o || !shares || !diff || !net || !pools_o) {
		if (o) json_decref(o);
		if (shares) json_decref(shares);
		if (diff) json_decref(diff);
		if (net) json_decref(net);
		if (pools_o) json_decref(pools_o);
		return NULL;
	}

	json_object_set_new(o, "algo", json_string(s->algo));
	json_object_set_new(o, "uptime_s", json_integer((json_int_t) s->uptime));
	json_object_set_new(o, "timestamp", json_integer(s->ts));
	json_object_set_new(o, "hashrate_hs", json_real(s->khs * 1000.0));
	json_object_set_new(o, "hashrate_avg_hs", json_null());   /* not tracked yet */
	json_object_set_new(o, "devices", json_integer(s->gpus));
	json_object_set_new(o, "threads", json_integer(opt_n_threads));

	json_object_set_new(shares, "accepted", json_integer(s->accepted));
	json_object_set_new(shares, "rejected", json_integer(s->rejected));
	json_object_set_new(shares, "stale", json_null());        /* per-pool only */
	json_object_set_new(shares, "solved", json_integer(s->solved));
	json_object_set_new(shares, "accepted_per_min", json_real(s->accps));
	json_object_set_new(o, "shares", shares);

	json_object_set_new(diff, "pool", jnum_or_null(stratum_diff, stratum_diff > 0.0));
	json_object_set_new(diff, "network", jnum_or_null(net_diff, net_diff > 0.0));
	json_object_set_new(diff, "best_share", jnum_or_null(pools[cur_pooln].best_share,
	                                                     pools[cur_pooln].best_share > 0.0));
	json_object_set_new(o, "difficulty", diff);

	json_object_set_new(net, "hashrate_hs", jnum_or_null(s->netkhs * 1000.0, s->netkhs > 0.0));
	json_object_set_new(o, "network", net);

	json_object_set_new(pools_o, "count", json_integer(s->pools));
	json_object_set_new(pools_o, "active", json_integer(cur_pooln));
	json_object_set_new(pools_o, "wait_time_s", json_integer(s->wait_time));
	json_object_set_new(o, "pools", pools_o);

	return o;
}

json_t *api_build_thread_json(const struct api_thread_snapshot *s)
{
	json_t *o = json_object();
	if (!o) return NULL;
	json_object_set_new(o, "id", json_integer(s->id));
	json_object_set_new(o, "device_id", json_integer(s->gpu_id));
	json_object_set_new(o, "hashrate_hs", json_real(s->khs * 1000.0));
	json_object_set_new(o, "accepted", json_integer(s->acc));
	json_object_set_new(o, "rejected", json_integer(s->rej));
	json_object_set_new(o, "hw_errors", json_integer(s->hwf));
	json_object_set_new(o, "intensity", json_real(s->intensity));
	json_object_set_new(o, "throughput", json_integer(s->throughput));
	return o;
}

json_t *api_build_pool_json(int index, bool active, const struct api_pool_snapshot *s)
{
	json_t *o = json_object(), *shares = json_object();
	if (!o || !shares) {
		if (o) json_decref(o);
		if (shares) json_decref(shares);
		return NULL;
	}
	json_object_set_new(o, "index", json_integer(index));
	json_object_set_new(o, "active", json_boolean(active));
	json_object_set_new(o, "name", json_string(s->name));
	json_object_set_new(o, "url", json_string(s->url));
	json_object_set_new(o, "user", json_string(s->user));   /* the password is never returned */
	json_object_set_new(o, "algo", json_string(s->algo ? s->algo : ""));

	json_object_set_new(shares, "accepted", json_integer(s->accepted));
	json_object_set_new(shares, "rejected", json_integer(s->rejected));
	json_object_set_new(shares, "stale", json_integer(s->stales));
	json_object_set_new(shares, "solved", json_integer(s->solved));
	json_object_set_new(shares, "accepted_per_min", json_null());
	json_object_set_new(o, "shares", shares);

	json_object_set_new(o, "type", json_string(s->stratum ? "stratum" : "getwork"));
	json_object_set_new(o, "status", json_string(s->connected ? "connected" : "disconnected"));
	json_object_set_new(o, "stale", json_integer(s->stales));
	json_object_set_new(o, "difficulty", jnum_or_null(s->diff, s->diff > 0.0));
	json_object_set_new(o, "best_share", jnum_or_null(s->best_share, s->best_share > 0.0));

	/* The binary API exposes JOB/H/N2SZ/N2; dropping them here would be a
	 * regression against it. */
	json_t *job = json_object();
	if (job) {
		json_object_set_new(job, "id", s->job_id[0] ? json_string(s->job_id) : json_null());
		json_object_set_new(job, "height", jint_or_null(s->height, s->height > 0));
		json_object_set_new(job, "extranonce2_size", json_integer(s->xnonce2_size));
		json_object_set_new(job, "extranonce2", s->xnonce2[0] ? json_string(s->xnonce2) : json_null());
		json_object_set_new(o, "job", job);
	}
	json_object_set_new(o, "ping_ms", jint_or_null(s->ping_ms, s->ping_ms > 0));
	json_object_set_new(o, "disconnects", json_integer(s->disconnects));
	json_object_set_new(o, "wait_time_s", json_integer(s->wait_time));
	/* Pool session length; process uptime is `uptime_s` on /summary. */
	/* null rather than a climbing number while the pool is down: work_time keeps
	 * counting through an outage, which reported a live session for a dead socket. */
	json_object_set_new(o, "session_s", jint_or_null(s->work_time, s->connected));
	json_object_set_new(o, "last_share_age_s", jint_or_null(s->last_share_age, s->last_share_age > 0));
	return o;
}

json_t *api_build_device_json(const struct api_gpuhw_snapshot *hw,
                              const struct api_thread_snapshot *th)
{
	json_t *o = json_object(), *g = json_object();
	char idbuf[16];
	if (!o || !g) {
		if (o) json_decref(o);
		if (g) json_decref(g);
		return NULL;
	}

	json_object_set_new(o, "id", json_integer(hw->gpu_id));
	json_object_set_new(o, "type", json_string("gpu"));
	json_object_set_new(o, "name", json_string(hw->card));
	json_object_set_new(o, "temp_c", jnum_or_null(hw->temp, hw->temp > 0.0f));
	json_object_set_new(o, "fan_pct", jint_or_null(hw->fan, hw->fan > 0));
	json_object_set_new(o, "fan_rpm", jint_or_null(hw->rpm, hw->rpm > 0));
	json_object_set_new(o, "clock_mhz", jint_or_null(hw->gpuf, hw->gpuf > 0));
	json_object_set_new(o, "mem_clock_mhz", jint_or_null(hw->memf, hw->memf > 0));
	json_object_set_new(o, "power_mw", jint_or_null(hw->power, hw->power > 0));
	json_object_set_new(o, "power_limit_mw", jint_or_null(hw->plim, hw->plim > 0));
	json_object_set_new(o, "hashrate_hs",
		th && th->valid ? json_real(th->khs * 1000.0) : json_null());
	json_object_set_new(o, "hashrate_per_watt_khs",
		th && th->valid && th->khw > 0.0 ? json_real(th->khw) : json_null());

	json_object_set_new(g, "bus_id", json_integer(hw->bus));
	json_object_set_new(g, "sm", json_integer(hw->arch));
	json_object_set_new(g, "mem_bytes", json_integer((json_int_t) hw->mem * 1024 * 1024));
	json_object_set_new(g, "pstate", hw->pstate[0] ? json_string(hw->pstate) : json_null());
	json_object_set_new(g, "base_clock_mhz", jint_or_null(hw->freq, hw->freq > 0));
	json_object_set_new(g, "base_mem_clock_mhz", jint_or_null(hw->memfreq, hw->memfreq > 0));
	snprintf(idbuf, sizeof(idbuf), "0x%04hx", hw->vid);
	json_object_set_new(g, "vendor_id", json_string(idbuf));
	snprintf(idbuf, sizeof(idbuf), "0x%04hx", hw->pid);
	json_object_set_new(g, "device_id", json_string(idbuf));
	json_object_set_new(g, "serial", hw->sn[0] ? json_string(hw->sn) : json_null());
	json_object_set_new(g, "bios", hw->bios[0] ? json_string(hw->bios) : json_null());
	json_object_set_new(g, "nvml_id", json_integer(hw->nvml_id));
	json_object_set_new(g, "nvapi_id", json_integer(hw->nvapi_id));
	json_object_set_new(g, "monitoring", json_boolean(hw->temp > 0.0f || hw->power > 0));
	json_object_set_new(o, "gpu", g);

	return o;
}

json_t *api_build_system_json(const struct api_system_snapshot *s)
{
	json_t *o = json_object();
	if (!o) return NULL;
	json_object_set_new(o, "os", json_string(s->os));
	json_object_set_new(o, "driver", s->driver[0] ? json_string(s->driver) : json_null());
	json_object_set_new(o, "cpus", json_integer(s->cpus));
	json_object_set_new(o, "cpu_temp_c", jint_or_null(s->cpu_temp_c, s->cpu_temp_c > 0));
	json_object_set_new(o, "cpu_clock_mhz", jint_or_null(s->cpu_clock_mhz, s->cpu_clock_mhz > 0));
	json_object_set_new(o, "cpu_fan_pct", jint_or_null(s->cpu_fan_pct, s->cpu_fan_pct >= 0));
	return o;
}

/* ------------------------------------------------------------- metrics */

/* Snapshot for GET /metrics. Deliberately not api_collect_thread(): that one
 * queries NVML/NVAPI per field, and a scraper polls every 15-60 s. Only plain
 * globals and the monitor thread's cached sample are read here. */
void api_collect_metrics(api_metrics_input *in)
{
	memset(in, 0, sizeof(*in));

	static char algo_buf[64];
	get_currentalgo(algo_buf, sizeof(algo_buf));

	in->name = PACKAGE_NAME;
	in->version = PACKAGE_VERSION;
	in->kind = "gpu";
	in->algo = algo_buf;
	in->uptime_s = difftime(time(NULL), api_startup_time);
	in->hashrate_hs = mining_is_parked() ? 0. : (double) global_hashrate;
	/* A benchmark run has no chain and a fresh connection has no pool
	 * difficulty; 0 is a difficulty no chain has, so the series is omitted
	 * rather than fabricated. The input is memset above: without these two
	 * flags this miner would publish neither. */
	in->has_net_difficulty = net_diff > 0.;
	in->net_difficulty = net_diff;
	in->has_pool_difficulty = stratum_diff > 0.;
	in->pool_difficulty = stratum_diff;

	const ctl_state_t st = api_ctl_get_state();
	in->control_state = api_ctl_state_name(st);
	in->mining_active = !abort_flag && opt_n_threads > 0 &&
		st != CTL_PAUSED && st != CTL_STOPPED && global_hashrate > 0.;

	/* The per-pool counters are not monotonic — a recycled pool slot takes its
	 * predecessor's totals with it — and a counter that goes backwards makes
	 * rate() report a spike that never happened. Accumulate deltas. */
	static uint64_t tot_acc = 0, tot_rej = 0, tot_stale = 0, tot_solved = 0;
	static uint32_t seen_acc = 0, seen_rej = 0, seen_stale = 0, seen_solved = 0;
	uint32_t acc = 0, rej = 0, stale = 0, solved = 0;
	for (int p = 0; p < num_pools; p++) {
		acc += pools[p].accepted_count;
		rej += pools[p].rejected_count;
		stale += pools[p].stales_count;
		solved += pools[p].solved_count;
	}
	tot_acc    += (acc    >= seen_acc)    ? (acc    - seen_acc)    : acc;
	tot_rej    += (rej    >= seen_rej)    ? (rej    - seen_rej)    : rej;
	tot_stale  += (stale  >= seen_stale)  ? (stale  - seen_stale)  : stale;
	tot_solved += (solved >= seen_solved) ? (solved - seen_solved) : solved;
	seen_acc = acc; seen_rej = rej; seen_stale = stale; seen_solved = solved;

	in->shares_accepted = tot_acc;
	in->shares_rejected = tot_rej;
	in->shares_stale = tot_stale;
	in->blocks_solved = tot_solved;

	for (int i = 0; i < opt_n_threads && in->ndevices < API_METRICS_MAX_DEVICES; i++) {
		struct cgpu_info *cgpu = &thr_info[i].gpu;
		api_metrics_device *d = &in->devices[in->ndevices++];
		d->valid = true;
		d->device = cgpu->gpu_id;
		snprintf(d->type, sizeof(d->type), "gpu");
		snprintf(d->algo, sizeof(d->algo), "%s", algo_buf);
		d->hashrate_hs = mining_is_parked() ? 0.0 : stats_get_speed(i, 0.0);

		/* Only what the monitor thread sampled; it does not run under --quiet,
		 * so "no temperature" is normal and a fabricated 0 would poison every
		 * average built on it. */
		d->has_temp  = cgpu->monitor.gpu_temp  > 0;
		d->temp_c    = (double) cgpu->monitor.gpu_temp;
		d->has_power = cgpu->monitor.gpu_power > 0;
		d->power_w   = (double) cgpu->monitor.gpu_power / 1000.0;   /* mW -> W */
		d->has_fan   = cgpu->monitor.gpu_fan   > 0;
		d->fan_pct   = (double) cgpu->monitor.gpu_fan;
		d->has_hw_errors = true;
		d->hw_errors = cgpu->hw_errors;
	}

	for (int p = 0; p < num_pools && in->npools < API_METRICS_MAX_POOLS; p++) {
		if (pools[p].type == POOL_UNUSED)
			continue;
		api_metrics_pool *mp = &in->pools[in->npools++];
		mp->index = p;
		mp->active = (p == cur_pooln);
		snprintf(mp->url, sizeof(mp->url), "%s", pools[p].url);
		/* One live socket: connection state exists only for the pool in use,
		 * and reporting 0 for the others would invent a fault. */
		mp->stratum = mp->active && (pools[p].type & POOL_STRATUM) != 0;
		mp->connected = mp->stratum ? (stratum.curl != NULL) : false;
		mp->disconnects = pools[p].disconnects;
		mp->has_last_share = pools[p].last_share_time != 0;
		mp->last_share_age_s = mp->has_last_share
			? difftime(time(NULL), pools[p].last_share_time) : 0.0;
	}
}
