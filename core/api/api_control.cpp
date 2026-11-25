// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Control core — see api_control.h and docs/api-rest.md section 7.
 */

#include <stdio.h>
#include <string.h>
#include <unistd.h>   // usleep() on Linux (Windows shim: compat/unistd.h)

#include "miner.h"
#include "algos.h"
#include "api_control.h"

extern void restart_threads(void);

struct api_ctl {
	volatile ctl_state_t state;
	volatile bool pause_requested;   /* the only field mining threads read  */
	volatile bool hold_connection;   /* stratum: do not (re)connect         */
	volatile int  parked;            /* the only field mining threads write */
	volatile uint32_t epoch;         /* ++ on every accepted mutation       */
	time_t   state_since;
	time_t   last_switch_ts;
	uint32_t switch_count;
	char     last_error[128];
	bool     enabled;
	int      min_interval_s;
	int      park_timeout_ms;
	pthread_mutex_t lock;            /* serialises control callers only     */
	pthread_mutex_t park_lock;       /* guards `parked` only                */
};

static struct api_ctl ctl = { CTL_RUNNING, false, false, 0, 0, 0, 0, 0, { 0 }, false, 15, 10000 };

/* --- control worker ----------------------------------------------------- */

/* Mutations run here, not on the single-threaded accept loop, which they would
 * stall for the whole barrier. One slot: a second request is refused, not
 * queued. The bounded wait is what makes a 202 possible. */
typedef enum { WK_STATE = 0, WK_PROFILE } wk_kind_t;

static struct {
	pthread_t       thread;
	pthread_mutex_t lock;
	pthread_cond_t  posted;          /* -> worker: a request is waiting     */
	pthread_cond_t  finished;        /* -> caller: your request is done     */
	bool            running;
	bool            stop;
	bool            pending;         /* posted, not yet picked up           */
	bool            busy;            /* picked up, still applying           */
	bool            done;
	uint32_t        seq;             /* identifies the slot's occupant      */
	wk_kind_t       kind;
	ctl_state_t     target;
	ctl_profile_req profile;         /* by value: see api_control.h         */
	ctl_result_t    result;
	char            error[160];
} wk;

static struct {
	pthread_mutex_t lock;
	bool  pending;
	bool  done;
	bool  ok;
	char  url[1024];
	int   index;      /* used when url is empty */
	int   algo;       /* pin the new entry's algo, -1 to leave it */
} poolreq;

static void *ctl_worker(void *arg);
static ctl_result_t profile_run(const ctl_profile_req *p);

void api_ctl_init(bool enabled, int min_interval_s, int park_timeout_ms)
{
	pthread_mutex_init(&ctl.lock, NULL);
	pthread_mutex_init(&ctl.park_lock, NULL);
	pthread_mutex_init(&poolreq.lock, NULL);
	ctl.enabled = enabled;
	/* 0 is a deliberate setting (no anti-flap), so only a negative falls back. */
	ctl.min_interval_s = min_interval_s >= 0 ? min_interval_s : 15;
	/* Generous on purpose: a thread only reaches the barrier between scans, and
	 * that gap is set by the algo and the job cadence, not by this code. On
	 * sha3t the median park is ~140ms but the p99 is ~5.4s. */
	ctl.park_timeout_ms = park_timeout_ms > 0 ? park_timeout_ms : 30000;
	ctl.state = CTL_RUNNING;
	ctl.state_since = time(NULL);

	if (!enabled)
		return;                       /* no flag, no thread */

	pthread_mutex_init(&wk.lock, NULL);
	pthread_cond_init(&wk.posted, NULL);
	pthread_cond_init(&wk.finished, NULL);
	if (pthread_create(&wk.thread, NULL, ctl_worker, NULL)) {
		applog(LOG_ERR, "control API: cannot start worker thread, disabled");
		ctl.enabled = false;
		return;
	}
	wk.running = true;

	applog(LOG_INFO, "control API enabled (min interval %ds, park timeout %dms)",
		ctl.min_interval_s, ctl.park_timeout_ms);
}

void api_ctl_shutdown(void)
{
	/* Clearing the request first means any thread still in the park loop
	 * leaves it rather than waiting out its slice while we exit. */
	ctl.pause_requested = false;
	ctl.hold_connection = false;

	if (!wk.running)
		return;

	pthread_mutex_lock(&wk.lock);
	wk.stop = true;
	wk.running = false;
	pthread_cond_broadcast(&wk.posted);
	pthread_mutex_unlock(&wk.lock);
	pthread_join(wk.thread, NULL);
}

/* --- mining-loop hooks ------------------------------------------------- */

bool api_ctl_wants_pause(void)
{
	return ctl.enabled && ctl.pause_requested;
}

void api_ctl_park(int thr_id)
{
	(void) thr_id;
	if (!ctl.enabled || !ctl.pause_requested)
		return;

	/* Announce, then wait in short slices. The control caller polls `parked`,
	 * so it must be incremented before the first sleep, not after. */
	pthread_mutex_lock(&ctl.park_lock);
	ctl.parked++;
	pthread_mutex_unlock(&ctl.park_lock);

	/* 10ms slices: an idle thread polling a bool costs nothing, and the slice
	 * is the floor on how long a resume takes to be honoured. */
	while (ctl.pause_requested && !abort_flag)
		usleep(10 * 1000);

	pthread_mutex_lock(&ctl.park_lock);
	ctl.parked--;
	pthread_mutex_unlock(&ctl.park_lock);
}

/* --- control side ------------------------------------------------------ */

const char *api_ctl_state_name(ctl_state_t s)
{
	switch (s) {
	case CTL_RUNNING:   return "running";
	case CTL_PAUSED:    return "paused";
	case CTL_STOPPED:   return "stopped";
	case CTL_SWITCHING: return "switching";
	}
	return "unknown";
}

static bool wait_for_park(int want, int timeout_ms)
{
	const int slice = 20;
	for (int waited = 0; waited < timeout_ms; waited += slice) {
		if (ctl.parked >= want)
			return true;
		if (abort_flag)
			return false;
		usleep(slice * 1000);
	}
	return ctl.parked >= want;
}

/* A resume is complete only once the barrier is empty: publishing `running`
 * earlier lets the next pause count a thread on its way out. */
static bool wait_for_drain(int timeout_ms)
{
	const int slice = 5;
	for (int waited = 0; waited < timeout_ms; waited += slice) {
		if (ctl.parked <= 0)
			return true;
		if (abort_flag)
			return false;
		usleep(slice * 1000);
	}
	return ctl.parked <= 0;
}

int api_ctl_threads_total(void) { return opt_n_threads; }

/* Threads are already parked, so no new share can appear; the wait lets
 * already-submitted ones be acked before the socket goes. */
static void drop_connection(void)
{
	extern struct stratum_ctx stratum;
	if (!have_stratum || !stratum.curl)
		return;
	usleep(500 * 1000);              /* bounded ack drain */
	stratum_disconnect_intentional(&stratum);
	applog(LOG_INFO, "control: pool connection released");
}

/* Worker thread. The park timeout is the miner's, not the caller's: an
 * impatient caller gets 202, it does not shorten or abandon the barrier. */
static ctl_result_t ctl_apply(ctl_state_t target)
{
	if (pthread_mutex_trylock(&ctl.lock) != 0)
		return CTL_BUSY;              /* a mutation is already in flight */

	if (ctl.state == CTL_SWITCHING) {
		pthread_mutex_unlock(&ctl.lock);
		return CTL_BUSY;
	}

	if (ctl.state == target) {
		pthread_mutex_unlock(&ctl.lock);
		return CTL_OK;
	}

	const ctl_state_t previous = ctl.state;
	const bool want_parked = (target == CTL_PAUSED || target == CTL_STOPPED);

	ctl.state = CTL_SWITCHING;
	ctl.last_error[0] = '\0';

	if (want_parked) {
		ctl.pause_requested = true;
		restart_threads();            /* break the scan loops now */

		const int timeout = ctl.park_timeout_ms;
		if (!wait_for_park(opt_n_threads, timeout)) {
			/* Abandoned, not forced: restore and report. */
			ctl.pause_requested = false;
			ctl.state = previous;
			snprintf(ctl.last_error, sizeof(ctl.last_error),
				"only %d of %d threads parked within %dms",
				ctl.parked, opt_n_threads, timeout);
			pthread_mutex_unlock(&ctl.lock);
			return CTL_TIMEOUT;
		}
		ctl.hold_connection = (target == CTL_STOPPED);
		if (target == CTL_STOPPED)
			drop_connection();
	} else {
		/* Resume: clear the request, then wait for the barrier to empty. */
		ctl.pause_requested = false;
		ctl.hold_connection = false;
		restart_threads();
		if (!wait_for_drain(ctl.park_timeout_ms)) {
			ctl.state = previous;
			snprintf(ctl.last_error, sizeof(ctl.last_error),
				"%d thread(s) still parked after %dms", ctl.parked, ctl.park_timeout_ms);
			pthread_mutex_unlock(&ctl.lock);
			return CTL_TIMEOUT;
		}
	}

	ctl.state = target;
	ctl.state_since = time(NULL);
	ctl.switch_count++;
	ctl.epoch++;

	pthread_mutex_unlock(&ctl.lock);

	applog(LOG_INFO, "control: %s -> %s (epoch %u)",
		api_ctl_state_name(previous), api_ctl_state_name(target), ctl.epoch);
	return CTL_OK;
}

static void *ctl_worker(void *arg)
{
	(void) arg;
	pthread_mutex_lock(&wk.lock);
	while (!wk.stop) {
		if (!wk.pending) {
			pthread_cond_wait(&wk.posted, &wk.lock);
			continue;
		}
		const wk_kind_t kind = wk.kind;
		const ctl_state_t target = wk.target;
		static ctl_profile_req job;   /* worker-private copy of the payload */
		if (kind == WK_PROFILE)
			job = wk.profile;
		wk.pending = false;
		wk.busy = true;
		pthread_mutex_unlock(&wk.lock);

		const ctl_result_t rc = (kind == WK_PROFILE) ? profile_run(&job)
		                                             : ctl_apply(target);

		pthread_mutex_lock(&wk.lock);
		wk.result = rc;
		snprintf(wk.error, sizeof(wk.error), "%s",
			ctl.last_error[0] ? ctl.last_error : "");
		wk.busy = false;
		wk.done = true;
		pthread_cond_broadcast(&wk.finished);
	}
	pthread_mutex_unlock(&wk.lock);
	return NULL;
}

/* Post one job, wait up to wait_ms. wait_ms <= 0 is the "fire and poll" mode;
 * on timeout the caller gets CTL_ACCEPTED and the job keeps running. */
static ctl_result_t submit(wk_kind_t kind, ctl_state_t target,
                           const ctl_profile_req *profile, int wait_ms,
                           char *err, size_t errlen)
{
	pthread_mutex_lock(&wk.lock);

	if (!wk.running) {
		pthread_mutex_unlock(&wk.lock);
		return CTL_DISABLED;
	}
	if (wk.pending || wk.busy) {
		pthread_mutex_unlock(&wk.lock);
		return CTL_BUSY;
	}

	const uint32_t mine = ++wk.seq;
	wk.kind    = kind;
	wk.target  = target;
	if (profile)
		wk.profile = *profile;
	wk.pending = true;
	wk.done    = false;
	wk.error[0] = '\0';
	pthread_cond_signal(&wk.posted);

	if (wait_ms <= 0) {
		pthread_mutex_unlock(&wk.lock);
		return CTL_ACCEPTED;
	}

	struct timeval now;
	gettimeofday(&now, NULL);
	const long long us = (long long) now.tv_usec + (long long) wait_ms * 1000;

	struct timespec abstime;
	abstime.tv_sec  = now.tv_sec + (time_t) (us / 1000000);
	abstime.tv_nsec = (long) (us % 1000000) * 1000;

	/* seq guards against waking on someone else's completion. */
	while (!(wk.done && wk.seq == mine)) {
		if (pthread_cond_timedwait(&wk.finished, &wk.lock, &abstime) != 0)
			break;
	}

	ctl_result_t rc = CTL_ACCEPTED;
	if (wk.done && wk.seq == mine) {
		rc = wk.result;
		if (err && errlen && wk.error[0])
			snprintf(err, errlen, "%s", wk.error);
	}
	pthread_mutex_unlock(&wk.lock);
	return rc;
}

ctl_result_t api_ctl_set_state(ctl_state_t target, int wait_ms)
{
	if (!ctl.enabled)
		return CTL_DISABLED;
	if (target == CTL_SWITCHING)
		return CTL_BUSY;

	/* No-op: same state. Answer success without advancing epoch, so a manager
	 * can re-assert what it wants instead of tracking what it last sent. */
	if (ctl.state == target)
		return CTL_OK;

	return submit(WK_STATE, target, NULL, wait_ms, NULL, 0);
}

ctl_result_t api_ctl_profile(const ctl_profile_req *req, int wait_ms,
                             char *err, size_t errlen)
{
	if (!ctl.enabled)
		return CTL_DISABLED;

	/* Only a real reconfiguration is throttled. A body carrying nothing but
	 * `run` is a run-state verb wearing a different hat, and the contract exempts
	 * those (docs/api-rest.md section 7.1). */
	const bool reconfigures = req->has_algo || req->has_pool ||
	                          req->has_pool_index || req->nparams > 0;
	if (reconfigures && !api_ctl_ready_for_switch())
		return CTL_THROTTLED;

	return submit(WK_PROFILE, CTL_RUNNING, req, wait_ms, err, errlen);
}

/* A posted-but-unfinished request IS a switch in progress, including the window
 * where ctl_apply has stored the target but still holds the lock — otherwise
 * "poll until state leaves switching" returns too early. */
ctl_state_t api_ctl_get_state(void)
{
	return (wk.pending || wk.busy) ? CTL_SWITCHING : ctl.state;
}
uint32_t    api_ctl_epoch(void)         { return ctl.epoch; }
uint32_t    api_ctl_switch_count(void)  { return ctl.switch_count; }
time_t      api_ctl_state_since(void)   { return ctl.state_since; }
time_t      api_ctl_last_switch(void)   { return ctl.last_switch_ts; }
int         api_ctl_parked(void)        { return ctl.parked; }
bool        api_ctl_enabled(void)       { return ctl.enabled; }
bool        api_ctl_hold_connection(void) { return ctl.enabled && ctl.hold_connection; }
int         api_ctl_min_interval(void)  { return ctl.min_interval_s; }
const char *api_ctl_last_error(void)    { return ctl.last_error[0] ? ctl.last_error : NULL; }

/* Profile switches only; the run-state verbs are exempt (docs/api-rest.md
 * section 7.1). The clock is started by api_ctl_note_switch(). */
bool api_ctl_ready_for_switch(void)
{
	if (!ctl.last_switch_ts)
		return true;
	return difftime(time(NULL), ctl.last_switch_ts) >= (double) ctl.min_interval_s;
}

int api_ctl_retry_after_s(void)
{
	if (api_ctl_ready_for_switch())
		return 0;
	int left = ctl.min_interval_s - (int) difftime(time(NULL), ctl.last_switch_ts);
	return left > 0 ? left : 1;
}

void api_ctl_note_switch(void)
{
	ctl.last_switch_ts = time(NULL);
}

/* --- algo parameters ---------------------------------------------------- */

/* Owned by ccminer.cpp, which is C++: declared here rather than in miner.h,
 * whose one big extern "C" block would give them the wrong linkage. */
extern void parse_arg(int key, char *arg);
extern uint32_t yescrypt_param_N, yescrypt_param_r, yescrypt_param_p;
extern char *yescrypt_key;
extern size_t yescrypt_key_len;
extern uint32_t gpus_intensity[MAX_GPUS];
extern int device_bfactor[MAX_GPUS];
extern int device_lookup_gap[MAX_GPUS];
extern int device_interactive[MAX_GPUS];
extern int device_texturecache[MAX_GPUS];
extern char *device_config[MAX_GPUS];
extern char *opt_scratchpad_url;

static const ctl_param_def_t param_defs[] = {
	{ "n",              CTLP_INT,        false },
	{ "r",              CTLP_INT,        false },
	{ "p",              CTLP_INT,        false },
	{ "key",            CTLP_STRING,     false },
	{ "intensity",      CTLP_FLOAT_LIST, false },
	{ "launch_config",  CTLP_STRING,     false },
	{ "lookup_gap",     CTLP_INT_LIST,   false },
	{ "texture_cache",  CTLP_INT_LIST,   false },
	{ "interactive",    CTLP_INT_LIST,   false },
	{ "bfactor",        CTLP_INT_LIST,   false },
	{ "scratchpad_url", CTLP_STRING,     true  },
};

/* Which algo consumes which. `intensity` is the only universal one: everything
 * else is rejected for an algo that would ignore it. */
static const char *p_yescrypt[]  = { "n", "r", "p", "key", "intensity" };
static const char *p_scrypt[]    = { "launch_config", "lookup_gap", "texture_cache",
                                     "interactive", "intensity" };
static const char *p_cn[]        = { "launch_config", "bfactor", "intensity" };
static const char *p_wildkeccak[]= { "scratchpad_url", "launch_config", "intensity" };
static const char *p_default[]   = { "intensity" };

const ctl_param_def_t *api_ctl_param_defs(size_t *count)
{
	if (count) *count = ARRAY_SIZE(param_defs);
	return param_defs;
}

const ctl_param_def_t *api_ctl_param_find(const char *name)
{
	for (size_t i = 0; i < ARRAY_SIZE(param_defs); i++)
		if (!strcmp(param_defs[i].name, name))
			return &param_defs[i];
	return NULL;
}

const char **api_ctl_algo_params(int algo, size_t *count)
{
	switch (algo) {
	/* Only the generic yescrypt reads the globals; the rN variants pass their
	 * own constants to the same kernel, so a parameter there would be a lie. */
	case ALGO_YESCRYPT:
		*count = ARRAY_SIZE(p_yescrypt);   return p_yescrypt;
	case ALGO_SCRYPT:
	case ALGO_SCRYPT_JANE:
		*count = ARRAY_SIZE(p_scrypt);     return p_scrypt;
	case ALGO_CRYPTONIGHT:
	case ALGO_CRYPTOLIGHT:
		*count = ARRAY_SIZE(p_cn);         return p_cn;
	case ALGO_WILDKECCAK:
		*count = ARRAY_SIZE(p_wildkeccak); return p_wildkeccak;
	default:
		*count = ARRAY_SIZE(p_default);    return p_default;
	}
}

bool api_ctl_algo_accepts(int algo, const char *name)
{
	size_t count = 0;
	const char **names = api_ctl_algo_params(algo, &count);
	for (size_t i = 0; i < count; i++)
		if (!strcmp(names[i], name))
			return true;
	return false;
}

/* Every value is range-checked here before parse_arg() sees it, because
 * parse_arg's own error path is show_usage_and_exit() — reachable from the
 * network it would turn a bad parameter into a remote process kill. */
static bool value_ok(const ctl_param_def_t *def, const char *v, char *err, size_t errlen)
{
	if (def->type == CTLP_STRING) {
		if (!strcmp(def->name, "key") && strlen(v) > YESCRYPT_KEY_MAX) {
			snprintf(err, errlen, "key: at most %d bytes", YESCRYPT_KEY_MAX);
			return false;
		}
		return true;
	}

	const char *p = v;
	int fields = 0;
	while (*p) {
		char *end = NULL;
		double d = strtod(p, &end);
		if (end == p) {
			snprintf(err, errlen, "%s: '%s' is not a number", def->name, v);
			return false;
		}
		if (!strcmp(def->name, "intensity") && (d < 0.0 || d > 31.0)) {
			snprintf(err, errlen, "intensity must be 0..31");
			return false;
		}
		if (def->type != CTLP_FLOAT_LIST && d != (double) (long) d) {
			snprintf(err, errlen, "%s must be an integer", def->name);
			return false;
		}
		if (d < 0.0 || d > 1e9) {
			snprintf(err, errlen, "%s is out of range", def->name);
			return false;
		}
		fields++;
		p = end;
		if (*p == ',') p++;
		else if (*p) {
			snprintf(err, errlen, "%s: unexpected '%c'", def->name, *p);
			return false;
		}
		if (fields > MAX_GPUS) {
			snprintf(err, errlen, "%s: too many values", def->name);
			return false;
		}
	}
	if (!fields) {
		snprintf(err, errlen, "%s is empty", def->name);
		return false;
	}
	if (def->type == CTLP_INT && fields != 1) {
		snprintf(err, errlen, "%s takes a single value", def->name);
		return false;
	}
	return true;
}

/* The startup defaults, captured before anything is changed, so a null can put
 * a parameter back exactly where the command line left it. */
static struct {
	bool captured;
	uint32_t yN, yr, yp;
	char *ykey;
	uint32_t intensity[MAX_GPUS];
	int bfactor[MAX_GPUS], lookup_gap[MAX_GPUS];
	int interactive[MAX_GPUS], texturecache[MAX_GPUS];
	char *config[MAX_GPUS];
	char *scratchpad;
} defaults;

static void capture_defaults(void)
{
	if (defaults.captured)
		return;
	defaults.yN = yescrypt_param_N;
	defaults.yr = yescrypt_param_r;
	defaults.yp = yescrypt_param_p;
	defaults.ykey = yescrypt_key ? strdup(yescrypt_key) : NULL;
	defaults.scratchpad = opt_scratchpad_url ? strdup(opt_scratchpad_url) : NULL;
	for (int i = 0; i < MAX_GPUS; i++) {
		defaults.intensity[i] = gpus_intensity[i];
		defaults.bfactor[i] = device_bfactor[i];
		defaults.lookup_gap[i] = device_lookup_gap[i];
		defaults.interactive[i] = device_interactive[i];
		defaults.texturecache[i] = device_texturecache[i];
		defaults.config[i] = device_config[i];
	}
	defaults.captured = true;
}

static void param_reset(const char *name)
{
	capture_defaults();
	if (!strcmp(name, "n")) yescrypt_param_N = defaults.yN;
	else if (!strcmp(name, "r")) yescrypt_param_r = defaults.yr;
	else if (!strcmp(name, "p")) yescrypt_param_p = defaults.yp;
	else if (!strcmp(name, "key")) {
		free(yescrypt_key);
		yescrypt_key = defaults.ykey ? strdup(defaults.ykey) : NULL;
		yescrypt_key_len = yescrypt_key ? strlen(yescrypt_key) : 0;
	}
	else if (!strcmp(name, "scratchpad_url")) {
		free(opt_scratchpad_url);
		opt_scratchpad_url = defaults.scratchpad ? strdup(defaults.scratchpad) : NULL;
	}
	else for (int i = 0; i < MAX_GPUS; i++) {
		if (!strcmp(name, "intensity"))          gpus_intensity[i] = defaults.intensity[i];
		else if (!strcmp(name, "bfactor"))       device_bfactor[i] = defaults.bfactor[i];
		else if (!strcmp(name, "lookup_gap"))    device_lookup_gap[i] = defaults.lookup_gap[i];
		else if (!strcmp(name, "interactive"))   device_interactive[i] = defaults.interactive[i];
		else if (!strcmp(name, "texture_cache")) device_texturecache[i] = defaults.texturecache[i];
		else if (!strcmp(name, "launch_config")) device_config[i] = defaults.config[i];
	}
}

bool api_ctl_param_check(int algo, const char *name, const char *value,
                         char *err, size_t errlen)
{
	const ctl_param_def_t *def = api_ctl_param_find(name);
	if (!def) {
		snprintf(err, errlen, "unknown parameter '%s'", name);
		return false;
	}
	if (!api_ctl_algo_accepts(algo, name)) {
		snprintf(err, errlen, "algo %s does not accept parameter '%s'",
			algo_names[algo] ? algo_names[algo] : "?", name);
		return false;
	}
	if (!value)
		return true;                  /* null is always legal: reset */
	return value_ok(def, value, err, errlen);
}

bool api_ctl_param_apply(int algo, const char *name, const char *value,
                         char *err, size_t errlen)
{
	const ctl_param_def_t *def = api_ctl_param_find(name);
	if (!def) {
		snprintf(err, errlen, "unknown parameter '%s'", name);
		return false;
	}
	if (!api_ctl_algo_accepts(algo, name)) {
		snprintf(err, errlen, "algo %s does not accept parameter '%s'",
			algo_names[algo] ? algo_names[algo] : "?", name);
		return false;
	}

	capture_defaults();

	if (!value) {
		param_reset(name);
		return true;
	}
	if (!value_ok(def, value, err, errlen))
		return false;

	/* parse_arg() owns the comma-list and per-device fan-out quirks; it gets a
	 * writable copy because it runs strtok over the argument. */
	char arg[CTL_PARAM_VALUE_MAX + 32];
	snprintf(arg, sizeof(arg), "%s", value);

	if (!strcmp(name, "n") || !strcmp(name, "r") || !strcmp(name, "p")) {
		/* One component at a time, so the other two keep their values. */
		const uint32_t v = (uint32_t) atoi(value);
		if (!strcmp(name, "n")) yescrypt_param_N = v;
		else if (!strcmp(name, "r")) yescrypt_param_r = v;
		else yescrypt_param_p = v;
		return true;
	}
	if (!strcmp(name, "key")) {
		free(yescrypt_key);
		yescrypt_key = strdup(value);
		yescrypt_key_len = strlen(value);
		return true;
	}
	if (!strcmp(name, "scratchpad_url")) { parse_arg('k', arg); return true; }
	if (!strcmp(name, "intensity"))      { parse_arg('i', arg); return true; }
	if (!strcmp(name, "launch_config"))  { parse_arg('l', arg); return true; }
	if (!strcmp(name, "lookup_gap"))     { parse_arg('L', arg); return true; }
	if (!strcmp(name, "interactive"))    { parse_arg(1050, arg); return true; }
	if (!strcmp(name, "texture_cache"))  { parse_arg(1051, arg); return true; }
	if (!strcmp(name, "bfactor"))        { parse_arg(1055, arg); return true; }

	snprintf(err, errlen, "parameter '%s' has no application path", name);
	return false;
}

const char *api_ctl_param_get(int algo, const char *name, char *buf, size_t buflen)
{
	if (!api_ctl_algo_accepts(algo, name))
		return NULL;

	if (!strcmp(name, "n")) {
		if (!yescrypt_param_N) return NULL;
		snprintf(buf, buflen, "%u", yescrypt_param_N); return buf;
	}
	if (!strcmp(name, "r")) {
		if (!yescrypt_param_r) return NULL;
		snprintf(buf, buflen, "%u", yescrypt_param_r); return buf;
	}
	if (!strcmp(name, "p")) {
		if (!yescrypt_param_p) return NULL;
		snprintf(buf, buflen, "%u", yescrypt_param_p); return buf;
	}
	if (!strcmp(name, "key"))
		return yescrypt_key ? (snprintf(buf, buflen, "%s", yescrypt_key), buf) : NULL;
	if (!strcmp(name, "scratchpad_url"))
		return opt_scratchpad_url ? (snprintf(buf, buflen, "%s", opt_scratchpad_url), buf) : NULL;
	if (!strcmp(name, "launch_config"))
		return device_config[0] ? (snprintf(buf, buflen, "%s", device_config[0]), buf) : NULL;
	if (!strcmp(name, "intensity")) {
		if (!gpus_intensity[0]) return NULL;
		snprintf(buf, buflen, "%u", gpus_intensity[0]); return buf;
	}
	if (!strcmp(name, "bfactor"))    { snprintf(buf, buflen, "%d", device_bfactor[0]); return buf; }
	if (!strcmp(name, "lookup_gap")) { snprintf(buf, buflen, "%d", device_lookup_gap[0]); return buf; }
	if (!strcmp(name, "interactive")){ snprintf(buf, buflen, "%d", device_interactive[0]); return buf; }
	if (!strcmp(name, "texture_cache")) { snprintf(buf, buflen, "%d", device_texturecache[0]); return buf; }
	return NULL;
}

/* Sticky store. Small and fixed: a rig rotates between a handful of algos, and
 * an unbounded cache here would be a memory leak with extra steps. */
#define CTL_STICKY_ALGOS 8

static struct {
	int  algo;
	bool used;
	int  n;
	char name[CTL_MAX_PARAMS][CTL_PARAM_NAME_MAX];
	char value[CTL_MAX_PARAMS][CTL_PARAM_VALUE_MAX];
	bool isnull[CTL_MAX_PARAMS];
} sticky[CTL_STICKY_ALGOS];

static int sticky_slot(int algo, bool create)
{
	int free_slot = -1, oldest = 0;
	for (int i = 0; i < CTL_STICKY_ALGOS; i++) {
		if (sticky[i].used && sticky[i].algo == algo) return i;
		if (!sticky[i].used && free_slot < 0) free_slot = i;
	}
	if (!create) return -1;
	if (free_slot < 0) free_slot = oldest;   /* overwrite slot 0, see above */
	memset(&sticky[free_slot], 0, sizeof(sticky[0]));
	sticky[free_slot].used = true;
	sticky[free_slot].algo = algo;
	return free_slot;
}

void api_ctl_params_remember(int algo, const char *name, const char *value)
{
	const int s = sticky_slot(algo, true);
	if (s < 0) return;
	for (int i = 0; i < sticky[s].n; i++) {
		if (!strcmp(sticky[s].name[i], name)) {
			sticky[s].isnull[i] = (value == NULL);
			snprintf(sticky[s].value[i], CTL_PARAM_VALUE_MAX, "%s", value ? value : "");
			return;
		}
	}
	if (sticky[s].n >= CTL_MAX_PARAMS) return;
	const int i = sticky[s].n++;
	snprintf(sticky[s].name[i], CTL_PARAM_NAME_MAX, "%s", name);
	snprintf(sticky[s].value[i], CTL_PARAM_VALUE_MAX, "%s", value ? value : "");
	sticky[s].isnull[i] = (value == NULL);
}

void api_ctl_params_restore(int algo)
{
	const int s = sticky_slot(algo, false);
	if (s < 0)
		return;
	char err[128];
	for (int i = 0; i < sticky[s].n; i++)
		api_ctl_param_apply(algo, sticky[s].name[i],
			sticky[s].isnull[i] ? NULL : sticky[s].value[i], err, sizeof(err));
}

/* --- pool switch handshake ---------------------------------------------- */

bool api_ctl_pool_request_take(char *url, size_t urllen, int *index, int *algo)
{
	if (!ctl.enabled || !poolreq.pending)
		return false;
	pthread_mutex_lock(&poolreq.lock);
	const bool got = poolreq.pending;
	if (got) {
		snprintf(url, urllen, "%s", poolreq.url);
		*index = poolreq.index;
		*algo = poolreq.algo;
		poolreq.pending = false;
	}
	pthread_mutex_unlock(&poolreq.lock);
	return got;
}

bool api_ctl_pool_request_pending(void)
{
	return ctl.enabled && poolreq.pending;
}

static volatile bool switch_in_flight = false;

bool api_ctl_switch_in_flight(void)
{
	return ctl.enabled && switch_in_flight;
}

void api_ctl_pool_request_done(bool ok)
{
	pthread_mutex_lock(&poolreq.lock);
	poolreq.ok = ok;
	poolreq.done = true;
	pthread_mutex_unlock(&poolreq.lock);
}

/* The stratum thread only reaches the handover between lines, so a quiet pool
 * makes a switch slow, never unsafe. */
static bool pool_switch_via_stratum(const char *packed, int index, int algo,
                                    char *err, size_t errlen)
{
	if (!have_stratum) {
		/* getwork/GBT: no stratum thread owns anything, so the old direct
		 * path is the safe one here. */
		const bool ok = (index >= 0) ? pool_switch(-1, index)
		                             : pool_switch_url_algo((char *) packed, algo);
		if (!ok) {
			snprintf(err, errlen, "pool switch failed");
			return false;
		}
		return true;
	}

	pthread_mutex_lock(&poolreq.lock);
	snprintf(poolreq.url, sizeof(poolreq.url), "%s", packed ? packed : "");
	poolreq.index = index;
	poolreq.algo = algo;
	poolreq.done = false;
	poolreq.ok = false;
	poolreq.pending = true;
	pthread_mutex_unlock(&poolreq.lock);

	const int slice = 20;
	for (int waited = 0; waited < ctl.park_timeout_ms; waited += slice) {
		if (poolreq.done)
			break;
		if (abort_flag)
			break;
		usleep(slice * 1000);
	}

	pthread_mutex_lock(&poolreq.lock);
	const bool done = poolreq.done, ok = poolreq.ok;
	poolreq.pending = false;
	pthread_mutex_unlock(&poolreq.lock);

	if (!done) {
		snprintf(err, errlen, "stratum thread did not take the pool switch within %dms",
			ctl.park_timeout_ms);
		return false;
	}
	if (!ok) {
		snprintf(err, errlen, "pool switch failed");
		return false;
	}
	return true;
}

/* --- profile: algo + params + pool + run, applied atomically ------------ */

/*
 * Worker thread, every mining thread parked. Each step records what it replaced
 * so a failure walks back before the threads are released: a half-switched
 * miner (new algo, old pool) mines for nobody while looking healthy.
 */
static bool profile_apply(const ctl_profile_req *p, char *err, size_t errlen)
{
	const int   old_algo = opt_algo;
	const bool  algo_changed = p->has_algo && p->algo != opt_algo;
	bool variant_changed = false;
	bool params_touched = false;
	char old_variant[32];

	/* Which yespower coin was selected, for the same walk-back reason: the algo
	 * int does not carry it, so restoring opt_algo alone would leave the previous
	 * algo running the new coin's (N, r, pers). */
	snprintf(old_variant, sizeof(old_variant), "%s", yespower_variant_name());

	if (algo_changed)
		opt_algo = (enum sha_algos) p->algo;

	/* Not gated on the enum moving: `algo_name` exists because the int loses the
	 * coin, and two yespower coins can share one enum with different (N, r, pers).
	 * Mirrors the same rule in pools.cpp. */
	if (p->has_algo &&
	    (opt_algo == ALGO_YESPOWER || opt_algo == ALGO_YESPOWERR16)) {
		if (!yespower_set_variant(p->algo_name)) {
			opt_algo = (enum sha_algos) old_algo;
			yespower_set_variant(old_variant);
			snprintf(err, errlen, "algo '%s' has no yespower parameter entry",
			         p->algo_name);
			return false;
		}
		variant_changed = strcmp(old_variant, yespower_variant_name()) != 0;
	}

	/* Sticky params for the incoming algo first, so an explicit params object
	 * in this same request wins over what was remembered. */
	if (algo_changed)
		api_ctl_params_restore(opt_algo);

	for (int i = 0; i < p->nparams; i++) {
		const char *v = p->pnull[i] ? NULL : p->pvalue[i];
		if (!api_ctl_param_apply(opt_algo, p->pname[i], v, err, errlen)) {
			if (algo_changed) {
				opt_algo = (enum sha_algos) old_algo;
				yespower_set_variant(old_variant);
				api_ctl_params_restore(opt_algo);
			} else if (variant_changed) {
				/* opt_algo never moved, so restoring it is a no-op and
				* api_ctl_params_restore would undo this request's own
				* params. Only the coin needs walking back. */
				yespower_set_variant(old_variant);
			}
			return false;
		}
		api_ctl_params_remember(opt_algo, p->pname[i], v);
		params_touched = true;
	}
	(void) params_touched;

	if (p->has_pool || p->has_pool_index) {
		switch_in_flight = true;
		/* Drain window before the pool changes. Threads are parked, so no new
		 * share can be found, but a share found just before the barrier is
		 * submitted asynchronously and would otherwise land on the new
		 * connection as "Invalid job id" — real work, thrown away. */
		if (have_stratum)
			usleep(500 * 1000);
	}

	if (p->has_pool) {
		char packed[1024];
		if (p->pool_user[0]) {
			const char *sep = strstr(p->pool_url, "://");
			const size_t schemelen = sep ? (size_t) (sep - p->pool_url) + 3 : 0;
			snprintf(packed, sizeof(packed), "%.*s%s:%s@%s",
				(int) schemelen, p->pool_url, p->pool_user,
				p->pool_pass[0] ? p->pool_pass : "x", p->pool_url + schemelen);
		} else {
			snprintf(packed, sizeof(packed), "%s", p->pool_url);
		}
		if (!pool_switch_via_stratum(packed, -1, (int) opt_algo, err, errlen)) {
			if (algo_changed) {
				opt_algo = (enum sha_algos) old_algo;
				yespower_set_variant(old_variant);
				api_ctl_params_restore(opt_algo);
			} else if (variant_changed) {
				/* opt_algo never moved, so restoring it is a no-op and
				* api_ctl_params_restore would undo this request's own
				* params. Only the coin needs walking back. */
				yespower_set_variant(old_variant);
			}
			return false;
		}
	} else if (p->has_pool_index) {
		if (!pool_switch_via_stratum(NULL, p->pool_index, -1, err, errlen)) {
			if (algo_changed) {
				opt_algo = (enum sha_algos) old_algo;
				yespower_set_variant(old_variant);
				api_ctl_params_restore(opt_algo);
			} else if (variant_changed) {
				/* opt_algo never moved, so restoring it is a no-op and
				* api_ctl_params_restore would undo this request's own
				* params. Only the coin needs walking back. */
				yespower_set_variant(old_variant);
			}
			return false;
		}
	}

	/* Wait for the pool to come up: this window is the only one where the
	 * reconnect escalation is suppressed, so it is where the manager's choice
	 * is guaranteed to hold. A pool that never connects is reported, not
	 * rolled back — rolling one back is itself a switch. */
	if ((p->has_pool || p->has_pool_index) && have_stratum) {
		extern struct stratum_ctx stratum;
		const int slice = 100;
		int waited = 0;
		while (!stratum.curl && !abort_flag && waited < ctl.park_timeout_ms) {
			usleep(slice * 1000);
			waited += slice;
		}
		if (!stratum.curl)
			snprintf(ctl.last_error, sizeof(ctl.last_error),
				"pool selected but not connected after %dms", waited);
	}

	if (algo_changed || variant_changed) {
		/* pool_switch() cannot bump this here: opt_algo was assigned above.  A
		 * variant-only change must bump it too, since yespower sizes its buffers
		 * and uploads `pers` under its init guard.  Last, after every rollback.
		 * The stats reset is skipped there for the same reason. */
		pool_reset_algo_stats();
		algo_switch_gen++;
		if (algo_changed)
		applog(LOG_NOTICE, "control: algo %s -> %s",
			algo_names[old_algo] ? algo_names[old_algo] : "?",
			algo_names[opt_algo] ? algo_names[opt_algo] : "?");
		else
			applog(LOG_NOTICE, "control: yespower variant %s -> %s",
				old_variant, yespower_variant_name());
	}
	return true;
}

/* Park, apply, restore the run state the request asked for. */
static ctl_result_t profile_run(const ctl_profile_req *p)
{
	char err[160] = { 0 };

	if (pthread_mutex_trylock(&ctl.lock) != 0)
		return CTL_BUSY;

	const ctl_state_t previous = ctl.state;
	const bool was_parked = ctl.pause_requested;

	ctl.state = CTL_SWITCHING;
	ctl.last_error[0] = '\0';

	if (!was_parked) {
		ctl.pause_requested = true;
		restart_threads();
		if (!wait_for_park(opt_n_threads, ctl.park_timeout_ms)) {
			ctl.pause_requested = false;
			snprintf(ctl.last_error, sizeof(ctl.last_error),
				"only %d of %d threads parked within %dms",
				ctl.parked, opt_n_threads, ctl.park_timeout_ms);
			ctl.state = previous;
			restart_threads();
			wait_for_drain(ctl.park_timeout_ms);
			pthread_mutex_unlock(&ctl.lock);
			return CTL_TIMEOUT;
		}
	}

	const bool ok = profile_apply(p, err, sizeof(err));
	switch_in_flight = false;
	if (!ok)
		snprintf(ctl.last_error, sizeof(ctl.last_error), "%s", err);

	/* Default when the request says nothing: keep the state we came in with. */
	ctl_state_t target = previous;
	if (ok && p->has_run)
		target = p->run;

	const bool want_parked = (target == CTL_PAUSED || target == CTL_STOPPED);
	ctl.pause_requested = want_parked;
	ctl.hold_connection = (target == CTL_STOPPED);
	restart_threads();
	if (!want_parked)
		wait_for_drain(ctl.park_timeout_ms);

	ctl.state = target;
	if (ok) {
		ctl.state_since = time(NULL);
		ctl.switch_count++;
		ctl.epoch++;
		api_ctl_note_switch();
	}
	pthread_mutex_unlock(&ctl.lock);

	if (ok)
		applog(LOG_INFO, "control: profile applied (epoch %u, state %s)",
			ctl.epoch, api_ctl_state_name(target));
	else
		applog(LOG_WARNING, "control: profile rejected and rolled back: %s", err);

	/* A rejected profile is a 409 like a busy one; the message tells them apart. */
	return ok ? CTL_OK : CTL_BUSY;
}
