/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Runtime control: pause, resume and stop mining without restarting the
 * process. Contract: docs/api-rest.md section 7.
 *
 * A park barrier, not thread cancellation: every mining thread is asked to
 * idle and nothing changes until all of them have, so the CUDA context and the
 * stratum connection survive a switch.
 *
 * Threading: mining threads only read pause_requested and only write parked.
 * The mutex serialises control callers against each other, never the mining
 * loop.
 */

#ifndef API_CONTROL_H
#define API_CONTROL_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	CTL_RUNNING = 0,
	CTL_PAUSED,
	CTL_STOPPED,
	CTL_SWITCHING
} ctl_state_t;

/* Result of a control request, mapped to HTTP status by the caller. */
typedef enum {
	CTL_OK = 0,          /* applied                                  -> 200 */
	CTL_ACCEPTED,        /* still running, poll the state            -> 202 */
	CTL_BUSY,            /* another mutation in flight               -> 409 */
	CTL_TIMEOUT,         /* threads did not park; rolled back        -> 409 */
	CTL_THROTTLED,       /* inside --api-control-min-interval        -> 429 */
	CTL_DISABLED         /* --api-control not given                  -> 403 */
} ctl_result_t;

/* Called once at startup. Starts the worker thread only when enabled. */
void api_ctl_init(bool enabled, int min_interval_s, int park_timeout_ms);
void api_ctl_shutdown(void);

/* --- mining-loop hooks ------------------------------------------------- */

/* The whole hook. One leading condition in wanna_mine(). */
bool api_ctl_wants_pause(void);

/* Called from the miner loop's idle branch once a thread has released its
 * resources. Blocks in short slices while the pause stands, so a resume is
 * seen quickly; returns when mining may continue. */
void api_ctl_park(int thr_id);

/* --- control side ------------------------------------------------------ */

/* Run-state verbs. Never throttled: an emergency stop has to work, and a
 * manager re-asserting a state must not be told to come back later. */
ctl_result_t api_ctl_set_state(ctl_state_t target, int wait_ms);

/* --- algo parameters ---------------------------------------------------- */

/*
 * A whitelist, not a passthrough: only these names, only for the algos that
 * consume them. A parameter the target algo would ignore is rejected, never
 * dropped silently.
 */
typedef enum {
	CTLP_INT = 0,
	CTLP_STRING,
	CTLP_INT_LIST,      /* comma-separated, one per device */
	CTLP_FLOAT_LIST     /* comma-separated, one per device */
} ctl_param_type_t;

typedef struct {
	const char *name;
	ctl_param_type_t type;
	bool slow;          /* cannot complete inside a request; answers 202 */
} ctl_param_def_t;

/* The whole accepted set, and the subset a given algo consumes. */
const ctl_param_def_t *api_ctl_param_defs(size_t *count);
const ctl_param_def_t *api_ctl_param_find(const char *name);
bool api_ctl_algo_accepts(int algo, const char *name);
const char **api_ctl_algo_params(int algo, size_t *count);

/* Apply one parameter. Call with every mining thread parked. NULL value resets
 * to the startup default. False + err on a value the miner will not accept. */
bool api_ctl_param_apply(int algo, const char *name, const char *value,
                         char *err, size_t errlen);

/* Current value of a parameter, or NULL when unset. Writes into buf. */
const char *api_ctl_param_get(int algo, const char *name, char *buf, size_t buflen);

/* Sticky store: last params used per algo, re-applied when a profile switch
 * omits `params`. */
void api_ctl_params_remember(int algo, const char *name, const char *value);
void api_ctl_params_restore(int algo);

/* --- profile switches --------------------------------------------------- */

/* Throttled, unlike the run-state verbs (docs/api-rest.md section 7.1). */
bool api_ctl_ready_for_switch(void);
int  api_ctl_retry_after_s(void);
void api_ctl_note_switch(void);

/*
 * One atomic change of algo + params + pool + run state.
 *
 * Passed by value: when wait_ms expires the request outlives the caller (that
 * is what the 202 path is), so a caller-owned struct would dangle.
 */
#define CTL_MAX_PARAMS 12
#define CTL_PARAM_NAME_MAX 24
#define CTL_PARAM_VALUE_MAX 256

typedef struct {
	bool has_algo;
	int  algo;
	char algo_name[32];   /* as sent: yespower coin variants share one algo and
	                       * differ only in (N, r, pers), which the int loses */

	bool has_pool;
	char pool_url[512];
	char pool_user[256];
	char pool_pass[128];
	bool has_pool_index;
	int  pool_index;

	bool has_run;
	ctl_state_t run;

	int  nparams;
	char pname[CTL_MAX_PARAMS][CTL_PARAM_NAME_MAX];
	char pvalue[CTL_MAX_PARAMS][CTL_PARAM_VALUE_MAX];
	bool pnull[CTL_MAX_PARAMS];    /* explicit null: reset to startup default */
} ctl_profile_req;

/* Callers validate first: anything rejectable without touching the miner is a
 * 400 and must not reach the barrier. */
bool api_ctl_param_check(int algo, const char *name, const char *value,
                         char *err, size_t errlen);

ctl_result_t api_ctl_profile(const ctl_profile_req *req, int wait_ms,
                             char *err, size_t errlen);

/* --- pool switch handshake ---------------------------------------------- */

/*
 * A pool switch must run on the stratum thread: pool_switch() replaces the
 * global stratum struct while stratum_recv_line() reads it under no lock.
 * The control side posts the url here and waits for the ack.
 */
/* index >= 0 means switch to an existing pool slot; otherwise use url. */
bool api_ctl_pool_request_take(char *url, size_t urllen, int *index, int *algo);
bool api_ctl_pool_request_pending(void);

/* True while a control-requested pool switch is being applied; the reconnect
 * escalation (failover, then proper_exit) must not fire in that window. */
bool api_ctl_switch_in_flight(void);
void api_ctl_pool_request_done(bool ok);

ctl_state_t api_ctl_get_state(void);
uint32_t    api_ctl_epoch(void);
uint32_t    api_ctl_switch_count(void);
time_t      api_ctl_state_since(void);
time_t      api_ctl_last_switch(void);
int         api_ctl_parked(void);
int         api_ctl_threads_total(void);
bool        api_ctl_enabled(void);
bool        api_ctl_hold_connection(void);
int         api_ctl_min_interval(void);
const char *api_ctl_last_error(void);
const char *api_ctl_state_name(ctl_state_t s);

#ifdef __cplusplus
}
#endif

#endif /* API_CONTROL_H */
