// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Neutral snapshots of everything the API reports, and the renderers over them:
 *
 *     api_collect_*()        read the live globals into a snapshot
 *     api_format_*_binary()  legacy K=V;K=V;| form
 *     api_build_*_json()     JSON form
 *
 * so both renderers report the same numbers. The binary output is a
 * compatibility surface (api/index.php, api/summary.pl, HiveOS) and must stay
 * byte-identical; api/tests/golden.py is the gate.
 *
 * Snapshots hold copies, never pointers into miner state: the API thread reads
 * live globals without locks, so each field is read once and then formatted.
 *
 * Keep out of miner.h — that header is one large extern "C" block, and these
 * are C++-linkage declarations.
 */

#ifndef API_MODEL_H
#define API_MODEL_H

#include <stdint.h>
#include <stddef.h>

#include <jansson.h>

#include "api_metrics.h"

/* Version of the binary API contract reported as API= in the summary record. */
#ifndef APIVERSION
#define APIVERSION "1.9"
#endif

/* ------------------------------------------------------------------ summary */

struct api_summary_snapshot {
	const char *name;        /* string literals, stable for process lifetime */
	const char *version;
	const char *api_version;
	char algo[64];
	int gpus;
	double khs;
	uint32_t solved, accepted, rejected;
	double accps;
	double diff;
	double netkhs;
	uint32_t pools;
	uint32_t wait_time;
	double uptime;
	uint32_t ts;
};

/* ------------------------------------------------- per-thread (GPU) status */

struct api_thread_snapshot {
	bool valid;
	int id;                   /* miner thread index */
	int gpu_id;
	int16_t bus;
	char card[64];
	float temp;
	uint32_t power;
	uint16_t fan, rpm;
	uint32_t freq, memfreq;   /* base clocks, MHz */
	uint32_t gpuf, memf;      /* current clocks, MHz */
	double khs, khw;
	uint32_t plim;
	unsigned acc;
	uint32_t rej;
	uint32_t hwf;
	double intensity;
	uint32_t throughput;
};

/* --------------------------------------------------------------------- pool */

struct api_pool_snapshot {
	char name[64];            /* pool name, or short_url when unnamed */
	const char *algo;
	char url[512];
	char user[192];           /* empty unless stratum */
	uint32_t solved, accepted, rejected, stales;
	uint32_t height;
	char job_id[128];
	double diff;
	double best_share;
	int xnonce2_size;
	char xnonce2[96];
	uint32_t ping_ms;
	uint32_t disconnects;
	uint32_t wait_time;
	uint32_t work_time;
	uint32_t last_share_age;
	bool stratum;             /* false for getwork/GBT */
	bool connected;
};

/* ---------------------------------------------------------- GPU hardware */

struct api_gpuhw_snapshot {
	bool valid;
	int gpu_id;
	int16_t bus;
	char card[64];
	uint16_t arch;
	uint32_t mem;
	float temp;
	uint16_t fan, rpm;
	uint32_t freq, memfreq;
	uint32_t gpuf, memf;
	char pstate[8];
	uint32_t power;
	uint32_t plim;
	uint16_t vid, pid;
	int nvml_id, nvapi_id;
	char sn[64];
	char bios[64];
};

/* ------------------------------------------------------- system hardware */

struct api_system_snapshot {
	char os[64];
	char driver[32];
	int cpus;
	int cpu_temp_c;
	uint32_t cpu_clock_mhz;
	int cpu_fan_pct;          /* -1 = no sensor on this platform */
};

/* -------------------------------------------------------------- collectors */

void api_collect_summary(struct api_summary_snapshot *s);
void api_collect_thread(int thr_id, struct api_thread_snapshot *s);
void api_collect_pool(int pooln, struct api_pool_snapshot *s);
void api_collect_gpuhw(int gpu_id, struct api_gpuhw_snapshot *s);
void api_collect_system(struct api_system_snapshot *s);

/* ------------------------------------------------------- binary renderers */

int api_format_summary_binary(const struct api_summary_snapshot *s, char *out, size_t outlen);
int api_format_thread_binary(const struct api_thread_snapshot *s, char *out, size_t outlen);
int api_format_pool_binary(const struct api_pool_snapshot *s, char *out, size_t outlen);
int api_format_gpuhw_binary(const struct api_gpuhw_snapshot *s, char *out, size_t outlen);
int api_format_system_binary(const struct api_system_snapshot *s, char *out, size_t outlen);

/* --------------------------------------------------------- JSON renderers */
/* Same snapshots, second renderer. Unit change against the binary API,
 * per docs/api-rest.md: JSON reports H/s where the binary reported kH/s. */

json_t *api_build_miner_json(void);
json_t *api_build_summary_json(const struct api_summary_snapshot *s);
json_t *api_build_thread_json(const struct api_thread_snapshot *s);
json_t *api_build_pool_json(int index, bool active, const struct api_pool_snapshot *s);
json_t *api_build_device_json(const struct api_gpuhw_snapshot *hw,
                              const struct api_thread_snapshot *th);
json_t *api_build_system_json(const struct api_system_snapshot *s);

/* ------------------------------------------------------------- metrics */

/* Fills the Prometheus input struct from cached state only — no vendor
 * telemetry calls, because a scraper polls this every 15-60 s. */
void api_collect_metrics(api_metrics_input *in);

#endif /* API_MODEL_H */
