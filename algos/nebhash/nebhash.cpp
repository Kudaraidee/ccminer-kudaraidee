/* Host glue for nebhash v2 (NEB mainnet, KEP1) mining.  The consensus code and
 * the CUDA scan are the Apache-2.0 reference from SilasEigen/neb, see
 * LICENSE.NEB.  This file owns the dataset, the per-job models and one scan
 * handle per (mining thread, dataset slot). */
#include "algos/nebhash/nebhash.h"

#include "miner.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
/* The copied reference files are compiled as C; their public headers do not
 * all carry extern "C" guards, so include them under C linkage here. */
extern "C" {
#include "algos/nebhash/blake3_min.h"
#include "algos/nebhash/shares.h"
#include "algos/nebhash/chacha8.h"
#include "algos/nebhash/nebhash2.h"
#include "algos/nebhash/nebhash2_cuda.h"
}

#include <cuda_runtime.h>

#include <vector>
#include <memory>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cmath>

#define NEB_MAX_SLOTS   16
/* One scan window must finish well inside a pool job's lifetime. The reference
 * kernel does ~1.7k attempts/s on arch 1 and ~0.5k/s on arch 2 (GTX 1050 Ti),
 * so a 1M-nonce window would take many minutes and every job would go stale
 * before the first hit could be submitted. */
#define NEB_SCAN_CAP    1024        /* nonces per scan_run call */
#define NEB_MAX_HITS    (MAX_NONCES + 1)

namespace {

struct Dataset {
	uint32_t n = 0;
	uint32_t nin = 0;
	int arch = 0;
	std::vector<uint8_t> values;   /* n * nin, values 0..7 */
	std::vector<uint8_t> labels;   /* n, values 0..1 */
};

struct Model {
	uint8_t hash[32];
	int slot = 0;
	int arch = 0;
	std::vector<int8_t> weights;
	std::shared_ptr<neb2_model> ref;   /* 130 KB; keep off the stack */
};

struct ScanHandle {
	neb2_scan *h = nullptr;
	int arch = 0;
	uint8_t model_hash[32];
};

std::vector<Dataset> g_datasets;
/* Mining threads read the current model while the stratum thread installs the
 * next one. Publish whole shared_ptrs so an in-flight verification keeps its
 * model alive instead of racing with a vector reallocation. */
std::mutex g_state_lock;
std::vector<std::shared_ptr<Model>> g_models;
uint8_t g_dataset_hash[32] = { 0 };
uint8_t g_target[32] = { 0 };         /* set by mining.set_target */
ScanHandle *g_scans[MAX_GPUS][NEB_MAX_SLOTS];
char g_dataset_path[512] = { 0 };

uint32_t read_le32(const uint8_t *p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

void write_le32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

void append_bytes(std::vector<uint8_t> &v, const void *p, size_t n)
{
	const uint8_t *b = (const uint8_t *)p;
	if (n)
		v.insert(v.end(), b, b + n);
}

/* H(tag, parts...) is BLAKE3 over the tag bytes followed by the parts, with no
 * separators (spec/node.md section 3). */
void hash_parts(const char *tag, size_t taglen,
                const uint8_t *a, size_t alen,
                const uint8_t *b, size_t blen,
                const uint8_t *c, size_t clen, uint8_t out[32])
{
	std::vector<uint8_t> buf;
	buf.reserve(taglen + alen + blen + clen);
	append_bytes(buf, tag, taglen);
	append_bytes(buf, a, alen);
	append_bytes(buf, b, blen);
	append_bytes(buf, c, clen);
	blake3_hash(buf.data(), buf.size(), out);
}

int dataset_get(void *ctx, uint32_t index, uint8_t *x)
{
	Dataset *ds = (Dataset *)ctx;
	if (index >= ds->n)
		return 0;
	memcpy(x, ds->values.data() + (size_t)index * ds->nin, ds->nin);
	return ds->labels[index];
}

std::shared_ptr<Model> find_model(const uint8_t hash[32])
{
	std::lock_guard<std::mutex> lock(g_state_lock);
	for (size_t i = 0; i < g_models.size(); i++)
		if (!memcmp(g_models[i]->hash, hash, 32))
			return g_models[i];
	return std::shared_ptr<Model>();
}

/* Compare two 256-bit little-endian integers: return true when a <= b. */
bool le_le(const uint8_t a[32], const uint8_t b[32])
{
	for (int i = 31; i >= 0; i--) {
		if (a[i] != b[i])
			return a[i] < b[i];
	}
	return true;
}

/* Re-check a GPU hit with the portable reference before submitting it. */
int verify_ticket(const Model &m, int slot, const uint8_t header[80], uint8_t ticket[32])
{
	if (slot < 0 || slot >= (int)g_datasets.size() || !m.ref)
		return -1;
	Dataset &ds = g_datasets[slot];
	const neb2_arch *a = neb2_arch_get(ds.arch);
	if (!a || a->id != m.arch)
		return -1;

	uint8_t key[32];
	blake3_hash(header, 80, key);

	neb2_dataset nd;
	nd.n = ds.n;
	nd.get = dataset_get;
	nd.ctx = &ds;

	std::vector<int32_t> grad((size_t)a->nweights);
	uint8_t digest[32];
	return neb2_share_eval(m.ref.get(), &nd, key, grad.data(), digest, ticket,
	                       NULL, NULL);
}

bool ensure_scan(int thr_id, int slot, const uint8_t model_hash[32])
{
	if (thr_id < 0 || thr_id >= MAX_GPUS || slot < 0 || slot >= NEB_MAX_SLOTS)
		return false;
	int dev = device_map[thr_id];
	if (dev < 0 || slot >= (int)g_datasets.size())
		return false;

	std::shared_ptr<Model> m = find_model(model_hash);
	if (!m)
		return false;
	Dataset &ds = g_datasets[slot];
	const neb2_arch *a = neb2_arch_get(ds.arch);
	if (!a || a->id != m->arch)
		return false;

	ScanHandle *&ref = g_scans[thr_id][slot];
	if (!ref)
		ref = new ScanHandle();

	if (!ref->h) {
		int rc = neb2_scan_open(dev, a->id, ds.values.data(), ds.labels.data(),
		                        ds.n, m->weights.data(), NEB_SCAN_CAP,
		                        NEB_MAX_HITS, &ref->h);
		if (rc != 0) {
			applog(LOG_ERR, "nebhash: neb2_scan_open failed on GPU %d (%d)", dev, rc);
			return false;
		}
		ref->arch = a->id;
		memset(ref->model_hash, 0, 32);
	} else if (ref->arch != a->id) {
		neb2_scan_close(ref->h);
		ref->h = NULL;
		return ensure_scan(thr_id, slot, model_hash);
	}

	if (memcmp(ref->model_hash, model_hash, 32)) {
		int rc = neb2_scan_set_model(ref->h, m->weights.data());
		if (rc != 0) {
			applog(LOG_ERR, "nebhash: model upload failed on GPU %d (%d)", dev, rc);
			return false;
		}
		memcpy(ref->model_hash, model_hash, 32);
	}
	return true;
}

void close_all_scans()
{
	for (int t = 0; t < MAX_GPUS; t++) {
		for (int s = 0; s < NEB_MAX_SLOTS; s++) {
			if (g_scans[t][s]) {
				neb2_scan_close(g_scans[t][s]->h);
				delete g_scans[t][s];
				g_scans[t][s] = NULL;
			}
		}
	}
}

} /* namespace */

void nebhash_set_dataset_option(const char *path)
{
	if (!path)
		return;
	snprintf(g_dataset_path, sizeof(g_dataset_path), "%s", path);
}

const char *nebhash_dataset_option(void)
{
	return g_dataset_path[0] ? g_dataset_path : NULL;
}

int nebhash_dataset_loaded(void)
{
	return !g_datasets.empty();
}

void nebhash_dataset_hash(uint8_t out[32])
{
	memcpy(out, g_dataset_hash, 32);
}

int nebhash_load_dataset_file(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) {
		applog(LOG_ERR, "nebhash: cannot open dataset %s", path);
		return -1;
	}
	if (fseek(f, 0, SEEK_END) != 0) {
		fclose(f);
		return -1;
	}
	long filelen = ftell(f);
	if (filelen <= 8 || fseek(f, 0, SEEK_SET) != 0) {
		fclose(f);
		return -1;
	}
	std::vector<uint8_t> bytes((size_t)filelen);
	if (fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
		fclose(f);
		applog(LOG_ERR, "nebhash: short read on dataset %s", path);
		return -1;
	}
	fclose(f);

	if (memcmp(bytes.data(), "NMDS", 4)) {
		applog(LOG_ERR, "nebhash: %s has no NMDS header", path);
		return -1;
	}
	uint32_t count = read_le32(bytes.data() + 4);
	if (!count || count > NEB_MAX_SLOTS) {
		applog(LOG_ERR, "nebhash: dataset count %u out of range", count);
		return -1;
	}

	size_t pos = 8;
	std::vector<Dataset> loaded;
	for (uint32_t i = 0; i < count; i++) {
		if (pos + 12 > bytes.size() || memcmp(bytes.data() + pos, "NMD2", 4)) {
			applog(LOG_ERR, "nebhash: dataset %u has no NMD2 header", i);
			return -1;
		}
		uint32_t arch_id = read_le32(bytes.data() + pos + 4);
		uint32_t n = read_le32(bytes.data() + pos + 8);
		const neb2_arch *a = neb2_arch_get((int)arch_id);
		if (!a || !n) {
			applog(LOG_ERR, "nebhash: dataset %u has unsupported arch %u", i, arch_id);
			return -1;
		}
		size_t need = (size_t)a->nin + 1;
		size_t total = 12 + (size_t)n * need;
		if (pos + total > bytes.size()) {
			applog(LOG_ERR, "nebhash: dataset %u is truncated", i);
			return -1;
		}

		Dataset ds;
		ds.arch = (int)arch_id;
		ds.nin = (uint32_t)a->nin;
		ds.n = n;
		ds.values.resize((size_t)n * a->nin);
		ds.labels.resize(n);
		for (uint32_t j = 0; j < n; j++) {
			const uint8_t *rec = bytes.data() + pos + 12 + (size_t)j * need;
			memcpy(ds.values.data() + (size_t)j * a->nin, rec, a->nin);
			ds.labels[j] = rec[a->nin];
			if (ds.labels[j] >= NEB2_NOUT) {
				applog(LOG_ERR, "nebhash: dataset %u sample %u has bad label", i, j);
				return -1;
			}
		}
		pos += total;
		loaded.push_back(std::move(ds));
	}
	if (pos != bytes.size()) {
		applog(LOG_ERR, "nebhash: trailing bytes in dataset file");
		return -1;
	}

	close_all_scans();
	g_models.clear();
	g_datasets = std::move(loaded);
	blake3_hash(bytes.data(), bytes.size(), g_dataset_hash);
	return 0;
}

int nebhash_have_model(const uint8_t hash[32])
{
	return find_model(hash) != NULL;
}

int nebhash_set_model(int slot, const uint8_t hash[32],
                      const uint8_t *packed, size_t packed_len)
{
	if (slot < 0 || slot >= (int)g_datasets.size())
		return -1;
	const Dataset &ds = g_datasets[slot];
	const neb2_arch *a = neb2_arch_get(ds.arch);
	if (!a || !packed || packed_len != (size_t)a->nweights / 2)
		return -1;

	Model m;
	memcpy(m.hash, hash, 32);
	m.slot = slot;
	m.arch = a->id;
	m.weights.resize((size_t)a->nweights);
	for (int i = 0; i < a->nweights; i++) {
		uint8_t nibble = (i & 1) ? (packed[i / 2] >> 4) : (packed[i / 2] & 15);
		m.weights[i] = (int8_t)((int8_t)(nibble << 4) >> 4);
	}

	m.ref.reset(new neb2_model);
	memset(m.ref.get(), 0, sizeof(neb2_model));
	m.ref->arch = a->id;
	memcpy(m.ref->w, m.weights.data(), (size_t)a->nweights);

	uint8_t wh[32], rh[32];
	neb2_model_hashes(m.ref.get(), wh, rh);
	if (memcmp(wh, hash, 32)) {
		applog(LOG_ERR, "nebhash: set_model weights_hash mismatch");
		return -1;
	}

	std::lock_guard<std::mutex> lock(g_state_lock);
	std::shared_ptr<Model> published = std::make_shared<Model>(std::move(m));
	for (size_t i = 0; i < g_models.size(); i++) {
		if (!memcmp(g_models[i]->hash, hash, 32)) {
			g_models[i] = published;
			return 0;
		}
	}
	/* Models arrive every block; only a few are needed for hit verification. */
	if (g_models.size() >= 4)
		g_models.erase(g_models.begin());
	g_models.push_back(published);
	return 0;
}

void nebhash_set_target(const uint8_t le[32])
{
	std::lock_guard<std::mutex> lock(g_state_lock);
	memcpy(g_target, le, 32);
}

void nebhash_get_target(uint8_t le[32])
{
	std::lock_guard<std::mutex> lock(g_state_lock);
	memcpy(le, g_target, 32);
}

double nebhash_target_to_diff(const uint8_t le[32])
{
	long double t = 0.0L;
	for (int i = 31; i >= 0; i--)
		t = t * 256.0L + le[i];
	if (!(t > 0.0L))
		return 0.0;
	return (double)(ldexpl(1.0L, 256) / t);
}

int nebhash_build_header(uint32_t version, const uint8_t parent[32],
                         const uint8_t *cb_prefix, size_t cb_prefix_len,
                         const uint8_t *xn1, size_t xn1_len,
                         const uint8_t *xn2, size_t xn2_len,
                         const uint8_t *cb_suffix, size_t cb_suffix_len,
                         const uint8_t *branch, size_t branch_count,
                         uint32_t tx_count, const uint8_t shares_root[32],
                         uint32_t ntime, uint32_t bits, uint8_t header[80])
{
	std::vector<uint8_t> coinbase;
	coinbase.reserve(cb_prefix_len + xn1_len + xn2_len + cb_suffix_len);
	append_bytes(coinbase, cb_prefix, cb_prefix_len);
	append_bytes(coinbase, xn1, xn1_len);
	append_bytes(coinbase, xn2, xn2_len);
	append_bytes(coinbase, cb_suffix, cb_suffix_len);
	if (coinbase.empty())
		return -1;

	uint8_t coinhash[32], root[32], le[4];
	blake3_hash(coinbase.data(), coinbase.size(), coinhash);
	hash_parts("leaf", 4, coinhash, 32, NULL, 0, NULL, 0, root);
	for (size_t i = 0; i < branch_count; i++)
		hash_parts("node", 4, root, 32, branch + i * 32, 32, NULL, 0, root);
	write_le32(le, tx_count);
	hash_parts("body", 4, le, 4, root, 32, shares_root, 32, root);

	write_le32(header, version);
	memcpy(header + 4, parent, 32);
	memcpy(header + 36, root, 32);
	write_le32(header + 68, ntime);
	write_le32(header + 72, bits);
	write_le32(header + 76, 0);   /* nonce, rolled by the scan */
	return 0;
}

int nebhash_scanhash(int thr_id, struct work *work, uint32_t max_nonce,
                     unsigned long *hashes_done)
{
	*hashes_done = 0;
	if (!work->is_nebhash || !nebhash_dataset_loaded())
		return -1;

	int slot = (int)work->neb_slot;
	if (!ensure_scan(thr_id, slot, work->neb_model_hash))
		return -1;

	uint32_t start = work->data[19];
	uint32_t count = max_nonce;
	if (count == 0)
		return 0;
	if (count > NEB_SCAN_CAP)
		count = NEB_SCAN_CAP;
	if (start > UINT32_MAX - count)
		count = UINT32_MAX - start;
	if (count == 0)
		return 0;

	uint8_t header[80];
	memcpy(header, work->neb_header, 80);
	uint32_t hits[NEB_MAX_HITS + 1];
	uint32_t nhits = 0;
	int rc = neb2_scan_run(g_scans[thr_id][slot]->h, header, start, count,
	                       (const uint8_t *)work->target, hits, &nhits);
	if (rc != 0) {
		applog(LOG_ERR, "nebhash: scan failed on GPU %d (%d)", device_map[thr_id], rc);
		return -1;
	}

	*hashes_done = count;
	work->data[19] = start + count;   /* deliberate uint32 wrap */

	std::shared_ptr<Model> m = find_model(work->neb_model_hash);
	int found = 0;
	for (uint32_t i = 0; i < nhits && found < MAX_NONCES; i++) {
		uint32_t nonce = hits[i];
		write_le32(header + 76, nonce);
		uint8_t ticket[32];
		if (!m || verify_ticket(*m, slot, header, ticket) != 0)
			continue;
		if (!le_le(ticket, (const uint8_t *)work->target))
			continue;
		work->nonces[found] = nonce;
		bn_set_target_ratio(work, (uint32_t *)ticket, found);
		found++;
	}
	return found;
}

void nebhash_free(int thr_id)
{
	if (thr_id < 0 || thr_id >= MAX_GPUS)
		return;
	for (int s = 0; s < NEB_MAX_SLOTS; s++) {
		if (g_scans[thr_id][s]) {
			neb2_scan_close(g_scans[thr_id][s]->h);
			delete g_scans[thr_id][s];
			g_scans[thr_id][s] = NULL;
		}
	}
}
