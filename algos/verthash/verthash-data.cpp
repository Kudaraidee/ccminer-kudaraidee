// SPDX-License-Identifier: GPL-3.0-or-later
//
// Verthash mining-data-file management + generator (host side).
// Provenance: cpuminer-opt algo/verthash/Verthash.c (CryptoGraphics, GPLv2);
// graph construction ported verbatim, sha3 routed through algos/verthash's
// tiny_sha3, logging routed to stderr. Self-contained (no miner.h).

#include "verthash-data.h"
#include "sph/sha3.h"   // vendored FIPS-202 tiny_sha3 (0x06 pad)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "sph/sph_sha2.h"
}

#define VH_HASH_OUT_SIZE  32
#define VH_BYTE_ALIGNMENT 16

// Known-good SHA-256 of the canonical Vertcoin verthash.dat, in natural
// (sha256-emitted) byte order. NOTE: the cpuminer-opt source comment prints the
// byte-reversed form (0x48aa21d7...); this array is the order sha256 actually
// produces, so it can be memcmp'd directly.
static const uint8_t verthash_dat_sha256[32] = {
    0xa5, 0x55, 0x31, 0xe8, 0x43, 0xcd, 0x56, 0xb0,
    0x10, 0x11, 0x4a, 0xaf, 0x63, 0x25, 0xb0, 0xd5,
    0x29, 0xec, 0xf8, 0x8f, 0x8a, 0xd4, 0x76, 0x39,
    0xb6, 0xed, 0xed, 0xaf, 0xd7, 0x21, 0xaa, 0x48
};

//-----------------------------------------------------------------------------
int verthash_data_load(const char *path, uint8_t **out_buf, size_t *out_size)
{
    *out_buf = NULL;
    *out_size = 0;

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "verthash: cannot open data file '%s'\n", path); return -1; }

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long long sz =
#if defined(_WIN32)
        _ftelli64(f);
#else
        ftello(f);
#endif
    if (sz <= (long long) VH_HASH_OUT_SIZE) {
        fprintf(stderr, "verthash: data file '%s' too small (%lld bytes)\n", path, sz);
        fclose(f);
        return -1;
    }
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = (uint8_t *) malloc((size_t) sz);
    if (!buf) { fprintf(stderr, "verthash: OOM allocating %lld bytes\n", sz); fclose(f); return -2; }

    // Read in chunks: a single fread of >2 GiB is unreliable on some CRTs.
    size_t total = 0, want = (size_t) sz;
    while (total < want) {
        size_t chunk = want - total;
        if (chunk > (64u << 20)) chunk = 64u << 20;
        size_t got = fread(buf + total, 1, chunk, f);
        if (got == 0) { fprintf(stderr, "verthash: short read on '%s'\n", path); free(buf); fclose(f); return -1; }
        total += got;
    }
    fclose(f);

    *out_buf = buf;
    *out_size = want;
    return 0;
}

//-----------------------------------------------------------------------------
uint32_t verthash_data_mdiv(size_t size)
{
    return (uint32_t) (((size - VH_HASH_OUT_SIZE) / VH_BYTE_ALIGNMENT) + 1);
}

//-----------------------------------------------------------------------------
int verthash_data_verify(const uint8_t *buf, size_t size)
{
    uint8_t digest[32];
    sph_sha256_context ctx;
    sph_sha256_init(&ctx);
    // hash in bounded chunks (size_t len ok, but keep it explicit)
    size_t off = 0;
    while (off < size) {
        size_t chunk = size - off;
        if (chunk > (256u << 20)) chunk = 256u << 20;
        sph_sha256(&ctx, buf + off, chunk);
        off += chunk;
    }
    sph_sha256_close(&ctx, digest);
    return memcmp(digest, verthash_dat_sha256, 32) == 0;
}

//=============================================================================
// verthash.dat generator (ported verbatim from cpuminer-opt Verthash.c).
//=============================================================================

#define NODE_SIZE 32

struct Graph {
    FILE *db;
    int64_t log2;
    int64_t pow2;
    uint8_t *pk;
    int64_t index;
};

static int64_t Log2(int64_t x) { int64_t r = 0; for (; x > 1; x >>= 1) r++; return r; }
static int64_t bfsToPost(struct Graph *g, const int64_t node) { return node & ~g->pow2; }
static int64_t numXi(int64_t index) { return (1 << ((uint64_t) index)) * (index + 1) * index; }

static void WriteId(struct Graph *g, uint8_t *Node, const int64_t id)
{
    fseek(g->db, (long) (id * NODE_SIZE), SEEK_SET);
    fwrite(Node, 1, NODE_SIZE, g->db);
}
static void WriteNode(struct Graph *g, uint8_t *Node, const int64_t id)
{
    WriteId(g, Node, bfsToPost(g, id));
}
static void NewNode(struct Graph *g, const int64_t id, uint8_t *hash) { WriteNode(g, hash, id); }

static uint8_t *GetId(struct Graph *g, const int64_t id)
{
    fseek(g->db, (long) (id * NODE_SIZE), SEEK_SET);
    uint8_t *node = (uint8_t *) malloc(NODE_SIZE);
    const size_t bytes_read = fread(node, 1, NODE_SIZE, g->db);
    if (bytes_read != NODE_SIZE) { free(node); return NULL; }
    return node;
}
static uint8_t *GetNode(struct Graph *g, const int64_t id) { return GetId(g, bfsToPost(g, id)); }

static uint32_t WriteVarInt(uint8_t *buffer, int64_t val)
{
    memset(buffer, 0, NODE_SIZE);
    uint64_t uval = ((uint64_t) (val)) << 1;
    if (val < 0) uval = ~uval;
    uint32_t i = 0;
    while (uval >= 0x80) { buffer[i] = (uint8_t) uval | 0x80; uval >>= 7; i++; }
    buffer[i] = (uint8_t) uval;
    return i;
}

static void ButterflyGraph(struct Graph *g, int64_t index, int64_t *count)
{
    if (index == 0) index = 1;

    int64_t numLevel = 2 * index;
    int64_t perLevel = (int64_t) (1 << (uint64_t) index);
    int64_t begin = *count - perLevel;
    int64_t level, i;

    for (level = 1; level < numLevel; level++) {
        for (i = 0; i < perLevel; i++) {
            int64_t prev;
            int64_t shift = index - level;
            if (level > numLevel / 2) shift = level - numLevel / 2;
            if (((i >> (uint64_t) shift) & 1) == 0) prev = i + (1 << (uint64_t) shift);
            else prev = i - (1 << (uint64_t) shift);

            uint8_t *parent0 = GetNode(g, begin + (level - 1) * perLevel + prev);
            uint8_t *parent1 = GetNode(g, *count - perLevel);
            uint8_t *buf = (uint8_t *) malloc(NODE_SIZE);
            WriteVarInt(buf, *count);
            uint8_t *hashInput = (uint8_t *) malloc(NODE_SIZE * 4);
            memcpy(hashInput, g->pk, NODE_SIZE);
            memcpy(hashInput + NODE_SIZE, buf, NODE_SIZE);
            memcpy(hashInput + (NODE_SIZE * 2), parent0, NODE_SIZE);
            memcpy(hashInput + (NODE_SIZE * 3), parent1, NODE_SIZE);

            uint8_t *hashOutput = (uint8_t *) malloc(NODE_SIZE);
            sha3(hashInput, NODE_SIZE * 4, hashOutput, NODE_SIZE);

            NewNode(g, *count, hashOutput);
            (*count)++;

            free(hashOutput); free(hashInput); free(parent0); free(parent1); free(buf);
        }
    }
}

static void XiGraphIter(struct Graph *g, int64_t index)
{
    int64_t count = g->pow2;

    int8_t stackSize = 5;
    int64_t *stack = (int64_t *) malloc(sizeof(int64_t) * stackSize);
    for (int i = 0; i < 5; i++) stack[i] = index;

    int8_t graphStackSize = 5;
    int32_t *graphStack = (int32_t *) malloc(sizeof(int32_t) * graphStackSize);
    for (int i = 0; i < 5; i++) graphStack[i] = graphStackSize - i - 1;

    int64_t i = 0;
    int64_t graph = 0;
    int64_t pow2index = 1 << ((uint64_t) index);

    for (i = 0; i < pow2index; i++) {
        uint8_t *buf = (uint8_t *) malloc(NODE_SIZE);
        WriteVarInt(buf, count);
        uint8_t *hashInput = (uint8_t *) malloc(NODE_SIZE * 2);
        memcpy(hashInput, g->pk, NODE_SIZE);
        memcpy(hashInput + NODE_SIZE, buf, NODE_SIZE);
        uint8_t *hashOutput = (uint8_t *) malloc(NODE_SIZE);

        sha3(hashInput, NODE_SIZE * 2, hashOutput, NODE_SIZE);
        NewNode(g, count, hashOutput);
        count++;

        free(hashOutput); free(hashInput); free(buf);
    }

    if (index == 1) { ButterflyGraph(g, index, &count); return; }

    while (stackSize != 0 && graphStackSize != 0) {
        index = stack[stackSize - 1];
        graph = graphStack[graphStackSize - 1];

        stackSize--;
        if (stackSize > 0) {
            int64_t *tempStack = (int64_t *) malloc(sizeof(int64_t) * (stackSize));
            memcpy(tempStack, stack, sizeof(int64_t) * (stackSize));
            free(stack); stack = tempStack;
        }

        graphStackSize--;
        if (graphStackSize > 0) {
            int32_t *tempGraphStack = (int32_t *) malloc(sizeof(int32_t) * (graphStackSize));
            memcpy(tempGraphStack, graphStack, sizeof(int32_t) * (graphStackSize));
            free(graphStack); graphStack = tempGraphStack;
        }

        int8_t indicesSize = 5;
        int64_t *indices = (int64_t *) malloc(sizeof(int64_t) * indicesSize);
        for (int k = 0; k < indicesSize; k++) indices[k] = index - 1;

        int8_t graphsSize = 5;
        int32_t *graphs = (int32_t *) malloc(sizeof(int32_t) * graphsSize);
        for (int k = 0; k < graphsSize; k++) graphs[k] = graphsSize - k - 1;

        int64_t pow2indexInner = 1 << ((uint64_t) index);
        int64_t pow2indexInner_1 = 1 << ((uint64_t) index - 1);

        if (graph == 0) {
            uint64_t sources = count - pow2indexInner;
            for (i = 0; i < pow2indexInner_1; i++) {
                uint8_t *parent0 = GetNode(g, sources + i);
                uint8_t *parent1 = GetNode(g, sources + i + pow2indexInner_1);
                uint8_t *buf = (uint8_t *) malloc(NODE_SIZE);
                WriteVarInt(buf, count);
                uint8_t *hashInput = (uint8_t *) malloc(NODE_SIZE * 4);
                memcpy(hashInput, g->pk, NODE_SIZE);
                memcpy(hashInput + NODE_SIZE, buf, NODE_SIZE);
                memcpy(hashInput + (NODE_SIZE * 2), parent0, NODE_SIZE);
                memcpy(hashInput + (NODE_SIZE * 3), parent1, NODE_SIZE);
                uint8_t *hashOutput = (uint8_t *) malloc(NODE_SIZE);
                sha3(hashInput, NODE_SIZE * 4, hashOutput, NODE_SIZE);
                NewNode(g, count, hashOutput);
                count++;
                free(hashOutput); free(hashInput); free(parent0); free(parent1); free(buf);
            }
        } else if (graph == 1) {
            uint64_t firstXi = count;
            for (i = 0; i < pow2indexInner_1; i++) {
                uint64_t nodeId = firstXi + i;
                uint8_t *parent = GetNode(g, firstXi - pow2indexInner_1 + i);
                uint8_t *buf = (uint8_t *) malloc(NODE_SIZE);
                WriteVarInt(buf, nodeId);
                uint8_t *hashInput = (uint8_t *) malloc(NODE_SIZE * 3);
                memcpy(hashInput, g->pk, NODE_SIZE);
                memcpy(hashInput + NODE_SIZE, buf, NODE_SIZE);
                memcpy(hashInput + (NODE_SIZE * 2), parent, NODE_SIZE);
                uint8_t *hashOutput = (uint8_t *) malloc(NODE_SIZE);
                sha3(hashInput, NODE_SIZE * 3, hashOutput, NODE_SIZE);
                NewNode(g, count, hashOutput);
                count++;
                free(hashOutput); free(hashInput); free(parent); free(buf);
            }
        } else if (graph == 2) {
            uint64_t secondXi = count;
            for (i = 0; i < pow2indexInner_1; i++) {
                uint64_t nodeId = secondXi + i;
                uint8_t *parent = GetNode(g, secondXi - pow2indexInner_1 + i);
                uint8_t *buf = (uint8_t *) malloc(NODE_SIZE);
                WriteVarInt(buf, nodeId);
                uint8_t *hashInput = (uint8_t *) malloc(NODE_SIZE * 3);
                memcpy(hashInput, g->pk, NODE_SIZE);
                memcpy(hashInput + NODE_SIZE, buf, NODE_SIZE);
                memcpy(hashInput + (NODE_SIZE * 2), parent, NODE_SIZE);
                uint8_t *hashOutput = (uint8_t *) malloc(NODE_SIZE);
                sha3(hashInput, NODE_SIZE * 3, hashOutput, NODE_SIZE);
                NewNode(g, count, hashOutput);
                count++;
                free(hashOutput); free(hashInput); free(parent); free(buf);
            }
        } else if (graph == 3) {
            uint64_t secondButter = count;
            for (i = 0; i < pow2indexInner_1; i++) {
                uint64_t nodeId = secondButter + i;
                uint8_t *parent = GetNode(g, secondButter - pow2indexInner_1 + i);
                uint8_t *buf = (uint8_t *) malloc(NODE_SIZE);
                WriteVarInt(buf, nodeId);
                uint8_t *hashInput = (uint8_t *) malloc(NODE_SIZE * 3);
                memcpy(hashInput, g->pk, NODE_SIZE);
                memcpy(hashInput + NODE_SIZE, buf, NODE_SIZE);
                memcpy(hashInput + (NODE_SIZE * 2), parent, NODE_SIZE);
                uint8_t *hashOutput = (uint8_t *) malloc(NODE_SIZE);
                sha3(hashInput, NODE_SIZE * 3, hashOutput, NODE_SIZE);
                NewNode(g, count, hashOutput);
                count++;
                free(hashOutput); free(hashInput); free(parent); free(buf);
            }
        } else {
            uint64_t sinks = count;
            uint64_t sources = sinks + pow2indexInner - numXi(index);
            for (i = 0; i < pow2indexInner_1; i++) {
                uint64_t nodeId0 = sinks + i;
                uint64_t nodeId1 = sinks + i + pow2indexInner_1;
                uint8_t *parent0 = GetNode(g, sinks - pow2indexInner_1 + i);
                uint8_t *parent1_0 = GetNode(g, sources + i);
                uint8_t *parent1_1 = GetNode(g, sources + i + pow2indexInner_1);
                uint8_t *buf = (uint8_t *) malloc(NODE_SIZE);
                WriteVarInt(buf, nodeId0);
                uint8_t *hashInput = (uint8_t *) malloc(NODE_SIZE * 4);
                memcpy(hashInput, g->pk, NODE_SIZE);
                memcpy(hashInput + NODE_SIZE, buf, NODE_SIZE);
                memcpy(hashInput + (NODE_SIZE * 2), parent0, NODE_SIZE);
                memcpy(hashInput + (NODE_SIZE * 3), parent1_0, NODE_SIZE);
                uint8_t *hashOutput0 = (uint8_t *) malloc(NODE_SIZE);
                sha3(hashInput, NODE_SIZE * 4, hashOutput0, NODE_SIZE);
                WriteVarInt(buf, nodeId1);
                memcpy(hashInput, g->pk, NODE_SIZE);
                memcpy(hashInput + NODE_SIZE, buf, NODE_SIZE);
                memcpy(hashInput + (NODE_SIZE * 2), parent0, NODE_SIZE);
                memcpy(hashInput + (NODE_SIZE * 3), parent1_1, NODE_SIZE);
                uint8_t *hashOutput1 = (uint8_t *) malloc(NODE_SIZE);
                sha3(hashInput, NODE_SIZE * 4, hashOutput1, NODE_SIZE);
                NewNode(g, nodeId0, hashOutput0);
                NewNode(g, nodeId1, hashOutput1);
                count += 2;
                free(parent0); free(parent1_0); free(parent1_1); free(buf);
                free(hashInput); free(hashOutput0); free(hashOutput1);
            }
        }

        if ((graph == 0 || graph == 3) || ((graph == 1 || graph == 2) && index == 2)) {
            ButterflyGraph(g, index - 1, &count);
        } else if (graph == 1 || graph == 2) {
            int64_t *tempStack = (int64_t *) malloc(sizeof(int64_t) * (stackSize + indicesSize));
            memcpy(tempStack, stack, stackSize * sizeof(int64_t));
            memcpy(tempStack + stackSize, indices, indicesSize * sizeof(int64_t));
            stackSize += indicesSize;
            free(stack); stack = tempStack;

            int32_t *tempGraphStack = (int32_t *) malloc(sizeof(int32_t) * (graphStackSize + graphsSize));
            memcpy(tempGraphStack, graphStack, graphStackSize * sizeof(int32_t));
            memcpy(tempGraphStack + graphStackSize, graphs, graphsSize * sizeof(int32_t));
            graphStackSize += graphsSize;
            free(graphStack); graphStack = tempGraphStack;
        }

        free(indices); free(graphs);
    }

    free(stack); free(graphStack);
}

static struct Graph *NewGraph(int64_t index, const char *targetFile, uint8_t *pk)
{
    // "wb+" always truncates the target, so always (re)generate. (The reference
    // skipped generation when the file pre-existed -- which, combined with the
    // truncating open, left an empty file. We overwrite unconditionally.)
    FILE *db = fopen(targetFile, "wb+");
    if (!db) { fprintf(stderr, "verthash: cannot create '%s'\n", targetFile); return NULL; }

    int64_t size = numXi(index);
    int64_t log2 = Log2(size) + 1;
    int64_t pow2 = 1 << ((uint64_t) log2);

    struct Graph *g = (struct Graph *) malloc(sizeof(struct Graph));
    if (!g) { fclose(db); return NULL; }

    g->db = db; g->log2 = log2; g->pow2 = pow2; g->pk = pk; g->index = index;

    XiGraphIter(g, index);

    fclose(db);
    return g;
}

int verthash_generate_data_file(const char *output_file_name)
{
    const char *hashInput = "Verthash Proof-of-Space Datafile";
    uint8_t *pk = (uint8_t *) malloc(NODE_SIZE);
    if (!pk) { fprintf(stderr, "verthash: pk alloc failed\n"); return -1; }

    sha3(hashInput, 32, pk, NODE_SIZE);

    int64_t index = 17;
    struct Graph *g = NewGraph(index, output_file_name, pk);
    free(pk);
    if (!g) { fprintf(stderr, "verthash: data file creation failed\n"); return -1; }
    free(g);
    return 0;
}
