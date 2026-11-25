#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

#ifdef __cplusplus
extern "C" {
#endif
	extern uint32_t opt_cuda_syncmode;
#define BITSTREAM_BUF_SIZE ((32) * (AES_BLOCK_SIZE))
#define N_NEIGHBORS (3)
#define SALT_LEN (32)
#define INLEN_MAX (1ull<<20)
#define TCOST_MIN 1ull
#define SCOST_MIN (1)
#define SCOST_MAX (UINT32_MAX)
#define BLOCKS_MIN (1ull)
#define THREADS_MAX 4096
#define BLOCK_SIZE (32)
#define UNUSED __attribute__ ((unused))
#define MAXRESULTS 8
//uint32_t num_cuda_threads = 64;
//uint32_t num_cuda_blocks = 48;
#define npt 1
#define blocksize 512 //was 512
	struct bitstream {
		bool initialized;
		uint8_t *zeros;
		SHA256_CTX c;
#if   OPENSSL_VERSION_NUMBER >= 0x10100000L
		EVP_CIPHER_CTX* ctx;
#else
		//EVP_CIPHER_CTX ctx;
#endif
	};

	struct hash_state;

	struct hash_state {
		uint64_t counter;
		uint64_t n_blocks;
		bool has_mixed;
		uint8_t *buffer;
		struct bitstream bstream;
		const struct balloon_options *opts;
	};

	struct balloon_options {
		int64_t s_cost;
		int32_t t_cost;
	};

	void balloon_128(unsigned char *input, unsigned char *output);
	void balloon_hash(unsigned char *input, unsigned char *output, int64_t s_cost, int32_t t_cost);
	void balloon(unsigned char *input, unsigned char *output, int32_t len, int64_t s_cost, int32_t t_cost);
	void balloon_reset();
	void balloon_init(struct balloon_options *opts, int64_t s_cost, int32_t t_cost);

	void bytes_to_littleend8_uint64(const uint8_t *bytes, uint64_t *out);
	int bitstream_fill_buffer(struct bitstream *b, void *out, size_t outlen);

	int hash_state_init(struct hash_state *s, const struct balloon_options *opts, const uint8_t salt[SALT_LEN]);
	int hash_state_free(struct hash_state *s);
	int hash_state_fill(struct hash_state *s, const uint8_t *in, size_t inlen);
	void hash_state_mix(struct hash_state *s, int32_t mixrounds);
	int hash_state_extract(const struct hash_state *s, uint8_t out[BLOCK_SIZE]);

	void balloon_128_orig(unsigned char *input, unsigned char *output);
	void balloon_128_openssl(unsigned char *input, unsigned char *output);
	void balloon_orig(unsigned char *input, unsigned char *output, int32_t len, int64_t s_cost, int32_t t_cost);
	void balloon_openssl(unsigned char *input, unsigned char *output, int32_t len, int64_t s_cost, int32_t t_cost);
	void hash_state_mix_orig(struct hash_state *s, int32_t mixrounds);
	void hash_state_mix_openssl(struct hash_state *s, int32_t mixrounds);


	uint32_t balloon_128_cuda(int thr_id, unsigned char *input, unsigned char *output, uint32_t *target, uint32_t max_nonce, uint32_t num_threads, uint32_t *is_winning, uint32_t num_blocks, uint32_t *h_nounce);
	uint32_t cuda_balloon(int thr_id, unsigned char *input, unsigned char *output, int32_t len, int64_t s_cost, int32_t t_cost, uint32_t *target, uint32_t max_nonce, uint32_t num_threads, uint32_t *is_winning, uint32_t num_blocks, uint32_t *h_nounce);
	void reset_host_prebuf(int thr_id);
	void balloon_cuda_init(int thr_id, uint32_t opt_cuda_syncmode, uint32_t num_threads, uint32_t num_blocks);
	void balloon_cuda_free(int thr_id);

	int cuda_query();



#ifdef __cplusplus
}
#endif