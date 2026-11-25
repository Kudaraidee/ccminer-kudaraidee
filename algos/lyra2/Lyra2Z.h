/**
 * Header file for the Lyra2 Password Hashing Scheme (PHS).
 *
 * Author: The Lyra PHC team (http://www.lyra-kdf.net/) -- 2014.
 *
 * This software is hereby placed in the public domain.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef LYRA2Z_H_
#define LYRA2Z_H_

#include <stdint.h>

typedef unsigned char byte;

//Block length required so Blake2's Initialization Vector (IV) is not overwritten (THIS SHOULD NOT BE MODIFIED)
#define BLOCK_LEN_BLAKE2_SAFE_INT64 8                                   //512 bits (=64 bytes, =8 uint64_t)
#define BLOCK_LEN_BLAKE2_SAFE_BYTES (BLOCK_LEN_BLAKE2_SAFE_INT64 * 8)   //same as above, in bytes


#ifdef BLOCK_LEN_BITS
        #define BLOCK_LEN_INT64 (BLOCK_LEN_BITS/64)      //Block length: 768 bits (=96 bytes, =12 uint64_t)
        #define BLOCK_LEN_BYTES (BLOCK_LEN_BITS/8)       //Block length, in bytes
#else   //default block lenght: 768 bits
        #define BLOCK_LEN_INT64 12                       //Block length: 768 bits (=96 bytes, =12 uint64_t)
        #define BLOCK_LEN_BYTES (BLOCK_LEN_INT64 * 8)    //Block length, in bytes
#endif

// Row indices use the generic (modulo) form, so nRows need not be a power of 2:
// lyra2z uses 8 rows, lyra2z330 uses 330. LYRA2Z allocates its own memory matrix;
// LYRA2Z_reuse takes a caller-owned one (nRows * nCols * BLOCK_LEN_BYTES bytes),
// which a hashing loop wants since lyra2z330's matrix is ~7.7 MB.
int LYRA2Z(void *K, int64_t kLen, const void *pwd, int32_t pwdlen, const void *salt, int32_t saltlen, int64_t timeCost, const int16_t nRows, const int16_t nCols);
int LYRA2Z_reuse(uint64_t *wholeMatrix, void *K, int64_t kLen, const void *pwd, int32_t pwdlen, const void *salt, int32_t saltlen, int64_t timeCost, const int16_t nRows, const int16_t nCols);

#endif /* LYRA2_H_ */
