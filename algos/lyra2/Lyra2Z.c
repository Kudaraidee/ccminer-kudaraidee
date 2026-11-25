/**
 * Implementation of the Lyra2 Password Hashing Scheme (PHS).
 *
 * Author: The Lyra PHC team (http://www.lyra-kdf.net/) -- 2014.
 *
 * This software is hereby placed in the public domain.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "Lyra2Z.h"
#include "Sponge.h"

/**
 * Executes Lyra2 based on the G function from Blake2b. This version supports salts and passwords
 * whose combined length is smaller than the size of the memory matrix, (i.e., (nRows x nCols x b) bits,
 * where "b" is the underlying sponge's bitrate). In this implementation, the "basil" is composed by all
 * integer parameters (treated as type "unsigned int") in the order they are provided, plus the value
 * of nCols, (i.e., basil = kLen || pwdlen || saltlen || timeCost || nRows || nCols).
 *
 * @param wholeMatrix Caller-owned memory matrix, nRows * nCols * BLOCK_LEN_BYTES
 *                    bytes. Its previous contents are irrelevant, so one buffer
 *                    can be reused across hashes.
 * @param K The derived key to be output by the algorithm
 * @param kLen Desired key length
 * @param pwd User password
 * @param pwdlen Password length
 * @param salt Salt
 * @param saltlen Salt length
 * @param timeCost Parameter to determine the processing time (T)
 * @param nRows Number or rows of the memory matrix (R)
 * @param nCols Number of columns of the memory matrix (C)
 *
 * @return 0 if the key is generated correctly; -1 if there is an error (usually due to lack of memory for allocation)
 */
int LYRA2Z_reuse(uint64_t *wholeMatrix, void *K, int64_t kLen, const void *pwd, int32_t pwdlen, const void *salt, int32_t saltlen, int64_t timeCost, const int16_t nRows, const int16_t nCols)
{
	//============================= Basic variables ============================//
	int64_t row = 2; //index of row to be processed
	int64_t prev = 1; //index of prev (last row ever computed/modified)
	int64_t rowa = 0; //index of row* (a previous row, deterministically picked during Setup and randomly picked while Wandering)
	int64_t tau; //Time Loop iterator
	int64_t step = 1; //Visitation step (used during Setup and Wandering phases)
	int64_t window = 2; //Visitation window (used to define which rows can be revisited during Setup)
	int64_t gap = 1; //Modifier to the step, assuming the values 1 or -1
	int64_t i; //auxiliary iteration counter
	int64_t v64; // 64bit var for memcpy
	//==========================================================================/

	//===================== Pointers into the memory matrix ====================//
	// The caller owns the matrix (nRows rows of ROW_LEN_BYTES); rows are indexed
	// arithmetically, so no per-call allocation happens here at all. Every row is
	// fully written by the Setup phase before it can be read, so a reused (dirty)
	// matrix yields the same digest as a fresh one -- only the pwd/salt/basil
	// input blocks are explicitly cleared below.

	const int64_t ROW_LEN_INT64 = BLOCK_LEN_INT64 * nCols;
	// for Lyra2REv2, nCols = 4, v1 was using 8
	const int64_t BLOCK_LEN = BLOCK_LEN_BLAKE2_SAFE_INT64;

	#define ROW(idx) (&wholeMatrix[(size_t)(idx) * (size_t)ROW_LEN_INT64])
	//==========================================================================/

	//============= Getting the password + salt + basil padded with 10*1 ===============//
	//OBS.:The memory matrix will temporarily hold the password: not for saving memory,
	//but this ensures that the password copied locally will be overwritten as soon as possible

	//First, we clean enough blocks for the password, salt, basil and padding
	int64_t nBlocksInput = ((saltlen + pwdlen + 6 * sizeof(uint64_t)) / BLOCK_LEN_BLAKE2_SAFE_BYTES) + 1;

	byte *ptrByte = (byte*) wholeMatrix;

	//Prepends the password
	memcpy(ptrByte, pwd, pwdlen);
	ptrByte += pwdlen;

	//Concatenates the salt
	memcpy(ptrByte, salt, saltlen);
	ptrByte += saltlen;

	memset(ptrByte, 0, (size_t) (nBlocksInput * BLOCK_LEN_BLAKE2_SAFE_BYTES - (saltlen + pwdlen)));

	//Concatenates the basil: every integer passed as parameter, in the order they are provided by the interface
	memcpy(ptrByte, &kLen, sizeof(int64_t));
	ptrByte += sizeof(uint64_t);
	v64 = pwdlen;
	memcpy(ptrByte, &v64, sizeof(int64_t));
	ptrByte += sizeof(uint64_t);
	v64 = saltlen;
	memcpy(ptrByte, &v64, sizeof(int64_t));
	ptrByte += sizeof(uint64_t);
	v64 = timeCost;
	memcpy(ptrByte, &v64, sizeof(int64_t));
	ptrByte += sizeof(uint64_t);
	v64 = nRows;
	memcpy(ptrByte, &v64, sizeof(int64_t));
	ptrByte += sizeof(uint64_t);
	v64 = nCols;
	memcpy(ptrByte, &v64, sizeof(int64_t));
	ptrByte += sizeof(uint64_t);

	//Now comes the padding
	*ptrByte = 0x80; //first byte of padding: right after the password
	ptrByte = (byte*) wholeMatrix; //resets the pointer to the start of the memory matrix
	ptrByte += nBlocksInput * BLOCK_LEN_BLAKE2_SAFE_BYTES - 1; //sets the pointer to the correct position: end of incomplete block
	*ptrByte ^= 0x01; //last byte of padding: at the end of the last incomplete block
	//==========================================================================/

	//======================= Initializing the Sponge State ====================//
	//Sponge state: 16 uint64_t, BLOCK_LEN_INT64 words of them for the bitrate (b) and the remainder for the capacity (c)
	uint64_t state[16];
	initState(state);
	//==========================================================================/

	//================================ Setup Phase =============================//
	//Absorbing salt, password and basil: this is the only place in which the block length is hard-coded to 512 bits
	uint64_t *ptrWord = wholeMatrix;
	for (i = 0; i < nBlocksInput; i++) {
		absorbBlockBlake2Safe(state, ptrWord); //absorbs each block of pad(pwd || salt || basil)
		ptrWord += BLOCK_LEN; //goes to next block of pad(pwd || salt || basil)
	}

	//Initializes M[0] and M[1]
	reducedSqueezeRow0(state, ROW(0), nCols); //The locally copied password is most likely overwritten here

	reducedDuplexRow1(state, ROW(0), ROW(1), nCols);

	do {
		//M[row] = rand; //M[row*] = M[row*] XOR rotW(rand)

		reducedDuplexRowSetup(state, ROW(prev), ROW(rowa), ROW(row), nCols);

		//updates the value of row* (deterministically picked during Setup))
		rowa = (rowa + step) & (window - 1);
		//update prev: it now points to the last row ever computed
		prev = row;
		//updates row: goes to the next row to be computed
		row++;

		//Checks if all rows in the window where visited.
		if (rowa == 0) {
		step = window + gap; //changes the step: approximately doubles its value
		window *= 2; //doubles the size of the re-visitation window
		gap = -gap; //inverts the modifier to the step
	}

	} while (row < nRows);
	//==========================================================================/

	//============================ Wandering Phase =============================//
	row = 0; //Resets the visitation to the first row of the memory matrix
	for (tau = 1; tau <= timeCost; tau++) {
		//Step is approximately half the number of all rows of the memory matrix for an odd tau; otherwise, it is -1
		step = (tau % 2 == 0) ? -1 : nRows / 2 - 1;
		do {
			//Selects a pseudorandom index row*
			//------------------------------------------------------------------------------------------
			// The "generic" (modulo) form: required for a non-power-of-2 nRows
			// (lyra2z330 uses 330). For a power-of-2 nRows this is bit-identical
			// to the classic `& (nRows-1)` mask, including the negative-step wrap,
			// because the modulo is evaluated on the unsigned value -- so the
			// lyra2z (8 rows) digest is unchanged. Matches the reference miners.
			rowa = (uint64_t)state[0] % (uint64_t)nRows;
			//------------------------------------------------------------------------------------------

			//Performs a reduced-round duplexing operation over M[row*] XOR M[prev], updating both M[row*] and M[row]
			reducedDuplexRow(state, ROW(prev), ROW(rowa), ROW(row), nCols);

			//update prev: it now points to the last row ever computed
			prev = row;

			//updates row: goes to the next row to be computed
			//------------------------------------------------------------------------------------------
			row = (int64_t)((uint64_t)(row + step) % (uint64_t)nRows);
			//------------------------------------------------------------------------------------------

		} while (row != 0);
	}

	//============================ Wrap-up Phase ===============================//
	//Absorbs the last block of the memory matrix
	absorbBlock(state, ROW(rowa));

	//Squeezes the key
	squeeze(state, K, (unsigned int) kLen);

	return 0;
}
#undef ROW

/**
 * Same as LYRA2Z_reuse, but allocates (and frees) the memory matrix itself.
 * Convenient for one-shot callers such as the host-side verification of a GPU
 * candidate; a hashing loop should reuse one matrix instead, since for
 * lyra2z330 this allocation is ~7.7 MB per call.
 */
int LYRA2Z(void *K, int64_t kLen, const void *pwd, int32_t pwdlen, const void *salt, int32_t saltlen, int64_t timeCost, const int16_t nRows, const int16_t nCols)
{
	const size_t sz = (size_t)BLOCK_LEN_INT64 * nCols * 8 * nRows;
	uint64_t *wholeMatrix = malloc(sz);
	if (wholeMatrix == NULL)
		return -1;

	int rc = LYRA2Z_reuse(wholeMatrix, K, kLen, pwd, pwdlen, salt, saltlen, timeCost, nRows, nCols);

	free(wholeMatrix);
	return rc;
}

