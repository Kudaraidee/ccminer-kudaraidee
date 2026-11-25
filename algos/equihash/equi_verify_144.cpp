// Host-side consensus verifier for Equihash 144/5.
//
// equi_tromp.h derives everything from WN/WK, so the two variants are the same
// verifier in separate TUs with different defines and distinct namespaces.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "blake2/blake2.h"      // blake2b_state/param + eq_blake2b_* (extern "C")
#include "equi_verify.h"

#define WN 144
#define WK 5

namespace tromp144 {
#include "equi_tromp.h"         // params, setheader, verifyrec, verify
}

extern "C" int eq_verify_144(const char *headernonce, const char *personal,
                             const uint32_t *indices)
{
	// verify() takes a non-const array; duped() sorts a copy and verifyrec
	// walks it read-only.
	uint32_t idx[1 << WK];
	memcpy(idx, indices, sizeof(idx));
	return tromp144::verify(idx, headernonce, HEADERNONCELEN, personal);
}
