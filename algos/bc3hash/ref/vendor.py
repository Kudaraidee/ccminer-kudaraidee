"""Vendor BitcoinIII Core's crypto/bc3 into algos/bc3hash/ref/ for the host oracle.

Source: BitcoinIII Core v31.1.2 (commit ac3dbe8b47), src/crypto/bc3/. Files are
copied verbatim except for these anchor-checked patches:

  uint.hpp  Wide = unsigned __int128      -> bc3_u128 (bc3_shim.h; native on gcc)
  mix.cpp   __builtin_prefetch            -> bc3_prefetch (bc3_shim.h)
  mix.cpp   euclid_batch / binary-Euclid inverse256 (signed __int128,
            __builtin_clzll/ctzll)       -> Fermat: a^(p-2) mod F256 via the
            coin's own Mod::pow. The inverse mod a prime is unique and both
            return 0 for a == 0 (mod p), so the output is identical.

usage: python vendor.py <bitcoiniii-core checkout>
"""
import os
import sys

core = sys.argv[1]
src_dir = os.path.join(core, "src", "crypto", "bc3")
here = os.path.dirname(os.path.abspath(__file__))
FILES = ["uint.hpp", "constants.hpp", "mix.cpp", "mix_internal.hpp", "hash.cpp",
         "standard_hashes.hpp", "standard_hashes.cpp", "include/bc3/bc3.hpp"]


def patch(text, old, new, name):
    n = text.count(old)
    if n != 1:
        sys.exit(f"{name}: anchor found {n}x, expected 1: {old[:60]!r}")
    return text.replace(old, new)


def cut(text, start, end, new, name):
    a = text.find(start)
    b = text.find(end, a)
    if a < 0 or b < 0 or text.count(start) != 1:
        sys.exit(f"{name}: cut anchors not found / not unique")
    return text[:a] + new + text[b + len(end):]


FERMAT = """} // namespace
// ccminer oracle: Fermat inversion replaces the coin's binary Euclid, which
// needs signed 128-bit integers (see vendor.py). Same unique result, including
// inverse256(0) == 0.
U256 inverse256(U256 a)
{
    a = F256::normalize(a);
    if (a.zero()) return {};
    return F256::pow(a, F256::modulus() - U256(2));
}
"""

for f in FILES:
    text = open(os.path.join(src_dir, f), encoding="utf-8", newline="").read()
    if f == "uint.hpp":
        text = patch(text, "#include <utility>\n", "#include <utility>\n\n#include \"bc3_shim.h\"\n", f)
        text = patch(text, "using Wide = unsigned __int128;", "using Wide = bc3_u128;", f)
    if f == "mix.cpp":
        text = patch(text, "__builtin_prefetch(points[n], 0, 3);", "bc3_prefetch(points[n]);", f)
        text = cut(text, "// Coefficients stay in [0,p).", "    return u == one ? x : y;\n}\n", FERMAT, f)
    for bad in ("__int128", "__builtin_"):
        if bad in text:
            sys.exit(f"{f}: {bad} survives the patches")
    out = os.path.join(here, f)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8", newline="").write(text)
    print("vendored", f)

# MIT: the copyright and permission notice travels with the copied files
lic = open(os.path.join(core, "COPYING"), encoding="utf-8", newline="").read()
open(os.path.join(here, "COPYING"), "w", encoding="utf-8", newline="").write(lic)
print("vendored COPYING")
