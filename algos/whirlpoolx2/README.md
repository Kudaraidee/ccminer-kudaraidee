# whirlpoolx2 — CapStash PoW (`-a whirlpoolx2`)

## Provenance

New port (2026-07). Cloned from the fork's (unbuilt) `algos/x15/whirlpoolx.cu` +
`cuda_whirlpoolx.cu` — the heavily-optimised, midstate-precomputed Whirlpool-80
kernel (djm34 / tpruvot / SP / Provos Alexis lineage, MIT). Symbols renamed to
`whirlpoolx2_*` so both could coexist; only the 512→256 fold offset differs.

- **Consensus (authoritative):** CapStash-Core `src/primitives/block.cpp`
  `CBlockHeader::GetPoWHash()`.

## Consensus definition

```
buf[80] = LE header ( nVersion | hashPrevBlock | hashMerkleRoot | nTime | nBits | nNonce )
wh[64]  = Whirlpool512(buf, 80)                  // one standard ISO Whirlpool-512
out[32] : for i in 0..31: out[i] = wh[i] ^ wh[i+32]   // fold 512->256, clean halves
PoWhash = uint256(out)                           // little-endian compare, fulltest
```

- **Single** Whirlpool-512 over the 80-byte header (no second pass despite the
  "x2" name), folded to 256 bits by XORing the two clean 32-byte halves.
- The fold offset is the *only* algorithmic difference from whirlpoolx, which
  folds at offset 16 (the overlapping-halves Vanillacoin/XVC quirk). In 64-bit
  words: whirlpoolx2 out = (w0^w4, w1^w5, w2^w6, w3^w7); whirlpoolx = (w0^w2,
  w1^w3, w2^w4, w3^w5).
- Standard Bitcoin-Core-fork header/stratum: 32-bit nonce at `pdata[19]`,
  big-endian header words, sha256d merkle (the `default` path), LE (MSW-last)
  256-bit target compare (`vhash64[7] <= ptarget[7]`, `fulltest`) — the
  whirlpool-coin path, **not** the sha256 MSW-first path.

## Fold in the fused kernel (the non-obvious part)

The whirlpoolx kernel never materialises the full 512-bit state. It computes
only the **single top fold word** needed for the target screen and re-verifies
candidates on the CPU. So the offset lives in three device spots, not one
(`cuda_whirlpoolx2.cu`):

1. precompute feed-forward `atLastCalc = h[3] ^ h[7]`   (was `h[3] ^ h[5]`)
2. precompute last-round key word `ROUND_ELT(tmp11, 7,6,5,4,3,2,1,0)` for
   `c_xtra[1]`   (was the word-5 pattern `5,4,3,2,1,0,7,6`)
3. final screen `xor3(c_xtra[1], ROUND_ELT(tmp, 3,...), ROUND_ELT(tmp, 7,6,5,4,3,2,1,0)) <= pTarget[3]`
   (was the word-5 pattern) — i.e. top word `wh[3] ^ wh[7]`.

The host reference `whirlpoolx2_hash` (sph_whirlpool + fold at +32) re-verifies
every GPU candidate before submit, so a wrong device fold is loud (persistent
"does not validate on CPU" + zero accepts), never a silent wrong hash.

## Validation

- Live CapStash pool: "accepted N/N, 0 rejects" is the definitive gate (host
  re-verify + fold cross-check baked into scanhash).
- `--benchmark` at loosened `ptarget[7]` fires candidates (Whirlpool-80 is fast,
  hundreds of MH/s) → non-vacuous CPU re-verify.
