# sha256dv (Veil)

SHA256Dv is Veil's PoW: ordinary double SHA-256 over an 80-byte stage2 buffer,
but with a 64-bit nonce (`nonce_hi:nonce_lo`), a coin-specific stage2 layout
and a bespoke stratum notify/submit path — a distinct algo, **not** a sha256d
kernel variant (which is why it has its own folder; see docs/coding-guideline.md §2).

Since the 2026-07 migration the host transform and constants come from
`cuda/sha256_device.cuh`; the kernel keeps its fused form (3-round host
prehash via `d_pre`, round-60 early-out, full MSW-first target compare) and an
init-time GPU/CPU self-test.

## Benchmarks

| Date | Card | Driver | CUDA | Intensity | Hashrate | Notes |
|---|---|---|---|---|---|---|
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (24) | 1346.0 MH/s | baseline before shared-header migration |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (24) | 1346.4 MH/s | after shared-header migration (02017c8) |
