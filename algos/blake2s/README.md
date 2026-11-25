# blake2s (`-a blake2s`)

BLAKE2s proof-of-work (tpruvot lineage, GPLv3), relocated from `Algo256/`.

- `blake2s.cu` — `scanhash_blake2s`; self-contained (the BLAKE2s device kernel
  is in this TU). CPU reference via `sph/blake2s.c`.

## Layout

Relocation only (layout B) — pure `git mv` + build-system repoint, no symbol or
include changes (the file's includes are `sph/*` + system headers, resolved
repo-root-relative and unaffected by the move).

## Validation

Rebuild + benchmark re-validation owed.
