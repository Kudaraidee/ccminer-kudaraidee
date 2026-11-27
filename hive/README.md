# HiveOS custom miner

`hive/` contains the callbacks HiveOS expects from a custom miner. They use the
stock custom-miner variables (`CUSTOM_URL`, `CUSTOM_ALGO`, `CUSTOM_TEMPLATE`,
`CUSTOM_PASS`, `CUSTOM_USER_CONFIG`) and the binary/telnet API on port `4068`.

## Files

| File | Purpose |
| :-- | :-- |
| `h-manifest.conf` | Miner name, version, config/log paths and API port. |
| `h-config.sh` | Writes `CUSTOM_CONFIG_FILENAME` from the wallet/rig settings. |
| `h-run.sh` | Starts `./ccminer -b 127.0.0.1:$MINER_API_PORT <config>`. |
| `h-stats.sh` | Publishes `$khs` and `$stats` JSON for the HiveOS agent. |
| `package.sh` | Creates `ccminer-kudaraidee-<version>.tar.gz` for installation. |

The `.sh` files are executable in git; HiveOS requires that.

## Build and install

On a machine with the CUDA 11.8 toolkit:

```bash
./build.sh
./hive/package.sh
```

Copy `ccminer-kudaraidee-1.4.0.tar.gz` to the rig (for example with
`hive-shell` or scp) and install it with:

```bash
custom-get https://example.org/ccminer-kudaraidee-1.4.0.tar.gz
```

Then select **Custom** as the miner in the Flight Sheet, choose
`ccminer-kudaraidee` and fill in:

| Field | Value |
| :-- | :-- |
| Coin | Any coin; the algo is taken from the Algo field |
| Algo | ccminer `-a` name, e.g. `kawpow`, `nebhash`, `ethash` |
| URL | Pool URL, including `stratum+ssl://` when needed |
| Wallet | Template passed to `-u`, e.g. `wallet.worker` |
| Password | Passed to `-p`, e.g. `c=NEB` or `x=password` |

Everything in **User config** is appended verbatim. For Nebhash the official
dataset must be present on the rig:

```text
--neb-dataset=/hive/miners/custom/ccminer-kudaraidee/neb-datasets.bin
```

Put `neb-datasets.bin` in the miner directory (or change the path above). The
pool's `dataset_hash` is checked at subscribe time; a mismatched file is
refused.

## Stats

`h-stats.sh` reads `summary` and `threads` from the binary API and emits:

- `$khs` — total hashrate in kH/s, summed from per-GPU `KHS=`
- `$stats` — JSON with `hs`, `hs_units: "khs"`, `temp`, `fan`, `uptime`,
  `ar: [accepted, rejected]`, `bus_numbers`, `algo` and `ver`

A GPU reporting `TEMP=0.0` is treated as not mining and contributes `0` to
`$khs`. If the API cannot be reached, `$khs` is `0` and `$stats` is an empty
but valid JSON object, so the agent keeps reporting instead of breaking.

The API port is fixed at `4068` in `h-manifest.conf`; do not pass a different
`-b` value in User config unless you edit the manifest too.
