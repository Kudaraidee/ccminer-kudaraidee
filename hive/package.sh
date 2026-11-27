#!/usr/bin/env bash
# Build the HiveOS custom-miner archive expected by /hive/miners/custom/custom-get.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
srcdir="$(cd "$here/.." && pwd)"
version="$(sed -n 's/^CUSTOM_VERSION=//p' "$here/h-manifest.conf" | head -n 1)"
binary="$srcdir/build-linux/ccminer"

if [[ ! -x $binary ]]; then
	"$srcdir/build.sh"
fi

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
minerdir="$stage/ccminer-kudaraidee"
mkdir -p "$minerdir"

install -m 0755 "$binary" "$minerdir/ccminer"
install -m 0755 "$here/h-config.sh" "$minerdir/h-config.sh"
install -m 0755 "$here/h-run.sh" "$minerdir/h-run.sh"
install -m 0755 "$here/h-stats.sh" "$minerdir/h-stats.sh"
install -m 0644 "$here/h-manifest.conf" "$minerdir/h-manifest.conf"
install -m 0644 "$here/README.md" "$minerdir/README.md"

archive="$srcdir/ccminer-kudaraidee-$version.tar.gz"
tar -czf "$archive" -C "$stage" ccminer-kudaraidee
echo "$archive"
