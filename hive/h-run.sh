#!/usr/bin/env bash
# HiveOS custom-miner launcher for ccminer-kudaraidee.
# Based on https://github.com/minershive/hiveos-linux/tree/master/hive/miners/ccminer
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
. ./h-manifest.conf

if pgrep -x ccminer >/dev/null 2>&1; then
	echo -e "${RED-}${CUSTOM_NAME} miner is already running${NOCOLOR-}"
	exit 1
fi

# Release TIME_WAIT sockets left by a previous miner on the API port. Some
# HiveOS images do not ship netstat/killcx, so this cleanup is optional.
if command -v netstat >/dev/null 2>&1 && command -v killcx >/dev/null 2>&1; then
	while true; do
		connections="$(netstat -anp 2>/dev/null | grep TIME_WAIT | grep "$MINER_API_PORT" | awk '{print $5}')"
		[[ -z $connections ]] && break
		for con in $connections; do
			killcx "$con" lo
		done
	done
fi

CUSTOM_LOG_BASEDIR="$(dirname "$CUSTOM_LOG_BASENAME")"
mkdir -p "$CUSTOM_LOG_BASEDIR"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/hive/lib"

./ccminer -b "127.0.0.1:$MINER_API_PORT" $(< "${CUSTOM_CONFIG_FILENAME:?CUSTOM_CONFIG_FILENAME is not set}") 2>&1 |
	tee --append "${CUSTOM_LOG_BASENAME}.log"
exit "${PIPESTATUS[0]}"
