#!/usr/bin/env bash
# HiveOS custom-miner config generator for ccminer-kudaraidee.
# Sourced by /hive/bin/custom after wallet/rig variables are in scope.

_hive_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z ${CUSTOM_VERSION:-} || -z ${CUSTOM_CONFIG_FILENAME:-} ]]; then
	_config_override=${CUSTOM_CONFIG_FILENAME:-}
	_version_override=${CUSTOM_VERSION:-}
	[[ -f "$_hive_dir/h-manifest.conf" ]] && . "$_hive_dir/h-manifest.conf"
	[[ -n $_config_override ]] && CUSTOM_CONFIG_FILENAME=$_config_override
	[[ -n $_version_override ]] && CUSTOM_VERSION=$_version_override
fi

# HiveOS asks the base custom h-config.sh for the version; report the manifest value.
miner_ver() {
	echo "${CUSTOM_VERSION:-}"
}

# The agent also sources h-config.sh while collecting stats, where the wallet
# variables are not in scope. Do not overwrite the live config in that case.
conf=""
if [[ -n ${CUSTOM_ALGO:-} && -n ${CUSTOM_URL:-} && -n ${CUSTOM_TEMPLATE:-} ]]; then
	conf=" -a ${CUSTOM_ALGO} -o ${CUSTOM_URL} -u ${CUSTOM_TEMPLATE}"
	[[ -n ${CUSTOM_PASS:-} ]] && conf+=" -p ${CUSTOM_PASS}"
	[[ -n ${CUSTOM_USER_CONFIG:-} ]] && conf+=" ${CUSTOM_USER_CONFIG}"
	printf '%s\n' "$conf" > "${CUSTOM_CONFIG_FILENAME:?CUSTOM_CONFIG_FILENAME is not set}"
fi

echo "$conf"
