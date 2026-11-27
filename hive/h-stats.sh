#!/usr/bin/env bash
# HiveOS custom-miner stats for ccminer-kudaraidee.
# Sourced by the HiveOS agent, so this script must define $khs (kH/s) and $stats.

_hive_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z ${MINER_API_PORT:-} ]]; then
	[[ -f "$_hive_dir/h-manifest.conf" ]] && . "$_hive_dir/h-manifest.conf"
fi

_ccminer_field() {
	printf '%s\n' "$1" | tr ';' '\n' | grep -m1 "^$2=" | sed -e 's/.*=//'
}

_ccminer_json_array() {
	(($#)) || { printf '[]'; return; }
	printf '%s\n' "$@" | jq -cs 'map(tonumber? // 0)'
}

api_timeout=${API_TIMEOUT:-5}
khs=0
stats='{"hs":[],"hs_units":"khs","temp":[],"fan":[],"uptime":0,"ar":[0,0],"bus_numbers":[],"algo":"","ver":""}'

raw_threads="$(echo "threads" | nc -w "$api_timeout" localhost "${MINER_API_PORT}" 2>/dev/null)"
nc_rc=$?
threads="$(printf '%s' "$raw_threads" | tr -d '\0')"
if [[ $nc_rc -ne 0 || -z $threads ]]; then
	echo -e "${YELLOW-}Failed to read ${CUSTOM_NAME:-ccminer-kudaraidee} stats from localhost:${MINER_API_PORT}${NOCOLOR-}"
	return 0 2>/dev/null || exit 0
fi

raw_summary="$(echo "summary" | nc -w 2 localhost "${MINER_API_PORT}" 2>/dev/null)"
summary="$(printf '%s' "$raw_summary" | tr -d '\0')"

uptime="$(_ccminer_field "$summary" UPTIME)"
algo="$(_ccminer_field "$summary" ALGO)"
ac="$(_ccminer_field "$summary" ACC)"
rj="$(_ccminer_field "$summary" REJ)"
ver="$(_ccminer_field "$summary" VER)"

[[ $uptime =~ ^[0-9]+$ ]] || uptime=0
[[ $ac =~ ^[0-9]+$ ]] || ac=0
[[ $rj =~ ^[0-9]+$ ]] || rj=0
[[ -n $algo ]] || algo="${CUSTOM_ALGO:-}"
[[ -n $ver ]] || ver="${CUSTOM_VERSION:-}"

striplines="$(printf '%s\n' "$threads" | tr '|' '\n' | tr ';' '\n' | tr -cd '\11\12\15\40-\176')"

mapfile -t cckhs < <(printf '%s\n' "$striplines" | grep 'KHS=' | sed -e 's/.*=//')
mapfile -t cctemps < <(printf '%s\n' "$striplines" | grep 'TEMP=' | sed -e 's/.*=//')
mapfile -t ccbusids < <(printf '%s\n' "$striplines" | grep 'BUS=' | sed -e 's/.*=//')
mapfile -t ccfans < <(printf '%s\n' "$striplines" | grep 'FAN=' | sed -e 's/.*=//')

# A zero temperature means this GPU did not come up; do not credit its hashrate.
for i in "${!cckhs[@]}"; do
	[[ ${cctemps[$i]:-} == "0.0" ]] && cckhs[$i]=0
done

khs="$(awk -v values="${cckhs[*]:-}" 'BEGIN {
	n = split(values, a, " ")
	total = 0
	for (i = 1; i <= n; i++) if (a[i] != "") total += a[i]
	printf "%.3f", total
}')"

hs_json="$(_ccminer_json_array "${cckhs[@]}")"
temp_json="$(_ccminer_json_array "${cctemps[@]}")"
fan_json="$(_ccminer_json_array "${ccfans[@]}")"
bus_json="$(_ccminer_json_array "${ccbusids[@]}")"

stats="$(jq -n \
	--argjson hs "$hs_json" \
	--argjson temp "$temp_json" \
	--argjson fan "$fan_json" \
	--argjson uptime "$uptime" \
	--argjson ar "[$ac, $rj]" \
	--argjson bus_numbers "$bus_json" \
	--arg algo "$algo" \
	--arg ver "$ver" \
	'{hs: $hs, hs_units: "khs", temp: $temp, fan: $fan, uptime: $uptime,
	  ar: $ar, bus_numbers: $bus_numbers, algo: $algo, ver: $ver}')"
