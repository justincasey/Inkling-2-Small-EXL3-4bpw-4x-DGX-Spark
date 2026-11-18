#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# memguard.sh: one rank's memory guard, run on the node beside rexl3-serve (`./inkling2.sh start` ships it to
# BASE_DIR/run/memguard.sh and starts it before the rank). It waits for the rank's pid file, then samples MemAvailable
# and swap every GUARD_SAMPLE_S seconds and SIGKILLs the rank (and its children) when MemAvailable stays below the floor
# for 3 consecutive samples, or when swap use grows more than GUARD_SWAP_MAX_MIB above what it was at arming: on GB10 the
# GPU and the host share one memory pool, and a node that starts swapping takes sshd down with it instead of failing a
# request. The reason goes to GUARD_LOG and, as a `level=fatal` line, to the rank's own log. The guard exits when the rank
# is gone.
#
# Floors: GUARD_START_MIB while GUARD_STAGE_FILE says anything but `ready` (the pack is loading; `start` writes `ready`
# once rank 0 answers /v1/models), GUARD_MIB after. While loading, page cache is dropped every DROP_CACHES_EVERY_S seconds
# (0: never): GB10 allocations count only free memory, and a 73 GiB load leaves cache the allocator cannot reclaim in time.
#
# Records (GUARD_MIN_FILE, rewritten every sample, key=value lines): MemAvailable now, its minimum since arming (overall,
# while loading, once ready) and since the last memwatch mark: GUARD_WINDOW_FILE holds a token (`./inkling2.sh memwatch
# begin` writes a new one on every rank); when the token changes the window minimum restarts. Every GUARD_REPORT_S
# seconds a `memguard ... min_mib=` line also goes to GUARD_LOG.
#
# Variables (by name; no positional arguments):
#   GUARD_PIDFILE GUARD_RANK GUARD_LOG GUARD_MIN_FILE GUARD_STAGE_FILE GUARD_WINDOW_FILE GUARD_SERVE_LOG
#   GUARD_START_MIB GUARD_MIB GUARD_SWAP_MAX_MIB DROP_CACHES_EVERY_S
#   optional: GUARD_SAMPLE_S (0.5) GUARD_REPORT_S (60) GUARD_ARM_TIMEOUT_S (60) MEMINFO (/proc/meminfo, for tests)
#             GUARD_DROP_CMD (sudo -n sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches')
set -uo pipefail

for v in GUARD_PIDFILE GUARD_RANK GUARD_LOG GUARD_MIN_FILE GUARD_STAGE_FILE GUARD_WINDOW_FILE GUARD_SERVE_LOG \
  GUARD_START_MIB GUARD_MIB GUARD_SWAP_MAX_MIB DROP_CACHES_EVERY_S; do
  [[ -n ${!v:-} ]] || { echo "memguard: $v is not set" >&2; exit 2; }
done
for v in GUARD_START_MIB GUARD_MIB GUARD_SWAP_MAX_MIB DROP_CACHES_EVERY_S; do
  [[ ${!v} =~ ^[0-9]+$ ]] || { echo "memguard: $v=${!v} is not a whole number" >&2; exit 2; }
done
(( GUARD_MIB > 0 && GUARD_START_MIB > 0 )) || { echo "memguard: a floor of 0 MiB guards nothing" >&2; exit 2; }
meminfo=${MEMINFO:-/proc/meminfo}
sample_s=${GUARD_SAMPLE_S:-0.5}
report_s=${GUARD_REPORT_S:-60}
arm_timeout_s=${GUARD_ARM_TIMEOUT_S:-60}
drop_cmd=${GUARD_DROP_CMD:-"sudo -n sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'"}

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { printf '%s memguard rank=%s %s\n' "$(now)" "$GUARD_RANK" "$*" >> "$GUARD_LOG"; }
# MemAvailable and swap in use, MiB
read_mem() {
  awk '/^MemAvailable:/ {a = int($2 / 1024)} /^SwapTotal:/ {t = $2} /^SwapFree:/ {f = $2}
       END {if (a == "") exit 1; printf "%d %d\n", a, int((t - f) / 1024)}' "$meminfo"
}

read -r avail swap_base < <(read_mem) || true
[[ ${avail:-} =~ ^[0-9]+$ && ${swap_base:-} =~ ^[0-9]+$ ]] || { say "FAIL cannot read MemAvailable/swap from $meminfo"; exit 1; }
if (( DROP_CACHES_EVERY_S > 0 )); then
  bash -c "$drop_cmd" >/dev/null 2>&1 || { say "FAIL cannot drop page cache ($drop_cmd); set DROP_CACHES_EVERY_S=0 to run without"; exit 1; }
fi
say "waiting for $GUARD_PIDFILE (MemAvailable ${avail} MiB, swap ${swap_base} MiB)"

# arm on the rank's pid
pid=
for (( i = 0; i < arm_timeout_s * 4; i++ )); do
  if [[ -s $GUARD_PIDFILE ]]; then read -r pid < "$GUARD_PIDFILE"; [[ $pid =~ ^[0-9]+$ ]] && break; pid=; fi
  sleep 0.25
done
[[ -n $pid ]] || { say "FAIL no pid in $GUARD_PIDFILE after ${arm_timeout_s}s"; exit 1; }
kill -0 "$pid" 2>/dev/null || { say "FAIL pid $pid is not running"; exit 1; }
started=$(now)
say "armed pid=$pid floor_loading=${GUARD_START_MIB}MiB floor_ready=${GUARD_MIB}MiB swap_base=${swap_base}MiB swap_max_growth=${GUARD_SWAP_MAX_MIB}MiB drop_every=${DROP_CACHES_EVERY_S}s sample=${sample_s}s"

min=-1 min_at='' min_loading=-1 min_ready=-1 window='' window_min=-1 window_at='' low=0
last_drop=$SECONDS last_report=$SECONDS
kill_rank() {
  local why=$1
  { say "KILL $why: SIGKILL pid $pid and its children"
    ps -o pid,rss,vsz,args -p "$pid" | cut -c1-200
    head -8 "$meminfo"; } >> "$GUARD_LOG" 2>&1
  pkill -9 -P "$pid" 2>/dev/null
  kill -9 "$pid" 2>/dev/null
  printf '%s rexl3 level=fatal event=memguard_kill rank=%s reason="%s"\n' "$(now)" "$GUARD_RANK" "$why" >> "$GUARD_SERVE_LOG"
}

while kill -0 "$pid" 2>/dev/null; do
  read -r avail swap < <(read_mem) || true
  if ! [[ ${avail:-} =~ ^[0-9]+$ && ${swap:-} =~ ^[0-9]+$ ]]; then
    kill_rank "cannot read $meminfo"; break   # blind is not guarded
  fi
  stage=loading; floor=$GUARD_START_MIB
  if [[ -s $GUARD_STAGE_FILE ]] && read -r s < "$GUARD_STAGE_FILE" && [[ $s == ready ]]; then stage=ready floor=$GUARD_MIB; fi
  mark=; [[ -s $GUARD_WINDOW_FILE ]] && read -r mark < "$GUARD_WINDOW_FILE"
  if [[ $mark != "$window" ]]; then window=$mark window_min=-1; fi
  t=$(now)
  if (( min < 0 || avail < min )); then min=$avail min_at=$t; fi
  if (( window_min < 0 || avail < window_min )); then window_min=$avail window_at=$t; fi
  if [[ $stage == ready ]]; then (( min_ready < 0 || avail < min_ready )) && min_ready=$avail
  else (( min_loading < 0 || avail < min_loading )) && min_loading=$avail; fi
  printf 'rank=%s\npid=%s\nstarted=%s\nupdated=%s\nstage=%s\nnow_mib=%s\nmin_mib=%s\nmin_at=%s\nloading_min_mib=%s\nready_min_mib=%s\nwindow=%s\nwindow_min_mib=%s\nwindow_min_at=%s\nswap_base_mib=%s\nswap_now_mib=%s\nfloor_mib=%s\n' \
    "$GUARD_RANK" "$pid" "$started" "$t" "$stage" "$avail" "$min" "$min_at" "$min_loading" "$min_ready" "$window" \
    "$window_min" "$window_at" "$swap_base" "$swap" "$floor" > "$GUARD_MIN_FILE"
  if (( swap - swap_base > GUARD_SWAP_MAX_MIB )); then
    kill_rank "swap grew $((swap - swap_base)) MiB > ${GUARD_SWAP_MAX_MIB} MiB (MemAvailable ${avail} MiB)"; break
  fi
  if (( avail < floor )); then low=$((low + 1)); else low=0; fi
  if (( low >= 3 )); then
    kill_rank "MemAvailable ${avail} MiB < ${floor} MiB ($stage floor) for 3 samples"; break
  fi
  if (( SECONDS - last_report >= report_s )); then
    say "stage=$stage avail_mib=$avail min_mib=$min window_min_mib=$window_min swap_growth_mib=$((swap - swap_base))"
    last_report=$SECONDS
  fi
  if [[ $stage == loading ]] && (( DROP_CACHES_EVERY_S > 0 && SECONDS - last_drop >= DROP_CACHES_EVERY_S )); then
    bash -c "$drop_cmd" >/dev/null 2>&1 || say "WARN page-cache drop failed"
    last_drop=$SECONDS
  fi
  sleep "$sample_s"
done
say "pid $pid gone; guard exits (min_mib=$min at $min_at, loading_min_mib=$min_loading, ready_min_mib=$min_ready)"
