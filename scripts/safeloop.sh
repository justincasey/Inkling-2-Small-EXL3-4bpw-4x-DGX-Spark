#!/bin/bash
# safeloop.sh: run cluster experiments with a guaranteed way back to the last known-good engine.
#
#   safeloop.sh run LABEL CANDIDATE_SHA [PHASES] [TIMEOUT_S]   one autoresearch.sh run of the HEAD of CANDIDATE_SRC
#                                                              (HEAD must equal CANDIDATE_SHA). PHASES empty = full run:
#                                                              11/11 gates + score => the candidate is PROMOTED to known-good.
#                                                              A screen (PHASES set) never promotes.
#   safeloop.sh cmd TIMEOUT_S 'shell command'                  any lab command under a timeout (logits check, 1M, ...)
#   safeloop.sh restore                                        put the known-good engine back and serving
#
# Every mode ends the same way: the known-good engine is installed on all four ranks and serving on :8000, whatever
# happened (gate failure, boot failure, timeout, kill signal). Only `run` of a full passing run changes what known-good is.
#
# State: .work/safeloop/state.env (GOOD_SHA, GOOD_FALLBACK), history: .work/safeloop/history.log
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

MODE="${1:-}"; shift || true
CAND_SRC="${CANDIDATE_SRC:-$HOME/projects/rexl3-inkling2-a}"
export REXL3_SRC="$CAND_SRC"          # inkling2.sh build reads the engine repository from here
STATE=.work/safeloop/state.env
HISTORY=.work/safeloop/history.log
LOCK=/tmp/inkling2-safeloop.lock
HEALTH=http://192.168.1.121:8000/health
NODES=(192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18)
mkdir -p .work/safeloop

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { echo "safeloop: FAIL: $*" >&2; exit 1; }
note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$HISTORY"; }

# ---- lock (mkdir; macOS has no flock). A dead holder's lock is reclaimed. ---------------------------------------------
if ! mkdir "$LOCK" 2>/dev/null; then
  holder=$(cat "$LOCK/pid" 2>/dev/null || echo "?")
  if [[ "$holder" =~ ^[0-9]+$ ]] && ! kill -0 "$holder" 2>/dev/null; then
    rm -rf "$LOCK"; mkdir "$LOCK" || { echo "safeloop: cannot take $LOCK" >&2; exit 3; }
  else
    echo "safeloop: another run holds $LOCK (pid $holder); refusing to run two GPU jobs at once" >&2; exit 3
  fi
fi
echo $$ > "$LOCK/pid"

# ---- known-good state ----------------------------------------------------------------------------------------------
[[ -f $STATE ]] || die "no $STATE (seed GOOD_SHA and GOOD_FALLBACK first)"
# shellcheck disable=SC1090
source "$STATE"
resolve() { git -C "$CAND_SRC" rev-parse --verify "$1^{commit}" 2>/dev/null; }
GOOD_FULL=$(resolve "$GOOD_SHA") || die "known-good $GOOD_SHA is not a commit in $CAND_SRC"

restore_env_example() { git diff --quiet -- .env.example || { git checkout -- .env.example; log "restored .env.example from git"; }; }
stop_engine() { ./inkling2.sh stop >/dev/null 2>&1 || true; }
wait_health() {   # seconds
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do
    curl -fs -m 5 "$HEALTH" 2>/dev/null | grep -q '"status":"ok"' && return 0
    sleep 5
  done
  return 1
}
installed_commits() { ./inkling2.sh doctor 2>&1 | grep -o '\] engine [0-9a-f]\{40\}' | awk '{print $3}' | sort -u | paste -sd' ' -; }

# ensure_good: known-good installed on all four ranks, started, healthy. Falls back once to GOOD_FALLBACK.
ensure_good() {
  local why="$1" inst full
  log "restoring known-good ${GOOD_SHA:0:12} ($why)"
  stop_engine; restore_env_example
  for full in "$GOOD_FULL" "$(resolve "${GOOD_FALLBACK:-HEAD}" || true)"; do
    [[ -n $full ]] || continue
    inst=$(installed_commits)
    if [[ $inst != "$full" ]]; then
      log "installing ${full:0:12} (installed: ${inst:-none})"
      if ! ./inkling2.sh build "$full" > .work/safeloop/restore-build.log 2>&1; then
        log "build of ${full:0:12} failed (.work/safeloop/restore-build.log)"; continue
      fi
    fi
    if ./inkling2.sh start > .work/safeloop/restore-start.log 2>&1 && wait_health 300; then
      log "serving ${full:0:12} on :8000"; note "RESTORED engine=${full:0:12} after: $why"; return 0
    fi
    log "start of ${full:0:12} failed (.work/safeloop/restore-start.log)"; stop_engine
  done
  note "RESTORE-FAILED after: $why"
  die "could not restore a serving engine; the cluster is DOWN (see .work/safeloop/restore-*.log)"
}

CHILD=""
on_exit() { rm -rf "$LOCK"; }
# Kill a process and everything below it (the harness is a tree of bash, ssh and python processes; killing only the direct child leaves them running).
killtree() { local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -TERM "$p" 2>/dev/null || true; }
# A trap runs only when the foreground command ends, so children run in the background and the script `wait`s for them: a signal then acts at once.
on_signal() {
  trap - INT TERM
  log "signal: stopping the children, restoring known-good"
  [[ -z $CHILD ]] || killtree "$CHILD"
  sleep 3
  ensure_good "signal during $MODE ${LABEL:-}"
  exit 130
}
trap on_exit EXIT
trap on_signal INT TERM

preflight() {
  git diff --quiet -- .env.example || { log ".env.example was modified; restoring"; git checkout -- .env.example; }
  [[ -z $(git status --porcelain --untracked-files=no) ]] || { git status --short >&2; die "recipe repo has uncommitted tracked changes"; }
  [[ -z $(git -C "$CAND_SRC" status --porcelain --untracked-files=no) ]] || die "engine checkout $CAND_SRC has uncommitted changes"
  stop_engine
  local h used
  for h in "${NODES[@]}"; do
    used=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "spark@$h" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l' 2>&1) || die "cannot reach $h"
    [[ $used == 0 ]] || die "GPU on $h is busy ($used compute apps)"
  done
}

case "$MODE" in
  restore)
    ensure_good "manual restore"
    ;;

  cmd)
    TIMEOUT_S="${1:?usage: safeloop.sh cmd TIMEOUT_S 'command'}"; CMD="${2:?command}"
    LABEL="cmd"
    preflight
    log "running under timeout ${TIMEOUT_S}s: $CMD"
    timeout "$TIMEOUT_S" bash -c "$CMD" & CHILD=$!
    wait "$CHILD"; RC=$?; CHILD=""
    note "CMD rc=$RC: $CMD"
    ensure_good "after lab command (rc=$RC)"
    exit "$RC"
    ;;

  run)
    LABEL="${1:?usage: safeloop.sh run LABEL CANDIDATE_SHA [PHASES] [TIMEOUT_S]}"
    CAND="${2:?candidate sha}"; PHASES="${3:-}"; TIMEOUT_S="${4:-4200}"
    CAND_FULL=$(resolve "$CAND") || die "candidate $CAND is not a commit in $CAND_SRC"
    [[ $(git -C "$CAND_SRC" rev-parse HEAD) == "$CAND_FULL" ]] || die "HEAD of $CAND_SRC is not $CAND (autoresearch.sh builds HEAD)"
    preflight
    FULL=0; [[ -z $PHASES ]] && FULL=1
    log "run $LABEL: candidate ${CAND_FULL:0:12} (known-good ${GOOD_SHA:0:12}) phases=${PHASES:-FULL} timeout=${TIMEOUT_S}s"
    START=$(date +%s)
    CANDIDATE_SRC="$CAND_SRC" PHASES="$PHASES" timeout "$TIMEOUT_S" ./autoresearch.sh > ".work/safeloop/${LABEL}.log" 2>&1 & CHILD=$!
    wait "$CHILD"; RC=$?; CHILD=""
    ELAPSED=$(( $(date +%s) - START ))
    RUN_DIR=$(grep -o -m1 'run=bench/runs/autoresearch-[0-9-]*' ".work/safeloop/${LABEL}.log" | head -1 | cut -d= -f2)
    SCORE=$(grep -E '^METRIC target_score=' "${RUN_DIR:-/nonexistent}/metrics.txt" 2>/dev/null | cut -d= -f2)
    [[ $RC == 124 ]] && WHY="timeout after ${TIMEOUT_S}s" || WHY="exit code $RC"
    if [[ $RC == 0 && $FULL == 1 ]]; then
      log "PASS (full): ${CAND_FULL:0:12} score ${SCORE:-n/a}, ${ELAPSED}s, $RUN_DIR -> promoted to known-good"
      printf 'GOOD_SHA=%s\nGOOD_FALLBACK=%s\nGOOD_SCORE=%s\nGOOD_DATE=%s\n' "${CAND_FULL:0:12}" "${GOOD_FULL:0:12}" "${SCORE:-0}" "$(date +%F)" > "$STATE"
      note "PROMOTE label=$LABEL engine=${CAND_FULL:0:12} score=${SCORE:-n/a} run=$RUN_DIR"
      GOOD_SHA=${CAND_FULL:0:12}; GOOD_FULL=$CAND_FULL
      ensure_good "promoted ${CAND_FULL:0:12}"
    elif [[ $RC == 0 ]]; then
      log "PASS (screen $PHASES): ${CAND_FULL:0:12}, ${ELAPSED}s, $RUN_DIR (not promoted)"
      note "SCREEN-PASS label=$LABEL engine=${CAND_FULL:0:12} phases=$PHASES run=$RUN_DIR"
      ensure_good "after screen $LABEL"
    else
      log "FAIL: ${CAND_FULL:0:12} ($WHY, ${ELAPSED}s, .work/safeloop/${LABEL}.log, $RUN_DIR)"
      note "FAIL label=$LABEL engine=${CAND_FULL:0:12} why=$WHY run=${RUN_DIR:-none}"
      ensure_good "candidate $LABEL failed: $WHY"
      exit 1
    fi
    ;;

  *)
    echo "usage: safeloop.sh run LABEL SHA [PHASES] [TIMEOUT_S] | cmd TIMEOUT_S 'command' | restore" >&2; exit 2
    ;;
esac
