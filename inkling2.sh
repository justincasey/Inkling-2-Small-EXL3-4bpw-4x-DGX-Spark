#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# inkling2.sh: Inkling 2 Small (0xSero EXL3 4.0 bpw pack) on four DGX Sparks, TP4 on the REXL3 engine.
# Run from any control machine with key-based ssh to the four nodes.
#
# The pack is staged split: rank r holds only its routed experts (E mod 4 == r, files model-experts-LL-0r) plus every
# shared file (non-expert shards, drafter, config, tokenizer), about 91 GB per node. config/pack-ranks.tsv is the
# assignment (scripts/pack_ranks.py checks it against the files' own headers); intake/manifest.tsv holds the sha256s.
#
# Usage: ./inkling2.sh <command> [args]
#   doctor         read-only: GPU idle and alike on every rank, memory, foreign containers, staged pack files,
#                  fabric reachability between all ranks, pinned NCCL library, installed engine commit
#   fetch          each rank downloads its own expert files; rank 0 downloads the shared files and sends them to
#                  ranks 1-3 over the fabric; then verify
#   verify         sha256 of every file each rank holds, all four ranks in parallel (no download)
#   build [REV]    build rexl3-serve + rexl3-cluster on rank 0 from the local engine repository (REXL3_SRC) at
#                  REXL3_COMMIT (config/pins.sh) or the lab revision REV, install under BASE_DIR/bin and ship the
#                  binaries to ranks 1-3 over the fabric; every rank must report the same sha256 and commit
#   files RANK     print the pack paths RANK holds
#   print-serve [RANK]
#                  print the exact command line (environment included) each rank runs; `start` runs exactly this
#   start          preflight (the doctor's gates, plus an installed engine), then ranks 3, 2, 1 and 0 detached, each
#                  under its memory guard (scripts/memguard.sh); waits up to READY_TIMEOUT_S for /health (200: engine ready) on rank 0
#                  while watching every rank for `level=fatal` or death (then stops all and prints the log tails);
#                  prints the capacity line and boot_s. DUMP_DIR, EXTRA_FLAGS and EXTRA_ENV reach every rank alike
#   stop           rank 0 first, then ranks 1-3 (SIGTERM, STOP_GRACE_S, SIGKILL), then the guards; checks the GPUs idle
#   status         per rank: pid, uptime, guard, MemAvailable now and the guard's minimum, GPU, last log line;
#                  rank 0's /v1/models
#   logs [RANK] [N]
#                  last N (default 40) lines of a rank's log, every rank if RANK is omitted or `all`
#   smoke          one chat and one raw completion against rank 0 (bench/tools/smoke.py); fails on non-200 or empty
#   memwatch begin|report
#                  begin: open a measurement window on every guard; report: per rank MemAvailable minimum in the
#                  window, while loading, once ready, overall (the guards' records)
#   record NAME    create bench/records/<date>-NAME/meta.json: engine commit and sha256 from the installed binaries,
#                  pack revision, pins, configuration and every rank's print-serve line
#   help
#
# Configuration: environment > .env > defaults below. Every key is documented in .env.example.
set -euo pipefail
if (( BASH_VERSINFO[0] < 4 )); then echo "inkling2.sh needs bash >= 4 (macOS: brew install bash)" >&2; exit 1; fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ── configuration: caller env > .env > defaults ────────────────────────────────────────────────
declare -A _CALLER=()
while IFS= read -r _k; do _CALLER[$_k]=1; done < <(compgen -e)
if [[ -f $ROOT/.env ]]; then
  while IFS= read -r _line || [[ -n $_line ]]; do
    [[ $_line =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _k=${BASH_REMATCH[1]} _v=${BASH_REMATCH[2]}
    [[ -n ${_CALLER[$_k]:-} ]] && continue
    _v=${_v#"${_v%%[![:space:]]*}"}
    if [[ $_v =~ ^\"(.*)\"[[:space:]]*(#.*)?$ || $_v =~ ^\'(.*)\'[[:space:]]*(#.*)?$ ]]; then _v=${BASH_REMATCH[1]}
    else _v=${_v%%[[:space:]]#*}; [[ $_v == \#* ]] && _v=; _v=${_v%"${_v##*[![:space:]]}"}; fi
    printf -v "$_k" '%s' "$_v"; export "${_k?}"
  done < "$ROOT/.env"
fi
unset _CALLER _k _v _line
# shellcheck source=config/pins.sh
source "$ROOT/config/pins.sh"

: "${NODES:=192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18}"            # management addresses, rank 0 first
: "${FABRIC_ADDRS:=10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.4}"            # each rank's rail-0 address on the switch
: "${SSH_USER:=spark}"
: "${MODELS_DIR:=/home/spark/models}"
: "${HF_BIN:=/home/spark/venvs/hf/bin/hf}"
: "${BASE_DIR:=/home/spark/inkling-2-small-exl3-4bpw-4x-dgx-spark}"           # engine build, binaries, logs on each node
: "${REXL3_SRC:=$HOME/projects/REXL3}"                                          # local engine git repository (this machine)
: "${ALLOWED_CONTAINERS:=mentatd glm53-nfs}"                                    # containers that may run beside the ranks
: "${SERVED_NAME:=inkling-2-small-exl3-4bpw-4x-dgx-spark}"
: "${PORT:=8000}"                                                               # rank 0's HTTP API
: "${API_KEY:=}"                                                                # sent as a bearer token by the tools when set
: "${TP_PORT:=29571}"                                                           # rank 0's TP rendezvous on its rail-0 address
: "${MAX_MODEL_LEN:=1048576}"
: "${SLOTS:=4}"
: "${PREFILL_CHUNK:=4096}"
: "${EXTRA_FLAGS:=}"                                                            # more rexl3-serve flags, every rank alike
: "${EXTRA_ENV:=}"                                                              # more KEY=VALUE words, every rank alike
: "${DUMP_DIR:=}"                                                               # absolute: REXL3_INKLING2_DUMP on every rank
: "${READY_TIMEOUT_S:=1800}"                                                    # start: wait this long for /health
: "${STOP_GRACE_S:=60}"                                                         # stop: SIGTERM, then SIGKILL after this
: "${MIN_AVAIL_GIB:=100}"                                                       # doctor/start: MemAvailable before a boot
: "${GUARD_START_MIB:=2048}"                                                    # guard floor while the pack loads
: "${GUARD_MIB:=1024}"                                                          # guard floor once ready
: "${GUARD_SWAP_MAX_MIB:=64}"                                                   # guard: swap growth that kills the rank...
: "${GUARD_SWAP_AVAIL_MIB:=16384}"                                              # ...while MemAvailable is under this
: "${DROP_CACHES_EVERY_S:=20}"                                                  # guard: page-cache drop while loading
RANKS_FILE=config/pack-ranks.tsv

read -r -a NODE <<<"$NODES"
read -r -a FABRIC <<<"$FABRIC_ADDRS"
(( ${#NODE[@]} == 4 )) || { echo "NODES must list exactly four management addresses (got ${#NODE[@]})" >&2; exit 1; }
(( ${#FABRIC[@]} == 4 )) || { echo "FABRIC_ADDRS must list exactly four addresses (got ${#FABRIC[@]})" >&2; exit 1; }
for _k in PORT TP_PORT MAX_MODEL_LEN SLOTS PREFILL_CHUNK READY_TIMEOUT_S STOP_GRACE_S MIN_AVAIL_GIB GUARD_START_MIB GUARD_MIB \
  GUARD_SWAP_MAX_MIB GUARD_SWAP_AVAIL_MIB DROP_CACHES_EVERY_S; do
  [[ ${!_k} =~ ^[0-9]+$ ]] || { echo "$_k must be a whole number (got '${!_k}')" >&2; exit 1; }
done
[[ -z $DUMP_DIR || $DUMP_DIR == /* ]] || { echo "DUMP_DIR must be an absolute path on the nodes (got '$DUMP_DIR')" >&2; exit 1; }
for _k in $EXTRA_ENV; do
  [[ $_k =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "EXTRA_ENV takes KEY=VALUE words (got '$_k')" >&2; exit 1; }
done
unset _k
MODEL_DIR=$MODELS_DIR/$MODEL_NAME
BIN=$BASE_DIR/bin/rexl3-serve
API=http://${NODE[0]}:$PORT
# One ssh connection per node, reused by the start/status polling loops.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new
  -o ControlMaster=auto -o "ControlPath=/tmp/inkling2-ssh-%C" -o ControlPersist=60)

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

# remote RANK SCRIPT [VAR=value ...]: run SCRIPT under `bash -s` on one rank with the variables exported. The script
# travels on stdin as one brace group with stdin from /dev/null, so no child that reads stdin can eat the rest of it.
remote() {
  local r=$1 script=$2 kv prelude="set -euo pipefail"$'\n'; shift 2
  for kv in "$@"; do prelude+="export ${kv%%=*}=$(printf '%q' "${kv#*=}")"$'\n'; done
  ssh "${SSH_OPTS[@]}" "$SSH_USER@${NODE[$r]}" bash -s <<<"$prelude{"$'\n'"$script"$'\n'"} </dev/null"
}
# wait_all MSG: wait on the jobs in pids, die with MSG if any failed (explicit PIDs: an exited job leaves the table).
pids=()
wait_all() { local p rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; pids=(); (( rc == 0 )) || die "$1"; }

# rank_files RANK [shared|experts|all]: the pack paths RANK holds, from config/pack-ranks.tsv.
rank_files() {
  local r=$1 which=${2:-all}
  [[ -f $ROOT/$RANKS_FILE ]] || die "missing $RANKS_FILE (scripts/pack_ranks.py)"
  awk -F'\t' -v r="$r" -v w="$which" '
    ($1 == "*" && w != "experts") || ($1 == r && w != "shared") {print $3}' "$ROOT/$RANKS_FILE"
}
# rank_manifest RANK: `sha256  path` lines for the files RANK holds (sha256sum -c format).
rank_manifest() {
  awk -F'\t' 'NR == FNR {want[$1] = 1; next} ($3 in want) {print $1 "  " $3}' \
    <(rank_files "$1") "$ROOT/$MANIFEST_FILE"
}

cmd_files() {
  local r=${1:-}; [[ $r =~ ^[0-3]$ ]] || die "usage: files RANK (0-3)"
  rank_files "$r"
}

verify_rank() {   # verify_rank RANK: sha256 of every file the rank holds, 12 at a time
  local r=$1 manifest; manifest=$(rank_manifest "$r")
  (( $(wc -l <<<"$manifest") == $(rank_files "$r" | wc -l) )) || die "rank $r: $MANIFEST_FILE lacks some of its files"
  remote "$r" '
    cd "$MODEL_DIR" 2>/dev/null || { echo "rank $RANK: $MODEL_DIR missing" >&2; exit 1; }
    want=$(mktemp); trap "rm -f $want" EXIT; printf "%s\n" "$MANIFEST" > "$want"
    if awk "{print \$2}" "$want" | xargs -P 12 -n 1 sha256sum 2>/dev/null | sort -k2 | diff -q - <(sort -k2 "$want") >/dev/null
    then echo "rank $RANK: $(wc -l < "$want") files verified"
    else echo "rank $RANK: sha256 mismatch or missing file under $MODEL_DIR" >&2; exit 1; fi' \
    MODEL_DIR="$MODEL_DIR" RANK="$r" MANIFEST="$manifest"
}
cmd_verify() {
  local r rc=0; for r in 0 1 2 3; do verify_rank "$r" & pids+=($!); done
  for r in "${pids[@]}"; do wait "$r" || rc=1; done; pids=(); return $rc
}

download() {   # download RANK FILES: hf download FILES at the pinned revision into MODEL_DIR on RANK
  local r=$1 files=$2
  log "rank $r: download $(wc -w <<<"$files") files"
  remote "$r" 'mkdir -p "$MODEL_DIR"
    HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 "$HF_BIN" download "$REPO" --revision "$REV" --local-dir "$MODEL_DIR" $FILES >/dev/null
    echo "rank $RANK: downloaded"' \
    MODEL_DIR="$MODEL_DIR" HF_BIN="$HF_BIN" REPO="$MODEL_REPO" REV="$MODEL_REV" FILES="$files" RANK="$r"
}
send_shared() {   # send_shared TO: rsync rank 0's shared files to rank TO over its fabric address
  local to=$1
  log "rank 0 -> rank $to over ${FABRIC[$to]}: shared files"
  remote 0 '
    ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER@$IP" mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR" && printf "%s\n" $FILES | rsync -a --whole-file --files-from=- . "$USER@$IP:$MODEL_DIR/"' \
    USER="$SSH_USER" IP="${FABRIC[$to]}" MODEL_DIR="$MODEL_DIR" FILES="$(rank_files 0 shared | tr '\n' ' ')"
}

cmd_fetch() {
  [[ $MODEL_REV =~ ^[0-9a-f]{40}$ ]] || die "MODEL_REV must be a full commit sha (config/pins.sh)"
  local r
  for r in 0 1 2 3; do
    remote "$r" 'ping -c 1 -W 2 "$IP" >/dev/null' IP="${FABRIC[$r]}" || die "rank $r: fabric address ${FABRIC[$r]} unreachable from itself"
  done
  log "$MODEL_REPO@$MODEL_REV -> $MODEL_DIR: each rank its experts, rank 0 the shared files"
  download 0 "$(rank_files 0 | tr '\n' ' ')" & pids+=($!)
  for r in 1 2 3; do download "$r" "$(rank_files "$r" experts | tr '\n' ' ')" & pids+=($!); done
  wait_all "a rank's download failed"
  for r in 1 2 3; do send_shared "$r" & pids+=($!); done
  wait_all "shared-file fan-out failed"
  cmd_verify || die "checksum verification failed"
  log "fetch: every rank holds and has verified its files"
}

# every SCRIPT [VAR=value ...]: run on all ranks in parallel, with RANK set to the rank; output printed per rank in
# order, prefixed [rN]; fails if any rank fails.
every() {
  local r rc=0 tmp; tmp=$(mktemp -d)
  local -a ep=()
  for r in 0 1 2 3; do ( remote "$r" "$@" RANK="$r" >"$tmp/$r" 2>&1 && echo 0 >"$tmp/$r.rc" || echo $? >"$tmp/$r.rc" ) & ep+=($!); done
  wait "${ep[@]}" || true
  for r in 0 1 2 3; do sed "s/^/[r$r] /" "$tmp/$r"; [[ $(<"$tmp/$r.rc") == 0 ]] || rc=1; done
  rm -rf "$tmp"
  return $rc
}

# cmd_doctor [start]: read-only. Each rank checks itself; a FAIL line fails the doctor. Driver and engine lines are then
# compared across ranks. `start` (the start preflight) also requires an installed engine.
cmd_doctor() {
  local mode=${1:-} r out bad=0 tmp; tmp=$(mktemp -d)
  for r in 0 1 2 3; do
    remote "$r" '
      fail() { echo "FAIL $*"; }
      gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)
      [[ $gpu == *GB10* ]] && echo "driver ${gpu##*, }" || fail "no NVIDIA GB10 (nvidia-smi: ${gpu:-none})"
      busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
      if [[ -z $busy ]]; then echo "ok   gpu idle"
      else
        fail "GPU in use:"
        while IFS=", " read -r p _; do
          echo "     $(ps -o pid=,user=,etime=,args= -p "$p" 2>/dev/null | cut -c1-160 || echo "$p (gone)")"
        done <<<"$busy"
      fi
      # the lab GPU lock (another agent tests the engine under `flock /tmp/rexl3-gpu.lock`): free, or name its holders
      if [[ -e $GPU_LOCK ]] && ! flock -n -x "$GPU_LOCK" true 2>/dev/null; then
        fail "$GPU_LOCK is held:"
        pgrep -a -f "[f]lock .*$GPU_LOCK" | cut -c1-160 | sed "s/^/     /" || true
      else echo "ok   $GPU_LOCK free"; fi
      ours=$(pgrep -f "^$BASE_DIR/bin/[r]exl3-serve( |$)" | paste -sd" " - || true)
      [[ -z $ours ]] && echo "ok   no rexl3-serve of this recipe running" || fail "rexl3-serve already running here (pid $ours): ./inkling2.sh stop"
      avail=$(awk "/MemAvailable/ {printf \"%d\", \$2/1048576}" /proc/meminfo)
      (( avail >= MIN_AVAIL )) && echo "ok   MemAvailable ${avail} GiB" || fail "MemAvailable ${avail} GiB < $MIN_AVAIL"
      for c in $(docker ps --format "{{.Names}}" 2>/dev/null); do
        [[ " $ALLOWED " == *" $c "* ]] || fail "foreign container running: $c"
      done
      missing=0
      while read -r size path; do
        [[ $(stat -c %s "$MODEL_DIR/$path" 2>/dev/null) == "$size" ]] || missing=$((missing + 1))
      done <<<"$FILES"
      (( missing == 0 )) && echo "ok   pack: $(wc -l <<<"$FILES") files at full size under $MODEL_DIR (verify checks bytes)" \
        || fail "pack: $missing files missing or short under $MODEL_DIR (./inkling2.sh fetch)"
      for ip in $PEERS; do ping -c 1 -W 2 "$ip" >/dev/null || fail "fabric: $ip unreachable"; done
      echo "ok   fabric: $PEERS reachable"
      [[ $(sha256sum "$NCCL_LIB" 2>/dev/null | cut -d" " -f1) == "$NCCL_SHA256" ]] && echo "ok   nccl: $NCCL_LIB (pinned sha256)" \
        || fail "nccl: $NCCL_LIB missing or not the pinned build (config/pins.sh NCCL_SHA256)"
      # an NCCL config file key the serve environment does not override would make this rank differ unseen by the boot gate
      for f in /etc/nccl.conf "$HOME/.nccl.conf"; do
        [[ -f $f ]] || continue
        for k in $(sed -n "s/^[[:space:]]*\([A-Z_][A-Z0-9_]*\)=.*/\1/p" "$f"); do
          [[ " $NCCL_KEYS " == *" $k "* ]] || fail "nccl: $f sets $k, which the serve environment does not override (config/pins.sh NCCL_ENV)"
        done
      done
      if [[ -x $BASE_DIR/bin/rexl3-serve ]]; then
        commit=$("$BASE_DIR/bin/rexl3-serve" --version 2>/dev/null | awk "\$1 == \"commit\" {print \$2}")
        [[ $commit == "$(cat "$BASE_DIR/bin/rexl3-serve.commit" 2>/dev/null)" ]] \
          || fail "engine: --version reports commit ${commit:-none}, rexl3-serve.commit says otherwise"
        echo "engine ${commit:-unknown} $(sha256sum "$BASE_DIR/bin/rexl3-serve" | cut -d" " -f1)"
      else echo "engine none"; fi' \
      MODEL_DIR="$MODEL_DIR" BASE_DIR="$BASE_DIR" ALLOWED="$ALLOWED_CONTAINERS" PEERS="${FABRIC[*]}" NCCL_LIB="$NCCL_LIB" NCCL_SHA256="$NCCL_SHA256" \
      MIN_AVAIL="$MIN_AVAIL_GIB" GPU_LOCK=/tmp/rexl3-gpu.lock NCCL_KEYS="$(printf '%s\n' "${NCCL_ENV[@]}" | cut -d= -f1 | paste -sd' ' -)" \
      FILES="$(awk -F'\t' -v r="$r" '$1 == "*" || $1 == r {print $2, $3}' "$ROOT/$RANKS_FILE")" \
      >"$tmp/$r" 2>&1 & pids+=($!)
  done
  wait_all "ssh to a rank failed"
  out=$(for r in 0 1 2 3; do sed "s/^/[r$r] /" "$tmp/$r"; done); rm -rf "$tmp"
  echo "$out"
  grep -q '\] FAIL' <<<"$out" && bad=1
  (( $(grep '\] driver ' <<<"$out" | awk '{print $3}' | sort -u | wc -l) == 1 )) || { echo "FAIL driver versions differ across ranks"; bad=1; }
  (( $(grep '\] engine ' <<<"$out" | awk '{print $3, $4}' | sort -u | wc -l) == 1 )) || { echo "FAIL engine binaries differ across ranks (./inkling2.sh build)"; bad=1; }
  if [[ $mode == start ]] && grep -q '\] engine none' <<<"$out"; then echo "FAIL no engine installed (./inkling2.sh build)"; bad=1; fi
  grep -q "\] engine $REXL3_COMMIT " <<<"$out" || echo "note installed engine is not REXL3_COMMIT ${REXL3_COMMIT:0:12} (lab build or none)"
  (( bad == 0 )) && log "doctor: all four ranks ready" || die "doctor found problems"
}

cmd_build() {
  local rev=${1:-$REXL3_COMMIT} sha
  sha=$(git -C "$REXL3_SRC" rev-parse --verify "$rev^{commit}" 2>/dev/null) || die "$rev is not a commit in $REXL3_SRC"
  git -C "$REXL3_SRC" merge-base --is-ancestor "$REXL3_UPSTREAM_BASE" "$sha" \
    || die "$sha does not descend from the upstream base ${REXL3_UPSTREAM_BASE:0:12}"
  log "build engine ${sha:0:12} on rank 0 (${NODE[0]})"
  git -C "$REXL3_SRC" archive --format=tar "$sha" | ssh "${SSH_OPTS[@]}" "$SSH_USER@${NODE[0]}" \
    "rm -rf '$BASE_DIR/src.new' && mkdir -p '$BASE_DIR/src.new' && tar -x -C '$BASE_DIR/src.new'"
  remote 0 '
    pgrep -f "[r]exl3-serve" >/dev/null && { echo "FAIL rexl3-serve is running on rank 0: stop it first" >&2; exit 1; }
    rm -rf "$BASE_DIR/src" && mv "$BASE_DIR/src.new" "$BASE_DIR/src" && mkdir -p "$BASE_DIR/bin" "$BASE_DIR/logs"
    cd "$BASE_DIR/src"
    export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH CUDA_HOME=/usr/local/cuda REXL3_CUDA_ARCH=sm_121 \
      REXL3_COMMIT="$SHA" CARGO_TARGET_DIR="$BASE_DIR/target"
    cargo +"$TOOLCHAIN" build --release --locked -p gb-server --bin rexl3-serve -p gb-cluster --bin rexl3-cluster \
      > "$BASE_DIR/logs/build.log" 2>&1 \
      || { grep -E "^error" -A6 "$BASE_DIR/logs/build.log" | head -60; echo "FAIL build (log: $BASE_DIR/logs/build.log)" >&2; exit 1; }
    v=$("$BASE_DIR/target/release/rexl3-serve" --version 2>/dev/null)
    grep -q "^commit *$SHA$" <<<"$v" || { echo "FAIL the built binary reports another commit:" >&2; echo "$v" >&2; exit 1; }
    grep -Eq "^cuda_arch *sm_121$" <<<"$v" || { echo "FAIL the built binary is not sm_121:" >&2; echo "$v" >&2; exit 1; }
    for b in rexl3-serve rexl3-cluster; do cp "$BASE_DIR/target/release/$b" "$BASE_DIR/bin/$b.new" && mv -f "$BASE_DIR/bin/$b.new" "$BASE_DIR/bin/$b"; done
    echo "$SHA" > "$BASE_DIR/bin/rexl3-serve.commit"
    echo "rank 0: built $(sha256sum "$BASE_DIR/bin/rexl3-serve" | cut -c1-12)"' \
    BASE_DIR="$BASE_DIR" SHA="$sha" TOOLCHAIN="$RUST_TOOLCHAIN" || die "build failed on rank 0"
  local r
  for r in 1 2 3; do
    remote 0 '
      ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER@$IP" "pgrep -f \"[r]exl3-serve\" >/dev/null && exit 1; mkdir -p $BASE_DIR/bin $BASE_DIR/logs" \
        || { echo "FAIL rexl3-serve is running on $IP: stop it first" >&2; exit 1; }
      rsync -a "$BASE_DIR/bin/" "$USER@$IP:$BASE_DIR/bin/"' USER="$SSH_USER" IP="${FABRIC[$r]}" BASE_DIR="$BASE_DIR" & pids+=($!)
  done
  wait_all "shipping the engine failed"
  local sums; sums=$(every 'echo "$(cat "$BASE_DIR/bin/rexl3-serve.commit") $(sha256sum "$BASE_DIR/bin/rexl3-serve" "$BASE_DIR/bin/rexl3-cluster" | cut -c1-64 | paste -sd" " -)"' BASE_DIR="$BASE_DIR") \
    || die "could not read the installed binaries"
  echo "$sums"
  (( $(sed 's/^\[r.\] //' <<<"$sums" | sort -u | wc -l) == 1 )) || die "the four ranks hold different binaries"
  log "build: engine ${sha:0:12} installed on all four ranks"
}

# ── serving ─────────────────────────────────────────────────────────────────────────────────────
# serve_cmd RANK: the one command line RANK runs, environment included. `env -i`: nothing from the ssh session reaches
# the rank, so its NCCL_* set is exactly config/pins.sh NCCL_ENV (+ EXTRA_ENV) on every rank. print-serve prints this
# string and start hands this very string to the rank. Flags are identical on every rank but --tp-rank; rank 0 alone
# serves HTTP, the followers take the same --host/--port (contract with the engine).
serve_cmd() {
  local r=$1 w
  local -a extra_env extra_flags env args
  read -r -a extra_env <<<"$EXTRA_ENV"
  read -r -a extra_flags <<<"$EXTRA_FLAGS"
  env=(HOME="/home/$SSH_USER" USER="$SSH_USER" LOGNAME="$SSH_USER"
    PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    "${NCCL_ENV[@]}" REXL3_NCCL_LIB="$NCCL_LIB" RUST_BACKTRACE=1)
  [[ -z $DUMP_DIR ]] || env+=(REXL3_INKLING2_DUMP="$DUMP_DIR")
  env+=("${extra_env[@]}")
  args=(--model "$MODEL_DIR" --tp-ranks 4 --tp-rank "$r" --tp-peer "${FABRIC[0]}:$TP_PORT"
    --max-model-len "$MAX_MODEL_LEN" --slots "$SLOTS" --prefill-chunk "$PREFILL_CHUNK" --kv-dtype int8 --spec none
    --no-vision --allow-unvalidated-pack --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT"
    "${extra_flags[@]}")
  printf 'exec env -i'
  for w in "${env[@]}" "$BIN" "${args[@]}"; do printf ' %q' "$w"; done
  printf '\n'
}

cmd_print_serve() {
  local r ranks=(0 1 2 3)
  if [[ -n ${1:-} ]]; then [[ $1 =~ ^[0-3]$ ]] || die "usage: print-serve [RANK 0-3]"; ranks=("$1"); fi
  for r in "${ranks[@]}"; do
    printf '# rank %s  %s@%s  rail-0 %s' "$r" "$SSH_USER" "${NODE[$r]}" "${FABRIC[$r]}"
    (( r == 0 )) && printf '  TP rendezvous %s:%s, HTTP :%s' "${FABRIC[0]}" "$TP_PORT" "$PORT"
    printf '\n'
    serve_cmd "$r"
  done
}

AUTH=(); [[ -z $API_KEY ]] || AUTH=(-H "Authorization: Bearer $API_KEY")
# ready = /health answers 200: the HTTP front end listens (and /v1/models answers) from the first second, the engine only once
# the group is loaded; /health is 503 until then
api_up() { curl -fsS -m 5 -o /dev/null "${AUTH[@]}" "$API/health" 2>/dev/null; }

# launch_rank RANK: the guard first (it validates itself, then waits for the rank's pid file), then the rank, detached
# (setsid nohup; the previous log becomes .prev); fails, with the rank killed, unless the guard armed on the rank's pid.
launch_rank() {
  local r=$1
  remote "$r" '
    mkdir -p "$BASE_DIR/logs" "$BASE_DIR/run"
    log=$BASE_DIR/logs/rank$RANK.log glog=$BASE_DIR/logs/memguard-rank$RANK.log gmin=$BASE_DIR/logs/memguard-rank$RANK.min
    pidf=$BASE_DIR/run/rank$RANK.pid gpidf=$BASE_DIR/run/memguard-rank$RANK.pid stage=$BASE_DIR/run/stage-rank$RANK
    if [[ -s $pidf ]] && kill -0 "$(<"$pidf")" 2>/dev/null; then echo "FAIL rank $RANK: already running (pid $(<"$pidf"))" >&2; exit 1; fi
    if [[ -n $DUMP ]]; then
      mkdir -p "$DUMP/rank$RANK"
      if [[ -n $(ls -A "$DUMP/rank$RANK") ]]; then
        echo "FAIL rank $RANK: $DUMP/rank$RANK is not empty (chunk numbers of two boots would mix)" >&2; exit 1
      fi
    fi
    for f in "$log" "$glog" "$gmin"; do if [[ -f $f ]]; then mv -f "$f" "$f.prev"; fi; done
    rm -f "$pidf" "$gpidf"
    printf "%s\n" "$GUARD_SRC" > "$BASE_DIR/run/memguard.sh"
    echo loading > "$stage"
    GUARD_PIDFILE=$pidf GUARD_RANK=$RANK GUARD_LOG=$glog GUARD_MIN_FILE=$gmin GUARD_STAGE_FILE=$stage \
    GUARD_WINDOW_FILE=$BASE_DIR/run/memwatch.window GUARD_SERVE_LOG=$log GUARD_START_MIB=$G_START GUARD_MIB=$G_MIB \
    GUARD_SWAP_MAX_MIB=$G_SWAP GUARD_SWAP_AVAIL_MIB=$G_SWAP_AVAIL DROP_CACHES_EVERY_S=$G_DROP \
      setsid nohup bash "$BASE_DIR/run/memguard.sh" >> "$glog" 2>&1 < /dev/null &
    gpid=$!
    echo "$gpid" > "$gpidf"
    for _ in $(seq 120); do   # 30 s: the first page-cache drop of the guard includes a sync
      if grep -q " waiting for " "$glog" 2>/dev/null || ! kill -0 "$gpid" 2>/dev/null; then break; fi
      sleep 0.25
    done
    if ! kill -0 "$gpid" 2>/dev/null || ! grep -q " waiting for " "$glog"; then
      echo "FAIL rank $RANK: the memory guard did not start, so the rank is not started:" >&2; cat "$glog" >&2; exit 1
    fi
    cd "$BASE_DIR"
    # the lab GPU lock, held exclusively for the rank'"'"'s whole life: fd 9 survives the exec into rexl3-serve and closes with it
    setsid nohup bash -c "exec 9>>/tmp/rexl3-gpu.lock; flock -x -n 9 || { echo \"FAIL rank $RANK: /tmp/rexl3-gpu.lock is held by another GPU job\"; exit 75; }; $CMD" \
      > "$log" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$pidf"
    for _ in $(seq 40); do
      if grep -q " armed pid=$pid " "$glog" || ! kill -0 "$gpid" 2>/dev/null; then break; fi
      sleep 0.25
    done
    if ! grep -q " armed pid=$pid " "$glog" || ! kill -0 "$gpid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true; kill "$gpid" 2>/dev/null || true
      echo "FAIL rank $RANK: the memory guard did not arm on pid $pid; the rank is killed:" >&2; tail -n 5 "$glog" >&2; exit 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "FAIL rank $RANK: rexl3-serve exited at once:" >&2; tail -n 20 "$log" >&2; exit 1; fi
    echo "rank $RANK: pid $pid, guard $gpid, log $log"' \
    BASE_DIR="$BASE_DIR" RANK="$r" DUMP="$DUMP_DIR" CMD="$(serve_cmd "$r")" GUARD_SRC="$(<"$ROOT/scripts/memguard.sh")" \
    G_START="$GUARD_START_MIB" G_MIB="$GUARD_MIB" G_SWAP="$GUARD_SWAP_MAX_MIB" G_SWAP_AVAIL="$GUARD_SWAP_AVAIL_MIB" G_DROP="$DROP_CACHES_EVERY_S"
}

# Boot watch, per rank: the rank and its guard alive, no `level=fatal` line; prints the last event seen.
PROBE='
  log=$BASE_DIR/logs/rank$RANK.log rc=0
  pid=$(cat "$BASE_DIR/run/rank$RANK.pid" 2>/dev/null || true)
  gpid=$(cat "$BASE_DIR/run/memguard-rank$RANK.pid" 2>/dev/null || true)
  if [[ -z $pid ]] || ! kill -0 "$pid" 2>/dev/null; then echo "DEAD rexl3-serve (pid ${pid:-none})"; rc=1; fi
  if [[ -z $gpid ]] || ! kill -0 "$gpid" 2>/dev/null; then echo "DEAD memory guard (pid ${gpid:-none})"; rc=1; fi
  if grep -q "level=fatal" "$log" 2>/dev/null; then echo "FATAL $(grep -m1 "level=fatal" "$log" | cut -c1-300)"; rc=1; fi
  echo "last $(grep -o "event=[A-Za-z0-9_]*" "$log" 2>/dev/null | tail -n 1 || true)"
  exit $rc'

print_tails() {
  every 'echo "== $BASE_DIR/logs/rank$RANK.log"; tail -n 40 "$BASE_DIR/logs/rank$RANK.log" 2>/dev/null || echo "(no log)"
    echo "== memguard"; tail -n 6 "$BASE_DIR/logs/memguard-rank$RANK.log" 2>/dev/null || echo "(no guard log)"' BASE_DIR="$BASE_DIR" || true
}

cmd_start() {
  cmd_doctor start
  local r out t0 boot_s last=0
  t0=$(date +%s)
  # followers first: rank 0 waits for all three, the followers retry their connection to it
  for r in 3 2 1 0; do
    log "rank $r (${NODE[$r]}): launch"
    if ! launch_rank "$r"; then
      log "rank $r did not launch: stopping every rank"; print_tails; ( cmd_stop ) || true; die "start failed at rank $r"
    fi
  done
  log "waiting up to ${READY_TIMEOUT_S}s for $API/health"
  until api_up; do
    if ! out=$(every "$PROBE" BASE_DIR="$BASE_DIR"); then
      echo "$out"; print_tails; log "a rank failed during boot: stopping every rank"; ( cmd_stop ) || true
      die "boot failed after $(( $(date +%s) - t0 ))s"
    fi
    if (( $(date +%s) - t0 >= READY_TIMEOUT_S )); then
      echo "$out"; print_tails; log "not ready after ${READY_TIMEOUT_S}s: stopping every rank"; ( cmd_stop ) || true
      die "boot timed out (READY_TIMEOUT_S=$READY_TIMEOUT_S)"
    fi
    if (( $(date +%s) - last >= 30 )); then
      log "booting $(( $(date +%s) - t0 ))s: $(grep -o '^\[r.\] last .*' <<<"$out" | sed 's/ last / /' | paste -sd' ' -)"
      last=$(date +%s)
    fi
    sleep 5
  done
  boot_s=$(( $(date +%s) - t0 ))
  every 'echo ready > "$BASE_DIR/run/stage-rank$RANK"' BASE_DIR="$BASE_DIR" >/dev/null \
    || die "ready, but a guard could not be told (it keeps the loading floor GUARD_START_MIB)"
  every 'grep -h -E "event=(inkling2_plan|tp_group_ready|tp_follower_ready)" "$BASE_DIR/logs/rank$RANK.log" | tail -n 2 | cut -c1-400' \
    BASE_DIR="$BASE_DIR" || true
  log "start: ready at $API/v1 ($SERVED_NAME)"
  echo "boot_s=$boot_s"
}

# stop_rank RANK: SIGTERM the rank (pid file, and any rexl3-serve of this BASE_DIR), SIGKILL after STOP_GRACE_S, then
# its guard (after the rank: a guard exits by itself once its rank is gone).
stop_rank() {
  remote "$1" '
    live=""
    for p in $(cat "$BASE_DIR"/run/rank[0-3].pid 2>/dev/null || true) $(pgrep -f "^$BASE_DIR/bin/[r]exl3-serve( |$)" || true); do
      if kill -0 "$p" 2>/dev/null && [[ " $live " != *" $p "* ]]; then live+=" $p"; fi
    done
    how=none
    if [[ -n $live ]]; then
      kill -TERM $live 2>/dev/null || true; how=SIGTERM
      for (( i = 0; i < GRACE; i++ )); do
        still=""; for p in $live; do if kill -0 "$p" 2>/dev/null; then still+=" $p"; fi; done
        if [[ -z $still ]]; then break; fi
        sleep 1
      done
      still=""; for p in $live; do if kill -0 "$p" 2>/dev/null; then still+=" $p"; fi; done
      if [[ -n $still ]]; then kill -KILL $still 2>/dev/null || true; how="SIGKILL after ${GRACE}s"; sleep 1; fi
    fi
    for g in $(cat "$BASE_DIR"/run/memguard-rank[0-3].pid 2>/dev/null || true) $(pgrep -f "^bash $BASE_DIR/run/[m]emguard.sh" || true); do
      kill "$g" 2>/dev/null || true
    done
    rm -f "$BASE_DIR"/run/rank[0-3].pid "$BASE_DIR"/run/memguard-rank[0-3].pid "$BASE_DIR"/run/stage-rank[0-3]
    echo "stopped: ${how}${live:+ (pid$live)}"' BASE_DIR="$BASE_DIR" GRACE="$STOP_GRACE_S"
}

cmd_stop() {
  local r rc=0 out
  log "stop rank 0 (${NODE[0]}) first"
  stop_rank 0 2>&1 | sed 's/^/[r0] /' || rc=1
  for r in 1 2 3; do ( stop_rank "$r" 2>&1 | sed "s/^/[r$r] /" ) & pids+=($!); done
  for r in "${pids[@]}"; do wait "$r" || rc=1; done; pids=()
  (( rc == 0 )) || die "could not stop every rank (ssh)"
  for _ in $(seq 10); do   # the driver frees a dead process's memory within seconds
    out=$(every 'busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
      if [[ -z $busy ]]; then echo "gpu idle"; else echo "GPU in use: $busy"; exit 1; fi') && break
    sleep 2
  done
  echo "$out"
  grep -q 'GPU in use' <<<"$out" && die "a GPU is still in use after stop"
  log "stop: every rank stopped, every GPU idle"
}

cmd_status() {
  every '
    log=$BASE_DIR/logs/rank$RANK.log glog=$BASE_DIR/logs/memguard-rank$RANK.log gmin=$BASE_DIR/logs/memguard-rank$RANK.min
    pid=$(cat "$BASE_DIR/run/rank$RANK.pid" 2>/dev/null || true)
    gpid=$(cat "$BASE_DIR/run/memguard-rank$RANK.pid" 2>/dev/null || true)
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
      up=$(ps -o etimes= -p "$pid" | tr -d " ")
      printf "serve   pid %s, up %dh%02dm%02ds\n" "$pid" $((up / 3600)) $((up % 3600 / 60)) $((up % 60))
    else echo "serve   down${pid:+ (stale pid file: $pid)}"; fi
    if [[ -n $gpid ]] && kill -0 "$gpid" 2>/dev/null; then echo "guard   pid $gpid, $(cat "$BASE_DIR/run/stage-rank$RANK" 2>/dev/null || echo "?") floor"
    elif grep -q " KILL " "$glog" 2>/dev/null; then echo "guard   FIRED: $(grep " KILL " "$glog" | tail -n 1)"
    else echo "guard   off"; fi
    get() { sed -n "s/^$1=//p" "$gmin" 2>/dev/null; }
    avail=$(awk "/^MemAvailable:/ {printf \"%d\", \$2 / 1024}" /proc/meminfo)
    if [[ -f $gmin ]]; then
      echo "memory  MemAvailable $avail MiB now; guard min $(get min_mib) MiB at $(get min_at) (loading $(get loading_min_mib), ready $(get ready_min_mib), window $(get window_min_mib)); swap $(get swap_now_mib) MiB (base $(get swap_base_mib))"
    else echo "memory  MemAvailable $avail MiB now; no guard record"; fi
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed")
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | paste -sd";" - || true)
    echo "gpu     util/mem/temp/power: $gpu; processes: ${apps:-none}"
    echo "log     $(tail -n 1 "$log" 2>/dev/null | cut -c1-200)"' BASE_DIR="$BASE_DIR" || true
  local models
  if models=$(curl -fsS -m 5 "${AUTH[@]}" "$API/v1/models" 2>&1); then echo "rank 0 $API/v1/models: $models"
  else echo "rank 0 $API/v1/models: down ($models)"; return 1; fi
}

cmd_logs() {
  local which=${1:-all} n=${2:-40} r
  [[ $which == all || $which =~ ^[0-3]$ ]] || die "usage: logs [RANK|all] [N]"
  [[ $n =~ ^[0-9]+$ ]] || die "usage: logs [RANK|all] [N]"
  local -a ranks=(0 1 2 3); [[ $which == all ]] || ranks=("$which")
  for r in "${ranks[@]}"; do
    echo "== rank $r (${NODE[$r]}) $BASE_DIR/logs/rank$r.log"
    remote "$r" 'tail -n "$N" "$BASE_DIR/logs/rank$RANK.log" 2>/dev/null || echo "(no log)"
      if grep -q " KILL " "$BASE_DIR/logs/memguard-rank$RANK.log" 2>/dev/null; then
        echo "== the memory guard killed this rank:"; grep -A 10 " KILL " "$BASE_DIR/logs/memguard-rank$RANK.log" | tail -n 11
      fi' BASE_DIR="$BASE_DIR" RANK="$r" N="$n" || echo "(ssh failed)"
  done
}

cmd_smoke() { API_KEY="$API_KEY" python3 "$ROOT/bench/tools/smoke.py" --url "$API" --model "$SERVED_NAME"; }

# memwatch begin: a new window token on every rank (each guard restarts its window minimum when it sees it);
# memwatch report: every guard's record. Guards run only while their rank runs; the record stays after.
cmd_memwatch() {
  case ${1:-} in
    begin)
      local token; token="$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM"
      every 'mkdir -p "$BASE_DIR/run" && echo "$TOKEN" > "$BASE_DIR/run/memwatch.window"' BASE_DIR="$BASE_DIR" TOKEN="$token" \
        || die "could not open the window on every rank"
      echo "memwatch window=$token (memwatch report reads each rank's minimum since now)" ;;
    report)
      every '
        f=$BASE_DIR/logs/memguard-rank$RANK.min
        [[ -f $f ]] || { echo "no guard record ($f): the rank has not run under a guard"; exit 1; }
        get() { sed -n "s/^$1=//p" "$f"; }
        cur=$(cat "$BASE_DIR/run/memwatch.window" 2>/dev/null || true)
        w=$(get window_min_mib) wat=$(get window_min_at)
        if [[ -z $cur || $(get window) != "$cur" ]]; then w=n/a wat=n/a; fi
        gpid=$(cat "$BASE_DIR/run/memguard-rank$RANK.pid" 2>/dev/null || true)
        if [[ -n $gpid ]] && kill -0 "$gpid" 2>/dev/null; then g=running; else g=stopped; fi
        echo "memwatch rank=$RANK guard=$g window=${cur:-none} window_min_mib=$w window_min_at=$wat loading_min_mib=$(get loading_min_mib) ready_min_mib=$(get ready_min_mib) min_mib=$(get min_mib) min_at=$(get min_at) now_mib=$(get now_mib) updated=$(get updated)"' \
        BASE_DIR="$BASE_DIR" ;;
    *) die "usage: memwatch begin|report" ;;
  esac
}

cmd_record() {
  local name=${1:-} day dir vers serve
  [[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "usage: record NAME (letters, digits, . _ -)"
  day=$(date -u +%Y-%m-%d) dir=$ROOT/bench/records/$day-$name
  [[ ! -e $dir ]] || die "$dir exists"
  vers=$(every '[[ -x $BIN ]] || { echo "no engine at $BIN"; exit 1; }
    "$BIN" --version 2>/dev/null; echo "sha256 $(sha256sum "$BIN" | cut -d" " -f1)"' BIN="$BIN") \
    || { echo "$vers"; die "could not read the installed engine on every rank"; }
  serve=$(cmd_print_serve)
  REC_DIR="$dir" REC_NAME="$name" REC_DAY="$day" REC_VERSIONS="$vers" REC_SERVE="$serve" REC_NODES="$NODES" \
  REC_FABRIC="$FABRIC_ADDRS" REC_SSH_USER="$SSH_USER" REC_NCCL_ENV="$(printf '%s\n' "${NCCL_ENV[@]}")" \
  REC_GIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo none)" \
  REC_DIRTY="$(git -C "$ROOT" status --porcelain 2>/dev/null | grep -c . || true)" \
  REC_MODEL_REPO="$MODEL_REPO" REC_MODEL_REV="$MODEL_REV" REC_MODEL_DIR="$MODEL_DIR" REC_PINNED="$REXL3_COMMIT" \
  REC_UPSTREAM_BASE="$REXL3_UPSTREAM_BASE" REC_NCCL_LIB="$NCCL_LIB" REC_NCCL_SHA="$NCCL_SHA256" \
  REC_CONFIG="$(for k in PROFILE SERVED_NAME PORT TP_PORT MAX_MODEL_LEN SLOTS PREFILL_CHUNK EXTRA_FLAGS EXTRA_ENV DUMP_DIR \
    GUARD_START_MIB GUARD_MIB GUARD_SWAP_MAX_MIB GUARD_SWAP_AVAIL_MIB DROP_CACHES_EVERY_S; do printf '%s=%s\n' "$k" "${!k:-}"; done)" \
  python3 - <<'PY' || die "could not write the record"
import json, os, re
e = os.environ
ranks = {}
for line in e["REC_VERSIONS"].splitlines():
    m = re.match(r"\[r(\d)\] (\S+)\s+(.*)$", line)
    if m:
        ranks.setdefault(int(m.group(1)), {})[m.group(2)] = m.group(3).strip()
if sorted(ranks) != [0, 1, 2, 3]:
    raise SystemExit("record: not every rank reported its engine")
for key in ("commit", "sha256"):
    vals = {ranks[r].get(key) for r in ranks}
    if len(vals) != 1 or None in vals:
        raise SystemExit(f"record: the ranks' engine {key} differ: {sorted(map(str, vals))}")
serve, cur = [], None
for line in e["REC_SERVE"].splitlines():
    if line.startswith("# rank "):
        cur = int(line.split()[2])
    elif cur is not None:
        serve.append({"rank": cur, "host": e["REC_NODES"].split()[cur], "fabric": e["REC_FABRIC"].split()[cur],
                      "command": line})
v0 = ranks[0]
meta = {
    "name": e["REC_NAME"], "date": e["REC_DAY"],
    "recipe": {"commit": e["REC_GIT"], "uncommitted_files": int(e["REC_DIRTY"] or 0)},
    "pack": {"repo": e["REC_MODEL_REPO"], "revision": e["REC_MODEL_REV"], "dir": e["REC_MODEL_DIR"]},
    "engine": {"commit": v0["commit"], "sha256": v0["sha256"], "pinned_commit": e["REC_PINNED"],
               "lab_build": v0["commit"] != e["REC_PINNED"], "upstream_base": e["REC_UPSTREAM_BASE"],
               "version": {k: v for k, v in v0.items() if k != "sha256"}},
    "nccl": {"lib": e["REC_NCCL_LIB"], "sha256": e["REC_NCCL_SHA"], "env": e["REC_NCCL_ENV"].split("\n")},
    "config": dict(l.split("=", 1) for l in e["REC_CONFIG"].splitlines()),
    "ranks": serve,
    "results": [],
}
os.makedirs(e["REC_DIR"])
with open(os.path.join(e["REC_DIR"], "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
    f.write("\n")
print(os.path.join(e["REC_DIR"], "meta.json"))
PY
}

cmd_help() { sed -n '3,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

case ${1:-help} in
  doctor) cmd_doctor ;;
  fetch) cmd_fetch ;;
  build) shift; cmd_build "$@" ;;
  verify) cmd_verify ;;
  files) shift; cmd_files "$@" ;;
  print-serve) shift; cmd_print_serve "$@" ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs) shift; cmd_logs "$@" ;;
  smoke) cmd_smoke ;;
  memwatch) shift; cmd_memwatch "$@" ;;
  record) shift; cmd_record "$@" ;;
  help|-h|--help) cmd_help ;;
  *) cmd_help >&2; exit 2 ;;
esac
