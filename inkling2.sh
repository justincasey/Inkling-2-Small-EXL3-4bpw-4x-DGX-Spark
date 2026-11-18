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
RANKS_FILE=config/pack-ranks.tsv

read -r -a NODE <<<"$NODES"
read -r -a FABRIC <<<"$FABRIC_ADDRS"
(( ${#NODE[@]} == 4 )) || { echo "NODES must list exactly four management addresses (got ${#NODE[@]})" >&2; exit 1; }
(( ${#FABRIC[@]} == 4 )) || { echo "FABRIC_ADDRS must list exactly four addresses (got ${#FABRIC[@]})" >&2; exit 1; }
MODEL_DIR=$MODELS_DIR/$MODEL_NAME
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

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

# every SCRIPT [VAR=value ...]: run on all ranks in parallel; output printed per rank in order, prefixed [rN];
# fails if any rank fails.
every() {
  local r rc=0 tmp; tmp=$(mktemp -d)
  local -a ep=()
  for r in 0 1 2 3; do ( remote "$r" "$@" >"$tmp/$r" 2>&1 && echo 0 >"$tmp/$r.rc" || echo $? >"$tmp/$r.rc" ) & ep+=($!); done
  wait "${ep[@]}" || true
  for r in 0 1 2 3; do sed "s/^/[r$r] /" "$tmp/$r"; [[ $(<"$tmp/$r.rc") == 0 ]] || rc=1; done
  rm -rf "$tmp"
  return $rc
}

cmd_doctor() {
  local r out bad=0 tmp; tmp=$(mktemp -d)
  # Each rank checks itself; a FAIL line fails the doctor. Driver and engine lines are then compared across ranks.
  for r in 0 1 2 3; do
    remote "$r" '
      fail() { echo "FAIL $*"; }
      gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)
      [[ $gpu == *GB10* ]] && echo "driver ${gpu##*, }" || fail "no NVIDIA GB10 (nvidia-smi: ${gpu:-none})"
      busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
      [[ -z $busy ]] && echo "ok   gpu idle" || fail "GPU in use: $busy"
      avail=$(awk "/MemAvailable/ {printf \"%d\", \$2/1048576}" /proc/meminfo)
      (( avail >= 100 )) && echo "ok   MemAvailable ${avail} GiB" || fail "MemAvailable ${avail} GiB < 100"
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
      if [[ -x $BASE_DIR/bin/rexl3-serve ]]; then
        echo "engine $(cat "$BASE_DIR/bin/rexl3-serve.commit" 2>/dev/null) $(sha256sum "$BASE_DIR/bin/rexl3-serve" | cut -c1-12)"
      else echo "engine none"; fi' \
      MODEL_DIR="$MODEL_DIR" BASE_DIR="$BASE_DIR" ALLOWED="$ALLOWED_CONTAINERS" PEERS="${FABRIC[*]}" NCCL_LIB="$NCCL_LIB" NCCL_SHA256="$NCCL_SHA256" \
      FILES="$(awk -F'\t' -v r="$r" '$1 == "*" || $1 == r {print $2, $3}' "$ROOT/$RANKS_FILE")" \
      >"$tmp/$r" 2>&1 & pids+=($!)
  done
  wait_all "ssh to a rank failed"
  out=$(for r in 0 1 2 3; do sed "s/^/[r$r] /" "$tmp/$r"; done); rm -rf "$tmp"
  echo "$out"
  grep -q '\] FAIL' <<<"$out" && bad=1
  (( $(grep '\] driver ' <<<"$out" | awk '{print $3}' | sort -u | wc -l) == 1 )) || { echo "FAIL driver versions differ across ranks"; bad=1; }
  (( $(grep '\] engine ' <<<"$out" | awk '{print $3, $4}' | sort -u | wc -l) == 1 )) || { echo "FAIL engine binaries differ across ranks (./inkling2.sh build)"; bad=1; }
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

cmd_help() { sed -n '3,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

case ${1:-help} in
  doctor) cmd_doctor ;;
  fetch) cmd_fetch ;;
  build) shift; cmd_build "$@" ;;
  verify) cmd_verify ;;
  files) shift; cmd_files "$@" ;;
  help|-h|--help) cmd_help ;;
  *) cmd_help >&2; exit 2 ;;
esac
