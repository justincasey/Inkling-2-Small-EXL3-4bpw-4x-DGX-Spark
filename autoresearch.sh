#!/usr/bin/env bash
# The benchmark for Inkling 2 Small (REXL3 engine, TP4 on four DGX Sparks): one candidate engine commit, one fixed workload,
# printed as `METRIC name=value` lines. Exits 0 when every gate that ran passed, non-zero on a failed gate, boot, build or
# clock check. Segment A of docs/AUTORESEARCH-HANDOFF.md: strictly bit-preserving work (see "Not in this harness").
#
#   CANDIDATE_SRC=~/projects/rexl3-inkling2-ar ./autoresearch.sh        a full run: every phase but prefill-1m, a score
#   PHASES=warm,decode ./autoresearch.sh                                a screen: those phases only, target_score 0
#   PIN=1 ./autoresearch.sh                                             pin or extend bench/reference.json and
#                                                                       bench/reference-vision.json (unmodified engine only)
#
# The candidate is the HEAD commit of the engine checkout CANDIDATE_SRC (default ~/projects/rexl3-inkling2-ar, branch
# `inkling2-ar` off `inkling2` @ f4e27ca); its tracked files must be clean, because the build ships the commit. It is built
# with `./inkling2.sh build <sha>` unless all four ranks already run it. Every other setting comes from the tracked file
# .env.example, copied over .env at the start: an experiment that changes the serve configuration changes that file (or the
# engine); the caller's environment cannot, so runs stay comparable. The GPUs must be free: if the DeepSeek stack holds them,
# `./inkling2.sh stop-other` first and `restore-other` when the session ends (the handoff's rule); this script stops only
# its own engine, before and after.
#
# Phases (PHASES, comma list; unset = a full run: every phase below except prefill-1m, the only way to get a score), in order:
#   host, goldens        engine stopped: `inkling2.sh engine-test host|goldens` on rank 0 from the installed source
#   boot A               PREFILL_CHUNK as in .env.example (4096); capacity line, memwatch window, clock/memory sampling
#   smoke, warm, decode, conc, prefill      bench/tools/bench.py (see its header): decode c = 1 / 4 / 8, cold 32K, 128K and 256K prefill
#   prefill-32k          a quick screen alone: the cold 32K prefill only (a full run has `prefill`, which includes it)
#   vision, spec, cache  bench/tools/gates.py;  chunk: exactness record of boot A;  needle: 200K needle
#   prefill-1m           not in a full run (about 25 minutes; user 2026-11-18: routine runs stop at 256K, the largest contexts are
#                        benchmarked later): PHASES=prefill-1m ./autoresearch.sh, a screen with target_score 0, one cold 1M prompt
#   boot B               only for `chunk`: PREFILL_CHUNK=1000, exactness record, diff against boot A's
#
# Primary metric target_score (higher is better; 1.0 = both user targets met on 4 lanes of 1M):
#   sqrt(decode_c1_tok_s / 70 * prefill_32k_tok_s / 4000) * min(1, lanes_at_max_context / 4)
# It prints 0 unless every gate below printed `PASS <id>`; target_score_raw keeps the number. Secondary metrics: every raw
# metric of bench.py, boot_s, kv_tokens, free_after_gib, prefix_gib, mem_min_gib / mem_min_gib_norm (per rank raw too), gpu_clock_mhz_*.
#
# Fixed gates (each prints `PASS <id>` or `FAIL <id>`, matched whole-line by bench/tools/score.py; a log per id in the run dir):
#   pin        every bench.py reply (decode x4, concurrent x4 and x8, cold 32K / 128K / 256K, and 1M when run) equals bench/reference.json byte for byte
#   spec       gates.py spec: drafted == serial, concurrent == solo
#   cache      gates.py cache: cached == uncached
#   chunk      exactness.py record at prefill chunk 4096 (boot A) and 1000 (boot B), identical
#   vision     gates.py vision: text and image replies equal bench/reference-vision.json
#   needle     needle.py: the 200K needle is found
#   capacity   the capacity line says lanes_at_max_context=4 with kv_tokens=4194304
#   memory     per-rank MemAvailable minimum over the boot and the run (the guards' record), normalized to the recorded 6.03 GiB
#              prefix cache (the plan gives 4.5-6.0 GiB depending on node state; memcheck.py), >= 8 GiB, every guard still running
#   smoke      ./inkling2.sh smoke
#   host       the engine's Inkling 2 host tests (gb-model, gb-cuda, gb-server) compile and pass
#   goldens    testdata/gpu is byte-identical to the pinned engine's, and the five Inkling 2 GPU golden tests pass
#
# Reference replies. `PIN=1` runs the workload on the engine under test and writes the references, or extends them with entries they
# lack (the 256K prefill, 2026-11-18) while requiring every entry already pinned to be reproduced byte for byte (the vision reference
# is compared when it exists). Only for the unmodified pinned engine (config/pins.sh REXL3_COMMIT); pin once per entry, then commit.
# Off limits for the loop: oracle/, bench/reference*.json once pinned, bench/tools/gates.py thresholds, config/pins.sh pack
# pins, config/pack-ranks.tsv, the memory guard's floors (scripts/memguard.sh, GUARD_* and MEMORY_FLOOR_GIB defaults).
#
# GPU clocks (site policy, user 2026-11-10): every rank's GPU SM clock is locked at GPU_CLOCKS (default 2200) MHz
# (`inkling2.sh gpu-clocks`); `keep` skips the lock. All four are sampled every 5 s with MemAvailable; the run fails when a
# sample is above the lock, or when there are none. The lock stays after the run (it is the site's setting).
#
# Not in this harness: the `oracle` gate (oracle/teacher_force.py against the emulating oracle). It belongs to bit-changing
# work (Segment B), which needs the user's decision on the byte-exact policy and a new segment; run it by hand per
# docs/ORACLE-GAP.md "Reproduce" until then.
#
# The body is functions and one call on the last line, so bash has parsed all of it before it runs: replacing this file
# while a run is in flight cannot change what that run executes.
set -euo pipefail
cd "$(dirname "$0")"

PHASES_KNOWN="warm decode conc prefill-32k prefill prefill-1m smoke vision spec cache chunk needle host goldens"
REQUIRED_GATES="pin spec cache chunk vision needle capacity memory smoke host goldens"
REF=bench/reference.json
REF_VISION=bench/reference-vision.json
EXPECT_KV_TOKENS=4194304   # 4 lanes x 1,048,576

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { local rc=${2:-1}; echo "FAIL $1" >&2; exit "$rc"; }
has() { [[ ,$PHASES_RUN, == *,"$1",* ]]; }
metric() { echo "METRIC $1=$2" >> "$METRICS"; }

# gate ID CMD...: run CMD (its output in $RUN/ID.log), record and print PASS/FAIL ID
gate() {
  local id=$1; shift
  log "gate $id: $*"
  if "$@" > "$RUN/$id.log" 2>&1; then echo "PASS $id" | tee -a "$GATES"
  else echo "FAIL $id" | tee -a "$GATES"; tail -n 25 "$RUN/$id.log" >&2; fi
}

# bench LABEL PHASES [bench.py args]: one bench.py invocation; its METRIC lines are kept, the rest is logged
bench() {
  local label=$1 ph=$2 rc=0; shift 2
  log "bench.py $ph"
  python3 bench/tools/bench.py --phases "$ph" --out "$RUN/bench-$label.json" "$@" > "$RUN/bench-$label.out" 2> "$RUN/bench-$label.err" || rc=$?
  { grep '^METRIC ' "$RUN/bench-$label.out" || true; } >> "$METRICS"
  tail -n 40 "$RUN/bench-$label.err" >&2
  BENCH_RAN=1
  (( rc == 0 )) || BENCH_RC=1
  return 0
}

# drop_caches: the page cache on every rank. A boot's capacity plan reads free memory, so the cache cargo leaves after the
# engine tests shrank boot A's prefix cache from 6.03 to 4.52 GiB (free_after_gib 41 -> 39) in the pin run: every boot starts clean.
drop_caches() {
  local -a host pids=(); read -r -a host <<<"$HOSTS"
  local h p rc=0
  for h in "${host[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_USER@$h" 'sync; sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"' & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  (( rc == 0 )) || die "could not drop the page cache on every rank"
}

# boot LABEL [VAR=value ...]: ./inkling2.sh start; sets BOOT_S and keeps the capacity line in $RUN/boot-LABEL.log
boot() {
  local label=$1; shift
  log "boot $label ($*)"
  drop_caches
  if ! env "$@" ./inkling2.sh start > "$RUN/boot-$label.log" 2>&1; then
    tail -n 60 "$RUN/boot-$label.log" >&2; die "boot $label failed"
  fi
  BOOTED=1
  BOOT_S=$(sed -n 's/^boot_s=//p' "$RUN/boot-$label.log" | tail -n 1)
  [[ -n $BOOT_S ]] || die "boot $label: no boot_s line"
}

# samples of every rank, every 5 s: `EPOCH RANK SM_MHZ GPU_UTIL% MEMAVAILABLE_MIB` per line
sampler() {
  local -a host; read -r -a host <<<"$HOSTS"
  local i
  while true; do
    for i in 0 1 2 3; do
      ( timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_USER@${host[$i]}" \
          "echo \"\$(date +%s) $i \$(nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ') \$(awk '/^MemAvailable:/ {print int(\$2/1024)}' /proc/meminfo)\"" \
          >> "$RUN/samples.txt" 2>/dev/null || true ) &
    done
    wait
    sleep 5
  done
}

cleanup() {
  local rc=$?
  trap - EXIT
  [[ -z ${SAMPLER_PID:-} ]] || kill "$SAMPLER_PID" 2>/dev/null || true
  if [[ -n ${BOOTED:-} ]]; then ./inkling2.sh logs all 300 > "$RUN/server.log" 2>&1 || true; fi
  ./inkling2.sh stop > "$RUN/stop-after.log" 2>&1 || echo "WARN could not stop the engine: see $RUN/stop-after.log" >&2
  [[ -n ${PRINTED:-} ]] || { cat "$METRICS" 2>/dev/null || true; }   # a run that ended early still reports what it measured
  exit "$rc"
}

# the capacity line of rank 0's boot: kv_tokens, lanes, free_after_gib, prefix_gib
capacity_gate() {
  local plan lanes kv
  plan=$(grep -h 'event=inkling2_plan' "$RUN/boot-a.log" | head -n 1)
  [[ -n $plan ]] || { echo "no inkling2_plan line in $RUN/boot-a.log"; return 1; }
  field() { grep -o " $1=[^ ]*" <<<"$plan" | head -n 1 | cut -d= -f2; }
  lanes=$(field lanes_at_max_context); kv=$(field kv_tokens)
  metric lanes_at_max_context "$lanes"
  metric kv_tokens "$kv"
  metric free_after_gib "$(field free_after_gib)"
  metric prefix_gib "$(field prefix_gib)"
  echo "$plan"
  [[ $lanes == 4 && $kv == "$EXPECT_KV_TOKENS" ]] \
    || { echo "capacity: lanes_at_max_context=$lanes kv_tokens=$kv, want 4 and $EXPECT_KV_TOKENS"; return 1; }
}

# per-rank MemAvailable minimum since each guard armed (loading included), normalized to the recorded prefix cache, against
# the engine floor, every guard running (bench/tools/memcheck.py)
memory_gate() {
  ./inkling2.sh memwatch report > "$RUN/memwatch.txt" 2>&1 || { cat "$RUN/memwatch.txt"; return 1; }
  cat "$RUN/memwatch.txt"
  python3 bench/tools/memcheck.py --report "$RUN/memwatch.txt" --boot-log "$RUN/boot-a.log" | tee -a "$METRICS"
}

goldens_gate() {
  if ! git -C "$CAND" diff --quiet "$PINNED" HEAD -- testdata/gpu; then
    echo "testdata/gpu differs from the pinned engine ${PINNED:0:12} (a golden was changed):"
    git -C "$CAND" diff --stat "$PINNED" HEAD -- testdata/gpu
    return 1
  fi
  echo "testdata/gpu identical to the pinned engine ${PINNED:0:12}"
  timeout 7200 ./inkling2.sh engine-test goldens
}

vision_gate() {
  timeout 1800 python3 bench/tools/gates.py vision "$RUN/vision.json" || return 1
  if [[ -n $PIN && ! -e $REF_VISION ]]; then cp "$RUN/vision.json" "$REF_VISION"; echo "pinned $REF_VISION"
  else python3 bench/tools/gates.py diff "$REF_VISION" "$RUN/vision.json"; fi
}

chunk_gate() {   # after boot B: record at chunk 1000, diff against boot A's record at 4096
  timeout 3600 python3 bench/tools/exactness.py record "$RUN/exactness-chunk1000.json" --label chunk1000 || return 1
  python3 bench/tools/exactness.py diff "$RUN/exactness-chunk4096.json" "$RUN/exactness-chunk1000.json"
}

# METRIC lines and the clock check from the samples file
clock_summary() {
  local f=$RUN/samples.txt n max med minmem
  n=$(awk 'NF == 5' "$f" 2>/dev/null | wc -l | tr -d ' ')
  max=$(awk 'NF == 5 && $3 > m {m = $3} END {print m + 0}' "$f" 2>/dev/null || echo 0)
  med=$(awk 'NF == 5 && $4 >= 30 {print $3}' "$f" 2>/dev/null | sort -n | awk '{a[NR] = $1} END {print NR ? a[int((NR + 1) / 2)] : 0}')
  minmem=$(awk 'NF == 5 && (m == "" || $5 < m) {m = $5} END {printf "%.2f", m / 1024}' "$f" 2>/dev/null || echo 0)
  metric gpu_clock_samples "$n"
  metric gpu_clock_mhz_max "$max"
  metric gpu_clock_mhz_busy_median "$med"
  metric min_mem_available_gib_sampled "$minmem"
  [[ $GPU_CLOCKS == keep ]] && return 0
  if (( n == 0 )); then echo "FAIL no GPU clock samples: the $GPU_CLOCKS MHz lock could not be verified" >&2; return 1; fi
  if (( max > GPU_CLOCKS + 25 )); then echo "FAIL a GPU clock sample of $max MHz is above the $GPU_CLOCKS MHz lock" >&2; return 1; fi
}

setup() {
  local k p need_ref=0
  PIN=${PIN:-}; GPU_CLOCKS=${GPU_CLOCKS:-2200}
  FULL=0
  if [[ -z ${PHASES:-} ]]; then PHASES_RUN=$PHASES_ALL; FULL=1; else PHASES_RUN=$PHASES; fi
  if [[ -n $PIN ]]; then FULL=0; fi
  for p in ${PHASES_RUN//,/ }; do
    [[ " $PHASES_KNOWN " == *" $p "* ]] || die "unknown phase '$p' (known: $PHASES_KNOWN)" 2
  done
  [[ $GPU_CLOCKS == keep || $GPU_CLOCKS =~ ^[1-9][0-9]*$ ]] || die "GPU_CLOCKS is a clock in MHz or keep (got '$GPU_CLOCKS')" 2
  # the serve configuration is .env.example only: drop every setting a caller's shell may carry
  while IFS='=' read -r k _; do if [[ $k =~ ^[A-Z][A-Z0-9_]*$ ]]; then unset "$k"; fi; done < <(grep -vE '^(#|$)' .env.example)
  RUN=bench/runs/autoresearch-$(date +%Y%m%d-%H%M%S)
  mkdir -p "$RUN"
  GATES=$RUN/gates.txt; METRICS=$RUN/metrics.txt; : > "$GATES"; : > "$METRICS"
  # shellcheck disable=SC1090
  source <(grep -E '^(NODES|SSH_USER)=' .env.example)
  HOSTS=$NODES

  CAND=${CANDIDATE_SRC:-$HOME/projects/rexl3-inkling2-ar}
  git -C "$CAND" rev-parse --git-dir >/dev/null 2>&1 || die "CANDIDATE_SRC=$CAND is not a git checkout of the engine" 2
  [[ -z $(git -C "$CAND" status --porcelain --untracked-files=no) ]] \
    || die "uncommitted changes in $CAND: commit the candidate first (the build ships the commit)" 2
  SHA=$(git -C "$CAND" rev-parse HEAD)
  PINNED=$(sed -n 's/^REXL3_COMMIT=\([0-9a-f]\{40\}\).*/\1/p' config/pins.sh)
  [[ -n $PINNED ]] || die "no REXL3_COMMIT in config/pins.sh" 2
  export REXL3_SRC=$CAND
  if [[ -n $PIN ]]; then
    [[ $SHA == "$PINNED" ]] || die "PIN=1 is for the unmodified pinned engine ${PINNED:0:12}; $CAND is at ${SHA:0:12}" 2
  else
    for p in warm decode conc prefill-32k prefill prefill-1m; do if has "$p"; then need_ref=1; fi; done
    if (( need_ref )) && [[ ! -e $REF ]]; then die "no $REF: pin it once with PIN=1 on the unmodified engine" 2; fi
    if has vision && [[ ! -e $REF_VISION ]]; then die "no $REF_VISION: pin it once with PIN=1 on the unmodified engine" 2; fi
  fi
  echo "ASI engine=${SHA:0:12} candidate_src=$CAND phases=$PHASES_RUN full=$FULL pin=${PIN:-no} clocks=$GPU_CLOCKS run=$RUN"

  cp .env.example .env
  bash tests/test_recipe.sh > "$RUN/static.log" 2>&1 || { tail -n 30 "$RUN/static.log" >&2; die "static recipe tests failed ($RUN/static.log)"; }
}

cluster_ready() {
  local out installed
  ./inkling2.sh stop > "$RUN/stop-before.log" 2>&1 \
    || { tail -n 20 "$RUN/stop-before.log" >&2; die "could not stop the engine or a GPU is in use (the DeepSeek stack? ./inkling2.sh stop-other)" 2; }
  if [[ $GPU_CLOCKS != keep ]]; then
    ./inkling2.sh gpu-clocks "$GPU_CLOCKS" > "$RUN/clocks.log" 2>&1 || { cat "$RUN/clocks.log" >&2; die "could not lock the GPU clocks at $GPU_CLOCKS MHz"; }
  fi
  out=$(./inkling2.sh doctor 2>&1) || { echo "$out" >&2; die "doctor: the cluster is not ready" 2; }
  installed=$(grep -o '\] engine [0-9a-f]\{40\}' <<<"$out" | awk '{print $3}' | sort -u | paste -sd' ' -)
  if [[ $installed == "$SHA" ]]; then log "engine ${SHA:0:12} already installed on all four ranks"
  else
    log "build engine ${SHA:0:12} (installed: ${installed:-none})"
    ./inkling2.sh build "$SHA" > "$RUN/build.log" 2>&1 || { tail -n 40 "$RUN/build.log" >&2; die "build failed ($RUN/build.log)"; }
  fi
}

run_all() {
  PHASES_ALL="host,goldens,smoke,warm,decode,conc,prefill,vision,spec,cache,chunk,needle"
  BENCH_RAN=0; BENCH_RC=0; BOOTED=""; SAMPLER_PID=""; PRINTED=""; BOOT_A_S=0
  setup
  trap cleanup EXIT
  cluster_ready

  if has host; then gate host timeout 7200 ./inkling2.sh engine-test host; fi
  if has goldens; then gate goldens goldens_gate; fi

  local p need_boot=0
  for p in smoke warm decode conc prefill-32k prefill prefill-1m vision spec cache chunk needle; do if has "$p"; then need_boot=1; fi; done
  if (( need_boot )); then
    boot a
    BOOT_A_S=$BOOT_S
    metric boot_s "$BOOT_A_S"
    gate capacity capacity_gate
    ./inkling2.sh memwatch begin > "$RUN/memwatch-begin.txt" 2>&1 || die "could not open the memwatch window"
    sampler & SAMPLER_PID=$!
    boot_a_phases
  fi

  if (( BOOTED )); then clock_summary || echo "FAIL clock" >> "$GATES"; fi
  finish
}

# everything that runs on boot A, then boot B for the chunk gate
boot_a_phases() {
  local ph="" p chunk_a_ok=0 mode=(--ref "$REF")
  if [[ -n $PIN ]]; then mode=(--pin "$REF" --engine "$SHA"); fi
  if has smoke; then gate smoke timeout 900 ./inkling2.sh smoke; fi
  # the chunk record comes first among the requests that read the prefix cache, so boot A's is as cold as boot B's
  if has chunk; then
    if timeout 3600 python3 bench/tools/exactness.py record "$RUN/exactness-chunk4096.json" --label chunk4096 > "$RUN/chunk4096.log" 2>&1; then chunk_a_ok=1
    else tail -n 20 "$RUN/chunk4096.log" >&2; echo "FAIL chunk" | tee -a "$GATES"; fi
  fi
  for p in warm decode conc prefill-32k prefill; do if has "$p"; then ph+="$p,"; fi; done
  if [[ -n $PIN ]] && has prefill-1m; then ph+="prefill-1m,"; fi
  if [[ -n $ph ]]; then bench main "${ph%,}" "${mode[@]}"; fi
  if has vision; then gate vision vision_gate; fi
  if has spec; then gate spec timeout 7200 python3 bench/tools/gates.py spec "$RUN/spec.json"; fi
  if has cache; then gate cache timeout 10800 python3 bench/tools/gates.py cache "$RUN/cache.json"; fi
  if has needle; then gate needle timeout 7200 python3 bench/tools/needle.py --tokens 200000 --depth 0.5 --out "$RUN/needle-200k.json"; fi
  if [[ -z $PIN ]] && has prefill-1m; then bench long1m prefill-1m "${mode[@]}"; fi
  if (( BENCH_RAN )); then
    if (( BENCH_RC == 0 )); then echo "PASS pin" | tee -a "$GATES"; else echo "FAIL pin" | tee -a "$GATES"; fi
  fi
  gate memory memory_gate

  if (( chunk_a_ok )); then
    ./inkling2.sh stop > "$RUN/stop-a.log" 2>&1 || die "could not stop boot A"
    boot b PREFILL_CHUNK=1000
    gate chunk chunk_gate
  fi
}

finish() {
  local failed rc=0
  failed=$(grep '^FAIL ' "$GATES" | cut -d' ' -f2 | sort -u | paste -sd, - || true)
  if (( FULL )); then python3 bench/tools/score.py --metrics "$METRICS" --gates "$GATES" --required "$REQUIRED_GATES" >> "$METRICS"
  else python3 bench/tools/score.py --metrics "$METRICS" --gates "$GATES" --screen >> "$METRICS"; fi
  cat "$METRICS"; PRINTED=1
  echo "ASI engine=${SHA:0:12} boot_s=$BOOT_A_S failed_gates=${failed:-none} run=$RUN"
  if [[ -n $PIN ]]; then echo "PINNED: $REF and $REF_VISION written or extended (engine ${SHA:0:12}); review the diff and commit them"; fi
  [[ -z $failed ]] || rc=1
  exit "$rc"
}

run_all "$@"; exit
