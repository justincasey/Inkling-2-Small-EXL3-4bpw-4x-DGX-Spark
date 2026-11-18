#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Static checks of the recipe: no ssh, no GPU, nothing on the nodes. Run before any hardware run (AGENTS.md).
#   - bash -n on every shell script
#   - print-serve: four rank commands with the engine contract's flags, identical but for --tp-rank, an identical
#     environment (every NCCL_* variable included) on every rank, and DUMP_DIR / EXTRA_ENV / EXTRA_FLAGS on every rank;
#     SPEC / PREFIX_CACHE / VISION / MEMORY_FLOOR_GIB map to their flags (or variable) on every rank, defaults included,
#     and bad values are refused
#   - start runs exactly what print-serve prints (one builder, serve_cmd)
#   - the usage header lists every command of the dispatcher
#   - python tools compile and answer --help
#   - the memory guard's logic on a fake /proc/meminfo: kill on a low floor, kill on swap growth, minimum records and
#     the memwatch window, refusal of a broken configuration
#   - merge_dump.py on synthetic dumps (tests/test_merge_dump.py; with oracle/.venv also through oracle/compare.py)
#   - gates.py / synth_images.py pure parts: PNG writer decodes back, record diffs, usage parsing (tests/test_gates.py)
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 1
BASH_BIN=$(command -v bash)
fails=0
ok() { printf 'ok   %s\n' "$*"; }
bad() { printf 'FAIL %s\n' "$*"; fails=$((fails + 1)); }

# ── shell syntax ──
for f in inkling2.sh config/pins.sh scripts/*.sh tests/*.sh; do
  if "$BASH_BIN" -n "$f"; then ok "bash -n $f"; else bad "bash -n $f"; fi
done

# ── print-serve: the engine contract ──
# The caller's environment beats .env: pin the tunables to their contract defaults so a local .env cannot hide a bug.
serve_env=(MAX_MODEL_LEN=1048576 SLOTS=4 PREFILL_CHUNK=4096 TP_PORT=29571 EXTRA_FLAGS= EXTRA_ENV= DUMP_DIR=
  SPEC=dspark PREFIX_CACHE=on VISION=on MEMORY_FLOOR_GIB=
  NODES="192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18" FABRIC_ADDRS="10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.4")
# check_serve LABEL WANT_ENV WANT_FLAGS PRINT_SERVE_OUTPUT WANT_SERVE; WANT_SERVE: "spec=S prefix=on|off|auto|MiB
# vision=on|off floor=GiB-or-empty"
check_serve() {
  SERVE_TEXT=$4 python3 - "$@" <<'PY'
import os, shlex, sys
label, want_env, want_flags = sys.argv[1], sys.argv[2], sys.argv[3]
want = dict(kv.split("=", 1) for kv in sys.argv[5].split())
text = os.environ["SERVE_TEXT"]
lines = [l for l in text.splitlines() if l and not l.startswith("#")]
heads = [l for l in text.splitlines() if l.startswith("# rank ")]
errs = []
if len(lines) != 4 or len(heads) != 4:
    errs.append(f"expected 4 rank commands, got {len(lines)}")
envs, argvs = [], []
for r, line in enumerate(lines):
    w = shlex.split(line)
    if w[:3] != ["exec", "env", "-i"]:
        errs.append(f"rank {r}: does not start with `exec env -i`")
        continue
    b = next((i for i, x in enumerate(w) if x.endswith("/bin/rexl3-serve")), None)
    if b is None:
        errs.append(f"rank {r}: no .../bin/rexl3-serve")
        continue
    env = dict(x.split("=", 1) for x in w[3:b])
    args = w[b + 1:]
    envs.append(env)
    argvs.append(args)
    def val(flag):
        return args[args.index(flag) + 1] if flag in args else None
    if val("--tp-rank") != str(r):
        errs.append(f"rank {r}: --tp-rank {val('--tp-rank')}")
    for flag, v in [("--tp-ranks", "4"), ("--tp-peer", "10.100.10.1:29571"), ("--max-model-len", "1048576"),
                    ("--slots", "4"), ("--prefill-chunk", "4096"), ("--kv-dtype", "int8"), ("--spec", want["spec"]),
                    ("--host", "0.0.0.0")]:
        if val(flag) != v:
            errs.append(f"rank {r}: {flag} {val(flag)!r}, contract {v!r}")
    if args.count("--spec") != 1:
        errs.append(f"rank {r}: --spec given {args.count('--spec')} times")
    p = want["prefix"]
    got = (val("--prefix-cache"), val("--prefix-cache-mib"))
    exp = {"on": ("on", None), "off": ("off", None), "auto": (None, None)}.get(p, (None, p))
    if got != exp:
        errs.append(f"rank {r}: PREFIX_CACHE={p}: (--prefix-cache, --prefix-cache-mib) {got}, want {exp}")
    if ("--no-vision" in args) != (want["vision"] == "off"):
        errs.append(f"rank {r}: VISION={want['vision']}, --no-vision {'present' if '--no-vision' in args else 'absent'}")
    if env.get("REXL3_MEMORY_FLOOR_GIB") != (want["floor"] or None):
        errs.append(f"rank {r}: REXL3_MEMORY_FLOOR_GIB={env.get('REXL3_MEMORY_FLOOR_GIB')!r}, want {want['floor'] or None!r}")
    for flag in ("--allow-unvalidated-pack",):
        if flag not in args:
            errs.append(f"rank {r}: {flag} missing")
    for flag in ("--model", "--served-model-name", "--port"):
        if not val(flag):
            errs.append(f"rank {r}: {flag} missing")
    if not (val("--model") or "").endswith("/Inkling-2-Small-EXL3-4bpw"):
        errs.append(f"rank {r}: --model {val('--model')}")
    for kv in want_env.split():
        k, v = kv.split("=", 1)
        if env.get(k) != v:
            errs.append(f"rank {r}: {k}={env.get(k)!r}, want {v!r}")
    if want_flags and " ".join(args[-len(want_flags.split()):]) != want_flags:
        errs.append(f"rank {r}: extra flags {want_flags!r} not at the end")
    hca = env.get("NCCL_IB_HCA", "")
    if "rocep1s0f1" not in hca:
        errs.append(f"rank {r}: NCCL_IB_HCA {hca!r} lacks rail 0's rocep1s0f1")
if envs and any(e != envs[0] for e in envs):
    errs.append("the environment differs across ranks: " + str([sorted(set(e.items()) ^ set(envs[0].items())) for e in envs]))
nccl = {k for k in envs[0] if k.startswith("NCCL_")} if envs else set()
strip = lambda a: [x for i, x in enumerate(a) if not (a[i - 1:i] == ["--tp-rank"] or x == "--tp-rank")]
if argvs and any(strip(a) != strip(argvs[0]) for a in argvs):
    errs.append("flags differ across ranks beyond --tp-rank")
for e in errs:
    print(f"FAIL print-serve ({label}): {e}")
if not errs:
    print(f"ok   print-serve ({label}): 4 ranks, contract flags, {sys.argv[5]}, identical env ({len(nccl)} NCCL_* variables)")
sys.exit(1 if errs else 0)
PY
}
base_env="NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_GID_INDEX=3 NCCL_IB_TC=106 NCCL_MAX_NCHANNELS=4 REXL3_NCCL_LIB=/home/spark/rexl3/lib/libnccl.so.2"
out=$(env "${serve_env[@]}" "$BASH_BIN" ./inkling2.sh print-serve 2>&1) || bad "print-serve exited non-zero: $out"
check_serve defaults "$base_env" "" "$out" "spec=dspark prefix=on vision=on floor=" || fails=$((fails + 1))
grep -q '^# rank 0 .*vision tower here' <<<"$out" && [[ $(grep -c '^# rank [123] .*no vision tower (rank 0 encodes; same flags)' <<<"$out") == 3 ]] \
  && ok "print-serve says rank 0 alone holds the vision tower" || bad "print-serve does not say where the vision tower is"
out=$(env "${serve_env[@]}" DUMP_DIR=/home/spark/inkling2-dump EXTRA_ENV="REXL3_TEST_A=1 REXL3_TEST_B=x" \
  EXTRA_FLAGS="--kv-tokens 65536" "$BASH_BIN" ./inkling2.sh print-serve 2>&1) || bad "print-serve with passthrough exited non-zero: $out"
check_serve passthrough "$base_env REXL3_INKLING2_DUMP=/home/spark/inkling2-dump REXL3_TEST_A=1 REXL3_TEST_B=x" \
  "--kv-tokens 65536" "$out" "spec=dspark prefix=on vision=on floor=" || fails=$((fails + 1))
out=$(env "${serve_env[@]}" SPEC=none PREFIX_CACHE=2048 VISION=off MEMORY_FLOOR_GIB=6.5 "$BASH_BIN" ./inkling2.sh print-serve 2>&1) \
  || bad "print-serve (serial, MiB cache, no vision, floor) exited non-zero: $out"
check_serve "serial, MiB cache, no vision, floor" "$base_env REXL3_MEMORY_FLOOR_GIB=6.5" "" "$out" \
  "spec=none prefix=2048 vision=off floor=6.5" || fails=$((fails + 1))
grep -q 'vision tower' <<<"$out" && bad "print-serve names a vision tower with VISION=off" || ok "VISION=off: print-serve names no vision tower"
out=$(env "${serve_env[@]}" SPEC=copy PREFIX_CACHE=off "$BASH_BIN" ./inkling2.sh print-serve 2>&1) || bad "print-serve (copy, off) exited non-zero: $out"
check_serve "copy, cache off" "$base_env" "" "$out" "spec=copy prefix=off vision=on floor=" || fails=$((fails + 1))
out=$(env "${serve_env[@]}" SPEC=auto PREFIX_CACHE=auto "$BASH_BIN" ./inkling2.sh print-serve 2>&1) || bad "print-serve (auto, auto) exited non-zero: $out"
check_serve "auto, cache auto" "$base_env" "" "$out" "spec=auto prefix=auto vision=on floor=" || fails=$((fails + 1))
for r in 0 1 2 3; do
  one=$(env "${serve_env[@]}" "$BASH_BIN" ./inkling2.sh print-serve "$r" | grep -v '^#')
  [[ $one == "$(grep -v '^#' <<<"$(env "${serve_env[@]}" "$BASH_BIN" ./inkling2.sh print-serve)" | sed -n "$((r + 1))p")" ]] \
    && ok "print-serve $r == line $r of print-serve" || bad "print-serve $r differs from the full listing"
done
if env DUMP_DIR=relative/dir "$BASH_BIN" ./inkling2.sh print-serve >/dev/null 2>&1; then bad "a relative DUMP_DIR was accepted"; else ok "a relative DUMP_DIR is refused"; fi
if env EXTRA_ENV="notakeyvalue" "$BASH_BIN" ./inkling2.sh print-serve >/dev/null 2>&1; then bad "EXTRA_ENV without KEY= was accepted"; else ok "EXTRA_ENV without KEY= is refused"; fi
# (an empty value is the default, as for every key: `${KEY:=default}`)
for kv in SPEC=mtp SPEC=DSpark PREFIX_CACHE=pool PREFIX_CACHE=1.5 VISION=yes MEMORY_FLOOR_GIB=0.1 \
  MEMORY_FLOOR_GIB=abc MEMORY_FLOOR_GIB=-2 MEMORY_FLOOR_GIB=8GiB; do
  if env "${serve_env[@]}" "$kv" "$BASH_BIN" ./inkling2.sh print-serve >/dev/null 2>&1; then bad "$kv was accepted"; else ok "$kv is refused"; fi
done
# the defaults a fresh checkout boots with (the caller's environment and .env aside) and .env.example's values
for kv in SPEC:dspark PREFIX_CACHE:on VISION:on MEMORY_FLOOR_GIB:; do
  k=${kv%%:*} v=${kv#*:}
  grep -q "^: \"\${$k:=$v}\"" inkling2.sh && ok "inkling2.sh defaults $k to '$v'" || bad "inkling2.sh does not default $k to '$v'"
  grep -Eq "^$k=$v( |$)" .env.example && ok ".env.example sets $k=$v" || bad ".env.example does not set $k=$v"
done
grep -q 'skip=(); \[\[ $VISION == on \]\] || skip=(--skip-image)' inkling2.sh \
  && ok "smoke skips the image check exactly when VISION=off" || bad "smoke does not follow VISION"
# start must hand the rank the print-serve string itself
if grep -q 'CMD="$(serve_cmd "$r")"' inkling2.sh && grep -q '^    serve_cmd "$r"$' inkling2.sh \
  && grep -q 'setsid nohup bash -c "exec 9>>/tmp/rexl3-gpu.lock; flock -x -n 9 .*; $CMD"' inkling2.sh; then
  ok "start and print-serve share serve_cmd (run under the exclusive GPU lock)"
else bad "start does not run the serve_cmd string print-serve prints"; fi

# ── usage header lists every command ──
helptext=$("$BASH_BIN" ./inkling2.sh help)
cmds=$(awk '/^case \$\{1:-help\} in/ {on = 1; next} on && /^esac/ {exit} on && /^  [a-z-]+(\|[a-z-]+)*\)/ {sub(/\).*/, ""); sub(/^  /, ""); print}' inkling2.sh)
[[ -n $cmds ]] || bad "no dispatcher found in inkling2.sh"
for c in $cmds; do
  [[ $c == *"|"* ]] && continue   # help|-h|--help
  if grep -Eq "^  $c( |$)" <<<"$helptext"; then ok "usage lists $c"; else bad "usage header does not list '$c'"; fi
done
for k in TP_PORT MAX_MODEL_LEN SLOTS PREFILL_CHUNK SPEC PREFIX_CACHE VISION MEMORY_FLOOR_GIB EXTRA_FLAGS EXTRA_ENV DUMP_DIR \
  READY_TIMEOUT_S STOP_GRACE_S MIN_AVAIL_GIB GUARD_START_MIB GUARD_MIB GUARD_SWAP_MAX_MIB DROP_CACHES_EVERY_S; do
  grep -Eq "^#? ?$k=" .env.example && ok ".env.example documents $k" || bad ".env.example lacks $k"
done

# ── python ──
if python3 -m py_compile bench/tools/*.py scripts/*.py tests/*.py; then ok "py_compile bench/tools scripts tests"
else bad "py_compile"; fi
for t in bench/tools/*.py; do
  grep -q '^def main' "$t" || continue   # modules (inkling_api, stlite)
  if python3 "$t" --help >/dev/null 2>&1; then ok "$t --help"; else bad "$t --help"; fi
done

# ── memory guard logic (fake meminfo; local processes) ──
tmp=$(mktemp -d)
trap 'kill $(jobs -p) 2>/dev/null; [[ -n ${rpid:-} ]] && kill "$rpid" 2>/dev/null; rm -rf "$tmp"' EXIT
meminfo() { printf 'MemTotal: 125000000 kB\nMemAvailable: %d kB\nSwapTotal: 16777216 kB\nSwapFree: %d kB\n' \
  "$(( $1 * 1024 ))" "$(( 16777216 - $2 * 1024 ))" > "$tmp/meminfo.new"; mv "$tmp/meminfo.new" "$tmp/meminfo"; }
# shellcheck disable=SC2034  # gpid and rpid are read by the wait_for conditions (eval)
guard() {   # guard NAME: start a guard over a fresh `sleep` rank; sets gpid, rpid
  local n=$1
  : > "$tmp/$n.serve.log"; echo loading > "$tmp/$n.stage"
  MEMINFO=$tmp/meminfo GUARD_PIDFILE=$tmp/$n.pid GUARD_RANK=0 GUARD_LOG=$tmp/$n.guard.log GUARD_MIN_FILE=$tmp/$n.min \
    GUARD_STAGE_FILE=$tmp/$n.stage GUARD_WINDOW_FILE=$tmp/$n.window GUARD_SERVE_LOG=$tmp/$n.serve.log \
    GUARD_START_MIB=2048 GUARD_MIB=1024 GUARD_SWAP_MAX_MIB=64 DROP_CACHES_EVERY_S=0 GUARD_SAMPLE_S=0.05 \
    "$BASH_BIN" scripts/memguard.sh & gpid=$!
  rpid=$( (sleep 300 >/dev/null 2>&1 & echo $!) )   # not a job of this shell: no "Killed" chatter
  echo "$rpid" > "$tmp/$n.pid"
}
wait_for() { local i; for (( i = 0; i < 100; i++ )); do eval "$1" && return 0; sleep 0.05; done; return 1; }
get() { sed -n "s/^$2=//p" "$tmp/$1.min"; }

meminfo 60000 100
guard low
wait_for 'grep -q " armed pid=$rpid " "$tmp/low.guard.log" 2>/dev/null' && ok "guard arms on the rank's pid file" || bad "guard did not arm"
meminfo 1500 100   # above the ready floor, below the loading floor: loading kills
wait_for '! kill -0 $rpid 2>/dev/null' && ok "guard SIGKILLs the rank below the loading floor" || bad "guard did not kill below the floor"
wait_for '! kill -0 $gpid 2>/dev/null' && ok "guard exits after its rank" || bad "guard outlived its rank"
grep -q 'level=fatal event=memguard_kill' "$tmp/low.serve.log" && ok "the kill reason is a level=fatal line in the rank log" \
  || bad "no level=fatal line in the rank log"
[[ $(get low min_mib) == 1500 && $(get low loading_min_mib) == 1500 ]] && ok "guard records the minimum (1500 MiB)" \
  || bad "guard minimum: $(get low min_mib)"

meminfo 60000 100
guard ready
echo ready > "$tmp/ready.stage"
wait_for 'grep -q " armed " "$tmp/ready.guard.log" 2>/dev/null' || bad "guard did not arm (ready)"
meminfo 1500 100   # ready floor 1024: survives
sleep 0.5
kill -0 "$rpid" 2>/dev/null && ok "once ready, 1500 MiB is above the floor" || bad "guard killed above the ready floor"
echo w1 > "$tmp/ready.window"; meminfo 3000 100
wait_for '[[ $(get ready window) == w1 && $(get ready window_min_mib) == 3000 ]]' \
  && ok "memwatch window restarts its minimum (3000 MiB, overall $(get ready min_mib))" || bad "window minimum: $(get ready window_min_mib)"
[[ $(get ready ready_min_mib) == 1500 ]] && ok "ready minimum recorded" || bad "ready minimum: $(get ready ready_min_mib)"
meminfo 3000 200   # swap +100 MiB > 64
wait_for '! kill -0 $rpid 2>/dev/null' && ok "guard SIGKILLs the rank on swap growth" || bad "guard ignored swap growth"
grep -q 'swap grew 100 MiB' "$tmp/ready.guard.log" && ok "the guard log says why" || bad "no swap reason in the guard log"
wait_for '! kill -0 $gpid 2>/dev/null' || bad "guard outlived its rank (swap)"

if GUARD_PIDFILE=$tmp/x GUARD_RANK=0 GUARD_LOG=$tmp/x.log GUARD_MIN_FILE=$tmp/x.min GUARD_STAGE_FILE=$tmp/x.s \
  GUARD_WINDOW_FILE=$tmp/x.w GUARD_SERVE_LOG=$tmp/x.serve GUARD_START_MIB=2048 GUARD_MIB=0 GUARD_SWAP_MAX_MIB=64 \
  DROP_CACHES_EVERY_S=0 MEMINFO=$tmp/meminfo "$BASH_BIN" scripts/memguard.sh 2>/dev/null; then bad "guard accepted a 0 MiB floor"
else ok "guard refuses a 0 MiB floor"; fi
if GUARD_PIDFILE=$tmp/x GUARD_RANK=0 GUARD_LOG=$tmp/x.log GUARD_MIN_FILE=$tmp/x.min GUARD_STAGE_FILE=$tmp/x.s \
  GUARD_WINDOW_FILE=$tmp/x.w GUARD_SERVE_LOG=$tmp/x.serve GUARD_START_MIB=2048 GUARD_MIB=1024 GUARD_SWAP_MAX_MIB=64 \
  DROP_CACHES_EVERY_S=20 GUARD_DROP_CMD=false MEMINFO=$tmp/meminfo "$BASH_BIN" scripts/memguard.sh 2>/dev/null; then
  bad "guard started without page-cache drop permission"
else ok "guard refuses to start when it cannot drop the page cache"; fi

# ── merge_dump ──
cmp=(); [[ -x oracle/.venv/bin/python ]] && cmp=(--compare-python oracle/.venv/bin/python)
if res=$(python3 tests/test_merge_dump.py "${cmp[@]}" 2>&1); then grep '^ok' <<<"$res"; else echo "$res"; bad "tests/test_merge_dump.py"; fi

# ── gates / smoke helpers ──
if res=$(python3 tests/test_gates.py 2>&1); then grep '^ok' <<<"$res"; else echo "$res"; bad "tests/test_gates.py"; fi

echo
if (( fails == 0 )); then echo "test_recipe: all checks passed"; else echo "test_recipe: $fails check(s) failed"; exit 1; fi
