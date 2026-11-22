#!/bin/bash
# loop_probe.sh 'label|KEY=VAL ...' ...   boots the HEAD of CANDIDATE_SRC (default the Segment B worktree) once per variant, runs scripts/loop_probe.py against it
# (32 prompts, 512 greedy tokens each) and compares every variant with the first one: first differing token, repetition loops, speed. Results in
# .work/loop/STAMP/. It builds, boots and stops the engine: run it under `scripts/safeloop.sh cmd TIMEOUT`.
set -u
cd "$(dirname "$0")/.." || exit 2
export CANDIDATE_SRC=${CANDIDATE_SRC:-$HOME/projects/rexl3-inkling2-b}
export REXL3_SRC=$CANDIDATE_SRC
OUT=${OUT:-.work/loop/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
SHA=$(git -C "$CANDIDATE_SRC" rev-parse --short=12 HEAD)
./inkling2.sh stop >/dev/null 2>&1
./inkling2.sh build "$SHA" 2>&1 | tail -1
names=()
for v in "$@"; do
  label=${v%%|*}; envs=${v#*|}; [[ $v == *'|'* ]] || envs=""
  names+=("$label")
  echo "== $label  env: ${envs:-(none)}"
  ./inkling2.sh stop >/dev/null 2>&1
  EXTRA_ENV="$envs" ./inkling2.sh start 2>&1 | grep -E 'boot_s=|FAIL|fatal'
  python3 scripts/loop_probe.py run "$OUT/$label.json" | tail -3
  python3 scripts/loop_probe.py run "$OUT/$label.json" | tail -3
  if [[ -n ${NEEDLE:-} ]]; then   # 200K needle at three depths with a 512-token answer budget (the gate's is 128): is the code found, and in how many reasoning tokens
    for d in 0.1 0.5 0.9; do
      python3 bench/tools/needle.py --tokens 200000 --depth "$d" --max-tokens 512 --out "$OUT/needle-$label-$d.json" >/dev/null 2>&1
      python3 - "$OUT/needle-$label-$d.json" "$label" "$d" <<'PY'
import json, sys
l = json.load(open(sys.argv[1]))["lanes"][0]
u = l["usage"]
print(f"needle {sys.argv[2]:6} depth {sys.argv[3]}: found={l['found']} answer={l['answer']!r} completion={u['completion_tokens']} reasoning={u['completion_tokens_details']['reasoning_tokens']} finish={l['finish_reason']}")
PY
    done
  fi
done
./inkling2.sh stop 2>&1 | tail -1
for l in "${names[@]:1}"; do
  echo "== ${names[0]} -> $l"
  python3 scripts/loop_probe.py compare "$OUT/${names[0]}.json" "$OUT/$l.json" | tee "$OUT/compare-${names[0]}-$l.txt"
done
echo "results: $OUT"
