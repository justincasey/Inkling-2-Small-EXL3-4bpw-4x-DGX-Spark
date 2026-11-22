#!/bin/bash
# quality_vs_spec.sh LABEL [KEY=VAL ...]   the engine's drift from the spec oracle, with oracle/ untouched (docs/ORACLE-GAP.md, "Reproduce" and "Gates").
# Boots the HEAD of CANDIDATE_SRC (default the Segment B worktree) with the full dump and the given extra env, runs bench/tools/greedy.py over the four spec
# fixtures on rank 0 (the engine's own greedy tokens, then each prompt + those tokens as one teacher-forced prefill the dump holds), gathers the four ranks'
# dumps on rank 0, builds each fixture's engine dump with bench/tools/merge_dump.py and compares it with the fixture by oracle/compare.py with the
# `none` gate set (the spec oracle: KL mean <= 2.5e-2, top-1 >= 0.95, per-layer hidden cosine median >= 0.95). Everything runs from the checked-in
# recipe tools and the oracle's own files on rank 0 (/home/spark/inkling2-oracle: the fixtures and a venv with torch); results come back to
# .work/quality/LABEL/ (greedy.json, cmp/NAME.json, summary.txt). It boots and stops the engine: run it under `scripts/safeloop.sh cmd TIMEOUT`.
set -u
cd "$(dirname "$0")/.." || exit 2
LABEL=${1:?usage: quality_vs_spec.sh LABEL [KEY=VAL ...]}; shift
export CANDIDATE_SRC=${CANDIDATE_SRC:-$HOME/projects/rexl3-inkling2-b}
export REXL3_SRC=$CANDIDATE_SRC
H=(192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18)
SSH="ssh -o BatchMode=yes"
OUT=.work/quality/$LABEL
ORA=/home/spark/inkling2-oracle
DUMP=/home/spark/q-dump
W=/home/spark/q-work
mkdir -p "$OUT/cmp"
SHA=$(git -C "$CANDIDATE_SRC" rev-parse --short=12 HEAD)
for h in "${H[@]}"; do $SSH spark@$h "rm -rf $DUMP"; done
$SSH spark@${H[0]} "rm -rf $W && mkdir -p $W/tools $W/eng $W/cmp"
scp -q -o BatchMode=yes bench/tools/greedy.py bench/tools/inkling_api.py bench/tools/stlite.py bench/tools/merge_dump.py spark@${H[0]}:$W/tools/ || { echo "FAIL: copying the tools"; exit 1; }
./inkling2.sh stop >/dev/null 2>&1
./inkling2.sh build "$SHA" 2>&1 | tail -1
echo "engine $SHA, env: $*"
DUMP_DIR=$DUMP EXTRA_ENV="$*" EXTRA_FLAGS="--kv-tokens 262144" ./inkling2.sh start 2>&1 | grep -E 'boot_s=|FAIL|fatal'
$SSH spark@${H[0]} "cd $W/tools && INKLING2_URL=http://127.0.0.1:8000 python3 greedy.py --fixtures $ORA/fixtures --teacher-force --out $W/greedy.json" 2>&1 | tail -12 | tee "$OUT/greedy.txt"
./inkling2.sh stop 2>&1 | tail -1
for r in 1 2 3; do
  $SSH spark@${H[$r]} "tar -C $DUMP -cf - rank$r" | $SSH spark@${H[0]} "tar -C $DUMP -xf -" || { echo "FAIL: gathering rank $r"; exit 1; }
done
for name in chat-capital-effort0 chat-sky-effort0.9 raw-counting raw-long-window; do
  echo "== $name"
  $SSH spark@${H[0]} "cd $W && python3 tools/merge_dump.py --dump $DUMP --fixture $ORA/fixtures/$name.safetensors --greedy $W/greedy.json --name $name --out $W/eng/$name.safetensors 2>&1 | tail -2 && $ORA/.venv/bin/python $ORA/compare.py $ORA/fixtures/$name.safetensors $W/eng/$name.safetensors --json $W/cmp/$name.json --gates none 2>&1 | tail -14"
done 2>&1 | tee "$OUT/summary.txt"
scp -q -o BatchMode=yes spark@${H[0]}:$W/greedy.json "$OUT/greedy.json"
scp -q -o BatchMode=yes "spark@${H[0]}:$W/cmp/*.json" "$OUT/cmp/"
for h in "${H[@]}"; do $SSH spark@$h "rm -rf $DUMP"; done
$SSH spark@${H[0]} "rm -rf $W"
echo "results: $OUT"
