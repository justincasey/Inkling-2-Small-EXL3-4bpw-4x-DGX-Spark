#!/bin/bash
# logit_fidelity.sh 'label|KEY=VAL ...' ...   how far the logits of engine variants are from an unmodified boot's, row by row, whole vocabulary.
# One build of the HEAD of CANDIDATE_SRC (default the Segment B worktree), then one cold N-id prefill (N=9292, prefix cache off) on a reference boot (no
# extra env, or REF_ENV) and on one boot per variant, all with REXL3_INKLING2_DUMP_LOGITS_ONLY=1 so each rank writes its logits shard of every row. Each rank then
# compares its shard of each variant with the reference's (scripts/logit_stats.c, compiled on the node), the small result files come back, and
# scripts/logit_fidelity.py combines the four shards: KL(reference || variant) and top-1 agreement per row. A variant with an empty env is a control: it
# must show KL 0 and agreement 1 (the same build twice is byte-identical). Results: .work/fid/STAMP/fidelity.txt.
# It builds, boots and stops the engine: run it under `scripts/safeloop.sh cmd TIMEOUT`.
set -u
cd "$(dirname "$0")/.." || exit 2
export CANDIDATE_SRC=${CANDIDATE_SRC:-$HOME/projects/rexl3-inkling2-b}
export REXL3_SRC=$CANDIDATE_SRC
N=${N:-9292}
H=(192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18)
SSH="ssh -o BatchMode=yes"
OUT=${OUT:-.work/fid/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
SHA=$(git -C "$CANDIDATE_SRC" rev-parse --short=12 HEAD)
on_all() { local h; for h in "${H[@]}"; do $SSH spark@$h "$1" || return 1; done; }
on_all 'rm -rf /home/spark/plp-fid-* /home/spark/fid-stats'
./inkling2.sh stop >/dev/null 2>&1
./inkling2.sh build "$SHA" 2>&1 | tail -1
echo "engine $SHA: crates compiled by this build: $($SSH spark@${H[0]} "grep -c 'Compiling gb-' /home/spark/inkling-2-small-exl3-4bpw-4x-dgx-spark/logs/build.log")"
boot() {  # name env
  ./inkling2.sh stop >/dev/null 2>&1
  DUMP_DIR=/home/spark/plp-fid-$1 EXTRA_ENV="REXL3_INKLING2_DUMP_LOGITS_ONLY=1 $2" ./inkling2.sh start 2>&1 | grep -E 'boot_s=|FAIL|fatal'
  python3 scripts/plp_req.py "$N" 2>&1 | tail -1
}
boot ref "${REF_ENV:-}"
names=()
for v in "$@"; do
  label=${v%%|*}; envs=${v#*|}; [[ $v == *'|'* ]] || envs=""
  names+=("$label")
  echo "== $label  env: ${envs:-(none)}"
  boot "$label" "$envs"
done
./inkling2.sh stop 2>&1 | tail -1
for h in "${H[@]}"; do
  scp -q -o BatchMode=yes scripts/logit_stats.c spark@$h:/home/spark/logit_stats.c && $SSH spark@$h 'gcc -O2 -o /home/spark/logit_stats /home/spark/logit_stats.c -lm' || { echo "FAIL: compiling logit_stats on $h"; exit 1; }
done
for r in 0 1 2 3; do
  h=${H[$r]}
  scp -q -o BatchMode=yes spark@$h:/home/spark/plp-fid-ref/rank$r/chunk00000.json "$OUT/ref-rank$r.json"
done
# the label of a row is the prompt's next token (the last row has none): from the reference's chunk descriptions (rank 0 holds them all), one file a chunk on every node
mkdir -p "$OUT/ref-chunks" "$OUT/labels"
scp -q -o BatchMode=yes "spark@${H[0]}:/home/spark/plp-fid-ref/rank0/chunk*.json" "$OUT/ref-chunks/"
python3 - "$OUT" <<'PY' || { echo "FAIL: building the labels"; exit 1; }
import glob, json, os, struct, sys
out = sys.argv[1]
files = sorted(glob.glob(os.path.join(out, "ref-chunks", "chunk*.json")))
chunks = [json.load(open(f)) for f in files]
toks = [t for c in chunks for t in c["tokens"]]
assert len(toks) == sum(c["rows"] for c in chunks), "the chunks' token lists are not their rows"
at = 0
for f, c in zip(files, chunks):
    lab = [toks[at + i + 1] if at + i + 1 < len(toks) else -1 for i in range(c["rows"])]
    open(os.path.join(out, "labels", os.path.basename(f)[:-5] + ".i32"), "wb").write(struct.pack(f"<{len(lab)}i", *lab))
    at += c["rows"]
PY
for h in "${H[@]}"; do $SSH spark@$h 'rm -rf /home/spark/fid-labels; mkdir -p /home/spark/fid-labels' && scp -q -o BatchMode=yes "$OUT"/labels/*.i32 spark@$h:/home/spark/fid-labels/ || { echo "FAIL: labels to $h"; exit 1; }; done
for label in "${names[@]}"; do
  mkdir -p "$OUT/$label"
  for r in 0 1 2 3; do
    h=${H[$r]}
    # shard width from the file size and the chunk's row count; one background job per rank
    (
      $SSH spark@$h 'bash -s' "$r" "$label" <<'REMOTE'
r=$1; v=$2; cd /home/spark; mkdir -p fid-stats
for f in plp-fid-ref/rank$r/chunk*.logits.f32; do
  c=${f##*/}; c=${c%.logits.f32}
  n=$(python3 -c "import json,os,sys; print(os.path.getsize(sys.argv[2]) // 4 // json.load(open(sys.argv[1]))['rows'])" plp-fid-ref/rank$r/$c.json "$f")
  base=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['vocab_base'])" plp-fid-ref/rank$r/$c.json)
  ./logit_stats "$f" plp-fid-$v/rank$r/$c.logits.f32 "$n" fid-stats/$v-rank$r-$c.stats fid-labels/$c.i32 "$base" >/dev/null || exit 1
done
REMOTE
    ) &
  done
  wait
  for r in 0 1 2 3; do
    h=${H[$r]}
    scp -q -o BatchMode=yes "spark@$h:/home/spark/fid-stats/$label-rank$r-chunk*.stats" "$OUT/$label/" || { echo "FAIL: stats of $label from $h"; exit 1; }
    for f in "$OUT/$label"/$label-rank$r-chunk*.stats; do mv "$f" "$OUT/$label/${f##*/$label-}"; done
  done
done
on_all 'rm -rf /home/spark/plp-fid-* /home/spark/fid-stats /home/spark/fid-labels /home/spark/logit_stats /home/spark/logit_stats.c'
python3 scripts/logit_fidelity.py "$OUT" "${names[@]}" | tee "$OUT/fidelity.txt"
echo "results: $OUT"
