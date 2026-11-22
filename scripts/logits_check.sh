#!/bin/bash
# logits_check.sh: the byte-for-byte logits check of docs/exec-plans/completed/autoresearch-segment-a.md. Three cold boots, one cold N-id prefill each (N=9292,
# prefix cache off): the pinned engine f4e27ca, the HEAD of CANDIDATE_SRC with the full dump, and the HEAD with REXL3_INKLING2_DUMP_LOGITS_ONLY=1 (two lanes
# on). Every rank's logits shard of every row is compared with `cmp` against the pinned engine's, per rank and file: all must be identical (48 comparisons).
# It builds, boots and stops the engine: run it under `scripts/safeloop.sh cmd`.
set -u
cd "$(dirname "$0")/.." || exit 2
export CANDIDATE_SRC=${CANDIDATE_SRC:-$HOME/projects/rexl3-inkling2-a}
export REXL3_SRC=$CANDIDATE_SRC
N=${N:-9292}
H=(192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18)
for h in "${H[@]}"; do ssh -o BatchMode=yes spark@$h 'rm -rf /home/spark/plp-A /home/spark/plp-Bfull /home/spark/plp-Blanes'; done
FIN=$(git -C "$CANDIDATE_SRC" rev-parse --short=12 HEAD)
run() {  # name sha extra_env
  local name=$1 sha=$2 xenv=$3
  ./inkling2.sh stop >/dev/null 2>&1
  ./inkling2.sh build $sha 2>&1 | tail -1
  echo "  $name: crates compiled by this build: $(ssh -o BatchMode=yes spark@${H[0]} "grep -c 'Compiling gb-' /home/spark/inkling-2-small-exl3-4bpw-4x-dgx-spark/logs/build.log") (gb-cuda: $(ssh -o BatchMode=yes spark@${H[0]} "grep -c 'Compiling gb-cuda' /home/spark/inkling-2-small-exl3-4bpw-4x-dgx-spark/logs/build.log"))"
  DUMP_DIR=/home/spark/plp-$name EXTRA_ENV="$xenv" ./inkling2.sh start 2>&1 | grep -E 'boot_s=|FAIL|fatal'
  python3 scripts/plp_req.py $N 2>&1 | tail -1
}
run A f4e27ca ""
run Bfull $FIN ""
run Blanes $FIN "REXL3_INKLING2_DUMP_LOGITS_ONLY=1"
./inkling2.sh stop 2>&1 | tail -1
same=0; diff=0
for r in 0 1 2 3; do
  h=${H[$r]}
  out=$(ssh -o BatchMode=yes spark@$h 'cd /home/spark; for f in plp-A/rank'$r'/chunk*.logits.f32 plp-A/rank'$r'/chunk*.json; do b=${f##*/}; for v in Bfull Blanes; do if cmp -s "$f" plp-$v/rank'$r'/$b; then echo "same $v $b"; else echo "DIFF $v $b"; fi; done; done')
  echo "== rank $r: $(echo "$out" | grep -c '^same') same, $(echo "$out" | grep -c '^DIFF') differ"
  echo "$out" | grep '^DIFF' | head -3
done
for h in "${H[@]}"; do ssh -o BatchMode=yes spark@$h 'rm -rf /home/spark/plp-A /home/spark/plp-Bfull /home/spark/plp-Blanes'; done
