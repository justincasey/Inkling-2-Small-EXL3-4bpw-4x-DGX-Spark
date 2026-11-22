#!/bin/bash
# ops_dump_smoke.sh: the oracle's op-level dump path on four ranks. One boot of the HEAD of CANDIDATE_SRC with REXL3_INKLING2_DUMP_OPS=1 and one cold
# prefill of N ids (default 1,200: a chunk whose layer sums are past the reduce-scatter size, where the f32 and bf16 gathers differ). Rank 0 alone holds
# the f32 sums of the dump, but every rank must take the same collective: a rank-0-only choice is a size mismatch inside one NCCL group, which hangs or
# corrupts. Passes when the request completes and rank 0 wrote its dump files. Run it under `scripts/safeloop.sh cmd`.
set -u
cd "$(dirname "$0")/.." || exit 2
export CANDIDATE_SRC=${CANDIDATE_SRC:-$HOME/projects/rexl3-inkling2-a}
export REXL3_SRC=$CANDIDATE_SRC
N=${N:-1200}
H=(192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18)
D=/home/spark/ops-smoke
clean() { for h in "${H[@]}"; do ssh -o BatchMode=yes spark@"$h" "rm -rf $D"; done; }
clean
./inkling2.sh stop >/dev/null 2>&1
./inkling2.sh build "$(git -C "$CANDIDATE_SRC" rev-parse HEAD)" 2>&1 | tail -1
DUMP_DIR=$D EXTRA_ENV="REXL3_INKLING2_DUMP_OPS=1" ./inkling2.sh start 2>&1 | grep -E 'boot_s=|FAIL|fatal'
rc=0
out=$(timeout 900 python3 scripts/plp_req.py "$N" 2>&1); prc=$?
echo "$out" | tail -1
[[ $prc == 0 ]] || rc=1
files=$(ssh -o BatchMode=yes spark@"${H[0]}" "ls $D/rank0 2>/dev/null | wc -l")
echo "rank 0 dump files: $files"
[[ $files -gt 0 ]] || rc=1
./inkling2.sh stop 2>&1 | tail -1
clean
if (( rc == 0 )); then echo "PASS ops-dump smoke ($N ids, four ranks)"; else echo "FAIL ops-dump smoke"; fi
exit $rc
