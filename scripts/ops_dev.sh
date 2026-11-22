#!/bin/bash
# ops_dev.sh TESTBIN FILTER...   sync the dev engine tree's crates to rank 0's lab tree, build `cargo test -p gb-cuda --test TESTBIN`, and run each
# FILTER (a test-name substring) under the node's GPU lock. Needs the engine stopped (run it under `safeloop.sh cmd`). The lab tree and its target
# directory are separate from the engine's (docs/AUTORESEARCH-HANDOFF.md section 6).
set -u
SRC=${LAB_SRC:-$HOME/projects/rexl3-inkling2-dev}
B=/home/spark/inkling-2-small-exl3-4bpw-4x-dgx-spark
HOST=192.168.1.121
BIN_NAME=$1; shift
rsync -a --delete "$SRC/crates/" spark@$HOST:$B/lab/crates/ || exit 1
ssh -o BatchMode=yes spark@$HOST "cd $B/lab && export PATH=/usr/local/cuda/bin:\$HOME/.cargo/bin:\$PATH CUDA_HOME=/usr/local/cuda REXL3_CUDA_ARCH=sm_121 REXL3_COMMIT=lab CARGO_TARGET_DIR=$B/target-lab && cargo +1.89.0 test --release --locked -p gb-cuda --test $BIN_NAME --no-run > /tmp/ops_dev_build.log 2>&1; rc=\$?; grep -E '^error|csrc.*error|nvcc' -A8 /tmp/ops_dev_build.log | head -40; exit \$rc" || { echo "ops_dev: the build of $BIN_NAME failed"; exit 1; }
for f in "$@"; do
  echo "== $BIN_NAME $f"
  ssh -o BatchMode=yes spark@$HOST "cd $B/lab && exec 9>>/tmp/rexl3-gpu.lock && flock -x -n 9 && BIN=\$(ls -t $B/target-lab/release/deps/$BIN_NAME-* | grep -v '\\.d\$' | head -1) && REXL3_INKLING2_DIR=/home/spark/models/Inkling-2-Small-EXL3-4bpw \$BIN $f --test-threads=1 2>&1 | grep -E '^test |test result|panicked|FAILED|failed${OPS_GREP:+|$OPS_GREP}' | head -${OPS_LINES:-30}"
done
