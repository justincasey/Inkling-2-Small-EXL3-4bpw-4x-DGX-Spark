#!/bin/bash
# quad_lat.sh [--no-build] ["NCCL_X=1 NCCL_Y=2" ...]: build crates/cuda/tests/nccl_oneshot_quad.rs in the lab tree on rank 0, copy the binary to
# ranks 1-3, and run it on all four (each under its node's GPU lock, the serve environment from config/pins.sh plus the given words) once per
# argument (no argument: the pinned environment alone). Prints rank 0's table. Needs the engine stopped.
set -u
cd /Users/justin/recipes/Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark
source config/pins.sh
NODES=(192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18)
FAB=(10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.4)
B=/home/spark/inkling-2-small-exl3-4bpw-4x-dgx-spark
BUILD=1; [[ ${1:-} == --no-build ]] && { BUILD=0; shift; }
SIZES=${QUAD_SIZES:-}
if (( BUILD )); then
  rsync -a --delete ${LAB_SRC:-$HOME/projects/rexl3-inkling2-dev}/crates/ spark@${NODES[0]}:$B/lab/crates/ || exit 1
  ssh -o BatchMode=yes spark@${NODES[0]} "cd $B/lab && export PATH=/usr/local/cuda/bin:\$HOME/.cargo/bin:\$PATH CUDA_HOME=/usr/local/cuda REXL3_CUDA_ARCH=sm_121 REXL3_COMMIT=lab CARGO_TARGET_DIR=$B/target-lab && cargo +1.89.0 test --release --locked -p gb-cuda --test nccl_oneshot_quad --no-run 2>&1 | grep -E '^error' -A8 | head -20; cp \$(ls -t $B/target-lab/release/deps/nccl_oneshot_quad-* | grep -v '\.d\$' | head -1) $B/quad_bin; for ip in ${FAB[1]} ${FAB[2]} ${FAB[3]}; do rsync -a $B/quad_bin spark@\$ip:$B/quad_bin; done" || exit 1
fi
PORT=29777
run_once() {   # words...
  local words="$*" r pids=() out
  PORT=$((PORT + 1))
  for r in 0 1 2 3; do
    ssh -o BatchMode=yes spark@${NODES[$r]} "exec 9>>/tmp/rexl3-gpu.lock; flock -x 9; exec env -i HOME=/home/spark USER=spark PATH=/usr/local/cuda/bin:/usr/bin:/bin ${NCCL_ENV[*]} REXL3_NCCL_LIB=$NCCL_LIB $words REXL3_NCCL_RANK=$r REXL3_NCCL_RANKS=4 REXL3_NCCL_PEER=${FAB[0]}:$PORT REXL3_QUAD_SIZES=$SIZES $B/quad_bin nccl_oneshot_quad_latency --nocapture --test-threads=1 2>&1" > /tmp/quad-r$r.log &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "== [${words:-pinned env}]"
  grep -h -A8 "rank 0 (4 ranks)" /tmp/quad-r0.log | sed -E 's/^nccl_oneshot_quad rank 0 \(4 ranks\): //' | head -9
  grep -h -E "\(exactness\)|\(mesh exactness\)|test result" /tmp/quad-r?.log | sed -E 's/^nccl_oneshot_quad //' | head -24
  grep -l -E "panicked|watchdog" /tmp/quad-r?.log 2>/dev/null | while read f; do echo "FAILED: $f"; grep -E "panicked|watchdog" -A2 "$f" | head -4; done
}
if (( $# == 0 )); then run_once ""; else for w in "$@"; do run_once "$w"; done; fi
