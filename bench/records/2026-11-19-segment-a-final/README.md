# Segment A (bit-preserving speed), final state

`meta.json`: the installed final engine (457d637, a lab build off the pinned f4e27ca). `harness-runs/`: each kept harness run of the session
(`./autoresearch.sh`, `bench/runs/autoresearch-*`, gitignored) with its metrics, gates, capacity plan, memwatch and the raw bench data;
`bitwise-logits.txt`: the byte-for-byte logits check. The numbers quoted in `docs/exec-plans/active/autoresearch-segment-a.md`,
`docs/AUTORESEARCH-HANDOFF.md` and the README come from these files.

| run | engine | change | target_score | decode_c1 | prefill 32K / 128K / 256K |
|---|---|---|---:|---:|---|
| 02 | f4e27ca | baseline | 0.4422 | 42.18 | 1,298 / 1,199 / 1,080 |
| 03 | 8bd0ed5 | prefill all-reduce as reduce-scatter + all-gather | 0.5071 | 42.14 | 1,709 / 1,528 / 1,339 |
| 04 | 50bdb1f | parallel MoE plan | 0.5160 | 42.08 | 1,772 / 1,577 / 1,375 |
| 05 | e0ee556 | Curve draft budget | 0.5248 | 43.41 | 1,777 / 1,573 / 1,372 |
| 06 | f3bf927 | two-lane prefill | 0.5450 | 43.33 | 1,920 / 1,684 / 1,441 |
| 07 | 2a6ba1d | thin lin_narrow tiles | 0.5520 | 44.37 | 1,923 / 1,686 / 1,439 |
| 08 | dc9b356 | HEAD after the env-gated logits-only dump (no performance change) | 0.5519 | 44.35 | 1,923 / 1,686 / 1,443 |
| 09 | 6f04cbc | MoE prefill A fragments one step ahead (MCG experts) | 0.5572 | 44.25 | 1,965 / 1,716 / 1,465 |
| 10 | b1c4de1 | staged wide linear, L2-grouped blocks | 0.5718 | 44.27 | 2,068 / 1,798 / 1,524 |
| 11 | a8a7602 | PDL on the MoE and dense decode chains (default on) | 0.5820 | 45.97 | 2,063 / 1,793 / 1,519 |
| 12 | 72d8d10 | L2 warm of w_q and the router gate in decode windows | 0.5917 | 47.52 | 2,063 / 1,794 / 1,518 |
| 13 | 23eb6f3 | MoE prefill grids sized by the chunk's picks | 0.5931 | 47.32 | 2,081 / 1,810 / 1,529 |
| 14 | a9d01bc | drafter Markov walk kernels | 0.6025 | 48.73 | 2,086 / 1,811 / 1,533 |
| 15 | 457d637 | narrow linear: 12 chunks of loads in flight | 0.6139 | 50.62 | 2,085 / 1,812 / 1,534 |
| 16 | 457d637 | the same commit on a verified clean build (the final sweep) | 0.6143 | 50.75 | 2,082 / 1,810 / 1,534 |

All fifteen runs: 11 of 11 gates. `1m-*`: cold 1,008,317-token prefill, 4 x 1,048,576-token lanes, pin and memory PASS (793 tok/s on
both engines); the 1M runs of the later engines `1m-b1c4de1`, `1m-72d8d10`, `1m-23eb6f3` and `1m-457d637`: 814, 815, 818 and 818 tok/s, pin and memory PASS. Run 12's decode_c1 of 47.52 was a good boot: eleven boots of that build gave 46.3-47.7 (mean 47.0), see the exec plan. Decode of the final engine is steadier: six boots (runs 15 and 16 and four screens) gave 49.5-50.7, mean 50.2. Runs 14 to 16 were built after the first attempt of the final checks showed that `inkling2.sh build` could link stale crates (see the end of `bitwise-logits.txt` and the handoff's traps); run 16 and the 1M run are on builds that recompiled all ten gb-* crates.

The engine worktree's HEAD after the benchmarked `72d8d10` is `3bddc4d`: it adds only the PDL ordering probe (`csrc/pdl_probe.cu`, `tests/pdl_order.rs`) to the
production code of `72d8d10`; the later decode screens (the interleaved PDL off / on set) ran on it.
