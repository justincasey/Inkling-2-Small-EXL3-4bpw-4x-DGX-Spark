# Segment A (bit-preserving speed), final state

`meta.json`: the installed final engine (72d8d10, a lab build off the pinned f4e27ca). `harness-runs/`: each kept harness run of the session
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

All eleven runs: 11 of 11 gates. `1m-*`: cold 1,008,317-token prefill, 4 x 1,048,576-token lanes, pin and memory PASS (793 tok/s on
both engines); the 1M runs of the later engines `1m-b1c4de1` and `1m-72d8d10`: 814 and 815 tok/s, pin and memory PASS. Run 12's decode_c1 of 47.52 is a good boot: nine boots of that build gave 46.3-47.7 (mean 46.9), see the exec plan.
