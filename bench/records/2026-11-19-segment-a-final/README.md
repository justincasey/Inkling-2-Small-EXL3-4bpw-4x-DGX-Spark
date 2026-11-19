# Segment A (bit-preserving speed), final state

`meta.json`: the installed final engine (dc9b356, a lab build off the pinned f4e27ca). `runs/`: each kept harness run of the session
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
| 08 | dc9b356 | final HEAD (adds the env-gated logits-only dump) | 0.5519 | 44.35 | 1,923 / 1,686 / 1,443 |

All seven runs: 11 of 11 gates. `1m-*`: cold 1,008,317-token prefill, 4 x 1,048,576-token lanes, pin and memory PASS (793 tok/s on
both engines).
