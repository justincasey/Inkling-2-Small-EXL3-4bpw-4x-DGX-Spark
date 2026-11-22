# Segment A v1 (bit-preserving speed), 2026-11-21

`meta.json`: the installed engine, `a73ff7a` (engine branch `inkling2-a`, a lab build off the pinned `f4e27ca`); written by `./inkling2.sh record` before the recipe commit that holds this
record, so its `recipe.commit` is that commit's parent and its 8 uncommitted files are this commit's scripts and docs. `harness-runs/`: each kept run of the
session (`./autoresearch.sh`, `bench/runs/autoresearch-*`, gitignored) with its metrics, gates, capacity plan, memwatch and raw bench data.
`bitwise-logits.txt`: the byte-for-byte logits check. `tests.txt`: the primitive-level proofs. The numbers quoted in
`docs/exec-plans/completed/autoresearch-segment-a.md`, `docs/AUTORESEARCH-HANDOFF.md` and the README come from these files. Runs 18 to 20 and every screen went
through `scripts/safeloop.sh` (docs/SELF-IMPROVEMENT-LOOP.md): the known-good engine was restored after each (run 17 predates the wrapper).

## Full runs (11 of 11 gates each)

| run | engine | change since the previous | target_score | decode_c1 | prefill 32K / 128K / 256K |
|---|---|---|---:|---:|---|
| 17 (`harness-runs/run17-bd307f5`) | bd307f5 | fused MoE groups | 0.6158 | 50.60 | 2,098 / 1,828 / 1,543 |
| 18 | 1721743 | bf16 layer partials on the wire (1eb7142) and the 4-rank RoCE mesh for decode collectives | 0.6212 | 53.33 | 2,026 / 1,742 / 1,487 |
| 19 | 58c91f2 | K and V projections in the `w_q` launch, bf16 reduce-scatter and all-gather of the prefill layer sums, batched conv commits, drafter-chain PDL | 0.6442 | 53.69 | 2,164 / 1,867 / 1,571 |
| 20 | a73ff7a | rank-symmetric choice of the gather (oracle dump runs), nothing else | **0.6464** | **54.14** | **2,161 / 1,868 / 1,575** |

`1m-58c91f2`: cold 1,008,317-token prefill, 4 x 1,048,576-token lanes, pin and memory PASS: 830 tok/s (`1m-bd307f5`: 821 on `bd307f5`; 818 on `457d637`, in
`bench/records/2026-11-19-segment-a-final/harness-runs/1m-457d637`), decode after it 31.2 tok/s, lowest free memory 9.85 GiB (floor 8). Acceptance 0.4134 and
3.0431 tokens a round in every run: the drafter's arithmetic did not move.

The mesh costs prefill: run 18's 32K / 128K / 256K are 3.4 / 4.7 / 3.6% below run 17 (decode +5.4%). The 2026-11-19 screens (`harness-runs/mesh-screens-1119/`, `index.txt`
with each run's engine commit, mesh state and numbers) split the same way: mesh on, 9 screens, 2,008-2,046 at 32K (median 2,027; one more on-screen with an unlogged
state, 2,032); mesh off, 30 screens, 1,939-2,138 (median 2,081, 13 of 30 at 2,120 or more: the off boots are bimodal and sometimes slow for a reason not found). A "park"
variant of the mesh (queue pairs back to INIT during prefill) did not recover it: 2,027 / 2,031 / 2,038 on `9ff2f8a`, `646c7c1`, `b3e1d06` (in the index). Net score effect
of the mesh +0.9% (0.6158 to 0.6212); the cause of the prefill cost is not found. The bf16 collectives of run 19 more than pay it back.

## Screens of the new switches (`harness-runs/screen-*`, every one `PASS pin`)

Builds: `ab1` ran `489418d`; `ab2`-`ab4` ran `58c91f2` (the dev HEAD moved between the first and the second boot: see docs/SELF-IMPROVEMENT-LOOP.md).

| screen | switches | decode_c1 | prefill 32K |
|---|---|---:|---:|
| ab1 | K/V pack, RS bf16, batched commits on | 54.04 | 2,090 |
| ab2 | the same three off (all-gather off with them) | 53.19 | 2,029 |
| ab3 | all on, all-gather included | 54.03 | 2,166 |
| ab4 | the same three off | 53.41 | 2,002 |
| ag1 / ag3 | all on | | 2,164 / 2,168 |
| ag2 / ag4 | `REXL3_INKLING2_AG_BF16=0` | | 2,067 / 2,090 |

Reading: K/V pack and batched commits +1.4% of decode (the off arm, 53.30, equals run 18's 53.33: the drafter-chain PDL is worth nothing measurable);
bf16 reduce-scatter +3.1% of 32K prefill, bf16 all-gather another +4.4%, 2,015 to 2,166 together. The prefill gain is larger than the bytes alone predict:
NCCL's send/recv kernels hold SMs while they run, and half the bytes is half the time they hold them `[INFERENCE]`.

## Not run, not done

- The 1M prefill was run on `58c91f2`, not on `a73ff7a` (the difference is the oracle-dump predicate, which no benchmark path reaches).
- Not built (each under 1% by estimate): fusing the expert-remap into the grouping kernel, splitting `markov_step` over eight blocks, walk truncation, packing the
  layer partials straight from their producers.
