# Execution plan: autoresearch, Segment A (bit-preserving speed)

Session: `autoresearch/session-20261118`, metric `target_score` (see `autoresearch.sh`, `docs/AUTORESEARCH-HANDOFF.md` §3a).
Engine candidates: worktree `~/projects/rexl3-inkling2-ar` (branch `inkling2-ar` off `inkling2` @ `f4e27ca`).

## Status

- [x] harness built, references pinned from the unmodified engine, baseline logged (segment 1 run #1, score 0.4431, 11/11 gates;
      segment 2 baseline run #2 after the 256K change, 0.4422, 11/11 gates)
- [x] profile of one decode window and one 32K prefill on all four ranks (below)
- [x] byte-exact policy decision (user, 2026-11-18): Segment A only
- [x] user, 2026-11-18: prefill work and routine runs stop at 256K
- [x] profile of a cold 256K prefill (below): the all-reduce path is 41.6%, attention 21.4%
- [x] Segment A changes kept: runs #3 to #7, then #9 and #10 after the second kernel pass and #11 and #12 after the third (decode; Results, below); run #8 confirmed the HEAD between them;
      numbers are in `bench/records/2026-11-19-segment-a-final/`
- [x] byte-for-byte logits check of `f4e27ca` against the engine, lanes off and on, on `dc9b356` and again on the final `b1c4de1` (record, `bitwise-logits.txt`)
- [x] 1M benchmark on `44cd901` (pin and memory PASS, 793 tok/s); to be re-run on the final engine of the second kernel pass
- [ ] open items: "What is left" below (none started)

## Baseline (unmodified `f4e27ca`, eager, clocks locked at 2200 MHz)

| Metric | Value | Target |
|---|---:|---:|
| `decode_c1_tok_s` (mean of four prompt types) | 42.2 | 70 |
| per type: prose / structured / json / code | 22.0 / 26.4 / 53.2 / 67.3 | |
| DSpark acceptance / tokens per round | 0.307 / 3.13 (prose 0.14 and 1.96, structured 0.21 and 2.44, json 0.58 and 5.02, code 0.76 and 6.32) | |
| `prefill_32k_tok_s` | 1,300 (25.2 s) | 4,000 |
| prefill 128K / 256K / 1M | 1,199 (109 s) / 1,080 (243 s) / 688 (not routine) tok/s | 3,200 / n/a / 1,700 (mine) |
| decode c4 / c8 (two waves of 4 slots) | 52.1 / 57-64 (c8 is noisy) | 100 at c8 (mine) |
| decode at 32K / 128K / 256K / 1M context | 40.4 / 31.8 / 37.6 / 28.1 (256K: the reply stops at 125 tokens) | |
| `target_score_raw` | 0.443 | 1.0 |

Spread over repeats: `decode_c1` 42.06-42.44 (0.9%, seven runs), `prefill_32k` 1,297.8-1,304.2 (0.5%, five runs).

Segment 2 (user, 2026-11-18): routine runs stop at 256K; the 1M prefill is a separate benchmark for later
(`PHASES=prefill-1m ./autoresearch.sh`). A full run is 29 minutes instead of 49. **The profile below was taken at 32K.** At 128K
the engine does 1,199 tok/s and at 256K 1,080 against 1,300 at 32K, so attention (global layers, int8 dequant per tile) takes a
growing share of the prefill; the collectives' cost per token does not grow. Profile a cold 256K prefill before choosing between
the collective work (item 1) and attention work.

## Profile

Method: `PROFILE_NSYS="--delay=135 --duration=60 --trace=cuda,nvtx --sample=none --cpuctxsw=none" ./inkling2.sh start` (every rank
under `nsys profile`, reports in `BASE_DIR/prof/`), then `bench/tools/profile_driver.py T0 135` (a warm pass outside the window,
then prose 128 tokens, code 128 tokens, a cold 32K prefill with 16 tokens), `nsys export -t sqlite`, and
`bench/tools/nsys_phases.py RANK.sqlite decode:0:7.9 prefill:8.2:32.8` (phases are seconds from the rank's first kernel; only rank
0's prefill slice is aligned, the decode slice is the same on all four). Engine `f4e27ca`, graphs do not exist.

### Decode (7.9 s window: 76 prose rounds + 18 code rounds = 94 verify rounds, 8 rows each, plus DSpark proposals)

Rank 0; ranks 1-3 are within 3% on every line except MoE (rank 3: 2,417 ms, rank 0: 2,669 ms) and NCCL (rank 3: 1,714 ms, rank 0: 1,590 ms).

| What | ms in window | share of 7,900 ms | ms per round | kernels |
|---|---:|---:|---:|---:|
| `moe_gemv_kernel` (routed experts, EXL3) | 2,669 | 33.8% | 28.4 | 7,968 |
| `lin_narrow_kernel` (bf16 narrow-row linears) | 2,102 | 26.6% | 22.4 | 22,240 |
| `ncclDevKernel_SendRecv` (all-reduces, includes waiting for peers) | 1,590 | 20.1% | 16.9 | 9,562 |
| GPU idle (host gaps between kernels) | 425 | 5.4% | 4.5 | |
| everything else (drafter, attention, mix, convs, sums, norms) | 1,114 | 14.1% | 11.8 | about 85,000 |

- 124,269 kernels in 94 rounds: about 1,320 launches and 100 collectives per round (the code says about 1,500 and 220 kernels
  counting drafter commit and `apply_pending`). Host launch calls cost 1.2 s of the 7.9 s, but the GPU is idle only 5.4% of
  the time: **CUDA graphs can win at most about 5%** on this engine as it stands.
- `lin_narrow_kernel`: 22.4 ms per round for the bf16 attention weights of one rank (9.42 GB / 4 = 2.36 GB) is about 105 GB/s, 44% of
  the 237 GB/s read roofline `[INFERENCE: assumes the kernel's launches are those weights]`.
- NCCL: 166 us per all-reduce on average for 147 KB payloads. Rank 3 does less MoE work than rank 0 and waits longer, so part of
  this is expert-load skew between ranks (lockstep at each all-reduce), not wire time `[INFERENCE]`.
- Round time (request wall minus its time to first token, over its rounds): prose 89 ms for 1.96 tokens (22 tok/s: the handoff's
  unrecorded serial rate is about 21, so speculation gains nothing there), code 93 ms for 6.3 tokens. "Per round" in the table is the
  7.9 s window over 94 rounds, so it includes the three request prefills (about 0.6 s).

### Prefill 32K (rank 0, 24.6 s, 8 chunks of 4,096 rows)

| What | ms | share | launches |
|---|---:|---:|---:|
| `ncclDevKernel_SendRecv` (13.5 ms each, 75.5 MB partials to 3 peers) | 10,115 | 41.2% | 750 |
| `tp_sum_slots_vec_k` (rank-order sum of the 4 partials) | 2,310 | 9.4% | 750 |
| MoE `gateup_kernel` + `down_kernel` | 3,103 + 1,696 = 4,799 | 19.6% | 1,128 + 1,125 |
| `lin_wide_kernel` (bf16 linears) | 2,554 | 10.4% | 2,156 |
| attention (`attn_kernel`) | 1,150 | 4.7% | 375 |
| `mix_kernel` / `plan_kernel` (MoE grouping, 1 CTA, 2.2 ms each) | 848 / 835 | 3.5% / 3.4% | 751 / 383 |
| `rot_in`, `conv`, `combine_f32`, `prologue`, `mix_norm` | 516 + 507 + 365 + 321 + 127 | 7.3% | |

GPU idle 0.3%: prefill is communication and compute bound, not launch bound. **Half of it (12.4 s) is the all-reduce path.**
The collective is "send the full partial to every peer, then sum in rank order": 226 MB out and in per rank per call. The same bits
come from a reduce-scatter (each rank receives only its quarter of every peer's partial, sums it in rank order) followed by an
all-gather: half the bytes, a quarter of the sum work.

### Prefill 256K (user: the largest context of a routine run), rank 0, 240 s, 64 chunks of 4,096 rows

Same method, `profile_driver.py T0 135 262144` with `--duration=330`; one cold 262,144-token prompt, 16 tokens, 243.7 s request.
Ranks 1-3 within 1.5% of rank 0 on every line (rank 3 does 12% more MoE and waits 7% less in NCCL).

| What | s | share | per call |
|---|---:|---:|---:|
| `ncclDevKernel_SendRecv` | 81.1 | 33.8% | 13.4 ms, flat from the first chunk to the last |
| `tp_sum_slots_vec_k` | 18.6 | 7.7% | 3.07 ms, flat |
| `attn_kernel` (all 48 layers' launches) | 51.3 | 21.4% | mean 2.7 ms per launch in the first chunks, 31.4 ms in the last (about 1.5 s of each 4.4 s chunk) |
| MoE `gateup_kernel` + `down_kernel` | 24.9 + 13.6 = 38.5 | 16.1% | flat |
| `lin_wide_kernel` | 20.7 | 8.6% | flat |
| `plan_kernel` + `mix_kernel` | 6.7 + 6.9 | 5.7% | flat |
| `rot_in`, `conv`, `combine_f32`, `prologue`, `mix_norm`, ... | 4.2 + 4.1 + 3.0 + 2.6 + 1.1 | 6.0% | flat |

GPU idle 0.2%. Per-token time grows from 0.77 ms (32K) to 0.93 ms (256K) because attention grows with the context and everything
else is flat. The all-reduce path is **99.7 s of 240 s (41.6%)**: item 1 saves about half the NCCL bytes and three quarters of
the sum work, 40.6 s + 14 s = about 54 s `[INFERENCE: assumes NCCL time scales with bytes]`, so 256K goes from 1,080 to about
1,400 tok/s and 32K from 1,300 to about 1,800. After it, attention is the largest single kernel at 256K (51 s of 185 s).

### Decode on the final engine (`dc9b356`, ranks 2 and 3, 103 verify rounds in 7.89 s, nsys `--trace=cuda`)

76.7 ms a round of wall time, GPU idle 6.1-6.2% (launch gaps). Per round, rank 2 / rank 3: `lin_narrow` 20.5 / 20.3 ms, `moe_gemv` 19.2 / 18.5,
`ncclDevKernel_SendRecv` 20.0 / 21.0 (includes waiting for the other ranks), other kernels 9.9 / 9.8, drafter walk (`markov_*`) 2.4 / 2.3.
`lin_narrow` shapes (rank 1): (410,1) 42.5 launches 160 us 6.8 ms; the unembed (3,141,1) 1.8 launches 2,255 us 4.0 ms (463 MB in 2.25 ms:
205 GB/s); the 64-thread `w_o` grid (576,1) 46.8 launches 78 us 3.7 ms (79 us before the thin tiles: no gain); (576,1) 8.8 launches 319 us
2.8 ms; the small shapes (32,1) 70.8 launches 18 us (27 us before), (48,1) 24 us (31), (33,1) 18 us (27): the thin tiles saved about 1.6 ms a round.
Not comparable round for round with the `f4e27ca` profile above: the Curve budget changes how many rows a verify round carries.
The rank-skew share of the collectives was **not** separated: rank 0's kernel records were lost (below) and the per-collective durations of
ranks 1-3 align across ranks with a correlation of only 0.16-0.41.

nsys sessions of the lane engine lost rank 0 twice and completed once: with `--trace=cuda,nvtx` (prefill-only driver) during the 16-token decode
after the 32K prefill, with `--trace=cuda --cuda-memory-usage=false` (prefill-only driver) not at all, and with `--trace=cuda ...` (prose,
code, then 32K) during decode windows, before the 32K prefill. The loss came 148-162 s after launch, not at `--delay + --duration` (195 s);
no fatal line, the guard did not act. Cause not found. Without nsys the lane engine lost no rank in about 15 boots (runs #6-#8, screens,
two 1M runs) nor in a 16-cycle stress (cold prefills of 4,100-33,000 tokens, each followed at once by four concurrent decodes; every
recurring prompt gave the same text). The dying rank's kernel records are lost with it: use the other ranks' files.

## What the profile changes

The handoff ranked CUDA graphs first for decode (the RED-SNOW precedent). Here the GPU is 94.6% busy, so graphs are worth at most 5%.
Revised order for Segment A (every item keeps arithmetic order, so `pin`, `spec`, `cache`, `chunk` must stay green):

| # | Change | Where the time is | Estimate (upper bound, not measured) |
|---|---|---|---|
| 1 | Prefill: all-reduce as reduce-scatter + all-gather with the same rank-order sum | NCCL 10.1 s + sum 2.3 s of 24.6 s | about -6 s, prefill_32k about 1,300 to 1,700 |
| 2 | Prefill: pipeline the chunk in row halves so a layer's collective overlaps the next half's MoE/linears (row invariance: `inkling2_attn.rs:17`, `inkling2/mod.rs:2`) | remaining comm | up to another -5 s: about 2,400 tok/s |
| 3 | Decode: `lin_narrow_kernel` to about 85% of read bandwidth, same accumulation order per output | 22.4 ms of 84 per round | about -10 ms per round, +13% |
| 4 | Decode: cost-priced draft length (wire `BudgetPolicy` with a measured 1..8-row cost table, truncate the Markov walk at k): output-neutral per `crates/engine/tests/budget.rs:77` | prose and structured run at the serial rate with an 8-row verify | prose/structured +10-20% `[INFERENCE]`, code/json unchanged |
| 5 | Decode: collective latency (166 us each, 100 per round): Nemotron's RoCE `TpComm` (`roce_*.cu`) or NCCL settings, same rank-order sum | 16.9 ms per round | about -8 ms per round if wire time is ~80 us |
| 6 | Decode: CUDA graphs for the verify window and the proposal (`window_graph.rs`, `Rows::set_bounds` already exist) | 4.5 ms per round of gaps | at most +5% |
| 7 | Prefill: `plan_kernel` (1 CTA, 2.2 ms, 383 launches) parallelized | 0.8 s | about -0.7 s |
| 8 | Prefill, 256K: global attention (`attn_kernel`, int8 dequant per tile): stage dequantized K/V tiles in shared memory per CTA group without changing the online-softmax order | 51.3 s of 240 s at 256K (21%); in the last chunks 1.5 s of each 4.4 s chunk, about 190 ms per global layer `[INFERENCE: local layers are cheap]` | up to -20 s at 256K if the kernel reaches 2x `[INFERENCE]`; grows with context |

Measured (run #12, final engine `72d8d10`): `target_score` 0.592, decode 47.5 tok/s (a good boot: 46.3-47.7 over nine boots, mean 46.9), prefill_32k 2,063, 1M prefill 815 (run #10 on `b1c4de1`: 0.572 with decode 44.3 and prefill 2,068; run #8 on `dc9b356`: 0.552). The original estimate of a Segment A ceiling (decode 55-60, prefill_32k 2,400-2,600, score 0.65-0.75) was not reached: the estimates above were upper bounds from the first profile, and items 2, 3 and 5 paid less than they assumed; the second kernel pass (runs #9 and #10) found +7% of prefill in two kernels the first pass had called near their peak, and the third (runs #11 and #12) +6% of decode in launch overlap and L2 warming, which the first pass had bounded at 2%.

Status of each item (2026-11-19):

| # | Status | What was measured |
|---|---|---|
| 1 | done, run #3 | prefill_32k 1,298 to 1,709 |
| 2 | done, run #6 (two lanes) | 1,777 to 1,920 (+8%, not the +30% estimated): the group is paced by rank 3 (see "Lane profile") |
| 3 | done in part, run #7 (thin tiles) | decode +2.4%; the narrow shapes that remain were not re-derived against a bandwidth ceiling except the unembed (205 GB/s) |
| 4 | done, run #5 (Curve budget); walk truncation and the estimator constants not done | decode +3%; truncation about +1.2% and estimator +0.6% were estimated, not built |
| 5 | NCCL settings tried, nothing moved (below); a 4-rank RoCE transport is **not attempted** | the "-8 ms" estimate is unsupported: the 155 us mean (72 us median) per collective includes time waiting for the other ranks, and its floor was not isolated |
| 6 | superseded by PDL, run #11 | the 2% bound for graphs was wrong: PDL on the dense and MoE chains gave +3.8% of decode; graphs themselves were not tried (the drafter chain and the glue are the launches left) |
| 7 | done, run #4 | prefill_32k 1,709 to 1,772 |
| 8 | **open, not attempted** | attention is 1.3 s of 17.1 s at 32K (7.6%) and about 26% at 256K, where the score does not look; its efficiency against a roofline was not measured |
| new | partly done, run #9; the rest measured and dropped | MoE prefill kernel: A prefetch for MCG +4.5% of a layer; deeper weight prefetch, a fused group list, a joint MoE for both lanes (+0.5%), 8,192-row chunks: no gain or not allowed (Results, below) |
| new | done, run #10 | `lin_wide`: staged kernel and L2-grouped block order, 32K prefill +5.5% |

The open items are upper-bounded, not measured: if every one paid its whole estimate (walk truncation and estimator +1.8% decode,
narrow-kernel bandwidth up to +4%, a 4-rank transport up to +5%, attention staging up to +4% of 32K prefill, prefill elementwise
fusion a few %, MoE prefill kernel up to +7% of 32K prefill) the score would be about 0.58 to 0.61 `[ESTIMATE]`. Reaching 70 and 4,000 needs more than bit-preserving work: decode at serial speed is bounded near 46 tok/s by the bf16 attention
bytes (`docs/ROOFLINE.md`), and 4,000 tok/s at 32K would need the MoE prefill (6 s of 17 s) and everything else to halve together.

Correction (2026-11-19): an earlier version of this plan and of the session notes called the MoE prefill and `lin_wide` "near tensor peak".
That was not measured. From the 32K profile the MoE `gateup` + `down` take 4.8 s over 376 chunk-layers on rank 0, 12.8 ms each, against
about 5.6 ms of math per 4,096-row chunk-layer at 93 TFLOP/s dense (`docs/AUTORESEARCH-HANDOFF.md` section 2): about 44%. The kernel
(`moe_prefill.cu`) decodes each expert's EXL3 trellis once per group of up to 64 rows (about 85 pairs reach an expert per chunk, so most
experts decode about twice) and its header states that the group size cannot change a bit. `lin_wide` was never compared with a roofline.

## Decision taken by the user: Segment A only (2026-11-18). Kept for the record: what Segment B would buy

Segment B changes bits and needs: the `oracle` gate (not wired in the harness), a new pin, a new segment (old comparisons void).
What it would buy, from the profile:

- tensor-core MoE prefill for this pack's codebook (`glm_p2_moe.cu` variant): the 4.8 s MoE part of the prefill, 19.6%;
- requantizing the BF16 attention weights to EXL3 in a new pack: `lin_narrow` 22.4 ms per round to roughly 6-8 ms (4 bpw is a quarter
  of the bf16 bytes; 52% of a token's weight bytes), the serial ceiling from 46 to about 70 tok/s (handoff §5);
- another attention merge order: attention is only 4.7% of the prefill at 32K and 1.6% of a decode round (low value there; its
  share at 256K is not profiled yet).

The loop stays in Segment A; ask again before any Segment B work.

## What is left (2026-11-19, after the second kernel pass)

Nothing below is started. Each needs a screen first (`PHASES=warm,decode`, `warm,prefill-32k`) and a full run only to keep it; each is worth a
few percent of one metric at best, which is a point or two of `target_score`.

1. Decode: walk truncation (`BudgetPolicy::Plan` already asks the drafter for `ProposeRequest.max` drafts; the Inkling backend runs all 7 steps
   and the round cost has no draft-step cost) with the Curve estimator constants (decay 0.97, prior weight 2; +0.5-0.6% in a sweep, noise
   level). The walk is about 0.39 ms a step (`markov_logits` 252 us + `markov_step` 129 us), so a 3-step average on the prose and structured
   streams saves about 1.2 ms of a 77 ms round: about +0.8% of decode overall `[ESTIMATE]`.
2. Prefill: the wide `w_o` linear (75 MB of fp32 output against 1,536 of K; flat under the staged kernel) and the output store; the
   `lin_wide2_kernel` still keeps 48 bytes of stack at the 128-register cap of two blocks an SM.
3. Prefill: attention at 32K is 1.35 s on rank 3 in 256 blocks of one block an SM (5.3 waves, up to 11% of tail); at 128K and 256K it is
   larger and outside the score. Its grid is head-major already. First measure it against a roofline.
4. Decode: a 4-rank collective transport. First isolate the no-compute one-shot latency at 147 KB over this fabric
   (`crates/cuda/tests/nccl_oneshot_pair.rs` `latency` phase does this for two nodes): the saving is bounded by what the NCCL one-shot costs
   above that floor, and the time other ranks spend waiting is not recoverable by any transport.
5. Prefill: the memory-bound elementwise kernels on the slowest rank (`mix` 1.05 s, `rot_in` 0.63, `tp_sum` 0.58, `conv` 0.53, `combine` 0.40,
   `prologue` 0.37 on rank 3): each is near its DRAM floor on its own; only fusing passes would move them.

Before any final claim run the 1M benchmark (`PHASES=prefill-1m`) and the byte-for-byte logits check
(`bench/records/2026-11-19-segment-a-final/bitwise-logits.txt`: `DUMP_DIR=... EXTRA_ENV=REXL3_INKLING2_DUMP_LOGITS_ONLY=1`, `cmp` against a boot
of `f4e27ca`). Lab branches of the engine worktree that hold measured-and-dropped code: `lab/fused-groups`, `lab/joint-moe`.

## Results (segment 2; engine commits on branch `inkling2-ar` of `~/projects/rexl3-inkling2-ar`)

| Run | Engine | Change | `target_score` | decode_c1 | prefill 32K / 128K / 256K | Gates |
|---|---|---|---:|---:|---|---|
| #2 | `f4e27ca` | baseline | 0.4422 | 42.18 | 1,298 / 1,199 / 1,080 | 11/11 |
| #3 | `8bd0ed5` | prefill all-reduce as reduce-scatter + all-gather for messages >= 2 MiB (`NcclComm::rsag_sum_f32`, `GroupComm::sum`): same rank-order sum, same bits, half the network bytes | 0.5071 | 42.14 | 1,709 / 1,528 / 1,339 | 11/11 |
| #4 | `50bdb1f` | MoE plan of the prefill from the multi-block planner already in `moe_prefill.cu` (`PrefillScratch::use_parallel_plan(true)`, the Qwen Flash-Next setting): same plan, 2.2 ms a layer less | 0.5160 | 42.08 | 1,772 / 1,577 / 1,375 | 11/11 |
| #5 | `e0ee556` | draft budget `BudgetPolicy::Curve` over a measured round-cost grid (`engine.rs` `round_cost`; one stream measured, two or more flat = every draft verified as before): prose 22.0 to 24.8, structured 26.4 to 28.5 tok/s | 0.5248 | 43.41 | 1,777 / 1,573 / 1,372 | 11/11 |
| #6 | `8ee0fe3` | a prefill chunk of 1,024+ rows as two lanes (`Inkling2Rank::forward_lanes`): row views of the step buffers, layers interleaved A.pre B.pre A.mid B.mid, the four collectives of a layer on a comm stream under the other lane's compute | 0.5450 | 43.33 | 1,920 / 1,684 / 1,441 | 11/11 |
| #7 | `44cd901` | `lin_narrow_kernel<NT>`: 8-column (thin) blocks when the 16-column grid would run under 4,096 warps (the decode small-n shapes and `w_o`, 150+ launches a round); each column's accumulation order is unchanged | 0.5520 | 44.37 | 1,923 / 1,686 / 1,439 | 11/11 |
| #8 | `dc9b356` | final-HEAD confirmation after the env-gated logits-only dump: no performance change | 0.5519 | 44.35 | 1,923 / 1,686 / 1,443 | 11/11 |
| #9 | `8aa5122` | MoE prefill: the A fragments one k step ahead (`kAp`) for the 4-bit MCG experts; same operands, same bits | 0.5572 | 44.25 | 1,965 / 1,716 / 1,465 | 11/11 |
| #10 | `b1c4de1` | wide bf16 linear: `lin_wide2_kernel` (64-wide k stages, two buffers, `cp.async`, one barrier a stage) and row blocks walking the weight columns in groups of 8 (L2 reuse); same chunk order per output | 0.5718 | 44.27 | 2,068 / 1,798 / 1,524 | 11/11 |
| #11 | `a8a7602` | programmatic dependent launch (PDL) on by default for the MoE decode chain (mask bit 4, existing) and a new bit 8 for the dense chain (narrow linear, mix, conv, attention, router select, vocabulary, all-reduce sum); each such kernel waits for its predecessor before any access; timing only | 0.5820 | 45.97 | 2,063 / 1,793 / 1,519 | 11/11 |
| #12 | `fbc7a1a` (tree of `72d8d10`) | decode windows load the next layer's first 16 MB of `w_q` and the layer's router gate through L2 on a second stream during the two all-reduces; no value changes | 0.5917 | 47.52 | 2,063 / 1,794 / 1,518 | 11/11 |

Lane profile (rank 0, cold 32K, 17.2 s, `--trace=cuda`): comm kernels 10.3 s of which 84% run under compute (1.7 s exposed); compute is 13.55 s on rank 0 and 16.0 s on rank 3 (experts E mod 4 give rank 3 about 12% more MoE rows), so the group is paced by rank 3. Splitting the rows makes the MoE kernels about 25% slower in total (gateup 2.0 ms per 2,048-row lane against 2.75 ms per 4,096-row chunk). Prefill is within about 7% of rank 3's compute: only kernel work helps from here.

Thin tiles (run #7): decode `lin_narrow` shapes were 16 to 24 blocks (256 to 384 columns) and a 64-thread `w_o` grid at 79 us (about 120 GB/s, half of peak): too few warps in flight. A block of one n8 tile doubles the blocks with the same per-column accumulation (the invariance test covers narrow == thin == wide at every row count and output mode; goldens pass). Screens: threshold 1,024 warps 44.07, 4,096 warps 44.18 (noise-level better); full run 44.37 (43.33 before), every prompt type +1.7 to 2.5%.

Decode kernels left after run #7 (profile of `f4e27ca`, rank 0, per round, before the thin tiles; the final-engine numbers are in the profile section): MoE gemv 27 ms (gate/up 465 us a layer, down 226 us; not compared with a bandwidth bound: the bytes a layer reads depend on how many distinct experts the verify rows reach, which was not counted), the 100 one-shot all-reduces 15.6 ms (155 us each; waiting for the other ranks is included and was not separated), `lin_narrow` 22 ms (unembed 4.1 ms, 205 GB/s), drafter walk `markov_logits` + `markov_step` 2.4 ms (6.2 steps a round), launch gaps of 2.6 us: 315 ms of 7.9 s.

1M check of the lane engine (`f3bf927`, `PHASES=prefill-1m`, run `bench/runs/autoresearch-20261119-015435`): pin PASS at 1,008,317 tokens (byte-exact), memory PASS (rank 0 minimum 9.53 GiB against the 8 GiB floor), capacity 4 lanes at 1,048,576 tokens; prefill 1,271.9 s = 793 tok/s (baseline 688), decode after it 27.3 tok/s. The same check on the final engine `44cd901` (run `bench/runs/autoresearch-20261119-025735`): pin PASS, memory PASS (minimum 9.83 GiB normalized), 4 lanes, 793 tok/s, decode 27.55 tok/s. Run #7's installed build was `2a6ba1d`, the same tree as `44cd901` (the commit was amended for its message only).

Decode idle (rank 0, 7.9 s of decode): 425 ms of gaps between kernels, 315 ms of them in 121,000 gaps under 5 us and under 2 ms of host-side gaps above 100 us, so CUDA graphs are worth at most about 2% of a round.

Kernel findings of the second pass (ncu with root counters, `sudo -n`, on the real-layer benches `bench_real_moe_layer` and `bench_inkling2_ops`; both are `#[ignore]`, run on rank 0 with `REXL3_INKLING2_DIR` and `REXL3_INKLING2_RANK=0`):
- MoE prefill `gateup` (4-tile groups): tensor pipe 46%, L2 47%, issue slots 30%, one 8-warp block an SM (224 registers, 82 KB shared): latency-bound, not decode-bound (removing the trellis decode saved 9%, the block reduce 3%, both within a noisy bench). Prefetching the A fragments one step ahead for the MCG codebook (the other codebooks had it) took a real layer of 4,096 rows from 18.7-19.4 to 17.9 ms: run #9. Weight prefetch deeper than one step (per group size, at 2,048 and 4,096 rows) did nothing. `rot_in` is near its DRAM bound (about 230 MB moved in 1.3 ms); a grid-stride launch took 1.43 to 1.31 ms: dropped.
- In lane mode (2,048 rows a lane, about 42 pairs an expert) the MoE reads each expert's weights twice a chunk-layer: about 2.8 GB, 11.6 ms at 237 GB/s of the 15.9 ms the gate/up and down kernels take; the lane penalty on the MoE is about 4.4 ms a chunk-layer. A fused group list (one launch, an expert's groups adjacent; branch `lab/fused-groups` of the engine worktree) gained 0.5-0.9 ms in the 4,096-row bench and nothing in the engine (1,965.6 against 1,966.9), because a lane has about one group an expert. A joint MoE for both lanes (one weight pass) would save about 4.4 ms and expose about 2.5 ms more of the all-reduces: an estimated +0 to +5% of prefill, not built. 8,192-row chunks would halve the weight passes, but the engine caps rows at 4,096 and the extra buffers (about 2 GB) would come out of the prefix cache and fail the memory gate.
- `lin_wide`: 3.10 s of the 32K lane prefill on rank 0, 1.87 s of it one launch a layer-lane of n=3,328 x k=4,608 at 23 TFLOP/s. In isolation that shape ran at 38 TFLOP/s against 50-60 for the 1,536-column shapes: the plain grid order read the 30.7 MB weight matrix once per row block (past L2). Row blocks walking the weight columns in groups of 8 took it from 1,649 to 1,184 us (2,048 rows); staging alone gave -15% on the q-like shapes and -25% on the router shape, nothing on `w_o` (75 MB of fp32 output): run #10.
- Shared code: `moe_prefill.cu` and `exl3_grouped.cuh` also serve other families, and `kAp` now also covers 4-bit MCG experts. After the change the shared tests of those paths pass on rank 0: `moe_prefill` (21 with the new `four_bit_mcg_prefill_equals_the_decode_path`, which instantiates 4-bit MCG through the shared `gb_moe_pf_gateup` / `gb_moe_pf_down`; the older tests use the MUL1 codebook only), `moe_owned`, `moe_subbanks`, `glm_p2_moe` and the CPU plan test. No other family's GPU tests are part of the harness. The wide-linear kernels are Inkling-only (`inkling2_linear.cu`); their invariance test has two 1,157-row shapes (a full group of 8 row blocks and a tail in `wide_block`'s order) compared with the narrow kernels, which have no such order.
- A joint MoE for both lanes (`moe_joint`: one MoE call over the chunk's rows between the per-lane router and the per-lane all-reduces; branch `lab/joint-moe`, byte-exact by the row-invariance contract) measured 2,083 tok/s against 2,068-2,072: +0.5%, within noise. The weight-pass saving (about 4.4 ms a chunk-layer) is cancelled by the exposed all-reduces: AR2 of the first lane has nothing left to hide behind and AR1 of the second hides only behind the first lane's conv/mix/router.
- The L2 hit rate of the 4-tile gate/up kernel is 85% and its DRAM read is about 160 GB/s of 273: neither the tensor pipe (46%), L2 (47%) nor DRAM saturates; the kernel is limited by latency at 8 warps an SM.
- Decode MoE gemv at 8 rows: about 111 MB in 0.43 ms, about 95% of the DRAM peak: nothing to win.

Third pass (decode):
- Launch overlap was worth far more than the idle gaps suggested. The launch gaps in the decode profile were 2.6 us each, 315 ms of 7.9 s, and I had put CUDA graphs at no more than 2% of a round; PDL for the MoE chain alone (the `REXL3_GLM_PDL=4` that already existed, timing only) gave +0.6-1.0%, and with the dense chain (1,138 of the 1,301 launches of a round) +3.5-3.9% (46.04 and 46.20 against 44.47 in screens, 45.97 in the full run). The remaining 163 launches are the drafter chain and shared glue. Each converted kernel starts with `pdl_wait(); pdl_trigger();` and its launch goes through `gbp_pdl::launch_pdl`; the mask is set by the rank (`use_pdl_unless_env_says_otherwise`, `REXL3_GLM_PDL` wins, 0 turns it off). The kernel tests (ops, attention, MoE, the dspark and vision goldens) were run with the mask off and with every bit on.
- Warming L2 during the all-reduces. Decode is DRAM-bound on weights, and during an all-reduce (about 100-150 us, twice a layer) the DRAM is idle. L2 is 24 MiB; a re-read of up to 16 MB runs at 1.0-1.3 TB/s against 0.25 TB/s from DRAM. In isolation (`bench_l2_prefetch_before_a_wait`: an all-reduce stand-in of 150 us, then the 3,328 x 4,608 linear at 8 rows) loading 8 / 16 / 24 MB of the weights on a second stream took the sequence from 289 to 254 / 221 / 215 us. `prefetch.global.L2` and `cp.async.bulk.prefetch.L2` do nothing on this part, and loads whose results are dead are removed by ptxas: the loads must be consumed. In the engine the gain is about +1.2-2% of decode, much less than the isolated 70 us a layer, and it varies from boot to boot: decode_c1 of this build over nine boots 46.29-47.69 (mean 46.9; 46.52, 46.61, 46.45, 47.69 in screens with the router warm on, 47.48, 47.30, 46.56, 46.29 with defaults, 47.52 in the full run) against 45.97-46.20 over five boots without it. 8 MB of `w_q` did nothing, 12 and 16 MB gave +0.9%, 24 MB lost 0.6% (it evicts useful lines). The guess for the spread is where the weights land against the L2 sets; it was not tested.
- Dropped in this pass: the lane split other than half (a lab knob; 40 / 45 / 50 / 55 / 60% gave 2,069 / 2,064 / 2,074 / 2,056 / 2,058 tok/s at 32K); attention row blocks launched last-first (2,072 against 2,068-2,074: no gain).

Measured and dropped: NCCL `PROTO` LL / LL128, `MAX_NCHANNELS` 2 / 8, `P2P_NET_CHUNKSIZE` 256K / 512K / 1M (all within noise); `lin_narrow` unroll 4 (+0.9%, noise); a highest-priority comm stream (no effect); the Curve estimator's decay 0.97 with prior weight 2 to 4 (+0.5 to 0.6%, noise level).

Profile of a cold 256K prefill on `8bd0ed5` (rank 0, 196 s, after run #3): attention 51.5 s (26.5%), NCCL 49.9 s (25.6%, was 81.1 s;
7.9 ms per all-reduce, two phases of 3.9 ms), MoE gateup+down 38.0 s (19.6%), `lin_wide` 20.5 s (10.6%), `mix_kernel` 6.8 s,
`plan_kernel` 6.75 s (gone in run #4), `tp_sum_slots_vec_k` 4.9 s (was 18.6 s), `rot_in` 4.2 s, `conv` 4.0 s, `combine_f32` 3.0 s.
The measured all-reduce saving at 256K is 99.7 s to 54.8 s, against the 54 s estimated.

## Not done

- Decode moved through the draft budget (run #5) and the thin `lin_narrow` tiles (run #7); the collective and lane changes are prefill-only.
- `oracle`, Segment B and other families' golden tests are not part of the harness.
