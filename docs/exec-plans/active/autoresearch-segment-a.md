# Execution plan: autoresearch, Segment A (bit-preserving speed)

Session: `autoresearch/session-20261118`, metric `target_score` (see `autoresearch.sh`, `docs/AUTORESEARCH-HANDOFF.md` §3a).
Engine candidates: worktree `~/projects/rexl3-inkling2-ar` (branch `inkling2-ar` off `inkling2` @ `f4e27ca`).

## Status

- [x] harness built, references pinned from the unmodified engine, baseline logged (run #1, score 0.4431, 11/11 gates)
- [x] profile of one decode window and one 32K prefill on all four ranks (below)
- [ ] byte-exact policy decision (user): Segment B allowed or not (below)
- [ ] first Segment A change (order in "What to do first")

## Baseline (unmodified `f4e27ca`, eager, clocks locked at 2200 MHz)

| Metric | Value | Target |
|---|---:|---:|
| `decode_c1_tok_s` (mean of four prompt types) | 42.2 | 70 |
| per type: prose / structured / json / code | 22.0 / 26.4 / 53.2 / 67.3 | |
| DSpark acceptance / tokens per round | 0.307 / 3.13 (prose 0.14 and 1.96, structured 0.21 and 2.44, json 0.58 and 5.02, code 0.76 and 6.32) | |
| `prefill_32k_tok_s` | 1,302 (25.2 s) | 4,000 |
| prefill 128K / 1M | 1,193 / 688 tok/s | 3,200 / 1,700 (mine) |
| decode c4 / c8 (two waves of 4 slots) | 52.2 / 57-60 | 100 at c8 (mine) |
| decode at 32K / 128K / 1M context | 40.3 / 31.6 / 28.1 | |
| `target_score_raw` | 0.443 | 1.0 |

Spread over repeats: `decode_c1` 0.6% (four runs), `prefill_32k` 0.02% (two runs).

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

Segment A ceiling `[ESTIMATE]`: decode about 55-60 tok/s (mean of four), prefill_32k about 2,400-2,600: `target_score` about 0.65-0.75.
Reaching 70 and 4,000 needs more than bit-preserving work: the MoE prefill (4.8 s) and linears (2.6 s) are 30% of today's prefill and
more than half of what remains after items 1-2, and decode at serial speed is bounded near 46 tok/s by the bf16 attention bytes
(`docs/ROOFLINE.md`).

## Decision needed from the user (handoff §4)

Segment B changes bits and needs: the `oracle` gate (not wired in the harness), a new pin, a new segment (old comparisons void).
What it would buy, from the profile:

- tensor-core MoE prefill for this pack's codebook (`glm_p2_moe.cu` variant): the 4.8 s MoE part of the prefill, 19.6%;
- requantizing the BF16 attention weights to EXL3 in a new pack: `lin_narrow` 22.4 ms per round to roughly 6-8 ms (4 bpw is a quarter
  of the bf16 bytes; 52% of a token's weight bytes), the serial ceiling from 46 to about 70 tok/s (handoff §5);
- another attention merge order: attention is only 4.7% of the prefill at 32K and 1.6% of a decode round: low value.

Until the user decides, the loop stays in Segment A.

## What to do first

Item 1 (reduce-scatter + all-gather, prefill) and item 3 (`lin_narrow`) are independent files and each is checked by one full run.
Item 4 needs a measured cost table first: time a solo request at 1, 2, 4 and 8 verify rows (`"draft": false` for 1) and read the
per-position acceptance counters `rexl3_spec_position_*` on `/metrics`, which also say what k the three content types want.
Run screens (`PHASES=warm,prefill` or `warm,decode`) per change, a full run only to keep one.

## Not done

- No Segment A change has been made yet; the engine worktree equals `f4e27ca`.
- `oracle`, Segment B and other families' golden tests are not part of the harness.
