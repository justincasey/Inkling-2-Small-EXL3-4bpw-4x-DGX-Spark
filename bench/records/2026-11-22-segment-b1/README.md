# Segment B1: the dense core in MXINT8 (2026-11-22)

Engine `c4a3a26` (branch `inkling2-b` of the engine worktree `~/projects/rexl3-inkling2-b`, from Segment A's `a73ff7a`), pinned in `config/pins.sh` of the recipe branch `segment-b1-candidate` (tag `segment-b1`; the session branch keeps Segment A's pin) and run through the full harness twice (the pin run, then the full run: the second boot reproduces the pinned replies byte for byte). **This changes bits.**
The attention projections (the fused `w_q` matrix and `w_o` of every layer), the rank's unembed shard and the drafter's block projections are served as int8 codes
with one power-of-two scale (E8M0) per 32 weights of a row (the OCP MXINT8 layout), made on the host at load from the pack's bf16: half the bytes a decode round reads
from DRAM for 3.7 of its 13.7 GB. Every kernel dequantizes to the exact bf16 value and runs the same bf16 mma chains as before, so an output is the bf16 engine's output on
the dequantized weights, bit for bit, at every row count (`tests.txt`; `fidelity/int8-real-vs-fake`: 9,292 rows of logits, 0 bits of difference between the real engine and the
bf16 kernels on weights rounded through the quantizer). `REXL3_INKLING2_INT8=0` serves bf16 (Segment A's replies). The engine, its tests and the quantizer are in the engine
commits of that branch; `tests/test_mxint8.py` of this repo re-implements the format from its definition and reproduces the engine's quantization of the golden weights bit for bit.

## What it bought and cost

| | Segment A `a73ff7a` (run 20, `../2026-11-22-segment-a-v1/harness-runs/run20-a73ff7a-final`) | B1 `c4a3a26` (full run, `harness-runs/full-20261122-032109`) |
|---|---:|---:|
| `decode_c1_tok_s` (mean of four prompt types) | 54.137 | **69.423** (see the next section: not a like-for-like gain) |
| prose / structured / json / code | 31.628 / 35.777 / 65.318 / 83.825 | 35.455 / 78.467 / 72.447 / 91.322 |
| `decode_c4` / `c8` (concurrent) | 62.678 / 67.533 | 77.337 / 83.926 |
| decode at 32K / 128K / 256K context | 49.757 / 41.425 / 43.226 | 62.622 / 37.038 / 47.922 (other replies: not comparable) |
| tokens a round | 3.0431 | 3.4771 |
| cold prefill 32K / 128K / 256K tok/s | 2,161.0 / 1,868.0 / 1,574.9 | 2,133.1 / 1,846.7 / 1,565.7 (-1.3 / -1.1 / -0.6%) |
| weights a rank (`weights_gib` of the plan) | 73.94 GiB | 72.33 GiB |
| lowest free memory `mem_min_gib` (rank 0) / normalized | 10.31 / 9.55 GiB | 11.93 / 11.63 GiB |
| `free_after_gib` / `prefix_gib` (planner; varies a little from boot to boot: the pin run planned 41.62 / 6.64) | 40.00 / 5.27 | 40.62 / 5.73 |

The prefill costs 0.6-1.3% because a prefill chunk's wide kernels read bf16: the first launch of a layer's attention weights dequantizes them into a cache
(`DequantCache`, 49 MB, shared by the two lanes). The decode kernels read the int8 bytes directly (`kernels/narrow_bf16_vs_int8.txt`: unembed 2,201 -> 982 us, `w_q` 171 -> 73,
`w_o` 72 -> 34, `q` 77 -> 35 per launch, x2.0-2.4, 237-247 GB/s of the bytes read).

## The decode number is not a like-for-like gain

A bits-changing engine changes the greedy replies, and the benchmark's decode speed depends on the reply: DSpark accepts more tokens a round on repetitive text. In the
int8 engine the *structured* prompt's reply falls into a loop after about 250 tokens (`<!-- harness-canary: canary-payments-gateway-7f3a9c1e -->` repeated, distinct
4-gram ratio 0.27 against 0.97 with bf16) and decodes at 78 tok/s instead of 34: that one prompt carries 10.6 of the 14.8 tok/s the mean rose by.

**The clean measure: the same replies, only the kernels differ.** The probe engine (`REXL3_INKLING2_INT8=0 REXL3_INKLING2_FAKE_MXINT8=` all five groups: the bf16 kernels on weights rounded through the
quantizer) produces exactly the replies of the real int8 engine (its pin gate passes against the references the real engine wrote; tokens a round 3.4771 in both). Two interleaved pairs of boots
(`harness-runs/screen-fake-*` and `screen-real-*`, `warm,decode`): `decode_c1` 59.38 / 59.04 with the bf16 kernels, 68.98 / 68.59 with the int8 kernels, **x1.162**; by prompt prose x1.216, structured
x1.162, json x1.148, code x1.152. That is the kernels' gain (about the 14.8% fewer bytes a round, plus the lower launch times). The rest of the 52.3 -> 68.8 (x1.316) is x1.13 from other
replies: tokens a round 3.0431 -> 3.4771 is acceptance on different text, a draw and not a property of the format (the e4m3 engine drew 3.1703).

Measures that compare different replies agree within their noise and are kept for the record:

* the same-session screens (`harness-runs/20261122-015409` and `-020256` bf16, `-015933` and `-020634` int8, interleaved boots): prose 30.1 / 30.4 -> 35.3 / 35.3 (+17%),
  code 80.8 / 81.4 -> 90.7 / 91.3 (+12%), json 63.1 / 63.5 -> 72.0 / 72.2 (+14%);
* 34 other prompts decoded 512 greedy tokens each by both engines (`loop-probe/`, `scripts/loop_probe.sh`): on the 26 that loop in neither, 50.0 -> 56.3 tok/s (x1.125);
  over all 34, 53.6 -> 63.6, which includes loops.

So B1's decode gain is **+16% from the kernels (identical replies)**, not the +28% the harness mean shows (69.4 against 54.1). The harness's `target_score_raw` of B1 (0.7272 in the full run, `target_score` 0 because needle fails) is inflated by the different replies in the same way.

## Fidelity

**Against the bf16 engine, 9,292 rows of logits over the whole vocabulary** (`fidelity/`, `scripts/logit_fidelity.sh`; KL of the bf16 engine's distribution from the variant's, top-1
agreement; rows with a reference margin of at least 0.5 agree in every variant). The quantized-weights probes ran the bf16 kernels on weights rounded through the format
(`REXL3_INKLING2_FAKE_MXINT8`), which the real engine reproduces exactly:

| variant | KL mean | KL p99 | KL max | top-1 |
|---|---:|---:|---:|---:|
| same build twice (control) | 0 | 0 | 0 | 1.0000 |
| e4m3 elements, attention `w_q` | 4.84e-4 | 9.8e-3 | 0.351 | 0.9952 |
| e4m3, `w_o` | 4.67e-4 | 1.04e-2 | 0.318 | 0.9937 |
| e4m3, unembed | 1.64e-5 | 3.2e-4 | 4.7e-3 | 0.9990 |
| e4m3, all three | 5.29e-4 | 1.13e-2 | 0.212 | 0.9947 |
| **int8, all three** | **3.81e-4** | 8.6e-3 | 0.102 | 0.9944 |

Any one of the attention groups gives about the same drift as all of them: the model's routing and near-ties make a perturbation of this size saturate (the median KL is 5e-7 in
every row of the table but the unembed's, 1e-8), so the precision of the format moves it little. The 45 to 59 disagreeing rows of 9,292 of the attention variants all sit at a reference margin under 0.5
(all-three e4m3: 49 rows, 45 of them under 0.05; int8: 52 rows, 44 under 0.05).

**An averaging measure, the negative log-likelihood of the prompt's own next token** (`fidelity/int8-real-vs-bf16-nll`, the same 9,292-token prompt, 9,291 rows with a next token; flips average out, a
systematic cost would show as a positive difference): bf16 0.7742, int8 0.7740, **int8 - bf16 = -0.00013 +- 0.00031** (1 standard error, t = -0.40); the control (the same build twice) 0 +- 0.
No cost is detectable above about 0.0006 nats (2 s.e.), under 0.1% of the NLL. The limit of the measure: the prompt is the benchmark's synthetic filler text, which the model finds easy (NLL 0.77), not
natural text; the same measure on a natural-language set would be the stronger evidence and was not run. It was run for int8 only (the e4m3 engine is not in the branch).

**Against the spec oracle, the four fixtures of `docs/ORACLE-GAP.md`, `oracle/compare.py` with the `none` gate set** (`quality-vs-spec/`, `scripts/quality_vs_spec.sh`; the oracle
is untouched; the Segment A engine is the control):

| engine | chat-capital | chat-sky | raw-counting | raw-long-window |
|---|---|---|---|---|
| bf16 (A) | pass, KL 1.19e-2, top-1 0.975 | pass, KL 4.21e-3, 0.959 (the engine's token differs at 23, oracle margin 0.047: a tie) | pass, 4.58e-4, 1.000 | pass, 2.07e-4, 1.000 |
| e4m3, real | pass, 1.05e-2, 0.975 | **fail**: its token differs at 4 where the oracle's margin is 0.1115 (the gate excuses 0.1), top-1 0.933 | pass, 1.13e-3, 1.000 | pass, 3.10e-4, 1.000 |
| int8, probe | pass, 1.06e-2, 0.950 | pass, 1.33e-3, 0.959 (all 24 tokens equal the oracle's) | pass, 6.54e-4, 1.000 | pass, 2.93e-4, 1.000 |
| **int8, the real engine `c4a3a26`** | pass | pass | pass | pass |

That is why the element is int8 and not e4m3: the e4m3 engine failed the drift gate on one fixture at a near-tie. The fixtures are 30 to 50 positions (and one 1,126), so a single
near-tie decides a gate; int8 is also closer to bf16 in the routing sets (identical expert sets with the spec oracle: chat-sky 0.870 against e4m3's 0.825 and bf16's 0.892).

**Against int8, and what the choice rests on.** The choice rests on one fixture's near-tie (margin 0.1115 against the gate's 0.1) and on routing-set overlap, which a different draw of
fixtures could reverse. The long-prompt drift cannot rank the formats (int8 3.8e-4 and e4m3 5.3e-4 mean KL, 52 and 49 flipped rows, although the unit tests bound the relative RMS weight error at 0.01 for int8 and 0.04 for e4m3: it saturates).
On the harness's structured prompt the e4m3 engine did *not* loop (45.2 tok/s, `harness-runs/20261122-002017`, a normal runbook reply) while int8 does (78 tok/s, the canary line repeated); the
34-prompt loop probe was run for bf16 and int8 only, never for e4m3 (its engine is in the branch history, `90abc85` to `4963e87`; `06151f4` makes int8 the default), so nothing says e4m3 loops less over a population.
An averaging measure was added afterwards (the NLL of the prompt's own next token, below) and was run for int8 only.

## The harness: 10 of 11 gates (`harness-runs/full-20261122-032109/gates.txt`; the pin run's agrees; the full run took 24 minutes, the `pin` gate: every reply of the second boot equals the pinned one)

host, goldens, capacity, smoke, vision, spec, cache, pin, memory, chunk pass; **needle fails**: at 200K tokens the engine finds the pass-code in its reasoning
(`found_in_reasoning: true`) but spends 118 reasoning tokens (bf16: 76) and the 128-token answer budget of `bench/tools/needle.py` cuts the answer to `OTTER-2624-`.
With a 512-token budget, at depths 0.1 / 0.5 / 0.9 both engines find the code (`loop-probe/needle-*.json`): bf16 reasons 71 / 76 / 91 tokens, int8 73 / 118 / 153.
The gate's thresholds are off limits to the loop; whether the budget moves is the user's decision (see the plan). Nothing was promoted: B1 is not the known-good engine.

## Loops

Greedy decoding of this base model loops on some prompts with bf16 too. Over the 34 prompts: replies identical to bf16's on 5, median first difference at token 36; loops
(distinct 4-gram ratio below 0.5) 5 with bf16 and 8 with int8 (three new: structured, `extra19`, `extra20`; none of the five old ones went away), mean ratio 0.699 and 0.714.
Eight against five of 34 is not a significant difference; it is not evidence of none either.

## Not done

* The correctness gate of `docs/ORACLE-GAP.md` (`oracle/teacher_force.py` against the *emulating* oracle) was not run: it needs an `mxint8` emulation in
  `oracle/inkling2_ref.py`, and `oracle/` is off limits to the loop. The chain that stands in for it: the kernels equal the bf16 kernels on the dequantized weights (tests), the
  bf16 engine passed that gate (Segment A), the real engine equals the probe on 9,292 rows, and the format is cross-checked by an independent implementation.
* The MXFP8 (e4m3) engine of this session is in the engine branch's history, not at its tip: e4m3 is the host default from `90abc85` through `4963e87` (`REXL3_INKLING2_FAKE_ELEM=int8` there rounds through int8 in the probe); `06151f4` makes int8 the host default.
* `scripts/lab/mma_peak.cu` (`kernels/mma_peak.txt`): on this GPU `mma.sync` e4m3 m16n8k32 runs at 215 TFLOP/s, twice bf16's 107.5, and the block-scaled form
  (`kind::mxf8f6f4`, ue8m0 scales) at the same 215: the prerequisite of Segment B2 holds. B2 itself was not started (see the plan).
* (A 1M-token prefill run on this engine: done, see "1M tokens" below.)

## 1M tokens

`PHASES=prefill-1m` on `c4a3a26` (`harness-runs/prefill-1m-20261122-034942`, and again as the pin extension, `prefill-1m-pin-20261122-041741`): cold 1,008,317-token prefill on four 1,048,576-token lanes
827.3 and 827.1 tok/s (Segment A's `58c91f2`: 830.4, `../2026-11-22-segment-a-v1/harness-runs/1m-58c91f2`), decode after it 31.49 tok/s (A: 31.23), lowest free memory 10.43 and 10.34 GiB (A: 9.85;
floor 8), capacity and memory gates PASS. The pin gate of the first run failed only because the reference held no 1M entry; the second run wrote it (`PIN=1`, the already pinned entries reproduced).

Files: `fidelity/` (each run's `fidelity.txt`), `quality-vs-spec/` (greedy.json, summary.txt and `cmp/*.json` for bf16, e4m3 real, int8 probe, int8 real), `harness-runs/` (screens and the pin run),
`loop-probe/`, `kernels/`, `tests.txt`, the Segment A references (`reference*-segment-a.json`).
