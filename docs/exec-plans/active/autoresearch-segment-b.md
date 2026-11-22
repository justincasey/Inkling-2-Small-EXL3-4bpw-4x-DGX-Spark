# Execution plan: autoresearch, Segment B (bit-changing speed)

Session: `autoresearch/session-20261118` after the Segment A tags (`segment-a-v1` here, `inkling2-segment-a-v1` on the engine at `a73ff7a`); B1's pin is on `segment-b1-candidate`; metric `target_score` as before
(`autoresearch.sh`), but **comparisons with Segment A are void**: B changes the replies, and the benchmark's decode speed depends on the reply (see the B1 record).
User (2026-11-22): do all of Segment A first, tag it, then the two B items: (1) the dense core in 8-bit weights, (2) an FP8 MoE prefill. Every B change needs the oracle gate and a
re-pin. Engine worktrees: `~/projects/rexl3-inkling2-b` (branch `inkling2-b`, the frozen one the screens and the harness build) and `~/projects/rexl3-inkling2-dev`
(`inkling2-b-dev`, where code is written). Nothing of B is pushed and nothing is promoted: the known-good engine is still Segment A's `a73ff7a` (`.work/safeloop/state.env`).

## Status

- [x] harness on `c4a3a26`, the pin run and a full run (`bench/records/2026-11-22-segment-b1/harness-runs/`): 10 of 11 gates both times, `pin` reproduced byte for byte on the second boot; **needle fails on the 128-token answer budget** (the code is found in the reasoning)
- [x] format chosen by measurement: e4m3 (MXFP8) failed one spec-oracle drift fixture, int8 passes all four (record `bench/records/2026-11-22-segment-b1/`)
- [x] new references written by `PIN=1` (the 1M reply included) with `config/pins.sh` on **branch `segment-b1-candidate`** (tag `segment-b1`); this branch keeps Segment A's pin and references (`f4e27ca`) until the user decides, so the stack it builds and serves is the validated one
- [x] B1 engine `c4a3a26`: attention `w_q` / `w_o`, the unembed shard and the drafter's block projections as MXINT8; decode 2.0-2.4x faster per launch; replies equal the bf16 engine's on the dequantized weights bit for bit
- [ ] **user decisions below**
- [x] 1M-token prefill on `c4a3a26`: 827.3 / 827.1 tok/s, decode 31.49 after it, lowest free memory 10.43 / 10.34 GiB, capacity and memory PASS; the 1M reply is pinned (record `README.md`, "1M tokens")
- [ ] B2: not started; the measured prerequisites and an estimate are below
- [ ] before B1 could be promoted (all engineering, none started): (a) make `REXL3_INKLING2_INT8` part of the boot gate's facts (`I2Facts` in `crates/model/src/families/inkling2/tp.rs`, which already carries
      `kv_dtype`, the collective and the drafter label), so a rank that quantized another set of groups is refused instead of serving different bits: a change to the engine, so a new commit and a
      re-pin; (b) the NLL measure of the B1 record on a natural-language set (the one run is the benchmark's synthetic filler text); (c) merge `segment-b1-candidate` (the pin and references) into the session branch

## Decisions for the user

1. **The needle gate.** `bench/tools/needle.py --tokens 200000 --depth 0.5` allows 128 tokens; the int8 engine reasons 118 tokens before answering (bf16: 76) and the answer is cut at `OTTER-2624-`.
   With 512 tokens both engines find the code at depths 0.1 / 0.5 / 0.9 (reasoning 71 / 76 / 91 against 73 / 118 / 153). The thresholds are off limits to the loop, so B1 stays at 10 of 11 and is not
   promoted. Options: raise the budget (256 is enough for the three depths measured), keep it and drop B1, or find another numerics that happens to answer in 128 (a lottery: not recommended).
2. **The oracle's correctness gate.** `oracle/teacher_force.py` against the emulating oracle needs an `mxint8` emulation in `oracle/inkling2_ref.py` (the weights rounded through the same quantizer),
   and `oracle/` is off limits to the loop. What exists instead: the spec-oracle drift gate on the four fixtures (passes for int8), exact equality of the real engine with the bf16 kernels on
   rounded weights (9,292 rows, every kernel), and an independent re-implementation of the format (`tests/test_mxint8.py`). If the teacher-forced gate is required, say so and allow the oracle edit.
3. **Promote B1 or not.** Gains: decode **x1.162 at identical replies** (the bf16 kernels on rounded weights against the int8 kernels, two interleaved pairs; by prompt x1.15 to x1.22), prefill -0.6 to -1.3%,
   +1.6 GiB of lowest free memory and 1.6 GiB less weights a rank. Costs: every reply differs from Segment A's (first difference at tokens 17-165 on the pinned seven), three of 34 prompts loop that did not
   (five loop with bf16 as well), and the harness's `decode_c1` (69.42) holds, besides the kernels' x1.162, the acceptance on different text (x1.13, a draw) and one looping reply (+10.6 of the +15.3 over Segment A's 54.14).

## B1 as built

* Format: int8 codes (-127..127) with one power-of-two scale (E8M0) per 32 weights of a row, the OCP MXINT8 layout; the scale is `2^(floor(log2 amax) - 6)` (amax in 64..128 of it), clamped to exponents 27..134.
  Host quantizer `crates/cuda/src/inkling2/mxint8.rs` (on every core of the host, at load; the boots took the same time: `boot_s` 114 and 114 with int8, 120 and 114 with bf16), cross-checked by `tests/test_mxint8.py`.
* Layout: a row is cut into pairs of 32-wide chunks; byte `16 t + 8 c + j` of a 64-byte pair is weight `64 p + 32 c + 8 t + j`, so a lane of the narrow kernel reads its 8 weights of two chunks as one 16-byte load;
  scales `[n][k/32]`.
* Kernels (`csrc/inkling2_linear.cu`): `gb_inkling2_lin_i8` (the narrow kernel, dequantizing in registers: integer to float by the 2^23 trick, `cvt.rn.bf16x2.f32`, one `fma.rn.bf16x2` with the
  power-of-two scale: exact) and `gb_inkling2_dequant_i8` (the prefill's route: the first linear of a layer dequantizes into the 49 MB `DequantCache`, the wide bf16 kernels read that; the two lanes share it).
  Unembed and the drafter's block projections always use the narrow kernel; the drafter's context K/V projections run over a prefill's rows and stay bf16; a global layer's K and V stay bf16.
* Switch: `REXL3_INKLING2_INT8` (unset: every group; `0`: bf16, Segment A's replies; a comma list of `attn_qkv attn_o unembed drafter_attn drafter_mlp`). The memory plan prices the int8 bytes and the cache.
* Probe: `REXL3_INKLING2_FAKE_MXINT8` rounds the same groups through the quantizer and runs the bf16 kernels (the fidelity measurements, and the proof that the real engine adds nothing).
* Tools (all run under `scripts/safeloop.sh cmd`, which restores the known-good engine): `logit_fidelity.sh` (KL and top-1 of variants against a reference boot over a 9,292-token prompt, whole vocabulary),
  `quality_vs_spec.sh` (the four spec fixtures through `oracle/compare.py`), `loop_probe.sh` (34 prompts: loops, drift, speed; `NEEDLE=1` adds 200K needles with a 512-token budget), `lab/mma_peak.cu`,
  `ops_dev.sh` (now fails on a build error, a failing test or an empty filter: an earlier run of stale binaries is how a compile error was once missed).

## B2: an FP8 MoE prefill: what is known

* `scripts/lab/mma_peak.cu` on this GPU: `mma.sync` bf16 107.5 TFLOP/s, e4m3 m16n8k32 215.2, block-scaled e4m3 (`kind::mxf8f6f4`, ue8m0, needs `sm_121a`) 215.0: the 2x of the tensor pipe is there, and the
  block-scaled form applies a scale per 32 k of both operands inside the mma.
* Where the prefill goes (Segment A's profile, `docs/exec-plans/completed/autoresearch-segment-a.md`): the MoE gate/up/down kernels are 4.8 s of the 15.2 s of a 32K prefill (12.8 ms a chunk-layer against 5.6 ms of
  math at 93 TFLOP/s), "latency-bound, not decode-bound": tensor pipe 46%, L2 47%, issue slots 30%, 8 warps an SM (removing the trellis decode saved 9%). A faster pipe does not make a latency-bound kernel
  faster in proportion. If only the 5.6 ms of math halved, a chunk-layer would take 10.0 ms: 376 x 2.8 ms = 1.05 s off 15.2 s, the prefill 6.9% shorter, +7.4% tok/s; reaching 4,000 tok/s needs 1.9x.
* What B2 would cost: a new grouped expert GEMM over EXL3's trellis weights decoded to e4m3 (a rounding of weights that are already 4-bit codes) with the rotated activations quantized per row, in the
  family-shared `moe_prefill.cu` (whose header promises prefill == decode bit for bit: B2 gives that up), and an oracle emulation of both roundings. The fidelity cost lands on the largest part of the model.
* Recommendation: do not build B2 as specified. The bit-exact route to the same kernel (more resident warps per SM for the 4-tile gate/up kernel, which the profile says is the limit) is Segment A work
  and keeps the pin. If B2 is still wanted, the first step is a profile of the MoE kernel at two blocks an SM, not an FP8 kernel.

## What B1 did not touch

The MoE experts (EXL3, the bulk of a decode round's bytes), the mesh and collectives, the vision tower, the router (bf16, 384 rows), the conv weights, the embedding.
