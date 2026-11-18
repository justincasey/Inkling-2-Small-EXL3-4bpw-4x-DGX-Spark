# Engine vs oracle: the gap, explained

The first TP4 boots of the REXL3 `inkling2` engine (records `2026-11-18-inkling2-tp4-base`) matched the oracle's
greedy tokens but failed `oracle/compare.py`'s logit gates on the two chat fixtures (KL mean 4.2e-3 and 1.2e-2 against
1e-3; top-1 agreement 0.959 and 0.975 against 0.99) and the hidden-state gate (cosine ≥ 0.999) on every fixture, with
routing differing on 11–29 % of (token, layer) pairs. This document finds out why. Evidence:
`bench/records/2026-11-18-inkling2-oracle-gap/`.

## Conclusion

**No bug.** Fed identical inputs, every op of the engine agrees with the oracle to within fp32 summation-order noise,
once the oracle reproduces the engine's three deliberate lossy choices; the router picks the same experts on every one
of the 60,720 (token, layer) pairs checked. The free-running gap is the model amplifying small numeric differences
through routing flips:

1. **Per op, the gap is the three lossy choices** (teacher-forced, identical inputs; medians over layers):
   - **int8 KV of the global layers** is the largest single error: the global layers' attention core differs from the
     spec oracle by 0.9–1.4 % (relative, per row) and by 0.006–0.034 % from the emulating oracle (30–200x less).
   - **bf16 P in the attention kernel** dominates the local layers' attention core: 0.09–0.14 % → 0.018–0.033 %.
   - **fp16 expert GEMM inputs**: the MoE output 0.20 % → 0.077 %.
   - What is left after emulating all three is at the noise floor measured between the oracle and a second fp32
     implementation of itself (table below).
2. **Free-running, the oracle reproduces the engine's drift on its own.** Switching the three choices on in the oracle
   moves it away from the spec oracle exactly as far as the engine is (chat-capital: KL mean 1.31e-2, hidden cosine
   min 0.549, identical routing 78 %; the engine: 1.19e-2, 0.552, 82 %). A ~1e-3 change anywhere flips near-tied
   routes; a flipped expert moves that token's next-layer input by percents, which flips more routes. The engine
   against the *emulating* oracle still drifts (KL mean 4.8e-3, hidden min 0.956, routing 82 % identical), because
   what is left per op (fp32 summation order, ~1e-3) is itself a perturbation of that size: the emulating oracle at
   tp = 1 drifts as far from itself at tp = 4 (see "Noise floor, free-running").
3. Consequence for the gates: a per-token free-running gate cannot separate a bug from summation order for this model.
   The correctness gate is now op-level and teacher-forced (`oracle/teacher_force.py`, against the emulating oracle);
   `compare.py` keeps measured drift gates for both oracles.

## Method

- **Emulation in the oracle** (`inkling2_ref.Emulation`, `gen_fixtures.py --emulate`, default off; definitions in
  `docs/NUMERICS.md`, "Engine emulation"): `kv8` (int8 K/V with the pow2 scale per (token, KV head) of
  `nemotron_kv8.cuh`, global layers), `attn_bf16_p` (the kernel's absolute 512/256-key chunks of 64-key tiles, online
  softmax with bf16 P for P·V), `moe_fp16` (fp16 rotated GEMM inputs of the EXL3 experts). Unit tests:
  `oracle/tests/test_emulation.py` (the int8 quantiser against its definition and boundaries, tiled attention = exact
  attention without rounding, emulated prefill + decode = recompute, kv8 touches global layers only, the fp16 input
  path in process = worker pool).
- **Sub-op dump in the engine** (engine `inkling2-gap` e91148a, dump-only change): with `REXL3_INKLING2_DUMP_OPS=1`
  rank 0 also writes, per layer, the attention core of its 12 heads, the attention output after `attn_sconv`, the MoE
  input, the router logits and the all-reduced MoE output (`bench/tools/merge_dump.py` merges them). The forward is
  unchanged: the dump boot's logits, hidden states and picks are byte-identical to the earlier boot's (0f0d9fe forward).
- **Teacher forcing** (`oracle/teacher_force.py`): each oracle op is fed the engine's own inputs of that op, over all
  rows of the prompt (so the K/V and conv state of earlier rows come from the engine's inputs too): attention (four
  emulation variants), mix, router, experts with the engine's picks and weights (fp16 emulation on/off), conv +
  residual. The noise floor is the emulating oracle against a second fp32 implementation of itself (`Fp32Accum`:
  float32 projections, mix and expert GEMMs). Unit tests: `oracle/tests/test_teacher_force.py` (an oracle-written dump
  agrees exactly; an altered plane fails in its own op and layer only).

## Evidence: identical inputs (teacher-forced)

Row-median relative error ‖o − e‖/‖e‖, median over the layers of each kind (engine against the oracle variant named).
Attention core = rank 0's 12 heads before `wo`; attention a = the sublayer output after the all-reduce and
`attn_sconv`.

| fixture (rows) | op | layers | spec oracle | + bf16 P | + int8 KV | emulating | noise floor |
|---|---|---|---|---|---|---|---|
| chat-capital (40) | attention core | local | 1.35e-3 | 1.75e-4 | 1.35e-3 | 1.75e-4 | 2.37e-4 |
| | | global | 1.36e-2 | 1.36e-2 | 1.36e-3 | 1.12e-4 | 5.44e-4 |
| | attention a | local | 2.54e-3 | 1.08e-3 | 2.54e-3 | 1.08e-3 | 1.15e-3 |
| | | global | 1.25e-2 | 1.25e-2 | 2.58e-3 | 1.14e-3 | 2.04e-3 |
| chat-sky (49) | attention core | local | 1.34e-3 | 1.80e-4 | 1.34e-3 | 1.80e-4 | 2.38e-4 |
| | | global | 1.32e-2 | 1.32e-2 | 1.44e-3 | 5.91e-5 | 7.07e-4 |
| | attention a | local | 2.52e-3 | 1.13e-3 | 2.52e-3 | 1.13e-3 | 1.22e-3 |
| | | global | 1.12e-2 | 1.12e-2 | 2.53e-3 | 1.06e-3 | 1.94e-3 |
| raw-counting (50) | attention core | local | 1.26e-3 | 2.28e-4 | 1.26e-3 | 2.28e-4 | 2.65e-4 |
| | | global | 1.37e-2 | 1.38e-2 | 1.20e-3 | 2.36e-4 | 8.95e-4 |
| | attention a | local | 2.46e-3 | 1.22e-3 | 2.46e-3 | 1.22e-3 | 1.39e-3 |
| | | global | 1.19e-2 | 1.19e-2 | 2.66e-3 | 1.57e-3 | 2.38e-3 |
| raw-long-window (1126) | attention core | local | 9.07e-4 | 3.27e-4 | 9.07e-4 | 3.27e-4 | 3.44e-4 |
| | | global | 9.08e-3 | 9.09e-3 | 9.68e-4 | 3.35e-4 | 9.87e-4 |
| | attention a | local | 2.06e-3 | 1.24e-3 | 2.06e-3 | 1.24e-3 | 1.33e-3 |
| | | global | 8.30e-3 | 8.31e-3 | 2.35e-3 | 1.54e-3 | 2.26e-3 |

| fixture | MoE output: spec oracle | + fp16 inputs | noise floor (fp32 GEMMs) | worst row, emulating |
|---|---|---|---|---|
| chat-capital | 2.04e-3 | 7.75e-4 | 3.38e-4 | 2.2e-3 (cos 0.9999976) |
| chat-sky | 2.00e-3 | 7.62e-4 | 3.33e-4 | 3.6e-3 (cos 0.9999938) |

(The experts were compared on the two chat fixtures; the raw fixtures were checked without them.)

- **Router, identical inputs:** the engine's picks equal the oracle's on all 60,720 (token, layer) pairs of the four
  fixtures (1,920 + 2,352 + 2,400 + 54,048), including the closest call, an 8th-vs-9th margin of 1.3e-9 on
  raw-long-window. The engine's selection equals a float64 selection on its own logits everywhere; its fp32 logits
  differ from float64 by at most 7.6e-6 (3.2e-7 of Σ|x·w|); its weights by at most 9.5e-7. No flip exists, so none
  needed the documented bound (2·max|Δσ| per row) to excuse it.
- **Mix (AttnRes + norm):** at most 0.019 % of elements differ, worst row 5.0e-4 relative (noise floor 5.0e-4).
- **Conv + residual:** the engine's next partial block is reproduced bit for bit from its MoE output except for 2, 2,
  3 and 98 elements out of 8.8M, 10.8M, 11.1M and 249M (4e-7: fp32-vs-float64 at a bf16 rounding boundary).
- **Determinism:** the dump boot (e91148a) reproduced the earlier boot's (0f0d9fe) logits, hidden states and picks
  byte for byte; greedy tokens equal the earlier boots' on all four fixtures.

## Evidence: free-running (compare.py)

Engine dump (prompt + the engine's greedy tokens, teacher-forced prefill) against an oracle fixture that greedily
decoded on its own.

| comparison | fixture | tokens | KL mean | KL max | top-1 | hidden cos min / worst layer median | h_final cos min / median | identical routing |
|---|---|---|---|---|---|---|---|---|
| engine vs spec oracle | raw-counting | equal | 4.6e-4 | 3.5e-3 | 1.000 | 0.699 / 0.9936 | 0.9947 / 0.9991 | 71 % |
| | chat-sky | excused tie (0.047) | 4.2e-3 | 0.156 | 0.959 | 0.956 / 0.9997 | 0.9878 / 0.9993 | 89 % |
| | chat-capital | equal | 1.19e-2 | 0.360 | 0.975 | 0.552 / 0.9980 | 0.9592 / 0.9983 | 82 % |
| | raw-long-window | equal | 2.1e-4 | 4.1e-3 | 1.000 | 0.333 / 0.9751 | 0.6625 / 0.9931 | 55 % |
| engine vs emulating oracle | chat-sky | last token differs | 4.4e-3 | 0.160 | 0.959 | 0.953 / 0.99968 | 0.9878 / 0.99958 | 88 % |
| | chat-capital | equal | 4.8e-3 | 0.045 | 0.975 | 0.956 / 0.9979 | 0.9755 / 0.99759 | 82 % |
| spec oracle vs emulating oracle | chat-sky | equal | 1.1e-3 | 0.012 | 0.959 | 0.964 / 0.99955 | 0.9918 | 87 % |
| | chat-capital | equal | 1.31e-2 | 0.302 | 0.950 | 0.549 / 0.9962 | 0.9656 | 78 % |
| engine vs emulating oracle | raw-counting | equal | 5.8e-4 | 3.9e-3 | 1.000 | 0.715 / 0.9960 | 0.9933 / 0.99895 | 72 % |
| emulating oracle tp=4 vs tp=1 | chat-sky | equal | 1.4e-3 | 0.034 | 1.000 | 0.898 / 0.99974 | 0.9841 / 0.99950 | 89 % |
| | chat-capital | equal | 9.5e-3 | 0.175 | 1.000 | 0.894 / 0.99764 | 0.9744 / 0.99842 | 83 % |

- The top-1 disagreements against the spec oracle sit at oracle margins of 0.060 and 0.047 (row 1 of both chats, the
  same two-token prefix); every generated token matched it.
- Free-running, routing differs from layer 0–2 on (the first flips at margins ≤ 4e-4) and the fraction grows with
  depth, while teacher-forced it never differs: the flips are caused by input differences, not by the router.

## Noise floor, free-running

The same oracle, emulating the engine, at tp = 1 and at tp = 4 differs only in where the bf16 roundings of the
all-reduce fall: a perturbation of ~1e-3 per op, the size of the engine's residual (teacher-forced table). Free-running,
it drifts as far as the engine does from the emulating oracle: KL mean 9.5e-3 / 1.4e-3 (engine 4.8e-3 / 4.4e-3), hidden
cosine min 0.894 / 0.898 (engine 0.956 / 0.953), routing 83 % / 89 % identical (engine 82 % / 88 %), on chat-capital /
chat-sky. Neither pair meets the original per-token gates. The engine's last chat-sky token (153606) is the emulating
oracle's second choice at a margin of 0.062, just outside the 0.05 tie allowance; the spec oracle had it first.

## Gates

**Correctness (binding): `oracle/teacher_force.py` against the emulating oracle** (exit 1 on failure). Per layer:
attention core row-median ≤ 2e-3, attention output row-median ≤ 6e-3, every mix row ≤ 2e-3, MoE output row-median
≤ 1.5e-3 (fp16 inputs emulated, the engine's picks), router logits within 1e-6 of Σ|x·w|, the engine's selection equal
to a float64 selection on its logits, and no pick difference outside its bound. Measured maxima over the four fixtures:
1.0e-3, 2.4e-3, 5.0e-4, 9.1e-4, 3.2e-7, consistent, no flip. The limits fail the spec oracle (62–63 layer failures on
the chats), an oracle without int8 KV (15–16, the global layers) and one without fp16 experts (47–48). Missing bf16 P
alone stays under the limits (local attention core 0.9–1.4e-3 against ≤ 1.0e-3 emulated): the emulation still applies,
the gate just cannot isolate it.

**Drift (documented, `oracle/compare.py`), chosen by the oracle fixture's `emulate`:**

| gate | `engine` (emulating oracle) | `none` (spec oracle) | measured worst (engine) |
|---|---|---|---|
| tokens | equal or first difference at margin < 0.05 | < 0.1 | spec: excused at 0.047; emulating: chat-sky's last token |
| KL mean / max | ≤ 1e-3 / 1e-2 | ≤ 2.5e-2 / 1.0 | 1.19e-2 / 0.36 (spec) |
| top-1 agreement | ≥ 0.99, none outside ties | ≥ 0.95, none outside 0.1 | 0.959 (spec) |
| hidden cosine | every token ≥ 0.999 | per-layer median ≥ 0.95 | median 0.9751 (spec, long, layer 27) |
| h_final cosine | every token ≥ 0.999 | median ≥ 0.99 | median 0.9931 (spec, long) |

The `engine` set keeps the original numbers: no implementation that differs from the oracle in fp32 summation order
meets them on these prompts (the engine fails them on both chat fixtures, and so does the oracle against itself at
tp = 1, above), so a failure there is a measurement, not a verdict; the teacher-forced gate decides. The `none` set
passes all four fixtures with about 2x headroom. Partial emulations have no gate set (`--gates` forces one).

## Reproduce

```bash
# engine with the sub-op dump (any inkling2 commit with REXL3_INKLING2_DUMP_OPS), one boot
./inkling2.sh build e91148a
DUMP_DIR=/home/spark/inkling2-dump-ops EXTRA_ENV=REXL3_INKLING2_DUMP_OPS=1 EXTRA_FLAGS="--kv-tokens 262144" ./inkling2.sh start
INKLING2_URL=http://127.0.0.1:8000 python3 bench/tools/greedy.py --fixtures FIXTURES --teacher-force --out greedy.json
./inkling2.sh stop     # gather rank0..3 under DUMP, then per fixture:
python3 bench/tools/merge_dump.py --dump DUMP --fixture FIXTURES/NAME.safetensors --greedy greedy.json --out NAME.safetensors
# oracle host (full pack), CPU only
python oracle/teacher_force.py --pack FULL --engine NAME.safetensors --out tf-NAME.json --workers 6 --threads 4
python oracle/gen_fixtures.py --pack FULL --out fixtures-engine --emulate engine --workers 6 --threads 4 --verify-recompute
python oracle/compare.py fixtures-engine/NAME.safetensors NAME.safetensors --json cmp-NAME.json
```

## Not done

- The emulating oracle's `raw-long-window` fixture and the experts of the teacher-forced raw fixtures were stopped to
  free the oracle host (the attention, mix, router and residual checks of both raw fixtures are complete).
- The dump covers prefill rows only; decode windows are covered by the engine's decode == prefill tests and the greedy
  token equality.
