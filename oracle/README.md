# Inkling 2 Small oracle

The oracle is an independent, deterministic CPU reference forward of the Inkling 2 Small target text model over the
exact pack `0xSero/Inkling-2-Small-EXL3-4bpw@996dc404`. It produces the fixtures that the REXL3 `inkling2` port is checked
against: per-layer hidden states, logits, greedy tokens and router choices. Each fixture comes with the tolerances it is
checked under.

- Ops: `docs/PORT-SPEC.md` §3: embedding, AttnRes block residual, projections, the four short convs (with the dynamic
  K/V conv on global layers), l2 clips, value norm, relative logits, log scaling, windowed and full attention with
  sinks and output gate, router, softcapped SwiGLU experts, final mix/norm, unembed, softcap, vocab slice.
- Numerics: `docs/NUMERICS.md` (U1–U14 and the rounding points). Math between storage points runs in float64; bf16 and
  fp32 rounding happen exactly where the spec stores a value. The oracle is therefore batch-, chunk- and
  order-invariant: prefill + decode equals a full recompute bit for bit.
- TP4 rounding is emulated by default (`tp=4`):
  - rank r holds q heads `[12r, 12r+12)` and experts with E mod 4 == r;
  - each rank's partial is rounded to bf16, the partials are summed in rank order in fp32, and the sum is rounded to bf16.
- Out of scope: the DSpark drafter and the vision tower. The oracle exports the drafter's 8 target features and
  refuses image placeholder ids.

## Layout

| Path | What |
|---|---|
| `inkling2_ref.py` | the model: config, rounding, ops, cache, `Inkling2Ref.forward/layer`, `generate`, `greedy` |
| `exl3.py` | EXL3 decode in numpy: codebooks, trellis unpack, Hadamard rotations, dense decode, layer forward |
| `exl3dq/` | Rust CLI over REXL3's CPU decoder `gb-exl3` (path dependency on `/Users/justin/projects/rexl3-inkling2/crates/exl3`); used for cross-checks only |
| `pack.py` | pack folder as weight and expert sources: index-driven positional reads, EXL3 groups, optional worker pool |
| `st.py` | minimal safetensors reader and writer |
| `gen_fixtures.py` | fixture generator over a full pack folder |
| `compare.py` | engine dump vs oracle fixture: metrics and pass/fail gates |
| `prompts.jsonl` | default fixture prompts (chat-rendered and raw text, including one past the 1024 window) |
| `synthetic.py` | tiny config, random weights and synthetic EXL3 packs for the tests |
| `tools/verify_exl3_real.py` | fetches real expert groups by HTTP range, checks the numpy decode against gb-exl3, records weight and FFN stats |
| `records/exl3-real-verify.json` | output of that check |
| `tests/` | pytest suite; `tests/hf_transcription.py` is an independent float64 transcription of the HF eager model |

## Setup and tests (this Mac)

```bash
uv venv oracle/.venv --python 3.12
uv pip install --python oracle/.venv/bin/python -r oracle/requirements.txt
oracle/.venv/bin/python -m pytest oracle/tests -q
```

`cargo` (toolchain 1.89, pinned in `exl3dq/rust-toolchain.toml`) builds `exl3dq` on first use by the EXL3 tests. The
real-group tests run when `oracle/.cache/real-groups/` exists, i.e. after `tools/verify_exl3_real.py`.

What the tests cover:

- **Op structure.** With rounding off (`Rounding.exact()`), the oracle equals an independent float64 transcription of
  HF `modeling_inkling2.py@fc00b5b3` (F.conv1d convs, unfold/flip dynamic conv, concatenated-sink softmax, SDPA block
  residual) to 1e-9, for both tp=1 and tp=4.
- **Rounding cost.** Each rounded op stays within one bf16 step of the unrounded op (mix + norm, conv) or a few steps
  (attention, MoE). Each rounded layer stays within 12 steps.
- **Single ops.**
  - AttnRes schedule: which query, how many blocks, 3 queries on global layers.
  - Conv streaming == batch: static and dynamic, every chunking.
  - Tap and lag orientation.
  - Relative-bias distances.
  - Window masks: 1024-style window including self, and full causal.
  - Sink and gate against an explicit softmax with a zero-value sink logit.
  - Router: ties go to the lower index; bias changes the selection but not the weights.
  - Softcapped activation, with the sigmoid on the raw gate.
  - τ, the rank-order all-reduce, and numpy vs torch bf16 rounding, ties included.
- **Cache.** Prefill + decode == recompute at many split patterns, with sequences longer than the window. Cached ==
  uncached continuation. The ring keeps window − 1 keys. Expert ownership is E mod 4.
- **EXL3.** Codebook tables (3inst, mcg, mul1), W_q bits and the dense decode are bit-identical to gb-exl3 at several
  widths. `forward` matches gb-exl3 to 1e-12, and gb-exl3 and the port both refuse a wrong marker.
- **Pack path.** A synthetic pack in the real layout (EXL3 mcg 4-bit experts striped E mod 4, index, config) gives:
  - prefill + decode == recompute;
  - a worker pool that matches in-process results bit for bit;
  - a missing expert file raises an error;
  - `gen_fixtures.py --verify-recompute` writes a fixture;
  - `compare.py` passes a clean dump and a noisy one, and fails a broken layer, naming it.

## EXL3 dequantization

The routed experts are EXL3 `mcg` 4.0 bpw with `suh`/`svh` fp16 scales. The oracle decodes them on the CPU with
`exl3.py`, a numpy port of REXL3's CPU reference `gb-exl3` (`crates/exl3/src/lib.rs`: `codebook`, `states`, `unpack`,
`fwht_128`, `decode_weight`, `forward`). The marker tensor must hold `0xCBAC1FED`, as gb-exl3's scan requires.

An expert layer is applied as

```
y = ((((x ⊙ suh) H_K) W_q) H_N) ⊙ svh
```

in float64, where W_q holds the exact fp16 codebook values. This is gb-exl3's `forward`. The dense
`decode_weight` is the bit-exact cross-check form.

Verification (`tools/verify_exl3_real.py`, record `records/exl3-real-verify.json`, run on this Mac): 30 real groups
(gate, up and down of experts 0, 1, 2, 3 and 383 in layers 0 and 47), fetched by HTTP range from the pinned revision.

- **Bit-exact.** W_q fp16 bits and the float64 dense weights are identical to gb-exl3 (`exl3dq wq|dense`) for all 30
  groups. Every marker is `0xcbac1fed`.
- **Sane outputs.**
  - Effective weight RMS is 0.01471–0.01473 for gate/up (≈ 4608^-½) and 0.02078–0.02080 for down (≈ 2304^-½).
  - On unit-RMS inputs scaled by the layer's `mlp_norm`, gate/up output RMS tracks the input RMS: 0.62–0.63 at layer 0,
    0.97–0.98 at layer 47.
  - Expert output RMS is 0.23–0.26 (layer 0) and 0.55–0.58 (layer 47).
  - All values are finite and consistent across experts.
- **Cost.** Decoding one expert and running its FFN on 16 tokens took 0.42–0.47 s. The verification process, holding
  all 30 fetched groups and decoding one at a time, peaked at 1380 MiB RSS.

## Fixtures

`gen_fixtures.py` prefills each prompt, greedy-decodes from the cache, and writes `OUT/NAME.safetensors`:

| Tensor | Shape | Meaning |
|---|---|---|
| `tokens` | I64 [N] | prompt + generated tokens |
| `logits` | F32 [N−1, 200058] | row t is the soft-capped prediction for token t+1 |
| `hidden` | BF16 [49, N−1, 4608] | `[l]` = partial block P entering layer l; `[48]` = P after layer 47. Drafter features are `hidden[5, 11, …, 41]` |
| `h_final` | BF16 [N−1, 4608] | final mix + norm output (drafter feature 48) |
| `router_idx` / `router_w` | I32 / F32 [48, N−1, 8] | selected experts in selection order, and their weights |
| `router_margin` | F32 [48, N−1] | 8th minus 9th selection score |

Metadata: `prompt_len`, `tp`, `pack_config_sha256`, `oracle_source_sha256`, `numerics_version`. `OUT/manifest.json`
holds each fixture's sha256, the generated ids and text, and timings.

### Runner on the real pack: dev-only full copy on rank 0 (chosen)

No node holds the whole pack: rank r stages only experts E mod 4 == r. Two ways to run the oracle were considered:

- **(a)** A per-node expert partial-sum protocol.
- **(b)** A full copy of the pack on one node.

The oracle runs as **(b), one process over a full copy on 192.168.1.121**. That host has 1.27 TB free; the copy adds the
144 expert files of ranks 1–3, 220.7 GB (`intake/manifest.tsv`). Reasons for (b):

- the reference stays a single process with no network protocol;
- it touches nothing the engine's TP machinery does;
- the oracle's TP rounding is already defined per rank, so (b) produces the same bits that (a) would.

The copy is not made yet. Make it once staging has passed `./inkling2.sh verify` on all four nodes. On .121:

```bash
FULL=/home/spark/models/Inkling-2-Small-EXL3-4bpw-full
SRC=/home/spark/models/Inkling-2-Small-EXL3-4bpw
mkdir -p "$FULL"
cp -al "$SRC"/. "$FULL"/                      # hard links to rank 0's files: no extra space
for r in 1 2 3; do                            # rails as in .env.example (FABRIC_ADDRS); LAN IPs work too, slower
  ip=$(echo 10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.4 | cut -d' ' -f$((r + 1)))
  rsync -a --whole-file "spark@$ip:$SRC/model-experts-*-0$r.safetensors" "$FULL"/
done
```

Then, from the recipe checkout, check the copied files against the intake manifest:

```bash
awk -F'\t' '$3 ~ /^model-experts-..-0[123]\./ {print $1"  "$3}' intake/manifest.tsv > /tmp/full.sha256
scp /tmp/full.sha256 spark@192.168.1.121:/tmp/
ssh spark@192.168.1.121 'cd /home/spark/models/Inkling-2-Small-EXL3-4bpw-full && sha256sum -c --quiet /tmp/full.sha256'
```

Environment and run on .121. The run is CPU-only and never uses the GPU, so it can share the node with the
`spark-ds41f` stack. Keep it niced and small:

```bash
rsync -a --exclude .venv --exclude .cache --exclude exl3dq/target oracle/ spark@192.168.1.121:~/inkling2-oracle/
ssh spark@192.168.1.121
python3 -m venv ~/inkling2-oracle/.venv
~/inkling2-oracle/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
~/inkling2-oracle/.venv/bin/pip install numpy==2.5.5 tokenizers==0.23.3
nice -n 19 ~/inkling2-oracle/.venv/bin/python ~/inkling2-oracle/gen_fixtures.py \
  --pack /home/spark/models/Inkling-2-Small-EXL3-4bpw-full --out ~/inkling2-oracle/fixtures \
  --workers 3 --threads 4 --verify-recompute
```

Resources:

- **Memory.** Weights are read per op by positional reads; nothing is mmapped or cached. The main process holds one
  layer's dense matrices in float64 at a time; the largest are `wq`/`gate_attn`/`wo` at 226 MB each, and an unembed
  chunk is 604 MB. A worker holds one expert at a time: three W_q matrices at 85 MB each in float64, plus
  temporaries. Neither process has been measured on a Spark. Start with `--workers 3`, watch RSS against the 6 GiB
  budget the shared nodes allow, and record the first run.
- **Time.** [estimate, from the Mac measurement only] Each expert costs about 0.45 s per decode. One decode step
  touches 8 experts × 48 layers, about 3 min single-process. A prefill costs one decode per unique expert per layer;
  the 1119-token `raw-long-window` prompt (pack tokenizer) can touch all 384 experts in each layer. Workers divide
  this. The Spark CPU has not been measured.
- **What remains.**
  1. Staging finished and verified.
  2. The full copy above.
  3. The venv.
  4. One run per prompt.
  5. The fixtures copied back and recorded under `bench/records/<date>/` with the manifest.

  `--verify-recompute` doubles the cost and turns the run into its own cached == uncached check.

## Tolerances

`python oracle/compare.py ORACLE.safetensors ENGINE.safetensors [--json report.json]`: the engine dump uses the
fixture layout; `hidden`, `h_final` and `router_*` are optional. The command exits 1 when a gate fails.

| Check | Gate | Why |
|---|---|---|
| Generated tokens | identical, or the first difference falls where the oracle's top-1 − top-2 < 0.05 (an excused tie) | greedy decoding is discrete; a near-tie may flip under fp32 vs float64 accumulation |
| Logits (positions before any divergence) | KL(oracle ‖ engine) mean ≤ 1e-3 and max ≤ 1e-2 nats; top-1 agreement ≥ 99%; no top-1 disagreement where the margin ≥ 0.05 | bf16 activations and the engine's fp16 EXL3 intermediates move logits by small amounts; a real bug moves them by orders of magnitude more |
| `hidden` (every layer, every token) | cosine ≥ 0.999 | per-op error is bf16-level (tests); 0.999 is about 4.5 % relative error, room for accumulation over 48 layers but not for a wrong op. The report names the worst layer |
| `h_final` | cosine ≥ 0.999 | same |
| Router | informative: identical-set fraction, slot overlap, largest margin at which a flip happened | a flip at a margin near 0 is rounding; a flip at a wide margin points to the router or its input |

These gates are provisional. Tighten them once the first engine-vs-oracle comparison is recorded under
`bench/records/`, and quote no measured agreement before that. The byte-exact targets (drafted == serial,
cached == uncached, concurrent == solo, chunk-width invariance) are engine-vs-pinned-engine checks; this oracle
anchors correctness, not bit equality.
