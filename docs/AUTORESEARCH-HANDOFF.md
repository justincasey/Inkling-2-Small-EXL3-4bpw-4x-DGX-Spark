# Autoresearch handoff: Inkling 2 Small on REXL3, TP4 on four DGX Sparks

Goal of the next phase: make the verified base recipe fast enough to meet the targets below, without giving up
any correctness gate. Start from this file. Then read `README.md` (verified state), `docs/ROOFLINE.md` (ceilings),
`docs/DECISIONS.md`, `docs/ORACLE-GAP.md`, and the skills `autoresearch-kernel-harness` and `gb10-cuda-kernel-perf`.

## 1. Targets (user, 2026-11-17)

| Metric | Target | Ceiling (`docs/ROOFLINE.md`) | Measured today |
|---|---:|---:|---|
| decode, 1 request, ≤ 32K context, DSpark on | **70 tok/s** | 46 serial; 57–69 with DSpark at k = 3 | not measured (eager decode, no CUDA graphs) |
| prefill, cold 32K prompt | **4,000+ tok/s** | 9.5K | not measured |
| prefill 128K / 1M | 3,200 / 1,700 (set by me, not the user) | 7.6K / 2.7K | one 1M lane: 1,047,598 tokens in 1,802 s, prefill and answer included (`2026-11-18-inkling2-tp4-base`) |
| decode, 8 requests aggregate, short context | ≥ 100 tok/s (set by me, not the user) | 127 | not measured |
| capacity | **4 lanes × 1,048,576 tokens** (user: 4–8) | 4 at int8 KV | met: `lanes_at_max_context=4` |

Hold these while optimizing; each is a gate.
- **Byte-exact (user):** replies equal the pinned engine reference; drafted == serial; cached == uncached;
  concurrent == solo; chunk width changes nothing.
- **Configuration:** int8 KV, 4 × 1M lanes, DSpark, prefix cache and vision stay on.

## 2. Where things stand

- **Engine:** `~/projects/REXL3`, branch `inkling2` @ `f4e27ca` (pinned in `config/pins.sh`, nothing pushed).
  - Family code: `crates/model/src/families/inkling2/`.
  - Kernels: `crates/cuda/src/inkling2/`, `crates/cuda/src/inkling2_attn*.rs`, `crates/cuda/csrc/inkling2_*.cu`.
  - Lane branches `inkling2-{attn,moe,vision,draft,fwd,gap,feat}` are merged into `inkling2`.
- **Recipe:** this repo @ `2b39202`.
  - Controller: `./inkling2.sh` (doctor, build REV, start, stop, smoke, memwatch, record, stop-other/restore-other).
  - Gate tools: `bench/tools/{gates,exactness,needle,smoke,greedy}.py`.
  - Oracle: `oracle/`.
- **Cluster:**
  - Inkling stopped.
  - DeepSeek-V4.1 4x stopped (user gave this lane the GPUs). `./inkling2.sh restore-other` brings it back.
  - Sessions: run `./inkling2.sh stop-other` at the start and `restore-other` at the end, unless told otherwise.
- **Memory headroom is the binding resource.** At the target configuration MemAvailable is 9.9 GiB on rank 0 and
  11.2–12.1 GiB on the others under load. Rank 0 also holds the vision tower and HTTP. Two consequences:
  - Any new buffer (CUDA graph pools, scratch, overlap staging) must come from the prefix cache (6.03 GiB), not
    from the floor. Watch `free_after_gib` and `memwatch` in every run.
  - A guard kill at boot has happened before (`boot-attempt1-killed.txt`).

### Known costs (kernel microbenchmarks, one GB10, from the lane agents' commit messages)

| Op | 1 row | verify window of 8 | 4,096-row prefill chunk |
|---|---:|---:|---:|
| MoE layer, rank-3 real experts, L2 flushed | 0.02–0.17 ms | 1.29–1.47 ms (4 rows: 0.55–0.68) | 20.0–21.3 ms |
| global attention layer (int8 KV) at 4K / 128K / 1M context | 33 µs / 0.33 ms / 2.5 ms | 37 µs / 0.42 ms / 3.3 ms | 32K prefill: 105 ms per global layer, 16 ms per local layer |
| unembed shard 50,256 × 4,608 | 2.25 ms (206 GB/s) | same | — |
| AttnRes mix (9 sources, 3 queries) | 14 µs | — | 2.06 ms |

Arithmetic from these (`[INFERENCE]`, to be replaced by profiles):

**Decode verify window of 8 rows.** About 48 × 1.4 ms = 67 ms of MoE per verify, roughly at the bandwidth roofline
(about 15 unique experts per rank per layer). At 3.8 tokens per round that caps DSpark near 55 tok/s before any
other cost.
- Levers: shorter or adaptive draft length, and acceptance.
- Not a lever: the MoE kernel itself, which is already near the bandwidth roofline.

**Prefill at 32K.** MoE alone is about 8 chunks × 48 × 20 ms ≈ 7.7 s, which caps prefill near 4K tok/s.
- MoE: 5.6 ms of math per chunk at peak, so it runs at about 28% of tensor peak.
- Global attention: 98 ms per layer, about 35 ms at peak.

## 3. Harness to build first (no speed work before it exists)

Follow `autoresearch-kernel-harness`. The patterns closest to this lane are CYBERFROST's harness
(`~/recipes/CYBERFROST-3.8-1x-DGX-Spark/autoresearch.sh`, REXL3, byte-exact against a pinned reference) and
DeepSeek-V4.1 4x's (`~/recipes/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark/autoresearch.sh`, four nodes).

**Benchmark** (`bench/tools/bench.py`, new; tokens from `usage.completion_tokens`, never stream chunks):

| Phase | What it runs |
|---|---|
| warm | discard the first pass after boot |
| decode | c = 1 over a fixed prompt set ≤ 32K (prose, code, structured, JSON), greedy, 512 tokens, DSpark on. Report tok/s, acceptance, tokens per round |
| decode-conc | c = 4 and c = 8 aggregate |
| decode-long | c = 1 at 128K and ~1M context (lane prefilled first) |
| prefill | cold 32K / 128K (per-run nonce, `"prefix_cache": false`), `prompt_tokens / prefill_seconds` from server timing |
| prefill-1m | cold 1M, one lane; long, so only on full runs, never screens |

**Primary metric**, printed as 0 unless every gate passes:

```
target_score = sqrt(decode_c1_tok_s / 70 * prefill_32k_tok_s / 4000) * min(1, lanes_at_max_context / 4)
```

Also print `target_score_raw`, every raw metric, `boot_s`, `free_after_gib` and per-rank MemAvailable minima.

**Fixed gates.** List their ids in the harness header and match each `PASS <id>` exactly.

| Gate id | Check |
|---|---|
| `pin` | every reply equals `bench/reference.json` byte for byte. Pin it once from engine `f4e27ca` with this harness before any change |
| `spec` | `gates.py spec` (drafted == serial, concurrent == solo) |
| `cache` | `gates.py cache` |
| `chunk` | `exactness.py record` at prefill chunk 4096 and 1000, then `diff` |
| `vision` | image replies equal the pinned ones |
| `needle` | 200K needle |
| `capacity` | the capacity line still says `lanes_at_max_context=4` with `kv_tokens=4194304` |
| `memory` | per-rank MemAvailable minimum ≥ 8 GiB (the engine floor); no guard kill |
| `smoke` | `./inkling2.sh smoke` |
| `host` | gb-model inkling2/family/serve and gb-cuda inkling2 host tests compile and pass |
| `goldens` | `testdata/gpu/golden_inkling2_*.txt` unchanged, and other families' goldens unchanged |
| `oracle` | `oracle/teacher_force.py` passes against the emulating oracle. Run only when a kernel's numerics change (re-baseline segments, see §4) |

**Screens and full runs.** `PHASES=warm,decode` or `warm,prefill` is a screen: no score, raw numbers only, used to
rank ideas. A full run keeps one. Full four-node boots take about 90–105 s each.

**Off limits for the loop:**
- `oracle/` and the fixtures on .121;
- `bench/reference.json` once pinned;
- `bench/tools/gates.py` thresholds;
- `config/pins.sh` pack pins and `config/pack-ranks.tsv`;
- the memory guard's floors.

**Mechanics:**
- **Build** with `./inkling2.sh build <rev>`. It builds on rank 0 and ships the same binaries to all ranks, which
  must agree by sha256. One candidate at a time: the four GPUs form one group.
- **Locks:** each rank holds `flock /tmp/rexl3-gpu.lock` for its life. Single-node kernel experiments take the same
  lock exclusively and stay short.
- **Clocks:** `GPU_CLOCKS=2200` lock as in the DeepSeek and CYBERFROST recipes (site policy); the harness sets it through
  `./inkling2.sh gpu-clocks` and samples every rank's clock.

### 3a. The harness as built (2026-11-18)

`./autoresearch.sh` (header: contract, gates, phases), `bench/tools/{bench,score,memcheck}.py`, controller commands
`gpu-clocks` and `engine-test`, `tests/test_bench.py`. The references are pinned from the unmodified engine
(`bench/reference.json`, `bench/reference-vision.json`). Engine candidates are commits of the worktree
`~/projects/rexl3-inkling2-ar` (branch `inkling2-ar` off `inkling2` @ `f4e27ca`). Where it differs from §3:

- `decode-long` has no phase of its own: the cold 32K, 128K and 1M prefill requests each generate 128 tokens, so
  `decode_ctx32k/128k/1m_tok_s` come from the same requests as `prefill_*_tok_s`.
- Times come from the server's histograms `rexl3_request_ttft_seconds` (prefill) and `rexl3_request_e2e_seconds`
  (decode = (completion_tokens - 1) / (e2e - ttft)), read around solo requests. `decode_c1_tok_s` is the mean of the
  four per-type rates; c = 8 runs two waves of the 4 slots.
- The 1M prompt goes as text (about 1,008,000 tokens): the server refuses a request body of more than 300,000 JSON
  nodes, so a million ids cannot be sent. The 32K and 128K prompts go as ids.
- `goldens` = `testdata/gpu` byte-identical to the pinned engine's, plus the five Inkling 2 GPU golden tests passing
  without skipping. Other families' golden tests are not run (their writers are `#[ignore]` and need a wrapper).
- `memory` normalizes each rank's minimum to the recorded 6.03 GiB prefix cache (`memcheck.py`). The capacity plan reads
  free memory at boot and gives the cache what is left: 6.03 GiB on an idle node, 4.52 GiB after the engine tests had
  run, whatever `drop_caches` does. Raw minima differ by about 1.4 GiB between those boots.
- The `oracle` gate is not wired in (Segment B; see §4).

Baseline of the unmodified engine, from the pin run, one validation full run and two screens (four decode runs in all):
`decode_c1_tok_s` 42.2 (prose 22.0, code 67.4, structured 26.4, json 53.1; acceptance 0.307, 3.13 tokens a round),
`prefill_32k_tok_s` 1,304, `prefill_128k_tok_s` 1,199, `prefill_1m_tok_s` 689 (1,463 s), `decode_c4` 52.0, `decode_c8` 57.3,
decode at 32K / 128K / 1M context 40.4 / 31.8 / 28.1, `target_score_raw` 0.443. Repeat spread: `decode_c1_tok_s` within 0.6%
over the four runs, `prefill_32k_tok_s` within 0.02% over two. A full run takes 49 minutes, 24 of them the 1M prefill.

## 4. Byte-exact policy: a decision to take before tuning

The pin is the replies of engine `f4e27ca`. Any change to a kernel's summation order, tiling, precision path or
fusion boundary changes output bits and fails `pin`, even when it is numerically better. So the work splits in two:

- **Segment A, strictly bit-preserving** (no user decision needed): CUDA graphs, launch fusion that keeps every
  arithmetic order, overlapping collectives with compute, scheduling, memory placement, prefetch, draft-length
  policy (drafted == serial makes it output-neutral), host overhead.
- **Segment B, re-baselining** (ask the user first). Kernel rewrites that change bits: tensor-core MoE prefill for
  mcg, attention kernels with another merge order, requantizing the BF16 attention weights. Each needs:
  - the `oracle` gate passing;
  - every other gate passing;
  - a new pin;
  - a new segment, with old comparisons voided.

  REXL3's CYBERFROST precedent was "every change byte exact against the pinned reference". Confirm with the user
  whether this lane keeps that rule or allows re-pins.

The 70 tok/s decode target most likely needs Segment B or a large acceptance gain (§5). Raise that early with
measurements, not at the end.

## 5. Ideas, ranked by expected value

### Decode (Segment A first)

1. **CUDA graphs for decode and verify windows.** Today every step is eager (the Forward agent observed about
   21 tok/s serial, unrecorded). REXL3 families already capture graphs per shape bucket (see `gb10-cuda-kernel-perf`:
   graphs were the largest decode step on RED-SNOW). Kernels must read positions and lengths from device memory.
   The attention plan (`Rows::upload_on`, `set_bounds`) is already graph-ready (`agent://Inkling2Attention` notes).
2. **Draft length.** With 0.41 acceptance and 3.8 tokens per round, an 8-row verify costs about 67 ms of MoE.
   - Try a fixed k of 3–5.
   - Try cost-priced adaptive verification: DeepSeek 4x's `ADAPTIVE_VERIFICATION_ROW_COST` idea and the DSpark
     draft budget.
   - Output-neutral. Measure tokens/s, not acceptance.
3. **Collectives.** 2 all-reduces per layer plus 10 per draft proposal, NCCL one-shot rank-ordered sum.
   - Measure their share with nsys.
   - Overlap the MoE partial all-reduce with the next layer's AttnRes mix where the dependence allows.
   - Keep the rank-order sum: it is part of the bits.
4. **Host and launch overhead** per step: span vs busy in nsys (`pprof.py` / `zprof.py` from `gb10-cuda-kernel-perf`).
5. **Unembed shard** (2.25 ms per step at 206 GB/s, already near bandwidth). It runs twice per round (target verify,
   drafter head). Check that the drafter's folded Markov head doesn't re-read the shard needlessly.
6. **Segment B:** requantize BF16 attention weights (52% of a token's weight bytes) to EXL3 in a new pack. That
   raises the serial ceiling to about 70 and DSpark to about 90+, but it changes the pack and the pin, so it is the
   user's call.

### Prefill

1. **MoE prefill:**
   - Segment A: grouped-expert scheduling and decode-once tiles, keeping each tile's arithmetic order
     (`gb10-cuda-kernel-perf` "decode-once prompt kernel pattern"; check bits).
   - Segment B: an mcg variant of the P2 tensor-core MoE (`glm_p2_moe.cu`, Mul1-only today).
2. **Global attention prefill** (98 ms per layer at 32K): int8 dequant per tile dominates.
   - A: stage dequantized K/V tiles in shared memory once per CTA group without changing the online-softmax order.
   - B: other chunking.
3. **Prefill chunk width:** bits are chunk-invariant, so try 8,192 for throughput. Memory: CYBERFROST paid about
   3 GiB for 8,192, which we do not have on rank 0 without shrinking the prefix cache.
4. **Collectives overlap in prefill:** the DeepSeek 4x analysis found collectives were about 26% of a 128K prefill.
5. **Vision tower** on rank 0 (fp32 attention, no tensor cores): only matters for image-heavy prompts; low priority.

## 6. Traps already paid for

- `pkill -f pattern` inside an ssh command matches its own shell: use `[p]attern`.
- Python multiprocessing workers of a killed oracle parent survived as orphans and starved rank 0. The oracle now
  watches its parent; check `ps` on .121 before a boot anyway.
- The ssh control socket path is too long under macOS `$TMPDIR` for `spark-ds41f`'s `bin/spark`: use `TMPDIR=/tmp`
  (already in `STOP_OTHER`/`RESTORE_OTHER`).
- gb-model lib tests need the gitignored `testdata/nemotron_omni/moe/layer1_fixture{,_repaired}.safetensors`. Create
  empty placeholders for the run and delete them; never commit them.
- The engine's auto KV pool (5.77M tokens) left rank 0 too tight. Keep `KV_TOKENS = SLOTS × MAX_MODEL_LEN`.
- Rank 0 alone carries the vision tower (1.06 GiB plus a 0.56 GiB reserve). Lowering `--max-image-tokens` buys memory
  if a Segment A change needs it.

## 7. First steps

1. `./inkling2.sh stop-other`, then `doctor`, `build` (pinned `f4e27ca`), `start`, `smoke`.
2. Write `autoresearch.sh` plus `bench/tools/bench.py` per §3.
3. Pin `bench/reference.json` on the unmodified engine.
4. Run the baseline: full run three times, interleaved boots; record noise per metric.
5. Profile one decode step and one 32K prefill with nsys on all ranks (graphs off), and attribute time to MoE,
   attention, linears, collectives, host and the drafter. Write the profile into the exec plan before choosing the
   first idea.
6. Ask the user the §4 question with the baseline and profile in hand.
7. Iterate per `autoresearch-kernel-harness` loop discipline: one change per run, screen first, commit kept changes on
   a branch off `inkling2` with the hash in the log, `discard` flat results, keep short notes.
8. When stopped: log the current run, `./inkling2.sh restore-other` (unless told otherwise), and write the next-ideas
   list into the exec plan.
