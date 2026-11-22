# Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark

Thinking Machines' Inkling 2 Small (594B total / 17B active MoE, multimodal) served from
[0xSero/Inkling-2-Small-EXL3-4bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw) @ `996dc404` on four DGX
Sparks, tensor parallel 4, on a fork of the [REXL3](https://github.com/REXL3/REXL3) engine (Rust + CUDA), behind an
OpenAI-compatible API on rank 0.

> **Status: verified base recipe.** Correctness, determinism and capacity are recorded below. Performance work
> (CUDA graphs, kernel speed, decode/prefill targets in `docs/ROOFLINE.md`) is the next phase; no speed number is quoted
> here.

## What runs

| | |
|---|---|
| pack | `0xSero/Inkling-2-Small-EXL3-4bpw` @ `996dc404c1feb3abd5330af8a3c52270ffe0008d`: routed experts EXL3 mcg 4.0 bpw, everything else BF16 |
| staging | split: rank r holds experts E mod 4 == r plus every shared file, 91.02 GB per node (`config/pack-ranks.tsv`) |
| engine | REXL3 fork branch `inkling2` @ `f4e27ca` on upstream `06181b1` (v0.1.2): new `inkling2` family, kernels, four-rank group |
| nodes | rank 0 `192.168.1.121` (HTTP, vision), 1 `.122`, 2 `.123`, 3 `.18`; NCCL 2.30.7 over both rails of the CRS804 switch, TC 106 |
| context | 1,048,576 tokens per lane, 4 lanes, int8 KV |
| features | DSpark speculative decoding (the pack's drafter), prefix cache, images, reasoning, tool calls |

## Quick start

```bash
cp .env.example .env            # defaults are the recorded configuration
./inkling2.sh stop-other        # only if the DeepSeek-V4.1 stack holds the GPUs (STOP_OTHER)
./inkling2.sh doctor            # GPUs idle and alike, memory, pack files, fabric, pinned NCCL, engine
./inkling2.sh fetch             # first time: each rank downloads its share, sha256-verified (~45 min)
./inkling2.sh build             # REXL3_COMMIT from config/pins.sh, built on rank 0, shipped to ranks 1-3
./inkling2.sh start             # ranks 3..0 under memory guards; ready in ~90-105 s
./inkling2.sh smoke             # chat, completion, tool call, reasoning on/off, image
./inkling2.sh stop
```

`./inkling2.sh help` lists every command; `.env.example` documents every setting. `build` needs the engine repository
(`REXL3_SRC`, default `~/projects/REXL3`) on the control machine. The pack is unvalidated in REXL3's list, so ranks run
with `--allow-unvalidated-pack`; its split-staging identity is `sha256:71d76043…` (every rank computes the same).

## Verification

All on engine `f4e27ca`, default configuration, record
[`bench/records/2026-11-18-inkling2-final-base/`](bench/records/2026-11-18-inkling2-final-base/) unless noted.

| Check | Result | Evidence |
|---|---|---|
| capacity | 4 lanes × 1,048,576 tokens, int8 KV + int8 draft plane 4,644 B/token/rank, prefix cache 6.03 GiB, weights 73.94 GiB/rank | `boot.txt` |
| four concurrent 1M lanes | 4 needles at 1,044,231-1,044,275 prompt tokens, depths 0.2-0.8, all admitted and found; 6,195 s wall | `needle-4x1M.json` |
| memory under load | MemAvailable minimum 9.9 GiB (rank 0) to 11.7 GiB per rank | `memwatch-4x1M.txt`, `memwatch-bootA.txt` |
| drafted == serial | identical ids for short/medium/long (2,030)/xlong (8,334)/73K-token prompts and 4 concurrent; acceptance 0.41, 3.8 tokens/round | `spec.json` |
| drafted boot == `SPEC=none` boot | identical | `spec-serial-boot.json` |
| concurrent == solo | identical (part of the spec gate) | `spec.json` |
| cached == uncached | identical at 28-73,166 tokens and for shared 1K/70K prefixes; cold requests never served from cache | `cache.json` |
| chunk width / boot to boot | `--prefill-chunk` 4096 vs 1000 on two boots: identical (16-8,334-token prompts); the first 64 ids also equal the first TP4 engine's (`0f0d9fe`, before DSpark/prefix cache/vision) | `exactness/` |
| vision | three synthetic images described correctly (colours named); image and text replies identical across two boots | `vision-boot{A,B}.json` |
| smoke | chat, completion, tool call, reasoning on/off, image | `smoke.txt` |
| oracle (fp64 reference) | greedy tokens equal on all four fixtures; op-level gate against the engine-emulating oracle passes; no engine bug | [`docs/ORACLE-GAP.md`](docs/ORACLE-GAP.md), `bench/records/2026-11-18-inkling2-oracle-gap/` |

Earlier records: first TP4 boots, 200K and 1M single-lane needles (`2026-11-18-inkling2-tp4-base/`), features boot
(`2026-11-18-inkling2-tp4-features/`).

## Design notes

- **Numerics.** The engine is its own byte-exact reference. Its three deliberate lossy choices (int8 KV on the 8
  full-attention layers, bf16 attention probabilities, fp16 expert GEMM inputs) are emulated by the oracle for the
  binding op-level gate; free-running drift against the pure oracle is measured, not gated tightly, because near-tie
  router flips compound over depth (`docs/ORACLE-GAP.md`, `docs/NUMERICS.md`).
- **Memory.** `docs/FIT.md` (estimate and measured). `KV_TOKENS` defaults to SLOTS × MAX_MODEL_LEN; the memory guard
  kills a rank at 2 GiB (loading) / 1 GiB (ready) MemAvailable, or on swap growth under the engine's 8 GiB floor.
- **Decisions** with evidence: `docs/DECISIONS.md`. Op-by-op model spec: `docs/PORT-SPEC.md`.

## Performance harness

`./autoresearch.sh` is the benchmark of the performance phase ([`docs/AUTORESEARCH-HANDOFF.md`](docs/AUTORESEARCH-HANDOFF.md)):
it builds the candidate engine commit (`CANDIDATE_SRC`, default `~/projects/rexl3-inkling2-ar`), locks the GPU clocks at
2200 MHz, runs the engine's host and golden tests, boots the group twice (prefill chunk 4096, then 1000), runs
`bench/tools/bench.py` (decode c = 1/4/8, cold 32K/128K/256K prefill, replies pinned in `bench/reference.json`) and the
correctness gates, and prints `METRIC` lines. `target_score` is 0 unless every gate in the script's header printed `PASS`.

```bash
./inkling2.sh stop-other                 # once per session, if the DeepSeek stack holds the GPUs; restore-other at the end
PIN=1 ./autoresearch.sh                  # on the unmodified pinned engine only: writes or extends the two reference files
./autoresearch.sh                        # a full run (about 25 minutes; no 1M prefill)
PHASES=warm,prefill-32k ./autoresearch.sh  # a screen: raw numbers, target_score 0 (also warm,decode and warm,prefill)
PHASES=prefill-1m ./autoresearch.sh      # the 1M-token prefill benchmark (about 25 minutes), a screen
```

Scope: Segment A (bit-preserving, complete: tags `segment-a-v1` here and `inkling2-segment-a-v1` on the engine) and Segment B1 (bit-changing: the dense core as MXINT8,
`bench/records/2026-11-22-segment-b1/`). The pins and references are B1's engine `c4a3a26` (`config/pins.sh`; Segment A's references are in the B1 record, the
`segment-a-v1` tag holds the A pins). The `oracle` gate (`oracle/teacher_force.py` against the emulating oracle) is not wired in and was not run for B1: it needs an `mxint8`
emulation in `oracle/`, which the loop may not edit; the spec-oracle drift gate (`scripts/quality_vs_spec.sh`) was run instead.

Segment B1 (engine `c4a3a26`): 10 of 11 gates, the needle gate fails on its 128-token answer budget (the code is found in the reasoning). Like for like the decode is
+12 to +14% (the harness's `decode_c1` 54.1 to 69.0 includes one reply that falls into a repetition loop), cold prefill -0.6 to -0.9%, lowest free memory +0.96 GiB.

Segment A result (record `bench/records/2026-11-22-segment-a-v1/`, engine `a73ff7a` against the pinned `f4e27ca`, all 11 gates): `target_score` 0.4422 to
0.6464, decode (mean of four prompt types, one request) 42.2 to 54.1 tok/s, cold prefill at 32K / 128K / 256K 1,298 / 1,199 / 1,080 to 2,161 / 1,868 / 1,575 tok/s,
cold 1M prefill 689 to 830 tok/s (4 x 1,048,576-token lanes, memory gate passed). Every row's logits of a 9,292-token prompt are byte-identical to the
pinned engine's on all four ranks (`bitwise-logits.txt`, 48 of 48). The 70 tok/s and 4,000 tok/s targets are not met: decode is within about 6% of the
bandwidth bound of the BF16 core (`docs/exec-plans/completed/autoresearch-segment-a.md`, "Seventh pass"), prefill needs faster MoE and linear kernels.
The experiments ran under `scripts/safeloop.sh`, which restores the last known-good engine whatever happens (`docs/SELF-IMPROVEMENT-LOOP.md`).

## Known gaps (next phase)

- Decode runs eagerly (CUDA graphs would save at most about 2%: the idle time is 2.6 us launch gaps). Segment A (bit-preserving) work
  is recorded in `bench/records/2026-11-22-segment-a-v1/` and `docs/exec-plans/completed/autoresearch-segment-a.md`, with what was left undone.
  Targets and ceilings: `docs/ROOFLINE.md`. Segment B (bit-changing, `docs/exec-plans/active/autoresearch-segment-b.md`): the int8 dense core is built and pinned but
  not promoted (the needle gate's 128-token answer budget; decisions listed in the plan); the FP8 MoE prefill is assessed, not built.
- `prompt_logprobs` is not served.
- The pack's fingerprint is not in REXL3's validated list (`--allow-unvalidated-pack`).
- REXL3 refuses `json_schema` / `json_object` response formats.

## Credits and licenses

- Model: Thinking Machines, Inkling 2 Small (Apache-2.0). Pack: 0xSero (Apache-2.0).
- Engine: REXL3 (AGPL-3.0); this recipe builds the fork branch `inkling2`. EXL3 format: turboderp's exllamav3 (MIT).
- `oracle/exl3dq/` links REXL3's `gb-exl3` crate (AGPL-3.0) as a development tool only.
- NCCL (NVIDIA, BSD-3-Clause).
