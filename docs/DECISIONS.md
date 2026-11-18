# Recipe decisions

One bullet per decision: what, why, evidence, risk. "User" marks a decision the owner made.

## Model and pack
- **Pack:** `0xSero/Inkling-2-Small-EXL3-4bpw` @ `996dc404c1feb3abd5330af8a3c52270ffe0008d` (user). Routed experts EXL3
  mcg 4.0 bpw uniform; attention, embeddings, routers, vision and the drafter BF16 as in the official NVFP4 release.
  0xSero reports KLD 0.0213 / top-1 97.35% against the dequantized source (model card). No REXL3 quantizer is
  involved: `rexl3-quant` is source-only in v0.1.2, reads only F32/F16/BF16 and its teacher implements Qwen3.5 only.
- **Expert split:** rank r owns experts E mod 4 == r. The pack stripes them that way (`model-experts-LL-SS` holds
  E mod 4 == SS); `scripts/pack_ranks.py` checked all 192 expert headers and found no exception.
- **Draft:** the pack's DSpark drafter (`mtp.safetensors`, 5 layers, block 8, Markov RNN head).

## Topology and engine
- **Engine:** REXL3 from the latest upstream (user): `upstream/main` @ `06181b1` (v0.1.2), branch `inkling2` in the
  worktree `~/projects/rexl3-inkling2`. Upstream has no Inkling 2 family and verifies two ranks only; the family, its
  kernels and the first four-rank group are this lane's work.
- **Topology:** TP4, one rank per Spark, rank 0 = .121 (HTTP, vision). Fabric: the CRS804 switch (the ring is
  retired, skill `dgx-spark-cluster`).

## Targets (user)
- 1,048,576-token context per lane; 4–8 lanes, as many as fit.
- Decode 70 tok/s (c=1, ≤ 32K, DSpark on); prefill 4,000+ tok/s (32K).
- Byte-exact decode and prefill against a **pinned engine reference with int8 KV** (user, 2026-11-17): replies equal
  the pinned engine's; drafted == serial, cached == uncached, concurrent == solo, chunk width does not change a byte.
  int8 KV is REXL3's Nemotron format (per token and KV head, power-of-two scale); REXL3 rejects FP8 KV
  (`server/src/main.rs`, `nemotron_h_omni/tp_arm.rs`). About 4.5 GiB per 1M lane per rank: 4 full 1M lanes.
- Reference numerics: the engine is its own byte-exact reference; the fp32 oracle (`oracle/`) checks it by
  tolerance. PORT-SPEC's reference ambiguities U1-U14 are settled in `docs/NUMERICS.md`.

## Memory
- Weights per rank 74.45 GiB (rank 0); KV pool ~19 GiB per rank; int8 KV ~4.5 GiB per 1M lane per rank (BF16 would be
  9.04) (`docs/FIT.md`).

## Operations
- **Staging:** split, not four full copies (user): 91.02 GB per node at `/home/spark/models/Inkling-2-Small-EXL3-4bpw`;
  each rank downloads its own experts, rank 0 the shared files and sends them over rail 0 (`./inkling2.sh fetch`),
  sha256 per rank (`./inkling2.sh verify`). Cost: changing the rank count means re-staging.
- **Owning stack:** `spark-ds41f` (DeepSeek-V4.1 4x, `~/recipes/spark-ds41f`, `config/cluster.json`) normally owns
  all four GPUs. Stopped for the build (user, 2026-11-17) with `cluster stop --apply --parallel`; containers kept.
  Restore: `RESTORE_OTHER` in `.env.example` (`cluster start --replace`).
