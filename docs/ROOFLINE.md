# Roofline and targets

Arithmetic on hardware ceilings and the pack census; no engine number is in this file.

## Hardware (one GB10)

| Fact | Value | Source |
|---|---|---|
| sustained read bandwidth | 233.8–237 GB/s | GLM-5.3-EXL3-3.38bpw-4x `docs/ROOFLINE.md` (`tools/roofline.cu`); skill `gb10-cuda-kernel-perf` |
| dense bf16 / fp16 GEMM, 8192³ | 93.4 / 91.9 TFLOP/s | GLM-5.3-EXL3-3.38bpw-4x `docs/ROOFLINE.md` |
| fabric | one 200G cage per node, two rails ≈ 98 Gb/s each, CRS804 switch, lossless TC 106 | skill `dgx-spark-cluster` |

## Decode: bytes read per token, per rank

| Item | GB |
|---|---:|
| 8 of 384 experts × 48 layers, 4.0 bpw (12.23 B params / 4) | 1.53 |
| BF16 attention (9.42 GB / 4) | 2.36 |
| routers + unembed share | 0.63 |
| **weights** | **4.52** |
| KV at context L | 8 KiB × L (8.6 GB at 1M) |

About 96 small all-reduces a token (two per layer) at ~25 µs each add ~2.4 ms (estimate). A verify block of n rows
touches 384·(1 − (1 − 8/384)ⁿ) experts per layer: 31 at n = 4, 60 at n = 8.

| Case | Ceiling (read + comm) | Target |
|---|---:|---:|
| c=1, ≤ 32K, serial | 46 tok/s | 40 |
| c=1, ≤ 32K, DSpark (k = 3, ~43 ms/verify, needs ≥ 3 accepted a round) | 69+ | **70** (user) |
| c=1, 1M, BF16 KV | 17 | 14 |
| c=8 aggregate, short context | 127 | 100 |

70 tok/s at c=1 needs either DSpark acceptance of about 3.2 tokens a round or fewer attention bytes (BF16 attention
is 52% of a token's weight reads); the latter changes the pack and so must happen before the byte-exact baseline is pinned.

## Prefill: FLOPs per token

34 GFLOP of linears (2 × 17 B active) + 98,304 × L FLOPs of full attention (8 layers, averaged over a prompt of
length L) + ~2 GFLOP (sliding layers, drafter context). Cluster peak ≈ 4 × 93 TFLOP/s.

| Prompt | GFLOP/token | Ceiling | Target |
|---|---:|---:|---:|
| 32K | 39 | 9.5K tok/s | **4,000** (user: 4,000+) |
| 128K | 49 | 7.6K | 3,200 |
| 512K | 88 | 4.2K | 2,400 |
| 1M | 139 | 2.7K | 1,700 |

Full attention equals the linears' cost at L ≈ 350K and dominates beyond it.
