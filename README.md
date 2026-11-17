# Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark

> **Status: scaffold.** No hardware run yet. Do not quote a number until a record exists under `bench/records/`.

Serve **0xSero/Inkling-2-Small-EXL3-4bpw** @ `996dc404c1feb3abd5330af8a3c52270ffe0008d` on 4 DGX Spark(s), behind an OpenAI-compatible endpoint.

## Lane

| | |
|---|---|
| pack | `0xSero/Inkling-2-Small-EXL3-4bpw` |
| revision | `996dc404c1feb3abd5330af8a3c52270ffe0008d` |
| topology | TP=4 |
| nodes | 192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18 |
| engine | TODO (image digest + fork shas in `config/pins.sh`) |
| exact pack size | see `intake/intake.json` |
| weights per rank | see `intake/intake.json` (`fit`) |

## Open risks

- Fit is an estimate until the first boot records `MemAvailable` at ready and under load.
- Engine/quant support: TODO (name the kernels and the refusals found).

## Install / run

TODO after the controller exists. The command surface is in `docs/RECIPE_INFLATION_PIPELINE.md`.

## Benchmarks and evals

See `EVAL-MATRIX.md` in the pipeline templates. Planned: smoke, probe, decode/prefill/needle/spec/pin/check,
quality, tools, longctx, api-test, sixcat.

## Credits and licenses

TODO.
