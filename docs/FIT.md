# Fit — 0xSero/Inkling-2-Small-EXL3-4bpw on 4 Spark(s)

Generated from `intake/intake.json`. Re-measure on the first boot.

## Exact pack

See `intake/manifest.tsv` (byte authority) and `intake/census.json` (per-module dtypes/shapes/bytes).

## Per-rank weights

| Component | GiB | Sharded? |
|---|---:|---|
| routed experts | | TP |
| non-routed quantized | | TP |
| replicated (norms/head) | | no |
| draft | | TP |
| **total** | | |

## KV budget

- node memory: __ GiB
- weights per rank: __ GiB
- guard + activation reserve: __ GiB
- headroom: __ GiB
- bytes/token/rank: __ KiB
- tokens: __

## Profiles (all `PROFILE_MEASURED=0`)

| Profile | Window × streams | KV | Status |
|---|---|---:|---|
| bootstrap | | | unmeasured |
