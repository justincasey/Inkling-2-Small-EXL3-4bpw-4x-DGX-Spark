# Agent Instructions — Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark

This is a serving-recipe repository. A change is correct only when the pinned stack serves and a record says so.

## Read first

- `README.md` — lane, pack, pins, risks.
- `docs/FIT.md` — per-rank weights and KV math.
- `intake/intake.json` — manifest/census/footprint evidence.
- `docs/DECISIONS.md` — decisions and traps.

## Rules

- Pins in `config/pins.sh`; not tunable from `.env`.
- No invented numbers: a quoted number needs `bench/records/<date>/meta.json`.
- Fail closed: no flag that skips a gate; the memory guard is mandatory.
- Remote scripts receive variables by name (no `$1`).
- Merged serve config, never a quant block mounted over `config.json`.
- Run the static tests before any hardware run.

## Before finishing

```bash
bash tests/test_recipe.sh
python3 tests/test_pack_layout.py
python3 -m py_compile bench/*.py scripts/*.py tests/*.py
```
