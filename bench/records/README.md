# Records

One directory per boot configuration, `bench/records/<date>-<name>/`, created by `./inkling2.sh record NAME` while the
engine to be recorded is installed (the helper reads the installed binaries over ssh). The README quotes only records.
This repository has none yet.

## meta.json

Written by `record`; the run's results are added next to it.

| Key | What |
|---|---|
| `name`, `date` | the record's name and UTC date (the directory name) |
| `recipe.commit`, `recipe.uncommitted_files` | this repository's HEAD and how many files differed from it |
| `pack.repo`, `pack.revision`, `pack.dir` | `config/pins.sh` MODEL_REPO / MODEL_REV and the staged path |
| `engine.commit`, `engine.sha256` | from every rank's installed `rexl3-serve --version` and the binary's sha256 (all four must agree) |
| `engine.pinned_commit`, `engine.lab_build` | `config/pins.sh` REXL3_COMMIT, and whether the installed commit differs from it |
| `engine.version` | rank 0's full `--version` table (nvcc, cuda_arch, kernel_hash, cublas, ...) |
| `nccl.lib`, `nccl.sha256`, `nccl.env` | the pinned NCCL library and the `NCCL_ENV` every rank runs with |
| `config` | the tunables: PROFILE, SERVED_NAME, PORT, TP_PORT, MAX_MODEL_LEN, SLOTS, PREFILL_CHUNK, SPEC, PREFIX_CACHE, VISION, MEMORY_FLOOR_GIB, EXTRA_FLAGS, EXTRA_ENV, DUMP_DIR, GUARD_* |
| `ranks[]` | per rank: `rank`, `host`, `fabric` and `command`, the exact `print-serve` line it runs |
| `results[]` | filled by hand or by the run: one entry per result file of the directory |

## Result files (next to meta.json)

- `boot.txt`: `./inkling2.sh start` output (capacity line `rexl3 event=inkling2_plan ...`, `boot_s=`).
- `memwatch.txt`: `./inkling2.sh memwatch report` after a `memwatch begin` at the start of the measured window
  (per rank MemAvailable minimum in the window, while loading, once ready).
- `smoke.txt`: `./inkling2.sh smoke` (chat, completion, tool call, reasoning on/off, image unless VISION=off; each
  check's usage with `usage.spec`).
- `greedy.json`: `bench/tools/greedy.py --fixtures ... --out` (engine greedy vs oracle fixtures).
- `compare-<fixture>.json`: `oracle/compare.py FIXTURE ENGINE --json` on `bench/tools/merge_dump.py` output.
- `exactness-concurrent.json`, `exactness-<label>.json` and their `exactness.py diff` output.
- `needle-<tokens>[-lanes4].json`: `bench/tools/needle.py --out`.
- `gates-spec.json`: `bench/tools/gates.py spec` (drafted == serial per prompt incl. a > 64K one, 4 concurrent drafted
  vs solo serial; `usage.spec` per reply, acceptance and tokens per round); `gates-spec-serial.json` the same on a
  SPEC=none boot (`--serial-boot`) and the `gates.py spec-diff` output between the two.
- `gates-cache.json`: `bench/tools/gates.py cache` (cached == uncached, cold 0 / warm > 0 cached tokens, shared
  prefixes > 1K and > 64K tokens; `/metrics` prefix counter deltas per request).
- `gates-vision-<label>.json`: `bench/tools/gates.py vision` (synthetic images and the exactness text prompts; a
  VISION=off boot with `--no-images`) and the `gates.py diff` output between two boots.
