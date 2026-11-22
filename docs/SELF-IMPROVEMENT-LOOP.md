# The self-improvement loop

One agent edits the engine, builds it on the four Sparks, measures it and decides, without a person watching every step. The cluster is
shared by one serving engine and one experiment at a time, so the loop's first rule is that **whatever happens, the last known-good engine
is installed on all four ranks and serving on :8000 when the loop's tooling exits**. This file says how that is arranged and how an
experiment is run inside it. The tools are in `scripts/`; their state is machine-local under `.work/` (gitignored).

## The parts

| Part | What it is |
|---|---|
| `~/projects/rexl3-inkling2-a` (branch `inkling2-a`) | The validated line of the engine: every commit on it passed the full harness (it is only fast-forwarded to a commit that did). Default `CANDIDATE_SRC` of the scripts. |
| `~/projects/rexl3-inkling2-dev` (branch `inkling2-a-dev`, then `inkling2-b-dev` for Segment B) | Where code is written and committed. Never the target of a cluster run that is still in flight. |
| `~/projects/rexl3-inkling2-screen` (detached) | A frozen checkout for screens: `autoresearch.sh` builds the HEAD of its `CANDIDATE_SRC` at the start of **every** run, so a commit made in the dev tree during a multi-boot screen silently changes the later boots. Point screens here and `git checkout --detach SHA` it. |
| `~/projects/rexl3-inkling2-b` (branch `inkling2-b`) | Segment B's frozen line: fast-forwarded to the dev branch's commit before a run; the scripts of Segment B default to it. |
| `.work/safeloop/state.env` | `GOOD_SHA`, `GOOD_FALLBACK`, `GOOD_SCORE`, `GOOD_DATE`: the known-good engine and the one before it. Seed it by hand once from a commit with a logged 11/11 run. |
| `.work/safeloop/history.log` | One line per run, restore and promotion. |
| `scripts/safeloop.sh` | The wrapper (below). |
| `scripts/ab_screen.sh` | Interleaved screens: one `autoresearch.sh` per variant, `EXTRA_ENV` set per variant and restored from git. |
| `scripts/ops_dev.sh`, `scripts/quad_dev.sh` | The engine's single-node GPU tests (rank 0's lab tree and target directory, never the engine's) and the four-rank exactness test. |
| `scripts/logits_check.sh`, `scripts/plp_req.py` | The byte-for-byte logits check against the pinned engine. |
| `scripts/ops_dump_smoke.sh` | The oracle's op-level dump path on four ranks (rank-symmetry of the collectives). |
| `scripts/logit_fidelity.sh`, `logit_stats.c`, `logit_fidelity.py` | KL and top-1 of engine variants against a reference boot, per row over the whole vocabulary (`REF_ENV` sets the reference's env); a control (same build twice) must show 0. |
| `scripts/quality_vs_spec.sh` | The engine's drift from the spec oracle on the four fixtures of `docs/ORACLE-GAP.md` (`oracle/compare.py`, `none` gate set), `oracle/` untouched. |
| `scripts/loop_probe.{sh,py}` | 34 prompts, 512 greedy tokens each: repetition loops, first difference between two engines, speed; `NEEDLE=1` adds 200K needles with a 512-token budget. |
| `scripts/lab/mma_peak.cu` | Register-only `mma.sync` throughput per operand type (the ceiling kernels are measured against). |

## `scripts/safeloop.sh`

```
safeloop.sh run LABEL SHA [PHASES] [TIMEOUT_S]   one autoresearch.sh run of the HEAD of CANDIDATE_SRC (HEAD must be SHA)
safeloop.sh cmd TIMEOUT_S 'shell command'        any lab command (screens, tests, the logits check, the 1M run)
safeloop.sh restore                              install the known-good engine and start it
```

- **Mutual exclusion.** A `mkdir` lock with a pid file (macOS has no `flock`); a second run is refused, a dead holder's lock is reclaimed.
- **Preflight.** Tracked files of the recipe and of the engine checkout clean, `.env.example` pristine (a screen edits `EXTRA_ENV` and restores it), the
  engine stopped, no compute process on any GPU.
- **`run` with no PHASES** is a full run: only on 11 of 11 gates and a score does the candidate become `GOOD_SHA` (the old one becomes the fallback).
  A screen (PHASES set) never promotes. A failing run, a boot failure, a timeout or a kill is recorded and rolled back.
- **Every mode ends in `ensure_good`**: stop the engine, restore `.env.example`, install `GOOD_SHA` on all four ranks if they do not already run it
  (`inkling2.sh build` of that commit; on failure the same for the fallback), start it, wait for `/health`.
- **Signals.** Children run in the background and the script `wait`s, so a TERM or INT is handled at once: the whole process tree of the child is
  killed (`killtree`), then the known-good engine is restored. A `kill -9` of the wrapper itself is the one case it cannot cover: run
  `scripts/safeloop.sh restore` by hand.
- **Timeouts** are per run (`TIMEOUT_S`, default 4,200 s for a full run, which takes about 25 minutes).

## An experiment, step by step

1. Write the change in the dev tree. A change that must not alter any bit (everything of Segment A) gets an environment switch that restores the old
   path (`REXL3_INKLING2_FUSE_KV=0`, `..._RS_BF16=0`, `..._AG_BF16=0`, `..._COMMIT_LAYERS=0`), so one build can be measured on and off, and a test that proves
   the claim at the level of the primitive (a kernel pair, a collective) before anything runs end to end.
2. `REXL3_NO_CUDA=1 cargo check -p gb-model --lib` and `-p gb-cuda --tests` in the dev tree (there is no nvcc on the Mac: a `.cu` error shows only at build).
3. Primitive tests on the cluster, under the wrapper: `scripts/safeloop.sh cmd 4800 "export LAB_SRC=... CANDIDATE_SRC=...; scripts/ops_dev.sh inkling2_ops FILTER...; QUAD_SIZES=64 scripts/quad_dev.sh"`.
4. Screens: `scripts/ab_screen.sh warm,decode,prefill-32k 'on|' 'off|KEY=0 ...'` repeated to interleave, from the frozen worktree. Every screen prints `PASS pin`
   (every reply byte for byte against `bench/reference.json`). Boot-to-boot spread is about 1% of decode and 3% of prefill: one sample says little, two
   interleaved pairs say something.
5. Promote: commit on the dev branch, fast-forward `inkling2-a`, `scripts/safeloop.sh run LABEL SHA "" 5400`.
6. Before any claim: `scripts/logits_check.sh` and `PHASES=prefill-1m ./autoresearch.sh`, both under `safeloop.sh cmd`; `scripts/ops_dump_smoke.sh` after any change to a collective.

## What the loop does not do

- It does not touch `config/pins.sh`, `bench/reference*.json`, `oracle/`, the gate thresholds or the memory guard's floors (off limits since the first session).
- It cannot recover from a node that is down or from a lost network: `ensure_good` fails loudly (`RESTORE-FAILED` in the history, exit 1) and the cluster is left stopped.
- It does not run the oracle gate: bit-changing work (Segment B) needs a new pin and the oracle flow of `docs/ORACLE-GAP.md`. Segment B1's re-pin (`config/pins.sh`, `bench/reference*.json`) was made by hand,
  authorised by the user, never by the loop, and lives on the recipe branch `segment-b1-candidate` (the session branch keeps Segment A's pin); nothing of B was promoted; the teacher-forced gate needs an
  `mxint8` emulation in `oracle/` (an edit the loop may not make).

## Mistakes the loop has already caught or made (kept so they are not repeated)

- A second job started while a harness run was booting stopped the boot (runs 18 and 19 of the session failed for it). The lock is the answer.
- A screen that spans a commit builds the new HEAD for its later boots (the all-knobs-off arm of one screen ran a build with one more commit than the on arm).
- A lab build leaves artifacts newer than any commit: lab builds go to the lab tree and target directory (`docs/AUTORESEARCH-HANDOFF.md` section 6).
- A choice of collective derived from state that only rank 0 has (the oracle dump's capture) is a size mismatch inside one NCCL group. Every choice a rank makes
  between collectives must be a function of the message size and of flags every rank has.
- `ops_dev.sh` once ran a stale test binary after a compile error (the build step's output was filtered, its exit status lost) and printed passing tests of the previous build: it now stops on a failed
  build and fails on a failing or empty filter.
- Editing a script a running job executes changes the job (bash reads a script as it runs): scripts are edited between runs, never during one.
- A quantization format chosen on one metric can fail another: e4m3 elements failed one spec-oracle fixture at a near-tie, int8 elements passed all four (record `bench/records/2026-11-22-segment-b1/`). Decide
  from the gates the project already has, and report what a single sample cannot show.
- A bits-changing change moves the benchmark's own reply, and decode speed depends on the reply: one looping reply inflated the harness's decode mean by 10.6 tok/s. Read the per-prompt numbers and, with
  `scripts/loop_probe.sh`, a population of prompts before quoting a gain.
