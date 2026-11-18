# Plan — Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark

## Lane

- pack: `0xSero/Inkling-2-Small-EXL3-4bpw` @ `996dc404c1feb3abd5330af8a3c52270ffe0008d`, staged split per rank
- topology: TP4 on 192.168.1.121 (rank 0) .122 .123 .18, CRS804 switch
- engine: REXL3 `upstream/main` @ `06181b1` (v0.1.2), branch `inkling2` (`~/projects/rexl3-inkling2`)
- targets: 1M per lane × 4–8 lanes; decode 70 tok/s; prefill 4,000+ tok/s; byte-exact (`docs/DECISIONS.md`,
  `docs/ROOFLINE.md`)

## Phases

This phase ends at a working, verified base recipe (user, 2026-11-17): boots TP4, serves, passes the correctness
gates, first boot recorded. Performance toward the targets is the next phase (autoresearch harness + iterations).

- [x] intake: manifest, census, per-rank file split (`intake/`, `config/pack-ranks.tsv`), fit (`docs/FIT.md`)
- [x] staging: `./inkling2.sh fetch` + `verify` on all four nodes (91.02 GB each, sha256-verified)
- [x] spec: op-by-op forward of target, drafter, vision (`docs/PORT-SPEC.md`, `docs/NUMERICS.md`)
- [x] oracle: fp64 reference (`oracle/`) + real-pack fixtures (4 prompts incl. one past the 1024 window); gap explained,
      binding op-level gate (`docs/ORACLE-GAP.md`)
- [x] engine family `inkling2` host half: config, inventory, bind (E mod 4), memory model, chat format
- [x] kernels: attention (relative logits, log scaling, l2 clips, sinks, gate, window/full, int8 KV), short convs,
      softcapped MoE activation, fp32 router, AttnRes mix, logit softcap, bf16-input linear, vision tower, DSpark drafter
- [x] forward assembly + TP4 bring-up: first four-rank REXL3 group on the switch (`bench/records/2026-11-18-inkling2-tp4-base`)
- [x] 4 x 1M int8 lanes, DSpark wired, prefix cache, vision through the group (engine `f4e27ca`, pinned)
- [x] controller: doctor, fetch, verify, build, start/stop/status/logs/smoke/print-serve/memwatch/record,
      stop-other/restore-other, memory guard
- [x] correctness gates (`bench/records/2026-11-18-inkling2-final-base`): smoke (chat, completion, tool call, reasoning,
      image), oracle, drafted == serial, cached == uncached, concurrent == solo, chunk width, boot to boot, 4 x 1M needles.
      Not run in this phase: a quality benchmark and the API conformance suite (next phase's harness).
- [x] stop: hand off to the autoresearch phase

## Open questions

- 70 tok/s: DSpark acceptance is 0.41 (3.8 tokens/round) on the gate prompts; requantizing BF16 attention before a
  new baseline (changes the pack) remains the other lever. Decode is eager (no CUDA graphs) and unmeasured here.
- Four-rank REXL3 groups now run on this switch (this recipe); upstream still lists R > 2 as unmeasured.
- REXL3 refuses `json_schema` response formats; acceptable for the clients?
- Pack identity under split staging: settled. The engine's fingerprint of a split-staged folder is the split form (config, every
  plain tensor, the index's tensor-to-shard manifest), the same on every rank: `sha256:71d760435893535f318c51038935514e4e93fe5d58507edebc78c3a75cda5079`
  (compared at the TP boot gate). Pin it in a `VALIDATED_*` entry before dropping `--allow-unvalidated-pack`.
