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
- [ ] oracle: fp64 reference (`oracle/`, done) + real-pack fixtures (running on rank 0)
- [x] engine family `inkling2` host half: config, inventory, bind (E mod 4), memory model, chat format
- [ ] kernels: attention (relative logits, log scaling, l2 clips, sinks, gate, window/full, int8 KV), short convs,
      softcapped MoE activation, fp32 router, AttnRes mix, logit softcap, bf16-input linear
- [x] forward assembly + TP4 bring-up: first four-rank REXL3 group on the switch (engine `inkling2-fwd`, record
      `bench/records/2026-11-18-inkling2-tp4-base`)
- [ ] KV pool with 1M lanes, DSpark drafter, vision tower
- [x] controller: doctor, fetch, verify, build
- [x] controller: start/stop/status/logs/smoke/print-serve/memwatch/record, memory guard (STOP_OTHER/RESTORE_OTHER: open)
- [ ] correctness ladder: smoke → probe → oracle compare → pin/check → drafted == serial → cached == uncached →
      needle → quality → tools → api-test; first boot record
- [ ] stop: hand off to the autoresearch phase

## Open questions

- 70 tok/s: DSpark acceptance (unmeasured) vs requantizing BF16 attention before the baseline (changes the pack).
- REXL3 four-rank groups are built but unmeasured upstream (`docs/examples/cluster.4node.json`).
- REXL3 refuses `json_schema` response formats; acceptable for the clients?
- Pack identity under split staging: settled. The engine's fingerprint of a split-staged folder is the split form (config, every
  plain tensor, the index's tensor-to-shard manifest), the same on every rank: `sha256:71d760435893535f318c51038935514e4e93fe5d58507edebc78c3a75cda5079`
  (compared at the TP boot gate). Pin it in a `VALIDATED_*` entry before dropping `--allow-unvalidated-pack`.
