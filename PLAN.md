# Plan — Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark

## Lane

- pack: `0xSero/Inkling-2-Small-EXL3-4bpw` @ `996dc404c1feb3abd5330af8a3c52270ffe0008d`, staged split per rank
- topology: TP4 on 192.168.1.121 (rank 0) .122 .123 .18, CRS804 switch
- engine: REXL3 `upstream/main` @ `06181b1` (v0.1.2), branch `inkling2` (`~/projects/rexl3-inkling2`)
- targets: 1M per lane × 4–8 lanes; decode 70 tok/s; prefill 4,000+ tok/s; byte-exact (`docs/DECISIONS.md`,
  `docs/ROOFLINE.md`)

## Phases

- [x] intake: manifest, census, per-rank file split (`intake/`, `config/pack-ranks.tsv`), fit (`docs/FIT.md`)
- [ ] staging: `./inkling2.sh fetch` + `verify` on all four nodes
- [ ] spec: op-by-op forward of target, drafter, vision (`docs/PORT-SPEC.md`)
- [ ] oracle: layer-at-a-time reference over the pack, token/logit fixtures
- [ ] engine family `inkling2`: config, loader/bind (E mod 4 expert split), forward on one rank vs oracle
- [ ] kernels: attention (relative logits, log scaling, l2 clips, sinks, gate, window/full), short convs, softcapped
      MoE activation, logit softcap, attention-residual blocks
- [ ] TP4 bring-up: first four-rank REXL3 group on the switch
- [ ] KV pool with 1M lanes, DSpark drafter, vision tower
- [ ] controller: doctor/build/start/stop/status/logs/smoke/print-serve, memory guard, STOP_OTHER/RESTORE_OTHER
- [ ] eval ladder: smoke → probe → spec → pin/check → needle → quality → tools → api-test → bench → longctx
- [ ] first boot record, README numbers, autoresearch toward the targets

## Open questions

- 70 tok/s: DSpark acceptance (unmeasured) vs requantizing BF16 attention before the baseline (changes the pack).
- REXL3 four-rank groups are built but unmeasured upstream (`docs/examples/cluster.4node.json`).
- REXL3 refuses `json_schema` response formats; acceptable for the clients?
