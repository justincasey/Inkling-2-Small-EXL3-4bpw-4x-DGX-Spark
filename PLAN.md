# Plan — Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark

## Lane

- pack: `0xSero/Inkling-2-Small-EXL3-4bpw` @ `996dc404c1feb3abd5330af8a3c52270ffe0008d`
- topology: TP=4 on 192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18
- engine: TODO

## Phases

- [ ] intake: manifest, census, overlay, fit (`intake/`)
- [ ] pins: image digest, fork shas, patch, overlay sha
- [ ] controller: doctor/fetch/verify/build/start/stop/status/logs/print-serve
- [ ] static tests: print-serve invariants, gates, manifest/census/overlay, patch apply
- [ ] eval harness: smoke, probe, bench, quality, tools, longctx, api-test, sixcat
- [ ] first boot: record ready memory, guard minimum, KV capacity
- [ ] correctness: quant dispatch, spec acceptance, token pin, needles, quality/tools
- [ ] records + README numbers
- [ ] autoresearch baseline

## Open questions

- TODO
