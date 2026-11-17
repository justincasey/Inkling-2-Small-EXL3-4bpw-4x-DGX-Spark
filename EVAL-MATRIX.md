# Eval matrix

Run in this order. Stop at the first failure.

| # | Gate | Tool | Pass condition | Proves |
|---|---|---|---|---|
| 1 | boot | `start` | `/v1/models` answers; ready memory logged | the stack loads and the guard held |
| 2 | smoke | `smoke` | chat coherent, tool call parses, reasoning separated, quant dispatch is the validated mode | the serving path is not silently wrong |
| 3 | probe | `probe` | models/max_model_len, tokenize, usage, reasoning field, tool parser, spec counters, effort | every later measurement is meaningful |
| 4 | spec | `bench spec` | tokens/step > 1.0 | the draft is accepted, not merely loaded |
| 5 | token pin | `bench pin` then `check` | first run writes; later runs match | numerics/regression oracle |
| 6 | needle | `bench needle` | code found at 10% and 50% depth | long-context recall at the profile's window |
| 7 | quality | `quality` | baseline recorded; later runs >= baseline | quantization did not break answers |
| 8 | tools | `tools` | baseline recorded; schema-valid calls | agent surface works |
| 9 | API | `api-test` | 0 FAIL | client contract |
| 10 | speed | `bench decode/prefill` | raw numbers, then compare to a record | throughput |
| 11 | longctx | `longctx` | PASS at every length <= window | recall + prefill curve |
| 12 | SixCat | `sixcat` | score recorded | cross-recipe comparability |

Warm-up: one short + one long request before any timed run. Discard one 20 s decode pass after boot.
Never count SSE chunks for tokens; use `usage.completion_tokens`.
