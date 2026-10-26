# Gemma 3n native batched decode: Metal validation

Issue #2561 / PR #2628. Measured on an Apple M3 Ultra with 512 GiB memory, using `metal,accelerate` release builds. Baseline source: `1cb18ccc`; candidate source: `f01241ba` (later commits only add reports). Both use MLX pin `7ca40033`. Baseline measurements were collected on October 26 and candidate measurements on October 27, 2026, Korea time.

## Serving results

The server used `--parallel 4 --ctx-size 32768 --ignore-eos --cache-ram 0`. The concurrency script used a nominal 512-token plain prompt and 128 output tokens, after a 32-output-token warmup. Values are single-run aggregate output tokens per wall-clock second, including prefill; they are not pure decode rates or a statistical confidence estimate. No other validation workload ran during the candidate measurements.

| Checkpoint | Main C=1 | Native C=1 | Main C=2 | Native C=2 | Main C=4 | Native C=4 | C=4 improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma-3n-e2b-it-4bit | 87.6 | 88.3 | 124.3 | 146.0 | 128.6 | 221.5 | 72.2% |
| gemma-3n-e4b-it-4bit | 68.5 | 68.4 | 97.9 | 110.8 | 100.2 | 162.8 | 62.5% |

C=1 changed by +0.8% and -0.15%, respectively; this single sample shows effectively unchanged performance, not proof of zero variance. Native C=4 is 2.51x C=1 on E2B and 2.38x on E4B. Every request succeeded. Both candidate startup logs report `decode path: native batched forward`; baseline logs report per-row fallback. The optional `/metrics` endpoint was unavailable, so no per-step metric attribution is claimed.

## Correctness and numerical differences

`mlxcel generate -m models/<checkpoint> -p Hello -n 128 --temp 0` produced identical completion bytes before and after: E2B 220 bytes, E4B 188 bytes, excluding loading/timing lines and retaining the echoed prompt. Both stopped naturally after 51 generated tokens. The complete B=1 teacher-forced trace files also match the baseline byte-for-byte.

The trace command used `docs/architecture.md 64 4 8 512` with `MLXCEL_TRACE_BATCH_ROWS=1` and `=4`, covering 256 positions per checkpoint. Decided positions have a reference top-two logit gap of at least 2.0.

| Checkpoint | Top-1 disagreement | Decided disagreement | Largest reference gap at disagreement | Perplexity B=1 → B=4 |
|---|---:|---:|---:|---:|
| E2B | 2/256 | 0/147 | 0.125 | 27.3600 → 27.5361 (+0.643%) |
| E4B | 4/256 | 0/150 | 0.125 | 22.8674 → 22.6462 (-0.967%) |

All disagreeing reference tokens rank second in the candidate. This is consistent with near-tie numerical sensitivity across batch-shaped kernels, but the B=4 graph is not bit-identical: maximum logit deltas on the reference token are 6.25 (E2B) and 10.9375 (E4B). No decided-position mismatch was observed in this sample. Baseline fallback B=1/B=4 traces have no differences.

## Validation and limits

- Release build, workspace/all-targets Clippy with `-D warnings`, and formatting passed.
- Workspace `test-fast` with `metal,accelerate`, serial test execution, and `--no-fail-fast`: 13,826 passed, 10 failed, 441 ignored (nested filtered child runs excluded).
- All 10 failures are the same SAM3, SAM3D Body, Sapiens2 and DINOv2 numerical-oracle failures recorded during #2546. Names, panic locations, and actual/expected values match exactly. They were independently reproduced on clean `01047d75` before either change; implicated files remain identical and #2561 does not change their dependencies. The suite is therefore not wholly green, but no new failure was found.
- Focused Gemma 3n tests cover three different prompt lengths across a sliding-window boundary, E2B/E4B-shaped KV sharing, full/sliding attention, per-layer inputs, B=0 and exact B=1 routing. Coverage tests pin native text and delegating VLM classification.
- CUDA and ROCm checkpoint performance were not measured on this host. Paged-cache support and audio/vision tower batching are outside this change.

Raw local evidence is under `/private/tmp/mlxcel-chain-evidence/{main2561,branch2561}-gemma-3n-*`; full test output is `/private/tmp/mlxcel-chain-2561-tests.log`. These temporary paths are local evidence, not durable repository artifacts; the numerical results and limitations are preserved here.
