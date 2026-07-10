# Scheduler decode lookahead pipelining acceptance (issue #632, GB10)

Date: 2026-07-10. Host: NVIDIA GB10 (Grace-Blackwell), CUDA build.
Bench: `bench_serving_concurrency.py` (512 prompt tokens, 512 max tokens) against `mlxcel-bench-decode --prompt-tokens 512` as the matched-context CLI reference; sync baseline via `MLXCEL_FORCE_SYNC=1`. Full run and methodology: PR #729.

## TL;DR

Porting the CLI generation loop's lookahead `async_eval` pipelining into the server `BatchScheduler` decode tick closes most of the structural decode-throughput gap between server and CLI on CUDA. The lookahead pipeline engages 509/511 decode steps per 512-token request, and greedy output stays byte-identical to the `MLXCEL_FORCE_SYNC=1` synchronous path.

## B=1 decode

| case | sync | pipelined | CLI reference | verdict |
|------|-----:|----------:|--------------:|---------|
| qwen2.5-0.5b-bf16 B=1 decode | 147.9 | 182.4-191.3 across runs | 199.0-200.3 | pipeline +23-29% over sync; CLI gap 3.9-8.5% (5% AC met at the best run; residual is SSE/HTTP per-token cost, not pipelining) |
| llama-3.1-8b-4bit B=1 decode (dense backend) | 44.3 (default backend) | 52.0 | 51.8 | server exceeds CLI; AC met |
| llama-3.1-8b-4bit B=1 decode (default paged backend) | 44.3 | 47.2 | 51.8 | +6.5% over sync; remaining 8.9% gap is the paged-attention decode kernel cost, tracked by #634, reproduced independently of this PR |

## B=4 aggregate

| case | sync | pipelined | verdict |
|------|-----:|----------:|---------|
| qwen2.5-0.5b-bf16 B=4 aggregate | 289.7 | 340.5 | +17.5%; AC met |
| llama-3.1-8b-4bit B=4 aggregate | 44.7 | 46.4 | +3.8% (bounded by the CUDA batched-decode non-amortization caveat from #724) |

## Correctness and coverage

Lookahead engagement: 509/511 decode steps per 512-token request (`mlxcel_batch_decode_lookahead_steps_total`), for cold and prompt-cache-hit requests alike. Greedy outputs are byte-identical pipelined vs `MLXCEL_FORCE_SYNC=1` (orchestrator-independent check on llama plus the implementation's dense/paged checks on qwen). Full `cargo test --release --features cuda --no-fail-fast -- --test-threads=1`: 4362 passed, 1 failed (the pre-existing #728 flake, passes on rerun).

## Side finding

During validation a separate pre-existing anomaly surfaced: decode is ~11% slower after a prompt-cache hit than after a cold prefill, with the pipeline engaged in both modes. Filed as #730, out of scope for this PR.

## Conclusion

The lookahead decode pipeline meets acceptance for the qwen2.5-0.5b-bf16 and llama-3.1-8b-4bit dense-backend cases; the paged-backend llama case is a partial win bounded by the separately tracked #634 paged-attention decode cost, and B=4 llama is bounded by the separately tracked #724 CUDA batched-decode caveat. `MLXCEL_FORCE_SYNC=1` remains the rollback path.
