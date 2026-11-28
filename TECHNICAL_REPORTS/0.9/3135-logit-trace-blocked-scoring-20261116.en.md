# Technical Report: PR #3135 - Score logit_trace rows in blocks so a 9216-wide forward fits

**Date**: 2026-11-16
**Status**: Completed
**Risk Level**: Low (measurement harness only; no model or runtime code changed)

## Executive Summary

Issue #3132 reported that a single 9216-token Llama 4 Scout forward ran out of Metal memory on `main`. Per-layer peak-memory readouts showed the model forward is not the cause: a lazily run 9216-wide forward peaks at 67.38 GB, 6.6 GB above the 60.74 GB of weights, and 3.3 GB of that is the returned logits. The abort came from `examples/logit_trace`. `emit_rows_to` scored every row of the chunk over the 202,048-entry vocabulary at once (f32 cast, log-softmax, `argpartition`). The harness now scores 1024 rows at a time, so a 9216-wide trace peaks at 71.18 GB and a 16384-wide trace at 75.48 GB.

## 1. Problem Statement

The Llama 4 RoPE measurement (#3126) had to replace one 9216-wide forward with two 4608-wide forwards, because the single forward aborted with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. The note it left suggested a model-side memory problem past the 8192 attention chunk. If that had been true, every long single-pass Llama 4 prefill would be affected, not only the trace.

## 3. Technical Decisions

- **Instrument before fixing.** The issue listed three model-side suspects: a dense chunked mask, MoE intermediates growing with width, and lazy cache state keeping prefill inputs alive. Per-layer readouts (MLX `get_active_memory` / `get_peak_memory` with `reset_peak_memory`) ruled out all three. No stage peaked more than 0.9 GB above the memory live before it, and each layer added only its 38 MB of K/V. The readouts were temporary and are not in the PR.
- **Block the scoring, not the forward.** Every scoring step is per row, so scoring in blocks gives exactly the same output at a bounded peak. With 1024-row blocks, scoring adds 4.8 GB above the logits at any width. Whole-chunk scoring added 22.3 GB at 4608 rows and roughly twice that at 9216.
- **Pin the model-side property anyway.** A new test asserts that a one-row prefill past the chunk builds no dense chunk mask, so the class of memory growth that was ruled out cannot come back unnoticed.

## 7. Change Summary

| Stage | Width | Peak, main (GB) | Peak, branch (GB) |
|---|---|---|---|
| Forward as one lazy graph, through `lm_head` | 9216 | 67.38 | |
| Forward as one lazy graph, through `lm_head` | 16384 | | 72.60 |
| Trace scoring | 4608 | 85.79 | |
| Trace scoring | 9216 | out of memory | 71.18 |
| Trace scoring | 16384 | | 75.48 |

Output:
- The two-4608 trace is byte-identical before and after the change.
- One 9216-wide forward against two 4608-wide forwards: 0 of 3651 decided positions disagree. Overall top-1 disagreement is 0.358%, every case at a reference gap under 0.44. Perplexity is 11.7995 against 11.7967.
- `mlxcel generate --temp 0` at 8,584 prompt tokens writes the same 64 tokens with single-pass prefill as with the default chunking. A 15,563-token single-pass prefill completes.

Tests:
- `blocked_scoring_matches_whole_chunk_scoring` checks blocks of 1, 2, 4, 6 and 1024 rows against whole-chunk scoring.
- `a_one_row_prefill_past_the_chunk_builds_no_dense_chunk_mask` covers the model-side property.

Results: `docs/benchmark_results/llama4-long-prefill-memory-m5max-2026-11-16.md`.

## 8. Follow-up Actions

- None for Llama 4. Other harnesses that score a whole chunk over a large vocabulary (`examples/perplexity`) could hit the same limit at long widths; this PR does not change them.
