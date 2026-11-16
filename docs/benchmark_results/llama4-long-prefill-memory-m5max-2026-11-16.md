# Llama 4 Scout single long-forward memory (M5 Max, 2026-11-16)

Issue #3132. The Llama 4 RoPE measurement ([`llama4-rope-m5max-2026-11-16.md`](llama4-rope-m5max-2026-11-16.md)) could not run a single 9216-wide forward of `llama-4-scout-17b-16e-instruct-4bit` through `examples/logit_trace`: it aborted with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Per-layer instrumentation shows the model forward is not the cause. The abort came from the trace harness scoring all 9216 rows over the 202,048-entry vocabulary at once.

## Setup

- Host: Apple M5 Max, 128 GB, macOS 27. GPU shared with another job under a lock, one GPU process at a time.
- Checkpoint: `models/llama-4-scout-17b-16e-instruct-4bit` (60.7 GB active once loaded).
- `main` is `5ee08ec6`. The forward and whole-chunk scoring rows were read from a release build of `main` with temporary, environment-gated readouts of MLX `get_active_memory` / `get_peak_memory`, with `reset_peak_memory` after each readout. In per-layer mode every layer's attention and feed-forward output is evaluated before reading. In whole-forward mode only the embedding and the `lm_head` output are, so the forward runs as one lazy graph like a plain call. The `branch` rows come from this change.
- Corpus: `docs/architecture.md` followed by `docs/installation.md`, the same as the RoPE measurement.

## Where the memory goes

Peak memory in GB (`llama4-long-prefill-memory-m5max-2026-11-16/peak-memory.csv`):

| Stage | Width | main | branch |
|---|---|---|---|
| Weights loaded | 9216 | 60.74 | |
| Largest per-layer stage (layer 47 MoE FFN), per-layer eval | 9216 | 64.03 | |
| Forward as one lazy graph, through `lm_head` | 4608 | 64.12 | |
| Forward as one lazy graph, through `lm_head` | 9216 | 67.38 | |
| Forward as one lazy graph, through `lm_head` | 16384 | | 72.60 |
| Trace scoring | 4608 | 85.79 | |
| Trace scoring | 9216 | out of memory | 71.18 |
| Trace scoring | 16384 | | 75.48 |

With per-layer evaluation no stage peaks more than 0.9 GB above the memory live before it, and live memory grows by the 38 MB of K/V each layer appends. The whole 9216-wide forward run lazily peaks at 67.38 GB, 6.6 GB above the weights, of which 3.3 GB is the `[1, 9216, 202048]` bf16 logits it returns. None of the issue's suspects shows up: the chunked layers take no dense mask (the new `a_one_row_prefill_past_the_chunk_builds_no_dense_chunk_mask` test pins this), and neither the MoE intermediates nor unevaluated cache state grows with width beyond that.

The excess was in `emit_rows_to`. It cast the whole chunk's logits to f32, took a log-softmax and ran `argpartition` over every row at once. At 4608 rows that added 22.3 GB on top of the forward's 63.5 GB, and at 9216 rows the same work needed roughly twice that, past what the GPU can map next to 66 GB of live weights, K/V and logits. Scoring 1024 rows at a time bounds it at 4.8 GB above the logits whatever the width, so a 16384-wide forward traces at 75.5 GB.

## Output

Every scoring step is per row, so blocks write the same trace: the two-4608 trace from the branch is byte-identical to the one from `main`, and `blocked_scoring_matches_whole_chunk_scoring` checks blocks of 1, 2, 4, 6 and 1024 rows against whole-chunk scoring.

The single 9216-wide forward against the two 4608-wide forwards it replaces, `scripts/compare_logit_traces.py --decided 2` (`llama4-long-prefill-memory-m5max-2026-11-16/trace-summary.csv`):

| Reference | Candidate | Positions | Top-1 disagreement | Decided | Decided disagreement | ppl reference | ppl candidate |
|---|---|---|---|---|---|---|---|
| Two 4608-wide forwards (`4608 2 8 4608`) | One 9216-wide forward (`9216 1 8 0`) | 9216 | 0.358% (33) | 3651 | 0.000% | 11.7995 | 11.7967 |

Every disagreement sits where the reference's top two were less than 0.44 apart, and 32 of the 33 land at the candidate's rank 2: the rounding class that a different forward width gives the quantized matmuls. The 16384-wide single forward traces 16384 positions at perplexity 10.283.

## Generation

`mlxcel generate --temp 0 -n 64` on the corpus head with a summary request (8,584 prompt tokens), run with `MLXCEL_PREFILL_CHUNK=0` (one prefill forward) and with the default 2048-token chunks, wrote the same 64 tokens. The same command at 15,563 prompt tokens with `MLXCEL_PREFILL_CHUNK=0` completed with a coherent summary. Both prompts are past the 8192 attention chunk, so the decode steps run past it too. No model code changed.
