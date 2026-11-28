# PR #3392: Inkling 2 native batched decode and padded prefill

**Date**: 2026-11-24
**Status**: Draft PR open (closes #3377). Narrow suites pass; validated on an M3 Ultra (512 GB) on `models/inkling-2-mini` and `models/inkling-2-mini-4bit`.

## Executive Summary

Inkling 2 served B > 1 by running every row as its own forward, so concurrent requests got no batching gain, and it declined padded prefill. This PR runs one forward over every row and lets the server prefill a cohort of cold prompts in one padded pass. At 8 concurrent requests the server's aggregate throughput goes from 25.9 to 62.7 tok/s on the native checkpoint and from 30.3 to 62.7 tok/s on the 4-bit export. B = 1 is byte-identical to main. B = 4 and B = 8 agree with B = 1 on every decided trace position.

## 1. Problem Statement

`batched_decode_kind()` returned `PerRowFallback`: `forward_batched_with_ids` sliced each row and called `forward_with_sequence_id`, so decode cost grew linearly with the batch. Every one of Mini's 48 layer visits owns an `InklingLayerCache` per sequence (a chronological KV cache plus four f32 convolution histories), and those histories differ in length between rows, so the states cannot simply be stacked.

| Risk | Impact | Handling |
|---|---|---|
| Row states of different lengths mixed in one tensor | Wrong context for a short row | Histories stacked left-padded; the padding never reaches the `kernel - 1` rows a convolution reads |
| Per-row trim no longer exact | Speculative rollback or padding trim corrupts a row | Each row gets back exactly the history its own forward keeps |
| B = 1 output or kernels change | Regression for single-request users | One row keeps the single-sequence graph |
| Padding read by real positions | Wrong prefill | Right padding, causal masks and convolutions, exact per-row trim |

## 2. Design

### 2.1 Batched forward (`src/models/inkling2/batched.rs`)

`forward_rows` opens the rows' states with `with_batched_sequence_states` (creating fresh caches for a cold prefill row) and runs the existing hidden pass with one cache set per row. Projections, the four short convolutions, attention residual mixes, Mini's message bank and controllers, the experts and the head see `[B, L]`. The attention core (log scaling by each row's offset, KV append, sliding trim, banded relative mask, sink column, SDPA) runs per row, as Inkling 1 does since #2673; `Inkling2Attention::forward` was split into the batched part and `attend`.

`with_stacked_state` handles a convolution slot. One row passes its own state through unchanged. Several rows are stacked to `[B, R_max, C]` with zero rows prepended to the shorter histories. After the convolution, row `i` keeps the newest `min(kernel - 1 + margin, R_i + L)` rows of the stacked next history, the same rows its single-row forward keeps. Every history holds at least `kernel - 1` rows, so the padding is never read.

### 2.2 Unchanged paths

`route_batched_decode_with_ids` sends one row to `forward_with_sequence_id`, so B = 1 runs the old graph. The #3369 gates are untouched: the short convolution kernel still takes up to 4 rows (it reads a batched history through its strides), head preparation and the grouped experts stay `B == 1`, so B > 4 runs the ops convolution. The DSpark verify forward calls `hidden_with_captures`, now a one-row wrapper over the same pass.

### 2.3 Padded prefill

The scheduler right-pads a cohort and trims each row's padding through `trim_state`. Every Inkling 2 mask and convolution is causal, so real positions never read padding, and the sliding KV margin already covers any trim up to the forward length. A convolution history keeps only 16 extra rows outside a speculative verify. A batched forward of more than one token therefore enters `conv::WideHistoryScope`, which keeps as many rows as the forward appends. `supports_batched_prefill` and `supports_padded_prefill` are now true, also through `Inkling2VlModel`. The wider history lives from the cohort forward until the next decode step drops it.

## 3. Validation

| Check | Result |
|---|---|
| Tiny model, B = 3 decode, rows at 3/12/30 tokens, trim on one row, a row leaving | equal to single-row decodes within 2e-4 (observed 3e-5, f32) |
| Tiny model, cohort right-padded to 48 (padding 43/25/8) | last-row logits, KV, convolution tails and the next 4 decode steps equal unpadded prefill |
| Mutation: padding on the wrong side of the stacked history | both tests fail (diff about 3) |
| Traces B = 1, main against PR | byte-identical, both checkpoints |
| Traces B = 1 against B = 4 / B = 8 | native 2/256 and 3/256 top-1, 0/98 decided; 4-bit 7/256, 0/98 decided; largest gap 0.19 |
| Greedy 128 tokens | identical between main and PR, both checkpoints |

Server throughput, ABAB, 501-token prompts, 128 tokens, `--parallel 8`, aggregate tok/s (mean of two rounds):

| Checkpoint | C1 main / PR | C4 main / PR | C8 main / PR |
|---|---|---|---|
| inkling-2-mini | 26.5 / 26.5 | 26.6 / 48.6 | 26.0 / 62.6 |
| inkling-2-mini-4bit | 30.7 / 30.7 | 30.9 / 50.6 | 30.3 / 62.9 |

Time to first token at C8 drops from about 4.2 s to 3.6 s (native) through the batched prefill cohort. Selectors: `--lib inkling` (168), `--lib coverage_tests` (18), `--lib prefill_cohort` (23), `--lib pad_trim` (4).

## 4. Limitations and follow-ups

- Free-running concurrent requests at temperature 0 are not byte-identical to their solo answers. They split after 47 to 238 characters at near-ties, and with `--max-batch-prefill 1` the splits stay, so they come from batched decode arithmetic (matmuls over B rows), the class the traces measure.
- The attention core still runs once per row. A left-padded batched SDPA with per-row masks could remove that loop if profiles show it matters.
- Inkling 2 Small is not measured here (#3373); its path is the same without the loop.
