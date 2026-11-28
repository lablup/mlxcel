# PR #3417: Acceptance-aware DSpark verify width for Inkling 2

Issue #3410, follow-up to #3280 / PR #3405.

## Problem

PR #3405 made the Inkling 2 verify block byte-identical to decode, which enabled DSpark by default. On Small the drafter still lost on prose (0.90x classic). The cause is cost. Small routes every token to 8 of 384 NVFP4 experts, so each extra verify row adds about 8 more experts' worth of weight reads. An 8-row round measured 150 ms on the gather path against a 52 to 56 ms decode step. Prose accepts about 2.3 tokens per round, which is less than that round costs.

## Mechanism

1. **Width chosen per round.** The row-invariant forward from #3280 makes a block of any width from 1 to 16 equal to that many decode steps, so a target can declare `exact_at_every_verify_width`. For such targets the round loop uses `ExpectedThroughputWidth`, which works as follows:
   - It keeps a decayed estimate of the conditional acceptance at each draft position, with a prior that carries the previous position's value forward.
   - It keeps an EMA of the measured cost of each width it has run, and interpolates linearly for widths it has not.
   - Each round it picks the width with the highest E(tokens)/C(width). Width 1 is a classic decode step with no draft.
   - After 8 classic steps in a row it re-tries a drafted width, so the drafter can come back when the text turns repetitive.
   - The first round at any width is not used as a cost sample. It compiles that width's kernels (230 to 260 ms against 80 steady), and in early runs that one sample made the policy abandon the width for good.
2. **Pipelined classic steps.** When run one at a time, classic steps cost 57 to 58 ms against a 52 ms decode step, because each step waited for its own read-back. `classic_steps.rs` now builds step i+1 on the lazy argmax of step i, the same way classic decode does. It keeps every step's captured hidden row so the next draft sees the full committed context.
3. **Grouped expert matvec on Small.** Upstream gates this kernel on `loop_config`, and Small has none. The kernel decodes each expert's weights once for all the rows routed to it, and its output equals the gather path row for row. Measured on Small:

   | Rows | gather path | grouped kernel |
   |---|---|---|
   | 3 | 81 ms | 75 ms |
   | 8 | 150 ms | 129 ms |

## Results

Small, 256 greedy tokens, ABAB:

| Prompt | Before | After | Target |
|---|---|---|---|
| prose | 0.90x | 1.12x (21.6 vs 19.3 tok/s) | 1.0x |
| code | 1.54x | 1.58x (30.4 vs 19.2 tok/s) | 1.5x |

- In every run the output text equals classic decode.
- The policy settles on mostly 3-row rounds for prose and on 3 to 8 rows for code.
- Mini shows no regression. Its prose gain over the fixed width is larger.

## Other items

- **Server prefill chunk.** The server burst used to prefill Inkling 2 in one forward. It now follows `--prefill-chunk-size` through the new `DFlashTargetModel::prefill_burst` hook. Other targets keep their default.
- **Mini 4-bit per-row projections.** These cost at most 4.1 ms per 8-row round, an upper bound because the plain comparison forward also skips per-row sliding attention. A quantized row-invariant kernel is not justified for that saving.

## Risks

- The policy depends on wall-clock costs, so heavy host load can move it toward narrower widths. Output text cannot change, only throughput.
- The grouped kernel was checked on Small with the per-visit diagnostic at widths 2, 4 and 8, and also after an 1100-token prompt.
