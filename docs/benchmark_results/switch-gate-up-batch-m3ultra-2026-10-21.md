# Switch gate/up kernel batch decode A/B on M3 Ultra, 2026-10-21

Issue #2515. An A/B of the switch gate/up Metal kernel against the gather path at batch decode, served through the OpenAI-compatible server (`scripts/bench_serving_concurrency.py`), on `models/olmoe-1b-7b-0125-4bit` (64 experts, top_k 8, hidden 2048, dff 1024, affine q4 group 64, f16 activations and scales). This is the first real-checkpoint measurement of the extended `SwitchGLU::forward` wire-in; the single-stream baseline in `switch-gate-up-baseline-m3ultra-2026-10-21.md` covers inkling.

## What was measured

- Arm A: default env. The #268 fused single-token kernel at batch 1, and the switch gate/up kernel at batch 2 to 7 (`total < 64`).
- Arm B: `MLXCEL_FUSED_MOE=0`. The gather path everywhere.
- Four server boots in ABBA order (A, B, B, A), one discarded warm-up pass per boot, two measured passes of the concurrency 1..8 ladder, 512-token prompts, 128 max tokens, indexers paused via `scripts/with_indexers_paused.sh`, `caffeinate -i`.

## Coverage boundary

| concurrency | decode routes (tokens x top_k) | arm A kernel | arm B kernel |
|---|---|---|---|
| 1 | 1 x 8 = 8 | #268 fused single-token | gather |
| 2..7 | 16..56 | switch gate/up | gather |
| 8 | 64 | sorted path | sorted path |

Batch 8 is the control cell: both arms run the same sorted path. Batch 1 compares the #268 single-token kernel, not the switch gate/up kernel.

## Aggregate throughput (tok/s, ABBA means)

| conc | A (kernel) | B (gather) | delta |
|---|---|---|---|
| 1 | 363.4 | 347.1 | +4.7% |
| 2 | 503.5 | 522.7 | -3.8% |
| 3 | 629.6 | 598.6 | +5.2% |
| 4 | 615.8 | 581.9 | +5.8% |
| 5 | 457.9 | 450.7 | +1.6% |
| 6 | 493.1 | 486.1 | +1.4% |
| 7 | 520.7 | 515.4 | +1.1% |
| 8 | 528.2 | 525.2 | +0.6% |

Within-arm spread across the two passes is under 0.5% at every cell except batch 7 arm B (2.6%, one pass dipped to 507.2).

## Per-stream decode (tok/s, ABBA means)

| conc | A | B | delta |
|---|---|---|---|
| 1 | 403.0 | 384.0 | +4.9% |
| 2 | 277.3 | 288.9 | -4.0% |
| 3 | 226.3 | 230.2 | -1.7% |
| 4 | 169.2 | 168.5 | +0.4% |
| 5 | 188.7 | 189.6 | -0.5% |
| 6 | 186.1 | 188.1 | -1.1% |
| 7 | 173.4 | 176.8 | -1.9% |
| 8 | 152.6 | 155.2 | -1.7% |

## TTFT (ms, ABBA means)

| conc | A | B | delta |
|---|---|---|---|
| 1 | 38.2 | 39.3 | -1.1 |
| 2 | 47.9 | 47.7 | +0.2 |
| 3 | 46.1 | 83.7 | -37.6 |
| 4 | 75.4 | 117.7 | -42.3 |
| 5 | 277.1 | 311.5 | -34.4 |
| 6 | 419.8 | 454.7 | -34.9 |
| 7 | 534.0 | 571.6 | -37.6 |
| 8 | 627.9 | 652.3 | -24.5 |

TTFT jumps by an order of magnitude from batch 4 to batch 5 in both arms; that is a server batch-scheduler shape, not a kernel effect.

## Reading

- The control cell (batch 8, identical code) shows +0.6% aggregate and -1.7% per-stream, which is the A/B noise floor for this protocol.
- Batch 2 is the one attributable regression. Its TTFT is identical (47.9 vs 47.7 ms), so the aggregate gap comes from decode: per-stream 277.3 vs 288.9 tok/s, -4.0%, outside the control noise floor. At 16 routes the kernel is about 4% slower than the gather.
- Batch 3 to 7 show no measurable per-stream kernel effect: -0.5 to -1.9%, inside the control noise floor of -1.7%. Their aggregate gains (+1.1 to +5.8%) track a consistent TTFT advantage of 25 to 42 ms whose source this protocol cannot isolate; the 512-token prefill is 4096+ routes and runs the sorted path in both arms, so the advantage is not the prefill kernel. The plausible drivers are the first multi-token decode step or the #268 single-token kernel during batch admission, and separating them would need a dedicated trace.
- Batch 1 (+4.7% aggregate) is the #268 cell, consistent with its earlier validation, not a switch gate/up result.

## End-to-end parity

Four identical concurrent requests at batch 4, temperature 0, 32 max tokens, prompt "Write a short paragraph about the history of computing.": the kernel arm and the gather arm produced byte-identical completions for all four requests, and the four concurrent requests within each arm were also byte-identical. The two paths agree at batch shapes end to end.

## Environment

| Field | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB unified memory, macOS 26.6.1 (25G76) |
| Apple GPU generation | 15 (`applegpu_g15d`) |
| Build | `cargo build --release --bin mlxcel-server` (workspace, rust LTO) |
| Branch | `perf/moe-switch-gate-up` at `0f43e98e` (PR #2514) |
| Harness | `scripts/bench_serving_concurrency.py` against `mlxcel-server --port 8113` |
| Wrapper | `caffeinate -i`, indexers paused (`scripts/with_indexers_paused.sh`) |
| Target | `models/olmoe-1b-7b-0125-4bit` |
| Raw results | `bench-results/switch-gate-up-batch/20261021-122959/` |

## Follow-up

The batch 2 cell (16 routes) is a small real loss. A gate refinement declining the kernel at `tokens >= 2 && total < 24` would remove it while keeping inkling's measured `tokens == 1` win, but it rests on one cell of one checkpoint family against a noisy control; a focused re-measure of the batch-2 cell should confirm the finding before the shared gate changes.
