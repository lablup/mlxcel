# PR #2931: Count only decode-window tokens in decode tok/s

**Date**: 2026-11-10
**Status**: Completed (merged 2026-11-10; closes #2929)

## Executive Summary

Most mlxcel generation paths sample and evaluate the first token during prefill, before the decode timer starts, yet divided all N generated tokens by `decode_ms`. The timed window holds only N-1 forward steps, so the reported decode rate was N/(N-1) too high: 0.79% at 128 tokens. The speculative engine already used N-1, which meant speculative and plain decode rows were measured under different rules. The PR adds one helper, `mlxcel_core::generate::decode_tok_per_sec(window_tokens, decode_ms)`, routes every path's rate through it, and passes N-1 wherever the first token comes out of prefill. `scripts/bench_mlxlm.py` converts mlx-lm and mlx-vlm `generation_tps` to the same convention so the "vs mlx-lm" columns still compare like with like. `decode_ms` and `generated_tokens` are unchanged; only the rate moves. On Llama-3.2-1B-Instruct-4bit (M1 Ultra, 512-token prompt, 128 tokens), `bench_decode.sh` reports 392.16 tok/s = 127 / 323.84 ms, against 395.26 under the old rule, and `bench_mlxlm.py` reports 389.61 for mlx-lm against mlx-lm's own 392.69. 15 files, +244/-54.

## 1. Problem Statement

A decode rate is a count divided by a time, and both sides have to describe the same interval. In the direct engine, `complete_prefill` returns the first token already sampled and evaluated, and `decode_start` is taken after it. The window that `decode_ms` measures therefore contains the forward steps for tokens 2 through N. Dividing N by that window credits the window with a token it did not produce.

The error is small (N/(N-1)) but systematic, and it was inconsistent across paths:

- The speculative engine already divided `decode_count`, which excludes the prefill token.
- Engine direct completion (`mlxcel-bench-decode`, `bench_decode.sh`, plain `generate`), MTP single and batched, the MTP and DFlash recomputations in `generate`, and the `mlxcel-bench-engine` server probe divided N.
- The server's `predicted_per_second` already followed the N-1 rule, as llama-server does.

So a speculative-vs-plain comparison carried a 0.79% bias toward plain decode at 128 tokens, and the bias grows as N shrinks (about 3.2% at 32 tokens). Speculative decoding speedups are often measured in single-digit percent, so a bias of this size can change a conclusion.

## 2. Change Summary

### 2.1 One helper for the rule

`src/lib/mlxcel-core/src/generate.rs` gains:

```rust
pub fn decode_tok_per_sec(window_tokens: usize, decode_ms: f64) -> f64
```

It returns `window_tokens / (decode_ms / 1000)` and 0.0 when either the count or the window is zero. Its doc comment states the rule, the N/(N-1) error, and the relationship to llama-server and mlx-lm. The `GenerationStats::decode_tok_per_sec` field doc points at it. Before this PR, each path had its own copy of the same `if decode_ms > 0.0 { n / (decode_ms / 1000.0) } else { 0.0 }` expression, which is how the numerators drifted apart.

### 2.2 Per-path numerators

| Path | Numerator after the PR | Reason |
|------|------------------------|--------|
| `engine/direct.rs` (bench-decode, `bench_decode.sh`, plain `generate`) | `gen_count - 1` | First token comes from `complete_prefill`, before `decode_start` |
| `speculative/mtp/generator.rs::build_stats` | `gen_count - 1` | First bonus comes from `prefill_and_seed` |
| `speculative/mtp/round_loop_batched.rs` | sum over rows of `len - 1` | Each row has its own prefill bonus |
| `commands/generate.rs` (offline MTP) and `generate_dflash.rs` | `tokens.len() - 1` | Same prefill bonus, recomputed at the command layer |
| `server/engine_probe/server_engine.rs` | `tokens.len() - 1` | `decode_ms` starts when the first token arrives |
| `bin/bench_engine/output.rs` | `generated_tokens - 1` | Both measurement paths produce the first token before the window; the human-readable line now prints the window token count |
| `engine/speculative.rs` | unchanged (`decode_count`) | Already correct; only routed through the helper |

Paths left at N because every token is produced inside their window: pipeline-parallel `generate` and the diffusion generators (`diffusion_gemma`, `llada2_moe`). DFlash round loops and the server's `predicted_per_second` were already correct and are untouched.

All subtractions use `saturating_sub(1)`, so a run that produced only the prefill token yields a 0 count and the helper returns 0.0 instead of a negative or undefined rate.

### 2.3 mlx-lm baseline conversion

mlx-lm and mlx-vlm restart their clock after the first token in `stream_generate` but still count all N tokens. Their `generation_tps` therefore has the same N/(N-1) inflation mlxcel just removed. Without a conversion, fixing only mlxcel would have shifted every "vs mlx-lm" ratio by 0.79% against mlxcel.

`scripts/bench_mlxlm.py` gains `decode_window(gen_tokens, generation_tps)`. It reconstructs the window as `N / generation_tps` and returns `(window_ms, (N - 1) / window)`. It returns `(None, None)` for missing, non-numeric, or non-positive inputs, which replaces a bare `except Exception: pass` in the CSV writer. A single-token run yields a window but a rate of 0.0.

### 2.4 Documentation

`docs/benchmarks.md` gains a "How decode tok/s is counted" section with the per-path table, the mlx-lm discrepancy, and the restatement formula for old rows: `(generated_tokens - 1) / decode_ms`. Recorded CSVs under `benchmarks/` and `docs/benchmark_results/` are left as recorded. The CHANGELOG entry states the 0.79% shift.

## 3. Measurements

Llama-3.2-1B-Instruct-4bit, M1 Ultra, 512-token prompt, 128 generated tokens.

| Quantity | Value |
|----------|-------|
| `decode_ms` (unchanged by the PR) | 323.84 ms |
| mlxcel decode tok/s, new rule (127 / 323.84 ms) | 392.16 |
| mlxcel decode tok/s, old rule (128 / 323.84 ms) | 395.26 |
| mlx-lm `generation_tps`, raw | 392.69 |
| mlx-lm converted by `bench_mlxlm.py` | 389.61 |

The arithmetic checks: 127 / 0.32384 s = 392.17 and 128 / 0.32384 s = 395.26; 392.69 × 127/128 = 389.62. The last digit of each comes from rounding `decode_ms` and the reported rates.

The mlxcel-to-mlx-lm ratio is the same under both conventions (395.26 / 392.69 and 392.16 / 389.61 are both about 1.0065). That is the property the conversion exists to preserve: absolute numbers drop by N/(N-1), and the cross-runtime comparison does not move.

## 4. Technical Decisions

### 4.1 N-1, not N with a wider window

The other consistent fix would be to start the decode timer before the first token and keep N. That would fold the first token's sampling and evaluation (and, on some paths, the tail of prefill) into `decode_ms`, change the meaning of a field that existing CSVs, dashboards, and the server already report, and break the match with llama-server's `predicted_per_second`. Changing only the numerator keeps `decode_ms` and `generated_tokens` byte-compatible with history, so any old row can be restated from fields it already has.

### 4.2 Convert the mlx-lm baseline instead of copying its convention

Adopting mlx-lm's N-over-(N-1)-steps rule would have made mlxcel match one reference and diverge from llama-server and from its own speculative engine. The PR picks the rule that describes the measured interval, and adapts the external number at the one place it enters mlxcel's CSVs.

### 4.3 A shared helper rather than per-site fixes

The bug existed because seven sites each had their own copy of the division. Putting the rule in one function with a doc comment makes "which count goes in" an explicit argument at every call site, which is where the decision belongs: the helper cannot know whether a given path produced its first token inside the window, but every caller now has to say so.

### 4.4 Leave recorded CSVs alone

Rewriting historical rows would have mixed regenerated and measured values in files that are treated as records. The docs give the restatement formula instead, and the cut-off date (rows before 2026-11-10 used N everywhere except the speculative engine).

## 5. Verification and Limits

- Real checkpoint run as in Section 3.
- New unit tests: `rate_counts_only_window_tokens` and `rate_is_zero_without_a_window` for the helper; `session_stats_rate_excludes_the_prefill_bonus` (MTP single); `batched_stats_rate_excludes_each_rows_prefill_bonus` (MTP batched, including a bonus-only batch that must report 0.0); the server probe test renamed to `decode_rate_excludes_the_first_token_from_the_window` with a single-token case; bench-engine fixtures moved from 100 to 101 tokens so the expected rates stay the same; and `tests/test_bench_mlxlm_decode_window.py` for the Python conversion, including CSV string inputs and missing values.
- `cargo clippy --workspace --all-targets -D warnings` and `cargo fmt --check` passed. `cargo test --workspace --profile test-fast --no-fail-fast` passed 15373 tests with 8 failures outside the touched code: four root-lib model tests (dots1, qwen4_exp snapshot) that fail only in the parallel run and also fail with the change stashed, and four `mlxcel-server` argument tests that pass alone and with `--test-threads=1`.

Open items a maintainer should know about:

- The server `/metrics` per-request decode histogram is out of scope. Its `generation_only_ms` source differs by worker (florence2 and nemotron_parse use full elapsed time), so it needs a separate pass and may still disagree with the CLI rate.
- `decode_window` assumes mlx-lm and mlx-vlm keep the "clock restarts after the first token, count all N" semantics. If upstream changes `stream_generate`, the conversion silently becomes wrong. No test pins the upstream behavior.
- Historical CSV rows and new rows now sit side by side under different conventions. Anyone comparing across 2026-11-10 has to restate the older rows.
- Only the plain direct path was validated on a real checkpoint. The MTP, DFlash, and server-probe changes are covered by unit tests on the stats builders, not by a real-model run.

## 6. Change Summary Table

| Item | Value |
|------|-------|
| Files changed | 15 |
| Lines added | +244 |
| Lines deleted | -54 |
| Tests added or changed | 2 helper tests, 2 MTP stats tests, 1 server probe test, 2 bench-engine fixtures, 4 Python tests |

| Category | Summary |
|----------|---------|
| Correctness | Decode rate counts N-1 on prefill-seeded paths; one helper for every path |
| Benchmark tooling | mlx-lm and mlx-vlm baseline converted to the same convention |
| Documentation | `docs/benchmarks.md` counting rule and restatement formula; CHANGELOG entry |

## 7. Learning Points

- When a timer starts after an event, the count must start after it too. The first token is the usual place this goes wrong in LLM benchmarks, because prefill naturally ends by producing it.
- A small, constant bias is worth fixing when it differs between the things being compared. Here the plain and speculative paths disagreed, and that disagreement mattered more than the absolute 0.79%.
- When fixing a convention, check every external number that is compared against it. Fixing only one side of a ratio moves the ratio.
