# Technical Report: PR #3042 - fix(cli): report XLA prefill and decode separately

**Date**: 2026-11-14
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

The OpenXLA backend's `generate --profile` output reported prefill as zero and divided every generated token by the whole run, so prefill sat inside the decode rate. A kernel trace showed 413.5 ms of prefill kernel time (about 590 ms wall) for a 42-token prompt, which diluted short runs so heavily that one unchanged environment measured anywhere from 51.5 to 132.6 tok/s depending only on `n`. The session now returns the two windows it already executes as `GreedyTiming`, and the CLI reports them separately; decode reads 154.8 to 165.9 tok/s at `n=64` to `512`, flat within 7 percent.

---

## 1. Problem Statement

### 1.1 Background

The OpenXLA single-sequence session already splits its work: `generate_streaming_greedy` seeds KV with the prompt prefix, then starts decoding from the last prompt token. The CLI, however, discarded that structure and measured the whole call.

### 1.2 Existing Issues

- `generate_xla` built its `GenerationStats` with `generation_stats_from_duration(...)`, a helper that hardcodes `prefill_time_ms: 0.0` and computes `decode_tok_per_sec` as `generated_tokens / total_time`, where `total_time` starts before prefill.
- The profile line therefore always printed `Prefill: 0.00 ms (0.00 tok/s)`, and the decode figure was really tokens over prefill plus decode.
- On gfx1151 with llama-3.2-1b-instruct-4bit, a 42-token prompt costs about 590 ms of prefill. The same binary measured 51.5 tok/s at `n=64`, 75.7 at `n=128`, 114.0 at `n=256` and 132.6 at `n=512`, all of which were artifacts of the fixed cost, none a decode rate.
- The MLX path does this correctly through `DirectEngine`, which times prefill up to the first token and counts `N - 1` tokens in the decode window.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Existing benchmarks compared across arms remain valid but their absolute figures are misread | Wrong optimization decisions | High, already observed |
| The timed entry point changes the token stream | Wrong output | Low: the untimed path delegates to it and a device gate compares streams |
| Other XLA paths (images, audio, Gemma3n prepared) are wrongly assumed fixed | Partial fix mistaken for complete | Medium: documented in code and changelog |

---

## 2. Technical Review

### 2.1 Security

None. No input, authentication or data-handling surface changes; the change adds two `f64` wall-time measurements to an internal result type.

### 2.2 Performance

The change is instrumentation only and does not alter execution. What it changes is the measurement:

| n | before (tok/s) | after decode (tok/s) | after prefill (tok/s) |
|---|---|---|---|
| 64 | 51.5 | 159.6 | 68.7 |
| 128 | 75.7 | 165.9 | 65.8 |
| 256 | 114.0 | 154.8 | 71.3 |
| 512 | 132.6 | 162.2 | 69.5 |

The "before" column is tokens over prefill plus decode; the "after" columns are the two windows reported separately. Decode is flat in `n`, which is the acceptance criterion.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. `generate_streaming_greedy` keeps its signature and behavior; the timed variant is additive.
- **New Dependencies**: none.
- **Compatibility**: all XLA devices and both text and multimodal entry points; the image, audio and Gemma3n prepared paths keep the whole-run accounting.

### 2.4 Code Quality

Two new ignored device gates in `tests/xla_greedy_timing.rs`, one for the phase split and token-stream identity, one for the zero budget. `cargo fmt` and scoped `cargo clippy` report nothing new.

---

## 3. Technical Decisions

### 3.1 Time the existing calls instead of restructuring generation

**Context:** the split already exists inside the session; only the durations were lost.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Drive `prefill` and `decode_step` from the CLI directly | No new session API | Duplicates the split, EOS handling and capacity validation, which is exactly the bug-prone part |
| **Chosen: `generate_streaming_greedy_timed`** | One execution path, no duplicated contract | Adds a result type to the crate's public surface |

**Rationale:** the terminating-token contract (issue #963) lives in `drive_greedy_loop`; reimplementing the loop in the CLI would put a second copy of it next to the one the server also uses. Timing around the existing calls keeps one implementation.

**Trade-offs:** a new public `GreedyTiming` type and one more method to keep documented.

### 3.2 Count all generated tokens in the decode window

**Context:** the two backends split the phases differently.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Reuse the MLX `N - 1` convention | One rule everywhere | The XLA first token is produced by `decode_step`, so excluding it would understate the window |
| **Chosen: count all N on XLA** | Matches what the window contains | The two backends need different comments |

**Rationale:** `decode_tok_per_sec` is defined over the tokens produced inside the measured window. The MLX first token comes out of prefill and is outside its window; the XLA first token comes out of `decode_step` and is inside it. The rule is the same, the arithmetic differs. Both the method documentation and the CLI comment state this.

### 3.3 Keep the other XLA paths on the old accounting

Image, audio and Gemma3n prepared inputs do not route through `generate_streaming_greedy`, so their phase split would need separate work. Leaving them on `generation_stats_from_duration` is recorded in the changelog rather than silently assumed fixed.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
generate_xla
  start_time
  session.generate_greedy(...)        # prefill + decode, no split
  stats = generation_stats_from_duration(total)

[After]
generate_xla
  session.generate_streaming_greedy_timed(...)
    prefill_ms  = time(prefill(prefix))
    decode_ms   = time(decode_step(last) + drive_greedy)
  stats = GenerationStats { prefill_time_ms, decode_time_ms, ... }
```

### 4.2 Key Code Changes

**File: `src/lib/mlxcel-xla/src/lib.rs`**

```rust
pub struct GreedyTiming {
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

let prefill_started = Instant::now();
self.prefill(&prompt_tokens[..split])?;
let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
let decode_started = Instant::now();
let first = self.decode_step(prompt_tokens[split])?;
let tokens = self.drive_greedy(first, max_new_tokens, eos_token_ids, on_token)?;
let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
```

**Reason for change:** the session already performs these two phases; only their durations needed to escape.

**File: `src/commands/generate.rs`**

```rust
let (tokens, timing) = s
    .generate_streaming_greedy_timed(prompt_tokens, max_tokens, &eos, |_| true)?;
xla_timing = Some(timing);
```

and, after the match, a `GenerationStats` built from `timing.prefill_ms` and `timing.decode_ms` instead of the wall clock.

### 4.3 Data Model Changes

None.

---

## 5. Learning Points

### 5.1 A fixed cost inside a rate

A rate over a window that contains a fixed cost is not comparable across window sizes. Here a 590 ms prefill was divided into every decode figure, so the same hardware looked 2.6 times faster at `n=512` than at `n=64`. The first symptom was not a wrong-looking number but a number that moved with an unrelated parameter.

### 5.2 Cross-check a benchmark against the trace

The profiler was the arbiter. `rocprofv3 --kernel-trace` showed the prefill dispatch block (693 kernels, 413.5 ms) sitting in the same wall-clock window the CLI attributed to decode. Without that trace, the discrepancy could have been attributed to clocks, thermal state or the code under test, all of which were plausible.

### 5.3 Delegation keeps one contract

Wrapping the existing calls preserved the EOS and budget semantics for free. Reimplementing the loop in the CLI would have created a second copy of a contract that already has a test.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| prefill | Prompt pass that seeds KV and produces the first token on the MLX path | The fixed cost that was hidden in the decode rate |
| decode window | Time spent producing tokens after the first | The quantity a decode rate should measure |
| `GreedyTiming` | The two wall-time windows the XLA session returns | The new seam between session and CLI |

### Related PRs/Issues

- Issue #3041: the filed bug with the measured evidence.
- PR #3031: the decode-step scalar reuse whose quoted absolute figure includes this prefill, so its 150.1 tok/s combined median is a prefill-inclusive rate; the A/B comparison itself remains valid.
- PR #2976: split load timing out of `xla_batch_bench`; the same class of accounting error on a different surface.
- Issue #2929: established that decode tok/s counts only tokens inside the timed window.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 5 |
| Lines added | +288 |
| Lines deleted | -4 |
| Tests added | 2 ignored device gates |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Correctness | 1 | Phase-accurate statistics for the XLA CLI path |
| Tests | 1 | `xla_greedy_timing` gate |
| Documentation | 1 | CHANGELOG entry |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `0dcc68f3` | fix | fix(cli): report XLA prefill and decode separately (#3041) |

---

## 8. Follow-up Actions

### Required

- [x] Device gates and the `n` sweep.
- [ ] None outstanding.

### Monitoring Required

- Decode rates quoted for the OpenXLA backend after this change are not comparable with figures quoted before it; benchmark tables and issues that cite the old numbers should say which accounting they used.

### Future Improvements

- Extend the same timing to the image, audio and Gemma3n prepared paths.
- Re-baseline the #3031 comparison with phase-accurate numbers, which is also the prerequisite for attributing the remaining non-kernel time in the decode step.

---

## Appendix

### A. Test Results

```
cargo test --features "xla-iree test-utils" --test xla_greedy_timing -- --ignored --test-threads=1   # 2 passed
cargo check --features xla-iree                                                                      # pass
cargo fmt --all -- --check                                                                           # pass
```

### B. Performance Benchmarks

Rebuilt binary, native quant, context capacity 1024, one warm-up run before the sweep:

| n | prefill | prefill tok/s | decode | decode tok/s |
|---|---|---|---|---|
| 64 | 597.04 ms | 68.67 | 400.96 ms | 159.62 |
| 128 | 623.11 ms | 65.80 | 771.54 ms | 165.90 |
| 256 | 574.99 ms | 71.31 | 1654.14 ms | 154.76 |
| 512 | 590.05 ms | 69.49 | 3157.45 ms | 162.16 |

### C. References

- `src/lib/mlxcel-core/src/generate.rs`, `decode_tok_per_sec`, for the window-token rule.
- `src/lib/mlxcel-core/src/engine/direct.rs`, the MLX path's prefill/decode split.
