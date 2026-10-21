# PR #2521: expose GLiNER2.5 CLI options, fix token-classification leftovers

**Date**: 2026-10-21
**Status**: Completed
**Risk Level**: Low
**Languages**: Rust

---

## Executive Summary

Four token-classification leftovers from issues #2383/#2387/#2390/#2391 are implemented in one PR. The headline item is a shipped capability that was unreachable from the CLI: `Gliner2_5Model` supported `max_len` and a `char` word splitter for scripts without spaces, but `mlxcel extract` exposed neither; both are now CLI flags wired through to the loaded model. The other three items: the Viterbi decoder accumulates in f64 to match upstream's f64 bias adds, the Privacy Filter band mask is built once per chunk per forward instead of per chunk per layer, and malformed checkpoint tensor shapes now error at load with the tensor named instead of aborting inside MLX. Two pre-existing defects found during verification are fixed on the way.

---

## 1. Problem Statement

### 1.1 Background

The token-classification subsystem shipped GLiNER2.5 support (model-side `max_len` validation, `set_word_splitter`), but the CLI surface was not wired. Item 1 means a shipped capability was unreachable from the real code flow.

### 1.2 Existing Issues

- **Item 1**: `ExtractOptions` had only `include_confidence`/`include_spans`; no `--max-len` or `--word-splitter` flags existed.
- **Item 2**: the Viterbi decoder summed in f32 while upstream adds f64 biases.
- **Item 3**: the Privacy Filter band mask was rebuilt per chunk per layer.
- **Item 4**: malformed tensor shapes aborted inside MLX instead of erroring at load.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Shipped capability unreachable from CLI | Medium (CJK extraction unusable without a config edit) | Deterministic |
| Malformed checkpoint aborts the process | Low (user-facing error quality) | On malformed checkpoints |

---

## 2. Technical Review

### 2.1 Security

CLEAN (pr-security-checker, 0 CRITICAL/HIGH). CLI input validation is fail-fast: non-integer `--max-len` refused by clap before the range check; the range check (1..=32768) reuses the existing error text; `--word-splitter` accepts exactly `whitespace|char`. `validate_shapes` runs first in `from_weights`, before any graph build, returning descriptive errors (no panic, no MLX abort). The band windows flow as an immutable `&[BandWindow]` with no interior mutability; the only mutable static is the `BAND_MASKS_BUILT` test counter (AtomicUsize). No new unsafe code.

### 2.2 Performance

The band mask hoisting is strictly less work: one build per chunk per forward instead of per chunk per layer (counter test: 6 vs 12 for 40 tokens in 7-query chunks, 2 layers). The f64 lattice doubles the accumulator word size on a label-count-sized array, negligible. `validate_shapes` is one pass of hash lookups plus FFI shape reads at load time only, off the hot path.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. Default behavior is unchanged when the new flags are absent (the old `load_span_extractor` signature is kept and delegates with `SpanExtractorLoadOptions::default()`); the public `decode(&[Vec<f32>]) -> Vec<usize>` signature is kept; JSON output shapes of `tag` and `extract` are unchanged.
- **New Dependencies**: none.

### 2.4 Code Quality

- The item-2 fixture genuinely requires f64: at V = 2^24 the f32 ULP is 2, so the f32 path rounds the t2 E_X increment (2^24 + 1.0, round-half-even) back to 2^24, ties the O-first background path, and fails to close the entity; f64 keeps 16777217 and closes it.
- The item-4 validation list covers every weight key the graph consumes; the `.scales` sibling skip matches UnifiedLinear's own quantization keying (`format!("{}.scales", prefix)`), validated against the real 8bit checkpoint (score.scales [33,10], packed score.weight [33,160] U32).
- token_classification filter 87/0, extract 9/0, openai_privacy_filter 12/0 (CUDA, TF32=0, gpu-lock, three filtered-run passes); fmt and clippy -D warnings clean.

---

## 3. Technical Decisions

### 3.1 Fail-fast CLI validation before any model load

Both flags are validated in `gliner_load_options()` before `resolve_model_source_with_override` and `initialize_runtime_checked`, so out-of-range `--max-len` produces the existing config error text before any model loads.

### 3.2 The `.scales` skip key

The standalone `score.weight` rank/width check skips the width check when a quantized score plane is present, and the `.scales` sibling keying strips the `.weight` suffix (MLX affine stores the scales sibling at the projection prefix: `score.scales` for key `score.weight`). The pre-existing check `{key}.scales` false-positived on the 8bit checkpoint and failed two tests on the first run; fixed and verified against the genuine layout.

### 3.3 f64 accumulation, signature kept

The lattice arithmetic converts to f64 with `f64::from` at every emission and bias add; `NEGATIVE_INFINITY` and `best_of` are f64; the public signature and tie-break order are unchanged.

---

## 4. Implementation Details

**Item 1** — `src/commands/extract.rs` (flags + parse), `src/token_classification/loader.rs` (`SpanExtractorLoadOptions`, wiring), `src/token_classification/gliner2_5.rs` (`check_max_len` shared with the config path, `set_max_len`/`set_word_splitter` reachable), `src/commands/extract_tests.rs`.

**Item 2** — `src/token_classification/viterbi.rs`, `viterbi_tests.rs`.

**Items 3, 4** — `src/models/openai_privacy_filter_layers.rs` (`band_windows` once per forward, `BAND_MASKS_BUILT`), `openai_privacy_filter.rs` (`validate_shapes` first in `from_weights`), `openai_privacy_filter_tests.rs`.

**Docs** — `docs/token-classification.md`: both flags documented in the `mlxcel extract` options paragraph; the stale "has no CLI flag yet" statement in the GLiNER2.5 Input paragraph replaced.

---

## 5. Verification

| Check | Result |
|---|---|
| token_classification filter (CUDA, TF32=0, gpu-lock, --test-threads=1) | 87 passed / 0 failed |
| extract filter (bin harness) | 9 passed / 0 failed |
| openai_privacy_filter filter | 12 passed / 0 failed |
| Fail-before (item 1) | tests written first; red run failed to compile with "cannot find function `gliner_load_options`" |
| Real-checkpoint regression | privacy_filter_8bit and acceptance_sentence pass; extract CJK gate matches the mlx-vlm fixture |
| Flags in --help | tag_and_extract_are_discoverable_in_help |
| fmt / clippy -D warnings | clean / clean |
| pr-reviewer | all items PASS; one MEDIUM docs gap (fixed by 17f35020) |
| pr-security-checker | CLEAN; three LOW hardening notes recorded |

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 11 (10 code/test + 1 docs) |
| Tests added | ~10 (per item: CLI x4, f64 x1, band-mask counter x1, malformed-shapes x1, help x1, plus fixture updates) |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `62b980e7` | feat | feat(token-classification): expose GLiNER2.5 CLI options, fix leftovers |
| `17f35020` | docs | docs(token-classification): document the GLiNER2.5 extract flags |

---

## 7. Follow-up Actions

### Future Improvements (recorded LOW notes)

- `validate_shapes`'s quantized skip short-circuits before the presence check: a checkpoint carrying `.scales` without `.weight` passes validation and is caught downstream by UnifiedLinear's "Weight not found" error (no panic, only a less specific message). Optional hardening: check presence before the skip.
- `--max-len`/`--word-splitter` are silently ignored for non-GLiNER span extractors (documented "GLiNER2.5 only" in --help); a runtime warning would prevent a user believing the flag applied.
- The config `max_len` path duplicates the range check inline instead of reusing `check_max_len` (error text identical, no behavior risk).

### Monitoring Required

- First real CJK extraction runs through the new flags in the next release cycle.
