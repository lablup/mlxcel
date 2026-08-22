# Technical Report: PR #1281 - An order-independent checkpoint layout verdict for RT-DETRv2

## Executive Summary

`needs_sanitize` decides whether a freshly loaded RT-DETRv2 weight map is raw HuggingFace layout and therefore needs the rename plus transpose pipeline. It walked `weights.keys()`, which is a `HashMap`, and returned from inside the loop as soon as either an MLX marker or an HF marker was seen. For a map carrying both marker families the verdict was decided by whichever key the randomized walk reached first, so the same file could be judged differently on different runs. The fix scans every key before deciding and applies an explicit precedence.

This is the fourth instance of one root-cause class found in a single review sweep, after #1265 (test fixtures), #1267 (`lang_bias.rs`) and #1277 (distributed registry accessors).

## 1. Problem Statement

The loop set `has_mlx_marker` for keys prefixed `vision.backbone.` or `vision.hybrid_encoder.`, set `has_hf_marker` for six HF markers, and then returned `false` or `true` immediately. Its comment claimed it waited "once both decisions are unambiguous", which the code did not do: it returned on the first flag of either kind.

The two directions are not symmetric in cost. A spurious `false` on a genuine HF checkpoint skips the rename, and the load then fails a moment later when the model looks up `vision.backbone.*` names the raw map does not have. A spurious `true` on an already-MLX checkpoint runs the pipeline over already-transposed weights and silently double-transposes the convolutions, which the module documentation at the top of the same file explicitly warns about. A double-transposed conv is a shape-valid tensor, so nothing downstream can flag it and the model emits confident, wrong detections.

Reachability, stated without overclaiming. Two HF markers are unanchored substring tests, `.contains(".convolution.")` and `.contains(".normalization.")`, and a correctly converted MLX checkpoint should not trip them, because `rename_stripped` in the same file rewrites `.convolution.` to `.conv.` and `.normalization.` to `.bn.`. So for well formed inputs on both sides this was latent rather than live. It becomes reachable for a partially converted or hand merged checkpoint, an MLX checkpoint shipping auxiliary tensors under HF field names, or a future MLX layout that keeps any `.normalization.` shaped key. The point is that the function had no defense against that case and failed toward the corrupting direction at random.

## 2. Technical Decisions

### 2.1 Scan every key, then decide, with MLX winning

Deciding after the full scan removes the dependence on iteration order. MLX takes precedence because it is the recoverable direction, per the asymmetry above.

### 2.2 A warning rather than a hard error

The issue raised whether a map carrying both families should be rejected loudly, since it is a layout the loader does not actually understand. It is not rejected, for three reasons recorded in the PR. MLX-wins is already the safe direction and the wrong guess in that direction still fails loudly a moment later at name lookup, so "silent" describes the verdict rather than the outcome. The detector is knowingly over-broad, since the unanchored `contains` tests are out of scope to change here, and making the mixed case fatal would build a hard load failure on top of a known-loose predicate. And `needs_sanitize` is `pub` returning `bool`, so a `Result` would thread a signature change through `RtDetrV2Model::load` for a case the issue itself calls latent.

Most of the value of "loud" is recovered with a `tracing::warn!` on the mixed case naming one offending key per family. That diagnostic picks the lexicographically first match per family rather than the first the walk encounters, so the warning does not reinherit the very nondeterminism the fix removes.

### 2.3 The misleading comment and the dead return

The comment now describes what the code does. The trailing `has_hf_marker` return after the loop, reachable only when no key set either flag and therefore only ever `false`, no longer reads as though it could return `true`.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/vision/detection/rt_detr_v2/sanitize.rs` | Full scan before the verdict with explicit MLX precedence; `tracing::warn!` on the mixed case with order-independent key selection; comment corrected; dead return made explicit; 2 new tests |

`RtDetrV2Model::load` is untouched: the signature stayed `bool`, so the call site is byte-identical.

## 4. Review Findings

The load-bearing requirement was proving the regression test fails against the unfixed function, because this bug class readily produces tests that pass before and after. Run against the original code, the new test reported that `needs_sanitize` returned `true` for **27 of 64** mixed-marker maps freshly built from the same key set, and printed the full 64-element verdict vector. The existing `needs_sanitize_detects_layout` passed in that same run, confirming that its green status was never evidence about this defect: it uses one single-key MLX map and one single-key HF map, so it structurally cannot observe an ordering dependence.

The test builds a fresh `HashMap` inside each of its 64 iterations rather than reusing one, because `RandomState` randomizes per map instance and not merely per process. That property was measured separately while filing #1267: ten maps built from the same five keys produced nine distinct iteration orders inside one process.

## 5. Validation

Measured on GB10 (DGX Spark, CUDA sm_121, Linux aarch64).

The branch was cut from `ca114220` and `main` had since taken #1278 and #1280, so the commit was rebased onto `bf69b83e` before gating and the gate ran on the tree that merges rather than on the stale base. After the rebase the diff against `main` is one file.

- `make verify-test-cuda` at the rebased tree: **8229 passed, 0 failed, 311 ignored**, 101 suites, exit 0, no link or compile errors.
- `cargo test --profile test-fast --features cuda --lib vision::detection::rt_detr_v2::sanitize`: 14 passed, exit 0, and green across six consecutive runs, which is 384 independently constructed maps with no differing verdict.
- `cargo fmt --all -- --check`, `cargo check --lib --tests --features cuda`: exit 0.

Not verified by observation: no real RT-DETRv2 checkpoint was loaded. The integration criterion holds by construction, since the signature is unchanged and only the mixed case changed behavior, but it was deliberately left unticked on the issue rather than claimed.

## 6. Related Work

- #1276: the issue this closes, filed from the review sweep on PR #1268.
- #1265 and PR #1266, #1267 and PR #1269, #1277: the sibling instances of the same class.
- Left alone deliberately: the `needs_sanitize` doc comment names `decoder.layers.` as an MLX marker, but the markers actually tested are `vision.backbone.` and `vision.hybrid_encoder.`. Correcting it means touching the marker set, which the issue puts out of scope. Recorded as a comment on the issue instead.
