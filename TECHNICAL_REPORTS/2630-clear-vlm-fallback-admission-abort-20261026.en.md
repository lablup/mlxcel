# PR #2630: Clear VLM fallback state when admission aborts

**Date**: 2026-10-26
**Status**: Complete on Metal (M1 Ultra) for Qwen3-VL; other families validated by unit tests only
**Languages**: Rust
**Risk Level**: Medium

## Executive Summary

PR #2630 (issue #2625) closes a cross-request state leak in server admission. Media preparation writes a request's media state into model-level fallback slots before the request has a sequence id, and admission moves that state under the id with the `bind_*_to_sequence` calls. Three admission returns sit between the two steps, and none of them emptied the slots, so the next request's bind adopted the failed request's state. A text-only request never overwrites these slots, so it decoded against another request's MRoPE deltas, DeepStack features, `per_layer_inputs`, cross-attention image, or Falcon-OCR rope delta. On a real Qwen3-VL server the text-only request after an aborted image request produced one wrong token instead of its solo answer; with the fix it matches solo.

## 1. Problem Statement

`bind_fallback_to_sequence` takes whatever the fallback holds, and the text-only branch of `prepare_request_vlm_embeddings` returns without touching the fallback. The binding is therefore only correct if every admission that prepares media also reaches its bind. Three returns in `src/server/batch/scheduler/admission.rs` broke that:

- `allocate_sequence_state()` failing with `Server busy` after the early preparation that `--enable-vlm-prefix-cache` runs before a sequence id exists.
- The early preparation failing part way, for example on cancellation after a family wrote its slot. The old comment said there was nothing to clean up here.
- The non-early preparation failing after allocation. `release_sequence_caches` drops per-sequence entries, never the fallback.

## 2. Change Summary

| Area | Change |
|---|---|
| `LoadedModel` | `clear_vlm_fallback_state()` drops every fallback slot that admission binds per sequence: the Qwen VL runtimes (MRoPE, plus DeepStack where present), Qwen3-Omni, Gemma 4 (both runtimes), Gemma 3n, Llama 3.2 Vision and Falcon-OCR. Per-sequence entries are untouched. |
| Admission | Calls the clear on all three abort returns. |
| `QwenVlRuntime` | `clear_pending_media_state()` is a required method with no default, so a new Qwen-style runtime cannot skip it silently. |
| State containers | New clear methods for the Gemma 4 and Gemma 3n `per_layer_inputs` containers and Falcon-OCR state; `clear_mrope_state()` for the GLM-4V-MoE, Qwen3.5 and Qwen4-Exp text models. |

## 3. Technical Decisions

### 3.1 Clear only families whose slot is drained on every admission

The clear is safe only where the fallback is empty while requests run, which holds when admission binds (and so drains) the slot on every request. The first version also cleared PaddleOCR-VL, ERNIE-4.5-MoE-VL and Hunyuan-VL. pr-reviewer found that admission never binds those three per sequence: `bind_qwen_vl_mrope_state_to_sequence` only matches the `Qwen` and `Qwen3OmniMoe` runtimes. Their in-flight requests decode from the fallback itself, so clearing it when another request aborted would have wiped a live image request's positions. They were removed from the clear.

### 3.2 Clear at the abort, not at the next bind

The alternative was to make every bind validate that the fallback belongs to the binding request. That needs a request identity on the fallback, which preparation does not have, and it would leave the aborted request's tensors resident until the next admission. Clearing at the abort keeps the invariant local to the code that breaks it and frees the tensors immediately.

## 4. Validation

- **Failing-first unit tests:** `src/server/batch/scheduler/vlm_fallback_abort_tests.rs` drives the real `enqueue_request` on a two-sequence pool. There is one test per abort path on Qwen3-VL, plus an allocation failure on Llama 3.2 Vision. With the admission clears reverted, all four fail; with them, all pass. Two state-level tests cover the Gemma containers. The tests seed the fallback directly because the tiny test models have no vision tower.
- **Real checkpoint**, `models/qwen3-vl-4b-instruct-4bit`, on `mlxcel-server --parallel 2 --max-queue-depth 1 --enable-vlm-prefix-cache`:
  - Setup: two long text requests fill the active rows and image request Y is queued. Image request X, sent at the same instant, is refused with `Server busy` after its early preparation.
  - On `main`, the following text-only request T returned `"` instead of its solo answer. On this PR, T matches solo.
  - Y has to be an image request, because a queued text-only prefill clears the Qwen3-VL fallback by accident and hides the leak.
- **Regression:** the concurrent-vs-solo check from PR #2610 still matches solo. Workspace clippy, fmt and the 221 scheduler tests pass after merging the latest `main`.

## 5. Learning Points

- A fallback-plus-bind design is only correct if every path that fills the fallback reaches the bind. Abort paths are where this breaks, and a comment saying "nothing to clean up" was wrong because the cleanup lives in model state, not scheduler state.
- A real repro depended on queue timing. Two coincidences hid the bug: a text-only queued prefill clearing the slot, and the HTTP queue gate rejecting X before preparation. The harness had to arrange both away.

## 6. Follow-up and Limitations

- PaddleOCR-VL, ERNIE-4.5-MoE-VL and Hunyuan-VL are never bound per sequence, so they share one MRoPE fallback across requests even without an abort. This is pre-existing, and the PR body names it as out of scope.
- No real-checkpoint runs for Llama 3.2 Vision, Gemma 4 / 3n or Falcon-OCR.

## References

- Issue #2625, related #2567 and PR #2610
- `src/loaded_model_capabilities.rs`, `src/server/batch/scheduler/admission.rs`, `src/multimodal/qwen_vl.rs`
