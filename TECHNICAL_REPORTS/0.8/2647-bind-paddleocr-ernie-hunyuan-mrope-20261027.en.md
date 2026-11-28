# PR #2647: Bind PaddleOCR, ERNIE and Hunyuan VL MRoPE per sequence

**Date**: 2026-10-27
**Status**: Complete on Metal (M1 Ultra) for PaddleOCR-VL bf16 and HunyuanOCR 4-bit; ERNIE-4.5-MoE-VL validated by unit tests only
**Languages**: Rust
**Risk Level**: Medium

## Executive Summary

PR #2647 (issue #2634) gives PaddleOCR-VL, ERNIE-4.5-MoE-VL and Hunyuan-VL the same per-sequence MRoPE lifecycle the Qwen VL family already had in `mlxcel-server`. Admission never bound these three families, so every in-flight request decoded from the single `MRopeState` fallback slot, which holds the positions of whichever image request was prepared last. On real checkpoints with `--parallel 4`, a concurrent PaddleOCR-VL image request and a later text-only request both decoded repeated digits instead of their solo answers, and a HunyuanOCR text-only request diverged from its solo output. With the fix, every concurrent output matches its solo run on both models.

## 1. Problem Statement

These three text models already stored MRoPE state in `MRopeState` (fallback slot plus `HashMap<SequenceId, MRopeEntry>`), and release already went through `release_mrope_sequence`. The gap was in `LoadedModel`: `bind_qwen_vl_mrope_state_to_sequence`, `take_qwen_vl_mrope_entry` and `install_qwen_vl_mrope_entry` matched only `VlmRuntimeRef::Qwen` and `Qwen3OmniMoe`. For the other three runtimes the bind was a silent no-op. `MRopeState::with_entry` and `batched_decode_positions` resolve an unknown id through the fallback, so nothing failed loudly; requests simply shared positions.

PR #2630 had deliberately left these families out of `clear_vlm_fallback_state()`. While they decoded from the fallback, clearing it on another request's admission abort would have wiped a live request's positions. That exclusion was correct for the old state and is what this PR resolves.

The issue's premise needed a long enough text prompt to show. A short text request with 40 tokens matched solo on main; the divergence appeared once the text request's positions overlapped the leftover image positions.

## 2. Change Summary

| Area | Change |
|---|---|
| `LoadedModel` | `PaddleOcr`, `Ernie45MoeVl` and `HunyuanVl` arms in bind, take, install and `clear_vlm_fallback_state`. Take and install use only the MRoPE half of `QwenVlMRopeSnapshot`, like the Qwen3-Omni arm. The comment explaining the exclusion from the clear is removed. |
| Text models | `take_mrope_entry` / `install_mrope_entry` on the PaddleOCR-VL, ERNIE-4.5-MoE-VL and Hunyuan-VL text models, mirroring `qwen3_vl.rs`. |
| Scheduler | Comments only in `admission.rs` and `decode_tick.rs`; the call sites already ran for every model. |
| `qwen_mrope_state.rs` | `// Used by:` lists name every family that uses each helper. |
| Tests | New `server::batch::scheduler::mrope_family_binding_tests` (21 tests) drives real `enqueue_request` admission, preemption, a `Server busy` abort and release for each family. `vlm_fallback_abort_tests` now expects these families to be cleared. |

## 3. Technical Decisions

- **Reuse the Qwen-named entry points instead of adding new ones.** The scheduler already calls `bind_qwen_vl_mrope_state_to_sequence` and the take/install pair for every model, so adding match arms extends coverage with no scheduler change. The names are now broader than their scope; renaming them is cosmetic and was left out to keep the diff small.
- **Leave `with_entry` falling back for a missing id.** The CLI runs with `seq_id == None` and relies on the fallback. The reviewer and security checker confirmed every server prefill, decode, speculative and `route_batched_decode_with_ids` path passes an id, and the three wrappers use `BatchedDecodeKind::Native`, so the fallback is no longer read by the server for these families. Three CLI guard tests pin that the fallback still works.
- **Text-only requests get an empty bound entry.** Binding drains the fallback even when it is empty, so a text-only request owns an empty entry rather than reading whatever the fallback holds later.

## 4. Validation

- Red commit 4116a975: 15 of 18 new tests failed on main; the 3 passing ones are the CLI guards that must pass on both sides. The reviewer confirmed the tests fail with the fix reverted.
- M1 Ultra, `mlxcel-server --parallel 4`, greedy, 120 tokens, two image OCR requests then one long text request:

| Checkpoint | main 5610f8f5 (A / B / T match solo) | branch |
|---|---|---|
| `mlx-community/PaddleOCR-VL-bfloat16` | no / yes / no | yes / yes / yes |
| `hadeseus/HunyuanOCR-mlx-4bit` | yes / yes / no | yes / yes / yes |

- Solo outputs and CLI `mlxcel generate` output (image and text prompts) are identical between main and the branch.
- Orchestrator gate on the branch merged with main 11463e50: fmt, cross-repo refs, workspace clippy with `-D warnings`, and the scheduler, model and vision test modules.

## 5. Not Verified

- ERNIE-4.5-MoE-VL on a real checkpoint: the only MLX checkpoint is the 28B model. Its arms are identical in shape to the other two and covered by the unit tests.
- `mlx-community/PaddleOCR-VL-4bit` crashes in the vision tower on main (`[broadcast_shapes] Shapes (N,1152) and (N,144)`) in both CLI and server. This is unrelated to MRoPE binding and needs its own issue.

## 6. Learning Points

- A silent `_ => {}` arm in a capability dispatcher turns "family not wired" into "family shares global state". When a state container supports per-sequence entries, the dispatcher that binds it should cover every runtime that owns one, and a test should iterate the runtimes rather than pick one.
- Cross-request leaks can need a specific overlap to be visible. A negative result from a short prompt is not evidence that state is isolated.
