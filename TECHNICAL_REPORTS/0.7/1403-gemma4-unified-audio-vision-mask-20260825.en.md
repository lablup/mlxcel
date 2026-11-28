# Technical Report: PR #1403 - Preserve Gemma4 Vision Masks with Audio

**Date**: 2026-08-25
**Author**: mlxcel contributors
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: High

---

## Executive Summary

PR #1403 keeps Gemma 4 Unified's blockwise bidirectional attention active for contiguous image and video token spans when audio tokens share the prompt. It removes the incorrect audio-presence gates from both mask-construction paths, leaves audio outside all vision blocks, and adds regression coverage proving that only vision positions gain forward attention.

---

## 1. Problem Statement

### 1.1 Background

Gemma 4 Unified checkpoints can declare `use_bidirectional_attention: "vision"`. During prefill, positions inside one contiguous image or video token span should attend to the entire span, while text and audio rows retain their ordinary causal or windowed mask.

### 1.2 Existing Issues

- **Host-side policy error**: `compute_vision_block_ids` returned `None` whenever any audio token appeared, even when valid vision spans were present.
- **Graph-side policy error**: The embeddings-driven prefill path repeated the same audio-count gate after deriving vision positions with MLX operations.
- **Reachable degradation**: Gemma 4 Unified already accepts combined image and audio requests, so production prompts could silently run their image span fully causal.
- **Incorrect regression pin**: A unit test explicitly required the degraded `None` result for mixed prompts.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood before fix |
|------|--------|-----------------------|
| Image understanding degrades only when audio is also supplied | High | High |
| Host and embeddings-driven prefill mask behavior diverges from checkpoint intent | High | High |
| Future refactors preserve the wrong mixed-modality gate | Medium | High |

---

## 2. Change Summary

### 2.1 Audio-Independent Vision Blocks

The host helper now requires only an enabled checkpoint policy, a prefill sequence longer than one token, and at least one image or video token. Each contiguous vision run receives a non-negative block id; every other position, including audio, remains `-1`.

### 2.2 Matching MLX Graph Path

`Gemma4UnifiedModel::block_ids_array_for` no longer constructs or reads an audio-presence scalar. It still checks vision presence and uses the existing vision mask, block-start detection, cumulative numbering, and non-vision `-1` assignment, so the graph path has the same contract as the host helper.

### 2.3 Exact Mask Regression Coverage

The obsolete audio-disables-overlay test now expects `Some([-1, 0, 0, -1, -1])`. A new additive-mask test builds `[BOI, image, image, EOI, audio, audio, text]` and proves that the first image token can attend forward to the second image token while the first audio token remains unable to attend forward to the second audio token.

### 2.4 Documentation

The supported-model entry now explicitly states that audio tokens stay outside Gemma 4 Unified vision/video blocks and retain causal rows.

---

## 3. Technical Decisions

### 3.1 Define the Overlay Solely from Vision Runs

**Decision:** Audio presence does not participate in the overlay gate.

**Rationale:** The overlay relation is `same non-negative vision block`. Because audio positions are always `-1`, they cannot enter a same-block match and need no separate disabling rule.

**Trade-off:** Mixed prompts materialize the same vision overlay mask as image-only prompts rather than using the cheaper fully causal fallback, which is required for checkpoint correctness.

### 3.2 Preserve Shared Helper Semantics

**Decision:** Change the shared block-id helper rather than adding a Gemma 4 Unified call-site exception.

**Rationale:** DiffusionGemma also consumes the helper but uses `audio: -1`, so removing the audio gate does not alter its inputs. One contract prevents host and graph mask policies from drifting.

---

## 4. Review and Quality Findings

### 4.1 Implementation Review

The implementation reviewer found no unresolved correctness issues. Focused review covered the host helper, MLX graph operations, additive-mask semantics, DiffusionGemma reuse, and documentation.

### 4.2 Security and Performance Review

No CRITICAL, HIGH, MEDIUM, or LOW security/performance findings remained. The change removes one scalar reduction/readback from the graph path and does not add shape-dependent allocation or untrusted indexing. Audio tokens remain excluded structurally through block id `-1`.

### 4.3 Compatibility

- **Breaking changes**: None to CLI or HTTP interfaces.
- **New dependencies**: None.
- **Behavior change**: Combined image/video and audio prompts now honor the checkpoint's vision-bidirectional policy instead of falling back to fully causal vision rows.

---

## 5. Validation

- `cargo test --workspace --profile test-fast --features metal,accelerate` passed after final review.
- `cargo clippy --workspace --all-targets -- -D warnings` passed.
- `cargo fmt --all -- --check` and `git diff --check` passed.
- The focused Gemma 4 Unified mask selection passed 10 tests; broader Gemma 4 Unified selection passed 50 tests; DiffusionGemma selection passed 39 tests with 2 ignored.
- The local upstream regression `test_gemma4_unified_audio_tokens_keep_vision_overlay` passed, independently confirming the mixed-audio mask relation.
- A real local Gemma 4 Unified 12B 4-bit checkpoint on Metal completed image+audio and image-only greedy runs with finite, fluent output. The 16 kHz mixed input expanded to the same 324-token prompt length as the local reference run.
- Full real-generation token equality was not established: mlxcel and the local reference shared the opening token sequence and semantic answer but diverged in later descriptive wording. Separately linked pre-change and patched image-only binaries also diverged in later greedy wording despite unchanged image-only mask values, so exact real-output comparison remains sensitive to optimized build numerics.

---

## 6. Change Statistics

| Item | Value |
|------|-------|
| Files changed | 4 |
| Lines added | 44 |
| Lines deleted | 22 |
| Implementation commits | 1 |

### Related Commit

| Hash | Type | Message |
|------|------|---------|
| `8dc5e0169` | fix | Keep Gemma4 Unified vision overlay with audio |

---

## 7. Follow-up Considerations

- Add a stable first-logit or mask-tensor real-checkpoint parity harness if greedy token equality must be enforced across independently linked optimized binaries.
- Keep future multimodal mask gates based on the modality that participates in the overlay rather than unrelated modalities sharing the prompt.
- Preserve the exact audio `-1` invariant when extending Gemma 4 Unified batching or speculative prefill paths.

---

## References

- Issue #1344: mixed audio incorrectly disabling the vision overlay
- PR #1403: host and graph mask correction with regression coverage
- `docs/supported-models.md`: Gemma 4 Unified multimodal mask behavior
