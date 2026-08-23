# Technical Report: PR #1386 - apply the linear rope_scaling factor on Gemma 3 global-attention layers

**Date**: 2026-08-23
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust, C++ (cxx bridge)
**Risk Level**: Medium-High (changes Gemma 3 decode numerics, and the enabling change is a signature edit on a launcher shared with Qwen3 and Qwen3-MoE)

---

## Executive Summary

Gemma 3 checkpoints from 4B up declare `rope_scaling: {"rope_type": "linear", "factor": 8.0}`, which applies to the global-attention layers only; the sliding layers keep an unscaled RoPE at `rope_local_base_freq`. mlxcel parsed the block and never read it, so every layer rotated at `scale = 1.0` and the global layers saw positions eight times larger than the model was trained with.

This is the third instance of the same defect class in this batch, after the shared Llama decoder (#1355) and with Qwen3 filed as a fourth (#1388). The interesting content here is not the fix, which is small, but three decisions around it: reusing rather than re-deriving the machinery #1355 built two hours earlier, choosing a cross-family C++ signature change over the cheaper fallback that would have cost a supported family its fused kernel, and a validation near-miss where the probe was too weak to distinguish the broken binary from the fixed one.

## 1. Problem Statement

### 1.1 Background

`src/models/gemma3.rs` declared `pub rope_scaling: Option<HashMap<String, serde_json::Value>>` and referenced it exactly twice: the declaration and a `None` in a defaults constructor. Nothing consumed it. All three RoPE call sites passed a literal `1.0` scale.

The per-layer rule the reference implements, with `is_sliding = (l + 1) % sliding_window_pattern != 0`:

- sliding layer: `base = rope_local_base_freq`, `scale = 1.0`
- global layer: `base = rope_theta`, `scale = 1 / factor`

so `theta_{p,i} = (p * scale) * base^(-2i/head_dim)`, and a global layer at offset 4096 with `factor = 8` must rotate exactly as an unscaled layer at position 512.

### 1.2 Existing Issues

Same signature as its siblings: fluent output at every prompt length, no error, no warning, and a divergence that grows with position. `models/gemma-3-4b-it-4bit` declares the block; `models/gemma-3-1b-it-4bit` declares none, which made the 1B checkpoint an exact control rather than merely a second data point.

### 1.3 Risk Assessment

Medium-high, driven by the enabling change rather than the fix. The fused quantized launcher `fused_qkv_project_split_norm_rope` had no scale parameter at all, so honoring the block on the fused path required editing a cxx bridge signature shared by Gemma3, Qwen3 and Qwen3-MoE. A transposed scalar there would compile cleanly and silently corrupt three families.

## 2. Change Summary

9 files, roughly 557 insertions before review fixes.

| Area | Change |
|---|---|
| `src/models/gemma3.rs` | `global_rope_scale()`, `layer_rope_params()`, `Attention::rope_scale`, all three RoPE sites |
| `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.{h,cpp}` | `rope_scale` parameter on the fused launcher, forwarded to `mx::fast::rope` |
| `src/lib/mlxcel-core/src/lib.rs`, `layers.rs` | Bridge declaration and Rust wrapper threading |
| `src/models/qwen3.rs`, `qwen3_moe.rs` | Pass `1.0` |
| `src/models/rope_utils.rs` | `printable_label` and `is_usable_scalar` made `pub(crate)` for reuse |
| `src/models/gemma3_tests.rs` (new) | 12 tests |
| `docs/supported-models.md` | Gemma 3 `rope_scaling` policy and the deliberate asymmetry with the Llama path |

## 3. Technical Decisions

### 3.1 Reusing #1355's reader instead of writing a third copy

#1355 landed `src/models/rope_utils.rs` earlier the same day with a `serde_json::Map`-based `type`-then-`rope_type` lookup, specifically because a derived `#[serde(rename = "type", alias = "rope_type")]` field hard-errors with `duplicate field` on the five local checkpoints that spell both keys. Gemma 3 reads through the same `RopeScalingSpec::from_lookup`, so that trap cannot reappear here, and the both-keys case is pinned by a test.

Review caught the one place this was not carried through: `global_rope_scale` open-coded `factor > 0.0 && factor.is_finite()` rather than calling the shared `is_usable_scalar`, whose own doc says it exists so the arms cannot screen to different standards. Semantics were identical, so this was drift rather than a defect, but it is exactly the drift the helper was written to prevent. It now calls the helper, which carries a `// Used by:` line naming Gemma 3 and a note on why Gemma 3 bypasses `RopeScalingKind::resolve`.

### 3.2 A load error here, a warning on the Llama path, and why that asymmetry is correct

#1355 has a nearly identical acceptance criterion requiring unsupported `rope_type` values to be named load errors, and implementing it literally there would have stopped `models/internvl3-1b` from loading. It was deliberately softened to warn-and-continue.

The same criterion is **safe to implement literally on Gemma 3**, and the reason is structural rather than incidental. VLM loaders route an arbitrary `text_config` into `llama3::ModelArgs` (eight of them do), so an unimplemented scheme arriving there belongs to a model whose text backbone merely happens to be Llama-shaped. No Gemma 3 checkpoint does that: the only configs reaching these args are Gemma 3's own, and every local one declares `linear`. Failing loudly therefore cannot strand a model that used to load.

This asymmetry is now recorded in `docs/supported-models.md` next to the Llama paragraph, because two adjacent families behaving differently on the same input is precisely what that file exists to explain. The risk was real: having just finished #1355, the natural move is to pattern-match and copy its warn-and-continue behavior, which would have been the wrong call here.

### 3.3 Extending the C++ signature rather than gating the fused path off

Two options existed. (a) Add `rope_scale` to the launcher and thread it through every caller. (b) Gate the fused call on `rope_scale == 1.0` so global layers fall back to the graph path.

(b) is cheaper and touches no C++, but it costs Gemma 3 4B, 12B and 27B their fused prefill kernel on every global layer, and this repository has a documented history of quietly losing kernels that way. (a) was taken. Verification that it was safe is empirical rather than argued: Qwen3 and Qwen3-MoE come out **byte-identical** to the pre-change binary on real checkpoints.

The correctness question that mattered is parameter position. `rope_scale` sits between `rope_base` and `rms_eps` in all four layers (bridge declaration, C++ header, C++ definition, Rust wrapper), and the C++ passes it to MLX's scale slot, not the base slot. That was confirmed against MLX's own declaration `rope(x, dims, traditional, base, scale, offset)` and the Metal kernel's `float L = scale * (pos.y + batch_offset)`, rather than by reading the Rust side alone. A transposition would have compiled and produced fluent, wrong output for three families.

## 4. Verification

### 4.1 A probe that the buggy binary already passed

The first token-exactness probe hit an exact 0.00 logprob tie at step 28, so it was rebuilt. The second probe was token-exact, and that was the problem: **the pre-change binary was already 64/64 on it**. It proved the fixed binary matches the oracle, and proved nothing about whether the fix changes anything.

This is the mirror image of #1355's near-miss. There, a genuine improvement looked like a failure because the discriminating step was noise-decided. Here, a real defect looked absent because the probe never exercised the position range where the two frequency schedules diverge. Both failure modes produce a green result.

The resolution was a multi-needle retrieval probe. Final results against the mlx-lm oracle on identical weights:

| Probe | Prompt tokens | Oracle tokens | Pre-change match | Post-change match | Min top-2 margin |
|---|---|---|---|---|---|
| multi-needle, five codes | 2710 | 46 | 19 | **46** | 2.875 |
| multi-needle, two codes in prose | 2697 | 42 | 20 | **42** | 1.125 |
| needle + section number | 2706 | 27 | 3 | **27** | 0.125 |
| continue the document | 3140 | 64 | 64 | **64** | 1.75 |

The pre-change binary emits the wrong archive code (`XC-1057-RM` where the oracle emits `SV-3390-LD`) at a step whose oracle margin is 27.0 nats, so that divergence is decided by the model, not by rounding. The smallest margin anywhere in the 46-step probe is 2.875, so no step in it is noise-decided.

A probe is only evidence if it separates the two binaries **and** its discriminating step has a healthy margin. Checking one without the other is how both near-misses in this batch happened.

### 4.2 The control

`models/gemma-3-1b-it-4bit` declares no `rope_scaling` and is byte-identical to the pre-change binary on all three prompts, including the 3140-token one. That is the other half of the claim: the change must alter scaled configs and nothing else.

The exactness is analytic as well as empirical. An absent block returns `Ok(1.0)` before any arithmetic, and on the fused path the C++ replaced a `1.0f` literal with a parameter carrying `1.0f`, so the emitted graph node is identical rather than numerically close.

### 4.3 Gate

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8338 passed, 0 failed. `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings` and `cargo fmt --all -- --check` clean. Server batched decode verified with two concurrent 2710-token requests matching their single-request outputs, and a `"rope_type": "yarn"` config copy verified to fail with the named load error.

## 5. Findings from Review

Nothing above MEDIUM. Four applied: the docs entry above, the shared-predicate reuse above, a rustdoc heading split across two lines (which rendered as a heading plus a stray paragraph), and an unnecessary `pub(crate)`.

Three declined with reasons. A positional `(bool, f32, f32)` return of two same-typed floats is the same swap-and-still-compile hazard one level up, but it is pinned by a test asserting base and scale separately for all twelve layers, and refactoring near numerics late in review carries its own risk. A pathological `factor: 1e30` passes the positive-and-finite screen and yields effectively positionless attention; upstream screens nothing at all here and no real checkpoint does it, so the boundary is left where #1385 put it. And the misleading Qwen3 comments were filed as #1388 rather than fixed in a Gemma PR.

## 6. Adjacent Issues Filed

**#1388**: Qwen3 and Qwen3-MoE declare `rope_scaling` and never read it, the same class one family over. Latent rather than live: local plain Qwen3 checkpoints omit the block entirely, and the qwen3-vl variants parse their own struct in `qwen3_vl.rs`. Upstream is asymmetric in a way worth knowing before anyone fixes it: `mlx_lm/models/qwen3.py` passes `scaling_config` into `initialize_rope`, so `qwen3.rs` genuinely diverges, while `qwen3_moe.py` builds a plain `nn.RoPE`, so `qwen3_moe.rs` currently matches upstream and fixing it would deliberately deviate.

**#1387**: `scripts/ci/check_cross_repo_refs.py` classifies any bare `#NNN` at or above 1000 as likely-upstream, on an assumption its own comment states and that expired when the repository passed issue 1000. The #1385 merge commit alone produces 31 hits, 28 of them misclassified. The check is advisory so CI stays green, but its purpose is stopping internal issue numbers from reaching the public tree, and a guard that mostly cries wolf stops being read. Re-tuning the constant reschedules the same failure, so the issue proposes inverting it the way `check_crate_versions.py` inverts its crate list.

## 7. What Remains Unverified

Gemma 3 12B and 27B are not present locally; only 4B and 1B were exercised. The pipeline stage executor and the tensor-parallel runtime were verified by inspection (the stage executor has no RoPE call of its own, and `local_gemma3_args` clones `rope_scaling`), not by a multi-device run.

One pre-existing gap recorded so nobody later assumes this PR covered it: `Gemma3StageModel::load` parses the top-level `config.json` into `ModelArgs`, so on a Gemma 3 VLM checkpoint it never sees `text_config.rope_scaling`. That path is effectively unreachable today, because `ModelArgs` is `#[serde(default)]` and would already mis-read every other text field from a VLM-shaped config.

## 8. Learning Points

- A probe that passes identically on the broken and the fixed binary is not evidence, however green it looks. Confirm the probe separates them before trusting that it validates anything.
- The two ways a parity run misleads are opposites and both look like success: a real improvement hidden by a noise-decided step, and a real defect hidden by a probe that never reaches the divergent regime. Check the margin and check the separation.
- Two adjacent families can correctly behave differently on the same malformed input. The reason here is structural (VLM loaders funnel arbitrary `text_config` into the Llama args and nothing does that to Gemma 3), so it belongs in the docs, not in a code comment.
- When a helper exists to prevent divergence, a third caller open-coding its body is the divergence arriving. Reach for `pub(crate)` and a `Used by:` line instead.
- Verify a cross-language signature change against the callee's own declaration and kernel source, not against the caller you just wrote. A transposed scalar of the same type is invisible to both compilers.
