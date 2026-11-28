# PR #3404: Resolve the late XLA vs eager greedy divergences (Gemma3, Phi4MM vision, Gemma3n pin)

**Date**: 2026-11-26
**Status**: PR open (closes #3391; part of #932 and epic #566). Probes pass on the pinned checkpoints on GB10 CUDA; margins recorded at both assert sites

## Executive Summary

Three late XLA versus eager greedy divergences remained after #3365. This PR measures the top-2 logit margins on both sides at each divergence step, then decides per item: a defect is fixed, a near-tie is documented with its margin. All three resolve without a graph change: Gemma3 is a documented cross-backend near-tie (both arms rank the same top-2 candidate set; eager picks 2258 by exactly one bf16 ulp, XLA picks 3355), Phi4MM's vision path is bit-identical at every stage and the reported divergence does not reproduce on current main, and Gemma3n's prefix-decode top1 re-pins to 236798 (both production CUDA paths agree; only the third-party mlx-vlm CPU oracle ranks 854). Two margin probes gate the parity a future change would break.

## 1. Problem Statement

### 1.1 The three divergences

- **Gemma3** `gemma-3-4b-it-4bit`, `test_image.png`, `-n 16 --temp 0`: the first 15 tokens match eager; token 16 is eager 2258 versus XLA 3355, on both cuda and local-task. The 34-stage boundary gate passes, so the divergence is a ranking flip, not a numeric gap.
- **Phi4MM vision** `phi-4-multimodal-instruct` (needs the LoRA layout from #3380): 8 tokens matched, then XLA said "white line in the middle" and eager said "on the left side". XLA f32 and f16 vision gave the same tokens, so the gap sat upstream of the decoder.
- **Gemma3n** canonical lib test `pinned_e2b_intermediates_prefill_decode_and_greedy_match` failed at `src/models/gemma3n.rs`: "pinned production-CUDA prefix-decode MLX top1 drifted", pinned 854, observed 236798. The code comment called it a close-ranking cross-device flip, which was unproven.

### 1.2 Why measure before deciding

#3291 and #3390 established the rule: eager's result is the reference, and near-ties resolved the other way are recorded, not "fixed" by moving rounding boundaries. A margin within one dtype step at the divergence magnitude is a near-tie; a real numeric gap shows a same-candidate-set failure with a large margin.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/loading/vlm_gemma3_margin_tests.rs` (new) | `pinned_gemma3_greedy_margin_probe`: drives the eager and XLA greedy loops the way the CLI does, records per-step top-2 margins, and asserts top-2 candidate-set agreement at every step (a real numeric divergence fails it). |
| `src/lib/mlxcel-xla/src/batch.rs` | `XlaPreparedPrefillLogits::decode_ragged_logits`, the teacher-forced continuation shim; gated `all(iree, any(test-support, diagnostics))` and delegating to the ungated `decode_ragged_logits_with_modes` with a zero adapter-modes vector, so both `test-support` and `diagnostics` builds compile (the review fix). |
| `src/loading/vlm_phi4mm_margin_tests.rs` (new) | `pinned_phi4mm_vision_bisect_probe`: per-stage bit-identity between eager and XLA through preprocessing, per-crop encoder hidden states, pooled and projected features. |
| `src/models/gemma3n.rs` | Prefix-decode top1 re-pinned 854 to 236798, measured margins documented at both assert sites. |

3 files plus module declarations in `src/loading/vlm.rs` and `vlm_gemma.rs`. No tolerance or threshold changed.

## 3. Technical Decisions

### 3.1 Candidate-set agreement as the near-tie gate

The Gemma3 probe asserts the top-2 candidate-set agreement at every step instead of pinning a token: a real numeric gap changes which candidates are in the top-2, while a near-tie keeps the set and flips the ranking. The measured margins at the divergence step: eager top1 2258 (36.000) over 3355 (35.750) by 0.250, exactly one bf16 ulp at a ~36-logit magnitude; XLA top1 3355 (36.274) over 2258 (35.408) by 0.866. Steps 0-14 agree on top-1 on both arms.

### 3.2 The Gemma3n re-pin evidence

Both production CUDA paths agree: the eager prefix-decode ranks 236798 first by 0.3125 at a ~14-logit magnitude, and the XLA side agrees by 0.375. Only the third-party mlx-vlm 0.7.8 CPU oracle still ranks 854 by 0.75. The flip is cross-build (MLX pin advanced by #3134 three times since the 551039ee capture) and cross-device, not a graph defect. Per the issue's re-pin rule the pin moves 854 to 236798 with the margins recorded next to it; `mlx_logits` stays 236798 and `mlx_greedy` stays [236798, 236789].

### 3.3 The shim's feature gating

`IreeRaggedLlama::decode_ragged_logits` is `#[cfg(feature = "diagnostics")]`, so a shim gated `any(test-support, diagnostics)` but calling it broke the `test-support`-only build (the root `test-utils` feature forwards `mlxcel-xla?/test-support`), including the probe's own documented recipe. The shim delegates to the ungated `decode_ragged_logits_with_modes` with a zero adapter-modes vector of length `b_max()`, mirroring the diagnostics-gated wrapper's body (same OneD and length validation). Both feature sets compile; `cargo check -p mlxcel-xla --features test-support` and the xla-diagnostics build both pass.

## 4. Results

GB10 CUDA, pinned checkpoints, gpu-lock.

| Item | Result |
|---|---|
| Gemma3 margin probe | passed (200.68s): steps 0-14 top-1 agreement, step 15 same candidate set, near-tie documented |
| Phi4MM bisect probe | all stages max abs 0.0; CLI greedy identical 16 tokens on both backends |
| Gemma3n re-pinned one-row test | 1 passed, 0 failed |
| `cargo check -p mlxcel-xla --features test-support` | passes (the review fix) |
| `cargo check --no-default-features --features xla-diagnostics` | passes |

## 5. Limits

- **Phi4MM's reported divergence does not reproduce.** The vision path is bit-identical and the CLI produces identical tokens; the earlier observation was taken before #3380's LoRA layout fix landed. The probe now gates that parity.
- **The probes are `#[ignore]`d behind pinned-checkpoint env vars**, matching the diagnostics-test conventions; they do not run in CI.
- **Pre-existing host limitation.** The dev-profile integration test `pipeline_ci_multi_stage_real_models` does not link on this host with the cuda feature combination (AArch64 BL-range overflow, same class as the earlier continuum-router finding); test builds use `--lib`.
- **The CPU oracle is third-party.** mlx-vlm 0.7.8 ranks 854 first on the Gemma3n prefix-decode; only its margin is recorded, not gated.

## 6. Learning Points

- Measure top-2 margins before deciding near-tie versus defect: a candidate-set agreement probe distinguishes a ranking flip (near-tie) from a real numeric gap (different candidates) without pinning a token.
- A test-utils consumer of mlxcel-xla breaks when a shim's feature gate is narrower than its callee's; forward to the ungated variant instead of widening the callee's gate.
- A pinned value captured from an earlier MLX pin decays as MLX advances; record the capture commit next to the pin so the drift is attributable.

## Related Commits

| Hash | Type | Message |
|---|---|---|
| `2fe6dce` | test | gate the Gemma3 and Phi4MM greedy divergences with margin probes |
| `ab0029c` | fix | re-pin the Gemma3n prefix-decode top1 at 236798 |
| `d4a8ec8` | fix | call the ungated ragged-decode variant from the diagnostic shim |
| `9dc0b01` | docs | deduplicate the Gemma3n re-pin comment and align the probe byte order |
| `64cf984` | docs | reflow the Gemma3n re-capture paragraph lead-in |

Related: #3391 (this change), #3365 (found the divergences), #3380 (Phi4MM LoRA layout), #3291 and #3390 (the near-tie rule), #3134 (MLX pin advances), #932, #566.

## Verification

- Gemma3 margin probe passed first-hand under gpu-lock (200.68s), margins matching the recorded values; Phi4MM CLI greedy re-verified first-hand, identical 16 tokens.
- `cargo test -p mlxcel --lib --no-run` compiles; `cargo fmt --check` clean; `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings` clean.
- Implementation review: approve after the shim gating fix; security review: approve, no CRITICAL/HIGH.
