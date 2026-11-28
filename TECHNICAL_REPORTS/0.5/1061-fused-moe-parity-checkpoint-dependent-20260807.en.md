# Technical Report: PR #1061 - docs(moe): record that fused decode-MoE byte-identity is checkpoint-dependent

**Date**: 2026-08-07
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Completed
**Languages**: Rust doc comments, Markdown
**Risk Level**: Low (documentation only; the investigation cleared the kernel)

---

## Executive Summary

The fused single-token decode-MoE kernel changed greedy output on the newly added Klear family, which read as a regression against #268's "byte-identical greedy output" claim. Measurement showed the opposite: the fused kernel is roughly 6x *closer* to an all-f32 ground truth than the `gather_qmm` reference path it disagrees with, on both Klear and `qwen3-30b-a3b`. #268's claim still holds exactly where it was made, so it never generalized rather than regressed. This PR corrects the documentation and records why `MLXCEL_FUSED_MOE=0` is the right switch when reference-diffing a MoE port.

---

## 1. Problem Statement

### 1.1 Background

Issue #1045 was filed while validating PR #1044 (the Klear port). On `Kwai-Klear/Klear-46B-A2.5B-Instruct` converted locally to 4-bit:

- A 5-token prefill, where the kernel does not engage since it is gated on single-token input, reproduced the mlx-lm reference's top-5 logits exactly.
- A 1-token probe with `MLXCEL_FUSED_MOE=0` also matched exactly.
- The same probe with the kernel enabled shifted every value by roughly 0.2 on a logit of 7.5 and reshuffled the near-ties below rank 2.
- End to end, greedy decode produced a different but equally coherent continuation.

The kernel is shared (`src/models/switch_layers.rs`, `SwitchGLU::forward_fused_kernel`) and reached by every family in its `Used by:` list, so the blast radius was unknown.

### 1.2 Existing Issues

- **The doc comment read as a general guarantee.** `fused_moe_enabled` said the kernel is "byte-identical or within the documented f16 jitter class" across the validated MoE set, which a reader would take as a property of the kernel.
- **A 3% shift on a logit is large for pure reordering,** as the issue noted, so "it is just jitter" could not be assumed.
- **There was no recorded guidance for reference-diffing.** A new MoE port that matched at prefill and diverged at decode would look like a porting bug.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A real numeric defect in a default-on kernel shared by nineteen families | Critical | Unknown at filing; this is what the investigation had to settle |
| A future port loses days chasing a porting bug that is actually the kernel gate | Medium | High, with no documented guidance |
| The false "byte-identical" guarantee is relied on for a parity gate | Medium | Moderate |

---

## 2. Technical Review

### 2.1 The instrument

`MLXCEL_FUSED_MOE_PARITY_CHECK=1`, which already existed from the #886 corruption triage, is the right instrument and needed no new code. Per MoE call it compares three things: the fused kernel, the `gather_qmm` production fallback, and an all-f32 dequantize-and-matmul ground truth over the same selected experts. It also re-runs the fused pair on identical inputs to check bitwise determinism.

Setting `MLXCEL_FUSED_MOE_PARITY_THRESHOLD=0` makes every call report rather than only the outliers.

### 2.2 Results

Normalized RMS, 96 decode MoE calls per checkpoint, M1 Ultra.

| checkpoint | fused vs truth | gather_qmm vs truth | fused vs gather_qmm | fused closer to truth |
|---|---|---|---|---|
| `Klear-46B-A2.5B-Instruct` 4-bit (256 experts, top-8, dff 896, 32 layers) | **1.655e-3** | 1.025e-2 | 1.040e-2 | **96/96**, median 6.21x |
| `qwen3-30b-a3b` 4-bit (128 experts, top-8, dff 768, 48 layers) | **1.650e-3** | 1.022e-2 | 1.034e-2 | **96/96**, median 6.16x |

Spreads: Klear fused 1.539e-3 to 1.717e-3 and reference 3.535e-3 to 1.373e-2; qwen3 fused 1.106e-3 to 2.105e-3 and reference 3.034e-3 to 1.306e-2. Zero non-deterministic reruns across 192 calls.

Two readings follow.

**The fused kernel is the more accurate path.** Its deviation from truth is a stable ~1.65e-3 on both families while `gather_qmm`'s tracks the checkpoint, so `fused-vs-gather_qmm` (1.04e-2) is very nearly `gather_qmm-vs-truth` (1.02e-2). The disagreement the issue measured is mostly the *reference's* distance from truth. The 0.2-on-7.5 logit gap is that difference accumulated over 32 layers plus a router, not a 3% error in the kernel.

**It never generalized, and did not regress.** Re-running #268's claim at current HEAD on `qwen3-30b-a3b`, 64 greedy tokens at `-t 0 --no-chat-template`, produced byte-identical text with the kernel on and off; only the timing lines differ.

### 2.3 Blast-radius survey

The kernel is shared, so the real question is not which family is broken (none are) but which family's routing is knife-edge enough for a sub-ulp difference to flip a greedy argmax. The two measured families bracket it:

- `qwen3-30b-a3b` routes top-8 of 128 and never flips.
- Klear routes top-8 of 256 and additionally blends its shared expert through a learned 2-way softmax over an `mlp.coefficient` head, so it has two amplification stages where qwen3 has one. It flips.

Since the numeric behaviour is family-independent to three digits, running the probe across the remaining callers would re-measure the same ~1.65e-3 rather than surface a new defect. What varies is routing sharpness, which is a checkpoint property.

### 2.4 Performance

Kernel on measured 83.1 tok/s against 71.1 off on `qwen3-30b-a3b`, but that is one sample each on a heavily loaded box. Recorded as directionally consistent with #268's ~3.5% rather than as a measurement.

---

## 3. Technical Decisions

### 3.1 No code change

**Context:** The issue's acceptance criteria branch on the determination. "If a defect: a fix, plus a parity test that would have caught it. If benign: documentation."

**Rationale for the benign branch:** the fused path is closer to ground truth than the reference path in 96 of 96 calls on each of two families, and every rerun was bitwise deterministic. There is nothing to fix and nothing new to pin. `fused_moe_parity_tests` already pins the kernel against the f32 reference, and `MLXCEL_FUSED_MOE_PARITY_CHECK` already measures exactly what settled this.

**Trade-off:** a reader who expected a kernel change may read a documentation-only PR as the issue being dismissed. The doc comment carries the table for that reason, so the evidence sits next to the claim rather than only in the issue thread.

### 3.2 Where the guidance lives

Three places, chosen by who would look:

- `src/models/switch_layers.rs`, for someone reading the gate.
- `docs/adding-models.md`, for someone mid-port whose decode diverged.
- `docs/environment-variables.md`, for someone scanning the switch list.

---

## 4. Implementation Details

`fused_moe_enabled`'s doc no longer claims byte-identity across the validated set. It states that byte-identical greedy output is not a general property and never was, carries the two-checkpoint table, and explains that whether the difference flips a greedy argmax depends on how knife-edge the checkpoint's routing is.

`forward_fused_kernel`'s bare "byte-identical greedy output" is scoped to the checkpoint it was measured on and points at `fused_moe_enabled` for the full picture.

`docs/adding-models.md` gains a subsection under the reference-selection guidance, saying to set `MLXCEL_FUSED_MOE=0` when reference-diffing a MoE port. It gives the mechanism (the kernel engages only at `l == 1`, so a prefill comparison is unaffected while a decode comparison is not) and the diagnostic value: a port that looks exact at prefill and diverges at decode is very likely hitting this rather than a porting bug, so rule it out first.

---

## 5. Lessons

- **"Which of two disagreeing paths is right" is answerable, and often neither is the reference.** Comparing both against an independent ground truth turned an apparent kernel defect into a measurement showing the reference is the less accurate of the two. Diffing the two paths against each other could never have produced that.
- **A claim's scope is part of the claim.** #268's measurement was correct and remains correct; only the sentence generalizing it was wrong. The fix is to scope the sentence, not to re-litigate the measurement.
- **The right instrument may already be in the tree.** `MLXCEL_FUSED_MOE_PARITY_CHECK` was built for a different investigation (#886) and answered this one with no new code, including the all-f32 ground truth that made the conclusion possible.
