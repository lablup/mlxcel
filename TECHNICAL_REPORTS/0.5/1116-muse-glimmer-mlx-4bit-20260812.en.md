# Technical Report: PR #1116 - feat(models): support Muse Glimmer MLX 4-bit

**Date**: 2026-08-12
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1116 adds checkpoint-specific support for the pinned `mlx-community/Muse-Glimmer-30B-4bit` affine-Q4 conversion while preserving the merged dense BF16 Muse path. It normalizes mlx-vlm weight namespaces, propagates the root quantization contract into the text stack, uses quantization-aware text and vision-fusion layers, and keeps the 50-layer vision tower dense.

The security review found and fixed two fail-open metadata cases before PR submission: a non-string `quantization.mode` could bypass the mode comparison, and a missing affine `.biases` sidecar could make the shared loader infer a block-float format. Both VLM and text-only loaders now enforce the same pinned affine contract before kernel selection.

---

## 1. Problem Statement

### 1.1 Background

PR #1101 established Muse Glimmer with the 59.55 GB BF16 checkpoint, but its loader deliberately rejected quantization sidecars. The public mlx-community conversion is approximately 19.41 GB and uses a different root namespace plus root-level affine-Q4 metadata, so it could not be interpreted safely by the BF16-only boundary.

### 1.2 Existing Limitations

- mlx-vlm exports `language_model.*` and unwrapped vision roots that did not match mlxcel's canonical Muse namespaces.
- The decoder consumed `text_config`, while the published quantization contract lived at the JSON root.
- Dense-only fusion projections could not load the quantized adapter and projector.
- The one-shot CLI displayed Muse recipient envelopes and internal `to=self` reasoning even though server routes already separated them.
- Accepting arbitrary quantization metadata would risk selecting incompatible native kernels or producing silently incorrect output.

### 1.3 Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Alias keys overwrite canonical tensors | High | Collision-safe normalization rejects duplicate canonical destinations |
| Malformed metadata selects an unintended quantization mode | High | Exact affine contract, type checks, parameter validation, and required sidecars |
| Quantized vision tower reaches an unqualified path | High | Vision-tower sidecars are rejected before model construction |
| Internal reasoning appears in CLI output | Medium | Reuse the server recipient parser and hide `to=self` by default |

---

## 2. Technical Review

### 2.1 Security and Correctness

- Root `quantization`, compatibility `quantization_config`, and nested `text_config.quantization` must agree.
- `mode`, when present, must be a string equal to `affine`; group size and bit width pass the shared quantization validator.
- Normalized weight aliases cannot collide with one another or with canonical keys.
- Every supported `.scales` tensor requires matching `.weight` and `.biases`; orphan biases, global scales, and vision-tower sidecars are rejected.
- The text-only loader calls the same weight-map validator as the VLM loader, closing a boundary-specific bypass.
- No new dependency, network request, authentication/authorization path, filesystem write, subprocess, or unsafe block was added.
- CLI channel rendering delegates to the existing ATEM/Muse parser. The transformation is linear in already-generated output and does not broaden server-visible data.

No Critical or High issue remains after the two review fixes. The residual risk is limited to the explicitly pinned checkpoint-format contract and hardware qualifications documented below.

### 2.2 Performance

| Scenario | Result |
|----------|--------|
| BF16 text decode baseline | 4.25 tok/s |
| Q4 warm text prefill | 12.43 tok/s |
| Q4 warm text decode | 13.34 tok/s |
| Q4 image prefill | 5.80 tok/s including first-use compilation |
| Q4 image decode | 13.15 tok/s |
| Post-security-fix exact-source text run | 11.41 prefill / 13.07 decode tok/s |

The Q4 decode path is approximately 3.1x the recorded BF16 rate on the same NVIDIA GB10. The actual image run inserted 64 vision patch tokens and accurately described the solid orange-red fixture.

### 2.3 Compatibility

- Qualified checkpoint: `mlx-community/Muse-Glimmer-30B-4bit` revision `3e7677d7a40d348a3daba263a2b1c0aa41910710`.
- Qualified hardware: Linux/aarch64 NVIDIA GB10 with CUDA.
- Preserved path: canonical dense BF16 Muse checkpoint from PR #1101.
- Explicitly unsupported: video, quantized vision tower, Turbo/INT8 KV, speculative/DFlash, LoRA/adapters, TP/PP, XLA/IREE/OpenXLA, distributed, and disaggregated serving.
- No new package dependency or wire-format change.

---

## 3. Technical Decisions

### 3.1 Normalize Names Without Copying Tensor Data

**Decision:** Move weight handles into a new collision-checked map after converting published mlx-vlm roots to canonical Muse roots.

**Rationale:** This keeps one runtime namespace for BF16 and Q4 model construction while avoiding tensor-data duplication. Rejecting collisions prevents a crafted alias from silently replacing a canonical weight.

### 3.2 Keep the Vision Tower Dense

**Decision:** Enable quantization-aware layers only for the text stack, LM head, vision adapter, and vision projection.

**Rationale:** The pinned conversion leaves its vision tower dense. Generalizing beyond that evidence would expose untested shapes and kernels, so vision-tower sidecars fail closed.

### 3.3 Share Recipient Parsing Between Server and CLI

**Decision:** Export the server's Muse channel renderer for one-shot CLI display rather than implement a second parser.

**Rationale:** One parser preserves identical `to=self` and `to=user` semantics and reduces the chance that reasoning or structural tokens leak through a divergent CLI implementation.

---

## 4. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 16 |
| Lines added | 477 |
| Lines deleted | 69 |
| Commits | 2 |

### Major Areas

| Area | Summary |
|------|---------|
| Loading/config | Root quantization inheritance, namespace normalization, sidecar and mode validation |
| Model/vision | Quantization-aware text and fusion layers with dense tower preservation |
| CLI/tool channels | Shared Muse recipient rendering and reasoning suppression |
| Tests | Alias, collision, config disagreement, malformed mode, sidecar, CLI, and real-checkpoint coverage |
| Documentation | Pinned revision, size, support boundary, and GB10 throughput |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `4ea5aca2` | feat | support Muse Glimmer MLX 4-bit |
| `9a23ed2e` | fix | enforce Muse affine Q4 contract |

Follow-up to PR #1101.

---

## 5. Validation and Follow-up

### Passed

- `cargo fmt --all --check`
- Clippy for library and binaries with `-D warnings`
- 100 Muse-filtered library tests
- Two CLI help/metadata regressions
- CUDA release build with `MLX_CUDA_ARCHITECTURES=121`
- Real pinned-checkpoint text and image generation on NVIDIA GB10
- Final exact-source text run loaded in 0.480 seconds and printed only `The capital of France is Paris.` without recipient or control-token leakage

### Remaining Boundary

- Hosted PR checks must pass before merge.
- Apple Silicon/Metal is not qualified by this report.
- Additional quantization formats and the unsupported execution modes remain separate, evidence-gated follow-ups.
