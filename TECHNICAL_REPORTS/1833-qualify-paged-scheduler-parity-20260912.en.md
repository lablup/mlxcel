# Technical Report: PR #1833 - test: qualify paged scheduler parity

**Date**: 2026-09-12
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle (implementation review, security and performance review, finalization)
**Status**: Completed (the reported LLM-jp VLM divergence did not reproduce on the current CUDA build; the verified contract is deliberately narrower)
**Languages**: Rust, Markdown
**Risk Level**: Low (documentation and ignored real-model tests only; no request-path code changed)

---

## Executive Summary

PR #1833 replaces two unconditional claims that paged decode is byte-identical to dense decode with the narrower contract the repository can actually demonstrate. The integration test now records the full selector-logit row for each greedy token, requires identical greedy tokens, bounds relative RMS at `5e-5`, and adds a CUDA qwen3 arm that is run with TF32 disabled. The serving documentation now distinguishes throughput defaults from token-exact oracle settings and identifies `--decode-storage-backend dense` as the first backend-only bisect before using `--max-batch-size 1`, which also changes automatic storage selection.

The original `llm-jp/llm-jp-4-vl-9b-beta` report was re-run on an NVIDIA GB10. The CLI, default server, default-width server forced to dense, and `--max-batch-size 1` server all produced the same 15-token continuation for the 297-token prompt. This does not prove universal VLM parity or fused paged-v2 parity: the prompt remains below the fused dispatch floor and exercises gather-then-SDPA.

---

## Problem Statement

The previous documentation stated that paged decode was byte-identical to the dense backend. Its cited evidence, `tests/paged_scheduler_parity.rs`, was an ignored real-model test covering qwen3 and llama3 Fp16 dense-natural caches. It had no CUDA-specific arm and no VLM arm, so it could not support a guarantee spanning VLM front ends, model-owned caches, Turbo or quantized KV storage, and every CUDA reduction geometry.

Issue #1773 reported that `mlxcel generate` and the continuous-batching server differed by one Japanese token for an LLM-jp VLM checkpoint. A result obtained with `--max-batch-size 1` was not a clean batch-width comparison: with automatic storage selection, that option also moves decode storage from paged to dense. Without an explicit dense run at the default admission width, the observed recovery could be attributed to the wrong dimension.

The existing token-only parity test also left numerical drift invisible until it changed an argmax. A near tie may change output even when aggregate error is small, while a hard-coded Japanese continuation would couple a cache-backend contract to one model, processor, prompt, and vocabulary. The test needed to compare the backend property directly instead.

### Risk of leaving the old contract

| Risk | Impact | Likelihood |
|------|--------|------------|
| Operators treat throughput defaults as a universal token-exactness promise | Medium | Medium |
| `--max-batch-size 1` results are misdiagnosed as batch-width effects instead of storage-backend effects | High | Medium |
| CUDA numerical drift remains invisible until a greedy decision flips | Medium | Medium |
| A VLM-specific symptom drives unrelated kernel, reduction, or MLX-pin changes without a reproducing bisect | High | Medium |

---

## Change Summary

- The documentation scopes paged/dense parity to scheduler-shaped qwen3 and llama3 Fp16 dense-natural-cache runs at B=1/B=2, plus the explicit CUDA qwen3 numerical contract. VLM front ends, model-owned caches, Turbo/quantized KV modes, and all CUDA reduction geometries remain outside the claim.
- The continuous-batching guide now says that server defaults define a serving-throughput contract, not universal token-exact equivalence with the CLI. `--no-batch` selects the legacy worker, `--max-batch-size 1` keeps the scheduler but causes automatic storage to select dense, and `--decode-storage-backend dense` isolates storage while preserving default admission width.
- `DecodeStepTrace` associates every emitted token with the logit row that selected it. The first row is the terminal prefill row; later rows come from the preceding decode step. Review corrected an earlier off-by-one version that compared only post-token next logits.
- Full-vocabulary logits are converted to f32 on the host. The test rejects empty, NaN, and infinite rows, computes `RMS(actual - reference) / RMS(reference)`, requires the result to stay at or below `5e-5`, and separately requires greedy-token equality.
- A CUDA-gated qwen3 test constructs the same scheduler-shaped paged layout as the existing tests and compares it with a dense allocation. Evidence commands set `MLX_ENABLE_TF32=0`; `MLXCEL_REQUIRE_MODELS=1` turns a missing checkpoint from a soft skip into a failure.

---

## Technical Decisions

### Separate the storage-backend bisect from batch width

The selected first comparison is default admission width plus `--decode-storage-backend dense`. This changes only the decode storage choice under investigation. `--max-batch-size 1` remains useful as a scheduler-shaped single-request diagnostic, but it is not treated as a pure batch-width arm because `auto` resolves to dense at that width.

No scheduler behavior was changed. The PR documents how the existing selection works so future investigations do not build a causal claim on a confounded experiment.

### Test numerical and selection properties, not a sentence

The contract has two layers. Exact token equality catches user-visible greedy divergence. Relative RMS over the complete selector-logit row catches backend drift even when the argmax remains stable. This is more reusable than asserting the Japanese word involved in #1773 and keeps model processing, tokenizer behavior, and prompt wording outside a cache-storage test.

The trace is aligned to decisions rather than forward calls. If token `t[n]` is emitted from logits `L[n]`, the trace stores `(t[n], L[n])`; therefore the first comparison includes the terminal prefill logits instead of beginning one step late.

### Prefer bounded relative error over byte identity

Byte identity was stronger than the available evidence and unsuitable as a universal CUDA claim. A `5e-5` relative-RMS envelope, with TF32 disabled, defines a measurable numerical contract while preserving exact greedy decisions. Empty and non-finite inputs fail explicitly so NaN cannot accidentally bypass the comparison.

### Keep gather and fused paged evidence distinct

The 297-token LLM-jp request is below the 4,096-token fused paged-v2 dispatch floor. Its default-server run therefore validates the gather-then-SDPA paged path only. The report and documentation retain that boundary instead of presenting the successful bisect as fused-kernel evidence.

---

## Validation

### LLM-jp VLM CUDA bisect

The bisect used spark-101: NVIDIA GB10, compute capability 12.1, driver 580.173.02, Linux aarch64. The checkpoint was `/home/inureyes/models/mlx/llm-jp-4-vl-9b-beta`, the solid-red image SHA-256 was `5eafcdbe57b88e9c12ef8ac4cc3eee45f9c3433b7f929d867e5dd4d7832a8aed`, and every numerical run set `MLX_ENABLE_TF32=0`.

| Arm | Admission/storage choice | Result |
|-----|--------------------------|--------|
| `mlxcel generate` | CLI cache path | Same 15-token continuation |
| Server with `--decode-storage-backend dense` | Default max batch, forced dense | Same 15-token continuation |
| Default server | Default max batch, `decode_storage=auto` | Same 15-token continuation |
| Server with `--max-batch-size 1` | Scheduler retained, `auto` resolves dense | Same 15-token continuation |

The reported `キャンバス` divergence did not reproduce on the current build. The old installed binary was excluded because it could not load `llmjpvl`, not treated as evidence for a hardware-generation difference.

### Paged/dense contract

At final head `2dc42d9db33072a45fe26c36849eefacfbcdccee`, the following GB10 command required the checkpoint and passed:

```text
MLXCEL_REQUIRE_MODELS=1 MLX_ENABLE_TF32=0 cargo test --test paged_scheduler_parity --release --features cuda cuda_paged_scheduler_qwen3_matches_dense_within_logit_contract -- --ignored --nocapture
```

Dense and paged greedy traces were identical:

```text
[1079, 264, 5458, 304, 279, 220, 16, 15, 339, 11972, 13, 358, 614, 311, 3270, 264]
```

The result was 1 passed, 0 failed. The broader ignored integration target passed 5/5 at review commit `363d6648`; qwen3 executed and llama3 soft-skipped because its checkpoint was absent on the host. CUDA clippy, local format checking, `git diff --check`, Metal/Accelerate integration-target compilation, focused clippy, and all required CI checks also passed.

---

## Learning Points

- **A diagnostic flag can change more than its name suggests.** `--max-batch-size 1` changes admission width and automatic KV storage selection. A useful bisect varies one dimension explicitly before combining controls.
- **Numerical traces must be aligned with decisions.** Comparing the logits returned after a token while labeling them as the logits that selected that token omits the first decision and shifts every row. Recording selector logits makes the invariant explicit.
- **A passing soft-skip is not hardware evidence.** Real-model evidence commands should require model presence. Exploratory and CI runs may keep soft-skip behavior, but their output must not be reported as a checkpoint-backed pass.
- **Backend and hardware generation are separate hypotheses.** The investigation used an actual CUDA host and discarded an incompatible old binary. It did not infer an Apple-generation or CUDA-generation cause and did not widen into MLX-pin or kernel changes without a reproducer.

---

## Change Statistics

| Item | Value |
|------|-------|
| Files changed | 3 |
| Lines added | 219 |
| Lines deleted | 37 |
| Runtime request-path changes | 0 |
| New CUDA contract tests | 1 |

| Category | Summary |
|----------|---------|
| Tests | Added selector-logit traces, relative-RMS validation, finite-value guards, required-checkpoint evidence mode, and a CUDA qwen3 arm |
| Documentation | Replaced universal byte-identity language with the measured scope and documented unconfounded diagnostic controls |
| Runtime | No production scheduler, cache, kernel, reduction, or model code changed |

### Related commits

| Hash | Summary |
|------|---------|
| `51f277f` | Introduced the scoped parity contract and CUDA arm |
| `032353c` / `9e5f192` | Added and then fully reverted a temporary branch-scoped GB10 diagnostic workflow |
| `2967db0` | Clarified trace terminology before review exposed its alignment issue |
| `363d664` | Aligned every logit row with the token it selected |
| `2dc42d9` | Required model presence for evidence and hardened numerical validation |

---

## Follow-up Actions

- No kernel, reduction-order, scheduler, or MLX-pin fix is justified by the current non-reproduction.
- If the LLM-jp divergence reappears on a current pinned build, preserve the four-arm backend bisect, capture selector logits at the first divergent decision, and file the resulting correctness defect separately before expanding scope.
- Fused paged-v2, VLM-specific caches, model-owned caches, and Turbo/quantized KV storage need their own evidence before the documentation scope can be widened.

---

## References

- Issue #1773: server and CLI diverge by one token on LLM-jp-4-vl-9B
- `tests/paged_scheduler_parity.rs`
- `docs/CONTINUOUS_BATCHING.md`
- `docs/turbo-kv-cache.md`
