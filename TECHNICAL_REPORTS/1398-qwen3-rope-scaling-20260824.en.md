# Technical Report: PR #1398 - Apply Qwen3 RoPE Scaling End to End

**Date**: 2026-08-24

**Author**: Jeongkyu Shin

**Status**: Completed

**Languages**: Rust, C++, CUDA, Markdown

**Risk Level**: Medium

## Executive Summary

PR #1398 consumes the `rope_scaling` blocks that dense Qwen3 and Qwen3-MoE previously parsed and discarded. The resolved plan is constructed once per model, propagated through normal, batched, pipeline-parallel, and tensor-parallel paths, and applied consistently to graph and fused Q/K-normalization RoPE launchers. Linear scaling remains eligible for the fused launcher; frequency-table schemes use the graph path. Unsupported or malformed schemes produce a checkpoint-keyed one-shot warning instead of silently decoding unscaled.

Real-model qualification ran locally on the attached NVIDIA GB10 with existing 4-bit 4B and 30B-A3B weights. GitHub Actions downloaded no large checkpoints; the abandoned one-off workflow and its run were cancelled and removed from the final change.

## 1. Problem Statement

`qwen3::ModelArgs` and `qwen3_moe::ModelArgs` both deserialized `rope_scaling`, but every attention layer still passed `1.0` to RoPE. A checkpoint declaring `linear`, `llama3`, or another scaling scheme therefore loaded successfully while silently using an unscaled position table. The defect was latent for the repository's unscaled checkpoints, but it would degrade long-context behavior for future or third-party scaled checkpoints.

## 2. Change Summary

| Area | Change |
|---|---|
| Shared resolution | Reuse `RopeScalingSpec` and `RopeScalingKind`; resolve once with a checkpoint-specific diagnostic label |
| Dense Qwen3 | Carry `rope_scale` and optional frequencies into regular, batched, fused, and graph attention paths |
| Qwen3-MoE | Apply the same plan to fused decode and graph prefill/decode paths |
| Fused primitive | Document and test that its position scale has the same semantics as `fast_rope` |
| Distributed paths | Preserve the resolved configuration through pipeline-stage construction and tensor-parallel argument localization |
| Diagnostics | Warn once per checkpoint and scheme for malformed or unsupported scaling instead of silently using `1.0` |
| Regression suite | Cover key precedence, invalid factors, exact warning cardinality, hand-computed rotations, fused/graph parity, batched attention, real loaders, pipeline parallelism, and tensor parallelism |

## 3. Technical Decisions

### 3.1 Resolve once and carry an attention plan

Each model resolves the config into `Default`, `Linear { scale }`, or `Llama3 { freqs }` before building its layers. A linear factor of 8 becomes the MLX position multiplier `0.125`. Attention blocks receive owned handles to the same resolved plan, avoiding configuration lookups and repeated frequency arithmetic in forward calls.

### 3.2 Keep linear scaling on the fused path

The Qwen3 fused QKV/QK-normalization launcher already accepts a scalar RoPE position multiplier. Linear scaling therefore passes `1 / factor` directly and retains the optimized decode path. A precomputed frequency table cannot be expressed by that launcher, so `llama3` routing deliberately falls back to `fast_rope_with_freqs`.

### 3.3 Warn rather than fail arbitrary VLM-backed configs

MiniCPM-o can deserialize an arbitrary full config into `qwen3::ModelArgs`. Turning an unsupported scaling type into a hard load error could take an otherwise working VLM offline. The implementation preserves its former unscaled fallback but makes the fallback visible exactly once, keyed by checkpoint name and scheme. Missing, non-finite, zero, or negative linear factors follow the same named outcome.

### 3.4 Keep hosted CI lightweight and perform large-model qualification locally

The repository's established boundary is retained: Actions may exercise small, approximately 0.6B model fixtures, while tests requiring larger checkpoints run locally against weights already present on the qualification machine. The temporary 4B/30B hosted workflow was cancelled and deleted. The final PR adds no workflow that downloads those checkpoints.

## 4. Verification

### 4.1 Deterministic code gates

- `cargo fmt --all -- --check`: passed.
- Focused Qwen3 suite on integrated latest `main`: 110 passed, 0 failed, 20 ignored.
- `cargo clippy -p mlxcel --all-targets -- -D warnings`: passed.
- Correctness review, security/performance review, and finalization review found no unresolved blocker after the checkpoint-keyed warning fix.
- PR #1395 was integrated before the final gate. Its Turbo4/MLA change does not overlap Qwen3 RoPE, and the focused suite remained green.

### 4.2 Local NVIDIA real-model qualification

Hardware and runtime:

- NVIDIA GB10, driver 580.173.02, CUDA 13.0, compute capability 12.1.
- CUDA release build: `MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda --bin mlxcel`.
- Existing local checkpoints only: `models/mlx/qwen3-4b-4bit` and `models/mlx/qwen3-30b-a3b-4bit`.

Unscaled regression, temperature 0, seed 0, 128 generated tokens:

- Qwen3 4B: the PR and latest `main` generated the same 128-token text.
- Qwen3 30B-A3B: the PR and latest `main` generated the same 128-token text.

Scaled dense qualification:

- A hard-linked temporary view of the local 4B checkpoint changed only `config.json`, injecting `{"rope_type":"linear","factor":8.0}`; the original checkpoint was untouched.
- The non-degenerate prompt contained 3,009 tokenizer tokens, exceeding the required 2,048-token boundary.
- With the graph Q/K-normalization path selected to match upstream's graph RoPE evaluation, both mlxcel and mlx-lm 0.31.3 generated `The quick. The quick brown fox.` for the eight-token comparison.
- Latest `main`, which ignores the injected block, diverged from the scaled result. The PR's default fused decode produced the same prefix and then entered the issue's documented f16 reduction-order jitter class; the numerical fused-versus-graph rotation tests pass at the repository tolerance.
- The upstream CUDA oracle used mlx 0.32.1 with its CUDA 12.9 NVRTC and matching official CUDA 12.9 runtime headers. This repaired a CUDA-12-NVRTC/CUDA-13-system-header mismatch in the temporary oracle environment; it did not download or alter model weights.

### 4.3 Hosted validation boundary

The attempted one-off large-checkpoint Actions run was explicitly cancelled and is not counted as evidence. Standard repository CI remains responsible for formatting, lint, compilation, metadata, and small-fixture coverage; the GB10 runs above are the real 4B/30B qualification record.

## 5. Integration Notes

- The `rope_scaling` map remains intentionally map-shaped so configs carrying both `type` and `rope_type` parse normally; `type` wins as upstream specifies.
- Dense Qwen3 and Qwen3-MoE use the same shared resolution code, but only dense Qwen3 can be token-diffed against mlx-lm because upstream Qwen3-MoE still constructs an unscaled RoPE module.
- Batched, pipeline-parallel, and tensor-parallel tests assert that distributed construction does not reset the resolved scale to `1.0`.

## 6. Related Work

- Issue #1388: parsed-but-unused Qwen3 and Qwen3-MoE RoPE scaling.
- PR #1398: implementation, review corrections, and local qualification documented here.
- Issues #1340 and #1355: Gemma 3 and shared Llama RoPE scaling precedents.
- PR #1395: adjacent Turbo4/MLA correction integrated before final validation.
