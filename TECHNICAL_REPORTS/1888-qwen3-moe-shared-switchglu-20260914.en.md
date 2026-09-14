# Technical Report: PR #1888 - refactor(moe): migrate qwen3_moe onto the shared SwitchGLU

**Date**: 2026-09-14
**Status**: Completed; merge pending
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

`qwen3_moe` kept a private copy of `SwitchLinear`, `SwitchGLU` and `forward_fused_kernel`, and `qwen3_vl_moe` (which also builds the Qwen3-Omni-MoE thinker) built that copy by hand. The copy never received the fused-kernel `dff` upper bound that the shared `switch_layers::SwitchGLU` gained in #311 and #643, so `MLXCEL_FUSED_MOE_MAX_DFF` was silently ignored for both families. PR #1888 deletes the copy and puts both families on the shared type. On GB10 the greedy token ids of `qwen3-30b-a3b-4bit` and `qwen3-vl-30b-a3b-4bit` are identical before and after, on the fused path and with `MLXCEL_FUSED_MOE=0`; the decode trace still shows `path=fused tokens=1`; and a cap of 512 now moves both families to `gather_qmm`, where the base binary ignored it.

## 1. Problem Statement

### 1.1 Background

The fused single-token decode kernel (#268), the shared `forward_fused_kernel` and the Qwen3-MoE copy landed together in `7fd4dcac` (#275). The shared function then gained the `dff` bound in `d846fd4f` (#311), made backend-aware in `37931d7f` (#643): 4096 on Metal, 8192 on CUDA, overridable by `MLXCEL_FUSED_MOE_MAX_DFF`. Above the bound `gather_qmm` already saturates the GPU and the two-kernel fused path is a measured net loss (phi-3.5-moe at Dff 6400 regresses on M1 Ultra).

### 1.2 Existing Issues

- **Missing bound**: the copy's `forward_fused_kernel` had no `dff` check, so a Qwen3-MoE or Qwen3-VL-MoE variant with experts wider than the bound would take the fused path in the loss regime, and the env override did nothing for either family.
- **Wrong documentation**: `switch_layers.rs` said the cap governed Qwen3-MoE and listed Qwen3Moe as a user of the shared type.
- **Guards that could not fire**: the copy had no `mode` field (its `gather_qmm` call hardcoded `"affine"` and its loader required `.biases`) and no activation check. Neither mattered, because the copy cannot load a non-affine checkpoint at all.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A wide-expert Qwen3-MoE variant decodes slower on the fused path | Medium | Low (no local checkpoint is above the bound) |
| The copy drifts again the next time the shared guard set changes | Medium | High (it already happened once) |
| The migration silently drops the fused path on existing checkpoints | High | Low, ruled out by the trace in Appendix B |

## 2. Technical Review

### 2.1 Equivalence on existing checkpoints

Both implementations run the same ops: the same `do_sort` threshold of 64 routed slots, the same `gather_sort` and `scatter_unsort`, `compiled_swiglu_activation`, and the same `gather_qmm` arguments (`transpose=true`, mode `"affine"`). Both call `fused_moe_expert_kernel` with the same 18 arguments in the same order. The shared type adds four declines (activation, mode, missing biases, `dff`), none of which fires on an affine 4-bit plane of Dff 768 or 2560.

Two loader behaviors change:

- **Stored bit width**: the shared loader stores bits inferred from the tensor shapes (`packed_in * 32 / (num_groups * group_size)`), where the copy stored the configured value. Every expert plane of the four local checkpoints was read from the safetensors headers: 144 planes each for `qwen3-30b-a3b-4bit`, `qwen3-moe-4bit` and `qwen3-vl-30b-a3b-4bit`, and 186 for `qwen3-coder-480b-a35b-instruct-4bit`. All solve to 4 bits with `.biases` shaped like `.scales`, so the stored triple is identical.
- **Missing biases**: an affine plane without `.biases` now fails with the `validate_quantization_biases` message instead of `Weight not found: ...biases`. No local checkpoint lacks biases.

### 2.2 Security

The load-time protection the copy had (`validate_expert_quantization_params` on the declared pair) is kept by `SwitchLinear::from_stacked_parts`, which additionally cross-checks the packing against the tensor shapes, the biases shape against the scales, and the mode against the biases. Every added refusal rejects only a load that would otherwise have aborted inside `gather_qmm`, whose C++ throw crosses the cxx bridge as an uncatchable abort.

### 2.3 Performance

Decode tok/s on `qwen3-30b-a3b-4bit` (`mlxcel-bench-decode -n 128 --ignore-eos --warmup-tokens 20`), 15 runs per arm, base and branch interleaved with the arm order alternating per round:

| Arm | n | min | median | max | mean |
|-----|---|-----|--------|-----|------|
| base `b8d10fb1` | 15 | 66.21 | 78.00 | 87.76 | 77.70 |
| branch | 15 | 61.27 | 73.63 | 86.28 | 73.85 |

The ranges overlap almost entirely. Single-run decode on GB10 is bimodal and the split between modes differs (base 11 high and 4 low, branch 9 high and 6 low); inside the high mode the means are 81.06 and 79.65. A Mann-Whitney test gives p = 0.21, and the paired per-round difference is -3.85 tok/s with a standard deviation of 10.21 (t = -1.46). No change is distinguishable. The only added per-call work is the shared bound's `std::env::var` lookup and `metal_is_available()` query, 48 calls per token, on the order of 10 microseconds against a 13 millisecond token.

### 2.4 Compatibility

- **Breaking changes**: `qwen3_moe::SwitchLinear` and `qwen3_moe::SwitchGLU` are no longer exported. The only in-tree users were `qwen3_vl_moe` and the family tests, both migrated.
- **Newly loadable layout**: the shared loader falls back to stacking unstacked `experts.{idx}` tensors, so a raw Hugging Face Qwen3-MoE checkpoint that failed with `Weight not found` now reaches that fallback, as Qwen2-MoE already did.
- **New dependencies**: none.

## 3. Technical Decisions

### 3.1 Migrate the type rather than patch the copy

| Option | Pros | Cons |
|--------|------|------|
| Add the `dff` check to the copy | One-line fix | Keeps the duplicate that drifted in the first place |
| **Chosen: move both families onto `switch_layers::SwitchGLU`** | One guard set for every shared-kernel family; removes about 330 lines of model code | The two loader behavior changes in 2.1 |

`SwitchGLU::from_weights` (affine) is used rather than `from_weights_with_mode`: non-affine Qwen3-MoE loading is out of scope, and the shared guard covers it whenever it arrives.

### 3.2 Guard tests drive the family block loader

The #958 guard tests previously drove the copy's loader. With no family loader left they drive each family's `SparseMoeBlock::from_weights`, which pins both that the shared loader carries the bound and that the family hands it the pair from its own config type (`ModelArgs`, or `Qwen3VLMoeConfig`, whose accessors fall back to 0).

## 4. Implementation Details

- `src/models/qwen3_moe.rs`: the local types, their loaders and the `validate_expert_quantization_params` import are removed; `SparseMoeBlock::experts` is the shared `SwitchGLU`, built by `SwitchGLU::from_weights(weights, "{prefix}.switch_mlp", group_size, bits)`. The `forward`, `forward_profiled` and batched call sites are unchanged.
- `src/models/qwen3_vl_moe.rs`: `load_switch_glu`, `load_switch_linear` and the now-unused `get_weight_copy` are removed; the block loader calls the shared constructor.
- `src/models/switch_layers.rs`: a test helper `insert_honest_affine_swiglu_experts` quantizes real affine 4-bit SwiGLU planes; a new test pins the mxfp4 decline; the docs name the shared fused-kernel callers and the paths that still bypass the bound, correct the family counts, and qualify the greedy-parity claim.
- Docs: `docs/environment-variables.md` and `docs/benchmark_results/fused-moe-decode-kernel-design.md` record that the two families read the bound.

## 5. Learning Points

### 5.1 An id comparison cannot prove a dispatch

Byte-identical greedy ids would also pass if the migration had silently fallen back to `gather_qmm`, whenever the two paths agree on the prompt. Two independent signals close that gap. The `MLXCEL_PROFILE_QWEN3_MOE_DETAIL` trace names the path per call and shows `path=fused tokens=1` for all 144 decode MoE calls on both binaries. A cap below the checkpoint's Dff (`MLXCEL_FUSED_MOE_MAX_DFF=512`) then separates the two implementations directly: the base trace stays at 144 fused calls, the branch trace shows 144 `gather_qmm` calls.

`qwen3_vl_moe` has no trace, so the per-process first-launch cost of the fused CUDA kernel (about 1.3 s on the first MoE call) serves as the signature. Default runs take 2.26 to 2.55 s for 64 tokens on both binaries and `MLXCEL_FUSED_MOE=0` runs take 1.27 to 1.34 s; with the 512 cap the base stays at 2.31 to 2.35 s while the branch drops to 1.28 to 1.32 s.

### 5.2 Fused and gather greedy parity is prompt-dependent

The shared docs said 64 greedy tokens agree with the kernel on and off on `qwen3-30b-a3b`. On the prompt used here the two paths agree for 39 generated tokens and then diverge, deterministically and identically on both binaries. That also made the fused-id comparison discriminating on this checkpoint: a silent fallback would have diverged at token 39.

### 5.3 Mutation-check a decline test

A decline test that asserts `None` passes for any reason the function declines. The Dff 64 positive control on the same geometry rules out the other guards, and a temporary mutant that disabled only the `dff > max_dff` branch made both new decline tests fail, which confirms they pin the bound itself.

## 6. Further Learning

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `gather_qmm` | MLX gathered quantized matmul over stacked experts | The fallback and prefill path |
| `fused_moe_expert_kernel` | Two-launch custom kernel for single-token SwiGLU experts | The path whose guard set lacked the bound |
| `MLXCEL_FUSED_MOE_MAX_DFF` | Expert-width cap for the fused path | Now read by both families |

Related: #268, #275, #311, #643, #958, #1045, #1803, #1859.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 7 (plus this report) |
| Lines | +524 / -432 |
| Tests added | 3 (Dff decline through the qwen3_moe loader, Dff decline through the qwen3_vl_moe loader, mxfp4 decline) |
| Tests retargeted | 2 (#958 guards for both families) |

| Hash | Type | Message |
|------|------|---------|
| `44e58eeb` | refactor | move qwen3_moe experts onto the shared SwitchGLU |
| `a78d39e9` | test | pin the dff decline through the Qwen3-VL-MoE loader too |
| `b7431338` | docs | state which families read MLXCEL_FUSED_MOE_MAX_DFF |
| `92e3faf9` | docs | qualify the qwen3-30b-a3b fused greedy parity claim |
| `c314b253` | docs | add technical report for PR #1888 |
| `0c80472a` | docs | name the Qwen3-Omni thinker and fix issue references |

## 8. Follow-up Actions

### Future Improvements

- `qwen3_next.rs` (Qwen3Next, Qwen3.5, the qwen3_omni_moe talker) carries the same SwiGLU kernel copy with the same missing bound, mode and activation guards.
- `gemma4.rs` drives the GeGLU kernel with the mode check but no bound; consolidating it needs a GeGLU `SwitchGluActivation` first.
- NemotronH's opt-in `MLXCEL_FUSED_MOE_RELU2` path launches the down kernel without the bound.
- Ten families keep a local `SwitchGLU` with no fused path: deepseek, deepseek_v2, deepseek_v3, deepseek_v32, ernie4_5_moe, exaone_moe, glm4_moe, glm4_moe_lite, hunyuan_moe, llama4.
- The shared bound reads the environment and queries the backend on every call; caching it in a `OnceLock`, as `fused_moe_enabled` does, would remove that work.
- `deepseek_v4_moe` calls `validate_expert_quantization_params` but is missing from its Used-by list.
- Security review, pre-existing in the shared loader: the three expert planes are never cross-checked against each other, and the router's row count is never compared with the stacked expert count, so a checkpoint whose planes disagree can reach an unchecked kernel index. The unstacked `experts.{idx}` fallback also stacks without a declared expert count and calls `stack` on tensors whose shapes are not compared first. Both belong in the shared `SwitchGLU` loader so every family gains them.
- The `MLXCEL_FUSED_MOE` family list in `docs/benchmark_results/fused-moe-decode-kernel-design.md` ("eleven model paths") is stale.
- The `OpenXLA feature compile` CI job fails on `main` at `b8d10fb1` with unused-import errors in `src/models/mod.rs` and the server modules under `--no-default-features`; it fails the same way on this PR and is unrelated to it.

## Appendix

### A. Test Results

All 38 tests under `models::qwen3_moe`, `models::qwen3_vl_moe` and `models::switch_layers` pass on GB10 (`test-fast` profile, `--features cuda`), each run in its own process, at `92e3faf9` and again at the final `0c80472a`. The Dff decline positive controls ran (no skip message). `cargo fmt --check` and `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests --no-deps -- -D warnings` are clean. Not run: the `metal,accelerate` gate (not runnable on Linux) and the workspace-wide `verify-test-cuda`.

### B. Verification on GB10 (sm_121, CUDA, release build)

| Check | Base `b8d10fb1` | Branch |
|-------|-----------------|--------|
| `qwen3-30b-a3b-4bit` ids, fused (64 tokens, `-t 0`) | reference | identical |
| `qwen3-30b-a3b-4bit` ids, `MLXCEL_FUSED_MOE=0` | reference | identical |
| `qwen3-vl-30b-a3b-4bit` ids, fused | reference | identical |
| `qwen3-vl-30b-a3b-4bit` ids, `MLXCEL_FUSED_MOE=0` | reference | identical |
| Decode trace, `-n 4` | 144 `path=fused tokens=1`, 0 `gather_qmm` | 144 `path=fused tokens=1`, 0 `gather_qmm` |
| Decode trace, `MLXCEL_FUSED_MOE_MAX_DFF=512` | 144 fused (bound ignored) | 144 `path=gather_qmm tokens=1` |
| `mlxcel-server` `qwen3-30b-a3b-4bit`, temperature 0, 64 tokens | reference | identical content and reasoning |
| `mlxcel-server` `qwen3-vl-30b-a3b-4bit`, temperature 0, 64 tokens | reference | identical, coherent, same opening tokens as the CLI |

The branch column was measured with binaries built at `a78d39e9`; the later commits change comments and docs only. The ids, trace, cap and server checks were repeated with binaries rebuilt at the final `0c80472a` and matched the base again. The same final binaries added the Qwen3-Omni-MoE thinker (`qwen3-omni-30b-a3b-instruct-4bit`, text-only): ids identical to the base with the fused path and with `MLXCEL_FUSED_MOE=0`, where the two paths diverge from each other at token 26; with `MLXCEL_FUSED_MOE_MAX_DFF=512` the base ids equal its fused ids and the branch ids equal the `MLXCEL_FUSED_MOE=0` ids.

`qwen3-coder-480b-a35b-instruct-4bit` was checked from its headers only. The kernel driver reported 0 `NV_ERR_NO_MEMORY` events throughout.
