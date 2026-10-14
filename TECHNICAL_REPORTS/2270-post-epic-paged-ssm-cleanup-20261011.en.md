# Technical Report: PR #2270 - Post-epic paged and SSM cleanup

**Date**: 2026-10-11
**Status**: Completed and verified on CUDA (GB10)
**Languages**: C++, Rust, Python, Bash, Markdown
**Risk Level**: Medium

## Executive summary

PR #2270 completes the three post-epic tasks in issue #2230. It removes the unused dense-pointer and rotating paged compatibility surface, replaces its role in pooled decode tests with an independent float32 attention oracle, resolves the Gemma 3 and Llama 4 model-owned storage flags through a complete paired/null serving sweep, and fixes CUDA bf16 SSM parity at the first numerically divergent stage.

The SSM defect was caused by the pinned MLX type-promotion table selecting bf16 for `float32 + bfloat16`. `compute_dt_compiled` cast `dt` to float32 but left `dt_bias` in bf16, so the compiled softplus result was rounded before any backend kernel received it. Casting both operands to float32 restores the documented recurrence for Metal, CUDA, and HIP without changing a tolerance. A deterministic Granite checkpoint comparison produced the same 64 generated token IDs through the fused CUDA kernel and graph paths.

The storage sweep ran 60 fresh-server arms and 180 successful client cells. None of the six prompt/concurrency cells for either family cleared the repeated-paged null spread in either direction, so both `supports_paged_decode_backend()` flags remain `true`.

## 1. Problem statement

### 1.1 Obsolete compatibility surface

After #2210 and #2217, production models no longer called the dense-pointer or rotating paged decode compatibility routes. Their C++ entry points, CXX bridge declarations, Rust wrappers and validators, cache metadata, and direct FFI tests remained only to support each other. Four active pooled-kernel tests still used the old dense fallback as their oracle, preventing removal until those tests had an independent reference.

### 1.2 Unresolved model-owned storage default

Gemma 3 and Llama 4 own their KV rows and compute attention from dense storage, while their advertised paged support still causes a multi-slot server to allocate paged storage and mirror each decode step into it. Earlier Gemma 3 numbers overlapped their run-to-run spread, and Llama 4 had no measurement. The flags therefore required a per-family decision based on paired dense-versus-paged deltas and a repeated-paged null arm.

### 1.3 CUDA bf16 SSM parity failure

The Granite bf16 SSM cases failed against the float32 graph reference with normalized state RMS around `1.8e-3` to `2.5e-3`, while f32 cases and the Nemotron-H bf16-activation/f32-`A_log` case passed. The same failure existed before the epic. The acceptance criteria required measuring `dt'`, decay, and state before changing production code and prohibited widening the existing tolerance without a derivation.

## 2. Implementation

### 2.1 Compatibility removal and independent oracle

The PR deletes the dense and rotating compatibility C++ implementations and declarations, their bridge entries, Rust fallback wrappers and validators, `PagedDecodeMetadata`, `RotatingPagedDecodeMetadata`, and tests that existed only for those APIs. The pooled empty-batch case remains under its current pooled behavior.

The four pooled decode tests now use a test-only `dense_reference_decode_attention` helper. It converts query, key, and value tensors to float32, repeats KV heads for grouped-query attention, computes `softmax(q k^T * scale) v`, and concatenates the reference rows. Each test preserves its original absolute RMS `<5e-3` assertion and adds normalized RMS `<5e-3`. A temporary `1.01` multiplication of the pooled output made all four tests fail near normalized RMS `0.010`; the mutation was removed before commit.

### 2.2 Model-owned storage measurement

The benchmark starts a fresh `mlxcel serve --parallel 8` process for every arm. It covers two models, 512- and 4096-token prompts, five rounds, and `paged-a`, `dense`, and `paged-b` arms in rotating order. Each arm measures concurrency 1, 4, and 8 for 128 generated tokens. The outer command holds `gpu-lock`; an inner fail-closed gate requires a sustained quiet host, a free port before and after the wait, no CI/build/foreign-model activity, sufficient memory, and clean NVIDIA driver counters.

The analyzer refuses to calculate a decision without the completion marker, clean checkout and artifact provenance, matching configured and built MLX commits, exact measured script hashes, the ordered 60-arm schedule, 61 accepted quiet-gate records, 60 nonempty backend-marked server logs, and exactly 180 successful client keys. `paged-a` is the paired baseline, while `paged-b / paged-a - 1` defines the null spread. Dense wins a cell only when its median paired delta exceeds `max(0, null_high)`; paged wins only below `min(0, null_low)`.

### 2.3 Shared SSM precision correction

The first large divergence appeared in compiled `dt'`, whose output dtype was bf16. The pinned MLX promotion table explained why: the graph added float32 `dt` to bf16 `dt_bias`, and the result followed the table's bf16 entry. The PR explicitly casts `dt_bias` to float32 beside `dt` before the compiled add, softplus, and clip.

`compute_dt_compiled` runs before backend dispatch. The same correction therefore supplies float32 `dt'` semantics to Metal, CUDA, and HIP without duplicating backend-specific logic. The CUDA recurrence kernel, float32 reference, and tolerances remain unchanged.

## 3. Measured results

### 3.1 SSM intermediate deviations

For Granite seed 2069, normalized RMS/max deviations against the stable float32 graph were:

| Stage | Before | After |
|---|---:|---:|
| Compiled `dt'` | `1.957140831e-3 / 5.875451502e-3` | `2.890811210e-8 / 2.002812756e-7` |
| Compiled decay | `3.008783834e-3 / 1.075881018e-2` | `6.809246822e-10 / 4.717584582e-9` |
| Kernel state | `2.532402801e-3 / 4.127457397e-2` | `2.838613609e-8 / 1.701009653e-6` |

The correction reduces the downstream deviation by roughly five orders of magnitude and brings both formerly failing CUDA tests inside their existing budgets.

### 3.2 Storage sweep

Rates below are aggregate generated tokens per second. The paired column shows the five-round median and full paired range; the null column is the repeated-paged range.

| Model | Prompt | C | paged-a | dense | paged-b | Dense paired delta | Null spread | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Gemma 3 | 512 | 1 | 154.4 | 154.0 | 154.0 | +0.52% (-2.76..+3.63) | -2.37..+3.84% | unresolved |
| Gemma 3 | 512 | 4 | 361.0 | 363.6 | 360.6 | +1.29% (-2.11..+2.49) | -1.35..+3.52% | unresolved |
| Gemma 3 | 512 | 8 | 194.9 | 197.2 | 194.4 | +1.16% (+0.25..+6.37) | -0.88..+3.80% | unresolved |
| Gemma 3 | 4096 | 1 | 114.6 | 114.4 | 113.0 | -0.18% (-1.82..+0.26) | -2.33..+1.31% | unresolved |
| Gemma 3 | 4096 | 4 | 219.4 | 216.6 | 218.6 | -0.60% (-2.74..+0.54) | -3.45..+0.50% | unresolved |
| Gemma 3 | 4096 | 8 | 184.4 | 185.8 | 184.0 | +0.49% (-0.53..+1.98) | -3.13..+1.91% | unresolved |
| Llama 4 | 512 | 1 | 11.7 | 11.8 | 11.8 | +0.85% (+0.00..+83.08) | -0.85..+81.54% | unresolved |
| Llama 4 | 512 | 4 | 31.0 | 30.8 | 30.7 | +0.65% (-3.53..+2.61) | -0.97..+3.53% | unresolved |
| Llama 4 | 512 | 8 | 17.8 | 17.9 | 17.7 | +0.00% (-1.69..+13.29) | -1.13..+10.76% | unresolved |
| Llama 4 | 4096 | 1 | 8.0 | 8.0 | 8.0 | +1.25% (+0.00..+1.27) | +0.00..+2.53% | unresolved |
| Llama 4 | 4096 | 4 | 22.9 | 23.1 | 23.0 | +0.44% (-0.43..+2.25) | -2.99..+4.05% | unresolved |
| Llama 4 | 4096 | 8 | 14.7 | 14.7 | 14.5 | +0.68% (-0.68..+3.55) | -2.03..+2.84% | unresolved |

No cell satisfies either clear-win rule. `auto` on a multi-slot server and an explicit `paged` request continue to select paged storage for both model families, including pool allocation and the per-step mirror. Their attention path continues to use dense rows.

### 3.3 Observed first-arm anomaly

The first Llama 4 512-token round began with `paged-a`. At concurrency 1, that arm recorded 6.5 aggregate tok/s and 12690.4 ms mean TTFT. The adjacent dense and repeated-paged arms recorded 11.9 and 11.8 aggregate tok/s with 4072.6 and 4128.2 ms TTFT. Rounds 2 through 5 were approximately 11.7 to 11.9 aggregate tok/s across all arms.

The first round therefore contributes a +83.08% dense paired delta and +81.54% null delta. Every record remains in the result. The host gate accepted the arm, but the evidence does not establish its cause; a cold or warm-up effect is only a hypothesis. The run was not filtered or repeated after inspection. Under the registered rule, the paired median is +0.85% and does not clear the +81.54% null upper bound.

## 4. Verification

| Area | Result |
|---|---|
| Removed compatibility expression across `src/` | No matches |
| Four focused pooled CUDA tests | Passed |
| Temporary pooled-output `1.01` mutation | All four tests failed; mutation removed |
| Focused CUDA SSM parity | Five passed, including both former failures |
| Granite real checkpoint | Fused kernel and graph produced identical 64/64 greedy token IDs |
| Granite token sequence SHA-256 | `9df6f4fc2abbfcb121fa5dca3ad1e140e8db5b67d28118a905b93022f8905829` |
| CUDA release build | Passed in 19m43s |
| Full `make verify-test-cuda` under GPU lock | Exit 0, zero failures |
| Workspace/all-target CUDA clippy and formatting | Passed |
| Dtype-key, port-dispatch, and llama-compat contracts | Passed |
| Storage analyzer | Validated 60 arms and 180 successful cells |
| Benchmark protocol fixtures | 9 passed with warnings treated as errors |
| Raw storage and SSM manifests | Passed |
| Independent implementation, security, and evidence reviews | No remaining findings |

The Granite comparison used prompt `Explain the Mamba state update.`, 64 greedy tokens, temperature 0, and deterministic SDPA. The graph arm set `MLXCEL_SSM_CUDA_KERNEL=0`.

## 5. Provenance and limitations

The storage run's contemporaneous `environment.txt` records clean benchmark checkout `35da2dbe`. The measured release binary was built from production commit `2cb92404`; the binary does not embed a commit, so this association is reconstructed from the preserved build-session context, artifact hash and mtime, ancestry, and the exact intervening path set. `2cb92404` is an ancestor of `35da2dbe`, and all intervening changes are confined to the four benchmark protocol files. No Rust, C++, model, or build-definition source differs. The original environment and measured script bytes remain unchanged, while a separate derived provenance record documents the proof and its limitation.

The committed release log proves that the release build completed in 19m43s but does not print HEAD. The raw package retains all client/server output, the 61 accepted host-gate records, ordered schedule, analyzer result, completion marker, script hashes, MLX pin, derived provenance, SSM stage output, Granite comparison, and checksums.

## 6. Compatibility, security, and performance

The removed APIs had no production callers. The SSM fix changes numeric preprocessing only where a bf16 bias previously caused unintended rounding and adds no per-token host synchronization or new dispatch. No new runtime dependency or public option is introduced.

Both model-owned flags remain enabled, so serving behavior is stable. The benchmark provides no clear throughput winner within its observed repeatability. Independent security review found no production-code issue. Benchmark review initially found fail-open host/provenance handling; the final gate and analyzer reject nonquiet, incomplete, reordered, dirty, or backend-substituted evidence before calculation.

## 7. Related records

- Issue #2230: post-epic cleanup requirements
- PR #2270: implementation and evidence
- `docs/benchmark_results/model-owned-storage-gb10-2026-10-11.md`: dated storage decision
- `docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11/`: complete storage protocol and raw output
- `docs/benchmark_results/data/ssm-bf16-parity-gb10-2026-10-11/`: original stage deviations and Granite parity artifacts
