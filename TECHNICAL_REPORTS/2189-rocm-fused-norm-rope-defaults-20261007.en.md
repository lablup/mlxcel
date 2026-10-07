# Technical Report: PR #2189 - Enable Fused Add-RMSNorm and RoPE-Append by Default on ROCm

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); head `9874c3c4` (rebased onto origin/main `fc21784f`), PR open, pending merge. Closes #2145 (part of #1801).

**Languages**: Rust (mlxcel-core `layers.rs`; mlxcel `models/llama3.rs`, `models/llama3_tests.rs`), Python (`scripts/rocm_decode_profile.py`, `tests/test_rocm_decode_profile.py`), Markdown (README, `docs/environment-variables.md`, `docs/installation.md`, `docs/benchmarks.md`, two results pages), CSV (two benchmark files)

**Risk Level**: Low. The two HIP ports this PR turns on are byte-identical to the ROCm graph they replace, so the default change cannot move a logit on the models that were traced. Metal and CUDA builds compile the same `false` constants as before. The residual risk is in model families that now take the norm port by default on ROCm without a local checkpoint (Gemma, IQuest Loop Coder).

## Executive Summary

PR #2107 (closing #2063) ported `fused_add_rms_norm` and `fused_rope_qk_append` to HIP and left both opt-in, because on gfx1151 the decode gains (Llama 3.1 8B +0.3%, Qwen2.5 7B up to +1.7%) sat inside run-to-run drift. The maintainer then decided in #2145 to turn both on by default on ROCm only. The reasoning is asymmetric: the ports cost no accuracy (every on/off logit-trace pair was byte-identical) and no median was slower, so there is nothing to lose by defaulting them on, even though the gain cannot be distinguished from noise. This PR is the implementation of that decision, not a performance claim.

The change is two lines of logic: `FUSED_ADD_RMSNORM_DEFAULT` and `FUSED_ROPE_APPEND_DEFAULT` in `src/lib/mlxcel-core/src/layers.rs` become `cfg!(feature = "rocm")`. The environment variables keep overriding in both directions. One side effect needed a real fix: the Llama 3.1 `rope_scaling` bypass notice asked the gate whether the RoPE kernel was requested, so with the default on it would have printed on every Llama 3.1 run on ROCm. A new `layers::fused_flag_explicit_value` lets the notice fire only for an explicit truthy `MLXCEL_FUSED_ROPE_APPEND`.

The default-on build was re-measured on gfx1151: Llama 3.1 -0.2% and Qwen2.5 +0.8% against `=0`, both noise. `w8` logit traces are byte-identical between the default and both variables at `0`, and `rocprofv3` shows the default path dispatching the ported kernels. The unit's `make verify-rocm` at `9874c3c4` printed OK with 11,996 passed, 0 failed and 383 ignored; the orchestrator re-ran the full gate on the same head before merge.

## 1. Problem Statement

### 1.1 The state after #2107

`FUSED_ADD_RMSNORM_DEFAULT` and `FUSED_ROPE_APPEND_DEFAULT` were plain `false` for every backend. Their doc comments recorded why: on Metal (M1 Ultra) the op-level microbench and the decode measurement did not justify the fusions, and the RoPE fusion had one reproducible below-parity cell (hidden 8192, batch 1, 0.89x to 0.94x). The #2063 ROCm measurement was appended to those comments with the conclusion "not a clear win, so ROCm keeps the shared default".

The ROCm numbers from #2107:

| Model | Arm | Off | On | Change |
|---|---|---|---|---|
| Llama-3.1-8B | norm only (RoPE bypassed by `rope_scaling`) | 37.85 | 37.95 | +0.3% |
| Qwen2.5-7B | norm / RoPE / both | 46.53 | 47.23 / 46.80 / 47.32 | +1.5% / +0.6% / +1.7% |
| Qwen3-30B-A3B | control | | | +0.3% |

On beat off in 6 of 7 Qwen2.5 rounds, but the size of the gain drifted between runs as much as the off arm's spread.

### 1.2 The decision

The maintainer's call in #2145 rests on correctness, not speed. Both ports produce output byte-identical to the graph on ROCm (every pair at `w1`, `w8` and `w1ctx512`, and 0/230 and 0/274 decided mismatches against Metal). A default that cannot change a token and was never slower in a median has no downside on that backend, and running the ported kernels by default also means they get exercised by every ROCm user instead of only by people who know the variables exist. On Metal the picture is different (the RoPE fusion has a measured regression cell, and the Metal norm kernel differs from the graph by a rounding step), so Metal and CUDA stay off.

## 2. Change Summary

| Area | Change |
|---|---|
| `mlxcel-core/src/layers.rs` | Both `FUSED_*_DEFAULT` constants become `cfg!(feature = "rocm")`; doc comments rewritten per backend with the #2107 numbers and the decision. `fused_flag_enabled_from` now delegates to the new public `fused_flag_explicit_value(value) -> Option<bool>`. New tests `fused_905_defaults_are_on_for_rocm_builds_only` and `fused_flag_explicit_value_separates_a_setting_from_the_default` |
| `src/models/llama3.rs` | `fused_rope_launcher_requested` uses the new `fused_rope_append_explicitly_requested(value)` for `MLXCEL_FUSED_ROPE_APPEND` instead of the gate; notice and launcher-order doc comments updated |
| `src/models/llama3_tests.rs` | `the_rope_append_bypass_notice_needs_an_explicit_truthy_value` |
| `scripts/rocm_decode_profile.py`, `tests/test_rocm_decode_profile.py` | The #2063 roles count as reached with the shipped defaults; the test asserts the default tuple instead of `()` |
| Docs | README ROCm bullet; `environment-variables.md` Default column "off (Metal, CUDA) / on (ROCm)"; ROCm row in `installation.md`; `benchmarks.md`; the note in `rocm-correctness-gfx1151-2026-09-30.md`; a Decision section with the re-measurement in `rocm-fused-norm-rope-gfx1151-2026-10-05.md` |
| `benchmarks/` | `rocm_strixhalo-gfx1151_2026-10-07_fused-default-{on,off}.csv` |

Three commits: the default flip with the notice fix and tests (`c1309a0e`), the re-measurement (`3f5783c6`), and README, docs and decode-profile alignment (`9874c3c4`). 13 files, 197 insertions and 65 deletions.

## 3. Design

### 3.1 `cfg!(feature = "rocm")` on the constants

```rust
pub(crate) const FUSED_ADD_RMSNORM_DEFAULT: bool = cfg!(feature = "rocm");
pub(crate) const FUSED_ROPE_APPEND_DEFAULT: bool = cfg!(feature = "rocm");
```

The `rocm` feature here is mlxcel-core's own (`src/lib/mlxcel-core/Cargo.toml`), which the root `rocm` feature enables. A ROCm build compiles no Metal or CUDA backend, so the build flag already identifies the backend. This follows PR #2098, which set the ROCm `MLXCEL_FUSED_MOE_SGY` default to 2 under `#ifdef MLXCEL_BRIDGE_ROCM_BACKEND` on the C++ side for the same reason.

The alternatives the issue ruled out:

- **A runtime backend comparison** (`gpu_kernel_backend() == Rocm`) or a `metal_is_available() || cuda_is_available()` gate. `scripts/ci/check_kernel_port_dispatch.py` (the `verify-kernel-port-dispatch` gate) rejects those patterns in kernel-port dispatch code, and they would add a runtime branch to answer a question the compiler already knows the answer to.
- **A separate ROCm constant or a ROCm-only gate function.** That would duplicate the `OnceLock` gates and the parser and give the two backends a chance to drift. With `cfg!` there is one constant per fusion and one place to flip it, which keeps the "flip here" contract the doc comments describe.

`cfg!` rather than `#[cfg]` items keeps both values type-checked on every build and lets the tests compare against `cfg!(feature = "rocm")` directly, so each backend's test run checks its own default.

### 3.2 Env override in both directions

`fused_flag_enabled_from` keeps its contract. `0`/`false`/`off`/`no` disable and `1`/`true`/`on`/`yes` enable on every backend (case-insensitive, trimmed); unset or unrecognised takes the compiled-in default, so a typo does not silently change the decode graph. On ROCm, `MLXCEL_FUSED_ADD_RMSNORM=0 MLXCEL_FUSED_ROPE_APPEND=0` restores the graph; on Metal and CUDA, `=1` still opts in. The gates still read each variable once into a `OnceLock`, so the setting must be in place before the first decode.

The refactor splits the parser: `fused_flag_explicit_value(value)` returns `Some(false)`, `Some(true)` or `None`, and `fused_flag_enabled_from` is now `fused_flag_explicit_value(value).unwrap_or(default)`. Behavior is identical; the split exists so a caller can ask "did the user set this?" separately from "is it on?".

### 3.3 The Llama 3.1 bypass notice

Llama 3.1's `rope_scaling` builds a frequency table, which the fused RoPE kernel cannot take, so `Attention::forward` routes around it regardless of the flag. `report_fused_rope_bypass_once` prints a one-time stderr notice when a bypass reason holds and a variable asked for a fused RoPE launcher. Its doc comment promised that a user who never sets the variables never sees it.

Before this PR, `fused_rope_launcher_requested` answered the `MLXCEL_FUSED_ROPE_APPEND` arm by asking the gate (`fused_rope_append_enabled()`), which was correct while the default was off: the gate was true only if the user had opted in, and asking the gate instead of checking presence kept `=0` from being reported as a lost kernel. With the default on in a ROCm build, the gate is true for every unset run, so every Llama 3.1 run on ROCm would have printed the notice.

The fix asks a different question. `fused_rope_append_explicitly_requested(value)` is `fused_flag_explicit_value(value) == Some(true)`, so the notice fires only for an explicit truthy value. Unset, unrecognised and `0` stay silent on every backend. The other two variables in the list (`MLXCEL_FUSED_CAUSAL_PREFILL`, `MLXCEL_ENABLE_FUSED_QKV_SPLIT_ROPE`) are presence-enabled and unchanged.

One consequence to note: an unrecognised value such as `MLXCEL_FUSED_ROPE_APPEND=maybe` on ROCm takes the default (on) but does not trigger the notice. That matches the intent, since the user did not clearly ask for the kernel.

### 3.4 Tests

- `fused_905_defaults_are_on_for_rocm_builds_only` asserts both constants equal `cfg!(feature = "rocm")`, then under `#[cfg(feature = "rocm")]` checks unset is on and `0` is off, and under `#[cfg(not(...))]` checks unset is off and `1` is on. Each backend's test run therefore pins its own default and the override in the direction that matters there.
- `fused_flag_explicit_value_separates_a_setting_from_the_default` covers `None`, unrecognised values (`""`, `maybe`, `2`), both spellings of each recognised set, and whitespace and case.
- `the_rope_append_bypass_notice_needs_an_explicit_truthy_value` covers the unset case every default ROCm Llama 3.1 run hits, plus the falsy and truthy sets.
- The existing `fused_905_flags_follow_the_default_and_respect_explicit_values` and the parity tests already read the constant rather than a literal, so they needed no change.

### 3.5 The decode profile script

`scripts/rocm_decode_profile.py` classifies which decode roles each ROCm kernel-port unit reaches. For #2063 it returned `()` for the default and listed the roles only as opt-in. It now returns the same tuple for both, `add_rms_join_post_attn` plus `rope_append` when the checkpoint has no `rope_scaling` table, with a note pointing at the `cfg!` constants and #2145.

## 4. Re-measurement

gfx1151, `scripts/bench_decode.sh` at pp512/tg128, default (no variables) against `MLXCEL_FUSED_ADD_RMSNORM=0 MLXCEL_FUSED_ROPE_APPEND=0`, arm order alternated per round, medians of 3. Every run went through `scripts/rocm_gpu_guard.sh` with the host-wide lock from #2146; 12 of 12 runs were clean on the first attempt. The measurement ran on the pre-rebase SHA of the first commit (`57b59629`).

| Model | Default (on) | `=0` (off) | Change | Paired (on minus off), by round |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct-4bit | 38.09 | 38.15 | -0.2% | +0.02 / -0.49 / -0.05 |
| Qwen2.5-7B-Instruct-4bit | 47.90 | 47.51 | +0.8% | +0.39 / +0.30 / -0.04 |

Both changes are inside run-to-run noise. Llama 3.1 runs only the norm port, and its -0.2% comes from one off run at 38.58, above every other run of either arm. Qwen2.5 runs both ports and is slightly above off in two of three rounds. Neither number is a speedup claim, and neither argues against the decision, which never depended on a speedup.

Qwen2.5 prefill was 2.7% to 6.9% higher with the default in every round (median 1609.96 against 1544.08). The #2107 run did not show this, and three rounds cannot separate it from drift, so the results page records it without drawing a conclusion.

## 5. Verification

All on gfx1151:

- **Byte identity.** `logit_trace` `w8` for both models, default against both variables at `0`: the files are byte-identical (`cmp`).
- **The default path reaches the kernels.** `rocprofv3 --kernel-trace` on an 8-token `mlxcel generate` with no variables set: Qwen2.5 dispatched `fused_add_rms_norm` and `fused_rope_qk_append` 252 times each (28 layers x 9 forward passes) and no graph RoPE kernel; with both variables at `0` it dispatched `rope_single_1d` instead. Llama 3.1 dispatched the norm port 288 times (32 x 9) and kept `rope_single_freqs_1d` / `rope_freqs`. This is what proves the flip took effect in the binary, not only in the constant.
- **Bypass notice.** Absent with the default on Llama 3.1, present with `MLXCEL_FUSED_ROPE_APPEND=1`, absent with `=0`.
- **Targeted tests.** `layers::tests::fused`, `fused_norm_parity_tests` and `fused_rope_parity_tests` under `--features rocm`: 32 passed. `models::llama3::`: 27 passed, including `greedy_decode_is_token_identical_to_the_unfused_baseline`, which also passes with both variables at `0`. `tests.test_rocm_decode_profile`: 26 passed.
- **Script gates.** `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: pass. `verify-kernel-port-dispatch` passing is the check that no runtime backend comparison was introduced.
- **Unit's `make verify-rocm`** at `9874c3c4` (rebased onto `fc21784f`): OK. Clippy and the ROCm smoke test passed; 11,996 passed, 0 failed, 383 ignored across 152 test binaries.
- **Orchestrator's `make verify-rocm`**: re-run on the same head before merge.

Not verified: Metal and CUDA, which are not available on this host. On those builds both constants evaluate to `false` exactly as before, and the notice change only narrows when it prints.

## 6. Technical Decisions

- **Default on for ROCm on correctness grounds, not speed.** A byte-identical port that is never slower in a median has no downside; the decision is explicitly not a claim that the fusion is faster.
- **Keep Metal and CUDA off.** The Metal measurements included a reproducible regression cell for the RoPE fusion, and the Metal norm kernel can differ from its graph by the rounding of the residual sum. Nothing in #2107 or this PR changes that evidence.
- **The build feature is the backend.** `cfg!(feature = "rocm")` follows PR #2098's precedent, adds no runtime branch, and keeps `verify-kernel-port-dispatch` passing.
- **One constant per fusion, one parser.** No ROCm-only duplicate of the gate, so the two backends cannot drift apart in parsing or caching.
- **Separate "explicitly set" from "enabled".** `fused_flag_explicit_value` exists because the bypass notice needs the first question and the gate answers the second. Once the default differs by backend, they are no longer the same.
- **Re-measure the shipped default, not the opt-in.** The new run compares the default against `=0`, which is the comparison a ROCm user now faces, and records it next to the original #2107 numbers.

## 7. Residual Risks and Follow-ups

- **Unverified model families.** Gemma and IQuest Loop Coder also call `fused_add_rms_norm` (`src/models/gemma.rs`, `src/models/iquestloopcoder.rs`) and now take the norm port by default on ROCm, but no checkpoint for either is available on this host. Gemma's `(1 + w)` weight convention is covered only by the tolerance tests in `fused_norm_parity_tests`, not by a logit trace or a generation run. VLMs whose text backbone is `Llama3Model` or the Qwen2 path also inherit the defaults; they share the traced Llama/Qwen2.5 code path but were not run individually. A trace on a Gemma checkpoint is the obvious follow-up.
- **Wave64 (CDNA) is untested.** All measurements and traces are on RDNA 3.5 gfx1151 (wave32). The default now applies to any ROCm target the build supports.
- **Qwen2.5 prefill shift.** The 2.7% to 6.9% prefill increase is unexplained and may be drift; a longer run would settle it.
- **The gain is noise-level.** If a later ROCm change makes the graph path faster than the ports, the default should be revisited. `MLXCEL_FUSED_ADD_RMSNORM=0` / `MLXCEL_FUSED_ROPE_APPEND=0` remain the kill switches, and the constants remain the one place to flip.

## 8. Learning Points

- **A byte-identical port changes the default question.** When a replacement kernel cannot change a single logit, "is it clearly faster?" is the wrong bar; "is it ever slower?" is enough, and the default can follow.
- **A notice tied to a gate changes meaning when the default changes.** Asking the gate was right while the gate meant "the user opted in". Flipping the default silently changed what the gate meant, and anything that inferred user intent from it had to switch to reading the explicit value.
- **Prove the default path in the binary.** Unit tests pin the constant, but the `rocprofv3` dispatch counts (252 and 288) are what show a default build actually launches the ported kernels and that Llama 3.1 still keeps the graph RoPE.
- **Let the compiler answer build-time questions.** When the build flag already determines the backend, `cfg!` is simpler and safer than a runtime backend check, and the repository's dispatch gate enforces that pattern.
