# Technical Report: PR #1990, Align Metal runtime controls with the built backend

**Date**: 2026-09-27
**Status**: Implemented and locally validated; pending merge.
**Languages**: Rust, C++, Markdown
**Risk Level**: Medium

## Executive Summary

This change fixes automatic MTP exactness recovery in plain macOS release builds. MLX's CMake configuration enabled Metal by default, but the C++ bridge exposed the live QMV and command-buffer controls only when Cargo's optional `metal` feature was selected. A plain `cargo build --release` therefore ran on Metal while its Rust-visible QMV setter was an inert stub. The exactness gate reported that it had retried without `qmv_wide`, but it had measured the wide kernel twice.

The build now resolves the Metal backend once and uses the same decision for CMake and the C++ bridge. Failed probes retain both the initial and narrow-retry verdicts. A successful narrow retry keeps that mode active; a failed or unrunnable retry restores the previous wide mode.

Real M5 Max checks with the plain release binary passed for Gemma 4 31B QAT and Qwen 3.8 MTP. Automatic retry and an independently started `MLXCEL_QMV_WIDE=0` process produced identical outputs and acceptance statistics for each checkpoint.

## 1. Problem Statement

The automatic retry from issue #1187 disables `qmv_wide` when block verification differs from the single-token chain. The Gemma 4 work in PR #1987 exercised that retry on another geometry. In the original M5 Ultra report, the initial Gemma probe differed in 240,460 of 524,288 logit bytes at layer 0 sliding attention. The retry logged that disabling the wide kernel did not help, while a fresh process started with `MLXCEL_QMV_WIDE=0` passed.

The discrepancy came from two build-time decisions. CMake enabled Metal for every macOS build by default. The bridge compiled its real runtime-control calls only under Cargo's optional `metal` feature. The plain release server therefore had an active Metal backend and stubbed controls.

The old diagnostic also reduced a failed retry to a boolean. It discarded the retry's divergence location and byte counts, which obscured whether the second arm ran and how it differed.

## 2. Technical Review

### 2.1 Backend selection

`mlxcel-core/build.rs` now resolves Metal availability from the native build host and `MLXCEL_BUILD_METAL`. The result configures both `MLX_BUILD_METAL` and `MLXCEL_BRIDGE_METAL_BACKEND`. Plain macOS builds default to Metal; an explicit `MLXCEL_BUILD_METAL=OFF` disables both the backend and its controls.

Pure unit tests cover the macOS default, accepted explicit values, the disabled non-macOS policy, and named errors for invalid macOS values. A live example toggles QMV both ways and round-trips the Metal command-buffer budget, restoring the prior process state on exit.

### 2.2 Exactness diagnostics and state

The retry returns its complete `BlockChainExactness` verdict. Decline snapshots and warning logs now retain separate initial and narrow-retry reasons, including localized layer information and differing-byte counts.

An exact retry leaves narrow QMV selected because re-enabling the wide path would invalidate the decision just measured. Failed and `NotRun` retries restore wide QMV. Operator-pinned `MLXCEL_QMV_WIDE` values still skip the automatic retry and are named as the reason.

### 2.3 Compatibility and security

The backend policy change is limited to build configuration and existing runtime controls. Linux and other non-Metal builds retain inert Metal controls. No endpoint, authentication behavior, unsafe block, or checkpoint format changes.

The workaround for older affected binaries remains starting a fresh process with `MLXCEL_QMV_WIDE=0`. The variable is initialized per process, applies only to Metal QMV dispatch, may reduce verify throughput, and does not bypass the model's exactness probe.

## 3. Validation

| Check | Result |
|---|---|
| Plain-release runtime-control example | QMV false/true observed correctly; command-buffer budget 17 round-tripped |
| Gemma 4 31B QAT, automatic retry | Passed at block size 4; 64 output tokens; 33/84 accepted over 28 rounds |
| Gemma 4 31B QAT, pinned narrow | Passed; output and acceptance matched automatic retry |
| Gemma output identity | 268 bytes; SHA-256 `55ad7d86e23b6132e4b3b2af1d5f9cdda4ed552baf26690d76a410bfe823d66d` |
| Qwen 3.8, automatic retry | Passed at block size 3; 64 output tokens; 35/52 accepted over 26 rounds |
| Qwen 3.8, pinned narrow | Passed; output and acceptance matched automatic retry |
| Qwen output identity | 328 bytes; SHA-256 `3aca47725fce1ffda00d959c4d9758eccc9b2bd04daaf9c4f6af575f1e6add05` |
| Format and focused retry tests | Passed |
| Full `make verify` workspace gate | Passed |
| Focused operator-pin suite | Passed 16/16 with the variable unset, pinned to 0, and pinned to 1 |

The Gemma automatic run first reproduced the known wide-path difference of 240,460/524,288 bytes, then passed under the runtime-selected narrow path. Its generated output also matched the explicit-Metal release baseline.

These are correctness runs. Their single-sample timings are not used as throughput evidence.

The full gate passed before the final test-only isolation adjustment. The affected focused suite was then rebuilt and passed in all three ambient operator-pin modes.

## 4. Change Summary

| Item | Value |
|---|---:|
| Implementation files | 8 |
| Primary commit | `d939279f` |

| Category | Summary |
|---|---|
| Build correctness | One Metal decision now controls both MLX and bridge compilation |
| Runtime behavior | Plain release builds can actually toggle QMV and command-buffer settings |
| Diagnostics | Initial and retry verdicts remain distinct and complete |
| Tests | Added build-policy, retry-state, diagnostic, and live bridge checks |
| Documentation | Documented plain-build behavior, verification example, and old-binary workaround |

## 5. Learning Points

A runtime backend query is insufficient when the control surface was compiled under a different condition. Backend construction and every backend-specific bridge must consume the same resolved build decision.

A retry diagnostic should preserve each arm's measured result. Reporting only that a retry failed can hide a no-op control, while locations and byte counts reveal whether the second measurement actually changed.

## 6. Limitations

The corrected real-model validation covers one M5 Max host, Gemma 4 31B QAT, and Qwen 3.8. It does not establish a general performance delta or guarantee that narrow QMV makes every model exact. The user's original M5 Ultra host was unavailable for the corrected-binary rerun. Cross-host compilation and physical Metal-OFF/CUDA execution remain unvalidated. The exactness gate remains authoritative for each loaded model and block size.

## 7. References

- Issue #1988: diagnose automatic QMV retry failure on M5 Ultra
- Issue #1187: automatic narrow-QMV exactness retry
- PR #1987: Gemma 4 31B MTP exactness
- `docs/installation.md`
- `src/lib/mlxcel-core/examples/metal_runtime_switch_probe.rs`
