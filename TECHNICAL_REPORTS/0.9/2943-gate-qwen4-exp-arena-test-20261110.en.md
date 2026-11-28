# PR #2943: Gate the qwen4_exp arena zero-copy test to macOS

**Date**: 2026-11-10
**Status**: Completed (PR open as draft; closes #2939)
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

PR #2943 fixes a test that failed deterministically on CUDA since #2912. `models::qwen4_exp::stream::arena::tests::a_write_into_a_slot_is_seen_through_the_wrapped_array` asserts that MLX wraps a page-aligned expert plane without copying it, which only the Metal allocator can do. The test module is now compiled only on macOS, and a new non-macOS test pins the behavior that makes the gate safe: `ExpertStream::resolve` refuses streaming off Metal. No production code changes. On GB10 (CUDA) the `qwen4_exp::stream` tests pass, 33 of 33.

## 1. Problem Statement

`from_bytes_nocopy` (`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`) calls MLX's `array(void*, Shape, Dtype, deleter)` constructor. That constructor asks `allocator::make_buffer` to adopt the pointer and falls back to malloc plus copy when it returns `nullptr`. Metal implements `make_buffer` with `newBuffer(ptr, ...)`; the CUDA allocator does not override it, so the base class returns `nullptr` and every wrap on CUDA is a copy. The arena test compares `array_data_address` with the plane pointer, so it fails on every CUDA run (3 of 3 isolated runs, and the single failure in an 11,641-test lib run). Main CI does not run the CUDA lib suite, which is why #2912 merged with it red.

Production was not affected. `ExpertStream::resolve` returns an error for `MLXCEL_EXPERT_STREAM=1` and `Ok(None)` for `auto` when the target is not macOS, and `LayerStream::new` rejects a plane MLX copied, so CUDA never builds a detached arena.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/models/qwen4_exp_stream_arena.rs` | The test module is `#[cfg(all(test, target_os = "macos"))]`, with a comment giving the reason |
| `src/models/qwen4_exp_stream_tests.rs` | New `#[cfg(not(target_os = "macos"))]` test `resolve_refuses_streaming_off_metal` and its env-restoring drop guard |

## 3. Technical Decisions

### 3.1 Gate the test, not the bridge

Making `from_bytes_nocopy` zero-copy on CUDA would need a `make_buffer` for the CUDA allocator in the pinned MLX, which is upstream work, and nothing on CUDA would use it while streaming stays Metal-only. The issue ruled the bridge out of scope, so the test's platform now matches the feature's.

### 3.2 Gate the whole module

The arena `mod tests` holds this one test and a `use super::*;`. Gating only the function would leave the import unused on Linux, and clippy runs with `-D warnings`. The module-level cfg compiles the whole block out instead.

### 3.3 Replace the lost Linux coverage with the refusal itself

Gating removes the only arena check on Linux, so the new test asserts the invariant the gate relies on: with `MLXCEL_EXPERT_STREAM=1`, `resolve` returns an error containing "Metal only", and with `auto` it returns `Ok(None)`. If someone later enables streaming on CUDA, this test fails and points at the arena test that needs to run there. The test passes a nonexistent model directory on purpose, since the platform check runs before any file is opened.

### 3.4 Env mutation discipline

The test holds `crate::test_support::env_lock::env_lock()` for its whole run, as every env-mutating test in the crate must. A drop guard declared after the lock saves the previous value with `var_os` and restores or removes it. Locals drop in reverse order, so the restore runs before the lock is released, and it runs on a panic too. `set_var` is wrapped in `unsafe` with a SAFETY note, as edition 2024 requires.

## 4. Validation

- `gpu-lock run --tag issue-2939 -- cargo test --profile test-fast --features cuda -p mlxcel --lib qwen4_exp::stream -- --test-threads=1`: 33 passed, 0 failed, on spark-102 (GB10). The new test is among them, and the arena test is no longer compiled on Linux.
- `cargo clippy -p mlxcel --lib --tests --profile test-fast --features cuda -- -D warnings`, `cargo fmt --all -- --check`, `scripts/ci/check_env_var_docs.py` and `scripts/insert_apache_header.py --check`: clean.

## 5. Not Verified

The macOS cfg path was not compiled: this host has no macOS target installed. By reading the cfg, macOS sees the arena module exactly as before and does not see the new test or its guard, so neither leaves an unused item there. The macOS CI job is the first build that checks this.

## 6. Learning Points

- MLX's no-copy array constructor depends on `allocator::make_buffer`, which only Metal implements in the pinned MLX. A test that asserts a zero-copy wrap is a Metal test, whatever module it lives in.
- When a test is gated off a platform, add a test on that platform for the check that keeps the platform away from the code path, so the gate cannot hide a later behavior change.
