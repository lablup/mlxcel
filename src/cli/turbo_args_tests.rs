// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Unit tests for the shared TurboQuant KV-cache CLI argument group.
//!
//! Resolution semantics for `resolve_kv_cache_mode` /
//! `env_fallback_cache_type_*` are covered by the larger suite in
//! `src/server/cli_input_tests.rs`, which exercises the same helpers via
//! the re-exports from `mlxcel::server`. The tests here pin down the
//! `apply_to_environment` translation that is unique to this module.

use mlxcel_core::cache::KVCacheMode;

use super::{TurboKvCacheArgs, resolve_kv_cache_mode};
// `std::env::set_var` / `remove_var` are not thread-safe. Tests that touch
// the process environment must serialize through the crate-wide
// `ENV_LOCK` — a per-module lock would race with the env
// mutations in unrelated modules of the same test binary.
use crate::test_support::env_lock::env_lock;

fn boundary_env_key() -> &'static str {
    mlxcel_core::cache::turbo::BOUNDARY_V_ENV
}

#[test]
fn apply_to_environment_translates_some_into_env_var() {
    let _guard = env_lock();
    // SAFETY: tests are serialized through `env_lock`.
    unsafe {
        std::env::remove_var(boundary_env_key());
    }

    let args = TurboKvCacheArgs {
        turbo_boundary_v: Some(4),
        ..TurboKvCacheArgs::default()
    };
    args.apply_to_environment();

    assert_eq!(
        std::env::var(boundary_env_key()).expect("env var must be set"),
        "4"
    );

    // SAFETY: tests are serialized through `env_lock`.
    unsafe {
        std::env::remove_var(boundary_env_key());
    }
}

#[test]
fn apply_to_environment_with_none_leaves_env_untouched() {
    let _guard = env_lock();
    // SAFETY: tests are serialized through `env_lock`.
    unsafe {
        std::env::set_var(boundary_env_key(), "preexisting");
    }

    let args = TurboKvCacheArgs::default();
    args.apply_to_environment();

    assert_eq!(
        std::env::var(boundary_env_key()).as_deref(),
        Ok("preexisting"),
        "None must be a no-op so caller-set env vars survive"
    );

    // SAFETY: tests are serialized through `env_lock`.
    unsafe {
        std::env::remove_var(boundary_env_key());
    }
}

#[test]
fn apply_to_environment_preserves_negative_inputs_for_runtime_clamping() {
    // The boundary-V parser inside mlxcel-core treats negative values as 0,
    // and the CLI surface forwards the integer verbatim. Pin that contract:
    // a negative `--turbo-boundary-v` must still set the env var so the
    // runtime is the single source of truth for clamping/validation.
    let _guard = env_lock();
    // SAFETY: tests are serialized through `env_lock`.
    unsafe {
        std::env::remove_var(boundary_env_key());
    }

    let args = TurboKvCacheArgs {
        turbo_boundary_v: Some(-1),
        ..TurboKvCacheArgs::default()
    };
    args.apply_to_environment();

    assert_eq!(
        std::env::var(boundary_env_key()).expect("env var must be set"),
        "-1"
    );

    // SAFETY: tests are serialized through `env_lock`.
    unsafe {
        std::env::remove_var(boundary_env_key());
    }
}

// Finding 1 — `--cache-type-k fp16 --cache-type-v turbo3` was
// rejected by the split-flag resolver even though the legacy
// `--kv-cache-mode fp16+turbo3` shorthand worked, and the help text on all
// three binaries advertises `fp16+turbo3` as a supported value. Pin the
// fixed arm so a regression on the split-flag path fails loudly.

/// K=fp16, V=turbo3 (canonical short form) → Turbo3Asym.
#[test]
fn resolve_split_flags_fp16_k_turbo3_v_returns_turbo3_asym() {
    let mode = resolve_kv_cache_mode(Some("fp16"), Some("turbo3"), None)
        .expect("fp16 + turbo3 split flags must resolve");
    assert_eq!(mode, KVCacheMode::Turbo3Asym);
}

/// K=fp16, V=turbo3-asym (explicit alias) → Turbo3Asym.
#[test]
fn resolve_split_flags_fp16_k_turbo3_asym_v_returns_turbo3_asym() {
    let mode = resolve_kv_cache_mode(Some("fp16"), Some("turbo3-asym"), None)
        .expect("fp16 + turbo3-asym split flags must resolve");
    assert_eq!(mode, KVCacheMode::Turbo3Asym);
}

/// K=fp16, V=fp16+turbo3 (canonical long form, accepted by `KVCacheMode::from_str`).
#[test]
fn resolve_split_flags_fp16_k_fp16_plus_turbo3_v_returns_turbo3_asym() {
    let mode = resolve_kv_cache_mode(Some("fp16"), Some("fp16+turbo3"), None)
        .expect("fp16 + fp16+turbo3 split flags must resolve");
    assert_eq!(mode, KVCacheMode::Turbo3Asym);
}

/// Symmetric Turbo3 is intentionally not offered (see `KVCacheMode::Turbo3Asym`
/// docs). K=turbo3 + V=turbo3 must fail and the error message must list the
/// supported pairs, including the new `fp16 / turbo3` row.
#[test]
fn resolve_split_flags_symmetric_turbo3_is_rejected_with_helpful_error() {
    let err = resolve_kv_cache_mode(Some("turbo3"), Some("turbo3"), None)
        .expect_err("symmetric turbo3 must be rejected");
    assert!(
        err.contains("unsupported"),
        "error must explain that the pair is unsupported, got: {err}"
    );
    assert!(
        err.contains("fp16   / turbo3"),
        "error must list the supported `fp16 / turbo3` row, got: {err}"
    );
}
