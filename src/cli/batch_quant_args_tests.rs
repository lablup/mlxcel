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

//! Unit tests for the continuous-batching KV quantization flag group.

use std::ffi::OsString;

use clap::Parser;

use super::*;

const ENV_KEYS: [&str; 4] = [
    "LLAMA_ARG_KV_BITS",
    "LLAMA_ARG_KV_GROUP_SIZE",
    "LLAMA_ARG_KV_QUANT_SCHEME",
    "LLAMA_ARG_KV_SKIP_LAST_LAYER",
];

/// Restores clap's KV environment bindings when a parse completes or panics.
struct EnvMask(Vec<(&'static str, Option<OsString>)>);

impl EnvMask {
    fn clear() -> Self {
        let previous = ENV_KEYS
            .into_iter()
            .map(|key| (key, std::env::var_os(key)))
            .collect();
        for key in ENV_KEYS {
            // SAFETY: `parse` holds the crate-wide environment lock for this
            // guard's entire lifetime.
            #[allow(unsafe_code)]
            unsafe {
                std::env::remove_var(key);
            }
        }
        Self(previous)
    }
}

impl Drop for EnvMask {
    fn drop(&mut self) {
        for (key, value) in self.0.drain(..) {
            // SAFETY: `parse` declares its lock guard before this mask, so the
            // lock outlives restoration during reverse-order drop.
            #[allow(unsafe_code)]
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

#[derive(Parser)]
#[command(allow_negative_numbers = true)]
struct Probe {
    #[command(flatten)]
    quant: BatchKvQuantArgs,
}

/// Parse an argv, with the process environment held still.
///
/// Every field in this group carries a `LLAMA_ARG_*` binding, so an ambient
/// deployment environment could otherwise decide what these CLI-only tests
/// parse. The crate-wide lock makes clearing and restoring those keys safe.
fn parse(argv: &[&str]) -> BatchKvQuantArgs {
    let _env_guard = crate::test_support::env_lock::env_lock();
    let _env_mask = EnvMask::clear();
    let mut full = vec!["probe"];
    full.extend_from_slice(argv);
    Probe::try_parse_from(full)
        .unwrap_or_else(|e| panic!("{argv:?} must parse: {e}"))
        .quant
}

/// The hand-written `Default` impl and the clap `default_value_t` attributes
/// are two independent statements of the same defaults; this pins them to
/// each other so they cannot silently fork.
#[test]
fn default_impl_matches_an_empty_command_line() {
    let parsed = parse(&[]);
    let default = BatchKvQuantArgs::default();
    assert_eq!(parsed.kv_bits, default.kv_bits);
    assert_eq!(parsed.kv_group_size, default.kv_group_size);
    assert_eq!(parsed.kv_quant_scheme, default.kv_quant_scheme);
    assert_eq!(parsed.kv_skip_last_layer, default.kv_skip_last_layer);
}

#[test]
fn each_flag_overrides_its_default() {
    let parsed = parse(&[
        "--kv-bits",
        "8",
        "--kv-group-size",
        "32",
        "--kv-quant-scheme",
        "uniform",
        "--kv-skip-last-layer",
        "false",
    ]);
    assert_eq!(parsed.kv_bits, 8);
    assert_eq!(parsed.kv_group_size, 32);
    assert_eq!(parsed.kv_quant_scheme.as_deref(), Some("uniform"));
    assert!(!parsed.kv_skip_last_layer);
}
