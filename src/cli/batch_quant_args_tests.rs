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

use clap::Parser;

use super::*;

#[derive(Parser)]
#[command(allow_negative_numbers = true)]
struct Probe {
    #[command(flatten)]
    quant: BatchKvQuantArgs,
}

/// Parse an argv, with the process environment held still.
///
/// Every field in this group except `--kv-quant-scheme`'s absence carries a
/// `LLAMA_ARG_*` binding, so clap reads the process environment on each
/// parse; the crate-wide env lock keeps concurrent tests from deciding what
/// this one parsed.
fn parse(argv: &[&str]) -> BatchKvQuantArgs {
    let _env_guard = crate::test_support::env_lock::env_lock();
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
