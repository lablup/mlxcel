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

//! Unit tests for the b10621 prompt-cache and batching flag group.

use clap::Parser;

use super::*;

#[derive(Parser)]
#[command(allow_negative_numbers = true)]
struct Probe {
    #[command(flatten)]
    cache: CacheCompatArgs,
}

/// Parse an argv, with the process environment held still.
///
/// Every field in this group carries a `LLAMA_ARG_*` binding, so clap reads
/// the process environment on each parse. Another test in this binary mutating
/// one of those keys concurrently would decide what this one parsed, which is
/// what the crate-wide env lock exists to prevent. The guard is returned to the
/// caller-visible scope by being held for the whole call.
fn parse(argv: &[&str]) -> CacheCompatArgs {
    let _env_guard = crate::test_support::env_lock::env_lock();
    let mut full = vec!["probe"];
    full.extend_from_slice(argv);
    Probe::try_parse_from(full)
        .unwrap_or_else(|e| panic!("{argv:?} must parse: {e}"))
        .cache
}

#[test]
fn an_empty_group_leaves_every_decision_to_mlxcels_own_flags() {
    let resolved = parse(&[]).resolve().expect("no flags is not an error");
    assert_eq!(resolved, CacheCompatResolution::default());
    assert_eq!(resolved.prompt_cache_enabled, None);
}

#[test]
fn cache_prompt_and_no_cache_prompt_resolve_to_a_tri_state() {
    assert_eq!(
        parse(&["--cache-prompt"])
            .resolve()
            .unwrap()
            .prompt_cache_enabled,
        Some(true)
    );
    assert_eq!(
        parse(&["--no-cache-prompt"])
            .resolve()
            .unwrap()
            .prompt_cache_enabled,
        Some(false)
    );
}

#[test]
fn the_last_prompt_cache_flag_on_the_command_line_wins() {
    // `overrides_with` on both directions: a wrapper script that appends
    // `--no-cache-prompt` to a base command line carrying `--cache-prompt`
    // must end up disabled, and the reverse must end up enabled.
    assert_eq!(
        parse(&["--cache-prompt", "--no-cache-prompt"])
            .resolve()
            .unwrap()
            .prompt_cache_enabled,
        Some(false)
    );
    assert_eq!(
        parse(&["--no-cache-prompt", "--cache-prompt"])
            .resolve()
            .unwrap()
            .prompt_cache_enabled,
        Some(true)
    );
}

#[test]
fn cache_reuse_zero_is_upstreams_default_and_is_accepted() {
    let resolved = parse(&["--cache-reuse", "0"])
        .resolve()
        .expect("0 means no KV-shift reuse, which is what mlxcel does");
    assert_eq!(resolved, CacheCompatResolution::default());
}

#[test]
fn a_positive_cache_reuse_is_refused_with_what_is_missing() {
    let err = parse(&["--cache-reuse", "256"])
        .resolve()
        .expect_err("KV-shift chunk reuse is not implemented");
    assert!(err.contains("--cache-reuse 256"), "{err}");
    // The message has to name the missing operation, not just say "no".
    assert!(err.contains("re-bas"), "{err}");
    assert!(err.contains("#1453"), "{err}");
}

#[test]
fn a_negative_cache_reuse_is_out_of_domain() {
    let err = parse(&["--cache-reuse", "-1"])
        .resolve()
        .expect_err("a chunk size cannot be negative");
    assert!(err.contains("zero or positive"), "{err}");
}

#[test]
fn cache_ram_is_read_in_mebibytes() {
    assert_eq!(
        parse(&["--cache-ram", "512"])
            .resolve()
            .unwrap()
            .capacity_bytes,
        Some(512 * 1024 * 1024)
    );
    // b10621's own default, spelled out, must land on 8 GiB rather than being
    // mistaken for bytes.
    assert_eq!(
        parse(&["--cache-ram", "8192"])
            .resolve()
            .unwrap()
            .capacity_bytes,
        Some(8192 * 1024 * 1024)
    );
}

#[test]
fn cache_ram_sentinels_follow_upstream() {
    // `-1` = no limit, `0` = disable. `0` must reach the config as a real zero
    // rather than as "unset", because `PromptCacheConfig::is_enabled` reads
    // `capacity_bytes > 0` and that is how the disable takes effect.
    assert_eq!(
        parse(&["--cache-ram", "-1"])
            .resolve()
            .unwrap()
            .capacity_bytes,
        Some(usize::MAX)
    );
    assert_eq!(
        parse(&["--cache-ram", "0"])
            .resolve()
            .unwrap()
            .capacity_bytes,
        Some(0)
    );
}

#[test]
fn a_negative_cache_ram_other_than_the_sentinel_is_refused() {
    let err = parse(&["--cache-ram", "-2"])
        .resolve()
        .expect_err("-1 is the only negative upstream defines");
    assert!(err.contains("out of domain"), "{err}");
}

/// `LLAMA_ARG_CONT_BATCHING=0` and `LLAMA_ARG_CACHE_PROMPT=0` are how a
/// deployment turns these off without a command line. A plain boolean flag
/// would read a falsy environment value as "nothing was set"; the optional
/// value is what keeps the two apart.
#[test]
fn a_falsy_value_is_a_disable_rather_than_an_absence() {
    let disabled = parse(&["--cache-prompt=false", "--cont-batching=false"])
        .resolve()
        .expect("valid");
    assert_eq!(disabled.prompt_cache_enabled, Some(false));
    assert!(disabled.single_sequence_decode);

    let enabled = parse(&["--cache-prompt=true", "--cont-batching=true"])
        .resolve()
        .expect("valid");
    assert_eq!(enabled.prompt_cache_enabled, Some(true));
    assert!(!enabled.single_sequence_decode);
}

#[test]
fn no_cont_batching_pins_the_decode_width() {
    assert!(
        parse(&["--no-cont-batching"])
            .resolve()
            .unwrap()
            .single_sequence_decode
    );
    assert!(
        !parse(&["--cont-batching"])
            .resolve()
            .unwrap()
            .single_sequence_decode
    );
    assert!(!parse(&[]).resolve().unwrap().single_sequence_decode);
}

#[test]
fn the_last_batching_flag_on_the_command_line_wins() {
    assert!(
        parse(&["--cont-batching", "--no-cont-batching"])
            .resolve()
            .unwrap()
            .single_sequence_decode
    );
    assert!(
        !parse(&["--no-cont-batching", "--cont-batching"])
            .resolve()
            .unwrap()
            .single_sequence_decode
    );
}

#[test]
fn cache_reuse_is_checked_before_anything_else_resolves() {
    // A command line that is wrong in two ways must report the one that names
    // an unimplemented capability, not the one that names a domain error.
    let err = parse(&["--cache-reuse", "256", "--cache-ram", "-2"])
        .resolve()
        .expect_err("both halves are invalid");
    assert!(err.contains("--cache-reuse"), "{err}");
}
