// Copyright 2025-2026 Lablup Inc.
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

//! The CLI-flag to server-option mapping (issue #2173, ADR 0007 table).

use std::path::{Path, PathBuf};

use mlxcel::server::ServerStartupConfig;
use serde_json::json;

use super::*;

fn sampling(argv: &[&str]) -> SamplingOptions {
    use clap::Parser;
    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        sampling: SamplingOptions,
    }
    let mut full = vec!["probe"];
    full.extend_from_slice(argv);
    Probe::parse_from(full).sampling
}

fn settings(argv: &[&str], max_tokens: usize) -> CliServerSettings {
    let flags: Vec<String> = argv.iter().map(|a| a.to_string()).collect();
    let was_set = move |long: &str, short: Option<char>| {
        flags.iter().any(|arg| {
            arg == &format!("--{long}")
                || arg.starts_with(&format!("--{long}="))
                || short
                    .is_some_and(|s| arg.starts_with(&format!("-{s}")) && !arg.starts_with("--"))
        })
    };
    settings_from_flags(
        &sampling(argv),
        ServerSamplingOptions::default(),
        max_tokens,
        KVCacheMode::Fp16,
        &was_set,
    )
}

#[test]
fn bare_cli_keeps_greedy_base_and_server_windows() {
    let s = settings(&[], UNLIMITED_MAX_TOKENS);
    let startup = s.startup(Path::new("/models/m"));
    // Greedy base sampler, deferring to generation_config.json (not "set").
    assert_eq!(startup.temperature, 0.0);
    assert!(!startup.temperature_was_set);
    assert_eq!(startup.top_k, 0);
    assert!(!startup.top_k_was_set);
    assert_eq!(startup.top_p, 1.0);
    assert!(!startup.top_p_was_set);
    assert_eq!(startup.min_p, 0.0);
    // ADR 0007: the server's windows replace the CLI's full-history -1.
    let defaults = ServerStartupConfig::default();
    assert_eq!(startup.repeat_last_n, defaults.repeat_last_n);
    assert_eq!(startup.dry_penalty_last_n, defaults.dry_penalty_last_n);
    assert!(
        startup.dry_sequence_breakers.is_empty(),
        "server default breaker set"
    );
    // One slot, unlimited -n, prompt cache per the caller.
    assert_eq!(startup.n_parallel, 1);
    assert_eq!(startup.max_batch_size, Some(1));
    assert_eq!(startup.n_predict, -1);
    assert!(startup.prompt_cache.enabled);
    assert!(!startup.warmup);
    assert!(s.request_fields().is_empty());
}

#[test]
fn explicit_flags_map_one_to_one() {
    let s = settings(
        &[
            "-t",
            "0.7",
            "--top-k",
            "20",
            "--top-p",
            "0.9",
            "--min-p",
            "0.05",
            "--repetition-penalty",
            "1.1",
            "--repeat-last-n",
            "-1",
            "--dry-multiplier",
            "0.8",
            "--dry-penalty-last-n",
            "128",
            "--seed",
            "7",
            "--p-less",
        ],
        256,
    );
    let startup = s.startup(Path::new("/models/m"));
    assert_eq!(startup.temperature, 0.7);
    assert!(startup.temperature_was_set);
    assert_eq!(startup.top_k, 20);
    assert!(startup.top_k_was_set);
    assert_eq!(startup.top_p, 0.9);
    assert!(startup.top_p_was_set);
    assert_eq!(startup.min_p, 0.05);
    assert_eq!(startup.repeat_penalty, 1.1);
    assert_eq!(
        startup.repeat_last_n,
        i32::MAX as usize,
        "-1 = full history"
    );
    assert_eq!(startup.dry_multiplier, 0.8);
    assert_eq!(startup.dry_penalty_last_n, 128);
    assert_eq!(startup.seed, Some(7));
    assert_eq!(startup.n_predict, 256);
    assert_eq!(s.request_fields().get("p_less"), Some(&json!(true)));
}

#[test]
fn server_sampling_flags_reach_the_server_options() {
    let mut s = settings(&[], UNLIMITED_MAX_TOKENS);
    s.extra = ServerSamplingOptions {
        frequency_penalty: Some(0.3),
        presence_penalty: Some(0.4),
        xtc_probability: Some(0.5),
        xtc_threshold: Some(0.2),
        mirostat: Some(2),
        mirostat_eta: Some(0.2),
        mirostat_tau: Some(4.0),
        dynatemp_range: Some(0.5),
        dynatemp_exponent: Some(1.5),
        dry_sequence_breakers: vec!["none".to_string()],
        stop: vec!["END".to_string()],
    };
    s.prompt_cache = false;
    s.warmup = true;
    s.draft_model = Some(PathBuf::from("/models/drafter"));
    s.draft_kind = Some("dflash".to_string());
    let startup = s.startup(Path::new("/models/m"));
    assert_eq!(startup.frequency_penalty, 0.3);
    assert_eq!(startup.presence_penalty, 0.4);
    assert_eq!(startup.xtc_probability, 0.5);
    assert_eq!(startup.xtc_threshold, 0.2);
    assert_eq!(startup.mirostat, 2);
    assert_eq!(startup.mirostat_eta, 0.2);
    assert_eq!(startup.mirostat_tau, 4.0);
    assert_eq!(startup.dynatemp_range, 0.5);
    assert_eq!(startup.dynatemp_exponent, 1.5);
    assert_eq!(startup.dry_sequence_breakers, vec!["none".to_string()]);
    assert!(!startup.prompt_cache.enabled);
    assert!(startup.warmup);
    assert_eq!(
        startup.draft_model_path,
        Some(PathBuf::from("/models/drafter"))
    );
    assert_eq!(startup.draft_kind.as_deref(), Some("dflash"));
    assert_eq!(s.request_fields().get("stop"), Some(&json!(["END"])));
}

#[test]
fn request_fields_parse_as_a_chat_request() {
    let mut s = settings(&["--p-less"], UNLIMITED_MAX_TOKENS);
    s.extra.stop = vec!["###".to_string()];
    let body = mlxcel::cli::in_process_client::chat_request_body(
        json!([{ "role": "user", "content": "hi" }]),
        &s,
    );
    let request = mlxcel::server::in_process::chat::chat_request_from_json(body).expect("parses");
    assert_eq!(request.params.p_less, Some(true));
    assert_eq!(request.params.stop, Some(vec!["###".to_string()]));
    assert!(request.stream);
    assert_eq!(
        request.params.temperature, None,
        "sampling defaults live on the server"
    );
}
