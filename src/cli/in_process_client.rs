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

//! The settings a CLI client gives the server it starts in-process (issue
//! #2173, ADR 0007 decision table), shared by `mlxcel run`, the chat REPL and
//! the `mlxcel-engine-parity` arm that checks `run -p` against the server.
//!
//! The CLI is the in-process server's only client, so every sampling flag maps
//! one to one onto the `mlxcel-server` option of the same meaning, and the
//! server resolves each request from those options exactly as it resolves
//! requests from its own command line ([`CliServerSettings::startup`]). The
//! two request-only knobs, `--p-less` and `--stop`, ride on the request body
//! instead ([`CliServerSettings::request_fields`]). A flag the user did not
//! pass leaves the server default in place, with one documented exception:
//! the base sampler stays greedy (temperature 0, top-k 0, top-p 1, min-p 0)
//! when `generation_config.json` is silent, so a bare `mlxcel run -p` stays
//! reproducible.

use std::path::{Path, PathBuf};

use clap::Args;
use mlxcel_core::cache::KVCacheMode;
use serde_json::{Map, Value, json};

use crate::server::ServerStartupConfig;

/// Sampling flags the server has and the CLI sampling group did not, exposed
/// on `mlxcel run` with the `mlxcel-server` spellings and defaults.
#[derive(Args, Debug, Clone, Default, PartialEq)]
#[command(next_help_heading = "Server Sampling Options")]
pub struct ServerSamplingOptions {
    /// Frequency penalty (0.0 = disabled), as `mlxcel-server --frequency-penalty`.
    #[arg(long = "frequency-penalty", value_name = "N")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (0.0 = disabled), as `mlxcel-server --presence-penalty`.
    #[arg(long = "presence-penalty", value_name = "N")]
    pub presence_penalty: Option<f32>,

    /// XTC removal probability (0.0 = disabled).
    #[arg(long = "xtc-probability", value_name = "N")]
    pub xtc_probability: Option<f32>,

    /// XTC probability threshold (values above 0.5 make XTC inert).
    #[arg(long = "xtc-threshold", value_name = "N")]
    pub xtc_threshold: Option<f32>,

    /// Mirostat sampling (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    #[arg(long = "mirostat", value_name = "N")]
    pub mirostat: Option<i32>,

    /// Mirostat learning rate, parameter eta.
    #[arg(long = "mirostat-lr", value_name = "N")]
    pub mirostat_eta: Option<f32>,

    /// Mirostat target entropy, parameter tau.
    #[arg(long = "mirostat-ent", value_name = "N")]
    pub mirostat_tau: Option<f32>,

    /// Dynamic temperature range (0.0 = disabled).
    #[arg(long = "dynatemp-range", value_name = "N")]
    pub dynatemp_range: Option<f32>,

    /// Dynamic temperature exponent.
    #[arg(long = "dynatemp-exp", value_name = "N")]
    pub dynatemp_exponent: Option<f32>,

    /// DRY sequence breaker strings (default: "\n", ":", "\"", "*"; "none" =
    /// no breakers), comma-separated, as `mlxcel-server --dry-sequence-breaker`.
    #[arg(
        long = "dry-sequence-breaker",
        value_delimiter = ',',
        value_name = "STRING"
    )]
    pub dry_sequence_breakers: Vec<String>,

    /// Stop generating when the reply produces this string (repeatable). Sent
    /// as the request's `stop` field.
    #[arg(long = "stop", value_name = "STRING")]
    pub stop: Vec<String>,
}

/// Everything a CLI client sets on its in-process server, resolved from the
/// command line before the model path is known.
#[derive(Debug, Clone, PartialEq)]
pub struct CliServerSettings {
    /// `--temp` / `-t` when given; otherwise the greedy CLI base applies
    /// unless `generation_config.json` sets a temperature.
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: f32,
    pub typical_p: f32,
    pub top_n_sigma: f32,
    pub p_less: bool,
    pub repeat_penalty: f32,
    /// `--repeat-last-n` when given (`-1` = full history); the server default
    /// window (64) otherwise.
    pub repeat_last_n: Option<i32>,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    /// `--dry-penalty-last-n` when given (`-1` = full history); the server
    /// default window (64) otherwise.
    pub dry_penalty_last_n: Option<i64>,
    pub seed: Option<u64>,
    pub extra: ServerSamplingOptions,
    /// `-n`; `None` is the unlimited `-1`.
    pub max_tokens: Option<usize>,
    pub kv_cache_mode: KVCacheMode,
    pub adapter: Option<PathBuf>,
    pub draft_model: Option<PathBuf>,
    pub draft_kind: Option<String>,
    pub draft_block_size: Option<u32>,
    /// The cross-request prompt cache: on for the chat REPL (the server
    /// default), off for a one-shot `-p` run that has nothing to reuse.
    pub prompt_cache: bool,
    /// Run the server's one-token warmup before the first request.
    pub warmup: bool,
}

impl Default for CliServerSettings {
    /// A command line with no sampling flags: the greedy CLI base, the server
    /// windows, unlimited `-n`, FP16 KV, prompt cache on, no warmup.
    fn default() -> Self {
        Self {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: 0.0,
            typical_p: 1.0,
            top_n_sigma: 0.0,
            p_less: false,
            repeat_penalty: 1.0,
            repeat_last_n: None,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: None,
            seed: None,
            extra: ServerSamplingOptions::default(),
            max_tokens: None,
            kv_cache_mode: KVCacheMode::Fp16,
            adapter: None,
            draft_model: None,
            draft_kind: None,
            draft_block_size: None,
            prompt_cache: true,
            warmup: false,
        }
    }
}

impl CliServerSettings {
    /// The startup configuration of the in-process server for `model_path`:
    /// one slot (`--parallel 1`, `--max-batch-size 1`, so decode storage
    /// resolves to dense, ADR 0007), the CLI's sampling flags as server
    /// defaults, the prompt cache per [`Self::prompt_cache`], the speculative
    /// drafter, and the KV cache mode.
    pub fn startup(&self, model_path: &Path) -> ServerStartupConfig {
        let mut startup = ServerStartupConfig {
            model_path: model_path.to_path_buf(),
            adapter_path: self.adapter.clone(),
            n_parallel: 1,
            max_batch_size: Some(1),
            warmup: self.warmup,
            n_predict: self
                .max_tokens
                .map_or(-1, |n| i32::try_from(n).unwrap_or(i32::MAX)),
            temperature: self.temperature.unwrap_or(0.0),
            temperature_was_set: self.temperature.is_some(),
            top_k: self.top_k.unwrap_or(0),
            top_k_was_set: self.top_k.is_some(),
            top_p: self.top_p.unwrap_or(1.0),
            top_p_was_set: self.top_p.is_some(),
            min_p: self.min_p,
            typical_p: self.typical_p,
            top_n_sigma: self.top_n_sigma,
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            kv_cache_mode: self.kv_cache_mode,
            draft_model_path: self.draft_model.clone(),
            draft_kind: self.draft_kind.clone(),
            draft_block_size: self.draft_block_size,
            ..ServerStartupConfig::default()
        };
        if let Some(n) = self.repeat_last_n {
            // `-1` is the CLI's full-history sentinel; the server window is
            // unsigned, so it becomes the widest window the sampler takes.
            startup.repeat_last_n = usize::try_from(n).unwrap_or(i32::MAX as usize);
        }
        if let Some(n) = self.dry_penalty_last_n {
            startup.dry_penalty_last_n = i32::try_from(n.max(-1)).unwrap_or(i32::MAX);
        }
        let extra = &self.extra;
        if let Some(v) = extra.frequency_penalty {
            startup.frequency_penalty = v;
        }
        if let Some(v) = extra.presence_penalty {
            startup.presence_penalty = v;
        }
        if let Some(v) = extra.xtc_probability {
            startup.xtc_probability = v;
        }
        if let Some(v) = extra.xtc_threshold {
            startup.xtc_threshold = v;
        }
        if let Some(v) = extra.mirostat {
            startup.mirostat = v;
        }
        if let Some(v) = extra.mirostat_eta {
            startup.mirostat_eta = v;
        }
        if let Some(v) = extra.mirostat_tau {
            startup.mirostat_tau = v;
        }
        if let Some(v) = extra.dynatemp_range {
            startup.dynatemp_range = v;
        }
        if let Some(v) = extra.dynatemp_exponent {
            startup.dynatemp_exponent = v;
        }
        startup.dry_sequence_breakers = extra.dry_sequence_breakers.clone();
        if !self.prompt_cache {
            startup.prompt_cache.enabled = false;
        }
        startup
    }

    /// The request-body fields the CLI sets on every request: the two
    /// sampling knobs the server has only per request.
    pub fn request_fields(&self) -> Map<String, Value> {
        let mut fields = Map::new();
        if self.p_less {
            fields.insert("p_less".to_string(), json!(true));
        }
        if !self.extra.stop.is_empty() {
            fields.insert("stop".to_string(), json!(self.extra.stop));
        }
        fields
    }
}

/// The `/v1/chat/completions` body for a conversation.
#[must_use]
pub fn chat_request_body(messages: Value, settings: &CliServerSettings) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), json!("mlxcel-run"));
    body.insert("messages".to_string(), messages);
    body.insert("stream".to_string(), json!(true));
    body.extend(settings.request_fields());
    Value::Object(body)
}

/// The `/v1/completions` body for a `--no-chat-template` prompt.
#[must_use]
pub fn completion_request_body(prompt: &str, settings: &CliServerSettings) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), json!("mlxcel-run"));
    body.insert("prompt".to_string(), json!(prompt));
    body.insert("stream".to_string(), json!(true));
    body.extend(settings.request_fields());
    Value::Object(body)
}
