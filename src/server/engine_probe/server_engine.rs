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

//! The server decode path, driven in-process at one sequence at a time.
//!
//! [`ServerEngine`] builds exactly what `start_server` builds, minus the HTTP
//! listener: the [`ServerConfig`] comes from [`build_server_config`] over the
//! default [`ServerStartupConfig`] (the `mlxcel-server` defaults), the prompt
//! cache store from [`resolve_prompt_cache_store`], and the model worker from
//! [`ModelProvider::new_with_server_config_and_prompt_cache`], which spawns the
//! same `BatchScheduler` worker thread the server runs. Requests enter through
//! the provider's request channel with pre-tokenized ids, so tokenization and
//! HTTP stay outside the timed region while admission, prefill, decode and the
//! finish step are the real scheduler code.
//!
//! One request is in flight at a time, so the scheduler decodes a batch of one
//! while keeping the server's default admission width (`--parallel 4`). That
//! width matters: paged decode storage is only available when the worker's
//! `max_batch_size` is above one, so `--max-batch-size 1` would silently turn
//! every paged run into a dense one.
//!
//! [`build_server_config`]: crate::server::startup::build_server_config
//! [`resolve_prompt_cache_store`]: crate::server::startup::resolve_prompt_cache_store

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mlxcel_core::generate::SamplingConfig;

use crate::server::batch::{BatchObservability, RequestPriority};
use crate::server::config::PromptCacheRequestContext;
use crate::server::prompt_cache::key::{MultimodalDigest, resolve_session_key};
use crate::server::startup::{build_server_config, resolve_prompt_cache_store, warmup_model};
use crate::server::{
    ApiKeys, BatchMetrics, DecodeStorageBackend, ModelProvider, ServerConfig,
    ServerGenerateOptions, ServerStartupConfig,
};

/// How long [`ServerEngine::shutdown`] waits for the worker thread to release
/// the model before giving up.
const WORKER_EXIT_TIMEOUT: Duration = Duration::from_secs(120);

/// Cache-key template dimension for probe requests. Probe requests never share
/// a bucket with real chat or raw-prompt traffic.
const PROBE_TEMPLATE_SIG: &str = "mlxcel:engine-probe:v1";

/// Server knobs the probes vary. Everything else is the `mlxcel-server`
/// default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerEngineOptions {
    /// Decode storage for server sequences. `Auto` is requested as `Paged`:
    /// the scheduler resolves the two identically (paged when the worker can
    /// serve it, dense otherwise), but only an explicit `Paged` request
    /// records the fallback, which is what lets the probe report the storage
    /// it actually measured.
    pub decode_storage: DecodeStorageBackend,
    /// `--prefill-chunk-size`; `None` keeps the server default (512).
    pub prefill_chunk_size: Option<usize>,
    /// Whether the cross-request prompt cache is enabled (`--no-prompt-cache`
    /// when `false`). The server default is enabled.
    pub prompt_cache: bool,
}

impl Default for ServerEngineOptions {
    fn default() -> Self {
        Self {
            decode_storage: DecodeStorageBackend::Auto,
            prefill_chunk_size: None,
            prompt_cache: true,
        }
    }
}

/// One request to the server engine.
#[derive(Debug, Clone)]
pub struct ServerEngineRequest<'a> {
    /// Prompt token ids, used as-is (no server-side tokenization).
    pub prompt_tokens: &'a [i32],
    /// Sampling configuration, passed through unchanged; the scheduler only
    /// merges the checkpoint's EOS ids into the stop set, as it does for
    /// every request.
    pub sampling: SamplingConfig,
    /// Generation budget.
    pub max_tokens: usize,
    /// b10621 `ignore_eos`: suppress every end-of-generation token.
    pub ignore_eos: bool,
    /// Attach a prompt-cache context so the request can adopt a stored prefix
    /// and donate its cache back. `None` keeps the request out of the cache,
    /// as a raw route with `cache_prompt: false` does.
    pub prompt_cache: Option<ProbeCacheKey>,
}

/// The prompt-cache identity of a probe request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeCacheKey {
    /// Session key; requests with different keys never share an entry, so
    /// independent cases cannot hit each other's donated prefixes.
    pub session: String,
    /// History-boundary tokens (the conversation rendered without the
    /// generation prompt), or `None` for a context without a boundary.
    pub history_tokens: Option<Vec<i32>>,
}

/// What one server-engine request produced, with timings taken on the calling
/// thread.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerEngineRun {
    /// Generated token ids in order, as the scheduler reports them.
    pub tokens: Vec<i32>,
    /// Prompt length the scheduler admitted.
    pub prompt_tokens: usize,
    /// Leading prompt tokens supplied by an adopted prompt-cache prefix.
    pub cached_tokens: usize,
    /// Prompt tokens the scheduler actually forwarded for this request.
    pub forwarded_prefill_tokens: u64,
    /// Request send to the first-token prefill event, in milliseconds.
    pub ttft_ms: f64,
    /// First-token event to request completion, in milliseconds. Covers the
    /// generation of every token after the first.
    pub decode_ms: f64,
    /// The scheduler's own `prompt_eval_ms` (millisecond resolution).
    pub server_prompt_eval_ms: u64,
    /// The scheduler's own `generation_only_ms` (millisecond resolution).
    pub server_generation_ms: u64,
    /// OpenAI finish reason (`stop` or `length`).
    pub finish_reason: String,
    /// Prompt-cache entries this request stored when it finished.
    pub prompt_cache_inserts: u64,
    /// Why the prompt cache declined to store or adopt for this request, when
    /// it recorded a decline while the request ran.
    pub prompt_cache_reject: Option<&'static str>,
    /// Pooled paged-attention kernel launches (fused v2 plus gather fallback,
    /// summed over layers and steps) while this request ran, read from the
    /// process-wide `mlxcel_core::cache::paged_batch_decode_stats` counters.
    /// Zero on dense storage. Zero on paged storage too when the model keeps
    /// its KV in model-owned per-sequence state (Gemma 3, Llama 4, Qwen 3.5):
    /// at B=1 those families decode through their dense caches and only the
    /// block table is paged, so this field is the evidence of which attention
    /// a "paged" run actually used.
    pub paged_decode_launches: u64,
}

/// Process-wide count of pooled paged-attention launches (fused kernel plus
/// gather fallback). One probe request is in flight at a time, so the delta
/// across a request belongs to it.
fn paged_kernel_launches() -> u64 {
    let stats = mlxcel_core::cache::paged_batch_decode_stats();
    stats.v2_launches + stats.gather_fallbacks
}

impl ServerEngineRun {
    /// Decode throughput in the `mlxcel-bench-decode` convention: generated
    /// tokens over the post-first-token time.
    #[must_use]
    pub fn decode_tok_per_sec(&self) -> f64 {
        if self.decode_ms > 0.0 {
            self.tokens.len() as f64 / (self.decode_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// The in-process server engine. Dropping it shuts the worker down without
/// waiting; call [`Self::shutdown`] to wait for the model to be released.
pub struct ServerEngine {
    provider: ModelProvider,
    config: ServerConfig,
    observability: Arc<BatchObservability>,
    model_path: PathBuf,
    options: ServerEngineOptions,
}

impl ServerEngine {
    /// Build the server configuration, prompt cache store and model worker the
    /// way `start_server` does, then run the server's own one-token warmup.
    pub fn start(model_path: &Path, options: ServerEngineOptions) -> Result<Self> {
        // `start_server` turns CUDA graph capture off for the families that
        // need it (Gemma 4, #688) on the main thread before any worker exists;
        // do the same so the worker's own load-site call is a no-op here too.
        crate::loading::maybe_disable_cuda_graphs_for_model_for_path(model_path);
        let mut startup = ServerStartupConfig {
            model_path: model_path.to_path_buf(),
            decode_storage_backend: Some(match options.decode_storage {
                DecodeStorageBackend::Auto => DecodeStorageBackend::Paged,
                other => other,
            }),
            ..ServerStartupConfig::default()
        };
        if let Some(chunk) = options.prefill_chunk_size {
            startup.prefill_chunk_size = chunk;
        }
        if !options.prompt_cache {
            startup.prompt_cache.enabled = false;
        }
        let mut config = build_server_config(&startup, ApiKeys::default());
        let batch_metrics = Arc::new(BatchMetrics::new());
        let observability = Arc::new(BatchObservability::new());
        let store = resolve_prompt_cache_store(&mut config, model_path, &batch_metrics);
        let provider = ModelProvider::new_with_server_config_and_prompt_cache(
            model_path.to_path_buf(),
            None,
            &config,
            store,
            batch_metrics,
            observability.clone(),
        )
        .context("failed to start the server model worker")?;
        warmup_model(&provider).context("server warmup request failed")?;
        Ok(Self {
            provider,
            config,
            observability,
            model_path: model_path.to_path_buf(),
            options,
        })
    }

    /// The options this engine was started with.
    #[must_use]
    pub fn options(&self) -> ServerEngineOptions {
        self.options
    }

    /// The prefill chunk the scheduler runs with.
    #[must_use]
    pub fn prefill_chunk_size(&self) -> usize {
        self.config.prefill_chunk_size
    }

    /// Whether the prompt cache store is live for this engine.
    #[must_use]
    pub fn prompt_cache_enabled(&self) -> bool {
        self.provider.prompt_cache().is_some()
    }

    /// The decode storage the scheduler resolved: paged only when paged was
    /// requested (directly or through `Auto`) and the worker did not fall back
    /// to dense. This is the sequence storage, not the attention kernel: a
    /// model-owned family resolves to paged yet decodes through dense caches
    /// at B=1, which [`ServerEngineRun::paged_decode_launches`] shows.
    #[must_use]
    pub fn effective_decode_storage(&self) -> DecodeStorageBackend {
        match self.options.decode_storage {
            DecodeStorageBackend::Dense => DecodeStorageBackend::Dense,
            DecodeStorageBackend::Auto | DecodeStorageBackend::Paged => {
                if self.observability.snapshot().decode_storage_fallbacks > 0 {
                    DecodeStorageBackend::Dense
                } else {
                    DecodeStorageBackend::Paged
                }
            }
        }
    }

    /// Where the history-boundary split lands for a cold prefill of
    /// `prompt_tokens` with `history_tokens` as the boundary render, or
    /// `None` when the scheduler would prefill it unsplit (dense-KV family,
    /// prompt cache off, boundary snapshots disabled, or no usable boundary).
    #[must_use]
    pub fn history_boundary(&self, history_tokens: &[i32], prompt_tokens: &[i32]) -> Option<usize> {
        if !self.prompt_cache_enabled()
            || !self.provider.supports_snapshot_reuse()
            || crate::server::prompt_cache::boundary_snapshot_disabled()
        {
            return None;
        }
        crate::server::batch::scheduler::history_boundary_len(
            history_tokens,
            prompt_tokens,
            self.config.prompt_cache.min_prefix_tokens,
        )
    }

    /// Run one request to completion and time it on the calling thread.
    pub fn run(&self, request: &ServerEngineRequest<'_>) -> Result<ServerEngineRun> {
        let prompt_cache_ctx = request
            .prompt_cache
            .as_ref()
            .map(|key| PromptCacheRequestContext {
                model_id: self.model_path.display().to_string(),
                lora_id: None,
                template_sig: PROBE_TEMPLATE_SIG.to_string(),
                session_key: resolve_session_key(Some(&key.session), None).to_string(),
                mm_digest: MultimodalDigest::empty(),
                history_prompt: None,
                history_prefix_tokens: key.history_tokens.clone(),
            });
        let options = ServerGenerateOptions {
            max_tokens: request.max_tokens,
            sampling: request.sampling.clone(),
            stop_sequences: None,
            n_indent: 0,
            reasoning_budget_message: None,
            t_max_predict_ms: None,
            ignore_eos: request.ignore_eos,
            priority: RequestPriority::Normal,
            logprobs: Default::default(),
            dry_breaker_strings: None,
            logit_bias: Vec::new(),
            logit_bias_texts: Vec::new(),
            post_sampling_probs: false,
            reasoning_budget: Default::default(),
            thinking_enter_block_on_start: false,
            reasoning_control: None,
            prompt_cache_ctx,
            structured: None,
            retention: Default::default(),
            lora_scales: None,
            grammar: None,
            image_soft_tokens: None,
            pre_rendered_prompt_tokens: Some(request.prompt_tokens.to_vec()),
        };
        let live = self.config.live_settings();
        let before = self.observability.snapshot();
        let paged_before = paged_kernel_launches();
        let mut first_token_at: Option<Instant> = None;
        let start = Instant::now();
        let result = self.provider.generate_with_live_with_prefill(
            String::new(),
            options,
            &live,
            |stats| {
                if stats.first_token && first_token_at.is_none() {
                    first_token_at = Some(Instant::now());
                }
            },
        )?;
        let done = Instant::now();
        let after = self.observability.snapshot();
        let paged_after = paged_kernel_launches();
        // A request that finished inside prefill (immediate EOS) never stamps a
        // first token; its whole wall time is then prefill.
        let first = first_token_at.unwrap_or(done);
        Ok(ServerEngineRun {
            tokens: result.generated_token_ids,
            prompt_tokens: result.prompt_tokens,
            cached_tokens: result.cached_tokens,
            forwarded_prefill_tokens: after
                .total_prefill_tokens
                .saturating_sub(before.total_prefill_tokens),
            ttft_ms: first.duration_since(start).as_secs_f64() * 1000.0,
            decode_ms: done.duration_since(first).as_secs_f64() * 1000.0,
            server_prompt_eval_ms: result.prompt_eval_ms,
            server_generation_ms: result.generation_only_ms,
            finish_reason: result.finish_reason,
            prompt_cache_inserts: after
                .prompt_cache_inserts
                .saturating_sub(before.prompt_cache_inserts),
            paged_decode_launches: paged_after.saturating_sub(paged_before),
            prompt_cache_reject: after
                .prompt_cache_last_reject
                .filter(|r| {
                    before
                        .prompt_cache_last_reject
                        .is_none_or(|b| b.at_unix_ms != r.at_unix_ms || b.seq_id != r.seq_id)
                })
                .map(|r| r.reason),
        })
    }

    /// Ask the worker to exit and wait until it has released the model, so a
    /// following engine (or the CLI path) does not hold two copies.
    pub fn shutdown(self) -> Result<()> {
        self.provider.shutdown_worker();
        let observer = self.provider.worker_exit_observer();
        if !observer.wait_timeout(WORKER_EXIT_TIMEOUT) {
            anyhow::bail!(
                "server model worker did not exit within {}s",
                WORKER_EXIT_TIMEOUT.as_secs()
            );
        }
        if let Some(message) = observer.panic_message() {
            anyhow::bail!("server model worker panicked: {message}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_with(tokens: usize, decode_ms: f64) -> ServerEngineRun {
        ServerEngineRun {
            tokens: vec![0; tokens],
            prompt_tokens: 10,
            cached_tokens: 0,
            forwarded_prefill_tokens: 10,
            ttft_ms: 5.0,
            decode_ms,
            server_prompt_eval_ms: 4,
            server_generation_ms: 40,
            finish_reason: "length".to_string(),
            prompt_cache_inserts: 0,
            prompt_cache_reject: None,
            paged_decode_launches: 0,
        }
    }

    #[test]
    fn decode_rate_is_tokens_over_post_first_token_time() {
        assert!((run_with(50, 500.0).decode_tok_per_sec() - 100.0).abs() < 1e-9);
        assert_eq!(run_with(50, 0.0).decode_tok_per_sec(), 0.0);
        assert_eq!(run_with(0, 500.0).decode_tok_per_sec(), 0.0);
    }

    #[test]
    fn default_options_are_the_server_defaults() {
        let options = ServerEngineOptions::default();
        assert_eq!(options.decode_storage, DecodeStorageBackend::Auto);
        assert_eq!(options.prefill_chunk_size, None);
        assert!(options.prompt_cache);
    }
}
