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

//! The model server, started in-process without an HTTP listener (issue
//! #2173, epic #2166 Phase 5, ADR 0007 "CLI front end").
//!
//! [`InProcessServer::start`] builds what `start_server` builds for the chat
//! model, minus the listener and the side models: the [`ServerConfig`] from
//! [`build_server_config`], the prompt cache store from
//! [`resolve_prompt_cache_store`], the model worker from
//! [`ModelProvider::new_with_server_config_and_prompt_cache`] (the same
//! `BatchScheduler` thread `mlxcel-server` runs), the tokenizer and chat
//! template from [`load_chat_front`], and an [`AppState`] over them, then runs
//! the server's own one-token warmup when the startup config asks for it.
//!
//! Two clients drive it. `mlxcel run` and its chat REPL submit chat turns
//! through [`InProcessServer::chat`], which runs the `/v1/chat/completions`
//! request path ([`crate::server::routes::chat_generation`]) and consumes the
//! worker's token stream through the server's `StreamFilter`, the way
//! llama-cli is a client of llama-server but without the loopback socket and
//! SSE hop. The parity and benchmark probes
//! ([`crate::server::engine_probe::ServerEngine`]) submit pre-tokenized
//! requests through the provider directly.
//!
//! [`build_server_config`]: crate::server::startup::build_server_config
//! [`resolve_prompt_cache_store`]: crate::server::startup::resolve_prompt_cache_store
//! [`load_chat_front`]: crate::server::startup::load_chat_front

pub mod chat;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::server::batch::BatchObservability;
use crate::server::startup::{
    build_server_config, detect_model_media_support, load_chat_front, resolve_prompt_cache_store,
    run_startup_warmup,
};
use crate::server::{
    ApiKeys, AppState, BatchMetrics, ModelMediaSupport, ModelProvider, ServerConfig,
    ServerStartupConfig,
};

pub use chat::{ChatDelta, ChatTurn, ChatTurnToolCall};

/// How long [`InProcessServer::shutdown`] waits for the worker thread to
/// release the model before giving up.
const WORKER_EXIT_TIMEOUT: Duration = Duration::from_secs(120);

/// The chat model server, running in this process with no listener.
///
/// Dropping it asks the worker to exit without waiting; call
/// [`Self::shutdown`] to wait until the model is released.
pub struct InProcessServer {
    state: AppState,
    observability: Arc<BatchObservability>,
    model_path: PathBuf,
    /// Drives the async halves of the chat request path (template rendering,
    /// media resolution, grammar compilation) on the calling thread. Model
    /// work runs on the worker thread, never on this runtime.
    runtime: tokio::runtime::Runtime,
}

impl InProcessServer {
    /// Start the server for `startup.model_path` the way `start_server` does
    /// and run its warmup when `startup.warmup` is set.
    pub fn start(startup: &ServerStartupConfig) -> Result<Self> {
        let model_path = startup.model_path.clone();
        // `start_server` turns CUDA graph capture off for the families that
        // need it (Gemma 4, #688) before any worker exists; do the same so the
        // worker's own load-site call is a no-op here too.
        crate::loading::maybe_disable_cuda_graphs_for_model_for_path(&model_path);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to build the in-process server runtime")?;
        let mut config = build_server_config(startup, ApiKeys::default());
        let (tokenizer, chat_template) = load_chat_front(startup)?;
        let batch_metrics = Arc::new(BatchMetrics::new());
        let observability = Arc::new(BatchObservability::new());
        let store = resolve_prompt_cache_store(&mut config, &model_path, &batch_metrics);
        let provider = ModelProvider::new_with_server_config_and_prompt_cache(
            model_path.clone(),
            startup.adapter_path.clone(),
            &config,
            store.clone(),
            batch_metrics.clone(),
            observability.clone(),
        )
        .context("failed to start the server model worker")?;
        if startup.warmup {
            // `start_server`'s decision: the same family skips, and a failed
            // warmup is logged, not fatal.
            run_startup_warmup(&model_path, &provider);
        }
        let state = AppState::with_observability(
            Arc::new(provider),
            config,
            chat_template,
            tokenizer,
            model_path.clone(),
            batch_metrics,
            observability.clone(),
        )
        .with_media_support(detect_model_media_support(&model_path))
        .with_prompt_cache(store);
        Ok(Self {
            state,
            observability,
            model_path,
            runtime,
        })
    }

    /// The model worker's request channel.
    #[must_use]
    pub fn provider(&self) -> &ModelProvider {
        &self.state.model_provider
    }

    /// The server configuration the worker was built from.
    #[must_use]
    pub fn config(&self) -> &ServerConfig {
        &self.state.config
    }

    /// The scheduler's observability counters.
    #[must_use]
    pub fn observability(&self) -> &Arc<BatchObservability> {
        &self.observability
    }

    /// The model directory the server loaded.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// The tokenizer the chat request path renders and decodes with.
    #[must_use]
    pub fn tokenizer(&self) -> &crate::tokenizer::MlxcelTokenizer {
        &self.state.tokenizer
    }

    /// Image, audio and video support detected from the checkpoint.
    #[must_use]
    pub fn media_support(&self) -> ModelMediaSupport {
        self.state.media_support
    }

    /// Whether the prompt-prefix cache store is live.
    #[must_use]
    pub fn prompt_cache_enabled(&self) -> bool {
        self.state.prompt_cache.is_some()
    }

    /// Whether chat requests render with the generic fallback template: the
    /// checkpoint ships no chat template of its own (a likely base model) and
    /// no native renderer stands in for one. Answered from the processor
    /// loaded at start, so the model directory is not read again.
    #[must_use]
    pub fn uses_generic_chat_template(&self) -> bool {
        self.state.chat_template.kimi_k3().is_none()
            && self.state.chat_template.is_generic_default()
    }

    /// Why the loaded checkpoint cannot answer chat requests (an embedding,
    /// speech or image-task model on the chat worker), or `None` when it can.
    #[must_use]
    pub fn chat_unavailable_reason(&self) -> Option<String> {
        crate::server::routes::chat_unavailable_message(&self.state)
    }

    pub(crate) fn state(&self) -> &AppState {
        &self.state
    }

    pub(crate) fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    /// Ask the worker to exit and wait until it has released the model.
    pub fn shutdown(self) -> Result<()> {
        let provider = &self.state.model_provider;
        provider.shutdown_worker();
        let observer = provider.worker_exit_observer();
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

impl Drop for InProcessServer {
    fn drop(&mut self) {
        // Idempotent: a worker that `shutdown` already stopped ignores it.
        self.state.model_provider.shutdown_worker();
    }
}
