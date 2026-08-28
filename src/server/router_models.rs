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

//! Router-mode model pool (llama-server b10621 compatible, issue #1438).
//!
//! b10621's router server discovers models under `--models-dir`, spawns one
//! child llama-server process per loaded model, and proxies requests to the
//! child the request's `model` field names. mlxcel serves the same HTTP
//! surface from one process: each pool entry, once loaded, owns a full
//! [`AppState`] plus its axum [`Router`], and the dispatcher forwards the
//! request into that sub-app in-process instead of over a child socket. The
//! state machine mirrors upstream's `UNLOADED -> LOADING -> LOADED` (a
//! failed load returns to `unloaded` with a failure recorded, which is
//! b10621's failed shape), `--models-max` bounds the concurrently loaded
//! set with LRU eviction, and `--models-autoload` (plus the per-request
//! `?autoload=` override) loads on demand.
//!
//! Confinement: entry names come only from discovery, requests resolve names
//! through the registry (a request can never smuggle a path), and a
//! discovered entry whose canonical path escapes the canonical models
//! directory (a symlink pointing outside it) is skipped at scan time.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::config::ServerConfig;
use super::{AppState, ChatTemplateProcessor, ModelProvider};

/// b10621 `server_model_status` (the subset an in-process pool reaches).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterModelStatus {
    Unloaded,
    Loading,
    Loaded,
}

impl RouterModelStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unloaded => "unloaded",
            Self::Loading => "loading",
            Self::Loaded => "loaded",
        }
    }
}

/// One discovered model.
pub struct RouterModelEntry {
    pub name: String,
    pub path: PathBuf,
    state: Mutex<EntryState>,
    last_used: AtomicI64,
}

#[derive(Default)]
struct EntryState {
    /// The loaded sub-app; `Some` while loading or loaded.
    app: Option<LoadedApp>,
    /// Set when the last load failed; cleared by the next load attempt.
    failed: bool,
}

#[derive(Clone)]
struct LoadedApp {
    state: AppState,
    router: axum::Router,
}

impl RouterModelEntry {
    fn status(&self) -> RouterModelStatus {
        let guard = match self.state.lock() {
            Ok(guard) => guard,
            Err(_) => return RouterModelStatus::Unloaded,
        };
        match &guard.app {
            None => RouterModelStatus::Unloaded,
            Some(app) => {
                if app.state.model_provider.is_loaded() {
                    RouterModelStatus::Loaded
                } else if app.state.model_provider.is_chat_unavailable() {
                    // The worker gave up: the b10621 failed shape is
                    // "unloaded" with a failure recorded.
                    RouterModelStatus::Unloaded
                } else {
                    RouterModelStatus::Loading
                }
            }
        }
    }

    fn failed(&self) -> bool {
        let Ok(guard) = self.state.lock() else {
            return false;
        };
        guard.failed
            || guard
                .app
                .as_ref()
                .is_some_and(|app| app.state.model_provider.is_chat_unavailable())
    }

    /// b10621 `is_running`: loading or loaded.
    pub fn is_running(&self) -> bool {
        matches!(
            self.status(),
            RouterModelStatus::Loading | RouterModelStatus::Loaded
        )
    }

    fn router(&self) -> Option<axum::Router> {
        self.state
            .lock()
            .ok()?
            .app
            .as_ref()
            .map(|app| app.router.clone())
    }
}

/// A status snapshot for `GET /models` and the SSE stream.
#[derive(Debug, Clone)]
pub struct RouterModelSnapshot {
    pub name: String,
    pub status: RouterModelStatus,
    pub failed: bool,
    pub vision: bool,
    pub audio: bool,
}

/// The pool shared by every router-mode handler.
pub struct RouterPool {
    entries: RwLock<BTreeMap<String, Arc<RouterModelEntry>>>,
    models_dir: PathBuf,
    /// Template for each per-model [`ServerConfig`]; the router's own CLI
    /// arguments apply to every model, the b10621 base-preset overlay.
    base_config: ServerConfig,
    pub models_max: usize,
    pub autoload_default: bool,
    events: tokio::sync::broadcast::Sender<serde_json::Value>,
    /// Serializes model loads so two concurrent autoloads cannot race the
    /// capacity check or contend the accelerator during weight upload.
    load_lock: tokio::sync::Mutex<()>,
}

/// Why a name failed to resolve or load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterPoolError {
    MissingName,
    NotFound(String),
    NotLoaded,
    LoadFailed(String),
    Capacity(String),
}

impl RouterPool {
    pub fn new(
        models_dir: PathBuf,
        base_config: ServerConfig,
        models_max: usize,
        autoload_default: bool,
    ) -> anyhow::Result<Self> {
        let (events, _) = tokio::sync::broadcast::channel(256);
        let pool = Self {
            entries: RwLock::new(BTreeMap::new()),
            models_dir,
            base_config,
            models_max,
            autoload_default,
            events,
            load_lock: tokio::sync::Mutex::new(()),
        };
        pool.rescan()?;
        Ok(pool)
    }

    /// Scan `models_dir` for checkpoint directories and reconcile the
    /// registry (b10621 `load_models`). New directories appear as unloaded
    /// entries; removed directories drop their entry unless the model is
    /// still running.
    pub fn rescan(&self) -> anyhow::Result<()> {
        let discovered = discover_models(&self.models_dir)?;
        let mut entries = self
            .entries
            .write()
            .map_err(|_| anyhow::anyhow!("router pool poisoned"))?;
        for (name, path) in &discovered {
            entries.entry(name.clone()).or_insert_with(|| {
                Arc::new(RouterModelEntry {
                    name: name.clone(),
                    path: path.clone(),
                    state: Mutex::new(EntryState::default()),
                    last_used: AtomicI64::new(0),
                })
            });
        }
        entries.retain(|name, entry| discovered.contains_key(name) || entry.is_running());
        drop(entries);
        self.notify("models_reload", "*", serde_json::Value::Null);
        Ok(())
    }

    /// Subscribe to the model-event stream (`GET /models/sse`).
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<serde_json::Value> {
        self.events.subscribe()
    }

    /// b10621 `notify_sse` event shape: `{"model", "event"[, "data"]}`.
    fn notify(&self, event: &str, model: &str, data: serde_json::Value) {
        let mut payload = serde_json::json!({ "model": model, "event": event });
        if !data.is_null() {
            payload["data"] = data;
        }
        let _ = self.events.send(payload);
    }

    fn notify_status(&self, entry: &RouterModelEntry) {
        let mut data = serde_json::json!({ "status": entry.status().as_str() });
        if entry.failed() {
            data["failed"] = true.into();
        }
        self.notify("status_change", &entry.name, data);
    }

    /// Resolve a request's model name (b10621 `router_validate_model`):
    /// resolves through the registry only, so a path can never be smuggled
    /// in, and enforces the not-loaded refusal when autoload is off.
    pub fn resolve(
        &self,
        name: &str,
        autoload: bool,
    ) -> Result<Arc<RouterModelEntry>, RouterPoolError> {
        if name.is_empty() {
            return Err(RouterPoolError::MissingName);
        }
        let entry = self
            .entries
            .read()
            .ok()
            .and_then(|entries| entries.get(name).cloned())
            .ok_or_else(|| RouterPoolError::NotFound(name.to_string()))?;
        if !autoload && !entry.is_running() {
            return Err(RouterPoolError::NotLoaded);
        }
        Ok(entry)
    }

    pub fn get(&self, name: &str) -> Option<Arc<RouterModelEntry>> {
        self.entries.read().ok()?.get(name).cloned()
    }

    pub fn snapshot(&self) -> Vec<RouterModelSnapshot> {
        let entries = match self.entries.read() {
            Ok(entries) => entries,
            Err(_) => return Vec::new(),
        };
        entries
            .values()
            .map(|entry| {
                let (vision, audio) = entry
                    .state
                    .lock()
                    .ok()
                    .and_then(|guard| {
                        guard.app.as_ref().map(|app| {
                            (app.state.media_support.image, app.state.media_support.audio)
                        })
                    })
                    .unwrap_or_else(|| {
                        let support = super::startup::detect_model_media_support(&entry.path);
                        (support.image, support.audio)
                    });
                RouterModelSnapshot {
                    name: entry.name.clone(),
                    status: entry.status(),
                    failed: entry.failed(),
                    vision,
                    audio,
                }
            })
            .collect()
    }

    fn running_count(&self) -> usize {
        self.entries
            .read()
            .map(|entries| entries.values().filter(|e| e.is_running()).count())
            .unwrap_or(0)
    }

    /// Begin loading `name` (b10621 `server_models::load`), evicting the LRU
    /// loaded model first when `--models-max` is reached. Returns without
    /// waiting; [`Self::ensure_ready`] waits.
    pub async fn begin_load(&self, name: &str) -> Result<Arc<RouterModelEntry>, RouterPoolError> {
        let entry = self
            .get(name)
            .ok_or_else(|| RouterPoolError::NotFound(name.to_string()))?;
        let _permit = self.load_lock.lock().await;
        if entry.is_running() {
            return Ok(entry);
        }

        // Capacity: evict least-recently-used loaded entries until a slot
        // frees. A pool whose running entries are all still loading cannot be
        // evicted from and refuses the new load instead of thrashing.
        if self.models_max > 0 {
            while self.running_count() >= self.models_max {
                let victim = self
                    .entries
                    .read()
                    .ok()
                    .and_then(|entries| {
                        entries
                            .values()
                            .filter(|e| e.status() == RouterModelStatus::Loaded)
                            .min_by_key(|e| e.last_used.load(Ordering::Relaxed))
                            .cloned()
                    })
                    .ok_or_else(|| {
                        RouterPoolError::Capacity(format!(
                            "models_max ({}) reached and every loaded model is busy loading",
                            self.models_max
                        ))
                    })?;
                tracing::info!(
                    "router: evicting LRU model '{}' to load '{}' (models_max {})",
                    victim.name,
                    name,
                    self.models_max
                );
                self.unload_entry(&victim);
            }
        }

        // Construct the sub-app. The provider constructor returns fast (the
        // weights load on the worker thread), which is what makes `loading`
        // an observable state.
        match build_model_app(&entry.name, &entry.path, &self.base_config) {
            Ok((state, router)) => {
                if let Ok(mut guard) = entry.state.lock() {
                    guard.app = Some(LoadedApp { state, router });
                    guard.failed = false;
                }
                entry
                    .last_used
                    .store(chrono::Utc::now().timestamp_millis(), Ordering::Relaxed);
                // b10621 emits `model_status` when the instance is placed;
                // the loaded/failed transition later arrives as
                // `status_change` (see `notify_status`).
                self.notify(
                    "model_status",
                    &entry.name,
                    serde_json::json!({ "status": entry.status().as_str() }),
                );
                Ok(entry)
            }
            Err(err) => {
                if let Ok(mut guard) = entry.state.lock() {
                    guard.app = None;
                    guard.failed = true;
                }
                self.notify_status(&entry);
                Err(RouterPoolError::LoadFailed(err.to_string()))
            }
        }
    }

    /// Wait until `name` is loaded (b10621 `ensure_model_ready`), beginning
    /// the load when needed. Bounded by `timeout`.
    pub async fn ensure_ready(
        &self,
        name: &str,
        timeout: std::time::Duration,
    ) -> Result<Arc<RouterModelEntry>, RouterPoolError> {
        let entry = self.begin_load(name).await?;
        let started = std::time::Instant::now();
        loop {
            match entry.status() {
                RouterModelStatus::Loaded => {
                    self.notify_status(&entry);
                    return Ok(entry);
                }
                RouterModelStatus::Unloaded => {
                    self.notify_status(&entry);
                    return Err(RouterPoolError::LoadFailed(format!(
                        "model '{name}' failed to load"
                    )));
                }
                RouterModelStatus::Loading => {
                    if started.elapsed() > timeout {
                        return Err(RouterPoolError::LoadFailed(format!(
                            "model '{name}' did not become ready within {}s",
                            timeout.as_secs()
                        )));
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
        }
    }

    fn unload_entry(&self, entry: &RouterModelEntry) {
        if let Ok(mut guard) = entry.state.lock() {
            // Dropping the pool's AppState drops its ModelProvider sender;
            // the worker thread exits on channel disconnect and the weights
            // free. In-flight requests hold their own AppState clone, so
            // they finish before the last Arc drops: graceful by refcount.
            guard.app = None;
            guard.failed = false;
        }
        self.notify_status(entry);
    }

    /// Unload `name` (b10621 `server_models::unload` through
    /// `POST /models/unload`).
    pub fn unload(&self, name: &str) -> Result<(), RouterPoolError> {
        let entry = self
            .get(name)
            .ok_or_else(|| RouterPoolError::NotFound(name.to_string()))?;
        if !entry.is_running() {
            return Err(RouterPoolError::NotLoaded);
        }
        self.unload_entry(&entry);
        Ok(())
    }

    /// Route a request into `entry`'s sub-app.
    pub async fn dispatch(
        &self,
        entry: &RouterModelEntry,
        request: axum::http::Request<axum::body::Body>,
        stamp_last_used: bool,
    ) -> axum::response::Response {
        use tower::ServiceExt;
        if stamp_last_used {
            entry
                .last_used
                .store(chrono::Utc::now().timestamp_millis(), Ordering::Relaxed);
        }
        let Some(router) = entry.router() else {
            return super::routes::slots::llama_invalid_request("model is not loaded");
        };
        match router.oneshot(request).await {
            Ok(response) => response,
            Err(err) => match err {},
        }
    }
}

/// Build the per-model serving stack: tokenizer, chat template, provider,
/// [`AppState`], and the model's own axum app (without the CORS layer; the
/// router's top level owns CORS so headers are emitted exactly once).
fn build_model_app(
    name: &str,
    model_path: &Path,
    base_config: &ServerConfig,
) -> anyhow::Result<(AppState, axum::Router)> {
    let mut config = base_config.clone();
    // The name a child answers with is its directory name; the router's own
    // --alias never leaks into children, the b10621 preset-strip rule.
    config.model_alias = Some(name.to_string());
    config.model_aliases = vec![name.to_string()];

    let tokenizer = crate::tokenizer::load_tokenizer(model_path)?;
    let chat_template = ChatTemplateProcessor::from_model_path(model_path)?.unwrap_or_default();
    let batch_metrics = Arc::new(super::state::BatchMetrics::new());
    let batch_observability = Arc::new(super::batch::BatchObservability::new());
    let provider = ModelProvider::new_with_server_config(
        model_path.to_path_buf(),
        None,
        &config,
        batch_metrics.clone(),
        batch_observability.clone(),
    )?;
    let media_support = super::startup::detect_model_media_support(model_path);
    let state = AppState::with_observability(
        Arc::new(provider),
        config,
        chat_template,
        tokenizer,
        model_path.to_path_buf(),
        batch_metrics,
        batch_observability,
    )
    .with_media_support(media_support);
    let router = super::app::create_app_without_cors(state.clone());
    Ok((state, router))
}

/// Discover checkpoint directories directly under `models_dir`.
///
/// A model is a directory containing `config.json`. The name is the
/// directory's file name. An entry whose canonical path escapes the
/// canonical models directory (a symlink pointing outside it) is skipped
/// with a warning: the models directory is the confinement boundary the
/// router promises (#1438).
pub fn discover_models(models_dir: &Path) -> anyhow::Result<BTreeMap<String, PathBuf>> {
    let canonical_root = models_dir.canonicalize().map_err(|e| {
        anyhow::anyhow!(
            "--models-dir {}: cannot open models directory: {e}",
            models_dir.display()
        )
    })?;
    let mut found = BTreeMap::new();
    for entry in std::fs::read_dir(&canonical_root)? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let Ok(canonical) = path.canonicalize() else {
            continue;
        };
        if !canonical.starts_with(&canonical_root) {
            tracing::warn!(
                "router: skipping '{name}': resolves outside the models directory ({})",
                canonical.display()
            );
            continue;
        }
        if !canonical.is_dir() || !canonical.join("config.json").is_file() {
            continue;
        }
        found.insert(name, canonical);
    }
    Ok(found)
}

#[cfg(test)]
#[path = "router_models_tests.rs"]
mod router_models_tests;
