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

//! Router-mode model pool (llama-server b10621 compatible, issue #1438).
//!
//! b10621's router server discovers models from three sources: its download
//! cache (removable, `POST /models` downloads into it), the `--models-dir`
//! directory, and `--models-preset` INI sections, with name collisions
//! resolved cache < models-dir < preset. It spawns one child llama-server
//! process per loaded model and proxies requests to the child the request's
//! `model` field names. mlxcel serves the same HTTP surface from one
//! process: each pool entry, once loaded, owns a full [`AppState`] plus its
//! axum [`Router`], and the dispatcher forwards the request into that
//! sub-app in-process instead of over a child socket. The state machine
//! mirrors upstream's `UNLOADED -> LOADING -> LOADED` plus the transient
//! `DOWNLOADING` while `POST /models` fetches a repository into the cache (a
//! failed load returns to `unloaded` with a failure recorded, which is
//! b10621's failed shape), `--models-max` bounds the concurrently loaded set
//! with LRU eviction, and `--models-autoload` (plus the per-request
//! `?autoload=` override) loads on demand.
//!
//! Confinement: entry names come only from discovery, the cache store, or
//! operator-authored presets; requests resolve names through the registry (a
//! request can never smuggle a path); a discovered entry whose canonical
//! path escapes the canonical models directory (a symlink pointing outside
//! it) is skipped at scan time; and cache downloads/removals go through the
//! store's sanitized `<owner>/<name>` composition with containment
//! re-asserted before any deletion (see [`super::router_cache`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use axum::body::Body;
use futures::StreamExt;

use super::model_provider::WorkerExitObserver;
use super::router_cache::CacheSource;
use super::router_lifecycle::{
    DownloadState, ErrorBody, LifecycleCoordinator, LifecycleSnapshot, ModelLifecycle,
    ModelLifecycleState, OperationError, OperationKind, OperationResult, OperationState,
    OperationTarget, stable_model_identity,
};
use super::router_presets::{PresetCliOverrides, PresetSection, RouterPresets};
use super::{AppState, ChatTemplateProcessor, ModelProvider, ServerStartupConfig};

const ROUTER_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const ROUTER_WORKER_EXIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
const ROUTER_LOAD_READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);
pub const ROUTER_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(150);

#[cfg(test)]
type RescanAfterSnapshotHook = Arc<dyn Fn() + Send + Sync + 'static>;

/// b10621 `server_model_status` (the subset an in-process pool reaches;
/// `downloaded` is upstream's "erase on next reload" marker, which the
/// in-process pool replaces with an immediate rescan).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterModelStatus {
    Unloaded,
    Loading,
    Loaded,
    Downloading,
}

impl RouterModelStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unloaded => "unloaded",
            Self::Loading => "loading",
            Self::Loaded => "loaded",
            Self::Downloading => "downloading",
        }
    }
}

/// b10621 `server_model_source`: where an entry came from, which decides
/// `can_remove` (only cache entries are removable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterModelSource {
    Cache,
    ModelsDir,
    Preset,
}

impl RouterModelSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cache => "cache",
            Self::ModelsDir => "models_dir",
            Self::Preset => "preset",
        }
    }
}

/// One discovered model.
pub struct RouterModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub source: RouterModelSource,
    /// Preset aliases (b10621 lists them on the model object and resolves
    /// them for request routing).
    pub aliases: Vec<String>,
    pub tags: Vec<String>,
    pub ui_model_id: String,
    pub source_key_hash: String,
    /// Hidden by a preset's `dedup-cache-models` (still resolvable by name).
    pub hidden: bool,
    /// The preset section that shaped this entry, for the `status.preset`
    /// INI block; `None` when no named section applies.
    preset: Option<PresetSection>,
    /// Per-model server config: the router's own CLI config overlaid with
    /// this model's preset (issue #1438).
    config: super::config::ServerConfig,
    state: Mutex<EntryState>,
    lifecycle: Arc<ModelLifecycle>,
    last_used: AtomicI64,
}

#[derive(Default)]
struct EntryState {
    /// The loaded sub-app; `Some` while loading or loaded.
    app: Option<LoadedApp>,
    /// Set when the last load or download failed; cleared by the next
    /// attempt.
    failed: bool,
    /// `Some` while `POST /models` is downloading this entry into the cache.
    download: Option<DownloadInFlight>,
}

struct DownloadInFlight {
    cancel: Arc<AtomicBool>,
    /// b10621 `loaded_info` during a download:
    /// `{"progress": {url: {"done": n, "total": n}}}`.
    progress: serde_json::Value,
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
        if guard.download.is_some() {
            return RouterModelStatus::Downloading;
        }
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

    /// b10621 `is_running`: loading or loaded (a downloading entry is not
    /// running; it has no weights to serve).
    pub fn is_running(&self) -> bool {
        matches!(
            self.status(),
            RouterModelStatus::Loading | RouterModelStatus::Loaded
        )
    }

    pub fn reserves_capacity(&self) -> bool {
        self.lifecycle.reserves_capacity()
    }

    fn worker_exit_observer(&self) -> Option<Arc<WorkerExitObserver>> {
        self.state.lock().ok().and_then(|guard| {
            guard
                .app
                .as_ref()
                .map(|app| app.state.model_provider.worker_exit_observer())
        })
    }

    fn worker_exit_observed(&self) -> bool {
        self.worker_exit_observer()
            .as_ref()
            .map(|observer| observer.observed())
            .unwrap_or(true)
    }

    pub fn is_downloading(&self) -> bool {
        self.state
            .lock()
            .map(|guard| guard.download.is_some())
            .unwrap_or(false)
    }

    fn router(&self) -> Option<axum::Router> {
        self.state
            .lock()
            .ok()?
            .app
            .as_ref()
            .map(|app| app.router.clone())
    }

    pub fn lifecycle_snapshot(&self) -> LifecycleSnapshot {
        self.lifecycle.snapshot()
    }

    pub fn lifecycle_revision(&self) -> u64 {
        self.lifecycle.revision()
    }

    fn download_progress_json(&self) -> Option<serde_json::Value> {
        self.state
            .lock()
            .ok()?
            .download
            .as_ref()
            .map(|d| d.progress.clone())
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
    pub source: RouterModelSource,
    pub aliases: Vec<String>,
    pub tags: Vec<String>,
    pub hidden: bool,
    /// `{"progress": {...}}` while downloading (b10621 `loaded_info`).
    pub download_info: Option<serde_json::Value>,
    /// The preset section as INI text (b10621 `status.preset`).
    pub preset_ini: Option<String>,
}

/// Model sources the pool reconciles on every rescan (issue #1438).
#[derive(Default)]
pub struct RouterSources {
    /// `--models-dir` discovery root.
    pub models_dir: Option<PathBuf>,
    /// The model cache (mlxcel model store) with its downloader.
    pub cache: Option<CacheSource>,
    /// Parsed `--models-preset` sections.
    pub presets: RouterPresets,
}

/// The pool shared by every router-mode handler.
pub struct RouterPool {
    entries: RwLock<BTreeMap<String, Arc<RouterModelEntry>>>,
    sources: RouterSources,
    /// Template for each per-model [`ServerStartupConfig`]; the router's own
    /// CLI arguments apply to every model (the b10621 base-preset overlay),
    /// and each model's preset section overlays underneath them.
    base_startup: ServerStartupConfig,
    api_keys: super::ApiKeys,
    cli_overrides: PresetCliOverrides,
    pub models_max: usize,
    pub autoload_default: bool,
    events: tokio::sync::broadcast::Sender<serde_json::Value>,
    lifecycle: Arc<LifecycleCoordinator>,
    /// Serializes model loads so two concurrent autoloads cannot race the
    /// capacity check or contend the accelerator during weight upload.
    load_lock: tokio::sync::Mutex<()>,
    #[cfg(test)]
    rescan_after_snapshot_hook: Mutex<Option<RescanAfterSnapshotHook>>,
}

/// Why a name failed to resolve, load, download, or be removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterPoolError {
    MissingName,
    NotFound(String),
    NotLoaded,
    LoadFailed(String),
    Capacity(String),
    /// b10621: only cache-sourced models are removable.
    NotRemovable(String),
    /// `POST /models` on a name the pool already has.
    AlreadyExists(String),
    /// A WebUI operation was recorded but rejected before background execution.
    OperationRejected(ErrorBody),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterModelAction {
    Load,
    Unload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterShutdownReport {
    pub attempted: usize,
    pub completed: Vec<String>,
    pub remaining: Vec<String>,
    pub timed_out: bool,
}

impl RouterPool {
    pub fn new(
        sources: RouterSources,
        base_startup: ServerStartupConfig,
        api_keys: super::ApiKeys,
        cli_overrides: PresetCliOverrides,
        models_max: usize,
        autoload_default: bool,
    ) -> anyhow::Result<Self> {
        let (events, _) = tokio::sync::broadcast::channel(256);
        let pool = Self {
            entries: RwLock::new(BTreeMap::new()),
            sources,
            base_startup,
            api_keys,
            cli_overrides,
            models_max,
            autoload_default,
            events,
            lifecycle: Arc::new(LifecycleCoordinator::new()),
            load_lock: tokio::sync::Mutex::new(()),
            #[cfg(test)]
            rescan_after_snapshot_hook: Mutex::new(None),
        };
        pool.rescan()?;
        Ok(pool)
    }

    #[cfg(test)]
    pub(super) fn set_rescan_after_snapshot_hook(&self, hook: Option<RescanAfterSnapshotHook>) {
        *self
            .rescan_after_snapshot_hook
            .lock()
            .expect("rescan hook mutex poisoned") = hook;
    }

    /// Build the per-model [`super::config::ServerConfig`] by overlaying the
    /// entry's preset section onto a clone of the router's startup config and
    /// re-running the CLI's own resolution pipeline, so preset keys and CLI
    /// flags resolve identically (per-model `generation_config.json` sampling
    /// defaults included).
    fn build_entry_config(
        &self,
        name: &str,
        path: &Path,
        section: &PresetSection,
    ) -> super::config::ServerConfig {
        let mut startup = self.base_startup.clone();
        startup.model_path = path.to_path_buf();
        super::router_presets::apply_section_to_startup(&mut startup, section, &self.cli_overrides);
        let mut config = super::startup::build_server_config(&startup, self.api_keys.clone());
        // The name a model answers with is its directory name / repo id; the
        // router's own --alias never leaks into models, the b10621
        // preset-strip rule. Preset aliases do apply.
        config.model_alias = Some(name.to_string());
        let mut aliases = vec![name.to_string()];
        aliases.extend(section.aliases.iter().cloned());
        config.model_aliases = aliases;
        config
    }

    fn ui_identity_for(&self, name: &str, source: RouterModelSource) -> (String, String) {
        let (source_key, rank) = match source {
            RouterModelSource::Cache => ("cache:mlxcel-managed-cache-v1".to_string(), 0),
            RouterModelSource::ModelsDir => ("models_dir:primary-redacted-root".to_string(), 1),
            RouterModelSource::Preset => (format!("preset:{name}"), 2),
        };
        stable_model_identity(source.as_str(), rank, &source_key, name)
    }

    /// Scan every source and reconcile the registry (b10621 `load_models`):
    /// cache snapshots, then `--models-dir` directories (overriding cache
    /// names), then preset sections (defining new models or re-sourcing
    /// discovered ones). New names appear as unloaded entries; removed names
    /// drop their entry unless the model is still running or downloading.
    pub fn rescan(&self) -> anyhow::Result<()> {
        // Phase 1: enumerate sources and build replacement entries without
        // holding the registry lock (config building reads the checkpoint's
        // generation_config.json).
        let mut discovered: BTreeMap<String, (PathBuf, RouterModelSource)> = BTreeMap::new();
        if let Some(cache) = &self.sources.cache {
            for (name, path) in cache.list() {
                discovered.insert(name, (path, RouterModelSource::Cache));
            }
        }
        if let Some(dir) = &self.sources.models_dir {
            for (name, path) in discover_models(dir)? {
                discovered.insert(name, (path, RouterModelSource::ModelsDir));
            }
        }
        for (name, section) in &self.sources.presets.models {
            if let Some(path) = &section.model_path {
                discovered.insert(name.clone(), (path.clone(), RouterModelSource::Preset));
            } else if let Some(repo) = &section.hf_repo {
                let Some(cache) = &self.sources.cache else {
                    tracing::warn!(
                        "router: preset '[{name}]' names hf-repo '{repo}' but no model cache \
                         is configured; skipping"
                    );
                    continue;
                };
                let path = cache.snapshot_dir(repo);
                if path.join("config.json").is_file() {
                    discovered.insert(name.clone(), (path, RouterModelSource::Preset));
                } else {
                    tracing::warn!(
                        "router: preset '[{name}]' names hf-repo '{repo}', which is not in the \
                         cache; download it first (POST /models or `mlxcel download {repo}`)"
                    );
                }
            } else if let Some((path, _)) = discovered.get(name.as_str()).cloned() {
                // Overlay-only section: the model keeps its discovered path
                // but is re-sourced as preset, upstream's merge rule.
                discovered.insert(name.clone(), (path, RouterModelSource::Preset));
            } else {
                tracing::warn!(
                    "router: preset '[{name}]' names no checkpoint (model= / hf-repo=) and \
                     matches no discovered model"
                );
            }
        }

        // A preset with `dedup-cache-models` hides cache entries that resolve
        // to the snapshot the preset itself serves.
        let mut hidden_names: Vec<String> = Vec::new();
        for (pname, section) in &self.sources.presets.models {
            if !section.dedup_cache_models {
                continue;
            }
            let preset_path = discovered.get(pname.as_str()).map(|(p, _)| p.clone());
            let Some(preset_path) = preset_path else {
                continue;
            };
            for (cname, (cpath, csource)) in &discovered {
                if cname != pname && *csource == RouterModelSource::Cache && *cpath == preset_path {
                    hidden_names.push(cname.clone());
                }
            }
        }

        let existing: BTreeMap<String, Arc<RouterModelEntry>> = self
            .entries
            .read()
            .map_err(|_| anyhow::anyhow!("router pool poisoned"))?
            .clone();

        #[cfg(test)]
        if let Some(hook) = self
            .rescan_after_snapshot_hook
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
        {
            hook();
        }

        let mut rebuilt: BTreeMap<String, Arc<RouterModelEntry>> = BTreeMap::new();
        for (name, (path, source)) in &discovered {
            if let Some(entry) = existing.get(name)
                && (entry.reserves_capacity() || entry.is_downloading())
            {
                // A running model keeps serving its current configuration;
                // source changes apply on its next load (upstream unloads on
                // preset change, which the in-process pool defers to the
                // operator's own unload/load cycle).
                rebuilt.insert(name.clone(), entry.clone());
                continue;
            }
            let section = self.sources.presets.for_model(name);
            let config = self.build_entry_config(name, path, &section);
            let (ui_model_id, source_key_hash) = self.ui_identity_for(name, *source);
            rebuilt.insert(
                name.clone(),
                Arc::new(RouterModelEntry {
                    name: name.clone(),
                    path: path.clone(),
                    source: *source,
                    aliases: section.aliases.clone(),
                    tags: section.tags.clone(),
                    ui_model_id,
                    source_key_hash,
                    hidden: hidden_names.contains(name),
                    preset: self.sources.presets.models.get(name).cloned(),
                    config,
                    state: Mutex::new(EntryState::default()),
                    lifecycle: Arc::new(ModelLifecycle::new(DownloadState::Complete)),
                    last_used: AtomicI64::new(0),
                }),
            );
        }
        // Keep running or downloading entries whose source vanished (or, for
        // downloads, never existed yet).
        for (name, entry) in &existing {
            if !rebuilt.contains_key(name) && (entry.reserves_capacity() || entry.is_downloading())
            {
                rebuilt.insert(name.clone(), entry.clone());
            }
        }

        let mut entries = self
            .entries
            .write()
            .map_err(|_| anyhow::anyhow!("router pool poisoned"))?;
        for (name, current) in entries.iter() {
            if current.reserves_capacity() || current.is_downloading() {
                rebuilt.insert(name.clone(), current.clone());
            }
        }
        *entries = rebuilt;
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
        if let Some(progress) = entry.download_progress_json() {
            // b10621 carries the download progress as the entry's
            // `loaded_info`/`progress` blocks on `status_change`.
            data["info"] = progress;
        }
        self.notify("status_change", &entry.name, data);
    }

    pub fn lifecycle_coordinator(&self) -> Arc<LifecycleCoordinator> {
        self.lifecycle.clone()
    }

    fn notify_lifecycle(&self, entry: &RouterModelEntry) {
        self.lifecycle.publish_model_revision(
            &entry.ui_model_id,
            entry.lifecycle.revision(),
            entry.lifecycle_snapshot(),
        );
    }

    /// Resolve a request's model name (b10621 `router_validate_model`):
    /// resolves through the registry only (name or preset alias), so a path
    /// can never be smuggled in, and enforces the not-loaded refusal when
    /// autoload is off.
    pub fn resolve(
        &self,
        name: &str,
        autoload: bool,
    ) -> Result<Arc<RouterModelEntry>, RouterPoolError> {
        if name.is_empty() {
            return Err(RouterPoolError::MissingName);
        }
        let entry = self
            .lookup(name)
            .ok_or_else(|| RouterPoolError::NotFound(name.to_string()))?;
        if !autoload && !entry.is_running() {
            return Err(RouterPoolError::NotLoaded);
        }
        Ok(entry)
    }

    /// Exact-name lookup (load/unload/delete address models by name only).
    pub fn get(&self, name: &str) -> Option<Arc<RouterModelEntry>> {
        self.entries.read().ok()?.get(name).cloned()
    }

    pub fn get_by_model_id(&self, model_id: &str) -> Option<Arc<RouterModelEntry>> {
        self.entries
            .read()
            .ok()?
            .values()
            .find(|entry| entry.ui_model_id == model_id)
            .cloned()
    }

    /// Name-or-alias lookup (request routing).
    pub fn lookup(&self, name: &str) -> Option<Arc<RouterModelEntry>> {
        let entries = self.entries.read().ok()?;
        if let Some(entry) = entries.get(name) {
            return Some(entry.clone());
        }
        entries
            .values()
            .find(|e| e.aliases.iter().any(|a| a == name))
            .cloned()
    }

    pub fn runtime_model_ids(&self) -> Vec<String> {
        self.entries
            .read()
            .map(|entries| {
                entries
                    .values()
                    .filter(|entry| entry.reserves_capacity() || entry.is_downloading())
                    .map(|entry| entry.ui_model_id.clone())
                    .collect()
            })
            .unwrap_or_default()
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
                    source: entry.source,
                    aliases: entry.aliases.clone(),
                    tags: entry.tags.clone(),
                    hidden: entry.hidden,
                    download_info: entry.download_progress_json(),
                    preset_ini: entry
                        .preset
                        .as_ref()
                        .map(|section| section.to_ini(&entry.name)),
                }
            })
            .collect()
    }

    /// Entries whose preset asked for `load-on-startup`.
    pub fn load_on_startup_names(&self) -> Vec<String> {
        self.sources
            .presets
            .models
            .iter()
            .filter(|(_, section)| section.load_on_startup)
            .map(|(name, _)| name.clone())
            .collect()
    }

    fn running_count(&self) -> usize {
        self.entries
            .read()
            .map(|entries| entries.values().filter(|e| e.reserves_capacity()).count())
            .unwrap_or(0)
    }

    /// Begin loading `name` (b10621 `server_models::load`), evicting the LRU
    /// loaded model first when `--models-max` is reached. Returns without
    /// waiting; [`Self::ensure_ready`] waits.
    pub async fn begin_load(&self, name: &str) -> Result<Arc<RouterModelEntry>, RouterPoolError> {
        self.begin_load_with_policy(name, None, true, None).await
    }

    async fn begin_load_with_policy(
        &self,
        name: &str,
        eviction_target: Option<&str>,
        allow_lru_eviction: bool,
        expected_revision: Option<u64>,
    ) -> Result<Arc<RouterModelEntry>, RouterPoolError> {
        let entry = self
            .get(name)
            .ok_or_else(|| RouterPoolError::NotFound(name.to_string()))?;
        if entry.is_downloading() {
            return Err(RouterPoolError::LoadFailed(format!(
                "model '{name}' is still downloading"
            )));
        }
        let _permit = self.load_lock.lock().await;
        let _entry_permit = entry.lifecycle.operation_guard().await;
        reject_stale_revision(&entry, expected_revision)?;
        if entry.is_running() {
            return Ok(entry.clone());
        }
        if entry.reserves_capacity() {
            return Err(RouterPoolError::Capacity(format!(
                "model '{name}' still owns resources; worker exit has not been observed"
            )));
        }

        // Capacity: evict least-recently-used loaded entries until a slot
        // frees. A pool whose running entries are all still loading cannot be
        // evicted from and refuses the new load instead of thrashing.
        if self.models_max > 0 {
            let mut explicit_victim = eviction_target.map(ToString::to_string);
            while self.running_count() >= self.models_max {
                let victim = if let Some(target_name) = explicit_victim.take() {
                    let victim = self
                        .get(&target_name)
                        .ok_or_else(|| RouterPoolError::NotFound(target_name.clone()))?;
                    if victim.name == entry.name {
                        return Err(RouterPoolError::Capacity(
                            "eviction target cannot be the model being loaded".to_string(),
                        ));
                    }
                    if victim.lifecycle_snapshot().active_requests > 0
                        || victim.lifecycle.state() != ModelLifecycleState::Ready
                    {
                        return Err(RouterPoolError::Capacity(format!(
                            "eviction target '{}' is busy",
                            victim.name
                        )));
                    }
                    victim
                } else if allow_lru_eviction {
                    self.entries
                        .read()
                        .ok()
                        .and_then(|entries| {
                            entries
                                .values()
                                .filter(|e| {
                                    e.status() == RouterModelStatus::Loaded
                                        && e.lifecycle_snapshot().active_requests == 0
                                        && e.lifecycle.state() == ModelLifecycleState::Ready
                                })
                                .min_by_key(|e| e.last_used.load(Ordering::Relaxed))
                                .cloned()
                        })
                        .ok_or_else(|| {
                            RouterPoolError::Capacity(format!(
                                "models_max ({}) reached and every loaded model is busy",
                                self.models_max
                            ))
                        })?
                } else {
                    return Err(RouterPoolError::Capacity(format!(
                        "models_max ({}) reached; provide an explicit eviction target",
                        self.models_max
                    )));
                };
                tracing::info!(
                    "router: evicting model '{}' to load '{}' (models_max {})",
                    victim.name,
                    name,
                    self.models_max
                );
                self.unload_entry(&victim).await?;
            }
        }

        entry.lifecycle.mark_loading();
        self.notify_lifecycle(&entry);

        // Construct the sub-app. The provider constructor returns fast (the
        // weights load on the worker thread), which is what makes `loading`
        // an observable state; the tokenizer and chat-template reads are
        // still filesystem work, so they run on the blocking pool rather
        // than stalling the async runtime.
        let (path, config) = (entry.path.clone(), entry.config.clone());
        let built = tokio::task::spawn_blocking(move || build_model_app(&path, config))
            .await
            .unwrap_or_else(|join_err| Err(anyhow::anyhow!(join_err.to_string())));
        match built {
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
                self.spawn_ready_monitor(entry.clone());
                Ok(entry.clone())
            }
            Err(err) => {
                if let Ok(mut guard) = entry.state.lock() {
                    guard.app = None;
                    guard.failed = true;
                }
                entry.lifecycle.mark_failed(err.to_string(), true);
                self.notify_lifecycle(&entry);
                self.notify_status(&entry);
                Err(RouterPoolError::LoadFailed(err.to_string()))
            }
        }
    }

    fn spawn_ready_monitor(&self, entry: Arc<RouterModelEntry>) {
        let lifecycle = self.lifecycle.clone();
        tokio::spawn(async move {
            loop {
                match entry.status() {
                    RouterModelStatus::Loaded => {
                        if entry.lifecycle.state() == ModelLifecycleState::Loading {
                            entry.lifecycle.mark_ready();
                            lifecycle.publish_model_revision(
                                &entry.ui_model_id,
                                entry.lifecycle.revision(),
                                entry.lifecycle_snapshot(),
                            );
                        }
                        break;
                    }
                    RouterModelStatus::Unloaded => {
                        if entry.lifecycle.state() == ModelLifecycleState::Loading {
                            let observer = entry.worker_exit_observer();
                            let worker_observed = observer
                                .as_ref()
                                .map(|observer| observer.observed())
                                .unwrap_or(true);
                            entry
                                .lifecycle
                                .mark_failed("model worker failed during load", worker_observed);
                            lifecycle.publish_model_revision(
                                &entry.ui_model_id,
                                entry.lifecycle.revision(),
                                entry.lifecycle_snapshot(),
                            );
                            if let Some(observer) = observer
                                && !worker_observed
                            {
                                wait_for_failed_worker_exit(
                                    entry.clone(),
                                    lifecycle.clone(),
                                    observer,
                                )
                                .await;
                            }
                        }
                        break;
                    }
                    RouterModelStatus::Loading => {
                        if entry.lifecycle.state() != ModelLifecycleState::Loading {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                    RouterModelStatus::Downloading => break,
                }
            }
        });
    }

    /// Wait until `name` is loaded (b10621 `ensure_model_ready`), beginning
    /// the load when needed. Bounded by `timeout`.
    pub async fn ensure_ready(
        &self,
        name: &str,
        timeout: std::time::Duration,
    ) -> Result<Arc<RouterModelEntry>, RouterPoolError> {
        let entry = self.begin_load(name).await?;
        self.wait_for_ready_entry(entry, name, timeout).await
    }

    async fn wait_for_ready_entry(
        &self,
        entry: Arc<RouterModelEntry>,
        name: &str,
        timeout: std::time::Duration,
    ) -> Result<Arc<RouterModelEntry>, RouterPoolError> {
        let started = std::time::Instant::now();
        loop {
            match entry.status() {
                RouterModelStatus::Loaded => {
                    entry.lifecycle.mark_ready();
                    self.notify_lifecycle(&entry);
                    self.notify_status(&entry);
                    return Ok(entry.clone());
                }
                RouterModelStatus::Unloaded | RouterModelStatus::Downloading => {
                    let worker_observed = entry.worker_exit_observed();
                    entry
                        .lifecycle
                        .mark_failed(format!("model '{name}' failed to load"), worker_observed);
                    self.notify_lifecycle(&entry);
                    self.notify_status(&entry);
                    return Err(RouterPoolError::LoadFailed(format!(
                        "model '{name}' failed to load"
                    )));
                }
                RouterModelStatus::Loading => {
                    if started.elapsed() > timeout {
                        let message = format!(
                            "model '{name}' did not become ready within {}s",
                            timeout.as_secs()
                        );
                        entry.lifecycle.mark_loading_blocked(message.clone());
                        self.notify_lifecycle(&entry);
                        self.notify_status(&entry);
                        return Err(RouterPoolError::LoadFailed(message));
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
        }
    }

    pub async fn shutdown_all(
        self: &Arc<Self>,
        timeout: std::time::Duration,
    ) -> RouterShutdownReport {
        let entries: Vec<_> = self
            .entries
            .read()
            .map(|entries| {
                entries
                    .values()
                    .filter(|entry| entry.reserves_capacity() || entry.is_downloading())
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        for entry in &entries {
            if entry.lifecycle.begin_drain() {
                self.notify_lifecycle(entry);
                self.notify_status(entry);
            }
        }
        let attempted = entries.len();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        for entry in entries {
            let pool = self.clone();
            let name = entry.name.clone();
            let tx = tx.clone();
            tokio::spawn(async move {
                let result = pool.unload(&name).await;
                let _ = tx.send((name, result));
            });
        }
        drop(tx);

        let deadline = tokio::time::sleep(timeout);
        tokio::pin!(deadline);
        let mut completed = Vec::new();
        let mut observed = 0usize;
        let mut timed_out = false;
        while observed < attempted {
            tokio::select! {
                message = rx.recv() => {
                    match message {
                        Some((name, Ok(()))) => {
                            observed += 1;
                            completed.push(name);
                        }
                        Some((_name, Err(_))) => {
                            observed += 1;
                        }
                        None => break,
                    }
                }
                _ = &mut deadline => {
                    timed_out = true;
                    break;
                }
            }
        }
        let remaining = self
            .entries
            .read()
            .map(|entries| {
                entries
                    .values()
                    .filter(|entry| entry.reserves_capacity() || entry.is_downloading())
                    .map(|entry| entry.name.clone())
                    .collect()
            })
            .unwrap_or_default();
        RouterShutdownReport {
            attempted,
            completed,
            remaining,
            timed_out,
        }
    }

    pub fn submit_model_action(
        self: &Arc<Self>,
        model_id: &str,
        action: RouterModelAction,
        expected_revision: u64,
        idempotency_key: &str,
        eviction_target_id: Option<&str>,
    ) -> Result<super::router_lifecycle::OperationAccepted, RouterPoolError> {
        let kind = match action {
            RouterModelAction::Load => OperationKind::ModelLoad,
            RouterModelAction::Unload => OperationKind::ModelUnload,
        };
        let fingerprint = format!(
            "{kind:?}:{model_id}:{expected_revision}:{}",
            eviction_target_id.unwrap_or("")
        );
        let accepted = self
            .lifecycle
            .begin_operation(
                kind,
                OperationTarget::Model {
                    model_id: model_id.to_string(),
                    requested_revision: Some(expected_revision),
                },
                Some(idempotency_key),
                fingerprint,
            )
            .map_err(|err| match err {
                OperationError::Conflict { .. } => {
                    RouterPoolError::LoadFailed("conflicting idempotency key".to_string())
                }
                OperationError::TooManyActive => {
                    RouterPoolError::Capacity("too many active lifecycle operations".to_string())
                }
            })?;
        if accepted.idempotent_replay {
            return Ok(accepted);
        }

        let Some(entry) = self.get_by_model_id(model_id) else {
            self.lifecycle.update_operation(
                &accepted.operation_id,
                OperationState::Failed,
                None,
                Some(ErrorBody {
                    code: "not_found".to_string(),
                    message: format!("model id '{model_id}' not found"),
                    retryable: true,
                    field_errors: None,
                    operation_id: Some(accepted.operation_id.clone()),
                }),
            );
            return Err(RouterPoolError::NotFound(model_id.to_string()));
        };
        if entry.lifecycle.revision() != expected_revision {
            let error = ErrorBody {
                code: "stale_revision".to_string(),
                message: "catalog entry revision changed; refresh before retrying".to_string(),
                retryable: true,
                field_errors: Some(vec![super::router_lifecycle::FieldError {
                    field: "expected_revision".to_string(),
                    code: "stale".to_string(),
                    message: format!(
                        "expected {expected_revision} but current revision is {}",
                        entry.lifecycle.revision()
                    ),
                }]),
                operation_id: Some(accepted.operation_id.clone()),
            };
            self.lifecycle.update_operation(
                &accepted.operation_id,
                OperationState::Failed,
                None,
                Some(error.clone()),
            );
            return Err(RouterPoolError::OperationRejected(error));
        }

        let eviction_target_name = match eviction_target_id {
            Some(target_id) => match self.get_by_model_id(target_id) {
                Some(target) => Some(target.name.clone()),
                None => {
                    self.lifecycle.update_operation(
                        &accepted.operation_id,
                        OperationState::Failed,
                        None,
                        Some(ErrorBody {
                            code: "not_found".to_string(),
                            message: format!("eviction target id '{target_id}' not found"),
                            retryable: true,
                            field_errors: None,
                            operation_id: Some(accepted.operation_id.clone()),
                        }),
                    );
                    return Err(RouterPoolError::NotFound(target_id.to_string()));
                }
            },
            None => None,
        };

        if let Some((err, code, message)) = match action {
            RouterModelAction::Load if entry.is_running() => Some((
                RouterPoolError::LoadFailed(format!("model '{}' is already running", entry.name)),
                "conflict",
                format!("model '{}' is already running", entry.name),
            )),
            RouterModelAction::Load if entry.reserves_capacity() => Some((
                RouterPoolError::Capacity(format!(
                    "model '{}' still owns resources; worker exit has not been observed",
                    entry.name
                )),
                "conflict",
                format!("model '{}' still owns resources", entry.name),
            )),
            RouterModelAction::Unload if !entry.reserves_capacity() && !entry.is_downloading() => {
                Some((
                    RouterPoolError::NotLoaded,
                    "invalid_request",
                    format!("model '{}' is not loaded", entry.name),
                ))
            }
            _ => None,
        } {
            self.lifecycle.update_operation(
                &accepted.operation_id,
                OperationState::Failed,
                None,
                Some(ErrorBody {
                    code: code.to_string(),
                    message,
                    retryable: true,
                    field_errors: None,
                    operation_id: Some(accepted.operation_id.clone()),
                }),
            );
            return Err(err);
        }
        let pool = self.clone();
        let operation_id = accepted.operation_id.clone();
        let model_name = entry.name.clone();
        tokio::spawn(async move {
            pool.lifecycle
                .update_operation(&operation_id, OperationState::Running, None, None);
            let result = match action {
                RouterModelAction::Load => match pool
                    .begin_load_with_policy(
                        &model_name,
                        eviction_target_name.as_deref(),
                        false,
                        Some(expected_revision),
                    )
                    .await
                {
                    Ok(entry) => pool
                        .wait_for_ready_entry(entry, &model_name, ROUTER_LOAD_READY_TIMEOUT)
                        .await
                        .map(|_| ()),
                    Err(err) => Err(err),
                },
                RouterModelAction::Unload => {
                    pool.unload_with_expected(&model_name, Some(expected_revision))
                        .await
                }
            };
            let entry = pool.get(&model_name);
            let lifecycle = entry.as_ref().map(|entry| entry.lifecycle_snapshot());
            match (result, lifecycle) {
                (Ok(()), Some(lifecycle)) => {
                    let model_id = entry
                        .as_ref()
                        .map(|e| e.ui_model_id.clone())
                        .unwrap_or(model_name.clone());
                    let revision = entry.as_ref().map(|e| e.lifecycle.revision()).unwrap_or(1);
                    let result = match action {
                        RouterModelAction::Load => OperationResult::ModelLoad {
                            model_id,
                            revision,
                            lifecycle,
                        },
                        RouterModelAction::Unload => OperationResult::ModelUnload {
                            model_id,
                            revision,
                            lifecycle,
                        },
                    };
                    pool.lifecycle.update_operation(
                        &operation_id,
                        OperationState::Succeeded,
                        Some(result),
                        None,
                    );
                }
                (Err(RouterPoolError::OperationRejected(mut error)), _) => {
                    if error.operation_id.is_none() {
                        error.operation_id = Some(operation_id.clone());
                    }
                    pool.lifecycle.update_operation(
                        &operation_id,
                        OperationState::Failed,
                        None,
                        Some(error),
                    );
                }
                (Err(err), _) => {
                    pool.lifecycle.update_operation(
                        &operation_id,
                        OperationState::Failed,
                        None,
                        Some(ErrorBody {
                            code: "conflict".to_string(),
                            message: format!("{err:?}"),
                            retryable: true,
                            field_errors: None,
                            operation_id: Some(operation_id.clone()),
                        }),
                    );
                }
                (Ok(()), None) => {
                    pool.lifecycle.update_operation(
                        &operation_id,
                        OperationState::Failed,
                        None,
                        Some(ErrorBody {
                            code: "not_found".to_string(),
                            message: "model disappeared before operation finished".to_string(),
                            retryable: true,
                            field_errors: None,
                            operation_id: Some(operation_id.clone()),
                        }),
                    );
                }
            }
        });
        Ok(accepted)
    }

    async fn unload_entry(&self, entry: &RouterModelEntry) -> Result<(), RouterPoolError> {
        self.unload_entry_with_expected(entry, None).await
    }

    async fn unload_entry_with_expected(
        &self,
        entry: &RouterModelEntry,
        expected_revision: Option<u64>,
    ) -> Result<(), RouterPoolError> {
        let _entry_permit = entry.lifecycle.operation_guard().await;
        reject_stale_revision(entry, expected_revision)?;
        let app = entry.state.lock().ok().and_then(|guard| guard.app.clone());
        let Some(app) = app else {
            return Err(RouterPoolError::NotLoaded);
        };
        entry.lifecycle.begin_drain();
        self.notify_lifecycle(entry);
        self.notify_status(entry);
        if !entry
            .lifecycle
            .wait_for_zero_active(ROUTER_DRAIN_TIMEOUT)
            .await
        {
            entry.lifecycle.mark_drain_blocked(format!(
                "model '{}' still has active requests after {}s drain timeout",
                entry.name,
                ROUTER_DRAIN_TIMEOUT.as_secs()
            ));
            self.notify_lifecycle(entry);
            return Err(RouterPoolError::Capacity(format!(
                "model '{}' is still draining active requests",
                entry.name
            )));
        }

        entry.lifecycle.mark_unloading();
        self.notify_lifecycle(entry);
        let observer = app.state.model_provider.worker_exit_observer();
        let _ = app.state.model_provider.shutdown_worker();
        let observed =
            tokio::task::spawn_blocking(move || observer.wait_timeout(ROUTER_WORKER_EXIT_TIMEOUT))
                .await
                .unwrap_or(false);
        if !observed {
            tracing::warn!(
                model = %entry.name,
                ui_model_id = %entry.ui_model_id,
                timeout_secs = ROUTER_WORKER_EXIT_TIMEOUT.as_secs(),
                "router: worker exit observation timed out during unload"
            );
            entry.lifecycle.mark_drain_blocked(format!(
                "model '{}' worker exit was not observed within {}s",
                entry.name,
                ROUTER_WORKER_EXIT_TIMEOUT.as_secs()
            ));
            self.notify_lifecycle(entry);
            return Err(RouterPoolError::Capacity(format!(
                "model '{}' worker exit is still pending",
                entry.name
            )));
        }
        tracing::info!(
            model = %entry.name,
            ui_model_id = %entry.ui_model_id,
            "router: worker exit observed during unload"
        );
        if let Ok(mut guard) = entry.state.lock() {
            guard.app = None;
            guard.failed = false;
        }
        drop(app);
        entry.lifecycle.mark_unloaded();
        self.notify_lifecycle(entry);
        self.notify_status(entry);
        Ok(())
    }

    /// Unload `name` (b10621 `server_models::unload` through
    /// `POST /models/unload`). Unloading a downloading model cancels the
    /// download, upstream's own unload-during-download behavior.
    pub async fn unload(&self, name: &str) -> Result<(), RouterPoolError> {
        self.unload_with_expected(name, None).await
    }

    async fn unload_with_expected(
        &self,
        name: &str,
        expected_revision: Option<u64>,
    ) -> Result<(), RouterPoolError> {
        let entry = self
            .get(name)
            .ok_or_else(|| RouterPoolError::NotFound(name.to_string()))?;
        reject_stale_revision(&entry, expected_revision)?;
        let cancel_download = entry
            .state
            .lock()
            .ok()
            .and_then(|guard| {
                guard.download.as_ref().map(|download| {
                    download.cancel.store(true, Ordering::Relaxed);
                })
            })
            .is_some();
        if cancel_download {
            return self.wait_for_download_stop(&entry).await;
        }
        if !entry.reserves_capacity() {
            return Err(RouterPoolError::NotLoaded);
        }
        self.unload_entry_with_expected(&entry, expected_revision)
            .await
    }

    async fn wait_for_download_stop(
        &self,
        entry: &RouterModelEntry,
    ) -> Result<(), RouterPoolError> {
        let started = std::time::Instant::now();
        while entry.is_downloading() {
            if started.elapsed() > ROUTER_WORKER_EXIT_TIMEOUT {
                return Err(RouterPoolError::LoadFailed(format!(
                    "model '{}' download did not stop within {}s",
                    entry.name,
                    ROUTER_WORKER_EXIT_TIMEOUT.as_secs()
                )));
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
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
        let lease = match entry.lifecycle.clone().try_request_lease() {
            Ok(lease) => lease,
            Err(message) => return super::routes::slots::llama_invalid_request(message),
        };
        match router.oneshot(request).await {
            Ok(response) => response_with_lease(response, lease),
            Err(err) => match err {},
        }
    }
}

fn stale_revision_error(entry: &RouterModelEntry, expected_revision: u64) -> ErrorBody {
    ErrorBody {
        code: "stale_revision".to_string(),
        message: "catalog entry revision changed; refresh before retrying".to_string(),
        retryable: true,
        field_errors: Some(vec![super::router_lifecycle::FieldError {
            field: "expected_revision".to_string(),
            code: "stale".to_string(),
            message: format!(
                "expected {expected_revision} but current revision is {}",
                entry.lifecycle.revision()
            ),
        }]),
        operation_id: None,
    }
}

fn reject_stale_revision(
    entry: &RouterModelEntry,
    expected_revision: Option<u64>,
) -> Result<(), RouterPoolError> {
    if let Some(expected_revision) = expected_revision
        && entry.lifecycle.revision() != expected_revision
    {
        return Err(RouterPoolError::OperationRejected(stale_revision_error(
            entry,
            expected_revision,
        )));
    }
    Ok(())
}

fn response_with_lease(
    response: axum::response::Response,
    lease: super::router_lifecycle::RequestLease,
) -> axum::response::Response {
    let (parts, body) = response.into_parts();
    let stream = body.into_data_stream();
    let leased = futures::stream::unfold((stream, Some(lease)), |(mut stream, lease)| async move {
        stream.next().await.map(|item| (item, (stream, lease)))
    });
    axum::response::Response::from_parts(parts, Body::from_stream(leased))
}

async fn wait_for_failed_worker_exit(
    entry: Arc<RouterModelEntry>,
    lifecycle: Arc<LifecycleCoordinator>,
    observer: Arc<WorkerExitObserver>,
) {
    while entry.lifecycle.state() == ModelLifecycleState::Failed && !observer.observed() {
        let observer_for_wait = observer.clone();
        let observed = tokio::task::spawn_blocking(move || {
            observer_for_wait.wait_timeout(std::time::Duration::from_secs(1))
        })
        .await
        .unwrap_or(false);
        if observed {
            break;
        }
    }
    if entry.lifecycle.state() == ModelLifecycleState::Failed && observer.observed() {
        if let Ok(mut guard) = entry.state.lock() {
            guard.app = None;
            guard.failed = true;
        }
        entry.lifecycle.mark_worker_exit_observed();
        tracing::info!(
            model = %entry.name,
            ui_model_id = %entry.ui_model_id,
            "router: worker exit observed after failed load"
        );
        lifecycle.publish_model_revision(
            &entry.ui_model_id,
            entry.lifecycle.revision(),
            entry.lifecycle_snapshot(),
        );
    }
}

/// Cache download and removal (issue #1438). Split from the core pool impl
/// so the state machine above stays readable.
impl RouterPool {
    /// Whether a model cache (the mlxcel store) is configured for this pool.
    pub fn has_cache(&self) -> bool {
        self.sources.cache.is_some()
    }

    /// Normalize a requested `POST /models` name into the cache's repo-id
    /// spelling. Errors when no cache is configured or the name cannot form a
    /// valid repository id.
    pub fn normalize_cache_name(&self, name: &str) -> anyhow::Result<String> {
        let cache = self
            .sources
            .cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no model cache is configured"))?;
        cache.normalize_name(name)
    }

    /// Synchronously validate that `repo_id` is fetchable (b10621's metadata
    /// probe before it starts a download). Blocking.
    pub fn validate_cache_repo(&self, repo_id: &str) -> anyhow::Result<()> {
        let cache = self
            .sources
            .cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no model cache is configured"))?;
        cache.validate(repo_id)
    }

    /// Start downloading `name` into the cache (`POST /models`). The name
    /// must already be validated (non-empty, normalized, not present,
    /// metadata probe passed). Registers a transient `downloading` entry,
    /// then fetches on a background task, forwarding progress into the SSE
    /// stream and rescanning when the snapshot lands.
    pub fn start_download(self: &Arc<Self>, name: &str) -> Result<(), RouterPoolError> {
        let Some(cache) = &self.sources.cache else {
            return Err(RouterPoolError::LoadFailed(
                "no model cache is configured (set --model-store-root, MLXCEL_MODELS_DIR, or \
                 MLXCEL_CACHE_DIR)"
                    .to_string(),
            ));
        };
        if self.lookup(name).is_some() {
            return Err(RouterPoolError::AlreadyExists(name.to_string()));
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let path = cache.snapshot_dir(name);
        let section = PresetSection::default();
        let config = self.build_entry_config(name, &path, &section);
        let (ui_model_id, source_key_hash) = self.ui_identity_for(name, RouterModelSource::Cache);
        let entry = Arc::new(RouterModelEntry {
            name: name.to_string(),
            path,
            source: RouterModelSource::Cache,
            aliases: Vec::new(),
            tags: Vec::new(),
            ui_model_id,
            source_key_hash,
            hidden: false,
            preset: None,
            config,
            state: Mutex::new(EntryState {
                app: None,
                failed: false,
                download: Some(DownloadInFlight {
                    cancel: cancel.clone(),
                    progress: serde_json::json!({ "progress": {} }),
                }),
            }),
            lifecycle: Arc::new(ModelLifecycle::new(DownloadState::Downloading)),
            last_used: AtomicI64::new(0),
        });
        {
            let mut entries = self
                .entries
                .write()
                .map_err(|_| RouterPoolError::LoadFailed("router pool poisoned".into()))?;
            entries.insert(entry.name.clone(), entry.clone());
        }
        self.notify_status(&entry);

        let pool = self.clone();
        let task_entry = entry;
        let repo = name.to_string();
        tokio::spawn(async move {
            let hooks = crate::downloader::DownloadHooks {
                progress: Some(pool.download_progress_hook(&task_entry)),
                cancel: Some(cancel),
            };
            let blocking_pool = pool.clone();
            let blocking_repo = repo.clone();
            let result = tokio::task::spawn_blocking(move || {
                let Some(cache) = &blocking_pool.sources.cache else {
                    return Err(anyhow::anyhow!("no model cache configured"));
                };
                cache.download(&blocking_repo, hooks)
            })
            .await
            .unwrap_or_else(|join_err| Err(anyhow::anyhow!(join_err.to_string())));

            let ok = result.is_ok();
            if let Ok(mut guard) = task_entry.state.lock() {
                guard.download = None;
                guard.failed = !ok;
            }
            task_entry.lifecycle.mark_download_terminal(if ok {
                DownloadState::Complete
            } else if result
                .as_ref()
                .err()
                .is_some_and(crate::downloader::is_download_cancelled)
            {
                DownloadState::Incomplete
            } else {
                DownloadState::Failed
            });
            pool.notify_lifecycle(&task_entry);
            match &result {
                Ok(()) => tracing::info!("router: download of '{repo}' finished"),
                Err(err) if crate::downloader::is_download_cancelled(err) => {
                    tracing::info!("router: download of '{repo}' cancelled");
                }
                Err(err) => tracing::warn!("router: download of '{repo}' failed: {err:#}"),
            }
            // b10621 order: the terminal download event first, then the
            // reload that reconciles the entry (a finished snapshot becomes a
            // regular cache entry; a failed one drops out of the list).
            pool.notify(
                if ok {
                    "download_finished"
                } else {
                    "download_failed"
                },
                &repo,
                serde_json::Value::Null,
            );
            if let Err(err) = pool.rescan() {
                tracing::warn!("router: rescan after download of '{repo}' failed: {err:#}");
            }
        });
        Ok(())
    }

    /// The per-chunk progress hook: updates the entry's progress block and
    /// forwards a throttled `download_progress` event (unthrottled SSE at
    /// chunk granularity would flood every subscriber).
    fn download_progress_hook(
        self: &Arc<Self>,
        entry: &Arc<RouterModelEntry>,
    ) -> Arc<dyn Fn(&str, u64, u64) + Send + Sync> {
        let pool = self.clone();
        let entry = entry.clone();
        let last_emit: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        Arc::new(move |url: &str, done: u64, total: u64| {
            let snapshot = {
                let Ok(mut guard) = entry.state.lock() else {
                    return;
                };
                let Some(download) = guard.download.as_mut() else {
                    return;
                };
                download.progress["progress"][url] =
                    serde_json::json!({ "done": done, "total": total });
                download.progress.clone()
            };
            let terminal = total > 0 && done >= total;
            let due = {
                let Ok(mut last) = last_emit.lock() else {
                    return;
                };
                let now = std::time::Instant::now();
                let due = terminal
                    || last.is_none_or(|t| {
                        now.duration_since(t) >= std::time::Duration::from_millis(250)
                    });
                if due {
                    *last = Some(now);
                }
                due
            };
            if due {
                pool.notify("download_progress", &entry.name, snapshot);
            }
        })
    }

    /// Remove `name` from the cache (`DELETE /models`, b10621
    /// `server_models::remove`): cancel an in-flight download or stop a
    /// running instance, delete the snapshot from disk (containment-checked
    /// against the store root), drop the entry, and emit `model_remove`.
    pub async fn remove(&self, name: &str) -> Result<(), RouterPoolError> {
        let entry = self
            .get(name)
            .ok_or_else(|| RouterPoolError::NotFound(name.to_string()))?;
        if entry.source != RouterModelSource::Cache {
            return Err(RouterPoolError::NotRemovable(name.to_string()));
        }
        let Some(cache) = &self.sources.cache else {
            return Err(RouterPoolError::NotRemovable(name.to_string()));
        };

        if let Ok(guard) = entry.state.lock()
            && let Some(download) = &guard.download
        {
            tracing::info!("router: cancelling download for model '{name}'");
            download.cancel.store(true, Ordering::Relaxed);
        }
        // Wait for the download worker to acknowledge the cancel (it clears
        // the download state in its completion block).
        let waited = std::time::Instant::now();
        while entry.is_downloading() {
            if waited.elapsed() > std::time::Duration::from_secs(120) {
                return Err(RouterPoolError::LoadFailed(format!(
                    "model '{name}' download did not stop in time"
                )));
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        if entry.reserves_capacity() {
            tracing::info!("router: stopping model instance '{name}' before removal");
            self.unload_entry(&entry).await?;
        }

        cache
            .remove(name)
            .map_err(|err| RouterPoolError::LoadFailed(err.to_string()))?;
        if let Ok(mut entries) = self.entries.write() {
            entries.remove(name);
        }
        self.notify("model_remove", name, serde_json::Value::Null);
        Ok(())
    }
}

/// Build the per-model serving stack: tokenizer, chat template, provider,
/// [`AppState`], and the model's own axum app (without the CORS layer; the
/// router's top level owns CORS so headers are emitted exactly once).
fn build_model_app(
    model_path: &Path,
    config: super::config::ServerConfig,
) -> anyhow::Result<(AppState, axum::Router)> {
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
    // This runs on the blocking pool (see the `spawn_blocking` in the loader),
    // so the allowlist canonicalization, its writable-directory warning and
    // the ffmpeg probe happen here once per model load rather than on a
    // request's Tokio worker (issue #1766).
    let video_dir_allowlist = super::startup::resolve_video_request_inputs(media_support);
    let state = AppState::with_observability(
        Arc::new(provider),
        config,
        chat_template,
        tokenizer,
        model_path.to_path_buf(),
        batch_metrics,
        batch_observability,
    )
    .with_media_support(media_support)
    .with_video_dir_allowlist(video_dir_allowlist);
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
