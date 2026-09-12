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

//! Unit tests for the router-mode pool (issue #1438): discovery and its
//! confinement boundary, name resolution, the autoload gate, failed loads,
//! the SSE event stream, and the cache source (list / download / remove with
//! the full b10621 event vocabulary).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{
    RouterModelSource, RouterModelStatus, RouterPool, RouterPoolError, RouterSources,
    discover_models,
};
use crate::downloader::DownloadHooks;
use crate::server::ServerStartupConfig;
use crate::server::router_cache::{CacheSource, RouterDownloader};
use crate::server::router_presets::{PresetCliOverrides, parse_preset_text};

fn temp_models_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel-router-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create models dir");
    dir
}

fn add_fake_model(root: &std::path::Path, name: &str) {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("model dir");
    std::fs::write(dir.join("config.json"), "{}").expect("config.json");
}

/// A downloader that materializes a fake snapshot locally, driving the same
/// hooks the HuggingFace downloader drives. `delay_until_cancel` makes the
/// download hang until the cancel flag flips, for cancellation tests.
struct FakeDownloader {
    fail: bool,
    delay_until_cancel: bool,
    downloads: AtomicUsize,
}

impl FakeDownloader {
    fn ok() -> Arc<Self> {
        Arc::new(Self {
            fail: false,
            delay_until_cancel: false,
            downloads: AtomicUsize::new(0),
        })
    }

    fn failing() -> Arc<Self> {
        Arc::new(Self {
            fail: true,
            delay_until_cancel: false,
            downloads: AtomicUsize::new(0),
        })
    }

    fn hanging() -> Arc<Self> {
        Arc::new(Self {
            fail: false,
            delay_until_cancel: true,
            downloads: AtomicUsize::new(0),
        })
    }
}

impl RouterDownloader for FakeDownloader {
    fn validate(&self, _repo_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    fn download(
        &self,
        repo_id: &str,
        dest_root: &Path,
        hooks: DownloadHooks,
    ) -> anyhow::Result<()> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        let url = format!("https://example.invalid/{repo_id}/config.json");
        if let Some(progress) = &hooks.progress {
            progress(&url, 1, 2);
        }
        if self.delay_until_cancel {
            let cancel = hooks.cancel.clone().expect("cancel flag");
            let started = std::time::Instant::now();
            while !cancel.load(Ordering::Relaxed) {
                if started.elapsed() > std::time::Duration::from_secs(10) {
                    anyhow::bail!("fake download was never cancelled");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            return Err(anyhow::Error::new(crate::downloader::DownloadCancelled));
        }
        if self.fail {
            anyhow::bail!("fake network failure");
        }
        let dest = dest_root.join(repo_id);
        std::fs::create_dir_all(&dest)?;
        std::fs::write(dest.join("config.json"), "{}")?;
        if let Some(progress) = &hooks.progress {
            progress(&url, 2, 2);
        }
        Ok(())
    }
}

fn sources_dir_only(root: PathBuf) -> RouterSources {
    RouterSources {
        models_dir: Some(root),
        cache: None,
        presets: Default::default(),
    }
}

fn pool_from(sources: RouterSources, max: usize, autoload: bool) -> RouterPool {
    RouterPool::new(
        sources,
        ServerStartupConfig::default(),
        Default::default(),
        PresetCliOverrides::default(),
        max,
        autoload,
    )
    .expect("pool")
}

fn pool(root: PathBuf, max: usize, autoload: bool) -> RouterPool {
    pool_from(sources_dir_only(root), max, autoload)
}

/// Drain currently queued events into a Vec of (event, model) pairs.
fn drain_events(
    events: &mut tokio::sync::broadcast::Receiver<serde_json::Value>,
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    while let Ok(event) = events.try_recv() {
        out.push((
            event["event"].as_str().unwrap_or_default().to_string(),
            event["model"].as_str().unwrap_or_default().to_string(),
        ));
    }
    out
}

async fn wait_for_event(
    events: &mut tokio::sync::broadcast::Receiver<serde_json::Value>,
    wanted: &str,
) -> serde_json::Value {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let event = events.recv().await.expect("event stream open");
            if event["event"] == wanted {
                return event;
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for '{wanted}'"))
}

#[test]
fn discovery_finds_checkpoint_dirs_and_skips_noise() {
    let root = temp_models_dir("discover");
    add_fake_model(&root, "model-a");
    add_fake_model(&root, "model-b");
    // Noise: a bare file, a dir without config.json, a dotdir.
    std::fs::write(root.join("stray.txt"), "x").unwrap();
    std::fs::create_dir_all(root.join("not-a-model")).unwrap();
    std::fs::create_dir_all(root.join(".hidden")).unwrap();
    let found = discover_models(&root).expect("scan");
    assert_eq!(
        found.keys().cloned().collect::<Vec<_>>(),
        vec!["model-a", "model-b"]
    );
}

#[cfg(unix)]
#[test]
fn discovery_skips_symlinks_that_escape_the_models_dir() {
    let root = temp_models_dir("symlink");
    let outside = temp_models_dir("symlink-outside");
    add_fake_model(&outside, "escapee");
    std::os::unix::fs::symlink(outside.join("escapee"), root.join("escapee")).unwrap();
    add_fake_model(&root, "legit");
    let found = discover_models(&root).expect("scan");
    assert_eq!(found.keys().cloned().collect::<Vec<_>>(), vec!["legit"]);
}

#[test]
fn missing_models_dir_is_a_startup_error() {
    let missing = std::env::temp_dir().join("mlxcel-router-definitely-missing");
    let _ = std::fs::remove_dir_all(&missing);
    assert!(discover_models(&missing).is_err());
}

#[tokio::test]
async fn resolution_matches_b10621s_refusals() {
    let root = temp_models_dir("resolve");
    add_fake_model(&root, "known");
    let pool = pool(root, 4, true);

    assert!(matches!(
        pool.resolve("", true),
        Err(RouterPoolError::MissingName)
    ));
    assert!(matches!(
        pool.resolve("nope", true),
        Err(RouterPoolError::NotFound(name)) if name == "nope"
    ));
    // Autoload off plus not running: b10621's "model is not loaded".
    assert!(matches!(
        pool.resolve("known", false),
        Err(RouterPoolError::NotLoaded)
    ));
    // Autoload on: resolvable while still unloaded.
    assert!(pool.resolve("known", true).is_ok());
}

#[tokio::test]
async fn a_failed_load_reports_unloaded_with_failure() {
    let root = temp_models_dir("failed-load");
    // config.json exists but nothing else: the tokenizer load fails fast.
    add_fake_model(&root, "broken");
    let pool = pool(root, 4, true);
    let mut events = pool.subscribe();

    let result = pool.begin_load("broken").await;
    assert!(matches!(result, Err(RouterPoolError::LoadFailed(_))));
    let entry = pool.get("broken").expect("entry");
    assert_eq!(entry.status(), RouterModelStatus::Unloaded);
    let snapshot = pool
        .snapshot()
        .into_iter()
        .find(|s| s.name == "broken")
        .expect("snapshot");
    assert!(snapshot.failed, "failed load must be visible");

    // The SSE stream carried the b10621 event shape.
    let event = events.try_recv().expect("status event");
    assert_eq!(event["model"], "broken");
    assert_eq!(event["event"], "status_change");
    assert_eq!(event["data"]["status"], "unloaded");
    assert_eq!(event["data"]["failed"], true);
}

#[tokio::test]
async fn ensure_ready_propagates_load_failure() {
    let root = temp_models_dir("ensure");
    add_fake_model(&root, "broken");
    let pool = pool(root, 4, true);
    let result = pool
        .ensure_ready("broken", std::time::Duration::from_secs(5))
        .await;
    assert!(matches!(result, Err(RouterPoolError::LoadFailed(_))));
}

#[tokio::test]
async fn unload_refuses_a_model_that_is_not_running() {
    let root = temp_models_dir("unload");
    add_fake_model(&root, "idle");
    let pool = pool(root, 4, true);
    assert_eq!(pool.unload("idle").await, Err(RouterPoolError::NotLoaded));
    assert_eq!(
        pool.unload("ghost").await,
        Err(RouterPoolError::NotFound("ghost".into()))
    );
}

#[tokio::test]
async fn rescan_picks_up_new_directories_and_drops_removed_ones() {
    let root = temp_models_dir("rescan");
    add_fake_model(&root, "first");
    let pool = pool(root.clone(), 4, true);
    assert!(pool.get("first").is_some());
    assert!(pool.get("second").is_none());

    add_fake_model(&root, "second");
    std::fs::remove_dir_all(root.join("first")).unwrap();
    pool.rescan().expect("rescan");
    assert!(pool.get("second").is_some());
    assert!(pool.get("first").is_none(), "removed dir drops its entry");
}

// ── Cache source (#1438: POST /models, DELETE /models, SSE vocabulary) ──────

#[tokio::test]
async fn cache_snapshots_list_as_removable_cache_entries() {
    let cache_root = temp_models_dir("cache-list");
    add_fake_model(&cache_root.join("mlx-community"), "tiny");
    let models_dir = temp_models_dir("cache-list-dir");
    add_fake_model(&models_dir, "local");
    let pool = pool_from(
        RouterSources {
            models_dir: Some(models_dir),
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    );
    let snapshots = pool.snapshot();
    let cache_entry = snapshots
        .iter()
        .find(|s| s.name == "mlx-community/tiny")
        .expect("cache entry listed");
    assert_eq!(cache_entry.source, RouterModelSource::Cache);
    let dir_entry = snapshots
        .iter()
        .find(|s| s.name == "local")
        .expect("dir entry");
    assert_eq!(dir_entry.source, RouterModelSource::ModelsDir);
}

#[tokio::test]
async fn models_dir_wins_a_name_collision_with_the_cache() {
    let cache_root = temp_models_dir("collide-cache");
    add_fake_model(&cache_root, "same-name");
    let models_dir = temp_models_dir("collide-dir");
    add_fake_model(&models_dir, "same-name");
    let pool = pool_from(
        RouterSources {
            models_dir: Some(models_dir.clone()),
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    );
    let entry = pool.get("same-name").expect("entry");
    assert_eq!(entry.source, RouterModelSource::ModelsDir);
    assert_eq!(
        entry.path,
        models_dir.join("same-name").canonicalize().unwrap()
    );
}

#[tokio::test]
async fn a_download_emits_the_b10621_event_sequence_and_lands_in_the_cache() {
    let cache_root = temp_models_dir("dl-ok");
    let downloader = FakeDownloader::ok();
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root.clone(), downloader.clone())),
            presets: Default::default(),
        },
        4,
        true,
    ));
    let mut events = pool.subscribe();

    pool.start_download("mlx-community/new-model")
        .expect("start");
    // Transient entry is visible as `downloading` immediately.
    let entry = pool
        .get("mlx-community/new-model")
        .expect("transient entry");
    assert_eq!(entry.status(), RouterModelStatus::Downloading);
    assert!(entry.is_downloading());

    wait_for_event(&mut events, "download_finished").await;
    // The rescan after the download lists the snapshot as a cache entry.
    wait_for_event(&mut events, "models_reload").await;
    let entry = pool.get("mlx-community/new-model").expect("cache entry");
    assert_eq!(entry.source, RouterModelSource::Cache);
    assert_eq!(entry.status(), RouterModelStatus::Unloaded);
    assert!(!entry.is_downloading());
    assert!(
        cache_root
            .join("mlx-community/new-model/config.json")
            .is_file(),
        "snapshot landed in the cache"
    );
    assert_eq!(downloader.downloads.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn download_progress_events_carry_per_url_done_and_total() {
    let cache_root = temp_models_dir("dl-progress");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    ));
    let mut events = pool.subscribe();
    pool.start_download("mlx-community/progress-model")
        .expect("start");
    let event = wait_for_event(&mut events, "download_progress").await;
    assert_eq!(event["model"], "mlx-community/progress-model");
    let progress = event["data"]["progress"]
        .as_object()
        .expect("per-url progress map");
    let (_, first) = progress.iter().next().expect("one url");
    assert!(first.get("done").is_some() && first.get("total").is_some());
}

#[tokio::test]
async fn a_failed_download_emits_download_failed_and_drops_the_entry() {
    let cache_root = temp_models_dir("dl-fail");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root, FakeDownloader::failing())),
            presets: Default::default(),
        },
        4,
        true,
    ));
    let mut events = pool.subscribe();
    pool.start_download("mlx-community/broken-model")
        .expect("start");
    wait_for_event(&mut events, "download_failed").await;
    wait_for_event(&mut events, "models_reload").await;
    assert!(
        pool.get("mlx-community/broken-model").is_none(),
        "failed download drops out of the list"
    );
}

#[tokio::test]
async fn start_download_refuses_an_existing_name() {
    let cache_root = temp_models_dir("dl-dup");
    add_fake_model(&cache_root.join("mlx-community"), "present");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    ));
    assert!(matches!(
        pool.start_download("mlx-community/present"),
        Err(RouterPoolError::AlreadyExists(_))
    ));
}

#[tokio::test]
async fn remove_deletes_a_cache_model_from_disk_and_emits_model_remove() {
    let cache_root = temp_models_dir("rm-ok");
    add_fake_model(&cache_root.join("mlx-community"), "doomed");
    let pool = pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root.clone(), FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    );
    let mut events = pool.subscribe();
    pool.remove("mlx-community/doomed").await.expect("remove");
    assert!(pool.get("mlx-community/doomed").is_none());
    assert!(
        !cache_root.join("mlx-community/doomed").exists(),
        "snapshot removed from disk"
    );
    let names: Vec<(String, String)> = drain_events(&mut events);
    assert!(
        names.contains(&(
            "model_remove".to_string(),
            "mlx-community/doomed".to_string()
        )),
        "model_remove emitted: {names:?}"
    );
}

#[tokio::test]
async fn remove_refuses_models_dir_and_unknown_entries() {
    let models_dir = temp_models_dir("rm-refuse");
    add_fake_model(&models_dir, "local");
    let cache_root = temp_models_dir("rm-refuse-cache");
    let pool = pool_from(
        RouterSources {
            models_dir: Some(models_dir),
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    );
    assert!(matches!(
        pool.remove("local").await,
        Err(RouterPoolError::NotRemovable(name)) if name == "local"
    ));
    assert!(matches!(
        pool.remove("ghost").await,
        Err(RouterPoolError::NotFound(_))
    ));
}

#[tokio::test]
async fn remove_cancels_an_in_flight_download() {
    let cache_root = temp_models_dir("rm-cancel");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root, FakeDownloader::hanging())),
            presets: Default::default(),
        },
        4,
        true,
    ));
    let mut events = pool.subscribe();
    pool.start_download("mlx-community/hanging").expect("start");
    assert!(
        pool.get("mlx-community/hanging")
            .expect("entry")
            .is_downloading()
    );

    pool.remove("mlx-community/hanging")
        .await
        .expect("remove cancels");
    assert!(pool.get("mlx-community/hanging").is_none());
    let names = drain_events(&mut events);
    assert!(
        names.iter().any(|(e, _)| e == "download_failed"),
        "cancelled download reports download_failed: {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|(e, m)| e == "model_remove" && m == "mlx-community/hanging"),
        "model_remove emitted: {names:?}"
    );
}

// ── Presets (#1438: --models-preset translation) ────────────────────────────

#[tokio::test]
async fn preset_sections_define_models_with_aliases_and_tags() {
    let checkpoint_root = temp_models_dir("preset-def");
    add_fake_model(&checkpoint_root, "ckpt");
    let ini = format!(
        "[my-model]\nmodel = {}\nalias = short, alt\ntags = prod\n",
        checkpoint_root.join("ckpt").display()
    );
    let presets = parse_preset_text(&ini).expect("parse");
    let pool = pool_from(
        RouterSources {
            models_dir: None,
            cache: None,
            presets,
        },
        4,
        true,
    );
    let entry = pool.get("my-model").expect("preset entry");
    assert_eq!(entry.source, RouterModelSource::Preset);
    assert_eq!(entry.aliases, vec!["short", "alt"]);
    assert_eq!(entry.tags, vec!["prod"]);
    // Aliases resolve for request routing.
    assert!(pool.lookup("short").is_some());
    // The model object reproduces the section as INI text.
    let snapshot = pool
        .snapshot()
        .into_iter()
        .find(|s| s.name == "my-model")
        .expect("snapshot");
    let ini_text = snapshot.preset_ini.expect("preset ini");
    assert!(ini_text.starts_with("[my-model]\n"), "{ini_text}");
    assert!(!ini_text.contains("alias"), "alias stripped: {ini_text}");
}

#[tokio::test]
async fn preset_overlay_resources_a_discovered_model_and_dedup_hides_cache_twins() {
    let cache_root = temp_models_dir("preset-dedup");
    add_fake_model(&cache_root.join("mlx-community"), "twin");
    let ini = "[served]\nhf-repo = mlx-community/twin\ndedup-cache-models = 1\n";
    let presets = parse_preset_text(ini).expect("parse");
    let pool = pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets,
        },
        4,
        true,
    );
    let served = pool.get("served").expect("preset entry");
    assert_eq!(served.source, RouterModelSource::Preset);
    let twin = pool
        .snapshot()
        .into_iter()
        .find(|s| s.name == "mlx-community/twin")
        .expect("cache twin");
    assert!(twin.hidden, "cache twin hidden by dedup-cache-models");
}

#[tokio::test]
async fn load_on_startup_names_come_from_the_preset() {
    let checkpoint_root = temp_models_dir("preset-los");
    add_fake_model(&checkpoint_root, "ckpt");
    let ini = format!(
        "[eager]\nmodel = {}\nload-on-startup = 1\n[lazy]\nmodel = {}\n",
        checkpoint_root.join("ckpt").display(),
        checkpoint_root.join("ckpt").display()
    );
    let presets = parse_preset_text(&ini).expect("parse");
    let pool = pool_from(
        RouterSources {
            models_dir: None,
            cache: None,
            presets,
        },
        4,
        true,
    );
    assert_eq!(pool.load_on_startup_names(), vec!["eager"]);
}

async fn wait_for_operation_state(
    pool: &RouterPool,
    operation_id: &str,
    terminal: &[crate::server::router_lifecycle::OperationState],
) -> crate::server::router_lifecycle::Operation {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if let Some(operation) = pool.lifecycle_coordinator().get_operation(operation_id)
            && terminal.contains(&operation.state)
        {
            return operation;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "operation {operation_id} did not reach {terminal:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn ui_model_action_replay_wins_over_stale_revision_after_failure() {
    let root = temp_models_dir("ui-replay-stale");
    add_fake_model(&root, "broken");
    let pool = Arc::new(pool(root, 4, true));
    let entry = pool.get("broken").expect("entry");
    let model_id = entry.ui_model_id.clone();
    let revision = entry.lifecycle.revision();

    let first = pool
        .submit_model_action(
            &model_id,
            super::RouterModelAction::Load,
            revision,
            "replay-load-0001",
            None,
        )
        .expect("accepted");
    let failed = wait_for_operation_state(
        &pool,
        &first.operation_id,
        &[crate::server::router_lifecycle::OperationState::Failed],
    )
    .await;
    assert_eq!(
        failed.state,
        crate::server::router_lifecycle::OperationState::Failed
    );
    assert_ne!(
        entry.lifecycle.revision(),
        revision,
        "failed load must advance revision"
    );

    let replay = pool
        .submit_model_action(
            &model_id,
            super::RouterModelAction::Load,
            revision,
            "replay-load-0001",
            None,
        )
        .expect("idempotent replay must not be rejected as stale");
    assert!(replay.idempotent_replay);
    assert_eq!(replay.operation_id, first.operation_id);
    assert_eq!(
        replay.state,
        crate::server::router_lifecycle::OperationState::Failed
    );
}

#[tokio::test]
async fn ui_load_operation_waits_for_terminal_worker_state() {
    let root = temp_models_dir("ui-load-terminal");
    add_fake_model(&root, "broken");
    let root_display = root.display().to_string();
    let pool = Arc::new(pool(root, 4, true));
    let entry = pool.get("broken").expect("entry");
    let accepted = pool
        .submit_model_action(
            &entry.ui_model_id,
            super::RouterModelAction::Load,
            entry.lifecycle.revision(),
            "load-terminal-0001",
            None,
        )
        .expect("accepted");
    let terminal = wait_for_operation_state(
        &pool,
        &accepted.operation_id,
        &[
            crate::server::router_lifecycle::OperationState::Succeeded,
            crate::server::router_lifecycle::OperationState::Failed,
        ],
    )
    .await;
    assert_eq!(
        terminal.state,
        crate::server::router_lifecycle::OperationState::Failed,
        "a broken checkpoint must not be reported as succeeded immediately after begin_load"
    );
    assert!(terminal.result.is_none());
    assert_eq!(
        terminal.error.as_ref().map(|error| error.code.as_str()),
        Some("conflict")
    );
    let rendered = serde_json::to_string(&terminal).expect("operation json");
    assert!(
        !rendered.contains(&root_display),
        "operation history must not expose local paths: {rendered}"
    );
    assert!(
        !entry
            .lifecycle_snapshot()
            .last_error
            .unwrap_or_default()
            .contains(&root_display),
        "lifecycle error must not expose local paths"
    );
}

#[tokio::test]
async fn ui_load_requires_explicit_eviction_when_capacity_is_reserved() {
    let root = temp_models_dir("ui-capacity");
    add_fake_model(&root, "resident");
    add_fake_model(&root, "candidate");
    let root_display = root.display().to_string();
    let pool = Arc::new(pool(root, 1, true));
    let resident = pool.get("resident").expect("resident");
    resident.lifecycle.mark_ready();
    let candidate = pool.get("candidate").expect("candidate");

    let accepted = pool
        .submit_model_action(
            &candidate.ui_model_id,
            super::RouterModelAction::Load,
            candidate.lifecycle.revision(),
            "capacity-load-0001",
            None,
        )
        .expect("accepted");
    let terminal = wait_for_operation_state(
        &pool,
        &accepted.operation_id,
        &[crate::server::router_lifecycle::OperationState::Failed],
    )
    .await;
    assert_eq!(
        terminal.state,
        crate::server::router_lifecycle::OperationState::Failed
    );
    assert_eq!(
        terminal.error.as_ref().map(|error| error.code.as_str()),
        Some("conflict")
    );
    assert!(
        terminal
            .error
            .as_ref()
            .map(|error| !error.message.contains(&root_display))
            .unwrap_or(false),
        "capacity refusal must be redacted: {terminal:?}"
    );
    assert_eq!(
        resident.lifecycle.state(),
        crate::server::router_lifecycle::ModelLifecycleState::Ready
    );
}

#[tokio::test]
async fn rescan_and_remove_preserve_reserved_entries_until_release() {
    let cache_root = temp_models_dir("reserved-cache");
    add_fake_model(&cache_root.join("mlx-community"), "reserved");
    let pool = pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root.clone(), FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    );
    let entry = pool.get("mlx-community/reserved").expect("entry");
    entry.lifecycle.mark_loading();
    std::fs::remove_dir_all(cache_root.join("mlx-community/reserved")).unwrap();
    pool.rescan().expect("rescan");
    assert!(
        pool.get("mlx-community/reserved").is_some(),
        "reserved entry must not disappear during rescan"
    );
    assert_eq!(
        pool.remove("mlx-community/reserved").await,
        Err(RouterPoolError::NotLoaded),
        "cache deletion must not remove a lifecycle-reserved entry without an observed unload"
    );
}

#[test]
fn rescan_preserves_entry_that_becomes_reserved_after_snapshot_clone() {
    let cache_root = temp_models_dir("reserved-after-clone");
    add_fake_model(&cache_root.join("mlx-community"), "race");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    ));
    let entry = pool.get("mlx-community/race").expect("entry");
    assert!(!entry.reserves_capacity());

    let (snapshot_tx, snapshot_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
    pool.set_rescan_after_snapshot_hook(Some(Arc::new(move || {
        snapshot_tx.send(()).expect("notify snapshot clone");
        release_rx
            .lock()
            .expect("release receiver mutex")
            .recv()
            .expect("wait for reservation");
    })));

    let rescan_pool = pool.clone();
    let rescan_thread = std::thread::spawn(move || rescan_pool.rescan());
    snapshot_rx.recv().expect("rescan reached snapshot hook");
    entry.lifecycle.mark_loading();
    release_tx.send(()).expect("release rescan");
    rescan_thread
        .join()
        .expect("rescan thread")
        .expect("rescan succeeds");
    pool.set_rescan_after_snapshot_hook(None);

    let current = pool.get("mlx-community/race").expect("entry after rescan");
    assert!(
        Arc::ptr_eq(&entry, &current),
        "final rescan write must preserve the entry that became reserved after the stale snapshot clone"
    );
    assert_eq!(
        current.lifecycle.state(),
        crate::server::router_lifecycle::ModelLifecycleState::Loading
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn begin_load_reservation_is_atomic_with_current_registry_entry() {
    let cache_root = temp_models_dir("load-current-reservation");
    add_fake_model(&cache_root.join("mlx-community"), "atomic");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    ));
    let entry = pool.get("mlx-community/atomic").expect("entry");

    let (checked_tx, checked_rx) = std::sync::mpsc::channel();
    let (release_checked_tx, release_checked_rx) = std::sync::mpsc::channel();
    let release_checked_rx = Arc::new(std::sync::Mutex::new(release_checked_rx));
    pool.set_load_current_check_before_reservation_hook(Some(Arc::new(move || {
        checked_tx.send(()).expect("notify current check");
        release_checked_rx
            .lock()
            .expect("release current-check mutex")
            .recv()
            .expect("release current-check hook");
    })));

    let (reserved_tx, reserved_rx) = std::sync::mpsc::channel();
    let (release_reserved_tx, release_reserved_rx) = std::sync::mpsc::channel();
    let release_reserved_rx = Arc::new(std::sync::Mutex::new(release_reserved_rx));
    pool.set_load_after_reservation_hook(Some(Arc::new(move || {
        reserved_tx.send(()).expect("notify reservation");
        release_reserved_rx
            .lock()
            .expect("release reservation mutex")
            .recv()
            .expect("release reservation hook");
    })));

    let (snapshot_tx, snapshot_rx) = std::sync::mpsc::channel();
    pool.set_rescan_after_snapshot_hook(Some(Arc::new(move || {
        snapshot_tx.send(()).expect("notify stale snapshot clone");
    })));

    let load_pool = pool.clone();
    let load_task = tokio::spawn(async move { load_pool.begin_load("mlx-community/atomic").await });
    checked_rx
        .recv()
        .expect("begin_load reached current-check/reservation boundary");

    let rescan_pool = pool.clone();
    let rescan_thread = std::thread::spawn(move || rescan_pool.rescan());
    snapshot_rx.recv().expect("rescan cloned stale snapshot");
    release_checked_tx
        .send(())
        .expect("release current-check hook");
    reserved_rx.recv().expect("load marked reservation");
    rescan_thread
        .join()
        .expect("rescan thread")
        .expect("rescan succeeds");

    let current = pool.get("mlx-community/atomic").expect("current entry");
    assert!(
        Arc::ptr_eq(&entry, &current),
        "rescan must not replace the entry between the final current check and the load reservation"
    );
    assert!(
        current.reserves_capacity(),
        "entry must be reserved before rescan is allowed to publish its rebuilt registry"
    );

    pool.set_rescan_after_snapshot_hook(None);
    pool.set_load_current_check_before_reservation_hook(None);
    pool.set_load_after_reservation_hook(None);
    release_reserved_tx
        .send(())
        .expect("release reservation hook");
    let _ = load_task.await.expect("load task");
}

#[test]
fn recreated_cache_entry_keeps_model_id_but_gets_fresh_revision() {
    let cache_root = temp_models_dir("aba-revision");
    add_fake_model(&cache_root.join("mlx-community"), "same");
    let pool = pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root.clone(), FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    );
    let first = pool.get("mlx-community/same").expect("entry");
    let first_id = first.ui_model_id.clone();
    let first_revision = first.lifecycle_revision();
    std::fs::remove_dir_all(cache_root.join("mlx-community/same")).unwrap();
    pool.rescan().expect("drop entry");
    assert!(pool.get("mlx-community/same").is_none());

    add_fake_model(&cache_root.join("mlx-community"), "same");
    pool.rescan().expect("recreate entry");
    let recreated = pool.get("mlx-community/same").expect("recreated");
    assert_eq!(recreated.ui_model_id, first_id);
    assert_ne!(
        recreated.lifecycle_revision(),
        first_revision,
        "same stable model id must not reuse an old revision after remove/rescan/re-add"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_ui_action_rejects_same_id_recreated_before_execution() {
    let cache_root = temp_models_dir("queued-same-id");
    add_fake_model(&cache_root.join("mlx-community"), "queued");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root.clone(), FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    ));
    let entry = pool.get("mlx-community/queued").expect("entry");
    let model_id = entry.ui_model_id.clone();
    let revision = entry.lifecycle_revision();

    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
    pool.set_model_action_before_execute_hook(Some(Arc::new(move || {
        let _ = entered_tx.send(());
        release_rx
            .lock()
            .expect("release receiver mutex")
            .recv()
            .expect("release queued action");
    })));

    let accepted = pool
        .submit_model_action(
            &model_id,
            super::RouterModelAction::Load,
            revision,
            "queued-same-id-0001",
            None,
        )
        .expect("accepted");
    entered_rx.recv().expect("background reached hook");
    pool.remove("mlx-community/queued")
        .await
        .expect("remove old entry");
    add_fake_model(&cache_root.join("mlx-community"), "queued");
    pool.rescan().expect("recreate same id");
    let recreated = pool.get("mlx-community/queued").expect("recreated");
    assert_eq!(recreated.ui_model_id, model_id);
    assert_ne!(recreated.lifecycle_revision(), revision);
    release_tx.send(()).expect("release hook");

    let terminal = wait_for_operation_state(
        &pool,
        &accepted.operation_id,
        &[crate::server::router_lifecycle::OperationState::Failed],
    )
    .await;
    assert_eq!(
        terminal.error.as_ref().map(|error| error.code.as_str()),
        Some("stale_revision")
    );
    pool.set_model_action_before_execute_hook(None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_ui_load_rejects_same_id_recreated_eviction_target_before_execution() {
    let cache_root = temp_models_dir("queued-target-same-id");
    add_fake_model(&cache_root.join("mlx-community"), "resident");
    add_fake_model(&cache_root.join("mlx-community"), "candidate");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: None,
            cache: Some(CacheSource::new(cache_root.clone(), FakeDownloader::ok())),
            presets: Default::default(),
        },
        1,
        true,
    ));
    let candidate = pool.get("mlx-community/candidate").expect("candidate");
    let candidate_id = candidate.ui_model_id.clone();
    let candidate_revision = candidate.lifecycle_revision();
    let resident = pool.get("mlx-community/resident").expect("resident");
    let target_id = resident.ui_model_id.clone();
    let target_revision = resident.lifecycle_revision();

    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
    pool.set_model_action_before_execute_hook(Some(Arc::new(move || {
        let _ = entered_tx.send(());
        release_rx
            .lock()
            .expect("release receiver mutex")
            .recv()
            .expect("release queued action");
    })));

    let accepted = pool
        .submit_model_action(
            &candidate_id,
            super::RouterModelAction::Load,
            candidate_revision,
            "queued-target-same-id-0001",
            Some(&target_id),
        )
        .expect("accepted");
    entered_rx.recv().expect("background reached hook");
    std::fs::remove_dir_all(cache_root.join("mlx-community/resident")).unwrap();
    pool.rescan().expect("drop target");
    assert!(pool.get("mlx-community/resident").is_none());
    add_fake_model(&cache_root.join("mlx-community"), "resident");
    pool.rescan().expect("recreate target");
    let recreated = pool.get("mlx-community/resident").expect("recreated");
    assert_eq!(recreated.ui_model_id, target_id);
    assert_ne!(recreated.lifecycle_revision(), target_revision);
    recreated.lifecycle.mark_ready();
    release_tx.send(()).expect("release hook");

    let terminal = wait_for_operation_state(
        &pool,
        &accepted.operation_id,
        &[crate::server::router_lifecycle::OperationState::Failed],
    )
    .await;
    assert_eq!(
        terminal.error.as_ref().map(|error| error.code.as_str()),
        Some("stale_revision")
    );
    pool.set_model_action_before_execute_hook(None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_ui_load_rejects_different_id_eviction_target_same_name_before_execution() {
    let cache_root = temp_models_dir("queued-target-different-id-cache");
    let models_dir = temp_models_dir("queued-target-different-id-dir");
    add_fake_model(&cache_root, "resident");
    add_fake_model(&cache_root, "candidate");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: Some(models_dir.clone()),
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets: Default::default(),
        },
        1,
        true,
    ));
    let candidate = pool.get("candidate").expect("candidate");
    let candidate_id = candidate.ui_model_id.clone();
    let candidate_revision = candidate.lifecycle_revision();
    let resident = pool.get("resident").expect("resident");
    assert_eq!(resident.source, RouterModelSource::Cache);
    let target_id = resident.ui_model_id.clone();

    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
    pool.set_model_action_before_execute_hook(Some(Arc::new(move || {
        let _ = entered_tx.send(());
        release_rx
            .lock()
            .expect("release receiver mutex")
            .recv()
            .expect("release queued action");
    })));

    let accepted = pool
        .submit_model_action(
            &candidate_id,
            super::RouterModelAction::Load,
            candidate_revision,
            "queued-target-different-id-0001",
            Some(&target_id),
        )
        .expect("accepted");
    entered_rx.recv().expect("background reached hook");
    add_fake_model(&models_dir, "resident");
    pool.rescan().expect("models_dir replaces target");
    let replacement = pool.get("resident").expect("replacement");
    assert_eq!(replacement.source, RouterModelSource::ModelsDir);
    assert_ne!(replacement.ui_model_id, target_id);
    replacement.lifecycle.mark_ready();
    release_tx.send(()).expect("release hook");

    let terminal = wait_for_operation_state(
        &pool,
        &accepted.operation_id,
        &[crate::server::router_lifecycle::OperationState::Failed],
    )
    .await;
    assert_eq!(
        terminal.error.as_ref().map(|error| error.code.as_str()),
        Some("stale_revision")
    );
    pool.set_model_action_before_execute_hook(None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_ui_action_rejects_different_id_with_same_name_before_execution() {
    let cache_root = temp_models_dir("queued-different-id-cache");
    let models_dir = temp_models_dir("queued-different-id-dir");
    add_fake_model(&cache_root, "same-name");
    let pool = Arc::new(pool_from(
        RouterSources {
            models_dir: Some(models_dir.clone()),
            cache: Some(CacheSource::new(cache_root, FakeDownloader::ok())),
            presets: Default::default(),
        },
        4,
        true,
    ));
    let entry = pool.get("same-name").expect("cache entry");
    assert_eq!(entry.source, RouterModelSource::Cache);
    let model_id = entry.ui_model_id.clone();
    let revision = entry.lifecycle_revision();

    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
    pool.set_model_action_before_execute_hook(Some(Arc::new(move || {
        let _ = entered_tx.send(());
        release_rx
            .lock()
            .expect("release receiver mutex")
            .recv()
            .expect("release queued action");
    })));

    let accepted = pool
        .submit_model_action(
            &model_id,
            super::RouterModelAction::Load,
            revision,
            "queued-different-id-0001",
            None,
        )
        .expect("accepted");
    entered_rx.recv().expect("background reached hook");
    add_fake_model(&models_dir, "same-name");
    pool.rescan().expect("models_dir replaces same name");
    let replacement = pool.get("same-name").expect("replacement");
    assert_eq!(replacement.source, RouterModelSource::ModelsDir);
    assert_ne!(replacement.ui_model_id, model_id);
    release_tx.send(()).expect("release hook");

    let terminal = wait_for_operation_state(
        &pool,
        &accepted.operation_id,
        &[crate::server::router_lifecycle::OperationState::Failed],
    )
    .await;
    assert_eq!(
        terminal.error.as_ref().map(|error| error.code.as_str()),
        Some("stale_revision")
    );
    pool.set_model_action_before_execute_hook(None);
}

#[tokio::test]
async fn response_body_drop_releases_active_request_lease() {
    let lifecycle = Arc::new(crate::server::router_lifecycle::ModelLifecycle::new(
        crate::server::router_lifecycle::DownloadState::Complete,
    ));
    lifecycle.mark_ready();
    let lease = lifecycle.clone().try_request_lease().expect("lease");
    assert_eq!(lifecycle.snapshot().active_requests, 1);
    let response = axum::response::Response::new(axum::body::Body::from("hello"));
    let leased = super::response_with_lease(response, lease);
    assert_eq!(lifecycle.snapshot().active_requests, 1);
    drop(leased);
    assert_eq!(lifecycle.snapshot().active_requests, 0);
}

#[tokio::test]
async fn shutdown_reports_pending_reserved_entries_within_bound() {
    let root = temp_models_dir("shutdown-pending");
    add_fake_model(&root, "reserved");
    let pool = Arc::new(pool(root, 4, true));
    let entry = pool.get("reserved").expect("entry");
    entry.lifecycle.mark_loading();
    let report = pool
        .shutdown_all(std::time::Duration::from_millis(50))
        .await;
    assert_eq!(report.attempted, 1);
    assert!(report.remaining.contains(&"reserved".to_string()));
    assert_eq!(
        entry.lifecycle.state(),
        crate::server::router_lifecycle::ModelLifecycleState::Draining
    );
}
