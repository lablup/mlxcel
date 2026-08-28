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

//! Unit tests for the router-mode pool (issue #1438): discovery and its
//! confinement boundary, name resolution, the autoload gate, failed loads,
//! and the SSE event stream.

use std::path::PathBuf;

use super::{RouterModelStatus, RouterPool, RouterPoolError, discover_models};
use crate::server::config::ServerConfig;

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

fn pool(root: PathBuf, max: usize, autoload: bool) -> RouterPool {
    RouterPool::new(root, ServerConfig::default(), max, autoload).expect("pool")
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
    assert_eq!(pool.unload("idle"), Err(RouterPoolError::NotLoaded));
    assert_eq!(
        pool.unload("ghost"),
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
