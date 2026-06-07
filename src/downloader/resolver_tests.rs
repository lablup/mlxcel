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

//! Unit tests for the repo-id-aware `-m/--model` resolver (issue #94).
//!
//! Network-free by construction: every test exercises a reuse branch (existing
//! path, repo-id shape, legacy CWD, HF-cache, mlxcel store) or a parse/error
//! branch. The download branch (2d) is the only path that touches the network
//! and is covered end-to-end by the acceptance criteria, not by a unit test.

use super::*;
use std::fs;
use std::path::{Path, PathBuf};
// Env-var-sensitive tests must serialize through the crate-wide `ENV_LOCK`
// Rust 2024's `set_var`/`remove_var` are `unsafe` because
// libc's env block has no internal lock and concurrent mutation is UB.
use crate::test_support::env_lock::env_lock;

/// Restore an env var to a prior captured state.
fn restore_env(key: &str, prev: Option<String>) {
    unsafe {
        match prev {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

/// Create a complete (loadable) snapshot directory: `dir` plus a `config.json`.
fn make_complete_snapshot(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), b"{}").unwrap();
}

// ── is_repo_id_shape ────────────────────────────────────────────────────────

#[test]
fn repo_id_shape_accepts_owner_name() {
    assert!(is_repo_id_shape("mlx-community/Qwen3-4B-4bit"));
    assert!(is_repo_id_shape("Qwen/Qwen3-4B-4bit"));
    assert!(is_repo_id_shape("owner/model"));
    // `.`, `_`, `-` are all valid HF id characters.
    assert!(is_repo_id_shape("some.owner_1/my-model.v2"));
}

#[test]
fn repo_id_shape_rejects_non_owner_name() {
    // No slash → bare name (not the `owner/name` shape this resolver gates on).
    assert!(!is_repo_id_shape("gpt2"));
    // More than one slash → relative path, not a repo-id.
    assert!(!is_repo_id_shape("models/foo/bar"));
    assert!(!is_repo_id_shape("a/b/c"));
    // Empty segments.
    assert!(!is_repo_id_shape("/model"));
    assert!(!is_repo_id_shape("owner/"));
    assert!(!is_repo_id_shape("/"));
    assert!(!is_repo_id_shape(""));
    // Illegal characters per the locked `[A-Za-z0-9._-]` segment set: space,
    // `~`, `:`. (`~` is not in the allowed set, so `~/model` is rejected.)
    assert!(!is_repo_id_shape("owner name/model"));
    assert!(!is_repo_id_shape("~/model"));
    assert!(!is_repo_id_shape("owner/mo:del"));
}

#[test]
fn repo_id_shape_matches_locked_dot_owner_spec() {
    // The locked regex `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$` includes `.`, so a
    // value like `./models` *does* match the shape. Back-compat is preserved
    // regardless: `resolve_model_source` checks `value.exists()` (branch 1)
    // before the repo-id branch, so an existing `./models` directory is used
    // as a path. Only a non-existent `./models` would fall through to the
    // repo-id branch (and then fail at the HF API as a malformed owner) — an
    // acceptable user error, not a back-compat regression.
    assert!(is_repo_id_shape("./models"));
}

// ── snapshot_is_complete ────────────────────────────────────────────────────

#[test]
fn snapshot_is_complete_requires_config_json() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("snap");
    // Missing entirely.
    assert!(!snapshot_is_complete(&dir));
    // Exists but no config.json → still incomplete (would fail to load).
    fs::create_dir_all(&dir).unwrap();
    assert!(!snapshot_is_complete(&dir));
    // With config.json → complete.
    fs::write(dir.join("config.json"), b"{}").unwrap();
    assert!(snapshot_is_complete(&dir));
}

#[test]
fn snapshot_is_complete_false_for_file_path() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("not-a-dir");
    fs::write(&file, b"x").unwrap();
    assert!(!snapshot_is_complete(&file));
}

// ── resolve_model_source: branch 1 (existing path, byte-identical) ───────────

#[test]
fn existing_directory_path_is_used_verbatim() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("local-model");
    fs::create_dir_all(&model).unwrap();

    let resolved = resolve_model_source(&model).unwrap();
    assert_eq!(resolved, model);
}

#[test]
fn existing_file_path_is_used_verbatim() {
    // The on-disk check uses `Path::exists`, which is true for files too. A
    // model loader will reject a file, but the resolver must not second-guess
    // an explicit existing path — that is the byte-identical back-compat rule.
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("weights.safetensors");
    fs::write(&file, b"x").unwrap();

    let resolved = resolve_model_source(&file).unwrap();
    assert_eq!(resolved, file);
}

#[test]
fn existing_path_shaped_like_repo_id_is_still_used_verbatim() {
    // A local directory literally named `owner/name` that exists on disk must
    // be used as-is (branch 1 wins over the repo-id branch). We create the
    // directory under a temp root and resolve via its full path.
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("owner").join("name");
    fs::create_dir_all(&model).unwrap();

    let resolved = resolve_model_source(&model).unwrap();
    assert_eq!(resolved, model);
}

// ── resolve_model_source: branch 4 (error) ──────────────────────────────────

#[test]
fn nonexistent_non_repo_id_value_errors_clearly() {
    // A value that is neither an existing path, an `owner/name` repo-id, nor a
    // bare model-name segment. Since issue #112 a bare segment (e.g.
    // `definitely-not-here`) resolves as `mlx-community/<name>`, so the error
    // arm now requires an illegal segment character such as a space.
    let err = resolve_model_source(Path::new("not a model name")).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("neither an existing path"), "got: {msg}");
    assert!(msg.contains("owner/name"), "got: {msg}");
}

#[test]
fn nonexistent_multi_segment_path_errors_clearly() {
    // `a/b/c` has too many slashes to be a repo-id and does not exist.
    let err = resolve_model_source(Path::new("no/such/nested/path")).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("neither an existing path"), "got: {msg}");
}

// ── locate_cached_snapshot: branch 2a (legacy ./models/<basename>) ───────────

#[test]
fn locate_prefers_legacy_cwd_models_when_complete() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd_models = tmp.path().join("models");
    let legacy = cwd_models.join("Qwen3-4B-4bit");
    make_complete_snapshot(&legacy);

    // Point the mlxcel store and HF cache at empty temp dirs so only the
    // legacy location can hit.
    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    let empty_store = tmp.path().join("store");
    let empty_hf = tmp.path().join("hf");
    fs::create_dir_all(&empty_hf).unwrap();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &empty_store);
        std::env::set_var("HF_HUB_CACHE", &empty_hf);
        std::env::remove_var("HF_HOME");
    }

    let hit = locate_cached_snapshot("mlx-community/Qwen3-4B-4bit", None, &cwd_models, None);

    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    assert_eq!(hit, Some(legacy));
}

#[test]
fn locate_ignores_incomplete_legacy_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd_models = tmp.path().join("models");
    // Legacy dir exists but has no config.json → must be skipped.
    fs::create_dir_all(cwd_models.join("Qwen3-4B-4bit")).unwrap();

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    let empty_store = tmp.path().join("store");
    let empty_hf = tmp.path().join("hf");
    fs::create_dir_all(&empty_hf).unwrap();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &empty_store);
        std::env::set_var("HF_HUB_CACHE", &empty_hf);
        std::env::remove_var("HF_HOME");
    }

    let hit = locate_cached_snapshot("mlx-community/Qwen3-4B-4bit", None, &cwd_models, None);

    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    assert_eq!(hit, None);
}

// ── locate_cached_snapshot: branch 2b (HuggingFace cache reuse) ──────────────

#[test]
fn locate_reuses_hf_cache_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hf");
    // Build a fake HF hub snapshot: models--owner--model/snapshots/<sha>/config.json
    let sha = "0123456789abcdef0123456789abcdef01234567";
    let repo_dir = hub.join("models--owner--model");
    let snap = repo_dir.join("snapshots").join(sha);
    make_complete_snapshot(&snap);
    let refs = repo_dir.join("refs");
    fs::create_dir_all(&refs).unwrap();
    fs::write(refs.join("main"), sha).unwrap();

    // Empty legacy + empty mlxcel store so only the HF cache can hit.
    let cwd_models = tmp.path().join("no-models");

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    let empty_store = tmp.path().join("store");
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &empty_store);
        std::env::set_var("HF_HUB_CACHE", &hub);
        std::env::remove_var("HF_HOME");
    }

    let hit = locate_cached_snapshot("owner/model", None, &cwd_models, None);

    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    assert_eq!(hit, Some(snap));
}

// ── locate_cached_snapshot: branch 2c (mlxcel global store) ──────────────────

#[test]
fn locate_uses_mlxcel_store_when_complete() {
    let tmp = tempfile::tempdir().unwrap();
    let store_root = tmp.path().join("store");
    // mlxcel store layout: <root>/models/<owner>/<name>/config.json
    let store_dir = store_root.join("models").join("owner").join("model");
    make_complete_snapshot(&store_dir);

    // Empty legacy + empty HF cache so only the mlxcel store can hit.
    let cwd_models = tmp.path().join("no-models");
    let empty_hf = tmp.path().join("hf");
    fs::create_dir_all(&empty_hf).unwrap();

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &store_root);
        std::env::set_var("HF_HUB_CACHE", &empty_hf);
        std::env::remove_var("HF_HOME");
    }

    let hit = locate_cached_snapshot("owner/model", None, &cwd_models, None);

    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    assert_eq!(hit, Some(store_dir));
}

// ── locate_cached_snapshot: --models-dir override (issue #107) ───────────────

#[test]
fn locate_uses_override_models_root_for_store_probe() {
    let tmp = tempfile::tempdir().unwrap();
    // The OVERRIDE models root holds the snapshot directly (no `models/`
    // subdir): <override>/<owner>/<name>/config.json.
    let override_root = tmp.path().join("custom-store");
    let store_dir = override_root.join("owner").join("model");
    make_complete_snapshot(&store_dir);

    // Decoy cache-root store that must be IGNORED when the override is passed.
    // Stage a different (also-complete) snapshot there to prove we do not read
    // it. Empty legacy + empty HF cache so only the store probe can hit.
    let cwd_models = tmp.path().join("no-models");
    let decoy_cache = tmp.path().join("decoy-cache");
    make_complete_snapshot(&decoy_cache.join("models").join("owner").join("model"));
    let empty_hf = tmp.path().join("hf");
    fs::create_dir_all(&empty_hf).unwrap();

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_models_dir = std::env::var("MLXCEL_MODELS_DIR").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &decoy_cache);
        std::env::remove_var("MLXCEL_MODELS_DIR");
        std::env::set_var("HF_HUB_CACHE", &empty_hf);
        std::env::remove_var("HF_HOME");
    }

    let hit = locate_cached_snapshot("owner/model", None, &cwd_models, Some(&override_root));

    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("MLXCEL_MODELS_DIR", prev_models_dir);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    // The hit must be the override-root snapshot, not the decoy cache-root one.
    assert_eq!(hit, Some(store_dir));
}

#[test]
fn resolve_model_source_with_override_reuses_override_store() {
    // End-to-end (network-free): a repo-id that is not a path but IS present
    // under the override models root resolves there with no download.
    let tmp = tempfile::tempdir().unwrap();
    let override_root = tmp.path().join("custom-store");
    let store_dir = override_root.join("owner").join("model");
    make_complete_snapshot(&store_dir);
    let empty_hf = tmp.path().join("hf");
    fs::create_dir_all(&empty_hf).unwrap();

    // Run from a CWD with no `./models` so the legacy probe misses.
    let run_dir = tmp.path().join("run");
    fs::create_dir_all(&run_dir).unwrap();

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_models_dir = std::env::var("MLXCEL_MODELS_DIR").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    let prev_cwd = std::env::current_dir().ok();
    unsafe {
        // Point the cache-root store elsewhere (empty) to prove the override wins.
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path().join("empty-cache"));
        std::env::remove_var("MLXCEL_MODELS_DIR");
        std::env::set_var("HF_HUB_CACHE", &empty_hf);
        std::env::remove_var("HF_HOME");
    }
    std::env::set_current_dir(&run_dir).unwrap();

    let resolved =
        resolve_model_source_with_override(Path::new("owner/model"), Some(&override_root));

    if let Some(cwd) = prev_cwd {
        let _ = std::env::set_current_dir(cwd);
    }
    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("MLXCEL_MODELS_DIR", prev_models_dir);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    assert_eq!(resolved.unwrap(), store_dir);
}

#[test]
fn locate_returns_none_on_total_miss() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd_models = tmp.path().join("no-models");
    let empty_hf = tmp.path().join("hf");
    fs::create_dir_all(&empty_hf).unwrap();
    let empty_store = tmp.path().join("store");

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &empty_store);
        std::env::set_var("HF_HUB_CACHE", &empty_hf);
        std::env::remove_var("HF_HOME");
    }

    let hit = locate_cached_snapshot("owner/never-downloaded", None, &cwd_models, None);

    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    assert_eq!(hit, None);
}

// ── precedence: legacy CWD beats both HF cache and mlxcel store ──────────────

#[test]
fn legacy_cwd_wins_over_hf_and_store() {
    let tmp = tempfile::tempdir().unwrap();

    // Populate all three locations for the same repo-id.
    let cwd_models = tmp.path().join("models");
    let legacy = cwd_models.join("model");
    make_complete_snapshot(&legacy);

    let hub = tmp.path().join("hf");
    let sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let repo_dir = hub.join("models--owner--model");
    make_complete_snapshot(&repo_dir.join("snapshots").join(sha));
    let refs = repo_dir.join("refs");
    fs::create_dir_all(&refs).unwrap();
    fs::write(refs.join("main"), sha).unwrap();

    let store_root = tmp.path().join("store");
    make_complete_snapshot(&store_root.join("models").join("owner").join("model"));

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &store_root);
        std::env::set_var("HF_HUB_CACHE", &hub);
        std::env::remove_var("HF_HOME");
    }

    let hit = locate_cached_snapshot("owner/model", None, &cwd_models, None);

    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    // Legacy CWD must win the precedence race.
    assert_eq!(hit, Some(legacy));
}

// ── end-to-end reuse via resolve_model_source (store hit, no download) ───────

#[test]
fn resolve_model_source_reuses_store_without_download() {
    // A repo-id that does not exist as a path but is already in the mlxcel
    // store must resolve to the store dir with no network access.
    let tmp = tempfile::tempdir().unwrap();
    let store_root = tmp.path().join("store");
    let store_dir = store_root.join("models").join("owner").join("model");
    make_complete_snapshot(&store_dir);
    let empty_hf = tmp.path().join("hf");
    fs::create_dir_all(&empty_hf).unwrap();

    // Run from a CWD with no `./models` so branch 2a misses. The resolver uses
    // a relative `./models`, so we chdir into an empty temp dir for this test.
    let run_dir = tmp.path().join("run");
    fs::create_dir_all(&run_dir).unwrap();

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    let prev_cwd = std::env::current_dir().ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &store_root);
        std::env::set_var("HF_HUB_CACHE", &empty_hf);
        std::env::remove_var("HF_HOME");
    }
    std::env::set_current_dir(&run_dir).unwrap();

    let resolved = resolve_model_source(Path::new("owner/model"));

    if let Some(cwd) = prev_cwd {
        let _ = std::env::set_current_dir(cwd);
    }
    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    assert_eq!(resolved.unwrap(), store_dir);
}

/// Smoke check that the legacy models constant did not drift from the
/// downloader's own basename helper expectations.
#[test]
fn legacy_models_dir_basename_roundtrip() {
    assert_eq!(LEGACY_MODELS_DIR, "models");
    let legacy = PathBuf::from(LEGACY_MODELS_DIR).join(repo_basename("owner/model"));
    assert_eq!(legacy, PathBuf::from("models/model"));
}

// ── resolve_model_source: branch 3 (bare name → default org, issue #112) ─────

#[test]
fn bare_name_expands_to_default_org_and_reuses_store() {
    // A bare, prefix-less name resolves as `mlx-community/<name>` and, when that
    // repo is already in the mlxcel store, reuses it with no network access.
    let tmp = tempfile::tempdir().unwrap();
    let store_root = tmp.path().join("store");
    let store_dir = store_root
        .join("models")
        .join("mlx-community")
        .join("Qwen3-4B-4bit");
    make_complete_snapshot(&store_dir);
    let empty_hf = tmp.path().join("hf");
    fs::create_dir_all(&empty_hf).unwrap();
    let run_dir = tmp.path().join("run");
    fs::create_dir_all(&run_dir).unwrap();

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_default_org = std::env::var("MLXCEL_DEFAULT_ORG").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    let prev_cwd = std::env::current_dir().ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &store_root);
        std::env::remove_var("MLXCEL_DEFAULT_ORG");
        std::env::set_var("HF_HUB_CACHE", &empty_hf);
        std::env::remove_var("HF_HOME");
    }
    std::env::set_current_dir(&run_dir).unwrap();

    let resolved = resolve_model_source(Path::new("Qwen3-4B-4bit"));

    if let Some(cwd) = prev_cwd {
        let _ = std::env::set_current_dir(cwd);
    }
    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("MLXCEL_DEFAULT_ORG", prev_default_org);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    assert_eq!(resolved.unwrap(), store_dir);
}

#[test]
fn bare_name_honors_default_org_override() {
    // `MLXCEL_DEFAULT_ORG` overrides the `mlx-community` default.
    let tmp = tempfile::tempdir().unwrap();
    let store_root = tmp.path().join("store");
    let store_dir = store_root.join("models").join("acme").join("my-model");
    make_complete_snapshot(&store_dir);
    let empty_hf = tmp.path().join("hf");
    fs::create_dir_all(&empty_hf).unwrap();
    let run_dir = tmp.path().join("run");
    fs::create_dir_all(&run_dir).unwrap();

    let _guard = env_lock();
    let prev_cache_dir = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_default_org = std::env::var("MLXCEL_DEFAULT_ORG").ok();
    let prev_hf_cache = std::env::var("HF_HUB_CACHE").ok();
    let prev_hf_home = std::env::var("HF_HOME").ok();
    let prev_cwd = std::env::current_dir().ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", &store_root);
        std::env::set_var("MLXCEL_DEFAULT_ORG", "acme");
        std::env::set_var("HF_HUB_CACHE", &empty_hf);
        std::env::remove_var("HF_HOME");
    }
    std::env::set_current_dir(&run_dir).unwrap();

    let resolved = resolve_model_source(Path::new("my-model"));

    if let Some(cwd) = prev_cwd {
        let _ = std::env::set_current_dir(cwd);
    }
    restore_env("MLXCEL_CACHE_DIR", prev_cache_dir);
    restore_env("MLXCEL_DEFAULT_ORG", prev_default_org);
    restore_env("HF_HUB_CACHE", prev_hf_cache);
    restore_env("HF_HOME", prev_hf_home);

    assert_eq!(resolved.unwrap(), store_dir);
}

#[test]
fn default_org_falls_back_when_unset_or_blank() {
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_DEFAULT_ORG").ok();

    unsafe { std::env::remove_var("MLXCEL_DEFAULT_ORG") };
    assert_eq!(default_org(), "mlx-community");

    unsafe { std::env::set_var("MLXCEL_DEFAULT_ORG", "   ") };
    assert_eq!(default_org(), "mlx-community");

    unsafe { std::env::set_var("MLXCEL_DEFAULT_ORG", "acme") };
    assert_eq!(default_org(), "acme");

    restore_env("MLXCEL_DEFAULT_ORG", prev);
}

#[test]
fn bare_name_with_invalid_default_org_errors_without_network() {
    // A slash in MLXCEL_DEFAULT_ORG would yield a multi-segment repo-id; the
    // resolver rejects it up front rather than attempting a malformed download.
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_DEFAULT_ORG").ok();
    unsafe { std::env::set_var("MLXCEL_DEFAULT_ORG", "owner/extra") };

    let err = resolve_model_source(Path::new("my-model")).unwrap_err();
    let msg = format!("{err}");

    restore_env("MLXCEL_DEFAULT_ORG", prev);

    assert!(msg.contains("MLXCEL_DEFAULT_ORG"), "got: {msg}");
    assert!(msg.contains("invalid repo-id"), "got: {msg}");
}

// ── normalize_repo_id (issue #171): shared bare-name → default-org expansion ──
//
// `normalize_repo_id` is the funnel the `download` verb uses so a bare,
// prefix-less name expands to `<default-org>/<name>` exactly like resolver
// step 3. These mirror the branch-3 (#112) cases above but exercise the shared
// helper directly (no filesystem / network), asserting only the returned id.

#[test]
fn normalize_bare_name_prepends_default_org() {
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_DEFAULT_ORG").ok();
    unsafe { std::env::remove_var("MLXCEL_DEFAULT_ORG") };

    let resolved = normalize_repo_id("gemma-4-26B-A4B-it-qat-4bit").unwrap();

    restore_env("MLXCEL_DEFAULT_ORG", prev);
    assert_eq!(resolved, "mlx-community/gemma-4-26B-A4B-it-qat-4bit");
}

#[test]
fn normalize_bare_name_honors_default_org_override() {
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_DEFAULT_ORG").ok();
    unsafe { std::env::set_var("MLXCEL_DEFAULT_ORG", "acme") };

    let resolved = normalize_repo_id("my-model").unwrap();

    restore_env("MLXCEL_DEFAULT_ORG", prev);
    assert_eq!(resolved, "acme/my-model");
}

#[test]
fn normalize_bare_name_falls_back_on_blank_default_org() {
    // A blank / whitespace-only override falls back to the `mlx-community`
    // default, matching `default_org`.
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_DEFAULT_ORG").ok();
    unsafe { std::env::set_var("MLXCEL_DEFAULT_ORG", "   ") };

    let resolved = normalize_repo_id("my-model").unwrap();

    restore_env("MLXCEL_DEFAULT_ORG", prev);
    assert_eq!(resolved, "mlx-community/my-model");
}

#[test]
fn normalize_passes_through_owner_name_unchanged() {
    // An explicit `owner/name` id (or any multi-slash value) is not a bare
    // segment, so it is returned verbatim — idempotent, no second org prefix.
    // The default-org env var is irrelevant here; we pin it to a non-default
    // value to prove the expansion branch never runs for a slash-bearing id.
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_DEFAULT_ORG").ok();
    unsafe { std::env::set_var("MLXCEL_DEFAULT_ORG", "acme") };

    assert_eq!(
        normalize_repo_id("mlx-community/Qwen3-4B-4bit").unwrap(),
        "mlx-community/Qwen3-4B-4bit"
    );
    assert_eq!(normalize_repo_id("a/b/c").unwrap(), "a/b/c");

    restore_env("MLXCEL_DEFAULT_ORG", prev);
}

#[test]
fn normalize_bare_name_with_invalid_default_org_errors() {
    // A `/` in MLXCEL_DEFAULT_ORG expands the bare name into a multi-segment
    // (invalid) repo-id; the normalizer rejects it up front with the same
    // actionable `bad_default_org_error` the resolver uses.
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_DEFAULT_ORG").ok();
    unsafe { std::env::set_var("MLXCEL_DEFAULT_ORG", "owner/extra") };

    let err = normalize_repo_id("my-model").unwrap_err();
    let msg = format!("{err}");

    restore_env("MLXCEL_DEFAULT_ORG", prev);

    assert!(msg.contains("MLXCEL_DEFAULT_ORG"), "got: {msg}");
    assert!(msg.contains("invalid repo-id"), "got: {msg}");
}
