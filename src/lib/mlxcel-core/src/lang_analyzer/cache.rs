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

//! Disk cache for `TokenLanguageIndex` (B4 — vocab-hash keyed, postcard 1.x).
//!
//! # Cache key
//! `vocab_hash = hex(sha256(tokenizer.json bytes))[..16]`
//!
//! # Location
//! `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/tokenizer-scripts/<vocab_hash>.bin`
//!
//! # Invalidation rules (§7.4)
//! - File missing → build and write.
//! - `version` field mismatch → rebuild and overwrite.
//! - `--lang-bias-rebuild-cache` / `rebuild: bool` → force rebuild.
//! - Corrupted postcard data → rename to `*.broken.<epoch>.bak` then rebuild.

use std::path::PathBuf;

use tokenizers::Tokenizer;

use super::{LangAnalyzerError, TokenLanguageIndex, CURRENT_VERSION};

/// Sub-directory under the mlxcel cache root that holds tokenizer index files.
pub const CACHE_SUBDIR: &str = "tokenizer-scripts";

// ============================================================================
// Cache root resolution
// ============================================================================

/// Resolve the mlxcel cache root directory.
///
/// Reads `MLXCEL_CACHE_DIR` from the environment; falls back to
/// `$HOME/.cache/mlxcel` via [`dirs::home_dir`]. Returns `None` only when
/// neither `MLXCEL_CACHE_DIR` nor a home directory can be determined.
///
/// This is the single source of truth for the cache root across the codebase.
/// The tokenizer language-analysis disk cache stores its files under
/// `cache_root()/tokenizer-scripts/`, and the downloader's global model store
/// (issue #93) stores model snapshots under `cache_root()/models/`. Sharing
/// one resolver keeps the `MLXCEL_CACHE_DIR` override semantics identical for
/// every consumer.
pub fn cache_root() -> Option<PathBuf> {
    std::env::var_os("MLXCEL_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache/mlxcel")))
}

// ============================================================================
// Cache path resolution
// ============================================================================

/// Resolve the cache file path for a given `vocab_hash`.
///
/// Reads `MLXCEL_CACHE_DIR` from the environment; falls back to
/// `$HOME/.cache/mlxcel` via [`dirs::home_dir`].
///
/// # Panics
/// Panics only if neither `MLXCEL_CACHE_DIR` nor a home directory can be
/// determined. In practice this should not happen on any supported platform.
pub fn cache_path(vocab_hash: &str) -> PathBuf {
    let base = cache_root().expect("no home directory and MLXCEL_CACHE_DIR not set");
    base.join(CACHE_SUBDIR).join(format!("{vocab_hash}.bin"))
}

// ============================================================================
// try_load
// ============================================================================

/// Attempt to load a cached `TokenLanguageIndex` for `vocab_hash`.
///
/// Returns `Some(index)` only when:
/// - The cache file exists.
/// - The file deserializes without error.
/// - The stored `version` equals [`CURRENT_VERSION`].
///
/// On a version mismatch the corrupted/stale file is left in place (the
/// caller will overwrite it via [`save`]).
///
/// On a **postcard decode failure** the corrupt file is renamed to
/// `<original>.broken.<epoch_secs>.bak` before returning `None`, so the
/// caller can build fresh without worrying about re-encountering the same
/// corrupt bytes.
pub fn try_load(vocab_hash: &str) -> Option<TokenLanguageIndex> {
    let path = cache_path(vocab_hash);
    let bytes = std::fs::read(&path).ok()?;

    match postcard::from_bytes::<TokenLanguageIndex>(&bytes) {
        Ok(idx) if idx.version == CURRENT_VERSION => Some(idx),
        Ok(_) => {
            // Version mismatch — stale cache. Leave the file; the caller will
            // overwrite it.
            None
        }
        Err(_) => {
            // Corrupted file — rename it aside so it does not interfere with
            // future loads.
            let epoch = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let mut bak_path = path.clone();
            // Build extension: <hash>.bin → <hash>.bin.broken.<epoch>.bak
            let mut ext = path
                .extension()
                .map(|e| e.to_string_lossy().into_owned())
                .unwrap_or_default();
            if !ext.is_empty() {
                ext.push('.');
            }
            ext.push_str(&format!("broken.{epoch}.bak"));
            bak_path.set_extension(ext);
            let _ = std::fs::rename(&path, &bak_path);
            None
        }
    }
}

// ============================================================================
// save
// ============================================================================

/// Serialize `index` to the cache file for `index.vocab_hash`, using an
/// atomic write (temp file → rename).
///
/// Creates all intermediate directories if they do not exist.
pub fn save(index: &TokenLanguageIndex) -> Result<(), LangAnalyzerError> {
    let path = cache_path(&index.vocab_hash);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = postcard::to_allocvec(index)?;
    // Write to a sibling temp file first to ensure atomicity.
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

// ============================================================================
// load_or_build
// ============================================================================

/// One-stop helper used by downstream (B8) to obtain a `TokenLanguageIndex`.
///
/// # Algorithm
/// 1. Compute `vocab_hash` cheaply from `tokenizer_json_bytes`.
/// 2. Unless `rebuild` is `true`, attempt to load from disk via [`try_load`].
/// 3. On cache miss (or forced rebuild), call [`TokenLanguageIndex::build`]
///    and persist the result via [`save`].
///
/// The `tokenizer_json_bytes` parameter must be the raw bytes of the model's
/// `tokenizer.json` (the same bytes used to construct `tokenizer`). They are
/// needed to compute the `vocab_hash` and to pass to `build`.
pub fn load_or_build(
    tokenizer: &Tokenizer,
    tokenizer_json_bytes: &[u8],
    rebuild: bool,
) -> Result<TokenLanguageIndex, LangAnalyzerError> {
    let hash = TokenLanguageIndex::compute_vocab_hash(tokenizer_json_bytes);

    if !rebuild {
        if let Some(idx) = try_load(&hash) {
            return Ok(idx);
        }
    }

    let idx = TokenLanguageIndex::build(tokenizer, tokenizer_json_bytes)?;
    save(&idx)?;
    Ok(idx)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Tests that mutate `MLXCEL_CACHE_DIR` (or any other env var) must
    // serialize through the crate-wide `ENV_LOCK` from
    // `mlxcel_core::test_support::env_lock`. Per-module locks would race
    // with env mutations in unrelated modules of the same test binary —
    // libc's env block has no internal lock and concurrent
    // `setenv`/`getenv` is undefined behavior.
    use crate::test_support::env_lock::env_lock;

    // -----------------------------------------------------------------------
    // Per-test unique mock tokenizer JSONs
    //
    // Each test uses a distinct `marker_*` vocab key so that the SHA-256
    // hash over the bytes differs per test. This prevents accidental hash
    // collisions between parallel tests.
    // -----------------------------------------------------------------------

    fn mock_json(marker: &str) -> String {
        format!(
            r#"{{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
    {{"id": 1, "content": "<s>",   "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {{
    "type": "WordLevel",
    "vocab": {{
      "<unk>": 0, "<s>": 1,
      "hello": 2, "world": 3, "test": 4,
      "{marker}": 5
    }},
    "unk_token": "<unk>"
  }}
}}"#
        )
    }

    fn make_tokenizer(json: &str) -> Tokenizer {
        Tokenizer::from_bytes(json.as_bytes()).expect("failed to parse mock tokenizer JSON")
    }

    fn build_test_index(json: &str) -> TokenLanguageIndex {
        let tok = make_tokenizer(json);
        TokenLanguageIndex::build(&tok, json.as_bytes()).expect("build should succeed")
    }

    // -------------------------------------------------------------------------
    // cache_path_uses_mlxcel_cache_dir_override
    // -------------------------------------------------------------------------

    /// When `MLXCEL_CACHE_DIR` is set, `cache_path` must use it as the root.
    #[test]
    fn cache_path_uses_mlxcel_cache_dir_override() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let override_dir = tmp.path().to_path_buf();

        let _guard = env_lock();
        std::env::set_var("MLXCEL_CACHE_DIR", &override_dir);
        let path = cache_path("deadbeef12345678");
        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        assert!(
            path.starts_with(&override_dir),
            "expected path to start with {override_dir:?}, got {path:?}"
        );
        assert!(
            path.to_string_lossy().contains("tokenizer-scripts"),
            "expected 'tokenizer-scripts' sub-dir in path"
        );
        assert!(
            path.to_string_lossy().ends_with("deadbeef12345678.bin"),
            "expected filename to be <hash>.bin"
        );
    }

    // -------------------------------------------------------------------------
    // cache_roundtrip
    // -------------------------------------------------------------------------

    /// `save` followed by `try_load` must return an equivalent index.
    #[test]
    fn cache_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let json = mock_json("marker_roundtrip");
        let idx = build_test_index(&json);

        let _guard = env_lock();
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());
        save(&idx).expect("save should succeed");
        let loaded = try_load(&idx.vocab_hash);
        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        let loaded = loaded.expect("try_load should return Some after save");
        assert_eq!(loaded.vocab_hash, idx.vocab_hash);
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert_eq!(loaded.tokens.len(), idx.tokens.len());
    }

    // -------------------------------------------------------------------------
    // cache_version_mismatch_rebuilds
    // -------------------------------------------------------------------------

    /// A cache file with a `version` field != `CURRENT_VERSION` must cause
    /// `try_load` to return `None` (stale cache).
    #[test]
    fn cache_version_mismatch_returns_none() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let json = mock_json("marker_version_mismatch");

        // Build a valid index but manually set version to a fake future value.
        let mut idx = build_test_index(&json);
        idx.version = CURRENT_VERSION + 99; // future version — cache is stale

        let _guard = env_lock();
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());
        save(&idx).expect("save should succeed");
        let result = try_load(&idx.vocab_hash);
        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        assert!(
            result.is_none(),
            "try_load should return None on version mismatch"
        );
    }

    /// a v2 cache file must auto-invalidate and rebuild to v3.
    ///
    /// Simulates a earlier deployment: writes a cache file carrying the old
    /// `version = 2` tag and exercises `load_or_build` with `rebuild=false`.
    /// The old file must be discarded and a fresh v3 index written in its
    /// place. This is the key backward-compatibility guarantee for Phase 1
    /// users who upgrade to the byte-fragment-aware build.
    #[test]
    fn cache_v2_auto_rebuilds_to_v3() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let json = mock_json("marker_v2_to_v3");
        let json_bytes = json.as_bytes().to_vec();
        let tok = make_tokenizer(&json);

        // Build a valid index, then stamp it as v2 and save.
        let mut idx_v2 = build_test_index(&json);
        idx_v2.version = 2;

        let _guard = env_lock();
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());
        save(&idx_v2).expect("save v2 cache");

        // `try_load` must refuse the v2 cache.
        let loaded = try_load(&idx_v2.vocab_hash);
        assert!(
            loaded.is_none(),
            "try_load must reject v2 cache once CURRENT_VERSION advances"
        );

        // `load_or_build` must rebuild to the current version.
        let rebuilt =
            load_or_build(&tok, &json_bytes, false).expect("load_or_build should rebuild v2→v3");
        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        assert_eq!(
            rebuilt.version, CURRENT_VERSION,
            "rebuilt cache must carry CURRENT_VERSION (now 3): got {}",
            rebuilt.version
        );
        // The on-disk file must have been rewritten (not carrying v2 anymore).
        assert_eq!(rebuilt.vocab_hash, idx_v2.vocab_hash);
    }

    // -------------------------------------------------------------------------
    // cache_corrupted_moves_aside_and_none
    // -------------------------------------------------------------------------

    /// Writing garbage bytes to the cache path must cause `try_load` to:
    /// 1. Return `None`.
    /// 2. Rename the corrupt file to a `.broken.<epoch>.bak` sibling.
    #[test]
    fn cache_corrupted_moves_aside_and_none() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let hash = "aabbccddeeff0011";

        let _guard = env_lock();
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());
        let path = cache_path(hash);
        std::fs::create_dir_all(path.parent().unwrap()).expect("create dirs");
        std::fs::write(&path, b"not valid postcard data!!!").expect("write garbage");
        let result = try_load(hash);
        let path_still_exists = path.exists();
        let cache_dir = path.parent().unwrap().to_path_buf();
        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        assert!(
            result.is_none(),
            "try_load should return None on corrupt file"
        );

        // The original file must no longer exist.
        assert!(
            !path_still_exists,
            "corrupt cache file should have been renamed away"
        );

        // A .broken.<epoch>.bak sibling must exist.
        let bak_exists = std::fs::read_dir(&cache_dir)
            .expect("read cache dir")
            .filter_map(|e| e.ok())
            .any(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.contains("broken") && name.ends_with(".bak")
            });
        assert!(bak_exists, "a .broken.*.bak sibling should exist");
    }

    // -------------------------------------------------------------------------
    // cache_load_or_build_hits_disk_second_time
    // -------------------------------------------------------------------------

    /// The second call to `load_or_build` must load from disk without invoking
    /// `TokenLanguageIndex::build` again. We verify this by comparing the
    /// modification time of the cache file: it must not change between the
    /// two calls.
    #[test]
    fn cache_load_or_build_hits_disk_second_time() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let json = mock_json("marker_disk_hit");
        let json_bytes = json.as_bytes().to_vec();
        let tok = make_tokenizer(&json);

        let _guard = env_lock();
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());

        // First call — must build and save.
        let idx1 =
            load_or_build(&tok, &json_bytes, false).expect("first load_or_build should succeed");
        let path = cache_path(&idx1.vocab_hash);
        let mtime1 = std::fs::metadata(&path)
            .expect("cache file must exist after first call")
            .modified()
            .expect("mtime");

        // Second call — must load from disk (no rebuild, mtime unchanged).
        let idx2 =
            load_or_build(&tok, &json_bytes, false).expect("second load_or_build should succeed");
        let mtime2 = std::fs::metadata(&path)
            .expect("cache file must still exist")
            .modified()
            .expect("mtime");

        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        assert_eq!(
            idx1.vocab_hash, idx2.vocab_hash,
            "both calls must produce the same vocab_hash"
        );
        assert_eq!(
            mtime1, mtime2,
            "cache file must not be rewritten on second call (disk hit expected)"
        );
    }

    // -------------------------------------------------------------------------
    // cache_rebuild_flag_forces_rebuild
    // -------------------------------------------------------------------------

    /// When `rebuild = true`, `load_or_build` must overwrite the existing
    /// cache file even if a valid cache already exists.
    #[test]
    fn cache_rebuild_flag_forces_rebuild() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let json = mock_json("marker_rebuild_flag");
        let json_bytes = json.as_bytes().to_vec();
        let tok = make_tokenizer(&json);

        let _guard = env_lock();
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());

        // First call — build and save.
        let idx1 = load_or_build(&tok, &json_bytes, false).expect("first load_or_build");
        let path = cache_path(&idx1.vocab_hash);
        let mtime1 = std::fs::metadata(&path)
            .expect("cache file must exist")
            .modified()
            .expect("mtime");

        // Sleep briefly to ensure the OS clock advances enough for mtime to differ.
        // Most filesystems have at least 1-second mtime resolution.
        std::thread::sleep(std::time::Duration::from_millis(1100));

        // Second call with rebuild=true — must overwrite.
        let idx2 = load_or_build(&tok, &json_bytes, true).expect("second load_or_build (rebuild)");
        let mtime2 = std::fs::metadata(&path)
            .expect("cache file must still exist")
            .modified()
            .expect("mtime");

        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        assert_eq!(
            idx1.vocab_hash, idx2.vocab_hash,
            "rebuild must produce the same vocab_hash"
        );
        assert_ne!(
            mtime1, mtime2,
            "cache file must be rewritten when rebuild=true"
        );
    }

    // -------------------------------------------------------------------------
    // Integration criterion — real tokenizer smoke test
    // -------------------------------------------------------------------------

    /// Loads a real `tokenizer.json` from a model in `models/`, runs
    /// `load_or_build` twice, and confirms the second call hits the disk
    /// cache (mtime unchanged).
    ///
    /// Skipped gracefully when no model is available (CI environments).
    #[test]
    fn cache_load_or_build_real_tokenizer_smoke() {
        let candidates = [
            "models/smollm-135m-4bit/tokenizer.json",
            "models/Qwen2.5-7B-Instruct-4bit/tokenizer.json",
            "models/Meta-Llama-3.1-8B-Instruct-4bit/tokenizer.json",
        ];

        let found = candidates.iter().find(|p| std::path::Path::new(p).exists());
        let Some(tok_path) = found else {
            eprintln!(
                "[skip] no model tokenizer.json found; skipping real-tokenizer cache smoke test"
            );
            return;
        };

        let tmp = tempfile::tempdir().expect("tempdir");
        let json_bytes = std::fs::read(tok_path).expect("read tokenizer.json");
        let tok = Tokenizer::from_bytes(&json_bytes).expect("parse tokenizer");

        let _guard = env_lock();
        std::env::set_var("MLXCEL_CACHE_DIR", tmp.path());

        // First call — build and persist.
        let idx1 =
            load_or_build(&tok, &json_bytes, false).expect("first load_or_build on real tokenizer");
        assert_eq!(idx1.version, CURRENT_VERSION);
        assert_eq!(idx1.vocab_hash.len(), 16);

        let path = cache_path(&idx1.vocab_hash);
        assert!(
            path.exists(),
            "cache file must exist after first load_or_build"
        );
        let mtime1 = std::fs::metadata(&path).unwrap().modified().unwrap();

        // Second call — must load from disk.
        let idx2 = load_or_build(&tok, &json_bytes, false)
            .expect("second load_or_build on real tokenizer");
        let mtime2 = std::fs::metadata(&path).unwrap().modified().unwrap();

        std::env::remove_var("MLXCEL_CACHE_DIR");
        drop(_guard);

        assert_eq!(idx1.vocab_hash, idx2.vocab_hash);
        assert_eq!(
            mtime1, mtime2,
            "second call must not rewrite the cache file (disk hit expected)"
        );

        eprintln!(
            "[ok] real-tokenizer cache smoke: vocab={}, hash={}, path={}",
            idx1.tokens.len(),
            idx1.vocab_hash,
            path.display()
        );
    }
}
