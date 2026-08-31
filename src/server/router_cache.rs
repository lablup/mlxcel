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

//! Router-mode model cache source (llama-server b10621, issue #1438).
//!
//! b10621's router lists its download cache as removable model entries
//! (`source: "cache"`, `can_remove: true`), lets `POST /models` download a
//! new HuggingFace repository into that cache, and lets `DELETE /models`
//! remove a cache entry from disk. mlxcel's equivalent cache is its model
//! store (the directory `--model-store-root` / `MLXCEL_MODELS_DIR` /
//! `MLXCEL_CACHE_DIR` resolve, holding `<owner>/<name>` MLX snapshots), so
//! the router's cache source wraps that store.
//!
//! Confinement: every path in and out of the store goes through
//! [`crate::downloader::store`]'s sanitized `<owner>/<name>` composition, and
//! removal re-asserts containment under the store root before deleting
//! (`remove_model_with_override`), so an HTTP-supplied model name can never
//! escape the cache directory in either direction. The downloader is behind a
//! trait so tests exercise the full download flow without the network.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::downloader::{self, DownloadHooks, DownloadOptions};

/// How a router downloads a repository into the cache. The production
/// implementation is [`HfRouterDownloader`]; tests substitute a fake that
/// writes files locally and drives the same hooks.
pub trait RouterDownloader: Send + Sync {
    /// Synchronously validate that `repo_id` names a fetchable repository
    /// (b10621 validates by fetching metadata before answering `POST
    /// /models`). Blocking; called from a blocking-capable thread.
    fn validate(&self, repo_id: &str) -> anyhow::Result<()>;

    /// Download `repo_id` into the models root at `dest_root`, reporting
    /// progress and honoring cancellation through `hooks`. Blocking.
    fn download(&self, repo_id: &str, dest_root: &Path, hooks: DownloadHooks)
    -> anyhow::Result<()>;
}

/// The real HuggingFace-backed downloader.
#[derive(Debug, Default, Clone, Copy)]
pub struct HfRouterDownloader;

impl RouterDownloader for HfRouterDownloader {
    fn validate(&self, repo_id: &str) -> anyhow::Result<()> {
        downloader::probe_repo(repo_id, None)
    }

    fn download(
        &self,
        repo_id: &str,
        dest_root: &Path,
        hooks: DownloadHooks,
    ) -> anyhow::Result<()> {
        downloader::download_repo_with_hooks(
            DownloadOptions {
                repo_id: repo_id.to_string(),
                local_dir: None,
                models_dir: Some(dest_root.to_path_buf()),
                revision: None,
                token: None,
                include: Vec::new(),
                force: false,
            },
            hooks,
        )
    }
}

/// The router's model cache: the mlxcel model store plus a downloader.
pub struct CacheSource {
    root: PathBuf,
    downloader: Arc<dyn RouterDownloader>,
}

impl std::fmt::Debug for CacheSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheSource")
            .field("root", &self.root)
            .finish()
    }
}

impl CacheSource {
    pub fn new(root: PathBuf, downloader: Arc<dyn RouterDownloader>) -> Self {
        Self { root, downloader }
    }

    /// The models root this cache lists, downloads into, and removes from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Enumerate complete snapshots as `(repo_id, path)` pairs. Uses the
    /// store's own listing (a directory counts only when it holds a
    /// `config.json`), so a half-finished download is not offered as a model.
    pub fn list(&self) -> Vec<(String, PathBuf)> {
        crate::downloader::list_models_with_override(Some(&self.root))
            .into_iter()
            .map(|m| (m.repo_id, m.path))
            .collect()
    }

    /// The snapshot directory `repo_id` would occupy (whether or not it
    /// exists yet). Repo-id segments are sanitized by the store so the
    /// composed path cannot escape the root.
    pub fn snapshot_dir(&self, repo_id: &str) -> PathBuf {
        crate::downloader::model_dir_with_override(repo_id, Some(&self.root))
            .unwrap_or_else(|| self.root.join(repo_id))
    }

    /// Normalize a requested model name into the repo id used as the cache
    /// entry name (bare names expand to the default organization, exactly as
    /// `-m <name>` resolution does).
    pub fn normalize_name(&self, name: &str) -> anyhow::Result<String> {
        downloader::normalize_repo_id(name)
    }

    /// Validate that `repo_id` is fetchable (metadata probe, no file
    /// downloads). Blocking.
    pub fn validate(&self, repo_id: &str) -> anyhow::Result<()> {
        self.downloader.validate(repo_id)
    }

    /// Download `repo_id` into the cache. Blocking; run on a worker thread.
    pub fn download(&self, repo_id: &str, hooks: DownloadHooks) -> anyhow::Result<()> {
        self.downloader.download(repo_id, &self.root, hooks)
    }

    /// Remove `repo_id`'s snapshot from the cache. Deleting is contained to
    /// the store root by `remove_model_under`'s re-assertion; a missing
    /// snapshot (for example a cancelled download that never completed a
    /// file) is not an error, matching b10621's best-effort
    /// `common_download_remove`.
    pub fn remove(&self, repo_id: &str) -> anyhow::Result<()> {
        use crate::downloader::{RemoveOutcome, remove_model_with_override};
        match remove_model_with_override(repo_id, None, Some(&self.root)) {
            Ok(RemoveOutcome::Removed { path, size_bytes }) => {
                tracing::info!(
                    "router: removed cache model '{repo_id}' ({size_bytes} bytes) at {}",
                    path.display()
                );
                Ok(())
            }
            Ok(RemoveOutcome::HfCacheOnly { hf_path }) => {
                // The read-only HuggingFace cache is not ours to manage; the
                // router's cache never lists it, so this arm is unreachable
                // from HTTP. Refuse rather than pretend.
                anyhow::bail!(
                    "model '{repo_id}' only exists in the read-only HuggingFace cache at {}",
                    hf_path.display()
                )
            }
            Ok(RemoveOutcome::NotFound) => Ok(()),
            Err(err) => Err(anyhow::anyhow!(err.to_string())),
        }
    }
}
