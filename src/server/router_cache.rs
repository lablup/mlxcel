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

//! Router-mode model cache source (llama-server b10621, issue #1438).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;

use crate::downloader::{self, DownloadHooks, TokenMode};

pub const STAGING_DIR: &str = ".mlxcel-staging";

/// How a router downloads a repository into the cache. The production
/// implementation is [`HfRouterDownloader`]; tests substitute a fake that
/// writes files locally and drives the same hooks.
pub trait RouterDownloader: Send + Sync {
    /// Synchronously validate that `repo_id` names a fetchable repository.
    fn validate(&self, repo_id: &str, revision: Option<&str>) -> anyhow::Result<()>;

    /// Download `repo_id` into the models root at `dest_root`, reporting
    /// progress and honoring cancellation through `hooks`. Blocking.
    fn download(
        &self,
        repo_id: &str,
        revision: Option<&str>,
        dest_root: &Path,
        hooks: DownloadHooks,
    ) -> anyhow::Result<()>;
}

/// The real HuggingFace-backed downloader.
#[derive(Debug, Default, Clone, Copy)]
pub struct HfRouterDownloader;

impl RouterDownloader for HfRouterDownloader {
    fn validate(&self, repo_id: &str, revision: Option<&str>) -> anyhow::Result<()> {
        downloader::probe_repo_with_token_mode(repo_id, revision, TokenMode::Anonymous)
    }

    fn download(
        &self,
        repo_id: &str,
        revision: Option<&str>,
        dest_root: &Path,
        hooks: DownloadHooks,
    ) -> anyhow::Result<()> {
        #[cfg(not(unix))]
        {
            let _ = (repo_id, revision, dest_root, hooks);
            anyhow::bail!("router-managed HuggingFace downloads require Unix directory-fd safety");
        }
        #[cfg(unix)]
        {
            let final_dir = downloader::model_dir_with_override(repo_id, Some(dest_root))
                .ok_or_else(|| {
                    anyhow::anyhow!("cannot resolve cache destination for '{repo_id}'")
                })?;
            let mut stage = AnchoredStage::create(dest_root, repo_id)?;
            let begin_publish = hooks.begin_publish.clone();
            let result = downloader::download_repo_to_existing_dir_fd(
                repo_id,
                revision,
                stage.stage_fd(),
                final_dir.clone(),
                hooks,
            );
            if let Err(err) = result {
                stage.cleanup();
                return Err(err);
            }
            if let Some(begin_publish) = begin_publish
                && !begin_publish()
            {
                stage.cleanup();
                return Err(anyhow::Error::new(crate::downloader::DownloadCancelled));
            }
            stage.publish().with_context(|| {
                format!("failed to publish downloaded snapshot for '{repo_id}'")
            })?;
            Ok(())
        }
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

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn list(&self) -> Vec<(String, PathBuf)> {
        crate::downloader::list_models_with_override(Some(&self.root))
            .into_iter()
            .filter(|m| regular_file_exists(&m.path.join("config.json")))
            .map(|m| (m.repo_id, m.path))
            .collect()
    }

    pub fn snapshot_dir(&self, repo_id: &str) -> PathBuf {
        crate::downloader::model_dir_with_override(repo_id, Some(&self.root))
            .unwrap_or_else(|| self.root.join(repo_id))
    }

    pub fn normalize_name(&self, name: &str) -> anyhow::Result<String> {
        downloader::normalize_repo_id(name)
    }

    pub fn validate(&self, repo_id: &str, revision: Option<&str>) -> anyhow::Result<()> {
        self.downloader.validate(repo_id, revision)
    }

    pub fn download(
        &self,
        repo_id: &str,
        revision: Option<&str>,
        hooks: DownloadHooks,
    ) -> anyhow::Result<()> {
        self.downloader
            .download(repo_id, revision, &self.root, hooks)
    }

    pub fn remove(&self, repo_id: &str) -> anyhow::Result<()> {
        #[cfg(not(unix))]
        {
            let _ = repo_id;
            anyhow::bail!("router-managed cache removal requires Unix directory-fd safety");
        }
        #[cfg(unix)]
        {
            match anchored_remove::remove_managed_snapshot(&self.root, repo_id)? {
                Some(removed) => {
                    tracing::info!(
                        "router: removed managed cache model '{repo_id}' ({} bytes) at {}",
                        removed.size_bytes,
                        removed.path.display()
                    );
                    Ok(())
                }
                None => Ok(()),
            }
        }
    }
}

fn regular_file_exists(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_file())
        .unwrap_or(false)
}

#[cfg(unix)]
#[path = "router_cache/anchored_delete.rs"]
mod anchored_delete;

#[cfg(unix)]
#[path = "router_cache/anchored_publish.rs"]
mod anchored_publish;

#[cfg(unix)]
#[path = "router_cache/anchored_remove.rs"]
mod anchored_remove;

#[cfg(unix)]
use anchored_publish::AnchoredStage;
