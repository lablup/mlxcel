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

//! Shared clap argument struct for the `download` subcommand.
//!
//! Both `mlxcel` (`src/main.rs`) and `mlxcel-server`
//! (`src/bin/mlx_server.rs`) flatten this same struct under their respective
//! subcommand wrappers so the flag surface stays identical between binaries
//! without manual sync.

use clap::Args;
use std::path::PathBuf;

/// Arguments for the `download` subcommand shared by `mlxcel` and
/// `mlxcel-server`.
///
/// The struct is intentionally narrow: every field maps 1:1 to a field on
/// [`crate::downloader::DownloadOptions`], and the binary entry points are
/// thin shims that call [`crate::downloader::download_repo`].
#[derive(Args, Debug, Clone)]
#[command(after_help = "\
Examples:
  # Default destination is the global store at
  # ${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>:
  mlxcel download mlx-community/Qwen3-4B-4bit
  # -> ~/.cache/mlxcel/models/mlx-community/Qwen3-4B-4bit/

  # If the repo is already in your HuggingFace cache (HF_HUB_CACHE / HF_HOME),
  # the download is skipped and the existing snapshot is reused.

  # Explicit local directory (opt-out of the global store):
  mlxcel download mlx-community/Qwen3-4B-4bit --local-dir /tmp/qwen

  # Specific revision (branch, tag, or commit hash):
  mlxcel download mlx-community/Qwen3-4B-4bit --revision main

  # Gated repo with auth token (also reads HF_TOKEN / HUGGING_FACE_HUB_TOKEN):
  mlxcel download meta-llama/Llama-3.1-8B-Instruct --token hf_xxx

  # Force re-download even if files are already present:
  mlxcel download mlx-community/Qwen3-4B-4bit --force")]
pub struct DownloadArgs {
    /// HuggingFace repository id, e.g. `mlx-community/Qwen3-4B-4bit`.
    #[arg(value_name = "REPO_ID")]
    pub repo_id: String,

    /// Local destination directory (opt-out of the global store).
    ///
    /// When omitted, the snapshot lands in the location-independent global
    /// store at `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>`
    /// (e.g. `mlx-community/Qwen3-4B-4bit` ->
    /// `~/.cache/mlxcel/models/mlx-community/Qwen3-4B-4bit`). An existing
    /// HuggingFace cache copy is reused instead of re-downloading. Pass this
    /// flag to write the snapshot to an explicit path instead.
    #[arg(long, value_name = "PATH")]
    pub local_dir: Option<PathBuf>,

    /// Repository revision (branch, tag, or commit hash). Defaults to
    /// `main`.
    #[arg(long, value_name = "REV")]
    pub revision: Option<String>,

    /// HuggingFace authentication token for gated/private repositories.
    ///
    /// When omitted, falls back to the `HF_TOKEN` environment variable, then
    /// `HUGGING_FACE_HUB_TOKEN`, then anonymous access.
    #[arg(long, value_name = "TOKEN")]
    pub token: Option<String>,

    /// Re-download every file even if it already exists locally.
    #[arg(long, default_value_t = false)]
    pub force: bool,
}
