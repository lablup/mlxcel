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

//! `--cache-list` / `-cl`: the checkpoints in mlxcel's model store, in
//! b10621's output format (issue #1448, epic #1431).
//!
//! b10621 prints
//!
//! ```text
//! number of models in cache: 2
//!    1. ggml-org/embeddinggemma-300M-GGUF:Q8_0
//!    2. ggml-org/Qwen3-0.6B-GGUF:Q8_0
//! ```
//!
//! (verified against the pinned macOS arm64 binary), which is a header line
//! plus `%4zu. %s` per entry, then `exit(0)`. This reproduces exactly that
//! shape over mlxcel's own store, which is the directory
//! `crate::downloader::models_root` resolves: `<root>/<owner>/<name>`
//! snapshots of MLX SafeTensors checkpoints, rather than llama.cpp's GGUF
//! cache. Each entry is the repository id mlxcel's `-m` accepts, so the
//! output is directly copy-pasteable into the next command.
//!
//! A directory only counts as a checkpoint when it actually holds one
//! (`config.json` or a `*.safetensors` file). A half-finished download or an
//! unrelated directory under the store root would otherwise be offered as a
//! model that cannot load.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Upstream reference: <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>

use std::path::Path;

/// True when `dir` holds a loadable MLX checkpoint.
fn is_checkpoint(dir: &Path) -> bool {
    if dir.join("config.json").is_file() {
        return true;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
    })
}

/// Sorted repository ids of every checkpoint under `root`.
///
/// Both layouts the store produces are recognised: `<owner>/<name>` for a
/// normal HuggingFace id, and a bare `<name>` for an id without a slash. A
/// directory that is itself a checkpoint is never also descended into, so a
/// checkpoint whose own subdirectory happens to hold safetensors shards is
/// listed once.
#[must_use]
pub fn cached_model_ids(root: &Path) -> Vec<String> {
    let mut ids = Vec::new();
    let Ok(top_level) = std::fs::read_dir(root) else {
        return ids;
    };
    for owner_entry in top_level.flatten() {
        let owner_path = owner_entry.path();
        if !owner_path.is_dir() {
            continue;
        }
        let owner = owner_entry.file_name().to_string_lossy().into_owned();
        if owner.starts_with('.') {
            continue;
        }
        if is_checkpoint(&owner_path) {
            ids.push(owner);
            continue;
        }
        let Ok(models) = std::fs::read_dir(&owner_path) else {
            continue;
        };
        for model_entry in models.flatten() {
            let model_path = model_entry.path();
            if !model_path.is_dir() {
                continue;
            }
            let name = model_entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            if is_checkpoint(&model_path) {
                ids.push(format!("{owner}/{name}"));
            }
        }
    }
    ids.sort();
    ids
}

/// Render the `--cache-list` report for an already-resolved store root.
///
/// `root` is `None` when neither `MLXCEL_MODELS_DIR`, `MLXCEL_CACHE_DIR`, nor
/// a home directory could be resolved; that is reported as an empty cache
/// rather than an error, exactly as b10621 reports an absent cache directory.
#[must_use]
pub fn render_cache_list(root: Option<&Path>) -> String {
    let ids = root.map(cached_model_ids).unwrap_or_default();
    let mut out = format!("number of models in cache: {}\n", ids.len());
    for (index, id) in ids.iter().enumerate() {
        out.push_str(&format!("{:>4}. {id}\n", index + 1));
    }
    out
}

/// Render the `--cache-list` report for the live model store.
///
/// `models_dir` is the inline `--models-dir <path>` flag, so `--cache-list`
/// reports the store the same invocation would load from rather than the
/// default one.
#[must_use]
pub fn render_store_cache_list(models_dir: Option<&Path>) -> String {
    let root = crate::downloader::models_root(models_dir);
    render_cache_list(root.as_deref())
}

#[cfg(test)]
#[path = "cache_list_tests.rs"]
mod tests;
