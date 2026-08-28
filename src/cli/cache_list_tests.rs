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

//! Unit tests for `--cache-list` (issue #1448).

use super::*;

/// Create `<root>/<repo_id>` and drop a `config.json` in it.
fn seed_checkpoint(root: &Path, repo_id: &str) {
    let dir = root.join(repo_id);
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("config.json"), "{}").expect("write config");
}

#[test]
fn an_empty_store_reports_zero_models_in_b10621s_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_eq!(
        render_cache_list(Some(dir.path())),
        "number of models in cache: 0\n"
    );
}

#[test]
fn an_unresolvable_store_root_reports_an_empty_cache() {
    assert_eq!(render_cache_list(None), "number of models in cache: 0\n");
}

#[test]
fn the_rendered_layout_matches_the_pinned_binarys_output_shape() {
    // Verified against the pinned macOS arm64 llama-server, which prints a
    // header line then `%4zu. %s` per entry.
    let dir = tempfile::tempdir().expect("tempdir");
    seed_checkpoint(dir.path(), "mlx-community/Qwen3-4B-4bit");
    seed_checkpoint(dir.path(), "mlx-community/gemma-3-4b-it-4bit");

    let expected = concat!(
        "number of models in cache: 2\n",
        "   1. mlx-community/Qwen3-4B-4bit\n",
        "   2. mlx-community/gemma-3-4b-it-4bit\n",
    );
    assert_eq!(render_cache_list(Some(dir.path())), expected);
}

#[test]
fn a_safetensors_only_directory_counts_as_a_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = dir.path().join("mlx-community/no-config");
    std::fs::create_dir_all(&model).expect("mkdir");
    std::fs::write(model.join("model.safetensors"), b"").expect("write");

    assert_eq!(
        cached_model_ids(dir.path()),
        vec!["mlx-community/no-config".to_owned()]
    );
}

#[test]
fn a_directory_holding_no_checkpoint_is_not_listed() {
    // A half-finished download or an unrelated directory under the store root
    // would otherwise be offered as a model that cannot load.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("mlx-community/incomplete")).expect("mkdir");
    std::fs::write(dir.path().join("mlx-community/incomplete/README.md"), "wip").expect("write");

    assert!(cached_model_ids(dir.path()).is_empty());
}

#[test]
fn a_bare_repo_id_without_an_owner_is_listed_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    seed_checkpoint(dir.path(), "gpt2");
    assert_eq!(cached_model_ids(dir.path()), vec!["gpt2".to_owned()]);
}

#[test]
fn a_checkpoint_is_never_descended_into() {
    // A checkpoint whose own subdirectory holds shards must be listed once,
    // under its own id, not once per subdirectory.
    let dir = tempfile::tempdir().expect("tempdir");
    seed_checkpoint(dir.path(), "gpt2");
    let nested = dir.path().join("gpt2").join("shards");
    std::fs::create_dir_all(&nested).expect("mkdir");
    std::fs::write(nested.join("model.safetensors"), b"").expect("write");

    assert_eq!(cached_model_ids(dir.path()), vec!["gpt2".to_owned()]);
}

#[test]
fn hidden_directories_are_skipped() {
    // `.locks` and `.cache` are HuggingFace bookkeeping, not checkpoints.
    let dir = tempfile::tempdir().expect("tempdir");
    seed_checkpoint(dir.path(), ".cache/huggingface");
    seed_checkpoint(dir.path(), "mlx-community/.tmp-download");
    seed_checkpoint(dir.path(), "mlx-community/Qwen3-4B-4bit");

    assert_eq!(
        cached_model_ids(dir.path()),
        vec!["mlx-community/Qwen3-4B-4bit".to_owned()]
    );
}

#[test]
fn the_listing_is_sorted_so_the_output_is_stable() {
    let dir = tempfile::tempdir().expect("tempdir");
    for repo in ["zz/last", "aa/first", "mm/middle"] {
        seed_checkpoint(dir.path(), repo);
    }
    assert_eq!(
        cached_model_ids(dir.path()),
        vec![
            "aa/first".to_owned(),
            "mm/middle".to_owned(),
            "zz/last".to_owned()
        ]
    );
}

#[test]
fn every_listed_id_is_a_repository_id_the_model_flag_accepts() {
    let dir = tempfile::tempdir().expect("tempdir");
    seed_checkpoint(dir.path(), "mlx-community/Qwen3-4B-4bit");
    for id in cached_model_ids(dir.path()) {
        assert!(!id.starts_with('/'), "{id} is a path, not a repository id");
        assert!(!id.ends_with('/'), "{id} has a trailing separator");
        assert!(
            id.matches('/').count() <= 1,
            "{id} is not an <owner>/<name> repository id"
        );
    }
}
