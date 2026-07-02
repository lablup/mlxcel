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

//! Offline snapshot-completeness gate for the load/resolve path (issue #465).
//!
//! The downloader already has a strong completeness gate (`snapshot_complete`)
//! keyed on the network-fetched wanted-set. The resolver, however, historically
//! reused a local snapshot after only checking that `config.json` was present.
//! An interrupted download that fetched `config.json` and only some safetensors
//! shards therefore passed that weak gate, the re-download was skipped, and the
//! model loader died at the first missing weight with a bare
//! `Weight not found: ...`.
//!
//! [`classify_snapshot`] closes that gap **without a network round-trip** by
//! deriving the expected weight set from the snapshot's own
//! `model.safetensors.index.json`. It deliberately mirrors the loader's
//! stale-index tolerance ([`mlxcel_core::weights`]): a repackaged mlx-community
//! quant that ships the original full-precision index (whose shard names no
//! longer match the on-disk quant files) is reported [`SnapshotState::Complete`]
//! and never re-fetched, exactly as the loader would glob-load it.

use std::collections::HashSet;
use std::path::Path;

/// The manifest file every materialized snapshot has. Its absence means the
/// directory is not a snapshot at all (an empty or unrelated `models/` folder).
const SNAPSHOT_MARKER: &str = "config.json";

/// Single-file weight name used by non-sharded snapshots (no shard index).
const SINGLE_WEIGHT_FILE: &str = "model.safetensors";

/// Sharded-weight manifest name.
const SHARD_INDEX_FILE: &str = "model.safetensors.index.json";

/// Offline completeness verdict for a candidate local snapshot directory,
/// produced by [`classify_snapshot`].
#[derive(Debug, PartialEq, Eq)]
pub(super) enum SnapshotState {
    /// `config.json` is present and every weight file the snapshot needs — each
    /// shard named by a local `model.safetensors.index.json`, or at least one
    /// non-zero `*.safetensors` for a single-file / repackaged layout — is
    /// present and non-zero. Safe to hand to the model loader.
    Complete,
    /// `config.json` is present but one or more weight shards named by the local
    /// index are missing or zero-byte: the hallmark of an interrupted download.
    /// Carries the missing shard names for the re-fetch message.
    Incomplete { missing: Vec<String> },
    /// Not a materialized snapshot at all (not a directory, or no `config.json`).
    Absent,
}

/// Classify a candidate snapshot directory against the full weight set — the
/// load-path completeness gate for issue #465.
///
/// This is the offline counterpart of the downloader's own `snapshot_complete`
/// (which keys on the network-fetched wanted-set): here the expected weights are
/// derived from the snapshot's own `model.safetensors.index.json` so an
/// interrupted download (config.json + only some shards) is caught before the
/// loader is handed a doomed path.
pub(super) fn classify_snapshot(dir: &Path) -> SnapshotState {
    if !dir.is_dir() || !dir.join(SNAPSHOT_MARKER).exists() {
        return SnapshotState::Absent;
    }
    match mlxcel_core::weights::parse_shard_index(dir) {
        // Sharded layout: the index names every weight shard.
        Ok(Some(shards)) => classify_sharded(dir, &shards),
        // No index: single-file (or already-globbed) layout. Complete as long as
        // at least one non-zero `*.safetensors` is present.
        Ok(None) => {
            if has_nonzero_safetensors(dir) {
                SnapshotState::Complete
            } else {
                SnapshotState::Incomplete {
                    missing: vec![SINGLE_WEIGHT_FILE.to_string()],
                }
            }
        }
        // The index exists but does not parse — often a truncated/zero-byte
        // index from an interrupted download. Defer to the loader's glob when
        // any weights are present; otherwise there is nothing loadable.
        Err(_) => {
            if has_nonzero_safetensors(dir) {
                SnapshotState::Complete
            } else {
                SnapshotState::Incomplete {
                    missing: vec![SHARD_INDEX_FILE.to_string()],
                }
            }
        }
    }
}

/// Completeness verdict for a sharded snapshot given the shard names its index
/// references.
fn classify_sharded(dir: &Path, shards: &[String]) -> SnapshotState {
    let missing: Vec<String> = shards
        .iter()
        .filter(|name| !shard_present(dir, name))
        .cloned()
        .collect();
    if missing.is_empty() {
        return SnapshotState::Complete;
    }
    // Some index shards are absent. Distinguish an interrupted download (the
    // on-disk shards are a subset of the index) from a repackaged mlx-community
    // quant whose stale full-precision index names shards that never matched the
    // on-disk quant files. In the latter case the directory holds a
    // `*.safetensors` the index does NOT name; the loader globs those and loads
    // fine, so we must not re-fetch.
    let indexed: HashSet<&str> = shards.iter().map(String::as_str).collect();
    let has_unindexed = on_disk_safetensors(dir)
        .iter()
        .any(|name| !indexed.contains(name.as_str()));
    if has_unindexed {
        SnapshotState::Complete
    } else {
        SnapshotState::Incomplete { missing }
    }
}

/// True when `name` is a plain shard filename that exists in `dir` and is
/// non-zero. Non-plain names (separators, traversal) can never be a legitimate
/// on-disk shard and are treated as missing — mirroring the loader's
/// `validate_index_shards` guard so a malicious index cannot point the check at
/// an out-of-directory file.
fn shard_present(dir: &Path, name: &str) -> bool {
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || Path::new(name).components().count() != 1
    {
        return false;
    }
    super::file_exists_nonempty(&dir.join(name))
}

/// The non-zero `*.safetensors` filenames directly inside `dir` (non-recursive).
/// The shard index (`*.safetensors.index.json`) is excluded because its
/// extension is `json`, not `safetensors`.
fn on_disk_safetensors(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "safetensors")
            && super::file_exists_nonempty(&path)
            && let Some(name) = path.file_name().and_then(|n| n.to_str())
        {
            out.push(name.to_string());
        }
    }
    out
}

/// True when `dir` holds at least one non-zero `*.safetensors` file.
fn has_nonzero_safetensors(dir: &Path) -> bool {
    !on_disk_safetensors(dir).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn write(path: &Path, bytes: &[u8]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    /// Build a `model.safetensors.index.json` whose `weight_map` references the
    /// given shard filenames.
    fn write_index(dir: &Path, shards: &[&str]) {
        let entries: Vec<String> = shards
            .iter()
            .enumerate()
            .map(|(i, s)| format!("\"tensor.{i}\": \"{s}\""))
            .collect();
        let json = format!("{{\"weight_map\": {{{}}}}}", entries.join(", "));
        write(&dir.join(SHARD_INDEX_FILE), json.as_bytes());
    }

    #[test]
    fn absent_when_not_a_dir_or_no_config() {
        let tmp = tempfile::tempdir().unwrap();
        // Missing entirely.
        assert_eq!(
            classify_snapshot(&tmp.path().join("nope")),
            SnapshotState::Absent
        );
        // Directory without config.json.
        let dir = tmp.path().join("snap");
        fs::create_dir_all(&dir).unwrap();
        write(&dir.join("model.safetensors"), b"w");
        assert_eq!(classify_snapshot(&dir), SnapshotState::Absent);
        // A plain file path (config marker check on a file is Absent).
        let file = tmp.path().join("file");
        write(&file, b"x");
        assert_eq!(classify_snapshot(&file), SnapshotState::Absent);
    }

    #[test]
    fn complete_single_file() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("snap");
        write(&dir.join("config.json"), b"{}");
        write(&dir.join("model.safetensors"), b"weights");
        assert_eq!(classify_snapshot(&dir), SnapshotState::Complete);
    }

    #[test]
    fn incomplete_single_file_no_weights() {
        // config.json present but no safetensors and no index: an interrupted
        // download that stopped right after config.json.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("snap");
        write(&dir.join("config.json"), b"{}");
        assert_eq!(
            classify_snapshot(&dir),
            SnapshotState::Incomplete {
                missing: vec![SINGLE_WEIGHT_FILE.to_string()],
            }
        );
    }

    #[test]
    fn complete_sharded_all_present() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("snap");
        write(&dir.join("config.json"), b"{}");
        write_index(
            &dir,
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        write(&dir.join("model-00001-of-00002.safetensors"), b"a");
        write(&dir.join("model-00002-of-00002.safetensors"), b"b");
        assert_eq!(classify_snapshot(&dir), SnapshotState::Complete);
    }

    #[test]
    fn incomplete_sharded_missing_shard() {
        // The reported bug: index references N shards, only some on disk, and
        // the on-disk shards are a subset of the index (interrupted download).
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("snap");
        write(&dir.join("config.json"), b"{}");
        write_index(
            &dir,
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        write(&dir.join("model-00001-of-00002.safetensors"), b"a");
        // model-00002 missing.
        assert_eq!(
            classify_snapshot(&dir),
            SnapshotState::Incomplete {
                missing: vec!["model-00002-of-00002.safetensors".to_string()],
            }
        );
    }

    #[test]
    fn incomplete_sharded_zero_byte_shard() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("snap");
        write(&dir.join("config.json"), b"{}");
        write_index(
            &dir,
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        write(&dir.join("model-00001-of-00002.safetensors"), b"a");
        // Present but zero bytes → still counts as missing.
        write(&dir.join("model-00002-of-00002.safetensors"), b"");
        assert_eq!(
            classify_snapshot(&dir),
            SnapshotState::Incomplete {
                missing: vec!["model-00002-of-00002.safetensors".to_string()],
            }
        );
    }

    #[test]
    fn complete_stale_index_repackaged_quant() {
        // A repackaged mlx-community quant: the shipped index still names the
        // original full-precision shards (absent), but the directory holds a
        // differently-named quant file. Must NOT be flagged incomplete — the
        // loader globs it and loads fine.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("snap");
        write(&dir.join("config.json"), b"{}");
        write_index(
            &dir,
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        // On-disk file is NOT named by the (stale) index.
        write(&dir.join("model.safetensors"), b"quant");
        assert_eq!(classify_snapshot(&dir), SnapshotState::Complete);
    }

    #[test]
    fn incomplete_malformed_index_no_weights() {
        // A truncated/zero-byte index with nothing else loadable → re-fetch.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("snap");
        write(&dir.join("config.json"), b"{}");
        write(&dir.join(SHARD_INDEX_FILE), b""); // unparseable
        assert_eq!(
            classify_snapshot(&dir),
            SnapshotState::Incomplete {
                missing: vec![SHARD_INDEX_FILE.to_string()],
            }
        );
    }

    #[test]
    fn complete_malformed_index_with_weights() {
        // A malformed index but with loadable weights present: defer to the
        // loader's glob rather than re-fetching a working model.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("snap");
        write(&dir.join("config.json"), b"{}");
        write(&dir.join(SHARD_INDEX_FILE), b"{ this is not json");
        write(&dir.join("model.safetensors"), b"weights");
        assert_eq!(classify_snapshot(&dir), SnapshotState::Complete);
    }

    #[test]
    fn traversal_shard_name_counts_as_missing() {
        // A malicious index naming a shard outside the directory must never be
        // counted as present, even if such a file exists elsewhere.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("snap");
        write(&dir.join("config.json"), b"{}");
        // The out-of-dir file the malicious index tries to point at.
        write(&tmp.path().join("outside.safetensors"), b"secret");
        write_index(&dir, &["../outside.safetensors"]);
        assert_eq!(
            classify_snapshot(&dir),
            SnapshotState::Incomplete {
                missing: vec!["../outside.safetensors".to_string()],
            }
        );
    }
}
