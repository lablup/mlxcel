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

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::catalog_types::{MAX_CONFIG_BYTES, MAX_DISK_DEPTH, MAX_DISK_FILES, MAX_INDEX_BYTES};

pub(super) struct Completeness {
    pub(super) ok: bool,
    pub(super) reason: String,
}

pub(super) fn completeness(path: &Path) -> Completeness {
    if !regular_file_exists(&path.join("config.json")) {
        return Completeness {
            ok: false,
            reason: "config.json is missing".to_string(),
        };
    }
    let index = path.join("model.safetensors.index.json");
    if regular_file_exists(&index) {
        return indexed_completeness(path, &index);
    }
    let entries = match bounded_read_dir_paths(path) {
        Ok(entries) => entries,
        Err(reason) => {
            return Completeness { ok: false, reason };
        }
    };
    let has_shard = entries.iter().any(|entry| {
        entry.extension().is_some_and(|ext| ext == "safetensors") && regular_nonzero_file(entry)
    });
    if has_shard {
        Completeness {
            ok: true,
            reason: String::new(),
        }
    } else {
        Completeness {
            ok: false,
            reason: "no non-empty SafeTensors shard was found".to_string(),
        }
    }
}

fn indexed_completeness(path: &Path, index: &Path) -> Completeness {
    let (json, err) = read_json_bounded(index, MAX_INDEX_BYTES);
    let Some(value) = json else {
        return Completeness {
            ok: false,
            reason: err.unwrap_or_else(|| "safetensors index is unreadable".to_string()),
        };
    };
    let Some(map) = value.get("weight_map").and_then(Value::as_object) else {
        return Completeness {
            ok: false,
            reason: "safetensors index has no weight_map".to_string(),
        };
    };
    if map.is_empty() {
        return Completeness {
            ok: false,
            reason: "safetensors index has an empty weight_map".to_string(),
        };
    }
    let mut files = BTreeMap::<PathBuf, ()>::new();
    for file in map.values().filter_map(Value::as_str) {
        let relative = Path::new(file);
        if file.contains("..") || relative.is_absolute() || relative.components().count() != 1 {
            return Completeness {
                ok: false,
                reason: "safetensors index contains an unsafe shard path".to_string(),
            };
        }
        files.insert(PathBuf::from(file), ());
        if files.len() > MAX_DISK_FILES {
            return Completeness {
                ok: false,
                reason: "safetensors index exceeds the catalog shard limit".to_string(),
            };
        }
    }
    for file in files.keys() {
        let shard = path.join(file);
        if !regular_nonzero_file(&shard) {
            return Completeness {
                ok: false,
                reason: format!("required shard '{}' is missing or empty", file.display()),
            };
        }
    }
    Completeness {
        ok: true,
        reason: String::new(),
    }
}

pub(super) fn read_json_bounded(path: &Path, max_bytes: u64) -> (Option<Value>, Option<String>) {
    let meta = match fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(err) => {
            return (
                None,
                Some(format!("{} is unreadable: {err}", safe_path_label(path))),
            );
        }
    };
    if !meta.file_type().is_file() {
        return (
            None,
            Some(format!("{} is not a regular file", safe_path_label(path))),
        );
    }
    if meta.len() > max_bytes {
        return (
            None,
            Some(format!(
                "{} exceeds the bounded metadata limit",
                safe_path_label(path)
            )),
        );
    }
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(err) => {
            return (
                None,
                Some(format!("{} is unreadable: {err}", safe_path_label(path))),
            );
        }
    };
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    if let Err(err) = file.by_ref().take(max_bytes + 1).read_to_end(&mut bytes) {
        return (
            None,
            Some(format!(
                "{} could not be read: {err}",
                safe_path_label(path)
            )),
        );
    }
    match serde_json::from_slice(&bytes) {
        Ok(value) => (Some(value), None),
        Err(err) => (
            None,
            Some(format!(
                "{} is malformed JSON: {err}",
                safe_path_label(path)
            )),
        ),
    }
}

pub(super) fn format_for(path: &Path) -> (Option<String>, Option<String>) {
    if regular_file_exists(&path.join("model.safetensors.index.json")) {
        return (Some("safetensors".to_string()), None);
    }
    let entries = match bounded_read_dir_paths(path) {
        Ok(entries) => entries,
        Err(reason) => return (None, Some(reason)),
    };
    (
        entries
            .iter()
            .any(|entry| {
                entry.extension().is_some_and(|ext| ext == "safetensors")
                    && regular_file_exists(entry)
            })
            .then(|| "safetensors".to_string()),
        None,
    )
}

pub(super) fn disk_size(path: &Path) -> (Option<u64>, Option<String>) {
    let mut stack = vec![(path.to_path_buf(), 0usize)];
    let mut total = 0u64;
    let mut visited = 0usize;
    while let Some((current, depth)) = stack.pop() {
        visited += 1;
        if depth > MAX_DISK_DEPTH || visited > MAX_DISK_FILES {
            return (
                None,
                Some("disk size traversal exceeded the catalog bound".to_string()),
            );
        }
        let meta = match fs::symlink_metadata(&current) {
            Ok(meta) => meta,
            Err(err) => {
                return (
                    None,
                    Some(format!(
                        "{} is unreadable: {err}",
                        safe_path_label(&current)
                    )),
                );
            }
        };
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_file() {
            total = total.saturating_add(meta.len());
        } else if meta.is_dir() {
            let entries = match fs::read_dir(&current) {
                Ok(entries) => entries,
                Err(err) => {
                    return (
                        None,
                        Some(format!(
                            "{} is unreadable: {err}",
                            safe_path_label(&current)
                        )),
                    );
                }
            };
            for entry in entries.filter_map(Result::ok) {
                if stack.len() + visited >= MAX_DISK_FILES {
                    return (
                        None,
                        Some("disk size traversal exceeded the catalog bound".to_string()),
                    );
                }
                stack.push((entry.path(), depth + 1));
            }
        }
    }
    (Some(total), None)
}

fn regular_file_exists(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_file())
        .unwrap_or(false)
}

fn regular_nonzero_file(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_file() && meta.len() > 0)
        .unwrap_or(false)
}

pub(super) fn content_fingerprint(path: &Path) -> Option<String> {
    let mut facts = Vec::new();
    push_required_metadata_fact(
        &mut facts,
        "config",
        &path.join("config.json"),
        MAX_CONFIG_BYTES,
    )?;
    let index = path.join("model.safetensors.index.json");
    if regular_file_exists(&index) {
        push_required_metadata_fact(&mut facts, "index", &index, MAX_INDEX_BYTES)?;
        push_index_shard_facts(&mut facts, path, &index);
    } else {
        push_direct_shard_facts(&mut facts, path);
    }
    let digest = Sha256::digest(facts.join("\n").as_bytes());
    Some(format!("sha256:{}", hex_digest(&digest)))
}

fn push_required_metadata_fact(
    facts: &mut Vec<String>,
    label: &str,
    path: &Path,
    max_bytes: u64,
) -> Option<()> {
    push_bounded_file_state(facts, label, path, max_bytes).then_some(())
}

fn push_direct_shard_facts(facts: &mut Vec<String>, path: &Path) {
    let entries = match bounded_read_dir_paths(path) {
        Ok(entries) => entries,
        Err(reason) => {
            facts.push(format!("direct:bounded:{reason}"));
            return;
        }
    };
    let mut shards: Vec<PathBuf> = entries
        .into_iter()
        .filter(|entry| {
            entry.extension().is_some_and(|ext| ext == "safetensors") && regular_file_exists(entry)
        })
        .collect();
    shards.sort();
    for (idx, shard) in shards.iter().enumerate() {
        push_path_state(facts, &format!("direct:{idx}"), shard);
    }
}

fn push_index_shard_facts(facts: &mut Vec<String>, model_path: &Path, index: &Path) {
    let (json, err) = read_json_bounded(index, MAX_INDEX_BYTES);
    let Some(value) = json else {
        facts.push(format!(
            "index-shards:unreadable:{}",
            err.unwrap_or_else(|| "unknown".to_string())
        ));
        return;
    };
    let Some(map) = value.get("weight_map").and_then(Value::as_object) else {
        facts.push("index-shards:no-weight-map".to_string());
        return;
    };
    let mut shards = BTreeMap::<String, ()>::new();
    for file in map.values().filter_map(Value::as_str) {
        let relative = Path::new(file);
        if file.contains("..") || relative.is_absolute() || relative.components().count() != 1 {
            facts.push("index-shards:unsafe-path".to_string());
            return;
        }
        shards.insert(file.to_string(), ());
        if shards.len() > MAX_DISK_FILES {
            facts.push("index-shards:overflow".to_string());
            break;
        }
    }
    for (idx, file) in shards.keys().take(MAX_DISK_FILES).enumerate() {
        push_path_state(facts, &format!("indexed:{idx}"), &model_path.join(file));
    }
}

fn push_path_state(facts: &mut Vec<String>, label: &str, path: &Path) -> bool {
    let Ok(meta) = fs::symlink_metadata(path) else {
        facts.push(format!("{label}:missing"));
        return false;
    };
    let file_type = meta.file_type();
    let kind = if file_type.is_file() {
        "file"
    } else if file_type.is_dir() {
        "dir"
    } else if file_type.is_symlink() {
        "symlink"
    } else {
        "other"
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    facts.push(format!("{label}:{kind}:{}:{mtime}", meta.len()));
    true
}

fn push_bounded_file_state(
    facts: &mut Vec<String>,
    label: &str,
    path: &Path,
    max_bytes: u64,
) -> bool {
    let Ok(meta) = fs::symlink_metadata(path) else {
        facts.push(format!("{label}:missing"));
        return false;
    };
    if !meta.file_type().is_file() {
        push_path_state(facts, label, path);
        return false;
    }
    if meta.len() > max_bytes {
        facts.push(format!("{label}:file:{}:too-large", meta.len()));
        return false;
    }
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(_) => {
            facts.push(format!("{label}:file:{}:unreadable", meta.len()));
            return false;
        }
    };
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    if file
        .by_ref()
        .take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .is_err()
    {
        facts.push(format!("{label}:file:{}:read-error", meta.len()));
        return false;
    }
    let digest = Sha256::digest(&bytes);
    let mtime = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    facts.push(format!(
        "{label}:file:{}:{mtime}:sha256:{}",
        meta.len(),
        hex_digest(&digest)
    ));
    true
}

fn bounded_read_dir_paths(path: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = fs::read_dir(path)
        .map_err(|err| format!("{} is unreadable: {err}", safe_path_label(path)))?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|err| format!("{} is unreadable: {err}", safe_path_label(path)))?;
        if paths.len() >= MAX_DISK_FILES {
            return Err("directory entry count exceeded the catalog bound".to_string());
        }
        paths.push(entry.path());
    }
    Ok(paths)
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn safe_path_label(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("catalog metadata file")
        .to_string()
}
