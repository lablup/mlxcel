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

//! WebUI model catalog projection.
//!
//! This adapter is intentionally metadata-only: it reads bounded config and
//! index files from already-discovered router entries, never tokenizers,
//! providers, SafeTensors payloads, or remote repositories.

#[path = "catalog_detection.rs"]
mod catalog_detection;
#[path = "catalog_fs.rs"]
mod catalog_fs;
#[path = "catalog_metadata.rs"]
mod catalog_metadata;
#[path = "catalog_types.rs"]
mod catalog_types;

#[allow(unused_imports)]
pub use catalog_metadata::{single_model_entry, single_model_entry_from_state};
pub use catalog_types::*;

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use crate::server::router_lifecycle::{LifecycleSnapshot, ModelLifecycleState, SCHEMA_VERSION};
use crate::server::router_models::RouterCatalogModel;

use catalog_metadata::{apply_runtime_fields, catalog_entry};

static CATALOG_CACHE: OnceLock<Mutex<BTreeMap<String, (String, CatalogEntry)>>> = OnceLock::new();

pub fn clear_catalog_cache() {
    if let Some(cache) = CATALOG_CACHE.get()
        && let Ok(mut cache) = cache.lock()
    {
        cache.clear();
    }
}

pub fn catalog_change_signatures(models: Vec<RouterCatalogModel>) -> BTreeMap<String, String> {
    models
        .into_iter()
        .take(MAX_INVENTORY)
        .filter(|model| !model.hidden)
        .map(|model| {
            let fingerprint = catalog_metadata::catalog_cache_fingerprint(&model.path)
                .unwrap_or_else(|| "missing-config".to_string());
            let signature = format!(
                "{}:{}:{}:{}:{}",
                model.source.as_str(),
                model.path.display(),
                model.name,
                model.generation,
                fingerprint
            );
            (model.ui_model_id, signature)
        })
        .collect()
}

pub fn count_changed_entries(
    before: &BTreeMap<String, String>,
    after: &BTreeMap<String, String>,
) -> u64 {
    let mut changed = 0u64;
    for (id, after_sig) in after {
        if before.get(id) != Some(after_sig) {
            changed += 1;
        }
    }
    for id in before.keys() {
        if !after.contains_key(id) {
            changed += 1;
        }
    }
    changed
}

fn cached_catalog_entry(model: RouterCatalogModel) -> CatalogEntry {
    let id = model.ui_model_id.clone();
    let fingerprint = catalog_metadata::catalog_cache_fingerprint(&model.path)
        .unwrap_or_else(|| "missing-config".to_string());
    let provider_key = model
        .provider_capabilities
        .map(|cap| format!("provider:{}:{}", cap.image_input, cap.audio_input))
        .unwrap_or_else(|| "provider:none".to_string());
    let key = format!(
        "{}:{}:{}:{}",
        model.source.as_str(),
        model.path.display(),
        fingerprint,
        provider_key
    );
    let cache = CATALOG_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(cache) = cache.lock()
        && let Some((cached_key, entry)) = cache.get(&id)
        && cached_key == &key
    {
        let mut entry = entry.clone();
        apply_runtime_fields(&mut entry, &model);
        return entry;
    }
    let entry = catalog_entry(model);
    if let Ok(mut cache) = cache.lock() {
        if cache.len() >= MAX_INVENTORY {
            cache.clear();
        }
        cache.insert(id, (key, entry.clone()));
    }
    entry
}
use catalog_types::{DEFAULT_LIMIT, MAX_INVENTORY, MAX_LIMIT};

pub fn list_catalog(
    models: Vec<RouterCatalogModel>,
    query: &CatalogQuery,
    server_instance_id: String,
    snapshot_sequence: u64,
) -> Result<CatalogListResponse, CatalogError> {
    let limit = query.limit.unwrap_or(DEFAULT_LIMIT);
    if !(1..=MAX_LIMIT).contains(&limit) {
        return Err(CatalogError::InvalidField {
            field: "limit",
            message: "limit must be between 1 and 200".to_string(),
        });
    }
    validate_search(query.q.as_deref())?;
    let offset = parse_cursor(query.cursor.as_deref())?;
    let filters = Filters::parse(query)?;
    let mut entries: Vec<_> = models
        .into_iter()
        .take(MAX_INVENTORY)
        .filter(|model| !model.hidden)
        .map(cached_catalog_entry)
        .filter(|entry| filters.matches(entry))
        .collect();
    entries.sort_by(|a, b| a.identity.id.cmp(&b.identity.id));
    let total = entries.len();
    let page: Vec<_> = entries.into_iter().skip(offset).take(limit).collect();
    let next = (offset + page.len() < total).then(|| format!("o_{}", offset + page.len()));
    Ok(CatalogListResponse {
        schema_version: SCHEMA_VERSION.to_string(),
        items: page,
        pagination: Pagination {
            limit,
            next_cursor: next,
            total_known: Some(total),
        },
        server_instance_id,
        snapshot_sequence,
    })
}

pub fn get_catalog_entry(
    models: Vec<RouterCatalogModel>,
    model_id: &str,
) -> Result<CatalogEntry, CatalogError> {
    models
        .into_iter()
        .take(MAX_INVENTORY)
        .filter(|model| !model.hidden)
        .map(cached_catalog_entry)
        .find(|entry| entry.identity.id == model_id)
        .ok_or(CatalogError::NotFound)
}

fn validate_search(q: Option<&str>) -> Result<(), CatalogError> {
    if q.is_some_and(|q| q.len() > 128) {
        return Err(CatalogError::InvalidField {
            field: "q",
            message: "q must be at most 128 bytes".to_string(),
        });
    }
    Ok(())
}

fn parse_cursor(cursor: Option<&str>) -> Result<usize, CatalogError> {
    let Some(cursor) = cursor else { return Ok(0) };
    if cursor.len() > 512 {
        return Err(CatalogError::InvalidField {
            field: "cursor",
            message: "cursor is too long".to_string(),
        });
    }
    let Some(raw) = cursor.strip_prefix("o_") else {
        return Err(CatalogError::InvalidField {
            field: "cursor",
            message: "cursor must be returned by a previous catalog response".to_string(),
        });
    };
    raw.parse::<usize>()
        .map_err(|_| CatalogError::InvalidField {
            field: "cursor",
            message: "cursor must be returned by a previous catalog response".to_string(),
        })
}

struct Filters {
    q: Option<String>,
    source: Option<CatalogSourceKind>,
    task: Option<TaskKind>,
    lifecycle: Option<String>,
    support: Option<bool>,
    completeness: Option<bool>,
}

impl Filters {
    fn parse(query: &CatalogQuery) -> Result<Self, CatalogError> {
        Ok(Self {
            q: query.q.as_ref().map(|q| q.to_ascii_lowercase()),
            source: parse_source(query.source.as_deref())?,
            task: parse_task(query.task.as_deref())?,
            lifecycle: parse_lifecycle(query.lifecycle.as_deref())?,
            support: parse_support_filter(query.support.as_deref())?,
            completeness: parse_completeness_filter(query.completeness.as_deref())?,
        })
    }

    fn matches(&self, entry: &CatalogEntry) -> bool {
        if let Some(q) = &self.q {
            let haystack = format!(
                "{} {} {} {}",
                entry.identity.display_name,
                entry.identity.inference_id,
                entry.metadata.model_type.as_deref().unwrap_or_default(),
                entry.metadata.architecture.as_deref().unwrap_or_default()
            )
            .to_ascii_lowercase();
            if !haystack.contains(q) {
                return false;
            }
        }
        if self
            .source
            .is_some_and(|source| source != entry.identity.source)
        {
            return false;
        }
        if self.task.is_some_and(|task| {
            !entry.metadata.input_tasks.contains(&task)
                && !entry.metadata.output_tasks.contains(&task)
        }) {
            return false;
        }
        if self
            .lifecycle
            .as_ref()
            .is_some_and(|state| *state != lifecycle_name(&entry.lifecycle))
        {
            return false;
        }
        if self.support.is_some_and(|wanted| wanted != entry.supported) {
            return false;
        }
        if self
            .completeness
            .is_some_and(|wanted| wanted != entry.complete)
        {
            return false;
        }
        true
    }
}

fn parse_lifecycle(value: Option<&str>) -> Result<Option<String>, CatalogError> {
    match value {
        None => Ok(None),
        Some("unloaded" | "loading" | "ready" | "draining" | "unloading" | "failed") => {
            Ok(value.map(ToString::to_string))
        }
        Some(_) => Err(CatalogError::InvalidField {
            field: "lifecycle",
            message: "lifecycle must be unloaded, loading, ready, draining, unloading or failed"
                .to_string(),
        }),
    }
}

fn parse_support_filter(value: Option<&str>) -> Result<Option<bool>, CatalogError> {
    match value {
        None => Ok(None),
        Some("true" | "supported") => Ok(Some(true)),
        Some("false" | "unsupported") => Ok(Some(false)),
        Some(_) => Err(CatalogError::InvalidField {
            field: "support",
            message: "support must be true, false, supported or unsupported".to_string(),
        }),
    }
}

fn parse_completeness_filter(value: Option<&str>) -> Result<Option<bool>, CatalogError> {
    match value {
        None => Ok(None),
        Some("true" | "complete") => Ok(Some(true)),
        Some("false" | "incomplete") => Ok(Some(false)),
        Some(_) => Err(CatalogError::InvalidField {
            field: "completeness",
            message: "completeness must be true, false, complete or incomplete".to_string(),
        }),
    }
}

fn parse_source(source: Option<&str>) -> Result<Option<CatalogSourceKind>, CatalogError> {
    match source {
        None => Ok(None),
        Some("cache") => Ok(Some(CatalogSourceKind::Cache)),
        Some("models_dir") => Ok(Some(CatalogSourceKind::ModelsDir)),
        Some("preset") => Ok(Some(CatalogSourceKind::Preset)),
        Some("single_model") => Ok(Some(CatalogSourceKind::SingleModel)),
        Some(_) => Err(CatalogError::InvalidField {
            field: "source",
            message: "source must be cache, models_dir, preset or single_model".to_string(),
        }),
    }
}

fn parse_task(task: Option<&str>) -> Result<Option<TaskKind>, CatalogError> {
    match task {
        None => Ok(None),
        Some("chat") => Ok(Some(TaskKind::Chat)),
        Some("completion") => Ok(Some(TaskKind::Completion)),
        Some("embedding") => Ok(Some(TaskKind::Embedding)),
        Some("rerank") => Ok(Some(TaskKind::Rerank)),
        Some("audio_transcription") => Ok(Some(TaskKind::AudioTranscription)),
        Some("audio_speech") => Ok(Some(TaskKind::AudioSpeech)),
        Some("vision_input") => Ok(Some(TaskKind::VisionInput)),
        Some("image_generation") => Ok(Some(TaskKind::ImageGeneration)),
        Some(_) => Err(CatalogError::InvalidField {
            field: "task",
            message: "task is not a recognized WebUI task".to_string(),
        }),
    }
}

fn lifecycle_name(lifecycle: &LifecycleSnapshot) -> &'static str {
    match lifecycle.state {
        ModelLifecycleState::Unloaded => "unloaded",
        ModelLifecycleState::Loading => "loading",
        ModelLifecycleState::Ready => "ready",
        ModelLifecycleState::Draining => "draining",
        ModelLifecycleState::Unloading => "unloading",
        ModelLifecycleState::Failed => "failed",
    }
}

#[cfg(test)]
#[path = "catalog_tests.rs"]
mod tests;
