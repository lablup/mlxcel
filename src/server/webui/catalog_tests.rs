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

use std::path::{Path, PathBuf};

use crate::server::router_lifecycle::{
    DownloadState, LifecycleSnapshot, ModelLifecycleState, stable_model_identity,
};
use crate::server::router_models::{RouterCatalogModel, RouterModelSource};

use super::*;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel-catalog-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn lifecycle() -> LifecycleSnapshot {
    LifecycleSnapshot {
        state: ModelLifecycleState::Unloaded,
        download: DownloadState::Complete,
        busy: false,
        active_requests: 0,
        draining_requests: 0,
        worker_exit_observed: true,
        last_error: None,
    }
}

fn lifecycle_with(
    state: ModelLifecycleState,
    busy: bool,
    active_requests: usize,
) -> LifecycleSnapshot {
    LifecycleSnapshot {
        state,
        download: DownloadState::Complete,
        busy,
        active_requests,
        draining_requests: 0,
        worker_exit_observed: true,
        last_error: None,
    }
}

fn write_model(root: &Path, name: &str, model_type: &str) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("model dir");
    std::fs::write(
        dir.join("config.json"),
        format!(r#"{{"model_type":"{model_type}","quantization_config":{{"bits":4}}}}"#),
    )
    .expect("config");
    std::fs::write(dir.join("model.safetensors"), b"weights").expect("weights");
    dir
}

fn model(name: &str, path: PathBuf, source: RouterModelSource) -> RouterCatalogModel {
    model_with_lifecycle(name, path, source, lifecycle(), 1)
}

fn model_with_lifecycle(
    name: &str,
    path: PathBuf,
    source: RouterModelSource,
    lifecycle: LifecycleSnapshot,
    revision: u64,
) -> RouterCatalogModel {
    let (ui_model_id, source_key_hash) =
        stable_model_identity(source.as_str(), 1, "test-root", name);
    RouterCatalogModel {
        name: name.to_string(),
        path,
        source,
        aliases: Vec::new(),
        tags: Vec::new(),
        ui_model_id,
        source_key_hash,
        hidden: false,
        lifecycle,
        revision,
        generation: 1,
        provider_capabilities: None,
    }
}

#[test]
fn empty_catalog_is_valid_and_paginated() {
    let response = list_catalog(Vec::new(), &CatalogQuery::default(), "srv_test".into(), 0)
        .expect("empty catalog");
    assert!(response.items.is_empty());
    assert_eq!(response.pagination.limit, 50);
    assert_eq!(response.pagination.total_known, Some(0));
}

#[test]
fn metadata_scan_reports_supported_complete_model_without_loading_provider() {
    let root = temp_dir("supported");
    let path = write_model(&root, "qwen", "qwen3");
    let entry = get_catalog_entry(
        vec![model("qwen", path, RouterModelSource::ModelsDir)],
        &stable_model_identity("models_dir", 1, "test-root", "qwen").0,
    )
    .expect("entry");
    assert_eq!(entry.metadata.model_type.as_deref(), Some("qwen3"));
    assert_eq!(entry.metadata.architecture.as_deref(), Some("qwen3"));
    assert!(entry.complete);
    assert!(entry.supported);
    assert!(entry.metadata.input_tasks.contains(&TaskKind::Chat));
    assert!(entry.metadata.output_tasks.contains(&TaskKind::Rerank));
    assert!(entry.metadata.memory_estimate_bytes.is_none());
    assert!(
        entry
            .metadata
            .unknown_reasons
            .memory_estimate_bytes
            .is_some()
    );
}

#[test]
fn incomplete_index_and_unsupported_dense_variant_stay_inspectable() {
    let root = temp_dir("incomplete-unsupported");
    let incomplete = root.join("broken");
    std::fs::create_dir_all(&incomplete).expect("dir");
    std::fs::write(incomplete.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
    std::fs::write(
        incomplete.join("model.safetensors.index.json"),
        r#"{"weight_map":{"model.embed_tokens.weight":"missing.safetensors"}}"#,
    )
    .unwrap();
    let unsupported = write_model(&root, "arcee", "arcee");
    let items = vec![
        model("broken", incomplete, RouterModelSource::ModelsDir),
        model("arcee", unsupported, RouterModelSource::ModelsDir),
    ];
    let page = list_catalog(items, &CatalogQuery::default(), "srv".into(), 0).unwrap();
    let broken = page
        .items
        .iter()
        .find(|entry| entry.identity.inference_id == "broken")
        .unwrap();
    assert!(!broken.complete);
    assert!(
        broken
            .metadata
            .support
            .complete_reason
            .as_deref()
            .unwrap()
            .contains("missing")
    );
    let arcee = page
        .items
        .iter()
        .find(|entry| entry.identity.inference_id == "arcee")
        .unwrap();
    assert!(arcee.metadata.model_type.is_none());
    assert!(arcee.metadata.unknown_reasons.model_type.is_some());
    assert!(!arcee.metadata.support.architecturally_supported);
    assert!(!arcee.supported);
}

#[test]
fn detection_authority_overrides_declared_model_type_for_embedding_layouts() {
    let root = temp_dir("embedding-detection");
    let path = write_model(&root, "embed", "qwen3");
    std::fs::create_dir_all(path.join("1_Pooling")).expect("pooling dir");
    std::fs::write(
        path.join("1_Pooling/config.json"),
        r#"{"pooling_mode_lasttoken":true}"#,
    )
    .expect("pooling config");

    let entry = catalog_entry(model("embed", path, RouterModelSource::ModelsDir));
    assert_eq!(
        entry.metadata.model_type.as_deref(),
        Some("qwen3_embedding")
    );
    assert_eq!(
        entry.metadata.architecture.as_deref(),
        Some("qwen3_embedding")
    );
    assert!(entry.metadata.output_tasks.contains(&TaskKind::Embedding));
    assert!(!entry.metadata.output_tasks.contains(&TaskKind::Chat));
}

#[test]
fn catalog_cache_invalidates_when_shard_metadata_changes() {
    clear_catalog_cache();
    let root = temp_dir("shard-cache");

    let direct = write_model(&root, "direct", "qwen3");
    let direct_id = stable_model_identity("models_dir", 1, "test-root", "direct").0;
    assert!(
        get_catalog_entry(
            vec![model(
                "direct",
                direct.clone(),
                RouterModelSource::ModelsDir
            )],
            &direct_id,
        )
        .unwrap()
        .complete
    );
    std::fs::remove_file(direct.join("model.safetensors")).expect("remove direct shard");
    let direct_after = get_catalog_entry(
        vec![model("direct", direct, RouterModelSource::ModelsDir)],
        &direct_id,
    )
    .unwrap();
    assert!(!direct_after.complete);
    assert!(
        direct_after
            .metadata
            .support
            .complete_reason
            .as_deref()
            .unwrap()
            .contains("no non-empty SafeTensors")
    );

    let indexed = root.join("indexed");
    std::fs::create_dir_all(&indexed).expect("indexed dir");
    std::fs::write(
        indexed.join("config.json"),
        r#"{"model_type":"qwen3","quantization_config":{"bits":4}}"#,
    )
    .unwrap();
    std::fs::write(
        indexed.join("model.safetensors.index.json"),
        r#"{"weight_map":{"model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}"#,
    )
    .unwrap();
    std::fs::write(indexed.join("model-00001-of-00001.safetensors"), b"weights").unwrap();
    let indexed_id = stable_model_identity("models_dir", 1, "test-root", "indexed").0;
    assert!(
        get_catalog_entry(
            vec![model(
                "indexed",
                indexed.clone(),
                RouterModelSource::ModelsDir,
            )],
            &indexed_id,
        )
        .unwrap()
        .complete
    );
    std::fs::write(indexed.join("model-00001-of-00001.safetensors"), b"").unwrap();
    let indexed_after = get_catalog_entry(
        vec![model("indexed", indexed, RouterModelSource::ModelsDir)],
        &indexed_id,
    )
    .unwrap();
    assert!(!indexed_after.complete);
    assert!(
        indexed_after
            .metadata
            .support
            .complete_reason
            .as_deref()
            .unwrap()
            .contains("missing or empty")
    );
}

#[test]
fn catalog_filters_validate_lifecycle_support_and_completeness() {
    let invalid_lifecycle = list_catalog(
        Vec::new(),
        &CatalogQuery {
            lifecycle: Some("started".to_string()),
            ..Default::default()
        },
        "srv".into(),
        0,
    )
    .unwrap_err();
    assert!(matches!(
        invalid_lifecycle,
        CatalogError::InvalidField {
            field: "lifecycle",
            ..
        }
    ));

    let invalid_support = list_catalog(
        Vec::new(),
        &CatalogQuery {
            support: Some("complete".to_string()),
            ..Default::default()
        },
        "srv".into(),
        0,
    )
    .unwrap_err();
    assert!(matches!(
        invalid_support,
        CatalogError::InvalidField {
            field: "support",
            ..
        }
    ));

    let invalid_completeness = list_catalog(
        Vec::new(),
        &CatalogQuery {
            completeness: Some("supported".to_string()),
            ..Default::default()
        },
        "srv".into(),
        0,
    )
    .unwrap_err();
    assert!(matches!(
        invalid_completeness,
        CatalogError::InvalidField {
            field: "completeness",
            ..
        }
    ));
}

#[test]
fn non_chat_task_and_removal_reasons_are_truthful() {
    let root = temp_dir("embedding-cache");
    let path = write_model(&root, "embed", "bert");
    let cache_entry = catalog_entry(model("owner/embed", path, RouterModelSource::Cache));
    assert!(
        cache_entry
            .metadata
            .output_tasks
            .contains(&TaskKind::Embedding)
    );
    assert!(!cache_entry.metadata.output_tasks.contains(&TaskKind::Chat));
    assert!(cache_entry.removal.eligible);

    let root = temp_dir("readonly");
    let path = write_model(&root, "flat", "qwen3");
    let readonly = catalog_entry(model("flat", path, RouterModelSource::ModelsDir));
    assert!(!readonly.removable);
    assert!(
        readonly
            .removal
            .reason
            .as_deref()
            .unwrap()
            .contains("managed cache")
    );
}

#[path = "catalog_contract_tests.rs"]
mod catalog_contract_tests;
