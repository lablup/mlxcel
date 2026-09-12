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
use crate::server::router_models::{
    RouterCatalogModel, RouterCatalogProviderCapabilities, RouterModelSource,
};

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
        catalog_epoch: 1,
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
    assert_eq!(arcee.metadata.model_type.as_deref(), Some("arcee"));
    assert!(arcee.metadata.unknown_reasons.model_type.is_none());
    assert!(arcee.metadata.architecture.is_none());
    assert!(arcee.metadata.unknown_reasons.architecture.is_some());
    assert!(!arcee.metadata.support.architecturally_supported);
    assert!(!arcee.supported);
}

#[test]
fn architecture_uses_detection_authority_for_embedding_layouts() {
    let root = temp_dir("embedding-detection");
    let path = write_model(&root, "embed", "qwen3");
    std::fs::create_dir_all(path.join("1_Pooling")).expect("pooling dir");
    std::fs::write(
        path.join("1_Pooling/config.json"),
        r#"{"pooling_mode_lasttoken":true}"#,
    )
    .expect("pooling config");

    let entry = catalog_entry(model("embed", path, RouterModelSource::ModelsDir));
    assert_eq!(entry.metadata.model_type.as_deref(), Some("qwen3"));
    assert_eq!(
        entry.metadata.architecture.as_deref(),
        Some("qwen3_embedding")
    );
    assert!(entry.metadata.output_tasks.contains(&TaskKind::Embedding));
    assert!(!entry.metadata.output_tasks.contains(&TaskKind::Chat));
}

#[test]
fn catalog_cache_uses_epoch_and_projects_fresh_provider_lifecycle() {
    clear_catalog_cache();
    let root = temp_dir("epoch-cache");
    let path = write_model(&root, "vision", "qwen2_vl");
    let id = stable_model_identity("models_dir", 1, "test-root", "vision").0;
    let first = get_catalog_entry(
        vec![model("vision", path.clone(), RouterModelSource::ModelsDir)],
        &id,
    )
    .unwrap();
    assert!(first.complete);
    assert!(
        first
            .capabilities
            .iter()
            .any(|capability| capability.task == TaskKind::VisionInput && !capability.available)
    );

    std::fs::remove_file(path.join("model.safetensors")).expect("remove shard");
    reset_heavy_metadata_probe_count();
    let mut same_epoch = model_with_lifecycle(
        "vision",
        path.clone(),
        RouterModelSource::ModelsDir,
        lifecycle_with(ModelLifecycleState::Ready, true, 3),
        9,
    );
    same_epoch.provider_capabilities = Some(RouterCatalogProviderCapabilities {
        image_input: true,
        audio_input: false,
    });
    let cached = get_catalog_entry(vec![same_epoch], &id).unwrap();
    assert!(cached.complete, "same epoch must reuse cached metadata");
    assert_eq!(cached.identity.revision, 9);
    assert_eq!(cached.lifecycle.active_requests, 3);
    assert_eq!(heavy_metadata_probe_count(), 0);
    assert!(
        cached
            .capabilities
            .iter()
            .any(|capability| capability.task == TaskKind::VisionInput && capability.available)
    );

    let mut next_epoch = model("vision", path, RouterModelSource::ModelsDir);
    next_epoch.catalog_epoch = 2;
    let refreshed = get_catalog_entry(vec![next_epoch], &id).unwrap();
    assert!(!refreshed.complete);
    assert!(
        refreshed
            .metadata
            .support
            .complete_reason
            .as_deref()
            .unwrap()
            .contains("no non-empty SafeTensors")
    );
}

#[test]
fn raw_model_type_and_declared_architectures_preserve_bounded_config_strings() {
    let root = temp_dir("raw-config-fields");
    let dir = root.join("mixed-case");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"QwEn3","architectures":["Qwen3ForCausalLM","Vendor.Custom-1"],"quantization_config":{"bits":4}}"#,
    )
    .unwrap();
    std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();

    let entry = catalog_entry(model("mixed-case", dir, RouterModelSource::ModelsDir));
    assert_eq!(entry.metadata.model_type.as_deref(), Some("QwEn3"));
    assert_eq!(entry.metadata.architecture.as_deref(), Some("qwen3"));
    assert_eq!(
        entry.metadata.declared_architectures.as_deref(),
        Some(
            &[
                "Qwen3ForCausalLM".to_string(),
                "Vendor.Custom-1".to_string()
            ][..]
        )
    );
    assert!(entry.metadata.unknown_reasons.model_type.is_none());
    assert!(
        entry
            .metadata
            .unknown_reasons
            .declared_architectures
            .is_none()
    );
}

#[test]
fn invalid_raw_config_strings_are_null_with_reasons() {
    let root = temp_dir("invalid-raw-config-fields");
    let too_long_model_type = "x".repeat(MAX_MODEL_TYPE_BYTES + 1);
    let too_many_architectures = (0..=MAX_DECLARED_ARCHITECTURES)
        .map(|idx| format!(r#""Arch{idx}""#))
        .collect::<Vec<_>>()
        .join(",");
    let too_long_architecture = "A".repeat(MAX_DECLARED_ARCHITECTURE_BYTES + 1);
    let cases = [
        (
            "non-string-model-type",
            r#"{"model_type":3,"architectures":["Arch"],"quantization_config":{"bits":4}}"#
                .to_string(),
            "model_type",
            "model_type must be a string",
        ),
        (
            "too-long-model-type",
            format!(
                r#"{{"model_type":"{too_long_model_type}","architectures":["Arch"],"quantization_config":{{"bits":4}}}}"#
            ),
            "model_type",
            "model_type exceeds 128 bytes",
        ),
        (
            "non-array-architectures",
            r#"{"model_type":"qwen3","architectures":"Arch","quantization_config":{"bits":4}}"#
                .to_string(),
            "architectures",
            "architectures must be an array of strings",
        ),
        (
            "too-many-architectures",
            format!(
                r#"{{"model_type":"qwen3","architectures":[{too_many_architectures}],"quantization_config":{{"bits":4}}}}"#
            ),
            "architectures",
            "architectures has more than 16 entries",
        ),
        (
            "too-long-architecture",
            format!(
                r#"{{"model_type":"qwen3","architectures":["{too_long_architecture}"],"quantization_config":{{"bits":4}}}}"#
            ),
            "architectures",
            "architectures entry exceeds 128 bytes",
        ),
    ];
    for (name, config, field, reason) in cases {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        let entry = catalog_entry(model(name, dir, RouterModelSource::ModelsDir));
        match field {
            "model_type" => {
                assert!(entry.metadata.model_type.is_none(), "{name}");
                assert_eq!(
                    entry.metadata.unknown_reasons.model_type.as_deref(),
                    Some(reason),
                    "{name}"
                );
            }
            "architectures" => {
                assert!(entry.metadata.declared_architectures.is_none(), "{name}");
                assert_eq!(
                    entry
                        .metadata
                        .unknown_reasons
                        .declared_architectures
                        .as_deref(),
                    Some(reason),
                    "{name}"
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn directory_entry_bounds_count_junk_before_filtering() {
    let root = temp_dir("entry-bound");
    let path = write_model(&root, "junk", "qwen3");
    for idx in 0..MAX_DISK_FILES {
        std::fs::write(path.join(format!("junk-{idx:04}.txt")), b"x").unwrap();
    }
    let entry = catalog_entry(model("junk", path, RouterModelSource::ModelsDir));
    assert!(!entry.complete);
    assert!(
        entry
            .metadata
            .support
            .complete_reason
            .as_deref()
            .unwrap()
            .contains("directory entry count exceeded")
    );
    assert!(
        entry
            .metadata
            .unknown_reasons
            .format
            .as_deref()
            .unwrap()
            .contains("directory entry count exceeded")
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
