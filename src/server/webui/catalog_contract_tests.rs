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

use super::*;

#[test]
fn thousand_entry_inventory_paginates_without_duplicates() {
    let root = temp_dir("page1000");
    let path = write_model(&root, "shared", "qwen3");
    let mut models = Vec::new();
    for idx in 0..1000 {
        models.push(model(
            &format!("model-{idx:04}"),
            path.clone(),
            RouterModelSource::ModelsDir,
        ));
    }
    let cache = CatalogProjectionCache::new();
    let mut cursor = None;
    let mut seen = std::collections::BTreeSet::new();
    loop {
        let page = list_catalog_with_cache(
            &cache,
            models.clone(),
            &CatalogQuery {
                limit: Some(200),
                cursor: cursor.clone(),
                ..Default::default()
            },
            "srv".into(),
            1,
        )
        .unwrap();
        for item in &page.items {
            assert!(
                seen.insert(item.identity.id.clone()),
                "duplicate {}",
                item.identity.id
            );
        }
        cursor = page.pagination.next_cursor;
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(seen.len(), 1000);
    cache.clear();
    let after_refresh = list_catalog_with_cache(
        &cache,
        models,
        &CatalogQuery {
            limit: Some(200),
            ..Default::default()
        },
        "srv".into(),
        2,
    )
    .unwrap();
    for item in &after_refresh.items {
        assert!(
            seen.contains(&item.identity.id),
            "refresh changed stable id"
        );
    }
}

#[test]
fn symlink_loop_and_large_config_are_bounded_metadata() {
    let root = temp_dir("bounds");
    let looped = write_model(&root, "looped", "qwen3");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&looped, looped.join("self-loop")).expect("symlink");
    let entry = catalog_entry(model("looped", looped, RouterModelSource::ModelsDir));
    assert!(
        entry.metadata.disk_bytes.is_some(),
        "symlink loop must not be followed"
    );

    let large = root.join("large");
    std::fs::create_dir_all(&large).unwrap();
    std::fs::write(
        large.join("config.json"),
        vec![b' '; (MAX_CONFIG_BYTES + 1) as usize],
    )
    .unwrap();
    std::fs::write(large.join("model.safetensors"), b"weights").unwrap();
    let entry = catalog_entry(model("large", large, RouterModelSource::ModelsDir));
    assert!(entry.metadata.model_type.is_none());
    assert!(
        entry
            .metadata
            .unknown_reasons
            .model_type
            .unwrap()
            .contains("bounded metadata limit")
    );
}

#[test]
fn bounded_detection_rejects_dflash_without_reading_weights() {
    let root = temp_dir("dflash");
    let path = write_model(&root, "draft", "qwen3");
    std::fs::write(
        path.join("config.json"),
        r#"{"model_type":"qwen3","dflash_config":{}}"#,
    )
    .expect("config");

    let entry = catalog_entry(model("draft", path, RouterModelSource::ModelsDir));
    assert_eq!(entry.metadata.model_type.as_deref(), Some("qwen3"));
    assert!(!entry.metadata.support.architecturally_supported);
    let reason = entry
        .metadata
        .unknown_reasons
        .architecture
        .as_deref()
        .unwrap();
    assert!(reason.contains("DFlash"));
    assert!(!reason.contains(root.to_str().unwrap()));
}

#[test]
fn shared_detection_errors_do_not_leak_absolute_catalog_paths() {
    let root = temp_dir("sanitized-detection-error");
    let path = write_model(&root, "unsupported-embedding", "arcee");
    std::fs::write(
        path.join("config.json"),
        r#"{"model_type":"arcee","architectures":["BertModel"]}"#,
    )
    .expect("config");

    let entry = catalog_entry(model(
        "unsupported-embedding",
        path,
        RouterModelSource::ModelsDir,
    ));
    assert_eq!(entry.metadata.model_type.as_deref(), Some("arcee"));
    let reason = entry
        .metadata
        .unknown_reasons
        .architecture
        .as_deref()
        .unwrap();
    assert!(reason.contains("model directory is an embedding checkpoint"));
    assert!(!reason.contains(root.to_str().unwrap()));
}

#[test]
fn empty_weight_map_is_incomplete_and_reasons_do_not_leak_absolute_paths() {
    let root = temp_dir("empty-index");
    let path = root.join("empty-index-model");
    std::fs::create_dir_all(&path).expect("model dir");
    std::fs::write(path.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
    std::fs::write(
        path.join("model.safetensors.index.json"),
        r#"{"weight_map":{}}"#,
    )
    .unwrap();

    let entry = catalog_entry(model("empty", path.clone(), RouterModelSource::ModelsDir));
    assert!(!entry.complete);
    assert!(
        entry
            .metadata
            .support
            .complete_reason
            .as_deref()
            .unwrap()
            .contains("empty weight_map")
    );
    let reason_blob = serde_json::to_string(&entry.metadata.unknown_reasons).unwrap();
    assert!(!reason_blob.contains(root.to_str().unwrap()));
}

#[test]
fn single_model_catalog_entry_is_read_only_and_uses_actual_inference_id() {
    let root = temp_dir("single");
    let path = write_model(&root, "single", "qwen3");
    let entry = single_model_entry(path, "actual-model".to_string(), lifecycle());
    assert_eq!(entry.identity.inference_id, "actual-model");
    assert_eq!(entry.identity.source, CatalogSourceKind::SingleModel);
    assert!(!entry.removable);
    assert!(entry.removal.reason.unwrap().contains("single-model"));
}

#[test]
fn catalog_cache_projects_fresh_lifecycle_without_metadata_rescan() {
    let root = temp_dir("fresh-lifecycle");
    let path = write_model(&root, "alpha", "qwen3");
    let id = stable_model_identity("models_dir", 1, "test-root", "alpha").0;
    let cache = CatalogProjectionCache::new();
    let first = get_catalog_entry_with_cache(
        &cache,
        vec![model_with_lifecycle(
            "alpha",
            path.clone(),
            RouterModelSource::ModelsDir,
            lifecycle(),
            1,
        )],
        &id,
    )
    .unwrap();
    assert_eq!(first.lifecycle.active_requests, 0);

    let second = get_catalog_entry_with_cache(
        &cache,
        vec![model_with_lifecycle(
            "alpha",
            path,
            RouterModelSource::ModelsDir,
            lifecycle_with(ModelLifecycleState::Ready, true, 3),
            9,
        )],
        &id,
    )
    .unwrap();
    assert_eq!(second.identity.revision, 9);
    assert_eq!(second.lifecycle.active_requests, 3);
    assert!(second.lifecycle.busy);
}

#[test]
fn actual_catalog_producer_matches_whole_expected_shape() {
    let root = temp_dir("whole-producer");
    let path = write_model(&root, "alpha", "qwen3");
    let model = model("alpha", path, RouterModelSource::ModelsDir);
    let response = list_catalog(
        vec![model],
        &CatalogQuery::default(),
        "srv_fixture".into(),
        7,
    )
    .unwrap();
    let mut actual = serde_json::to_value(&response).unwrap();
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/webui/examples/catalog.page.json"
    ))
    .expect("catalog fixture");

    // Keep this as a whole-producer fixture comparison instead of an inline
    // expected shape while masking values that are intentionally generated from
    // the temporary test directory.
    actual["items"][0]["identity"]["id"] = expected["items"][0]["identity"]["id"].clone();
    actual["items"][0]["identity"]["source_key_hash"] =
        expected["items"][0]["identity"]["source_key_hash"].clone();
    actual["items"][0]["identity"]["content_fingerprint"] =
        expected["items"][0]["identity"]["content_fingerprint"].clone();
    actual["items"][0]["metadata"]["disk_bytes"] =
        expected["items"][0]["metadata"]["disk_bytes"].clone();

    assert_eq!(actual, expected);
}

#[test]
fn catalog_fixture_round_trips_through_producer_dtos() {
    let mut value: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/webui/examples/catalog.page.json"
    ))
    .expect("catalog fixture");
    value.as_object_mut().unwrap().remove("$schemaName");
    let response: CatalogListResponse = serde_json::from_value(value).expect("catalog dto");
    assert_eq!(response.schema_version, SCHEMA_VERSION);
    assert_eq!(
        response.items[0].metadata.model_type.as_deref(),
        Some("qwen3")
    );
    assert!(
        response.items[0]
            .metadata
            .support
            .tested_checkpoint_reason
            .is_some()
    );
}

#[test]
fn cached_single_model_catalog_accessor_projects_fresh_app_state_provider() {
    use std::sync::{Arc, mpsc};

    use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig};
    use crate::tokenizer::MlxcelTokenizer;

    let root = temp_dir("single-state");
    let path = write_model(&root, "single", "qwen2_vl");
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let mut state = AppState::new(
        provider,
        ServerConfig {
            model_alias: Some("served-alias".to_string()),
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        path.clone(),
        batch_metrics,
    );
    state.media_support.image = true;
    let cache = CatalogProjectionCache::new();

    let entry = single_model_entry_from_state_with_cache(&cache, &state);
    assert_eq!(entry.identity.inference_id, "served-alias");
    assert_eq!(entry.identity.source, CatalogSourceKind::SingleModel);
    assert_eq!(entry.lifecycle.state, ModelLifecycleState::Ready);
    assert!(
        entry
            .capabilities
            .iter()
            .any(|capability| { capability.task == TaskKind::VisionInput && capability.available })
    );
    assert!(!entry.removable);

    std::fs::remove_file(path.join("model.safetensors")).expect("remove shard");
    cache.reset_heavy_metadata_probe_count();
    let repeated = single_model_entry_from_state_with_cache(&cache, &state);
    assert!(repeated.complete, "cache hit should reuse static metadata");
    assert_eq!(cache.heavy_metadata_probe_count(), 0);

    let unavailable_provider = Arc::new(ModelProvider::chat_unavailable_for_route_tests());
    let unavailable_metrics = unavailable_provider.batch_metrics().clone();
    let unavailable = AppState::new(
        unavailable_provider,
        ServerConfig {
            model_alias: Some("served-alias".to_string()),
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        path,
        unavailable_metrics,
    );
    let unavailable_entry = single_model_entry_from_state_with_cache(&cache, &unavailable);
    assert_eq!(
        unavailable_entry.lifecycle.state,
        ModelLifecycleState::Failed
    );
    assert!(
        unavailable_entry.capabilities.iter().any(|capability| {
            capability.task == TaskKind::VisionInput && !capability.available
        })
    );
    assert_eq!(cache.heavy_metadata_probe_count(), 0);
}

#[test]
fn bounded_catalog_detection_reports_unknown_when_weight_index_is_required() {
    let root = temp_dir("needs-index");
    let path = write_model(&root, "kimi", "kimi_k3");
    std::fs::write(
        path.join("config.json"),
        r#"{"model_type":"kimi_k3","vision_config":{}}"#,
    )
    .expect("config");
    std::fs::remove_file(path.join("model.safetensors")).expect("remove direct shard");

    let entry = catalog_entry(model("kimi", path, RouterModelSource::ModelsDir));
    assert_eq!(entry.metadata.model_type.as_deref(), Some("kimi_k3"));
    assert!(
        entry
            .metadata
            .unknown_reasons
            .architecture
            .as_deref()
            .unwrap()
            .contains("does not read SafeTensors headers")
    );
    assert!(!entry.supported);
}

#[test]
fn gemma4_without_bounded_variant_evidence_is_unknown_not_text() {
    let root = temp_dir("gemma4-needs-index");
    let path = write_model(&root, "gemma4", "gemma4");

    let entry = catalog_entry(model("gemma4", path, RouterModelSource::ModelsDir));
    assert_eq!(entry.metadata.model_type.as_deref(), Some("gemma4"));
    assert!(
        entry
            .metadata
            .unknown_reasons
            .architecture
            .as_deref()
            .unwrap()
            .contains("Gemma 4 text-vs-vision classification requires")
    );
    assert!(!entry.supported);
}

#[cfg(unix)]
#[test]
fn symlinked_config_index_and_shards_are_not_catalog_evidence() {
    let root = temp_dir("symlinked-catalog-evidence");

    let config_link = root.join("config-link");
    std::fs::create_dir_all(&config_link).unwrap();
    let outside_config = root.join("outside-config.json");
    std::fs::write(&outside_config, r#"{"model_type":"qwen3"}"#).unwrap();
    std::os::unix::fs::symlink(&outside_config, config_link.join("config.json")).unwrap();
    std::fs::write(config_link.join("model.safetensors"), b"weights").unwrap();
    let entry = catalog_entry(model(
        "config-link",
        config_link,
        RouterModelSource::ModelsDir,
    ));
    assert!(entry.metadata.model_type.is_none());
    assert!(!entry.complete);

    let index_link = write_model(&root, "index-link", "qwen3");
    std::fs::remove_file(index_link.join("model.safetensors")).unwrap();
    let outside_index = root.join("outside-index.json");
    std::fs::write(
        &outside_index,
        r#"{"weight_map":{"model.embed_tokens.weight":"real.safetensors"}}"#,
    )
    .unwrap();
    std::os::unix::fs::symlink(
        &outside_index,
        index_link.join("model.safetensors.index.json"),
    )
    .unwrap();
    let entry = catalog_entry(model(
        "index-link",
        index_link,
        RouterModelSource::ModelsDir,
    ));
    assert!(!entry.complete);
    assert!(entry.metadata.format.is_none());

    let shard_link = root.join("shard-link");
    std::fs::create_dir_all(&shard_link).unwrap();
    std::fs::write(shard_link.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
    let outside_shard = root.join("outside.safetensors");
    std::fs::write(&outside_shard, b"weights").unwrap();
    std::os::unix::fs::symlink(&outside_shard, shard_link.join("model.safetensors")).unwrap();
    let entry = catalog_entry(model(
        "shard-link",
        shard_link,
        RouterModelSource::ModelsDir,
    ));
    assert!(!entry.complete);
    assert!(entry.metadata.format.is_none());
}

#[cfg(unix)]
#[test]
fn gemma4_processor_config_symlink_is_not_variant_evidence() {
    let root = temp_dir("gemma4-processor-symlink");
    let path = write_model(&root, "gemma4", "gemma4");
    let outside = root.join("outside_processor_config.json");
    std::fs::write(&outside, r#"{"processor_class":"Gemma4Processor"}"#).unwrap();
    std::os::unix::fs::symlink(&outside, path.join("processor_config.json")).unwrap();

    let entry = catalog_entry(model("gemma4", path, RouterModelSource::ModelsDir));
    assert_eq!(entry.metadata.model_type.as_deref(), Some("gemma4"));
    assert!(
        entry
            .metadata
            .unknown_reasons
            .architecture
            .as_deref()
            .unwrap()
            .contains("Gemma 4 text-vs-vision classification requires")
    );
    assert!(!entry.supported);
}

#[cfg(unix)]
#[test]
fn pooling_parent_symlink_is_not_embedding_layout_evidence() {
    let root = temp_dir("pooling-parent-symlink");
    let path = write_model(&root, "qwen", "qwen3");
    let outside = root.join("outside_1_pooling");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(
        outside.join("config.json"),
        r#"{"pooling_mode_mean_tokens":true}"#,
    )
    .unwrap();
    std::os::unix::fs::symlink(&outside, path.join("1_Pooling")).unwrap();

    let entry = catalog_entry(model("qwen", path, RouterModelSource::ModelsDir));
    assert_eq!(entry.metadata.model_type.as_deref(), Some("qwen3"));
    assert!(entry.supported);
}
