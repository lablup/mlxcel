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

use std::collections::BTreeSet;

use super::*;

async fn paginate_catalog_ids(app: Router) -> BTreeSet<String> {
    let mut cursor = None;
    let mut seen = BTreeSet::new();
    loop {
        let uri = match cursor.as_deref() {
            Some(cursor) => format!("/ui-api/v1/catalog?limit=200&cursor={cursor}"),
            None => "/ui-api/v1/catalog?limit=200".to_string(),
        };
        let (status, body) = send(app.clone(), Method::GET, &uri, "", Some(ROUTER_KEY)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["pagination"]["total_known"], 1000);
        let items = body["items"].as_array().expect("items");
        assert!(!items.is_empty(), "catalog page must not be empty");
        for item in items {
            let id = item["identity"]["id"].as_str().expect("id").to_string();
            assert!(seen.insert(id.clone()), "duplicate catalog id {id}");
        }
        cursor = body["pagination"]["next_cursor"]
            .as_str()
            .map(str::to_string);
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(seen.len(), 1000);
    seen
}

async fn wait_for_refresh_success(app: Router, operation_id: &str) -> serde_json::Value {
    for _ in 0..100 {
        let (status, body) = send(
            app.clone(),
            Method::GET,
            &format!("/ui-api/v1/operations/{operation_id}"),
            "",
            Some(ROUTER_KEY),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        if body["state"] == "succeeded" {
            return body;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("catalog refresh operation {operation_id} did not finish");
}

async fn catalog_page(app: Router, uri: &str) -> serde_json::Value {
    let (status, body) = send(app, Method::GET, uri, "", Some(ROUTER_KEY)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

async fn small_catalog_page(app: Router) -> serde_json::Value {
    catalog_page(app, "/ui-api/v1/catalog?limit=10").await
}

async fn refresh_catalog(app: Router) -> serde_json::Value {
    let (status, accepted) = send(
        app.clone(),
        Method::POST,
        "/ui-api/v1/catalog/refresh",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    let operation_id = accepted["operation_id"]
        .as_str()
        .expect("operation id")
        .to_string();
    wait_for_refresh_success(app, &operation_id).await
}

fn item_by_inference_id<'a>(
    body: &'a serde_json::Value,
    inference_id: &str,
) -> &'a serde_json::Value {
    body["items"]
        .as_array()
        .expect("items")
        .iter()
        .find(|item| item["identity"]["inference_id"] == inference_id)
        .unwrap_or_else(|| panic!("catalog item {inference_id} not found in {body}"))
}

async fn catalog_model_type(app: Router, inference_id: &str) -> String {
    let page = catalog_page(
        app,
        &format!("/ui-api/v1/catalog?limit=10&q={inference_id}"),
    )
    .await;
    item_by_inference_id(&page, inference_id)["metadata"]["model_type"]
        .as_str()
        .expect("model_type")
        .to_string()
}

#[tokio::test]
async fn catalog_route_paginates_thousand_entries_across_http_refresh() {
    let root = temp_models_dir("ui-catalog-page1000");
    for idx in 0..1000 {
        add_catalog_model(&root, &format!("model-{idx:04}"), "qwen3");
    }
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let catalog_cache = state.catalog_cache.clone();
    let app = create_router_app_with_authenticated_ui(state);

    let before = paginate_catalog_ids(app.clone()).await;
    catalog_cache.reset_heavy_metadata_probe_count();
    let cached = paginate_catalog_ids(app.clone()).await;
    assert_eq!(cached, before, "cached HTTP catalog IDs changed");
    assert_eq!(
        catalog_cache.heavy_metadata_probe_count(),
        0,
        "repeated HTTP catalog cache hit performed heavyweight metadata probes"
    );
    let (status, accepted) = send(
        app.clone(),
        Method::POST,
        "/ui-api/v1/catalog/refresh",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    let operation_id = accepted["operation_id"]
        .as_str()
        .expect("operation id")
        .to_string();
    let terminal = wait_for_refresh_success(app.clone(), &operation_id).await;
    assert_eq!(terminal["result"]["scanned_entries"], 1000);

    let after = paginate_catalog_ids(app).await;
    assert_eq!(after, before, "catalog IDs changed across HTTP refresh");
}

#[tokio::test]
async fn catalog_route_cache_is_scoped_per_router_pool() {
    let root_a = temp_models_dir("ui-catalog-cache-pool-a");
    let root_b = temp_models_dir("ui-catalog-cache-pool-b");
    for idx in 0..1000 {
        let name = format!("model-{idx:04}");
        add_catalog_model(&root_a, &name, "qwen3");
        add_catalog_model(&root_b, &name, "qwen2");
    }
    let state_a = router_state_from(
        RouterSources {
            models_dir: Some(root_a),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let cache_a = state_a.catalog_cache.clone();
    let app_a = create_router_app_with_authenticated_ui(state_a);
    let state_b = router_state_from(
        RouterSources {
            models_dir: Some(root_b),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let cache_b = state_b.catalog_cache.clone();
    let app_b = create_router_app_with_authenticated_ui(state_b);

    let (ids_a, ids_b) = tokio::join!(
        paginate_catalog_ids(app_a.clone()),
        paginate_catalog_ids(app_b.clone())
    );
    assert_eq!(ids_a, ids_b, "identical names should keep stable IDs");
    let (model_type_a, model_type_b) = tokio::join!(
        catalog_model_type(app_a.clone(), "model-0000"),
        catalog_model_type(app_b.clone(), "model-0000")
    );
    assert_eq!(model_type_a, "qwen3");
    assert_eq!(model_type_b, "qwen2");

    cache_a.reset_heavy_metadata_probe_count();
    cache_b.reset_heavy_metadata_probe_count();
    let (terminal_b, warm_a) = tokio::join!(
        refresh_catalog(app_b.clone()),
        paginate_catalog_ids(app_a.clone())
    );
    assert_eq!(terminal_b["result"]["scanned_entries"], 1000);
    assert_eq!(
        warm_a, ids_a,
        "warm pool A IDs changed while pool B refreshed"
    );
    assert_eq!(
        cache_a.heavy_metadata_probe_count(),
        0,
        "pool B refresh evicted or invalidated warm pool A cache"
    );
    assert!(
        cache_b.heavy_metadata_probe_count() > 0,
        "pool B refresh should measure its own metadata"
    );
    assert_eq!(
        catalog_model_type(app_a, "model-0000").await,
        "qwen3",
        "pool B metadata leaked into pool A"
    );
}

#[tokio::test]
async fn explicit_refresh_detects_same_size_config_and_index_edits() {
    let root = temp_models_dir("ui-catalog-same-size-refresh");
    add_catalog_model(&root, "alpha", "qwen3");
    let indexed = root.join("indexed");
    std::fs::create_dir_all(&indexed).expect("indexed model dir");
    let config_qwen3 = r#"{"model_type":"qwen3","quantization_config":{"bits":4}}"#;
    let config_qwen2 = r#"{"model_type":"qwen2","quantization_config":{"bits":4}}"#;
    assert_eq!(config_qwen3.len(), config_qwen2.len());
    std::fs::write(indexed.join("config.json"), config_qwen3).expect("config");
    let index_a = r#"{"weight_map":{"model.embed_tokens.weight":"a.safetensors"}}"#;
    let index_b = r#"{"weight_map":{"model.embed_tokens.weight":"b.safetensors"}}"#;
    assert_eq!(index_a.len(), index_b.len());
    std::fs::write(indexed.join("model.safetensors.index.json"), index_a).expect("index");
    std::fs::write(indexed.join("a.safetensors"), b"weights-a").expect("shard a");
    std::fs::write(indexed.join("b.safetensors"), b"weights-b").expect("shard b");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root.clone()),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let app = create_router_app_with_authenticated_ui(state);

    let before = small_catalog_page(app.clone()).await;
    assert_eq!(
        item_by_inference_id(&before, "alpha")["metadata"]["model_type"],
        "qwen3"
    );
    let indexed_before_fingerprint =
        item_by_inference_id(&before, "indexed")["identity"]["content_fingerprint"].clone();

    std::fs::write(root.join("alpha/config.json"), config_qwen2).expect("edit config");
    std::fs::write(indexed.join("model.safetensors.index.json"), index_b).expect("edit index");
    let terminal = refresh_catalog(app.clone()).await;
    assert_eq!(terminal["result"]["scanned_entries"], 2);
    assert_eq!(terminal["result"]["changed_entries"], 2);

    let after = small_catalog_page(app).await;
    assert_eq!(
        item_by_inference_id(&after, "alpha")["metadata"]["model_type"],
        "qwen2"
    );
    let indexed_after = item_by_inference_id(&after, "indexed");
    assert_eq!(indexed_after["complete"], true);
    assert_ne!(
        indexed_after["identity"]["content_fingerprint"],
        indexed_before_fingerprint
    );
}

#[tokio::test]
async fn legacy_models_reload_invalidates_catalog_cache_by_epoch() {
    let root = temp_models_dir("ui-catalog-legacy-reload");
    add_catalog_model(&root, "alpha", "qwen3");
    let app = create_router_app_with_authenticated_ui(router_state_from(
        RouterSources {
            models_dir: Some(root.clone()),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    ));
    let before = small_catalog_page(app.clone()).await;
    assert_eq!(
        item_by_inference_id(&before, "alpha")["metadata"]["model_type"],
        "qwen3"
    );

    let config_qwen2 = r#"{"model_type":"qwen2","quantization_config":{"bits":4}}"#;
    std::fs::write(root.join("alpha/config.json"), config_qwen2).expect("edit config");
    let (status, models) = send(
        app.clone(),
        Method::GET,
        "/models?reload=1",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{models}");
    let after = small_catalog_page(app).await;
    assert_eq!(
        item_by_inference_id(&after, "alpha")["metadata"]["model_type"],
        "qwen2"
    );
}

#[tokio::test]
async fn cache_download_refreshes_catalog_inventory_after_empty_cache_hit() {
    let root = temp_models_dir("ui-catalog-download-root");
    let cache_root = temp_models_dir("ui-catalog-download-cache");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: Some(crate::server::router_cache::CacheSource::new(
                cache_root,
                std::sync::Arc::new(InstantDownloader),
            )),
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let mut events = state.pool.subscribe();
    let app = create_router_app_with_authenticated_ui(state);
    let empty = small_catalog_page(app.clone()).await;
    assert!(empty["items"].as_array().unwrap().is_empty());

    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/models",
        r#"{"model":"mlx-community/fresh"}"#,
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let event = events.recv().await.expect("events open");
            if event["event"] == "download_finished" {
                break;
            }
        }
    })
    .await
    .expect("download_finished");

    let after = small_catalog_page(app).await;
    let downloaded = item_by_inference_id(&after, "mlx-community/fresh");
    assert_eq!(downloaded["identity"]["source"], "cache");
    assert_eq!(downloaded["removal"]["eligible"], true);
}

#[tokio::test]
async fn mounted_dflash_catalog_matches_bounded_whole_fixture() {
    let root = temp_models_dir("ui-catalog-dflash-contract");
    add_catalog_model(&root, "dflash", "qwen3");
    let model_path = root.join("dflash");
    std::fs::write(
        model_path.join("config.json"),
        r#"{"model_type":"qwen3","dflash_config":{},"quantization_config":{"bits":4}}"#,
    )
    .expect("DFlash config");
    // This real loader entry point delegates to detect_model_type_with_probes;
    // structural DFlash rejection precedes all weight/probe inspection.
    let raw = crate::models::get_model_type(&model_path)
        .expect_err("a DFlash drafter is not a standalone model")
        .to_string();
    assert!(
        raw.chars().count() > 512,
        "exercise the actual long diagnostic"
    );
    assert!(raw.contains("not a standalone model."));
    let app = create_router_app_with_authenticated_ui(router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    ));
    let mut actual = catalog_page(app, "/ui-api/v1/catalog?limit=50").await;
    let metadata = &actual["items"][0]["metadata"];
    for reason in [
        &metadata["support"]["architecturally_supported_reason"],
        &metadata["unknown_reasons"]["architecture"],
    ] {
        let reason = reason.as_str().expect("unsupported architecture reason");
        assert_eq!(reason.chars().count(), 512);
        assert!(reason.contains("not a standalone model."));
        assert!(reason.ends_with('…'));
        assert!(!reason.contains(model_path.to_str().expect("test path")));
    }
    let mut expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/webui/examples/catalog.dflash-page.json"
    ))
    .expect("canonical DFlash fixture");
    expected
        .as_object_mut()
        .expect("fixture object")
        .remove("$schemaName");
    // Normalize only values derived from temporary paths and process identity.
    // Sequence, revisions, capabilities, reasons and every other field remain
    // part of the exact, schema-validated whole HTTP producer comparison.
    let server_id = actual["server_instance_id"].as_str().expect("server id");
    assert!(server_id.starts_with("srv_") && server_id.len() <= 128);
    assert!(
        server_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'~' | b'-'))
    );
    assert_eq!(actual["items"].as_array().expect("catalog items").len(), 1);
    let identity = &actual["items"][0]["identity"];
    let id = identity["id"].as_str().expect("model id");
    assert!(id.starts_with("mdl_") && id.len() == 47);
    assert!(
        id[4..]
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    );
    let hash = identity["source_key_hash"].as_str().expect("source hash");
    assert!(hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
    let fingerprint = identity["content_fingerprint"]
        .as_str()
        .expect("fingerprint");
    assert!(!fingerprint.is_empty() && fingerprint.len() <= 128);
    assert!(
        actual["items"][0]["metadata"]["disk_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    actual["server_instance_id"] = serde_json::json!("srv_fixture");
    for (field, replacement) in [
        ("id", "mdl_uB9xAUQSKlrb9ELybOgV-92lVC7XjiMXju6pwZZBbAU"),
        (
            "source_key_hash",
            "fb957973e2e6f9fb17bbb5bf2922d6bb67dde35b08c94fa80f8154722b58af2e",
        ),
        ("content_fingerprint", "fixture-content-fingerprint"),
    ] {
        actual["items"][0]["identity"][field] = serde_json::json!(replacement);
    }
    actual["items"][0]["metadata"]["disk_bytes"] = serde_json::json!(12345);
    assert_eq!(actual, expected, "mounted DFlash catalog contract drift");
}
