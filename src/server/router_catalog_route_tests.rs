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
    let app = create_router_app_with_authenticated_ui(state);

    let before = paginate_catalog_ids(app.clone()).await;
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
