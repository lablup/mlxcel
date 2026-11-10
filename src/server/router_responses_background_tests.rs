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

//! Background Responses on the router pool (issue #2901), through the real
//! router app and real per-model sub-apps over scripted providers:
//! retrieve, delete and cancel find the model that owns a response id without
//! a `model`, eviction leaves a model with a running job loaded, and an
//! explicit unload ends its jobs `failed`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{RouterServerState, create_router_app, response_affinity_id};
use crate::server::config::ServerConfig;
use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::responses_store::{ResponsesStore, ResponsesStoreConfig};
use crate::server::router_lifecycle::OperationState;
use crate::server::router_models::{
    ModelActionEvictionTarget, RouterModelAction, RouterPool, RouterSources,
};
use crate::server::router_presets::PresetCliOverrides;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerStartupConfig};

const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";
const WAIT: Duration = Duration::from_secs(10);

/// Eviction and explicit unload of a model with running jobs.
#[path = "router_responses_background_unload_tests.rs"]
mod unload;

/// What the test keeps of one loaded model: the script driver of its
/// provider and the store its sub-app serves from.
struct Loaded {
    handle: ScriptedStreamHandle,
    store: Arc<ResponsesStore>,
}

struct Harness {
    _root: tempfile::TempDir,
    pool: Arc<RouterPool>,
    app: Router,
    loaded: Arc<Mutex<HashMap<String, Loaded>>>,
}

impl Harness {
    /// A router over `models`, each a fake checkpoint directory, with room
    /// for `models_max` loaded at once. Nothing is loaded yet.
    fn new(models: &[&str], models_max: usize) -> Self {
        let root = tempfile::tempdir().expect("models dir");
        for name in models {
            let dir = root.path().join(name);
            std::fs::create_dir(&dir).expect("model dir");
            std::fs::write(dir.join("config.json"), "{}").expect("config.json");
        }
        let pool = Arc::new(
            RouterPool::new(
                RouterSources {
                    models_dir: Some(root.path().into()),
                    ..Default::default()
                },
                ServerStartupConfig::default(),
                Default::default(),
                PresetCliOverrides::default(),
                models_max,
                false,
            )
            .expect("pool"),
        );
        let loaded: Arc<Mutex<HashMap<String, Loaded>>> = Arc::default();
        let registry = loaded.clone();
        pool.set_model_app_factory_for_tests(Some(Arc::new(move |path, config| {
            let (options_tx, _options_rx) = std::sync::mpsc::channel();
            let (provider, handle) = ModelProvider::scripted_streaming_for_route_tests(options_tx);
            let provider = Arc::new(provider);
            let store = Arc::new(ResponsesStore::new(ResponsesStoreConfig::default()));
            let state = AppState::new(
                provider.clone(),
                config,
                ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
                crate::tokenizer::MlxcelTokenizer::stub(),
                path.to_path_buf(),
                provider.batch_metrics().clone(),
            )
            .with_responses_store(Some(store.clone()));
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("model directory name")
                .to_string();
            registry
                .lock()
                .expect("loaded registry")
                .insert(name, Loaded { handle, store });
            let router = crate::server::app::create_app_without_cors(state.clone());
            Ok((state, router))
        })));
        let app = create_router_app(RouterServerState {
            pool: pool.clone(),
            config: Arc::new(ServerConfig::default()),
            #[cfg(feature = "webui")]
            startup: Arc::new(ServerStartupConfig::default()),
            #[cfg(feature = "webui")]
            catalog_cache: Arc::new(crate::server::webui::catalog::CatalogProjectionCache::new()),
        });
        Self {
            _root: root,
            pool,
            app,
            loaded,
        }
    }

    async fn load(&self, name: &str) {
        self.pool
            .ensure_ready(name, WAIT)
            .await
            .unwrap_or_else(|err| panic!("load {name}: {err:?}"));
    }

    fn store(&self, name: &str) -> Arc<ResponsesStore> {
        self.loaded.lock().expect("loaded registry")[name]
            .store
            .clone()
    }

    /// Run `drive` against the script handle of `name`'s provider.
    fn script(&self, name: &str, drive: impl FnOnce(&ScriptedStreamHandle)) {
        drive(&self.loaded.lock().expect("loaded registry")[name].handle);
    }

    fn running(&self, name: &str) -> bool {
        self.pool.get(name).expect("entry").is_running()
    }

    async fn send(&self, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
            .expect("request builds");
        let response = self.app.clone().oneshot(request).await.expect("responds");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .expect("body collects");
        let body = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| panic!("{status} body is not JSON: {bytes:?}"));
        (status, body)
    }

    /// Create a background response on `model` and return its id.
    async fn create_background(&self, model: &str) -> String {
        let (status, body) = self
            .send(
                Method::POST,
                "/v1/responses",
                Some(json!({"model": model, "input": "hi", "background": true})),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "queued", "{body}");
        body["id"].as_str().expect("response id").to_string()
    }

    /// Create a job on `model` and feed its first token, so it is
    /// `in_progress` with the scripted worker holding the generation.
    async fn create_running(&self, model: &str) -> String {
        let id = self.create_background(model).await;
        self.script(model, |handle| handle.token("partial"));
        self.poll_until(&id, "in_progress").await;
        id
    }

    async fn get(&self, id: &str) -> (StatusCode, Value) {
        self.send(Method::GET, &format!("/v1/responses/{id}"), None)
            .await
    }

    /// Poll without a `model` until the response reports `status`.
    async fn poll_until(&self, id: &str, status: &str) -> Value {
        let deadline = Instant::now() + WAIT;
        loop {
            let (code, body) = self.get(id).await;
            assert_eq!(code, StatusCode::OK, "{id}: {body}");
            if body["status"] == status {
                return body;
            }
            assert!(
                Instant::now() < deadline,
                "response {id} never reached {status}: {body}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Submit a WebUI-style load of `name` that names `victim` as the model
    /// to evict, and return the terminal operation state with its error.
    async fn load_evicting(&self, name: &str, victim: &str) -> (OperationState, String) {
        let entry = self.pool.get(name).expect("entry");
        let victim = self.pool.get(victim).expect("victim entry");
        let accepted = self
            .pool
            .submit_model_action(
                &entry.ui_model_id,
                RouterModelAction::Load,
                entry.lifecycle_revision(),
                &format!("evict-{}", uuid::Uuid::new_v4().simple()),
                Some(ModelActionEvictionTarget::new(
                    &victim.ui_model_id,
                    victim.lifecycle_revision(),
                )),
            )
            .expect("load accepted");
        let coordinator = self.pool.lifecycle_coordinator();
        let deadline = Instant::now() + WAIT;
        loop {
            let operation = coordinator
                .get_operation(&accepted.operation_id)
                .expect("operation recorded");
            if matches!(
                operation.state,
                OperationState::Succeeded | OperationState::Failed | OperationState::Cancelled
            ) {
                let message = operation.error.map(|error| error.message);
                return (operation.state, message.unwrap_or_default());
            }
            assert!(Instant::now() < deadline, "load operation never finished");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

fn assert_not_found(status: StatusCode, body: &Value, id: &str) {
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["type"], "not_found", "{body}");
    assert_eq!(
        body["error"]["message"],
        format!("response '{id}' not found"),
        "{body}"
    );
}

#[tokio::test]
async fn retrieve_without_model_reaches_the_job_on_its_model() {
    let h = Harness::new(&["alpha", "beta"], 2);
    h.load("alpha").await;
    h.load("beta").await;

    let id = h.create_background("beta").await;
    assert_eq!(h.store("beta").active_background_jobs(), 1);
    assert_eq!(h.store("alpha").active_background_jobs(), 0);

    // No `model` anywhere: the router finds beta by the id.
    let (status, body) = h.get(&id).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], id.as_str());
    assert_eq!(body["status"], "queued");
    assert_eq!(body["background"], true);
    // The unprefixed alias and the resume form route the same way.
    let (status, body) = h.send(Method::GET, &format!("/responses/{id}"), None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], id.as_str());

    // An explicit model keeps the ordinary path: the owner answers, another
    // loaded model does not know the id, an unknown model is refused.
    let (status, body) = h
        .send(Method::GET, &format!("/v1/responses/{id}?model=beta"), None)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], id.as_str());
    let (status, body) = h
        .send(
            Method::GET,
            &format!("/v1/responses/{id}?model=alpha"),
            None,
        )
        .await;
    assert_not_found(status, &body, &id);
    let (status, body) = h
        .send(
            Method::GET,
            &format!("/v1/responses/{id}?model=gamma"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    h.script("beta", |handle| {
        handle.token("done");
        handle.finish();
    });
    let body = h.poll_until(&id, "completed").await;
    assert_eq!(body["output_text"], "done", "{body}");

    // The stored terminal object stays reachable, and DELETE finds it too.
    let (status, body) = h
        .send(Method::DELETE, &format!("/v1/responses/{id}"), None)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["deleted"], true, "{body}");
    let (status, body) = h.get(&id).await;
    assert_not_found(status, &body, &id);
}

#[tokio::test]
async fn cancel_without_model_returns_the_full_response() {
    let h = Harness::new(&["alpha", "beta"], 2);
    h.load("alpha").await;
    h.load("beta").await;
    let id = h.create_background("alpha").await;

    // The OpenAI SDK sends this POST with no body and no query.
    let (status, body) = h
        .send(Method::POST, &format!("/v1/responses/{id}/cancel"), None)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], id.as_str());
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "cancelled");
    assert_eq!(body["background"], true);
    assert_eq!(body["model"], "alpha", "{body}");
    assert!(body["output"].is_array(), "{body}");
    assert_eq!(h.store("alpha").active_background_jobs(), 0);

    // A repeat through the alias, and one naming the model in the query,
    // both return the same recorded object.
    let (status, again) = h
        .send(Method::POST, &format!("/responses/{id}/cancel"), None)
        .await;
    assert_eq!(status, StatusCode::OK, "{again}");
    assert_eq!(again, body);
    let (status, again) = h
        .send(
            Method::POST,
            &format!("/v1/responses/{id}/cancel?model=alpha"),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{again}");
    assert_eq!(again, body);

    // Release the scripted worker in case the request reached it.
    h.script("alpha", |handle| handle.finish());
}

#[tokio::test]
async fn unknown_response_id_is_404_not_a_missing_model() {
    let h = Harness::new(&["alpha", "beta"], 2);
    // Nothing loaded: the probe has nothing to ask.
    let (status, body) = h.get("resp_missing").await;
    assert_not_found(status, &body, "resp_missing");

    h.load("alpha").await;
    h.load("beta").await;
    for (method, path) in [
        (Method::GET, "/v1/responses/resp_missing"),
        (Method::GET, "/responses/resp_missing"),
        (Method::DELETE, "/v1/responses/resp_missing"),
        (Method::POST, "/v1/responses/resp_missing/cancel"),
        (Method::POST, "/responses/resp_missing/cancel"),
    ] {
        let (status, body) = h.send(method.clone(), path, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{method} {path}: {body}");
        assert_eq!(body["error"]["type"], "not_found", "{method} {path}");
    }

    // A create without a model is still the dispatcher's missing-name 400.
    let (status, body) = h
        .send(Method::POST, "/v1/responses", Some(json!({"input": "hi"})))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[test]
fn affinity_covers_retrieve_delete_and_cancel_only() {
    let id = |method: Method, path: &str| response_affinity_id(&method, path);
    assert_eq!(
        id(Method::GET, "/v1/responses/resp_1").as_deref(),
        Some("resp_1")
    );
    assert_eq!(
        id(Method::DELETE, "/responses/resp_1").as_deref(),
        Some("resp_1")
    );
    assert_eq!(
        id(Method::POST, "/v1/responses/resp_1/cancel").as_deref(),
        Some("resp_1")
    );
    assert_eq!(
        id(Method::GET, "/v1/responses/resp%5F1").as_deref(),
        Some("resp_1")
    );
    // Creates, the static routes, other methods and deeper paths keep the
    // ordinary model resolution.
    assert_eq!(id(Method::POST, "/v1/responses"), None);
    assert_eq!(id(Method::POST, "/v1/responses/resp_1"), None);
    assert_eq!(id(Method::POST, "/v1/responses/compact"), None);
    assert_eq!(id(Method::GET, "/v1/responses/compact"), None);
    assert_eq!(id(Method::POST, "/v1/responses/input_tokens"), None);
    assert_eq!(id(Method::GET, "/v1/responses/"), None);
    assert_eq!(id(Method::GET, "/v1/responses/resp_1/input_items"), None);
    assert_eq!(id(Method::PATCH, "/v1/responses/resp_1"), None);
    assert_eq!(id(Method::GET, "/v1/chat/completions/resp_1"), None);
}
