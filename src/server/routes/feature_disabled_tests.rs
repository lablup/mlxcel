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

//! Route tests for the b10621 disabled-feature stubs (#1435).

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use crate::server::model_provider::ModelProvider;
use crate::server::{AppState, ChatTemplateProcessor, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn app() -> axum::Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    );
    create_app(state)
}

#[tokio::test]
async fn tools_and_cors_proxy_answer_the_b10621_feature_disabled_envelope() {
    for (method, path) in [
        ("GET", "/tools"),
        ("POST", "/tools"),
        ("GET", "/cors-proxy"),
        ("POST", "/cors-proxy"),
    ] {
        let mut builder = Request::builder().method(method).uri(path);
        let body = if method == "POST" {
            builder = builder.header("content-type", "application/json");
            Body::from("{}")
        } else {
            Body::empty()
        };
        let response = app()
            .oneshot(builder.body(body).expect("request builds"))
            .await
            .expect("router responds");
        assert_eq!(
            response.status(),
            StatusCode::FORBIDDEN,
            "{method} {path} must answer 403"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body collects");
        // Byte-shaped like upstream's res_403: message + type, no code.
        assert_eq!(
            String::from_utf8_lossy(&bytes),
            r#"{"error":{"message":"this feature is disabled","type":"feature_disabled"}}"#,
            "{method} {path} must carry upstream's envelope"
        );
    }
}
