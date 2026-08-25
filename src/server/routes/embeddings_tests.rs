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

//! `/v1/embeddings` route tests over the stub embedding model, driven
//! through the real router so the request flow (auth-free, JSON body,
//! worker dispatch, response shape) is exercised end to end.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{NO_EMBEDDING_MODEL_MESSAGE, embedding_error_response};
use crate::embeddings::stub::{STUB_DIM, STUB_VOCAB_SIZE, stub_loaded_model};
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};
use crate::server::embedding_worker::EmbeddingWorkerProvider;
use crate::server::types::embeddings::{
    EmbedItem, EmbeddingEncoding, EmbeddingInput, decode_base64_f32,
};
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const STUB_MODEL_ID: &str = "stub-embedding";

fn stub_provider(multi_vector: bool, batch_size: usize) -> Arc<dyn EmbeddingModelProvider> {
    Arc::new(
        EmbeddingWorkerProvider::from_loader(
            STUB_MODEL_ID.to_string(),
            batch_size,
            8,
            Duration::from_secs(30),
            move || Ok(stub_loaded_model(multi_vector)),
        )
        .expect("stub embedding worker spawns"),
    )
}

fn app_with(provider: Option<Arc<dyn EmbeddingModelProvider>>) -> axum::Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let model_provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = model_provider.batch_metrics().clone();
    let state = AppState::new(
        model_provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_embedding_model(provider);
    create_app(state)
}

async fn post(app: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("route responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let value: Value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body")
    };
    (status, value)
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("float array")
        .iter()
        .map(|v| v.as_f64().expect("float") as f32)
        .collect()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[tokio::test]
async fn string_input_returns_one_vector() {
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, body) = post(app, "/v1/embeddings", json!({"input": "hello world"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], STUB_MODEL_ID);
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    let vector = floats(&data[0]["embedding"]);
    assert_eq!(vector.len(), STUB_DIM);
    assert!((norm(&vector) - 1.0).abs() < 1e-5);
    assert!(
        data[0].get("shape").is_none(),
        "no shape for single-vector float"
    );
}

#[tokio::test]
async fn alias_path_without_v1_prefix_is_mounted() {
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, _) = post(app, "/embeddings", json!({"input": "hello"})).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn list_input_preserves_order_across_micro_batches() {
    // batch_size 1 makes every text its own micro-batch after the length
    // sort; the response must still line up with the request order.
    let app = app_with(Some(stub_provider(false, 1)));
    let (status, body) = post(
        app,
        "/v1/embeddings",
        json!({"input": ["hello world a b", "hello", "world a"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    for (i, entry) in data.iter().enumerate() {
        assert_eq!(entry["index"], i);
    }
    // "hello" (index 1) is one-hot on id 3 after [CLS]/[SEP] contribute zeros.
    let v1 = floats(&data[1]["embedding"]);
    assert!((v1[3] - 1.0).abs() < 1e-5, "{v1:?}");
    // "world a" (index 2) has mass on ids 4 and 5 and none on 3.
    let v2 = floats(&data[2]["embedding"]);
    assert!(v2[3].abs() < 1e-6 && v2[4] > 0.5 && v2[5] > 0.5, "{v2:?}");
    // "hello world a b" (index 0) shares ids with both.
    let v0 = floats(&data[0]["embedding"]);
    assert!(v0[3] > 0.0 && v0[4] > 0.0 && v0[6] > 0.0, "{v0:?}");
}

#[tokio::test]
async fn token_id_input() {
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, body) = post(app, "/v1/embeddings", json!({"input": [3, 4]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Verbatim ids: no [CLS]/[SEP] added, so usage counts exactly two.
    assert_eq!(body["usage"]["prompt_tokens"], 2);
    let v = floats(&body["data"][0]["embedding"]);
    assert!((v[3] - v[4]).abs() < 1e-6 && v[3] > 0.5);

    let app = app_with(Some(stub_provider(false, 16)));
    let (status, body) = post(app, "/v1/embeddings", json!({"input": [[3], [4, 5]]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(body["usage"]["prompt_tokens"], 3);
}

#[tokio::test]
async fn token_id_above_vocab_is_400() {
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, body) = post(
        app,
        "/v1/embeddings",
        json!({"input": [3, STUB_VOCAB_SIZE]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("vocab_size")
    );
}

#[tokio::test]
async fn base64_encoding_roundtrips_f32() {
    let app = app_with(Some(stub_provider(false, 16)));
    let (_, float_body) = post(app, "/v1/embeddings", json!({"input": "hello world"})).await;
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, b64_body) = post(
        app,
        "/v1/embeddings",
        json!({"input": "hello world", "encoding_format": "base64"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{b64_body}");
    let encoded = b64_body["data"][0]["embedding"]
        .as_str()
        .expect("base64 string");
    let decoded = decode_base64_f32(encoded).expect("valid base64 f32");
    let expected = floats(&float_body["data"][0]["embedding"]);
    assert_eq!(decoded.len(), expected.len());
    for (d, e) in decoded.iter().zip(&expected) {
        assert_eq!(d.to_bits(), e.to_bits(), "bit-for-bit identical");
    }
    assert!(b64_body["data"][0].get("shape").is_none());
}

#[tokio::test]
async fn unsupported_encoding_format_is_400() {
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, _) = post(
        app,
        "/v1/embeddings",
        json!({"input": "hello", "encoding_format": "int8"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn dimensions_truncates_and_renormalizes() {
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, body) = post(
        app,
        "/v1/embeddings",
        json!({"input": [3, 4], "dimensions": 5}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v = floats(&body["data"][0]["embedding"]);
    assert_eq!(v.len(), 5);
    assert!((norm(&v) - 1.0).abs() < 1e-5, "re-normalized: {v:?}");

    for bad in [0, STUB_DIM + 1] {
        let app = app_with(Some(stub_provider(false, 16)));
        let (status, body) = post(
            app,
            "/v1/embeddings",
            json!({"input": "hello", "dimensions": bad}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "dimensions={bad}: {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }
}

#[tokio::test]
async fn empty_input_is_400() {
    for input in [json!([]), json!(""), json!(["hello", ""]), json!([[3], []])] {
        let app = app_with(Some(stub_provider(false, 16)));
        let (status, body) = post(app, "/v1/embeddings", json!({"input": input})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "input={input}: {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }
    // A missing or malformed `input` is also a 400, not a 422.
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, _) = post(app, "/v1/embeddings", json!({"model": "x"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, _) = post(app, "/v1/embeddings", json!({"input": [1.5]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn no_embedding_model_is_501() {
    let app = app_with(None);
    let (status, body) = post(app, "/v1/embeddings", json!({"input": "hello"})).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(body["error"]["type"], "not_implemented");
    assert_eq!(body["error"]["message"], NO_EMBEDDING_MODEL_MESSAGE);
}

#[tokio::test]
async fn model_mismatch_is_400_and_match_is_accepted() {
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, body) = post(
        app,
        "/v1/embeddings",
        json!({"input": "hello", "model": "other-model"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let app = app_with(Some(stub_provider(false, 16)));
    let (status, _) = post(
        app,
        "/v1/embeddings",
        json!({"input": "hello", "model": STUB_MODEL_ID}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn usage_counts_real_tokens() {
    let app = app_with(Some(stub_provider(false, 16)));
    // [CLS] hello world [SEP] = 4, [CLS] a [SEP] = 3, verbatim [3, 4] = 2.
    let (status, body) = post(
        app,
        "/v1/embeddings",
        json!({"input": [{"type": "text", "text": "hello world"}, {"type": "text", "text": "a"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["usage"]["prompt_tokens"], 7);
    assert_eq!(body["usage"]["total_tokens"], 7);
}

#[tokio::test]
async fn image_item_is_400_for_text_only_model() {
    let app = app_with(Some(stub_provider(false, 16)));
    let (status, body) = post(
        app,
        "/v1/embeddings",
        json!({"input": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("image"));
}

#[tokio::test]
async fn multi_vector_model_returns_token_rows_and_shape_in_base64() {
    let app = app_with(Some(stub_provider(true, 16)));
    let (status, body) = post(app, "/v1/embeddings", json!({"input": [[3, 4, 5]]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let rows = body["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(rows.len(), 3, "one row per real token");
    assert_eq!(floats(&rows[0]).len(), STUB_DIM);

    let app = app_with(Some(stub_provider(true, 16)));
    let (status, body) = post(
        app,
        "/v1/embeddings",
        json!({"input": [[3, 4, 5]], "encoding_format": "base64"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"][0]["shape"], json!([3, STUB_DIM]));
    let decoded = decode_base64_f32(body["data"][0]["embedding"].as_str().unwrap()).unwrap();
    assert_eq!(decoded.len(), 3 * STUB_DIM);
}

#[test]
fn error_mapping_matches_audio_routes() {
    let queue_full = embedding_error_response(EmbeddingError::QueueFull);
    assert_eq!(queue_full.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(queue_full.error.error_type, "server_busy");
    let timeout = embedding_error_response(EmbeddingError::Timeout);
    assert_eq!(timeout.status, StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(timeout.error.error_type, "server_timeout");
    let invalid = embedding_error_response(EmbeddingError::InvalidInput("bad".into()));
    assert_eq!(invalid.status, StatusCode::BAD_REQUEST);
    assert_eq!(invalid.error.error_type, "invalid_request_error");
    let internal = embedding_error_response(EmbeddingError::Internal("boom".into()));
    assert_eq!(internal.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(internal.error.message.contains("boom"));
}

#[test]
fn input_shapes_flatten_to_ordered_items() {
    let text: EmbeddingInput = serde_json::from_value(json!("a")).unwrap();
    assert_eq!(text.into_items(), vec![EmbedItem::Text("a".into())]);
    let tokens: EmbeddingInput = serde_json::from_value(json!([1, 2])).unwrap();
    assert_eq!(tokens.into_items(), vec![EmbedItem::Tokens(vec![1, 2])]);
    let texts: EmbeddingInput = serde_json::from_value(json!(["a", "b"])).unwrap();
    assert_eq!(texts.into_items().len(), 2);
    let lists: EmbeddingInput = serde_json::from_value(json!([[1], [2, 3]])).unwrap();
    assert_eq!(
        lists.into_items(),
        vec![EmbedItem::Tokens(vec![1]), EmbedItem::Tokens(vec![2, 3])]
    );
    let parts: EmbeddingInput = serde_json::from_value(json!([
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "file:///x.png"}}
    ]))
    .unwrap();
    assert_eq!(
        parts.into_items(),
        vec![
            EmbedItem::Text("a".into()),
            EmbedItem::ImageUrl("file:///x.png".into())
        ]
    );
    assert!(serde_json::from_value::<EmbeddingInput>(json!([1, "a"])).is_err());

    assert_eq!(
        EmbeddingEncoding::parse(None),
        Some(EmbeddingEncoding::Float)
    );
    assert_eq!(
        EmbeddingEncoding::parse(Some("base64")),
        Some(EmbeddingEncoding::Base64)
    );
    assert_eq!(EmbeddingEncoding::parse(Some("int8")), None);
}
