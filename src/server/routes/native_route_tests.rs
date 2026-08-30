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

//! Native versus OpenAI route separation (#1441).
//!
//! `llama-server` b10621 sends `/completion` and `/completions` to one handler
//! and `/v1/completions` to a different one, and does the same for
//! `/embedding` / `/embeddings` against `/v1/embeddings`. mlxcel answered the
//! OpenAI shape on all of them. These tests assert the split by the shape of
//! the body each path returns, which is the only thing a client can observe.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tower::ServiceExt;

use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn app() -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("native-route-test-model"),
        batch_metrics,
    );
    create_app(state)
}

async fn post(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app().oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, parsed)
}

/// The native completion body, whose shape is what separates the handlers.
fn native_prompt() -> serde_json::Value {
    serde_json::json!({"prompt": "hello", "n_predict": 4})
}

#[tokio::test]
async fn the_native_completion_paths_answer_the_native_shape() {
    // `content` at the top level with `tokens_predicted` beside it is the
    // native object; the OpenAI one nests text under `choices`.
    for path in ["/completion", "/completions"] {
        let (status, body) = post(path, native_prompt()).await;
        assert_eq!(status, StatusCode::OK, "{path}: {body}");
        assert!(
            body.get("content").is_some(),
            "{path} must answer the native shape, got {body}"
        );
        assert!(
            body.get("choices").is_none(),
            "{path} must not answer the OpenAI shape, got {body}"
        );
        assert!(body.get("tokens_predicted").is_some(), "{path}: {body}");
    }
}

#[tokio::test]
async fn the_v1_completion_path_stays_openai() {
    let (status, body) = post(
        "/v1/completions",
        serde_json::json!({"model": "native-route-test-model", "prompt": "hello", "max_tokens": 4}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body.get("choices").is_some(),
        "/v1/completions must stay OpenAI compatible, got {body}"
    );
    assert_eq!(body["object"], "text_completion");
    assert!(
        body.get("content").is_none(),
        "/v1/completions must not answer the native shape, got {body}"
    );
}

#[tokio::test]
async fn the_native_completion_response_carries_the_b10621_key_set() {
    let (_, body) = post("/completion", native_prompt()).await;
    for key in [
        "index",
        "content",
        "tokens",
        "id_slot",
        "stop",
        "model",
        "tokens_predicted",
        "tokens_evaluated",
        "generation_settings",
        "prompt",
        "has_new_line",
        "truncated",
        "stop_type",
        "stopping_word",
        "tokens_cached",
        "timings",
    ] {
        assert!(body.get(key).is_some(), "missing {key} in {body}");
    }
    assert!(
        body["timings"].get("cache_n").is_some(),
        "timings must lead with cache_n: {body}"
    );
    assert!(
        body["generation_settings"]
            .as_object()
            .is_some_and(|m| !m.is_empty()),
        "generation_settings must report the resolved settings, got {body}"
    );
}

#[tokio::test]
async fn the_native_completion_echoes_the_prompt_and_stop_metadata() {
    let (_, body) = post("/completion", native_prompt()).await;
    assert_eq!(body["prompt"], "hello");
    assert_eq!(body["index"], 0);
    assert_eq!(body["stop"], true);
    assert_eq!(body["stopping_word"], "");
    assert!(
        ["limit", "eos"].contains(&body["stop_type"].as_str().unwrap_or("")),
        "stop_type must be one of the reasons mlxcel can distinguish: {body}"
    );
}

/// A matched string stop sequence must reach the wire as b10621 reports it:
/// `stop_type: "word"` with the matched string in `stopping_word` (issue #1466).
/// Before the fix the field could only ever be `""`, because nothing on the MLX
/// serving path detected a stop-string match at all.
#[tokio::test]
async fn a_matched_stop_string_is_reported_as_stop_type_word() {
    let (_, body) = post(
        "/completion",
        serde_json::json!({"prompt": "hello", "n_predict": 30, "stop": ["5"]}),
    )
    .await;
    assert_eq!(body["stop_type"], "word", "{body}");
    assert_eq!(body["stopping_word"], "5", "{body}");
    // The request's stop list is echoed back in the resolved settings, so a
    // client can see the server acted on the value it sent.
    assert_eq!(
        body["generation_settings"]["stop"],
        serde_json::json!(["5"])
    );
}

/// `/completions` shares the handler, so the same mapping must hold there.
#[tokio::test]
async fn the_completions_alias_reports_the_matched_stop_string_too() {
    let (_, body) = post(
        "/completions",
        serde_json::json!({"prompt": "hello", "n_predict": 30, "stop": "5"}),
    )
    .await;
    assert_eq!(body["stop_type"], "word", "{body}");
    assert_eq!(body["stopping_word"], "5", "{body}");
}

#[tokio::test]
async fn n_predict_accepts_the_openai_aliases() {
    // b10621 declares `max_tokens` and `max_completion_tokens` as aliases, so
    // the same body reaches the native route whichever spelling a client uses.
    for key in ["n_predict", "max_tokens", "max_completion_tokens"] {
        let (status, body) = post(
            "/completion",
            serde_json::json!({"prompt": "hello", key: 4}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{key}: {body}");
        assert_eq!(
            body["generation_settings"]["n_predict"], 4,
            "{key} must resolve the token budget: {body}"
        );
    }
}

#[tokio::test]
async fn unsupported_native_fields_are_refused_with_a_diagnostic() {
    // The epic's rule is that a field whose value has observable semantics is
    // never silently ignored. Each of these names the field and the
    // alternative.
    for (field, body) in [
        ("n_cmpl", serde_json::json!({"prompt": "hi", "n_cmpl": 2})),
        ("n_cmpl", serde_json::json!({"prompt": "hi", "n": 2})),
        (
            "n_indent",
            serde_json::json!({"prompt": "hi", "n_indent": 4}),
        ),
        (
            "t_max_predict_ms",
            serde_json::json!({"prompt": "hi", "t_max_predict_ms": 500}),
        ),
        (
            "return_progress",
            serde_json::json!({"prompt": "hi", "return_progress": true}),
        ),
        (
            "return_tokens",
            serde_json::json!({"prompt": "hi", "return_tokens": true}),
        ),
    ] {
        let (status, response) = post("/completion", body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{field} must be refused, got {response}"
        );
        let message = response["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains(field),
            "the diagnostic must name {field}, got {message:?}"
        );
    }
}

/// `tokens_cached` is b10621's SLOT cache occupancy after the request, not
/// what the prefix cache supplied for it (#1477).
///
/// Six independent measurements against the pinned binary agree on
/// `tokens_evaluated + tokens_predicted - 1`: 5+8-1=12 on a limit stop,
/// 5+1-1=5 on the `n_predict: 0` prompt-only case (upstream still answers it
/// with `tokens_predicted: 1`), 8+24-1=31 and 11+40-1=50 on longer runs,
/// 5+200-1=204, and 15+4-1=18 on a fully cache-hit request, where the figure
/// is unchanged by the hit. `timings.cache_n` stays the cache-supplied count,
/// which is the different quantity `prompt_n` is derived from, so this test
/// pins both at once.
#[tokio::test]
async fn tokens_cached_reports_the_upstream_slot_occupancy_formula() {
    let (status, body) = post(
        "/completion",
        serde_json::json!({"prompt": "hello there", "n_predict": 4}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let evaluated = body["tokens_evaluated"].as_u64().expect("tokens_evaluated");
    let predicted = body["tokens_predicted"].as_u64().expect("tokens_predicted");
    let cached = body["tokens_cached"].as_u64().expect("tokens_cached");
    assert_eq!(
        cached,
        (evaluated + predicted).saturating_sub(1),
        "tokens_cached must be tokens_evaluated + tokens_predicted - 1 (saturating), got \
         {cached} from {evaluated} + {predicted}"
    );
    assert_eq!(
        body["timings"]["cache_n"].as_u64(),
        Some(0),
        "cache_n is the cache-supplied prompt count and is a different quantity"
    );
    assert_eq!(
        body["timings"]["prompt_n"].as_u64(),
        Some(evaluated),
        "prompt_n stays tokens_evaluated - cache_n"
    );
}

/// b10621 accepts `verbose` on the native route and ignores it (#1477).
///
/// Upstream writes its `__verbose` debug block only from the OAI-compat
/// response builders; the native `/completion` object IS
/// `to_json_non_oaicompat()`, so the field changes nothing there. Measured
/// against the pinned binary, the top-level key set with `verbose: true` is
/// identical to the key set without it. mlxcel used to refuse the field with
/// a 400, which was the divergence; it now matches upstream by accepting it
/// and changing nothing.
#[tokio::test]
async fn verbose_is_accepted_and_changes_nothing() {
    let (with_status, with_body) = post(
        "/completion",
        serde_json::json!({"prompt": "hello", "n_predict": 4, "verbose": true}),
    )
    .await;
    assert_eq!(with_status, StatusCode::OK, "{with_body}");
    let (without_status, without_body) = post(
        "/completion",
        serde_json::json!({"prompt": "hello", "n_predict": 4}),
    )
    .await;
    assert_eq!(without_status, StatusCode::OK, "{without_body}");

    let keys = |v: &serde_json::Value| {
        let mut k: Vec<String> = v
            .as_object()
            .expect("the native response is an object")
            .keys()
            .cloned()
            .collect();
        k.sort();
        k
    };
    assert_eq!(
        keys(&with_body),
        keys(&without_body),
        "verbose must not add or remove a key, as it does not upstream"
    );
    assert!(
        with_body.get("__verbose").is_none(),
        "the native route has no __verbose block upstream and must not invent one"
    );
}

#[tokio::test]
async fn the_inert_value_of_an_unsupported_field_is_accepted() {
    // A client that sends the whole schema at its defaults must not be turned
    // away: only a value that would change behavior is refused.
    let (status, body) = post(
        "/completion",
        serde_json::json!({
            "prompt": "hello",
            "n_predict": 4,
            "n_cmpl": 1,
            "n_indent": 0,
            "t_max_predict_ms": -1,
            "return_progress": false,
            "verbose": false,
            "return_tokens": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn an_unknown_native_field_is_ignored_as_upstream_ignores_it() {
    // b10621 has no deny-unknown-fields equivalent: an unrecognised key is
    // accepted and the request succeeds. Rejecting here would turn away a
    // request llama-server serves.
    let (status, body) = post(
        "/completion",
        serde_json::json!({"prompt": "hello", "n_predict": 4, "totally_unknown_field": 123}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn the_native_embedding_paths_are_mounted_and_separate_from_v1() {
    // No embedding model is loaded in this fixture, so all three answer 501.
    // What matters here is that the native paths resolve at all: `/embedding`
    // was not mounted before this change.
    for path in ["/embedding", "/embeddings", "/v1/embeddings"] {
        let (status, body) = post(path, serde_json::json!({"input": "hello"})).await;
        assert_ne!(status, StatusCode::NOT_FOUND, "{path} must be mounted");
        assert_ne!(
            status,
            StatusCode::METHOD_NOT_ALLOWED,
            "{path} must accept POST"
        );
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {body}");
    }
}

#[tokio::test]
async fn the_native_embedding_path_accepts_the_content_spelling() {
    // Upstream's legacy `/embedding` takes `{"content": ...}` rather than
    // `{"input": ...}`; reaching the same 501 proves the body parsed.
    let (status, _) = post("/embedding", serde_json::json!({"content": "hello"})).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
}

// ---------------------------------------------------------------------------
// `response_fields`, `stream_options` and the `n_predict` value domain (#1441)
//
// The expectations are captures of the pinned b10621 binary answering the same
// bodies against a real checkpoint.
// ---------------------------------------------------------------------------

fn keys(body: &serde_json::Value) -> Vec<String> {
    body.as_object().expect("object").keys().cloned().collect()
}

#[tokio::test]
async fn response_fields_projects_the_native_body() {
    let (status, body) = post(
        "/completion",
        serde_json::json!({
            "prompt": "hello",
            "n_predict": 4,
            "response_fields": ["content", "tokens_predicted"],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(keys(&body), ["content", "tokens_predicted"], "{body}");
}

#[tokio::test]
async fn response_fields_keys_a_slashed_path_by_the_whole_path() {
    let (_, body) = post(
        "/completion",
        serde_json::json!({
            "prompt": "hello",
            "n_predict": 4,
            "response_fields": ["generation_settings/n_predict", "timings/cache_n"],
        }),
    )
    .await;
    assert_eq!(
        keys(&body),
        ["generation_settings/n_predict", "timings/cache_n"],
        "{body}"
    );
    assert_eq!(body["generation_settings/n_predict"], 4, "{body}");
}

#[tokio::test]
async fn a_wrongly_typed_response_fields_is_ignored_rather_than_refused() {
    // Upstream reads the field with a `std::vector<std::string>` default, so a
    // string or a mixed array falls back to the whole object with a 200. A 422
    // here would turn away a request llama-server serves.
    for value in [
        serde_json::json!("content"),
        serde_json::json!(["content", 5]),
        serde_json::Value::Null,
    ] {
        let (status, body) = post(
            "/completion",
            serde_json::json!({"prompt": "hello", "n_predict": 4, "response_fields": value}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(
            body.get("timings").is_some(),
            "must be the full object: {body}"
        );
    }
}

#[tokio::test]
async fn stream_options_is_accepted_and_inert_on_the_native_route() {
    // Measured on the pinned binary: `stream_options.include_usage` changes
    // nothing on `/completion`, because the native final frame always carries
    // the counts and the timing block. mlxcel now declares the field so its
    // type is validated, and answers the same body with and without it.
    let (status, with_option) = post(
        "/completion",
        serde_json::json!({
            "prompt": "hello",
            "n_predict": 4,
            "stream_options": {"include_usage": true},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{with_option}");
    let (_, without_option) = post(
        "/completion",
        serde_json::json!({"prompt": "hello", "n_predict": 4}),
    )
    .await;
    assert_eq!(keys(&with_option), keys(&without_option));
    assert_eq!(with_option["content"], without_option["content"]);
    assert_eq!(
        with_option["tokens_predicted"],
        without_option["tokens_predicted"]
    );
}

#[tokio::test]
async fn a_non_object_stream_options_is_tolerated() {
    // `"stream_options": "garbage"` answers a normal completion upstream.
    let (status, body) = post(
        "/completion",
        serde_json::json!({"prompt": "hello", "n_predict": 4, "stream_options": "garbage"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn a_non_boolean_include_usage_is_refused_with_the_field_named() {
    let (status, body) = post(
        "/completion",
        serde_json::json!({
            "prompt": "hello",
            "n_predict": 4,
            "stream_options": {"include_usage": "yes"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("include_usage"), "{message:?}");
    assert!(message.contains("boolean"), "{message:?}");
}

#[tokio::test]
async fn n_predict_minus_one_is_accepted_as_the_unbounded_spelling() {
    // b10621's hard limits are [-1, INT32_MAX] with -1 meaning "as many as the
    // context allows". Before this change serde refused it with a 422.
    let (status, body) = post(
        "/completion",
        serde_json::json!({"prompt": "hello", "n_predict": -1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.get("content").is_some(), "{body}");
}

#[tokio::test]
async fn n_predict_zero_is_accepted_as_the_prompt_only_spelling() {
    let (status, body) = post(
        "/completion",
        serde_json::json!({"prompt": "hello", "n_predict": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["generation_settings"]["n_predict"], 0, "{body}");
}

#[tokio::test]
async fn n_predict_below_the_hard_limit_is_refused_with_the_upstream_wording() {
    let (status, body) = post(
        "/completion",
        serde_json::json!({"prompt": "hello", "n_predict": -2}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("n_predict"), "{message:?}");
    assert!(
        message.contains("-1 <= value <= 2147483647"),
        "the diagnostic must state upstream's domain, got {message:?}"
    );
}
