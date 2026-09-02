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

use crate::server::model_provider::{ScriptedStreamHandle, SpeculativeStats};
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn app() -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    create_app(state_for(Arc::new(
        ModelProvider::recording_for_route_tests(options_tx),
    )))
}

fn state_for(provider: Arc<ModelProvider>) -> AppState {
    let batch_metrics = provider.batch_metrics().clone();
    AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("native-route-test-model"),
        batch_metrics,
    )
}

fn json_request(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds")
}

async fn post(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    post_to(app(), path, body).await
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
        // `n_cmpl` above 1 is the one field still refused, and at one slot the
        // refusal is byte-equivalent to upstream's own (#1477).
        ("n_cmpl", serde_json::json!({"prompt": "hi", "n_cmpl": 2})),
        ("n_cmpl", serde_json::json!({"prompt": "hi", "n": 2})),
        // Value-domain refusals, upstream's own hard limits: n_indent is
        // 0..=INT32_MAX and t_max_predict_ms is -1..=INT64_MAX, both refused
        // rather than clamped by the pinned binary.
        (
            "n_indent",
            serde_json::json!({"prompt": "hi", "n_indent": -1}),
        ),
        (
            "t_max_predict_ms",
            serde_json::json!({"prompt": "hi", "t_max_predict_ms": -5}),
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

/// b10621 `return_tokens` (#1477): the top-level `tokens` array carries the
/// raw generated ids when the field is set and is empty otherwise.
///
/// Measured on the pinned binary: `The capital of France is` answered with 8
/// tokens returns `tokens: [12095, 13, 576, 6722, 315, 15344, 374, 21718]`
/// against `tokens_predicted: 8`, and the same request without the field
/// returns `tokens: []`. The ids come from the scheduler rather than from
/// re-tokenizing the answer, because a string stop sequence excludes its
/// matched text from `content` while its token still counts: with
/// `stop: ["Paris"]` the binary answers `content: " "` and `tokens: [12095]`.
#[tokio::test]
async fn return_tokens_projects_the_generated_ids_and_is_empty_without_it() {
    for path in ["/completion", "/completions"] {
        let (status, with) = post(
            path,
            serde_json::json!({"prompt": "hi", "n_predict": 2, "return_tokens": true}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{with}");
        assert_eq!(
            with["tokens"].as_array().map(Vec::len),
            Some(2),
            "{path} must return the generated ids under return_tokens: {with}"
        );

        for body in [
            serde_json::json!({"prompt": "hi", "n_predict": 2, "return_tokens": false}),
            serde_json::json!({"prompt": "hi", "n_predict": 2}),
        ] {
            let (status, without) = post(path, body).await;
            assert_eq!(status, StatusCode::OK, "{without}");
            assert_eq!(
                without["tokens"].as_array().map(Vec::len),
                Some(0),
                "{path} must return an empty array without return_tokens: {without}"
            );
        }
    }
}

/// b10621 `return_progress` (#1477) is accepted on a non-streaming request and
/// adds nothing there, matching upstream's `stream && return_progress` gate:
/// the pinned binary answering `{"return_progress": true}` without `stream`
/// returns the same key set as the request without it, with no
/// `prompt_progress`.
#[tokio::test]
async fn return_progress_is_accepted_and_adds_nothing_to_a_non_streaming_response() {
    let (status, body) = post(
        "/completion",
        serde_json::json!({"prompt": "hi", "n_predict": 2, "return_progress": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body.get("prompt_progress").is_none(),
        "prompt_progress belongs to streaming frames only: {body}"
    );
}

/// The two b10621 generation bounds are accepted at every value in their
/// domain and reach the generation options (#1477).
///
/// `n_indent: 0` and `t_max_predict_ms: 0` are upstream's disabled sentinels,
/// so they must be inert rather than refused; `-1` is the second disabled
/// spelling of `t_max_predict_ms`.
#[tokio::test]
async fn the_generation_bounds_are_accepted_across_their_value_domain() {
    for body in [
        serde_json::json!({"prompt": "hi", "n_indent": 0}),
        serde_json::json!({"prompt": "hi", "n_indent": 4}),
        serde_json::json!({"prompt": "hi", "t_max_predict_ms": -1}),
        serde_json::json!({"prompt": "hi", "t_max_predict_ms": 0}),
        serde_json::json!({"prompt": "hi", "t_max_predict_ms": 500}),
        serde_json::json!({"prompt": "hi", "n_indent": 4, "t_max_predict_ms": 500}),
    ] {
        let (status, response) = post("/completion", body.clone()).await;
        assert_eq!(status, StatusCode::OK, "{body} -> {response}");
    }
}

/// `generation_settings` reports every b10621 `task_params` key mlxcel
/// resolves and acts on (#1477).
///
/// The pinned binary's block has 49 keys. mlxcel reports 47 of them; the two
/// it omits are `backend_sampling`, whose entry records that mlxcel's sampler
/// IS the backend graph and has no CPU chain to switch to, and
/// `speculative.types`, which names upstream's draft-model type. Both
/// omissions are the omit-rather-than-invent policy `GET /props` records: a
/// key reported with an invented value would tell an operator a setting
/// steers generation when it steers nothing.
#[tokio::test]
async fn generation_settings_reports_the_b10621_task_params_keys_mlxcel_resolves() {
    let (status, body) = post("/completion", native_prompt()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let settings = body["generation_settings"]
        .as_object()
        .expect("generation_settings is an object");

    // The 49-key block captured from the pinned binary.
    let upstream = [
        "adaptive_decay",
        "adaptive_target",
        "backend_sampling",
        "chat_format",
        "dry_allowed_length",
        "dry_base",
        "dry_multiplier",
        "dry_penalty_last_n",
        "dry_sequence_breakers",
        "dynatemp_exponent",
        "dynatemp_range",
        "frequency_penalty",
        "generation_prompt",
        "grammar",
        "grammar_lazy",
        "grammar_triggers",
        "ignore_eos",
        "logit_bias",
        "lora",
        "max_tokens",
        "min_keep",
        "min_p",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "n_discard",
        "n_keep",
        "n_predict",
        "n_probs",
        "post_sampling_probs",
        "presence_penalty",
        "preserved_tokens",
        "reasoning_format",
        "reasoning_in_content",
        "repeat_last_n",
        "repeat_penalty",
        "samplers",
        "seed",
        "speculative.types",
        "stop",
        "stream",
        "temperature",
        "timings_per_token",
        "top_k",
        "top_n_sigma",
        "top_p",
        "typical_p",
        "xtc_probability",
        "xtc_threshold",
    ];
    let omitted = ["backend_sampling", "speculative.types"];

    for key in upstream {
        let present = settings.contains_key(key);
        if omitted.contains(&key) {
            assert!(!present, "{key} has no mlxcel analogue and must be omitted");
        } else {
            assert!(present, "missing {key} in {settings:?}");
        }
    }
    // No key outside upstream's block: an extension here would be read as a
    // b10621 setting by a client that trusts the name.
    for key in settings.keys() {
        assert!(
            upstream.contains(&key.as_str()),
            "{key} is not a b10621 task_params key"
        );
    }
    assert_eq!(settings.len(), upstream.len() - omitted.len());

    // The shapes the pinned binary uses for the keys #1477 added.
    assert_eq!(
        settings["samplers"].as_array().map(|v| v.len()),
        Some(9),
        "the fixed b10621 default chain has nine stages: {settings:?}"
    );
    assert!(settings["logit_bias"].is_array());
    assert!(settings["lora"].is_array());
    assert_eq!(settings["chat_format"], "Content-only");
    assert_eq!(settings["generation_prompt"], "");
    assert_eq!(settings["reasoning_format"], "deepseek");
    assert_eq!(settings["reasoning_in_content"], false);
    assert_eq!(settings["timings_per_token"], false);
}

// ---------------------------------------------------------------------------
// Speculative acceptance on the timings block (#1314)
// ---------------------------------------------------------------------------

/// The DFlash acceptance a drafted request reports in these tests: nine rounds
/// proposing 72 tokens of which 55 were accepted.
fn dflash_stats() -> SpeculativeStats {
    SpeculativeStats::from_counts(mlxcel_core::drafter::DrafterKind::Dflash, 9, 72, 55)
        .expect("nine rounds proposing 72 tokens is a speculative run")
}

/// The same app as [`app`], served by a provider that answers as a
/// DFlash-drafted request.
fn speculative_app() -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    create_app(state_for(Arc::new(
        ModelProvider::recording_for_route_tests_with_speculative(options_tx, dflash_stats()),
    )))
}

/// The same app again, but streaming step by step from the returned handle, so
/// a test can drive real content frames between the opening and closing ones.
fn scripted_speculative_app() -> (Router, ScriptedStreamHandle) {
    let (options_tx, _options_rx) = mpsc::channel();
    let (provider, handle) = ModelProvider::scripted_streaming_for_route_tests_with_speculative(
        options_tx,
        dflash_stats(),
    );
    (create_app(state_for(Arc::new(provider))), handle)
}

async fn post_to(
    app: Router,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let response = app
        .oneshot(json_request(path, body))
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, parsed)
}

/// The SSE frames of a streaming response, as parsed `data:` payloads with the
/// `[DONE]` sentinel dropped.
async fn stream_frames(app: Router, path: &str, body: serde_json::Value) -> Vec<serde_json::Value> {
    let response = app
        .oneshot(json_request(path, body))
        .await
        .expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        axum::body::to_bytes(response.into_body(), 1024 * 1024),
    )
    .await
    .expect("stream ends")
    .expect("body collects");
    String::from_utf8(bytes.to_vec())
        .expect("utf-8 stream")
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| l.trim() != "[DONE]")
        .map(|l| serde_json::from_str(l).expect("frame parses"))
        .collect()
}

#[tokio::test]
async fn the_native_timings_omit_the_draft_keys_without_a_drafter() {
    // A server started without `--draft-model` answers exactly the body it
    // answered before #1314. The keys are absent, not zero: a zero `draft_n`
    // would say a drafter ran and proposed nothing.
    let (status, body) = post("/completion", native_prompt()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let timings = body["timings"].as_object().expect("timings is an object");
    for key in ["draft_n", "draft_n_accepted", "draft_rounds", "draft_kind"] {
        assert!(!timings.contains_key(key), "{key} must be absent: {body}");
    }
}

#[tokio::test]
async fn the_native_timings_report_the_draft_counters_of_a_drafted_request() {
    // `draft_n` / `draft_n_accepted` are b10621's own optional pair, so a
    // client already reading llama-server timings reads these unchanged;
    // `draft_rounds` and `draft_kind` are the mlxcel extension beside them.
    for path in ["/completion", "/completions"] {
        let (status, body) = post_to(speculative_app(), path, native_prompt()).await;
        assert_eq!(status, StatusCode::OK, "{path}: {body}");
        assert_eq!(body["timings"]["draft_n"], 72, "{path}: {body}");
        assert_eq!(body["timings"]["draft_n_accepted"], 55, "{path}: {body}");
        assert_eq!(body["timings"]["draft_rounds"], 9, "{path}: {body}");
        assert_eq!(body["timings"]["draft_kind"], "dflash", "{path}: {body}");
    }
}

#[tokio::test]
async fn a_chat_completion_carries_no_timings_object_without_a_drafter() {
    // The OpenAI wire shape of a non-speculative deployment is unchanged: no
    // `timings` key at all, rather than one reporting zeros.
    let (status, body) = post(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "native-route-test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.get("timings").is_none(), "{body}");
}

#[tokio::test]
async fn a_chat_completion_carries_the_timings_object_of_a_drafted_request() {
    let (status, body) = post_to(
        speculative_app(),
        "/v1/chat/completions",
        serde_json::json!({
            "model": "native-route-test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["timings"]["draft_n"], 72, "{body}");
    assert_eq!(body["timings"]["draft_n_accepted"], 55, "{body}");
    assert_eq!(body["timings"]["draft_rounds"], 9, "{body}");
    assert_eq!(body["timings"]["draft_kind"], "dflash", "{body}");
    // The whole b10621 block, not the draft half alone: a client that probes
    // for `timings` and reads `predicted_per_second` off it has to find the
    // key it expects. Same object the native route answers with.
    assert_eq!(
        keys(&body["timings"]),
        [
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second",
            "draft_n",
            "draft_n_accepted",
            "draft_rounds",
            "draft_kind",
        ],
        "{body}"
    );
}

#[tokio::test]
async fn the_streaming_chat_finish_chunk_carries_the_timings_and_the_others_do_not() {
    // The finish chunk is where the totals are final. A mid-stream chunk that
    // carried a running `draft_n` would read as a total and be wrong, so only
    // the frame that already carries `finish_reason` carries the block. Driven
    // through the scripted provider so the negative half of the assertion runs
    // against a stream that really has content frames.
    let (app, handle) = scripted_speculative_app();
    let frames = tokio::spawn(stream_frames(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "native-route-test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "stream": true,
        }),
    ));
    handle.token("Hello");
    handle.token(" world");
    handle.finish();
    let frames = frames.await.expect("stream task joins");

    let mut content_frames = 0;
    let mut saw_finish = false;
    for chunk in &frames {
        if chunk["choices"][0]["finish_reason"].is_string() {
            saw_finish = true;
            assert_eq!(chunk["timings"]["draft_n"], 72, "{chunk}");
            assert_eq!(chunk["timings"]["draft_n_accepted"], 55, "{chunk}");
            assert_eq!(chunk["timings"]["draft_rounds"], 9, "{chunk}");
            assert_eq!(chunk["timings"]["draft_kind"], "dflash", "{chunk}");
        } else {
            if chunk["choices"][0]["delta"]["content"].is_string() {
                content_frames += 1;
            }
            assert!(
                chunk.get("timings").is_none(),
                "only the finish chunk carries the totals: {chunk}"
            );
        }
    }
    assert!(saw_finish, "the stream must end with a finish chunk");
    assert!(
        content_frames >= 2,
        "the negative half must run against real content frames, saw {content_frames}"
    );
}

#[tokio::test]
async fn the_native_stream_carries_the_draft_keys_on_the_final_frame_only() {
    // `timings_per_token` puts a timing block on every partial frame. Those
    // frames must stay at the nine b10621 keys even for a drafted request: the
    // acceptance counters are the run's totals and only exist once the round
    // loop has finished. The final frame is the one that carries all thirteen.
    let (app, handle) = scripted_speculative_app();
    let frames = tokio::spawn(stream_frames(
        app,
        "/completion",
        serde_json::json!({
            "prompt": "hello",
            "n_predict": 4,
            "stream": true,
            "timings_per_token": true,
        }),
    ));
    handle.token("Hello");
    handle.token(" world");
    handle.finish();
    let frames = frames.await.expect("stream task joins");

    let (final_frame, partials) = frames.split_last().expect("the stream has frames");
    assert_eq!(final_frame["stop"], true, "{final_frame}");
    assert_eq!(final_frame["timings"]["draft_n"], 72, "{final_frame}");
    assert_eq!(
        final_frame["timings"]["draft_n_accepted"], 55,
        "{final_frame}"
    );
    assert_eq!(final_frame["timings"]["draft_rounds"], 9, "{final_frame}");
    assert_eq!(
        final_frame["timings"]["draft_kind"], "dflash",
        "{final_frame}"
    );

    let mut timed_partials = 0;
    for frame in partials {
        assert_eq!(frame["stop"], false, "{frame}");
        let timings = frame["timings"]
            .as_object()
            .expect("timings_per_token puts a block on every partial frame");
        timed_partials += 1;
        for key in ["draft_n", "draft_n_accepted", "draft_rounds", "draft_kind"] {
            assert!(
                !timings.contains_key(key),
                "{key} must be absent from a partial frame: {frame}"
            );
        }
        assert_eq!(timings.len(), 9, "{frame}");
    }
    assert!(
        timed_partials >= 2,
        "the assertion must run against real partial frames, saw {timed_partials}"
    );
}
