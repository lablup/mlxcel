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

//! `/v1/decisions` route tests over the stub decision model, driven through
//! the real router (JSON body, validation, media gate, worker dispatch,
//! response shape).

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::NO_DECISION_MODEL_MESSAGE;
use crate::decision::stub::stub_loaded_decision_model;
use crate::server::decision_model::{
    DecisionMedia, DecisionModelProvider, DecisionProviderError, DecisionResult, MediaKind,
    Question, QuestionKind,
};
use crate::server::decision_worker::DecisionWorkerProvider;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const STUB_MODEL_ID: &str = "stub-decider";

/// A 1x1 PNG as a data URI.
const TINY_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// A provider that fails every request with one fixed error.
struct FailingProvider(fn() -> DecisionProviderError);

impl DecisionModelProvider for FailingProvider {
    fn predict(
        &self,
        _state: Option<Value>,
        _questions: Vec<(String, Question)>,
        _media: DecisionMedia,
        _temp_files: Vec<crate::multimodal::video::PrivateTempDir>,
    ) -> Result<DecisionResult, DecisionProviderError> {
        Err((self.0)())
    }

    fn model_id(&self) -> &str {
        "failing-decider"
    }

    fn created_at(&self) -> i64 {
        0
    }

    fn decision_types(&self) -> &[QuestionKind] {
        &[QuestionKind::Choice]
    }

    fn decision_media(&self) -> &[MediaKind] {
        &[]
    }

    fn shutdown_and_wait(&self, _timeout: Duration) -> anyhow::Result<()> {
        Ok(())
    }
}

/// A provider that declares audio and video but must never be reached by a
/// request that fails media acquisition.
struct MediaProvider;

impl DecisionModelProvider for MediaProvider {
    fn predict(
        &self,
        _state: Option<Value>,
        _questions: Vec<(String, Question)>,
        _media: DecisionMedia,
        _temp_files: Vec<crate::multimodal::video::PrivateTempDir>,
    ) -> Result<DecisionResult, DecisionProviderError> {
        Err(DecisionProviderError::Internal(
            "predict must not be reached".into(),
        ))
    }

    fn model_id(&self) -> &str {
        "media-decider"
    }

    fn created_at(&self) -> i64 {
        0
    }

    fn decision_types(&self) -> &[QuestionKind] {
        &[QuestionKind::Choice]
    }

    fn decision_media(&self) -> &[MediaKind] {
        &[MediaKind::Audio, MediaKind::Video]
    }

    fn shutdown_and_wait(&self, _timeout: Duration) -> anyhow::Result<()> {
        Ok(())
    }
}

fn stub_provider() -> Arc<dyn DecisionModelProvider> {
    Arc::new(
        DecisionWorkerProvider::from_loader(
            STUB_MODEL_ID.to_string(),
            8,
            Duration::from_secs(30),
            || Ok(stub_loaded_decision_model()),
        )
        .expect("stub decision worker spawns"),
    )
}

fn state_with(model_provider: ModelProvider) -> AppState {
    let model_provider = Arc::new(model_provider);
    let batch_metrics = model_provider.batch_metrics().clone();
    AppState::new(
        model_provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
}

fn app_with(provider: Option<Arc<dyn DecisionModelProvider>>) -> axum::Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let state = state_with(ModelProvider::recording_for_route_tests(options_tx))
        .with_decision_model(provider);
    create_app(state)
}

async fn send(
    app: axum::Router,
    method: Method,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut request = Request::builder().method(method).uri(path);
    let body = match body {
        Some(body) => {
            request = request.header("content-type", "application/json");
            Body::from(body.to_string())
        }
        None => Body::empty(),
    };
    let response = app
        .oneshot(request.body(body).expect("request builds"))
        .await
        .expect("route responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body")
    };
    (status, value)
}

async fn post(app: axum::Router, body: Value) -> (StatusCode, Value) {
    send(app, Method::POST, "/v1/decisions", Some(body)).await
}

fn ticket() -> Value {
    json!({
        "state": "The billing page charged me twice",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Route the ticket",
                "criteria": ["technical", "billing", "sales"]
            },
            "urgent": {"type": "noul"},
            "severity": {"type": "score", "criteria": ["low", "medium", "high"]}
        }
    })
}

#[tokio::test]
async fn a_valid_request_answers_every_question_in_order() {
    let (status, body) = post(app_with(Some(stub_provider())), ticket()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], STUB_MODEL_ID);
    assert_eq!(
        body["usage"],
        json!({"input_tokens": 6, "output_tokens": 0})
    );
    let names: Vec<&String> = body["answers"].as_object().unwrap().keys().collect();
    assert_eq!(names, ["topic", "urgent", "severity"]);

    let topic = &body["answers"]["topic"];
    assert_eq!(topic["type"], "choice");
    assert_eq!(topic["value"], "billing");
    let labels: Vec<&String> = topic["probabilities"].as_object().unwrap().keys().collect();
    assert_eq!(labels, ["technical", "billing", "sales"]);
    assert_eq!(topic["metadata"]["confidence"], 0.6);

    let urgent = &body["answers"]["urgent"];
    assert_eq!(urgent["type"], "bool", "noul answers as bool: {body}");
    assert_eq!(urgent["value"], false);
    assert_eq!(urgent["probability"], 0.2);

    let severity = &body["answers"]["severity"];
    assert_eq!(severity["type"], "score");
    let levels: Vec<&String> = severity["probabilities"]
        .as_object()
        .unwrap()
        .keys()
        .collect();
    assert_eq!(levels, ["0", "1", "2"]);
    assert_eq!(severity["metadata"]["legend"]["0"], "low");
}

#[tokio::test]
async fn every_validation_failure_is_400_with_the_contract_message() {
    let q = |questions: Value| json!({"state": "hi", "questions": questions});
    let cases = [
        (
            json!({"questions": {"q": {}}}),
            "state is required unless images or audio are given",
        ),
        (
            json!({"state": null, "questions": {"q": {}}}),
            "state is required unless images or audio are given",
        ),
        (
            json!({"state": "hi", "audio": "UklGRg==", "questions": {"q": {}}}),
            "This model does not support audio input",
        ),
        (
            json!({"state": "hi", "videos": ["clip.mp4"], "questions": {"q": {}}}),
            "This model does not support video input",
        ),
        (q(json!({})), "At least one named question is required"),
        (
            q(json!({"q": "x"})),
            "Questions must map names to question dictionaries",
        ),
        (
            q(json!({"q": {"type": "multi_label", "criteria": ["a"]}})),
            "This model does not support 'multi_label' decisions",
        ),
        (
            q(json!({"q": {"type": "ranking"}})),
            "This model does not support 'ranking' decisions",
        ),
        (
            q(json!({"q": {"criteria": ["a"]}})),
            "choice needs at least 2 criteria",
        ),
        (
            q(json!({"q": {"criteria": ["a", "a"]}})),
            "Decision labels must be unique",
        ),
        (
            q(json!({"q": {"criteria": ["a", ""]}})),
            "Decision labels must be nonempty strings",
        ),
        (
            q(json!({"q": {"type": "score", "criteria": {"a": 1, "b": 2}}})),
            "score criteria must be an ordered list",
        ),
        (
            q(json!({"q": {"type": "bool", "criteria": ["no", "yes"]}})),
            "bool criteria must map false and true to descriptions",
        ),
    ];
    let provider = stub_provider();
    for (request, expected) in cases {
        let (status, body) = post(app_with(Some(provider.clone())), request.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{request}: {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert_eq!(body["error"]["message"], expected, "{request}");
    }
}

#[tokio::test]
async fn threshold_is_checked_for_a_multi_label_model() {
    // The stub does not declare multi_label, so rule 7 is exercised through
    // the shared validator in `decision::mod_tests`; here a non-object body
    // and a wrongly typed media field are still 400s.
    let provider = stub_provider();
    let (status, body) = post(app_with(Some(provider.clone())), json!([1, 2])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = post(
        app_with(Some(provider)),
        json!({"state": "hi", "images": "a.png", "questions": {"q": {}}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["message"],
        "`images` must be a list of strings"
    );
}

#[tokio::test]
async fn images_reach_predict_and_stand_in_for_state() {
    let request = json!({
        "images": [TINY_PNG],
        "questions": {"screenshot_ok": {"type": "bool"}}
    });
    let (status, body) = post(app_with(Some(stub_provider())), request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let answer = &body["answers"]["screenshot_ok"];
    assert_eq!(answer["metadata"]["images"], 1, "{body}");
    assert_eq!(answer["value"], true);
}

#[tokio::test]
async fn an_undecodable_image_is_400() {
    let request = json!({
        "state": "hi",
        "images": ["data:image/png;base64,AAAA"],
        "questions": {"q": {"type": "bool"}}
    });
    let (status, body) = post(app_with(Some(stub_provider())), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("images[0]"), "{body}");
}

#[tokio::test]
async fn no_decision_model_is_501_naming_the_flag() {
    for path in ["/v1/decisions", "/decisions"] {
        let (status, body) = send(app_with(None), Method::POST, path, Some(ticket())).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {body}");
        assert_eq!(body["error"]["message"], NO_DECISION_MODEL_MESSAGE);
        assert!(NO_DECISION_MODEL_MESSAGE.contains("--decision-model"));
    }
}

#[tokio::test]
async fn a_model_naming_another_checkpoint_is_400() {
    let mut request = ticket();
    request["model"] = json!("some-other-model");
    let (status, body) = post(app_with(Some(stub_provider())), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("is not the served decision model"),
        "{body}"
    );
    let mut request = ticket();
    request["model"] = json!(STUB_MODEL_ID);
    let (status, _) = post(app_with(Some(stub_provider())), request).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn provider_failures_map_to_503_504_and_a_generic_500() {
    let cases: [(fn() -> DecisionProviderError, StatusCode); 4] = [
        (
            || DecisionProviderError::QueueFull,
            StatusCode::SERVICE_UNAVAILABLE,
        ),
        (
            || DecisionProviderError::Timeout,
            StatusCode::GATEWAY_TIMEOUT,
        ),
        (
            || DecisionProviderError::InvalidInput("Decider takes 2 to 255 options".into()),
            StatusCode::BAD_REQUEST,
        ),
        (
            || DecisionProviderError::Internal("secret worker detail".into()),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
    ];
    let request = json!({"state": "hi", "questions": {"q": {"criteria": ["a", "b"]}}});
    for (error, expected) in cases {
        let provider: Arc<dyn DecisionModelProvider> = Arc::new(FailingProvider(error));
        let (status, body) = post(app_with(Some(provider)), request.clone()).await;
        assert_eq!(status, expected, "{body}");
        if expected == StatusCode::INTERNAL_SERVER_ERROR {
            assert_eq!(body["error"]["message"], "Decision prediction failed");
        }
        if expected == StatusCode::BAD_REQUEST {
            assert_eq!(body["error"]["message"], "Decider takes 2 to 255 options");
        }
    }
}

#[tokio::test]
async fn the_decision_model_is_listed_in_v1_models() {
    let (status, body) = send(
        app_with(Some(stub_provider())),
        Method::GET,
        "/v1/models",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entry = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == STUB_MODEL_ID)
        .unwrap_or_else(|| panic!("decision model listed: {body}"));
    assert_eq!(entry["capabilities"], json!(["decision"]));
}

#[tokio::test]
async fn a_decision_only_server_names_the_route_and_flag_on_chat() {
    let state = state_with(ModelProvider::chat_unavailable_for_route_tests())
        .with_decision_model(Some(stub_provider()));
    let (status, body) = send(
        create_app(state),
        Method::POST,
        "/v1/chat/completions",
        Some(json!({"model": "route-test-model", "messages": [{"role": "user", "content": "hi"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(message.contains("/v1/decisions"), "{body}");
    assert!(message.contains("--decision-model"), "{body}");
}

fn wav_data_uri(seconds: usize, rate: u32) -> String {
    use base64::Engine;
    let samples = vec![0.1_f32; seconds * rate as usize];
    let wav = crate::audio::encode_wav_pcm16(&samples, rate, 1);
    format!(
        "data:audio/wav;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(wav)
    )
}

#[tokio::test]
async fn audio_over_the_decoded_duration_cap_is_a_400() {
    let provider: Arc<dyn DecisionModelProvider> = Arc::new(MediaProvider);
    let request = json!({
        "audio": wav_data_uri(crate::decision::media::MAX_DECISION_AUDIO_SECONDS + 1, 8_000),
        "questions": {"q": {"criteria": ["a", "b"]}},
    });
    let (status, body) = post(app_with(Some(provider)), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(
        body["error"]["message"],
        "audio: audio is 301 seconds long; decision audio is limited to 300 seconds"
    );
}

#[tokio::test]
async fn audio_within_the_cap_reaches_the_model() {
    let provider: Arc<dyn DecisionModelProvider> = Arc::new(MediaProvider);
    let request = json!({
        "audio": wav_data_uri(1, 16_000),
        "questions": {"q": {"criteria": ["a", "b"]}},
    });
    let (status, body) = post(app_with(Some(provider)), request).await;
    // The sentinel provider fails in predict, so a 500 proves decoding passed.
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert_eq!(body["error"]["message"], "Decision prediction failed");
}

#[test]
fn videos_need_ffmpeg_before_any_download() {
    use super::{VIDEO_NEEDS_FFMPEG_MESSAGE, check_video_support};
    let err = check_video_support(1, false).expect_err("no ffmpeg");
    assert_eq!(err.status, StatusCode::BAD_REQUEST);
    assert_eq!(err.error.message, VIDEO_NEEDS_FFMPEG_MESSAGE);
    assert!(check_video_support(1, true).is_ok());
    assert!(check_video_support(0, false).is_ok(), "no videos, no probe");
}

#[test]
fn the_video_count_cap_comes_before_the_ffmpeg_probe() {
    use super::{MAX_DECISION_VIDEOS, check_video_support};
    assert!(check_video_support(MAX_DECISION_VIDEOS, true).is_ok());
    let err = check_video_support(MAX_DECISION_VIDEOS + 1, false).expect_err("over the cap");
    assert_eq!(err.status, StatusCode::BAD_REQUEST);
    assert!(
        err.error.message.contains("at most 8"),
        "{}",
        err.error.message
    );
}

#[tokio::test]
async fn nine_videos_are_a_400_before_any_fetch() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    // Every URL points at this listener, so a fetch shows up as a connection.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("address");
    let connections = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&connections);
    let accept = tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            counter.fetch_add(1, Ordering::SeqCst);
            drop(socket);
        }
    });
    let videos: Vec<String> = (0..super::MAX_DECISION_VIDEOS + 1)
        .map(|index| format!("http://{address}/clip-{index}.mp4"))
        .collect();
    let provider: Arc<dyn DecisionModelProvider> = Arc::new(MediaProvider);
    let request =
        json!({"state": "hi", "videos": videos, "questions": {"q": {"criteria": ["a", "b"]}}});
    let (status, body) = post(app_with(Some(provider)), request).await;
    accept.abort();
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["message"],
        "`videos` holds 9 entries; at most 8 are allowed per request"
    );
    assert_eq!(connections.load(Ordering::SeqCst), 0, "a video was fetched");
}
