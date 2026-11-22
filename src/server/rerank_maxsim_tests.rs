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

//! MaxSim rerank provider tests over a stub multi-vector embedder, driven
//! through the real router so `/v1/rerank` validation, dispatch, sorting and
//! both response envelopes are exercised end to end.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{
    LateInteractionDecision, MaxSimRerankProvider, late_interaction_decision,
    resolve_late_interaction_reranker,
};
use crate::embeddings::{EmbedOptions, EmbedReply, EmbeddingVector, ImageInput, maxsim};
use crate::rerank::{RerankItem, RerankerKind};
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};
use crate::server::rerank_model::{RerankError, RerankModelProvider};
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const EMBEDDER_ID: &str = "stub-colbert";

/// A 1x1 PNG as a data URI.
const TINY_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

fn query_rows() -> Vec<Vec<f32>> {
    vec![vec![1.0, 0.0], vec![0.0, 1.0]]
}

fn relevant_rows() -> Vec<Vec<f32>> {
    vec![vec![0.6, 0.8], vec![1.0, 0.0], vec![0.0, 0.5]]
}

fn unrelated_rows() -> Vec<Vec<f32>> {
    vec![vec![0.0, 0.3]]
}

fn image_rows() -> Vec<Vec<f32>> {
    vec![vec![0.5, 0.5], vec![0.9, 0.1]]
}

/// Known rows and a token count per text; `None` is a worker error.
fn table(text: &str) -> Result<(Vec<Vec<f32>>, usize), EmbeddingError> {
    match text {
        "query" => Ok((query_rows(), 2)),
        "relevant" => Ok((relevant_rows(), 3)),
        "unrelated" => Ok((unrelated_rows(), 1)),
        "no rows" => Ok((Vec::new(), 0)),
        "nan" => Ok((vec![vec![f32::NAN, 0.0]], 1)),
        "busy" => Err(EmbeddingError::QueueFull),
        "broken" => Err(EmbeddingError::Internal("forward failed".to_string())),
        other => Err(EmbeddingError::InvalidInput(format!(
            "unknown text {other}"
        ))),
    }
}

fn matrix(rows: &[Vec<f32>]) -> EmbeddingVector {
    EmbeddingVector {
        values: rows.iter().flatten().copied().collect(),
        shape: vec![rows.len(), 2],
    }
}

/// A two-wide stand-in for a multi-vector embedder.
struct StubColEmbedder {
    multi_vector: bool,
    images: bool,
    calls: Mutex<Vec<String>>,
    /// Worker round trips: one per `embed_texts` or `embed_image` call.
    dispatches: Mutex<usize>,
}

impl StubColEmbedder {
    fn new(multi_vector: bool, images: bool) -> Arc<Self> {
        Arc::new(Self {
            multi_vector,
            images,
            calls: Mutex::new(Vec::new()),
            dispatches: Mutex::new(0),
        })
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls lock").clone()
    }

    fn dispatches(&self) -> usize {
        *self.dispatches.lock().expect("dispatches lock")
    }
}

impl EmbeddingModelProvider for StubColEmbedder {
    fn embed_texts(
        &self,
        texts: Vec<String>,
        _opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        *self.dispatches.lock().expect("dispatches lock") += 1;
        let mut vectors = Vec::new();
        let mut prompt_tokens = 0;
        for text in texts {
            self.calls.lock().expect("calls lock").push(text.clone());
            let (rows, tokens) = table(&text)?;
            vectors.push(matrix(&rows));
            prompt_tokens += tokens;
        }
        Ok(EmbedReply {
            vectors,
            prompt_tokens,
        })
    }

    fn embed_tokens(
        &self,
        _token_rows: Vec<Vec<u32>>,
        _opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        unreachable!("the MaxSim provider never embeds token ids")
    }

    fn embed_image(
        &self,
        _image: ImageInput,
        _opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        *self.dispatches.lock().expect("dispatches lock") += 1;
        self.calls
            .lock()
            .expect("calls lock")
            .push("<image>".to_string());
        Ok(EmbedReply {
            vectors: vec![matrix(&image_rows())],
            prompt_tokens: 4,
        })
    }

    fn model_id(&self) -> &str {
        EMBEDDER_ID
    }

    fn created_at(&self) -> i64 {
        7
    }

    fn dim(&self) -> usize {
        2
    }

    fn multi_vector(&self) -> bool {
        self.multi_vector
    }

    fn supports_images(&self) -> bool {
        self.images
    }

    fn vocab_size(&self) -> usize {
        16
    }

    fn max_length(&self) -> usize {
        64
    }

    fn shutdown_and_wait(&self, _timeout: std::time::Duration) -> anyhow::Result<()> {
        Ok(())
    }
}

/// The router as startup builds it: the embedder installed on
/// `/v1/embeddings` and the MaxSim provider resolved from it.
fn app_for(embedder: Arc<StubColEmbedder>) -> axum::Router {
    let embedding: Arc<dyn EmbeddingModelProvider> = embedder;
    let rerank = resolve_late_interaction_reranker(None, false, Some(&embedding), None);
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
    .with_embedding_model(Some(embedding))
    .with_rerank_model(rerank);
    create_app(state)
}

async fn send(
    app: axum::Router,
    method: Method,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(path);
    if body.is_some() {
        builder = builder.header("content-type", "application/json");
    }
    let request = builder
        .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
        .expect("request builds");
    let response = app.oneshot(request).await.expect("route responds");
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
    send(app, Method::POST, "/v1/rerank", Some(body)).await
}

fn score(value: &Value) -> f32 {
    value.as_f64().expect("numeric score") as f32
}

#[tokio::test]
async fn scores_are_maxsim_of_the_same_rows_in_document_order() {
    let embedder = StubColEmbedder::new(true, false);
    let (status, body) = post(
        app_for(Arc::clone(&embedder)),
        json!({"query": "query", "documents": ["unrelated", "relevant"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], EMBEDDER_ID, "{body}");

    let relevant = maxsim(&query_rows(), &relevant_rows());
    let unrelated = maxsim(&query_rows(), &unrelated_rows());
    assert!(relevant > 1.0, "a raw MaxSim sum, not a probability");
    let results = body["results"].as_array().expect("results");
    assert_eq!(results.len(), 2, "{body}");
    // Sorted by score: the relevant document (index 1) first.
    assert_eq!(results[0]["index"], 1, "{body}");
    assert_eq!(score(&results[0]["relevance_score"]), relevant);
    assert_eq!(results[1]["index"], 0, "{body}");
    assert_eq!(score(&results[1]["relevance_score"]), unrelated);
    // Query 2 + unrelated 1 + relevant 3.
    assert_eq!(body["usage"]["prompt_tokens"], 6, "{body}");
    assert_eq!(body["usage"]["total_tokens"], 6, "{body}");
    // Query first, then documents in order, all in one worker round trip.
    assert_eq!(embedder.calls(), ["query", "unrelated", "relevant"]);
    assert_eq!(embedder.dispatches(), 1);
}

#[tokio::test]
async fn top_n_and_return_documents_are_applied_after_scoring() {
    let (status, body) = post(
        app_for(StubColEmbedder::new(true, false)),
        json!({
            "query": "query",
            "documents": ["unrelated", "relevant"],
            "top_n": 1,
            "return_documents": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let results = body["results"].as_array().expect("results");
    assert_eq!(results.len(), 1, "{body}");
    assert_eq!(results[0]["index"], 1, "{body}");
    assert_eq!(results[0]["document"], "relevant", "{body}");
}

#[tokio::test]
async fn the_tei_texts_envelope_carries_the_same_scores() {
    let (status, body) = post(
        app_for(StubColEmbedder::new(true, false)),
        json!({"query": "query", "texts": ["unrelated", "relevant"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let results = body.as_array().expect("a bare TEI array");
    assert_eq!(results.len(), 2, "{body}");
    assert_eq!(results[0]["index"], 1, "{body}");
    assert_eq!(
        score(&results[0]["score"]),
        maxsim(&query_rows(), &relevant_rows())
    );
    assert_eq!(
        score(&results[1]["score"]),
        maxsim(&query_rows(), &unrelated_rows())
    );
}

#[tokio::test]
async fn instruction_is_400_with_a_late_interaction_message() {
    let embedder = StubColEmbedder::new(true, false);
    let (status, body) = post(
        app_for(Arc::clone(&embedder)),
        json!({"query": "query", "documents": ["relevant"], "instruction": "Find pages"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(
        message.contains("only supported by the generative rerankers"),
        "{body}"
    );
    assert!(
        message.contains("late-interaction (MaxSim) reranker"),
        "{body}"
    );
    assert!(!message.contains("sequence-classifier"), "{body}");
    assert!(embedder.calls().is_empty(), "rejected before dispatch");
}

#[tokio::test]
async fn an_image_document_on_a_text_only_embedder_is_400() {
    let (status, body) = post(
        app_for(StubColEmbedder::new(true, false)),
        json!({"query": "query", "documents": [{"image_url": TINY_PNG}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("does not accept images"),
        "{body}"
    );
}

#[tokio::test]
async fn image_documents_reach_an_image_embedder() {
    let embedder = StubColEmbedder::new(true, true);
    let (status, body) = post(
        app_for(Arc::clone(&embedder)),
        json!({"query": "query", "documents": [{"image_url": TINY_PNG}, "relevant"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let results = body["results"].as_array().expect("results");
    let by_index = |index: u64| {
        results
            .iter()
            .find(|result| result["index"] == index)
            .map(|result| score(&result["relevance_score"]))
            .expect("result present")
    };
    assert_eq!(by_index(0), maxsim(&query_rows(), &image_rows()));
    assert_eq!(by_index(1), maxsim(&query_rows(), &relevant_rows()));
    assert_eq!(body["usage"]["prompt_tokens"], 2 + 4 + 3, "{body}");
    // The texts go in one batch, then each image on its own.
    assert_eq!(embedder.calls(), ["query", "relevant", "<image>"]);
    assert_eq!(embedder.dispatches(), 2);
}

#[tokio::test]
async fn embedder_errors_map_to_the_rerank_status_codes() {
    let (status, body) = post(
        app_for(StubColEmbedder::new(true, false)),
        json!({"query": "query", "documents": ["relevant", "busy"]}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");

    let (status, body) = post(
        app_for(StubColEmbedder::new(true, false)),
        json!({"query": "broken", "documents": ["relevant"]}),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("forward failed"),
        "{body}"
    );
}

#[tokio::test]
async fn props_and_models_report_one_late_interaction_model() {
    let embedder = StubColEmbedder::new(true, false);
    let (status, body) = send(app_for(Arc::clone(&embedder)), Method::GET, "/props", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["capabilities"]["reranking"]["kind"],
        "late_interaction"
    );
    assert_eq!(body["capabilities"]["reranking"]["model"], EMBEDDER_ID);

    let (status, body) = send(app_for(embedder), Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let listed = body["data"]
        .as_array()
        .expect("data")
        .iter()
        .filter(|model| model["id"] == EMBEDDER_ID)
        .count();
    assert_eq!(listed, 1, "the shared id is listed once: {body}");
}

#[test]
fn a_document_without_rows_scores_zero() {
    let provider = MaxSimRerankProvider::new(StubColEmbedder::new(true, false), None);
    let scored = provider
        .rerank(
            RerankItem::text("query"),
            vec![RerankItem::text("no rows"), RerankItem::text("relevant")],
            None,
        )
        .expect("scores");
    assert_eq!(scored.scores[0], 0.0);
    assert_eq!(scored.scores[1], maxsim(&query_rows(), &relevant_rows()));
    assert_eq!(scored.prompt_tokens, 2 + 3);
    assert_eq!(provider.kind(), RerankerKind::LateInteraction);
    assert_eq!(provider.model_id(), EMBEDDER_ID);
    assert_eq!(provider.created_at(), 7);
    assert_eq!(provider.max_length(), 64);
    assert!(!provider.supports_images());
}

#[test]
fn queue_full_maps_to_rerank_queue_full() {
    let provider = MaxSimRerankProvider::new(StubColEmbedder::new(true, false), None);
    let err = provider
        .rerank(
            RerankItem::text("busy"),
            vec![RerankItem::text("relevant")],
            None,
        )
        .expect_err("queue full");
    assert!(matches!(err, RerankError::QueueFull), "{err:?}");
}

#[test]
fn install_decision_requires_a_multi_vector_embedder_and_no_reranker() {
    let single = StubColEmbedder::new(false, false);
    let multi = StubColEmbedder::new(true, false);
    assert_eq!(
        late_interaction_decision(None, false),
        LateInteractionDecision::NotApplicable
    );
    assert_eq!(
        late_interaction_decision(Some(single.as_ref()), false),
        LateInteractionDecision::NotApplicable
    );
    assert_eq!(
        late_interaction_decision(Some(single.as_ref()), true),
        LateInteractionDecision::NotApplicable
    );
    assert_eq!(
        late_interaction_decision(Some(multi.as_ref()), false),
        LateInteractionDecision::Install
    );
    assert_eq!(
        late_interaction_decision(Some(multi.as_ref()), true),
        LateInteractionDecision::ShadowedByReranker
    );
}

#[test]
fn a_single_vector_embedder_never_gets_the_provider() {
    let single: Arc<dyn EmbeddingModelProvider> = StubColEmbedder::new(false, false);
    assert!(resolve_late_interaction_reranker(None, false, Some(&single), None).is_none());

    let multi: Arc<dyn EmbeddingModelProvider> = StubColEmbedder::new(true, false);
    let installed = resolve_late_interaction_reranker(None, false, Some(&multi), None)
        .expect("installed over a multi-vector embedder");
    assert_eq!(installed.kind(), RerankerKind::LateInteraction);

    // A configured reranker wins, even one that failed to load.
    assert!(resolve_late_interaction_reranker(None, true, Some(&multi), None).is_none());
    let explicit = resolve_late_interaction_reranker(
        Some(Arc::new(MaxSimRerankProvider::new(
            StubColEmbedder::new(true, true),
            None,
        ))),
        false,
        Some(&multi),
        None,
    )
    .expect("the explicit reranker is kept");
    assert!(
        explicit.supports_images(),
        "the explicit provider, not a new one"
    );
}

#[tokio::test]
async fn many_text_documents_take_one_worker_round_trip() {
    // One dispatch per document would hold the worker for one reply timeout
    // per document and could hit QueueFull mid-request; the whole text side
    // of a request is one batch, as on `/v1/embeddings`.
    let embedder = StubColEmbedder::new(true, false);
    let documents: Vec<&str> = (0..500)
        .map(|i| if i % 2 == 0 { "relevant" } else { "unrelated" })
        .collect();
    let (status, body) = post(
        app_for(Arc::clone(&embedder)),
        json!({"query": "query", "documents": documents}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["results"].as_array().expect("results").len(), 500);
    assert_eq!(embedder.dispatches(), 1);
    assert_eq!(embedder.calls().len(), 501);
}

#[tokio::test]
async fn a_non_finite_maxsim_score_is_a_generic_500() {
    let (status, body) = post(
        app_for(StubColEmbedder::new(true, false)),
        json!({"query": "query", "documents": ["relevant", "nan"]}),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert_eq!(
        body["error"]["message"], "rerank inference returned an invalid numeric result",
        "{body}"
    );
}

#[test]
fn a_short_text_reply_is_an_internal_error_not_a_misaligned_score() {
    struct ShortReply(Arc<StubColEmbedder>);
    impl EmbeddingModelProvider for ShortReply {
        fn embed_texts(
            &self,
            texts: Vec<String>,
            opts: EmbedOptions,
        ) -> Result<EmbedReply, EmbeddingError> {
            let mut reply = self.0.embed_texts(texts, opts)?;
            reply.vectors.pop();
            Ok(reply)
        }
        fn embed_tokens(
            &self,
            token_rows: Vec<Vec<u32>>,
            opts: EmbedOptions,
        ) -> Result<EmbedReply, EmbeddingError> {
            self.0.embed_tokens(token_rows, opts)
        }
        fn embed_image(
            &self,
            image: ImageInput,
            opts: EmbedOptions,
        ) -> Result<EmbedReply, EmbeddingError> {
            self.0.embed_image(image, opts)
        }
        fn model_id(&self) -> &str {
            self.0.model_id()
        }
        fn created_at(&self) -> i64 {
            self.0.created_at()
        }
        fn dim(&self) -> usize {
            self.0.dim()
        }
        fn multi_vector(&self) -> bool {
            true
        }
        fn vocab_size(&self) -> usize {
            self.0.vocab_size()
        }
        fn max_length(&self) -> usize {
            self.0.max_length()
        }
        fn shutdown_and_wait(&self, timeout: std::time::Duration) -> anyhow::Result<()> {
            self.0.shutdown_and_wait(timeout)
        }
    }
    let provider = MaxSimRerankProvider::new(
        Arc::new(ShortReply(StubColEmbedder::new(true, false))),
        None,
    );
    let err = provider
        .rerank(
            RerankItem::text("query"),
            vec![RerankItem::text("relevant"), RerankItem::text("unrelated")],
            None,
        )
        .expect_err("a short reply");
    assert!(matches!(err, RerankError::Internal(_)), "{err:?}");
}
