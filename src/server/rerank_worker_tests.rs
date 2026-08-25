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

//! Rerank worker admission, timeout and panic-recovery gates, over the stub
//! reranker so no checkpoint is needed.

use std::sync::mpsc;
use std::time::Duration;

use super::RerankWorkerProvider;
use crate::rerank::stub::{STUB_BATCH_SIZE, STUB_MAX_LENGTH, stub_loaded_reranker};
use crate::rerank::{LoadedReranker, RerankItem, RerankScores, Reranker, RerankerKind};
use crate::server::rerank_model::{RerankError, RerankModelProvider};

fn provider() -> RerankWorkerProvider {
    RerankWorkerProvider::from_loader(
        "stub-rerank".to_string(),
        8,
        Duration::from_secs(30),
        || Ok(stub_loaded_reranker(RerankerKind::GenerativeText, false)),
    )
    .expect("stub worker spawns")
}

fn query() -> RerankItem {
    RerankItem::text("alpha beta")
}

#[test]
fn worker_loads_stub_and_reports_info() {
    let p = provider();
    assert_eq!(p.model_id(), "stub-rerank");
    assert_eq!(p.kind(), RerankerKind::GenerativeText);
    assert_eq!(p.max_length(), STUB_MAX_LENGTH);
    assert!(!p.supports_images());
    assert_eq!(p.info().batch_size, STUB_BATCH_SIZE);
    assert_eq!(p.info().model_type, "stub");
    assert!(p.created_at() > 0);
}

#[test]
fn worker_round_trips_scores_on_its_own_thread() {
    let p = provider();
    let scored = p
        .rerank(
            query(),
            vec![RerankItem::text("alpha beta"), RerankItem::text("gamma")],
            None,
        )
        .expect("scoring succeeds");
    assert_eq!(scored.scores.len(), 2);
    assert!(scored.scores[0] > scored.scores[1]);
    assert_eq!(scored.prompt_tokens, 4 + 3);
}

#[test]
fn worker_maps_a_family_error_to_internal() {
    let p = provider();
    let image = crate::rerank::ImageInput {
        image: image::DynamicImage::new_rgb8(4, 4),
    };
    let err = p
        .rerank(query(), vec![RerankItem::image(image)], None)
        .expect_err("the text-only stub rejects images");
    match err {
        RerankError::Internal(message) => {
            assert!(message.contains("does not accept images"), "{message}")
        }
        other => panic!("expected Internal, got {other:?}"),
    }
}

#[test]
fn worker_surfaces_loader_failure() {
    let result =
        RerankWorkerProvider::from_loader("broken".to_string(), 8, Duration::from_secs(30), || {
            Err(anyhow::anyhow!("synthetic load failure"))
        });
    let err = result.expect_err("loader failure propagates");
    assert!(err.to_string().contains("synthetic load failure"), "{err}");
}

/// A reranker whose scoring panics on a sentinel document, to prove the
/// worker survives and keeps serving.
struct PanickingReranker;

impl Reranker for PanickingReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn score(
        &self,
        _query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
    ) -> anyhow::Result<RerankScores> {
        if documents.iter().any(|d| d.text_or_empty() == "boom") {
            panic!("synthetic rerank panic");
        }
        Ok(RerankScores {
            scores: vec![0.5; documents.len()],
            prompt_tokens: documents.len(),
        })
    }

    fn max_length(&self) -> usize {
        STUB_MAX_LENGTH
    }

    fn batch_size(&self) -> usize {
        1
    }
}

fn panicking_loaded() -> LoadedReranker {
    LoadedReranker {
        reranker: Box::new(PanickingReranker),
        kind: RerankerKind::GenerativeText,
        model_type: "panicky".to_string(),
    }
}

#[test]
fn worker_survives_a_panicking_request_and_keeps_serving() {
    let p = RerankWorkerProvider::from_loader(
        "panicky".to_string(),
        8,
        Duration::from_secs(30),
        || Ok(panicking_loaded()),
    )
    .expect("worker spawns");

    let err = p
        .rerank(query(), vec![RerankItem::text("boom")], None)
        .expect_err("the sentinel document panics inside the family");
    match err {
        RerankError::Internal(message) => {
            assert!(message.contains("panic"), "{message}");
            assert!(!message.contains("no longer running"), "{message}");
        }
        other => panic!("expected Internal, got {other:?}"),
    }

    let scored = p
        .rerank(query(), vec![RerankItem::text("fine")], None)
        .expect("the worker still serves after the panic");
    assert_eq!(scored.scores.len(), 1);
}

/// A reranker that sleeps, for the timeout gate.
struct SlowReranker(Duration);

impl Reranker for SlowReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn score(
        &self,
        _query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
    ) -> anyhow::Result<RerankScores> {
        std::thread::sleep(self.0);
        Ok(RerankScores {
            scores: vec![0.25; documents.len()],
            prompt_tokens: 1,
        })
    }

    fn max_length(&self) -> usize {
        STUB_MAX_LENGTH
    }

    fn batch_size(&self) -> usize {
        1
    }
}

#[test]
fn request_timeout_returns_timeout_error() {
    let p =
        RerankWorkerProvider::from_loader("slow".to_string(), 8, Duration::from_millis(50), || {
            Ok(LoadedReranker {
                reranker: Box::new(SlowReranker(Duration::from_millis(400))),
                kind: RerankerKind::GenerativeText,
                model_type: "slow".to_string(),
            })
        })
        .expect("worker spawns");
    let err = p
        .rerank(query(), vec![RerankItem::text("alpha")], None)
        .expect_err("slower than the timeout");
    assert!(matches!(err, RerankError::Timeout), "{err:?}");
}

/// A reranker that blocks on a gate until the test releases it, so the
/// bounded queue can be filled deterministically.
struct BlockingReranker {
    gate: std::cell::Cell<Option<mpsc::Receiver<()>>>,
    started: mpsc::Sender<()>,
}

impl Reranker for BlockingReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn score(
        &self,
        _query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
    ) -> anyhow::Result<RerankScores> {
        if let Some(gate) = self.gate.take() {
            let _ = self.started.send(());
            let _ = gate.recv();
        }
        Ok(RerankScores {
            scores: vec![0.5; documents.len()],
            prompt_tokens: 1,
        })
    }

    fn max_length(&self) -> usize {
        STUB_MAX_LENGTH
    }

    fn batch_size(&self) -> usize {
        1
    }
}

#[test]
fn queue_full_returns_queuefull_error() {
    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let queue_depth = 2;
    // A short reply timeout keeps the queue-filling probes from blocking on a
    // worker that is deliberately parked; each one times out and the next
    // takes its slot until admission itself is refused.
    let worker = super::RerankWorker::spawn(
        "rerank-test-queuefull",
        queue_depth,
        Duration::from_millis(150),
        move || {
            Ok(LoadedReranker {
                reranker: Box::new(BlockingReranker {
                    gate: std::cell::Cell::new(Some(gate_rx)),
                    started: started_tx,
                }),
                kind: RerankerKind::GenerativeText,
                model_type: "blocking".to_string(),
            })
        },
    )
    .expect("worker spawns");

    std::thread::scope(|scope| {
        scope.spawn(|| {
            let _ = worker.score(query(), vec![RerankItem::text("a")], None);
        });
        started_rx.recv().expect("in-flight request started");

        let mut saw_queue_full = false;
        for _ in 0..(queue_depth + 1) {
            match worker.score(query(), vec![RerankItem::text("b")], None) {
                Err(RerankError::QueueFull) => {
                    saw_queue_full = true;
                    break;
                }
                Err(RerankError::Timeout) => continue,
                other => panic!("unexpected result while filling the queue: {other:?}"),
            }
        }
        assert!(saw_queue_full, "a full bounded queue must reject admission");
        drop(gate_tx);
    });
}
