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

//! Decision worker admission, timeout and load-failure behavior.

use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::decision::stub::stub_loaded_decision_model;
use crate::decision::{
    Answer, DecisionError, DecisionKind, DecisionModel, DecisionUsage, MediaPresence,
    validate_questions,
};

/// A model whose `predict` blocks until the test releases it.
struct GatedModel {
    gate: Arc<Mutex<mpsc::Receiver<()>>>,
}

impl DecisionModel for GatedModel {
    fn decision_types(&self) -> &[QuestionKind] {
        &[QuestionKind::Bool]
    }

    fn decision_media(&self) -> &[MediaKind] {
        &[]
    }

    fn predict(
        &mut self,
        _state: Option<&Value>,
        questions: &[(String, Question)],
        _media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionError> {
        let _ = self
            .gate
            .lock()
            .expect("gate lock")
            .recv_timeout(Duration::from_secs(10));
        Ok(DecisionResult {
            answers: questions
                .iter()
                .map(|(name, _)| (name.clone(), Answer::bool(0.5, Default::default())))
                .collect(),
            usage: DecisionUsage::default(),
        })
    }
}

fn gated_provider(
    queue_depth: usize,
    timeout: Duration,
) -> (DecisionWorkerProvider, mpsc::Sender<()>) {
    let (release, gate) = mpsc::channel();
    let gate = Arc::new(Mutex::new(gate));
    let provider =
        DecisionWorkerProvider::from_loader("gated".into(), queue_depth, timeout, move || {
            Ok(LoadedDecisionModel {
                model: Box::new(GatedModel { gate }),
                kind: DecisionKind::Stub,
                model_type: "gated".into(),
            })
        })
        .expect("worker spawns");
    (provider, release)
}

fn bool_question() -> Vec<(String, Question)> {
    validate_questions(
        &[QuestionKind::Bool],
        &[],
        Some(&json!("x")),
        &json!({"q": {"type": "bool"}}),
        MediaPresence::default(),
    )
    .expect("valid")
}

#[test]
fn the_stub_worker_reports_its_declarations_and_answers() {
    let provider =
        DecisionWorkerProvider::from_loader("stub".into(), 4, Duration::from_secs(30), || {
            Ok(stub_loaded_decision_model())
        })
        .expect("worker spawns");
    assert_eq!(provider.decision_media(), [MediaKind::Image]);
    assert!(provider.decision_types().contains(&QuestionKind::Score));
    let result = provider
        .predict(
            Some(json!("yes please")),
            bool_question(),
            DecisionMedia::default(),
        )
        .expect("predicts");
    assert_eq!(result.answers[0].1.to_json()["value"], true);
    provider
        .shutdown_and_wait(Duration::from_secs(5))
        .expect("worker exits");
}

#[test]
fn a_slow_prediction_times_out() {
    let (provider, release) = gated_provider(4, Duration::from_millis(50));
    let err = provider
        .predict(Some(json!("x")), bool_question(), DecisionMedia::default())
        .expect_err("times out");
    assert!(matches!(err, DecisionProviderError::Timeout), "{err:?}");
    release.send(()).ok();
}

#[test]
fn a_full_queue_is_rejected() {
    let (provider, release) = gated_provider(1, Duration::from_secs(10));
    let provider = Arc::new(provider);
    // The first request occupies the worker; the second fills the queue.
    let mut pending = Vec::new();
    for _ in 0..2 {
        let provider = Arc::clone(&provider);
        pending.push(std::thread::spawn(move || {
            provider.predict(Some(json!("x")), bool_question(), DecisionMedia::default())
        }));
        std::thread::sleep(Duration::from_millis(100));
    }
    let err = provider
        .predict(Some(json!("x")), bool_question(), DecisionMedia::default())
        .expect_err("queue full");
    assert!(matches!(err, DecisionProviderError::QueueFull), "{err:?}");
    for _ in 0..2 {
        release.send(()).ok();
    }
    for handle in pending {
        handle
            .join()
            .expect("joins")
            .expect("queued request completes");
    }
}

#[test]
fn a_non_decision_checkpoint_fails_to_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), r#"{"model_type": "qwen3"}"#).expect("write");
    let Err(err) = resolve_decision_provider(Some(dir.path()), "x".into(), 4, 30) else {
        panic!("explicit load failure is fatal");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains("failed to load --decision-model"),
        "{message}"
    );
    assert!(
        message.contains("is not a decision checkpoint"),
        "{message}"
    );
    assert!(
        resolve_decision_provider(None, String::new(), 4, 30)
            .is_ok_and(|provider| provider.is_none())
    );
}
