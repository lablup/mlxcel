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

use std::sync::mpsc;
use std::time::Duration;

use super::{EmbeddingWorker, EmbeddingWorkerProvider};
use crate::embeddings::stub::{STUB_DIM, STUB_MAX_LENGTH, STUB_VOCAB_SIZE, stub_loaded_model};
use crate::embeddings::{
    EmbedOptions, EmbeddingBatch, EmbeddingModel, EmbeddingOutput, LoadedEmbeddingModel,
};
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};

fn provider() -> EmbeddingWorkerProvider {
    EmbeddingWorkerProvider::from_loader(
        "stub-embed".to_string(),
        4,
        8,
        Duration::from_secs(30),
        || Ok(stub_loaded_model(false)),
    )
    .expect("stub worker spawns")
}

#[test]
fn worker_loads_stub_and_reports_info() {
    let p = provider();
    assert_eq!(p.model_id(), "stub-embed");
    assert_eq!(p.dim(), STUB_DIM);
    assert_eq!(p.max_length(), STUB_MAX_LENGTH);
    assert_eq!(p.vocab_size(), STUB_VOCAB_SIZE);
    assert!(!p.multi_vector());
    assert!(!p.supports_images());
    assert_eq!(p.info().batch_size, 4);
    assert!(p.created_at() > 0);
}

#[test]
fn worker_round_trips_texts_and_tokens_on_its_own_thread() {
    let p = provider();
    let reply = p
        .embed_texts(
            vec!["hello world".to_string(), "hello".to_string()],
            EmbedOptions::default(),
        )
        .expect("texts embed");
    assert_eq!(reply.vectors.len(), 2);
    // [CLS] hello world [SEP] + [CLS] hello [SEP]
    assert_eq!(reply.prompt_tokens, 7);
    let dot: f32 = reply.vectors[0]
        .values
        .iter()
        .zip(&reply.vectors[1].values)
        .map(|(a, b)| a * b)
        .sum();
    assert!(dot > 0.5, "texts sharing `hello` are similar: {dot}");

    let reply = p
        .embed_tokens(vec![vec![3, 4], vec![5]], EmbedOptions::default())
        .expect("tokens embed");
    assert_eq!(reply.prompt_tokens, 3);
    assert_eq!(reply.vectors[1].shape, vec![STUB_DIM]);
}

#[test]
fn worker_maps_engine_invalid_input() {
    let p = provider();
    let err = p
        .embed_tokens(vec![vec![STUB_VOCAB_SIZE as u32]], EmbedOptions::default())
        .expect_err("out-of-vocab id is rejected");
    assert!(matches!(err, EmbeddingError::InvalidInput(_)), "{err:?}");
    let err = p
        .embed_texts(vec![String::new()], EmbedOptions::default())
        .expect_err("empty text is rejected");
    assert!(matches!(err, EmbeddingError::InvalidInput(_)), "{err:?}");
}

#[test]
fn worker_surfaces_loader_failure() {
    let result = EmbeddingWorkerProvider::from_loader(
        "broken".to_string(),
        4,
        8,
        Duration::from_secs(30),
        || Err(anyhow::anyhow!("synthetic load failure")),
    );
    let err = result.expect_err("loader failure propagates");
    assert!(err.to_string().contains("synthetic load failure"), "{err}");
}

/// A model whose forward pass panics on a sentinel id, to prove the worker
/// survives and keeps serving.
struct PanickingModel;

impl EmbeddingModel for PanickingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> anyhow::Result<EmbeddingOutput> {
        let ids = mlxcel_core::utils::array_to_vec_f32(batch.input_ids);
        if ids.contains(&7.0) {
            panic!("synthetic embed panic");
        }
        let b = mlxcel_core::array_shape(batch.input_ids)[0];
        Ok(EmbeddingOutput {
            embeddings: mlxcel_core::ones(&[b, STUB_DIM as i32], mlxcel_core::dtype::FLOAT32),
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> crate::embeddings::PoolingMode {
        crate::embeddings::PoolingMode::Mean
    }

    fn embedding_dim(&self) -> usize {
        STUB_DIM
    }
}

fn panicking_loaded_model() -> LoadedEmbeddingModel {
    let mut loaded = stub_loaded_model(false);
    loaded.model = Box::new(PanickingModel);
    loaded
}

#[test]
fn worker_survives_a_panicking_request_and_keeps_serving() {
    let p = EmbeddingWorkerProvider::from_loader(
        "panicky".to_string(),
        4,
        8,
        Duration::from_secs(30),
        || Ok(panicking_loaded_model()),
    )
    .expect("worker spawns");

    let err = p
        .embed_tokens(vec![vec![7]], EmbedOptions::default())
        .expect_err("the sentinel id panics inside the engine");
    match err {
        EmbeddingError::Internal(message) => {
            assert!(message.contains("panic"), "{message}");
            assert!(!message.contains("no longer running"), "{message}");
        }
        other => panic!("expected Internal, got {other:?}"),
    }

    let reply = p
        .embed_tokens(vec![vec![3]], EmbedOptions::default())
        .expect("the worker still serves after the panic");
    assert_eq!(reply.vectors.len(), 1);
}

#[test]
fn request_timeout_returns_timeout_error() {
    struct SlowModel;
    impl EmbeddingModel for SlowModel {
        fn embed(&self, batch: &EmbeddingBatch) -> anyhow::Result<EmbeddingOutput> {
            std::thread::sleep(Duration::from_millis(400));
            let b = mlxcel_core::array_shape(batch.input_ids)[0];
            Ok(EmbeddingOutput {
                embeddings: mlxcel_core::ones(&[b, STUB_DIM as i32], mlxcel_core::dtype::FLOAT32),
                last_hidden_state: None,
            })
        }
        fn default_pooling(&self) -> crate::embeddings::PoolingMode {
            crate::embeddings::PoolingMode::Mean
        }
        fn embedding_dim(&self) -> usize {
            STUB_DIM
        }
    }

    let p = EmbeddingWorkerProvider::from_loader(
        "slow".to_string(),
        4,
        8,
        Duration::from_millis(50),
        || {
            let mut loaded = stub_loaded_model(false);
            loaded.model = Box::new(SlowModel);
            Ok(loaded)
        },
    )
    .expect("worker spawns");
    let err = p
        .embed_tokens(vec![vec![3]], EmbedOptions::default())
        .expect_err("slower than the timeout");
    assert!(matches!(err, EmbeddingError::Timeout), "{err:?}");
}

#[test]
fn queue_full_returns_queuefull_error() {
    struct BlockingModel {
        gate: std::cell::Cell<Option<mpsc::Receiver<()>>>,
        started: mpsc::Sender<()>,
    }
    impl EmbeddingModel for BlockingModel {
        fn embed(&self, batch: &EmbeddingBatch) -> anyhow::Result<EmbeddingOutput> {
            if let Some(gate) = self.gate.take() {
                let _ = self.started.send(());
                let _ = gate.recv();
            }
            let b = mlxcel_core::array_shape(batch.input_ids)[0];
            Ok(EmbeddingOutput {
                embeddings: mlxcel_core::ones(&[b, STUB_DIM as i32], mlxcel_core::dtype::FLOAT32),
                last_hidden_state: None,
            })
        }
        fn default_pooling(&self) -> crate::embeddings::PoolingMode {
            crate::embeddings::PoolingMode::Mean
        }
        fn embedding_dim(&self) -> usize {
            STUB_DIM
        }
    }

    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let queue_depth = 2;
    let worker = EmbeddingWorker::spawn(
        "embed-test-queuefull",
        queue_depth,
        Duration::from_millis(150),
        4,
        move || {
            let mut loaded = stub_loaded_model(false);
            loaded.model = Box::new(BlockingModel {
                gate: std::cell::Cell::new(Some(gate_rx)),
                started: started_tx,
            });
            Ok(loaded)
        },
        mlxcel_core::memory::clear_cache,
    )
    .expect("worker spawns");

    std::thread::scope(|scope| {
        scope.spawn(|| {
            let _ = worker.embed_tokens(vec![vec![3]], EmbedOptions::default());
        });
        started_rx.recv().expect("in-flight request started");

        let mut saw_queue_full = false;
        for _ in 0..(queue_depth + 1) {
            match worker.embed_tokens(vec![vec![3]], EmbedOptions::default()) {
                Err(EmbeddingError::QueueFull) => {
                    saw_queue_full = true;
                    break;
                }
                Err(EmbeddingError::Timeout) => continue,
                other => panic!("unexpected result while filling the queue: {other:?}"),
            }
        }
        assert!(saw_queue_full, "a full bounded queue must reject admission");
        drop(gate_tx);
    });
}

/// Log lines captured from a scoped subscriber (#3209 item d).
#[derive(Clone, Default)]
struct CapturedLog(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

struct CapturedWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for CapturedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log buffer").extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
    type Writer = CapturedWriter;
    fn make_writer(&'a self) -> CapturedWriter {
        CapturedWriter(std::sync::Arc::clone(&self.0))
    }
}

impl CapturedLog {
    fn snapshot_lines(&self) -> Vec<String> {
        let text = String::from_utf8(self.0.lock().expect("log buffer").clone()).expect("UTF-8");
        text.lines()
            .filter(|line| line.contains("embedding command finished"))
            .map(str::to_string)
            .collect()
    }
}

#[test]
fn worker_logs_memory_snapshot_per_command() {
    let log = CapturedLog::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(log.clone())
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let p = provider();
        p.embed_texts(vec!["hello".to_string()], EmbedOptions::default())
            .expect("texts embed");
        p.embed_tokens(vec![vec![3, 4]], EmbedOptions::default())
            .expect("tokens embed");
        // An `Err` result is a command too.
        p.embed_tokens(vec![vec![STUB_VOCAB_SIZE as u32]], EmbedOptions::default())
            .expect_err("out-of-vocab id is rejected");
        drop(p);

        // So is a request whose engine call panicked and was recovered.
        let panicky = EmbeddingWorkerProvider::from_loader(
            "panicky".to_string(),
            4,
            8,
            Duration::from_secs(30),
            || Ok(panicking_loaded_model()),
        )
        .expect("worker spawns");
        panicky
            .embed_tokens(vec![vec![7]], EmbedOptions::default())
            .expect_err("the sentinel id panics");
        drop(panicky);
    });
    let lines = log.snapshot_lines();
    assert_eq!(lines.len(), 4, "one snapshot per command: {lines:#?}");
    for line in &lines {
        assert!(line.contains("mlxcel::embeddings"), "{line}");
        for field in ["active_bytes=", "cache_bytes=", "peak_bytes="] {
            assert!(line.contains(field), "{field} missing: {line}");
        }
    }
}

/// Commands the counting hook below has seen; only
/// `worker_clears_cache_after_each_command` installs it.
static AFTER_COMMAND_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn count_after_command() {
    AFTER_COMMAND_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn worker_clears_cache_after_each_command() {
    use std::sync::atomic::Ordering;
    AFTER_COMMAND_CALLS.store(0, Ordering::SeqCst);
    let p = EmbeddingWorkerProvider::from_loader_with_after_command(
        "stub-embed".to_string(),
        4,
        8,
        Duration::from_secs(30),
        || Ok(stub_loaded_model(false)),
        count_after_command,
    )
    .expect("stub worker spawns");
    // Loading is not a command.
    assert_eq!(AFTER_COMMAND_CALLS.load(Ordering::SeqCst), 0);
    p.embed_texts(vec!["hello".to_string()], EmbedOptions::default())
        .expect("texts embed");
    // The hook runs after the reply is sent, so the caller may get here
    // first; dropping the provider joins the worker.
    p.embed_tokens(vec![vec![3, 4]], EmbedOptions::default())
        .expect("tokens embed");
    p.embed_tokens(vec![vec![STUB_VOCAB_SIZE as u32]], EmbedOptions::default())
        .expect_err("an `Err` result is a command too");
    drop(p);
    assert_eq!(AFTER_COMMAND_CALLS.load(Ordering::SeqCst), 3);

    let panicky = EmbeddingWorkerProvider::from_loader_with_after_command(
        "panicky".to_string(),
        4,
        8,
        Duration::from_secs(30),
        || Ok(panicking_loaded_model()),
        count_after_command,
    )
    .expect("worker spawns");
    panicky
        .embed_tokens(vec![vec![7]], EmbedOptions::default())
        .expect_err("the sentinel id panics");
    panicky
        .embed_tokens(vec![vec![3]], EmbedOptions::default())
        .expect("the worker still serves after the panic");
    drop(panicky);
    assert_eq!(
        AFTER_COMMAND_CALLS.load(Ordering::SeqCst),
        5,
        "once per command, including after a recovered panic"
    );
}
