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

//! Single dedicated-thread worker for the reranker model.
//!
//! MLX work is thread-affine (see [`super::audio_worker`] for the full
//! argument): the thread that loads the weights must be the thread that
//! evaluates them. [`RerankWorker`] owns that thread. It installs a
//! thread-local default stream, loads the [`LoadedReranker`] on itself, and
//! serves commands from a bounded channel one at a time. Prompt assembly,
//! tokenization, micro-batching and the forward pass all run inside the family
//! on this thread; the HTTP side only ever sees `f32` scores.
//!
//! The admission and timeout design is the embedding worker's: a bounded
//! `SyncSender` with `try_send` (full queue = [`RerankError::QueueFull`]), a
//! per-request reply timeout ([`RerankError::Timeout`]), and `catch_unwind`
//! around every scoring call so one bad request cannot kill the worker for the
//! rest of the process.

use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
    synchronize_thread_local_stream,
};

use crate::rerank::{
    LoadedReranker, RerankItem, RerankLoadOptions, RerankScores, RerankerKind,
    load_reranker_with_options,
};
use crate::server::rerank_model::{RerankError, RerankModelProvider};

/// Reported when a request cannot reach the worker thread because it has
/// already exited.
const WORKER_GONE: &str = "rerank worker thread is no longer running";

/// Static facts about the loaded reranker, sent back once the worker is ready.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RerankWorkerInfo {
    pub kind: RerankerKind,
    pub max_length: usize,
    pub batch_size: usize,
    pub supports_images: bool,
    pub model_type: String,
}

type Reply = mpsc::Sender<Result<RerankScores, RerankError>>;

/// A unit of work plus the channel to reply on.
enum RerankCommand {
    Score {
        query: Box<RerankItem>,
        documents: Vec<RerankItem>,
        instruction: Option<String>,
        respond: Reply,
    },
    Shutdown,
}

/// Owns the dedicated thread that loads and runs one reranker.
#[derive(Debug)]
pub(crate) struct RerankWorker {
    /// Bounded command channel; the `Mutex` makes the handle `Sync` and is
    /// held only for the enqueue, never across inference.
    sender: Mutex<mpsc::SyncSender<RerankCommand>>,
    /// Per-request reply timeout. Frees the caller's blocking thread; does not
    /// cancel the in-flight MLX work.
    request_timeout: Duration,
    info: RerankWorkerInfo,
    handle: Option<JoinHandle<()>>,
}

impl RerankWorker {
    /// Spawn the worker thread and block until the model has loaded.
    ///
    /// `loader` runs on the worker thread so every array it creates belongs to
    /// that thread's MLX context. `queue_depth` bounds the command channel
    /// (clamped to at least one) and `request_timeout` bounds the reply wait (a
    /// zero falls back to the default).
    pub(crate) fn spawn<L>(
        thread_name: &str,
        queue_depth: usize,
        request_timeout: Duration,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<LoadedReranker> + Send + 'static,
    {
        let capacity = queue_depth.max(1);
        let request_timeout = if request_timeout == Duration::ZERO {
            Duration::from_secs(crate::server::config::DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS)
        } else {
            request_timeout
        };
        let (command_tx, command_rx) = mpsc::sync_channel::<RerankCommand>(capacity);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<RerankWorkerInfo, String>>();

        let handle = thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || worker_loop(loader, command_rx, ready_tx))
            .map_err(|e| anyhow::anyhow!("failed to spawn rerank worker thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(info)) => Ok(Self {
                sender: Mutex::new(command_tx),
                request_timeout,
                info,
                handle: Some(handle),
            }),
            Ok(Err(message)) => {
                let _ = handle.join();
                Err(anyhow::anyhow!(
                    "rerank worker failed to load model: {message}"
                ))
            }
            Err(_) => {
                let _ = handle.join();
                Err(anyhow::anyhow!(
                    "rerank worker thread exited before reporting readiness"
                ))
            }
        }
    }

    /// Static facts about the loaded reranker.
    pub(crate) fn info(&self) -> &RerankWorkerInfo {
        &self.info
    }

    pub(crate) fn score(
        &self,
        query: RerankItem,
        documents: Vec<RerankItem>,
        instruction: Option<String>,
    ) -> Result<RerankScores, RerankError> {
        let (respond, reply) = mpsc::channel();
        self.dispatch(RerankCommand::Score {
            query: Box::new(query),
            documents,
            instruction,
            respond,
        })?;
        match reply.recv_timeout(self.request_timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(RerankError::Timeout),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(RerankError::Internal(WORKER_GONE.to_string()))
            }
        }
    }

    /// Enqueue without blocking; a full bounded queue is rejected with
    /// [`RerankError::QueueFull`] (load shedding).
    fn dispatch(&self, command: RerankCommand) -> Result<(), RerankError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| RerankError::Internal("rerank worker channel poisoned".to_string()))?;
        match sender.try_send(command) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => Err(RerankError::QueueFull),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(RerankError::Internal(WORKER_GONE.to_string()))
            }
        }
    }
}

impl Drop for RerankWorker {
    fn drop(&mut self) {
        // Blocking `send` on purpose: the worker keeps draining, so a slot
        // frees and the shutdown lands after the queued work.
        if let Ok(sender) = self.sender.lock() {
            let _ = sender.send(RerankCommand::Shutdown);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Body of the worker thread: install the stream, load the model, serve.
fn worker_loop<L>(
    loader: L,
    commands: mpsc::Receiver<RerankCommand>,
    ready: mpsc::Sender<Result<RerankWorkerInfo, String>>,
) where
    L: FnOnce() -> anyhow::Result<LoadedReranker>,
{
    let stream = new_thread_local_generation_stream();
    install_thread_local_default_stream(stream.as_ref());

    let loaded = match loader() {
        Ok(loaded) => loaded,
        Err(err) => {
            let _ = ready.send(Err(format!("{err:#}")));
            return;
        }
    };
    let info = RerankWorkerInfo {
        kind: loaded.kind,
        max_length: loaded.reranker.max_length(),
        batch_size: loaded.reranker.batch_size(),
        supports_images: loaded.reranker.supports_images(),
        model_type: loaded.model_type.clone(),
    };
    if ready.send(Ok(info)).is_err() {
        return;
    }
    drop(ready);

    // One command at a time under a panic boundary, then synchronize the
    // stream so dispatch and synchronization stay paired on this thread.
    // `AssertUnwindSafe` is sound here for the same reason as in the embedding
    // worker: the reranker is owned and used single-threadedly and holds no
    // cross-call invariants a recovered panic could leave torn.
    while let Ok(command) = commands.recv() {
        match command {
            RerankCommand::Score {
                query,
                documents,
                instruction,
                respond,
            } => {
                let result = run_guarded(stream.as_ref(), || {
                    loaded
                        .reranker
                        .score(&query, &documents, instruction.as_deref())
                        .map_err(|err| RerankError::Internal(format!("{err:#}")))
                });
                let _ = respond.send(result);
            }
            RerankCommand::Shutdown => break,
        }
    }
}

/// Run one scoring call under a panic boundary, then synchronize the stream.
fn run_guarded<F>(
    stream: Option<&mlxcel_core::UniquePtr<mlxcel_core::MlxThreadLocalStream>>,
    call: F,
) -> Result<RerankScores, RerankError>
where
    F: FnOnce() -> Result<RerankScores, RerankError>,
{
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call));
    synchronize_thread_local_stream(stream);
    match outcome {
        Ok(result) => result,
        Err(payload) => {
            let detail = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "rerank request panicked".to_string()
            };
            tracing::error!(
                target: "mlxcel::rerank",
                "rerank worker recovered from panic: {detail}"
            );
            Err(RerankError::Internal(format!(
                "rerank worker recovered from panic: {detail}"
            )))
        }
    }
}

/// [`RerankModelProvider`] backed by a [`RerankWorker`].
pub struct RerankWorkerProvider {
    worker: RerankWorker,
    model_id: String,
    created_at: i64,
}

impl std::fmt::Debug for RerankWorkerProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RerankWorkerProvider")
            .field("model_id", &self.model_id)
            .field("info", self.worker.info())
            .finish()
    }
}

impl RerankWorkerProvider {
    /// Spawn the worker and load the checkpoint at `model_path` on it.
    ///
    /// `model_id` is the id reported in responses and `/v1/models`. Returns
    /// `Err` when the thread cannot start or the checkpoint fails to load.
    pub fn load(
        model_path: &Path,
        model_id: String,
        queue_depth: usize,
        request_timeout: Duration,
        load_options: RerankLoadOptions,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.to_path_buf();
        Self::from_loader(model_id, queue_depth, request_timeout, move || {
            load_reranker_with_options(&model_path, load_options)
        })
    }

    /// Spawn the worker around an arbitrary loader (the production path above,
    /// or a test stub).
    pub(crate) fn from_loader<L>(
        model_id: String,
        queue_depth: usize,
        request_timeout: Duration,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<LoadedReranker> + Send + 'static,
    {
        let worker = RerankWorker::spawn("rerank-worker", queue_depth, request_timeout, loader)?;
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Ok(Self {
            worker,
            model_id,
            created_at,
        })
    }

    /// Static facts about the loaded reranker.
    pub(crate) fn info(&self) -> &RerankWorkerInfo {
        self.worker.info()
    }
}

impl RerankModelProvider for RerankWorkerProvider {
    fn rerank(
        &self,
        query: RerankItem,
        documents: Vec<RerankItem>,
        instruction: Option<String>,
    ) -> Result<RerankScores, RerankError> {
        self.worker.score(query, documents, instruction)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn created_at(&self) -> i64 {
        self.created_at
    }

    fn kind(&self) -> RerankerKind {
        self.worker.info().kind
    }

    fn supports_images(&self) -> bool {
        self.worker.info().supports_images
    }

    fn max_length(&self) -> usize {
        self.worker.info().max_length
    }
}

#[cfg(test)]
#[path = "rerank_worker_tests.rs"]
mod tests;
