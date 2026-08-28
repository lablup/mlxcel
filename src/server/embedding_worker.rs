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

//! Single dedicated-thread worker for the embedding model.
//!
//! MLX work is thread-affine (see [`super::audio_worker`] for the full
//! argument): the thread that loads the weights must be the thread that
//! evaluates them. [`EmbeddingWorker`] owns that thread. It installs a
//! thread-local default stream, loads the [`LoadedEmbeddingModel`] on
//! itself, wraps it in an [`EmbeddingEngine`], and serves commands from a
//! bounded channel one at a time. Tokenization, length-sorted
//! micro-batching, pooling and normalization all run inside the engine on
//! this thread; the HTTP side only ever sees `Vec<f32>` vectors.
//!
//! The admission and timeout design is the audio worker's: a bounded
//! `SyncSender` with `try_send` (full queue = [`EmbeddingError::QueueFull`]),
//! a per-request reply timeout ([`EmbeddingError::Timeout`]), and
//! `catch_unwind` around every engine call so one bad request cannot kill
//! the worker for the rest of the process.

use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
    synchronize_thread_local_stream,
};

use crate::embeddings::{
    EmbedOptions, EmbedReply, EmbeddingEngine, EmbeddingLoadOptions, ImageInput,
    LoadedEmbeddingModel, load_embedding_model_with_options,
};
use crate::models::ModelType;
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};

/// Reported when a request cannot reach the worker thread because it has
/// already exited.
const WORKER_GONE: &str = "embedding worker thread is no longer running";

/// Static facts about the loaded model, sent back once the worker is ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EmbeddingWorkerInfo {
    pub dim: usize,
    pub max_length: usize,
    pub vocab_size: usize,
    pub multi_vector: bool,
    pub supports_images: bool,
    pub batch_size: usize,
    pub model_type: ModelType,
    /// The pooling this checkpoint resolved, reported by `/props` (#1452).
    pub pooling: crate::embeddings::PoolingMode,
    /// The normalization a request that names none gets (#1452).
    pub embd_normalize: crate::embeddings::EmbdNormalize,
}

type Reply = mpsc::Sender<Result<EmbedReply, EmbeddingError>>;

/// A unit of work plus the channel to reply on.
enum EmbeddingCommand {
    Texts {
        texts: Vec<String>,
        opts: EmbedOptions,
        respond: Reply,
    },
    Tokens {
        rows: Vec<Vec<u32>>,
        opts: EmbedOptions,
        respond: Reply,
    },
    Image {
        image: Box<ImageInput>,
        opts: EmbedOptions,
        respond: Reply,
    },
    Shutdown,
}

/// Owns the dedicated thread that loads and runs one embedding model.
#[derive(Debug)]
pub(crate) struct EmbeddingWorker {
    /// Bounded command channel; the `Mutex` makes the handle `Sync` and is
    /// held only for the enqueue, never across inference.
    sender: Mutex<mpsc::SyncSender<EmbeddingCommand>>,
    /// Per-request reply timeout. Frees the caller's blocking thread; does
    /// not cancel the in-flight MLX work.
    request_timeout: Duration,
    info: EmbeddingWorkerInfo,
    handle: Option<JoinHandle<()>>,
}

impl EmbeddingWorker {
    /// Spawn the worker thread and block until the model has loaded.
    ///
    /// `loader` runs on the worker thread so every array it creates belongs
    /// to that thread's MLX context. `queue_depth` bounds the command
    /// channel (clamped to at least one), `request_timeout` bounds the reply
    /// wait (a zero falls back to the default), and `batch_size` is the
    /// micro-batch width handed to the engine.
    pub(crate) fn spawn<L>(
        thread_name: &str,
        queue_depth: usize,
        request_timeout: Duration,
        batch_size: usize,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<LoadedEmbeddingModel> + Send + 'static,
    {
        let capacity = queue_depth.max(1);
        let request_timeout = if request_timeout == Duration::ZERO {
            Duration::from_secs(crate::server::config::DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS)
        } else {
            request_timeout
        };
        let (command_tx, command_rx) = mpsc::sync_channel::<EmbeddingCommand>(capacity);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<EmbeddingWorkerInfo, String>>();

        let handle = thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || worker_loop(loader, batch_size, command_rx, ready_tx))
            .map_err(|e| anyhow::anyhow!("failed to spawn embedding worker thread: {e}"))?;

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
                    "embedding worker failed to load model: {message}"
                ))
            }
            Err(_) => {
                let _ = handle.join();
                Err(anyhow::anyhow!(
                    "embedding worker thread exited before reporting readiness"
                ))
            }
        }
    }

    /// Static facts about the loaded model.
    pub(crate) fn info(&self) -> EmbeddingWorkerInfo {
        self.info
    }

    pub(crate) fn embed_texts(
        &self,
        texts: Vec<String>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        let (respond, reply) = mpsc::channel();
        self.dispatch(EmbeddingCommand::Texts {
            texts,
            opts,
            respond,
        })?;
        self.wait(reply)
    }

    pub(crate) fn embed_tokens(
        &self,
        rows: Vec<Vec<u32>>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        let (respond, reply) = mpsc::channel();
        self.dispatch(EmbeddingCommand::Tokens {
            rows,
            opts,
            respond,
        })?;
        self.wait(reply)
    }

    pub(crate) fn embed_image(
        &self,
        image: ImageInput,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        let (respond, reply) = mpsc::channel();
        self.dispatch(EmbeddingCommand::Image {
            image: Box::new(image),
            opts,
            respond,
        })?;
        self.wait(reply)
    }

    fn wait(
        &self,
        reply: mpsc::Receiver<Result<EmbedReply, EmbeddingError>>,
    ) -> Result<EmbedReply, EmbeddingError> {
        match reply.recv_timeout(self.request_timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(EmbeddingError::Timeout),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(EmbeddingError::Internal(WORKER_GONE.to_string()))
            }
        }
    }

    /// Enqueue without blocking; a full bounded queue is rejected with
    /// [`EmbeddingError::QueueFull`] (load shedding).
    fn dispatch(&self, command: EmbeddingCommand) -> Result<(), EmbeddingError> {
        let sender = self.sender.lock().map_err(|_| {
            EmbeddingError::Internal("embedding worker channel poisoned".to_string())
        })?;
        match sender.try_send(command) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => Err(EmbeddingError::QueueFull),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(EmbeddingError::Internal(WORKER_GONE.to_string()))
            }
        }
    }
}

impl Drop for EmbeddingWorker {
    fn drop(&mut self) {
        // Blocking `send` on purpose: the worker keeps draining, so a slot
        // frees and the shutdown lands after the queued work.
        if let Ok(sender) = self.sender.lock() {
            let _ = sender.send(EmbeddingCommand::Shutdown);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Body of the worker thread: install the stream, load the model, serve.
fn worker_loop<L>(
    loader: L,
    batch_size: usize,
    commands: mpsc::Receiver<EmbeddingCommand>,
    ready: mpsc::Sender<Result<EmbeddingWorkerInfo, String>>,
) where
    L: FnOnce() -> anyhow::Result<LoadedEmbeddingModel>,
{
    let stream = new_thread_local_generation_stream();
    install_thread_local_default_stream(stream.as_ref());

    let engine = match loader() {
        Ok(loaded) => EmbeddingEngine::new(loaded, batch_size),
        Err(err) => {
            let _ = ready.send(Err(format!("{err:#}")));
            return;
        }
    };
    let info = EmbeddingWorkerInfo {
        dim: engine.dim(),
        max_length: engine.max_length(),
        vocab_size: engine.vocab_size(),
        multi_vector: engine.multi_vector(),
        supports_images: engine.supports_images(),
        batch_size: engine.batch_size(),
        model_type: engine.model_type(),
        pooling: engine.pooling(),
        embd_normalize: engine.default_normalize(),
    };
    if ready.send(Ok(info)).is_err() {
        return;
    }
    drop(ready);

    // One command at a time under a panic boundary, then synchronize the
    // stream so dispatch and synchronization stay paired on this thread.
    // `AssertUnwindSafe` is sound here for the same reason as in the audio
    // worker: the engine is owned and used single-threadedly and holds no
    // cross-call invariants a recovered panic could leave torn.
    while let Ok(command) = commands.recv() {
        match command {
            EmbeddingCommand::Texts {
                texts,
                opts,
                respond,
            } => {
                let result = run_guarded(stream.as_ref(), || {
                    engine.embed_texts(&texts, &opts).map_err(Into::into)
                });
                let _ = respond.send(result);
            }
            EmbeddingCommand::Tokens {
                rows,
                opts,
                respond,
            } => {
                let result = run_guarded(stream.as_ref(), || {
                    engine.embed_tokens(&rows, &opts).map_err(Into::into)
                });
                let _ = respond.send(result);
            }
            EmbeddingCommand::Image {
                image,
                opts,
                respond,
            } => {
                let result = run_guarded(stream.as_ref(), || {
                    engine.embed_image(*image, &opts).map_err(Into::into)
                });
                let _ = respond.send(result);
            }
            EmbeddingCommand::Shutdown => break,
        }
    }
}

/// Run one engine call under a panic boundary, then synchronize the stream.
fn run_guarded<F>(
    stream: Option<&mlxcel_core::UniquePtr<mlxcel_core::MlxThreadLocalStream>>,
    call: F,
) -> Result<EmbedReply, EmbeddingError>
where
    F: FnOnce() -> Result<EmbedReply, EmbeddingError>,
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
                "embedding request panicked".to_string()
            };
            tracing::error!(
                target: "mlxcel::embeddings",
                "embedding worker recovered from panic: {detail}"
            );
            Err(EmbeddingError::Internal(format!(
                "embedding worker recovered from panic: {detail}"
            )))
        }
    }
}

/// [`EmbeddingModelProvider`] backed by an [`EmbeddingWorker`].
pub struct EmbeddingWorkerProvider {
    worker: EmbeddingWorker,
    model_id: String,
    created_at: i64,
}

impl std::fmt::Debug for EmbeddingWorkerProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingWorkerProvider")
            .field("model_id", &self.model_id)
            .field("info", &self.worker.info())
            .finish()
    }
}

impl EmbeddingWorkerProvider {
    /// Spawn the worker and load the checkpoint at `model_path` on it.
    ///
    /// `model_id` is the id reported in responses and `/v1/models`. Returns
    /// `Err` when the thread cannot start or the checkpoint fails to load.
    pub fn load(
        model_path: &Path,
        model_id: String,
        batch_size: usize,
        queue_depth: usize,
        request_timeout: Duration,
        load_options: EmbeddingLoadOptions,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.to_path_buf();
        Self::from_loader(
            model_id,
            batch_size,
            queue_depth,
            request_timeout,
            move || load_embedding_model_with_options(&model_path, load_options),
        )
    }

    /// Spawn the worker around an arbitrary loader (the production path
    /// above, or a test stub).
    pub(crate) fn from_loader<L>(
        model_id: String,
        batch_size: usize,
        queue_depth: usize,
        request_timeout: Duration,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<LoadedEmbeddingModel> + Send + 'static,
    {
        let worker = EmbeddingWorker::spawn(
            "embedding-worker",
            queue_depth,
            request_timeout,
            batch_size,
            loader,
        )?;
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

    /// Static facts about the loaded model.
    pub(crate) fn info(&self) -> EmbeddingWorkerInfo {
        self.worker.info()
    }
}

impl EmbeddingModelProvider for EmbeddingWorkerProvider {
    fn embed_texts(
        &self,
        texts: Vec<String>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        self.worker.embed_texts(texts, opts)
    }

    fn embed_tokens(
        &self,
        token_rows: Vec<Vec<u32>>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        self.worker.embed_tokens(token_rows, opts)
    }

    fn embed_image(
        &self,
        image: ImageInput,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        self.worker.embed_image(image, opts)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn created_at(&self) -> i64 {
        self.created_at
    }

    fn dim(&self) -> usize {
        self.worker.info().dim
    }

    fn multi_vector(&self) -> bool {
        self.worker.info().multi_vector
    }

    fn pooling(&self) -> crate::embeddings::PoolingMode {
        self.worker.info().pooling
    }

    fn embd_normalize(&self) -> crate::embeddings::EmbdNormalize {
        self.worker.info().embd_normalize
    }

    fn supports_images(&self) -> bool {
        self.worker.info().supports_images
    }

    fn vocab_size(&self) -> usize {
        self.worker.info().vocab_size
    }

    fn max_length(&self) -> usize {
        self.worker.info().max_length
    }
}

#[cfg(test)]
#[path = "embedding_worker_tests.rs"]
mod tests;
