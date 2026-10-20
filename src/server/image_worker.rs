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

//! Single dedicated-thread worker for the image-generation model.
//!
//! MLX work is thread-affine (see [`super::embedding_worker`] for the full
//! argument): the thread that builds the model must be the thread that runs it.
//! [`ImageWorker`] owns that thread. It installs a thread-local default stream,
//! builds the model on itself, and serves commands from a bounded channel one
//! at a time. A diffusion run holds the GPU for seconds, so there is no
//! batching: requests are serialized.
//!
//! Admission and timeout follow the embedding worker: a bounded `SyncSender`
//! with `try_send` (a full queue is [`ImageError::QueueFull`]), a per-request
//! reply timeout ([`ImageError::Timeout`]), and `catch_unwind` around every
//! generation so one bad request cannot kill the worker for the rest of the
//! process.

use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
    synchronize_thread_local_stream,
};

use crate::imagegen::registry::{FAMILIES, ImageFamily, load_image_model_from};
use crate::imagegen::run::run_request;
use crate::imagegen::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
};
use crate::server::image_model::{ImageError, ImageModelProvider};

/// Default bound on queued requests (one more is running).
pub const DEFAULT_IMAGE_QUEUE_DEPTH: usize = 8;

/// Default per-request reply timeout. Diffusion at high step counts is slow, so
/// this is far above the embedding and audio defaults.
pub const DEFAULT_IMAGE_REQUEST_TIMEOUT_SECS: u64 = 900;

/// Reported when a request cannot reach the worker thread because it has
/// already exited.
const WORKER_GONE: &str = "image worker thread is no longer running";

/// Static facts about the loaded model, sent back once the worker is ready.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ImageWorkerInfo {
    pub family: &'static str,
    pub supports_generate: bool,
    pub supports_edit: bool,
    pub generate_defaults: ImageSamplingDefaults,
    pub edit_defaults: ImageSamplingDefaults,
    pub size_multiple: u32,
}

type Reply = mpsc::Sender<Result<Vec<GeneratedImage>, ImageError>>;

enum ImageCommand {
    Generate {
        request: Box<ImageRequest>,
        respond: Reply,
    },
    Shutdown,
}

/// Owns the dedicated thread that builds and runs one image model.
#[derive(Debug)]
pub(crate) struct ImageWorker {
    /// Bounded command channel; the `Mutex` makes the handle `Sync` and is held
    /// only for the enqueue, never across generation.
    sender: Mutex<mpsc::SyncSender<ImageCommand>>,
    request_timeout: Duration,
    info: ImageWorkerInfo,
    handle: Option<JoinHandle<()>>,
}

impl ImageWorker {
    /// Spawn the worker thread and block until the model has been built.
    ///
    /// `loader` runs on the worker thread so every array it creates belongs to
    /// that thread's MLX context. `queue_depth` is clamped to at least one and a
    /// zero `request_timeout` falls back to the default.
    pub(crate) fn spawn<L>(
        thread_name: &str,
        queue_depth: usize,
        request_timeout: Duration,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<Box<dyn ImageGenerationModel>> + Send + 'static,
    {
        let capacity = queue_depth.max(1);
        let request_timeout = if request_timeout == Duration::ZERO {
            Duration::from_secs(DEFAULT_IMAGE_REQUEST_TIMEOUT_SECS)
        } else {
            request_timeout
        };
        let (command_tx, command_rx) = mpsc::sync_channel::<ImageCommand>(capacity);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<ImageWorkerInfo, String>>();

        let handle = thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || worker_loop(loader, command_rx, ready_tx))
            .map_err(|e| anyhow::anyhow!("failed to spawn image worker thread: {e}"))?;

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
                    "image worker failed to load model: {message}"
                ))
            }
            Err(_) => {
                let _ = handle.join();
                Err(anyhow::anyhow!(
                    "image worker thread exited before reporting readiness"
                ))
            }
        }
    }

    pub(crate) fn info(&self) -> ImageWorkerInfo {
        self.info
    }

    pub(crate) fn generate(
        &self,
        request: ImageRequest,
    ) -> Result<Vec<GeneratedImage>, ImageError> {
        let (respond, reply) = mpsc::channel();
        self.dispatch(ImageCommand::Generate {
            request: Box::new(request),
            respond,
        })?;
        match reply.recv_timeout(self.request_timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(ImageError::Timeout),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(ImageError::Internal(WORKER_GONE.to_string()))
            }
        }
    }

    /// Enqueue without blocking; a full queue is load-shed with
    /// [`ImageError::QueueFull`].
    fn dispatch(&self, command: ImageCommand) -> Result<(), ImageError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| ImageError::Internal("image worker channel poisoned".to_string()))?;
        match sender.try_send(command) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => Err(ImageError::QueueFull),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(ImageError::Internal(WORKER_GONE.to_string()))
            }
        }
    }
}

impl Drop for ImageWorker {
    fn drop(&mut self) {
        // Blocking `send` on purpose: the worker keeps draining, so a slot
        // frees and the shutdown lands after the queued work.
        if let Ok(sender) = self.sender.lock() {
            let _ = sender.send(ImageCommand::Shutdown);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop<L>(
    loader: L,
    commands: mpsc::Receiver<ImageCommand>,
    ready: mpsc::Sender<Result<ImageWorkerInfo, String>>,
) where
    L: FnOnce() -> anyhow::Result<Box<dyn ImageGenerationModel>>,
{
    let stream = new_thread_local_generation_stream();
    install_thread_local_default_stream(stream.as_ref());

    let model = match loader() {
        Ok(model) => model,
        Err(err) => {
            let _ = ready.send(Err(format!("{err:#}")));
            return;
        }
    };
    let info = ImageWorkerInfo {
        family: model.family(),
        supports_generate: model.supports(ImageTask::Generate),
        supports_edit: model.supports(ImageTask::Edit),
        generate_defaults: model.defaults(ImageTask::Generate),
        edit_defaults: model.defaults(ImageTask::Edit),
        size_multiple: model.size_multiple(),
    };
    if ready.send(Ok(info)).is_err() {
        return;
    }
    drop(ready);

    // One command at a time under a panic boundary, then synchronize the
    // stream so dispatch and synchronization stay paired on this thread.
    // `AssertUnwindSafe` is sound for the reason given in the embedding
    // worker: the model is owned and used single-threadedly.
    while let Ok(command) = commands.recv() {
        match command {
            ImageCommand::Generate { request, respond } => {
                let result = run_guarded(stream.as_ref(), || {
                    run_request(model.as_ref(), &request)
                        .map_err(|e| ImageError::Internal(format!("{e:#}")))
                });
                let _ = respond.send(result);
            }
            ImageCommand::Shutdown => break,
        }
    }
}

fn run_guarded<F>(
    stream: Option<&mlxcel_core::UniquePtr<mlxcel_core::MlxThreadLocalStream>>,
    call: F,
) -> Result<Vec<GeneratedImage>, ImageError>
where
    F: FnOnce() -> Result<Vec<GeneratedImage>, ImageError>,
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
                "image request panicked".to_string()
            };
            tracing::error!(
                target: "mlxcel::imagegen",
                "image worker recovered from panic: {detail}"
            );
            Err(ImageError::Internal(format!(
                "image worker recovered from panic: {detail}"
            )))
        }
    }
}

/// [`ImageModelProvider`] backed by an [`ImageWorker`].
pub struct ImageWorkerProvider {
    worker: ImageWorker,
    model_id: String,
    created_at: i64,
}

impl std::fmt::Debug for ImageWorkerProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageWorkerProvider")
            .field("model_id", &self.model_id)
            .field("info", &self.worker.info())
            .finish()
    }
}

impl ImageWorkerProvider {
    /// Spawn the worker and load the checkpoint at `model_path` on it.
    ///
    /// Returns `Err` when the thread cannot start, no family is recognized, the
    /// family is not supported by this build, or the checkpoint fails to load.
    pub fn load(
        model_path: &Path,
        model_id: String,
        queue_depth: usize,
        request_timeout: Duration,
    ) -> anyhow::Result<Self> {
        Self::load_with_families(model_path, model_id, queue_depth, request_timeout, FAMILIES)
    }

    /// [`Self::load`] against an explicit family table, so tests can serve a
    /// stub family.
    pub(crate) fn load_with_families(
        model_path: &Path,
        model_id: String,
        queue_depth: usize,
        request_timeout: Duration,
        families: &'static [ImageFamily],
    ) -> anyhow::Result<Self> {
        let model_path = model_path.to_path_buf();
        Self::from_loader(model_id, queue_depth, request_timeout, move || {
            load_image_model_from(&model_path, families)
        })
    }

    /// Spawn the worker around an arbitrary loader.
    pub(crate) fn from_loader<L>(
        model_id: String,
        queue_depth: usize,
        request_timeout: Duration,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<Box<dyn ImageGenerationModel>> + Send + 'static,
    {
        let worker = ImageWorker::spawn("image-worker", queue_depth, request_timeout, loader)?;
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

    pub(crate) fn info(&self) -> ImageWorkerInfo {
        self.worker.info()
    }
}

impl ImageModelProvider for ImageWorkerProvider {
    fn generate(&self, req: ImageRequest) -> Result<Vec<GeneratedImage>, ImageError> {
        self.worker.generate(req)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn created_at(&self) -> i64 {
        self.created_at
    }

    fn family(&self) -> &str {
        self.worker.info().family
    }

    fn supports(&self, task: ImageTask) -> bool {
        match task {
            ImageTask::Generate => self.worker.info().supports_generate,
            ImageTask::Edit => self.worker.info().supports_edit,
        }
    }

    fn defaults(&self, task: ImageTask) -> ImageSamplingDefaults {
        match task {
            ImageTask::Generate => self.worker.info().generate_defaults,
            ImageTask::Edit => self.worker.info().edit_defaults,
        }
    }

    fn size_multiple(&self) -> u32 {
        self.worker.info().size_multiple
    }
}

#[cfg(test)]
#[path = "image_worker_tests.rs"]
mod tests;
