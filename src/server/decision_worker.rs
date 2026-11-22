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

//! Single dedicated-thread worker for the `--decision-model` checkpoint.
//!
//! The same design as [`super::rerank_worker`]: MLX work is thread-affine, so
//! the worker thread installs its own default stream, loads the
//! [`LoadedDecisionModel`] on itself and serves one command at a time from a
//! bounded channel. A full queue is [`DecisionProviderError::QueueFull`]
//! (`503`), a missed reply deadline is [`DecisionProviderError::Timeout`]
//! (`504`), and every prediction runs under `catch_unwind` so one bad request
//! cannot kill the worker.

use std::path::Path;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
    synchronize_thread_local_stream,
};
use serde_json::Value;

use crate::decision::loader::declarations;
use crate::decision::{DecisionKind, LoadedDecisionModel, check_result, load_decision_model};
use crate::server::decision_model::{
    DecisionMedia, DecisionModelProvider, DecisionProviderError, DecisionResult, MediaKind,
    Question, QuestionKind,
};
use crate::server::model_provider::{
    WorkerExitObserver, enqueue_shutdown_command, observe_worker_exit, wait_for_worker_exit,
};

/// Reported when a request cannot reach the worker thread.
const WORKER_GONE: &str = "decision worker thread is no longer running";

/// Static facts about the loaded model, sent back once the worker is ready.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionWorkerInfo {
    pub kind: DecisionKind,
    pub model_type: String,
    pub decision_types: Vec<QuestionKind>,
    pub decision_media: Vec<MediaKind>,
}

type Reply = mpsc::Sender<Result<DecisionResult, DecisionProviderError>>;

enum DecisionCommand {
    Predict {
        state: Option<Value>,
        questions: Vec<(String, Question)>,
        media: Box<DecisionMedia>,
        respond: Reply,
    },
    Shutdown,
}

/// Owns the dedicated thread that loads and runs one decision model.
#[derive(Debug)]
struct DecisionWorker {
    sender: Mutex<mpsc::SyncSender<DecisionCommand>>,
    request_timeout: Duration,
    info: DecisionWorkerInfo,
    handle: Option<JoinHandle<()>>,
    worker_exit: Arc<WorkerExitObserver>,
}

impl DecisionWorker {
    fn spawn<L>(queue_depth: usize, request_timeout: Duration, loader: L) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<LoadedDecisionModel> + Send + 'static,
    {
        let capacity = queue_depth.max(1);
        let request_timeout = if request_timeout == Duration::ZERO {
            Duration::from_secs(crate::server::config::DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS)
        } else {
            request_timeout
        };
        let (command_tx, command_rx) = mpsc::sync_channel::<DecisionCommand>(capacity);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<DecisionWorkerInfo, String>>();
        let handle = thread::Builder::new()
            .name("decision-worker".to_string())
            .spawn(move || worker_loop(loader, command_rx, ready_tx))
            .map_err(|e| anyhow::anyhow!("failed to spawn decision worker thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(info)) => {
                let (handle, worker_exit) = observe_worker_exit(handle);
                Ok(Self {
                    sender: Mutex::new(command_tx),
                    request_timeout,
                    info,
                    handle: Some(handle),
                    worker_exit,
                })
            }
            Ok(Err(message)) => {
                let _ = handle.join();
                Err(anyhow::anyhow!("{message}"))
            }
            Err(_) => {
                let _ = handle.join();
                Err(anyhow::anyhow!(
                    "decision worker thread exited before reporting readiness"
                ))
            }
        }
    }

    fn shutdown_and_wait(&self, timeout: Duration) -> anyhow::Result<()> {
        let started = std::time::Instant::now();
        let _ = enqueue_shutdown_command(&self.sender, DecisionCommand::Shutdown, timeout);
        let remaining = timeout.saturating_sub(started.elapsed());
        wait_for_worker_exit(&self.worker_exit, remaining, "decision model worker")
    }

    fn predict(
        &self,
        state: Option<Value>,
        questions: Vec<(String, Question)>,
        media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionProviderError> {
        let (respond, reply) = mpsc::channel();
        let command = DecisionCommand::Predict {
            state,
            questions,
            media: Box::new(media),
            respond,
        };
        {
            let sender = self.sender.lock().map_err(|_| {
                DecisionProviderError::Internal("decision worker channel poisoned".to_string())
            })?;
            match sender.try_send(command) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Full(_)) => return Err(DecisionProviderError::QueueFull),
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err(DecisionProviderError::Internal(WORKER_GONE.to_string()));
                }
            }
        }
        match reply.recv_timeout(self.request_timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(DecisionProviderError::Timeout),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(DecisionProviderError::Internal(WORKER_GONE.to_string()))
            }
        }
    }
}

impl Drop for DecisionWorker {
    fn drop(&mut self) {
        if let Ok(sender) = self.sender.lock() {
            let _ = sender.send(DecisionCommand::Shutdown);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop<L>(
    loader: L,
    commands: mpsc::Receiver<DecisionCommand>,
    ready: mpsc::Sender<Result<DecisionWorkerInfo, String>>,
) where
    L: FnOnce() -> anyhow::Result<LoadedDecisionModel>,
{
    let stream = new_thread_local_generation_stream();
    install_thread_local_default_stream(stream.as_ref());

    let mut loaded = match loader() {
        Ok(loaded) => loaded,
        Err(err) => {
            let _ = ready.send(Err(format!("{err:#}")));
            return;
        }
    };
    let (decision_types, decision_media) = declarations(loaded.model.as_ref());
    let info = DecisionWorkerInfo {
        kind: loaded.kind,
        model_type: loaded.model_type.clone(),
        decision_types,
        decision_media,
    };
    if ready.send(Ok(info)).is_err() {
        return;
    }
    drop(ready);

    // `AssertUnwindSafe` is sound for the rerank worker's reason: the model is
    // owned and used by this thread alone.
    while let Ok(command) = commands.recv() {
        match command {
            DecisionCommand::Predict {
                state,
                questions,
                media,
                respond,
            } => {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let result = loaded.model.predict(state.as_ref(), &questions, *media)?;
                    check_result(&questions, &result)?;
                    Ok::<_, crate::decision::DecisionError>(result)
                }));
                synchronize_thread_local_stream(stream.as_ref());
                let result = match outcome {
                    Ok(result) => result.map_err(DecisionProviderError::from),
                    Err(payload) => {
                        let detail = payload
                            .downcast_ref::<&str>()
                            .map(|s| (*s).to_string())
                            .or_else(|| payload.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "decision request panicked".to_string());
                        Err(DecisionProviderError::Internal(format!(
                            "decision worker recovered from panic: {detail}"
                        )))
                    }
                };
                let _ = respond.send(result);
            }
            DecisionCommand::Shutdown => break,
        }
    }
}

/// [`DecisionModelProvider`] backed by a decision worker thread.
pub struct DecisionWorkerProvider {
    worker: DecisionWorker,
    model_id: String,
    created_at: i64,
}

impl std::fmt::Debug for DecisionWorkerProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionWorkerProvider")
            .field("model_id", &self.model_id)
            .field("info", &self.worker.info)
            .finish()
    }
}

impl DecisionWorkerProvider {
    /// Spawn the worker and load the checkpoint at `model_path` on it.
    pub fn load(
        model_path: &Path,
        model_id: String,
        queue_depth: usize,
        request_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.to_path_buf();
        Self::from_loader(model_id, queue_depth, request_timeout, move || {
            load_decision_model(&model_path)
        })
    }

    /// Spawn the worker around an arbitrary loader (production or test stub).
    pub(crate) fn from_loader<L>(
        model_id: String,
        queue_depth: usize,
        request_timeout: Duration,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        L: FnOnce() -> anyhow::Result<LoadedDecisionModel> + Send + 'static,
    {
        let worker = DecisionWorker::spawn(queue_depth, request_timeout, loader)?;
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

    pub(crate) fn info(&self) -> &DecisionWorkerInfo {
        &self.worker.info
    }
}

impl DecisionModelProvider for DecisionWorkerProvider {
    fn predict(
        &self,
        state: Option<Value>,
        questions: Vec<(String, Question)>,
        media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionProviderError> {
        self.worker.predict(state, questions, media)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn created_at(&self) -> i64 {
        self.created_at
    }

    fn decision_types(&self) -> &[QuestionKind] {
        &self.worker.info.decision_types
    }

    fn decision_media(&self) -> &[MediaKind] {
        &self.worker.info.decision_media
    }

    fn shutdown_and_wait(&self, timeout: Duration) -> anyhow::Result<()> {
        self.worker.shutdown_and_wait(timeout)
    }
}

/// Load `--decision-model` at startup.
///
/// The flag is always explicit (a decision model is never served through
/// `-m`), so any failure is fatal: a command line that asked for
/// `/v1/decisions` never serves `501` for it. The worker reuses the
/// `--embedding-queue-depth` and `--embedding-request-timeout-secs` bounds,
/// as the rerank worker does.
pub(crate) fn resolve_decision_provider(
    path: Option<&Path>,
    model_id: String,
    queue_depth: usize,
    request_timeout_secs: u64,
) -> anyhow::Result<Option<Arc<dyn DecisionModelProvider>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    tracing::info!(
        path = %path.display(),
        model_id = %model_id,
        "Loading decision model for /v1/decisions"
    );
    let provider = DecisionWorkerProvider::load(
        path,
        model_id,
        queue_depth,
        Duration::from_secs(request_timeout_secs),
    )
    .map_err(|err| {
        err.context(format!(
            "failed to load --decision-model {}",
            path.display()
        ))
    })?;
    let info = provider.info();
    tracing::info!(
        kind = info.kind.as_str(),
        model_type = %info.model_type,
        types = ?info.decision_types.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        media = ?info.decision_media.iter().map(|m| m.field()).collect::<Vec<_>>(),
        "Decision model ready on /v1/decisions"
    );
    Ok(Some(Arc::new(provider)))
}

#[cfg(test)]
#[path = "decision_worker_tests.rs"]
mod tests;
