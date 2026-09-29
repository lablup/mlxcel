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

//! One long-lived MLX worker with a single-session reservation, serving the
//! `/v1/realtime` VoiceChat WebSocket (issue #1376).
//!
//! Port of `RealtimeVoiceChatEngine` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/server/realtime.py.
//! The model is loaded on the engine thread and the active session (which
//! borrows it) lives on that thread's stack, so no MLX array ever crosses a
//! thread. The async socket task sends `open` / `push` / `flush` / `cancel` /
//! `close` over a channel and awaits a one-shot reply, so the Tokio runtime
//! never runs MLX code. Events are serialized to their wire form (including
//! the base64 PCM16 audio) on the engine thread.
//!
//! The MLX buffer cache is cleared on `close` only: clearing it on every
//! frame raises per-frame latency.
//!
//! Used by: `server/routes/realtime.rs`, `server/startup.rs`

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, mpsc};
use std::thread::{self, JoinHandle};

use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
    synchronize_thread_local_stream,
};
use tokio::sync::oneshot;

use crate::models::nemotron_voicechat::streaming::VoiceChatError;
use crate::server::realtime_protocol::{ServerEvent, serialize_event};
pub use crate::server::realtime_session::{
    RealtimeModel, RealtimeSession, RealtimeSessionConfig, RealtimeSessionInfo,
};

/// Errors of an engine call.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RealtimeEngineError {
    /// `open` while another session is active on the engine thread.
    AlreadyActive,
    /// `push` / `flush` / `cancel` / `close` for a session that is not the
    /// active one (never opened, or already closed).
    NotActive,
    /// The session rejected the call (bad input, context limit, model error).
    Session(VoiceChatError),
    /// A session call panicked; the session was dropped.
    Panicked(String),
    /// The engine thread has exited.
    WorkerGone,
}

impl fmt::Display for RealtimeEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyActive => f.write_str("a realtime session is already active"),
            Self::NotActive => f.write_str("realtime session is not active"),
            Self::Session(err) => write!(f, "{err}"),
            Self::Panicked(msg) => write!(f, "realtime session failed: {msg}"),
            Self::WorkerGone => f.write_str("realtime engine thread is no longer running"),
        }
    }
}

impl std::error::Error for RealtimeEngineError {}

type EngineResult<T> = Result<T, RealtimeEngineError>;

enum CommandKind {
    Open(RealtimeSessionConfig),
    Session(SessionCommand),
}

/// A command on the active session.
enum SessionCommand {
    Push { samples: Vec<f32>, sample_rate: u32 },
    Flush { pad_partial: bool },
    Cancel,
    Close,
}

enum Reply {
    Opened(RealtimeSessionInfo),
    Events(Vec<ServerEvent>),
    Closed,
}

enum Message {
    Call {
        session_id: String,
        kind: CommandKind,
        /// `None` for a fire-and-forget `close` from a dropped socket task.
        reply: Option<oneshot::Sender<EngineResult<Reply>>>,
    },
    Shutdown,
}

/// Handle to the engine thread. `Send + Sync`; share it behind an `Arc`.
pub struct RealtimeVoiceChatEngine {
    commands: mpsc::Sender<Message>,
    reserved: Mutex<Option<String>>,
    model_id: String,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl fmt::Debug for RealtimeVoiceChatEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeVoiceChatEngine")
            .field("model_id", &self.model_id)
            .field("reserved", &self.is_reserved())
            .finish()
    }
}

impl RealtimeVoiceChatEngine {
    /// Start the engine thread, load the Nemotron VoiceChat checkpoint on it,
    /// and block until the load finishes. `model_id` is the name reported in
    /// `session.updated`.
    pub fn spawn(model_path: &Path, model_id: impl Into<String>) -> anyhow::Result<Self> {
        let path: PathBuf = model_path.to_path_buf();
        Self::spawn_with_loader(model_id, move || {
            crate::models::NemotronVoiceChatModel::load(&path).map_err(|err| format!("{err:#}"))
        })
        .map_err(anyhow::Error::msg)
    }

    /// Start the engine thread with a custom model loader (tests use a fake
    /// model). `loader` runs on the engine thread; the call blocks until it
    /// returns and reports its error.
    pub fn spawn_with_loader<M, L>(model_id: impl Into<String>, loader: L) -> Result<Self, String>
    where
        M: RealtimeModel + 'static,
        L: FnOnce() -> Result<M, String> + Send + 'static,
    {
        let (commands, receiver) = mpsc::channel::<Message>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        let handle = thread::Builder::new()
            .name("mlxcel-realtime".to_string())
            .spawn(move || worker_loop(loader, receiver, ready_tx))
            .map_err(|err| format!("failed to spawn the realtime engine thread: {err}"))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                commands,
                reserved: Mutex::new(None),
                model_id: model_id.into(),
                handle: Mutex::new(Some(handle)),
            }),
            Ok(Err(message)) => {
                let _ = handle.join();
                Err(format!(
                    "realtime engine failed to load the model: {message}"
                ))
            }
            Err(_) => {
                let _ = handle.join();
                Err("realtime engine thread exited before reporting readiness".to_string())
            }
        }
    }

    /// The model name reported in `session.updated`.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Admit `session_id` as the one connection; `false` when another holds
    /// the reservation.
    pub fn try_reserve(&self, session_id: &str) -> bool {
        let mut reserved = self.reserved.lock().unwrap_or_else(|e| e.into_inner());
        if reserved.is_some() {
            return false;
        }
        *reserved = Some(session_id.to_string());
        true
    }

    /// Release the reservation if `session_id` holds it.
    pub fn release(&self, session_id: &str) {
        let mut reserved = self.reserved.lock().unwrap_or_else(|e| e.into_inner());
        if reserved.as_deref() == Some(session_id) {
            *reserved = None;
        }
    }

    /// Whether a connection holds the reservation.
    pub fn is_reserved(&self) -> bool {
        self.reserved
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Create the streaming session (prefills the system prompt).
    pub async fn open(
        &self,
        session_id: &str,
        config: RealtimeSessionConfig,
    ) -> EngineResult<RealtimeSessionInfo> {
        match self.call(session_id, CommandKind::Open(config)).await? {
            Reply::Opened(info) => Ok(info),
            _ => Err(RealtimeEngineError::WorkerGone),
        }
    }

    /// Push PCM and return the wire events of every frame it completed.
    pub async fn push(
        &self,
        session_id: &str,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> EngineResult<Vec<ServerEvent>> {
        self.events(
            session_id,
            SessionCommand::Push {
                samples,
                sample_rate,
            },
        )
        .await
    }

    /// Finish the input; the events end with `response.done`.
    pub async fn flush(
        &self,
        session_id: &str,
        pad_partial: bool,
    ) -> EngineResult<Vec<ServerEvent>> {
        self.events(session_id, SessionCommand::Flush { pad_partial })
            .await
    }

    /// Stop the stream; the events end with `response.cancelled`.
    pub async fn cancel(&self, session_id: &str) -> EngineResult<Vec<ServerEvent>> {
        self.events(session_id, SessionCommand::Cancel).await
    }

    /// Drop the session (cancelling it first if it was not flushed) and clear
    /// the MLX buffer cache.
    pub async fn close(&self, session_id: &str) -> EngineResult<()> {
        self.call(session_id, CommandKind::Session(SessionCommand::Close))
            .await
            .map(|_| ())
    }

    /// Queue a `close` without waiting for it, for a socket task that is
    /// being dropped. The channel is FIFO, so a later `open` from the next
    /// connection runs after it.
    pub fn close_detached(&self, session_id: &str) {
        let _ = self.commands.send(Message::Call {
            session_id: session_id.to_string(),
            kind: CommandKind::Session(SessionCommand::Close),
            reply: None,
        });
    }

    async fn events(
        &self,
        session_id: &str,
        command: SessionCommand,
    ) -> EngineResult<Vec<ServerEvent>> {
        match self.call(session_id, CommandKind::Session(command)).await? {
            Reply::Events(events) => Ok(events),
            _ => Err(RealtimeEngineError::WorkerGone),
        }
    }

    async fn call(&self, session_id: &str, kind: CommandKind) -> EngineResult<Reply> {
        let (reply, receiver) = oneshot::channel();
        self.commands
            .send(Message::Call {
                session_id: session_id.to_string(),
                kind,
                reply: Some(reply),
            })
            .map_err(|_| RealtimeEngineError::WorkerGone)?;
        receiver
            .await
            .map_err(|_| RealtimeEngineError::WorkerGone)?
    }
}

impl Drop for RealtimeVoiceChatEngine {
    fn drop(&mut self) {
        let _ = self.commands.send(Message::Shutdown);
        let handle = self
            .handle
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }
}

/// The session the engine thread is driving.
struct Active<'m> {
    session_id: String,
    session: Box<dyn RealtimeSession + 'm>,
}

fn worker_loop<M, L>(
    loader: L,
    commands: mpsc::Receiver<Message>,
    ready: mpsc::Sender<Result<(), String>>,
) where
    M: RealtimeModel,
    L: FnOnce() -> Result<M, String>,
{
    // Install this thread's default MLX stream before any array exists, as
    // `AudioWorker` and `BatchScheduler::run` do.
    let stream = new_thread_local_generation_stream();
    install_thread_local_default_stream(stream.as_ref());

    let model = match catch_unwind(AssertUnwindSafe(loader)) {
        Ok(Ok(model)) => model,
        Ok(Err(message)) => {
            let _ = ready.send(Err(message));
            return;
        }
        Err(payload) => {
            let _ = ready.send(Err(crate::server::audio_worker::panic_message(
                payload.as_ref(),
                "model load",
            )));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    drop(ready);

    let mut active: Option<Active<'_>> = None;
    while let Ok(Message::Call {
        session_id,
        kind,
        reply,
    }) = commands.recv()
    {
        let is_close = matches!(kind, CommandKind::Session(SessionCommand::Close));
        // A panic on the model path must not take the thread (and with it
        // every later connection) down. The session it tore is dropped; the
        // model holds no cross-call state, so the next `open` starts clean.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            execute(&model, &mut active, &session_id, kind)
        }));
        synchronize_thread_local_stream(stream.as_ref());
        let result = outcome.unwrap_or_else(|payload| {
            active = None;
            let message = crate::server::audio_worker::panic_message(payload.as_ref(), "realtime");
            tracing::error!("realtime VoiceChat session panicked: {message}");
            Err(RealtimeEngineError::Panicked(message))
        });
        if let Err(err) = &result {
            tracing::warn!("realtime VoiceChat command failed: {err}");
        }
        if is_close {
            mlxcel_core::memory::clear_cache();
        }
        if let Some(reply) = reply {
            let _ = reply.send(result);
        }
    }
}

/// Run one command against the active session.
fn execute<'m, M: RealtimeModel>(
    model: &'m M,
    active: &mut Option<Active<'m>>,
    session_id: &str,
    kind: CommandKind,
) -> EngineResult<Reply> {
    let command = match kind {
        CommandKind::Open(config) => {
            if active.is_some() {
                return Err(RealtimeEngineError::AlreadyActive);
            }
            let session = model
                .open_session(&config)
                .map_err(RealtimeEngineError::Session)?;
            let info = session.info();
            *active = Some(Active {
                session_id: session_id.to_string(),
                session,
            });
            return Ok(Reply::Opened(info));
        }
        CommandKind::Session(command) => command,
    };
    let Some(current) = active.as_mut().filter(|a| a.session_id == session_id) else {
        return Err(RealtimeEngineError::NotActive);
    };
    let session = &mut current.session;
    let events = match command {
        SessionCommand::Push {
            samples,
            sample_rate,
        } => session
            .push_audio(&samples, sample_rate)
            .map_err(RealtimeEngineError::Session)?,
        SessionCommand::Flush { pad_partial } => session
            .flush(pad_partial)
            .map_err(RealtimeEngineError::Session)?,
        SessionCommand::Cancel => session.cancel(),
        SessionCommand::Close => {
            if !session.is_closed() {
                session.cancel();
            }
            *active = None;
            return Ok(Reply::Closed);
        }
    };
    Ok(Reply::Events(
        events.into_iter().map(serialize_event).collect(),
    ))
}

#[cfg(test)]
#[path = "realtime_engine_tests.rs"]
mod tests;
