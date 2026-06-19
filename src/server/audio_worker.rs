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

//! Single dedicated-thread worker for audio models.
//!
//! MLX work is thread-affine. The thread that creates a model's weight arrays
//! must also be the thread that evaluates them, because loading initializes the
//! full per-thread MLX scheduler context: the GPU stream this worker installs
//! plus the CPU stream that decode-time ops and scalar readback resolve on. The
//! generation path already follows this rule, every model is loaded and run on
//! one owned thread by `BatchScheduler`, which installs a thread-local default
//! stream once at the top of its loop. The audio path mirrors that here so a
//! transcription (or future synthesis) request never evaluates an MLX graph on
//! a thread that did not load the model. Evaluating on a foreign thread fails
//! with `There is no Stream(cpu, 0) in current thread`.
//!
//! [`AudioWorker`] owns that thread. It installs a thread-local default stream,
//! loads an [`AudioEngine`] on itself, then serves [`AudioCommand`]s from a
//! channel one at a time. The HTTP routes only send a command and block for the
//! reply, so the `spawn_blocking` pool thread that calls
//! [`AudioWorker::transcribe`] or [`AudioWorker::synthesize`] does no MLX work.
//!
//! The worker is engine-agnostic. The speech-to-text provider supplies a
//! Whisper engine today; the text-to-speech provider can supply its own engine
//! over the same channel and thread-ownership pattern.

use std::sync::Mutex;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
    synchronize_thread_local_stream,
};

use crate::server::audio_model::{
    AudioModelError, AudioModelKind, AudioSynthesizeInput, AudioSynthesizeOutput,
    AudioTranscribeInput, AudioTranscribeOutput,
};

/// Reported when a request cannot reach the worker thread because it has
/// already exited (a load that failed late, or a panic mid-request).
const WORKER_GONE: &str = "audio worker thread is no longer running";

/// Model logic that runs exclusively on an [`AudioWorker`]'s dedicated thread.
///
/// Implementors hold MLX array handles directly, with no `Mutex` and no
/// `unsafe impl Send`, because every call arrives on the single worker thread
/// that constructed them. A provider that services only one direction overrides
/// just that method and leaves the other with its default "kind not loaded"
/// body.
pub(crate) trait AudioEngine: 'static {
    /// Run speech-to-text. Default: this engine does not transcribe.
    fn transcribe(
        &mut self,
        _input: AudioTranscribeInput,
    ) -> Result<AudioTranscribeOutput, AudioModelError> {
        Err(AudioModelError::KindNotLoaded(AudioModelKind::Stt))
    }

    /// Run text-to-speech. Default: this engine does not synthesize.
    fn synthesize(
        &mut self,
        _input: AudioSynthesizeInput,
    ) -> Result<AudioSynthesizeOutput, AudioModelError> {
        Err(AudioModelError::KindNotLoaded(AudioModelKind::Tts))
    }
}

/// A unit of work plus the channel to reply on, handed to the worker thread.
enum AudioCommand {
    /// Speech-to-text request and its reply channel.
    Transcribe {
        input: AudioTranscribeInput,
        respond: mpsc::Sender<Result<AudioTranscribeOutput, AudioModelError>>,
    },
    /// Text-to-speech request and its reply channel.
    Synthesize {
        input: AudioSynthesizeInput,
        respond: mpsc::Sender<Result<AudioSynthesizeOutput, AudioModelError>>,
    },
    /// Graceful stop: break the loop so the thread can join.
    Shutdown,
}

/// Owns a dedicated thread that loads and runs one [`AudioEngine`].
///
/// The handle is `Send + Sync` even though the engine and its MLX arrays are
/// not, because the engine never leaves the worker thread: only command and
/// reply values (all `Send`) cross the channel.
#[derive(Debug)]
pub(crate) struct AudioWorker {
    /// Command channel to the worker thread. Wrapped in a `Mutex` so the
    /// `AudioWorker` is `Sync` (an `mpsc::Sender` is `Send` but not `Sync`); the
    /// lock is held only for the brief enqueue, not across inference.
    sender: Mutex<mpsc::Sender<AudioCommand>>,
    /// Join handle, taken in `Drop` after the loop is asked to stop.
    handle: Option<JoinHandle<()>>,
}

impl AudioWorker {
    /// Spawn the worker thread and block until the engine has loaded.
    ///
    /// `loader` runs on the worker thread, so every array it creates belongs to
    /// that thread's MLX context. Returns `Err` if the thread cannot be spawned
    /// or the engine fails to load, so the caller can leave the audio slot empty
    /// and keep serving rather than aborting startup.
    pub(crate) fn spawn<E, L>(thread_name: &str, loader: L) -> anyhow::Result<Self>
    where
        E: AudioEngine,
        L: FnOnce() -> anyhow::Result<E> + Send + 'static,
    {
        let (command_tx, command_rx) = mpsc::channel::<AudioCommand>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

        let handle = thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || worker_loop(loader, command_rx, ready_tx))
            .map_err(|e| anyhow::anyhow!("failed to spawn audio worker thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: Mutex::new(command_tx),
                handle: Some(handle),
            }),
            Ok(Err(message)) => {
                let _ = handle.join();
                Err(anyhow::anyhow!(
                    "audio worker failed to load model: {message}"
                ))
            }
            Err(_) => {
                let _ = handle.join();
                Err(anyhow::anyhow!(
                    "audio worker thread exited before reporting readiness"
                ))
            }
        }
    }

    /// Send a transcription request to the worker and block for its reply.
    pub(crate) fn transcribe(
        &self,
        input: AudioTranscribeInput,
    ) -> Result<AudioTranscribeOutput, AudioModelError> {
        let (respond, reply) = mpsc::channel();
        self.dispatch(AudioCommand::Transcribe { input, respond })?;
        match reply.recv() {
            Ok(result) => result,
            Err(_) => Err(AudioModelError::Inference(WORKER_GONE.to_string())),
        }
    }

    /// Send a synthesis request to the worker and block for its reply.
    pub(crate) fn synthesize(
        &self,
        input: AudioSynthesizeInput,
    ) -> Result<AudioSynthesizeOutput, AudioModelError> {
        let (respond, reply) = mpsc::channel();
        self.dispatch(AudioCommand::Synthesize { input, respond })?;
        match reply.recv() {
            Ok(result) => result,
            Err(_) => Err(AudioModelError::Inference(WORKER_GONE.to_string())),
        }
    }

    /// Enqueue a command, releasing the sender lock before the caller blocks on
    /// its reply channel so concurrent requests queue in the channel rather than
    /// serialize on the lock.
    fn dispatch(&self, command: AudioCommand) -> Result<(), AudioModelError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| AudioModelError::Inference("audio worker channel poisoned".to_string()))?;
        sender
            .send(command)
            .map_err(|_| AudioModelError::Inference(WORKER_GONE.to_string()))
    }
}

impl Drop for AudioWorker {
    fn drop(&mut self) {
        // Ask the loop to stop, then wait for the thread so the engine's MLX
        // handles are dropped on the thread that created them. Both steps are
        // best-effort: if the worker already exited, the send fails and the join
        // observes the prior exit, neither of which should escape a destructor.
        if let Ok(sender) = self.sender.lock() {
            let _ = sender.send(AudioCommand::Shutdown);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Body of the worker thread: install the stream, load the engine, then serve
/// commands until asked to stop or the channel closes.
fn worker_loop<E, L>(
    loader: L,
    commands: mpsc::Receiver<AudioCommand>,
    ready: mpsc::Sender<Result<(), String>>,
) where
    E: AudioEngine,
    L: FnOnce() -> anyhow::Result<E>,
{
    // (a) Install this thread's default MLX stream, exactly as
    // `BatchScheduler::run` does before it touches any array.
    let stream = new_thread_local_generation_stream();
    install_thread_local_default_stream(stream.as_ref());

    // (b) Load the engine on this thread so its weight arrays live in this
    // thread's MLX context. Report the outcome so `spawn` can return it.
    let mut engine = match loader() {
        Ok(engine) => engine,
        Err(err) => {
            let _ = ready.send(Err(err.to_string()));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        // The constructor stopped waiting; there is nobody to serve.
        return;
    }
    drop(ready);

    // (c) Serve one command at a time. After each, synchronize so dispatch and
    // synchronization stay paired on this thread's stream.
    while let Ok(command) = commands.recv() {
        match command {
            AudioCommand::Transcribe { input, respond } => {
                let result = engine.transcribe(input);
                synchronize_thread_local_stream(stream.as_ref());
                let _ = respond.send(result);
            }
            AudioCommand::Synthesize { input, respond } => {
                let result = engine.synthesize(input);
                synchronize_thread_local_stream(stream.as_ref());
                let _ = respond.send(result);
            }
            AudioCommand::Shutdown => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Engine that performs a trivial MLX op on the worker thread, proving the
    /// load-and-run-on-one-thread round-trip works end to end.
    struct TrivialEngine;

    impl AudioEngine for TrivialEngine {
        fn transcribe(
            &mut self,
            input: AudioTranscribeInput,
        ) -> Result<AudioTranscribeOutput, AudioModelError> {
            // Dispatch and force-evaluate a tiny array on this (worker) thread.
            // The fallible `try_eval` boundary keeps an MLX failure from
            // aborting the process, mirroring the real Whisper decode path.
            let arr = mlxcel_core::from_slice_f32(&[1.0_f32, 2.0, 3.0], &[3]);
            mlxcel_core::try_eval(&arr)
                .map_err(|e| AudioModelError::Inference(format!("eval failed: {e}")))?;
            Ok(AudioTranscribeOutput {
                text: format!("ok:{}", input.audio.len()),
                language: input.language,
                duration_seconds: None,
            })
        }
    }

    fn transcribe_input(len: usize, language: Option<&str>) -> AudioTranscribeInput {
        AudioTranscribeInput {
            audio: vec![0u8; len],
            filename: None,
            language: language.map(str::to_string),
            temperature: None,
            translate: false,
        }
    }

    #[test]
    fn worker_round_trip_runs_mlx_op_and_replies() {
        let worker = AudioWorker::spawn("audio-test-roundtrip", || Ok(TrivialEngine))
            .expect("worker spawns and engine loads");

        let out = worker
            .transcribe(transcribe_input(5, Some("en")))
            .expect("transcribe round-trips");
        assert_eq!(out.text, "ok:5");
        assert_eq!(out.language.as_deref(), Some("en"));

        // A second request reuses the same thread and the same installed stream.
        let out2 = worker
            .transcribe(transcribe_input(2, None))
            .expect("second transcribe round-trips");
        assert_eq!(out2.text, "ok:2");
        assert!(out2.language.is_none());
    }

    #[test]
    fn worker_surfaces_loader_failure() {
        let result = AudioWorker::spawn::<TrivialEngine, _>("audio-test-loadfail", || {
            Err(anyhow::anyhow!("synthetic load failure"))
        });
        let err = result.expect_err("loader failure propagates from spawn");
        assert!(
            err.to_string().contains("synthetic load failure"),
            "expected the loader error to be surfaced, got: {err}"
        );
    }

    #[test]
    fn default_engine_reports_unsupported_direction() {
        struct NoneEngine;
        impl AudioEngine for NoneEngine {}

        let worker = AudioWorker::spawn("audio-test-default", || Ok(NoneEngine))
            .expect("worker spawns with a do-nothing engine");
        let err = worker
            .synthesize(AudioSynthesizeInput {
                input: "hi".to_string(),
                voice: None,
                speed: None,
            })
            .expect_err("default synthesize is unsupported");
        assert!(matches!(
            err,
            AudioModelError::KindNotLoaded(AudioModelKind::Tts)
        ));
    }
}
