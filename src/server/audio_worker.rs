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
use std::time::Duration;

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
    /// Bounded command channel to the worker thread. Wrapped in a `Mutex` so the
    /// `AudioWorker` is `Sync` (an `mpsc::SyncSender` is `Send` but not `Sync`);
    /// the lock is held only for the brief enqueue, not across inference. The
    /// channel is bounded (admission control): `dispatch` uses `try_send`, so a
    /// full queue is rejected with [`AudioModelError::QueueFull`] rather than
    /// growing memory without bound (each queued command holds the full audio
    /// payload).
    sender: Mutex<mpsc::SyncSender<AudioCommand>>,
    /// Per-request reply timeout. `transcribe`/`synthesize` block on the reply
    /// channel for at most this long, then return [`AudioModelError::Timeout`].
    ///
    /// The timeout frees the caller's blocking `spawn_blocking` thread and
    /// returns a structured error; it does NOT cancel the in-flight MLX work on
    /// the worker (a single worker can only safely process one request at a
    /// time). When the worker eventually finishes, its reply send fails silently
    /// because the reply receiver was already dropped. This is intentional and
    /// matches the issue: a stuck request frees its thread instead of hanging.
    request_timeout: Duration,
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
    ///
    /// `queue_depth` bounds the command channel: at most `queue_depth` requests
    /// can wait behind the one in flight before admission is rejected with
    /// [`AudioModelError::QueueFull`]. `request_timeout` bounds how long a caller
    /// blocks for a reply before giving up with [`AudioModelError::Timeout`].
    pub(crate) fn spawn<E, L>(
        thread_name: &str,
        queue_depth: usize,
        request_timeout: Duration,
        loader: L,
    ) -> anyhow::Result<Self>
    where
        E: AudioEngine,
        L: FnOnce() -> anyhow::Result<E> + Send + 'static,
    {
        // A zero-capacity `sync_channel` is a rendezvous (no buffering), which is
        // not the admission semantics we want, so clamp to at least one queued
        // command. A `0` from config therefore falls back to a usable bound.
        let capacity = queue_depth.max(1);
        // A `Duration::ZERO` timeout would make `recv_timeout` return `Err(Timeout)`
        // before the worker thread can ever reply, rejecting every request. Fall back
        // to the documented default (120 s), mirroring the `queue_depth.max(1)` clamp
        // above so the worker is correct regardless of what the caller passes.
        let request_timeout = if request_timeout == Duration::ZERO {
            Duration::from_secs(crate::server::config::DEFAULT_AUDIO_REQUEST_TIMEOUT_SECS)
        } else {
            request_timeout
        };
        let (command_tx, command_rx) = mpsc::sync_channel::<AudioCommand>(capacity);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

        let handle = thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || worker_loop(loader, command_rx, ready_tx))
            .map_err(|e| anyhow::anyhow!("failed to spawn audio worker thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: Mutex::new(command_tx),
                request_timeout,
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

    /// Send a transcription request to the worker and block for its reply, up to
    /// the per-request timeout.
    pub(crate) fn transcribe(
        &self,
        input: AudioTranscribeInput,
    ) -> Result<AudioTranscribeOutput, AudioModelError> {
        let (respond, reply) = mpsc::channel();
        self.dispatch(AudioCommand::Transcribe { input, respond })?;
        match reply.recv_timeout(self.request_timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(AudioModelError::Timeout),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(AudioModelError::Inference(WORKER_GONE.to_string()))
            }
        }
    }

    /// Send a synthesis request to the worker and block for its reply, up to the
    /// per-request timeout.
    pub(crate) fn synthesize(
        &self,
        input: AudioSynthesizeInput,
    ) -> Result<AudioSynthesizeOutput, AudioModelError> {
        let (respond, reply) = mpsc::channel();
        self.dispatch(AudioCommand::Synthesize { input, respond })?;
        match reply.recv_timeout(self.request_timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(AudioModelError::Timeout),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(AudioModelError::Inference(WORKER_GONE.to_string()))
            }
        }
    }

    /// Enqueue a command without blocking, releasing the sender lock before the
    /// caller blocks on its reply channel so concurrent requests queue in the
    /// channel rather than serialize on the lock.
    ///
    /// `try_send` is the admission gate: a full bounded queue is rejected with
    /// [`AudioModelError::QueueFull`] (load shedding) instead of blocking the
    /// caller or letting queued payloads grow without bound. A disconnected
    /// channel means the worker thread has gone, reported as before.
    fn dispatch(&self, command: AudioCommand) -> Result<(), AudioModelError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| AudioModelError::Inference("audio worker channel poisoned".to_string()))?;
        match sender.try_send(command) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => Err(AudioModelError::QueueFull),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(AudioModelError::Inference(WORKER_GONE.to_string()))
            }
        }
    }
}

impl Drop for AudioWorker {
    fn drop(&mut self) {
        // Ask the loop to stop, then wait for the thread so the engine's MLX
        // handles are dropped on the thread that created them. Both steps are
        // best-effort: if the worker already exited, the send fails and the join
        // observes the prior exit, neither of which should escape a destructor.
        //
        // This is a blocking `SyncSender::send` (not `try_send`): on a full
        // bounded queue it waits for capacity, which is correct here because the
        // worker keeps draining commands, so a slot frees and the shutdown is
        // delivered after the queued work.
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
    //
    // Each engine call runs under `catch_unwind` so one panicking request cannot
    // tear down the worker thread. Without this boundary a panic on the
    // attacker-reachable forward path (the synthesis helpers carry many
    // `expect`/`unwrap` calls) would unwind out of the loop, close the channel,
    // and make every later audio request fail with `WORKER_GONE` for the rest of
    // the process: a permanent denial of service from a single bad request.
    // Recovering here turns that into a clean per-request `Inference` error while
    // the thread keeps serving.
    //
    // `&mut engine` is not `UnwindSafe`, so we assert it at this supervised
    // boundary. That is sound: the engine is owned and used single-threadedly on
    // this thread and is the only state the closure touches, the MLX graph
    // builders hold no cross-call invariants, so a recovered panic cannot leave
    // shared state observably torn for the next command. We always synchronize
    // the stream afterwards (panic or not) because a partially built or evaluated
    // graph may have left work queued, then send the reply: run -> sync -> send.
    while let Ok(command) = commands.recv() {
        match command {
            AudioCommand::Transcribe { input, respond } => {
                let result = run_guarded("transcription", stream.as_ref(), || {
                    engine.transcribe(input)
                });
                let _ = respond.send(result);
            }
            AudioCommand::Synthesize { input, respond } => {
                let result = run_guarded("synthesis", stream.as_ref(), || engine.synthesize(input));
                let _ = respond.send(result);
            }
            AudioCommand::Shutdown => break,
        }
    }
}

/// Run one engine call under a panic boundary, then synchronize this thread's
/// MLX stream, returning the call's `Result` or a recovered-panic error.
///
/// `label` names the direction ("transcription"/"synthesis") for the log and
/// fallback message; it never carries request input. The happy path forwards
/// the engine's `Ok`/`Err` unchanged, so non-panicking behavior is identical to
/// calling the engine directly. On a caught panic the payload string (if any) is
/// surfaced as [`AudioModelError::Inference`].
fn run_guarded<T, F>(
    label: &str,
    stream: Option<&mlxcel_core::UniquePtr<mlxcel_core::MlxThreadLocalStream>>,
    call: F,
) -> Result<T, AudioModelError>
where
    F: FnOnce() -> Result<T, AudioModelError>,
{
    // `AssertUnwindSafe`: see the supervised-boundary reasoning at the call site.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call));
    // Synchronize whether or not the call panicked: a partially built/evaluated
    // graph may have queued work on this stream that must drain before the next
    // command reuses it.
    synchronize_thread_local_stream(stream);
    match outcome {
        Ok(result) => result,
        Err(payload) => {
            let detail = panic_message(payload.as_ref(), label);
            // A recovered panic is a real fault worth surfacing; log it without
            // any request input content.
            tracing::error!(target: "mlxcel::audio", "audio worker recovered from panic during {label}: {detail}");
            Err(AudioModelError::Inference(format!(
                "audio worker recovered from panic: {detail}"
            )))
        }
    }
}

/// Extract a human-readable message from a caught panic payload, falling back to
/// a generic per-direction message when the payload is not a string.
fn panic_message(payload: &(dyn std::any::Any + Send), label: &str) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        format!("{label} panicked")
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
        let worker = AudioWorker::spawn("audio-test-roundtrip", 8, Duration::from_secs(30), || {
            Ok(TrivialEngine)
        })
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
        let result = AudioWorker::spawn::<TrivialEngine, _>(
            "audio-test-loadfail",
            8,
            Duration::from_secs(30),
            || Err(anyhow::anyhow!("synthetic load failure")),
        );
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

        let worker = AudioWorker::spawn("audio-test-default", 8, Duration::from_secs(30), || {
            Ok(NoneEngine)
        })
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

    /// Engine that panics on a sentinel input and succeeds otherwise. Used to
    /// prove the worker thread survives a panicking request instead of dying and
    /// failing every later request with `WORKER_GONE`.
    struct PanicEngine;

    impl AudioEngine for PanicEngine {
        fn transcribe(
            &mut self,
            input: AudioTranscribeInput,
        ) -> Result<AudioTranscribeOutput, AudioModelError> {
            if input.language.as_deref() == Some("panic") {
                panic!("synthetic transcribe panic");
            }
            // Touch MLX on the worker thread like `TrivialEngine`, so the success
            // path exercises the same stream the panic path left to synchronize.
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

    // Since issue #375 this test also represents release-build behavior. The
    // test harness always builds with `panic = "unwind"` regardless of the
    // profile, so `catch_unwind` in `run_guarded` has always worked here; the
    // release profile now also uses `panic = "unwind"`, so the release binary
    // contains the worker survives, server keeps serving behavior this asserts
    // (rather than aborting the process as it did under the former
    // `panic = "abort"`).
    #[test]
    fn worker_survives_engine_panic_and_keeps_serving() {
        let worker = AudioWorker::spawn("audio-test-panic", 8, Duration::from_secs(30), || {
            Ok(PanicEngine)
        })
        .expect("worker spawns and engine loads");

        // First request hits the panic path. The worker must recover and reply
        // with an `Inference` error, NOT die and surface `WORKER_GONE`. (The
        // default panic hook prints the panic to stderr during this call; that
        // is expected and does not fail the test.)
        let err = worker
            .transcribe(transcribe_input(7, Some("panic")))
            .expect_err("panicking request returns an error");
        match err {
            AudioModelError::Inference(message) => {
                assert_ne!(
                    message, WORKER_GONE,
                    "a recovered panic must not be reported as a dead worker"
                );
                assert!(
                    message.contains("panic"),
                    "recovered-panic error should mention the panic, got: {message}"
                );
            }
            other => panic!("expected an Inference error, got: {other:?}"),
        }

        // The key assertion: a normal request on the SAME worker still succeeds,
        // proving the thread survived the earlier panic.
        let out = worker
            .transcribe(transcribe_input(4, Some("en")))
            .expect("worker still serves after recovering from a panic");
        assert_eq!(out.text, "ok:4");
        assert_eq!(out.language.as_deref(), Some("en"));
    }

    /// A bounded command queue rejects admission once it is full. The single
    /// worker is held busy on an in-flight request so the buffer never drains;
    /// each buffered direct call returns via the per-request timeout and its
    /// command stays buffered, so after `queue_depth` slots fill the next
    /// admission is rejected with `QueueFull`.
    #[test]
    fn queue_full_returns_queuefull_error() {
        // Engine that blocks the worker thread on the first request until the
        // test opens the gate. Only the first pulled command reaches the gate;
        // buffered commands are never pulled while the worker is blocked here.
        struct BlockingEngine {
            gate: Option<mpsc::Receiver<()>>,
            started: mpsc::Sender<()>,
        }
        impl AudioEngine for BlockingEngine {
            fn transcribe(
                &mut self,
                input: AudioTranscribeInput,
            ) -> Result<AudioTranscribeOutput, AudioModelError> {
                if let Some(gate) = self.gate.take() {
                    let _ = self.started.send(());
                    let _ = gate.recv();
                }
                Ok(AudioTranscribeOutput {
                    text: format!("ok:{}", input.audio.len()),
                    language: input.language,
                    duration_seconds: None,
                })
            }
        }

        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let (started_tx, started_rx) = mpsc::channel::<()>();

        let queue_depth = 2;
        let worker = AudioWorker::spawn(
            "audio-test-queuefull",
            queue_depth,
            Duration::from_millis(150),
            move || {
                Ok(BlockingEngine {
                    gate: Some(gate_rx),
                    started: started_tx,
                })
            },
        )
        .expect("worker spawns and engine loads");

        std::thread::scope(|scope| {
            // Occupy the single worker with one in-flight request. This call
            // gives up after the per-request timeout, but the worker stays busy
            // inside the engine until the gate opens, so the buffer never drains.
            scope.spawn(|| {
                let _ = worker.transcribe(transcribe_input(1, None));
            });

            // Wait until the worker has pulled the in-flight request and is
            // blocked in the engine: now the buffer is empty and the worker busy.
            started_rx
                .recv()
                .expect("engine signals it began the in-flight request");

            // Fill the bounded buffer, then prove the next admission is rejected.
            let mut saw_queue_full = false;
            for _ in 0..(queue_depth + 1) {
                match worker.transcribe(transcribe_input(1, None)) {
                    Err(AudioModelError::QueueFull) => {
                        saw_queue_full = true;
                        break;
                    }
                    // The slot filled; the command stays buffered. Keep going.
                    Err(AudioModelError::Timeout) => continue,
                    other => panic!("unexpected result while filling the queue: {other:?}"),
                }
            }
            assert!(
                saw_queue_full,
                "a full bounded queue must reject admission with QueueFull"
            );

            // Open the gate so the worker drains and the scope thread can finish.
            drop(gate_tx);
        });
    }

    /// A request the worker cannot finish within the per-request timeout returns
    /// `Timeout` (freeing the caller's blocking thread), not `WORKER_GONE`.
    #[test]
    fn request_timeout_returns_timeout_error() {
        struct SlowEngine;
        impl AudioEngine for SlowEngine {
            fn transcribe(
                &mut self,
                input: AudioTranscribeInput,
            ) -> Result<AudioTranscribeOutput, AudioModelError> {
                // Sleep well past the worker's per-request timeout. This runs on
                // the worker thread (a plain std thread, not async), so a
                // blocking sleep is the right tool to model a slow request.
                std::thread::sleep(Duration::from_millis(500));
                Ok(AudioTranscribeOutput {
                    text: format!("ok:{}", input.audio.len()),
                    language: input.language,
                    duration_seconds: None,
                })
            }
        }

        let worker = AudioWorker::spawn("audio-test-timeout", 8, Duration::from_millis(50), || {
            Ok(SlowEngine)
        })
        .expect("worker spawns and engine loads");

        let err = worker
            .transcribe(transcribe_input(3, None))
            .expect_err("a request slower than the timeout must error");
        assert!(
            matches!(err, AudioModelError::Timeout),
            "expected Timeout, got: {err:?}"
        );
    }

    /// `Duration::ZERO` passed to `spawn` must not make every request time out
    /// immediately. The guard inside `spawn` clamps it to the documented default
    /// so a fast request still completes.
    #[test]
    fn zero_request_timeout_falls_back_to_default() {
        let worker = AudioWorker::spawn("audio-test-zero-timeout", 8, Duration::ZERO, || {
            Ok(TrivialEngine)
        })
        .expect("worker spawns with a zero timeout arg");

        // If Duration::ZERO were used as-is, recv_timeout(ZERO) would return
        // Err(Timeout) before the worker thread could reply. The guard inside
        // spawn clamps it, so a fast request completes normally.
        let out = worker
            .transcribe(transcribe_input(4, None))
            .expect("zero timeout falls back to default; fast request completes");
        assert_eq!(out.text, "ok:4");
    }
}
