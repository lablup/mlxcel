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

//! Terminal side of one in-process server turn, shared by `mlxcel run -p` and
//! the chat REPL (issue #2173).
//!
//! The server splits the stream into content and reasoning (its
//! `StreamFilter`); this module only decides what reaches the terminal:
//! content always, reasoning dimmed on a TTY when `--show-reasoning` is set,
//! and a one-line notice when a turn ended inside the reasoning channel or was
//! interrupted. It also holds the base-model notice both front ends print
//! after loading a checkpoint without a chat template. Ctrl-C during a turn sets the request's cancellation flag, so
//! the worker stops that request and the process keeps running.

use std::io::{self, IsTerminal, Write as IoWrite};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use mlxcel::server::in_process::{ChatDelta, ChatTurn, InProcessServer};

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// How a streamed turn is shown.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TurnDisplay {
    /// Print the reasoning channel too (dimmed on a TTY).
    pub(crate) show_reasoning: bool,
}

/// Prints deltas as they arrive and remembers whether any visible text went
/// out, so an all-reasoning turn can say so instead of looking hung.
pub(crate) struct TurnPrinter {
    show_reasoning: bool,
    dim: bool,
    saw_visible_text: bool,
    stdout: io::Stdout,
}

impl TurnPrinter {
    pub(crate) fn new(display: TurnDisplay) -> Self {
        let stdout = io::stdout();
        Self {
            show_reasoning: display.show_reasoning,
            dim: stdout.is_terminal(),
            saw_visible_text: false,
            stdout,
        }
    }

    pub(crate) fn on_delta(&mut self, delta: ChatDelta<'_>) {
        let (text, dimmed) = match delta {
            ChatDelta::Content(text) => {
                self.saw_visible_text |= !text.trim().is_empty();
                (text, false)
            }
            ChatDelta::Reasoning(text) if self.show_reasoning => {
                self.saw_visible_text |= !text.trim().is_empty();
                (text, self.dim)
            }
            ChatDelta::Reasoning(_) => return,
        };
        // The slice goes straight to the locked stdout, then a flush so the
        // text shows as it streams; stdout is line-buffered, so without the
        // flush a delta with no newline would wait for the next one.
        let mut out = self.stdout.lock();
        let _ = write_delta(&mut out, text, dimmed);
        let _ = out.flush();
    }

    /// The tail notices for a finished turn.
    pub(crate) fn finish(&self, turn: &ChatTurn, restart_hint: &str) {
        println!();
        if turn.cancelled {
            println!("[interrupted]");
        }
        if !turn.cancelled
            && !self.show_reasoning
            && !self.saw_visible_text
            && turn
                .reasoning
                .as_deref()
                .is_some_and(|r| !r.trim().is_empty())
        {
            println!();
            println!(
                "[All {} generated tokens went to the reasoning channel; the content channel is empty. {restart_hint} with --show-reasoning to see them.]",
                turn.result.completion_tokens
            );
        }
        for call in &turn.tool_calls {
            println!("[tool call] {}({})", call.name, call.arguments);
        }
    }
}

/// Write one delta's text, wrapped in the dim escape when `dimmed`.
fn write_delta<W: IoWrite>(out: &mut W, text: &str, dimmed: bool) -> io::Result<()> {
    if dimmed {
        out.write_all(DIM.as_bytes())?;
        out.write_all(text.as_bytes())?;
        out.write_all(RESET.as_bytes())
    } else {
        out.write_all(text.as_bytes())
    }
}

/// Say so when the checkpoint ships no chat template: it is likely a base
/// model, and the server renders turns with its generic default format.
/// Shared by the REPL and `mlxcel run -p`. The answer comes from the template
/// and tokenizer the in-process server already loaded; nothing is re-read
/// from the model directory.
pub(crate) fn warn_if_base_model(server: &InProcessServer, no_chat_template: bool) {
    if !base_model_notice_due(no_chat_template, || server.uses_generic_chat_template()) {
        return;
    }
    eprintln!(
        "Note: this model ships no chat template and is likely a base (non-instruction-tuned) model."
    );
    eprintln!(
        "      Chat responses will likely be incoherent or repetitive. Try an instruction-tuned"
    );
    eprintln!(
        "      variant (Gemma \"-it\", Llama and Qwen2.5 \"-Instruct\", Qwen3 without \"-Base\")."
    );
    eprintln!(
        "      Turns are rendered with the server's generic chat format; for raw text without"
    );
    eprintln!("      role markers, pass --no-chat-template.");
    eprintln!();
}

/// Whether the base-model notice is due. The generic-template question is a
/// closure so it is asked only when `--no-chat-template` is not set, and a
/// checkpoint with a template (the common case) costs one boolean.
fn base_model_notice_due(
    no_chat_template: bool,
    uses_generic_template: impl FnOnce() -> bool,
) -> bool {
    !no_chat_template && uses_generic_template()
}

/// Join the worker, then hand back the error that ended the command. A turn
/// error would otherwise leave through `?` with the worker still holding the
/// model, and the process could exit mid-teardown. A shutdown failure is
/// reported on stderr and does not replace the original error.
pub(crate) fn shutdown_then_fail<T>(server: InProcessServer, err: anyhow::Error) -> Result<T> {
    if let Err(shutdown_err) = server.shutdown() {
        eprintln!("warning: {shutdown_err:#}");
    }
    Err(err)
}

/// The cancellation flag of the request in flight, if any.
fn in_flight() -> &'static Mutex<Option<Arc<AtomicBool>>> {
    static IN_FLIGHT: OnceLock<Mutex<Option<Arc<AtomicBool>>>> = OnceLock::new();
    IN_FLIGHT.get_or_init(|| Mutex::new(None))
}

/// Die of SIGINT the way the default disposition does, without running
/// `atexit` handlers or static destructors on this signal thread while the
/// model worker may still hold the GPU. On Unix the default handler is
/// restored and the signal re-raised, so the parent sees a SIGINT death.
/// Windows' default Ctrl-C handling is itself an orderly process exit, so
/// there the process exits with status 130.
fn terminate_on_interrupt() -> ! {
    #[cfg(unix)]
    {
        // SAFETY: `signal` and `raise` are async-signal-safe libc calls with
        // no Rust-side invariants; resetting SIGINT to `SIG_DFL` and raising
        // it terminates the process as an unhandled Ctrl-C does.
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::raise(libc::SIGINT);
        }
    }
    // Reached only where the re-raise is unavailable or did not terminate.
    // SAFETY: `_exit` ends the process without running exit handlers, which
    // is the point; no Rust state is touched afterwards.
    #[cfg(unix)]
    unsafe {
        libc::_exit(130)
    }
    #[cfg(not(unix))]
    std::process::exit(130)
}

/// What one Ctrl-C does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterruptAction {
    /// The first Ctrl-C of a turn: the flag is now set and the worker stops
    /// that request.
    Cancel,
    /// No turn is running, or the turn is already cancelling and the user
    /// pressed Ctrl-C again: end the process.
    Terminate,
}

/// Apply a Ctrl-C to the in-flight turn's flag. A second Ctrl-C while the
/// flag is already set means the cancel is not taking effect (a long prefill
/// that does not poll it, a stuck worker), so it ends the process instead of
/// being swallowed.
fn interrupt_action(in_flight: Option<&AtomicBool>) -> InterruptAction {
    match in_flight {
        Some(cancel) if !cancel.swap(true, Ordering::AcqRel) => InterruptAction::Cancel,
        _ => InterruptAction::Terminate,
    }
}

/// Install the process's Ctrl-C handling once: during a turn it cancels that
/// turn through the worker (a second Ctrl-C of the same turn terminates the
/// process); outside a turn it terminates the process as the default handler
/// did ([`terminate_on_interrupt`]). The line editor reads in
/// raw mode, where Ctrl-C reaches it as a key rather than a signal, so the
/// REPL prompt keeps its own `^C` handling.
fn install_interrupt_handler() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let spawned = std::thread::Builder::new()
            .name("mlxcel-ctrl-c".to_string())
            .spawn(|| {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                runtime.block_on(async {
                    while tokio::signal::ctrl_c().await.is_ok() {
                        let flag = in_flight().lock().ok().and_then(|guard| guard.clone());
                        if interrupt_action(flag.as_deref()) == InterruptAction::Terminate {
                            terminate_on_interrupt();
                        }
                    }
                });
            });
        if let Err(err) = spawned {
            eprintln!("warning: Ctrl-C will end the process, not the turn ({err})");
        }
    });
}

/// Run `turn` with a fresh cancellation flag that Ctrl-C sets while it runs.
pub(crate) fn run_cancellable<T>(turn: impl FnOnce(Arc<AtomicBool>) -> Result<T>) -> Result<T> {
    install_interrupt_handler();
    let cancel = Arc::new(AtomicBool::new(false));
    if let Ok(mut guard) = in_flight().lock() {
        *guard = Some(cancel.clone());
    }
    let outcome = turn(cancel);
    if let Ok(mut guard) = in_flight().lock() {
        *guard = None;
    }
    outcome
}

#[cfg(test)]
#[path = "cli_turn_tests.rs"]
mod tests;
