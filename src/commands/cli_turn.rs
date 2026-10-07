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
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use mlxcel::server::chat_template::ChatTemplateProcessor;
use mlxcel::server::in_process::{ChatDelta, ChatTurn};

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
        let text = match delta {
            ChatDelta::Content(text) => {
                self.saw_visible_text |= !text.trim().is_empty();
                text.to_string()
            }
            ChatDelta::Reasoning(text) if self.show_reasoning => {
                self.saw_visible_text |= !text.trim().is_empty();
                if self.dim {
                    format!("{DIM}{text}{RESET}")
                } else {
                    text.to_string()
                }
            }
            ChatDelta::Reasoning(_) => return,
        };
        print!("{text}");
        let _ = self.stdout.flush();
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

/// Say so when the checkpoint ships no chat template: it is likely a base
/// model, and the server renders turns with its generic default format.
/// Shared by the REPL and `mlxcel run -p`.
pub(crate) fn warn_if_base_model(model_path: &Path, no_chat_template: bool) {
    if no_chat_template {
        return;
    }
    let has_template = ChatTemplateProcessor::from_model_path(model_path)
        .ok()
        .flatten()
        .is_some();
    let native = mlxcel::tokenizer::load_tokenizer(model_path)
        .ok()
        .is_some_and(|tokenizer| tokenizer.kimi_k3_control_ids().is_some());
    if has_template || native {
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

/// Install the process's Ctrl-C handling once: during a turn it cancels that
/// turn through the worker; outside a turn it terminates the process as the
/// default handler did ([`terminate_on_interrupt`]). The line editor reads in
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
                        match flag {
                            Some(cancel) => cancel.store(true, Ordering::Release),
                            None => terminate_on_interrupt(),
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
