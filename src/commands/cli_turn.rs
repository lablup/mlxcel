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
//! interrupted. Ctrl-C during a turn sets the request's cancellation flag, so
//! the worker stops that request and the process keeps running.

use std::io::{self, IsTerminal, Write as IoWrite};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
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
        if !self.show_reasoning
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

/// The cancellation flag of the request in flight, if any.
fn in_flight() -> &'static Mutex<Option<Arc<AtomicBool>>> {
    static IN_FLIGHT: OnceLock<Mutex<Option<Arc<AtomicBool>>>> = OnceLock::new();
    IN_FLIGHT.get_or_init(|| Mutex::new(None))
}

/// Install the process's Ctrl-C handling once: during a turn it cancels that
/// turn through the worker; outside a turn it exits with status 130, which is
/// what the default handler did. The line editor reads in raw mode, where
/// Ctrl-C reaches it as a key rather than a signal, so the REPL prompt keeps
/// its own `^C` handling.
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
                            None => std::process::exit(130),
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
