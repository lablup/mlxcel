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

//! Interactive multi-turn chat REPL (epic #92, issue #96), a client of the
//! in-process model server since epic #2166 Phase 5 (issue #2173).
//!
//! A line-edited, streaming chat loop in the spirit of `mlx_lm.chat` /
//! `ollama run`. The REPL keeps the terminal and the transcript; everything
//! about generation is the server's:
//!
//! * model resolution: [`mlxcel::downloader::resolve_model_source_with_override`]
//!   (the `-m`-accepts-a-repo-id resolver, issue #94);
//! * the model worker: [`InProcessServer`], started at one slot with the CLI's
//!   sampling flags as its defaults ([`super::cli_server`]);
//! * each turn: [`InProcessServer::chat`], the `/v1/chat/completions` request
//!   path (chat template, Kimi K3's native renderer, worker options, the
//!   server's `StreamFilter` for reasoning, tool-call parsing, stop strings);
//! * `--no-chat-template`: [`InProcessServer::complete`], the
//!   `/v1/completions` path over the raw transcript.
//!
//! ## Multi-turn context
//!
//! The whole transcript is sent every turn, as a chat client sends it. The
//! server's prompt cache (on for the REPL, the server default) reuses the KV
//! of the history prefix the previous turn left, so a follow-up prefills only
//! the new messages. Each conversation carries its own `prompt_cache_key`;
//! `/clear` empties the transcript and moves to a new key, so nothing cached
//! for the old conversation is adopted again.
//!
//! ## Interrupting a reply
//!
//! Ctrl-C while a reply streams cancels that request through the worker's
//! cancellation path (the one a disconnected HTTP client uses) and returns to
//! the prompt; the interrupted exchange is dropped from the transcript.
//! Ctrl-C at the prompt cancels the line, as before.

use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, anyhow};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use serde_json::json;

use mlxcel::initialize_runtime_checked;
use mlxcel::server::chat_template::{ChatMessage, ChatTemplateProcessor};
use mlxcel::server::in_process::InProcessServer;
use mlxcel::server::in_process::chat::{chat_request_from_json, completion_request_from_json};

use super::chat_transcript::{Turn, messages_json, transcript_messages};
use super::cli_turn::{TurnDisplay, TurnPrinter, run_cancellable};
use mlxcel::cli::in_process_client::CliServerSettings;

/// Triple-quote fence that opens / closes an ollama-style multiline input
/// block.
const MULTILINE_FENCE: &str = "\"\"\"";

/// Self-contained configuration for the interactive chat REPL, built by
/// `mlxcel run` and by `mlxcel generate` without `-p`.
#[derive(Debug, Clone)]
pub struct ChatOptions {
    /// Local model directory **or** a HuggingFace `owner/name` repo-id,
    /// resolved like `generate -m`. A bare name without a slash resolves as
    /// `mlx-community/<name>` (override the org with `MLXCEL_DEFAULT_ORG`).
    pub model: PathBuf,
    /// Model-store root override (`--models-dir`, issue #107).
    pub models_dir: Option<PathBuf>,
    /// Repository revision override (`--revision`, issue #1113).
    pub revision: Option<String>,
    /// What the in-process server is started with: sampling flags, `-n`,
    /// the KV cache mode, the adapter and the speculative drafter.
    pub(crate) server: CliServerSettings,
    /// Send the raw transcript as a completion instead of rendering the chat
    /// template (mirrors `generate --no-chat-template`).
    pub no_chat_template: bool,
    /// Also print the reasoning channel (dimmed on a TTY). The raw channel
    /// markers never print.
    pub show_reasoning: bool,
    /// Images attached to the first user message (`--image`).
    pub images: Vec<PathBuf>,
}

/// Outcome of interpreting a single submitted line / block.
enum Action {
    /// A complete user message ready to send to the model.
    Send(String),
    /// `/bye` (or EOF) — leave the REPL.
    Exit,
    /// Empty input — reprompt without sending.
    Empty,
}

/// Result of dispatching a slash command.
#[derive(Debug, PartialEq, Eq)]
enum SlashOutcome {
    /// Input was not a slash command — treat it as a user message.
    NotACommand,
    /// A command was handled; continue the loop (no conversation reset).
    Handled,
    /// `/clear` — the transcript was emptied; the caller also starts a new
    /// server-side conversation.
    Cleared,
    /// `/image <path>` — attach an image to the next message.
    Image(PathBuf),
    /// `/bye` — leave the REPL cleanly.
    Exit,
}

/// Families the chat worker cannot hold a conversation with, and what to run
/// instead. Checked before the weights load.
fn non_chat_family_error(model_path: &Path) -> Option<anyhow::Error> {
    use mlxcel::models::ModelType;
    let model_type = mlxcel::models::get_model_type(model_path).ok()?;
    let message = match model_type {
        ModelType::DiffusionGemma | ModelType::Llada2Moe => {
            "Block-diffusion models do not support interactive chat yet; use a one-shot prompt: \
             mlxcel generate -m <model> -p \"...\""
        }
        ModelType::Florence2VLM => {
            "Florence-2 is an image-task model without a chat surface; run a task instead: \
             mlxcel generate -m <model> --image <image> -p '<CAPTION>' (or <OD>, <OCR>, ...)"
        }
        ModelType::NemotronVoiceChat => {
            "Nemotron VoiceChat is a speech-to-speech model without a text chat surface; run a \
             turn instead: mlxcel generate -m <model> --audio question.wav --output-audio \
             answer.wav [-p '<system prompt>']"
        }
        ModelType::NemotronParseVLM => {
            "Nemotron-Parse is a document-parsing model without a chat surface; parse a page \
             instead: mlxcel generate -m <model> --image <page> -p \
             '</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>'"
        }
        _ => return None,
    };
    Some(anyhow!(message))
}

/// Run the interactive multi-turn chat REPL until the user exits.
///
/// # Errors
///
/// Returns an error if the model cannot be resolved or served, or the
/// terminal line editor cannot be initialized. A failed turn is reported and
/// the loop continues.
pub fn run_chat(mut opts: ChatOptions) -> Result<()> {
    let runtime = initialize_runtime_checked()?;
    super::generate::print_runtime_setup(&runtime);

    let model_path = mlxcel::downloader::resolve_model_source_with_override(
        &opts.model,
        opts.models_dir.as_deref(),
        opts.revision.as_deref(),
    )?;
    if let Some(err) = non_chat_family_error(&model_path) {
        return Err(err);
    }
    // issue #1350: the KV cache mode this family can really run, announced
    // once before the load banner.
    opts.server.kv_cache_mode = mlxcel::cli::turbo_args::resolve_and_announce_kv_cache_mode(
        opts.server.kv_cache_mode,
        &model_path,
    );
    // The REPL keeps the server default: prompt cache on, no warmup beyond
    // the first turn's own prefill.
    opts.server.prompt_cache = true;
    opts.server.warmup = false;

    println!("Loading model from {model_path:?}...");
    let load_start = Instant::now();
    let server = InProcessServer::start(&opts.server.startup(&model_path))?;
    println!(
        "Model loaded in {:.2}s.",
        load_start.elapsed().as_secs_f64()
    );
    if opts.server.max_tokens.is_none() {
        println!("Per-turn output: unlimited (-1) -> until EOS or the model context window.");
    }
    warn_if_base_model(&model_path, opts.no_chat_template);

    let mut editor = DefaultEditor::new()
        .map_err(|e| anyhow!("Failed to initialize the interactive line editor: {e}"))?;
    let interactive = io::stdin().is_terminal();
    print_banner(interactive);

    let display = TurnDisplay {
        show_reasoning: opts.show_reasoning,
    };
    let mut transcript: Vec<Turn> = Vec::new();
    let mut pending_images = opts.images.clone();
    let mut conversation = 0u64;
    let session = std::process::id();

    loop {
        let user_text = match read_input(&mut editor, interactive) {
            Action::Exit => break,
            Action::Empty => continue,
            Action::Send(text) => text,
        };
        match handle_slash_command(&user_text, &mut transcript) {
            SlashOutcome::Exit => break,
            SlashOutcome::Cleared => {
                conversation += 1;
                pending_images.clear();
                continue;
            }
            SlashOutcome::Image(path) => {
                println!("Image {} attached to the next message.", path.display());
                pending_images.push(path);
                continue;
            }
            SlashOutcome::Handled => continue,
            SlashOutcome::NotACommand => {}
        }

        transcript.push(Turn {
            message: ChatMessage {
                role: "user".to_string(),
                content: user_text,
            },
            images: std::mem::take(&mut pending_images),
        });
        let mut printer = TurnPrinter::new(display);
        let cache_key = format!("mlxcel-chat-{session}-{conversation}");
        let outcome = run_cancellable(|cancel| {
            if opts.no_chat_template {
                let body = mlxcel::cli::in_process_client::completion_request_body(
                    &concat_plaintext(&transcript_messages(&transcript)),
                    &opts.server,
                );
                server.complete(completion_request_from_json(body)?, cancel, |delta| {
                    printer.on_delta(delta)
                })
            } else {
                let mut body = mlxcel::cli::in_process_client::chat_request_body(
                    messages_json(&transcript)?,
                    &opts.server,
                );
                body["prompt_cache_key"] = json!(cache_key);
                server.chat(chat_request_from_json(body)?, cancel, |delta| {
                    printer.on_delta(delta)
                })
            }
        });
        match outcome {
            Ok(turn) => {
                printer.finish(&turn, "Restart");
                println!();
                if turn.cancelled {
                    // The interrupted exchange leaves the transcript, so the
                    // next turn does not answer a half-finished reply.
                    transcript.pop();
                } else {
                    transcript.push(Turn {
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: turn.content,
                        },
                        images: Vec::new(),
                    });
                }
            }
            Err(err) => {
                println!();
                eprintln!("error: {err:#}");
                transcript.pop();
            }
        }
    }

    println!("Bye!");
    server.shutdown()
}

/// Say so when the checkpoint ships no chat template: it is likely a base
/// model, and the server renders turns with its generic default format.
fn warn_if_base_model(model_path: &Path, no_chat_template: bool) {
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

/// Print the one-time greeting / help hint.
fn print_banner(interactive: bool) {
    println!();
    println!("mlxcel interactive chat. Type a message and press Enter.");
    println!("Commands: /bye (exit), /clear (reset conversation), /image <path>, /? or /help.");
    println!("Multiline: open and close a block with {MULTILINE_FENCE} on their own lines.");
    println!("Ctrl-C while a reply streams stops that reply.");
    if !interactive {
        // Piped / redirected stdin: say so instead of looking hung.
        eprintln!("(non-interactive stdin detected: reading messages until EOF)");
    }
    println!();
}

/// Read one logical user submission: a single line, or a `"""`-fenced
/// multiline block. Returns an [`Action`] describing what to do next.
fn read_input(editor: &mut DefaultEditor, interactive: bool) -> Action {
    let prompt = if interactive { ">>> " } else { "" };
    let line = match editor.readline(prompt) {
        Ok(line) => line,
        // Ctrl-D / EOF (also fired at end of a piped stdin) exits cleanly so a
        // non-interactive invocation never hangs.
        Err(ReadlineError::Eof) => return Action::Exit,
        // Ctrl-C cancels the current line without exiting the REPL.
        Err(ReadlineError::Interrupted) => {
            println!("(^C — type /bye to exit)");
            return Action::Empty;
        }
        Err(_) => return Action::Exit,
    };

    // Record the raw line in history (best-effort; history is non-essential).
    let _ = editor.add_history_entry(line.as_str());

    let trimmed = line.trim();

    // Triple-quote opens a multiline block: keep reading until the closing
    // fence (ollama-style). The opening fence may carry inline text after it.
    if let Some(rest) = trimmed.strip_prefix(MULTILINE_FENCE) {
        return read_multiline_block(editor, rest);
    }

    if trimmed.is_empty() {
        return Action::Empty;
    }

    Action::Send(trimmed.to_string())
}

/// Continue reading lines after an opening `"""` until the closing `"""`.
///
/// `first_rest` is whatever followed the opening fence on the same line. The
/// closing fence may appear alone or with trailing text before it; everything
/// up to (but not including) the fence is part of the message.
fn read_multiline_block(editor: &mut DefaultEditor, first_rest: &str) -> Action {
    let mut buf = String::new();

    // Inline text after the opening fence, e.g. `"""hello` starts the body.
    let first = first_rest.trim_start();
    if let Some(before) = first.strip_suffix(MULTILINE_FENCE) {
        // Single-line `"""body"""` form.
        return finalize_multiline(before);
    }
    if !first.is_empty() {
        buf.push_str(first);
        buf.push('\n');
    }

    loop {
        match editor.readline("... ") {
            Ok(line) => {
                let _ = editor.add_history_entry(line.as_str());
                if let Some(before) = line.strip_suffix(MULTILINE_FENCE) {
                    buf.push_str(before);
                    return finalize_multiline(&buf);
                }
                buf.push_str(&line);
                buf.push('\n');
            }
            // EOF mid-block: send whatever was accumulated so we never hang.
            Err(ReadlineError::Eof) => return finalize_multiline(&buf),
            Err(ReadlineError::Interrupted) => {
                println!("(^C — multiline input discarded)");
                return Action::Empty;
            }
            Err(_) => return Action::Exit,
        }
    }
}

/// Trim a finished multiline body and turn it into a [`Action::Send`], or
/// [`Action::Empty`] when nothing meaningful was entered.
fn finalize_multiline(body: &str) -> Action {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        Action::Empty
    } else {
        Action::Send(trimmed.to_string())
    }
}

/// Handle a slash command, returning a [`SlashOutcome`] that tells the loop
/// whether to send the input as a message, continue, reset, or exit.
fn handle_slash_command(input: &str, transcript: &mut Vec<Turn>) -> SlashOutcome {
    if !input.starts_with('/') {
        return SlashOutcome::NotACommand;
    }
    // First whitespace-delimited token is the command.
    let mut words = input.split_whitespace();
    let command = words.next().unwrap_or(input);
    match command {
        "/bye" => SlashOutcome::Exit,
        "/clear" => {
            transcript.clear();
            println!("Conversation cleared.");
            SlashOutcome::Cleared
        }
        "/image" => {
            let path = input["/image".len()..].trim();
            if path.is_empty() {
                println!("Usage: /image <path>");
                SlashOutcome::Handled
            } else {
                SlashOutcome::Image(PathBuf::from(path))
            }
        }
        "/?" | "/help" => {
            print_help();
            SlashOutcome::Handled
        }
        other => {
            println!("Unknown command: {other}. Type /? for the list of commands.");
            SlashOutcome::Handled
        }
    }
}

/// Print the slash-command help block (`/?`).
fn print_help() {
    println!("Available commands:");
    println!("  /bye           Exit the chat.");
    println!("  /clear         Reset the conversation (clears all prior turns).");
    println!("  /image <path>  Attach an image to the next message (vision models).");
    println!("  /?, /help      Show this help.");
    println!("  {MULTILINE_FENCE} ... {MULTILINE_FENCE}   Wrap multiline input as one message.");
}

/// Raw, role-less concatenation for the explicit `--no-chat-template` path
/// (and template render-failure fallback when the user has opted into raw
/// mode). One newline between turns, nothing else — mirrors offline
/// `generate --no-chat-template` so completion-style usage is unaffected by
/// the structured fallback added in issue #133.
fn concat_plaintext(conversation: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in conversation {
        out.push_str(&msg.content);
        out.push('\n');
    }
    out
}

#[cfg(test)]
#[path = "chat_tests.rs"]
mod tests;
