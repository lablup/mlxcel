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

//! Interactive multi-turn chat REPL (epic #92, issue #96).
//!
//! A line-edited, streaming chat loop in the spirit of `mlx_lm.chat` /
//! `ollama run`. It is deliberately a thin orchestration layer over the
//! **existing** generation machinery — it forks none of it:
//!
//! * model resolution → [`mlxcel::downloader::resolve_model_source`] (the same
//!   `-m`-accepts-a-repo-id resolver `generate` / `serve` / `inspect` use,
//!   issue #94),
//! * tokenization → [`mlxcel::tokenizer::MlxcelTokenizer`] via
//!   [`mlxcel::tokenizer::load_tokenizer`],
//! * prompt rendering → [`ChatTemplateProcessor`] (the exact chat-template path
//!   `generate` applies),
//! * sampling → [`build_sampling_config`] over [`ResolvedSamplingParams`] (the
//!   same `SamplingConfig` assembly `generate` uses),
//! * token generation → [`Session::generate_streaming`] (which delegates to the
//!   same `CxxGenerator` streaming decode loop the offline `generate` path drives),
//! * incremental detokenization → [`StreamingDecodeState`] (the server's own
//!   byte-fallback-safe streaming detokenizer, now shared).
//!
//! ## Multi-turn context
//!
//! The full conversation ([`ChatMessage`] history) is re-rendered through the
//! chat template every turn and the resulting prompt is fed to
//! `generate_streaming`. This is the correctness-first reuse path: the
//! generator re-prefills the accumulated context on each turn (it owns and
//! resets its KV cache per call), so context is preserved without
//! reimplementing a bespoke cross-turn KV-append loop that would diverge from
//! the canonical generation code. `/clear` simply empties the history.
//!
//! ## Reusable entry point
//!
//! [`run_chat`] is a free function taking a self-contained [`ChatOptions`] so
//! the forthcoming `mlxcel run` verb (issue #95) can dispatch straight into the
//! REPL without going through `GenerateArgs`. The `generate` subcommand calls
//! [`run_chat`] when invoked with no `-p/--prompt`.

use std::io::{self, IsTerminal, Write as IoWrite};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, anyhow};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use mlxcel::cli::max_tokens::{
    DEFAULT_CONTEXT_WINDOW_FALLBACK, UNLIMITED_MAX_TOKENS, resolve_unlimited_max_tokens,
};
use mlxcel::reasoning_stream;
use mlxcel::sampling::{ResolvedSamplingParams, build_sampling_config};
use mlxcel::server::chat_template::{ChatMessage, ChatTemplateProcessor};
use mlxcel::server::model_provider::model_worker::StreamingDecodeState;
use mlxcel::tokenizer::{MlxcelTokenizer, load_tokenizer};
use mlxcel::{LanguageModel, SamplingConfig, Session, initialize_runtime, select_backend};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::sampling::TokenBiasMap;

/// Triple-quote fence that opens / closes an ollama-style multiline input
/// block.
const MULTILINE_FENCE: &str = "\"\"\"";

/// Self-contained configuration for the interactive chat REPL.
///
/// Constructed by the `generate` subcommand (no-prompt path) and by the
/// future `mlxcel run` verb (issue #95). Keeping this a plain options struct —
/// rather than borrowing `GenerateArgs` — is what lets #95 reuse [`run_chat`]
/// without dragging in the offline generate flag surface.
#[derive(Debug, Clone)]
pub struct ChatOptions {
    /// Local model directory **or** a HuggingFace `owner/name` repo-id. Passed
    /// verbatim to [`mlxcel::downloader::resolve_model_source_with_override`],
    /// so a repo-id auto-downloads exactly like `generate -m <repo-id>`. A bare
    /// name without a slash (e.g. `Qwen3-4B-4bit`) is resolved as
    /// `mlx-community/<name>`; override the org with the `MLXCEL_DEFAULT_ORG`
    /// environment variable.
    pub model: PathBuf,
    /// Model-store root override (`--models-dir`, issue #107) threaded into the
    /// `-m` resolver so a repo-id resolves to / downloads under this root. `None`
    /// keeps the `MLXCEL_MODELS_DIR`-then-cache-root resolution.
    pub models_dir: Option<PathBuf>,
    /// Maximum number of tokens to generate per assistant turn.
    pub max_tokens: usize,
    /// Resolved sampling knobs (temperature / top-k / top-p / min-p /
    /// penalties). The same struct `generate` builds its `SamplingConfig` from.
    pub sampling: ResolvedSamplingParams,
    /// KV-cache quantization mode for the generator.
    pub kv_cache_mode: KVCacheMode,
    /// When `true`, skip chat-template application and feed raw user text to
    /// the model (mirrors `generate --no-chat-template`). Multi-turn context is
    /// then concatenated as plain text.
    pub no_chat_template: bool,
    /// When `true`, also print the model's reasoning channel (dimmed on a TTY)
    /// instead of suppressing it. Mirrors `generate --show-reasoning`. The raw
    /// channel markers are never printed regardless of this flag.
    pub show_reasoning: bool,
}

impl ChatOptions {
    /// Construct chat options with the conventional REPL defaults
    /// (greedy-friendly sampling left to the caller; everything else inert).
    ///
    /// Callers typically build [`ResolvedSamplingParams`] from their own CLI
    /// flags; this helper only fixes the non-sampling knobs so `mlxcel run`
    /// (issue #95) and the `generate` no-prompt path share one default surface.
    pub fn new(model: PathBuf, max_tokens: usize, sampling: ResolvedSamplingParams) -> Self {
        Self {
            model,
            models_dir: None,
            max_tokens,
            sampling,
            kv_cache_mode: KVCacheMode::Fp16,
            no_chat_template: false,
            show_reasoning: false,
        }
    }
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
enum SlashOutcome {
    /// Input was not a slash command — treat it as a user message.
    NotACommand,
    /// A command was handled; continue the loop (no conversation reset).
    Handled,
    /// `/clear` — conversation was reset; the caller should also reset
    /// generator state before the next turn.
    Cleared,
    /// `/bye` — leave the REPL cleanly.
    Exit,
}

/// Run the interactive multi-turn chat REPL until the user exits.
///
/// Reusable entry point for both the `generate` no-prompt path and the
/// `mlxcel run` verb (issue #95). Loads the model once, then loops: read a
/// line (or a `"""` multiline block), interpret slash commands, render the
/// accumulated conversation through the chat template, and stream the
/// assistant reply token-by-token via the shared inference [`Session`].
///
/// # Errors
///
/// Returns an error if the model cannot be resolved / loaded, the tokenizer
/// cannot be read, or the terminal line editor cannot be initialized. Per-turn
/// generation never aborts the loop; a `/clear` or a fresh turn always recovers.
pub fn run_chat(opts: ChatOptions) -> Result<()> {
    let runtime = initialize_runtime();
    println!("Runtime device: {}", runtime.device);

    // Reuse the exact `-m` resolver `generate` / `serve` / `inspect` use, so a
    // repo-id auto-downloads into the global store (epic #92, issues #93/#94),
    // honoring the `--models-dir` override (issue #107).
    let model_path = mlxcel::downloader::resolve_model_source_with_override(
        &opts.model,
        opts.models_dir.as_deref(),
    )?;

    println!("Loading model from {model_path:?}...");
    let load_start = Instant::now();
    let (model, _tok_from_load) = select_backend().load_model(&model_path)?;
    if matches!(
        model,
        mlxcel::LoadedModel::DiffusionGemma(_) | mlxcel::LoadedModel::Llada2Moe(_)
    ) {
        // The REPL drives the autoregressive CxxGenerator loop; the
        // block-diffusion / block-unmasking engines are one-shot only.
        return Err(anyhow!(
            "Block-diffusion models do not support interactive chat yet; use a one-shot prompt: \
             mlxcel generate -m <model> -p \"...\""
        ));
    }
    let tokenizer = load_tokenizer(&model_path)?;
    println!(
        "Model loaded in {:.2}s.",
        load_start.elapsed().as_secs_f64()
    );

    // llama.cpp parity (issue #476): the default unlimited `-n -1` caps each
    // reply at the model context window instead of a fixed number, so chat turns
    // run until EOS or the window fills. Read the window once; `stream_turn`
    // computes the per-turn budget (window - rendered prompt) so a growing
    // transcript never overruns the context. An explicit `-n N` is honored
    // verbatim every turn.
    let context_window =
        mlxcel::read_model_context_window(&model_path).unwrap_or(DEFAULT_CONTEXT_WINDOW_FALLBACK);
    if opts.max_tokens == UNLIMITED_MAX_TOKENS {
        println!(
            "Per-turn output: unlimited (-1) -> until EOS or the model context window \
             ({context_window} tokens)."
        );
    }

    // Same chat-template discovery as `generate`'s `load_cli_prompt`. `None`
    // (no template, or `--no-chat-template`) falls back to raw-text turns.
    let processor = if opts.no_chat_template {
        None
    } else {
        ChatTemplateProcessor::from_model_path(&model_path)
            .ok()
            .flatten()
    };
    if processor.is_none() && !opts.no_chat_template {
        eprintln!(
            "Note: this model ships no chat template and is likely a base (non-instruction-tuned) model."
        );
        eprintln!(
            "      Chat responses will likely be incoherent or repetitive — base models are not designed"
        );
        eprintln!("      for interactive conversation.");
        eprintln!();
        eprintln!(
            "      Try an instruction-tuned variant instead. Naming conventions vary by family:"
        );
        eprintln!(
            "      Gemma uses an \"-it\" suffix (e.g. gemma-4-e4b-it-4bit); Llama and Qwen2.5 use"
        );
        eprintln!(
            "      \"-Instruct\"; Qwen3/Qwen3.5 use the plain name, with \"-Base\" marking the"
        );
        eprintln!("      non-instruct variant.");
        eprintln!();
        eprintln!(
            "      Falling back to a generic User/Assistant prompt format to mitigate echo loops."
        );
        eprintln!("      For raw text mode without role markers, pass --no-chat-template;");
        eprintln!("      for one-shot completion, use `mlxcel generate -p <prompt>`.");
    }

    // Build the SamplingConfig once via the shared assembly used by `generate`.
    // Stop tokens come from the model's config, exactly like the offline path.
    let mut sampling = opts.sampling.clone();
    if sampling.stop_token_ids.is_empty() {
        sampling.stop_token_ids = mlxcel::read_eos_token_ids(&model_path);
    }
    let sampling_config = build_sampling_config(sampling);

    // issue #350: suppress this model's reserved multimodal placeholder ids
    // (audio / image / video span markers) in the interactive chat generator,
    // mirroring `run_generation_mode` for the one-shot path. `CxxGenerator`'s
    // `compose_sampling` injects this cached bias whenever the per-call sampling
    // config carries no token bias of its own (the chat config's stays empty),
    // so a placeholder id can never leak into a chat reply. Zero-cost for
    // non-multimodal models: the suppressed set is empty, the bias map stays
    // empty, and `apply_token_bias` short-circuits.
    let mut output_suppression = TokenBiasMap::new();
    output_suppression.suppress_tokens(&model.output_suppressed_token_ids());

    // One inference session for the whole chat (issue #448, ADR 0004). Under
    // default features `select_backend()` folds to MLX and the session wraps the
    // same `CxxGenerator`, so `generate_streaming` runs the identical loop and
    // resets its KV cache per call; re-rendering the full transcript each turn
    // preserves context without forking the generation loop.
    let mut session = select_backend().create_session(
        &model_path,
        model.num_layers(),
        opts.kv_cache_mode,
        output_suppression,
    )?;

    let mut editor = DefaultEditor::new()
        .map_err(|e| anyhow!("failed to initialize the interactive line editor: {e}"))?;
    let interactive = io::stdin().is_terminal();

    print_banner(interactive);

    let mut conversation: Vec<ChatMessage> = Vec::new();

    loop {
        let action = read_input(&mut editor, interactive);
        match action {
            Action::Exit => {
                println!("Bye!");
                break;
            }
            Action::Empty => continue,
            Action::Send(user_text) => {
                match handle_slash_command(&user_text, &mut conversation) {
                    SlashOutcome::Exit => {
                        println!("Bye!");
                        break;
                    }
                    SlashOutcome::Cleared => {
                        // `/clear` already reset the transcript; also drop any
                        // session-side state for a clean next prefill.
                        session.reset_with_model(&model);
                        continue;
                    }
                    SlashOutcome::Handled => continue,
                    SlashOutcome::NotACommand => {}
                }

                conversation.push(ChatMessage {
                    role: "user".to_string(),
                    content: user_text,
                });

                let prompt =
                    render_prompt(processor.as_ref(), &conversation, opts.no_chat_template);
                let reply = stream_turn(
                    &mut session,
                    &model,
                    &tokenizer,
                    &prompt,
                    opts.max_tokens,
                    context_window,
                    &sampling_config,
                    opts.show_reasoning,
                )?;

                conversation.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: reply,
                });
            }
        }
    }

    mlxcel_core::clear_memory_cache();
    Ok(())
}

/// Print the one-time greeting / help hint.
fn print_banner(interactive: bool) {
    println!();
    println!("mlxcel interactive chat. Type a message and press Enter.");
    println!("Commands: /bye (exit), /clear (reset conversation), /? or /help (this list).");
    println!("Multiline: open and close a block with {MULTILINE_FENCE} on their own lines.");
    if !interactive {
        // Piped / redirected stdin: rustyline still reads lines, but there is
        // no TTY to edit on. Make the degraded mode explicit instead of looking
        // hung.
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
fn handle_slash_command(input: &str, conversation: &mut Vec<ChatMessage>) -> SlashOutcome {
    if !input.starts_with('/') {
        return SlashOutcome::NotACommand;
    }
    // First whitespace-delimited token is the command.
    let command = input.split_whitespace().next().unwrap_or(input);
    match command {
        "/bye" => SlashOutcome::Exit,
        "/clear" => {
            conversation.clear();
            println!("Conversation cleared.");
            SlashOutcome::Cleared
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
    println!("  /?, /help      Show this help.");
    println!("  {MULTILINE_FENCE} ... {MULTILINE_FENCE}   Wrap multiline input as one message.");
}

/// Render the accumulated conversation into a model prompt.
///
/// Three paths, in priority order:
///
/// 1. `--no-chat-template` (explicit user opt-in) → [`concat_plaintext`]: raw
///    content-only concatenation, no role markers. Mirrors the offline
///    `generate --no-chat-template` mode for completion-style usage.
/// 2. A chat template is present → [`ChatTemplateProcessor::apply`]: the exact
///    path `generate` uses for a single turn, generalized to the full
///    transcript. Template render failure falls back to (3).
/// 3. No template found and the user did not opt out →
///    [`concat_userassistant_fallback`]: a minimal `User:` / `Assistant:`
///    pseudo-template (issue #133). A bare concatenation here leaves base
///    models without any structural cue and they collapse into raw-text echo
///    loops; labeling turns and cueing the next assistant turn substantially
///    reduces that pathology without claiming to give base models
///    chat-grade behavior.
fn render_prompt(
    processor: Option<&ChatTemplateProcessor>,
    conversation: &[ChatMessage],
    no_chat_template: bool,
) -> String {
    if no_chat_template {
        return concat_plaintext(conversation);
    }
    match processor {
        Some(p) => p
            .apply(conversation, None)
            .unwrap_or_else(|_| concat_userassistant_fallback(conversation)),
        None => concat_userassistant_fallback(conversation),
    }
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

/// Generic `User:` / `Assistant:` pseudo-template fallback for models that
/// ship no chat template (issue #133).
///
/// This is *not* a true chat template — no BOS/EOS markers, no model-specific
/// special tokens — and the upstream `processor.is_none()` warning still
/// fires telling the user the model is likely a base / non-instruction-tuned
/// variant. The point is narrower: a bare content-only concatenation leaves
/// the model without any structural cue and base models tend to collapse
/// into echo loops where they parrot the user's last line indefinitely. A
/// minimal role-labeled format with a trailing `Assistant:` cue (no newline)
/// nudges the model to produce an assistant turn next instead of continuing
/// to complete its own prompt.
fn concat_userassistant_fallback(conversation: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in conversation {
        match msg.role.as_str() {
            "user" => out.push_str("User: "),
            "assistant" => out.push_str("Assistant: "),
            "system" => out.push_str("System: "),
            other => {
                // Unknown role (e.g. "tool"): still mark it so the model has
                // something to anchor on rather than silently merging it into
                // the prior turn.
                out.push_str(other);
                out.push_str(": ");
            }
        }
        out.push_str(&msg.content);
        out.push_str("\n\n");
    }
    // No trailing newline: the bare `Assistant:` token is the cue that asks
    // the model for an assistant turn next.
    out.push_str("Assistant:");
    out
}

/// Tokenize, stream-generate, and print one assistant turn; return the decoded
/// reply text (to append to the transcript).
///
/// Tokenization matches `generate::tokenize_prompt` (skip the extra BOS when
/// the rendered prompt already embeds one). Generation goes through
/// [`Session::generate_streaming`] — which delegates to the same
/// `CxxGenerator::generate_streaming` loop the offline path uses — with a
/// per-token callback that streams text via [`StreamingDecodeState`].
#[allow(clippy::too_many_arguments)]
fn stream_turn<M: LanguageModel>(
    session: &mut Session,
    model: &M,
    tokenizer: &MlxcelTokenizer,
    prompt: &str,
    max_tokens: usize,
    context_window: usize,
    sampling_config: &SamplingConfig,
    show_reasoning: bool,
) -> Result<String> {
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    let prompt_tokens: Vec<i32> = tokenizer
        .encode(prompt, add_special)
        .map_err(|e| anyhow!("Tokenization failed: {e}"))?
        .iter()
        .map(|&x| x as i32)
        .collect();

    // llama.cpp parity (issue #476): an unlimited `-n -1` becomes the remaining
    // context (window minus this turn's rendered prompt) so the reply runs until
    // EOS or the window fills; an explicit `-n N` is used verbatim. Resolved
    // before the allocation below so the sentinel never reaches `with_capacity`.
    let max_tokens = resolve_unlimited_max_tokens(max_tokens, context_window, prompt_tokens.len());

    // Stream display through the shared incremental detokenizer (byte-fallback
    // safe), then split the reasoning channel out of the visible output so a
    // Gemma 4 (or Qwen-style) checkpoint's chain-of-thought and its raw
    // `<|channel>thought` / `<channel|>` markers never print (issue #884). For a
    // non-thinking model the filter is an inert passthrough, so output is
    // byte-identical to the pre-#884 path. The raw generated ids are collected
    // in parallel so the final turn text is decoded byte-exactly for history.
    let mut decode_state = StreamingDecodeState::new(tokenizer, &prompt_tokens);
    // Start the reasoning filter inside the channel when the rendered prompt
    // primed an open thinking marker (Qwen-style `<think>\n`); otherwise the
    // primed thought body and its raw `</think>` close marker would print, since
    // the open marker is in the prompt rather than the generated tokens.
    let markers = tokenizer.infer_thinking_markers();
    let mut filter = if reasoning_stream::prompt_primed_open_thinking(&markers, prompt) {
        reasoning_stream::ReasoningFilter::new_primed_open_thinking(&markers)
    } else {
        reasoning_stream::ReasoningFilter::new(&markers)
    };
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut stdout = io::stdout();
    let dim = stdout.is_terminal();

    session.generate_streaming(
        model,
        &prompt_tokens,
        max_tokens,
        sampling_config,
        |token_id| {
            generated_ids.push(token_id as u32);
            if let Some(text) = decode_state.on_token(token_id, tokenizer) {
                let visible =
                    reasoning_stream::render_visible(&filter.feed(&text), show_reasoning, dim);
                if !visible.is_empty() {
                    print!("{visible}");
                    let _ = stdout.flush();
                }
            }
            true
        },
    );

    // Flush the detokenizer's held-back tail (e.g. a multi-byte char split
    // across the final tokens), then the reasoning filter's own buffered tail
    // (a partial marker, or an unclosed thought block that stays hidden), so no
    // visible text is lost or duplicated.
    if let Some(tail) = decode_state.flush(tokenizer) {
        let visible = reasoning_stream::render_visible(&filter.feed(&tail), show_reasoning, dim);
        if !visible.is_empty() {
            print!("{visible}");
            let _ = stdout.flush();
        }
    }
    let visible = reasoning_stream::render_visible(&filter.flush(), show_reasoning, dim);
    if !visible.is_empty() {
        print!("{visible}");
        let _ = stdout.flush();
    }
    println!();
    println!();

    // Decode the full assistant turn (skip special tokens so template markers do
    // not leak into the next turn's rendered history). Kept as the byte-exact
    // turn text used for the transcript.
    let reply = tokenizer.decode(&generated_ids, true).unwrap_or_default();
    Ok(reply)
}

#[cfg(test)]
#[path = "chat_tests.rs"]
mod tests;
