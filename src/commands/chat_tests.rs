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

//! Unit tests for the interactive chat REPL's pure helpers (issue #96).
//!
//! These cover slash-command dispatch, multiline-block finalization, and
//! prompt rendering — the logic that does not require a loaded model. The
//! end-to-end streaming loop is exercised by the orchestrator's broader,
//! real-model integration runs.

use super::*;

fn user(content: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
    }
}

fn assistant(content: &str) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
        content: content.to_string(),
    }
}

#[test]
fn slash_bye_signals_exit() {
    let mut convo = vec![user("hi")];
    assert!(matches!(
        handle_slash_command("/bye", &mut convo),
        SlashOutcome::Exit
    ));
    // Exit must not mutate the transcript.
    assert_eq!(convo.len(), 1);
}

#[test]
fn slash_clear_resets_conversation() {
    let mut convo = vec![user("hi"), assistant("hello")];
    assert!(matches!(
        handle_slash_command("/clear", &mut convo),
        SlashOutcome::Cleared
    ));
    assert!(convo.is_empty());
}

#[test]
fn slash_help_aliases_are_handled_without_reset() {
    let mut convo = vec![user("hi")];
    assert!(matches!(
        handle_slash_command("/?", &mut convo),
        SlashOutcome::Handled
    ));
    assert!(matches!(
        handle_slash_command("/help", &mut convo),
        SlashOutcome::Handled
    ));
    assert_eq!(convo.len(), 1, "help must not mutate the transcript");
}

#[test]
fn unknown_slash_command_is_handled_not_sent() {
    let mut convo = Vec::new();
    assert!(matches!(
        handle_slash_command("/nope", &mut convo),
        SlashOutcome::Handled
    ));
}

#[test]
fn non_slash_input_is_not_a_command() {
    let mut convo = Vec::new();
    assert!(matches!(
        handle_slash_command("hello there", &mut convo),
        SlashOutcome::NotACommand
    ));
    // A message that merely contains a slash mid-string is still a message.
    assert!(matches!(
        handle_slash_command("what is 1/2", &mut convo),
        SlashOutcome::NotACommand
    ));
}

#[test]
fn slash_command_ignores_trailing_args() {
    let mut convo = vec![user("hi")];
    // `/clear` with trailing tokens still dispatches on the first token.
    assert!(matches!(
        handle_slash_command("/clear everything", &mut convo),
        SlashOutcome::Cleared
    ));
    assert!(convo.is_empty());
}

#[test]
fn finalize_multiline_trims_and_detects_empty() {
    assert!(matches!(finalize_multiline("   \n  "), Action::Empty));
    match finalize_multiline("  line one\nline two  ") {
        Action::Send(text) => assert_eq!(text, "line one\nline two"),
        _ => panic!("expected Send"),
    }
}

#[test]
fn concat_plaintext_joins_turns_with_newlines() {
    let convo = vec![user("first"), assistant("second"), user("third")];
    assert_eq!(concat_plaintext(&convo), "first\nsecond\nthird\n");
}

#[test]
fn render_prompt_without_template_falls_back_to_plaintext() {
    let convo = vec![user("hello")];
    // No processor and not forced: plain-text fallback.
    assert_eq!(render_prompt(None, &convo, false), "hello\n");
    // `no_chat_template` forces plain-text even if a processor were present.
    assert_eq!(render_prompt(None, &convo, true), "hello\n");
}

#[test]
fn render_prompt_with_template_renders_all_turns() {
    // A minimal Jinja chat template that simply marks each role+content.
    let template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>\
        {% endfor %}{% if add_generation_prompt %}<gen>{% endif %}"
        .to_string();
    let processor = ChatTemplateProcessor::with_template(template);
    let convo = vec![user("hi"), assistant("hello"), user("again")];

    let rendered = render_prompt(Some(&processor), &convo, false);
    assert_eq!(
        rendered,
        "<user>hi</user><assistant>hello</assistant><user>again</user><gen>"
    );
    // Multi-turn context is preserved: the prior assistant turn is present.
    assert!(rendered.contains("<assistant>hello</assistant>"));
}

#[test]
fn chat_options_new_sets_conventional_defaults() {
    let sampling = ResolvedSamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: None,
        repetition_penalty: 1.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: 0,
        dry_sequence_breakers: Vec::new(),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_token_ids: Vec::new(),
    };
    let opts = ChatOptions::new(PathBuf::from("models/foo"), 128, sampling);
    assert_eq!(opts.max_tokens, 128);
    assert!(!opts.no_chat_template);
    assert!(matches!(opts.kv_cache_mode, KVCacheMode::Fp16));
    assert!(opts.sampling.stop_token_ids.is_empty());
}
