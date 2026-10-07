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

//! Unit tests for the interactive chat REPL's pure helpers (issue #96).
//!
//! These cover slash-command dispatch, multiline-block finalization, and the
//! raw `--no-chat-template` transcript, the logic that does not need a loaded
//! model. Prompt rendering is the server's (issue #2173) and is tested there.

use super::*;

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: content.to_string(),
    }
}

fn user(content: &str) -> Turn {
    Turn {
        message: message("user", content),
        images: Vec::new(),
    }
}

fn assistant(content: &str) -> Turn {
    Turn {
        message: message("assistant", content),
        images: Vec::new(),
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
    // `concat_plaintext` is the raw `--no-chat-template` path: content only,
    // no role markers, one newline between turns.
    assert_eq!(
        concat_plaintext(&transcript_messages(&convo)),
        "first\nsecond\nthird\n"
    );
}

#[test]
fn slash_image_attaches_a_path_and_requires_one() {
    let mut convo = vec![user("hi")];
    assert_eq!(
        handle_slash_command("/image  photos/cat one.png ", &mut convo),
        SlashOutcome::Image(PathBuf::from("photos/cat one.png"))
    );
    assert_eq!(
        handle_slash_command("/image", &mut convo),
        SlashOutcome::Handled
    );
    assert_eq!(convo.len(), 1, "/image must not mutate the transcript");
}

#[test]
fn non_chat_families_are_refused_before_loading() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "florence2"}"#,
    )
    .expect("config");
    let err = non_chat_family_error(dir.path()).expect("Florence-2 has no chat surface");
    assert!(err.to_string().contains("image-task model"), "{err}");
    std::fs::write(dir.path().join("config.json"), r#"{"model_type": "llama"}"#).expect("config");
    assert!(non_chat_family_error(dir.path()).is_none());
}
