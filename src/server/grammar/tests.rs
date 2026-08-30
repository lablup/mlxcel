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

//! Lazy-grammar trigger tests.

use super::{GrammarTrigger, LazyGate, LazyOutcome};

fn feed(gate: &mut LazyGate, pieces: &[(u32, &str)]) -> Vec<String> {
    let mut activations = Vec::new();
    for (token, piece) in pieces {
        match gate.observe(*token, piece.as_bytes()) {
            LazyOutcome::StillWaiting => {}
            LazyOutcome::ActivateTokens(ids) => activations.push(format!("tokens:{ids:?}")),
            LazyOutcome::ActivateBytes(bytes) => {
                activations.push(format!("bytes:{}", String::from_utf8_lossy(&bytes)));
            }
        }
    }
    activations
}

#[test]
fn a_token_trigger_activates_on_its_own_id_and_replays_only_that_token() {
    let mut gate = LazyGate::new(&[GrammarTrigger::Token(42)]).unwrap();
    assert!(gate.awaiting());
    let acts = feed(&mut gate, &[(1, "hello "), (2, "world "), (42, "<tool>")]);
    assert_eq!(acts, vec!["tokens:[42]".to_string()]);
    assert!(!gate.awaiting());
}

#[test]
fn a_word_trigger_replays_from_the_start_of_the_match_not_the_buffer() {
    let mut gate = LazyGate::new(&[GrammarTrigger::Word("{\"name\"".to_string())]).unwrap();
    let acts = feed(
        &mut gate,
        &[(1, "sure, here goes: "), (2, "{\"name\""), (3, "x")],
    );
    assert_eq!(acts, vec!["bytes:{\"name\"".to_string()]);
}

#[test]
fn a_pattern_trigger_starts_at_the_first_non_empty_capture_group() {
    // b10621 takes the first non-empty capturing group as the start of the
    // constrained text, so the prose before the group is not replayed.
    let mut gate = LazyGate::new(&[GrammarTrigger::Pattern(
        "thinking\\.\\.\\.(\\{)".to_string(),
    )])
    .unwrap();
    let acts = feed(&mut gate, &[(1, "thinking..."), (2, "{")]);
    assert_eq!(acts, vec!["bytes:{".to_string()]);
}

#[test]
fn a_pattern_full_trigger_must_match_the_whole_buffer() {
    let mut gate = LazyGate::new(&[GrammarTrigger::PatternFull("[a-z]+".to_string())]).unwrap();
    // A prefix of the buffer matching is not enough while more text follows.
    let acts = feed(&mut gate, &[(1, "abc1")]);
    assert!(acts.is_empty(), "{acts:?}");
    let mut gate = LazyGate::new(&[GrammarTrigger::PatternFull("[a-z]+".to_string())]).unwrap();
    let acts = feed(&mut gate, &[(1, "abc")]);
    assert_eq!(acts, vec!["bytes:abc".to_string()]);
}

#[test]
fn a_multi_byte_character_split_across_tokens_does_not_break_matching() {
    let mut gate = LazyGate::new(&[GrammarTrigger::Word("é!".to_string())]).unwrap();
    let bytes = "é".as_bytes();
    // First token carries only the lead byte: the buffer is not valid UTF-8
    // yet and must not panic or spuriously fire.
    assert!(matches!(
        gate.observe(1, &bytes[..1]),
        LazyOutcome::StillWaiting
    ));
    match gate.observe(2, &[bytes[1], b'!']) {
        LazyOutcome::ActivateBytes(b) => assert_eq!(String::from_utf8_lossy(&b), "é!"),
        other => panic!(
            "expected activation, got {}",
            matches!(other, LazyOutcome::StillWaiting)
        ),
    }
}

#[test]
fn an_uncompilable_trigger_pattern_is_refused_rather_than_never_firing() {
    let err = LazyGate::new(&[GrammarTrigger::Pattern("(unclosed".to_string())]).unwrap_err();
    assert!(err.to_string().contains("invalid grammar trigger pattern"));
}

#[test]
fn a_gate_with_only_patterns_stays_awaiting_until_one_matches() {
    let mut gate = LazyGate::new(&[GrammarTrigger::Pattern("NEVER".to_string())]).unwrap();
    let acts = feed(&mut gate, &[(1, "a"), (2, "b"), (3, "c")]);
    assert!(acts.is_empty());
    assert!(gate.awaiting());
}
