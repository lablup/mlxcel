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

use super::tests::{ENC, decoder, encoded};
use super::*;

fn vocabulary() -> Vec<String> {
    ["<unk>", "\u{2581}hi", "\u{2581}there", "<en-US>", "!"]
        .iter()
        .map(|p| p.to_string())
        .collect()
}

fn frame(t: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::slice(&encoded(4), &[0, t, 0], &[1, t + 1, ENC as i32])
}

#[test]
fn rnnt_stream_delta_rule() {
    assert_eq!(transcript_delta("", "hi"), "hi");
    assert_eq!(transcript_delta("hi", "hi there"), " there");
    // A revised prefix yields the whole text.
    assert_eq!(transcript_delta("hi there", "hi their"), "hi their");
    assert_eq!(transcript_delta("same", "same"), "");
}

#[test]
fn rnnt_stream_accumulates_trimmed_text_and_deltas() {
    // Logits always favour "▁hi": every frame emits `max_symbols` tokens.
    let dec = decoder(&[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]);
    let vocab = vocabulary();
    let mut state = RnntStreamState::new(&dec);
    let first = state.step(&dec, &frame(0), 2, &vocab).unwrap();
    assert_eq!(
        first,
        Some(("hi hi".to_string(), "hi hi".to_string())),
        "leading word-boundary space is trimmed"
    );
    let second = state.step(&dec, &frame(1), 1, &vocab).unwrap();
    assert_eq!(second, Some((" hi".to_string(), "hi hi hi".to_string())));
    assert_eq!(state.tokens, vec![1, 1, 1]);
    assert_eq!(state.rnnt_state.last_token(), 1);
}

#[test]
fn rnnt_stream_special_tokens_advance_state_without_update() {
    // A language tag is special: it advances the decoder but not the text.
    let dec = decoder(&[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]);
    let vocab = vocabulary();
    let mut state = RnntStreamState::new(&dec);
    assert_eq!(state.step(&dec, &frame(0), 3, &vocab).unwrap(), None);
    assert_eq!(state.rnnt_state.last_token(), 3);
    assert!(state.tokens.is_empty());
    assert!(state.text.is_empty());

    // Blank-only frames and an empty vocabulary produce nothing.
    let blank = decoder(&[0.0, 0.0, 0.0, 0.0, 0.0, 10.0]);
    let mut state = RnntStreamState::new(&blank);
    assert_eq!(state.step(&blank, &frame(0), 3, &vocab).unwrap(), None);
    let hi = decoder(&[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]);
    let mut state = RnntStreamState::new(&hi);
    assert_eq!(state.step(&hi, &frame(0), 3, &[]).unwrap(), None);
    assert_eq!(state.rnnt_state.last_token(), hi.blank_id());
}
