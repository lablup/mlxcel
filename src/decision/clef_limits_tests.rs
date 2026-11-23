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

//! Clef request limits: the token cap enforced while the schema is encoded
//! and before each video is decoded, the placeholder-token guard, and the
//! path-free video error text. The tokenizer tests skip when
//! `models/clef-flash-mlx-8bit` is absent.

use std::cell::Cell;
use std::path::PathBuf;

use serde_json::json;

use super::DecisionError;
use super::MediaPresence;
use super::clef::video_error_message;
use super::clef_sequence::{
    ClefTokenizer, MAX_LENGTH, check_placeholders, gated_media_block, image_block,
    placeholder_count, sequence,
};
use super::clef_tests::{checkpoint, validated};
use crate::multimodal::video::VideoError;

fn tokenizer() -> Option<ClefTokenizer> {
    checkpoint(false).map(|dir| ClefTokenizer::load(&dir).expect("tokenizer"))
}

fn message(error: DecisionError) -> String {
    match error {
        DecisionError::InvalidInput(message) => message,
        other => panic!("not InvalidInput: {other:?}"),
    }
}

#[test]
fn an_oversized_schema_stops_encoding_at_the_cap() {
    let Some(tokenizer) = tokenizer() else {
        return;
    };
    // Each instruction is about MAX_LENGTH / 2 tokens; the third question is
    // never reached, so the count is a lower bound.
    let half = "word ".repeat(MAX_LENGTH / 2);
    let questions = validated(
        &json!({
            "a": {"type": "bool", "instructions": half},
            "b": {"type": "bool", "instructions": half},
            "c": {"type": "bool", "instructions": half},
        }),
        MediaPresence::default(),
    );
    let text = message(sequence(&tokenizer, None, &questions, &[]).expect_err("too long"));
    assert!(text.starts_with("schema requires at least "), "{text}");
    assert!(
        text.ends_with(" tokens before state; maximum is 16384"),
        "{text}"
    );
    let count: usize = text["schema requires at least ".len()..]
        .split(' ')
        .next()
        .and_then(|n| n.parse().ok())
        .expect("count");
    assert!(
        count > MAX_LENGTH && count < MAX_LENGTH + MAX_LENGTH / 2,
        "{count}"
    );

    // Overflow in the last piece of the schema keeps upstream's exact count.
    let long = "word ".repeat(MAX_LENGTH + 10);
    let questions = validated(
        &json!({"q": {"type": "choice", "criteria": {"a": "short", "z": long}}}),
        MediaPresence::default(),
    );
    let text = message(sequence(&tokenizer, None, &questions, &[]).expect_err("too long"));
    assert!(!text.contains("at least"), "{text}");
}

#[test]
fn videos_past_the_cap_are_never_decoded() {
    let Some(tokenizer) = tokenizer() else {
        return;
    };
    let calls = Cell::new(0usize);
    let prepare = |_index: usize| {
        calls.set(calls.get() + 1);
        Ok(((), 6_000usize, vec![0.0]))
    };
    // Two 6000-token videos fit beside 1000 text tokens; the third pushes the
    // block past the cap, so the fourth and fifth are refused undecoded.
    let error = gated_media_block(&tokenizer, &[], 5, 1_000, prepare).expect_err("over");
    assert_eq!(calls.get(), 3);
    assert!(message(error).starts_with("schema requires at least "));

    // Images that alone overflow refuse every video before decoding it.
    calls.set(0);
    let error = gated_media_block(&tokenizer, &[MAX_LENGTH], 2, 10, prepare).expect_err("over");
    assert_eq!(calls.get(), 0);
    assert!(message(error).contains("maximum is 16384"));

    // Within the cap every video is prepared, in order.
    calls.set(0);
    let (ids, videos) = gated_media_block(&tokenizer, &[4], 2, 10, prepare).expect("fits");
    assert_eq!((calls.get(), videos.len()), (2, 2));
    assert_eq!(placeholder_count(&tokenizer, &ids), 4 + 2 * 6_000);
}

#[test]
fn placeholder_tokens_in_the_text_are_refused_with_media() {
    let Some(tokenizer) = tokenizer() else {
        return;
    };
    let questions = validated(&json!({"q": {"type": "bool"}}), MediaPresence::default());
    let mut block = image_block(&tokenizer, 4);
    block.extend(tokenizer.encode("\n").expect("newline"));
    let pads = placeholder_count(&tokenizer, &block);
    assert_eq!(pads, 4);

    let clean = sequence(&tokenizer, Some(&json!("a cat")), &questions, &block).expect("clean");
    check_placeholders(&tokenizer, &clean.ids, pads).expect("clean text passes");

    let state = json!("look <|image_pad|> here");
    let injected = sequence(&tokenizer, Some(&state), &questions, &block).expect("built");
    let text = message(check_placeholders(&tokenizer, &injected.ids, pads).expect_err("refused"));
    assert!(text.contains("<|image_pad|>"), "{text}");

    let questions = validated(
        &json!({"q": {"type": "bool", "instructions": "<|video_pad|>?"}}),
        MediaPresence::default(),
    );
    let injected = sequence(&tokenizer, Some(&json!("a cat")), &questions, &block).expect("built");
    assert!(check_placeholders(&tokenizer, &injected.ids, pads).is_err());
}

#[test]
fn video_errors_do_not_name_the_server_path() {
    let path = PathBuf::from("/var/tmp/mlxcel-video-abc123/clip.mp4");
    let errors = [
        VideoError::EmptyVideo(path.clone()),
        VideoError::Probe {
            path: path.clone(),
            message: format!("{}: Invalid data found", path.display()),
        },
        VideoError::Extract {
            path: path.clone(),
            message: "4 decoded frame(s) exceed the budget".to_string(),
        },
    ];
    for error in &errors {
        let text = video_error_message(error);
        assert!(!text.contains("/var/tmp"), "{text}");
        assert!(text.contains("the video"), "{text}");
    }
    let limit = VideoError::DurationTooLong {
        seconds: 900.0,
        max_seconds: 600.0,
    };
    assert_eq!(video_error_message(&limit), limit.to_string());
}
