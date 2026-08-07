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

//! Processor wiring. The tokenizer half needs the real checkpoint and skips
//! without it, following `tests/florence2_fusion_parity.rs`; everything else
//! runs everywhere.

use std::path::Path;

use super::super::coords::florence2_loc_token_id;
use super::*;

const MODEL_DIR: &str = "models/Florence-2-base-ft-bf16";

fn processor() -> Option<Florence2Processor> {
    if !Path::new(MODEL_DIR).exists() {
        eprintln!("skipping: {MODEL_DIR} not present");
        return None;
    }
    Some(Florence2Processor::from_pretrained(Path::new(MODEL_DIR)).expect("load processor"))
}

/// The prompt the fusion parity test pins, reached through the task API
/// instead of a hard-coded id list. `<CAPTION>` must expand, tokenize, and
/// wrap in `<s>` / `</s>` to exactly those eight ids.
#[test]
fn caption_prompt_tokenizes_to_the_pinned_ids() {
    let Some(processor) = processor() else { return };
    let ids = processor
        .encode_prompt(Florence2Task::Caption, None)
        .expect("encode prompt");
    assert_eq!(ids, vec![0, 2264, 473, 5, 2274, 6190, 116, 2]);
}

/// Every task's prompt must survive the round trip through the tokenizer.
/// `<s>` and `</s>` come from the checkpoint's `RobertaProcessing`
/// post-processor, so their presence is what proves special tokens were added.
#[test]
fn every_task_prompt_round_trips_through_the_tokenizer() {
    let Some(processor) = processor() else { return };
    for task in Florence2Task::ALL {
        let input = task
            .takes_input()
            .then_some("<loc_52><loc_332><loc_932><loc_774>");
        let prompt = task.expand(input).expect("expand");
        let ids = processor.encode_prompt(task, input).expect("encode");
        assert_eq!(ids.first(), Some(&0), "{task} must start with <s>");
        assert_eq!(ids.last(), Some(&2), "{task} must end with </s>");

        let decoded = processor.decode_answer(&ids).expect("decode");
        assert_eq!(
            decoded,
            format!("<s>{prompt}</s>"),
            "{task} did not round trip"
        );
    }
}

/// Location tokens are single vocabulary entries, so a region input must cost
/// four ids and come back verbatim. If they tokenized as plain text the
/// coordinates would be spelled out character by character.
#[test]
fn region_inputs_tokenize_as_single_location_tokens() {
    let Some(processor) = processor() else { return };
    let region = "<loc_52><loc_332><loc_932><loc_774>";
    let ids = processor
        .encode_prompt(Florence2Task::RegionToCategory, Some(region))
        .expect("encode");

    let expected: Vec<i32> = [52u32, 332, 932, 774]
        .iter()
        .map(|bin| florence2_loc_token_id(*bin).expect("in range") as i32)
        .collect();
    let found = ids
        .windows(expected.len())
        .any(|window| window == expected.as_slice());
    assert!(found, "expected {expected:?} inside {ids:?}");
}

/// Decoding with `skip_special_tokens = true` would delete every coordinate,
/// leaving the spatial tasks silently empty. This pins the opposite.
#[test]
fn decoding_keeps_location_tokens() {
    let Some(processor) = processor() else { return };
    let ids: Vec<i32> = [0u32, 50269, 50369, 51268, 2]
        .iter()
        .map(|id| *id as i32)
        .collect();
    let decoded = processor.decode_answer(&ids).expect("decode");
    assert!(decoded.contains("<loc_0>"), "{decoded}");
    assert!(decoded.contains("<loc_100>"), "{decoded}");
    assert!(decoded.contains("<loc_999>"), "{decoded}");
}

/// End to end without a model: the post-processing half of `run` must turn a
/// decoded answer into coordinates against the original image size.
#[test]
fn post_process_reaches_the_parsers() {
    let result = postprocess::post_process(
        "<s>car<loc_52><loc_332><loc_932><loc_774></s>",
        Florence2Task::ObjectDetection,
        Florence2ImageSize::new(1000, 1000),
    );
    let Florence2TaskResult::Boxes { boxes, labels } = result else {
        panic!("expected boxes");
    };
    assert_eq!(labels, vec!["car".to_string()]);
    assert_eq!(boxes[0].to_array(), [52.5, 332.5, 932.5, 774.5]);
}

#[test]
fn missing_checkpoint_directory_is_an_error() {
    // `Florence2Processor` holds MLX-backed handles and is not `Debug`, so
    // the error is inspected without going through `expect_err`.
    let message = match Florence2Processor::from_pretrained(Path::new("models/no-such-florence2")) {
        Ok(_) => panic!("loading a missing checkpoint must fail"),
        Err(e) => format!("{e:#}"),
    };
    assert!(message.contains("Florence-2"), "{message}");
}
