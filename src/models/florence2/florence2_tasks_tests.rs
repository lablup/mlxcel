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

//! Task markers, prompt expansions, and post-processing routing.
//!
//! The expected strings are transcribed from the checkpoint's own
//! `processing_florence2.py` and pinned here byte for byte: a prompt that
//! drifts by a character is still a valid sentence, so nothing downstream
//! would catch it, but the model was trained on the exact wording.

use super::*;
use Florence2PostProcessingType as P;

/// The fifteen tasks and their exact expansions, as
/// `task_prompts_without_inputs` and `task_prompts_with_input` declare them.
const EXPECTED: &[(Florence2Task, &str, &str, P)] = &[
    (Ocr, "<OCR>", "What is the text in the image?", P::PureText),
    (
        OcrWithRegion,
        "<OCR_WITH_REGION>",
        "What is the text in the image, with regions?",
        P::Ocr,
    ),
    (
        Caption,
        "<CAPTION>",
        "What does the image describe?",
        P::PureText,
    ),
    (
        DetailedCaption,
        "<DETAILED_CAPTION>",
        "Describe in detail what is shown in the image.",
        P::PureText,
    ),
    (
        MoreDetailedCaption,
        "<MORE_DETAILED_CAPTION>",
        "Describe with a paragraph what is shown in the image.",
        P::PureText,
    ),
    (
        ObjectDetection,
        "<OD>",
        "Locate the objects with category name in the image.",
        P::DescriptionWithBoxes,
    ),
    (
        DenseRegionCaption,
        "<DENSE_REGION_CAPTION>",
        "Locate the objects in the image, with their descriptions.",
        P::DescriptionWithBoxes,
    ),
    (
        RegionProposal,
        "<REGION_PROPOSAL>",
        "Locate the region proposals in the image.",
        P::Boxes,
    ),
    (
        CaptionToPhraseGrounding,
        "<CAPTION_TO_PHRASE_GROUNDING>",
        "Locate the phrases in the caption: {input}",
        P::PhraseGrounding,
    ),
    (
        ReferringExpressionSegmentation,
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "Locate {input} in the image with mask",
        P::Polygons,
    ),
    (
        RegionToSegmentation,
        "<REGION_TO_SEGMENTATION>",
        "What is the polygon mask of region {input}",
        P::Polygons,
    ),
    (
        OpenVocabularyDetection,
        "<OPEN_VOCABULARY_DETECTION>",
        "Locate {input} in the image.",
        P::DescriptionWithBoxesOrPolygons,
    ),
    (
        RegionToCategory,
        "<REGION_TO_CATEGORY>",
        "What is the region {input}?",
        P::PureText,
    ),
    (
        RegionToDescription,
        "<REGION_TO_DESCRIPTION>",
        "What does the region {input} describe?",
        P::PureText,
    ),
    (
        RegionToOcr,
        "<REGION_TO_OCR>",
        "What text is in the region {input}?",
        P::PureText,
    ),
];

#[test]
fn every_task_is_covered_exactly_once() {
    assert_eq!(EXPECTED.len(), Florence2Task::ALL.len());
    for task in Florence2Task::ALL {
        let hits = EXPECTED.iter().filter(|(t, ..)| *t == task).count();
        assert_eq!(hits, 1, "{task} appears {hits} times in the expected table");
    }
}

#[test]
fn tokens_and_post_processing_types_match_the_checkpoint() {
    for (task, token, _, post) in EXPECTED {
        assert_eq!(task.token(), *token);
        assert_eq!(task.to_string(), *token);
        assert_eq!(task.post_processing_type(), *post, "{token}");
    }
}

/// The round trip the acceptance criteria call for: marker in, prompt out.
#[test]
fn prompt_expansion_matches_the_checkpoint() {
    for (task, token, template, _) in EXPECTED {
        if task.takes_input() {
            assert!(template.contains("{input}"), "{token}");
            let expanded = task.expand(Some("a green car")).expect("expand with input");
            assert_eq!(expanded, template.replace("{input}", "a green car"));
            assert!(!expanded.contains("{input}"), "{token} left a placeholder");
        } else {
            assert!(!template.contains("{input}"), "{token}");
            assert_eq!(task.expand(None).expect("expand"), *template);
        }
    }
}

/// Exactly seven tasks interpolate an input, matching
/// `task_prompts_with_input`.
#[test]
fn seven_tasks_take_an_input() {
    let with_input = Florence2Task::ALL
        .iter()
        .filter(|t| t.takes_input())
        .count();
    assert_eq!(with_input, 7);
}

/// Upstream would expand a missing input to a literal gap ("What is the region
/// ?") and let the model answer nonsense. Both mismatches are errors here.
#[test]
fn input_arity_mismatches_are_rejected() {
    assert!(RegionToCategory.expand(None).is_err());
    assert!(Caption.expand(Some("something")).is_err());

    let err = ObjectDetection.expand(Some("x")).expect_err("must reject");
    assert!(err.contains("<OD>"), "{err}");
    let err = RegionToOcr.expand(None).expect_err("must reject");
    assert!(err.contains("<REGION_TO_OCR>"), "{err}");
}

/// A region input is itself location tokens, and it must survive expansion
/// untouched so the tokenizer sees real vocabulary entries.
#[test]
fn region_inputs_pass_through_verbatim() {
    let region = "<loc_52><loc_332><loc_932><loc_774>";
    assert_eq!(
        RegionToCategory.expand(Some(region)).expect("expand"),
        format!("What is the region {region}?")
    );
}

#[test]
fn task_markers_parse_from_strings() {
    for task in Florence2Task::ALL {
        assert_eq!(task.token().parse::<Florence2Task>(), Ok(task));
    }
    // Bare names and mixed case, so a CLI flag need not quote angle brackets.
    assert_eq!("OD".parse(), Ok(ObjectDetection));
    assert_eq!("  <caption>  ".parse(), Ok(Caption));
    assert_eq!("ocr_with_region".parse(), Ok(OcrWithRegion));

    // `<CAPTION>` is a prefix of `<CAPTION_TO_PHRASE_GROUNDING>` up to the
    // underscore; matching must be exact, not prefix based.
    assert_eq!("<CAPTION>".parse(), Ok(Caption));
    assert_eq!(
        "<CAPTION_TO_PHRASE_GROUNDING>".parse(),
        Ok(CaptionToPhraseGrounding)
    );

    assert!("<NOT_A_TASK>".parse::<Florence2Task>().is_err());
    assert!("".parse::<Florence2Task>().is_err());
}
