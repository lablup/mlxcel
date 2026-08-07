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

//! Florence-2 task tokens and their prompt expansions.
//!
//! Florence-2 selects its behavior with a task marker such as `<OD>` or
//! `<CAPTION>`. These markers are **not vocabulary entries**: the checkpoint's
//! tokenizer has no id for any of them. The processor replaces the marker with
//! a plain English sentence *before* tokenization, and it is that sentence the
//! encoder sees. `<OD>` really means "Locate the objects with category name in
//! the image."
//!
//! Seven of the fifteen tasks additionally interpolate a caller-supplied
//! string into the prompt: a caption to ground, a phrase to segment, or a
//! region expressed as four `<loc_*>` tokens.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py
//! (`Florence2Processor.task_prompts_without_inputs`,
//! `task_prompts_with_input`, `tasks_answer_post_processing_type`,
//! `_construct_prompts`).

use std::fmt;
use std::str::FromStr;

use super::postprocess::Florence2PostProcessingType;

/// One of the fifteen Florence-2 task modes.
///
/// Not `#[non_exhaustive]`: the set is fixed by the released checkpoints'
/// training recipe, and callers matching exhaustively over it is the point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Florence2Task {
    Ocr,
    OcrWithRegion,
    Caption,
    DetailedCaption,
    MoreDetailedCaption,
    ObjectDetection,
    DenseRegionCaption,
    RegionProposal,
    CaptionToPhraseGrounding,
    ReferringExpressionSegmentation,
    RegionToSegmentation,
    OpenVocabularyDetection,
    RegionToCategory,
    RegionToDescription,
    RegionToOcr,
}

use Florence2Task::*;

impl Florence2Task {
    /// Every task, in the order upstream declares them.
    pub const ALL: [Florence2Task; 15] = [
        Ocr,
        OcrWithRegion,
        Caption,
        DetailedCaption,
        MoreDetailedCaption,
        ObjectDetection,
        DenseRegionCaption,
        CaptionToPhraseGrounding,
        ReferringExpressionSegmentation,
        RegionToSegmentation,
        OpenVocabularyDetection,
        RegionToCategory,
        RegionToDescription,
        RegionToOcr,
        RegionProposal,
    ];

    /// The task marker, for example `"<OD>"`. Callers pass this string form
    /// through a CLI or an API; it never reaches the tokenizer.
    pub fn token(self) -> &'static str {
        match self {
            Ocr => "<OCR>",
            OcrWithRegion => "<OCR_WITH_REGION>",
            Caption => "<CAPTION>",
            DetailedCaption => "<DETAILED_CAPTION>",
            MoreDetailedCaption => "<MORE_DETAILED_CAPTION>",
            ObjectDetection => "<OD>",
            DenseRegionCaption => "<DENSE_REGION_CAPTION>",
            RegionProposal => "<REGION_PROPOSAL>",
            CaptionToPhraseGrounding => "<CAPTION_TO_PHRASE_GROUNDING>",
            ReferringExpressionSegmentation => "<REFERRING_EXPRESSION_SEGMENTATION>",
            RegionToSegmentation => "<REGION_TO_SEGMENTATION>",
            OpenVocabularyDetection => "<OPEN_VOCABULARY_DETECTION>",
            RegionToCategory => "<REGION_TO_CATEGORY>",
            RegionToDescription => "<REGION_TO_DESCRIPTION>",
            RegionToOcr => "<REGION_TO_OCR>",
        }
    }

    /// Whether this task interpolates a caller-supplied string into its
    /// prompt.
    pub fn takes_input(self) -> bool {
        self.prompt_template().contains("{input}")
    }

    /// How the model's answer for this task must be parsed.
    pub fn post_processing_type(self) -> Florence2PostProcessingType {
        use Florence2PostProcessingType as P;
        match self {
            Ocr | Caption | DetailedCaption | MoreDetailedCaption | RegionToCategory
            | RegionToDescription | RegionToOcr => P::PureText,
            OcrWithRegion => P::Ocr,
            ObjectDetection | DenseRegionCaption => P::DescriptionWithBoxes,
            RegionProposal => P::Boxes,
            CaptionToPhraseGrounding => P::PhraseGrounding,
            ReferringExpressionSegmentation | RegionToSegmentation => P::Polygons,
            OpenVocabularyDetection => P::DescriptionWithBoxesOrPolygons,
        }
    }

    /// The raw prompt sentence, with `{input}` still in place for the tasks
    /// that take one.
    ///
    /// Copied byte for byte from the checkpoint, including the inconsistencies
    /// upstream carries: `<OCR_WITH_REGION>` has a comma before "with
    /// regions", `<REFERRING_EXPRESSION_SEGMENTATION>` and
    /// `<REGION_TO_SEGMENTATION>` end without a period while their neighbours
    /// have one. These are part of the trained input distribution, so
    /// "tidying" them changes what the model sees.
    fn prompt_template(self) -> &'static str {
        match self {
            Ocr => "What is the text in the image?",
            OcrWithRegion => "What is the text in the image, with regions?",
            Caption => "What does the image describe?",
            DetailedCaption => "Describe in detail what is shown in the image.",
            MoreDetailedCaption => "Describe with a paragraph what is shown in the image.",
            ObjectDetection => "Locate the objects with category name in the image.",
            DenseRegionCaption => "Locate the objects in the image, with their descriptions.",
            RegionProposal => "Locate the region proposals in the image.",
            CaptionToPhraseGrounding => "Locate the phrases in the caption: {input}",
            ReferringExpressionSegmentation => "Locate {input} in the image with mask",
            RegionToSegmentation => "What is the polygon mask of region {input}",
            OpenVocabularyDetection => "Locate {input} in the image.",
            RegionToCategory => "What is the region {input}?",
            RegionToDescription => "What does the region {input} describe?",
            RegionToOcr => "What text is in the region {input}?",
        }
    }

    /// Expand this task into the English sentence the tokenizer receives.
    ///
    /// Deviation from upstream, deliberate: a missing input on a task that
    /// needs one, or an input supplied to a task that does not, is an error
    /// here. Upstream's `_construct_prompts` scans for the task token inside a
    /// free-form string, so `"<REGION_TO_CATEGORY>"` with nothing after it
    /// silently expands to `"What is the region ?"` and the model answers
    /// nonsense. There is no way to tell that apart from a real prompt after
    /// the fact, so it is rejected at the boundary instead.
    pub fn expand(self, input: Option<&str>) -> Result<String, String> {
        let template = self.prompt_template();
        match (self.takes_input(), input) {
            (false, None) => Ok(template.to_string()),
            (false, Some(_)) => Err(format!(
                "Florence-2 task {} takes no input text",
                self.token()
            )),
            (true, None) => Err(format!(
                "Florence-2 task {} requires input text",
                self.token()
            )),
            (true, Some(text)) => Ok(template.replace("{input}", text)),
        }
    }
}

impl fmt::Display for Florence2Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.token())
    }
}

impl FromStr for Florence2Task {
    type Err = String;

    /// Parse a task marker. Accepts the bare name as well (`"OD"` for
    /// `"<OD>"`) and is case insensitive, so a CLI flag does not have to fight
    /// shell quoting to pass angle brackets.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        let bare = trimmed
            .strip_prefix('<')
            .and_then(|rest| rest.strip_suffix('>'))
            .unwrap_or(trimmed);
        Self::ALL
            .into_iter()
            .find(|task| {
                let token = task.token();
                token[1..token.len() - 1].eq_ignore_ascii_case(bare)
            })
            .ok_or_else(|| format!("unknown Florence-2 task: {s:?}"))
    }
}

#[cfg(test)]
#[path = "florence2_tasks_tests.rs"]
mod florence2_tasks_tests;
