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

//! The d1 System One prompt and readout, ported from mlx-vlm's
//! `mlx_vlm/models/d1/prompt.py` (the published `D1Model` defaults: no system
//! turn, JSON states, described options, no lead, no calibration).
//!
//! Everything here is pure text and token-id logic over a [`TokenEncoder`],
//! so it is tested without weights. The forward pass lives in
//! [`super::d1`].

use std::sync::OnceLock;

use serde_json::Value;

use super::validate::python_repr;
use super::{Criteria, DecisionError, Question, QuestionKind};

/// Chat-turn markers of the LFM2 template.
pub(crate) const IM_START: &str = "<|im_start|>";
pub(crate) const IM_END: &str = "<|im_end|>";
/// What the LFM2-VL chat template writes for one image content item.
pub(crate) const IMAGE_MARKUP: &str = "<image>";
/// Largest image area d1 reads before its own bicubic pre-downscale.
pub(crate) const VISION_MAX_PIXELS: u64 = 1024 * 1024;
/// Most score levels the readout is defined for.
pub(crate) const MAX_SCORE_LEVELS: usize = 10;

const YES_FORMS: [&str; 3] = ["yes", "Yes", "YES"];
const NO_FORMS: [&str; 3] = ["no", "No", "NO"];

/// Text to token ids without special-token insertion
/// (`tokenizer.encode(text, add_special_tokens=False)`). Special tokens
/// written in the text, such as `<|im_start|>`, still parse as one id.
pub(crate) trait TokenEncoder {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, DecisionError>;
}

impl TokenEncoder for crate::tokenizer::MlxcelTokenizer {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        self.encode(text, false)
            .map_err(|err| DecisionError::Internal(format!("tokenizer failed: {err}")))
    }
}

fn invalid(message: impl Into<String>) -> DecisionError {
    DecisionError::InvalidInput(message.into())
}

/// Python's `str()` of a JSON value, as an f-string would render it.
fn python_str(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(other) => python_repr(other),
        None => "None".to_string(),
    }
}

/// A question in d1's normalized form (`as_question`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum D1Question {
    /// `choice`: `(label, description)` in request order.
    Choice {
        instructions: String,
        options: Vec<(String, Option<String>)>,
    },
    /// `score`: level descriptions, lowest first.
    Score {
        instructions: String,
        levels: Vec<String>,
    },
    /// `bool` (read as `noul`): the rendered `Yes:` / `No:` descriptions when
    /// the request gave a non-empty criteria map.
    Noul {
        instructions: String,
        descriptions: Option<(String, String)>,
    },
}

/// Normalize one validated question the way upstream's `as_question` does:
/// `instructions` is required, a score has 2 to 10 levels, and bool criteria
/// may only describe `true` and `false`.
pub(crate) fn as_question(question: &Question) -> Result<D1Question, DecisionError> {
    let instructions = match question.raw.get("instructions") {
        None => return Err(invalid("a question needs `instructions`")),
        Some(value) => python_str(Some(value)),
    };
    match question.kind {
        QuestionKind::Choice => Ok(D1Question::Choice {
            instructions,
            options: question
                .labels()
                .iter()
                .map(|label| (label.name.clone(), label.description.clone()))
                .collect(),
        }),
        QuestionKind::Score => {
            let levels = question.levels();
            if !(2..=MAX_SCORE_LEVELS).contains(&levels.len()) {
                return Err(invalid(
                    "a score needs criteria: a list of 2 to 10 level descriptions, lowest first",
                ));
            }
            Ok(D1Question::Score {
                instructions,
                levels: levels.to_vec(),
            })
        }
        QuestionKind::Bool => {
            let descriptions = match question.raw.get("criteria") {
                Some(Value::Object(map)) if !map.is_empty() => {
                    if map.keys().any(|k| k != "true" && k != "false") {
                        let mut keys: Vec<&String> = map.keys().collect();
                        keys.sort();
                        let keys: Vec<String> = keys
                            .into_iter()
                            .map(|k| python_repr(&Value::String(k.clone())))
                            .collect();
                        return Err(invalid(format!(
                            "bool criteria describe \"true\" and \"false\", got keys {}",
                            keys.join(", ")
                        )));
                    }
                    Some((python_str(map.get("true")), python_str(map.get("false"))))
                }
                _ => None,
            };
            debug_assert!(matches!(question.criteria, Criteria::Bool { .. }));
            Ok(D1Question::Noul {
                instructions,
                descriptions,
            })
        }
        other => Err(invalid(format!(
            "This model does not support {} decisions",
            python_repr(&Value::String(other.as_str().to_string()))
        ))),
    }
}

/// Option codes before the single-token check: the labels themselves when
/// every label is one letter, else `A`, `B`, ... up to 26 options, else
/// `00`, `01`, ....
pub(crate) fn option_codes(labels: &[String]) -> Vec<String> {
    let stripped: Vec<&str> = labels.iter().map(|label| label.trim()).collect();
    let single_letters = !stripped.is_empty()
        && stripped.iter().all(|label| {
            let mut chars = label.chars();
            matches!((chars.next(), chars.next()), (Some(c), None) if c.is_alphabetic())
        });
    if single_letters {
        return stripped.into_iter().map(str::to_string).collect();
    }
    if labels.len() <= 26 {
        return (0..labels.len())
            .map(|i| char::from(b'A' + i as u8).to_string())
            .collect();
    }
    (0..labels.len()).map(|i| format!("{i:02}")).collect()
}

/// Codes tried, in order, when an option's own code is not one unused token:
/// `A`-`Z`, `00`-`99`, `a`-`z`, `#0`-`#199`, `AA`-`ZZ`.
fn fallback_pool() -> &'static [String] {
    static POOL: OnceLock<Vec<String>> = OnceLock::new();
    POOL.get_or_init(|| {
        let upper = || (b'A'..=b'Z').map(char::from);
        let mut pool: Vec<String> = upper().map(String::from).collect();
        pool.extend((0..100).map(|i| format!("{i:02}")));
        pool.extend((b'a'..=b'z').map(|c| char::from(c).to_string()));
        pool.extend((0..200).map(|i| format!("#{i}")));
        pool.extend(upper().flat_map(|a| upper().map(move |b| format!("{a}{b}"))));
        pool
    })
}

/// A distinct single-token code for every label: `(code, token_id)`.
pub(crate) fn aliases(
    encoder: &dyn TokenEncoder,
    labels: &[String],
) -> Result<Vec<(String, u32)>, DecisionError> {
    let mut used: Vec<u32> = Vec::with_capacity(labels.len());
    let mut out: Vec<(String, u32)> = Vec::with_capacity(labels.len());
    let take = |raw: &str,
                used: &mut Vec<u32>,
                out: &mut Vec<(String, u32)>|
     -> Result<bool, DecisionError> {
        let ids = encoder.encode_text(raw)?;
        if ids.len() != 1 || used.contains(&ids[0]) {
            return Ok(false);
        }
        out.push((raw.to_string(), ids[0]));
        used.push(ids[0]);
        Ok(true)
    };
    for code in option_codes(labels) {
        if take(&code, &mut used, &mut out)? {
            continue;
        }
        let mut found = false;
        for raw in fallback_pool() {
            if take(raw, &mut used, &mut out)? {
                found = true;
                break;
            }
        }
        if !found {
            return Err(invalid(format!(
                "no single-token alias left for {} options",
                labels.len()
            )));
        }
    }
    Ok(out)
}

/// The single-token ids of `texts`, first occurrence only.
fn single_token_ids(encoder: &dyn TokenEncoder, texts: &[&str]) -> Result<Vec<u32>, DecisionError> {
    let mut out = Vec::with_capacity(texts.len());
    for text in texts {
        let ids = encoder.encode_text(text)?;
        if ids.len() == 1 && !out.contains(&ids[0]) {
            out.push(ids[0]);
        }
    }
    Ok(out)
}

/// Token ids scored for each option, one group per option (`yes` then `no`
/// for a noul). A tokenizer without the needed single-token forms is an
/// internal error.
pub(crate) fn readout_ids(
    encoder: &dyn TokenEncoder,
    question: &D1Question,
) -> Result<Vec<Vec<u32>>, DecisionError> {
    match question {
        D1Question::Noul { .. } => {
            let yes = single_token_ids(encoder, &YES_FORMS)?;
            let no = single_token_ids(encoder, &NO_FORMS)?;
            if yes.is_empty() || no.is_empty() {
                return Err(DecisionError::Internal(
                    "tokenizer has no single-token yes/no".to_string(),
                ));
            }
            Ok(vec![yes, no])
        }
        D1Question::Score { levels, .. } => {
            let mut groups = Vec::with_capacity(levels.len());
            for level in 0..levels.len() {
                let group = single_token_ids(encoder, &[&level.to_string()])?;
                if group.is_empty() {
                    return Err(DecisionError::Internal(format!(
                        "score with {} levels needs single-token digits",
                        levels.len()
                    )));
                }
                groups.push(group);
            }
            Ok(groups)
        }
        D1Question::Choice { options, .. } => {
            let labels: Vec<String> = options.iter().map(|(label, _)| label.clone()).collect();
            let mut groups = Vec::with_capacity(labels.len());
            for (code, id) in aliases(encoder, &labels)? {
                let mut group = vec![id];
                group.extend(
                    single_token_ids(encoder, &[&format!(" {code}")])?
                        .into_iter()
                        .filter(|other| *other != id),
                );
                groups.push(group);
            }
            if groups.is_empty() {
                return Err(DecisionError::Internal(
                    "choice with no options".to_string(),
                ));
            }
            Ok(groups)
        }
    }
}

/// Option probabilities: each option's best token log-probability, softmaxed
/// over the options without temperature.
pub(crate) fn readout(groups: &[Vec<u32>], logz: &[f32]) -> Result<Vec<f64>, DecisionError> {
    let mut scores = Vec::with_capacity(groups.len());
    for group in groups {
        let mut best = f64::NEG_INFINITY;
        for &id in group {
            let value = logz.get(id as usize).copied().ok_or_else(|| {
                DecisionError::Internal(format!(
                    "answer token {id} is outside the {}-entry vocabulary",
                    logz.len()
                ))
            })?;
            best = best.max(f64::from(value));
        }
        scores.push(best);
    }
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    Ok(exps.into_iter().map(|e| e / total).collect())
}

/// The state rendered ahead of the question: a string as is, any other value
/// as two-space-indented JSON with non-ASCII text kept.
pub(crate) fn state_block(state: &Value) -> String {
    match state {
        Value::String(text) => format!("{text}\n\n"),
        other => {
            let json = serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string());
            format!("{json}\n\n")
        }
    }
}

/// The question text up to (not including) the end of the user turn.
pub(crate) fn question_block(
    encoder: &dyn TokenEncoder,
    question: &D1Question,
) -> Result<String, DecisionError> {
    Ok(match question {
        D1Question::Choice {
            instructions,
            options,
        } => {
            let labels: Vec<String> = options.iter().map(|(label, _)| label.clone()).collect();
            let codes = aliases(encoder, &labels)?;
            let lines: Vec<String> = options
                .iter()
                .zip(&codes)
                .map(|((label, description), (code, _))| {
                    let text = match description.as_deref() {
                        Some(text) if !text.is_empty() => text.to_string(),
                        _ => label.replace('_', " "),
                    };
                    format!("{code} {text}")
                })
                .collect();
            format!(
                "{instructions}\n\nOptions:\n{}\n\nReply with the option code only.",
                lines.join("\n")
            )
        }
        D1Question::Noul {
            instructions,
            descriptions,
        } => {
            let extra = match descriptions {
                Some((yes, no)) => format!("\nYes: {yes}\nNo: {no}"),
                None => String::new(),
            };
            format!("{instructions}{extra}\n\nReply with yes or no only.")
        }
        D1Question::Score {
            instructions,
            levels,
        } => {
            let legend: Vec<String> = levels
                .iter()
                .enumerate()
                .map(|(i, level)| format!("{i} {level}"))
                .collect();
            format!(
                "{instructions}\n\n{}\n\nReply with a single digit 0-{} only.",
                legend.join("\n"),
                levels.len() - 1
            )
        }
    })
}

/// Everything before the question, shared by every question on one state.
pub(crate) fn prefix_text(state: Option<&Value>, bos: &str, image_count: usize) -> String {
    let images = IMAGE_MARKUP.repeat(image_count);
    let body = match state {
        None => String::new(),
        Some(state) => format!("{}\nQUESTION:\n", state_block(state)),
    };
    format!("{bos}{IM_START}user\n{images}{body}")
}

/// The question and the assistant header, up to the answer slot.
pub(crate) fn suffix_text(
    encoder: &dyn TokenEncoder,
    question: &D1Question,
) -> Result<String, DecisionError> {
    Ok(format!(
        "{}{IM_END}\n{IM_START}assistant\n",
        question_block(encoder, question)?
    ))
}

/// Trunk and branch token ids before image expansion. One question folds
/// its suffix into the trunk (a single pass, no branches); several share the
/// prefix as the trunk, one branch each.
pub(crate) fn prompt_ids(
    encoder: &dyn TokenEncoder,
    bos: &str,
    state: Option<&Value>,
    questions: &[D1Question],
    image_count: usize,
) -> Result<(Vec<i32>, Vec<Vec<i32>>), DecisionError> {
    let encode = |text: &str| -> Result<Vec<i32>, DecisionError> {
        Ok(encoder
            .encode_text(text)?
            .into_iter()
            .map(|id| id as i32)
            .collect())
    };
    let mut prefix = prefix_text(state, bos, image_count);
    let mut suffixes = questions
        .iter()
        .map(|q| suffix_text(encoder, q))
        .collect::<Result<Vec<_>, _>>()?;
    if suffixes.len() == 1 {
        prefix.push_str(&suffixes.remove(0));
    }
    let branches = suffixes
        .iter()
        .map(|suffix| encode(suffix))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((encode(&prefix)?, branches))
}

/// An image downscaled to at most [`VISION_MAX_PIXELS`] in area, bicubic
/// (PIL `BICUBIC`, the Catmull-Rom kernel), after conversion to RGB. Each side
/// is `max(1, floor(side * scale))`.
pub(crate) fn cap_pixels(image: &image::DynamicImage) -> image::DynamicImage {
    let rgb = image::DynamicImage::ImageRgb8(image.to_rgb8());
    let (width, height) = (u64::from(rgb.width()), u64::from(rgb.height()));
    if width * height <= VISION_MAX_PIXELS {
        return rgb;
    }
    let scale = (VISION_MAX_PIXELS as f64 / (width * height) as f64).sqrt();
    let side = |value: u64| ((value as f64 * scale) as u32).max(1);
    rgb.resize_exact(
        side(width),
        side(height),
        image::imageops::FilterType::CatmullRom,
    )
}

/// Branch batches, longest first, each listed in request order: at most
/// `budget` padded tokens per batch, trunk included, and no branch padded
/// past twice its length.
pub(crate) fn branch_chunks(lengths: &[usize], trunk: usize, budget: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    // Stable sort, as Python's `sorted`, so equal lengths keep request order.
    order.sort_by_key(|&i| std::cmp::Reverse(lengths[i]));
    let mut out = Vec::new();
    let mut rest = order.as_slice();
    while !rest.is_empty() {
        let longest = lengths[rest[0]];
        let mut size = 1;
        while size < rest.len()
            && (size + 1) * (trunk + longest) <= budget
            && 2 * lengths[rest[size]] >= longest
        {
            size += 1;
        }
        let mut chunk = rest[..size].to_vec();
        chunk.sort_unstable();
        out.push(chunk);
        rest = &rest[size..];
    }
    out
}
