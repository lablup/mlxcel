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

//! Decider-2b score rows and answers: the host-side half of upstream's
//! `predict` and `_score_rows`, with no MLX in it.

use std::collections::HashMap;
use std::ops::Range;

use serde_json::{Map, Value, json};

use super::decider2_prompt::{RenderedQuestion, isolated_level_instruction};
use super::{Answer, Question};

/// Most padded tokens one forward may hold (`rows * longest <= 8192`).
pub(crate) const GROUP_TOKEN_BUDGET: usize = 8192;
/// Each group is right-padded to a multiple of this.
pub(crate) const PAD_MULTIPLE: usize = 64;

/// `decision_config` temperatures.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Temperatures {
    /// Fallback for a rendered type without its own entry.
    pub(crate) default: f32,
    /// Keyed by rendered type (`choice`, `score`, `noul`).
    pub(crate) by_type: HashMap<String, f32>,
}

impl Temperatures {
    fn for_kind(&self, kind: &str) -> f32 {
        self.by_type.get(kind).copied().unwrap_or(self.default)
    }
}

/// One scored prompt: a question (or one isolated score level) and its
/// options.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Row {
    /// The instruction placed after `Question:`.
    pub(crate) instruction: String,
    /// The option lines.
    pub(crate) options: Vec<String>,
    /// The calibration temperature the logits are divided by.
    pub(crate) temperature: f32,
}

/// Where a question's rows sit in the row list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Slot {
    /// First row.
    pub(crate) start: usize,
    /// Row count (the level count for an isolated score).
    pub(crate) count: usize,
    /// Whether the rows are isolated score levels.
    pub(crate) isolated: bool,
}

/// Upstream's row plan: an isolated score question becomes one `no`/`yes`
/// row per level at the score temperature, every other question one row.
pub(crate) fn plan_rows(
    rendered: &[RenderedQuestion],
    temperatures: &Temperatures,
    isolated_levels: bool,
) -> (Vec<Row>, Vec<Slot>) {
    let mut rows = Vec::new();
    let mut slots = Vec::with_capacity(rendered.len());
    for question in rendered {
        let temperature = temperatures.for_kind(question.kind);
        let start = rows.len();
        if isolated_levels && question.kind == "score" {
            for level in &question.legend {
                rows.push(Row {
                    instruction: isolated_level_instruction(&question.instruction, level),
                    options: vec!["no".to_string(), "yes".to_string()],
                    temperature,
                });
            }
            slots.push(Slot {
                start,
                count: question.legend.len(),
                isolated: true,
            });
        } else {
            rows.push(Row {
                instruction: question.instruction.clone(),
                options: question.options.clone(),
                temperature,
            });
            slots.push(Slot {
                start,
                count: 1,
                isolated: false,
            });
        }
    }
    (rows, slots)
}

/// Consecutive rows grouped while `rows_in_group * longest <= 8192`; a row
/// longer than the budget runs alone.
pub(crate) fn group_rows(lengths: &[usize]) -> Vec<Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0;
    while start < lengths.len() {
        let mut end = start + 1;
        let mut longest = lengths[start];
        while end < lengths.len() {
            let candidate = longest.max(lengths[end]);
            if (end - start + 1) * candidate > GROUP_TOKEN_BUDGET {
                break;
            }
            longest = candidate;
            end += 1;
        }
        groups.push(start..end);
        start = end;
    }
    groups
}

/// The padded length of a group: its longest prompt rounded up to 64.
pub(crate) fn padded_length(longest: usize) -> usize {
    longest.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE
}

/// `usage.input_tokens`: the prefix every prompt shares, counted once, plus
/// each prompt's remainder.
pub(crate) fn input_tokens(prompts: &[Vec<u32>]) -> usize {
    let Some(first) = prompts.first() else {
        return 0;
    };
    let shortest = prompts.iter().map(Vec::len).min().unwrap_or(0);
    let common = (0..shortest)
        .take_while(|&index| prompts.iter().all(|prompt| prompt[index] == first[index]))
        .count();
    common
        + prompts
            .iter()
            .map(|prompt| prompt.len() - common)
            .sum::<usize>()
}

/// `softmax(logits / temperature)` in f32, as upstream computes it.
pub(crate) fn calibrated_softmax(logits: &[f32], temperature: f32) -> Vec<f64> {
    let scaled: Vec<f32> = logits.iter().map(|logit| logit / temperature).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|value| (value - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.iter().map(|value| f64::from(value / total)).collect()
}

/// Python's `round(value, places)` for the magnitudes answers carry.
pub(crate) fn round_to(value: f64, places: usize) -> f64 {
    format!("{value:.places$}").parse().unwrap_or(value)
}

fn argmax(values: &[f64]) -> usize {
    values.iter().enumerate().fold(
        0,
        |best, (index, value)| {
            if *value > values[best] { index } else { best }
        },
    )
}

fn indexed(values: &[f64]) -> Value {
    Value::Object(
        values
            .iter()
            .enumerate()
            .map(|(index, value)| (index.to_string(), json!(round_to(*value, 4))))
            .collect(),
    )
}

/// Upstream's answer for one question from its rows' probabilities.
pub(crate) fn answer(
    question: &Question,
    rendered: &RenderedQuestion,
    slot: Slot,
    probabilities: &[Vec<f64>],
) -> Answer {
    let rows = &probabilities[slot.start..slot.start + slot.count];
    let (p, fit) = if slot.isolated {
        let fit: Vec<f64> = rows.iter().map(|row| row[1]).collect();
        let mass: f64 = fit.iter().sum();
        let divisor = if mass == 0.0 { 1e-9 } else { mass };
        (fit.iter().map(|value| value / divisor).collect(), Some(fit))
    } else {
        (rows[0].clone(), None)
    };
    if rendered.kind == "noul" {
        return Answer::bool(round_to(p[1], 4), Map::new());
    }
    let best = argmax(&p);
    let entropy: f64 = -p
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| value * value.ln())
        .sum::<f64>();
    let certainty = (1.0 - entropy / (p.len() as f64).ln()).max(0.0);
    let mut metadata = Map::new();
    metadata.insert("confidence".to_string(), json!(round_to(p[best], 4)));
    metadata.insert("certainty".to_string(), json!(round_to(certainty, 4)));
    if rendered.kind == "choice" {
        let labels = question.labels();
        return Answer::Choice {
            value: labels[best].name.clone(),
            probabilities: labels
                .iter()
                .zip(&p)
                .map(|(label, value)| (label.name.clone(), round_to(*value, 4)))
                .collect(),
            metadata,
        };
    }
    let legend: Map<String, Value> = rendered
        .legend
        .iter()
        .enumerate()
        .map(|(index, level)| (index.to_string(), Value::String(level.clone())))
        .collect();
    metadata.insert("legend".to_string(), Value::Object(legend));
    if let Some(fit) = fit {
        metadata.insert("level_fit".to_string(), indexed(&fit));
        metadata.insert(
            "fit_mass".to_string(),
            json!(round_to(fit.iter().sum::<f64>(), 4)),
        );
    }
    let expected: f64 = p
        .iter()
        .enumerate()
        .map(|(index, value)| index as f64 * value)
        .sum();
    Answer::Score {
        value: round_to(expected, 2),
        probabilities: p.iter().map(|value| round_to(*value, 4)).collect(),
        metadata,
    }
}
