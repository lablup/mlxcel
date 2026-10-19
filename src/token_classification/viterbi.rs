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

//! Constrained BIOES Viterbi decoding, a port of `mlx_vlm.privacy_filter.
//! ViterbiDecoder`.
//!
//! The lattice has one state per label. Legal transitions:
//!
//! - `O` from `O` (`background_stay`) or from a closed state, an `E` or `S`
//!   of any entity (`end_to_background`);
//! - every `B` and `S` from `O` (`background_to_start`) or from the best
//!   closed state (`end_to_start`);
//! - an entity's `I` and `E` from that entity's `B` or `I`
//!   (`inside_to_continue` / `inside_to_end`).
//!
//! Start states are `O`, every `B` and every `S`; end states are `O`, every
//! `E` and every `S`. Ties take the first option listed, and among closed or
//! end states the first candidate in upstream's order (every entity's `E`,
//! then every `S`; `O` first among end states), exactly as upstream's `>=`
//! and `np.argmax` over those gathered arrays do. Scores are f32 with a `-1e9` sentinel, not `-inf`, so the
//! sums stay finite as upstream's do.

use anyhow::{Result, bail};
use serde_json::Value;

use super::spans::{Boundary, LabelInfo, build_label_info};

/// The six calibrated transition biases, in upstream key order.
pub const VITERBI_BIAS_KEYS: [&str; 6] = [
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
];

/// Additive log-domain biases on the lattice transitions. All zero means the
/// plain constrained decoder.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TransitionBiases {
    pub background_stay: f32,
    pub background_to_start: f32,
    pub inside_to_continue: f32,
    pub inside_to_end: f32,
    pub end_to_background: f32,
    pub end_to_start: f32,
}

impl TransitionBiases {
    /// Read the six keys from a calibration `biases` object; every key must
    /// be present and numeric. Unknown keys are ignored, as upstream's loader
    /// copies only the known ones.
    pub fn from_json(biases: &Value) -> Result<Self> {
        let Some(map) = biases.as_object() else {
            bail!("Viterbi calibration biases must be an object, got {biases}");
        };
        let missing: Vec<&str> = VITERBI_BIAS_KEYS
            .iter()
            .copied()
            .filter(|key| !map.contains_key(*key))
            .collect();
        if !missing.is_empty() {
            let mut missing = missing;
            missing.sort_unstable();
            bail!("Viterbi calibration is missing biases: {missing:?}");
        }
        let read = |key: &str| -> Result<f32> {
            let Some(value) = map[key].as_f64() else {
                bail!(
                    "Viterbi calibration bias {key} must be a number, got {}",
                    map[key]
                );
            };
            Ok(value as f32)
        };
        Ok(Self {
            background_stay: read(VITERBI_BIAS_KEYS[0])?,
            background_to_start: read(VITERBI_BIAS_KEYS[1])?,
            inside_to_continue: read(VITERBI_BIAS_KEYS[2])?,
            inside_to_end: read(VITERBI_BIAS_KEYS[3])?,
            end_to_background: read(VITERBI_BIAS_KEYS[4])?,
            end_to_start: read(VITERBI_BIAS_KEYS[5])?,
        })
    }
}

const NEGATIVE_INFINITY: f32 = -1e9;

/// Linear-time constrained BIOES decoder.
pub struct ViterbiDecoder {
    label_info: LabelInfo,
    biases: TransitionBiases,
    /// Output index of each entity's `B`, `I`, `E`, `S`, in entity order.
    b: Vec<usize>,
    i: Vec<usize>,
    e: Vec<usize>,
    s: Vec<usize>,
    /// `O`, every `B`, every `S`.
    start_states: Vec<usize>,
    /// `O`, every `E`, every `S`.
    end_states: Vec<usize>,
    /// Every `E`, then every `S`.
    closed_states: Vec<usize>,
}

impl ViterbiDecoder {
    /// Index the tag set; every entity must carry all four of B, I, E, S.
    pub fn new(labels: &[String], biases: TransitionBiases) -> Result<Self> {
        let label_info = build_label_info(labels)?;
        let (mut b, mut i, mut e, mut s) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for name in &label_info.span_names[1..] {
            let states = &label_info.states_by_span[name];
            let missing: Vec<&str> = [
                (Boundary::Begin, "B"),
                (Boundary::End, "E"),
                (Boundary::Inside, "I"),
                (Boundary::Single, "S"),
            ]
            .iter()
            .filter(|(boundary, _)| !states.contains_key(boundary))
            .map(|(_, letter)| *letter)
            .collect();
            if !missing.is_empty() {
                bail!("Privacy Filter labels for '{name}' are missing {missing:?}");
            }
            b.push(states[&Boundary::Begin]);
            i.push(states[&Boundary::Inside]);
            e.push(states[&Boundary::End]);
            s.push(states[&Boundary::Single]);
        }
        let background = label_info.background;
        let start_states = [vec![background], b.clone(), s.clone()].concat();
        let end_states = [vec![background], e.clone(), s.clone()].concat();
        let closed_states = [e.clone(), s.clone()].concat();
        Ok(Self {
            label_info,
            biases,
            b,
            i,
            e,
            s,
            start_states,
            end_states,
            closed_states,
        })
    }

    pub fn label_info(&self) -> &LabelInfo {
        &self.label_info
    }

    pub fn biases(&self) -> &TransitionBiases {
        &self.biases
    }

    /// First maximum among `states`, as `np.argmax` over the gathered scores.
    fn best_of(scores: &[f32], states: &[usize]) -> (usize, f32) {
        let mut best = (states[0], scores[states[0]]);
        for &state in &states[1..] {
            if scores[state] > best.1 {
                best = (state, scores[state]);
            }
        }
        best
    }

    /// The best legal label path for `emissions[token][label]`.
    ///
    /// Rows must have `labels().len()` entries; a shorter row panics, so the
    /// pipeline checks widths before calling. Empty input decodes to an
    /// empty path.
    pub fn decode(&self, emissions: &[Vec<f32>]) -> Vec<usize> {
        let length = emissions.len();
        if length == 0 {
            return Vec::new();
        }
        let num_labels = self.label_info.labels.len();
        let background = self.label_info.background;
        let biases = &self.biases;

        let mut scores = vec![NEGATIVE_INFINITY; num_labels];
        for &state in &self.start_states {
            scores[state] = emissions[0][state];
        }
        let mut backpointers: Vec<Vec<usize>> = Vec::with_capacity(length.saturating_sub(1));

        for emission in emissions.iter().skip(1) {
            let mut next = vec![NEGATIVE_INFINITY; num_labels];
            let mut pointers = vec![usize::MAX; num_labels];

            let (best_closed_state, best_closed) = Self::best_of(&scores, &self.closed_states);

            let stay = scores[background] + biases.background_stay;
            let close = best_closed + biases.end_to_background;
            let (background_score, background_prev) = if stay >= close {
                (stay, background)
            } else {
                (close, best_closed_state)
            };
            next[background] = background_score + emission[background];
            pointers[background] = background_prev;

            let open = scores[background] + biases.background_to_start;
            let reopen = best_closed + biases.end_to_start;
            let (start_score, start_prev) = if open >= reopen {
                (open, background)
            } else {
                (reopen, best_closed_state)
            };
            for &state in self.b.iter().chain(&self.s) {
                next[state] = start_score + emission[state];
                pointers[state] = start_prev;
            }

            for entity in 0..self.b.len() {
                let (b, i, e) = (self.b[entity], self.i[entity], self.e[entity]);
                let (best_open, best_open_state) = if scores[i] > scores[b] {
                    (scores[i], i)
                } else {
                    (scores[b], b)
                };
                next[i] = best_open + biases.inside_to_continue + emission[i];
                next[e] = best_open + biases.inside_to_end + emission[e];
                pointers[i] = best_open_state;
                pointers[e] = best_open_state;
            }

            backpointers.push(pointers);
            scores = next;
        }

        let (mut last, _) = Self::best_of(&scores, &self.end_states);
        let mut path = vec![0usize; length];
        path[length - 1] = last;
        for step in (0..length - 1).rev() {
            last = backpointers[step][last];
            path[step] = last;
        }
        path
    }
}

#[cfg(test)]
#[path = "viterbi_tests.rs"]
mod viterbi_tests;
