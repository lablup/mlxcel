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

//! Tensor-rank screening for LoRA pairs, shared by every adapter path.
//!
//! Both application paths (fused and unfused runtime) only handle a pair of
//! 2-D matrices. mlx-lm's `LoRASwitchLinear`, which is how it adapts the
//! experts of a mixture-of-experts layer, writes 3-D `lora_a` / `lora_b`
//! tensors with one slice per expert. That used to reach the delta
//! computation and fail as a bare tensor-rank complaint; this names it as the
//! unsupported feature it is (issue #1577).
//!
//! Used by: `loader::validate_adapter_tensors`

/// The leading-axis length of a 3-D pair is the expert count.
const EXPERT_PAIR_RANK: usize = 3;

/// Collects pairs that are not two 2-D matrices, so the shared validator can
/// report them alongside every other violation in one error.
///
/// Expert adapters are aggregated into one line: an MoE adapter carries one on
/// every expert projection of every layer, and a line per pair would bury the
/// one fact the operator needs. Any other rank is reported per pair.
#[derive(Default)]
pub(super) struct PairRankScreen {
    /// `(count, first layer by name, its lora_a shape)`.
    expert_pairs: Option<(usize, String, Vec<i32>)>,
    other: Vec<String>,
}

impl PairRankScreen {
    /// Record one pair; returns `true` when both halves are 2-D.
    pub(super) fn admit(&mut self, layer: &str, a_shape: &[i32], b_shape: &[i32]) -> bool {
        if a_shape.len() == 2 && b_shape.len() == 2 {
            return true;
        }
        if a_shape.len() == EXPERT_PAIR_RANK || b_shape.len() == EXPERT_PAIR_RANK {
            let entry = self
                .expert_pairs
                .get_or_insert_with(|| (0, layer.to_string(), a_shape.to_vec()));
            entry.0 += 1;
            if layer < entry.1.as_str() {
                entry.1 = layer.to_string();
                entry.2 = a_shape.to_vec();
            }
        } else {
            self.other.push(format!(
                "{layer}.lora_a: not a pair of 2-D matrices (lora_a {a_shape:?}, lora_b \
                 {b_shape:?})"
            ));
        }
        false
    }

    /// Append the collected violations to `violations`.
    pub(super) fn report_into(self, violations: &mut Vec<String>) {
        if let Some((count, layer, a_shape)) = self.expert_pairs {
            violations.push(format!(
                "{layer}.lora_a: {}",
                moe_expert_adapter_reason(&a_shape, count)
            ));
        }
        violations.extend(self.other);
    }
}

/// The diagnostic for a mixture-of-experts expert adapter, without the
/// offending tensor's name so each caller can prefix its own context.
pub(super) fn moe_expert_adapter_reason(a_shape: &[i32], count: usize) -> String {
    let noun = if count == 1 { "pair" } else { "pairs" };
    format!(
        "mixture-of-experts expert adapters are not supported (3-D per-expert \
         lora_a {a_shape:?}, as mlx-lm's LoRASwitchLinear writes; {count} such {noun} in this \
         adapter). Only 2-D lora_a / lora_b pairs on dense projections can be applied; retrain \
         without the expert projections in the LoRA target set"
    )
}

#[cfg(test)]
#[path = "pair_rank_tests.rs"]
mod tests;
