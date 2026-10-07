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

//! Issue #2190: the batched Gemma 4 MTP adapter marks its verify forwards
//! the way the B = 1 adapter does, and on the row-wise geometries a B > 1
//! verify reproduces each row's standalone B = 1 linear verify bit for bit.
use mlxcel_core::speculative::mtp::target::MtpTarget;

use super::*;
use crate::models::gemma4::mtp_verify_log;
use crate::models::gemma4_tests::build_synthetic_wrapper_with_layer;

fn row_wise_wrapper(layer_type: &str) -> crate::models::gemma4::Gemma4Wrapper {
    let mut wrapper = build_synthetic_wrapper_with_layer(layer_type);
    wrapper.force_mtp_row_verify_for_test();
    wrapper
}

fn to_f32(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn row_of(array: &MlxArray, row: i32) -> Vec<f32> {
    to_f32(&mlxcel_core::utils::slice_axis(array, 0, row, row + 1))
}

/// AC-2: prefill forwards leave `mtp_verify` off and verify forwards set it,
/// on both the rectangular prefill and the per-row prefill. Before #2190 the
/// batched verify never set it, so the row-wise geometries' verify never
/// reached the decode-exact path.
#[test]
fn batched_adapter_marks_only_verify_forwards_as_mtp_verify() {
    let _runtime = crate::initialize_runtime();
    let sampler = SamplingConfig::greedy();
    for row_wise in [false, true] {
        let wrapper = if row_wise {
            row_wise_wrapper("sliding_attention")
        } else {
            build_synthetic_wrapper_with_layer("sliding_attention")
        };
        let adapter = Gemma4MtpBatchedTargetAdapter::new(&wrapper, 2);
        let _ = mtp_verify_log::take();
        let (bonuses, _) = adapter
            .prefill_and_seed_batched(&[vec![2, 3, 4], vec![5, 6, 7]], &sampler)
            .expect("prefill");
        let prefill = mtp_verify_log::take();
        assert!(!prefill.is_empty(), "prefill ran a sink-aware forward");
        assert!(
            prefill.iter().all(|&flag| !flag),
            "row_wise={row_wise}: prefill forwards must not be verify forwards: {prefill:?}"
        );
        let window: Vec<Vec<i32>> = bonuses.iter().map(|&b| vec![b, 1, 2]).collect();
        let _ = adapter
            .verify_forward_batched(&window, &sampler)
            .expect("verify");
        assert_eq!(
            mtp_verify_log::take(),
            [true],
            "row_wise={row_wise}: the verify forward is a verify forward"
        );
    }
}

/// One forced-accept round for both the batched adapter and the per-row B = 1
/// references, asserting every row's verify argmax and hidden state match.
struct Lockstep<'w> {
    batched: Gemma4MtpBatchedTargetAdapter<'w>,
    singles: Vec<Gemma4MtpTargetAdapter<'w>>,
    bonuses: Vec<i32>,
    label: String,
}

impl Lockstep<'_> {
    fn round(&mut self, drafts: [i32; 2], accepted: &[usize]) {
        let sampler = SamplingConfig::greedy();
        let logprobs = mlxcel_core::sampling::LogprobsConfig::default();
        let windows: Vec<Vec<i32>> = self
            .bonuses
            .iter()
            .map(|&b| vec![b, drafts[0], drafts[1]])
            .collect();
        let batched = self
            .batched
            .verify_forward_batched(&windows, &sampler)
            .expect("batched verify");
        let hidden = batched.captured.tensors[0].as_ref().unwrap();
        let mut next = Vec::with_capacity(self.singles.len());
        for (row, single) in self.singles.iter().enumerate() {
            let reference = single.verify_forward(&windows[row], &sampler, &logprobs);
            assert_eq!(
                batched.target_tokens_per_row[row], reference.target_tokens,
                "[{}] row {row}: verify argmax",
                self.label
            );
            assert_eq!(
                row_of(hidden, row as i32),
                to_f32(reference.captured.tensors[0].as_ref().unwrap()),
                "[{}] row {row}: verify hidden must be bit-identical",
                self.label
            );
            next.push(reference.target_tokens[accepted[row]]);
            let _ = single.verify_finalize(accepted[row], 3, reference.captured);
        }
        self.batched
            .verify_finalize_batched(accepted, 3, batched.captured)
            .expect("batched finalize");
        self.bonuses = next;
    }
}

/// The per-batch-row verify against independent B = 1 linear verifies, on
/// both attention families, for equal-length and ragged prompts, through
/// uniform and divergent rounds (rows accepting different counts, so later
/// rounds verify rows whose valid end lags the shared offset). The sliding
/// fixture's window is 8, so the ring wraps and per-row ring cursors matter.
#[test]
fn per_row_batched_verify_matches_b1_linear_verify() {
    let _runtime = crate::initialize_runtime();
    let sampler = SamplingConfig::greedy();
    let logprobs = mlxcel_core::sampling::LogprobsConfig::default();
    let accepts: [&[usize]; 5] = [&[0, 2, 1], &[2, 0, 1], &[1, 1, 2], &[0, 0, 0], &[2, 2, 0]];
    for layer_type in ["full_attention", "sliding_attention"] {
        for prompts in [
            vec![vec![2, 3, 4], vec![5, 6, 7], vec![1, 6, 2]],
            vec![vec![2, 3, 4], vec![5, 6, 7, 1, 2], vec![3]],
        ] {
            let batched_wrapper = row_wise_wrapper(layer_type);
            let single_wrappers: Vec<_> = prompts
                .iter()
                .map(|_| row_wise_wrapper(layer_type))
                .collect();
            let batched = Gemma4MtpBatchedTargetAdapter::new(&batched_wrapper, prompts.len());
            let (bonuses, _) = batched
                .prefill_and_seed_batched(&prompts, &sampler)
                .expect("per-row prefill");
            let singles: Vec<_> = single_wrappers
                .iter()
                .map(|w| Gemma4MtpTargetAdapter::new(w, None))
                .collect();
            for (row, (single, prompt)) in singles.iter().zip(&prompts).enumerate() {
                let (bonus, _, _) = single.prefill_and_seed(prompt, &sampler, &[], &logprobs);
                assert_eq!(bonuses[row], bonus, "[{layer_type}] row {row}: first bonus");
            }
            let mut lockstep = Lockstep {
                batched,
                singles,
                bonuses,
                label: format!(
                    "{layer_type}, lengths {:?}",
                    prompts.iter().map(Vec::len).collect::<Vec<_>>()
                ),
            };
            for accepted in accepts {
                lockstep.round([1, 5], accepted);
            }
        }
    }
}
