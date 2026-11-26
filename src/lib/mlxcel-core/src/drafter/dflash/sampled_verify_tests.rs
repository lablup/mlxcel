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

//! Distribution preservation of the sampled verify rows (issue #3436): a
//! Markov target and a deterministic drafter on the real round loop, whose
//! sampled output must have the joint distribution of plain sampling, and
//! whose presence penalty must see the accepted drafts.

use super::super::round_loop::{DFlashGenerator, SpeculativeTarget};
use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::sync::atomic::AtomicBool;

const VOCAB: usize = 4;

/// `p(next | prev)`, row `prev`.
const TABLE: [[f32; VOCAB]; VOCAB] = [
    [0.6, 0.2, 0.1, 0.1],
    [0.1, 0.5, 0.3, 0.1],
    [0.25, 0.25, 0.25, 0.25],
    [0.05, 0.15, 0.3, 0.5],
];

/// A target whose verify row `s` is `log TABLE[input[s]]`.
struct MarkovTarget;

struct MarkovOut {
    logits: UniquePtr<MlxArray>,
    hidden: UniquePtr<MlxArray>,
}

impl SpeculativeTarget for MarkovTarget {
    type Cache = ();
    type VerifyOut = MarkovOut;

    fn capture_layer_ids(&self) -> &[usize] {
        &[]
    }

    fn verify_forward(&self, input: &MlxArray, _caches: &mut [()]) -> MarkovOut {
        let k = ffi::array_shape(input)[1];
        ffi::eval(input);
        let ids: Vec<i32> =
            crate::utils::array_to_vec_f32(&ffi::astype(input, crate::dtype::FLOAT32))
                .into_iter()
                .map(|v| v as i32)
                .collect();
        let logits: Vec<f32> = ids
            .iter()
            .flat_map(|&prev| TABLE[prev as usize].iter().map(|p| p.ln()))
            .collect();
        MarkovOut {
            logits: ffi::from_slice_f32(&logits, &[1, k, VOCAB as i32]),
            hidden: ffi::zeros(&[1, k, 1], crate::dtype::FLOAT32),
        }
    }

    fn verify_forward_with_capture_layers(
        &self,
        input: &MlxArray,
        caches: &mut [()],
        _layers: &[usize],
    ) -> MarkovOut {
        self.verify_forward(input, caches)
    }

    fn rollback_partial(&self, _c: &mut [()], _o: &MarkovOut, _accepted: i32, _block: i32) {}

    fn concat_hidden_for_drafter(&self, out: &MarkovOut) -> UniquePtr<MlxArray> {
        ffi::copy(&out.hidden)
    }

    fn verify_logits<'a>(&self, out: &'a MarkovOut) -> &'a MlxArray {
        &out.logits
    }
}

/// Proposes `prev + 1, prev + 2, ...` (mod the vocabulary) from the bonus.
struct StepDrafter;

impl Drafter for StepDrafter {
    fn bind(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }
    fn reset(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }
    fn supports_sampled_verify(&self) -> bool {
        true
    }
    fn prefer_requested_block_size(&self) -> bool {
        true
    }
    fn draft_block(
        &mut self,
        last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        Ok((1..block_size as i32)
            .map(|i| (last_bonus + i) % VOCAB as i32)
            .collect())
    }
    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }
    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
}

struct NoLm;

impl LanguageModel for NoLm {
    fn forward(
        &self,
        _: &MlxArray,
        _: &mut [KVCache],
        _: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        ffi::zeros(&[1, 1, 1], crate::dtype::FLOAT32)
    }
    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }
    fn num_layers(&self) -> usize {
        0
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }
}

/// The three tokens after a fixed first bonus `0`, at block width `width`.
fn run_once(sampler: &SamplingConfig, width: u32) -> Vec<i32> {
    let mut generator = DFlashGenerator::new(Box::new(StepDrafter), sampler.clone(), width, 0);
    generator.set_token_history(Vec::new());
    let out = generator
        .run(
            &MarkovTarget,
            &NoLm,
            &mut [],
            0,
            ffi::zeros(&[1, 1, 1], crate::dtype::FLOAT32),
            &[],
            4,
            &AtomicBool::new(false),
            &crate::sampling::LogprobsConfig::default(),
        )
        .expect("round loop runs");
    out.tokens
}

#[test]
fn sampled_verify_keeps_the_plain_sampling_joint_distribution() {
    ffi::random_seed(3436);
    let sampler = SamplingConfig::default(); // temperature 1, no truncation
    const N: usize = 3000;
    for width in [2u32, 4] {
        let mut pairs = [[0usize; VOCAB]; VOCAB];
        let mut accepted_any = false;
        for _ in 0..N {
            let t = run_once(&sampler, width);
            assert_eq!(t.len(), 3, "width {width}: three tokens after the bonus");
            pairs[t[0] as usize][t[1] as usize] += 1;
            accepted_any |= t[0] == 1 || t[1] == (t[0] + 1) % VOCAB as i32;
        }
        assert!(
            accepted_any,
            "width {width}: the drafts must be accepted sometimes"
        );
        // Plain sampling: t0 ~ TABLE[0], t1 ~ TABLE[t0].
        for (a, row) in pairs.iter().enumerate() {
            for (b, &count) in row.iter().enumerate() {
                let p = (TABLE[0][a] * TABLE[a][b]) as f64;
                let freq = count as f64 / N as f64;
                let sd = (p * (1.0 - p) / N as f64).sqrt();
                assert!(
                    (freq - p).abs() <= 5.0 * sd + 1e-3,
                    "width {width}: P(t0 = {a}, t1 = {b}) = {p:.4}, sampled {freq:.4}"
                );
            }
        }
    }
}

#[test]
fn a_presence_penalty_sees_the_accepted_drafts() {
    ffi::random_seed(36);
    // A large presence penalty forbids every token already in the history,
    // the first bonus included, so plain sampling emits four distinct
    // tokens. A verify row that missed an accepted draft could repeat it.
    let sampler = SamplingConfig {
        presence_penalty: 50.0,
        ..SamplingConfig::default()
    };
    for _ in 0..200 {
        let mut seen = vec![0];
        seen.extend(run_once(&sampler, 4));
        let mut distinct = seen.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), seen.len(), "repeated token in {seen:?}");
    }
}

#[test]
fn feedback_samplers_never_take_the_sampled_verify_rows() {
    // Each verify row samples statelessly, so a sampler whose state carries
    // from one token to the next (Mirostat's mu, adaptive-p's EMA) would
    // restart on every row; the round loop and the burst gate refuse both.
    let mirostat = SamplingConfig {
        mirostat: 2,
        ..SamplingConfig::default()
    };
    let adaptive = SamplingConfig {
        adaptive_target: 0.5,
        ..SamplingConfig::default()
    };
    for sampler in [&mirostat, &adaptive] {
        assert!(sampler.needs_sampler_feedback_state());
        assert!(!super::sampled_verify_serves(sampler));
        assert!(!super::uses_sampled_verify(&StepDrafter, sampler));
    }
    assert!(super::uses_sampled_verify(
        &StepDrafter,
        &SamplingConfig::default()
    ));
}

#[test]
fn sampled_rows_report_the_penalty_adjusted_logits() {
    // Plain decode computes log-probabilities from the sampler-adjusted
    // logits; the sampled verify rows hand those back for the same report.
    let sampler = SamplingConfig {
        presence_penalty: 50.0,
        ..SamplingConfig::default()
    };
    let logits = ffi::zeros(&[1, 2, VOCAB as i32], crate::dtype::FLOAT32);
    let (tokens, adjusted) = super::sample_verify_rows(&logits, &[], &sampler, &[0]);
    assert_eq!(tokens.len(), 1, "no draft: only the bonus row is sampled");
    assert_ne!(tokens[0], 0, "the penalized token is never drawn");
    assert_eq!(ffi::array_shape(&adjusted), vec![1, 1, VOCAB as i32]);
    let row = crate::utils::array_to_vec_f32(&ffi::astype(&adjusted, crate::dtype::FLOAT32));
    assert!(
        row[0] <= -49.0,
        "token 0 carries the presence penalty: {row:?}"
    );
    assert!(row[1..].iter().all(|&v| v.abs() < 1e-6), "{row:?}");
}
