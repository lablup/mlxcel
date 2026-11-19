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

//! Test-only fixtures for the SSM `A = -exp(A_log)` overflow (#3046, #3057).
//!
//! f16 holds `exp(x)` only up to `x = 11.09`, and trained checkpoints carry
//! `A_log` up to 19.5. Computing `exp` in f16 gives `A = -inf`, and a step
//! whose `softplus` is zero then computes `0 * -inf = NaN`. A `dt_bias` of
//! [`ZERO_STEP_DT_BIAS`] drives every `softplus` step to exactly zero, so a
//! model that still runs `exp` in f16 returns NaN logits from a prefill.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// A `dt_bias` (or `dt_proj` bias) whose float32 `softplus` is exactly zero.
pub(crate) const ZERO_STEP_DT_BIAS: f32 = -200.0;

/// An f16 `A_log` of `shape` whose values ramp from 0.5 to 19.5 (the largest
/// `state-spaces/mamba-130m` carries); every value past 11.09 overflows an
/// f16 `exp`, and the smaller ones round differently in an f16 `exp`.
pub(crate) fn overflowing_f16_a_log(shape: &[i32]) -> UniquePtr<MlxArray> {
    let len: usize = shape.iter().map(|d| *d as usize).product();
    let last = len.saturating_sub(1).max(1) as f32;
    let data: Vec<f32> = (0..len).map(|i| 0.5 + 19.0 * i as f32 / last).collect();
    mlxcel_core::astype(&mlxcel_core::from_slice_f32(&data, shape), dtype::FLOAT16)
}

/// `len` float32 values of [`ZERO_STEP_DT_BIAS`].
pub(crate) fn zero_step_dt_bias(len: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[len], ZERO_STEP_DT_BIAS, dtype::FLOAT32)
}

/// The logits of a three-token prefill.
pub(crate) fn prefill_logits<M: LanguageModel>(model: &M) -> Vec<f32> {
    let seq = SequenceId::from_raw(3_057_000);
    model.prepare_sequence_state(seq);
    let mut caches = model.make_caches();
    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let logits = model.forward_with_sequence_id(&prompt, Some(seq), &mut caches, None);
    mlxcel_core::utils::array_to_vec_f32(&logits)
}

/// Prefill three tokens and assert every logit is finite.
pub(crate) fn assert_prefill_logits_finite<M: LanguageModel>(model: &M, what: &str) {
    let values = prefill_logits(model);
    let bad = values.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        bad,
        0,
        "{what}: {bad} of {} prefill logits are not finite",
        values.len()
    );
}

/// Build the model `build` makes from [`overflowing_f16_a_log`] of `shape`
/// and from a float32 copy of the same values, and assert the f16 load's
/// prefill logits are finite and bit-identical to the float32 load's. Both
/// hold only when `A = -exp(A_log)` is computed from a float32 copy: an f16
/// `exp` rounds every `A` to f16 and turns the large ones into `-inf`.
pub(crate) fn assert_f16_a_log_prefill_matches_its_f32_copy<M: LanguageModel>(
    build: impl Fn(&MlxArray) -> M,
    shape: &[i32],
    what: &str,
) {
    let a_log = overflowing_f16_a_log(shape);
    let from_f16 = build(&a_log);
    assert_prefill_logits_finite(&from_f16, what);
    let from_f32 = build(&mlxcel_core::astype(&a_log, dtype::FLOAT32));
    let (got, want) = (prefill_logits(&from_f16), prefill_logits(&from_f32));
    assert!(
        got.iter()
            .zip(&want)
            .all(|(g, w)| g.to_bits() == w.to_bits()),
        "{what}: an f16 A_log must give the logits of its float32 copy"
    );
}
