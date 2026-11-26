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

//! Reasoning-effort steering on the tiny random Raon-1 mini (#3435): the add
//! is applied (and fails these tests if skipped), `max` is the unsteered
//! model bit for bit, and each batch row runs at its own sequence's alpha on
//! the native batched decode, the per-row fallback and prefill.

use super::super::reasoning_effort_steering::ReasoningEffortSteering;
use super::tests::{ids, raw_weights, tiny_config, tiny_model, values};
use super::{Raon1MiniConfig, Raon1MiniModel};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::{MlxArray, dtype};

/// The release block, steering the tiny model's layer 2 (a MoE layer whose
/// fixture carries a vector).
fn steering(layers: &[usize]) -> ReasoningEffortSteering {
    ReasoningEffortSteering::from_config_value(&serde_json::json!({
        "reasoning_effort_steering": {
            "alpha": {"none": 0, "low": 0.3, "medium": 0.1, "high": 0.05, "xhigh": 0.02, "max": 0},
            "default": "high",
            "layers": layers,
        }
    }))
    .unwrap()
    .unwrap()
}

fn steered_config(layers: &[usize]) -> Raon1MiniConfig {
    let mut config = tiny_config();
    config.reasoning_effort_steering = Some(steering(layers));
    config
}

fn load(config: &Raon1MiniConfig) -> Result<Raon1MiniModel, String> {
    let weights = Raon1MiniModel::sanitize_weights(raw_weights(), config)?;
    Raon1MiniModel::from_weights(&weights, config)
}

/// The tiny random model with layer 2 steered (also the scheduler tests' model).
pub(crate) fn steered_model() -> Raon1MiniModel {
    load(&steered_config(&[2])).expect("steered tiny model loads")
}

/// Prefill logits of `tokens` at the model-default `level`.
fn prefill_at(model: &Raon1MiniModel, level: Option<&str>, tokens: &[i32]) -> Vec<f32> {
    model.set_reasoning_effort(None, level).unwrap();
    let logits = model.forward_with_caches(&ids(&[tokens]), &mut model.make_layer_caches(), None);
    values(&logits)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn vectors_load_in_float32_on_listed_layers_only() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = steered_model();
    let steering = model.reasoning_effort_steering();
    assert_eq!(steering.len(), 1);
    assert_eq!(steering[0].0, 2);
    assert_eq!(mlxcel_core::array_dtype(steering[0].1), dtype::FLOAT32);
    // Without the block the fixture's layer-2 vector is dropped.
    assert!(tiny_model().reasoning_effort_steering().is_empty());
}

#[test]
fn a_listed_layer_without_a_vector_or_a_dense_listed_layer_fails_to_load() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let err = load(&steered_config(&[2, 3]))
        .err()
        .expect("layer 3 has no vector");
    assert!(
        err.contains("Missing weight: model.layers.3.moe.reasoning_effort_steering"),
        "{err}"
    );
    let err = steered_config(&[0]).validate().unwrap_err();
    assert!(err.contains("layer 0, which is not a MoE layer"), "{err}");
}

#[test]
fn the_default_effort_steers_and_max_is_the_unsteered_model_bit_for_bit() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let tokens = [1, 2, 3, 4, 5];
    let unsteered = prefill_at(&tiny_model(), None, &tokens);
    let model = steered_model();
    let high = prefill_at(&model, None, &tokens);
    let low = prefill_at(&model, Some("low"), &tokens);
    let max = prefill_at(&model, Some("max"), &tokens);
    let none = prefill_at(&model, Some("none"), &tokens);
    let medium = prefill_at(&model, Some("medium"), &tokens);

    assert_eq!(max, unsteered, "max (alpha 0) must skip the add entirely");
    assert!(
        max_abs_diff(&high, &unsteered) > 1e-4,
        "the default effort changes the logits"
    );
    assert!(
        max_abs_diff(&low, &high) > 1e-4,
        "low and high steer differently"
    );
    // `none` resolves to NO_THINKING_ALPHA (0.1), the same alpha as medium.
    assert_eq!(none, medium);
    assert_eq!(model.reasoning_effort_alpha(None), 0.1);
}

/// Prefill `prompt` for `id` at `level` on `model`.
fn open(model: &Raon1MiniModel, id: SequenceId, level: &str, prompt: &[i32]) {
    model.prepare_sequence_state(id);
    model
        .set_sequence_reasoning_effort(Some(id), Some(level))
        .unwrap();
    let out =
        model.forward_with_sequence_id(&ids(&[prompt]), Some(id), &mut model.make_caches(), None);
    mlxcel_core::eval(&out);
}

fn row(logits: &MlxArray, r: i32) -> Vec<f32> {
    values(&mlxcel_core::utils::slice_axis(logits, 0, r, r + 1))
}

/// Two rows at different efforts in one batch equal each row run alone at its
/// effort, on the native batched decode (one token, uniform states) and on
/// the per-row fallback (a two-token step), within the tolerance of
/// `native_batched_decode_matches_per_sequence_decode`.
#[test]
fn batched_rows_at_different_efforts_match_each_row_alone() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let tol = 1e-5;
    let a = SequenceId::from_raw(91_000);
    let b = SequenceId::from_raw(91_001);
    let batched = steered_model();
    let (solo_a, solo_b) = (steered_model(), steered_model());
    open(&batched, a, "low", &[1, 2, 3]);
    open(&batched, b, "max", &[4, 5, 6, 7]);
    open(&solo_a, a, "low", &[1, 2, 3]);
    open(&solo_b, b, "max", &[4, 5, 6, 7]);
    // The default must not leak into either row.
    assert_eq!(batched.reasoning_effort_alpha(Some(a)), 0.3);
    assert_eq!(batched.reasoning_effort_alpha(Some(b)), 0.0);

    for (step, (ta, tb)) in [(&[8][..], &[9][..]), (&[10, 11][..], &[12, 13][..])]
        .into_iter()
        .enumerate()
    {
        let mut ca = batched.make_caches();
        let mut cb = batched.make_caches();
        let mut caches: Vec<&mut [mlxcel_core::layers::KVCache]> = vec![&mut ca, &mut cb];
        let got =
            batched.forward_batched_with_ids(&ids(&[ta, tb]), Some(&[a, b]), &mut caches, None);
        let want_a =
            solo_a.forward_with_sequence_id(&ids(&[ta]), Some(a), &mut solo_a.make_caches(), None);
        let want_b =
            solo_b.forward_with_sequence_id(&ids(&[tb]), Some(b), &mut solo_b.make_caches(), None);
        let (got_a, got_b) = (row(&got, 0), row(&got, 1));
        assert!(
            max_abs_diff(&got_a, &values(&want_a)) <= tol,
            "step {step} row a (low)"
        );
        assert!(
            max_abs_diff(&got_b, &values(&want_b)) <= tol,
            "step {step} row b (max)"
        );
    }

    // Swapping the efforts changes the rows, so the match above is not the
    // trivial one of an unsteered batch.
    let swapped = steered_model();
    open(&swapped, a, "max", &[1, 2, 3]);
    let solo_low = steered_model();
    open(&solo_low, a, "low", &[1, 2, 3]);
    let x =
        swapped.forward_with_sequence_id(&ids(&[&[8]]), Some(a), &mut swapped.make_caches(), None);
    let y = solo_low.forward_with_sequence_id(
        &ids(&[&[8]]),
        Some(a),
        &mut solo_low.make_caches(),
        None,
    );
    assert!(max_abs_diff(&values(&x), &values(&y)) > 1e-4);
}

#[test]
fn a_released_or_reprepared_sequence_returns_to_the_default() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = steered_model();
    let id = SequenceId::from_raw(91_100);
    model.prepare_sequence_state(id);
    model
        .set_sequence_reasoning_effort(Some(id), Some("low"))
        .unwrap();
    assert_eq!(model.sequence_reasoning_effort(id).as_deref(), Some("low"));
    model.release_sequence_state_by_id(id);
    assert_eq!(model.sequence_reasoning_effort(id), None);
    assert_eq!(model.reasoning_effort_alpha(Some(id)), 0.05);
    model
        .set_sequence_reasoning_effort(Some(id), Some("xhigh"))
        .unwrap();
    model.prepare_sequence_state(id);
    assert_eq!(model.reasoning_effort_alpha(Some(id)), 0.05);
    // An unknown level is refused and leaves the setting alone.
    let err = model
        .set_sequence_reasoning_effort(Some(id), Some("minimal"))
        .unwrap_err();
    assert!(err.contains("none, low, medium, high, xhigh, max"), "{err}");
    // A model without the block ignores every level.
    let plain = tiny_model();
    plain
        .set_sequence_reasoning_effort(None, Some("low"))
        .unwrap();
    assert_eq!(plain.reasoning_effort_alpha(None), 0.0);
}
