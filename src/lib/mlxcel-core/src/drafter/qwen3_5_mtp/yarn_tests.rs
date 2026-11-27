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

//! YaRN rotation in the Qwen 3.5 MTP drafter layer (issue #3479): the
//! default stays on plain `rope_theta`, and an installed table reaches the
//! attention forward through `fast_rope_with_freqs` plus rotary-lane scaling.

use super::layer::Qwen35MtpDecoderLayer;
use super::model::Qwen35MtpDraftModel;
use super::tests::{tiny_config, tiny_weights};
use crate::cache::KVCache;
use crate::ffi::{self, MlxArray};
use crate::rope_lanes::scale_rotary_lanes;
use crate::utils::array_to_vec_f32;
use cxx::UniquePtr;

fn tiny_layer() -> Qwen35MtpDecoderLayer {
    let config = tiny_config();
    Qwen35MtpDecoderLayer::from_weights(&tiny_weights(), "layers.0", config.text_config())
        .expect("tiny layer builds")
}

/// A `[1, 3, 8]` input with distinct values per lane, so the rotation and
/// the lane scaling both move the output.
fn layer_input() -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..24).map(|i| 0.1 + 0.05 * (i % 11) as f32).collect();
    ffi::from_slice_f32(&data, &[1, 3, 8])
}

fn forward_at(layer: &Qwen35MtpDecoderLayer, offset: i32) -> Vec<f32> {
    let mut cache = KVCache::new();
    let out = layer.forward(&layer_input(), true, &mut cache, offset);
    array_to_vec_f32(&out)
}

/// The divisor table plain RoPE uses for `theta` over `dims` rotary lanes:
/// `theta^(2i / dims)`, the form `fast_rope_with_freqs` takes.
fn plain_freqs(theta: f32, dims: i32) -> UniquePtr<MlxArray> {
    let table: Vec<f32> = (0..dims / 2)
        .map(|i| theta.powf(2.0 * i as f32 / dims as f32))
        .collect();
    ffi::from_slice_f32(&table, &[dims / 2])
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn drafter_without_yarn_keeps_plain_rope_theta() {
    let model = Qwen35MtpDraftModel::from_weights(&tiny_weights(), tiny_config())
        .expect("tiny drafter builds");
    assert!(model.yarn_rope().is_none(), "no table unless one is set");
    assert_eq!(model.config().text_config().rope_theta(), 10000.0);
    assert_eq!(model.config().text_config().max_position_embeddings, None);
}

#[test]
fn set_yarn_rope_installs_the_table_on_every_layer() {
    let mut model = Qwen35MtpDraftModel::from_weights(&tiny_weights(), tiny_config())
        .expect("tiny drafter builds");
    let freqs = ffi::from_slice_f32(&[3.0], &[1]);
    model.set_yarn_rope(&freqs, 1.25);
    let (table, factor) = model.yarn_rope().expect("table set");
    assert_eq!(array_to_vec_f32(table), vec![3.0]);
    assert_eq!(factor, 1.25);
}

#[test]
fn yarn_table_reaches_the_attention_forward() {
    let plain = forward_at(&tiny_layer(), 5);

    // The plain `rope_theta` table through the YaRN path with a unit factor
    // is the same rotation as `fast_rope`.
    let mut same_table = tiny_layer();
    same_table.set_yarn_rope(&plain_freqs(10000.0, 2), 1.0);
    let via_freqs = forward_at(&same_table, 5);
    assert!(
        max_abs_diff(&plain, &via_freqs) < 1e-5,
        "unit-factor table must match fast_rope: {plain:?} vs {via_freqs:?}"
    );

    // A stretched table changes the output: the drafter really rotates with
    // the installed frequencies, not with `rope_theta`.
    let mut stretched = tiny_layer();
    stretched.set_yarn_rope(&ffi::from_slice_f32(&[4.0], &[1]), 1.0);
    assert!(max_abs_diff(&plain, &forward_at(&stretched, 5)) > 1e-4);

    // The attention factor is applied as well.
    let mut scaled = tiny_layer();
    scaled.set_yarn_rope(&plain_freqs(10000.0, 2), 1.5);
    assert!(max_abs_diff(&plain, &forward_at(&scaled, 5)) > 1e-4);
}

#[test]
fn scale_rotary_lanes_scales_only_the_rotated_prefix() {
    let x = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
    assert_eq!(
        array_to_vec_f32(&scale_rotary_lanes(&x, 2, 2.0)),
        vec![2.0, 4.0, 3.0, 4.0]
    );
    assert_eq!(
        array_to_vec_f32(&scale_rotary_lanes(&x, 4, 0.5)),
        vec![0.5, 1.0, 1.5, 2.0]
    );
    assert_eq!(
        array_to_vec_f32(&scale_rotary_lanes(&x, 2, 1.0)),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}
