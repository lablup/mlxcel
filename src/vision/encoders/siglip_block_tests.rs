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

//! Regression guard for the SigLIP/CLIP encoder block now that it is shared
//! with the SigLIP text tower (`crate::models::siglip_text`).
//!
//! The block gained an optional attention mask so the text tower could reuse
//! it. Vision callers pass `None` and must keep producing exactly what they
//! produced before that parameter existed, so the golden below was captured
//! from the pre-change `EncoderLayer::forward(x)` on this fixed seed and is
//! asserted against the post-change `EncoderLayer::forward(x, None)`.

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{EncoderLayer, VisionMlpActivation};
use crate::vision::config::{VisionConfig, VisionHiddenActivation};

/// Deterministic linear congruential generator, so the golden below does not
/// depend on a random-number crate or its version.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }

    fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_f32() * scale).collect()
    }

    fn insert(
        &mut self,
        weights: &mut WeightMap,
        name: &str,
        shape: &[i32],
        scale: f32,
        offset: f32,
    ) {
        let count: usize = shape.iter().map(|&d| d as usize).product();
        let values: Vec<f32> = self
            .values(count, scale)
            .into_iter()
            .map(|v| v + offset)
            .collect();
        weights.insert(
            name.to_string(),
            mlxcel_core::from_slice_f32(&values, shape),
        );
    }
}

/// Weights for one encoder block at `hidden = 8`, `intermediate = 16`.
fn block_weights(rng: &mut Lcg, prefix: &str) -> WeightMap {
    let mut weights = WeightMap::new();
    for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        let base = format!("{prefix}.self_attn.{projection}");
        rng.insert(&mut weights, &format!("{base}.weight"), &[8, 8], 0.3, 0.0);
        rng.insert(&mut weights, &format!("{base}.bias"), &[8], 0.1, 0.0);
    }
    for norm in ["layer_norm1", "layer_norm2"] {
        let base = format!("{prefix}.{norm}");
        rng.insert(&mut weights, &format!("{base}.weight"), &[8], 0.1, 1.0);
        rng.insert(&mut weights, &format!("{base}.bias"), &[8], 0.1, 0.0);
    }
    rng.insert(
        &mut weights,
        &format!("{prefix}.mlp.fc1.weight"),
        &[16, 8],
        0.3,
        0.0,
    );
    rng.insert(
        &mut weights,
        &format!("{prefix}.mlp.fc1.bias"),
        &[16],
        0.1,
        0.0,
    );
    rng.insert(
        &mut weights,
        &format!("{prefix}.mlp.fc2.weight"),
        &[8, 16],
        0.3,
        0.0,
    );
    rng.insert(
        &mut weights,
        &format!("{prefix}.mlp.fc2.bias"),
        &[8],
        0.1,
        0.0,
    );
    weights
}

fn block_config() -> VisionConfig {
    VisionConfig {
        model_type: "siglip_vision_model".to_string(),
        num_hidden_layers: 1,
        hidden_size: 8,
        intermediate_size: 16,
        num_attention_heads: 2,
        patch_size: 16,
        image_size: 224,
        num_channels: 3,
        layer_norm_eps: 1e-6,
        hidden_act: VisionHiddenActivation::GeluPytorchTanh,
    }
}

/// The block and its input, both from the fixed seed the golden was captured
/// with. The input is drawn after the weights, so the draw order is part of
/// the fixture.
fn fixture() -> (EncoderLayer, UniquePtr<MlxArray>) {
    let mut rng = Lcg(0x5161_1DEF_0000_0001);
    let weights = block_weights(&mut rng, "block");
    let layer = EncoderLayer::from_weights(
        &weights,
        "block",
        &block_config(),
        64,
        4,
        VisionMlpActivation::PytorchTanh,
    )
    .unwrap();
    let x = mlxcel_core::from_slice_f32(&rng.values(32, 1.0), &[1, 4, 8]);
    (layer, x)
}

fn read(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(array);
    array_to_vec_f32(array)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Captured from `EncoderLayer::forward(x)` before the mask parameter was
/// added. Do not regenerate it to make a change pass: it exists to prove the
/// vision path is untouched.
const UNMASKED_GOLDEN: [f32; 32] = [
    0.755_600_8,
    -0.597_344_04,
    -0.328_453_24,
    -0.699_166_6,
    -1.035_612_3,
    0.818_498_1,
    0.919_871_2,
    0.990_668_83,
    0.629_887_9,
    0.344_315_92,
    -0.306_985_94,
    -0.355_890_04,
    0.010_203_063,
    -0.489_434_36,
    -0.277_053_12,
    0.709_256_8,
    -0.021_081,
    -1.029_360_9,
    -0.737_356,
    -0.224_977_97,
    0.526_652_3,
    0.689_434_65,
    0.069_147_02,
    1.257_028_3,
    0.719_936_2,
    0.482_109_55,
    0.964_234_5,
    1.272_394,
    -0.771_051_65,
    0.718_882_6,
    0.036_761_813,
    0.094_809_234,
];

#[test]
fn encoder_block_shared_with_vision_is_unchanged() {
    let (layer, x) = fixture();
    let out = read(&layer.forward(&x, None));
    assert_eq!(out.len(), UNMASKED_GOLDEN.len());
    let drift = max_abs_diff(&out, &UNMASKED_GOLDEN);
    assert!(
        drift <= 1e-6,
        "the unmasked encoder block drifted from the pre-refactor output by {drift}"
    );
}

#[test]
fn an_all_attend_mask_is_a_no_op_and_a_blocking_mask_is_not() {
    let (layer, x) = fixture();
    let unmasked = read(&layer.forward(&x, None));

    // Additive masks are 0.0 = attend, -inf = blocked. An all-zero mask must
    // reproduce the maskless path exactly, which is what lets vision callers
    // keep passing `None` while the text tower shares the block.
    let attend_all = mlxcel_core::from_slice_f32(&[0.0; 16], &[1, 1, 4, 4]);
    let masked = read(&layer.forward(&x, Some(&attend_all)));
    assert!(max_abs_diff(&unmasked, &masked) <= 1e-6);

    // A mask that actually blocks a key must reach the attention call, so the
    // parameter cannot be silently dropped.
    let mut blocking = [0.0f32; 16];
    for query in 0..4 {
        blocking[query * 4 + 3] = f32::NEG_INFINITY;
    }
    let blocked = read(&layer.forward(
        &x,
        Some(&mlxcel_core::from_slice_f32(&blocking, &[1, 1, 4, 4])),
    ));
    assert!(
        max_abs_diff(&unmasked, &blocked) > 1e-4,
        "blocking the last key column must change the block output"
    );
}
