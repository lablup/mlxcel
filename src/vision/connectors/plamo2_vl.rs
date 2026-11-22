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

//! PLaMo 2.1 VL `image_proj` adapter (`MLPImageProjector`, issue #1380).
//!
//! For tower features `x: [N, 1152]`:
//!
//! ```text
//! x = rms_norm(x, norm0.weight + 1.0, eps)   # PLaMo offset norm
//! x = x + bias0
//! x = linear1(x) + bias1                     # linear1 has no own bias
//! x = gelu(x)                                # exact erf, torch nn.GELU()
//! x = linear2(x) + bias2                     # -> text hidden size
//! ```
//!
//! The HF checkpoint stores the biases as `image_proj.biasN._bias`; the
//! mlx-vlm conversion renames them `image_proj.biasN.bias`. The loader
//! normalizes both to `.bias` before this module reads them.
//!
//! Used by: `vision::plamo2_vl::Plamo2VlModel`, `loading::load_plamo2_vl`.

use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// The additive offset PLaMo folds into every RMSNorm weight.
const NORM0_OFFSET: f32 = 1.0;

pub struct Plamo2VlImageProjector {
    norm0: RMSNorm,
    bias0: UniquePtr<MlxArray>,
    linear1: UnifiedLinear,
    bias1: UniquePtr<MlxArray>,
    linear2: UnifiedLinear,
    bias2: UniquePtr<MlxArray>,
}

fn required(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {key}"))
}

impl Plamo2VlImageProjector {
    /// Load `{prefix}.norm0.weight`, `{prefix}.bias{0,1,2}.bias` and
    /// `{prefix}.linear{1,2}` (plain or quantized). `eps` is the text config's
    /// `rms_norm_eps`.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let norm_weight = required(weights, &format!("{prefix}.norm0.weight"))?;
        let dtype = mlxcel_core::array_dtype(&norm_weight);
        let offset = mlxcel_core::full_f32(&[1], NORM0_OFFSET, dtype);
        let norm0 = RMSNorm::new(mlxcel_core::add(&norm_weight, &offset), eps);
        Ok(Self {
            norm0,
            bias0: required(weights, &format!("{prefix}.bias0.bias"))?,
            linear1: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.linear1"),
                group_size,
                bits,
            )?,
            bias1: required(weights, &format!("{prefix}.bias1.bias"))?,
            linear2: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.linear2"),
                group_size,
                bits,
            )?,
            bias2: required(weights, &format!("{prefix}.bias2.bias"))?,
        })
    }

    /// `[..., image_feature_size]` -> `[..., hidden_size]`.
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.norm0.forward(x);
        let x = mlxcel_core::add(&x, &self.bias0);
        let x = mlxcel_core::add(&self.linear1.forward(&x), &self.bias1);
        let x = mlxcel_core::gelu(&x);
        mlxcel_core::add(&self.linear2.forward(&x), &self.bias2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `norm0.weight = 0` makes the offset norm a unit-weight RMSNorm; the rest
    /// is checked against a hand computation on a 2x4 input.
    #[test]
    fn projector_forward_shape_and_offset_norm() {
        let _guard = crate::models::embedding_test_support::mlx_test_guard();
        let f32_ = mlxcel_core::dtype::FLOAT32;
        let mut weights = WeightMap::new();
        weights.insert("p.norm0.weight".into(), mlxcel_core::zeros(&[4], f32_));
        weights.insert(
            "p.bias0.bias".into(),
            mlxcel_core::from_slice_f32(&[0.0, 0.0, 0.0, 1.0], &[4]),
        );
        // linear1: [3, 4], picks components 0, 1 and 3.
        let w1 = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        weights.insert(
            "p.linear1.weight".into(),
            mlxcel_core::from_slice_f32(&w1, &[3, 4]),
        );
        weights.insert(
            "p.bias1.bias".into(),
            mlxcel_core::from_slice_f32(&[0.5, 0.0, 0.0], &[3]),
        );
        // linear2: [2, 3], sums of pairs.
        let w2 = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        weights.insert(
            "p.linear2.weight".into(),
            mlxcel_core::from_slice_f32(&w2, &[2, 3]),
        );
        weights.insert(
            "p.bias2.bias".into(),
            mlxcel_core::from_slice_f32(&[0.0, -1.0], &[2]),
        );
        let projector = Plamo2VlImageProjector::from_weights(&weights, "p", 1e-6, 64, 4).unwrap();

        let input = [1.0f32, -1.0, 1.0, -1.0, 2.0, 0.0, 0.0, 0.0];
        let x = mlxcel_core::from_slice_f32(&input, &[2, 4]);
        let out = projector.forward(&x);
        assert_eq!(mlxcel_core::array_shape(&out), vec![2, 2]);

        let gelu = |v: f64| 0.5 * v * (1.0 + libm_erf(v / std::f64::consts::SQRT_2));
        let mut expected = Vec::new();
        for row in input.chunks(4) {
            let rms = (row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / 4.0 + 1e-6).sqrt();
            let n: Vec<f64> = row.iter().map(|v| f64::from(*v) / rms).collect();
            let b0 = [n[0], n[1], n[2], n[3] + 1.0];
            let h = [gelu(b0[0] + 0.5), gelu(b0[1]), gelu(b0[3])];
            expected.push(h[0] + h[1]);
            expected.push(h[1] + h[2] - 1.0);
        }
        let flat = mlxcel_core::reshape(&out, &[4]);
        for (i, want) in expected.iter().enumerate() {
            let got =
                mlxcel_core::item_f32(&mlxcel_core::slice(&flat, &[i as i32], &[i as i32 + 1]));
            assert!(
                (f64::from(got) - want).abs() < 1e-4,
                "element {i}: {got} vs {want}"
            );
        }
    }

    /// Abramowitz-Stegun 7.1.26 erf (|error| < 1.5e-7), enough for a 1e-4 check.
    fn libm_erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        sign * y
    }
}
