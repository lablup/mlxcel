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

//! LFM2-VL connector: pixel unshuffle (space-to-depth) + multimodal projector.
//!
//! Port of the LFM2-VL `multi_modal_projector` path. The vision tower output
//! `(1, h*w, hidden)` is reshaped to the patch grid `(1, h, w, hidden)`,
//! space-to-depth downsampled by `downsample_factor = f` (each `f x f` block
//! packs into the channel dim, row-major with the original channel varying
//! fastest), then projected `LayerNorm -> Linear -> exact-GELU -> Linear` into
//! the text hidden size and flattened to `(T, text_hidden)` with
//! `T = ceil(h/f) * ceil(w/f)`.
//!
//! Used by: `vision::lfm2_vl::Lfm2VlModel`.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Space-to-depth by factor `f`. `x`: `[1, h, w, C]` -> `[1, ceil(h/f), ceil(w/f),
/// C*f*f]` with `out[0, ar, br, (dr*f + dc)*C + c] = in[0, ar*f + dr, br*f + dc, c]`
/// (zero-padding the high edge of each spatial axis up to a multiple of `f`).
pub(crate) fn pixel_unshuffle(x: &MlxArray, f: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    let (h, w, c) = (s[1], s[2], s[3]);
    let hp = ((h + f - 1) / f) * f;
    let wp = ((w + f - 1) / f) * f;

    let mut x = mlxcel_core::copy(x);
    let dtype = mlxcel_core::array_dtype(&x);
    if hp > h {
        let pad = mlxcel_core::zeros(&[1, hp - h, w, c], dtype);
        x = mlxcel_core::concatenate(&x, &pad, 1);
    }
    if wp > w {
        let pad = mlxcel_core::zeros(&[1, hp, wp - w, c], dtype);
        x = mlxcel_core::concatenate(&x, &pad, 2);
    }

    // [1, hp, wp, C] -> [1, hp/f, f, wp/f, f, C]
    let x = mlxcel_core::reshape(&x, &[1, hp / f, f, wp / f, f, c]);
    // -> [1, hp/f, wp/f, f(dr), f(dc), C]
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
    // -> [1, hp/f, wp/f, C*f*f] with (dr, dc, c) flattened, c fastest.
    mlxcel_core::reshape(&x, &[1, hp / f, wp / f, c * f * f])
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, eps))
}

/// LFM2-VL multimodal projector (pixel unshuffle + LayerNorm/MLP).
pub struct Lfm2VlConnector {
    layer_norm: Option<LayerNorm>,
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
    downsample_factor: i32,
}

impl Lfm2VlConnector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        downsample_factor: i32,
        use_layernorm: bool,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let layer_norm = if use_layernorm {
            // projector_use_layernorm; default eps 1e-5.
            Some(load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm"),
                1e-5,
            )?)
        } else {
            None
        };
        Ok(Self {
            layer_norm,
            linear_1: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.linear_1"),
                gs,
                bits,
            )?,
            linear_2: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.linear_2"),
                gs,
                bits,
            )?,
            downsample_factor: downsample_factor.max(1),
        })
    }

    /// `vision_out`: `[1, h*w, hidden]`; `grid`: `(h, w)`. Returns
    /// `[T, text_hidden]` image tokens, `T = ceil(h/f) * ceil(w/f)`.
    pub fn forward(&self, vision_out: &MlxArray, grid: (i32, i32)) -> UniquePtr<MlxArray> {
        let hidden = *mlxcel_core::array_shape(vision_out).last().unwrap();
        let (h, w) = grid;
        let x = mlxcel_core::reshape(vision_out, &[1, h, w, hidden]);
        let x = pixel_unshuffle(&x, self.downsample_factor);

        let x = match &self.layer_norm {
            Some(ln) => ln.forward(&x),
            None => x,
        };
        let x = self.linear_1.forward(&x);
        let x = mlxcel_core::gelu(&x); // exact (erf) GELU, projector_hidden_act = "gelu".
        let x = self.linear_2.forward(&x);

        let s = mlxcel_core::array_shape(&x);
        mlxcel_core::reshape(&x, &[s[1] * s[2], s[3]])
    }
}

#[cfg(test)]
mod tests {
    use super::pixel_unshuffle;

    /// Reference space-to-depth over row-major `[1, h, w, C]` data.
    fn reference(
        input: &[f32],
        h: usize,
        w: usize,
        c: usize,
        f: usize,
    ) -> (Vec<f32>, usize, usize, usize) {
        let hp = h.div_ceil(f) * f;
        let wp = w.div_ceil(f) * f;
        let (oh, ow, oc) = (hp / f, wp / f, c * f * f);
        let get = |y: usize, x: usize, ch: usize| -> f32 {
            if y < h && x < w {
                input[(y * w + x) * c + ch]
            } else {
                0.0
            }
        };
        let mut out = vec![0f32; oh * ow * oc];
        for ar in 0..oh {
            for br in 0..ow {
                for dr in 0..f {
                    for dc in 0..f {
                        for ch in 0..c {
                            let oidx = ((ar * ow + br) * oc) + (dr * f + dc) * c + ch;
                            out[oidx] = get(ar * f + dr, br * f + dc, ch);
                        }
                    }
                }
            }
        }
        (out, oh, ow, oc)
    }

    fn run_case(h: usize, w: usize, c: usize, f: usize) {
        let data: Vec<f32> = (0..(h * w * c)).map(|i| i as f32).collect();
        let x = mlxcel_core::from_slice_f32(&data, &[1, h as i32, w as i32, c as i32]);
        let out = pixel_unshuffle(&x, f as i32);
        mlxcel_core::eval(&out);
        let (want, oh, ow, oc) = reference(&data, h, w, c, f);
        assert_eq!(
            mlxcel_core::array_shape(&out),
            vec![1, oh as i32, ow as i32, oc as i32]
        );
        let flat = mlxcel_core::reshape(&out, &[(oh * ow * oc) as i32]);
        mlxcel_core::eval(&flat);
        for (i, &want_v) in want.iter().enumerate() {
            let cell = mlxcel_core::slice(&flat, &[i as i32], &[i as i32 + 1]);
            mlxcel_core::eval(&cell);
            assert!(
                (mlxcel_core::item_f32(&cell) - want_v).abs() < 1e-6,
                "pixel_unshuffle[{i}] mismatch"
            );
        }
    }

    #[test]
    fn even_grid_matches_reference() {
        run_case(4, 6, 3, 2);
    }

    #[test]
    fn odd_grid_zero_pads_matches_reference() {
        // 3x5 grid, f=2 -> padded to 4x6 -> 2x3 output.
        run_case(3, 5, 2, 2);
    }
}
