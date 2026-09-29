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

//! Load-time f32 copies of half-precision weights that only ever meet f32
//! activations (issue #2045).
//!
//! The Nemotron VoiceChat encoder, EAR-TTS backbone and MoG head run f32
//! activations against bf16 checkpoint weights, following the reference.
//! MLX's `matmul`, `addmm`, `conv_general`, `fast::layer_norm` and
//! `fast::rms_norm` all promote to the common dtype by inserting
//! `astype(weight, float32)` into the graph, so every frame re-casts the
//! same gigabytes of weights before the f32 kernel reads them again. The
//! bf16 to f32 conversion is exact, so casting once at load hands the
//! identical f32 values to the identical kernels: the outputs are
//! bit-identical, and each frame skips the cast traffic. The price is twice
//! the resident size of the promoted weights.
//!
//! A weight may be promoted only when every op that reads it promotes it
//! against an f32 activation. Weights combined with other weights in their
//! own dtype first (for example Gemma's `1 + weight` norms) or read by a
//! bf16 path (the EAR-TTS subword condition) must stay as stored, which is
//! why callers pass an explicit key filter.
//!
//! Used by: `VoiceChatPerception::from_weights` (FastConformer encoder and
//! `proj`), `RvqEarTtsModel::from_weights` (backbone and MoG projections).

use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Whether `dtype_id` is a half-precision float that promotes to f32.
fn is_half(dtype_id: i32) -> bool {
    dtype_id == dtype::BFLOAT16 || dtype_id == dtype::FLOAT16
}

/// The entries of `weights` whose key starts with `prefix`, keys unchanged,
/// with every bf16 / f16 entry accepted by `promote` cast to f32 and
/// evaluated. Other entries are shared (`copy`) as stored.
pub(crate) fn promoted_subset(
    weights: &WeightMap,
    prefix: &str,
    promote: impl Fn(&str) -> bool,
) -> WeightMap {
    let mut out = WeightMap::new();
    let mut cast: Vec<UniquePtr<MlxArray>> = Vec::new();
    let mut cast_keys = Vec::new();
    for (key, value) in weights.iter().filter(|(k, _)| k.starts_with(prefix)) {
        if is_half(mlxcel_core::array_dtype(value)) && promote(key) {
            cast.push(mlxcel_core::astype(value, dtype::FLOAT32));
            cast_keys.push(key.clone());
        } else {
            out.insert(key.clone(), mlxcel_core::copy(value));
        }
    }
    if !cast.is_empty() {
        let ptrs: Vec<*const MlxArray> = cast.iter().map(|a| &**a as *const MlxArray).collect();
        // SAFETY: every pointer refers to an array owned by `cast`, which
        // outlives the call.
        unsafe { mlxcel_core::eval_all(&ptrs) };
    }
    out.extend(cast_keys.into_iter().zip(cast));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(seed: u64, shape: &[i32]) -> UniquePtr<MlxArray> {
        let n: i32 = shape.iter().product();
        let mut s = seed;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        mlxcel_core::from_slice_f32(&v, shape)
    }

    fn bytes(a: &MlxArray) -> Vec<u8> {
        mlxcel_core::eval(a);
        mlxcel_core::array_to_raw_bytes(a)
    }

    fn bf16(a: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::astype(a, dtype::BFLOAT16)
    }

    fn f32_of(a: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::astype(a, dtype::FLOAT32)
    }

    #[test]
    fn precast_matmul_and_addmm_match_the_promoted_graph() {
        let w = bf16(&rand(1, &[48, 64]));
        let b = bf16(&rand(2, &[48]));
        let w32 = f32_of(&w);
        let b32 = f32_of(&b);
        for rows in [1, 2, 7] {
            let x = rand(3 + rows as u64, &[1, rows, 64]);
            let promoted = mlxcel_core::matmul(&x, &mlxcel_core::transpose(&w));
            let precast = mlxcel_core::matmul(&x, &mlxcel_core::transpose(&w32));
            assert_eq!(mlxcel_core::array_dtype(&promoted), dtype::FLOAT32);
            assert_eq!(bytes(&promoted), bytes(&precast));
            let promoted = mlxcel_core::addmm(&b, &x, &mlxcel_core::transpose(&w), 1.0, 1.0);
            let precast = mlxcel_core::addmm(&b32, &x, &mlxcel_core::transpose(&w32), 1.0, 1.0);
            assert_eq!(bytes(&promoted), bytes(&precast));
        }
    }

    #[test]
    fn precast_conv_and_layer_norm_match_the_promoted_graph() {
        let x = rand(9, &[1, 12, 16]);
        for (shape, groups) in [([32, 1, 16], 1), ([16, 9, 1], 16)] {
            let w = bf16(&rand(10, &shape));
            let promoted = mlxcel_core::try_conv1d(&x, &w, 1, 0, 1, groups).unwrap();
            let precast = mlxcel_core::try_conv1d(&x, &f32_of(&w), 1, 0, 1, groups).unwrap();
            assert_eq!(bytes(&promoted), bytes(&precast));
        }
        let g = bf16(&rand(11, &[16]));
        let beta = bf16(&rand(12, &[16]));
        let (g32, beta32) = (f32_of(&g), f32_of(&beta));
        // SAFETY: all four pointers refer to arrays that outlive the calls.
        let (promoted, precast) = unsafe {
            (
                mlxcel_core::fast_layer_norm(&x, &*g, &*beta, 1e-5),
                mlxcel_core::fast_layer_norm(&x, &*g32, &*beta32, 1e-5),
            )
        };
        assert_eq!(bytes(&promoted), bytes(&precast));
    }

    #[test]
    fn promoted_subset_filters_by_prefix_and_predicate() {
        let mut w = WeightMap::new();
        w.insert("a.proj.weight".into(), bf16(&rand(1, &[2, 2])));
        w.insert("a.norm.weight".into(), bf16(&rand(2, &[2])));
        w.insert("a.ids".into(), mlxcel_core::from_slice_i32(&[1, 2], &[2]));
        w.insert("b.proj.weight".into(), bf16(&rand(3, &[2, 2])));
        let out = promoted_subset(&w, "a.", |k| !k.contains("norm"));
        assert_eq!(out.len(), 3);
        assert!(!out.contains_key("b.proj.weight"));
        assert_eq!(
            mlxcel_core::array_dtype(&out["a.proj.weight"]),
            dtype::FLOAT32
        );
        assert_eq!(
            mlxcel_core::array_dtype(&out["a.norm.weight"]),
            dtype::BFLOAT16
        );
        assert_eq!(mlxcel_core::array_dtype(&out["a.ids"]), dtype::INT32);
        assert_eq!(
            bytes(&out["a.proj.weight"]),
            bytes(&f32_of(&w["a.proj.weight"]))
        );
    }
}
