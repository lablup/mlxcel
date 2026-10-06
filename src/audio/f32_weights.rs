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
//! On CUDA builds the precast is required, not only faster: mlxcel's CUDA
//! overlay of MLX's promotion table (`src/lib/mlx-cpp/patches-cuda/dtype.cpp`,
//! kept for the single-dtype bf16 decode graph of issue #636) resolves bf16
//! with f32 to bf16, so a bf16 weight left as stored would demote the f32
//! activation stream instead of promoting the weight (issue #2087). Code that
//! needs f32 from a stored bf16 tensor outside these subsets casts it
//! explicitly.
//!
//! A weight may be promoted only when every op that reads it promotes it
//! against an f32 activation. Weights combined with other weights in their
//! own dtype first (for example Gemma's `1 + weight` norms) or read by a
//! bf16 path (the EAR-TTS subword condition) must stay as stored, which is
//! why callers pass an explicit key filter.
//!
//! Used by: `VoiceChatPerception::from_weights` (FastConformer encoder and
//! `proj`), `RvqEarTtsModel::from_weights` (backbone and MoG projections,
//! `bos_emb`, `audio_prompt_projection_W`).

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

    /// The bf16 weight `w` as upstream MLX promotes it against an f32
    /// activation. Off CUDA that is MLX's own promotion inside the op, so `w`
    /// is passed as is and the tests compare the precast with it. The CUDA
    /// overlay resolves bf16 with f32 to bf16 instead (see
    /// `mixed_bf16_f32_promotion_follows_the_build`), so there the upstream
    /// promotion is spelled out as an unevaluated in-graph `astype`.
    fn as_promoted(w: &MlxArray) -> UniquePtr<MlxArray> {
        if cfg!(feature = "cuda") {
            f32_of(w)
        } else {
            mlxcel_core::copy(w)
        }
    }

    /// Issue #2087 probe: the dtype an f32 activation meets a bf16 operand in.
    /// Upstream MLX gives f32. mlxcel's CUDA overlay
    /// (`src/lib/mlx-cpp/patches-cuda/dtype.cpp`, issue #636) gives bf16 on
    /// every device of a CUDA build, because the promotion table is compiled
    /// into the library; ROCm and Metal use the upstream table. Production
    /// code must therefore cast explicitly wherever it needs f32 from a stored
    /// bf16 tensor, and this test fails loudly if either rule changes.
    #[test]
    fn mixed_bf16_f32_promotion_follows_the_build() {
        let x = rand(1, &[2, 16]);
        let w = bf16(&rand(2, &[16, 16]));
        let g = bf16(&rand(3, &[16]));
        let want = if cfg!(feature = "cuda") {
            dtype::BFLOAT16
        } else {
            dtype::FLOAT32
        };
        let outputs = [
            ("add(f32, bf16)", mlxcel_core::add(&x, &g)),
            ("add(bf16, f32)", mlxcel_core::add(&g, &x)),
            ("matmul(f32, bf16)", mlxcel_core::matmul(&x, &w)),
            ("matmul(bf16, f32)", mlxcel_core::matmul(&w, &f32_of(&w))),
            (
                "rms_norm(f32, bf16)",
                mlxcel_core::fast_rms_norm(&x, &g, 1e-5),
            ),
        ];
        for (name, out) in &outputs {
            assert_eq!(mlxcel_core::array_dtype(out), want, "{name}");
        }
        // An explicit cast is backend independent and exact.
        let cast = mlxcel_core::add(&x, &f32_of(&g));
        assert_eq!(mlxcel_core::array_dtype(&cast), dtype::FLOAT32);
    }

    #[test]
    fn precast_matmul_and_addmm_match_the_promoted_graph() {
        let w = bf16(&rand(1, &[48, 64]));
        let b = bf16(&rand(2, &[48]));
        let w32 = f32_of(&w);
        let b32 = f32_of(&b);
        for rows in [1, 2, 7] {
            let x = rand(3 + rows as u64, &[1, rows, 64]);
            let promoted = mlxcel_core::matmul(&x, &mlxcel_core::transpose(&as_promoted(&w)));
            let precast = mlxcel_core::matmul(&x, &mlxcel_core::transpose(&w32));
            assert_eq!(mlxcel_core::array_dtype(&promoted), dtype::FLOAT32);
            assert_eq!(bytes(&promoted), bytes(&precast));
            let promoted = mlxcel_core::addmm(
                &as_promoted(&b),
                &x,
                &mlxcel_core::transpose(&as_promoted(&w)),
                1.0,
                1.0,
            );
            let precast = mlxcel_core::addmm(&b32, &x, &mlxcel_core::transpose(&w32), 1.0, 1.0);
            assert_eq!(bytes(&promoted), bytes(&precast));
        }
    }

    #[test]
    fn precast_conv_and_layer_norm_match_the_promoted_graph() {
        let x = rand(9, &[1, 12, 16]);
        for (shape, groups) in [([32, 1, 16], 1), ([16, 9, 1], 16)] {
            let w = bf16(&rand(10, &shape));
            let promoted = mlxcel_core::try_conv1d(&x, &as_promoted(&w), 1, 0, 1, groups).unwrap();
            let precast = mlxcel_core::try_conv1d(&x, &f32_of(&w), 1, 0, 1, groups).unwrap();
            assert_eq!(bytes(&promoted), bytes(&precast));
        }
        let g = bf16(&rand(11, &[16]));
        let beta = bf16(&rand(12, &[16]));
        let (g32, beta32) = (f32_of(&g), f32_of(&beta));
        let (gp, betap) = (as_promoted(&g), as_promoted(&beta));
        // SAFETY: every pointer refers to an array that outlives the calls.
        let (promoted, precast) = unsafe {
            (
                mlxcel_core::fast_layer_norm(&x, &*gp, &*betap, 1e-5),
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
