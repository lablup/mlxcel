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

//! Probabilistic residual vector quantizer (inference path).
//!
//! Ports `ProbabilisticResidualVectorQuantizer` from
//! `mlx_audio/codec/models/nemotron_voicechat/codec.py`. Only the codebook
//! means (`prvq.mus_list.N`, `[codebook_size, latent_dim]`) are used; the
//! per-codebook variances (`prvq._variance_list.N.variance` in the
//! checkpoint) belong to the training/generative path and are not loaded.
//!
//! Dtypes mirror the reference: encoding runs against the f32 encoder
//! latents (bf16 codebooks promote), while decoding accumulates the 31
//! gathered codebook rows sequentially in the codebook dtype (bf16).

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::CodecConfig;
use super::layers::{copy_weight, native_sum_last, scalar_like};

pub(crate) struct ResidualQuantizer {
    /// Codebook means, one `[codebook_size, latent_dim]` per quantizer.
    means: Vec<UniquePtr<MlxArray>>,
    /// `sum(means * means, -1)` per codebook, computed in the codebook dtype.
    means_sq: Vec<UniquePtr<MlxArray>>,
    latent_dim: i32,
}

impl ResidualQuantizer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &CodecConfig,
    ) -> Result<Self, String> {
        let expected = [
            i32::try_from(config.codebook_size).map_err(|e| e.to_string())?,
            i32::try_from(config.latent_dim).map_err(|e| e.to_string())?,
        ];
        let mut means = Vec::with_capacity(config.num_quantizers);
        let mut means_sq = Vec::with_capacity(config.num_quantizers);
        for index in 0..config.num_quantizers {
            let key = format!("{prefix}.mus_list.{index}");
            let m = copy_weight(weights, &key)?;
            let shape = mlxcel_core::array_shape(&m);
            if shape != expected {
                return Err(format!(
                    "codec weight {key} has shape {shape:?}, expected {expected:?}"
                ));
            }
            means_sq.push(native_sum_last(&mlxcel_core::multiply(&m, &m), false));
            means.push(m);
        }
        Ok(Self {
            means,
            means_sq,
            latent_dim: expected[1],
        })
    }

    pub(crate) fn num_quantizers(&self) -> usize {
        self.means.len()
    }

    /// Greedy residual quantization of `[B, T, D]` latents into `[B, Q, T]`
    /// int32 code ids (squared-distance argmin per codebook).
    pub(crate) fn encode(&self, latents: &MlxArray) -> UniquePtr<MlxArray> {
        let mut residual = mlxcel_core::copy(latents);
        let mut codes = Vec::with_capacity(self.means.len());
        for (m, m_sq) in self.means.iter().zip(&self.means_sq) {
            let r_sq =
                mlxcel_core::sum_axis(&mlxcel_core::multiply(&residual, &residual), -1, true);
            let cross = mlxcel_core::matmul(&residual, &mlxcel_core::transpose(m));
            let scaled = mlxcel_core::multiply(&scalar_like(2.0, &cross), &cross);
            let distances = mlxcel_core::add(&mlxcel_core::subtract(&r_sq, &scaled), m_sq);
            let indices = mlxcel_core::argmin(&distances, -1, false);
            let chosen = mlxcel_core::take(m, &indices, 0);
            residual = mlxcel_core::subtract(&residual, &chosen);
            codes.push(indices);
        }
        let stacked = mlxcel_core::stack_owned(&codes, 1);
        mlxcel_core::astype(&stacked, mlxcel_core::dtype::INT32)
    }

    /// Sum the selected codebook rows of `[B, Q, T]` codes into `[B, T, D]`
    /// latents. Callers validate shape and range first.
    pub(crate) fn decode(&self, codes: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(codes);
        let (batch, quantizers, frames) = (shape[0], shape[1], shape[2]);
        let dtype = mlxcel_core::array_dtype(&self.means[0]);
        let mut latents = mlxcel_core::zeros(&[batch, frames, self.latent_dim], dtype);
        for (index, m) in self.means.iter().enumerate().take(quantizers as usize) {
            let q = index as i32;
            let ids = mlxcel_core::slice(codes, &[0, q, 0], &[batch, q + 1, frames]);
            let ids = mlxcel_core::reshape(&ids, &[batch, frames]);
            latents = mlxcel_core::add(&latents, &mlxcel_core::take(m, &ids, 0));
        }
        latents
    }
}
