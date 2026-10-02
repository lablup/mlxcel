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

//! Residual-VQ codebooks of the EAR-TTS decoder (`rvq_embs`).
//!
//! Ports `RVQEARTTSModel.depthsum_embedding` and `_rvq_encode_step` from
//! `mlx_vlm/models/nemotron_voicechat/tts.py`.
//!
//! Two details are kept on purpose because they decide which code an
//! `argmin` lands on:
//!
//! - The depth sum starts from `f32` zeros and adds the 31 gathered bf16
//!   rows one codebook at a time, in order. Gathering all rows and reducing
//!   would change the `f32` summation order.
//! - The squared norms `sum(e * e, -1)` stay in the codebook dtype (bf16)
//!   and are computed per codebook on a `[codebook_size, latent]` slice, the
//!   same reduction the reference runs on every call; they are cached here
//!   because they only depend on the weights.
//!
//! Index `codebook_size` (1024) is the mask token: it selects an all-zero
//! padding row, so masked codebooks contribute nothing to the embedding.

use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::norm_mlp::{native_sum_axis, scalar, weight_with_shape};

/// Per-codebook RVQ tables.
pub struct RvqCodebooks {
    /// `[codebook_size, latent]` rows of each codebook.
    embs: Vec<UniquePtr<MlxArray>>,
    /// `[codebook_size + 1, latent]`: the codebook plus a zero mask row.
    padded: Vec<UniquePtr<MlxArray>>,
    /// `[codebook_size]` squared norms in the codebook dtype.
    norms: Vec<UniquePtr<MlxArray>>,
    num_quantizers: i32,
    latent_size: i32,
}

impl RvqCodebooks {
    /// Load `{key}` shaped `[num_quantizers, codebook_size, latent_size]`.
    pub fn from_weights(
        weights: &WeightMap,
        key: &str,
        num_quantizers: usize,
        codebook_size: usize,
        latent_size: usize,
    ) -> Result<Self, String> {
        let (q, c, l) = (
            num_quantizers as i32,
            codebook_size as i32,
            latent_size as i32,
        );
        let all = weight_with_shape(weights, key, &[q, c, l])?;
        let dtype_id = mlxcel_core::array_dtype(&all);
        let zero_row = mlxcel_core::zeros(&[1, l], dtype_id);
        let mut embs = Vec::with_capacity(num_quantizers);
        let mut padded = Vec::with_capacity(num_quantizers);
        let mut norms = Vec::with_capacity(num_quantizers);
        for idx in 0..q {
            let emb = mlxcel_core::slice(&all, &[idx, 0, 0], &[idx + 1, c, l]);
            let emb = mlxcel_core::reshape(&emb, &[c, l]);
            let norm = native_sum_axis(&mlxcel_core::multiply(&emb, &emb), 1);
            padded.push(mlxcel_core::concatenate(&emb, &zero_row, 0));
            norms.push(norm);
            embs.push(emb);
        }
        let mut arrays: Vec<&MlxArray> = Vec::with_capacity(3 * num_quantizers);
        arrays.extend(embs.iter().map(|a| &**a));
        arrays.extend(padded.iter().map(|a| &**a));
        arrays.extend(norms.iter().map(|a| &**a));
        for a in arrays {
            mlxcel_core::eval(a);
        }
        Ok(Self {
            embs,
            padded,
            norms,
            num_quantizers: q,
            latent_size: l,
        })
    }

    /// Codebook `idx` of `code` (`[B, T, Q]`) as `[B, T]`.
    fn column(code: &MlxArray, idx: i32) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(code);
        let col = mlxcel_core::slice(code, &[0, 0, idx], &[s[0], s[1], idx + 1]);
        mlxcel_core::reshape(&col, &[s[0], s[1]])
    }

    /// `depthsum_embedding`: `sum_q table_q[code[..., q]]` in `f32`, with the
    /// mask index selecting a zero row. `code`: integer `[B, T, Q]`.
    pub fn depthsum_embedding(&self, code: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let s = mlxcel_core::array_shape(code);
        if s.len() != 3 || s[2] != self.num_quantizers {
            return Err(format!(
                "RVQ codes must be [batch, time, {}], got {s:?}",
                self.num_quantizers
            ));
        }
        let mut result = mlxcel_core::zeros(&[s[0], s[1], self.latent_size], dtype::FLOAT32);
        for (idx, table) in self.padded.iter().enumerate() {
            let rows = mlxcel_core::take(table, &Self::column(code, idx as i32), 0);
            // Cast the gathered rows explicitly: CUDA builds resolve
            // bf16 + f32 to bf16 (issue #2087). The cast is exact and is the
            // same `astype` upstream promotion inserts elsewhere.
            let rows = mlxcel_core::astype(&rows, dtype::FLOAT32);
            result = mlxcel_core::add(&result, &rows);
        }
        Ok(result)
    }

    /// `_rvq_encode_step`: greedily quantize `residual` (`[B, T, latent]`)
    /// into codebooks `start .. start + count`, keeping the other columns of
    /// `code`. Returns the updated int32 `[B, T, Q]` codes.
    pub fn encode_step(
        &self,
        residual: &MlxArray,
        code: &MlxArray,
        start: usize,
        count: usize,
    ) -> Result<UniquePtr<MlxArray>, String> {
        if start + count > self.num_quantizers as usize {
            return Err(format!(
                "RVQ step {start}..{} exceeds {} codebooks",
                start + count,
                self.num_quantizers
            ));
        }
        let mut pieces: Vec<UniquePtr<MlxArray>> = (0..self.num_quantizers)
            .map(|idx| mlxcel_core::astype(&Self::column(code, idx), dtype::INT32))
            .collect();
        let mut residual = mlxcel_core::copy(residual);
        for (offset, piece) in pieces[start..start + count].iter_mut().enumerate() {
            let idx = start + offset;
            let emb = &self.embs[idx];
            let dots = mlxcel_core::matmul(&residual, &mlxcel_core::transpose(emb));
            let dots = mlxcel_core::multiply(&scalar(2.0, mlxcel_core::array_dtype(&dots)), &dots);
            let distances = mlxcel_core::subtract(&self.norms[idx], &dots);
            let selected = mlxcel_core::argmin(&distances, -1, false);
            residual = mlxcel_core::subtract(&residual, &mlxcel_core::take(emb, &selected, 0));
            *piece = mlxcel_core::astype(&selected, dtype::INT32);
        }
        Ok(mlxcel_core::stack_owned(&pieces, -1))
    }
}
