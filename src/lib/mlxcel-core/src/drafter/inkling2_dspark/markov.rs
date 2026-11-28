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

//! Upstream `MarkovHead` (`markov_head_type = "rnn_hidden"`): a GRU-style
//! recurrence over the proposals whose output, through `hidden_out`, is
//! folded into the target LM head once at bind time.

use cxx::UniquePtr;

use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedLinear};
use crate::weights::WeightMap;

use super::model::{f32, norm_weight};

/// Rows of the LM head folded per tile, as upstream.
pub(super) const FOLD_TILE_ROWS: i32 = 4096;

pub struct MarkovHead {
    /// `[vocab, rank]` token table.
    markov_w1: UniquePtr<MlxArray>,
    /// `rnn_proj.weight[:, :2 * rank]` in f32, transposed: `[2 * rank, 3 * rank]`.
    state_weight: UniquePtr<MlxArray>,
    /// `rnn_proj.weight[:, 2 * rank:]` in f32, transposed: `[2 * hidden, 3 * rank]`.
    context_weight: UniquePtr<MlxArray>,
    /// `rnn_proj.bias` in f32.
    bias: UniquePtr<MlxArray>,
    /// `hidden_out.weight` `[hidden, rank]` and `hidden_out.bias` `[hidden]`.
    hidden_out_weight: UniquePtr<MlxArray>,
    hidden_out_bias: UniquePtr<MlxArray>,
    pub(super) dflare_norm: RMSNorm,
    pub(super) rank: i32,
    /// `lm_head @ hidden_out` `[vocab, rank]` and `lm_head @ hidden_out.bias`
    /// `[vocab]`, in the `hidden_out` dtype. Set by [`Self::fold`].
    folded: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
}

fn tensor(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|value| ffi::copy(value))
        .ok_or_else(|| format!("Inkling 2 DSpark weight not found: {name}"))
}

impl MarkovHead {
    pub(super) fn from_weights(
        weights: &WeightMap,
        hidden: usize,
        rank: usize,
        eps: f32,
    ) -> Result<Self, String> {
        let (hidden, rank) = (hidden as i32, rank as i32);
        let markov_w1 = tensor(weights, "markov_head.markov_w1.weight")?;
        let rnn = tensor(weights, "markov_head.rnn_proj.weight")?;
        let expected = [3 * rank, 2 * rank + 2 * hidden];
        if ffi::array_shape(&rnn) != expected {
            return Err(format!(
                "markov_head.rnn_proj.weight must be {expected:?}, got {:?}",
                ffi::array_shape(&rnn)
            ));
        }
        let columns = |start: i32, stop: i32| {
            let part = ffi::slice(&rnn, &[0, start], &[3 * rank, stop]);
            ffi::contiguous(&ffi::transpose_axes(&f32(&part), &[1, 0]), false)
        };
        let hidden_out_weight = tensor(weights, "markov_head.hidden_out.weight")?;
        if ffi::array_shape(&hidden_out_weight) != [hidden, rank] {
            return Err(format!(
                "markov_head.hidden_out.weight must be [{hidden}, {rank}], got {:?}",
                ffi::array_shape(&hidden_out_weight)
            ));
        }
        let rnn_bias = tensor(weights, "markov_head.rnn_proj.bias")?;
        Ok(Self {
            state_weight: columns(0, 2 * rank),
            context_weight: columns(2 * rank, 2 * rank + 2 * hidden),
            bias: f32(&rnn_bias),
            markov_w1,
            hidden_out_weight,
            hidden_out_bias: tensor(weights, "markov_head.hidden_out.bias")?,
            dflare_norm: norm_weight(weights, "markov_head.dflare_norm.weight", eps)?,
            rank,
            folded: None,
        })
    }

    pub(super) fn is_folded(&self) -> bool {
        self.folded.is_some()
    }

    /// Fold `hidden_out` into `lm_head` once, in f64 on the host over tiles
    /// of [`FOLD_TILE_ROWS`] vocabulary rows, then round to f32 and to the
    /// `hidden_out` dtype: upstream `MarkovHead.fold`. A quantized head is
    /// dequantized first, as upstream does.
    pub(super) fn fold(&mut self, lm_head: &UnifiedLinear) {
        let head = match lm_head {
            UnifiedLinear::Regular(linear) => ffi::copy(&linear.weight),
            UnifiedLinear::Quantized { .. } => lm_head.dequantized_weight(),
        };
        let shape = ffi::array_shape(&head);
        let (vocab, hidden) = (shape[0], shape[1]);
        let rank = self.rank as usize;
        let projection: Vec<f64> = crate::utils::array_to_vec_f32(&self.hidden_out_weight)
            .into_iter()
            .map(f64::from)
            .collect();
        let bias: Vec<f64> = crate::utils::array_to_vec_f32(&self.hidden_out_bias)
            .into_iter()
            .map(f64::from)
            .collect();
        let mut weight_out = Vec::with_capacity(vocab as usize * rank);
        let mut bias_out = Vec::with_capacity(vocab as usize);
        let mut start = 0;
        while start < vocab {
            let stop = (start + FOLD_TILE_ROWS).min(vocab);
            let rows =
                crate::utils::array_to_vec_f32(&ffi::slice(&head, &[start, 0], &[stop, hidden]));
            let (w, b) = fold_tile(&rows, &projection, &bias, hidden as usize, rank);
            weight_out.extend(w);
            bias_out.extend(b);
            start = stop;
        }
        let dtype = ffi::array_dtype(&self.hidden_out_weight);
        let weight = ffi::astype(
            &ffi::from_slice_f32(&weight_out, &[vocab, self.rank]),
            dtype,
        );
        let bias = ffi::astype(&ffi::from_slice_f32(&bias_out, &[vocab]), dtype);
        ffi::eval(&weight);
        ffi::eval(&bias);
        self.folded = Some((weight, bias));
    }

    /// `[B, L, 3 * rank]` f32: the context part of `rnn_proj`, computed once
    /// per block from the final hidden rows and the fused `f_rnn`.
    pub(super) fn context(&self, hidden: &MlxArray, f_rnn: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(hidden);
        let broadcast = ffi::broadcast_to(f_rnn, &shape);
        let inputs = crate::ops::concatenate(hidden, &broadcast, -1);
        ffi::add(
            &ffi::matmul(&f32(&inputs), &self.context_weight),
            &self.bias,
        )
    }

    /// One recurrence step: `(state, logit_bias)` for `state` `[B, rank]`,
    /// `previous` `[B]` token ids and `context` `[B, 3 * rank]` f32.
    pub(super) fn step(
        &self,
        state: &MlxArray,
        previous: &MlxArray,
        context: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let state_dtype = ffi::array_dtype(state);
        let embedded = ffi::take(&self.markov_w1, previous, 0);
        let inputs = crate::ops::concatenate(state, &embedded, -1);
        let raw = ffi::astype(
            &ffi::add(&ffi::matmul(&f32(&inputs), &self.state_weight), context),
            state_dtype,
        );
        let shape = ffi::array_shape(&raw);
        let part = |i: i32| ffi::slice(&raw, &[0, i * self.rank], &[shape[0], (i + 1) * self.rank]);
        let gate = ffi::astype(&ffi::sigmoid(&f32(&part(0))), state_dtype);
        let candidate = ffi::astype(&ffi::tanh(&f32(&part(1))), state_dtype);
        let keep = ffi::subtract(&ffi::ones_like(&gate), &gate);
        let state = ffi::add(
            &ffi::multiply(&gate, state),
            &ffi::multiply(&keep, &candidate),
        );
        let latent = ffi::astype(&ffi::tanh(&f32(&part(2))), state_dtype);
        let (weight, bias) = self
            .folded
            .as_ref()
            .expect("fold the LM head before stepping");
        let logit_bias = ffi::addmm(
            bias,
            &latent,
            &ffi::transpose_axes(weight, &[1, 0]),
            1.0,
            1.0,
        );
        (state, logit_bias)
    }
}

/// `rows @ projection` and `rows @ bias` in f64 for one tile, rounded to f32.
/// `rows` is `[tile, hidden]`, `projection` `[hidden, rank]`.
///
/// Used by: [`MarkovHead::fold`] and its reference test.
pub(super) fn fold_tile(
    rows: &[f32],
    projection: &[f64],
    bias: &[f64],
    hidden: usize,
    rank: usize,
) -> (Vec<f32>, Vec<f32>) {
    let tile = rows.len() / hidden;
    let mut weight = vec![0f32; tile * rank];
    let mut folded_bias = vec![0f32; tile];
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 32);
    let per_thread = tile.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for ((weight_chunk, bias_chunk), row_chunk) in weight
            .chunks_mut(per_thread * rank)
            .zip(folded_bias.chunks_mut(per_thread))
            .zip(rows.chunks(per_thread * hidden))
        {
            scope.spawn(move || {
                let mut acc = vec![0f64; rank];
                for (row_index, row) in row_chunk.chunks_exact(hidden).enumerate() {
                    acc.iter_mut().for_each(|a| *a = 0.0);
                    let mut bias_acc = 0f64;
                    for (h, &value) in row.iter().enumerate() {
                        let value = f64::from(value);
                        bias_acc += value * bias[h];
                        let projected = &projection[h * rank..(h + 1) * rank];
                        for (a, &p) in acc.iter_mut().zip(projected) {
                            *a += value * p;
                        }
                    }
                    for (out, &a) in weight_chunk[row_index * rank..(row_index + 1) * rank]
                        .iter_mut()
                        .zip(&acc)
                    {
                        *out = a as f32;
                    }
                    bias_chunk[row_index] = bias_acc as f32;
                }
            });
        }
    });
    (weight, folded_bias)
}
