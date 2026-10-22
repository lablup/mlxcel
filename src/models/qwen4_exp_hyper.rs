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

//! Qwen4-Exp low-rank gated hyper-connections (`Qwen4ExpGatedResidual`).
//!
//! The residual stream is `[B, L, hc_count * hidden]`: `hc_count` contiguous
//! copies of the hidden state. Each sublayer reads one stream mixed from all
//! of them and writes its output back into every stream with a per-stream
//! gate. This is not DeepSeek-V4's Sinkhorn `HyperConnection`.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::ResolvedQwen4ExpConfig;
use super::norm::ZeroCenteredRmsNorm;
use super::shapes;

pub(crate) struct GatedResidual {
    hc_norm: ZeroCenteredRmsNorm,
    input_mix_weight_down: UnifiedLinear,
    input_mix_weight_up: UnifiedLinear,
    block_inject_weight: Option<UnifiedLinear>,
    hc_count: i32,
    hidden_size: i32,
}

impl GatedResidual {
    /// `with_inject` is false only for the model-level `hyper_connection_mixer`,
    /// which collapses the streams once after the last layer.
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
        prefix: &str,
        with_inject: bool,
    ) -> Result<Self, String> {
        let raw = &config.raw;
        let streams = raw.hc_count * raw.hidden_size;
        let linear = |name: &str, rows: usize, columns: usize| {
            let module = format!("{prefix}.{name}");
            let (group_size, bits) = config.quantization.for_module(&module);
            shapes::expect_matrix(weights, &module, (group_size, bits), &[rows, columns])?;
            UnifiedLinear::from_weights(weights, &module, group_size, bits)
        };
        Ok(Self {
            hc_norm: ZeroCenteredRmsNorm::from_weights(
                weights,
                &format!("{prefix}.hc_norm"),
                streams,
                Some(raw.hidden_size),
                raw.rms_norm_eps,
            )?,
            input_mix_weight_down: linear("input_mix_weight_down", raw.hc_lowrank, streams)?,
            input_mix_weight_up: linear("input_mix_weight_up", streams, raw.hc_lowrank)?,
            block_inject_weight: if with_inject {
                Some(linear("block_inject_weight", raw.hc_count, streams)?)
            } else {
                None
            },
            hc_count: config.raw.hc_count as i32,
            hidden_size: config.raw.hidden_size as i32,
        })
    }

    fn streams(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let mut split = shape[..shape.len() - 1].to_vec();
        split.extend([self.hc_count, self.hidden_size]);
        mlxcel_core::reshape(x, &split)
    }

    fn mix(&self, normed: &MlxArray) -> UniquePtr<MlxArray> {
        let down = self.input_mix_weight_down.forward(normed);
        let activated = mlxcel_core::qwen4_hyper_activate(&down, self.hc_count);
        let projected = self.input_mix_weight_up.forward(&activated);
        mlxcel_core::qwen4_hyper_mix(&self.streams(&projected), &self.streams(normed))
    }

    /// The mixer path: the `[B, L, hidden]` mixed stream.
    pub(crate) fn collapse(&self, hyper: &MlxArray) -> UniquePtr<MlxArray> {
        self.mix(&self.hc_norm.forward(hyper))
    }

    /// Returns the sublayer input `[B, L, hidden]` and the per-stream injection
    /// weights `[B, L, hc_count]` (`2 * sigmoid(inject(normed) / hc_count)`).
    pub(crate) fn split(&self, hyper: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let normed = self.hc_norm.forward(hyper);
        let mixed = self.mix(&normed);
        let inject = self
            .block_inject_weight
            .as_ref()
            .expect("split() is only called on a sublayer hyper-connection");
        let projected = super::linear::row_invariant_projection(inject, &normed);
        let weights = mlxcel_core::qwen4_hyper_gate(&projected, self.hc_count);
        (mixed, weights)
    }

    /// `hyper + (branch[.., None, :] * weights[.., None]).flatten(-2)`.
    pub(crate) fn inject(
        &self,
        branch: &MlxArray,
        hyper: &MlxArray,
        weights: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let branch = mlxcel_core::expand_dims(branch, -2);
        let weights = mlxcel_core::expand_dims(weights, -1);
        let output = mlxcel_core::qwen4_hyper_inject(&branch, &weights, &self.streams(hyper));
        mlxcel_core::reshape(&output, &mlxcel_core::array_shape(hyper))
    }
}
