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

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::{InklingConfig, weight};
use crate::models::switch_layers::{SwitchGLU, moe_weighted_sum};

pub(crate) enum InklingMlp {
    Dense(InklingDenseMlp),
    Sparse(InklingSparseMoe),
}

impl InklingMlp {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &InklingConfig,
        index: usize,
    ) -> Result<Self, String> {
        if config.text_config.layer_is_dense(index) {
            Ok(Self::Dense(InklingDenseMlp::from_weights(
                weights, config, index,
            )?))
        } else {
            Ok(Self::Sparse(InklingSparseMoe::from_weights(
                weights, config, index,
            )?))
        }
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(moe) => moe.forward(x),
        }
    }
}

pub(crate) struct InklingDenseMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    global_scale: UniquePtr<MlxArray>,
}

impl InklingDenseMlp {
    fn from_weights(
        weights: &WeightMap,
        config: &InklingConfig,
        index: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{index}.mlp");
        let (group, bits, _) = config.quantization();
        let (expected_width, _) = config.text_config.widths()?;
        let gate_shape = weights
            .get(&format!("{prefix}.gate_proj.weight"))
            .map(|w| mlxcel_core::array_shape(w));
        if let Some(shape) = gate_shape
            && shape.len() == 2
            && shape[0] != expected_width as i32
        {
            return Err(format!(
                "{prefix}.gate_proj.weight: config dense width {expected_width} disagrees with {shape:?}"
            ));
        }
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                group,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.up_proj"),
                group,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                group,
                bits,
            )?,
            global_scale: weight(weights, &format!("{prefix}.global_scale"))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        let down = self.down_proj.forward(&activated);
        mlxcel_core::multiply(&down, &self.global_scale)
    }
}

pub(crate) struct InklingSparseMoe {
    gate_weight: UniquePtr<MlxArray>,
    correction_bias: UniquePtr<MlxArray>,
    global_scale: UniquePtr<MlxArray>,
    switch_mlp: SwitchGLU,
    gate_scale: Option<UniquePtr<MlxArray>>,
    out_scale: Option<UniquePtr<MlxArray>>,
    shared_gate: UnifiedLinear,
    shared_up: UnifiedLinear,
    shared_down: UnifiedLinear,
    n_routed: i32,
    n_shared: i32,
    top_k: i32,
    intermediate: i32,
    route_scale: f32,
}

impl InklingSparseMoe {
    fn from_weights(
        weights: &WeightMap,
        config: &InklingConfig,
        index: usize,
    ) -> Result<Self, String> {
        let text = &config.text_config;
        let prefix = format!("model.layers.{index}.mlp");
        let switch_prefix = format!("{prefix}.switch_mlp");
        let (group, bits, _) = config.quantization();
        let quantized = weights.contains_key(&format!("{switch_prefix}.gate_proj.scales"));
        let has_biases = weights.contains_key(&format!("{switch_prefix}.gate_proj.biases"));
        let (expert_group, expert_mode) = if quantized && !has_biases {
            (16, "nvfp4")
        } else {
            (group, "affine")
        };
        let (_, intermediate) = text.widths()?;
        let gate_scale = (quantized && !has_biases)
            .then(|| {
                weights
                    .get(&format!("{switch_prefix}.gate_scale"))
                    .map(|v| mlxcel_core::copy(v))
            })
            .flatten();
        let out_scale = (quantized && !has_biases)
            .then(|| {
                weights
                    .get(&format!("{switch_prefix}.out_scale"))
                    .map(|v| mlxcel_core::copy(v))
            })
            .flatten();
        let correction_bias = weight(weights, &format!("{prefix}.e_score_correction_bias"))?;
        Ok(Self {
            gate_weight: weight(weights, &format!("{prefix}.gate_weight"))?,
            correction_bias: mlxcel_core::astype(&correction_bias, dtype::FLOAT32),
            global_scale: weight(weights, &format!("{prefix}.global_scale"))?,
            switch_mlp: SwitchGLU::from_weights_with_mode(
                weights,
                &switch_prefix,
                expert_group,
                bits,
                expert_mode,
            )?,
            gate_scale,
            out_scale,
            shared_gate: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.shared_experts.gate_proj"),
                group,
                bits,
            )?,
            shared_up: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.shared_experts.up_proj"),
                group,
                bits,
            )?,
            shared_down: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.shared_experts.down_proj"),
                group,
                bits,
            )?,
            n_routed: text.n_routed_experts as i32,
            n_shared: text.n_shared_experts as i32,
            top_k: text.num_experts_per_tok as i32,
            intermediate: intermediate as i32,
            route_scale: text.route_scale,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let hidden = shape[2];
        let flat = mlxcel_core::reshape(x, &[-1, hidden]);
        let logits = mlxcel_core::matmul(&flat, &mlxcel_core::transpose(&self.gate_weight));
        let (indices, routed_weights, shared_weights) = route_weights(
            &logits,
            &self.correction_bias,
            &self.global_scale,
            self.n_routed,
            self.n_shared,
            self.top_k,
            self.route_scale,
        );
        let experts = self.switch_mlp.forward_with_expert_scales(
            &flat,
            &indices,
            self.gate_scale.as_deref(),
            self.out_scale.as_deref(),
        );
        let routed = moe_weighted_sum(&experts, &routed_weights, mlxcel_core::array_dtype(x));

        let shared_gate = self.shared_gate.forward(&flat);
        let shared_up = self.shared_up.forward(&flat);
        let shared = mlxcel_core::compiled_swiglu_activation(&shared_gate, &shared_up);
        let shared_weights =
            mlxcel_core::astype(&shared_weights, mlxcel_core::array_dtype(&shared));
        let gamma = mlxcel_core::repeat(&shared_weights, self.intermediate, -1);
        let shared = mlxcel_core::multiply(&shared, &gamma);
        let shared = self.shared_down.forward(&shared);
        let output = mlxcel_core::add(&routed, &shared);
        let output = mlxcel_core::astype(&output, mlxcel_core::array_dtype(x));
        mlxcel_core::reshape(&output, &[shape[0], shape[1], hidden])
    }
}

pub(crate) fn route_weights(
    logits: &MlxArray,
    correction_bias: &MlxArray,
    global_scale: &MlxArray,
    n_routed: i32,
    n_shared: i32,
    top_k: i32,
    route_scale: f32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let routed_logits = slice_axis(logits, -1, 0, n_routed);
    let routed_scores = mlxcel_core::sigmoid(&mlxcel_core::astype(&routed_logits, dtype::FLOAT32));
    let selection_scores = mlxcel_core::add(&routed_scores, correction_bias);
    let partition =
        mlxcel_core::argpartition(&mlxcel_core::negative(&selection_scores), top_k - 1, -1);
    let indices = slice_axis(&partition, -1, 0, top_k);
    let selected_logits = mlxcel_core::take_along_axis(&routed_logits, &indices, -1);
    let shared_logits = slice_axis(logits, -1, n_routed, n_routed + n_shared);
    let combined = mlxcel_core::concatenate(&selected_logits, &shared_logits, -1);
    let combined = mlxcel_core::astype(&combined, dtype::FLOAT32);
    let log_prob = mlxcel_core::negative(&mlxcel_core::logaddexp(
        &mlxcel_core::zeros_like(&combined),
        &mlxcel_core::negative(&combined),
    ));
    let normalizer = mlxcel_core::logsumexp_axis(&log_prob, -1, true);
    let weights = mlxcel_core::exp(&mlxcel_core::subtract(&log_prob, &normalizer));
    let weights = mlxcel_core::multiply_scalar(&weights, route_scale);
    let global = mlxcel_core::astype(global_scale, dtype::FLOAT32);
    let weights = mlxcel_core::multiply(&weights, &global);
    let routed = slice_axis(&weights, -1, 0, top_k);
    let shared = slice_axis(&weights, -1, top_k, top_k + n_shared);
    (mlxcel_core::astype(&indices, dtype::UINT32), routed, shared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn concatenated_shared_experts_equal_explicit_expert_sum() {
        const TOKENS: usize = 2;
        const SHARED: usize = 2;
        const WIDTH: usize = 2;
        const HIDDEN: usize = 3;
        let x_values = [0.5_f32, -1.0, 1.5, 2.0, 0.25, -0.75];
        let gate_values: Vec<f32> = (0..SHARED * WIDTH * HIDDEN)
            .map(|i| (i as f32 - 4.0) / 11.0)
            .collect();
        let up_values: Vec<f32> = (0..SHARED * WIDTH * HIDDEN)
            .map(|i| (i as f32 + 2.0) / 13.0)
            .collect();
        let down_values: Vec<f32> = (0..HIDDEN * SHARED * WIDTH)
            .map(|i| (i as f32 - 3.0) / 17.0)
            .collect();
        let weight_values = [0.2_f32, 0.8, 0.65, 0.35];

        let x = mlxcel_core::from_slice_f32(&x_values, &[TOKENS as i32, HIDDEN as i32]);
        let gate =
            mlxcel_core::from_slice_f32(&gate_values, &[(SHARED * WIDTH) as i32, HIDDEN as i32]);
        let up = mlxcel_core::from_slice_f32(&up_values, &[(SHARED * WIDTH) as i32, HIDDEN as i32]);
        let down =
            mlxcel_core::from_slice_f32(&down_values, &[HIDDEN as i32, (SHARED * WIDTH) as i32]);
        let weights = mlxcel_core::from_slice_f32(&weight_values, &[TOKENS as i32, SHARED as i32]);
        let gate_out = mlxcel_core::matmul(&x, &mlxcel_core::transpose(&gate));
        let up_out = mlxcel_core::matmul(&x, &mlxcel_core::transpose(&up));
        let activated = mlxcel_core::compiled_swiglu_activation(&gate_out, &up_out);
        let gamma = mlxcel_core::repeat(&weights, WIDTH as i32, -1);
        let actual = mlxcel_core::matmul(
            &mlxcel_core::multiply(&activated, &gamma),
            &mlxcel_core::transpose(&down),
        );

        let mut expected = vec![0.0_f32; TOKENS * HIDDEN];
        for token in 0..TOKENS {
            for shared in 0..SHARED {
                for intermediate in 0..WIDTH {
                    let row = shared * WIDTH + intermediate;
                    let mut gate_value = 0.0;
                    let mut up_value = 0.0;
                    for hidden in 0..HIDDEN {
                        let input = x_values[token * HIDDEN + hidden];
                        gate_value += gate_values[row * HIDDEN + hidden] * input;
                        up_value += up_values[row * HIDDEN + hidden] * input;
                    }
                    let activation = gate_value / (1.0 + (-gate_value).exp()) * up_value;
                    for hidden in 0..HIDDEN {
                        expected[token * HIDDEN + hidden] += weight_values[token * SHARED + shared]
                            * down_values[hidden * SHARED * WIDTH + row]
                            * activation;
                    }
                }
            }
        }
        let expected = mlxcel_core::from_slice_f32(&expected, &[TOKENS as i32, HIDDEN as i32]);
        assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
            &actual, &expected, 1e-5, 1e-5
        )));
    }
}
