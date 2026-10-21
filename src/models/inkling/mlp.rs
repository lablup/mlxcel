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

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::{InklingConfig, weight};
use crate::models::switch_layers::{
    QuantizedSwitchLinearRef, SwitchGLU, fused_moe_enabled, moe_weighted_sum,
};

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
}

impl mlxcel_core::inkling_layer::InklingFeedForward for InklingMlp {
    fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(input),
            Self::Sparse(moe) => moe.forward(input),
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
    /// route_scale * global_scale in float32: the routing-scale product the
    /// MoE route kernel reads from one buffer (issue #2479).
    wscale: UniquePtr<MlxArray>,
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

/// Whether the custom Inkling MoE Metal kernels can run: the resolved GPU
/// backend is Metal and `MLXCEL_INKLING_MOE_KERNEL=0` is not set. The kernels
/// replace the ~4,600-small-kernel ops chain that binds batch-1 decode
/// (issue #2479).
pub(crate) fn inkling_moe_kernels_available() -> bool {
    mlxcel_core::metal_is_available()
        && std::env::var("MLXCEL_INKLING_MOE_KERNEL").ok().as_deref() != Some("0")
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
        let global_scale = weight(weights, &format!("{prefix}.global_scale"))?;
        let wscale = mlxcel_core::multiply_scalar(
            &mlxcel_core::astype(&global_scale, dtype::FLOAT32),
            text.route_scale,
        );
        Ok(Self {
            gate_weight: weight(weights, &format!("{prefix}.gate_weight"))?,
            correction_bias: mlxcel_core::astype(&correction_bias, dtype::FLOAT32),
            global_scale,
            wscale,
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

    /// Quantized parts of the down projection when the routed down-combine
    /// kernel applies to this block: batch<=8 decode, affine q4/g64 experts
    /// with zero-point biases, and the down projection's
    /// one-64-wide-group-per-lane shape (in_dims 2048) with scales in the
    /// activation dtype (the mixed-dtype gather cliff).
    fn down_combine_parts(
        &self,
        x: &MlxArray,
        n_tokens: i32,
    ) -> Option<QuantizedSwitchLinearRef<'_>> {
        if n_tokens > 8 || self.top_k > 8 {
            return None;
        }
        let down = self.switch_mlp.down_quantized_parts()?;
        if down.bits != 4 || down.group_size != 64 || down.mode != "affine" {
            return None;
        }
        let wshape = mlxcel_core::array_shape(down.weight.as_ref().unwrap());
        if wshape.len() != 3 || wshape[2] * 8 != 2048 {
            return None;
        }
        if mlxcel_core::array_dtype(down.scales.as_ref().unwrap()) != mlxcel_core::array_dtype(x) {
            return None;
        }
        Some(down)
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let hidden = shape[2];
        let flat = mlxcel_core::reshape(x, &[-1, hidden]);
        let logits = mlxcel_core::matmul(&flat, &mlxcel_core::transpose(&self.gate_weight));

        // Fast path (issue #2479): the whole post-matmul routing chain is one
        // Metal kernel; at batch<=8 the routed down projection is one
        // dequant+combine dispatch. The ops chain runs ~4,600 small kernels
        // per batch-1 decode token and is GPU-bound. The launcher refuses
        // when the resolved backend has no port; the gate below pins Metal,
        // so an `Err` would mean the two disagree, and the fallback is the
        // graph routing chain rather than a null output.
        let route_kernel = inkling_moe_kernels_available();
        let fast_route = if route_kernel {
            let mut idx = mlxcel_core::UniquePtr::null();
            let mut wk = mlxcel_core::UniquePtr::null();
            let mut g = mlxcel_core::UniquePtr::null();
            mlxcel_core::inkling_moe_route_kernel(
                &logits,
                &self.correction_bias,
                &self.wscale,
                self.n_routed,
                self.n_shared,
                self.top_k,
                self.intermediate,
                &mut idx,
                &mut wk,
                &mut g,
            )
            .is_ok()
            .then_some((idx, wk, g))
        } else {
            None
        };
        let (indices, routed_weights, shared_weights, gamma) = match fast_route {
            Some((idx, wk, g)) => (idx, wk, None, Some(g)),
            None => {
                let (i, r, s) = route_weights(
                    &logits,
                    &self.correction_bias,
                    &self.global_scale,
                    self.n_routed,
                    self.n_shared,
                    self.top_k,
                    self.route_scale,
                );
                (i, r, Some(s), None)
            }
        };

        let n_tokens = mlxcel_core::array_shape(&flat)[0];
        // NVFP4 experts and any plane carrying ModelOpt gate/out scale
        // sidecars stay on the gather path; their per-expert scales have no
        // slot in either fused kernel.
        let scaled = self.gate_scale.is_some() || self.out_scale.is_some();
        let down_parts = if route_kernel && !scaled {
            self.down_combine_parts(x, n_tokens)
        } else {
            None
        };
        let down_combine = down_parts.is_some();
        // Single-token decode tries the fused affine kernel (issue #1859)
        // when the down-combine kernel does not apply.
        let fused = if !down_combine && !scaled && n_tokens == 1 && fused_moe_enabled() {
            self.switch_mlp
                .forward_fused_kernel(&flat, &indices, &routed_weights)
                .map(|out| mlxcel_core::reshape(&out, &[1, hidden]))
        } else {
            None
        };
        let routed = match fused {
            Some(out) => out,
            None => {
                let down_combined = match &down_parts {
                    Some(down) => {
                        let act = self.switch_mlp.forward_activated(&flat, &indices);
                        let act_flat = mlxcel_core::reshape(&act, &[n_tokens * self.top_k, -1]);
                        let mut out = mlxcel_core::UniquePtr::null();
                        // The launcher refuses when the resolved backend has
                        // no port; the gate above pins Metal, so an `Err`
                        // would mean the two disagree. Fall back to the
                        // expert-scales gather path rather than reading a
                        // null output.
                        // SAFETY: all inputs outlive the call; the kernel
                        // writes `out` and evaluates no graph.
                        unsafe {
                            mlxcel_core::inkling_moe_down_combine_kernel(
                                &act_flat,
                                down.weight.as_ref().unwrap(),
                                down.scales.as_ref().unwrap(),
                                down.biases.as_ref().unwrap(),
                                &indices,
                                &routed_weights,
                                self.top_k,
                                &mut out,
                            )
                        }
                        .is_ok()
                        .then(|| mlxcel_core::reshape(&out, &[n_tokens, hidden]))
                    }
                    None => None,
                };
                match down_combined {
                    Some(out) => out,
                    None => {
                        let experts = self.switch_mlp.forward_with_expert_scales(
                            &flat,
                            &indices,
                            self.gate_scale.as_deref(),
                            self.out_scale.as_deref(),
                        );
                        moe_weighted_sum(&experts, &routed_weights, mlxcel_core::array_dtype(x))
                    }
                }
            }
        };

        let shared_gate = self.shared_gate.forward(&flat);
        let shared_up = self.shared_up.forward(&flat);
        let shared = mlxcel_core::compiled_swiglu_activation(&shared_gate, &shared_up);
        let shared = match gamma {
            // The route kernel emits the shared-expert weights pre-broadcast
            // over the expert-major intermediate.
            Some(g) => mlxcel_core::multiply(&shared, &g),
            None => {
                let shared_weights = mlxcel_core::astype(
                    &shared_weights.unwrap(),
                    mlxcel_core::array_dtype(&shared),
                );
                let gamma = mlxcel_core::repeat(&shared_weights, self.intermediate, -1);
                mlxcel_core::multiply(&shared, &gamma)
            }
        };
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
    use super::*;
    use crate::models::switch_layers::insert_honest_affine_swiglu_experts;
    use serde_json::json;

    const FUSED_HIDDEN: i32 = 64;
    const FUSED_DFF: i32 = 64;
    const FUSED_ROUTED: i32 = 2;
    const FUSED_SHARED: i32 = 1;
    const FUSED_TOPK: i32 = 2;

    fn values(n: usize, start: f32, step: f32) -> Vec<f32> {
        (0..n)
            .map(|i| start + ((i * 7 + 3) % 19) as f32 * step)
            .collect()
    }

    fn honest_affine_sparse_moe() -> InklingSparseMoe {
        let config: InklingConfig = serde_json::from_value(json!({
            "quantization": {"group_size": 64, "bits": 4},
            "text_config": {
                "hidden_size": FUSED_HIDDEN,
                "num_hidden_layers": 1,
                "dense_mlp_idx": 0,
                "mlp_layer_types": ["sparse"],
                "intermediate_size": FUSED_DFF,
                "dense_intermediate_size": FUSED_DFF,
                "n_routed_experts": FUSED_ROUTED,
                "num_experts_per_tok": FUSED_TOPK,
                "n_shared_experts": FUSED_SHARED,
                "route_scale": 8.0
            }
        }))
        .expect("config parses");

        let mut weights = WeightMap::new();
        let prefix = "model.layers.0.mlp";
        insert_honest_affine_swiglu_experts(
            &mut weights,
            &format!("{prefix}.switch_mlp"),
            FUSED_ROUTED,
            FUSED_HIDDEN,
            FUSED_DFF,
        );
        let total_experts = FUSED_ROUTED + FUSED_SHARED;
        weights.insert(
            format!("{prefix}.gate_weight"),
            mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(
                    &values((total_experts * FUSED_HIDDEN) as usize, -0.2, 0.025),
                    &[total_experts, FUSED_HIDDEN],
                ),
                dtype::BFLOAT16,
            ),
        );
        weights.insert(
            format!("{prefix}.e_score_correction_bias"),
            mlxcel_core::from_slice_f32(&[0.1, -0.05], &[FUSED_ROUTED]),
        );
        weights.insert(
            format!("{prefix}.global_scale"),
            mlxcel_core::from_slice_f32(&[1.25], &[1]),
        );
        for (leaf, rows, cols, seed) in [
            ("gate_proj", FUSED_SHARED * FUSED_DFF, FUSED_HIDDEN, 1usize),
            ("up_proj", FUSED_SHARED * FUSED_DFF, FUSED_HIDDEN, 2),
            ("down_proj", FUSED_HIDDEN, FUSED_SHARED * FUSED_DFF, 3),
        ] {
            weights.insert(
                format!("{prefix}.shared_experts.{leaf}.weight"),
                mlxcel_core::astype(
                    &mlxcel_core::from_slice_f32(
                        &values((rows * cols) as usize, -0.15 * seed as f32, 0.01),
                        &[rows, cols],
                    ),
                    dtype::BFLOAT16,
                ),
            );
        }
        InklingSparseMoe::from_weights(&weights, &config, 0).expect("sparse MoE loads")
    }

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

    /// Issue #1859: `InklingSparseMoe::forward` never called
    /// `SwitchGLU::forward_fused_kernel`, so affine checkpoints like
    /// `inkling-small-mlx-4bit` (`gate_scale: None, out_scale: None`) always
    /// took the `forward_with_expert_scales` + `moe_weighted_sum` gather path
    /// at single-token decode. When the fused MoE ports are available and
    /// `MLXCEL_FUSED_MOE` is enabled, a single-token forward on an unscaled
    /// affine block must route through `forward_fused_kernel` and match its
    /// output bit-for-bit, while agreeing with the gather fallback within
    /// quantization tolerance.
    #[test]
    fn inkling_sparse_moe_routes_one_token_affine_through_fused_kernel() {
        if !mlxcel_core::fused_moe_kernels_available() {
            eprintln!("skipping: this backend has no fused MoE kernel port (#1803)");
            return;
        }
        if !fused_moe_enabled() {
            eprintln!("skipping: MLXCEL_FUSED_MOE disables the routing this test pins");
            return;
        }

        let moe = honest_affine_sparse_moe();
        let x = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(
                &values(FUSED_HIDDEN as usize, -0.4, 0.0125),
                &[1, 1, FUSED_HIDDEN],
            ),
            dtype::BFLOAT16,
        );
        let flat = mlxcel_core::reshape(&x, &[1, FUSED_HIDDEN]);
        let logits = mlxcel_core::matmul(&flat, &mlxcel_core::transpose(&moe.gate_weight));
        let (indices, routed_weights, shared_weights) = route_weights(
            &logits,
            &moe.correction_bias,
            &moe.global_scale,
            moe.n_routed,
            moe.n_shared,
            moe.top_k,
            moe.route_scale,
        );

        let fused_routed = moe
            .switch_mlp
            .forward_fused_kernel(&flat, &indices, &routed_weights)
            .expect("honest affine 4-bit experts of dff 64 must dispatch the fused kernel");
        let fused_routed = mlxcel_core::reshape(&fused_routed, &[1, FUSED_HIDDEN]);
        let gather_routed = moe_weighted_sum(
            &moe.switch_mlp.forward(&flat, &indices),
            &routed_weights,
            dtype::BFLOAT16,
        );

        let shared_gate = moe.shared_gate.forward(&flat);
        let shared_up = moe.shared_up.forward(&flat);
        let shared = mlxcel_core::compiled_swiglu_activation(&shared_gate, &shared_up);
        let shared_weights =
            mlxcel_core::astype(&shared_weights, mlxcel_core::array_dtype(&shared));
        let gamma = mlxcel_core::repeat(&shared_weights, moe.intermediate, -1);
        let shared = mlxcel_core::multiply(&shared, &gamma);
        let shared = moe.shared_down.forward(&shared);

        let fused_total = mlxcel_core::reshape(
            &mlxcel_core::astype(&mlxcel_core::add(&fused_routed, &shared), dtype::BFLOAT16),
            &[1, 1, FUSED_HIDDEN],
        );
        let gather_total = mlxcel_core::reshape(
            &mlxcel_core::astype(&mlxcel_core::add(&gather_routed, &shared), dtype::BFLOAT16),
            &[1, 1, FUSED_HIDDEN],
        );

        let actual = moe.forward(&x);
        assert!(
            mlxcel_core::item_bool(&mlxcel_core::array_equal(&actual, &fused_total, false)),
            "one-token affine InklingSparseMoe::forward must take the fused kernel path"
        );

        let actual_f32 =
            mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(&actual, dtype::FLOAT32));
        let gather_f32 = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
            &gather_total,
            dtype::FLOAT32,
        ));
        let diff_sq: f64 = actual_f32
            .iter()
            .zip(&gather_f32)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let ref_sq: f64 = gather_f32.iter().map(|b| f64::from(*b).powi(2)).sum();
        let nrms = (diff_sq / ref_sq.max(1e-20)).sqrt();
        assert!(
            actual_f32.iter().all(|v| v.is_finite()) && nrms < 0.1,
            "fused InklingSparseMoe output must agree with gather reference: nrms {nrms:e}"
        );
    }

    /// Issue #2479: the MoE route kernel must reproduce the graph routing
    /// chain — identical top-k indices, routing weights within one storage-dtype
    /// rounding, and the shared-expert gamma matching the graph's astype +
    /// repeat broadcast.
    #[test]
    fn inkling_moe_route_kernel_matches_graph_routing() {
        if !mlxcel_core::metal_is_available() {
            eprintln!("skipping: no Metal backend for the MoE route kernel (#2479)");
            return;
        }
        let moe = honest_affine_sparse_moe();
        let flat = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(
                &values(FUSED_HIDDEN as usize, -0.4, 0.0125),
                &[1, FUSED_HIDDEN],
            ),
            dtype::BFLOAT16,
        );
        let logits = mlxcel_core::matmul(&flat, &mlxcel_core::transpose(&moe.gate_weight));
        let (g_indices, g_routed, g_shared) = route_weights(
            &logits,
            &moe.correction_bias,
            &moe.global_scale,
            moe.n_routed,
            moe.n_shared,
            moe.top_k,
            moe.route_scale,
        );
        let mut k_indices = UniquePtr::<MlxArray>::null();
        let mut k_routed = UniquePtr::<MlxArray>::null();
        let mut k_gamma = UniquePtr::<MlxArray>::null();
        mlxcel_core::inkling_moe_route_kernel(
            &logits,
            &moe.correction_bias,
            &moe.wscale,
            moe.n_routed,
            moe.n_shared,
            moe.top_k,
            moe.intermediate,
            &mut k_indices,
            &mut k_routed,
            &mut k_gamma,
        )
        .expect("route kernel launch must succeed once metal_is_available() gates this test");
        assert!(
            mlxcel_core::item_bool(&mlxcel_core::array_equal(&k_indices, &g_indices, false)),
            "route kernel top-k indices must match argpartition"
        );
        let g_routed_f32 =
            mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(&g_routed, dtype::FLOAT32));
        let k_routed_f32 =
            mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(&k_routed, dtype::FLOAT32));
        for (g, k) in g_routed_f32.iter().zip(&k_routed_f32) {
            let denom = g.abs().max(1e-6);
            assert!(
                (g - k).abs() / denom < 0.02,
                "route kernel weights must match the graph chain: {g} vs {k}"
            );
        }
        let g_gamma = mlxcel_core::repeat(
            &mlxcel_core::astype(&g_shared, mlxcel_core::array_dtype(&k_gamma)),
            moe.intermediate,
            -1,
        );
        assert!(
            mlxcel_core::item_bool(&mlxcel_core::array_equal(&k_gamma, &g_gamma, false)),
            "route kernel gamma must match the graph astype + repeat broadcast"
        );
    }

    /// Issue #2479: a one-token forward on an affine q4/g64 block with the
    /// down projection's 2048-wide intermediate must dispatch the
    /// dequant+combine kernel and agree with the unfused gather chain within
    /// quantization tolerance.
    #[test]
    fn inkling_sparse_moe_routes_one_token_affine_through_down_combine_kernel() {
        if !inkling_moe_kernels_available() {
            eprintln!("skipping: this backend has no Inkling MoE kernel port (#2479)");
            return;
        }
        let hidden = 64_i32;
        let dff = 2048_i32;
        let config: InklingConfig = serde_json::from_value(json!({
            "quantization": {"group_size": 64, "bits": 4},
            "text_config": {
                "hidden_size": hidden,
                "num_hidden_layers": 1,
                "dense_mlp_idx": 0,
                "mlp_layer_types": ["sparse"],
                "intermediate_size": dff,
                "dense_intermediate_size": dff,
                "n_routed_experts": FUSED_ROUTED,
                "num_experts_per_tok": FUSED_TOPK,
                "n_shared_experts": FUSED_SHARED,
                "route_scale": 8.0
            }
        }))
        .expect("config parses");
        let mut weights = WeightMap::new();
        let prefix = "model.layers.0.mlp";
        insert_honest_affine_swiglu_experts(
            &mut weights,
            &format!("{prefix}.switch_mlp"),
            FUSED_ROUTED,
            hidden,
            dff,
        );
        let total_experts = FUSED_ROUTED + FUSED_SHARED;
        weights.insert(
            format!("{prefix}.gate_weight"),
            mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(
                    &values((total_experts * hidden) as usize, -0.2, 0.025),
                    &[total_experts, hidden],
                ),
                dtype::BFLOAT16,
            ),
        );
        weights.insert(
            format!("{prefix}.e_score_correction_bias"),
            mlxcel_core::from_slice_f32(&[0.1, -0.05], &[FUSED_ROUTED]),
        );
        weights.insert(
            format!("{prefix}.global_scale"),
            mlxcel_core::from_slice_f32(&[1.25], &[1]),
        );
        for (leaf, rows, cols, seed) in [
            ("gate_proj", FUSED_SHARED * dff, hidden, 1usize),
            ("up_proj", FUSED_SHARED * dff, hidden, 2),
            ("down_proj", hidden, FUSED_SHARED * dff, 3),
        ] {
            weights.insert(
                format!("{prefix}.shared_experts.{leaf}.weight"),
                mlxcel_core::astype(
                    &mlxcel_core::from_slice_f32(
                        &values((rows * cols) as usize, -0.15 * seed as f32, 0.01),
                        &[rows, cols],
                    ),
                    dtype::BFLOAT16,
                ),
            );
        }
        let moe = InklingSparseMoe::from_weights(&weights, &config, 0).expect("sparse MoE loads");

        let x = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&values(hidden as usize, -0.4, 0.0125), &[1, 1, hidden]),
            dtype::BFLOAT16,
        );
        let flat = mlxcel_core::reshape(&x, &[1, hidden]);
        let logits = mlxcel_core::matmul(&flat, &mlxcel_core::transpose(&moe.gate_weight));
        let (indices, routed_weights, shared_weights) = route_weights(
            &logits,
            &moe.correction_bias,
            &moe.global_scale,
            moe.n_routed,
            moe.n_shared,
            moe.top_k,
            moe.route_scale,
        );
        assert!(
            moe.down_combine_parts(&x, 1).is_some(),
            "affine q4/g64 experts of dff 2048 must pass the down-combine gate"
        );
        let gather_routed = moe_weighted_sum(
            &moe.switch_mlp.forward(&flat, &indices),
            &routed_weights,
            dtype::BFLOAT16,
        );
        let shared_gate = moe.shared_gate.forward(&flat);
        let shared_up = moe.shared_up.forward(&flat);
        let shared = mlxcel_core::compiled_swiglu_activation(&shared_gate, &shared_up);
        let shared_weights =
            mlxcel_core::astype(&shared_weights, mlxcel_core::array_dtype(&shared));
        let gamma = mlxcel_core::repeat(&shared_weights, moe.intermediate, -1);
        let shared = mlxcel_core::multiply(&shared, &gamma);
        let shared = moe.shared_down.forward(&shared);
        let expected = mlxcel_core::reshape(
            &mlxcel_core::astype(&mlxcel_core::add(&gather_routed, &shared), dtype::BFLOAT16),
            &[1, 1, hidden],
        );

        let actual = moe.forward(&x);
        let actual_f32 =
            mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(&actual, dtype::FLOAT32));
        let expected_f32 =
            mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(&expected, dtype::FLOAT32));
        let diff_sq: f64 = actual_f32
            .iter()
            .zip(&expected_f32)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let ref_sq: f64 = expected_f32.iter().map(|b| f64::from(*b).powi(2)).sum();
        let nrms = (diff_sq / ref_sq.max(1e-20)).sqrt();
        assert!(
            actual_f32.iter().all(|v| v.is_finite()) && nrms < 0.1,
            "down-combine output must agree with the gather reference: nrms {nrms:e}"
        );
    }

    /// Issue #1859 guard: if `gate_scale` or `out_scale` is present on an
    /// `InklingSparseMoe` block, single-token `forward` must NOT bypass those
    /// per-expert sidecar scales via `forward_fused_kernel`.
    #[test]
    fn inkling_sparse_moe_with_expert_scales_stays_on_scaled_gather_path() {
        let mut moe = honest_affine_sparse_moe();
        let gate_scale = mlxcel_core::from_slice_f32(&[2.5, 0.25], &[FUSED_ROUTED]);
        let out_scale = mlxcel_core::from_slice_f32(&[3.0, 0.5], &[FUSED_ROUTED]);
        moe.gate_scale = Some(mlxcel_core::copy(&gate_scale));
        moe.out_scale = Some(mlxcel_core::copy(&out_scale));

        let x = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(
                &values(FUSED_HIDDEN as usize, -0.4, 0.0125),
                &[1, 1, FUSED_HIDDEN],
            ),
            dtype::BFLOAT16,
        );
        let flat = mlxcel_core::reshape(&x, &[1, FUSED_HIDDEN]);
        let logits = mlxcel_core::matmul(&flat, &mlxcel_core::transpose(&moe.gate_weight));
        let (indices, routed_weights, shared_weights) = route_weights(
            &logits,
            &moe.correction_bias,
            &moe.global_scale,
            moe.n_routed,
            moe.n_shared,
            moe.top_k,
            moe.route_scale,
        );
        let experts = moe.switch_mlp.forward_with_expert_scales(
            &flat,
            &indices,
            Some(&gate_scale),
            Some(&out_scale),
        );
        let routed = moe_weighted_sum(&experts, &routed_weights, dtype::BFLOAT16);
        let shared_gate = moe.shared_gate.forward(&flat);
        let shared_up = moe.shared_up.forward(&flat);
        let shared = mlxcel_core::compiled_swiglu_activation(&shared_gate, &shared_up);
        let shared_weights =
            mlxcel_core::astype(&shared_weights, mlxcel_core::array_dtype(&shared));
        let gamma = mlxcel_core::repeat(&shared_weights, moe.intermediate, -1);
        let shared = mlxcel_core::multiply(&shared, &gamma);
        let shared = moe.shared_down.forward(&shared);
        let expected = mlxcel_core::reshape(
            &mlxcel_core::astype(&mlxcel_core::add(&routed, &shared), dtype::BFLOAT16),
            &[1, 1, FUSED_HIDDEN],
        );

        let actual = moe.forward(&x);
        assert!(
            mlxcel_core::item_bool(&mlxcel_core::array_equal(&actual, &expected, false)),
            "scaled InklingSparseMoe must stay on forward_with_expert_scales at one token"
        );
    }
}
