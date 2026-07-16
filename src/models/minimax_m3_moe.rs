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

//! MiniMax-M3 sparse MoE block: sigmoid router with a selection-only routing
//! bias, `num_experts_per_tok` routed experts through a clamp-SwiGLU
//! `SwitchGLU`, plus one shared expert.
//!
//! The shared expert is packed as switch-tensor index `num_local_experts` and
//! participates with a fixed score of `1.0` when its width equals the routed
//! expert width (`shared_intermediate_size == intermediate_size`); otherwise it
//! is a separate clamp-SwiGLU MLP added to the routed mixture.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::ModelArgs;
use super::layers::{DenseMlp, swigluoai};
use crate::models::switch_layers::{SwitchLinear, moe_weighted_sum};

/// Top-`k` expert routing with a selection-only bias.
///
/// `logits` is `[n, num_routed_experts]` (float32). The bias is added only for
/// the top-k *selection*; the returned mixture weights are the UNBIASED sigmoid
/// scores of the selected experts, normalized to sum 1 (when `norm_topk`) and
/// then multiplied by `routed_scaling_factor`. Returns `(indices [n, k] int32,
/// scores [n, k])`.
pub(super) fn route(
    logits: &MlxArray,
    bias: &MlxArray,
    k: i32,
    norm_topk: bool,
    routed_scaling_factor: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let scores = mlxcel_core::sigmoid(logits);
    let orig_scores = mlxcel_core::copy(&scores);
    let biased = mlxcel_core::add(&scores, bias);

    let neg = mlxcel_core::negative(&biased);
    let part = mlxcel_core::argpartition(&neg, k - 1, -1);
    let idx = slice_axis(&part, -1, 0, k);

    let topk = mlxcel_core::take_along_axis(&orig_scores, &idx, -1);
    let topk = if norm_topk && k > 1 {
        let sum = mlxcel_core::sum_axis(&topk, -1, true);
        let eps = mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::array_dtype(&topk));
        let sum = mlxcel_core::add(&sum, &eps);
        mlxcel_core::divide(&topk, &sum)
    } else {
        topk
    };
    let scale = mlxcel_core::full_f32(&[1], routed_scaling_factor, mlxcel_core::array_dtype(&topk));
    let topk = mlxcel_core::multiply(&topk, &scale);
    (idx, topk)
}

/// `SwitchGLU` over pre-stacked experts with the clamp-SwiGLU activation.
/// Mirrors the shared `switch_layers::SwitchGLU` non-sorted gather path but with
/// the `swigluoai` activation in place of the SiLU SwiGLU.
struct SwitchGluOai {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
    alpha: f32,
    limit: f32,
}

impl SwitchGluOai {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let x_exp = mlxcel_core::expand_dims(x, -2);
        let x_exp = mlxcel_core::expand_dims(&x_exp, -3);
        let x_gate = self.gate_proj.forward(&x_exp, indices, false);
        let x_up = self.up_proj.forward(&x_exp, indices, false);
        // gate is the gated ("glu") branch, up is the linear branch.
        let activated = swigluoai(&x_up, &x_gate, self.alpha, self.limit);
        let output = self.down_proj.forward(&activated, indices, false);
        mlxcel_core::squeeze_axis(&output, -2)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        alpha: f32,
        limit: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
            )?,
            alpha,
            limit,
        })
    }
}

/// How the single shared expert is realized.
enum SharedExpert {
    /// Packed into the switch tensors at `index`; participates with score 1.0.
    Packed { index: i32 },
    /// Separate clamp-SwiGLU MLP added to the routed mixture.
    Separate(DenseMlp),
}

/// Append the packed shared expert (`index`, score 1.0) to each token's routed
/// selection. `indices`/`scores` are `[n, k]`; returns `[n, k+1]` each.
pub(super) fn append_shared_expert(
    indices: &MlxArray,
    scores: &MlxArray,
    index: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let n = mlxcel_core::array_shape(indices)[0];
    let shared_idx = mlxcel_core::from_slice_i32(&vec![index; n as usize], &[n, 1]);
    let indices = mlxcel_core::astype(indices, mlxcel_core::dtype::INT32);
    let idx_full = mlxcel_core::concatenate(&indices, &shared_idx, -1);

    let shared_score = mlxcel_core::ones(&[n, 1], mlxcel_core::array_dtype(scores));
    let score_full = mlxcel_core::concatenate(scores, &shared_score, -1);
    (idx_full, score_full)
}

pub(super) struct MoeBlock {
    router: UnifiedLinear,
    bias: UniquePtr<MlxArray>,
    experts: SwitchGluOai,
    shared: SharedExpert,
    num_experts_per_tok: i32,
    norm_topk: bool,
    routed_scaling_factor: f32,
}

impl MoeBlock {
    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = orig_shape[orig_shape.len() - 1];
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden])
        } else {
            mlxcel_core::copy(x)
        };

        let x_f32 = mlxcel_core::astype(&x_flat, mlxcel_core::dtype::FLOAT32);
        let logits = self.router.forward(&x_f32);
        let (idx, scores) = route(
            &logits,
            &self.bias,
            self.num_experts_per_tok,
            self.norm_topk,
            self.routed_scaling_factor,
        );

        let out_dtype = mlxcel_core::array_dtype(&x_flat);
        let result = match &self.shared {
            SharedExpert::Packed { index } => {
                let (idx_full, score_full) = append_shared_expert(&idx, &scores, *index);
                let expert_out = self.experts.forward(&x_flat, &idx_full);
                moe_weighted_sum(&expert_out, &score_full, out_dtype)
            }
            SharedExpert::Separate(mlp) => {
                let idx = mlxcel_core::astype(&idx, mlxcel_core::dtype::INT32);
                let expert_out = self.experts.forward(&x_flat, &idx);
                let routed = moe_weighted_sum(&expert_out, &scores, out_dtype);
                let shared = mlp.forward(&x_flat);
                mlxcel_core::add(&routed, &shared)
            }
        };

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    pub(super) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate", prefix),
            group_size,
            args.gate_bits(),
        )?;

        let bias = if args.use_routing_bias {
            weights
                .get(&format!("{}.gate.e_score_correction_bias", prefix))
                .or_else(|| weights.get(&format!("{}.e_score_correction_bias", prefix)))
                .map(|w| mlxcel_core::copy(w))
                .unwrap_or_else(|| {
                    mlxcel_core::full_f32(
                        &[args.num_local_experts as i32],
                        0.0,
                        mlxcel_core::dtype::FLOAT32,
                    )
                })
        } else {
            mlxcel_core::full_f32(
                &[args.num_local_experts as i32],
                0.0,
                mlxcel_core::dtype::FLOAT32,
            )
        };

        let experts = SwitchGluOai::from_weights(
            weights,
            &format!("{}.switch_mlp", prefix),
            group_size,
            bits,
            args.swiglu_alpha,
            args.swiglu_limit,
        )?;

        // Packed when the shared width equals the routed width (the switch
        // tensors then carry the shared expert as row `num_local_experts`);
        // otherwise a separate MLP under `{prefix}.shared_experts`.
        let shared = if args.shared_expert_is_packed() {
            SharedExpert::Packed {
                index: args.num_local_experts as i32,
            }
        } else {
            SharedExpert::Separate(DenseMlp::from_weights(
                weights,
                args,
                &format!("{}.shared_experts", prefix),
            )?)
        };

        Ok(Self {
            router,
            bias,
            experts,
            shared,
            num_experts_per_tok: args.num_experts_per_tok as i32,
            norm_topk: args.norm_topk_prob,
            routed_scaling_factor: args.routed_scaling_factor,
        })
    }
}
