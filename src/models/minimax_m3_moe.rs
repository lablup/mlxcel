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

//! MiniMax-M3 sparse MoE block (`block_sparse_moe`): a sigmoid router with a
//! selection-only routing bias, `num_experts_per_tok` routed experts through a
//! clamp-SwiGLU `SwitchGLU`, plus one always-on shared expert.
//!
//! The real checkpoint stores experts under the Mixtral convention
//! (`block_sparse_moe.experts.{i}.w1/w2/w3.weight`, w1=gate_proj, w2=down_proj,
//! w3=up_proj) and the shared expert as a SEPARATE MLP
//! (`block_sparse_moe.shared_experts.{gate_proj,up_proj,down_proj}.weight`),
//! never packed into the switch tensors. The shared expert is loaded when its
//! tensors are present; otherwise the block is routed-only.

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
    (mlxcel_core::astype(&idx, mlxcel_core::dtype::INT32), topk)
}

/// Pick the expert projection leaf names for `{moe_prefix}.experts.{i}.*`.
///
/// The real MiniMax-M3 checkpoint uses the Mixtral convention
/// (`w1`=gate_proj, `w3`=up_proj, `w2`=down_proj); community conversions may use
/// the plain `gate_proj/up_proj/down_proj` names (individual or pre-stacked).
/// The returned triple is `[gate, up, down]`.
fn expert_proj_names(weights: &WeightMap, moe_prefix: &str) -> [&'static str; 3] {
    if weights.contains_key(&format!("{}.experts.0.w1.weight", moe_prefix)) {
        ["w1", "w3", "w2"]
    } else {
        ["gate_proj", "up_proj", "down_proj"]
    }
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

    /// Load routed experts from `{moe_prefix}.switch_mlp.{gate,up,down}`, which
    /// the shared loader resolves to either a pre-stacked tensor or per-expert
    /// `{moe_prefix}.experts.{i}.{leaf}` tensors. `proj_names` is `[gate, up,
    /// down]` (e.g. Mixtral `["w1", "w3", "w2"]`).
    fn from_weights(
        weights: &WeightMap,
        moe_prefix: &str,
        group_size: i32,
        bits: i32,
        alpha: f32,
        limit: f32,
        proj_names: [&str; 3],
    ) -> Result<Self, String> {
        let [gate, up, down] = proj_names;
        let switch_prefix = format!("{}.switch_mlp", moe_prefix);
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.{}", switch_prefix, gate),
                group_size,
                bits,
            )?,
            up_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.{}", switch_prefix, up),
                group_size,
                bits,
            )?,
            down_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.{}", switch_prefix, down),
                group_size,
                bits,
            )?,
            alpha,
            limit,
        })
    }
}

pub(super) struct MoeBlock {
    router: UnifiedLinear,
    bias: UniquePtr<MlxArray>,
    experts: SwitchGluOai,
    /// Separate always-on shared expert; `None` when the block is routed-only.
    shared: Option<DenseMlp>,
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
        let expert_out = self.experts.forward(&x_flat, &idx);
        let mut result = moe_weighted_sum(&expert_out, &scores, out_dtype);
        if let Some(shared) = &self.shared {
            let shared_out = shared.forward(&x_flat);
            result = mlxcel_core::add(&result, &shared_out);
        }

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    /// Load the MoE block at `moe_prefix` (e.g.
    /// `model.layers.{i}.block_sparse_moe`).
    pub(super) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        moe_prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate", moe_prefix),
            group_size,
            args.gate_bits(),
        )?;

        let bias = if args.use_routing_bias {
            weights
                .get(&format!("{}.e_score_correction_bias", moe_prefix))
                .or_else(|| weights.get(&format!("{}.gate.e_score_correction_bias", moe_prefix)))
                .map(|w| mlxcel_core::copy(w))
                .unwrap_or_else(|| zeros_bias(args.num_local_experts))
        } else {
            zeros_bias(args.num_local_experts)
        };

        let proj_names = expert_proj_names(weights, moe_prefix);
        let experts = SwitchGluOai::from_weights(
            weights,
            moe_prefix,
            group_size,
            bits,
            args.swiglu_alpha,
            args.swiglu_limit,
            proj_names,
        )?;

        // The shared expert is a separate MLP (never packed into the switch
        // tensors). Load it when its tensors are present.
        let shared_prefix = format!("{}.shared_experts", moe_prefix);
        let has_shared = weights.contains_key(&format!("{}.gate_proj.weight", shared_prefix));
        let shared = if args.n_shared_experts > 0 && has_shared {
            Some(DenseMlp::from_weights(weights, args, &shared_prefix)?)
        } else {
            None
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

fn zeros_bias(num_experts: usize) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[num_experts as i32], 0.0, mlxcel_core::dtype::FLOAT32)
}
