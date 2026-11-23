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

//! Kolibri-1's sparse block: upstream `MoEGate` + `SwitchGLU` + a SwiGLU
//! shared expert.
//!
//! The router differs from AFMoE's and from the DeepSeek `noaux_tc` gate in
//! one place that a shape check cannot see: the correction bias is added to
//! the **raw logits** for selection, while the weights are the sigmoid of the
//! unbiased logits. Adding the bias to `sigmoid(logits)` instead (the
//! DeepSeek-V3 rule) picks a different expert set whenever a bias exceeds the
//! logit gap, and the output stays finite and fluent. See [`route_logits`].

use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::afmoe::Mlp;
use crate::models::kolibri1::ModelArgs;
use crate::models::switch_layers::{SwitchGLU, fused_moe_enabled};

/// Upstream `MoEGate.__call__` on precomputed f32 router logits
/// `[tokens, num_experts]`. Returns `(indices, weights)`, both
/// `[tokens, top_k]`, with the indices ordered by descending selection value
/// and the weights in f32.
///
/// 1. `selection = logits + bias`
/// 2. top-`k` of `selection` (`argpartition(-selection, k - 1)`, upstream's
///    orientation, which decides the set when selection values tie)
/// 3. reorder those `k` by descending `selection`
/// 4. `weights = sigmoid(logits)` gathered at the reordered indices
/// 5. divide by `sum + 1e-20` only when `norm_topk_prob`
pub(crate) fn route_logits(
    logits: &MlxArray,
    bias: &MlxArray,
    top_k: i32,
    norm_topk_prob: bool,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let selection = mlxcel_core::add(logits, bias);
    let n_experts = *mlxcel_core::array_shape(&selection).last().unwrap_or(&0);
    let kth = (top_k - 1).clamp(0, (n_experts - 1).max(0));
    let order = mlxcel_core::argpartition(&mlxcel_core::negative(&selection), kth, -1);
    let indices = slice_axis(&order, -1, 0, top_k);
    let selected = mlxcel_core::take_along_axis(&selection, &indices, -1);
    let reorder = mlxcel_core::argsort(&mlxcel_core::negative(&selected), -1);
    let indices = mlxcel_core::take_along_axis(&indices, &reorder, -1);
    let weights = mlxcel_core::take_along_axis(&mlxcel_core::sigmoid(logits), &indices, -1);
    let weights = if norm_topk_prob {
        let sum = mlxcel_core::sum_axis(&weights, -1, true);
        let eps = mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::divide(&weights, &mlxcel_core::add(&sum, &eps))
    } else {
        weights
    };
    (indices, weights)
}

/// Upstream `SparseMoeBlock`.
pub struct Kolibri1Moe {
    /// `mlp.gate.weight` cast to f32 once at load and transposed to
    /// `[hidden, num_experts]`; upstream casts both operands to f32 on every
    /// call, which yields the same product.
    pub router_t: UniquePtr<MlxArray>,
    /// `mlp.gate.e_score_correction_bias` in f32, `[num_experts]`.
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub switch_mlp: SwitchGLU,
    pub shared_experts: Mlp,
    pub top_k: i32,
    pub norm_topk_prob: bool,
}

impl Kolibri1Moe {
    fn route(&self, x_flat: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let x32 = mlxcel_core::astype(x_flat, mlxcel_core::dtype::FLOAT32);
        let logits = mlxcel_core::matmul(&x32, &self.router_t);
        route_logits(
            &logits,
            &self.e_score_correction_bias,
            self.top_k,
            self.norm_topk_prob,
        )
    }

    /// The fused single-token decode kernel, or `None` where it declines
    /// (disabled, more than one token, or an unsupported configuration).
    fn try_fused_routed(
        &self,
        x_flat: &MlxArray,
        indices: &MlxArray,
        scores: &MlxArray,
        hidden: i32,
        fused_enabled: bool,
    ) -> Option<UniquePtr<MlxArray>> {
        if mlxcel_core::array_shape(x_flat)[0] != 1 || !fused_enabled {
            return None;
        }
        let fused_scores = mlxcel_core::astype(scores, mlxcel_core::array_dtype(x_flat));
        self.switch_mlp
            .forward_fused_kernel(x_flat, indices, &fused_scores)
            .map(|out| mlxcel_core::reshape(&out, &[1, hidden]))
    }

    fn forward_with(&self, x: &MlxArray, fused_enabled: bool) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = *orig_shape.last().unwrap_or(&0);
        let tokens: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
        let x_flat = mlxcel_core::reshape(x, &[tokens, hidden]);

        let (indices, scores) = self.route(&x_flat);
        let routed = match self.try_fused_routed(&x_flat, &indices, &scores, hidden, fused_enabled)
        {
            Some(out) => out,
            None => {
                // `(routed * scores[..., None]).sum(-2).astype(routed.dtype)`:
                // the f32 scores promote the combine, which sums in f32 and is
                // cast back once, so this is not `moe_weighted_sum` (that
                // helper casts the weights down first).
                let expert_out = self.switch_mlp.forward(&x_flat, &indices);
                let out32 = mlxcel_core::astype(&expert_out, mlxcel_core::dtype::FLOAT32);
                let weighted =
                    mlxcel_core::multiply(&out32, &mlxcel_core::expand_dims(&scores, -1));
                let summed = mlxcel_core::sum_axis(&weighted, -2, false);
                mlxcel_core::astype(&summed, mlxcel_core::array_dtype(&expert_out))
            }
        };
        let routed = mlxcel_core::reshape(&routed, &orig_shape);
        // The shared expert is added unconditionally, with no gate.
        mlxcel_core::add(&routed, &self.shared_experts.forward(x))
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.forward_with(x, fused_moe_enabled())
    }

    /// The whole block for one token through the fused kernel, or `None`
    /// where the kernel declines, so the caller never substitutes another
    /// arithmetic for a row it meant to fuse.
    fn try_full_fused(&self, x_row: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(x_row);
        let hidden = *shape.last().unwrap_or(&0);
        let x_flat = mlxcel_core::reshape(x_row, &[1, hidden]);
        let (indices, scores) = self.route(&x_flat);
        let routed = self.try_fused_routed(&x_flat, &indices, &scores, hidden, true)?;
        let routed = mlxcel_core::reshape(&routed, &shape);
        Some(mlxcel_core::add(
            &routed,
            &self.shared_experts.forward(x_row),
        ))
    }

    /// A native decode batch `[B, 1, hidden]`: keep the exact B=1 fused expert
    /// arithmetic per row when the kernel accepts this configuration, so a
    /// batched row matches the same row decoded alone; otherwise one
    /// whole-batch `SwitchGLU` pass. AFMoE makes the same choice.
    pub(crate) fn forward_batched_decode(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let fused_enabled = fused_moe_enabled();
        if shape.len() == 3
            && shape[0] >= 2
            && shape[1] == 1
            && fused_enabled
            && let Some(first) = self.try_full_fused(&slice_axis(x, 0, 0, 1))
        {
            let mut outputs = Vec::with_capacity(shape[0] as usize);
            outputs.push(first);
            for row in 1..shape[0] {
                outputs.push(self.forward_with(&slice_axis(x, 0, row, row + 1), true));
            }
            let refs: Vec<_> = outputs.iter().map(|output| &**output).collect();
            return mlxcel_core::concatenate_many(&refs, 0);
        }
        self.forward_with(x, fused_enabled)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let gate_key = format!("{prefix}.gate.weight");
        let gate = weights
            .get(&gate_key)
            .ok_or_else(|| format!("Weight not found: {gate_key}"))?;
        let router_t =
            mlxcel_core::transpose(&mlxcel_core::astype(gate, mlxcel_core::dtype::FLOAT32));
        // Upstream initializes the bias to zeros, so a checkpoint without one
        // routes on the raw logits.
        let bias_key = format!("{prefix}.gate.e_score_correction_bias");
        let e_score_correction_bias = match weights.get(&bias_key) {
            Some(bias) => mlxcel_core::astype(bias, mlxcel_core::dtype::FLOAT32),
            None => {
                mlxcel_core::full_f32(&[args.num_experts as i32], 0.0, mlxcel_core::dtype::FLOAT32)
            }
        };
        Ok(Self {
            router_t,
            e_score_correction_bias,
            switch_mlp: SwitchGLU::from_weights(
                weights,
                &format!("{prefix}.switch_mlp"),
                group_size,
                bits,
            )?,
            shared_experts: Mlp::load(
                weights,
                &format!("{prefix}.shared_experts"),
                group_size,
                bits,
            )?,
            top_k: args.num_experts_per_tok as i32,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}
