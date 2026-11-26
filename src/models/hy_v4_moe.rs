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

//! Hy4 MoE forward over the DeepSeek-V3.2 MoE weights ([`MoEBlock`]).
//!
//! The weights and their loader are V3.2's. The forward differs in three
//! places, so it is written here instead of changing the V3.2 one:
//!
//! * **float32 routing.** Upstream scores the router in float32
//!   (`group_expert_select`: `sigmoid(gates.astype(float32))`) and keeps
//!   `e_score_correction_bias` float32. The V3.2 port scores in the
//!   activation dtype; with 256 experts under bf16 activations that can pick a
//!   different top-8 near a tie.
//! * **float32 combine.** The float32 routing scores multiply the bf16 expert
//!   outputs, so the weighted sum runs in float32 and is cast back, as
//!   upstream's `weighted_expert_sum` does.
//! * **`swiglu_limit`.** The transformers reference (`HYV4Experts._apply_gate`
//!   in `modeling_hy_v4.py`) clamps the routed experts' gate pre-activation to
//!   at most `swiglu_limit` and the up projection to `[-limit, limit]` before
//!   `silu(gate) * up`. The dense layer-0 MLP and the shared expert are not
//!   clamped there. mlx-vlm applies no clamp; see the `hy_v4.rs` header for
//!   the decision.

use mlxcel_core::utils::{silu, slice_axis};
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::{MoEBlock, MoEGate, SwitchGLU};

/// noaux_tc expert selection in float32. Returns `(indices [.., k], scores
/// [.., k] f32)`.
fn route(gate: &MoEGate, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let f32_dtype = mlxcel_core::dtype::FLOAT32;
    let weight_t = mlxcel_core::transpose(&gate.weight);
    let gates = mlxcel_core::matmul(x, &weight_t);
    let scores = mlxcel_core::sigmoid(&mlxcel_core::astype(&gates, f32_dtype));
    let bias = mlxcel_core::astype(&gate.e_score_correction_bias, f32_dtype);
    let choice = mlxcel_core::add(&scores, &bias);

    let choice = if gate.n_group > 1 {
        let shape = mlxcel_core::array_shape(&choice);
        let n_experts = shape.last().copied().unwrap_or(1);
        let flat = mlxcel_core::reshape(&choice, &[-1, n_experts]);
        let masked = crate::models::switch_layers::group_mask_scores(
            &flat,
            gate.n_group as i32,
            gate.topk_group as i32,
        );
        mlxcel_core::reshape(&masked, &shape)
    } else {
        choice
    };

    let k = gate.top_k as i32;
    let order = mlxcel_core::argpartition(&mlxcel_core::negative(&choice), k - 1, -1);
    let indices = slice_axis(&order, -1, 0, k);
    let selected = mlxcel_core::take_along_axis(&scores, &indices, -1);
    let selected = if gate.top_k > 1 && gate.norm_topk_prob {
        let sum = mlxcel_core::sum_axis(&selected, -1, true);
        mlxcel_core::divide(&selected, &sum)
    } else {
        selected
    };
    let selected = mlxcel_core::multiply_scalar(&selected, gate.routed_scaling_factor);
    (indices, selected)
}

/// The routed experts, `[.., k, hidden]`, with the reference's clamp when
/// `swiglu_limit` is set.
fn routed_experts(
    experts: &SwitchGLU,
    x: &MlxArray,
    indices: &MlxArray,
    swiglu_limit: Option<f32>,
) -> UniquePtr<MlxArray> {
    let x = mlxcel_core::expand_dims(x, -2);
    let x = mlxcel_core::expand_dims(&x, -3);
    let gate = experts.gate_proj.forward(&x, indices);
    let up = experts.up_proj.forward(&x, indices);
    let (gate, up) = match swiglu_limit {
        Some(limit) => {
            let dtype = mlxcel_core::array_dtype(&gate);
            let high = mlxcel_core::full_f32(&[1], limit, dtype);
            let low = mlxcel_core::full_f32(&[1], -limit, dtype);
            (
                mlxcel_core::minimum(&gate, &high),
                mlxcel_core::clip(&up, &low, &high),
            )
        }
        None => (gate, up),
    };
    let hidden = mlxcel_core::multiply(&silu(&gate), &up);
    mlxcel_core::squeeze_axis(&experts.down_proj.forward(&hidden, indices), -2)
}

/// `sum_k(scores_k * expert_k(x)) + shared(x)` in `x`'s dtype.
pub(super) fn moe_forward(
    block: &MoEBlock,
    x: &MlxArray,
    swiglu_limit: Option<f32>,
) -> UniquePtr<MlxArray> {
    let (indices, scores) = route(&block.gate, x);
    let expert_out = routed_experts(&block.experts, x, &indices, swiglu_limit);
    let scores = mlxcel_core::expand_dims(&scores, -1);
    let weighted = mlxcel_core::multiply(&expert_out, &scores);
    let combined = mlxcel_core::sum_axis(&weighted, -2, false);
    let combined = mlxcel_core::astype(&combined, mlxcel_core::array_dtype(x));
    match &block.shared_experts {
        Some(shared) => mlxcel_core::add(&combined, &shared.forward(x)),
        None => combined,
    }
}
