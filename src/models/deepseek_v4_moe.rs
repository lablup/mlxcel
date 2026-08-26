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

//! DeepSeek-V4 MoE: hash-routed gating with `sqrtsoftplus` scoring and a
//! limited SwiGLU in both the routed experts and the shared expert.
//!
//! What is NOT V3 here, each silently wrong if done the V3 way:
//!
//! * **Scoring** is `sqrt(softplus(logits))` in float32 (`sqrtsoftplus`).
//!   `softmax` and `sigmoid` are also accepted; anything else is rejected at
//!   config validation. There is no V3 group-limited softmax and no
//!   `n_group` / `topk_group` machinery.
//! * **Hash routing**: layers with `layer_idx < num_hash_layers` (3 on the
//!   real checkpoint) pick expert INDICES from a `tid2eid` lookup table
//!   (`[vocab_size, top_k]`, indexed by the raw input token ids), not from
//!   the logits. The logits still supply the expert WEIGHTS, gathered at the
//!   hash-chosen indices. This is why `input_ids` threads through the whole
//!   block stack.
//! * **Selection vs weighting** on non-hash layers: `argpartition` runs over
//!   `scores + e_score_correction_bias` for selection ONLY; the returned
//!   weights are gathered from the UNBIASED scores. Same contract
//!   `bailing_moe` / `afmoe` / `klear` document; getting it wrong keeps the
//!   output finite and plausible.
//! * **`norm_topk_prob`** renormalises by `sum + 1e-20`, but only when the
//!   scoring function is not `softmax`; then `routed_scaling_factor` (1.5).
//! * **Limited SwiGLU**: `gate = min(gate, limit)`,
//!   `up = clip(up, -limit, limit)`, then `silu(gate) * up`, with
//!   `swiglu_limit = 10.0`, in the routed experts AND the shared expert. The
//!   shared expert's intermediate size is
//!   `moe_intermediate_size * n_shared_experts`.
//! * **Every layer is MoE**: there is no `first_k_dense_replace` dense
//!   prefix.
//!
//! The routed experts reuse the shared `switch_layers::SwitchLinear`
//! (per-path quantization: the real checkpoint ships them as mxfp4 at group
//! size 32 under an affine/64 top level) with the limited activation applied
//! between the projections; the shared `SwitchGLU` is not reused because its
//! activation is a plain SwiGLU. The fused MoE decode kernel
//! (`MLXCEL_FUSED_MOE`) is deliberately not wired: it has no clamp stage.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::switch_layers::{
    SwitchLinear, moe_weighted_sum, validate_expert_quantization_params,
    validate_expert_quantization_shapes,
};

use super::{ModelArgs, ScoringFunc, get_weight_copy};

/// `silu(min(gate, limit)) * clip(up, -limit, limit)` (`_limited_swiglu`).
/// A non-positive limit disables clamping, as in the reference.
pub(crate) fn limited_swiglu(gate: &MlxArray, up: &MlxArray, limit: f32) -> UniquePtr<MlxArray> {
    if limit > 0.0 {
        let dtype = mlxcel_core::array_dtype(gate);
        let limit_arr = mlxcel_core::full_f32(&[1], limit, dtype);
        let neg_limit = mlxcel_core::full_f32(&[1], -limit, dtype);
        let gate = mlxcel_core::minimum(gate, &limit_arr);
        let up = mlxcel_core::clip(up, &neg_limit, &limit_arr);
        let gate = mlxcel_core::utils::silu(&gate);
        mlxcel_core::multiply(&gate, &up)
    } else {
        let gate = mlxcel_core::utils::silu(gate);
        mlxcel_core::multiply(&gate, up)
    }
}

/// Apply the configured scoring function to f32 logits.
fn score_logits(logits_f32: &MlxArray, func: ScoringFunc) -> UniquePtr<MlxArray> {
    match func {
        ScoringFunc::Softmax => mlxcel_core::softmax_precise(logits_f32, -1),
        ScoringFunc::Sigmoid => mlxcel_core::sigmoid(logits_f32),
        ScoringFunc::SqrtSoftplus => mlxcel_core::sqrt(&mlxcel_core::utils::softplus(logits_f32)),
    }
}

/// Router: hash lookup for the first `num_hash_layers` layers, biased
/// argpartition for the rest.
pub(crate) enum Routing {
    /// `tid2eid` `[vocab_size, top_k]` int32; indices come from the token
    /// ids, weights from the logits.
    Hash { tid2eid: UniquePtr<MlxArray> },
    /// `e_score_correction_bias` `[n_routed_experts]` float32; biases the
    /// selection but never the weights.
    Biased { bias: UniquePtr<MlxArray> },
}

pub(crate) struct MoEGate {
    /// `[n_routed_experts, hidden_size]`.
    weight: UniquePtr<MlxArray>,
    routing: Routing,
    top_k: i32,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
    scoring: ScoringFunc,
}

impl MoEGate {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        hash: bool,
    ) -> Result<Self, String> {
        let weight = get_weight_copy(weights, &format!("{prefix}.weight"))?;
        let w_shape = mlxcel_core::array_shape(&weight);
        let expected = [args.n_routed_experts as i32, args.hidden_size as i32];
        if w_shape != expected {
            return Err(format!(
                "{prefix}.weight: expected shape {expected:?}, checkpoint ships {w_shape:?}"
            ));
        }
        let routing = if hash {
            let tid2eid = get_weight_copy(weights, &format!("{prefix}.tid2eid"))?;
            let t_shape = mlxcel_core::array_shape(&tid2eid);
            let expected = [args.vocab_size as i32, args.num_experts_per_tok as i32];
            if t_shape != expected {
                return Err(format!(
                    "{prefix}.tid2eid: expected shape {expected:?}, checkpoint ships {t_shape:?}"
                ));
            }
            // The checkpoint ships the table as int64; index arithmetic and
            // take_along_axis want int32 (reference sanitize does the same
            // cast).
            let tid2eid = mlxcel_core::astype(&tid2eid, mlxcel_core::dtype::INT32);
            Routing::Hash { tid2eid }
        } else {
            Routing::Biased {
                bias: get_weight_copy(weights, &format!("{prefix}.e_score_correction_bias"))?,
            }
        };
        Ok(Self {
            weight,
            routing,
            top_k: args.num_experts_per_tok as i32,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
            scoring: args.scoring_func_parsed().expect("validated at load"),
        })
    }

    /// Returns `(indices [B, L, K] int32, weights [B, L, K] f32)`.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        input_ids: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let w_t = mlxcel_core::transpose(&self.weight);
        let logits = mlxcel_core::matmul(x, &w_t);
        let logits = mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32);
        let scores = score_logits(&logits, self.scoring);

        let inds = match &self.routing {
            Routing::Hash { tid2eid } => {
                // Expert indices come from the token ids, not the logits.
                mlxcel_core::take(tid2eid, input_ids, 0)
            }
            Routing::Biased { bias } => {
                // Bias influences SELECTION only.
                let biased = mlxcel_core::add(&scores, bias);
                super::indexer::topk_indices(&biased, self.top_k)
            }
        };

        // Weights always come from the UNBIASED scores.
        let mut weights = mlxcel_core::take_along_axis(&scores, &inds, -1);
        if self.scoring != ScoringFunc::Softmax && self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&weights, -1, true);
            let eps = mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::dtype::FLOAT32);
            weights = mlxcel_core::divide(&weights, &mlxcel_core::add(&sum, &eps));
        }
        let weights = mlxcel_core::multiply_scalar(&weights, self.routed_scaling_factor);
        (inds, weights)
    }
}

/// Routed experts: three `SwitchLinear`s with the limited SwiGLU between.
struct LimitedSwitchGlu {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
    limit: f32,
}

impl LimitedSwitchGlu {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let x_exp = mlxcel_core::expand_dims(x, -2);
        let x_exp = mlxcel_core::expand_dims(&x_exp, -3);
        let gate = self.gate_proj.forward(&x_exp, indices, false);
        let up = self.up_proj.forward(&x_exp, indices, false);
        let h = limited_swiglu(&gate, &up, self.limit);
        let out = self.down_proj.forward(&h, indices, false);
        // `[..., top_k, 1, hidden] -> [..., top_k, hidden]` so
        // `moe_weighted_sum` broadcasts against `[..., top_k, 1]` scores.
        mlxcel_core::squeeze_axis(&out, -2)
    }
}

/// Shared expert / dense MLP with the same limited SwiGLU.
pub(crate) struct LimitedMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    limit: f32,
}

impl LimitedMlp {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.up_proj"),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                group_size,
                bits,
            )?,
            limit: args.swiglu_limit,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let h = limited_swiglu(&gate, &up, self.limit);
        self.down_proj.forward(&h)
    }
}

/// One V4 FFN block: hash/biased gate, routed limited-SwiGLU experts, plus
/// the always-on shared expert.
pub(crate) struct DeepseekV4MoE {
    gate: MoEGate,
    switch_mlp: LimitedSwitchGlu,
    shared_experts: LimitedMlp,
}

impl DeepseekV4MoE {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let hash = layer_idx < args.num_hash_layers;
        let gate = MoEGate::from_weights(weights, args, &format!("{prefix}.gate"), hash)?;

        let mut projs = Vec::with_capacity(3);
        for name in ["gate_proj", "up_proj", "down_proj"] {
            let proj_prefix = format!("{prefix}.switch_mlp.{name}");
            let (group_size, bits, mode) = args.expert_quantization(&proj_prefix);
            // Family-local expert wrapper: bound the declared pair before the
            // loader stores it (docs/adding-models.md, Quantization Parameter
            // Bounds), and cross-check it against the plane on disk when the
            // plane is quantized.
            validate_expert_quantization_params(&proj_prefix, group_size, bits)?;
            if let (Some(w), Some(s)) = (
                weights.get(&format!("{proj_prefix}.weight")),
                weights.get(&format!("{proj_prefix}.scales")),
            ) {
                validate_expert_quantization_shapes(
                    &proj_prefix,
                    &mlxcel_core::array_shape(w),
                    &mlxcel_core::array_shape(s),
                    group_size,
                    bits,
                )?;
            }
            projs.push(SwitchLinear::from_weights_with_mode(
                weights,
                &proj_prefix,
                group_size,
                bits,
                &mode,
            )?);
        }
        let down_proj = projs.pop().expect("three projections pushed");
        let up_proj = projs.pop().expect("three projections pushed");
        let gate_proj = projs.pop().expect("three projections pushed");

        Ok(Self {
            gate,
            switch_mlp: LimitedSwitchGlu {
                gate_proj,
                up_proj,
                down_proj,
                limit: args.swiglu_limit,
            },
            shared_experts: LimitedMlp::from_weights(
                weights,
                args,
                &format!("{prefix}.shared_experts"),
            )?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let (inds, scores) = self.gate.forward(x, input_ids);
        let y = self.switch_mlp.forward(x, &inds);
        let routed = moe_weighted_sum(&y, &scores, mlxcel_core::array_dtype(x));
        let shared = self.shared_experts.forward(x);
        mlxcel_core::add(&routed, &shared)
    }
}
