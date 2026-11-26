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

//! SiTU MLPs and the sigmoid-routed MXFP8 MoE of Raon-1 mini.
//!
//! ```text
//! situ(g, v) = (4 tanh(g / 4) sigmoid(g)) * (25 tanh(v / 25))     float32, rounded once
//! mlp(x)     = down(situ(gate(x), up(x)))
//! scores     = sigmoid(x.f32 @ W.f32^T)                           router in float32
//! idx        = top-k(scores + e_score_correction_bias)
//! w          = scores[idx] / (sum + 1e-20) * routed_scaling_factor
//! routed     = (sum_k experts_k(x).f32 * w_k).astype(dt)
//! out        = routed + shared_experts(x)                         added in the model dtype
//! ```
//!
//! The combine follows `Raon1MiniMoE.forward` (non-native path): the routed
//! sum is accumulated in float32 and rounded before the shared expert is
//! added. The routed experts are the original checkpoint's native MXFP8
//! planes (E4M3 codes, E8M0 scales per 32 inputs), which is MLX's `mxfp8`
//! layout as is, or an NVFP4 export's planes with their per-expert scale.

use crate::models::switch_layers::{SwitchGLU, SwitchGluActivation, situ_activation};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// MLX `mxfp8`: 8-bit E4M3 codes with one E8M0 scale per 32 inputs.
pub const MXFP8_GROUP_SIZE: i32 = 32;
pub const MXFP8_BITS: i32 = 8;

/// SiTU-GLU MLP (the dense layer-0 MLP and the shared expert).
pub struct RaonSituMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    beta: f32,
    linear_beta: f32,
}

impl RaonSituMlp {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        beta: f32,
        linear_beta: f32,
    ) -> Result<Self, String> {
        let linear =
            |leaf: &str| super::raon1_mini_quant::linear(weights, &format!("{prefix}.{leaf}"));
        Ok(Self {
            gate_proj: linear("gate_proj")?,
            up_proj: linear("up_proj")?,
            down_proj: linear("down_proj")?,
            beta,
            linear_beta,
        })
    }

    /// Input widths (issue #3457): `gate_proj` / `up_proj` take `hidden`,
    /// `down_proj` takes `intermediate`.
    pub fn check_input_widths(
        &self,
        prefix: &str,
        hidden: usize,
        intermediate: usize,
    ) -> Result<(), String> {
        self.gate_proj
            .check_input_width(&format!("{prefix}.gate_proj"), hidden)?;
        self.up_proj
            .check_input_width(&format!("{prefix}.up_proj"), hidden)?;
        self.down_proj
            .check_input_width(&format!("{prefix}.down_proj"), intermediate)
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        use mlxcel_core::row_invariant::linear;
        let gate = linear(&self.gate_proj, x);
        let up = linear(&self.up_proj, x);
        let act = situ_activation(&gate, &up, self.beta, Some(self.linear_beta));
        linear(&self.down_proj, &act)
    }
}

/// Routing options of [`RaonMoe`].
#[derive(Debug, Clone, Copy)]
pub struct MoeSpec {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub renormalize: bool,
    pub routed_scaling_factor: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub has_shared_expert: bool,
}

pub struct RaonMoe {
    /// `[hidden, num_experts]` float32 (the router runs in float32).
    router_t: UniquePtr<MlxArray>,
    /// `[num_experts]` float32.
    correction_bias: Option<UniquePtr<MlxArray>>,
    switch_mlp: SwitchGLU,
    shared_experts: Option<RaonSituMlp>,
    /// `moe.reasoning_effort_steering` (`[hidden]` float32), present only on
    /// a layer the config's `reasoning_effort_steering.layers` lists. The
    /// decoder layer adds `alpha * steering` to this MoE's output (#3435).
    steering: Option<UniquePtr<MlxArray>>,
    spec: MoeSpec,
}

impl RaonMoe {
    /// `steered` is whether the config lists this layer for reasoning-effort
    /// steering: its vector is then required, and otherwise dropped.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        spec: MoeSpec,
        steered: bool,
    ) -> Result<Self, String> {
        let gate_key = format!("{prefix}.gate.weight");
        let gate = weights
            .get(&gate_key)
            .ok_or_else(|| format!("Missing weight: {gate_key}"))?;
        let shape = mlxcel_core::array_shape(gate);
        if shape != [spec.num_experts as i32, spec.hidden_size as i32] {
            return Err(format!(
                "{gate_key}: expected [num_experts {}, hidden_size {}], got {shape:?} (the router \
                 must stay unquantized)",
                spec.num_experts, spec.hidden_size
            ));
        }
        let router_t = mlxcel_core::astype(gate, dtype::FLOAT32);
        let router_t = mlxcel_core::contiguous(&mlxcel_core::swap_axes(&router_t, 0, 1), false);
        let correction_bias = weights
            .get(&format!("{prefix}.gate.e_score_correction_bias"))
            .map(|b| mlxcel_core::reshape(&mlxcel_core::astype(b, dtype::FLOAT32), &[-1]));
        if let Some(b) = &correction_bias {
            let n = mlxcel_core::array_shape(b)[0];
            if n as usize != spec.num_experts {
                return Err(format!(
                    "{prefix}.gate.e_score_correction_bias: {n} entries for {} experts",
                    spec.num_experts
                ));
            }
        }

        let switch_prefix = format!("{prefix}.switch_mlp");
        for leaf in ["gate_proj", "up_proj", "down_proj"] {
            super::raon1_mini_quant::check_scales_match_weight(
                weights,
                &format!("{switch_prefix}.{leaf}"),
            )?;
            let key = format!("{switch_prefix}.{leaf}.weight");
            let w = weights
                .get(&key)
                .ok_or_else(|| format!("Missing weight: {key}"))?;
            let s = mlxcel_core::array_shape(w);
            if s.first().copied() != Some(spec.num_experts as i32) {
                return Err(format!(
                    "{key}: {s:?} does not stack num_experts {}",
                    spec.num_experts
                ));
            }
        }
        // The layout comes from the planes: MXFP8 g32 (the original
        // checkpoint) or NVFP4 g16 with a per-expert scale (the exports, #3437).
        let switch_mlp =
            SwitchGLU::from_weights(weights, &switch_prefix, MXFP8_GROUP_SIZE, MXFP8_BITS)?
                .with_activation(SwitchGluActivation::SiTU {
                    beta: spec.situ_beta,
                    linear_beta: Some(spec.situ_linear_beta),
                });

        let shared_experts = if spec.has_shared_expert {
            Some(RaonSituMlp::from_weights(
                weights,
                &format!("{prefix}.shared_experts"),
                spec.situ_beta,
                spec.situ_linear_beta,
            )?)
        } else {
            None
        };
        let steering = load_steering(weights, prefix, spec.hidden_size, steered)?;

        Ok(Self {
            router_t,
            correction_bias,
            switch_mlp,
            shared_experts,
            steering,
            spec,
        })
    }

    /// Input widths of the routed experts (`moe_intermediate`) and the
    /// shared experts (`shared_intermediate`) (issue #3457).
    pub fn check_input_widths(
        &self,
        prefix: &str,
        moe_intermediate: usize,
        shared_intermediate: usize,
    ) -> Result<(), String> {
        let hidden = self.spec.hidden_size;
        self.switch_mlp.check_input_widths(
            &format!("{prefix}.switch_mlp"),
            hidden,
            moe_intermediate,
        )?;
        match &self.shared_experts {
            Some(shared) => shared.check_input_widths(
                &format!("{prefix}.shared_experts"),
                hidden,
                shared_intermediate,
            ),
            None => Ok(()),
        }
    }

    /// The layer's reasoning-effort steering vector, when the layer is steered.
    pub fn steering_vector(&self) -> Option<&MlxArray> {
        self.steering.as_deref()
    }

    /// `y` steered by `alpha` (`[B, 1, 1]` float32, one alpha per row):
    /// `(y.f32 + alpha * c).astype(dt)`, rounded once, as the reference's
    /// `Raon1MiniDecoderLayer.forward` does. `y` unchanged when this layer
    /// carries no vector.
    pub fn steer(&self, y: UniquePtr<MlxArray>, alpha: &MlxArray) -> UniquePtr<MlxArray> {
        let Some(c) = self.steering.as_deref() else {
            return y;
        };
        let dt = mlxcel_core::array_dtype(&y);
        let shift = mlxcel_core::multiply(alpha, c);
        mlxcel_core::astype(
            &mlxcel_core::add(&mlxcel_core::astype(&y, dtype::FLOAT32), &shift),
            dt,
        )
    }

    /// `(top-k expert ids [N, k], float32 routing weights [N, k])` for `x_flat` `[N, D]`.
    pub fn route(&self, x_flat: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let logits = mlxcel_core::row_invariant::per_row(
            &mlxcel_core::astype(x_flat, dtype::FLOAT32),
            |x| mlxcel_core::matmul(x, &self.router_t),
        );
        let scores = mlxcel_core::sigmoid(&logits);
        let selection = match &self.correction_bias {
            Some(bias) => mlxcel_core::add(&scores, bias),
            None => mlxcel_core::copy(&scores),
        };
        let k = self.spec.top_k as i32;
        let n = mlxcel_core::array_shape(&selection)[0];
        let order = mlxcel_core::argpartition(&mlxcel_core::negative(&selection), k - 1, -1);
        let idx = mlxcel_core::slice(&order, &[0, 0], &[n, k]);
        let mut w = mlxcel_core::take_along_axis(&scores, &idx, -1);
        if self.spec.renormalize {
            let eps = mlxcel_core::full_f32(&[1], 1e-20, dtype::FLOAT32);
            let sum = mlxcel_core::add(&mlxcel_core::sum_axis(&w, -1, true), &eps);
            w = mlxcel_core::divide(&w, &sum);
        }
        if self.spec.routed_scaling_factor != 1.0 {
            w = mlxcel_core::multiply_scalar(&w, self.spec.routed_scaling_factor);
        }
        (idx, w)
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let d = shape[shape.len() - 1];
        let n: i32 = shape[..shape.len() - 1].iter().product();
        let flat = mlxcel_core::reshape(x, &[n, d]);
        let dt = mlxcel_core::array_dtype(&flat);
        let (idx, w) = self.route(&flat);
        let expert_out = self.switch_mlp.forward(&flat, &idx); // [N, k, D]
        let weighted = mlxcel_core::multiply(
            &mlxcel_core::astype(&expert_out, dtype::FLOAT32),
            &mlxcel_core::expand_dims(&w, -1),
        );
        let routed = mlxcel_core::astype(&mlxcel_core::sum_axis(&weighted, -2, false), dt);
        let y = match &self.shared_experts {
            Some(shared) => mlxcel_core::add(&routed, &shared.forward(&flat)),
            None => routed,
        };
        mlxcel_core::reshape(&y, &shape)
    }
}

/// The `[hidden]` float32 steering vector of a steered layer. A steered layer
/// without one is a load error; an unsteered layer's vector is dropped, as the
/// reference registers the buffer only on listed layers.
fn load_steering(
    weights: &WeightMap,
    prefix: &str,
    hidden_size: usize,
    steered: bool,
) -> Result<Option<UniquePtr<MlxArray>>, String> {
    let key = format!("{prefix}.reasoning_effort_steering");
    let Some(vector) = weights.get(&key) else {
        return if steered {
            Err(format!(
                "Missing weight: {key} (the config's reasoning_effort_steering lists this layer)"
            ))
        } else {
            Ok(None)
        };
    };
    if !steered {
        tracing::debug!("{key}: layer not listed in reasoning_effort_steering.layers; dropped");
        return Ok(None);
    }
    let shape = mlxcel_core::array_shape(vector);
    if shape != [hidden_size as i32] {
        return Err(format!(
            "{key}: expected [hidden_size {hidden_size}], got {shape:?}"
        ));
    }
    Ok(Some(mlxcel_core::astype(vector, dtype::FLOAT32)))
}
