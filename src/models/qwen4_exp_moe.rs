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

//! Qwen4-Exp sparse MoE block: the Qwen3.5 MoE (top-k softmax routing over
//! stacked experts plus a sigmoid-gated shared expert), loaded with each
//! module's own declared quantization.
//!
//! `SparseMoeBlock` in `qwen3_next.rs` takes one quantization pair for the
//! whole block, which cannot describe this checkpoint: its routed experts are
//! 2- or 4-bit per layer while the shared expert is 4-bit and the router is
//! unquantized. The stacked experts must get their exact pair, because
//! `SwitchLinear` hands it to `gather_qmm` without reconciling it against the
//! tensor shapes.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::qwen3_next::{MLP, Quantization, Qwen3NextConfig, SwitchGLU};

use super::config::ResolvedQwen4ExpConfig;
use super::shapes;

pub(crate) struct Qwen4ExpMoeBlock {
    router: UnifiedLinear,
    experts: SwitchGLU,
    shared_expert: MLP,
    shared_expert_gate: UnifiedLinear,
    num_experts_per_tok: i32,
    norm_topk_prob: bool,
}

impl Qwen4ExpMoeBlock {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let quantization = &config.quantization;
        let pair_of = |modules: &[String]| -> Result<(i32, i32), String> {
            let first = quantization.for_module(&modules[0]);
            for module in &modules[1..] {
                if quantization.for_module(module) != first {
                    return Err(format!(
                        "Qwen4-Exp {} and {} declare different quantization ({first:?} vs {:?}); \
                         one stacked-expert or shared-expert block needs one pair",
                        modules[0],
                        module,
                        quantization.for_module(module)
                    ));
                }
            }
            Ok(first)
        };
        let projections = |block: &str| -> Vec<String> {
            ["gate_proj", "up_proj", "down_proj"]
                .iter()
                .map(|name| format!("{prefix}.{block}.{name}"))
                .collect()
        };
        let experts_pair = pair_of(&projections("switch_mlp"))?;
        for module in projections("switch_mlp") {
            check_expert_packing(weights, &module, experts_pair)?;
        }
        let shared_pair = pair_of(&projections("shared_expert"))?;

        // The router's width is the expert index range `gather_qmm` reads the
        // stacked planes with, so both must be `num_experts`.
        let raw = &config.raw;
        let (experts, hidden) = (raw.num_experts, raw.hidden_size);
        let (routed, shared) = (
            raw.moe_intermediate_size,
            raw.shared_expert_intermediate_size,
        );
        for (name, dims) in [
            ("gate_proj", [experts, routed, hidden]),
            ("up_proj", [experts, routed, hidden]),
            ("down_proj", [experts, hidden, routed]),
        ] {
            let module = format!("{prefix}.switch_mlp.{name}");
            shapes::expect_matrix(weights, &module, experts_pair, &dims)?;
        }
        for (name, dims) in [
            ("gate_proj", [shared, hidden]),
            ("up_proj", [shared, hidden]),
            ("down_proj", [hidden, shared]),
        ] {
            let module = format!("{prefix}.shared_expert.{name}");
            shapes::expect_matrix(weights, &module, shared_pair, &dims)?;
        }
        let linear = |name: &str, rows: usize| {
            let module = format!("{prefix}.{name}");
            let (group_size, bits) = quantization.for_module(&module);
            shapes::expect_matrix(weights, &module, (group_size, bits), &[rows, hidden])?;
            UnifiedLinear::from_weights(weights, &module, group_size, bits)
        };
        Ok(Self {
            router: linear("gate", experts)?,
            experts: SwitchGLU::from_weights(
                weights,
                &loader_config(config, experts_pair),
                &format!("{prefix}.switch_mlp"),
            )?,
            shared_expert: MLP::from_weights(
                weights,
                &loader_config(config, shared_pair),
                &format!("{prefix}.shared_expert"),
            )?,
            shared_expert_gate: linear("shared_expert_gate", 1)?,
            num_experts_per_tok: config.raw.num_experts_per_tok as i32,
            norm_topk_prob: config.raw.norm_topk_prob,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let hidden = shape[shape.len() - 1];
        let tokens: i32 = shape[..shape.len() - 1].iter().product();
        let x_flat = mlxcel_core::reshape(x, &[tokens, hidden]);

        let logits = self.router.forward(&x_flat);
        let gates = mlxcel_core::softmax_precise(&logits, -1);
        let n_experts = *mlxcel_core::array_shape(&logits)
            .last()
            .expect("router logits");
        let kth = n_experts - self.num_experts_per_tok;
        let order = mlxcel_core::argpartition(&gates, kth, -1);
        let topk = mlxcel_core::slice(&order, &[0, kth], &[tokens, n_experts]);
        let mut scores = mlxcel_core::take_along_axis(&gates, &topk, -1);
        if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis_native_dtype(&scores, -1, true);
            scores = mlxcel_core::divide(&scores, &sum);
        }

        // The generic fused MoE kernel accumulates expert outputs in float32.
        // Qwen4 checkpoints require bf16 rounding before weighting and summing
        // the selected experts, including single-token decode.
        let expert_out = self.experts.forward(&x_flat, &topk);
        let scores = mlxcel_core::expand_dims(&scores, -1);
        let weighted = mlxcel_core::multiply(&expert_out, &scores);
        let routed = mlxcel_core::sum_axis_native_dtype(&weighted, -2, false);

        let shared = self.shared_expert.forward(&x_flat);
        let shared_gate = mlxcel_core::sigmoid(&super::linear::row_invariant_projection(
            &self.shared_expert_gate,
            x,
        ));
        let shared_gate = mlxcel_core::reshape(&shared_gate, &[tokens, 1]);
        let out = mlxcel_core::add(&routed, &mlxcel_core::multiply(&shared, &shared_gate));
        mlxcel_core::reshape(&out, &shape)
    }
}

/// A packed expert plane whose width disagrees with its declared bits would
/// reach `gather_qmm` and abort the process at the first forward pass, so it
/// is refused here by name.
fn check_expert_packing(
    weights: &WeightMap,
    module: &str,
    (group_size, bits): (i32, i32),
) -> Result<(), String> {
    let (Some(weight), Some(scales)) = (
        weights.get(&format!("{module}.weight")),
        weights.get(&format!("{module}.scales")),
    ) else {
        return Ok(());
    };
    let packed = i64::from(*mlxcel_core::array_shape(weight).last().unwrap_or(&0));
    let groups = i64::from(*mlxcel_core::array_shape(scales).last().unwrap_or(&0));
    let (unpacked, grouped) = (
        packed * 32 / i64::from(bits),
        groups * i64::from(group_size),
    );
    if unpacked == grouped {
        Ok(())
    } else {
        Err(format!(
            "Qwen4-Exp {module} is not packed at its declared bits={bits}, group_size={group_size}: \
             the packed weight unpacks to {unpacked} input columns but the scales cover {grouped}"
        ))
    }
}

/// A `Qwen3NextConfig` carrying only what `SwitchGLU` / `MLP` read: the
/// quantization pair and the MoE sizes.
fn loader_config(
    config: &ResolvedQwen4ExpConfig,
    (group_size, bits): (i32, i32),
) -> Qwen3NextConfig {
    let raw = &config.raw;
    Qwen3NextConfig {
        model_type: raw.model_type.clone(),
        hidden_size: raw.hidden_size,
        num_hidden_layers: raw.num_hidden_layers,
        intermediate_size: raw.moe_intermediate_size,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        head_dim: config.head_dim,
        linear_num_value_heads: raw.linear_num_value_heads,
        linear_num_key_heads: raw.linear_num_key_heads,
        linear_key_head_dim: raw.linear_key_head_dim,
        linear_value_head_dim: raw.linear_value_head_dim,
        linear_conv_kernel_dim: raw.linear_conv_kernel_dim,
        num_experts: raw.num_experts,
        num_experts_per_tok: raw.num_experts_per_tok,
        decoder_sparse_step: 1,
        moe_intermediate_size: raw.moe_intermediate_size,
        shared_expert_intermediate_size: raw.shared_expert_intermediate_size,
        mlp_only_layers: Vec::new(),
        full_attention_interval: raw.full_attention_interval,
        rms_norm_eps: raw.rms_norm_eps,
        vocab_size: raw.vocab_size,
        rope_theta: config.rope.rope_theta,
        partial_rotary_factor: config.rope.partial_rotary_factor,
        max_position_embeddings: Some(raw.max_position_embeddings),
        norm_topk_prob: raw.norm_topk_prob,
        tie_word_embeddings: raw.tie_word_embeddings,
        attention_bias: false,
        quantization: Some(Quantization { group_size, bits }),
    }
}
