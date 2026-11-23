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

//! Kolibri-1 checkpoint sanitize and shape validation.
//!
//! The released MLX conversion (`nativ-community/Kolibri-1-MLX-4bit`) ships
//! 174,253 tensors: each layer's 384 experts as separate
//! `mlp.experts.{e}.{gate,up,down}_proj.{weight,scales,biases}` planes, and the
//! router's selection bias as `moe.router.expert_bias`. Upstream's `sanitize`
//! renames the bias to `mlp.gate.e_score_correction_bias` and stacks the
//! planes into `mlp.switch_mlp.*`; [`sanitize_weights`] does the same through
//! the shared [`drain_numbered_expert_planes`], which looks every key up by
//! name, so the pass costs a fixed number of map probes per expert plane and
//! never scans the 174k-entry map.

use mlxcel_core::weights::WeightMap;

use crate::models::afmoe::{
    validate_mlp, validate_norm, validate_projection, validate_quantized_packing,
    validate_stacked_experts,
};
use crate::models::gpt2::dim_eq;
use crate::models::kolibri1::ModelArgs;
use crate::models::switch_layers::drain_numbered_expert_planes;

const PROJECTIONS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// Rename the selection bias and stack the per-expert planes, in place.
///
/// Per layer, mirroring upstream `LanguageModel.sanitize`:
///
/// - `moe.router.expert_bias` (or `mlp.expert_bias`) becomes
///   `mlp.gate.e_score_correction_bias` unless that key already exists.
/// - `mlp.experts.{proj}.{leaf}` (a pre-stacked export) becomes
///   `mlp.switch_mlp.{proj}.{leaf}`.
/// - Numbered `mlp.experts.{e}.{proj}.{leaf}` planes are drained and stacked
///   into `mlp.switch_mlp.{proj}.{leaf}`. A checkpoint with fewer planes than
///   `num_experts` stacks short and is refused by [`validate_weights`].
///
/// Idempotent: a map that already holds `switch_mlp.*` is left as it is.
pub fn sanitize_weights(weights: &mut WeightMap, args: &ModelArgs) {
    if args.tie_word_embeddings {
        weights.remove("lm_head.weight");
    }
    for layer in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        let target_bias = format!("{prefix}.mlp.gate.e_score_correction_bias");
        if !weights.contains_key(&target_bias) {
            for source in [
                format!("{prefix}.moe.router.expert_bias"),
                format!("{prefix}.mlp.expert_bias"),
            ] {
                if let Some(bias) = weights.remove(&source) {
                    weights.insert(target_bias.clone(), bias);
                    break;
                }
            }
        }
        for proj in PROJECTIONS {
            let stacked = format!("{prefix}.mlp.switch_mlp.{proj}");
            for leaf in ["weight", "scales", "biases"] {
                let target = format!("{stacked}.{leaf}");
                if weights.contains_key(&target) {
                    continue;
                }
                if let Some(tensor) = weights.remove(&format!("{prefix}.mlp.experts.{proj}.{leaf}"))
                {
                    weights.insert(target, tensor);
                }
            }
            if !weights.contains_key(&format!("{stacked}.weight")) {
                drain_numbered_expert_planes(
                    weights,
                    &format!("{prefix}.mlp.experts"),
                    proj,
                    &stacked,
                );
                // Materialize this layer's stack now, while its 384 source
                // planes are the only ones alive, so the copy happens at load
                // one layer at a time instead of for all layers inside the
                // first forward.
                for leaf in ["weight", "scales", "biases"] {
                    if let Some(tensor) = weights.get(&format!("{stacked}.{leaf}")) {
                        mlxcel_core::eval(tensor);
                    }
                }
            }
        }
    }
}

/// Reject a checkpoint whose real tensor shapes disagree with `config.json`
/// before any of them reaches MLX. Runs after [`sanitize_weights`].
pub fn validate_weights(weights: &WeightMap, args: &ModelArgs) -> Result<(), String> {
    let hidden = args.hidden_size;
    let head_dim = args.head_dim;
    let q_size = args.num_attention_heads * head_dim;
    let kv_size = args.num_key_value_heads * head_dim;
    let group_size = args.group_size();
    let bits = args.bits();

    validate_norm(weights, "model.norm.weight", hidden)?;
    validate_quantized_packing(weights, "model.embed_tokens", hidden, group_size, bits)?;
    if !args.tie_word_embeddings {
        validate_projection(
            weights,
            "lm_head",
            args.vocab_size,
            hidden,
            group_size,
            bits,
        )?;
    }

    for layer in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        let attn = format!("{prefix}.self_attn");
        for (leaf, out_features, in_features) in [
            ("q_proj", q_size, hidden),
            ("k_proj", kv_size, hidden),
            ("v_proj", kv_size, hidden),
            ("o_proj", hidden, q_size),
        ] {
            validate_projection(
                weights,
                &format!("{attn}.{leaf}"),
                out_features,
                in_features,
                group_size,
                bits,
            )?;
        }
        validate_norm(weights, &format!("{attn}.q_norm.weight"), head_dim)?;
        validate_norm(weights, &format!("{attn}.k_norm.weight"), head_dim)?;
        for leaf in [
            "input_layernorm",
            "post_attn_norm",
            "post_attention_layernorm",
            "post_ffn_norm",
        ] {
            validate_norm(weights, &format!("{prefix}.{leaf}.weight"), hidden)?;
        }

        let mlp = format!("{prefix}.mlp");
        // The router runs as an f32 matmul on the plain weight; upstream's
        // quant predicate never quantizes it, so a quantized one is refused
        // rather than dequantized behind the caller's back.
        let gate = format!("{mlp}.gate");
        if weights.contains_key(&format!("{gate}.scales")) {
            return Err(format!(
                "{gate} is quantized; Kolibri-1 routes on the unquantized router weight in f32"
            ));
        }
        validate_projection(weights, &gate, args.num_experts, hidden, group_size, bits)?;
        if let Some(bias) = weights.get(&format!("{gate}.e_score_correction_bias")) {
            let shape = mlxcel_core::array_shape(bias);
            if shape.len() != 1 || !dim_eq(shape[0], args.num_experts) {
                return Err(format!(
                    "unexpected {gate}.e_score_correction_bias shape {shape:?}: expected [{}]",
                    args.num_experts
                ));
            }
        }
        for (proj, out_features, in_features) in [
            ("gate_proj", args.moe_intermediate_size, hidden),
            ("up_proj", args.moe_intermediate_size, hidden),
            ("down_proj", hidden, args.moe_intermediate_size),
        ] {
            validate_stacked_experts(
                weights,
                &format!("{mlp}.switch_mlp.{proj}"),
                args.num_experts,
                out_features,
                in_features,
                group_size,
                bits,
            )?;
        }
        validate_mlp(
            weights,
            &format!("{mlp}.shared_experts"),
            hidden,
            args.shared_expert_intermediate_size,
            group_size,
            bits,
        )?;
    }
    Ok(())
}
