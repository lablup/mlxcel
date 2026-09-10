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

//! Per-model-family heuristics for the pipeline [`ModelProfile`] builder.
//!
//! Each model family either keeps the uniform dense-layer cost (default
//! `match` arm) or overrides it to account for MoE expert weights,
//! Gemma 4's double-wide MLP on KV-shared consumers, or similar layer
//! heterogeneity. See [`super::partition_profile::build_profile_from_json`]
//! for the entry point.
//!
//! Used by: `super::partition_profile::build_per_layer_bytes`,
//! `super::partition_profile::build_adjacency`

use serde_json::Value;

use super::partition::LayerAdjacencyGroup;

/// Populate the per-layer byte vector, accounting for MoE expert layers
/// and Gemma 4's double-wide MLP on KV-shared consumers.
pub(super) fn build_per_layer_bytes(
    root: &Value,
    text: Option<&Value>,
    model_type: &str,
    num_layers: usize,
    dense_layer_bytes: u64,
    hidden_size: u64,
    bits_per_weight: u64,
) -> Vec<u64> {
    let mut out = vec![dense_layer_bytes; num_layers];

    match model_type {
        "mixtral" | "qwen3_5_moe" | "qwen3_5_moe_vlm" | "exaone_moe" | "gpt_oss" | "glm4_moe"
        | "glm4_moe_lite" | "glm_moe_dsa" | "phi_moe" => {
            let experts = expert_count(text, root);
            let moe_intermediate = moe_intermediate_size(text, root).unwrap_or_else(|| {
                text.and_then(|t| t.get("intermediate_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(hidden_size * 4)
            });
            let expert_block =
                moe_expert_block_bytes(hidden_size, moe_intermediate, experts, bits_per_weight);
            let mlp_only = dense_mlp_bytes(hidden_size, moe_intermediate, bits_per_weight);
            let moe_layer_bytes = dense_layer_bytes
                .saturating_sub(mlp_only)
                .saturating_add(expert_block);
            for slot in out.iter_mut() {
                *slot = moe_layer_bytes;
            }
        }
        "deepseek_v3" | "deepseek_v2" | "deepseek" | "nemotron_h" => {
            let experts = expert_count(text, root);
            let first_dense = text
                .and_then(|t| t.get("first_k_dense_replace"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("first_k_dense_replace").and_then(|v| v.as_u64()))
                .unwrap_or(0) as usize;
            let moe_freq = text
                .and_then(|t| t.get("moe_layer_freq"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("moe_layer_freq").and_then(|v| v.as_u64()))
                .unwrap_or(1) as usize;
            let moe_intermediate = moe_intermediate_size(text, root).unwrap_or_else(|| {
                text.and_then(|t| t.get("intermediate_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(hidden_size * 4)
            });
            let expert_block =
                moe_expert_block_bytes(hidden_size, moe_intermediate, experts, bits_per_weight);
            let mlp_only = dense_mlp_bytes(hidden_size, moe_intermediate, bits_per_weight);
            for (i, slot) in out.iter_mut().enumerate() {
                let is_moe =
                    experts > 0 && i >= first_dense && moe_freq > 0 && i.is_multiple_of(moe_freq);
                if is_moe {
                    *slot = dense_layer_bytes
                        .saturating_sub(mlp_only)
                        .saturating_add(expert_block);
                }
            }
        }
        "llama4" | "llama4_vlm" => {
            let experts = expert_count(text, root);
            let step = text
                .and_then(|t| t.get("interleave_moe_layer_step"))
                .and_then(|v| v.as_u64())
                .or_else(|| {
                    root.get("interleave_moe_layer_step")
                        .and_then(|v| v.as_u64())
                })
                .unwrap_or(1) as usize;
            let moe_intermediate = text
                .and_then(|t| t.get("intermediate_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(hidden_size * 4);
            let dense_intermediate = text
                .and_then(|t| t.get("intermediate_size_mlp"))
                .and_then(|v| v.as_u64())
                .unwrap_or(moe_intermediate);
            let expert_block =
                moe_expert_block_bytes(hidden_size, moe_intermediate, experts, bits_per_weight);
            let dense_mlp_only = dense_mlp_bytes(hidden_size, dense_intermediate, bits_per_weight);
            let moe_mlp_only = dense_mlp_bytes(hidden_size, moe_intermediate, bits_per_weight);
            for (i, slot) in out.iter_mut().enumerate() {
                let is_moe_step = step > 0 && (i % step) == (step - 1);
                if is_moe_step {
                    *slot = dense_layer_bytes
                        .saturating_sub(moe_mlp_only)
                        .saturating_add(expert_block);
                } else {
                    *slot = dense_layer_bytes
                        .saturating_sub(moe_mlp_only)
                        .saturating_add(dense_mlp_only);
                }
            }
        }
        "jamba" => {
            let experts = expert_count(text, root);
            let expert_period = text
                .and_then(|t| t.get("expert_layer_period"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("expert_layer_period").and_then(|v| v.as_u64()))
                .unwrap_or(2) as usize;
            let expert_offset = text
                .and_then(|t| t.get("expert_layer_offset"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("expert_layer_offset").and_then(|v| v.as_u64()))
                .unwrap_or(0) as usize;
            let moe_intermediate = text
                .and_then(|t| t.get("intermediate_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(hidden_size * 4);
            let expert_block =
                moe_expert_block_bytes(hidden_size, moe_intermediate, experts, bits_per_weight);
            let mlp_only = dense_mlp_bytes(hidden_size, moe_intermediate, bits_per_weight);
            for (i, slot) in out.iter_mut().enumerate() {
                let is_moe = experts > 0
                    && expert_period > 0
                    && i >= expert_offset
                    && (i - expert_offset).is_multiple_of(expert_period);
                if is_moe {
                    *slot = dense_layer_bytes
                        .saturating_sub(mlp_only)
                        .saturating_add(expert_block);
                }
            }
        }
        "kimi_k3" => {
            kimi_k3_per_layer_bytes(text, root, &mut out, hidden_size, bits_per_weight);
        }
        "gemma4" | "gemma4_vlm" | "gemma3" | "gemma3_text" => {
            let num_shared = text
                .and_then(|t| t.get("num_kv_shared_layers"))
                .and_then(|v| v.as_u64())
                .or_else(|| root.get("num_kv_shared_layers").and_then(|v| v.as_u64()))
                .unwrap_or(0) as usize;
            let use_double = text
                .and_then(|t| t.get("use_double_wide_mlp"))
                .and_then(|v| v.as_bool())
                .or_else(|| root.get("use_double_wide_mlp").and_then(|v| v.as_bool()))
                .unwrap_or(false);
            if use_double && num_shared > 0 {
                let intermediate = text
                    .and_then(|t| t.get("intermediate_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(hidden_size * 4);
                let mlp_base = dense_mlp_bytes(hidden_size, intermediate, bits_per_weight);
                let mlp_double =
                    dense_mlp_bytes(hidden_size, intermediate.saturating_mul(2), bits_per_weight);
                let first_shared = num_layers.saturating_sub(num_shared);
                for (i, slot) in out.iter_mut().enumerate() {
                    if i >= first_shared {
                        *slot = dense_layer_bytes
                            .saturating_sub(mlp_base)
                            .saturating_add(mlp_double);
                    }
                }
            }
        }
        _ => {}
    }

    out
}

/// Kimi K3 per-layer bytes (issue #1334).
///
/// Layer `i` is KDA when `i + 1` is in `linear_attn_config.kda_layers`,
/// otherwise MLA, and MoE when `i >= first_k_dense_replace` on the
/// `moe_layer_freq` stride (layer 0 is dense on the published config). The
/// attention blocks differ in size: KDA carries `qkv_proj [3P, D]`, the
/// full-rank gate `[P, D]`, `o_proj [D, P]` and the small `f_a` / `f_b` /
/// `b` projections; MLA carries `q_a [q_lora, D]`, `q_b [H * q_head, q_lora]`,
/// `kv_a [kv_lora + rope, D]`, `kv_b [H * (nope + v), kv_lora]`, the output
/// gate `[P, D]` and `o_proj`. The routed experts are compressed-tensors
/// mxfp4 regardless of `bits_per_weight` (which describes the dense planes):
/// 4 bits per weight plus one E8M0 byte per 32-weight group, on the latent
/// width `routed_expert_hidden_size`. The router, the latent down / up
/// projections and the shared experts stay at `bits_per_weight`.
fn kimi_k3_per_layer_bytes(
    text: Option<&Value>,
    root: &Value,
    out: &mut [u64],
    hidden_size: u64,
    bits_per_weight: u64,
) {
    let get =
        |key: &str| -> Option<&Value> { text.and_then(|t| t.get(key)).or_else(|| root.get(key)) };
    let num = |key: &str, default: u64| get(key).and_then(|v| v.as_u64()).unwrap_or(default);
    let linear = get("linear_attn_config");
    let linear_num = |key: &str, default: u64| {
        linear
            .and_then(|l| l.get(key))
            .and_then(|v| v.as_u64())
            .unwrap_or(default)
    };
    let kda_layers: Vec<u64> = linear
        .and_then(|l| l.get("kda_layers"))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();
    let full_rank_gate = linear
        .and_then(|l| l.get("use_full_rank_gate"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let d = hidden_size;
    let kda_heads = linear_num("num_heads", 96);
    let kda_head_dim = linear_num("head_dim", 128);
    let kernel = linear_num("short_conv_kernel_size", 4);
    let p = kda_heads.saturating_mul(kda_head_dim);
    let heads = num("num_attention_heads", 96);
    let q_lora = get("q_lora_rank").and_then(|v| v.as_u64());
    let kv_lora = num("kv_lora_rank", 512);
    let nope = num("qk_nope_head_dim", 128);
    let rope = num("qk_rope_head_dim", 64);
    let v_head = num("v_head_dim", 128);
    let output_gate = get("mla_use_output_gate")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let experts = expert_count(text, root);
    let moe_intermediate = moe_intermediate_size(text, root).unwrap_or(0);
    let latent = get("routed_expert_hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(d);
    let shared = num("num_shared_experts", 0);
    let dense_intermediate = num("intermediate_size", d.saturating_mul(4));
    let first_dense = num("first_k_dense_replace", 0) as usize;
    let moe_freq = num("moe_layer_freq", 1) as usize;

    let dense = |params: u64| super::partition_profile::param_bytes(params, bits_per_weight);

    // KDA attention: qkv + gate + o + f_a/f_b + b + conv + A_log/dt_bias/o_norm.
    let kda_gate = if full_rank_gate {
        p.saturating_mul(d)
    } else {
        d.saturating_mul(kda_head_dim)
            .saturating_add(kda_head_dim.saturating_mul(p))
    };
    let kda_attn = dense(
        3u64.saturating_mul(p)
            .saturating_mul(d)
            .saturating_add(kda_gate)
            .saturating_add(d.saturating_mul(p))
            .saturating_add(d.saturating_mul(kda_head_dim))
            .saturating_add(kda_head_dim.saturating_mul(p))
            .saturating_add(kda_heads.saturating_mul(d))
            .saturating_add(3u64.saturating_mul(p).saturating_mul(kernel)),
    );
    // MLA attention: q (LoRA or plain) + kv_a + kv_b + output gate + o.
    let q_head = nope.saturating_add(rope);
    let q_params = match q_lora {
        Some(r) => d
            .saturating_mul(r)
            .saturating_add(r.saturating_mul(heads.saturating_mul(q_head))),
        None => d.saturating_mul(heads.saturating_mul(q_head)),
    };
    let hv = heads.saturating_mul(v_head);
    let mla_attn = dense(
        q_params
            .saturating_add(d.saturating_mul(kv_lora.saturating_add(rope)))
            .saturating_add(
                kv_lora.saturating_mul(heads.saturating_mul(nope.saturating_add(v_head))),
            )
            .saturating_add(if output_gate { d.saturating_mul(hv) } else { 0 })
            .saturating_add(hv.saturating_mul(d)),
    );

    // MoE block: mxfp4 experts on the latent, everything else dense-bit.
    let expert_params = experts
        .saturating_mul(3)
        .saturating_mul(moe_intermediate)
        .saturating_mul(latent);
    let expert_bytes = expert_params
        .saturating_mul(4)
        .saturating_div(8)
        .saturating_add(expert_params.saturating_div(32));
    let moe_dense_params = experts
        .saturating_mul(d)
        .saturating_add(2u64.saturating_mul(d).saturating_mul(latent))
        .saturating_add(
            3u64.saturating_mul(d)
                .saturating_mul(shared.saturating_mul(moe_intermediate)),
        );
    let moe_mlp = expert_bytes.saturating_add(dense(moe_dense_params));
    let dense_mlp = dense_mlp_bytes(d, dense_intermediate, bits_per_weight);
    let norms = dense(6u64.saturating_mul(d));

    for (i, slot) in out.iter_mut().enumerate() {
        let is_kda = kda_layers.contains(&(i as u64 + 1));
        let is_moe = experts > 0 && i >= first_dense && moe_freq > 0 && i.is_multiple_of(moe_freq);
        let attn = if is_kda { kda_attn } else { mla_attn };
        let mlp = if is_moe { moe_mlp } else { dense_mlp };
        *slot = attn.saturating_add(mlp).saturating_add(norms);
    }
}

/// Build the adjacency constraints implied by the model type.
///
/// Currently Gemma 4 is the only supported model with a mandatory adjacency
/// invariant: each of its `num_kv_shared_layers` consumers reads its keys
/// and values from the most recent earlier layer with the same
/// `layer_types[i]` value.
pub(super) fn build_adjacency(
    root: &Value,
    text: Option<&Value>,
    model_type: &str,
    num_layers: usize,
) -> Vec<LayerAdjacencyGroup> {
    let mut out = Vec::new();
    if !(model_type == "gemma4" || model_type == "gemma4_vlm") {
        return out;
    }
    let num_shared = text
        .and_then(|t| t.get("num_kv_shared_layers"))
        .and_then(|v| v.as_u64())
        .or_else(|| root.get("num_kv_shared_layers").and_then(|v| v.as_u64()))
        .unwrap_or(0) as usize;
    if num_shared == 0 || num_layers == 0 {
        return out;
    }
    let layer_types: Vec<String> = text
        .and_then(|t| t.get("layer_types"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default();
    if layer_types.len() < num_layers {
        return out;
    }
    let first_shared = num_layers.saturating_sub(num_shared);
    for consumer in first_shared..num_layers {
        let ltype = &layer_types[consumer];
        if let Some(source_rel) = layer_types[..first_shared].iter().rposition(|t| t == ltype) {
            let source = source_rel;
            out.push(LayerAdjacencyGroup {
                layers: source..(consumer + 1),
                reason: format!(
                    "gemma4 KV-shared layer {} reads keys/values from layer {}",
                    consumer, source
                ),
            });
        }
    }
    out
}

pub(super) fn expert_count(text: Option<&Value>, root: &Value) -> u64 {
    for key in [
        "num_local_experts",
        "n_routed_experts",
        "num_experts",
        "num_routed_experts",
    ] {
        if let Some(v) = text
            .and_then(|t| t.get(key))
            .and_then(|v| v.as_u64())
            .or_else(|| root.get(key).and_then(|v| v.as_u64()))
        {
            return v;
        }
    }
    0
}

pub(super) fn moe_intermediate_size(text: Option<&Value>, root: &Value) -> Option<u64> {
    for key in ["moe_intermediate_size", "expert_intermediate_size"] {
        if let Some(v) = text
            .and_then(|t| t.get(key))
            .and_then(|v| v.as_u64())
            .or_else(|| root.get(key).and_then(|v| v.as_u64()))
        {
            return Some(v);
        }
    }
    None
}

pub(super) fn moe_expert_block_bytes(
    hidden_size: u64,
    intermediate: u64,
    experts: u64,
    bits_per_weight: u64,
) -> u64 {
    let per_expert = hidden_size.saturating_mul(intermediate).saturating_mul(3);
    super::partition_profile::param_bytes(per_expert.saturating_mul(experts), bits_per_weight)
}

pub(super) fn dense_mlp_bytes(hidden_size: u64, intermediate: u64, bits_per_weight: u64) -> u64 {
    super::partition_profile::param_bytes(
        hidden_size.saturating_mul(intermediate).saturating_mul(3),
        bits_per_weight,
    )
}
