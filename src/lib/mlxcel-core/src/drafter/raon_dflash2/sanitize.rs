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

//! Weight inventory of the DFlash2 drafter.
//!
//! The bf16 checkpoint needs no renaming. Sanitize refuses the vendor NVFP4
//! copy (`Raon-1-mini-mlx/dflash2/model.safetensors`, keys `w.*` / `q.*`)
//! by name and checks the shape of every tensor the forward reads, so a
//! malformed checkpoint fails here instead of as an MLX shape throw.

use crate::ffi;
use crate::weights::WeightMap;

use super::config::RaonDFlash2Config;

/// Tensors the forward reads, with their expected shapes.
pub fn expected_shapes(config: &RaonDFlash2Config) -> Vec<(String, Vec<i32>)> {
    let h = config.hidden_size as i32;
    let q = (config.num_attention_heads * config.head_dim) as i32;
    let kv = (config.num_key_value_heads * config.head_dim) as i32;
    let hd = config.head_dim as i32;
    let inter = config.intermediate_size as i32;
    let taps = config.conv_taps as i32;
    let coef = 2 * taps * config.conv_groups() as i32;
    let rank = config.selector_rank as i32;
    let vocab = config.vocab_size as i32;
    let mut out = vec![
        (
            "fc.weight".to_string(),
            vec![h, config.target_feature_size as i32],
        ),
        ("hidden_norm.weight".to_string(), vec![h]),
        ("norm.weight".to_string(), vec![h]),
        (
            "candidate_selector.hidden_projection.weight".to_string(),
            vec![rank, h],
        ),
        (
            "candidate_selector.predecessor_codebook".to_string(),
            vec![vocab, rank],
        ),
        (
            "candidate_selector.successor_codebook".to_string(),
            vec![vocab, rank],
        ),
    ];
    for i in 0..config.num_hidden_layers {
        let p = format!("layers.{i}");
        let mut push = |leaf: &str, shape: Vec<i32>| out.push((format!("{p}.{leaf}"), shape));
        push("input_layernorm.weight", vec![h]);
        push("post_attention_layernorm.weight", vec![h]);
        push("self_attn.q_proj.weight", vec![q, h]);
        push("self_attn.k_proj.weight", vec![kv, h]);
        push("self_attn.v_proj.weight", vec![kv, h]);
        push("self_attn.o_proj.weight", vec![h, q]);
        push("self_attn.q_norm.weight", vec![hd]);
        push("self_attn.k_norm.weight", vec![hd]);
        push("mlp.gate_proj.weight", vec![inter, h]);
        push("mlp.up_proj.weight", vec![inter, h]);
        push("mlp.down_proj.weight", vec![h, inter]);
        for conv in ["attention_conv", "mlp_conv"] {
            push(&format!("{conv}.base_kernel"), vec![2, taps, h]);
            push(&format!("{conv}.kernel_projection.weight"), vec![coef, h]);
        }
    }
    out
}

/// Refuse the vendor NVFP4 format and check every tensor's shape.
///
/// # Errors
/// Returns a message naming the first missing or misshapen tensor.
pub fn sanitize_weights(weights: &mut WeightMap, config: &RaonDFlash2Config) -> Result<(), String> {
    if weights
        .keys()
        .any(|k| k.starts_with("w.") || k.starts_with("q.") || k.starts_with("head."))
    {
        return Err(
            "Raon DFlash2: this is the vendor NVFP4 drafter bundled with Raon-1-mini-mlx \
             (keys w.* / q.* with fmt.* metadata), which mlxcel does not load; download \
             KRAFTON/Raon-1-mini-dflash2 (bf16) instead"
                .to_string(),
        );
    }
    for (key, shape) in expected_shapes(config) {
        let w = weights
            .get(&key)
            .ok_or_else(|| format!("Raon DFlash2: missing weight {key}"))?;
        let got = ffi::array_shape(w);
        if got != shape {
            return Err(format!(
                "Raon DFlash2: {key} has shape {got:?}, expected {shape:?}"
            ));
        }
    }
    Ok(())
}
