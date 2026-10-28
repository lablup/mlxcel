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

//! Component weight loading for Ming-Image, a port of `weights.py`
//! (mlx-vlm `73d366ec`).
//!
//! Two layouts load. The original Diffusers tree
//! (`inclusionAI/Ming-Image-0.1-Design`) keeps the conditioner in three
//! source directories: `mllm/` (the BailingMoeV2 backbone under
//! `model.model.*`, per-expert MoE tensors, raw router names, plus a vision
//! tower, audio gates and an output head this port never reads), `connector/`
//! (a Qwen2 decoder under `model.*`) and `mlp/` (`query_tokens_dict.16x16`,
//! `proj_in`, `proj_out`, `proj_directvlm.{0,1}`). A converted MLX tree
//! (`nativ-community/Ming-Image-0.1-Design-MLX-*`) stores the merged
//! encoder in `text_encoder/` (`mllm.*`, `connector.*` and the bridge names),
//! each component directory's `config.json` carrying `mlx_format: true` and,
//! when quantized, a `quantization` block. Quantized scales and biases keep
//! their stored dtype.
//!
//! Every map is rekeyed to the names mlxcel's loaders read: the backbone to
//! `model.*` with `query_layernorm` / `key_layernorm`, the connector to
//! `model.layers.*` / `model.norm.weight`, the bridge to the upstream module
//! names. Per-expert tensors are stacked into `switch_mlp` one layer at a
//! time, so the load holds at most one layer's copies twice.

use std::path::Path;

use anyhow::{Context, Result};
use mlxcel_core::weights::WeightMap;
use serde_json::Value;

use super::config::MingImageConfig;
use crate::imagegen::quant::{QuantSpec, validate_quantized_weights};

/// The conditioner's weights, rekeyed per consumer.
pub struct TextEncoderWeights {
    /// BailingMoeV2 backbone, `model.*`.
    pub mllm: WeightMap,
    /// Qwen2 connector, `model.layers.*` and `model.norm.weight`.
    pub connector: WeightMap,
    /// `query_tokens`, `proj_in`, `proj_out`, `directvlm_norm`, `directvlm_proj`.
    pub bridge: WeightMap,
    /// Quantization of a converted tree, `None` for dense weights.
    pub quant: Option<QuantSpec>,
}

fn component_config(dir: &Path) -> Result<Value> {
    let path = dir.join("config.json");
    if !path.exists() {
        return Ok(Value::Null);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

/// Whether a component directory is an MLX-native conversion.
pub fn is_mlx_native(dir: &Path) -> Result<bool> {
    Ok(component_config(dir)?
        .get("mlx_format")
        .and_then(Value::as_bool)
        .unwrap_or(false))
}

/// The validated `quantization` block of a component's `config.json`.
pub fn read_quant(dir: &Path) -> Result<Option<QuantSpec>> {
    let config = component_config(dir)?;
    let Some(q) = config.get("quantization").filter(|q| q.is_object()) else {
        return Ok(None);
    };
    let int = |k: &str| {
        q.get(k)
            .and_then(Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
    };
    let mode = q.get("mode").and_then(Value::as_str).unwrap_or("affine");
    QuantSpec::validated(
        "Ming-Image",
        mode,
        int("bits").unwrap_or(0),
        int("group_size").unwrap_or(0),
    )
    .map(Some)
}

fn load_dir(dir: &Path, keep: impl FnMut(&str) -> bool) -> Result<WeightMap> {
    mlxcel_core::weights::load_weights_from_dir_filtered(dir, keep)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load Ming-Image weights from {}", dir.display()))
}

/// Raw backbone key to the Bailing loader's name, `None` for tensors this
/// port never reads (vision tower, audio gates, output head, projector).
fn mllm_key(key: &str) -> Option<String> {
    let rest = key.strip_prefix("model.model.")?;
    if rest.contains(".mlp.audio_gate.") {
        return None;
    }
    let k = format!("model.{rest}");
    Some(
        k.replace(".attention.q_norm.", ".attention.query_layernorm.")
            .replace(".attention.k_norm.", ".attention.key_layernorm."),
    )
}

/// Stack `{prefix}.experts.{e}.{proj}.weight` into
/// `{prefix}.switch_mlp.{proj}.weight` (upstream `_stack_experts`), one layer
/// at a time.
fn stack_experts(weights: &mut WeightMap, prefix: &str, num_experts: usize) -> Result<()> {
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        if !weights.contains_key(&format!("{prefix}.experts.0.{proj}.weight")) {
            continue;
        }
        let parts = (0..num_experts)
            .map(|e| {
                let key = format!("{prefix}.experts.{e}.{proj}.weight");
                weights
                    .remove(&key)
                    .with_context(|| format!("Ming-Image backbone weight missing: {key}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let stacked = mlxcel_core::stack_owned(&parts, 0);
        mlxcel_core::try_eval(&stacked).map_err(|e| anyhow::anyhow!("Ming-Image: {e}"))?;
        drop(parts);
        weights.insert(format!("{prefix}.switch_mlp.{proj}.weight"), stacked);
    }
    Ok(())
}

/// Raw bridge key to the upstream module name.
fn bridge_key(key: &str, query_scale: i32) -> Option<String> {
    if key == format!("query_tokens_dict.{query_scale}x{query_scale}") {
        return Some("query_tokens".into());
    }
    if key.starts_with("proj_in.") || key.starts_with("proj_out.") {
        return Some(key.into());
    }
    if key == "proj_directvlm.0.weight" {
        return Some("directvlm_norm.weight".into());
    }
    key.strip_prefix("proj_directvlm.1.")
        .map(|rest| format!("directvlm_proj.{rest}"))
}

/// Load the conditioner of `root` in either layout.
pub fn load_text_encoder(root: &Path, config: &MingImageConfig) -> Result<TextEncoderWeights> {
    let merged = root.join("text_encoder");
    let (mllm, connector, bridge, quant) = if is_mlx_native(&merged)? {
        let quant = read_quant(&merged)?;
        let all = load_dir(&merged, |_| true)?;
        let (mut mllm, mut connector, mut bridge) =
            (WeightMap::new(), WeightMap::new(), WeightMap::new());
        for (key, value) in all {
            if let Some(rest) = key.strip_prefix("mllm.") {
                mllm.insert(format!("model.{rest}"), value);
            } else if let Some(rest) = key.strip_prefix("connector.") {
                connector.insert(format!("model.{rest}"), value);
            } else {
                bridge.insert(key, value);
            }
        }
        (mllm, connector, bridge, quant)
    } else {
        let mut mllm = WeightMap::new();
        for (key, value) in load_dir(&root.join("mllm"), |k| mllm_key(k).is_some())? {
            mllm.insert(mllm_key(&key).expect("filtered"), value);
        }
        for layer in config.mllm.first_k_dense_replace..config.mllm.num_hidden_layers {
            stack_experts(
                &mut mllm,
                &format!("model.layers.{layer}.mlp"),
                config.mllm.num_experts(),
            )?;
        }
        let connector = load_dir(&root.join("connector"), |k| {
            k.starts_with("model.layers.") || k == "model.norm.weight"
        })?;
        let scale = (config.bridge.query_token_count as f64).sqrt().round() as i32;
        let mut bridge = WeightMap::new();
        for (key, value) in load_dir(&root.join("mlp"), |k| bridge_key(k, scale).is_some())? {
            bridge.insert(bridge_key(&key, scale).expect("filtered"), value);
        }
        (mllm, connector, bridge, None)
    };
    if let Some(spec) = &quant {
        for map in [&mllm, &connector, &bridge] {
            validate_quantized_weights(map, spec)?;
        }
    }
    Ok(TextEncoderWeights {
        mllm,
        connector,
        bridge,
        quant,
    })
}

/// Transformer tensors that upstream applies to a bf16 activation: Ming
/// feeds the DiT bf16 latents, a bf16 timestep and bf16 caption features, so
/// the embedders, every AdaLN modulation head (the timestep embedding stays
/// bf16) and the pre-attention path of the first noise-refiner and
/// context-refiner blocks compute in the checkpoint's bf16. Everything else
/// reads the f32 stream the first attention output starts.
#[must_use]
pub fn reads_bf16_activation(key: &str) -> bool {
    const BLOCK0: [&str; 6] = [
        "attention_norm1.weight",
        "attention.to_q.weight",
        "attention.to_k.weight",
        "attention.to_v.weight",
        "attention.norm_q.weight",
        "attention.norm_k.weight",
    ];
    if key.starts_with("x_embedder.")
        || key.starts_with("t_embedder.")
        || key.starts_with("cap_embedder.")
        || key.contains("adaLN_modulation.")
    {
        return true;
    }
    ["noise_refiner.0.", "context_refiner.0."]
        .iter()
        .any(|block| {
            key.strip_prefix(block)
                .is_some_and(|rest| BLOCK0.contains(&rest))
        })
}

/// Load the transformer of `root` (raw or converted, dense or quantized) in
/// the block layout of [`crate::imagegen::dit_blocks`], with every tensor in
/// the dtype that makes this build's arithmetic match upstream's.
///
/// Upstream keeps the weights bf16. A bf16 weight meeting an f32 activation
/// promotes to f32 there, but the CUDA build of mlxcel's MLX promotes the pair
/// to bf16, so the tensors that read the f32 stream are stored as f16 (f16
/// with f32 is f32 everywhere), or f32 when the cast is not exact
/// ([`crate::imagegen::z_image::weights::cast_floats`]); the
/// [`reads_bf16_activation`] tensors stay bf16. A quantized tree is
/// dequantized first (its scales are bf16, so the dense weights are bf16).
pub fn load_transformer(root: &Path) -> Result<(WeightMap, Vec<String>)> {
    let dir = root.join("transformer");
    let raw = load_dir(&dir, |_| true)?;
    let weights = if is_mlx_native(&dir)? {
        match read_quant(&dir)? {
            Some(spec) => dequantize(raw, &spec)?,
            None => raw,
        }
    } else {
        crate::imagegen::dit_blocks::sanitize_transformer_weights(raw)
    };
    let mut weights = weights;
    let widened =
        crate::imagegen::z_image::weights::cast_floats(&mut weights, reads_bf16_activation);
    Ok((weights, widened))
}

/// Upstream `sanitize_vae_weights` for the original tree: the causal-3D conv
/// kernels (`[O, I, T, kH, kW]`) keep their last temporal slice, the frame a
/// single image is computed from, and the `[C, 1, 1(, 1)]` RMS-norm gammas
/// flatten to `[C]`. The 2-D kernels stay in their Diffusers layout; the
/// shared VAE loader transposes them.
#[must_use]
pub fn sanitize_vae_weights(raw: WeightMap) -> WeightMap {
    raw.into_iter()
        .map(|(key, value)| {
            let shape = mlxcel_core::array_shape(&value);
            let value = if key.ends_with(".gamma") {
                mlxcel_core::reshape(&value, &[shape.iter().product()])
            } else if shape.len() == 5 {
                let t = shape[2];
                let last = mlxcel_core::slice(
                    &value,
                    &[0, 0, t - 1, 0, 0],
                    &[shape[0], shape[1], t, shape[3], shape[4]],
                );
                mlxcel_core::reshape(&last, &[shape[0], shape[1], shape[3], shape[4]])
            } else {
                value
            };
            (key, value)
        })
        .collect()
}

/// Load the decoder half of `<root>/vae/` in either layout.
pub fn load_vae_weights(root: &Path) -> Result<WeightMap> {
    let dir = root.join("vae");
    let raw = load_dir(&dir, |k| {
        !(k.starts_with("encoder.") || k.starts_with("quant_conv."))
    })?;
    if is_mlx_native(&dir)? {
        Ok(raw)
    } else {
        Ok(sanitize_vae_weights(raw))
    }
}

/// Replace each quantized `<p>.weight` / `.scales` / `.biases` triple with a
/// dense `<p>.weight`.
fn dequantize(mut weights: WeightMap, spec: &QuantSpec) -> Result<WeightMap> {
    validate_quantized_weights(&weights, spec)?;
    for prefix in crate::imagegen::quant::quantized_prefixes(&weights) {
        let scales = weights
            .remove(&format!("{prefix}.scales"))
            .context("scales disappeared")?;
        let biases = weights.remove(&format!("{prefix}.biases"));
        let key = format!("{prefix}.weight");
        let packed = weights.remove(&key).context("weight disappeared")?;
        let bias_ptr = biases
            .as_ref()
            .map_or(std::ptr::null(), |b| &**b as *const mlxcel_core::MlxArray);
        // SAFETY: `bias_ptr` is null or borrowed from `biases`, alive for the call.
        let dense = unsafe {
            mlxcel_core::dequantize(
                &packed,
                &scales,
                bias_ptr,
                spec.group_size,
                spec.bits,
                &spec.mode,
            )
        };
        weights.insert(key, dense);
    }
    Ok(weights)
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
