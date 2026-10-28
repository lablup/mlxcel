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

//! Ming-Image configuration, read from the component JSON files only.
//!
//! A port of `MingImageConfig` and its parts in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ming_image/config.py>
//! (mlx-vlm `73d366ec`): the DiT (`transformer/config.json`, the Z-Image
//! transformer config), the BailingMoeV2 conditioner backbone
//! (`mllm/config.json` `llm_config`, mapped as `_bailing_config` maps it),
//! the bidirectional Qwen2 connector (`connector/config.json`), the bridge
//! (`mlp/config.json`) and the 4-channel VAE (`vae/config.json`). Nothing here
//! reads a weight, so request checks run before any load.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

use crate::imagegen::pipeline::ImageSamplingDefaults;
use crate::imagegen::qwen_image::config::QwenImageVaeConfig;
use crate::imagegen::scheduler::DynamicShift;
use crate::imagegen::z_image::config::ZImageTransformerConfig;

/// Default denoising steps (`MingImageConfig.default_steps`).
pub const DEFAULT_STEPS: u32 = 12;
/// The only guidance the model is trained for (CFG off).
pub const GUIDANCE: f32 = 1.0;
/// Pixels per latent cell (the VAE's spatial factor).
pub const DOWNSAMPLE: u32 = 8;

/// The model's sampling defaults (12 steps, guidance 1.0).
#[must_use]
pub fn defaults() -> ImageSamplingDefaults {
    ImageSamplingDefaults {
        steps: DEFAULT_STEPS,
        guidance: GUIDANCE,
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Missing Ming-Image config: {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

fn positive(c: &Value, key: &str, default: i64) -> Result<i64> {
    match c.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_i64().filter(|n| *n > 0).with_context(|| {
            format!("Ming-Image config {key} must be a positive integer, got {v}")
        }),
    }
}

/// `mlp/config.json`: the learned query block and the two caption heads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgeConfig {
    /// `img_gen_scales[0] ** 2` learned query tokens.
    pub query_token_count: i32,
    /// `diffusion_c_input_dim`, the width of `cap_feats`.
    pub cap_feat_dim: i32,
    /// `diffusion_inner_dim`, the width of `cap_feats_2`.
    pub directvlm_dim: i32,
    /// Hidden-state indices concatenated into `cap_feats_2`.
    pub selected_layers: Vec<usize>,
}

impl BridgeConfig {
    pub fn from_json(c: &Value) -> Result<Self> {
        let scale = match c.get("img_gen_scales") {
            None | Some(Value::Null) => 16,
            Some(v) => v
                .as_array()
                .and_then(|a| a.first())
                .and_then(Value::as_i64)
                .filter(|n| (1..=64).contains(n))
                .with_context(|| {
                    format!("Ming-Image img_gen_scales must list a scale in 1..=64, got {v}")
                })?,
        };
        let selected_layers = match c.get("selected_hidden_states_layers") {
            None | Some(Value::Null) => vec![5, 12, 20],
            Some(v) => v
                .as_array()
                .and_then(|a| {
                    a.iter()
                        .map(|x| x.as_u64().map(|n| n as usize))
                        .collect::<Option<Vec<_>>>()
                })
                .filter(|l| !l.is_empty())
                .with_context(|| {
                    format!("Ming-Image selected_hidden_states_layers must be non-negative integers, got {v}")
                })?,
        };
        Ok(Self {
            query_token_count: (scale * scale) as i32,
            cap_feat_dim: positive(c, "diffusion_c_input_dim", 2560)? as i32,
            directvlm_dim: positive(c, "diffusion_inner_dim", 3840)? as i32,
            selected_layers,
        })
    }
}

/// The special tokens that frame the learned query block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTokens {
    pub patch: i32,
    pub start: i32,
    pub end: i32,
}

/// Everything about a checkpoint that does not need a weight.
#[derive(Debug, Clone)]
pub struct MingImageConfig {
    pub dit: ZImageTransformerConfig,
    pub bridge: BridgeConfig,
    pub vae: QwenImageVaeConfig,
    /// The BailingMoeV2 backbone, as upstream's `_bailing_config` builds it.
    pub mllm: crate::models::bailing_moe::ModelArgs,
    /// The Qwen2 connector blocks.
    pub connector: crate::models::llama3::ModelArgs,
    pub tokens: BlockTokens,
}

/// `llm_config` mapped onto the Bailing MoE config the way upstream's
/// `_bailing_config` maps it: sigmoid routing, QK norm, the expert bias from
/// `use_expert_bias`, the routed scaling from `routed_scaling_factor` (or
/// `moe_router_topk_scaling_factor`), no RoPE scaling (the checkpoint's
/// `video_rope` entry is not read upstream). The output head is never loaded,
/// so the mapping declares tied embeddings.
pub fn bailing_args(llm: &Value) -> Result<crate::models::bailing_moe::ModelArgs> {
    let get = |k: &str| llm.get(k).cloned().filter(|v| !v.is_null());
    let need = |k: &str| get(k).with_context(|| format!("Ming-Image llm_config has no {k}"));
    let scaling = get("routed_scaling_factor")
        .or_else(|| get("moe_router_topk_scaling_factor"))
        .unwrap_or(json!(2.5));
    let mapped = json!({
        "model_type": "bailing_moe_v2",
        "hidden_size": need("hidden_size")?,
        "intermediate_size": need("intermediate_size")?,
        "max_position_embeddings": get("max_position_embeddings").unwrap_or(json!(32768)),
        "moe_intermediate_size": need("moe_intermediate_size")?,
        "num_experts": need("num_experts")?,
        "num_shared_experts": need("num_shared_experts")?,
        "norm_topk_prob": get("norm_topk_prob").unwrap_or(json!(true)),
        "num_attention_heads": need("num_attention_heads")?,
        "num_experts_per_tok": need("num_experts_per_tok")?,
        "num_hidden_layers": need("num_hidden_layers")?,
        "num_key_value_heads": need("num_key_value_heads")?,
        "rms_norm_eps": get("rms_norm_eps").unwrap_or(json!(1e-6)),
        "rope_theta": get("rope_theta").unwrap_or(json!(600000.0)),
        "vocab_size": need("vocab_size")?,
        "first_k_dense_replace": get("first_k_dense_replace").unwrap_or(json!(1)),
        "use_qk_norm": true,
        "partial_rotary_factor": get("partial_rotary_factor").unwrap_or(json!(0.5)),
        "moe_router_enable_expert_bias": get("use_expert_bias").unwrap_or(json!(true)),
        "routed_scaling_factor": scaling,
        "score_function": "sigmoid",
        "n_group": get("n_group").unwrap_or(json!(8)),
        "topk_group": get("topk_group").unwrap_or(json!(4)),
        "tie_word_embeddings": true,
    });
    let args: crate::models::bailing_moe::ModelArgs =
        serde_json::from_value(mapped).context("cannot map Ming-Image llm_config")?;
    args.validate().map_err(anyhow::Error::msg)?;
    Ok(args)
}

/// Upstream `MingImageVAEConfig.from_dict` over the shared Qwen-Image VAE
/// config: `input_channels` (or `in_channels`, default 4) for both ends, and
/// the plain (non-residual) layout unless the file says otherwise.
pub fn vae_config(c: &Value) -> Result<QwenImageVaeConfig> {
    let mut vae = QwenImageVaeConfig::from_json(c)?;
    let channels = positive(c, "input_channels", positive(c, "in_channels", 4)?)? as i32;
    vae.in_channels = channels;
    vae.out_channels = positive(c, "out_channels", channels.into())? as i32;
    vae.is_residual = c
        .get("is_residual")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    vae.scaling_factor = Some(vae.scaling_factor.unwrap_or(8.0064));
    vae.shift_factor = Some(vae.shift_factor.unwrap_or(0.0));
    ensure!(
        vae.scaling_factor
            .is_some_and(|s| s.is_finite() && s != 0.0)
            && vae.shift_factor.is_some_and(f32::is_finite),
        "Ming-Image VAE scaling_factor must be finite and non-zero, shift_factor finite"
    );
    vae.validate()?;
    Ok(vae)
}

impl MingImageConfig {
    /// Read every component config under `root`.
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let mllm_json = read_json(&root.join("mllm/config.json"))?;
        let llm = mllm_json
            .get("llm_config")
            .context("Ming-Image mllm/config.json has no llm_config")?;
        let mllm = bailing_args(llm)?;
        let mut connector: crate::models::llama3::ModelArgs =
            serde_json::from_value(read_json(&root.join("connector/config.json"))?)
                .context("cannot parse the Ming-Image connector config")?;
        // A converted checkpoint quantizes the connector with the text encoder.
        connector.quantization = None;
        let token =
            |key: &str, default: i64| -> Result<i32> { Ok(positive(llm, key, default)? as i32) };
        let config = Self {
            dit: ZImageTransformerConfig::from_value(&read_json(
                &root.join("transformer/config.json"),
            )?)?,
            bridge: BridgeConfig::from_json(&read_json(&root.join("mlp/config.json"))?)?,
            vae: vae_config(&read_json(&root.join("vae/config.json"))?)?,
            mllm,
            connector,
            tokens: BlockTokens {
                patch: token("image_patch_token", 157_157)?,
                start: token("image_start_token", 157_158)?,
                end: token("image_end_token", 157_159)?,
            },
        };
        config.validate()?;
        Ok(config)
    }

    /// Cross-component checks upstream relies on implicitly.
    pub fn validate(&self) -> Result<()> {
        let layers = self.mllm.num_hidden_layers;
        if let Some(bad) = self.bridge.selected_layers.iter().find(|&&l| l > layers) {
            bail!(
                "Ming-Image selected hidden-state layer {bad} exceeds the {layers} backbone layers"
            );
        }
        ensure!(
            self.connector.hidden_size > 0 && self.connector.num_attention_heads > 0,
            "Ming-Image connector config is empty"
        );
        ensure!(
            self.bridge.cap_feat_dim == self.dit.text_embed_dim,
            "Ming-Image bridge width {} does not match the transformer cap_feat_dim {}",
            self.bridge.cap_feat_dim,
            self.dit.text_embed_dim
        );
        ensure!(
            self.bridge.directvlm_dim == self.dit.hidden_size,
            "Ming-Image directvlm width {} does not match the transformer dim {}",
            self.bridge.directvlm_dim,
            self.dit.hidden_size
        );
        ensure!(
            self.vae.z_dim == self.dit.in_channels,
            "Ming-Image VAE z_dim {} does not match the transformer in_channels {}",
            self.vae.z_dim,
            self.dit.in_channels
        );
        ensure!(
            self.vae.spatial_scale() == DOWNSAMPLE as i32,
            "Ming-Image VAE spatial scale is {}, the pipeline needs {DOWNSAMPLE}",
            self.vae.spatial_scale()
        );
        ensure!(
            self.vae.out_channels == 4,
            "Ming-Image VAE must decode 4 channels (RGBA), got {}",
            self.vae.out_channels
        );
        Ok(())
    }
}

/// Upstream `scheduler_shift` plus the fixed parts of the pipeline's
/// `FlowMatchEulerDiscreteScheduler`: base shift 0.5 at 256 tokens, max shift
/// 1.35 at the image's own length from 4096 tokens up, else 1.15 at 4096; no
/// terminal stretch.
#[must_use]
pub fn scheduler_shift(image_seq_len: u32) -> DynamicShift {
    let (max_image_seq_len, max_shift) = if image_seq_len >= 4096 {
        (image_seq_len, 1.35)
    } else {
        (4096, 1.15)
    };
    DynamicShift {
        base_shift: 0.5,
        max_shift,
        base_image_seq_len: 256,
        max_image_seq_len,
        shift_terminal: None,
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
