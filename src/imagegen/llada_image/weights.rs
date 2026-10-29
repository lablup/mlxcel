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

//! LLaDA-Image component loading.
//!
//! A port of `weights.py` and the `sanitize_*` functions in
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/llada_image>.
//! The raw Diffusers tree loads directly, every tensor in its stored dtype
//! (the published checkpoints are BF16), exactly as upstream's strict
//! `load_weights` does. The FP8 conversions upstream also restores are out of
//! scope and rejected by name. Each component is checked at load time: no
//! FP8 bytes, one floating dtype, and the shapes the modules expect.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::conditioning::{Dense, QueryFormer, TextProjection, tensor};
use super::config::{QueryFormerConfig, SigVqConfig, TextProjectionConfig, TransformerConfig};
use super::sigvq::SigVq;
use super::transformer::LladaImageTransformer;
use crate::imagegen::dit_blocks::{
    ADALN_EMBED_DIM, RopePrecision, sanitize_transformer_weights_for_patch,
};
use crate::imagegen::ernie_image::weights::{ConvLayout, match_conv_layout, tensor_layout};
use crate::imagegen::mage_flow::weights::component_metadata;
use crate::imagegen::vae::Flux2Vae;
use crate::models::Llada2MoeModel;
use crate::models::llada2_moe::{Llada2Numerics, ModelArgs};

/// Hidden width of the timestep MLP (`TimestepEmbedder(256, mid_size=1024)`).
const TIME_MID_DIM: i32 = 1024;

/// The text encoder's untied output head, read only by the VQ prior.
pub const LM_HEAD_KEY: &str = "model.lm_head.weight";

fn load_dir(dir: &Path, keep: impl FnMut(&str) -> bool) -> Result<WeightMap> {
    mlxcel_core::weights::load_weights_from_dir_index_filtered(dir, keep)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load LLaDA-Image weights from {}", dir.display()))
}

fn is_floating(dt: i32) -> bool {
    matches!(dt, dtype::FLOAT16 | dtype::FLOAT32 | dtype::BFLOAT16)
}

/// Reject FP8 conversions and mixed floating dtypes in one component.
pub fn check_component_dtypes(component: &str, weights: &WeightMap) -> Result<()> {
    if let Some(key) = weights.keys().find(|k| {
        k.ends_with(".weight_scale_inv") || (k.contains(".mlp.experts.") && k.ends_with("_scale"))
    }) {
        bail!(
            "FP8 LLaDA-Image checkpoints are not supported (found {key} in {component}); load the BF16 checkpoint"
        );
    }
    let mut floating = None;
    for (key, value) in weights {
        let dt = mlxcel_core::array_dtype(value);
        if dt == dtype::UINT8 {
            bail!(
                "{component} weight {key} is 8-bit; FP8 LLaDA-Image checkpoints are not supported"
            );
        }
        if is_floating(dt) {
            match floating {
                None => floating = Some(dt),
                Some(first) if first != dt => {
                    bail!("{component} mixes floating dtypes ({key} differs from the rest)")
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn expect_shape(weights: &WeightMap, key: &str, shape: &[i32]) -> Result<()> {
    let got = mlxcel_core::array_shape(&*tensor(weights, key)?);
    if got != shape {
        bail!("LLaDA-Image weight {key} has shape {got:?}, expected {shape:?}");
    }
    Ok(())
}

/// Upstream `sanitize_text_encoder_weights`: drop the LM head, strip
/// `model.language_model.`, and name the stacked experts the way the shared
/// `SwitchGLU` reads them (`mlp.switch_mlp.<proj>.weight`).
#[must_use]
pub fn sanitize_text_encoder_weights(weights: WeightMap) -> WeightMap {
    weights
        .into_iter()
        .filter(|(key, _)| key != LM_HEAD_KEY)
        .map(|(key, value)| {
            let key = key
                .strip_prefix("model.language_model.")
                .map_or(key.clone(), |rest| format!("model.{rest}"));
            let key = if key.contains(".mlp.experts.") {
                format!(
                    "{}.weight",
                    key.replace(".mlp.experts.", ".mlp.switch_mlp.")
                )
            } else {
                key
            };
            (key, value)
        })
        .collect()
}

fn text_encoder_args(dir: &Path) -> Result<ModelArgs> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let text = crate::models::sanitize_config_json(&text);
    serde_json::from_str(&text).with_context(|| format!("invalid {}", path.display()))
}

/// Load `<root>/text_encoder/` (without the LM head) as a LLaDA-2 MoE
/// decoder with the LLaDA-Image numerics.
pub fn load_text_encoder(root: &Path) -> Result<Llada2MoeModel> {
    let dir = root.join("text_encoder");
    let args = text_encoder_args(&dir)?;
    let raw = load_dir(&dir, |k| k != LM_HEAD_KEY)?;
    check_component_dtypes("text_encoder", &raw)?;
    let weights = sanitize_text_encoder_weights(raw);
    let (layers, hidden) = (args.num_hidden_layers, args.hidden_size as i32);
    expect_shape(
        &weights,
        "model.word_embeddings.weight",
        &[args.vocab_size as i32, hidden],
    )?;
    if layers > args.first_k_dense_replace {
        let e = args.num_experts as i32;
        let moe = args.moe_intermediate_size as i32;
        let p = format!("model.layers.{}.mlp.switch_mlp", args.first_k_dense_replace);
        expect_shape(
            &weights,
            &format!("{p}.gate_proj.weight"),
            &[e, moe, hidden],
        )?;
        expect_shape(
            &weights,
            &format!("{p}.down_proj.weight"),
            &[e, hidden, moe],
        )?;
    }
    Llada2MoeModel::from_weights_with_numerics(&weights, &args, Llada2Numerics::LladaImage)
        .map_err(anyhow::Error::msg)
        .context("cannot build the LLaDA-Image text encoder")
}

/// Load the untied `model.lm_head.weight` alone (upstream `load_lm_head`).
pub fn load_lm_head(root: &Path) -> Result<Dense> {
    let dir = root.join("text_encoder");
    let args = text_encoder_args(&dir)?;
    let weights = load_dir(&dir, |k| k == LM_HEAD_KEY)?;
    check_component_dtypes("text_encoder", &weights)?;
    let head = super::conditioning::shaped(
        &weights,
        LM_HEAD_KEY,
        &[args.vocab_size as i32, args.hidden_size as i32],
    )?;
    Dense::from_parts(&head, None)
}

pub fn load_queryformer(root: &Path) -> Result<QueryFormer> {
    let dir = root.join("queryformer");
    let config = QueryFormerConfig::from_dir(&dir)?;
    let weights = load_dir(&dir, |_| true)?;
    check_component_dtypes("queryformer", &weights)?;
    QueryFormer::from_weights(config, &weights)
}

pub fn load_text_projection(root: &Path) -> Result<TextProjection> {
    let dir = root.join("text_projection");
    let config = TextProjectionConfig::from_dir(&dir)?;
    let weights = load_dir(&dir, |_| true)?;
    check_component_dtypes("text_projection", &weights)?;
    TextProjection::from_weights(&config, &weights)
}

/// Upstream `sanitize_sigvq_weights`: keep only the `prior_*` tensors
/// without the encoder, rename the projector, and put the conv kernels in
/// MLX layout. The layout is decided once from the component metadata and
/// the result must match the module's shape, so a kernel is never permuted
/// twice.
pub fn sanitize_sigvq_weights(
    raw: WeightMap,
    config: &SigVqConfig,
    include_encoder: bool,
    layout: ConvLayout,
) -> Result<WeightMap> {
    let (dim, p) = (config.hidden_size, config.patch_size);
    let mut out = WeightMap::new();
    for (key, value) in raw {
        if !include_encoder && !key.starts_with("prior_") {
            continue;
        }
        let key = key
            .replace("prior_projector.net.0.proj.", "prior_projector.0.")
            .replace("prior_projector.net.2.", "prior_projector.1.");
        let value = if mlxcel_core::array_shape(&value).len() == 4 {
            let target = match key.as_str() {
                "visual.patch_embed.proj.weight" => [dim, p, p, config.in_channels],
                "vqmodel.quant_conv.weight" => [config.codebook_embed_dim, 1, 1, dim],
                _ => bail!("unexpected SigVQ convolution weight {key}"),
            };
            match_conv_layout(&value, &target, &key, layout)?
        } else {
            value
        };
        out.insert(key, value);
    }
    Ok(out)
}

pub fn load_sigvq(root: &Path, include_encoder: bool) -> Result<SigVq> {
    let dir = root.join("sigvq");
    let config = SigVqConfig::from_dir(&dir)?;
    let raw = load_dir(&dir, |k| include_encoder || k.starts_with("prior_"))?;
    check_component_dtypes("sigvq", &raw)?;
    let layout = tensor_layout(&component_metadata(&dir)?);
    let weights = sanitize_sigvq_weights(raw, &config, include_encoder, layout)?;
    SigVq::from_weights(config, &weights, include_encoder)
}

/// Split a fused `[3D, in]` / `[2D, in]` tensor into equal row blocks.
fn split_rows(value: &MlxArray, parts: i32) -> Vec<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(value);
    let rows = shape[0] / parts;
    (0..parts)
        .map(|i| {
            let mut start = vec![0; shape.len()];
            let mut stop = shape.clone();
            start[0] = i * rows;
            stop[0] = (i + 1) * rows;
            mlxcel_core::slice(value, &start, &stop)
        })
        .collect()
}

/// Upstream `sanitize_transformer_weights`: the patch-1 key renames of the
/// shared Z-Image blocks, plus fused `to_qkv` / `w13` splits.
#[must_use]
pub fn sanitize_transformer_weights(raw: WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (key, value) in sanitize_transformer_weights_for_patch(raw, "1-1") {
        if key.contains(".attention.to_qkv.") {
            for (name, part) in ["to_q", "to_k", "to_v"].iter().zip(split_rows(&value, 3)) {
                out.insert(key.replace(".to_qkv.", &format!(".{name}.")), part);
            }
        } else if key.contains(".feed_forward.w13.") {
            for (name, part) in ["w1", "w3"].iter().zip(split_rows(&value, 2)) {
                out.insert(key.replace(".w13.", &format!(".{name}.")), part);
            }
        } else {
            out.insert(key, value);
        }
    }
    out
}

/// Shapes the shared blocks read without checking.
fn check_transformer_shapes(config: &TransformerConfig, weights: &WeightMap) -> Result<()> {
    let dim = config.dim;
    let ffn = (f64::from(dim) / 3.0 * 8.0) as i32;
    let block = |name: &str, count: i32, modulation: bool| -> Result<()> {
        for i in 0..count {
            let p = format!("{name}.{i}");
            for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
                expect_shape(
                    weights,
                    &format!("{p}.attention.{proj}.weight"),
                    &[dim, dim],
                )?;
            }
            expect_shape(weights, &format!("{p}.feed_forward.w1.weight"), &[ffn, dim])?;
            expect_shape(weights, &format!("{p}.feed_forward.w3.weight"), &[ffn, dim])?;
            expect_shape(weights, &format!("{p}.feed_forward.w2.weight"), &[dim, ffn])?;
            if modulation {
                expect_shape(
                    weights,
                    &format!("{p}.adaLN_modulation.0.weight"),
                    &[4 * dim, ADALN_EMBED_DIM],
                )?;
            }
        }
        Ok(())
    };
    block("noise_refiner", config.n_refiner_layers, true)?;
    block("context_refiner", config.n_refiner_layers, false)?;
    block("sigvq_refiner", config.n_refiner_layers, false)?;
    block("layers", config.n_layers, true)?;
    expect_shape(
        weights,
        "t_embedder.linear1.weight",
        &[TIME_MID_DIM, ADALN_EMBED_DIM],
    )?;
    expect_shape(
        weights,
        "t_embedder.linear2.weight",
        &[ADALN_EMBED_DIM, TIME_MID_DIM],
    )?;
    expect_shape(
        weights,
        "final_layer.linear.weight",
        &[config.in_channels, dim],
    )?;
    expect_shape(
        weights,
        "final_layer.adaLN_modulation.0.weight",
        &[dim, ADALN_EMBED_DIM],
    )?;
    Ok(())
}

pub fn load_transformer(root: &Path) -> Result<LladaImageTransformer> {
    load_transformer_with_rope(root, RopePrecision::RoundToInput)
}

/// [`load_transformer`] with a chosen RoPE precision (parity controls).
pub(crate) fn load_transformer_with_rope(
    root: &Path,
    rope: RopePrecision,
) -> Result<LladaImageTransformer> {
    let dir = root.join("transformer");
    let config = TransformerConfig::from_dir(&dir)?;
    let raw = load_dir(&dir, |_| true)?;
    check_component_dtypes("transformer", &raw)?;
    let weights = sanitize_transformer_weights(raw);
    check_transformer_shapes(&config, &weights)?;
    LladaImageTransformer::from_weights_with_rope(config, &weights, rope)
        .context("cannot build the LLaDA-Image transformer")
}

/// The FLUX.2 VAE (upstream `flux2.weights.load_vae`), with the checkpoint's
/// batch-norm epsilon, in [`Flux2Vae::default_dtype`].
pub fn load_vae(root: &Path, include_encoder: bool) -> Result<Flux2Vae> {
    Flux2Vae::load_with_dtype(root, include_encoder, Flux2Vae::default_dtype())
        .context("cannot build the LLaDA-Image VAE")
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
