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

//! LocateAnything (`locateanything`) VLM loader.
//!
//! `mlx-community/LocateAnything-3B-4bit` composition, verified against the
//! real `config.json` and `model.safetensors.index.json`:
//! - Text: `text_config.model_type == "qwen2"` (Qwen2.5-3B-Instruct geometry:
//!   hidden 2048, 36 layers, 16 heads, 2 kv heads, QKV bias,
//!   `tie_word_embeddings = true`, rope theta 1e6). 4-bit affine, group 64,
//!   including `embed_tokens` (which doubles as the tied LM head).
//! - Vision: `vision_config.model_type == "moonvit"` (MoonViT-SO-400M: hidden
//!   1152, 27 layers, 16 heads, intermediate 4304, patch 14, 64x64 learned
//!   position grid, 2x2 merge kernel). Plain bf16, **not** quantized: the
//!   checkpoint's `quantization` block lists only `language_model.*` entries.
//! - Connector `multi_modal_projector.{layer_norm,linear_1,linear_2}`: plain
//!   bf16, `layer_norm` sized 4608 = 1152 * 2 * 2.
//!
//! Weight-name mapping performed here mirrors upstream `Model.sanitize` plus
//! `VisionModel.sanitize`
//! (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/locateanything/):
//! - `vision_model.encoder.<rest>` / `vision_model.<rest>` -> `vision_tower.<rest>`
//! - `mlp1.{0,1,3}` -> `multi_modal_projector.{layer_norm,linear_1,linear_2}`
//! - `blocks.{i}.{wqkv,wo}` -> `blocks.{i}.attn.{wqkv,wo}` when stored flat
//! - `patch_embed.proj.weight` transposed to MLX channel-last layout
//! - `position_ids` dropped; `language_model.lm_head.weight` dropped when the
//!   text config ties embeddings.
//!
//! The released MLX conversion already ships the post-sanitize names
//! (`vision_tower.*`, `multi_modal_projector.*`, `language_model.model.*`), so
//! every rewrite above is a no-op there; they exist so an unconverted
//! `nvidia/LocateAnything-3B` checkpoint loads through the same path.
//!
//! Mixed precision: the top-level `quantization` block makes
//! [`load_vlm_weights_common`] treat the checkpoint as quantized and skip its
//! bf16 -> f16 conversion, which is right for the 4-bit text stack (its scales
//! and biases must stay bf16) but leaves the bf16 MoonViT tower and connector
//! untouched. On Apple Silicon those would JIT-crash on M5, so this loader
//! converts every remaining bf16 tensor to f16 except quantization `.scales` /
//! `.biases`, exactly as the InternVL loader does. On non-Apple hardware the
//! conversion is skipped and the tower stays bf16.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision::encoders::kimi_vl::{
    KimiVLVisionConfig, KimiVLVisionModel, MoonViTMlpActivation,
};
use crate::vision::locateanything::{LocateAnythingConnector, LocateAnythingVLM};
use crate::vision::processors::locateanything::LocateAnythingProcessor;

use super::{load_vlm_weights_common, parse_required_vlm_subconfig, read_sanitized_vlm_config};
use crate::loading::conv2d_weight_is_channel_last;

#[path = "vlm_locateanything_quant.rs"]
mod quant;

use quant::densify_mixed_precision_qkv;

/// Upstream `LAYER_NORM_EPS` in `locateanything/vision.py`. Kimi-VL's MoonViT
/// uses `1e-6`, so this cannot be left at the shared config default.
const MOONVIT_LAYER_NORM_EPS: f32 = 1e-5;

/// Upstream defaults from `LocateAnythingImageProcessor` / `config.py`.
const DEFAULT_IN_TOKEN_LIMIT: usize = 25_600;
const DEFAULT_IMAGE_TOKEN_INDEX: i32 = 151_665; // <IMG_CONTEXT>
const DEFAULT_IMG_START_TOKEN_ID: i32 = 151_666; // <img>
const DEFAULT_IMG_END_TOKEN_ID: i32 = 151_667; // </img>
/// Qwen2 chat defaults: `<|endoftext|>` and `<|im_end|>`.
const DEFAULT_EOS_TOKEN_IDS: [i32; 2] = [151_643, 151_645];

/// Upper bound on `vision_config.patch_size`; the released checkpoint declares
/// 14.
///
/// The ceiling exists so the geometry `patch_size` derives stays bounded.
/// [`LocateAnythingProcessor`] rounds each image side *up* to a multiple of
/// `merge * patch` and caps the result only with its own 511-patch grid
/// envelope, so the worst-case resized side is about `MAX_GRID_SIDE *
/// patch_size` pixels and the f32 patch buffer grows as `patch_size^2`. The
/// `in_token_limit` downscale cannot bound it either, because `(w / p) * (h /
/// p)` is 0 whenever `p` exceeds the image and the downscale never engages.
/// Uncapped, a one-line `patch_size: 100000` asks for a ~200000x200000 RGB
/// buffer and OOM-aborts the process.
///
/// 128 is generous next to every ViT/MoonViT patch size in this tree (14 and 16
/// throughout, with a single 48), so the released geometry is untouched. It is
/// kept equal to the processor's backstop constant of the same name.
const MAX_PATCH_SIZE: usize = 128;

/// Upper bound on each `vision_config.merge_kernel_size` axis; the released
/// checkpoint declares `[2, 2]`.
///
/// The merge kernel sets the `merge * patch` pad unit the processor rounds up
/// to, so it alone fixes the floor on the buffer even a 1x1 image
/// materializes, and its product must stay far inside `i32` for the
/// merged-token arithmetic in
/// [`crate::multimodal::locateanything_prompt::merged_token_count`]. Real
/// spatial merges are 1, 2 (LocateAnything and Kimi-VL), or 4. Kept equal to
/// the processor's backstop constant of the same name.
const MAX_MERGE_KERNEL: usize = 16;

/// LocateAnything's MoonViT sub-config.
///
/// The keys are the HuggingFace spelling (`num_hidden_layers`, `hidden_size`,
/// `num_attention_heads`, `merge_kernel_size`), which upstream normalizes into
/// `depth` / `embed_dim` / `num_heads` / `spatial_merge_size` inside
/// `VisionConfig.__post_init__`. mlxcel's shared MoonViT config
/// ([`KimiVLVisionConfig`]) uses the normalized spelling, so the mapping is
/// done explicitly here rather than relying on defaults happening to agree.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct LocateAnythingVisionConfig {
    #[serde(default = "default_vision_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_init_pos_emb")]
    pub init_pos_emb_height: usize,
    #[serde(default = "default_init_pos_emb")]
    pub init_pos_emb_width: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_merge_kernel_size")]
    pub merge_kernel_size: Vec<usize>,
}

fn default_vision_model_type() -> String {
    "moonvit".to_string()
}
fn default_hidden_size() -> usize {
    1152
}
fn default_num_hidden_layers() -> usize {
    27
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_intermediate_size() -> usize {
    4304
}
fn default_patch_size() -> usize {
    14
}
fn default_init_pos_emb() -> usize {
    64
}
fn default_num_channels() -> usize {
    3
}
fn default_merge_kernel_size() -> Vec<usize> {
    vec![2, 2]
}

impl LocateAnythingVisionConfig {
    /// `[merge_h, merge_w]`, defaulting each axis to 1 rather than 0 so a
    /// malformed config cannot produce a zero-sized merge.
    pub(crate) fn merge_kernel(&self) -> [usize; 2] {
        let h = self.merge_kernel_size.first().copied().unwrap_or(2).max(1);
        let w = self.merge_kernel_size.get(1).copied().unwrap_or(h).max(1);
        [h, w]
    }

    /// Translate into mlxcel's shared MoonViT config, applying the two genuine
    /// LocateAnything deltas: LayerNorm eps `1e-5` and the tanh-approximate
    /// GELU in the block MLP.
    ///
    /// This is also the one place `patch_size` and `merge_kernel_size` are
    /// range-checked, and it **rejects** rather than clamps. Both values are
    /// consumed twice and the two consumers have to agree: the processor
    /// derives the patch grid from them ([`LocateAnythingProcessor::new`]),
    /// while the MoonViT conv patch-embed and patch merger are sized from the
    /// [`KimiVLVisionConfig`] built here. Silently clamping one side would
    /// leave the processor emitting a grid the tower does not apply, which
    /// turns a loud failure into a quiet mis-computation.
    pub(crate) fn to_moonvit_config(&self) -> Result<KimiVLVisionConfig, String> {
        if self.model_type != "moonvit" {
            return Err(format!(
                "LocateAnything expects vision_config.model_type == \"moonvit\", got \"{}\"",
                self.model_type
            ));
        }
        if !(1..=MAX_PATCH_SIZE).contains(&self.patch_size) {
            return Err(format!(
                "LocateAnything requires vision_config.patch_size in 1..={MAX_PATCH_SIZE}, got {}",
                self.patch_size
            ));
        }
        // Only axes the config actually declares are range-checked; a short or
        // absent `merge_kernel_size` is filled by `merge_kernel()`'s defaults
        // below. An explicitly declared 0 is refused rather than floored to 1,
        // because a silent 1x1 merge would quadruple the `<IMG_CONTEXT>` run
        // while the connector still expects 4608-wide merged rows.
        if self
            .merge_kernel_size
            .iter()
            .any(|&axis| !(1..=MAX_MERGE_KERNEL).contains(&axis))
        {
            return Err(format!(
                "LocateAnything requires every vision_config.merge_kernel_size axis in \
                 1..={MAX_MERGE_KERNEL}, got {:?}",
                self.merge_kernel_size
            ));
        }
        let merge = self.merge_kernel();
        // The shared MoonViT tower carries one `spatial_merge_size`. Upstream
        // does the same (`spatial_merge_size = merge_kernel_size[0]`), and the
        // patch merger it feeds is square, so a non-square kernel would be
        // silently squared. Refuse instead of mis-shaping the merge.
        if merge[0] != merge[1] {
            return Err(format!(
                "LocateAnything supports only a square merge_kernel_size; got {merge:?}"
            ));
        }
        Ok(KimiVLVisionConfig {
            model_type: self.model_type.clone(),
            depth: self.num_hidden_layers,
            embed_dim: self.hidden_size,
            hidden_size: self.hidden_size,
            num_heads: self.num_attention_heads,
            patch_size: self.patch_size,
            num_channels: self.num_channels,
            intermediate_size: self.intermediate_size,
            init_pos_emb_height: self.init_pos_emb_height,
            init_pos_emb_width: self.init_pos_emb_width,
            spatial_merge_size: merge[0],
            // MoonViT has no temporal axis; LocateAnything is image-only.
            temporal_patch_size: 1,
            layer_norm_eps: MOONVIT_LAYER_NORM_EPS,
            mlp_activation: MoonViTMlpActivation::GeluTanh,
            // The released checkpoint quantizes only `language_model.*`; the
            // tower and connector are plain bf16. Leaving these at 0 keeps
            // `UnifiedLinear::from_weights` on its unquantized branch unless a
            // future checkpoint actually ships quantized vision planes, in
            // which case the loader below inherits the top-level block.
            quant_group_size: 0,
            quant_bits: 0,
        })
    }
}

/// Load a LocateAnything (`locateanything`) VLM: MoonViT tower + MLP connector
/// + Qwen2 language model.
pub(crate) fn load_locateanything_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let vision_config_raw: LocateAnythingVisionConfig = parse_required_vlm_subconfig(
        &full_config,
        "vision_config",
        "LocateAnything vision config",
    )?;
    let merge_kernel = vision_config_raw.merge_kernel();
    let mut vision_config = vision_config_raw
        .to_moonvit_config()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Text config, inheriting a top-level `quantization` block when the text
    // sub-config does not carry its own (the released checkpoint stores it
    // once, above the sub-configs).
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in config.json"))?;
    if text_config_value.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(&mut text_config_value, "LocateAnything text_config")?
            .insert("quantization".to_string(), q.clone());
    }
    let text_args: models::llama3::ModelArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse LocateAnything text_config: {e}"))?;

    let group_size = text_args.group_size();
    let bits = text_args.bits();

    // Inherit the top-level quantization into the vision config only when the
    // checkpoint actually quantized a vision plane. Probing the weight map is
    // the reliable signal: the `quantization` block itself is present for the
    // text stack in every released conversion.
    let mut weights = load_vlm_weights_common(model_path, None)?;
    weights = remap_locateanything_weights(weights, text_args.tie_word_embeddings);
    if weights.contains_key("vision_tower.blocks.0.attn.wqkv.scales") {
        vision_config.quant_group_size = group_size;
        vision_config.quant_bits = bits;
    }

    // The released conversion is `mixed_4_8`, not uniform 4-bit: `embed_tokens`
    // and some layers' `v_proj` / `down_proj` are 8-bit while the rest is
    // 4-bit. Every per-tensor loader reconciles its own width from the tensor
    // shapes, so those are fine, but the fused QKV projection concatenates
    // q/k/v along axis 0 and therefore needs one shared width.
    //
    // This has to run before the bf16 -> f16 pass below, not after it: MLX's
    // `dequantize` returns an array carrying the scales' dtype, and the scales
    // are bf16, so every dense q/k/v plane inserted here is a *new* bf16
    // tensor. Densifying after the conversion pass would leave those planes
    // bf16 with nothing left to convert them.
    let quant_mode = full_config
        .get("quantization")
        .and_then(|q| q.get("mode"))
        .and_then(|m| m.as_str())
        .unwrap_or("affine")
        .to_string();
    let densified = densify_mixed_precision_qkv(&mut weights, group_size, bits, &quant_mode)
        .map_err(|e| {
            anyhow::anyhow!("Failed to normalize LocateAnything mixed-precision QKV: {e}")
        })?;
    if densified > 0 {
        tracing::info!(
            target: "mlxcel::quant",
            layers = densified,
            "LocateAnything: dequantized mixed-precision q/k/v planes so the fused QKV \
             projection can concatenate them"
        );
    }

    // Mixed precision: `load_vlm_weights_common` skipped its bf16 -> f16 pass
    // because the checkpoint declares quantization, so convert the remaining
    // plain-bf16 tensors (tower + connector + q/k/v linear biases, plus the
    // planes just densified above) here while keeping quantization
    // scales/biases at bf16. The densified planes no longer carry `.scales` /
    // `.biases` of their own, so the keep-predicate does not exempt them.
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            key.ends_with(".scales") || key.ends_with(".biases")
        });
        if had_bf16 {
            models::warn_bf16_precision();
        }
    }

    // Qwen2 text backbone. Keys arrive as `language_model.model.*`; the Qwen2
    // loader wants `model.*` / `lm_head.*`.
    let text_weights = super::strip_language_model_prefix(language_weights_subset(&weights));
    let text_model = models::Qwen2Model::from_weights(&text_weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load LocateAnything Qwen2 text model: {e}"))?;
    drop(text_weights);

    let vision_model = KimiVLVisionModel::from_weights(&weights, &vision_config, "vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load LocateAnything MoonViT tower: {e}"))?;

    let input_dim = connector_input_dim(vision_config.hidden_size, merge_kernel)?;
    let connector = LocateAnythingConnector::from_weights(
        &weights,
        "multi_modal_projector",
        input_dim,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load LocateAnything connector: {e}"))?;

    let in_token_limit = read_in_token_limit(model_path).unwrap_or(DEFAULT_IN_TOKEN_LIMIT);
    let processor =
        LocateAnythingProcessor::new(vision_config.patch_size, merge_kernel, in_token_limit);

    let added_tokens = std::fs::read_to_string(model_path.join("added_tokens.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok());

    // `image_token_index` is authoritative for the merge position; the framing
    // tokens are resolved from the tokenizer because the config does not carry
    // them.
    let image_token_id = full_config
        .get("image_token_index")
        .and_then(|v| v.as_i64())
        .map(|n| n as i32)
        .unwrap_or_else(|| {
            resolve_added_token_id(
                added_tokens.as_ref(),
                "<IMG_CONTEXT>",
                DEFAULT_IMAGE_TOKEN_INDEX,
            )
        });
    let img_start_token_id =
        resolve_added_token_id(added_tokens.as_ref(), "<img>", DEFAULT_IMG_START_TOKEN_ID);
    let img_end_token_id =
        resolve_added_token_id(added_tokens.as_ref(), "</img>", DEFAULT_IMG_END_TOKEN_ID);
    let eos_token_ids = resolve_eos_token_ids(&full_config, added_tokens.as_ref());

    let vlm = LocateAnythingVLM {
        text_model,
        vision_model,
        connector,
        processor,
        image_token_id,
        img_start_token_id,
        img_end_token_id,
        merge_kernel_size: merge_kernel,
        eos_token_ids,
    };

    Ok(LoadedModel::LocateAnythingVLM(vlm))
}

/// Width of one merged row the patch merger hands the connector:
/// `hidden_size * merge_h * merge_w` (4608 for the released checkpoint).
///
/// Both factors are read from `vision_config`, so the product is formed with
/// `checked_mul` and narrowed with `i32::try_from` instead of a bare
/// `as i32`. `hidden_size: 1152` with `merge_kernel_size: [65536, 65536]` is
/// `1152 * 2^32`, whose low 32 bits are zero, and an `input_dim` of 0 reaches
/// [`LocateAnythingConnector::forward`]'s `reshape(image_features, &[-1, 0])`
/// and throws inside MLX. A C++ throw across the cxx boundary aborts the
/// process rather than unwinding into a `Result`, so the value has to be
/// rejected here, while it is still an `Err` a caller can report.
fn connector_input_dim(hidden_size: usize, merge_kernel: [usize; 2]) -> Result<i32> {
    hidden_size
        .checked_mul(merge_kernel[0])
        .and_then(|v| v.checked_mul(merge_kernel[1]))
        .and_then(|v| i32::try_from(v).ok())
        .filter(|&width| width > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "LocateAnything connector input width must be a positive i32, but \
                 vision_config.hidden_size {hidden_size} times merge_kernel_size \
                 {merge_kernel:?} is not"
            )
        })
}

/// Collect only the `language_model.*` keys so the Qwen2 backbone loader never
/// sees vision / connector tensors.
fn language_weights_subset(weights: &WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (key, value) in weights.iter() {
        if key.starts_with("language_model.") {
            out.insert(key.clone(), mlxcel_core::copy(value));
        }
    }
    out
}

/// Apply the LocateAnything weight-key remapping (upstream `Model.sanitize`
/// plus the MoonViT `VisionModel.sanitize`).
///
/// `tie_word_embeddings` gates dropping `language_model.lm_head.weight`:
/// upstream drops it unconditionally because the family always ties, but a
/// checkpoint that did ship an untied head would then silently lose it.
pub(crate) fn remap_locateanything_weights(raw: WeightMap, tie_word_embeddings: bool) -> WeightMap {
    let mut out = WeightMap::with_capacity(raw.len());

    for (key, value) in raw.into_iter() {
        if key.contains("position_ids") || key.contains("rotary_emb") {
            continue;
        }
        if tie_word_embeddings && key == "language_model.lm_head.weight" {
            continue;
        }

        if key.starts_with("vision_model.") || key.starts_with("vision_tower.") {
            let (new_key, value) = transform_vision_key(&key, value);
            out.insert(new_key, value);
            continue;
        }

        if let Some(rest) = key.strip_prefix("mlp1.") {
            if let Some(tail) = rest.strip_prefix("0.") {
                out.insert(format!("multi_modal_projector.layer_norm.{tail}"), value);
            } else if let Some(tail) = rest.strip_prefix("1.") {
                out.insert(format!("multi_modal_projector.linear_1.{tail}"), value);
            } else if let Some(tail) = rest.strip_prefix("3.") {
                out.insert(format!("multi_modal_projector.linear_2.{tail}"), value);
            } else {
                out.insert(key, value);
            }
            continue;
        }

        out.insert(key, value);
    }

    out
}

fn transform_vision_key(
    key: &str,
    value: mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
) -> (String, mlxcel_core::UniquePtr<mlxcel_core::MlxArray>) {
    // Upstream rewrites `vision_model.encoder.` and `vision_model.` alike onto
    // `vision_tower.`; an already-converted checkpoint arrives as
    // `vision_tower.` and is left alone.
    let mut new_key = if let Some(rest) = key.strip_prefix("vision_model.encoder.") {
        format!("vision_tower.{rest}")
    } else if let Some(rest) = key.strip_prefix("vision_model.") {
        format!("vision_tower.{rest}")
    } else {
        key.to_string()
    };

    // Fused-qkv / output projections stored flat under the block.
    if new_key.contains("blocks") && !new_key.contains("attn") {
        if new_key.contains("wqkv") {
            new_key = new_key.replace("wqkv", "attn.wqkv");
        } else if new_key.contains("wo.") {
            new_key = new_key.replace("wo.", "attn.wo.");
        }
    }

    // Conv patch-embed kernel -> MLX channel-last layout.
    if new_key.ends_with("patch_embed.proj.weight") {
        let shape = mlxcel_core::array_shape(&value);
        if shape.len() == 4 && !conv2d_weight_is_channel_last(&shape) {
            let transposed = mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1]);
            return (new_key, mlxcel_core::copy(&transposed));
        }
    }

    (new_key, value)
}

/// Read `in_token_limit` from `preprocessor_config.json`, falling back to
/// `processor_config.json`.
fn read_in_token_limit(model_path: &Path) -> Option<usize> {
    for name in ["preprocessor_config.json", "processor_config.json"] {
        let Ok(content) = std::fs::read_to_string(model_path.join(name)) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&content) else {
            continue;
        };
        let direct = value.get("in_token_limit").and_then(|v| v.as_u64());
        let nested = value
            .get("image_processor")
            .and_then(|ip| ip.get("in_token_limit"))
            .and_then(|v| v.as_u64());
        if let Some(limit) = direct.or(nested) {
            return Some(limit as usize);
        }
    }
    None
}

fn resolve_added_token_id(added_tokens: Option<&Value>, name: &str, default: i32) -> i32 {
    added_tokens
        .and_then(|v| v.get(name))
        .and_then(|id| id.as_i64())
        .map(|id| id as i32)
        .unwrap_or(default)
}

/// Resolve the EOS/stop token ids. The `Llama3Model` trait default returns
/// Llama-3 ids, which are wrong for a Qwen2 backbone, so the correct ids must
/// be supplied here for the CLI and server stop paths.
fn resolve_eos_token_ids(full_config: &Value, added_tokens: Option<&Value>) -> Vec<i32> {
    let mut ids = Vec::new();

    let mut push_from = |value: Option<&Value>| match value {
        Some(Value::Number(n)) => {
            if let Some(id) = n.as_i64()
                && !ids.contains(&(id as i32))
            {
                ids.push(id as i32);
            }
        }
        Some(Value::Array(arr)) => {
            for id in arr.iter().filter_map(|v| v.as_i64()) {
                if !ids.contains(&(id as i32)) {
                    ids.push(id as i32);
                }
            }
        }
        _ => {}
    };

    push_from(full_config.get("eos_token_id"));
    push_from(
        full_config
            .get("text_config")
            .and_then(|tc| tc.get("eos_token_id")),
    );

    if let Some(im_end) = added_tokens
        .and_then(|v| v.get("<|im_end|>"))
        .and_then(|id| id.as_i64())
    {
        let im_end = im_end as i32;
        if !ids.contains(&im_end) {
            ids.push(im_end);
        }
    }

    if ids.is_empty() {
        ids = DEFAULT_EOS_TOKEN_IDS.to_vec();
    }
    ids
}

#[cfg(test)]
#[path = "vlm_locateanything_tests.rs"]
mod tests;
