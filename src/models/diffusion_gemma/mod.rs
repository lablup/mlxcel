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

//! DiffusionGemma block-diffusion model (issue #217, phases 1-2).
//!
//! `google/diffusiongemma-26B-A4B-it` reuses the Gemma 4 26B-A4B MoE text
//! backbone but generates by denoising a canvas of up to `canvas_length`
//! tokens per block instead of autoregressive decoding. This module composes
//! the existing [`crate::models::gemma4`] building blocks:
//!
//! * Encoder mode (prompt prefill + committed-block append): the standard
//!   causal Gemma 4 forward writing dense Fp16 KV caches, with each layer's
//!   output scalar taken from the checkpoint's per-layer ENCODER scalars.
//! * Canvas (decoder) mode: the noisy canvas attends bidirectionally within
//!   itself and to the cached encoder prefix (read-only), preceded by the
//!   self-conditioning GeGLU module.
//! * The block-diffusion generation engine lives in [`generate`].
//!
//! Reference: `references/mlx-vlm/mlx_vlm/models/diffusion_gemma/` and
//! `references/mlx-vlm/mlx_vlm/generate/diffusion.py`.
//!
//! Phase 2 adds image input: when the checkpoint ships the Gemma 4 vision
//! tower (`model.encoder.vision_tower.*`) and the multimodal embedder
//! (`model.encoder.embed_vision.*`), the prompt prefill runs as an
//! embeddings pass that masked-scatters projected image features into the
//! image-token positions and applies the `use_bidirectional_attention ==
//! "vision"` overlay. Text-only checkpoints load with [`DiffusionVision`] =
//! `None` and the text path stays byte-identical to phase 1. Server serving
//! is phase 3.

mod generate;

pub use generate::{
    DiffusionFinishReason, DiffusionGenerateOptions, DiffusionGenerationStats,
    DiffusionSamplerKind, diffusion_debug_canvas_enabled,
};

use crate::models::gemma4::{
    Gemma4TextModel, QuantizationArgs, RMSNormNoScale, RootQuantization, TextConfig,
    overlay_block_bidirectional, parse_eos_ids,
};
use crate::vision::encoders::gemma4::{Gemma4VisionConfig, Gemma4VisionModel};
use crate::vision::gemma4_multimodal_embedder::Gemma4MultimodalEmbedder;
use crate::vision::gemma4_unified_mask::{UnifiedTokenIds, compute_vision_block_ids};
use crate::vision::processors::gemma4::{Gemma4ImageInput, Gemma4Processor};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde::Deserialize;
use std::path::Path;

/// Default EOS ids for the DiffusionGemma chat checkpoints (same set as
/// Gemma 4: `<eos>`, `<end_of_turn>`, and the tool-call terminator).
const DEFAULT_EOS_TOKEN_IDS: [i32; 3] = [1, 106, 50];

/// Default canvas length when `config.json` omits `canvas_length`.
const DEFAULT_CANVAS_LENGTH: usize = 256;

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// Embedded `generation_config.sampler_config` object.
#[derive(Debug, Clone, Deserialize)]
struct SamplerConfigRaw {
    #[serde(rename = "_cls_name", default)]
    cls_name: Option<String>,
    #[serde(default)]
    entropy_bound: Option<f32>,
}

/// Raw embedded `generation_config` object inside `config.json`.
#[derive(Debug, Clone, Default, Deserialize)]
struct GenerationConfigRaw {
    #[serde(default)]
    confidence_threshold: Option<f32>,
    #[serde(default)]
    stability_threshold: Option<usize>,
    #[serde(default)]
    max_denoising_steps: Option<usize>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
    #[serde(default)]
    t_min: Option<f32>,
    #[serde(default)]
    t_max: Option<f32>,
    #[serde(default)]
    sampler_config: Option<SamplerConfigRaw>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

/// Early-stopping knobs (`_diffusion_stable_and_confident` in the
/// reference). Present only when the checkpoint's `generation_config`
/// carries at least one of the two keys; absent means no early stop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffusionStoppingConfig {
    /// Mean per-position entropy must fall below this for an early stop.
    pub confidence_threshold: f32,
    /// Number of consecutive identical argmax canvases required.
    pub stability_threshold: usize,
}

/// Resolved generation configuration parsed from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionGenerationConfig {
    pub max_denoising_steps: usize,
    pub max_new_tokens: usize,
    pub t_min: f32,
    pub t_max: f32,
    pub entropy_bound: f32,
    pub stopping: Option<DiffusionStoppingConfig>,
    pub eos_token_ids: Vec<i32>,
}

impl Default for DiffusionGenerationConfig {
    fn default() -> Self {
        Self {
            max_denoising_steps: 48,
            max_new_tokens: 256,
            t_min: 0.4,
            t_max: 0.8,
            entropy_bound: 0.1,
            stopping: None,
            eos_token_ids: Vec::new(),
        }
    }
}

impl DiffusionGenerationConfig {
    fn from_raw(raw: &GenerationConfigRaw) -> Result<Self, String> {
        if let Some(sampler) = &raw.sampler_config
            && let Some(name) = sampler.cls_name.as_deref()
            && name != "EntropyBoundSamplerConfig"
        {
            return Err(format!(
                "DiffusionGemma: unsupported sampler_config._cls_name {name:?} \
                 (only EntropyBoundSamplerConfig is supported)"
            ));
        }
        let stopping = if raw.confidence_threshold.is_some() || raw.stability_threshold.is_some() {
            Some(DiffusionStoppingConfig {
                confidence_threshold: raw.confidence_threshold.unwrap_or(0.005),
                stability_threshold: raw.stability_threshold.unwrap_or(1),
            })
        } else {
            None
        };
        Ok(Self {
            max_denoising_steps: raw.max_denoising_steps.unwrap_or(48),
            max_new_tokens: raw.max_new_tokens.unwrap_or(256),
            t_min: raw.t_min.unwrap_or(0.4),
            t_max: raw.t_max.unwrap_or(0.8),
            entropy_bound: raw
                .sampler_config
                .as_ref()
                .and_then(|s| s.entropy_bound)
                .unwrap_or(0.1),
            stopping,
            eos_token_ids: parse_eos_ids(raw.eos_token_id.as_ref()),
        })
    }
}

/// Default soft tokens contributed by one image (`config.json` omits it).
const DEFAULT_VISION_SOFT_TOKENS_PER_IMAGE: usize = 280;

/// Top-level `config.json` arguments for `model_type == "diffusion_gemma"`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub text_config: serde_json::Value,
    #[serde(default)]
    pub canvas_length: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    generation_config: Option<GenerationConfigRaw>,
    #[serde(default)]
    pub quantization: Option<RootQuantization>,
    // Vision front-end (phase 2). All optional: text-only exports omit them.
    #[serde(default)]
    pub boi_token_id: Option<i32>,
    #[serde(default)]
    pub eoi_token_id: Option<i32>,
    #[serde(default)]
    pub image_token_id: Option<i32>,
    #[serde(default)]
    pub video_token_id: Option<i32>,
    #[serde(default)]
    pub vision_soft_tokens_per_image: Option<usize>,
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
}

impl ModelArgs {
    /// Parse and normalize the nested `text_config` into a Gemma 4
    /// [`TextConfig`].
    ///
    /// The DiffusionGemma `text_config` is shape-identical to
    /// `gemma-4-26b-a4b-it` but OMITS the presence-only flags that
    /// checkpoint sets, so the serde defaults would silently disable two
    /// structural features the weights require:
    ///
    /// * `attention_k_eq_v = true`: the 5 full-attention layers carry no
    ///   `v_proj` weights (values are the RMSNormNoScale of the raw K
    ///   projection), exactly like the upstream reference's
    ///   `v_proj = None if not sliding` construction.
    /// * `enable_moe_block = true`: every layer has the dual dense-MLP +
    ///   MoE feed-forward.
    ///
    /// Both are forced here after parsing. The KV-sharing / per-layer-input
    /// E-series features are explicitly zeroed because DiffusionGemma never
    /// uses them.
    pub fn text_args(&self) -> Result<TextConfig, String> {
        let mut config: TextConfig = serde_json::from_value(self.text_config.clone())
            .map_err(|e| format!("DiffusionGemma: failed to parse text_config: {e}"))?;
        config.attention_k_eq_v = true;
        config.enable_moe_block = true;
        config.num_kv_shared_layers = 0;
        config.use_double_wide_mlp = false;
        config.vocab_size_per_layer_input = 0;
        config.hidden_size_per_layer_input = 0;
        if config.quantization.is_none()
            && let Some(ref q) = self.quantization
        {
            config.quantization = Some(QuantizationArgs {
                group_size: q.group_size,
                bits: q.bits,
            });
        }
        Ok(config)
    }

    pub fn generation_config(&self) -> Result<DiffusionGenerationConfig, String> {
        match &self.generation_config {
            Some(raw) => DiffusionGenerationConfig::from_raw(raw),
            None => Ok(DiffusionGenerationConfig::default()),
        }
    }

    pub fn canvas_length(&self) -> usize {
        self.canvas_length.unwrap_or(DEFAULT_CANVAS_LENGTH)
    }

    pub fn image_token_id(&self) -> i32 {
        self.image_token_id.unwrap_or(258_880)
    }

    pub fn boi_token_id(&self) -> i32 {
        self.boi_token_id.unwrap_or(255_999)
    }

    pub fn eoi_token_id(&self) -> i32 {
        self.eoi_token_id.unwrap_or(258_882)
    }

    /// Video token id, or `-1` when the checkpoint has none (DiffusionGemma
    /// chat checkpoints set `video_token_id: null`). `-1` never matches a
    /// real token id, so it cleanly disables the video branch.
    pub fn video_token_id(&self) -> i32 {
        self.video_token_id.unwrap_or(-1)
    }

    pub fn vision_soft_tokens_per_image(&self) -> usize {
        self.vision_soft_tokens_per_image
            .unwrap_or(DEFAULT_VISION_SOFT_TOKENS_PER_IMAGE)
    }

    /// Parse the nested `vision_config` into a [`Gemma4VisionConfig`], or
    /// `None` when the checkpoint omits it (text-only export).
    pub fn vision_config(&self) -> Result<Option<Gemma4VisionConfig>, String> {
        match &self.vision_config {
            Some(value) => {
                let config: Gemma4VisionConfig = serde_json::from_value(value.clone())
                    .map_err(|e| format!("DiffusionGemma: failed to parse vision_config: {e}"))?;
                Ok(Some(config))
            }
            None => Ok(None),
        }
    }

    /// EOS ids: union of the top-level `eos_token_id`, the embedded
    /// generation_config `eos_token_id`, and the Gemma 4 defaults.
    pub fn eos_token_ids(&self, generation: &DiffusionGenerationConfig) -> Vec<i32> {
        let mut ids = parse_eos_ids(self.eos_token_id.as_ref());
        for &id in &generation.eos_token_ids {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        if ids.is_empty() {
            ids.extend_from_slice(&DEFAULT_EOS_TOKEN_IDS);
        }
        ids
    }
}

// ---------------------------------------------------------------------------
// Weight remapping (fused gate_up experts -> gemma4 SwitchGeGLU layout)
// ---------------------------------------------------------------------------

/// Split a fused `gate_up` expert tensor `[num_experts, 2 * moe_dim, K]`
/// along the OUTPUT axis into `(gate, up)` halves of `[num_experts, moe_dim,
/// K]` each (gate first, then up, matching the reference
/// `gate = gate_up[..., :moe]; up = gate_up[..., moe:]`).
///
/// Affine quantization is per-output-row (weight `[E, out, packed_in]`,
/// scales/biases `[E, out, in / group_size]`), so this split is numerically
/// exact for the packed weight, scales, and biases alike.
///
/// The split is performed on HOST bytes, building pristine dense arrays via
/// `from_bytes`. The obvious `slice` + `copy` graph route produces arrays
/// that `gather_qmm` reads incorrectly on the pinned MLX (out-of-bounds
/// style corruption that turns nondeterministic under allocator churn);
/// gather_qmm with byte-identical rebuilt buffers is deterministic, so the
/// host-side rebuild is load-bearing, not an optimization. This runs once
/// per layer at load time.
pub(crate) fn split_gate_up_tensor(
    tensor: &MlxArray,
    moe_dim: i32,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
    let shape = mlxcel_core::array_shape(tensor);
    if shape.len() != 3 {
        return Err(format!(
            "DiffusionGemma: fused gate_up tensor must be rank 3, got shape {shape:?}"
        ));
    }
    if shape[1] != 2 * moe_dim {
        return Err(format!(
            "DiffusionGemma: fused gate_up output dim {} does not match 2 * moe_intermediate_size \
             ({})",
            shape[1],
            2 * moe_dim
        ));
    }
    let dtype = mlxcel_core::array_dtype(tensor);
    let element_size = match dtype {
        d if d == dtype::UINT32 || d == dtype::INT32 || d == dtype::FLOAT32 => 4usize,
        d if d == dtype::FLOAT16 || d == dtype::BFLOAT16 => 2usize,
        other => {
            return Err(format!(
                "DiffusionGemma: unsupported fused gate_up dtype {other}"
            ));
        }
    };
    mlxcel_core::eval(tensor);
    let bytes = mlxcel_core::array_to_raw_bytes(tensor);
    let (num_experts, fused_rows, cols) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let half_rows = moe_dim as usize;
    let row_bytes = cols * element_size;
    let expected = num_experts * fused_rows * row_bytes;
    if bytes.len() != expected {
        return Err(format!(
            "DiffusionGemma: fused gate_up byte size mismatch (got {}, expected {expected})",
            bytes.len()
        ));
    }
    let half_bytes = half_rows * row_bytes;
    let mut gate_bytes = Vec::with_capacity(num_experts * half_bytes);
    let mut up_bytes = Vec::with_capacity(num_experts * half_bytes);
    for expert in 0..num_experts {
        let base = expert * fused_rows * row_bytes;
        gate_bytes.extend_from_slice(&bytes[base..base + half_bytes]);
        up_bytes.extend_from_slice(&bytes[base + half_bytes..base + 2 * half_bytes]);
    }
    let half_shape = [shape[0], moe_dim, shape[2]];
    // 16-bit dtypes must go through the f16 constructor: the generic
    // `from_bytes` path reads half the bytes for them (see
    // `from_bytes_f16` docs / the #125 serde corruption fix).
    let build = |data: &[u8]| -> UniquePtr<MlxArray> {
        if dtype == dtype::BFLOAT16 {
            mlxcel_core::from_bytes_f16(data, &half_shape, true)
        } else if dtype == dtype::FLOAT16 {
            mlxcel_core::from_bytes_f16(data, &half_shape, false)
        } else {
            mlxcel_core::from_bytes(data, &half_shape, dtype)
        }
    };
    let gate = build(&gate_bytes);
    let up = build(&up_bytes);
    mlxcel_core::eval(&gate);
    mlxcel_core::eval(&up);
    Ok((gate, up))
}

/// Rewrite the checkpoint's fused expert tensors onto the key layout
/// `gemma4::Experts::from_weights` expects:
///
/// * `…experts.gate_up_proj.{weight,scales,biases}` is split along the
///   output axis into `…experts.switch_glu.gate_proj.*` (rows `0..moe`) and
///   `…experts.switch_glu.up_proj.*` (rows `moe..2*moe`).
/// * `…experts.down_proj.*` is aliased to `…experts.switch_glu.down_proj.*`.
fn remap_fused_expert_weights(
    weights: &mut WeightMap,
    num_layers: usize,
    moe_dim: i32,
) -> Result<(), String> {
    for layer_idx in 0..num_layers {
        let prefix = format!("model.decoder.layers.{layer_idx}.experts");
        for suffix in ["weight", "scales", "biases"] {
            let fused_key = format!("{prefix}.gate_up_proj.{suffix}");
            if let Some(fused) = weights.remove(&fused_key) {
                let (gate, up) = split_gate_up_tensor(&fused, moe_dim)
                    .map_err(|e| format!("{e} (key: {fused_key})"))?;
                weights.insert(format!("{prefix}.switch_glu.gate_proj.{suffix}"), gate);
                weights.insert(format!("{prefix}.switch_glu.up_proj.{suffix}"), up);
            }
            let down_key = format!("{prefix}.down_proj.{suffix}");
            if let Some(down) = weights.remove(&down_key) {
                weights.insert(format!("{prefix}.switch_glu.down_proj.{suffix}"), down);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Self-conditioning module
// ---------------------------------------------------------------------------

/// Self-conditioning GeGLU MLP (`model.decoder.self_conditioning.*`).
///
/// Mirrors `diffusion_gemma.language.SelfConditioning`:
/// `signal = down_proj(gelu_approx(gate_proj(pre_norm(s))) * up_proj(pre_norm(s)))`
/// and `output = RMSNormNoScale(inputs_embeds + signal)`.
///
/// The post-norm applies even when the soft-embedding signal is absent: a
/// zero signal yields exactly `down_proj(gelu_approx(0) * 0) == 0` (no bias
/// terms anywhere in the chain), so the `None` fast path skips the MLP but
/// still normalizes.
pub(crate) struct SelfConditioning {
    pre_norm: RMSNorm,
    post_norm: RMSNormNoScale,
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl SelfConditioning {
    fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let pre_norm_key = format!("{prefix}.pre_norm.weight");
        let pre_norm_weight = weights
            .get(&pre_norm_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {pre_norm_key}"))?;
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);
        Ok(Self {
            pre_norm: RMSNorm::new(pre_norm_weight, config.rms_norm_eps),
            post_norm: RMSNormNoScale::new(config.hidden_size as i32, config.rms_norm_eps),
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
        })
    }

    pub(crate) fn forward(
        &self,
        inputs_embeds: &MlxArray,
        soft_embeddings: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        match soft_embeddings {
            None => self.post_norm.forward(inputs_embeds),
            Some(soft) => {
                let soft = mlxcel_core::astype(soft, mlxcel_core::array_dtype(inputs_embeds));
                let normed = self.pre_norm.forward(&soft);
                let gate = self.gate_proj.forward(&normed);
                let up = self.up_proj.forward(&normed);
                let activated = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
                let signal = self.down_proj.forward(&activated);
                self.post_norm
                    .forward(&mlxcel_core::add(inputs_embeds, &signal))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Vision front-end (phase 2)
// ---------------------------------------------------------------------------

/// DiffusionGemma vision front-end: the Gemma 4 vision tower plus the
/// multimodal embedder that projects tower features into the language
/// model's embedding space (`model.encoder.vision_tower.*`,
/// `model.encoder.embed_vision.*`).
///
/// Present only when the checkpoint ships vision weights; text-only exports
/// leave [`DiffusionGemmaModel::vision`] as `None`.
pub struct DiffusionVision {
    vision_tower: Gemma4VisionModel,
    embed_vision: Gemma4MultimodalEmbedder,
    pub image_token_id: i32,
    pub boi_token_id: i32,
    pub eoi_token_id: i32,
    pub video_token_id: i32,
    pub soft_tokens_per_image: usize,
}

impl DiffusionVision {
    /// Run the vision tower and the multimodal embedder for one preprocessed
    /// image, returning projected features `[1, num_soft_tokens, hidden]`.
    fn image_features(&self, image: &Gemma4ImageInput) -> UniquePtr<MlxArray> {
        let pixel_values = image
            .pixel_values
            .as_ref()
            .expect("DiffusionGemma image pixel_values must be non-null");
        let tower = self.vision_tower.forward(pixel_values, image.patch_grid);
        self.embed_vision.forward(&tower)
    }
}

/// Vision inputs prepared for one diffusion prefill: the scattered,
/// embed-scaled `inputs_embeds` for the expanded prompt and the per-position
/// vision block ids that drive the bidirectional overlay (or `None` when no
/// vision block is present / the overlay is disabled).
pub struct DiffusionVisionPrefill {
    pub(crate) inputs_embeds: UniquePtr<MlxArray>,
    pub(crate) vision_block_ids: Option<Vec<i32>>,
}

/// Result of preparing image input for one diffusion prompt: the expanded
/// prompt ids (with each image placeholder rewritten to
/// `boi + image_token * N + eoi`) and the matching vision prefill.
///
/// Shared by the CLI `generate` driver
/// ([`crate::commands::generate_diffusion`]) and the server diffusion worker
/// so the two surfaces preprocess images, expand the prompt, and build the
/// prefill through one code path.
pub struct PreparedDiffusionImagePrompt {
    pub expanded_ids: Vec<i32>,
    pub prefill: DiffusionVisionPrefill,
    /// Soft-token count contributed by each input image (in prompt order).
    pub num_soft_tokens: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// DiffusionGemma block-diffusion text model.
pub struct DiffusionGemmaModel {
    pub(crate) text: Gemma4TextModel,
    pub(crate) self_conditioning: SelfConditioning,
    /// Per-layer ENCODER output scalars
    /// (`model.encoder.language_model.layers.N.layer_scalar`).
    pub(crate) encoder_layer_scalars: Vec<UniquePtr<MlxArray>>,
    pub(crate) canvas_length: usize,
    pub(crate) generation_config: DiffusionGenerationConfig,
    pub(crate) eos_token_ids: Vec<i32>,
    pub(crate) embed_scale: f32,
    /// Vision front-end, present only when the checkpoint ships a tower.
    pub(crate) vision: Option<DiffusionVision>,
    /// Whether `text_config.use_bidirectional_attention == "vision"`: gates
    /// the bidirectional vision-block overlay on the encoder prefill.
    pub(crate) use_bidirectional_vision: bool,
    _weight_backing: super::sanitize::Gemma4WeightBacking,
}

impl DiffusionGemmaModel {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;

        // `use_clipped_linears` from the vision config decides whether the
        // tower's clipped-linear calibration tensors are kept. Text-only
        // checkpoints have no vision config; default to false (drop them).
        let use_clipped_linears = args
            .vision_config()?
            .map(|vc| vc.use_clipped_linears)
            .unwrap_or(false);
        let (mut weights, weight_backing) =
            super::sanitize::load_diffusion_gemma_weights_with_backing(
                model_dir,
                use_clipped_linears,
            )?;
        let mut model = Self::from_weights(&mut weights, &args)?;
        model._weight_backing = weight_backing;
        Ok(model)
    }

    /// Build the model from an already-loaded (and ownable) weight map.
    ///
    /// Takes `&mut` because the fused expert tensors are remapped in place
    /// onto the gemma4 SwitchGeGLU key layout before construction.
    pub fn from_weights(weights: &mut WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let config = args.text_args()?;
        let moe_dim = config.moe_intermediate_size.ok_or_else(|| {
            "DiffusionGemma: text_config.moe_intermediate_size is required".to_string()
        })? as i32;
        remap_fused_expert_weights(weights, config.num_hidden_layers, moe_dim)?;

        let text = Gemma4TextModel::from_weights(weights, &config, "model.decoder")?;
        let self_conditioning =
            SelfConditioning::from_weights(weights, &config, "model.decoder.self_conditioning")?;

        let mut encoder_layer_scalars = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let key = format!("model.encoder.language_model.layers.{layer_idx}.layer_scalar");
            let scalar = weights
                .get(&key)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Weight not found: {key}"))?;
            encoder_layer_scalars.push(scalar);
        }

        let generation_config = args.generation_config()?;
        let eos_token_ids = args.eos_token_ids(&generation_config);
        let embed_scale = (config.hidden_size as f32).sqrt();

        // Vision front-end: build it only when both a parsed vision config and
        // the tower weights are present. A text-only export (no vision_config,
        // no `model.encoder.vision_tower.*` keys) leaves `vision` as None.
        let vision = Self::build_vision(weights, &config, args)?;
        let use_bidirectional_vision = config
            .use_bidirectional_attention
            .as_deref()
            .map(|s| s == "vision")
            .unwrap_or(false);

        Ok(Self {
            text,
            self_conditioning,
            encoder_layer_scalars,
            canvas_length: args.canvas_length(),
            generation_config,
            eos_token_ids,
            embed_scale,
            vision,
            use_bidirectional_vision,
            _weight_backing: super::sanitize::Gemma4WeightBacking::default(),
        })
    }

    /// Build the optional vision front-end from `model.encoder.vision_tower.*`
    /// and `model.encoder.embed_vision.*`. Returns `None` for text-only
    /// checkpoints (no `vision_config`, or no tower weights present).
    fn build_vision(
        weights: &WeightMap,
        config: &TextConfig,
        args: &ModelArgs,
    ) -> Result<Option<DiffusionVision>, String> {
        let Some(vision_config) = args.vision_config()? else {
            return Ok(None);
        };
        // Tolerate a vision_config without tower weights (text-only shard set):
        // the projection weight is the canonical marker of a real front-end.
        if !weights.contains_key("model.encoder.embed_vision.embedding_projection.weight") {
            return Ok(None);
        }

        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let vision_tower = Gemma4VisionModel::from_weights(
            weights,
            "model.encoder.vision_tower",
            &vision_config,
            group_size,
            bits,
        )?;
        let embed_vision = Gemma4MultimodalEmbedder::from_weights(
            weights,
            "model.encoder.embed_vision",
            vision_config.hidden_size,
            vision_config.rms_norm_eps(),
            group_size,
            bits,
        )?;

        Ok(Some(DiffusionVision {
            vision_tower,
            embed_vision,
            image_token_id: args.image_token_id(),
            boi_token_id: args.boi_token_id(),
            eoi_token_id: args.eoi_token_id(),
            video_token_id: args.video_token_id(),
            soft_tokens_per_image: args.vision_soft_tokens_per_image(),
        }))
    }

    /// Whether this checkpoint loaded a vision front-end (phase 2 capable).
    pub fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    /// Vision token ids / soft-token count for prompt expansion, or `None`
    /// for a text-only checkpoint.
    pub fn vision(&self) -> Option<&DiffusionVision> {
        self.vision.as_ref()
    }

    pub fn config(&self) -> &TextConfig {
        &self.text.config
    }

    pub fn generation_config(&self) -> &DiffusionGenerationConfig {
        &self.generation_config
    }

    pub fn canvas_length(&self) -> usize {
        self.canvas_length
    }

    /// Allocate one dense Fp16 [`KVCache`] per layer.
    ///
    /// Phase 1 intentionally uses dense Fp16 caches for BOTH layer families
    /// (sliding behavior is enforced by masks / the canvas-side trim, exactly
    /// like the offset-aligned dynamic cache in the reference). A
    /// `RotatingKVCache` optimization and quantized KV modes are out of
    /// scope for this phase.
    pub fn make_diffusion_caches(&self) -> Vec<KVCache> {
        (0..self.text.layers.len())
            .map(|_| KVCache::new())
            .collect()
    }

    /// Encoder-mode forward: causal prefill of `input_ids` into `caches`,
    /// with each layer scaled by its ENCODER scalar.
    ///
    /// Returns the final pre-norm hidden state for the [`LanguageModel`]
    /// trait path; the diffusion engine ignores it (only the KV-cache writes
    /// matter — the reference never consumes the encoder hidden output, so
    /// callers should not project it unless they need trait-compatible
    /// logits).
    pub(crate) fn forward_encoder(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let embeds = self.text.embed_tokens.forward(input_ids);
        let embeds = mlxcel_core::multiply_scalar(&embeds, self.embed_scale);
        // Text path: no vision-block overlay. Byte-identical to the phase-1
        // forward_encoder for an all-text prompt.
        self.forward_encoder_embeds(&embeds, caches, mask, None)
    }

    /// Embeddings-input encoder forward (issue #217, phase 2).
    ///
    /// `inputs_embeds` is the already-`embed_scale`-scaled (and, for vision,
    /// feature-scattered) hidden state `[1, L, hidden]`. `vision_block_ids`,
    /// when present, is the per-position vision block-id vector that drives
    /// the bidirectional overlay (OR-ed on top of the causal / sliding-window
    /// masks for every layer), matching `_vision_block_overlay` /
    /// `_make_encoder_masks` in the reference. The overlay applies only when
    /// this routine builds its own masks (`mask == None`), i.e. the prompt
    /// prefill at cache offset 0; continuation encoder passes over committed
    /// blocks pass `vision_block_ids == None`.
    pub(crate) fn forward_encoder_embeds(
        &self,
        inputs_embeds: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        vision_block_ids: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs_embeds);
        let l = shape[1];
        let offset = caches.first().map(|c| c.offset).unwrap_or(0);
        let window = self.text.config.sliding_window as i32;

        // Always build explicit dense-axis masks: the dense Fp16 caches keep
        // the FULL key axis [0, offset + l), so the rotating-cache-shaped
        // helpers (which cap the sliding mask to the window width) do not
        // apply here.
        let (mut global_mask, mut sliding_mask) = match mask {
            Some(m) => (mlxcel_core::copy(m), mlxcel_core::copy(m)),
            None => (
                mlxcel_core::utils::create_causal_mask(l, offset),
                dense_windowed_causal_mask(l, offset, window),
            ),
        };

        // Bidirectional vision-block overlay: every contiguous image block
        // attends to itself in both directions, OR-ed onto the causal /
        // sliding base masks for ALL layers.
        if let Some(block_ids) = vision_block_ids {
            let ids = mlxcel_core::from_slice_i32(block_ids, &[block_ids.len() as i32]);
            global_mask = overlay_block_bidirectional(&global_mask, &ids);
            sliding_mask = overlay_block_bidirectional(&sliding_mask, &ids);
        }

        let mut h: Option<UniquePtr<MlxArray>> = None;
        for (i, layer) in self.text.layers.iter().enumerate() {
            let local_mask = if layer.layer_type == "full_attention" {
                &global_mask
            } else {
                &sliding_mask
            };
            let input = h
                .as_ref()
                .map(|p| p.as_ref().expect("non-null encoder hidden") as &MlxArray)
                .unwrap_or(inputs_embeds);
            let out = layer.forward_encoder_with_scalar(
                input,
                Some(local_mask),
                &mut caches[i],
                &self.encoder_layer_scalars[i],
            );
            h = Some(out);
        }
        h.unwrap_or_else(|| mlxcel_core::copy(inputs_embeds))
    }

    /// Build the embed-scaled, feature-scattered `inputs_embeds` for one
    /// vision prompt and the per-position vision block ids for the overlay.
    ///
    /// Mirrors `EncoderModel._embed_inputs`: image-token positions are
    /// replaced by `pad_token_id` for the embedding lookup, the text
    /// embeddings are scaled by `embed_scale`, and the RAW (unscaled) image
    /// features are masked-scattered into the image positions.
    ///
    /// `expanded_ids` is the prompt after image-placeholder expansion
    /// (`boi + image_token * soft_tokens + eoi`); batch is always 1.
    pub fn prepare_vision_prefill(
        &self,
        expanded_ids: &[i32],
        images: &[Gemma4ImageInput],
    ) -> Result<DiffusionVisionPrefill, String> {
        let vision = self.vision.as_ref().ok_or_else(|| {
            "DiffusionGemma: this checkpoint has no vision tower; image input is unsupported"
                .to_string()
        })?;
        if expanded_ids.is_empty() {
            return Err("DiffusionGemma: prompt must contain at least one token".to_string());
        }

        let l = expanded_ids.len() as i32;
        let hidden = self.text.config.hidden_size as i32;

        // Pad image/video positions to id 0 for the lookup (Gemma pad_token_id
        // == 0); those rows are overwritten by the scatter below.
        let lookup_ids: Vec<i32> = expanded_ids
            .iter()
            .map(|&id| {
                if id == vision.image_token_id || id == vision.video_token_id {
                    0
                } else {
                    id
                }
            })
            .collect();
        let lookup_arr = mlxcel_core::from_slice_i32(&lookup_ids, &[1, l]);
        let text_embeds = self.text.embed_tokens.forward(&lookup_arr);
        let inputs_embeds = mlxcel_core::multiply_scalar(&text_embeds, self.embed_scale);

        // Image features (raw embedder output, NOT scaled by embed_scale),
        // concatenated along the sequence axis in prompt order.
        let mut features: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(images.len());
        for image in images {
            features.push(vision.image_features(image));
        }
        if features.is_empty() {
            return Err("DiffusionGemma: image input requested but no images supplied".to_string());
        }
        let mut merged = mlxcel_core::copy(features[0].as_ref().expect("non-null image features"));
        for next in features.iter().skip(1) {
            merged = mlxcel_core::concatenate(
                &merged,
                next.as_ref().expect("non-null image features"),
                1,
            );
        }
        let merged = mlxcel_core::astype(&merged, mlxcel_core::array_dtype(&inputs_embeds));

        // Vision mask broadcast to the embedding shape, then masked-scatter.
        let is_vision: Vec<i32> = expanded_ids
            .iter()
            .map(|&id| i32::from(id == vision.image_token_id || id == vision.video_token_id))
            .collect();

        // The projected features must line up exactly with the image-token
        // positions. `masked_scatter` wraps source indices modulo the feature
        // count, so a mismatch (e.g. a config that points `image_token_id` at a
        // common text token, or a tower/processor soft-token disagreement)
        // would silently scatter the wrong rows rather than fault. Reject it.
        let vision_positions = is_vision.iter().filter(|&&v| v != 0).count();
        let feature_count = mlxcel_core::array_shape(&merged)
            .get(1)
            .copied()
            .unwrap_or(0) as usize;
        if vision_positions != feature_count {
            return Err(format!(
                "DiffusionGemma: image-token positions ({vision_positions}) do not match                  projected vision features ({feature_count}); check image_token_id and the                  soft-token expansion"
            ));
        }

        let mask_2d = mlxcel_core::from_slice_i32(&is_vision, &[1, l]);
        let mask_bool = mlxcel_core::astype(&mask_2d, dtype::BOOL);
        let mask_expanded =
            mlxcel_core::broadcast_to(&mlxcel_core::expand_dims(&mask_bool, -1), &[1, l, hidden]);
        let inputs_embeds = crate::vision::gemma4_multimodal_embedder::masked_scatter(
            &inputs_embeds,
            &mask_expanded,
            &merged,
        );

        // Per-position vision block ids for the bidirectional overlay.
        let vision_block_ids = if self.use_bidirectional_vision {
            compute_vision_block_ids(
                expanded_ids,
                UnifiedTokenIds {
                    image: vision.image_token_id,
                    video: vision.video_token_id,
                    audio: -1,
                },
                true,
            )
        } else {
            None
        };

        Ok(DiffusionVisionPrefill {
            inputs_embeds,
            vision_block_ids,
        })
    }

    /// Build the Gemma 4 image processor for this checkpoint.
    ///
    /// The DiffusionGemma `image_processor` is identical to the gemma-4 VLM
    /// family (size 224x224, patch 16, pooling_kernel_size 3, 280 soft
    /// tokens). Values are read from `processor_config.json` when present and
    /// fall back to those defaults otherwise. Returns `None` for a text-only
    /// checkpoint (no vision tower).
    pub fn build_image_processor(&self, model_dir: &Path) -> Option<Gemma4Processor> {
        let vision = self.vision.as_ref()?;
        let image_cfg = std::fs::read_to_string(model_dir.join("processor_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|cfg| cfg.get("image_processor").cloned());
        let get = |key: &str, default: usize| -> usize {
            image_cfg
                .as_ref()
                .and_then(|cfg| cfg.get(key))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        Some(Gemma4Processor::new(
            get("patch_size", 16),
            get("max_soft_tokens", vision.soft_tokens_per_image),
            get("pooling_kernel_size", 3),
        ))
    }

    /// Preprocess decoded images, expand the prompt with image placeholder
    /// tokens, and build the vision prefill.
    ///
    /// Shared by the CLI `generate` driver and the server diffusion worker:
    /// both decode their images upstream (CLI from `--image` paths, server
    /// from request bytes) and hand the [`image::DynamicImage`] values here.
    /// Errors when the checkpoint has no vision tower or no images are
    /// supplied.
    pub fn prepare_image_prompt(
        &self,
        model_dir: &Path,
        images: &[image::DynamicImage],
        prompt_tokens: &[i32],
    ) -> Result<PreparedDiffusionImagePrompt, String> {
        let vision = self.vision.as_ref().ok_or_else(|| {
            "This DiffusionGemma checkpoint does not include a vision tower; \
             run with a text-only prompt"
                .to_string()
        })?;
        if images.is_empty() {
            return Err("DiffusionGemma: image input requested but no images supplied".to_string());
        }
        let processor = self
            .build_image_processor(model_dir)
            .ok_or_else(|| "DiffusionGemma: this checkpoint has no vision tower".to_string())?;
        let processed: Vec<Gemma4ImageInput> = processor.preprocess(images);
        let num_soft_tokens: Vec<usize> = processed.iter().map(|img| img.num_soft_tokens).collect();

        let mut expanded_ids = prompt_tokens.to_vec();
        crate::vlm_runtime::expand_gemma4_image_tokens_pub(
            &mut expanded_ids,
            vision.image_token_id,
            vision.boi_token_id,
            vision.eoi_token_id,
            &num_soft_tokens,
        )
        .map_err(|e| format!("{e}"))?;

        let prefill = self.prepare_vision_prefill(&expanded_ids, &processed)?;
        Ok(PreparedDiffusionImagePrompt {
            expanded_ids,
            prefill,
            num_soft_tokens,
        })
    }

    /// Canvas (decoder-mode) forward: denoise one canvas against the
    /// read-only encoder prefix in `caches` and return softcapped logits
    /// `[1, canvas_len, vocab]`.
    ///
    /// `self_conditioning_embeddings` is the previous denoising step's soft
    /// embedding signal (`softmax(logits) @ embed_table * embed_scale`), or
    /// `None` on the first step (which still applies the self-conditioning
    /// post-norm — never skip the module).
    pub(crate) fn forward_canvas(
        &self,
        canvas_ids: &MlxArray,
        caches: &[KVCache],
        self_conditioning_embeddings: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let embeds = self.text.embed_tokens.forward(canvas_ids);
        let embeds = mlxcel_core::multiply_scalar(&embeds, self.embed_scale);
        let mut h = self
            .self_conditioning
            .forward(&embeds, self_conditioning_embeddings);

        let offset = caches.first().map(|c| c.offset).unwrap_or(0);
        for (layer, cache) in self.text.layers.iter().zip(caches.iter()) {
            let encoder_kv = cache.visible_state();
            let encoder_kv_refs = encoder_kv.as_ref().map(|(k, v)| {
                (
                    k.as_ref().expect("non-null encoder keys") as &MlxArray,
                    v.as_ref().expect("non-null encoder values") as &MlxArray,
                )
            });
            h = layer.forward_canvas(&h, encoder_kv_refs, offset);
        }

        let h = self.text.norm.forward(&h);
        let mut logits = self.text.embed_tokens.as_linear(&h);
        if let Some(cap) = self.text.config.final_logit_softcapping {
            logits = mlxcel_core::compiled_softcap(&logits, cap);
        }
        logits
    }
}

/// Build a `[size, size + offset]` additive causal mask with a sliding-window
/// lower bound over the FULL dense key axis.
///
/// Unlike [`mlxcel_core::utils::create_causal_mask_with_window`], this never
/// caps the key axis to the window width: the diffusion encoder keeps every
/// position resident in a dense [`KVCache`], so column `k` always maps to
/// logical key position `k`. Query row `j` (logical position `offset + j`)
/// may attend key `k` iff `k <= offset + j` (causal) and
/// `k > offset + j - window` (window lower bound).
pub(crate) fn dense_windowed_causal_mask(
    size: i32,
    offset: i32,
    window: i32,
) -> UniquePtr<MlxArray> {
    let total_len = size + offset;
    let ones = mlxcel_core::ones(&[size, total_len], dtype::FLOAT32);
    let causal = mlxcel_core::tril(&ones, offset);
    let band = mlxcel_core::triu(&ones, offset - window + 1);
    let allowed = mlxcel_core::multiply(&causal, &band);

    let zeros = mlxcel_core::zeros(&[size, total_len], dtype::FLOAT32);
    let neg_inf = mlxcel_core::full_f32(&[size, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
    let cond = mlxcel_core::greater(&allowed, &zeros);
    mlxcel_core::where_cond(&cond, &zeros, &neg_inf)
}

impl LanguageModel for DiffusionGemmaModel {
    /// Honest minimal trait forward: an encoder-mode causal pass (writing
    /// `caches`) followed by the final norm and tied-embedding logits. The
    /// CLI routes diffusion models to the block-diffusion engine BEFORE the
    /// autoregressive loop, so this exists for trait completeness (warmup,
    /// tooling) rather than as a generation path.
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.forward_encoder(input_ids, caches, mask);
        let hidden = self.text.norm.forward(&hidden);
        let mut logits = self.text.embed_tokens.as_linear(&hidden);
        if let Some(cap) = self.text.config.final_logit_softcapping {
            logits = mlxcel_core::compiled_softcap(&logits, cap);
        }
        logits
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.make_diffusion_caches()
    }

    fn num_layers(&self) -> usize {
        self.text.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    /// Block-diffusion generation is a model-owned loop over a single
    /// sequence; the batched/paged scheduler must never pick this model up.
    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
