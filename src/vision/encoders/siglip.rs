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

//! SigLIP/CLIP Vision Encoder
//!
//! Unified vision encoder supporting both SigLIP and CLIP architectures.
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llava/vision.py
//!
//! Architecture: VisionEmbeddings → [pre_layrnorm] → Encoder (N × EncoderLayer) → post_layernorm
//! - VisionEmbeddings: Conv2d patch embedding + learned position embedding [+ CLS token for CLIP]
//! - EncoderLayer: LayerNorm → Attention → residual → LayerNorm → MLP → residual
//! - Attention: standard multi-head with bias, uses scaled_dot_product_attention
//! - MLP: Linear → checkpoint-selected GELU → Linear
//!
//! Used by: Gemma3 VLM (SigLIP), LLaVA (CLIP or SigLIP), Idefics2 (via
//! `forward_with_position_ids`, bucketized variable-resolution tiles)

use super::{VisionEncoder, VisionEncoderOutput};
use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::config::{VisionConfig, VisionHiddenActivation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisionMlpActivation {
    Exact,
    PytorchTanh,
    FastSigmoid,
}

fn select_mlp_activation(
    hidden_act: VisionHiddenActivation,
    use_fast_gelu: bool,
) -> VisionMlpActivation {
    if use_fast_gelu {
        VisionMlpActivation::FastSigmoid
    } else {
        match hidden_act {
            VisionHiddenActivation::GeluPytorchTanh => VisionMlpActivation::PytorchTanh,
            VisionHiddenActivation::ExactGelu => VisionMlpActivation::Exact,
        }
    }
}

// Vision MLP.
struct VisionMLP {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    activation: VisionMlpActivation,
}

/// Hugging Face's `gelu_pytorch_tanh`, evaluated in F32 to avoid the BF16
/// `x.pow(3)` overflow that motivated the legacy exact-erf fallback.
///
/// Keep this SigLIP/CLIP-specific: changing the global `gelu_approx` helper
/// would silently alter model families whose checkpoints expect exact GELU.
fn gelu_pytorch_tanh(x: &MlxArray) -> UniquePtr<MlxArray> {
    let output_dtype = mlxcel_core::array_dtype(x);
    let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    let half = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32);
    let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
    let sqrt_two_over_pi = mlxcel_core::full_f32(&[1], 0.797_884_6, mlxcel_core::dtype::FLOAT32);
    let cubic_coefficient = mlxcel_core::full_f32(&[1], 0.044_715, mlxcel_core::dtype::FLOAT32);

    // Multiplication, rather than a generic power operation, is defined for
    // negative inputs and matches PyTorch's polynomial evaluation order.
    let squared = mlxcel_core::multiply(&x, &x);
    let cubed = mlxcel_core::multiply(&squared, &x);
    let cubic = mlxcel_core::multiply(&cubic_coefficient, &cubed);
    let inner = mlxcel_core::multiply(&sqrt_two_over_pi, &mlxcel_core::add(&x, &cubic));
    let cdf = mlxcel_core::multiply(&half, &mlxcel_core::add(&one, &mlxcel_core::tanh(&inner)));
    let activated = mlxcel_core::multiply(&x, &cdf);
    if output_dtype == mlxcel_core::dtype::FLOAT32 {
        activated
    } else {
        mlxcel_core::astype(&activated, output_dtype)
    }
}

impl VisionMLP {
    fn forward_impl(
        &self,
        x: &MlxArray,
        capture: bool,
    ) -> (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>) {
        let x = self.fc1.forward(x);
        let mut diagnostics = Vec::new();
        if capture {
            diagnostics.push(mlxcel_core::copy(&x));
        }
        let x = match self.activation {
            VisionMlpActivation::Exact => mlxcel_core::gelu_approx(&x),
            VisionMlpActivation::PytorchTanh => gelu_pytorch_tanh(&x),
            VisionMlpActivation::FastSigmoid => mlxcel_core::utils::gelu_sigmoid(&x),
        };
        if capture {
            diagnostics.push(mlxcel_core::copy(&x));
        }
        let x = self.fc2.forward(&x);
        if capture {
            diagnostics.push(mlxcel_core::copy(&x));
        }
        (x, diagnostics)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        activation: VisionMlpActivation,
    ) -> Result<Self, String> {
        let fc1 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc1", prefix), group_size, bits)?;
        let fc2 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc2", prefix), group_size, bits)?;
        Ok(Self {
            fc1,
            fc2,
            activation,
        })
    }
}

// Vision Attention.
struct VisionAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    scale: f32,
}

impl VisionAttention {
    fn forward_impl(
        &self,
        x: &MlxArray,
        capture: bool,
    ) -> (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>) {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let queries = self.q_proj.forward(x);
        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);
        let mut diagnostics = Vec::new();
        if capture {
            diagnostics.push(mlxcel_core::copy(&queries));
            diagnostics.push(mlxcel_core::copy(&keys));
            diagnostics.push(mlxcel_core::copy(&values));
        }

        let head_dim = mlxcel_core::array_shape(&queries)[2] / self.num_heads;

        // Reshape to [B, L, num_heads, head_dim] then transpose to [B, num_heads, L, head_dim]
        let queries = mlxcel_core::reshape(&queries, &[b, l, self.num_heads, head_dim]);
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = mlxcel_core::reshape(&keys, &[b, l, self.num_heads, head_dim]);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = mlxcel_core::reshape(&values, &[b, l, self.num_heads, head_dim]);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        // Scaled dot product attention (no mask for vision encoder)
        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &queries,
                &keys,
                &values,
                self.scale,
                std::ptr::null(),
                0.0,
                0,
            )
        };

        // Transpose back and reshape: [B, num_heads, L, head_dim] -> [B, L, D]
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, self.num_heads * head_dim]);
        if capture {
            diagnostics.push(mlxcel_core::copy(&output));
        }

        let output = self.out_proj.forward(&output);
        if capture {
            diagnostics.push(mlxcel_core::copy(&output));
        }
        (output, diagnostics)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: usize,
        dims: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let out_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.out_proj", prefix),
            group_size,
            bits,
        )?;

        let head_dim = dims / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: num_heads as i32,
            scale,
        })
    }
}

// Encoder Layer.
struct EncoderLayer {
    self_attn: VisionAttention,
    layer_norm1: LayerNorm,
    mlp: VisionMLP,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.forward_impl(x, false).0
    }

    fn forward_impl(
        &self,
        x: &MlxArray,
        capture: bool,
    ) -> (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>) {
        let mut diagnostics = Vec::new();
        // LayerNorm -> Attention -> residual
        let ln1 = self.layer_norm1.forward(x);
        if capture {
            diagnostics.push(mlxcel_core::copy(&ln1));
        }
        let (r, attention_diagnostics) = self.self_attn.forward_impl(&ln1, capture);
        diagnostics.extend(attention_diagnostics);
        let h = mlxcel_core::add(x, &r);
        if capture {
            diagnostics.push(mlxcel_core::copy(&h));
        }
        // LayerNorm -> MLP -> residual
        let ln2 = self.layer_norm2.forward(&h);
        if capture {
            diagnostics.push(mlxcel_core::copy(&ln2));
        }
        let (r, mlp_diagnostics) = self.mlp.forward_impl(&ln2, capture);
        diagnostics.extend(mlp_diagnostics);
        let output = mlxcel_core::add(&h, &r);
        if capture {
            diagnostics.push(mlxcel_core::copy(&output));
        }
        (output, diagnostics)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &VisionConfig,
        group_size: i32,
        bits: i32,
        activation: VisionMlpActivation,
    ) -> Result<Self, String> {
        let self_attn = VisionAttention::from_weights(
            weights,
            &format!("{}.self_attn", prefix),
            config.num_attention_heads,
            config.hidden_size,
            group_size,
            bits,
        )?;

        let layer_norm1 = load_layer_norm(
            weights,
            &format!("{}.layer_norm1", prefix),
            config.layer_norm_eps,
        )?;
        let layer_norm2 = load_layer_norm(
            weights,
            &format!("{}.layer_norm2", prefix),
            config.layer_norm_eps,
        )?;

        let mlp = VisionMLP::from_weights(
            weights,
            &format!("{}.mlp", prefix),
            group_size,
            bits,
            activation,
        )?;

        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }
}

// Vision Embeddings (Conv2d patch embedding + position embedding + optional CLS).
struct VisionEmbeddings {
    patch_embedding_weight: UniquePtr<MlxArray>,
    patch_embedding_bias: Option<UniquePtr<MlxArray>>,
    position_embedding: UnifiedEmbedding,
    /// CLS token embedding (CLIP only)
    class_embedding: Option<UniquePtr<MlxArray>>,
    num_patches: usize,
    /// Total position count: num_patches for SigLIP, num_patches+1 for CLIP
    num_positions: usize,
    patch_size: usize,
}

impl VisionEmbeddings {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Conv2d patch embedding: [B, H, W, C] -> [B, H/P, W/P, hidden]
        let patch_emb = if let Some(ref bias) = self.patch_embedding_bias {
            let conv = mlxcel_core::conv2d(
                x,
                &self.patch_embedding_weight,
                self.patch_size as i32,
                self.patch_size as i32,
                0,
                0,
                1,
                1,
                1,
            );
            mlxcel_core::add(&conv, bias)
        } else {
            mlxcel_core::conv2d(
                x,
                &self.patch_embedding_weight,
                self.patch_size as i32,
                self.patch_size as i32,
                0,
                0,
                1,
                1,
                1,
            )
        };

        // Flatten spatial dims: [B, H/P, W/P, hidden] -> [B, num_patches, hidden]
        let shape = mlxcel_core::array_shape(&patch_emb);
        let b = shape[0];
        let hidden = shape[3];
        let patch_emb = mlxcel_core::reshape(&patch_emb, &[b, self.num_patches as i32, hidden]);

        // For CLIP: prepend CLS token
        let embeddings = if let Some(ref cls_emb) = self.class_embedding {
            // Broadcast CLS: [hidden] -> [1, 1, hidden] -> [B, 1, hidden]
            let cls = mlxcel_core::reshape(cls_emb, &[1, 1, hidden]);
            let cls_broadcast = mlxcel_core::broadcast_to(&cls, &[b, 1, hidden]);
            // Concatenate: [B, 1, hidden] + [B, num_patches, hidden] -> [B, num_patches+1, hidden]
            mlxcel_core::concatenate(&cls_broadcast, &patch_emb, 1)
        } else {
            patch_emb
        };

        // Add position embeddings
        let position_ids: Vec<i32> = (0..self.num_positions as i32).collect();
        let pos_ids = mlxcel_core::from_slice_i32(&position_ids, &[1, self.num_positions as i32]);
        let pos_emb = self.position_embedding.forward(&pos_ids);

        mlxcel_core::add(&embeddings, &pos_emb)
    }

    /// SigLIP embed with caller-provided position ids and a *dynamic* patch
    /// count taken from the conv output (not `config.image_size`). Used by
    /// Idefics2, whose vision tower feeds variable-resolution tiles and maps each
    /// tile's `(grid_h, grid_w)` patch grid into the fixed position table via
    /// bucketized ids. Assumes a SigLIP tower (bias conv, no CLS token).
    /// `position_ids` must have length `grid_h * grid_w` (batch 1) and be
    /// pre-wrapped into `[0, num_positions)`.
    fn forward_with_position_ids(&self, x: &MlxArray, position_ids: &[i32]) -> UniquePtr<MlxArray> {
        let ps = self.patch_size as i32;
        let patch_emb = if let Some(ref bias) = self.patch_embedding_bias {
            let conv = mlxcel_core::conv2d(x, &self.patch_embedding_weight, ps, ps, 0, 0, 1, 1, 1);
            mlxcel_core::add(&conv, bias)
        } else {
            mlxcel_core::conv2d(x, &self.patch_embedding_weight, ps, ps, 0, 0, 1, 1, 1)
        };
        // [B, grid_h, grid_w, hidden] -> [B, grid_h*grid_w, hidden]
        let shape = mlxcel_core::array_shape(&patch_emb);
        let b = shape[0];
        let hidden = shape[3];
        let num_patches = shape[1] * shape[2];
        let patch_emb = mlxcel_core::reshape(&patch_emb, &[b, num_patches, hidden]);

        let pos_ids = mlxcel_core::from_slice_i32(position_ids, &[b, num_patches]);
        let pos_emb = self.position_embedding.forward(&pos_ids);
        mlxcel_core::add(&patch_emb, &pos_emb)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &VisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight_key = format!("{}.patch_embedding.weight", prefix);
        let bias_key = format!("{}.patch_embedding.bias", prefix);

        let mut patch_weight = weights
            .get(&weight_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_key))?;

        // Sanitize conv2d weights: check if they need transposing
        // PyTorch: [out, in, kH, kW] -> MLX: [out, kH, kW, in]
        let w_shape = mlxcel_core::array_shape(&patch_weight);
        if w_shape.len() == 4 {
            let (out_ch, dim1, dim2, _dim3) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
            // If out_channels >= kH and kH == kW, it's already in MLX format
            // Otherwise transpose from PyTorch format
            if !(out_ch >= dim1 && out_ch >= dim2 && dim1 == dim2) {
                patch_weight = mlxcel_core::transpose_axes(&patch_weight, &[0, 2, 3, 1]);
            }
        }

        let patch_bias = weights.get(&bias_key).map(|w| mlxcel_core::copy(w));

        let num_patches = (config.image_size / config.patch_size).pow(2);

        // Check for CLS token (CLIP models)
        let cls_key = format!("{}.class_embedding", prefix);
        let class_embedding = weights.get(&cls_key).map(|w| mlxcel_core::copy(w));

        let num_positions = if class_embedding.is_some() {
            num_patches + 1 // CLIP: patches + CLS
        } else {
            num_patches // SigLIP: patches only
        };

        let pos_emb = UnifiedEmbedding::from_weights(
            weights,
            &format!("{}.position_embedding", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            patch_embedding_weight: patch_weight,
            patch_embedding_bias: patch_bias,
            position_embedding: pos_emb,
            class_embedding,
            num_patches,
            num_positions,
            patch_size: config.patch_size,
        })
    }
}

// SigLIP/CLIP Vision Model.
pub struct SigLipVisionModel {
    embeddings: VisionEmbeddings,
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
    /// Pre-layernorm applied after embeddings (CLIP only, note Python typo: "pre_layrnorm")
    pre_layrnorm: Option<LayerNorm>,
    /// Which layer's hidden states to return (-2 = second to last, default for LLaVA)
    /// When None, returns final post-layernorm output (Gemma3 SigLIP behavior)
    vision_feature_layer: Option<i32>,
    /// "default" strips CLS token [:, 1:], "full" keeps all
    vision_feature_select_strategy: Option<String>,
}

impl SigLipVisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &VisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Self::from_weights_with_quant(weights, config, prefix, 64, 4)
    }

    pub fn from_weights_with_quant(
        weights: &WeightMap,
        config: &VisionConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Self::from_weights_with_quant_and_gelu(weights, config, prefix, group_size, bits, false)
    }

    /// Like [`Self::from_weights_with_quant`] but selects the encoder MLP GELU
    /// variant. `use_fast_gelu = true` uses the sigmoid `GELU(approx="fast")`
    /// that the Idefics2 vision tower uses. Otherwise `hidden_act` selects the
    /// explicit PyTorch tanh approximation, with exact-erf GELU as the
    /// compatibility default for missing and unknown values.
    pub fn from_weights_with_quant_and_gelu(
        weights: &WeightMap,
        config: &VisionConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
        use_fast_gelu: bool,
    ) -> Result<Self, String> {
        let emb_prefix = format!("{}.embeddings", prefix);
        let embeddings =
            VisionEmbeddings::from_weights(weights, &emb_prefix, config, group_size, bits)?;
        let activation = select_mlp_activation(config.hidden_act, use_fast_gelu);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_prefix = format!("{}.encoder.layers.{}", prefix, i);
            let layer = EncoderLayer::from_weights(
                weights,
                &layer_prefix,
                config,
                group_size,
                bits,
                activation,
            )?;
            layers.push(layer);
        }

        let post_layernorm = load_layer_norm(
            weights,
            &format!("{}.post_layernorm", prefix),
            config.layer_norm_eps,
        )?;

        // Try loading pre_layrnorm (CLIP only, note the typo in the weight name)
        let pre_layrnorm = load_layer_norm(
            weights,
            &format!("{}.pre_layrnorm", prefix),
            config.layer_norm_eps,
        )
        .ok();

        Ok(Self {
            embeddings,
            layers,
            post_layernorm,
            pre_layrnorm,
            vision_feature_layer: None,
            vision_feature_select_strategy: None,
        })
    }

    /// Configure for LLaVA-style feature selection from intermediate layers
    pub fn with_feature_selection(mut self, layer: i32, strategy: String) -> Self {
        self.vision_feature_layer = Some(layer);
        self.vision_feature_select_strategy = Some(strategy);
        self
    }

    /// Multi-tap forward for LLaVA-Next-style feature selection from several
    /// intermediate layers in a single pass. Builds the `num_layers + 1` entry
    /// hidden-state list `L` (`L[0]` = embeddings output, `L[j]` = output of the
    /// j-th encoder layer, all *before* `post_layernorm`) and returns the entries
    /// named by `taps` in order. A negative tap `v` resolves to `L[len + v]`.
    /// SigLIP-only (no CLS, no `pre_layrnorm`); the `"full"` select strategy is
    /// assumed (no token dropped), which is what Granite Vision uses.
    ///
    /// Used by: Granite Vision (`granite_vision`).
    pub fn forward_collect_layers(
        &self,
        pixel_values: &MlxArray,
        taps: &[i32],
    ) -> Vec<UniquePtr<MlxArray>> {
        let total = self.layers.len() as i32 + 1; // L[0..=num_layers]
        let resolved: Vec<i32> = taps
            .iter()
            .map(|&v| {
                let idx = if v < 0 { total + v } else { v };
                idx.clamp(0, total - 1)
            })
            .collect();
        let want: std::collections::HashSet<i32> = resolved.iter().copied().collect();

        let mut store: Vec<Option<UniquePtr<MlxArray>>> = (0..total).map(|_| None).collect();
        let mut h = self.embeddings.forward(pixel_values);
        if want.contains(&0) {
            store[0] = Some(mlxcel_core::copy(&h));
        }
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            let lidx = i as i32 + 1;
            if want.contains(&lidx) {
                store[lidx as usize] = Some(mlxcel_core::copy(&h));
            }
        }

        resolved
            .iter()
            .map(|&idx| mlxcel_core::copy(store[idx as usize].as_ref().unwrap()))
            .collect()
    }

    /// Idefics2-style forward: embed with caller-provided (bucketized) position
    /// ids over a dynamic patch grid, then run the encoder + `post_layernorm`.
    /// SigLIP-only (no CLS, no `pre_layrnorm`, no feature-layer selection);
    /// idefics2 consumes the post-layernorm last hidden state and its encoder
    /// attention is unmasked, matching the reference vision tower.
    pub fn forward_with_position_ids(
        &self,
        pixel_values: &MlxArray,
        position_ids: &[i32],
    ) -> UniquePtr<MlxArray> {
        let mut h = self
            .embeddings
            .forward_with_position_ids(pixel_values, position_ids);
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        self.post_layernorm.forward(&h)
    }
}

impl VisionEncoder for SigLipVisionModel {
    fn forward(&self, pixel_values: &MlxArray) -> VisionEncoderOutput {
        self.forward_impl(pixel_values, false).0
    }

    #[cfg(feature = "xla-diagnostics")]
    fn forward_with_hidden_state_diagnostics(
        &self,
        pixel_values: &MlxArray,
    ) -> (
        VisionEncoderOutput,
        Vec<UniquePtr<MlxArray>>,
        Vec<UniquePtr<MlxArray>>,
    ) {
        self.forward_impl(pixel_values, true)
    }
}

impl SigLipVisionModel {
    fn forward_impl(
        &self,
        pixel_values: &MlxArray,
        capture_hidden_states: bool,
    ) -> (
        VisionEncoderOutput,
        Vec<UniquePtr<MlxArray>>,
        Vec<UniquePtr<MlxArray>>,
    ) {
        // Embed patches (+ CLS for CLIP)
        let mut h = self.embeddings.forward(pixel_values);

        // Apply pre-layernorm if present (CLIP only)
        if let Some(ref pre_ln) = self.pre_layrnorm {
            h = pre_ln.forward(&h);
        }
        let mut hidden_states = Vec::new();
        let mut block0_diagnostics = Vec::new();
        if capture_hidden_states {
            hidden_states.push(mlxcel_core::copy(&h));
        }

        // Check if we need to collect hidden states for layer selection
        if let Some(feature_layer) = self.vision_feature_layer {
            // Collect hidden states from each layer
            let num_layers = self.layers.len() as i32;
            // Resolve negative index: -2 means second-to-last
            let target_layer = if feature_layer < 0 {
                (num_layers + feature_layer) as usize
            } else {
                feature_layer as usize
            };

            for (i, layer) in self.layers.iter().enumerate() {
                if capture_hidden_states && i == 0 {
                    (h, block0_diagnostics) = layer.forward_impl(&h, true);
                } else {
                    h = layer.forward(&h);
                }
                if capture_hidden_states {
                    hidden_states.push(mlxcel_core::copy(&h));
                }
                if i == target_layer {
                    // Select features from this layer (before post_layernorm)
                    let selected = self.apply_feature_select_strategy(&h);
                    return (
                        VisionEncoderOutput {
                            hidden_states: selected,
                        },
                        hidden_states,
                        block0_diagnostics,
                    );
                }
            }
            // Fallback: if target_layer >= num_layers, use last layer output
            let selected = self.apply_feature_select_strategy(&h);
            return (
                VisionEncoderOutput {
                    hidden_states: selected,
                },
                hidden_states,
                block0_diagnostics,
            );
        }

        // Default path (Gemma3 SigLIP): pass through all layers + post_layernorm
        for (i, layer) in self.layers.iter().enumerate() {
            if capture_hidden_states && i == 0 {
                (h, block0_diagnostics) = layer.forward_impl(&h, true);
            } else {
                h = layer.forward(&h);
            }
            if capture_hidden_states {
                hidden_states.push(mlxcel_core::copy(&h));
            }
        }

        let selected = self.post_layernorm.forward(&h);

        (
            VisionEncoderOutput {
                hidden_states: selected,
            },
            hidden_states,
            block0_diagnostics,
        )
    }
}

impl SigLipVisionModel {
    /// Apply vision_feature_select_strategy
    /// "default": strip CLS token ([:, 1:]) - removes first token
    /// "full": keep all tokens
    fn apply_feature_select_strategy(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let strategy = self
            .vision_feature_select_strategy
            .as_deref()
            .unwrap_or("default");

        match strategy {
            "default" => {
                // Strip CLS token: [:, 1:, :]
                let shape = mlxcel_core::array_shape(h);
                let seq_len = shape[1];
                // Slice from index 1 to end along axis 1
                mlxcel_core::slice(h, &[0, 1, 0], &[shape[0], seq_len, shape[2]])
            }
            "full" => mlxcel_core::copy(h),
            _ => {
                // Unknown strategy, default to stripping CLS
                let shape = mlxcel_core::array_shape(h);
                let seq_len = shape[1];
                mlxcel_core::slice(h, &[0, 1, 0], &[shape[0], seq_len, shape[2]])
            }
        }
    }
}

// Helper functions.
fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight_key = format!("{}.weight", prefix);
    let bias_key = format!("{}.bias", prefix);

    let weight = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", weight_key))?;

    let bias = weights.get(&bias_key).map(|w| mlxcel_core::copy(w));

    Ok(LayerNorm::new(weight, bias, eps))
}

#[cfg(test)]
mod tests {
    use super::{VisionMlpActivation, gelu_pytorch_tanh, select_mlp_activation};
    use crate::vision::config::VisionHiddenActivation;

    #[test]
    fn activation_selection_preserves_exact_and_fast_compatibility() {
        assert_eq!(
            select_mlp_activation(VisionHiddenActivation::ExactGelu, false),
            VisionMlpActivation::Exact
        );
        assert_eq!(
            select_mlp_activation(VisionHiddenActivation::GeluPytorchTanh, false),
            VisionMlpActivation::PytorchTanh
        );
        assert_eq!(
            select_mlp_activation(VisionHiddenActivation::GeluPytorchTanh, true),
            VisionMlpActivation::FastSigmoid
        );
    }

    #[test]
    fn pytorch_tanh_gelu_matches_hugging_face_f32_golden() {
        let input = mlxcel_core::from_slice_f32(&[-3.0, -1.0, 0.0, 1.0, 3.0], &[5]);
        let output = gelu_pytorch_tanh(&input);
        mlxcel_core::eval(&output);
        let expected = [-0.003_637_433, -0.158_808, 0.0, 0.841_192, 2.996_362_7];
        for (index, expected) in expected.into_iter().enumerate() {
            let value = mlxcel_core::slice(&output, &[index as i32], &[index as i32 + 1]);
            assert!(
                (mlxcel_core::item_f32(&value) - expected).abs() <= 2.0e-6,
                "GELU mismatch at {index}"
            );
        }
    }
}
