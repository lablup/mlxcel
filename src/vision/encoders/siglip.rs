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
//! - MLP: Linear → GELU(precise) → Linear
//!
//! Used by: Gemma3 VLM (SigLIP), LLaVA (CLIP or SigLIP)

use super::{VisionEncoder, VisionEncoderOutput};
use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::config::VisionConfig;

// Vision MLP.
struct VisionMLP {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl VisionMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.fc1.forward(x);
        let x = mlxcel_core::gelu_approx(&x); // GELU(approx="precise") matching Python
        self.fc2.forward(&x)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let fc1 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc1", prefix), group_size, bits)?;
        let fc2 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc2", prefix), group_size, bits)?;
        Ok(Self { fc1, fc2 })
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
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let queries = self.q_proj.forward(x);
        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);

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

        self.out_proj.forward(&output)
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
        // LayerNorm -> Attention -> residual
        let ln1 = self.layer_norm1.forward(x);
        let r = self.self_attn.forward(&ln1);
        let h = mlxcel_core::add(x, &r);
        // LayerNorm -> MLP -> residual
        let ln2 = self.layer_norm2.forward(&h);
        let r = self.mlp.forward(&ln2);
        mlxcel_core::add(&h, &r)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &VisionConfig,
        group_size: i32,
        bits: i32,
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

        let mlp = VisionMLP::from_weights(weights, &format!("{}.mlp", prefix), group_size, bits)?;

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
        let emb_prefix = format!("{}.embeddings", prefix);
        let embeddings =
            VisionEmbeddings::from_weights(weights, &emb_prefix, config, group_size, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_prefix = format!("{}.encoder.layers.{}", prefix, i);
            let layer =
                EncoderLayer::from_weights(weights, &layer_prefix, config, group_size, bits)?;
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
}

impl VisionEncoder for SigLipVisionModel {
    fn forward(&self, pixel_values: &MlxArray) -> VisionEncoderOutput {
        // Embed patches (+ CLS for CLIP)
        let mut h = self.embeddings.forward(pixel_values);

        // Apply pre-layernorm if present (CLIP only)
        if let Some(ref pre_ln) = self.pre_layrnorm {
            h = pre_ln.forward(&h);
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
                h = layer.forward(&h);
                if i == target_layer {
                    // Select features from this layer (before post_layernorm)
                    let selected = self.apply_feature_select_strategy(&h);
                    return VisionEncoderOutput {
                        hidden_states: selected,
                    };
                }
            }
            // Fallback: if target_layer >= num_layers, use last layer output
            let selected = self.apply_feature_select_strategy(&h);
            return VisionEncoderOutput {
                hidden_states: selected,
            };
        }

        // Default path (Gemma3 SigLIP): pass through all layers + post_layernorm
        for layer in &self.layers {
            h = layer.forward(&h);
        }

        let hidden_states = self.post_layernorm.forward(&h);

        VisionEncoderOutput { hidden_states }
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
