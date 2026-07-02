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

//! Mllama (Llama 3.2 Vision) tiled ViT image encoder.
//!
//! Faithful port of `references/mlx-vlm/mlx_vlm/models/mllama/vision.py`.
//!
//! The tower ingests `[B, num_media, num_tiles, C, H, W]` pixel values plus the
//! per-image `aspect_ratio_ids` / `aspect_ratio_mask`, runs a Conv2d patch
//! embedding with a prepended class token, adds gated tile + position
//! embeddings, applies a local (non-gated) transformer and a global (gated)
//! transformer, and returns the final hidden state concatenated with a fixed
//! set of intermediate-layer hidden states along the channel dim. The result is
//! `[B, num_media, num_tiles, num_patches, vision_output_dim]`, ready for the
//! multi-modal projector.

use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::mllama::config::MllamaVisionConfig;

fn get_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = get_copy(weights, &format!("{prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Full self-attention over all patches of a tile (no mask beyond the padded
/// aspect-ratio mask). No bias on any projection.
struct VisionAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim() as i32;
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            o_proj: linear("o_proj")?,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &MlxArray, mask: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let reshape_heads = |t: &MlxArray| {
            let t = mlxcel_core::reshape(t, &[b, l, self.num_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3])
        };
        let q = reshape_heads(&self.q_proj.forward(x));
        let k = reshape_heads(&self.k_proj.forward(x));
        let v = reshape_heads(&self.v_proj.forward(x));

        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let out = unsafe {
            mlxcel_core::layers::attention_from_ptr(&q, &k, &v, self.scale, mask_ptr, 0.0, 0)
        };

        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }
}

/// fc1 -> GELU(exact) -> fc2 (both projections carry a bias).
struct VisionMLP {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl VisionMLP {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc1"), group_size, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc2"), group_size, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Reference uses `nn.GELU()` (exact erf GELU).
        let h = mlxcel_core::gelu(&self.fc1.forward(x));
        self.fc2.forward(&h)
    }
}

/// A single ViT encoder layer. Global-transformer layers are gated: their
/// attention and MLP branches are scaled by a learned `tanh(gate)`.
struct EncoderLayer {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    self_attn: VisionAttention,
    mlp: VisionMLP,
    gate_attn: Option<UniquePtr<MlxArray>>,
    gate_ffn: Option<UniquePtr<MlxArray>>,
}

impl EncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
        is_gated: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let (gate_attn, gate_ffn) = if is_gated {
            (
                Some(get_copy(weights, &format!("{prefix}.gate_attn"))?),
                Some(get_copy(weights, &format!("{prefix}.gate_ffn"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            input_layernorm: load_layer_norm(
                weights,
                &format!("{prefix}.input_layernorm"),
                config.norm_eps,
            )?,
            post_attention_layernorm: load_layer_norm(
                weights,
                &format!("{prefix}.post_attention_layernorm"),
                config.norm_eps,
            )?,
            self_attn: VisionAttention::from_weights(
                weights,
                config,
                &format!("{prefix}.self_attn"),
                group_size,
                bits,
            )?,
            mlp: VisionMLP::from_weights(weights, &format!("{prefix}.mlp"), group_size, bits)?,
            gate_attn,
            gate_ffn,
        })
    }

    fn forward(&self, x: &MlxArray, mask: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        // Self-attention branch.
        let normed = self.input_layernorm.forward(x);
        let mut attn = self.self_attn.forward(&normed, mask);
        if let Some(gate) = &self.gate_attn {
            attn = mlxcel_core::multiply(&mlxcel_core::tanh(gate), &attn);
        }
        let x = mlxcel_core::add(x, &attn);

        // Feed-forward branch.
        let normed = self.post_attention_layernorm.forward(&x);
        let mut ff = self.mlp.forward(&normed);
        if let Some(gate) = &self.gate_ffn {
            ff = mlxcel_core::multiply(&mlxcel_core::tanh(gate), &ff);
        }
        mlxcel_core::add(&x, &ff)
    }
}

/// Learned per-tile aspect-ratio embedding (`nn.Embedding`), optionally gated.
/// The embedding table is quantized in `-4bit`/`-8bit` checkpoints, so it loads
/// through [`UnifiedEmbedding`] (which dequantizes on lookup).
struct AspectRatioEmbedding {
    embedding: UnifiedEmbedding,
    gate: Option<UniquePtr<MlxArray>>,
    max_num_tiles: i32,
    hidden_size: i32,
}

impl AspectRatioEmbedding {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            embedding: UnifiedEmbedding::from_weights(
                weights,
                &format!("{prefix}.embedding"),
                group_size,
                bits,
            )?,
            gate: Some(get_copy(weights, &format!("{prefix}.gate"))?),
            max_num_tiles: config.max_num_tiles as i32,
            hidden_size: config.hidden_size as i32,
        })
    }

    /// `hidden_state`: `[bm, max_num_tiles, num_patches, hidden]`.
    /// `aspect_ratio_ids`: int32 `[bm, 1]`.
    fn forward(&self, hidden_state: &MlxArray, aspect_ratio_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let embeddings = self.embedding.forward(aspect_ratio_ids);
        let mut embeddings =
            mlxcel_core::reshape(&embeddings, &[-1, self.max_num_tiles, 1, self.hidden_size]);
        if let Some(gate) = &self.gate {
            embeddings = mlxcel_core::multiply(&embeddings, &mlxcel_core::tanh(gate));
        }
        mlxcel_core::add(hidden_state, &embeddings)
    }
}

/// Learned position embedding: a base per-patch table blended with a per-tile
/// table by a single `tanh` gate.
struct PositionEmbedding {
    gate: UniquePtr<MlxArray>,
    embedding: UniquePtr<MlxArray>,
    tile_embedding: UnifiedEmbedding,
    max_num_tiles: i32,
    num_patches: i32,
    hidden_size: i32,
}

impl PositionEmbedding {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate: get_copy(weights, &format!("{prefix}.gate"))?,
            // The base per-patch table is a raw `nn.Parameter` (never quantized).
            embedding: get_copy(weights, &format!("{prefix}.embedding"))?,
            // The per-tile table is an `nn.Embedding`, quantized in 4/8-bit.
            tile_embedding: UnifiedEmbedding::from_weights(
                weights,
                &format!("{prefix}.tile_embedding"),
                group_size,
                bits,
            )?,
            max_num_tiles: config.max_num_tiles as i32,
            num_patches: config.num_patches() as i32,
            hidden_size: config.hidden_size as i32,
        })
    }

    /// `hidden_state`: `[bm, max_num_tiles, num_patches, hidden]`.
    fn forward(&self, hidden_state: &MlxArray, aspect_ratio_ids: &MlxArray) -> UniquePtr<MlxArray> {
        // Base position embedding, gated by (1 - tanh(gate)).
        let tanh_gate = mlxcel_core::tanh(&self.gate);
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&tanh_gate));
        let inv_gate = mlxcel_core::subtract(&one, &tanh_gate);
        let gated_pos = mlxcel_core::multiply(&inv_gate, &self.embedding);
        let gated_pos =
            mlxcel_core::reshape(&gated_pos, &[1, 1, self.num_patches, self.hidden_size]);
        let hidden_state = mlxcel_core::add(hidden_state, &gated_pos);

        // Per-tile position embedding, gated by tanh(gate).
        let tile = self.tile_embedding.forward(aspect_ratio_ids);
        let tile = mlxcel_core::reshape(
            &tile,
            &[-1, self.max_num_tiles, self.num_patches, self.hidden_size],
        );
        let gated_tile = mlxcel_core::multiply(&tanh_gate, &tile);
        mlxcel_core::add(&hidden_state, &gated_tile)
    }
}

/// A stack of ViT encoder layers that also returns every layer's output so the
/// tower can collect intermediate hidden states.
struct VisionEncoder {
    layers: Vec<EncoderLayer>,
}

impl VisionEncoder {
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
        num_layers: usize,
        is_gated: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(EncoderLayer::from_weights(
                weights,
                config,
                &format!("{prefix}.layers.{i}"),
                is_gated,
                group_size,
                bits,
            )?);
        }
        Ok(Self { layers })
    }

    /// Returns `(final_hidden_state, all_layer_hidden_states)`.
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>) {
        let mut h = mlxcel_core::copy(x);
        let mut states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            h = layer.forward(&h, mask);
            states.push(mlxcel_core::copy(&h));
        }
        (h, states)
    }
}

/// The Mllama vision tower.
pub struct MllamaVisionModel {
    patch_embedding_weight: UniquePtr<MlxArray>,
    class_embedding: UniquePtr<MlxArray>,
    gated_positional_embedding: PositionEmbedding,
    pre_tile_positional_embedding: AspectRatioEmbedding,
    post_tile_positional_embedding: AspectRatioEmbedding,
    layernorm_pre: LayerNorm,
    layernorm_post: LayerNorm,
    transformer: VisionEncoder,
    global_transformer: VisionEncoder,
    config: MllamaVisionConfig,
}

impl MllamaVisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        // Quantization params inherited from the checkpoint's top-level block.
        // The tower projections, MLPs, and tile/positional embedding tables are
        // quantized in `-4bit`/`-8bit` releases; `Unified{Linear,Embedding}`
        // fall back to plain tensors when `.scales` is absent (dense tower).
        let group_size = config.quant_group_size();
        let bits = config.quant_bits();

        // Conv2d patch weight; transpose PyTorch [out,in,kH,kW] -> MLX
        // [out,kH,kW,in] only when the checkpoint is not already channel-last.
        let mut patch_embedding_weight =
            get_copy(weights, &format!("{prefix}.patch_embedding.weight"))?;
        let w_shape = mlxcel_core::array_shape(&patch_embedding_weight);
        if w_shape.len() == 4
            && !(w_shape[0] >= w_shape[1] && w_shape[0] >= w_shape[2] && w_shape[1] == w_shape[2])
        {
            patch_embedding_weight =
                mlxcel_core::transpose_axes(&patch_embedding_weight, &[0, 2, 3, 1]);
        }

        Ok(Self {
            patch_embedding_weight,
            class_embedding: get_copy(weights, &format!("{prefix}.class_embedding"))?,
            gated_positional_embedding: PositionEmbedding::from_weights(
                weights,
                config,
                &format!("{prefix}.gated_positional_embedding"),
                group_size,
                bits,
            )?,
            pre_tile_positional_embedding: AspectRatioEmbedding::from_weights(
                weights,
                config,
                &format!("{prefix}.pre_tile_positional_embedding"),
                group_size,
                bits,
            )?,
            post_tile_positional_embedding: AspectRatioEmbedding::from_weights(
                weights,
                config,
                &format!("{prefix}.post_tile_positional_embedding"),
                group_size,
                bits,
            )?,
            layernorm_pre: load_layer_norm(
                weights,
                &format!("{prefix}.layernorm_pre"),
                config.norm_eps,
            )?,
            layernorm_post: load_layer_norm(
                weights,
                &format!("{prefix}.layernorm_post"),
                config.norm_eps,
            )?,
            transformer: VisionEncoder::from_weights(
                weights,
                config,
                &format!("{prefix}.transformer"),
                config.num_hidden_layers,
                false,
                group_size,
                bits,
            )?,
            global_transformer: VisionEncoder::from_weights(
                weights,
                config,
                &format!("{prefix}.global_transformer"),
                config.num_global_layers,
                true,
                group_size,
                bits,
            )?,
            config: config.clone(),
        })
    }

    /// Whether the tower loaded quantized projections (i.e. the checkpoint
    /// carried `.scales`/`.biases`). Drives nothing at runtime beyond the
    /// `UnifiedLinear` dispatch it already encapsulates; exposed for load-time
    /// inspection and the checkpoint-key parity tests.
    pub fn is_quantized(&self) -> bool {
        self.transformer
            .layers
            .first()
            .is_some_and(|layer| layer.self_attn.q_proj.is_quantized())
    }

    /// `pixel_values`: `[B, num_media, num_tiles, C, H, W]`.
    /// `aspect_ratio_ids`: int32 `[B, num_media]`.
    /// `aspect_ratio_mask`: int32 `[B, num_media, num_tiles]`.
    ///
    /// Returns `[B, num_media, num_tiles, num_patches, vision_output_dim]`.
    pub fn forward(
        &self,
        pixel_values: &MlxArray,
        aspect_ratio_ids: &MlxArray,
        aspect_ratio_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let cfg = &self.config;
        let shape = mlxcel_core::array_shape(pixel_values);
        let (batch, num_media, num_tiles, num_channels, height, width) =
            (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        let bm = batch * num_media;
        let hidden = cfg.hidden_size as i32;
        let patch = cfg.patch_size as i32;

        let aspect_ratio_ids = mlxcel_core::reshape(aspect_ratio_ids, &[bm, -1]);

        // Match the tower's weight dtype (f16 on Apple Silicon) so Conv2d does
        // not fault on an f32 input against f16 weights.
        let pixel_values = mlxcel_core::astype(
            pixel_values,
            mlxcel_core::array_dtype(&self.patch_embedding_weight),
        );

        // Patch embedding via Conv2d (channels-last in MLX).
        let pv = mlxcel_core::reshape(
            &pixel_values,
            &[bm * num_tiles, num_channels, height, width],
        );
        let pv = mlxcel_core::moveaxis(&pv, 1, 3);
        let patch_embeds = mlxcel_core::conv2d(
            &pv,
            &self.patch_embedding_weight,
            patch,
            patch,
            0,
            0,
            1,
            1,
            1,
        );
        // [bm*tiles, Hp, Wp, hidden] -> [bm*tiles, num_patches_no_cls, hidden].
        let num_patches_no_cls = (height / patch) * (width / patch);
        let hidden_state =
            mlxcel_core::reshape(&patch_embeds, &[bm * num_tiles, num_patches_no_cls, hidden]);

        // Pre-tile positional embedding over [bm, tiles, patches, hidden].
        let hidden_state =
            mlxcel_core::reshape(&hidden_state, &[bm, num_tiles, num_patches_no_cls, hidden]);
        let hidden_state = self
            .pre_tile_positional_embedding
            .forward(&hidden_state, &aspect_ratio_ids);

        // Prepend the class token.
        let hidden_state =
            mlxcel_core::reshape(&hidden_state, &[bm * num_tiles, num_patches_no_cls, hidden]);
        let cls = mlxcel_core::reshape(&self.class_embedding, &[1, 1, hidden]);
        let cls = mlxcel_core::broadcast_to(&cls, &[bm * num_tiles, 1, hidden]);
        let hidden_state = mlxcel_core::concatenate(&cls, &hidden_state, 1);
        let num_patches = num_patches_no_cls + 1;

        // Gated position embedding over [bm, tiles, patches, hidden].
        let hidden_state =
            mlxcel_core::reshape(&hidden_state, &[bm, num_tiles, num_patches, hidden]);
        let hidden_state = self
            .gated_positional_embedding
            .forward(&hidden_state, &aspect_ratio_ids);
        let hidden_state = self.layernorm_pre.forward(&hidden_state);

        // Pad patches up to a multiple of 8.
        let num_padding = (8 - (num_patches % 8)) % 8;
        let hidden_state = if num_padding > 0 {
            mlxcel_core::pad(&hidden_state, &[0, 0, 0, 0, 0, num_padding, 0, 0], 0.0)
        } else {
            hidden_state
        };
        let padded_patches = num_patches + num_padding;

        // Aspect-ratio attention mask (additive, [bm, 1, L, L]).
        let mask = prepare_aspect_ratio_attention_mask(
            &mlxcel_core::reshape(aspect_ratio_mask, &[bm, -1]),
            cfg.num_patches() as i32,
            padded_patches,
            num_tiles,
        );

        // Local (non-gated) transformer.
        let hidden_state =
            mlxcel_core::reshape(&hidden_state, &[bm, num_tiles * padded_patches, hidden]);
        let (encoded, intermediate_states) = self.transformer.forward(&hidden_state, Some(&mask));
        let hidden_state = self.layernorm_post.forward(&encoded);

        // Post-tile positional embedding then the global (gated) transformer.
        let hidden_state =
            mlxcel_core::reshape(&hidden_state, &[bm, num_tiles, padded_patches, hidden]);
        let hidden_state = self
            .post_tile_positional_embedding
            .forward(&hidden_state, &aspect_ratio_ids);
        let hidden_state =
            mlxcel_core::reshape(&hidden_state, &[bm, num_tiles * padded_patches, hidden]);
        let (global_encoded, _) = self.global_transformer.forward(&hidden_state, Some(&mask));

        // Strip the patch padding, restore the [B, media, tiles, patches, hidden] shape.
        let hidden_state =
            mlxcel_core::reshape(&global_encoded, &[bm, num_tiles, padded_patches, hidden]);
        let hidden_state = mlxcel_core::slice(
            &hidden_state,
            &[0, 0, 0, 0],
            &[bm, num_tiles, num_patches, hidden],
        );
        let hidden_state = mlxcel_core::reshape(
            &hidden_state,
            &[batch, num_media, num_tiles, num_patches, hidden],
        );

        // Collect intermediate hidden states, select the configured indices,
        // strip padding, and concatenate onto the channel dim.
        let intermediate = self.collect_intermediate(
            &intermediate_states,
            bm,
            num_tiles,
            padded_patches,
            num_patches,
            hidden,
            batch,
            num_media,
        );
        mlxcel_core::concatenate(&hidden_state, &intermediate, -1)
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_intermediate(
        &self,
        states: &[UniquePtr<MlxArray>],
        bm: i32,
        num_tiles: i32,
        padded_patches: i32,
        num_patches: i32,
        hidden: i32,
        batch: i32,
        num_media: i32,
    ) -> UniquePtr<MlxArray> {
        let indices = &self.config.intermediate_layers_indices;
        // Stack the selected layers on a new trailing axis, matching
        // `mx.stack(all_states, axis=-1)[..., indices]`.
        let selected: Vec<*const MlxArray> = indices
            .iter()
            .map(|&i| states[i].as_ref().expect("intermediate state") as *const MlxArray)
            .collect();
        // Each state is [bm, tiles*padded_patches, hidden]; stack on axis=-1.
        let stacked = mlxcel_core::stack(&selected, 2);
        // [bm, tiles*padded_patches, hidden, k] -> [bm, tiles, padded_patches, hidden*k].
        let k = indices.len() as i32;
        let inter = mlxcel_core::reshape(&stacked, &[bm, num_tiles, padded_patches, hidden * k]);
        let inter = mlxcel_core::slice(
            &inter,
            &[0, 0, 0, 0],
            &[bm, num_tiles, num_patches, hidden * k],
        );
        mlxcel_core::reshape(
            &inter,
            &[batch, num_media, num_tiles, num_patches, hidden * k],
        )
    }
}

/// Build the additive `[bm, 1, tiles*L, tiles*L]` attention mask that keeps
/// padding tiles and padding patches from attending. Faithful port of
/// `_prepare_aspect_ratio_attention_mask`.
fn prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: &MlxArray,
    num_patches: i32,
    target_length: i32,
    max_num_tiles: i32,
) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::dtype::FLOAT32;
    let mask = mlxcel_core::astype(aspect_ratio_mask, dtype);
    let bm = mlxcel_core::array_shape(&mask)[0];

    // [bm, tiles] -> [bm, tiles, 1, 1] -> tile over the patch axis.
    let mask = mlxcel_core::reshape(&mask, &[bm, max_num_tiles, 1, 1]);
    let mask = mlxcel_core::tile(&mask, &[1, 1, target_length, 1]);

    // Zero the trailing (padding) patches.
    let pad_patches = target_length - num_patches;
    let mask = if pad_patches > 0 {
        let valid = mlxcel_core::slice(&mask, &[0, 0, 0, 0], &[bm, max_num_tiles, num_patches, 1]);
        let zeros = mlxcel_core::zeros(&[bm, max_num_tiles, pad_patches, 1], dtype);
        mlxcel_core::concatenate(&valid, &zeros, 2)
    } else {
        mask
    };

    // Invert (0 -> 1, 1 -> 0), collapse to [bm, tiles*L, 1], outer-product,
    // scale by a large negative bias, add the head axis.
    let one = mlxcel_core::full_f32(&mlxcel_core::array_shape(&mask), 1.0, dtype);
    let inverted = mlxcel_core::subtract(&one, &mask);
    let inverted = mlxcel_core::reshape(&inverted, &[bm, max_num_tiles * target_length, 1]);
    let transposed = mlxcel_core::transpose_axes(&inverted, &[0, 2, 1]);
    let outer = mlxcel_core::matmul(&inverted, &transposed);
    let outer = mlxcel_core::multiply_scalar(&outer, -1e9);
    mlxcel_core::expand_dims(&outer, 1)
}

#[cfg(test)]
mod tests {
    use super::{MllamaVisionModel, prepare_aspect_ratio_attention_mask};
    use crate::models::mllama::MllamaVisionConfig;
    use mlxcel_core::weights::WeightMap;
    use mlxcel_core::{MlxArray, UniquePtr};

    fn value_at(mask: &mlxcel_core::MlxArray, i: i32, j: i32) -> f32 {
        // mask: [1, 1, L, L]
        let cell = mlxcel_core::slice(mask, &[0, 0, i, j], &[1, 1, i + 1, j + 1]);
        mlxcel_core::eval(&cell);
        mlxcel_core::item_f32(&cell)
    }

    #[test]
    fn all_valid_tiles_produce_a_zero_mask() {
        // 2 tiles, 2 patches each, both tiles valid, no patch padding.
        let ar_mask = mlxcel_core::from_slice_i32(&[1, 1], &[1, 2]);
        let mask = prepare_aspect_ratio_attention_mask(&ar_mask, 2, 2, 2);
        assert_eq!(mlxcel_core::array_shape(&mask), vec![1, 1, 4, 4]);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    value_at(&mask, i, j),
                    0.0,
                    "cell ({i},{j}) must be unmasked"
                );
            }
        }
    }

    #[test]
    fn padding_tile_is_masked_out() {
        // 2 tiles, 2 patches each; tile 1 is a padding tile (aspect ratio 0).
        // Positions belonging to tile 1 (flattened indices 2,3) must be masked.
        let ar_mask = mlxcel_core::from_slice_i32(&[1, 0], &[1, 2]);
        let mask = prepare_aspect_ratio_attention_mask(&ar_mask, 2, 2, 2);
        assert_eq!(value_at(&mask, 0, 0), 0.0);
        assert_eq!(value_at(&mask, 1, 1), 0.0);
        // Both endpoints in the padding tile -> masked.
        assert!(value_at(&mask, 2, 2) < -1e8);
        assert!(value_at(&mask, 3, 3) < -1e8);
    }

    // --- Real-checkpoint vision-tower key-set parity (issue #527 follow-up). ---
    //
    // These reconstruct the exact key names the real Llama-3.2-11B-Vision(-4bit)
    // checkpoint stores for the tower (verified against
    // `model.safetensors.index.json`) and feed them through
    // `MllamaVisionModel::from_weights`, asserting the loader resolves every key
    // it expects. They run on CPU with tiny tensors (no GPU, no real weights):
    // `from_weights` only copies tensors and reads shapes, so no kernel runs.

    /// Vision tower config mirroring the real tower's structure with a tiny
    /// layer count, exercising both the local and gated global transformer.
    fn tiny_vision_config(quantized: bool) -> MllamaVisionConfig {
        let quant = if quantized {
            r#", "quantization": { "group_size": 64, "bits": 4 }"#
        } else {
            ""
        };
        serde_json::from_str(&format!(
            r#"{{
                "image_size": 28,
                "patch_size": 14,
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_global_layers": 1,
                "num_attention_heads": 2,
                "max_num_tiles": 4,
                "intermediate_layers_indices": [0]
                {quant}
            }}"#
        ))
        .expect("tiny mllama vision config")
    }

    fn dense(map: &mut WeightMap, key: &str, shape: &[i32]) {
        let n: i32 = shape.iter().product();
        map.insert(
            key.to_string(),
            mlxcel_core::from_slice_f32(&vec![0.0; n as usize], shape),
        );
    }

    /// Insert an `nn.Linear` / `nn.Embedding` at `prefix`. When `quantized`,
    /// emit the affine triplet `.weight` (u32 packed) + `.scales` + `.biases`
    /// with shapes satisfying MLX's `packed_in * 32 == bits * num_groups *
    /// group_size` invariant (group_size 64 / bits 4: in=64 -> packed_in 8,
    /// num_groups 1); otherwise a plain float `.weight`.
    fn quant_or_dense(map: &mut WeightMap, prefix: &str, quantized: bool, bias: bool) {
        if quantized {
            map.insert(
                format!("{prefix}.weight"),
                mlxcel_core::from_slice_u32(&[0u32; 8 * 8], &[8, 8]),
            );
            dense(map, &format!("{prefix}.scales"), &[8, 1]);
            dense(map, &format!("{prefix}.biases"), &[8, 1]);
        } else {
            dense(map, &format!("{prefix}.weight"), &[8, 64]);
        }
        if bias {
            dense(map, &format!("{prefix}.bias"), &[8]);
        }
    }

    fn add_transformer_layers(
        map: &mut WeightMap,
        prefix: &str,
        stack: &str,
        count: usize,
        gated: bool,
        quantized: bool,
    ) {
        for i in 0..count {
            let p = format!("{prefix}.{stack}.layers.{i}");
            dense(map, &format!("{p}.input_layernorm.weight"), &[8]);
            dense(map, &format!("{p}.input_layernorm.bias"), &[8]);
            dense(map, &format!("{p}.post_attention_layernorm.weight"), &[8]);
            dense(map, &format!("{p}.post_attention_layernorm.bias"), &[8]);
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                quant_or_dense(map, &format!("{p}.self_attn.{proj}"), quantized, false);
            }
            // mlp fc1/fc2 carry a float bias in addition to quant scales/biases.
            quant_or_dense(map, &format!("{p}.mlp.fc1"), quantized, true);
            quant_or_dense(map, &format!("{p}.mlp.fc2"), quantized, true);
            if gated {
                dense(map, &format!("{p}.gate_attn"), &[1]);
                dense(map, &format!("{p}.gate_ffn"), &[1]);
            }
        }
    }

    /// Build the full vision-tower weight set exactly as the real checkpoint
    /// stores it under `prefix` (2 local + 1 gated global layer).
    fn build_vision_weights(prefix: &str, quantized: bool) -> WeightMap {
        let mut w = WeightMap::new();

        // Non-quantized leaf tensors: patch conv, class token, pre/post norms.
        dense(&mut w, &format!("{prefix}.class_embedding"), &[8]);
        dense(
            &mut w,
            &format!("{prefix}.patch_embedding.weight"),
            &[8, 3, 2, 2],
        );
        dense(&mut w, &format!("{prefix}.layernorm_pre.weight"), &[8]);
        dense(&mut w, &format!("{prefix}.layernorm_pre.bias"), &[8]);
        dense(&mut w, &format!("{prefix}.layernorm_post.weight"), &[8]);
        dense(&mut w, &format!("{prefix}.layernorm_post.bias"), &[8]);

        // Gated positional embedding: raw base table + gate, quantized tile table.
        dense(
            &mut w,
            &format!("{prefix}.gated_positional_embedding.embedding"),
            &[5, 8],
        );
        dense(
            &mut w,
            &format!("{prefix}.gated_positional_embedding.gate"),
            &[1],
        );
        quant_or_dense(
            &mut w,
            &format!("{prefix}.gated_positional_embedding.tile_embedding"),
            quantized,
            false,
        );

        // Pre/post tile aspect-ratio embeddings: quantized table + gate.
        for name in [
            "pre_tile_positional_embedding",
            "post_tile_positional_embedding",
        ] {
            quant_or_dense(
                &mut w,
                &format!("{prefix}.{name}.embedding"),
                quantized,
                false,
            );
            dense(&mut w, &format!("{prefix}.{name}.gate"), &[1]);
        }

        add_transformer_layers(&mut w, prefix, "transformer", 2, false, quantized);
        add_transformer_layers(&mut w, prefix, "global_transformer", 1, true, quantized);
        w
    }

    #[test]
    fn real_vision_tower_keys_load_and_stay_quantized() {
        let config = tiny_vision_config(true);
        let weights = build_vision_weights("vision_tower", true);
        let model = MllamaVisionModel::from_weights(&weights, &config, "vision_tower")
            .expect("real 4-bit vision_tower key set must resolve against the encoder");
        assert!(
            model.is_quantized(),
            "the 4-bit tower must load quantized projections, not raw packed weights"
        );
    }

    #[test]
    fn wrong_vision_prefix_reproduces_the_527_load_failure() {
        // Before this fix the loader passed "vision_model", which does not exist
        // in the real checkpoint. Guard against a regression to that prefix.
        let config = tiny_vision_config(true);
        let weights = build_vision_weights("vision_tower", true);
        // `MllamaVisionModel` is not `Debug`, so match rather than `expect_err`.
        let err = match MllamaVisionModel::from_weights(&weights, &config, "vision_model") {
            Ok(_) => panic!("the pre-fix 'vision_model' prefix must not resolve"),
            Err(e) => e,
        };
        assert!(
            err.contains("vision_model.patch_embedding.weight"),
            "expected the original #527 error, got: {err}"
        );
    }

    #[test]
    fn dense_vision_tower_falls_back_to_regular_linear() {
        // A hypothetical unquantized tower (no `.scales`) must still load, via
        // the Unified{Linear,Embedding} regular fallback.
        let config = tiny_vision_config(false);
        let weights = build_vision_weights("vision_tower", false);
        let model = MllamaVisionModel::from_weights(&weights, &config, "vision_tower")
            .expect("dense vision_tower key set must resolve");
        assert!(
            !model.is_quantized(),
            "a dense tower must not report quantized projections"
        );
    }

    // --- Padding-tile semantics (issue #527 perf follow-up). ---
    //
    // These tests document WHY the tower must keep processing the zero-padding
    // tiles even though they look like wasted compute. The aspect-ratio mask is
    // an outer product of the inverted validity vector, so it adds `-1e9` only
    // where BOTH the query and the key are padding positions. Real-tile queries
    // therefore attend to padding-tile keys with a zero bias (strictly positive
    // softmax weight), and padding-tile lanes ingest real-tile content in one
    // layer and feed it back to real-tile queries in the next. The padding
    // tiles act as trained register lanes: dropping them from the tower input
    // changes the real tiles' output. This matches the reference exactly
    // (mlx-vlm `_prepare_aspect_ratio_attention_mask` and the HF processor,
    // which always pads the tile axis to `max_image_tiles` before the tower).

    /// Real->padding and padding->real attention is UNMASKED; only
    /// padding->padding pairs carry the -1e9 bias. This is the structural
    /// reason a "run the tower on real tiles only" optimization is unsound.
    #[test]
    fn mask_keeps_real_to_padding_attention_open() {
        // 2 tiles, 2 patches each; tile 1 is a padding tile.
        // Flattened positions: 0,1 = tile 0 (real), 2,3 = tile 1 (padding).
        let ar_mask = mlxcel_core::from_slice_i32(&[1, 0], &[1, 2]);
        let mask = prepare_aspect_ratio_attention_mask(&ar_mask, 2, 2, 2);

        // Real query -> padding key: open (bias 0), NOT excluded.
        assert_eq!(value_at(&mask, 0, 2), 0.0);
        assert_eq!(value_at(&mask, 1, 3), 0.0);
        // Padding query -> real key: also open.
        assert_eq!(value_at(&mask, 2, 0), 0.0);
        assert_eq!(value_at(&mask, 3, 1), 0.0);
        // Only padding query -> padding key is masked.
        assert!(value_at(&mask, 2, 3) < -1e8);
        assert!(value_at(&mask, 3, 2) < -1e8);
    }

    // --- Forward-capable tiny tower harness. ---

    /// Tiny but forward-runnable tower: 4x4 image, patch 2 (4 patches + cls =
    /// 5, padded to 8), hidden 8, 2 heads, 2 local + 1 gated global layer,
    /// max_num_tiles 4.
    fn forward_vision_config() -> MllamaVisionConfig {
        serde_json::from_str(
            r#"{
                "image_size": 4,
                "patch_size": 2,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_global_layers": 1,
                "num_attention_heads": 2,
                "max_num_tiles": 4,
                "intermediate_layers_indices": [0]
            }"#,
        )
        .expect("forward-capable tiny mllama vision config")
    }

    /// Deterministic pseudo-random fill in roughly `[-0.5, 0.5]`.
    fn fill(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 131 + seed * 977 + 7) % 251) as f32 / 251.0 - 0.5)
            .collect()
    }

    fn put(map: &mut WeightMap, key: &str, shape: &[i32], seed: usize) {
        let n: i32 = shape.iter().product();
        map.insert(
            key.to_string(),
            mlxcel_core::from_slice_f32(&fill(n as usize, seed), shape),
        );
    }

    /// Full dense weight set with non-degenerate values so attention mixing is
    /// observable (the zero-filled loader-test weights would hide it).
    fn build_forward_weights(prefix: &str) -> WeightMap {
        let (h, inter, np, tiles, ar_rows) = (8, 16, 5, 4, 9);
        let mut w = WeightMap::new();

        put(&mut w, &format!("{prefix}.class_embedding"), &[h], 1);
        // PyTorch conv layout [out, in, kH, kW]; the loader transposes it.
        put(
            &mut w,
            &format!("{prefix}.patch_embedding.weight"),
            &[h, 3, 2, 2],
            2,
        );
        for (name, seed) in [("layernorm_pre", 3), ("layernorm_post", 5)] {
            put(&mut w, &format!("{prefix}.{name}.weight"), &[h], seed);
            put(&mut w, &format!("{prefix}.{name}.bias"), &[h], seed + 1);
        }

        put(
            &mut w,
            &format!("{prefix}.gated_positional_embedding.embedding"),
            &[np, h],
            7,
        );
        put(
            &mut w,
            &format!("{prefix}.gated_positional_embedding.gate"),
            &[1],
            8,
        );
        put(
            &mut w,
            &format!("{prefix}.gated_positional_embedding.tile_embedding.weight"),
            &[ar_rows, tiles * np * h],
            9,
        );
        for (name, seed) in [
            ("pre_tile_positional_embedding", 10),
            ("post_tile_positional_embedding", 12),
        ] {
            put(
                &mut w,
                &format!("{prefix}.{name}.embedding.weight"),
                &[ar_rows, tiles * h],
                seed,
            );
            put(&mut w, &format!("{prefix}.{name}.gate"), &[1], seed + 1);
        }

        let mut add_layers = |stack: &str, count: usize, gated: bool, base: usize| {
            for i in 0..count {
                let p = format!("{prefix}.{stack}.layers.{i}");
                let s = base + i * 20;
                put(&mut w, &format!("{p}.input_layernorm.weight"), &[h], s);
                put(&mut w, &format!("{p}.input_layernorm.bias"), &[h], s + 1);
                put(
                    &mut w,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[h],
                    s + 2,
                );
                put(
                    &mut w,
                    &format!("{p}.post_attention_layernorm.bias"),
                    &[h],
                    s + 3,
                );
                for (j, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
                    put(
                        &mut w,
                        &format!("{p}.self_attn.{proj}.weight"),
                        &[h, h],
                        s + 4 + j,
                    );
                }
                put(&mut w, &format!("{p}.mlp.fc1.weight"), &[inter, h], s + 8);
                put(&mut w, &format!("{p}.mlp.fc1.bias"), &[inter], s + 9);
                put(&mut w, &format!("{p}.mlp.fc2.weight"), &[h, inter], s + 10);
                put(&mut w, &format!("{p}.mlp.fc2.bias"), &[h], s + 11);
                if gated {
                    put(&mut w, &format!("{p}.gate_attn"), &[1], s + 12);
                    put(&mut w, &format!("{p}.gate_ffn"), &[1], s + 13);
                }
            }
        };
        add_layers("transformer", 2, false, 100);
        add_layers("global_transformer", 1, true, 200);
        w
    }

    fn forward_tower() -> MllamaVisionModel {
        let config = forward_vision_config();
        let weights = build_forward_weights("vision_tower");
        MllamaVisionModel::from_weights(&weights, &config, "vision_tower")
            .expect("forward-capable tiny tower must load")
    }

    /// Build `[1, 1, 4, 3, 4, 4]` pixel values: tile 0 is a real tile, tiles
    /// 1..4 carry the given per-tile fills (zero = processor padding tile).
    fn pixel_values_with_tiles(tile_fills: [Option<usize>; 4]) -> UniquePtr<MlxArray> {
        let per_tile = 3 * 4 * 4;
        let mut pixels = Vec::with_capacity(4 * per_tile);
        for fill_seed in tile_fills {
            match fill_seed {
                Some(seed) => pixels.extend(fill(per_tile, seed)),
                None => pixels.extend(std::iter::repeat_n(0.0f32, per_tile)),
            }
        }
        mlxcel_core::from_slice_f32(&pixels, &[1, 1, 4, 3, 4, 4])
    }

    fn max_abs_diff(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) -> f32 {
        let diff = mlxcel_core::subtract(a, b);
        let m = mlxcel_core::max_all(&mlxcel_core::abs(&diff));
        mlxcel_core::eval(&m);
        mlxcel_core::item_f32(&m)
    }

    /// The content of a padding-tile lane reaches the REAL tile's tower
    /// output. Two runs share an identical real tile 0 and differ only in
    /// tile 1 (zero vs non-zero content); the real tile's output changes, so
    /// padding lanes are live inputs to real-tile results, and slicing them
    /// away before the tower is NOT an output-preserving optimization.
    ///
    /// If this test ever starts failing (diff == 0), the mask semantics were
    /// changed to truly exclude padding tiles, and a real-tiles-only tower
    /// fast path would become legal. Revisit issue #527's perf follow-up then.
    #[test]
    fn padding_tile_content_reaches_real_tile_output() {
        let tower = forward_tower();
        let ar_ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
        let ar_mask = mlxcel_core::from_slice_i32(&[1, 0, 0, 0], &[1, 1, 4]);

        let zero_padding = pixel_values_with_tiles([Some(50), None, None, None]);
        let perturbed_padding = pixel_values_with_tiles([Some(50), Some(60), None, None]);

        let out_zero = tower.forward(&zero_padding, &ar_ids, &ar_mask);
        let out_perturbed = tower.forward(&perturbed_padding, &ar_ids, &ar_mask);

        // [1, 1, 4, 5, 16]: hidden (8) + one intermediate layer (8).
        assert_eq!(
            mlxcel_core::array_shape(&out_zero),
            vec![1, 1, 4, 5, 16],
            "tiny tower output contract"
        );

        // Compare ONLY the real tile (tile 0).
        let real_zero = mlxcel_core::slice(&out_zero, &[0, 0, 0, 0, 0], &[1, 1, 1, 5, 16]);
        let real_perturbed =
            mlxcel_core::slice(&out_perturbed, &[0, 0, 0, 0, 0], &[1, 1, 1, 5, 16]);
        let diff = max_abs_diff(&real_zero, &real_perturbed);
        assert!(
            diff > 1e-6,
            "padding-tile content must leak into the real tile's output through \
             the unmasked real->padding attention (measured diff {diff}); if this \
             is now 0, the mask semantics changed and a real-tiles-only tower \
             fast path may have become legal"
        );
    }
}
