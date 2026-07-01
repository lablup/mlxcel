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

use mlxcel_core::layers::{LayerNorm, Linear};
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
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim() as i32;
        Ok(Self {
            q_proj: Linear::from_weights(weights, &format!("{prefix}.q_proj"))?,
            k_proj: Linear::from_weights(weights, &format!("{prefix}.k_proj"))?,
            v_proj: Linear::from_weights(weights, &format!("{prefix}.v_proj"))?,
            o_proj: Linear::from_weights(weights, &format!("{prefix}.o_proj"))?,
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
    fc1: Linear,
    fc2: Linear,
}

impl VisionMLP {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            fc1: Linear::from_weights(weights, &format!("{prefix}.fc1"))?,
            fc2: Linear::from_weights(weights, &format!("{prefix}.fc2"))?,
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
            )?,
            mlp: VisionMLP::from_weights(weights, &format!("{prefix}.mlp"))?,
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
struct AspectRatioEmbedding {
    weight: UniquePtr<MlxArray>,
    gate: Option<UniquePtr<MlxArray>>,
    max_num_tiles: i32,
    hidden_size: i32,
}

impl AspectRatioEmbedding {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: get_copy(weights, &format!("{prefix}.embedding.weight"))?,
            gate: Some(get_copy(weights, &format!("{prefix}.gate"))?),
            max_num_tiles: config.max_num_tiles as i32,
            hidden_size: config.hidden_size as i32,
        })
    }

    /// `hidden_state`: `[bm, max_num_tiles, num_patches, hidden]`.
    /// `aspect_ratio_ids`: int32 `[bm, 1]`.
    fn forward(&self, hidden_state: &MlxArray, aspect_ratio_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let embeddings = mlxcel_core::embedding(&self.weight, aspect_ratio_ids);
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
    tile_embedding_weight: UniquePtr<MlxArray>,
    max_num_tiles: i32,
    num_patches: i32,
    hidden_size: i32,
}

impl PositionEmbedding {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            gate: get_copy(weights, &format!("{prefix}.gate"))?,
            embedding: get_copy(weights, &format!("{prefix}.embedding"))?,
            tile_embedding_weight: get_copy(weights, &format!("{prefix}.tile_embedding.weight"))?,
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
        let tile = mlxcel_core::embedding(&self.tile_embedding_weight, aspect_ratio_ids);
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
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaVisionConfig,
        prefix: &str,
        num_layers: usize,
        is_gated: bool,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(EncoderLayer::from_weights(
                weights,
                config,
                &format!("{prefix}.layers.{i}"),
                is_gated,
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
            )?,
            pre_tile_positional_embedding: AspectRatioEmbedding::from_weights(
                weights,
                config,
                &format!("{prefix}.pre_tile_positional_embedding"),
            )?,
            post_tile_positional_embedding: AspectRatioEmbedding::from_weights(
                weights,
                config,
                &format!("{prefix}.post_tile_positional_embedding"),
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
            )?,
            global_transformer: VisionEncoder::from_weights(
                weights,
                config,
                &format!("{prefix}.global_transformer"),
                config.num_global_layers,
                true,
            )?,
            config: config.clone(),
        })
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
    use super::prepare_aspect_ratio_attention_mask;

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
}
