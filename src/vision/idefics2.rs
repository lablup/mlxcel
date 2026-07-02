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

//! Idefics2 (`idefics2`) Vision-Language Model.
//!
//! Faithful port of
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/idefics2/idefics2.py
//! (connector + perceiver resampler + vision/text fusion) and its Mistral text
//! backbone.
//!
//! Composition (for `mlx-community/idefics2-8b-4bit`, `HuggingFaceM4/idefics2-8b`):
//! - `vision_model`: a SigLIP encoder ([`SigLipVisionModel`]). Idefics2 reuses
//!   the SigLIP vision tower (Conv2d patch embed + learned position embedding, N
//!   encoder layers, `post_layernorm`, no CLS token, `gelu_pytorch_tanh` MLP) via
//!   its `forward_with_position_ids` path. The tower runs unmasked (matching the
//!   reference, which uses the patch mask only to compute position ids), and each
//!   tile's `(grid_h, grid_w)` patch grid is mapped into the fixed 70x70 position
//!   table by the bucketized ids in [`bucketize_position_ids`].
//! - `connector` ([`Idefics2Connector`]): a `modality_projection` SwiGLU MLP that
//!   maps the SigLIP hidden size into the text hidden size, followed by an
//!   [`Idefics2PerceiverResampler`] that compresses each tile's patch sequence
//!   into `n_latents` (64) learned query slots via cross-attention. Unlike the
//!   SmolVLM pixel-shuffle connector, the token count per tile is fixed at
//!   `n_latents` regardless of the patch grid.
//! - `language_model`: a Mistral backbone. Mistral is architecturally Llama
//!   (RoPE every layer, SwiGLU MLP, RMSNorm, GQA), so we reuse mlxcel's
//!   [`crate::models::Llama3Model`] exactly as the SmolVLM runtime reuses it for
//!   its Llama backbone. This gives the full VLM runtime surface
//!   (`forward_with_embeddings`, batched decode, sequence-state) for free.
//!
//! Vision/text fusion: the connector produces `num_image_token` (= `n_latents`,
//! 64) feature vectors per image tile in the text hidden size. Those replace the
//! `<image>` placeholder positions in the prompt embedding stream via
//! [`crate::vision::merge::merge_llava`], matching the Idefics2 `masked_scatter`
//! merge.
//!
//! Scope: this port feeds a single aspect-preserving tile per image (idefics2's
//! `do_image_splitting=False` mode), which validates to a correct caption on the
//! released checkpoint. The tile is resized to fit within the processor's
//! shortest/longest-edge budget (cropped to whole patches), and its `(grid_h,
//! grid_w)` patch grid is mapped into the fixed position table via the same
//! bucketized position ids the reference uses (see [`bucketize_position_ids`]).
//! Idefics2's optional multi-tile `do_image_splitting` (4 crops + a global tile)
//! is a resolution/quality refinement on the same per-tile math and is a
//! documented follow-up.
//!
//! Used by: `loading::load_idefics2_vlm`, `multimodal::vlm_runtime`.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::Llama3Model;
use crate::models::llama3::{MLP, ModelArgs};
use crate::vision::encoders::siglip::SigLipVisionModel;
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::idefics2::Idefics2Processor;

/// Idefics2 perceiver-resampler head configuration. The mlx-community 4-bit
/// checkpoint strips these fields from `perceiver_config`, so the loader falls
/// back to the released `HuggingFaceM4/idefics2-8b` defaults. `n_latents` and
/// `depth` are derived from the weights (latents shape and layer count); only
/// the attention head split needs these values.
#[derive(Debug, Clone, Copy)]
pub struct Idefics2PerceiverHeads {
    /// `resampler_n_heads` (query heads).
    pub n_heads: usize,
    /// `resampler_head_dim`.
    pub head_dim: usize,
    /// `num_key_value_heads` (grouped-query key/value heads).
    pub n_kv_heads: usize,
}

impl Default for Idefics2PerceiverHeads {
    fn default() -> Self {
        // HuggingFaceM4/idefics2-8b perceiver_config defaults.
        Self {
            n_heads: 16,
            head_dim: 96,
            n_kv_heads: 4,
        }
    }
}

/// Idefics2 bucketized vision position ids: map a `(grid_h, grid_w)` patch grid
/// into the fixed `num_patches_per_side^2` position table, matching the
/// reference `Idefics2VisionEmbeddings`. `num_patches_per_side = image_size /
/// patch_size` (70 for the released checkpoint). The reference's `- 1` offset
/// makes the first fractional coord's bucket `-1`, so a raw id can be negative;
/// MLX embedding indexing wraps negatives Python-style, so we wrap them into
/// `[0, num_positions)` here to match bit-for-bit.
fn bucketize_position_ids(grid_h: usize, grid_w: usize, num_patches_per_side: usize) -> Vec<i32> {
    let n = num_patches_per_side.max(1);
    let nf = n as f64;
    // boundaries = linspace(1/n, 1.0, n, endpoint=false)
    let step = (1.0 - 1.0 / nf) / nf;
    let boundaries: Vec<f64> = (0..n).map(|k| 1.0 / nf + k as f64 * step).collect();
    // bucket(grid): digitize(linspace(0,1,grid,endpoint=false), boundaries, right=true) - 1,
    // where digitize(x, boundaries, right=true) == count(boundaries < x).
    let bucket = |grid: usize| -> Vec<i32> {
        (0..grid)
            .map(|k| {
                let x = k as f64 / grid.max(1) as f64;
                let idx = boundaries.iter().filter(|&&b| b < x).count() as i32;
                idx - 1
            })
            .collect()
    };
    let bh = bucket(grid_h);
    let bw = bucket(grid_w);
    let num_positions = (n * n) as i32;
    let mut ids = Vec::with_capacity(grid_h * grid_w);
    for &h in &bh {
        for &w in &bw {
            let mut id = h * n as i32 + w;
            if id < 0 {
                id += num_positions; // Python-style wrap, matching MLX embedding.
            }
            ids.push(id);
        }
    }
    ids
}

/// One perceiver cross-attention block: latents attend to `concat(context,
/// latents)`. Grouped-query attention (16 query heads, 4 kv heads by default),
/// no RoPE, no causal mask.
struct Idefics2PerceiverAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Idefics2PerceiverAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        heads: Idefics2PerceiverHeads,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.k_proj"), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.v_proj"), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), group_size, bits)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads: heads.n_heads as i32,
            n_kv_heads: heads.n_kv_heads as i32,
            head_dim: heads.head_dim as i32,
            scale: (heads.head_dim as f32).powf(-0.5),
        })
    }

    /// `latents_normed`: `[B, n_latents, hidden]`; `context_normed`:
    /// `[B, ctx, hidden]`. Returns `[B, n_latents, hidden]`.
    fn forward(&self, latents_normed: &MlxArray, context_normed: &MlxArray) -> UniquePtr<MlxArray> {
        let lshape = mlxcel_core::array_shape(latents_normed);
        let b = lshape[0];
        let n_lat = lshape[1];

        // Queries from the latents only; keys/values from concat(context, latents).
        let q = self.q_proj.forward(latents_normed);
        let kv_input = mlxcel_core::concatenate(context_normed, latents_normed, 1);
        let k = self.k_proj.forward(&kv_input);
        let v = self.v_proj.forward(&kv_input);
        let kv_len = mlxcel_core::array_shape(&k)[1];

        // [B, n_lat, n_heads*hd] -> [B, n_heads, n_lat, hd]
        let q = mlxcel_core::reshape(&q, &[b, n_lat, self.n_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        // [B, kv_len, n_kv*hd] -> [B, n_kv, kv_len, hd]
        let k = mlxcel_core::reshape(&k, &[b, kv_len, self.n_kv_heads, self.head_dim]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::reshape(&v, &[b, kv_len, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Grouped-query SDPA (kv heads broadcast to query heads), no mask.
        let out = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q,
                &k,
                &v,
                self.scale,
                std::ptr::null(),
                0.0,
                0,
            )
        };

        // [B, n_heads, n_lat, hd] -> [B, n_lat, n_heads*hd]
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, n_lat, self.n_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }
}

/// One `Idefics2PerceiverLayer`: pre-norm cross-attention + pre-norm SwiGLU MLP,
/// each with a residual connection.
struct Idefics2PerceiverLayer {
    input_latents_norm: RMSNorm,
    input_context_norm: RMSNorm,
    self_attn: Idefics2PerceiverAttention,
    post_attention_layernorm: RMSNorm,
    mlp: MLP,
}

impl Idefics2PerceiverLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        args: &ModelArgs,
        heads: Idefics2PerceiverHeads,
        eps: f32,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        Ok(Self {
            input_latents_norm: load_rms_norm(
                weights,
                &format!("{prefix}.input_latents_norm"),
                eps,
            )?,
            input_context_norm: load_rms_norm(
                weights,
                &format!("{prefix}.input_context_norm"),
                eps,
            )?,
            self_attn: Idefics2PerceiverAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                heads,
                group_size,
                bits,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.post_attention_layernorm"),
                eps,
            )?,
            mlp: MLP::from_weights(weights, args, &format!("{prefix}.mlp"))?,
        })
    }

    /// `latents`: `[B, n_latents, hidden]`; `context`: `[B, ctx, hidden]`.
    fn forward(&self, latents: &MlxArray, context: &MlxArray) -> UniquePtr<MlxArray> {
        let residual = mlxcel_core::copy(latents);
        let latents_normed = self.input_latents_norm.forward(latents);
        let context_normed = self.input_context_norm.forward(context);
        let attn = self.self_attn.forward(&latents_normed, &context_normed);
        let latents = mlxcel_core::add(&residual, &attn);

        let residual = mlxcel_core::copy(&latents);
        let normed = self.post_attention_layernorm.forward(&latents);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&residual, &mlp_out)
    }
}

/// `Idefics2PerceiverResampler`: `n_latents` learned query slots refined through
/// `depth` cross-attention layers over the (modality-projected) image features,
/// then a final RMS norm. Output: `[B, n_latents, hidden]`.
struct Idefics2PerceiverResampler {
    latents: UniquePtr<MlxArray>,
    layers: Vec<Idefics2PerceiverLayer>,
    norm: RMSNorm,
    n_latents: i32,
    hidden: i32,
}

impl Idefics2PerceiverResampler {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        args: &ModelArgs,
        heads: Idefics2PerceiverHeads,
        eps: f32,
    ) -> Result<Self, String> {
        let latents = weights
            .get(&format!("{prefix}.latents"))
            .map(|v| mlxcel_core::copy(v))
            .ok_or_else(|| format!("Weight not found: {prefix}.latents"))?;
        let latents_shape = mlxcel_core::array_shape(&latents);
        let n_latents = latents_shape[0];
        let hidden = latents_shape[1];

        // Derive the layer count from the checkpoint (the 4-bit config strips
        // `resampler_depth`).
        let mut depth = 0usize;
        while weights
            .get(&format!(
                "{prefix}.layers.{depth}.input_latents_norm.weight"
            ))
            .is_some()
        {
            depth += 1;
        }
        if depth == 0 {
            return Err(format!("No perceiver layers found under {prefix}.layers"));
        }

        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            layers.push(Idefics2PerceiverLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                args,
                heads,
                eps,
            )?);
        }

        let norm = load_rms_norm(weights, &format!("{prefix}.norm"), eps)?;
        Ok(Self {
            latents,
            layers,
            norm,
            n_latents,
            hidden,
        })
    }

    /// `context`: `[B, ctx, hidden]` (modality-projected image features).
    /// Returns `[B, n_latents, hidden]`.
    fn forward(&self, context: &MlxArray) -> UniquePtr<MlxArray> {
        let batch = mlxcel_core::array_shape(context)[0];
        // Expand the shared latents to the batch (per-tile) dimension.
        let latents = mlxcel_core::reshape(&self.latents, &[1, self.n_latents, self.hidden]);
        let mut latents =
            mlxcel_core::broadcast_to(&latents, &[batch, self.n_latents, self.hidden]);
        for layer in &self.layers {
            latents = layer.forward(&latents, context);
        }
        self.norm.forward(&latents)
    }
}

/// The Idefics2 vision-to-language connector: a SwiGLU `modality_projection`
/// (SigLIP hidden -> text hidden) followed by the perceiver resampler.
pub struct Idefics2Connector {
    modality_projection: MLP,
    perceiver_resampler: Idefics2PerceiverResampler,
}

impl Idefics2Connector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        args: &ModelArgs,
        heads: Idefics2PerceiverHeads,
        eps: f32,
    ) -> Result<Self, String> {
        let modality_projection =
            MLP::from_weights(weights, args, &format!("{prefix}.modality_projection"))?;
        let perceiver_resampler = Idefics2PerceiverResampler::from_weights(
            weights,
            &format!("{prefix}.perceiver_resampler"),
            args,
            heads,
            eps,
        )?;
        Ok(Self {
            modality_projection,
            perceiver_resampler,
        })
    }

    /// `vision_features`: `[N, patches, vision_hidden]` -> `[N, n_latents,
    /// text_hidden]` projected, compressed image tokens.
    pub fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.modality_projection.forward(vision_features);
        self.perceiver_resampler.forward(&projected)
    }
}

/// Number of image-feature tokens emitted per tile (the perceiver latent count).
impl Idefics2Connector {
    pub fn num_image_token(&self) -> usize {
        self.perceiver_resampler.n_latents as usize
    }
}

/// Top-level Idefics2 (`idefics2`) VLM runtime.
pub struct Idefics2Model {
    pub text_model: Llama3Model,
    pub vision_model: SigLipVisionModel,
    pub connector: Idefics2Connector,
    pub processor: Idefics2Processor,
    /// Token id of the `<image>` placeholder (`image_token_id` in config.json;
    /// 32001 for the released Idefics2 checkpoints). Image features are merged at
    /// these positions.
    pub image_token_id: i32,
    /// Token id of `<fake_token_around_image>` (frames each image block). `0`
    /// when the tokenizer does not expose it.
    pub fake_image_token_id: i32,
    /// Number of image feature tokens emitted per tile (perceiver `n_latents`).
    pub num_image_token: usize,
    /// Vision patch size (14) — divides the tile into the patch grid.
    pub patch_size: usize,
    /// `image_size / patch_size` (70) — the side length of the fixed position
    /// table the bucketized ids index into.
    pub num_patches_per_side: usize,
    /// EOS/stop token ids consumed by the server stop path.
    pub eos_token_ids: Vec<i32>,
}

impl Idefics2Model {
    /// Compute merged input embeddings for a request that carries pixel values.
    /// Mirrors `Model.get_input_embeddings` in upstream `idefics2.py`.
    ///
    /// `pixel_values`: channels-first `[1, C, H, W]` for the single aspect-
    /// preserving tile (H, W are whole multiples of the patch size). The SigLIP
    /// tower is channels-last, so we transpose once here, matching the reference
    /// `pixel_values.transpose(0, 2, 3, 1)`, and feed bucketized position ids for
    /// the tile's `(grid_h, grid_w)` patch grid.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);
        // [1, C, H, W] -> [1, H, W, C] for the conv patch embed.
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);

        // Patch grid from the (already patch-aligned) tile dimensions.
        let shape = mlxcel_core::array_shape(&pv);
        let ps = self.patch_size.max(1) as i32;
        let grid_h = (shape[1] / ps).max(1) as usize;
        let grid_w = (shape[2] / ps).max(1) as usize;
        let position_ids = bucketize_position_ids(grid_h, grid_w, self.num_patches_per_side);

        // SigLIP tower with bucketized ids -> [1, grid_h*grid_w, vision_hidden].
        let vision_output = self
            .vision_model
            .forward_with_position_ids(&pv, &position_ids);

        // Run the connector in f32. The modality-projection SwiGLU (14336-wide)
        // and the perceiver MLP (16384-wide) overflow f16 to inf on some images,
        // which corrupts the merged prefix. The reference sidesteps this by
        // casting image features to the (f32) pixel dtype before the connector;
        // mirror that here. merge_llava casts the result back to the text dtype.
        let vision_output = mlxcel_core::astype(&vision_output, mlxcel_core::dtype::FLOAT32);

        // modality_projection + perceiver -> [1, num_image_token, text_hidden].
        let image_features = self.connector.forward(&vision_output);

        merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
    }
}

fn load_rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    Ok(RMSNorm::new(weight, eps))
}

// LanguageModel: text-only forward paths delegate straight to the underlying
// Mistral (Llama) backbone, including `forward_with_embeddings` used by the VLM
// runtime to inject pre-merged inputs. EOS ids are overridden so the server stop
// path uses Idefics2's configured stop tokens.
impl LanguageModel for Idefics2Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched(input_ids, batch_caches, mask)
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched_with_context(input_ids, batch_caches, mask, context)
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_batched_with_context_and_ids(
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        self.text_model.embed_tokens(input_ids)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn supports_batched_prefill(&self) -> bool {
        self.text_model.supports_batched_prefill()
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        self.text_model.supports_maskless_padded_prefill()
    }

    fn supports_paged_decode_backend(&self) -> bool {
        self.text_model.supports_paged_decode_backend()
    }
}

#[cfg(test)]
mod tests {
    use super::bucketize_position_ids;

    #[test]
    fn bucketize_matches_reference_for_34x45_grid() {
        // cats.jpg single-tile grid. Reference raw ids
        // [-71,-70,-68,-67,-65,-64,-62,-60] wrap (+4900) into the values below.
        let ids = bucketize_position_ids(34, 45, 70);
        assert_eq!(ids.len(), 34 * 45);
        assert_eq!(&ids[..8], &[4829, 4830, 4832, 4833, 4835, 4836, 4838, 4840]);
        // Every id indexes the 70*70 = 4900-entry position table.
        assert!(ids.iter().all(|&i| (0..4900).contains(&i)));
    }

    #[test]
    fn bucketize_full_grid_is_in_range() {
        let ids = bucketize_position_ids(70, 70, 70);
        assert_eq!(ids.len(), 4900);
        assert!(ids.iter().all(|&i| (0..4900).contains(&i)));
    }
}
