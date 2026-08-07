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

//! Jina VLM vision tower (SigLIP-so400m class) and its VL connector.
//!
//! The tower is a plain pre-norm ViT: a *linear* patch embedding over already
//! patchified pixels (`[B, n_patches, patch*patch*3]`), a learned absolute
//! positional table, no class token, LayerNorm (with bias) pre-norms, fused
//! `attn.qkv`, and a non-gated `ffn.up -> tanh-GELU -> ffn.down`. It is not the
//! HF SigLIP tensor layout, so [`crate::vision::encoders::siglip`] cannot load
//! it: that module expects `q_proj`/`k_proj`/`v_proj` and `layer_norm1/2`.
//!
//! What makes this family distinct is the connector, which is the Molmo recipe:
//!
//! 1. Features are taken from **two intermediate ViT layers** and concatenated
//!    on the feature axis (`vit_layers: [-4, -10]`, so 2 x 1152 = 2304 wide,
//!    which is exactly the input width of `vl_connector.pooling.q`).
//! 2. Two learned `pad_embed` rows are added where the per-patch coverage mask
//!    says the patch is fully or partially padding.
//! 3. The 27x27 patch grid is zero-padded to 28x28 and split into 2x2 windows;
//!    each window is pooled by a single cross-attention query equal to the mean
//!    of its four patches (`pooling_type: attention_meanq`).
//! 4. A SwiGLU projector maps the pooled 1152-wide vector into the 2048-wide
//!    text hidden size.
//!
//! Layer indexing has one trap. The reference collects `hidden_states` as the
//! 27 per-layer outputs **plus** the `post_lnorm` output, so the list has 28
//! entries and `vit_layers = [-4, -10]` resolves against 28, selecting the
//! outputs of layers 24 and 18 rather than 23 and 17. Layers 25/26 and
//! `post_norm` are then never consumed, so they are not even loaded.
//!
//! Reference: the checkpoint's own `blocks_jvlm.py` / `modeling_jvlm.py`
//! (`VisionLanguageConnector`, `JinaVLMVisionModel.forward`), cross-checked
//! against references/mlx-vlm/mlx_vlm/models/jina_vlm/vision.py.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Static configuration for the Jina VLM vision tower plus connector.
#[derive(Debug, Clone, PartialEq)]
pub struct JinaVlmVisionConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub head_dim: i32,
    pub patch_size: i32,
    /// Square crop side in pixels (`input_size[0]`).
    pub image_size: i32,
    pub num_channels: i32,
    pub intermediate_size: i32,
    pub layer_norm_eps: f32,
    pub use_cls_token: bool,
    pub post_layer_norm: bool,
    /// ViT layer indices to concatenate, in order (negative allowed).
    pub vit_layers: Vec<i32>,
    /// Text hidden size the projector emits.
    pub output_size: i32,
    /// Pooling cross-attention head count / head dim.
    pub pooling_num_heads: i32,
    pub pooling_head_dim: i32,
    pub connector_hidden_size: i32,
    pub pooling_h: i32,
    pub pooling_w: i32,
    pub group_size: i32,
    pub bits: i32,
}

impl Default for JinaVlmVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            head_dim: 72,
            patch_size: 14,
            image_size: 378,
            num_channels: 3,
            intermediate_size: 4304,
            layer_norm_eps: 1e-6,
            use_cls_token: false,
            post_layer_norm: true,
            vit_layers: vec![-4, -10],
            output_size: 2048,
            pooling_num_heads: 16,
            pooling_head_dim: 72,
            connector_hidden_size: 6144,
            pooling_h: 2,
            pooling_w: 2,
            group_size: 64,
            bits: 4,
        }
    }
}

impl JinaVlmVisionConfig {
    /// Patches per crop side (27 for 378/14).
    pub fn crop_patches(&self) -> i32 {
        self.image_size / self.patch_size
    }

    /// Flattened pixel width of one patch (588 for 14x14x3).
    pub fn patch_dim(&self) -> i32 {
        self.patch_size * self.patch_size * self.num_channels
    }

    /// Pooled tokens per crop side (round-up division: 14 for 27 / 2).
    pub fn token_length(&self) -> (i32, i32) {
        (
            self.crop_patches().div_euclid(self.pooling_h)
                + i32::from(self.crop_patches() % self.pooling_h != 0),
            self.crop_patches().div_euclid(self.pooling_w)
                + i32::from(self.crop_patches() % self.pooling_w != 0),
        )
    }

    /// Length of the reference `hidden_states` list: one entry per layer plus
    /// the `post_lnorm` output when it exists. Negative `vit_layers` resolve
    /// against this, not against `num_hidden_layers`.
    pub fn hidden_state_slots(&self) -> usize {
        self.num_hidden_layers + usize::from(self.post_layer_norm)
    }

    /// Resolve `vit_layers` to non-negative indices into the hidden-state list,
    /// preserving the configured order (the concatenation order matters because
    /// it fixes which half of `pooling.q`'s 2304-wide input each layer feeds).
    pub fn resolved_vit_layers(&self) -> Result<Vec<usize>, String> {
        let slots = self.hidden_state_slots() as i32;
        self.vit_layers
            .iter()
            .map(|&l| {
                let resolved = if l < 0 { l + slots } else { l };
                if resolved < 0 || resolved >= slots {
                    Err(format!(
                        "vit_layers entry {l} resolves to {resolved}, outside the \
                         {slots} available hidden states"
                    ))
                } else {
                    Ok(resolved as usize)
                }
            })
            .collect()
    }
}

// ViT MLP: non-gated `up -> tanh-GELU -> down`, both carrying biases.
struct JinaVlmViTMLP {
    up: UnifiedLinear,
    down: UnifiedLinear,
}

impl JinaVlmViTMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.up.forward(x);
        let h = mlxcel_core::gelu_approx(&h);
        self.down.forward(&h)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            up: UnifiedLinear::from_weights(weights, &format!("{prefix}.up"), group_size, bits)?,
            down: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down"),
                group_size,
                bits,
            )?,
        })
    }
}

/// Multi-head attention over a fused `qkv` projection.
///
/// Also serves the connector's pooling block, which is the same module with
/// `self_attn = false`: there the query comes from a separate `q` projection
/// and `k`/`v` from a fused `kv` projection over a different input.
struct JinaVlmAttention {
    /// `Some` for tower self-attention (`attn.qkv`).
    qkv: Option<UnifiedLinear>,
    /// `Some` for the connector's cross-attention (`pooling.q` + `pooling.kv`).
    q_and_kv: Option<(UnifiedLinear, UnifiedLinear)>,
    out: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl JinaVlmAttention {
    /// `inputs_kv` is `None` for self-attention.
    fn forward(&self, inputs_q: &MlxArray, inputs_kv: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let inner = self.num_heads * self.head_dim;

        let (q, k, v) = match (&self.qkv, &self.q_and_kv) {
            (Some(qkv), _) => {
                let fused = qkv.forward(inputs_q);
                (
                    mlxcel_core::slice_last_dim(&fused, 0, inner),
                    mlxcel_core::slice_last_dim(&fused, inner, inner * 2),
                    mlxcel_core::slice_last_dim(&fused, inner * 2, inner * 3),
                )
            }
            (None, Some((q_proj, kv_proj))) => {
                let kv_source = inputs_kv.unwrap_or(inputs_q);
                let q = q_proj.forward(inputs_q);
                let kv = kv_proj.forward(kv_source);
                (
                    q,
                    mlxcel_core::slice_last_dim(&kv, 0, inner),
                    mlxcel_core::slice_last_dim(&kv, inner, inner * 2),
                )
            }
            _ => unreachable!("attention must own either a fused qkv or a q/kv pair"),
        };

        let q_shape = mlxcel_core::array_shape(&q);
        let bsz = q_shape[0];
        let q_len = q_shape[1];
        let kv_len = mlxcel_core::array_shape(&k)[1];

        let q = mlxcel_core::reshape(&q, &[bsz, q_len, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[bsz, kv_len, self.num_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[bsz, kv_len, self.num_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

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

        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[bsz, q_len, inner]);
        self.out.forward(&out)
    }

    fn self_attention(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            qkv: Some(UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.qkv"),
                group_size,
                bits,
            )?),
            q_and_kv: None,
            out: UnifiedLinear::from_weights(weights, &format!("{prefix}.out"), group_size, bits)?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn cross_attention(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let q = UnifiedLinear::from_weights(weights, &format!("{prefix}.q"), group_size, bits)?;
        let kv = UnifiedLinear::from_weights(weights, &format!("{prefix}.kv"), group_size, bits)?;
        Ok(Self {
            qkv: None,
            q_and_kv: Some((q, kv)),
            out: UnifiedLinear::from_weights(weights, &format!("{prefix}.out"), group_size, bits)?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }
}

// ViT residual block.
struct JinaVlmVisionBlock {
    attn_norm: LayerNorm,
    attn: JinaVlmAttention,
    ffn_norm: LayerNorm,
    ffn: JinaVlmViTMLP,
}

impl JinaVlmVisionBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.attn_norm.forward(x);
        let attn_out = self.attn.forward(&normed, None);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.ffn_norm.forward(&h);
        let ffn_out = self.ffn.forward(&normed);
        mlxcel_core::add(&h, &ffn_out)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &JinaVlmVisionConfig,
    ) -> Result<Self, String> {
        Ok(Self {
            attn_norm: layer_norm_from_weights(
                weights,
                &format!("{prefix}.attn_norm"),
                config.layer_norm_eps,
            )?,
            attn: JinaVlmAttention::self_attention(
                weights,
                &format!("{prefix}.attn"),
                config.num_attention_heads,
                config.head_dim,
                config.group_size,
                config.bits,
            )?,
            ffn_norm: layer_norm_from_weights(
                weights,
                &format!("{prefix}.ffn_norm"),
                config.layer_norm_eps,
            )?,
            ffn: JinaVlmViTMLP::from_weights(
                weights,
                &format!("{prefix}.ffn"),
                config.group_size,
                config.bits,
            )?,
        })
    }
}

/// The SigLIP-class ViT tower, truncated to the deepest layer `vit_layers`
/// actually consumes.
pub struct JinaVlmVisionTower {
    patch_embed: UnifiedLinear,
    pos_embed: UniquePtr<MlxArray>,
    layers: Vec<JinaVlmVisionBlock>,
    post_norm: Option<LayerNorm>,
    /// Hidden-state indices to keep, in `vit_layers` order.
    vit_layers: Vec<usize>,
    /// Index of the `post_lnorm` slot in the reference hidden-state list.
    post_norm_slot: usize,
    patch_dim: i32,
}

impl JinaVlmVisionTower {
    /// Run the tower over `[B, n_patches, patch_dim]` and return the requested
    /// layers concatenated on the feature axis: `[B, n_patches, len(vit_layers)
    /// * hidden]`.
    pub fn forward_features(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // The processor emits exactly `patch_dim` pixels per patch; pad only if
        // a narrower layout ever reaches us so the matmul stays well-formed.
        let shape = mlxcel_core::array_shape(x);
        let width = shape[shape.len() - 1];
        let x = if width < self.patch_dim {
            let pad_width = [0, 0, 0, 0, 0, self.patch_dim - width];
            mlxcel_core::pad(x, &pad_width, 0.0)
        } else {
            mlxcel_core::copy(x)
        };

        let weight_dtype = mlxcel_core::array_dtype(&self.pos_embed);
        let x = if mlxcel_core::array_dtype(&x) != weight_dtype {
            mlxcel_core::astype(&x, weight_dtype)
        } else {
            x
        };

        let mut h = self.patch_embed.forward(&x);

        // pos_embed is stored 2-D `[num_patches, hidden]`; broadcast over batch.
        let pe_shape = mlxcel_core::array_shape(&self.pos_embed);
        let pe = mlxcel_core::reshape(&self.pos_embed, &[1, pe_shape[0], pe_shape[1]]);
        let pe = mlxcel_core::astype(&pe, mlxcel_core::array_dtype(&h));
        h = mlxcel_core::add(&h, &pe);

        // Keep only the hidden states the connector consumes; the tower is
        // already truncated, but a 27-layer stack of `[5, 729, 1152]` states
        // would still be ~200 MB of dead intermediates.
        let mut kept: Vec<Option<UniquePtr<MlxArray>>> =
            (0..self.vit_layers.len()).map(|_| None).collect();

        for (idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            stash_hidden_state(&self.vit_layers, idx, &h, &mut kept);
        }
        if let Some(post_norm) = &self.post_norm {
            let normed = post_norm.forward(&h);
            stash_hidden_state(&self.vit_layers, self.post_norm_slot, &normed, &mut kept);
        }

        let mut features: Option<UniquePtr<MlxArray>> = None;
        for slot in kept.into_iter() {
            let value = slot.expect("every requested ViT layer is produced by the tower");
            features = Some(match features {
                None => value,
                Some(acc) => mlxcel_core::concatenate(&acc, &value, -1),
            });
        }
        features.expect("vit_layers is never empty")
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &JinaVlmVisionConfig,
        vit_layers: Vec<usize>,
    ) -> Result<Self, String> {
        let post_norm_slot = config.num_hidden_layers;
        let needs_post_norm = vit_layers.contains(&post_norm_slot);

        // Layers beyond the deepest consumed index contribute nothing; skipping
        // them saves both load time and resident memory.
        let deepest = vit_layers
            .iter()
            .filter(|&&l| l < post_norm_slot)
            .copied()
            .max()
            .map(|l| l + 1)
            .unwrap_or(0);
        let num_layers = if needs_post_norm {
            config.num_hidden_layers
        } else {
            deepest.min(config.num_hidden_layers)
        };

        let patch_embed = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.patch_embed.proj"),
            config.group_size,
            config.bits,
        )?;
        let pos_embed = weights
            .get(&format!("{prefix}.pos_embed"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.pos_embed"))?;
        // `forward_features` indexes `pe_shape[0]` and `pe_shape[1]` to broadcast
        // the table over the batch. A 1-D table is a Rust panic on `[1]`, and a
        // 3-D one silently drops elements and then aborts inside MLX's reshape,
        // so the rank is pinned here where it is still an `Err`.
        let pe_shape = mlxcel_core::array_shape(&pos_embed);
        if pe_shape.len() != 2 {
            return Err(format!(
                "unexpected {prefix}.pos_embed shape {pe_shape:?}: expected a 2-D \
                 [num_patches, hidden] table"
            ));
        }

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(JinaVlmVisionBlock::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                config,
            )?);
        }

        let post_norm = if needs_post_norm {
            Some(layer_norm_from_weights(
                weights,
                &format!("{prefix}.post_norm"),
                config.layer_norm_eps,
            )?)
        } else {
            None
        };

        Ok(Self {
            patch_embed,
            pos_embed,
            layers,
            post_norm,
            vit_layers,
            post_norm_slot,
            patch_dim: config.patch_dim(),
        })
    }
}

/// SwiGLU projector. The fused `gate_up` is ordered `[up, gate]`.
struct JinaVlmProjector {
    gate_up: UnifiedLinear,
    down: UnifiedLinear,
}

impl JinaVlmProjector {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.gate_up.forward(x);
        let shape = mlxcel_core::array_shape(&projected);
        let half = shape[shape.len() - 1] / 2;
        let up = mlxcel_core::slice_last_dim(&projected, 0, half);
        let gate = mlxcel_core::slice_last_dim(&projected, half, half * 2);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down.forward(&activated)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_up: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_up"),
                group_size,
                bits,
            )?,
            down: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down"),
                group_size,
                bits,
            )?,
        })
    }
}

/// Vision tower + `vl_connector`, producing text-space image features.
pub struct JinaVlmVisionModel {
    tower: JinaVlmVisionTower,
    pooling: JinaVlmAttention,
    projector: JinaVlmProjector,
    /// `[2, hidden * len(vit_layers)]`: row 0 for fully padded patches, row 1
    /// for partially padded ones (`padding_embed_type: pad_and_partial_pad`).
    ///
    /// Optional on purpose: `padding_embed_type` is a config choice and a
    /// conversion that leaves it unset ships no `vl_connector.pad_embed`, so a
    /// missing tensor must load rather than fail. When it is absent
    /// [`JinaVlmVisionModel::apply_pad_embed`] is a no-op, which means padded
    /// and partially padded patches carry the tower's raw features instead of
    /// the learned pad markers. For a checkpoint that was trained with them,
    /// that degrades output on any image whose tiling does not divide evenly,
    /// silently and without a diagnostic.
    pad_embed: Option<UniquePtr<MlxArray>>,
    config: JinaVlmVisionConfig,
}

impl JinaVlmVisionModel {
    pub fn config(&self) -> &JinaVlmVisionConfig {
        &self.config
    }

    /// `images`: `[B, n_crops, n_patches, patch_dim]`.
    /// `image_masks`: `[B, n_crops, n_patches]` per-patch coverage in `[0, 1]`,
    /// or `-1` for the trailing sentinel row the processor appends.
    ///
    /// Returns `[B, n_crops, token_h * token_w, output_size]`.
    pub fn forward(&self, images: &MlxArray, image_masks: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(images);
        let batch = shape[0];
        let crops = shape[1];
        let patches = shape[2];
        let patch_dim = shape[3];

        let flat = mlxcel_core::reshape(images, &[batch * crops, patches, patch_dim]);

        // A crop consisting entirely of -1 is padding; its features are zeroed.
        // (Reference `JinaVLMVisionModel.forward`: `~all(patches == -1)`.)
        let neg_one = mlxcel_core::from_slice_f32(&[-1.0], &[1]);
        let is_neg = mlxcel_core::equal(&flat, &neg_one);
        let is_neg_i = mlxcel_core::astype(&is_neg, mlxcel_core::dtype::INT32);
        let per_patch = mlxcel_core::sum_axis(&is_neg_i, 2, false);
        let per_crop = mlxcel_core::sum_axis(&per_patch, 1, false);
        let total = mlxcel_core::from_slice_i32(&[patches * patch_dim], &[1]);
        let all_pad = mlxcel_core::greater_equal(&per_crop, &total);
        let one = mlxcel_core::from_slice_i32(&[1], &[1]);
        let all_pad_i = mlxcel_core::astype(&all_pad, mlxcel_core::dtype::INT32);
        let keep = mlxcel_core::subtract(&one, &all_pad_i);

        let features = self.tower.forward_features(&flat);
        let feat_shape = mlxcel_core::array_shape(&features);
        let feat_dim = feat_shape[feat_shape.len() - 1];

        let keep_f = mlxcel_core::astype(&keep, mlxcel_core::array_dtype(&features));
        let keep_f = mlxcel_core::reshape(&keep_f, &[batch * crops, 1, 1]);
        let features = mlxcel_core::multiply(&features, &keep_f);
        let features = mlxcel_core::reshape(&features, &[batch, crops, patches, feat_dim]);

        let features = self.apply_pad_embed(features, image_masks);
        self.pool_and_project(features, batch, crops, feat_dim)
    }

    /// `pad_and_partial_pad`: add `pad_embed[0]` where the mask is exactly 0 and
    /// `pad_embed[1]` where it is below 1 but not 0.
    fn apply_pad_embed(
        &self,
        features: UniquePtr<MlxArray>,
        image_masks: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let Some(pad_embed) = self.pad_embed.as_ref() else {
            return features;
        };
        let dtype = mlxcel_core::array_dtype(&features);
        let pe_shape = mlxcel_core::array_shape(pad_embed);
        let feat_dim = pe_shape[1];

        let pad0 = mlxcel_core::slice(pad_embed, &[0, 0], &[1, feat_dim]);
        let pad1 = mlxcel_core::slice(pad_embed, &[1, 0], &[2, feat_dim]);
        let pad0 = mlxcel_core::astype(&mlxcel_core::reshape(&pad0, &[1, 1, 1, feat_dim]), dtype);
        let pad1 = mlxcel_core::astype(&mlxcel_core::reshape(&pad1, &[1, 1, 1, feat_dim]), dtype);

        let zero = mlxcel_core::from_slice_f32(&[0.0], &[1]);
        let one = mlxcel_core::from_slice_f32(&[1.0], &[1]);
        let all_pad = mlxcel_core::equal(image_masks, &zero);
        let below_one = mlxcel_core::less(image_masks, &one);
        let partial_pad = mlxcel_core::logical_and(&below_one, &mlxcel_core::logical_not(&all_pad));

        let all_pad_f = mlxcel_core::expand_dims(&mlxcel_core::astype(&all_pad, dtype), -1);
        let partial_pad_f = mlxcel_core::expand_dims(&mlxcel_core::astype(&partial_pad, dtype), -1);

        let h = mlxcel_core::add(&features, &mlxcel_core::multiply(&pad0, &all_pad_f));
        mlxcel_core::add(&h, &mlxcel_core::multiply(&pad1, &partial_pad_f))
    }

    /// 2x2 mean-query cross-attention pooling followed by the SwiGLU projector.
    fn pool_and_project(
        &self,
        features: UniquePtr<MlxArray>,
        batch: i32,
        crops: i32,
        feat_dim: i32,
    ) -> UniquePtr<MlxArray> {
        let cfg = &self.config;
        let grid = cfg.crop_patches();
        let features = mlxcel_core::reshape(&features, &[batch, crops, grid, grid, feat_dim]);

        // Zero-pad an indivisible grid (27 -> 28) so 2x2 windows tile exactly.
        let pad_h = grid % cfg.pooling_h;
        let pad_w = grid % cfg.pooling_w;
        let (features, grid_h, grid_w) = if pad_h != 0 || pad_w != 0 {
            let pad_width = [0, 0, 0, 0, 0, pad_h, 0, pad_w, 0, 0];
            (
                mlxcel_core::pad(&features, &pad_width, 0.0),
                grid + pad_h,
                grid + pad_w,
            )
        } else {
            (features, grid, grid)
        };

        let blocks_h = grid_h / cfg.pooling_h;
        let blocks_w = grid_w / cfg.pooling_w;

        let windows = mlxcel_core::reshape(
            &features,
            &[
                batch,
                crops,
                blocks_h,
                cfg.pooling_h,
                blocks_w,
                cfg.pooling_w,
                feat_dim,
            ],
        );
        let windows = mlxcel_core::transpose_axes(&windows, &[0, 1, 2, 4, 3, 5, 6]);
        let windows = mlxcel_core::reshape(
            &windows,
            &[
                batch * crops * blocks_h * blocks_w,
                cfg.pooling_h * cfg.pooling_w,
                feat_dim,
            ],
        );

        let query = mlxcel_core::mean_axis(&windows, -2, true);
        let pooled = self.pooling.forward(&query, Some(&windows));

        let pooled_dim = {
            let s = mlxcel_core::array_shape(&pooled);
            s[s.len() - 1]
        };
        let pooled =
            mlxcel_core::reshape(&pooled, &[batch, crops, blocks_h * blocks_w, pooled_dim]);
        self.projector.forward(&pooled)
    }

    pub fn from_weights(
        weights: &WeightMap,
        vision_prefix: &str,
        connector_prefix: &str,
        config: JinaVlmVisionConfig,
    ) -> Result<Self, String> {
        let vit_layers = config.resolved_vit_layers()?;
        let tower = JinaVlmVisionTower::from_weights(weights, vision_prefix, &config, vit_layers)?;

        let pooling = JinaVlmAttention::cross_attention(
            weights,
            &format!("{connector_prefix}.pooling"),
            config.pooling_num_heads,
            config.pooling_head_dim,
            config.group_size,
            config.bits,
        )?;
        let projector = JinaVlmProjector::from_weights(
            weights,
            &format!("{connector_prefix}.projector"),
            config.group_size,
            config.bits,
        )?;
        // Absence is allowed (see the field's doc comment); a wrong shape is not.
        // `apply_pad_embed` reads `pe_shape[1]` and slices rows 0 and 1 out of
        // the table: a 1-D tensor panics on the index, and a single-row one is
        // silently clamped by MLX's `slice` to a zero-row result that then
        // aborts in the following `reshape`.
        let pad_embed = match weights.get(&format!("{connector_prefix}.pad_embed")) {
            Some(w) => {
                let shape = mlxcel_core::array_shape(w);
                if shape.len() != 2 || shape[0] != 2 {
                    return Err(format!(
                        "unexpected {connector_prefix}.pad_embed shape {shape:?}: expected a 2-D \
                         [2, hidden * len(vit_layers)] table, row 0 for fully padded patches and \
                         row 1 for partially padded ones"
                    ));
                }
                Some(mlxcel_core::copy(w))
            }
            None => None,
        };

        Ok(Self {
            tower,
            pooling,
            projector,
            pad_embed,
            config,
        })
    }
}

/// Copy `value` into every `kept` slot whose requested hidden-state index is
/// `slot`. A layer may legitimately be requested twice.
fn stash_hidden_state(
    wanted: &[usize],
    slot: usize,
    value: &MlxArray,
    kept: &mut [Option<UniquePtr<MlxArray>>],
) {
    for (i, &index) in wanted.iter().enumerate() {
        if index == slot {
            kept[i] = Some(mlxcel_core::copy(value));
        }
    }
}

fn layer_norm_from_weights(
    weights: &WeightMap,
    prefix: &str,
    eps: f32,
) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, eps))
}

#[cfg(test)]
#[path = "jina_vlm_tests.rs"]
pub(crate) mod tests;
