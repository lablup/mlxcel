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

//! Falcon-OCR decoder (`model_type: falcon_ocr`, ~300M).
//!
//! Falcon-OCR is an early-fusion OCR VLM with **no vision encoder**. A 16x16
//! image patch is projected straight into the token stream by a single linear
//! layer (`img_projector`), and one decoder processes image and text together
//! under a hybrid mask that is bidirectional inside each image block and causal
//! everywhere else.
//!
//! The decoder is Llama-derived but differs in five ways, all of which are
//! visible in the checkpoint's own `modeling_falcon_ocr.py`:
//!
//! 1. **Fused QKV** (`wqkv`), split `q | k | v` on the last axis.
//! 2. **Weightless normalization.** Every pre-norm in the stack is
//!    `F.rms_norm(x, (dim,))` with no learnable weight, including the per-head
//!    Q and K norms. The checkpoint ships exactly five tensors per layer and no
//!    `attention_norm` / `ffn_norm` / `q_norm` / `k_norm`; the norms are not
//!    missing, they are non-parametric. Only the final `norm` before the LM
//!    head is parametric, and it is the only place `norm_eps` applies.
//! 3. **Per-head attention sinks**, one learned logit per head appended to the
//!    softmax denominator (`layers.N.attention.sinks`, shape `[n_heads]`).
//! 4. **Squared-ReLU gated MLP** over an *interleaved* fused `w13`: the packed
//!    projection alternates gate and up rows, and the activation is
//!    `relu(gate)^2 * up`.
//! 5. **3-D rotary**: 1-D temporal rotary on the low half of each head, 2-D
//!    per-head spatial rotary on the high half. See [`super::falcon_ocr_rope`].
//!
//! K and V are expanded from `n_kv_heads` to `n_heads` **before** the rotary,
//! because the spatial rotary has per-head frequencies and each repeated copy
//! must receive its own rotation. The KV cache therefore holds `n_heads` heads.
//!
//! References, in order of authority for this checkpoint:
//! `tiiuae/Falcon-OCR` vendor code (`modeling_falcon_ocr.py`, `rope.py`,
//! `attention.py`), then mlx-vlm `mlx_vlm/models/falcon_ocr/language.py`.

use std::cell::RefCell;
use std::collections::HashMap;

use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use super::falcon_ocr_rope::{
    FalconOcrTokenIds, NONPARAM_RMS_EPS, apply_3d_rotary, golden_cos_sin, temporal_cos_sin,
    temporal_inv_freq,
};

// Config.

/// Falcon-OCR config.
///
/// The shipped `config.json` uses the raw `dim` / `n_layers` / `ffn_dim` key
/// scheme rather than the HF `hidden_size` / `num_hidden_layers` /
/// `intermediate_size` names, so every field accepts both spellings.
#[derive(Debug, Clone, Deserialize)]
pub struct FalconOcrConfig {
    #[serde(alias = "hidden_size")]
    pub dim: usize,
    #[serde(alias = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(alias = "num_attention_heads")]
    pub n_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default, alias = "num_key_value_heads")]
    pub n_kv_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(alias = "intermediate_size")]
    pub ffn_dim: usize,
    #[serde(default = "default_norm_eps", alias = "rms_norm_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_max_seq_len", alias = "max_position_embeddings")]
    pub max_seq_len: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_channel_size")]
    pub channel_size: usize,
    #[serde(default = "default_spatial_patch_size")]
    pub spatial_patch_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,

    #[serde(default = "default_eos_id")]
    pub eos_id: i32,
    #[serde(default = "default_img_id")]
    pub img_id: i32,
    #[serde(default = "default_img_end_id")]
    pub img_end_id: i32,
    #[serde(default = "default_image_cls_token_id")]
    pub image_cls_token_id: i32,
    #[serde(default = "default_reg1")]
    pub image_reg_1_token_id: i32,
    #[serde(default = "default_reg2")]
    pub image_reg_2_token_id: i32,
    #[serde(default = "default_reg3")]
    pub image_reg_3_token_id: i32,
    #[serde(default = "default_reg4")]
    pub image_reg_4_token_id: i32,

    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    #[serde(default)]
    pub group_size: i32,
    #[serde(default)]
    pub bits: i32,
}

fn default_norm_eps() -> f32 {
    1e-5
}
fn default_max_seq_len() -> usize {
    8192
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_channel_size() -> usize {
    3
}
fn default_spatial_patch_size() -> usize {
    16
}
fn default_temporal_patch_size() -> usize {
    1
}
fn default_eos_id() -> i32 {
    11
}
fn default_img_id() -> i32 {
    227
}
fn default_img_end_id() -> i32 {
    230
}
fn default_image_cls_token_id() -> i32 {
    244
}
fn default_reg1() -> i32 {
    245
}
fn default_reg2() -> i32 {
    246
}
fn default_reg3() -> i32 {
    247
}
fn default_reg4() -> i32 {
    248
}

impl FalconOcrConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.dim / self.n_heads)
    }
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads.unwrap_or(self.n_heads)
    }
    /// `temporal_patch_size * spatial_patch_size^2 * channel_size`, the input
    /// width of `img_projector`.
    pub fn patch_dim(&self) -> usize {
        self.temporal_patch_size
            * self.spatial_patch_size
            * self.spatial_patch_size
            * self.channel_size
    }
    pub fn token_ids(&self) -> FalconOcrTokenIds {
        FalconOcrTokenIds {
            img_id: self.img_id,
            image_cls_token_id: self.image_cls_token_id,
            img_end_id: self.img_end_id,
            image_reg_token_ids: [
                self.image_reg_1_token_id,
                self.image_reg_2_token_id,
                self.image_reg_3_token_id,
                self.image_reg_4_token_id,
            ],
        }
    }
    fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(0)
    }
    fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(0)
    }
}

// Per-request runtime state.

/// Positional state that a Falcon-OCR prefill computes once and a decode step
/// keeps consuming.
pub struct FalconOcrPrefillState {
    /// Temporal rope position per prompt token. An image block collapses onto
    /// one index, so this is not `0..len`.
    pub positions: Vec<i32>,
    /// `[1, L, 2]` spatial `(h, w)` coordinates, zero outside image blocks.
    pub pos_hw: Option<UniquePtr<MlxArray>>,
    /// `positions.last() + 1 - positions.len()`; decode adds it to the cache
    /// offset to recover the absolute temporal position.
    pub rope_delta: i32,
}

impl FalconOcrPrefillState {
    fn duplicate(&self) -> Self {
        Self {
            positions: self.positions.clone(),
            pos_hw: self.pos_hw.as_ref().map(|p| mlxcel_core::copy(p)),
            rope_delta: self.rope_delta,
        }
    }
}

/// Two-slot resolver: a per-`SequenceId` map for the server, and a "current"
/// fallback for the CLI and any caller that has no sequence id.
///
/// Without the map, a burst of concurrent server requests would let one row's
/// `rope_delta` (which is negative and image-size dependent) drive another
/// row's decode positions.
#[derive(Default)]
pub struct FalconOcrRuntimeState {
    sequences: RefCell<HashMap<SequenceId, FalconOcrPrefillState>>,
    fallback: RefCell<Option<FalconOcrPrefillState>>,
}

impl FalconOcrRuntimeState {
    pub fn set_current(&self, state: FalconOcrPrefillState) {
        *self.fallback.borrow_mut() = Some(state);
    }

    /// Move the pending fallback slot under `seq_id`.
    pub fn bind_to_sequence(&self, seq_id: SequenceId) {
        if let Some(state) = self.fallback.borrow_mut().take() {
            self.sequences.borrow_mut().insert(seq_id, state);
        }
    }

    pub fn release(&self, seq_id: SequenceId) {
        self.sequences.borrow_mut().remove(&seq_id);
    }

    /// Read a field out of the resolved entry without duplicating it.
    ///
    /// Resolution order matches [`Self::resolve`]: the per-sequence map first,
    /// then the fallback slot. The borrow lives only for the closure, which is
    /// what lets a decode step look at the entry without cloning the prompt's
    /// position vector or copying `pos_hw` across the FFI boundary.
    fn with_entry<T>(
        &self,
        seq_id: Option<SequenceId>,
        f: impl FnOnce(&FalconOcrPrefillState) -> T,
    ) -> Option<T> {
        if let Some(id) = seq_id {
            let map = self.sequences.borrow();
            if let Some(entry) = map.get(&id) {
                return Some(f(entry));
            }
        }
        self.fallback.borrow().as_ref().map(f)
    }

    /// The stashed rope delta for a decode step, or `None` when this row never
    /// went through a Falcon-OCR image prefill.
    ///
    /// A decode step reads nothing else out of the entry, so it must not pay
    /// for a full duplicate: the prompt's position vector is thousands of
    /// entries for an image prompt, and duplicating it per generated token
    /// makes decode quadratic in prompt length.
    fn decode_rope_delta(&self, seq_id: Option<SequenceId>) -> Option<i32> {
        self.with_entry(seq_id, |state| state.rope_delta)
    }

    /// Resolve by sequence id first, then the fallback slot.
    fn resolve(&self, seq_id: Option<SequenceId>) -> Option<FalconOcrPrefillState> {
        if let Some(id) = seq_id
            && let Some(entry) = self.sequences.borrow().get(&id)
        {
            return Some(entry.duplicate());
        }
        self.fallback.borrow().as_ref().map(|e| e.duplicate())
    }

    /// Resolve the state for a prefill of `len` tokens, evicting a stale one.
    ///
    /// The entry is written by the image stage before generation starts, and
    /// nothing in the generate loop tells the model when a *new* request began.
    /// A prefill whose length does not match the stashed positions is therefore
    /// the boundary: it belongs to a different prompt, so the old entry is
    /// dropped rather than left to drive the next decode's rope delta. This is
    /// what makes an image turn followed by a text-only turn safe.
    ///
    /// The length is compared through [`Self::with_entry`] so the eviction path
    /// does not duplicate an entry it is about to throw away.
    fn take_for_prefill(
        &self,
        seq_id: Option<SequenceId>,
        len: usize,
    ) -> Option<FalconOcrPrefillState> {
        match self.with_entry(seq_id, |state| state.positions.len() == len) {
            Some(true) => self.resolve(seq_id),
            Some(false) => {
                if let Some(id) = seq_id {
                    self.sequences.borrow_mut().remove(&id);
                }
                *self.fallback.borrow_mut() = None;
                None
            }
            None => None,
        }
    }
}

// Attention.

struct Attention {
    wqkv: UnifiedLinear,
    wo: UnifiedLinear,
    sinks: UniquePtr<MlxArray>,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        cfg: &FalconOcrConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let head_dim = cfg.head_dim() as i32;
        let sinks = weights
            .get(&format!("{prefix}.sinks"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.sinks"))?;
        Ok(Self {
            wqkv: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.wqkv"),
                cfg.group_size(),
                cfg.bits(),
            )?,
            wo: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.wo"),
                cfg.group_size(),
                cfg.bits(),
            )?,
            sinks,
            n_heads: cfg.n_heads as i32,
            n_kv_heads: cfg.n_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        cos_1d: &MlxArray,
        sin_1d: &MlxArray,
        golden: Option<(&MlxArray, &MlxArray)>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;

        // Weightless pre-norm, then the fused QKV projection.
        let x_norm = mlxcel_core::fast_rms_norm_no_weight(x, NONPARAM_RMS_EPS);
        let qkv = self.wqkv.forward(&x_norm);

        let q = mlxcel_core::slice_last_dim(&qkv, 0, q_dim);
        let k = mlxcel_core::slice_last_dim(&qkv, q_dim, q_dim + kv_dim);
        let v = mlxcel_core::slice_last_dim(&qkv, q_dim + kv_dim, q_dim + 2 * kv_dim);

        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim]);

        // Per-head weightless RMSNorm over head_dim for Q and K only.
        let q = mlxcel_core::fast_rms_norm_no_weight(&q, NONPARAM_RMS_EPS);
        let k = mlxcel_core::fast_rms_norm_no_weight(&k, NONPARAM_RMS_EPS);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Expand GQA groups before the rotary: the spatial rotary is per-head,
        // so every repeated K copy must be rotated with its own frequencies.
        let n_rep = self.n_heads / self.n_kv_heads;
        let (k, v) = if n_rep > 1 {
            (
                mlxcel_core::repeat(&k, n_rep, 1),
                mlxcel_core::repeat(&v, n_rep, 1),
            )
        } else {
            (k, v)
        };

        let q = apply_3d_rotary(&q, cos_1d, sin_1d, golden);
        let k = apply_3d_rotary(&k, cos_1d, sin_1d, golden);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);
        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let sinks_ptr =
            self.sinks
                .as_ref()
                .expect("falcon_ocr attention sinks must be present") as *const _;
        let attn = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention_with_sinks(
                &q, &cache_k, &cache_v, self.scale, mask_ptr, sinks_ptr,
            )
        };

        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, l, self.n_heads * self.head_dim]);
        self.wo.forward(&attn)
    }
}

// Squared-ReLU gated MLP over the de-interleaved fused projection.

struct MLP {
    w1: UnifiedLinear,
    w3: UnifiedLinear,
    w2: UnifiedLinear,
}

impl MLP {
    fn from_weights(
        weights: &WeightMap,
        cfg: &FalconOcrConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let (gs, bits) = (cfg.group_size(), cfg.bits());
        Ok(Self {
            w1: UnifiedLinear::from_weights(weights, &format!("{prefix}.w1"), gs, bits)?,
            w3: UnifiedLinear::from_weights(weights, &format!("{prefix}.w3"), gs, bits)?,
            w2: UnifiedLinear::from_weights(weights, &format!("{prefix}.w2"), gs, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x_norm = mlxcel_core::fast_rms_norm_no_weight(x, NONPARAM_RMS_EPS);
        let gate = self.w1.forward(&x_norm);
        let up = self.w3.forward(&x_norm);
        let activated = mlxcel_core::multiply(&mlxcel_core::compiled_relu_squared(&gate), &up);
        self.w2.forward(&activated)
    }
}

struct TransformerBlock {
    attention: Attention,
    feed_forward: MLP,
}

impl TransformerBlock {
    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        cos_1d: &MlxArray,
        sin_1d: &MlxArray,
        golden: Option<(&MlxArray, &MlxArray)>,
    ) -> UniquePtr<MlxArray> {
        let h = mlxcel_core::add(
            x,
            &self
                .attention
                .forward(x, cache, mask, cos_1d, sin_1d, golden),
        );
        mlxcel_core::add(&h, &self.feed_forward.forward(&h))
    }
}

// Model.

pub struct FalconOcrTextModel {
    pub config: FalconOcrConfig,
    tok_embeddings: UnifiedEmbedding,
    img_projector: UnifiedLinear,
    layers: Vec<TransformerBlock>,
    norm: RMSNorm,
    output: UnifiedLinear,
    freqs_cis_golden: UniquePtr<MlxArray>,
    inv_freq: Vec<f32>,
    pub state: FalconOcrRuntimeState,
}

impl FalconOcrTextModel {
    pub fn from_weights(weights: &WeightMap, cfg: &FalconOcrConfig) -> Result<Self, String> {
        let (gs, bits) = (cfg.group_size(), cfg.bits());
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(TransformerBlock {
                attention: Attention::from_weights(weights, cfg, &format!("layers.{i}.attention"))?,
                feed_forward: MLP::from_weights(weights, cfg, &format!("layers.{i}.feed_forward"))?,
            });
        }

        let norm_w = weights
            .get("norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Weight not found: norm.weight".to_string())?;
        let freqs_cis_golden = weights
            .get("freqs_cis_golden")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Weight not found: freqs_cis_golden".to_string())?;

        // Only the low half of every head takes the temporal rotary.
        let inv_freq = temporal_inv_freq(cfg.head_dim() / 2, cfg.rope_theta);

        Ok(Self {
            tok_embeddings: UnifiedEmbedding::from_weights(weights, "tok_embeddings", gs, bits)?,
            img_projector: UnifiedLinear::from_weights(weights, "img_projector", gs, bits)?,
            layers,
            norm: RMSNorm::new(norm_w, cfg.norm_eps),
            output: UnifiedLinear::from_weights(weights, "output", gs, bits)?,
            freqs_cis_golden,
            inv_freq,
            config: cfg.clone(),
            state: FalconOcrRuntimeState::default(),
        })
    }

    pub fn embed(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.tok_embeddings.forward(input_ids)
    }

    /// Project already-patchified pixels (`[N, patch_dim]`) into token space.
    pub fn project_patches(&self, patches: &MlxArray) -> UniquePtr<MlxArray> {
        self.img_projector.forward(patches)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match input_embeddings {
            Some(embeds) => mlxcel_core::copy(embeds),
            None => self.embed(input_ids),
        };
        let shape = mlxcel_core::array_shape(&h);
        let l = shape[1];
        let offset = caches.first().map(|c| c.offset).unwrap_or(0);

        // Temporal positions: a stashed prefill vector when this row went
        // through a Falcon-OCR image prefill, otherwise the plain contiguous
        // run (which is exactly what `temporal_positions` yields for text).
        //
        // The spatial rotary rides along in the same match because it only ever
        // fires during the prefill that carries image tokens: a decode step has
        // no image position and the rotation would be the identity anyway.
        // Deciding both at once is also what keeps the decode arm from touching
        // the stashed entry at all beyond its scalar delta.
        type GoldenTables = Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>;
        let (positions, golden): (Vec<i32>, GoldenTables) = if l > 1 {
            match self.state.take_for_prefill(seq_id, l as usize) {
                Some(state) => {
                    let golden = state
                        .pos_hw
                        .as_ref()
                        .map(|p| golden_cos_sin(&self.freqs_cis_golden, p));
                    (state.positions, golden)
                }
                None => ((offset..offset + l).collect(), None),
            }
        } else if l == 1
            && let Some(rope_delta) = self.state.decode_rope_delta(seq_id)
        {
            (vec![offset + rope_delta], None)
        } else {
            ((offset..offset + l).collect(), None)
        };
        let (cos_1d, sin_1d) = temporal_cos_sin(&positions, &self.inv_freq);

        let golden_ref = golden
            .as_ref()
            .map(|(c, s)| (c.as_ref().unwrap(), s.as_ref().unwrap()));

        // A prefill with no supplied mask (text-only prompt) still needs an
        // explicit array: the sink SDPA entry point has no "causal" mode.
        let fallback_mask =
            (l > 1 && mask.is_none()).then(|| mlxcel_core::utils::create_causal_mask(l, offset));
        let mask = mask.or_else(|| fallback_mask.as_ref().map(|m| m.as_ref().unwrap()));

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, cache, mask, &cos_1d, &sin_1d, golden_ref);
        }
        let h = self.norm.forward(&h);
        self.output.forward(&h)
    }
}

impl LanguageModel for FalconOcrTextModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, None, None, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, None, seq_id, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, input_embeddings, None, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, input_embeddings, seq_id, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        FalconOcrTextModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![self.config.eos_id]
    }

    /// Falcon-OCR cannot be prefilled in chunks.
    ///
    /// The prompt-shaped hybrid mask and the temporal-position vector are both
    /// computed once for the whole prompt, and `chunked_prefill_last_logits`
    /// re-enters `forward` with `mask = None` and a slice of the tokens, which
    /// would silently drop the bidirectional image block and misalign every
    /// position after the first chunk. This mirrors the upstream mlx-vlm
    /// `Model.no_chunked_prefill = True`.
    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    /// Tile-aligned padded prefill pads the token axis, which would desync the
    /// stashed per-token position vector and the `[1, 1, S, S]` mask.
    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn supports_batching(&self) -> bool {
        false
    }

    /// Falcon-OCR stores every request's KV in the caller-supplied slice, so the
    /// scheduler has to allocate a dense per-layer cache set for it.
    ///
    /// The trait default derives the layout from `supports_batching()`, and this
    /// model declines batching only because its per-request positional state is
    /// single-row, not because it owns its state. Without this override the
    /// server classifies it as a model-owned (SSM / recurrent) runtime and hands
    /// `forward` an *empty* cache slice, which makes `layers.iter().zip(caches)`
    /// run zero times: every decoder layer is skipped and the LM head reads the
    /// raw token embedding. That is silent, so it looks like a quality problem
    /// rather than a wiring one. Mirrors the same override on Phi4MM, which
    /// declines batching for an unrelated reason.
    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::dense_kv_cache(self.layers.len())
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.state.release(seq_id);
    }

    // `reset_runtime_state` is deliberately NOT overridden. The generate loop
    // calls it *after* the image stage has written the prefill state and
    // *before* the prefill forward consumes it, so clearing there would throw
    // away the positions and the mask geometry for the request about to run.
    // Staleness is handled at the prefill boundary by `take_for_prefill`.
}

#[cfg(test)]
#[path = "falcon_ocr_tests.rs"]
mod tests;
