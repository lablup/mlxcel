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

//! LongCat-Flash-Lite-Sparse (`longcat_flash_sparse`, `LongcatCausalLM`,
//! issue #2362).
//!
//! The LongCat-Flash ScMoE decoder (two MLA sub-blocks and one MoE shortcut
//! per layer, identity zero experts) with two changes:
//!
//! - LSA sparse attention. `self_attn.0` of each layer owns a lightning
//!   indexer (`relu(q_h . k) * weights_proj(x)_h`, summed over heads) whose
//!   top-`index_topk` key selection `self_attn.1` reuses. The first
//!   `index_init_tokens` columns and the `index_local_tokens` window ending at
//!   the query are always kept (forced to `+inf` before the causal mask).
//!   Decode gathers the selected latent rows in whichever layout the cache
//!   holds them (fp16, or the affine quantized latent of #3176 / #3189);
//!   prefill folds the selection into the causal mask.
//! - The n-gram input embedding fuses as
//!   `word + post_proj(concat(lookups)) / (1 + num_embedders)` with one
//!   projection over the concatenated lookups, where `longcat_flash_ngram`
//!   divides the whole sum.
//!
//! The MLA projections, MoE, router and dense MLPs are the LongCat ones in
//! [`super::longcat_flash_ngram`]. RoPE is the checkpoint's `deepseek_yarn`
//! block (traditional for the MLA, `indexer_rope_interleave` for the indexer).
//!
//! Each layer keeps three caches, all model-owned: the latent caches of both
//! sub-blocks (which follow the KV mode table) and the indexer-key cache
//! (always fp16), plus the n-gram token tail.
//!
//! Reference: mlx-vlm `mlx_vlm/models/longcat_flash_sparse/language.py`
//! (Blaizzy/mlx-vlm#2063, #2150).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::Path;

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::dtype;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::mla::LatentWindow;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use super::gpt2::validate_embedding_table;
use super::longcat_flash_ngram::{
    LongcatFlashMLA, LongcatFlashMLP, LongcatFlashMoE, LongcatFlashNgramConfig, MlaProjections,
    NgramEmbedding, QuantizationConfig, get_weight_copy,
};
use super::model_owned::KvCacheLayerModes;
use super::rope_utils::{RopeScalingKind, RopeScalingSpec};

/// The `config.json` `model_type` of this family. The original
/// `meituan-longcat` config omits the key; detection maps its
/// `LongcatCausalLM` architecture here.
pub const MODEL_TYPE: &str = "longcat_flash_sparse";

/// `nn.LayerNorm` default epsilon, for an indexer whose `index_k_norm_type`
/// is not `rms`.
const LAYER_NORM_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
pub struct LongcatFlashSparseConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// `LSA` (the sparse indexer) or `MLA` (dense attention).
    #[serde(default = "default_attention_method")]
    pub attention_method: String,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub moe_topk: usize,
    pub expert_ffn_hidden_size: usize,
    pub n_routed_experts: usize,
    #[serde(default)]
    pub zero_expert_num: usize,
    #[serde(default = "default_zero_expert_type")]
    pub zero_expert_type: String,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    // Upstream's `ModelConfig` defaults both LoRA scales on.
    #[serde(default = "default_true")]
    pub mla_scale_q_lora: bool,
    #[serde(default = "default_true")]
    pub mla_scale_kv_lora: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub router_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub index_n_heads: usize,
    #[serde(default = "default_index_head_dim")]
    pub index_head_dim: usize,
    #[serde(default)]
    pub index_topk: usize,
    /// Leading sink columns every query keeps.
    #[serde(default)]
    pub index_init_tokens: usize,
    /// Trailing local window (ending at the query) every query keeps.
    #[serde(default)]
    pub index_local_tokens: usize,
    #[serde(default = "default_index_k_norm_type")]
    pub index_k_norm_type: String,
    #[serde(default = "default_true")]
    pub indexer_rope_interleave: bool,
    /// `0` disables the n-gram embedding.
    #[serde(default)]
    pub oe_vocab_size_ratio: usize,
    #[serde(default = "default_oe_neighbor_num")]
    pub oe_neighbor_num: usize,
    #[serde(default = "default_oe_split_num")]
    pub oe_split_num: usize,
}

mlxcel_core::impl_rope_lifting_deserialize!(LongcatFlashSparseConfig);

fn default_model_type() -> String {
    MODEL_TYPE.to_string()
}
fn default_attention_method() -> String {
    "LSA".to_string()
}
fn default_zero_expert_type() -> String {
    "identity".to_string()
}
fn default_true() -> bool {
    true
}
fn default_index_head_dim() -> usize {
    128
}
fn default_index_k_norm_type() -> String {
    "rms".to_string()
}
fn default_oe_neighbor_num() -> usize {
    4
}
fn default_oe_split_num() -> usize {
    4
}

impl LongcatFlashSparseConfig {
    /// Whether attention uses the LSA indexer.
    pub fn uses_lsa(&self) -> bool {
        self.attention_method.eq_ignore_ascii_case("lsa")
    }

    fn uses_ngram(&self) -> bool {
        self.oe_vocab_size_ratio > 0
    }

    /// The same layer geometry as the shared LongCat config, which the MLA,
    /// MoE and MLP loaders read. The n-gram keys are renamed (`oe_*` to the
    /// `longcat_flash_ngram` names); `model_type` stays this family's, so the
    /// shared sanitizer leaves the embedding keys alone.
    pub(crate) fn base(&self) -> LongcatFlashNgramConfig {
        LongcatFlashNgramConfig {
            model_type: MODEL_TYPE.to_string(),
            hidden_size: self.hidden_size,
            ffn_hidden_size: self.ffn_hidden_size,
            moe_topk: self.moe_topk,
            expert_ffn_hidden_size: self.expert_ffn_hidden_size,
            n_routed_experts: self.n_routed_experts,
            zero_expert_num: self.zero_expert_num,
            num_layers: self.num_layers,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            num_attention_heads: self.num_attention_heads,
            kv_lora_rank: self.kv_lora_rank,
            q_lora_rank: self.q_lora_rank,
            qk_rope_head_dim: self.qk_rope_head_dim,
            qk_nope_head_dim: self.qk_nope_head_dim,
            v_head_dim: self.v_head_dim,
            routed_scaling_factor: self.routed_scaling_factor,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            mla_scale_q_lora: self.mla_scale_q_lora,
            mla_scale_kv_lora: self.mla_scale_kv_lora,
            attention_bias: self.attention_bias,
            zero_expert_type: self.zero_expert_type.clone(),
            ngram_vocab_size_ratio: self.oe_vocab_size_ratio,
            emb_neighbor_num: self.oe_neighbor_num,
            emb_split_num: self.oe_split_num,
            norm_topk_prob: self.norm_topk_prob,
            router_bias: self.router_bias,
            rope_scaling: self.rope_scaling.clone(),
            quantization: self.quantization.clone(),
        }
    }

    /// Refuse a config this port would run wrongly, naming the field.
    pub fn validate(&self) -> Result<(), String> {
        let method = self.attention_method.to_ascii_lowercase();
        if method != "lsa" && method != "mla" {
            return Err(format!(
                "longcat_flash_sparse: attention_method {:?} is not LSA or MLA",
                self.attention_method
            ));
        }
        if self.zero_expert_num > 0 && self.zero_expert_type != "identity" {
            return Err(format!(
                "longcat_flash_sparse: zero_expert_type {:?} is not supported (only identity)",
                self.zero_expert_type
            ));
        }
        for (name, value) in [
            ("num_layers", self.num_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("v_head_dim", self.v_head_dim),
            ("moe_topk", self.moe_topk),
        ] {
            if value == 0 || i32::try_from(value).is_err() {
                return Err(format!(
                    "longcat_flash_sparse: {name} ({value}) must be a positive i32"
                ));
            }
        }
        if !self.qk_rope_head_dim.is_multiple_of(2) {
            return Err(format!(
                "longcat_flash_sparse: qk_rope_head_dim ({}) must be even",
                self.qk_rope_head_dim
            ));
        }
        if self.uses_lsa() {
            if self.q_lora_rank == 0 {
                return Err(
                    "longcat_flash_sparse: LSA needs a query LoRA (q_lora_rank > 0), the \
                     indexer's input"
                        .to_string(),
                );
            }
            for (name, value) in [
                ("index_n_heads", self.index_n_heads),
                ("index_head_dim", self.index_head_dim),
                ("index_topk", self.index_topk),
            ] {
                if value == 0 || i32::try_from(value).is_err() {
                    return Err(format!(
                        "longcat_flash_sparse: {name} ({value}) must be a positive i32 when \
                         attention_method is LSA"
                    ));
                }
            }
            if self.index_head_dim < self.qk_rope_head_dim {
                return Err(format!(
                    "longcat_flash_sparse: index_head_dim ({}) is narrower than the rotated \
                     qk_rope_head_dim ({})",
                    self.index_head_dim, self.qk_rope_head_dim
                ));
            }
            for (name, value) in [
                ("index_init_tokens", self.index_init_tokens),
                ("index_local_tokens", self.index_local_tokens),
            ] {
                if i32::try_from(value).is_err() {
                    return Err(format!(
                        "longcat_flash_sparse: {name} ({value}) exceeds i32::MAX"
                    ));
                }
            }
        }
        Ok(())
    }

    fn group_size(&self) -> i32 {
        self.base().group_size()
    }

    fn bits(&self) -> i32 {
        self.base().bits()
    }
}

/// The rotary table of the MLA and the indexer: the checkpoint's YaRN block
/// (`deepseek_yarn` is the same table as `yarn`), or plain frequencies.
struct SparseRope {
    kind: RopeScalingKind,
    base: f32,
}

impl SparseRope {
    fn from_config(args: &LongcatFlashSparseConfig) -> Self {
        let spec = args.rope_scaling.as_ref().map(|block| {
            let mut spec = RopeScalingSpec::from_lookup(|key| block.get(key));
            // mlx-vlm's `initialize_rope` builds one `YarnRoPE` for both
            // spellings.
            if spec.rope_type.as_deref() == Some("deepseek_yarn") {
                spec.rope_type = Some("yarn".to_string());
            }
            spec
        });
        let kind = RopeScalingKind::resolve(
            spec.as_ref(),
            args.qk_rope_head_dim,
            args.rope_theta,
            Some(args.max_position_embeddings as f32),
            MODEL_TYPE,
        );
        Self {
            kind,
            base: args.rope_theta,
        }
    }

    fn duplicate(&self) -> Self {
        Self {
            kind: self.kind.duplicate(),
            base: self.base,
        }
    }

    /// Rotate the first `dims` features of `x` at `offset`, scaling them by
    /// the YaRN magnitude first, as upstream `YarnRoPE` does.
    fn apply(
        &self,
        x: &MlxArray,
        dims: i32,
        traditional: bool,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let mscale = self.kind.attn_scale();
        let scaled;
        let x = if mscale != 1.0 {
            let width = *mlxcel_core::array_shape(x).last().unwrap_or(&dims);
            let s = mlxcel_core::full_f32(&[1], mscale, mlxcel_core::array_dtype(x));
            scaled = if width > dims {
                let head = mlxcel_core::multiply(&slice_axis(x, -1, 0, dims), &s);
                mlxcel_core::concatenate(&head, &slice_axis(x, -1, dims, width), -1)
            } else {
                mlxcel_core::multiply(x, &s)
            };
            &*scaled
        } else {
            x
        };
        match self.kind.freqs() {
            Some(freqs) => mlxcel_core::fast_rope_with_freqs(
                x,
                dims,
                traditional,
                self.kind.scale(),
                offset,
                freqs,
            ),
            None => {
                mlxcel_core::fast_rope(x, dims, traditional, self.base, self.kind.scale(), offset)
            }
        }
    }
}

enum IndexerNorm {
    Rms(RMSNorm),
    Layer(LayerNorm),
}

impl IndexerNorm {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Rms(norm) => norm.forward(x),
            Self::Layer(norm) => norm.forward(x),
        }
    }
}

/// The LSA lightning indexer of a layer's `self_attn.0`.
struct LsaIndexer {
    wq_b: UnifiedLinear,
    wk: UnifiedLinear,
    k_norm: IndexerNorm,
    weights_proj: UnifiedLinear,
    n_heads: i32,
    head_dim: i32,
    rope_dims: i32,
    rope_traditional: bool,
    rope: SparseRope,
    selection: LsaSelection,
}

/// The selection rule of the indexer, apart from its weights.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LsaSelection {
    pub(crate) topk: i32,
    pub(crate) init_tokens: i32,
    pub(crate) local_tokens: i32,
}

impl LsaIndexer {
    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashSparseConfig,
        prefix: &str,
        rope: SparseRope,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), gs, bits)
        };
        let norm_weight = get_weight_copy(weights, &format!("{prefix}.k_norm.weight"))?;
        let k_norm = if args.index_k_norm_type.eq_ignore_ascii_case("rms") {
            IndexerNorm::Rms(RMSNorm::new(norm_weight, args.rms_norm_eps))
        } else {
            let bias = weights
                .get(&format!("{prefix}.k_norm.bias"))
                .map(|b| mlxcel_core::copy(b));
            IndexerNorm::Layer(LayerNorm::new(norm_weight, bias, LAYER_NORM_EPS))
        };
        Ok(Self {
            wq_b: linear("wq_b")?,
            wk: linear("wk")?,
            k_norm,
            weights_proj: linear("weights_proj")?,
            n_heads: args.index_n_heads as i32,
            head_dim: args.index_head_dim as i32,
            rope_dims: args.qk_rope_head_dim as i32,
            rope_traditional: args.indexer_rope_interleave,
            rope,
            selection: LsaSelection {
                topk: args.index_topk as i32,
                init_tokens: args.index_init_tokens as i32,
                local_tokens: args.index_local_tokens as i32,
            },
        })
    }

    /// Append this step's indexer keys to `cache` and select each query's
    /// key positions: `[B, 1, L, index_topk]`, or `None` while the window is
    /// at most `index_topk` long (every key is kept).
    fn select(
        &self,
        x: &MlxArray,
        qr: &MlxArray,
        cache: &mut KVCache,
    ) -> Option<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(x);
        let (b, s) = (shape[0], shape[1]);
        let offset = cache.offset;

        let q = self.wq_b.forward(qr);
        let q = mlxcel_core::reshape(&q, &[b, s, self.n_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let q = self
            .rope
            .apply(&q, self.rope_dims, self.rope_traditional, offset);

        let k = self.k_norm.forward(&self.wk.forward(x));
        let k = mlxcel_core::reshape(&k, &[b, 1, s, self.head_dim]);
        let k = self
            .rope
            .apply(&k, self.rope_dims, self.rope_traditional, offset);

        // The keys ride in an fp16 KV cache whose value slot is one zero
        // column; upstream stores a zero-width value.
        let placeholder = mlxcel_core::zeros(&[b, 1, s, 1], mlxcel_core::array_dtype(&k));
        let (keys, _) = cache.update_and_fetch(k, placeholder);

        let kv_len = mlxcel_core::array_shape(&keys)[2];
        if kv_len <= self.selection.topk {
            return None;
        }
        let weights = self.weights_proj.forward(x);
        let scores = indexer_scores(&q, &keys, &weights, self.n_heads, self.head_dim);
        Some(lsa_top_indices(&scores, kv_len - s, self.selection))
    }
}

/// `sum_h relu(q_h . k) * w_h * n_heads^-0.5 * head_dim^-0.5`, `[B, 1, L, S]`.
///
/// `q` `[B, H, L, D]`, `k` `[B, 1, S, D]`, `weights` `[B, L, H]`.
fn indexer_scores(
    q: &MlxArray,
    k: &MlxArray,
    weights: &MlxArray,
    n_heads: i32,
    head_dim: i32,
) -> UniquePtr<MlxArray> {
    let k_t = mlxcel_core::transpose_axes(k, &[0, 1, 3, 2]);
    let scores = mlxcel_core::relu(&mlxcel_core::matmul(q, &k_t));
    let scale = (n_heads as f32).powf(-0.5) * (head_dim as f32).powf(-0.5);
    let scale = mlxcel_core::full_f32(&[1], scale, mlxcel_core::array_dtype(weights));
    let w = mlxcel_core::multiply(weights, &scale);
    let w = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&w, &[0, 2, 1]), -1);
    let w = mlxcel_core::astype(&w, mlxcel_core::array_dtype(&scores));
    mlxcel_core::sum_axis(&mlxcel_core::multiply(&scores, &w), 1, true)
}

/// The causal keep mask of `l` queries at positions `start..start + l` over
/// `s` keys, `[l, s]`.
fn causal_keep(start: i32, l: i32, s: i32) -> UniquePtr<MlxArray> {
    let rows = mlxcel_core::reshape(&mlxcel_core::arange_i32(start, start + l, 1), &[l, 1]);
    let cols = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, s, 1), &[1, s]);
    mlxcel_core::less_equal(&cols, &rows)
}

/// Each query's top-`topk` key positions from indexer `scores` `[B, 1, L, S]`,
/// for queries at positions `start..start + L`.
///
/// The first `init_tokens` columns and the `local_tokens` columns ending at
/// the query are forced to `+inf`, then every key after the query to `-inf`,
/// so the forced columns win and no future key outranks a past one. A query
/// with fewer than `topk` past keys still gets `topk` positions; the extra
/// ones are future keys, which [`sparse_prefill_mask`] drops again with the
/// causal mask, as upstream's `sparse_mask & mask` does.
pub(crate) fn lsa_top_indices(
    scores: &MlxArray,
    start: i32,
    selection: LsaSelection,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(scores);
    let (l, s) = (shape[2], shape[3]);
    let causal = causal_keep(start, l, s);
    let mut scores = mlxcel_core::copy(scores);
    if selection.init_tokens > 0 || selection.local_tokens > 0 {
        let cols = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, s, 1), &[1, s]);
        let sink = mlxcel_core::less(
            &cols,
            &mlxcel_core::from_slice_i32(&[selection.init_tokens], &[1]),
        );
        let forced = if selection.local_tokens > 0 {
            let rows = mlxcel_core::reshape(&mlxcel_core::arange_i32(start, start + l, 1), &[l, 1]);
            let window_start = mlxcel_core::subtract(
                &rows,
                &mlxcel_core::from_slice_i32(&[selection.local_tokens], &[1]),
            );
            let local =
                mlxcel_core::logical_and(&causal, &mlxcel_core::greater(&cols, &window_start));
            mlxcel_core::logical_or(&sink, &local)
        } else {
            sink
        };
        let inf = mlxcel_core::full_f32(&[1], f32::INFINITY, mlxcel_core::array_dtype(&scores));
        scores = mlxcel_core::where_cond(&forced, &inf, &scores);
    }
    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, mlxcel_core::array_dtype(&scores));
    let scores = mlxcel_core::where_cond(&causal, &scores, &neg_inf);
    let kth = s - selection.topk;
    let part = mlxcel_core::argpartition(&scores, kth, -1);
    slice_axis(&part, -1, kth, s)
}

/// The additive mask of a multi-token sparse step: `0` where a key is both
/// selected (`indices` `[B, 1, L, K]`) and causal for queries at positions
/// `start..start + L`, `-inf` elsewhere; `[B, 1, L, S]` in `dtype`.
pub(crate) fn sparse_prefill_mask(
    indices: &MlxArray,
    start: i32,
    kv_len: i32,
    dtype: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(indices);
    let (b, l, k) = (shape[0], shape[2], shape[3]);
    let base = mlxcel_core::zeros(&[b, 1, l, kv_len], dtype::FLOAT32);
    let ones = mlxcel_core::ones(&[b, 1, l, k], dtype::FLOAT32);
    let filled = mlxcel_core::put_along_axis(&base, indices, &ones, -1);
    let half = mlxcel_core::full_f32(&[1], 0.5, dtype::FLOAT32);
    let keep = mlxcel_core::logical_and(
        &mlxcel_core::greater(&filled, &half),
        &causal_keep(start, l, kv_len),
    );
    let zero = mlxcel_core::full_f32(&[1], 0.0, dtype);
    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, dtype);
    mlxcel_core::where_cond(&keep, &zero, &neg_inf)
}

/// One MLA sub-block: the shared LongCat projections and attention, this
/// family's RoPE, and on `self_attn.0` the indexer.
struct SparseMla {
    inner: LongcatFlashMLA,
    rope: SparseRope,
    indexer: Option<LsaIndexer>,
}

impl SparseMla {
    /// The sub-block output and the key selection it used (its own on an
    /// indexer owner, `shared` otherwise).
    fn forward(
        &self,
        x: &MlxArray,
        latent_cache: &mut KVCache,
        index_cache: &mut KVCache,
        shared: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>) {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let MlaProjections {
            qr,
            q_nope,
            q_pe,
            kv_latent,
            k_pe,
        } = self.inner.project(x);

        let offset = latent_cache.offset;
        let rope_dims = self.inner.qk_rope_head_dim;
        let q_pe = self.rope.apply(&q_pe, rope_dims, true, offset);
        let k_pe = self.rope.apply(&k_pe, rope_dims, true, offset);

        let selected = match (&self.indexer, &qr) {
            (Some(indexer), Some(qr)) => indexer.select(x, qr, index_cache),
            _ => shared.map(mlxcel_core::copy),
        };

        let window = crate::models::mla_batched::store_latent_step(
            latent_cache,
            self.inner.geometry(),
            kv_latent,
            k_pe,
            MODEL_TYPE,
            true,
        );
        let absorbed = self.inner.absorbed(l, latent_cache);
        let output = match &selected {
            None => {
                debug_assert!(l == 1 || window.seq_len() == latent_cache.live_len());
                self.inner
                    .attend_window(window, &q_nope, &q_pe, absorbed, None, b, l)
            }
            // One query attends only its selected rows, gathered from the
            // window in the layout the cache stores (fp16 or the packed
            // quantized latent); every gathered row is causally valid.
            Some(indices) if l == 1 => {
                let gathered: LatentWindow = window.gather_rows(indices);
                self.inner
                    .attend_window(gathered, &q_nope, &q_pe, true, None, b, l)
            }
            Some(indices) => {
                let kv_len = window.seq_len();
                let mask = sparse_prefill_mask(
                    indices,
                    kv_len - l,
                    kv_len,
                    mlxcel_core::array_dtype(&q_pe),
                );
                self.inner
                    .attend_window(window, &q_nope, &q_pe, absorbed, Some(&mask), b, l)
            }
        };
        (output, selected)
    }
}

/// Per-layer model-owned caches.
pub(super) struct SparseLayerCaches {
    /// The latent caches of `self_attn.0` and `self_attn.1`.
    pub(super) latent: [KVCache; 2],
    /// The indexer keys of `self_attn.0` (always fp16; unused under MLA).
    pub(super) index: KVCache,
}

struct SparseDecoderLayer {
    self_attn: [SparseMla; 2],
    mlps: [LongcatFlashMLP; 2],
    input_layernorm: [RMSNorm; 2],
    post_attention_layernorm: [RMSNorm; 2],
    moe_mlp: LongcatFlashMoE,
}

impl SparseDecoderLayer {
    fn forward(&self, x: &MlxArray, caches: &mut SparseLayerCaches) -> UniquePtr<MlxArray> {
        let SparseLayerCaches { latent, index } = caches;
        let [latent0, latent1] = latent;
        let mut hidden = mlxcel_core::copy(x);
        let mut shortcut = None;
        let mut selection: Option<UniquePtr<MlxArray>> = None;
        for (i, latent_cache) in [latent0, latent1].into_iter().enumerate() {
            let normed = self.input_layernorm[i].forward(&hidden);
            let (attn_out, used) =
                self.self_attn[i].forward(&normed, latent_cache, index, selection.as_deref());
            if i == 0 {
                selection = used;
            }
            hidden = mlxcel_core::add(&hidden, &attn_out);

            let normed = self.post_attention_layernorm[i].forward(&hidden);
            if i == 0 {
                shortcut = Some(self.moe_mlp.forward(&normed));
            }
            hidden = mlxcel_core::add(&hidden, &self.mlps[i].forward(&normed));
            if i == 1
                && let Some(ref shortcut) = shortcut
            {
                hidden = mlxcel_core::add(&hidden, shortcut);
            }
        }
        hidden
    }

    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashSparseConfig,
        base: &LongcatFlashNgramConfig,
        rope: &SparseRope,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        let attention = |i: usize| -> Result<SparseMla, String> {
            let attn_prefix = format!("{prefix}.self_attn.{i}");
            let indexer_prefix = format!("{attn_prefix}.indexer");
            let has_indexer = weights.contains_key(&format!("{indexer_prefix}.wk.weight"));
            let indexer = if args.uses_lsa() && i == 0 {
                if !has_indexer {
                    return Err(format!(
                        "longcat_flash_sparse: attention_method is LSA but \
                         `{indexer_prefix}.wk.weight` is missing"
                    ));
                }
                Some(LsaIndexer::from_weights(
                    weights,
                    args,
                    &indexer_prefix,
                    rope.duplicate(),
                )?)
            } else if has_indexer {
                // Upstream builds the indexer on `self_attn.0` only and
                // `self_attn.1` reuses its selection; weights elsewhere
                // would be silently ignored.
                return Err(format!(
                    "longcat_flash_sparse: unexpected indexer weights under `{indexer_prefix}`"
                ));
            } else {
                None
            };
            Ok(SparseMla {
                inner: LongcatFlashMLA::from_weights(weights, base, &attn_prefix)?,
                rope: rope.duplicate(),
                indexer,
            })
        };
        let mlp = |i: usize| {
            LongcatFlashMLP::from_weights(
                weights,
                &format!("{prefix}.mlps.{i}"),
                args.hidden_size,
                gs,
                bits,
            )
        };
        let norm = |name: String| -> Result<RMSNorm, String> {
            Ok(RMSNorm::new(
                get_weight_copy(weights, &name)?,
                args.rms_norm_eps,
            ))
        };
        Ok(Self {
            self_attn: [attention(0)?, attention(1)?],
            mlps: [mlp(0)?, mlp(1)?],
            input_layernorm: [
                norm(format!("{prefix}.input_layernorm.0.weight"))?,
                norm(format!("{prefix}.input_layernorm.1.weight"))?,
            ],
            post_attention_layernorm: [
                norm(format!("{prefix}.post_attention_layernorm.0.weight"))?,
                norm(format!("{prefix}.post_attention_layernorm.1.weight"))?,
            ],
            moe_mlp: LongcatFlashMoE::from_weights(weights, base, &format!("{prefix}.mlp"))?,
        })
    }
}

/// `word + proj / (1 + num_embedders)`: the word embedding at full scale and
/// only the n-gram projection normalized. `longcat_flash_ngram` divides the
/// whole sum, which garbles this checkpoint's generation (upstream).
pub(crate) fn fuse_ngram(
    word: &MlxArray,
    proj: &MlxArray,
    num_embedders: usize,
) -> UniquePtr<MlxArray> {
    let divisor = mlxcel_core::full_f32(
        &[1],
        (1 + num_embedders) as f32,
        mlxcel_core::array_dtype(proj),
    );
    mlxcel_core::add(word, &mlxcel_core::divide(proj, &divisor))
}

/// The n-gram input embedding with one fused projection over the
/// concatenated lookups (`model.ngram_embeddings.post_proj`).
struct SparseNgramEmbedding {
    word_embeddings: UnifiedEmbedding,
    embedders: Vec<UnifiedEmbedding>,
    post_proj: UnifiedLinear,
    vocab_mods: HashMap<(usize, usize), Vec<i64>>,
    m: usize,
    k: usize,
    n: usize,
}

impl SparseNgramEmbedding {
    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashSparseConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        let base = args.base();
        let (m, num_embedders) = NgramEmbedding::validate_config(&base)?;
        let emb_dim = args.hidden_size / num_embedders;
        let word_embeddings = UnifiedEmbedding::from_weights(
            weights,
            &format!("{prefix}.word_embeddings"),
            gs,
            bits,
        )?;
        validate_embedding_table(
            &word_embeddings,
            &format!("{prefix}.word_embeddings"),
            args.vocab_size,
            "vocab_size",
            args.hidden_size,
            "hidden_size",
        )?;
        let mut embedders = Vec::with_capacity(num_embedders);
        for i in 0..num_embedders {
            let key = format!("{prefix}.embedders.{i}");
            let embedding = UnifiedEmbedding::from_weights(weights, &key, gs, bits)?;
            validate_embedding_table(
                &embedding,
                &key,
                m + i * 2 + 1,
                "oe_vocab_size_ratio * vocab_size + 2 * i + 1",
                emb_dim,
                "hidden_size / (oe_split_num * (oe_neighbor_num - 1))",
            )?;
            embedders.push(embedding);
        }
        let post_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.post_proj"), gs, bits)?;
        Ok(Self {
            word_embeddings,
            embedders,
            post_proj,
            vocab_mods: NgramEmbedding::compute_vocab_mods(
                args.oe_neighbor_num,
                args.oe_split_num,
                m,
                args.vocab_size,
            ),
            m,
            k: args.oe_split_num,
            n: args.oe_neighbor_num,
        })
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        context: &mut Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape[shape.len() - 1];
        let cur_context = NgramEmbedding::advance_context(input_ids, context, self.n);
        let word = self.word_embeddings.forward(input_ids);

        let shifted: HashMap<usize, UniquePtr<MlxArray>> = (2..=self.n)
            .map(|i| (i, NgramEmbedding::shift_right(&cur_context, i - 1)))
            .collect();
        let mut lookups = Vec::with_capacity(self.embedders.len());
        for i in 2..=self.n {
            for j in 0..self.k {
                let index = (i - 2) * self.k + j;
                let ids = NgramEmbedding::hashed_ids(
                    &cur_context,
                    &shifted,
                    &self.vocab_mods[&(i, j)],
                    i,
                    (self.m + index * 2 + 1) as i64,
                    seq_len,
                );
                lookups.push(self.embedders[index].forward(&ids));
            }
        }
        let refs: Vec<&MlxArray> = lookups.iter().map(|a| &**a).collect();
        let proj = self
            .post_proj
            .forward(&mlxcel_core::concatenate_many(&refs, -1));
        fuse_ngram(&word, &proj, self.embedders.len())
    }
}

enum SparseEmbedding {
    Ngram(SparseNgramEmbedding),
    Plain(UnifiedEmbedding),
}

/// Per-sequence state the model owns: three caches per layer and the n-gram
/// token tail. The family does not batch, so the engine hands `forward` an
/// empty cache slice and `make_caches` resets this.
pub(super) struct SparseState {
    pub(super) layers: Vec<SparseLayerCaches>,
    ngram_context: Option<UniquePtr<MlxArray>>,
}

impl SparseState {
    /// Layer `i`'s latent caches are built in `mode_for_layer(i)`; the
    /// indexer-key cache is always fp16.
    fn new(num_layers: usize, mode_for_layer: impl Fn(usize) -> KVCacheMode) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|i| {
                    let mode = mode_for_layer(i);
                    SparseLayerCaches {
                        latent: [KVCache::new_with_mode(mode), KVCache::new_with_mode(mode)],
                        index: KVCache::new(),
                    }
                })
                .collect(),
            ngram_context: None,
        }
    }
}

pub struct LongcatFlashSparseModel {
    embedding: SparseEmbedding,
    layers: Vec<SparseDecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    num_layers: usize,
    pub(super) state: RefCell<SparseState>,
    /// The per-layer KV modes the latent caches are built in: the injected
    /// table with every mode the latent pair cannot hold replaced by fp16.
    kv_cache_layer_modes: KvCacheLayerModes,
    kv_mode_warned: Cell<bool>,
    /// Whether an injected `int8` layer keeps the affine quantized latent,
    /// read from `mla::stores_quantized_latent_pair` at load (the family is
    /// in `mla::QUANTIZED_LATENT_FAMILIES`; `--mla-latent-quant` overrides).
    pub(crate) quantized_latent: Cell<bool>,
}

impl LongcatFlashSparseModel {
    pub fn load(model_path: &str) -> Result<(Self, usize), String> {
        let path = Path::new(model_path);
        let config_str = std::fs::read_to_string(path.join("config.json"))
            .map_err(|e| format!("Failed to read config: {e}"))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: LongcatFlashSparseConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config: {e}"))?;
        let weights = crate::models::load_text_weights(path, None)?;
        let weights = sanitize_weights(weights, &args)?;
        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args.vocab_size))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashSparseConfig,
    ) -> Result<Self, String> {
        args.validate()?;
        let gs = args.group_size();
        let bits = args.bits();
        let base = args.base();
        let embedding = if args.uses_ngram() {
            SparseEmbedding::Ngram(SparseNgramEmbedding::from_weights(
                weights,
                args,
                "model.ngram_embeddings",
            )?)
        } else {
            let embed = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;
            validate_embedding_table(
                &embed,
                "model.embed_tokens",
                args.vocab_size,
                "vocab_size",
                args.hidden_size,
                "hidden_size",
            )?;
            SparseEmbedding::Plain(embed)
        };
        let rope = SparseRope::from_config(args);
        let layers = (0..args.num_layers)
            .map(|i| {
                SparseDecoderLayer::from_weights(
                    weights,
                    args,
                    &base,
                    &rope,
                    &format!("model.layers.{i}"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let norm = RMSNorm::new(
            get_weight_copy(weights, "model.norm.weight")?,
            args.rms_norm_eps,
        );
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?;
        Ok(Self {
            embedding,
            layers,
            norm,
            lm_head,
            num_layers: args.num_layers,
            state: RefCell::new(SparseState::new(args.num_layers, |_| KVCacheMode::Fp16)),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
            kv_mode_warned: Cell::new(false),
            quantized_latent: Cell::new(mlxcel_core::mla::stores_quantized_latent_pair(MODEL_TYPE)),
        })
    }

    /// `row == None` projects every position, `Some(row)` only that one.
    fn internal_forward(
        &self,
        input_ids: &MlxArray,
        state: &mut SparseState,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let SparseState {
            layers,
            ngram_context,
        } = state;
        let mut hidden = match &self.embedding {
            SparseEmbedding::Ngram(ngram) => ngram.forward(input_ids, ngram_context),
            SparseEmbedding::Plain(embed) => embed.forward(input_ids),
        };
        for (layer, caches) in self.layers.iter().zip(layers.iter_mut()) {
            hidden = layer.forward(&hidden, caches);
        }
        let hidden = mlxcel_core::generate::select_row(hidden, row);
        self.lm_head.forward(&self.norm.forward(&hidden))
    }
}

impl LanguageModel for LongcatFlashSparseModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // The caller's slice is ignored (empty on the engine path): the
        // caches and the n-gram tail are model-owned.
        let mut state = self.state.borrow_mut();
        self.internal_forward(input_ids, &mut state, None)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<mlxcel_core::cache::SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        if input_embeddings.is_some() {
            return None;
        }
        let mut state = self.state.borrow_mut();
        Some(self.internal_forward(input_ids, &mut state, Some(last_pos)))
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn supports_batching(&self) -> bool {
        // One model-owned state slot.
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        // Pad positions would land in the model-owned caches and n-gram tail.
        false
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset the model-owned state; the returned vector is never read.
        self.reset_runtime_state();
        (0..self.num_layers).map(|_| KVCache::new()).collect()
    }

    fn reset_runtime_state(&self) {
        *self.state.borrow_mut() = SparseState::new(self.num_layers, |i| {
            self.kv_cache_layer_modes.mode_for_layer(i)
        });
    }

    /// Keep the per-layer modes the latent pair can hold: fp16, and the
    /// affine quantized latent (`int8`, at the width `--mla-latent-quant`
    /// selects) when the model opted in. Any other mode (every Turbo mode)
    /// is replaced by fp16 with one warning. The indexer keys stay fp16.
    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        let opted_in = self.quantized_latent.get();
        let kept: Vec<KVCacheMode> = modes
            .into_iter()
            .map(|mode| {
                let refusal = if opted_in && mode.latent_affine_bits().is_some() {
                    None
                } else {
                    mlxcel_core::mla::latent_pair_supports_mode(MODEL_TYPE, mode).err()
                };
                match refusal {
                    None => mode,
                    Some(reason) => {
                        if !self.kv_mode_warned.replace(true) {
                            tracing::warn!(
                                %mode,
                                "{MODEL_TYPE}: {reason}; layers requesting this KV cache mode \
                                 use fp16"
                            );
                        }
                        KVCacheMode::Fp16
                    }
                }
            })
            .collect();
        self.kv_cache_layer_modes.set(kept);
        self.reset_runtime_state();
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        Some(
            (0..self.num_layers)
                .map(|i| self.kv_cache_layer_modes.mode_for_layer(i))
                .collect(),
        )
    }
}

/// Bring a checkpoint to the layout [`LongcatFlashSparseModel::from_weights`]
/// reads.
///
/// The mlx-community conversions root every tensor at `language_model.`; the
/// original HF layout has per-expert MLPs, a fused `kv_b_proj`, and the
/// n-gram tables as `model.embed_tokens`, `model.oe_embed_tokens{i}` and
/// per-embedder `model.oe_embed_proj{i}`, whose projections concatenate along
/// the input axis into the one `post_proj` (upstream `sanitize`). The shared
/// LongCat sanitizer stacks the experts, decomposes `kv_b_proj` and drops
/// the MTP tensors.
pub fn sanitize_weights(
    weights: WeightMap,
    args: &LongcatFlashSparseConfig,
) -> Result<WeightMap, String> {
    let mut stripped = WeightMap::new();
    for (key, value) in weights {
        let key = match key.strip_prefix("language_model.") {
            Some(rest) => rest.to_string(),
            None => key,
        };
        stripped.insert(key, value);
    }
    let mut weights = super::longcat_flash_ngram::sanitize_weights(stripped, &args.base())?;
    if !args.uses_ngram() {
        return Ok(weights);
    }

    for part in ["weight", "scales", "biases"] {
        if let Some(w) = weights.remove(&format!("model.embed_tokens.{part}")) {
            weights.insert(format!("model.ngram_embeddings.word_embeddings.{part}"), w);
        }
    }
    let table_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.starts_with("model.oe_embed_tokens"))
        .cloned()
        .collect();
    for key in table_keys {
        let rest = &key["model.oe_embed_tokens".len()..];
        let (index, part) = split_indexed(rest, &key)?;
        let value = weights.remove(&key).expect("key listed above");
        weights.insert(
            format!("model.ngram_embeddings.embedders.{index}.{part}"),
            value,
        );
    }
    let mut proj_parts: HashMap<String, Vec<(usize, UniquePtr<MlxArray>)>> = HashMap::new();
    let proj_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.starts_with("model.oe_embed_proj"))
        .cloned()
        .collect();
    for key in proj_keys {
        let rest = &key["model.oe_embed_proj".len()..];
        let (index, part) = split_indexed(rest, &key)?;
        let value = weights.remove(&key).expect("key listed above");
        proj_parts
            .entry(part.to_string())
            .or_default()
            .push((index, value));
    }
    for (part, mut parts) in proj_parts {
        parts.sort_by_key(|(index, _)| *index);
        if parts
            .iter()
            .enumerate()
            .any(|(want, (index, _))| *index != want)
        {
            return Err(format!(
                "longcat_flash_sparse: model.oe_embed_proj*.{part} indices are not 0..{}",
                parts.len()
            ));
        }
        let refs: Vec<&MlxArray> = parts.iter().map(|(_, a)| &**a).collect();
        // Every per-embedder projection maps its lookup to `hidden`, so the
        // fused one reads the lookups concatenated along its input axis. An
        // affine plane packs along that axis by whole groups, so its scales
        // and biases concatenate the same way.
        let fused = mlxcel_core::concatenate_many(&refs, 1);
        weights.insert(format!("model.ngram_embeddings.post_proj.{part}"), fused);
    }
    Ok(weights)
}

/// `"{index}.{part}"` of an `oe_embed_*` key suffix.
fn split_indexed<'a>(rest: &'a str, key: &str) -> Result<(usize, &'a str), String> {
    let (index, part) = rest
        .trim_start_matches(['.', '_'])
        .split_once('.')
        .ok_or_else(|| format!("longcat_flash_sparse: unexpected n-gram tensor `{key}`"))?;
    let index = index
        .parse::<usize>()
        .map_err(|_| format!("longcat_flash_sparse: unexpected n-gram tensor `{key}`"))?;
    Ok((index, part))
}

#[cfg(test)]
#[path = "longcat_flash_sparse_tests.rs"]
mod tests;
