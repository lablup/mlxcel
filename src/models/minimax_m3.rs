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

//! MiniMax-M3 text model (`model_type: "minimax_m3"`).
//!
//! Hybrid dense/MoE decoder with block-sparse "MSA" attention. Unlike the
//! MiniMax-M2 family (`minimax.rs`, 256 experts, standard RMSNorm, dense
//! attention), M3 uses:
//! - a per-layer plan from `moe_layer_freq` (first layers dense, rest MoE),
//! - Gemma-style RMSNorm (weight+1) everywhere, including per-head Q/K norm,
//! - clamp-SwiGLU (`swigluoai`) experts and dense MLPs,
//! - a sigmoid router with a selection-only bias plus one shared expert
//!   (packed as switch-tensor index `num_local_experts` when its width equals
//!   the routed width), and
//! - a block-sparse indexer (`minimax_m3_indexer.rs`) on the sparse layers.
//!
//! The config parses a FLAT text config; a future VL wrapper (#764) constructs
//! the same [`ModelArgs`] from the checkpoint's nested `text_config` block. The
//! real 427B VL checkpoint cannot be loaded on the development machine, so the
//! validated surface is the synthetic reduced-config unit tests plus the
//! real-config parse test in `minimax_m3_tests.rs`.

#[path = "minimax_m3_config.rs"]
mod config;
#[path = "minimax_m3_indexer.rs"]
mod indexer;
#[path = "minimax_m3_layers.rs"]
mod layers;
#[path = "minimax_m3_moe.rs"]
mod moe;

#[cfg(test)]
#[path = "minimax_m3_tests.rs"]
mod minimax_m3_tests;

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

use crate::models::gemma::GemmaRMSNorm;
use layers::{Attention, DenseMlp};
use moe::MoeBlock;

pub use config::{ModelArgs, Quantization, SparseAttentionConfig};

// ============================================================================
// Decoder layer
// ============================================================================

enum Mlp {
    Dense(DenseMlp),
    Moe(MoeBlock),
}

impl Mlp {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Mlp::Dense(m) => m.forward(x),
            Mlp::Moe(m) => m.forward(x),
        }
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: GemmaRMSNorm,
    post_attention_layernorm: GemmaRMSNorm,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let sparse = if args.is_sparse_layer(layer_idx) {
            args.sparse_attention_config.as_ref()
        } else {
            None
        };
        let self_attn =
            Attention::from_weights(weights, args, sparse, &format!("{}.self_attn", prefix))?;

        // MoE layers store their FFN under `block_sparse_moe`; the leading dense
        // layers store a plain `mlp` (verbatim checkpoint layout).
        let mlp = if args.is_moe_layer(layer_idx) {
            Mlp::Moe(MoeBlock::from_weights(
                weights,
                args,
                &format!("{}.block_sparse_moe", prefix),
            )?)
        } else {
            Mlp::Dense(DenseMlp::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        };

        let input_layernorm = GemmaRMSNorm::new(
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?,
            args.rms_norm_eps,
        );
        let post_attention_layernorm = GemmaRMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.post_attention_layernorm.weight", prefix),
            )?,
            args.rms_norm_eps,
        );

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// ============================================================================
// Model
// ============================================================================

pub struct MiniMaxM3Model {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: GemmaRMSNorm,
    lm_head: Option<UnifiedLinear>,
}

impl MiniMaxM3Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_impl(input_ids, None, caches, mask)
    }

    /// Token embedding lookup, exposed for the VL wrapper's LLaVA-style merge
    /// (`src/vision/minimax_m3_vl.rs`).
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Decoder forward that optionally starts from precomputed input
    /// embeddings. When `input_embeddings` is `Some`, the embedding lookup is
    /// skipped and the provided (already vision-merged) embeddings are decoded
    /// directly; `input_ids` is then only used for shape/consistency by callers.
    pub fn forward_with_embeddings_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match input_embeddings {
            Some(embeds) => mlxcel_core::copy(embeds),
            None => self.embed_tokens.forward(input_ids),
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.norm.forward(&h);
        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = sanitize_weights(weights, &args);
        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(weights, args, i)?);
        }

        let norm = GemmaRMSNorm::new(
            get_weight_copy(weights, "model.norm.weight")?,
            args.rms_norm_eps,
        );

        let lm_head = if !args.tie_word_embeddings {
            let head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits).or_else(
                |_| UnifiedLinear::from_weights(weights, "model.lm_head", group_size, bits),
            )?;
            Some(head)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
}

// ============================================================================
// Weight sanitization
// ============================================================================

/// Normalize checkpoint key prefixes and drop non-text tensors so loading a flat
/// text export or a nested VL text tower both land on the `model.`-prefixed
/// layout the loader expects, and never fail on tensors this text decoder does
/// not implement.
///
/// - Vision skip: `vision_tower.*`, `multi_modal_projector.*`, and
///   `patch_merge_mlp.*` are dropped so a text-only load of the VL checkpoint
///   ignores the vision front-end.
/// - Prefix rewrites: `language_model.model.`, `model.language_model.`, and a
///   leading `language_model.` all collapse to `model.` (no-op for a flat text
///   export). `language_model.lm_head.` becomes `model.lm_head.`, which the
///   loader resolves via its `model.lm_head` fallback.
/// - MTP strip: any tensor for a layer index `>= num_hidden_layers`, or whose
///   path names an MTP / next-N module, is removed.
pub fn sanitize_weights(weights: WeightMap, args: &ModelArgs) -> WeightMap {
    let mut out = WeightMap::new();
    for (key, value) in weights.into_iter() {
        if is_non_text_key(&key) {
            continue;
        }
        let key = rewrite_language_model_prefix(&key);
        if is_mtp_key(&key, args.num_hidden_layers) {
            continue;
        }
        out.insert(key, value);
    }
    out
}

/// Vision / multimodal tensors to drop for a text-only load of the VL
/// checkpoint. Matches both the top-level layout and any `model.`-nested export.
fn is_non_text_key(key: &str) -> bool {
    const PREFIXES: [&str; 3] = [
        "vision_tower.",
        "multi_modal_projector.",
        "patch_merge_mlp.",
    ];
    PREFIXES
        .iter()
        .any(|p| key.starts_with(p) || key.starts_with(&format!("model.{}", p)))
}

fn rewrite_language_model_prefix(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        format!("model.{}", rest)
    } else if let Some(rest) = key.strip_prefix("language_model.model.") {
        format!("model.{}", rest)
    } else if let Some(rest) = key.strip_prefix("language_model.") {
        format!("model.{}", rest)
    } else {
        key.to_string()
    }
}

fn is_mtp_key(key: &str, num_hidden_layers: usize) -> bool {
    if key.contains("mtp") || key.contains("nextn") || key.contains("next_n") {
        return true;
    }
    // model.layers.{idx}... beyond the decoder depth are MTP prediction layers.
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() >= 3
        && parts[0] == "model"
        && parts[1] == "layers"
        && let Ok(idx) = parts[2].parse::<usize>()
    {
        return idx >= num_hidden_layers;
    }
    false
}

pub(crate) fn get_weight_copy(
    weights: &WeightMap,
    name: &str,
) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// ============================================================================
// LanguageModel
// ============================================================================

impl LanguageModel for MiniMaxM3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        MiniMaxM3Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        MiniMaxM3Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Fallback only; the runtime prefers the tokenizer/generation config.
        // MiniMax uses the `<eos>`/`[e~[` sentinel at 200020 in its 200k vocab.
        vec![200020]
    }
}
