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

//! Laguna (Poolside Laguna XS / S) model implementation using mlxcel-core.
//!
//! Laguna is a pre-norm decoder with QK-RMSNorm, a hybrid of
//! `full_attention` (YaRN, partial rotary) and `sliding_attention` (plain
//! RoPE, window 512) layers, per-layer query-head counts, a per-head softplus
//! attention output gate, and a sigmoid-routed top-k MoE with a correction
//! bias, a shared expert and a `moe_routed_scaling_factor`. The published
//! NVFP4 checkpoints keep the experts in the `compressed-tensors`
//! `nvfp4-pack-quantized` layout; [`crate::models::laguna_sanitize`] turns
//! that into MLX native NVFP4 planes bit for bit.
//!
//! References: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/laguna.py
//! for the graph, and the `modeling_laguna.py` / `configuration_laguna.py`
//! that ship inside each published checkpoint for the config semantics. The
//! two disagree in one place: the YaRN block's `attention_factor` is honored
//! here when the config declares one, which is what
//! `transformers.modeling_rope_utils._compute_yarn_parameters` does and
//! therefore what the checkpoint's own modeling file inherits, while mlx-lm
//! drops the key and always derives the factor from `factor`. It matters for
//! `mlx-community/Laguna-XS.2-4bit`, which declares `attention_factor: 1.0`
//! where the derived value would be 1.3466.
//!
//! Layer blocks live in [`crate::models::laguna_layers`]; this file holds
//! the config, the per-layer RoPE resolution, the model shell and the
//! `LanguageModel` wrapper.

use crate::models::laguna_layers::{DecoderLayer, LagunaCache};
use crate::models::laguna_sanitize::sanitize_weights;
use crate::models::mellum::{YarnRope, compute_yarn_rope};
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::cell::RefCell;
use std::path::Path;

pub const FULL_ATTENTION: &str = "full_attention";
pub const SLIDING_ATTENTION: &str = "sliding_attention";

/// Upper bounds on the architecture scalars a Laguna `config.json` may declare.
/// Same rationale as the other ports: `config.json` is untrusted input on the
/// `mlxcel generate -m <org>/<repo>` path, and these keep a hostile value from
/// sizing a loop, an allocation or a partition pivot. Both sit well above
/// Laguna S 2.1, the largest published member (48 layers, 256 experts).
const MAX_NUM_LAYERS: usize = 1024;
/// See [`MAX_NUM_LAYERS`].
const MAX_NUM_EXPERTS: usize = 4096;

/// Top-level `config.json` fields the loader reads.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Query heads per layer; defaults to `num_attention_heads` everywhere.
    #[serde(default)]
    pub num_attention_heads_per_layer: Option<Vec<usize>>,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// `"full_attention"` / `"sliding_attention"` per layer; defaults to all
    /// full attention.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    /// Per-layer-type RoPE dicts plus loose top-level scalars that apply to
    /// every layer type (`original_max_position_embeddings` on XS.2).
    #[serde(default)]
    pub rope_parameters: serde_json::Value,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Fallback when a `rope_parameters` entry lacks its own factor.
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    /// `true` / `"per-head"` for a per-head gate, any other truthy string for
    /// a per-element gate, `false` / absent for no gate.
    #[serde(default)]
    pub gating: serde_json::Value,
    #[serde(default)]
    pub swa_attention_sink_enabled: bool,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_one")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    /// `"dense"` / `"sparse"` per layer; overrides the step / only-layers
    /// schedule when present.
    #[serde(default)]
    pub mlp_layer_types: Option<Vec<String>>,
    #[serde(default = "default_routed_scaling_factor")]
    pub moe_routed_scaling_factor: f32,
    #[serde(default)]
    pub moe_router_logit_softcapping: f32,
    /// `"sigmoid"` (default) or `"sqrtsoftplus"`.
    #[serde(default)]
    pub moe_router_score_func: Option<String>,
    /// Legacy spelling: `false` selects softmax scores.
    #[serde(default)]
    pub moe_router_use_sigmoid: Option<bool>,
    #[serde(default)]
    pub moe_apply_router_weight_on_input: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    /// MLX-style affine quantization block.
    #[serde(default)]
    pub quantization: Option<Quantization>,
    /// HF-style block (`compressed-tensors` on the published NVFP4 exports).
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "laguna".to_string()
}
fn default_max_position_embeddings() -> usize {
    262_144
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_sliding_window() -> usize {
    512
}
fn default_true() -> bool {
    true
}
fn default_one() -> usize {
    1
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}

/// Quantization parameters handed to every projection loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantSpec {
    pub group_size: i32,
    pub bits: i32,
    pub mode: &'static str,
}

impl QuantSpec {
    /// MLX native NVFP4 (the compressed-tensors transcode target).
    pub const NVFP4: Self = Self {
        group_size: 16,
        bits: 4,
        mode: "nvfp4",
    };
}

/// Router score function (`moe_router_score_func` / `moe_router_use_sigmoid`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterScoreFunc {
    Sigmoid,
    SqrtSoftplus,
    Softmax,
}

/// Resolved RoPE for one layer type.
pub(crate) struct LayerRope {
    /// Plain-RoPE base (only consulted when `yarn` is `None`).
    pub(crate) base: f32,
    /// Leading head dims that are rotated.
    pub(crate) rotated_dims: usize,
    /// YaRN frequencies and attention factor for full-attention layers.
    pub(crate) yarn: Option<YarnRope>,
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

    /// Query-head count of layer `idx`.
    pub fn num_heads_for_layer(&self, idx: usize) -> Result<usize, String> {
        match &self.num_attention_heads_per_layer {
            Some(per_layer) => per_layer.get(idx).copied().ok_or_else(|| {
                format!(
                    "num_attention_heads_per_layer has {} entries but layer {idx} exists",
                    per_layer.len()
                )
            }),
            None => Ok(self.num_attention_heads),
        }
    }

    pub fn layer_type(&self, idx: usize) -> &str {
        self.layer_types
            .as_ref()
            .and_then(|types| types.get(idx))
            .map(String::as_str)
            .unwrap_or(FULL_ATTENTION)
    }

    pub fn is_sliding(&self, idx: usize) -> bool {
        self.layer_type(idx) == SLIDING_ATTENTION
    }

    /// Layer `idx` is MoE iff `mlp_layer_types[idx] == "sparse"`, or (without
    /// that list) `(idx + 1) % decoder_sparse_step == 0` and `idx` is not in
    /// `mlp_only_layers`.
    pub fn is_moe_layer(&self, idx: usize) -> bool {
        if let Some(types) = &self.mlp_layer_types
            && let Some(kind) = types.get(idx)
        {
            return kind == "sparse";
        }
        if self.num_experts == 0 {
            return false;
        }
        let step = self.decoder_sparse_step.max(1);
        (idx + 1).is_multiple_of(step) && !self.mlp_only_layers.contains(&idx)
    }

    pub fn router_score_func(&self) -> Result<RouterScoreFunc, String> {
        match self.moe_router_score_func.as_deref() {
            Some("sigmoid") => Ok(RouterScoreFunc::Sigmoid),
            Some("sqrtsoftplus") => Ok(RouterScoreFunc::SqrtSoftplus),
            Some("softmax") => Ok(RouterScoreFunc::Softmax),
            Some(other) => Err(format!(
                "unsupported moe_router_score_func {other:?}; expected \"sigmoid\" or \
                 \"sqrtsoftplus\""
            )),
            None => Ok(if self.moe_router_use_sigmoid == Some(false) {
                RouterScoreFunc::Softmax
            } else {
                RouterScoreFunc::Sigmoid
            }),
        }
    }

    /// Whether `quantization_config` declares the compressed-tensors
    /// `nvfp4-pack-quantized` layout.
    pub fn declares_compressed_tensors_nvfp4(&self) -> bool {
        let Some(cfg) = &self.quantization_config else {
            return false;
        };
        let is_nvfp4 = |v: &serde_json::Value| {
            v.get("format")
                .and_then(|f| f.as_str())
                .is_some_and(|f| f == "nvfp4-pack-quantized")
        };
        is_nvfp4(cfg)
            || cfg
                .get("config_groups")
                .and_then(|g| g.as_object())
                .is_some_and(|groups| groups.values().any(is_nvfp4))
    }

    /// Quantization parameters for the projections, given whether the weight
    /// map carries transcoded NVFP4 planes.
    pub fn quant_spec(&self, nvfp4_planes_present: bool) -> QuantSpec {
        if let Some(q) = &self.quantization {
            return QuantSpec {
                group_size: q.group_size,
                bits: q.bits,
                mode: "affine",
            };
        }
        if nvfp4_planes_present || self.declares_compressed_tensors_nvfp4() {
            return QuantSpec::NVFP4;
        }
        QuantSpec {
            group_size: 64,
            bits: 4,
            mode: "affine",
        }
    }

    /// RoPE for a layer type: the per-type `rope_parameters` entry overlaid
    /// on the loose top-level scalars, `rope_theta` falling back to the
    /// config's top-level value (default 500000), `partial_rotary_factor`
    /// falling back to the top-level config key, then 1.0.
    pub(crate) fn layer_rope(&self, layer_type: &str) -> LayerRope {
        let head_dim = self.head_dim();
        let mut merged = serde_json::Map::new();
        if let Some(obj) = self.rope_parameters.as_object() {
            for (k, v) in obj {
                if !v.is_object() {
                    merged.insert(k.clone(), v.clone());
                }
            }
            if let Some(entry) = obj.get(layer_type).and_then(|v| v.as_object()) {
                for (k, v) in entry {
                    merged.insert(k.clone(), v.clone());
                }
            }
        }
        if !merged.contains_key("rope_theta") {
            merged.insert(
                "rope_theta".to_string(),
                serde_json::json!(self.rope_theta.unwrap_or(500_000.0)),
            );
        }
        let partial = merged
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .or(self.partial_rotary_factor)
            .unwrap_or(1.0);
        // The lower bound tracks `head_dim` because `Ord::clamp` asserts
        // `min <= max` and panics otherwise, and `head_dim` is a config value.
        // [`ModelArgs::validate`] rejects a `head_dim` below 2 before any load
        // reaches here; this keeps the helper itself panic-free for callers
        // that build a `ModelArgs` directly.
        let rotated_dims =
            ((head_dim as f64 * partial).round() as usize).clamp(2.min(head_dim), head_dim) & !1;
        let base = merged
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(500_000.0) as f32;
        let params = serde_json::Value::Object(merged);
        let yarn = compute_yarn_rope(rotated_dims, &params).map(|mut yarn| {
            // HF applies the explicit `attention_factor` when the config
            // carries one, and derives it from `factor` otherwise.
            if let Some(af) = params.get("attention_factor").and_then(|v| v.as_f64()) {
                yarn.mscale = af as f32;
            }
            yarn
        });
        LayerRope {
            base,
            rotated_dims,
            yarn,
        }
    }

    /// Reject config values that reach MLX as an out-of-range partition pivot,
    /// an out-of-bounds gather index, or a NaN factory.
    ///
    /// The routing half mirrors [`crate::models::afmoe`] and
    /// [`crate::models::bailing_moe`]: the router selects `num_experts_per_tok`
    /// indices out of a row of `num_experts` scores through
    /// `argpartition(kth = num_experts - num_experts_per_tok)`, and MLX signals
    /// an out-of-range `kth` by throwing. An MLX C++ exception crossing the cxx
    /// bridge is an uncatchable abort at the first routed forward pass, not a
    /// load error, so it lands after the whole checkpoint is already resident.
    /// `num_experts_per_tok` reaches that state from the serde default alone: a
    /// config that declares sparse layers through `mlp_layer_types` but omits
    /// `num_experts_per_tok` yields 0, and `kth` is then the full expert count.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_hidden_layers == 0 || self.num_hidden_layers > MAX_NUM_LAYERS {
            return Err(format!(
                "Laguna num_hidden_layers ({}) must be between 1 and {MAX_NUM_LAYERS}",
                self.num_hidden_layers
            ));
        }
        let head_dim = self.head_dim();
        if head_dim < 2 {
            return Err(format!(
                "Laguna head_dim ({head_dim}) must be at least 2; RoPE rotates feature pairs, and \
                 the partial-rotary width `layer_rope` derives from it has no valid value below 2"
            ));
        }
        if !(0..self.num_hidden_layers).any(|i| self.is_moe_layer(i)) {
            return Ok(());
        }
        if self.num_experts == 0 || self.num_experts > MAX_NUM_EXPERTS {
            return Err(format!(
                "Laguna num_experts ({}) must be between 1 and {MAX_NUM_EXPERTS}",
                self.num_experts
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            return Err(format!(
                "Laguna num_experts_per_tok ({}) must be between 1 and num_experts ({}); the \
                 router selects that many indices out of a row of num_experts scores with \
                 argpartition, and an out-of-range pivot is an MLX throw, which crosses the cxx \
                 bridge as an uncatchable abort on a token rather than a load error",
                self.num_experts_per_tok, self.num_experts
            ));
        }
        if !self.moe_routed_scaling_factor.is_finite() {
            return Err(format!(
                "Laguna moe_routed_scaling_factor ({}) must be finite; it multiplies every routed \
                 expert weight, so a non-finite value makes every MoE output NaN and that NaN \
                 reaches the logits without anything throwing",
                self.moe_routed_scaling_factor
            ));
        }
        if !self.moe_router_logit_softcapping.is_finite() {
            return Err(format!(
                "Laguna moe_router_logit_softcapping ({}) must be finite; the softcap divides the \
                 router logits and multiplies the tanh back, so a non-finite cap turns every \
                 router score into NaN",
                self.moe_router_logit_softcapping
            ));
        }
        self.router_score_func()?;
        Ok(())
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                n.as_i64().map(|id| vec![id as i32]).unwrap_or_default()
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Laguna model.
pub struct LagunaModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub sliding_window: usize,
    pub eos_token_ids: Vec<i32>,
}

impl LagunaModel {
    /// Run the stack. `capture_layer_ids` names layers whose post-block
    /// residual stream is returned alongside the logits (in that order); an
    /// empty slice captures nothing.
    ///
    /// Every caller today passes an empty slice: the capture arm and
    /// [`crate::models::laguna_layers::LagunaCache::trim`] are staged for the
    /// DFlash drafter (#1351), which the port keeps out of scope, and neither
    /// is reachable until `verify_forward_with_capture_layers` is implemented
    /// for this family.
    pub fn forward_with_capture(
        &self,
        input_ids: &MlxArray,
        caches: &mut [LagunaCache],
        capture_layer_ids: &[usize],
    ) -> (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>) {
        debug_assert_eq!(caches.len(), self.layers.len(), "one cache per layer");
        // Each attention layer builds its own prefill mask from the keys its
        // cache returns (see `Attention::forward`); decode passes none.
        let mut h = self.embed_tokens.forward(input_ids);
        let mut captured: Vec<Option<UniquePtr<MlxArray>>> =
            (0..capture_layer_ids.len()).map(|_| None).collect();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i]);
            for (slot, &want) in capture_layer_ids.iter().enumerate() {
                if want == i {
                    captured[slot] = Some(mlxcel_core::copy(&h));
                }
            }
        }

        let hidden = captured
            .into_iter()
            .map(|slot| slot.unwrap_or_else(|| mlxcel_core::zeros_like(&h)))
            .collect();
        let h = self.norm.forward(&h);
        let logits = match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        };
        (logits, hidden)
    }

    pub fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [LagunaCache],
    ) -> UniquePtr<MlxArray> {
        self.forward_with_capture(input_ids, caches, &[]).0
    }

    /// `KVCache` on full-attention layers, `RotatingKVCache(sliding_window)`
    /// on sliding layers.
    pub fn make_caches(&self) -> Vec<LagunaCache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.self_attn.is_sliding {
                    LagunaCache::Rotating(RotatingKVCache::new(self.sliding_window as i32))
                } else {
                    LagunaCache::Standard(KVCache::new())
                }
            })
            .collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;

        // Ahead of the sanitizer, which walks `num_hidden_layers` and stacks
        // `num_experts` planes per MoE layer straight from this config.
        args.validate()?;
        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        sanitize_weights(&mut weights, &args)?;
        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args))
    }

    /// Build from an already sanitized weight map.
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        if args.moe_apply_router_weight_on_input {
            return Err(
                "Laguna: moe_apply_router_weight_on_input = true is not supported (no published \
                 checkpoint sets it)"
                    .to_string(),
            );
        }
        if let Some(types) = &args.layer_types
            && types.len() < args.num_hidden_layers
        {
            return Err(format!(
                "Laguna: layer_types has {} entries for {} layers",
                types.len(),
                args.num_hidden_layers
            ));
        }
        args.validate()?;
        let nvfp4_planes = weights.keys().any(|k| k.ends_with(".global_scale"));
        let quant = args.quant_spec(nvfp4_planes);

        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            "model.embed_tokens",
            quant.group_size,
            quant.bits,
        )?;

        let full_rope = args.layer_rope(FULL_ATTENTION);
        let sliding_rope = args.layer_rope(SLIDING_ATTENTION);
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let rope = if args.is_sliding(i) {
                &sliding_rope
            } else {
                &full_rope
            };
            layers.push(DecoderLayer::from_weights(weights, args, &quant, i, rope)?);
        }

        let norm_weight =
            crate::models::laguna_layers::get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights_with_mode(
                weights,
                "lm_head",
                quant.group_size,
                quant.bits,
                quant.mode,
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: args.sliding_window,
            eos_token_ids: args.eos_token_ids(),
        })
    }
}

/// `LanguageModel` wrapper owning the mixed full/sliding caches.
pub struct LagunaWrapper {
    pub model: LagunaModel,
    caches: RefCell<Vec<LagunaCache>>,
}

impl LagunaWrapper {
    pub fn new(model: LagunaModel) -> Self {
        let caches = model.make_caches();
        Self {
            model,
            caches: RefCell::new(caches),
        }
    }

    pub fn reset_caches(&self) {
        *self.caches.borrow_mut() = self.model.make_caches();
    }
}

impl mlxcel_core::generate::LanguageModel for LagunaWrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut caches = self.caches.borrow_mut();
        self.model.forward_with_caches(input_ids, &mut caches)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset the internal mixed caches; the returned slice is unused (the
        // model owns its full/sliding cache state).
        self.reset_caches();
        (0..self.model.layers.len())
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    fn supports_batching(&self) -> bool {
        // Mixed full/sliding caches live in a RefCell, which is not compatible
        // with per-sequence KV isolation.
        false
    }

    fn reset_runtime_state(&self) {
        self.reset_caches();
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.model.eos_token_ids.clone()
    }

    fn embed_tokens_module(&self) -> Option<UnifiedEmbedding> {
        Some(self.model.embed_tokens.clone_shared())
    }

    fn lm_head_module(&self) -> Option<UnifiedLinear> {
        self.model.lm_head.as_ref().map(UnifiedLinear::clone_shared)
    }
}
