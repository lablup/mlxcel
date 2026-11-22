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

//! Xing4.0 (`model_type: xing4_0`, `Xing4_0ForCausalLM`, issue #2356).
//!
//! The DeepSeek-V3 decoder (MLA attention, `noaux_tc` MoE, YaRN RoPE) wrapped
//! in a manifold hyper-connection residual stream of width `hc_mult`. Port of
//! mlx-vlm's `mlx_vlm/models/xing4_0/language.py`; attention, MoE and the
//! weight sanitize are the DeepSeek-V3 ones, reused by composition.
//!
//! The hidden state between blocks is `[B, L, hc, D]`. Each sublayer runs
//! `(post, comb, collapsed) = hc(x)`, feeds `collapsed` (`[B, L, D]`) through
//! its norm and the attention or MLP, and folds the output back with
//! `post[..., None] * out[:, :, None] + comb @ x`.
//!
//! The connection differs from DeepSeek-V4's (`deepseek_v4_hyper.rs`) in
//! every place that module calls silently wrong if approximated, so it is a
//! separate implementation, not a reuse:
//!
//! * the RMS norm runs in f32 but the `@ hc_fn.T` matmul runs in the
//!   activation dtype, and only its result is promoted to f32;
//! * `pre = sigmoid(..)` with no `+ hc_eps`;
//! * `comb` is clipped to `[mhc_h_res_clamp_min, mhc_h_res_clamp_max]` and
//!   exponentiated against its row max with no softmax normalization and no
//!   `+ eps`, followed by exactly `hc_sinkhorn_iters` row-then-column rounds
//!   (no initial column pass);
//! * the expand is `comb @ x`, not `comb.T @ residual`;
//! * the head is a plain `mean` over the streams (no `HyperHead` weights);
//! * collapse, expand and the head run in the activation dtype.

use std::path::Path;

use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, route_batched_decode};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::{Deserialize, Deserializer};

use crate::models::deepseek_v3::{
    DecoderLayer, DeepSeekV3Attention, DeepSeekV3Config, DeepSeekV3Model, MLPType,
};
use crate::models::rope_utils::{RopeScalingKind, RopeScalingSpec};

/// Upper bound on `hc_mult`. The released checkpoints use 4; the bound only
/// keeps `(2 + hc) * hc` and `hc * hidden_size` far from i32 overflow.
const MAX_HC_MULT: usize = 64;
/// Upper bound on `hc_sinkhorn_iters` (20 on the released checkpoints).
const MAX_SINKHORN_ITERS: usize = 1000;

fn default_hc_mult() -> usize {
    4
}
fn default_sinkhorn_iters() -> usize {
    20
}
fn default_hc_eps() -> f32 {
    1e-6
}
fn default_clamp_min() -> f32 {
    -30.0
}
fn default_clamp_max() -> f32 {
    30.0
}

/// The hyper-connection keys `Xing4_0Config` adds over DeepSeek-V3, with the
/// mlx-vlm `ModelConfig` defaults.
#[derive(Debug, Clone, Deserialize)]
struct HyperFields {
    #[serde(default = "default_hc_mult")]
    hc_mult: usize,
    #[serde(default = "default_sinkhorn_iters")]
    hc_sinkhorn_iters: usize,
    #[serde(default = "default_hc_eps")]
    hc_eps: f32,
    #[serde(default = "default_clamp_min")]
    mhc_h_res_clamp_min: f32,
    #[serde(default = "default_clamp_max")]
    mhc_h_res_clamp_max: f32,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

/// Xing4.0 configuration: the DeepSeek-V3 config plus the hyper-connection
/// fields.
#[derive(Debug, Clone)]
pub struct Xing40Config {
    pub base: DeepSeekV3Config,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub mhc_h_res_clamp_min: f32,
    pub mhc_h_res_clamp_max: f32,
    /// From `eos_token_id` (an integer or a list); `<_end>` = 2 when absent.
    pub eos_token_ids: Vec<i32>,
}

impl<'de> Deserialize<'de> for Xing40Config {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let base = DeepSeekV3Config::deserialize(&value).map_err(serde::de::Error::custom)?;
        let hyper = HyperFields::deserialize(&value).map_err(serde::de::Error::custom)?;
        let eos_token_ids = parse_eos(hyper.eos_token_id.as_ref());
        let config = Self {
            base,
            hc_mult: hyper.hc_mult,
            hc_sinkhorn_iters: hyper.hc_sinkhorn_iters,
            hc_eps: hyper.hc_eps,
            mhc_h_res_clamp_min: hyper.mhc_h_res_clamp_min,
            mhc_h_res_clamp_max: hyper.mhc_h_res_clamp_max,
            eos_token_ids,
        };
        config.validate().map_err(serde::de::Error::custom)?;
        Ok(config)
    }
}

fn parse_eos(value: Option<&serde_json::Value>) -> Vec<i32> {
    let ids: Vec<i32> = match value {
        Some(serde_json::Value::Number(n)) => n.as_i64().map(|v| v as i32).into_iter().collect(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_i64())
            .map(|v| v as i32)
            .collect(),
        _ => Vec::new(),
    };
    if ids.is_empty() { vec![2] } else { ids }
}

impl Xing40Config {
    /// Refuse hyper-connection settings the forward cannot run as declared.
    pub fn validate(&self) -> Result<(), String> {
        if self.hc_mult == 0 || self.hc_mult > MAX_HC_MULT {
            return Err(format!(
                "xing4_0: hc_mult {} is outside 1..={MAX_HC_MULT}",
                self.hc_mult
            ));
        }
        if self.hc_sinkhorn_iters > MAX_SINKHORN_ITERS {
            return Err(format!(
                "xing4_0: hc_sinkhorn_iters {} exceeds {MAX_SINKHORN_ITERS}",
                self.hc_sinkhorn_iters
            ));
        }
        if !(self.hc_eps.is_finite() && self.hc_eps >= 0.0) {
            return Err(format!(
                "xing4_0: hc_eps {} is not a non-negative finite number",
                self.hc_eps
            ));
        }
        let (lo, hi) = (self.mhc_h_res_clamp_min, self.mhc_h_res_clamp_max);
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(format!(
                "xing4_0: mhc_h_res_clamp_min {lo} / mhc_h_res_clamp_max {hi} is not a finite, ordered range"
            ));
        }
        let width = (self.hc_mult as u64) * (self.base.hidden_size as u64);
        if width > i32::MAX as u64 {
            return Err(format!(
                "xing4_0: hc_mult * hidden_size = {width} does not fit in i32"
            ));
        }
        Ok(())
    }
}

/// `x * scale + base` with `scale` a `[1]` slice; all f32.
fn scaled_shift(x: &MlxArray, scale: &MlxArray, base: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::add(&mlxcel_core::multiply(x, scale), base)
}

fn expect_shape(
    weights: &WeightMap,
    name: &str,
    expected: &[i32],
) -> Result<UniquePtr<MlxArray>, String> {
    let w = weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))?;
    let shape = mlxcel_core::array_shape(&w);
    if shape != expected {
        return Err(format!(
            "{name}: expected shape {expected:?}, checkpoint ships {shape:?}"
        ));
    }
    Ok(w)
}

/// One Xing hyper connection (`attn_hc` / `ffn_hc`).
pub(crate) struct XingHyperConnection {
    /// `[(2 + hc) * hc, hc * hidden]`, in the checkpoint dtype; cast to the
    /// activation dtype at the matmul.
    hc_fn: UniquePtr<MlxArray>,
    /// The `hc_scale` / `hc_base` slices, as f32 (the reference promotes them
    /// against the f32 mix).
    scale_pre: UniquePtr<MlxArray>,
    scale_post: UniquePtr<MlxArray>,
    scale_comb: UniquePtr<MlxArray>,
    base_pre: UniquePtr<MlxArray>,
    base_post: UniquePtr<MlxArray>,
    /// `[hc, hc]`.
    base_comb: UniquePtr<MlxArray>,
    clamp_min: UniquePtr<MlxArray>,
    clamp_max: UniquePtr<MlxArray>,
    eps: UniquePtr<MlxArray>,
    hc_mult: i32,
    sinkhorn_iters: usize,
    norm_eps: f32,
}

impl XingHyperConnection {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Xing40Config,
    ) -> Result<Self, String> {
        let hc = config.hc_mult as i32;
        let hidden = config.base.hidden_size as i32;
        let mix = (2 + hc) * hc;
        let hc_fn = expect_shape(weights, &format!("{prefix}.hc_fn"), &[mix, hc * hidden])?;
        let base = expect_shape(weights, &format!("{prefix}.hc_base"), &[mix])?;
        let scale = expect_shape(weights, &format!("{prefix}.hc_scale"), &[3])?;
        let f32_ = mlxcel_core::dtype::FLOAT32;
        let base = mlxcel_core::astype(&base, f32_);
        let scale = mlxcel_core::astype(&scale, f32_);
        let base_comb = slice_axis(&base, 0, 2 * hc, mix);
        let this = Self {
            hc_fn,
            scale_pre: slice_axis(&scale, 0, 0, 1),
            scale_post: slice_axis(&scale, 0, 1, 2),
            scale_comb: slice_axis(&scale, 0, 2, 3),
            base_pre: slice_axis(&base, 0, 0, hc),
            base_post: slice_axis(&base, 0, hc, 2 * hc),
            base_comb: mlxcel_core::reshape(&base_comb, &[hc, hc]),
            clamp_min: mlxcel_core::full_f32(&[1], config.mhc_h_res_clamp_min, f32_),
            clamp_max: mlxcel_core::full_f32(&[1], config.mhc_h_res_clamp_max, f32_),
            eps: mlxcel_core::full_f32(&[1], config.hc_eps, f32_),
            hc_mult: hc,
            sinkhorn_iters: config.hc_sinkhorn_iters,
            norm_eps: config.base.rms_norm_eps,
        };
        Ok(this)
    }

    /// `x` is `[B, L, hc, D]`. Returns `(post [B, L, hc], comb [B, L, hc, hc],
    /// collapsed [B, L, D])`, all in `x`'s dtype.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let shape = mlxcel_core::array_shape(x);
        let (b, l, h, d) = (shape[0], shape[1], shape[2], shape[3]);
        let hc = self.hc_mult;
        debug_assert_eq!(h, hc);
        let x_dtype = mlxcel_core::array_dtype(x);
        let f32_ = mlxcel_core::dtype::FLOAT32;

        // RMS-normalize the flattened state in f32, then mix in the
        // activation dtype and promote the result to f32.
        let flat = mlxcel_core::reshape(x, &[b, l, h * d]);
        let flat = mlxcel_core::astype(&flat, f32_);
        let flat = mlxcel_core::fast_rms_norm_no_weight(&flat, self.norm_eps);
        let flat = mlxcel_core::astype(&flat, x_dtype);
        let fn_t = mlxcel_core::transpose(&mlxcel_core::astype(&self.hc_fn, x_dtype));
        let mix = mlxcel_core::astype(&mlxcel_core::matmul(&flat, &fn_t), f32_);

        let pre = slice_axis(&mix, -1, 0, hc);
        let post = slice_axis(&mix, -1, hc, 2 * hc);
        let comb = slice_axis(&mix, -1, 2 * hc, (2 + hc) * hc);

        let pre = mlxcel_core::sigmoid(&scaled_shift(&pre, &self.scale_pre, &self.base_pre));
        let post = mlxcel_core::sigmoid(&scaled_shift(&post, &self.scale_post, &self.base_post));
        let post = mlxcel_core::multiply_scalar(&post, 2.0);

        let comb = mlxcel_core::reshape(&comb, &[b, l, hc, hc]);
        let comb = scaled_shift(&comb, &self.scale_comb, &self.base_comb);
        let comb = mlxcel_core::clip(&comb, &self.clamp_min, &self.clamp_max);
        let row_max = mlxcel_core::max_axis(&comb, -1, true);
        let mut comb = mlxcel_core::exp(&mlxcel_core::subtract(&comb, &row_max));
        for _ in 0..self.sinkhorn_iters {
            let row_sum = mlxcel_core::sum_axis(&comb, -1, true);
            comb = mlxcel_core::divide(&comb, &mlxcel_core::add(&row_sum, &self.eps));
            let col_sum = mlxcel_core::sum_axis(&comb, -2, true);
            comb = mlxcel_core::divide(&comb, &mlxcel_core::add(&col_sum, &self.eps));
        }

        // collapsed = (pre[..., None].astype(x.dtype) * x).sum(axis=2)
        let pre = mlxcel_core::astype(&mlxcel_core::expand_dims(&pre, -1), x_dtype);
        let collapsed =
            mlxcel_core::sum_axis_native_dtype(&mlxcel_core::multiply(&pre, x), 2, false);

        (
            mlxcel_core::astype(&post, x_dtype),
            mlxcel_core::astype(&comb, x_dtype),
            collapsed,
        )
    }
}

/// Fold a sublayer output back into the widened state:
/// `post[..., None] * out[:, :, None] + comb @ x`, in the activation dtype.
pub(crate) fn hc_expand(
    out: &MlxArray,
    x: &MlxArray,
    post: &MlxArray,
    comb: &MlxArray,
) -> UniquePtr<MlxArray> {
    let post = mlxcel_core::expand_dims(post, -1);
    let out = mlxcel_core::expand_dims(out, 2);
    let gated = mlxcel_core::multiply(&post, &out);
    let mixed = mlxcel_core::matmul(comb, x);
    mlxcel_core::add(&gated, &mixed)
}

/// Widen `[B, L, D]` embeddings to `[B, L, hc, D]` by repeating each row.
pub(crate) fn hc_widen(h: &MlxArray, hc_mult: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::repeat(&mlxcel_core::expand_dims(h, 2), hc_mult, 2)
}

/// The model head's collapse: `h.mean(axis=2)` in the activation dtype.
pub(crate) fn hc_head(h: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::mean_axis_native_dtype(h, 2, false)
}

/// One Xing decoder layer: the DeepSeek-V3 attention, MLP and norms, with
/// each residual replaced by a hyper connection.
pub struct XingDecoderLayer {
    pub(crate) attn_hc: XingHyperConnection,
    pub(crate) ffn_hc: XingHyperConnection,
    pub self_attn: DeepSeekV3Attention,
    pub mlp: MLPType,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

fn mlp_forward(mlp: &MLPType, x: &MlxArray) -> UniquePtr<MlxArray> {
    match mlp {
        MLPType::Dense(mlp) => mlp.forward(x),
        MLPType::MoE(moe) => moe.forward(x),
    }
}

impl XingDecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (post, comb, collapsed) = self.attn_hc.forward(x);
        let normed = self.input_layernorm.forward(&collapsed);
        let out = self.self_attn.forward(&normed, cache, mask);
        let x = hc_expand(&out, x, &post, &comb);

        let (post, comb, collapsed) = self.ffn_hc.forward(&x);
        let normed = self.post_attention_layernorm.forward(&collapsed);
        let out = mlp_forward(&self.mlp, &normed);
        hc_expand(&out, &x, &post, &comb)
    }

    pub fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
    ) -> UniquePtr<MlxArray> {
        let (post, comb, collapsed) = self.attn_hc.forward(x);
        let normed = self.input_layernorm.forward(&collapsed);
        let out = self.self_attn.forward_batched_decode(&normed, caches);
        let x = hc_expand(&out, x, &post, &comb);

        let (post, comb, collapsed) = self.ffn_hc.forward(&x);
        let normed = self.post_attention_layernorm.forward(&collapsed);
        let out = mlp_forward(&self.mlp, &normed);
        hc_expand(&out, &x, &post, &comb)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Xing40Config,
        rope: &RopeScalingKind,
        rope_base_override: f32,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let DecoderLayer {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        } = DecoderLayer::from_weights(weights, &config.base, layer_idx)?;
        let self_attn = match rope.freqs() {
            Some(freqs) => self_attn.with_rope_table(mlxcel_core::copy(freqs), rope.attn_scale()),
            // No table: the plain rotation runs at `rope_base`, which must carry
            // a `--rope-freq-base` override just as the table would have, or the
            // override is counted as applied while the rotation ignores it.
            None => DeepSeekV3Attention {
                rope_base: rope_base_override,
                ..self_attn
            },
        };
        let prefix = format!("model.layers.{layer_idx}");
        Ok(Self {
            attn_hc: XingHyperConnection::from_weights(
                weights,
                &format!("{prefix}.attn_hc"),
                config,
            )?,
            ffn_hc: XingHyperConnection::from_weights(
                weights,
                &format!("{prefix}.ffn_hc"),
                config,
            )?,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

/// Resolve the attention's RoPE table from the config's `rope_scaling` block
/// (YaRN on the released checkpoints). The DeepSeek-V3 attention rotates at
/// unit position scale with a table or at `rope_theta` without one, so a
/// `linear` block, which needs a position scale, is refused.
fn resolve_rope(config: &DeepSeekV3Config) -> Result<RopeScalingKind, String> {
    let spec = config
        .rope_scaling
        .as_ref()
        .map(|block| RopeScalingSpec::from_lookup(|key| block.get(key)));
    let kind = RopeScalingKind::resolve(
        spec.as_ref(),
        config.qk_rope_head_dim,
        config.rope_theta,
        Some(config.max_position_embeddings as f32),
        "xing4_0",
    );
    if let RopeScalingKind::Linear { .. } = kind {
        return Err(
            "xing4_0: a linear rope_scaling block is not supported by the MLA attention"
                .to_string(),
        );
    }
    Ok(kind)
}

/// Xing4.0 causal LM.
pub struct Xing40Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<XingDecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
    pub config: Xing40Config,
}

impl Xing40Model {
    /// The single-sequence forward, projecting every position or only `row`.
    fn forward_rows(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        let mut h = hc_widen(&h, self.config.hc_mult as i32);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, cache, mask);
        }
        let h = mlxcel_core::generate::select_row(hc_head(&h), row);
        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows(input_ids, caches, mask, None)
    }

    pub fn forward_batched_impl(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
    ) -> UniquePtr<MlxArray> {
        crate::models::mla_batched::validate_mla_decode_batch(
            "Xing40Model",
            input_ids,
            batch_caches,
            self.layers.len(),
        );
        let h = self.embed_tokens.forward(input_ids);
        let mut h = hc_widen(&h, self.config.hc_mult as i32);
        for layer_idx in 0..self.layers.len() {
            let mut layer_caches: Vec<&mut KVCache> = batch_caches
                .iter_mut()
                .map(|row| &mut row[layer_idx])
                .collect();
            h = self.layers[layer_idx].forward_batched_decode(&h, &mut layer_caches);
        }
        let h = self.norm.forward(&hc_head(&h));
        self.lm_head.forward(&h)
    }

    pub fn make_caches_impl(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, Xing40Config), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config: Xing40Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = Self::sanitize_weights(weights, &config)?;
        let model = Self::from_weights(&weights, &config)?;
        Ok((model, config))
    }

    /// Map the mlx-vlm `language_model.` root onto the text layout and apply
    /// the DeepSeek-V3 sanitize, which also drops the MTP trailer
    /// `model.layers.{num_hidden_layers}.*` (the upstream Xing sanitize step).
    pub fn sanitize_weights(
        mut weights: WeightMap,
        config: &Xing40Config,
    ) -> Result<WeightMap, String> {
        crate::models::embedding_sanitize::strip_language_model_wrapper(&mut weights);
        DeepSeekV3Model::sanitize_weights_with_args(weights, &config.base)
    }

    /// Build from an already sanitized weight map (the adapter route hands
    /// the raw map, so both layouts are accepted).
    pub fn from_weights(weights: &WeightMap, config: &Xing40Config) -> Result<Self, String> {
        if weights.keys().any(|k| k.starts_with("language_model.")) {
            let mut owned = WeightMap::new();
            for (k, v) in weights.iter() {
                owned.insert(k.clone(), mlxcel_core::copy(v));
            }
            let owned = Self::sanitize_weights(owned, config)?;
            return Self::build(&owned, config);
        }
        Self::build(weights, config)
    }

    fn build(weights: &WeightMap, config: &Xing40Config) -> Result<Self, String> {
        config.validate()?;
        let args = &config.base;
        let group_size = args.group_size();
        let bits = args.bits();
        let rope = resolve_rope(args)?;
        // `rope_theta` after any `--rope-freq-base` override, for the layers
        // that rotate without a table (a `default` scheme, or a block that
        // resolved to one). `RopeScalingKind::resolve` applies the same
        // override to the table it builds.
        let rope_base = crate::models::rope_overrides::resolve_base(args.rope_theta);
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;
        let layers = (0..args.num_hidden_layers)
            .map(|i| XingDecoderLayer::from_weights(weights, config, &rope, rope_base, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm_weight = weights
            .get("model.norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Weight not found: model.norm.weight".to_string())?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: config.clone(),
        })
    }
}

impl LanguageModel for Xing40Model {
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows(input_ids, caches, mask, None)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<mlxcel_core::cache::SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        // The trait-level embeddings forward ignores embeddings; decline so
        // that behavior stays the default's.
        if input_embeddings.is_some() {
            return None;
        }
        Some(self.forward_rows(input_ids, caches, mask, Some(last_pos)))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.make_caches_impl()
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        route_batched_decode(self, input_ids, batch_caches, |caches| {
            self.forward_batched_impl(input_ids, caches)
        })
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.config.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "xing4_0_tests.rs"]
mod tests;
