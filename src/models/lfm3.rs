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
// Derived from mlx-vlm's LFM3 (MIT project) and the Apache-2.0 checkpoint
// reference implementation, Copyright 2026 The HuggingFace Team. See NOTICE.

//! LiquidAI LFM3 causal text backbone. Full and sliding attention share the
//! same gated operator but use different rotary periods and cache lifetimes.

use super::gemma3::{Cache, CacheInterface};
use super::kv_snapshot::KvSnapshotNames;
use super::lfm2::{Quantization, ShortConv, ShortConvParams, transpose_short_conv_weights};
use super::llama3::EosTokenId;
use super::model_owned::{KvCacheLayerModes, ModelOwnedSequenceState};
use super::recurrent_snapshot::{push_optional, restore_optional};
use mlxcel_core as mx;
use mx::cache::{CachePool, KVCacheMode, LayerStateKind, SequenceId, SequenceStateLayout};
use mx::generate::{LanguageModel, ModelStateSnapshot};
use mx::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mx::weights::WeightMap;
use mx::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct Rope {
    rope_theta: f32,
    partial_rotary_factor: f32,
    rope_type: String,
}
impl Default for Rope {
    fn default() -> Self {
        Self {
            rope_theta: 1e6,
            partial_rotary_factor: 1.,
            rope_type: "default".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Lfm3Args {
    hidden_size: i32,
    intermediate_size: i32,
    vocab_size: i32,
    num_hidden_layers: usize,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: Option<i32>,
    layer_types: Option<Vec<String>>,
    sliding_window: Option<i32>,
    #[serde(rename = "conv_L_cache")]
    conv_l_cache: i32,
    conv_bias: bool,
    use_sandwich_norm: bool,
    norm_eps: f32,
    tie_word_embeddings: bool,
    eos_token_id: EosTokenId,
    #[serde(deserialize_with = "deserialize_rope_parameters")]
    rope_parameters: BTreeMap<String, Rope>,
    quantization: Option<Quantization>,
    quantization_config: serde_json::Value,
}

fn default_rope_parameters() -> BTreeMap<String, Rope> {
    BTreeMap::from([
        (
            "full_attention".into(),
            Rope {
                partial_rotary_factor: 0.5,
                ..Default::default()
            },
        ),
        (
            "sliding_attention".into(),
            Rope {
                rope_theta: 1e4,
                ..Default::default()
            },
        ),
    ])
}

fn deserialize_rope_parameters<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, Rope>, D::Error> {
    Ok(Option::deserialize(deserializer)?.unwrap_or_else(default_rope_parameters))
}

impl Default for Lfm3Args {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 7680,
            vocab_size: 128000,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: None,
            layer_types: None,
            sliding_window: Some(256),
            conv_l_cache: 7,
            conv_bias: false,
            use_sandwich_norm: true,
            norm_eps: 1e-5,
            tie_word_embeddings: true,
            eos_token_id: EosTokenId::Single(124900),
            quantization: None,
            quantization_config: serde_json::Value::Null,
            rope_parameters: default_rope_parameters(),
        }
    }
}
impl Lfm3Args {
    fn dim(&self) -> i32 {
        self.head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
    }

    fn validate(&self) -> Result<&[String], String> {
        if let Some(method) = self.quantization_config["quant_method"].as_str() {
            return Err(format!("LFM3 does not support {method} quantization"));
        }
        let types = self
            .layer_types
            .as_deref()
            .ok_or("LFM3 requires an explicit layer_types list")?;
        if types.len() != self.num_hidden_layers || types.is_empty() {
            return Err("layer_types must contain num_hidden_layers entries (at least one)".into());
        }
        if self.num_attention_heads <= 0
            || self.num_key_value_heads <= 0
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(
                "num_attention_heads must be a positive multiple of num_key_value_heads".into(),
            );
        }
        let dim = self.dim();
        if self.hidden_size <= 0
            || self.hidden_size.checked_mul(3).is_none()
            || dim <= 0
            || dim % 2 != 0
            || dim.checked_mul(self.num_attention_heads).is_none()
        {
            return Err("invalid hidden_size or head_dim".into());
        }
        if self.intermediate_size <= 0 || self.vocab_size <= 0 {
            return Err("intermediate_size and vocab_size must be positive".into());
        }
        if self.conv_l_cache < 1 {
            return Err("conv_L_cache must be positive".into());
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0. {
            return Err("norm_eps must be positive".into());
        }
        for kind in types {
            match kind.as_str() {
                "conv" => continue,
                "sliding_attention" if self.sliding_window.unwrap_or(0) <= 0 => {
                    return Err("Sliding attention requires a positive sliding_window".into());
                }
                "full_attention" | "sliding_attention" => {}
                _ => {
                    return Err(format!(
                        "Unsupported LFM3 layer type in layer_types: {kind}"
                    ));
                }
            }
            let r = self
                .rope_parameters
                .get(kind)
                .ok_or_else(|| format!("missing rope_parameters.{kind}"))?;
            if r.rope_type != "default" {
                return Err(format!(
                    "LFM3 supports default rotary embeddings only (rope_type: {})",
                    r.rope_type
                ));
            }
            if r.rope_theta <= 0. || !r.rope_theta.is_finite() {
                return Err("rope_theta must be positive".into());
            }
            if !(0.0..=1.0).contains(&r.partial_rotary_factor)
                || f64::from(dim) * f64::from(r.partial_rotary_factor) < 2.
            {
                return Err("Invalid partial_rotary_factor".into());
            }
        }
        Ok(types)
    }
    fn quant(&self) -> (i32, i32) {
        self.quantization
            .as_ref()
            .map_or((64, 4), |q| (q.group_size, q.bits))
    }
}

fn norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    let name = format!("{prefix}.weight");
    let weight = weights
        .get(&name)
        .ok_or_else(|| format!("Missing weight: {name}"))?;
    Ok(RMSNorm::new(mx::copy(weight), eps))
}

struct Attention {
    q: UnifiedLinear,
    k: UnifiedLinear,
    v: UnifiedLinear,
    o: UnifiedLinear,
    gate: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    heads: i32,
    kv_heads: i32,
    dim: i32,
    periods: UniquePtr<MlxArray>,
}

impl Attention {
    fn new(
        weights: &WeightMap,
        args: &Lfm3Args,
        prefix: &str,
        rope: &Rope,
    ) -> Result<Self, String> {
        let (gs, bits) = args.quant();
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), gs, bits)
        };
        let dim = args.dim();
        let periods = mx::rope_proportional::compute_proportional_rope_freqs(
            dim,
            rope.partial_rotary_factor,
            rope.rope_theta,
            1.,
        )
        .ok_or("Invalid partial_rotary_factor")?;
        Ok(Self {
            q: linear("q_proj")?,
            k: linear("k_proj")?,
            v: linear("v_proj")?,
            o: linear("o_proj")?,
            gate: linear("gate_proj")?,
            q_norm: norm(weights, &format!("{prefix}.q_norm"), args.norm_eps)?,
            k_norm: norm(weights, &format!("{prefix}.k_norm"), args.norm_eps)?,
            heads: args.num_attention_heads,
            kv_heads: args.num_key_value_heads,
            dim,
            periods,
        })
    }
    fn rotate(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        mx::fast_rope_with_freqs(x, self.dim, false, 1., offset, &self.periods)
    }
    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mx::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let project = |linear: &UnifiedLinear, heads: i32| {
            mx::reshape(&linear.forward(x), &[b, l, heads, self.dim])
        };
        let q = self.q_norm.forward(&project(&self.q, self.heads));
        let k = self.k_norm.forward(&project(&self.k, self.kv_heads));
        let q = self.rotate(&mx::transpose_axes(&q, &[0, 2, 1, 3]), cache.offset());
        let k = self.rotate(&mx::transpose_axes(&k, &[0, 2, 1, 3]), cache.offset());
        let v = mx::transpose_axes(&project(&self.v, self.kv_heads), &[0, 2, 1, 3]);
        let out = cache.attend(&q, k, v, (self.dim as f32).powf(-0.5), mask);
        let out = mx::transpose_axes(&out, &[0, 2, 1, 3]);
        let gate = mx::reshape(&mx::sigmoid(&self.gate.forward(x)), &[b, l, self.heads, 1]);
        self.o
            .forward(&mx::reshape(&mx::multiply(&out, &gate), &[b, l, -1]))
    }
}

enum Mixer {
    Conv(ShortConv),
    Attention(Attention),
}
enum LayerCache {
    Conv(Option<UniquePtr<MlxArray>>),
    Attention(Cache),
}
struct Layer {
    mixer: Mixer,
    kind: LayerStateKind,
    operator_norm: RMSNorm,
    ffn_norm: RMSNorm,
    post_operator: Option<RMSNorm>,
    post_ffn: Option<RMSNorm>,
    w1: UnifiedLinear,
    w2: UnifiedLinear,
    w3: UnifiedLinear,
}

impl Layer {
    fn new(weights: &WeightMap, args: &Lfm3Args, idx: usize, kind: &str) -> Result<Self, String> {
        let p = format!("model.layers.{idx}");
        let (group_size, bits) = args.quant();
        let (mixer, kind) = if kind == "conv" {
            let prefix = format!("{p}.conv");
            (
                Mixer::Conv(ShortConv::from_parts(
                    weights,
                    &prefix,
                    &ShortConvParams {
                        hidden_size: args.hidden_size,
                        l_cache: args.conv_l_cache,
                        require_conv_bias: args.conv_bias,
                        causal: true,
                        group_size,
                        bits,
                    },
                )?),
                LayerStateKind::Recurrent,
            )
        } else {
            let attn = Attention::new(
                weights,
                args,
                &format!("{p}.self_attn"),
                &args.rope_parameters[kind],
            )?;
            let kind = if kind == "full_attention" {
                LayerStateKind::Full
            } else {
                LayerStateKind::sliding(args.sliding_window.unwrap())
            };
            (Mixer::Attention(attn), kind)
        };
        let layer_norm = |name: &str| norm(weights, &format!("{p}.{name}"), args.norm_eps);
        let linear = |name: &str| {
            UnifiedLinear::from_weights(
                weights,
                &format!("{p}.feed_forward.{name}"),
                group_size,
                bits,
            )
        };
        Ok(Self {
            mixer,
            kind,
            operator_norm: layer_norm("operator_norm")?,
            ffn_norm: layer_norm("ffn_norm")?,
            post_operator: args
                .use_sandwich_norm
                .then(|| layer_norm("post_operator_norm"))
                .transpose()?,
            post_ffn: args
                .use_sandwich_norm
                .then(|| layer_norm("post_ffn_norm"))
                .transpose()?,
            w1: linear("w1")?,
            w2: linear("w2")?,
            w3: linear("w3")?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut LayerCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = self.operator_norm.forward(x);
        let h = match (&self.mixer, cache) {
            (Mixer::Conv(conv), LayerCache::Conv(state)) => {
                conv.forward_with_capture(&h, state, None, None)
            }
            (Mixer::Attention(attn), LayerCache::Attention(cache)) => {
                attn.forward(&h, cache.as_interface(), mask)
            }
            _ => unreachable!("cache kind is fixed at construction"),
        };
        let residual = |x: &MlxArray, h: UniquePtr<MlxArray>, norm: &Option<RMSNorm>| {
            let h = match norm {
                Some(norm) => norm.forward(&h),
                None => h,
            };
            mx::add(x, &h)
        };
        let x = residual(x, h, &self.post_operator);
        let h = self.ffn_norm.forward(&x);
        let h = self.w2.forward(&mx::compiled_swiglu_activation(
            &self.w1.forward(&h),
            &self.w3.forward(&h),
        ));
        residual(&x, h, &self.post_ffn)
    }
}

pub struct Lfm3Model {
    args: Lfm3Args,
    embed: UnifiedEmbedding,
    layers: Vec<Layer>,
    final_norm: RMSNorm,
    head: Option<UnifiedLinear>,
    state: ModelOwnedSequenceState<LayerCache>,
    modes: KvCacheLayerModes,
}
impl Lfm3Model {
    pub fn load(path: &str) -> Result<(Self, Lfm3Args), String> {
        let dir = Path::new(path);
        let json = std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
        let args: Lfm3Args =
            serde_json::from_str(&super::sanitize_config_json(&json)).map_err(|e| e.to_string())?;
        args.validate()?;
        let weights = super::load_text_weights(dir, None)?;
        Ok((Self::from_weights(args.clone(), weights)?, args))
    }
    pub fn from_weights(args: Lfm3Args, weights: WeightMap) -> Result<Self, String> {
        let types = args.validate()?;
        let mut canonical = WeightMap::new();
        for (key, w) in weights {
            let name = key.strip_prefix("language_model.").unwrap_or(&key);
            if args.tie_word_embeddings && name.starts_with("lm_head.") {
                continue;
            }
            if canonical.insert(name.to_owned(), w).is_some() {
                return Err(format!("duplicate LFM3 weight: {name}"));
            }
        }
        transpose_short_conv_weights(&mut canonical);
        let (gs, bits) = args.quant();
        let embed = UnifiedEmbedding::from_weights(&canonical, "model.embed_tokens", gs, bits)?;
        let final_norm = norm(&canonical, "model.embedding_norm", args.norm_eps)?;
        let head = (!args.tie_word_embeddings)
            .then(|| UnifiedLinear::from_weights(&canonical, "lm_head", gs, bits))
            .transpose()?;
        let layers = types
            .iter()
            .enumerate()
            .map(|(i, t)| Layer::new(&canonical, &args, i, t))
            .collect::<Result<Vec<_>, _>>()?;
        let model = Self {
            args,
            embed,
            layers,
            final_norm,
            head,
            state: ModelOwnedSequenceState::new(Vec::new()),
            modes: KvCacheLayerModes::new(),
        };
        model.reset_runtime_state();
        Ok(model)
    }
    fn caches(&self) -> Vec<LayerCache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(idx, layer)| match layer.kind {
                LayerStateKind::Full => LayerCache::Attention(Cache::Standard(
                    KVCache::new_with_mode(self.modes.mode_for_layer(idx)),
                )),
                LayerStateKind::Sliding { window } => {
                    LayerCache::Attention(Cache::Rotating(RotatingKVCache::new_with_mode(
                        window,
                        self.modes.rotating_mode_for_layer(idx, window),
                    )))
                }
                _ => LayerCache::Conv(None),
            })
            .collect()
    }
    fn run(
        &self,
        ids: &MlxArray,
        embeds: Option<&MlxArray>,
        seq: Option<SequenceId>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        self.state.with_or_create_sequence_state(
            seq,
            || self.caches(),
            |caches| {
                let mut h = embeds.map_or_else(|| self.embed.forward(ids), mx::copy);
                for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
                    h = layer.forward(&h, cache, None);
                }
                let h = self.final_norm.forward(&mx::generate::select_row(h, row));
                match &self.head {
                    Some(head) => head.forward(&h),
                    None => self.embed.as_linear(&h),
                }
            },
        )
    }
}

const NAMES: KvSnapshotNames = KvSnapshotNames::new("LFM3", "full", "sliding");
impl LanguageModel for Lfm3Model {
    fn forward(
        &self,
        ids: &MlxArray,
        _: &mut [KVCache],
        _: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run(ids, None, None, None)
    }
    fn make_caches(&self) -> Vec<KVCache> {
        self.reset_runtime_state();
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }
    fn num_layers(&self) -> usize {
        self.layers.len()
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        match &self.args.eos_token_id {
            EosTokenId::Single(id) => vec![*id],
            EosTokenId::Multiple(ids) => ids.clone(),
        }
    }
    fn supports_padded_prefill(&self) -> bool {
        false
    }
    fn reset_runtime_state(&self) {
        self.state.replace_internal(self.caches());
    }
    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.num_layers())
    }
    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.modes.set(modes);
        self.reset_runtime_state();
    }
    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.modes.clone_modes()
    }
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        self.modes
            .storage_modes(|i| self.layers.get(i).map_or(LayerStateKind::None, |l| l.kind))
    }
    fn prepare_sequence_state(&self, seq: SequenceId) {
        self.state.prepare_sequence_state(seq, self.caches());
    }
    fn release_sequence_state_by_id(&self, seq: SequenceId) {
        self.state.release_sequence_state(seq);
    }
    fn sync_sequence_storage(&self, seq: SequenceId, pool: &mut CachePool) -> Result<(), String> {
        self.state.with_existing_sequence_state(seq, |state| {
            let lengths = state
                .iter()
                .map(|cache| match cache {
                    LayerCache::Attention(Cache::Standard(c)) => c.live_len() as usize,
                    LayerCache::Attention(Cache::Rotating(c)) => c.visible_len() as usize,
                    LayerCache::Conv(_) => 0,
                })
                .collect::<Vec<_>>();
            pool.sync_paged_state_with_lengths(seq, &lengths)
        })?
    }
    fn embed_tokens(&self, ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed.forward(ids))
    }
    fn forward_with_sequence_id(
        &self,
        ids: &MlxArray,
        seq: Option<SequenceId>,
        _: &mut [KVCache],
        _: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run(ids, None, seq, None)
    }
    fn forward_with_embeddings(
        &self,
        ids: &MlxArray,
        embeds: Option<&MlxArray>,
        _: &mut [KVCache],
        _: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run(ids, embeds, None, None)
    }
    fn forward_with_embeddings_and_sequence_id(
        &self,
        ids: &MlxArray,
        embeds: Option<&MlxArray>,
        seq: Option<SequenceId>,
        _: &mut [KVCache],
        _: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run(ids, embeds, seq, None)
    }
    fn last_row_logits(
        &self,
        ids: &MlxArray,
        embeds: Option<&MlxArray>,
        seq: Option<SequenceId>,
        _: &mut [KVCache],
        _: Option<&MlxArray>,
        pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        Some(self.run(ids, embeds, seq, Some(pos)))
    }
    fn forward_batched(
        &self,
        ids: &MlxArray,
        caches: &mut [&mut [KVCache]],
        _: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Without sequence IDs only a single internal recurrent state is available.
        mx::generate::route_batched_decode_with_ids(self, ids, None, caches, |_| {
            panic!("LFM3 model-owned batched decode requires sequence ids")
        })
    }
    fn forward_batched_with_ids(
        &self,
        ids: &MlxArray,
        seq: Option<&[SequenceId]>,
        caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if let Some(seq) = seq {
            assert_eq!(seq.len(), caches.len(), "LFM3 sequence/cache row mismatch");
            assert_eq!(
                mx::array_shape(ids)[0] as usize,
                seq.len(),
                "LFM3 input/sequence row mismatch"
            );
            if seq.len() > 1 {
                self.state
                    .validate_sequence_states(seq)
                    .expect("LFM3 batched sequence state");
            }
        }
        crate::multimodal::batched_dispatch::forward_batched_with_seq_ids_dispatch(
            self, ids, seq, caches, mask,
        )
    }
    fn supports_snapshot_reuse(&self) -> bool {
        true
    }
    fn snapshot_sequence_state(&self, seq: SequenceId, len: usize) -> Option<ModelStateSnapshot> {
        self.state
            .with_sequence_state_ref(seq, |state| {
                let mut s = ModelStateSnapshot::new("lfm3", len);
                for (idx, c) in state.iter().enumerate() {
                    let prefix = format!("layer{idx}");
                    match c {
                        LayerCache::Conv(v) => {
                            push_optional(&mut s, format!("{prefix}.conv.state"), v)
                        }
                        LayerCache::Attention(kv) => {
                            if let Err(err) = kv.snapshot_into(&mut s, &prefix, NAMES) {
                                tracing::warn!("LFM3 prompt-cache snapshot skipped: {err}");
                                return None;
                            }
                        }
                    }
                }
                Some(s)
            })
            .flatten()
    }
    fn restore_sequence_state(
        &self,
        seq: SequenceId,
        s: &ModelStateSnapshot,
    ) -> Result<(), String> {
        if s.family() != "lfm3" {
            return Err(format!("cannot restore {} snapshot into LFM3", s.family()));
        }
        let mut caches = self.caches();
        for (idx, c) in caches.iter_mut().enumerate() {
            let prefix = format!("layer{idx}");
            match c {
                LayerCache::Attention(kv) => kv.restore_from(s, &prefix, NAMES)?,
                LayerCache::Conv(v) => {
                    *v = restore_optional(s, format!("{prefix}.conv.state"));
                    if s.token_len() > 0 && v.is_none() {
                        return Err(format!("missing {prefix}.conv.state"));
                    }
                }
            }
        }
        self.state.replace_sequence_state(seq, caches);
        Ok(())
    }
}

#[cfg(test)]
#[path = "lfm3_tests.rs"]
mod lfm3_tests;
