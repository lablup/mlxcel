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

//! KRAFTON Raon-1 mini (`model_type: "raon1_mini"`, `Raon1MiniForCausalLM`).
//!
//! A 25-layer text MoE on a 4-stream Raon_mHC residual (`raon_mhc.rs`): 18
//! Raon Delta Attention layers (`raon_rda.rs`) and 7 NoPE gated MLA layers
//! (the q-LoRA NoPE MLA of Kimi K3, `KimiK3MLAAttention`), a dense SiTU MLP
//! in layer 0 and a 128-expert top-6 sigmoid-routed MXFP8 MoE with one shared
//! expert elsewhere (`raon1_mini_moe.rs`), and an untied head.
//!
//! ```text
//! streams = broadcast(embed(ids), [B, L, n, D])
//! per layer, for (site, branch) in [(self_attn_raon_mhc, attn), (mlp_raon_mhc, mlp/moe)]:
//!     route   = site.route(streams)
//!     streams = mix(streams, branch(norm(aggregate(streams, route.h_pre))), route)
//! logits = lm_head(norm(mean(streams.f32, stream).astype(dt)))
//! ```
//!
//! The five `moe.reasoning_effort_steering` vectors are loaded and kept in
//! float32 but not applied: this is the model at steering alpha 0, which the
//! reference produces with `set_reasoning_effort("max")`. Applying them is
//! #3435.
//!
//! Reference: https://huggingface.co/KRAFTON/Raon-1-mini/blob/6691a076/modeling_raon1_mini.py

use super::kimi_k3::{KimiK3MLAAttention, NopeMlaSpec};
use super::model_owned::{KvCacheLayerModes, ModelOwnedSequenceState};
use super::raon_mhc::{self, RaonMhc};
use super::raon_rda::{RaonDeltaAttention, RaonDeltaCache, RdaSpec};
use super::raon1_mini_moe::{MoeSpec, RaonMoe, RaonSituMlp};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use std::path::Path;

pub use super::raon1_mini_config::Raon1MiniConfig;

#[path = "raon1_mini_lm.rs"]
mod lm;

/// Per-layer state: RDA conv window plus recurrent state, or the MLA latent
/// pair (`kv_latent`, `k_pe`) in a `KVCache`.
pub enum Raon1MiniLayerCache {
    Delta(RaonDeltaCache),
    Attn(KVCache),
}

impl Raon1MiniLayerCache {
    pub fn offset(&self) -> i32 {
        match self {
            Raon1MiniLayerCache::Delta(d) => d.offset,
            Raon1MiniLayerCache::Attn(kv) => kv.offset,
        }
    }
}

/// The masks one forward step applies under left padding.
#[derive(Default)]
pub(crate) struct StepMasks {
    /// `[B, L]` in the activation dtype: 1 for real tokens, 0 for padding.
    token_mask: Option<UniquePtr<MlxArray>>,
    /// `[B, 1, L, S]` additive (0 / -inf) mask over the MLA window.
    mla_mask: Option<UniquePtr<MlxArray>>,
}

/// Masks for a step of `l` tokens at cache offset `offset`, where row `b`
/// carries `left_pad[b]` padding tokens at absolute positions `0..pad`.
///
/// A query attends to the keys at or before it that are real, plus itself,
/// so a padded query's row never softmaxes over an empty set.
pub(crate) fn padding_masks(left_pad: &[i32], offset: i32, l: i32, act_dtype: i32) -> StepMasks {
    let b = left_pad.len() as i32;
    let s = offset + l;
    let pad = mlxcel_core::from_slice_i32(left_pad, &[b, 1, 1, 1]);
    let q_pos = mlxcel_core::reshape(&mlxcel_core::arange_i32(offset, s, 1), &[1, 1, l, 1]);
    let k_pos = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, s, 1), &[1, 1, 1, s]);
    let causal = mlxcel_core::less_equal(&k_pos, &q_pos);
    let real_key = mlxcel_core::greater_equal(&k_pos, &pad);
    let own = mlxcel_core::equal(&k_pos, &q_pos);
    let allowed = mlxcel_core::logical_and(&causal, &mlxcel_core::logical_or(&real_key, &own));
    let zero = mlxcel_core::full_f32(&[1], 0.0, act_dtype);
    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, act_dtype);
    let mla_mask = mlxcel_core::where_cond(&allowed, &zero, &neg_inf);
    let real_query = mlxcel_core::greater_equal(&q_pos, &pad); // [B, 1, L, 1]
    let token_mask = mlxcel_core::astype(&mlxcel_core::reshape(&real_query, &[b, l]), act_dtype);
    StepMasks {
        token_mask: Some(token_mask),
        mla_mask: Some(mla_mask),
    }
}

enum Mixer {
    Rda(RaonDeltaAttention),
    Mla(KimiK3MLAAttention),
}

enum Ffn {
    Dense(RaonSituMlp),
    Moe(RaonMoe),
}

impl Ffn {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Ffn::Dense(mlp) => mlp.forward(x),
            Ffn::Moe(moe) => moe.forward(x),
        }
    }
}

pub(crate) struct Raon1MiniDecoderLayer {
    layer_idx: usize,
    is_rda: bool,
    mixer: Mixer,
    ffn: Ffn,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    attn_mhc: RaonMhc,
    mlp_mhc: RaonMhc,
}

impl Raon1MiniDecoderLayer {
    fn forward(
        &self,
        streams: &MlxArray,
        cache: &mut Raon1MiniLayerCache,
        masks: &StepMasks,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_mixer(streams, |mixer, x| match (mixer, cache) {
            (Mixer::Rda(attn), Raon1MiniLayerCache::Delta(c)) => {
                attn.forward(x, c, masks.token_mask.as_deref())
            }
            (Mixer::Mla(attn), Raon1MiniLayerCache::Attn(c)) => {
                attn.forward(x, masks.mla_mask.as_deref(), c)
            }
            _ => panic!(
                "raon1_mini: cache kind mismatch at layer {}",
                self.layer_idx
            ),
        })
    }

    /// Both mHC sites and the FFN around `mixer`, which runs the token
    /// mixer (RDA or MLA) on the normalized branch input `x`. The single
    /// sequence forward and the native batched decode differ only there.
    fn forward_with_mixer(
        &self,
        streams: &MlxArray,
        mixer: impl FnOnce(&Mixer, &MlxArray) -> UniquePtr<MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let dt = mlxcel_core::array_dtype(streams);
        let route = self.attn_mhc.route(streams);
        let sf = mlxcel_core::astype(streams, dtype::FLOAT32);
        let x = self
            .input_layernorm
            .forward(&raon_mhc::aggregate(&sf, &route.h_pre, dt));
        let y = mixer(&self.mixer, &x);
        let streams = raon_mhc::mix(&sf, &y, &route, dt);

        let route = self.mlp_mhc.route(&streams);
        let sf = mlxcel_core::astype(&streams, dtype::FLOAT32);
        let x = self
            .post_attention_layernorm
            .forward(&raon_mhc::aggregate(&sf, &route.h_pre, dt));
        let y = self.ffn.forward(&x);
        raon_mhc::mix(&sf, &y, &route, dt)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Raon1MiniConfig,
        layer_idx: usize,
        act_dtype: i32,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let is_rda = config.is_rda_layer(layer_idx);
        let attn_prefix = format!("{prefix}.self_attn");
        let mixer = if is_rda {
            let spec = RdaSpec {
                hidden_size: config.hidden_size,
                num_heads: config.rda_num_heads(),
                head_dim: config.rda_head_dim(),
                kernel_size: config.rda_config.conv_kernel_size,
                gate_lower_bound: config.rda_config.gate_lower_bound,
                norm_eps: config.rms_norm_eps,
                group_size: 64,
                bits: 4,
            };
            Mixer::Rda(RaonDeltaAttention::from_weights(
                weights,
                &attn_prefix,
                spec,
            )?)
        } else {
            let spec = NopeMlaSpec {
                hidden_size: config.hidden_size,
                num_heads: config.num_attention_heads,
                q_lora_rank: (config.q_lora_rank > 0).then_some(config.q_lora_rank),
                kv_lora_rank: config.kv_lora_rank,
                qk_nope_head_dim: config.qk_nope_head_dim,
                qk_rope_head_dim: config.qk_rope_head_dim,
                v_head_dim: config.v_head_dim,
                use_output_gate: config.mla_use_output_gate,
                latent_norm_eps: config.mla_latent_norm_eps,
                group_size: 64,
                bits: 4,
                family: "raon1_mini",
            };
            Mixer::Mla(KimiK3MLAAttention::from_spec(weights, &spec, &attn_prefix)?)
        };

        let ffn = if config.is_moe_layer(layer_idx) {
            let spec = MoeSpec {
                hidden_size: config.hidden_size,
                num_experts: config.num_experts,
                top_k: config.num_experts_per_token,
                renormalize: config.moe_renormalize,
                routed_scaling_factor: config.routed_scaling_factor,
                situ_beta: config.activation_situ_beta,
                situ_linear_beta: config.activation_situ_linear_beta,
                has_shared_expert: config.num_shared_experts > 0,
            };
            Ffn::Moe(RaonMoe::from_weights(
                weights,
                &format!("{prefix}.moe"),
                spec,
            )?)
        } else {
            Ffn::Dense(RaonSituMlp::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                config.activation_situ_beta,
                config.activation_situ_linear_beta,
            )?)
        };

        let norm = |leaf: &str| -> Result<RMSNorm, String> {
            let key = format!("{prefix}.{leaf}.weight");
            let w = weights
                .get(&key)
                .ok_or_else(|| format!("Missing weight: {key}"))?;
            Ok(RMSNorm::new(mlxcel_core::copy(w), config.rms_norm_eps))
        };
        let site = |leaf: &str| {
            RaonMhc::from_weights(
                weights,
                &format!("{prefix}.{leaf}"),
                config.raon_mhc_streams,
                config.hidden_size,
                config.raon_mhc_selector_eps,
                config.raon_mhc_post_factor,
                act_dtype,
            )
        };
        Ok(Self {
            layer_idx,
            is_rda,
            mixer,
            ffn,
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            attn_mhc: site("self_attn_raon_mhc")?,
            mlp_mhc: site("mlp_raon_mhc")?,
        })
    }
}

pub struct Raon1MiniModel {
    pub embed_tokens: UnifiedEmbedding,
    layers: Vec<Raon1MiniDecoderLayer>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    streams: usize,
    act_dtype: i32,
    sequence_state: ModelOwnedSequenceState<Raon1MiniLayerCache>,
    kv_cache_layer_modes: KvCacheLayerModes,
    eos_token_ids: Vec<i32>,
}

impl Raon1MiniModel {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, Raon1MiniConfig), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config = Raon1MiniConfig::from_json_str(&config_str)?;
        tracing::info!(
            "[Raon1Mini] {} layers ({} RDA, {} MLA), {} experts top-{}, {} mHC streams",
            config.num_hidden_layers,
            config.num_hidden_layers - config.mla_layer_indices().len(),
            config.mla_layer_indices().len(),
            config.num_experts,
            config.num_experts_per_token,
            config.raon_mhc_streams
        );
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = Self::sanitize_weights(weights, &config)?;
        let mut model = Self::from_weights(&weights, &config)?;
        let from_dir = crate::loading::read_eos_token_ids(model_dir);
        if !from_dir.is_empty() {
            model.eos_token_ids = from_dir;
        }
        Ok((model, config))
    }

    /// See `raon1_mini_sanitize.rs`. Idempotent.
    pub fn sanitize_weights(
        weights: WeightMap,
        config: &Raon1MiniConfig,
    ) -> Result<WeightMap, String> {
        super::raon1_mini_sanitize::sanitize_weights(weights, config)
    }

    /// [`Self::sanitize_weights`] then [`Self::from_weights`] over an
    /// in-memory checkpoint (the config-backed weight route). The map is
    /// left holding the sanitized tensors.
    pub fn from_checkpoint_weights(
        weights: &mut WeightMap,
        config: &Raon1MiniConfig,
    ) -> Result<Self, String> {
        let sanitized = Self::sanitize_weights(std::mem::take(weights), config)?;
        *weights = sanitized;
        Self::from_weights(weights, config)
    }

    pub fn from_weights(weights: &WeightMap, config: &Raon1MiniConfig) -> Result<Self, String> {
        config.validate()?;
        let embed_key = "model.embed_tokens.weight";
        let embed = weights
            .get(embed_key)
            .ok_or_else(|| format!("Missing weight: {embed_key}"))?;
        let shape = mlxcel_core::array_shape(embed);
        if shape != [config.vocab_size as i32, config.hidden_size as i32] {
            return Err(format!(
                "{embed_key}: expected [vocab_size {}, hidden_size {}], got {shape:?}",
                config.vocab_size, config.hidden_size
            ));
        }
        let act_dtype = mlxcel_core::array_dtype(embed);
        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", 64, 4)?;
        let layers = (0..config.num_hidden_layers)
            .map(|i| Raon1MiniDecoderLayer::from_weights(weights, config, i, act_dtype))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = weights
            .get("model.norm.weight")
            .ok_or_else(|| "Missing weight: model.norm.weight".to_string())?;
        let lm_head = if weights.contains_key("lm_head.weight") {
            Some(UnifiedLinear::from_weights(weights, "lm_head", 64, 4)?)
        } else if config.tie_word_embeddings {
            None
        } else {
            return Err("Missing weight: lm_head.weight (tie_word_embeddings is false)".into());
        };
        let kv_cache_layer_modes = KvCacheLayerModes::new();
        let internal = make_layer_caches(&layers, &kv_cache_layer_modes);
        Ok(Self {
            embed_tokens,
            layers,
            norm: RMSNorm::new(mlxcel_core::copy(norm), config.rms_norm_eps),
            lm_head,
            streams: config.raon_mhc_streams,
            act_dtype,
            sequence_state: ModelOwnedSequenceState::new(internal),
            kv_cache_layer_modes,
            eos_token_ids: config.eos_token_ids(),
        })
    }

    pub fn activation_dtype(&self) -> i32 {
        self.act_dtype
    }

    pub fn make_layer_caches(&self) -> Vec<Raon1MiniLayerCache> {
        make_layer_caches(&self.layers, &self.kv_cache_layer_modes)
    }

    /// Logits for every position of `input_ids` against explicit `caches`.
    /// `left_pad` (one count per row) masks left padding out of the RDA
    /// state and the MLA window; the caller passes the same counts for
    /// every step of those rows.
    pub fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Raon1MiniLayerCache],
        left_pad: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        let masks = match left_pad {
            Some(pad) => {
                let l = mlxcel_core::array_shape(input_ids)[1];
                let offset = caches.first().map_or(0, Raon1MiniLayerCache::offset);
                padding_masks(pad, offset, l, self.act_dtype)
            }
            None => StepMasks::default(),
        };
        let h = self.embed_tokens.forward(input_ids);
        self.forward_hidden(h, caches, &masks, None)
    }

    /// Decoder stack, stream reduction, final norm and head over embeddings
    /// `h`, for every position (`row == None`) or only `row`.
    pub(crate) fn forward_hidden(
        &self,
        h: UniquePtr<MlxArray>,
        caches: &mut [Raon1MiniLayerCache],
        masks: &StepMasks,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        assert_eq!(
            caches.len(),
            self.layers.len(),
            "raon1_mini: one cache per layer"
        );
        let mut streams = raon_mhc::expand_streams(&h, self.streams);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            streams = layer.forward(&streams, cache, masks);
        }
        self.head(&streams, row)
    }

    /// Stream reduction, final norm and LM head, for every position
    /// (`row == None`) or only `row`.
    pub(crate) fn head(&self, streams: &MlxArray, row: Option<usize>) -> UniquePtr<MlxArray> {
        let h = raon_mhc::reduce_streams(streams, self.act_dtype);
        let h = mlxcel_core::generate::select_row(h, row);
        let h = self.norm.forward(&h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }

    /// Steering vectors per MoE layer, `(layer, [hidden] float32)`.
    pub fn reasoning_effort_steering(&self) -> Vec<(usize, &MlxArray)> {
        self.layers
            .iter()
            .filter_map(|layer| match &layer.ffn {
                Ffn::Moe(moe) => moe.steering_vector().map(|v| (layer.layer_idx, v)),
                Ffn::Dense(_) => None,
            })
            .collect()
    }
}

fn make_layer_caches(
    layers: &[Raon1MiniDecoderLayer],
    modes: &KvCacheLayerModes,
) -> Vec<Raon1MiniLayerCache> {
    layers
        .iter()
        .enumerate()
        .map(|(idx, layer)| {
            if layer.is_rda {
                Raon1MiniLayerCache::Delta(RaonDeltaCache::new())
            } else {
                Raon1MiniLayerCache::Attn(KVCache::new_with_mode(
                    super::mla_batched::latent_pair_cache_mode(modes.mode_for_layer(idx)),
                ))
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "raon1_mini_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "raon1_mini_block_tests.rs"]
mod block_tests;
