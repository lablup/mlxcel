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

//! Paradigma Limite (`model_type: limite`, `LimiteForCausalLM`, issue #2360).
//!
//! Reference: `mlx_vlm/models/limite/language.py` (Blaizzy/mlx-vlm#2331) and
//! the checkpoint's own `modeling_limite.py`.
//!
//! A dense decoder whose every norm is a weightless RMSNorm with the dtype's
//! `finfo.eps`, sliding layers (`sliding_window + 1` keys) interleaved with
//! NoPE global layers, value embeddings, XSA, a per-head attention gate,
//! learned residual lambdas, a MUDD mixer on two layers that reads a learned
//! mix of earlier layer outputs, and a sigmoid-softcapped tied head whose
//! bounded values are what the sampler receives. The numerics live in
//! [`super::limite_layers`]; this file holds the config contract, the
//! loader and the model-owned mixed cache.

use crate::models::limite_layers::{
    AttnGate, Cache, Lambdas, LimiteAttention, LimiteLayer, LimiteMlp, MuddWay, PartialRope,
    ValueEmbed, Xsa, fold, mudd_combine, positions_from_offset, positions_from_offsets, rms_norm,
};
use crate::models::model_owned::KvCacheLayerModes;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::layers::{KVCache, Linear, RotatingKVCache, UnifiedEmbedding};
use mlxcel_core::utils::{
    PrefillAttention, SlidingPrefillKeys, prefill_causal_mask, prefill_sliding_window_mask,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[path = "limite_config.rs"]
mod config;
pub use config::ModelArgs;
#[path = "limite_shapes.rs"]
mod shapes;

/// The MUDD weights of one mixing layer.
struct MuddLayer {
    taps: Vec<usize>,
    attn: MuddWay,
    resid: Option<MuddWay>,
}

pub struct LimiteModel {
    embed_tokens: UnifiedEmbedding,
    value_embeds: Option<UnifiedEmbedding>,
    layers: Vec<LimiteLayer>,
    rope: PartialRope,
    /// `mudd.dense1` in the activation dtype, `[inter, hidden]`.
    mudd_dense1: Option<UniquePtr<MlxArray>>,
    mudd: BTreeMap<usize, MuddLayer>,
    /// History slots a MUDD layer reads (`hist[j]` = output of layer `j-1`).
    retained: BTreeSet<usize>,
    final_softcap: f32,
    /// Sigmoid softcap `(a, b, c)` of the head, `None` for raw logits.
    softcap: Option<(f32, f32, f32)>,
    /// Keys a sliding layer keeps: `sliding_window + 1`.
    sliding_keys: i32,
    activation_dtype: i32,
    eos_token_ids: Vec<i32>,
}

fn weight<'a>(weights: &'a WeightMap, name: &str) -> Result<&'a MlxArray, String> {
    weights
        .get(name)
        .map(|w| w.as_ref().expect("weight map holds live arrays"))
        .ok_or_else(|| format!("Limite: weight not found: {name}"))
}

fn dense(weights: &WeightMap, name: &str) -> Result<Linear, String> {
    Ok(Linear::new(
        mlxcel_core::copy(weight(weights, &format!("{name}.weight"))?),
        None,
    ))
}

fn folded(weights: &WeightMap, name: &str, scale: &MlxArray) -> Result<Linear, String> {
    Ok(Linear::new(
        fold(scale, weight(weights, &format!("{name}.weight"))?),
        None,
    ))
}

/// `t.astype(dt)`, with rows `[layer, :n]` when `row` is given.
fn rows_as(t: &MlxArray, row: Option<(i32, i32)>, dt: i32) -> UniquePtr<MlxArray> {
    let t = match row {
        Some((layer, n)) => {
            let r = mlxcel_core::utils::slice_axis(t, 0, layer, layer + 1);
            let shape = mlxcel_core::array_shape(&r);
            let r = mlxcel_core::reshape(&r, &shape[1..]);
            mlxcel_core::utils::slice_axis(&r, 0, 0, n)
        }
        None => mlxcel_core::copy(t),
    };
    mlxcel_core::astype(&t, dt)
}

impl LimiteModel {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;
        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", 64, 4)?;
        if embed_tokens.is_quantized() {
            return Err("Limite: quantized checkpoints are not supported".into());
        }
        shapes::check_shapes(weights, args)?;
        let dt = mlxcel_core::array_dtype(embed_tokens.weight());
        let value_embeds = if args.ve_layers.is_empty() {
            None
        } else {
            Some(UnifiedEmbedding::from_weights(
                weights,
                "model.value_embeds",
                64,
                4,
            )?)
        };

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(Self::layer(weights, args, i, dt)?);
        }

        let (mudd_dense1, mudd) = if args.mudd && !args.mudd_layers.is_empty() {
            let dense1 = rows_as(weight(weights, "model.mudd.dense1")?, None, dt);
            let mut mixers = BTreeMap::new();
            for (&layer, taps) in &args.tap_indices()? {
                let n = taps.len() as i32;
                let way = |w: &str, b: &str| -> Result<MuddWay, String> {
                    let at = Some((layer as i32, n));
                    Ok(MuddWay {
                        dense2: rows_as(weight(weights, w)?, at, dt),
                        bias: rows_as(weight(weights, b)?, at, dt),
                    })
                };
                let attn = way("model.mudd.dense2", "model.mudd.bias")?;
                let resid = if args.mudd_mlp {
                    Some(way("model.mudd.dense2_mlp", "model.mudd.bias_mlp")?)
                } else {
                    None
                };
                mixers.insert(
                    layer,
                    MuddLayer {
                        taps: taps.clone(),
                        attn,
                        resid,
                    },
                );
            }
            (Some(dense1), mixers)
        } else {
            (None, BTreeMap::new())
        };
        let retained = mudd.values().flat_map(|m| m.taps.iter().copied()).collect();

        let sliding_keys = crate::models::model_owned::cache_window(
            "Limite",
            args.sliding_window + 1,
            layers.iter().any(|l| !l.attn.is_global),
        )?;
        Ok(Self {
            embed_tokens,
            value_embeds,
            layers,
            rope: PartialRope::new(
                args.rope_n_pairs as i32,
                args.head_dim as i32,
                args.rope_base_local,
            ),
            mudd_dense1,
            mudd,
            retained,
            final_softcap: args.final_softcap,
            softcap: args.sigmoid_softcap()?,
            sliding_keys,
            activation_dtype: dt,
            eos_token_ids: args.eos_token_ids(),
        })
    }

    fn layer(
        weights: &WeightMap,
        args: &ModelArgs,
        i: usize,
        dt: i32,
    ) -> Result<LimiteLayer, String> {
        let p = format!("model.layers.{i}");
        let a = format!("{p}.self_attn");
        let qkv_scale = weight(weights, &format!("{a}.qkv_scale"))?;
        let o_scale = weight(weights, &format!("{a}.o_scale"))?;
        let is_global = args.global_layers.contains(&i);
        let value_embed = if args.ve_layers.contains(&i) {
            let gate = weight(weights, &format!("{a}.ve_gate"))?;
            let heads = args.ve_stored_heads.min(args.num_key_value_heads) as i32;
            let gate = mlxcel_core::utils::slice_axis(gate, 0, 0, heads);
            Some(ValueEmbed {
                gate: mlxcel_core::astype(&gate, dt),
                gate_scale: args.ve_gate_scale,
                stored_heads: args.ve_stored_heads as i32,
                ve_dim: args.ve_dim as i32,
            })
        } else {
            None
        };
        let xsa = if args.xsa && args.xsa_layers.contains(&i) {
            let alpha = weight(weights, &format!("{a}.xsa_alpha"))?;
            let alpha = mlxcel_core::tanh(&mlxcel_core::astype(alpha, dtype::FLOAT32));
            Some(Xsa {
                alpha: mlxcel_core::reshape(&alpha, &[1, 1, args.num_attention_heads as i32, 1]),
                eps: args.xsa_normalize_eps,
            })
        } else {
            None
        };
        let gate = if args.attn_gate_channels > 0 {
            let w = weight(weights, &format!("{a}.attn_gate"))?;
            Some(AttnGate {
                weight: mlxcel_core::astype(w, dt),
                scale: args.attn_gate_scale,
            })
        } else {
            None
        };
        let attn = LimiteAttention {
            q_proj: folded(weights, &format!("{a}.q_proj"), qkv_scale)?,
            k_proj: folded(weights, &format!("{a}.k_proj"), qkv_scale)?,
            v_proj: folded(weights, &format!("{a}.v_proj"), qkv_scale)?,
            o_proj: folded(weights, &format!("{a}.o_proj"), o_scale)?,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim: args.head_dim as i32,
            scale: args.attention_softmax_scale,
            uses_rope: !(args.global_nope && is_global),
            is_global,
            window: if is_global {
                0
            } else {
                (args.sliding_window + 1) as i32
            },
            value_embed,
            xsa,
            gate,
        };
        let mlp = LimiteMlp {
            gate_proj: dense(weights, &format!("{p}.mlp.gate_proj"))?,
            up_proj: dense(weights, &format!("{p}.mlp.up_proj"))?,
            down_proj: dense(weights, &format!("{p}.mlp.down_proj"))?,
        };
        // Cast once here; the per-layer `scale_by` cast is then a no-op.
        let lambda = |n: &str| -> Result<UniquePtr<MlxArray>, String> {
            Ok(mlxcel_core::astype(
                weight(weights, &format!("{p}.{n}"))?,
                dt,
            ))
        };
        Ok(LimiteLayer {
            attn,
            mlp,
            lambdas: Lambdas {
                resid_attn: lambda("resid_lambda_attn")?,
                post_attn: lambda("post_lambda_attn")?,
                resid_mlp: lambda("resid_lambda_mlp")?,
                post_mlp: lambda("post_lambda_mlp")?,
            },
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// The decoder stack over `x` (already `rms_norm(embedding)`), calling
    /// `attend(layer_index, layer, attn_in)` for each attention output.
    fn decode_stack(
        &self,
        mut x: UniquePtr<MlxArray>,
        mut attend: impl FnMut(usize, &LimiteLayer, &MlxArray) -> UniquePtr<MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut history: BTreeMap<usize, UniquePtr<MlxArray>> = BTreeMap::new();
        if self.retained.contains(&0) {
            history.insert(0, mlxcel_core::array_handle_clone(&x));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let (attn_in, base) = match (self.mudd.get(&index), &self.mudd_dense1) {
                (Some(mixer), Some(dense1)) => {
                    let values: Vec<&MlxArray> = mixer
                        .taps
                        .iter()
                        .map(|t| {
                            history
                                .get(t)
                                .map(|h| h.as_ref().expect("live history"))
                                .expect("validated MUDD tap is retained")
                        })
                        .collect();
                    let attn_in = rms_norm(&mudd_combine(&values, &x, dense1, &mixer.attn));
                    let base = mixer
                        .resid
                        .as_ref()
                        .map(|way| mudd_combine(&values, &x, dense1, way));
                    (attn_in, base)
                }
                _ => (rms_norm(&x), None),
            };
            let attn_out = attend(index, layer, &attn_in);
            x = layer.residual(base.as_deref().unwrap_or(&*x), &attn_out);
            if self.retained.contains(&(index + 1)) {
                history.insert(index + 1, mlxcel_core::array_handle_clone(&x));
            }
        }
        if self.final_softcap > 0.0 {
            let cap = self.final_softcap;
            let dt = mlxcel_core::array_dtype(&x);
            let s = crate::models::limite_layers::scalar(cap, dt);
            x = mlxcel_core::multiply(&s, &mlxcel_core::tanh(&mlxcel_core::divide(&x, &s)));
        }
        x
    }

    /// Final norm and the sigmoid-softcapped tied head, in f32.
    fn head(&self, h: UniquePtr<MlxArray>, row: Option<usize>) -> UniquePtr<MlxArray> {
        let h = rms_norm(&mlxcel_core::generate::select_row(h, row));
        let raw = self.embed_tokens.as_linear(&h);
        match self.softcap {
            Some((a, b, c)) => {
                let f = |v| crate::models::limite_layers::scalar(v, dtype::FLOAT32);
                let raw = mlxcel_core::astype(&raw, dtype::FLOAT32);
                let z = mlxcel_core::divide(&mlxcel_core::add(&raw, &f(b)), &f(c));
                mlxcel_core::multiply(&f(a), &mlxcel_core::sigmoid(&z))
            }
            None => raw,
        }
    }

    fn embed(&self, input_ids: &MlxArray) -> (UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>) {
        let x = rms_norm(&self.embed_tokens.forward(input_ids));
        let ve = self.value_embeds.as_ref().map(|e| e.forward(input_ids));
        (x, ve)
    }

    /// Single-row forward over `caches`, projecting every position
    /// (`row == None`) or only `row` (the prefill's sampled row, #3070).
    pub(crate) fn forward_rows(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let (batch, seq_len) = (shape[0], shape[1]);
        let attn = &self.layers[0].attn;
        let attention = PrefillAttention::new(
            attn.num_heads as usize,
            attn.num_kv_heads as usize,
            attn.head_dim as usize,
        );
        let dt = self.activation_dtype;
        let first = |global: bool| self.layers.iter().position(|l| l.attn.is_global == global);
        let mask_full = first(true)
            .and_then(|i| prefill_causal_mask(seq_len, caches[i].offset(), batch, attention, dt));
        let mask_sliding = first(false).and_then(|i| {
            prefill_sliding_window_mask(
                seq_len,
                caches[i].offset(),
                self.sliding_keys,
                batch,
                attention,
                dt,
                SlidingPrefillKeys::Rotating,
            )
        });
        let rope = self
            .rope
            .tables(&positions_from_offset(caches[0].offset(), seq_len));
        let (x, ve) = self.embed(input_ids);
        let h = self.decode_stack(x, |i, layer, attn_in| {
            let mask = if layer.attn.is_global {
                mask_full.as_deref()
            } else {
                mask_sliding.as_deref()
            };
            layer
                .attn
                .forward(attn_in, ve.as_deref(), &rope, &mut caches[i], mask)
        });
        self.head(h, row)
    }

    /// Width-one decode of `B` rows, each over its own cache set.
    fn forward_batched_decode(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Vec<Cache>],
    ) -> UniquePtr<MlxArray> {
        let offsets: Vec<i32> = caches.iter().map(|row| row[0].offset()).collect();
        let rope = self.rope.tables(&positions_from_offsets(&offsets));
        let (x, ve) = self.embed(input_ids);
        let h = self.decode_stack(x, |i, layer, attn_in| {
            let mut rows: Vec<&mut Cache> = caches.iter_mut().map(|row| &mut row[i]).collect();
            layer
                .attn
                .forward_batched_decode(attn_in, ve.as_deref(), &rope, &mut rows)
        });
        self.head(h, None)
    }

    /// Per-layer caches in the resolved KV cache modes; `None` keeps FP16.
    pub(crate) fn make_caches_with_modes(&self, modes: Option<&KvCacheLayerModes>) -> Vec<Cache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                if layer.attn.is_global {
                    let mode = modes.map_or(KVCacheMode::Fp16, |m| m.mode_for_layer(i));
                    Cache::Standard(KVCache::new_with_mode(mode))
                } else {
                    let window = self.sliding_keys;
                    let mode =
                        modes.map_or(KVCacheMode::Fp16, |m| m.rotating_mode_for_layer(i, window));
                    Cache::Rotating(RotatingKVCache::new_with_mode(window, mode))
                }
            })
            .collect()
    }
}

#[path = "limite_wrapper.rs"]
mod wrapper;
pub use wrapper::LimiteWrapper;
