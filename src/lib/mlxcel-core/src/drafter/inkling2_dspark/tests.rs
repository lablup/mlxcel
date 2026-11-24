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

//! Tiny synthetic checkpoint checks for the Inkling 2 DSpark drafter: config
//! validation, and `mix`, the Markov step, the fold and `draft_block` against
//! an f64 host reference written from upstream `dspark.py`. Every weight is
//! f32, so upstream's BF16 roundings are identities and the device result
//! must match the reference to float tolerance.

use std::collections::HashMap;

use cxx::UniquePtr;
use serde_json::{Value, json};

use super::markov::{MarkovHead, fold_tile};
use super::model::FuseWeights;
use super::*;
use crate::cache::KVCache;
use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::{Embedding, Linear};
use crate::weights::WeightMap;

const H: usize = 8;
const HEADS: usize = 2;
const KV: usize = 1;
const D: usize = 4;
const INTER: usize = 12;
const LAYERS: usize = 2;
const RANK: usize = 2;
const VOCAB: usize = 16;
const UNPADDED: usize = 14;
const BLOCK: usize = 4;
const WINDOW: usize = 3;
const TAPS: usize = 2;
const EPS: f32 = 1e-6;
const THETA: f32 = 10000.0;
const CAP: f64 = 30.0;

fn config_json() -> Value {
    json!({
        "model_type": "inkling2",
        "text_config": {
            "hidden_size": H,
            "vocab_size": VOCAB,
            "num_hidden_layers": 3,
            "unpadded_vocab_size": UNPADDED,
            "final_logit_softcapping": CAP,
        },
        "mtp_config": {
            "hidden_size": H,
            "intermediate_size": INTER,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV,
            "head_dim": D,
            "markov_rank": RANK,
            "sliding_window_size": WINDOW,
            "rope_theta": THETA,
            "rms_norm_eps": EPS,
            "target_layer_ids": [1, 3],
            "local_layer_ids": [0],
            "block_size": BLOCK,
            "markov_head_type": "rnn_hidden",
        }
    })
}

fn config() -> Inkling2DsparkConfig {
    Inkling2DsparkConfig::from_json(&config_json()).expect("valid tiny config")
}

// ── Config validation ──────────────────────────────────────────────────────

fn refusal(edit: impl FnOnce(&mut Value)) -> String {
    let mut json = config_json();
    edit(&mut json);
    match Inkling2DsparkConfig::from_json(&json) {
        Ok(_) => panic!("config should be refused: {json}"),
        Err(message) => message,
    }
}

fn set(json: &mut Value, field: &str, value: Value) {
    json["mtp_config"][field] = value;
}

#[test]
fn the_tiny_config_parses_with_the_target_shape() {
    let config = config();
    assert_eq!(config.target.visit_count, 3);
    assert_eq!(config.target.model_type, "inkling2");
    assert_eq!(config.unpadded_vocab_size, UNPADDED);
    assert_eq!(config.final_logit_softcapping, CAP as f32);
    assert_eq!(config.default_verify_width(), BLOCK);
}

#[test]
fn a_looped_target_counts_every_visit() {
    let mut json = config_json();
    json["model_type"] = json!("inkling2_loop");
    json["text_config"]["num_hidden_layers"] = json!(4);
    json["text_config"]["loop_config"] = json!({"core_start": 1, "core_end": 3, "num_passes": 2});
    set(&mut json, "target_layer_ids", json!([1, 6]));
    let config = Inkling2DsparkConfig::from_json(&json).expect("looped config");
    assert_eq!(config.target.visit_count, 6);
    assert_eq!(config.target.loop_config, Some((1, 3, 2)));
    // A list ending at the layer count instead of the visit count is refused.
    set(&mut json, "target_layer_ids", json!([1, 4]));
    assert!(
        Inkling2DsparkConfig::from_json(&json)
            .unwrap_err()
            .contains("visit count 6")
    );
}

type Edit = Box<dyn FnOnce(&mut Value)>;

#[test]
fn each_invalid_field_is_refused_by_name() {
    let cases: Vec<(&str, Edit)> = vec![
        ("model_type", Box::new(|j| j["model_type"] = json!("llama"))),
        (
            "requires mtp_config",
            Box::new(|j| {
                j.as_object_mut().unwrap().remove("mtp_config");
            }),
        ),
        (
            "missing block_size",
            Box::new(|j| {
                j["mtp_config"]
                    .as_object_mut()
                    .unwrap()
                    .remove("block_size");
            }),
        ),
        (
            "markov_head_type",
            Box::new(|j| set(j, "markov_head_type", json!("linear"))),
        ),
        ("q_bias", Box::new(|j| set(j, "q_bias", json!(true)))),
        ("o_bias = true", Box::new(|j| set(j, "o_bias", json!(true)))),
        (
            "hidden_size 16",
            Box::new(|j| set(j, "hidden_size", json!(16))),
        ),
        ("block_size 1", Box::new(|j| set(j, "block_size", json!(1)))),
        (
            "num_hidden_layers 0",
            Box::new(|j| {
                set(j, "num_hidden_layers", json!(0));
                set(j, "local_layer_ids", json!([]));
            }),
        ),
        (
            "markov_rank 0",
            Box::new(|j| set(j, "markov_rank", json!(0))),
        ),
        (
            "not a multiple of num_key_value_heads 3",
            Box::new(|j| set(j, "num_key_value_heads", json!(3))),
        ),
        ("head_dim 3", Box::new(|j| set(j, "head_dim", json!(3)))),
        (
            "sliding_window_size 0",
            Box::new(|j| set(j, "sliding_window_size", json!(0))),
        ),
        (
            "local_layer_ids entry 2",
            Box::new(|j| set(j, "local_layer_ids", json!([2]))),
        ),
        (
            "target_layer_ids [3, 1]",
            Box::new(|j| set(j, "target_layer_ids", json!([3, 1]))),
        ),
        (
            "target_layer_ids [3]",
            Box::new(|j| set(j, "target_layer_ids", json!([3]))),
        ),
        (
            "target_layer_ids [1, 2]",
            Box::new(|j| set(j, "target_layer_ids", json!([1, 2]))),
        ),
        (
            "entries must be >= 0",
            Box::new(|j| set(j, "target_layer_ids", json!([-1, 3]))),
        ),
        (
            "rope_theta must be a number",
            Box::new(|j| set(j, "rope_theta", json!("big"))),
        ),
    ];
    for (needle, edit) in cases {
        let message = refusal(edit);
        assert!(
            message.contains(needle),
            "refusal for {needle:?} should name it, got: {message}"
        );
    }
}

#[test]
fn verify_widths_outside_the_trained_block_are_refused() {
    let config = config();
    assert!(config.verify_width_error(1).is_some());
    assert!(config.verify_width_error(2).is_none());
    assert!(config.verify_width_error(BLOCK).is_none());
    let wide = config.verify_width_error(BLOCK + 1).expect("too wide");
    assert!(wide.contains("2..=4"), "{wide}");
}

#[test]
fn directory_detection_needs_the_sidecar() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&config_json()).unwrap(),
    )
    .unwrap();
    // A plain Inkling 2 export (the 4-bit conversion): no sidecar, no drafter.
    assert!(!is_inkling2_dspark_dir(dir.path()));
    assert!(peek_configured_block_size(dir.path()).is_none());
    assert!(Inkling2DsparkConfig::from_dir(dir.path()).is_err());

    std::fs::write(dir.path().join(MTP_WEIGHTS_FILE), b"").unwrap();
    assert!(is_inkling2_dspark_dir(dir.path()));
    assert!(crate::drafter::dflash::config::is_dflash_drafter_dir(
        dir.path()
    ));
    assert_eq!(peek_configured_block_size(dir.path()), Some(BLOCK));
    assert_eq!(
        crate::drafter::resolve_drafter_kind(dir.path(), None).expect("auto kind"),
        DrafterKind::Dflash
    );
}

// ── Synthetic weights and the host reference ───────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn values(&mut self, count: usize, scale: f32, base: f32) -> Vec<f32> {
        (0..count).map(|_| base + scale * self.next()).collect()
    }
}

/// Host copies of every tensor, keyed as in `mtp.safetensors`, plus the
/// target's `embed`, `embed_norm` and `lm_head`.
struct Tensors(HashMap<String, (Vec<f32>, Vec<usize>)>);

impl Tensors {
    fn new(seed: u64) -> Self {
        let mut rng = Lcg(seed);
        let mut map = HashMap::new();
        let mut put = |name: String, shape: Vec<usize>, scale: f32, base: f32| {
            let count = shape.iter().product();
            map.insert(name, (rng.values(count, scale, base), shape));
        };
        put("fuse_weight".into(), vec![LAYERS + 1, TAPS], 1.0, 0.0);
        put("mask_embed".into(), vec![H], 1.0, 0.0);
        put("norm.weight".into(), vec![H], 0.2, 1.0);
        for l in 0..LAYERS {
            let p = format!("layers.{l}");
            for norm in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
                put(format!("{p}.{norm}.weight"), vec![H], 0.2, 1.0);
            }
            put(
                format!("{p}.mlp.gate_proj.weight"),
                vec![INTER, H],
                0.5,
                0.0,
            );
            put(format!("{p}.mlp.up_proj.weight"), vec![INTER, H], 0.5, 0.0);
            put(
                format!("{p}.mlp.down_proj.weight"),
                vec![H, INTER],
                0.5,
                0.0,
            );
            let a = format!("{p}.self_attn");
            put(format!("{a}.q_proj.weight"), vec![HEADS * D, H], 0.6, 0.0);
            for name in ["k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"] {
                put(format!("{a}.{name}.weight"), vec![KV * D, H], 0.6, 0.0);
            }
            put(format!("{a}.o_proj.weight"), vec![H, HEADS * D], 0.6, 0.0);
            for norm in ["q_norm", "k_norm", "k_norm_ctx"] {
                put(format!("{a}.{norm}.weight"), vec![D], 0.2, 1.0);
            }
        }
        let m = "markov_head";
        put(format!("{m}.dflare_norm.weight"), vec![H], 0.2, 1.0);
        put(format!("{m}.hidden_out.weight"), vec![H, RANK], 0.7, 0.0);
        put(format!("{m}.hidden_out.bias"), vec![H], 0.3, 0.0);
        put(format!("{m}.markov_w1.weight"), vec![VOCAB, RANK], 1.0, 0.0);
        put(
            format!("{m}.rnn_proj.weight"),
            vec![3 * RANK, 2 * RANK + 2 * H],
            0.6,
            0.0,
        );
        put(format!("{m}.rnn_proj.bias"), vec![3 * RANK], 0.3, 0.0);
        put("target.embed".into(), vec![VOCAB, H], 1.0, 0.0);
        put("target.embed_norm".into(), vec![H], 0.2, 1.0);
        put("target.lm_head".into(), vec![VOCAB, H], 1.5, 0.0);
        Self(map)
    }

    fn host(&self, name: &str) -> Vec<f64> {
        self.0[name].0.iter().map(|&v| f64::from(v)).collect()
    }

    fn array(&self, name: &str) -> UniquePtr<MlxArray> {
        let (data, shape) = &self.0[name];
        let shape: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        ffi::from_slice_f32(data, &shape)
    }

    fn weight_map(&self) -> WeightMap {
        self.0
            .keys()
            .filter(|name| !name.starts_with("target."))
            .map(|name| (name.clone(), self.array(name)))
            .collect()
    }

    fn binding(&self, shape: Inkling2TargetShape) -> Inkling2DsparkBinding {
        Inkling2DsparkBinding {
            embed_tokens: crate::layers::UnifiedEmbedding::Regular(Embedding::new(
                self.array("target.embed"),
            )),
            embed_norm: crate::layers::RMSNorm::new(self.array("target.embed_norm"), EPS),
            lm_head: crate::layers::UnifiedLinear::Regular(Linear::new(
                self.array("target.lm_head"),
                None,
            )),
            shape,
        }
    }
}

/// `x @ w.T` for `w` `[out, in]`.
fn linear(w: &[f64], x: &[f64]) -> Vec<f64> {
    let input = x.len();
    w.chunks_exact(input)
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn rms(x: &[f64], w: &[f64]) -> Vec<f64> {
    let mean = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean + f64::from(EPS)).sqrt();
    x.iter().zip(w).map(|(v, w)| v * inv * w).collect()
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Non-traditional RoPE over one head row at position `pos`.
fn rope(x: &[f64], pos: usize) -> Vec<f64> {
    let half = x.len() / 2;
    let mut out = x.to_vec();
    for i in 0..half {
        let freq = f64::from(THETA).powf(-(2.0 * i as f64) / x.len() as f64);
        let (sin, cos) = (pos as f64 * freq).sin_cos();
        out[i] = x[i] * cos - x[i + half] * sin;
        out[i + half] = x[i] * sin + x[i + half] * cos;
    }
    out
}

/// Per-head `[heads][D]` view of a projected row with a per-head norm.
fn head_rows(x: &[f64], norm: Option<&[f64]>) -> Vec<Vec<f64>> {
    x.chunks_exact(D)
        .map(|h| norm.map_or_else(|| h.to_vec(), |w| rms(h, w)))
        .collect()
}

/// Upstream `Model.mix` for one row of captured hidden `[TAPS * H]`.
fn host_mix(t: &Tensors, hidden_row: &[f64], row: usize, norm: &[f64]) -> Vec<f64> {
    let fuse = t.host("fuse_weight");
    let alpha = softmax(&fuse[row * TAPS..(row + 1) * TAPS]);
    let mut mixed = vec![0.0; H];
    for (tap, a) in alpha.iter().enumerate() {
        for (m, v) in mixed.iter_mut().zip(&hidden_row[tap * H..(tap + 1) * H]) {
            *m += a * v;
        }
    }
    rms(&mixed, norm)
}

/// One drafter layer's context cache: rope'd keys and values of the single
/// KV head, and the RoPE offset.
#[derive(Default)]
struct HostCache {
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    offset: usize,
}

struct HostDrafter<'a> {
    t: &'a Tensors,
    caches: Vec<HostCache>,
}

impl<'a> HostDrafter<'a> {
    fn new(t: &'a Tensors) -> Self {
        Self {
            t,
            caches: (0..LAYERS).map(|_| HostCache::default()).collect(),
        }
    }

    fn attention(&mut self, layer: usize, x: &[Vec<f64>], context: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let t = self.t;
        let a = format!("layers.{layer}.self_attn");
        let w = |name: &str| t.host(&format!("{a}.{name}"));
        let local = layer == 0;
        let cache = &mut self.caches[layer];
        let mut context = context;
        if local && context.len() > WINDOW {
            let skip = context.len() - WINDOW;
            cache.offset += skip;
            context = &context[skip..];
        }
        for (j, row) in context.iter().enumerate() {
            let key = head_rows(
                &linear(&w("k_proj_ctx.weight"), row),
                Some(&w("k_norm_ctx.weight")),
            );
            cache.keys.push(rope(&key[0], cache.offset + j));
            cache.values.push(linear(&w("v_proj_ctx.weight"), row));
        }
        cache.offset += context.len();
        if local && cache.keys.len() > WINDOW {
            let drop = cache.keys.len() - WINDOW;
            cache.keys.drain(..drop);
            cache.values.drain(..drop);
        }
        let pos = cache.offset;
        let mut keys = cache.keys.clone();
        let mut values = cache.values.clone();
        for (i, row) in x.iter().enumerate() {
            let key = head_rows(&linear(&w("k_proj.weight"), row), Some(&w("k_norm.weight")));
            keys.push(rope(&key[0], pos + i));
            values.push(linear(&w("v_proj.weight"), row));
        }
        let scale = (D as f64).powf(-0.5);
        x.iter()
            .enumerate()
            .map(|(i, row)| {
                let queries =
                    head_rows(&linear(&w("q_proj.weight"), row), Some(&w("q_norm.weight")));
                let mut out = Vec::with_capacity(HEADS * D);
                for query in &queries {
                    let q = rope(query, pos + i);
                    let scores: Vec<f64> = keys
                        .iter()
                        .map(|k| k.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>() * scale)
                        .collect();
                    let p = softmax(&scores);
                    let mut head = vec![0.0; D];
                    for (pj, v) in p.iter().zip(&values) {
                        for (h, vv) in head.iter_mut().zip(v) {
                            *h += pj * vv;
                        }
                    }
                    out.extend(head);
                }
                linear(&w("o_proj.weight"), &out)
            })
            .collect()
    }

    fn mlp(&self, layer: usize, x: &[f64]) -> Vec<f64> {
        let w = |name: &str| self.t.host(&format!("layers.{layer}.mlp.{name}.weight"));
        let gate = linear(&w("gate_proj"), x);
        let up = linear(&w("up_proj"), x);
        let hidden: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| g * sigmoid(*g) * u)
            .collect();
        linear(&w("down_proj"), &hidden)
    }

    /// Upstream `Model.draft_block`, greedy, returning the proposals and the
    /// top-1 minus top-2 logit margin of each pick.
    fn draft_block(&mut self, hidden: &[Vec<f64>], anchor: usize, block: usize) -> (Vec<i32>, f64) {
        let t = self.t;
        let embed = t.host("target.embed");
        let mut x = vec![rms(
            &embed[anchor * H..(anchor + 1) * H],
            &t.host("target.embed_norm"),
        )];
        x.extend((1..BLOCK).map(|_| t.host("mask_embed")));
        let mut residual: Option<Vec<Vec<f64>>> = None;
        for layer in 0..LAYERS {
            let norm = |name: &str| t.host(&format!("layers.{layer}.{name}.weight"));
            let context: Vec<Vec<f64>> = hidden
                .iter()
                .map(|row| host_mix(t, row, layer, &norm("dflare_norm")))
                .collect();
            let summed: Vec<Vec<f64>> = match &residual {
                None => x.clone(),
                Some(r) => x.iter().zip(r).map(|(a, b)| add(a, b)).collect(),
            };
            let normalized: Vec<Vec<f64>> = summed
                .iter()
                .map(|row| rms(row, &norm("input_layernorm")))
                .collect();
            let attended = self.attention(layer, &normalized, &context);
            let summed: Vec<Vec<f64>> = attended
                .iter()
                .zip(&summed)
                .map(|(a, b)| add(a, b))
                .collect();
            x = summed
                .iter()
                .map(|row| self.mlp(layer, &rms(row, &norm("post_attention_layernorm"))))
                .collect();
            residual = Some(summed);
        }
        let residual = residual.unwrap();
        let rows: Vec<Vec<f64>> = x
            .iter()
            .zip(&residual)
            .map(|(a, b)| rms(&add(a, b), &t.host("norm.weight")))
            .collect();
        let count = block - 1;
        let rows = &rows[1..=count];

        let lm_head = t.host("target.lm_head");
        let ho = t.host("markov_head.hidden_out.weight");
        let ho_bias = t.host("markov_head.hidden_out.bias");
        let fold_w: Vec<f64> = (0..VOCAB)
            .flat_map(|v| {
                let row = &lm_head[v * H..(v + 1) * H];
                (0..RANK)
                    .map(|r| (0..H).map(|h| row[h] * ho[h * RANK + r]).sum::<f64>())
                    .collect::<Vec<_>>()
            })
            .collect();
        let fold_b = linear(&lm_head, &ho_bias);
        let rnn = t.host("markov_head.rnn_proj.weight");
        let rnn_bias = t.host("markov_head.rnn_proj.bias");
        let width = 2 * RANK + 2 * H;
        let pick = |cols: std::ops::Range<usize>| -> Vec<f64> {
            (0..3 * RANK)
                .flat_map(|o| rnn[o * width + cols.start..o * width + cols.end].to_vec())
                .collect()
        };
        let state_w = pick(0..2 * RANK);
        let context_w = pick(2 * RANK..width);
        let f_rnn = host_mix(
            t,
            hidden.last().unwrap(),
            LAYERS,
            &t.host("markov_head.dflare_norm.weight"),
        );
        let w1 = t.host("markov_head.markov_w1.weight");

        let mut state = vec![0.0; RANK];
        let mut previous = anchor;
        let mut tokens = Vec::with_capacity(count);
        let mut margin = f64::INFINITY;
        for row in rows {
            let mut inputs = row.clone();
            inputs.extend(&f_rnn);
            let context = add(&linear(&context_w, &inputs), &rnn_bias);
            let mut inputs = state.clone();
            inputs.extend(&w1[previous * RANK..(previous + 1) * RANK]);
            let raw = add(&linear(&state_w, &inputs), &context);
            let gate: Vec<f64> = raw[..RANK].iter().map(|&v| sigmoid(v)).collect();
            let candidate: Vec<f64> = raw[RANK..2 * RANK].iter().map(|v| v.tanh()).collect();
            state = (0..RANK)
                .map(|i| gate[i] * state[i] + (1.0 - gate[i]) * candidate[i])
                .collect();
            let latent: Vec<f64> = raw[2 * RANK..].iter().map(|v| v.tanh()).collect();
            let bias = add(&linear(&fold_w, &latent), &fold_b);
            let logits: Vec<f64> = add(&linear(&lm_head, row), &bias)
                .iter()
                .map(|l| CAP * (l / CAP).tanh())
                .take(UNPADDED)
                .collect();
            let mut order: Vec<usize> = (0..UNPADDED).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            margin = margin.min(logits[order[0]] - logits[order[1]]);
            previous = order[0];
            tokens.push(previous as i32);
        }
        (tokens, margin)
    }
}

fn hidden_rows(rng: &mut Lcg, rows: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            rng.values(TAPS * H, 1.5, 0.0)
                .into_iter()
                .map(f64::from)
                .collect()
        })
        .collect()
}

fn device_rows(rows: &[Vec<f64>]) -> UniquePtr<MlxArray> {
    let flat: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
    ffi::from_slice_f32(&flat, &[1, rows.len() as i32, (TAPS * H) as i32])
}

fn tokens_of(tokens: &MlxArray) -> Vec<i32> {
    crate::utils::array_to_vec_f32(&ffi::astype(tokens, crate::dtype::FLOAT32))
        .into_iter()
        .map(|v| v as i32)
        .collect()
}

fn assert_close(device: &MlxArray, host: &[f64], what: &str) {
    let device = crate::utils::array_to_vec_f32(&ffi::astype(device, crate::dtype::FLOAT32));
    assert_eq!(device.len(), host.len(), "{what}: length");
    for (i, (d, h)) in device.iter().zip(host).enumerate() {
        assert!(
            (f64::from(*d) - h).abs() <= 1e-4 * (1.0 + h.abs()),
            "{what}[{i}]: device {d} vs host {h}"
        );
    }
}

// ── Pieces against the reference ───────────────────────────────────────────

#[test]
fn mix_matches_the_host_reference_for_every_row() {
    let t = Tensors::new(11);
    let config = config();
    let fuse = FuseWeights::new(&t.array("fuse_weight"), crate::dtype::FLOAT32, &config)
        .expect("fuse weights");
    let mut rng = Lcg(5);
    let hidden = hidden_rows(&mut rng, 3);
    let norm_w = t.host("layers.1.dflare_norm.weight");
    let norm = crate::layers::RMSNorm::new(t.array("layers.1.dflare_norm.weight"), EPS);
    for row in 0..=LAYERS {
        let device = fuse.mix(&device_rows(&hidden), row, &norm);
        let host: Vec<f64> = hidden
            .iter()
            .flat_map(|h| host_mix(&t, h, row, &norm_w))
            .collect();
        assert_close(&device, &host, &format!("mix row {row}"));
    }
    assert_eq!(fuse.last_row(), LAYERS);
}

#[test]
fn a_misshapen_fuse_weight_is_refused() {
    let config = config();
    let wrong = ffi::zeros(&[LAYERS as i32, TAPS as i32], crate::dtype::FLOAT32);
    let message = FuseWeights::new(&wrong, crate::dtype::FLOAT32, &config)
        .err()
        .expect("refused");
    assert!(message.contains("fuse_weight must be [3, 2]"), "{message}");
}

#[test]
fn fold_tile_matches_a_naive_f64_product() {
    let mut rng = Lcg(3);
    let (tile, hidden, rank) = (37, 5, 3);
    let rows = rng.values(tile * hidden, 1.0, 0.0);
    let projection: Vec<f64> = rng
        .values(hidden * rank, 1.0, 0.0)
        .into_iter()
        .map(f64::from)
        .collect();
    let bias: Vec<f64> = rng
        .values(hidden, 1.0, 0.0)
        .into_iter()
        .map(f64::from)
        .collect();
    let (weight, folded_bias) = fold_tile(&rows, &projection, &bias, hidden, rank);
    for r in 0..tile {
        let row = &rows[r * hidden..(r + 1) * hidden];
        for k in 0..rank {
            let expected: f64 = (0..hidden)
                .map(|h| f64::from(row[h]) * projection[h * rank + k])
                .sum();
            assert_eq!(weight[r * rank + k], expected as f32);
        }
        let expected: f64 = (0..hidden).map(|h| f64::from(row[h]) * bias[h]).sum();
        assert_eq!(folded_bias[r], expected as f32);
    }
}

#[test]
fn the_markov_step_matches_the_host_reference() {
    let t = Tensors::new(23);
    let mut head = MarkovHead::from_weights(&t.weight_map(), H, RANK, EPS).expect("head");
    head.fold(&crate::layers::UnifiedLinear::Regular(Linear::new(
        t.array("target.lm_head"),
        None,
    )));
    let mut rng = Lcg(9);
    let x: Vec<f64> = rng.values(H, 1.0, 0.0).into_iter().map(f64::from).collect();
    let f_rnn: Vec<f64> = rng.values(H, 1.0, 0.0).into_iter().map(f64::from).collect();
    let state: Vec<f64> = rng
        .values(RANK, 0.5, 0.0)
        .into_iter()
        .map(f64::from)
        .collect();
    let previous = 7usize;

    let to_device = |v: &[f64], shape: &[i32]| {
        let flat: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        ffi::from_slice_f32(&flat, shape)
    };
    let context = head.context(
        &to_device(&x, &[1, 1, H as i32]),
        &to_device(&f_rnn, &[1, 1, H as i32]),
    );
    let context = ffi::squeeze_axis(&context, 1);
    let (next_state, bias) = head.step(
        &to_device(&state, &[1, RANK as i32]),
        &ffi::from_slice_i32(&[previous as i32], &[1]),
        &context,
    );

    let rnn = t.host("markov_head.rnn_proj.weight");
    let width = 2 * RANK + 2 * H;
    let mut raw = t.host("markov_head.rnn_proj.bias");
    let w1 = t.host("markov_head.markov_w1.weight");
    let mut inputs = state.clone();
    inputs.extend(&w1[previous * RANK..(previous + 1) * RANK]);
    inputs.extend(&x);
    inputs.extend(&f_rnn);
    for (o, value) in raw.iter_mut().enumerate() {
        *value += (0..width)
            .map(|i| rnn[o * width + i] * inputs[i])
            .sum::<f64>();
    }
    let expected_state: Vec<f64> = (0..RANK)
        .map(|i| {
            let g = sigmoid(raw[i]);
            g * state[i] + (1.0 - g) * raw[RANK + i].tanh()
        })
        .collect();
    let latent: Vec<f64> = raw[2 * RANK..].iter().map(|v| v.tanh()).collect();
    let lm_head = t.host("target.lm_head");
    let ho = t.host("markov_head.hidden_out.weight");
    let ho_bias = t.host("markov_head.hidden_out.bias");
    // lm_head @ (hidden_out @ latent + hidden_out.bias)
    let projected = add(&linear(&ho, &latent), &ho_bias);
    let expected_bias = linear(&lm_head, &projected);
    assert_close(&next_state, &expected_state, "markov state");
    assert_close(&bias, &expected_bias, "markov logit bias");
}

// ── draft_block and binding ────────────────────────────────────────────────

struct FakeTarget {
    tensors: Tensors,
    shape: Option<Inkling2TargetShape>,
}

impl LanguageModel for FakeTarget {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        ffi::zeros(&[1], crate::dtype::FLOAT32)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        3
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0]
    }

    fn inkling2_dspark_binding(&self) -> Option<Inkling2DsparkBinding> {
        self.shape.clone().map(|shape| self.tensors.binding(shape))
    }
}

fn drafter_and_target(seed: u64) -> (Inkling2DsparkDraftModel, FakeTarget) {
    let tensors = Tensors::new(seed);
    let config = config();
    let drafter =
        Inkling2DsparkDraftModel::from_weights(config.clone(), &tensors.weight_map(), (64, 4))
            .expect("tiny drafter");
    let target = FakeTarget {
        tensors,
        shape: Some(config.target),
    };
    (drafter, target)
}

#[test]
fn draft_block_matches_the_host_reference_across_rounds() {
    let (mut drafter, target) = drafter_and_target(31);
    drafter.reset(&target).expect("bind");
    let mut host = HostDrafter::new(&target.tensors);
    let mut rng = Lcg(77);
    // Round 1 hands over more rows than the local layer's window, round 2
    // grows the window from the cache, round 3 asks for a narrower block
    // than the trained one (the end of a token budget).
    for (round, (rows, block, anchor)) in [(5, BLOCK, 3), (2, BLOCK, 9), (1, 3, 12)]
        .into_iter()
        .enumerate()
    {
        let hidden = hidden_rows(&mut rng, rows);
        let (expected, margin) = host.draft_block(&hidden, anchor, block);
        assert!(
            margin > 1e-3,
            "round {round}: the reference pick is a near tie ({margin}); change the seed"
        );
        let tokens = drafter
            .draft_block_tokens(anchor as i32, &device_rows(&hidden), block)
            .expect("draft");
        ffi::eval(&tokens);
        let got: Vec<i32> = tokens_of(&tokens);
        assert_eq!(got, expected, "round {round} proposals");
        assert!(got.iter().all(|&t| (t as usize) < UNPADDED));
    }
}

#[test]
fn reset_clears_the_drafter_context() {
    let (mut drafter, target) = drafter_and_target(31);
    let mut rng = Lcg(5);
    let first = hidden_rows(&mut rng, 4);
    let second = hidden_rows(&mut rng, 2);
    drafter.reset(&target).unwrap();
    let fresh = drafter
        .draft_block_tokens(2, &device_rows(&second), BLOCK)
        .unwrap();
    let fresh = tokens_of(&fresh);
    drafter.reset(&target).unwrap();
    drafter
        .draft_block_tokens(2, &device_rows(&first), BLOCK)
        .unwrap();
    drafter.reset(&target).unwrap();
    let again = drafter
        .draft_block_tokens(2, &device_rows(&second), BLOCK)
        .unwrap();
    assert_eq!(tokens_of(&again), fresh);
}

#[test]
fn bind_refuses_a_target_that_is_not_its_inkling2() {
    let (mut drafter, mut target) = drafter_and_target(1);
    let mut shape = target.shape.clone().unwrap();
    shape.vocab_size = 2 * VOCAB;
    target.shape = Some(shape);
    let message = match drafter.bind(&target) {
        Err(DrafterError::BindFailed { reason }) => reason,
        other => panic!("expected a bind refusal, got {other:?}"),
    };
    assert!(message.contains("vocab_size"), "{message}");
    assert!(message.contains(&VOCAB.to_string()) && message.contains(&(2 * VOCAB).to_string()));

    target.shape = None;
    let message = match drafter.validate_target_compat(&target) {
        Err(DrafterError::BindFailed { reason }) => reason,
        other => panic!("expected a bind refusal, got {other:?}"),
    };
    assert!(message.contains("matching Inkling 2 target"), "{message}");
}

#[test]
fn drafting_refuses_sampling_unbound_use_and_bad_shapes() {
    let (mut drafter, target) = drafter_and_target(1);
    let mut rng = Lcg(2);
    let hidden = device_rows(&hidden_rows(&mut rng, 2));
    let greedy = SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    };
    assert!(matches!(
        drafter.draft_block_array(1, Some(&hidden), BLOCK, &greedy),
        Err(DrafterError::DraftFailed { .. })
    ));
    drafter.reset(&target).unwrap();
    let sampling = SamplingConfig {
        temperature: 0.8,
        top_k: 0,
        ..SamplingConfig::default()
    };
    assert!(matches!(
        drafter.draft_block_array(1, Some(&hidden), BLOCK, &sampling),
        Err(DrafterError::GreedyOnly { .. })
    ));
    assert!(drafter.draft_block_array(1, None, BLOCK, &greedy).is_err());
    assert!(
        drafter
            .draft_block_array(1, Some(&hidden), BLOCK + 1, &greedy)
            .is_err()
    );
    let narrow = ffi::zeros(&[1, 2, H as i32], crate::dtype::FLOAT32);
    assert!(
        drafter
            .draft_block_array(1, Some(&narrow), BLOCK, &greedy)
            .is_err()
    );
    let proposals = drafter
        .draft_block(1, Some(&hidden), BLOCK, &greedy)
        .unwrap();
    assert_eq!(proposals.len(), BLOCK - 1);
    assert!(drafter.is_inkling2_dspark() && drafter.greedy_only());
    assert_eq!(drafter.dflash_target_layer_ids(), Some(&[1usize, 3][..]));
    assert_eq!(drafter.configured_block_size(), Some(BLOCK));
}
