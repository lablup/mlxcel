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

//! Hy4 model tests over a synthetic two-layer checkpoint (issue #2354): layer
//! 0 is `full` with a dense MLP, layer 1 is `shared` with a MoE MLP, which is
//! the smallest stack that exercises every Hy4-specific path.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{HyV4Config, HyV4Model, ParityArms};

/// Dimensions the fixture varies. The quantized-latent tests need
/// `kv_lora_rank` and `index_head_dim` to be whole affine groups of 64.
#[derive(Clone, Copy)]
pub(crate) struct Tiny {
    pub(crate) index_topk: i32,
    pub(crate) kv_lora: i32,
    pub(crate) index_head_dim: i32,
    pub(crate) swiglu_limit: f32,
}

impl Tiny {
    pub(crate) const fn new(index_topk: i32) -> Self {
        Self {
            index_topk,
            kv_lora: 4,
            index_head_dim: 4,
            swiglu_limit: 10.0,
        }
    }
}

pub(crate) const HIDDEN: i32 = 6;
pub(crate) const HEADS: i32 = 2;
const VOCAB: i32 = 16;
/// Two rotation planes, so the interleaved and half-split pairings differ.
const ROPE: i32 = 4;

pub(crate) fn tiny_config_json(tiny: Tiny) -> serde_json::Value {
    serde_json::json!({
        "model_type": "hy_v4",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "q_lora_rank": 8,
        "kv_lora_rank": tiny.kv_lora,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": ROPE,
        "v_head_dim": 2,
        "index_topk": tiny.index_topk,
        "index_n_heads": 2,
        "index_head_dim": tiny.index_head_dim,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 8,
        "routed_scaling_factor": 2.0,
        "hc_mult": 2,
        "hc_eps": 1e-6,
        "hc_magnitude": 2.0,
        "indexer_types": ["full", "shared"],
        "mlp_layer_types": ["dense", "sparse"],
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "swiglu_limit": tiny.swiglu_limit,
        "eos_token_id": 7,
    })
}

pub(crate) fn tiny_config(tiny: Tiny) -> HyV4Config {
    serde_json::from_value(tiny_config_json(tiny)).expect("tiny hy_v4 config parses")
}

/// Deterministic, sign-varying weights; `key` seeds the phase so no two
/// tensors are the same.
fn varied(key: &str, shape: &[i32]) -> UniquePtr<MlxArray> {
    let seed = key.bytes().map(f32::from).sum::<f32>() * 0.071;
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| 0.45 * (0.37 * i as f32 + seed).sin())
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// Norm weights near one.
fn near_one(key: &str, shape: &[i32]) -> UniquePtr<MlxArray> {
    let base = varied(key, shape);
    mlxcel_core::add(
        &mlxcel_core::multiply_scalar(&base, 0.2),
        &mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32),
    )
}

pub(crate) fn tiny_weights(tiny: Tiny) -> WeightMap {
    let (k, i) = (tiny.kv_lora, tiny.index_head_dim);
    let mut weights = WeightMap::new();
    let mut put = |key: String, shape: &[i32]| {
        let value = if key.contains("norm") {
            near_one(&key, shape)
        } else {
            varied(&key, shape)
        };
        weights.insert(key, value);
    };

    put("model.embed_tokens.weight".into(), &[VOCAB, HIDDEN]);
    put("model.norm.weight".into(), &[HIDDEN]);
    put("lm_head.weight".into(), &[VOCAB, HIDDEN]);
    put("model.hc_head.hc_head_fn".into(), &[2, 2 * HIDDEN]);
    put("model.hc_head.hc_head_base".into(), &[2]);
    put("model.hc_head.hc_head_scale".into(), &[1]);

    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        put(format!("{p}.input_layernorm.weight"), &[HIDDEN]);
        put(format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]);
        for hc in ["hc_attn_layer", "hc_mlp_layer"] {
            put(format!("{p}.{hc}.hc_pre.hc_fn"), &[4, 2 * HIDDEN]);
            put(format!("{p}.{hc}.hc_pre.hc_base"), &[4]);
            put(format!("{p}.{hc}.hc_pre.hc_scale"), &[2]);
        }

        let a = format!("{p}.self_attn");
        put(format!("{a}.q_a_proj.weight"), &[8, HIDDEN]);
        put(format!("{a}.q_a_layernorm.weight"), &[8]);
        put(format!("{a}.q_b_proj.weight"), &[HEADS * (2 + ROPE), 8]);
        put(
            format!("{a}.kv_a_proj_with_mqa.weight"),
            &[k + ROPE, HIDDEN],
        );
        put(format!("{a}.kv_a_layernorm.weight"), &[k]);
        put(format!("{a}.embed_q.weight"), &[HEADS, k, 2]);
        put(format!("{a}.unembed_out.weight"), &[HEADS, 2, k]);
        put(format!("{a}.o_proj.weight"), &[HIDDEN, HEADS * 2]);
        put(format!("{a}.linear_gate.weight"), &[HEADS * 2, HIDDEN]);
        put(format!("{a}.learnable_sink_param"), &[HEADS]);
        if layer == 0 {
            put(format!("{a}.indexer.wk.weight"), &[i, HIDDEN]);
            put(format!("{a}.indexer.wq_b.weight"), &[2 * i, 8]);
            put(format!("{a}.indexer.weights_proj.weight"), &[2, HIDDEN]);
            put(format!("{a}.indexer.k_norm.weight"), &[i]);
            put(format!("{a}.indexer.k_norm.bias"), &[i]);
        }

        let m = format!("{p}.mlp");
        if layer == 0 {
            put(format!("{m}.gate_proj.weight"), &[16, HIDDEN]);
            put(format!("{m}.up_proj.weight"), &[16, HIDDEN]);
            put(format!("{m}.down_proj.weight"), &[HIDDEN, 16]);
        } else {
            put(format!("{m}.gate.weight"), &[4, HIDDEN]);
            put(format!("{m}.gate.e_score_correction_bias"), &[4]);
            put(format!("{m}.switch_mlp.gate_proj.weight"), &[4, 8, HIDDEN]);
            put(format!("{m}.switch_mlp.up_proj.weight"), &[4, 8, HIDDEN]);
            put(format!("{m}.switch_mlp.down_proj.weight"), &[4, HIDDEN, 8]);
            put(format!("{m}.shared_experts.gate_proj.weight"), &[8, HIDDEN]);
            put(format!("{m}.shared_experts.up_proj.weight"), &[8, HIDDEN]);
            put(format!("{m}.shared_experts.down_proj.weight"), &[HIDDEN, 8]);
        }
    }
    weights
}

pub(crate) fn tiny_model_with(tiny: Tiny, arms: ParityArms) -> HyV4Model {
    HyV4Model::from_weights_with_arms(&tiny_weights(tiny), &tiny_config(tiny), arms)
        .expect("tiny hy_v4 model loads")
}

/// Two layers with top-k 2, so any prompt longer than two tokens is sparse.
fn tiny_model() -> HyV4Model {
    tiny_model_with(Tiny::new(2), ParityArms::default())
}

pub(crate) fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

pub(crate) fn values(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::utils::array_to_vec_f32(&array)
}

pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

/// Logits of the last position after feeding `chunks` one forward each.
pub(crate) fn last_logits(
    model: &HyV4Model,
    caches: &mut [KVCache],
    chunks: &[&[i32]],
) -> Vec<f32> {
    let mut last = Vec::new();
    for chunk in chunks {
        let logits = model.forward(&ids(chunk), caches, None);
        let shape = mlxcel_core::array_shape(&logits);
        assert_eq!(shape, [1, chunk.len() as i32, VOCAB]);
        let row = mlxcel_core::slice(&logits, &[0, shape[1] - 1, 0], &[1, shape[1], VOCAB]);
        last = values(&row);
    }
    assert!(last.iter().all(|v| v.is_finite()), "non-finite logits");
    last
}

pub(crate) const PROMPT: &[i32] = &[3, 1, 4, 1, 5, 9];

fn load_error(weights: &WeightMap, config: &HyV4Config) -> String {
    HyV4Model::from_weights_with_arms(weights, config, ParityArms::default())
        .err()
        .expect("load must be refused")
}

/// `hc_*` tensors built for `hc_mult = 2` under a config that says 4.
#[test]
fn load_rejects_hc_tensors_that_disagree_with_hc_mult_times_hidden() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let tiny = Tiny::new(2);
    let mut json = tiny_config_json(tiny);
    json["hc_mult"] = 4.into();
    let config: HyV4Config = serde_json::from_value(json).expect("parses");
    let err = load_error(&tiny_weights(tiny), &config);
    assert!(err.contains("hc_attn_layer.hc_pre.hc_fn"), "{err}");
    assert!(err.contains("hc_mult * hidden_size = 24"), "{err}");
}

/// The checkpoint and `indexer_types` must agree on which layers own an
/// indexer, in both directions, and the original layout is refused by name.
#[test]
fn load_rejects_indexer_ownership_that_disagrees_with_the_checkpoint() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let tiny = Tiny::new(2);

    let mut json = tiny_config_json(tiny);
    json["indexer_types"] = serde_json::json!(["full", "full"]);
    let config: HyV4Config = serde_json::from_value(json).expect("parses");
    let err = load_error(&tiny_weights(tiny), &config);
    assert!(err.contains("indexer_types[1] is `full`"), "{err}");
    assert!(err.contains("ships no indexer weights"), "{err}");

    let mut weights = tiny_weights(tiny);
    weights.insert(
        "model.layers.1.self_attn.indexer.wk.weight".to_string(),
        varied("extra", &[4, HIDDEN]),
    );
    let err = load_error(&weights, &tiny_config(tiny));
    assert!(err.contains("indexer_types[1] is `shared`"), "{err}");

    let mut weights = tiny_weights(tiny);
    weights.insert(
        "model.layers.0.self_attn.kv_b_proj.weight".to_string(),
        varied("kv_b", &[8, 4]),
    );
    let err = load_error(&weights, &tiny_config(tiny));
    assert!(err.contains("`kv_b_proj`"), "{err}");

    let mut weights = tiny_weights(tiny);
    weights.insert(
        "model.layers.0.self_attn.learnable_sink_param".to_string(),
        varied("sinks", &[HEADS + 1]),
    );
    let err = load_error(&weights, &tiny_config(tiny));
    assert!(err.contains("learnable_sink_param"), "{err}");
    assert!(err.contains("num_attention_heads"), "{err}");
}

/// The full layer owns the indexer and caches `[kv_latent | indexer key]`;
/// the shared layer owns none and caches the latent alone.
#[test]
fn shared_layer_owns_no_indexer_and_no_indexer_key_cache() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let tiny = Tiny::new(2);
    let model = tiny_model();
    assert!(model.layers[0].self_attn.mla.indexer.is_some());
    assert!(!model.layers[0].self_attn.shares_indexer);
    assert!(model.layers[1].self_attn.mla.indexer.is_none());
    assert!(model.layers[1].self_attn.shares_indexer);

    let mut caches = model.make_caches();
    last_logits(&model, &mut caches, &[PROMPT, &[2]]);
    let key_width = |cache: &KVCache| {
        mlxcel_core::array_shape(cache.keys.as_ref().expect("cache holds keys"))[3]
    };
    assert_eq!(key_width(&caches[0]), tiny.kv_lora + tiny.index_head_dim);
    assert_eq!(key_width(&caches[1]), tiny.kv_lora);
    assert_eq!(caches[0].live_len(), PROMPT.len() as i32 + 1);
    assert_eq!(caches[1].live_len(), PROMPT.len() as i32 + 1);
}

/// The model hands the full layer's selection to the shared layer. Running
/// the two layers by hand with the selection carried over reproduces
/// `forward`; clearing it before the shared layer (dense there) does not.
#[test]
fn forward_carries_the_full_layers_selection_into_the_shared_layer() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    let expected = values(&model.forward(&ids(PROMPT), &mut model.make_caches(), None));

    let by_hand = |carry: bool| {
        let mut caches = model.make_caches();
        let h = model.embed_tokens.forward(&ids(PROMPT));
        let h = mlxcel_core::expand_dims(&h, 2);
        let shape = [1, PROMPT.len() as i32, model.hc_mult, HIDDEN];
        let h = mlxcel_core::broadcast_to(&h, &shape);
        let mut topk = None;
        let h = model.layers[0].forward(&h, None, &mut caches[0], &mut topk);
        assert!(topk.is_some(), "six keys against index_topk 2 must select");
        if !carry {
            topk = None;
        }
        let h = model.layers[1].forward(&h, None, &mut caches[1], &mut topk);
        let h = model.norm.forward(&model.hc_head.forward(&h));
        values(&model.lm_head.as_ref().expect("untied head").forward(&h))
    };
    let carried = max_abs_diff(&by_hand(true), &expected);
    assert!(carried < 1e-6, "carried selection differs by {carried}");
    let dropped = max_abs_diff(&by_hand(false), &expected);
    assert!(dropped > 1e-4, "the shared layer ignored the selection");
}

/// One prefill, a chunked prefill and prefill-then-decode agree on the last
/// position, on the sparse path (`kv_len > index_topk`, where the prefill
/// masks and the decode gathers) and on the dense fallback.
#[test]
fn prefill_chunked_prefill_and_decode_agree_on_sparse_and_dense_paths() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    for index_topk in [2, 2048] {
        let model = tiny_model_with(Tiny::new(index_topk), ParityArms::default());
        let whole = last_logits(&model, &mut model.make_caches(), &[PROMPT]);
        let chunked = last_logits(
            &model,
            &mut model.make_caches(),
            &[&PROMPT[..4], &PROMPT[4..]],
        );
        let decoded = last_logits(
            &model,
            &mut model.make_caches(),
            &[&PROMPT[..5], &PROMPT[5..]],
        );
        for (name, other) in [("chunked", &chunked), ("decoded", &decoded)] {
            let diff = max_abs_diff(&whole, other);
            assert!(diff < 2e-4, "top-k {index_topk}: {name} differs by {diff}");
        }
    }
}

/// The top-k selection is load-bearing on the tiny model: with `index_topk`
/// 2 the last position attends to two keys, with 2048 to all of them.
#[test]
fn sparse_selection_changes_the_logits() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let sparse = tiny_model();
    let dense = tiny_model_with(Tiny::new(2048), ParityArms::default());
    let a = last_logits(&sparse, &mut sparse.make_caches(), &[PROMPT]);
    let b = last_logits(&dense, &mut dense.make_caches(), &[PROMPT]);
    assert!(max_abs_diff(&a, &b) > 1e-4);
}

/// Dropping the sinks, the gate or the routed-expert clamp, or switching the
/// RoPE pairing, each changes the output, on a prefill and on a decode step.
/// This is the synthetic stand-in for the real-checkpoint parity arms: an arm
/// that left the logits unchanged would mean the term is not wired in.
#[test]
fn every_parity_arm_changes_the_logits() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    // A limit the tiny activations exceed, so the clamp bites.
    let tiny = Tiny {
        swiglu_limit: 0.05,
        ..Tiny::new(2)
    };
    let run = |arms: ParityArms| {
        let model = tiny_model_with(tiny, arms);
        let mut caches = model.make_caches();
        let prefill = last_logits(&model, &mut caches, &[PROMPT]);
        let decode = last_logits(&model, &mut caches, &[&[2]]);
        (prefill, decode)
    };
    let (base_prefill, base_decode) = run(ParityArms::default());
    let arms = [
        ("no_sinks", ParityArms::parse("no_sinks").unwrap()),
        ("no_gate", ParityArms::parse("no_gate").unwrap()),
        (
            "no_swiglu_limit",
            ParityArms::parse("no_swiglu_limit").unwrap(),
        ),
        (
            "rope_half_split",
            ParityArms::parse("rope_half_split").unwrap(),
        ),
    ];
    for (name, arm) in arms {
        let (prefill, decode) = run(arm);
        let prefill_diff = max_abs_diff(&base_prefill, &prefill);
        let decode_diff = max_abs_diff(&base_decode, &decode);
        assert!(prefill_diff > 1e-4, "{name}: prefill moved {prefill_diff}");
        assert!(decode_diff > 1e-4, "{name}: decode moved {decode_diff}");
    }
}

#[test]
fn eos_and_layer_count_come_from_the_config() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    assert_eq!(LanguageModel::num_layers(&model), 2);
    assert_eq!(LanguageModel::eos_token_ids(&model), [7]);
    assert_eq!(LanguageModel::make_caches(&model).len(), 2);
}

crate::models::last_row_logits_harness::last_row_case!(
    hy_v4_last_row_logits_match_the_full_forward,
    tiny_model
);

/// The default per-row batched decode equals sequential single-row decode,
/// with rows on both sides of the DSA threshold in one batch.
#[test]
fn hy_v4_batched_decode_matches_sequential_rows() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        &[&[1], &[2, 3, 4], &[5, 6, 7, 8, 9]],
        &[10, 11, 12],
        1e-5,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
}
