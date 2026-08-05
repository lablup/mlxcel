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

//! Unit tests for the DBRX config surface, its weight-key contract, and the two
//! behaviors that no shape assertion can catch: the QKV clamp and the
//! norm-attn-norm residual routing.
//!
//! Everything here is checkpoint-free. The config test parses the real
//! `mlx-community/dbrx-instruct-4bit` field set (including the
//! `PretrainedConfig` noise that upstream dumps into `attn_config` /
//! `ffn_config`), and the model tests build tiny synthetic weight maps whose key
//! names mirror the checkpoint.

use super::{Attention, DbrxModel, ModelArgs};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// Config surface.

/// The `mlx-community/dbrx-instruct-4bit` config. The nested blocks are
/// reproduced with their real `PretrainedConfig` clutter, because that clutter
/// is exactly what a strict deserializer would choke on.
const DBRX_INSTRUCT_CONFIG: &str = r#"{
    "architectures": ["DbrxForCausalLM"],
    "attn_config": {
        "return_dict": true,
        "output_hidden_states": false,
        "torch_dtype": null,
        "tie_word_embeddings": true,
        "max_length": 20,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "id2label": { "0": "LABEL_0", "1": "LABEL_1" },
        "model_type": "",
        "attn_pdrop": 0,
        "clip_qkv": 8,
        "kv_n_heads": 8,
        "rope_theta": 500000
    },
    "d_model": 6144,
    "emb_pdrop": 0.0,
    "ffn_config": {
        "return_dict": true,
        "tie_word_embeddings": true,
        "temperature": 1.0,
        "model_type": "",
        "ffn_act_fn": { "name": "silu" },
        "ffn_hidden_size": 10752,
        "moe_num_experts": 16,
        "moe_top_k": 4,
        "moe_jitter_eps": 0,
        "moe_loss_weight": 0.05,
        "moe_normalize_expert_weights": 1,
        "uniform_expert_assignment": false
    },
    "initializer_range": 0.02,
    "max_seq_len": 32768,
    "model_type": "dbrx",
    "n_heads": 48,
    "n_layers": 40,
    "output_router_logits": false,
    "quantization": { "group_size": 64, "bits": 4 },
    "resid_pdrop": 0.0,
    "router_aux_loss_coef": 0.05,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.39.2",
    "use_cache": true,
    "vocab_size": 100352
}"#;

fn dbrx_instruct() -> ModelArgs {
    serde_json::from_str(DBRX_INSTRUCT_CONFIG).expect("the real config must parse")
}

#[test]
fn the_real_config_parses_including_the_nested_blocks() {
    let args = dbrx_instruct();
    assert_eq!(args.model_type, "dbrx");
    assert_eq!(args.d_model, 6144);
    assert_eq!(args.n_heads, 48);
    assert_eq!(args.n_layers, 40);
    assert_eq!(args.vocab_size, 100_352);

    // Geometry lives one level down, not at the top level like the Llama family.
    assert_eq!(args.num_kv_heads(), 8);
    assert_eq!(args.rope_theta(), 500_000.0);
    assert_eq!(args.ffn_hidden_size(), 10_752);
    assert_eq!(args.moe_num_experts(), 16);
    assert_eq!(args.moe_top_k(), 4);

    // 6144 / 48. GQA groups 48 query heads onto 8 KV heads.
    assert_eq!(args.head_dim(), 128);

    assert!(!args.tie_word_embeddings);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

#[test]
fn clip_qkv_parses_from_a_json_integer() {
    // Upstream writes `"clip_qkv": 8`, not `8.0`. Declaring the field as a float
    // still has to accept the integer spelling.
    let args = dbrx_instruct();
    assert_eq!(args.clip_qkv(), Some(8.0));
}

#[test]
fn an_absent_clip_qkv_is_none_rather_than_zero() {
    // A defaulted `f32` would make this 0.0, which clamps every activation to
    // zero and produces a model that loads and emits pure noise. The distinction
    // between "no clipping" and "clip to zero" has to survive parsing.
    let cfg = r#"{
        "model_type": "dbrx",
        "vocab_size": 100352,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 2,
        "attn_config": { "kv_n_heads": 2, "rope_theta": 500000 },
        "ffn_config": { "ffn_hidden_size": 128, "moe_num_experts": 4, "moe_top_k": 2 }
    }"#;
    let args: ModelArgs = serde_json::from_str(cfg).expect("config without clip_qkv must parse");
    assert_eq!(args.clip_qkv(), None);

    let explicit_null = cfg.replace(r#""kv_n_heads": 2"#, r#""kv_n_heads": 2, "clip_qkv": null"#);
    let args: ModelArgs =
        serde_json::from_str(&explicit_null).expect("an explicit null must parse too");
    assert_eq!(args.clip_qkv(), None);
}

#[test]
fn a_null_eos_token_id_falls_back_to_the_tokenizer_id() {
    // Published DBRX configs carry no eos id at all; it lives only in
    // `tokenizer_config.json` as `<|endoftext|>` = 100257. Returning an empty
    // list here would leave generation with no stop condition.
    let args = dbrx_instruct();
    assert_eq!(args.eos_token_ids(), vec![100_257]);

    let with_eos = DBRX_INSTRUCT_CONFIG.replace(
        r#""model_type": "dbrx","#,
        r#""model_type": "dbrx", "eos_token_id": 7,"#,
    );
    let args: ModelArgs = serde_json::from_str(&with_eos).expect("must parse");
    assert_eq!(args.eos_token_ids(), vec![7]);

    let with_list = DBRX_INSTRUCT_CONFIG.replace(
        r#""model_type": "dbrx","#,
        r#""model_type": "dbrx", "eos_token_id": [7, 9],"#,
    );
    let args: ModelArgs = serde_json::from_str(&with_list).expect("the list form must parse");
    assert_eq!(args.eos_token_ids(), vec![7, 9]);
}

// Model construction and forward.

fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
        "model_type": "dbrx",
        "vocab_size": 64,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "attn_config": { "kv_n_heads": 2, "clip_qkv": 8, "rope_theta": 500000 },
        "ffn_config": { "ffn_hidden_size": 16, "moe_num_experts": 4, "moe_top_k": 2 },
        "tie_word_embeddings": false
    }"#,
    )
    .expect("the tiny config must parse")
}

fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let d = args.d_model as i32;
    let ff = args.ffn_hidden_size() as i32;
    let vocab = args.vocab_size as i32;
    let head_dim = args.head_dim() as i32;
    let qkv_out = (args.n_heads as i32 + 2 * args.num_kv_heads() as i32) * head_dim;

    let mut w = WeightMap::new();
    w.insert("transformer.wte.weight".into(), filled(&[vocab, d]));
    w.insert("transformer.norm_f.weight".into(), ones(&[d]));
    w.insert("lm_head.weight".into(), filled(&[vocab, d]));

    for layer in 0..args.n_layers {
        let p = format!("transformer.blocks.{layer}");
        w.insert(format!("{p}.norm_attn_norm.norm_1.weight"), ones(&[d]));
        w.insert(format!("{p}.norm_attn_norm.norm_2.weight"), ones(&[d]));
        w.insert(
            format!("{p}.norm_attn_norm.attn.Wqkv.weight"),
            filled(&[qkv_out, d]),
        );
        w.insert(
            format!("{p}.norm_attn_norm.attn.out_proj.weight"),
            filled(&[d, d]),
        );
        w.insert(
            format!("{p}.ffn.router.layer.weight"),
            filled(&[args.moe_num_experts() as i32, d]),
        );
        for e in 0..args.moe_num_experts() {
            let ep = format!("{p}.ffn.experts.{e}");
            // gate=w1 and up=v1 project d -> ff; down=w2 projects ff -> d.
            w.insert(format!("{ep}.w1.weight"), filled(&[ff, d]));
            w.insert(format!("{ep}.v1.weight"), filled(&[ff, d]));
            w.insert(format!("{ep}.w2.weight"), filled(&[d, ff]));
        }
    }
    w
}

#[test]
fn from_weights_builds_the_full_stack_from_checkpoint_key_names() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = DbrxModel::from_weights(&weights, &args).expect("must build");

    assert_eq!(model.num_layers(), args.n_layers);
    assert_eq!(model.blocks.len(), 2);
    assert!(
        model.lm_head.is_some(),
        "tie_word_embeddings is false, so a separate lm_head must load"
    );
}

#[test]
fn a_missing_expert_projection_is_an_error_not_a_silent_skip() {
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    // Drop the very first expert's up projection. The per-expert stacker walks
    // contiguously from index 0, so losing expert 0 must fail loudly.
    weights.remove("transformer.blocks.0.ffn.experts.0.v1.weight");
    assert!(DbrxModel::from_weights(&weights, &args).is_err());
}

#[test]
fn forward_produces_vocab_sized_logits_and_advances_the_cache() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = DbrxModel::from_weights(&weights, &args).expect("must build");

    let mut caches = model.make_caches();
    assert_eq!(caches.len(), args.n_layers);

    let tokens = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);
    let logits = LanguageModel::forward(&model, &tokens, &mut caches, None);
    mlxcel_core::eval(&logits);

    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 4, args.vocab_size as i32]
    );
    assert_eq!(caches[0].offset, 4, "prefill must advance the KV cache");

    // A follow-on single token continues from the prefill offset.
    let next = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
    let logits = LanguageModel::forward(&model, &next, &mut caches, None);
    mlxcel_core::eval(&logits);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 1, args.vocab_size as i32]
    );
    assert_eq!(caches[0].offset, 5);
}

#[test]
fn eos_ids_reach_the_built_model() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = DbrxModel::from_weights(&weights, &args).expect("must build");
    assert_eq!(LanguageModel::eos_token_ids(&model), vec![100_257]);
}

// The two behaviors that shapes cannot distinguish.

#[test]
fn clip_qkv_actually_clamps_the_projection() {
    // A dropped clamp changes no shape and still produces fluent-looking output,
    // so the only way to see it is to feed the projection past the limit and
    // check the attention output changes when the limit changes.
    //
    // Build one attention block with a tight clamp and an identical one with no
    // clamp; the same input must produce different results.
    let args = tiny_args();
    let weights = tiny_weights(&args);

    let clipped =
        Attention::from_weights(&weights, &args, "transformer.blocks.0.norm_attn_norm.attn")
            .expect("must build");

    let mut unclipped_args = tiny_args();
    unclipped_args.attn_config.clip_qkv = None;
    let unclipped = Attention::from_weights(
        &weights,
        &unclipped_args,
        "transformer.blocks.0.norm_attn_norm.attn",
    )
    .expect("must build");

    assert_eq!(clipped.clip_qkv, Some(8.0));
    assert_eq!(unclipped.clip_qkv, None);

    // Large inputs push the fused projection well past +/-8.
    let big: Vec<f32> = (0..(args.d_model as usize))
        .map(|i| 40.0 + i as f32)
        .collect();
    let x = mlxcel_core::from_slice_f32(&big, &[1, 1, args.d_model as i32]);

    let mut cache_a = KVCache::new();
    let a = clipped.forward(&x, &mut cache_a, None);
    let mut cache_b = KVCache::new();
    let b = unclipped.forward(&x, &mut cache_b, None);

    let diff = mlxcel_core::sum_all(&mlxcel_core::abs(&mlxcel_core::subtract(&a, &b)));
    mlxcel_core::eval(&diff);
    assert!(
        mlxcel_core::item_f32(&diff) > 1e-3,
        "clamping the QKV projection must change the attention output; \
         an unclamped path would be numerically identical"
    );
}

#[test]
fn the_second_norm_feeds_the_ffn_but_the_unnormalized_residual_is_added_back() {
    // norm-attn-norm routes `norm_2(residual)` into the FFN while adding the FFN
    // output to `residual` itself. Wiring the FFN to add back its own normalized
    // input instead is a plausible mistake that keeps every shape intact.
    //
    // Detect it structurally: scale `norm_2` far away from unity. If the block
    // added back the normalized tensor, the output would track that scaling; it
    // must not, because only the FFN input is normalized.
    let args = tiny_args();

    let mut weights = tiny_weights(&args);
    let model_a = DbrxModel::from_weights(&weights, &args).expect("must build");
    let tokens = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let mut caches = model_a.make_caches();
    let out_a = LanguageModel::forward(&model_a, &tokens, &mut caches, None);
    mlxcel_core::eval(&out_a);

    // Zero the second norm's weight. That makes the FFN input identically zero,
    // so the FFN contributes only its bias-free projection of zero (i.e. zero),
    // and the block output must collapse to the attention residual alone.
    let d = args.d_model as i32;
    for layer in 0..args.n_layers {
        weights.insert(
            format!("transformer.blocks.{layer}.norm_attn_norm.norm_2.weight"),
            mlxcel_core::from_slice_f32(&vec![0.0; d as usize], &[d]),
        );
    }
    let model_b = DbrxModel::from_weights(&weights, &args).expect("must build");
    let mut caches = model_b.make_caches();
    let out_b = LanguageModel::forward(&model_b, &tokens, &mut caches, None);
    mlxcel_core::eval(&out_b);

    let diff = mlxcel_core::sum_all(&mlxcel_core::abs(&mlxcel_core::subtract(&out_a, &out_b)));
    mlxcel_core::eval(&diff);
    assert!(
        mlxcel_core::item_f32(&diff) > 1e-3,
        "norm_2 must gate the FFN input; if it did not, zeroing it would not \
         change the block output"
    );
}

// Fixtures.

fn filled(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state: u32 = 0x9E37_79B9;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}
