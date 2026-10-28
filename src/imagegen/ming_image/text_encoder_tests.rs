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

//! The conditioner's layout and shapes on a tiny synthetic checkpoint, and
//! its templated ids against upstream's on the real tokenizer.

use serde_json::json;

use super::*;
use crate::imagegen::ming_image::config::{bailing_args, vae_config};
use crate::imagegen::ming_image::parity_tests::{model_dir, parity};
use crate::imagegen::testing::{det, values};
use crate::imagegen::z_image::config::ZImageTransformerConfig;

const TOKENS: BlockTokens = BlockTokens {
    patch: 157_157,
    start: 157_158,
    end: 157_159,
};
const HIDDEN: i32 = 32;
const CONNECTOR: i32 = 24;
const CAP_DIM: i32 = 16;
const DIT_DIM: i32 = 20;
const QUERY: i32 = 256;

#[test]
fn the_query_span_and_text_positions_follow_upstream() {
    let ids = [9, 8, TOKENS.start, TOKENS.patch, TOKENS.patch, TOKENS.end];
    let (start, text) = query_layout(&ids, TOKENS, 2).unwrap();
    assert_eq!(start, 3);
    assert_eq!(text, [0, 1]);
    // No span, a short span, or a split span is an error.
    assert!(query_layout(&[1, 2, 3], TOKENS, 2).is_err());
    assert!(query_layout(&ids, TOKENS, 3).is_err());
    let split = [TOKENS.patch, 4, TOKENS.patch];
    assert!(query_layout(&split, TOKENS, 2).is_err());
}

fn tiny_config() -> MingImageConfig {
    let mllm = bailing_args(&json!({
        "hidden_size": HIDDEN,
        "intermediate_size": 40,
        "moe_intermediate_size": 16,
        "num_experts": 8,
        "num_shared_experts": 1,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 3,
        "num_key_value_heads": 2,
        "vocab_size": 157_184,
        "n_group": 2,
        "topk_group": 1
    }))
    .unwrap();
    let connector = serde_json::from_value(json!({
        "model_type": "qwen2",
        "hidden_size": CONNECTOR,
        "num_hidden_layers": 2,
        "intermediate_size": 40,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "vocab_size": 64
    }))
    .unwrap();
    MingImageConfig {
        dit: ZImageTransformerConfig {
            hidden_size: DIT_DIM,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            in_channels: 16,
            text_embed_dim: CAP_DIM,
            num_hidden_layers: 1,
            n_refiner_layers: 1,
            n_context_refiner_layers: 1,
            patch_size: 2,
            f_patch_size: 1,
            adaln_embed_dim: 8,
            rope_sections: vec![2, 4, 4],
            rope_theta: 256.0,
            norm_eps: 1e-5,
            timestep_scale: 1000.0,
        },
        bridge: BridgeConfig {
            query_token_count: QUERY,
            cap_feat_dim: CAP_DIM,
            directvlm_dim: DIT_DIM,
            selected_layers: vec![1, 2, 3],
        },
        vae: vae_config(&json!({
            "base_dim": 96,
            "dim_mult": [1, 2, 4, 4],
            "input_channels": 4,
            "num_res_blocks": 2,
            "temperal_downsample": [false, true, true],
            "z_dim": 16
        }))
        .unwrap(),
        mllm,
        connector,
        tokens: TOKENS,
    }
}

/// Raw-layout tiny weights: per-expert MoE tensors with both routers, a
/// Qwen2 connector with q/k/v biases, and the bridge.
fn tiny_weights(config: &MingImageConfig) -> TextEncoderWeights {
    let a = &config.mllm;
    let hidden = a.hidden_size as i32;
    let experts = a.num_experts() as i32;
    let moe_ff = a.moe_intermediate_size() as i32;
    let head_dim = a.head_dim() as i32;
    let mut mllm = WeightMap::new();
    let put = |map: &mut WeightMap, k: String, shape: &[i32], norm: bool| {
        let fan = *shape.last().unwrap();
        let v = if norm {
            det(&k, shape, 0.2, 1.0)
        } else if shape.len() == 1 {
            det(&k, shape, 0.1, 0.0)
        } else {
            det(&k, shape, 0.8 / f64::from(fan).sqrt(), 0.0)
        };
        map.insert(k, mlxcel_core::astype(&v, dtype::BFLOAT16));
    };
    put(
        &mut mllm,
        "model.word_embeddings.weight".into(),
        &[a.vocab_size as i32, hidden],
        false,
    );
    put(&mut mllm, "model.norm.weight".into(), &[hidden], true);
    for layer in 0..a.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            put(&mut mllm, format!("{p}.{norm}.weight"), &[hidden], true);
        }
        for norm in ["query_layernorm", "key_layernorm"] {
            put(
                &mut mllm,
                format!("{p}.attention.{norm}.weight"),
                &[head_dim],
                true,
            );
        }
        put(
            &mut mllm,
            format!("{p}.attention.query_key_value.weight"),
            &[a.qkv_out_features() as i32, hidden],
            false,
        );
        put(
            &mut mllm,
            format!("{p}.attention.dense.weight"),
            &[hidden, hidden],
            false,
        );
        let mlp = format!("{p}.mlp");
        if !a.is_moe_layer(layer) {
            let ff = a.intermediate_size as i32;
            put(
                &mut mllm,
                format!("{mlp}.gate_proj.weight"),
                &[ff, hidden],
                false,
            );
            put(
                &mut mllm,
                format!("{mlp}.up_proj.weight"),
                &[ff, hidden],
                false,
            );
            put(
                &mut mllm,
                format!("{mlp}.down_proj.weight"),
                &[hidden, ff],
                false,
            );
            continue;
        }
        for router in ["gate", "image_gate"] {
            put(
                &mut mllm,
                format!("{mlp}.{router}.weight"),
                &[experts, hidden],
                false,
            );
            put(
                &mut mllm,
                format!("{mlp}.{router}.expert_bias"),
                &[experts],
                false,
            );
        }
        for e in 0..experts {
            let ex = format!("{mlp}.experts.{e}");
            put(
                &mut mllm,
                format!("{ex}.gate_proj.weight"),
                &[moe_ff, hidden],
                false,
            );
            put(
                &mut mllm,
                format!("{ex}.up_proj.weight"),
                &[moe_ff, hidden],
                false,
            );
            put(
                &mut mllm,
                format!("{ex}.down_proj.weight"),
                &[hidden, moe_ff],
                false,
            );
        }
        let s = format!("{mlp}.shared_experts");
        put(
            &mut mllm,
            format!("{s}.gate_proj.weight"),
            &[moe_ff, hidden],
            false,
        );
        put(
            &mut mllm,
            format!("{s}.up_proj.weight"),
            &[moe_ff, hidden],
            false,
        );
        put(
            &mut mllm,
            format!("{s}.down_proj.weight"),
            &[hidden, moe_ff],
            false,
        );
    }

    let c = &config.connector;
    let (d, ff) = (c.hidden_size as i32, c.intermediate_size as i32);
    let kv = (c.num_key_value_heads.unwrap() * (c.hidden_size / c.num_attention_heads)) as i32;
    let mut connector = WeightMap::new();
    for layer in 0..c.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        for (proj, out) in [("q_proj", d), ("k_proj", kv), ("v_proj", kv)] {
            put(
                &mut connector,
                format!("{p}.self_attn.{proj}.weight"),
                &[out, d],
                false,
            );
            put(
                &mut connector,
                format!("{p}.self_attn.{proj}.bias"),
                &[out],
                false,
            );
        }
        put(
            &mut connector,
            format!("{p}.self_attn.o_proj.weight"),
            &[d, d],
            false,
        );
        put(
            &mut connector,
            format!("{p}.mlp.gate_proj.weight"),
            &[ff, d],
            false,
        );
        put(
            &mut connector,
            format!("{p}.mlp.up_proj.weight"),
            &[ff, d],
            false,
        );
        put(
            &mut connector,
            format!("{p}.mlp.down_proj.weight"),
            &[d, ff],
            false,
        );
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            put(&mut connector, format!("{p}.{norm}.weight"), &[d], true);
        }
    }
    put(&mut connector, "model.norm.weight".into(), &[d], true);

    let b = &config.bridge;
    let wide = HIDDEN * b.selected_layers.len() as i32;
    let mut bridge = WeightMap::new();
    put(
        &mut bridge,
        "query_tokens".into(),
        &[b.query_token_count, hidden],
        false,
    );
    put(&mut bridge, "proj_in.weight".into(), &[d, hidden], false);
    put(&mut bridge, "proj_in.bias".into(), &[d], false);
    put(
        &mut bridge,
        "proj_out.weight".into(),
        &[b.cap_feat_dim, d],
        false,
    );
    put(
        &mut bridge,
        "proj_out.bias".into(),
        &[b.cap_feat_dim],
        false,
    );
    put(&mut bridge, "directvlm_norm.weight".into(), &[wide], true);
    put(
        &mut bridge,
        "directvlm_proj.weight".into(),
        &[b.directvlm_dim, wide],
        false,
    );
    put(
        &mut bridge,
        "directvlm_proj.bias".into(),
        &[b.directvlm_dim],
        false,
    );
    TextEncoderWeights {
        mllm,
        connector,
        bridge,
        quant: None,
    }
}

/// The ids upstream's `_tokenize` produced for the poster prompt.
fn fixture_ids() -> Vec<i32> {
    values(&parity()["gen.input_ids"])
        .into_iter()
        .map(|v| v as i32)
        .collect()
}

#[test]
fn caption_streams_have_the_query_and_text_shapes_for_the_fixture_prompt() {
    let config = tiny_config();
    let encoder = TextEncoder::build(None, &config, tiny_weights(&config)).unwrap();
    let ids = fixture_ids();
    let text_len = ids.len() as i32 - QUERY - 2;
    assert!(text_len > 0);
    let states = encoder.encode_ids(&ids).unwrap();
    let e = &states.encoded;
    assert_eq!(mlxcel_core::array_shape(&e.cap_feats), [1, QUERY, CAP_DIM]);
    assert_eq!(
        mlxcel_core::array_shape(&e.cap_feats_2),
        [1, text_len, DIT_DIM]
    );
    assert_eq!(mlxcel_core::array_dtype(&e.cap_feats), dtype::BFLOAT16);
    assert_eq!(mlxcel_core::array_dtype(&e.cap_feats_2), dtype::BFLOAT16);
    assert_eq!(
        mlxcel_core::array_shape(&states.query_hidden),
        [1, QUERY, HIDDEN]
    );
    assert_eq!(
        mlxcel_core::array_shape(&states.text_states),
        [1, text_len, 3 * HIDDEN]
    );
    assert!(values(&e.cap_feats).iter().all(|v| v.is_finite()));
    assert!(
        encoder.encode("a prompt").is_err(),
        "no tokenizer was given"
    );
}

#[test]
fn the_query_span_reads_the_learned_tokens_not_the_patch_embedding() {
    let config = tiny_config();
    let ids = fixture_ids();
    let encode = |w: TextEncoderWeights| {
        let e = TextEncoder::build(None, &config, w).unwrap();
        values(&e.encode_ids(&ids).unwrap().encoded.cap_feats)
    };
    let base = encode(tiny_weights(&config));

    // Another embedding row for the patch token changes nothing: the span's
    // inputs are the learned query tokens.
    let mut w = tiny_weights(&config);
    let table = w.mllm.remove("model.word_embeddings.weight").unwrap();
    let rows = mlxcel_core::array_shape(&table)[0];
    let mut scale = vec![1.0_f32; rows as usize];
    scale[TOKENS.patch as usize] = -3.0;
    let scale = mlxcel_core::from_slice_f32(&scale, &[rows, 1]);
    let table = mlxcel_core::astype(&mlxcel_core::multiply(&table, &scale), dtype::BFLOAT16);
    w.mllm.insert("model.word_embeddings.weight".into(), table);
    assert_eq!(encode(w), base);

    // Other learned query tokens move cap_feats.
    let mut w = tiny_weights(&config);
    let q = w.bridge.remove("query_tokens").unwrap();
    w.bridge
        .insert("query_tokens".into(), mlxcel_core::negative(&q));
    assert_ne!(encode(w), base);
}

#[test]
fn templated_ids_match_upstream_on_the_real_tokenizer() {
    let Some(root) = model_dir("ming-image-0.1-design") else {
        return;
    };
    let config = MingImageConfig::from_model_dir(&root).unwrap();
    let formatter = PromptFormatter::load(&root.join("mllm"), &config).unwrap();
    let prompt = crate::imagegen::ming_image::parity_tests::GEN_PROMPT;
    assert_eq!(formatter.input_ids(prompt).unwrap(), fixture_ids());
    // A prompt that spells out a block token is a 400.
    let err = formatter.prompt_ids("<|image_patch|> hi").err();
    if let Some(err) = err {
        assert!(
            err.downcast_ref::<InvalidImageRequest>().is_some(),
            "{err:#}"
        );
    }
}
