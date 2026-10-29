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

use super::*;
use crate::models::bailing_moe::FeedForward;

/// A three-layer Ming-shaped config: one dense prefix layer, then MoE layers
/// with sigmoid scores, grouped routing, a correction bias and routed scaling.
fn args() -> ModelArgs {
    serde_json::from_str(
        r#"{
            "model_type": "bailing_moe_v2",
            "hidden_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 40,
            "moe_intermediate_size": 16,
            "num_experts": 8,
            "num_shared_experts": 1,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "first_k_dense_replace": 1,
            "rms_norm_eps": 1e-06,
            "rope_theta": 600000,
            "partial_rotary_factor": 0.5,
            "use_qk_norm": true,
            "moe_router_enable_expert_bias": true,
            "routed_scaling_factor": 2.5,
            "score_function": "sigmoid",
            "n_group": 2,
            "topk_group": 1,
            "tie_word_embeddings": false,
            "vocab_size": 24
        }"#,
    )
    .expect("the tiny config parses")
}

/// Seeded uniform values in `[-0.5, 0.5)`.
fn filled(shape: &[i32], seed: u32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state = 0x9E37_79B9_u32 ^ seed.wrapping_mul(0x85EB_CA6B);
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

/// A raw-layout checkpoint for [`args`] whose MoE layers carry both a text
/// `gate` and an `image_gate`, each with its own weight and correction bias.
fn weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let experts = args.num_experts() as i32;
    let moe_ff = args.moe_intermediate_size() as i32;
    let head_dim = args.head_dim() as i32;
    let mut seed = 0;
    let mut next = |shape: &[i32]| {
        seed += 1;
        filled(shape, seed)
    };
    let mut w = WeightMap::new();
    w.insert(
        "model.word_embeddings.weight".into(),
        next(&[args.vocab_size as i32, hidden]),
    );
    w.insert("model.norm.weight".into(), ones(&[hidden]));
    w.insert(
        "lm_head.weight".into(),
        next(&[args.vocab_size as i32, hidden]),
    );
    for layer in 0..args.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            w.insert(format!("{p}.{norm}.weight"), ones(&[hidden]));
        }
        for norm in ["query_layernorm", "key_layernorm"] {
            w.insert(format!("{p}.attention.{norm}.weight"), ones(&[head_dim]));
        }
        w.insert(
            format!("{p}.attention.query_key_value.weight"),
            next(&[args.qkv_out_features() as i32, hidden]),
        );
        w.insert(
            format!("{p}.attention.dense.weight"),
            next(&[hidden, hidden]),
        );
        let mlp = format!("{p}.mlp");
        if !args.is_moe_layer(layer) {
            let ff = args.intermediate_size as i32;
            w.insert(format!("{mlp}.gate_proj.weight"), next(&[ff, hidden]));
            w.insert(format!("{mlp}.up_proj.weight"), next(&[ff, hidden]));
            w.insert(format!("{mlp}.down_proj.weight"), next(&[hidden, ff]));
            continue;
        }
        for router in ["gate", IMAGE_ROUTER] {
            w.insert(format!("{mlp}.{router}.weight"), next(&[experts, hidden]));
            w.insert(format!("{mlp}.{router}.expert_bias"), next(&[experts]));
        }
        for e in 0..experts {
            let ex = format!("{mlp}.experts.{e}");
            w.insert(format!("{ex}.gate_proj.weight"), next(&[moe_ff, hidden]));
            w.insert(format!("{ex}.up_proj.weight"), next(&[moe_ff, hidden]));
            w.insert(format!("{ex}.down_proj.weight"), next(&[hidden, moe_ff]));
        }
        let s = format!("{mlp}.shared_experts");
        w.insert(format!("{s}.gate_proj.weight"), next(&[moe_ff, hidden]));
        w.insert(format!("{s}.up_proj.weight"), next(&[moe_ff, hidden]));
        w.insert(format!("{s}.down_proj.weight"), next(&[hidden, moe_ff]));
    }
    w
}

/// The same checkpoint with each layer's `image_gate` moved into `gate`, so
/// the plain model routes every token with the image router.
fn image_router_as_text_router(weights: &WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (k, v) in weights {
        if k.contains(".mlp.gate.") {
            continue;
        }
        out.insert(
            k.replace(".mlp.image_gate.", ".mlp.gate."),
            mlxcel_core::copy(v),
        );
    }
    out
}

fn sparse(model: &BailingMoeModel, layer: usize) -> &BailingMoeSparseBlock {
    match &model.layers[layer].mlp {
        FeedForward::Sparse(block) => block,
        FeedForward::Dense(_) => panic!("layer {layer} is dense"),
    }
}

fn host(x: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32))
}

/// Rows `rows` of `x` (`[1, L, D]`) as `[1, rows.len(), D]`.
fn take_rows(x: &MlxArray, rows: &[i32]) -> UniquePtr<MlxArray> {
    let idx = mlxcel_core::from_slice_i32(rows, &[rows.len() as i32]);
    mlxcel_core::take(x, &idx, 1)
}

fn bool_mask(values: &[bool]) -> UniquePtr<MlxArray> {
    let ints: Vec<i32> = values.iter().map(|&b| i32::from(b)).collect();
    let m = mlxcel_core::from_slice_i32(&ints, &[1, values.len() as i32]);
    mlxcel_core::astype(&m, mlxcel_core::dtype::BOOL)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn per_token_router_selection_equals_each_router_alone_on_its_tokens() {
    let args = args();
    let w = weights(&args);
    let model = BailingMoeModel::from_weights(&w, &args).unwrap();
    let routers = ImageRouters::from_weights(&w, &args).unwrap();
    let image_model =
        BailingMoeModel::from_weights(&image_router_as_text_router(&w), &args).unwrap();
    let hidden = args.hidden_size as i32;
    let x = filled(&[1, 6, hidden], 991);
    let mask = [false, true, true, false, true, false];
    let (text_rows, image_rows): (Vec<i32>, Vec<i32>) = (0..6).partition(|&i| !mask[i as usize]);

    for layer in 1..args.num_hidden_layers {
        let block = sparse(&model, layer);
        let gate = routers.get(layer).expect("MoE layer has an image router");
        let mixed = host(&block.forward_image_routed(&x, gate, &bool_mask(&mask)));
        let text_alone = host(&block.forward(&take_rows(&x, &text_rows)));
        let image_alone = host(&sparse(&image_model, layer).forward(&take_rows(&x, &image_rows)));
        let row = |v: &[f32], r: usize| v[r * hidden as usize..(r + 1) * hidden as usize].to_vec();
        for (j, &r) in text_rows.iter().enumerate() {
            let d = max_abs_diff(&row(&mixed, r as usize), &row(&text_alone, j));
            assert!(d < 1e-5, "layer {layer} text token {r}: {d}");
        }
        for (j, &r) in image_rows.iter().enumerate() {
            let d = max_abs_diff(&row(&mixed, r as usize), &row(&image_alone, j));
            assert!(d < 1e-5, "layer {layer} image token {r}: {d}");
        }
        // The routers disagree, so the test is not vacuous.
        let all_text = host(&block.forward(&x));
        assert!(max_abs_diff(&mixed, &all_text) > 1e-3, "layer {layer}");
    }
}

#[test]
fn an_all_text_mask_is_bit_identical_to_the_plain_forward() {
    let args = args();
    let w = weights(&args);
    let model = BailingMoeModel::from_weights(&w, &args).unwrap();
    let routers = ImageRouters::from_weights(&w, &args).unwrap();
    let x = filled(&[1, 5, args.hidden_size as i32], 5);
    let block = sparse(&model, 1);
    let routed = block.forward_image_routed(&x, routers.get(1).unwrap(), &bool_mask(&[false; 5]));
    assert_eq!(host(&routed), host(&block.forward(&x)));
}

#[test]
fn hidden_states_are_every_layer_input_then_the_final_norm() {
    let args = args();
    let w = weights(&args);
    let model = BailingMoeModel::from_weights(&w, &args).unwrap();
    let ids = mlxcel_core::from_slice_i32(&[3, 1, 4, 1, 5, 9], &[1, 6]);
    let embeds = model.word_embeddings.forward(&ids);
    let states = model.hidden_states_from_embeds(&embeds, None).unwrap();
    assert_eq!(states.len(), args.num_hidden_layers + 1);
    assert_eq!(host(&states[0]), host(&embeds));
    // The last state is what the plain forward feeds its output head.
    let head = model.lm_head.as_ref().unwrap();
    let mut caches = model.make_caches();
    let logits = model.forward(&ids, &mut caches, None);
    assert_eq!(host(&head.forward(&states[3])), host(&logits));

    // With the routers and a mixed mask only the image tokens' states move.
    let routers = ImageRouters::from_weights(&w, &args).unwrap();
    let mask = bool_mask(&[false, false, false, true, true, true]);
    let routed = model
        .hidden_states_from_embeds(&embeds, Some((&routers, &mask)))
        .unwrap();
    let d = args.hidden_size;
    let (plain, mixed) = (host(&states[2]), host(&routed[2]));
    assert_eq!(
        plain[..3 * d],
        mixed[..3 * d],
        "causal text prefix unchanged"
    );
    assert!(max_abs_diff(&plain[3 * d..], &mixed[3 * d..]) > 1e-4);
    // The dense prefix layer has no image router.
    assert!(routers.get(0).is_none());
}

#[test]
fn image_routers_reject_a_missing_router_or_a_wrong_bias() {
    let args = args();
    let mut w = weights(&args);
    w.insert(
        "model.layers.2.mlp.image_gate.expert_bias".into(),
        filled(&[5], 1),
    );
    let err = ImageRouters::from_weights(&w, &args).err().unwrap();
    assert!(err.contains("image_gate.expert_bias"), "{err}");
    w.remove("model.layers.2.mlp.image_gate.weight");
    let err = ImageRouters::from_weights(&w, &args).err().unwrap();
    assert!(
        err.contains("model.layers.2.mlp.image_gate.weight"),
        "{err}"
    );
}

#[test]
fn an_image_mask_that_does_not_cover_every_token_is_an_error() {
    let args = args();
    let w = weights(&args);
    let model = BailingMoeModel::from_weights(&w, &args).unwrap();
    let routers = ImageRouters::from_weights(&w, &args).unwrap();
    let ids = mlxcel_core::from_slice_i32(&[3, 1, 4, 1, 5, 9], &[1, 6]);
    let embeds = model.word_embeddings.forward(&ids);

    let short = bool_mask(&[false, true, true]);
    let err = model
        .hidden_states_from_embeds(&embeds, Some((&routers, &short)))
        .err()
        .expect("a short mask is rejected");
    assert!(err.contains("image mask"), "{err}");

    let ints = mlxcel_core::from_slice_i32(&[0, 0, 0, 1, 1, 1], &[1, 6]);
    let err = model
        .hidden_states_from_embeds(&embeds, Some((&routers, &ints)))
        .err()
        .expect("a non-boolean mask is rejected");
    assert!(err.contains("boolean"), "{err}");

    let flat = mlxcel_core::reshape(&embeds, &[6, args.hidden_size as i32]);
    let err = model
        .hidden_states_from_embeds(&flat, None)
        .err()
        .expect("rank-2 embeds are rejected");
    assert!(err.contains("[B, L, hidden]"), "{err}");
}
