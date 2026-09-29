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

//! Unit tests for the embedding-driven Gemma 3 backbone (random weights).

use super::*;
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;

fn rand(key: &mut u64, shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
    *key += 1;
    let k = mlxcel_core::random_key(*key);
    let x = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, &*k) };
    mlxcel_core::multiply_scalar(&x, scale)
}

fn tiny_args() -> ModelArgs {
    ModelArgs {
        hidden_size: 8,
        num_hidden_layers: 4,
        intermediate_size: 16,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        sliding_window: 4,
        sliding_window_pattern: 2,
        vocab_size: 1,
        ..ModelArgs::default()
    }
}

fn tiny_weights(prefix: &str, args: &ModelArgs) -> WeightMap {
    let mut key = 7u64;
    let mut w = WeightMap::new();
    let (h, i, hd) = (
        args.hidden_size as i32,
        args.intermediate_size as i32,
        args.head_dim as i32,
    );
    let qd = (args.num_attention_heads * args.head_dim) as i32;
    let kd = (args.num_key_value_heads * args.head_dim) as i32;
    for l in 0..args.num_hidden_layers {
        let p = format!("{prefix}.layers.{l}");
        let mut put = |name: &str, shape: &[i32], scale: f32| {
            w.insert(format!("{p}.{name}"), rand(&mut key, shape, scale));
        };
        put("self_attn.q_proj.weight", &[qd, h], 0.3);
        put("self_attn.k_proj.weight", &[kd, h], 0.3);
        put("self_attn.v_proj.weight", &[kd, h], 0.3);
        put("self_attn.o_proj.weight", &[h, qd], 0.3);
        put("self_attn.q_norm.weight", &[hd], 0.1);
        put("self_attn.k_norm.weight", &[hd], 0.1);
        put("mlp.gate_proj.weight", &[i, h], 0.3);
        put("mlp.up_proj.weight", &[i, h], 0.3);
        put("mlp.down_proj.weight", &[h, i], 0.3);
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            put(&format!("{norm}.weight"), &[h], 0.1);
        }
    }
    w.insert(format!("{prefix}.norm.weight"), rand(&mut key, &[h], 0.1));
    w
}

fn to_vec(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

#[test]
fn caches_follow_the_sliding_global_pattern() {
    let args = tiny_args();
    let backbone = Gemma3Backbone::from_weights(&tiny_weights("bb", &args), "bb", &args).unwrap();
    let caches = backbone.make_caches();
    assert_eq!(caches.len(), 4);
    let globals: Vec<bool> = (0..4).map(|i| caches.is_global(i)).collect();
    assert_eq!(globals, vec![false, true, false, true]);
    assert_eq!(caches.offset(), 0);
}

#[test]
fn missing_layer_weight_is_an_error() {
    let args = tiny_args();
    let mut weights = tiny_weights("bb", &args);
    weights.remove("bb.layers.3.mlp.up_proj.weight");
    let err = Gemma3Backbone::from_weights(&weights, "bb", &args)
        .err()
        .unwrap();
    assert!(err.contains("bb.layers.3.mlp.up_proj"), "{err}");
}

#[test]
fn incremental_steps_match_one_shot_prefill() {
    let args = tiny_args();
    let backbone = Gemma3Backbone::from_weights(&tiny_weights("bb", &args), "bb", &args).unwrap();
    let mut key = 99u64;
    // Five tokens overflow the four-token sliding window in the one-shot
    // call, exercising the explicit sliding mask.
    let x = rand(&mut key, &[2, 5, 8], 1.0);

    let mut full_caches = backbone.make_caches();
    let full = backbone.forward_embeds(&x, &mut full_caches).unwrap();
    assert_eq!(mlxcel_core::array_shape(&full), vec![2, 5, 8]);
    assert_eq!(full_caches.offset(), 5);

    let mut caches = backbone.make_caches();
    let head = mlxcel_core::slice(&x, &[0, 0, 0], &[2, 2, 8]);
    let _ = backbone.forward_embeds(&head, &mut caches).unwrap();
    let mut last = None;
    for t in 2..5 {
        let xt = mlxcel_core::slice(&x, &[0, t, 0], &[2, t + 1, 8]);
        last = Some(backbone.forward_embeds(&xt, &mut caches).unwrap());
    }
    let last = to_vec(&last.unwrap());
    let want = to_vec(&mlxcel_core::slice(&full, &[0, 4, 0], &[2, 5, 8]));
    for (a, b) in last.iter().zip(&want) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn wrong_cache_count_is_rejected() {
    let args = tiny_args();
    let backbone = Gemma3Backbone::from_weights(&tiny_weights("bb", &args), "bb", &args).unwrap();
    let other = Gemma3Backbone::from_weights(
        &tiny_weights(
            "x",
            &ModelArgs {
                num_hidden_layers: 2,
                ..tiny_args()
            },
        ),
        "x",
        &ModelArgs {
            num_hidden_layers: 2,
            ..tiny_args()
        },
    )
    .unwrap();
    let mut key = 1u64;
    let x = rand(&mut key, &[1, 1, 8], 1.0);
    assert!(
        backbone
            .forward_embeds(&x, &mut other.make_caches())
            .is_err()
    );
}
