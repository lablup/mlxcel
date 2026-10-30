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

//! The conditioning entry point on a tiny random Ministral 3, plus an ignored
//! check of the real `text_encoder/` against upstream's hidden states.

use std::path::PathBuf;

use mlxcel_core::dtype;
use mlxcel_core::weights::{WeightMap, load_safetensors};

use super::*;
use crate::imagegen::testing::{rel_rms, values};

const HIDDEN: i32 = 16;
const LAYERS: usize = 3;

fn tiny_args() -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "ministral3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "rope_parameters": {
            "rope_theta": 1000000.0,
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 16384
        }
    }))
    .unwrap()
}

fn tiny_model() -> Ministral3Model {
    let args = tiny_args();
    let mut w = WeightMap::new();
    let mut seed = 7_u64;
    let mut put = |w: &mut WeightMap, key: &str, shape: &[i32], scale: f32, offset: f32| {
        seed += 1;
        let key_arr = mlxcel_core::random_key(seed);
        // SAFETY: `key_arr` is alive for the call.
        let x = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, &*key_arr) };
        let x = mlxcel_core::add(
            &mlxcel_core::multiply(&x, &mlxcel_core::full_f32(&[1], scale, dtype::FLOAT32)),
            &mlxcel_core::full_f32(&[1], offset, dtype::FLOAT32),
        );
        w.insert(key.to_string(), x);
    };
    put(&mut w, "model.embed_tokens.weight", &[32, HIDDEN], 1.0, 0.0);
    put(&mut w, "model.norm.weight", &[HIDDEN], 0.1, 1.5);
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        put(
            &mut w,
            &format!("{p}.self_attn.q_proj.weight"),
            &[16, HIDDEN],
            0.3,
            0.0,
        );
        put(
            &mut w,
            &format!("{p}.self_attn.k_proj.weight"),
            &[8, HIDDEN],
            0.3,
            0.0,
        );
        put(
            &mut w,
            &format!("{p}.self_attn.v_proj.weight"),
            &[8, HIDDEN],
            0.3,
            0.0,
        );
        put(
            &mut w,
            &format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, 16],
            0.3,
            0.0,
        );
        put(
            &mut w,
            &format!("{p}.mlp.gate_proj.weight"),
            &[32, HIDDEN],
            0.3,
            0.0,
        );
        put(
            &mut w,
            &format!("{p}.mlp.up_proj.weight"),
            &[32, HIDDEN],
            0.3,
            0.0,
        );
        put(
            &mut w,
            &format!("{p}.mlp.down_proj.weight"),
            &[HIDDEN, 32],
            0.3,
            0.0,
        );
        put(
            &mut w,
            &format!("{p}.input_layernorm.weight"),
            &[HIDDEN],
            0.1,
            1.0,
        );
        put(
            &mut w,
            &format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
            0.1,
            1.0,
        );
    }
    Ministral3Model::from_weights(&w, &args).unwrap()
}

const IDS: [i32; 5] = [1, 5, 9, 2, 17];

fn hidden(model: &Ministral3Model, layers: Option<usize>, norm: bool) -> Vec<f32> {
    let x = mlxcel_core::from_slice_i32(&IDS, &[1, IDS.len() as i32]);
    let mut caches = model.make_caches();
    values(
        &model
            .forward_hidden_states(&x, &mut caches, layers, norm)
            .unwrap(),
    )
}

#[test]
fn conditioning_is_the_penultimate_layer_without_the_final_norm() {
    let model = tiny_model();
    let cond = conditioning_hidden_states(&model, &IDS).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&cond),
        [1, IDS.len() as i32, HIDDEN]
    );
    let cond = values(&cond);
    // Exactly the first `layers - 1` layers, no norm.
    assert_eq!(cond, hidden(&model, Some(LAYERS - 1), false));
    // Not the last layer, and not normalized: the final norm applied to the
    // conditioning gives the normalized penultimate state, which differs.
    let all = hidden(&model, None, false);
    assert!(rel_rms(&cond, &all) > 1e-2, "conditioning ran every layer");
    let normed = hidden(&model, Some(LAYERS - 1), true);
    assert!(
        rel_rms(&cond, &normed) > 1e-2,
        "conditioning applied the final norm"
    );
    let renormed = {
        let x = mlxcel_core::from_slice_f32(&cond, &[1, IDS.len() as i32, HIDDEN]);
        values(&model.norm.forward(&x))
    };
    assert!(rel_rms(&renormed, &normed) < 1e-6);
}

#[test]
fn the_full_entry_point_reproduces_the_generation_forward() {
    let model = tiny_model();
    let x = mlxcel_core::from_slice_i32(&IDS, &[1, IDS.len() as i32]);
    let mut caches = model.make_caches();
    let logits = values(&model.forward_with_caches(&x, &mut caches));
    let mut caches = model.make_caches();
    let h = model
        .forward_hidden_states(&x, &mut caches, None, true)
        .unwrap();
    let via_entry = values(&model.embed_tokens.as_linear(&h));
    assert_eq!(logits, via_entry);
}

#[test]
fn the_entry_point_rejects_too_many_layers_or_too_few_caches() {
    let model = tiny_model();
    let x = mlxcel_core::from_slice_i32(&IDS, &[1, IDS.len() as i32]);
    let mut caches = model.make_caches();
    assert!(
        model
            .forward_hidden_states(&x, &mut caches, Some(LAYERS + 1), false)
            .is_err()
    );
    let mut short = model.make_caches();
    short.truncate(1);
    assert!(
        model
            .forward_hidden_states(&x, &mut short, Some(2), false)
            .is_err()
    );
    assert!(
        model
            .forward_hidden_states(&x, &mut short, Some(1), false)
            .is_ok()
    );
}

fn turbo_root() -> Option<PathBuf> {
    let root = std::env::var_os("MLXCEL_ERNIE_IMAGE_TURBO")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/mlx/ernie-image-turbo")
        });
    root.join("text_encoder")
        .exists()
        .then_some(root)
        .or_else(|| {
            eprintln!("skipping: the ERNIE-Image-Turbo checkpoint is absent");
            None
        })
}

/// Token ids and penultimate-layer hidden states of the real encoder against
/// upstream's (the oracle run in `MLXCEL_ERNIE_IMAGE_ORACLE_DIR`).
#[test]
#[ignore = "needs models/mlx/ernie-image-turbo, MLXCEL_ERNIE_IMAGE_ORACLE_DIR and a GPU"]
fn the_real_encoder_matches_upstream() {
    let (Some(root), Some(dir)) = (
        turbo_root(),
        std::env::var_os("MLXCEL_ERNIE_IMAGE_ORACLE_DIR").map(PathBuf::from),
    ) else {
        return;
    };
    let oracle = load_safetensors(dir.join("turbo_1024.safetensors")).unwrap();
    let encoder = ErnieTextEncoder::load(&root).unwrap();
    let prompt = "A red panda serving tea in a moonlit bamboo forest";
    let ids = encoder.token_ids(prompt).unwrap();
    let want_ids: Vec<i32> = values(&oracle["ids_0"]).iter().map(|v| *v as i32).collect();
    assert_eq!(ids, want_ids);
    assert_eq!(encoder.token_ids("").unwrap(), vec![1]);
    let ours = encoder.encode(prompt).unwrap();
    let err = rel_rms(&values(&ours), &values(&oracle["text_0"]));
    eprintln!("ERNIE-Image conditioning rel-RMS vs upstream: {err:.3e}");
    assert!(err < 2e-2, "conditioning rel-RMS {err}");
}

#[test]
fn quantized_configs_are_refused_before_any_weight_loads() {
    let path = std::path::Path::new("config.json");
    for cfg in [
        serde_json::json!({"quantization": {"bits": 4}}),
        serde_json::json!({"quantization_config": {"quant_method": "modelopt"}}),
        serde_json::json!({"text_config": {"quantization": {"bits": 4}}}),
    ] {
        let err = reject_quantized_config(&cfg, path).unwrap_err().to_string();
        assert!(err.contains("quantization"), "{err}");
    }
    for cfg in [
        serde_json::json!({"hidden_size": 8}),
        serde_json::json!({"text_config": {"hidden_size": 8}, "quantization": null}),
    ] {
        assert!(reject_quantized_config(&cfg, path).is_ok());
    }
}
