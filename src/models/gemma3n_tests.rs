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

//! Gemma 3n KV-shared RoPE offset (#2379, after ml-explore/mlx-lm#1886).
//!
//! A tiny f32 model with two KV-shared layers, built from deterministic
//! host-side weights, must give the same last-token logits whether its
//! prompt arrives in one pass, in chunks, or one token at a time. Before the
//! fix a KV-shared layer rotated its queries at its owner cache's offset after
//! the owner had appended the current pass, so each split moved the shared
//! layers' query positions by a different amount. Run with
//! `MLX_ENABLE_TF32=0` so the f32 matmuls are f32.

use super::{Gemma3nLanguageModel, TextConfig};
use mlxcel_core::weights::WeightMap;

const HIDDEN: i32 = 32;
const LAYERS: i32 = 4;
const INTERMEDIATE: i32 = 64;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const HEAD_DIM: i32 = 16;
const VOCAB: i32 = 64;
const PER_LAYER: i32 = 8;
const ALTUP: i32 = 4;
const LAUREL: i32 = 4;

/// Mirrors upstream's `test_gemma3n_kv_shared_layers_ignore_chunking`, with
/// four AltUp planes as the shipped checkpoints use. `sliding_window` exceeds
/// the 12-token prompt, so the sliding layers never drop a key and the test
/// isolates the RoPE offset from the window logic.
const TINY_CONFIG: &str = r#"{
    "model_type": "gemma3n_text",
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "intermediate_size": 64,
    "num_attention_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "vocab_size": 64,
    "num_key_value_heads": 1,
    "num_kv_shared_layers": 2,
    "vocab_size_per_layer_input": 64,
    "sliding_window": 32,
    "max_position_embeddings": 128,
    "rope_local_base_freq": 1.0,
    "rope_theta": 1000.0,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    "activation_sparsity_pattern": [0.0, 0.0, 0.0, 0.0],
    "hidden_size_per_layer_input": 8,
    "altup_num_inputs": 4,
    "altup_coef_clip": 120.0,
    "altup_correct_scale": true,
    "altup_active_idx": 0,
    "laurel_rank": 4
}"#;

const PROMPT: [i32; 12] = [5, 17, 33, 2, 60, 41, 8, 23, 50, 11, 3, 29];

/// Deterministic pseudo-random tensor: `amplitude * sin(golden * i + seed)`,
/// offset by `center`.
fn tensor(
    shape: &[i32],
    seed: f32,
    center: f32,
    amplitude: f32,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| center + amplitude * (i as f32 * 0.618_034 + seed).sin())
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn tiny_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    let mut seed = 0.0f32;
    let mut add = |name: String, shape: &[i32], center: f32, amplitude: f32| {
        seed += 1.37;
        weights.insert(name, tensor(shape, seed, center, amplitude));
    };
    let linear = 0.25;
    let norm = (1.0, 0.1);
    add("lm.embed_tokens.weight".into(), &[VOCAB, HIDDEN], 0.0, 0.5);
    add(
        "lm.embed_tokens_per_layer.weight".into(),
        &[VOCAB, LAYERS * PER_LAYER],
        0.0,
        0.5,
    );
    add(
        "lm.per_layer_model_projection.weight".into(),
        &[LAYERS * PER_LAYER, HIDDEN],
        0.0,
        linear,
    );
    add(
        "lm.per_layer_projection_norm.weight".into(),
        &[PER_LAYER],
        norm.0,
        norm.1,
    );
    for i in 0..ALTUP - 1 {
        add(
            format!("lm.altup_projections.{i}.weight"),
            &[HIDDEN, HIDDEN],
            0.0,
            linear,
        );
        add(
            format!("lm.altup_unembed_projections.{i}.weight"),
            &[HIDDEN, HIDDEN],
            0.0,
            linear,
        );
    }
    add("lm.norm.weight".into(), &[HIDDEN], norm.0, norm.1);

    for layer in 0..LAYERS {
        let p = format!("lm.layers.{layer}");
        for (leaf, shape) in [
            ("self_attn.q_proj.weight", [HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.k_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.v_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.o_proj.weight", [HIDDEN, HEADS * HEAD_DIM]),
            ("mlp.gate_proj.weight", [INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj.weight", [INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj.weight", [HIDDEN, INTERMEDIATE]),
            ("altup.correction_coefs.weight", [ALTUP, ALTUP]),
            ("altup.prediction_coefs.weight", [ALTUP * ALTUP, ALTUP]),
            ("altup.modality_router.weight", [ALTUP, HIDDEN]),
            ("laurel.linear_left.weight", [LAUREL, HIDDEN]),
            ("laurel.linear_right.weight", [HIDDEN, LAUREL]),
            ("per_layer_input_gate.weight", [PER_LAYER, HIDDEN]),
            ("per_layer_projection.weight", [HIDDEN, PER_LAYER]),
        ] {
            add(format!("{p}.{leaf}"), &shape, 0.0, linear);
        }
        for (leaf, dim) in [
            ("self_attn.q_norm.weight", HEAD_DIM),
            ("self_attn.k_norm.weight", HEAD_DIM),
            ("input_layernorm.weight", HIDDEN),
            ("post_attention_layernorm.weight", HIDDEN),
            ("pre_feedforward_layernorm.weight", HIDDEN),
            ("post_feedforward_layernorm.weight", HIDDEN),
            ("altup.correct_output_scale", HIDDEN),
            ("altup.router_norm.weight", HIDDEN),
            ("laurel.post_laurel_norm.weight", HIDDEN),
            ("post_per_layer_input_norm.weight", HIDDEN),
        ] {
            add(format!("{p}.{leaf}"), &[dim], norm.0, norm.1);
        }
    }
    weights
}

fn tiny_model() -> Gemma3nLanguageModel {
    let config: TextConfig = serde_json::from_str(TINY_CONFIG).expect("tiny config parses");
    assert_eq!(config.first_kv_shared_layer_idx(), 2);
    Gemma3nLanguageModel::from_weights(&tiny_weights(), &config, "lm").expect("tiny model loads")
}

/// Feed `PROMPT` in the given chunk sizes over one set of caches and return
/// the last position's logits.
fn last_logits(model: &Gemma3nLanguageModel, chunks: &[usize]) -> Vec<f32> {
    assert_eq!(chunks.iter().sum::<usize>(), PROMPT.len());
    let mut caches = model.make_caches();
    let mut start = 0;
    let mut logits = None;
    for &len in chunks {
        let ids = &PROMPT[start..start + len];
        let input = mlxcel_core::from_slice_i32(ids, &[1, len as i32]);
        let out = model.forward(&input, &mut caches);
        mlxcel_core::eval(&out);
        logits = Some(out);
        start += len;
    }
    let logits = logits.expect("at least one chunk");
    let last = *mlxcel_core::array_shape(&logits).get(1).unwrap();
    let row = mlxcel_core::slice(&logits, &[0, last - 1, 0], &[1, last, VOCAB]);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    mlxcel_core::array_to_raw_bytes(&row)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Every split of the prompt must give the same last-token logits.
///
/// Every multi-pass split is compared with `[5, 4, 3]` at 1e-3; measured on
/// CUDA they agree to about 1e-6 with the fix and drift by 5.6e-3 to 4e-2
/// without it. The single 12-row pass is compared at 1e-2 instead: the Gemma
/// 3n MLP and the AltUp input round their activations to bf16 (matching the
/// bf16 checkpoints), so the 12-row matmuls and narrower ones can land one
/// bf16 ulp apart, which measures 2.7e-3 here with the fix and 2.7e-2 to
/// 6.7e-2 without it.
#[test]
fn kv_shared_layers_ignore_chunking() {
    let model = tiny_model();
    let whole = last_logits(&model, &[12]);
    assert_eq!(whole.len(), VOCAB as usize);
    assert!(whole.iter().all(|v| v.is_finite()), "{whole:?}");
    // A degenerate model would pass trivially: require a spread of logits.
    let spread = whole.iter().cloned().fold(f32::MIN, f32::max)
        - whole.iter().cloned().fold(f32::MAX, f32::min);
    assert!(spread > 0.1, "logit spread {spread}");

    let chunked = last_logits(&model, &[5, 4, 3]);
    for split in [vec![1; 12], vec![2; 6], vec![11, 1], vec![6, 6]] {
        let diff = max_abs_diff(&chunked, &last_logits(&model, &split));
        assert!(
            diff < 1e-3,
            "split {split:?} differs from [5, 4, 3] by {diff}"
        );
    }
    let decode = last_logits(&model, &[1; 12]);
    for (name, other) in [("[5, 4, 3]", &chunked), ("token-by-token decode", &decode)] {
        let diff = max_abs_diff(&whole, other);
        assert!(diff < 1e-2, "{name} differs from one pass by {diff}");
    }
}
