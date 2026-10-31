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

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use super::*;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel};
use crate::embeddings::pooling::{PoolingMode, normalize_l2, truncate_dimensions};
use crate::models::ModelType;
use crate::models::detection::{get_model_type, is_embedding_checkpoint};
use crate::models::embedding_test_support::{mlx_test_guard, to_vec};

fn temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "mlxcel_embedding_gemma2_{label}_{}_{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn det_tensor(shape: &[i32], seed: u32) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let mut s = seed.wrapping_add(1);
    let data: Vec<f32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let unit = (s >> 8) as f32 / ((1u32 << 24) as f32);
            (unit - 0.5) * 0.2
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn synthetic_config(num_layers: usize, sliding_window: i32) -> Value {
    let layer_types: Vec<&str> = (0..num_layers)
        .map(|i| {
            if i == num_layers - 1 && num_layers > 1 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let mut per_layer_config = serde_json::Map::new();
    if num_layers > 1 {
        let last_key = format!("{:02}", num_layers - 1);
        per_layer_config.insert(
            last_key,
            json!({
                "head_dim": 16,
                "num_key_value_heads": 1,
                "rope_theta": 1_000_000.0
            }),
        );
    }
    json!({
        "architectures": ["EmbeddingGemma2Model"],
        "model_type": "embedding_gemma2",
        "text_config": {
            "model_type": "embedding_gemma2_text",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "embedding_dim": 24,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "sliding_window": sliding_window,
            "layer_types": layer_types,
            "per_layer_config": per_layer_config
        }
    })
}

fn build_synthetic_weights(cfg: &EmbeddingGemma2Config) -> WeightMap {
    let mut w = WeightMap::new();
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let ple_d = cfg.ple_dim;
    let n_layers = cfg.num_hidden_layers as i32;
    let emb_d = cfg.embedding_dim as i32;

    w.insert(
        "language_model.embed_tokens.weight".to_string(),
        det_tensor(&[cfg.vocab_size, h], 1),
    );
    w.insert(
        "language_model.ple.per_layer_model_projection.weight".to_string(),
        det_tensor(&[n_layers * ple_d, h], 2),
    );
    w.insert(
        "language_model.ple.per_layer_projection_norm.weight".to_string(),
        mlxcel_core::ones(&[ple_d], mlxcel_core::dtype::FLOAT32),
    );

    for layer in &cfg.layers {
        let i = layer.layer_idx;
        let q_out = layer.num_attention_heads * layer.head_dim;
        let kv_out = layer.num_key_value_heads * layer.head_dim;
        let p = format!("language_model.layers.{i}");
        let seed_base = (i as u32 + 1) * 100;

        for norm_name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            w.insert(
                format!("{p}.{norm_name}.weight"),
                mlxcel_core::ones(&[h], mlxcel_core::dtype::FLOAT32),
            );
        }
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            det_tensor(&[q_out, h], seed_base + 1),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            det_tensor(&[kv_out, h], seed_base + 2),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            det_tensor(&[kv_out, h], seed_base + 3),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            det_tensor(&[h, q_out], seed_base + 4),
        );
        w.insert(
            format!("{p}.self_attn.q_norm.weight"),
            mlxcel_core::ones(&[layer.head_dim], mlxcel_core::dtype::FLOAT32),
        );
        w.insert(
            format!("{p}.self_attn.k_norm.weight"),
            mlxcel_core::ones(&[layer.head_dim], mlxcel_core::dtype::FLOAT32),
        );
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            det_tensor(&[inter, h], seed_base + 5),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            det_tensor(&[inter, h], seed_base + 6),
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            det_tensor(&[h, inter], seed_base + 7),
        );
        w.insert(
            format!("{p}.ple_block.per_layer_input_gate.weight"),
            det_tensor(&[ple_d, h], seed_base + 8),
        );
        w.insert(
            format!("{p}.ple_block.per_layer_projection.weight"),
            det_tensor(&[h, ple_d], seed_base + 9),
        );
        w.insert(
            format!("{p}.ple_block.post_per_layer_input_norm.weight"),
            mlxcel_core::ones(&[h], mlxcel_core::dtype::FLOAT32),
        );
        w.insert(
            format!("{p}.layer_scalar"),
            mlxcel_core::ones(&[1], mlxcel_core::dtype::FLOAT32),
        );
    }

    w.insert(
        "language_model.norm.weight".to_string(),
        mlxcel_core::ones(&[h], mlxcel_core::dtype::FLOAT32),
    );
    w.insert(
        "language_model.embedding_projection.weight".to_string(),
        det_tensor(&[emb_d, h], 999),
    );
    w
}

#[test]
fn detection_accepts_embedding_gemma2_variants_and_preserves_bf16() {
    let dir_mm = temp_dir("mm");
    let cfg_mm = json!({
        "architectures": ["EmbeddingGemma2Model"],
        "model_type": "embedding_gemma2",
        "text_config": { "model_type": "embedding_gemma2_text", "hidden_size": 512 },
        "vision_config": { "model_type": "gemma4_vision", "hidden_size": 768 }
    });
    fs::write(dir_mm.join("config.json"), cfg_mm.to_string()).unwrap();
    assert_eq!(
        is_embedding_checkpoint(&dir_mm, &cfg_mm).unwrap(),
        Some(ModelType::EmbeddingGemma2)
    );
    assert_eq!(get_model_type(&dir_mm).unwrap(), ModelType::EmbeddingGemma2);
    assert!(!crate::models::bf16_to_f16_at_load(false, Some(&cfg_mm)));
    fs::remove_dir_all(&dir_mm).unwrap();

    let dir_txt = temp_dir("txt");
    let cfg_txt = json!({
        "model_type": "embedding_gemma2_text",
        "hidden_size": 512
    });
    fs::write(dir_txt.join("config.json"), cfg_txt.to_string()).unwrap();
    assert_eq!(
        is_embedding_checkpoint(&dir_txt, &cfg_txt).unwrap(),
        Some(ModelType::EmbeddingGemma2)
    );
    assert_eq!(
        get_model_type(&dir_txt).unwrap(),
        ModelType::EmbeddingGemma2
    );
    assert!(!crate::models::bf16_to_f16_at_load(false, Some(&cfg_txt)));
    fs::remove_dir_all(&dir_txt).unwrap();
}

#[test]
fn per_layer_config_resolves_local_and_global_layers() {
    let raw = json!({
        "model_type": "embedding_gemma2",
        "text_config": {
            "num_hidden_layers": 24,
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "embedding_dim": 768,
            "rope_theta": 10000.0,
            "sliding_window": 512,
            "per_layer_config": {
                "05": { "head_dim": 512, "num_key_value_heads": 1, "rope_theta": 1000000.0 },
                "11": { "head_dim": 512, "num_key_value_heads": 1, "rope_theta": 1000000.0 },
                "17": { "head_dim": 512, "num_key_value_heads": 1, "rope_theta": 1000000.0 },
                "23": { "head_dim": 512, "num_key_value_heads": 1, "rope_theta": 1000000.0 }
            }
        }
    });
    let cfg = EmbeddingGemma2Config::from_json(&raw, None).expect("config parses");
    assert_eq!(cfg.layers.len(), 24);
    for (i, layer) in cfg.layers.iter().enumerate() {
        if matches!(i, 5 | 11 | 17 | 23) {
            assert!(!layer.is_sliding, "layer {i} should be full_attention");
            assert_eq!(layer.head_dim, 512);
            assert_eq!(layer.num_key_value_heads, 1);
            assert_eq!(layer.rope_theta, 1_000_000.0);
        } else {
            assert!(layer.is_sliding, "layer {i} should be sliding_attention");
            assert_eq!(layer.head_dim, 256);
            assert_eq!(layer.num_key_value_heads, 2);
            assert_eq!(layer.rope_theta, 10_000.0);
        }
    }
}

#[test]
fn sanitize_and_quant_params_handle_hf_and_mlx_layouts() {
    let _guard = mlx_test_guard();
    let mut w = WeightMap::new();
    let one = || mlxcel_core::ones(&[1], mlxcel_core::dtype::FLOAT32);
    w.insert(
        "model.language_model.model.embed_tokens.weight".into(),
        one(),
    );
    w.insert("layers.0.per_layer_input_gate.weight".into(), one());
    w.insert("per_layer_model_projection.weight".into(), one());
    w.insert("embedding_projection.weight".into(), one());
    w.insert("model.vision_tower.patch_embed.weight".into(), one());
    w.insert("lm_head.weight".into(), one());
    w.insert("model.lm_head.weight".into(), one());
    w.insert("language_model.lm_head.weight".into(), one());

    sanitize_embedding_gemma2_weights(&mut w);
    assert!(w.contains_key("language_model.embed_tokens.weight"));
    assert!(w.contains_key("language_model.layers.0.ple_block.per_layer_input_gate.weight"));
    assert!(w.contains_key("language_model.ple.per_layer_model_projection.weight"));
    assert!(w.contains_key("language_model.embedding_projection.weight"));
    assert!(w.contains_key("vision_tower.patch_embed.weight"));
    assert!(!w.contains_key("lm_head.weight"));
    assert!(!w.contains_key("model.lm_head.weight"));
    assert!(!w.contains_key("language_model.lm_head.weight"));

    let q_cfg = json!({
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "language_model.ple": false,
            "vision_tower": { "group_size": 32, "bits": 8 }
        }
    });
    assert_eq!(
        quant_params_for_prefix(&q_cfg, "language_model.layers.0.self_attn"),
        (64, 4)
    );
    assert_eq!(
        quant_params_for_prefix(&q_cfg, "vision_tower.encoder"),
        (32, 8)
    );
    assert_eq!(
        quant_params_for_prefix(&q_cfg, "language_model.ple.per_layer_model_projection"),
        (0, 0)
    );
    assert_eq!(
        quant_params_for_prefix(
            &q_cfg,
            "language_model.layers.0.ple_block.per_layer_input_gate"
        ),
        (0, 0)
    );
}

#[test]
fn synthetic_forward_padding_invariance_bidirectionality_and_matryoshka() {
    let _guard = mlx_test_guard();
    let raw_cfg = synthetic_config(2, 2);
    let parsed_cfg = EmbeddingGemma2Config::from_json(&raw_cfg, None).expect("config parses");
    let mut weights = build_synthetic_weights(&parsed_cfg);
    let model = EmbeddingGemma2Model::from_weights(&mut weights, &raw_cfg, PoolingMode::Mean, true)
        .unwrap();

    // 1. Single row vs padded batch invariance
    let ids_single = mlxcel_core::from_slice_i32(&[3, 7, 11], &[1, 3]);
    let mask_single = mlxcel_core::from_slice_i32(&[1, 1, 1], &[1, 3]);
    let out_single = model
        .embed(&EmbeddingBatch {
            input_ids: &ids_single,
            attention_mask: &mask_single,
            token_type_ids: None,
            images: None,
        })
        .unwrap();
    let v_single = to_vec(&out_single.embeddings);
    assert_eq!(v_single.len(), 24);

    let ids_padded = mlxcel_core::from_slice_i32(&[3, 7, 11, 59, 61], &[1, 5]);
    let mask_padded = mlxcel_core::from_slice_i32(&[1, 1, 1, 0, 0], &[1, 5]);
    let out_padded = model
        .embed(&EmbeddingBatch {
            input_ids: &ids_padded,
            attention_mask: &mask_padded,
            token_type_ids: None,
            images: None,
        })
        .unwrap();
    let v_padded = to_vec(&out_padded.embeddings);
    for (a, b) in v_single.iter().zip(v_padded.iter()) {
        assert!((a - b).abs() < 1e-5, "padding leaked: {a} vs {b}");
    }

    // 2. Bidirectionality: changing token 2 changes token 0's hidden representation
    let ids_alt = mlxcel_core::from_slice_i32(&[3, 7, 29], &[1, 3]);
    let h1 = model
        .forward_multimodal(&ids_single, &mask_single, None, None)
        .unwrap();
    let h2 = model
        .forward_multimodal(&ids_alt, &mask_single, None, None)
        .unwrap();
    let h1_vec = to_vec(&h1);
    let h2_vec = to_vec(&h2);
    let tok0_diff: f32 = h1_vec[..24]
        .iter()
        .zip(&h2_vec[..24])
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        tok0_diff > 1e-4,
        "expected bidirectional attention to affect token 0 when token 2 changes, diff={tok0_diff}"
    );

    // 3. Matryoshka truncation + L2 normalization
    for dim in [24usize, 16, 8] {
        let trunc = truncate_dimensions(&out_single.embeddings, dim);
        let normed = normalize_l2(&trunc);
        let vals = to_vec(&normed);
        assert_eq!(vals.len(), dim);
        let norm_sq: f32 = vals.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-5,
            "norm^2={norm_sq} at dim={dim}"
        );
    }
}

#[test]
fn sliding_window_mask_blocks_tokens_beyond_window() {
    let _guard = mlx_test_guard();
    // 1-layer sliding-only model with sliding_window = 1 (allows |q - k| <= 1)
    let raw_cfg = synthetic_config(1, 1);
    let parsed_cfg = EmbeddingGemma2Config::from_json(&raw_cfg, None).expect("config parses");
    assert!(parsed_cfg.layers[0].is_sliding);
    let mut weights = build_synthetic_weights(&parsed_cfg);
    let model = EmbeddingGemma2Model::from_weights(&mut weights, &raw_cfg, PoolingMode::Mean, true)
        .unwrap();

    let mask = mlxcel_core::from_slice_i32(&[1, 1, 1, 1], &[1, 4]);
    let ids_a = mlxcel_core::from_slice_i32(&[2, 5, 8, 11], &[1, 4]);
    let ids_b = mlxcel_core::from_slice_i32(&[2, 5, 8, 42], &[1, 4]);

    let h_a = to_vec(&model.forward_multimodal(&ids_a, &mask, None, None).unwrap());
    let h_b = to_vec(&model.forward_multimodal(&ids_b, &mask, None, None).unwrap());

    // Token 0 is at distance 3 from token 3 (> sliding_window=1), so in a 1-layer
    // sliding model token 0's output must be unaffected by changing token 3.
    for i in 0..24 {
        assert!(
            (h_a[i] - h_b[i]).abs() < 1e-6,
            "sliding window leaked from pos 3 to pos 0 at dim {i}: {} vs {}",
            h_a[i],
            h_b[i]
        );
    }
}

#[test]
fn expand_image_tokens_text_only_and_vision_paths() {
    let raw_cfg = synthetic_config(2, 2);
    let parsed_cfg = EmbeddingGemma2Config::from_json(&raw_cfg, None).expect("config parses");
    let mut weights = build_synthetic_weights(&parsed_cfg);
    let model = EmbeddingGemma2Model::from_weights(&mut weights, &raw_cfg, PoolingMode::Mean, true)
        .unwrap();

    let dummy_img = image::DynamicImage::new_rgb8(32, 32);
    let images = [ImageInput { image: dummy_img }];

    // Text-only model (no vision tower loaded): expand_image_tokens returns ids unchanged
    let ids = vec![101u32, 202u32];
    let out = model.expand_image_tokens(&ids, &images).unwrap();
    assert_eq!(out, ids);
    assert_eq!(model.format_text("hello", None), "hello");
    assert_eq!(model.format_text("", None), "");
}

#[test]
fn short_sequence_sliding_window_bypass_matches_explicit_window_mask() {
    let _guard = mlx_test_guard();
    // sliding_window = 10, input length = 4 (4 <= 10 + 1)
    let raw_cfg = synthetic_config(2, 10);
    let parsed_cfg = EmbeddingGemma2Config::from_json(&raw_cfg, None).expect("config parses");
    let mut weights = build_synthetic_weights(&parsed_cfg);
    let model = EmbeddingGemma2Model::from_weights(&mut weights, &raw_cfg, PoolingMode::Mean, true)
        .unwrap();

    let ids = mlxcel_core::from_slice_i32(&[2, 5, 8, 11], &[1, 4]);
    let mask = mlxcel_core::from_slice_i32(&[1, 1, 1, 1], &[1, 4]);
    let out = model.forward_multimodal(&ids, &mask, None, None).unwrap();
    let v_out = to_vec(&out);
    assert_eq!(v_out.len(), 4 * 24);

    // Verify empty sequence handling returns clean zero tensor without panicking
    let empty_ids = mlxcel_core::from_slice_i32(&[], &[1, 0]);
    let empty_mask = mlxcel_core::from_slice_i32(&[], &[1, 0]);
    let empty_out = model
        .forward_multimodal(&empty_ids, &empty_mask, None, None)
        .unwrap();
    assert_eq!(mlxcel_core::array_shape(&empty_out), vec![1, 0, 24]);
}

#[test]
fn config_bounds_guard_against_malformed_nonpositive_values() {
    let bad_cfg = json!({
        "text_config": {
            "vocab_size": -10,
            "hidden_size": 0,
            "intermediate_size": -50,
            "num_hidden_layers": 0,
            "num_attention_heads": -4,
            "num_key_value_heads": 0,
            "head_dim": -128,
            "hidden_size_per_layer_input": -1
        }
    });
    let cfg = EmbeddingGemma2Config::from_json(&bad_cfg, None).expect("config parses");
    assert!(cfg.vocab_size >= 1);
    assert!(cfg.hidden_size >= 1);
    assert!(cfg.intermediate_size >= 1);
    assert!(cfg.num_hidden_layers >= 1);
    assert_eq!(cfg.sliding_window, 512);
    assert!(cfg.ple_dim >= 1);
    assert_eq!(cfg.layers.len(), 1);
    assert!(cfg.layers[0].num_attention_heads >= 1);
    assert!(cfg.layers[0].num_key_value_heads >= 1);
    assert!(cfg.layers[0].head_dim >= 1);
}

// Issue #2798: the sliding window is validated, not clamped, before the
// bidirectional band reads it.
#[test]
fn config_rejects_a_sliding_window_that_cannot_back_the_band() {
    for window in [json!(0), json!(-512), json!(i64::from(i32::MAX) + 1)] {
        let cfg = json!({ "text_config": { "sliding_window": window } });
        let err = EmbeddingGemma2Config::from_json(&cfg, None)
            .expect_err("an unusable sliding window must fail the parse");
        assert!(err.contains("EmbeddingGemma 2"), "{err}");
        assert!(err.contains("sliding_window"), "{err}");
    }
    let all_global = json!({
        "text_config": {
            "sliding_window": 0,
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "full_attention"]
        }
    });
    let cfg = EmbeddingGemma2Config::from_json(&all_global, None)
        .expect("an all-global config keeps any unused window");
    assert_eq!(cfg.sliding_window, 0);
}

#[test]
#[ignore = "requires local mlx-community/embeddinggemma-2-bf16 or 4bit checkpoint"]
fn real_checkpoint_embeddinggemma2_parity_and_ranking() {
    use crate::embeddings::{EmbedOptions, EmbeddingEngine, load_embedding_model};
    use crate::models::embedding_test_support::{cosine, local_checkpoint};

    let _guard = mlx_test_guard();
    let Some(path) = local_checkpoint("mlx-community/embeddinggemma-2-bf16")
        .or_else(|| local_checkpoint("mlx-community/embeddinggemma-2-4bit"))
        .or_else(|| local_checkpoint("google/embeddinggemma-2"))
    else {
        return;
    };

    let loaded = load_embedding_model(&path).unwrap();
    assert_eq!(loaded.model_type, ModelType::EmbeddingGemma2);
    assert_eq!(loaded.limits.dim, 768);
    let engine = EmbeddingEngine::new(loaded, 8);

    let reply = engine
        .embed_texts(
            &[
                "task: search result | query: Which planet is known as the Red Planet?".to_string(),
                "title: none | text: Venus is the second planet from the Sun.".to_string(),
                "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.".to_string(),
                "title: Baking | text: Sourdough bread requires flour, water, salt, and fermentation.".to_string(),
            ],
            &EmbedOptions::default(),
        )
        .unwrap();
    let vecs: Vec<Vec<f32>> = reply
        .vectors
        .into_iter()
        .map(|v| {
            assert!(!v.is_multi_vector(), "expected single-vector output");
            v.values
        })
        .collect();
    assert_eq!(vecs[0].len(), 768);
    let query = &vecs[0];
    let venus_sim = cosine(query, &vecs[1]);
    let mars_sim = cosine(query, &vecs[2]);
    let baking_sim = cosine(query, &vecs[3]);

    // Reference validation numbers from validation.json:
    // retrieval_scores_venus_mars: [0.6774646639823914, 0.8348115086555481]
    assert!(
        (venus_sim - 0.6775).abs() < 0.05,
        "expected venus similarity ~0.6775, got {venus_sim}"
    );
    assert!(
        (mars_sim - 0.8348).abs() < 0.05,
        "expected mars similarity ~0.8348, got {mars_sim}"
    );
    assert!(
        mars_sim > venus_sim + 0.10,
        "expected positive document similarity ({mars_sim}) > negative ({venus_sim})"
    );
    assert!(
        venus_sim > baking_sim,
        "expected venus ({venus_sim}) > baking ({baking_sim})"
    );
}
