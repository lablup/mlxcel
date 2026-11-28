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

use crate::loading::special::try_load_special_model_from_weights;
use crate::models::embedding_test_support::{Rng, ids_array, mlx_test_guard, temp_dir, to_vec};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use serde_json::{Value, json};

const PUBLISHED: &str = include_str!("../../tests/fixtures/lfm3/config.json");

fn tiny_config() -> Value {
    json!({"model_type":"lfm3","hidden_size":16,"num_attention_heads":2,"num_key_value_heads":1,
        "num_hidden_layers":4,"vocab_size":32,"intermediate_size":24,"sliding_window":16,
        "layer_types":["conv","conv","full_attention","sliding_attention"],"conv_L_cache":3})
}

fn load(config: &Value, mut weights: WeightMap) -> Result<crate::LoadedModel, String> {
    let dir = temp_dir("lfm3");
    let config = config.to_string();
    std::fs::write(dir.join("config.json"), &config).unwrap();
    let result = crate::models::get_model_type(&dir)
        .and_then(|kind| try_load_special_model_from_weights(kind, &dir, &config, &mut weights));
    std::fs::remove_dir_all(dir).unwrap();
    result
        .map_err(|e| format!("{e:#}"))?
        .ok_or_else(|| "no weight loader".into())
}

#[test]
fn published_config_reaches_weight_loader() {
    let error = load(&serde_json::from_str(PUBLISHED).unwrap(), WeightMap::new())
        .err()
        .unwrap();
    assert!(error.contains("embed_tokens"), "{error}");
}

#[test]
fn rejects_invalid_layer_and_rope_config_before_loading_weights() {
    for (key, value, expected) in [
        ("layer_types", Value::Null, "layer_types"),
        ("layer_types", json!(["conv"]), "num_hidden_layers"),
        (
            "layer_types",
            json!(["conv", "bogus", "conv", "conv"]),
            "bogus",
        ),
        ("sliding_window", json!(0), "sliding_window"),
        ("sliding_window", json!(-1), "sliding_window"),
        (
            "rope_parameters",
            json!({"full_attention":{"rope_type":"linear"},"sliding_attention":{}}),
            "rope_type",
        ),
        ("num_attention_heads", json!(0), "num_attention_heads"),
        ("conv_L_cache", json!(0), "conv_L_cache"),
    ] {
        let mut config = tiny_config();
        if value.is_null() {
            config.as_object_mut().unwrap().remove(key);
        } else {
            config[key] = value;
        }
        let error = load(&config, WeightMap::new()).err().unwrap();
        assert!(error.contains(expected), "{key}: {error}");
    }
}

#[test]
fn null_rope_parameters_use_the_default_periods_and_rotary_factors() {
    let mut config = tiny_config();
    config["rope_parameters"] = Value::Null;
    let args: super::Lfm3Args = serde_json::from_value(config).unwrap();
    args.validate().unwrap();
    let full = &args.rope_parameters["full_attention"];
    let sliding = &args.rope_parameters["sliding_attention"];
    assert_eq!((full.rope_theta, full.partial_rotary_factor), (1e6, 0.5));
    assert_eq!(
        (sliding.rope_theta, sliding.partial_rotary_factor),
        (1e4, 1.0)
    );
}

#[test]
fn int8_attention_skips_snapshot_donation() {
    let _guard = mlx_test_guard();
    let model = load(&tiny_config(), weights(false)).unwrap();
    model.set_kv_cache_layer_modes(vec![mlxcel_core::cache::KVCacheMode::Int8; 4]);
    let seq = SequenceId::from_raw(17);
    model.prepare_sequence_state(seq);
    assert!(model.snapshot_sequence_state(seq, 0).is_some());
    let logits = model.forward_with_sequence_id(&ids_array(&[1], 1, 1), Some(seq), &mut [], None);
    mlxcel_core::eval(&logits);
    assert!(model.snapshot_sequence_state(seq, 1).is_none());
}

fn weights(converted: bool) -> WeightMap {
    let mut rng = Rng::new(3206);
    let mut map = WeightMap::new();
    let mut add = |key: String, shape: &[i32]| {
        let mut w = rng.tensor(shape, 0.5);
        if converted && key.ends_with("conv.conv.weight") {
            w = mlxcel_core::transpose_axes(&w, &[0, 2, 1]);
        }
        let name = if converted {
            format!("language_model.{key}")
        } else {
            key
        };
        map.insert(name, w);
    };
    add("model.embed_tokens.weight".into(), &[32, 16]);
    add("model.embedding_norm.weight".into(), &[16]);
    add("lm_head.weight".into(), &[32, 16]);
    for i in 0..4 {
        let mut layer =
            |name: &str, shape: &[i32]| add(format!("model.layers.{i}.{name}.weight"), shape);
        for name in [
            "operator_norm",
            "ffn_norm",
            "post_ffn_norm",
            "post_operator_norm",
        ] {
            layer(name, &[16]);
        }
        layer("feed_forward.w1", &[24, 16]);
        layer("feed_forward.w3", &[24, 16]);
        layer("feed_forward.w2", &[16, 24]);
        if i < 2 {
            layer("conv.in_proj", &[48, 16]);
            layer("conv.out_proj", &[16, 16]);
            layer("conv.conv", &[16, 1, 3]);
        } else {
            for (name, shape) in [
                ("q_proj", vec![16, 16]),
                ("k_proj", vec![8, 16]),
                ("v_proj", vec![8, 16]),
                ("o_proj", vec![16, 16]),
                ("gate_proj", vec![2, 16]),
                ("q_norm", vec![8]),
                ("k_norm", vec![8]),
            ] {
                layer(&format!("self_attn.{name}"), &shape);
            }
        }
    }
    map
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() <= tolerance, "index {i}: {a} vs {b}");
    }
}

#[test]
fn wrapped_chunks_decode_and_snapshot_match_full_windowed_prefill() {
    use super::{Cache, LayerCache, Lfm3Model};
    use mlxcel_core::layers::KVCache;
    use mlxcel_core::utils::create_causal_mask_with_window_full;

    let _guard = mlx_test_guard();
    let args = serde_json::from_value(tiny_config()).unwrap();
    let reference = Lfm3Model::from_weights(args, weights(false)).unwrap();
    let converted = load(&tiny_config(), weights(true)).unwrap();
    let ids: Vec<i32> = (0..301).map(|i| (i * 13 + i / 7) % 32).collect();
    let all = ids_array(&ids, 1, 301);
    // Keep all keys in ordinary dense caches. Only the explicit band limits
    // the last layer during this forward pass.
    let band = create_causal_mask_with_window_full(301, 0, Some(16));
    let mut h = reference.embed.forward(&all);
    for (idx, layer) in reference.layers.iter().enumerate() {
        let mut cache = if idx < 2 {
            LayerCache::Conv(None)
        } else {
            LayerCache::Attention(Cache::Standard(KVCache::new()))
        };
        h = layer.forward(&h, &mut cache, (idx == 3).then_some(&*band));
    }
    let expected = to_vec(&reference.embed.as_linear(&reference.final_norm.forward(&h)));
    let seq = SequenceId::from_raw(11);
    converted.prepare_sequence_state(seq);
    let empty = converted.snapshot_sequence_state(seq, 0).unwrap();
    converted.restore_sequence_state(seq, &empty).unwrap();
    let mut start = 0;
    for end in [1, 64, 300, 301] {
        let chunk = ids_array(&ids[start..end], 1, (end - start) as i32);
        let logits = converted.forward_with_sequence_id(&chunk, Some(seq), &mut [], None);
        close(&to_vec(&logits), &expected[start * 32..end * 32], 5e-4);
        if end == 300 {
            let snapshot = converted.snapshot_sequence_state(seq, end).unwrap();
            converted.release_sequence_state_by_id(seq);
            converted.restore_sequence_state(seq, &snapshot).unwrap();
        }
        start = end;
    }
}

#[test]
fn identified_batched_rows_keep_independent_histories() {
    let _guard = mlx_test_guard();
    let model = load(&tiny_config(), weights(false)).unwrap();
    let ids = [SequenceId::from_raw(11), SequenceId::from_raw(23)];
    let histories = [vec![2, 3], vec![4, 7, 9, 2, 5]];
    let mut expected = Vec::new();
    for (idx, history) in histories.iter().enumerate() {
        let input = ids_array(history, 1, history.len() as i32);
        model.prepare_sequence_state(ids[idx]);
        let _ = model.forward_with_sequence_id(&input, Some(ids[idx]), &mut [], None);
        let single = load(&tiny_config(), weights(false)).unwrap();
        let mut all = history.clone();
        all.push(8 + idx as i32);
        expected.push(to_vec(
            &single
                .last_row_logits(
                    &ids_array(&all, 1, all.len() as i32),
                    None,
                    None,
                    &mut [],
                    None,
                    all.len() - 1,
                )
                .unwrap(),
        ));
    }
    let rows = model.forward_batched_with_ids(
        &ids_array(&[9, 8], 2, 1),
        Some(&[ids[1], ids[0]]),
        &mut [&mut [], &mut []],
        None,
    );
    let actual = to_vec(&rows);
    close(&actual[..32], &expected[1], 5e-4);
    close(&actual[32..], &expected[0], 5e-4);
}

#[test]
fn published_schedule_and_rotary_pairs() {
    use super::{Attention, Lfm3Args};
    let _guard = mlx_test_guard();
    let args: Lfm3Args = serde_json::from_str(PUBLISHED).unwrap();
    let schedule = args.validate().unwrap();
    for (kind, count) in [
        ("conv", 13),
        ("full_attention", 6),
        ("sliding_attention", 5),
    ] {
        assert_eq!(
            schedule.iter().filter(|t| t.as_str() == kind).count(),
            count
        );
    }
    let args = Lfm3Args {
        hidden_size: 128,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        ..args
    };
    let mut rng = Rng::new(3206);
    let mut w = WeightMap::new();
    for (name, shape) in [
        ("q_proj", vec![128, 128]),
        ("k_proj", vec![128, 128]),
        ("v_proj", vec![128, 128]),
        ("o_proj", vec![128, 128]),
        ("gate_proj", vec![1, 128]),
        ("q_norm", vec![128]),
        ("k_norm", vec![128]),
    ] {
        rng.insert(&mut w, &format!("a.{name}.weight"), &shape, 0.5);
    }
    let input = rng.values(128, 1.);
    let x = mlxcel_core::from_slice_f32(&input, &[1, 1, 1, 128]);
    for (kind, angles, base) in [
        ("full_attention", 32, 1e6_f32),
        ("sliding_attention", 64, 1e4_f32),
    ] {
        let attn = Attention::new(&w, &args, "a", &args.rope_parameters[kind]).unwrap();
        let out = to_vec(&attn.rotate(&x, 7));
        let mut expected = input.clone();
        for i in 0..angles {
            let angle = 7. * base.powf(-2. * i as f32 / 128.);
            let (s, c) = angle.sin_cos();
            expected[i] = input[i] * c - input[i + 64] * s;
            expected[i + 64] = input[i + 64] * c + input[i] * s;
        }
        close(&out, &expected, 2e-6);
    }
}

#[test]
fn single_token_gate_and_sandwich_norm_match_scalar_equations() {
    use super::{Cache, Layer, LayerCache, Lfm3Args};
    let _guard = mlx_test_guard();
    let w = weights(false);
    let tensor = |key: &str| to_vec(&w[&format!("model.layers.2.{key}.weight")]);
    let linear = |key: &str, x: &[f32]| {
        tensor(key)
            .chunks_exact(x.len())
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>())
            .collect::<Vec<_>>()
    };
    let norm = |key: &str, x: &[f32]| {
        let denom = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5).sqrt();
        tensor(key)
            .iter()
            .zip(x)
            .map(|(w, v)| w * (v / denom))
            .collect::<Vec<_>>()
    };
    let x = Rng::new(999).values(16, 1.);
    let h = norm("operator_norm", &x);
    // One causal key: softmax is 1 for each query head, both using the same value head.
    let value = linear("self_attn.v_proj", &h);
    let gate = linear("self_attn.gate_proj", &h);
    let gated: Vec<f32> = gate
        .iter()
        .flat_map(|g| value.iter().map(move |v| v / (1. + (-g).exp())))
        .collect();
    for sandwich in [true, false] {
        let mut args: Lfm3Args = serde_json::from_value(tiny_config()).unwrap();
        args.use_sandwich_norm = sandwich;
        let layer = Layer::new(&w, &args, 2, "full_attention").unwrap();
        let mut h = linear("self_attn.o_proj", &gated);
        if sandwich {
            h = norm("post_operator_norm", &h);
        }
        let residual: Vec<_> = x.iter().zip(h).map(|(a, b)| a + b).collect();
        let ff = norm("ffn_norm", &residual);
        let activated: Vec<_> = linear("feed_forward.w1", &ff)
            .into_iter()
            .zip(linear("feed_forward.w3", &ff))
            .map(|(a, b)| a / (1. + (-a).exp()) * b)
            .collect();
        let mut h = linear("feed_forward.w2", &activated);
        if sandwich {
            h = norm("post_ffn_norm", &h);
        }
        let expected: Vec<_> = residual.iter().zip(h).map(|(a, b)| a + b).collect();
        let mut cache = LayerCache::Attention(Cache::Standard(mlxcel_core::layers::KVCache::new()));
        let actual = layer.forward(
            &mlxcel_core::from_slice_f32(&x, &[1, 1, 16]),
            &mut cache,
            None,
        );
        close(&to_vec(&actual), &expected, 2e-5);
    }
}

#[test]
fn untied_head_and_tied_sanitize() {
    let _guard = mlx_test_guard();
    let mut config = tiny_config();
    config["tie_word_embeddings"] = json!(false);
    let mut w = weights(false);
    w.remove("lm_head.weight");
    let error = load(&config, w).err().unwrap();
    assert!(error.contains("lm_head.weight"), "{error}");
    let mut w = weights(true);
    w.insert(
        "language_model.lm_head.weight".into(),
        mlxcel_core::zeros(&[32, 16], mlxcel_core::dtype::FLOAT32),
    );
    let untied = load(&config, w).unwrap();
    let id = ids_array(&[2], 1, 1);
    assert_eq!(to_vec(&untied.forward(&id, &mut [], None)), vec![0.; 32]);
    config["tie_word_embeddings"] = json!(true);
    let mut w = weights(false);
    w.remove("lm_head.weight");
    let tied = load(&config, w).unwrap();
    let out = to_vec(&tied.forward(&id, &mut [], None));
    assert!(out.iter().any(|v| v.abs() > 0.1));
    close(
        &out,
        &to_vec(
            &load(&config, weights(true))
                .unwrap()
                .forward(&id, &mut [], None),
        ),
        1e-6,
    );
}

#[derive(serde::Deserialize)]
struct Prompt {
    ids: Vec<i32>,
    greedy: Vec<i32>,
}

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lfm3")
        .join(name)
}
fn prompts() -> Vec<Prompt> {
    serde_json::from_slice(&std::fs::read(fixture("prompts.json")).unwrap()).unwrap()
}

#[test]
#[ignore = "requires the original LiquidAI/LFM3-1.7B checkpoint"]
fn real_f32_last_position_logits() {
    let _guard = mlx_test_guard();
    let dir =
        crate::models::embedding_test_support::local_checkpoint("LiquidAI/LFM3-1.7B").unwrap();
    let args = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let mut weights = mlxcel_core::weights::load_weights_from_dir(dir).unwrap();
    for tensor in weights.values_mut() {
        *tensor = mlxcel_core::astype(tensor, mlxcel_core::dtype::FLOAT32);
    }
    let model = super::Lfm3Model::from_weights(args, weights).unwrap();
    let refs = mlxcel_core::weights::load_safetensors(fixture("last_logits.safetensors")).unwrap();
    for (idx, prompt) in prompts().iter().enumerate() {
        let ids = ids_array(&prompt.ids, 1, prompt.ids.len() as i32);
        let mut caches = model.make_caches();
        let actual = to_vec(
            &model
                .last_row_logits(&ids, None, None, &mut caches, None, prompt.ids.len() - 1)
                .unwrap(),
        );
        let expected = to_vec(&refs[&idx.to_string()]);
        assert!(actual.iter().all(|v| v.is_finite()));
        let error = crate::models::embedding_test_support::max_abs_diff(&actual, &expected);
        let scale = expected.iter().map(|v| v.abs()).fold(0_f32, f32::max);
        eprintln!(
            "prompt {idx}: {} tokens, abs max {error}, relative infinity norm {}",
            prompt.ids.len(),
            error / scale
        );
        assert!(error <= 1e-4 * scale);
    }
}

#[test]
#[ignore = "requires LiquidAI/LFM3-1.7B-MLX-8bit"]
fn real_eight_bit_greedy_tokens() {
    let _guard = mlx_test_guard();
    let dir =
        crate::models::embedding_test_support::local_checkpoint("LiquidAI/LFM3-1.7B-MLX-8bit")
            .unwrap();
    let (model, _) = super::Lfm3Model::load(dir.to_str().unwrap()).unwrap();
    for (idx, prompt) in prompts().iter().enumerate() {
        let mut caches = model.make_caches();
        let mut ids = ids_array(&prompt.ids, 1, prompt.ids.len() as i32);
        let mut generated = Vec::new();
        for step in 0..32 {
            let row = if step == 0 { prompt.ids.len() - 1 } else { 0 };
            let logits = model
                .last_row_logits(&ids, None, None, &mut caches, None, row)
                .unwrap();
            let next = mlxcel_core::item_i32(&mlxcel_core::argmax(&logits, -1, false));
            generated.push(next);
            ids = ids_array(&[next], 1, 1);
        }
        eprintln!("prompt {idx}: {generated:?}");
        assert_eq!(generated, prompt.greedy, "prompt {idx}");
    }
}

#[test]
#[ignore = "requires LiquidAI/LFM3-1.7B-MLX-8bit"]
fn real_chunked_prefill_across_many_window_wraps() {
    let _guard = mlx_test_guard();
    let dir =
        crate::models::embedding_test_support::local_checkpoint("LiquidAI/LFM3-1.7B-MLX-8bit")
            .unwrap();
    let (model, _) = super::Lfm3Model::load(dir.to_str().unwrap()).unwrap();
    let prompts = prompts();
    let tail = &prompts[0].ids;
    let mut ids: Vec<_> = prompts[2]
        .ids
        .iter()
        .copied()
        .cycle()
        .take(4826 - tail.len())
        .collect();
    ids.extend_from_slice(tail);
    let mut expected = None;
    for size in [64, 300, 2048] {
        let mut caches = model.make_caches();
        let mut prediction = None;
        for chunk in ids.chunks(size) {
            let tokens = ids_array(chunk, 1, chunk.len() as i32);
            let logits = model
                .last_row_logits(&tokens, None, None, &mut caches, None, chunk.len() - 1)
                .unwrap();
            prediction = Some(mlxcel_core::item_i32(&mlxcel_core::argmax(
                &logits, -1, false,
            )));
        }
        let prediction = prediction.unwrap();
        eprintln!("4826-token prefill, chunk {size}: {prediction}");
        assert_eq!(*expected.get_or_insert(prediction), prediction);
    }
}

#[test]
fn kv_estimate_counts_only_attention_and_caps_sliding_layers() {
    use crate::execution::kv_arch::estimate_kv_arch_from_config;
    let cfg = serde_json::from_str(PUBLISHED).unwrap();
    let estimate =
        estimate_kv_arch_from_config(&cfg, 8192, mlxcel_core::cache::KVCacheMode::Fp16, 1).unwrap();
    assert_eq!(estimate.total_bytes, (6 * 8192 + 5 * 256) * 2 * 4 * 128 * 2);
    assert_eq!(estimate.marginal_bytes_per_token, 6 * 2 * 4 * 128 * 2);
}

#[test]
fn validates_large_layer_schedule_and_rope_boundaries() {
    let mut args: super::Lfm3Args = serde_json::from_value(tiny_config()).unwrap();
    args.num_hidden_layers = 200_000;
    args.layer_types = Some(vec!["full_attention".into(); args.num_hidden_layers]);
    assert_eq!(args.validate().unwrap().len(), 200_000);
    args.layer_types
        .as_mut()
        .unwrap()
        .last_mut()
        .unwrap()
        .clone_from(&"unknown".to_owned());
    assert!(args.validate().unwrap_err().contains("unknown"));
    let mut args: super::Lfm3Args = serde_json::from_value(tiny_config()).unwrap();
    args.sliding_window = Some(1);
    args.conv_l_cache = 1;
    assert!(args.validate().is_ok());
    for factor in [0., 0.01, 1.01, f32::MAX] {
        args.rope_parameters
            .get_mut("full_attention")
            .unwrap()
            .partial_rotary_factor = factor;
        assert!(
            args.validate()
                .unwrap_err()
                .contains("partial_rotary_factor")
        );
    }
    args.hidden_size = i32::MAX;
    assert!(args.validate().unwrap_err().contains("hidden_size"));
}
