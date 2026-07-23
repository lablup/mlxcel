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

//! Pinned real-checkpoint parity for Phi-4 Multimodal audio.
//!
//! The fixture values were captured from the official Transformers remote code
//! at revision `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`, using bf16 eager
//! attention and greedy decoding. Run on a host with the pinned checkpoint:
//!
//! ```text
//! MLXCEL_PHI4MM_MODEL=/path/to/checkpoint \
//! MLXCEL_PHI4MM_REVISION=93f923e1a7727d1c4f446756212d9d3e8fcc5d81 \
//! cargo test --release --features cuda --test phi4mm_audio_parity -- --nocapture
//! ```

use std::path::PathBuf;

use mlxcel::{LanguageModel, LoadedModel};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/phi4mm_audio_parity.json"))
        .expect("valid Phi4MM parity fixture")
}

fn read_flat_prefix(value: &mlxcel_core::MlxArray, count: usize) -> Vec<f32> {
    let value = mlxcel_core::astype(value, mlxcel_core::dtype::FLOAT32);
    let flat = mlxcel_core::reshape(&value, &[-1]);
    mlxcel_core::eval(&flat);
    (0..count)
        .map(|index| {
            mlxcel_core::item_f32(&mlxcel_core::slice(
                &flat,
                &[index as i32],
                &[index as i32 + 1],
            ))
        })
        .collect()
}

fn read_3d(value: &mlxcel_core::MlxArray, row: i32, column: i32) -> f32 {
    let value = mlxcel_core::astype(value, mlxcel_core::dtype::FLOAT32);
    let item = mlxcel_core::slice(&value, &[0, row, column], &[1, row + 1, column + 1]);
    mlxcel_core::eval(&item);
    mlxcel_core::item_f32(&item)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}]={actual}, expected {expected} +/- {tolerance}"
        );
    }
}

fn argmax_last(logits: &mlxcel_core::MlxArray) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let last = mlxcel_core::slice(logits, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
    let token = mlxcel_core::argmax_last_axis(&last);
    mlxcel_core::eval(&token);
    mlxcel_core::item_i32(&token)
}

#[test]
fn pinned_phi4mm_audio_intermediates_kv_and_greedy_tokens_match() {
    let Ok(model_dir) = std::env::var("MLXCEL_PHI4MM_MODEL") else {
        eprintln!("Skipping Phi4MM audio parity: MLXCEL_PHI4MM_MODEL is not set");
        return;
    };
    let fixture = fixture();
    let expected_revision = fixture["revision"].as_str().unwrap();
    assert_eq!(
        std::env::var("MLXCEL_PHI4MM_REVISION").as_deref(),
        Ok(expected_revision),
        "set MLXCEL_PHI4MM_REVISION to attest the immutable checkpoint snapshot"
    );

    let model_dir = PathBuf::from(model_dir);
    let (loaded, _tokenizer) = mlxcel::load_model(&model_dir).expect("load pinned Phi4MM");
    let LoadedModel::Phi4MMVLM(model) = loaded else {
        panic!("fixture checkpoint must load as Phi4MMVLM");
    };

    let audio_path = model_dir.join(fixture["audio"].as_str().unwrap());
    let (samples, sample_rate) =
        mlxcel::audio::load_wav_file(&audio_path).expect("load fixture WAV");
    let batch = model
        .extract_audio(&[(samples, sample_rate)])
        .expect("extract pinned audio features");
    let expected_shape: Vec<i32> = fixture["feature_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(mlxcel_core::array_shape(&batch.clips[0]), expected_shape);
    assert_eq!(
        batch.embed_sizes,
        vec![fixture["audio_embed_size"].as_u64().unwrap() as usize]
    );
    for probe in fixture["feature_probes"].as_array().unwrap() {
        let probe = probe.as_array().unwrap();
        let row = probe[0].as_i64().unwrap() as i32;
        let column = probe[1].as_i64().unwrap() as i32;
        let expected = probe[2].as_f64().unwrap() as f32;
        let actual = read_3d(&batch.clips[0], row, column);
        assert!(
            (actual - expected).abs() <= 2e-3,
            "feature[{row},{column}]={actual}, expected {expected}"
        );
    }

    let encoded = model
        .audio_encoder
        .forward(&batch.clips[0])
        .expect("encode audio");
    let expected_encoder: Vec<f32> = fixture["encoder_first"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    assert_close(
        &read_flat_prefix(&encoded, expected_encoder.len()),
        &expected_encoder,
        0.025,
        "encoder",
    );

    let projected = model.audio_projection.forward(&encoded, false);
    let expected_projection: Vec<f32> = fixture["projection_first"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    assert_close(
        &read_flat_prefix(&projected, expected_projection.len()),
        &expected_projection,
        0.025,
        "projection",
    );

    let input_ids: Vec<i32> = fixture["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap() as i32)
        .collect();
    let input = mlxcel_core::from_slice_i32(&input_ids, &[1, input_ids.len() as i32]);
    let embeddings = model
        .get_input_embeddings_with_audio(&input, &[], &batch)
        .expect("merge pinned audio embeddings");
    let mut caches = LanguageModel::make_caches(&model);
    let logits = LanguageModel::forward_with_embeddings(
        &model,
        &input,
        Some(&embeddings.inputs_embeds),
        &mut caches,
        None,
    );
    mlxcel_core::eval(&logits);
    assert!(
        caches
            .iter()
            .all(|cache| cache.offset == input_ids.len() as i32)
    );
    let expected_kv_shape: Vec<i32> = fixture["kv_buffer_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap() as i32)
        .collect();
    assert!(caches.iter().all(|cache| {
        cache
            .keys
            .as_ref()
            .is_some_and(|keys| mlxcel_core::array_shape(keys) == expected_kv_shape)
            && cache
                .values
                .as_ref()
                .is_some_and(|values| mlxcel_core::array_shape(values) == expected_kv_shape)
    }));

    let expected_tokens: Vec<i32> = fixture["greedy_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap() as i32)
        .collect();
    let first = argmax_last(&logits);
    assert_eq!(first, expected_tokens[0]);
    let next = mlxcel_core::from_slice_i32(&[first], &[1, 1]);
    let logits = LanguageModel::forward(&model, &next, &mut caches, None);
    mlxcel_core::eval(&logits);
    assert_eq!(argmax_last(&logits), expected_tokens[1]);
    assert!(
        caches
            .iter()
            .all(|cache| cache.offset == input_ids.len() as i32 + 1)
    );
}
