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

use mlxcel_core::utils::array_to_vec_f32;

use super::*;

fn tiny_config() -> InklingAudioConfig {
    InklingAudioConfig {
        model_type: "inkling_audio".into(),
        n_mel_bins: 3,
        mel_vocab_size: 4,
        text_hidden_size: 2,
        rms_norm_eps: 1e-6,
        max_frames_per_chunk: 1,
    }
}

#[test]
fn tower_sums_channels_then_norms_across_chunks() {
    let config = tiny_config();
    let mut table = vec![0.0f32; 12 * 2];
    for row in 0..12 {
        table[row * 2] = row as f32 + 1.0;
        table[row * 2 + 1] = 2.0 * row as f32 + 0.5;
    }
    let mut weights = WeightMap::new();
    weights.insert(
        "audio_tower.embed_audio_tokens.weight".into(),
        mlxcel_core::from_slice_f32(&table, &[12, 2]),
    );
    weights.insert(
        "audio_tower.norm.weight".into(),
        mlxcel_core::from_slice_f32(&[1.0, 0.5], &[2]),
    );
    let tower = InklingAudioTower::from_weights(&weights, &config, 64, 4).unwrap();
    let ids = mlxcel_core::from_slice_i32(&[0, 0, 0, 1, 2, 3], &[2, 3]);
    let actual = tower.forward(&ids).unwrap();
    let actual = array_to_vec_f32(&actual);

    let rows = [[0usize, 4, 8], [1usize, 6, 11]];
    let mut expected = Vec::new();
    for selected in rows {
        let x0: f32 = selected.iter().map(|row| table[row * 2]).sum();
        let x1: f32 = selected.iter().map(|row| table[row * 2 + 1]).sum();
        let rms = ((x0 * x0 + x1 * x1) * 0.5 + config.rms_norm_eps).sqrt();
        expected.push(x0 / rms);
        expected.push(x1 / rms * 0.5);
    }
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "value {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn request_ceiling_joins_6000_chunks_once_in_order() {
    let config = tiny_config();
    let frames = 6_000usize;
    assert_eq!(config.max_frames_per_chunk, 1);
    let chunks: Vec<Vec<usize>> = (0..frames)
        .step_by(config.max_frames_per_chunk)
        .map(|frame| vec![frame])
        .collect();
    let mut calls = 0usize;
    let mut arity = 0usize;
    let output = join_chunks(chunks, |chunks| {
        calls += 1;
        arity = chunks.len();
        Ok::<_, String>(chunks.iter().flatten().copied().collect::<Vec<_>>())
    })
    .unwrap()
    .unwrap();

    assert_eq!(calls, 1, "the tower must build one final concatenate node");
    assert_eq!(
        arity, 6_000,
        "all request-ceiling chunks must share that node"
    );
    assert_eq!(output, (0..frames).collect::<Vec<_>>());
}

#[test]
fn concatenate_many_preserves_256_array_cardinality_and_order() {
    let chunks: Vec<UniquePtr<MlxArray>> = (0..256)
        .map(|value| mlxcel_core::from_slice_f32(&[value as f32], &[1]))
        .collect();
    let arrays: Vec<&MlxArray> = chunks.iter().map(|chunk| chunk.as_ref().unwrap()).collect();
    let output = mlxcel_core::concatenate_many(&arrays, 0);
    mlxcel_core::eval(&output);

    assert_eq!(mlxcel_core::array_shape(&output), [256]);
    assert_eq!(
        array_to_vec_f32(&output),
        (0..256).map(|value| value as f32).collect::<Vec<_>>()
    );
}

#[test]
fn tower_rejects_wrong_shape_and_empty_frames() {
    let config = tiny_config();
    let mut weights = WeightMap::new();
    weights.insert(
        "audio_tower.embed_audio_tokens.weight".into(),
        mlxcel_core::zeros(&[12, 2], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        "audio_tower.norm.weight".into(),
        mlxcel_core::ones(&[2], mlxcel_core::dtype::FLOAT32),
    );
    let tower = InklingAudioTower::from_weights(&weights, &config, 64, 4).unwrap();
    assert!(
        tower
            .forward(&mlxcel_core::zeros(&[1, 2], mlxcel_core::dtype::INT32))
            .is_err()
    );
    assert!(
        tower
            .forward(&mlxcel_core::zeros(&[0, 3], mlxcel_core::dtype::INT32))
            .is_err()
    );
    assert!(
        tower
            .forward(&mlxcel_core::from_slice_i32(&[0, 4, 0], &[1, 3]))
            .is_err()
    );
    assert!(
        tower
            .forward(&mlxcel_core::zeros(&[1, 3], mlxcel_core::dtype::FLOAT32))
            .is_err()
    );
}
