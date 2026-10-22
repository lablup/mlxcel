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

#![allow(clippy::excessive_precision)]

use std::path::{Path, PathBuf};

use mlxcel_core::weights::WeightMap;

use super::*;

fn filled(shape: &[i32], value: f32) -> UniquePtr<MlxArray> {
    if value == 0.0 {
        mlxcel_core::zeros(shape, mlxcel_core::dtype::FLOAT32)
    } else if value == 1.0 {
        mlxcel_core::ones(shape, mlxcel_core::dtype::FLOAT32)
    } else {
        let count = shape.iter().map(|value| *value as usize).product();
        mlxcel_core::from_slice_f32(&vec![value; count], shape)
    }
}

fn insert(weights: &mut WeightMap, key: impl Into<String>, shape: &[i32], value: f32) {
    weights.insert(key.into(), filled(shape, value));
}

fn tiny_components(
    image_size: usize,
    patch_size: usize,
    depth: usize,
    registers: usize,
) -> (WeightMap, Dinov2Config) {
    let hidden = 2;
    let mut config = Dinov2Config::vits14();
    config.hidden_size = hidden;
    config.num_hidden_layers = depth;
    config.num_attention_heads = 1;
    config.mlp_ratio = 1.0;
    config.image_size = image_size;
    config.patch_size = patch_size;
    config.num_register_tokens = registers;
    config.intermediate_layers = vec![depth - 1];

    let mut weights = WeightMap::new();
    insert(
        &mut weights,
        "patch_embed.proj.weight",
        &[hidden as i32, patch_size as i32, patch_size as i32, 3],
        0.0,
    );
    insert(&mut weights, "patch_embed.proj.bias", &[hidden as i32], 0.0);
    insert(&mut weights, "cls_token", &[1, 1, hidden as i32], -2.0);
    insert(&mut weights, "mask_token", &[1, 1, hidden as i32], 3.0);
    let side = image_size / patch_size;
    let position_count = (side * side + 1) * hidden;
    let position_values = (0..position_count)
        .map(|index| index as f32 / 997.0)
        .collect::<Vec<_>>();
    weights.insert(
        "pos_embed".into(),
        mlxcel_core::from_slice_f32(
            &position_values,
            &[1, (side * side + 1) as i32, hidden as i32],
        ),
    );
    if registers > 0 {
        let values = (0..registers * hidden)
            .map(|index| 20.0 + index as f32)
            .collect::<Vec<_>>();
        weights.insert(
            "register_tokens".into(),
            mlxcel_core::from_slice_f32(&values, &[1, registers as i32, hidden as i32]),
        );
    }
    for index in 0..depth {
        let prefix = format!("blocks.{index}");
        for norm in ["norm1", "norm2"] {
            insert(
                &mut weights,
                format!("{prefix}.{norm}.weight"),
                &[hidden as i32],
                1.0,
            );
            insert(
                &mut weights,
                format!("{prefix}.{norm}.bias"),
                &[hidden as i32],
                0.0,
            );
        }
        insert(
            &mut weights,
            format!("{prefix}.attn.qkv.weight"),
            &[(hidden * 3) as i32, hidden as i32],
            0.0,
        );
        insert(
            &mut weights,
            format!("{prefix}.attn.qkv.bias"),
            &[(hidden * 3) as i32],
            0.0,
        );
        insert(
            &mut weights,
            format!("{prefix}.attn.proj.weight"),
            &[hidden as i32, hidden as i32],
            0.0,
        );
        insert(
            &mut weights,
            format!("{prefix}.attn.proj.bias"),
            &[hidden as i32],
            0.0,
        );
        insert(
            &mut weights,
            format!("{prefix}.mlp.fc1.weight"),
            &[hidden as i32, hidden as i32],
            0.0,
        );
        insert(
            &mut weights,
            format!("{prefix}.mlp.fc1.bias"),
            &[hidden as i32],
            0.0,
        );
        insert(
            &mut weights,
            format!("{prefix}.mlp.fc2.weight"),
            &[hidden as i32, hidden as i32],
            0.0,
        );
        insert(
            &mut weights,
            format!("{prefix}.mlp.fc2.bias"),
            &[hidden as i32],
            0.0,
        );
        insert(
            &mut weights,
            format!("{prefix}.ls1.gamma"),
            &[hidden as i32],
            1.0,
        );
        insert(
            &mut weights,
            format!("{prefix}.ls2.gamma"),
            &[hidden as i32],
            1.0,
        );
    }
    insert(&mut weights, "norm.weight", &[hidden as i32], 1.0);
    insert(&mut weights, "norm.bias", &[hidden as i32], 0.0);
    (weights, config)
}

fn tiny_model(image_size: usize, patch_size: usize, depth: usize, registers: usize) -> Dinov2 {
    let (weights, config) = tiny_components(image_size, patch_size, depth, registers);
    Dinov2::from_weights(&weights, config).unwrap()
}

fn values(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(array);
    mlxcel_core::utils::array_to_vec_f32(array)
}

fn assert_probes(array: &MlxArray, expected: &[(usize, f32)]) {
    let actual = values(array);
    for &(index, expected) in expected {
        assert!(
            (actual[index] - expected).abs() <= 5e-5,
            "probe {index}: expected {expected}, got {}",
            actual[index]
        );
    }
}

#[test]
fn positional_interpolation_matches_hugging_face_modes() {
    let tokens = filled(&[1, 1441, 2], 0.0);
    let mut model = tiny_model(518, 14, 1, 0);
    let identity_tokens = filled(&[1, 1370, 2], 0.0);
    let identity = model
        .interpolate_pos_encoding(&identity_tokens, 518, 518)
        .unwrap();
    assert_probes(
        &identity,
        &[(0, 0.0), (1, 0.00100300903), (2739, 2.74724173)],
    );

    let probes = &[
        (0, 0.0),
        (1, 0.00100300903),
        (2, 0.00625283225),
        (90, 0.0786915049),
        (1440, 1.36884570),
        (2878, 2.74057937),
        (2881, 2.74299431),
    ];
    let output = model.interpolate_pos_encoding(&tokens, 448, 630).unwrap();
    assert_eq!(mlxcel_core::array_shape(&output), [1, 1441, 2]);
    assert_probes(&output, probes);

    model.config.interpolate_antialias = true;
    let output = model.interpolate_pos_encoding(&tokens, 448, 630).unwrap();
    assert_probes(
        &output,
        &[
            (2, 0.00824080594),
            (90, 0.0806141868),
            (1440, 1.36699963),
            (2878, 2.73854280),
            (2881, 2.74100685),
        ],
    );

    model.config.interpolate_offset = 0.1;
    let output = model.interpolate_pos_encoding(&tokens, 448, 630).unwrap();
    assert_probes(
        &output,
        &[
            (2, 0.00615100376),
            (90, 0.0784910545),
            (1440, 1.36549544),
            (2878, 2.73395109),
            (2881, 2.73645735),
        ],
    );
}

#[test]
fn masks_registers_and_zero_register_path_preserve_token_contract() {
    let image = filled(&[1, 4, 4, 3], 0.0);
    let model = tiny_model(4, 2, 1, 2);
    let mask_values = mlxcel_core::from_slice_f32(&[1.0, 0.0, 0.0, 0.0], &[1, 4]);
    let mask = mlxcel_core::astype(&mask_values, mlxcel_core::dtype::BOOL);
    let tokens = model.prepare_tokens(&image, Some(&mask)).unwrap();
    assert_eq!(mlxcel_core::array_shape(&tokens), [1, 7, 2]);
    let actual = values(&tokens);
    assert_eq!(&actual[2..6], &[20.0, 21.0, 22.0, 23.0]);
    assert!((actual[6] - (3.0 + 2.0 / 997.0)).abs() < 1e-6);

    let no_registers = tiny_model(4, 2, 1, 0);
    let tokens = no_registers.prepare_tokens(&image, None).unwrap();
    assert_eq!(mlxcel_core::array_shape(&tokens), [1, 5, 2]);
}

#[test]
fn mask_token_is_cast_to_f16_patch_dtype() {
    let (mut weights, config) = tiny_components(4, 2, 1, 0);
    for value in weights.values_mut() {
        *value = mlxcel_core::astype(value, mlxcel_core::dtype::FLOAT16);
    }
    weights.insert("mask_token".into(), filled(&[1, 1, 2], 3.0));
    let model = Dinov2::from_weights(&weights, config).unwrap();
    let image = mlxcel_core::astype(&filled(&[1, 4, 4, 3], 0.0), mlxcel_core::dtype::FLOAT16);
    let mask = mlxcel_core::astype(
        &mlxcel_core::from_slice_i32(&[1, 0, 0, 0], &[1, 4]),
        mlxcel_core::dtype::BOOL,
    );
    let tokens = model.prepare_tokens(&image, Some(&mask)).unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(&tokens),
        mlxcel_core::dtype::FLOAT16
    );
}

#[test]
fn public_scoped_constructor_keeps_parent_checkpoint_reusable() {
    let (weights, config) = tiny_components(4, 2, 1, 0);
    let mut parent = WeightMap::new();
    for (key, value) in weights {
        parent.insert(format!("image_backbone.{key}"), value);
    }
    insert(&mut parent, "dense_head.weight", &[1], 7.0);
    let model = Dinov2::from_hf_weights(&parent, config, "image_backbone").unwrap();
    assert!(parent.contains_key("dense_head.weight"));
    let output = model.forward(&filled(&[1, 4, 4, 3], 0.0), None).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&output.last_hidden_state),
        [1, 5, 2]
    );
}

#[test]
fn intermediate_layers_are_normalized_ordered_and_exclude_registers() {
    let mut model = tiny_model(4, 2, 12, 2);
    model.config.intermediate_layers = vec![4, 11];
    let image = filled(&[1, 4, 4, 3], 0.0);
    let output = model.get_intermediate_layers(&image, &[4, 11]).unwrap();
    assert_eq!(output.len(), 2);
    for (patches, class) in &output {
        assert_eq!(mlxcel_core::array_shape(patches), [1, 4, 2]);
        assert_eq!(mlxcel_core::array_shape(class), [1, 2]);
        for row in values(patches).chunks_exact(2) {
            assert!((row[0] + row[1]).abs() < 1e-5);
        }
    }
    assert_eq!(values(&output[0].0), values(&output[1].0));
    let error = model.get_intermediate_layers(&image, &[12]).err().unwrap();
    assert!(error.to_string().contains("outside depth 12"));
    let error = model
        .get_intermediate_layers(&image, &[4, 4])
        .err()
        .unwrap();
    assert!(error.to_string().contains("must be unique"));
}

#[test]
fn invalid_spatial_shapes_and_size_configs_are_rejected() {
    let model = tiny_model(4, 2, 1, 0);
    let image = filled(&[1, 3, 4, 3], 0.0);
    let error = model.forward(&image, None).err().unwrap();
    assert!(error.to_string().contains("positive multiples of 2"));

    let non_square = serde_json::json!({"image_size": [518, 504]});
    assert!(serde_json::from_value::<Dinov2Config>(non_square).is_err());
    let wrong_length = serde_json::json!({"patch_size": [14]});
    assert!(serde_json::from_value::<Dinov2Config>(wrong_length).is_err());

    let mut invalid = Dinov2Config::vits14();
    invalid.num_hidden_layers = 0;
    assert!(invalid.validate().is_err());

    let mut duplicate_layers = Dinov2Config::vits14();
    duplicate_layers.intermediate_layers = vec![4, 4];
    assert!(duplicate_layers.validate().is_err());

    let empty_weights = WeightMap::new();
    assert!(Attention::from_weights(&empty_weights, "attn", 3, 2, true).is_err());
    assert!(Mlp::from_weights(&empty_weights, "mlp", 2, usize::MAX).is_err());

    let oversized = filled(&[1, 2, 32_770, 3], 0.0);
    let error = model.prepare_tokens(&oversized, None).err().unwrap();
    assert!(error.to_string().contains("patch count exceeds"));
}

#[test]
fn sanitizer_fuses_hf_projections_and_keeps_convolution_channels_last() {
    use super::super::dinov2_weights::sanitize_weights;

    let mut weights = WeightMap::new();
    for (name, value) in [("query", 1.0), ("key", 2.0), ("value", 3.0)] {
        insert(
            &mut weights,
            format!("dinov2.encoder.layer.0.attention.attention.{name}.weight"),
            &[2, 2],
            value,
        );
    }
    for (name, value) in [("gate_proj", 4.0), ("up_proj", 5.0)] {
        insert(
            &mut weights,
            format!("dinov2.encoder.layer.0.mlp.{name}.weight"),
            &[2, 2],
            value,
        );
    }
    insert(&mut weights, "patch_embed.proj.weight", &[2, 2, 2, 3], 7.0);
    let sanitized = sanitize_weights(weights).unwrap();
    let qkv = sanitized.get("blocks.0.attn.qkv.weight").unwrap();
    assert_eq!(mlxcel_core::array_shape(qkv), [6, 2]);
    assert_eq!(
        values(qkv),
        vec![1.0; 4]
            .into_iter()
            .chain(vec![2.0; 4])
            .chain(vec![3.0; 4])
            .collect::<Vec<_>>()
    );
    let w12 = sanitized.get("blocks.0.mlp.w12.weight").unwrap();
    assert_eq!(mlxcel_core::array_shape(w12), [4, 2]);
    let twice = sanitize_weights(sanitized).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(twice.get("patch_embed.proj.weight").unwrap()),
        [2, 2, 2, 3]
    );

    let mut giant_weights = WeightMap::new();
    insert(
        &mut giant_weights,
        "encoder.layer.0.mlp.weights_in.weight",
        &[8, 2],
        1.0,
    );
    insert(
        &mut giant_weights,
        "encoder.layer.0.mlp.weights_out.weight",
        &[2, 4],
        1.0,
    );
    let giant_weights = sanitize_weights(giant_weights).unwrap();
    assert!(giant_weights.contains_key("blocks.0.mlp.w12.weight"));
    assert!(giant_weights.contains_key("blocks.0.mlp.w3.weight"));

    let giant = Dinov2Config::vitg14();
    assert_eq!(giant.hidden_size, 1536);
    assert_eq!(giant.num_hidden_layers, 40);
    assert_eq!(giant.num_attention_heads, 24);
    assert!(giant.use_swiglu_ffn);
    giant.validate().unwrap();
}

fn fixture_tensor(path: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensor = tensors.tensor(name).unwrap();
    assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
    let values = tensor
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (tensor.shape().to_vec(), values)
}

fn assert_relative_rms(actual: &MlxArray, expected: &[f32], name: &str) {
    let actual = values(actual);
    assert_eq!(actual.len(), expected.len(), "{name} element count");
    let (squared_error, squared_reference) = actual.iter().zip(expected).fold(
        (0.0_f64, 0.0_f64),
        |(error, reference), (actual, expected)| {
            let delta = f64::from(*actual) - f64::from(*expected);
            (
                error + delta * delta,
                reference + f64::from(*expected) * f64::from(*expected),
            )
        },
    );
    let relative_rms = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("{name} relative RMS: {relative_rms:.9e}");
    assert!(relative_rms < 1e-3, "{name} relative RMS {relative_rms}");
}

fn run_reference(checkpoint: PathBuf, fixture_name: &str, expected_tokens: usize) {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/dinov2")
        .join(fixture_name);
    let (input_shape, input) = fixture_tensor(&fixture, "pixel_values_nhwc");
    let input = mlxcel_core::from_slice_f32(
        &input,
        &input_shape
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>(),
    );
    let model = Dinov2::load(&checkpoint).unwrap();
    let output = model.forward(&input, None).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&output.last_hidden_state),
        [1, expected_tokens as i32, 384]
    );
    for (name, actual) in [
        ("last_hidden_state", &*output.last_hidden_state),
        ("pooler_output", &*output.pooler_output),
        ("hidden_patch_tokens", &*output.hidden_patch_tokens),
    ] {
        let (_, expected) = fixture_tensor(&fixture, name);
        assert_relative_rms(actual, &expected, name);
    }
}

#[test]
fn public_hf_small_checkpoints_match_reference_outputs() {
    let Ok(standard) = std::env::var("MLXCEL_TEST_DINOV2_SMALL_DIR") else {
        eprintln!("skipping DINOv2 reference parity: MLXCEL_TEST_DINOV2_SMALL_DIR is unset");
        return;
    };
    let Ok(registers) = std::env::var("MLXCEL_TEST_DINOV2_REGISTER_SMALL_DIR") else {
        eprintln!(
            "skipping register DINOv2 parity: MLXCEL_TEST_DINOV2_REGISTER_SMALL_DIR is unset"
        );
        return;
    };
    let _device_lock = mlxcel_core::streams::lock_default_device();
    let _cpu_device = mlxcel_core::streams::DefaultDeviceGuard::cpu();
    run_reference(PathBuf::from(standard), "small_reference.safetensors", 257);
    run_reference(
        PathBuf::from(registers),
        "register_small_reference.safetensors",
        261,
    );
}
