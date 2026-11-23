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

//! Quantized learned parameters in the Inkling 2 vision tower (#3228): the
//! affine-quantized export stores the position, class and layer-scale tables
//! with `.scales` / `.biases` sidecars beside its quantized linears.

use super::*;

const GROUP: i32 = 64;
const BITS: i32 = 4;
const HIDDEN: i32 = 128;
const TEXT_HIDDEN: i32 = 64;
const GRID: i32 = 4;

struct Rng(u64);

impl Rng {
    fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    }
}

fn config() -> Inkling2VisionConfig {
    serde_json::from_value(serde_json::json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "posemb_grid_size": GRID,
        "projection_hidden_dim": 128,
    }))
    .expect("tiny vision config")
}

/// Every tensor the tower reads, dense f32.
fn dense_weights() -> WeightMap {
    let mut rng = Rng(7);
    let mut weights = WeightMap::new();
    let mut put = |key: &str, shape: &[i32], scale: f32, offset: f32| {
        let count = shape.iter().product::<i32>() as usize;
        let values: Vec<f32> = rng
            .values(count, scale)
            .into_iter()
            .map(|v| v + offset)
            .collect();
        weights.insert(key.to_string(), mlxcel_core::from_slice_f32(&values, shape));
    };
    let mlp = HIDDEN * 4;
    put(
        "vision_tower.embeddings.patch_embedding.weight",
        &[HIDDEN, 1200],
        0.05,
        0.0,
    );
    put(
        "vision_tower.embeddings.position_embedding",
        &[GRID * GRID, HIDDEN],
        0.5,
        0.0,
    );
    put(
        "vision_tower.embeddings.class_embedding",
        &[HIDDEN],
        0.5,
        0.0,
    );
    put(
        "vision_tower.embeddings.cls_position_embedding",
        &[HIDDEN],
        0.5,
        0.0,
    );
    for norm in [
        "vision_tower.pre_layernorm",
        "vision_tower.layers.0.layernorm_before",
        "vision_tower.layers.0.layernorm_after",
    ] {
        put(&format!("{norm}.weight"), &[HIDDEN], 0.1, 1.0);
        put(&format!("{norm}.bias"), &[HIDDEN], 0.1, 0.0);
    }
    for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        put(
            &format!("vision_tower.layers.0.self_attn.{projection}.weight"),
            &[HIDDEN, HIDDEN],
            0.1,
            0.0,
        );
    }
    put(
        "vision_tower.layers.0.mlp.fc1.weight",
        &[mlp, HIDDEN],
        0.1,
        0.0,
    );
    put(
        "vision_tower.layers.0.mlp.fc2.weight",
        &[HIDDEN, mlp],
        0.05,
        0.0,
    );
    put("vision_tower.layers.0.lambda_1", &[HIDDEN], 0.2, 0.5);
    put("vision_tower.layers.0.lambda_2", &[HIDDEN], 0.2, 0.5);
    put("vision_tower.proj_in.weight", &[128, HIDDEN * 4], 0.05, 0.0);
    put(
        "vision_tower.proj_out.weight",
        &[TEXT_HIDDEN, 128],
        0.1,
        0.0,
    );
    weights
}

fn sidecar(key: &str, suffix: &str) -> String {
    key.strip_suffix(".weight").map_or_else(
        || format!("{key}.{suffix}"),
        |prefix| format!("{prefix}.{suffix}"),
    )
}

/// Keys the affine export quantizes: every linear whose input width is a
/// multiple of the group, and the learned tables.
const QUANTIZED: &[&str] = &[
    "vision_tower.embeddings.position_embedding",
    "vision_tower.embeddings.class_embedding",
    "vision_tower.embeddings.cls_position_embedding",
    "vision_tower.layers.0.lambda_1",
    "vision_tower.layers.0.lambda_2",
    "vision_tower.layers.0.self_attn.q_proj.weight",
    "vision_tower.layers.0.self_attn.k_proj.weight",
    "vision_tower.layers.0.self_attn.v_proj.weight",
    "vision_tower.layers.0.self_attn.out_proj.weight",
    "vision_tower.layers.0.mlp.fc1.weight",
    "vision_tower.layers.0.mlp.fc2.weight",
    "vision_tower.proj_in.weight",
    "vision_tower.proj_out.weight",
];

/// The quantized map, and the dense map holding exactly what it dequantizes to.
fn quantized_and_reference() -> (WeightMap, WeightMap) {
    let mut quantized = dense_weights();
    let mut reference = dense_weights();
    for &key in QUANTIZED {
        let dense = quantized.remove(key).expect("dense tensor");
        let shape = mlxcel_core::array_shape(&dense);
        let width = *shape.last().expect("rank >= 1");
        let flat = mlxcel_core::reshape(&dense, &[-1, width]);
        let triple = mlxcel_core::quantize_weights(&flat, GROUP, BITS);
        let (packed, scales, biases) = (
            mlxcel_core::quantized_weights_w(&triple),
            mlxcel_core::quantized_weights_scales(&triple),
            mlxcel_core::quantized_weights_biases(&triple),
        );
        // SAFETY: the affine triple comes from `quantize_weights` above and
        // every plane outlives the call.
        let restored = unsafe {
            mlxcel_core::dequantize(
                &packed,
                &scales,
                &*biases as *const MlxArray,
                GROUP,
                BITS,
                "affine",
            )
        };
        reference.insert(key.to_string(), mlxcel_core::reshape(&restored, &shape));
        let mut packed_shape = shape.clone();
        *packed_shape.last_mut().expect("rank >= 1") = mlxcel_core::array_shape(&packed)[1];
        let mut sidecar_shape = shape.clone();
        *sidecar_shape.last_mut().expect("rank >= 1") = mlxcel_core::array_shape(&scales)[1];
        quantized.insert(
            key.to_string(),
            mlxcel_core::reshape(&packed, &packed_shape),
        );
        quantized.insert(
            sidecar(key, "scales"),
            mlxcel_core::reshape(&scales, &sidecar_shape),
        );
        quantized.insert(
            sidecar(key, "biases"),
            mlxcel_core::reshape(&biases, &sidecar_shape),
        );
    }
    (quantized, reference)
}

/// One 2 x 1 tile image (80 x 40 pixels).
fn pixels() -> UniquePtr<MlxArray> {
    let values = Rng(11).values(2 * 40 * 40 * 3, 1.0);
    mlxcel_core::from_slice_f32(&values, &[2, 40, 40, 3])
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let diff = mlxcel_core::abs(&mlxcel_core::subtract(a, b));
    mlxcel_core::item_f32(&mlxcel_core::max_all(&diff))
}

#[test]
fn quantized_tables_and_linears_match_their_dequantized_dense_tower() {
    let (quantized, reference) = quantized_and_reference();
    let config = config();
    let quantized = Inkling2VisionTower::from_weights(&quantized, &config, (GROUP, BITS))
        .expect("quantized tower loads");
    let dense = Inkling2VisionTower::from_weights(&reference, &config, (GROUP, BITS))
        .expect("dense tower loads");
    let grids = [(2, 1)];
    let got = quantized
        .forward(&pixels(), &grids)
        .expect("quantized forward");
    let want = dense.forward(&pixels(), &grids).expect("dense forward");
    assert_eq!(mlxcel_core::array_shape(&got), [2, TEXT_HIDDEN]);
    let diff = max_abs_diff(&got, &want);
    let scale = mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(&want)));
    assert!(
        diff <= 1e-3 * scale.max(1.0),
        "max |diff| {diff} against max |x| {scale}"
    );
}

#[test]
fn a_dense_checkpoint_still_loads() {
    let tower = Inkling2VisionTower::from_weights(&dense_weights(), &config(), (GROUP, BITS))
        .expect("dense tower loads");
    let out = tower.forward(&pixels(), &[(2, 1)]).expect("dense forward");
    assert_eq!(mlxcel_core::array_shape(&out), [2, TEXT_HIDDEN]);
}

#[test]
fn a_malformed_sidecar_is_a_load_error_naming_the_tensor() {
    let (mut quantized, _) = quantized_and_reference();
    let key = "vision_tower.layers.0.lambda_1";
    quantized.insert(
        sidecar(key, "scales"),
        mlxcel_core::from_slice_f32(&[1.0; 3], &[3]),
    );
    let error = Inkling2VisionTower::from_weights(&quantized, &config(), (GROUP, BITS))
        .err()
        .expect("a wrong scales shape must fail");
    assert!(error.contains(key), "error does not name {key}: {error}");
}
