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

//! Florence-2 DaViT vision-backbone parity against the mlx-vlm reference.
//!
//! Pins the four-stage DaViT tower (ConvEmbed pre/post norm, depthwise
//! positional convs, windowed spatial attention, channel attention with the
//! token-count query scale) against reference activations captured from the
//! mlx-vlm florence2 `VisionModel`
//! (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/vision.py)
//! running the same `models/Florence-2-base-ft-bf16` checkpoint with weights
//! cast bf16 -> f16, matching mlxcel's Apple Silicon precision policy. Skips
//! when the checkpoint is absent (CI has no Metal and no weights).
//!
//! The input is a deterministic synthetic pixel tensor rather than a decoded
//! image: Florence-2 image preprocessing belongs to the processor sub-issue,
//! and a closed-form tensor is the only way to guarantee both runtimes see
//! bit-identical input today. Both sides build
//! `x[0, c, h, w] = ((h * 768 + w) * 3 + c) % 251 / 251.0 - 0.5` in f32 with
//! the same IEEE-754 operation order, then cast to f16.
//!
//! To regenerate the pins, in a virtualenv holding `mlx` and `numpy`, with
//! the mlx-vlm florence2 `vision.py` and `config.py` importable as a package:
//! load `models/Florence-2-base-ft-bf16/model.safetensors` with `mx.load`,
//! keep the `vision_tower.`-prefixed tensors with that prefix stripped, run
//! them through `VisionModel.sanitize`, cast every tensor to `mx.float16`,
//! build `VisionModel(VisionConfig.from_dict(config["vision_config"]))`, and
//! drive its stage loop (`for conv, blks in zip(model.convs, model.blocks)`)
//! over `mx.array(pixels).astype(mx.float16)` with `pixels` built by the same
//! closed form as `synthetic_pixels` below. Print each stage's shape, the
//! first 16 values of its output, and its mean / standard deviation.

use std::path::Path;

use mlxcel::models::{Florence2DaViT, Florence2VisionConfig};

const MODEL_DIR: &str = "models/Florence-2-base-ft-bf16";
const IMAGE_SIDE: i32 = 768;

/// Per-stage `(H, W)` grid and channel width for a 768x768 input.
const REF_STAGE_SHAPES: &[(i32, i32, i32)] = &[
    (192, 192, 128),
    (96, 96, 256),
    (48, 48, 512),
    (24, 24, 1024),
];

// Reference activations from the mlx-vlm florence2 VisionModel (f16 weights,
// f32 readout) on the same checkpoint and the synthetic input above. The full
// digit strings are the exact f16-representable reference values; truncating
// them would move the pins off the values the reference produced.
#[allow(clippy::excessive_precision)]
const REF_STAGE_FIRST16: &[&[f32]] = &[
    &[
        0.09832763671875,
        -0.171630859375,
        0.204345703125,
        0.0914306640625,
        -0.07818603515625,
        0.07391357421875,
        -0.145751953125,
        -0.01416015625,
        -0.0079345703125,
        -0.038116455078125,
        -0.30908203125,
        -0.05322265625,
        -0.0084228515625,
        -0.06329345703125,
        -0.0166473388671875,
        0.14501953125,
    ],
    &[
        0.05743408203125,
        -0.0853271484375,
        2.6015625,
        -0.1715087890625,
        -0.1055908203125,
        0.07421875,
        -0.1572265625,
        -0.15771484375,
        0.11279296875,
        0.2137451171875,
        0.12353515625,
        0.057281494140625,
        -0.032470703125,
        -0.069580078125,
        -0.00994873046875,
        -0.07830810546875,
    ],
    &[
        -0.054931640625,
        -0.0341796875,
        0.1148681640625,
        -0.0047607421875,
        0.0008544921875,
        0.06121826171875,
        -0.06121826171875,
        -0.0980224609375,
        0.18994140625,
        -0.01214599609375,
        0.11181640625,
        -0.073486328125,
        0.1759033203125,
        -0.06317138671875,
        -0.1063232421875,
        -0.10406494140625,
    ],
    &[
        0.164794921875,
        -1.3779296875,
        -0.1307373046875,
        -0.7822265625,
        0.16650390625,
        -0.4638671875,
        0.32275390625,
        -1.51171875,
        0.0391845703125,
        -0.50732421875,
        0.2088623046875,
        -1.4443359375,
        0.1802978515625,
        -0.759765625,
        0.420654296875,
        -0.8876953125,
    ],
];

/// Per-stage `(mean, std)` of the full stage output.
const REF_STAGE_STATS: &[(f32, f32)] = &[
    (-0.001656, 0.119339),
    (0.010804, 0.290888),
    (0.074503, 1.458821),
    (-0.487382, 0.524704),
];

/// Absolute tolerance per stage. Both sides execute in f16 with different op
/// ordering, so the bound scales with the activation magnitude at that stage
/// (`absmax` was 1.54 / 8.03 / 65.5 / 3.97 respectively). Each bound is a
/// small multiple of one f16 ulp at its stage's peak, and every one is at
/// least 25x below that stage's own standard deviation. Observed deviations
/// on Apple Silicon were 3.4e-4 / 4.3e-3 / 7.4e-4 / 3.1e-3.
const STAGE_TOL: &[f32] = &[2e-3, 1.2e-2, 6e-2, 1.2e-2];

/// Tolerance on the whole-tensor mean / standard deviation per stage.
const STAGE_STAT_TOL: &[f32] = &[1e-4, 3e-4, 3e-3, 3e-4];

/// `x[0, c, h, w] = ((h * side + w) * 3 + c) % 251 / 251.0 - 0.5`, NCHW.
///
/// Every operation is a single IEEE-754 f32 step over exactly represented
/// integers, so the Python reference reproduces this bit for bit.
fn synthetic_pixels(side: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; (3 * side * side) as usize];
    for c in 0..3i64 {
        for h in 0..side as i64 {
            for w in 0..side as i64 {
                let raw = ((h * side as i64 + w) * 3 + c) % 251;
                let value = raw as f32 / 251.0f32 - 0.5f32;
                out[(c * side as i64 * side as i64 + h * side as i64 + w) as usize] = value;
            }
        }
    }
    out
}

fn to_vec_f32(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let dev = (g - w).abs();
        if dev > worst {
            worst = dev;
            worst_at = i;
        }
    }
    eprintln!("{what}: max abs deviation {worst} at index {worst_at} (tol {tol})");
    assert!(
        worst <= tol,
        "{what}[{worst_at}]: got {}, reference {} (deviation {worst}, tol {tol})",
        got[worst_at],
        want[worst_at]
    );
}

#[test]
fn florence2_davit_matches_mlx_vlm_reference() {
    if !Path::new(MODEL_DIR).exists() {
        eprintln!("skipping florence2_vision_parity: {MODEL_DIR} not present");
        return;
    }

    let model = Florence2DaViT::load(Path::new(MODEL_DIR)).expect("load florence2 DaViT backbone");
    let config = model.config();
    assert_eq!(config.dim_embed, vec![128, 256, 512, 1024]);
    assert_eq!(config.depths, vec![1, 1, 9, 1]);
    assert_eq!(config.num_groups, vec![4, 8, 16, 32]);
    assert_eq!(config.window_size, 12);
    // The backbone emits dim_embed[-1]; projection_dim is the fusion stage's.
    assert_eq!(config.output_dim(), 1024);
    assert_eq!(config.projection_dim, 768);

    let pixels = synthetic_pixels(IMAGE_SIDE);
    let pixel_values = mlxcel_core::from_slice_f32(&pixels, &[1, 3, IMAGE_SIDE, IMAGE_SIDE]);
    let pixel_values = mlxcel_core::astype(&pixel_values, mlxcel_core::dtype::FLOAT16);

    let stages = model.forward_stages(&pixel_values);
    assert_eq!(stages.len(), REF_STAGE_SHAPES.len());

    for (i, (out, size)) in stages.iter().enumerate() {
        let (ref_h, ref_w, ref_c) = REF_STAGE_SHAPES[i];
        assert_eq!(*size, (ref_h, ref_w), "stage {i}: grid size");
        assert_eq!(
            mlxcel_core::array_shape(out),
            vec![1, ref_h * ref_w, ref_c],
            "stage {i}: output shape"
        );

        let values = to_vec_f32(out);
        assert_close(
            &values[..16],
            REF_STAGE_FIRST16[i],
            STAGE_TOL[i],
            &format!("stage {i} first16"),
        );

        let n = values.len() as f64;
        let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = values
            .iter()
            .map(|v| {
                let d = *v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let (ref_mean, ref_std) = REF_STAGE_STATS[i];
        let stat_tol = STAGE_STAT_TOL[i];
        eprintln!(
            "stage {i}: mean {mean:.6} (ref {ref_mean:.6}), std {:.6} (ref {ref_std:.6})",
            var.sqrt()
        );
        assert!(
            (mean as f32 - ref_mean).abs() <= stat_tol,
            "stage {i} mean: got {mean}, reference {ref_mean} (tol {stat_tol})"
        );
        assert!(
            (var.sqrt() as f32 - ref_std).abs() <= stat_tol,
            "stage {i} std: got {}, reference {ref_std} (tol {stat_tol})",
            var.sqrt()
        );
    }

    // The plain forward must agree with the last stage of forward_stages.
    let final_out = Florence2DaViT::forward(&model, &pixel_values);
    assert_eq!(
        mlxcel_core::array_shape(&final_out),
        vec![1, 576, 1024],
        "final backbone output shape"
    );
    assert_close(
        &to_vec_f32(&final_out)[..16],
        REF_STAGE_FIRST16[3],
        STAGE_TOL[3],
        "final output first16",
    );
}

#[test]
fn florence2_vision_config_parses_real_checkpoint() {
    if !Path::new(MODEL_DIR).exists() {
        eprintln!("skipping florence2_vision_parity config check: {MODEL_DIR} not present");
        return;
    }
    let raw = std::fs::read_to_string(Path::new(MODEL_DIR).join("config.json"))
        .expect("read florence2 config.json");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("parse florence2 config.json");
    // The real checkpoint carries model_type "" inside vision_config; the
    // DaViT family is identified by the parent model_type plus structure.
    assert_eq!(
        value.get("model_type").and_then(|v| v.as_str()),
        Some("florence2")
    );
    assert_eq!(
        value
            .get("vision_config")
            .and_then(|v| v.get("model_type"))
            .and_then(|v| v.as_str()),
        Some("")
    );
    let config = Florence2VisionConfig::from_model_config(&value).expect("parse vision config");
    assert_eq!(config.num_stages(), 4);
    assert_eq!(config.output_dim(), 1024);
    assert_eq!(config.patch_prenorm, vec![false, true, true, true]);
    assert!((config.drop_path_rate - 0.1).abs() < 1e-6);
    assert_eq!(
        config.image_feature_source,
        vec![
            "spatial_avg_pool".to_string(),
            "temporal_avg_pool".to_string()
        ]
    );
    assert!(config.image_pos_embed.is_some());
    assert!(config.visual_temporal_embedding.is_some());
}
