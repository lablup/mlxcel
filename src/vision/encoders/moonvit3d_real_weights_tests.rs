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

//! The `#[ignore]`d real-weights harness for the Kimi K3 MoonViT3D tower.

use super::*;
use mlxcel_core::dtype;
use mlxcel_core::utils::array_to_vec_f32;

fn to_vec(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(a);
    array_to_vec_f32(a)
}

// ---------------------------------------------------------------------------
// Real weights (ignored): tower + projector against the out-of-band reference
// ---------------------------------------------------------------------------

/// Relative mean-absolute error `mean(|a - b|) / mean(|b|)`.
fn rel_mean_abs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from((x - y).abs())).sum();
    let den: f64 = b.iter().map(|y| f64::from(y.abs())).sum();
    num / den.max(1e-12)
}

fn json_f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

/// The MoonViT3D tower and the `patchmergerv2` projector on the real bf16
/// weights of `models/kimi-k3-8l-mxfp4` (shards 00095 / 00096), compared with
/// `tests/fixtures/kimi_k3_vision/reference.json`, which
/// `tests/fixtures/kimi_k3_vision/generate_reference.py` computes in numpy
/// from the same shards and the same image, independently of this code.
///
/// Run with:
/// `cargo test --profile test-fast --features metal,accelerate --lib vision::encoders::moonvit3d -- --ignored kimi_k3_tower_real_weights --nocapture`
#[test]
#[ignore = "needs models/kimi-k3-8l-mxfp4 (about 1.5 GB of vision shards)"]
fn kimi_k3_tower_real_weights() {
    use crate::vision::kimi_k3_vl::{KimiK3VLModel, sanitize_kimi_k3_vision_weights};
    use crate::vision::processors::kimi_k3::KimiK3ImageProcessor;
    use std::path::Path;

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let reference_path = root.join("tests/fixtures/kimi_k3_vision/reference.json");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&reference_path).expect("reference.json"))
            .expect("parse reference.json");
    let checkpoint = root.join(reference["checkpoint"].as_str().expect("checkpoint"));
    assert!(
        checkpoint.join("config.json").exists(),
        "missing checkpoint {}",
        checkpoint.display()
    );

    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(checkpoint.join("config.json")).expect("config.json"),
    )
    .expect("parse config.json");
    let cfg: MoonViT3DConfig =
        serde_json::from_value(config["vision_config"].clone()).expect("vision_config");
    cfg.validate().expect("published vision_config");

    // Preprocessing: the same image, compared value for value on a sample.
    let image_path = root.join(reference["image"].as_str().expect("image"));
    let image = image::open(&image_path).expect("open fixture image");
    let processor = KimiK3ImageProcessor::from_model_dir(&checkpoint).expect("preprocessor config");
    let prepared = processor
        .preprocess_with_grid(&[image])
        .expect("preprocess");
    let item = prepared.images[0];
    let grid = json_f32s(&reference["grid_thw"]);
    assert_eq!(
        (item.grid.t, item.grid.h, item.grid.w),
        (grid[0] as i32, grid[1] as i32, grid[2] as i32)
    );
    assert_eq!(
        item.plan.num_tokens as u64,
        reference["num_tokens"].as_u64().unwrap()
    );
    let pixels = to_vec(&prepared.pixel_values);
    let pixel_head = json_f32s(&reference["pixel_values"]["head"]);
    for (i, (a, b)) in pixels.iter().zip(&pixel_head).enumerate() {
        assert!((a - b).abs() < 1e-5, "pixel {i}: {a} vs {b}");
    }
    let pixel_mean_abs =
        pixels.iter().map(|v| f64::from(v.abs())).sum::<f64>() / pixels.len() as f64;
    let ref_pixel_mean_abs = reference["pixel_values"]["mean_abs"].as_f64().unwrap();
    assert!(
        (pixel_mean_abs - ref_pixel_mean_abs).abs() / ref_pixel_mean_abs < 1e-5,
        "pixel mean_abs {pixel_mean_abs} vs {ref_pixel_mean_abs}"
    );
    println!(
        "pixels: {} values, mean_abs {pixel_mean_abs:.6} (reference {ref_pixel_mean_abs:.6})",
        pixels.len()
    );

    // Weights: only the vision shards.
    let raw = mlxcel_core::weights::load_weights_from_dir_index_filtered(&checkpoint, |k| {
        k.starts_with("vision_tower.") || k.starts_with("mm_projector.")
    })
    .expect("load vision shards");
    assert_eq!(raw.len(), 168, "vision + projector tensor count");
    let raw = sanitize_kimi_k3_vision_weights(raw);

    let sample_stride = reference["projected"]["sample_col_stride"]
        .as_u64()
        .unwrap() as usize;
    let ref_sample: Vec<f32> = reference["projected"]["sample"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(json_f32s)
        .collect();
    let ref_row_mean_abs = json_f32s(&reference["projected"]["row_mean_abs"]);
    let ref_final_mean_abs = reference["stages"]["final_norm"]["mean_abs"]
        .as_f64()
        .unwrap();
    let ref_projected_mean_abs = reference["projected"]["mean_abs"].as_f64().unwrap();
    let text_hidden = cfg.text_hidden_size;

    let mut results = Vec::new();
    for (label, target_dtype, gate) in [
        ("f32", Some(dtype::FLOAT32), 1e-2),
        ("bf16 (checkpoint dtype)", None, 5e-2),
    ] {
        let weights: mlxcel_core::weights::WeightMap = raw
            .iter()
            .map(|(k, v)| {
                let v = match target_dtype {
                    Some(d) => mlxcel_core::astype(v, d),
                    None => mlxcel_core::copy(v),
                };
                (k.clone(), v)
            })
            .collect();
        let tower =
            MoonViT3DVisionModel::from_weights(&weights, &cfg, "vision_tower").expect("tower");
        let projector = KimiK3VLModel::projector_from_weights(&weights, &cfg).expect("projector");
        let run_dtype = target_dtype.unwrap_or(dtype::BFLOAT16);
        let pv = mlxcel_core::astype(&prepared.pixel_values, run_dtype);
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);
        let grids = prepared.grids();

        let final_norm = tower.forward(&pv, &grids).expect("tower forward");
        let final_norm = to_vec(&final_norm[0]);
        let final_mean_abs =
            final_norm.iter().map(|v| f64::from(v.abs())).sum::<f64>() / final_norm.len() as f64;
        let final_err = (final_mean_abs - ref_final_mean_abs).abs() / ref_final_mean_abs;

        let merged = tower.forward_merged(&pv, &grids).expect("merge");
        let projected = projector.forward(&merged).expect("projector");
        let projected = to_vec(&projected);
        assert_eq!(projected.len(), item.plan.num_tokens as usize * text_hidden);
        assert!(
            projected.iter().all(|v| v.is_finite()),
            "{label}: non-finite projector output"
        );

        let rows = item.plan.num_tokens as usize;
        let sample: Vec<f32> = (0..rows)
            .flat_map(|r| (0..text_hidden).step_by(sample_stride).map(move |c| (r, c)))
            .map(|(r, c)| projected[r * text_hidden + c])
            .collect();
        let sample_err = rel_mean_abs(&sample, &ref_sample);
        let row_mean_abs: Vec<f32> = projected
            .chunks(text_hidden)
            .map(|row| row.iter().map(|v| v.abs()).sum::<f32>() / text_hidden as f32)
            .collect();
        let row_err = rel_mean_abs(&row_mean_abs, &ref_row_mean_abs);
        let projected_mean_abs =
            projected.iter().map(|v| f64::from(v.abs())).sum::<f64>() / projected.len() as f64;
        println!(
            "{label}: final_norm mean_abs {final_mean_abs:.5} (ref {ref_final_mean_abs:.5}, rel err \
             {final_err:.2e}); projected mean_abs {projected_mean_abs:.5} (ref \
             {ref_projected_mean_abs:.5}); sample rel mean-abs err {sample_err:.3e} over {} values; \
             per-token mean_abs rel err {row_err:.3e}; gate {gate:.0e}",
            sample.len()
        );
        results.push((label, sample_err, row_err, gate));
    }
    for (label, sample_err, row_err, gate) in results {
        assert!(
            sample_err < gate,
            "{label}: projected sample rel mean-abs err {sample_err:.3e} >= {gate:.0e}"
        );
        assert!(
            row_err < gate,
            "{label}: per-token mean_abs rel err {row_err:.3e} >= {gate:.0e}"
        );
    }
}
