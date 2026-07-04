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

//! DeepSeek-OCR SAM + CLIP encoder parity against the mlx-vlm reference.
//!
//! Gated on `models/deepseek-ocr-4bit` (vision weights are bf16 there). Feeds
//! a deterministic input and prints the SAM / CLIP output signatures for a
//! numeric diff against `scratchpad/ref_dsocr_enc2.py`.

use std::path::PathBuf;

use mlxcel::vision::encoders::deepseekocr_clip::{ClipConfig, ClipEncoder};
use mlxcel::vision::encoders::deepseekocr_sam::{SamConfig, SamEncoder};
use mlxcel_core::MlxArray;

fn model_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/deepseek-ocr-4bit");
    dir.exists().then_some(dir)
}

fn sig(a: &MlxArray, name: &str) -> f32 {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    let flat = mlxcel_core::reshape(&f, &[-1]);
    let am = mlxcel_core::mean_axis(&mlxcel_core::abs(&flat), 0, false);
    mlxcel_core::eval(&am);
    let absmean = mlxcel_core::item_f32(&am);
    let first4: Vec<f32> = (0..4)
        .map(|k| {
            let x = mlxcel_core::slice(&flat, &[k], &[k + 1]);
            mlxcel_core::eval(&x);
            (mlxcel_core::item_f32(&x) * 100000.0).round() / 100000.0
        })
        .collect();
    println!(
        "[{name}] shape={:?} absmean={absmean:.5} first4={first4:?}",
        mlxcel_core::array_shape(a),
    );
    absmean
}

#[test]
fn sam_and_clip_encoders_match_reference() {
    let Some(dir) = model_dir() else {
        eprintln!("Skipping DeepSeek-OCR encoder parity: checkpoint not found");
        return;
    };
    let weights = mlxcel_core::weights::load_weights_from_dir(&dir).expect("load weights");

    // Deterministic input: ((i % 255)/255 - 0.5)/0.5, (1, 1024, 1024, 3), bf16.
    let n = 1024usize * 1024 * 3;
    let v: Vec<f32> = (0..n)
        .map(|i| (((i % 255) as f32) / 255.0 - 0.5) / 0.5)
        .collect();
    let x = mlxcel_core::from_slice_f32(&v, &[1, 1024, 1024, 3]);
    let x = mlxcel_core::astype(&x, mlxcel_core::dtype::BFLOAT16);

    let sam = SamEncoder::from_weights(&weights, "sam_model", SamConfig::default())
        .expect("build SAM encoder");
    let sam_out = sam.forward(&x);
    mlxcel_core::eval(&sam_out);
    let sam_am = sig(&sam_out, "SAM");

    let clip = ClipEncoder::from_weights(&weights, "vision_model", ClipConfig::default())
        .expect("build CLIP encoder");
    let clip_out = clip.forward(&sam_out);
    mlxcel_core::eval(&clip_out);
    let clip_am = sig(&clip_out, "CLIP");

    assert_eq!(mlxcel_core::array_shape(&sam_out), vec![1, 16, 16, 1024]);
    assert_eq!(mlxcel_core::array_shape(&clip_out), vec![1, 257, 1024]);
    // Reference (mlx-vlm) on this deterministic input: SAM absmean 0.04304,
    // CLIP absmean 0.14356. bf16 tolerance widens through the CLIP's 24 layers.
    assert!(
        (sam_am - 0.04304).abs() < 0.002,
        "SAM absmean {sam_am} off reference"
    );
    assert!(
        (clip_am - 0.14356).abs() < 0.005,
        "CLIP absmean {clip_am} off reference"
    );
}
