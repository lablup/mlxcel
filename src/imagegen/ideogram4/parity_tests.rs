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

//! Seed-fixed parity of the real `ideogram-4-fp8` snapshot against the
//! upstream CUDA oracle (`tests/fixtures/generate_ideogram4_fixtures.py`,
//! mlx 0.32.4 CUDA on NVIDIA GB10, mlx-vlm 0.7.8, `MLX_ENABLE_TF32=0`).
//!
//! Case `turbo_512`: the issue's red-square JSON caption, 512x512,
//! `V4_TURBO_12`, seed 0. Always checked against the committed summary
//! (`tests/fixtures/imagegen/ideogram4_parity.safetensors`): the token ids,
//! the initial noise (bit-identical), the 4x4-token-pooled final latents and
//! the 8x8-pooled image. With `MLXCEL_IDEOGRAM4_ORACLE_DIR` set to the
//! generator's `--out` directory it also checks the full-resolution PNG
//! (PSNR) and reports, as drift, the text features, the first step's
//! velocities, the full final latents, and a rerun on upstream's own text
//! features. `MLXCEL_IDEOGRAM4_OUT` names a directory to write the PNG to.
//!
//! Run (GPU, about 60 GB peak, one at a time):
//! `MLX_ENABLE_TF32=0 <test binary> imagegen::ideogram4::parity_tests --ignored --test-threads=1 --nocapture`

use std::path::{Path, PathBuf};

use mlxcel_core::weights::{WeightMap, load_safetensors};
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::json;

use super::pipeline::{Denoiser, SequenceLayout, initial_noise};
use super::scheduler::{get_schedule_for_resolution, make_step_intervals};
use super::tests::checkpoint;
use super::*;
use crate::imagegen::testing::{fnv1a32_bytes, psnr, rel_rms, values};

const PROMPT: &str = r#"{"high_level_description":"A red square icon.","compositional_deconstruction":{"background":"A white background.","elements":[{"type":"obj","desc":"A centered red square."}]}}"#;

/// Per-case bounds, each set from the value measured on GB10 (in the comment
/// at the call site) with headroom for kernel accumulation order.
struct Bounds {
    /// rel-RMS of the 4x4-token-pooled final latents.
    latents_pool4: f64,
    /// PSNR of the 8x8-pooled image against upstream's, dB.
    image_pool8_psnr: f64,
    /// PSNR of the full PNG against upstream's (needs the oracle directory).
    png_psnr: f64,
}

fn summary() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/ideogram4_parity.safetensors"),
    )
    .expect("parity summary loads")
}

fn oracle_dir() -> Option<PathBuf> {
    std::env::var_os("MLXCEL_IDEOGRAM4_ORACLE_DIR").map(PathBuf::from)
}

fn ints(a: &MlxArray) -> Vec<i32> {
    values(a).into_iter().map(|v| v as i32).collect()
}

/// Mean over `k x k` image-token patches of `[gh * gw, c]` latents.
fn pool_tokens(z: &[f32], gh: usize, gw: usize, c: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f64; (gh / k) * (gw / k) * c];
    for y in 0..gh {
        for x in 0..gw {
            let o = ((y / k) * (gw / k) + x / k) * c;
            for ch in 0..c {
                out[o + ch] += f64::from(z[(y * gw + x) * c + ch]);
            }
        }
    }
    out.iter().map(|v| (v / (k * k) as f64) as f32).collect()
}

/// `round(mean)` over `k x k` pixel patches of an HWC image, as `u8`.
fn pool_image(pixels: &[u8], h: usize, w: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0.0_f64; (h / k) * (w / k) * 3];
    for y in 0..h {
        for x in 0..w {
            let o = ((y / k) * (w / k) + x / k) * 3;
            for c in 0..3 {
                out[o + c] += f64::from(pixels[(y * w + x) * 3 + c]);
            }
        }
    }
    out.iter()
        .map(|v| (v / (k * k) as f64).round() as u8)
        .collect()
}

fn slice_text(features: &MlxArray, text: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(features);
    mlxcel_core::slice(features, &[0, 0, 0], &[1, text, s[2]])
}

/// Denoise `seed` from `features` and decode it.
fn run(
    model: &Ideogram4Model,
    prepared: &PreparedRequest,
    layout: &SequenceLayout,
    features: &MlxArray,
    width: u32,
    height: u32,
    seed: u64,
) -> (Vec<f32>, Vec<u8>) {
    let conditional = model.conditional.borrow();
    let unconditional = model.unconditional.borrow();
    let llm = conditional
        .as_ref()
        .unwrap()
        .project_llm_features(features, &layout.indicator)
        .unwrap();
    let denoiser = Denoiser {
        conditional: conditional.as_ref().unwrap(),
        unconditional: unconditional.as_ref().unwrap(),
        llm: &llm,
        layout,
    };
    let z = denoiser
        .denoise(
            initial_noise(seed, layout.num_image_tokens),
            &prepared.plan,
            width,
            height,
        )
        .unwrap();
    let pixels = model.decode(&z, layout).unwrap();
    (values(&z), pixels)
}

fn run_case(case: &str, root: &Path, req: &ImageRequest, bounds: &Bounds) {
    let (width, height) = (req.width, req.height);
    let model = Ideogram4Model::load(root).unwrap();
    let prepared = model.prepare(req).unwrap();
    let want = summary();
    assert_eq!(
        prepared.token_ids,
        ints(&want[&format!("{case}.token_ids")]),
        "{case} token ids"
    );
    let layout = SequenceLayout::new(prepared.token_ids.len(), width, height);

    let noise = initial_noise(req.seed, layout.num_image_tokens);
    let fnv = fnv1a32_bytes(&mlxcel_core::array_to_raw_bytes(&noise));
    let want_fnv = mlxcel_core::array_to_raw_bytes(&want[&format!("{case}.noise_fnv")]);
    let want_fnv = u32::from_le_bytes(want_fnv[..4].try_into().unwrap());
    assert_eq!(fnv, want_fnv, "{case} initial noise checksum");
    assert_eq!(
        values(&noise)[..64],
        values(&want[&format!("{case}.noise_head")])[..],
        "{case} initial noise head"
    );
    eprintln!("{case}: initial noise bit-identical (FNV-1a {fnv:#010x})");

    let features = model
        .encode_text(&prepared.token_ids, layout.num_image_tokens)
        .unwrap();
    let oracle = oracle_dir().map(|dir| {
        load_safetensors(dir.join(format!("{case}.safetensors"))).expect("oracle case loads")
    });
    if let Some(oracle) = &oracle {
        let err = rel_rms(
            &values(&slice_text(&features, layout.num_text_tokens)),
            &values(&oracle["llm_features_text"]),
        );
        eprintln!("{case}: text features rel-RMS {err:.4e} (drift)");
    }
    model.ensure_transformers_and_vae().unwrap();

    if let Some(oracle) = &oracle {
        let conditional = model.conditional.borrow();
        let unconditional = model.unconditional.borrow();
        let llm = conditional
            .as_ref()
            .unwrap()
            .project_llm_features(&features, &layout.indicator)
            .unwrap();
        let denoiser = Denoiser {
            conditional: conditional.as_ref().unwrap(),
            unconditional: unconditional.as_ref().unwrap(),
            llm: &llm,
            layout: &layout,
        };
        let plan = &prepared.plan;
        let schedule = get_schedule_for_resolution(height, width, plan.mu, plan.std);
        let t0 = schedule.at(make_step_intervals(plan.steps).unwrap()[plan.steps as usize]);
        let (pos, neg) = denoiser.velocities(&noise, t0).unwrap();
        let pos_err = rel_rms(&values(&pos), &values(&oracle["step0_pos"]));
        let neg_err = rel_rms(&values(&neg), &values(&oracle["step0_neg"]));
        eprintln!(
            "{case}: first-step velocities rel-RMS conditional {pos_err:.4e}, unconditional {neg_err:.4e} (drift)"
        );
    }

    let (latents, pixels) = run(
        &model, &prepared, &layout, &features, width, height, req.seed,
    );
    let (gh, gw) = (layout.grid_h as usize, layout.grid_w as usize);
    let pooled = pool_tokens(&latents, gh, gw, 128, 4);
    let drift = rel_rms(&pooled, &values(&want[&format!("{case}.latents_pool4")]));
    let image_pool: Vec<u8> = pool_image(&pixels, height as usize, width as usize, 8);
    let want_pool: Vec<u8> = values(&want[&format!("{case}.image_pool8")])
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let image_psnr = psnr(&image_pool, &want_pool);
    eprintln!("{case}: latents_pool4 rel-RMS {drift:.4e}, image_pool8 PSNR {image_psnr:.2} dB");
    if let Some(dir) = std::env::var_os("MLXCEL_IDEOGRAM4_OUT") {
        let path = PathBuf::from(dir).join(format!("{case}.png"));
        image::RgbImage::from_raw(width, height, pixels.clone())
            .unwrap()
            .save(&path)
            .unwrap();
        eprintln!("{case}: wrote {}", path.display());
    }
    assert!(
        drift <= bounds.latents_pool4,
        "{case} latents_pool4 rel-RMS {drift} > {}",
        bounds.latents_pool4
    );
    assert!(
        image_psnr >= bounds.image_pool8_psnr,
        "{case} image_pool8 PSNR {image_psnr} < {}",
        bounds.image_pool8_psnr
    );

    let (Some(oracle), Some(dir)) = (oracle, oracle_dir()) else {
        return;
    };
    let reference = image::open(dir.join(format!("{case}.png")))
        .unwrap()
        .to_rgb8();
    let full = rel_rms(&latents, &values(&oracle["final_latents"]));
    let png_psnr = psnr(&pixels, reference.as_raw());
    eprintln!("{case}: final latents rel-RMS {full:.4e} (drift), PNG PSNR {png_psnr:.2} dB");

    // Upstream's own text features: isolates the transformers, sampler and
    // VAE from the text encoder's bf16 drift.
    let padding = mlxcel_core::zeros(
        &[
            1,
            layout.num_image_tokens,
            mlxcel_core::array_shape(&features)[2],
        ],
        mlxcel_core::dtype::FLOAT32,
    );
    let upstream_features = mlxcel_core::concatenate(&oracle["llm_features_text"], &padding, 1);
    drop(features);
    let (latents, pixels) = run(
        &model,
        &prepared,
        &layout,
        &upstream_features,
        width,
        height,
        req.seed,
    );
    let full = rel_rms(&latents, &values(&oracle["final_latents"]));
    let cond_psnr = psnr(&pixels, reference.as_raw());
    eprintln!(
        "{case} on upstream's text features: final latents rel-RMS {full:.4e} (drift), PNG PSNR {cond_psnr:.2} dB"
    );
    assert!(
        png_psnr >= bounds.png_psnr,
        "{case} PNG PSNR {png_psnr} < {}",
        bounds.png_psnr
    );
}

#[test]
#[ignore = "runs the real ideogram-4-fp8 checkpoint (GPU, about 60 GB peak)"]
fn turbo_512_matches_upstream() {
    let Some(root) = checkpoint() else { return };
    let mut req = ImageRequest::generate(PROMPT, 0);
    req.extra
        .insert("sampler_preset".into(), json!("V4_TURBO_12"));
    run_case(
        "turbo_512",
        &root,
        &req,
        // Measured on GB10: latents_pool4 3.16e-2, image_pool8 43.7 dB, PNG
        // 37.1 dB (final latents 6.2e-2 drift). On upstream's own text
        // features: PNG 46.0 dB, final latents 2.9e-2. First-step velocities
        // on the same noise: 6.2e-3 conditional, 3.2e-3 unconditional.
        &Bounds {
            latents_pool4: 6e-2,
            image_pool8_psnr: 38.0,
            png_psnr: 30.0,
        },
    );
}
