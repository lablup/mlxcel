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

//! Seed-fixed parity of whole runs against upstream on the real checkpoints.
//!
//! `#[ignore]`: each needs a 20 GB checkpoint and a GPU. The checkpoints are
//! read from `models/mlx/z-image-turbo-bf16` and `models/mlx/z-image-bf16`
//! (override with `MLXCEL_Z_IMAGE_TURBO` / `MLXCEL_Z_IMAGE_BASE`); a missing
//! one skips. The committed `tests/fixtures/imagegen/z_image_parity.safetensors`
//! (see `tests/fixtures/generate_z_image_fixtures.py`) holds the upstream
//! initial-noise checksum, 4x4-pooled final latents and 4x4-pooled image per
//! case. With `MLXCEL_Z_IMAGE_ORACLE_DIR` pointing at the generator's `--out`
//! the full-resolution latents and PNGs are compared too (and the img2img case,
//! whose source image lives there, runs). `MLXCEL_Z_IMAGE_OUT` saves the PNGs.
//!
//! ```bash
//! MLX_ENABLE_TF32=0 <test binary> z_image::tests --ignored --test-threads=1
//! ```

use std::path::{Path, PathBuf};

use super::*;
use crate::imagegen::output::encode_png;
use mlxcel_core::weights::{WeightMap, load_safetensors};
use mlxcel_core::{MlxArray, dtype};

/// Final-latents relative RMS bounds, per case.
///
/// The issue asks for 1e-2. That is below upstream's own reproducibility on
/// this host (mlx 0.32.4 CUDA, GB10, TF32 off, Turbo case, final latents and
/// PNG PSNR against the unmodified upstream run): computing only the caption
/// embedder's linear in f32 instead of bf16 moves upstream by 2.7e-2 (41.3 dB);
/// running only its text encoder's attention in bf16 instead of f32 moves it
/// by 1.0e-1 (32.1 dB, caption features 1.9e-3); TF32 on moves it by 9.5e-2
/// (31.6 dB). Eight Euler steps through 30 joint layers amplify bf16-level
/// differences in the caption stream. The bounds sit above this port's
/// measured drift (Turbo 8.6e-2, base 6.2e-2, img2img 1.7e-2) and within the
/// floor; the PSNR bound and the bit-identical initial noise are the issue's.
/// Running this port's conditioner attention in f32 like upstream (#2666)
/// leaves the caption features at 9.3e-3 and only reshuffles the drift
/// (Turbo 6.4e-2, base 1.0e-1, img2img 1.8e-2).
const TURBO_LATENTS: f64 = 1.2e-1;
const BASE_LATENTS: f64 = 1.0e-1;
const EDIT_LATENTS: f64 = 5.0e-2;
const MIN_PSNR_DB: f64 = 30.0;

fn config(shift: f32) -> ZImageConfig {
    ZImageConfig::new(config::ZImageTransformerConfig::default(), shift).unwrap()
}

fn edit_request(strength: Option<f64>, images: usize) -> ImageRequest {
    let mut req = ImageRequest::generate("a fox", 1);
    req.task = ImageTask::Edit;
    req.reference_images = vec![image::DynamicImage::new_rgb8(16, 16); images];
    if let Some(strength) = strength {
        req.extra
            .insert("strength".into(), serde_json::json!(strength));
    }
    req
}

fn rejection(config: &ZImageConfig, req: &ImageRequest) -> String {
    let err = validate_request(config, req).expect_err("request is rejected");
    err.downcast_ref::<InvalidImageRequest>()
        .expect("a content error is InvalidImageRequest (400 on the server)")
        .0
        .clone()
}

#[test]
fn validate_request_rejects_bad_strength_and_image_count() {
    let turbo = config(3.0);
    assert_eq!(
        validate_request(&turbo, &edit_request(None, 1))
            .unwrap()
            .strength,
        0.6
    );
    for strength in [0.0, 1.5, -0.1] {
        let message = rejection(&turbo, &edit_request(Some(strength), 1));
        assert!(message.contains("strength must be in (0, 1]"), "{message}");
    }
    for images in [0, 2] {
        let message = rejection(&turbo, &edit_request(None, images));
        assert!(message.contains("exactly one source image"), "{message}");
    }
}

#[test]
fn validate_request_checks_guidance_and_steps_per_variant() {
    let (turbo, base) = (config(3.0), config(6.0));
    let mut req = ImageRequest::generate("a fox", 1);
    req.guidance = Some(4.0);
    rejection(&turbo, &req);
    assert!(validate_request(&base, &req).is_ok());
    req.guidance = None;
    req.steps = Some(1);
    rejection(&base, &req);
    let mut bad = ImageRequest::generate("a fox", 1);
    bad.extra
        .insert("cfg_truncation".into(), serde_json::json!(2.0));
    rejection(&base, &bad);
}

fn checkpoint(var: &str, name: &str) -> Option<PathBuf> {
    let path = std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/mlx")
            .join(name)
    });
    if path.join("transformer/config.json").exists() {
        Some(path)
    } else {
        eprintln!("skipping: {} is absent", path.display());
        None
    }
}

fn values(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(a, dtype::FLOAT32))
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}

fn fnv1a32(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811C_9DC5_u32, |h, b| {
        (h ^ u32::from(*b)).wrapping_mul(0x0100_0193)
    })
}

/// Average-pool `[.., H, W]` laid out as `rows x cols` planes by 4.
fn pool4(data: &[f32], planes: usize, h: usize, w: usize, interleaved: usize) -> Vec<f32> {
    // `interleaved` > 1 means HWC with that many channels; otherwise CHW planes.
    let (ph, pw) = (h / 4, w / 4);
    let mut out = Vec::with_capacity(planes * ph * pw * interleaved.max(1));
    if interleaved > 1 {
        for y in 0..ph {
            for x in 0..pw {
                for c in 0..interleaved {
                    let mut s = 0.0;
                    for dy in 0..4 {
                        for dx in 0..4 {
                            s += data[((4 * y + dy) * w + 4 * x + dx) * interleaved + c];
                        }
                    }
                    out.push(s / 16.0);
                }
            }
        }
    } else {
        for p in 0..planes {
            for y in 0..ph {
                for x in 0..pw {
                    let mut s = 0.0;
                    for dy in 0..4 {
                        for dx in 0..4 {
                            s += data[p * h * w + (4 * y + dy) * w + 4 * x + dx];
                        }
                    }
                    out.push(s / 16.0);
                }
            }
        }
    }
    out
}

fn committed() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/z_image_parity.safetensors"),
    )
    .expect("parity fixture loads")
}

fn oracle_dir() -> Option<PathBuf> {
    std::env::var_os("MLXCEL_Z_IMAGE_ORACLE_DIR").map(PathBuf::from)
}

fn save_png(case: &str, image: &GeneratedImage) {
    if let Some(dir) = std::env::var_os("MLXCEL_Z_IMAGE_OUT") {
        let path = Path::new(&dir).join(format!("{case}.png"));
        std::fs::write(&path, encode_png(image).unwrap()).unwrap();
        eprintln!("{case}: wrote {}", path.display());
    }
}

/// Run `req` on `model`, check it against the committed summary of `case`
/// (and the full oracle output when available), and return the image.
fn check_case(
    model: &ZImageModel,
    case: &str,
    req: &ImageRequest,
    max_latents: f64,
) -> GeneratedImage {
    let fixture = committed();
    if let Some(dir) = oracle_dir() {
        // Stage diagnostics: the conditioner alone, then one transformer
        // forward on upstream's own inputs.
        let full = load_safetensors(dir.join(format!("{case}.safetensors"))).unwrap();
        let cap = model.encode(&req.prompt).unwrap();
        let cap_err = rel_rms(&values(&cap), &values(&full["cap0"]));
        let cap_ref = mlxcel_core::astype(&full["cap0"], mlxcel_core::array_dtype(&cap));
        let sigma_start = if req.task == ImageTask::Generate {
            0
        } else {
            let options = ZImageOptions::from_extra(&req.extra).unwrap();
            img2img_start_index(req.steps.unwrap(), options.strength)
        };
        let steps = req.steps.unwrap();
        let schedule = match req.task {
            ImageTask::Generate => model.config.generate_schedule(steps).unwrap(),
            ImageTask::Edit => model.config.edit_schedule(steps).unwrap(),
        };
        let t = mlxcel_core::from_slice_f32(&[1.0 - schedule.sigmas()[sigma_start]], &[1]);
        let pred = model
            .transformer
            .forward(&full["start_latents"], &t, &cap_ref)
            .unwrap();
        let pred_err = rel_rms(&values(&pred), &values(&full["first_pred"]));
        eprintln!(
            "{case}: caption features rel-RMS {cap_err:.3e}, first transformer forward (upstream inputs) rel-RMS {pred_err:.3e}"
        );
    }
    let t0 = std::time::Instant::now();
    let latents = model
        .with_conditioning(req, |cond| model.sample_latents(req, 0, cond))
        .unwrap();
    let pixels = pipeline::decode_to_pixels(&model.vae, &latents).unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    let image = GeneratedImage {
        pixels,
        width: req.width,
        height: req.height,
        channels: 3,
        seed: req.seed,
        revised_prompt: None,
    };
    save_png(case, &image);
    let (h, w) = (req.height as usize / 8, req.width as usize / 8);
    let final_latents = values(&latents);

    if req.task == ImageTask::Generate {
        let noise = pipeline::initial_noise(req.seed, 16, req.width, req.height);
        let noise_bytes = mlxcel_core::array_to_raw_bytes(&noise);
        let want = mlxcel_core::array_to_raw_bytes(&fixture[&format!("{case}.start_fnv")]);
        let want = u32::from_le_bytes(want[..4].try_into().unwrap());
        assert_eq!(
            fnv1a32(&noise_bytes),
            want,
            "{case}: initial noise checksum"
        );
        let head = values(&fixture[&format!("{case}.start_head")]);
        assert_eq!(
            &values(&noise)[..head.len()],
            head.as_slice(),
            "{case}: noise head"
        );
    }
    let pooled = pool4(&final_latents, 16, h, w, 1);
    let latent_err = rel_rms(&pooled, &values(&fixture[&format!("{case}.latents_pool4")]));
    let image_f32: Vec<f32> = image.pixels.iter().map(|&p| f32::from(p)).collect();
    let pooled_image: Vec<u8> = pool4(&image_f32, 1, req.height as usize, req.width as usize, 3)
        .iter()
        .map(|v| v.round_ties_even() as u8)
        .collect();
    let want_image = mlxcel_core::array_to_raw_bytes(&fixture[&format!("{case}.image_pool4")]);
    let pooled_psnr = psnr(&pooled_image, &want_image);
    eprintln!(
        "{case}: {elapsed:.1}s, pooled latents rel-RMS {latent_err:.3e}, pooled image PSNR {pooled_psnr:.2} dB"
    );
    let pooled_err = latent_err;

    if let Some(dir) = oracle_dir() {
        let full = load_safetensors(dir.join(format!("{case}.safetensors"))).unwrap();
        let latent_err = rel_rms(&final_latents, &values(&full["final_latents"]));
        let png = image::open(dir.join(format!("{case}.png")))
            .unwrap()
            .to_rgb8();
        let image_psnr = psnr(&image.pixels, png.as_raw());
        eprintln!(
            "{case}: full-res final latents rel-RMS {latent_err:.3e}, PNG PSNR {image_psnr:.2} dB"
        );
        assert!(latent_err < max_latents, "{case}: latents {latent_err}");
        assert!(image_psnr >= MIN_PSNR_DB, "{case}: PSNR {image_psnr}");
    }
    assert!(
        pooled_err < max_latents,
        "{case}: pooled latents {pooled_err}"
    );
    assert!(
        pooled_psnr >= MIN_PSNR_DB,
        "{case}: pooled PSNR {pooled_psnr}"
    );
    image
}

fn red_panda(steps: u32, guidance: f32) -> ImageRequest {
    let mut req = ImageRequest::generate("A red panda serving tea in a moonlit bamboo forest", 42);
    req.steps = Some(steps);
    req.guidance = Some(guidance);
    req
}

#[test]
#[ignore = "needs models/mlx/z-image-turbo-bf16 and a GPU"]
fn turbo_generation_and_edit_match_upstream() {
    let Some(root) = checkpoint("MLXCEL_Z_IMAGE_TURBO", "z-image-turbo-bf16") else {
        return;
    };
    let model = ZImageModel::load(&root).unwrap();
    assert_eq!(model.config().variant, config::Variant::Turbo);
    check_case(&model, "turbo", &red_panda(8, 0.0), TURBO_LATENTS);

    let Some(dir) = oracle_dir() else {
        eprintln!("skipping the img2img case: MLXCEL_Z_IMAGE_ORACLE_DIR is unset");
        return;
    };
    let fox = image::open(dir.join("fox.png")).unwrap();
    let mut req = ImageRequest::generate("the same fox at night under a full moon", 42);
    req.task = ImageTask::Edit;
    req.reference_images = vec![fox];
    req.steps = Some(8);
    req.guidance = Some(0.0);
    req.extra.insert("strength".into(), serde_json::json!(0.6));
    check_case(&model, "edit", &req, EDIT_LATENTS);
}

#[test]
#[ignore = "needs models/mlx/z-image-bf16 and a GPU"]
fn base_generation_with_cfg_matches_upstream() {
    let Some(root) = checkpoint("MLXCEL_Z_IMAGE_BASE", "z-image-bf16") else {
        return;
    };
    let model = ZImageModel::load(&root).unwrap();
    assert_eq!(model.config().variant, config::Variant::Base);
    let mut req = red_panda(50, 4.0);
    req.negative_prompt = Some("blurry, low quality".into());
    req.extra
        .insert("cfg_truncation".into(), serde_json::json!(1.0));
    check_case(&model, "base", &req, BASE_LATENTS);
}
