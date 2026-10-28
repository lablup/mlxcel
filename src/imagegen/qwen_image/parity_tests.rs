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

//! Seed-fixed parity of the whole pipeline against upstream mlx-vlm
//! `73d366ec` on the CUDA backend.
//!
//! `tests/fixtures/imagegen/qwen_image_parity.safetensors` is the committed
//! summary (`generate_qwen_image_fixtures.py real`): per case the initial
//! noise checksum, the 8x8-cell average pools of the final latents and the
//! 32x32-pixel average pools of the image. The full tensors and PNGs of the
//! same run stay outside the repository; the ignored tests read them from
//! `MLXCEL_QWEN_IMAGE_ORACLE_DIR` when it is set (the edit case needs it for
//! its reference PNG) and write this port's PNGs to `MLXCEL_QWEN_IMAGE_OUT`.
//!
//! Gates, per case (generate: 1024x1024, 4 steps, seed 7; edit: one RGBA
//! 1024x1024 reference, 4 steps, seed 7):
//! * the initial noise is bit-identical to upstream's;
//! * the pooled final latents stay within the case's [`Bounds`]; a broken
//!   port (swapped RoPE axes, a dropped modulation row) lands far outside
//!   them (numbers at [`GEN_BOUNDS`]);
//! * the decoded PNG is within [`PNG_PSNR`] of upstream's;
//! * the edit without the KV cache is within [`PNG_PSNR`] of the edit with it.
//!
//! The full-resolution final-latent rel-RMS is printed as drift only.

use std::path::{Path, PathBuf};

use image::DynamicImage;
use mlxcel_core::MlxArray;
use mlxcel_core::weights::{WeightMap, load_safetensors};

use super::pipeline::{self, initial_noise};
use super::*;
use crate::imagegen::testing::{fnv1a32_bytes, pool_planes, psnr, rel_rms, values};

pub(crate) const GEN_PROMPT: &str = "A red ceramic teapot on a wooden table in soft window light";
pub(crate) const EDIT_PROMPT: &str = "Change the teapot to cobalt blue";
const SEED: u64 = 7;
const STEPS: u32 = 4;
const SIZE: u32 = 1024;

/// Least PSNR between this port's PNG and upstream's, and between the edit
/// with and without the KV cache.
const PNG_PSNR: f64 = 30.0;

pub(crate) fn model_dir() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/mlx/qwen-image-2.1");
    root.join("processor/tokenizer.json")
        .is_file()
        .then_some(root)
}

pub(crate) fn tiny_refs() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/qwen_image_tiny_refs.safetensors"),
    )
    .expect("fixture loads")
}

pub(crate) fn parity() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/qwen_image_parity.safetensors"),
    )
    .expect("fixture loads")
}

fn oracle_dir() -> Option<PathBuf> {
    std::env::var_os("MLXCEL_QWEN_IMAGE_ORACLE_DIR").map(PathBuf::from)
}

/// `<MLXCEL_QWEN_IMAGE_ORACLE_DIR>/<case>.safetensors`, when the variable is set.
pub(crate) fn oracle(case: &str) -> Option<WeightMap> {
    let path = oracle_dir()?.join(format!("{case}.safetensors"));
    Some(load_safetensors(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

fn save_png(name: &str, pixels: &[u8], channels: u8) {
    let Some(dir) = std::env::var_os("MLXCEL_QWEN_IMAGE_OUT") else {
        return;
    };
    let path = Path::new(&dir).join(name);
    let color = if channels == 4 {
        image::ExtendedColorType::Rgba8
    } else {
        image::ExtendedColorType::Rgb8
    };
    image::save_buffer(&path, pixels, SIZE, SIZE, color).expect("write PNG");
}

/// FNV-1a of the bf16 bits and the first 16 values of the initial noise.
fn noise_summary(seed: u64) -> (u32, Vec<f32>) {
    let grid = (SIZE / config::DOWNSAMPLE) as i32;
    let noise = initial_noise(grid, grid, 64, seed);
    let noise = mlxcel_core::contiguous(&noise, false);
    mlxcel_core::eval(&noise);
    let bytes = mlxcel_core::try_array_to_raw_bytes(&noise).unwrap();
    (fnv1a32_bytes(&bytes), values(&noise)[..16].to_vec())
}

#[test]
fn initial_noise_is_bit_identical_to_upstream() {
    let want = parity();
    for case in ["gen", "edit"] {
        let (fnv, head) = noise_summary(SEED);
        let raw = mlxcel_core::try_array_to_raw_bytes(&want[&format!("{case}.noise_fnv")]).unwrap();
        let want_fnv = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        assert_eq!(fnv, want_fnv, "{case} noise checksum");
        assert_eq!(
            head,
            values(&want[&format!("{case}.noise_head")]),
            "{case} noise head"
        );
    }
    // Another seed draws other noise.
    assert_ne!(noise_summary(SEED + 1).0, noise_summary(SEED).0);
}

/// Bounds on the pooled final latents of one case.
struct Bounds {
    /// rel-RMS of the 8x8-cell pools against upstream's.
    latents_pool: f64,
}

/// `[1, h*w, C]` latents to `[C, h/8, w/8]` cell means, as upstream's
/// `pool_tokens`.
fn latents_pool(latents: &MlxArray) -> Vec<f32> {
    let grid = (SIZE / config::DOWNSAMPLE) as usize;
    let chw = mlxcel_core::transpose_axes(latents, &[0, 2, 1]);
    pool_planes(&values(&chw), 64, grid, grid, 8)
}

/// `[H, W, C]` bytes to `[C, H/32, W/32]` cell means, as upstream's `pool_image`.
fn image_pool(pixels: &[u8], channels: usize) -> Vec<f32> {
    let n = SIZE as usize;
    let mut planes = vec![0.0f32; channels * n * n];
    for (i, &p) in pixels.iter().enumerate() {
        planes[(i % channels) * n * n + i / channels] = f32::from(p);
    }
    pool_planes(&planes, channels, n, n, 32)
}

fn request(
    task: ImageTask,
    references: Vec<DynamicImage>,
    extra: serde_json::Value,
) -> ImageRequest {
    let mut req = ImageRequest::generate(
        if task == ImageTask::Edit {
            EDIT_PROMPT
        } else {
            GEN_PROMPT
        },
        SEED,
    );
    req.task = task;
    req.reference_images = references;
    req.width = SIZE;
    req.height = SIZE;
    req.steps = Some(STEPS);
    req.extra = extra.as_object().cloned().unwrap_or_default();
    req
}

/// Check one finished image of `case` (`tag` names its pools in the
/// summary) against the bounds and, with the oracle directory, upstream's
/// PNG.
fn check_image(
    case: &str,
    tag: &str,
    latents: &MlxArray,
    pixels: &[u8],
    channels: usize,
    bounds: &Bounds,
) {
    let want = parity();
    let drift = rel_rms(
        &latents_pool(latents),
        &values(&want[&format!("{case}.{tag}latents_pool")]),
    );
    let pool_psnr = psnr(
        &image_pool(pixels, channels)
            .iter()
            .map(|v| v.round() as u8)
            .collect::<Vec<_>>(),
        &values(&want[&format!("{case}.{tag}image_pool")])
            .iter()
            .map(|v| v.round() as u8)
            .collect::<Vec<_>>(),
    );
    eprintln!("{case} {tag}: latents pool rel-RMS {drift:.4e}, image pool PSNR {pool_psnr:.2} dB");
    assert!(
        drift <= bounds.latents_pool,
        "{case} {tag}latents pool rel-RMS {drift:.4e} > {}",
        bounds.latents_pool
    );
}

fn oracle_png_psnr(case: &str, key: &str, latents: &MlxArray, pixels: &[u8]) -> Option<f64> {
    let full = oracle(case)?;
    let drift = rel_rms(
        &values(latents),
        &values(&full[&format!("{key}final_latents")]),
    );
    let theirs = mlxcel_core::try_array_to_raw_bytes(&full[&format!("{key}pixels")]).unwrap();
    let png = psnr(pixels, &theirs);
    eprintln!("{case} {key}: final latents rel-RMS {drift:.4e} (drift), PNG PSNR {png:.2} dB");
    Some(png)
}

/// Generation: 1024x1024, 4 steps, seed 7. Measured 5.6e-3 (bf16, the CUDA
/// default) and 8.5e-3 (f16 transformer). Broken arms on the same case:
/// RoPE height and width axes swapped 7.7e-1, every token on the real
/// timestep's modulation row (the zero-timestep row dropped) 2.3e-1.
const GEN_BOUNDS: Bounds = Bounds { latents_pool: 2e-2 };
/// Edit with one RGBA reference, 4 steps, seed 7 (KV cache on and off).
/// Measured 4.5e-3 (bf16) and 7.1e-3 (f16 transformer).
const EDIT_BOUNDS: Bounds = Bounds { latents_pool: 2e-2 };

#[test]
#[ignore = "runs the real Qwen-Image-2.1 checkpoint (GPU, ~33 GB)"]
fn generate_matches_upstream() {
    let root = model_dir().expect("models/mlx/qwen-image-2.1 is present");
    let model = QwenImageModel::load(&root).unwrap();
    let req = request(ImageTask::Generate, Vec::new(), serde_json::json!({}));
    let resolved = model.resolve(&req).unwrap();
    let cond = model.encode(&req, &resolved, &[]).unwrap();
    assert!(model.text_encoder.borrow().is_none(), "conditioner evicted");
    let latents = model.sample_latents(&resolved, &cond, None, SEED).unwrap();
    let pixels = model.decode(&latents, &resolved, 3).unwrap();
    save_png("qwen_image_gen.png", &pixels, 3);
    check_image("gen", "", &latents, &pixels, 3, &GEN_BOUNDS);
    let Some(png) = oracle_png_psnr("gen_full", "", &latents, &pixels) else {
        return;
    };
    assert!(png >= PNG_PSNR, "generate PNG PSNR {png:.2} < {PNG_PSNR}");
    // Per stage, on upstream's own inputs.
    let full = oracle("gen_full").unwrap();
    let slot = model.denoiser.borrow();
    let denoiser = slot.as_ref().unwrap();
    let dt = denoiser.transformer.dtype();
    let step0 = denoiser
        .transformer
        .forward(
            &mlxcel_core::astype(&full["step0_input"], dt),
            &mlxcel_core::astype(&full["cond"], dt),
            &full["step0_timestep"],
            (1, 64, 64),
            None,
            None,
        )
        .unwrap();
    let rel = rel_rms(&values(&step0), &values(&full["step0_out"]));
    eprintln!("gen: first-step velocity on upstream's inputs rel-RMS {rel:.4e}");
    let rel = rel_rms(&values(&cond.positive.hidden), &values(&full["cond"]));
    eprintln!("gen: conditioner rel-RMS {rel:.4e}");
    let theirs = mlxcel_core::try_array_to_raw_bytes(&full["pixels"]).unwrap();
    let decoded =
        pipeline::decode_to_pixels(&denoiser.vae, &full["final_latents"], 64, 64, 3).unwrap();
    let vae_psnr = psnr(&decoded, &theirs);
    eprintln!("gen: VAE decode of upstream's final latents PSNR {vae_psnr:.2} dB");
    assert!(vae_psnr >= 40.0, "VAE decode PSNR {vae_psnr:.2}");
}

#[test]
#[ignore = "runs the real Qwen-Image-2.1 checkpoint (GPU, ~35 GB); needs MLXCEL_QWEN_IMAGE_ORACLE_DIR"]
fn edit_matches_upstream_with_and_without_the_kv_cache() {
    let root = model_dir().expect("models/mlx/qwen-image-2.1 is present");
    let dir = oracle_dir().expect("MLXCEL_QWEN_IMAGE_ORACLE_DIR holds edit_ref_rgba.png");
    let reference = image::open(dir.join("edit_ref_rgba.png")).unwrap();
    let model = QwenImageModel::load(&root).unwrap();
    let mut images = Vec::new();
    for (tag, kv) in [("kv_", true), ("nokv_", false)] {
        let req = request(
            ImageTask::Edit,
            vec![reference.clone()],
            serde_json::json!({ "use_kv_cache": kv }),
        );
        let resolved = model.resolve(&req).unwrap();
        assert_eq!((resolved.width, resolved.height), (SIZE, SIZE));
        let references = model.references(&req, &resolved).unwrap();
        let cond = model.encode(&req, &resolved, &references).unwrap();
        let (ref_latents, shapes) = model.reference_latents(&references).unwrap();
        if let Some(full) = oracle("edit_full") {
            let rel = rel_rms(&values(&ref_latents), &values(&full["ref_latents"]));
            eprintln!("edit: reference latents rel-RMS {rel:.4e}");
            let rel = rel_rms(&values(&cond.positive.hidden), &values(&full["cond"]));
            eprintln!("edit: conditioner rel-RMS {rel:.4e}");
        }
        let latents = model
            .sample_latents(&resolved, &cond, Some((&ref_latents, &shapes)), SEED)
            .unwrap();
        let pixels = model.decode(&latents, &resolved, 4).unwrap();
        save_png(
            &format!("qwen_image_edit_{tag}.png").replace("_.png", ".png"),
            &pixels,
            4,
        );
        check_image("edit", tag, &latents, &pixels, 4, &EDIT_BOUNDS);
        let png = oracle_png_psnr("edit_full", tag, &latents, &pixels).unwrap();
        assert!(png >= PNG_PSNR, "edit {tag} PNG PSNR {png:.2} < {PNG_PSNR}");
        images.push(pixels);
    }
    let between = psnr(&images[0], &images[1]);
    eprintln!("edit: KV cache on vs off PNG PSNR {between:.2} dB");
    assert!(between >= PNG_PSNR, "KV cache on vs off PSNR {between:.2}");
    // The first step on upstream's own inputs.
    let full = oracle("edit_full").unwrap();
    let mask: Vec<bool> = values(&full["pad_mask"])
        .iter()
        .map(|v| *v != 0.0)
        .collect();
    let slot = model.denoiser.borrow();
    let transformer = &slot.as_ref().unwrap().transformer;
    let dt = transformer.dtype();
    let step0 = transformer
        .forward(
            &mlxcel_core::astype(&full["step0_input"], dt),
            &mlxcel_core::astype(&full["cond"], dt),
            &full["step0_timestep"],
            (1, 64, 64),
            Some(&transformer::EditLayout {
                reference_shapes: &[(1, 64, 64)],
                image_pad_mask: &mask,
            }),
            None,
        )
        .unwrap();
    let rel = rel_rms(&values(&step0), &values(&full["step0_out"]));
    eprintln!("edit: first-step velocity on upstream's inputs rel-RMS {rel:.4e}");
}
