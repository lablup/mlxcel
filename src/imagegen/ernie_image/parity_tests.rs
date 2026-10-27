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

//! Real-checkpoint parity against upstream (mlx-vlm 0.7.8 on mlx 0.32.4 CUDA).
//!
//! `tests/fixtures/imagegen/ernie_image_parity.safetensors` (written by
//! `tests/fixtures/generate_ernie_image_fixtures.py`) holds, per case, the
//! initial-noise checksum and head, the 8x8-pooled final latents and the
//! 8x8-pooled image. With `MLXCEL_ERNIE_IMAGE_ORACLE_DIR` pointing at the
//! generator's `--out`, the full-resolution final latents and PNG are checked
//! too, plus two stage diagnostics (the conditioning, and one transformer
//! forward on upstream's own inputs). `MLXCEL_ERNIE_IMAGE_OUT` saves the PNGs.
//!
//! Gates: bit-identical initial noise, per-case final-latent and PNG PSNR
//! bounds (see [`Bounds`]), and, with the oracle directory, the first
//! transformer forward on upstream's own inputs. The final-latent relative
//! RMS is reported as drift.
//!
//! Every forward at batch 1 reproduces upstream bit for bit, so the Turbo
//! text-to-image case matches exactly. The two classifier-free-guidance cases
//! run the transformer at batch 2, where one small GEMM (the text projection,
//! `[2, 13, 3072] x [3072, 4096]`) rounds differently in mlxcel's pinned MLX
//! and the mlx 0.32.4 wheel; small-M bf16 GEMMs pick different kernels in the
//! two builds, and 36 layers over many steps amplify a last-bit difference.
//! Upstream reproduces the same spread against itself: evaluating only that
//! projection one batch row at a time moves upstream's own edit case by
//! 1.39e-1 (32.1 dB) and its base case by 2.89e-1 (23.7 dB), against this
//! port's 1.35e-1 (32.2 dB) and 3.03e-1 (24.2 dB).
//!
//! The epic's 30 dB PNG criterion is NOT met by the 50-step base case
//! (24.2 dB). The control `base_8step_1024` runs the same checkpoint, prompt,
//! seed, size and CFG batch-2 forward over 8 steps and reaches 31.43 dB
//! (final latents 1.64e-1), so the batched forward is sound and the base
//! miss is the 50-step loop amplifying the GEMM rounding difference. The two
//! 50-step images are compositionally identical (same panda, teacup, teapot
//! and moon; the tail differs in detail).

use std::path::{Path, PathBuf};

use mlxcel_core::weights::{WeightMap, load_safetensors};

use super::*;
use crate::imagegen::output::encode_png;
use crate::imagegen::testing::{fnv1a32_bytes, pool_planes, psnr, rel_rms, values};

const PROMPT: &str = "A red panda serving tea in a moonlit bamboo forest";
/// Pooling factor of the committed summary.
const POOL: usize = 8;

/// Per-case bounds.
#[derive(Clone, Copy)]
struct Bounds {
    /// Final latents (and the 8x8-pooled latents), relative RMS.
    latents: f64,
    /// PNG PSNR floor, full resolution and pooled.
    psnr_db: f64,
}

/// Turbo text-to-image at batch 1: measured bit-exact (0, 57.0 dB). The
/// bound leaves room for a last-bit kernel change after an MLX bump. A wrong
/// port lands far above it: before the GELU and the biased linears rounded
/// like upstream, this case drifted to 3.3e-1 (24.8 dB).
const TURBO: Bounds = Bounds {
    latents: 1.0e-2,
    psnr_db: 30.0,
};
/// Turbo SDEdit with CFG at batch 2: measured 1.35e-1 (32.2 dB); upstream
/// against itself 1.39e-1 (32.1 dB).
const EDIT: Bounds = Bounds {
    latents: 2.5e-1,
    psnr_db: 30.0,
};
/// Base, 50 CFG steps at batch 2: measured 3.03e-1 (24.2 dB); upstream
/// against itself 2.89e-1 (23.7 dB). This case does NOT meet the epic's 30 dB
/// PNG criterion, and the 20 dB floor does not claim to: it only rejects a
/// grossly broken pipeline. It is not a wrong-port detector on its own (the
/// wrong Turbo port above would pass it); [`MAX_FIRST_FORWARD`] and the
/// [`BASE_8_STEP`] control at the 30 dB criterion are.
const BASE: Bounds = Bounds {
    latents: 4.5e-1,
    psnr_db: 20.0,
};
/// Control for [`BASE`]: the same CFG batch-2 forward over 8 steps, held to
/// the epic's 30 dB criterion. Measured 1.64e-1 (31.43 dB).
const BASE_8_STEP: Bounds = Bounds {
    latents: 2.5e-1,
    psnr_db: 30.0,
};
/// One transformer forward on upstream's own inputs (start latents, timestep,
/// padded text), unamplified by the sampling loop. Measured 0 (Turbo),
/// 4.9e-2 (edit, batch 2), 5.0e-3 (base, batch 2); the wrong port above gave
/// 2.0e-1.
const MAX_FIRST_FORWARD: f64 = 1.0e-1;

fn checkpoint(var: &str, name: &str) -> Option<PathBuf> {
    let path = std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/mlx")
            .join(name)
    });
    if path.join("transformer").exists() {
        Some(path)
    } else {
        eprintln!("skipping: {} is absent", path.display());
        None
    }
}

fn committed() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/ernie_image_parity.safetensors"),
    )
    .expect("parity fixture loads")
}

fn oracle_dir() -> Option<PathBuf> {
    std::env::var_os("MLXCEL_ERNIE_IMAGE_ORACLE_DIR").map(PathBuf::from)
}

fn save_png(case: &str, image: &GeneratedImage) {
    if let Some(dir) = std::env::var_os("MLXCEL_ERNIE_IMAGE_OUT") {
        let path = Path::new(&dir).join(format!("{case}.png"));
        std::fs::write(&path, encode_png(image).unwrap()).unwrap();
        eprintln!("{case}: wrote {}", path.display());
    }
}

/// 8x8 average pool of an interleaved RGB image, rounded like numpy.
fn pool8_rgb(pixels: &[u8], h: usize, w: usize) -> Vec<u8> {
    let (ph, pw) = (h / POOL, w / POOL);
    let mut out = Vec::with_capacity(ph * pw * 3);
    for y in 0..ph {
        for x in 0..pw {
            for c in 0..3 {
                let mut s = 0.0_f64;
                for dy in 0..POOL {
                    for dx in 0..POOL {
                        s += f64::from(pixels[((POOL * y + dy) * w + POOL * x + dx) * 3 + c]);
                    }
                }
                out.push((s / (POOL * POOL) as f64).round_ties_even() as u8);
            }
        }
    }
    out
}

fn u32_scalar(a: &MlxArray) -> u32 {
    let bytes = mlxcel_core::array_to_raw_bytes(a);
    u32::from_le_bytes(bytes[..4].try_into().unwrap())
}

fn check_case(model: &ErnieImageModel, case: &str, req: &ImageRequest, bounds: Bounds) {
    let fixture = committed();
    let options = check_request(false, req).unwrap();
    let steps = req.steps.unwrap();
    let guidance = req.guidance.unwrap();
    let text = model.text_batch(&req.prompt, "", guidance).unwrap();
    let full =
        oracle_dir().map(|dir| load_safetensors(dir.join(format!("{case}.safetensors"))).unwrap());

    if let Some(full) = &full {
        // Stage diagnostics: the conditioning, then one transformer forward on
        // upstream's own inputs.
        let last = usize::from(guidance > 1.0);
        let ours = model.text_encoder().encode(&req.prompt).unwrap();
        let cond_err = rel_rms(&values(&ours), &values(&full[&format!("text_{last}")]));
        let start = values(&full["start_index"])[0] as usize;
        let scheduler = ErnieScheduler::new(steps).unwrap();
        let init = &full["init_latents"];
        let dt = mlxcel_core::array_dtype(init);
        let batch = if guidance > 1.0 {
            mlxcel_core::concatenate(init, init, 0)
        } else {
            mlxcel_core::astype(init, dt)
        };
        let b = mlxcel_core::array_shape(&batch)[0];
        let t = mlxcel_core::astype(
            &mlxcel_core::full_f32(
                &[b],
                scheduler.timesteps()[start],
                mlxcel_core::dtype::FLOAT32,
            ),
            dt,
        );
        let lengths: Vec<i32> = values(&full["text_lengths"])
            .iter()
            .map(|v| *v as i32)
            .collect();
        let pred = model
            .transformer()
            .forward(&batch, &t, &full["text_padded"], &lengths)
            .unwrap();
        let pred_err = rel_rms(&values(&pred), &values(&full["step0_prediction"]));
        eprintln!(
            "{case}: conditioning rel-RMS {cond_err:.3e}, first transformer forward (upstream inputs) rel-RMS {pred_err:.3e}"
        );
        assert!(cond_err < 1e-2, "{case}: conditioning {cond_err}");
        assert!(
            pred_err < MAX_FIRST_FORWARD,
            "{case}: first transformer forward {pred_err}"
        );
    }

    let t0 = std::time::Instant::now();
    let latents = model
        .sample_latents(req, req.seed, steps, &text, &options)
        .unwrap();
    let pixels = pipeline::decode_to_pixels(model.vae(), &latents).unwrap();
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

    // Bit-identical initial noise (the img2img noise has the same shape, key
    // and dtype as the text-to-image start). A case missing from the committed
    // summary (the oracle-only control) checks against the full oracle file.
    let shape = mlxcel_core::array_shape(&latents);
    let noise = pipeline::initial_noise(req.seed, shape[1], req.width, req.height);
    let noise_bytes = mlxcel_core::array_to_raw_bytes(&noise);
    let final_latents = values(&latents);
    let summarized = fixture.contains_key(&format!("{case}.start_fnv"));
    if !summarized {
        let full = full
            .as_ref()
            .expect("a case outside the committed summary needs the oracle directory");
        assert_eq!(
            noise_bytes,
            mlxcel_core::array_to_raw_bytes(&full["noise"]),
            "{case}: initial noise"
        );
    }
    let pooled_check = summarized.then(|| {
        assert_eq!(
            fnv1a32_bytes(&noise_bytes),
            u32_scalar(&fixture[&format!("{case}.start_fnv")]),
            "{case}: initial noise checksum"
        );
        let head = values(&fixture[&format!("{case}.start_head")]);
        assert_eq!(
            &values(&noise)[..head.len()],
            head.as_slice(),
            "{case}: noise head"
        );
        let (h, w) = (shape[2] as usize, shape[3] as usize);
        let pooled = pool_planes(&final_latents, shape[1] as usize, h, w, POOL);
        let pooled_err = rel_rms(&pooled, &values(&fixture[&format!("{case}.latents_pool8")]));
        let pooled_image = pool8_rgb(&image.pixels, req.height as usize, req.width as usize);
        let want_image =
            mlxcel_core::array_to_raw_bytes(&fixture[&format!("{case}.image_pool8")]);
        let pooled_psnr = psnr(&pooled_image, &want_image);
        eprintln!(
            "{case}: {elapsed:.1}s, pooled latents rel-RMS {pooled_err:.3e}, pooled image PSNR {pooled_psnr:.2} dB"
        );
        (pooled_err, pooled_psnr)
    });

    if let (Some(full), Some(dir)) = (&full, oracle_dir()) {
        let latent_err = rel_rms(&final_latents, &values(&full["final_latents"]));
        let png = image::open(dir.join(format!("{case}.png")))
            .unwrap()
            .to_rgb8();
        let image_psnr = psnr(&image.pixels, png.as_raw());
        eprintln!(
            "{case}: full-res final latents rel-RMS {latent_err:.3e} (drift), PNG PSNR {image_psnr:.2} dB"
        );
        assert!(
            latent_err < bounds.latents,
            "{case}: final latents {latent_err}"
        );
        assert!(image_psnr >= bounds.psnr_db, "{case}: PSNR {image_psnr}");
    }
    if let Some((pooled_err, pooled_psnr)) = pooled_check {
        assert!(
            pooled_err < bounds.latents,
            "{case}: pooled latents {pooled_err}"
        );
        assert!(
            pooled_psnr >= bounds.psnr_db,
            "{case}: pooled PSNR {pooled_psnr}"
        );
    }
}

fn red_panda(size: u32, steps: u32, guidance: f32) -> ImageRequest {
    let mut req = ImageRequest::generate(PROMPT, 42);
    req.width = size;
    req.height = size;
    req.steps = Some(steps);
    req.guidance = Some(guidance);
    req.extra
        .insert("use_prompt_enhancer".into(), serde_json::json!(false));
    req
}

#[test]
#[ignore = "needs models/mlx/ernie-image-turbo and a GPU"]
fn turbo_generation_and_edit_match_upstream() {
    let Some(root) = checkpoint("MLXCEL_ERNIE_IMAGE_TURBO", "ernie-image-turbo") else {
        return;
    };
    let model = ErnieImageModel::load(&root).unwrap();
    assert_eq!(model.variant(), ErnieVariant::Turbo);
    check_case(&model, "turbo_1024", &red_panda(1024, 8, 1.0), TURBO);

    let Some(dir) = oracle_dir() else {
        eprintln!("skipping the SDEdit case: MLXCEL_ERNIE_IMAGE_ORACLE_DIR is unset");
        return;
    };
    let mut req = red_panda(1024, 8, 3.0);
    req.task = ImageTask::Edit;
    req.reference_images = vec![image::open(dir.join("turbo_1024.png")).unwrap()];
    req.extra
        .insert("image_strength".into(), serde_json::json!(0.7));
    check_case(&model, "turbo_edit_1024", &req, EDIT);
}

#[test]
#[ignore = "needs models/mlx/ernie-image and a GPU"]
fn base_generation_with_cfg_matches_upstream() {
    let Some(root) = checkpoint("MLXCEL_ERNIE_IMAGE_BASE", "ernie-image") else {
        return;
    };
    let model = ErnieImageModel::load(&root).unwrap();
    assert_eq!(model.variant(), ErnieVariant::Base);
    check_case(&model, "base_1024", &red_panda(1024, 50, 4.0), BASE);
}

/// Base with CFG at batch 2 like `base_1024`, but 8 steps: separates a wrong
/// batched forward from step-count amplification (see [`BASE_8_STEP`]). It
/// is not in the committed summary, so it needs the oracle directory.
#[test]
#[ignore = "needs models/mlx/ernie-image, MLXCEL_ERNIE_IMAGE_ORACLE_DIR and a GPU"]
fn base_cfg_eight_step_control_matches_upstream() {
    let Some(root) = checkpoint("MLXCEL_ERNIE_IMAGE_BASE", "ernie-image") else {
        return;
    };
    if oracle_dir().is_none() {
        eprintln!("skipping: MLXCEL_ERNIE_IMAGE_ORACLE_DIR is unset");
        return;
    }
    let model = ErnieImageModel::load(&root).unwrap();
    check_case(
        &model,
        "base_8step_1024",
        &red_panda(1024, 8, 4.0),
        BASE_8_STEP,
    );
}
