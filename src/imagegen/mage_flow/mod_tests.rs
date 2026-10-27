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

//! Request resolution of [`MageFlowModel`] (no weights), the seeded initial
//! noise against upstream's, and (ignored, GPU and the real checkpoints) the
//! whole-run parity of Mage-Flow-Turbo and Mage-Flow-Edit-Turbo against the
//! upstream CUDA oracle (`generate_mage_flow_fixtures.py real`).

use std::path::{Path, PathBuf};

use image::DynamicImage;
use serde_json::json;

use super::conditioner::tests::{EDIT_PROMPT, PROMPT, edit_references, model_dir, oracle, parity};
use super::transformer::tests::{rel_rms, to_vec};
use super::*;

/// A checkpoint directory named `name` with the files [`MageFlowModel::open`]
/// reads and no weights.
fn fake_checkpoint(name: &str) -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join(name);
    for dir in ["transformer", "vae", "text_encoder"] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
    }
    std::fs::write(
        root.join("transformer/config.json"),
        r#"{"hidden_size": 3072, "num_heads": 24, "axes_dim": [16, 56, 56]}"#,
    )
    .unwrap();
    std::fs::write(root.join("text_encoder/config.json"), "{}").unwrap();
    std::fs::write(root.join("text_encoder/tokenizer.json"), "{}").unwrap();
    std::fs::write(
        root.join("model_index.json"),
        r#"{"_class_name": "MageFlowPipeline"}"#,
    )
    .unwrap();
    (tmp, root)
}

fn request(task: ImageTask) -> ImageRequest {
    let mut req = ImageRequest::generate("a teapot", 7);
    req.task = task;
    if task == ImageTask::Edit {
        req.reference_images = vec![DynamicImage::new_rgb8(1000, 750)];
    }
    req
}

fn invalid(err: anyhow::Error) -> String {
    let message = format!("{err:#}");
    assert!(
        err.downcast_ref::<InvalidImageRequest>().is_some(),
        "not an InvalidImageRequest: {message}"
    );
    message
}

#[test]
fn the_variant_fixes_task_and_defaults() {
    let (_tmp, root) = fake_checkpoint("Mage-Flow-Edit-Turbo");
    let model = MageFlowModel::open(&root).unwrap();
    assert_eq!(model.family(), "mage_flow");
    assert_eq!(model.variant(), Some("mage-flow-edit-turbo"));
    assert!(model.supports(ImageTask::Edit));
    assert!(!model.supports(ImageTask::Generate));
    // Defined for both tasks so the server can report them.
    assert_eq!(model.defaults(ImageTask::Generate).steps, 4);
    assert_eq!(model.defaults(ImageTask::Edit).guidance, 1.0);
    assert_eq!((model.default_edit_size())(1000, 750), (992, 736));

    let (_tmp, root) = fake_checkpoint("mage-flow-base");
    let model = MageFlowModel::open(&root).unwrap();
    assert!(model.supports(ImageTask::Generate));
    assert_eq!(model.defaults(ImageTask::Generate).steps, 30);
    assert_eq!(model.defaults(ImageTask::Generate).guidance, 5.0);
}

#[test]
fn a_wrong_task_is_an_invalid_request_naming_the_variant() {
    let (_tmp, root) = fake_checkpoint("mage-flow-edit-turbo");
    let model = MageFlowModel::open(&root).unwrap();
    let message = invalid(
        model
            .check_request(&request(ImageTask::Generate))
            .unwrap_err(),
    );
    assert!(
        message.contains("Mage-Flow-Edit-Turbo is an image-edit checkpoint"),
        "{message}"
    );
    // The shared validation names it too.
    let err = crate::imagegen::run::validate_request(&model, &request(ImageTask::Generate))
        .unwrap_err()
        .to_string();
    assert!(err.contains("(variant mage-flow-edit-turbo)"), "{err}");

    let (_tmp, root) = fake_checkpoint("mage-flow-turbo");
    let model = MageFlowModel::open(&root).unwrap();
    let message = invalid(model.check_request(&request(ImageTask::Edit)).unwrap_err());
    assert!(
        message.contains("Mage-Flow-Turbo is a text-to-image checkpoint"),
        "{message}"
    );
}

#[test]
fn sizes_and_options_are_checked_before_any_weights_load() {
    let (_tmp, root) = fake_checkpoint("mage-flow-turbo");
    let model = MageFlowModel::open(&root).unwrap();
    let mut req = request(ImageTask::Generate);
    assert!(model.check_request(&req).is_ok());
    req.width = 256;
    let message = invalid(model.check_request(&req).unwrap_err());
    assert!(
        message.contains("width must be in [512, 2048]"),
        "{message}"
    );
    let mut req = request(ImageTask::Generate);
    req.extra = json!({"max_size": 1024}).as_object().unwrap().clone();
    let message = invalid(model.check_request(&req).unwrap_err());
    assert!(message.contains("edits only"), "{message}");
}

#[test]
fn cfg_negatives_follow_upstream() {
    let (_tmp, root) = fake_checkpoint("mage-flow");
    let model = MageFlowModel::open(&root).unwrap();
    let req = request(ImageTask::Generate);
    // Guidance 5 and the default negative " ".
    assert_eq!(model.resolve(&req).unwrap().negative.as_deref(), Some(" "));
    let mut req = request(ImageTask::Generate);
    req.negative_prompt = Some(String::new());
    // An empty negative disables CFG for generation.
    assert_eq!(model.resolve(&req).unwrap().negative, None);
    let mut req = request(ImageTask::Generate);
    req.guidance = Some(1.0);
    req.negative_prompt = Some("blurry".into());
    assert_eq!(model.resolve(&req).unwrap().negative, None);
    let mut req = request(ImageTask::Generate);
    req.extra = json!({"negative_prompt": "blurry"})
        .as_object()
        .unwrap()
        .clone();
    assert_eq!(
        model.resolve(&req).unwrap().negative.as_deref(),
        Some("blurry")
    );

    let (_tmp, root) = fake_checkpoint("mage-flow-edit");
    let model = MageFlowModel::open(&root).unwrap();
    let mut req = request(ImageTask::Edit);
    req.width = 992;
    req.height = 736;
    req.negative_prompt = Some(String::new());
    // An edit falls back to " " instead.
    let resolved = model.resolve(&req).unwrap();
    assert_eq!(resolved.negative.as_deref(), Some(" "));
    assert_eq!((resolved.steps, resolved.guidance), (30, 5.0));
    // `max_size` sizes the output from the first reference.
    req.extra = json!({"max_size": 1024}).as_object().unwrap().clone();
    let resolved = model.resolve(&req).unwrap();
    assert_eq!((resolved.width, resolved.height), (1024, 768));
}

fn fnv1a32_bytes(data: &[u8]) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for b in data {
        h ^= u32::from(*b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

#[test]
fn initial_noise_is_bit_identical_to_upstream() {
    let noise = pipeline::initial_noise(32, 32, 42);
    assert_eq!(mlxcel_core::array_shape(&noise), [1, 1024, 128]);
    assert_eq!(
        mlxcel_core::array_dtype(&noise),
        mlxcel_core::dtype::BFLOAT16
    );
    let values = to_vec(&noise);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let want = parity();
    for case in ["turbo", "edit"] {
        let raw = mlxcel_core::try_array_to_raw_bytes(&want[&format!("{case}.start_fnv")]).unwrap();
        let fnv = u32::from_le_bytes(raw[..4].try_into().unwrap());
        assert_eq!(fnv1a32_bytes(&bytes), fnv, "{case}");
        assert_eq!(
            &values[..64],
            to_vec(&want[&format!("{case}.start_head")]),
            "{case}"
        );
    }
}

/// `pool4` of the generator: 4x4 means over the grid axes of `[h, w, c]`.
fn pool4(values: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; (h / 4) * (w / 4) * c];
    for y in 0..h {
        for x in 0..w {
            for k in 0..c {
                out[((y / 4) * (w / 4) + x / 4) * c + k] += values[(y * w + x) * c + k] / 16.0;
            }
        }
    }
    out
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()
}

fn rel_rms_vec(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den).sqrt() as f32
}

/// Per-case bounds, each set from the value measured on GB10 (in the comment
/// at the call site) with headroom for kernel accumulation order.
struct Bounds {
    /// rel-RMS of the 4x4-pooled final latents.
    latents_pool4: f32,
    /// PSNR of the 4x4-pooled image against upstream's, dB.
    image_pool4_psnr: f64,
    /// PSNR of the full PNG against upstream's (needs the oracle directory).
    png_psnr: f64,
}

/// PNG PSNR bound when the denoiser runs on upstream's own conditioning.
const ORACLE_COND_PNG_PSNR: f64 = 30.0;

fn oracle_png(case: &str) -> Option<Vec<f32>> {
    let dir = std::env::var_os("MLXCEL_MAGE_FLOW_ORACLE_DIR")?;
    let png = image::open(PathBuf::from(dir).join(format!("{case}.png")))
        .unwrap()
        .to_rgb8();
    Some(png.as_raw().iter().map(|&p| f32::from(p)).collect())
}

fn as_f32(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&p| f32::from(p)).collect()
}

/// Run one oracle case end to end and check it against the oracle.
///
/// With `MLXCEL_MAGE_FLOW_ORACLE_DIR` set it also reruns the denoiser and
/// VAE on upstream's own conditioner output, which isolates them from the
/// conditioner's bf16 drift (see `conditioner::tests::check_cond`), and
/// gates that run at [`ORACLE_COND_PNG_PSNR`].
fn run_case(case: &str, root: &Path, req: &ImageRequest, bounds: &Bounds) {
    let model = MageFlowModel::load(root).unwrap();
    let resolved = model.resolve(req).unwrap();
    let mut cond = model.encode(req, &resolved).unwrap();
    assert!(model.conditioner.borrow().is_none(), "conditioner evicted");
    let latents = model.sample_latents(req, &resolved, &cond, 0).unwrap();
    let pixels = model.decode(&latents, &resolved).unwrap();
    let want = parity();

    let latents = to_vec(&latents);
    let latents_pool = pool4(&latents, 32, 32, 128);
    let drift = rel_rms_vec(
        &latents_pool,
        &to_vec(&want[&format!("{case}.latents_pool4")]),
    );
    let image = as_f32(&pixels);
    let image_pool: Vec<f32> = pool4(&image, 512, 512, 3)
        .iter()
        .map(|v| v.round())
        .collect();
    let image_psnr = psnr(&image_pool, &to_vec(&want[&format!("{case}.image_pool4")]));
    eprintln!("{case}: latents_pool4 rel-RMS {drift:.4e}, image_pool4 PSNR {image_psnr:.2} dB");
    assert!(
        drift <= bounds.latents_pool4,
        "{case} latents_pool4 rel-RMS {drift} > {}",
        bounds.latents_pool4
    );
    assert!(
        image_psnr >= bounds.image_pool4_psnr,
        "{case} image_pool4 PSNR {image_psnr} < {}",
        bounds.image_pool4_psnr
    );
    let (Some(oracle), Some(reference)) = (oracle(case), oracle_png(case)) else {
        return;
    };
    let full = rel_rms_vec(&latents, &to_vec(&oracle["final_latents"]));
    let png_psnr = psnr(&image, &reference);
    eprintln!("{case}: final latents rel-RMS {full:.4e} (drift), PNG PSNR {png_psnr:.2} dB");
    assert!(
        png_psnr >= bounds.png_psnr,
        "{case} PNG PSNR {png_psnr} < {}",
        bounds.png_psnr
    );
    if let Some(refs) = oracle.get("reference_latents") {
        let pixels = pipeline::reference_pixels(&req.reference_images, 512, 512).unwrap();
        let err = rel_rms(&pixels, &oracle["reference_pixels"]);
        eprintln!("{case}: reference pixels rel-RMS {err:.4e}");
        assert!(err < 1e-2, "{case} reference pixels rel-RMS {err}");
        let slot = model.denoiser.borrow();
        let vae = &slot.as_ref().unwrap().vae;
        let tokens = pipeline::reference_tokens(vae, &pixels, true, 42).unwrap();
        let err = rel_rms(&tokens, &mlxcel_core::reshape(refs, &[1, 2048, 128]));
        eprintln!("{case}: reference latents rel-RMS {err:.4e} (drift)");
    }
    cond.positive = mlxcel_core::astype(&oracle["cond"], mlxcel_core::dtype::BFLOAT16);
    let latents = model.sample_latents(req, &resolved, &cond, 0).unwrap();
    let pixels = model.decode(&latents, &resolved).unwrap();
    let full = rel_rms_vec(&to_vec(&latents), &to_vec(&oracle["final_latents"]));
    let png_psnr = psnr(&as_f32(&pixels), &reference);
    eprintln!(
        "{case} on upstream's conditioning: final latents rel-RMS {full:.4e} (drift), \
         PNG PSNR {png_psnr:.2} dB"
    );
    assert!(
        png_psnr >= ORACLE_COND_PNG_PSNR,
        "{case} on upstream's conditioning: PNG PSNR {png_psnr} < {ORACLE_COND_PNG_PSNR}"
    );
}

#[test]
#[ignore = "runs the real Mage-Flow-Turbo checkpoint (GPU, ~17 GB)"]
fn turbo_run_matches_upstream() {
    let root = model_dir("mage-flow-turbo").expect("models/mlx/mage-flow-turbo");
    let mut req = ImageRequest::generate(PROMPT, 42);
    req.steps = Some(4);
    req.guidance = Some(1.0);
    run_case(
        "turbo",
        &root,
        &req,
        // Measured: latents_pool4 4.75e-2, image_pool4 39.0 dB, PNG 38.4 dB
        // (on upstream's conditioning: PNG 42.3 dB).
        &Bounds {
            latents_pool4: 8e-2,
            image_pool4_psnr: 32.0,
            png_psnr: 30.0,
        },
    );
}

#[test]
#[ignore = "runs the real Mage-Flow-Edit-Turbo checkpoint (GPU, ~18 GB)"]
fn edit_run_matches_upstream() {
    let root = model_dir("mage-flow-edit-turbo").expect("models/mlx/mage-flow-edit-turbo");
    let mut req = ImageRequest::generate(EDIT_PROMPT, 42);
    req.task = ImageTask::Edit;
    req.reference_images = edit_references(&root);
    req.steps = Some(4);
    req.guidance = Some(1.0);
    run_case(
        "edit",
        &root,
        &req,
        // Measured: latents_pool4 2.13e-1, image_pool4 22.1 dB, PNG 21.8 dB;
        // on upstream's conditioning PNG 39.9 dB. The gap is the conditioner:
        // its bf16 hidden states sit 2.7e-1 from upstream's (as far as
        // upstream's own bf16 run sits from f32, see
        // `conditioner::tests::check_cond`), and upstream itself lands at
        // 22.0 dB from its own PNG when only its conditioner runs in f32.
        &Bounds {
            latents_pool4: 3.5e-1,
            image_pool4_psnr: 18.0,
            png_psnr: 18.0,
        },
    );
}
