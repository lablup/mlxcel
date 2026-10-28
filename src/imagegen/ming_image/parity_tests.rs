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
//! `tests/fixtures/imagegen/ming_image_parity.safetensors` is the committed
//! summary (`generate_ming_image_fixtures.py real`): per case the templated
//! input ids, the initial-noise checksum and head, the 8x8-cell average pools
//! of the final latents and the 32x32-pixel average pools of the RGBA image.
//! The full tensors (conditioning, the first step's transformer inputs and
//! output, the final latents, the pixels) of the same run stay outside the
//! repository; the ignored tests read them from `MLXCEL_MING_IMAGE_ORACLE_DIR`
//! when it is set and write this port's PNGs to `MLXCEL_MING_IMAGE_OUT`.
//!
//! Cases (1024x1024, 12 steps, seed 0): `gen` (the poster prompt) and `rgba`
//! (a transparent-background prompt) on the bf16 checkpoint, gated; `q4` (the
//! poster on the 4-bit checkpoint against upstream on that checkpoint),
//! recorded only. Gates per gated case:
//! * the initial noise is bit-identical to upstream's;
//! * the pooled final latents stay within the case's bound ([`GEN_POOL`],
//!   [`RGBA_POOL`]), which deliberately broken ports exceed (numbers there);
//! * the PNG is at least [`PNG_PSNR`] dB from upstream's (all four channels;
//!   the alpha channel alone is reported).
//!
//! Per stage, on upstream's own inputs: the conditioner on the same ids, the
//! first step's velocity, and the VAE decode of upstream's final latents. The
//! full-resolution final-latent rel-RMS is printed as drift only.

use std::path::{Path, PathBuf};

use mlxcel_core::MlxArray;
use mlxcel_core::weights::{WeightMap, load_safetensors};

use super::*;
use crate::imagegen::testing::{fnv1a32_bytes, pool_planes, psnr, rel_rms, values};

pub(crate) const GEN_PROMPT: &str = "a minimalist poster, bold word HELLO, blue background";
pub(crate) const RGBA_PROMPT: &str = "RGBA, 4-channel, transparent background, a red apple icon";
const SEED: u64 = 0;
const STEPS: u32 = 12;
const SIZE: u32 = 1024;

/// Least PSNR between this port's PNG and upstream's.
const PNG_PSNR: f64 = 30.0;
/// Pooled final-latent rel-RMS bound of `gen`.
const GEN_POOL: f64 = 5e-2;
/// Pooled final-latent rel-RMS bound of `rgba`.
const RGBA_POOL: f64 = 5e-2;

/// `models/mlx/<name>` when its tokenizer is present.
pub(crate) fn model_dir(name: &str) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/mlx")
        .join(name);
    root.join("mllm/tokenizer.json").is_file().then_some(root)
}

pub(crate) fn parity() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/ming_image_parity.safetensors"),
    )
    .expect("fixture loads")
}

/// `<MLXCEL_MING_IMAGE_ORACLE_DIR>/<case>_full.safetensors`, when set.
fn oracle(case: &str) -> Option<WeightMap> {
    let path = PathBuf::from(std::env::var_os("MLXCEL_MING_IMAGE_ORACLE_DIR")?)
        .join(format!("{case}_full.safetensors"));
    Some(load_safetensors(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

fn save_png(name: &str, pixels: &[u8]) {
    let Some(dir) = std::env::var_os("MLXCEL_MING_IMAGE_OUT") else {
        return;
    };
    let path = Path::new(&dir).join(name);
    image::save_buffer(&path, pixels, SIZE, SIZE, image::ExtendedColorType::Rgba8)
        .expect("write PNG");
}

fn noise_fnv(seed: u64) -> (u32, Vec<f32>) {
    let noise = mlxcel_core::contiguous(&pipeline::initial_noise(seed, 16, SIZE, SIZE), false);
    mlxcel_core::eval(&noise);
    let bytes = mlxcel_core::try_array_to_raw_bytes(&noise).unwrap();
    (fnv1a32_bytes(&bytes), values(&noise)[..16].to_vec())
}

#[test]
fn initial_noise_is_bit_identical_to_upstream() {
    let want = parity();
    let (fnv, head) = noise_fnv(SEED);
    for case in ["gen", "rgba"] {
        let raw = mlxcel_core::try_array_to_raw_bytes(&want[&format!("{case}.noise_fnv")]).unwrap();
        assert_eq!(
            fnv,
            u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
            "{case}"
        );
        assert_eq!(head, values(&want[&format!("{case}.noise_head")]), "{case}");
    }
    assert_ne!(noise_fnv(SEED + 1).0, fnv);
}

/// `[1, C, 1, h, w]` latents to `[C, h/8, w/8]` cell means.
fn latents_pool(latents: &MlxArray) -> Vec<f32> {
    let s = mlxcel_core::array_shape(latents);
    let (c, h, w) = (s[1] as usize, s[3] as usize, s[4] as usize);
    pool_planes(&values(latents), c, h, w, 8)
}

/// `[H, W, 4]` bytes to `[4, H/32, W/32]` cell means.
fn image_pool(pixels: &[u8]) -> Vec<f32> {
    let n = SIZE as usize;
    let mut planes = vec![0.0f32; 4 * n * n];
    for (i, &p) in pixels.iter().enumerate() {
        planes[(i % 4) * n * n + i / 4] = f32::from(p);
    }
    pool_planes(&planes, 4, n, n, 32)
}

fn alpha(pixels: &[u8]) -> Vec<u8> {
    pixels.iter().skip(3).step_by(4).copied().collect()
}

fn request(prompt: &str) -> ImageRequest {
    let mut req = ImageRequest::generate(prompt, SEED);
    req.width = SIZE;
    req.height = SIZE;
    req.steps = Some(STEPS);
    req
}

fn ids(case: &str) -> Vec<i32> {
    values(&parity()[&format!("{case}.input_ids")])
        .into_iter()
        .map(|v| v as i32)
        .collect()
}

/// With `<oracle>/<case>_layers.safetensors` (the generator's `layers`
/// mode), every backbone state's rel-RMS on the query span and on the text
/// positions, with and without the image router.
fn layer_diagnosis(
    checkpoint: &str,
    case: &str,
    encoder: &text_encoder::TextEncoder,
    ids: &[i32],
) -> anyhow::Result<()> {
    let Some(dir) = std::env::var_os("MLXCEL_MING_IMAGE_ORACLE_DIR") else {
        return Ok(());
    };
    let path = Path::new(&dir).join(format!("{case}_layers.safetensors"));
    if !path.is_file() {
        return Ok(());
    }
    let want = load_safetensors(&path).map_err(anyhow::Error::msg)?;
    let start = ids.iter().position(|&id| id == 157_157).unwrap() as i32;
    let len = ids.len() as i32;
    let span = |a: &MlxArray, from: i32, to: i32| {
        let h = mlxcel_core::array_shape(a)[2];
        values(&mlxcel_core::slice(a, &[0, from, 0], &[1, to, h]))
    };
    // One layer at a time on upstream's own input state.
    let layers = want.keys().filter(|k| k.starts_with("state.")).count() - 1;
    for i in 0..layers.saturating_sub(1) {
        let out = encoder.backbone_layer(i, &want[&format!("state.{i}")], ids)?;
        let theirs = &want[&format!("state.{}", i + 1)];
        let q = rel_rms(
            &span(&out, start, start + 256),
            &span(theirs, start, start + 256),
        );
        let t = rel_rms(&span(&out, 0, start - 1), &span(theirs, 0, start - 1));
        eprintln!(
            "{checkpoint} {case}: layer {i:2} alone on upstream's input rel-RMS query {q:.3e} text {t:.3e}"
        );
    }
    for routed in [true, false] {
        let states = encoder.backbone_states(ids, routed)?;
        for (i, ours) in states.iter().enumerate() {
            let theirs = &want[&format!("state.{i}")];
            let q = rel_rms(
                &span(ours, start, start + 256),
                &span(theirs, start, start + 256),
            );
            let t = rel_rms(&span(ours, 0, start - 1), &span(theirs, 0, start - 1));
            let tail = rel_rms(
                &span(ours, start + 256, len),
                &span(theirs, start + 256, len),
            );
            eprintln!(
                "{checkpoint} {case} image_routing={routed}: state {i:2} rel-RMS query {q:.3e} text {t:.3e} end {tail:.3e}"
            );
        }
    }
    Ok(())
}

/// The per-case results a parity run reports.
struct CaseResult {
    pool: f64,
    png: Option<f64>,
}

/// Run `cases` of `checkpoint` end to end and per stage; returns each case's
/// pooled-latent rel-RMS and, with the oracle, its PNG PSNR.
fn run_cases(checkpoint: &str, cases: &[(&str, &str)]) -> Vec<CaseResult> {
    let root =
        model_dir(checkpoint).unwrap_or_else(|| panic!("models/mlx/{checkpoint} is present"));
    let model = MingImageModel::load(&root).unwrap();
    let want = parity();

    // Conditioner: our ids equal upstream's, then the streams on them.
    let mut conds = Vec::new();
    model
        .with_text_encoder(|encoder| {
            for (case, prompt) in cases {
                let ours = encoder.formatter().unwrap().input_ids(prompt)?;
                assert_eq!(ours, ids(case), "{case}: templated ids");
                layer_diagnosis(checkpoint, case, encoder, &ours)?;
                let states = encoder.encode_ids(&ours)?;
                if let Some(full) = oracle(case) {
                    for (name, got) in [
                        ("query_hidden", &states.query_hidden),
                        ("text_states", &states.text_states),
                        ("cap_feats", &states.encoded.cap_feats),
                        ("cap_feats_2", &states.encoded.cap_feats_2),
                    ] {
                        let rel = rel_rms(&values(got), &values(&full[&format!("cond.{name}")]));
                        eprintln!("{checkpoint} {case}: conditioner {name} rel-RMS {rel:.4e}");
                    }
                }
                conds.push(states.encoded);
            }
            Ok(())
        })
        .unwrap();
    model.evict_text_encoder();

    let resolved = resolve_request(&request(GEN_PROMPT)).unwrap();
    let mut results = Vec::new();
    for ((case, _), cond) in cases.iter().zip(&conds) {
        if let Some(full) = oracle(case) {
            model
                .with_denoiser(|d| {
                    // One step on upstream's inputs (its own conditioning).
                    let up = text_encoder::Encoded {
                        cap_feats: mlxcel_core::copy(&full["cond.cap_feats"]),
                        cap_feats_2: mlxcel_core::copy(&full["cond.cap_feats_2"]),
                    };
                    let out = d.transformer.forward(
                        &full["step0_x"],
                        &full["step0_t"],
                        &up.cap_feats,
                        Some(&up.cap_feats_2),
                    )?;
                    let rel = rel_rms(&values(&out), &values(&full["step0_out"]));
                    eprintln!("{checkpoint} {case}: first-step velocity on upstream's inputs rel-RMS {rel:.4e}");
                    let theirs = mlxcel_core::try_array_to_raw_bytes(&full["pixels"]).unwrap();
                    let decoded = pipeline::decode_to_pixels(&d.vae, &full["final_latents"])?;
                    let p = psnr(&decoded, &theirs);
                    eprintln!("{checkpoint} {case}: VAE decode of upstream's final latents PSNR {p:.2} dB");
                    Ok(())
                })
                .unwrap();
        }
        let latents = model.sample_latents(&resolved, cond, SEED).unwrap();
        let pixels = model.decode(&latents).unwrap();
        save_png(&format!("ming_image_{checkpoint}_{case}.png"), &pixels);
        let pool = rel_rms(
            &latents_pool(&latents),
            &values(&want[&format!("{case}.latents_pool")]),
        );
        let to_u8 = |v: Vec<f32>| v.iter().map(|x| x.round() as u8).collect::<Vec<_>>();
        let pool_psnr = psnr(
            &to_u8(image_pool(&pixels)),
            &to_u8(values(&want[&format!("{case}.image_pool")])),
        );
        let transparent =
            alpha(&pixels).iter().filter(|&&a| a < 16).count() as f64 / f64::from(SIZE * SIZE);
        eprintln!(
            "{checkpoint} {case}: latents pool rel-RMS {pool:.4e}, image pool PSNR {pool_psnr:.2} dB, alpha<16 fraction {transparent:.3}"
        );
        let png = oracle(case).map(|full| {
            let drift = rel_rms(&values(&latents), &values(&full["final_latents"]));
            let theirs = mlxcel_core::try_array_to_raw_bytes(&full["pixels"]).unwrap();
            let png = psnr(&pixels, &theirs);
            let alpha_psnr = psnr(&alpha(&pixels), &alpha(&theirs));
            eprintln!(
                "{checkpoint} {case}: final latents rel-RMS {drift:.4e} (drift), PNG PSNR {png:.2} dB, alpha PSNR {alpha_psnr:.2} dB"
            );
            png
        });
        results.push(CaseResult { pool, png });
    }
    results
}

#[test]
#[ignore = "runs the real bf16 Ming-Image checkpoint (GPU, ~55 GB peak)"]
fn bf16_matches_upstream() {
    let results = run_cases(
        "ming-image-0.1-design",
        &[("gen", GEN_PROMPT), ("rgba", RGBA_PROMPT)],
    );
    for (r, (case, bound)) in results.iter().zip([("gen", GEN_POOL), ("rgba", RGBA_POOL)]) {
        assert!(
            r.pool <= bound,
            "{case}: latents pool rel-RMS {:.4e} > {bound}",
            r.pool
        );
        if let Some(png) = r.png {
            assert!(png >= PNG_PSNR, "{case}: PNG PSNR {png:.2} < {PNG_PSNR}");
        }
    }
}

#[test]
#[ignore = "runs the real 4-bit Ming-Image checkpoint (GPU); recorded, not gated"]
fn q4_is_recorded_against_upstream_on_the_same_checkpoint() {
    let results = run_cases("ming-image-0.1-design-mlx-4bit", &[("q4", GEN_PROMPT)]);
    let r = &results[0];
    eprintln!(
        "q4: latents pool rel-RMS {:.4e}, PNG PSNR {:?}",
        r.pool, r.png
    );
}

#[test]
#[ignore = "runs the real bf16 Ming-Image conditioner (GPU); needs MLXCEL_MING_IMAGE_ORACLE_DIR with gen_layers"]
fn conditioner_layers_against_upstream() {
    let root = model_dir("ming-image-0.1-design").expect("models/mlx/ming-image-0.1-design");
    let model = MingImageModel::load(&root).unwrap();
    model
        .with_text_encoder(|encoder| {
            let ids = encoder.formatter().unwrap().input_ids(GEN_PROMPT)?;
            layer_diagnosis("ming-image-0.1-design", "gen", encoder, &ids)
        })
        .unwrap();
}
