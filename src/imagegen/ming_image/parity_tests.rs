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
//! * the pooled final latents stay within the case's bound: [`RGBA_POOL`],
//!   which deliberately broken ports exceed, and for `gen` the loose sanity
//!   bound [`GEN_POOL`] (see below);
//! * `rgba`: the PNG is at least [`PNG_PSNR`] dB from upstream's, and its
//!   alpha channel alone at least [`ALPHA_PSNR`] dB;
//! * `gen` is a documented conditioner-precision exception: its end-to-end
//!   PNG (24.98 dB) is recorded, not gated. The 20-layer BailingMoeV2
//!   backbone amplifies rounding-level per-layer differences (1e-4 to 7.5e-4
//!   per layer on upstream's own inputs; the final norm alone is bit-exact)
//!   into a 0.2 query-span drift, and upstream perturbed against itself at the
//!   same drift lands at 16.37 dB on this prompt. Its gate is the run on
//!   upstream's conditioning ([`ORACLE_COND_PNG_PSNR`] and per-case pooled
//!   bounds that dropping `cap_feats_2` exceeds), the conditioner's own
//!   rel-RMS bounds on upstream's ids (which the wrong-router conditioner
//!   exceeds, checked in the test), and a loose end-to-end sanity bound.
//!
//! Per stage, on upstream's own inputs: the conditioner on the same ids, the
//! first step's velocity, and the VAE decode of upstream's final latents. The
//! full-resolution final-latent rel-RMS is printed as drift only.

use std::path::{Path, PathBuf};

use mlxcel_core::weights::{WeightMap, load_safetensors};
use mlxcel_core::{MlxArray, UniquePtr};

use super::*;
use crate::imagegen::testing::{
    assert_noise_head, fnv1a32_bytes, noise_checksum_is_exact, pool_planes, psnr, rel_rms, values,
};

/// The f32 draw's head is gated to this many ulps. On M1 Ultra the Metal GPU
/// misses upstream's CUDA draw by one ulp at element 0, and its whole-draw
/// checksum is 1107051371 against CUDA's 909872115 (the CPU stream gives a
/// third value), so the checksum is checked on CUDA only. See
/// [`noise_checksum_is_exact`].
const MAX_NOISE_ULPS: u32 = 2;

pub(crate) const GEN_PROMPT: &str = "a minimalist poster, bold word HELLO, blue background";
pub(crate) const RGBA_PROMPT: &str = "RGBA, 4-channel, transparent background, a red apple icon";
const SEED: u64 = 0;
const STEPS: u32 = 12;
const SIZE: u32 = 1024;

/// Least PSNR between this port's PNG and upstream's.
const PNG_PSNR: f64 = 30.0;
/// Least alpha-channel PSNR of the gated RGBA case against upstream's.
const ALPHA_PSNR: f64 = 30.0;
/// Conditioner bounds on upstream's ids (bf16): `cap_feats` measured 2.05e-1
/// (gen and rgba); `cap_feats_2` 5.5e-2 / 7.0e-2; the text states 1.2e-2 /
/// 6.2e-3. The query tokens on the text router exceed the `cap_feats` bound
/// (measured 4.59e-1).
const COND_CAP_FEATS: f64 = 0.3;
const COND_CAP_FEATS_2: f64 = 0.12;
const COND_TEXT: f64 = 3e-2;
/// Pooled final-latent rel-RMS bound of `gen` end to end, a sanity bound
/// only (conditioner-precision exception, see the module docs): measured
/// 3.37e-1. The end-to-end pools cannot tell a broken conditioner apart
/// here (the query tokens on the text router land at 1.07e-1), so the
/// conditioner bounds above carry that check; dropping `cap_feats_2` lands
/// at 7.7e-1 on upstream's conditioning.
const GEN_POOL: f64 = 0.6;
/// Pooled final-latent rel-RMS bound of `rgba`: measured 3.03e-2.
const RGBA_POOL: f64 = 6e-2;

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
fn initial_noise_matches_upstream() {
    let want = parity();
    let (fnv, head) = noise_fnv(SEED);
    for case in ["gen", "rgba"] {
        let raw = mlxcel_core::try_array_to_raw_bytes(&want[&format!("{case}.noise_fnv")]).unwrap();
        if noise_checksum_is_exact() {
            assert_eq!(
                fnv,
                u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                "{case}"
            );
        }
        assert_noise_head(
            &head,
            &values(&want[&format!("{case}.noise_head")]),
            0,
            MAX_NOISE_ULPS,
            case,
        );
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
    // One layer at a time on upstream's own input state; the last layer is
    // compared with upstream's pre-norm output, and the final norm alone on
    // that pre-norm output with upstream's normed state.
    let layers = want.keys().filter(|k| k.starts_with("state.")).count() - 1;
    for i in 0..layers {
        let out = encoder.backbone_layer(i, &want[&format!("state.{i}")], ids)?;
        let key = if i + 1 == layers {
            "prenorm".to_string()
        } else {
            format!("state.{}", i + 1)
        };
        let Some(theirs) = want.get(&key) else {
            continue;
        };
        let q = rel_rms(
            &span(&out, start, start + 256),
            &span(theirs, start, start + 256),
        );
        let t = rel_rms(&span(&out, 0, start - 1), &span(theirs, 0, start - 1));
        eprintln!(
            "{checkpoint} {case}: layer {i:2} alone on upstream's input rel-RMS query {q:.3e} text {t:.3e}"
        );
    }
    if let Some(prenorm) = want.get("prenorm") {
        let normed = encoder.backbone_norm(prenorm);
        let theirs = &want[&format!("state.{layers}")];
        let q = rel_rms(
            &span(&normed, start, start + 256),
            &span(theirs, start, start + 256),
        );
        let t = rel_rms(&span(&normed, 0, start - 1), &span(theirs, 0, start - 1));
        eprintln!(
            "{checkpoint} {case}: final norm alone on upstream's input rel-RMS query {q:.3e} text {t:.3e}"
        );
        let ours = encoder.backbone_states(ids, true)?;
        let ours_pre = encoder.backbone_layer(layers - 1, &ours[layers - 1], ids)?;
        let pq = rel_rms(
            &span(&ours_pre, start, start + 256),
            &span(prenorm, start, start + 256),
        );
        let pt = rel_rms(&span(&ours_pre, 0, start - 1), &span(prenorm, 0, start - 1));
        let nq = rel_rms(
            &span(&encoder.backbone_norm(&ours_pre), start, start + 256),
            &span(theirs, start, start + 256),
        );
        eprintln!(
            "{checkpoint} {case}: full-stack pre-norm drift rel-RMS query {pq:.3e} text {pt:.3e}; after the norm query {nq:.3e}"
        );
        // Where the post-norm query error sits: share of the squared error in
        // the 16 channels of largest norm weight magnitude.
        let w = values(&encoder.backbone_norm(&mlxcel_core::full_f32(
            &[1, 1, 2048],
            1.0,
            mlxcel_core::dtype::FLOAT32,
        )));
        let mut top: Vec<usize> = (0..w.len()).collect();
        top.sort_by(|a, b| w[*b].abs().total_cmp(&w[*a].abs()));
        let (o, u) = (
            span(&encoder.backbone_norm(&ours_pre), start, start + 256),
            span(theirs, start, start + 256),
        );
        let mut tot = 0.0f64;
        let mut big = 0.0f64;
        let set: std::collections::HashSet<usize> = top[..16].iter().copied().collect();
        for (k, (x, y)) in o.iter().zip(&u).enumerate() {
            let e = f64::from(x - y).powi(2);
            tot += e;
            if set.contains(&(k % 2048)) {
                big += e;
            }
        }
        eprintln!(
            "{checkpoint} {case}: post-norm query squared error in the 16 largest-|gamma| channels {:.1}% (norm weight max |w| {:.2}, median {:.3})",
            100.0 * big / tot.max(1e-30),
            w[top[0]].abs(),
            {
                let mut a: Vec<f32> = w.iter().map(|x| x.abs()).collect();
                a.sort_by(f32::total_cmp);
                a[a.len() / 2]
            }
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
    /// rel-RMS of `cap_feats`, `cap_feats_2` and the text states against
    /// upstream's on the same ids (with the oracle directory).
    cond: Option<[f64; 3]>,
    pool: f64,
    png: Option<f64>,
    alpha_psnr: Option<f64>,
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
    let mut cond_metrics = Vec::new();
    model
        .with_text_encoder(|encoder| {
            for (case, prompt) in cases {
                let ours = encoder.formatter().unwrap().input_ids(prompt)?;
                assert_eq!(ours, ids(case), "{case}: templated ids");
                layer_diagnosis(checkpoint, case, encoder, &ours)?;
                let states = encoder.encode_ids(&ours)?;
                if let Some(full) = oracle(case) {
                    let rel = |got: &MlxArray, name: &str| {
                        rel_rms(&values(got), &values(&full[&format!("cond.{name}")]))
                    };
                    cond_metrics.push(Some([
                        rel(&states.encoded.cap_feats, "cap_feats"),
                        rel(&states.encoded.cap_feats_2, "cap_feats_2"),
                        rel(&states.text_states, "text_states"),
                    ]));
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
                if cond_metrics.len() < conds.len() + 1 {
                    cond_metrics.push(None);
                }
                conds.push(states.encoded);
            }
            Ok(())
        })
        .unwrap();
    model.evict_text_encoder();

    let resolved = resolve_request(&request(GEN_PROMPT)).unwrap();
    let mut results = Vec::new();
    for (((case, _), cond), cond_rel) in cases.iter().zip(&conds).zip(cond_metrics) {
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
        let mut alpha_out = None;
        let png = oracle(case).map(|full| {
            let drift = rel_rms(&values(&latents), &values(&full["final_latents"]));
            let theirs = mlxcel_core::try_array_to_raw_bytes(&full["pixels"]).unwrap();
            let png = psnr(&pixels, &theirs);
            let alpha_psnr = psnr(&alpha(&pixels), &alpha(&theirs));
            eprintln!(
                "{checkpoint} {case}: final latents rel-RMS {drift:.4e} (drift), PNG PSNR {png:.2} dB, alpha PSNR {alpha_psnr:.2} dB"
            );
            alpha_out = Some(alpha_psnr);
            png
        });
        results.push(CaseResult {
            cond: cond_rel,
            pool,
            png,
            alpha_psnr: alpha_out,
        });
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
    let (poster, rgba) = (&results[0], &results[1]);
    for (case, r) in [("gen", poster), ("rgba", rgba)] {
        if let Some([cap, cap2, text]) = r.cond {
            assert!(
                cap <= COND_CAP_FEATS,
                "{case}: cap_feats rel-RMS {cap:.4e} > {COND_CAP_FEATS}"
            );
            assert!(
                cap2 <= COND_CAP_FEATS_2,
                "{case}: cap_feats_2 rel-RMS {cap2:.4e} > {COND_CAP_FEATS_2}"
            );
            assert!(
                text <= COND_TEXT,
                "{case}: text states rel-RMS {text:.4e} > {COND_TEXT}"
            );
        }
    }
    assert!(
        poster.pool <= GEN_POOL,
        "gen: latents pool rel-RMS {:.4e} > {GEN_POOL}",
        poster.pool
    );
    eprintln!(
        "gen: PNG PSNR {:?} (recorded; conditioner-precision exception)",
        poster.png
    );
    assert!(
        rgba.pool <= RGBA_POOL,
        "rgba: latents pool rel-RMS {:.4e} > {RGBA_POOL}",
        rgba.pool
    );
    if let (Some(png), Some(a)) = (rgba.png, rgba.alpha_psnr) {
        assert!(png >= PNG_PSNR, "rgba: PNG PSNR {png:.2} < {PNG_PSNR}");
        assert!(a >= ALPHA_PSNR, "rgba: alpha PSNR {a:.2} < {ALPHA_PSNR}");
    }
    // The conditioner bound rejects a broken conditioner: the query tokens
    // routed by the text router.
    if let Some(full) = oracle("gen") {
        let root = model_dir("ming-image-0.1-design").unwrap();
        let model = MingImageModel::load(&root).unwrap();
        let wrong = model
            .with_text_encoder(|e| Ok(e.encode_ids_routed(&ids("gen"), false)?.encoded))
            .unwrap();
        let rel = rel_rms(&values(&wrong.cap_feats), &values(&full["cond.cap_feats"]));
        eprintln!("gen with the text router on the query tokens: cap_feats rel-RMS {rel:.4e}");
        assert!(
            rel > COND_CAP_FEATS,
            "the wrong-router arm ({rel:.4e}) passes {COND_CAP_FEATS}"
        );
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

/// Least PSNR of this port's transformer, scheduler and VAE run on
/// upstream's own `cap_feats` / `cap_feats_2`, against upstream's PNG
/// (measured 32.45 dB gen, 52.69 dB rgba).
const ORACLE_COND_PNG_PSNR: f64 = 30.0;
/// Pooled final-latent rel-RMS bounds of that run, per case.
const ORACLE_COND_POOL: [(&str, f64); 2] = [("gen", ORACLE_GEN_POOL), ("rgba", ORACLE_RGBA_POOL)];
/// Measured 3.90e-2 (gen) and 1.38e-3 (rgba); without `cap_feats_2`,
/// 7.73e-1 and 5.68e-1.
const ORACLE_GEN_POOL: f64 = 8e-2;
const ORACLE_RGBA_POOL: f64 = 1e-2;

/// The denoising loop with `cap_feats_2` dropped (the broken-DiT arm).
fn denoise_without_cap2(model: &MingImageModel, cap_feats: &MlxArray) -> UniquePtr<MlxArray> {
    model
        .with_denoiser(|d| {
            let schedule = pipeline::schedule(&d.transformer, SIZE, SIZE, STEPS)?;
            let mut latents = pipeline::initial_noise(SEED, 16, SIZE, SIZE);
            for (i, &sigma) in schedule
                .sigmas()
                .iter()
                .take(schedule.num_steps())
                .enumerate()
            {
                let t = mlxcel_core::astype(
                    &mlxcel_core::from_slice_f32(&[1.0 - sigma], &[1]),
                    mlxcel_core::dtype::BFLOAT16,
                );
                let x = mlxcel_core::astype(&latents, mlxcel_core::dtype::BFLOAT16);
                let pred = d.transformer.forward(&x, &t, cap_feats, None)?;
                let pred = mlxcel_core::astype(&pred, mlxcel_core::dtype::FLOAT32);
                latents = schedule.step(&mlxcel_core::negative(&pred), i, &latents)?;
                mlxcel_core::eval(&latents);
            }
            Ok(latents)
        })
        .unwrap()
}

#[test]
#[ignore = "runs the real bf16 Ming-Image denoiser (GPU); needs MLXCEL_MING_IMAGE_ORACLE_DIR"]
fn bf16_on_upstream_conditioning_matches_upstream() {
    let root = model_dir("ming-image-0.1-design").expect("models/mlx/ming-image-0.1-design");
    if std::env::var_os("MLXCEL_MING_IMAGE_ORACLE_DIR").is_none() {
        eprintln!("skipping: set MLXCEL_MING_IMAGE_ORACLE_DIR");
        return;
    }
    let model = MingImageModel::load(&root).unwrap();
    let resolved = resolve_request(&request(GEN_PROMPT)).unwrap();
    let want = parity();
    for (case, bound) in ORACLE_COND_POOL {
        let full = oracle(case).unwrap();
        let cond = text_encoder::Encoded {
            cap_feats: mlxcel_core::copy(&full["cond.cap_feats"]),
            cap_feats_2: mlxcel_core::copy(&full["cond.cap_feats_2"]),
        };
        let latents = model.sample_latents(&resolved, &cond, SEED).unwrap();
        let pixels = model.decode(&latents).unwrap();
        save_png(&format!("ming_image_oracle_cond_{case}.png"), &pixels);
        let theirs = mlxcel_core::try_array_to_raw_bytes(&full["pixels"]).unwrap();
        let drift = rel_rms(&values(&latents), &values(&full["final_latents"]));
        let pool = rel_rms(
            &latents_pool(&latents),
            &values(&want[&format!("{case}.latents_pool")]),
        );
        let png = psnr(&pixels, &theirs);
        let alpha_psnr = psnr(&alpha(&pixels), &alpha(&theirs));
        eprintln!(
            "{case} on upstream's conditioning: latents pool rel-RMS {pool:.4e}, final latents rel-RMS {drift:.4e}, PNG PSNR {png:.2} dB, alpha PSNR {alpha_psnr:.2} dB"
        );
        assert!(
            pool <= bound,
            "{case}: latents pool rel-RMS {pool:.4e} > {bound}"
        );
        assert!(
            png >= ORACLE_COND_PNG_PSNR,
            "{case}: PNG PSNR {png:.2} < {ORACLE_COND_PNG_PSNR}"
        );
        // Dropping the second caption stream lands outside the bound.
        let skipped = denoise_without_cap2(&model, &cond.cap_feats);
        let skipped = rel_rms(
            &latents_pool(&skipped),
            &values(&want[&format!("{case}.latents_pool")]),
        );
        eprintln!(
            "{case} on upstream's conditioning without cap_feats_2: latents pool rel-RMS {skipped:.4e}"
        );
        assert!(
            skipped > bound,
            "{case}: the no-cap_feats_2 arm ({skipped:.4e}) passes {bound}"
        );
    }
}
