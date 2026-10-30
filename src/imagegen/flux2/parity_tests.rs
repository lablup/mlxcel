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

//! Seed-fixed parity against upstream mlx-vlm on the real 4B checkpoint.
//!
//! Fixtures in `tests/fixtures/flux2/` come from
//! `tests/fixtures/flux2/generate_flux2_fixtures.py` (mlx-vlm 0.7.8 at git
//! 73d366ec, mlx 0.32.4, CUDA backend on an NVIDIA GB10, `MLX_ENABLE_TF32=0`).
//! The tests are `#[ignore]`: they need `models/mlx/flux.2-klein-4b-bf16`
//! (`black-forest-labs/FLUX.2-klein-4B`) and a GPU, and run with
//! `MLX_ENABLE_TF32=0`.
//!
//! The port is checked in three layers:
//!
//! - the transformer in f32 against upstream in f32 on identical inputs, which
//!   must agree to f32 rounding (measured: bit-identical);
//! - the VAE decoding upstream's own final latents, against upstream's PNG;
//! - the whole bf16 pipeline, against upstream's PNG (PSNR) and final latents.
//!
//! The bf16 final latents cannot meet a 1e-2 relative RMS: bf16 matmuls with
//! long reductions (the 7680-wide context embedder, the text encoder) round
//! differently in mlxcel's MLX build and the oracle's mlx 0.32.4, and four
//! Euler steps amplify that. Upstream run in f32 differs from upstream in bf16
//! by 7.1e-2 on the same case, so the end-to-end latent bound is a sanity
//! bound that a wrong port (relative RMS near 1.4) cannot pass.

use std::path::{Path, PathBuf};

use super::Flux2Model;
use super::embeddings::AffineLinears;
use super::pipeline::RunOptions;
use super::prompt::{DEFAULT_MAX_SEQUENCE_LENGTH, prepare_text_ids};
use super::transformer::{Flux2Transformer, TransformerInput};
use crate::imagegen::latent::prepare_packed_latents;

/// f32 transformer against f32 upstream.
const F32_REL_RMS_MAX: f64 = 1e-5;
/// bf16 end-to-end final latents (see the module docs).
const BF16_LATENT_REL_RMS_MAX: f64 = 0.2;
const PSNR_MIN: f64 = 30.0;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn checkpoint() -> Option<PathBuf> {
    let dir = std::env::var_os("MLXCEL_TEST_FLUX2_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("models/mlx/flux.2-klein-4b-bf16"));
    dir.join("transformer").is_dir().then_some(dir)
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/fixtures/flux2").join(name)
}

/// `(raw bf16 bytes, f32 values)` of a `*_bf16_bits` (uint16) tensor.
fn bf16_fixture(file: &str, key: &str) -> (Vec<u8>, Vec<f32>) {
    bf16_tensor(&fixture(file), key)
}

/// [`bf16_fixture`] for a safetensors file at any path.
///
/// Used by: Flux2 and Bonsai parity tests.
pub(crate) fn bf16_tensor(path: &Path, key: &str) -> (Vec<u8>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let t = tensors.tensor(key).unwrap();
    let raw = t.data().to_vec();
    let values = raw
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect();
    (raw, values)
}

fn f32_fixture(file: &str, key: &str) -> Vec<f32> {
    f32_tensor(&fixture(file), key)
}

/// [`f32_fixture`] for a safetensors file at any path.
///
/// Used by: Flux2 and Bonsai parity tests.
pub(crate) fn f32_tensor(path: &Path, key: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    tensors
        .tensor(key)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub(crate) fn rel_rms(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (g, w) in got.iter().zip(want) {
        num += f64::from(g - w).powi(2);
        den += f64::from(*w).powi(2);
    }
    (num / den).sqrt()
}

pub(crate) fn psnr(got: &[u8], want: &[u8]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mse = got
        .iter()
        .zip(want)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>()
        / got.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

pub(crate) fn png_pixels(path: &Path) -> Vec<u8> {
    image::open(path).unwrap().to_rgb8().into_raw()
}

pub(crate) fn pixels(array: &mlxcel_core::MlxArray) -> Vec<u8> {
    let one = mlxcel_core::copy(array);
    mlxcel_core::eval(&one);
    mlxcel_core::array_to_raw_bytes(&one).to_vec()
}

/// The initial packed noise is bit-identical to upstream's (bf16 values are
/// exact in f32, so comparing f32 bit patterns compares the bf16 bits).
fn assert_noise_matches(file: &str, seed: u64) {
    let (_, want) = bf16_fixture(file, "noise_bf16_bits");
    let noise =
        prepare_packed_latents(seed, 512, 512, 1, 32, 8, mlxcel_core::dtype::BFLOAT16).unwrap();
    let got = mlxcel_core::utils::array_to_vec_f32(&noise.latents);
    assert_eq!(got.len(), want.len());
    let mismatches = got
        .iter()
        .zip(&want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    assert_eq!(
        mismatches,
        0,
        "initial noise differs from upstream in {mismatches} of {} values (first: {:?} vs {:?})",
        got.len(),
        &got[..4],
        &want[..4]
    );
}

fn options() -> RunOptions {
    RunOptions {
        max_sequence_length: DEFAULT_MAX_SEQUENCE_LENGTH,
        tiled_vae: None,
    }
}

#[test]
#[ignore = "needs models/mlx/flux.2-klein-4b-bf16 and a GPU"]
fn klein_4b_generate_matches_upstream() {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping: FLUX.2-klein-4B checkpoint is absent");
        return;
    };
    let file = "generate_512_seed9909.safetensors";
    let (_, want_latents) = bf16_fixture(file, "final_latents_bf16_bits");
    assert_noise_matches(file, 9909);

    let model = Flux2Model::load(&dir).unwrap();
    let pipeline = model.pipeline();
    assert_eq!(pipeline.compute_dtype(), mlxcel_core::dtype::BFLOAT16);
    let prompt = "A cinematic portrait of a glass bonsai tree on a moonlit desk";
    let latents = pipeline
        .generate_latents(prompt, None, 9909, 4, 512, 512, 1.0, 1, &options())
        .unwrap();
    let got = mlxcel_core::utils::array_to_vec_f32(&latents);
    let rms = rel_rms(&got, &want_latents);
    let image = pipeline.decode(&latents, 512, 512, None, true).unwrap();
    let p = psnr(
        &pixels(&image),
        &png_pixels(&fixture("generate_512_seed9909.png")),
    );
    eprintln!(
        "flux2 generate parity (VAE dtype {}): final latents rel RMS {rms:.3e}, PNG PSNR {p:.2} dB",
        pipeline.vae_dtype()
    );
    assert!(
        rms < BF16_LATENT_REL_RMS_MAX,
        "final latents rel RMS {rms:e}"
    );
    assert!(p >= PSNR_MIN, "PSNR {p} dB");
}

#[test]
#[ignore = "needs models/mlx/flux.2-klein-4b-bf16 and a GPU"]
fn klein_4b_edit_matches_upstream() {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping: FLUX.2-klein-4B checkpoint is absent");
        return;
    };
    let file = "edit_512_seed123.safetensors";
    let (_, want_latents) = bf16_fixture(file, "final_latents_bf16_bits");
    assert_noise_matches(file, 123);

    let model = Flux2Model::load(&dir).unwrap();
    // The loaded model hands the surfaces upstream's edit size, not the
    // reference's own (the 512x512 fixture maps to itself).
    let edit_size = crate::imagegen::ImageGenerationModel::default_edit_size(&model);
    assert_eq!(edit_size(4000, 3000), (1168, 880));
    assert_eq!(edit_size(512, 512), (512, 512));
    let pipeline = model.pipeline();
    let reference = image::open(fixture("edit_reference.png")).unwrap();
    let latents = pipeline
        .edit_latents(
            "Add black sunglasses",
            None,
            &[reference],
            123,
            4,
            512,
            512,
            1.0,
            &options(),
        )
        .unwrap();
    let got = mlxcel_core::utils::array_to_vec_f32(&latents);
    let rms = rel_rms(&got, &want_latents);
    let image = pipeline.decode(&latents, 512, 512, None, false).unwrap();
    let p = psnr(
        &pixels(&image),
        &png_pixels(&fixture("edit_512_seed123.png")),
    );
    eprintln!(
        "flux2 edit parity (VAE dtype {}): final latents rel RMS {rms:.3e}, PNG PSNR {p:.2} dB",
        pipeline.vae_dtype()
    );
    assert!(
        rms < BF16_LATENT_REL_RMS_MAX,
        "final latents rel RMS {rms:e}"
    );
    assert!(p >= PSNR_MIN, "PSNR {p} dB");
}

#[test]
#[ignore = "needs models/mlx/flux.2-klein-4b-bf16 and a GPU"]
fn klein_4b_transformer_f32_matches_upstream_f32() {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping: FLUX.2-klein-4B checkpoint is absent");
        return;
    };
    let f32 = mlxcel_core::dtype::FLOAT32;
    let file = "transformer_step0_f32.safetensors";
    let (_, embeds) = bf16_fixture(file, "embeds_bf16_bits");
    let want = f32_fixture(file, "pred_first128_f32");
    let embeds = mlxcel_core::from_slice_f32(&embeds, &[1, 32, 7680]);
    let noise = prepare_packed_latents(9909, 512, 512, 1, 32, 8, f32).unwrap();
    let model = Flux2Model::load(&dir).unwrap();
    let pipeline = model.pipeline();
    let transformer = Flux2Transformer::load(
        &pipeline.layout.transformer,
        pipeline.transformer_config.clone(),
        &AffineLinears { quant: None },
        f32,
    )
    .unwrap();
    // The f32 noise is upstream's bf16 noise before rounding; round it the
    // same way upstream's first step saw it.
    let latents = mlxcel_core::astype(
        &mlxcel_core::astype(&noise.latents, mlxcel_core::dtype::BFLOAT16),
        f32,
    );
    let text_ids = prepare_text_ids(1, 32);
    let out = transformer
        .forward(
            &TransformerInput {
                hidden: &latents,
                encoder: &embeds,
                timestep: 1000.0,
                img_ids: &noise.latent_ids,
                txt_ids: &text_ids,
                guidance: None,
            },
            None,
        )
        .unwrap();
    assert_eq!(mlxcel_core::array_dtype(&out), f32);
    let first = mlxcel_core::slice(&out, &[0, 0, 0], &[1, 128, 128]);
    let got = mlxcel_core::utils::array_to_vec_f32(&first);
    let rms = rel_rms(&got, &want);
    let exact = got
        .iter()
        .zip(&want)
        .filter(|(g, w)| g.to_bits() == w.to_bits())
        .count();
    eprintln!(
        "flux2 f32 transformer parity: rel RMS {rms:.3e}, {exact} of {} values bit-identical",
        got.len()
    );
    assert!(rms < F32_REL_RMS_MAX, "f32 transformer rel RMS {rms:e}");
}

/// The bucketed prompt embeddings of the generate case against the ones
/// upstream fed its first step (`embeds_bf16_bits`, `[1, 32, 7680]`).
/// Recorded drift: 1.19e-2 on GB10, with the encoder's attention in bf16 or
/// in f32 like upstream (#2666), so the bound is a sanity bound.
#[test]
#[ignore = "needs models/mlx/flux.2-klein-4b-bf16 and a GPU"]
fn klein_4b_prompt_embeds_match_upstream() {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping: FLUX.2-klein-4B checkpoint is absent");
        return;
    };
    let prompt = "A cinematic portrait of a glass bonsai tree on a moonlit desk";
    let (_, want) = bf16_fixture("transformer_step0_f32.safetensors", "embeds_bf16_bits");
    let model = Flux2Model::load(&dir).unwrap();
    let pipeline = model.pipeline();
    let embeds = pipeline
        .encode(prompt, DEFAULT_MAX_SEQUENCE_LENGTH, true)
        .unwrap();
    assert_eq!(mlxcel_core::array_shape(&embeds.embeds), vec![1, 32, 7680]);
    let got = mlxcel_core::utils::array_to_vec_f32(&embeds.embeds);
    let err = rel_rms(&got, &want);
    eprintln!("flux2 prompt embeds parity: rel RMS {err:.3e}");
    assert!(err < 0.1, "prompt embeds rel RMS {err:e}");
}

/// Decode upstream's final latents with this VAE and compare to its PNG.
fn vae_only_psnr(file: &str, png: &str, auto_mode: bool) -> f64 {
    let dir = checkpoint().expect("checkpoint");
    let (_, latents) = bf16_fixture(file, "final_latents_bf16_bits");
    let latents = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&latents, &[1, 128, 32, 32]),
        mlxcel_core::dtype::BFLOAT16,
    );
    let model = Flux2Model::load(&dir).unwrap();
    let pipeline = model.pipeline();
    pipeline.ensure_transformer_and_vae().unwrap();
    let image = pipeline
        .decode(&latents, 512, 512, None, auto_mode)
        .unwrap();
    psnr(&pixels(&image), &png_pixels(&fixture(png)))
}

#[test]
#[ignore = "needs models/mlx/flux.2-klein-4b-bf16 and a GPU"]
fn klein_4b_vae_decodes_upstream_latents_like_upstream() {
    if checkpoint().is_none() {
        eprintln!("skipping: FLUX.2-klein-4B checkpoint is absent");
        return;
    }
    let tiled = vae_only_psnr(
        "generate_512_seed9909.safetensors",
        "generate_512_seed9909.png",
        true,
    );
    let untiled = vae_only_psnr(
        "edit_512_seed123.safetensors",
        "edit_512_seed123.png",
        false,
    );
    eprintln!("flux2 VAE-only parity: tiled {tiled:.2} dB, untiled {untiled:.2} dB");
    assert!(tiled >= PSNR_MIN, "tiled VAE PSNR {tiled} dB");
    assert!(untiled >= PSNR_MIN, "untiled VAE PSNR {untiled} dB");
}

/// A batch decodes image by image with the tiling one image gets, so image
/// `i` of a batch is exactly what its latents decode to alone, and the batch
/// never runs the full-resolution decoder on every image at once.
#[test]
#[ignore = "needs models/mlx/flux.2-klein-4b-bf16 and a GPU"]
fn klein_4b_batched_decode_equals_each_image_decoded_alone() {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping: FLUX.2-klein-4B checkpoint is absent");
        return;
    };
    let (_, a) = bf16_fixture(
        "generate_512_seed9909.safetensors",
        "final_latents_bf16_bits",
    );
    let (_, b) = bf16_fixture("edit_512_seed123.safetensors", "final_latents_bf16_bits");
    let one = |v: &[f32]| {
        mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(v, &[1, 128, 32, 32]),
            mlxcel_core::dtype::BFLOAT16,
        )
    };
    let (a, b) = (one(&a), one(&b));
    let batch = mlxcel_core::concatenate(&a, &b, 0);
    let model = Flux2Model::load(&dir).unwrap();
    let pipeline = model.pipeline();
    pipeline.ensure_transformer_and_vae().unwrap();
    let decode = |latents: &mlxcel_core::MlxArray| {
        pixels(&pipeline.decode(latents, 512, 512, None, true).unwrap())
    };
    let batched = decode(&batch);
    let alone = [decode(&a), decode(&b)].concat();
    assert_eq!(batched.len(), alone.len());
    assert!(
        batched == alone,
        "a batched decode differs from per-image decodes"
    );
}

#[test]
#[ignore = "needs the FLUX.2 Klein checkpoints under models/mlx"]
fn local_checkpoints_resolve_to_their_variant_and_defaults() {
    use super::config::Flux2Variant::{self, *};
    let cases: [(&str, Flux2Variant, u32, f32); 6] = [
        ("flux.2-klein-4b-bf16", Klein4b, 4, 1.0),
        ("flux.2-klein-base-4b", KleinBase4b, 50, 4.0),
        ("flux.2-klein-9b", Klein9b, 4, 1.0),
        ("flux.2-klein-9b-kv", Klein9bKv, 4, 1.0),
        ("flux2-klein-4b-4bit", Klein4b, 4, 1.0),
        ("flux2-klein-9b-4bit", Klein9b, 4, 1.0),
    ];
    for (name, want, steps, guidance) in cases {
        let dir = root().join("models/mlx").join(name);
        if !dir.is_dir() {
            eprintln!("skipping {name}: absent");
            continue;
        }
        let got = Flux2Variant::from_dir(&dir).unwrap();
        assert_eq!(got, want, "{name}");
        assert_eq!(got.default_steps(), steps, "{name}");
        assert_eq!(got.default_guidance(), guidance, "{name}");
    }
}
