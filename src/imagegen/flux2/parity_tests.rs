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

use std::path::{Path, PathBuf};

use super::Flux2Model;
use super::pipeline::RunOptions;
use super::prompt::DEFAULT_MAX_SEQUENCE_LENGTH;
use crate::imagegen::latent::prepare_packed_latents;

const REL_RMS_MAX: f64 = 1e-2;
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
    let bytes = std::fs::read(fixture(file)).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let t = tensors.tensor(key).unwrap();
    let raw = t.data().to_vec();
    let values = raw
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect();
    (raw, values)
}

fn rel_rms(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (g, w) in got.iter().zip(want) {
        num += f64::from(g - w).powi(2);
        den += f64::from(*w).powi(2);
    }
    (num / den).sqrt()
}

fn psnr(got: &[u8], want: &[u8]) -> f64 {
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

fn png_pixels(path: &Path) -> Vec<u8> {
    image::open(path).unwrap().to_rgb8().into_raw()
}

fn pixels(array: &mlxcel_core::MlxArray) -> Vec<u8> {
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
    eprintln!("flux2 generate parity: final latents rel RMS {rms:.3e}, PNG PSNR {p:.2} dB");
    assert!(rms < REL_RMS_MAX, "final latents rel RMS {rms:e}");
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
    eprintln!("flux2 edit parity: final latents rel RMS {rms:.3e}, PNG PSNR {p:.2} dB");
    assert!(rms < REL_RMS_MAX, "final latents rel RMS {rms:e}");
    assert!(p >= PSNR_MIN, "PSNR {p} dB");
}
