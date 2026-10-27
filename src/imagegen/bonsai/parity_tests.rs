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

//! Seed-fixed parity against upstream mlx-vlm on the real Bonsai checkpoint.
//!
//! Fixtures in `tests/fixtures/bonsai/` come from
//! `tests/fixtures/bonsai/generate_bonsai_fixtures.py`: upstream's Bonsai
//! pipeline (mlx-vlm 0.7.8 at git 73d366ec, mlx 0.32.4, CUDA backend on an
//! NVIDIA GB10, `MLX_ENABLE_TF32=0`) with the Metal-only fused norm-RoPE
//! kernels replaced by their arithmetic and the checkpoint's own `vae/` as
//! the decoder (see the script). The tests are `#[ignore]`: they need
//! `models/mlx/bonsai-image-ternary-4b-mlx-2bit` and a GPU.

use std::path::{Path, PathBuf};

use super::BonsaiModel;
use super::config::{PackedQuantConfig, TRANSFORMER_DIR};
use super::loader::PackedLinears;
use crate::imagegen::flux2::config::{Flux2TransformerConfig, Flux2Variant};
use crate::imagegen::flux2::parity_tests::{bf16_tensor, pixels, png_pixels, psnr, rel_rms};
use crate::imagegen::flux2::pipeline::RunOptions;
use crate::imagegen::flux2::prompt::DEFAULT_MAX_SEQUENCE_LENGTH;
use crate::imagegen::flux2::transformer::{Flux2Transformer, TransformerInput};
use crate::imagegen::latent::prepare_packed_latents;

const PROMPT: &str = "A tiny glass bonsai tree on a moonlit desk";
const CASE: &str = "generate_512_seed9909";
const PSNR_MIN: f64 = 30.0;
/// End-to-end bf16 final latents: a sanity bound, not a precision claim.
///
/// Measured 1.0e-1 against the fixture. Upstream disagrees with itself by
/// 3.5e-2 on the same case when its FLUX.2 transformer replaces `klein_fast`
/// (same weights, same arithmetic in a different op order), and its own
/// bf16 conditioner is 1.95e-2 from f32 (see
/// `bonsai_prompt_embeds_match_upstream`); four Euler steps amplify that. The
/// port's graph is checked in f32 by
/// `bonsai_transformer_f32_matches_upstream_f32`. A wrong port scores about 1.4.
const LATENT_REL_RMS_MAX: f64 = 0.2;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn checkpoint() -> Option<PathBuf> {
    let dir = std::env::var_os("MLXCEL_TEST_BONSAI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("models/mlx/bonsai-image-ternary-4b-mlx-2bit"));
    dir.join(TRANSFORMER_DIR).is_dir().then_some(dir)
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/fixtures/bonsai").join(name)
}

fn options() -> RunOptions {
    RunOptions {
        max_sequence_length: DEFAULT_MAX_SEQUENCE_LENGTH,
        tiled_vae: None,
    }
}

#[test]
#[ignore = "needs models/mlx/bonsai-image-ternary-4b-mlx-2bit and a GPU"]
fn bonsai_generate_matches_upstream() {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping: Bonsai checkpoint is absent");
        return;
    };
    let file = fixture(&format!("{CASE}.safetensors"));
    let (_, want_noise) = bf16_tensor(&file, "noise_bf16_bits");
    let (_, want_latents) = bf16_tensor(&file, "final_latents_bf16_bits");

    let noise =
        prepare_packed_latents(9909, 512, 512, 1, 32, 8, mlxcel_core::dtype::BFLOAT16).unwrap();
    let got_noise = mlxcel_core::utils::array_to_vec_f32(&noise.latents);
    let mismatches = got_noise
        .iter()
        .zip(&want_noise)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    assert_eq!(got_noise.len(), want_noise.len());
    assert_eq!(
        mismatches, 0,
        "initial noise differs in {mismatches} values"
    );

    let model = BonsaiModel::load(&dir).unwrap();
    let pipeline = model.pipeline();
    let latents = pipeline
        .generate_latents(PROMPT, None, 9909, 4, 512, 512, 1.0, 1, &options())
        .unwrap();
    let rms = rel_rms(
        &mlxcel_core::utils::array_to_vec_f32(&latents),
        &want_latents,
    );
    let image = pipeline.decode(&latents, 512, 512, None, true).unwrap();
    let p = psnr(
        &pixels(&image),
        &png_pixels(&fixture(&format!("{CASE}.png"))),
    );
    eprintln!("bonsai generate parity: final latents rel RMS {rms:.3e}, PNG PSNR {p:.2} dB");
    assert!(rms < LATENT_REL_RMS_MAX, "final latents rel RMS {rms:e}");
    assert!(p >= PSNR_MIN, "PSNR {p} dB");
}

/// The VAE decoding upstream's own final latents, against upstream's PNG.
#[test]
#[ignore = "needs models/mlx/bonsai-image-ternary-4b-mlx-2bit and a GPU"]
fn bonsai_vae_decodes_upstream_latents_like_upstream() {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping: Bonsai checkpoint is absent");
        return;
    };
    let (_, latents) = bf16_tensor(
        &fixture(&format!("{CASE}.safetensors")),
        "final_latents_bf16_bits",
    );
    let latents = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&latents, &[1, 128, 32, 32]),
        mlxcel_core::dtype::BFLOAT16,
    );
    let model = BonsaiModel::load(&dir).unwrap();
    let pipeline = model.pipeline();
    pipeline.ensure_transformer_and_vae().unwrap();
    let image = pipeline.decode(&latents, 512, 512, None, true).unwrap();
    let p = psnr(
        &pixels(&image),
        &png_pixels(&fixture(&format!("{CASE}.png"))),
    );
    eprintln!("bonsai VAE-only parity: {p:.2} dB");
    assert!(p >= PSNR_MIN, "VAE PSNR {p} dB");
}

/// One transformer call on upstream's exact step-0 inputs. The inputs hold
/// the 7.5 MiB prompt embeddings, so they are not committed: write them with
/// `generate_bonsai_fixtures.py --step0 DIR` and point
/// `MLXCEL_TEST_BONSAI_STEP0` at `DIR/step0.safetensors`.
#[test]
#[ignore = "needs models/mlx/bonsai-image-ternary-4b-mlx-2bit, a GPU and MLXCEL_TEST_BONSAI_STEP0"]
fn bonsai_transformer_step0_matches_upstream() {
    let (Some(dir), Some(step0)) = (
        checkpoint(),
        std::env::var_os("MLXCEL_TEST_BONSAI_STEP0").map(PathBuf::from),
    ) else {
        eprintln!("skipping: Bonsai checkpoint or MLXCEL_TEST_BONSAI_STEP0 is absent");
        return;
    };
    let bf16 = mlxcel_core::dtype::BFLOAT16;
    let tensor = |key: &str, shape: &[i32]| {
        let (_, v) = bf16_tensor(&step0, key);
        mlxcel_core::astype(&mlxcel_core::from_slice_f32(&v, shape), bf16)
    };
    let latents = tensor("latents_bf16_bits", &[1, 1024, 128]);
    let embeds = tensor("embeds_bf16_bits", &[1, 512, 7680]);
    let (_, want) = bf16_tensor(&step0, "pred_bf16_bits");
    let text_ids = crate::imagegen::flux2::prompt::prepare_text_ids(1, 512);
    let noise = prepare_packed_latents(9909, 512, 512, 1, 32, 8, bf16).unwrap();

    let packed = dir.join(TRANSFORMER_DIR);
    let config = Flux2TransformerConfig::resolve(&packed, Flux2Variant::Klein4b).unwrap();
    let source = PackedLinears::new(PackedQuantConfig::from_dir(&packed).unwrap());
    let transformer = Flux2Transformer::load(&packed, config, &source, bf16).unwrap();
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
    let rms = rel_rms(&mlxcel_core::utils::array_to_vec_f32(&out), &want);
    eprintln!("bonsai step-0 transformer parity (bf16, identical inputs): rel RMS {rms:.3e}");
    assert!(rms < 1e-2, "step-0 transformer rel RMS {rms:e}");
}

/// The prompt embeddings upstream's 4-bit Qwen3 conditioner produces for the
/// fixture prompt, padded to 512 tokens (`bucketed_seq_len=False`).
///
/// Measured: rel RMS 2.3e-2 overall, 8.8e-3 on the first 16 tokens and up to
/// 1.7e-1 on the padding rows. Against upstream's conditioner run in f32
/// (dequantized weights), upstream bf16 is 1.95e-2 off and this port 1.74e-2,
/// with the same growth along the padding, so the bound is a sanity bound
/// (bf16 rounding in the 36-layer encoder) that a wrong port cannot pass.
#[test]
#[ignore = "needs models/mlx/bonsai-image-ternary-4b-mlx-2bit, a GPU and MLXCEL_TEST_BONSAI_STEP0"]
fn bonsai_prompt_embeds_match_upstream() {
    let (Some(dir), Some(step0)) = (
        checkpoint(),
        std::env::var_os("MLXCEL_TEST_BONSAI_STEP0").map(PathBuf::from),
    ) else {
        eprintln!("skipping: Bonsai checkpoint or MLXCEL_TEST_BONSAI_STEP0 is absent");
        return;
    };
    let (_, want) = bf16_tensor(&step0, "embeds_bf16_bits");
    let model = BonsaiModel::load(&dir).unwrap();
    let embeds = model
        .pipeline()
        .encode(PROMPT, DEFAULT_MAX_SEQUENCE_LENGTH, false)
        .unwrap();
    assert_eq!(mlxcel_core::array_shape(&embeds.embeds), vec![1, 512, 7680]);
    let got = mlxcel_core::utils::array_to_vec_f32(&embeds.embeds);
    let rms = rel_rms(&got, &want);
    let row = 7680;
    for (a, b) in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 512)] {
        eprintln!(
            "  tokens {a}..{b}: rel RMS {:.3e}",
            rel_rms(&got[a * row..b * row], &want[a * row..b * row])
        );
    }
    for (l, (a, b)) in [(9, (0, 2560)), (18, (2560, 5120)), (27, (5120, 7680))] {
        let pick = |v: &[f32]| -> Vec<f32> {
            v.chunks_exact(row).flat_map(|r| r[a..b].to_vec()).collect()
        };
        eprintln!(
            "  layer {l}: rel RMS {:.3e}",
            rel_rms(&pick(&got), &pick(&want))
        );
    }
    eprintln!("bonsai prompt embeds parity: rel RMS {rms:.3e}");
    assert!(rms < 0.1, "prompt embeds rel RMS {rms:e}");
}

/// The transformer in f32 with Bonsai's packed weights against upstream's
/// FLUX.2 transformer in f32 on identical inputs (see the fixture script).
#[test]
#[ignore = "needs models/mlx/bonsai-image-ternary-4b-mlx-2bit and a GPU"]
fn bonsai_transformer_f32_matches_upstream_f32() {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping: Bonsai checkpoint is absent");
        return;
    };
    let f32 = mlxcel_core::dtype::FLOAT32;
    let file = fixture("transformer_step0_f32.safetensors");
    let (_, embeds) = bf16_tensor(&file, "embeds_bf16_bits");
    let want = crate::imagegen::flux2::parity_tests::f32_tensor(&file, "pred_first128_f32");
    let embeds = mlxcel_core::from_slice_f32(&embeds, &[1, 32, 7680]);
    let noise = prepare_packed_latents(9909, 512, 512, 1, 32, 8, f32).unwrap();
    let latents = mlxcel_core::astype(
        &mlxcel_core::astype(&noise.latents, mlxcel_core::dtype::BFLOAT16),
        f32,
    );
    let packed = dir.join(TRANSFORMER_DIR);
    let config = Flux2TransformerConfig::resolve(&packed, Flux2Variant::Klein4b).unwrap();
    let source = PackedLinears::new(PackedQuantConfig::from_dir(&packed).unwrap());
    let transformer = Flux2Transformer::load(&packed, config, &source, f32).unwrap();
    let text_ids = crate::imagegen::flux2::prompt::prepare_text_ids(1, 32);
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
        "bonsai f32 transformer parity: rel RMS {rms:.3e}, {exact} of {} values bit-identical",
        got.len()
    );
    assert!(rms < 1e-5, "f32 transformer rel RMS {rms:e}");
}
