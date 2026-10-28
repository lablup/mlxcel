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

//! Mage-VAE parity against upstream's `MageVAE` on the real checkpoint
//! (`vae.*` of `tests/fixtures/imagegen/mage_flow_tiny_refs.safetensors`,
//! written by `generate_mage_flow_fixtures.py tiny --vae <checkpoint>`).
//!
//! Each stage is gated on upstream's own input at 5e-3 relative RMS: the CoD
//! decoder, DiCo blocks 7 and 20 of the denoiser, the pixel head, and the
//! encoder's last DiCo block and posterior head. The end-to-end decode and
//! encode are only bounded (at a level a wrong port is far above): 21 bf16
//! DiCo blocks amplify single-ulp reduction-order ties, and the oracle's
//! CUDA `mx.mean` of bf16 accumulates in bf16. Measured end to end:
//! decode 1.0e-2, encode mean 2.4e-2.
//! The real-weight tests skip when `models/mlx/mage-flow-turbo` is absent.

use std::path::PathBuf;

use mlxcel_core::weights::WeightMap;

use super::*;
use crate::imagegen::mage_flow::transformer::tests::{refs, rel_rms, to_vec};
use crate::imagegen::testing::det;
use mlxcel_core::dtype;

const STAGE_TOL: f32 = 5e-3;
/// Same-input bound on the last denoiser block, whose only difference from
/// upstream is the oracle's bf16 `mx.mean` accumulation (measured 9.8e-3;
/// 0.0 during the port against upstream patched to accumulate in f32).
const DEEP_BLOCK_TOL: f32 = 2e-2;
/// Bound on the end-to-end drift; a wrong port lands near 1.
const END_TO_END_BOUND: f32 = 5e-2;

fn checkpoint() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/mlx/mage-flow-turbo");
    root.join("vae").is_dir().then_some(root)
}

fn bf16(refs: &WeightMap, key: &str) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(&refs[key], dtype::BFLOAT16)
}

fn gate(name: &str, got: &MlxArray, want: &MlxArray, tol: f32) {
    let err = rel_rms(got, want);
    eprintln!("{name}: rel-RMS {err:.3e}");
    assert!(err < tol, "{name} rel-RMS {err} >= {tol}");
}

fn test_latent() -> UniquePtr<MlxArray> {
    mlxcel_core::astype(
        &det("mage_vae.latent", &[1, 4, 6, 128], 1.0, 0.0),
        dtype::BFLOAT16,
    )
}

fn test_image() -> UniquePtr<MlxArray> {
    let image = det("mage_vae.image", &[1, 32, 48, 3], 0.9, 0.0);
    let one = mlxcel_core::full_f32(&[1], 1.0, dtype::FLOAT32);
    let image = mlxcel_core::clip(&image, &mlxcel_core::negative(&one), &one);
    mlxcel_core::astype(&image, dtype::BFLOAT16)
}

#[test]
fn decoder_stages_match_upstream_on_the_real_vae() {
    let Some(root) = checkpoint() else {
        eprintln!("skipping: models/mlx/mage-flow-turbo is absent");
        return;
    };
    let vae = crate::imagegen::mage_flow::weights::load_vae(&root, false).unwrap();
    let refs = refs();
    let condition = vae.cod_decoder.forward(&test_latent());
    gate("cod", &condition, &refs["vae.cod"], STAGE_TOL);
    let condition = bf16(&refs, "vae.cod");
    let temb = vae.denoiser.temb(&condition);
    // Block 20 sees activations large enough that the oracle's bf16-accumulated
    // spatial mean is off by more than one bf16 ulp; against upstream patched
    // to accumulate in f32 the same block was bit-exact (0.0) during the port.
    for (index, tol) in [(7, STAGE_TOL), (20, DEEP_BLOCK_TOL)] {
        let input = bf16(&refs, &format!("vae.dec_block{index}_in"));
        let out = vae.denoiser.block(index).forward(&input, &temb);
        gate(
            &format!("denoiser block {index}"),
            &out,
            &refs[&format!("vae.dec_block{index}_out")],
            tol,
        );
    }
    let pixels = vae
        .denoiser
        .pixel_head(&condition, &bf16(&refs, "vae.dec_block20_out"));
    gate("pixel head", &pixels, &refs["vae.decode"], STAGE_TOL);

    let out = vae.decode(&test_latent());
    assert_eq!(mlxcel_core::array_shape(&out), [1, 64, 96, 3]);
    assert_eq!(mlxcel_core::array_dtype(&out), dtype::BFLOAT16);
    gate(
        "decode (end to end)",
        &out,
        &refs["vae.decode"],
        END_TO_END_BOUND,
    );
    // Loaded without its encoder.
    let image = mlxcel_core::zeros(&[1, 32, 32, 3], dtype::BFLOAT16);
    assert!(vae.encode_moments(&image).is_err());
}

#[test]
fn encoder_stages_match_upstream_on_the_real_vae() {
    let Some(root) = checkpoint() else {
        eprintln!("skipping: models/mlx/mage-flow-turbo is absent");
        return;
    };
    let vae = crate::imagegen::mage_flow::weights::load_vae(&root, true).unwrap();
    let refs = refs();
    let encoder = vae.encoder.as_ref().unwrap();
    let temb = encoder.temb(1, dtype::BFLOAT16);
    let out = encoder
        .block(20)
        .forward(&bf16(&refs, "vae.enc_block20_in"), &temb);
    gate(
        "encoder block 20",
        &out,
        &refs["vae.enc_block20_out"],
        STAGE_TOL,
    );
    let (mean, logvar) = encoder.moments(&bf16(&refs, "vae.enc_block20_out"));
    gate("posterior mean", &mean, &refs["vae.encode_mean"], STAGE_TOL);
    gate(
        "posterior logvar",
        &logvar,
        &refs["vae.encode_logvar"],
        STAGE_TOL,
    );

    let image = test_image();
    let (mean, logvar) = vae.encode_moments(&image).unwrap();
    gate(
        "encode mean (end to end)",
        &mean,
        &refs["vae.encode_mean"],
        END_TO_END_BOUND,
    );
    gate(
        "encode logvar (end to end)",
        &logvar,
        &refs["vae.encode_logvar"],
        END_TO_END_BOUND,
    );
    assert!(to_vec(&logvar).iter().all(|v| (-20.0..=10.0).contains(v)));
    // The posterior mean is the no-sample encode; a sample differs from it.
    let key = mlxcel_core::random_key(42);
    let plain = vae.encode(&image, false, &key).unwrap();
    assert_eq!(to_vec(&plain), to_vec(&mean));
    let sampled = vae.encode(&image, true, &key).unwrap();
    assert_ne!(to_vec(&sampled), to_vec(&mean));
    // Sides that are not multiples of 16 are refused.
    let odd = mlxcel_core::zeros(&[1, 24, 32, 3], dtype::BFLOAT16);
    assert!(vae.encode_moments(&odd).is_err());
}

#[test]
fn local_attention_windows_round_trip_with_edge_padding() {
    // 40x70 cells pad to 64x96: two by three 32-cell windows.
    let x = det("windows", &[1, 40, 70, 4], 1.0, 0.0);
    let (patches, geometry) = decoder::LocalAttentionBlock::patchify(&x);
    assert_eq!(mlxcel_core::array_shape(&patches), [6, 1024, 4]);
    let back = decoder::LocalAttentionBlock::unpatchify(&patches, &geometry);
    assert_eq!(to_vec(&back), to_vec(&x));
    // Edge padding repeats the last row and column.
    let padded = decoder::edge_pad(&x, 2, 3);
    assert_eq!(mlxcel_core::array_shape(&padded), [1, 42, 73, 4]);
    let v = to_vec(&padded);
    let src = to_vec(&x);
    let at = |v: &[f32], w: usize, y: usize, xx: usize, c: usize| v[(y * w + xx) * 4 + c];
    assert_eq!(at(&v, 73, 41, 72, 1), at(&src, 70, 39, 69, 1));
    assert_eq!(at(&v, 73, 10, 71, 2), at(&src, 70, 10, 69, 2));
}

#[test]
fn layer_norm_is_deterministic_on_384_wide_bf16_rows() {
    // The encoder's `norm_out` shape. MLX's fused CUDA layer_norm returned a
    // few wrong rows per call here (see `LayerNorm::forward`).
    let x = mlxcel_core::astype(&det("ln.x", &[2, 32, 32, 384], 1.0, 0.0), dtype::BFLOAT16);
    let mut w = WeightMap::new();
    w.insert(
        "ln.weight".to_string(),
        mlxcel_core::astype(&det("ln.w", &[384], 1.0, 1.0), dtype::BFLOAT16),
    );
    w.insert(
        "ln.bias".to_string(),
        mlxcel_core::astype(&det("ln.b", &[384], 0.5, 0.0), dtype::BFLOAT16),
    );
    let norm = crate::imagegen::mage_flow::layers::LayerNorm::affine(&w, "ln", 1e-6).unwrap();
    let first = norm.forward(&x);
    assert_eq!(mlxcel_core::array_dtype(&first), dtype::BFLOAT16);
    let first = to_vec(&first);
    for _ in 0..10 {
        assert_eq!(to_vec(&norm.forward(&x)), first);
    }
    // Against an f64 layer norm of the same bf16 inputs: one bf16 rounding.
    let (xs, ws, bs) = (to_vec(&x), to_vec(&w["ln.weight"]), to_vec(&w["ln.bias"]));
    for (row, got) in xs.chunks(384).zip(first.chunks(384)) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / 384.0;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / 384.0;
        for (i, (&v, &g)) in row.iter().zip(got).enumerate() {
            let want =
                (f64::from(v) - mean) / (var + 1e-6).sqrt() * f64::from(ws[i]) + f64::from(bs[i]);
            assert!(
                (f64::from(g) - want).abs() <= want.abs() / 128.0 + 1e-2,
                "{g} vs {want}"
            );
        }
    }
}
