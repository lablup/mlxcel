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

//! The VAE against upstream `QwenImageVAE`: a tiny residual config
//! (Qwen-Image's layout) and a tiny plain config (Ming-Image's) whose
//! weights the fixture stores in Diffusers' torch layout, decoding a fixed
//! latent and encoding a fixed image; (ignored) the real checkpoint's VAE on
//! a fixed latent and image; and the row-banded convolution against the
//! plain one.

use super::*;
use crate::imagegen::qwen_image::parity_tests::{model_dir, tiny_refs};
use crate::imagegen::testing::{det, rel_rms, values};

fn tiny_config(is_residual: bool) -> QwenImageVaeConfig {
    QwenImageVaeConfig {
        base_dim: 4,
        decoder_base_dim: 6,
        z_dim: 4,
        dim_mult: vec![1, 2, 2],
        num_res_blocks: 1,
        temperal_downsample: vec![false, true],
        in_channels: 4,
        out_channels: 4,
        is_residual,
        scaling_factor: None,
        shift_factor: None,
        // Identity normalization: `decode_normalized` is then `decode`.
        latents_mean: vec![0.0; 4],
        latents_std: vec![1.0; 4],
    }
}

fn tiny_vae(refs: &WeightMap, tag: &str, mlx_layout: bool) -> QwenImageVae {
    let prefix = format!("vae_{tag}.w.");
    let weights: WeightMap = refs
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(&prefix).map(|k| {
                let v = if mlx_layout
                    && mlxcel_core::array_shape(v).len() == 4
                    && !k.ends_with(".gamma")
                {
                    mlxcel_core::transpose_axes(v, &[0, 2, 3, 1])
                } else {
                    mlxcel_core::copy(v)
                };
                (k.to_string(), v)
            })
        })
        .collect();
    QwenImageVae::new(tiny_config(tag == "res"), weights, true).expect("tiny VAE builds")
}

/// NCTHW with one frame to NHWC.
fn nhwc(a: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(a);
    mlxcel_core::transpose_axes(
        &mlxcel_core::reshape(a, &[s[0], s[1], s[3], s[4]]),
        &[0, 2, 3, 1],
    )
}

/// NCTHW with one frame to NCHW.
fn nchw(a: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(a);
    mlxcel_core::reshape(a, &[s[0], s[1], s[3], s[4]])
}

fn assert_close(name: &str, ours: &MlxArray, theirs: &MlxArray, tol: f64) {
    assert_eq!(
        mlxcel_core::array_shape(ours),
        mlxcel_core::array_shape(theirs),
        "{name}"
    );
    let rel = rel_rms(&values(ours), &values(theirs));
    eprintln!("{name}: rel-RMS {rel:.3e}");
    assert!(rel < tol, "{name}: rel-RMS {rel:.3e}");
}

#[test]
fn tiny_residual_and_plain_vaes_match_upstream() {
    let refs = tiny_refs();
    for tag in ["res", "plain"] {
        for mlx_layout in [false, true] {
            let vae = tiny_vae(&refs, tag, mlx_layout);
            let decoded = vae.decode_normalized(&nchw(&refs["vae.latent"])).unwrap();
            assert_close(
                &format!("{tag} decode"),
                &decoded,
                &nhwc(&refs[&format!("vae_{tag}.decode")]),
                1e-4,
            );
            let mean = vae.encode_normalized(&nhwc(&refs["vae.image"])).unwrap();
            assert_close(
                &format!("{tag} encode"),
                &mean,
                &nchw(&refs[&format!("vae_{tag}.encode_mean")]),
                1e-4,
            );
        }
    }
}

#[test]
fn decoding_without_the_encoder_works_and_encoding_errors() {
    let refs = tiny_refs();
    let weights: WeightMap = refs
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("vae_res.w.")
                .filter(|k| k.starts_with("decoder.") || k.starts_with("post_quant_conv."))
                .map(|k| (k.to_string(), mlxcel_core::copy(v)))
        })
        .collect();
    let vae = QwenImageVae::new(tiny_config(true), weights, false).unwrap();
    vae.decode_normalized(&nchw(&refs["vae.latent"])).unwrap();
    assert!(vae.encode_normalized(&nhwc(&refs["vae.image"])).is_err());
}

#[test]
fn banded_convolution_equals_the_plain_one() {
    let mut w = WeightMap::new();
    w.insert(
        "c.weight".into(),
        det("qiv.band_w", &[6, 3, 3, 5], 0.2, 0.0),
    );
    w.insert("c.bias".into(), det("qiv.band_b", &[6], 0.1, 0.0));
    let conv = Conv::load(&w, "c", 1, 1).unwrap();
    let x = det("qiv.band_x", &[1, 13, 7, 5], 1.0, 0.0);
    let plain = conv.forward_banded(&x, i64::MAX);
    // 7 * 6 elements per row: bands of one, two and five output rows.
    for limit in [42, 84, 210] {
        let banded = conv.forward_banded(&x, limit);
        let (a, b) = (values(&banded), values(&plain));
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        eprintln!("band limit {limit}: max abs {max}");
        assert!(max < 1e-5, "band limit {limit}: max abs {max}");
    }
}

#[test]
#[ignore = "loads the real Qwen-Image-2.1 VAE (GPU, ~1 GB)"]
fn real_vae_decodes_a_fixed_latent_like_upstream() {
    let root = model_dir().expect("models/mlx/qwen-image-2.1 is present");
    let refs = tiny_refs();
    let mut config = QwenImageVaeConfig::from_dir(&root.join("vae")).unwrap();
    // The fixture decodes a raw latent and keeps the raw posterior mean.
    config.latents_mean = vec![0.0; 64];
    config.latents_std = vec![1.0; 64];
    let vae = QwenImageVae::load(&root.join("vae"), config, true).unwrap();
    let decoded = vae
        .decode_normalized(&nchw(&det("qiv.real_latent", &[1, 64, 1, 4, 6], 1.0, 0.0)))
        .unwrap();
    let theirs = nhwc(&refs["vae_real.decode"]);
    let rel = rel_rms(&values(&decoded), &values(&theirs));
    eprintln!("real VAE decode rel-RMS vs upstream: {rel:.3e}");
    assert!(rel < 1e-3, "decode rel-RMS {rel:.3e}");
    let image = mlxcel_core::astype(
        &det("qiv.real_image", &[1, 4, 1, 64, 96], 1.0, 0.0),
        dtype::BFLOAT16,
    );
    let mean = vae.encode_normalized(&nhwc(&image)).unwrap();
    let rel = rel_rms(
        &values(&mean),
        &values(&nchw(&refs["vae_real.encode_mean"])),
    );
    eprintln!("real VAE encode rel-RMS vs upstream: {rel:.3e}");
    assert!(rel < 1e-3, "encode rel-RMS {rel:.3e}");
}
