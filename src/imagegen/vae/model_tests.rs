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

//! Parity of [`Flux2Vae`] against the upstream Python (`mlx_vlm`), using
//! fixtures generated once by `tests/fixtures/generate_imagegen_fixtures.py`:
//!
//! - `vae_tiny.safetensors`: a small random-weight VAE (4 latent channels,
//!   block channels 4/4/8/8, 2 norm groups), Diffusers key names and PyTorch
//!   conv layout, plus the upstream outputs for decode, packed decode and encode.
//! - `vae_ernie_refs.safetensors`: upstream f32 outputs of the real
//!   `baidu/ERNIE-Image-Turbo` VAE, used by the `#[ignore]` checkpoint test.

use std::path::PathBuf;

use super::*;
use mlxcel_core::weights::{WeightMap, load_safetensors};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/imagegen")
        .join(name)
}

fn tiny_config() -> Flux2VaeConfig {
    Flux2VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 4,
        block_out_channels: vec![4, 4, 8, 8],
        layers_per_block: 2,
        norm_num_groups: 2,
        ..Flux2VaeConfig::default()
    }
}

/// Weights and `ref.*` tensors of a fixture file, split apart.
fn load_fixture(name: &str) -> (WeightMap, WeightMap) {
    let mut all = load_safetensors(fixture(name)).expect("fixture loads");
    let refs_keys: Vec<String> = all
        .keys()
        .filter(|k| k.starts_with("ref."))
        .cloned()
        .collect();
    let mut refs = WeightMap::new();
    for key in refs_keys {
        let value = all.remove(&key).unwrap();
        refs.insert(key.trim_start_matches("ref.").to_string(), value);
    }
    (all, refs)
}

fn to_vec(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

/// `sqrt(mean((a - b)^2)) / sqrt(mean(b^2))`.
fn rel_rms(a: &MlxArray, b: &MlxArray) -> f32 {
    assert_eq!(
        mlxcel_core::array_shape(a),
        mlxcel_core::array_shape(b),
        "shape"
    );
    let (a, b) = (to_vec(a), to_vec(b));
    let num: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

fn tiny_vae(weights: WeightMap, include_encoder: bool, dt: i32) -> Result<Flux2Vae> {
    Flux2Vae::from_raw_weights(tiny_config(), weights, include_encoder, dt)
}

/// Re-lay the PyTorch fixture's conv weights as MLX `(out, kH, kW, in)`.
fn to_mlx_layout(weights: WeightMap) -> WeightMap {
    weights
        .into_iter()
        .map(|(k, v)| {
            let v = if mlxcel_core::array_shape(&v).len() == 4 {
                mlxcel_core::transpose_axes(&v, &[0, 2, 3, 1])
            } else {
                v
            };
            (k, v)
        })
        .collect()
}

const F32_TOL: f32 = 1e-4;

/// A smooth `[1, 3, n, n]` image in `[-1, 1]` (sinusoids), which a VAE can
/// reconstruct well; `n` a multiple of 8.
fn smooth_image(n: usize) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity(3 * n * n);
    for c in 0..3 {
        for y in 0..n {
            for x in 0..n {
                let (u, v) = (x as f32 / (n - 1) as f32, y as f32 / (n - 1) as f32);
                let value = match c {
                    0 => (3.0 * u + v).sin(),
                    1 => (2.0 * v - u).cos(),
                    _ => u * v * 2.0 - 0.5,
                };
                data.push(value.clamp(-1.0, 1.0));
            }
        }
    }
    mlxcel_core::from_slice_f32(&data, &[1, 3, n as i32, n as i32])
}

#[test]
fn tiny_decode_matches_upstream() {
    let (weights, refs) = load_fixture("vae_tiny.safetensors");
    let vae = tiny_vae(weights, false, dtype::FLOAT32).unwrap();
    let out = vae.decode(&refs["latents"]).unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), [1, 3, 32, 32]);
    let err = rel_rms(&out, &refs["decode"]);
    assert!(err < F32_TOL, "decode rel-RMS {err}");
}

#[test]
fn tiny_packed_decode_matches_upstream() {
    let (weights, refs) = load_fixture("vae_tiny.safetensors");
    let vae = tiny_vae(weights, false, dtype::FLOAT32).unwrap();
    let out = vae.decode_packed_latents(&refs["packed"]).unwrap();
    let err = rel_rms(&out, &refs["decode_packed"]);
    assert!(err < F32_TOL, "decode_packed_latents rel-RMS {err}");
}

#[test]
fn tiny_encode_matches_upstream() {
    let (weights, refs) = load_fixture("vae_tiny.safetensors");
    let vae = tiny_vae(weights, true, dtype::FLOAT32).unwrap();
    assert!(vae.has_encoder());
    let out = vae.encode(&refs["image"]).unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), [1, 4, 4, 4]);
    let err = rel_rms(&out, &refs["encode"]);
    assert!(err < F32_TOL, "encode rel-RMS {err}");
}

#[test]
fn already_converted_mlx_layout_is_not_permuted_twice() {
    let (weights, refs) = load_fixture("vae_tiny.safetensors");
    let vae = tiny_vae(to_mlx_layout(weights), true, dtype::FLOAT32).unwrap();
    let decode = vae.decode(&refs["latents"]).unwrap();
    assert!(rel_rms(&decode, &refs["decode"]) < F32_TOL);
    let encode = vae.encode(&refs["image"]).unwrap();
    assert!(rel_rms(&encode, &refs["encode"]) < F32_TOL);
}

#[test]
fn conv_layout_is_detected_from_an_unambiguous_tensor() {
    let config = tiny_config();
    let (weights, _) = load_fixture("vae_tiny.safetensors");
    assert_eq!(
        detect_conv_layout(&weights, &config).unwrap(),
        ConvLayout::Torch
    );
    assert_eq!(
        detect_conv_layout(&to_mlx_layout(weights), &config).unwrap(),
        ConvLayout::Mlx
    );
}

#[test]
fn an_unrecognizable_conv_shape_is_an_error() {
    let config = tiny_config();
    let mut weights = WeightMap::new();
    weights.insert(
        "decoder.conv_in.weight".to_string(),
        mlxcel_core::zeros(&[8, 5, 5, 4], dtype::FLOAT32),
    );
    let err = detect_conv_layout(&weights, &config)
        .unwrap_err()
        .to_string();
    assert!(err.contains("decoder.conv_in.weight"), "{err}");
}

#[test]
fn a_three_channel_latent_falls_back_to_the_conv_out_probe() {
    // latent_channels == kernel size makes decoder.conv_in ambiguous.
    let config = Flux2VaeConfig {
        latent_channels: 3,
        ..tiny_config()
    };
    let mut weights = WeightMap::new();
    weights.insert(
        "decoder.conv_in.weight".to_string(),
        mlxcel_core::zeros(&[8, 3, 3, 3], dtype::FLOAT32),
    );
    weights.insert(
        "decoder.conv_out.weight".to_string(),
        mlxcel_core::zeros(&[3, 3, 3, 4], dtype::FLOAT32),
    );
    assert_eq!(
        detect_conv_layout(&weights, &config).unwrap(),
        ConvLayout::Mlx
    );
    weights.insert(
        "decoder.conv_out.weight".to_string(),
        mlxcel_core::zeros(&[3, 4, 3, 3], dtype::FLOAT32),
    );
    assert_eq!(
        detect_conv_layout(&weights, &config).unwrap(),
        ConvLayout::Torch
    );
}

#[test]
fn only_vae_tensors_are_kept_and_counters_are_dropped() {
    let (weights, _) = load_fixture("vae_tiny.safetensors");
    assert!(weights.contains_key("bn.num_batches_tracked"));
    assert!(weights.contains_key("unused.weight"));
    let prepared = prepare_weights(weights, &tiny_config(), false, dtype::FLOAT32).unwrap();
    assert!(!prepared.contains_key("bn.num_batches_tracked"));
    assert!(!prepared.contains_key("unused.weight"));
    assert!(
        !prepared
            .keys()
            .any(|k| k.starts_with("encoder.") || k.starts_with("quant_conv."))
    );
    assert!(prepared.contains_key("decoder.conv_in.weight"));
    assert!(prepared.contains_key("bn.running_mean"));
}

#[test]
fn the_encoder_is_only_required_when_asked_for() {
    let (mut weights, refs) = load_fixture("vae_tiny.safetensors");
    weights.retain(|k, _| !k.starts_with("encoder.") && !k.starts_with("quant_conv."));
    let vae = tiny_vae(weights, false, dtype::FLOAT32).unwrap();
    assert!(!vae.has_encoder());
    let err = vae.encode(&refs["image"]).err().unwrap().to_string();
    assert!(err.contains("without encoder weights"), "{err}");

    let (mut weights, _) = load_fixture("vae_tiny.safetensors");
    weights.retain(|k, _| !k.starts_with("encoder."));
    let err = tiny_vae(weights, true, dtype::FLOAT32).err().unwrap();
    assert!(format!("{err:#}").contains("encoder."), "{err:#}");
}

#[test]
fn a_missing_weight_is_named() {
    let (mut weights, _) = load_fixture("vae_tiny.safetensors");
    weights.remove("decoder.mid_block.attentions.0.to_q.weight");
    let err = tiny_vae(weights, false, dtype::FLOAT32).err().unwrap();
    assert!(
        format!("{err:#}").contains("decoder.mid_block.attentions.0.to_q.weight"),
        "{err:#}"
    );
}

#[test]
fn a_vae_without_batch_norm_cannot_decode_packed_latents_but_can_decode() {
    let (mut weights, refs) = load_fixture("vae_tiny.safetensors");
    weights.retain(|k, _| !k.starts_with("bn."));
    let vae = tiny_vae(weights, false, dtype::FLOAT32).unwrap();
    assert!(vae.decode(&refs["latents"]).is_ok());
    let err = vae
        .decode_packed_latents(&refs["packed"])
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("bn.*"), "{err}");
}

#[test]
fn a_vae_without_post_quant_conv_skips_that_stage() {
    // Z-Image's FLUX.1 AutoencoderKL layout: no post_quant_conv, no bn.
    let (mut weights, refs) = load_fixture("vae_tiny.safetensors");
    weights.retain(|k, _| !k.starts_with("post_quant_conv.") && !k.starts_with("bn."));
    let vae = tiny_vae(weights, false, dtype::FLOAT32).unwrap();
    let out = vae.decode(&refs["latents"]).unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), [1, 3, 32, 32]);
    // The skipped 1x1 conv changes the result.
    assert!(rel_rms(&out, &refs["decode"]) > 1e-3);
}

#[test]
fn scaling_and_shift_follow_the_config() {
    let (weights, refs) = load_fixture("vae_tiny.safetensors");
    let plain = tiny_vae(weights, false, dtype::FLOAT32).unwrap();
    let (weights, _) = load_fixture("vae_tiny.safetensors");
    let scaled = Flux2Vae::from_raw_weights(
        Flux2VaeConfig {
            scaling_factor: 2.0,
            shift_factor: 0.5,
            ..tiny_config()
        },
        weights,
        false,
        dtype::FLOAT32,
    )
    .unwrap();
    // decode(x) with (scale 2, shift .5) == decode of x/2 + .5 with (1, 0).
    let half = mlxcel_core::divide(
        &refs["latents"],
        &mlxcel_core::full_f32(&[1], 2.0, dtype::FLOAT32),
    );
    let shifted = mlxcel_core::add(&half, &mlxcel_core::full_f32(&[1], 0.5, dtype::FLOAT32));
    let expected = plain.decode(&shifted).unwrap();
    let actual = scaled.decode(&refs["latents"]).unwrap();
    assert!(rel_rms(&actual, &expected) < 1e-6);
}

#[test]
fn a_singleton_frame_axis_is_accepted_everywhere() {
    let (weights, refs) = load_fixture("vae_tiny.safetensors");
    let vae = tiny_vae(weights, true, dtype::FLOAT32).unwrap();
    let framed = mlxcel_core::reshape(&refs["latents"], &[1, 4, 1, 4, 4]);
    assert!(rel_rms(&vae.decode(&framed).unwrap(), &refs["decode"]) < F32_TOL);
    let framed = mlxcel_core::reshape(&refs["image"], &[1, 3, 1, 32, 32]);
    assert!(rel_rms(&vae.encode(&framed).unwrap(), &refs["encode"]) < F32_TOL);
    assert!(
        vae.decode(&mlxcel_core::zeros(&[1, 4, 2, 4, 4], dtype::FLOAT32))
            .is_err()
    );
    assert!(
        vae.decode(&mlxcel_core::zeros(&[4, 4], dtype::FLOAT32))
            .is_err()
    );
}

#[test]
fn half_precision_decodes_stay_finite_and_close_to_f32() {
    let (_, refs) = load_fixture("vae_tiny.safetensors");
    for (dt, bound) in [(dtype::FLOAT16, 2e-2), (dtype::BFLOAT16, 8e-2)] {
        let (weights, _) = load_fixture("vae_tiny.safetensors");
        let vae = tiny_vae(weights, false, dt).unwrap();
        let latents = mlxcel_core::astype(&refs["latents"], dt);
        let out = vae.decode(&latents).unwrap();
        assert_eq!(
            mlxcel_core::array_dtype(&out),
            dt,
            "output keeps the compute dtype"
        );
        let out = mlxcel_core::astype(&out, dtype::FLOAT32);
        assert!(to_vec(&out).iter().all(|v| v.is_finite()), "dtype {dt}");
        let err = rel_rms(&out, &refs["decode"]);
        assert!(err < bound, "dtype {dt}: rel-RMS {err}");
    }
}

#[test]
fn the_default_dtype_follows_the_project_policy() {
    let dt = Flux2Vae::default_dtype();
    assert!(dt == dtype::FLOAT16 || dt == dtype::BFLOAT16);
}

/// The real checkpoint, against the upstream output for fixed inputs.
///
/// Needs `models/ERNIE-Image-Turbo/vae/` (`huggingface-cli download
/// baidu/ERNIE-Image-Turbo --include "vae/*" "model_index.json" --local-dir
/// models/ERNIE-Image-Turbo`); skips when it is absent.
#[test]
#[ignore = "needs models/ERNIE-Image-Turbo/vae (baidu/ERNIE-Image-Turbo)"]
fn ernie_image_turbo_vae_matches_upstream() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/ERNIE-Image-Turbo");
    if !model_dir
        .join("vae")
        .join("diffusion_pytorch_model.safetensors")
        .exists()
    {
        eprintln!("skipping: {} is not downloaded", model_dir.display());
        return;
    }
    let (_, refs) = load_fixture("vae_ernie_refs.safetensors");

    // Same arithmetic as upstream: structure, layout and glue must agree.
    //
    // The encode reference differs from stock upstream in one tensor: upstream's
    // `_match_conv_layout` returns a weight unchanged when its shape equals the
    // target, and the encoder's RGB `conv_in` is `(128, 3, 3, 3)` in both the
    // PyTorch and the MLX layout, so upstream keeps the PyTorch data order there
    // and its encoder is wrong (encode -> decode reconstructs with rel-RMS 0.96;
    // with the layout fixed, 0.096). The fixture is generated with that one
    // tensor transposed. `detect_conv_layout` decides the layout once per
    // checkpoint, so this build does not have the defect.
    let vae = Flux2Vae::load_with_dtype(&model_dir, true, dtype::FLOAT32).unwrap();
    let decode = vae.decode(&refs["latents"]).unwrap();
    assert_eq!(mlxcel_core::array_shape(&decode), [1, 3, 64, 64]);
    let err = rel_rms(&decode, &refs["f32.decode"]);
    eprintln!("f32 decode rel-RMS vs upstream = {err:e}");
    assert!(err < 5e-3, "f32 decode rel-RMS {err}");
    let packed = vae.decode_packed_latents(&refs["packed"]).unwrap();
    let err = rel_rms(&packed, &refs["f32.decode_packed"]);
    eprintln!("f32 decode_packed_latents rel-RMS vs upstream = {err:e}");
    assert!(err < 5e-3, "f32 decode_packed_latents rel-RMS {err}");
    let encode = vae.encode(&refs["image"]).unwrap();
    let err = rel_rms(&encode, &refs["f32.encode"]);
    eprintln!("f32 encode rel-RMS vs layout-fixed upstream = {err:e}");
    assert!(err < 5e-3, "f32 encode rel-RMS {err}");

    // Encode then decode must reconstruct a smooth image: this is what fails
    // when a conv layout is wrong, whatever a reference says. (The fixture's
    // noise image is incompressible, so it cannot serve here.)
    let smooth = smooth_image(64);
    let rebuilt = vae.decode(&vae.encode(&smooth).unwrap()).unwrap();
    let err = rel_rms(&rebuilt, &smooth);
    eprintln!("f32 encode->decode reconstruction of a smooth image rel-RMS = {err:e}");
    assert!(err < 0.25, "encode->decode reconstruction rel-RMS {err}");

    // The dtype this build uses by default must stay finite and close.
    let dt = Flux2Vae::default_dtype();
    let vae = Flux2Vae::load_with_dtype(&model_dir, false, dt).unwrap();
    let out = vae
        .decode(&mlxcel_core::astype(&refs["latents"], dt))
        .unwrap();
    let out = mlxcel_core::astype(&out, dtype::FLOAT32);
    assert!(
        to_vec(&out).iter().all(|v| v.is_finite()),
        "default dtype produced inf/NaN"
    );
    let err = rel_rms(&out, &refs["f32.decode"]);
    eprintln!("default dtype {dt}: decode rel-RMS vs upstream f32 = {err}");
    assert!(err < 5e-3, "default dtype decode rel-RMS {err}");
}
