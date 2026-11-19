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

use super::*;

#[test]
fn linspace_hits_both_endpoints() {
    let values = torch_linspace(1.0, 0.25, 10);
    assert_eq!(values.len(), 10);
    assert!((values[0] - 1.0).abs() < 1e-6, "{values:?}");
    assert!((values[9] - 0.25).abs() < 1e-6, "{values:?}");
}

#[test]
fn the_schedule_decreases_strictly_and_ends_at_zero() {
    let sigmas = sigmas_for(50, 5.0);
    assert_eq!(sigmas.last().copied(), Some(0.0));
    assert!(sigmas.len() > 2, "{sigmas:?}");
    for pair in sigmas.windows(2) {
        assert!(pair[0] > pair[1], "sigmas must decrease strictly: {pair:?}");
    }
}

#[test]
fn every_sigma_sits_on_the_training_grid() {
    let mut training = grid(TRAINING_STEPS, 1.0, TRAINING_STEPS);
    training.reverse();
    for sigma in sigmas_for(20, 5.0) {
        if sigma == 0.0 {
            continue;
        }
        assert!(
            training.iter().any(|t| (t - sigma).abs() < 1e-6),
            "sigma {sigma} is not on the training grid"
        );
    }
}

#[test]
fn timesteps_are_sigmas_scaled_by_the_grid() {
    let sigmas = vec![1.0, 0.5, 0.0];
    assert_eq!(timesteps_for(&sigmas), vec![1000.0, 500.0]);
}

#[test]
fn zero_steps_yield_an_empty_schedule_instead_of_underflowing() {
    assert!(torch_linspace(1.0, 0.0, 0).is_empty());
    assert_eq!(sigmas_for(0, 5.0), vec![0.0]);
}

/// The tiny FLUX.2 VAE fixture (`tests/fixtures/imagegen/vae_tiny.safetensors`,
/// 4 latent channels) without its encoder, and its upstream `packed` latents.
fn tiny_decoder_only_vae(
    keep_stats: bool,
) -> (crate::imagegen::vae::Flux2Vae, UniquePtr<MlxArray>) {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/imagegen/vae_tiny.safetensors");
    let mut weights = mlxcel_core::weights::load_safetensors(path).expect("fixture loads");
    let packed = weights
        .remove("ref.packed")
        .expect("fixture has packed latents");
    weights.retain(|k, _| {
        !k.starts_with("ref.") && !k.starts_with("encoder.") && !k.starts_with("quant_conv.")
    });
    // The MOSS export's spelling: bare statistics, or none at all.
    for name in ["running_mean", "running_var"] {
        let value = weights.remove(&format!("bn.{name}")).unwrap();
        if keep_stats {
            weights.insert(name.to_string(), value);
        }
    }
    let config = crate::imagegen::vae::Flux2VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 4,
        block_out_channels: vec![4, 4, 8, 8],
        layers_per_block: 2,
        norm_num_groups: 2,
        ..crate::imagegen::vae::Flux2VaeConfig::default()
    };
    let vae =
        crate::imagegen::vae::Flux2Vae::from_raw_weights(config, weights, false, dtype::FLOAT32)
            .expect("tiny VAE loads");
    (vae, packed)
}

#[test]
fn decoding_denormalizes_with_the_exported_statistics() {
    let (vae, packed) = tiny_decoder_only_vae(true);
    let pixels = decode_to_pixels(&vae, &packed).unwrap();
    let expected = vae.decode_packed_latents(&packed).unwrap();
    let shape = mlxcel_core::array_shape(&expected);
    assert_eq!(pixels.len(), (shape[2] * shape[3] * 3) as usize);
    let unit = mlxcel_core::clip(
        &mlxcel_core::add(
            &mlxcel_core::divide(&expected, &scalar_like(2.0, &expected)),
            &scalar_like(0.5, &expected),
        ),
        &scalar_like(0.0, &expected),
        &scalar_like(1.0, &expected),
    );
    let nhwc = mlxcel_core::transpose_axes(&unit, &[0, 2, 3, 1]);
    let scaled = mlxcel_core::multiply(&nhwc, &scalar_like(255.0, &nhwc));
    let bytes = mlxcel_core::astype(&mlxcel_core::round(&scaled), dtype::UINT8);
    assert_eq!(pixels, mlxcel_core::try_array_to_raw_bytes(&bytes).unwrap());
}

#[test]
fn a_decoder_without_batch_norm_statistics_is_refused() {
    let (vae, packed) = tiny_decoder_only_vae(false);
    let err = format!("{:#}", decode_to_pixels(&vae, &packed).unwrap_err());
    assert!(err.contains("bn.*"), "{err}");
}
