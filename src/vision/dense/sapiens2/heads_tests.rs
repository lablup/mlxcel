// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn wrapping_bf16_values(length: usize, seed: u32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let input = (index as u32).wrapping_add(seed);
            let hash = input.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let bits = (0x3800 + ((hash >> 8) % 2048)) | ((hash & 1) << 15);
            f32::from_bits(bits << 16)
        })
        .collect()
}

#[test]
fn sapiens2_bf16_instance_norm_matches_upstream_fused_path() {
    let input = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &[
                -0.447265625,
                0.84375,
                1.140625,
                -0.08935546875,
                0.80078125,
                -0.1640625,
                0.267578125,
                -1.3125,
            ],
            &[1, 2, 2, 2],
        ),
        mlxcel_core::dtype::BFLOAT16,
    );
    let output = instance_norm(&input);
    assert_eq!(
        mlxcel_core::utils::array_to_vec_f32(&output),
        vec![
            -1.484375,
            1.3359375,
            1.1640625,
            0.119140625,
            0.6015625,
            0.0216064453125,
            -0.2890625,
            -1.4765625,
        ]
    );
}

#[test]
fn sapiens2_bf16_deconv4_matches_upstream_gemm_path() {
    let input = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &[
                -0.447265625,
                0.84375,
                1.140625,
                -0.08935546875,
                0.80078125,
                -0.1640625,
                0.267578125,
                -1.3125,
            ],
            &[1, 2, 2, 2],
        ),
        mlxcel_core::dtype::BFLOAT16,
    );
    let weight = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &[
                0.76171875,
                -0.73046875,
                0.97265625,
                -0.57421875,
                1.015625,
                -1.8203125,
                2.765625,
                0.6484375,
                1.3671875,
                -0.70703125,
                0.82421875,
                -0.67578125,
                0.240234375,
                0.38671875,
                -0.50390625,
                0.06201171875,
                0.345703125,
                -0.408203125,
                0.1015625,
                -0.546875,
                0.142578125,
                -0.62890625,
                0.5625,
                -0.07763671875,
                0.86328125,
                -0.7109375,
                2.40625,
                0.5,
                1.0546875,
                1.453125,
                1.1796875,
                -0.63671875,
                -0.1826171875,
                0.8203125,
                -0.310546875,
                0.055908203125,
                2.515625,
                1.171875,
                -0.65234375,
                0.28515625,
                -0.2275390625,
                0.703125,
                1.53125,
                0.4453125,
                -0.484375,
                0.9921875,
                -0.62890625,
                -0.6328125,
                -0.6953125,
                -1.8203125,
                -0.62890625,
                -4.0,
                -1.5078125,
                -0.4921875,
                -0.5,
                -0.73828125,
                1.6953125,
                0.328125,
                1.40625,
                -0.326171875,
                0.462890625,
                -0.173828125,
                0.0027923583984375,
                0.349609375,
            ],
            &[2, 4, 4, 2],
        ),
        mlxcel_core::dtype::BFLOAT16,
    );
    let gemm_weight = super::deconv::gemm_weight(&weight);
    let output = super::deconv::forward(&input, &gemm_weight, 2);
    assert_eq!(
        mlxcel_core::utils::array_to_vec_f32(&output),
        vec![
            -0.9375,
            -0.30859375,
            1.84375,
            0.73046875,
            1.28125,
            1.453125,
            0.2392578125,
            -0.640625,
            0.3671875,
            -3.359375,
            2.109375,
            0.326171875,
            2.96875,
            -1.484375,
            2.875,
            -2.546875,
            0.11669921875,
            0.2490234375,
            3.21875,
            0.016357421875,
            2.328125,
            1.3515625,
            0.62890625,
            -0.88671875,
            0.1708984375,
            0.15234375,
            0.84375,
            1.078125,
            1.2109375,
            4.8125,
            0.86328125,
            0.2421875,
        ]
    );
}

#[cfg(feature = "rocm")]
#[test]
fn sapiens2_bf16_deconv4_rocm_layout_matches_upstream_graph() {
    let _device_lock = mlxcel_core::streams::lock_default_device();
    if !mlxcel_core::default_device_is_gpu()
        || mlxcel_core::rocm_arch::device_gfx_target() != Some("gfx1151")
    {
        eprintln!("skipping Sapiens2 BF16 deconv layout oracle: requires the gfx1151 GPU");
        return;
    }

    // Exact MLX 0.32.5 / bff10220 ROCm oracle. The former value-identical
    // slice/concatenate weight graph produces 21.5 at this element because its
    // layout selects a different large-GEMM accumulation path.
    const CHANNELS: usize = 256;
    let weight = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &wrapping_bf16_values(CHANNELS * 4 * 4 * CHANNELS, 2371),
            &[CHANNELS as i32, 4, 4, CHANNELS as i32],
        ),
        mlxcel_core::dtype::BFLOAT16,
    );
    let columns = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &wrapping_bf16_values(9 * 4 * CHANNELS, 991),
            &[9, (4 * CHANNELS) as i32],
        ),
        mlxcel_core::dtype::BFLOAT16,
    );
    let output = mlxcel_core::matmul(&columns, &super::deconv::gemm_weight(&weight));
    let actual = mlxcel_core::utils::array_to_vec_f32(&output);
    assert_eq!(actual[4 * 4 * CHANNELS + 272], 21.375);
}

#[cfg(feature = "rocm")]
#[test]
fn sapiens2_bf16_silu_rocm_layout_matches_upstream_graph() {
    let _device_lock = mlxcel_core::streams::lock_default_device();
    if !mlxcel_core::default_device_is_gpu()
        || mlxcel_core::rocm_arch::device_gfx_target() != Some("gfx1151")
    {
        eprintln!("skipping Sapiens2 BF16 SiLU layout oracle: requires the gfx1151 GPU");
        return;
    }

    const CHANNELS: usize = 256;
    let channels_first = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &wrapping_bf16_values(256 * CHANNELS, 2371),
            &[1, CHANNELS as i32, 16, 16],
        ),
        mlxcel_core::dtype::BFLOAT16,
    );
    let input = mlxcel_core::transpose_axes(&channels_first, &[0, 2, 3, 1]);
    let weight = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &wrapping_bf16_values(CHANNELS * CHANNELS, 991),
            &[CHANNELS as i32, CHANNELS as i32],
        ),
        mlxcel_core::dtype::BFLOAT16,
    );
    let weight = mlxcel_core::transpose(&weight);
    let bias = mlxcel_core::zeros(&[CHANNELS as i32], mlxcel_core::dtype::BFLOAT16);

    let compiled = mlxcel_core::utils::silu(&input);
    let compiled = mlxcel_core::reshape(&compiled, &[256, CHANNELS as i32]);
    let compiled = mlxcel_core::addmm(&bias, &compiled, &weight, 1.0, 1.0);
    let compiled = mlxcel_core::utils::array_to_vec_f32(&compiled);
    assert_eq!(compiled[33 * CHANNELS + 12], 0.046142578125);

    let eager = mlxcel_core::multiply(&input, &mlxcel_core::sigmoid(&input));
    let eager = mlxcel_core::reshape(&eager, &[256, CHANNELS as i32]);
    let eager = mlxcel_core::addmm(&bias, &eager, &weight, 1.0, 1.0);
    let eager = mlxcel_core::utils::array_to_vec_f32(&eager);
    assert_eq!(eager[33 * CHANNELS + 12], 0.0458984375);
}

impl Sapiens2Head {
    pub(crate) fn all_stage_metrics(
        &self,
        normalized_tokens: &MlxArray,
        features: &MlxArray,
        reference_path: &std::path::Path,
    ) -> Vec<super::super::diagnostic::StageMetric> {
        match self {
            Self::Deconv(head) => {
                head.all_stage_metrics(normalized_tokens, features, reference_path)
            }
            Self::PixelShuffle(_) => {
                panic!("the all-stage diagnostic currently requires a pose/seg deconvolution head")
            }
        }
    }
}

impl DeconvHead {
    fn all_stage_metrics(
        &self,
        normalized_tokens: &MlxArray,
        features: &MlxArray,
        reference_path: &std::path::Path,
    ) -> Vec<super::super::diagnostic::StageMetric> {
        let reference = mlxcel_core::weights::load_safetensors(reference_path)
            .expect("load exact-pin Sapiens2 head reference");
        let mut expected_keys = std::collections::BTreeSet::from([
            "norm".to_owned(),
            "features".to_owned(),
            "heatmaps".to_owned(),
        ]);
        expected_keys
            .extend((0..self.deconvs.len() * 3).map(|index| format!("deconv_layers_{index}")));
        expected_keys.extend((0..self.convs.len() * 3).map(|index| format!("conv_layers_{index}")));
        let reference_keys: std::collections::BTreeSet<_> = reference.keys().cloned().collect();
        assert_eq!(
            reference_keys, expected_keys,
            "unexpected exact-pin head reference schema"
        );
        let mut metrics = Vec::new();
        for (name, actual) in [("norm", normalized_tokens), ("features", features)] {
            let (max_abs, relative_rms) =
                super::super::diagnostic::parity_metrics(actual, &reference[name]);
            metrics.push((format!("head.{name}"), max_abs, relative_rms));
        }

        let mut hidden = mlxcel_core::copy(features);
        for (index, layer) in self.deconvs.iter().enumerate() {
            let raw = if let Some(weight) = &layer.gemm_weight {
                let output_channels = mlxcel_core::array_shape(&layer.weight)[0];
                super::deconv::forward(&hidden, weight, output_channels)
            } else {
                let (padding, output_padding) = match layer.kernel {
                    3 => (1, 1),
                    2 => (0, 0),
                    _ => unreachable!("validated in constructor"),
                };
                mlxcel_core::conv_transpose2d(
                    &hidden,
                    &layer.weight,
                    2,
                    2,
                    padding,
                    padding,
                    1,
                    1,
                    output_padding,
                    output_padding,
                    1,
                )
            };
            let normalized = instance_norm(&raw);
            let activated = mlxcel_core::utils::silu(&normalized);
            for (offset, actual) in [&raw, &normalized, &activated].into_iter().enumerate() {
                let name = format!("deconv_layers_{}", index * 3 + offset);
                let (max_abs, relative_rms) =
                    super::super::diagnostic::parity_metrics(actual, &reference[name.as_str()]);
                metrics.push((format!("head.{name}"), max_abs, relative_rms));
            }
            hidden = activated;
        }
        for (index, layer) in self.convs.iter().enumerate() {
            let raw = layer.0.forward(&hidden);
            let normalized = instance_norm(&raw);
            let activated = mlxcel_core::utils::silu(&normalized);
            for (offset, actual) in [&raw, &normalized, &activated].into_iter().enumerate() {
                let name = format!("conv_layers_{}", index * 3 + offset);
                let (max_abs, relative_rms) =
                    super::super::diagnostic::parity_metrics(actual, &reference[name.as_str()]);
                metrics.push((format!("head.{name}"), max_abs, relative_rms));
            }
            hidden = activated;
        }
        // Compare the captured final tensor with an independent production
        // head call; the manual graph above is only for layer localization.
        let heatmaps = self.forward(features);
        let (max_abs, relative_rms) =
            super::super::diagnostic::parity_metrics(&heatmaps, &reference["heatmaps"]);
        metrics.push(("head.heatmaps".into(), max_abs, relative_rms));
        metrics
    }
}
