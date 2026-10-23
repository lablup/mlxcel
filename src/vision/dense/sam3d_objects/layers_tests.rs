// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Numerical contracts that distinguish guidance bias and norm semantics.
use super::*;

#[test]
fn unconditional_cross_attention_retains_value_and_output_bias() -> Result<()> {
    let mut weights = WeightMap::new();
    for (name, values, shape) in [
        ("a.to_q.weight", vec![1., 0., 0., 1.], vec![2, 2]),
        ("a.to_kv.weight", vec![0.; 8], vec![4, 2]),
        ("a.to_kv.bias", vec![19., -12., 2., -3.], vec![4]),
        ("a.to_out.weight", vec![2., 1., -1., 3.], vec![2, 2]),
        ("a.to_out.bias", vec![5., 7.], vec![2]),
    ] {
        weights.insert(name.into(), mx::from_slice_f32(&values, &shape));
    }
    let attention = Attention::from_weights(&weights, "a", 1, true, false)?;
    let condition = attention.prepare(None)?;
    let x = mx::zeros(&[1, 3, 2], mx::dtype::FLOAT32);
    let actual = attention.forward(&x, Some(&condition))?;
    assert_eq!(mx::array_shape(&actual), [1, 3, 2]);
    assert_eq!(
        mx::utils::array_to_vec_f32(&actual),
        [6., -4., 6., -4., 6., -4.]
    );
    Ok(())
}

#[test]
fn head_norm_keeps_bf16_and_finite_zero_heads() -> Result<()> {
    let x = mx::astype(
        &mx::from_slice_f32(&[0., 0., 3., 4.], &[1, 1, 2, 2]),
        mx::dtype::BFLOAT16,
    );
    let gamma = mx::full_f32(&[2, 2], 1., mx::dtype::BFLOAT16);
    let actual = mx::compiled_objects_head_norm(&x, &gamma)?;
    assert_eq!(mx::array_dtype(&actual), mx::dtype::BFLOAT16);
    let values = mx::utils::array_to_vec_f32(&actual);
    assert_eq!(&values[..2], &[0., 0.]);
    assert!((values[2] - 0.6 * 2_f32.sqrt()).abs() < 0.004);
    assert!((values[3] - 0.8 * 2_f32.sqrt()).abs() < 0.004);
    Ok(())
}

#[test]
fn compiled_layers_reject_invalid_shapes_and_dtypes() {
    let x = mx::zeros(&[1, 2, 4], mx::dtype::BFLOAT16);
    let wrong_width = mx::zeros(&[1, 1, 3], mx::dtype::BFLOAT16);
    let wrong_dtype = mx::zeros(&[1, 1, 4], mx::dtype::FLOAT32);
    assert!(mx::compiled_objects_modulate(&x, &wrong_width, &x).is_err());
    assert!(mx::compiled_objects_modulate(&x, &wrong_dtype, &x).is_err());
    assert!(mx::compiled_objects_gated_residual(&x, &wrong_width, &x).is_err());
    assert!(mx::compiled_objects_head_norm(&x, &wrong_width).is_err());
}

#[test]
fn position_embedding_preserves_axis_order_and_zero_padding() {
    let coords = mx::from_slice_i32(&[0, 0, 0], &[1, 3]);
    let actual = position_embedding(&coords, 14);
    assert_eq!(mx::array_shape(&actual), [1, 14]);
    assert_eq!(
        mx::utils::array_to_vec_f32(&actual),
        [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.]
    );
}

fn layer_oracle() -> Result<WeightMap> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sam3d_objects/layers.safetensors");
    mx::weights::load_safetensors(path).map_err(anyhow::Error::msg)
}
fn exact_bf16(actual: &MlxArray, expected: &MlxArray, name: &str) {
    assert_eq!(mx::array_dtype(actual), mx::dtype::BFLOAT16, "{name}");
    assert_eq!(mx::array_dtype(expected), mx::dtype::BFLOAT16, "{name}");
    assert_eq!(mx::array_shape(actual), mx::array_shape(expected), "{name}");
    let actual = mx::utils::array_to_vec_f32(actual);
    let expected = mx::utils::array_to_vec_f32(expected);
    assert!(actual.iter().all(|x| x.is_finite()), "{name}");
    assert_eq!(actual, expected, "{name} differs from pinned upstream BF16");
}

#[test]
fn compiled_layers_match_pinned_bf16_oracle() -> Result<()> {
    let w = layer_oracle()?;
    // The fixture records equivalence for these inputs; it does not establish
    // that eager and compiled operations always agree. Test cached calls too.
    for _ in 0..2 {
        exact_bf16(
            &*mx::compiled_objects_modulate(
                &w["modulate.x"],
                &w["modulate.scale"],
                &w["modulate.shift"],
            )?,
            &w["modulate.out"],
            "modulate",
        );
        exact_bf16(
            &*mx::compiled_objects_gated_residual(
                &w["residual.x"],
                &w["residual.h"],
                &w["residual.gate"],
            )?,
            &w["residual.out"],
            "gated residual",
        );
        exact_bf16(
            &*mx::compiled_objects_head_norm(&w["head_norm.x"], &w["head_norm.gamma"])?,
            &w["head_norm.out"],
            "head norm",
        );
    }
    Ok(())
}

#[test]
fn compiled_objects_gelu_matches_pinned_weak_scalar_oracle() -> Result<()> {
    let w = layer_oracle()?;
    let actual = mx::compiled_objects_gelu(&w["gelu.x"])?;
    exact_bf16(&actual, &w["gelu.out"], "weak scalar erf GELU");
    let promoted = mx::astype(&w["gelu.f32_constants"], mx::dtype::BFLOAT16);
    let promoted = mx::utils::array_to_vec_f32(&promoted);
    let expected = mx::utils::array_to_vec_f32(&w["gelu.out"]);
    assert_eq!(
        promoted
            .iter()
            .zip(expected)
            .filter(|(a, b)| **a != *b)
            .count(),
        3857
    );
    assert!(mx::compiled_objects_gelu(&mx::zeros(&[2], mx::dtype::INT32)).is_err());
    Ok(())
}

#[test]
fn head_norm_constructor_rejects_malformed_gamma_rank_and_shape() -> Result<()> {
    let mut w = WeightMap::new();
    w.insert("norm.gamma".into(), mx::zeros(&[2, 4], mx::dtype::BFLOAT16));
    assert!(HeadNorm::from_weights(&w, "norm", 2, 4).is_ok());
    for shape in [vec![], vec![8], vec![1, 2, 4], vec![4, 2], vec![2, 3]] {
        w.insert("norm.gamma".into(), mx::zeros(&shape, mx::dtype::BFLOAT16));
        let error = HeadNorm::from_weights(&w, "norm", 2, 4)
            .err()
            .ok_or_else(|| anyhow::anyhow!("accepted invalid gamma shape {shape:?}"))?;
        assert!(error.to_string().contains("norm.gamma"), "{error}");
    }
    assert!(HeadNorm::from_weights(&w, "norm", 0, 4).is_err());
    assert!(HeadNorm::from_weights(&w, "norm", 2, usize::MAX).is_err());
    Ok(())
}

fn tanh_oracle() -> Result<WeightMap> {
    use sha2::{Digest, Sha256};
    let (name, digest) = if mx::default_device_is_gpu() {
        (
            "gelu-tanh-power.safetensors",
            "ee4aed043dcbd10ad487b082e7a2ef7d00ae1745b5232a5f1945c92921c8a1ef",
        )
    } else {
        (
            "gelu-tanh-power-cpu.safetensors",
            "fe493ef267bdb06a95e630c694f0dc9be456a34fbc1e2f54879d3c91210685e2",
        )
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sam3d_objects")
        .join(name);
    let bytes = std::fs::read(&path)?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        digest,
        "authenticated tanh GELU fixture changed"
    );
    mx::weights::load_safetensors(path).map_err(anyhow::Error::msg)
}

#[test]
fn compiled_objects_gelu_tanh_matches_authentic_power_oracle() -> Result<()> {
    let w = tanh_oracle()?;
    let gpu = mx::default_device_is_gpu();
    let f32_differing = if gpu { 55 } else { 1 };
    for (prefix, differing) in [("probe", 4), ("bf16_grid", 32), ("f32_grid", f32_differing)] {
        // Transcendentals differ between native backends. Only F32 requires
        // a hardware-specific oracle; all eight BF16 arrays match CPU/GPU.
        if prefix == "f32_grid" && gpu && mx::rocm_arch::device_gfx_target() != Some("gfx1151") {
            eprintln!(
                "Skipping only F32 tanh GELU oracle: authenticated GPU capture requires gfx1151"
            );
            continue;
        }
        let x = &w[&format!("{prefix}_x")];
        let expected = &w[&format!("{prefix}_upstream")];
        let power = &w[&format!("{prefix}_power")];
        let old = &w[&format!("{prefix}_multiply")];
        let actual = mx::compiled_objects_gelu_tanh(x)?;
        assert_eq!(mx::array_dtype(&actual), mx::array_dtype(x), "{prefix}");
        assert_eq!(
            mx::array_shape(&actual),
            mx::array_shape(expected),
            "{prefix}"
        );
        assert_eq!(
            mx::array_to_raw_bytes(&actual),
            mx::array_to_raw_bytes(expected),
            "{prefix}"
        );
        assert_eq!(
            mx::array_to_raw_bytes(power),
            mx::array_to_raw_bytes(expected),
            "{prefix}"
        );
        let expected_values = mx::utils::array_to_vec_f32(expected);
        let old_values = mx::utils::array_to_vec_f32(old);
        assert_eq!(
            expected_values
                .iter()
                .zip(&old_values)
                .filter(|(a, b)| a != b)
                .count(),
            differing,
            "{prefix} must distinguish the previous graph"
        );
        // Check the actual replaced native path against its captured control.
        let one = mx::full_f32(&[], 1.0, mx::array_dtype(x));
        let old_native = mx::compiled_geglu_approx_activation(x, &one);
        assert_eq!(
            mx::array_to_raw_bytes(&old_native),
            mx::array_to_raw_bytes(old),
            "{prefix} old-graph control"
        );
        // Exercise a different shape with the same cached shapeless function.
        let reshaped = mx::reshape(x, &[2, -1]);
        let again = mx::compiled_objects_gelu_tanh(&reshaped)?;
        assert_eq!(
            mx::array_to_raw_bytes(&again),
            mx::array_to_raw_bytes(expected),
            "{prefix} cached shape"
        );
    }
    assert!(mx::compiled_objects_gelu_tanh(&mx::zeros(&[2], mx::dtype::INT32)).is_err());
    Ok(())
}

#[test]
fn objects_feed_forward_routes_through_power_gelu() -> Result<()> {
    let oracle = tanh_oracle()?;
    let mut weights = WeightMap::new();
    let identity: Vec<f32> = (0..16)
        .map(|i| if i / 4 == i % 4 { 1.0 } else { 0.0 })
        .collect();
    for index in [0, 2] {
        weights.insert(
            format!("f.mlp.{index}.weight"),
            mx::astype(&mx::from_slice_f32(&identity, &[4, 4]), mx::dtype::BFLOAT16),
        );
        weights.insert(
            format!("f.mlp.{index}.bias"),
            mx::zeros(&[4], mx::dtype::BFLOAT16),
        );
    }
    let layer = FeedForward::from_weights(&weights, "f")?;
    let input = mx::reshape(&oracle["probe_x"], &[1, 4]);
    let expected = mx::reshape(&oracle["probe_upstream"], &[1, 4]);
    exact_bf16(
        &*layer.forward(&input)?,
        &expected,
        "feed-forward power GELU routing",
    );
    Ok(())
}

/// Numerical coverage of the multi-wave reduction, not a deterministic race
/// reproducer: the original kernel also passed the bounded external probe.
#[test]
fn layer_norm_multiwave_mean_snapshot_is_stable() {
    for width in [32, 128, 512, 1024] {
        let values: Vec<f32> = (0..1024 * width)
            .map(|i| if i % 2 == 0 { 7.0 } else { 9.0 })
            .collect();
        for (dtype, tolerance) in [
            (mx::dtype::FLOAT32, 1e-6),
            (mx::dtype::FLOAT16, 1e-3),
            (mx::dtype::BFLOAT16, 8e-3),
        ] {
            let mut previous = None;
            for _ in 0..4 {
                let x = mx::astype(&mx::from_slice_f32(&values, &[1024, width]), dtype);
                // SAFETY: null affine pointers request the supported no-affine path.
                let y =
                    unsafe { mx::fast_layer_norm(&x, std::ptr::null(), std::ptr::null(), 1e-5) };
                let actual = mx::utils::array_to_vec_f32(&mx::astype(&y, mx::dtype::FLOAT32));
                for (i, &value) in actual.iter().enumerate() {
                    let expected = if i % 2 == 0 { -1.0 } else { 1.0 } / (1.0_f32 + 1e-5).sqrt();
                    assert!(
                        (value - expected).abs() < tolerance,
                        "width={width}, index={i}, value={value}"
                    );
                }
                let bits: Vec<_> = actual.iter().map(|v| v.to_bits()).collect();
                if let Some(previous) = &previous {
                    assert_eq!(&bits, previous, "unstable layer norm at width={width}");
                }
                previous = Some(bits);
            }
        }
    }
}
