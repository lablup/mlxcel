// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Objects focal recovery contracts, including the actual saved-input oracle.
use super::*;
use mlxcel_core as mx;
use sha2::{Digest, Sha256};

fn exact(actual: &MlxArray, expected: &MlxArray, label: &str) {
    assert_eq!(mx::array_dtype(actual), mx::dtype::FLOAT32, "{label}");
    assert_eq!(mx::array_dtype(expected), mx::dtype::FLOAT32, "{label}");
    assert_eq!(
        mx::array_shape(actual),
        mx::array_shape(expected),
        "{label}"
    );
    mx::eval(actual);
    mx::eval(expected);
    let actual = mx::array_to_raw_bytes(actual);
    let expected = mx::array_to_raw_bytes(expected);
    let difference = actual
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .position(|(a, b)| a != b);
    assert!(
        difference.is_none(),
        "{label}: first different F32 element {difference:?}"
    );
}

#[test]
fn objects_focal_solver_authentic_backend_oracle() -> Result<()> {
    // Reductions and fused arithmetic are backend-specific. CPU and gfx1151
    // fixtures use the same actual sampled inputs and exact per-step states.
    let gpu = mx::default_device_is_gpu();
    if gpu && mx::rocm_arch::device_gfx_target() != Some("gfx1151") {
        eprintln!("Skipping Objects focal GPU oracle: authenticated GPU capture requires gfx1151");
        return Ok(());
    }
    let load = |name: &str, digest: &str| -> Result<mx::weights::WeightMap> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sam3d_objects")
            .join(name);
        let bytes = std::fs::read(&path)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            digest,
            "authenticated {name} changed"
        );
        mx::weights::load_safetensors(path).map_err(anyhow::Error::msg)
    };
    let inputs = load(
        "focal-solver-inputs.safetensors",
        "d1be5971cd37960e5b5eb31b2fee95169af8e711cb82d0fcd4b5d3c6ec65ba9d",
    )?;
    let (states_file, states_hash) = if gpu {
        (
            "focal-solver-states.safetensors",
            "2c3c8afc0eb3c6a41687ceb2eed37ef510b69a5c694e5739fc6b9b86434f031b",
        )
    } else {
        (
            "focal-solver-states-cpu.safetensors",
            "cf5bf5c2eb1826242631177469e95ed2408872676c0d86c843283234f89bc3c5",
        )
    };
    let oracle = load(states_file, states_hash)?;
    let uv = &inputs["uv"];
    let xyz = &inputs["xyz"];
    assert_eq!(mx::array_shape(uv), [1, 4096, 2]);
    assert_eq!(mx::array_shape(xyz), [1, 4096, 3]);
    let states = mx::objects_focal_shift_states_for_test(uv, xyz)?;
    assert_eq!(mx::array_shape(&states), [31, 1, 7]);
    assert_eq!(mx::array_shape(&oracle["states"]), [31, 1, 7]);
    // Compare each state separately to report the first divergent iteration.
    for step in 0..31 {
        let actual = mx::slice(&states, &[step, 0, 0], &[step + 1, 1, 7]);
        let expected = mx::slice(&oracle["states"], &[step, 0, 0], &[step + 1, 1, 7]);
        exact(
            &actual,
            &expected,
            &format!("LM state {step}: shift,lambda,done,cost,g,h,f"),
        );
    }
    // Probe and production share arithmetic, but the production path must also
    // match without retaining or evaluating any diagnostic states.
    for _ in 0..2 {
        exact(
            &*mx::compiled_objects_focal_shift(uv, xyz)?,
            &oracle["focal_shift"],
            "final focal/shift",
        );
    }
    Ok(())
}

#[test]
fn objects_focal_wrapper_recovers_known_camera() -> Result<()> {
    let uv = normalized_view_plane_uv_with_policy(64, 64, None, Policy::Objects)?;
    let u = mx::slice(&uv, &[0, 0, 0], &[64, 64, 1]);
    let v = mx::slice(&uv, &[0, 0, 1], &[64, 64, 2]);
    let z = mx::add(
        &scalar(2.0),
        &mx::add(
            &mx::multiply(&u, &scalar(0.4)),
            &mx::multiply(&v, &scalar(0.3)),
        ),
    );
    let xy = mx::multiply(&uv, &mx::divide(&z, &scalar(1.7)));
    let affine = mx::expand_dims(
        &mx::concatenate(&xy, &mx::subtract(&z, &scalar(0.35)), -1),
        0,
    );
    let (focal, shift) = Policy::Objects.recover(&affine, None, None)?;
    assert!((mx::item_f32(&focal) - 1.7).abs() < 1e-4);
    assert!((mx::item_f32(&shift) - 0.35).abs() < 1e-4);
    Ok(())
}

#[test]
fn objects_focal_wrapper_preserves_degenerate_and_fixed_fallbacks() -> Result<()> {
    let points = mx::broadcast_to(
        &mx::from_slice_f32(&[0., 0., 1.], &[1, 1, 1, 3]),
        &[1, 64, 64, 3],
    );
    let fixed = mx::from_slice_f32(&[1.7], &[1]);
    for count in [0, 1] {
        let mut valid = vec![0; 4096];
        valid[0] = count;
        let mask = mx::astype(&mx::from_slice_i32(&valid, &[1, 64, 64]), mx::dtype::BOOL);
        for policy in [Policy::Standard, Policy::Objects] {
            let (focal, shift) = policy.recover(&points, Some(&mask), None)?;
            assert_eq!(mx::item_f32(&focal), 1.0);
            assert_eq!(mx::item_f32(&shift), 0.0);
            let (focal, shift) = policy.recover(&points, Some(&mask), Some(&fixed))?;
            exact(&focal, &fixed, "fixed focal preserved");
            assert_eq!(mx::item_f32(&shift), 0.0);
        }
    }
    Ok(())
}

#[test]
fn objects_focal_bridge_rejects_invalid_shapes_and_dtype() {
    let uv = mx::zeros(&[1, 4, 2], mx::dtype::FLOAT32);
    let xyz = mx::zeros(&[1, 4, 3], mx::dtype::FLOAT32);
    for wrong in [
        mx::zeros(&[4, 2], mx::dtype::FLOAT32),
        mx::zeros(&[2, 4, 2], mx::dtype::FLOAT32),
        mx::zeros(&[1, 3, 2], mx::dtype::FLOAT32),
        mx::zeros(&[1, 4, 3], mx::dtype::FLOAT32),
        mx::zeros(&[1, 4, 2], mx::dtype::FLOAT16),
        mx::zeros(&[1, 4, 2], mx::dtype::INT32),
        mx::zeros(&[1, 0, 2], mx::dtype::FLOAT32),
    ] {
        assert!(mx::compiled_objects_focal_shift(&wrong, &xyz).is_err());
        assert!(mx::objects_focal_shift_states_for_test(&wrong, &xyz).is_err());
    }
    for wrong in [
        mx::zeros(&[4, 3], mx::dtype::FLOAT32),
        mx::zeros(&[1, 4, 2], mx::dtype::FLOAT32),
        mx::zeros(&[1, 4, 3], mx::dtype::FLOAT16),
        mx::zeros(&[1, 4, 3], mx::dtype::INT32),
    ] {
        assert!(mx::compiled_objects_focal_shift(&uv, &wrong).is_err());
        assert!(mx::objects_focal_shift_states_for_test(&uv, &wrong).is_err());
    }
}

#[test]
fn objects_focal_bridge_rejects_overflowing_broadcast_metadata() {
    // Lazy views need no huge allocation; reject byte overflow before any solver op.
    let (batch, count) = if usize::BITS == 64 {
        (1 << 30, i32::MAX)
    } else {
        (1 << 15, 1 << 15)
    };
    let one = mx::from_slice_f32(&[1.], &[]);
    let uv = mx::broadcast_to(&one, &[batch, count, 2]);
    let xyz = mx::broadcast_to(&one, &[batch, count, 3]);
    for trace in [false, true] {
        let result = if trace {
            mx::objects_focal_shift_states_for_test(&uv, &xyz)
        } else {
            mx::compiled_objects_focal_shift(&uv, &xyz)
        };
        let error = result
            .err()
            .expect("overflow must precede solver operations");
        assert!(error.to_string().contains("byte size overflows size_t"));
    }
}
