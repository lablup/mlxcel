// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Small authenticated MHR geometry and deterministic accumulation contracts.
use super::super::mhr_utils;
use super::*;
use sha2::{Digest, Sha256};

fn exact(actual: &MlxArray, expected: &MlxArray, name: &str) -> Result<()> {
    assert_eq!(mx::array_dtype(actual), mx::array_dtype(expected), "{name}");
    assert_eq!(mx::array_shape(actual), mx::array_shape(expected), "{name}");
    let a = mx::try_array_to_raw_bytes(actual)?;
    let b = mx::try_array_to_raw_bytes(expected)?;
    let mismatch = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .position(|(x, y)| x != y);
    assert!(
        mismatch.is_none(),
        "{name}: first different F32 element {mismatch:?}"
    );
    Ok(())
}
/// Fields whose values pass through CPU transcendentals (`atan2`, `sin`,
/// `cos`) and so carry the capture host's math library.
const LIBM_FIELDS: [&str; 4] = ["body_params", "hand_params", "fk_state", "fk_global_rot"];
/// Largest distance from the x86_64 Linux capture allowed in [`LIBM_FIELDS`]
/// on any other host.
///
/// The fixture was captured with MLX's CPU stream on x86_64 Linux (glibc
/// `std::atan2`/`sin`/`cos`). On aarch64 Linux, glibc's own implementations
/// of those functions round differently: on an NVIDIA GB10 (DGX Spark,
/// aarch64, glibc) 41 of the 266 `body_params` and 6 of the 54 `hand_params`
/// elements differ by exactly 1 ULP, and every other field, including
/// `fk_state` and `fk_global_rot`, matches bit for bit (issue #2739). On macOS, MLX's CPU backend evaluates the same
/// functions through Apple's math library (`asd::` vector math in
/// <https://github.com/ml-explore/mlx/blob/main/mlx/backend/cpu/simd/accelerate_simd.h>).
/// On an M1 Ultra, 27 of the 388 elements in these four fields differ by 1 or
/// 2 ULP and every other field matches bit for bit. Seven of those elements are
/// single `atan2` calls, and upstream `mlx` 0.32.4 `mx.arctan2` on the CPU
/// stream of that host returns exactly mlxcel's values for them (and differs
/// from the fixture at exactly those seven). The other 20 come from longer
/// rotation and FK chains and were not recaptured upstream on macOS. The bound
/// is the larger of the two measured aarch64 hosts; only x86_64 Linux, the
/// capture host, stays bit-exact.
const NON_CAPTURE_HOST_MAX_ULP: i64 = 2;

fn within_ulp(actual: &MlxArray, expected: &MlxArray, name: &str, max_ulp: i64) -> Result<()> {
    assert_eq!(mx::array_dtype(actual), mx::dtype::FLOAT32, "{name}");
    assert_eq!(mx::array_dtype(expected), mx::dtype::FLOAT32, "{name}");
    assert_eq!(mx::array_shape(actual), mx::array_shape(expected), "{name}");
    let a = mx::try_array_to_raw_bytes(actual)?;
    let b = mx::try_array_to_raw_bytes(expected)?;
    // Map IEEE bits onto a monotonic integer line so the difference counts ULP.
    let ordered = |bytes: &[u8]| {
        let bits = i32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
        i64::from(if bits < 0 { i32::MIN - bits } else { bits })
    };
    for (index, (x, y)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        let distance = (ordered(x) - ordered(y)).abs();
        assert!(
            distance <= max_ulp,
            "{name}[{index}] differs from the capture by {distance} ULP"
        );
    }
    Ok(())
}

/// Bit-exact on the capture host's math library, ULP-bounded elsewhere for
/// the transcendental fields only.
fn geometry(actual: &MlxArray, expected: &MlxArray, name: &str) -> Result<()> {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) || !LIBM_FIELDS.contains(&name) {
        exact(actual, expected, name)
    } else {
        within_ulp(actual, expected, name, NON_CAPTURE_HOST_MAX_ULP)
    }
}

#[test]
fn sam3d_body_mhr_authentic_cpu_geometry_oracle() -> Result<()> {
    // This fixture records CPU libm/reductions on x86_64 Linux. Run it on the
    // CPU even in a GPU build, and restore the caller's device when the test
    // finishes. See `NON_CAPTURE_HOST_MAX_ULP` for other hosts' math libraries.
    let _device_lock = mx::streams::lock_default_device();
    let _cpu_device = mx::streams::DefaultDeviceGuard::cpu();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sam3d_body/geometry.safetensors");
    assert_eq!(
        format!("{:x}", Sha256::digest(std::fs::read(&path)?)),
        "7ab6a64bb6f0459ad6687ce50253a9e0f3e627d78e975460384c1ce9eed33b26"
    );
    let w = mx::weights::load_safetensors(path).map_err(anyhow::Error::msg)?;
    for (name, actual) in [
        (
            "rot6d_matrix",
            mhr_utils::rot6d_to_rotmat(&w["rot6d_input"])?,
        ),
        (
            "euler_zyx",
            mhr_utils::rotmat_to_euler_zyx(&w["rot6d_matrix"])?,
        ),
        (
            "euler_xyz",
            mhr_utils::batch_xyz_from_6d(&w["rot6d_input"])?,
        ),
        (
            "quaternion_matrix",
            mhr_utils::quat_to_rotmat(&w["quaternions"])?,
        ),
        (
            "matrix_quaternion",
            mhr_utils::rotmat_to_quat(&w["quaternion_matrix"])?,
        ),
        ("body_params", mhr_utils::compact_body(&w["body_cont"])?),
        ("hand_params", mhr_utils::compact_hand(&w["hand_cont"])?),
    ] {
        geometry(&actual, &w[name], name)?;
    }
    let (_, parents) = weights::i32_weight(&w, "fk_parents", &[2])?;
    let fk = kinematics::forward(
        &w["fk_dofs"],
        &w["fk_offsets"],
        &w["fk_prerotations"],
        &parents,
    )?;
    for (name, actual) in [
        ("fk_state", fk.state),
        ("fk_global_pos", fk.position),
        ("fk_global_rot", fk.rotation),
        ("fk_global_scale", fk.scale),
    ] {
        geometry(&actual, &w[name], name)?;
    }
    let (_, indices) = weights::i32_weight(&w, "scatter_indices", &[5])?;
    let output = ordered_scatter(&mx::reshape(&w["scatter_values"], &[1, 5, 1]), &indices, 3)?;
    exact(
        &mx::reshape(&output, &[3]),
        &w["scatter_result"],
        "ordered scatter",
    )
}
#[test]
fn sam3d_body_mhr_scatter_preserves_order_batches_and_columns() -> Result<()> {
    let input = mx::from_slice_f32(
        &[1e8, 2., 1., 3., -1e8, 4., 1., -1e8, 2., 1., 3., 1e8],
        &[2, 3, 2],
    );
    let output = ordered_scatter(&input, &[0, 0, 0], 2)?;
    let expected = mx::from_slice_f32(&[0., 9., 0., 0., 6., 0., 0., 0.], &[2, 2, 2]);
    exact(&output, &expected, "ordered F32 accumulation")?;
    assert!(ordered_scatter(&input, &[0, -1, 0], 2).is_err());
    assert!(ordered_scatter(&input, &[0, 0, 2], 2).is_err());
    assert!(ordered_scatter(&input, &[0, 0], 2).is_err());
    assert!(ordered_scatter(&input, &[0, 0, 0], usize::MAX).is_err());
    Ok(())
}
#[test]
fn sam3d_body_mhr_rejects_invalid_rotations_and_topology() -> Result<()> {
    assert!(mhr_utils::rot6d_to_rotmat(&mx::zeros(&[2, 5], mx::dtype::FLOAT32)).is_err());
    assert!(mhr_utils::rot6d_to_rotmat(&mx::zeros(&[2, 6], mx::dtype::FLOAT16)).is_err());
    assert!(mhr_utils::rotmat_to_quat(&mx::zeros(&[3, 2], mx::dtype::FLOAT32)).is_err());
    assert!(mhr_utils::compact_body(&mx::zeros(&[260], mx::dtype::FLOAT32)).is_err());
    assert!(weights::parents_valid(&[-1, 1]).is_err());
    assert!(weights::parents_valid(&[-1, 2, 0]).is_err());
    assert!(weights::parents_valid(&[-2, 0]).is_err());
    assert!(weights::parents_valid(&[]).is_err());
    let mut w = WeightMap::new();
    w.insert("weight".into(), mx::from_slice_f32(&[f32::NAN], &[1]));
    assert!(f32_weight(&w, "weight", &[1]).is_err());
    w.insert("weight".into(), mx::from_slice_i32(&[0], &[1]));
    assert!(f32_weight(&w, "weight", &[1]).is_err());
    assert!(i32_weight(&w, "weight", &[2]).is_err());
    assert!(i32_weight(&w, "missing", &[1]).is_err());
    assert!(weights::reserve::<f32>(usize::MAX).is_err());
    Ok(())
}
#[test]
fn sam3d_body_mhr_half_turn_quaternion_branches() -> Result<()> {
    let q = mx::from_slice_f32(
        &[
            0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
        ],
        &[4, 4],
    );
    let r = mhr_utils::quat_to_rotmat(&q)?;
    let recovered = mhr_utils::rotmat_to_quat(&r)?;
    exact(&recovered, &q, "all Shepperd branches")
}

#[test]
fn sam3d_body_mhr_blend_fk_skinning_batches() -> Result<()> {
    // A three-joint chain with inverse bind transforms that cancel its rest
    // offsets. Two influences on vertex zero exercise weighted skinning.
    let f = |values: &[f32], shape: &[i32]| mx::from_slice_f32(values, shape);
    let z = |shape: &[i32]| mx::zeros(shape, mx::dtype::FLOAT32);
    let mut transform = vec![0.; 21 * 249];
    transform[0] = 1.;
    let mut shapes = vec![0.; 45 * 2 * 3];
    shapes[0] = 0.5;
    let model = MhrBody {
        joints: 3,
        vertices: 2,
        offsets: f(&[0., 0., 0., 1., 0., 0., 0., 2., 0.], &[3, 3]),
        prerotations: f(&[0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.], &[3, 4]),
        parents: vec![-1, 0, 1],
        transform: f(&transform, &[21, 249]),
        base: f(&[1., 2., 3., -2., 0., 1.], &[2, 3]),
        shapes: f(&shapes, &[45, 2, 3]),
        expressions: z(&[72, 2, 3]),
        inverse_bind: f(
            &[
                0., 0., 0., 0., 0., 0., 1., 1., -1., 0., 0., 0., 0., 0., 1., 1., -1., -2., 0., 0.,
                0., 0., 1., 1.,
            ],
            &[3, 8],
        ),
        skin_indices: mx::from_slice_i32(&[0, 1, 2], &[3]),
        skin_weights: f(&[0.25, 0.75, 1.], &[3]),
        vertex_indices: mx::from_slice_i32(&[0, 0, 1], &[3]),
        vertex_host: vec![0, 0, 1],
        corrective_input: mx::from_slice_i32(&[0, 4], &[2]),
        corrective_output: vec![0, 1],
        corrective_weights: f(&[1., 1.], &[2]),
        corrective_linear: z(&[6, 3000]),
        minmax_min: f(&[-1.], &[1]),
        minmax_max: f(&[1.], &[1]),
        minmax_indices: mx::from_slice_i32(&[0], &[1]),
        minmax_host: vec![0],
    };
    let mut shape = vec![0.; 2 * 45];
    shape[0] = 2.;
    shape[45] = -2.;
    let mut parameters = vec![0.; 2 * 204];
    parameters[0] = 2.;
    parameters[204] = -3.;
    let shape = f(&shape, &[2, 45]);
    let parameters = f(&parameters, &[2, 204]);
    let (vertices, state) = model.forward(&shape, &parameters, Some(&z(&[2, 72])))?;
    exact(
        &vertices,
        &f(
            &[4., 2., 3., 0., 0., 1., -3., 2., 3., -5., 0., 1.],
            &[2, 2, 3],
        ),
        "blend and skinning",
    )?;
    exact(
        &last(&state, 0, 3),
        &f(
            &[
                2., 0., 0., 3., 0., 0., 3., 2., 0., -3., 0., 0., -2., 0., 0., -2., 2., 0.,
            ],
            &[2, 3, 3],
        ),
        "FK positions",
    )?;
    // Values outside limits remain unchanged in forward, as in upstream JIT.
    let limited = model.apply_parameter_limits(&parameters)?;
    exact(
        &last(&limited, 0, 1),
        &f(&[1., -1.], &[2, 1]),
        "optional limits helper",
    )?;
    let without_expression = model.forward(&shape, &parameters, None)?.0;
    exact(&without_expression, &vertices, "optional zero expression")?;
    Ok(())
}
