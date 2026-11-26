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

//! What the cuda-oxide route refuses, and how (#2833).
//!
//! Two different things must hold. A configuration the artifact does not serve
//! is decided when the graph node is built, takes the existing CUDA kernel,
//! and records why. A failure once the cuda-oxide node is being evaluated is
//! an error to the caller: nothing reroutes it to the existing kernel after
//! the fact. The tests here also run the wrong-manifest and missing-artifact
//! cases after a successful run in the same process, with the module loaded
//! and its PTX in MLX's disk cache, because a cache hit skips the payload and
//! must not be able to stand in for a refused artifact.

use std::ffi::OsStr;

use crate::cuda_oxide_artifact::{ArtifactError, Lane, TARGET_CAPABILITY, validate};
use crate::cuda_oxide_artifact_tests::{committed, repin, write_layout};
use crate::cuda_oxide_fused_norm::{self as oxide, Fallback, InstallError};
use crate::cuda_oxide_fused_norm_fixtures::{
    Case, RouteGuard, bits_of, case_arrays, delta, fused_lazy, inputs, run_fused,
};
use crate::streams::DefaultDeviceGuard;
use crate::{UniquePtr, try_eval, zeros};

fn case(act: Lane, w: Lane) -> Case {
    Case::new(act, w, 5, 1024, 1e-6, 0.0, false)
}

fn install() {
    oxide::install_embedded().expect("the embedded artifact validates and targets this device");
}

/// Run `c` and require that it took the existing kernel for `reason`, with
/// the cuda-oxide kernel never launched.
fn assert_takes_existing_kernel(c: &Case, index: usize, reason: Fallback) -> Vec<u32> {
    let arrays = case_arrays(c, &inputs(c, index));
    let before = oxide::route_counters();
    let out = run_fused(c, &arrays);
    let moved = delta(before);
    assert_eq!(
        (moved.oxide_routes, moved.launches, moved.native_routes),
        (0, 0, 1),
        "{} must take the existing kernel",
        c.name
    );
    assert_eq!(oxide::last_fallback(), reason, "{}", c.name);
    out.normed
}

#[test]
fn the_route_is_off_until_an_artifact_is_installed() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    assert!(!oxide::is_installed());
    let c = case(Lane::F32, Lane::F32);
    let existing = assert_takes_existing_kernel(&c, 1, Fallback::NotInstalled);

    install();
    assert!(oxide::is_installed());
    let arrays = case_arrays(&c, &inputs(&c, 1));
    let before = oxide::route_counters();
    assert_eq!(run_fused(&c, &arrays).normed, existing);
    assert_eq!(delta(before).launches, 1);

    oxide::uninstall();
    assert_takes_existing_kernel(&c, 1, Fallback::NotInstalled);
}

#[test]
fn the_runtime_switch_decides_whether_anything_is_installed() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    for off in [None, Some(""), Some("0"), Some("off"), Some("maybe")] {
        assert!(oxide::install_from_switches(off, None).is_none(), "{off:?}");
        assert!(!oxide::is_installed(), "{off:?} must not install");
    }
    assert!(matches!(
        oxide::install_from_switches(Some("1"), None),
        Some(Ok(()))
    ));
    assert!(oxide::is_installed());

    // An artifact directory that does not hold the artifact disables the
    // route; it does not fall back to the embedded copy.
    let empty = tempfile::tempdir().expect("tempdir");
    let result = oxide::install_from_switches(Some("on"), Some(empty.path().as_os_str()));
    assert!(matches!(
        result,
        Some(Err(InstallError::Artifact(ArtifactError::Missing { .. })))
    ));
    assert!(!oxide::is_installed());

    // An empty directory value means "not set".
    assert!(matches!(
        oxide::install_from_switches(Some("yes"), Some(OsStr::new(""))),
        Some(Ok(()))
    ));
    assert!(oxide::is_installed());
}

#[test]
fn a_dtype_pair_without_an_entry_takes_the_existing_kernel() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    // The module has five entries; these weight dtypes are not among them.
    let unsupported = [
        case(Lane::F32, Lane::F16),
        case(Lane::F32, Lane::Bf16),
        case(Lane::F16, Lane::Bf16),
        case(Lane::Bf16, Lane::F16),
    ];
    let existing: Vec<Vec<u32>> = unsupported
        .iter()
        .enumerate()
        .map(|(i, c)| assert_takes_existing_kernel(c, i, Fallback::NotInstalled))
        .collect();

    install();
    for (i, c) in unsupported.iter().enumerate() {
        let got = assert_takes_existing_kernel(c, i, Fallback::DtypePair);
        assert_eq!(got, existing[i], "{}", c.name);
    }
    // A supported pair right after still takes the cuda-oxide route.
    let c = case(Lane::F16, Lane::F32);
    let before = oxide::route_counters();
    run_fused(&c, &case_arrays(&c, &inputs(&c, 9)));
    assert_eq!(delta(before).launches, 1);
    assert_eq!(oxide::last_fallback(), Fallback::Taken);
}

#[test]
fn more_elements_than_the_validated_limit_take_the_existing_kernel() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    install();
    // 70000 x 65536 is 4.59e9 elements, past 2^32-1. The arrays are lazy and
    // are never evaluated, so nothing of that size is allocated; the route is
    // decided on shapes alone.
    let dt = Lane::F16.mlx_dtype();
    let x = zeros(&[70_000, 65_536], dt);
    let residual = zeros(&[70_000, 65_536], dt);
    let weight = zeros(&[65_536], dt);
    let before = oxide::route_counters();
    let _lazy = fused_lazy(&x, &residual, &weight, 1e-6, 0.0);
    let moved = delta(before);
    assert_eq!((moved.oxide_routes, moved.native_routes), (0, 1));
    assert_eq!(oxide::last_fallback(), Fallback::ElementLimit);

    // Just inside the limit the route is taken (still lazy, never evaluated).
    let x = zeros(&[65_535, 65_536], dt);
    let residual = zeros(&[65_535, 65_536], dt);
    let before = oxide::route_counters();
    let _lazy = fused_lazy(&x, &residual, &weight, 1e-6, 0.0);
    assert_eq!(delta(before).oxide_routes, 1);
}

#[test]
fn a_cpu_default_device_never_takes_the_route() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    install();
    let c = case(Lane::F32, Lane::F32);
    let arrays = case_arrays(&c, &inputs(&c, 2));
    let _cpu = DefaultDeviceGuard::cpu();
    let before = oxide::route_counters();
    let (mut normed, mut new_residual) = (UniquePtr::null(), UniquePtr::null());
    // The existing launcher refuses the CPU stream; what matters here is that
    // the adapter stepped aside first instead of building a node for it.
    let _ = crate::fused_add_rms_norm(
        &arrays.x,
        &arrays.residual,
        &arrays.weight,
        c.eps,
        0.0,
        &mut normed,
        &mut new_residual,
    );
    let moved = delta(before);
    assert_eq!((moved.oxide_routes, moved.native_routes), (0, 1));
    assert_eq!(oxide::last_fallback(), Fallback::DeviceNotGpu);
}

#[test]
fn a_device_of_another_architecture_is_refused_before_anything_is_built() {
    assert!(oxide::check_target(Some(TARGET_CAPABILITY)).is_ok());
    for device in [Some((9, 0)), Some((8, 6)), Some((12, 0)), None] {
        match oxide::check_target(device) {
            Err(InstallError::Target { .. }) => {}
            other => panic!("{device:?} must be refused, got {other:?}"),
        }
    }
}

#[test]
fn a_warm_cache_cannot_mask_a_wrong_manifest_or_a_missing_artifact() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    let c = case(Lane::Bf16, Lane::Bf16);
    let existing = assert_takes_existing_kernel(&c, 3, Fallback::NotInstalled);

    // Warm both caches: the module is loaded in this process and its PTX is in
    // MLX's disk cache under the embedded artifact's module name.
    install();
    let arrays = case_arrays(&c, &inputs(&c, 3));
    let before = oxide::route_counters();
    assert_eq!(run_fused(&c, &arrays).normed, existing);
    assert_eq!(delta(before).launches, 1);

    let (ptx, documents) = committed();

    // Wrong manifest: one entry pins another digest.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut wrong = documents.clone();
    wrong[0].manifest = wrong[0].manifest.replace(
        &validate(&ptx, &documents).expect("valid").sha256,
        &"0".repeat(64),
    );
    write_layout(dir.path(), &ptx, &wrong);
    assert!(matches!(
        oxide::install_from_dir(dir.path()),
        Err(InstallError::Artifact(ArtifactError::Digest { .. }))
    ));
    assert!(
        !oxide::is_installed(),
        "a refused artifact must not stay installed"
    );
    assert_eq!(
        assert_takes_existing_kernel(&c, 3, Fallback::NotInstalled),
        existing
    );

    // Missing artifact: manifests and schemas, no PTX.
    let dir = tempfile::tempdir().expect("tempdir");
    write_layout(dir.path(), &ptx, &documents);
    std::fs::remove_file(
        dir.path()
            .join("artifacts")
            .join(crate::cuda_oxide_artifact::PTX_FILE),
    )
    .expect("remove");
    assert!(matches!(
        oxide::install_from_dir(dir.path()),
        Err(InstallError::Artifact(ArtifactError::Missing { .. }))
    ));
    assert_takes_existing_kernel(&c, 3, Fallback::NotInstalled);

    // A rebuilt artifact, consistently re-pinned, is accepted under a new
    // module name: it cannot be served the first artifact's cache entry, and
    // it does launch.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut rebuilt_ptx = ptx.clone();
    rebuilt_ptx.extend_from_slice(b"\n// rebuilt for the cache identity test\n");
    let mut rebuilt = documents.clone();
    repin(&mut rebuilt, &rebuilt_ptx);
    write_layout(dir.path(), &rebuilt_ptx, &rebuilt);
    assert_ne!(
        validate(&rebuilt_ptx, &rebuilt).expect("valid").module_name,
        validate(&ptx, &documents).expect("valid").module_name
    );
    oxide::install_from_dir(dir.path()).expect("the rebuilt artifact installs");
    let before = oxide::route_counters();
    assert_eq!(run_fused(&c, &arrays).normed, existing);
    assert_eq!(
        (delta(before).launches, delta(before).native_routes),
        (1, 0)
    );
}

#[test]
fn an_evaluation_time_failure_is_an_error_not_a_retry() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    let c = case(Lane::F32, Lane::F32);
    let arrays = case_arrays(&c, &inputs(&c, 4));
    let existing = assert_takes_existing_kernel(&c, 4, Fallback::NotInstalled);
    let (ptx, documents) = committed();
    let entries = validate(&ptx, &documents).expect("valid").entries;

    // A payload that cannot load. It is installed around the validator on
    // purpose: validation is what normally keeps this from happening, and the
    // question here is what the node does if a load fails anyway.
    oxide::install_unchecked(
        b"this is not a PTX module\n",
        "mlxcel_oxide_fused_norm_test_unloadable_payload",
        TARGET_CAPABILITY,
        &entries,
    )
    .expect("the adapter accepts the description");
    let before = oxide::route_counters();
    let (normed, _new_residual) =
        fused_lazy(&arrays.x, &arrays.residual, &arrays.weight, c.eps, 0.0);
    assert_eq!(
        delta(before).oxide_routes,
        1,
        "the route is chosen at build time"
    );
    assert!(try_eval(&normed).is_err(), "the load failure must surface");
    // A second node finds the first attempt's entry in the PTX disk cache.
    // MLX discards an entry that fails to load and republishes it once; the
    // payload is the same, so the failure surfaces again.
    let (normed, _new_residual) =
        fused_lazy(&arrays.x, &arrays.residual, &arrays.weight, c.eps, 0.0);
    assert!(try_eval(&normed).is_err(), "and keep surfacing");
    let moved = delta(before);
    assert_eq!(
        (moved.launches, moved.native_routes),
        (0, 0),
        "a failed node neither launches nor reroutes to the existing kernel"
    );

    // The real payload under a target the device is not: refused by the node
    // itself when it is evaluated.
    oxide::install_unchecked(
        &ptx,
        "mlxcel_oxide_fused_norm_test_wrong_target",
        (9, 0),
        &entries,
    )
    .expect("the adapter accepts the description");
    let before = oxide::route_counters();
    let (normed, _new_residual) =
        fused_lazy(&arrays.x, &arrays.residual, &arrays.weight, c.eps, 0.0);
    let error = try_eval(&normed).expect_err("a target mismatch must surface");
    assert!(error.to_string().contains("sm_90"), "{error}");
    assert_eq!(
        (delta(before).launches, delta(before).native_routes),
        (0, 0)
    );

    // The stream is still usable, on both implementations.
    oxide::uninstall();
    assert_eq!(
        assert_takes_existing_kernel(&c, 4, Fallback::NotInstalled),
        existing
    );
    install();
    let before = oxide::route_counters();
    let (normed, _) = fused_lazy(&arrays.x, &arrays.residual, &arrays.weight, c.eps, 0.0);
    assert_eq!(bits_of(&normed, c.act), existing);
    assert_eq!(delta(before).launches, 1);
}
