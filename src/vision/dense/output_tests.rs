// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::vision::dense::predictor::PersonKeypoints;

#[test]
fn synthetic_output_round_trips_tensor_depth_and_keypoints() {
    // MLX does not support f64 tensors on GPU streams. Hold the shared device
    // lock while this normalization regression exercises an f64 CPU tensor.
    let _device_lock = mlxcel_core::streams::lock_default_device();
    let _cpu_device = mlxcel_core::streams::DefaultDeviceGuard::cpu();
    let dir = tempfile::TempDir::new().unwrap();
    let mut output = DenseOutput::default();
    let depth = mlxcel_core::from_slice_f32(&[f32::NAN, 1.0, 2.0, 3.0], &[2, 2]);
    output.tensors.insert(
        "depth".into(),
        mlxcel_core::astype(&depth, mlxcel_core::dtype::FLOAT64),
    );
    output.tensors.insert(
        "normal".into(),
        mlxcel_core::from_slice_f32(
            &[
                -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0,
            ],
            &[2, 2, 3],
        ),
    );
    output.keypoints.push(PersonKeypoints {
        bbox: [1.0, 2.0, 3.0, 4.0],
        points: vec![[5.0, 6.0, 0.9]],
        names: Some(vec!["nose".into()]),
    });
    let info = DenseOutputInfo {
        family: "stub",
        task: "depth",
        input_size: (2, 2),
        elapsed_ms: 1.25,
        is_video: false,
    };

    let paths = write_dense_output(dir.path(), &output, &info).unwrap();
    assert!(
        paths
            .iter()
            .any(|path| path.ends_with("result.safetensors"))
    );
    let bytes = std::fs::read(dir.path().join("result.safetensors")).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut names = tensors.names();
    names.sort_unstable();
    assert_eq!(names, vec!["depth", "normal"]);
    assert_eq!(tensors.tensor("depth").unwrap().dtype(), Dtype::F32);
    assert_eq!(tensors.tensor("depth").unwrap().shape(), &[2, 2]);
    assert_eq!(tensors.tensor("normal").unwrap().shape(), &[2, 2, 3]);

    let depth = image::open(dir.path().join("depth.png")).unwrap();
    assert_eq!(depth.color(), image::ColorType::L16);
    let depth = depth.to_luma16();
    assert_eq!(depth.get_pixel(0, 0).0[0], 0, "NaN preview is zero");
    assert_eq!(depth.get_pixel(1, 0).0[0], 0, "finite minimum is zero");
    assert_eq!(depth.get_pixel(1, 1).0[0], u16::MAX);

    let meta: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
    assert_eq!(meta["maps"]["depth"]["min"], 1.0);
    assert_eq!(meta["maps"]["depth"]["max"], 3.0);
    let keypoints: Vec<PersonKeypoints> =
        serde_json::from_slice(&std::fs::read(dir.path().join("keypoints.json")).unwrap()).unwrap();
    assert_eq!(keypoints, output.keypoints);
}

#[test]
fn video_depth_writes_one_preview_per_stacked_frame() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut output = DenseOutput::default();
    output.tensors.insert(
        "depth".into(),
        mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[2, 1, 2]),
    );
    write_dense_output(
        dir.path(),
        &output,
        &DenseOutputInfo {
            family: "stub",
            task: "depth",
            input_size: (2, 1),
            elapsed_ms: 0.0,
            is_video: true,
        },
    )
    .unwrap();
    assert!(dir.path().join("depth/000000.png").is_file());
    assert!(dir.path().join("depth/000001.png").is_file());
}

#[test]
fn one_frame_video_still_uses_a_numbered_preview_directory() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut output = DenseOutput::default();
    output.tensors.insert(
        "depth".into(),
        mlxcel_core::from_slice_f32(&[0.0, 1.0], &[1, 1, 2]),
    );
    write_dense_output(
        dir.path(),
        &output,
        &DenseOutputInfo {
            family: "stub",
            task: "depth",
            input_size: (2, 1),
            elapsed_ms: 0.0,
            is_video: true,
        },
    )
    .unwrap();
    assert!(dir.path().join("depth/000000.png").is_file());
    assert!(!dir.path().join("depth.png").exists());
}

#[test]
fn video_preview_rejects_an_unstacked_image_map() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut output = DenseOutput::default();
    output.tensors.insert(
        "depth".into(),
        mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[2, 2]),
    );
    let error = write_dense_output(
        dir.path(),
        &output,
        &DenseOutputInfo {
            family: "stub",
            task: "depth",
            input_size: (2, 2),
            elapsed_ms: 0.0,
            is_video: true,
        },
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("unsupported scalar-map shape [2, 2]"),
        "{error}"
    );
}

#[test]
fn reruns_remove_only_stale_writer_owned_artifacts() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("keep.txt"), "keep").unwrap();
    std::fs::create_dir(dir.path().join("depth")).unwrap();
    std::fs::write(dir.path().join("depth/keep.txt"), "keep").unwrap();

    let mut image_output = DenseOutput::default();
    image_output.tensors.insert(
        "depth".into(),
        mlxcel_core::from_slice_f32(&[0.0, 1.0], &[1, 2]),
    );
    image_output.keypoints.push(PersonKeypoints {
        bbox: [0.0; 4],
        points: Vec::new(),
        names: None,
    });
    write_dense_output(
        dir.path(),
        &image_output,
        &DenseOutputInfo {
            family: "stub",
            task: "depth",
            input_size: (2, 1),
            elapsed_ms: 0.0,
            is_video: false,
        },
    )
    .unwrap();
    assert!(dir.path().join("depth.png").is_file());
    assert!(dir.path().join("keypoints.json").is_file());

    let mut three_frames = DenseOutput::default();
    three_frames.tensors.insert(
        "depth".into(),
        mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0], &[3, 1, 1]),
    );
    write_dense_output(
        dir.path(),
        &three_frames,
        &DenseOutputInfo {
            family: "stub",
            task: "depth",
            input_size: (1, 1),
            elapsed_ms: 0.0,
            is_video: true,
        },
    )
    .unwrap();
    assert!(!dir.path().join("depth.png").exists());
    assert!(!dir.path().join("keypoints.json").exists());
    assert!(dir.path().join("depth/000002.png").is_file());

    let mut one_frame = DenseOutput::default();
    one_frame.tensors.insert(
        "depth".into(),
        mlxcel_core::from_slice_f32(&[4.0], &[1, 1, 1]),
    );
    write_dense_output(
        dir.path(),
        &one_frame,
        &DenseOutputInfo {
            family: "stub",
            task: "depth",
            input_size: (1, 1),
            elapsed_ms: 0.0,
            is_video: true,
        },
    )
    .unwrap();
    assert!(dir.path().join("depth/000000.png").is_file());
    assert!(!dir.path().join("depth/000001.png").exists());
    assert!(!dir.path().join("depth/000002.png").exists());

    write_dense_output(
        dir.path(),
        &image_output,
        &DenseOutputInfo {
            family: "stub",
            task: "depth",
            input_size: (2, 1),
            elapsed_ms: 0.0,
            is_video: false,
        },
    )
    .unwrap();
    assert!(dir.path().join("depth.png").is_file());
    assert!(!dir.path().join("depth/000000.png").exists());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("keep.txt")).unwrap(),
        "keep"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("depth/keep.txt")).unwrap(),
        "keep"
    );
}

#[cfg(unix)]
#[test]
fn video_preview_replaces_a_directory_symlink_without_touching_its_target() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::TempDir::new().unwrap();
    let external = tempfile::TempDir::new().unwrap();
    std::fs::write(external.path().join("000000.png"), "external preview").unwrap();
    std::fs::write(external.path().join("keep.txt"), "external sentinel").unwrap();
    std::fs::write(
        external.path().join("result.safetensors"),
        "external tensors",
    )
    .unwrap();
    std::fs::write(external.path().join("meta.json"), "external metadata").unwrap();
    symlink(external.path(), dir.path().join("depth")).unwrap();
    symlink(
        external.path().join("result.safetensors"),
        dir.path().join("result.safetensors"),
    )
    .unwrap();
    symlink(
        external.path().join("meta.json"),
        dir.path().join("meta.json"),
    )
    .unwrap();

    let mut output = DenseOutput::default();
    output.tensors.insert(
        "depth".into(),
        mlxcel_core::from_slice_f32(&[0.0, 1.0], &[1, 1, 2]),
    );
    write_dense_output(
        dir.path(),
        &output,
        &DenseOutputInfo {
            family: "stub",
            task: "depth",
            input_size: (2, 1),
            elapsed_ms: 0.0,
            is_video: true,
        },
    )
    .unwrap();

    assert!(dir.path().join("depth").is_dir());
    assert!(!dir.path().join("depth").is_symlink());
    assert!(dir.path().join("depth/000000.png").is_file());
    assert!(dir.path().join("result.safetensors").is_file());
    assert!(!dir.path().join("result.safetensors").is_symlink());
    assert!(dir.path().join("meta.json").is_file());
    assert!(!dir.path().join("meta.json").is_symlink());
    assert_eq!(
        std::fs::read_to_string(external.path().join("000000.png")).unwrap(),
        "external preview"
    );
    assert_eq!(
        std::fs::read_to_string(external.path().join("keep.txt")).unwrap(),
        "external sentinel"
    );
    assert_eq!(
        std::fs::read_to_string(external.path().join("result.safetensors")).unwrap(),
        "external tensors"
    );
    assert_eq!(
        std::fs::read_to_string(external.path().join("meta.json")).unwrap(),
        "external metadata"
    );
}
