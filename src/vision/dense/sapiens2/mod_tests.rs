// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(serde::Deserialize)]
struct StagePoseReference {
    keypoints: Vec<Vec<Vec<f32>>>,
    scores: Vec<Vec<f32>>,
}

fn write_stage_dump(path: &std::path::Path, tensors: &[(&str, &mlxcel_core::MlxArray)]) {
    use safetensors::tensor::{Dtype, TensorView};
    use sha2::{Digest, Sha256};

    let owned: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, tensor)| {
            let tensor = mlxcel_core::astype(tensor, mlxcel_core::dtype::FLOAT32);
            let shape = mlxcel_core::array_shape(&tensor)
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect();
            let bytes = mlxcel_core::utils::array_to_vec_f32(&tensor)
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            ((*name).to_owned(), shape, bytes)
        })
        .collect();
    let views = owned
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).expect("valid stage tensor"),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mut metadata = std::collections::HashMap::new();
    for (name, shape, bytes) in &owned {
        metadata.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "sha256": format!("{:x}", Sha256::digest(bytes)),
            })
            .to_string(),
        );
    }
    safetensors::serialize_to_file(&views, Some(metadata), path)
        .expect("write Sapiens2 stage dump");
    let file = std::fs::read(path).expect("read Sapiens2 stage dump");
    eprintln!(
        "Sapiens2 stage dump: path={}, sha256={:x}",
        path.display(),
        Sha256::digest(file)
    );
}

fn parity_metrics(actual: &mlxcel_core::MlxArray, expected: &mlxcel_core::MlxArray) -> (f32, f32) {
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected)
    );
    let actual = mlxcel_core::astype(actual, mlxcel_core::dtype::FLOAT32);
    let expected = mlxcel_core::astype(expected, mlxcel_core::dtype::FLOAT32);
    assert_eq!(
        mlxcel_core::item_f32(&mlxcel_core::astype(
            &mlxcel_core::all_all(&mlxcel_core::isfinite(&actual)),
            mlxcel_core::dtype::FLOAT32,
        )),
        1.0
    );
    assert_eq!(
        mlxcel_core::item_f32(&mlxcel_core::astype(
            &mlxcel_core::all_all(&mlxcel_core::isfinite(&expected)),
            mlxcel_core::dtype::FLOAT32,
        )),
        1.0
    );
    let difference = mlxcel_core::subtract(&actual, &expected);
    let max_abs = mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(&difference)));
    let rms =
        mlxcel_core::item_f32(&mlxcel_core::mean_all(&mlxcel_core::square(&difference))).sqrt();
    let scale =
        mlxcel_core::item_f32(&mlxcel_core::mean_all(&mlxcel_core::square(&expected))).sqrt();
    (max_abs, rms / scale.max(f32::MIN_POSITIVE))
}

fn assert_pose_coordinates(
    keypoints: &mlxcel_core::MlxArray,
    scores: &mlxcel_core::MlxArray,
    reference_json: &str,
) {
    let reference: StagePoseReference =
        serde_json::from_str(reference_json).expect("parse pose stage reference");
    let actual = mlxcel_core::utils::array_to_vec_f32(keypoints);
    let expected: Vec<f32> = reference
        .keypoints
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let max_error = actual
        .chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .map(|(actual, expected)| {
            ((actual[0] - expected[0]).powi(2) + (actual[1] - expected[1]).powi(2)).sqrt()
        })
        .fold(0.0f32, f32::max);
    let actual_scores = mlxcel_core::utils::array_to_vec_f32(scores);
    let expected_scores: Vec<f32> = reference.scores.into_iter().flatten().collect();
    let max_score_error = actual_scores
        .iter()
        .zip(expected_scores)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "Sapiens2 pose output: max_keypoint_error={max_error:.6}px, max_score_error={max_score_error:.9}"
    );
    assert!(max_error <= 1.5, "maximum keypoint error {max_error} px");
    assert!(
        max_score_error <= 0.05,
        "maximum keypoint score error {max_score_error}"
    );
}

#[test]
fn pose_boxes_default_empty_clip_and_validate() {
    let opts = Map::new();
    assert_eq!(
        parse_pose_boxes(&opts, 1280, 720).unwrap(),
        [[0.0, 0.0, 1279.0, 719.0]]
    );

    let mut opts = Map::new();
    opts.insert("boxes".into(), serde_json::json!([]));
    assert!(parse_pose_boxes(&opts, 1280, 720).unwrap().is_empty());

    opts.insert("boxes".into(), serde_json::json!([[-4, -3, 1300, 800]]));
    assert_eq!(
        parse_pose_boxes(&opts, 1280, 720).unwrap(),
        [[0.0, 0.0, 1280.0, 720.0]]
    );

    opts.insert("boxes".into(), serde_json::json!([[0, 0, 3]]));
    assert!(parse_pose_boxes(&opts, 1280, 720).is_err());
    opts.insert("boxes".into(), serde_json::json!([[10, 10, 5, 20]]));
    assert!(parse_pose_boxes(&opts, 1280, 720).is_err());

    opts.insert(
        "boxes".into(),
        Value::Array(vec![serde_json::json!([0, 0, 10, 10]); MAX_POSE_BOXES + 1]),
    );
    assert!(parse_pose_boxes(&opts, 1280, 720).is_err());
}

#[test]
#[ignore = "requires SAPIENS2_POSE_MODEL, SAPIENS2_POSE_STAGES_REFERENCE and SAPIENS2_POSE_OUTPUT_REFERENCE"]
fn sapiens2_pose_real_checkpoint_stage_parity() {
    let model_path = std::env::var("SAPIENS2_POSE_MODEL").expect("set SAPIENS2_POSE_MODEL");
    let reference_path = std::env::var("SAPIENS2_POSE_STAGES_REFERENCE")
        .expect("set SAPIENS2_POSE_STAGES_REFERENCE");
    let output_reference_path = std::env::var("SAPIENS2_POSE_OUTPUT_REFERENCE")
        .expect("set SAPIENS2_POSE_OUTPUT_REFERENCE for the stage-reference backend");
    let reference = mlxcel_core::weights::load_safetensors(reference_path)
        .expect("load upstream stage reference");
    let model = Sapiens2Model::load(std::path::Path::new(&model_path)).unwrap();
    let image = image::open("tests/fixtures/sapiens2/person.png").unwrap();
    let source = image_to_nhwc_255(&image);
    let bbox = [
        0.0,
        0.0,
        image.width() as f32 - 1.0,
        image.height() as f32 - 1.0,
    ];
    let input_size = (model.config.image_size.1, model.config.image_size.0);
    let (crop, center, scale) = pose::warp_to_input(&source, bbox, input_size).unwrap();
    let pixels = normalize_sapiens2(&crop).unwrap();
    if let Some(reference_dir) = std::env::var_os("SAPIENS2_POSE_ALL_STAGES_REFERENCE") {
        let metrics = model.all_stage_metrics(&pixels, std::path::Path::new(&reference_dir));
        for (name, max_abs, relative_rms) in &metrics {
            eprintln!(
                "Sapiens2 all-stage {name}: max_abs={max_abs:.9}, relative_rms={relative_rms:.9}"
            );
        }
        if let Some((name, max_abs, relative_rms)) = metrics
            .iter()
            .find(|(_, max_abs, relative_rms)| *max_abs != 0.0 || *relative_rms != 0.0)
        {
            eprintln!(
                "Sapiens2 first divergent stage: {name}, max_abs={max_abs:.9}, relative_rms={relative_rms:.9}"
            );
        } else {
            eprintln!("Sapiens2 all transformer and pose-head stages are bit-exact");
        }
    }
    let ModelOutput::Task {
        prediction: heatmaps,
        ..
    } = model.forward(&pixels).unwrap()
    else {
        panic!("pose model returned backbone output")
    };
    let (crop_keypoints, scores) = pose::udp_decode_batch(&heatmaps, input_size, 11).unwrap();
    let keypoints =
        pose::keypoints_to_image(&crop_keypoints, input_size, &[center], &[scale]).unwrap();
    let boxes = mlxcel_core::from_slice_f32(&bbox, &[1, 4]);
    let dump_path = std::env::var_os("SAPIENS2_POSE_ACTUAL_STAGES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "/tmp/sapiens2-rust-pose-stages.safetensors".into());
    write_stage_dump(
        &dump_path,
        &[
            ("pixel_values", &pixels),
            ("heatmaps", &heatmaps),
            ("keypoints", &keypoints),
            ("scores", &scores),
            ("boxes", &boxes),
        ],
    );
    if let Ok(block_reference_path) = std::env::var("SAPIENS2_POSE_BLOCK0_REFERENCE") {
        let block_reference = mlxcel_core::weights::load_safetensors(block_reference_path)
            .expect("load upstream block-zero reference");
        let block_stages = model.backbone.first_block_stages(&pixels);
        let block_dump_path = std::env::var_os("SAPIENS2_POSE_ACTUAL_BLOCK0")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "/tmp/sapiens2-rust-pose-block0.safetensors".into());
        let tensors: Vec<(&str, &mlxcel_core::MlxArray)> = block_stages
            .iter()
            .map(|(name, tensor)| {
                (
                    name.as_str(),
                    tensor.as_ref().expect("MLX block-zero stage allocation"),
                )
            })
            .collect();
        write_stage_dump(&block_dump_path, &tensors);
        for name in [
            "periods", "patch", "tokens", "sin", "cos", "ln1", "qkv", "q_norm", "k_norm", "q_rope",
            "k_rope", "v", "sdpa", "block0",
        ] {
            let (max_abs, relative_rms) =
                parity_metrics(&block_stages[name], &block_reference[name]);
            eprintln!(
                "Sapiens2 block0 {name}: max_abs={max_abs:.9}, relative_rms={relative_rms:.9}"
            );
        }
    }

    assert_eq!(center, [639.5, 359.5]);
    assert_eq!(scale, [1598.75, 2131.6667]);
    let (pixel_max_abs, pixel_relative_rms) = parity_metrics(&pixels, &reference["pixel_values"]);
    eprintln!(
        "Sapiens2 pose pixels: max_abs={pixel_max_abs:.9}, relative_rms={pixel_relative_rms:.9}"
    );
    assert!(
        pixel_max_abs <= 1e-5,
        "pose preprocessing differs from upstream"
    );

    if let Some(expected_heatmaps) = reference.get("heatmaps") {
        let (heatmap_max_abs, heatmap_relative_rms) = parity_metrics(&heatmaps, expected_heatmaps);
        eprintln!(
            "Sapiens2 pose heatmaps: max_abs={heatmap_max_abs:.9}, relative_rms={heatmap_relative_rms:.9}; dump={}",
            dump_path.display()
        );
        assert!(
            heatmap_relative_rms < 1e-2,
            "pose heatmaps differ from upstream"
        );

        let ModelOutput::Task {
            prediction: upstream_input_heatmaps,
            ..
        } = model.forward(&reference["pixel_values"]).unwrap()
        else {
            panic!("pose model returned backbone output")
        };
        let (upstream_input_max_abs, upstream_input_relative_rms) =
            parity_metrics(&upstream_input_heatmaps, expected_heatmaps);
        eprintln!(
            "Sapiens2 pose heatmaps from exact upstream input: max_abs={upstream_input_max_abs:.9}, relative_rms={upstream_input_relative_rms:.9}"
        );
        assert!(
            upstream_input_relative_rms < 1e-2,
            "pose model differs from upstream with exact input"
        );
    }
    let output_reference =
        std::fs::read_to_string(output_reference_path).expect("read pose output reference");
    assert_pose_coordinates(&keypoints, &scores, &output_reference);
}
