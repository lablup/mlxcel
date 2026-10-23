// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Default-only depth and conditioning diagnostic; no sampling or acceptance claim.
use super::*;

/// Preserve NaNs and infinities in point maps so their masks can be compared
/// before computing finite-value errors. Other stage tensors use strict capture.
fn capture_pointmap(
    tensors: &mut BTreeMap<String, Tensor>,
    name: String,
    value: &MlxArray,
) -> Result<()> {
    ensure!(
        mx::array_dtype(value) == mx::dtype::FLOAT32,
        "pointmap must be F32"
    );
    mx::eval(value);
    ensure!(
        tensors
            .insert(
                name,
                Tensor {
                    dtype: Dtype::F32,
                    shape: mx::array_shape(value).iter().map(|v| *v as usize).collect(),
                    bytes: mx::array_to_raw_bytes(value),
                }
            )
            .is_none(),
        "duplicate diagnostic tensor"
    );
    Ok(())
}

#[test]
#[ignore = "requires authenticated checkpoint/input assets and an exclusive GPU slot"]
fn sam3d_objects_default_stage_capture() -> Result<()> {
    let path = env_path("MLXCEL_SAM3D_OBJECTS_MODEL")?;
    let input_dir = env_path("MLXCEL_SAM3D_OBJECTS_INPUT_DIR")?;
    let output = env_path("MLXCEL_SAM3D_OBJECTS_DIAGNOSTIC_DIR")?;
    ensure!(!output.exists(), "refuse to overwrite a diagnostic capture");
    ensure!(
        std::env::var("MLX_ENABLE_TF32").as_deref() == Ok("0"),
        "disable TF32"
    );
    let runtime = crate::initialize_runtime_checked()?;
    eprintln!("Runtime device: {}", runtime.device);
    ensure!(
        runtime.device == crate::RuntimeDevice::Gpu,
        "stage capture requires GPU"
    );
    let config = bundle::config(&path)?;
    let image_path = input_dir.join("object.png");
    let mask_path = input_dir.join("mask.png");
    let image = image::open(&image_path)?;
    let mask = image::open(&mask_path)?;
    ensure!(
        config.depth_model.is_some(),
        "default capture requires bundled depth"
    );
    processing::validate_image_mask(&image, Some(&mask), config.image_size, None, true)?;
    super::super::super::checkpoint_schema::validate_headers(&path, &config)?;
    let points = bundle::estimate_depth(&path, &config, &image)?
        .context("default capture did not estimate depth")?;
    let mut tensors = BTreeMap::new();
    capture_pointmap(&mut tensors, "depth.pointmap".into(), &points)?;
    capture(
        &mut tensors,
        "depth.finite".into(),
        &mx::astype(&mx::isfinite(&points), mx::dtype::INT32),
    )?;
    let (inputs, metadata) =
        processing::prepare_inputs(&image, Some(&mask), Some(&points), config.image_size)?;
    ensure!(
        metadata.conditioned,
        "default capture must be pointmap-conditioned"
    );
    capture(
        &mut tensors,
        "pointmap.shift".into(),
        &mx::from_slice_f32(&metadata.shift, &[3]),
    )?;
    capture(
        &mut tensors,
        "pointmap.scale".into(),
        &mx::from_slice_f32(&[metadata.scale; 3], &[3]),
    )?;
    for (name, value) in &inputs {
        if name == "pointmap" || name == "rgb_pointmap" {
            capture_pointmap(&mut tensors, format!("input.{name}"), value)?;
        } else {
            capture(&mut tensors, format!("input.{name}"), value)?;
        }
    }
    let model = Model::load(&path, &config)?;
    for (name, value) in model.structure_condition.point_stage_diagnostics(&inputs)? {
        if name.ends_with(".resized") {
            capture_pointmap(&mut tensors, format!("point.{name}"), &value)?;
        } else {
            capture(&mut tensors, format!("point.{name}"), &value)?;
        }
    }
    let condition =
        model
            .structure_condition
            .forward(&model.backbones, &inputs, &mut FeatureCache::new())?;
    capture(&mut tensors, "structure.condition".into(), &condition)?;
    ensure!(
        tensors.len() == 17,
        "unexpected default diagnostic tensor count"
    );
    std::fs::create_dir(&output)?;
    let views = tensors
        .iter()
        .map(|(name, value)| {
            Ok((
                name.as_str(),
                TensorView::new(value.dtype, value.shape.clone(), &value.bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize_to_file(views, None, &output.join("default-stages.safetensors"))?;
    let receipt = json!({
        "purpose":"diagnostic-only", "case":"default", "estimate_depth":true,
        "depth_num_tokens":config.depth_num_tokens, "runtime_device":runtime.device.to_string(),
        "tf32":"0", "image":image_path, "mask":mask_path, "tensor_count":tensors.len(),
        "pointmap_conditioned":metadata.conditioned,
        "nonfinite_policy":"preserve point-map NaN/Inf masks; compare masks before finite errors",
        "source_identity":"supplied by external frozen process receipt"
    });
    std::fs::write(
        output.join("default-stages.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}
