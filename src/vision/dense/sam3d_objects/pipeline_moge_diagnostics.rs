// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Eager MoGe observations for first-divergence diagnosis, never parity acceptance.
use super::*;
use std::{cell::RefCell, rc::Rc};

fn capture_raw(tensors: &mut BTreeMap<String, Tensor>, name: &str, value: &MlxArray) -> Result<()> {
    let dtype = match mx::array_dtype(value) {
        mx::dtype::FLOAT32 => Dtype::F32,
        mx::dtype::INT32 => Dtype::I32,
        mx::dtype::BOOL => Dtype::BOOL,
        other => anyhow::bail!("unexpected MoGe diagnostic dtype {other} for {name}"),
    };
    let shape = mx::array_shape(value)
        .into_iter()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let element_bytes = if dtype == Dtype::BOOL { 1usize } else { 4usize };
    let bytes = shape.iter().try_fold(element_bytes, |size, dimension| {
        size.checked_mul(*dimension)
            .context("diagnostic tensor byte overflow")
    })?;
    let retained = tensors.values().try_fold(bytes, |size, tensor| {
        size.checked_add(tensor.bytes.len())
            .context("diagnostic total byte overflow")
    })?;
    ensure!(
        retained <= 512 * 1024 * 1024,
        "diagnostic host capture exceeds 512 MiB"
    );
    ensure!(
        !tensors.contains_key(name),
        "duplicate diagnostic tensor {name}"
    );
    mx::eval(value);
    let bytes = mx::array_to_raw_bytes(value);
    tensors.insert(
        name.to_owned(),
        Tensor {
            dtype,
            shape,
            bytes,
        },
    );
    Ok(())
}

#[test]
#[ignore = "requires authenticated checkpoint/input assets and an exclusive GPU slot"]
fn sam3d_objects_moge_stage_capture() -> Result<()> {
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
        "capture requires bundled depth"
    );
    processing::validate_image_mask(&image, Some(&mask), config.image_size, None, true)?;
    super::super::super::checkpoint_schema::validate_headers(&path, &config)?;
    let tensors = Rc::new(RefCell::new(BTreeMap::new()));
    let sink = Rc::clone(&tensors);
    let points = crate::vision::dense::moge3::diagnostics::with_capture(
        move |name, value| capture_raw(&mut sink.borrow_mut(), name, value),
        || bundle::estimate_depth(&path, &config, &image),
    )?
    .context("capture did not estimate depth")?;
    let mut tensors = Rc::try_unwrap(tensors)
        .map_err(|_| anyhow::anyhow!("MoGe observer retained its sink"))?
        .into_inner();
    capture_raw(&mut tensors, "depth.pointmap", &points)?;
    capture_raw(
        &mut tensors,
        "depth.finite",
        &mx::astype(&mx::isfinite(&points), mx::dtype::INT32),
    )?;
    ensure!(
        tensors.len() == 43,
        "unexpected MoGe diagnostic tensor count: {}",
        tensors.len()
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
    safetensors::serialize_to_file(views, None, &output.join("moge-stages.safetensors"))?;
    let receipt = json!({
        "purpose":"diagnostic-only", "case":"default-moge", "estimate_depth":true,
        "depth_num_tokens":config.depth_num_tokens, "runtime_device":runtime.device.to_string(),
        "tf32":"0", "image":image_path, "mask":mask_path, "tensor_count":tensors.len(),
        "capture_bytes":tensors.values().map(|tensor| tensor.bytes.len()).sum::<usize>(),
        "observation":"eager host serialization at selected existing MoGe operations, with two explicitly recomputed preprocessing observations",
        "recomputed_keys":["moge.encoder.resized", "moge.encoder.normalized"],
        "nonfinite_policy":"preserve raw NaN/Inf bytes; compare classes before finite errors",
        "corroboration":"compare depth.pointmap against uninstrumented default-stage capture before attributing differences",
        "source_identity":"supplied by external frozen process receipt"
    });
    std::fs::write(
        output.join("moge-stages.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}
