// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Ignored stage capture for locating parity failures; never an acceptance substitute.
use super::*;
use safetensors::{Dtype, tensor::TensorView};
use std::{collections::BTreeMap, path::PathBuf};

struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
fn capture(tensors: &mut BTreeMap<String, Tensor>, name: String, value: &MlxArray) -> Result<()> {
    let dtype = match mx::array_dtype(value) {
        mx::dtype::FLOAT32 => Dtype::F32,
        mx::dtype::BFLOAT16 => Dtype::BF16,
        mx::dtype::INT32 => Dtype::I32,
        other => anyhow::bail!("unexpected diagnostic dtype {other} for {name}"),
    };
    mx::eval(value);
    ensure!(
        mx::item_bool(&mx::all_all(&mx::isfinite(value))),
        "nonfinite {name}"
    );
    ensure!(
        tensors
            .insert(
                name,
                Tensor {
                    dtype,
                    shape: mx::array_shape(value).iter().map(|v| *v as usize).collect(),
                    bytes: mx::array_to_raw_bytes(value),
                }
            )
            .is_none(),
        "duplicate diagnostic tensor"
    );
    Ok(())
}
fn env_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("set {name}"))
}

#[test]
#[ignore = "requires authenticated checkpoint/input assets and an exclusive GPU slot"]
fn sam3d_objects_rgba_stage_capture() -> Result<()> {
    let path = env_path("MLXCEL_SAM3D_OBJECTS_MODEL")?;
    let input = env_path("MLXCEL_SAM3D_OBJECTS_INPUT_DIR")?.join("object-rgba.png");
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
    super::super::checkpoint_schema::validate_headers(&path, &config)?;
    let image = image::open(&input)?;
    let (inputs, metadata) = processing::prepare_inputs(&image, None, None, config.image_size)?;
    ensure!(
        !metadata.conditioned,
        "diagnostic must exercise RGBA pointmap dropout"
    );
    let mut tensors = BTreeMap::new();
    for (name, value) in &inputs {
        capture(&mut tensors, format!("input.{name}"), value)?;
    }
    let model = Model::load(&path, &config)?;
    let pixels = crate::vision::dense::interpolate::resize_bilinear_nhwc(
        &inputs["image"],
        (config.image_size, config.image_size),
        false,
        false,
    )?;
    let pixels = mx::astype(
        &*crate::vision::dense::preprocess::normalize_imagenet(&pixels)?,
        model.backbones[0].dtype(),
    );
    capture(&mut tensors, "dino.image.normalized_pixels".into(), &pixels)?;
    for (name, value) in model.backbones[0].objects_stage_diagnostics(&pixels)? {
        capture(&mut tensors, format!("dino.image.{name}"), &value)?;
    }
    let mut cache = FeatureCache::new();
    let tokens = model
        .structure_condition
        .forward(&model.backbones, &inputs, &mut cache)?;
    capture(&mut tensors, "structure.condition".into(), &tokens)?;
    for (name, (norm, prenorm)) in &cache {
        capture(&mut tensors, format!("dino.{name}.norm"), norm)?;
        capture(&mut tensors, format!("dino.{name}.prenorm"), prenorm)?;
    }
    let keys = mx::random_split_key(&mx::random_key(42), 6);
    let state: State = LATENTS
        .iter()
        .enumerate()
        .map(|(i, (name, channels))| {
            (
                (*name).to_owned(),
                normal(
                    &keys,
                    i as i32,
                    &[
                        1,
                        if *name == "shape" {
                            config.latent_resolution.pow(3) as i32
                        } else {
                            1
                        },
                        *channels,
                    ],
                ),
            )
        })
        .collect();
    let condition = model.structure.prepare_condition(Some(&tokens))?;
    let uncondition = model.structure.prepare_condition(None)?;
    let time = mx::zeros(&[1], mx::dtype::FLOAT32);
    let delta = mx::zeros(&[1], mx::dtype::FLOAT32);
    let (velocity, uncond) =
        model
            .structure
            .guided(&state, &time, &condition, &uncondition, Some(&delta))?;
    let times = schedule(config.ss_steps, config.ss_rescale_t);
    for (name, value) in &state {
        capture(&mut tensors, format!("initial.{name}"), value)?;
        capture(
            &mut tensors,
            format!("velocity.conditioned.{name}"),
            &velocity[name],
        )?;
        capture(
            &mut tensors,
            format!("velocity.unconditioned.{name}"),
            &uncond[name],
        )?;
        let next = update(
            value,
            &velocity[name],
            Some(&uncond[name]),
            config.ss_guidance,
            times[1] - times[0],
        );
        capture(&mut tensors, format!("step1.{name}"), &next)?;
    }
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
    safetensors::serialize_to_file(views, None, &output.join("stages.safetensors"))?;
    let metadata = json!({
        "purpose":"diagnostic-only", "case":"rgba-dropout", "seed":42,
        "structure_steps":config.ss_steps, "captured_step":1,
        "runtime_device":runtime.device.to_string(), "tf32":"0",
        "input":input, "tensor_count":tensors.len(),
        "source_identity":"supplied by external frozen process receipt"
    });
    std::fs::write(
        output.join("stages.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}

#[path = "pipeline_sampling_diagnostics.rs"]
mod sampling;

#[path = "pipeline_default_diagnostics.rs"]
mod default;

#[path = "pipeline_moge_diagnostics.rs"]
mod moge;
