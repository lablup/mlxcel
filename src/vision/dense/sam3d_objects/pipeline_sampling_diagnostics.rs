// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Full structure trajectory diagnostics; never a reconstruction acceptance substitute.
use super::*;

#[test]
#[ignore = "requires authenticated checkpoint/input assets and an exclusive GPU slot"]
fn sam3d_objects_rgba_sampling_capture() -> Result<()> {
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
        "sampling capture requires GPU"
    );
    let config = bundle::config(&path)?;
    super::super::super::checkpoint_schema::validate_headers(&path, &config)?;
    ensure!(
        config.ss_steps == 25,
        "capture requires released 25-step schedule"
    );
    let image = image::open(&input)?;
    let (inputs, metadata) = processing::prepare_inputs(&image, None, None, config.image_size)?;
    ensure!(
        !metadata.conditioned,
        "diagnostic requires pointmap dropout"
    );
    let model = Model::load(&path, &config)?;
    let mut cache = FeatureCache::new();
    let tokens = model
        .structure_condition
        .forward(&model.backbones, &inputs, &mut cache)?;
    let condition = model.structure.prepare_condition(Some(&tokens))?;
    let uncondition = model.structure.prepare_condition(None)?;
    let keys = mx::random_split_key(&mx::random_key(42), 6);
    let mut state: State = LATENTS
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
    let mut tensors = BTreeMap::new();
    let mut steps = Vec::new();
    let times = schedule(config.ss_steps, config.ss_rescale_t);
    for (i, t) in times.windows(2).enumerate() {
        let prefix = format!("step{:02}", i + 1);
        let time = mx::from_slice_f32(&[(t[0] * 1000.0) as f32], &[1]);
        let dt = t[1] - t[0];
        let delta = mx::zeros(&[1], mx::dtype::FLOAT32);
        let guided = config.ss_guidance > 0.0 && t[0] <= 0.5;
        capture(&mut tensors, format!("{prefix}.time"), &time)?;
        capture(
            &mut tensors,
            format!("{prefix}.dt"),
            &mx::from_slice_f32(&[dt as f32], &[]),
        )?;
        for (name, value) in &state {
            capture(&mut tensors, format!("{prefix}.pre.{name}"), value)?;
        }
        for (name, value) in model.structure.timestep_diagnostics(&time, &delta) {
            capture(&mut tensors, format!("{prefix}.embedding.{name}"), &value)?;
        }
        let (velocity, uncond) = if guided {
            let (v, u) =
                model
                    .structure
                    .guided(&state, &time, &condition, &uncondition, Some(&delta))?;
            (v, Some(u))
        } else {
            (
                model
                    .structure
                    .forward(&state, &time, &condition, Some(&delta))?,
                None,
            )
        };
        for (name, value) in &velocity {
            capture(
                &mut tensors,
                format!("{prefix}.velocity.conditioned.{name}"),
                value,
            )?;
        }
        if let Some(u) = &uncond {
            for (name, value) in u {
                capture(
                    &mut tensors,
                    format!("{prefix}.velocity.unconditioned.{name}"),
                    value,
                )?;
            }
        }
        state = state
            .iter()
            .map(|(name, value)| {
                Ok((
                    name.clone(),
                    update(
                        value,
                        velocity.get(name).context("missing velocity")?,
                        uncond.as_ref().and_then(|u| u.get(name)).map(|u| &**u),
                        config.ss_guidance,
                        dt,
                    ),
                ))
            })
            .collect::<Result<State>>()?;
        for (name, value) in &state {
            capture(&mut tensors, format!("{prefix}.post.{name}"), value)?;
        }
        steps.push(json!({"step":i+1, "guided":guided,
            "t0_bits":format!("{:016x}", t[0].to_bits()),
            "t1_bits":format!("{:016x}", t[1].to_bits()),
            "dt_bits":format!("{:016x}", dt.to_bits())}));
        eprintln!(
            "sam3d_objects: diagnostic structure {}/{}",
            i + 1,
            config.ss_steps
        );
    }
    let shape = state.get("shape").context("missing final shape")?;
    capture(&mut tensors, "final.shape".into(), shape)?;
    let logits = model.occupancy.forward(shape)?;
    capture(&mut tensors, "decoder.logits".into(), &logits)?;
    let (grid, factor) = occupancy_grid(&logits, config.downsample_ss_dist)?;
    capture(&mut tensors, "decoder.coords".into(), &grid.coords)?;
    ensure!(
        tensors.len() == 673,
        "unexpected released sampling diagnostic schema"
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
    safetensors::serialize_to_file(views, None, &output.join("sampling.safetensors"))?;
    std::fs::write(
        output.join("sampling.json"),
        serde_json::to_vec_pretty(&json!({
            "purpose":"diagnostic-only", "case":"rgba-dropout", "seed":42,
            "structure_steps":config.ss_steps, "steps":steps, "occupancy_factor":factor,
            "runtime_device":runtime.device.to_string(), "tf32":"0", "input":input,
            "tensor_count":tensors.len(), "source_identity":"supplied by external frozen process receipt"
        }))?,
    )?;
    Ok(())
}
