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
    // Retain handles only: evaluating a velocity or one state field here changes
    // the native graph boundary and invalidates comparison with the real sampler.
    let mut deferred: BTreeMap<String, Array> = state
        .iter()
        .map(|(name, value)| (format!("step01.pre.{name}"), mx::copy(value)))
        .collect();
    let times = schedule(config.ss_steps, config.ss_rescale_t);
    for (i, t) in times.windows(2).enumerate() {
        let time = mx::from_slice_f32(&[(t[0] * 1000.0) as f32], &[1]);
        let delta = mx::zeros(&[1], mx::dtype::FLOAT32);
        let guided = config.ss_guidance > 0.0 && t[0] <= 0.5;
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
                        t[1] - t[0],
                    ),
                ))
            })
            .collect::<Result<State>>()?;
        submit_state(&state);
        for (name, value) in &state {
            deferred.insert(format!("step{:02}.post.{name}", i + 1), mx::copy(value));
        }
        eprintln!(
            "sam3d_objects: diagnostic structure {}/{}",
            i + 1,
            config.ss_steps
        );
    }
    let shape = state.get("shape").context("missing final shape")?;
    let logits = model.occupancy.forward(shape)?;
    mx::async_eval(&logits);
    let (grid, factor) = occupancy_grid(&logits, config.downsample_ss_dist)?;
    deferred.insert("final.shape".into(), mx::copy(shape));
    deferred.insert("decoder.logits".into(), mx::copy(&logits));
    deferred.insert("decoder.coords".into(), mx::copy(&grid.coords));
    let mut tensors = BTreeMap::new();
    for (name, value) in deferred {
        capture(&mut tensors, name, &value)?;
    }
    // These scalars describe the same host schedule, reconstructed only after
    // the trajectory so diagnostic graph nodes cannot perturb model execution.
    let mut steps = Vec::new();
    for (i, t) in times.windows(2).enumerate() {
        let dt = t[1] - t[0];
        capture(
            &mut tensors,
            format!("step{:02}.time", i + 1),
            &mx::from_slice_f32(&[(t[0] * 1000.0) as f32], &[1]),
        )?;
        capture(
            &mut tensors,
            format!("step{:02}.dt", i + 1),
            &mx::from_slice_f32(&[dt as f32], &[]),
        )?;
        steps.push(
            json!({"step":i+1, "guided":config.ss_guidance > 0.0 && t[0] <= 0.5,
            "t0_bits":format!("{:016x}",t[0].to_bits()),
            "t1_bits":format!("{:016x}",t[1].to_bits()),
            "dt_bits":format!("{:016x}",dt.to_bits())}),
        );
    }
    ensure!(
        tensors.len() == 183,
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
            "time_dt_observation":"reconstructed-after-trajectory", "materialization":"deferred-until-after-occupancy",
            "runtime_device":runtime.device.to_string(), "tf32":"0", "input":input,
            "tensor_count":tensors.len(), "source_identity":"supplied by external frozen process receipt"
        }))?,
    )?;
    Ok(())
}
