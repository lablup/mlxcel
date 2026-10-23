// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Opt-in genuine predictor capture. The external driver authenticates source,
//! executable, checkpoint, input, invocation and process exits before comparison.
use anyhow::{Context, Result, ensure};
use mlxcel::vision::dense::{DenseInput, DenseOutputInfo, load_dense_model, write_dense_output};
use mlxcel_core as mx;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};
fn path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("set {name}"))
}
fn hash(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = [0; 1024 * 1024];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
#[test]
#[ignore = "requires authenticated real checkpoint/input and an exclusive GPU slot"]
fn sam3d_body_real_model_capture() -> Result<()> {
    let model = path("BODY_MODEL_DIR")?;
    let image_path = path("BODY_IMAGE")?;
    let out = path("BODY_CAPTURE_DIR")?;
    ensure!(!out.exists(), "capture output must be fresh");
    ensure!(
        std::env::var("MLX_ENABLE_TF32").as_deref() == Ok("0"),
        "Body parity requires TF32=0"
    );
    let options: Value =
        serde_json::from_str(&std::env::var("BODY_OPTIONS_JSON").unwrap_or_else(|_| "{}".into()))?;
    let options = options
        .as_object()
        .context("Body options must be an object")?;
    let runtime = mlxcel::initialize_runtime_checked()?;
    eprintln!("Runtime device: {}", runtime.device);
    ensure!(
        runtime.device == mlxcel::RuntimeDevice::Gpu,
        "Body real parity requires a resolved GPU"
    );
    let image = image::open(&image_path)?;
    let size = (image.width(), image.height());
    let predictor = load_dense_model(&model)?;
    let start = Instant::now();
    let output = predictor.predict(&DenseInput::Image(image), options)?;
    let elapsed = start.elapsed().as_secs_f64();
    ensure!(
        output.tensors.len() == 4,
        "Body tensor set must contain exactly four outputs"
    );
    for (name, shape) in [
        ("vertices", vec![18439, 3]),
        ("keypoints_3d", vec![70, 3]),
        ("joint_coords", vec![127, 3]),
        ("camera", vec![3]),
    ] {
        let t = output
            .tensors
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        ensure!(
            mx::array_shape(t) == shape && mx::array_dtype(t) == mx::dtype::FLOAT32,
            "Body output schema mismatch {name}"
        );
        ensure!(
            mx::item_bool(&mx::all_all(&mx::isfinite(t))),
            "non-finite {name}"
        );
    }
    ensure!(
        output.meshes.len() == 1
            && output.meshes[0].1.vertices.len() == 18439
            && output.meshes[0].1.faces.len() == 36874
            && output.meshes[0].1.vertex_colors.is_none(),
        "Body mesh schema mismatch"
    );
    ensure!(
        output.keypoints.is_empty() && output.instances.is_empty(),
        "Body must not populate 2D keypoints or detections"
    );
    write_dense_output(
        &out,
        &output,
        &DenseOutputInfo {
            family: "sam3d_body",
            task: "body_mesh",
            input_size: size,
            elapsed_ms: elapsed * 1000.0,
            is_video: false,
        },
    )?;
    let mut artifacts = serde_json::Map::new();
    for name in [
        "result.safetensors",
        "mesh.obj",
        "meta.json",
        "instances.json",
    ] {
        artifacts.insert(name.into(),json!({"sha256":hash(&out.join(name))?,"bytes":std::fs::metadata(out.join(name))?.len()}));
    }
    let executable = std::env::current_exe()?;
    let receipt = json!({"schema":1,"family":"sam3d_body","execution_api":"DensePredictor.predict","runtime_device":runtime.device.to_string(),"tf32":"0","options":options,"model":model,"image":image_path,"image_sha256":hash(&image_path)?,"executable":executable,"executable_sha256":hash(&executable)?,"elapsed_seconds":elapsed,"metadata":output.meta,"artifacts":artifacts});
    std::fs::write(out.join("run.json"), serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}
