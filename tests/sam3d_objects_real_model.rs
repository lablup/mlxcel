// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Opt-in full-default Objects capture; comparison uses the authenticated Python fixture harness.
use anyhow::{Context, Result, ensure};
use mlxcel::vision::dense::{
    DenseInput, DenseOutputInfo, load_dense_model, predictor::DenseSidecars, write_dense_output,
};
use mlxcel_core as mx;
use safetensors::{Dtype, tensor::TensorView};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};
fn hash(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut b = [0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn env_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("set {name}"))
}
struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
fn f32_tensor(shape: Vec<usize>, values: impl IntoIterator<Item = f32>) -> Tensor {
    Tensor {
        dtype: Dtype::F32,
        shape,
        bytes: values.into_iter().flat_map(f32::to_le_bytes).collect(),
    }
}

#[test]
#[ignore = "requires authenticated checkpoint/input assets and an exclusive GPU slot"]
fn sam3d_objects_full_default_capture() -> Result<()> {
    let model = env_path("MLXCEL_SAM3D_OBJECTS_MODEL")?;
    let inputs = env_path("MLXCEL_SAM3D_OBJECTS_INPUT_DIR")?;
    let out = env_path("MLXCEL_SAM3D_OBJECTS_OUTPUT_DIR")?;
    let source_manifest = env_path("MLXCEL_SAM3D_OBJECTS_SOURCE_MANIFEST")?;
    let case = std::env::var("MLXCEL_SAM3D_OBJECTS_CASE").unwrap_or_else(|_| "default".into());
    ensure!(
        matches!(case.as_str(), "default" | "rgba-dropout"),
        "unknown Objects capture case"
    );
    ensure!(!out.exists(), "refuse to overwrite previous model capture");
    ensure!(
        std::env::var("MLX_ENABLE_TF32").as_deref() == Ok("0"),
        "disable TF32 for authenticated parity"
    );
    let canonical: Value =
        serde_json::from_str(include_str!("fixtures/sam3d_objects/provenance.json"))?;
    for (name, expected) in canonical["checkpoint"]["files"]
        .as_object()
        .context("checkpoint manifest")?
    {
        let path = model.join(name);
        ensure!(
            std::fs::metadata(&path)?.len()
                == expected["bytes"].as_u64().context("checkpoint length")?
                && hash(&path)? == expected["sha256"].as_str().context("checkpoint hash")?,
            "checkpoint authentication failed: {name}"
        );
    }
    let image_name = if case == "default" {
        "object.png"
    } else {
        "object-rgba.png"
    };
    let names = if case == "default" {
        vec![image_name, "mask.png"]
    } else {
        vec![image_name]
    };
    let mut input_hashes = BTreeMap::new();
    for name in names {
        let actual = hash(&inputs.join(name))?;
        ensure!(
            Some(actual.as_str()) == canonical["inputs"][name]["sha256"].as_str(),
            "input hash mismatch {name}"
        );
        input_hashes.insert(name, actual);
    }
    let manifest: BTreeMap<String, String> =
        serde_json::from_reader(File::open(&source_manifest)?)?;
    for (name, expected) in &manifest {
        ensure!(
            &hash(&Path::new(env!("CARGO_MANIFEST_DIR")).join(name))? == expected,
            "source changed since frozen build manifest: {name}"
        );
    }
    let executable = std::env::current_exe()?;
    let executable_hash = hash(&executable)?;
    let runtime = mlxcel::initialize_runtime_checked()?;
    eprintln!("Runtime device: {}", runtime.device);
    ensure!(
        runtime.device == mlxcel::RuntimeDevice::Gpu,
        "Objects parity requires a resolved GPU device"
    );
    let image = image::open(inputs.join(image_name))?;
    let size = (image.width(), image.height());
    let mask = if case == "default" {
        Some(image::open(inputs.join("mask.png"))?)
    } else {
        None
    };
    let sidecars = DenseSidecars {
        mask,
        pointmap: None,
    };
    let options = json!({"seed":42,"ss_steps":25,"slat_steps":25,"formats":["gaussian","mesh"],"estimate_depth":case=="default"});
    let predictor = load_dense_model(&model)?;
    let start = Instant::now();
    let output = predictor.predict_with_sidecars(
        &DenseInput::Image(image),
        &sidecars,
        options.as_object().context("options object")?,
    )?;
    let elapsed = start.elapsed().as_secs_f64();
    let required = [
        "coords",
        "latents",
        "pointmap_conditioned",
        "pose.rotation",
        "pose.rotation_matrix",
        "pose.scale",
        "pose.translation",
    ];
    ensure!(
        output.tensors.len() == required.len()
            && required.iter().all(|n| output.tensors.contains_key(*n)),
        "compact output keys mismatch"
    );
    ensure!(
        mx::array_shape(&output.tensors["pointmap_conditioned"]) == [1]
            && mx::array_dtype(&output.tensors["pointmap_conditioned"]) == mx::dtype::BOOL,
        "conditioning schema must be BOOL[1]"
    );
    ensure!(
        mx::array_dtype(&output.tensors["coords"]) == mx::dtype::INT32,
        "coords must be int32"
    );
    let conditioned = mx::item_bool(&output.tensors["pointmap_conditioned"]);
    ensure!(
        conditioned == (case == "default"),
        "wrong conditioning path"
    );
    let info = DenseOutputInfo {
        family: "sam3d_objects",
        task: "mesh",
        input_size: size,
        elapsed_ms: elapsed * 1000.0,
        is_video: false,
    };
    write_dense_output(&out, &output, &info)?;
    let compact_bytes = std::fs::read(out.join("result.safetensors"))?;
    let compact = safetensors::SafeTensors::deserialize(&compact_bytes)?;
    ensure!(
        compact.tensor("pointmap_conditioned")?.dtype() == Dtype::BOOL
            && compact.tensor("pointmap_conditioned")?.shape() == [1],
        "serialized conditioning schema mismatch"
    );
    ensure!(
        compact.tensor("coords")?.dtype() == Dtype::I32,
        "serialized coords lost integer dtype"
    );
    drop(compact_bytes);
    let mut tensors = BTreeMap::new();
    for (name, array) in &output.tensors {
        if name == "pointmap_conditioned" {
            continue;
        }
        let shape = mx::array_shape(array).iter().map(|v| *v as usize).collect();
        let tensor = if name == "coords" {
            Tensor {
                dtype: Dtype::I32,
                shape,
                bytes: mx::array_to_raw_bytes(array),
            }
        } else {
            let v = mx::utils::array_to_vec_f32(array);
            ensure!(v.iter().all(|v| v.is_finite()), "nonfinite {name}");
            f32_tensor(shape, v)
        };
        tensors.insert(name.clone(), tensor);
    }
    for (name, mesh) in &output.meshes {
        mesh.validate()?;
        ensure!(
            !mesh.vertices.is_empty() && !mesh.faces.is_empty(),
            "empty mesh acceptance output"
        );
        tensors.insert(
            format!("{name}.vertices"),
            f32_tensor(
                vec![mesh.vertices.len(), 3],
                mesh.vertices.iter().flatten().copied(),
            ),
        );
        tensors.insert(
            format!("{name}.vertex_colors"),
            f32_tensor(
                vec![
                    mesh.vertex_colors.as_ref().expect("Objects colors").len(),
                    6,
                ],
                mesh.vertex_colors
                    .as_ref()
                    .expect("Objects colors")
                    .iter()
                    .flatten()
                    .copied(),
            ),
        );
        tensors.insert(
            format!("{name}.faces"),
            Tensor {
                dtype: Dtype::U32,
                shape: vec![mesh.faces.len(), 3],
                bytes: mesh
                    .faces
                    .iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            },
        );
    }
    for (name, g) in &output.gaussians {
        g.validate()?;
        ensure!(!g.positions.is_empty(), "empty Gaussian acceptance output");
        for (key, values) in [
            ("positions", &g.positions),
            ("sh_dc", &g.sh_dc),
            ("scales", &g.scales),
        ] {
            tensors.insert(
                format!("{name}.{key}"),
                f32_tensor(vec![values.len(), 3], values.iter().flatten().copied()),
            );
        }
        tensors.insert(
            format!("{name}.rotations"),
            f32_tensor(
                vec![g.rotations.len(), 4],
                g.rotations.iter().flatten().copied(),
            ),
        );
        tensors.insert(
            format!("{name}.opacities"),
            f32_tensor(vec![g.opacities.len(), 1], g.opacities.iter().copied()),
        );
    }
    ensure!(
        tensors.len() == 14,
        "full parity capture must contain all 14 tensors"
    );
    let views = tensors
        .iter()
        .map(|(name, t)| {
            Ok((
                name.as_str(),
                TensorView::new(t.dtype, t.shape.clone(), &t.bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize_to_file(views, None, &out.join("full.safetensors"))?;
    let artifacts = [
        "full.safetensors",
        "result.safetensors",
        "mesh.obj",
        "gaussian.ply",
        "meta.json",
    ]
    .into_iter()
    .map(|n| Ok((n, hash(&out.join(n))?)))
    .collect::<Result<BTreeMap<_, _>>>()?;
    let receipt = json!({"case":case,"runtime_device":runtime.device.to_string(),"tf32":"0","options":options,"checkpoint":canonical["checkpoint"],"upstream_source_revision":canonical["source_revision"],"executable":executable,"executable_sha256":executable_hash,"source_manifest_sha256":hash(&source_manifest)?,"source_files":manifest,"inputs":input_hashes,"pointmap_conditioned":conditioned,"elapsed_seconds":elapsed,"artifact_sha256":artifacts,"outputs":tensors.iter().map(|(n,t)|(n,json!({"shape":t.shape,"dtype":format!("{:?}",t.dtype)}))).collect::<BTreeMap<_,_>>()});
    serde_json::to_writer_pretty(File::create(out.join("run.json"))?, &receipt)?;
    Ok(())
}
