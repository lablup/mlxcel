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

//! Opt-in parity against the published MoGe3 ViT-L and ViT-g checkpoints.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use mlxcel::vision::dense::{DenseInput, DenseOutput, DensePredictor};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

const SOURCE_REVISION: &str = "abd3b2d7b728cb5372d4e092260da325e5ded209";
const IMAGE_SHA256: &str = "2a7ba590d525fe31f5627c73b2e83a5b63b3cba3840fb10ae9d726187aca6769";
const NATIVE_SHA256: &str = "9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094";
const BINDING_SHA256: &str = "f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc";
const VITL_SHA256: &str = "d21ea545b5f760b0d34d6538902378f6daa9d4c79dc5843891bbebec7c4da87b";
const VITG_SHA256: &str = "91d3a142dd78437acec9fb386f1bb61e214e389b9bba02dd4cf6160117e200e5";

struct Case {
    manifest_key: &'static str,
    reference_env: &'static str,
    checkpoint_sha256: &'static str,
    requested_options: Value,
}

struct ModelPin {
    manifest_key: &'static str,
    repository: &'static str,
    revision: &'static str,
    model_sha256: &'static str,
}

fn sha256(path: &Path) -> String {
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).expect("read parity asset");
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    format!("{:x}", digest.finalize())
}

fn manifest() -> Value {
    serde_json::from_str(include_str!("fixtures/moge3/full-reference-manifest.json"))
        .expect("parse committed MoGe3 reference manifest")
}

fn checkpoint_manifest() -> Value {
    serde_json::from_str(include_str!("fixtures/moge3/checkpoints.json"))
        .expect("parse committed MoGe3 checkpoint manifest")
}

fn authenticate_checkpoint(model_dir: &Path, pin: &ModelPin) {
    let manifest = checkpoint_manifest();
    let receipt = &manifest[pin.manifest_key];
    assert_eq!(receipt["repository"], pin.repository);
    assert_eq!(receipt["revision"], pin.revision);
    assert_eq!(
        receipt["files"]["model.safetensors"]["sha256"],
        pin.model_sha256
    );
    for filename in [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ] {
        let expected = receipt["files"][filename]["sha256"]
            .as_str()
            .unwrap_or_else(|| panic!("missing {} {filename} hash", pin.manifest_key));
        let expected_bytes = receipt["files"][filename]["bytes"]
            .as_u64()
            .unwrap_or_else(|| panic!("missing {} {filename} size", pin.manifest_key));
        let path = model_dir.join(filename);
        assert_eq!(
            path.metadata().expect("checkpoint file metadata").len(),
            expected_bytes,
            "MoGe3 {} {filename} size differs from the published checkpoint pin",
            pin.manifest_key
        );
        assert_eq!(
            sha256(&path),
            expected,
            "MoGe3 {} {filename} differs from the published checkpoint pin",
            pin.manifest_key
        );
    }
}

fn authenticate_reference(case: &Case, path: &Path) {
    let manifest = manifest();
    let receipt = &manifest[case.manifest_key];
    assert_eq!(receipt["options"], case.requested_options);
    assert_eq!(receipt["source_revision"], SOURCE_REVISION);
    assert_eq!(receipt["image_sha256"], IMAGE_SHA256);
    assert_eq!(receipt["tf32"], "0");
    assert_eq!(receipt["checkpoint_sha256"], case.checkpoint_sha256);
    assert_eq!(receipt["runtime_receipt"]["version"], "0.32.5");
    assert_eq!(receipt["runtime_receipt"]["native_sha256"], NATIVE_SHA256);
    assert_eq!(receipt["runtime_receipt"]["binding_sha256"], BINDING_SHA256);
    let expected_device = if mlxcel_core::default_device_is_gpu() {
        "Device(gpu, 0)"
    } else {
        "Device(cpu, 0)"
    };
    assert_eq!(receipt["device"], expected_device);
    assert_eq!(
        sha256(path),
        receipt["output_sha256"]
            .as_str()
            .expect("manifest output hash")
    );
    assert_eq!(
        sha256(Path::new("tests/fixtures/sam3/cats.png")),
        IMAGE_SHA256
    );
}

fn predictor_options(requested: &Value) -> Map<String, Value> {
    let mut options = requested
        .as_object()
        .expect("requested options map")
        .clone();
    options.retain(|_, value| !value.is_null());
    options.insert("task".into(), Value::String("geometry".into()));
    options
}

fn predict(predictor: &dyn DensePredictor, case: &Case) -> DenseOutput {
    assert_eq!(
        std::env::var("MLX_ENABLE_TF32").as_deref(),
        Ok("0"),
        "real MoGe3 parity requires MLX_ENABLE_TF32=0 before process startup"
    );
    let image = image::open("tests/fixtures/sam3/cats.png").unwrap();
    predictor
        .predict(
            &DenseInput::Image(image),
            &predictor_options(&case.requested_options),
        )
        .unwrap()
}

fn load_reference(case: &Case) -> mlxcel_core::weights::WeightMap {
    let path =
        std::env::var(case.reference_env).unwrap_or_else(|_| panic!("set {}", case.reference_env));
    let path = Path::new(&path);
    authenticate_reference(case, path);
    mlxcel_core::weights::load_safetensors(path).expect("load authentic MoGe3 reference")
}

fn f32_values(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32))
}

fn value_class(value: f32) -> u8 {
    if value.is_nan() {
        1
    } else if value.is_infinite() && value.is_sign_positive() {
        2
    } else if value.is_infinite() {
        3
    } else {
        4
    }
}

fn relative_rms(actual: &[f32], expected: &[f32]) -> f64 {
    let (error, scale, count) = actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual.is_finite() && expected.is_finite())
        .fold((0.0, 0.0, 0usize), |(error, scale, count), (a, e)| {
            let a = f64::from(*a);
            let e = f64::from(*e);
            (error + (a - e).powi(2), scale + e.powi(2), count + 1)
        });
    assert!(count > 0, "reference comparison has no finite values");
    (error / count as f64).sqrt() / (scale / count as f64).sqrt().max(f64::MIN_POSITIVE)
}

fn assert_parity(output: &DenseOutput, reference: &mlxcel_core::weights::WeightMap) {
    let dense_threshold = if mlxcel_core::default_device_is_gpu() {
        5e-3
    } else {
        1e-4
    };
    let actual_keys: BTreeSet<&str> = output.tensors.keys().map(String::as_str).collect();
    let expected_keys: BTreeSet<&str> = reference.keys().map(String::as_str).collect();
    assert_eq!(actual_keys, expected_keys, "MoGe3 output key set changed");
    for key in expected_keys {
        let actual = &output.tensors[key];
        let expected = &reference[key];
        assert_eq!(
            mlxcel_core::array_shape(actual),
            mlxcel_core::array_shape(expected),
            "MoGe3 output shape differs for {key}"
        );
        if key == "mask" {
            assert_eq!(mlxcel_core::array_dtype(actual), mlxcel_core::dtype::BOOL);
            assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
                actual, expected, false
            )));
            continue;
        }
        assert_eq!(
            mlxcel_core::array_dtype(actual),
            mlxcel_core::dtype::FLOAT32,
            "MoGe3 must preserve F32 output for {key}"
        );
        let actual_values = f32_values(actual);
        let expected_values = f32_values(expected);
        let actual_classes: Vec<u8> = actual_values.iter().copied().map(value_class).collect();
        let expected_classes: Vec<u8> = expected_values.iter().copied().map(value_class).collect();
        assert_eq!(
            actual_classes, expected_classes,
            "MoGe3 NaN/infinity/finite support differs for {key}"
        );
        assert!(
            actual_values.iter().any(|value| value.is_finite()),
            "MoGe3 {key} has no finite values"
        );
        if key.starts_with("intrinsics") {
            let max_abs = actual_values
                .iter()
                .zip(&expected_values)
                .filter(|(actual, expected)| actual.is_finite() && expected.is_finite())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs <= 1e-3, "{key} max error {max_abs} exceeds 1e-3");
            eprintln!("MoGe3 {key}: max_abs={max_abs:.9}");
        } else {
            let error = relative_rms(&actual_values, &expected_values);
            assert!(
                error <= dense_threshold,
                "{key} relative RMS {error} exceeds {dense_threshold}"
            );
            eprintln!("MoGe3 {key}: relative_rms={error:.9}");
        }
    }
}

fn run(model_env: &str, pin: ModelPin, cases: &[Case]) {
    let model = std::env::var(model_env).unwrap_or_else(|_| panic!("set {model_env}"));
    authenticate_checkpoint(Path::new(&model), &pin);
    let predictor = mlxcel::vision::dense::moge3::load(model.as_ref()).unwrap();
    for case in cases {
        let output = predict(predictor.as_ref(), case);
        let reference = load_reference(case);
        assert_parity(&output, &reference);
    }
}

fn default_options() -> Value {
    json!({
        "resolution_level": 9,
        "num_tokens": null,
        "refine_steps": 3,
        "return_per_step": true,
        "force_projection": true,
        "apply_mask": true,
        "fov_x": null
    })
}

#[test]
#[ignore = "requires published ViT-L checkpoint and authentic references"]
fn moge3_vitl_real_checkpoint_parity_all_options() {
    run(
        "MOGE3_VITL_MODEL",
        ModelPin {
            manifest_key: "vitl",
            repository: "mlx-community/moge-3-vitl-mlx-fp32",
            revision: "f5ceca7512506731fb1c50307c313da76aac500f",
            model_sha256: VITL_SHA256,
        },
        &[
            Case {
                manifest_key: "vitl-gpu-resolution9",
                reference_env: "MOGE3_VITL_REFERENCE",
                checkpoint_sha256: VITL_SHA256,
                requested_options: default_options(),
            },
            Case {
                manifest_key: "vitl-no-refine",
                reference_env: "MOGE3_VITL_NO_REFINE_REFERENCE",
                checkpoint_sha256: VITL_SHA256,
                requested_options: json!({"resolution_level":9,"num_tokens":null,"refine_steps":0,"return_per_step":false,"force_projection":false,"apply_mask":true,"fov_x":null}),
            },
            Case {
                manifest_key: "vitl-fixed-unmasked",
                reference_env: "MOGE3_VITL_FIXED_UNMASKED_REFERENCE",
                checkpoint_sha256: VITL_SHA256,
                requested_options: json!({"resolution_level":9,"num_tokens":null,"refine_steps":3,"return_per_step":true,"force_projection":false,"apply_mask":false,"fov_x":60}),
            },
            Case {
                manifest_key: "vitl-fixed-projected",
                reference_env: "MOGE3_VITL_FIXED_PROJECTED_REFERENCE",
                checkpoint_sha256: VITL_SHA256,
                requested_options: json!({"resolution_level":0,"num_tokens":3600,"refine_steps":0,"return_per_step":false,"force_projection":true,"apply_mask":false,"fov_x":60}),
            },
        ],
    );
}

#[test]
#[ignore = "requires published ViT-g checkpoint and authentic reference"]
fn moge3_vitg_resolution9_real_checkpoint_parity() {
    run(
        "MOGE3_VITG_MODEL",
        ModelPin {
            manifest_key: "vitg",
            repository: "mlx-community/moge-3-vitg-mlx-fp32",
            revision: "c437fec8ea9e39ad27ea82367fd4df8b8c0619ce",
            model_sha256: VITG_SHA256,
        },
        &[Case {
            manifest_key: "vitg-gpu-resolution9",
            reference_env: "MOGE3_VITG_REFERENCE",
            checkpoint_sha256: VITG_SHA256,
            requested_options: default_options(),
        }],
    );
}
