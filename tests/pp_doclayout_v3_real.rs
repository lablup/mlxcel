// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Ignored real CLI acceptance for PP-DocLayoutV3 -> Falcon-OCR.

use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const FIXTURE_DIR: &str = "tests/fixtures/pp_doclayout_v3";
const HASHNUKE_ACTIVE_FILES: &[&str] = &["config.json", "model.safetensors"];
const PADDLE_ACTIVE_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
];

#[derive(Deserialize)]
struct Acceptance {
    image_sha256: String,
    falcon_checkpoint_revision: String,
    falcon_model_sha256: String,
    falcon_checkpoint_manifest_sha256: String,
    anchors: Vec<String>,
    detector_expected_count: usize,
    detector_reading_order: Vec<u32>,
}

#[derive(Deserialize)]
struct CheckpointManifest {
    revision: String,
    files: Vec<ManifestFile>,
}

#[derive(Deserialize)]
struct ManifestFile {
    name: String,
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct DetectorOutput {
    detections: Vec<CliDetection>,
}

#[derive(Deserialize)]
struct CliDetection {
    label: String,
    label_id: usize,
    confidence: f32,
    reading_order: u32,
    #[serde(rename = "box")]
    box_: CliBox,
}

#[derive(Deserialize)]
struct CliBox {
    l: f32,
    t: f32,
    r: f32,
    b: f32,
}

#[derive(Deserialize)]
struct RawExpected {
    results: std::collections::BTreeMap<String, ExpectedLayout>,
}

#[derive(Deserialize)]
struct ExpectedLayout {
    rust_expected_detections: Vec<ExpectedDetection>,
}

#[derive(Deserialize)]
struct ExpectedDetection {
    bbox: [f32; 4],
    score: f32,
    label: usize,
    class_name: String,
    reading_order: u32,
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIR)
        .join(name)
}

fn required_path(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {variable} for the real PP-DocLayoutV3/Falcon-OCR test"))
}

fn sha256(path: &Path) -> String {
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|error| panic!("failed to open {} for SHA-256: {error}", path.display()));
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("failed to hash {}: {error}", path.display()));
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    format!("{:x}", digest.finalize())
}

fn authenticate_checkpoint(
    directory: &Path,
    manifest_path: &Path,
    required: &[&str],
) -> CheckpointManifest {
    let manifest: CheckpointManifest =
        serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
    assert_eq!(
        directory.file_name().and_then(|name| name.to_str()),
        Some(manifest.revision.as_str()),
        "checkpoint directory must be the pinned manifest revision"
    );
    for required_name in required {
        assert!(
            is_plain_file_name(required_name),
            "required checkpoint input must be a plain filename"
        );
        let matching: Vec<_> = manifest
            .files
            .iter()
            .filter(|file| file.name == *required_name)
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "checkpoint manifest must contain {required_name} exactly once"
        );
        let expected = matching[0];
        assert!(
            is_plain_file_name(&expected.name),
            "manifest checkpoint filename must not escape its directory"
        );
        let actual_path = directory.join(&expected.name);
        let metadata = std::fs::metadata(&actual_path)
            .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", actual_path.display()));
        assert_eq!(
            metadata.len(),
            expected.bytes,
            "{} byte count",
            expected.name
        );
        assert_eq!(
            sha256(&actual_path),
            expected.sha256,
            "{} SHA-256",
            expected.name
        );
    }
    manifest
}

fn is_plain_file_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

fn run(mut command: Command, name: &str, artifact_dir: &Path, artifact_stem: &str) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {name}: {error}"));
    std::fs::write(
        artifact_dir.join(format!("{artifact_stem}.stdout")),
        &output.stdout,
    )
    .unwrap();
    std::fs::write(
        artifact_dir.join(format!("{artifact_stem}.stderr")),
        &output.stderr,
    )
    .unwrap();
    let status = output.status.code().map_or_else(
        || "terminated_by_signal\n".to_string(),
        |code| format!("exit_code={code}\n"),
    );
    std::fs::write(artifact_dir.join(format!("{artifact_stem}.status")), status).unwrap();
    assert!(
        output.status.success(),
        "{name} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn assert_bound_hip_device(output: &Output, name: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("[mlx-rocm] bound HIP device"),
        "{name} did not report an actual bound HIP device:\n{stderr}"
    );
}

fn assert_detector_semantics(actual: &[CliDetection], expected: &[ExpectedDetection]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.label_id, expected.label,
            "detection {index} label id"
        );
        assert_eq!(actual.label, expected.class_name, "detection {index} label");
        assert_eq!(
            actual.reading_order, expected.reading_order,
            "detection {index} order"
        );
        assert!(
            (actual.confidence - expected.score).abs() <= 1.0e-3,
            "detection {index} score: {} != {}",
            actual.confidence,
            expected.score
        );
        for (axis, value) in [actual.box_.l, actual.box_.t, actual.box_.r, actual.box_.b]
            .into_iter()
            .enumerate()
        {
            assert!(
                (value - expected.bbox[axis]).abs() <= 2.0,
                "detection {index} bbox[{axis}]: {value} != {}",
                expected.bbox[axis]
            );
        }
    }
}

#[test]
#[ignore = "requires both released PP-DocLayoutV3 checkpoints, Falcon-OCR, and a GPU"]
fn real_cli_preserves_two_column_model_order_into_falcon_ocr() {
    let binary = env!("CARGO_BIN_EXE_mlxcel");
    let detectors = [
        (
            "hashnuke",
            required_path("MLXCEL_TEST_PP_DOCLAYOUT_HASHNUKE_DIR"),
            fixture("hashnuke-checkpoint-manifest.json"),
            HASHNUKE_ACTIVE_FILES,
        ),
        (
            "paddle",
            required_path("MLXCEL_TEST_PP_DOCLAYOUT_PADDLE_DIR"),
            fixture("paddle-checkpoint-manifest.json"),
            PADDLE_ACTIVE_FILES,
        ),
    ];
    let falcon = required_path("MLXCEL_TEST_FALCON_OCR_DIR");
    let page = fixture("two_column_page.png");
    let acceptance: Acceptance =
        serde_json::from_slice(&std::fs::read(fixture("falcon_ocr_expected.json")).unwrap())
            .unwrap();
    assert_eq!(sha256(&page), acceptance.image_sha256);
    assert_eq!(
        sha256(&fixture("falcon-checkpoint-manifest.json")),
        acceptance.falcon_checkpoint_manifest_sha256
    );
    let falcon_manifest = authenticate_checkpoint(
        &falcon,
        &fixture("falcon-checkpoint-manifest.json"),
        &[
            "config.json",
            "model.safetensors",
            "model_args.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    );
    assert_eq!(
        falcon_manifest.revision,
        acceptance.falcon_checkpoint_revision
    );
    assert_eq!(
        sha256(&falcon.join("model.safetensors")),
        acceptance.falcon_model_sha256
    );

    let expected: RawExpected =
        serde_json::from_slice(&std::fs::read(fixture("expected_detections.json")).unwrap())
            .unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let artifact_dir = match std::env::var_os("MLXCEL_TEST_PP_DOCLAYOUT_ARTIFACT_DIR") {
        Some(path) => {
            let path = PathBuf::from(path);
            std::fs::create_dir(&path).unwrap_or_else(|error| {
                panic!(
                    "MLXCEL_TEST_PP_DOCLAYOUT_ARTIFACT_DIR must name a fresh directory ({}): {error}",
                    path.display()
                )
            });
            path
        }
        None => temporary.path().to_path_buf(),
    };
    let mut layout_path = None;
    for (layout, detector, manifest, active_files) in detectors {
        authenticate_checkpoint(&detector, &manifest, active_files);
        let mut detect = Command::new(binary);
        detect
            .env("MLXCEL_DEVICE", "gpu")
            .env("MLX_ENABLE_TF32", "0")
            .args(["detect", "--format", "json", "--threshold", "0.3"])
            .arg("--model")
            .arg(&detector)
            .arg("--image")
            .arg(&page);
        let artifact_stem = format!("detect-{layout}");
        let detected = run(
            detect,
            &format!("mlxcel detect ({layout})"),
            &artifact_dir,
            &artifact_stem,
        );
        assert_bound_hip_device(&detected, &format!("mlxcel detect ({layout})"));
        let path = artifact_dir.join(format!("{artifact_stem}.stdout"));
        eprintln!(
            "PP-DocLayoutV3 actual_layout={layout} actual_device=hip actual_layout_json_sha256={:x}",
            Sha256::digest(&detected.stdout)
        );
        let parsed: DetectorOutput = serde_json::from_slice(&detected.stdout).unwrap();
        assert_eq!(parsed.detections.len(), acceptance.detector_expected_count);
        assert_eq!(
            parsed
                .detections
                .iter()
                .map(|detection| detection.reading_order)
                .collect::<Vec<_>>(),
            acceptance.detector_reading_order
        );
        assert_detector_semantics(
            &parsed.detections,
            &expected.results[layout].rust_expected_detections,
        );
        if layout == "hashnuke" {
            layout_path = Some(path);
        }
    }
    let layout_path = layout_path.expect("HashNuke layout output");

    let mut generate = Command::new(binary);
    generate
        .env("MLXCEL_DEVICE", "gpu")
        .env("MLX_ENABLE_TF32", "0")
        .args(["generate", "--temp", "0", "--max-tokens", "512"])
        .arg("--model")
        .arg(&falcon)
        .arg("--image")
        .arg(&page)
        .arg("--layout-detections")
        .arg(&layout_path);
    let generated = run(
        generate,
        "mlxcel generate --layout-detections",
        &artifact_dir,
        "falcon-ocr",
    );
    assert_bound_hip_device(&generated, "mlxcel generate --layout-detections");
    let transcript = String::from_utf8(generated.stdout).unwrap();
    let normalized = transcript
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let mut previous = None;
    for anchor in &acceptance.anchors {
        let position = normalized
            .find(&anchor.to_lowercase())
            .unwrap_or_else(|| panic!("Falcon-OCR transcript is missing source anchor {anchor:?}"));
        if let Some(previous) = previous {
            assert!(
                position > previous,
                "source anchors are not in model reading order"
            );
        }
        previous = Some(position);
    }
    eprintln!(
        "Falcon-OCR actual_device=hip actual_transcript_sha256={:x}",
        Sha256::digest(transcript.as_bytes())
    );
}
