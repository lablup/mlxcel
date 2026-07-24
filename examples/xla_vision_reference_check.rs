// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Ordered first-divergence gate for the native IREE LLaVA vision path.
//!
//! This intentionally consumes issue #862's independently generated HF capture
//! rather than regenerating an oracle from the implementation under test.

use std::fs;
use std::path::{Path, PathBuf};

use mlxcel_xla::IreeVisionDiagnosticProjector;
use serde_json::{Value, json};

const BLOCK0_STAGES: [&str; 12] = [
    "vision_block0_layer_norm1",
    "vision_block0_q_proj",
    "vision_block0_k_proj",
    "vision_block0_v_proj",
    "vision_block0_attention_context",
    "vision_block0_attention_output",
    "vision_block0_attention_residual",
    "vision_block0_layer_norm2",
    "vision_block0_mlp_fc1",
    "vision_block0_mlp_activation",
    "vision_block0_mlp_fc2",
    "vision_block0_output",
];

fn argument(flag: &str) -> Option<String> {
    let args = std::env::args().collect::<Vec<_>>();
    args.iter()
        .position(|argument| argument == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn required_path(flag: &str) -> PathBuf {
    argument(flag)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("missing required {flag}"))
}

fn f32_file(path: &Path) -> Vec<f32> {
    fs::read(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect()
}

fn array<'a>(manifest: &'a Value, case: &str, stage: &str) -> &'a Value {
    manifest["cases"]
        .as_array()
        .expect("manifest cases must be an array")
        .iter()
        .find(|entry| entry["name"] == case)
        .unwrap_or_else(|| panic!("reference has no case {case:?}"))["arrays"]
        .get(stage)
        .unwrap_or_else(|| panic!("reference case {case:?} has no stage {stage:?}"))
}

fn tolerance(manifest: &Value, stage: &str) -> (f64, f64) {
    let value = &manifest["tolerances"]["float32"][stage];
    (
        value["atol"]
            .as_f64()
            .unwrap_or_else(|| panic!("missing float32 {stage} atol")),
        value["rtol"]
            .as_f64()
            .unwrap_or_else(|| panic!("missing float32 {stage} rtol")),
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ComparisonStats {
    max_absolute: f64,
    max_relative: f64,
    failures: usize,
    non_finite_count: usize,
}

fn comparison_stats(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) -> ComparisonStats {
    assert_eq!(actual.len(), expected.len(), "comparison lengths differ");
    let mut stats = ComparisonStats {
        max_absolute: 0.0,
        max_relative: 0.0,
        failures: 0,
        non_finite_count: 0,
    };
    for (&observed, &reference) in actual.iter().zip(expected) {
        if !observed.is_finite() || !reference.is_finite() {
            stats.failures += 1;
            stats.non_finite_count += 1;
            continue;
        }
        let absolute = f64::from((observed - reference).abs());
        let relative = absolute / f64::from(reference.abs()).max(f64::MIN_POSITIVE);
        stats.max_absolute = stats.max_absolute.max(absolute);
        stats.max_relative = stats.max_relative.max(relative);
        if absolute > atol + rtol * f64::from(reference.abs()) {
            stats.failures += 1;
        }
    }
    stats
}

fn compare(
    reference_dir: &Path,
    manifest: &Value,
    case: &str,
    stage: &str,
    actual: &[f32],
) -> Value {
    let metadata = array(manifest, case, stage);
    assert_eq!(metadata["dtype"], "float32", "{stage} oracle dtype");
    let path = reference_dir.join(
        metadata["file"]
            .as_str()
            .unwrap_or_else(|| panic!("{stage} oracle file must be a string")),
    );
    let expected = f32_file(&path);
    assert_eq!(
        actual.len(),
        expected.len(),
        "{stage} element count differs"
    );
    let (atol, rtol) = tolerance(manifest, stage);
    let stats = comparison_stats(actual, &expected, atol, rtol);
    json!({
        "stage": stage,
        "elements": actual.len(),
        "atol": atol,
        "rtol": rtol,
        "max_absolute": stats.max_absolute,
        "max_relative": stats.max_relative,
        "failures": stats.failures,
        "non_finite_count": stats.non_finite_count,
        "passed": stats.failures == 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_rejects_non_finite_actual_and_reference_values() {
        let actual_nan = comparison_stats(&[1.0, f32::NAN], &[1.0, 2.0], 1e-6, 1e-6);
        assert_eq!(actual_nan.failures, 1);
        assert_eq!(actual_nan.non_finite_count, 1);

        let reference_inf = comparison_stats(&[1.0, 2.0], &[1.0, f32::INFINITY], 1e-6, 1e-6);
        assert_eq!(reference_inf.failures, 1);
        assert_eq!(reference_inf.non_finite_count, 1);
    }
}

fn main() {
    let model = required_path("--model");
    let reference_dir = required_path("--reference");
    let case = argument("--case").unwrap_or_else(|| "image_text".to_string());
    let device = argument("--device").unwrap_or_else(|| "local-task".to_string());
    let report_path = argument("--report").map(PathBuf::from);
    let manifest: Value = serde_json::from_slice(
        &fs::read(reference_dir.join("manifest.json")).expect("read reference manifest"),
    )
    .expect("parse reference manifest");
    let pixel_metadata = array(&manifest, &case, "processor_pixel_values");
    let pixel_shape = pixel_metadata["shape"]
        .as_array()
        .expect("pixel shape must be an array")
        .iter()
        .map(|dimension| dimension.as_u64().expect("pixel dimension") as usize)
        .collect::<Vec<_>>();
    assert_eq!(
        pixel_shape.first(),
        Some(&1),
        "diagnostic gate accepts one image"
    );
    let pixels = f32_file(
        &reference_dir.join(
            pixel_metadata["file"]
                .as_str()
                .expect("pixel oracle file must be a string"),
        ),
    );
    let mut projector = IreeVisionDiagnosticProjector::load(&model, &device)
        .unwrap_or_else(|error| panic!("load IREE vision projector: {error}"));
    let projection = projector
        .project(&pixels)
        .unwrap_or_else(|error| panic!("invoke IREE vision projector: {error}"));

    let mut comparisons = Vec::new();
    comparisons.push(compare(
        &reference_dir,
        &manifest,
        &case,
        "vision_hidden_state_00",
        &projection.hidden_states[0],
    ));
    for (stage, values) in BLOCK0_STAGES.iter().zip(&projection.block0_states) {
        comparisons.push(compare(&reference_dir, &manifest, &case, stage, values));
    }
    for (index, values) in projection.hidden_states.iter().enumerate().skip(1) {
        let stage = format!("vision_hidden_state_{index:02}");
        comparisons.push(compare(&reference_dir, &manifest, &case, &stage, values));
    }
    comparisons.push(compare(
        &reference_dir,
        &manifest,
        &case,
        "selected_vision_features",
        &projection.selected_vision_features,
    ));
    comparisons.push(compare(
        &reference_dir,
        &manifest,
        &case,
        "projected_image_features",
        &projection.projected_image_features,
    ));
    let first_divergence = comparisons
        .iter()
        .find(|comparison| comparison["passed"] == false)
        .map(|comparison| comparison["stage"].clone());
    let report = json!({
        "schema": 1,
        "producer": "mlxcel-xla-iree-vision",
        "device": device,
        "case": case,
        "artifact_fingerprint": projector.artifact_fingerprint(),
        "hidden_shape": projection.hidden_shape,
        "projected_shape": projection.projected_shape,
        "metrics": {
            "pixel_upload_bytes": projection.metrics.pixel_upload_bytes,
            "diagnostic_transfer_bytes": projection.metrics.projected_transfer_bytes,
            "elapsed_seconds": projection.metrics.elapsed_seconds,
        },
        "first_divergence": first_divergence,
        "comparisons": comparisons,
        "passed": first_divergence.is_none(),
    });
    if let Some(path) = report_path {
        fs::write(
            &path,
            serde_json::to_vec_pretty(&report).expect("serialize report"),
        )
        .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize report")
    );
    assert!(
        first_divergence.is_none(),
        "IREE vision first divergence: {}",
        first_divergence
            .as_ref()
            .and_then(Value::as_str)
            .unwrap_or("unknown")
    );
}
