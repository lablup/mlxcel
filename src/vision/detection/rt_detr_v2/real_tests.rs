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

//! Regression against the RT-DETRv2 behavior that predates shared dispatch.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::{Detection, RtDetrV2Predictor};

const FIXTURE_DIR: &str = "tests/fixtures/pp_doclayout_v3";

#[derive(Deserialize)]
struct Expected {
    threshold: f32,
    numeric_max_abs_tolerance: f32,
    detections: Vec<ExpectedDetection>,
}

#[derive(Deserialize)]
struct ExpectedDetection {
    bbox: [f32; 4],
    score: f32,
    label: usize,
    class_name: String,
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIR)
        .join(name)
}

fn sort_geometrically(detections: &mut [Detection]) {
    detections.sort_by(|left, right| {
        left.bbox[1]
            .total_cmp(&right.bbox[1])
            .then_with(|| left.bbox[0].total_cmp(&right.bbox[0]))
            .then_with(|| right.score.total_cmp(&left.score))
            .then_with(|| left.label.cmp(&right.label))
    });
}

#[test]
#[ignore = "requires the pinned Docling Layout Heron checkpoint"]
fn shared_detector_keeps_the_authentic_rt_detr_cli_contract() {
    let _device_lock = mlxcel_core::streams::lock_default_device();
    assert!(mlxcel_core::gpu_backend_available());
    let _gpu_device = mlxcel_core::streams::DefaultDeviceGuard::gpu();
    assert!(mlxcel_core::default_device_is_gpu());
    let checkpoint = std::env::var_os("MLXCEL_TEST_RT_DETR_HERON_DIR")
        .map(PathBuf::from)
        .expect("set MLXCEL_TEST_RT_DETR_HERON_DIR to the pinned checkpoint");
    let expected: Expected =
        serde_json::from_slice(&std::fs::read(fixture("rt_detr_heron_expected.json")).unwrap())
            .unwrap();
    let predictor = RtDetrV2Predictor::from_pretrained(checkpoint, expected.threshold).unwrap();
    let mut actual = predictor
        .predict_path(fixture("two_column_page.png"))
        .unwrap()
        .detections;
    assert_eq!(
        mlxcel_core::hardware::gpu_backend_kind(),
        mlxcel_core::hardware::GpuBackendKind::Rocm
    );
    eprintln!("RT-DETRv2 regression actual_backend=rocm");
    sort_geometrically(&mut actual);
    assert_eq!(actual.len(), expected.detections.len());
    let tolerance = expected.numeric_max_abs_tolerance;
    for (index, (actual, expected)) in actual.iter().zip(&expected.detections).enumerate() {
        assert_eq!(actual.reading_order, None, "detection {index} order");
        assert_eq!(actual.label, expected.label, "detection {index} label");
        assert_eq!(
            actual.class_name, expected.class_name,
            "detection {index} class"
        );
        assert!(
            (actual.score - expected.score).abs() <= tolerance,
            "detection {index} score: {} != {}",
            actual.score,
            expected.score
        );
        for axis in 0..4 {
            assert!(
                (actual.bbox[axis] - expected.bbox[axis]).abs() <= tolerance,
                "detection {index} bbox[{axis}]: {} != {}",
                actual.bbox[axis],
                expected.bbox[axis]
            );
        }
    }
}
