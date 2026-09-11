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

use super::*;
use crate::distributed::pipeline::partition::{ModelProfile, StageAssignment};

fn profile() -> ModelProfile {
    ModelProfile::uniform(8, 100, 50, 50)
}

fn stage(
    stage_index: usize,
    device_id: &str,
    layer_range: std::ops::Range<usize>,
    has_embedding: bool,
    has_lm_head: bool,
    estimated: u64,
) -> StageAssignment {
    StageAssignment {
        stage_index,
        device_id: device_id.to_string(),
        layer_range,
        has_embedding,
        has_lm_head,
        estimated_memory_bytes: estimated,
    }
}

#[test]
fn report_captures_imbalance_ratio() {
    let assignments = vec![
        stage(0, "d0", 0..4, true, false, 450),
        stage(1, "d1", 4..8, false, true, 450),
    ];
    let report = build_quality_report(&profile(), &assignments);
    assert_eq!(report.max_stage_estimated_bytes, 450);
    assert_eq!(report.min_stage_estimated_bytes, 450);
    assert_eq!(report.imbalance_pct, 100);
}

#[test]
fn report_flags_skewed_plan() {
    let assignments = vec![
        stage(0, "d0", 0..2, true, false, 250),
        stage(1, "d1", 2..8, false, true, 650),
    ];
    let report = build_quality_report(&profile(), &assignments);
    assert!(report.imbalance_pct > 100);
    // 650 / 250 = 2.6x ~= 260.
    assert!(report.imbalance_pct >= 200);
}

#[test]
fn populate_actual_memory_sets_measurements() {
    let assignments = vec![
        stage(0, "d0", 0..4, true, false, 450),
        stage(1, "d1", 4..8, false, true, 450),
    ];
    let mut report = build_quality_report(&profile(), &assignments);
    populate_actual_memory(&mut report, &[480, 460]);
    assert_eq!(report.per_stage[0].actual_bytes, Some(480));
    assert_eq!(report.per_stage[1].actual_bytes, Some(460));
    // 480 / 450 = 1.066x -> 106.
    assert_eq!(report.per_stage[0].drift_percent(), Some(106));
    // 460 / 450 = 1.022x -> 102.
    assert_eq!(report.per_stage[1].drift_percent(), Some(102));
}

#[test]
fn format_report_is_multiline_and_stable() {
    let assignments = vec![
        stage(0, "d0", 0..4, true, false, 450),
        stage(1, "d1", 4..8, false, true, 450),
    ];
    let mut report = build_quality_report(&profile(), &assignments);
    populate_actual_memory(&mut report, &[480, 460]);
    let text = format_quality_report(&report);
    // One header line plus two stage lines, no warnings section.
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3, "unexpected log shape: {text}");
    assert!(lines[0].contains("pipeline partition quality"));
    assert!(lines[1].contains("stage 0 device 'd0'"));
    assert!(lines[2].contains("stage 1 device 'd1'"));
    assert!(lines[1].contains("drift=1.6x"));
    assert!(lines[2].contains("drift=1.2x"));
}

#[test]
fn format_report_prints_warnings_block() {
    let assignments = vec![
        stage(0, "d0", 0..4, true, false, 450),
        stage(1, "d1", 4..8, false, true, 450),
    ];
    let mut report = build_quality_report(&profile(), &assignments);
    report.warnings.push("synthetic-warning".into());
    let text = format_quality_report(&report);
    assert!(text.contains("warnings:"));
    assert!(text.contains("- synthetic-warning"));
}

#[test]
fn summarize_quality_warnings_passes_through() {
    let mut report = build_quality_report(&profile(), &[stage(0, "d0", 0..8, true, true, 900)]);
    report.warnings.push("balancer-warning".into());
    let warnings = summarize_quality_warnings(&report);
    assert_eq!(warnings, vec!["balancer-warning".to_string()]);
}
