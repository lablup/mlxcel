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

use super::*;

#[test]
fn localization_names_first_layer_kind_and_subop() {
    let stage = |layer, kind: &str, op, rows| Stage {
        layer,
        kind: kind.to_owned(),
        op,
        rows,
    };
    let block = vec![
        stage(
            0,
            "sliding_attention",
            "attention output",
            vec![vec![1], vec![2]],
        ),
        stage(0, "sliding_attention", "MLP output", vec![vec![3], vec![4]]),
        stage(
            1,
            "full_attention",
            "attention output",
            vec![vec![5], vec![6]],
        ),
    ];
    let mut chain: Vec<Vec<Stage>> = (0..2)
        .map(|row| {
            block
                .iter()
                .map(|s| Stage {
                    rows: vec![s.rows[row].clone()],
                    ..s.clone()
                })
                .collect()
        })
        .collect();
    assert_eq!(first_divergence(&block, &chain), None);
    chain[1][1].rows[0][0] = 9;
    chain[0][2].rows[0][0] = 9;
    assert_eq!(
        first_divergence(&block, &chain).as_deref(),
        Some("first divergence: layer 0 (sliding_attention) MLP output at position 1")
    );
}

#[test]
fn capture_scope_resets_on_unwind() {
    let failed = std::panic::catch_unwind(|| with_capture(|| panic!("abort probe")));
    assert!(failed.is_err());
    let (_, stages) = with_capture(|| ());
    assert!(stages.is_empty());
}

#[test]
fn malformed_trace_cannot_report_equality() {
    let stage = Stage {
        layer: 0,
        kind: "sliding_attention".into(),
        op: "attention output",
        rows: vec![vec![1]],
    };
    assert!(first_divergence(&[], &[]).is_some());
    assert!(
        first_divergence(
            std::slice::from_ref(&stage),
            &[vec![stage.clone(), stage.clone()]]
        )
        .is_some()
    );
    let mut missing = stage.clone();
    missing.rows.clear();
    assert!(first_divergence(&[missing.clone()], &[vec![missing]]).is_some());
}
