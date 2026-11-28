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

/// Runs a three-layer toy forward under capture: every layer adds its index
/// to the residual, and `seed` flips one element of one row at one layer.
fn toy_forward(rows: usize, seed: Option<(usize, usize)>) -> Vec<Stage> {
    let (_, stages) = with_capture(|| {
        let mut h: Vec<f32> = (0..rows * 4).map(|i| i as f32).collect();
        for layer in 0..3 {
            for value in &mut h {
                *value += layer as f32;
            }
            if let Some((seed_layer, seed_row)) = seed
                && seed_layer == layer
                && seed_row < rows
            {
                h[seed_row * 4 + 2] += 0.5;
            }
            let tensor = mlxcel_core::from_slice_f32(&h, &[1, rows as i32, 4]);
            capture(layer, "toy", "layer output", &tensor);
        }
    });
    stages
}

/// The chain arm of the toy: row `row` of the block's input, alone.
fn toy_chain(rows: usize) -> Vec<Vec<Stage>> {
    let block = toy_forward(rows, None);
    (0..rows)
        .map(|row| {
            block
                .iter()
                .map(|stage| Stage {
                    rows: vec![stage.rows[row].clone()],
                    ..stage.clone()
                })
                .collect()
        })
        .collect()
}

/// Issue #3409: on a seeded mismatch the helper reports the first divergent
/// layer and row, although the error is still there in every later layer.
#[test]
fn a_seeded_mismatch_is_reported_at_its_first_layer_and_row() {
    let chain = toy_chain(4);
    assert_eq!(locate(&toy_forward(4, None), &chain), Ok(None));

    let block = toy_forward(4, Some((1, 2)));
    let found = locate(&block, &chain)
        .expect("the traces line up")
        .expect("the seeded mismatch is found");
    assert_eq!(
        (found.stage, found.layer, found.position),
        (1, 1, 2),
        "{found}"
    );
    assert_eq!(found.total_bytes, 16);
    assert!((1..=4).contains(&found.differing_bytes), "{found:?}");
    assert_eq!(
        first_divergence(&block, &chain).as_deref(),
        Some("first divergence: layer 1 (toy) layer output at position 2")
    );

    // The per-stage report shows the clean layer, then the seeded row
    // carrying its difference through the layers after it.
    let lines = report(&block, &chain);
    assert_eq!(lines.len(), 3);
    assert!(
        lines[0].ends_with("r0 0/16 r1 0/16 r2 0/16 r3 0/16"),
        "{}",
        lines[0]
    );
    for line in &lines[1..] {
        assert!(line.contains("r1 0/16"), "{line}");
        assert!(!line.contains("r2 0/16"), "{line}");
    }
}

#[test]
fn capture_is_inert_outside_a_scope() {
    assert!(!capturing());
    let tensor = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 1, 2]);
    capture(0, "toy", "layer output", &tensor);
    let (inside, stages) = with_capture(capturing);
    assert!(inside);
    assert!(stages.is_empty());
}

#[test]
fn a_report_of_misaligned_traces_is_empty() {
    let block = toy_forward(2, None);
    assert!(report(&block, &[]).is_empty());
    assert_eq!(
        locate(&block, &toy_chain(3)),
        Err("probe trace row counts differ")
    );
}
