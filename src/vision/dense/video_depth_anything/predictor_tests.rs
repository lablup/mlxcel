// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;
#[test]
fn padding_matches_upstream() {
    let padding = |frames| (FRAME_STEP - frames % FRAME_STEP) % FRAME_STEP + OVERLAP;
    assert_eq!(padding(40), 14);
    assert_eq!(padding(100), 20);
    assert_eq!(padding(44), 10);
    assert_eq!(padding(1), 31);
}
#[test]
fn window_starts_match_upstream() {
    assert_eq!(window_starts(40), [0, 22]);
    assert_eq!(window_starts(100), [0, 22, 44, 66, 88]);
}
#[test]
fn keyframe_conditioning_uses_the_previous_conditioned_window() {
    let plans = conditioned_source_indices(100);
    assert_eq!(
        &plans[1][..OVERLAP],
        &[0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
    );
    assert_eq!(&plans[1][OVERLAP..], &(32..54).collect::<Vec<_>>());
    assert_eq!(plans[2][0], 0);
    // Conditioning is recursive: plan 2 takes position 12 from plan 1,
    // whose source frame is 34 after the first 22-frame advance.
    assert_eq!(plans[2][1], 34);
}
#[test]
fn still_workset_counts_both_padded_window_copies() {
    let target = 14;
    let expected = 2 + INFER_LEN * 2 + target * target * 3 * INFER_LEN * 2;
    assert_eq!(host_working_values(1, 1, target, target).unwrap(), expected);
    assert_eq!(conditioned_source_indices(1).len(), 1);
    assert_eq!(conditioned_source_indices(FRAME_STEP).len(), 1);
    assert_eq!(conditioned_source_indices(FRAME_STEP + 1).len(), 2);
}

#[test]
fn model_workset_bounds_custom_large_geometry_but_keeps_published_size() {
    let config = super::super::config::VdaConfig {
        encoder: "vitl".into(),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let published_host = host_working_values(64, 1920 * 1080, 518, 924).unwrap();
    let published_model = model_working_values(&config, 518, 924).unwrap();
    assert!(published_host <= MAX_HOST_WORKING_VALUES);
    assert!(published_host + published_model <= MAX_TOTAL_WORKING_VALUES);

    let custom_host = host_working_values(1, 1, 1372, 1372).unwrap();
    let custom_model = model_working_values(&config, 1372, 1372).unwrap();
    assert!(custom_host + custom_model > MAX_TOTAL_WORKING_VALUES);
}

#[test]
fn singular_alignment_uses_identity() {
    assert_eq!(
        compute_scale_and_shift(&[vec![1.0; 2]], &[vec![3.0; 2]], 2).unwrap(),
        (1.0, 0.0)
    );
}
#[test]
fn nonsingular_alignment_recovers_scale_and_shift() {
    let prediction = vec![vec![1.0, 2.0, 4.0]];
    let target = vec![vec![7.0, 9.0, 13.0]];
    let (scale, shift) = compute_scale_and_shift(&prediction, &target, 3).unwrap();
    assert!((scale - 2.0).abs() < 1e-5);
    assert!((shift - 5.0).abs() < 1e-5);
}
#[test]
fn interpolation_includes_both_endpoints() {
    let mut windows = vec![vec![vec![1.0]; INFER_LEN], vec![vec![3.0]; INFER_LEN]];
    windows[1][0][0] = 1.0;
    windows[1][1][0] = 1.0;
    let mut output = Vec::new();
    let mut reference = Vec::new();
    align_window(&mut output, &mut reference, windows.remove(0), true, 1).unwrap();
    align_window(&mut output, &mut reference, windows.remove(0), true, 1).unwrap();
    assert_eq!(output[INFER_LEN - INTERP_LEN][0], 1.0);
    assert_eq!(output[INFER_LEN - 1][0], 3.0);
}
