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

#[test]
fn sentinels_preserve_original_feature_rows_and_non_contiguous_targets() {
    let plan = MolmoSparseAddPlan::from_image_input_idx(&[-100, 3, -1, 0], 5, 2, 8, 8).unwrap();
    assert_eq!(
        plan.pairs(),
        [
            MolmoSparseAddPair {
                feature_row: 1,
                target_position: 3,
            },
            MolmoSparseAddPair {
                feature_row: 3,
                target_position: 0,
            },
        ]
    );
    let mut text = vec![10.0; 10];
    let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    plan.apply(&mut text, &features).unwrap();
    assert_eq!(
        text,
        [17.0, 18.0, 10.0, 10.0, 10.0, 10.0, 13.0, 14.0, 10.0, 10.0]
    );
}

#[test]
fn rejects_duplicate_and_out_of_range_targets_before_merge() {
    assert!(matches!(
        MolmoSparseAddPlan::from_image_input_idx(&[1, -100, 1], 3, 2, 4, 4),
        Err(MolmoSparseAddError::DuplicateTarget { .. })
    ));
    assert!(matches!(
        MolmoSparseAddPlan::from_image_input_idx(&[-1, 3], 3, 2, 4, 4),
        Err(MolmoSparseAddError::TargetOutOfRange { .. })
    ));
}

#[test]
fn empty_active_features_are_valid_and_leave_text_unchanged() {
    let plan = MolmoSparseAddPlan::from_image_input_idx(&[-100, -1], 2, 2, 4, 4).unwrap();
    let mut text = vec![1.0, 2.0, 3.0, 4.0];
    plan.apply(&mut text, &[5.0, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(text, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn replacement_negative_fixture_is_caught_by_additive_oracle() {
    fn oracle_logits(embeddings: &[f32]) -> [f32; 2] {
        let active = &embeddings[2..4];
        [active[0] + 2.0 * active[1], 3.0 * active[0] - active[1]]
    }

    let plan = MolmoSparseAddPlan::from_image_input_idx(&[1], 2, 2, 1, 2).unwrap();
    let original = vec![2.0, 3.0, 5.0, 7.0];
    let features = vec![11.0, 13.0];
    let mut additive = original.clone();
    plan.apply(&mut additive, &features).unwrap();

    let mut replacement = original;
    replacement[2..4].copy_from_slice(&features);
    assert_eq!(additive, [2.0, 3.0, 16.0, 20.0]);
    assert_ne!(replacement, additive);
    assert_eq!(oracle_logits(&additive), [56.0, 28.0]);
    assert_ne!(
        oracle_logits(&replacement),
        oracle_logits(&additive),
        "the downstream logit oracle must reject replacement semantics"
    );
}

#[test]
fn capacity_shape_and_nonfinite_fail_closed() {
    assert!(matches!(
        MolmoSparseAddPlan::from_image_input_idx(&[0, 1], 2, 2, 1, 2),
        Err(MolmoSparseAddError::Capacity {
            dimension: "feature rows",
            ..
        })
    ));
    let plan = MolmoSparseAddPlan::from_image_input_idx(&[0], 1, 2, 1, 1).unwrap();
    assert!(matches!(
        plan.apply(&mut [0.0, 0.0], &[f32::NAN, 1.0]),
        Err(MolmoSparseAddError::NonFinite {
            tensor: "projected features",
            ..
        })
    ));
}

#[test]
fn processor_patch_rows_and_projected_rows_have_distinct_shapes() {
    let prepared = prepared_image(1.0);
    let plan = prepared.sparse_add_plan(4, 2, 2, 4, 2, 4).unwrap();
    assert_eq!(
        plan.pairs(),
        [MolmoSparseAddPair {
            feature_row: 1,
            target_position: 2,
        }]
    );
}

#[test]
fn invalid_mask_domain_and_late_overflow_are_atomic_failures() {
    let mut invalid_mask = prepared_image(1.5);
    assert!(matches!(
        invalid_mask.validate(2, 4, 2),
        Err(MolmoSparseAddError::MaskValue { .. })
    ));
    invalid_mask.image_masks.fill(-1.0);
    invalid_mask.validate(2, 4, 2).unwrap();

    let plan = MolmoSparseAddPlan::from_image_input_idx(&[0, 1], 2, 1, 2, 2).unwrap();
    let original = vec![1.0, f32::MAX];
    let mut text = original.clone();
    let error = plan.apply(&mut text, &[2.0, f32::MAX]).unwrap_err();
    assert!(matches!(
        error,
        MolmoSparseAddError::NonFinite {
            tensor: "merged embeddings",
            flat_index: 1,
        }
    ));
    assert_eq!(text, original, "failed merge must not modify any row");
}

fn prepared_image(mask: f32) -> MolmoPreparedImage {
    MolmoPreparedImage {
        pixel_values: vec![0.0; 2 * 4 * 3],
        crop_count: 2,
        patches_per_crop: 4,
        patch_width: 3,
        image_masks: vec![mask; 2 * 4],
        projected_rows_per_crop: 1,
        image_input_idx: vec![-100, 2],
    }
}
