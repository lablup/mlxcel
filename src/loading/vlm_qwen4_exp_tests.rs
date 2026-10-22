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

use mlxcel_core::weights::WeightMap;
use serde_json::json;

use super::{check_merger_width, reject_deepstack};

#[test]
fn deepstack_indexes_are_refused_by_name() {
    let error = reject_deepstack(&json!({"vision_config": {"deepstack_visual_indexes": [8, 16]}}))
        .expect_err("a non-empty deepstack list must not load");
    assert!(
        error.to_string().contains("deepstack_visual_indexes"),
        "{error}"
    );
}

#[test]
fn absent_or_empty_deepstack_indexes_load() {
    for config in [
        json!({"vision_config": {}}),
        json!({"vision_config": {"deepstack_visual_indexes": []}}),
        json!({"vision_config": {"deepstack_visual_indexes": null}}),
    ] {
        reject_deepstack(&config).unwrap_or_else(|error| panic!("{config}: {error}"));
    }
}

#[test]
fn vision_features_must_match_the_text_width() {
    let merger = |rows: i32| {
        let mut vision = WeightMap::new();
        vision.insert(
            "vision_tower.merger.linear_fc2.weight".to_string(),
            mlxcel_core::zeros(&[rows, 8], mlxcel_core::dtype::FLOAT32),
        );
        vision
    };
    check_merger_width(&merger(32), 32).expect("a 32-wide merger feeds a 32-wide text model");
    let error = check_merger_width(&merger(16), 32)
        .expect_err("a 16-wide merger must not load against a 32-wide text model");
    assert!(
        error
            .to_string()
            .contains("linear_fc2.weight has shape [16, 8]"),
        "{error}"
    );
}

#[test]
fn optiq_folded_norms_restore_exact_scales_without_touching_gated_norms() {
    use mlxcel_core::{dtype, utils::array_to_vec_f32};
    for optiq in [false, true] {
        let mut weights = WeightMap::new();
        let norms = [
            "model.layers.0.attn_hyper_connection.hc_norm.weight",
            "model.layers.3.self_attn.q_norm.weight",
            "model.layers.1.ple.norm_key.weight",
            "model.layers.0.linear_attn.norm.weight",
        ];
        for name in norms {
            weights.insert(
                name.into(),
                mlxcel_core::astype(
                    &mlxcel_core::from_slice_f32(&[0.125, 1.0078125, 2.5], &[3]),
                    dtype::BFLOAT16,
                ),
            );
        }
        let config = if optiq {
            json!({"optiq_vision": {"sidecar": "optiq/vision.safetensors"}})
        } else {
            json!({})
        };
        super::restore_optiq_zero_centered_norms(&mut weights, &config);
        for name in norms {
            let expected = if optiq && !name.ends_with("linear_attn.norm.weight") {
                vec![-0.875, 0.0078125, 1.5]
            } else {
                vec![0.125, 1.0078125, 2.5]
            };
            assert_eq!(array_to_vec_f32(&weights[name]), expected, "{name}");
        }
    }
}
