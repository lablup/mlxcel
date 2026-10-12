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
fn duplicate_schema_fields_are_rejected() {
    for yaml in [
        "version: 1\nversion: 1\noperations: []\n",
        "version: 1\noperations: [{op: scale, pattern: '*', factor: 1, factor: 2}]\n",
    ] {
        assert!(parse_config_str(yaml, None).is_err());
    }
}

#[test]
fn shipped_examples_keep_their_enum_and_numeric_syntax() {
    let options = || {
        serde_saphyr::options! {
            merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary,
            strict_booleans: true,
            reject_non_finite_typeless_float: false,
        }
    };
    for yaml in [
        include_str!("../../../../examples/surgery/add_example.yaml"),
        include_str!("../../../../examples/surgery/full_example.yaml"),
        include_str!("../../../../examples/surgery/interpolate_slerp_example.yaml"),
        include_str!("../../../../examples/surgery/prune_head_example.yaml"),
        include_str!("../../../../examples/surgery/replace_example.yaml"),
        include_str!("../../../../examples/surgery/scale_only.yaml"),
    ] {
        let config: SurgeryConfig = serde_saphyr::from_str_with_options(yaml, options()).unwrap();
        assert_eq!(config.version, SUPPORTED_SCHEMA_VERSION);
        assert!(!config.operations.is_empty());
    }
}

#[test]
fn non_finite_factors_reach_operation_validation() {
    for value in [
        ".nan", ".NaN", ".inf", "-.inf", "+.Inf", "'.nan'", "'.inf'", "'-.inf'",
    ] {
        let yaml =
            format!("version: 1\noperations: [{{op: scale, pattern: '*', factor: {value}}}]");
        let error = parse_config_str(&yaml, None)
            .err()
            .expect("non-finite rejected")
            .to_string();
        assert!(
            error.contains("scale") && error.contains("finite"),
            "{value}: {error}"
        );
    }
}

#[test]
fn non_finite_coefficients_reach_add_validation() {
    for value in [".nan", ".inf", "-.inf", "'.nan'", "'.inf'", "'-.inf'"] {
        let yaml = format!(
            "version: 1\noperations: [{{op: add, pattern: '*', source: donor.safetensors, alpha: {value}}}]"
        );
        let error = parse_config_str(&yaml, None)
            .err()
            .expect("non-finite rejected")
            .to_string();
        assert!(
            error.contains("(add)") && error.contains("alpha must be finite"),
            "{value}: {error}"
        );
    }
}

#[test]
fn non_finite_ratios_reach_interpolate_validation() {
    for value in [".nan", ".inf", "-.inf", "'.nan'", "'.inf'", "'-.inf'"] {
        let yaml = format!(
            "version: 1\noperations: [{{op: interpolate, pattern: '*', source_a: a.safetensors, source_b: b.safetensors, method: lerp, ratio: {value}}}]"
        );
        let error = parse_config_str(&yaml, None)
            .err()
            .expect("non-finite rejected")
            .to_string();
        assert!(
            error.contains("(interpolate)") && error.contains("ratio must be a finite number"),
            "{value}: {error}"
        );
    }
}
