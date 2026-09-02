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

//! Unit tests for the runtime (unfused) adapter staging path.
//!
//! The runtime path shares `validate_adapter_tensors` with fusion, so these
//! tests cover the wiring rather than re-testing every rule: that the shared
//! validator is reached, that its report names the adapter directory, and that
//! a well-formed adapter still stages (issue #1328).

use super::*;
use crate::lora::test_support::{adapter_pair_tensors, ones_tensor, temp_dir, write_adapter_dir};

/// A base weight map holding one `[out, in]` projection per named layer.
fn base_map(layers: &[(&str, i32, i32)]) -> WeightMap {
    let mut base = WeightMap::new();
    for (layer, out_features, in_features) in layers {
        base.insert(
            format!("{layer}.weight"),
            mlxcel_core::ones(&[*out_features, *in_features], mlxcel_core::dtype::FLOAT32),
        );
    }
    base
}

fn spec_for(dir: &std::path::Path) -> LoraAdapterSpec {
    LoraAdapterSpec {
        path: dir.to_path_buf(),
        scale: 1.0,
        apply: true,
    }
}

#[test]
fn runtime_staging_refuses_a_pair_with_no_base_weight() {
    // Serving unfused has exactly the fused path's failure mode: a term whose
    // base weight does not exist is never claimed by any layer, so the model
    // answers from base weights while the adapter is reported as loaded.
    let dir = temp_dir("runtime_missing");
    write_adapter_dir(&dir, "lora", 2, adapter_pair_tensors("other", 4, 2, 3));

    let set = RuntimeLoraSet::from_specs(&[spec_for(&dir)]).expect("a lora adapter config");
    let base = base_map(&[("layer", 3, 4)]);

    let err = stage_runtime_adapters(&base, &set)
        .expect_err("an adapter that maps onto nothing must not stage")
        .to_string();
    assert!(err.contains(&dir.display().to_string()), "{err}");
    assert!(err.contains("other.lora_a"), "{err}");
    assert!(err.contains("no base weight"), "{err}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn runtime_staging_refuses_an_unknown_adapter_tensor() {
    let dir = temp_dir("runtime_leaf");
    let mut tensors = adapter_pair_tensors("layer", 4, 2, 3);
    tensors.insert("layer.m".to_string(), ones_tensor(3, 1));
    write_adapter_dir(&dir, "lora", 2, tensors);

    let set = RuntimeLoraSet::from_specs(&[spec_for(&dir)]).expect("a lora adapter config");
    let base = base_map(&[("layer", 3, 4)]);

    let err = stage_runtime_adapters(&base, &set)
        .expect_err("a magnitude vector must not be dropped on this path either")
        .to_string();
    assert!(err.contains("layer.m"), "{err}");
    assert!(err.contains("not a LoRA tensor"), "{err}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn runtime_set_refuses_a_dora_adapter() {
    let dir = temp_dir("runtime_dora");
    write_adapter_dir(&dir, "dora", 2, adapter_pair_tensors("layer", 4, 2, 3));

    let err = RuntimeLoraSet::from_specs(&[spec_for(&dir)])
        .expect_err("DoRA is refused before any serving state is built")
        .to_string();
    assert!(err.contains("DoRA"), "{err}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_well_formed_runtime_adapter_still_stages() {
    // Positive control for the shared validator: the happy path must reach
    // `runtime_lora::stage`. The staging slot is thread-local, so this drains
    // it again rather than leaving terms for the next test on this thread.
    let dir = temp_dir("runtime_ok");
    write_adapter_dir(&dir, "lora", 2, adapter_pair_tensors("layer", 4, 2, 3));

    let set = RuntimeLoraSet::from_specs(&[spec_for(&dir)]).expect("a lora adapter config");
    let base = base_map(&[("layer", 3, 4)]);

    stage_runtime_adapters(&base, &set).expect("a well-formed adapter stages");
    let unclaimed = mlxcel_core::runtime_lora::drain_unclaimed();
    assert_eq!(
        unclaimed,
        vec!["layer".to_string()],
        "the pair must have been staged under its base weight prefix"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
