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

//! A mixture-of-experts expert adapter is refused by name on the fused and the
//! unfused path alike (issue #1577).

use std::collections::HashMap;

use mlxcel_core::weights::WeightMap;

use crate::lora::runtime::{RuntimeLoraSet, stage_runtime_adapters};
use crate::lora::test_support::{
    OwnedTensor, adapter_pair_tensors, ones_tensor_3d, temp_dir, write_adapter_dir,
};
use crate::lora::{LoraAdapterSpec, apply_lora_adapters};

const EXPERTS: usize = 4;

/// An MoE base: one `[experts, out, in]` expert projection per layer, plus a
/// dense attention projection.
fn moe_base() -> WeightMap {
    let mut base = WeightMap::new();
    for layer in 0..2 {
        base.insert(
            format!("model.layers.{layer}.mlp.switch_mlp.gate_proj.weight"),
            mlxcel_core::ones(&[EXPERTS as i32, 3, 4], mlxcel_core::dtype::FLOAT32),
        );
        base.insert(
            format!("model.layers.{layer}.self_attn.q_proj.weight"),
            mlxcel_core::ones(&[4, 4], mlxcel_core::dtype::FLOAT32),
        );
    }
    base
}

/// mlx-lm `LoRASwitchLinear` tensors on two layers' expert projections, plus
/// an ordinary 2-D pair, written as an adapter directory.
fn moe_adapter(name: &str) -> std::path::PathBuf {
    let dir = temp_dir(name);
    let mut tensors: HashMap<String, OwnedTensor> = HashMap::new();
    for layer in 0..2 {
        let stem = format!("model.layers.{layer}.mlp.switch_mlp.gate_proj");
        tensors.insert(format!("{stem}.lora_a"), ones_tensor_3d(EXPERTS, 4, 2));
        tensors.insert(format!("{stem}.lora_b"), ones_tensor_3d(EXPERTS, 2, 3));
    }
    tensors.extend(adapter_pair_tensors(
        "model.layers.0.self_attn.q_proj",
        4,
        2,
        4,
    ));
    write_adapter_dir(&dir, "lora", 2, tensors);
    dir
}

fn assert_names_the_moe_limitation(err: &str) {
    assert!(
        err.contains("mixture-of-experts expert adapters are not supported"),
        "{err}"
    );
    assert!(err.contains("LoRASwitchLinear"), "{err}");
    // One aggregated line for the whole adapter, naming the first layer.
    assert!(err.contains("2 such pairs"), "{err}");
    assert!(
        err.contains("model.layers.0.mlp.switch_mlp.gate_proj.lora_a"),
        "{err}"
    );
    assert!(
        !err.contains("model.layers.1.mlp.switch_mlp.gate_proj.lora_a"),
        "the limitation is reported once, not per layer: {err}"
    );
    assert!(!err.contains("Expected 2D"), "{err}");
}

#[test]
fn the_fused_path_names_an_expert_adapter_as_unsupported() {
    let dir = moe_adapter("moe_fused");

    let err = apply_lora_adapters(&moe_base(), &dir)
        .err()
        .expect("an expert adapter cannot be fused")
        .to_string();
    assert_names_the_moe_limitation(&err);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_runtime_path_names_an_expert_adapter_as_unsupported() {
    let dir = moe_adapter("moe_runtime");
    let set = RuntimeLoraSet::from_specs(&[LoraAdapterSpec {
        path: dir.clone(),
        scale: 1.0,
        apply: true,
    }])
    .expect("a lora adapter config");

    let err = stage_runtime_adapters(&moe_base(), &set)
        .expect_err("an expert adapter cannot be served unfused")
        .to_string();
    assert_names_the_moe_limitation(&err);
    assert!(!mlxcel_core::runtime_lora::has_pending());

    let _ = std::fs::remove_dir_all(&dir);
}
