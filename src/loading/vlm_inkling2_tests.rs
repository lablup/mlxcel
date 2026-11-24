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

//! The layer-by-layer evaluation order of the Inkling 2 loader (#3373).

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;

use super::{TEXT_PREFIX, layer_groups, materialize_by_layer};

fn map(names: &[&str]) -> WeightMap {
    names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let base = mlxcel_core::from_slice_f32(&[i as f32, 1.0, 2.0, 3.0], &[2, 2]);
            // A lazy contiguous copy of a transposed view, the shape of the
            // expert split that made the first token peak on Small.
            let value =
                mlxcel_core::contiguous(&mlxcel_core::transpose_axes(&base, &[1, 0]), false);
            (name.to_string(), value)
        })
        .collect()
}

#[test]
fn every_decoder_layer_is_its_own_group_in_order_and_towers_stay_out() {
    let p = TEXT_PREFIX;
    let names = [
        format!("{p}layers.10.mlp.switch_mlp.gate_proj.weight"),
        format!("{p}layers.1.mlp.switch_mlp.up_proj.weight"),
        format!("{p}layers.1.attn.wq.weight"),
        format!("{p}layers.0.attn.wq.weight"),
        format!("{p}embed_tokens.weight"),
        format!("{p}norm.weight"),
        "language_model.lm_head.weight".to_string(),
        "vision_tower.blocks.0.attn.qkv.weight".to_string(),
    ];
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let weights = map(&refs);
    let groups = layer_groups(&weights, 11);
    assert_eq!(groups.len(), 12, "one group per layer plus the rest");
    assert_eq!(groups[0], [names[3].as_str()]);
    assert_eq!(groups[1], [names[2].as_str(), names[1].as_str()]);
    assert!(groups[2..10].iter().all(Vec::is_empty));
    assert_eq!(groups[10], [names[0].as_str()], "layers.10 is not layers.1");
    assert_eq!(
        groups[11],
        [names[6].as_str(), names[4].as_str(), names[5].as_str()]
    );
    let grouped: usize = groups.iter().map(Vec::len).sum();
    assert_eq!(grouped, names.len() - 1, "the vision tower stays lazy");
}

#[test]
fn a_layer_index_past_the_config_falls_into_the_rest() {
    let name = format!("{TEXT_PREFIX}layers.7.attn.wq.weight");
    let weights = map(&[name.as_str()]);
    let groups = layer_groups(&weights, 2);
    assert_eq!(groups[2], [name.as_str()]);
}

#[test]
fn layer_materialization_keeps_the_values_and_the_cache_limit() {
    let _lock = mlxcel_core::streams::lock_default_device();
    let p = TEXT_PREFIX;
    let names = [
        format!("{p}layers.0.attn.wq.weight"),
        format!("{p}layers.1.attn.wq.weight"),
        format!("{p}norm.weight"),
    ];
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let weights = map(&refs);
    let limit = mlxcel_core::memory::set_cache_limit(1 << 30);
    materialize_by_layer(&weights, 2);
    assert_eq!(mlxcel_core::memory::set_cache_limit(limit), 1 << 30);
    for (i, name) in names.iter().enumerate() {
        let values = array_to_vec_f32(&weights[name]);
        assert_eq!(values, [i as f32, 2.0, 1.0, 3.0], "{name}");
    }
}
