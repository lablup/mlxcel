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

//! Weight-key sanitizer for the Laguna DFlash drafter.
//!
//! Strips a leading `model.`, fuses a split `self_attn.{q,k,v}_proj` into
//! `self_attn.qkv_proj` (q, k, v row order), rejects a layer that carries
//! both spellings or an incomplete split set, and finally checks that the
//! key set equals the published layout exactly (quantized variants may add
//! `.scales` / `.biases` next to each projection `.weight`).

use std::collections::BTreeSet;

use crate::ffi;
use crate::ops::concatenate;
use crate::weights::WeightMap;

use super::config::LagunaDFlashConfig;

const MODEL_PREFIX: &str = "model.";
const QUANT_SIDECARS: [&str; 2] = ["scales", "biases"];
const PROJECTIONS: [&str; 7] = [
    "fc",
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "self_attn.g_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// Sanitize `weights` in place for [`LagunaDFlashConfig`] `config`.
pub fn sanitize_weights(
    weights: &mut WeightMap,
    config: &LagunaDFlashConfig,
) -> Result<(), String> {
    strip_model_prefix(weights);
    for layer in 0..config.num_hidden_layers {
        fuse_split_qkv(weights, layer)?;
    }
    check_key_set(weights, config)
}

fn strip_model_prefix(weights: &mut WeightMap) {
    let renames: Vec<(String, String)> = weights
        .keys()
        .filter(|k| k.starts_with(MODEL_PREFIX))
        .map(|k| (k.clone(), k[MODEL_PREFIX.len()..].to_string()))
        .collect();
    for (old, new) in renames {
        if let Some(v) = weights.remove(&old) {
            weights.insert(new, v);
        }
    }
}

/// Fuse `layers.{i}.self_attn.{q,k,v}_proj.<leaf>` into
/// `layers.{i}.self_attn.qkv_proj.<leaf>` for every leaf the split set
/// carries (`weight`, and the quantization sidecars when present).
fn fuse_split_qkv(weights: &mut WeightMap, layer: usize) -> Result<(), String> {
    let attn = format!("layers.{layer}.self_attn");
    let fused_key = format!("{attn}.qkv_proj.weight");
    let split: Vec<String> = ["q_proj", "k_proj", "v_proj"]
        .iter()
        .map(|p| format!("{attn}.{p}.weight"))
        .collect();
    let present: Vec<bool> = split.iter().map(|k| weights.contains_key(k)).collect();
    let any_split = present.iter().any(|p| *p);
    if !any_split {
        return Ok(());
    }
    if weights.contains_key(&fused_key) {
        return Err(format!(
            "{attn}: both a fused qkv_proj and split q/k/v projections are present; the \
             checkpoint must carry one spelling"
        ));
    }
    if !present.iter().all(|p| *p) {
        let missing: Vec<&str> = split
            .iter()
            .zip(present.iter())
            .filter(|(_, p)| !**p)
            .map(|(k, _)| k.as_str())
            .collect();
        return Err(format!(
            "{attn}: incomplete split q/k/v projection set, missing {missing:?}"
        ));
    }
    let mut leaves = vec!["weight"];
    leaves.extend(QUANT_SIDECARS.iter().copied());
    for leaf in leaves {
        let keys: Vec<String> = ["q_proj", "k_proj", "v_proj"]
            .iter()
            .map(|p| format!("{attn}.{p}.{leaf}"))
            .collect();
        let have: Vec<bool> = keys.iter().map(|k| weights.contains_key(k)).collect();
        if !have.iter().any(|h| *h) {
            continue;
        }
        if !have.iter().all(|h| *h) {
            return Err(format!(
                "{attn}: split q/k/v `{leaf}` tensors are incomplete; all three or none must \
                 be present"
            ));
        }
        let q = weights
            .remove(&keys[0])
            .ok_or_else(|| format!("missing {}", keys[0]))?;
        let k = weights
            .remove(&keys[1])
            .ok_or_else(|| format!("missing {}", keys[1]))?;
        let v = weights
            .remove(&keys[2])
            .ok_or_else(|| format!("missing {}", keys[2]))?;
        let qk = concatenate(&q, &k, 0);
        let qkv = concatenate(&qk, &v, 0);
        weights.insert(
            format!("{attn}.qkv_proj.{leaf}"),
            ffi::contiguous(&qkv, false),
        );
    }
    Ok(())
}

/// The exact `.weight` key set of a Laguna DFlash checkpoint.
pub fn expected_weight_keys(config: &LagunaDFlashConfig) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    keys.insert("fc.weight".to_string());
    keys.insert("hidden_norm.weight".to_string());
    keys.insert("norm.weight".to_string());
    for i in 0..config.target_layer_ids.len() {
        keys.insert(format!("aux_hidden_norms.{i}.weight"));
    }
    for i in 0..config.num_hidden_layers {
        let p = format!("layers.{i}");
        for leaf in [
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "self_attn.g_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            keys.insert(format!("{p}.{leaf}.weight"));
        }
    }
    keys
}

fn is_projection_key(key: &str) -> bool {
    PROJECTIONS.iter().any(|p| {
        key == format!("{p}.weight")
            || (key.starts_with("layers.") && key.ends_with(&format!(".{p}.weight")))
    })
}

fn check_key_set(weights: &WeightMap, config: &LagunaDFlashConfig) -> Result<(), String> {
    let expected = expected_weight_keys(config);
    let mut allowed: BTreeSet<String> = expected.clone();
    for key in &expected {
        if is_projection_key(key) {
            let stem = key.trim_end_matches("weight");
            for sidecar in QUANT_SIDECARS {
                allowed.insert(format!("{stem}{sidecar}"));
            }
        }
    }
    let actual: BTreeSet<String> = weights.keys().cloned().collect();
    let missing: Vec<&String> = expected.difference(&actual).collect();
    let unexpected: Vec<&String> = actual.difference(&allowed).collect();
    if missing.is_empty() && unexpected.is_empty() {
        return Ok(());
    }
    let mut msg =
        String::from("Laguna DFlash drafter weight keys do not match the expected layout");
    if !missing.is_empty() {
        msg.push_str(&format!("; missing {missing:?}"));
    }
    if !unexpected.is_empty() {
        msg.push_str(&format!("; unexpected {unexpected:?}"));
    }
    Err(msg)
}
