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

//! [#1564] Tests for the weight materialization scope.
//!
//! The scope's decisions (which tensors it evaluates, which it leaves lazy) are
//! checked through the real weights loader on a small safetensors file, so they
//! run on whatever device MLX defaults to, CPU included.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::weight_materialize::{
    MaterializationPolicy, MaterializationScope, WeightTransferStats, is_multimodal_tower_tensor,
};
use crate::weights::{WeightMap, load_weights_from_dir};

fn temp_dir(tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "mlxcel_weight_materialize_{tag}_{}_{seq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write `names`, each an f32 vector of four values `[k, k+1, k+2, k+3]`.
fn write_checkpoint(dir: &Path, names: &[&str]) {
    let mut data: Vec<u8> = Vec::new();
    let mut header = serde_json::Map::new();
    for (k, name) in names.iter().enumerate() {
        let start = data.len();
        for i in 0..4 {
            data.extend_from_slice(&((k * 10 + i) as f32).to_le_bytes());
        }
        header.insert(
            (*name).to_string(),
            serde_json::json!({"dtype": "F32", "shape": [4], "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_string(&header).unwrap();
    let mut file = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&data).unwrap();
}

const NAMES: &[&str] = &[
    "language_model.model.embed_tokens.weight",
    "language_model.model.norm.weight",
    "language_model.mtp.fc.weight",
    "vision_tower.blocks.0.attn.qkv.weight",
];

/// Load `NAMES` inside a scope, keep everything but the MTP tensor, and
/// finish the scope while the kept tensors are still held.
fn load_and_finish(
    policy: MaterializationPolicy,
) -> (crate::weight_materialize::MaterializationReport, WeightMap) {
    let dir = temp_dir(policy.as_str());
    write_checkpoint(&dir, NAMES);
    let scope = MaterializationScope::begin_with(policy);
    let mut weights = load_weights_from_dir(&dir).unwrap();
    weights.remove("language_model.mtp.fc.weight");
    let report = scope.finish().expect("outermost scope reports");
    std::fs::remove_dir_all(&dir).ok();
    (report, weights)
}

#[test]
fn eager_materializes_kept_text_tensors_only() {
    let (report, weights) = load_and_finish(MaterializationPolicy::Eager);
    assert_eq!(report.registered, 4);
    assert_eq!(report.materialized, 2, "{report:?}");
    assert_eq!(report.materialized_bytes, 2 * 16);
    assert_eq!(report.skipped_multimodal, 1, "vision tower stays lazy");
    assert_eq!(
        report.skipped_unreferenced, 1,
        "dropped MTP tensor is not read"
    );
    // The materialized arrays still hold the checkpoint values.
    let norm = &weights["language_model.model.norm.weight"];
    assert_eq!(
        crate::utils::array_to_vec_f32(norm),
        vec![10.0, 11.0, 12.0, 13.0]
    );
    // The lazy vision tensor still loads on first use.
    let vision = &weights["vision_tower.blocks.0.attn.qkv.weight"];
    assert_eq!(
        crate::utils::array_to_vec_f32(vision),
        vec![30.0, 31.0, 32.0, 33.0]
    );
}

#[test]
fn lazy_policy_materializes_nothing() {
    let (report, weights) = load_and_finish(MaterializationPolicy::Lazy);
    assert_eq!(report.registered, 4);
    assert_eq!(report.materialized, 0);
    assert_eq!(report.skipped_multimodal + report.skipped_unreferenced, 0);
    let embed = &weights["language_model.model.embed_tokens.weight"];
    assert_eq!(
        crate::utils::array_to_vec_f32(embed),
        vec![0.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn nested_scope_is_inert_and_outer_scope_sees_everything() {
    let dir = temp_dir("nested");
    write_checkpoint(&dir, NAMES);
    let outer = MaterializationScope::begin_with(MaterializationPolicy::Eager);
    let inner = MaterializationScope::begin_with(MaterializationPolicy::Eager);
    let weights = load_weights_from_dir(&dir).unwrap();
    assert!(
        inner.finish().is_none(),
        "a nested scope must not finish the load"
    );
    let report = outer.finish().unwrap();
    assert_eq!(report.registered, NAMES.len());
    assert_eq!(report.materialized, 3);
    drop(weights);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dropped_scope_discards_registry_and_later_loads_are_unregistered() {
    let dir = temp_dir("dropped");
    write_checkpoint(&dir, NAMES);
    {
        let _scope = MaterializationScope::begin_with(MaterializationPolicy::Eager);
        let _ = load_weights_from_dir(&dir).unwrap();
        // A failed load returns early here; nothing is evaluated.
    }
    let fresh = MaterializationScope::begin_with(MaterializationPolicy::Eager);
    let report = fresh.finish().unwrap();
    assert_eq!(
        report.registered, 0,
        "the dropped scope's entries must not leak"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn loads_outside_a_scope_register_nothing() {
    let dir = temp_dir("outside");
    write_checkpoint(&dir, NAMES);
    let weights = load_weights_from_dir(&dir).unwrap();
    let scope = MaterializationScope::begin_with(MaterializationPolicy::Eager);
    let report = scope.finish().unwrap();
    assert_eq!(report.registered, 0);
    drop(weights);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn policy_parsing() {
    assert_eq!(
        MaterializationPolicy::parse(None).unwrap(),
        MaterializationPolicy::Eager
    );
    assert_eq!(
        MaterializationPolicy::parse(Some("")).unwrap(),
        MaterializationPolicy::Eager
    );
    assert_eq!(
        MaterializationPolicy::parse(Some(" LAZY ")).unwrap(),
        MaterializationPolicy::Lazy
    );
    assert_eq!(
        MaterializationPolicy::parse(Some("eager")).unwrap(),
        MaterializationPolicy::Eager
    );
    let error = MaterializationPolicy::parse(Some("sometimes")).unwrap_err();
    assert!(error.contains("MLXCEL_WEIGHT_MATERIALIZATION"), "{error}");
}

#[test]
fn multimodal_tower_names() {
    for name in [
        "vision_tower.blocks.0.attn.qkv.weight",
        "model.visual.blocks.0.mlp.fc1.weight",
        "model.vision_tower.encoder.layers.0.weight",
        "model.encoder.vision_tower.patch_embed.weight",
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "multi_modal_projector.linear_1.weight",
        "audio_tower.layers.0.fc1.weight",
        "model.embed_vision.embedding_projection.weight",
    ] {
        assert!(is_multimodal_tower_tensor(name), "{name}");
    }
    for name in [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "model.layers.3.mlp.switch_mlp.gate_proj.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
        // A tower name deeper than the third component is a layer's own name.
        "model.layers.0.visual.weight",
    ] {
        assert!(!is_multimodal_tower_tensor(name), "{name}");
    }
}

#[test]
fn transfer_stats_difference_and_fields() {
    let earlier = WeightTransferStats {
        tensors: 10,
        bytes: 1_000,
        pinned_tensors: 9,
        pageable_tensors: 1,
        memcpy_calls: 12,
        read_ns: 5_000_000,
        wait_ns: 1_000_000,
        total_ns: 7_000_000,
    };
    let later = WeightTransferStats {
        tensors: 1_857,
        bytes: 15_130_000_000 + 1_000,
        pinned_tensors: 1_856,
        pageable_tensors: 1,
        memcpy_calls: 2_000,
        read_ns: 2_005_000_000,
        wait_ns: 301_000_000,
        total_ns: 2_507_000_000,
    };
    let delta = later.since(&earlier);
    assert_eq!(delta.tensors, 1_847);
    assert_eq!(delta.pageable_tensors, 0);
    let line = delta.format_fields();
    for field in [
        "weights=1847/15.13GB",
        "staging=2000.00ms",
        "h2d_wait=300.00ms",
        "upload=2500.00ms",
        "pinned=1847",
        "pageable=0",
        "memcpy=1988",
    ] {
        assert!(line.contains(field), "missing {field} in {line}");
    }
    // Counters never run backwards; a reversed pair saturates to zero.
    assert_eq!(earlier.since(&later), WeightTransferStats::default());
}
