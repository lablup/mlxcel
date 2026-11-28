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

//! Unit tests for the expert arena (issue #2882).

use std::path::Path;

use super::slots::{LayerSlots, split_rows};
use super::source::{ExpertSource, PLANE_SUFFIXES, write_contiguous_file};
use super::{
    MIN_SLOTS, StreamRequest, arena_bytes_for, check_streamable, mtp_discouraged, should_stream,
    slots_per_layer,
};

#[test]
fn a_window_never_evicts_its_own_slots() {
    let mut slots = LayerSlots::new(8, 2);
    // Fill both slots.
    slots.map_window(&[0, 1]).unwrap();
    // Expert 2 must evict expert 1, because expert 0 is in this window.
    let mapping = slots.map_window(&[0, 2]).unwrap();
    assert_eq!(mapping.misses, vec![(2, 1)]);
    assert_eq!(mapping.hits, 1);
    assert_eq!(mapping.slots[0], 0);
    assert_eq!(mapping.slots[1], 1);
    // And when both window experts are resident, nothing moves.
    let again = slots.map_window(&[2, 0]).unwrap();
    assert!(again.misses.is_empty());
    assert_eq!(again.slots, vec![1, 0]);
}

#[test]
fn an_evicted_expert_refills_on_its_next_use() {
    let mut slots = LayerSlots::new(4, 1);
    slots.map_window(&[0]).unwrap();
    slots.map_window(&[1]).unwrap();
    let mapping = slots.map_window(&[0]).unwrap();
    assert_eq!(mapping.misses.len(), 1);
    assert_eq!(mapping.misses[0].0, 0);
}

#[test]
fn eviction_takes_the_least_recently_used_slot() {
    let mut slots = LayerSlots::new(8, 3);
    slots.map_window(&[0, 1, 2]).unwrap();
    slots.map_window(&[0, 2]).unwrap();
    let mapping = slots.map_window(&[5]).unwrap();
    // Expert 1 went untouched longest.
    assert_eq!(mapping.misses, vec![(5, 1)]);
}

#[test]
fn remapped_indices_recover_the_original_routing() {
    let mut slots = LayerSlots::new(32, 8);
    let routing = [3, 7, 3, 20, 7, 31];
    let mapping = slots.map_window(&routing).unwrap();
    let mut slot_to_expert = [u32::MAX; 8];
    for (expert, slot) in &mapping.misses {
        slot_to_expert[*slot as usize] = *expert;
    }
    let recovered: Vec<u32> = mapping
        .slots
        .iter()
        .map(|slot| slot_to_expert[*slot as usize])
        .collect();
    assert_eq!(recovered, routing);
}

#[test]
fn a_window_larger_than_the_arena_is_refused() {
    let mut slots = LayerSlots::new(8, 2);
    assert!(slots.map_window(&[0, 1, 2]).is_none());
}

#[test]
fn split_rows_keeps_each_range_within_capacity_and_covers_every_row() {
    // Four rows of two experts; capacity three.
    let experts = [0, 1, 2, 3, 4, 5, 0, 1];
    let ranges = split_rows(&experts, 2, 3);
    // No two neighbouring rows fit together: each adds two new experts to two.
    assert_eq!(ranges, vec![0..1, 1..2, 2..3, 3..4]);
    let covered: usize = ranges.iter().map(|r| r.end - r.start).sum();
    assert_eq!(covered, 4);
    for range in &ranges {
        let mut distinct = experts[range.start * 2..range.end * 2].to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert!(distinct.len() <= 3);
    }
}

/// Expand `(expert, rows)` into the ascending ids a sorted window holds.
fn sorted_window(groups: &[(u32, usize)]) -> Vec<u32> {
    groups
        .iter()
        .flat_map(|&(expert, rows)| std::iter::repeat_n(expert, rows))
        .collect()
}

/// Every position once, no expert split, at most `capacity` experts per batch.
fn assert_batches_valid(sorted: &[u32], batches: &[Vec<usize>], capacity: usize) {
    let mut seen: Vec<usize> = batches.iter().flatten().copied().collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..sorted.len()).collect::<Vec<_>>());
    let mut owner = std::collections::HashMap::new();
    for (index, batch) in batches.iter().enumerate() {
        let mut experts: Vec<u32> = batch.iter().map(|&p| sorted[p]).collect();
        experts.dedup();
        assert!(experts.len() <= capacity);
        for expert in experts {
            assert_eq!(*owner.entry(expert).or_insert(index), index, "expert split");
        }
    }
}

#[test]
fn expert_groups_stay_whole_and_within_capacity() {
    use super::slots::plan_expert_batches;
    let sorted = [0, 0, 1, 2, 2, 2, 3, 5, 5, 8];
    let batches = plan_expert_batches(&sorted, 3, None);
    assert_eq!(batches.len(), 2);
    assert_batches_valid(&sorted, &batches, 3);
    assert_eq!(
        plan_expert_batches(&sorted, 10, None),
        vec![(0..10).collect::<Vec<_>>()]
    );
    assert!(plan_expert_batches(&[], 3, None).is_empty());
}

#[test]
fn batches_of_rare_experts_are_balanced_into_the_sorted_kernel_band() {
    use super::slots::{batches_within_ratio, plan_expert_batches};
    // Popular experts first, so cutting in expert order would leave whole
    // batches of rare experts below the four rows per slot the kernel needs.
    let groups: Vec<(u32, usize)> = (0..256u32)
        .map(|expert| (expert, if expert < 32 { 64 } else { 2 }))
        .collect();
    let sorted = sorted_window(&groups);
    let batches = plan_expert_batches(&sorted, 64, Some(4..64));
    assert_eq!(batches.len(), 4);
    assert_batches_valid(&sorted, &batches, 64);
    assert!(batches_within_ratio(&batches, 64, &(4..64)));
}

#[test]
fn a_batch_never_reaches_the_tall_tile_the_resident_tensor_would_not_use() {
    use super::slots::{batches_within_ratio, plan_expert_batches};
    let groups: Vec<(u32, usize)> = (0..64u32).map(|expert| (expert, 320)).collect();
    let sorted = sorted_window(&groups);
    let unbounded = plan_expert_batches(&sorted, 64, None);
    assert_eq!(unbounded.len(), 1, "one batch fits the slots");
    assert!(!batches_within_ratio(&unbounded, 64, &(4..64)));
    let batches = plan_expert_batches(&sorted, 64, Some(4..64));
    assert_batches_valid(&sorted, &batches, 64);
    assert!(batches_within_ratio(&batches, 64, &(4..64)));
}

#[test]
fn a_band_out_of_reach_is_reported_rather_than_met() {
    use super::slots::{batches_within_ratio, plan_expert_batches};
    let groups: Vec<(u32, usize)> = (0..128u32).map(|expert| (expert, 1)).collect();
    let sorted = sorted_window(&groups);
    let batches = plan_expert_batches(&sorted, 64, Some(4..64));
    assert_batches_valid(&sorted, &batches, 64);
    assert!(!batches_within_ratio(&batches, 64, &(4..64)));
}

#[test]
fn the_decision_rule_is_explicit_then_the_working_set_share() {
    assert!(should_stream(StreamRequest::On, 0, 1000));
    assert!(!should_stream(StreamRequest::Off, 1 << 40, 1000));
    assert!(!should_stream(StreamRequest::Auto, 599, 1000));
    assert!(should_stream(StreamRequest::Auto, 601, 1000));
    assert!(!should_stream(StreamRequest::Auto, 1 << 40, 0));
}

#[test]
fn the_environment_values_parse_and_bad_ones_are_refused() {
    assert_eq!(StreamRequest::parse(None).unwrap(), StreamRequest::Auto);
    assert_eq!(
        StreamRequest::parse(Some("auto")).unwrap(),
        StreamRequest::Auto
    );
    assert_eq!(StreamRequest::parse(Some("1")).unwrap(), StreamRequest::On);
    assert_eq!(
        StreamRequest::parse(Some("off")).unwrap(),
        StreamRequest::Off
    );
    assert!(StreamRequest::parse(Some("maybe")).is_err());
}

#[test]
fn slots_round_down_and_stay_within_bounds() {
    let expert = 1_700_000;
    // 12 GB over 48 layers is about 147 experts; rounded down to 144.
    assert_eq!(slots_per_layer(12_000_000_000, 48, expert, 512), 144);
    // Tiny arenas hold the floor; huge ones stop at the expert count.
    assert_eq!(slots_per_layer(1, 48, expert, 512), MIN_SLOTS);
    assert_eq!(slots_per_layer(u64::MAX, 48, expert, 512), 512);
}

#[test]
fn the_arena_never_exceeds_the_experts_without_a_memory_probe() {
    assert_eq!(arena_bytes_for(None, 10, 100), 100);
}

#[test]
fn a_drafter_is_discouraged_only_when_streaming_below_96_gb() {
    assert!(mtp_discouraged(64 << 30, true));
    assert!(!mtp_discouraged(64 << 30, false));
    assert!(!mtp_discouraged(128 << 30, true));
}

/// Plane row sizes of the synthetic checkpoint; every value is arbitrary but
/// each plane gets distinct bytes so a swapped offset would be caught.
const ROW_BYTES: [usize; 9] = [16, 8, 8, 16, 8, 8, 32, 16, 16];
const EXPERTS: usize = 6;
const LAYERS: usize = 2;

fn byte_of(layer: usize, plane: usize, expert: usize, index: usize) -> u8 {
    (layer * 97 + plane * 31 + expert * 13 + index * 7) as u8
}

/// The header and payload of the synthetic checkpoint, for tests to alter.
fn checkpoint_parts() -> (serde_json::Map<String, serde_json::Value>, Vec<u8>) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for layer in 0..LAYERS {
        for (plane, suffix) in PLANE_SUFFIXES.iter().enumerate() {
            let start = data.len();
            for expert in 0..EXPERTS {
                for index in 0..ROW_BYTES[plane] {
                    data.push(byte_of(layer, plane, expert, index));
                }
            }
            header.insert(
                format!("model.layers.{layer}.mlp.switch_mlp.{suffix}"),
                serde_json::json!({
                    "dtype": "U8",
                    "shape": [EXPERTS, 1, ROW_BYTES[plane]],
                    "data_offsets": [start, data.len()],
                }),
            );
        }
    }
    (header, data)
}

fn write_shard(path: &Path, header: &serde_json::Map<String, serde_json::Value>, data: &[u8]) {
    let header = serde_json::to_vec(header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(data);
    std::fs::write(path, file).unwrap();
}

fn write_checkpoint(dir: &Path) {
    let (header, data) = checkpoint_parts();
    write_shard(&dir.join("model.safetensors"), &header, &data);
}

fn expected_record(layer: usize, expert: usize) -> Vec<u8> {
    (0..PLANE_SUFFIXES.len())
        .flat_map(|plane| {
            (0..ROW_BYTES[plane]).map(move |index| byte_of(layer, plane, expert, index))
        })
        .collect()
}

fn read_record(source: &ExpertSource, layer: usize, expert: usize) -> Vec<u8> {
    let mut planes: Vec<Vec<u8>> = ROW_BYTES.iter().map(|len| vec![0; *len]).collect();
    let mut dest: Vec<&mut [u8]> = planes.iter_mut().map(Vec::as_mut_slice).collect();
    source.read_expert(layer, expert, &mut dest).unwrap();
    planes.concat()
}

#[test]
fn the_plane_table_reads_every_expert_exactly() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    let source = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap();
    assert!(!source.uses_contiguous_file());
    let record: usize = ROW_BYTES.iter().sum();
    assert_eq!(
        source.total_expert_bytes(),
        (record * EXPERTS * LAYERS) as u64
    );
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            assert_eq!(
                read_record(&source, layer, expert),
                expected_record(layer, expert)
            );
        }
    }
}

#[test]
fn the_contiguous_file_round_trips_every_expert_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    let report = write_contiguous_file(dir.path(), LAYERS, EXPERTS).unwrap();
    assert_eq!(report.layers, LAYERS);
    let source = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap();
    assert!(
        source.uses_contiguous_file(),
        "the manifest must be picked up"
    );
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            assert_eq!(
                read_record(&source, layer, expert),
                expected_record(layer, expert)
            );
        }
    }
}

#[test]
fn a_contiguous_file_for_another_checkpoint_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    write_contiguous_file(dir.path(), LAYERS, EXPERTS).unwrap();
    let manifest = dir.path().join(super::source::CONTIGUOUS_MANIFEST);
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    value["fingerprint"] = serde_json::json!("different");
    std::fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let source = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap();
    assert!(!source.uses_contiguous_file());
}

#[test]
fn a_contiguous_file_is_ignored_once_its_shard_is_replaced() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    write_contiguous_file(dir.path(), LAYERS, EXPERTS).unwrap();
    assert!(
        ExpertSource::open(dir.path(), LAYERS, EXPERTS)
            .unwrap()
            .uses_contiguous_file()
    );
    // Same layout and size, as a re-quantized checkpoint would have.
    let shard = std::fs::File::options()
        .write(true)
        .open(dir.path().join("model.safetensors"))
        .unwrap();
    shard
        .set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(60))
        .unwrap();
    drop(shard);
    assert!(
        !ExpertSource::open(dir.path(), LAYERS, EXPERTS)
            .unwrap()
            .uses_contiguous_file()
    );
}

#[test]
fn a_truncated_contiguous_file_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    let report = write_contiguous_file(dir.path(), LAYERS, EXPERTS).unwrap();
    let file = std::fs::File::options()
        .write(true)
        .open(&report.path)
        .unwrap();
    file.set_len(report.bytes - 1).unwrap();
    drop(file);
    assert!(
        !ExpertSource::open(dir.path(), LAYERS, EXPERTS)
            .unwrap()
            .uses_contiguous_file()
    );
}

#[test]
fn a_checkpoint_without_stacked_planes_is_refused_by_name() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), {
        let header = b"{}";
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header);
        file
    })
    .unwrap();
    let error = ExpertSource::open(dir.path(), 1, 4).unwrap_err();
    assert!(error.contains("stacked switch_mlp"), "{error}");
}

/// The first plane of the first layer: 16 bytes per expert, at offset zero.
const FIRST_KEY: &str = "model.layers.0.mlp.switch_mlp.gate_proj.weight";

/// `open` on a checkpoint whose first plane has `field` set to `value`.
fn open_with(field: &str, value: serde_json::Value) -> Result<ExpertSource, String> {
    let dir = tempfile::tempdir().unwrap();
    let (mut header, data) = checkpoint_parts();
    header[FIRST_KEY][field] = value;
    write_shard(&dir.path().join("model.safetensors"), &header, &data);
    ExpertSource::open(dir.path(), LAYERS, EXPERTS)
}

#[test]
fn a_shape_that_disagrees_with_the_stored_bytes_is_refused() {
    use serde_json::json;
    for (shape, expected) in [
        (json!([EXPERTS, 2, 16]), "does not match shape"),
        (json!([EXPERTS, 1, 1 << 20]), "does not match shape"),
        (json!([EXPERTS, 1, 0]), "out of range"),
        (json!([EXPERTS, -1, -16]), "out of range"),
        (json!([EXPERTS, 1u64 << 33, 1]), "out of range"),
        (json!([EXPERTS, "1", 16]), "non-integer"),
        (json!([EXPERTS, 1.5, 16]), "non-integer"),
        (json!([EXPERTS, "1", 1, 16]), "non-integer"),
    ] {
        let error = open_with("shape", shape).unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
    // Two bytes per element: 16 bytes are 8 elements, not 16.
    let error = open_with("dtype", json!("U16")).unwrap_err();
    assert!(error.contains("does not match shape"), "{error}");
}

#[test]
fn offsets_past_the_end_of_the_file_are_refused() {
    use serde_json::json;
    let row = (ROW_BYTES[0] * EXPERTS) as u64;
    let (_, data) = checkpoint_parts();
    let past = data.len() as u64;
    let error = open_with("data_offsets", json!([past, past + row])).unwrap_err();
    assert!(error.contains("past the end"), "{error}");
    let top = u64::MAX;
    let error = open_with("data_offsets", json!([top - row, top])).unwrap_err();
    assert!(error.contains("past the end"), "{error}");
}

#[test]
fn a_duplicate_alias_of_a_plane_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (mut header, data) = checkpoint_parts();
    let plane = header[FIRST_KEY].clone();
    header.insert(format!("language_model.{FIRST_KEY}"), plane);
    write_shard(&dir.path().join("model.safetensors"), &header, &data);
    let error = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap_err();
    assert!(error.contains("duplicate"), "{error}");
}

#[test]
fn a_header_length_beyond_the_file_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut file = u64::MAX.to_le_bytes().to_vec();
    file.extend_from_slice(b"{}");
    std::fs::write(dir.path().join("model.safetensors"), file).unwrap();
    let error = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap_err();
    assert!(error.contains("header size"), "{error}");
}

#[test]
fn an_index_naming_a_shard_outside_the_model_is_refused() {
    let outer = tempfile::tempdir().unwrap();
    let model = outer.path().join("model");
    std::fs::create_dir(&model).unwrap();
    // A well-formed shard waits at the escaping path, so only containment can
    // refuse it.
    write_checkpoint(outer.path());
    let absolute = outer.path().join("model.safetensors");
    for name in ["../model.safetensors", absolute.to_str().unwrap()] {
        let index = serde_json::json!({ "weight_map": { "a": name } });
        std::fs::write(
            model.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        let error = ExpertSource::open(&model, LAYERS, EXPERTS).unwrap_err();
        assert!(error.contains("relative path inside"), "{error}");
    }
}

#[test]
fn impossible_counts_are_errors_rather_than_panics() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    assert!(ExpertSource::open(dir.path(), LAYERS, 0).is_err());
    assert!(ExpertSource::open(dir.path(), 0, EXPERTS).is_err());
    assert!(ExpertSource::open(dir.path(), 4097, EXPERTS).is_err());
    assert!(ExpertSource::open(dir.path(), usize::MAX, EXPERTS).is_err());
}

#[test]
fn fewer_experts_than_the_slot_floor_do_not_panic() {
    assert_eq!(slots_per_layer(1, 4, 1024, 32), 32);
    assert_eq!(slots_per_layer(u64::MAX, 4, 1024, 32), 32);
    assert_eq!(slots_per_layer(1, 4, 1024, 0), 0);
    assert_eq!(slots_per_layer(1, 4, 1024, 64), MIN_SLOTS);
}

#[test]
fn a_manifest_with_an_inconsistent_layer_entry_is_ignored() {
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    write_contiguous_file(dir.path(), LAYERS, EXPERTS).unwrap();
    let manifest = dir.path().join(super::source::CONTIGUOUS_MANIFEST);
    let good: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    let record: u64 = ROW_BYTES.iter().sum::<usize>() as u64;
    for (field, layer, value) in [
        ("offset", 1, json!(record * EXPERTS as u64 - 1)),
        ("offset", 0, json!(1)),
        ("record_bytes", 0, json!(record + 1)),
        ("record_bytes", 1, json!(u64::MAX)),
    ] {
        let mut altered = good.clone();
        altered["layers"][layer][field] = value;
        std::fs::write(&manifest, serde_json::to_vec(&altered).unwrap()).unwrap();
        let source = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap();
        assert!(!source.uses_contiguous_file(), "{field} of layer {layer}");
    }
    let mut altered = good.clone();
    altered["num_experts"] = json!(EXPERTS + 1);
    std::fs::write(&manifest, serde_json::to_vec(&altered).unwrap()).unwrap();
    let source = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap();
    assert!(!source.uses_contiguous_file(), "num_experts");
    std::fs::write(&manifest, serde_json::to_vec(&good).unwrap()).unwrap();
    let source = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap();
    assert!(source.uses_contiguous_file(), "the unaltered manifest");
}

#[cfg(unix)]
#[test]
fn a_contiguous_file_linked_outside_the_model_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    let report = write_contiguous_file(dir.path(), LAYERS, EXPERTS).unwrap();
    let moved = elsewhere.path().join("records");
    std::fs::rename(&report.path, &moved).unwrap();
    std::os::unix::fs::symlink(&moved, &report.path).unwrap();
    let source = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap();
    assert!(!source.uses_contiguous_file());
}

#[cfg(unix)]
#[test]
fn writing_the_contiguous_file_replaces_planted_symlinks() {
    use super::source::{CONTIGUOUS_FILE, CONTIGUOUS_MANIFEST};
    let dir = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    for (name, victim) in [(CONTIGUOUS_FILE, "data"), (CONTIGUOUS_MANIFEST, "manifest")] {
        let victim = elsewhere.path().join(victim);
        std::fs::write(&victim, b"precious").unwrap();
        std::os::unix::fs::symlink(&victim, dir.path().join(name)).unwrap();
    }
    write_contiguous_file(dir.path(), LAYERS, EXPERTS).unwrap();
    for name in ["data", "manifest"] {
        assert_eq!(
            std::fs::read(elsewhere.path().join(name)).unwrap(),
            b"precious"
        );
    }
    for name in [CONTIGUOUS_FILE, CONTIGUOUS_MANIFEST] {
        let meta = std::fs::symlink_metadata(dir.path().join(name)).unwrap();
        assert!(meta.file_type().is_file(), "{name} must be a regular file");
    }
    let source = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap();
    assert!(source.uses_contiguous_file());
}

#[test]
fn only_a_whole_page_arena_of_at_least_top_k_slots_can_stream() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    let source = ExpertSource::open(dir.path(), LAYERS, EXPERTS).unwrap();
    // The smallest plane is 8 bytes a row, so 2048 slots are one 16 KiB page.
    assert!(check_streamable(&source, &[2048, 2048], 4).is_ok());
    let error = check_streamable(&source, &[2048, 2], 4).unwrap_err();
    assert!(error.contains("layer 1"), "{error}");
    let error = check_streamable(&source, &[64, 2048], 4).unwrap_err();
    assert!(error.contains("whole number"), "{error}");
}

/// Restores `EXPERT_STREAM_ENV` on drop. Declare it after the `env_lock` guard.
#[cfg(not(target_os = "macos"))]
struct ExpertStreamEnvRestore(Option<std::ffi::OsString>);

#[cfg(not(target_os = "macos"))]
impl Drop for ExpertStreamEnvRestore {
    fn drop(&mut self) {
        // SAFETY: serialized via env_lock, which the owner of this guard still holds.
        unsafe {
            match self.0.take() {
                Some(value) => std::env::set_var(super::EXPERT_STREAM_ENV, value),
                None => std::env::remove_var(super::EXPERT_STREAM_ENV),
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn resolve_refuses_streaming_off_metal() {
    let _lock = crate::test_support::env_lock::env_lock();
    let _restore = ExpertStreamEnvRestore(std::env::var_os(super::EXPERT_STREAM_ENV));
    let dir = Path::new("/nonexistent");

    // SAFETY: serialized via env_lock.
    unsafe { std::env::set_var(super::EXPERT_STREAM_ENV, "1") };
    let error =
        super::ExpertStream::resolve(dir, 2, 8, 2, 0).expect_err("=1 must be refused off Metal");
    assert!(error.contains("Metal only"), "unexpected error: {error}");

    // SAFETY: serialized via env_lock.
    unsafe { std::env::set_var(super::EXPERT_STREAM_ENV, "auto") };
    let resolved = super::ExpertStream::resolve(dir, 2, 8, 2, 0).expect("auto must not error");
    assert!(resolved.is_none(), "auto must stay resident off Metal");
}
