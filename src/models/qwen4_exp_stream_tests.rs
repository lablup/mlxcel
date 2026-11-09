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
    MIN_SLOTS, StreamRequest, arena_bytes_for, mtp_discouraged, should_stream, slots_per_layer,
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

#[test]
fn expert_groups_stay_whole_and_within_capacity() {
    use super::slots::split_expert_groups;
    let sorted = [0, 0, 1, 2, 2, 2, 3, 5, 5, 8];
    let ranges = split_expert_groups(&sorted, 3);
    // Six experts at capacity three is two batches of three experts.
    assert_eq!(ranges, vec![0..6, 6..10]);
    for range in &ranges {
        let mut distinct = sorted[range.clone()].to_vec();
        distinct.dedup();
        assert!(distinct.len() <= 3);
    }
    // An expert's rows are never split across two batches.
    assert_eq!(split_expert_groups(&sorted, 10), vec![0..10]);
    assert!(split_expert_groups(&[], 3).is_empty());
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

fn write_checkpoint(dir: &Path) {
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
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
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
