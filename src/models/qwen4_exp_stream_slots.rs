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

//! Slot bookkeeping of the SSD expert arena (issue #2882): which expert lives
//! in which slot of one layer, and which slots a window may overwrite.
//!
//! Pure logic, so eviction can be tested without Metal.

/// Marks an expert that holds no slot, or a slot that holds no expert.
const NONE: u32 = u32::MAX;

/// The slot assignment for one window of routed experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowMapping {
    /// Slot of each routed expert, in input order.
    pub(crate) slots: Vec<u32>,
    /// `(expert, slot)` pairs that must be read before the window runs.
    pub(crate) misses: Vec<(u32, u32)>,
    /// Routed experts already resident, counted once per distinct expert.
    pub(crate) hits: usize,
}

/// One layer's expert-to-slot table with least-recently-used replacement.
#[derive(Debug)]
pub(crate) struct LayerSlots {
    expert_to_slot: Vec<u32>,
    slot_to_expert: Vec<u32>,
    /// Stamp of the window that last used each slot; 0 means never.
    last_use: Vec<u64>,
    clock: u64,
}

impl LayerSlots {
    pub(crate) fn new(num_experts: usize, num_slots: usize) -> Self {
        Self {
            expert_to_slot: vec![NONE; num_experts],
            slot_to_expert: vec![NONE; num_slots],
            last_use: vec![0; num_slots],
            clock: 0,
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slot_to_expert.len()
    }

    /// Assign a slot to every expert of `experts`, or `None` when the window
    /// routes to more distinct experts than the arena has slots.
    ///
    /// A slot used by this window is never chosen as a victim: hits are
    /// stamped before any eviction, and each eviction stamps its slot.
    pub(crate) fn map_window(&mut self, experts: &[u32]) -> Option<WindowMapping> {
        self.clock += 1;
        let stamp = self.clock;

        let mut distinct = experts.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() > self.capacity() {
            return None;
        }

        let mut hits = 0;
        for &expert in &distinct {
            let slot = self.expert_to_slot[expert as usize];
            if slot != NONE {
                self.last_use[slot as usize] = stamp;
                hits += 1;
            }
        }

        let mut misses = Vec::new();
        for &expert in &distinct {
            if self.expert_to_slot[expert as usize] != NONE {
                continue;
            }
            let victim = self.victim(stamp);
            let previous = self.slot_to_expert[victim];
            if previous != NONE {
                self.expert_to_slot[previous as usize] = NONE;
            }
            self.slot_to_expert[victim] = expert;
            self.expert_to_slot[expert as usize] = victim as u32;
            self.last_use[victim] = stamp;
            misses.push((expert, victim as u32));
        }

        let slots = experts
            .iter()
            .map(|&expert| self.expert_to_slot[expert as usize])
            .collect();
        Some(WindowMapping {
            slots,
            misses,
            hits,
        })
    }

    /// The least recently used slot outside the current window. An empty slot
    /// has stamp 0, so it is always taken before a filled one.
    fn victim(&self, stamp: u64) -> usize {
        self.last_use
            .iter()
            .enumerate()
            .filter(|&(_, &used)| used != stamp)
            .min_by_key(|&(_, &used)| used)
            .map(|(slot, _)| slot)
            .expect("the window fits, so a slot outside it exists")
    }
}

/// Split `rows` (each `k` routed experts wide) into consecutive row ranges
/// that each route to at most `capacity` distinct experts.
///
/// Used when one prefill chunk touches more experts of a layer than the arena
/// has slots: the ranges run one after another, with their output evaluated
/// between them so a later range can reuse the slots.
pub(crate) fn split_rows(
    experts: &[u32],
    k: usize,
    capacity: usize,
) -> Vec<std::ops::Range<usize>> {
    let rows = experts.len() / k.max(1);
    let mut ranges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut start = 0;
    for row in 0..rows {
        let row_experts = &experts[row * k..(row + 1) * k];
        let new = row_experts.iter().filter(|e| !seen.contains(*e)).count();
        // `new` over-counts a row's own duplicates; the top-k of one row has none.
        if seen.len() + new > capacity && row > start {
            ranges.push(start..row);
            start = row;
            seen.clear();
        }
        seen.extend(row_experts.iter().copied());
    }
    if rows > start {
        ranges.push(start..rows);
    }
    ranges
}

/// Split `sorted` (expert ids in ascending order) into contiguous ranges of
/// at most `capacity` distinct experts, cutting only between experts and
/// spreading the experts evenly over the fewest ranges.
///
/// Used for a sorted prefill window larger than the arena: each expert's rows
/// stay together, so every expert's matmul sees the rows the resident path
/// gives it.
pub(crate) fn split_expert_groups(sorted: &[u32], capacity: usize) -> Vec<std::ops::Range<usize>> {
    let mut starts = Vec::new();
    for (index, expert) in sorted.iter().enumerate() {
        if index == 0 || sorted[index - 1] != *expert {
            starts.push(index);
        }
    }
    let groups = starts.len();
    if groups == 0 {
        return Vec::new();
    }
    let batches = groups.div_ceil(capacity.max(1));
    let mut ranges = Vec::with_capacity(batches);
    for batch in 0..batches {
        let first = batch * groups / batches;
        let last = (batch + 1) * groups / batches;
        let end = starts.get(last).copied().unwrap_or(sorted.len());
        ranges.push(starts[first]..end);
    }
    ranges
}
