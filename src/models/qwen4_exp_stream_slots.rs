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

/// Divide `sorted` (expert ids in ascending order) into batches of positions
/// holding at most `capacity` distinct experts, keeping each expert's rows in
/// one batch.
///
/// Used for a sorted prefill window. MLX picks the sorted matmul and its tile
/// height from `rows / tensor experts`, so a batch run on a `capacity`-slot
/// tensor only gets the resident kernel when its own ratio falls in the same
/// band. `ratio`, when given, is that band: the fewest batches that respect
/// the capacity and the band's upper end are filled largest expert first into
/// the lightest batch, because batches of neighbouring rare experts would
/// otherwise fall below the band's lower end. Use [`batches_within_ratio`] to
/// learn whether the result met it.
pub(crate) fn plan_expert_batches(
    sorted: &[u32],
    capacity: usize,
    ratio: Option<std::ops::Range<usize>>,
) -> Vec<Vec<usize>> {
    let mut starts = Vec::new();
    for (index, expert) in sorted.iter().enumerate() {
        if index == 0 || sorted[index - 1] != *expert {
            starts.push(index);
        }
    }
    if starts.is_empty() {
        return Vec::new();
    }
    let capacity = capacity.max(1);
    let group_end = |group: usize| starts.get(group + 1).copied().unwrap_or(sorted.len());
    let mut count = starts.len().div_ceil(capacity);
    if let Some(ratio) = ratio
        && ratio.end != usize::MAX
    {
        let most_rows = (ratio.end.saturating_mul(capacity) - 1).max(1);
        count = count.max(sorted.len().div_ceil(most_rows));
    }

    let mut by_size: Vec<usize> = (0..starts.len()).collect();
    by_size.sort_by_key(|&group| std::cmp::Reverse(group_end(group) - starts[group]));
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut rows = vec![0usize; count];
    for group in by_size {
        let lightest = (0..count)
            .filter(|&batch| groups[batch].len() < capacity)
            .min_by_key(|&batch| rows[batch])
            .expect("count batches of capacity groups hold every group");
        groups[lightest].push(group);
        rows[lightest] += group_end(group) - starts[group];
    }
    groups
        .into_iter()
        .map(|mut batch| {
            batch.sort_unstable();
            batch
                .into_iter()
                .flat_map(|group| starts[group]..group_end(group))
                .collect()
        })
        .collect()
}

/// Whether every batch's `rows / capacity` lies inside `ratio`.
pub(crate) fn batches_within_ratio(
    batches: &[Vec<usize>],
    capacity: usize,
    ratio: &std::ops::Range<usize>,
) -> bool {
    batches
        .iter()
        .all(|batch| ratio.contains(&(batch.len() / capacity.max(1))))
}
