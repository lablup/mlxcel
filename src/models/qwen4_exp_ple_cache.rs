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

//! Bounded LRU cache of PLE row records, sized in bytes and grown in steps.
//!
//! Records live in step-sized slabs rather than per-row allocations, so a
//! half-million-row cache costs one allocation per step and eviction reuses a
//! slot in place. Growth goes through [`RowCacheGrowth`]: when the policy
//! refuses the next step the cache stops growing and evicts instead.

use std::collections::HashMap;
use std::sync::Arc;

/// Bytes of memory each growth step adds.
pub const GROWTH_STEP_BYTES: u64 = 64 * 1024 * 1024;

/// Bookkeeping charged per record against the byte budget, so slabs plus
/// bookkeeping never exceed it: 16 bytes for the key and the two list links
/// (reserved a step at a time) and 39 for a hash map entry at worst, 17 bytes
/// per bucket with up to `2 * 8 / 7` buckets per record.
const ENTRY_OVERHEAD_BYTES: u64 = 56;

const NONE: u32 = u32::MAX;

/// Decides whether the cache may take another growth step. The default always
/// agrees; #2883 plugs in its availability check here.
pub trait RowCacheGrowth: Send + Sync {
    fn can_grow(&self, next_step_bytes: u64) -> bool {
        let _ = next_step_bytes;
        true
    }
}

/// [`RowCacheGrowth`] that always agrees.
#[derive(Debug, Default)]
pub struct AlwaysGrow;
impl RowCacheGrowth for AlwaysGrow {}

pub(crate) struct RowCache {
    stride: usize,
    capacity_rows: u32,
    slab_rows: u32,
    slabs: Vec<Box<[u8]>>,
    map: HashMap<u64, u32>,
    keys: Vec<u64>,
    prev: Vec<u32>,
    next: Vec<u32>,
    head: u32,
    tail: u32,
    growth: Arc<dyn RowCacheGrowth>,
}

impl RowCache {
    pub(crate) fn new(budget_bytes: u64, stride: usize, growth: Arc<dyn RowCacheGrowth>) -> Self {
        let per_row = (stride as u64).saturating_add(ENTRY_OVERHEAD_BYTES).max(1);
        let capacity_rows = (budget_bytes / per_row).min(u64::from(u32::MAX - 1)) as u32;
        let slab_rows = (GROWTH_STEP_BYTES / (stride.max(1) as u64))
            .clamp(1, u64::from(capacity_rows.max(1))) as u32;
        Self {
            stride,
            capacity_rows,
            slab_rows,
            slabs: Vec::new(),
            map: HashMap::new(),
            keys: Vec::new(),
            prev: Vec::new(),
            next: Vec::new(),
            head: NONE,
            tail: NONE,
            growth,
        }
    }

    pub(crate) fn capacity_rows(&self) -> u32 {
        self.capacity_rows
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    /// Bytes of record storage currently allocated.
    #[cfg(test)]
    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.slabs.iter().map(|slab| slab.len() as u64).sum()
    }

    fn slot_range(&self, slot: u32) -> (usize, std::ops::Range<usize>) {
        let slab = (slot / self.slab_rows) as usize;
        let within = (slot % self.slab_rows) as usize * self.stride;
        (slab, within..within + self.stride)
    }

    fn unlink(&mut self, slot: u32) {
        let (prev, next) = (self.prev[slot as usize], self.next[slot as usize]);
        match prev {
            NONE => self.head = next,
            _ => self.next[prev as usize] = next,
        }
        match next {
            NONE => self.tail = prev,
            _ => self.prev[next as usize] = prev,
        }
    }

    fn push_front(&mut self, slot: u32) {
        self.prev[slot as usize] = NONE;
        self.next[slot as usize] = self.head;
        match self.head {
            NONE => self.tail = slot,
            head => self.prev[head as usize] = slot,
        }
        self.head = slot;
    }

    /// Copy the record for `id` into `dest` and mark it most recently used.
    pub(crate) fn get(&mut self, id: u64, dest: &mut [u8]) -> bool {
        let Some(&slot) = self.map.get(&id) else {
            return false;
        };
        if self.head != slot {
            self.unlink(slot);
            self.push_front(slot);
        }
        let (slab, range) = self.slot_range(slot);
        dest.copy_from_slice(&self.slabs[slab][range]);
        true
    }

    /// One fresh slot, growing a step if the policy agrees and the budget
    /// allows, else `None` so the caller evicts.
    fn fresh_slot(&mut self) -> Option<u32> {
        let used = self.keys.len() as u32;
        if used >= self.capacity_rows {
            return None;
        }
        if used as usize / self.slab_rows as usize >= self.slabs.len() {
            // The last step stops at the capacity instead of overshooting it.
            let step_rows = self.slab_rows.min(self.capacity_rows - used) as usize;
            let step = step_rows as u64 * self.stride as u64;
            if !self.growth.can_grow(step) {
                return None;
            }
            self.slabs.push(vec![0u8; step as usize].into_boxed_slice());
            self.keys.reserve_exact(step_rows);
            self.prev.reserve_exact(step_rows);
            self.next.reserve_exact(step_rows);
        }
        self.keys.push(0);
        self.prev.push(NONE);
        self.next.push(NONE);
        Some(used)
    }

    /// Insert `record` for `id`; a cache that cannot hold it drops it.
    pub(crate) fn insert(&mut self, id: u64, record: &[u8]) {
        if self.capacity_rows == 0 || self.map.contains_key(&id) {
            return;
        }
        let slot = match self.fresh_slot() {
            Some(slot) => slot,
            None if self.tail != NONE => {
                let slot = self.tail;
                self.unlink(slot);
                self.map.remove(&self.keys[slot as usize]);
                slot
            }
            None => return,
        };
        self.keys[slot as usize] = id;
        self.map.insert(id, slot);
        self.push_front(slot);
        let (slab, range) = self.slot_range(slot);
        self.slabs[slab][range].copy_from_slice(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(value: u8) -> [u8; 4] {
        [value; 4]
    }

    fn cache(budget_rows: u64) -> RowCache {
        RowCache::new(
            budget_rows * (4 + ENTRY_OVERHEAD_BYTES),
            4,
            Arc::new(AlwaysGrow),
        )
    }

    #[test]
    fn evicts_the_least_recently_used_row() {
        let mut cache = cache(2);
        let mut out = [0u8; 4];
        cache.insert(1, &record(1));
        cache.insert(2, &record(2));
        assert!(cache.get(1, &mut out));
        assert_eq!(out, record(1));
        cache.insert(3, &record(3));
        assert!(!cache.get(2, &mut out));
        assert!(cache.get(1, &mut out));
        assert!(cache.get(3, &mut out));
        assert_eq!(out, record(3));
    }

    #[test]
    fn a_zero_budget_stores_nothing() {
        let mut cache = cache(0);
        cache.insert(1, &record(1));
        assert_eq!(cache.len(), 0);
        assert!(!cache.get(1, &mut [0u8; 4]));
    }

    struct Refuse;
    impl RowCacheGrowth for Refuse {
        fn can_grow(&self, _next_step_bytes: u64) -> bool {
            false
        }
    }

    #[test]
    fn a_refusing_growth_policy_allocates_nothing() {
        let mut cache = RowCache::new(1024, 4, Arc::new(Refuse));
        cache.insert(1, &record(1));
        assert_eq!(cache.allocated_bytes(), 0);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn budget_counts_the_entry_overhead() {
        assert_eq!(
            RowCache::new(60, 4, Arc::new(AlwaysGrow)).capacity_rows(),
            1
        );
        assert_eq!(
            RowCache::new(59, 4, Arc::new(AlwaysGrow)).capacity_rows(),
            0
        );
    }

    #[test]
    fn slabs_stop_at_the_capacity_between_growth_steps() {
        // A capacity of one and a half growth steps.
        let stride = 1 << 20;
        let rows = GROWTH_STEP_BYTES / stride as u64 * 3 / 2;
        let budget = rows * (stride as u64 + ENTRY_OVERHEAD_BYTES);
        let mut cache = RowCache::new(budget, stride, Arc::new(AlwaysGrow));
        let record = vec![7u8; stride];
        for id in 0..rows + 5 {
            cache.insert(id, &record);
        }
        assert_eq!(cache.len() as u64, rows);
        assert_eq!(cache.allocated_bytes(), rows * stride as u64);
        let mut out = vec![0u8; stride];
        assert!(cache.get(rows + 4, &mut out));
        assert_eq!(out, record);
    }
}
