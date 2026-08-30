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

//! Shared helpers for bounded in-memory server stores.

use std::cmp::Ordering;
use std::io;
use std::time::Instant;

use serde::Serialize;

/// Count JSON bytes without allocating a second serialized copy of the value.
///
/// The result is intentionally approximate for memory budgeting: it tracks the
/// wire-size contribution of nested strings and JSON values and uses saturating
/// arithmetic so malformed or extreme inputs cannot overflow the accounting.
pub(crate) fn serialized_json_len_saturating<T>(value: &T) -> usize
where
    T: Serialize + ?Sized,
{
    let mut writer = SaturatingByteCounter::default();
    match serde_json::to_writer(&mut writer, value) {
        Ok(()) => writer.bytes,
        Err(_) => usize::MAX,
    }
}

#[derive(Default)]
struct SaturatingByteCounter {
    bytes: usize,
}

impl io::Write for SaturatingByteCounter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.bytes = self.bytes.saturating_add(buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Exact LRU index key. Stores keep one of these per live entry and remove it
/// whenever an entry is deleted or refreshed, so stale metadata cannot grow
/// independently of the map.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct LruKey {
    pub(crate) last_accessed: Instant,
    pub(crate) sequence: u64,
    pub(crate) id: String,
}

impl Ord for LruKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.last_accessed
            .cmp(&other.last_accessed)
            .then_with(|| self.sequence.cmp(&other.sequence))
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for LruKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
