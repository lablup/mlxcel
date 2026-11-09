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

//! Lookup counters of the external n-gram table and their stats line.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::ple_io::PleIoMode;

/// Counters behind the stats line, in the spirit of upstream's `LookupStats`.
#[derive(Debug, Default)]
pub struct PleStats {
    pub(super) lookups: AtomicU64,
    pub(super) rows: AtomicU64,
    pub(super) hits: AtomicU64,
    pub(super) misses: AtomicU64,
    pub(super) bytes_read: AtomicU64,
    pub(super) elapsed_ns: AtomicU64,
    pub(super) prefetched_rows: AtomicU64,
}

/// A point-in-time copy of [`PleStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PleStatsSnapshot {
    pub lookups: u64,
    pub rows: u64,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub elapsed_ns: u64,
    pub prefetched_rows: u64,
}

impl PleStats {
    pub fn snapshot(&self) -> PleStatsSnapshot {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        PleStatsSnapshot {
            lookups: load(&self.lookups),
            rows: load(&self.rows),
            hits: load(&self.hits),
            misses: load(&self.misses),
            bytes_read: load(&self.bytes_read),
            elapsed_ns: load(&self.elapsed_ns),
            prefetched_rows: load(&self.prefetched_rows),
        }
    }
}

/// The most recently opened table's counters, for the stats line. A process
/// serves one such table at a time; the slot keeps the printer model-agnostic.
pub(super) static LAST_TABLE: Mutex<Option<(PleIoMode, Arc<PleStats>)>> = Mutex::new(None);

/// One stats line for the external table used in this process, or `None` when
/// no table has been opened.
pub fn ple_stats_line() -> Option<String> {
    let guard = LAST_TABLE.lock().ok()?;
    let (mode, stats) = guard.as_ref()?;
    let s = stats.snapshot();
    let hit_rate = if s.rows == 0 {
        0.0
    } else {
        s.hits as f64 * 100.0 / s.rows as f64
    };
    Some(format!(
        "[PLE table ({}): {} lookups, {} rows, {:.1}% hit, {} misses, {:.1} MiB read, {:.1} ms, {} rows gathered ahead]",
        mode.label(),
        s.lookups,
        s.rows,
        hit_rate,
        s.misses,
        s.bytes_read as f64 / (1024.0 * 1024.0),
        s.elapsed_ns as f64 / 1e6,
        s.prefetched_rows,
    ))
}
