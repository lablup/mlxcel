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
//! Fixed-input tests of the availability arithmetic; none needs a Mac.

use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const PAGE: u64 = 16 * 1024;

fn status(available_gib: u64) -> HostMemoryStatus {
    HostMemoryStatus {
        total_bytes: 128 * GIB,
        available_now_bytes: available_gib * GIB,
        reserve_bytes: memory_reserve_bytes(128 * GIB),
        wired_bytes: 0,
    }
}

#[test]
fn reserve_is_seven_gib_until_fifteen_percent_exceeds_it() {
    assert_eq!(memory_reserve_bytes(0), 7 * GIB);
    assert_eq!(memory_reserve_bytes(16 * GIB), 7 * GIB);
    assert_eq!(memory_reserve_bytes(32 * GIB), 7 * GIB);
    // 15% of 64 GiB is 9.6 GiB.
    assert_eq!(memory_reserve_bytes(64 * GIB), 64 * GIB * 15 / 100);
    assert_eq!(memory_reserve_bytes(128 * GIB), 128 * GIB * 15 / 100);
}

#[test]
fn reserve_arithmetic_does_not_overflow() {
    assert_eq!(
        memory_reserve_bytes(u64::MAX),
        (u128::from(u64::MAX) * 15 / 100) as u64
    );
}

#[test]
fn inactive_anonymous_pages_are_not_available() {
    // 100 inactive pages but only 30 are file-backed: the other 70 belong to
    // programs macOS would have to compress.
    let counts = VmPageCounts {
        free: 10,
        purgeable: 5,
        inactive: 90,
        speculative: 10,
        file_backed: 30,
        wired: 1,
    };
    assert_eq!(available_now_pages(&counts), 10 + 5 + 30);
}

#[test]
fn file_cache_is_counted_when_the_inactive_list_is_smaller() {
    let counts = VmPageCounts {
        free: 10,
        purgeable: 0,
        inactive: 20,
        speculative: 5,
        file_backed: 1000,
        wired: 0,
    };
    assert_eq!(available_now_pages(&counts), 10 + 25);
}

#[test]
fn status_scales_pages_by_the_page_size() {
    let counts = VmPageCounts {
        free: 1000,
        purgeable: 24,
        inactive: 0,
        speculative: 0,
        file_backed: 0,
        wired: 64,
    };
    let status = HostMemoryStatus::from_counts(64 * GIB, PAGE, &counts);
    assert_eq!(status.available_now_bytes, 1024 * PAGE);
    assert_eq!(status.wired_bytes, 64 * PAGE);
    assert_eq!(status.reserve_bytes, memory_reserve_bytes(64 * GIB));
}

#[test]
fn shortfall_is_measured_against_available_minus_reserve() {
    // 80 GiB available, 19.2 GiB reserve: 60.8 GiB usable.
    let status = status(80);
    let usable = 80 * GIB - memory_reserve_bytes(128 * GIB);
    assert_eq!(status.usable_bytes(), usable);
    assert_eq!(status.shortfall_bytes(usable), 0);
    assert_eq!(status.shortfall_bytes(usable + 1), 1);
    assert_eq!(status.shortfall_bytes(0), 0);
}

#[test]
fn availability_below_the_reserve_leaves_nothing_usable() {
    let status = status(5);
    assert_eq!(status.usable_bytes(), 0);
    assert_eq!(status.shortfall_bytes(GIB), GIB);
}

#[test]
fn wired_ceiling_leaves_the_reserve() {
    assert_eq!(
        status(0).wired_ceiling_bytes(),
        128 * GIB - memory_reserve_bytes(128 * GIB)
    );
}

fn counting_gate(refresh: Duration, available_gib: u64) -> (AvailabilityGate, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let gate = AvailabilityGate::with_probe(refresh, move || {
        seen.fetch_add(1, Ordering::Relaxed);
        Some(status(available_gib))
    });
    (gate, calls)
}

#[test]
fn gate_refuses_a_step_that_would_cross_the_reserve() {
    let (gate, _) = counting_gate(Duration::ZERO, 20);
    let usable = status(20).usable_bytes();
    assert!(gate.allows_step_at(usable, Instant::now()));
    assert!(!gate.allows_step_at(usable + 1, Instant::now()));
}

#[test]
fn gate_reuses_one_query_within_the_refresh_interval() {
    let (gate, calls) = counting_gate(Duration::from_secs(60), 20);
    let now = Instant::now();
    for _ in 0..1000 {
        assert!(!gate.allows_step_at(1 << 50, now));
    }
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(!gate.allows_step_at(1 << 50, now + Duration::from_secs(61)));
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn gate_counts_steps_approved_since_its_last_query() {
    // A stale reading cannot see earlier steps, so they are charged here.
    let (gate, _) = counting_gate(Duration::from_secs(60), 20);
    let usable = status(20).usable_bytes();
    let now = Instant::now();
    assert!(gate.allows_step_at(usable / 2, now));
    assert!(gate.allows_step_at(usable / 2, now));
    assert!(!gate.allows_step_at(usable / 2, now));
}

#[test]
fn a_refused_step_is_not_charged() {
    let (gate, _) = counting_gate(Duration::from_secs(60), 20);
    let usable = status(20).usable_bytes();
    let now = Instant::now();
    assert!(!gate.allows_step_at(usable + 1, now));
    assert!(gate.allows_step_at(usable, now));
}

#[test]
fn gate_approves_everything_when_availability_is_unknown() {
    let gate = AvailabilityGate::with_probe(Duration::ZERO, || None);
    assert!(gate.allows_step_at(u64::MAX, Instant::now()));
}

#[test]
fn speculative_pages_in_the_kernel_free_count_are_counted_once() {
    let counts = VmPageCounts {
        free: free_pages_excluding_speculative(1000, 300),
        inactive: 200,
        speculative: 300,
        file_backed: 10_000,
        ..VmPageCounts::default()
    };
    assert_eq!(counts.free, 700);
    assert_eq!(available_now_pages(&counts), 1200);
    assert_eq!(free_pages_excluding_speculative(100, 300), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn macos_queries_return_plausible_figures() {
    let status = host_memory_status().expect("host_statistics64 on macOS");
    assert!(status.total_bytes > 0);
    assert!(status.available_now_bytes <= status.total_bytes);
    assert!(status.wired_bytes > 0 && status.wired_bytes <= status.total_bytes);
    let footprint = process_footprint().expect("task_info on macOS");
    assert!(footprint.phys_footprint_bytes > 0);
    assert!(footprint.phys_footprint_bytes < status.total_bytes);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn availability_is_reported_as_unknown_off_macos() {
    assert_eq!(host_memory_status(), None);
    assert_eq!(process_footprint(), None);
}
