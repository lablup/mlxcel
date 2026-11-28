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
use super::*;
use std::cell::RefCell;

const PAGE: usize = 16 * 1024;

#[test]
fn only_explicit_true_values_turn_wiring_on() {
    for on in ["1", "true", "ON", " yes "] {
        assert!(parse_wire_weights(Some(on)), "{on}");
    }
    for off in ["0", "", "false", "off", "2"] {
        assert!(!parse_wire_weights(Some(off)), "{off}");
    }
    assert!(!parse_wire_weights(None));
}

#[test]
fn ranges_round_out_to_pages_and_merge() {
    let merged = merge_page_ranges(
        &[
            (PAGE * 10 + 5, 10),     // inside page 10
            (PAGE * 10 + 100, PAGE), // pages 10 and 11
            (PAGE * 20, PAGE * 2),   // separate
            (PAGE * 22, PAGE),       // touches the previous range
            (0, PAGE),               // an unavailable array
            (PAGE * 40, 0),          // an empty array
        ],
        PAGE,
    );
    assert_eq!(merged, vec![(PAGE * 10, PAGE * 2), (PAGE * 20, PAGE * 3)]);
}

/// A fake system whose wired total rises by what is locked.
struct FakeKernel {
    wired: RefCell<u64>,
    locks: RefCell<Vec<(usize, usize)>>,
    unlocks: RefCell<Vec<(usize, usize)>>,
}

impl FakeKernel {
    fn new(wired: u64) -> Self {
        Self {
            wired: RefCell::new(wired),
            locks: RefCell::default(),
            unlocks: RefCell::default(),
        }
    }

    fn run(&self, ranges: &[(usize, usize)], ceiling: u64, chunk: usize) -> WireReport {
        wire_ranges(
            ranges,
            ceiling,
            chunk,
            || Some(*self.wired.borrow()),
            |address, len| {
                self.locks.borrow_mut().push((address, len));
                *self.wired.borrow_mut() += len as u64;
                Ok(())
            },
            |address, len| {
                self.unlocks.borrow_mut().push((address, len));
                *self.wired.borrow_mut() -= len as u64;
            },
        )
    }
}

#[test]
fn everything_is_wired_below_the_ceiling() {
    let kernel = FakeKernel::new(0);
    let report = kernel.run(&[(PAGE, PAGE * 4)], 1 << 40, PAGE * 2);
    assert_eq!(report.requested_bytes, (PAGE * 4) as u64);
    assert_eq!(report.wired_bytes, (PAGE * 4) as u64);
    assert!(!report.stopped_at_ceiling);
    assert_eq!(kernel.locks.borrow().len(), 2);
    assert!(kernel.unlocks.borrow().is_empty());
}

#[test]
fn a_chunk_that_crosses_the_ceiling_is_unwired_and_the_pass_stops() {
    let kernel = FakeKernel::new(0);
    // Room for one chunk and a bit; the second chunk crosses.
    let ceiling = (PAGE * 3) as u64;
    let report = kernel.run(&[(PAGE, PAGE * 8)], ceiling, PAGE * 2);
    assert_eq!(report.wired_bytes, (PAGE * 2) as u64);
    assert!(report.stopped_at_ceiling);
    assert_eq!(*kernel.unlocks.borrow(), vec![(PAGE * 3, PAGE * 2)]);
    assert!(*kernel.wired.borrow() <= ceiling);
}

#[test]
fn nothing_is_wired_when_the_system_is_already_at_the_ceiling() {
    let kernel = FakeKernel::new(1000);
    let report = kernel.run(&[(PAGE, PAGE)], 1000, PAGE);
    assert_eq!(report.wired_bytes, 0);
    assert!(report.stopped_at_ceiling);
    assert!(kernel.locks.borrow().is_empty());
}

#[test]
fn an_unreadable_wired_total_stops_the_pass() {
    let report = wire_ranges(
        &[(PAGE, PAGE)],
        1 << 40,
        PAGE,
        || None,
        |_, _| Ok(()),
        |_, _| {},
    );
    assert_eq!(report.wired_bytes, 0);
    assert!(report.stopped_at_ceiling);
}

#[test]
fn a_failed_lock_is_reported_and_later_ranges_still_run() {
    let report = wire_ranges(
        &[(PAGE, PAGE), (PAGE * 4, PAGE)],
        1 << 40,
        PAGE,
        || Some(0),
        |address, _| if address == PAGE { Err(12) } else { Ok(()) },
        |_, _| {},
    );
    assert_eq!(report.failed_ranges, 1);
    assert_eq!(report.first_errno, Some(12));
    assert_eq!(report.wired_bytes, PAGE as u64);
    assert!(!report.stopped_at_ceiling);
}

#[cfg(target_os = "macos")]
#[test]
fn a_real_evaluated_buffer_can_be_wired() {
    let data = vec![1.0_f32; 1 << 20];
    let array = crate::from_slice_f32(&data, &[data.len() as i32]);
    crate::eval(&array);
    let address = crate::ffi::array_data_address(&array);
    let len = crate::ffi::array_nbytes(&array);
    assert_ne!(address, 0, "an evaluated array has a host address");
    let page = 16 * 1024;
    let merged = merge_page_ranges(&[(address, len)], page);
    let (start, span) = merged[0];
    macos::mlock(start, span).expect("mlock of an evaluated MLX buffer");
    macos::munlock(start, span);
}

#[cfg(target_os = "macos")]
#[test]
fn only_a_contiguous_view_has_an_address() {
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let matrix = crate::from_slice_f32(&data, &[3, 4]);
    let columns = crate::ffi::slice(&matrix, &[0, 1], &[3, 3]);
    let row = crate::ffi::slice(&matrix, &[1, 0], &[2, 4]);
    crate::eval(&columns);
    crate::eval(&row);
    assert_eq!(crate::ffi::array_data_address(&columns), 0);
    assert_ne!(crate::ffi::array_data_address(&row), 0);
}

#[test]
fn an_unevaluated_array_has_no_address() {
    let data = vec![1.0_f32; 16];
    let lazy = crate::add(
        &crate::from_slice_f32(&data, &[16]),
        &crate::from_slice_f32(&data, &[16]),
    );
    assert_eq!(crate::ffi::array_data_address(&lazy), 0);
}
