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

//! `materialized_state_tail` must not keep its full-length source alive once
//! the layer's own consumers have run. These cases read process-wide MLX
//! memory, so they live in their own test binary: in the lib suite, tests on
//! other threads allocate while they measure.

use mlxcel_core::utils::{STATE_TAIL_EAGER_MIN_ROWS, materialized_state_tail};
use std::sync::Mutex;

/// The cases share the process's MLX memory counters; run them one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

const ROWS: i32 = 4099;
const WIDTH: i32 = 16384;
const SOURCE_BYTES: u64 = ROWS as u64 * WIDTH as u64 * 2;
// The release case must take the eager path.
const _: () = assert!(ROWS >= STATE_TAIL_EAGER_MIN_ROWS);

/// Builds a fresh `[1, ROWS, WIDTH]` f16 source, stores a 3-row tail with
/// `store`, evaluates only a consumer of the source (as a layer evaluates its
/// output), drops every other handle, and returns the bytes still active.
fn retained_after_consumer(
    store: impl Fn(&mlxcel_core::MlxArray) -> cxx::UniquePtr<mlxcel_core::MlxArray>,
) -> u64 {
    mlxcel_core::clear_memory_cache();
    mlxcel_core::synchronize_default();
    let before = mlxcel_core::memory::active_memory();
    let ones = mlxcel_core::ones(&[1, ROWS, WIDTH], mlxcel_core::dtype::FLOAT16);
    let two = mlxcel_core::full_f32(&[1], 2.0, mlxcel_core::dtype::FLOAT16);
    let source = mlxcel_core::multiply(&ones, &two);
    let tail = store(&source);
    let consumer = mlxcel_core::sum_all(&source);
    drop(source);
    drop(ones);
    mlxcel_core::eval(&consumer);
    drop(consumer);
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();
    let retained = mlxcel_core::memory::active_memory().saturating_sub(before);
    mlxcel_core::eval(&tail);
    assert_eq!(mlxcel_core::array_shape(&tail), vec![1, 3, WIDTH]);
    retained
}

#[test]
fn long_state_tail_releases_its_source_after_the_consumer_runs() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let retained = retained_after_consumer(|src| materialized_state_tail(src, 3));
    assert!(
        retained < SOURCE_BYTES / 8,
        "the tail kept {retained} bytes alive; the source is {SOURCE_BYTES}"
    );
}

/// The control: a lazy `contiguous(slice)` tail, the form the models used,
/// does keep the source. Without it the guard above could pass vacuously.
#[test]
fn lazy_contiguous_tail_keeps_its_source() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let retained = retained_after_consumer(|src| {
        let starts = [0, ROWS - 3, 0];
        mlxcel_core::contiguous(&mlxcel_core::slice(src, &starts, &[1, ROWS, WIDTH]), false)
    });
    assert!(
        retained >= SOURCE_BYTES / 2,
        "the lazy tail kept only {retained} bytes; the source is {SOURCE_BYTES}"
    );
}
