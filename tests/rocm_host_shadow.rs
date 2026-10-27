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

//! The Windows ROCm host shadow flushes CPU writes and only CPU writes
//! (issue #2643, `overlays/rocm/LOCAL_FIXES.md` item 67).
//!
//! On Windows every host access to a GPU buffer goes through a pinned shadow
//! copy. Before #2643 `Buffer::raw_ptr()` marked the shadow dirty on every
//! access, so a plain host read of a token or logit buffer cost a full
//! host-to-device copy before the next kernel. Now the bridge's read paths
//! (`item_*`, the byte readers) take a read-only accessor that leaves the
//! buffer clean, while `array::data<T>()`, the path every CPU writer in MLX
//! takes, still marks it dirty. This test drives both through the allocator's
//! shadow counters:
//!
//! - a host read followed by a kernel launch adds no upload;
//! - a host write followed by a kernel launch adds exactly one upload and the
//!   kernel sees the written value;
//! - a read, then a write, then a kernel launch flushes;
//! - a recycled buffer that still carries its previous owner's shadow neither
//!   flushes it nor serves it to a reader;
//! - every shadow this test created is released with its buffer once the
//!   cache is cleared, so the byte counter returns to where it started.
//!
//! Skips on any other backend and on Linux ROCm, where the CPU dereferences
//! device memory and no shadow exists. Run on a Windows ROCm host with:
//!
//! ```sh
//! cargo test --release --features rocm --test rocm_host_shadow -- --test-threads=1
//! ```

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::rocm_host_shadow::{
    HostShadowStats, host_read_f32_for_test as read_f32, host_shadow_stats,
    host_write_f32_for_test as write_f32,
};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr};

/// 1031 floats: 4124 bytes, which the allocator rounds to 8192. Below the
/// 16 KiB page so it never shares a size with the recycle probe below.
const N: usize = 1031;
/// 5000 floats: 20000 bytes, rounded to two 16 KiB pages. No other array in
/// this test has that size, so the one buffer of it that is dropped is the one
/// the cache hands out next.
const N_RECYCLE: usize = 5000;

fn on_windows_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm && cfg!(windows)
}

fn settle(label: &str, a: &MlxArray) {
    if let Err(err) = mlxcel_core::try_eval(a) {
        panic!("{label}: evaluation failed: {err}");
    }
    mlxcel_core::synchronize_default();
}

fn stats() -> HostShadowStats {
    mlxcel_core::synchronize_default();
    host_shadow_stats()
}

fn read(label: &str, a: &MlxArray, index: usize) -> f32 {
    match read_f32(a, index) {
        Ok(v) => v,
        Err(err) => panic!("{label}: host read failed: {err}"),
    }
}

fn write(label: &str, a: &MlxArray, index: usize, value: f32) {
    if let Err(err) = write_f32(a, index, value) {
        panic!("{label}: host write failed: {err}");
    }
}

/// A device-resident array of `n` copies of `value`: the output of a kernel,
/// so its buffer has no shadow until the host touches it.
fn device_filled(n: usize, value: f32, one: &MlxArray) -> UniquePtr<MlxArray> {
    let host = mlxcel_core::from_slice_f32(&vec![value - 1.0; n], &[n as i32]);
    let out = mlxcel_core::add(&host, one);
    settle("device_filled", &out);
    out
}

#[test]
fn host_reads_do_not_flush_and_host_writes_do() {
    if !on_windows_rocm() {
        eprintln!("skipping: host shadows exist only on Windows ROCm");
        return;
    }
    let _lock = lock_default_device();
    let _gpu = DefaultDeviceGuard::gpu();

    mlxcel_core::clear_memory_cache();
    let start = stats();

    // `one` is host-built, so its first kernel use flushes it once; after that
    // it is clean and every later `add` reads it without a copy.
    let one = mlxcel_core::from_slice_f32(&[1.0], &[1]);
    let a = device_filled(N, 2.0, &one);
    let base = stats();

    // (i) A host read, then a kernel reading the same buffer: no upload.
    assert_eq!(read("read a[0]", &a, 0), 2.0);
    let after_read = stats();
    assert_eq!(
        after_read.downloads,
        base.downloads + 1,
        "the first host read of a device-resident buffer copies it down once"
    );
    let shadow_bytes = after_read.bytes - base.bytes;
    assert!(
        (N * 4..=16 * 1024).contains(&(shadow_bytes as usize)),
        "the shadow is counted at the buffer's allocator size, got {shadow_bytes} bytes"
    );
    let b = mlxcel_core::add(&a, &one);
    settle("b = a + 1", &b);
    let after_kernel = stats();
    assert_eq!(
        after_kernel.uploads, after_read.uploads,
        "a kernel reading a buffer the host only read must not flush its shadow"
    );
    assert_eq!(read("read b[0]", &b, 0), 3.0);

    // (ii) A host write, then a kernel reading the buffer: one upload, and the
    // kernel computes from the written value.
    let before_write = stats();
    write("write a[0] = 7", &a, 0, 7.0);
    let c = mlxcel_core::add(&a, &one);
    settle("c = a + 1", &c);
    let after_write_kernel = stats();
    assert_eq!(
        after_write_kernel.uploads,
        before_write.uploads + 1,
        "a kernel reading a buffer the host wrote flushes its shadow once"
    );
    assert_eq!(
        read("read c[0]", &c, 0),
        8.0,
        "the flushed write reached the device"
    );

    // (iii) Read, then write, then kernel: the read must not hide the write.
    assert_eq!(read("read a[1]", &a, 1), 2.0);
    let before_rw = stats();
    write("write a[1] = 9", &a, 1, 9.0);
    let d = mlxcel_core::add(&a, &one);
    settle("d = a + 1", &d);
    let after_rw = stats();
    assert_eq!(
        after_rw.uploads,
        before_rw.uploads + 1,
        "a write after a read still flushes before the next kernel"
    );
    assert_eq!(read("read d[1]", &d, 1), 10.0);
    assert_eq!(
        read("read d[0]", &d, 0),
        8.0,
        "the earlier write is still on the device"
    );

    // (iv) A recycled buffer keeps its previous owner's shadow. The new owner's
    // kernel must not flush that stale shadow over its output, and a reader of
    // the new owner must not be served the stale bytes.
    let x = mlxcel_core::from_slice_f32(&vec![5.0; N_RECYCLE], &[N_RECYCLE as i32]);
    let y = mlxcel_core::add(&x, &one);
    settle("y = x + 1", &y);
    // Refresh x's shadow after the flush so it is a clean, valid copy of 5.0s
    // when the buffer goes back to the cache.
    assert_eq!(read("read x[0]", &x, 0), 5.0);
    drop(x);
    let before_recycle = stats();
    let z = mlxcel_core::add(&y, &one);
    settle("z = y + 1 (recycled buffer)", &z);
    let after_recycle = stats();
    assert_eq!(
        after_recycle.uploads, before_recycle.uploads,
        "the recycled buffer's stale shadow must not be flushed over the kernel's output"
    );
    assert_eq!(
        read("read z[0]", &z, 0),
        7.0,
        "a reader of the recycled buffer must get the device bytes, not the stale shadow"
    );

    // (v) Every shadow goes with its buffer. Drop the arrays, clear the cache
    // (recycled buffers keep their shadows until then) and the counter is back
    // where it started.
    drop(a);
    drop(b);
    drop(c);
    drop(d);
    drop(y);
    drop(z);
    drop(one);
    mlxcel_core::clear_memory_cache();
    let end = stats();
    assert_eq!(
        end.bytes, start.bytes,
        "shadow bytes must return to the starting value once every buffer is released"
    );
    assert!(end.downloads > start.downloads && end.uploads > start.uploads);
}

/// 2500 floats: 10000 bytes, a size no other array in this file has, so after
/// the cache is cleared the only buffer of it is the one under test.
const N_DONATE: usize = 2500;

#[test]
fn a_kernel_writing_a_read_buffer_in_place_invalidates_its_shadow() {
    if !on_windows_rocm() {
        eprintln!("skipping: host shadows exist only on Windows ROCm");
        return;
    }
    let _lock = lock_default_device();
    let _gpu = DefaultDeviceGuard::gpu();

    let one = mlxcel_core::from_slice_f32(&[1.0], &[1]);
    let a = device_filled(N_DONATE, 2.0, &one);
    // Free the host-built input `device_filled` dropped, and its shadow, so a
    // buffer of this size can only come from `a` itself below.
    mlxcel_core::clear_memory_cache();

    // The read leaves `a`'s shadow clean and valid: it mirrors the 2.0s.
    assert_eq!(read("read a[0]", &a, 0), 2.0);
    let before = stats();

    // Nothing else holds `a`, so the add donates its buffer and the kernel
    // overwrites the bytes the valid shadow mirrors, in place.
    let a2 = mlxcel_core::add(&a, &one);
    drop(a);
    settle("a2 = a + 1 (donated)", &a2);
    let after_kernel = stats();
    assert_eq!(
        after_kernel.uploads, before.uploads,
        "a clean shadow is not flushed before the kernel"
    );

    assert_eq!(
        read("read a2[0]", &a2, 0),
        3.0,
        "a reader after an in-place kernel write must get the device bytes, not the shadow"
    );
    let after_read = stats();
    assert_eq!(
        after_read.bytes, after_kernel.bytes,
        "the output reused the donated buffer and its shadow (no new shadow was allocated)"
    );
    assert_eq!(
        after_read.downloads,
        after_kernel.downloads + 1,
        "the kernel invalidated the shadow, so the read copied it down again"
    );
    assert_eq!(read("read a2[last]", &a2, N_DONATE - 1), 3.0);
    let again = stats();
    assert_eq!(
        again.downloads, after_read.downloads,
        "a second read with no kernel in between copies nothing"
    );

    drop(a2);
    drop(one);
    mlxcel_core::clear_memory_cache();
}

#[test]
fn empty_arrays_get_no_shadow_and_read_back_empty() {
    if !on_windows_rocm() {
        eprintln!("skipping: host shadows exist only on Windows ROCm");
        return;
    }
    let _lock = lock_default_device();
    let _gpu = DefaultDeviceGuard::gpu();

    let before = stats();
    // Building an empty array from host data writes its zero-byte buffer
    // through `data<T>()`, and the byte reader takes the read-only path. A
    // zero-byte buffer used to reach hipHostMalloc(0), which returns null, and
    // both calls threw (the constructor across a non-fallible bridge call,
    // aborting the process).
    let empty = mlxcel_core::from_slice_f32(&[], &[0]);
    match mlxcel_core::try_array_to_raw_bytes(&empty) {
        Ok(bytes) => assert!(bytes.is_empty(), "an empty array has no bytes"),
        Err(err) => panic!("reading an empty array failed: {err}"),
    }
    let after = stats();
    assert_eq!(
        after.bytes, before.bytes,
        "a zero-byte buffer gets no shadow"
    );
    drop(empty);
}
