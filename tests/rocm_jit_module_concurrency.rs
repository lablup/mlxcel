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

//! The ROCm JIT module cache is safe to use from several threads at once
//! (issue #2183).
//!
//! `get_jit_module` in the ROCm overlay memoises compiled modules in one
//! process-global map. It used to do an unlocked `find` and then a
//! `try_emplace` whose constructor compiles through hiprtc, so two threads
//! that missed the same key both compiled and inserted, and an insert that
//! rehashed the table could run under another thread's `find`. The server
//! launches kernels from several worker threads (scheduler, embedding,
//! rerank, audio), each on its own stream, so both cases are reachable. Every
//! evaluation also takes `hipEvent_t` handles from the backend's event pool,
//! which was unlocked as well and failed these tests first.
//!
//! The probe (`mlxcel_core::rocm_faults::jit_race_probe_array`) is one custom
//! kernel per `variant`, so a variant's first launch in this process is a
//! cache miss and every later launch is a hit. Each worker thread installs
//! its own thread-local stream, as the server workers do, and waits on a
//! barrier so the launches overlap:
//!
//! - `same_kernel_first_launch_from_many_threads`: 8 threads first-launch the
//!   same new variant, 16 rounds;
//! - `distinct_kernels_first_launch_while_others_hit_cache`: 8 threads each
//!   first-launch their own new variant while another thread keeps launching
//!   a compiled one, 16 rounds.
//!
//! Every output must be exactly `2 * i + 1`. A crash, a hang or a wrong value
//! is a failure. Measured on gfx1151 over 20 runs of this file each: with the
//! lock in `get_jit_module` removed, 0 runs failed, so for the JIT cache these
//! tests are a crash guard, not a deterministic reproduction. With the
//! unlocked `HipEventPool` of `event.hip` restored (LOCAL_FIXES item 36), all
//! 20 failed (10 with `hipEventRecord ... invalid resource handle`, 8 SIGABRT,
//! 2 SIGSEGV). With both locks, 0 failed. Skips on any other backend. Run on a
//! ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_jit_module_concurrency
//! ```

#![cfg(feature = "rocm")]

use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::rocm_faults::jit_race_probe_array;
use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
};

const ROUNDS: i32 = 16;
const THREADS: i32 = 8;
const LEN: i32 = 256;

/// Variant ranges per test. Variants are process-global, and libtest may run
/// both tests at once in this binary, so the ranges must not overlap.
const SAME_KERNEL_BASE: i32 = 1_000;
const WARM_VARIANT: i32 = 2_000;
const DISTINCT_KERNEL_BASE: i32 = 3_000;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// Launch `variant` on `arange(LEN)` on the calling thread's default stream
/// and check every element is exactly `2 * i + 1`.
fn launch_and_check(variant: i32) -> Result<(), String> {
    let inp = mlxcel_core::arange_f32(0.0, LEN as f32, 1.0);
    let out = jit_race_probe_array(&inp, variant)
        .map_err(|e| format!("variant {variant}: building the probe failed: {e}"))?;
    mlxcel_core::try_eval(&out)
        .map_err(|e| format!("variant {variant}: evaluating the probe failed: {e}"))?;
    let bytes = mlxcel_core::array_to_raw_bytes(&out);
    if bytes.len() != LEN as usize * 4 {
        return Err(format!(
            "variant {variant}: expected {} output bytes, got {}",
            LEN * 4,
            bytes.len()
        ));
    }
    for (i, b) in bytes.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let want = 2.0 * i as f32 + 1.0;
        if got != want {
            return Err(format!(
                "variant {variant}: element {i} is {got}, expected {want}"
            ));
        }
    }
    Ok(())
}

/// Join every handle and fail with every thread's error, so one bad thread
/// does not hide another.
fn join_all(handles: Vec<thread::ScopedJoinHandle<'_, Result<(), String>>>, round: i32) {
    let errors: Vec<String> = handles
        .into_iter()
        .enumerate()
        .filter_map(|(t, h)| match h.join() {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(format!("thread {t}: {e}")),
            Err(_) => Some(format!("thread {t}: panicked")),
        })
        .collect();
    assert!(errors.is_empty(), "round {round}: {}", errors.join("; "));
}

#[test]
fn same_kernel_first_launch_from_many_threads() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    for round in 0..ROUNDS {
        // A variant no thread has launched yet: every thread misses the cache
        // at once, and only one module may be built for all of them.
        let variant = SAME_KERNEL_BASE + round;
        let barrier = Barrier::new(THREADS as usize);
        thread::scope(|s| {
            let handles = (0..THREADS)
                .map(|_| {
                    let barrier = &barrier;
                    s.spawn(move || {
                        let stream = new_thread_local_generation_stream();
                        install_thread_local_default_stream(stream.as_ref());
                        barrier.wait();
                        launch_and_check(variant)
                    })
                })
                .collect();
            join_all(handles, round);
        });
    }
}

#[test]
fn distinct_kernels_first_launch_while_others_hit_cache() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    // Compile the warm variant once, so the hitter only ever finds it.
    launch_and_check(WARM_VARIANT).expect("warming the cache-hit variant");

    for round in 0..ROUNDS {
        // The hitter waits on the same barrier, then keeps launching (and so
        // looking up) the warm variant until every new variant is in, so its
        // finds overlap the inserts and any rehash they cause.
        let barrier = Barrier::new(THREADS as usize + 1);
        let inserts_done = AtomicBool::new(false);
        thread::scope(|s| {
            let hitter = {
                let barrier = &barrier;
                let inserts_done = &inserts_done;
                s.spawn(move || -> Result<(), String> {
                    let stream = new_thread_local_generation_stream();
                    install_thread_local_default_stream(stream.as_ref());
                    barrier.wait();
                    loop {
                        launch_and_check(WARM_VARIANT)?;
                        if inserts_done.load(Ordering::Acquire) {
                            return Ok(());
                        }
                    }
                })
            };
            let inserters: Vec<_> = (0..THREADS)
                .map(|t| {
                    let barrier = &barrier;
                    let variant = DISTINCT_KERNEL_BASE + round * THREADS + t;
                    s.spawn(move || {
                        let stream = new_thread_local_generation_stream();
                        install_thread_local_default_stream(stream.as_ref());
                        barrier.wait();
                        launch_and_check(variant)
                    })
                })
                .collect();
            // Join the inserters before stopping the hitter, then report both.
            let inserter_results: Vec<_> = inserters.into_iter().map(|h| h.join()).collect();
            inserts_done.store(true, Ordering::Release);
            let mut errors: Vec<String> = inserter_results
                .into_iter()
                .enumerate()
                .filter_map(|(t, r)| match r {
                    Ok(Ok(())) => None,
                    Ok(Err(e)) => Some(format!("inserter {t}: {e}")),
                    Err(_) => Some(format!("inserter {t}: panicked")),
                })
                .collect();
            match hitter.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => errors.push(format!("hitter: {e}")),
                Err(_) => errors.push("hitter: panicked".to_string()),
            }
            assert!(errors.is_empty(), "round {round}: {}", errors.join("; "));
        });
    }
}
