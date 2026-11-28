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

//! The NemotronH decode-handle registry must be safe to use from many threads
//! at once (#3124).
//!
//! Every `NemotronHModel` registers a C++ handle when it is built and frees it
//! in `Drop`, so parallel libtest workers (and concurrent server model loads)
//! insert into and erase from one process-wide map at the same time. An
//! unguarded map corrupted the heap under `cargo test --lib server::batch::`:
//! SIGSEGV inside `nemotron_free_model`, a malloc freelist trap in an
//! unrelated allocation, and a "pointer being freed was not allocated" abort
//! from the map's static destructor at `exit` after every test had passed.

use std::collections::HashSet;
use std::sync::{Arc, Barrier};

use super::*;

const THREADS: usize = 8;
const ROUNDS: usize = 4000;

/// Registers a model with no layers. The registry only stores the weight
/// pointers and never dereferences them outside `nemotron_decode_step`, which
/// this test does not call.
fn register_empty(stub: &MlxArray) -> u64 {
    // SAFETY: every reference points at `stub`, which outlives the handle
    // because each handle is freed before its thread drops `stub`. The empty
    // slices register a model with no layers, and no decode step runs.
    unsafe {
        nemotron_register_model(
            stub,
            stub,
            stub,
            stub,
            stub,
            stub,
            std::ptr::null(),
            &[],
            &[],
            &[],
            &[],
            &[],
            1e-5,
            64,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            false,
            0,
            0,
            0,
            0.0,
            0.0,
        )
    }
}

#[test]
fn concurrent_register_and_free_issue_distinct_handles_without_corruption() {
    let barrier = Arc::new(Barrier::new(THREADS));
    let workers: Vec<_> = (0..THREADS)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let stub = from_slice_f32(&[0.0], &[1]);
                let mut issued = Vec::with_capacity(ROUNDS);
                let mut live = Vec::with_capacity(ROUNDS / 2);
                barrier.wait();
                for round in 0..ROUNDS {
                    let handle = register_empty(&stub);
                    issued.push(handle);
                    // Free half of the handles right away so erases race the
                    // other threads' inserts and rehashes, as model drops do.
                    if round % 2 == 0 {
                        nemotron_free_model(handle);
                    } else {
                        live.push(handle);
                    }
                }
                for handle in live {
                    nemotron_free_model(handle);
                }
                issued
            })
        })
        .collect();

    let mut seen = HashSet::new();
    for worker in workers {
        let issued = worker.join().expect("registry worker panicked");
        for handle in issued {
            assert_ne!(handle, 0, "0 is the unregistered sentinel");
            assert!(seen.insert(handle), "handle {handle} was issued twice");
        }
    }
    assert_eq!(seen.len(), THREADS * ROUNDS);
}
