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

//! The token read of a pipelined classic step must not wait for the next
//! step, which is already queued behind it on the same stream (issue #3430).

use std::time::{Duration, Instant};

use super::materialize_argmax_i32_vec;
use crate::{MlxArray, UniquePtr, dtype, ffi};

/// About 0.1 s of matmuls on an M3 Ultra, queued after the token.
fn queue_long_work() -> UniquePtr<MlxArray> {
    let w = crate::multiply_scalar(&ffi::ones(&[2048, 2048], dtype::FLOAT32), 1.0 / 2048.0);
    let mut x = ffi::ones(&[2048, 2048], dtype::FLOAT32);
    for _ in 0..96 {
        x = ffi::matmul(&x, &w);
    }
    ffi::async_eval(&x);
    x
}

#[test]
fn reading_a_queued_token_does_not_wait_for_later_work() {
    if !ffi::metal_is_available() {
        return;
    }
    // Warm up the kernels so the timed rounds measure the queue only.
    let warm = queue_long_work();
    ffi::eval(&warm);

    let logits = ffi::from_slice_f32(&[0.1, 0.9, 0.3, 0.2], &[1, 1, 4]);
    let token = ffi::astype(&ffi::argmax_last_axis(&logits), dtype::INT32);
    ffi::async_eval(&token);
    let later = queue_long_work();

    let started = Instant::now();
    let value = materialize_argmax_i32_vec(&token, 1);
    let read = started.elapsed();
    let started = Instant::now();
    ffi::eval(&later);
    let rest = started.elapsed();

    assert_eq!(value, vec![1]);
    // Waiting for the queued work would put all of it in `read` and leave
    // `rest` near zero; the read must leave most of it still running.
    assert!(
        rest > read && rest > Duration::from_millis(5),
        "the token read waited for the work queued after it: read {read:?}, rest {rest:?}"
    );
}
