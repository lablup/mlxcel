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

//! `MLXCEL_SAMPLING_GUMBEL=0` kill switch for the Gumbel-max sampler (#900).
//!
//! The C++ gate reads the variable once per process (the sampler runs per
//! token, so a `getenv` per call would land in the very measurement the kernel
//! exists to improve). A test that mutates the environment therefore has to own
//! the process: this file deliberately holds exactly ONE `#[test]`, which sets
//! the variable before any sampling call can resolve the static. Do not add a
//! second test here, and do not move this into the crate's unit-test binary.
//!
//! Run:
//!   cargo test --release --features metal,accelerate \
//!     --test sampling_gumbel_kill_switch

use mlxcel_core::{
    MlxArray, array_to_raw_bytes, from_slice_f32, fused_sample, fused_sample_categorical,
    gumbel_max_sample, random_seed, sampling_gumbel_available,
};

fn token_ids(tokens: &MlxArray) -> Vec<u32> {
    array_to_raw_bytes(tokens)
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn falsy_env_restores_the_categorical_sampling_path() {
    // SAFETY: `set_var` mutates the process-global environment and is unsound
    // when another thread reads it concurrently. This is the first statement of
    // the only test in this binary, so no other thread exists yet and nothing
    // has resolved the C++ gate's `static` yet.
    unsafe { std::env::set_var("MLXCEL_SAMPLING_GUMBEL", "0") };

    assert!(
        !sampling_gumbel_available(),
        "MLXCEL_SAMPLING_GUMBEL=0 did not disable the Gumbel-max sampler"
    );

    // A GPU is still needed to compare the two sampling arms; on a CPU-only
    // build the assertion above is the whole test.
    if !mlxcel_core::is_gpu_available() {
        return;
    }

    let vocab = 1024usize;
    let rows = 32usize;
    let logits: Vec<f32> = (0..vocab)
        .map(|i| {
            let x = (i as f32 - vocab as f32 / 2.0) / 64.0;
            3.0 - x * x
        })
        .collect();
    let mut tiled = Vec::with_capacity(rows * vocab);
    for _ in 0..rows {
        tiled.extend_from_slice(&logits);
    }
    let batched = from_slice_f32(&tiled, &[rows as i32, vocab as i32]);

    // With the kill switch on, the no-filter path must be the pre-#900
    // `random::categorical` graph, bit-for-bit.
    random_seed(0x0900_00FF);
    let through_fused = token_ids(&fused_sample(&batched, 1.0, 0, 1.0, 0.0));
    random_seed(0x0900_00FF);
    let categorical = token_ids(&fused_sample_categorical(&batched, 1.0, 0, 1.0, 0.0));
    assert_eq!(
        through_fused, categorical,
        "kill switch set, but fused_sample did not reproduce the categorical path"
    );

    // ... and it must NOT be the kernel, which the same seed would reproduce
    // exactly if the gate were inert.
    random_seed(0x0900_00FF);
    let gumbel = token_ids(&gumbel_max_sample(&batched, 1.0));
    assert_ne!(
        through_fused, gumbel,
        "kill switch set, but fused_sample still produced the Gumbel-max stream"
    );

    // The kernel entry point itself stays callable: the switch gates routing,
    // not the kernel, so an explicit caller and the benchmark still work.
    assert_eq!(gumbel.len(), rows);
    assert!(gumbel.iter().all(|&id| (id as usize) < vocab));
}
