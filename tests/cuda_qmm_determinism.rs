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

//! Regression test for issue #910: CUDA quantized-matmul (qmm_sm80) shared
//! memory epilogue race.
//!
//! The qmm_sm80 kernel stages its output tile into shared memory that
//! aliases the A/B cp.async pipeline slots (SharedStorage is a union).
//! Before the fix the epilogue did not retire in-flight cp.async groups
//! before overwriting the union, so a late-landing async copy corrupted the
//! staged C tile: every M >= 8 quantized matmul (prefill, MoE gather_qmm)
//! was bitwise non-deterministic at temp 0 on every 4-bit CUDA model, with
//! occasional garbage tokens. Pre-fix this test failed on 23/23 repeat
//! iterations with an 8-token prefill on GB10 (sm_121); post-fix it passes
//! and compute-sanitizer racecheck reports 0 hazards.
//!
//! Runs only with the `cuda` feature and skips when the model checkpoint is
//! absent. Run on a CUDA host with:
//!
//! ```sh
//! MLX_CUDA_ARCHITECTURES=<arch> cargo test --release --features cuda \
//!     --test cuda_qmm_determinism
//! ```
//!
//! `MLXCEL_TEST_QMM_MODEL_DIR` overrides the default 4-bit model directory.

#![cfg(feature = "cuda")]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use mlxcel::generate::LanguageModel;
use mlxcel_core::layers::KVCache;

const DEFAULT_MODEL_DIR: &str = "models/llama-3.2-1b-4bit";

/// Byte-level hash of an array's contents after forcing evaluation.
fn hash_array(arr: &mlxcel_core::UniquePtr<mlxcel_core::MlxArray>) -> u64 {
    let f = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    let bytes = mlxcel_core::array_to_raw_bytes(&f);
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Repeats an identical prefill + fixed-token decode sequence with fresh KV
/// caches and asserts that the raw logits of every step are byte-identical
/// across iterations. The prefill length must be >= 8 so the quantized
/// matmuls take the qmm_sm80 tile path (M * B < 8 dispatches to qmv, which
/// was never affected).
#[test]
fn temp0_quantized_forward_is_bitwise_deterministic() {
    let model_dir = std::env::var("MLXCEL_TEST_QMM_MODEL_DIR")
        .unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    if !Path::new(&model_dir).exists() {
        eprintln!("skipping cuda_qmm_determinism: {model_dir} not present");
        return;
    }

    const ITERS: usize = 10;
    const PREFILL_LEN: usize = 64;
    const DECODE_STEPS: usize = 8;

    let (model, _) = mlxcel::load_model(Path::new(&model_dir)).expect("load model");

    // Fixed, sampling-free input schedule so every iteration performs the
    // identical computation regardless of what the logits contain.
    let prompt: Vec<i32> = (0..PREFILL_LEN)
        .map(|i| 100 + (i as i32 * 37) % 900)
        .collect();

    let mut reference: Option<Vec<u64>> = None;
    for iter in 0..ITERS {
        let mut caches: Vec<KVCache> = model.make_caches();
        let mut step_hashes = Vec::with_capacity(1 + DECODE_STEPS);

        let input = mlxcel_core::from_slice_i32(&prompt, &[1, PREFILL_LEN as i32]);
        let logits = model.forward(&input, &mut caches, None);
        step_hashes.push(hash_array(&logits));

        for s in 0..DECODE_STEPS {
            let tok = [500 + (s as i32 * 13) % 400];
            let input = mlxcel_core::from_slice_i32(&tok, &[1, 1]);
            let logits = model.forward(&input, &mut caches, None);
            step_hashes.push(hash_array(&logits));
        }
        mlxcel_core::synchronize_default();

        match &reference {
            None => reference = Some(step_hashes),
            Some(reference) => {
                let first_bad = step_hashes
                    .iter()
                    .zip(reference.iter())
                    .position(|(a, b)| a != b);
                assert_eq!(
                    &step_hashes, reference,
                    "iteration {iter} produced different logits than iteration 0 \
                     (first divergent step: {first_bad:?}, 0 = prefill); \
                     qmm_sm80 output is non-deterministic (issue #910)"
                );
            }
        }
    }
}
