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

//! Regression tests for issue #2128: with `MLXCEL_SDPA_DETERMINISTIC=1` the
//! decode SDPA call, and a Qwen3 decode past 256 KV positions, must be bitwise
//! deterministic.
//!
//! Once a KV cache reaches 256 positions, a one-row decode SDPA goes to cuDNN
//! over the whole cache buffer with a padding mask. On GB10 (sm_121, cuDNN
//! 9.27) the heuristics' first engine for the head_dim 128 graph, a stream-K
//! engine, rounds differently in about 1 call in 100, and Qwen3 temp-0 output
//! changes from run to run. That stays the default because it is the fast
//! path at long context; the switch routes decode to `sdpa_vector` instead.
//!
//! Runs only with the `cuda` feature; the Qwen3 test skips when its checkpoint
//! (`MLXCEL_TEST_QWEN3_MODEL_DIR`) is absent, the SDPA test needs none:
//!
//! ```sh
//! cargo test --profile test-fast --features cuda --test cuda_sdpa_determinism
//! ```
//!
//! This binary turns the switch on before `main` unless the environment
//! already sets it, so running it with `MLXCEL_SDPA_DETERMINISTIC=0` shows the
//! default path's nondeterminism as failures. The switch lives in its own test
//! binary so that `cuda_qmm_determinism` keeps guarding the default dispatch.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use mlxcel::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

const DEFAULT_QWEN3_MODEL_DIR: &str = "models/mlx/qwen3-0.6b-4bit";

/// Turn on `MLXCEL_SDPA_DETERMINISTIC` before MLX latches it into a static,
/// unless the environment sets it explicitly.
#[ctor::ctor(unsafe)]
fn enable_deterministic_sdpa() {
    if std::env::var_os("MLXCEL_SDPA_DETERMINISTIC").is_none() {
        // SAFETY: ctor runs before main, on one thread, before anything else
        // can read the environment.
        unsafe { std::env::set_var("MLXCEL_SDPA_DETERMINISTIC", "1") };
    }
}

fn hash_array(arr: &MlxArray) -> u64 {
    let f = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    let mut h = DefaultHasher::new();
    mlxcel_core::array_to_raw_bytes(&f).hash(&mut h);
    h.finish()
}

fn fixed_f16(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state = seed;
    let values: Vec<f32> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    let out = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&values, shape),
        mlxcel_core::dtype::FLOAT16,
    );
    mlxcel_core::eval(&out);
    out
}

/// Repeats one decode SDPA call: Qwen3-1.7B's attention shape (16 query heads
/// over 8 KV heads, head_dim 128), k and v the first `len` rows of an
/// `extent`-row cache buffer, as `KVCache::update_and_fetch` hands them over.
fn assert_decode_sdpa_is_repeatable(extent: i32, len: i32) {
    const HEADS: i32 = 16;
    const KV_HEADS: i32 = 8;
    const HEAD_DIM: i32 = 128;
    const CALLS: usize = 2000;

    let k_buffer = fixed_f16(&[1, KV_HEADS, extent, HEAD_DIM], 1);
    let v_buffer = fixed_f16(&[1, KV_HEADS, extent, HEAD_DIM], 2);
    let q = fixed_f16(&[1, HEADS, 1, HEAD_DIM], 3);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut outputs: BTreeMap<u64, usize> = BTreeMap::new();
    for _ in 0..CALLS {
        let k = mlxcel_core::slice(&k_buffer, &[0, 0, 0, 0], &[1, KV_HEADS, len, HEAD_DIM]);
        let v = mlxcel_core::slice(&v_buffer, &[0, 0, 0, 0], &[1, KV_HEADS, len, HEAD_DIM]);
        // Evaluated slices, as a decode step's cache views are by the time
        // SDPA runs; `use_cudnn_for_decoding` needs their strides.
        mlxcel_core::eval(&k);
        mlxcel_core::eval(&v);
        // SAFETY: a null mask pointer is the documented "no mask" argument.
        let out = unsafe {
            mlxcel_core::layers::attention_from_ptr(&q, &k, &v, scale, std::ptr::null(), 0.0, 0)
        };
        *outputs.entry(hash_array(&out)).or_insert(0) += 1;
    }
    assert_eq!(
        outputs.len(),
        1,
        "decode SDPA over {len} of {extent} cache rows gave {} distinct outputs over {CALLS} \
         identical calls (counts {:?}); MLXCEL_SDPA_DETERMINISTIC={:?} (issue #2128)",
        outputs.len(),
        outputs.values().collect::<Vec<_>>(),
        std::env::var("MLXCEL_SDPA_DETERMINISTIC").ok()
    );
}

/// The shape that reproduced #2128: 300 keys in a 512-row buffer. Without the
/// switch, cuDNN gave two distinct outputs in 4000 calls (16 to 46 of them the
/// minority, across runs); with it, `sdpa_vector`'s one-pass kernel gives one.
#[test]
fn cudnn_decode_sdpa_over_a_padded_kv_cache_is_bitwise_deterministic() {
    assert_decode_sdpa_is_repeatable(512, 300);
}

/// Past 1024 keys `sdpa_vector` splits the key axis into blocks and reduces
/// them in a second pass; that reduction must be in a fixed order too.
#[test]
fn two_pass_vector_decode_sdpa_is_bitwise_deterministic() {
    assert_decode_sdpa_is_repeatable(2048, 1500);
}

/// The model-level reproduction: 64 prompt positions and 400 fixed decode
/// tokens take the KV cache past 256 positions. Without the switch,
/// Qwen3-1.7B's logits diverged between two in-process repeats from decode
/// step 225 to 247 in every comparison made, and Qwen3-0.6B's from step 215
/// to 220.
#[test]
fn temp0_qwen3_decode_past_the_cudnn_sdpa_threshold_is_bitwise_deterministic() {
    const ITERS: usize = 3;
    const PREFILL_LEN: usize = 64;
    const DECODE_STEPS: usize = 400;

    let model_dir = std::env::var("MLXCEL_TEST_QWEN3_MODEL_DIR")
        .unwrap_or_else(|_| DEFAULT_QWEN3_MODEL_DIR.to_string());
    if !Path::new(&model_dir).exists() {
        eprintln!("skipping: {model_dir} not present");
        return;
    }
    let (model, _) = mlxcel::load_model(Path::new(&model_dir)).expect("load model");
    let prompt: Vec<i32> = (0..PREFILL_LEN)
        .map(|i| 100 + (i as i32 * 37) % 900)
        .collect();

    let mut reference: Option<Vec<u64>> = None;
    for iter in 0..ITERS {
        let mut caches: Vec<KVCache> = model.make_caches();
        let mut step_hashes = Vec::with_capacity(1 + DECODE_STEPS);
        let input = mlxcel_core::from_slice_i32(&prompt, &[1, PREFILL_LEN as i32]);
        step_hashes.push(hash_array(&model.forward(&input, &mut caches, None)));
        for s in 0..DECODE_STEPS {
            let tok = [500 + (s as i32 * 13) % 400];
            let input = mlxcel_core::from_slice_i32(&tok, &[1, 1]);
            step_hashes.push(hash_array(&model.forward(&input, &mut caches, None)));
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
                    &step_hashes,
                    reference,
                    "{model_dir}: iteration {iter} produced different logits than iteration 0 \
                     (first divergent step: {first_bad:?}, 0 = prefill) with \
                     MLXCEL_SDPA_DETERMINISTIC={:?} (issue #2128)",
                    std::env::var("MLXCEL_SDPA_DETERMINISTIC").ok()
                );
            }
        }
    }
}
