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

//! Regression test for issue #2128: with `MLXCEL_SDPA_DETERMINISTIC=1` the
//! decode SDPA call must be bitwise deterministic.
//!
//! Once a KV cache reaches 256 positions, a one-row decode SDPA goes to cuDNN
//! over the whole cache buffer with a padding mask. On GB10 (sm_121, cuDNN
//! 9.27) the heuristics' first engine for the head_dim 128 graph uses stream-K
//! work distribution, which combines partial key reductions in completion
//! order: about 1 call in 100 rounds differently, and Qwen3 temp-0 output
//! changes from run to run. That stays the default because it is the fast
//! path at long context; the switch routes decode to `sdpa_vector` instead.
//! Without the switch the shape below gave two distinct outputs in 4000 calls
//! (28 to 46 of them the minority, across five runs); with it, one.
//!
//! No checkpoint is needed. Runs only with the `cuda` feature:
//!
//! ```sh
//! cargo test --profile test-fast --features cuda --test cuda_sdpa_determinism
//! ```
//!
//! The test turns the switch on unless the environment already sets it, so
//! running it with `MLXCEL_SDPA_DETERMINISTIC=0` shows the default path's
//! nondeterminism as a failure.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use mlxcel_core::{MlxArray, UniquePtr};

/// Turn on `MLXCEL_SDPA_DETERMINISTIC` before the first SDPA call reads it,
/// unless the caller set it explicitly.
fn enable_deterministic_sdpa() {
    if std::env::var_os("MLXCEL_SDPA_DETERMINISTIC").is_none() {
        // SAFETY: this binary's only test calls this before any MLX work, so no
        // other thread reads the environment concurrently.
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

/// Qwen3-1.7B's attention shape (16 query heads over 8 KV heads, head_dim 128)
/// at a 300-key decode step inside a 512-position cache buffer: the slice is
/// what `KVCache::update_and_fetch` hands the attention, so SDPA unslices it
/// to the buffer and takes cuDNN with a padding mask.
#[test]
fn cudnn_decode_sdpa_over_a_padded_kv_cache_is_bitwise_deterministic() {
    const HEADS: i32 = 16;
    const KV_HEADS: i32 = 8;
    const HEAD_DIM: i32 = 128;
    const EXTENT: i32 = 512;
    const LEN: i32 = 300;
    const CALLS: usize = 2000;

    enable_deterministic_sdpa();
    let k_buffer = fixed_f16(&[1, KV_HEADS, EXTENT, HEAD_DIM], 1);
    let v_buffer = fixed_f16(&[1, KV_HEADS, EXTENT, HEAD_DIM], 2);
    let q = fixed_f16(&[1, HEADS, 1, HEAD_DIM], 3);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut outputs: BTreeMap<u64, usize> = BTreeMap::new();
    for _ in 0..CALLS {
        let k = mlxcel_core::slice(&k_buffer, &[0, 0, 0, 0], &[1, KV_HEADS, LEN, HEAD_DIM]);
        let v = mlxcel_core::slice(&v_buffer, &[0, 0, 0, 0], &[1, KV_HEADS, LEN, HEAD_DIM]);
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
        "decode SDPA gave {} distinct outputs over {CALLS} identical calls (counts {:?}); \
         MLXCEL_SDPA_DETERMINISTIC={:?} does not make decode deterministic (issue #2128)",
        outputs.len(),
        outputs.values().collect::<Vec<_>>(),
        std::env::var("MLXCEL_SDPA_DETERMINISTIC").ok()
    );
}
