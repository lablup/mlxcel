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

//! Fused paged-attention decode kernel throughput bench (epic #116 Phase 6,
//! #123).
//!
//! Compares the fused Metal kernel (`paged_decode_attention_pooled` with the
//! native path enabled, which reads scattered pool blocks directly) against the
//! gather-then-SDPA reference (`paged_decode_attention_pooled_fallback`, which
//! re-materialises a contiguous K/V every step) over a real `PagedBlockPool`.
//!
//! This is the #123 counterpart to `examples/page_gather_microbench.rs` (the
//! ADR 0001 spike): the spike measured gather overhead against a contiguous
//! SDPA lower bound; this measures the kernel that removes that gather. The
//! sequences are grown with interleaved block writes so their physical pool
//! rows are scattered (the gather pays its real scatter cost).
//!
//! Run:
//!   cargo run --release --features metal,accelerate \
//!     --example paged_attention_kernel_bench
//! Run under `caffeinate -i` and let the machine cool between sweeps; Apple
//! Silicon down-clocks under sustained load.

use std::time::{Duration, Instant};

use mlxcel_core::cache::{PagedBlockPool, PagedKvLayout, PagedSequenceState};
use mlxcel_core::layers::{paged_decode_attention_pooled, paged_decode_attention_pooled_fallback};
use mlxcel_core::{MlxArray, UniquePtr, astype, eval, from_slice_f32, synchronize_default};

const HEAD_DIM: i32 = 128;
const Q_HEADS: i32 = 32;
const KV_HEADS: i32 = 8;
const BLOCK_SIZE: usize = 32;
const LAYER: usize = 0;
const WARMUP: usize = 20;
const ITERS: usize = 50;

/// Deterministic pseudo-random f32 fill (a cheap LCG; only the timing matters,
/// but distinct values keep softmax non-degenerate).
fn pseudo_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32;
            u / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn f16(vals: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    astype(&from_slice_f32(vals, shape), mlxcel_core::dtype::FLOAT16)
}

/// warmup loop, then a `synchronize_default()`-bracketed timed loop that evals
/// each result. Matches the harness in `page_gather_microbench.rs`.
fn time_body<F>(mut body: F) -> Duration
where
    F: FnMut() -> UniquePtr<MlxArray>,
{
    for _ in 0..WARMUP {
        let out = body();
        eval(&out);
    }
    synchronize_default();
    let start = Instant::now();
    for _ in 0..ITERS {
        let out = body();
        eval(&out);
    }
    synchronize_default();
    start.elapsed()
}

/// Build a pool holding `batch` sequences of `ctx` tokens each, grown with
/// interleaved block writes so physical rows are scattered. Returns the pool
/// plus the per-sequence states.
fn build_pool(batch: usize, ctx: usize) -> (PagedBlockPool, Vec<PagedSequenceState>) {
    let layout = PagedKvLayout::uniform(
        1,
        BLOCK_SIZE,
        BLOCK_SIZE * KV_HEADS as usize * HEAD_DIM as usize * 2,
    )
    .expect("valid layout");
    let mut pool = PagedBlockPool::new(layout);
    let mut states: Vec<PagedSequenceState> = (0..batch)
        .map(|_| PagedSequenceState::new(pool.layout()))
        .collect();

    let n_blocks = ctx / BLOCK_SIZE;
    for blk in 0..n_blocks {
        for (seq_idx, state) in states.iter_mut().enumerate() {
            pool.append_tokens(state, LAYER, BLOCK_SIZE)
                .expect("append");
            let block_id = *state
                .layer(LAYER)
                .unwrap()
                .block_ids
                .last()
                .expect("block id");
            let seed = (seq_idx as u64) * 1_000_003 + blk as u64 + 1;
            let k = f16(
                &pseudo_f32(seed, KV_HEADS as usize * BLOCK_SIZE * HEAD_DIM as usize),
                &[1, KV_HEADS, BLOCK_SIZE as i32, HEAD_DIM],
            );
            let v = f16(
                &pseudo_f32(
                    seed.wrapping_mul(7),
                    KV_HEADS as usize * BLOCK_SIZE * HEAD_DIM as usize,
                ),
                &[1, KV_HEADS, BLOCK_SIZE as i32, HEAD_DIM],
            );
            pool.write_block(block_id, LAYER, 0, &k, &v).expect("write");
        }
    }
    (pool, states)
}

fn per_call_us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6 / ITERS as f64
}

fn run_config(batch: usize, ctx: usize) {
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let (pool, states) = build_pool(batch, ctx);
    let state_refs: Vec<&PagedSequenceState> = states.iter().collect();

    let q = f16(
        &pseudo_f32(
            0x5151_u64.wrapping_add(ctx as u64),
            batch * Q_HEADS as usize * HEAD_DIM as usize,
        ),
        &[batch as i32, Q_HEADS, 1, HEAD_DIM],
    );

    let gather = time_body(|| {
        paged_decode_attention_pooled_fallback(&q, &pool, &state_refs, LAYER, scale).unwrap()
    });
    let fused = time_body(|| {
        paged_decode_attention_pooled(&q, &pool, &state_refs, LAYER, scale, true).unwrap()
    });

    let g = per_call_us(gather);
    let f = per_call_us(fused);
    let speedup = g / f;
    println!(
        "  batch={batch:>2} ctx={ctx:>6}  gather={g:>9.1}us  fused={f:>9.1}us  speedup={speedup:>5.2}x"
    );
}

fn main() {
    println!("=== mlxcel fused paged-attention decode kernel bench (#123) ===");
    println!(
        "head_dim={HEAD_DIM} q_heads={Q_HEADS} kv_heads={KV_HEADS} block_size={BLOCK_SIZE} dtype=f16"
    );
    println!("warmup={WARMUP} iters={ITERS}");
    println!("Tip: run under `caffeinate -i` and let the machine cool between sweeps.");
    println!();
    for &ctx in &[4096usize, 16384] {
        for &batch in &[1usize, 4, 8, 16] {
            run_config(batch, ctx);
        }
    }
}
