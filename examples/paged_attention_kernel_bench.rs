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
//! #123; adaptive selector #331).
//!
//! Compares the fused kernel (raw `PagedBlockPool::paged_decode_fused`, which
//! reads scattered pool blocks directly) against the gather-then-SDPA reference
//! (`paged_decode_attention_pooled_fallback`, which re-materialises a contiguous
//! K/V every step) over a real `PagedBlockPool`, and prints the decision the
//! adaptive selector (`select_pooled_paged_dispatch`, #331) makes for each shape
//! so both dispatch arms are visible. The fused kernel runs its Metal JIT body
//! on Apple and the #634 CUDA port on NVIDIA, so this bench exercises the native
//! arm on either GPU backend.
//!
//! This is the #123 counterpart to `examples/page_gather_microbench.rs` (the
//! ADR 0001 spike): the spike measured gather overhead against a contiguous
//! SDPA lower bound; this measures the kernel that removes that gather. The
//! sequences are grown with interleaved block writes so their physical pool
//! rows are scattered (the gather pays its real scatter cost).
//!
//! The fused kernel reads one contiguous pool buffer per side, so with chunked
//! slabs (#235) it can only run while a layer fits in a single 32-block slab
//! (1024 tokens at block_size 32); past that it declines and the `fused` column
//! reads `declined`. The selector encodes exactly that (plus the ADR 0001
//! batch/context regime), so the two are cross-checked here.
//!
//! Run (Apple):
//!   cargo run --release --features metal,accelerate \
//!     --example paged_attention_kernel_bench
//! Run (CUDA):
//!   cargo run --release --features cuda --example paged_attention_kernel_bench
//! On Apple Silicon run under `caffeinate -i` and let the machine cool between
//! sweeps; it down-clocks under sustained load.

use std::time::{Duration, Instant};

use mlxcel_core::cache::{PagedBlockPool, PagedKvLayout, PagedSequenceState};
use mlxcel_core::layers::{
    PagedDecodeBackend, PagedDecodeDispatch, paged_decode_attention_pooled_fallback,
    select_pooled_paged_dispatch,
};
use mlxcel_core::{MlxArray, UniquePtr, astype, eval, from_slice_f32, synchronize_default};

const HEAD_DIM: i32 = 128;
const Q_HEADS: i32 = 32;
const KV_HEADS: i32 = 8;
const BLOCK_SIZE: usize = 32;
const LAYER: usize = 0;
const WARMUP: usize = 20;
const ITERS: usize = 50;

/// Backend the fused kernel actually dispatches on: Metal on Apple, the #634
/// CUDA port on NVIDIA, else CPU (no native kernel). Drives the selector label
/// so the printed `select=` column matches production dispatch on this host.
fn bench_backend() -> PagedDecodeBackend {
    if mlxcel_core::metal_is_available() {
        PagedDecodeBackend::Metal
    } else if mlxcel_core::cuda_is_available() {
        PagedDecodeBackend::Cuda
    } else {
        PagedDecodeBackend::Other
    }
}

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

    // Selector decision for this shape on this host's backend (Metal on Apple,
    // the #634 CUDA port on NVIDIA). Mirrors what production dispatch picks.
    let visible_len = state_refs
        .iter()
        .map(|s| s.layer(LAYER).map_or(0, |l| l.visible_len()))
        .max()
        .unwrap_or(0);
    let slabs = pool.slab_count(LAYER);
    let decision = select_pooled_paged_dispatch(batch, visible_len, slabs, bench_backend());
    let pick = match decision {
        PagedDecodeDispatch::Native => "native",
        PagedDecodeDispatch::Gather => "gather",
    };

    // Arm A: gather-then-SDPA reference (strategy A).
    let gather = time_body(|| {
        paged_decode_attention_pooled_fallback(&q, &pool, &state_refs, LAYER, scale).unwrap()
    });
    let g = per_call_us(gather);

    // Arm B: raw fused kernel (strategy B), bypassing the selector. Declines
    // (returns None) once the layer spans more than one slab, so probe first.
    let fused_available = pool
        .paged_decode_fused(&q, &state_refs, LAYER, scale)
        .unwrap()
        .is_some();

    if fused_available {
        let fused = time_body(|| {
            pool.paged_decode_fused(&q, &state_refs, LAYER, scale)
                .unwrap()
                .unwrap()
        });
        let f = per_call_us(fused);
        let speedup = g / f;
        println!(
            "  batch={batch:>2} ctx={ctx:>6} slabs={slabs:>3} select={pick}  gather={g:>9.1}us  fused={f:>9.1}us  speedup={speedup:>5.2}x"
        );
    } else {
        println!(
            "  batch={batch:>2} ctx={ctx:>6} slabs={slabs:>3} select={pick}  gather={g:>9.1}us  fused=declined(multi-slab)"
        );
    }
}

fn main() {
    println!("=== mlxcel fused paged-attention decode kernel bench (#123, selector #331) ===");
    println!(
        "head_dim={HEAD_DIM} q_heads={Q_HEADS} kv_heads={KV_HEADS} block_size={BLOCK_SIZE} dtype=f16"
    );
    println!("warmup={WARMUP} iters={ITERS}");
    println!("Tip: run under `caffeinate -i` and let the machine cool between sweeps.");
    println!(
        "Note: the fused kernel runs only single-slab (<= {} tokens here); larger contexts decline and read `gather`.",
        BLOCK_SIZE * 32
    );
    println!();
    // Contexts 1k/4k/16k and batches 1/4/8 (issue #331 acceptance criterion 4);
    // 512 and batch 16 are kept as extra single-slab / high-batch reference rows.
    // The pool's slab count is per-layer across ALL sequences, so a batched
    // layer spans B * ceil(ctx / block_size) rows: B>=4 at any of these contexts
    // is already multi-slab and the kernel declines. select=gather throughout.
    for &ctx in &[512usize, 1024, 4096, 16384] {
        for &batch in &[1usize, 4, 8, 16] {
            run_config(batch, ctx);
        }
    }

    // Single-slab batched island: the only regime where a B>=4 layer still fits
    // one 32-row slab (total rows = B * ceil(ctx / 32) <= 32), so the kernel is
    // actually serviceable and the selector picks native. This is the reachable
    // remnant of ADR 0001's "batched moderate-context" win after chunked slabs
    // (#235) shrank the servable context; it exercises the native arm end to end.
    println!();
    println!("-- single-slab batched island (select=native, fused kernel live) --");
    for &(batch, ctx) in &[(4usize, 128usize), (4, 256), (8, 128)] {
        run_config(batch, ctx);
    }
}
