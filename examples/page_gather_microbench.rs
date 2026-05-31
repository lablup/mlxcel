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

//! Synthetic op-level microbench for paged KV decode attention.
//!
//! Part of epic #116 (unified paged KV cache), Phase 0 spike (#117). This
//! measures the decode-step cost of gathering scattered physical KV blocks on
//! MLX so we can decide between two attention strategies for the paged store:
//!
//! * (A) gather-then-SDPA: use `take` + `reshape` + `transpose` to materialize
//!   a contiguous per-sequence K/V from scattered blocks, then call the fused
//!   `scaled_dot_product_attention`. Reuses existing FFI, no new kernel.
//! * (B) a fused Metal paged-attention kernel that reads scattered blocks
//!   directly with no gather copy.
//!
//! The bench is fully synthetic: it allocates fake K/V with `zeros` (values do
//! not matter, only timing) and times three decode-step attention paths plus a
//! per-step block-append (`slice_update`) across a sweep of context lengths,
//! batch sizes, and block sizes. It loads no model. Output feeds ADR 0001
//! (`docs/adr/0001-paged-attention-gather-vs-fused-kernel.md`): the gather
//! overhead vs contiguous SDPA, and the `take`/`slice_update` cost of two pool
//! tensor layouts, decide (A)-first and the pool layout.
//!
//! Reproduce (use `caffeinate -i` so the host does not idle-throttle the GPU
//! mid-run, and let the machine run cool between sweeps for stable numbers):
//!
//! ```text
//! caffeinate -i cargo run --release --features metal,accelerate \
//!     --example page_gather_microbench
//! ```
//!
//! Or via the wrapper: `scripts/run_page_gather_microbench.sh`.

use std::time::{Duration, Instant};

use clap::Parser;
use mlxcel_core::{
    MlxArray, UniquePtr, eval, fast_scaled_dot_product_attention, from_slice_i32, reshape,
    slice_update, synchronize_default, take, transpose_axes, zeros,
};

// MLX dtype ids (see src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp:50-51).
const F16: i32 = 9;

#[derive(Parser, Debug)]
#[command(name = "page_gather_microbench")]
struct Args {
    /// Per-head dimension.
    #[arg(long, default_value = "128")]
    head_dim: i32,

    /// Number of query heads.
    #[arg(long, default_value = "32")]
    q_heads: i32,

    /// Number of key/value heads (GQA groups).
    #[arg(long, default_value = "8")]
    kv_heads: i32,

    /// Comma-separated batch sizes to sweep.
    #[arg(long, default_value = "1,4")]
    batch_sizes: String,

    /// Comma-separated context lengths to sweep.
    #[arg(long, default_value = "1024,4096,16384,32768")]
    context_lengths: String,

    /// Comma-separated block sizes to sweep.
    #[arg(long, default_value = "16,32,64")]
    block_sizes: String,

    /// Warmup iterations per measured body.
    #[arg(long, default_value = "20")]
    warmup: usize,

    /// Timed iterations per measured body.
    #[arg(long, default_value = "50")]
    iters: usize,
}

/// Parse a comma-separated list of `usize` (whitespace tolerant).
fn parse_usize_list(s: &str) -> Vec<usize> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid list value: {t:?}"))
        })
        .collect()
}

/// Per-call microseconds for a timed body.
fn per_call_us(total: Duration, iters: usize) -> f64 {
    (total.as_nanos() as f64 / iters as f64) / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // --- parse_usize_list ---

    #[test]
    fn parse_usize_list_single() {
        assert_eq!(parse_usize_list("42"), vec![42]);
    }

    #[test]
    fn parse_usize_list_multi() {
        assert_eq!(parse_usize_list("1,4,16"), vec![1, 4, 16]);
    }

    #[test]
    fn parse_usize_list_whitespace() {
        assert_eq!(parse_usize_list(" 8 , 32 , 64 "), vec![8, 32, 64]);
    }

    #[test]
    fn parse_usize_list_trailing_comma() {
        // A trailing comma produces an empty token that is filtered out.
        assert_eq!(parse_usize_list("1,2,"), vec![1, 2]);
    }

    #[test]
    fn parse_usize_list_empty() {
        assert!(parse_usize_list("").is_empty());
    }

    // --- per_call_us ---

    #[test]
    fn per_call_us_round() {
        // 10ms total / 10 iters = 1000us per call.
        let total = Duration::from_millis(10);
        assert!((per_call_us(total, 10) - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn per_call_us_fractional() {
        // 1ms total / 4 iters = 250us per call.
        let total = Duration::from_millis(1);
        assert!((per_call_us(total, 4) - 250.0).abs() < 1e-6);
    }

    // --- ctx_pad / frag_pct math (inlined from run_config) ---

    fn ctx_frag(ctx: usize, block: usize) -> (usize, f64) {
        let nb = ctx.div_ceil(block);
        let ctx_pad = nb * block;
        let frag_pct = (ctx_pad - ctx) as f64 / ctx as f64 * 100.0;
        (ctx_pad, frag_pct)
    }

    #[test]
    fn ctx_pad_exact_multiple() {
        // ctx=1024, block=32 → already aligned, 0% fragmentation.
        let (pad, frag) = ctx_frag(1024, 32);
        assert_eq!(pad, 1024);
        assert!(frag.abs() < 1e-9);
    }

    #[test]
    fn ctx_pad_non_multiple() {
        // ctx=1000, block=32 → rounds up to 1024 (32 * 32), frag = 24/1000 * 100.
        let (pad, frag) = ctx_frag(1000, 32);
        assert_eq!(pad, 1024);
        let expected = 24.0 / 1000.0 * 100.0;
        assert!((frag - expected).abs() < 1e-9);
    }

    #[test]
    fn ctx_pad_block_size_1() {
        // block=1 always produces 0% frag.
        let (pad, frag) = ctx_frag(777, 1);
        assert_eq!(pad, 777);
        assert!(frag.abs() < 1e-9);
    }

    #[test]
    fn ctx_pad_ctx_equals_block() {
        // ctx == block → exactly one block, 0% frag.
        let (pad, frag) = ctx_frag(64, 64);
        assert_eq!(pad, 64);
        assert!(frag.abs() < 1e-9);
    }
}

/// Human-readable line for a single measured body.
fn fmt_per_call(label: &str, total: Duration, iters: usize) {
    println!(
        "  {:<26} total={:>8.2}ms  per_call={:>9.3}us  ({} iters)",
        label,
        total.as_secs_f64() * 1000.0,
        per_call_us(total, iters),
        iters
    );
}

/// Eval two arrays built on the same step. `keep` is returned so `time_body`
/// can eval it; `also` is evaled here so both outputs are forced. Used to time
/// the gather of K and V together (both are needed before SDPA can run).
fn eval_pair(keep: UniquePtr<MlxArray>, also: &MlxArray) -> UniquePtr<MlxArray> {
    eval(also);
    keep
}

/// Time a closure that builds an `MlxArray` per iteration. Warms up, then runs
/// `iters` timed iterations. Each iteration evals its result; a single
/// `synchronize_default()` brackets the timed region (matching the eval-per-iter
/// pattern in `examples/bridge_overhead_microbench.rs`).
fn time_body<F>(warmup: usize, iters: usize, mut body: F) -> Duration
where
    F: FnMut() -> UniquePtr<MlxArray>,
{
    for _ in 0..warmup {
        let out = body();
        eval(&out);
    }
    synchronize_default();
    let start = Instant::now();
    for _ in 0..iters {
        let out = body();
        eval(&out);
    }
    synchronize_default();
    start.elapsed()
}

/// `scale = 1 / sqrt(head_dim)`.
fn sdpa_scale(head_dim: i32) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

/// Fused SDPA with a null mask. A single decode query attends to all visible
/// keys, so no causal mask is needed.
fn sdpa(q: &MlxArray, k: &MlxArray, v: &MlxArray, scale: f32) -> UniquePtr<MlxArray> {
    // SAFETY: `fast_scaled_dot_product_attention` is a thin FFI wrapper over the
    // MLX fast SDPA kernel. The null mask pointer is the documented "no mask"
    // sentinel; q/k/v outlive the call.
    unsafe { fast_scaled_dot_product_attention(q, k, v, scale, std::ptr::null()) }
}

/// One measured config row. Timings are stored as raw totals over `iters`
/// iterations; per-call microseconds are derived at print time via
/// `per_call_us` so there is no double division.
struct Row {
    batch: usize,
    ctx: usize,
    ctx_pad: usize,
    block: usize,
    frag_pct: f64,
    iters: usize,
    contig_sdpa: Duration,
    gather_a_only: Duration,
    gather_a_sdpa: Duration,
    gather_b_only: Duration,
    gather_b_sdpa: Duration,
    sliceupd_a: Duration,
    sliceupd_b: Duration,
}

impl Row {
    fn contig_sdpa_us(&self) -> f64 {
        per_call_us(self.contig_sdpa, self.iters)
    }
    fn gather_a_only_us(&self) -> f64 {
        per_call_us(self.gather_a_only, self.iters)
    }
    fn gather_a_sdpa_us(&self) -> f64 {
        per_call_us(self.gather_a_sdpa, self.iters)
    }
    fn gather_b_only_us(&self) -> f64 {
        per_call_us(self.gather_b_only, self.iters)
    }
    fn gather_b_sdpa_us(&self) -> f64 {
        per_call_us(self.gather_b_sdpa, self.iters)
    }
    fn sliceupd_a_us(&self) -> f64 {
        per_call_us(self.sliceupd_a, self.iters)
    }
    fn sliceupd_b_us(&self) -> f64 {
        per_call_us(self.sliceupd_b, self.iters)
    }
    fn overhead_a_pct(&self) -> f64 {
        (self.gather_a_sdpa_us() - self.contig_sdpa_us()) / self.contig_sdpa_us() * 100.0
    }
    fn overhead_b_pct(&self) -> f64 {
        (self.gather_b_sdpa_us() - self.contig_sdpa_us()) / self.contig_sdpa_us() * 100.0
    }
}

#[allow(clippy::too_many_arguments)]
fn run_config(
    head_dim: i32,
    q_heads: i32,
    kv_heads: i32,
    batch: usize,
    ctx: usize,
    block: usize,
    warmup: usize,
    iters: usize,
) -> Row {
    let d = head_dim;
    let hq = q_heads;
    let hkv = kv_heads;
    let bs = block as i32;
    let b = batch as i32;
    let scale = sdpa_scale(head_dim);

    // Blocks per sequence and the padded (block-aligned) context length. We
    // attend over `ctx_pad` keys on every path so the comparison is apples to
    // apples; `frag` is the internal fragmentation the block size induces.
    let nb = ctx.div_ceil(block);
    let ctx_pad = nb * block;
    let frag_pct = (ctx_pad - ctx) as f64 / ctx as f64 * 100.0;
    let t_pad = ctx_pad as i32;
    let total_blocks = batch * nb;
    // 2x slack forces scattered (non-contiguous) physical ids.
    let pool_blocks = total_blocks * 2;

    // Decode query: one new token per sequence.
    let q = zeros(&[b, hq, 1, d], F16);

    // Scattered, deterministic, unique block ids: reverse order over the pool.
    let ids: Vec<i32> = (0..total_blocks)
        .map(|i| (pool_blocks - 1 - i) as i32)
        .collect();
    let block_ids = from_slice_i32(&ids, &[total_blocks as i32]);

    // Path 1 inputs: contiguous per-sequence K/V (the lower bound).
    let kc = zeros(&[b, hkv, t_pad, d], F16);
    let vc = zeros(&[b, hkv, t_pad, d], F16);

    // Layout A pools: [num_blocks, block_size, n_kv_heads, head_dim].
    let pool_k_a = zeros(&[pool_blocks as i32, bs, hkv, d], F16);
    let pool_v_a = zeros(&[pool_blocks as i32, bs, hkv, d], F16);
    let new_block_a = zeros(&[1, bs, hkv, d], F16);

    // Layout B pools (head-split): [n_kv_heads, num_blocks, block_size, head_dim].
    let pool_k_b = zeros(&[hkv, pool_blocks as i32, bs, d], F16);
    let pool_v_b = zeros(&[hkv, pool_blocks as i32, bs, d], F16);
    let new_block_b = zeros(&[hkv, 1, bs, d], F16);

    // Pre-eval all inputs once so allocation/fill cost is not in the timed
    // region (mirrors `bench_matmul_512` pre-evaling its inputs).
    for arr in [
        &q,
        &block_ids,
        &kc,
        &vc,
        &pool_k_a,
        &pool_v_a,
        &new_block_a,
        &pool_k_b,
        &pool_v_b,
        &new_block_b,
    ] {
        eval(arr);
    }
    synchronize_default();

    // Layout A gather: take(axis=0) + reshape + transpose for K and V.
    // [pool, BS, Hkv, D] --take--> [total_blocks, BS, Hkv, D]
    //   --reshape--> [B, T_pad, Hkv, D] --transpose--> [B, Hkv, T_pad, D].
    let gather_a = |q: &MlxArray, attend: bool| -> UniquePtr<MlxArray> {
        let kg = take(&pool_k_a, &block_ids, 0);
        let kg = reshape(&kg, &[b, t_pad, hkv, d]);
        let kg = transpose_axes(&kg, &[0, 2, 1, 3]);
        let vg = take(&pool_v_a, &block_ids, 0);
        let vg = reshape(&vg, &[b, t_pad, hkv, d]);
        let vg = transpose_axes(&vg, &[0, 2, 1, 3]);
        if attend {
            sdpa(q, &kg, &vg, scale)
        } else {
            eval_pair(kg, &vg)
        }
    };

    // Layout B gather (head-split): take(axis=1) + reshape + transpose.
    // [Hkv, pool, BS, D] --take--> [Hkv, total_blocks, BS, D]
    //   --reshape--> [Hkv, B, T_pad, D] --transpose--> [B, Hkv, T_pad, D].
    let gather_b = |q: &MlxArray, attend: bool| -> UniquePtr<MlxArray> {
        let kg = take(&pool_k_b, &block_ids, 1);
        let kg = reshape(&kg, &[hkv, b, t_pad, d]);
        let kg = transpose_axes(&kg, &[1, 0, 2, 3]);
        let vg = take(&pool_v_b, &block_ids, 1);
        let vg = reshape(&vg, &[hkv, b, t_pad, d]);
        let vg = transpose_axes(&vg, &[1, 0, 2, 3]);
        if attend {
            sdpa(q, &kg, &vg, scale)
        } else {
            eval_pair(kg, &vg)
        }
    };

    // Path 1: contiguous SDPA (baseline / lower bound).
    let contig_sdpa = time_body(warmup, iters, || sdpa(&q, &kc, &vc, scale));
    // Path 2 (layout A): gather-only, then gather + SDPA.
    let gather_a_only = time_body(warmup, iters, || gather_a(&q, false));
    let gather_a_sdpa = time_body(warmup, iters, || gather_a(&q, true));
    // Path 3 (layout B): gather-only, then gather + SDPA.
    let gather_b_only = time_body(warmup, iters, || gather_b(&q, false));
    let gather_b_sdpa = time_body(warmup, iters, || gather_b(&q, true));
    // Path 4: per-step block append (slice_update of one fresh block).
    let sliceupd_a = time_body(warmup, iters, || {
        slice_update(&pool_k_a, &new_block_a, &[0, 0, 0, 0], &[1, bs, hkv, d])
    });
    let sliceupd_b = time_body(warmup, iters, || {
        slice_update(&pool_k_b, &new_block_b, &[0, 0, 0, 0], &[hkv, 1, bs, d])
    });

    Row {
        batch,
        ctx,
        ctx_pad,
        block,
        frag_pct,
        iters,
        contig_sdpa,
        gather_a_only,
        gather_a_sdpa,
        gather_b_only,
        gather_b_sdpa,
        sliceupd_a,
        sliceupd_b,
    }
}

fn main() {
    let args = Args::parse();

    let batch_sizes = parse_usize_list(&args.batch_sizes);
    let context_lengths = parse_usize_list(&args.context_lengths);
    let block_sizes = parse_usize_list(&args.block_sizes);

    println!("=== mlxcel page-gather microbench (epic #116 Phase 0 / #117) ===");
    println!(
        "head_dim={} q_heads={} kv_heads={} dtype=f16",
        args.head_dim, args.q_heads, args.kv_heads
    );
    println!("warmup={} iters={}", args.warmup, args.iters);
    println!("Tip: run under `caffeinate -i` and let the machine cool between sweeps.");
    println!();

    let mut rows: Vec<Row> = Vec::new();
    for &batch in &batch_sizes {
        for &ctx in &context_lengths {
            for &block in &block_sizes {
                println!("--- batch={batch} ctx={ctx} block={block} ---");
                let row = run_config(
                    args.head_dim,
                    args.q_heads,
                    args.kv_heads,
                    batch,
                    ctx,
                    block,
                    args.warmup,
                    args.iters,
                );
                println!("  ctx_pad={} frag={:.2}%", row.ctx_pad, row.frag_pct);
                fmt_per_call("contig_sdpa", row.contig_sdpa, row.iters);
                fmt_per_call("gatherA_only", row.gather_a_only, row.iters);
                fmt_per_call("gatherA_sdpa", row.gather_a_sdpa, row.iters);
                fmt_per_call("gatherB_only", row.gather_b_only, row.iters);
                fmt_per_call("gatherB_sdpa", row.gather_b_sdpa, row.iters);
                fmt_per_call("sliceupd_A", row.sliceupd_a, row.iters);
                fmt_per_call("sliceupd_B", row.sliceupd_b, row.iters);
                println!(
                    "  overheadA={:.1}%  overheadB={:.1}%",
                    row.overhead_a_pct(),
                    row.overhead_b_pct()
                );
                println!();
                rows.push(row);
            }
        }
    }

    // Aligned human-readable summary table.
    println!("=== summary (per-call microseconds) ===");
    println!(
        "{:>5} {:>7} {:>7} {:>5} {:>7} | {:>12} {:>12} {:>12} {:>12} {:>12} {:>11} {:>11} | {:>10} {:>10}",
        "B",
        "ctx",
        "ctxpad",
        "blk",
        "frag%",
        "contig_sdpa",
        "gatherA_only",
        "gatherA_sdpa",
        "gatherB_only",
        "gatherB_sdpa",
        "sliceupd_A",
        "sliceupd_B",
        "overheadA%",
        "overheadB%",
    );
    for r in &rows {
        println!(
            "{:>5} {:>7} {:>7} {:>5} {:>7.2} | {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>11.3} {:>11.3} | {:>10.1} {:>10.1}",
            r.batch,
            r.ctx,
            r.ctx_pad,
            r.block,
            r.frag_pct,
            r.contig_sdpa_us(),
            r.gather_a_only_us(),
            r.gather_a_sdpa_us(),
            r.gather_b_only_us(),
            r.gather_b_sdpa_us(),
            r.sliceupd_a_us(),
            r.sliceupd_b_us(),
            r.overhead_a_pct(),
            r.overhead_b_pct(),
        );
    }
    println!();

    // Machine-readable CSV block (one line per config, each prefixed `CSV:`).
    println!(
        "CSV:batch,ctx,ctx_pad,block,frag_pct,contig_sdpa_us,gatherA_only_us,gatherA_sdpa_us,gatherB_only_us,gatherB_sdpa_us,sliceupd_a_us,sliceupd_b_us,overhead_a_pct,overhead_b_pct"
    );
    for r in &rows {
        println!(
            "CSV:{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            r.batch,
            r.ctx,
            r.ctx_pad,
            r.block,
            r.frag_pct,
            r.contig_sdpa_us(),
            r.gather_a_only_us(),
            r.gather_a_sdpa_us(),
            r.gather_b_only_us(),
            r.gather_b_sdpa_us(),
            r.sliceupd_a_us(),
            r.sliceupd_b_us(),
            r.overhead_a_pct(),
            r.overhead_b_pct(),
        );
    }
}
