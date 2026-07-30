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

//! Correctness matrix for paged-attention decode v2 (issue #898).
//!
//! Compares the v2 kernels elementwise against the gather-then-SDPA reference
//! (ADR 0001 strategy A, the same reference v1 is validated against) over a
//! sweep of head dimensions, GQA ratios, page sizes, contexts, and batch sizes,
//! with a randomized partial last page per request.
//!
//! ## What it drives, and what it deliberately does not
//!
//! The harness builds the pool tensors and the CSR page table directly and
//! calls [`mlxcel_core::paged_v2::run_decode_v2`], rather than going through
//! `PagedBlockPool`. That is not a shortcut around the pool: the pool allocates
//! physical rows in 32-block slabs, and both fused kernels read one contiguous
//! buffer per side, so any context past ~1024 tokens at page size 32 is
//! multi-slab and *declined* by v1 and v2 alike. A pool-driven matrix could
//! therefore only test the short-context corner. The CSR builder itself is
//! covered by `cache::paged_csr` unit tests, and the pool entry point by
//! `paged_v2::launch` unit tests; this harness covers the kernels at scale.
//!
//! Physical pages are assigned in reverse order out of a pool with 2x slack, so
//! the page table is genuinely scattered and a kernel that ignored `indices`
//! would fail immediately rather than accidentally pass on a contiguous layout.
//!
//! ## The committed default is a subset
//!
//! The issue's full matrix is head_dim {64, 128} x GQA {1, 4, 8} x block
//! {16, 32, 64} x context {512, 4096, 16384, 32768} x batch {1, 4, 8}: 216
//! configurations, several of which allocate gigabytes. The default sweep here
//! is the 24-configuration subset
//!
//!   head_dim {64, 128} x GQA {1, 4, 8} x block {32} x context {512, 4096}
//!   x batch {1, 4}
//!
//! which runs in seconds. `--full` expands to the complete matrix; the axis
//! flags override any subset in between. Always state which sweep produced a
//! recorded number.
//!
//! Run:
//!
//! ```text
//! cargo run --release --features metal,accelerate \
//!     --example paged_decode_v2_correctness
//!
//! # The full 216-configuration matrix (allocates up to several GB).
//! cargo run --release --features metal,accelerate \
//!     --example paged_decode_v2_correctness -- --full
//!
//! # One axis at a time.
//! cargo run --release --features metal,accelerate \
//!     --example paged_decode_v2_correctness -- \
//!     --contexts 16384,32768 --batches 1 --blocks 32
//! ```

use clap::Parser;
use mlxcel_core::cache::PagedCsrView;
use mlxcel_core::paged_v2::{PagedDecodeGeometry, PagedDecodePlan, device_target_ctas};
use mlxcel_core::{
    MlxArray, UniquePtr, astype, eval, fast_scaled_dot_product_attention, from_slice_f32, reshape,
    slice, synchronize_default, take, transpose_axes,
};

const F16: i32 = mlxcel_core::dtype::FLOAT16;
const F32: i32 = mlxcel_core::dtype::FLOAT32;

#[derive(Parser, Debug)]
#[command(name = "paged_decode_v2_correctness")]
struct Args {
    /// Query heads. The GQA ratio divides this to give the KV head count.
    #[arg(long, default_value = "32")]
    q_heads: i32,

    /// Comma-separated head dimensions.
    #[arg(long, default_value = "64,128")]
    head_dims: String,

    /// Comma-separated GQA ratios (`q_heads / kv_heads`).
    #[arg(long, default_value = "1,4,8")]
    gqa: String,

    /// Comma-separated page (block) sizes.
    #[arg(long, default_value = "32")]
    blocks: String,

    /// Comma-separated context lengths.
    #[arg(long, default_value = "512,4096")]
    contexts: String,

    /// Comma-separated batch sizes.
    #[arg(long, default_value = "1,4")]
    batches: String,

    /// Expand to the full matrix from issue #898 (216 configurations, several
    /// of them multi-gigabyte). Overrides every axis flag.
    #[arg(long)]
    full: bool,

    /// Relative-error ceiling. The issue's suggested f16-KV tolerance is 2e-2.
    #[arg(long, default_value = "2e-2")]
    tolerance: f32,

    /// Also force `pages_per_chunk = 1` for every configuration, which puts
    /// every request through the merge kernel with a maximally ragged grouping.
    #[arg(long)]
    force_merge: bool,

    /// Seed for the synthetic K/V/Q values.
    #[arg(long, default_value = "20260898")]
    seed: u64,
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid list value: {t:?}"))
        })
        .collect()
}

/// Deterministic pseudo-random values in [-0.5, 0.5). Distinct values keep the
/// softmax non-degenerate, and the fixed seed makes any failure reproducible.
fn pseudo_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1) | 1;
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
    astype(&from_slice_f32(vals, shape), F16)
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = astype(a, F32);
    eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// One measured configuration.
struct Row {
    head_dim: usize,
    gqa: usize,
    block: usize,
    ctx: usize,
    batch: usize,
    pages_per_chunk: i32,
    num_chunks: usize,
    total_ctas: usize,
    needs_merge: bool,
    max_rel: f32,
    rms_rel: f32,
    pass: bool,
}

/// Max and RMS deviation, both relative to the reference's own scale. A plain
/// absolute bound would be meaningless for an output whose magnitude depends on
/// the value distribution.
fn deviations(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "output length mismatch");
    let peak = want.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
    let mut max_abs = 0.0f32;
    let mut sq_err = 0.0f64;
    let mut sq_ref = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        sq_err += f64::from(d) * f64::from(d);
        sq_ref += f64::from(*w) * f64::from(*w);
    }
    let n = got.len().max(1) as f64;
    let rms_err = (sq_err / n).sqrt();
    let rms_ref = (sq_ref / n).sqrt().max(1e-12);
    (max_abs / peak, (rms_err / rms_ref) as f32)
}

/// Gather-then-SDPA reference for one request: take its pages out of the pool,
/// restore token order, slice the visible window, and run fused SDPA.
fn reference_for_request(
    pool_k: &MlxArray,
    pool_v: &MlxArray,
    q_row: &MlxArray,
    rows: &[i32],
    first_offset: i32,
    seq_len: i32,
    block: i32,
    hkv: i32,
    dim: i32,
    scale: f32,
) -> UniquePtr<MlxArray> {
    let ids = from_slice_i32_local(rows);
    let span = rows.len() as i32 * block;
    let gather = |pool: &MlxArray| -> UniquePtr<MlxArray> {
        let g = take(pool, &ids, 0); // [pages, block, Hkv, D]
        let g = reshape(&g, &[1, span, hkv, dim]);
        let g = transpose_axes(&g, &[0, 2, 1, 3]); // [1, Hkv, span, D]
        slice(
            &g,
            &[0, 0, first_offset, 0],
            &[1, hkv, first_offset + seq_len, dim],
        )
    };
    let k = gather(pool_k);
    let v = gather(pool_v);
    // SAFETY: thin FFI wrapper over the MLX fast SDPA kernel; the null mask
    // pointer is the documented "no mask" sentinel and q/k/v outlive the call.
    unsafe { fast_scaled_dot_product_attention(q_row, &k, &v, scale, std::ptr::null()) }
}

fn from_slice_i32_local(v: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(v, &[v.len() as i32])
}

#[allow(clippy::too_many_arguments)]
fn run_config(
    q_heads: i32,
    head_dim: usize,
    gqa: usize,
    block: usize,
    ctx: usize,
    batch: usize,
    tolerance: f32,
    force_merge: bool,
    seed: u64,
) -> Row {
    let dim = head_dim as i32;
    let hq = q_heads;
    let hkv = hq / gqa as i32;
    let bs = block as i32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Per-request visible length: `ctx` for request 0, then a randomized
    // partial tail so the last page is only partly valid and every request has
    // a different page count.
    let jitter = pseudo_f32(seed ^ 0xC0FFEE, batch);
    let seq_lens: Vec<usize> = (0..batch)
        .map(|r| {
            if r == 0 {
                ctx
            } else {
                // jitter is in [-0.5, 0.5), so this lands in [0.55, 1.0] of ctx.
                let frac = (0.75 + jitter[r] * 0.5).clamp(0.55, 1.0);
                ((ctx as f32 * frac) as usize).max(1)
            }
        })
        .collect();
    // Half the requests start mid-page, exercising `first_page_offset`.
    let first_offsets: Vec<usize> = (0..batch)
        .map(|r| if r % 2 == 1 { (r * 7) % block } else { 0 })
        .collect();

    let page_counts: Vec<usize> = seq_lens
        .iter()
        .zip(&first_offsets)
        .map(|(&len, &off)| (len + off).div_ceil(block))
        .collect();
    let total_pages: usize = page_counts.iter().sum();
    // 2x slack, and reverse assignment, so physical rows are scattered.
    let pool_blocks = (total_pages * 2).max(1);

    let pool_elems = pool_blocks * block * hkv as usize * head_dim;
    let pool_k = f16(
        &pseudo_f32(seed, pool_elems),
        &[pool_blocks as i32, bs, hkv, dim],
    );
    let pool_v = f16(
        &pseudo_f32(seed ^ 0x5EED, pool_elems),
        &[pool_blocks as i32, bs, hkv, dim],
    );
    let q_f16 = f16(
        &pseudo_f32(seed ^ 0x51D, batch * hq as usize * head_dim),
        &[batch as i32, hq, 1, dim],
    );
    let q_f32 = astype(&q_f16, F32);
    eval(&pool_k);
    eval(&pool_v);
    eval(&q_f16);
    synchronize_default();

    // CSR page table with reverse-order physical rows.
    let mut indices: Vec<i32> = Vec::with_capacity(total_pages);
    let mut indptr: Vec<i32> = vec![0];
    let mut last_page_len: Vec<i32> = Vec::with_capacity(batch);
    let mut first_page_offset: Vec<i32> = Vec::with_capacity(batch);
    let mut lens_i32: Vec<i32> = Vec::with_capacity(batch);
    let mut assigned = 0usize;
    for r in 0..batch {
        for _ in 0..page_counts[r] {
            indices.push((pool_blocks - 1 - assigned) as i32);
            assigned += 1;
        }
        indptr.push(indices.len() as i32);
        let total = first_offsets[r] + seq_lens[r];
        let lpl = total - (page_counts[r] - 1) * block;
        last_page_len.push(lpl as i32);
        first_page_offset.push(first_offsets[r] as i32);
        lens_i32.push(seq_lens[r] as i32);
    }
    let view = PagedCsrView {
        page_size: bs,
        indices: indices.clone(),
        indptr: indptr.clone(),
        last_page_len,
        first_page_offset,
        seq_lens: lens_i32,
        rope_offsets: seq_lens.iter().map(|&l| l as i32).collect(),
    };
    view.validate().expect("hand-built CSR view is consistent");

    // v2 output.
    let geometry = PagedDecodeGeometry {
        q_heads: hq,
        kv_heads: hkv,
        head_dim: dim,
        page_size: bs,
    };
    let target = device_target_ctas();
    let plan = if force_merge {
        PagedDecodePlan::with_chunk_size(
            geometry,
            &page_counts,
            1,
            target,
            mlxcel_core::autotune::Source::Default,
        )
    } else {
        PagedDecodePlan::heuristic(geometry, &page_counts, target)
    };
    plan.validate().expect("plan is well formed");
    let ctx_arrays =
        mlxcel_core::paged_v2::V2Context::build(&q_f32, &pool_k, &pool_v, &view, geometry, scale)
            .expect("v2 context builds");
    let got = to_vec_f32(&ctx_arrays.launch(&plan).expect("v2 launch"));

    // Reference: gather-then-SDPA, one request at a time.
    let mut want: Vec<f32> = Vec::with_capacity(batch * hq as usize * head_dim);
    for r in 0..batch {
        let q_row = slice(&q_f16, &[r as i32, 0, 0, 0], &[r as i32 + 1, hq, 1, dim]);
        let begin = indptr[r] as usize;
        let end = indptr[r + 1] as usize;
        let out = reference_for_request(
            &pool_k,
            &pool_v,
            &q_row,
            &indices[begin..end],
            first_offsets[r] as i32,
            seq_lens[r] as i32,
            bs,
            hkv,
            dim,
            scale,
        );
        want.extend(to_vec_f32(&out));
    }

    let (max_rel, rms_rel) = deviations(&got, &want);
    Row {
        head_dim,
        gqa,
        block,
        ctx,
        batch,
        pages_per_chunk: plan.pages_per_chunk,
        num_chunks: plan.num_chunks,
        total_ctas: plan.total_ctas,
        needs_merge: plan.needs_merge,
        max_rel,
        rms_rel,
        pass: max_rel <= tolerance,
    }
}

fn main() {
    let args = Args::parse();

    let (head_dims, gqas, blocks, contexts, batches) = if args.full {
        (
            vec![64usize, 128],
            vec![1usize, 4, 8],
            vec![16usize, 32, 64],
            vec![512usize, 4096, 16384, 32768],
            vec![1usize, 4, 8],
        )
    } else {
        (
            parse_list(&args.head_dims),
            parse_list(&args.gqa),
            parse_list(&args.blocks),
            parse_list(&args.contexts),
            parse_list(&args.batches),
        )
    };

    println!("=== paged decode v2 correctness matrix (issue #898) ===");
    println!(
        "q_heads={} tolerance={:.1e} force_merge={} seed={} target_ctas={}",
        args.q_heads,
        args.tolerance,
        args.force_merge,
        args.seed,
        device_target_ctas()
    );
    println!(
        "sweep: head_dim {head_dims:?} x gqa {gqas:?} x block {blocks:?} x ctx {contexts:?} x batch {batches:?} = {} configs",
        head_dims.len() * gqas.len() * blocks.len() * contexts.len() * batches.len()
    );
    println!("reference: gather-then-SDPA over the same pool (ADR 0001 strategy A)");
    println!();

    let mut rows: Vec<Row> = Vec::new();
    for &head_dim in &head_dims {
        for &gqa in &gqas {
            if args.q_heads % gqa as i32 != 0 {
                println!("skip: gqa {gqa} does not divide q_heads {}", args.q_heads);
                continue;
            }
            for &block in &blocks {
                for &ctx in &contexts {
                    for &batch in &batches {
                        let row = run_config(
                            args.q_heads,
                            head_dim,
                            gqa,
                            block,
                            ctx,
                            batch,
                            args.tolerance,
                            args.force_merge,
                            args.seed,
                        );
                        println!(
                            "  d={:<4} gqa={:<2} blk={:<3} ctx={:<6} B={:<2} ppc={:<5} chunks={:<6} ctas={:<7} merge={:<5} max_rel={:.3e} rms_rel={:.3e} {}",
                            row.head_dim,
                            row.gqa,
                            row.block,
                            row.ctx,
                            row.batch,
                            row.pages_per_chunk,
                            row.num_chunks,
                            row.total_ctas,
                            row.needs_merge,
                            row.max_rel,
                            row.rms_rel,
                            if row.pass { "PASS" } else { "FAIL" }
                        );
                        rows.push(row);
                    }
                }
            }
        }
    }

    println!();
    println!(
        "CSV:head_dim,gqa,block,ctx,batch,pages_per_chunk,num_chunks,total_ctas,needs_merge,max_rel,rms_rel,pass"
    );
    for r in &rows {
        println!(
            "CSV:{},{},{},{},{},{},{},{},{},{:.6e},{:.6e},{}",
            r.head_dim,
            r.gqa,
            r.block,
            r.ctx,
            r.batch,
            r.pages_per_chunk,
            r.num_chunks,
            r.total_ctas,
            r.needs_merge,
            r.max_rel,
            r.rms_rel,
            r.pass
        );
    }

    let failed = rows.iter().filter(|r| !r.pass).count();
    let worst_max = rows.iter().fold(0.0f32, |a, r| a.max(r.max_rel));
    let worst_rms = rows.iter().fold(0.0f32, |a, r| a.max(r.rms_rel));
    println!();
    println!(
        "{} configurations, {failed} failing; worst max_rel {worst_max:.3e}, worst rms_rel {worst_rms:.3e}, tolerance {:.1e}",
        rows.len(),
        args.tolerance
    );
    if failed > 0 {
        std::process::exit(1);
    }
}
