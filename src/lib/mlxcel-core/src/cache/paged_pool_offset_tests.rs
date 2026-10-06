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

//! Pool slabs past 2^32 elements (issue #2153).
//!
//! The fused paged-attention kernels address one layer's pool slab per side.
//! Their per-token pool base, `(row * page + slot) * Hkv * D + kv_head * D`,
//! was 32-bit in all six bodies (v1 decode and v2 partial, each for Metal,
//! CUDA and HIP), so a slab holding more than 2^32 elements silently read a
//! different row of the same buffer. The server sizes that slab as
//! `ceil(ctx / 32) * batch` blocks, so long contexts at a high `--parallel`
//! reach it (Llama-3.1-8B: more than 131,072 blocks of `[32, 8, 128]`).
//!
//! The fast tests read the kernel sources and pin the widening in every body,
//! on any host, because no single host can run all three backends, and check
//! that the merge launcher refuses sizes its 32-bit bodies cannot index.
//! The `#[ignore]`d one needs about 17 GiB of device memory: it builds a real
//! slab one block past the wrap, writes K/V into that block only, and checks
//! the v1 and v2 kernels against attention computed on the host from the same
//! values. On the 32-bit bodies the block's address wraps to row 0, which is
//! all zeros, so both kernels return zeros and the test fails.

use cxx::UniquePtr;

use crate::autotune::Source;
use crate::cache::PagedCsrView;
use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::paged_v2::{PagedDecodeGeometry, PagedDecodePlan, V2Context};

const PAGED_V1_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../mlx-cpp/turbo/paged_attention.cpp"
));
const PAGED_V2_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../mlx-cpp/turbo/paged_attention_v2.cpp"
));
const PAGED_HIP_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../mlx-cpp/turbo/paged_attention_hip.h"
));

/// Every `<type> base = <expr>;` declaration in `src`, as `(type, expr)`.
fn base_declarations(src: &str) -> Vec<(String, String)> {
    src.lines()
        .filter_map(|line| {
            let (lhs, rhs) = line.split_once('=')?;
            let mut words = lhs.split_whitespace();
            let ty = words.next()?;
            (words.next()? == "base" && words.next().is_none())
                .then(|| (ty.to_string(), rhs.trim().to_string()))
        })
        .collect()
}

#[test]
fn every_fused_body_computes_the_pool_base_in_64_bits() {
    const V1: &str = "block_size + slot";
    const V2: &str = "page_size + entry";
    // Each file holds two bodies, in this order.
    let files = [
        (
            "paged_attention.cpp",
            PAGED_V1_SRC,
            [("ulong", V1), ("uint64_t", V1)],
        ),
        (
            "paged_attention_v2.cpp",
            PAGED_V2_SRC,
            [("ulong", V2), ("uint64_t", V2)],
        ),
        (
            "paged_attention_hip.h",
            PAGED_HIP_SRC,
            [("uint64_t", V1), ("uint64_t", V2)],
        ),
    ];
    for (name, src, bodies) in files {
        let decls = base_declarations(src);
        assert_eq!(
            decls.len(),
            2,
            "{name}: expected two pool-base declarations, found {decls:?}"
        );
        for ((ty, expr), (want_ty, row_term)) in decls.iter().zip(bodies) {
            // `row` is widened before the multiply: `(T)(row * page)` would
            // still wrap in 32 bits.
            let want =
                format!("(({want_ty})row * {row_term}) * ({want_ty})stride_kv + kv_head * dim;");
            assert_eq!(
                (ty.as_str(), expr.as_str()),
                (want_ty, want.as_str()),
                "{name}: the pool base must be 64-bit"
            );
        }
    }
}

/// The merge kernel indexes `v_in` and its output in 32 bits, so its launcher
/// refuses either past `UINT32_MAX` elements with an `Err` instead of wrapping.
/// The inputs are unevaluated broadcasts, so nothing near that size is
/// allocated: the refusal reads only shapes and comes before any launch.
#[test]
fn merge_refuses_inputs_or_outputs_past_u32_elements() {
    let merge = |n: i32, h: i32, d: i32, m: i32| {
        let scalar = ffi::from_slice_f32(&[0.0], &[1]);
        let v_in = ffi::broadcast_to(&scalar, &[n, h, d]);
        let lse_in = ffi::broadcast_to(&scalar, &[n, h]);
        let zero = ffi::from_slice_i32(&[0], &[1]);
        let o_indptr = ffi::broadcast_to(&zero, &[m + 1]);
        let (mut v_out, mut lse_out) = (UniquePtr::null(), UniquePtr::null());
        ffi::paged_attention_merge_states(&v_in, &lse_in, &o_indptr, &mut v_out, &mut lse_out)
            .map_err(|e| e.what().to_string())
    };
    // v_in: 4,194,305 * 8 * 128 = 2^32 + 1024 elements.
    let err = merge(4_194_305, 8, 128, 1).expect_err("an oversized v_in is refused");
    assert!(
        err.contains("paged_attention_merge_states") && err.contains("UINT32_MAX"),
        "{err}"
    );
    // Output: 4,194,305 merged rows of [8, 128] from a one-row input.
    let err = merge(1, 8, 128, 4_194_305).expect_err("an oversized output is refused");
    assert!(err.contains("UINT32_MAX"), "{err}");
}

// ── the slab past 2^32 elements ─────────────────────────────────────────────

const PAGE: i32 = 32;
const HKV: i32 = 8;
const HQ: i32 = 32;
const DIM: i32 = 128;
/// 131,072 blocks of `[32, 8, 128]` is exactly 2^32 elements; one more block
/// puts every element of the last block past the wrap.
const BLOCKS: i32 = 131_073;
const LAST: i32 = BLOCKS - 1;

fn host_values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn to_vec_f32(arr: &MlxArray) -> Vec<f32> {
    let as_f32 = ffi::astype(arr, dtype::FLOAT32);
    ffi::eval(&as_f32);
    ffi::array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .collect()
}

/// An f16 pool `[BLOCKS, PAGE, HKV, DIM]` that is zero everywhere except the
/// last block, which holds `block` (`[1, PAGE, HKV, DIM]` f16).
///
/// Built from two zero halves rather than one: each `concatenate` input is
/// copied by one launch, and on ROCm a strided copy of 2^32 elements asks for
/// a 2^32-thread grid, which HIP refuses (`hipErrorInvalidConfiguration`).
/// That is a limit of MLX's copy (#2184), not of the kernels under test, so the test
/// stays clear of it.
fn pool_with_last_block(block: &MlxArray) -> UniquePtr<MlxArray> {
    let half = ffi::zeros(&[LAST / 2, PAGE, HKV, DIM], dtype::FLOAT16);
    let parts = [
        &*half as *const MlxArray,
        &*half as *const MlxArray,
        block as *const MlxArray,
    ];
    // SAFETY: every pointer comes from a live reference that outlives the call.
    let pool = unsafe { ffi::concatenate(&parts, 0) };
    ffi::eval(&pool);
    pool
}

/// Host attention for one query token: `q` is `[HQ * DIM]`, `k` and `v` are
/// `[PAGE][HKV][DIM]`. Returns `[HQ * DIM]`.
fn host_attention(q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
    let (hkv, d, n_rep) = (HKV as usize, DIM as usize, (HQ / HKV) as usize);
    let mut out = vec![0.0f32; q.len()];
    for h in 0..HQ as usize {
        let kvh = h / n_rep;
        let qh = &q[h * d..(h + 1) * d];
        let scores: Vec<f64> = (0..PAGE as usize)
            .map(|t| {
                let kt = &k[(t * hkv + kvh) * d..(t * hkv + kvh + 1) * d];
                qh.iter()
                    .zip(kt)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>()
                    * f64::from(scale)
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        for (i, o) in out[h * d..(h + 1) * d].iter_mut().enumerate() {
            let acc: f64 = (0..PAGE as usize)
                .map(|t| weights[t] * f64::from(v[(t * hkv + kvh) * d + i]))
                .sum();
            *o = (acc / total) as f32;
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "allocates two 8 GiB f16 pools; run with --ignored on a device with ~20 GiB free"]
fn paged_pool_past_u32_elements_matches_gather() {
    crate::test_support::kernel_ports::require_paged_attention_port!();
    let block_elems = (PAGE * HKV * DIM) as usize;
    let block_shape = [1, PAGE, HKV, DIM];

    // Round the host values through f16 so the reference sees exactly what the
    // pool stores.
    let k_block = ffi::astype(
        &ffi::from_slice_f32(&host_values(0x2153, block_elems), &block_shape),
        dtype::FLOAT16,
    );
    let v_block = ffi::astype(
        &ffi::from_slice_f32(&host_values(0x5132, block_elems), &block_shape),
        dtype::FLOAT16,
    );
    let (k_host, v_host) = (to_vec_f32(&k_block), to_vec_f32(&v_block));
    let q_host = host_values(0x0f16, (HQ * DIM) as usize);
    let q = ffi::from_slice_f32(&q_host, &[1, HQ, 1, DIM]);
    let scale = 1.0 / (DIM as f32).sqrt();
    let reference = host_attention(&q_host, &k_host, &v_host, scale);

    let k_pool = pool_with_last_block(&k_block);
    drop(k_block);
    let v_pool = pool_with_last_block(&v_block);
    drop(v_block);

    // The pool really holds the block past the wrap and nothing at row 0, which
    // is where a 32-bit base lands (131,072 * 32 * 1024 = 2^32 wraps to 0).
    let block_of = |pool: &MlxArray, row: i32| {
        to_vec_f32(&ffi::slice(
            pool,
            &[row, 0, 0, 0],
            &[row + 1, PAGE, HKV, DIM],
        ))
    };
    assert_eq!(block_of(&k_pool, LAST), k_host, "K's last block reads back");
    assert_eq!(block_of(&v_pool, LAST), v_host, "V's last block reads back");
    assert!(
        block_of(&k_pool, 0).iter().all(|x| *x == 0.0),
        "K row 0 is zero"
    );
    assert!(
        block_of(&v_pool, 0).iter().all(|x| *x == 0.0),
        "V row 0 is zero"
    );

    // v1: one sequence whose only block is the last pool row.
    let rows = ffi::from_slice_i32(&[LAST], &[1]);
    let row_offsets = ffi::from_slice_i32(&[0, 1], &[2]);
    let logical_starts = ffi::from_slice_i32(&[0], &[1]);
    let visible_lens = ffi::from_slice_i32(&[PAGE], &[1]);
    let v1 = ffi::paged_attention_decode(
        &q,
        &k_pool,
        &v_pool,
        &rows,
        &row_offsets,
        &logical_starts,
        &visible_lens,
        scale,
        0,
    )
    .expect("v1 launches");
    let v1_diff = max_abs_diff(&to_vec_f32(&v1), &reference);

    // v2: the same sequence as a one-page CSR view, one chunk, no merge.
    let view = PagedCsrView {
        page_size: PAGE,
        indices: vec![LAST],
        indptr: vec![0, 1],
        last_page_len: vec![PAGE],
        first_page_offset: vec![0],
        seq_lens: vec![PAGE],
        rope_offsets: vec![PAGE],
    };
    let geometry = PagedDecodeGeometry {
        q_heads: HQ,
        kv_heads: HKV,
        head_dim: DIM,
        page_size: PAGE,
    };
    let plan = PagedDecodePlan::with_chunk_size(geometry, &[1], 1, 1, Source::Default);
    let ctx = V2Context::build(&q, &k_pool, &v_pool, &view, geometry, scale).expect("view");
    let v2 = ctx.launch(&plan).expect("v2 launches");
    let v2_diff = max_abs_diff(&to_vec_f32(&v2), &reference);

    assert!(
        v1_diff < 1e-3 && v2_diff < 1e-3,
        "a pool row past 2^32 elements must be read from its own address: \
         max |v1 - ref| = {v1_diff}, max |v2 - ref| = {v2_diff}"
    );
}
