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

//! Per-op block-vs-chain byte parity on Metal, model-free (issue #1165).
//!
//! `tests/qwen38_mtp_chain_parity.rs` shows *that* a `T = K` verify block can
//! disagree with the `K`-step single-token decode chain at temperature 0. It
//! cannot show *which primitive* moved, because it drives a whole 27B model.
//! This file drives the primitives directly at the published Qwen 3.8 27B
//! shapes, loads no checkpoint, and finishes in about a second.
//!
//! For each op the same input is pushed through two arms:
//!
//! - **block** — one call on a `[1, T, D]` input.
//! - **chain** — `T` calls on `[1, 1, D]` slices of that same input.
//!
//! and position `t` of the block output is compared byte-for-byte against
//! call `t` of the chain. Any difference is a `T = 1` versus `T >= 2` kernel
//! dispatch difference inside MLX, since the arithmetic is identical by
//! construction.
//!
//! ## What is asserted and what is only reported
//!
//! `fast_rms_norm`, `fast_rope` and causal SDPA are **asserted** equal: the
//! speculative verify path depends on all three being position-independent,
//! and nothing else in the tree pins that. `matmul` and `quantized_matmul`
//! are **reported, not asserted**: whether MLX gives bit-equal results across
//! its matrix-vector and matrix-matrix kernels is a property this repository
//! does not control, and a red suite would say nothing a reader can act on.
//!
//! ## Invocation
//!
//! ```bash
//! cargo test --test metal_block_vs_chain_op_parity --release --features metal,accelerate -- --nocapture
//! ```

use mlxcel::initialize_runtime;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Qwen 3.8 27B hidden size.
const HIDDEN: i32 = 5120;
/// Qwen 3.8 27B MLP intermediate size (gate/up projection output).
const MLP_OUT: i32 = 17408;
/// Published `mlx-community` 4-bit affine quantization parameters.
const GROUP: i32 = 64;
const BITS: i32 = 4;
/// The drafter's published `block_size`, i.e. the verify width in production.
const BLOCK_T: i32 = 3;
/// Block widths swept to locate where equality breaks. Spans MLX's
/// `get_qmv_batch_limit` range (6 to 32 by architecture and operand size).
const SWEEP: &[i32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 18, 24, 32];

/// Attention shape: 40 query heads, 8 KV heads, head_dim 128.
const N_HEADS: i32 = 40;
const N_KV_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
/// Cached prefix length the verify block attends over.
const PREFIX: i32 = 64;

fn randn(shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    unsafe { mlxcel_core::random_normal(shape, dt, std::ptr::null()) }
}

/// Position `t` of a `[1, T, ...]` array, as a `[1, 1, ...]` slice.
fn position(arr: &MlxArray, t: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(arr);
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.clone();
    starts[1] = t;
    stops[1] = t + 1;
    mlxcel_core::slice(arr, &starts, &stops)
}

/// Position `t` of a `[1, H, T, D]` array, as a `[1, H, 1, D]` slice.
fn position_axis2(arr: &MlxArray, t: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(arr);
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.clone();
    starts[2] = t;
    stops[2] = t + 1;
    mlxcel_core::slice(arr, &starts, &stops)
}

fn bytes(arr: &MlxArray) -> Vec<u8> {
    mlxcel_core::eval(arr);
    mlxcel_core::array_to_raw_bytes(arr)
}

/// FNV-1a 64, so a moved digest proves an arm actually changed.
fn digest(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Differing byte count per position, block arm vs chain arm.
struct Parity {
    per_position: Vec<usize>,
    total_bytes: usize,
    block_digest: u64,
    chain_digest: u64,
}

impl Parity {
    fn equal(&self) -> bool {
        self.per_position.iter().all(|d| *d == 0)
    }

    fn verdict(&self) -> String {
        if self.equal() {
            "equal".to_string()
        } else {
            format!(
                "DIVERGES ({} of {} bytes per position)",
                self.per_position
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                self.total_bytes
            )
        }
    }
}

/// Run both arms and compare. `block` gets the whole `[1, T, ...]` input;
/// `single` gets one position at a time.
fn compare<B, S>(
    t: i32,
    block: B,
    single: S,
    slice_position: fn(&MlxArray, i32) -> UniquePtr<MlxArray>,
) -> Parity
where
    B: FnOnce() -> UniquePtr<MlxArray>,
    S: Fn(i32) -> UniquePtr<MlxArray>,
{
    let block_out = block();
    mlxcel_core::eval(&block_out);

    let mut per_position = Vec::with_capacity(t as usize);
    let mut total_bytes = 0usize;
    let mut block_all: Vec<u8> = Vec::new();
    let mut chain_all: Vec<u8> = Vec::new();

    for i in 0..t {
        let block_slice = slice_position(&block_out, i);
        let block_bytes = bytes(&block_slice);
        let chain_out = single(i);
        let chain_bytes = bytes(&chain_out);
        assert_eq!(
            block_bytes.len(),
            chain_bytes.len(),
            "arm shapes disagree at position {i}"
        );
        total_bytes = block_bytes.len();
        per_position.push(
            block_bytes
                .iter()
                .zip(chain_bytes.iter())
                .filter(|(a, b)| a != b)
                .count(),
        );
        block_all.extend_from_slice(&block_bytes);
        chain_all.extend_from_slice(&chain_bytes);
    }

    Parity {
        per_position,
        total_bytes,
        block_digest: digest(&block_all),
        chain_digest: digest(&chain_all),
    }
}

struct Quantized {
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: Option<UniquePtr<MlxArray>>,
    group_size: i32,
    mode: &'static str,
}

fn quantize_mode(out: i32, in_dim: i32, group_size: i32, mode: &'static str) -> Quantized {
    let dense = randn(&[out, in_dim], dtype::FLOAT16);
    let q = mlxcel_core::quantize_weights_with_mode(&dense, group_size, BITS, mode);
    Quantized {
        w: mlxcel_core::quantized_weights_w(&q),
        scales: mlxcel_core::quantized_weights_scales(&q),
        biases: mlxcel_core::quantized_weights_has_biases(&q)
            .then(|| mlxcel_core::quantized_weights_biases(&q)),
        group_size,
        mode,
    }
}

/// The published `mlx-community` quantization: affine, 4-bit, group 64.
fn quantize(out: i32, in_dim: i32) -> Quantized {
    quantize_mode(out, in_dim, GROUP, "affine")
}

fn quantized_forward(q: &Quantized, x: &MlxArray) -> UniquePtr<MlxArray> {
    let biases = match &q.biases {
        Some(b) => b.as_ref().unwrap() as *const MlxArray,
        None => std::ptr::null(),
    };
    unsafe {
        mlxcel_core::quantized_matmul(x, &q.w, &q.scales, biases, true, q.group_size, BITS, q.mode)
    }
}

fn skip_unless_metal() -> bool {
    if mlxcel_core::metal_is_available() {
        return false;
    }
    eprintln!("skipping: this diagnostic is about Metal kernel dispatch");
    true
}

/// The headline table: every op at the production verify width.
#[test]
fn per_op_block_vs_chain_parity_at_block_three() {
    if skip_unless_metal() {
        return;
    }
    let _runtime = initialize_runtime();
    mlxcel_core::random_seed(1165);

    let t = BLOCK_T;
    let x_f16 = randn(&[1, t, HIDDEN], dtype::FLOAT16);
    let x_f32 = mlxcel_core::astype(&x_f16, dtype::FLOAT32);

    println!("\n=== block (T = {t}) vs single-token chain, Qwen3.8-27B shapes ===\n");

    let mlp = quantize(MLP_OUT, HIDDEN);
    let p = compare(
        t,
        || quantized_forward(&mlp, &x_f16),
        |i| quantized_forward(&mlp, &position(&x_f16, i)),
        position,
    );
    println!(
        "quantized_matmul {HIDDEN} -> {MLP_OUT} (affine {BITS}-bit, group {GROUP}) : {}",
        p.verdict()
    );

    let o_proj = quantize(HIDDEN, HIDDEN);
    let p = compare(
        t,
        || quantized_forward(&o_proj, &x_f16),
        |i| quantized_forward(&o_proj, &position(&x_f16, i)),
        position,
    );
    let mxfp4 = quantize_mode(HIDDEN, HIDDEN, 32, "mxfp4");
    let p_mxfp4 = compare(
        t,
        || quantized_forward(&mxfp4, &x_f16),
        |i| quantized_forward(&mxfp4, &position(&x_f16, i)),
        position,
    );

    println!(
        "quantized_matmul {HIDDEN} -> {HIDDEN} (o_proj shape)                     : {}",
        p.verdict()
    );
    // MLX takes `qmv_wide` for every non-affine mode on every GPU generation
    // (`use_qmv_wide` in mlx/backend/metal/quantized.cpp), so this row is the
    // control showing the split is a (mode, generation) property and not a
    // property of the chip alone.
    println!(
        "quantized_matmul {HIDDEN} -> {HIDDEN} (mxfp4, group 32)                  : {}",
        p_mxfp4.verdict()
    );

    let w_f16 = randn(&[HIDDEN, HIDDEN], dtype::FLOAT16);
    let p = compare(
        t,
        || mlxcel_core::matmul(&x_f16, &w_f16),
        |i| mlxcel_core::matmul(&position(&x_f16, i), &w_f16),
        position,
    );
    println!(
        "dense matmul f16 {HIDDEN} x {HIDDEN}                                   : {}",
        p.verdict()
    );

    let w_f32 = mlxcel_core::astype(&w_f16, dtype::FLOAT32);
    let p = compare(
        t,
        || mlxcel_core::matmul(&x_f32, &w_f32),
        |i| mlxcel_core::matmul(&position(&x_f32, i), &w_f32),
        position,
    );
    println!(
        "dense matmul f32 {HIDDEN} x {HIDDEN}                                   : {}",
        p.verdict()
    );
    println!(
        "    f32 digests: block {:#018x}  chain {:#018x}",
        p.block_digest, p.chain_digest
    );

    let norm_w = randn(&[HIDDEN], dtype::FLOAT16);
    let p = compare(
        t,
        || mlxcel_core::fast_rms_norm(&x_f16, &norm_w, 1e-6),
        |i| mlxcel_core::fast_rms_norm(&position(&x_f16, i), &norm_w, 1e-6),
        position,
    );
    println!(
        "fast_rms_norm                                                    : {}",
        p.verdict()
    );
    assert!(
        p.equal(),
        "fast_rms_norm must be position-independent; the verify path assumes it"
    );

    let rope_x = randn(&[1, N_HEADS, t, HEAD_DIM], dtype::FLOAT16);
    let p = compare(
        t,
        || mlxcel_core::fast_rope(&rope_x, HEAD_DIM, false, 10000.0, 1.0, 0),
        |i| {
            mlxcel_core::fast_rope(
                &position_axis2(&rope_x, i),
                HEAD_DIM,
                false,
                10000.0,
                1.0,
                i,
            )
        },
        position_axis2,
    );
    println!(
        "fast_rope (matching per-position offsets)                         : {}",
        p.verdict()
    );
    assert!(
        p.equal(),
        "fast_rope must agree between a T-wide call and per-position calls at matching offsets"
    );

    let q = randn(&[1, N_HEADS, t, HEAD_DIM], dtype::FLOAT16);
    let k = randn(&[1, N_KV_HEADS, PREFIX + t, HEAD_DIM], dtype::FLOAT16);
    let v = randn(&[1, N_KV_HEADS, PREFIX + t, HEAD_DIM], dtype::FLOAT16);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let p = compare(
        t,
        || mlxcel_core::fast_scaled_dot_product_attention_causal(&q, &k, &v, scale),
        |i| {
            // Position i of the block sees the prefix plus positions 0..=i.
            let visible = PREFIX + i + 1;
            let k_i = mlxcel_core::slice(&k, &[0, 0, 0, 0], &[1, N_KV_HEADS, visible, HEAD_DIM]);
            let v_i = mlxcel_core::slice(&v, &[0, 0, 0, 0], &[1, N_KV_HEADS, visible, HEAD_DIM]);
            mlxcel_core::fast_scaled_dot_product_attention_causal(
                &position_axis2(&q, i),
                &k_i,
                &v_i,
                scale,
            )
        },
        position_axis2,
    );
    println!(
        "SDPA, causal T = K block vs per-position                          : {}",
        p.verdict()
    );
    assert!(
        p.equal(),
        "causal SDPA must agree between a T-wide block and the per-position form"
    );
    println!();
}

/// Where does equality break? Sweeps the block width for the two matmul
/// flavors, reported only.
#[test]
fn matmul_block_width_sweep() {
    if skip_unless_metal() {
        return;
    }
    let _runtime = initialize_runtime();
    mlxcel_core::random_seed(1165);

    let o_proj = quantize(HIDDEN, HIDDEN);
    let mlp = quantize(MLP_OUT, HIDDEN);
    let w_f16 = randn(&[HIDDEN, HIDDEN], dtype::FLOAT16);

    println!("\n=== block width sweep, Qwen3.8-27B projection shapes ===\n");
    println!(
        "    T   q4 {HIDDEN}->{HIDDEN}   q4 {HIDDEN}->{MLP_OUT}   dense f16 {HIDDEN}x{HIDDEN}"
    );
    for &t in SWEEP {
        let x = randn(&[1, t, HIDDEN], dtype::FLOAT16);
        let q = compare(
            t,
            || quantized_forward(&o_proj, &x),
            |i| quantized_forward(&o_proj, &position(&x, i)),
            position,
        );
        let m = compare(
            t,
            || quantized_forward(&mlp, &x),
            |i| quantized_forward(&mlp, &position(&x, i)),
            position,
        );
        let d = compare(
            t,
            || mlxcel_core::matmul(&x, &w_f16),
            |i| mlxcel_core::matmul(&position(&x, i), &w_f16),
            position,
        );
        let cell = |p: &Parity| if p.equal() { "equal" } else { "DIVERGES" };
        println!("{t:5}   {:<16}   {:<17}   {}", cell(&q), cell(&m), cell(&d));
    }
    println!();
}
