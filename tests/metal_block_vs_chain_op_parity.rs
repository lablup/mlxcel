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
//! and Gemma 4 12B projection shapes, loads no checkpoint, and finishes in
//! about a second. Two families rather than one because the shipping gate
//! probes Qwen 3.5 (#1186) while the Gemma 4 arm still returns
//! unconditionally true (#1188), and the same MLX dispatch decides both.
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
/// The drafter's published `block_size`, i.e. the verify width in production.
const BLOCK_T: i32 = 3;
/// Block widths swept to locate where equality breaks. Spans MLX's
/// `get_qmv_batch_limit` range (6 to 32 by architecture and operand size),
/// and includes 18, which is that limit for Gemma 4's attention shapes.
const SWEEP: &[i32] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 24, 32,
];

/// One quantized projection to compare, as the model actually ships it.
///
/// `bits` and `group` are carried per shape rather than assumed, because
/// Gemma 4 12B ships its MLP at 8 bits and everything else at 4 (the
/// per-tensor overrides in its `config.json` `quantization` block), and
/// `mode` decides `use_qmv_wide` on every generation.
struct ProjShape {
    label: &'static str,
    in_dim: i32,
    out_dim: i32,
    bits: i32,
    group: i32,
    mode: &'static str,
}

/// Qwen 3.8 27B, the pairing #1165 shipped MTP against. Hidden 5120,
/// MLP 17408, affine 4-bit group 64 throughout.
const QWEN38_SHAPES: &[ProjShape] = &[
    ProjShape {
        label: "qwen3.8-27b gate/up   5120 -> 17408 (affine 4-bit g64)",
        in_dim: HIDDEN,
        out_dim: MLP_OUT,
        bits: 4,
        group: 64,
        mode: "affine",
    },
    ProjShape {
        label: "qwen3.8-27b o_proj    5120 -> 5120  (affine 4-bit g64)",
        in_dim: HIDDEN,
        out_dim: HIDDEN,
        bits: 4,
        group: 64,
        mode: "affine",
    },
    ProjShape {
        label: "qwen3.8-27b o_proj    5120 -> 5120  (mxfp4 g32)",
        in_dim: HIDDEN,
        out_dim: HIDDEN,
        bits: 4,
        group: 32,
        mode: "mxfp4",
    },
];

/// Gemma 4 12B (`gemma4_unified_text`): hidden 3840, MLP 15360, 16 query
/// heads and 8 KV heads at head_dim 256, vocab 262144.
///
/// The two attention rows and the two MLP rows sit in **different**
/// `get_qmv_batch_limit` buckets: 3840 and 4096 are both at or below 4096
/// (limit 18 on Apple GPU generation 13/14 architecture size `d`), while
/// 15360 is above it (limit 12). So this family's byte-identity cliff is
/// per-projection and the model-wide answer is the minimum over shapes,
/// which is the reason the shipping gate probes the model rather than
/// consulting a shape table (#1186).
///
/// `lm_head` (3840 -> 262144) is deliberately absent: it lands in the same
/// above-4096 bucket the MLP rows already cover, and synthesizing it would
/// cost a 2 GB dense f16 intermediate in a test that otherwise runs in
/// under a second.
const GEMMA4_SHAPES: &[ProjShape] = &[
    ProjShape {
        label: "gemma4-12b  q_proj    3840 -> 4096  (affine 4-bit g64)",
        in_dim: 3840,
        out_dim: 4096,
        bits: 4,
        group: 64,
        mode: "affine",
    },
    ProjShape {
        label: "gemma4-12b  kv_proj   3840 -> 2048  (affine 4-bit g64)",
        in_dim: 3840,
        out_dim: 2048,
        bits: 4,
        group: 64,
        mode: "affine",
    },
    ProjShape {
        label: "gemma4-12b  gate/up   3840 -> 15360 (affine 8-bit g64)",
        in_dim: 3840,
        out_dim: 15360,
        bits: 8,
        group: 64,
        mode: "affine",
    },
    ProjShape {
        label: "gemma4-12b  down      15360 -> 3840 (affine 8-bit g64)",
        in_dim: 15360,
        out_dim: 3840,
        bits: 8,
        group: 64,
        mode: "affine",
    },
];

/// Attention shape: 40 query heads, 8 KV heads, head_dim 128.
const N_HEADS: i32 = 40;
const N_KV_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
/// Cached prefix length the verify block attends over.
const PREFIX: i32 = 64;

/// Seed for the synthetic operands, overridable with
/// `MLXCEL_TEST_OP_PARITY_SEED`.
///
/// Not decoration. When a kernel pair differs by only a byte or two out of
/// ten thousand, whether *any* byte differs on a given draw is itself
/// draw-dependent, so a single seed can report `equal` for a shape that is
/// genuinely on a different kernel. Sweeping the seed is how that gets
/// separated from a real equality (M5 Max, 2026-08-17: the mxfp4 row read
/// `equal` on one harness and `DIVERGES` at 1 to 9 bytes on four seeds of
/// this one).
fn operand_seed() -> u64 {
    std::env::var("MLXCEL_TEST_OP_PARITY_SEED")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(1165)
}

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
    bits: i32,
    group_size: i32,
    mode: &'static str,
}

fn quantize_shape(shape: &ProjShape) -> Quantized {
    let dense = randn(&[shape.out_dim, shape.in_dim], dtype::FLOAT16);
    let q = mlxcel_core::quantize_weights_with_mode(&dense, shape.group, shape.bits, shape.mode);
    Quantized {
        w: mlxcel_core::quantized_weights_w(&q),
        scales: mlxcel_core::quantized_weights_scales(&q),
        biases: mlxcel_core::quantized_weights_has_biases(&q)
            .then(|| mlxcel_core::quantized_weights_biases(&q)),
        bits: shape.bits,
        group_size: shape.group,
        mode: shape.mode,
    }
}

fn quantized_forward(q: &Quantized, x: &MlxArray) -> UniquePtr<MlxArray> {
    let biases = match &q.biases {
        Some(b) => b.as_ref().unwrap() as *const MlxArray,
        None => std::ptr::null(),
    };
    unsafe {
        mlxcel_core::quantized_matmul(
            x,
            &q.w,
            &q.scales,
            biases,
            true,
            q.group_size,
            q.bits,
            q.mode,
        )
    }
}

/// Run every shape in `family` at width `t` and print one line each.
fn report_family(family: &[ProjShape], t: i32) {
    for shape in family {
        let x = randn(&[1, t, shape.in_dim], dtype::FLOAT16);
        let weights = quantize_shape(shape);
        let parity = compare(
            t,
            || quantized_forward(&weights, &x),
            |i| quantized_forward(&weights, &position(&x, i)),
            position,
        );
        println!("{:<52} : {}", shape.label, parity.verdict());
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
    mlxcel_core::random_seed(operand_seed());

    let t = BLOCK_T;
    let x_f16 = randn(&[1, t, HIDDEN], dtype::FLOAT16);
    let x_f32 = mlxcel_core::astype(&x_f16, dtype::FLOAT32);

    println!("\n=== block (T = {t}) vs single-token chain ===\n");

    // Qwen 3.5 / 3.8, the family the shipping gate probes today (#1186).
    // The mxfp4 row is the control that separates "this chip" from "this
    // mode": MLX takes `qmv_wide` for every non-affine mode on every GPU
    // generation (`use_qmv_wide` in mlx/backend/metal/quantized.cpp), so a
    // machine whose affine rows are equal should still diverge there.
    report_family(QWEN38_SHAPES, t);

    // Gemma 4, whose MTP gate arm still returns unconditionally true
    // (#1188). Same quantized projections, so the same mechanism should
    // apply; these rows are what makes that measurable on a given GPU
    // without loading a checkpoint.
    report_family(GEMMA4_SHAPES, t);

    let w_f16 = randn(&[HIDDEN, HIDDEN], dtype::FLOAT16);
    let p = compare(
        t,
        || mlxcel_core::matmul(&x_f16, &w_f16),
        |i| mlxcel_core::matmul(&position(&x_f16, i), &w_f16),
        position,
    );
    println!(
        "{:<52} : {}",
        format!("dense matmul f16     {HIDDEN} -> {HIDDEN}  (no quantization)"),
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
        "{:<52} : {}",
        format!("dense matmul f32     {HIDDEN} -> {HIDDEN}  (no quantization)"),
        p.verdict()
    );
    println!(
        // `0x` written out rather than via `{:#...}`: the `#` in a hex format
        // spec reads as a bare issue reference to `scripts/ci/check_cross_repo_refs.py`.
        "    f32 digests: block 0x{:016x}  chain 0x{:016x}",
        p.block_digest, p.chain_digest
    );

    let norm_w = randn(&[HIDDEN], dtype::FLOAT16);
    let p = compare(
        t,
        || mlxcel_core::fast_rms_norm(&x_f16, &norm_w, 1e-6),
        |i| mlxcel_core::fast_rms_norm(&position(&x_f16, i), &norm_w, 1e-6),
        position,
    );
    println!("{:<52} : {}", "fast_rms_norm", p.verdict());
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
        "{:<52} : {}",
        "fast_rope (matching per-position offsets)",
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
        "{:<52} : {}",
        "SDPA, causal T = K block vs per-position",
        p.verdict()
    );
    assert!(
        p.equal(),
        "causal SDPA must agree between a T-wide block and the per-position form"
    );
    println!();
}

/// Where does equality break? Sweeps the block width per projection shape.
///
/// The two Gemma 4 columns are the point of this table: 3840 -> 4096 and
/// 3840 -> 15360 sit in different `get_qmv_batch_limit` buckets (18 and 12
/// on Apple GPU generation 13/14, architecture size `d`), so on a machine
/// where affine stays on the `qmv` kernel they should stop agreeing at
/// different widths. A model's own cliff is the minimum over its shapes,
/// which is why the shipping gate measures the model instead of consulting
/// a table (#1186, #1188). Reported, never asserted.
#[test]
fn matmul_block_width_sweep() {
    if skip_unless_metal() {
        return;
    }
    let _runtime = initialize_runtime();
    mlxcel_core::random_seed(operand_seed());

    // One representative per distinct (operand bucket, bits) combination
    // across the two families, plus an unquantized control.
    let columns: &[(&str, &ProjShape)] = &[
        ("qwen q4 5120->5120", &QWEN38_SHAPES[1]),
        ("qwen q4 5120->17408", &QWEN38_SHAPES[0]),
        ("gemma q4 3840->4096", &GEMMA4_SHAPES[0]),
        ("gemma q8 3840->15360", &GEMMA4_SHAPES[2]),
    ];
    let weights: Vec<Quantized> = columns.iter().map(|(_, s)| quantize_shape(s)).collect();
    let dense = randn(&[HIDDEN, HIDDEN], dtype::FLOAT16);

    println!("\n=== block width sweep ===\n");
    print!("    T");
    for (label, _) in columns {
        print!("   {label:<21}");
    }
    println!("   dense f16 {HIDDEN}x{HIDDEN}");

    for &t in SWEEP {
        print!("{t:5}");
        for (column, weight) in columns.iter().zip(&weights) {
            let x = randn(&[1, t, column.1.in_dim], dtype::FLOAT16);
            let parity = compare(
                t,
                || quantized_forward(weight, &x),
                |i| quantized_forward(weight, &position(&x, i)),
                position,
            );
            print!(
                "   {:<21}",
                if parity.equal() { "equal" } else { "DIVERGES" }
            );
        }
        let x = randn(&[1, t, HIDDEN], dtype::FLOAT16);
        let parity = compare(
            t,
            || mlxcel_core::matmul(&x, &dense),
            |i| mlxcel_core::matmul(&position(&x, i), &dense),
            position,
        );
        println!("   {}", if parity.equal() { "equal" } else { "DIVERGES" });
    }
    println!();
}
