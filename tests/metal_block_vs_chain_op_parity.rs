//! Which primitive stops being bitwise equal between a `T = 1` chain and a
//! `T = K` block on this GPU?
//!
//! ## Why this exists
//!
//! Speculative decoding rests on an equivalence: verifying `K` positions in
//! one `T = K` forward must reproduce what `K` separate `T = 1` decode steps
//! would have produced. The gated-delta chain-parity kernel (#1165) fixed one
//! violation of that equivalence. This file asks which *other* primitives
//! violate it, by comparing, per op, one `T = K` call against `K` single-token
//! calls on the real Qwen3.8-27B shapes.
//!
//! It loads no model and takes about a second.
//!
//! ## What it pins and what it only reports
//!
//! `fast_rms_norm`, `fast_rope` and SDPA are **asserted** equal: the verify
//! path depends on them being position-independent, and a regression there
//! would be a real defect worth failing on.
//!
//! Matmul is **reported, not asserted**. On M5 Max every linear op measured
//! here differs between `T = 1` and `T >= 2`, which is a property of MLX's
//! kernel selection rather than something this repository can assert away;
//! turning it into a failing test would only produce a permanently red suite.
//! The sweep test prints where the break happens so the same command can be
//! run on other Apple Silicon generations and compared.
//!
//! ## Run
//!
//! ```bash
//! cargo test --test metal_block_vs_chain_op_parity --release --features metal,accelerate -- --nocapture
//! ```

use mlxcel::initialize_runtime;
use mlxcel_core::{MlxArray, UniquePtr};

const H: i32 = 5120;
const INTER: i32 = 17408;
const HEAD_DIM: i32 = 256;
const N_Q_HEADS: i32 = 24;
const N_KV_HEADS: i32 = 4;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
const BLOCK: i32 = 3;

fn fill(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.37 + seed).sin()) * scale)
        .collect()
}

fn f16(data: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    let arr = mlxcel_core::from_slice_f32(data, shape);
    mlxcel_core::astype(&arr, mlxcel_core::dtype::FLOAT16)
}

/// `x[:, t:t+1, ...]` on a `[1, T, ...]` tensor.
fn slice_t(arr: &MlxArray, t: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(arr);
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.clone();
    starts[1] = t;
    stops[1] = t + 1;
    mlxcel_core::slice(arr, &starts, &stops)
}

/// `x[:, :, t:t+1, :]` on a `[1, heads, T, dim]` tensor.
fn slice_seq(arr: &MlxArray, t: i32, heads: i32, dim: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::slice(arr, &[0, 0, t, 0], &[1, heads, t + 1, dim])
}

fn raw(arr: &MlxArray) -> Vec<u8> {
    mlxcel_core::eval(arr);
    mlxcel_core::array_to_raw_bytes(arr)
}

/// Bytes that differ between a block position and its single-token counterpart.
fn diff_bytes(block_pos: &MlxArray, single: &MlxArray) -> (usize, usize) {
    let a = raw(block_pos);
    let b = raw(single);
    let differing = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    (differing, a.len())
}

fn quantize(
    w: &MlxArray,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    (
        mlxcel_core::quantize_weights_w(w, GROUP_SIZE, BITS),
        mlxcel_core::quantize_weights_scales(w, GROUP_SIZE, BITS),
        mlxcel_core::quantize_weights_biases(w, GROUP_SIZE, BITS),
    )
}

fn qlinear(
    x: &MlxArray,
    qw: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
) -> UniquePtr<MlxArray> {
    // SAFETY: `biases` outlives the call; no separate linear bias is supplied.
    unsafe {
        mlxcel_core::quantized_linear_forward(
            x,
            qw,
            scales,
            biases as *const MlxArray,
            std::ptr::null(),
            GROUP_SIZE,
            BITS,
            "affine",
        )
    }
}

/// Report a per-position comparison. Returns true when every position matched.
fn report(name: &str, block: &MlxArray, chain: &[UniquePtr<MlxArray>]) -> bool {
    let mut equal = true;
    for (t, single) in chain.iter().enumerate() {
        let (differing, total) = diff_bytes(
            &slice_t(block, t as i32),
            single.as_ref().expect("chain output"),
        );
        if differing > 0 {
            equal = false;
            println!("  [DIFF]  {name}: position {t}, {differing}/{total} bytes differ");
        }
    }
    if equal {
        println!("  [ok]    {name}: block == chain at every position");
    }
    equal
}

/// Quantized linear. `T = 1` is a matrix-vector product, `T >= 2` a
/// matrix-matrix product, and MLX instantiates different kernels for the two.
fn quantized_linear_case(out_features: i32, label: &str) -> bool {
    let w = f16(
        &fill((out_features * H) as usize, 0.11, 0.05),
        &[out_features, H],
    );
    let (qw, scales, biases) = quantize(&w);
    let biases = biases
        .as_ref()
        .expect("affine quantization produces biases");
    let x = f16(&fill((BLOCK * H) as usize, 0.7, 0.3), &[1, BLOCK, H]);

    let block = qlinear(&x, &qw, &scales, biases);
    let chain: Vec<_> = (0..BLOCK)
        .map(|t| qlinear(&slice_t(&x, t), &qw, &scales, biases))
        .collect();
    report(label, &block, &chain)
}

/// The same question without the dequantization path in the way.
fn dense_matmul_case() -> bool {
    let w = f16(&fill((H * H) as usize, 0.31, 0.02), &[H, H]);
    let x = f16(&fill((BLOCK * H) as usize, 0.9, 0.3), &[1, BLOCK, H]);
    let block = mlxcel_core::matmul(&x, &w);
    let chain: Vec<_> = (0..BLOCK)
        .map(|t| mlxcel_core::matmul(&slice_t(&x, t), &w))
        .collect();
    report("dense matmul f16 [1,T,5120] x [5120,5120]", &block, &chain)
}

/// Float32 variant. MLX gates its M5 Neural Accelerator gemm on
/// `is_nax_available() && (enable_tf32() || dtype != float32)`
/// (`mlx/backend/metal/matmul.cpp`). For float16 the second clause is always
/// true, so the NAX kernel cannot be avoided; for float32 it can, with
/// `MLX_ENABLE_TF32=0`. Running this test under both settings attributes how
/// much of the divergence the NAX kernel is responsible for. The printed
/// digests are the arm-attribution control: if the block digest does not move
/// between the two settings, the toggle changed nothing and no conclusion may
/// be drawn from it.
fn dense_matmul_f32_case() -> bool {
    let w = mlxcel_core::from_slice_f32(&fill((H * H) as usize, 0.31, 0.02), &[H, H]);
    let x = mlxcel_core::from_slice_f32(&fill((BLOCK * H) as usize, 0.9, 0.3), &[1, BLOCK, H]);
    let block = mlxcel_core::matmul(&x, &w);
    let chain: Vec<_> = (0..BLOCK)
        .map(|t| mlxcel_core::matmul(&slice_t(&x, t), &w))
        .collect();
    println!(
        "  [hash]  f32 block digest {:016x}, f32 chain[0] digest {:016x}",
        digest(&raw(&block)),
        digest(&raw(chain[0].as_ref().expect("chain output")))
    );
    report(
        "dense matmul f32 [1,T,5120] x [5120,5120] (MLX_ENABLE_TF32-gated)",
        &block,
        &chain,
    )
}

/// FNV-1a over raw bytes. Only used to tell two dispatches apart.
fn digest(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn rms_norm_case() -> bool {
    let weight = f16(&fill(H as usize, 0.05, 1.0), &[H]);
    let x = f16(&fill((BLOCK * H) as usize, 0.44, 0.6), &[1, BLOCK, H]);
    let block = mlxcel_core::fast_rms_norm(&x, &weight, 1e-6);
    let chain: Vec<_> = (0..BLOCK)
        .map(|t| mlxcel_core::fast_rms_norm(&slice_t(&x, t), &weight, 1e-6))
        .collect();
    report("fast_rms_norm [1,T,5120]", &block, &chain)
}

/// RoPE over `[B, heads, T, head_dim]`, with the chain arm applying the
/// matching per-position offset.
fn rope_case() -> bool {
    let n = (N_Q_HEADS * BLOCK * HEAD_DIM) as usize;
    let x = f16(&fill(n, 0.23, 0.5), &[1, N_Q_HEADS, BLOCK, HEAD_DIM]);
    let offset = 128;
    let block = mlxcel_core::fast_rope(&x, HEAD_DIM, false, 10000.0, 1.0, offset);

    let mut equal = true;
    for t in 0..BLOCK {
        let single = mlxcel_core::fast_rope(
            &slice_seq(&x, t, N_Q_HEADS, HEAD_DIM),
            HEAD_DIM,
            false,
            10000.0,
            1.0,
            offset + t,
        );
        let (differing, total) = diff_bytes(&slice_seq(&block, t, N_Q_HEADS, HEAD_DIM), &single);
        if differing > 0 {
            equal = false;
            println!(
                "  [DIFF]  fast_rope [1,24,T,256]: position {t}, {differing}/{total} bytes differ"
            );
        }
    }
    if equal {
        println!("  [ok]    fast_rope [1,24,T,256]: block == chain at every position");
    }
    equal
}

/// SDPA the way the two paths call it: a `T = K` causal block against the
/// per-position form `Qwen3NextAttention::attend_per_position` uses on the
/// verify path.
fn sdpa_block_vs_per_position_case() -> bool {
    let prefix = 64;
    let kv_len = prefix + BLOCK;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let kv_n = (N_KV_HEADS * kv_len * HEAD_DIM) as usize;
    let k = f16(&fill(kv_n, 0.61, 0.4), &[1, N_KV_HEADS, kv_len, HEAD_DIM]);
    let v = f16(&fill(kv_n, 0.13, 0.4), &[1, N_KV_HEADS, kv_len, HEAD_DIM]);
    let q = f16(
        &fill((N_Q_HEADS * BLOCK * HEAD_DIM) as usize, 0.87, 0.3),
        &[1, N_Q_HEADS, BLOCK, HEAD_DIM],
    );

    let block = mlxcel_core::causal_attention(&q, &k, &v, scale, 0.0, 0);

    let mut equal = true;
    for t in 0..BLOCK {
        let q_t = slice_seq(&q, t, N_Q_HEADS, HEAD_DIM);
        let want = prefix + t + 1;
        let k_t = mlxcel_core::slice(&k, &[0, 0, 0, 0], &[1, N_KV_HEADS, want, HEAD_DIM]);
        let v_t = mlxcel_core::slice(&v, &[0, 0, 0, 0], &[1, N_KV_HEADS, want, HEAD_DIM]);
        let single = mlxcel_core::layers::attention(&q_t, &k_t, &v_t, scale, None, 0.0, 0);
        let (differing, total) = diff_bytes(&slice_seq(&block, t, N_Q_HEADS, HEAD_DIM), &single);
        if differing > 0 {
            equal = false;
            println!(
                "  [DIFF]  sdpa causal block vs per-position: position {t}, \
                 {differing}/{total} bytes differ"
            );
        }
    }
    if equal {
        println!("  [ok]    sdpa causal block == per-position chain");
    }
    equal
}

/// Per-op block-versus-chain equality on the real 27B shapes.
///
/// Asserts the ops the verify path needs to be position-independent. Matmul is
/// printed only; see the module docs for why it is not asserted.
#[test]
fn block_vs_chain_op_parity() {
    if !mlxcel_core::metal_is_available() {
        eprintln!("skipping: Metal-only diagnostic");
        return;
    }
    let _runtime = initialize_runtime();

    println!("\nBlock (T={BLOCK}) versus single-token chain, bitwise, Qwen3.8-27B shapes:\n");

    let q_mlp = quantized_linear_case(INTER, "quantized_linear [1,T,5120] -> 17408");
    let q_out = quantized_linear_case(H, "quantized_linear [1,T,5120] -> 5120");
    let dense_f16 = dense_matmul_case();
    let dense_f32 = dense_matmul_f32_case();
    let norm = rms_norm_case();
    let rope = rope_case();
    let sdpa = sdpa_block_vs_per_position_case();

    println!("\nSummary:");
    for (name, ok) in [
        ("quantized_linear 5120->17408", q_mlp),
        ("quantized_linear 5120->5120", q_out),
        ("dense matmul f16", dense_f16),
        ("dense matmul f32", dense_f32),
        ("fast_rms_norm", norm),
        ("fast_rope", rope),
        ("sdpa block vs per-position", sdpa),
    ] {
        println!("  {:<32} {}", name, if ok { "equal" } else { "DIVERGES" });
    }
    println!();

    assert!(
        norm,
        "fast_rms_norm must be position-independent: the speculative verify path \
         relies on a T=K norm reproducing the T=1 decode norm"
    );
    assert!(
        rope,
        "fast_rope must be position-independent at matching offsets: the verify \
         path applies it over a T=K block and decode over single positions"
    );
    assert!(
        sdpa,
        "SDPA must agree between a causal T=K block and the per-position form \
         the verify path uses (Qwen3NextAttention::attend_per_position)"
    );
}

/// One block length, both linear flavors. Returns `(quantized_equal, dense_equal)`.
fn sweep_case(t: i32) -> (bool, bool) {
    let w = f16(&fill((H * H) as usize, 0.11, 0.05), &[H, H]);
    let (qw, scales, biases) = quantize(&w);
    let biases = biases
        .as_ref()
        .expect("affine quantization produces biases");
    let x = f16(&fill((t * H) as usize, 0.7, 0.3), &[1, t, H]);

    let qblock = qlinear(&x, &qw, &scales, biases);
    let dblock = mlxcel_core::matmul(&x, &w);

    let mut q_equal = true;
    let mut d_equal = true;
    for step in 0..t {
        let x_t = slice_t(&x, step);
        if diff_bytes(
            &slice_t(&qblock, step),
            &qlinear(&x_t, &qw, &scales, biases),
        )
        .0 > 0
        {
            q_equal = false;
        }
        if diff_bytes(&slice_t(&dblock, step), &mlxcel_core::matmul(&x_t, &w)).0 > 0 {
            d_equal = false;
        }
    }
    (q_equal, d_equal)
}

/// Locate the block length at which matmul stops matching the single-token
/// chain.
///
/// MLX switches between a matrix-vector and a matrix-matrix quantized kernel on
/// a per-architecture threshold (`get_qmv_batch_limit` in
/// `mlx/backend/metal/quantized.cpp`, 6 to 32 depending on architecture
/// generation and operand size, with `qmm` taken only when
/// `M >= vector_limit`). If that threshold were the whole story, equality would
/// hold below it and break above it. Print the curve rather than trusting the
/// reading, and compare it across GPU generations.
#[test]
fn block_length_sweep_locates_the_kernel_switch() {
    if !mlxcel_core::metal_is_available() {
        eprintln!("skipping: Metal-only diagnostic");
        return;
    }
    let _runtime = initialize_runtime();
    println!("\nBlock length sweep, 5120 -> 5120, f16 activations:\n");
    println!("    T   quantized 4-bit   dense f16");
    for t in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 18, 24, 32] {
        let (q, d) = sweep_case(t);
        println!(
            "  {:>3}   {:<15}   {}",
            t,
            if q { "equal" } else { "DIVERGES" },
            if d { "equal" } else { "DIVERGES" }
        );
    }
    println!();
}
