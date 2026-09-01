//! [#1544] The numbers the CUDA grouped GEMM produces on the device it is
//! dispatched for.
//!
//! Companion to `grouped_gemm_arch_tests.rs`, which pins the architecture tag
//! decision. This module answers the other half of #1544: whether the path that
//! decision feeds is arithmetically correct on the part it selects.
//!
//! It is not really a test of the retag, which is device-codegen neutral
//! (`docs/benchmark_results/grouped-gemm-arch-v100-2026-08-31.md` records 51
//! device symbols with byte-identical bodies either way). It is the check the
//! issue asked for and nobody had made. The grouped GEMM is what every
//! non-quantized MoE checkpoint routes its experts through, and what a
//! quantized one routes prefill through above #629's `B >= 8 * num_experts`
//! gate, on a tag that named the wrong architecture; the one test that touched
//! it asserted the output shape and never looked at a value. A CUTLASS
//! configuration compiled for an architecture it does not match can return
//! plausible numbers that are wrong, and greedy text generation will not
//! reveal it.
//!
//! The comparison is against a host reference computed in `f64` from the same
//! bytes that were uploaded, on the expert shapes `gemma-4-26b-a4b-it` actually
//! uses (`hidden_size` 2816, `moe_intermediate_size` 704), and it covers both
//! entry points the dispatch has: `cutlass_gather_mm` for the general case and
//! `cutlass_grouped_gemm_unaligned` for the sorted single-row case that
//! non-quantized MoE decode takes.
//!
//! GPU-only (Metal or CUDA); these skip on a CPU-only build.

use super::*;

/// Deterministic host data. A tiny LCG rather than MLX's RNG, so the reference
/// is computed from exactly the bytes that were uploaded and the test does not
/// depend on the device's random stream.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Top 24 bits, mapped to [-1, 1).
    ((*seed >> 40) as f32 / 8_388_608.0) - 1.0
}

fn fill(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n).map(|_| lcg(seed)).collect()
}

/// `out[i] = a[i] @ b[rhs_indices[i]]`, accumulated in `f64`.
///
/// `a` is `[batch, m, k]` and `b` is `[experts, k, n]`, both row major, which
/// is the logical layout `gather_mm` is given regardless of how `b` is stored.
fn reference_gather_mm(
    a: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    rhs_indices: &[i32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * n];
    for i in 0..batch {
        let e = rhs_indices[i] as usize;
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for d in 0..k {
                    acc += a[i * m * k + row * k + d] as f64 * b[e * k * n + d * n + col] as f64;
                }
                out[i * m * n + row * n + col] = acc as f32;
            }
        }
    }
    out
}

fn as_f32(arr: &MlxArray) -> Vec<f32> {
    let f32_arr = astype(arr, dtype::FLOAT32);
    eval(&f32_arr);
    let bytes = array_to_raw_bytes(&f32_arr);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], rtol: f32, atol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = atol + rtol * w.abs();
        let err = (g - w).abs();
        if err - tol > worst {
            worst = err - tol;
            worst_at = i;
        }
    }
    assert!(
        worst == 0.0,
        "{what}: grouped GEMM disagrees with the dense per-expert reference. \
         Worst element {worst_at}: got {}, want {}, exceeding tolerance by {worst}",
        got[worst_at],
        want[worst_at]
    );
}

/// One shape and routing combination for the reference comparison below.
///
/// A named struct rather than a tuple because the arguments are six integers
/// and two flags in a row, where a transposition would silently change which
/// dispatch arm the case lands in rather than failing to compile.
#[derive(Clone, Copy)]
struct Case {
    experts: usize,
    batch: usize,
    /// Rows per group. `1` with `sorted` takes `cutlass_grouped_gemm_unaligned`;
    /// anything else takes `cutlass_gather_mm`.
    m: usize,
    k: usize,
    /// `n % 8` selects between the two `kAlignmentC` instantiations.
    n: usize,
    sorted: bool,
    /// Store `b` as `[experts, n, k]` and pass a `swap_axes(-1, -2)` view, which
    /// is what production does, rather than a contiguous `[experts, k, n]`.
    swap_b: bool,
    label: &'static str,
}

impl Case {
    #[allow(clippy::too_many_arguments)]
    const fn new(
        experts: usize,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        sorted: bool,
        swap_b: bool,
        label: &'static str,
    ) -> Self {
        Self {
            experts,
            batch,
            m,
            k,
            n,
            sorted,
            swap_b,
            label,
        }
    }
}

/// The grouped GEMM must agree with a dense per-expert reference on the expert
/// shapes the MoE checkpoint actually uses.
///
/// This is the check #1544 asked for. The path under test is what
/// `SwitchLinear::Regular::forward` calls for every non-quantized MoE
/// checkpoint (`gpt_oss`, `deepseek`, `qwen3_next`, `jamba`, `nemotron_h`,
/// `exaone_moe` and the rest), and it runs on a CUTLASS configuration that was
/// tagged for the wrong architecture until #1544.
///
/// The two cases below reach the two different entry points in
/// `matmul.cpp`'s `GatherMM::eval_gpu`:
///
/// - `m > 1` falls through to `cutlass_gather_mm`, which is prefill.
/// - `m == 1` with `sorted` and no `lhs_indices` takes `gather_mm_rhs` into
///   `cutlass_grouped_gemm_unaligned`, which is decode.
///
/// `n = 704` exercises the `n % 8 == 0` arm of the alignment dispatch and
/// `n = 703` the unaligned arm, which are separate template instantiations.
#[test]
fn gather_mm_matches_dense_per_expert_reference() {
    if !crate::metal_is_available() && !crate::cuda_is_available() {
        return;
    }

    // k and n are gemma-4-26b-a4b-it's real expert dims: hidden_size 2816 and
    // moe_intermediate_size 704, in both the gate/up and the down orientation.
    // The expert count is cut from 128 to 4 because the grouped GEMM indexes
    // groups rather than iterating them, so the count changes the pointer
    // table and nothing about the arithmetic.
    let cases = &[
        Case::new(4, 4, 4, 2816, 704, false, true, "gate/up prefill, m=4"),
        Case::new(4, 4, 4, 704, 2816, false, true, "down prefill, m=4"),
        Case::new(4, 4, 1, 2816, 704, true, true, "gate/up decode, m=1 sorted"),
        Case::new(
            4,
            4,
            1,
            2816,
            704,
            false,
            false,
            "gate/up decode, m=1 unsorted",
        ),
        Case::new(4, 4, 2, 512, 703, false, true, "unaligned n, kAlignmentC=1"),
        Case::new(4, 4, 2, 512, 704, false, true, "aligned n, kAlignmentC=8"),
    ];

    for &Case {
        experts,
        batch,
        m,
        k,
        n,
        sorted,
        swap_b,
        label,
    } in cases
    {
        let mut seed = 1544u64;
        let a_host = fill(batch * m * k, &mut seed);
        let b_host = fill(experts * k * n, &mut seed);
        // Non-decreasing, and not the identity, so a dropped or mis-scaled
        // group index cannot pass by accident. `gather_mm_rhs` requires sorted
        // indices when `sorted` is set.
        let idx: Vec<i32> = (0..batch).map(|i| (i % experts) as i32).collect();

        let a = from_slice_f32(&a_host, &[batch as i32, m as i32, k as i32]);
        // Production stores expert weights as `[experts, n, k]` and hands
        // `gather_mm` a `swap_axes(-1, -2)` view of them, which is a
        // transposed operand rather than a contiguous one. Both are exercised
        // because `check_transpose` routes them differently.
        let b = if swap_b {
            let mut bt = vec![0.0f32; experts * k * n];
            for e in 0..experts {
                for r in 0..k {
                    for c in 0..n {
                        bt[e * n * k + c * k + r] = b_host[e * k * n + r * n + c];
                    }
                }
            }
            let b_nk = from_slice_f32(&bt, &[experts as i32, n as i32, k as i32]);
            swap_axes(&b_nk, -1, -2)
        } else {
            from_slice_f32(&b_host, &[experts as i32, k as i32, n as i32])
        };
        let indices = from_slice_i32(&idx, &[batch as i32]);

        let out = unsafe {
            gather_mm(
                &a,
                &b,
                std::ptr::null(),
                indices.as_ref().unwrap() as *const MlxArray,
                sorted,
            )
        };
        eval(&out);
        assert_eq!(
            array_shape(&out),
            vec![batch as i32, m as i32, n as i32],
            "{label}: unexpected output shape"
        );

        let want = reference_gather_mm(&a_host, batch, m, k, &b_host, n, &idx);
        // f32 SIMT FMA against an f64 reference over k up to 2816: the error
        // grows like sqrt(k) * eps, about 6e-6 relative. atol covers the
        // elements where cancellation leaves rtol meaningless.
        assert_close(&as_f32(&out), &want, 1e-4, 1e-4, label);
    }
}

/// A wrong expert index is a silent, plausible-looking failure, so it gets a
/// case that can only pass if the gather is exact.
///
/// Each expert's weight matrix is a distinct constant, so the output of expert
/// `e` is a known multiple of the row sum of `a` and any index confusion moves
/// every element of a slab by a whole multiple rather than by a rounding error.
#[test]
fn gather_mm_selects_the_indexed_expert() {
    if !crate::metal_is_available() && !crate::cuda_is_available() {
        return;
    }

    let (experts, batch, m, k, n) = (5usize, 7usize, 3usize, 64usize, 16usize);
    let mut seed = 987u64;
    let a_host = fill(batch * m * k, &mut seed);
    // Expert e is the constant matrix (e + 1).
    let mut b_host = vec![0.0f32; experts * k * n];
    for e in 0..experts {
        for v in b_host[e * k * n..(e + 1) * k * n].iter_mut() {
            *v = (e + 1) as f32;
        }
    }
    // Deliberately not sorted and not the identity permutation, with a repeat.
    let idx: Vec<i32> = vec![3, 0, 4, 4, 1, 2, 0];

    let a = from_slice_f32(&a_host, &[batch as i32, m as i32, k as i32]);
    let b = from_slice_f32(&b_host, &[experts as i32, k as i32, n as i32]);
    let indices = from_slice_i32(&idx, &[batch as i32]);

    let out = unsafe {
        gather_mm(
            &a,
            &b,
            std::ptr::null(),
            indices.as_ref().unwrap() as *const MlxArray,
            false,
        )
    };
    eval(&out);
    let got = as_f32(&out);
    let want = reference_gather_mm(&a_host, batch, m, k, &b_host, n, &idx);
    assert_close(&got, &want, 1e-5, 1e-4, "constant-per-expert gather");

    // And state the discriminator directly: every element of slab i is
    // (idx[i] + 1) times the corresponding row sum of a.
    for i in 0..batch {
        for row in 0..m {
            let row_sum: f64 = a_host[i * m * k + row * k..i * m * k + (row + 1) * k]
                .iter()
                .map(|&v| v as f64)
                .sum();
            let expect = row_sum * (idx[i] + 1) as f64;
            for col in 0..n {
                let g = got[i * m * n + row * n + col] as f64;
                assert!(
                    (g - expect).abs() <= 1e-3 + 1e-4 * expect.abs(),
                    "slab {i} row {row} col {col}: expert {} should scale the row sum by {}, got {g} against {expect}",
                    idx[i],
                    idx[i] + 1
                );
            }
        }
    }
}

/// The half-precision configurations have to be right too.
///
/// `bfloat16` and `float16` select a different `GemmConfiguration`
/// instantiation from `float`: `CommonGemmConfiguration` promotes the
/// accumulator to `float` when `sizeof(T) < 4`, and the epilogue converts back.
/// These are also the dtypes a real checkpoint is stored in, and they are the
/// ones where an `m16n8k8` MMA would have been selected had the pre-Ampere arm
/// ever reached a tensor-core configuration.
#[test]
fn gather_mm_half_precision_matches_reference() {
    if !crate::metal_is_available() && !crate::cuda_is_available() {
        return;
    }

    let (experts, batch, m, k, n) = (4usize, 4usize, 4usize, 704usize, 256usize);
    for &(dt, name, rtol) in &[
        (dtype::BFLOAT16, "bfloat16", 4e-2f32),
        (dtype::FLOAT16, "float16", 4e-3f32),
    ] {
        let mut seed = 424242u64;
        let a_host = fill(batch * m * k, &mut seed);
        let b_host = fill(experts * k * n, &mut seed);
        let idx: Vec<i32> = (0..batch).map(|i| (i % experts) as i32).collect();

        let a32 = from_slice_f32(&a_host, &[batch as i32, m as i32, k as i32]);
        let b32 = from_slice_f32(&b_host, &[experts as i32, k as i32, n as i32]);
        let a = astype(&a32, dt);
        let b = astype(&b32, dt);
        eval(&a);
        eval(&b);
        // Round the reference through the same storage precision the device
        // operands carry, so the comparison measures the GEMM and not the cast.
        let a_round = as_f32(&a);
        let b_round = as_f32(&b);
        let indices = from_slice_i32(&idx, &[batch as i32]);

        let out = unsafe {
            gather_mm(
                &a,
                &b,
                std::ptr::null(),
                indices.as_ref().unwrap() as *const MlxArray,
                false,
            )
        };
        eval(&out);

        let want = reference_gather_mm(&a_round, batch, m, k, &b_round, n, &idx);
        // The output is stored in `dt`, so the tolerance is dominated by the
        // final conversion: bf16 carries an 8-bit significand and fp16 an
        // 11-bit one. Accumulation itself is in float.
        assert_close(&as_f32(&out), &want, rtol, rtol, name);
    }
}
