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

//! Numeric-parity tests for the fused q/k RoPE + KV-append-layout kernel (#905).
//!
//! The reference is the exact graph the Llama3 attention fallback builds:
//! `slice_last_dim` -> `reshape` -> `transpose` -> `fast_rope`. Comparing
//! against that rather than against a re-derived rotation is deliberate; the
//! failure mode this class of kernel actually has is a rotation that is
//! self-consistent but disagrees with `fast::rope` (wrong frequency base,
//! wrong pairing convention, off-by-one position), and only the real op catches
//! that.
//!
//! Position handling gets its own coverage because it is where the cache
//! variants differ. `RingSlidingKVCache` hands out **absolute** positions, not
//! buffer slots, so a kernel that quietly folded a window origin into the
//! position would decode fluent nonsense on rotated caches while passing every
//! offset-zero test. The offset sweep here includes values past a rotating
//! window's capacity for that reason, and the multi-token case pins that token
//! `t` rotates at `positions_base + t`.
//!
//! GPU-only: the kernel JITs through `mx.fast.metal_kernel` / `cuda_kernel` /
//! `hip_kernel`, so these tests skip on a CPU-only build and fail, rather than
//! skip, on a GPU build whose predicate answers false.
//!
//! Run on Apple Silicon:
//!   cargo test --release -p mlxcel-core --lib --features metal,accelerate \
//!     fused_rope_parity_tests

use super::*;

/// Head geometry of a Llama3-8B-class attention block: 32 query heads, 8 KV
/// heads, head_dim 128. Small enough to keep the tests quick, real enough that
/// the GQA head mapping and the q/k/v column offsets are actually exercised.
const N_HEADS: i32 = 32;
const N_KV_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
const ROPE_BASE: f32 = 500000.0;

/// The device lock for a test that runs the `fused_rope_qk_append` port, or
/// `None` (skip) on a build without a GPU backend. Same contract as
/// `gpu_kernel_or_skip` in `fused_norm_parity_tests.rs`: every GPU backend has
/// the port since #2063, so a false predicate on one fails the test.
fn gpu_kernel_or_skip() -> Option<crate::streams::DefaultDeviceLock> {
    use crate::hardware::{GpuBackendKind, gpu_backend_kind};
    if gpu_backend_kind() == GpuBackendKind::None {
        eprintln!("skipping: no GPU backend, so no fused_rope_qk_append port");
        return None;
    }
    let lock = crate::streams::lock_default_device();
    assert!(
        crate::fused_rope_qk_append_available(),
        "every GPU backend has a fused_rope_qk_append port, so the predicate must be true"
    );
    Some(lock)
}

fn flatten_f32(arr: &MlxArray) -> Vec<f32> {
    let a = astype(arr, dtype::FLOAT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn normalized_deviation(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "length mismatch in deviation check");
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    let mut max_abs = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        diff_sq += d * d;
        ref_sq += (*y as f64) * (*y as f64);
        max_abs = max_abs.max(d.abs());
    }
    let ref_rms = (ref_sq / b.len() as f64).sqrt().max(1e-20);
    (
        (diff_sq / a.len() as f64).sqrt() / ref_rms,
        max_abs / ref_rms,
    )
}

/// The rotation is a pair of fp32 multiply-adds around a sine and a cosine that
/// both paths compute from a bit-identical fp32 angle, using the same function
/// as MLX's own RoPE kernel for that backend: `metal::fast::cos` / `sin` on
/// Metal, libdevice `cosf` / `sinf` on CUDA (#1049), `sincosf` on ROCm
/// (#2063). So the trig call itself
/// contributes no divergence, and what is left is the final rounding to the
/// activation dtype plus FMA contraction differing between two separately
/// compiled kernels. One f16 ulp on a 4-sigma element is about 2e-3 relative to
/// the RMS; the budget allows a few of those.
///
/// This premise is load-bearing and was wrong once. Until #1049 the CUDA body
/// called `__cosf` / `__sinf`, whose range reduction degrades outside
/// `[-pi, pi]`, and RoPE angles are unreduced: at position 131071 the `p = 0`
/// pair rotates by about 1.3e5 radians. That is a divergence this budget does
/// not cover and must not be made to cover, so a failure here at large offsets
/// means the trig calls have drifted apart again, not that the tolerance is
/// too tight.
fn assert_close(label: &str, got: &MlxArray, want: &MlxArray) {
    assert_close_within(label, got, want, 2e-3, 1.2e-2);
}

/// [`assert_close`] for a bf16 activation: the f16 budget above is about one
/// bf16 ulp on a 2-sigma element, so a single rounding flip on a backend where
/// the two paths contract differently would fail it. The bf16 budget is the
/// one `fused_norm_parity_tests.rs` uses for the same ulp reason.
fn assert_close_for(label: &str, got: &MlxArray, want: &MlxArray, dt: i32) {
    if dt == dtype::BFLOAT16 {
        assert_close_within(label, got, want, 1.6e-2, 7e-2);
    } else {
        assert_close(label, got, want);
    }
}

fn assert_close_within(label: &str, got: &MlxArray, want: &MlxArray, rms_tol: f64, max_tol: f64) {
    let (nrms, nmax) = normalized_deviation(&flatten_f32(got), &flatten_f32(want));
    assert!(
        nrms < rms_tol && nmax < max_tol,
        "{label}: normalized rms {nrms:.3e} (tol {rms_tol:.1e}), normalized max {nmax:.3e} (tol {max_tol:.1e})"
    );
}

fn assert_shape(label: &str, arr: &MlxArray, want: &[i32]) {
    assert_eq!(array_shape(arr), want, "{label}: unexpected shape");
}

/// A random row-contiguous fused-QKV projection output `[B, L, (Hq + 2*Hkv)*D]`.
fn random_qkv(seed: u64, batch: i32, seq: i32) -> UniquePtr<MlxArray> {
    random_qkv_as(seed, batch, seq, dtype::FLOAT16)
}

/// [`random_qkv`] in an explicit activation dtype.
fn random_qkv_as(seed: u64, batch: i32, seq: i32, dt: i32) -> UniquePtr<MlxArray> {
    random_seed(seed);
    let cols = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM;
    let raw = unsafe { random_normal(&[batch, seq, cols], dtype::FLOAT32, std::ptr::null()) };
    let qkv = astype(&raw, dt);
    eval(&qkv);
    qkv
}

fn run_fused(
    qkv: &MlxArray,
    rope_dims: i32,
    traditional: bool,
    positions_base: i32,
    dest_layout: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let mut q = UniquePtr::null();
    let mut k = UniquePtr::null();
    let mut v = UniquePtr::null();
    crate::fused_rope_qk_append(
        qkv,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        rope_dims,
        ROPE_BASE,
        1.0,
        traditional,
        positions_base,
        dest_layout,
        &mut q,
        &mut k,
        &mut v,
    )
    .expect("gpu_kernel_or_skip() checked the port table, so the launcher must not refuse");
    eval(&q);
    eval(&k);
    eval(&v);
    (q, k, v)
}

/// The graph the Llama3 attention fallback builds, op for op.
fn reference_graph(
    qkv: &MlxArray,
    rope_dims: i32,
    traditional: bool,
    positions_base: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let shape = array_shape(qkv);
    let (b, l) = (shape[0], shape[1]);
    let q_size = N_HEADS * HEAD_DIM;
    let kv_size = N_KV_HEADS * HEAD_DIM;

    let q = slice_last_dim(qkv, 0, q_size);
    let k = slice_last_dim(qkv, q_size, q_size + kv_size);
    let v = slice_last_dim(qkv, q_size + kv_size, q_size + 2 * kv_size);

    let q = transpose_axes(&reshape(&q, &[b, l, N_HEADS, HEAD_DIM]), &[0, 2, 1, 3]);
    let k = transpose_axes(&reshape(&k, &[b, l, N_KV_HEADS, HEAD_DIM]), &[0, 2, 1, 3]);
    let v = transpose_axes(&reshape(&v, &[b, l, N_KV_HEADS, HEAD_DIM]), &[0, 2, 1, 3]);

    let q = fast_rope(&q, rope_dims, traditional, ROPE_BASE, 1.0, positions_base);
    let k = fast_rope(&k, rope_dims, traditional, ROPE_BASE, 1.0, positions_base);
    eval(&q);
    eval(&k);
    eval(&v);
    (q, k, v)
}

/// Offsets a decode step can actually carry. The large values are the point:
/// `RingSlidingKVCache` positions are absolute and keep climbing past the
/// window capacity, so a kernel that assumed a bounded offset would only fail
/// here.
const OFFSETS: &[i32] = &[0, 1, 7, 63, 511, 4096, 131071];

#[test]
fn fused_rope_append_matches_graph_rope_across_offsets() {
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    for (i, &offset) in OFFSETS.iter().enumerate() {
        let qkv = random_qkv(600 + i as u64, 1, 1);
        let (q, k, v) = run_fused(&qkv, HEAD_DIM, false, offset, 0);
        let (want_q, want_k, want_v) = reference_graph(&qkv, HEAD_DIM, false, offset);

        assert_shape("q", &q, &[1, N_HEADS, 1, HEAD_DIM]);
        assert_shape("k", &k, &[1, N_KV_HEADS, 1, HEAD_DIM]);
        assert_shape("v", &v, &[1, N_KV_HEADS, 1, HEAD_DIM]);

        assert_close(&format!("q offset={offset}"), &q, &want_q);
        assert_close(&format!("k offset={offset}"), &k, &want_k);
        assert_close(&format!("v offset={offset}"), &v, &want_v);
    }
}

/// Every activation dtype a checkpoint decodes in. The kernel is keyed on `T`,
/// so each dtype is a separately compiled kernel (a separate hipRTC cache entry
/// on ROCm) and gets its own comparison; f16 alone would leave the f32 and bf16
/// loads and stores untested. Offsets include the long-context angle where a
/// trig mismatch between the two paths would show.
#[test]
fn fused_rope_append_matches_graph_rope_every_dtype() {
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    for &dt in &[dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        for &(seq, offset) in &[(1i32, 0i32), (1, 4096), (1, 131071), (3, 29)] {
            let qkv = random_qkv_as(1600 + seq as u64 + offset as u64, 2, seq, dt);
            let (q, k, v) = run_fused(&qkv, HEAD_DIM, false, offset, 0);
            let (want_q, want_k, want_v) = reference_graph(&qkv, HEAD_DIM, false, offset);
            assert_eq!(array_dtype(&q), dt, "q keeps the activation dtype");
            assert_close_for(
                &format!("q dt={dt} seq={seq} offset={offset}"),
                &q,
                &want_q,
                dt,
            );
            assert_close_for(
                &format!("k dt={dt} seq={seq} offset={offset}"),
                &k,
                &want_k,
                dt,
            );
            assert_close_for(
                &format!("v dt={dt} seq={seq} offset={offset}"),
                &v,
                &want_v,
                dt,
            );
        }
    }
}

/// Multi-token windows (prefill, or a speculative batch) must rotate token `t`
/// at `positions_base + t`, which is what makes a resumed prefill land on the
/// same rotation as the decode steps that follow it.
#[test]
fn fused_rope_append_multi_token_positions_are_absolute() {
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    for &(seq, offset) in &[(4i32, 0i32), (7, 1), (16, 4093), (3, 131069)] {
        let qkv = random_qkv(700 + seq as u64 + offset as u64, 1, seq);
        let (q, k, v) = run_fused(&qkv, HEAD_DIM, false, offset, 0);
        let (want_q, want_k, want_v) = reference_graph(&qkv, HEAD_DIM, false, offset);
        assert_shape("q", &q, &[1, N_HEADS, seq, HEAD_DIM]);
        assert_close(&format!("q seq={seq} offset={offset}"), &q, &want_q);
        assert_close(&format!("k seq={seq} offset={offset}"), &k, &want_k);
        assert_close(&format!("v seq={seq} offset={offset}"), &v, &want_v);
    }
}

/// Batch > 1 exercises the `[B, ...]` addressing on both the input row stride
/// and the two output layouts; a decode batch shares one position base.
#[test]
fn fused_rope_append_matches_graph_rope_batched() {
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    let qkv = random_qkv(808, 4, 2);
    let (q, k, v) = run_fused(&qkv, HEAD_DIM, false, 37, 0);
    let (want_q, want_k, want_v) = reference_graph(&qkv, HEAD_DIM, false, 37);
    assert_shape("q", &q, &[4, N_HEADS, 2, HEAD_DIM]);
    assert_shape("k", &k, &[4, N_KV_HEADS, 2, HEAD_DIM]);
    assert_close("batched q", &q, &want_q);
    assert_close("batched k", &k, &want_k);
    assert_close("batched v", &v, &want_v);
}

/// The interleaved (traditional) convention pairs `2p` with `2p + 1` instead of
/// `p` with `p + dims/2`. Getting this wrong produces correctly shaped tensors
/// and fluent-looking garbage, which is exactly why the Llama3 attention bypasses
/// `forward_split_rope` for traditional-RoPE checkpoints; this path takes the
/// flag for real, so it has to be pinned.
#[test]
fn fused_rope_append_traditional_matches_graph_rope() {
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    for &offset in &[0i32, 13, 2048] {
        let qkv = random_qkv(900 + offset as u64, 1, 3);
        let (q, k, _) = run_fused(&qkv, HEAD_DIM, true, offset, 0);
        let (want_q, want_k, _) = reference_graph(&qkv, HEAD_DIM, true, offset);
        assert_close(&format!("traditional q offset={offset}"), &q, &want_q);
        assert_close(&format!("traditional k offset={offset}"), &k, &want_k);
    }
}

/// Partial rotary: only the first `rope_dims` of each head rotate and the tail
/// is copied through untouched. The tail is handled by the leftover threads of
/// the same dispatch, so a mis-sized tail loop would silently drop or duplicate
/// elements.
#[test]
fn fused_rope_append_partial_rope_dims_copies_the_tail() {
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    for &rope_dims in &[64i32, 96, 32] {
        for &traditional in &[false, true] {
            let qkv = random_qkv(1100 + rope_dims as u64, 1, 2);
            let (q, k, _) = run_fused(&qkv, rope_dims, traditional, 19, 0);
            let (want_q, want_k, _) = reference_graph(&qkv, rope_dims, traditional, 19);
            assert_close(
                &format!("partial q dims={rope_dims} traditional={traditional}"),
                &q,
                &want_q,
            );
            assert_close(
                &format!("partial k dims={rope_dims} traditional={traditional}"),
                &k,
                &want_k,
            );
        }
    }
}

/// The paged pool layout must be the dense layout with the head and token axes
/// swapped, and nothing else: same values, same rotation, different addressing.
/// Wiring for this layout belongs to issue #899, so this test is what keeps it
/// honest until then.
#[test]
fn fused_rope_append_paged_layout_is_the_dense_layout_transposed() {
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    let qkv = random_qkv(1313, 2, 5);
    let (q_dense, k_dense, v_dense) = run_fused(&qkv, HEAD_DIM, false, 71, 0);
    let (q_paged, k_paged, v_paged) = run_fused(&qkv, HEAD_DIM, false, 71, 1);

    assert_shape("paged k", &k_paged, &[2, 5, N_KV_HEADS, HEAD_DIM]);
    assert_shape("paged v", &v_paged, &[2, 5, N_KV_HEADS, HEAD_DIM]);
    // Q is in attention order for both layouts; only K/V follow `dest_layout`.
    assert_shape("paged q", &q_paged, &[2, N_HEADS, 5, HEAD_DIM]);

    let k_dense_as_paged = transpose_axes(&k_dense, &[0, 2, 1, 3]);
    let v_dense_as_paged = transpose_axes(&v_dense, &[0, 2, 1, 3]);
    eval(&k_dense_as_paged);
    eval(&v_dense_as_paged);

    assert_close("paged q vs dense q", &q_paged, &q_dense);
    assert_close("paged k vs dense k", &k_paged, &k_dense_as_paged);
    assert_close("paged v vs dense v", &v_paged, &v_dense_as_paged);
}

/// V rides through the dispatch purely for the relayout, so it must come out
/// bit-identical to the projection's V block, not merely close: any rotation
/// leaking onto V would be a silent correctness bug that the tolerance-based
/// checks above could absorb.
#[test]
fn fused_rope_append_leaves_v_bit_identical() {
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    let qkv = random_qkv(1414, 1, 3);
    let (_, _, v) = run_fused(&qkv, HEAD_DIM, false, 12345, 0);

    let q_size = N_HEADS * HEAD_DIM;
    let kv_size = N_KV_HEADS * HEAD_DIM;
    let v_ref = slice_last_dim(&qkv, q_size + kv_size, q_size + 2 * kv_size);
    let v_ref = transpose_axes(
        &reshape(&v_ref, &[1, 3, N_KV_HEADS, HEAD_DIM]),
        &[0, 2, 1, 3],
    );
    let v_ref = contiguous(&v_ref, false);
    eval(&v_ref);

    assert_eq!(
        array_to_raw_bytes(&v),
        array_to_raw_bytes(&v_ref),
        "V must be a pure relayout of the projection's V block"
    );
}

/// LoRA fusion and the surgery tooling rewrite projection weights in place, so
/// the kernel sees a differently scaled `qkv` than the checkpoint produced. It
/// reads the projection output and nothing derived from the weights, so a
/// rewritten weight has to flow through unchanged.
#[test]
fn fused_rope_append_matches_graph_with_scaled_projection() {
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    let base = random_qkv(1515, 1, 2);
    // Stand-in for a merged adapter: the projection output after the merge is a
    // scaled version of the checkpoint's, in the same dtype.
    let scaled = astype(
        &multiply(
            &astype(&base, dtype::FLOAT32),
            &full_f32(&[1], 0.43, dtype::FLOAT32),
        ),
        dtype::FLOAT16,
    );
    eval(&scaled);

    let (q, k, v) = run_fused(&scaled, HEAD_DIM, false, 256, 0);
    let (want_q, want_k, want_v) = reference_graph(&scaled, HEAD_DIM, false, 256);
    assert_close("scaled-projection q", &q, &want_q);
    assert_close("scaled-projection k", &k, &want_k);
    assert_close("scaled-projection v", &v, &want_v);
}

/// The kill switch is the contract the measure-then-keep policy in #905 rests
/// on, so the gate has to actually follow the environment variable. Run the
/// file once normally and once with `MLXCEL_FUSED_ROPE_APPEND=0`.
#[test]
fn fused_rope_append_gate_follows_the_kill_switch() {
    // Same precedence as the add-RMSNorm gate: explicit value wins, unset or
    // unrecognised keeps the compiled-in default. Asserting `!disabled` would
    // pin default-on and break whenever a measurement flips the constant.
    let raw = std::env::var("MLXCEL_FUSED_ROPE_APPEND").ok();
    let expected = match raw.as_deref().map(|v| v.trim().to_ascii_lowercase()) {
        Some(ref v) if v == "0" || v == "false" || v == "off" || v == "no" => false,
        Some(ref v) if v == "1" || v == "true" || v == "on" || v == "yes" => true,
        _ => crate::layers::FUSED_ROPE_APPEND_DEFAULT,
    };
    assert_eq!(
        crate::layers::fused_rope_append_enabled(),
        expected,
        "gate does not match MLXCEL_FUSED_ROPE_APPEND"
    );
}

/// On ROCm the port computes the angle in the overlay's `rope.hip` order and
/// takes both sides with the same `sincosf`, so it reproduces the graph's
/// `fast_rope` exactly rather than within the cross-backend tolerance above.
/// Byte identity is the stronger pin, including at the long-context angle.
///
/// The graph runs two differently compiled kernels: `rope_single_1d` for one
/// row-contiguous token of one sequence (batch-1 decode, the case that
/// matters most) and `rope` otherwise, and the two round their second output
/// differently. Both shapes are covered here; the batch-1 single-token cases
/// are the ones a port that matched only `rope` failed. The two long windows
/// are where an f16 port that rounded through f32 failed (see
/// `fused_rope_append_hip.h`): about one f16 element in eight thousand lands
/// on the other side, so short windows rarely show it.
#[test]
fn fused_rope_append_is_byte_identical_to_the_rocm_graph() {
    use crate::hardware::{GpuBackendKind, gpu_backend_kind};
    if gpu_backend_kind() != GpuBackendKind::Rocm {
        return;
    }
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    let as_contiguous = |a: &MlxArray| {
        let c = contiguous(a, false);
        eval(&c);
        array_to_raw_bytes(&c)
    };
    for &dt in &[dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        for &(batch, seq, offset, traditional) in &[
            (1i32, 1i32, 512i32, false),
            (1, 1, 4096, false),
            (1, 1, 131071, false),
            (1, 1, 777, true),
            (1, 5, 29, false),
            (1, 512, 0, false),
            (2, 1, 0, false),
            (2, 1, 4096, false),
            (2, 1, 131071, false),
            (2, 5, 29, false),
            (2, 3, 2048, true),
            (3, 186, 4555, false),
            (1, 64, 9110, true),
        ] {
            let seed = 1700 + (batch * 7 + seq) as u64 + offset as u64;
            let qkv = random_qkv_as(seed, batch, seq, dt);
            let (q, k, v) = run_fused(&qkv, HEAD_DIM, traditional, offset, 0);
            let (want_q, want_k, want_v) = reference_graph(&qkv, HEAD_DIM, traditional, offset);
            let label = format!(
                "dt={dt} batch={batch} seq={seq} offset={offset} traditional={traditional}"
            );
            assert_eq!(as_contiguous(&q), as_contiguous(&want_q), "q {label}");
            assert_eq!(as_contiguous(&k), as_contiguous(&want_k), "k {label}");
            assert_eq!(as_contiguous(&v), as_contiguous(&want_v), "v {label}");
        }
    }
}
