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

//! Numeric checks of the block-float (`mxfp8`, `mxfp4`) matmuls mlxcel runs,
//! against a reference computed on the host (issue #1807).
//!
//! Two call sites are covered, each through the layer that production code
//! uses rather than a bare FFI call:
//!
//! * the MoE expert path, [`SwitchLinear::forward`] over MLX `gather_qmm`
//!   (`transpose = true`, no `lhs_indices`), in the shapes `SwitchGLU`
//!   produces: unsorted decode and small-prefill batches, and the sorted
//!   large-prefill batch built by [`gather_sort`];
//! * the dense projection, `UnifiedLinear::forward` over `quantized_matmul`,
//!   which every tensor of a vendor FP8 block checkpoint reaches after
//!   `fp8_block::requantize_block_fp8_weights` converts it to `mxfp8`.
//!
//! The reference decodes the packed codes and E8M0 scales on the host
//! (`E4M3` for `mxfp8`, `E2M1` for `mxfp4`, scale `2^(e - 127)`) and
//! accumulates in f64 from the activations as the device holds them, so it
//! shares no kernel with the backend under test. A separate test pins the
//! host decoder to MLX `dequantize`, so a wrong reference cannot pass
//! silently.
//!
//! # Which ROCm branches this reaches
//!
//! `GatherQMM::eval_gpu` in `patches-rocm/mlx/backend/rocm/quantized/qmm.hip`
//! gates every specialised path (grouped WMMA prefill, expert-batched, tiled,
//! wide, idot and the warp-shared fast kernel) on `mode_ == Affine`, so a
//! non-affine mode reaches exactly one launch: the
//! `gather_qmv_kernel<T, uint8_t, BITS, 32, AFFINE=false>` block that
//! `LOCAL_FIXES.md` item 10 added. Inside it the cases below cover:
//!
//! * `T` = `hip_bfloat16`, `__half` and `float` (all three dtype arms), with
//!   `BITS` = 8 and 4 (both bit arms);
//! * `implicit_lhs = false`: the unsorted decode (`B = 4`) and unsorted
//!   prefill (`B = 24`) cases, and the `M = 4` multi-row case;
//! * `implicit_lhs = true` (`use_sorted_rhs_schedule`: sorted, `M == 1`,
//!   `B >= 16`, `B / E >= 4`) with `implicit_x_batch_stride = K`, the sorted
//!   prefill case (`B = 128`, `E = 8`), and with stride 0, the sorted
//!   shared-activation case (`B = 32`, `E = 8`);
//! * `N = 320`, so the second column block runs its `col >= N` bound check.
//!
//! This is the default path. The expert-batched kernel from item 9 (on by
//! default for sorted affine bf16 and f16 prefill since lablup/mlxcel#2066)
//! is affine-only and cannot be reached from a block-float mode, so no case
//! targets it; `tests/rocm_gather_qmm_expert_batched.rs` covers it.
//!
//! The tests are not backend-gated: the tolerances are what a correct
//! backend meets from output rounding alone, and
//! `mxfp_matmuls_match_host_reference_on_cpu_device` runs a reduced matrix on
//! MLX's CPU backend (both modes, unsorted and sorted gather, multi-row, dense
//! decode, bf16 and f16) to show the bound is not tuned to one GPU.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::{SwitchLinear, gather_sort};
use crate::models::sanitize::f8_e4m3_to_f32;

const GROUP_SIZE: usize = 32;
const EXPERTS: usize = 8;
const OUT_FEATURES: usize = 320;
const IN_FEATURES: usize = 256;
/// Output width of the CPU arm. The GPU arms use `OUT_FEATURES = 320` so the
/// second column block runs its bound check; on the CPU that only costs
/// scalar quantize and matmul time.
const CPU_OUT_FEATURES: usize = 64;

/// The two block-float modes the ROCm gather path accepts.
#[derive(Debug, Clone, Copy)]
enum Mode {
    Mxfp8,
    Mxfp4,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Mxfp8 => "mxfp8",
            Self::Mxfp4 => "mxfp4",
        }
    }

    fn bits(self) -> usize {
        match self {
            Self::Mxfp8 => 8,
            Self::Mxfp4 => 4,
        }
    }

    fn decode(self, code: u32) -> f32 {
        match self {
            Self::Mxfp8 => f8_e4m3_to_f32(code as u8),
            Self::Mxfp4 => {
                const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                let magnitude = E2M1[(code & 7) as usize];
                if code & 8 != 0 { -magnitude } else { magnitude }
            }
        }
    }
}

/// Activation dtype and the relative L2 error a correct backend stays under.
///
/// The bound comes from rounding to the activation dtype (unit roundoff 2^-9
/// for bf16, 2^-11 for f16), with room for where a backend rounds its
/// partial sums. Measured on gfx1151 across every case below: at most
/// 1.8e-3 (bf16), 2.2e-4 (f16) and 2.9e-7 (f32). MLX's CPU backend, which
/// rounds partials to the activation dtype, measured up to 8.2e-3 (bf16) and
/// 1.0e-3 (f16) on the reduced matrix the CPU test runs, and 7.9e-3, 1.0e-3
/// and 1.2e-7 on the full matrix, so the bf16 and f16 bounds sit about 2.4x
/// above the least accurate correct backend seen. The f32 bound is not set by those numbers:
/// on CUDA sm80 and later the sorted prefill case takes MLX's grouped GEMM
/// (`B >= 8 * E`), which with `MLX_ENABLE_TF32` at its default of on runs
/// f32 through TF32 tensor cores, rounding the activations to a 10-bit
/// mantissa (unit roundoff 2^-11, about 4.9e-4). 2e-3 covers that with room
/// and is still three orders of magnitude below the defect these tests exist
/// for, fp weights read with the affine formula (`LOCAL_FIXES.md` item 10),
/// which measured between 1.0 and 1e34.
const ACTIVATIONS: [(i32, &str, f64); 3] = [
    (dtype::BFLOAT16, "bf16", 2e-2),
    (dtype::FLOAT16, "f16", 4e-3),
    (dtype::FLOAT32, "f32", 2e-3),
];

/// Deterministic values in `[-1, 1)`, with the magnitude stepped per group
/// of 32 so the E8M0 scales span several exponents instead of one.
fn synthetic(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0;
            let exponent = ((i / GROUP_SIZE) % 7) as i32 - 3;
            unit * 2f32.powi(exponent)
        })
        .collect()
}

fn to_f32_vec(arr: &MlxArray) -> Vec<f32> {
    let as_f32 = mlxcel_core::astype(arr, dtype::FLOAT32);
    mlxcel_core::eval(&as_f32);
    mlxcel_core::array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn to_u32_vec(arr: &MlxArray) -> Vec<u32> {
    let as_u32 = mlxcel_core::astype(arr, dtype::UINT32);
    mlxcel_core::eval(&as_u32);
    mlxcel_core::array_to_raw_bytes(&as_u32)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A quantized weight stack plus its host-decoded values.
struct QuantizedStack {
    packed: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    /// `[rows, IN_FEATURES]` row-major, the values the kernel must use.
    decoded: Vec<f32>,
}

/// Quantize `[..., rows, IN_FEATURES]` f32 weights with MLX and decode the
/// packed planes on the host.
fn quantize_stack(mode: Mode, shape: &[i32], seed: u64) -> QuantizedStack {
    let len: usize = shape.iter().map(|&d| d as usize).product();
    let source = mlxcel_core::from_slice_f32(&synthetic(len, seed), shape);
    let quantized = mlxcel_core::quantize_weights_with_mode(
        &source,
        GROUP_SIZE as i32,
        mode.bits() as i32,
        mode.name(),
    );
    assert!(
        !mlxcel_core::quantized_weights_has_biases(&quantized),
        "{} must not carry an affine bias plane",
        mode.name()
    );
    let packed = mlxcel_core::quantized_weights_w(&quantized);
    let scales = mlxcel_core::quantized_weights_scales(&quantized);
    mlxcel_core::eval(&packed);
    mlxcel_core::eval(&scales);
    assert_eq!(mlxcel_core::array_dtype(&scales), dtype::UINT8);

    let words = to_u32_vec(&packed);
    let exponents = to_u32_vec(&scales);
    let rows = len / IN_FEATURES;
    let words_per_row = IN_FEATURES * mode.bits() / 32;
    let groups_per_row = IN_FEATURES / GROUP_SIZE;
    assert_eq!(words.len(), rows * words_per_row);
    assert_eq!(exponents.len(), rows * groups_per_row);

    let mask = (1u32 << mode.bits()) - 1;
    let mut decoded = Vec::with_capacity(len);
    for row in 0..rows {
        for col in 0..IN_FEATURES {
            let bit = col * mode.bits();
            let word = words[row * words_per_row + bit / 32];
            let code = (word >> (bit % 32)) & mask;
            let exponent = exponents[row * groups_per_row + col / GROUP_SIZE] as i32;
            decoded.push(mode.decode(code) * 2f32.powi(exponent - 127));
        }
    }
    QuantizedStack {
        packed,
        scales,
        decoded,
    }
}

/// `x_row . w[n]` in f64 for every output row `n` of the row-major `w`.
fn project(x_row: &[f32], w: &[f32]) -> Vec<f64> {
    w.chunks_exact(x_row.len())
        .map(|w_row| {
            x_row
                .iter()
                .zip(w_row)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum()
        })
        .collect()
}

fn relative_l2(actual: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(actual.len(), reference.len());
    let mut err = 0.0f64;
    let mut norm = 0.0f64;
    for (&a, &r) in actual.iter().zip(reference) {
        err += (f64::from(a) - r).powi(2);
        norm += r * r;
    }
    (err / norm).sqrt()
}

fn assert_close(label: &str, actual: &[f32], reference: &[f64], tolerance: f64) {
    assert!(
        actual.iter().all(|v| v.is_finite()),
        "{label}: output holds non-finite values"
    );
    let error = relative_l2(actual, reference);
    assert!(
        error <= tolerance,
        "{label}: relative L2 error {error:.3e} exceeds {tolerance:.1e}"
    );
}

/// How the gathered batch was laid out, which decides the ROCm branch.
#[derive(Debug, Clone, Copy)]
enum GatherCase {
    /// `SwitchGLU` decode: one token, `top_k = 4`, unsorted.
    DecodeUnsorted,
    /// `SwitchGLU` prefill below the sort threshold: 6 tokens x 4 experts.
    PrefillUnsorted,
    /// `SwitchGLU` prefill above the sort threshold: 32 tokens x 4 experts,
    /// sorted through `gather_sort`, which selects the sorted rhs schedule.
    PrefillSorted,
    /// One shared activation row gathered against 32 sorted expert slots,
    /// four per expert, the sorted schedule with a zero activation stride.
    /// The slot count is load-bearing: the ROCm schedule needs `B >= 16` and
    /// `B / E >= 4`, so 16 slots over 8 experts would miss it.
    SortedSharedActivation,
    /// Four rows per gathered batch (`M = 4`).
    MultiRow,
}

/// Dense row counts: decode, a short verify block and a prefill chunk.
const DENSE_ROWS: [usize; 3] = [1, 4, 64];

const GATHER_CASES: [GatherCase; 5] = [
    GatherCase::DecodeUnsorted,
    GatherCase::PrefillUnsorted,
    GatherCase::PrefillSorted,
    GatherCase::SortedSharedActivation,
    GatherCase::MultiRow,
];

/// Deterministic expert choices, `top_k` distinct experts per token.
fn routing(tokens: usize, top_k: usize) -> Vec<u32> {
    (0..tokens)
        .flat_map(|t| (0..top_k).map(move |j| ((t * 3 + j * 5 + 1) % EXPERTS) as u32))
        .collect()
}

/// Run one gather case through `SwitchLinear::forward` and return the
/// output and the host reference, both flattened row-major.
fn run_gather(
    layer: &SwitchLinear,
    decoded: &[f32],
    case: GatherCase,
    act_dtype: i32,
    out_features: usize,
) -> (Vec<f32>, Vec<f64>) {
    let k = IN_FEATURES;
    // `slots_per_x` is how many consecutive gathered slots read one
    // activation batch entry: `top_k` for the unsorted layouts (MLX
    // broadcasts `[tokens, 1]` against `[tokens, top_k]`), 1 once
    // `gather_sort` has expanded the tokens, all 32 for the shared row.
    let (x, indices, sorted, slots_per_x) = match case {
        GatherCase::DecodeUnsorted | GatherCase::PrefillUnsorted | GatherCase::MultiRow => {
            let (tokens, top_k, rows) = match case {
                GatherCase::DecodeUnsorted => (1, 4, 1),
                GatherCase::PrefillUnsorted => (6, 4, 1),
                _ => (3, 2, 4),
            };
            let x = mlxcel_core::from_slice_f32(
                &synthetic(tokens * rows * k, 7),
                &[tokens as i32, 1, rows as i32, k as i32],
            );
            let indices = mlxcel_core::from_slice_u32(
                &routing(tokens, top_k),
                &[tokens as i32, top_k as i32],
            );
            (mlxcel_core::astype(&x, act_dtype), indices, false, top_k)
        }
        GatherCase::PrefillSorted => {
            let (tokens, top_k) = (32, 4);
            let x = mlxcel_core::from_slice_f32(
                &synthetic(tokens * k, 11),
                &[tokens as i32, 1, 1, k as i32],
            );
            let x = mlxcel_core::astype(&x, act_dtype);
            let indices = mlxcel_core::from_slice_u32(
                &routing(tokens, top_k),
                &[tokens as i32, top_k as i32],
            );
            let (sorted_x, sorted_idx, _inv) = gather_sort(&x, &indices);
            (sorted_x, sorted_idx, true, 1)
        }
        GatherCase::SortedSharedActivation => {
            let x = mlxcel_core::from_slice_f32(&synthetic(k, 13), &[1, 1, k as i32]);
            let slots: Vec<u32> = (0..32).map(|i| (i / 4) as u32).collect();
            let indices = mlxcel_core::from_slice_u32(&slots, &[32]);
            (mlxcel_core::astype(&x, act_dtype), indices, true, 32)
        }
    };

    let out = layer.forward(&x, &indices, sorted);
    mlxcel_core::eval(&out);
    assert_eq!(mlxcel_core::array_dtype(&out), act_dtype, "{case:?}");

    // The reference reads the activations back from the device, so it uses
    // exactly the values the kernel received (after the dtype cast and, for
    // the sorted case, after `gather_sort`'s reordering).
    let x_shape = mlxcel_core::array_shape(&x);
    let rows = x_shape[x_shape.len() - 2] as usize;
    let x_host = to_f32_vec(&x);
    let mut reference = Vec::new();
    for (b, &expert) in to_u32_vec(&indices).iter().enumerate() {
        for row in 0..rows {
            let x_row = &x_host[((b / slots_per_x) * rows + row) * k..][..k];
            let w = &decoded[expert as usize * out_features * k..][..out_features * k];
            reference.extend(project(x_row, w));
        }
    }
    (to_f32_vec(&out), reference)
}

fn switch_linear(mode: Mode, out_features: usize) -> (SwitchLinear, Vec<f32>) {
    let stack = quantize_stack(
        mode,
        &[EXPERTS as i32, out_features as i32, IN_FEATURES as i32],
        3,
    );
    let mut weights = WeightMap::new();
    weights.insert("moe.proj.weight".into(), stack.packed);
    weights.insert("moe.proj.scales".into(), stack.scales);
    let layer = SwitchLinear::from_weights_with_mode(
        &weights,
        "moe.proj",
        GROUP_SIZE as i32,
        mode.bits() as i32,
        mode.name(),
    )
    .unwrap_or_else(|err| panic!("{} expert stack must load: {err}", mode.name()));
    (layer, stack.decoded)
}

fn check_gather(
    mode: Mode,
    out_features: usize,
    cases: &[GatherCase],
    activations: &[(i32, &str, f64)],
) {
    let (layer, decoded) = switch_linear(mode, out_features);
    for &case in cases {
        for &(act_dtype, act_name, tolerance) in activations {
            let (out, reference) = run_gather(&layer, &decoded, case, act_dtype, out_features);
            assert_close(
                &format!("gather_qmm {} {case:?} {act_name}", mode.name()),
                &out,
                &reference,
                tolerance,
            );
        }
    }
}

/// The dense projection a requantized FP8 block checkpoint runs, at decode
/// (`M = 1`), a short verify block (`M = 4`) and a prefill chunk (`M = 64`).
fn check_dense(
    mode: Mode,
    out_features: usize,
    row_counts: &[usize],
    activations: &[(i32, &str, f64)],
) {
    let stack = quantize_stack(mode, &[out_features as i32, IN_FEATURES as i32], 5);
    let mut weights = WeightMap::new();
    weights.insert("proj.weight".into(), stack.packed);
    weights.insert("proj.scales".into(), stack.scales);
    // `from_weights` infers the mode from the absent `.biases` plane, the
    // same route the Qwen3.5 loader takes for a requantized checkpoint.
    let layer =
        UnifiedLinear::from_weights(&weights, "proj", GROUP_SIZE as i32, mode.bits() as i32)
            .unwrap_or_else(|err| panic!("{} projection must load: {err}", mode.name()));
    let k = IN_FEATURES;
    for &rows in row_counts {
        let source =
            mlxcel_core::from_slice_f32(&synthetic(rows * k, 17), &[1, rows as i32, k as i32]);
        for &(act_dtype, act_name, tolerance) in activations {
            let x = mlxcel_core::astype(&source, act_dtype);
            let out = layer.forward(&x);
            mlxcel_core::eval(&out);
            let x_host = to_f32_vec(&x);
            let reference: Vec<f64> = (0..rows)
                .flat_map(|r| project(&x_host[r * k..][..k], &stack.decoded))
                .collect();
            assert_close(
                &format!("quantized_matmul {} M={rows} {act_name}", mode.name()),
                &to_f32_vec(&out),
                &reference,
                tolerance,
            );
        }
    }
}

/// The host decoder is the reference, so pin it to MLX's own `dequantize`
/// first: a layout mistake here would otherwise make every check below
/// compare against the wrong numbers.
#[test]
fn mxfp_host_decode_matches_mlx_dequantize() {
    let _device = lock_default_device();
    for mode in [Mode::Mxfp8, Mode::Mxfp4] {
        let stack = quantize_stack(mode, &[OUT_FEATURES as i32, IN_FEATURES as i32], 19);
        // SAFETY: `packed` and `scales` are live arrays from `quantize_stack`,
        // and a null `biases` pointer is what `dequantize` documents for the
        // bias-free block-float modes (`quantize_stack` asserted there is no
        // bias plane).
        let dequantized = unsafe {
            mlxcel_core::dequantize(
                &stack.packed,
                &stack.scales,
                std::ptr::null(),
                GROUP_SIZE as i32,
                mode.bits() as i32,
                mode.name(),
            )
        };
        let device_values = to_f32_vec(&dequantized);
        assert_eq!(device_values.len(), stack.decoded.len());
        let mismatches = device_values
            .iter()
            .zip(&stack.decoded)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "{}: host decode disagrees with dequantize",
            mode.name()
        );
    }
}

#[test]
fn mxfp8_gather_qmm_matches_host_reference() {
    let _device = lock_default_device();
    check_gather(Mode::Mxfp8, OUT_FEATURES, &GATHER_CASES, &ACTIVATIONS);
}

#[test]
fn mxfp4_gather_qmm_matches_host_reference() {
    let _device = lock_default_device();
    check_gather(Mode::Mxfp4, OUT_FEATURES, &GATHER_CASES, &ACTIVATIONS);
}

#[test]
fn mxfp8_quantized_matmul_matches_host_reference() {
    let _device = lock_default_device();
    check_dense(Mode::Mxfp8, OUT_FEATURES, &DENSE_ROWS, &ACTIVATIONS);
}

/// A reduced matrix on MLX's CPU backend: evidence that the tolerances are
/// what a correct implementation meets, not a bound fitted to one GPU.
///
/// MLX's CPU `fp_qmm_t` is a scalar loop, so the full matrix took about
/// 38 s under `--profile test-fast` while holding the default-device lock
/// that serializes other tests, on every backend. The CPU arm keeps the
/// cases that decide the tolerance and drops the ones that only add
/// runtime: both modes, one unsorted and one sorted gather case (with
/// the activation stride at 0), the multi-row gather case, and the dense
/// decode and short-verify shapes, in bf16 (the loosest bound) and f16, on
/// a 64-wide output instead of 320.
/// f32 is left to the GPU arms, where it is exact to 3e-7. The `M = 64`
/// dense case and the large sorted prefill are the slowest and add no
/// new kernel branch on the CPU. The GPU arms above run the full matrix.
#[test]
fn mxfp_matmuls_match_host_reference_on_cpu_device() {
    let _device = lock_default_device();
    let _cpu = DefaultDeviceGuard::cpu();
    let cases = [
        GatherCase::PrefillUnsorted,
        GatherCase::SortedSharedActivation,
        GatherCase::MultiRow,
    ];
    let activations = [ACTIVATIONS[0], ACTIVATIONS[1]];
    check_gather(Mode::Mxfp8, CPU_OUT_FEATURES, &cases, &activations);
    check_gather(Mode::Mxfp4, CPU_OUT_FEATURES, &cases, &activations);
    check_dense(Mode::Mxfp8, CPU_OUT_FEATURES, &[1, 4], &activations);
}
