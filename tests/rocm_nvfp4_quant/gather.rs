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

//! NVFP4 `gather_qmm` against a per-expert dequantized reference, with and
//! without MLX's per-expert global scale (issue #2617).
//!
//! The layouts are the ones `SwitchLinear::forward` passes: unsorted decode
//! (`x` as `[T, 1, 1, K]`, indices `[T, top_k]`), sorted prefill (`x` as
//! `[T * top_k, 1, K]`, indices sorted by expert, `B = 128 >= 64` so eight
//! rows share a block), and an activation with three rows per expert (`x` as
//! `[T, 1, 3, K]`, also eight rows a block). With a global scale each expert
//! is quantized relative to its own `global_scale[e]`, which the kernel
//! applies as `global_scale[e] / (448 * 6)`.
//!
//! A sorted gather with `M == 1` and `B >= 64` takes the expert-batched
//! kernel (`nvfp4_gather_expert_batched_kernel`, issue #2665), so the sorted
//! prefill case and the sorted out-of-range case above run it, and
//! `sorted_prefill_over_128_experts_matches_per_expert_reference` runs it at
//! the expert count of the Nemotron-H checkpoints it was measured on.

use super::*;

const EXPERTS: i32 = 16;
const N: i32 = 256;
const K: i32 = 1024;
const TOP_K: i32 = 4;

/// An NVFP4 expert stack `[EXPERTS, N, K]` and its f32 dequantization,
/// global scale included.
struct Experts {
    packed: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    global_scale: Option<UniquePtr<MlxArray>>,
    dense: UniquePtr<MlxArray>,
}

fn experts(with_global_scale: bool) -> Experts {
    experts_of(EXPERTS, with_global_scale)
}

/// [`experts`] with `count` experts.
fn experts_of(count: i32, with_global_scale: bool) -> Experts {
    experts_shaped(count, N, with_global_scale)
}

/// [`experts`] with `count` experts of `n` output columns.
fn experts_shaped(count: i32, n: i32, with_global_scale: bool) -> Experts {
    let w = normal(&[count, n, K], dtype::FLOAT32);
    if !with_global_scale {
        let (packed, scales) = quantize(&w, None);
        let dense = dequantize_cpu(&packed, &scales);
        assert_reconstructs("expert stack", &dense, &w);
        return Experts {
            packed,
            scales,
            global_scale: None,
            dense,
        };
    }
    // Each expert relative to its own scale, at least its largest magnitude
    // (MLX requires every encoded block scale to fit E4M3).
    let mut gs = Vec::new();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    for e in 0..count {
        let slice = {
            let _guard = DefaultDeviceGuard::cpu();
            cpu_eval(
                "expert slice",
                mlxcel_core::copy(&mlxcel_core::slice(&w, &[e, 0, 0], &[e + 1, n, K])),
            )
        };
        let amax = {
            let _guard = DefaultDeviceGuard::cpu();
            let a = cpu_eval("amax", mlxcel_core::max_all(&mlxcel_core::abs(&slice)));
            mlxcel_core::item_f32(&a)
        };
        let value = amax * (1.0 + 0.5 * (e % 3) as f32);
        let (p, s) = quantize(&slice, Some(&mlxcel_core::from_slice_f32(&[value], &[1])));
        gs.push(value);
        packed.push(p);
        scales.push(s);
    }
    let _guard = DefaultDeviceGuard::cpu();
    let cat = |parts: &[UniquePtr<MlxArray>]| {
        let refs: Vec<&MlxArray> = parts.iter().map(|p| &**p).collect();
        cpu_eval("stack", mlxcel_core::concatenate_many(&refs, 0))
    };
    let (packed, scales) = (cat(&packed), cat(&scales));
    let factors: Vec<f32> = gs.iter().map(|g| g / GLOBAL_SCALE_DENOM).collect();
    let factors = mlxcel_core::from_slice_f32(&factors, &[count, 1, 1]);
    let dense = cpu_eval(
        "scaled dense",
        mlxcel_core::multiply(&dequantize_cpu(&packed, &scales), &factors),
    );
    assert_reconstructs("global-scaled expert stack", &dense, &w);
    Experts {
        packed,
        scales,
        global_scale: Some(mlxcel_core::from_slice_f32(&gs, &[count])),
        dense,
    }
}

/// Row `r` of `x_rows` (`[R, M, K]`) times expert `expert_of[r]`, `[R, M, N]`,
/// in f32 on the CPU, in chunks of 32 rows to keep the gathered experts small.
fn reference(x_rows: &MlxArray, dense: &MlxArray, expert_of: &[i32]) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::cpu();
    let shape = mlxcel_core::array_shape(x_rows);
    let (rows, m) = (shape[0], shape[1]);
    let parts: Vec<UniquePtr<MlxArray>> = (0..rows)
        .step_by(32)
        .map(|s| {
            let e = (s + 32).min(rows);
            let idx = mlxcel_core::from_slice_i32(&expert_of[s as usize..e as usize], &[e - s]);
            let picked = mlxcel_core::take(dense, &idx, 0);
            let xs = mlxcel_core::slice(x_rows, &[s, 0, 0], &[e, m, K]);
            let xs = mlxcel_core::astype(&xs, dtype::FLOAT32);
            cpu_eval(
                "reference chunk",
                mlxcel_core::matmul(&xs, &mlxcel_core::swap_axes(&picked, -1, -2)),
            )
        })
        .collect();
    let refs: Vec<&MlxArray> = parts.iter().map(|p| &**p).collect();
    cpu_eval("reference", mlxcel_core::concatenate_many(&refs, 0))
}

/// One gather case: `tokens` tokens of `m` rows each, routed to `TOP_K`
/// experts, sorted (one input row per output row, by expert) or not.
fn run_case(ex: &Experts, dt: i32, tokens: i32, m: i32, sorted: bool) {
    let label = format!(
        "gather_qmm T={tokens} M={m} sorted={sorted} global_scale={} {}",
        ex.global_scale.is_some(),
        dtype_name(dt)
    );
    let rows = tokens * TOP_K;
    let indices: Vec<i32> = (0..rows).map(|i| (i * 7 + 3) % EXPERTS).collect();
    let x = normal(&[tokens, 1, m, K], dt);
    let mut order: Vec<i32> = (0..rows).collect();
    if sorted {
        order.sort_by_key(|&r| indices[r as usize]);
    }
    let expert_of: Vec<i32> = order.iter().map(|&r| indices[r as usize]).collect();
    // One input row block per output row, in output order.
    let x_rows = {
        let _guard = DefaultDeviceGuard::cpu();
        let token_of: Vec<i32> = order.iter().map(|r| r / TOP_K).collect();
        let tok = mlxcel_core::from_slice_i32(&token_of, &[rows]);
        let flat = mlxcel_core::reshape(&x, &[tokens, m, K]);
        cpu_eval("input rows", mlxcel_core::take(&flat, &tok, 0))
    };
    let (x_in, rhs) = {
        let _guard = DefaultDeviceGuard::cpu();
        let (x_in, flat, shape) = if sorted {
            (mlxcel_core::copy(&x_rows), &expert_of, vec![rows])
        } else {
            (mlxcel_core::copy(&x), &indices, vec![tokens, TOP_K])
        };
        let rhs = mlxcel_core::astype(&mlxcel_core::from_slice_i32(flat, &shape), dtype::UINT32);
        (cpu_eval("inputs", x_in), cpu_eval("indices", rhs))
    };
    let got = {
        let _guard = DefaultDeviceGuard::gpu();
        let gs_ptr = ex
            .global_scale
            .as_ref()
            .map_or(std::ptr::null(), |g| &**g as *const MlxArray);
        // SAFETY: `rhs` and the global scale outlive the call; null is the
        // bridge's documented value for absent biases and lhs indices.
        let y = unsafe {
            mlxcel_core::gather_qmm(
                &x_in,
                &ex.packed,
                &ex.scales,
                std::ptr::null(),
                std::ptr::null(),
                &*rhs as *const MlxArray,
                true,
                GROUP_SIZE,
                BITS,
                sorted,
                MODE,
                gs_ptr,
            )
        };
        eval_ok(&label, &y);
        y
    };
    assert_eq!(mlxcel_core::array_dtype(&got), dt, "{label}: dtype");
    let got = {
        let _guard = DefaultDeviceGuard::cpu();
        cpu_eval("reshape", mlxcel_core::reshape(&got, &[rows, m, N]))
    };
    check(&label, dt, &got, &reference(&x_rows, &ex.dense, &expert_of));
}

#[test]
fn gather_qmm_matches_per_expert_reference() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(26173);
    for with_global_scale in [false, true] {
        let ex = experts(with_global_scale);
        for dt in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
            for (tokens, m, sorted) in [(1, 1, false), (8, 1, false), (32, 1, true), (4, 3, false)]
            {
                run_case(&ex, dt, tokens, m, sorted);
            }
        }
    }
}

/// An expert index at or past the expert count selects no weights: the row is
/// written as zeros and nothing is read out of bounds, as the other ROCm
/// gather kernels do. Unsorted (one row a block) and sorted (`B = 128`, eight
/// rows a block, the out-of-range rows at the end of the last runs) layouts,
/// with the global scale so its per-expert read is covered too.
#[test]
fn gather_qmm_zeroes_rows_routed_past_the_last_expert() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(26174);
    let ex = experts(true);
    let rows: i32 = 128;
    let past_end = [EXPERTS, EXPERTS + 1, i32::MAX];
    for dt in [dtype::FLOAT32, dtype::BFLOAT16] {
        for sorted in [false, true] {
            let label = format!(
                "gather_qmm out-of-range experts sorted={sorted} {}",
                dtype_name(dt)
            );
            let mut expert_of: Vec<i32> = (0..rows)
                .map(|i| {
                    if i % 5 == 2 {
                        past_end[(i % 3) as usize]
                    } else {
                        (i * 7 + 3) % EXPERTS
                    }
                })
                .collect();
            if sorted {
                expert_of.sort_unstable();
            }
            let in_range: Vec<f32> = expert_of
                .iter()
                .map(|&e| if e < EXPERTS { 1.0 } else { 0.0 })
                .collect();
            let x = normal(&[rows, 1, K], dt);
            let rhs = {
                let _guard = DefaultDeviceGuard::cpu();
                cpu_eval(
                    "indices",
                    mlxcel_core::astype(
                        &mlxcel_core::from_slice_i32(&expert_of, &[rows]),
                        dtype::UINT32,
                    ),
                )
            };
            let got = {
                let _guard = DefaultDeviceGuard::gpu();
                let gs = ex.global_scale.as_ref().expect("global scale");
                // SAFETY: `rhs` and the global scale outlive the call; null is
                // the bridge's documented value for absent biases and lhs
                // indices.
                let y = unsafe {
                    mlxcel_core::gather_qmm(
                        &x,
                        &ex.packed,
                        &ex.scales,
                        std::ptr::null(),
                        std::ptr::null(),
                        &*rhs as *const MlxArray,
                        true,
                        GROUP_SIZE,
                        BITS,
                        sorted,
                        MODE,
                        &**gs as *const MlxArray,
                    )
                };
                eval_ok(&label, &y);
                y
            };
            let _guard = DefaultDeviceGuard::cpu();
            let got = cpu_eval(
                "reshape",
                mlxcel_core::astype(&mlxcel_core::reshape(&got, &[rows, 1, N]), dtype::FLOAT32),
            );
            let clamped: Vec<i32> = expert_of.iter().map(|&e| e.min(EXPERTS - 1)).collect();
            let mask = mlxcel_core::from_slice_f32(&in_range, &[rows, 1, 1]);
            let want = cpu_eval(
                "masked reference",
                mlxcel_core::multiply(&reference(&x, &ex.dense, &clamped), &mask),
            );
            check(&label, dt, &got, &want);
            let out_of_range: Vec<f32> = in_range.iter().map(|v| 1.0 - v).collect();
            let leak = cpu_eval(
                "out-of-range rows",
                mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::multiply(
                    &got,
                    &mlxcel_core::from_slice_f32(&out_of_range, &[rows, 1, 1]),
                ))),
            );
            assert_eq!(
                mlxcel_core::item_f32(&leak),
                0.0,
                "{label}: a row routed past the last expert must be zero"
            );
        }
    }
}

/// The sorted prefill route at 128 experts (issue #2665): `B = 512` rows in
/// runs of uneven length (expert 5 holds a run longer than one row chunk,
/// most experts a few rows, some none), with and without the per-expert
/// global scale, for f32, f16 and bf16. Each output row checks both the
/// expert whose weights the block read and that expert's global scale, so a
/// wrong expert offset or a dropped or misindexed global scale fails here.
#[test]
fn sorted_prefill_over_128_experts_matches_per_expert_reference() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(26175);
    const E: i32 = 128;
    const ROWS: i32 = 512;
    let mut expert_of: Vec<i32> = (0..ROWS)
        .map(|r| {
            if r % 9 == 0 {
                5
            } else {
                (r * r * 7 + 3 * r + 1) % E
            }
        })
        .collect();
    expert_of.sort_unstable();
    // N = 250 leaves a partial group of four columns (250 = 62 * 4 + 2) and
    // a partial block of 64, so the kernel's column clamp and store guard
    // are exercised.
    for (n, with_global_scale) in [(N, false), (N, true), (250, true)] {
        let ex = experts_shaped(E, n, with_global_scale);
        for dt in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
            let label = format!(
                "gather_qmm sorted prefill E={E} B={ROWS} N={n} global_scale={with_global_scale} {}",
                dtype_name(dt)
            );
            let x = normal(&[ROWS, 1, K], dt);
            let rhs = {
                let _guard = DefaultDeviceGuard::cpu();
                cpu_eval(
                    "indices",
                    mlxcel_core::astype(
                        &mlxcel_core::from_slice_i32(&expert_of, &[ROWS]),
                        dtype::UINT32,
                    ),
                )
            };
            let got = {
                let _guard = DefaultDeviceGuard::gpu();
                let gs_ptr = ex
                    .global_scale
                    .as_ref()
                    .map_or(std::ptr::null(), |g| &**g as *const MlxArray);
                // SAFETY: `rhs` and the global scale outlive the call; null is
                // the bridge's documented value for absent biases and lhs
                // indices.
                let y = unsafe {
                    mlxcel_core::gather_qmm(
                        &x,
                        &ex.packed,
                        &ex.scales,
                        std::ptr::null(),
                        std::ptr::null(),
                        &*rhs as *const MlxArray,
                        true,
                        GROUP_SIZE,
                        BITS,
                        true,
                        MODE,
                        gs_ptr,
                    )
                };
                eval_ok(&label, &y);
                y
            };
            assert_eq!(mlxcel_core::array_dtype(&got), dt, "{label}: dtype");
            let got = {
                let _guard = DefaultDeviceGuard::cpu();
                cpu_eval("reshape", mlxcel_core::reshape(&got, &[ROWS, 1, n]))
            };
            check(&label, dt, &got, &reference(&x, &ex.dense, &expert_of));
        }
    }
}
