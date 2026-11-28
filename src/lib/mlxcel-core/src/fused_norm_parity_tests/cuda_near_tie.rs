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

//! What the CUDA fused kernel and the CUDA graph agree on, and what they do
//! not (issue #3501).
//!
//! On CUDA the two are not byte-identical, unlike the ROCm port. MLX's
//! `rms_norm` at bf16 dim 2048 launches `rms_norm_small` with a 128-thread
//! block and two 8-wide chunks per thread (four warp partials; its `ALIGNED`
//! argument was not observed and selects only the load function, not the
//! summation order); the fused kernel
//! launches 512 threads of 4 elements and folds 16 warp partials. Both use the
//! same xor-butterfly warp reduction, but the different partition sums the
//! squares in a different fp32 order, so the normalizer can differ in its last
//! bits. Where it straddles a rounding boundary of `bf16(x * inv)`, an element
//! moves one bf16 ulp, and the gain multiply can widen that to two at the
//! output. On the Llama 3.2 1B prefill that issue measured, two of 2048 rows
//! differed (19 and 27 elements), each side matching the float64 reference on
//! one of them.
//!
//! So the pin is structural rather than a tolerance: `new_residual` is
//! byte-identical, and every `normed` row of either side is reproduced exactly
//! by the graph's formula with one fp32 normalizer a few ulps from the exact
//! one. A rounding-order or gain defect moves elements independently of any
//! single per-row scalar, so it fails here (rounding `w * x` before the
//! normalizer instead of after was checked to fail), while a reduction-order
//! difference passes. An eps error small enough to hide inside the 32-ulp
//! search window is below bf16 resolution by construction. The test also
//! asserts that some rows do differ, so the day the kernel matches MLX's order
//! it fails and asks for the ROCm-style byte-identity pin instead. Making the
//! kernel byte-identical means copying MLX's
//! launch geometry and chunked loads, and the cuda-oxide artifact is a bitwise
//! port of the current kernel, so that change also needs the artifact
//! regenerated.

use super::*;

/// Round an f32 to bf16, nearest with ties to even.
fn round_bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

/// Order key of a bf16 value held in an f32: its distance from another key is
/// the distance in bf16 ulps.
fn bf16_key(v: f32) -> i64 {
    let bits = i64::from(v.to_bits() >> 16);
    if bits & 0x8000 != 0 {
        -(bits & 0x7fff)
    } else {
        bits
    }
}

/// Normalizer offsets searched around the fp32 rounding of the exact one.
const NORMALIZER_ULPS: i32 = 32;
/// Seeds of 4096 rows each. More seeds make a chance-zero count less likely.
const SEEDS: u64 = 6;

/// Whether some fp32 normalizer within [`NORMALIZER_ULPS`] of the exact one
/// reproduces `got` from `row` and `weight` by the graph's formula,
/// `bf16(w * bf16(x * inv))`.
fn row_fits_one_normalizer(row: &[f32], weight: &[f32], got: &[f32]) -> bool {
    let mean_sq = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / row.len() as f64;
    let exact = (1.0 / (mean_sq + f64::from(EPS)).sqrt()) as f32;
    let reproduces = |inv: f32| {
        row.iter()
            .zip(weight)
            .zip(got)
            .all(|((x, w), g)| round_bf16(w * round_bf16(x * inv)).to_bits() == g.to_bits())
    };
    let (mut up, mut down) = (exact, exact);
    for _ in 0..=NORMALIZER_ULPS {
        if reproduces(up) || reproduces(down) {
            return true;
        }
        up = f32::from_bits(up.to_bits() + 1);
        down = f32::from_bits(down.to_bits() - 1);
    }
    false
}

#[test]
fn cuda_fused_add_rms_norm_differs_from_the_graph_only_by_its_normalizer() {
    use crate::hardware::{GpuBackendKind, gpu_backend_kind};
    if gpu_backend_kind() != GpuBackendKind::Cuda {
        return;
    }
    let Some(_device) = gpu_kernel_or_skip() else {
        return;
    };
    let (rows, dim) = (4096, 2048);
    let mut differing_rows = 0usize;
    for seed in 0..SEEDS {
        let (delta, residual, weight) = random_case(7000 + seed, rows, dim, dtype::BFLOAT16);
        let (normed, new_residual) = run_fused(&delta, &residual, &weight, EPS, 0.0);
        let norm = RMSNorm::new(copy(&weight), EPS);
        let (want_normed, want_residual) =
            crate::layers::graph_add_rms_norm(&norm, &delta, &residual);
        assert_eq!(
            raw_bytes(&new_residual),
            raw_bytes(&want_residual),
            "seed {seed}: new_residual differs from the CUDA graph add"
        );
        let x = flatten_f32(&want_residual);
        let w = flatten_f32(&weight);
        let fused = flatten_f32(&normed);
        let graph = flatten_f32(&want_normed);
        let width = dim as usize;
        for r in 0..rows as usize {
            let span = r * width..(r + 1) * width;
            if fused[span.clone()] == graph[span.clone()] {
                continue;
            }
            differing_rows += 1;
            // One bf16 ulp of the intermediate `bf16(x * inv)`, which the gain
            // multiply can carry to two at the output (#3501 measured both).
            let max_ulp = fused[span.clone()]
                .iter()
                .zip(&graph[span.clone()])
                .map(|(a, b)| (bf16_key(*a) - bf16_key(*b)).abs())
                .max()
                .unwrap_or(0);
            assert!(
                max_ulp <= 2,
                "seed {seed} row {r}: normed differs from the graph by {max_ulp} bf16 ulps"
            );
            for (side, out) in [("fused", &fused), ("graph", &graph)] {
                assert!(
                    row_fits_one_normalizer(&x[span.clone()], &w, &out[span.clone()]),
                    "seed {seed} row {r}: the {side} row is not the graph formula at any \
                     normalizer within {NORMALIZER_ULPS} fp32 ulps of the exact one"
                );
            }
        }
    }
    eprintln!(
        "bf16 dim {dim}: {differing_rows} of {} rows differ by normalizer rounding",
        SEEDS as usize * rows as usize
    );
    // Measured on GB10 (6 seeds: 5 of 24576 rows; 4 of 16384 with the first four). Zero is possible by chance
    // if MLX's reduction order moves, so a failure here is a prompt to look,
    // not proof of byte identity.
    assert!(
        differing_rows > 0,
        "no row differs from the graph on these inputs. Before concluding the CUDA fused \
         kernel now matches MLX, check MLX's dim-2048 dispatch (`dispatch_num_chunks` block \
         size and chunk count in rms_norm.cu) against `fused_norm_threads`; only if the \
         geometry matches, replace this test with a byte-identity pin like the ROCm one"
    );
}
