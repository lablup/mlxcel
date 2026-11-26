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

//! The #2832 numerical matrix through the MLX adapter (#2833).
//!
//! 130 cases (26 per dtype pair, 2,181,390 elements): widths 1 to 8192
//! including odd and tail widths, 1 to 33 rows, the Gemma `weight_bias = 1`
//! fold, `eps = 1e-5`, and one NaN / +Inf case per pair. Every case runs three
//! ways on the same input bytes: the cuda-oxide node inside MLX, the existing
//! fused CUDA kernel, and the `add` + `fast_rms_norm` graph.
//!
//! The pass criteria are the ones #2832 froze before its oracle ran, the
//! budgets of `fused_norm_parity_tests::tolerance_for`; bitwise identity to
//! the existing kernel is counted and printed, not required. The route
//! counters are asserted around each phase, so the comparison cannot pass by
//! the launcher quietly taking the existing kernel on the cuda-oxide pass.
//!
//!     MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 cargo test --profile test-fast \
//!         -p mlxcel-core --lib --features cuda cuda_oxide_fused_norm \
//!         -- --test-threads=1 --nocapture

use crate::cuda_oxide_artifact::Lane;
use crate::cuda_oxide_fused_norm as oxide;
use crate::cuda_oxide_fused_norm_fixtures::{
    Case, OutBits, RouteGuard, case_arrays, case_matrix, decode, delta, inputs, run_fused,
    run_graph,
};
use crate::fused_norm_parity_tests::tolerance_for;

const EXPECTED_CASES: usize = 130;
const EXPECTED_ELEMENTS: usize = 2_181_390;

/// Normalized RMS and max deviation of `got` from `want` over finite rows,
/// both divided by the RMS of `want`, plus the number of differing patterns.
fn compare(
    got: &[u32],
    want: &[u32],
    lane: Lane,
    dim: usize,
    skip_rows: &[bool],
) -> (f64, f64, usize) {
    assert_eq!(got.len(), want.len(), "output length");
    let (mut sq, mut max, mut ref_sq) = (0f64, 0f64, 0f64);
    let (mut n, mut differing) = (0usize, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if skip_rows[i / dim] {
            continue;
        }
        if g != w {
            differing += 1;
        }
        let (gv, wv) = (f64::from(decode(lane, *g)), f64::from(decode(lane, *w)));
        sq += (gv - wv) * (gv - wv);
        max = max.max((gv - wv).abs());
        ref_sq += wv * wv;
        n += 1;
    }
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let ref_rms = (ref_sq / n as f64).sqrt().max(1e-20);
    ((sq / n as f64).sqrt() / ref_rms, max / ref_rms, differing)
}

/// NaN with NaN, an infinity with the same-signed infinity, finite with finite
/// bit for bit.
fn same_class_exact(g: f32, w: f32) -> bool {
    if g.is_nan() || w.is_nan() {
        g.is_nan() && w.is_nan()
    } else if g.is_infinite() || w.is_infinite() {
        g == w
    } else {
        g.to_bits() == w.to_bits()
    }
}

/// Whether the non-finite rows of `got` agree with `want` by class.
fn nonfinite_rows_agree(got: &[u32], want: &[u32], lane: Lane, dim: usize, rows: &[bool]) -> bool {
    got.iter()
        .zip(want)
        .enumerate()
        .filter(|(i, _)| rows[i / dim])
        .all(|(_, (g, w))| same_class_exact(decode(lane, *g), decode(lane, *w)))
}

struct Reference {
    case: Case,
    nonfinite_rows: Vec<bool>,
    fused: OutBits,
    graph: OutBits,
}

#[test]
fn the_2832_matrix_passes_through_the_adapter_on_the_oxide_route() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    let cases = case_matrix();
    assert_eq!(cases.len(), EXPECTED_CASES, "the matrix must be complete");
    assert_eq!(
        cases.iter().map(Case::elements).sum::<usize>(),
        EXPECTED_ELEMENTS,
        "the matrix must be the #2832 matrix"
    );

    // Pass 1, nothing installed: the existing fused kernel and the graph path.
    let before = oxide::route_counters();
    let references: Vec<Reference> = cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let inp = inputs(case, index);
            let arrays = case_arrays(case, &inp);
            Reference {
                case: case.clone(),
                nonfinite_rows: inp.nonfinite_rows(case),
                fused: run_fused(case, &arrays),
                graph: run_graph(case, &arrays),
            }
        })
        .collect();
    let reference_pass = delta(before);
    assert_eq!(
        (reference_pass.oxide_routes, reference_pass.launches),
        (0, 0),
        "the reference pass must not touch the cuda-oxide route"
    );
    assert_eq!(reference_pass.native_routes, EXPECTED_CASES as u64);
    assert_eq!(oxide::last_fallback(), oxide::Fallback::NotInstalled);

    // Pass 2, the embedded artifact installed: the same inputs, rebuilt from
    // the same seeds, through the cuda-oxide node.
    oxide::install_embedded().expect("the embedded artifact validates and targets this device");
    let before = oxide::route_counters();
    let mut failures = Vec::new();
    let (mut bitwise_fused, mut elements, mut nonfinite_cases) = (0usize, 0usize, 0usize);
    for (index, reference) in references.iter().enumerate() {
        let case = &reference.case;
        let inp = inputs(case, index);
        let arrays = case_arrays(case, &inp);
        let step = oxide::route_counters();
        let ours = run_fused(case, &arrays);
        let moved = delta(step);
        assert_eq!(
            (moved.oxide_routes, moved.launches, moved.native_routes),
            (1, 1, 0),
            "{}: the launcher must take the cuda-oxide route and launch once",
            case.name
        );
        assert_eq!(oxide::last_fallback(), oxide::Fallback::Taken);

        let skip = &reference.nonfinite_rows;
        let (dim, lane) = (case.dim, case.act);
        let (f_rms, f_max, f_diff) =
            compare(&ours.normed, &reference.fused.normed, lane, dim, skip);
        let (g_rms, g_max, _) = compare(&ours.normed, &reference.graph.normed, lane, dim, skip);
        let (_, _, res_f_diff) = compare(
            &ours.new_residual,
            &reference.fused.new_residual,
            lane,
            dim,
            skip,
        );
        let (_, _, res_g_diff) = compare(
            &ours.new_residual,
            &reference.graph.new_residual,
            lane,
            dim,
            skip,
        );
        let nonfinite_ok = [
            (&ours.normed, &reference.fused.normed),
            (&ours.normed, &reference.graph.normed),
            (&ours.new_residual, &reference.fused.new_residual),
            (&ours.new_residual, &reference.graph.new_residual),
        ]
        .iter()
        .all(|(got, want)| nonfinite_rows_agree(got, want, lane, dim, skip));

        let (rms_tol, max_tol) = tolerance_for(lane.mlx_dtype());
        let compared = skip.iter().filter(|s| !**s).count() * dim;
        let residual_budget = if lane == Lane::F32 { 0 } else { compared / 100 };
        let pass = f_rms < rms_tol
            && f_max < max_tol
            && g_rms < rms_tol
            && g_max < max_tol
            && res_f_diff == 0
            && res_g_diff <= residual_budget
            && nonfinite_ok;
        if !pass {
            failures.push(format!(
                "{}: fused rms {f_rms:.3e} max {f_max:.3e}, graph rms {g_rms:.3e} max {g_max:.3e} (tol {rms_tol:.1e}/{max_tol:.1e}), residual differs from fused in {res_f_diff}, from graph in {res_g_diff} (budget {residual_budget}), non-finite rows agree: {nonfinite_ok}",
                case.name
            ));
        }
        if f_diff == 0 && res_f_diff == 0 && nonfinite_ok {
            bitwise_fused += 1;
        }
        if skip.iter().any(|s| *s) {
            nonfinite_cases += 1;
        }
        elements += case.elements();
    }
    let oxide_pass = delta(before);

    println!(
        "RESULT oxide-adapter-matrix cases={} elements={} failed={} bitwise_identical_to_fused_cuda={} nonfinite_cases={} oxide_routes={} launches={} native_routes={}",
        references.len(),
        elements,
        failures.len(),
        bitwise_fused,
        nonfinite_cases,
        oxide_pass.oxide_routes,
        oxide_pass.launches,
        oxide_pass.native_routes
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(elements, EXPECTED_ELEMENTS);
    assert_eq!(nonfinite_cases, 5, "one NaN / +Inf case per dtype pair");
    assert_eq!(
        (
            oxide_pass.oxide_routes,
            oxide_pass.launches,
            oxide_pass.native_routes
        ),
        (EXPECTED_CASES as u64, EXPECTED_CASES as u64, 0),
        "every case must run on the cuda-oxide route, none on the existing kernel"
    );
}
