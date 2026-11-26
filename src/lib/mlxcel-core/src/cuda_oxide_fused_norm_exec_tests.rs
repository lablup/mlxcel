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

//! The cuda-oxide node inside MLX lazy evaluation (#2833).
//!
//! The numerical matrix says the kernel computes the right values when it is
//! evaluated alone. These tests are about everything around it that MLX owns:
//! that nothing is launched when the graph is built, that the node orders
//! correctly against its producers and consumers (including cuBLAS matmuls,
//! which MLX captures as child graphs), that a graph replayed many times and
//! buffers recycled by the allocator still give the same bytes, that inputs
//! and sibling outputs dropped by the caller stay alive until the launch, and
//! that several threads can drive it at once.
//!
//! Each test computes its reference with the existing fused CUDA kernel on the
//! same inputs and requires identical bytes, which #2832 measured for every
//! case of its matrix on this device, and asserts through the launch counter
//! that the cuda-oxide kernel really ran.
//!
//! `MLX_USE_CUDA_GRAPHS` is read once per process, so graph capture on and off
//! are two runs of this file, not two tests in it:
//!
//!     ... cargo test ... cuda_oxide_fused_norm -- --test-threads=1
//!     MLX_USE_CUDA_GRAPHS=0 ... cargo test ... cuda_oxide_fused_norm -- --test-threads=1

use crate::cuda_oxide_artifact::Lane;
use crate::cuda_oxide_fused_norm as oxide;
use crate::cuda_oxide_fused_norm_fixtures::{
    Case, CaseArrays, Inputs, OutBits, RouteGuard, bits_of, case_arrays, delta, fused_lazy, inputs,
    run_fused,
};
use crate::{MlxArray, UniquePtr, add, astype, eval, from_slice_f32, matmul};

fn case(act: Lane, w: Lane, rows: usize, dim: usize) -> Case {
    Case::new(act, w, rows, dim, 1e-6, 0.0, false)
}

/// A `[dim, dim]` f32 mixing matrix with entries of order `1 / sqrt(dim)`, so
/// a matmul through it keeps activations at unit scale.
fn mixing_matrix(dim: usize, seed: usize) -> UniquePtr<MlxArray> {
    let probe = Case::new(Lane::F32, Lane::F32, dim, dim, 1e-6, 0.0, false);
    let scale = 1.0 / (dim as f32).sqrt();
    let values: Vec<f32> = inputs(&probe, seed).x.iter().map(|v| v * scale).collect();
    let m = from_slice_f32(&values, &[dim as i32, dim as i32]);
    eval(&m);
    m
}

/// `fused -> add -> fused`: the second join consumes both outputs of the
/// first. Returns the unevaluated result.
fn two_joins(arrays: &CaseArrays, eps: f32) -> UniquePtr<MlxArray> {
    let (normed, residual) = fused_lazy(&arrays.x, &arrays.residual, &arrays.weight, eps, 0.0);
    let delta = add(&normed, &arrays.x);
    let (normed2, residual2) = fused_lazy(&delta, &residual, &arrays.weight, eps, 1.0);
    add(&normed2, &residual2)
}

/// `matmul -> fused -> matmul -> fused`, joined at the end: each join sits
/// between two captured cuBLAS child graphs and feeds both of its outputs
/// onward. Returns the unevaluated result.
fn joins_between_matmuls(arrays: &CaseArrays, m: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    let h0 = matmul(&arrays.x, m);
    let (n1, r1) = fused_lazy(&h0, &arrays.residual, &arrays.weight, eps, 0.0);
    let h1 = matmul(&n1, m);
    let side = matmul(&r1, m);
    let (n2, r2) = fused_lazy(&h1, &r1, &arrays.weight, eps, 1.0);
    add(&add(&n2, &r2), &side)
}

/// The reference bytes of `build` on the existing fused CUDA kernel.
fn on_existing_kernel(build: impl FnOnce() -> UniquePtr<MlxArray>, lane: Lane) -> Vec<u32> {
    oxide::uninstall();
    let before = oxide::route_counters();
    let bits = bits_of(&build(), lane);
    assert_eq!(
        delta(before).launches,
        0,
        "the reference must not use cuda-oxide"
    );
    bits
}

fn install() {
    oxide::install_embedded().expect("the embedded artifact validates and targets this device");
}

#[test]
fn a_lazy_chain_launches_nothing_until_it_is_evaluated() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    for (act, w) in [(Lane::F32, Lane::F32), (Lane::Bf16, Lane::F32)] {
        let c = case(act, w, 5, 1024);
        let arrays = case_arrays(&c, &inputs(&c, 1));
        let expected = on_existing_kernel(|| two_joins(&arrays, c.eps), act);

        install();
        let before = oxide::route_counters();
        let out = two_joins(&arrays, c.eps);
        let built = delta(before);
        assert_eq!(
            (built.oxide_routes, built.launches, built.native_routes),
            (2, 0, 0),
            "building the graph must route both joins and launch neither"
        );
        let got = bits_of(&out, act);
        assert_eq!(delta(before).launches, 2, "evaluation launches both joins");
        assert_eq!(got, expected, "{}", c.name);
    }
}

#[test]
fn a_replayed_graph_gives_the_same_bytes_every_time() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    const REPLAYS: usize = 60;
    let c = case(Lane::F16, Lane::F16, 5, 1024);
    // Three inputs of one shape: the graph key repeats, so MLX reuses the
    // instantiated graph and only rebinds the buffers.
    let sets: Vec<CaseArrays> = (0..3)
        .map(|i| case_arrays(&c, &inputs(&c, 10 + i)))
        .collect();
    let expected: Vec<Vec<u32>> = sets
        .iter()
        .map(|arrays| on_existing_kernel(|| two_joins(arrays, c.eps), c.act))
        .collect();

    install();
    let before = oxide::route_counters();
    for replay in 0..REPLAYS {
        let which = replay % sets.len();
        let got = bits_of(&two_joins(&sets[which], c.eps), c.act);
        assert_eq!(got, expected[which], "replay {replay}");
    }
    let ran = delta(before);
    assert_eq!(
        (ran.oxide_routes, ran.launches, ran.native_routes),
        (2 * REPLAYS as u64, 2 * REPLAYS as u64, 0)
    );
}

#[test]
fn joins_order_correctly_between_captured_child_graphs() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    const ROUNDS: usize = 12;
    let c = case(Lane::F32, Lane::F32, 7, 1024);
    let m = mixing_matrix(c.dim, 3);
    let sets: Vec<CaseArrays> = (0..2)
        .map(|i| case_arrays(&c, &inputs(&c, 20 + i)))
        .collect();
    let expected: Vec<Vec<u32>> = sets
        .iter()
        .map(|arrays| on_existing_kernel(|| joins_between_matmuls(arrays, &m, c.eps), c.act))
        .collect();

    install();
    let before = oxide::route_counters();
    for round in 0..ROUNDS {
        let which = round % sets.len();
        // One evaluation of the whole chain: the matmuls and the joins land in
        // the same encoder, so a missing dependency edge would let a join read
        // a matmul output before it is written, or the reverse.
        let got = bits_of(&joins_between_matmuls(&sets[which], &m, c.eps), c.act);
        assert_eq!(got, expected[which], "round {round}");
    }
    let ran = delta(before);
    assert_eq!((ran.launches, ran.native_routes), (2 * ROUNDS as u64, 0));
}

#[test]
fn outputs_survive_allocator_reuse() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    let cases = [
        case(Lane::F32, Lane::F32, 5, 1024),
        case(Lane::F16, Lane::F16, 33, 2048),
        case(Lane::Bf16, Lane::Bf16, 1, 128),
        case(Lane::F32, Lane::F32, 2, 4097),
        case(Lane::F16, Lane::F32, 5, 1024),
    ];
    let arrays: Vec<CaseArrays> = cases
        .iter()
        .enumerate()
        .map(|(i, c)| case_arrays(c, &inputs(c, 30 + i)))
        .collect();
    oxide::uninstall();
    let expected: Vec<OutBits> = cases
        .iter()
        .zip(&arrays)
        .map(|(c, a)| run_fused(c, a))
        .collect();

    install();
    let before = oxide::route_counters();
    // Held across every round below: its buffers must not be handed out again
    // while the handles are alive.
    let (kept_normed, kept_residual) = fused_lazy(
        &arrays[0].x,
        &arrays[0].residual,
        &arrays[0].weight,
        cases[0].eps,
        0.0,
    );
    eval(&kept_normed);

    const ROUNDS: usize = 40;
    for round in 0..ROUNDS {
        let which = round % cases.len();
        // Outputs are dropped at the end of each round and the shapes rotate,
        // so the pool hands the same buffers to outputs of other sizes.
        let got = run_fused(&cases[which], &arrays[which]);
        assert_eq!(got.normed, expected[which].normed, "round {round}");
        assert_eq!(
            got.new_residual, expected[which].new_residual,
            "round {round}"
        );
        if round % 9 == 8 {
            crate::clear_memory_cache();
        }
    }
    assert_eq!(bits_of(&kept_normed, cases[0].act), expected[0].normed);
    assert_eq!(
        bits_of(&kept_residual, cases[0].act),
        expected[0].new_residual
    );
    assert_eq!(delta(before).launches, ROUNDS as u64 + 1);
}

#[test]
fn inputs_and_sibling_outputs_dropped_by_the_caller_stay_alive() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    let c = case(Lane::Bf16, Lane::Bf16, 16, 8192);
    let inp: Inputs = inputs(&c, 40);
    oxide::uninstall();
    let expected = run_fused(&c, &case_arrays(&c, &inp));

    install();
    let before = oxide::route_counters();
    // The caller keeps one output only. The inputs and the other output are
    // referenced by the lazy node alone when it is evaluated.
    let new_residual = {
        let arrays = case_arrays(&c, &inp);
        let (normed, new_residual) =
            fused_lazy(&arrays.x, &arrays.residual, &arrays.weight, c.eps, 0.0);
        drop(normed);
        new_residual
    };
    assert_eq!(bits_of(&new_residual, c.act), expected.new_residual);

    // The same through `async_eval`: the handles go away while the launch may
    // still be in flight.
    let normed = {
        let arrays = case_arrays(&c, &inp);
        let (normed, new_residual) =
            fused_lazy(&arrays.x, &arrays.residual, &arrays.weight, c.eps, 0.0);
        crate::async_eval(&normed);
        drop(new_residual);
        normed
    };
    crate::clear_memory_cache();
    assert_eq!(bits_of(&normed, c.act), expected.normed);
    assert_eq!(delta(before).launches, 2);
}

#[test]
fn non_contiguous_inputs_are_copied_like_the_existing_port_copies_them() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    let c = case(Lane::F16, Lane::F32, 6, 1000);
    let inp = inputs(&c, 50);
    let (rows, dim) = (c.rows as i32, c.dim as i32);
    oxide::uninstall();
    let expected = run_fused(&c, &case_arrays(&c, &inp));

    // x arrives as the transpose of a [dim, rows] array (strided), residual as
    // a slice two rows into a taller array (an offset view), and weight as a
    // slice of a longer vector.
    let mut x_t = vec![0f32; c.elements()];
    for r in 0..c.rows {
        for d in 0..c.dim {
            x_t[d * c.rows + r] = inp.x[r * c.dim + d];
        }
    }
    let mut tall = vec![0f32; 2 * c.dim];
    tall.extend_from_slice(&inp.residual);
    let mut long = vec![9f32; 3];
    long.extend_from_slice(&inp.weight);

    let act = c.act.mlx_dtype();
    let x = crate::transpose(&astype(&from_slice_f32(&x_t, &[dim, rows]), act));
    let residual = crate::slice(
        &astype(&from_slice_f32(&tall, &[rows + 2, dim]), act),
        &[2, 0],
        &[rows + 2, dim],
    );
    let weight = crate::slice(&from_slice_f32(&long, &[dim + 3]), &[3], &[dim + 3]);
    for a in [&x, &residual, &weight] {
        eval(a);
    }

    install();
    let before = oxide::route_counters();
    let (normed, new_residual) = fused_lazy(&x, &residual, &weight, c.eps, c.weight_bias);
    assert_eq!(bits_of(&normed, c.act), expected.normed);
    assert_eq!(bits_of(&new_residual, c.act), expected.new_residual);
    assert_eq!(
        (delta(before).launches, delta(before).native_routes),
        (1, 0)
    );
}

#[test]
fn several_threads_can_drive_the_route_at_once() {
    let Some(_route) = RouteGuard::on_target_device() else {
        return;
    };
    const ITERATIONS: usize = 25;
    let cases = [
        case(Lane::F32, Lane::F32, 5, 1024),
        case(Lane::F16, Lane::F16, 5, 4096),
        case(Lane::Bf16, Lane::F32, 3, 8191),
        case(Lane::F16, Lane::F32, 33, 2048),
    ];
    oxide::uninstall();
    let expected: Vec<OutBits> = cases
        .iter()
        .enumerate()
        .map(|(i, c)| run_fused(c, &case_arrays(c, &inputs(c, 60 + i))))
        .collect();

    install();
    let before = oxide::route_counters();
    // Each thread builds and evaluates its own arrays, as a request worker
    // does.
    std::thread::scope(|scope| {
        for (i, (c, want)) in cases.iter().zip(&expected).enumerate() {
            scope.spawn(move || {
                let arrays = case_arrays(c, &inputs(c, 60 + i));
                for iteration in 0..ITERATIONS {
                    let got = run_fused(c, &arrays);
                    assert_eq!(got.normed, want.normed, "{} #{iteration}", c.name);
                    assert_eq!(
                        got.new_residual, want.new_residual,
                        "{} #{iteration}",
                        c.name
                    );
                }
            });
        }
    });
    let ran = delta(before);
    let total = (cases.len() * ITERATIONS) as u64;
    assert_eq!(
        (ran.oxide_routes, ran.launches, ran.native_routes),
        (total, total, 0)
    );
}
