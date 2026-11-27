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

//! Op-level comparison of the three implementations of the pre-norm residual
//! join through MLX (issue #2834): the graph path (`add` + `fast_rms_norm`),
//! the existing fused CUDA kernel, and the cuda-oxide node behind the same
//! launcher (#2833). It loads no model.
//!
//! ```text
//! MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121 \
//!     cargo build --release -p mlxcel-core --features cuda --example oxide_fused_norm_bench
//! gpu-lock run --tag issue-2834-op -- target/release/examples/oxide_fused_norm_bench \
//!     --mode iso --shapes model --rounds 5 --csv op.csv
//! ```
//!
//! Four modes, because they answer different questions:
//!
//! - `iso`: build, `eval`, `synchronize`, once per iteration, each iteration
//!   timed. This is also the process the profiler is attached to with
//!   `MLX_USE_CUDA_GRAPHS=0` to read kernel durations.
//! - `pipe`: build and `async_eval` every iteration, one `synchronize` at the
//!   end, which is what a decode loop does (docs/benchmarks.md, "An op-level
//!   number is not a decode number").
//! - `build`: build the lazy node and drop it, never evaluating: the host cost
//!   of the launcher alone, with no launch and no kernel.
//! - `cold`: one arm per fresh process. Times the first use of the op after the
//!   device context exists, then the second, then the steady state, so module
//!   load and kernel compilation are separated from kernel latency.
//!
//! Arms are interleaved by round and the order rotates. Every block records
//! the adapter's route counters over exactly its own calls and is marked
//! invalid when they do not match the arm: a silent fallback is not a data
//! point. The cuda-oxide arm exists only in a build made with
//! `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1`; elsewhere it is refused.

#[path = "oxide_fused_norm_bench/args.rs"]
mod args;
#[path = "oxide_fused_norm_bench/record.rs"]
mod record;

use std::fmt::Write as _;
use std::process::ExitCode;
use std::time::Instant;

use mlxcel_core::layers::{RMSNorm, graph_add_rms_norm};
use mlxcel_core::{
    MlxArray, UniquePtr, add, astype, async_eval_pair, clear_memory_cache, copy, dtype, eval,
    from_slice_f32, fused_add_rms_norm, fused_add_rms_norm_available, random_normal, random_seed,
    synchronize_default,
};

use args::{Args, Arm, Mode, Shape, dtype_name, parse_args};
use record::{CSV_HEADER, Recorder, Stats, append, stats};

const EPS: f32 = 1e-5;

/// Route counters of the adapter, `(oxide_routes, native_routes, launches)`;
/// zeros in a build without it.
fn counters() -> (u64, u64, u64) {
    mlxcel_core::fused_norm_route::fused_norm_route_snapshot()
        .map_or((0, 0, 0), |s| (s.oxide_routes, s.native_routes, s.launches))
}

/// Put the implementation `arm` needs behind the fused launcher.
fn select(arm: Arm) -> Result<(), String> {
    #[cfg(mlxcel_cuda_oxide_fused_norm)]
    {
        use mlxcel_core::cuda_oxide_fused_norm as oxide;
        match arm {
            Arm::Oxide => {
                if !oxide::is_installed() {
                    oxide::install_embedded().map_err(|e| format!("cuda-oxide install: {e}"))?;
                }
            }
            Arm::Graph | Arm::Fused => oxide::uninstall(),
        }
        Ok(())
    }
    #[cfg(not(mlxcel_cuda_oxide_fused_norm))]
    {
        if arm == Arm::Oxide {
            return Err(
                "the oxide arm needs a build made with MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Whether the counter deltas over `calls` calls are the ones `arm` implies.
fn route_valid(arm: Arm, calls: u64, delta: (u64, u64, u64)) -> bool {
    let has_adapter = mlxcel_core::fused_norm_route::fused_norm_route_snapshot().is_some();
    match arm {
        Arm::Graph => delta == (0, 0, 0),
        // Without the adapter nothing counts the existing kernel's calls.
        Arm::Fused => {
            delta
                == if has_adapter {
                    (0, calls, 0)
                } else {
                    (0, 0, 0)
                }
        }
        Arm::Oxide => delta == (calls, 0, calls),
    }
}

struct Inputs {
    delta: UniquePtr<MlxArray>,
    residual: UniquePtr<MlxArray>,
    weight: UniquePtr<MlxArray>,
    norm: RMSNorm,
}

fn randn(shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    // SAFETY: the null pointer is the documented "no explicit key" sentinel
    // for `random_normal`; the returned array owns its data.
    let raw = unsafe { random_normal(shape, dtype::FLOAT32, std::ptr::null()) };
    let out = astype(&raw, dt);
    eval(&out);
    out
}

fn inputs(shape: Shape) -> Inputs {
    // The models pass `[batch, tokens, hidden]`; rows are batch * tokens.
    let dims = [1, shape.rows, shape.dim];
    let delta = randn(&dims, shape.act);
    let residual = randn(&dims, shape.act);
    let values: Vec<f32> = (0..shape.dim)
        .map(|i| 1.0 + 0.01 * ((i % 7) as f32))
        .collect();
    let weight = astype(&from_slice_f32(&values, &[shape.dim]), shape.weight);
    eval(&weight);
    let norm = RMSNorm::new(copy(&weight), EPS);
    eval(&norm.weight);
    Inputs {
        delta,
        residual,
        weight,
        norm,
    }
}

/// Build one residual join with `arm`'s implementation, lazily.
fn build(arm: Arm, inp: &Inputs) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    match arm {
        Arm::Graph => graph_add_rms_norm(&inp.norm, &inp.delta, &inp.residual),
        Arm::Fused | Arm::Oxide => {
            let mut normed = UniquePtr::null();
            let mut new_residual = UniquePtr::null();
            fused_add_rms_norm(
                &inp.delta,
                &inp.residual,
                &inp.weight,
                EPS,
                0.0,
                &mut normed,
                &mut new_residual,
            )
            .expect("main checked fused_add_rms_norm_available(), so the launcher must not refuse");
            (normed, new_residual)
        }
    }
}

/// Build, evaluate both outputs, and wait for the device.
fn run_once(arm: Arm, inp: &Inputs) {
    let (normed, new_residual) = build(arm, inp);
    async_eval_pair(&normed, &new_residual);
    synchronize_default();
}

fn delta_since(before: (u64, u64, u64)) -> (u64, u64, u64) {
    let now = counters();
    (now.0 - before.0, now.1 - before.1, now.2 - before.2)
}

fn run_iso(args: &Args, rec: &mut Recorder) -> Result<(), String> {
    for &shape in &args.shapes {
        let inp = inputs(shape);
        for round in 0..args.rounds {
            for slot in 0..args.arms.len() {
                let arm = args.arms[(slot + round) % args.arms.len()];
                select(arm)?;
                for _ in 0..args.warmup {
                    run_once(arm, &inp);
                }
                let before = counters();
                let mut samples = Vec::with_capacity(args.iters);
                for _ in 0..args.iters {
                    let start = Instant::now();
                    run_once(arm, &inp);
                    samples.push(start.elapsed().as_secs_f64() * 1e6);
                }
                let delta = delta_since(before);
                let valid = route_valid(arm, args.iters as u64, delta);
                rec.push(
                    args,
                    "iso",
                    arm,
                    shape,
                    round,
                    args.iters,
                    &stats(&mut samples),
                    delta,
                    valid,
                );
            }
        }
        drop(inp);
        clear_memory_cache();
    }
    Ok(())
}

fn run_pipe(args: &Args, rec: &mut Recorder) -> Result<(), String> {
    for &shape in &args.shapes {
        let inp = inputs(shape);
        for round in 0..args.rounds {
            for slot in 0..args.arms.len() {
                let arm = args.arms[(slot + round) % args.arms.len()];
                select(arm)?;
                for _ in 0..args.warmup {
                    run_once(arm, &inp);
                }
                let before = counters();
                synchronize_default();
                let start = Instant::now();
                // Keep the previous step's outputs alive while the next one is
                // enqueued, as a decode loop holds the step it has not read.
                let mut previous = None;
                for _ in 0..args.iters {
                    let (normed, new_residual) = build(arm, &inp);
                    async_eval_pair(&normed, &new_residual);
                    previous = Some((normed, new_residual));
                }
                synchronize_default();
                let per_call = start.elapsed().as_secs_f64() * 1e6 / args.iters as f64;
                drop(previous);
                let delta = delta_since(before);
                let valid = route_valid(arm, args.iters as u64, delta);
                let s = Stats {
                    mean_us: per_call,
                    median_us: per_call,
                    p90_us: per_call,
                    min_us: per_call,
                };
                rec.push(
                    args, "pipe", arm, shape, round, args.iters, &s, delta, valid,
                );
            }
        }
        drop(inp);
        clear_memory_cache();
    }
    Ok(())
}

/// Host cost of building the lazy node alone: nothing is evaluated, so no
/// kernel is launched and no CUDA API is called. This is what separates the
/// launcher's own work (argument arrays, template arguments, kernel lookup)
/// from the launch and the kernel.
fn run_build(args: &Args, rec: &mut Recorder) -> Result<(), String> {
    for &shape in &args.shapes {
        let inp = inputs(shape);
        for round in 0..args.rounds {
            for slot in 0..args.arms.len() {
                let arm = args.arms[(slot + round) % args.arms.len()];
                select(arm)?;
                for _ in 0..args.warmup {
                    run_once(arm, &inp);
                }
                let before = counters();
                let start = Instant::now();
                for _ in 0..args.iters {
                    drop(build(arm, &inp));
                }
                let per_call = start.elapsed().as_secs_f64() * 1e6 / args.iters as f64;
                let delta = delta_since(before);
                // Routed at construction, never launched.
                let calls = args.iters as u64;
                let valid = match arm {
                    Arm::Oxide => delta == (calls, 0, 0),
                    _ => route_valid(arm, calls, delta),
                };
                let s = Stats {
                    mean_us: per_call,
                    median_us: per_call,
                    p90_us: per_call,
                    min_us: per_call,
                };
                rec.push(
                    args, "build", arm, shape, round, args.iters, &s, delta, valid,
                );
            }
        }
        drop(inp);
        clear_memory_cache();
    }
    Ok(())
}

/// One arm, one fresh process: device context, then the first use of the op,
/// then the second, then the steady state.
fn run_cold(args: &Args, rec: &mut Recorder) -> Result<(), String> {
    let arm = args.arms[0];
    let start = Instant::now();
    let ones = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let warm = add(&ones, &ones);
    eval(&warm);
    synchronize_default();
    let context_ms = start.elapsed().as_secs_f64() * 1e3;

    let mut csv_extra = String::new();
    for (index, &shape) in args.shapes.iter().enumerate() {
        let inp = inputs(shape);
        synchronize_default();
        let before = counters();
        let start = Instant::now();
        // The first fused join installs the cuda-oxide artifact in production
        // (`ensure_from_env`), so validation and installation belong to the
        // first use of that arm.
        select(arm)?;
        let select_us = start.elapsed().as_secs_f64() * 1e6;
        run_once(arm, &inp);
        let first_us = start.elapsed().as_secs_f64() * 1e6;
        let start = Instant::now();
        run_once(arm, &inp);
        let second_us = start.elapsed().as_secs_f64() * 1e6;
        let mut samples = Vec::with_capacity(args.iters);
        for _ in 0..args.iters {
            let start = Instant::now();
            run_once(arm, &inp);
            samples.push(start.elapsed().as_secs_f64() * 1e6);
        }
        let calls = args.iters as u64 + 2;
        let delta = delta_since(before);
        let valid = route_valid(arm, calls, delta);
        let s = stats(&mut samples);
        rec.push(
            args,
            "cold-steady",
            arm,
            shape,
            index,
            args.iters,
            &s,
            delta,
            valid,
        );
        let _ = writeln!(
            csv_extra,
            "COLD,{},{},{},{},{},{},{index},{context_ms:.3},{select_us:.3},{first_us:.3},{second_us:.3},{:.3},{valid}",
            args.label,
            arm.name(),
            dtype_name(shape.act),
            dtype_name(shape.weight),
            shape.rows,
            shape.dim,
            s.median_us,
        );
        println!(
            "cold  {:<5} {:>5} x {:<5} shape #{index}: context {context_ms:.1} ms, select {select_us:.1} us, first use {first_us:.1} us, second {second_us:.1} us, steady median {:.1} us",
            arm.name(),
            shape.rows,
            shape.dim,
            s.median_us,
        );
    }
    println!(
        "\nCOLD CSV\ntag,label,arm,act,weight,rows,dim,shape_index,context_ms,select_us,first_use_us,second_use_us,steady_median_us,route_valid"
    );
    print!("{csv_extra}");
    if let Some(path) = &args.csv {
        let cold_path = format!("{path}.cold.csv");
        let header = "tag,label,arm,act,weight,rows,dim,shape_index,context_ms,select_us,first_use_us,second_use_us,steady_median_us,route_valid\n";
        append(&cold_path, header, &csv_extra)?;
    }
    Ok(())
}

fn run(args: &Args) -> Result<usize, String> {
    if !fused_add_rms_norm_available() {
        return Err("fused_add_rms_norm is unavailable on this build or device".to_string());
    }
    random_seed(2834);
    println!(
        "oxide_fused_norm_bench: adapter {}, MLX_USE_CUDA_GRAPHS={}",
        if mlxcel_core::fused_norm_route::fused_norm_route_snapshot().is_some() {
            "built"
        } else {
            "absent"
        },
        std::env::var("MLX_USE_CUDA_GRAPHS").unwrap_or_else(|_| "unset".to_string()),
    );
    let mut rec = Recorder {
        csv: String::new(),
        invalid: 0,
    };
    match args.mode {
        Mode::Build => run_build(args, &mut rec)?,
        Mode::Iso => run_iso(args, &mut rec)?,
        Mode::Pipe => run_pipe(args, &mut rec)?,
        Mode::Cold => run_cold(args, &mut rec)?,
    }
    if let Some(path) = &args.csv {
        append(path, &format!("{CSV_HEADER}\n"), &rec.csv)?;
    }
    Ok(rec.invalid)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("oxide_fused_norm_bench: {message}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(0) => ExitCode::SUCCESS,
        Ok(invalid) => {
            eprintln!("oxide_fused_norm_bench: {invalid} block(s) ran on the wrong route");
            ExitCode::FAILURE
        }
        Err(message) => {
            eprintln!("oxide_fused_norm_bench: {message}");
            ExitCode::FAILURE
        }
    }
}
