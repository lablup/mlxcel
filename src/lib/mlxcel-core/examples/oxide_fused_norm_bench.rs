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
//! Three modes, because they answer different questions:
//!
//! - `iso`: build, `eval`, `synchronize`, once per iteration, each iteration
//!   timed. This is also the process the profiler is attached to with
//!   `MLX_USE_CUDA_GRAPHS=0` to read kernel durations.
//! - `pipe`: build and `async_eval` every iteration, one `synchronize` at the
//!   end, which is what a decode loop does (docs/benchmarks.md, "An op-level
//!   number is not a decode number").
//! - `cold`: one arm per fresh process. Times the first use of the op after the
//!   device context exists, then the second, then the steady state, so module
//!   load and kernel compilation are separated from kernel latency.
//!
//! Arms are interleaved by round and the order rotates. Every block records
//! the adapter's route counters over exactly its own calls and is marked
//! invalid when they do not match the arm: a silent fallback is not a data
//! point. The cuda-oxide arm exists only in a build made with
//! `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1`; elsewhere it is refused.

use std::fmt::Write as _;
use std::process::ExitCode;
use std::time::Instant;

use mlxcel_core::layers::{RMSNorm, graph_add_rms_norm};
use mlxcel_core::{
    MlxArray, UniquePtr, add, astype, async_eval_pair, clear_memory_cache, copy, dtype, eval,
    from_slice_f32, fused_add_rms_norm, fused_add_rms_norm_available, random_normal, random_seed,
    synchronize_default,
};

const EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Graph,
    Fused,
    Oxide,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Graph => "graph",
            Arm::Fused => "fused",
            Arm::Oxide => "oxide",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "graph" => Some(Arm::Graph),
            "fused" => Some(Arm::Fused),
            "oxide" => Some(Arm::Oxide),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape {
    act: i32,
    weight: i32,
    rows: i32,
    dim: i32,
}

fn dtype_name(dt: i32) -> &'static str {
    match dt {
        dtype::FLOAT32 => "f32",
        dtype::FLOAT16 => "f16",
        dtype::BFLOAT16 => "bf16",
        _ => "other",
    }
}

fn parse_dtype(s: &str) -> Option<i32> {
    match s {
        "f32" => Some(dtype::FLOAT32),
        "f16" => Some(dtype::FLOAT16),
        "bf16" => Some(dtype::BFLOAT16),
        _ => None,
    }
}

/// The residual joins of the #2834 checkpoint (hidden size 2048, bf16
/// activations and weight): one decode step at B=1 and B=4, and the prefill of
/// a 128-token and a 2048-token prompt at B=1 plus the 128-token one at B=4.
fn model_shapes() -> Vec<Shape> {
    [1, 4, 128, 512, 2048]
        .into_iter()
        .map(|rows| Shape {
            act: dtype::BFLOAT16,
            weight: dtype::BFLOAT16,
            rows,
            dim: 2048,
        })
        .collect()
}

/// The shape list of the #2832 standalone benchmark
/// (`spike/cuda-oxide-fused-norm/results/bench-sm_121.json`).
fn bench2832_shapes() -> Vec<Shape> {
    let mut shapes = Vec::new();
    for dt in [dtype::FLOAT32, dtype::BFLOAT16, dtype::FLOAT16] {
        for (rows, dim) in [
            (1, 1024),
            (1, 4096),
            (1, 8192),
            (5, 4096),
            (33, 2048),
            (64, 8192),
        ] {
            shapes.push(Shape {
                act: dt,
                weight: dt,
                rows,
                dim,
            });
        }
    }
    shapes
}

fn parse_shapes(spec: &str) -> Result<Vec<Shape>, String> {
    let mut shapes = Vec::new();
    for item in spec.split(',').filter(|s| !s.is_empty()) {
        match item {
            "model" => shapes.extend(model_shapes()),
            "bench2832" => shapes.extend(bench2832_shapes()),
            custom => {
                // act:weight:ROWSxDIM
                let parts: Vec<&str> = custom.split(':').collect();
                let bad = || format!("shape {custom:?}: expected act:weight:ROWSxDIM");
                if parts.len() != 3 {
                    return Err(bad());
                }
                let (rows, dim) = parts[2].split_once('x').ok_or_else(bad)?;
                shapes.push(Shape {
                    act: parse_dtype(parts[0]).ok_or_else(bad)?,
                    weight: parse_dtype(parts[1]).ok_or_else(bad)?,
                    rows: rows.parse().map_err(|_| bad())?,
                    dim: dim.parse().map_err(|_| bad())?,
                });
            }
        }
    }
    if shapes.is_empty() {
        return Err("no shapes selected".to_string());
    }
    Ok(shapes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Iso,
    Pipe,
    Cold,
}

struct Args {
    mode: Mode,
    arms: Vec<Arm>,
    shapes: Vec<Shape>,
    iters: usize,
    warmup: usize,
    rounds: usize,
    csv: Option<String>,
    label: String,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        mode: Mode::Iso,
        arms: vec![Arm::Graph, Arm::Fused, Arm::Oxide],
        shapes: model_shapes(),
        iters: 200,
        warmup: 20,
        rounds: 5,
        csv: None,
        label: String::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--mode" => {
                args.mode = match value()?.as_str() {
                    "iso" => Mode::Iso,
                    "pipe" => Mode::Pipe,
                    "cold" => Mode::Cold,
                    other => return Err(format!("unknown mode {other:?}")),
                }
            }
            "--arms" => {
                args.arms = value()?
                    .split(',')
                    .map(|s| Arm::parse(s).ok_or_else(|| format!("unknown arm {s:?}")))
                    .collect::<Result<_, _>>()?;
            }
            "--shapes" => args.shapes = parse_shapes(&value()?)?,
            "--iters" => args.iters = value()?.parse().map_err(|e| format!("--iters: {e}"))?,
            "--warmup" => args.warmup = value()?.parse().map_err(|e| format!("--warmup: {e}"))?,
            "--rounds" => args.rounds = value()?.parse().map_err(|e| format!("--rounds: {e}"))?,
            "--csv" => args.csv = Some(value()?),
            "--label" => args.label = value()?,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if args.arms.is_empty() || args.iters == 0 || args.rounds == 0 {
        return Err("--arms, --iters and --rounds must be non-empty".to_string());
    }
    if args.mode == Mode::Cold && args.arms.len() != 1 {
        return Err("--mode cold measures one arm per process; pass a single --arms".to_string());
    }
    Ok(args)
}

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

struct Stats {
    mean_us: f64,
    median_us: f64,
    p90_us: f64,
    min_us: f64,
}

fn stats(samples_us: &mut [f64]) -> Stats {
    samples_us.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    let n = samples_us.len();
    let median_us = if n.is_multiple_of(2) {
        (samples_us[n / 2 - 1] + samples_us[n / 2]) / 2.0
    } else {
        samples_us[n / 2]
    };
    Stats {
        mean_us: samples_us.iter().sum::<f64>() / n as f64,
        median_us,
        p90_us: samples_us[((n as f64 * 0.9).ceil() as usize).clamp(1, n) - 1],
        min_us: samples_us[0],
    }
}

const CSV_HEADER: &str = "label,mode,arm,act,weight,rows,dim,round,iters,warmup,mean_us,median_us,\
p90_us,min_us,oxide_routes,native_routes,launches,route_valid";

struct Recorder {
    csv: String,
    invalid: usize,
}

impl Recorder {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        args: &Args,
        mode: &str,
        arm: Arm,
        shape: Shape,
        round: usize,
        iters: usize,
        s: &Stats,
        delta: (u64, u64, u64),
        valid: bool,
    ) {
        if !valid {
            self.invalid += 1;
        }
        let _ = writeln!(
            self.csv,
            "{},{mode},{},{},{},{},{},{round},{iters},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{valid}",
            args.label,
            arm.name(),
            dtype_name(shape.act),
            dtype_name(shape.weight),
            shape.rows,
            shape.dim,
            args.warmup,
            s.mean_us,
            s.median_us,
            s.p90_us,
            s.min_us,
            delta.0,
            delta.1,
            delta.2,
        );
        println!(
            "{mode:<5} {:<5} {:>4}/{:<4} {:>5} x {:<5} round {round}  mean {:>9.3} us  median {:>9.3}  p90 {:>9.3}  min {:>9.3}  routes {delta:?}{}",
            arm.name(),
            dtype_name(shape.act),
            dtype_name(shape.weight),
            shape.rows,
            shape.dim,
            s.mean_us,
            s.median_us,
            s.p90_us,
            s.min_us,
            if valid { "" } else { "  INVALID ROUTE" },
        );
    }
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

fn append(path: &str, header: &str, body: &str) -> Result<(), String> {
    use std::io::Write as _;
    let exists = std::path::Path::new(path).exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{path}: {e}"))?;
    if !exists {
        file.write_all(header.as_bytes())
            .map_err(|e| format!("{path}: {e}"))?;
    }
    file.write_all(body.as_bytes())
        .map_err(|e| format!("{path}: {e}"))
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
