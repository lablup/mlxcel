// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Filtered-sampling microbenchmark for the dual-pivot rejection kernel
//! (issue #901). The #900 harness with the filter matrix the issue specifies.
//!
//! Measures the sampling step *alone*, with no model attached: a synthetic
//! `[batch, vocab]` logits tensor is sampled repeatedly and the two arms are
//! compared at every point of the vocab x batch x filter-config matrix.
//!
//! - **baseline** is `fused_sample_categorical`, the stock chain: temperature
//!   scale, `argpartition` top-k, `argsort` + `cumsum` top-p, compiled min-p,
//!   `random::categorical`.
//! - **rejection** is the kernel: one softmax and a shrinking probability
//!   interval resolved by two pivots per vocabulary sweep. The isolated mode
//!   calls `fused_sample_rejection`, the forced entry point, so the kernel is
//!   measured on every cell whether or not production routes it; that entry
//!   point also reads the per-row converged flags back and runs the
//!   `argpartition` fallback, so it synchronizes. The pipelined mode calls
//!   `fused_sample`, the production entry point, which never evaluates
//!   anything.
//!
//! ## Two timing modes, and why one of them is not optional
//!
//! **`iso`** is the classic op-level measurement: construct the graph, `eval`,
//! `synchronize_default()`, so the GPU work is retired inside the timed region.
//! The reported number is the median over `--iters` repetitions after `--warmup`
//! discarded ones.
//!
//! **`pipe`** reproduces what a decode loop actually does. `generate.rs` and the
//! batch scheduler are software pipelines: each iteration builds the next
//! forward and the next sample, `async_eval`s them, and only then reads the
//! PREVIOUS step's token. So each step is timed with a synthetic forward
//! enqueued ahead of the sampler and the previous step's tokens read back after,
//! with one synchronize at the end of the run rather than one per iteration.
//!
//! `iso` alone is not enough, and this harness learned that the hard way. A
//! sampler that evaluates anything internally drains the queue inside the
//! caller's build phase and collapses the pipeline. `iso` cannot see that,
//! because it synchronizes around every iteration anyway, so the sync the
//! sampler forces is already paid. The first cut of #901 read the rejection
//! kernel's converged flags back to host inside `fused_sample`; `iso` scored it
//! 1.14x to 1.17x FASTER at vocab 152064 while end-to-end decode on Qwen3-0.6B
//! measured **1.7x slower**. Read `pipe_x`, and treat a large `iso_x` with a
//! poor `pipe_x` as a sampler that is synchronizing.
//!
//! The synthetic forward is a chain of 1024x1024 matmuls, calibrated at startup
//! to roughly `PIPELINE_FORWARD_TARGET_US`. It is a stand-in for the model, not
//! a model: it exists so the pipeline has real GPU work to overlap with, which
//! is the condition under which a forced sync becomes expensive.
//!
//! Each row reports two things besides the timings. `rounds` is the worst
//! per-row rejection round count the kernel consumed for that shape, which is
//! the number that explains the timings: top-p accepts in one or two rounds,
//! top-k needs more and the count grows with the vocabulary, and every round is
//! another full-row sweep on a single threadgroup. `routed` is whether
//! `fused_sample` would actually send that configuration to the kernel in
//! production.
//!
//! The rejection arm is measured for EVERY cell, routed or not, because it goes
//! through the forced entry point. A `routed=no` row is therefore a real
//! measurement of a path production does not take, kept so the routing policy
//! can be re-derived from the table rather than trusted. Read the `routed=yes`
//! rows for what shipping costs, and the whole table for whether the policy is
//! still right.
//!
//! The header prints the dispatch outcome each arm recorded, read from the same
//! record `mlxcel-server` announces at INFO. Those two lines are the proof that
//! the arms ran on different paths: issue #899 shipped a benchmark that compared
//! the fallback against itself for a full sweep, and this harness is built so
//! that cannot happen silently. If both lines name the same path, the table
//! below them is measuring nothing.
//!
//! **Memory mode: warm** (see `docs/benchmarks.md`). The logits tensor is
//! allocated once and reused, which is the representative mode here: in
//! production the LM head writes the logits row immediately before the sampler
//! reads it. The largest point (batch 8, vocab 152064) reads under 5 MiB.
//!
//! Run (Apple):
//!   cargo run --release --features metal,accelerate \
//!     --example rejection_sampling_microbench
//! Run (CUDA):
//!   cargo run --release --features cuda --example rejection_sampling_microbench
//!
//! Options:
//!   --iters N     timed repetitions per point (default 200)
//!   --warmup N    discarded repetitions per point (default 30)
//!   --csv PATH    also write the table as CSV
//!
//! On Apple Silicon run under `caffeinate -i` and let the machine cool between
//! sweeps; it down-clocks under sustained load.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use mlxcel_core::{
    MlxArray, UniquePtr, array_to_raw_bytes, async_eval_pair, eval, from_slice_f32, fused_sample,
    fused_sample_categorical, fused_sample_rejection, is_gpu_available, matmul, random_seed,
    rejection_cap_overflow_launches, rejection_cap_overflow_rows, reset_sampling_dispatch,
    sampling_dispatch_recorded_report, sampling_rejection_available, sampling_rejection_max_rounds,
    sampling_rejection_probe, sampling_rejection_routes, synchronize_default,
};

/// Target duration for the synthetic forward in the pipelined mode. Roughly a
/// small-model decode step, which is the regime where a sampler-forced sync
/// costs the most.
const PIPELINE_FORWARD_TARGET_US: f64 = 2000.0;

/// Side length of the synthetic forward's square matmul.
const FORWARD_DIM: i32 = 1024;

const VOCABS: [i32; 3] = [32_768, 65_536, 152_064];
const BATCHES: [i32; 3] = [1, 4, 8];
const TEMPERATURE: f32 = 1.0;

/// The filter matrix issue #901 specifies: `(label, top_k, top_p, min_p)`.
const CONFIGS: [(&str, i32, f32, f32); 4] = [
    ("top-k=40", 40, 1.0, 0.0),
    ("top-p=0.9", 0, 0.9, 0.0),
    ("min-p=0.05", 0, 1.0, 0.05),
    ("top-k+top-p", 40, 0.9, 0.0),
];

struct Options {
    iters: usize,
    warmup: usize,
    csv: Option<String>,
}

fn parse_options() -> Options {
    let mut opts = Options {
        iters: 200,
        warmup: 30,
        csv: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iters" => {
                opts.iters = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--iters needs a positive integer");
            }
            "--warmup" => {
                opts.warmup = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--warmup needs a non-negative integer");
            }
            "--csv" => {
                opts.csv = Some(args.next().expect("--csv needs a path"));
            }
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(opts.iters > 0, "--iters must be positive");
    opts
}

/// Deterministic pseudo-logits with a realistic decode-step shape: a broad
/// low-probability floor with a handful of dominant tokens. Same generator as
/// the #900 harness, so the two benchmarks describe the same input.
fn synthetic_logits(batch: i32, vocab: i32) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity((batch as usize) * (vocab as usize));
    for row in 0..batch {
        let mut state = 0x2545_F491_4F6C_DD1Du64 ^ (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for _ in 0..vocab {
            // xorshift64* for a reproducible spread without pulling in a crate.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 16_777_216.0;
            data.push(-6.0 + 14.0 * u * u * u);
        }
    }
    from_slice_f32(&data, &[batch, vocab])
}

/// A chain of `links` square matmuls, unevaluated. Stand-in for the model
/// forward that a real decode step enqueues before it samples.
fn forward_chain(lhs: &MlxArray, seed: &MlxArray, links: usize) -> UniquePtr<MlxArray> {
    let mut acc = matmul(lhs, seed);
    for _ in 1..links.max(1) {
        acc = matmul(lhs, &acc);
    }
    acc
}

/// Operand pair plus the chain length that makes one `forward_chain` take about
/// [`PIPELINE_FORWARD_TARGET_US`].
struct SyntheticForward {
    lhs: UniquePtr<MlxArray>,
    seed: UniquePtr<MlxArray>,
    links: usize,
    measured_us: f64,
}

impl SyntheticForward {
    fn calibrate() -> Self {
        let side = (FORWARD_DIM * FORWARD_DIM) as usize;
        let data: Vec<f32> = (0..side).map(|i| ((i % 251) as f32) * 0.001).collect();
        let lhs = from_slice_f32(&data, &[FORWARD_DIM, FORWARD_DIM]);
        let seed = from_slice_f32(&data, &[FORWARD_DIM, FORWARD_DIM]);

        // Warm the kernel, then time a fixed probe chain and scale from it.
        const PROBE_LINKS: usize = 8;
        for _ in 0..3 {
            let out = forward_chain(&lhs, &seed, PROBE_LINKS);
            eval(&out);
        }
        synchronize_default();
        let start = Instant::now();
        let out = forward_chain(&lhs, &seed, PROBE_LINKS);
        eval(&out);
        synchronize_default();
        let probe_us = start.elapsed().as_secs_f64() * 1e6;

        let per_link = (probe_us / PROBE_LINKS as f64).max(1.0);
        let links = ((PIPELINE_FORWARD_TARGET_US / per_link).round() as usize).clamp(1, 512);

        let start = Instant::now();
        let out = forward_chain(&lhs, &seed, links);
        eval(&out);
        synchronize_default();
        let measured_us = start.elapsed().as_secs_f64() * 1e6;

        Self {
            lhs,
            seed,
            links,
            measured_us,
        }
    }

    fn enqueue(&self) -> UniquePtr<MlxArray> {
        forward_chain(&self.lhs, &self.seed, self.links)
    }
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn time_arm<F>(logits: &MlxArray, opts: &Options, mut sample: F) -> Duration
where
    F: FnMut(&MlxArray) -> UniquePtr<MlxArray>,
{
    for _ in 0..opts.warmup {
        let tokens = sample(logits);
        eval(&tokens);
    }
    synchronize_default();

    let mut samples = Vec::with_capacity(opts.iters);
    for _ in 0..opts.iters {
        let start = Instant::now();
        let tokens = sample(logits);
        eval(&tokens);
        synchronize_default();
        samples.push(start.elapsed());
    }
    median(samples)
}

/// Per-step time in the pipelined regime a decode loop actually runs.
///
/// Mirrors `generate.rs`: enqueue the forward and the sample for step n, hand
/// both to `async_eval`, then read step n-1's tokens. One synchronize at the
/// end, not one per iteration, so a sampler that evaluates internally shows up
/// as lost overlap instead of being hidden by the harness's own sync.
fn time_arm_pipelined<F>(
    logits: &MlxArray,
    opts: &Options,
    forward: &SyntheticForward,
    mut sample: F,
) -> Duration
where
    F: FnMut(&MlxArray) -> UniquePtr<MlxArray>,
{
    for _ in 0..opts.warmup {
        let fwd = forward.enqueue();
        let tokens = sample(logits);
        async_eval_pair(&fwd, &tokens);
        let _ = array_to_raw_bytes(&tokens);
    }
    synchronize_default();

    let start = Instant::now();
    let mut prev: Option<UniquePtr<MlxArray>> = None;
    for _ in 0..opts.iters {
        let fwd = forward.enqueue();
        let tokens = sample(logits);
        async_eval_pair(&fwd, &tokens);
        if let Some(previous) = prev.take() {
            let _ = array_to_raw_bytes(&previous);
        }
        prev = Some(tokens);
    }
    if let Some(previous) = prev {
        let _ = array_to_raw_bytes(&previous);
    }
    synchronize_default();
    start.elapsed() / opts.iters as u32
}

/// Worst per-row round count the kernel consumed for this shape.
fn worst_rounds(
    logits: &MlxArray,
    batch: i32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    cap: i32,
) -> u32 {
    let stacked = sampling_rejection_probe(logits, TEMPERATURE, top_k, top_p, min_p, cap);
    let flat: Vec<u32> = array_to_raw_bytes(&stacked)
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let rows = batch as usize;
    flat[2 * rows..3 * rows].iter().copied().max().unwrap_or(0)
}

/// Print the dispatch outcomes recorded since the last reset.
fn print_dispatch(prefix: &str) {
    let report = sampling_dispatch_recorded_report();
    for line in report.lines() {
        println!("{prefix}{line}");
    }
}

fn main() {
    let opts = parse_options();

    if !is_gpu_available() {
        eprintln!("No GPU backend available; the sampling kernel cannot run here.");
        return;
    }
    let cap = sampling_rejection_max_rounds();
    println!(
        "Dual-pivot rejection sampling microbenchmark (#901)  iters={} warmup={}  \
         routing_enabled={}  round_cap={cap}",
        opts.iters,
        opts.warmup,
        sampling_rejection_available()
    );

    // Prove the two arms take different paths before timing anything.
    reset_sampling_dispatch();
    let probe_logits = synthetic_logits(4, 32_768);
    let _ = fused_sample_categorical(&probe_logits, TEMPERATURE, 40, 0.9, 0.0);
    print_dispatch("  baseline arm  -> ");
    reset_sampling_dispatch();
    let _ = fused_sample_rejection(&probe_logits, TEMPERATURE, 40, 0.9, 0.0, cap);
    print_dispatch("  rejection arm (iso)  -> ");
    reset_sampling_dispatch();
    let _ = fused_sample(&probe_logits, TEMPERATURE, 0, 0.9, 0.0);
    print_dispatch("  rejection arm (pipe) -> ");
    println!();

    let forward = SyntheticForward::calibrate();
    println!(
        "pipelined mode: synthetic forward = {} chained {}x{} matmuls, {:.0}us measured",
        forward.links, FORWARD_DIM, FORWARD_DIM, forward.measured_us
    );
    println!();

    println!(
        "{:>12} {:>8} {:>6} {:>7} {:>7} {:>11} {:>11} {:>8} {:>12} {:>11} {:>8}",
        "config",
        "vocab",
        "batch",
        "rounds",
        "routed",
        "iso_base_us",
        "iso_rej_us",
        "iso_x",
        "pipe_base_us",
        "pipe_rej_us",
        "pipe_x"
    );

    let mut csv = String::from(
        "config,vocab,batch,worst_rounds,routed,iso_baseline_us,iso_rejection_us,iso_speedup,\
         pipe_baseline_us,pipe_rejection_us,pipe_speedup\n",
    );
    for (label, top_k, top_p, min_p) in CONFIGS {
        for vocab in VOCABS {
            for batch in BATCHES {
                let logits = synthetic_logits(batch, vocab);
                let rounds = worst_rounds(&logits, batch, top_k, top_p, min_p, cap);

                // Reseed before each arm so both consume the same RNG stream
                // position; the draw itself is part of what is being measured.
                // The pipelined arm samples through `fused_sample`, the
                // production entry point, because the whole point of that mode
                // is to measure what production pays INCLUDING any
                // synchronization the sampler forces on its caller. On a
                // `routed=no` row that means both pipelined arms are the same
                // code and `pipe_x` should read about 1.00x; that is not a
                // wasted cell, it is the check that the routing policy is
                // actually in force. The isolated arm keeps the forced entry
                // point so the kernel itself is measured on every cell,
                // routed or not.
                random_seed(0x0901_BEEF);
                let iso_base = time_arm(&logits, &opts, |x| {
                    fused_sample_categorical(x, TEMPERATURE, top_k, top_p, min_p)
                });
                random_seed(0x0901_BEEF);
                let iso_rej = time_arm(&logits, &opts, |x| {
                    fused_sample_rejection(x, TEMPERATURE, top_k, top_p, min_p, cap)
                });
                random_seed(0x0901_BEEF);
                let pipe_base = time_arm_pipelined(&logits, &opts, &forward, |x| {
                    fused_sample_categorical(x, TEMPERATURE, top_k, top_p, min_p)
                });
                random_seed(0x0901_BEEF);
                let pipe_rej = time_arm_pipelined(&logits, &opts, &forward, |x| {
                    fused_sample(x, TEMPERATURE, top_k, top_p, min_p)
                });

                let iso_base_us = iso_base.as_secs_f64() * 1e6;
                let iso_rej_us = iso_rej.as_secs_f64() * 1e6;
                let iso_x = iso_base_us / iso_rej_us;
                let pipe_base_us = pipe_base.as_secs_f64() * 1e6;
                let pipe_rej_us = pipe_rej.as_secs_f64() * 1e6;
                let pipe_x = pipe_base_us / pipe_rej_us;
                let routed = sampling_rejection_routes(vocab, top_k, top_p, min_p);
                let routed_text = if routed { "yes" } else { "no" };
                println!(
                    "{label:>12} {vocab:>8} {batch:>6} {rounds:>7} {routed_text:>7} \
                     {iso_base_us:>11.2} {iso_rej_us:>11.2} {iso_x:>7.2}x \
                     {pipe_base_us:>12.2} {pipe_rej_us:>11.2} {pipe_x:>7.2}x"
                );
                let _ = writeln!(
                    csv,
                    "{label},{vocab},{batch},{rounds},{routed_text},{iso_base_us:.3},\
                     {iso_rej_us:.3},{iso_x:.4},{pipe_base_us:.3},{pipe_rej_us:.3},{pipe_x:.4}"
                );
            }
        }
    }

    println!(
        "\ncap overflow: {} rows across {} launches",
        rejection_cap_overflow_rows(),
        rejection_cap_overflow_launches()
    );
    println!(
        "routing policy: the kernel replaces a sort, so it is routed only where the stock chain \
         sorts (top-p active), and top-k + top-p only at vocab <= 32768. A routed=no row is a \
         path production does not take."
    );
    println!(
        "read pipe_x, not iso_x: iso synchronizes around every iteration and is structurally \
         blind to a sync the sampler forces on its caller. A large iso_x with a poor pipe_x \
         means the sampler is synchronizing."
    );
    print_dispatch("dispatch: ");

    if let Some(path) = opts.csv {
        match std::fs::write(&path, csv) {
            Ok(()) => println!("\nwrote {path}"),
            Err(e) => eprintln!("\nfailed to write {path}: {e}"),
        }
    }
}
