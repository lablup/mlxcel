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
//! - **rejection** is `fused_sample_rejection`, the kernel: one softmax and a
//!   shrinking probability interval resolved by two pivots per vocabulary
//!   sweep, plus the host readback of the per-row converged flag and the
//!   `argpartition` fallback the production path pays for.
//!
//! The rejection arm is deliberately timed WITH the readback. That readback is
//! a device sync inside the sampler, and hiding it would make the number
//! unrepresentative of what a decode step actually costs.
//!
//! Both arms are timed the same way: construct the graph, `eval`, then
//! `synchronize_default()` so the GPU work is retired inside the timed region.
//! The reported number is the median over `--iters` repetitions after
//! `--warmup` discarded ones.
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
    MlxArray, UniquePtr, array_to_raw_bytes, eval, from_slice_f32, fused_sample_categorical,
    fused_sample_rejection, is_gpu_available, random_seed, rejection_cap_overflow_launches,
    rejection_cap_overflow_rows, reset_sampling_dispatch, sampling_dispatch_recorded_report,
    sampling_rejection_available, sampling_rejection_max_rounds, sampling_rejection_probe,
    sampling_rejection_routes, synchronize_default,
};

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
    print_dispatch("  rejection arm -> ");
    println!();

    println!(
        "{:>12} {:>8} {:>6} {:>7} {:>7} {:>14} {:>15} {:>9}",
        "config", "vocab", "batch", "rounds", "routed", "baseline_us", "rejection_us", "speedup"
    );

    let mut csv =
        String::from("config,vocab,batch,worst_rounds,routed,baseline_us,rejection_us,speedup\n");
    for (label, top_k, top_p, min_p) in CONFIGS {
        for vocab in VOCABS {
            for batch in BATCHES {
                let logits = synthetic_logits(batch, vocab);
                let rounds = worst_rounds(&logits, batch, top_k, top_p, min_p, cap);

                // Reseed before each arm so both consume the same RNG stream
                // position; the draw itself is part of what is being measured.
                random_seed(0x0901_BEEF);
                let baseline = time_arm(&logits, &opts, |x| {
                    fused_sample_categorical(x, TEMPERATURE, top_k, top_p, min_p)
                });
                random_seed(0x0901_BEEF);
                let rejection = time_arm(&logits, &opts, |x| {
                    fused_sample_rejection(x, TEMPERATURE, top_k, top_p, min_p, cap)
                });

                let baseline_us = baseline.as_secs_f64() * 1e6;
                let rejection_us = rejection.as_secs_f64() * 1e6;
                let speedup = baseline_us / rejection_us;
                let routed = sampling_rejection_routes(vocab, top_k, top_p, min_p);
                let routed_text = if routed { "yes" } else { "no" };
                println!(
                    "{label:>12} {vocab:>8} {batch:>6} {rounds:>7} {routed_text:>7} \
                     {baseline_us:>14.2} {rejection_us:>15.2} {speedup:>8.2}x"
                );
                let _ = writeln!(
                    csv,
                    "{label},{vocab},{batch},{rounds},{routed_text},{baseline_us:.3},\
                     {rejection_us:.3},{speedup:.4}"
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
    print_dispatch("dispatch: ");

    if let Some(path) = opts.csv {
        match std::fs::write(&path, csv) {
            Ok(()) => println!("\nwrote {path}"),
            Err(e) => eprintln!("\nfailed to write {path}: {e}"),
        }
    }
}
