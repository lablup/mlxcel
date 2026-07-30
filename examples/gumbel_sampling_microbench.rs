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

//! Sampling-step microbenchmark for the Gumbel-max kernel (issue #900).
//!
//! Measures the sampling step *alone*, with no model attached: a synthetic
//! `[batch, vocab]` logits tensor is sampled repeatedly and the two arms are
//! compared at every point of the vocab x batch matrix the issue specifies.
//!
//! - **baseline** is `fused_sample_categorical`, the pre-#900 path: temperature
//!   scale plus `mlx::core::random::categorical`, which normalises over the
//!   whole vocabulary per token per row.
//! - **gumbel** is `gumbel_max_sample`, the kernel: one index-carrying max
//!   reduction with in-kernel Philox noise, no softmax and no sort.
//!
//! Both arms are timed the same way: construct the graph, `eval`, then
//! `synchronize_default()` so the GPU work is actually retired inside the
//! timed region. The reported number is the median over `--iters` repetitions
//! after `--warmup` discarded ones, which is the statistic the autotuner uses
//! (issue #906) and is robust to the occasional scheduler hiccup that would
//! skew a mean.
//!
//! No model is loaded and no tokenizer is touched, so this isolates the
//! sampling step from decode. Pair it with a real end-to-end decode run to see
//! the throughput effect at batch 1 and batch 4.
//!
//! **Memory mode: warm** (see `docs/benchmarks.md`). The logits tensor is
//! allocated once and reused across iterations, so it is resident in the
//! last-level cache after the first read. Unlike the paged-KV harnesses, that
//! is the *representative* mode here rather than an upper bound: in production
//! the logits row is written by the LM head immediately before the sampler
//! reads it, so it is warm on a real decode step too. The largest point in the
//! matrix (batch 8, vocab 152064) reads under 5 MiB, well inside any
//! last-level cache, so there is no cold variant to run. Both arms read the
//! identical buffer, so the comparison is unaffected either way.
//!
//! Run (Apple):
//!   cargo run --release --features metal,accelerate \
//!     --example gumbel_sampling_microbench
//! Run (CUDA):
//!   cargo run --release --features cuda --example gumbel_sampling_microbench
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
    MlxArray, UniquePtr, eval, from_slice_f32, fused_sample_categorical, gumbel_max_sample,
    gumbel_sample_num_splits, is_gpu_available, random_seed, sampling_gumbel_available,
    synchronize_default,
};

const VOCABS: [i32; 3] = [32_768, 65_536, 152_064];
const BATCHES: [i32; 3] = [1, 4, 8];
const TEMPERATURE: f32 = 1.0;

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
/// low-probability floor with a handful of dominant tokens. The exact values do
/// not matter for timing (neither arm is data-dependent), but a degenerate
/// all-equal tensor would be an unrepresentative input for the categorical
/// arm's normalisation.
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

fn main() {
    let opts = parse_options();

    if !is_gpu_available() {
        eprintln!("No GPU backend available; the sampling kernel cannot run here.");
        return;
    }
    println!(
        "Gumbel-max sampling microbenchmark (#900)  iters={} warmup={}  \
         routing_enabled={}",
        opts.iters,
        opts.warmup,
        sampling_gumbel_available()
    );
    println!(
        "{:>8} {:>6} {:>7} {:>14} {:>14} {:>9}",
        "vocab", "batch", "splits", "baseline_us", "gumbel_us", "speedup"
    );

    let mut csv = String::from("vocab,batch,splits,baseline_us,gumbel_us,speedup\n");
    for vocab in VOCABS {
        for batch in BATCHES {
            let logits = synthetic_logits(batch, vocab);
            let splits = gumbel_sample_num_splits(batch, vocab);

            // Reseed before each arm so both consume the same RNG stream
            // position; the draw itself is part of what is being measured.
            random_seed(0x0900_BEEF);
            let baseline = time_arm(&logits, &opts, |x| {
                fused_sample_categorical(x, TEMPERATURE, 0, 1.0, 0.0)
            });
            random_seed(0x0900_BEEF);
            let gumbel = time_arm(&logits, &opts, |x| gumbel_max_sample(x, TEMPERATURE));

            let baseline_us = baseline.as_secs_f64() * 1e6;
            let gumbel_us = gumbel.as_secs_f64() * 1e6;
            let speedup = baseline_us / gumbel_us;
            println!(
                "{vocab:>8} {batch:>6} {splits:>7} {baseline_us:>14.2} \
                 {gumbel_us:>14.2} {speedup:>8.2}x"
            );
            let _ = writeln!(
                csv,
                "{vocab},{batch},{splits},{baseline_us:.3},{gumbel_us:.3},{speedup:.4}"
            );
        }
    }

    if let Some(path) = opts.csv {
        match std::fs::write(&path, csv) {
            Ok(()) => println!("\nwrote {path}"),
            Err(e) => eprintln!("\nfailed to write {path}: {e}"),
        }
    }
}
