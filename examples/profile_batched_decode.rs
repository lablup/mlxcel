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

//! Direct profiling of forward() vs forward_batched() decode performance.
//!
//! Bypasses the server and scheduler to measure raw model-level throughput.
//! This isolates whether the split-attention batched decode actually saves
//! time compared to sequential forward() calls.
//!
//! # Usage
//! ```bash
//! cargo run --release --example profile_batched_decode -- \
//!   -m models/llama3.1-8b-4bit \
//!   --batch-sizes 1,2,4,8 \
//!   --decode-steps 50 \
//!   --warmup 5
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use mlxcel::LanguageModel;

/// Profile batched vs sequential decode at the model level.
#[derive(Parser, Debug)]
#[command(name = "profile_batched_decode")]
struct Args {
    /// Model path.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Comma-separated batch sizes to test.
    #[arg(long, default_value = "1,2,4,8", value_delimiter = ',')]
    batch_sizes: Vec<usize>,

    /// Number of decode steps per measurement.
    #[arg(long, default_value = "50")]
    decode_steps: usize,

    /// Warmup decode steps (discarded).
    #[arg(long, default_value = "10")]
    warmup: usize,

    /// Number of runs per configuration (reports median).
    #[arg(long, default_value = "3")]
    runs: usize,

    /// Prompt tokens to prefill before decode.
    #[arg(long, default_value = "8")]
    prompt_len: usize,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

fn main() {
    let args = Args::parse();

    // Initialize MLX runtime
    mlxcel::initialize_runtime();

    println!("=== Batched Decode Profiling ===");
    println!("Model: {}", args.model.display());
    println!(
        "Config: decode_steps={}, warmup={}, runs={}, prompt_len={}",
        args.decode_steps, args.warmup, args.runs, args.prompt_len
    );
    println!();

    // Load model
    let load_start = Instant::now();
    let (model, _tokenizer) = mlxcel::load_model(&args.model).expect("Failed to load model");
    println!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f64());
    println!("  supports_batching: {}", model.supports_batching());
    println!("  num_layers: {}", model.num_layers());
    println!();

    // Create fake prompt tokens
    let prompt_tokens: Vec<i32> = (1..=args.prompt_len as i32).collect();

    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Batch Size", "Seq TPS", "Batch TPS", "Speedup", "Seq ms/step", "Bat ms/step"
    );
    println!("{}", "-".repeat(78));

    for &batch_size in &args.batch_sizes {
        // --- Sequential path: B separate forward() calls ---
        let mut seq_tps_values: Vec<f64> = Vec::new();

        for _run in 0..args.runs {
            // Create B independent cache sets and prefill each
            let mut all_caches: Vec<Vec<mlxcel_core::layers::KVCache>> =
                (0..batch_size).map(|_| model.make_caches()).collect();

            // Prefill each sequence
            let prompt_array =
                mlxcel_core::from_slice_i32(&prompt_tokens, &[1, args.prompt_len as i32]);
            let mask = if args.prompt_len > 1 {
                Some(mlxcel_core::utils::create_causal_mask(
                    args.prompt_len as i32,
                    0,
                ))
            } else {
                None
            };

            for caches in &mut all_caches {
                let logits = model.forward(&prompt_array, caches.as_mut_slice(), mask.as_deref());
                mlxcel_core::eval(&logits);
            }

            // Dummy last token for decode
            let last_tokens: Vec<i32> = vec![42; batch_size];

            // Warmup decode steps (sequential)
            for _step in 0..args.warmup {
                for (i, caches) in all_caches.iter_mut().enumerate() {
                    let input = mlxcel_core::from_slice_i32(&[last_tokens[i]], &[1, 1]);
                    let logits = model.forward(&input, caches.as_mut_slice(), None);
                    mlxcel_core::eval(&logits);
                }
            }

            // Timed decode steps (sequential: B × forward per step)
            let start = Instant::now();
            for _step in 0..args.decode_steps {
                for (i, caches) in all_caches.iter_mut().enumerate() {
                    let input = mlxcel_core::from_slice_i32(&[last_tokens[i]], &[1, 1]);
                    let logits = model.forward(&input, caches.as_mut_slice(), None);
                    mlxcel_core::eval(&logits);
                }
            }
            let elapsed = start.elapsed();
            let total_tokens = args.decode_steps * batch_size;
            let tps = total_tokens as f64 / elapsed.as_secs_f64();
            seq_tps_values.push(tps);
        }

        let seq_tps = median(&mut seq_tps_values);

        // --- Batched path: 1 forward_batched() call for B sequences ---
        let mut batch_tps_values: Vec<f64> = Vec::new();

        if model.supports_batching() && batch_size > 1 {
            for _run in 0..args.runs {
                let mut all_caches: Vec<Vec<mlxcel_core::layers::KVCache>> =
                    (0..batch_size).map(|_| model.make_caches()).collect();

                // Prefill each sequence
                let prompt_array =
                    mlxcel_core::from_slice_i32(&prompt_tokens, &[1, args.prompt_len as i32]);
                let mask = if args.prompt_len > 1 {
                    Some(mlxcel_core::utils::create_causal_mask(
                        args.prompt_len as i32,
                        0,
                    ))
                } else {
                    None
                };

                for caches in &mut all_caches {
                    let logits =
                        model.forward(&prompt_array, caches.as_mut_slice(), mask.as_deref());
                    mlxcel_core::eval(&logits);
                }

                let last_tokens: Vec<i32> = vec![42; batch_size];

                // Warmup with batched decode
                for _step in 0..args.warmup {
                    let input = mlxcel_core::from_slice_i32(&last_tokens, &[batch_size as i32, 1]);
                    let mut batch_cache_refs: Vec<&mut [mlxcel_core::layers::KVCache]> =
                        all_caches.iter_mut().map(|c| c.as_mut_slice()).collect();
                    let logits = model.forward_batched(&input, &mut batch_cache_refs, None);
                    mlxcel_core::eval(&logits);
                }

                // Timed batched decode steps (1 × forward_batched per step)
                let start = Instant::now();
                for _step in 0..args.decode_steps {
                    let input = mlxcel_core::from_slice_i32(&last_tokens, &[batch_size as i32, 1]);
                    let mut batch_cache_refs: Vec<&mut [mlxcel_core::layers::KVCache]> =
                        all_caches.iter_mut().map(|c| c.as_mut_slice()).collect();
                    let logits = model.forward_batched(&input, &mut batch_cache_refs, None);
                    mlxcel_core::eval(&logits);
                }
                let elapsed = start.elapsed();
                let total_tokens = args.decode_steps * batch_size;
                let tps = total_tokens as f64 / elapsed.as_secs_f64();
                batch_tps_values.push(tps);
            }
        }

        let batch_tps = if batch_tps_values.is_empty() {
            seq_tps
        } else {
            median(&mut batch_tps_values)
        };

        let speedup = batch_tps / seq_tps;
        let seq_ms_step = 1000.0 * args.decode_steps as f64
            / (seq_tps / batch_size as f64)
            / args.decode_steps as f64;
        let batch_ms_step = 1000.0 * args.decode_steps as f64
            / (batch_tps / batch_size as f64)
            / args.decode_steps as f64;

        println!(
            "{:<12} {:>12.1} {:>12.1} {:>12.2}x {:>12.2} {:>12.2}",
            batch_size, seq_tps, batch_tps, speedup, seq_ms_step, batch_ms_step
        );
    }

    println!();
    println!("Legend:");
    println!("  Seq TPS    = total tok/s using B × forward() per step");
    println!("  Batch TPS  = total tok/s using 1 × forward_batched() per step");
    println!("  Speedup    = Batch TPS / Seq TPS (>1.0 means batching wins)");
    println!("  ms/step    = wall time per decode step (all B sequences)");
}
