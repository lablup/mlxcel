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

//! Compare the native paged decode kernel against the dense gather fallback.
//!
//! This bypasses the server and scheduler so the measurement focuses on the
//! model-level batched decode hot path.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use mlxcel::{DecodeBatchContext, LanguageModel};

#[derive(Parser, Debug)]
#[command(name = "profile_paged_decode_kernel")]
struct Args {
    /// Model path.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Batch size for batched decode.
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Number of timed decode steps.
    #[arg(long, default_value = "50")]
    decode_steps: usize,

    /// Warmup decode steps.
    #[arg(long, default_value = "10")]
    warmup: usize,

    /// Number of benchmark runs.
    #[arg(long, default_value = "3")]
    runs: usize,

    /// Prompt length for prefill before decode.
    #[arg(long, default_value = "32")]
    prompt_len: usize,

    /// Logical paged block size.
    #[arg(long, default_value = "32")]
    block_size: i32,
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

fn run_profile<M: LanguageModel>(
    model: &M,
    batch_size: usize,
    prompt_len: usize,
    warmup: usize,
    decode_steps: usize,
    context: DecodeBatchContext,
) -> f64 {
    let prompt_tokens: Vec<i32> = (1..=prompt_len as i32).collect();
    let prompt = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_len as i32]);
    let mask = if prompt_len > 1 {
        Some(mlxcel_core::utils::create_causal_mask(prompt_len as i32, 0))
    } else {
        None
    };

    let mut all_caches: Vec<Vec<mlxcel_core::layers::KVCache>> =
        (0..batch_size).map(|_| model.make_caches()).collect();
    for caches in &mut all_caches {
        let logits = model.forward(&prompt, caches.as_mut_slice(), mask.as_deref());
        mlxcel_core::eval(&logits);
    }

    let decode_tokens: Vec<i32> = vec![42; batch_size];
    for _ in 0..warmup {
        let input = mlxcel_core::from_slice_i32(&decode_tokens, &[batch_size as i32, 1]);
        let mut batch_cache_refs: Vec<&mut [mlxcel_core::layers::KVCache]> =
            all_caches.iter_mut().map(|c| c.as_mut_slice()).collect();
        let logits =
            model.forward_batched_with_context(&input, &mut batch_cache_refs, None, Some(&context));
        mlxcel_core::eval(&logits);
    }

    let start = Instant::now();
    for _ in 0..decode_steps {
        let input = mlxcel_core::from_slice_i32(&decode_tokens, &[batch_size as i32, 1]);
        let mut batch_cache_refs: Vec<&mut [mlxcel_core::layers::KVCache]> =
            all_caches.iter_mut().map(|c| c.as_mut_slice()).collect();
        let logits =
            model.forward_batched_with_context(&input, &mut batch_cache_refs, None, Some(&context));
        mlxcel_core::eval(&logits);
    }
    let elapsed = start.elapsed();
    (decode_steps * batch_size) as f64 / elapsed.as_secs_f64()
}

fn main() {
    let args = Args::parse();

    mlxcel::initialize_runtime();
    let (model, _tokenizer) = mlxcel::load_model(&args.model).expect("failed to load model");
    assert!(
        model.supports_batching(),
        "paged decode kernel profiling requires a batching-capable model"
    );

    let fallback_ctx = DecodeBatchContext::paged_with_native(args.block_size, false);
    let native_ctx = DecodeBatchContext::paged_with_native(args.block_size, true);
    let mut fallback_runs = Vec::with_capacity(args.runs);
    let mut native_runs = Vec::with_capacity(args.runs);

    for _ in 0..args.runs {
        fallback_runs.push(run_profile(
            &model,
            args.batch_size,
            args.prompt_len,
            args.warmup,
            args.decode_steps,
            fallback_ctx,
        ));
        native_runs.push(run_profile(
            &model,
            args.batch_size,
            args.prompt_len,
            args.warmup,
            args.decode_steps,
            native_ctx,
        ));
    }

    let fallback_tps = median(&mut fallback_runs);
    let native_tps = median(&mut native_runs);

    println!("model={}", args.model.display());
    println!(
        "batch_size={} prompt_len={} decode_steps={} block_size={}",
        args.batch_size, args.prompt_len, args.decode_steps, args.block_size
    );
    println!("fallback_tok_per_sec={fallback_tps:.2}");
    println!("native_tok_per_sec={native_tps:.2}");
    println!("speedup={:.3}x", native_tps / fallback_tps);
}
