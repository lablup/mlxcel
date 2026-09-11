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

fn main() {
    use mlxcel::generate::LanguageModel;
    use mlxcel_core::layers::KVCache;
    use std::path::Path;
    use std::time::Instant;

    let (model, _) =
        mlxcel::load_model(Path::new("models/Meta-Llama-3.1-8B-Instruct-4bit")).unwrap();

    let input = mlxcel_core::from_slice_i32(&[9906], &[1, 1]);

    // Warmup
    println!("=== Rust mlxcel Forward Pass Benchmark ===");
    println!("Warming up...");
    for _ in 0..10 {
        let mut caches: Vec<KVCache> = model.make_caches();
        let logits = model.forward(&input, &mut caches, None);
        mlxcel_core::eval(&logits);
    }
    mlxcel_core::synchronize_default();

    // First token test - fresh cache each time
    let num_runs = 50;
    let mut first_token_times = Vec::new();
    for _ in 0..num_runs {
        let mut caches: Vec<KVCache> = model.make_caches();
        mlxcel_core::synchronize_default();

        let start = Instant::now();
        let logits = model.forward(&input, &mut caches, None);
        mlxcel_core::eval(&logits);
        mlxcel_core::synchronize_default();
        first_token_times.push(start.elapsed());
    }

    // Decode test - reuse cache
    let mut caches: Vec<KVCache> = model.make_caches();
    let _ = model.forward(&input, &mut caches, None);
    mlxcel_core::synchronize_default();

    let mut decode_times = Vec::new();
    for _ in 0..num_runs {
        mlxcel_core::synchronize_default();
        let start = Instant::now();
        let logits = model.forward(&input, &mut caches, None);
        mlxcel_core::eval(&logits);
        mlxcel_core::synchronize_default();
        decode_times.push(start.elapsed());
    }

    // Statistics
    let first_avg = first_token_times
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / first_token_times.len() as f64;
    let first_min = first_token_times
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    let first_max = first_token_times
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .fold(0.0, f64::max);

    let decode_avg = decode_times
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / decode_times.len() as f64;
    let decode_min = decode_times
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    let decode_max = decode_times
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .fold(0.0, f64::max);

    println!();
    println!("First token (fresh cache, n={}):", num_runs);
    println!("  Avg: {:.2} ms", first_avg);
    println!("  Min: {:.2} ms", first_min);
    println!("  Max: {:.2} ms", first_max);
    println!("  Implied tok/s: {:.2}", 1000.0 / first_avg);
    println!();
    println!("Decode token (positions 1-{}, n={}):", num_runs, num_runs);
    println!("  Avg: {:.2} ms", decode_avg);
    println!("  Min: {:.2} ms", decode_min);
    println!("  Max: {:.2} ms", decode_max);
    println!("  Implied tok/s: {:.2}", 1000.0 / decode_avg);
    println!();
    println!("Target (Python mlx-lm): ~11.36 ms / ~88 tok/s");
}
