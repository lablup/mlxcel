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

//! What a speculative verify block costs at each width, in pipelined decode
//! steps (issue #2091).
//!
//! Loads a model, prefills a fixed 300-token prompt, then times:
//!
//! - a pipelined one-token step, as `CxxGenerator` runs it (the next step is
//!   submitted from the still-lazy argmax before the host reads it);
//! - a synchronous forward of `w` tokens plus the argmax and its host read,
//!   for `w` in 1..=10, trimming `w - 1` positions afterwards so the cache
//!   grows one token per round like a rejected verify.
//!
//! Each width reports its median over 40 rounds and its ratio to the
//! pipelined step. `DraftPolicy` in `speculative/prompt_lookup.rs` quotes
//! these ratios for GB10; rerun this after an MLX pin bump that moves the
//! quantized-matmul kernel boundaries.
//!
//! ```bash
//! cargo build --release --features cuda --example verify_width_cost
//! gpu-lock run --tag verify-width -- \
//!     target/release/examples/verify_width_cost models/mlx/qwen3-8b-4bit
//! ```
fn main() {
    use mlxcel::generate::LanguageModel;
    use std::path::Path;
    use std::time::Instant;

    let path = std::env::args().nth(1).expect("model path");
    let (model, _) = mlxcel::load_model(Path::new(&path)).unwrap();
    // The rounds below roll a rejected block back with a cache trim; on a
    // model that cannot do that the cache would grow `w` tokens per round and
    // skew every ratio without saying so.
    assert!(
        mlxcel::supports_prompt_lookup(&model),
        "{path}: verify blocks cannot be rolled back on this model"
    );
    let prompt: Vec<i32> = (0..300).map(|i| 1000 + (i * 37) % 5000).collect();

    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    let fresh = || {
        let mut caches = model.make_caches();
        let input = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);
        let logits = model.forward(&input, &mut caches, None);
        mlxcel_core::eval(&logits);
        caches
    };

    // Pipelined one-token steps, as CxxGenerator runs them.
    let pipelined = |n: usize| {
        let mut caches = fresh();
        let mut y = mlxcel_core::from_slice_i32(&[1234], &[1, 1]);
        let start = Instant::now();
        let mut prev = mlxcel_core::reshape(&y, &[1, 1]);
        for _ in 0..n {
            let logits = model.forward(&y, &mut caches, None);
            let tok = mlxcel_core::argmax_last_axis(&logits);
            let next = mlxcel_core::reshape(&tok, &[1, 1]);
            mlxcel_core::async_eval(&next);
            let _ = mlxcel_core::item_i32(&prev);
            prev = mlxcel_core::reshape(&next, &[1, 1]);
            y = next;
        }
        let _ = mlxcel_core::item_i32(&prev);
        start.elapsed().as_secs_f64() * 1000.0 / n as f64
    };

    // Synchronous width-w forwards; keep one token per round like a rejected
    // verify, so the cache grows the way the decode loop grows it.
    let sync_width = |w: usize, n: usize| {
        let mut caches = fresh();
        let mut times = Vec::new();
        for i in 0..n + 5 {
            let tokens: Vec<i32> = (0..w)
                .map(|j| 2000 + ((i * 7 + j * 13) % 3000) as i32)
                .collect();
            let start = Instant::now();
            let input = mlxcel_core::from_slice_i32(&tokens, &[1, w as i32]);
            let logits = model.forward(&input, &mut caches, None);
            let am = mlxcel_core::argmax_last_axis(&logits);
            mlxcel_core::eval(&am);
            let _ = mlxcel_core::array_to_raw_bytes(&am);
            for c in caches.iter_mut() {
                c.trim(w as i32 - 1);
            }
            if i >= 5 {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        median(times)
    };

    for _ in 0..2 {
        let _ = pipelined(20);
        for w in 1..=10 {
            let _ = sync_width(w, 3);
        }
    }
    let pipe = median((0..5).map(|_| pipelined(60)).collect());
    println!("pipelined step: {pipe:.3} ms");
    for w in 1..=10 {
        let t = sync_width(w, 40);
        println!(
            "sync width {w:2}: {t:.3} ms = {:.2} pipelined steps",
            t / pipe
        );
    }
}
