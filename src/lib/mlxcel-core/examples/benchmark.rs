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

//! Benchmark for mlxcel-core low-level and high-level LLM operations

use mlxcel_core::{self, dtype};
use std::time::Instant;

// Typical LLM dimensions (Llama 8B-style)
const BATCH_SIZE: i32 = 1;
const SEQ_LEN: i32 = 1; // Single token generation
const HIDDEN_DIM: i32 = 4096;
const INTERMEDIATE_DIM: i32 = 14336; // 3.5x hidden
const NUM_HEADS: i32 = 32;
const HEAD_DIM: i32 = 128;

fn benchmark_cxx_matmul(iterations: usize) -> f64 {
    let shape_a = &[512_i32, 512];
    let shape_b = &[512_i32, 512];

    // Create arrays
    let a = mlxcel_core::ones(shape_a, dtype::FLOAT32);
    let b = mlxcel_core::ones(shape_b, dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        let c = mlxcel_core::matmul(&a, &b);
        mlxcel_core::eval(&c);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let c = mlxcel_core::matmul(&a, &b);
        mlxcel_core::eval(&c);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_cxx_add(iterations: usize) -> f64 {
    let shape = &[1024_i32, 1024];

    let a = mlxcel_core::ones(shape, dtype::FLOAT32);
    let b = mlxcel_core::ones(shape, dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        let c = mlxcel_core::add(&a, &b);
        mlxcel_core::eval(&c);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let c = mlxcel_core::add(&a, &b);
        mlxcel_core::eval(&c);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_cxx_chain_ops(iterations: usize) -> f64 {
    let shape = &[512_i32, 512];

    let a = mlxcel_core::ones(shape, dtype::FLOAT32);
    let b = mlxcel_core::ones(shape, dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        let c = mlxcel_core::add(&a, &b);
        let d = mlxcel_core::multiply(&c, &a);
        let e = mlxcel_core::subtract(&d, &b);
        mlxcel_core::eval(&e);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let c = mlxcel_core::add(&a, &b);
        let d = mlxcel_core::multiply(&c, &a);
        let e = mlxcel_core::subtract(&d, &b);
        mlxcel_core::eval(&e);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

// High-level LLM operation benchmarks

fn benchmark_rms_norm_high_level(iterations: usize) -> f64 {
    let x = mlxcel_core::ones(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], dtype::FLOAT32);
    let weight = mlxcel_core::ones(&[HIDDEN_DIM], dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        let out = mlxcel_core::rms_norm(&x, &weight, 1e-5);
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let out = mlxcel_core::rms_norm(&x, &weight, 1e-5);
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_rms_norm_decomposed(iterations: usize) -> f64 {
    // Simulate doing RMS norm with individual (decomposed) operations
    let x = mlxcel_core::ones(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], dtype::FLOAT32);
    let weight = mlxcel_core::ones(&[HIDDEN_DIM], dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        // RMS norm = x * rsqrt(mean(x^2) + eps) * weight
        let x_sq = mlxcel_core::square(&x);
        let mean_sq = mlxcel_core::mean_axis(&x_sq, -1, true);
        let eps_arr = mlxcel_core::full_f32(&[1], 1e-5, dtype::FLOAT32);
        let sum = mlxcel_core::add(&mean_sq, &eps_arr);
        let norm = mlxcel_core::rsqrt(&sum);
        let normalized = mlxcel_core::multiply(&x, &norm);
        let out = mlxcel_core::multiply(&normalized, &weight);
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let x_sq = mlxcel_core::square(&x);
        let mean_sq = mlxcel_core::mean_axis(&x_sq, -1, true);
        let eps_arr = mlxcel_core::full_f32(&[1], 1e-5, dtype::FLOAT32);
        let sum = mlxcel_core::add(&mean_sq, &eps_arr);
        let norm = mlxcel_core::rsqrt(&sum);
        let normalized = mlxcel_core::multiply(&x, &norm);
        let out = mlxcel_core::multiply(&normalized, &weight);
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_softmax_high_level(iterations: usize) -> f64 {
    // Attention scores: [batch, heads, seq, seq]
    let scores = mlxcel_core::ones(&[BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN], dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        let out = mlxcel_core::softmax(&scores, -1);
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let out = mlxcel_core::softmax(&scores, -1);
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_softmax_decomposed(iterations: usize) -> f64 {
    // Softmax decomposed: exp(x - max(x)) / sum(exp(x - max(x)))
    let scores = mlxcel_core::ones(&[BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN], dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        let max_val = mlxcel_core::max_axis(&scores, -1, true);
        let shifted = mlxcel_core::subtract(&scores, &max_val);
        let exp_x = mlxcel_core::exp(&shifted);
        let sum_exp = mlxcel_core::sum_axis(&exp_x, -1, true);
        let out = mlxcel_core::divide(&exp_x, &sum_exp);
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let max_val = mlxcel_core::max_axis(&scores, -1, true);
        let shifted = mlxcel_core::subtract(&scores, &max_val);
        let exp_x = mlxcel_core::exp(&shifted);
        let sum_exp = mlxcel_core::sum_axis(&exp_x, -1, true);
        let out = mlxcel_core::divide(&exp_x, &sum_exp);
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_swiglu_high_level(iterations: usize) -> f64 {
    // MLP input: [batch, seq, hidden]
    let x = mlxcel_core::ones(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], dtype::FLOAT32);
    // Gate/Up projection: [intermediate, hidden]
    let gate_proj = mlxcel_core::ones(&[INTERMEDIATE_DIM, HIDDEN_DIM], dtype::FLOAT32);
    let up_proj = mlxcel_core::ones(&[INTERMEDIATE_DIM, HIDDEN_DIM], dtype::FLOAT32);
    // Down projection: [hidden, intermediate]
    let down_proj = mlxcel_core::ones(&[HIDDEN_DIM, INTERMEDIATE_DIM], dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        let out = mlxcel_core::swiglu_mlp_forward(&x, &gate_proj, &up_proj, &down_proj);
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let out = mlxcel_core::swiglu_mlp_forward(&x, &gate_proj, &up_proj, &down_proj);
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_swiglu_decomposed(iterations: usize) -> f64 {
    // SwiGLU: down(silu(gate(x)) * up(x))
    let x = mlxcel_core::ones(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], dtype::FLOAT32);
    let gate_proj = mlxcel_core::ones(&[INTERMEDIATE_DIM, HIDDEN_DIM], dtype::FLOAT32);
    let up_proj = mlxcel_core::ones(&[INTERMEDIATE_DIM, HIDDEN_DIM], dtype::FLOAT32);
    let down_proj = mlxcel_core::ones(&[HIDDEN_DIM, INTERMEDIATE_DIM], dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        // gate(x) - matmul with transpose
        let gate_proj_t = mlxcel_core::transpose(&gate_proj);
        let gate = mlxcel_core::matmul(&x, &gate_proj_t);
        // up(x)
        let up_proj_t = mlxcel_core::transpose(&up_proj);
        let up = mlxcel_core::matmul(&x, &up_proj_t);
        // silu(gate) = gate * sigmoid(gate)
        let sigmoid_gate = mlxcel_core::sigmoid(&gate);
        let silu_gate = mlxcel_core::multiply(&gate, &sigmoid_gate);
        // silu(gate) * up
        let activated = mlxcel_core::multiply(&silu_gate, &up);
        // down(activated)
        let down_proj_t = mlxcel_core::transpose(&down_proj);
        let out = mlxcel_core::matmul(&activated, &down_proj_t);
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let gate_proj_t = mlxcel_core::transpose(&gate_proj);
        let gate = mlxcel_core::matmul(&x, &gate_proj_t);
        let up_proj_t = mlxcel_core::transpose(&up_proj);
        let up = mlxcel_core::matmul(&x, &up_proj_t);
        let sigmoid_gate = mlxcel_core::sigmoid(&gate);
        let silu_gate = mlxcel_core::multiply(&gate, &sigmoid_gate);
        let activated = mlxcel_core::multiply(&silu_gate, &up);
        let down_proj_t = mlxcel_core::transpose(&down_proj);
        let out = mlxcel_core::matmul(&activated, &down_proj_t);
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_attention_high_level(iterations: usize) -> f64 {
    // Q, K, V: [batch, heads, seq, head_dim]
    let q = mlxcel_core::ones(&[BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype::FLOAT32);
    let k = mlxcel_core::ones(&[BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype::FLOAT32);
    let v = mlxcel_core::ones(&[BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype::FLOAT32);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Warmup
    for _ in 0..5 {
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, scale, std::ptr::null())
        };
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, scale, std::ptr::null())
        };
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_attention_decomposed(iterations: usize) -> f64 {
    // SDPA decomposed: softmax(Q @ K^T / sqrt(d)) @ V
    let q = mlxcel_core::ones(&[BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype::FLOAT32);
    let k = mlxcel_core::ones(&[BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype::FLOAT32);
    let v = mlxcel_core::ones(&[BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype::FLOAT32);
    let scale_val = 1.0 / (HEAD_DIM as f32).sqrt();
    let scale = mlxcel_core::full_f32(&[1], scale_val, dtype::FLOAT32);
    // Axes for K^T: [batch, heads, seq, head_dim] -> [batch, heads, head_dim, seq]
    let transpose_axes = &[0_i32, 1, 3, 2];

    // Warmup
    for _ in 0..5 {
        // K^T: [batch, heads, head_dim, seq]
        let k_t = mlxcel_core::transpose_axes(&k, transpose_axes);
        // Q @ K^T
        let scores = mlxcel_core::matmul(&q, &k_t);
        // scores / sqrt(d)
        let scaled_scores = mlxcel_core::multiply(&scores, &scale);
        // softmax
        let attn_weights = mlxcel_core::softmax(&scaled_scores, -1);
        // attn @ V
        let out = mlxcel_core::matmul(&attn_weights, &v);
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let k_t = mlxcel_core::transpose_axes(&k, transpose_axes);
        let scores = mlxcel_core::matmul(&q, &k_t);
        let scaled_scores = mlxcel_core::multiply(&scores, &scale);
        let attn_weights = mlxcel_core::softmax(&scaled_scores, -1);
        let out = mlxcel_core::matmul(&attn_weights, &v);
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_linear_high_level(iterations: usize) -> f64 {
    let x = mlxcel_core::ones(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], dtype::FLOAT32);
    let weight = mlxcel_core::ones(&[HIDDEN_DIM, HIDDEN_DIM], dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        let out = unsafe { mlxcel_core::linear_forward(&x, &weight, std::ptr::null()) };
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let out = unsafe { mlxcel_core::linear_forward(&x, &weight, std::ptr::null()) };
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_linear_decomposed(iterations: usize) -> f64 {
    let x = mlxcel_core::ones(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], dtype::FLOAT32);
    let weight = mlxcel_core::ones(&[HIDDEN_DIM, HIDDEN_DIM], dtype::FLOAT32);

    // Warmup
    for _ in 0..5 {
        let w_t = mlxcel_core::transpose(&weight);
        let out = mlxcel_core::matmul(&x, &w_t);
        mlxcel_core::eval(&out);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let w_t = mlxcel_core::transpose(&weight);
        let out = mlxcel_core::matmul(&x, &w_t);
        mlxcel_core::eval(&out);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           mlxcel-core High-Level Operations Benchmark                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Batch size: {}", BATCH_SIZE);
    println!("  Sequence length: {}", SEQ_LEN);
    println!("  Hidden dim: {}", HIDDEN_DIM);
    println!("  Intermediate dim: {}", INTERMEDIATE_DIM);
    println!("  Num heads: {}", NUM_HEADS);
    println!("  Head dim: {}", HEAD_DIM);
    println!();

    let iterations = 100;

    // Low-level ops
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Low-level operations (baseline)");
    println!("───────────────────────────────────────────────────────────────────");

    let matmul_time = benchmark_cxx_matmul(iterations);
    println!("  Matmul (512x512):      {:.4} ms", matmul_time * 1000.0);

    let add_time = benchmark_cxx_add(iterations);
    println!("  Add (1024x1024):       {:.4} ms", add_time * 1000.0);

    let chain_time = benchmark_cxx_chain_ops(iterations);
    println!("  Chain ops (add->mul->sub): {:.4} ms", chain_time * 1000.0);

    // High-level vs decomposed comparison
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("High-level ops vs Decomposed (FFI call reduction benefit)");
    println!("───────────────────────────────────────────────────────────────────");

    let rms_high = benchmark_rms_norm_high_level(iterations);
    let rms_decomp = benchmark_rms_norm_decomposed(iterations);
    let rms_speedup = rms_decomp / rms_high;
    println!("RMS Norm:");
    println!("  High-level:    {:.4} ms", rms_high * 1000.0);
    println!(
        "  Decomposed:    {:.4} ms  (7 FFI calls)",
        rms_decomp * 1000.0
    );
    println!("  Speedup:       {:.2}x", rms_speedup);
    println!();

    let soft_high = benchmark_softmax_high_level(iterations);
    let soft_decomp = benchmark_softmax_decomposed(iterations);
    let soft_speedup = soft_decomp / soft_high;
    println!("Softmax:");
    println!("  High-level:    {:.4} ms", soft_high * 1000.0);
    println!(
        "  Decomposed:    {:.4} ms  (5 FFI calls)",
        soft_decomp * 1000.0
    );
    println!("  Speedup:       {:.2}x", soft_speedup);
    println!();

    let linear_high = benchmark_linear_high_level(iterations);
    let linear_decomp = benchmark_linear_decomposed(iterations);
    let linear_speedup = linear_decomp / linear_high;
    println!("Linear:");
    println!("  High-level:    {:.4} ms", linear_high * 1000.0);
    println!(
        "  Decomposed:    {:.4} ms  (2 FFI calls)",
        linear_decomp * 1000.0
    );
    println!("  Speedup:       {:.2}x", linear_speedup);
    println!();

    let attn_high = benchmark_attention_high_level(iterations);
    let attn_decomp = benchmark_attention_decomposed(iterations);
    let attn_speedup = attn_decomp / attn_high;
    println!("Scaled Dot-Product Attention:");
    println!("  High-level:    {:.4} ms", attn_high * 1000.0);
    println!(
        "  Decomposed:    {:.4} ms  (5 FFI calls)",
        attn_decomp * 1000.0
    );
    println!("  Speedup:       {:.2}x", attn_speedup);
    println!();

    let swiglu_high = benchmark_swiglu_high_level(iterations);
    let swiglu_decomp = benchmark_swiglu_decomposed(iterations);
    let swiglu_speedup = swiglu_decomp / swiglu_high;
    println!("SwiGLU MLP:");
    println!("  High-level:    {:.4} ms", swiglu_high * 1000.0);
    println!(
        "  Decomposed:    {:.4} ms  (9 FFI calls)",
        swiglu_decomp * 1000.0
    );
    println!("  Speedup:       {:.2}x", swiglu_speedup);
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Summary");
    println!("───────────────────────────────────────────────────────────────────");

    // Estimate total FFI calls per transformer layer (approximation)
    // A typical layer has: 2x RMS norm, 4x linear (Q,K,V,O), 1x SDPA, 3x linear (gate,up,down)
    // Decomposed: ~2*7 + 4*2 + 5 + 3*2 = 35 FFI calls per layer
    // High-level: ~2 + 4 + 1 + 1 = 8 FFI calls per layer
    let est_decomposed_per_layer =
        2.0 * rms_decomp + 4.0 * linear_decomp + attn_decomp + swiglu_decomp;
    let est_highlevel_per_layer = 2.0 * rms_high + 4.0 * linear_high + attn_high + swiglu_high;
    let layer_speedup = est_decomposed_per_layer / est_highlevel_per_layer;

    println!("Estimated per-layer time:");
    println!(
        "  High-level ops:  {:.4} ms",
        est_highlevel_per_layer * 1000.0
    );
    println!(
        "  Decomposed ops:  {:.4} ms",
        est_decomposed_per_layer * 1000.0
    );
    println!("  Potential speedup: {:.2}x", layer_speedup);
    println!();

    // For a 32-layer model generating 50 tokens
    let tokens = 50;
    let layers = 32;
    let total_decomposed = est_decomposed_per_layer * layers as f64 * tokens as f64;
    let total_highlevel = est_highlevel_per_layer * layers as f64 * tokens as f64;

    println!("Projected generation time (32 layers, 50 tokens):");
    println!(
        "  High-level ops:  {:.2} ms ({:.2} tok/s)",
        total_highlevel * 1000.0,
        tokens as f64 / total_highlevel
    );
    println!(
        "  Decomposed ops:  {:.2} ms ({:.2} tok/s)",
        total_decomposed * 1000.0,
        tokens as f64 / total_decomposed
    );
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
}
