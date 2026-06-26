//! Rust-native StableHLO text emitter for Llama-3.2-1B (spike #451).
//!
//! Usage:
//!   emit p0      [out.mlir]   single dot_general, toolchain round-trip gate (P0)
//!   emit probe   [out.mlir]   syntax probe for the riskiest op forms
//!   emit decode  [out.mlir]   full Llama-3.2-1B decode_step (P1)
//!   emit prefill [out.mlir]   full Llama-3.2-1B bucketed prefill

mod builder;
mod config;
mod model;
mod rope;

use std::io::Write;

fn p0_matmul() -> String {
    // [4,8] x [8,3] -> [4,3], the minimal toolchain round-trip.
    "module {\n  func.func public @main(%a: tensor<4x8xf32>, %b: tensor<8x3xf32>) -> tensor<4x3xf32> {\n    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x3xf32>) -> tensor<4x3xf32>\n    return %0 : tensor<4x3xf32>\n  }\n}\n".to_string()
}

fn probe() -> String {
    // Exercises the op forms not covered by the plain matmul, especially
    // dynamic_update_slice (JAX used scatter, so this spelling is novel here),
    // a batched dot_general, dynamic_slice, and reduce.
    "module {\n  func.func public @main(%cache: tensor<2x4x3xf32>, %upd: tensor<1x1x3xf32>, %idx: tensor<i32>, %q: tensor<2x4x5xf32>, %kv: tensor<6x2x5xf32>, %scores: tensor<8x7xf32>) -> (tensor<2x4x3xf32>, tensor<3x3xf32>, tensor<2x4x6xf32>, tensor<8xf32>) {\n    %c0 = stablehlo.constant dense<0> : tensor<i32>\n    %c1 = stablehlo.constant dense<1> : tensor<i32>\n    %0 = stablehlo.dynamic_update_slice %cache, %upd, %c1, %idx, %c0 : (tensor<2x4x3xf32>, tensor<1x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x3xf32>\n    %1 = stablehlo.dynamic_slice %cache, %c1, %c0, %c0, sizes = [1, 4, 3] : (tensor<2x4x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x3xf32>\n    %2 = stablehlo.reshape %1 : (tensor<1x4x3xf32>) -> tensor<4x3xf32>\n    %3 = stablehlo.slice %2 [1:4, 0:3] : (tensor<4x3xf32>) -> tensor<3x3xf32>\n    %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>\n    %4 = stablehlo.reduce(%scores init: %init) applies stablehlo.add across dimensions = [1] : (tensor<8x7xf32>, tensor<f32>) -> tensor<8xf32>\n    %5 = stablehlo.dot_general %q, %kv, batching_dims = [0] x [1], contracting_dims = [2] x [2] : (tensor<2x4x5xf32>, tensor<6x2x5xf32>) -> tensor<2x4x6xf32>\n    return %0, %3, %5, %4 : tensor<2x4x3xf32>, tensor<3x3xf32>, tensor<2x4x6xf32>, tensor<8xf32>\n  }\n}\n".to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map(|s| s.as_str()).unwrap_or("decode");
    let out = args.get(2);

    let text = match kind {
        "p0" => p0_matmul(),
        "probe" => probe(),
        "decode" => model::emit_decode(&config::Config::llama_3_2_1b()),
        "prefill" => model::emit_prefill(&config::Config::llama_3_2_1b()),
        other => {
            eprintln!("unknown kind: {other} (use p0 | probe | decode | prefill)");
            std::process::exit(2);
        }
    };

    match out {
        Some(path) => {
            let mut f = std::fs::File::create(path).expect("create output file");
            f.write_all(text.as_bytes()).expect("write output");
            eprintln!("wrote {} ({} bytes)", path, text.len());
        }
        None => {
            print!("{text}");
        }
    }
}
