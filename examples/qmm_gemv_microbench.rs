// Quantized-matmul GEMV effective-bandwidth microbench.
//
// Motivated by the #680 investigation: gemma-4-12b decode streams ~10.1 GB
// of weights per token but sustains only ~140-150 GB/s effective on GB10
// (~52% of the 273 GB/s peak), while llama-3.1-8b reaches ~200+ GB/s
// (~75%). Decode is weight-bandwidth-bound, so the per-shape efficiency of
// the quantized-matmul GEMV kernels (B=1, single token) IS the decode
// throughput. This bench measures achieved GB/s per (out, in, bits) layer
// shape, decoupled from the model runtime.
//
// Method: for each shape, quantize a bf16 weight at group_size 64 into
// `copies` independent buffers (round-robin so the working set exceeds L2
// and every read hits DRAM), then time batches of T back-to-back
// `quantized_matmul` GEMVs whose outputs are folded with tiny adds into a
// single root so one eval submits the whole batch (the production decode
// regime: many qmm nodes per step, graphs split by MLX's Spark caps).
// Effective GB/s = T * bytes(w + scales + biases) / batch_time.
//
// Usage: cargo run --release --features cuda --example qmm_gemv_microbench
//          [ T_PER_BATCH ] [ ROUNDS ]
// Defaults: T_PER_BATCH=64, ROUNDS=5 (best round reported).

use mlxcel_core::{
    MlxArray, UniquePtr, add, astype, eval, from_slice_f32, quantize_weights_biases,
    quantize_weights_scales, quantize_weights_w, quantized_matmul, synchronize_default,
};
use std::time::Instant;

const GROUP_SIZE: i32 = 64;
const BF16: i32 = 12;

struct Shape {
    name: &'static str,
    out: i32,
    inp: i32,
    bits: i32,
}

// Layer shapes from the local checkpoints' config.json:
//   gemma-4-12b: hidden 3840, intermediate 15360 (MLP is 8-bit in the mixed
//     checkpoint), q 16*256=4096, kv 8*256=2048, attention 4-bit
//   llama-3.1-8b: hidden 4096, intermediate 14336, q 4096, kv 1024, all 4-bit
//   gemma-4-31b: hidden 5376, intermediate 21504, all 4-bit
//   qwen3.5-9b: hidden 4096, intermediate 12288, all 4-bit
const SHAPES: &[Shape] = &[
    // gemma-4-12b MLP as shipped (8-bit) and the requantization counterfactual (4-bit)
    Shape {
        name: "g12b_mlp_gate_8b",
        out: 15360,
        inp: 3840,
        bits: 8,
    },
    Shape {
        name: "g12b_mlp_down_8b",
        out: 3840,
        inp: 15360,
        bits: 8,
    },
    Shape {
        name: "g12b_mlp_gate_4b",
        out: 15360,
        inp: 3840,
        bits: 4,
    },
    Shape {
        name: "g12b_mlp_down_4b",
        out: 3840,
        inp: 15360,
        bits: 4,
    },
    // gemma-4-12b attention (4-bit as shipped)
    Shape {
        name: "g12b_attn_q_4b",
        out: 4096,
        inp: 3840,
        bits: 4,
    },
    Shape {
        name: "g12b_attn_kv_4b",
        out: 2048,
        inp: 3840,
        bits: 4,
    },
    Shape {
        name: "g12b_attn_o_4b",
        out: 3840,
        inp: 4096,
        bits: 4,
    },
    // llama-3.1-8b (control that reaches ~75% of peak end-to-end)
    Shape {
        name: "l8b_mlp_gate_4b",
        out: 14336,
        inp: 4096,
        bits: 4,
    },
    Shape {
        name: "l8b_mlp_down_4b",
        out: 4096,
        inp: 14336,
        bits: 4,
    },
    Shape {
        name: "l8b_attn_q_4b",
        out: 4096,
        inp: 4096,
        bits: 4,
    },
    Shape {
        name: "l8b_attn_kv_4b",
        out: 1024,
        inp: 4096,
        bits: 4,
    },
    // gemma-4-31b and qwen3.5-9b MLP (uniform 4-bit checkpoints)
    Shape {
        name: "g31b_mlp_gate_4b",
        out: 21504,
        inp: 5376,
        bits: 4,
    },
    Shape {
        name: "g31b_mlp_down_4b",
        out: 5376,
        inp: 21504,
        bits: 4,
    },
    Shape {
        name: "q9b_mlp_gate_4b",
        out: 12288,
        inp: 4096,
        bits: 4,
    },
];

fn weight_bytes(out: i64, inp: i64, bits: i64) -> i64 {
    let packed = out * inp * bits / 8;
    let groups = out * (inp / GROUP_SIZE as i64);
    // bf16 scales + bf16 biases: 2 bytes each per group
    packed + groups * 4
}

fn make_bf16(shape: &[i32], seed: usize) -> UniquePtr<MlxArray> {
    let total: usize = shape.iter().map(|&d| d as usize).product();
    let data: Vec<f32> = (0..total)
        .map(|i| (((i + seed * 7919) as f32) * 0.000271).sin() * 0.05)
        .collect();
    let f32arr = from_slice_f32(&data, shape);
    astype(&f32arr, BF16)
}

struct QuantizedWeight {
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
}

fn quantize(shape: &Shape, seed: usize) -> QuantizedWeight {
    let wf = make_bf16(&[shape.out, shape.inp], seed);
    let w = quantize_weights_w(&wf, GROUP_SIZE, shape.bits);
    let scales = quantize_weights_scales(&wf, GROUP_SIZE, shape.bits);
    let biases = quantize_weights_biases(&wf, GROUP_SIZE, shape.bits);
    eval(&w);
    eval(&scales);
    eval(&biases);
    QuantizedWeight { w, scales, biases }
}

fn run_batch(x: &MlxArray, weights: &[QuantizedWeight], t: usize, bits: i32) -> f64 {
    let start = Instant::now();
    let mut acc: Option<UniquePtr<MlxArray>> = None;
    for i in 0..t {
        let qw = &weights[i % weights.len()];
        // SAFETY: all arrays outlive the call; biases pointer is valid.
        let y = unsafe {
            quantized_matmul(
                x,
                &qw.w,
                &qw.scales,
                qw.biases.as_ref().unwrap() as *const MlxArray,
                true,
                GROUP_SIZE,
                bits,
                "affine",
            )
        };
        acc = Some(match acc {
            None => y,
            Some(a) => add(&a, &y),
        });
    }
    let root = acc.unwrap();
    eval(&root);
    synchronize_default();
    start.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t_per_batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    println!(
        "=== qmm GEMV effective-bandwidth microbench === T={} ROUNDS={} group_size={}",
        t_per_batch, rounds, GROUP_SIZE
    );
    println!(
        "{:<20} {:>4} {:>7} {:>7} {:>9} {:>10} {:>9}",
        "shape", "bits", "out", "in", "MB/call", "us/call", "GB/s"
    );

    for shape in SHAPES {
        let bytes = weight_bytes(shape.out as i64, shape.inp as i64, shape.bits as i64);
        // Round-robin enough copies that the working set exceeds any L2/SLC.
        let copies = ((128 * 1024 * 1024) / bytes).clamp(2, 12) as usize;

        let weights: Vec<QuantizedWeight> = (0..copies).map(|s| quantize(shape, s + 1)).collect();
        let x = make_bf16(&[1, 1, shape.inp], 0);
        eval(&x);
        synchronize_default();

        // Warmup: one full batch (compiles kernels, primes the allocator).
        let _ = run_batch(&x, &weights, t_per_batch, shape.bits);

        let mut best = f64::MAX;
        for _ in 0..rounds {
            let dt = run_batch(&x, &weights, t_per_batch, shape.bits);
            if dt < best {
                best = dt;
            }
        }

        let us_per_call = best * 1e6 / t_per_batch as f64;
        let gbps = (bytes as f64 * t_per_batch as f64) / best / 1e9;
        println!(
            "{:<20} {:>4} {:>7} {:>7} {:>9.2} {:>10.1} {:>9.1}",
            shape.name,
            shape.bits,
            shape.out,
            shape.inp,
            bytes as f64 / 1e6,
            us_per_call,
            gbps
        );
    }

    println!();
    println!(
        "note: GB/s counts quantized weight + scales + biases bytes only (activations are negligible at B=1); {} weight copies round-robin per shape keep reads out of cache.",
        "2-12"
    );
}
