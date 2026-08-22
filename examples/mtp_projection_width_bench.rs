// MTP verify LM-head projection cost as a function of projected width.
//
// Issue #1179 asks whether a device-side early-exit verify walk is worth
// building. Early exit cannot skip the transformer forward (the whole block
// is one forward); all it can skip is the tail of the hidden-to-logits
// projection plus its argmax. This bench measures the ceiling of that
// saving: the cost of projecting W positions through the LM head at the
// real head shapes, for W from 1 to the widest block anyone runs. If the
// curve is flat in W, the projection is weight-read-bound and early exit
// has nothing to win at any accept rate.
//
// Method mirrors `qmm_gemv_microbench.rs`: quantize a bf16 weight at the
// real (out, in, bits, group) shape into enough independent copies that the
// working set defeats caches, then time T back-to-back projection+argmax
// graphs folded into one root so one eval submits the batch. The argmax is
// part of the measured graph because the device walk would fuse it.
//
// Guard (issue #1179 "the two arms must be provably different"): each row
// re-derives the logits and argmax shapes from a check call and prints
// them; a row only counts if the logits shape is [1, W, out].
//
// Usage: cargo run --release --features metal,accelerate \
//          --example mtp_projection_width_bench [ T_PER_BATCH ] [ ROUNDS ]
// Defaults: T_PER_BATCH=16, ROUNDS=5 (best round reported).

use mlxcel_core::{
    MlxArray, UniquePtr, add, argmax_last_axis, astype, eval, from_slice_f32,
    quantize_weights_biases, quantize_weights_scales, quantize_weights_w, quantized_matmul,
    synchronize_default,
};
use std::time::Instant;

const GROUP_SIZE: i32 = 64;
const BF16: i32 = 12;

struct Head {
    name: &'static str,
    out: i32,
    inp: i32,
    bits: i32,
}

// The two MTP-target LM heads in production today, from the local
// checkpoints' config.json: Gemma 4 12B ties the 262144-token embedding at
// hidden 3840; Qwen 3.8 27B has an untied 248320 x 5120 head. Both affine
// 4-bit group 64 in the mlx-community checkpoints.
const HEADS: &[Head] = &[
    Head {
        name: "gemma4-12b tied head",
        out: 262144,
        inp: 3840,
        bits: 4,
    },
    Head {
        name: "qwen3.8-27b lm_head",
        out: 248320,
        inp: 5120,
        bits: 4,
    },
];

const WIDTHS: &[i32] = &[1, 2, 3, 4, 5, 8, 16, 32];

fn weight_bytes(out: i64, inp: i64, bits: i64) -> i64 {
    let packed = out * inp * bits / 8;
    let groups = out * (inp / GROUP_SIZE as i64);
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

fn quantize(head: &Head, seed: usize) -> QuantizedWeight {
    let wf = make_bf16(&[head.out, head.inp], seed);
    let w = quantize_weights_w(&wf, GROUP_SIZE, head.bits);
    let scales = quantize_weights_scales(&wf, GROUP_SIZE, head.bits);
    let biases = quantize_weights_biases(&wf, GROUP_SIZE, head.bits);
    eval(&w);
    eval(&scales);
    eval(&biases);
    QuantizedWeight { w, scales, biases }
}

/// One projection+argmax graph: `[1, W, in] -> [1, W, out] -> [1, W]`.
fn project_argmax(x: &MlxArray, qw: &QuantizedWeight, bits: i32) -> UniquePtr<MlxArray> {
    // SAFETY: all arrays outlive the call; the biases pointer is valid.
    let logits = unsafe {
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
    argmax_last_axis(&logits)
}

fn run_batch(x: &MlxArray, weights: &[QuantizedWeight], t: usize, bits: i32) -> f64 {
    let start = Instant::now();
    let mut acc: Option<UniquePtr<MlxArray>> = None;
    for i in 0..t {
        let ids = project_argmax(x, &weights[i % weights.len()], bits);
        acc = Some(match acc {
            None => ids,
            Some(a) => add(&a, &ids),
        });
    }
    let root = acc.unwrap();
    eval(&root);
    synchronize_default();
    start.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t_per_batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    println!(
        "=== MTP LM-head projection width bench (issue #1179) === T={} ROUNDS={} group_size={}",
        t_per_batch, rounds, GROUP_SIZE
    );

    for head in HEADS {
        let bytes = weight_bytes(head.out as i64, head.inp as i64, head.bits as i64);
        let copies = ((256 * 1024 * 1024) / bytes).clamp(2, 8) as usize;
        let weights: Vec<QuantizedWeight> = (0..copies).map(|s| quantize(head, s + 1)).collect();

        println!(
            "\n{} ({} x {}, {}-bit, {:.0} MB/read, {} copies)",
            head.name,
            head.out,
            head.inp,
            head.bits,
            bytes as f64 / 1e6,
            copies
        );
        println!(
            "{:>5} {:>22} {:>10} {:>9} {:>12}",
            "W", "logits shape (guard)", "ms/call", "GB/s", "vs W=1"
        );

        let mut per_w_ms: Vec<(i32, f64)> = Vec::new();
        for &w in WIDTHS {
            let x = make_bf16(&[1, w, head.inp], 0);
            eval(&x);
            synchronize_default();

            // Guard: derive the shapes this arm actually projects.
            let check = unsafe {
                quantized_matmul(
                    &x,
                    &weights[0].w,
                    &weights[0].scales,
                    weights[0].biases.as_ref().unwrap() as *const MlxArray,
                    true,
                    GROUP_SIZE,
                    head.bits,
                    "affine",
                )
            };
            let logits_shape = mlxcel_core::array_shape(&check);
            assert_eq!(
                logits_shape,
                vec![1, w, head.out],
                "guard: this arm is not projecting W={w} positions"
            );

            let _ = run_batch(&x, &weights, t_per_batch, head.bits); // warmup
            let mut best = f64::MAX;
            for _ in 0..rounds {
                best = best.min(run_batch(&x, &weights, t_per_batch, head.bits));
            }
            let ms_per_call = best * 1e3 / t_per_batch as f64;
            let gbs = (bytes as f64 * t_per_batch as f64) / best / 1e9;
            let vs_w1 = per_w_ms
                .first()
                .map(|&(_, base)| format!("{:+.1}%", (ms_per_call / base - 1.0) * 100.0))
                .unwrap_or_default();
            println!(
                "{:>5} {:>22} {:>10.3} {:>9.0} {:>12}",
                w,
                format!("{:?}", logits_shape),
                ms_per_call,
                gbs,
                vs_w1
            );
            per_w_ms.push((w, ms_per_call));
        }

        // The early-exit ceiling at block K with A positions kept is
        // t(K) - t(A); print the widest useful contrast per K.
        println!("early-exit ceiling (t(K) - t(1), the most a device walk could save):");
        for &(w, ms) in per_w_ms.iter().skip(1) {
            let base = per_w_ms[0].1;
            println!("  K={:<3} ceiling = {:.3} ms/round", w, ms - base);
        }
    }
}
