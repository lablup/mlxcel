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

//! One dense decode layer at the checkpoint's GQA shape. Run with
//! `--model DIR --contexts 4096,16384,32768 --csv rows.csv`.
//! Inputs are resident, reused (one warm-buffer rotation). Every sample builds
//! a fresh graph, evaluates its outputs and waits for the default stream.
//! The readback control copies already-evaluated f32 coordinates and norms:
//! it measures the transfer alone, excluding rotation and CPU index lookup.

use std::fmt::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use mlxcel_core as mx;
use mx::cache::turbo::{DELEGATED_FOLD_BLOCK, TurboQuantParams, quant, sparse_v};
use mx::{MlxArray, UniquePtr, dtype};
use serde::Deserialize;

#[derive(Parser)]
struct Options {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "4096,16384,32768")]
    contexts: Vec<i32>,
    #[arg(long, value_delimiter = ',', default_value = "1,4096,16384")]
    quant_tokens: Vec<i32>,
    #[arg(long)]
    csv: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Shape {
    num_key_value_heads: i32,
    num_attention_heads: i32,
    hidden_size: i32,
    head_dim: Option<i32>,
    num_hidden_layers: usize,
}

type Array = UniquePtr<MlxArray>;

fn input(shape: &[i32], seed: u32, dt: i32) -> Array {
    let size = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d.try_into().ok()?))
        .expect("invalid input shape");
    let mut state = seed;
    let mut word = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let arr = if dt == dtype::UINT8 {
        let values: Vec<_> = (0..size).map(|_| word() as u8).collect();
        mx::from_bytes(&values, shape, dt)
    } else {
        let values: Vec<_> = (0..size)
            .map(|_| (word() >> 8) as f32 / 8388608.0 - 1.0)
            .collect();
        mx::astype(&mx::from_slice_f32(&values, shape), dt)
    };
    mx::eval(&arr);
    arr
}

// Real norm/rescale sidecars have one stored value per KV head and token.
fn sidecar(h: i32, t: i32, seed: u32, dt: i32) -> Array {
    mx::add(
        &input(&[1, h, t, 1], seed, dt),
        &mx::full_f32(&[1], 4.0, dt),
    )
}

fn measure(
    csv: &mut String,
    mode: &str,
    stage: &str,
    tokens: i32,
    mut body: impl FnMut() -> Vec<Array>,
) {
    let mut times = Vec::with_capacity(20);
    for i in 0..25 {
        mx::synchronize_default();
        let start = Instant::now();
        let outputs = body();
        for output in &outputs {
            mx::eval(output);
        }
        mx::synchronize_default();
        let us = start.elapsed().as_secs_f64() * 1e6;
        if i >= 5 {
            times.push(us);
        }
    }
    times.sort_by(f64::total_cmp);
    let median = (times[9] + times[10]) / 2.0;
    let line = format!(
        "{mode},{stage},{tokens},20,{median:.3},{:.3},{:.3}",
        times[0], times[19]
    );
    println!("{line}");
    writeln!(csv, "{line}").unwrap();
}

fn dequant_v(v: &MlxArray, scale: &MlxArray, params: &TurboQuantParams) -> Array {
    if sparse_v::kernel_enabled() {
        let cb = mx::from_slice_f32(&params.codebook.centroids, &[16]);
        mx::turbo4_delegated_bulk_dequant_rotated(v, scale, &cb, params.head_dim as i32).unwrap()
    } else {
        quant::dequantize_v_turbo4_rotated(v, scale, params)
    }
}

fn layer(csv: &mut String, shape: &Shape, params: &TurboQuantParams, t: i32) {
    let d = params.head_dim as i32;
    let h = shape.num_key_value_heads;
    let q = input(&[1, shape.num_attention_heads, 1, d], 13, dtype::FLOAT16);
    let k = input(&[1, h, t, d], 31, dtype::FLOAT16);
    let dense_v = input(&[1, h, t, d], 41, dtype::FLOAT16);
    let scale = 1.0 / (d as f32).sqrt();
    let attention = |q: &MlxArray, k: &MlxArray, v: &MlxArray| {
        mx::layers::attention(q, k, v, scale, None, 0.0, 0)
    };
    measure(csv, "fp16", "attention", t, || {
        vec![attention(&q, &k, &dense_v)]
    });
    for mode in ["turbo4-asym", "turbo4", "turbo4-delegated"] {
        // A point just after a 128-token fold has a 128-token hot tail.
        let hot = if mode == "turbo4-delegated" {
            t.min(DELEGATED_FOLD_BLOCK)
        } else {
            0
        };
        let cold = t - hot;
        let vp = input(&[1, h, cold, d / 2], 18, dtype::UINT8);
        let vr = sidecar(h, cold, 67, dtype::FLOAT16);
        let kp = input(&[1, h, t, d / 2], 23, dtype::UINT8);
        let kn = sidecar(h, t, 71, dtype::FLOAT16);
        let hv = input(&[1, h, hot, d], 29, dtype::FLOAT16);
        mx::eval(&vr);
        mx::eval(&kn);
        let keys = || quant::dequantize_k_turbo4_rotated(&kp, &kn, params);
        let values = || dequant_v(&vp, &vr, params);
        measure(csv, mode, "dequant", t, || {
            let mut out = Vec::new();
            if cold > 0 {
                out.push(values());
            }
            if mode == "turbo4" {
                out.push(keys());
            }
            out
        });
        let kr = (mode == "turbo4").then(keys);
        let qr = (mode == "turbo4")
            .then(|| mx::astype(&quant::turbo4_k_rotate(&q, params), dtype::FLOAT16));
        let v = if cold == 0 {
            mx::copy(&hv)
        } else if hot == 0 {
            values()
        } else {
            mx::concatenate(
                &values(),
                &mx::astype(&quant::turbo4_v_rotate(&hv, params), dtype::FLOAT16),
                2,
            )
        };
        let material_k = kr.as_deref().unwrap_or(&k);
        let material_q = qr.as_deref().unwrap_or(&q);
        mx::eval(material_k);
        mx::eval(material_q);
        mx::eval(&v);
        measure(csv, mode, "materialized_sdpa", t, || {
            vec![attention(material_q, material_k, &v)]
        });
        let full = || match mode {
            "turbo4" => {
                sparse_v::attention_turbo4_dequant_sdpa(&q, &kp, &kn, &vp, &vr, params, scale, None)
            }
            "turbo4-asym" => {
                sparse_v::attention_turbo4_asym_dequant_sdpa(&q, &k, &vp, &vr, params, scale, None)
            }
            _ => sparse_v::attention_turbo4_delegated_dequant_sdpa(
                &q,
                &k,
                Some(&vp),
                Some(&vr),
                Some(&hv),
                params,
                cold,
                hot,
                scale,
                None,
            )
            .unwrap(),
        };
        measure(csv, mode, "full", t, || vec![full()]);
        if mx::default_device_is_gpu() && mx::turbo_packed_attention_available() && cold > 0 {
            measure(csv, mode, "packed", t, || {
                let cb = mx::from_slice_f32(&params.codebook.centroids, &[16]);
                let sym = mode == "turbo4";
                let query =
                    sym.then(|| mx::astype(&quant::turbo4_k_rotate(&q, params), dtype::FLOAT16));
                let tail = (hot > 0)
                    .then(|| mx::astype(&quant::turbo4_v_rotate(&hv, params), dtype::FLOAT16));
                let rot = mx::turbo_packed_attention(
                    query.as_deref().unwrap_or(&q),
                    if sym { &kp } else { &k },
                    &kn,
                    &vp,
                    &vr,
                    tail.as_deref().unwrap_or(&q),
                    &cb,
                    scale,
                    sym,
                )
                .unwrap();
                vec![quant::turbo4_v_inverse_rotate(&rot, params)]
            });
        }
    }
}

fn main() {
    let options = Options::parse();
    let shape: Shape =
        serde_json::from_slice(&std::fs::read(options.model.join("config.json")).unwrap()).unwrap();
    let h = shape.num_key_value_heads;
    assert!(h > 0 && shape.num_attention_heads > 0 && shape.num_attention_heads % h == 0);
    let d = shape
        .head_dim
        .unwrap_or(shape.hidden_size / shape.num_attention_heads);
    assert!(d >= 2 && (d as u32).is_power_of_two());
    assert!(
        options
            .contexts
            .iter()
            .chain(&options.quant_tokens)
            .all(|&t| t > 0)
    );
    mx::memory::set_cache_limit(4 << 30);
    let params = TurboQuantParams::new(d as u32, 42);
    println!(
        "layers={} q_heads={} kv_heads={h} dim={d}",
        shape.num_hidden_layers, shape.num_attention_heads
    );
    let mut csv = String::from("mode,stage,tokens,iters,median_us,min_us,max_us\n");
    for t in options.contexts {
        layer(&mut csv, &shape, &params, t);
        mx::clear_memory_cache();
    }
    for t in options.quant_tokens {
        let v = input(&[1, h, t, d], 79, dtype::FLOAT16);
        let coords = mx::astype(&v, dtype::FLOAT32);
        let norms = sidecar(h, t, 73, dtype::FLOAT32);
        mx::eval(&coords);
        mx::eval(&norms);
        measure(&mut csv, "quantize", "readback_control", t, || {
            std::hint::black_box(mx::array_to_raw_bytes(&coords));
            std::hint::black_box(mx::array_to_raw_bytes(&norms));
            vec![]
        });
        measure(&mut csv, "quantize", "full", t, || {
            let (packed, norms, rescale) = quant::quantize_v_turbo4(&v, &params);
            vec![packed, norms, rescale]
        });
        mx::clear_memory_cache();
    }
    if let Some(path) = options.csv {
        std::fs::write(path, csv).unwrap();
    }
}
