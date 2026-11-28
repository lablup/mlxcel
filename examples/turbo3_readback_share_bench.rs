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

//! Turbo3Asym host-readback share bench (issue #3274).
//!
//! `--kv-cache-mode turbo3-asym` keeps K in fp16 and V as 3-bit PolarQuant
//! indices. Before #3274 both halves of that codec left the device: the
//! quantizer read the rotated coordinates back for a host centroid lookup, and
//! the dequantizer read the whole visible packed window back, unpacked it on
//! one CPU thread and uploaded the indices again, once per layer per decode
//! step. This bench times those host segments against the device work around
//! them, at one model's real V layer shape.
//!
//! It uses only API that exists on both sides of #3274, so the same source
//! builds on the tree before the change and on the tree after it. The host
//! segment is timed as a stand-alone reproduction built from the public host
//! helpers (`array_to_raw_bytes`, `unpack_3bit_per_token`, `from_bytes`, and
//! for the quantizer `nearest_centroid_indices_with_boundaries` and
//! `pack_3bit_per_token`), so its cost can still be read after the codec
//! stopped paying it. The full `dequantize_v_turbo3` and `quantize_v_turbo3`
//! calls are timed as they are in the tree the bench was built from: before
//! #3274 they contain the host segment, after it they do not.
//!
//! Per visible length the dequantize rows are:
//!
//! - `dequant_host_segment`: readback of the packed window, host unpack, and
//!   upload of the dense index tensor (with the three parts as separate rows);
//! - `dequant_device_rest`: centroid gather, norm correction, inverse rotation
//!   and rescale, from an index tensor already on the device;
//! - `dequant_full_call`: `dequantize_v_turbo3` as built;
//! - `attention_fp16`: `layers::attention` for one decode token over the
//!   materialized fp16 V (the fp16 floor of the layer);
//! - `layer_call`: `dequantize_v_turbo3` and that attention in one graph, the
//!   per-layer cost of a Turbo3Asym decode step.
//!
//! The quantize rows, per new-token count, are `quant_host_segment` (readback
//! of the rotated coordinates, host lookup, host pack, upload),
//! `quant_device_rest` (norm and rotation) and `quant_full_call`.
//!
//! The graph is lazy, so every timed region ends with `eval` on its outputs
//! and a `synchronize_default()`, and starts after one. Each row is the median
//! of `--iters` iterations (at least 20) after `--warmup` untimed ones, with
//! the minimum and maximum next to it.
//!
//! Multiply a per-layer time by the number of Turbo3 layers to compare it with
//! a decode step from `mlxcel-bench-decode`: the boundary policy keeps the
//! first and last layers in fp16, and the bench prints the count for the
//! model.
//!
//! Run (CUDA):
//!   cargo run --release --features cuda --example turbo3_readback_share_bench -- \
//!     --model models/<llama-3.1-8b 4-bit checkpoint> --contexts 4096,16384,32768
//! Run (Apple):
//!   cargo run --release --features metal,accelerate \
//!     --example turbo3_readback_share_bench -- --model models/<checkpoint>
//!
//! `--csv PATH` also writes the rows as CSV. Without `--model` the shape comes
//! from `--kv-heads`, `--q-heads`, `--head-dim` and `--layers` (defaults: the
//! Llama 3.1 8B shape, 8 / 32 / 128 / 32).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::cache::turbo::codebook::nearest_centroid_indices_with_boundaries;
use mlxcel_core::cache::turbo::pack3::{
    pack_3bit_per_token, packed_bytes_per_token_3bit, unpack_3bit_per_token,
};
use mlxcel_core::cache::turbo::quant3::{
    TurboQuantParams3, dequantize_v_turbo3, quantize_v_turbo3,
};
use mlxcel_core::cache::turbo::{boundary_v_layers_from_env, resolve_layer_modes};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Seed of the codec parameters. Any value works: the timings do not depend on
/// the sign vectors.
const SEED: u32 = 42;

struct Shape {
    kv_heads: i32,
    q_heads: i32,
    head_dim: i32,
    layers: usize,
}

struct Options {
    shape: Shape,
    model: Option<PathBuf>,
    contexts: Vec<i32>,
    quant_tokens: Vec<i32>,
    iters: usize,
    warmup: usize,
    csv: Option<PathBuf>,
}

/// One timed stage: the median and the range over the timed iterations.
struct Row {
    section: &'static str,
    stage: &'static str,
    tokens: i32,
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

fn parse_list(value: &str, flag: &str) -> Vec<i32> {
    let list: Vec<i32> = value
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<i32>()
                .ok()
                .filter(|&n| n > 0)
                .unwrap_or_else(|| panic!("{flag}: `{item}` is not a positive integer"))
        })
        .collect();
    assert!(!list.is_empty(), "{flag}: empty list");
    list
}

/// Read the V layer shape from a checkpoint's `config.json`.
fn shape_from_model(model: &Path) -> Shape {
    let path = model.join("config.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
    let config: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("cannot parse {}: {err}", path.display()));
    let field = |name: &str| config.get(name).and_then(serde_json::Value::as_i64);
    let q_heads = field("num_attention_heads").expect("config.json: num_attention_heads");
    let kv_heads = field("num_key_value_heads").unwrap_or(q_heads);
    let head_dim = field("head_dim").unwrap_or_else(|| {
        field("hidden_size").expect("config.json: hidden_size or head_dim") / q_heads
    });
    let layers = field("num_hidden_layers").expect("config.json: num_hidden_layers");
    Shape {
        kv_heads: kv_heads as i32,
        q_heads: q_heads as i32,
        head_dim: head_dim as i32,
        layers: layers as usize,
    }
}

fn parse_args() -> Options {
    let mut options = Options {
        shape: Shape {
            kv_heads: 8,
            q_heads: 32,
            head_dim: 128,
            layers: 32,
        },
        model: None,
        contexts: vec![4096, 16384, 32768],
        quant_tokens: vec![1, 4096, 16384],
        iters: 20,
        warmup: 5,
        csv: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("{flag} needs a value"))
        };
        let number = |text: String, flag: &str| {
            text.parse::<usize>()
                .ok()
                .filter(|&n| n > 0)
                .unwrap_or_else(|| panic!("{flag}: `{text}` is not a positive integer"))
        };
        match flag.as_str() {
            "--model" | "-m" => options.model = Some(PathBuf::from(value())),
            "--contexts" => options.contexts = parse_list(&value(), "--contexts"),
            "--quant-tokens" => options.quant_tokens = parse_list(&value(), "--quant-tokens"),
            "--iters" => options.iters = number(value(), "--iters"),
            "--warmup" => options.warmup = number(value(), "--warmup"),
            "--kv-heads" => options.shape.kv_heads = number(value(), "--kv-heads") as i32,
            "--q-heads" => options.shape.q_heads = number(value(), "--q-heads") as i32,
            "--head-dim" => options.shape.head_dim = number(value(), "--head-dim") as i32,
            "--layers" => options.shape.layers = number(value(), "--layers"),
            "--csv" => options.csv = Some(PathBuf::from(value())),
            other => panic!(
                "unknown argument `{other}`; flags: --model DIR, --contexts LIST, \
                 --quant-tokens LIST, --iters N, --warmup N, --kv-heads N, --q-heads N, \
                 --head-dim N, --layers N, --csv PATH"
            ),
        }
    }
    if let Some(model) = options.model.as_ref() {
        options.shape = shape_from_model(model);
    }
    assert!(
        options.iters >= 20,
        "--iters must be at least 20 for a usable median; got {}",
        options.iters
    );
    options
}

/// Deterministic pseudo-random words (xorshift-multiply). Only the timing
/// matters, but distinct values keep the unpack and the softmax non-degenerate.
fn pseudo_words(seed: u64, n: usize) -> impl Iterator<Item = u64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n).map(move |_| {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    })
}

fn pseudo_f16(seed: u64, shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let vals: Vec<f32> = pseudo_words(seed, n)
        .map(|w| (w >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        .collect();
    let arr = mlxcel_core::astype(&mlxcel_core::from_slice_f32(&vals, shape), dtype::FLOAT16);
    mlxcel_core::eval(&arr);
    arr
}

fn pseudo_u8(seed: u64, shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let bytes: Vec<u8> = pseudo_words(seed, n).map(|w| (w >> 40) as u8).collect();
    let arr = mlxcel_core::from_bytes(&bytes, shape, dtype::UINT8);
    mlxcel_core::eval(&arr);
    arr
}

/// Time `body` over `iters` iterations after `warmup` untimed ones. Every
/// iteration is bracketed: the clock starts after a `synchronize_default()`
/// and stops after the body's outputs are evaluated and the device is
/// synchronized again. `body` returns the arrays to evaluate (possibly none,
/// for a region that is host work only).
fn time_stage<F>(
    options: &Options,
    section: &'static str,
    stage: &'static str,
    tokens: i32,
    mut body: F,
) -> Row
where
    F: FnMut() -> Vec<UniquePtr<MlxArray>>,
{
    let mut samples_us = Vec::with_capacity(options.iters);
    for iteration in 0..options.warmup + options.iters {
        mlxcel_core::synchronize_default();
        let start = Instant::now();
        let outputs = body();
        for out in &outputs {
            mlxcel_core::eval(out);
        }
        mlxcel_core::synchronize_default();
        let elapsed = start.elapsed();
        if iteration >= options.warmup {
            samples_us.push(elapsed.as_secs_f64() * 1e6);
        }
    }
    samples_us.sort_by(f64::total_cmp);
    let n = samples_us.len();
    let median_us = if n % 2 == 1 {
        samples_us[n / 2]
    } else {
        (samples_us[n / 2 - 1] + samples_us[n / 2]) * 0.5
    };
    Row {
        section,
        stage,
        tokens,
        median_us,
        min_us: samples_us[0],
        max_us: samples_us[n - 1],
    }
}

/// The device half of `dequantize_v_turbo3` from an index tensor that is
/// already on the device: centroid gather, norm correction, inverse rotation,
/// rescale. The same ops in the same order as the codec.
fn dequant_device_rest(
    indices: &MlxArray,
    v_norms: &MlxArray,
    params: &TurboQuantParams3,
) -> UniquePtr<MlxArray> {
    use mlxcel_core::{
        astype, divide, from_slice_f32, full_f32, maximum, multiply, sqrt, sum_axis, take, wht,
    };
    let d = params.head_dim as i32;
    let centroids = from_slice_f32(
        &params.codebook.centroids,
        &[params.codebook.centroids.len() as i32],
    );
    let y_hat = take(&centroids, indices, 0);
    let y_norm = sqrt(&sum_axis(&multiply(&y_hat, &y_hat), -1, true));
    let eps = full_f32(&[1], 1e-10, dtype::FLOAT32);
    let y_unit = divide(&y_hat, &maximum(&y_norm, &eps));
    let signs1 = from_slice_f32(&params.signs1, &[1, 1, 1, d]);
    let signs2 = from_slice_f32(&params.signs2, &[1, 1, 1, d]);
    let v_unit = multiply(&wht(&multiply(&y_unit, &signs2)), &signs1);
    let v_full = multiply(&v_unit, &astype(v_norms, dtype::FLOAT32));
    astype(&v_full, dtype::FLOAT16)
}

/// The device half of `quantize_v_turbo3` up to the rotated coordinates:
/// `(v_rot, norms)`, the same ops in the same order as the codec.
fn quant_device_rest(
    v: &MlxArray,
    params: &TurboQuantParams3,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    use mlxcel_core::{
        astype, divide, from_slice_f32, full_f32, greater, multiply, sqrt, sum_axis, where_cond,
        wht,
    };
    let d = params.head_dim as i32;
    let v_f32 = astype(v, dtype::FLOAT32);
    let norm = sqrt(&sum_axis(&multiply(&v_f32, &v_f32), -1, true));
    let zero = full_f32(&[1], 0.0, dtype::FLOAT32);
    let one = full_f32(&[1], 1.0, dtype::FLOAT32);
    let safe = where_cond(&greater(&norm, &zero), &norm, &one);
    let unit = divide(&v_f32, &safe);
    let signs1 = from_slice_f32(&params.signs1, &[1, 1, 1, d]);
    let signs2 = from_slice_f32(&params.signs2, &[1, 1, 1, d]);
    let v_rot = multiply(&wht(&multiply(&unit, &signs1)), &signs2);
    (v_rot, astype(&norm, dtype::FLOAT16))
}

/// Dequantize rows at one visible length.
fn bench_dequant(options: &Options, params: &TurboQuantParams3, visible: i32, rows: &mut Vec<Row>) {
    let shape = &options.shape;
    let (h, d) = (shape.kv_heads, shape.head_dim);
    let bytes_per_token = packed_bytes_per_token_3bit(d);
    let total_tokens = (h * visible) as usize;
    let seed = visible as u64;

    // Stored state of one layer at this visible length: any bytes are a valid
    // packed window, so pseudo-random bytes stand in for it.
    let v_packed = pseudo_u8(seed, &[1, h, visible, bytes_per_token]);
    let v_norms = pseudo_f16(seed + 1, &[1, h, visible, 1]);
    let keys = pseudo_f16(seed + 2, &[1, h, visible, d]);
    let query = pseudo_f16(seed + 3, &[1, shape.q_heads, 1, d]);
    let scale = 1.0 / (d as f32).sqrt();

    let section = "dequantize";
    let none = Vec::<UniquePtr<MlxArray>>::new;

    // Host segment, whole and in its three parts.
    rows.push(time_stage(
        options,
        section,
        "dequant_host_segment",
        visible,
        || {
            mlxcel_core::eval(&v_packed);
            let raw = mlxcel_core::array_to_raw_bytes(&v_packed);
            let indices = unpack_3bit_per_token(&raw, d, total_tokens);
            vec![mlxcel_core::from_bytes(
                &indices,
                &[1, h, visible, d],
                dtype::UINT8,
            )]
        },
    ));
    rows.push(time_stage(
        options,
        section,
        "dequant_host_readback",
        visible,
        || {
            mlxcel_core::eval(&v_packed);
            std::hint::black_box(mlxcel_core::array_to_raw_bytes(&v_packed));
            none()
        },
    ));
    let raw = mlxcel_core::array_to_raw_bytes(&v_packed);
    rows.push(time_stage(
        options,
        section,
        "dequant_host_unpack",
        visible,
        || {
            std::hint::black_box(unpack_3bit_per_token(&raw, d, total_tokens));
            none()
        },
    ));
    let indices_host = unpack_3bit_per_token(&raw, d, total_tokens);
    rows.push(time_stage(
        options,
        section,
        "dequant_host_upload",
        visible,
        || {
            vec![mlxcel_core::from_bytes(
                &indices_host,
                &[1, h, visible, d],
                dtype::UINT8,
            )]
        },
    ));

    // Device remainder from an index tensor that is already on the device.
    let indices = mlxcel_core::from_bytes(&indices_host, &[1, h, visible, d], dtype::UINT8);
    mlxcel_core::eval(&indices);
    rows.push(time_stage(
        options,
        section,
        "dequant_device_rest",
        visible,
        || vec![dequant_device_rest(&indices, &v_norms, params)],
    ));

    // The codec call as built.
    rows.push(time_stage(
        options,
        section,
        "dequant_full_call",
        visible,
        || vec![dequantize_v_turbo3(&v_packed, &v_norms, params)],
    ));

    // Attention over the materialized fp16 V: the fp16 floor of the layer.
    let v_fp16 = dequantize_v_turbo3(&v_packed, &v_norms, params);
    mlxcel_core::eval(&v_fp16);
    rows.push(time_stage(
        options,
        section,
        "attention_fp16",
        visible,
        || {
            vec![mlxcel_core::layers::attention(
                &query, &keys, &v_fp16, scale, None, 0.0, 0,
            )]
        },
    ));

    // Dequantize and attention in one graph: one layer of a decode step.
    rows.push(time_stage(options, section, "layer_call", visible, || {
        let v = dequantize_v_turbo3(&v_packed, &v_norms, params);
        vec![mlxcel_core::layers::attention(
            &query, &keys, &v, scale, None, 0.0, 0,
        )]
    }));
}

/// Quantize rows at one new-token count.
fn bench_quant(
    options: &Options,
    params: &TurboQuantParams3,
    new_tokens: i32,
    rows: &mut Vec<Row>,
) {
    let shape = &options.shape;
    let (h, d) = (shape.kv_heads, shape.head_dim);
    let total_tokens = (h * new_tokens) as usize;
    let bytes_per_token = packed_bytes_per_token_3bit(d);
    let v = pseudo_f16(0x5EED + new_tokens as u64, &[1, h, new_tokens, d]);
    let section = "quantize";

    // Host segment from rotated coordinates that are already evaluated:
    // readback, host centroid lookup, host pack, upload.
    let (v_rot, _) = quant_device_rest(&v, params);
    mlxcel_core::eval(&v_rot);
    let n_centroids = params.codebook.centroids.len();
    rows.push(time_stage(
        options,
        section,
        "quant_host_segment",
        new_tokens,
        || {
            mlxcel_core::eval(&v_rot);
            let raw = mlxcel_core::array_to_raw_bytes(&v_rot);
            let coords: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let indices: Vec<u8> = nearest_centroid_indices_with_boundaries(
                &coords,
                &params.codebook.boundaries,
                n_centroids,
            )
            .into_iter()
            .map(|i| (i & 0x07) as u8)
            .collect();
            let packed = pack_3bit_per_token(&indices, d, total_tokens);
            vec![mlxcel_core::from_bytes(
                &packed,
                &[1, h, new_tokens, bytes_per_token],
                dtype::UINT8,
            )]
        },
    ));

    // Device remainder: norm and rotation.
    rows.push(time_stage(
        options,
        section,
        "quant_device_rest",
        new_tokens,
        || {
            let (v_rot, norms) = quant_device_rest(&v, params);
            vec![v_rot, norms]
        },
    ));

    // The codec call as built.
    rows.push(time_stage(
        options,
        section,
        "quant_full_call",
        new_tokens,
        || {
            let (packed, norms) = quantize_v_turbo3(&v, params);
            vec![packed, norms]
        },
    ));
}

fn main() {
    let options = parse_args();
    let shape = &options.shape;
    let params = TurboQuantParams3::new(shape.head_dim as u32, SEED);
    let boundary = boundary_v_layers_from_env();
    let turbo_layers = resolve_layer_modes(KVCacheMode::Turbo3Asym, shape.layers, boundary)
        .iter()
        .filter(|&&mode| mode == KVCacheMode::Turbo3Asym)
        .count();
    let backend = if mlxcel_core::cuda_is_available() {
        "cuda"
    } else if mlxcel_core::metal_is_available() {
        "metal"
    } else {
        "other"
    };

    println!("=== mlxcel Turbo3Asym host-readback share bench (#3274) ===");
    if let Some(model) = options.model.as_ref() {
        println!("model={}", model.display());
    }
    println!(
        "backend={backend} kv_heads={} q_heads={} head_dim={} layers={} turbo3_layers={turbo_layers} \
         (boundary_v={boundary})",
        shape.kv_heads, shape.q_heads, shape.head_dim, shape.layers
    );
    println!(
        "warmup={} iters={} (median, min..max per row; every region eval-bracketed)",
        options.warmup, options.iters
    );
    println!();

    let mut rows = Vec::new();
    for &visible in &options.contexts {
        bench_dequant(&options, &params, visible, &mut rows);
        mlxcel_core::clear_memory_cache();
    }
    for &new_tokens in &options.quant_tokens {
        bench_quant(&options, &params, new_tokens, &mut rows);
        mlxcel_core::clear_memory_cache();
    }

    let mut csv = String::from(
        "section,stage,tokens,kv_heads,q_heads,head_dim,turbo3_layers,warmup,iters,\
         median_us,min_us,max_us\n",
    );
    let mut last = (rows[0].section, rows[0].tokens);
    for row in &rows {
        if (row.section, row.tokens) != last {
            println!();
            last = (row.section, row.tokens);
        }
        println!(
            "  {:<10} tokens={:>6} {:<22} median={:>11.1}us  min={:>11.1}us  max={:>11.1}us",
            row.section, row.tokens, row.stage, row.median_us, row.min_us, row.max_us
        );
        writeln!(
            csv,
            "{},{},{},{},{},{},{turbo_layers},{},{},{:.1},{:.1},{:.1}",
            row.section,
            row.stage,
            row.tokens,
            shape.kv_heads,
            shape.q_heads,
            shape.head_dim,
            options.warmup,
            options.iters,
            row.median_us,
            row.min_us,
            row.max_us
        )
        .expect("write to a String");
    }

    // Shares per visible length: the host segment against one layer's call.
    println!();
    let stage_us = |stage: &str, tokens: i32| {
        rows.iter()
            .find(|row| row.stage == stage && row.tokens == tokens)
            .map(|row| row.median_us)
    };
    for &visible in &options.contexts {
        let (Some(host), Some(full), Some(layer), Some(attention)) = (
            stage_us("dequant_host_segment", visible),
            stage_us("dequant_full_call", visible),
            stage_us("layer_call", visible),
            stage_us("attention_fp16", visible),
        ) else {
            continue;
        };
        println!(
            "  visible={visible:>6}: host segment {:>9.1}us = {:>5.1}% of the dequantize call as \
             built, {:>5.1}% of the layer call as built; layer call x {turbo_layers} layers = \
             {:>8.2} ms; fp16 attention x {turbo_layers} = {:>7.2} ms",
            host,
            100.0 * host / full,
            100.0 * host / layer,
            layer * turbo_layers as f64 / 1e3,
            attention * turbo_layers as f64 / 1e3,
        );
    }
    for &new_tokens in &options.quant_tokens {
        let (Some(host), Some(full)) = (
            stage_us("quant_host_segment", new_tokens),
            stage_us("quant_full_call", new_tokens),
        ) else {
            continue;
        };
        println!(
            "  new_tokens={new_tokens:>6}: quantize host segment {:>9.1}us = {:>5.1}% of the \
             quantize call as built ({:>9.1}us)",
            host,
            100.0 * host / full,
            full
        );
    }

    if let Some(path) = options.csv.as_ref() {
        std::fs::write(path, csv)
            .unwrap_or_else(|err| panic!("cannot write {}: {err}", path.display()));
        println!();
        println!("wrote {}", path.display());
    }
}
