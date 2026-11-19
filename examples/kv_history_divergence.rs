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

//! Criterion 1 of #2778: compare logits *and* KV contents between a history
//! built by single-token steps and the same history built by width-W blocks.
//!
//! # What this isolates
//!
//! The issue reports that a temperature-0 run using a block verification path
//! can leave the plain-decode stream, and that a target-only replay with a
//! block-built KV history reproduces the flip. Two candidate causes exist:
//!
//!   (i)  the final forward is width-sensitive at equal history, and
//!   (ii) the history itself is written differently when it is appended a
//!        block at a time, so the two arms attend over different KV bytes.
//!
//! A logits-only comparison cannot separate them. This example holds the token
//! history identical, makes the *final* step one token wide in both arms, and
//! then reports both the KV bytes and the next-token logits together, so (ii)
//! is measured rather than inferred.
//!
//! # Arms
//!
//! Both arms absorb the same `context` tokens in one prefill forward. They
//! then append the same `n` history tokens: arm `inc` one token at a time,
//! arm `blk` in `width`-token blocks. Finally both take one identical
//! one-token step and report that step's logits. The final step is *not* the
//! thing under test, so its width is held fixed; only the stored history may
//! differ.
//!
//! With `width = 1` the two arms issue literally the same forward sequence and
//! must agree byte for byte. That is the negative control: if it fails, the
//! harness is broken and no other row means anything.
//!
//! # Why the existing harness cannot express this
//!
//! `examples/logit_trace.rs` rebuilds `make_caches()` inside its chunk loop,
//! so a model whose caches live in that slice starts every chunk empty and
//! never accumulates a history. The `MLXCEL_TRACE_CONTIGUOUS` lever that does
//! accumulate requires a Gemma 4 mode, and Gemma 4 owns its attention caches
//! internally, so the harness cannot read them back. This example therefore drives `forward` directly and keeps
//! one cache slice alive across every append.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features metal,accelerate \
//!   --example kv_history_divergence -- MODEL_DIR CORPUS.txt [WIDTH] [CONTEXT] [MAX_N]
//! ```
//!
//! `WIDTH` is the block width (default 8), `CONTEXT` the prefill length before
//! the first append (default 32), `MAX_N` the longest history to scan (default
//! 64, rounded down to a multiple of `WIDTH`).
//!
//! Every `WIDTH`-th history length from `WIDTH` to `MAX_N` is one traced
//! position, so a run traces `MAX_N / WIDTH` positions whether or not their KV
//! differs. The model must keep its KV in the slice `make_caches()` returns;
//! a model that owns its caches yields no readable KV and the run stops with
//! an error instead of reporting "no divergence".
//!
//! Output is TSV on stdout, prefixed with `#` metadata lines. Diagnostics go
//! to stderr.

use anyhow::{Context, Result};
use mlxcel::LanguageModel;
use mlxcel::distributed::kv_cache_serde::serialize::extract_kv_cache_entry;
use mlxcel_core::{array_shape, array_to_raw_bytes, astype, dtype, eval, from_slice_i32};

/// One layer's extracted K and V: bytes, shape, and MLX dtype code.
type LayerKv = ((Vec<u8>, Vec<i32>), (Vec<u8>, Vec<i32>), i32);

/// KV bytes for every layer, plus the element layout they are encoded in.
/// The layout is carried alongside rather than assumed: the same byte deltas
/// mean different things for fp16, bf16, or int8 storage.
struct Snapshot {
    layers: Vec<Option<LayerKv>>,
}
/// Absorb `context` tokens, append `n` tokens `width` at a time, then take one
/// one-token step. Returns the stored KV and that step's logits.
fn probe(
    model: &dyn LanguageModel,
    ids: &[i32],
    context: i32,
    width: i32,
    n: i32,
) -> Result<(Snapshot, Vec<f32>)> {
    let mut caches = model.make_caches();

    let ctx = from_slice_i32(&ids[..context as usize], &[1, context]);
    let out = model.forward(&ctx, &mut caches, None);
    eval(&out);

    let blocks = n / width;
    for b in 0..blocks {
        let from = (context + b * width) as usize;
        let to = from + width as usize;
        let block = from_slice_i32(&ids[from..to], &[1, width]);
        let out = model.forward(&block, &mut caches, None);
        eval(&out);
    }

    // Everything above may legitimately differ between arms. From here on the
    // two arms are identical work, so any difference is history-borne.
    let snap = snapshot(&caches)?;
    let at = (context + blocks * width) as usize;
    let step = from_slice_i32(&ids[at..at + 1], &[1, 1]);
    let out = model.forward(&step, &mut caches, None);
    eval(&out);
    Ok((snap, last_row_f32(&out)?))
}

/// Copy the filled portion of every layer's KV to Rust-owned bytes.
///
/// `extract_kv_cache_entry` already slices to `live_len()`, so this compares
/// exactly the bytes attention would read, not the pre-allocated padding.
/// It returns `(None, None)` for modes it cannot serialize (the packed turbo
/// modes), so those layers are `None` here.
fn snapshot(caches: &[mlxcel_core::cache::KVCache]) -> Result<Snapshot> {
    let layers = caches
        .iter()
        .map(|c| {
            let entry = extract_kv_cache_entry(c);
            match (entry.keys, entry.values) {
                (Some(k), Some(v)) => {
                    anyhow::ensure!(
                        k.dtype == v.dtype,
                        "K dtype {} differs from V dtype {}",
                        k.dtype,
                        v.dtype
                    );
                    let dt = k.dtype;
                    Ok(Some(((k.data, k.shape), (v.data, v.shape), dt)))
                }
                _ => Ok(None),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Snapshot { layers })
}

/// Name and element width of the KV dtypes this harness can decode. Any
/// other code is an error: reading an unknown layout as bytes would turn a
/// format mismatch into a fake divergence.
fn kv_layout(code: i32) -> Result<(&'static str, usize)> {
    match code {
        dtype::FLOAT16 => Ok(("float16", 2)),
        dtype::BFLOAT16 => Ok(("bfloat16", 2)),
        dtype::FLOAT32 => Ok(("float32", 4)),
        dtype::INT8 => Ok(("int8", 1)),
        other => anyhow::bail!("unsupported KV dtype code {other}"),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 3,
        "usage: kv_history_divergence MODEL_DIR CORPUS.txt [WIDTH] [CONTEXT] [MAX_N]"
    );
    let model_dir = &args[1];
    let corpus = &args[2];
    let width: i32 = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(8);
    let context: i32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(32);
    let max_n: i32 = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(64);

    anyhow::ensure!(width > 0, "WIDTH must be positive");
    anyhow::ensure!(context > 0, "CONTEXT must be positive");
    anyhow::ensure!(max_n >= width, "MAX_N must be at least WIDTH");
    let max_n = max_n / width * width;

    let runtime = mlxcel::initialize_runtime_checked().map_err(|e| anyhow::anyhow!("{e}"))?;
    let text = std::fs::read_to_string(corpus).with_context(|| format!("reading {corpus}"))?;
    let (model, tokenizer) = mlxcel::load_model(std::path::Path::new(model_dir))
        .with_context(|| format!("loading {model_dir}"))?;

    let ids: Vec<i32> = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .iter()
        .map(|&t| t as i32)
        .collect();

    anyhow::ensure!(
        ids.len() as i32 > context + max_n,
        "corpus has {} tokens, need {}",
        ids.len(),
        context + max_n + 1
    );

    println!("# model\t{model_dir}");
    println!("# device\t{}", runtime.device);
    println!("# width\t{width}\tcontext\t{context}\tmax_n\t{max_n}");
    println!(
        "# columns\tpos\tkv_bytes\tkv_diff_bytes\tkv_max_abs\tkv_max_rel\t\
         kv_peak\tkv_rms\tkv_rms_rel\tkv_perturbed_pct\tmax_abs_delta\t\
         d_top1\tmargin_inc\tmargin_blk\tmargin_consumed\targmax_inc\targmax_blk\tflip"
    );

    let mut flips = 0usize;
    let mut kv_diff_positions = 0usize;
    let mut control_ok = true;
    // Worst observed logit delta and the narrowest margin it had to cross.
    // Reported together so the "did it have room to flip?" question is
    // answered by measurement rather than by re-reading 17-column rows.
    let mut worst_delta = 0f32;
    let mut worst_delta_pos = 0i32;
    let mut narrowest_margin = f32::INFINITY;
    let mut margin_consumed_count = 0usize;
    let mut positions = 0usize;
    let mut seen_dtype: Option<i32> = None;

    for n in (width..=max_n).step_by(width as usize) {
        let (snap_inc, log_inc) = probe(&model, &ids, context, 1, n)?;
        let (snap_blk, log_blk) = probe(&model, &ids, context, width, n)?;

        // Negative control: re-running the same arm must reproduce itself, or
        // the comparison below is measuring harness noise.
        let (_, log_inc2) = probe(&model, &ids, context, 1, n)?;
        if log_inc != log_inc2 {
            control_ok = false;
        }
        anyhow::ensure!(
            snap_inc.layers.len() == snap_blk.layers.len(),
            "arms built {} and {} cache layers",
            snap_inc.layers.len(),
            snap_blk.layers.len()
        );
        anyhow::ensure!(
            snap_inc.layers.iter().any(Option::is_some),
            "no layer exposed readable KV: the model owns its caches or uses a \
             storage mode extract_kv_cache_entry cannot serialize"
        );

        let mut kv_bytes = 0usize;
        let mut kv_diff = 0usize;
        let mut kv_abs = 0f32;
        let mut kv_peak = 0f32;
        let mut kv_sumsq = 0f64;
        let mut kv_elems = 0usize;
        let mut kv_perturbed = 0usize;
        for (li, (a, b)) in snap_inc
            .layers
            .iter()
            .zip(snap_blk.layers.iter())
            .enumerate()
        {
            let ((ka, va, dt), (kb, vb, dt_b)) = match (a, b) {
                (Some((ka, va, dt)), Some((kb, vb, dt_b))) => ((ka, va, *dt), (kb, vb, *dt_b)),
                (None, None) => continue,
                _ => anyhow::bail!("layer {li}: KV readable in one arm only"),
            };
            anyhow::ensure!(dt == dt_b, "layer {li}: arms stored dtypes {dt} and {dt_b}");
            anyhow::ensure!(
                ka.1 == kb.1 && va.1 == vb.1,
                "layer {li}: arms stored different KV shapes ({:?}/{:?} vs {:?}/{:?})",
                ka.1,
                va.1,
                kb.1,
                vb.1
            );
            match seen_dtype {
                None => seen_dtype = Some(dt),
                Some(prev) => anyhow::ensure!(
                    prev == dt,
                    "layer {li}: KV dtype {dt} differs from earlier layers ({prev})"
                ),
            }
            let elem = kv_layout(dt)?;
            {
                let ((ka, _), (va, _), (kb, _), (vb, _)) = (ka, va, kb, vb);
                kv_bytes += ka.len() + va.len();
                kv_diff += count_diff(ka, kb) + count_diff(va, vb);
                for (x, y) in [(ka, kb), (va, vb)] {
                    let (d, p, rms, pert, tot) = max_abs_elem(x, y, elem);
                    kv_abs = kv_abs.max(d);
                    kv_peak = kv_peak.max(p);
                    kv_sumsq += (rms as f64) * (rms as f64) * tot as f64;
                    kv_elems += tot;
                    kv_perturbed += pert;
                }
            }
        }
        let kv_rel = if kv_peak > 0.0 { kv_abs / kv_peak } else { 0.0 };
        // The max alone is an extreme-value statistic: with millions of
        // elements a single unlucky lane dominates it. RMS and the perturbed
        // fraction describe the whole population, so they are what a claim
        // about "how big is the divergence" has to rest on.
        let kv_rms = if kv_elems > 0 {
            (kv_sumsq / kv_elems as f64).sqrt() as f32
        } else {
            0.0
        };
        let kv_rms_rel = if kv_peak > 0.0 { kv_rms / kv_peak } else { 0.0 };
        let kv_perturbed_pct = if kv_elems > 0 {
            100.0 * kv_perturbed as f64 / kv_elems as f64
        } else {
            0.0
        };
        if kv_diff > 0 {
            kv_diff_positions += 1;
        }

        let max_abs = log_inc
            .iter()
            .zip(log_blk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let (ai, ab) = (argmax(&log_inc), argmax(&log_blk));
        // A delta only becomes a flip when it exceeds the gap between the
        // top-1 and top-2 logits. Measuring the delta alone says how far the
        // histories drifted; the margin says whether drifting there could
        // ever change the emitted token. Both arms are reported so a flip
        // caused by a narrow plain-side margin is not misread as a wide one.
        let margin = |v: &[f32]| -> f32 {
            let mut best = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for &x in v {
                if x > best {
                    second = best;
                    best = x;
                } else if x > second {
                    second = x;
                }
            }
            best - second
        };
        let (m_inc, m_blk) = (margin(&log_inc), margin(&log_blk));
        // `max_abs` spans the whole vocabulary, so comparing it to the top-1
        // gap asks whether *any* logit moved further than the lead, which is
        // not what decides the emitted token: a token 20000 ranks down can
        // move by 3.0 without ever entering contention. The decisive
        // quantity is the perturbation at the incremental arm's winner, which
        // is the single logit whose value has to hold up for the plain stream
        // to be reproduced.
        let d_top1 = log_blk[ai] - log_inc[ai];
        let margin_gone = d_top1.abs() > m_inc;
        positions += 1;
        if max_abs > worst_delta {
            worst_delta = max_abs;
            worst_delta_pos = n;
        }
        narrowest_margin = narrowest_margin.min(m_inc.min(m_blk));
        if margin_gone {
            margin_consumed_count += 1;
        }
        let flip = ai != ab;
        if flip {
            flips += 1;
        }

        println!(
            "{n}\t{kv_bytes}\t{kv_diff}\t{kv_abs:.6}\t{kv_rel:.6}\t{kv_peak:.6}\t\
             {kv_rms:.8}\t{kv_rms_rel:.10}\t{kv_perturbed_pct:.4}\t\
             {max_abs:.6}\t{d_top1:.6}\t{m_inc:.6}\t{m_blk:.6}\t{margin_gone}\t\
             {ai}\t{ab}\t{flip}"
        );
    }

    println!("# arm_repeats_identical\t{control_ok}");
    match seen_dtype {
        Some(code) => {
            let (name, n) = kv_layout(code)?;
            println!("# kv_dtype\t{name}\telem_bytes\t{n}");
        }
        None => println!("# kv_dtype\tnone"),
    }
    println!("# positions_with_kv_diff\t{kv_diff_positions}");
    println!("# positions_with_flip\t{flips}");
    println!("# positions_traced\t{positions}");
    println!("# worst_logit_delta\t{worst_delta:.6}\tpos\t{worst_delta_pos}");
    println!(
        "# narrowest_margin\t{}\tmargin_consumed_positions\t{margin_consumed_count}",
        if narrowest_margin.is_finite() {
            format!("{narrowest_margin:.6}")
        } else {
            "n/a".to_string()
        }
    );
    println!(
        "# verdict\t{}",
        if !control_ok {
            "HARNESS UNRELIABLE: the same arm did not reproduce itself"
        } else if kv_diff_positions == 0 && flips == 0 {
            "no divergence: width-independent on this model/range"
        } else {
            "divergence measured"
        }
    );
    if !control_ok {
        anyhow::bail!("negative control failed: the same arm did not reproduce itself");
    }
    Ok(())
}

fn count_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// Largest absolute difference between two element runs, in float space for
/// float dtypes and in integer steps for int8, together with the peak
/// magnitude seen in either arm.
///
/// The peak is what makes the delta interpretable: an absolute error is only
/// meaningful against the scale of the values it perturbs, and that scale is a
/// property of the model's activations, not of the harness.
///
/// A raw bit-pattern delta is not a magnitude for a float format: fp16 stores
/// exponent and mantissa together, so one increment of the exponent field looks
/// like a huge integer jump while representing a tiny value change. Decode
/// first so this column means what it says.
fn max_abs_elem(
    a: &[u8],
    b: &[u8],
    (name, _): (&'static str, usize),
) -> (f32, f32, f32, usize, usize) {
    let mut worst = 0f32;
    let mut peak = 0f32;
    let mut sumsq = 0f64;
    let mut perturbed = 0usize;
    let mut total = 0usize;
    let mut take = |x: f32, y: f32| {
        peak = peak.max(x.abs()).max(y.abs());
        let d = (x - y).abs();
        total += 1;
        if d.is_finite() {
            if d > 0.0 {
                perturbed += 1;
            }
            sumsq += (d as f64) * (d as f64);
            if d > worst {
                worst = d;
            }
        }
    };
    match name {
        "float16" | "bfloat16" => {
            let decode = |c: &[u8]| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                if name == "bfloat16" {
                    f32::from_bits(u32::from(bits) << 16)
                } else {
                    half_to_f32(bits)
                }
            };
            for (ca, cb) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
                take(decode(ca), decode(cb));
            }
        }
        "float32" => {
            for (ca, cb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
                take(
                    f32::from_le_bytes([ca[0], ca[1], ca[2], ca[3]]),
                    f32::from_le_bytes([cb[0], cb[1], cb[2], cb[3]]),
                );
            }
        }
        _ => {
            // int8: signed integer steps, so -1 versus 0 is one step, not 255.
            for (&xa, &xb) in a.iter().zip(b.iter()) {
                take(f32::from(xa as i8), f32::from(xb as i8));
            }
        }
    }
    let rms = if total > 0 {
        (sumsq / total as f64).sqrt() as f32
    } else {
        0.0
    };
    (worst, peak, rms, perturbed, total)
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let man = (bits & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        0x1f => {
            if man == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

fn last_row_f32(out: &mlxcel_core::MlxArray) -> Result<Vec<f32>> {
    let shape = array_shape(out);
    let vocab = *shape.last().context("logits has no last dim")?;
    let rows = if shape.len() >= 2 {
        shape[shape.len() - 2]
    } else {
        1
    };
    let flat = astype(out, mlxcel_core::dtype::FLOAT32);
    eval(&flat);
    let bytes = array_to_raw_bytes(&flat);
    let all: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let start = ((rows - 1) * vocab) as usize;
    Ok(all[start..start + vocab as usize].to_vec())
}
