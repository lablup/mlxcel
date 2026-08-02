// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Decode-step benchmark for matrix-absorbed MLA (issue #907).
//!
//! ## What this measures, and what it does not
//!
//! One MLA attention block per step, on synthetic tensors with MLA geometry: the
//! per-step cache append, the attention over the whole cached window, and the
//! folds around it. It does **not** run a real DeepSeek checkpoint. MLA is a
//! DeepSeek-family feature and no MLA checkpoint fits the development host, so
//! quoting these numbers as end-to-end model throughput would be wrong. They are
//! the throughput of the layer that changed, which is where the whole effect
//! lives: nothing else in the model differs between the arms.
//!
//! ## Three arms
//!
//! * `decompressed` reproduces the pre-#907 path: `kv_b_proj` up-projects the
//!   new token, the per-head K and V go into the cache, and SDPA reads
//!   `num_heads * (qk_head_dim + v_head_dim)` per cached token.
//! * `absorbed` is Stage 1: the cache holds `(ckv, kpe)` and SDPA reads
//!   `kv_lora_rank + qk_rope_head_dim` per cached token, for one latent head.
//! * `split_kv` is Stage 2: the same latent cache, with the range cut into
//!   chunks whose partial states are merged by issue #898's
//!   `paged_attention_merge_states`. Its partial producer is composed from MLX
//!   ops, so it is expected to be *slower* than Stage 1 here; the arm exists to
//!   show the decomposition running end to end and to be the baseline a fused
//!   partial kernel is measured against.
//!
//! ## Proving which path an arm ran
//!
//! Every arm prints `paths=...` from `mlxcel_core::mla::stats`, taken and reset
//! around the measured region. Issue #899 shipped a fused decode path that never
//! activated and whose benchmark compared the fallback against itself; an arm
//! here whose counter for its own path is 0 is visibly not measuring what its
//! name says. Do not report a number from an arm whose `paths=` line disagrees
//! with its label.
//!
//! Run (Metal):
//!
//! ```text
//! cargo run --release --features metal,accelerate \
//!     --example mla_absorbed_decode_bench
//!
//! # The issue's grid.
//! cargo run --release --features metal,accelerate \
//!     --example mla_absorbed_decode_bench -- \
//!     --contexts 4096,16384,32768 --batches 1,4 --steps 64 --warmup 16
//!
//! # DeepSeek-V3 geometry (128 heads) instead of V2-Lite's 16.
//! cargo run --release --features metal,accelerate \
//!     --example mla_absorbed_decode_bench -- --heads 128 --layers 61
//! ```

use clap::Parser;
use mlxcel_core::cache::KVCache;
use mlxcel_core::mla::{
    MlaAbsorbedProjections, MlaGeometry, MlaLatentCache, MlaSplitPlan, absorbed_decode,
    absorbed_decode_split_kv, decompressed_bytes_per_token, latent_bytes_per_token, stats,
};
use mlxcel_core::{MlxArray, UniquePtr};

const F16: i32 = mlxcel_core::dtype::FLOAT16;

#[derive(Parser, Debug)]
#[command(
    name = "mla_absorbed_decode_bench",
    about = "Decode throughput and KV memory for absorbed MLA (issue #907)"
)]
struct Args {
    /// Cached context lengths to sweep.
    #[arg(long, value_delimiter = ',', default_value = "4096,16384")]
    contexts: Vec<i32>,
    /// Batch sizes to sweep.
    #[arg(long, value_delimiter = ',', default_value = "1,4")]
    batches: Vec<i32>,
    /// Measured decode steps per arm.
    #[arg(long, default_value_t = 32)]
    steps: usize,
    /// Unmeasured warmup steps per arm.
    #[arg(long, default_value_t = 8)]
    warmup: usize,
    /// Query heads. 16 is DeepSeek-V2-Lite, 128 is DeepSeek-V3.
    #[arg(long, default_value_t = 16)]
    heads: usize,
    /// kv_lora_rank.
    #[arg(long, default_value_t = 512)]
    rank: usize,
    /// qk_nope_head_dim.
    #[arg(long, default_value_t = 128)]
    nope: usize,
    /// qk_rope_head_dim.
    #[arg(long, default_value_t = 64)]
    rope: usize,
    /// v_head_dim.
    #[arg(long, default_value_t = 128)]
    v_dim: usize,
    /// Decoder layers, for the whole-model KV memory table.
    #[arg(long, default_value_t = 27)]
    layers: usize,
    /// KV memory budget in GiB for the max-servable-context table.
    #[arg(long, default_value_t = 16.0)]
    kv_budget_gib: f64,
    /// Skip the split-KV arm.
    #[arg(long)]
    no_split: bool,
}

/// xorshift64* in [-1, 1), so a run is reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn vec(&mut self, n: usize, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
                (unit * 2.0 - 1.0) * amplitude
            })
            .collect()
    }
    fn array(&mut self, shape: &[i32], amplitude: f32) -> UniquePtr<MlxArray> {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let a = mlxcel_core::from_slice_f32(&self.vec(n, amplitude), shape);
        mlxcel_core::astype(&a, F16)
    }
}

fn main() {
    let args = Args::parse();
    let geometry = MlaGeometry {
        num_heads: args.heads,
        kv_lora_rank: args.rank,
        qk_nope_head_dim: args.nope,
        qk_rope_head_dim: args.rope,
        v_head_dim: args.v_dim,
    };
    if let Err(e) = geometry.check() {
        eprintln!("{e}");
        std::process::exit(2);
    }
    let scale = (geometry.q_head_dim() as f32).powf(-0.5);

    println!("# MLA absorbed decode (issue #907)");
    println!(
        "geometry: heads={} kv_lora_rank={} qk_nope={} qk_rope={} v_head={} layers={}",
        geometry.num_heads,
        geometry.kv_lora_rank,
        geometry.qk_nope_head_dim,
        geometry.qk_rope_head_dim,
        geometry.v_head_dim,
        args.layers
    );
    print_kv_memory_table(&geometry, args.layers, args.kv_budget_gib);

    let mut rng = Rng::new(0x907_907);
    // One dense `kv_b_proj` shared by both arms, so the decompressed arm and the
    // fold describe the same model.
    let kv_b_rows = (geometry.num_heads * geometry.kv_b_rows_per_head()) as i32;
    let kv_b = rng.array(&[kv_b_rows, geometry.kv_lora_rank as i32], 0.05);
    let kv_b_t = mlxcel_core::transpose(&kv_b);
    let proj = MlaAbsorbedProjections::from_dense(&kv_b, geometry).expect("fold kv_b_proj");
    println!(
        "fold: {} dense elements per layer ({:.1} MiB in f16), {:.1} MiB over {} layers",
        proj.element_count(),
        (proj.element_count() * 2) as f64 / (1024.0 * 1024.0),
        (proj.element_count() * 2 * args.layers) as f64 / (1024.0 * 1024.0),
        args.layers
    );

    println!();
    println!(
        "{:<14} {:>7} {:>8} {:>11} {:>11} {:>10}  paths",
        "arm", "batch", "context", "ms/step", "steps/s", "KV MiB"
    );

    for &context in &args.contexts {
        for &batch in &args.batches {
            let _ = stats::take();
            run_arm(
                "decompressed",
                batch,
                context,
                &args,
                geometry,
                scale,
                &mut rng,
                Arm::Decompressed { kv_b_t: &kv_b_t },
            );
            run_arm(
                "absorbed",
                batch,
                context,
                &args,
                geometry,
                scale,
                &mut rng,
                Arm::Absorbed { proj: &proj },
            );
            if !args.no_split {
                run_arm(
                    "split_kv",
                    batch,
                    context,
                    &args,
                    geometry,
                    scale,
                    &mut rng,
                    Arm::SplitKv { proj: &proj },
                );
            }
        }
    }
}

enum Arm<'a> {
    /// `kv_b_t` is `kv_b_proj^T`, `[kv_lora_rank, H * (qk_nope + v)]`, so the
    /// arm's per-step up-projection is one matmul rather than a linear layer.
    Decompressed {
        kv_b_t: &'a MlxArray,
    },
    Absorbed {
        proj: &'a MlaAbsorbedProjections,
    },
    SplitKv {
        proj: &'a MlaAbsorbedProjections,
    },
}

impl Arm<'_> {
    fn expected_path(&self) -> &'static str {
        match self {
            Arm::Decompressed { .. } => "decompressed",
            Arm::Absorbed { .. } => "absorbed_composed",
            Arm::SplitKv { .. } => "absorbed_split_kv",
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    label: &str,
    batch: i32,
    context: i32,
    args: &Args,
    geometry: MlaGeometry,
    scale: f32,
    rng: &mut Rng,
    arm: Arm<'_>,
) {
    let heads = geometry.num_heads as i32;
    let rank = geometry.kv_lora_rank as i32;
    let rope = geometry.qk_rope_head_dim as i32;
    let nope = geometry.qk_nope_head_dim as i32;
    let v_dim = geometry.v_head_dim as i32;

    let mut cache = KVCache::new();
    // Prefill the cache to `context` in the layout this arm decodes against.
    match arm {
        Arm::Decompressed { .. } => {
            let keys = rng.array(&[batch, heads, context, nope + rope], 0.5);
            let values = rng.array(&[batch, heads, context, v_dim], 0.5);
            cache.update(keys, values);
        }
        Arm::Absorbed { .. } | Arm::SplitKv { .. } => {
            let ckv = rng.array(&[batch, 1, context, rank], 0.5);
            let kpe = rng.array(&[batch, 1, context, rope], 0.5);
            cache.update(ckv, kpe);
        }
    }
    cache.eval_state();
    let kv_mib = cache.nbytes() as f64 / (1024.0 * 1024.0);

    // Per-step inputs, generated once: the benchmark measures attention, not
    // random-number generation.
    let q_nope = rng.array(&[batch, heads, 1, nope], 0.5);
    let q_pe = rng.array(&[batch, heads, 1, rope], 0.5);
    let new_latent = rng.array(&[batch, 1, 1, rank], 0.5);
    let new_kpe = rng.array(&[batch, 1, 1, rope], 0.5);
    mlxcel_core::eval(&q_nope);
    mlxcel_core::eval(&q_pe);
    mlxcel_core::eval(&new_latent);
    mlxcel_core::eval(&new_kpe);

    let step = |cache: &mut KVCache| {
        match arm {
            Arm::Decompressed { kv_b_t } => {
                // What the pre-#907 path pays per token: up-project the new
                // latent into per-head K and V, then attend over the
                // decompressed window.
                let latent_2d = mlxcel_core::reshape(&new_latent, &[batch, 1, rank]);
                let kv = mlxcel_core::matmul(&latent_2d, kv_b_t);
                let kv = mlxcel_core::reshape(&kv, &[batch, 1, heads, nope + v_dim]);
                let kv = mlxcel_core::transpose_axes(&kv, &[0, 2, 1, 3]);
                let k_nope = mlxcel_core::utils::slice_axis(&kv, -1, 0, nope);
                let values = mlxcel_core::utils::slice_axis(&kv, -1, nope, -1);
                let k_pe = mlxcel_core::utils::repeat_kv(&new_kpe, heads);
                let keys = mlxcel_core::concatenate(&k_nope, &k_pe, -1);
                let queries = mlxcel_core::concatenate(&q_nope, &q_pe, -1);
                let (keys, values) = cache.update_and_fetch(keys, values);
                stats::record(mlxcel_core::mla::MlaDecodePath::Decompressed);
                mlxcel_core::causal_attention(&queries, &keys, &values, scale, 0.0, 0)
            }
            Arm::Absorbed { proj } => {
                let mut view = MlaLatentCache::wrap(cache, geometry).expect("fp16 non-paged cache");
                let (ckv, kpe) = view
                    .update_and_fetch(mlxcel_core::copy(&new_latent), mlxcel_core::copy(&new_kpe));
                absorbed_decode(&q_nope, &q_pe, &ckv, &kpe, proj, scale, None)
            }
            Arm::SplitKv { proj } => {
                let mut view = MlaLatentCache::wrap(cache, geometry).expect("fp16 non-paged cache");
                let (ckv, kpe) = view
                    .update_and_fetch(mlxcel_core::copy(&new_latent), mlxcel_core::copy(&new_kpe));
                let kv_len = view.seq_len();
                let plan = MlaSplitPlan::heuristic(
                    batch as usize,
                    geometry.num_heads,
                    kv_len,
                    mlxcel_core::paged_v2::device_target_ctas(),
                );
                absorbed_decode_split_kv(&q_nope, &q_pe, &ckv, &kpe, proj, scale, &plan)
                    .expect("split-kv decode")
            }
        }
    };

    for _ in 0..args.warmup {
        let out = step(&mut cache);
        mlxcel_core::eval(&out);
    }
    mlxcel_core::synchronize_default();
    let _ = stats::take();

    let start = std::time::Instant::now();
    for _ in 0..args.steps {
        let out = step(&mut cache);
        mlxcel_core::eval(&out);
    }
    mlxcel_core::synchronize_default();
    let elapsed = start.elapsed().as_secs_f64();
    let counts = stats::take();

    let ms = elapsed * 1000.0 / args.steps as f64;
    println!(
        "{:<14} {:>7} {:>8} {:>11.4} {:>11.1} {:>10.1}  {}",
        label,
        batch,
        context,
        ms,
        args.steps as f64 / elapsed,
        kv_mib,
        counts.summary()
    );
    let expected = arm.expected_path();
    if !counts
        .summary()
        .contains(&format!("{expected}={}", args.steps))
    {
        println!(
            "  WARNING: arm \"{label}\" expected {expected}={} but recorded {}. \
             Do not report this row.",
            args.steps,
            counts.summary()
        );
    }
}

/// The issue's KV-memory table: bytes per token per layer before and after, and
/// the context a fixed budget then buys.
fn print_kv_memory_table(geometry: &MlaGeometry, layers: usize, budget_gib: f64) {
    let before = decompressed_bytes_per_token(geometry, 2);
    let after = latent_bytes_per_token(geometry, 2);
    let budget = budget_gib * 1024.0 * 1024.0 * 1024.0;
    let ctx = |per_token_per_layer: usize| -> u64 {
        (budget / (per_token_per_layer * layers) as f64) as u64
    };
    println!(
        "kv bytes/token/layer: decompressed={before} latent={after} ({:.2}x reduction)",
        before as f64 / after.max(1) as f64
    );
    println!(
        "kv bytes/token (all {layers} layers): decompressed={} latent={}",
        before * layers,
        after * layers
    );
    println!(
        "max context under a {budget_gib:.0} GiB KV budget: decompressed={} tokens latent={} tokens",
        ctx(before),
        ctx(after)
    );
}
