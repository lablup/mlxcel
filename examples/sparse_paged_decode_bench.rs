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

//! Block-sparse decode: additive mask versus fused page indirection (issue #904).
//!
//! ## Why this is synthetic
//!
//! The two families issue #904 targets are DeepSeek-V3.2 (~685B) and
//! MiniMax-M3 (427B). Neither checkpoint fits on the development machine, and
//! `minimax_m3_indexer.rs` has said so since the family landed. A
//! checkpoint-driven benchmark is therefore not available, and inventing one
//! would measure nothing. What *is* measurable without weights is the thing the
//! change actually replaces: one decoder layer's attention, at the real head
//! geometry and the real sparse configuration, with the same selection driving
//! both arms.
//!
//! So this harness times a single attention step, not a model. Read the numbers
//! as "what happens to the attention op", and multiply by the sparse layer count
//! to reason about a whole forward pass. It cannot tell you end-to-end tokens
//! per second, and it does not claim to.
//!
//! ## The two arms
//!
//! | arm | what it runs |
//! |---|---|
//! | `mask` | dense SDPA over the whole live window with the block-drop mask added to it, which is what `minimax_m3_layers.rs` does today |
//! | `sparse` | [`mlxcel_core::paged_v2::run_sparse_decode`] over the same selection |
//!
//! Both arms consume the **same** selected block ids, so the comparison is not
//! confounded by a different selection, and the harness checks their outputs
//! agree before reporting any timing. A timing comparison between two arms that
//! disagree numerically is not a comparison.
//!
//! ## Proving which path each arm took
//!
//! Every arm prints its dispatch outcome on stdout, by name, before its timing.
//! The `sparse` arm additionally prints the [`SparseDecodeStats`] delta it
//! caused, so "the fused kernel ran N times" is in the output rather than
//! inferred. #899 shipped a fused decode path that silently never activated and
//! then benchmarked the fallback against itself; the resulting clean-looking
//! null was nearly accepted. An arm that cannot say what it ran is not evidence.
//!
//! ## Running it
//!
//! ```text
//! # The default sweep: MiniMax-M3 geometry at 8K / 16K / 32K.
//! cargo run --release --features metal,accelerate \
//!     --example sparse_paged_decode_bench
//!
//! # More repetitions, and a wider context.
//! cargo run --release --features metal,accelerate \
//!     --example sparse_paged_decode_bench -- --contexts 16384,32768,65536 --reps 5
//!
//! # Confirm the kill switch really restores the mask path.
//! MLXCEL_SPARSE_PAGED_ATTENTION=0 cargo run --release \
//!     --features metal,accelerate --example sparse_paged_decode_bench
//! ```
//!
//! Run it serialized on an otherwise quiet machine and repeat it; a decode
//! attention step at these sizes is well under a millisecond, so anything else
//! resident on the GPU dominates.

use std::time::Instant;

use clap::Parser;
use mlxcel_core::cache::sparse_csr::{ContiguousCacheLayout, selection_from_blocks};
use mlxcel_core::paged_v2::{
    SparseDecodeInputs, SparseDecodeOutcome, min_sparsity_ratio, report_sparse_outcome_once,
    run_sparse_decode, sparse_decode_stats, sparse_paged_enabled,
};
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Parser, Debug)]
#[command(about = "Block-sparse decode: additive mask versus fused page indirection (issue #904)")]
struct Args {
    /// Live context lengths to sweep.
    #[arg(long, value_delimiter = ',', default_value = "8192,16384,32768")]
    contexts: Vec<i32>,

    /// Query heads. MiniMax-M3 uses 64.
    #[arg(long, default_value_t = 64)]
    q_heads: i32,

    /// KV heads. MiniMax-M3 uses 4.
    #[arg(long, default_value_t = 4)]
    kv_heads: i32,

    /// Head dimension. MiniMax-M3 uses 128.
    #[arg(long, default_value_t = 128)]
    head_dim: i32,

    /// Tokens per key block (`sparse_block_size`).
    #[arg(long, default_value_t = 128)]
    block_size: i32,

    /// Selected blocks (`sparse_topk_blocks`).
    #[arg(long, default_value_t = 16)]
    topk_blocks: i32,

    /// Timed steps per repetition.
    #[arg(long, default_value_t = 200)]
    steps: usize,

    /// Repetitions; the reported figure is the median.
    #[arg(long, default_value_t = 3)]
    reps: usize,

    /// Skip the numerical agreement check between the arms.
    #[arg(long, default_value_t = false)]
    skip_parity: bool,
}

/// xorshift64* in [-1, 1).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn random_array(shape: &[i32], rng: &mut Rng, dtype: i32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    mlxcel_core::astype(&mlxcel_core::from_slice_f32(&data, shape), dtype)
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn max_rel_error(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |acc, v| acc.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs() / scale))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// One synthetic decode step: the K/V allocations, the query, and the selected
/// blocks that both arms share.
struct Step {
    /// `[1, kv_heads + 1, ctx, head_dim]` f16. One head wider than V, matching
    /// MiniMax-M3's index key riding on the K head axis.
    k_alloc: UniquePtr<MlxArray>,
    /// `[1, kv_heads, ctx, head_dim]` f16.
    v_alloc: UniquePtr<MlxArray>,
    /// `[1, q_heads, 1, head_dim]` f16.
    q: UniquePtr<MlxArray>,
    /// `[1, budget]` i32 selected whole-block ids.
    blocks: UniquePtr<MlxArray>,
    /// `[1, 1, 1, ctx]` additive keep/drop mask over the same selection.
    mask: UniquePtr<MlxArray>,
    ctx: i32,
    num_blocks: i32,
    tail_start: i32,
    budget: i32,
    selected: i32,
}

fn build_step(args: &Args, ctx: i32, seed: u64) -> Option<Step> {
    let num_blocks = (ctx + args.block_size - 1) / args.block_size;
    if args.topk_blocks >= num_blocks || args.topk_blocks < 2 {
        return None;
    }
    let budget = args.topk_blocks - 1;
    let tail_start = (num_blocks - 1) * args.block_size;

    let mut rng = Rng::new(seed);
    let f16 = mlxcel_core::dtype::FLOAT16;
    let k_alloc = random_array(&[1, args.kv_heads + 1, ctx, args.head_dim], &mut rng, f16);
    let v_alloc = random_array(&[1, args.kv_heads, ctx, args.head_dim], &mut rng, f16);
    let q = random_array(&[1, args.q_heads, 1, args.head_dim], &mut rng, f16);

    // A deterministic, scattered block choice standing in for the indexer's
    // top-k. Both arms consume it, so its content does not bias the comparison
    // as long as it is the same on both sides.
    let mut chosen: Vec<i32> = Vec::with_capacity(budget as usize);
    let mut j = 0i64;
    while (chosen.len() as i32) < budget {
        let candidate = ((j * 2_654_435_761) % i64::from(num_blocks - 1)) as i32;
        if !chosen.contains(&candidate) {
            chosen.push(candidate);
        }
        j += 1;
    }
    let blocks = mlxcel_core::from_slice_i32(&chosen, &[1, budget]);

    // The additive mask the `mask` arm uses: 0 on selected columns (the chosen
    // whole blocks plus the final block), -inf elsewhere. Built once on the
    // host so the timed region measures attention, not mask construction; the
    // real mask path also builds it on device per step, so the `mask` arm here
    // is if anything flattered.
    let mut keep = vec![f32::NEG_INFINITY; ctx as usize];
    for &b in &chosen {
        for t in b * args.block_size..(b + 1) * args.block_size {
            keep[t as usize] = 0.0;
        }
    }
    for t in tail_start..ctx {
        keep[t as usize] = 0.0;
    }
    let selected = keep.iter().filter(|v| **v == 0.0).count() as i32;
    let mask = mlxcel_core::from_slice_f32(&keep, &[1, 1, 1, ctx]);

    Some(Step {
        k_alloc,
        v_alloc,
        q,
        blocks,
        mask,
        ctx,
        num_blocks,
        tail_start,
        budget,
        selected,
    })
}

impl Step {
    /// The live K/V windows the mask arm attends over. At `ctx == capacity`
    /// these are the whole allocations; the slice is kept so the arm matches
    /// what the model does after `update_and_fetch`.
    fn windows(&self, kv_heads: i32, head_dim: i32) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let k = mlxcel_core::slice(
            &self.k_alloc,
            &[0, 0, 0, 0],
            &[1, kv_heads, self.ctx, head_dim],
        );
        let v = mlxcel_core::slice(
            &self.v_alloc,
            &[0, 0, 0, 0],
            &[1, kv_heads, self.ctx, head_dim],
        );
        (k, v)
    }

    fn selection(
        &self,
        kv_heads: i32,
        block_size: i32,
    ) -> Result<mlxcel_core::cache::SparseSelection, String> {
        let layout = ContiguousCacheLayout::from_shape(&mlxcel_core::array_shape(&self.k_alloc))?;
        selection_from_blocks(
            &layout,
            kv_heads,
            block_size,
            &self.blocks,
            self.tail_start,
            self.ctx,
        )
    }
}

/// Run the mask arm once.
fn run_mask(step: &Step, args: &Args) -> UniquePtr<MlxArray> {
    let (k, v) = step.windows(args.kv_heads, args.head_dim);
    let scale = (args.head_dim as f32).powf(-0.5);
    let mask_ptr = &*step.mask as *const MlxArray;
    unsafe { mlxcel_core::layers::attention_from_ptr(&step.q, &k, &v, scale, mask_ptr, 0.0, 0) }
}

/// Run the sparse arm once, returning the output and the dispatch outcome.
fn run_sparse(
    step: &Step,
    args: &Args,
    selection: &mlxcel_core::cache::SparseSelection,
) -> (Option<UniquePtr<MlxArray>>, SparseDecodeOutcome) {
    let inputs = SparseDecodeInputs {
        q: &step.q,
        k_alloc: &step.k_alloc,
        v_alloc: &step.v_alloc,
        kv_heads: args.kv_heads,
        live_len: step.ctx,
        scale: (args.head_dim as f32).powf(-0.5),
    };
    let (out, outcome) = run_sparse_decode(&inputs, selection);
    // The counters live behind the reporter, not behind the launch, so that a
    // caller which declines before ever reaching `run_sparse_decode` still
    // shows up in the totals. A harness that skips this reports zeroes.
    report_sparse_outcome_once(&outcome);
    (out, outcome)
}

/// Median milliseconds per step over `reps` repetitions of `steps` steps.
fn time_ms<F>(reps: usize, steps: usize, mut once: F) -> f64
where
    F: FnMut() -> Option<UniquePtr<MlxArray>>,
{
    // Warmup: JIT compilation of the kernel body happens on first use.
    for _ in 0..8 {
        if let Some(out) = once() {
            mlxcel_core::eval(&out);
        }
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        for _ in 0..steps {
            if let Some(out) = once() {
                mlxcel_core::eval(&out);
            }
        }
        samples.push(start.elapsed().as_secs_f64() * 1000.0 / steps as f64);
    }
    median(samples)
}

fn main() {
    let args = Args::parse();

    println!("# sparse paged decode bench (issue #904)");
    println!(
        "# geometry: q_heads {} / kv_heads {} / head_dim {}, block_size {}, topk_blocks {}",
        args.q_heads, args.kv_heads, args.head_dim, args.block_size, args.topk_blocks
    );
    println!(
        "# {} timed steps x {} repetitions, median reported",
        args.steps, args.reps
    );
    println!(
        "# MLXCEL_SPARSE_PAGED_ATTENTION: fused path {}",
        if sparse_paged_enabled() {
            "ENABLED"
        } else {
            "DISABLED (kill switch)"
        }
    );
    println!(
        "# MLXCEL_SPARSE_PAGED_MIN_SPARSITY: {}x required (set 0 to measure the declined regime)",
        min_sparsity_ratio()
    );
    println!();

    for (i, &ctx) in args.contexts.iter().enumerate() {
        let Some(step) = build_step(&args, ctx, 0x904_0000 + i as u64) else {
            println!(
                "ctx {ctx}: skipped, {} blocks is not a sparse regime for topk_blocks {}",
                (ctx + args.block_size - 1) / args.block_size,
                args.topk_blocks
            );
            continue;
        };
        let sparsity = f64::from(ctx) / f64::from(step.selected);
        println!("== ctx {ctx} ==");
        println!(
            "   {} blocks, budget {} whole + tail [{}, {}), {} selected rows per (sequence, head), {sparsity:.1}x sparsity",
            step.num_blocks, step.budget, step.tail_start, ctx, step.selected
        );

        let selection = match step.selection(args.kv_heads, args.block_size) {
            Ok(selection) => selection,
            Err(reason) => {
                println!("   sparse arm: SELECTION REJECTED ({reason})");
                continue;
            }
        };

        // State which path each arm takes, before timing either of them.
        let before = sparse_decode_stats();
        let (probe, outcome) = run_sparse(&step, &args, &selection);
        let dispatched = probe.is_some();
        if let Some(out) = &probe {
            mlxcel_core::eval(out);
        }
        let after = sparse_decode_stats();
        println!("   mask   arm: dense SDPA over {ctx} columns with an additive block mask");
        println!(
            "   sparse arm: {} -> {}",
            if dispatched {
                "FUSED PAGE INDIRECTION"
            } else {
                "FELL BACK (nothing fused)"
            },
            outcome.describe()
        );
        println!(
            "   sparse arm counters: fused +{}, fallbacks +{}",
            after.fused - before.fused,
            after.fallbacks - before.fallbacks
        );

        if !dispatched {
            println!(
                "   NOT COMPARABLE: the sparse arm did not run the fused kernel, so any timing \
                 below would compare the mask path against itself."
            );
            println!();
            continue;
        }

        if !args.skip_parity {
            let want = to_vec_f32(&run_mask(&step, &args));
            let got = to_vec_f32(probe.as_ref().unwrap());
            let err = max_rel_error(&got, &want);
            println!("   agreement: max relative error {err:.3e} (f16 storage)");
            if err > 5e-2 {
                println!(
                    "   ABORTING this context: the arms disagree, so a timing comparison is \
                     meaningless."
                );
                println!();
                continue;
            }
        }

        mlxcel_core::memory::reset_peak_memory();
        let base_peak = mlxcel_core::memory::peak_memory();
        let mask_ms = time_ms(args.reps, args.steps, || Some(run_mask(&step, &args)));
        let mask_peak = mlxcel_core::memory::peak_memory().saturating_sub(base_peak);

        mlxcel_core::memory::reset_peak_memory();
        let base_peak = mlxcel_core::memory::peak_memory();
        let sparse_ms = time_ms(args.reps, args.steps, || {
            run_sparse(&step, &args, &selection).0
        });
        let sparse_peak = mlxcel_core::memory::peak_memory().saturating_sub(base_peak);

        println!(
            "   mask   {mask_ms:8.4} ms/step   peak transient {:>10} B",
            mask_peak
        );
        println!(
            "   sparse {sparse_ms:8.4} ms/step   peak transient {:>10} B",
            sparse_peak
        );
        println!(
            "   speedup {:.2}x   transient delta {:+} B",
            mask_ms / sparse_ms.max(f64::MIN_POSITIVE),
            sparse_peak as i64 - mask_peak as i64
        );
        println!();
    }

    let stats = sparse_decode_stats();
    println!(
        "# totals: {} fused launches, {} fallbacks",
        stats.fused, stats.fallbacks
    );
}
