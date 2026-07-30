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

//! `mlxcel tune`: offline kernel autotuning (issue #906).
//!
//! Profiles the tuning matrix for the ops registered in
//! [`mlxcel_core::autotune::ops`] and writes the winners to the persistent
//! tactic cache under `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/autotune`. It
//! always profiles, so it works with `MLXCEL_AUTOTUNE` unset (the default);
//! the entries it writes take effect in later runs that set
//! `MLXCEL_AUTOTUNE=cache` or `MLXCEL_AUTOTUNE=1`.
//!
//! The paged-decode sweep is fully synthetic: the split-count optimum is a
//! launch-shape property (batch, heads, head dim, visible context), so the
//! sweep allocates pool tensors of the right shape rather than loading a
//! checkpoint. `--model` only reads `config.json` to take the head geometry
//! from a real architecture instead of the defaults; no weights are loaded.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use mlxcel_core::autotune::ops::cuda_kernel_knobs::{
    QmmShape, QmmTileOp, QmvMultirowOp, QmvShape, TILE_M_CAP_BLACKWELL,
};
use mlxcel_core::autotune::ops::paged_decode_splits::{DecodeShape, PagedDecodeSplitsOp};
use mlxcel_core::autotune::profile::{
    DEFAULT_MAX_REPS, DEFAULT_MIN_IMPROVEMENT, DEFAULT_REPS, DEFAULT_SAMPLE_BUDGET_US,
    DEFAULT_WARMUP, DEFAULT_WARMUP_BUDGET_US,
};
use mlxcel_core::autotune::{ProfileConfig, TacticStore, tune_and_store};
use mlxcel_core::{MlxArray, UniquePtr, eval, from_slice_i32, synchronize_default, zeros};

/// MLX f16 / f32 dtype ids.
const F16: i32 = 9;
const F32: i32 = 10;

/// Ops `mlxcel tune` knows how to profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum TuneOp {
    /// v1 paged-attention decode `NumSplits` launch shape (Metal + CUDA).
    PagedDecodeSplits,
    /// Blackwell qmm CTA `tile_m` (CUDA only, unvalidated).
    QmmTile,
    /// Multirow-qmv row-window ceiling (CUDA only, unvalidated).
    QmvMultirow,
}

impl TuneOp {
    fn label(self) -> &'static str {
        match self {
            TuneOp::PagedDecodeSplits => "paged-decode-splits",
            TuneOp::QmmTile => "qmm-tile",
            TuneOp::QmvMultirow => "qmv-multirow",
        }
    }

    fn is_cuda_only(self) -> bool {
        matches!(self, TuneOp::QmmTile | TuneOp::QmvMultirow)
    }
}

#[derive(Args, Debug)]
pub(crate) struct TuneArgs {
    /// Ops to tune. Repeat the flag for several; omit to tune every op that
    /// the live backend supports.
    #[arg(long = "op", value_enum)]
    pub(crate) ops: Vec<TuneOp>,

    /// Print the tuning matrix and exit without profiling.
    #[arg(long)]
    pub(crate) dry_run: bool,

    /// Model directory whose `config.json` supplies the head geometry for the
    /// paged-decode sweep (`head_dim`, `num_attention_heads`,
    /// `num_key_value_heads`). No weights are loaded.
    #[arg(long, short = 'm', value_name = "PATH")]
    pub(crate) model: Option<PathBuf>,

    /// Per-head dimension for the paged-decode sweep.
    #[arg(long, default_value = "128")]
    pub(crate) head_dim: i32,

    /// Query heads for the paged-decode sweep.
    #[arg(long, default_value = "32")]
    pub(crate) q_heads: i32,

    /// Key/value heads (GQA groups) for the paged-decode sweep.
    #[arg(long, default_value = "8")]
    pub(crate) kv_heads: i32,

    /// KV block size for the paged-decode sweep.
    #[arg(long, default_value = "32")]
    pub(crate) block_size: usize,

    /// Comma-separated batch sizes to sweep.
    #[arg(long, default_value = "1,2,4,8")]
    pub(crate) batch_sizes: String,

    /// Comma-separated visible context lengths to sweep.
    #[arg(long, default_value = "1024,4096,16384")]
    pub(crate) context_lengths: String,

    /// Minimum untimed warmup repetitions per candidate.
    #[arg(long, default_value_t = DEFAULT_WARMUP)]
    pub(crate) warmup: usize,

    /// Wall-clock warmup per candidate, milliseconds. Warmup runs past
    /// `--warmup` until this elapses, which is what brings the GPU to a steady
    /// clock before anything is recorded.
    #[arg(long, default_value_t = DEFAULT_WARMUP_BUDGET_US / 1000.0)]
    pub(crate) warmup_ms: f64,

    /// Minimum timed repetitions per candidate; the median is the candidate's
    /// score. The determinism guard wants at least 5.
    #[arg(long, default_value_t = DEFAULT_REPS)]
    pub(crate) reps: usize,

    /// Ceiling on the adaptively-grown repetition count.
    #[arg(long, default_value_t = DEFAULT_MAX_REPS)]
    pub(crate) max_reps: usize,

    /// Wall-clock sampling budget per candidate, milliseconds. Repetitions
    /// scale as `budget / per-launch cost`, so cheap cells (the noisy ones) get
    /// many more samples than expensive ones.
    #[arg(long, default_value_t = DEFAULT_SAMPLE_BUDGET_US / 1000.0)]
    pub(crate) sample_ms: f64,

    /// Floor on the relative win over the default a candidate must clear to be
    /// selected (0.02 = 2%). The effective threshold is the larger of this and
    /// the measured spread of the two medians being compared, so a noisy host
    /// converges back to today's behavior instead of to a coin flip.
    #[arg(long, default_value_t = DEFAULT_MIN_IMPROVEMENT)]
    pub(crate) min_improvement: f64,

    /// CTA `tile_m` ceiling the qmm sweep tunes within. Must match the cap the
    /// running binary's `make_cta_tiler` applies, otherwise the entry this
    /// writes will not be found at startup.
    #[arg(long, default_value_t = TILE_M_CAP_BLACKWELL)]
    pub(crate) qmm_tile_cap: i64,
}

impl TuneArgs {
    fn profile_config(&self) -> ProfileConfig {
        ProfileConfig {
            warmup: self.warmup,
            warmup_budget_us: self.warmup_ms * 1000.0,
            // The determinism guard is documented as median-of-5-or-more, so a
            // lower floor is refused rather than honored.
            reps: self.reps.max(DEFAULT_REPS),
            max_reps: self.max_reps,
            sample_budget_us: self.sample_ms * 1000.0,
            min_improvement: self.min_improvement,
        }
        .sanitized()
    }

    /// Ops to run: the explicit selection, or every op the backend supports.
    fn selected_ops(&self) -> Vec<TuneOp> {
        if !self.ops.is_empty() {
            let mut ops = self.ops.clone();
            ops.dedup();
            return ops;
        }
        let mut ops = vec![TuneOp::PagedDecodeSplits];
        if mlxcel_core::cuda_is_available() {
            ops.push(TuneOp::QmmTile);
            ops.push(TuneOp::QmvMultirow);
        }
        ops
    }
}

/// Parse a comma-separated list of positive `usize` values.
pub(crate) fn parse_usize_list(s: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for token in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let v: usize = token
            .parse()
            .with_context(|| format!("invalid list value {token:?}"))?;
        if v == 0 {
            anyhow::bail!("list values must be positive, got {token:?}");
        }
        out.push(v);
    }
    if out.is_empty() {
        anyhow::bail!("list must contain at least one positive value");
    }
    Ok(out)
}

/// Head geometry for the paged-decode sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HeadGeometry {
    pub(crate) head_dim: i32,
    pub(crate) q_heads: i32,
    pub(crate) kv_heads: i32,
}

/// Read the head geometry out of a checkpoint `config.json`.
///
/// Falls back to the caller's defaults for any field the config omits, so a
/// config without an explicit `head_dim` (the common case, where it is
/// `hidden_size / num_attention_heads`) still yields a usable geometry.
pub(crate) fn head_geometry_from_config(
    config: &serde_json::Value,
    fallback: HeadGeometry,
) -> HeadGeometry {
    let as_i32 = |key: &str| {
        config
            .get(key)
            .and_then(serde_json::Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
    };
    let q_heads_cfg = as_i32("num_attention_heads");
    let q_heads = q_heads_cfg.unwrap_or(fallback.q_heads);
    // A config that names attention heads but not KV heads is multi-head
    // attention, so KV heads equal query heads. A config that names neither
    // says nothing at all, so the caller's fallback stands.
    let kv_heads = as_i32("num_key_value_heads")
        .or(q_heads_cfg)
        .unwrap_or(fallback.kv_heads);
    let head_dim = as_i32("head_dim")
        .or_else(|| {
            let hidden = as_i32("hidden_size")?;
            if q_heads > 0 {
                Some(hidden / q_heads)
            } else {
                None
            }
        })
        .unwrap_or(fallback.head_dim);
    HeadGeometry {
        head_dim: head_dim.max(1),
        q_heads: q_heads.max(1),
        kv_heads: kv_heads.max(1),
    }
}

fn load_head_geometry(model: &Path, fallback: HeadGeometry) -> Result<HeadGeometry> {
    let path = model.join("config.json");
    let raw =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    // VLM checkpoints nest the language model's geometry.
    let node = config
        .get("text_config")
        .or_else(|| config.get("language_config"))
        .unwrap_or(&config);
    Ok(head_geometry_from_config(node, fallback))
}

/// Synthetic pool + block-table operands for one paged-decode launch shape.
struct DecodeWorkload {
    q: UniquePtr<MlxArray>,
    pool_k: UniquePtr<MlxArray>,
    pool_v: UniquePtr<MlxArray>,
    rows: UniquePtr<MlxArray>,
    row_offsets: UniquePtr<MlxArray>,
    logical_starts: UniquePtr<MlxArray>,
    visible_lens: UniquePtr<MlxArray>,
}

/// Physical pool rows for a `(batch, blocks-per-sequence)` sweep cell.
///
/// Reverse order over a pool with 2x slack, so the rows a sequence reads are
/// genuinely scattered. Matching `examples/page_gather_microbench.rs` keeps the
/// tuned access pattern comparable with the microbench numbers.
pub(crate) fn scattered_rows(batch: usize, blocks_per_seq: usize) -> Vec<i32> {
    let total = batch * blocks_per_seq;
    let pool = total * 2;
    (0..total).map(|i| (pool - 1 - i) as i32).collect()
}

/// `row_offsets[b]` is the start of sequence `b`'s rows; length `batch + 1`.
pub(crate) fn row_offsets_for(batch: usize, blocks_per_seq: usize) -> Vec<i32> {
    (0..=batch).map(|b| (b * blocks_per_seq) as i32).collect()
}

fn build_decode_workload(
    geom: HeadGeometry,
    batch: usize,
    context: usize,
    block_size: usize,
) -> DecodeWorkload {
    let blocks_per_seq = context.div_ceil(block_size);
    let pool_blocks = (batch * blocks_per_seq * 2).max(1) as i32;
    let bs = block_size as i32;

    let q = zeros(&[batch as i32, geom.q_heads, 1, geom.head_dim], F32);
    let pool_k = zeros(&[pool_blocks, bs, geom.kv_heads, geom.head_dim], F16);
    let pool_v = zeros(&[pool_blocks, bs, geom.kv_heads, geom.head_dim], F16);

    let rows_vec = scattered_rows(batch, blocks_per_seq);
    let rows = from_slice_i32(&rows_vec, &[rows_vec.len() as i32]);
    let offsets_vec = row_offsets_for(batch, blocks_per_seq);
    let row_offsets = from_slice_i32(&offsets_vec, &[offsets_vec.len() as i32]);
    let starts_vec = vec![0i32; batch];
    let logical_starts = from_slice_i32(&starts_vec, &[batch as i32]);
    let lens_vec = vec![context as i32; batch];
    let visible_lens = from_slice_i32(&lens_vec, &[batch as i32]);

    for arr in [
        &q,
        &pool_k,
        &pool_v,
        &rows,
        &row_offsets,
        &logical_starts,
        &visible_lens,
    ] {
        eval(arr);
    }
    synchronize_default();

    DecodeWorkload {
        q,
        pool_k,
        pool_v,
        rows,
        row_offsets,
        logical_starts,
        visible_lens,
    }
}

fn tune_paged_decode(args: &TuneArgs, store: &TacticStore, geom: HeadGeometry) -> Result<()> {
    let batches = parse_usize_list(&args.batch_sizes)?;
    let contexts = parse_usize_list(&args.context_lengths)?;
    println!(
        "paged-decode-splits: head_dim={} q_heads={} kv_heads={} block_size={}",
        geom.head_dim, geom.q_heads, geom.kv_heads, args.block_size
    );
    // `spread` is the selection's relative sample dispersion and `guard` the
    // improvement it had to clear (min_improvement, or the combined spread when
    // that is larger). A row whose guard dwarfs its speedup was decided by the
    // noise floor, not by the kernel.
    println!(
        "  {:>6} {:>8} | {:>12} {:>13} {:>9} {:>7} {:>5} {:>7} {:>8}",
        "batch", "context", "default", "selected", "best_us", "spread", "reps", "guard", "speedup"
    );

    for &batch in &batches {
        for &context in &contexts {
            if args.dry_run {
                println!("  {batch:>6} {context:>8} | (dry run)");
                continue;
            }
            let w = build_decode_workload(geom, batch, context, args.block_size);
            let shape = DecodeShape {
                batch,
                q_heads: geom.q_heads,
                kv_heads: geom.kv_heads,
                head_dim: geom.head_dim,
                context,
            };
            let op = PagedDecodeSplitsOp::new(
                &w.q,
                &w.pool_k,
                &w.pool_v,
                &w.rows,
                &w.row_offsets,
                &w.logical_starts,
                &w.visible_lens,
                1.0 / (geom.head_dim as f32).sqrt(),
                shape,
            );
            match tune_and_store(&op, store, args.profile_config()) {
                Some((_, record, result)) => println!(
                    "  {batch:>6} {context:>8} | {:>12} {:>13} {:>9.1} {:>6.1}% {:>5} {:>6.1}% {:>8}",
                    result
                        .default_us
                        .map_or_else(|| "n/a".to_string(), |v| format!("{v:.1}us")),
                    record.tactic.label,
                    record.latency_us,
                    record.spread * 100.0,
                    record.reps,
                    record.required_improvement * 100.0,
                    result
                        .speedup_over_default()
                        .map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}x")),
                ),
                None => println!("  {batch:>6} {context:>8} | skipped (no usable measurement)"),
            }
        }
    }
    Ok(())
}

fn tune_qmm_tile(args: &TuneArgs, store: &TacticStore) -> Result<()> {
    let shape = QmmShape::canonical();
    println!(
        "qmm-tile (UNVALIDATED, CUDA only): m={} n={} k={} cap={}",
        shape.m, shape.n, shape.k, args.qmm_tile_cap
    );
    if args.dry_run {
        return Ok(());
    }
    let op = QmmTileOp::new(shape, args.qmm_tile_cap).with_workload();
    report(tune_and_store(&op, store, args.profile_config()));
    Ok(())
}

fn tune_qmv_multirow(args: &TuneArgs, store: &TacticStore) -> Result<()> {
    let shape = QmvShape::canonical();
    println!(
        "qmv-multirow (UNVALIDATED, CUDA only): n={} k={} total_rows={}",
        shape.n, shape.k, shape.total_rows
    );
    if args.dry_run {
        return Ok(());
    }
    let op = QmvMultirowOp::new(shape).with_workload();
    report(tune_and_store(&op, store, args.profile_config()));
    Ok(())
}

fn report(
    outcome: Option<(
        mlxcel_core::autotune::TuneKey,
        mlxcel_core::autotune::TacticRecord,
        mlxcel_core::autotune::ProfileResult,
    )>,
) {
    match outcome {
        Some((key, record, result)) => println!(
            "  {} -> {} ({:.1}us +/-{:.1}% over {} reps, guard {:.1}%, speedup {})",
            key.display(),
            record.tactic.label,
            record.latency_us,
            record.spread * 100.0,
            record.reps,
            record.required_improvement * 100.0,
            result
                .speedup_over_default()
                .map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}x")),
        ),
        None => println!("  skipped (env override set, or no usable measurement)"),
    }
}

/// Entry point for `mlxcel tune`.
pub(crate) fn run_tune(args: TuneArgs) -> Result<()> {
    let store = TacticStore::from_cache_root();
    match store.dir() {
        Some(dir) => println!("autotune cache: {}", dir.display()),
        None => println!(
            "autotune cache: unavailable (no resolvable cache root); results will not persist"
        ),
    }
    let cfg = args.profile_config();
    println!(
        "profile: warmup>={} ({:.0}ms) reps {}..{} ({:.0}ms budget) min_improvement={:.3} (+ measured spread)",
        cfg.warmup,
        cfg.warmup_budget_us / 1000.0,
        cfg.reps,
        cfg.max_reps,
        cfg.sample_budget_us / 1000.0,
        cfg.min_improvement
    );

    let fallback = HeadGeometry {
        head_dim: args.head_dim,
        q_heads: args.q_heads,
        kv_heads: args.kv_heads,
    };
    let geom = match args.model.as_deref() {
        Some(path) => load_head_geometry(path, fallback)?,
        None => fallback,
    };

    for op in args.selected_ops() {
        if op.is_cuda_only() && !mlxcel_core::cuda_is_available() {
            println!("{}: skipped (CUDA backend not available)", op.label());
            continue;
        }
        match op {
            TuneOp::PagedDecodeSplits => tune_paged_decode(&args, &store, geom)?,
            TuneOp::QmmTile => tune_qmm_tile(&args, &store)?,
            TuneOp::QmvMultirow => tune_qmv_multirow(&args, &store)?,
        }
    }
    println!("done. Set MLXCEL_AUTOTUNE=cache (or 1) to consume these tactics at runtime.");
    Ok(())
}

#[cfg(test)]
#[path = "tune_tests.rs"]
mod tune_tests;
