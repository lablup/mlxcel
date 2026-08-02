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

//! Decode-attention benchmark for two-level cascade attention (issue #903).
//!
//! ## What this measures, and what it does not
//!
//! One layer's paged decode attention per step, over a synthetic pool whose
//! shared pages are *genuinely* one refcounted block per page (sequence 0 writes
//! the prefix, the rest adopt its block ids through `retain_block`, exactly the
//! fork `CachePool::clone_detached_paged_prefix` performs for an APC hit). It
//! does not run a model. Everything outside attention is identical between the
//! arms, so the ratio here is the ratio of the thing that changed; multiply it
//! into a model by the share of step time attention occupies at that context.
//!
//! The KV is fixed across steps rather than appended to, so a cell measures the
//! attention at exactly the stated context instead of a moving average over a
//! growing one.
//!
//! ## Three arms
//!
//! * `flat` is the production #899 path: one v2 launch over the whole batch,
//!   which reads the shared span once per request.
//! * `cascade` is #903: the shared span attended once for the subgroup, the
//!   per-request suffixes attended separately, and one merge.
//! * `no-share` is the overhead gate. The same shapes with no sharing at all,
//!   timing the flat launch with and without the cascade detection scan running
//!   in front of it, which is the cost every non-sharing workload pays for this
//!   feature existing.
//!
//! ## Proving which path an arm ran
//!
//! Every cascade cell prints `members=`, `shared_pages=` and the two chunk
//! counts taken from the launch that was just timed, and asserts that level 0
//! really folded the member queries onto the head axis. Every flat cell prints
//! its chunk count. Issue #899 shipped a fused decode path that never activated
//! and whose benchmark compared the fallback against itself; a cascade line
//! reporting `members=1`, or missing entirely, is visibly not measuring what its
//! name says. This prints to **stdout**, not through `tracing`: the `mlxcel`
//! binaries install no subscriber, so an `info!` here would be invisible in the
//! one situation it exists for.
//!
//! Run (Metal):
//!
//! ```text
//! cargo run --release --features metal,accelerate --example cascade_attention_bench
//!
//! # The grid issue #903 asks for.
//! cargo run --release --features metal,accelerate --example cascade_attention_bench -- \
//!     --shared 2048,8192 --batches 4,8 --tail 256 --steps 200 --warmup 40 --reps 5
//!
//! # Llama-3-8B geometry (32 q heads, 8 kv heads, head_dim 128) is the default;
//! # a Qwen3-4B-shaped run is:
//! cargo run --release --features metal,accelerate --example cascade_attention_bench -- \
//!     --q-heads 32 --kv-heads 8 --head-dim 128 --page-size 32
//! ```

use std::time::Instant;

use clap::Parser;
use mlxcel_core::cache::{PagedBlockId, PagedBlockPool, PagedKvLayout, PagedSequenceState};
use mlxcel_core::paged_v2::{
    PagedDecodeGeometry, PagedDecodePlan, build_cascade_plan, detect_shared_prefix,
    run_cascade_decode, run_decode_v2,
};
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Parser, Debug)]
#[command(
    name = "cascade_attention_bench",
    about = "Decode attention throughput for cascade (shared-prefix) attention (issue #903)"
)]
struct Args {
    /// Shared prefix lengths in tokens, rounded down to whole pages.
    #[arg(long, value_delimiter = ',', default_value = "2048,8192")]
    shared: Vec<usize>,
    /// Concurrent sequences sharing the prefix.
    #[arg(long, value_delimiter = ',', default_value = "4,8")]
    batches: Vec<usize>,
    /// Private tokens each sequence holds behind the shared prefix.
    #[arg(long, default_value_t = 256)]
    tail: usize,
    /// Measured decode steps per repetition.
    #[arg(long, default_value_t = 200)]
    steps: usize,
    /// Unmeasured warmup steps per repetition (also pays the kernel JIT).
    #[arg(long, default_value_t = 40)]
    warmup: usize,
    /// Repetitions per arm, so dispersion is visible from one invocation.
    #[arg(long, default_value_t = 5)]
    reps: usize,
    /// Query heads.
    #[arg(long, default_value_t = 32)]
    q_heads: i32,
    /// Key/value heads.
    #[arg(long, default_value_t = 8)]
    kv_heads: i32,
    /// Head dimension.
    #[arg(long, default_value_t = 128)]
    head_dim: i32,
    /// Pool page size in tokens.
    #[arg(long, default_value_t = 32)]
    page_size: usize,
    /// Skip the no-sharing overhead gate.
    #[arg(long)]
    no_overhead_gate: bool,
}

const F16: i32 = mlxcel_core::dtype::FLOAT16;

/// xorshift64* in [-1, 1), so a run is reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
                (unit * 2.0 - 1.0) * 0.5
            })
            .collect()
    }
}

/// A pool plus the sequence states of one decode batch.
struct Fixture {
    pool: PagedBlockPool,
    states: Vec<PagedSequenceState>,
    q: UniquePtr<MlxArray>,
    geometry: PagedDecodeGeometry,
    scale: f32,
}

impl Fixture {
    /// `shared_pages == 0` builds the no-sharing variant: every sequence owns
    /// its whole range.
    #[allow(clippy::too_many_arguments)]
    fn build(
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        page_size: usize,
        shared_pages: usize,
        tail: usize,
        batch: usize,
        seed: u64,
    ) -> Self {
        let hkv = kv_heads as usize;
        let dim = head_dim as usize;
        let layout = PagedKvLayout::uniform(1, page_size, page_size * hkv * dim * 2).unwrap();
        let mut pool = PagedBlockPool::new(layout);
        let mut rng = Rng::new(seed);
        let shared_len = shared_pages * page_size;
        let total_len = shared_len + tail;
        // The fused kernels read one contiguous pool buffer per side, so every
        // block of the batch has to land in the first slab. The 32-row default
        // caps that at 1024 tokens across the whole batch, which is below every
        // shape this benchmark exists to measure; size the slab to the batch up
        // front, which is what a server does through --ctx-size.
        let needed = batch * total_len.div_ceil(page_size) + 1;
        pool.set_slab_blocks(needed.next_power_of_two())
            .expect("slab sizing");

        let mut states: Vec<PagedSequenceState> = Vec::with_capacity(batch);
        let mut shared_blocks: Vec<PagedBlockId> = Vec::new();
        for b in 0..batch {
            let mut state = PagedSequenceState::new(pool.layout());
            let from_page = if shared_pages == 0 || b == 0 {
                pool.append_tokens(&mut state, 0, total_len).unwrap();
                if shared_pages > 0 && b == 0 {
                    shared_blocks = state.layer(0).unwrap().block_ids[..shared_pages].to_vec();
                }
                0
            } else {
                for id in &shared_blocks {
                    pool.retain_block(*id).unwrap();
                }
                {
                    let layer = state.layer_mut(0).unwrap();
                    layer.block_ids = shared_blocks.clone();
                    layer.len = shared_len;
                }
                pool.append_tokens(&mut state, 0, tail).unwrap();
                shared_pages
            };
            // Fill only the pages this sequence owns; shared pages were already
            // written by sequence 0.
            let block_ids = state.layer(0).unwrap().block_ids.clone();
            for block_id in block_ids.iter().skip(from_page) {
                let shape = [1, kv_heads, page_size as i32, head_dim];
                let k = mlxcel_core::astype(
                    &mlxcel_core::from_slice_f32(&rng.vec(hkv * page_size * dim), &shape),
                    F16,
                );
                let v = mlxcel_core::astype(
                    &mlxcel_core::from_slice_f32(&rng.vec(hkv * page_size * dim), &shape),
                    F16,
                );
                pool.write_block(*block_id, 0, 0, &k, &v).unwrap();
            }
            states.push(state);
        }

        let q_values = rng.vec(batch * q_heads as usize * dim);
        let q = mlxcel_core::from_slice_f32(&q_values, &[batch as i32, q_heads, 1, head_dim]);
        mlxcel_core::eval(&q);
        Self {
            pool,
            states,
            q,
            geometry: PagedDecodeGeometry {
                q_heads,
                kv_heads,
                head_dim,
                page_size: page_size as i32,
            },
            scale: 1.0 / (dim as f32).sqrt(),
        }
    }

    fn view(&self) -> mlxcel_core::cache::paged_csr::PagedCsrView {
        let refs: Vec<&PagedSequenceState> = self.states.iter().collect();
        self.pool.paged_csr_view(&refs, 0).unwrap()
    }

    fn pools(&self) -> (&MlxArray, &MlxArray) {
        self.pool.single_slab_tensors(0).expect("single-slab pool")
    }
}

/// Median, min and max of per-repetition milliseconds-per-step.
struct Timing {
    median: f64,
    min: f64,
    max: f64,
}

impl Timing {
    fn from(mut reps: Vec<f64>) -> Self {
        reps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self {
            median: reps[reps.len() / 2],
            min: reps[0],
            max: reps[reps.len() - 1],
        }
    }
    fn line(&self) -> String {
        format!(
            "{:8.4} ms/step (min {:.4}, max {:.4})",
            self.median, self.min, self.max
        )
    }
}

/// Time `body` for `reps` repetitions of `warmup + steps` calls.
fn time_arm<F>(args: &Args, mut body: F) -> Timing
where
    F: FnMut(),
{
    let mut reps = Vec::with_capacity(args.reps);
    for _ in 0..args.reps {
        for _ in 0..args.warmup {
            body();
        }
        let t0 = Instant::now();
        for _ in 0..args.steps {
            body();
        }
        reps.push(t0.elapsed().as_secs_f64() * 1000.0 / args.steps.max(1) as f64);
    }
    Timing::from(reps)
}

fn main() {
    let args = Args::parse();
    println!("# cascade attention decode benchmark (issue #903)");
    println!(
        "# geometry: q_heads={} kv_heads={} head_dim={} page_size={} tail={} \
         steps={} warmup={} reps={}",
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.page_size,
        args.tail,
        args.steps,
        args.warmup,
        args.reps
    );
    println!(
        "# arms are timed on the same fixture; each cell prints the launch stats \
         that prove which path it ran"
    );
    println!();

    for &shared_tokens in &args.shared {
        let shared_pages = shared_tokens / args.page_size;
        for &batch in &args.batches {
            let fx = Fixture::build(
                args.q_heads,
                args.kv_heads,
                args.head_dim,
                args.page_size,
                shared_pages,
                args.tail,
                batch,
                0x903 + batch as u64,
            );
            let view = fx.view();
            let (pool_k, pool_v) = fx.pools();
            let target = mlxcel_core::paged_v2::device_target_ctas();

            // -- flat --
            let flat_plan = PagedDecodePlan::heuristic(fx.geometry, &view.page_counts(), target);
            let flat = time_arm(&args, || {
                let out = run_decode_v2(&fx.q, pool_k, pool_v, &view, fx.scale)
                    .unwrap()
                    .expect("v2 serves this shape");
                mlxcel_core::eval(&out);
            });

            // -- cascade --
            // Detection runs against the real page table, with the same
            // thresholds the production path uses, so a cell that does not
            // qualify is reported rather than forced.
            let group = detect_shared_prefix(
                &view,
                mlxcel_core::paged_v2::min_shared_pages(),
                mlxcel_core::paged_v2::min_members(),
            );
            let Some(group) = group else {
                println!(
                    "shared={shared_tokens:5} batch={batch}  flat {}  cascade DECLINED \
                     (no subgroup clears MLXCEL_CASCADE_MIN_SHARED_PAGES / _MIN_MEMBERS); \
                     flat chunks={}",
                    flat.line(),
                    flat_plan.num_chunks
                );
                continue;
            };
            let plan = build_cascade_plan(&view, group).expect("cascade plan");
            let members = plan.members();
            let shared_pages_actual = plan.group.shared_pages;

            // One launch outside the timed region, to read the stats that
            // attribute the arm and to pay the level-0 kernel JIT.
            let (_, stats) = run_cascade_decode(
                &fx.q,
                pool_k,
                pool_v,
                &plan,
                fx.geometry,
                fx.scale,
                target,
            )
            .expect("cascade launch");
            assert_eq!(
                stats.prefix_q_heads,
                args.q_heads * members as i32,
                "level 0 did not fold the member queries onto the head axis; \
                 this arm is not running cascade"
            );

            let cascade = time_arm(&args, || {
                let (out, _) = run_cascade_decode(
                    &fx.q,
                    pool_k,
                    pool_v,
                    &plan,
                    fx.geometry,
                    fx.scale,
                    target,
                )
                .expect("cascade launch");
                mlxcel_core::eval(&out);
            });

            println!("shared={shared_tokens:5} batch={batch}");
            println!(
                "  path=flat     {}  chunks={}",
                flat.line(),
                flat_plan.num_chunks
            );
            println!(
                "  path=cascade  {}  members={members} shared_pages={shared_pages_actual} \
                 shared_tokens={} prefix_chunks={} suffix_chunks={} prefix_q_heads={}",
                cascade.line(),
                plan.shared_tokens(),
                stats.prefix_chunks,
                stats.suffix_chunks,
                stats.prefix_q_heads
            );
            println!(
                "  speedup       {:.3}x  (flat median / cascade median)",
                flat.median / cascade.median
            );
            println!();
        }
    }

    if args.no_overhead_gate {
        return;
    }

    // -- overhead gate: what a workload with no sharing pays for the feature. --
    println!("# no-sharing overhead gate (issue #903 requires < 1% end to end)");
    let shared_tokens = *args.shared.iter().max().unwrap_or(&2048);
    let per_request = shared_tokens + args.tail;
    for &batch in &args.batches {
        let fx = Fixture::build(
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.page_size,
            0,
            per_request,
            batch,
            0x0_5A_A5 ^ batch as u64,
        );
        let view = fx.view();
        let (pool_k, pool_v) = fx.pools();
        let min_pages = mlxcel_core::paged_v2::min_shared_pages();
        let min_members = mlxcel_core::paged_v2::min_members();
        assert!(
            detect_shared_prefix(&view, min_pages, min_members).is_none(),
            "the no-sharing fixture is sharing something; the gate would be meaningless"
        );

        let plain = time_arm(&args, || {
            let out = run_decode_v2(&fx.q, pool_k, pool_v, &view, fx.scale)
                .unwrap()
                .expect("v2 serves this shape");
            mlxcel_core::eval(&out);
        });
        let gated = time_arm(&args, || {
            // Exactly what the production path adds when cascade is enabled and
            // the batch shares nothing: one detection scan, then the flat
            // launch it would have run anyway.
            let found = detect_shared_prefix(&view, min_pages, min_members);
            assert!(found.is_none());
            let out = run_decode_v2(&fx.q, pool_k, pool_v, &view, fx.scale)
                .unwrap()
                .expect("v2 serves this shape");
            mlxcel_core::eval(&out);
        });
        let overhead = (gated.median - plain.median) / plain.median * 100.0;
        println!(
            "batch={batch} ctx={per_request}  flat {}  flat+detection {}  overhead {:+.3}% \
             of the attention step",
            plain.line(),
            gated.line(),
            overhead
        );
    }
}
