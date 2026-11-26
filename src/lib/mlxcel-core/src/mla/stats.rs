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

//! Which MLA attention path each step actually took (issue #907).
//!
//! Issue #899 shipped a fused decode path that never activated, and the
//! before/after benchmark compared the fallback against itself and produced a
//! clean-looking null. The counters here exist so that cannot repeat: every MLA
//! attention call records the path it took, and the benchmark harness prints
//! the snapshot for each arm. An arm that claims "absorbed" while
//! `absorbed_composed` is 0 is visibly a measurement of the fallback.
//!
//! `tracing` is deliberately not used. The `mlxcel` CLI installs no tracing
//! subscriber at all (only `src/server/startup.rs` does), so a `tracing::info!`
//! from a CLI-reachable path emits nothing no matter what `RUST_LOG` says.
//! Counters plus an explicit print in the harness are observable everywhere.
//!
//! Every call is counted twice: once in the process-global counters that
//! [`take`] and [`report_if_requested`] read, and once in a per-thread tally
//! that [`scoped`] reads (issue #3298). A test asserting an exact count uses
//! [`scoped`], because the global counters also see every other MLA test the
//! harness runs in parallel. Path selection happens while the graph is built,
//! on the thread that calls the model's `forward`, so a test's own forwards
//! always land in its own tally.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// The MLA attention paths a step can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MlaDecodePath {
    /// Per-head K/V materialized from the latent; dense SDPA. The pre-#907
    /// path and the default. The cache may still hold the latent, as in the
    /// `deepseek_v3`-family MLA, where materializing is a per-step choice.
    Decompressed,
    /// A multi-token step that attended directly against the latent under the
    /// [`super::use_absorbed_mla`] cost model (speculative verify blocks and
    /// chunked-prefill continuations behind a long cache).
    AbsorbedPrefill,
    /// Absorbed single-token decode. Stage 1 is composed from MLX ops in
    /// `deepseek_v2`; the other MLA families run their fused latent SDPA here.
    AbsorbedComposed,
    /// Stage 2: absorbed decode split across latent chunks, merged with issue
    /// #898's `paged_attention_merge_states`.
    AbsorbedSplitKv,
    /// Absorbed single-token decode on the fused Metal kernel (issue #3121):
    /// both score halves, the softmax and the latent-value sum in one launch
    /// plus a merge, instead of the composed rope-score mask and SDPA.
    AbsorbedFused,
}

impl MlaDecodePath {
    /// Stable label for printed output and test assertions.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Decompressed => "decompressed",
            Self::AbsorbedPrefill => "absorbed_prefill",
            Self::AbsorbedComposed => "absorbed_composed",
            Self::AbsorbedSplitKv => "absorbed_split_kv",
            Self::AbsorbedFused => "absorbed_fused",
        }
    }
}

/// Every path, in report order.
pub const MLA_DECODE_PATHS: [MlaDecodePath; 5] = [
    MlaDecodePath::Decompressed,
    MlaDecodePath::AbsorbedPrefill,
    MlaDecodePath::AbsorbedComposed,
    MlaDecodePath::AbsorbedSplitKv,
    MlaDecodePath::AbsorbedFused,
];

static DECOMPRESSED: AtomicU64 = AtomicU64::new(0);
static ABSORBED_PREFILL: AtomicU64 = AtomicU64::new(0);
static ABSORBED_COMPOSED: AtomicU64 = AtomicU64::new(0);
static ABSORBED_SPLIT_KV: AtomicU64 = AtomicU64::new(0);
static ABSORBED_FUSED: AtomicU64 = AtomicU64::new(0);

fn counter(path: MlaDecodePath) -> &'static AtomicU64 {
    match path {
        MlaDecodePath::Decompressed => &DECOMPRESSED,
        MlaDecodePath::AbsorbedPrefill => &ABSORBED_PREFILL,
        MlaDecodePath::AbsorbedComposed => &ABSORBED_COMPOSED,
        MlaDecodePath::AbsorbedSplitKv => &ABSORBED_SPLIT_KV,
        MlaDecodePath::AbsorbedFused => &ABSORBED_FUSED,
    }
}

/// Record one MLA attention call on `path`.
///
/// `Relaxed` because the counters are diagnostics: they are read after the
/// measured region, never used to order anything.
#[inline]
pub fn record(path: MlaDecodePath) {
    // Inside [`as_one_call`] the first path is held and recorded once when
    // the scope ends.
    if ONE_CALL.with(|slot| match slot.get() {
        Some(held) => {
            slot.set(Some(held.or(Some(path))));
            true
        }
        None => false,
    }) {
        return;
    }
    counter(path).fetch_add(1, Ordering::Relaxed);
    // `try_with` so a record from a thread-local destructor during thread
    // teardown cannot panic; the global count above still lands.
    let _ = THREAD_COUNTS.try_with(|tally| {
        let mut counts = tally.get();
        counts.bump(path);
        tally.set(counts);
    });
}

/// Run `f` as one MLA attention call: whatever `f` records counts once, on
/// the first path it took. Nested inside another such scope, `f` folds into
/// the outer one.
///
/// A verify block that runs every row as its decode step's single query
/// (issue #3418) is one attention call per layer, as a block-wide step is;
/// without this a width-K block would read as K decode steps (#3421).
///
/// Used by: the per-row verify branches of `DeepSeekV3Attention::forward`,
/// the DeepSeek V3.2 `MLAAttention::verify_block_attention` and the DeepSeek
/// V2 `MLAAttention::forward_absorbed`.
pub fn as_one_call<R>(f: impl FnOnce() -> R) -> R {
    if ONE_CALL.with(Cell::get).is_some() {
        return f();
    }
    struct Finish;
    impl Drop for Finish {
        fn drop(&mut self) {
            let held = ONE_CALL.with(|slot| slot.replace(None)).flatten();
            if let Some(path) = held {
                record(path);
            }
        }
    }
    ONE_CALL.with(|slot| slot.set(Some(None)));
    let _finish = Finish;
    f()
}

thread_local! {
    /// `Some` inside [`as_one_call`], holding the first path recorded there.
    static ONE_CALL: Cell<Option<Option<MlaDecodePath>>> = const { Cell::new(None) };

    /// Calls recorded on this thread since it started. Never reset: [`scoped`]
    /// subtracts a before-reading from an after-reading, so nested scopes and
    /// a global [`take`] in between do not disturb each other.
    static THREAD_COUNTS: Cell<MlaDispatchCounts> = const { Cell::new(MlaDispatchCounts::ZERO) };
}

/// Run `f` and return the MLA calls recorded **on the calling thread** while
/// it ran, alongside its result.
///
/// Unlike a [`snapshot`] before and after, the counts exclude calls made by
/// other threads, so an exact-count assertion holds while other MLA tests run
/// concurrently. Calls made by threads that `f` spawns are not included.
/// Scopes nest: an inner scope's calls are also counted by the outer one.
///
/// Used by: the MLA dispatch assertions in `src/models/*_tests.rs`.
pub fn scoped<R>(f: impl FnOnce() -> R) -> (R, MlaDispatchCounts) {
    let before = THREAD_COUNTS.with(Cell::get);
    let result = f();
    let after = THREAD_COUNTS.with(Cell::get);
    (result, after.since(&before))
}

/// Environment variable that makes a run print the dispatch counters when it
/// ends.
pub const TRACE_STATS_ENV: &str = "MLXCEL_TRACE_MLA_STATS";

/// Print and reset the counters on stderr as `# mla_paths\t<summary>` when
/// [`TRACE_STATS_ENV`] is `1`. A run that took no MLA path prints nothing.
///
/// Used by: `mlxcel generate`, `mlxcel-bench-decode` and
/// `examples/logit_trace`, so an arm can prove which path it ran.
pub fn report_if_requested() {
    if std::env::var(TRACE_STATS_ENV).as_deref() != Ok("1") {
        return;
    }
    let counts = take();
    if counts.total() > 0 {
        eprintln!("# mla_paths\t{}", counts.summary());
    }
}

/// Per-path call counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MlaDispatchCounts {
    pub decompressed: u64,
    pub absorbed_prefill: u64,
    pub absorbed_composed: u64,
    pub absorbed_split_kv: u64,
    pub absorbed_fused: u64,
}

impl MlaDispatchCounts {
    /// All counts zero; usable in `const` context, unlike `Default`.
    const ZERO: Self = Self {
        decompressed: 0,
        absorbed_prefill: 0,
        absorbed_composed: 0,
        absorbed_split_kv: 0,
        absorbed_fused: 0,
    };

    fn slot(&mut self, path: MlaDecodePath) -> &mut u64 {
        match path {
            MlaDecodePath::Decompressed => &mut self.decompressed,
            MlaDecodePath::AbsorbedPrefill => &mut self.absorbed_prefill,
            MlaDecodePath::AbsorbedComposed => &mut self.absorbed_composed,
            MlaDecodePath::AbsorbedSplitKv => &mut self.absorbed_split_kv,
            MlaDecodePath::AbsorbedFused => &mut self.absorbed_fused,
        }
    }

    fn bump(&mut self, path: MlaDecodePath) {
        let slot = self.slot(path);
        *slot = slot.wrapping_add(1);
    }

    /// Per-path difference from an earlier reading of the same monotonic tally.
    /// Wrapping, so the difference stays exact even if a counter wrapped.
    fn since(&self, earlier: &Self) -> Self {
        let mut out = Self::ZERO;
        for path in MLA_DECODE_PATHS {
            *out.slot(path) = self.get(path).wrapping_sub(earlier.get(path));
        }
        out
    }

    /// Count for one path.
    #[must_use]
    pub const fn get(&self, path: MlaDecodePath) -> u64 {
        match path {
            MlaDecodePath::Decompressed => self.decompressed,
            MlaDecodePath::AbsorbedPrefill => self.absorbed_prefill,
            MlaDecodePath::AbsorbedComposed => self.absorbed_composed,
            MlaDecodePath::AbsorbedSplitKv => self.absorbed_split_kv,
            MlaDecodePath::AbsorbedFused => self.absorbed_fused,
        }
    }

    /// Total recorded calls.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.decompressed
            + self.absorbed_prefill
            + self.absorbed_composed
            + self.absorbed_split_kv
            + self.absorbed_fused
    }

    /// One-line `path=count` summary, in [`MLA_DECODE_PATHS`] order.
    ///
    /// This is what a benchmark arm prints so its executed path is provable
    /// from the transcript rather than inferred from the flag it was given.
    #[must_use]
    pub fn summary(&self) -> String {
        MLA_DECODE_PATHS
            .iter()
            .map(|p| format!("{}={}", p.label(), self.get(*p)))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Read the process-global counters without clearing them.
///
/// These include every thread's calls; a test that asserts on the calls its
/// own code made uses [`scoped`] instead.
#[must_use]
pub fn snapshot() -> MlaDispatchCounts {
    MlaDispatchCounts {
        decompressed: DECOMPRESSED.load(Ordering::Relaxed),
        absorbed_prefill: ABSORBED_PREFILL.load(Ordering::Relaxed),
        absorbed_composed: ABSORBED_COMPOSED.load(Ordering::Relaxed),
        absorbed_split_kv: ABSORBED_SPLIT_KV.load(Ordering::Relaxed),
        absorbed_fused: ABSORBED_FUSED.load(Ordering::Relaxed),
    }
}

/// Read the process-global counters and reset them to zero. The per-thread
/// tally [`scoped`] reads is unaffected.
///
/// A benchmark calls this between arms so each arm's summary describes only
/// that arm.
#[must_use]
pub fn take() -> MlaDispatchCounts {
    MlaDispatchCounts {
        decompressed: DECOMPRESSED.swap(0, Ordering::Relaxed),
        absorbed_prefill: ABSORBED_PREFILL.swap(0, Ordering::Relaxed),
        absorbed_composed: ABSORBED_COMPOSED.swap(0, Ordering::Relaxed),
        absorbed_split_kv: ABSORBED_SPLIT_KV.swap(0, Ordering::Relaxed),
        absorbed_fused: ABSORBED_FUSED.swap(0, Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_distinct_and_stable() {
        let mut seen = std::collections::HashSet::new();
        for p in MLA_DECODE_PATHS {
            assert!(seen.insert(p.label()), "duplicate label {}", p.label());
        }
        assert_eq!(MlaDecodePath::AbsorbedComposed.label(), "absorbed_composed");
        assert_eq!(MlaDecodePath::AbsorbedSplitKv.label(), "absorbed_split_kv");
    }

    #[test]
    fn summary_names_every_path_so_a_zero_is_visible() {
        let counts = MlaDispatchCounts {
            decompressed: 7,
            ..Default::default()
        };
        let s = counts.summary();
        for p in MLA_DECODE_PATHS {
            assert!(s.contains(p.label()), "{s} is missing {}", p.label());
        }
        // The trap this exists to catch: an "absorbed" arm whose absorbed
        // counter is zero must read as zero, not be omitted.
        assert!(s.contains("absorbed_composed=0"), "{s}");
        assert_eq!(counts.total(), 7);
        assert_eq!(counts.get(MlaDecodePath::Decompressed), 7);
    }

    /// Two threads record different paths at the same time; each `scoped`
    /// result holds only its own thread's calls (issue #3298). A
    /// before/after `snapshot()` would see both threads' calls here.
    #[test]
    fn scoped_counts_only_the_calling_threads_calls() {
        // `record` also bumps the global counters, which other mlxcel-core
        // tests assert on under this lock.
        let _guard = crate::mla::testkit::serial();
        const CALLS: u64 = 1000;
        let barrier = std::sync::Barrier::new(2);
        let run = |path: MlaDecodePath| {
            let ((), counts) = scoped(|| {
                // Both threads are inside their scope before either records,
                // and neither leaves before the other has finished.
                barrier.wait();
                for _ in 0..CALLS {
                    record(path);
                }
                barrier.wait();
            });
            counts
        };
        let (a, b) = std::thread::scope(|s| {
            let a = s.spawn(|| run(MlaDecodePath::Decompressed));
            let b = s.spawn(|| run(MlaDecodePath::AbsorbedFused));
            (a.join().unwrap(), b.join().unwrap())
        });
        assert_eq!(
            a,
            MlaDispatchCounts {
                decompressed: CALLS,
                ..Default::default()
            }
        );
        assert_eq!(
            b,
            MlaDispatchCounts {
                absorbed_fused: CALLS,
                ..Default::default()
            }
        );
        let _ = take();
    }

    #[test]
    fn as_one_call_records_the_first_path_once() {
        let _guard = crate::mla::testkit::serial();
        let (rows, counts) = scoped(|| {
            as_one_call(|| {
                record(MlaDecodePath::AbsorbedFused);
                as_one_call(|| record(MlaDecodePath::AbsorbedComposed));
                record(MlaDecodePath::AbsorbedFused);
                3
            })
        });
        assert_eq!(rows, 3);
        assert_eq!(
            counts,
            MlaDispatchCounts {
                absorbed_fused: 1,
                ..Default::default()
            }
        );
        let ((), after) = scoped(|| record(MlaDecodePath::Decompressed));
        assert_eq!(after.total(), 1, "the scope is closed again");
        let ((), empty) = scoped(|| as_one_call(|| ()));
        assert_eq!(
            empty.total(),
            0,
            "a scope that records nothing adds nothing"
        );
        let _ = take();
    }

    #[test]
    fn scoped_nests_and_survives_a_global_take() {
        let _guard = crate::mla::testkit::serial();
        let (inner, outer) = scoped(|| {
            record(MlaDecodePath::AbsorbedComposed);
            let ((), inner) = scoped(|| {
                record(MlaDecodePath::AbsorbedSplitKv);
                let _ = take();
                record(MlaDecodePath::AbsorbedSplitKv);
            });
            inner
        });
        assert_eq!(inner.absorbed_split_kv, 2);
        assert_eq!(inner.total(), 2);
        assert_eq!(outer.absorbed_composed, 1);
        assert_eq!(outer.absorbed_split_kv, 2);
        assert_eq!(outer.total(), 3);
        let _ = take();
    }
}
