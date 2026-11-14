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

//! Eviction policy and configuration for the prompt prefix cache.
//!
//! The policy is intentionally small: LRU with a TTL escape hatch and both
//! byte-budget and entry-count caps. A production deployment can revisit
//! this layer without touching the store's public surface.

use std::fmt;
use std::time::Duration;

use super::block_hash::{ApcHashAlgo, DEFAULT_APC_BLOCK_SIZE};

/// Runtime configuration for the prompt-prefix cache store.
///
/// CLI/env parsing for these fields is tracked. Until then
/// construct an instance via [`PromptCacheConfig::default`] or the explicit
/// constructor and hand it to [`super::PromptCacheStore::with_config`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PromptCacheConfig {
    /// Toggle the entire feature. When `false`, callers should skip
    /// constructing a [`super::PromptCacheStore`] entirely so no memory is
    /// reserved. The store itself also honors this value as an early-out.
    pub enabled: bool,
    /// Total byte budget across all entries. Inserts that would exceed this
    /// after eviction are rejected.
    pub capacity_bytes: usize,
    /// Whether [`Self::capacity_bytes`] came from an operator flag/env.
    /// Startup may raise the compiled-in default for a model whose single
    /// conversation outgrows it, but never an operator cap.
    pub capacity_bytes_explicit: bool,
    /// Upper bound on the number of live cache entries. Oldest entries are
    /// evicted first once this cap is hit.
    pub max_entries: usize,
    /// Time-to-live for an entry since its last successful lookup. Lazy
    /// sweep: entries are checked and possibly expired on lookup and on
    /// [`super::PromptCacheStore::evict_if_needed`].
    pub ttl: Duration,
    /// Minimum number of prompt tokens required before an entry is eligible
    /// to be inserted. Helps avoid polluting the cache with tiny prefixes
    /// that can't really amortize the detach/adopt overhead.
    pub min_prefix_tokens: usize,
    /// Automatic Prefix Caching (APC) configuration.
    ///
    /// APC is an additive, opt-in feature layered on top of the existing
    /// prompt-prefix cache. It computes block-granularity hash chains so
    /// finer-grained KV reuse becomes possible. When [`ApcConfig::enabled`]
    /// is `false` (the default) no extra work is done and behaviour is
    /// identical to the earlier store.
    pub apc: ApcConfig,
    /// Byte budget dedicated to exact-prefix recurrent/model-owned snapshots.
    pub snapshot_capacity_bytes: usize,
    /// Whether [`Self::snapshot_capacity_bytes`] came from an operator flag/env
    /// or explicit builder call. Startup may raise the compiled-in default with
    /// a model-aware recommendation, but must not override an operator cap.
    pub snapshot_capacity_bytes_explicit: bool,
    /// Maximum live exact-prefix snapshot entries.
    pub snapshot_max_entries: usize,
    /// Time-to-live for snapshot entries since their last successful restore.
    pub snapshot_ttl: Duration,
    /// Keep each session's most recent restore source out of the
    /// session-chain supersede rule, so a request that edits or compacts the
    /// last turn resumes where that turn started. See
    /// [`keep_previous_turn_enabled`]; startup copies the switch in here.
    pub keep_previous_turn: bool,
}

/// Operator kill switch for the history-boundary snapshot (issue #1143).
///
/// The feature is on by default: it is what makes the prompt cache hit at all
/// for snapshot-only families. But it costs a second chat-template render and a
/// second tokenization on the request-dispatch thread, plus an extra graph
/// launch and a full model-state copy on the foreground prefill, and a
/// deployment that serves only single-turn traffic pays all of that for reuse it
/// will never claim. `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1` restores the
/// pre-#1143 request path end to end.
///
/// It lives here rather than in the scheduler because both ends of the feature
/// have to honor it: the route decides whether to render the history form at
/// all, and the scheduler decides whether to split the prefill. Gating only the
/// scheduler would leave the render and the encode being paid for a snapshot
/// that is then never taken.
///
/// Read once per process, so it costs nothing per request and setting it after
/// start has no effect.
///
/// Used by: `server::chat_request::prepare_chat_request_with_cache`,
/// `server::batch::scheduler::BatchScheduler::resolve_history_boundary`
pub fn boundary_snapshot_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("MLXCEL_DISABLE_BOUNDARY_SNAPSHOT")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Default prompt length at or above which the system-prompt root snapshot is
/// taken once `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT` is on (#2885 Part B).
pub const SYSTEM_ROOT_DEFAULT_MIN_PROMPT_TOKENS: usize = 2048;

/// Opt-in switch for the system-prompt root snapshot (#2885 Part B).
///
/// A snapshot-only family can reuse a stored state only when a new prompt
/// starts with exactly its tokens. The history boundary (#1143) covers the next
/// turn of the same chat; it does not cover a fresh chat that shares only the
/// system prompt and tool list with earlier ones, which is the common shape of
/// an agent client. With this switch on, a chat prompt of at least the returned
/// number of tokens also stores a snapshot at the end of its system block (the
/// system messages and the tool list, plus the first user turn's header), so
/// the next chat with the same system prompt starts after it.
///
/// `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT`:
/// * unset, empty or `0`: off (the default);
/// * `1`, `on` or `true`: on, for prompts of at least
///   [`SYSTEM_ROOT_DEFAULT_MIN_PROMPT_TOKENS`] tokens;
/// * any other integer: on, with that minimum prompt length.
///
/// Ignored when the history boundary itself is disabled
/// ([`boundary_snapshot_disabled`]), since both ride the same render and the
/// same prefill split. Read once per process.
///
/// Used by: `server::chat_request::prepare_chat_request_with_cache`,
/// `server::batch::scheduler::BatchScheduler::resolve_history_boundary`
pub fn system_root_min_prompt_tokens() -> Option<usize> {
    #[cfg(test)]
    if let Some(value) = SYSTEM_ROOT_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        if boundary_snapshot_disabled() {
            return None;
        }
        parse_system_root(
            std::env::var("MLXCEL_PROMPT_CACHE_SYSTEM_ROOT")
                .ok()
                .as_deref(),
        )
    })
}

#[cfg(test)]
thread_local! {
    static SYSTEM_ROOT_TEST_OVERRIDE: std::cell::Cell<Option<Option<usize>>> =
        const { std::cell::Cell::new(None) };
}

/// Pin [`system_root_min_prompt_tokens`] for the current test thread until the
/// guard drops, without touching the process environment.
#[cfg(test)]
pub(crate) fn override_system_root_for_test(value: Option<usize>) -> impl Drop {
    struct Reset(Option<Option<usize>>);
    impl Drop for Reset {
        fn drop(&mut self) {
            SYSTEM_ROOT_TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let previous = SYSTEM_ROOT_TEST_OVERRIDE.with(|cell| cell.replace(Some(value)));
    Reset(previous)
}

/// Parse an `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT` value; see
/// [`system_root_min_prompt_tokens`]. Invalid text warns once and stays off.
pub(crate) fn parse_system_root(value: Option<&str>) -> Option<usize> {
    let raw = value?.trim();
    match raw.to_ascii_lowercase().as_str() {
        "" | "0" | "off" | "false" => None,
        "1" | "on" | "true" => Some(SYSTEM_ROOT_DEFAULT_MIN_PROMPT_TOKENS),
        other => match other.parse::<usize>() {
            Ok(min) => Some(min),
            Err(_) => {
                tracing::warn!(
                    "MLXCEL_PROMPT_CACHE_SYSTEM_ROOT={raw:?} is neither a switch nor a token count; the system-prompt root snapshot stays off"
                );
                None
            }
        },
    }
}

/// Default spacing of the periodic prefill checkpoints once
/// `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL` is on (#2885 Part B).
pub const CHECKPOINT_DEFAULT_INTERVAL_TOKENS: usize = 16_384;

/// Smallest interval the environment variable accepts. Each checkpoint is one
/// more prefill split and one more state copy, so a tiny interval would turn
/// a long prompt into many short forwards.
const CHECKPOINT_MIN_INTERVAL_TOKENS: usize = 256;

/// Opt-in switch for periodic prefill checkpoints (#2885 Part B).
///
/// The history boundary (#1143) and the system-prompt root cover a chat that
/// appends to its previous turn and a new chat that shares only the system
/// block. Neither covers an agent that rewrites or compacts its history a few
/// thousand tokens back into a long context: the history snapshot no longer
/// matches and the root is far behind the edit. With this switch on, a
/// snapshot-family prefill also splits at every multiple of the returned
/// interval before its history boundary and stores a snapshot there, so such
/// a request resumes from the last checkpoint before the edit.
///
/// The points are absolute multiples of the interval, so they depend only on
/// the prompt prefix: a later prompt that shares the first `k * interval`
/// tokens splits there too, and a hit on that checkpoint forwards the same
/// pieces as the cold run of the same prompt.
///
/// `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL`:
/// * unset, empty or `0`: off (the default);
/// * `1`, `on` or `true`: on, every [`CHECKPOINT_DEFAULT_INTERVAL_TOKENS`];
/// * any other integer of at least 256: on, with that interval.
///
/// Ignored when the history boundary is disabled ([`boundary_snapshot_disabled`]).
/// Read once per process.
///
/// Used by: `server::batch::scheduler::BatchScheduler::checkpoint_splits`
pub fn checkpoint_interval_tokens() -> Option<usize> {
    #[cfg(test)]
    if let Some(value) = CHECKPOINT_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        if boundary_snapshot_disabled() {
            return None;
        }
        parse_checkpoint_interval(
            std::env::var("MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL")
                .ok()
                .as_deref(),
        )
    })
}

#[cfg(test)]
thread_local! {
    static CHECKPOINT_TEST_OVERRIDE: std::cell::Cell<Option<Option<usize>>> =
        const { std::cell::Cell::new(None) };
}

/// Pin [`checkpoint_interval_tokens`] for the current test thread until the
/// guard drops, without touching the process environment. Accepts any
/// interval, including ones below the environment variable's floor.
#[cfg(test)]
pub(crate) fn override_checkpoint_interval_for_test(value: Option<usize>) -> impl Drop {
    struct Reset(Option<Option<usize>>);
    impl Drop for Reset {
        fn drop(&mut self) {
            CHECKPOINT_TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let previous = CHECKPOINT_TEST_OVERRIDE.with(|cell| cell.replace(Some(value)));
    Reset(previous)
}

/// Parse an `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL` value; see
/// [`checkpoint_interval_tokens`]. Invalid text or an interval below the
/// floor warns and stays off.
pub(crate) fn parse_checkpoint_interval(value: Option<&str>) -> Option<usize> {
    let raw = value?.trim();
    match raw.to_ascii_lowercase().as_str() {
        "" | "0" | "off" | "false" => None,
        "1" | "on" | "true" => Some(CHECKPOINT_DEFAULT_INTERVAL_TOKENS),
        other => match other.parse::<usize>() {
            Ok(interval) if interval >= CHECKPOINT_MIN_INTERVAL_TOKENS => Some(interval),
            _ => {
                tracing::warn!(
                    "MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL={raw:?} is neither a switch nor a token interval of at least {CHECKPOINT_MIN_INTERVAL_TOKENS}; periodic checkpoints stay off"
                );
                None
            }
        },
    }
}

/// Opt-in switch that keeps the previous turn's restore point (#2885 Part B).
///
/// The session-chain supersede rule keeps one boundary and one completion
/// snapshot per conversation. Turn N restores from turn N-1's snapshot, and
/// inserting turn N's own snapshot supersedes it, so a request that replaces
/// or compacts turn N's user message finds no snapshot before the edit: turn
/// N's history boundary ends after that message. With this switch on, the
/// store remembers, per session, the snapshot the latest turn restored from
/// and exempts it from the supersede rule until a later turn restores from
/// something else. A conversation then holds one more snapshot, and such an
/// edit resumes where the previous turn started instead of at the
/// system-prompt root.
///
/// `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN`: `1`, `on` or `true` turns it on;
/// anything else, including unset, leaves it off. Ignored when the history
/// boundary is disabled ([`boundary_snapshot_disabled`]). Read once per
/// process.
///
/// Used by: `server::startup::resolve_prompt_cache_store`
pub fn keep_previous_turn_enabled() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        !boundary_snapshot_disabled()
            && parse_switch(
                std::env::var("MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN")
                    .ok()
                    .as_deref(),
            )
    })
}

/// Parse an on/off environment switch: `1`, `on` or `true` (any case) is on.
pub(crate) fn parse_switch(value: Option<&str>) -> bool {
    value.is_some_and(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "on" | "true"
        )
    })
}

/// Operator kill switch for the background prompt-cache warm-up (issue #1144).
///
/// Separate from [`boundary_snapshot_disabled`] on purpose. The boundary
/// snapshot is foreground work that buys the next turn a hit; the warm-up is
/// speculative background work that buys that hit a bigger prefix. An operator
/// may reasonably want the first without the second: the warm-up spends idle
/// GPU time and a second snapshot chain per conversation on a turn that may
/// never arrive. Setting `MLXCEL_DISABLE_CACHE_WARMUP=1` keeps #1143 and drops
/// #1144, which is also the arm separation a benchmark needs to attribute the
/// warm-up's effect.
///
/// Read once per process; setting it after start has no effect.
///
/// Used by: `server::routes::chat::submit_next_turn_warmup`,
/// `server::batch::scheduler::BatchScheduler::enqueue_prompt_cache_warmup`
pub fn cache_warmup_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED
        .get_or_init(|| std::env::var("MLXCEL_DISABLE_CACHE_WARMUP").ok().as_deref() == Some("1"))
}

/// Automatic Prefix Caching configuration knobs.
///
/// Mirrors the upstream `mlx-vlm` PR #1114 / WIP #1103 surface so callers can
/// tune block size, count, and hash algorithm via CLI flags or `APC_*` env
/// vars. See [`super::block_hash`] for how `block_size` and `hash` interact.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ApcConfig {
    /// Master switch for APC. When `false`, the block-hash chain is not
    /// computed and the existing whole-prefix cache path is the only one
    /// active.
    pub enabled: bool,
    /// Number of tokens per APC block. Defaults to
    /// [`DEFAULT_APC_BLOCK_SIZE`] (16) to match upstream.
    pub block_size: usize,
    /// Optional hard cap on the number of block-hash entries APC tracks. When
    /// `None`, the budget is derived from
    /// [`PromptCacheConfig::max_entries`] (a heuristic — the actual block
    /// count is bounded by `max_entries * (avg_prefix_len / block_size)`).
    pub num_blocks: Option<usize>,
    /// Hash algorithm used for block hashes. Defaults to SHA-256 for upstream
    /// wire-compat.
    pub hash: ApcHashAlgo,
}

impl Default for ApcConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            block_size: DEFAULT_APC_BLOCK_SIZE,
            num_blocks: None,
            hash: ApcHashAlgo::Sha256,
        }
    }
}

impl ApcConfig {
    /// Whether APC is fully active.
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.block_size > 0
    }
}

impl PromptCacheConfig {
    /// Default capacity in bytes: 2 GiB.
    pub const DEFAULT_CAPACITY_BYTES: usize = 2 * 1024 * 1024 * 1024;
    /// Default maximum entry count.
    pub const DEFAULT_MAX_ENTRIES: usize = 1024;
    /// Default TTL: 3600 seconds.
    pub const DEFAULT_TTL_SECONDS: u64 = 3600;
    /// Default minimum prompt-prefix length before caching kicks in.
    pub const DEFAULT_MIN_PREFIX_TOKENS: usize = 32;
    /// Default snapshot capacity: snapshots are smaller than KV blocks, so
    /// reserve one quarter of the KV budget for them by default.
    pub const DEFAULT_SNAPSHOT_CAPACITY_BYTES: usize = Self::DEFAULT_CAPACITY_BYTES / 4;
    /// Default snapshot entry count: exact-prefix snapshots are smaller, so
    /// keep more of them than detached KV entries.
    pub const DEFAULT_SNAPSHOT_MAX_ENTRIES: usize = Self::DEFAULT_MAX_ENTRIES * 4;
    /// Default snapshot TTL: allow longer multi-turn chat reuse than KV blocks.
    pub const DEFAULT_SNAPSHOT_TTL_SECONDS: u64 = Self::DEFAULT_TTL_SECONDS * 2;

    /// Build a fully-specified config. Prefer [`PromptCacheConfig::default`]
    /// unless a caller has a reason to deviate.
    pub fn new(
        enabled: bool,
        capacity_bytes: usize,
        max_entries: usize,
        ttl: Duration,
        min_prefix_tokens: usize,
    ) -> Self {
        Self {
            enabled,
            capacity_bytes,
            capacity_bytes_explicit: false,
            max_entries,
            ttl,
            min_prefix_tokens,
            apc: ApcConfig::default(),
            snapshot_capacity_bytes: Self::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            snapshot_capacity_bytes_explicit: false,
            snapshot_max_entries: Self::DEFAULT_SNAPSHOT_MAX_ENTRIES,
            snapshot_ttl: Duration::from_secs(Self::DEFAULT_SNAPSHOT_TTL_SECONDS),
            keep_previous_turn: false,
        }
    }

    /// Builder-style: keep each session's previous-turn restore source; see
    /// [`Self::keep_previous_turn`].
    #[must_use]
    pub fn with_keep_previous_turn(mut self, keep: bool) -> Self {
        self.keep_previous_turn = keep;
        self
    }

    /// Builder-style: attach APC configuration. Returns `self` so callers can
    /// chain `.with_apc(...)` after [`PromptCacheConfig::new`].
    #[must_use]
    pub fn with_apc(mut self, apc: ApcConfig) -> Self {
        self.apc = apc;
        self
    }

    /// Builder-style: override exact-prefix snapshot byte capacity.
    ///
    /// This marks the value as operator-supplied so startup's model-aware
    /// default sizing does not replace it.
    #[must_use]
    pub fn with_snapshot_capacity_bytes(mut self, capacity_bytes: usize) -> Self {
        self.snapshot_capacity_bytes = capacity_bytes;
        self.snapshot_capacity_bytes_explicit = true;
        self
    }

    /// Builder-style: override exact-prefix snapshot limits.
    #[must_use]
    pub fn with_snapshot_limits(
        mut self,
        capacity_bytes: usize,
        max_entries: usize,
        ttl: Duration,
    ) -> Self {
        self.snapshot_capacity_bytes = capacity_bytes;
        self.snapshot_capacity_bytes_explicit = true;
        self.snapshot_max_entries = max_entries;
        self.snapshot_ttl = ttl;
        self
    }

    /// Config variant with the feature disabled. Safe to hand to the store
    /// constructor; the resulting store is a cheap no-op.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Whether the store should accept any insert at all.
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.capacity_bytes > 0 && self.max_entries > 0
    }

    /// Whether the APC block-hash mode is active. Implies the store itself is
    /// enabled — APC layers on top of the regular prompt-prefix cache.
    pub fn apc_enabled(&self) -> bool {
        self.is_enabled() && self.apc.is_enabled()
    }
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capacity_bytes: Self::DEFAULT_CAPACITY_BYTES,
            capacity_bytes_explicit: false,
            max_entries: Self::DEFAULT_MAX_ENTRIES,
            ttl: Duration::from_secs(Self::DEFAULT_TTL_SECONDS),
            min_prefix_tokens: Self::DEFAULT_MIN_PREFIX_TOKENS,
            apc: ApcConfig::default(),
            snapshot_capacity_bytes: Self::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            snapshot_capacity_bytes_explicit: false,
            snapshot_max_entries: Self::DEFAULT_SNAPSHOT_MAX_ENTRIES,
            snapshot_ttl: Duration::from_secs(Self::DEFAULT_SNAPSHOT_TTL_SECONDS),
            keep_previous_turn: false,
        }
    }
}

/// Aggregate store statistics, intended for both tests and the metrics
/// bridge tracked.
///
/// This is a pure snapshot: values are captured under the store's lock and
/// returned by value so callers can release the lock immediately.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PromptCacheStats {
    /// Number of live entries in the store.
    pub entries: usize,
    /// Sum of `size_bytes` across all live entries.
    pub bytes: usize,
    /// Lifetime count of successful inserts.
    pub inserts: u64,
    /// Lifetime count of inserts rejected because the single entry exceeded
    /// the configured byte budget.
    pub rejections_oversized: u64,
    /// Lifetime count of lookup calls.
    pub lookups: u64,
    /// Lifetime count of lookup calls that returned `Some(..)`.
    pub hits: u64,
    /// Lifetime count of entries evicted due to LRU pressure (entry-cap or
    /// byte-cap).
    pub evictions_lru: u64,
    /// Lifetime count of entries evicted because the TTL expired.
    pub evictions_ttl: u64,
    /// Number of live exact-prefix recurrent/model-owned snapshots.
    pub snapshot_entries: usize,
    /// Sum of bytes across live snapshot entries.
    pub snapshot_bytes: usize,
    /// Lifetime count of successful snapshot inserts.
    pub snapshot_inserts: u64,
    /// Lifetime count of snapshot inserts rejected as oversized.
    pub snapshot_rejections_oversized: u64,
    /// Lifetime count of snapshot lookups.
    pub snapshot_lookups: u64,
    /// Lifetime count of snapshot lookup hits.
    pub snapshot_hits: u64,
    /// Lifetime count of snapshot LRU evictions.
    pub snapshot_evictions_lru: u64,
    /// Lifetime count of snapshot TTL evictions.
    pub snapshot_evictions_ttl: u64,
    /// Lifetime count of snapshots removed because a longer snapshot from the
    /// same session superseded them. Held apart from `snapshot_evictions_lru`
    /// so capacity tuning can tell deterministic in-session replacement from
    /// genuine budget pressure.
    pub snapshot_supersedes: u64,
    /// Lifetime count of snapshot inserts whose own session chain immediately
    /// lost a snapshot to the snapshot LRU cap. A non-zero value means the
    /// store could not retain one live snapshot per producer chain.
    pub snapshot_self_evictions: u64,
}

impl fmt::Display for PromptCacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "entries={} bytes={} inserts={} hits={}/{} lru_evict={} ttl_evict={} reject_oversized={} snapshot_entries={} snapshot_bytes={} snapshot_hits={}/{} snapshot_self_evict={}",
            self.entries,
            self.bytes,
            self.inserts,
            self.hits,
            self.lookups,
            self.evictions_lru,
            self.evictions_ttl,
            self.rejections_oversized,
            self.snapshot_entries,
            self.snapshot_bytes,
            self.snapshot_hits,
            self.snapshot_lookups,
            self.snapshot_self_evictions,
        )
    }
}

#[cfg(test)]
mod system_root_tests {
    use super::{SYSTEM_ROOT_DEFAULT_MIN_PROMPT_TOKENS, parse_system_root};

    #[test]
    fn system_root_switch_parses_off_on_and_thresholds() {
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("off"),
            Some("false"),
            Some("nope"),
        ] {
            assert_eq!(parse_system_root(off), None, "{off:?}");
        }
        for on in ["1", "on", "TRUE", " true "] {
            assert_eq!(
                parse_system_root(Some(on)),
                Some(SYSTEM_ROOT_DEFAULT_MIN_PROMPT_TOKENS),
                "{on:?}"
            );
        }
        assert_eq!(parse_system_root(Some("4096")), Some(4096));
        assert_eq!(parse_system_root(Some("2")), Some(2));
    }

    #[test]
    fn checkpoint_interval_parses_off_on_and_rejects_tiny_intervals() {
        use super::{CHECKPOINT_DEFAULT_INTERVAL_TOKENS, parse_checkpoint_interval};
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("off"),
            Some("false"),
            Some("nope"),
            Some("255"),
        ] {
            assert_eq!(parse_checkpoint_interval(off), None, "{off:?}");
        }
        for on in ["1", "on", "TRUE", " true "] {
            assert_eq!(
                parse_checkpoint_interval(Some(on)),
                Some(CHECKPOINT_DEFAULT_INTERVAL_TOKENS),
                "{on:?}"
            );
        }
        assert_eq!(parse_checkpoint_interval(Some("256")), Some(256));
        assert_eq!(parse_checkpoint_interval(Some("8192")), Some(8192));
    }
}
