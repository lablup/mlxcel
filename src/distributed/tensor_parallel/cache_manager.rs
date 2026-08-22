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

//! Tensor-parallel KV cache management.
//!
//! In tensor-parallel attention, each rank computes attention for a subset of
//! heads. The KV cache is similarly sharded: each rank stores cache entries
//! only for its assigned KV heads, providing proportional memory savings.
//!
//! Key components:
//!
//! - [`TPCacheConfig`] — per-rank cache configuration (heads, layers, budget)
//! - [`ShardedCacheAllocation`] — tracks a single sequence's cache on one rank
//! - [`TPCacheManager`] — per-rank cache tracking, admission, and eviction
//! - [`TPCacheMemoryReport`] — per-rank and aggregate memory accounting
//! - [`CacheSizeEstimate`] — estimated memory per sequence on one rank
//! - [`EvictionSignal`] — coordinated eviction broadcast from rank 0
//! - [`EvictionPolicy`] — configurable eviction strategy (LRU, LeastTokens)
//! - [`compute_per_rank_cache_size`] — memory needed per sequence per rank
//! - [`coordinate_eviction`] — synchronized eviction across all TP ranks
//! - [`aggregate_memory_reports`] — combine per-rank reports into a summary
//!
//! Used by: tensor_parallel attention forward pass, continuous batching scheduler

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use anyhow::{Result, ensure};

use super::parallel_attention::{
    AttentionType, KVAssignment, TPAttentionConfig, kv_head_assignment,
};

/// Unique identifier for a sequence in the KV cache.
/// Re-exported from pipeline cache_manager for consistency across the codebase.
pub type SequenceId = u64;

/// Configuration for tensor-parallel KV cache on a single rank.
#[derive(Debug, Clone, PartialEq)]
pub struct TPCacheConfig {
    /// This rank's index (0-based).
    pub tp_rank: usize,
    /// Total number of TP ranks.
    pub tp_size: usize,
    /// Total number of KV heads in the full model.
    pub total_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Number of layers in the model.
    pub num_layers: usize,
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Maximum number of concurrent sequences.
    pub max_sequences: usize,
    /// Memory budget for KV cache on this rank (bytes).
    pub memory_budget_bytes: u64,
    /// Bytes per element in the KV cache (e.g., 2 for float16, 4 for float32).
    pub bytes_per_element: usize,
    /// Memory pressure threshold (0.0 to 1.0). When usage exceeds this
    /// fraction, eviction signals are generated.
    pub pressure_threshold: f64,
}

impl TPCacheConfig {
    /// Compute the number of local KV heads on this rank.
    ///
    /// Uses the same assignment logic as `parallel_attention::kv_head_assignment`.
    pub fn local_kv_heads(&self) -> usize {
        let assignment = kv_head_assignment(self.tp_rank, self.tp_size, self.total_kv_heads);
        assignment.num_heads()
    }

    /// Compute the KV assignment for this rank.
    pub fn kv_assignment(&self) -> KVAssignment {
        kv_head_assignment(self.tp_rank, self.tp_size, self.total_kv_heads)
    }

    /// Classify the attention type based on total head counts.
    pub fn attention_type(&self, total_heads: usize) -> AttentionType {
        let attn_config = TPAttentionConfig {
            tp_rank: self.tp_rank,
            tp_size: self.tp_size,
            total_heads,
            total_kv_heads: self.total_kv_heads,
            head_dim: self.head_dim,
            sliding_window: None,
        };
        attn_config.attention_type()
    }

    /// Memory consumed per token per layer for KV cache on this rank.
    ///
    /// Each token stores K and V projections:
    /// `2 * local_kv_heads * head_dim * bytes_per_element`
    pub fn bytes_per_token_per_layer(&self) -> u64 {
        let local_kv = self.local_kv_heads();
        // 2 for K and V
        2 * local_kv as u64 * self.head_dim as u64 * self.bytes_per_element as u64
    }

    /// Estimate memory required for a sequence with the given token count.
    pub fn estimate_memory(&self, num_tokens: usize) -> u64 {
        self.bytes_per_token_per_layer() * num_tokens as u64 * self.num_layers as u64
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.tp_size >= 1, "tp_size must be >= 1");
        ensure!(
            self.tp_rank < self.tp_size,
            "tp_rank {rank} out of range for tp_size {size}",
            rank = self.tp_rank,
            size = self.tp_size
        );
        ensure!(self.total_kv_heads > 0, "total_kv_heads must be > 0");
        ensure!(self.head_dim > 0, "head_dim must be > 0");
        ensure!(self.num_layers > 0, "num_layers must be > 0");
        ensure!(self.max_seq_len > 0, "max_seq_len must be > 0");
        ensure!(self.max_sequences > 0, "max_sequences must be > 0");
        ensure!(
            self.memory_budget_bytes > 0,
            "memory_budget_bytes must be > 0"
        );
        ensure!(self.bytes_per_element > 0, "bytes_per_element must be > 0");
        ensure!(
            (0.0..=1.0).contains(&self.pressure_threshold),
            "pressure_threshold must be in [0.0, 1.0], got {t}",
            t = self.pressure_threshold
        );
        // When KV heads can be sharded, they must divide evenly.
        if self.total_kv_heads >= self.tp_size {
            ensure!(
                self.total_kv_heads.is_multiple_of(self.tp_size),
                "total_kv_heads ({kv}) must be divisible by tp_size ({tp}) when sharding",
                kv = self.total_kv_heads,
                tp = self.tp_size
            );
        }
        Ok(())
    }
}

/// Tracks the KV cache allocation for a single sequence on one TP rank.
#[derive(Debug, Clone)]
pub struct ShardedCacheAllocation {
    /// Sequence identifier.
    pub sequence_id: SequenceId,
    /// Number of local KV heads stored in this allocation.
    pub local_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Memory currently allocated for this sequence (bytes).
    pub allocated_memory_bytes: u64,
    /// Current cache offset (number of tokens cached so far).
    pub current_offset: usize,
    /// Prompt length (set at admission).
    pub prompt_len: usize,
    /// When this allocation was created.
    pub created_at: Instant,
    /// When this allocation was last accessed (for LRU eviction).
    pub last_accessed: Instant,
}

impl ShardedCacheAllocation {
    /// Update the offset (number of cached tokens) and recalculate memory.
    pub fn update_offset(
        &mut self,
        new_offset: usize,
        bytes_per_token_per_layer: u64,
        num_layers: usize,
    ) {
        self.current_offset = new_offset;
        self.allocated_memory_bytes =
            bytes_per_token_per_layer * new_offset as u64 * num_layers as u64;
        self.last_accessed = Instant::now();
    }

    /// Touch the allocation (update last_accessed without changing offset).
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }
}

impl fmt::Display for ShardedCacheAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TPCacheAlloc[seq={seq} kv_heads={kv} offset={off} mem={mem}B]",
            seq = self.sequence_id,
            kv = self.local_kv_heads,
            off = self.current_offset,
            mem = self.allocated_memory_bytes,
        )
    }
}

/// Estimated memory per sequence on one TP rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheSizeEstimate {
    /// Number of local KV heads on this rank.
    pub local_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// Bytes per token per layer on this rank.
    pub bytes_per_token_per_layer: u64,
    /// Whether KV heads are replicated (not sharded) on this rank.
    pub is_replicated: bool,
}

impl CacheSizeEstimate {
    /// Estimate memory for a given number of tokens.
    pub fn estimate_for_tokens(&self, num_tokens: usize) -> u64 {
        self.bytes_per_token_per_layer * num_tokens as u64 * self.num_layers as u64
    }
}

impl fmt::Display for CacheSizeEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheSizeEstimate[kv_heads={kv} head_dim={hd} layers={l} per_token_layer={b}B replicated={r}]",
            kv = self.local_kv_heads,
            hd = self.head_dim,
            l = self.num_layers,
            b = self.bytes_per_token_per_layer,
            r = self.is_replicated,
        )
    }
}

/// Compute the per-rank cache size estimate from a TP cache configuration.
pub fn compute_per_rank_cache_size(config: &TPCacheConfig) -> CacheSizeEstimate {
    let assignment = config.kv_assignment();
    let local_kv_heads = assignment.num_heads();
    let is_replicated = assignment.is_replicated();
    let bytes_per_token_per_layer =
        2 * local_kv_heads as u64 * config.head_dim as u64 * config.bytes_per_element as u64;

    CacheSizeEstimate {
        local_kv_heads,
        head_dim: config.head_dim,
        num_layers: config.num_layers,
        bytes_per_token_per_layer,
        is_replicated,
    }
}

/// Per-rank memory usage report.
#[derive(Debug, Clone, PartialEq)]
pub struct TPCacheMemoryReport {
    /// TP rank this report is for.
    pub tp_rank: usize,
    /// Memory currently used (bytes).
    pub used_bytes: u64,
    /// Total memory capacity (bytes).
    pub capacity_bytes: u64,
    /// Utilization fraction [0.0, 1.0].
    pub utilization: f64,
    /// Number of active sequences.
    pub total_sequences: usize,
    /// Number of local KV heads on this rank.
    pub local_kv_heads: usize,
    /// Whether KV heads are replicated on this rank.
    pub is_replicated: bool,
}

impl fmt::Display for TPCacheMemoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TPCacheReport[rank={rank} used={used}/{cap}B ({util:.1}%) seqs={seqs} kv_heads={kv} replicated={rep}]",
            rank = self.tp_rank,
            used = self.used_bytes,
            cap = self.capacity_bytes,
            util = self.utilization * 100.0,
            seqs = self.total_sequences,
            kv = self.local_kv_heads,
            rep = self.is_replicated,
        )
    }
}

/// Aggregate memory report across all TP ranks.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateMemoryReport {
    /// Per-rank reports.
    pub per_rank: Vec<TPCacheMemoryReport>,
    /// Total used memory across all ranks (bytes).
    pub total_used_bytes: u64,
    /// Total capacity across all ranks (bytes).
    pub total_capacity_bytes: u64,
    /// Overall utilization fraction.
    pub overall_utilization: f64,
    /// Maximum utilization across any single rank.
    pub max_rank_utilization: f64,
}

impl fmt::Display for AggregateMemoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AggregateCache[ranks={n} total={used}/{cap}B ({util:.1}%) max_rank={max:.1}%]",
            n = self.per_rank.len(),
            used = self.total_used_bytes,
            cap = self.total_capacity_bytes,
            util = self.overall_utilization * 100.0,
            max = self.max_rank_utilization * 100.0,
        )
    }
}

/// Aggregate per-rank memory reports into a summary.
pub fn aggregate_memory_reports(reports: &[TPCacheMemoryReport]) -> AggregateMemoryReport {
    let total_used: u64 = reports.iter().map(|r| r.used_bytes).sum();
    let total_cap: u64 = reports.iter().map(|r| r.capacity_bytes).sum();
    let overall = if total_cap > 0 {
        total_used as f64 / total_cap as f64
    } else {
        0.0
    };
    let max_util = reports
        .iter()
        .map(|r| r.utilization)
        .fold(0.0_f64, f64::max);

    AggregateMemoryReport {
        per_rank: reports.to_vec(),
        total_used_bytes: total_used,
        total_capacity_bytes: total_cap,
        overall_utilization: overall,
        max_rank_utilization: max_util,
    }
}

/// Signal broadcast from rank 0 to coordinate eviction across all TP ranks.
///
/// All ranks must evict the same sequences simultaneously to maintain
/// consistency. Eviction decisions are made by rank 0 and broadcast.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionSignal {
    /// Sequences to evict, in order.
    pub sequence_ids: Vec<SequenceId>,
    /// Reason for the eviction.
    pub reason: EvictionReason,
    /// Rank that originated the signal (always 0 for coordinated eviction).
    pub source_rank: usize,
}

impl fmt::Display for EvictionSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EvictionSignal[from=rank-{rank} seqs={n} reason={reason}]",
            rank = self.source_rank,
            n = self.sequence_ids.len(),
            reason = self.reason,
        )
    }
}

/// Reason for cache eviction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvictionReason {
    /// Memory pressure exceeded threshold.
    MemoryPressure,
    /// Sequence completed (EOS or max tokens).
    SequenceComplete,
    /// Explicit eviction request (e.g., client cancellation).
    ExplicitRequest,
}

impl fmt::Display for EvictionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryPressure => write!(f, "memory_pressure"),
            Self::SequenceComplete => write!(f, "sequence_complete"),
            Self::ExplicitRequest => write!(f, "explicit_request"),
        }
    }
}

/// Policy for selecting sequences to evict under memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EvictionPolicy {
    /// Evict the least recently used sequence.
    #[default]
    LRU,
    /// Evict the sequence with the fewest cached tokens.
    LeastTokens,
}

impl fmt::Display for EvictionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LRU => write!(f, "LRU"),
            Self::LeastTokens => write!(f, "LeastTokens"),
        }
    }
}

/// Per-rank KV cache manager for tensor parallelism.
///
/// Each TP rank has its own `TPCacheManager` that tracks cache allocations
/// for only the KV heads assigned to that rank. Cache operations (append,
/// slice, rotate) are independent per rank. Eviction decisions are
/// coordinated by rank 0 via [`EvictionSignal`].
///
/// Used by: tensor_parallel attention forward pass, continuous batching scheduler
#[derive(Debug)]
pub struct TPCacheManager {
    /// Configuration for this rank.
    config: TPCacheConfig,
    /// Active cache allocations, keyed by sequence ID.
    allocations: HashMap<SequenceId, ShardedCacheAllocation>,
    /// Current total memory usage across all allocations (bytes).
    used_memory_bytes: u64,
    /// Eviction policy for selecting eviction candidates.
    eviction_policy: EvictionPolicy,
    /// Cached value: number of local KV heads on this rank.
    local_kv_heads: usize,
    /// Cached value: bytes per token per layer for this rank.
    bytes_per_token_per_layer: u64,
    /// Whether KV heads are replicated on this rank.
    is_replicated: bool,
}

impl TPCacheManager {
    /// Create a new cache manager for a TP rank.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: TPCacheConfig) -> Result<Self> {
        config.validate()?;
        let estimate = compute_per_rank_cache_size(&config);
        Ok(Self {
            local_kv_heads: estimate.local_kv_heads,
            bytes_per_token_per_layer: estimate.bytes_per_token_per_layer,
            is_replicated: estimate.is_replicated,
            config,
            allocations: HashMap::new(),
            used_memory_bytes: 0,
            eviction_policy: EvictionPolicy::default(),
        })
    }

    /// Set the eviction policy.
    #[must_use]
    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// TP rank of this manager.
    pub fn tp_rank(&self) -> usize {
        self.config.tp_rank
    }

    /// TP size.
    pub fn tp_size(&self) -> usize {
        self.config.tp_size
    }

    /// Number of local KV heads on this rank.
    pub fn local_kv_heads(&self) -> usize {
        self.local_kv_heads
    }

    /// Whether KV heads are replicated (not sharded) on this rank.
    pub fn is_replicated(&self) -> bool {
        self.is_replicated
    }

    /// Number of active sequences in the cache.
    pub fn active_sequences(&self) -> usize {
        self.allocations.len()
    }

    /// Total memory currently used (bytes).
    pub fn used_memory(&self) -> u64 {
        self.used_memory_bytes
    }

    /// Available memory remaining (bytes).
    pub fn available_memory(&self) -> u64 {
        self.config
            .memory_budget_bytes
            .saturating_sub(self.used_memory_bytes)
    }

    /// Current memory usage as a fraction of the budget [0.0, 1.0].
    pub fn memory_usage_fraction(&self) -> f64 {
        if self.config.memory_budget_bytes == 0 {
            return 1.0;
        }
        self.used_memory_bytes as f64 / self.config.memory_budget_bytes as f64
    }

    /// Whether memory pressure exceeds the configured threshold.
    pub fn is_under_pressure(&self) -> bool {
        self.memory_usage_fraction() >= self.config.pressure_threshold
    }

    /// Allocate cache for a new sequence.
    ///
    /// # Arguments
    ///
    /// * `sequence_id` — unique identifier for the sequence
    /// * `seq_len` — initial number of tokens to cache (prompt length)
    ///
    /// # Returns
    ///
    /// The allocation if successful, or an error if the sequence is already
    /// cached, the max sequence count is reached, or there is insufficient memory.
    pub fn allocate_cache(
        &mut self,
        sequence_id: SequenceId,
        seq_len: usize,
    ) -> Result<ShardedCacheAllocation> {
        ensure!(
            !self.allocations.contains_key(&sequence_id),
            "sequence {sequence_id} already cached on rank {rank}",
            rank = self.config.tp_rank
        );
        ensure!(
            self.allocations.len() < self.config.max_sequences,
            "max sequences ({max}) reached on rank {rank}",
            max = self.config.max_sequences,
            rank = self.config.tp_rank
        );

        let required = self.config.estimate_memory(seq_len);
        let available = self.available_memory();
        ensure!(
            required <= available,
            "insufficient memory on rank {rank}: need {required}B, have {available}B",
            rank = self.config.tp_rank,
        );

        let now = Instant::now();
        let alloc = ShardedCacheAllocation {
            sequence_id,
            local_kv_heads: self.local_kv_heads,
            head_dim: self.config.head_dim,
            allocated_memory_bytes: required,
            current_offset: seq_len,
            prompt_len: seq_len,
            created_at: now,
            last_accessed: now,
        };

        self.used_memory_bytes += required;
        self.allocations.insert(sequence_id, alloc.clone());
        Ok(alloc)
    }

    /// Free cache for a sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence is not found.
    pub fn free_cache(&mut self, sequence_id: SequenceId) -> Result<()> {
        let alloc = self.allocations.remove(&sequence_id).ok_or_else(|| {
            anyhow::anyhow!(
                "sequence {sequence_id} not found on rank {rank}",
                rank = self.config.tp_rank
            )
        })?;
        self.used_memory_bytes = self
            .used_memory_bytes
            .saturating_sub(alloc.allocated_memory_bytes);
        Ok(())
    }

    /// Update the cache offset for a sequence (e.g., after appending tokens).
    ///
    /// Adjusts allocated memory to reflect the new offset.
    pub fn update_offset(&mut self, sequence_id: SequenceId, new_offset: usize) -> Result<()> {
        let alloc = self.allocations.get_mut(&sequence_id).ok_or_else(|| {
            anyhow::anyhow!(
                "sequence {sequence_id} not found on rank {rank}",
                rank = self.config.tp_rank
            )
        })?;
        let old_mem = alloc.allocated_memory_bytes;
        alloc.update_offset(
            new_offset,
            self.bytes_per_token_per_layer,
            self.config.num_layers,
        );
        let new_mem = alloc.allocated_memory_bytes;
        debug_assert!(
            self.used_memory_bytes >= old_mem,
            "memory accounting invariant violated: used={used} < old_alloc={old}",
            used = self.used_memory_bytes,
            old = old_mem,
        );
        self.used_memory_bytes = self
            .used_memory_bytes
            .saturating_sub(old_mem)
            .saturating_add(new_mem);
        Ok(())
    }

    /// Generate a memory usage report for this rank.
    pub fn memory_report(&self) -> TPCacheMemoryReport {
        TPCacheMemoryReport {
            tp_rank: self.config.tp_rank,
            used_bytes: self.used_memory_bytes,
            capacity_bytes: self.config.memory_budget_bytes,
            utilization: self.memory_usage_fraction(),
            total_sequences: self.allocations.len(),
            local_kv_heads: self.local_kv_heads,
            is_replicated: self.is_replicated,
        }
    }

    /// Check memory pressure and return an eviction signal if the threshold
    /// is exceeded. Only rank 0 should use this to coordinate eviction.
    ///
    /// The returned signal contains only enough candidates (in eviction-priority
    /// order) to bring memory usage below the pressure threshold.
    pub fn check_pressure(&self) -> Option<EvictionSignal> {
        if !self.is_under_pressure() {
            return None;
        }

        let all_candidates = self.select_eviction_candidates();
        if all_candidates.is_empty() {
            return None;
        }

        // Select only enough candidates to bring usage below the threshold.
        let target_bytes =
            (self.config.memory_budget_bytes as f64 * self.config.pressure_threshold) as u64;
        let mut projected_used = self.used_memory_bytes;
        let mut selected = Vec::new();

        for seq_id in all_candidates {
            if projected_used <= target_bytes {
                break;
            }
            if let Some(alloc) = self.allocations.get(&seq_id) {
                projected_used = projected_used.saturating_sub(alloc.allocated_memory_bytes);
                selected.push(seq_id);
            }
        }

        if selected.is_empty() {
            return None;
        }

        Some(EvictionSignal {
            sequence_ids: selected,
            reason: EvictionReason::MemoryPressure,
            source_rank: self.config.tp_rank,
        })
    }

    /// Apply an eviction signal (typically broadcast from rank 0).
    ///
    /// Evicts all sequences listed in the signal. Sequences not found are
    /// silently skipped (idempotent).
    pub fn apply_eviction(&mut self, signal: &EvictionSignal) -> Result<()> {
        for &seq_id in &signal.sequence_ids {
            if let Some(alloc) = self.allocations.remove(&seq_id) {
                self.used_memory_bytes = self
                    .used_memory_bytes
                    .saturating_sub(alloc.allocated_memory_bytes);
            }
        }
        Ok(())
    }

    /// Get the allocation for a specific sequence, if present.
    pub fn get_allocation(&self, sequence_id: SequenceId) -> Option<&ShardedCacheAllocation> {
        self.allocations.get(&sequence_id)
    }

    /// Iterate over all active allocations.
    pub fn allocations(&self) -> impl Iterator<Item = &ShardedCacheAllocation> {
        self.allocations.values()
    }

    /// Reference to the cache configuration.
    pub fn config(&self) -> &TPCacheConfig {
        &self.config
    }

    /// Select all active sequences ordered by the current eviction policy.
    ///
    /// Returns ALL sequence IDs in eviction priority order (first = evict first).
    /// Callers are responsible for selecting a subset sufficient to relieve pressure.
    fn select_eviction_candidates(&self) -> Vec<SequenceId> {
        let mut entries: Vec<_> = self.allocations.values().collect();

        // The `sequence_id` component is what makes each sort key a TOTAL
        // order, and it must stay. `sort_by_key` is stable, `entries` comes out
        // of a `HashMap` whose iteration order `RandomState` randomizes per
        // instance, and `check_pressure` consumes only a prefix of this list.
        // Without the tie-break, allocations that tie on the policy key keep
        // hash order and which of them actually loses its cache changes from
        // run to run.
        match self.eviction_policy {
            EvictionPolicy::LRU => {
                entries.sort_by_key(|a| (a.last_accessed, a.sequence_id));
            }
            EvictionPolicy::LeastTokens => {
                entries.sort_by_key(|a| (a.current_offset, a.sequence_id));
            }
        }

        entries.iter().map(|a| a.sequence_id).collect()
    }
}

impl fmt::Display for TPCacheManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TPCacheManager[rank={rank}/{tp} kv_heads={kv} seqs={seqs}/{max} mem={used}/{cap}B ({util:.1}%) replicated={rep}]",
            rank = self.config.tp_rank,
            tp = self.config.tp_size,
            kv = self.local_kv_heads,
            seqs = self.allocations.len(),
            max = self.config.max_sequences,
            used = self.used_memory_bytes,
            cap = self.config.memory_budget_bytes,
            util = self.memory_usage_fraction() * 100.0,
            rep = self.is_replicated,
        )
    }
}

/// Coordinate eviction across all TP ranks.
///
/// Applies the eviction signal to all managers. All ranks must evict the
/// same sequences to maintain consistency.
///
/// # Arguments
///
/// * `managers` — mutable references to all TP rank cache managers
/// * `signal` — the eviction signal (typically from rank 0)
pub fn coordinate_eviction(
    managers: &mut [&mut TPCacheManager],
    signal: &EvictionSignal,
) -> Result<()> {
    for mgr in managers.iter_mut() {
        mgr.apply_eviction(signal)?;
    }
    Ok(())
}

/// Collect memory reports from all TP ranks and aggregate them.
pub fn collect_memory_reports(managers: &[&TPCacheManager]) -> AggregateMemoryReport {
    let reports: Vec<_> = managers.iter().map(|m| m.memory_report()).collect();
    aggregate_memory_reports(&reports)
}

/// Check all TP ranks for memory pressure and return the most urgent signal.
///
/// When multiple ranks are under pressure, the rank with the highest
/// utilization is selected as the source.
pub fn check_tp_pressure(managers: &[&TPCacheManager]) -> Option<EvictionSignal> {
    let mut best_signal: Option<EvictionSignal> = None;
    let mut best_utilization: f64 = -1.0;

    for mgr in managers {
        if let Some(signal) = mgr.check_pressure() {
            let util = mgr.memory_usage_fraction();
            if util > best_utilization {
                best_utilization = util;
                best_signal = Some(signal);
            }
        }
    }

    best_signal
}

#[cfg(test)]
#[path = "cache_manager_tests.rs"]
mod tests;
