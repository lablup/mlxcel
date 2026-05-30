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

//! Pipeline-aware KV cache management for pipeline parallelism.
//!
//! Each pipeline stage only runs a subset of layers, so each stage allocates
//! KV cache only for its assigned layers. Cache management decisions
//! (admitting new sequences, evicting sequences under memory pressure) are
//! coordinated across all stages to maintain consistency.
//!
//! Key components:
//!
//! - [`PipelineCacheConfig`] — per-stage cache configuration
//! - [`StageCacheAllocation`] — tracks cache allocation for a single sequence
//! - [`CacheAdmissionRequest`] — request to admit a new sequence
//! - [`AdmissionDecision`] — outcome of an admission request
//! - [`PipelineCacheManager`] — per-stage cache tracking and admission
//! - [`EvictionEvent`] — broadcast when a sequence is evicted
//! - [`PreemptionSignal`] — emitted under memory pressure
//! - [`PreemptionPolicy`] — configurable eviction strategy
//! - [`CacheMetadataSync`] — cross-stage cache state consistency
//!
//! Used by: pipeline execution loop, batch scheduler, server startup

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::time::Instant;

use anyhow::{Result, ensure};

/// Unique identifier for a sequence in the KV cache.
pub type SequenceId = u64;

/// Configuration for per-stage KV cache allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineCacheConfig {
    /// Index of this pipeline stage (0-based).
    pub stage_index: u32,
    /// Total number of stages in the pipeline.
    pub num_stages: u32,
    /// Range of layer indices assigned to this stage (half-open).
    pub layer_range: Range<usize>,
    /// Maximum number of sequences that can be cached simultaneously.
    pub max_sequences: usize,
    /// Total memory budget for KV cache on this stage (bytes).
    pub memory_budget_bytes: u64,
    /// Memory consumed per layer per token for KV cache (bytes).
    /// Depends on hidden_size, num_kv_heads, head_dim, and dtype.
    pub bytes_per_layer_per_token: u64,
    /// Memory pressure threshold (0.0 to 1.0). When usage exceeds this
    /// fraction of the budget, preemption signals are emitted.
    pub pressure_threshold: f64,
}

impl PipelineCacheConfig {
    /// Number of layers assigned to this stage.
    pub fn num_layers(&self) -> usize {
        self.layer_range.end - self.layer_range.start
    }

    /// Estimate memory required for a sequence with the given token count
    /// on this stage (considering only the layers assigned here).
    pub fn estimate_memory(&self, num_tokens: usize) -> u64 {
        self.num_layers() as u64 * num_tokens as u64 * self.bytes_per_layer_per_token
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.num_stages >= 2,
            "pipeline requires at least 2 stages, got {}",
            self.num_stages
        );
        ensure!(
            self.stage_index < self.num_stages,
            "stage_index {} out of range for {}-stage pipeline",
            self.stage_index,
            self.num_stages
        );
        ensure!(
            self.layer_range.end > self.layer_range.start,
            "empty layer range on stage {}",
            self.stage_index
        );
        ensure!(self.max_sequences > 0, "max_sequences must be > 0");
        ensure!(
            self.memory_budget_bytes > 0,
            "memory_budget_bytes must be > 0"
        );
        ensure!(
            self.bytes_per_layer_per_token > 0,
            "bytes_per_layer_per_token must be > 0"
        );
        ensure!(
            (0.0..=1.0).contains(&self.pressure_threshold),
            "pressure_threshold must be in [0.0, 1.0], got {}",
            self.pressure_threshold
        );
        Ok(())
    }
}

/// Tracks the KV cache allocation for a single sequence on one stage.
#[derive(Debug, Clone)]
pub struct StageCacheAllocation {
    /// Sequence identifier.
    pub sequence_id: SequenceId,
    /// Layer range this allocation covers (same as the stage's layer range).
    pub layer_range: Range<usize>,
    /// Memory currently allocated for this sequence (bytes).
    pub allocated_memory_bytes: u64,
    /// Current cache offset (number of tokens cached so far).
    pub current_offset: usize,
    /// Prompt length (set at admission, used for metadata sync).
    pub prompt_len: usize,
    /// When this allocation was created.
    pub created_at: Instant,
    /// When this allocation was last accessed (for LRU eviction).
    pub last_accessed: Instant,
}

impl StageCacheAllocation {
    /// Update the offset (number of cached tokens) and recalculate memory.
    pub fn update_offset(&mut self, new_offset: usize, bytes_per_layer_per_token: u64) {
        self.current_offset = new_offset;
        let num_layers = self.layer_range.end - self.layer_range.start;
        self.allocated_memory_bytes =
            num_layers as u64 * new_offset as u64 * bytes_per_layer_per_token;
        self.last_accessed = Instant::now();
    }

    /// Touch the allocation (update last_accessed without changing offset).
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }
}

impl fmt::Display for StageCacheAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheAlloc[seq={} layers={}..{} offset={} mem={}B]",
            self.sequence_id,
            self.layer_range.start,
            self.layer_range.end,
            self.current_offset,
            self.allocated_memory_bytes,
        )
    }
}

/// Request to admit a new sequence into the KV cache across all stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheAdmissionRequest {
    /// Sequence to admit.
    pub sequence_id: SequenceId,
    /// Length of the prompt (number of tokens to cache initially).
    pub prompt_len: usize,
    /// Estimated maximum generation length (for memory reservation).
    /// If zero, only prompt_len is used for the initial estimate.
    pub estimated_max_tokens: usize,
}

impl CacheAdmissionRequest {
    /// Create a new admission request.
    pub fn new(sequence_id: SequenceId, prompt_len: usize) -> Self {
        Self {
            sequence_id,
            prompt_len,
            estimated_max_tokens: 0,
        }
    }

    /// Set the estimated maximum generation length.
    #[must_use]
    pub fn with_estimated_max_tokens(mut self, max_tokens: usize) -> Self {
        self.estimated_max_tokens = max_tokens;
        self
    }

    /// Effective token count for memory estimation.
    pub fn effective_tokens(&self) -> usize {
        if self.estimated_max_tokens > 0 {
            self.prompt_len + self.estimated_max_tokens
        } else {
            self.prompt_len
        }
    }
}

/// Outcome of a cache admission request.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdmissionDecision {
    /// Sequence was admitted; cache space has been reserved.
    Admitted,
    /// Sequence was rejected.
    Rejected(RejectionReason),
}

/// Reason why a sequence was rejected from the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RejectionReason {
    /// Maximum number of concurrent sequences reached.
    MaxSequencesReached,
    /// Insufficient memory on this stage.
    InsufficientMemory {
        required_bytes: u64,
        available_bytes: u64,
    },
    /// Sequence is already cached.
    AlreadyCached,
    /// Stage rejected by another stage in the pipeline.
    RejectedByStage { stage_index: u32 },
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxSequencesReached => write!(f, "maximum sequences reached"),
            Self::InsufficientMemory {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "insufficient memory: need {required_bytes}B, have {available_bytes}B"
            ),
            Self::AlreadyCached => write!(f, "sequence already cached"),
            Self::RejectedByStage { stage_index } => {
                write!(f, "rejected by stage {stage_index}")
            }
        }
    }
}

impl RejectionReason {
    /// Short, low-cardinality label for metric dimensions and traces.
    ///
    /// Used by: `/metrics` admission rejection counters,
    /// chrome-tracing rejection events.
    pub fn metric_label(&self) -> &'static str {
        match self {
            Self::MaxSequencesReached => "sequence_cap",
            Self::InsufficientMemory { .. } => "memory",
            Self::AlreadyCached => "already_cached",
            Self::RejectedByStage { .. } => "rejected_by_stage",
        }
    }
}

/// Enriched OOM diagnostic produced by [`coordinated_admission_with_attribution`]
/// Operators see which stage rejected, why, and where the
/// offending stage's KV occupancy sits at the moment of rejection.
///
/// Used by: server HTTP error path, `/metrics` counters, chrome-tracing
/// `admission_reject` events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionDiagnostic {
    /// Stage that rejected the sequence.
    pub stage_index: u32,
    /// Underlying rejection reason.
    pub reason: RejectionReason,
    /// Byte occupancy currently held by the rejecting stage's KV cache.
    pub used_memory_bytes: u64,
    /// Byte budget configured on that stage.
    pub budget_bytes: u64,
    /// Active sequence count on the rejecting stage at the moment of the
    /// rejection.
    pub active_sequences: usize,
}

impl fmt::Display for AdmissionDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let frac = if self.budget_bytes == 0 {
            0.0
        } else {
            (self.used_memory_bytes as f64 / self.budget_bytes as f64) * 100.0
        };
        write!(
            f,
            "admission rejected on stage {} ({}): used {}B / {}B ({:.1}%), {} active sequences",
            self.stage_index,
            self.reason,
            self.used_memory_bytes,
            self.budget_bytes,
            frac,
            self.active_sequences
        )
    }
}

/// Event broadcast when a sequence is evicted from the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionEvent {
    /// Sequence that was evicted.
    pub sequence_id: SequenceId,
    /// Stage that initiated the eviction.
    pub initiating_stage: u32,
    /// Reason for the eviction.
    pub reason: EvictionReason,
}

/// Reason for cache eviction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvictionReason {
    /// Memory pressure triggered eviction.
    MemoryPressure,
    /// Sequence completed (EOS or max tokens).
    SequenceComplete,
    /// Explicit eviction request (e.g., client cancellation).
    ExplicitRequest,
    /// Preemption to make room for a higher-priority sequence.
    Preemption,
}

impl fmt::Display for EvictionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryPressure => write!(f, "memory_pressure"),
            Self::SequenceComplete => write!(f, "sequence_complete"),
            Self::ExplicitRequest => write!(f, "explicit_request"),
            Self::Preemption => write!(f, "preemption"),
        }
    }
}

impl fmt::Display for EvictionEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Eviction[seq={} from_stage={} reason={}]",
            self.sequence_id, self.initiating_stage, self.reason,
        )
    }
}

/// Signal emitted when memory pressure requires preemption.
#[derive(Debug, Clone)]
pub struct PreemptionSignal {
    /// Sequences recommended for eviction, in priority order (evict first last).
    pub sequence_ids: Vec<SequenceId>,
    /// Stage that detected the pressure.
    pub source_stage: u32,
    /// Current memory usage fraction on the source stage.
    pub memory_usage_fraction: f64,
    /// Reason for the preemption.
    pub reason: PreemptionReason,
}

/// Reason for preemption.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PreemptionReason {
    /// Memory usage exceeded the pressure threshold.
    MemoryPressure,
    /// Need to admit a new sequence but no room available.
    AdmissionRequired,
}

impl fmt::Display for PreemptionSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Preemption[stage={} usage={:.1}% evict={} seqs reason={:?}]",
            self.source_stage,
            self.memory_usage_fraction * 100.0,
            self.sequence_ids.len(),
            self.reason,
        )
    }
}

/// Policy for selecting sequences to evict under memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PreemptionPolicy {
    /// Evict the least recently used sequence.
    #[default]
    LRU,
    /// Evict the sequence with the shortest cached context.
    Shortest,
    /// Evict the sequence with the longest cached context
    /// (frees the most memory per eviction).
    Longest,
}

impl fmt::Display for PreemptionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LRU => write!(f, "LRU"),
            Self::Shortest => write!(f, "Shortest"),
            Self::Longest => write!(f, "Longest"),
        }
    }
}

/// Cache metadata for cross-stage synchronization.
///
/// When the coordinator (typically stage 0) updates cache state, it
/// broadcasts a `CacheMetadataSync` to all other stages so they can
/// maintain a consistent view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheMetadataSync {
    /// Sequence whose metadata is being synchronized.
    pub sequence_id: SequenceId,
    /// Current cache offset (number of tokens cached).
    pub current_offset: usize,
    /// Prompt length (immutable after admission).
    pub prompt_len: usize,
    /// Whether the sequence is still active.
    pub is_active: bool,
    /// Source stage that produced this sync message.
    pub source_stage: u32,
}

impl fmt::Display for CacheMetadataSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheSync[seq={} offset={} prompt={} active={} from=stage-{}]",
            self.sequence_id,
            self.current_offset,
            self.prompt_len,
            self.is_active,
            self.source_stage,
        )
    }
}

/// Per-stage KV cache manager.
///
/// Tracks cache allocations for all sequences on a single pipeline stage,
/// handles admission requests, eviction, and memory pressure monitoring.
///
/// In a multi-stage pipeline, each stage has its own `PipelineCacheManager`.
/// Coordinated admission and eviction are handled by calling the appropriate
/// methods on each stage's manager and checking that all stages agree.
///
/// Used by: pipeline execution loop, batch scheduler
#[derive(Debug)]
pub struct PipelineCacheManager {
    /// Configuration for this stage.
    config: PipelineCacheConfig,
    /// Active cache allocations, keyed by sequence ID.
    allocations: HashMap<SequenceId, StageCacheAllocation>,
    /// Current total memory usage across all allocations (bytes).
    used_memory_bytes: u64,
    /// Preemption policy for selecting eviction candidates.
    preemption_policy: PreemptionPolicy,
}

impl PipelineCacheManager {
    /// Create a new cache manager for a pipeline stage.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: PipelineCacheConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            allocations: HashMap::new(),
            used_memory_bytes: 0,
            preemption_policy: PreemptionPolicy::default(),
        })
    }

    /// Set the preemption policy.
    #[must_use]
    pub fn with_preemption_policy(mut self, policy: PreemptionPolicy) -> Self {
        self.preemption_policy = policy;
        self
    }

    /// Stage index of this manager.
    pub fn stage_index(&self) -> u32 {
        self.config.stage_index
    }

    /// Number of layers managed by this stage.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers()
    }

    /// Layer range assigned to this stage.
    pub fn layer_range(&self) -> &Range<usize> {
        &self.config.layer_range
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

    /// Try to admit a new sequence to this stage's cache.
    ///
    /// Checks capacity (max sequences) and memory budget. Does NOT
    /// coordinate with other stages -- the caller must check all stages
    /// and only finalize admission if all return `Admitted`.
    pub fn request_admission(&mut self, req: &CacheAdmissionRequest) -> AdmissionDecision {
        // Check if already cached.
        if self.allocations.contains_key(&req.sequence_id) {
            return AdmissionDecision::Rejected(RejectionReason::AlreadyCached);
        }

        // Check sequence count limit.
        if self.allocations.len() >= self.config.max_sequences {
            return AdmissionDecision::Rejected(RejectionReason::MaxSequencesReached);
        }

        // Check memory budget.
        let required = self.config.estimate_memory(req.effective_tokens());
        let available = self.available_memory();
        if required > available {
            return AdmissionDecision::Rejected(RejectionReason::InsufficientMemory {
                required_bytes: required,
                available_bytes: available,
            });
        }

        // Admit: create the allocation.
        let now = Instant::now();
        let alloc = StageCacheAllocation {
            sequence_id: req.sequence_id,
            layer_range: self.config.layer_range.clone(),
            allocated_memory_bytes: self.config.estimate_memory(req.prompt_len),
            current_offset: req.prompt_len,
            prompt_len: req.prompt_len,
            created_at: now,
            last_accessed: now,
        };
        self.used_memory_bytes += alloc.allocated_memory_bytes;
        self.allocations.insert(req.sequence_id, alloc);

        AdmissionDecision::Admitted
    }

    /// Evict a sequence from this stage's cache.
    ///
    /// Returns an `EvictionEvent` describing the eviction, or an error
    /// if the sequence is not found.
    pub fn evict(
        &mut self,
        sequence_id: SequenceId,
        reason: EvictionReason,
    ) -> Result<EvictionEvent> {
        let alloc = self.allocations.remove(&sequence_id).ok_or_else(|| {
            anyhow::anyhow!(
                "sequence {sequence_id} not found on stage {}",
                self.config.stage_index
            )
        })?;
        debug_assert!(
            self.used_memory_bytes >= alloc.allocated_memory_bytes,
            "memory accounting underflow: used={}B alloc={}B seq={}",
            self.used_memory_bytes,
            alloc.allocated_memory_bytes,
            sequence_id,
        );
        self.used_memory_bytes = self
            .used_memory_bytes
            .saturating_sub(alloc.allocated_memory_bytes);

        Ok(EvictionEvent {
            sequence_id,
            initiating_stage: self.config.stage_index,
            reason,
        })
    }

    /// Broadcast an eviction from another stage: evict the sequence locally.
    ///
    /// Returns `Ok(())` if the sequence was found and evicted, or if it
    /// was already absent (idempotent). Returns error only on internal
    /// inconsistency.
    pub fn apply_eviction_broadcast(&mut self, event: &EvictionEvent) -> Result<()> {
        if let Some(alloc) = self.allocations.remove(&event.sequence_id) {
            debug_assert!(
                self.used_memory_bytes >= alloc.allocated_memory_bytes,
                "memory accounting underflow in broadcast: used={}B alloc={}B seq={}",
                self.used_memory_bytes,
                alloc.allocated_memory_bytes,
                event.sequence_id,
            );
            self.used_memory_bytes = self
                .used_memory_bytes
                .saturating_sub(alloc.allocated_memory_bytes);
        }
        // Idempotent: if the sequence is already gone, that's fine.
        Ok(())
    }

    /// Update cache metadata for a sequence from a cross-stage sync message.
    ///
    /// Adjusts the local allocation's offset and memory to match the
    /// coordinator's state. If the sequence is marked inactive in the
    /// sync, it is evicted.
    pub fn apply_metadata_sync(&mut self, sync: &CacheMetadataSync) -> Result<()> {
        if !sync.is_active {
            // Sequence has been deactivated; evict locally.
            if self.allocations.contains_key(&sync.sequence_id) {
                self.evict(sync.sequence_id, EvictionReason::SequenceComplete)?;
            }
            return Ok(());
        }

        if let Some(alloc) = self.allocations.get_mut(&sync.sequence_id) {
            let old_mem = alloc.allocated_memory_bytes;
            alloc.update_offset(sync.current_offset, self.config.bytes_per_layer_per_token);
            // Adjust total used memory.
            debug_assert!(
                self.used_memory_bytes >= old_mem,
                "memory accounting underflow in metadata sync: used={}B old_alloc={}B seq={}",
                self.used_memory_bytes,
                old_mem,
                sync.sequence_id,
            );
            self.used_memory_bytes = self
                .used_memory_bytes
                .saturating_sub(old_mem)
                .saturating_add(alloc.allocated_memory_bytes);
        }
        // If the sequence is not present locally, ignore (it may have
        // been evicted already by a race or the sequence was never
        // admitted to this stage).
        Ok(())
    }

    /// Check memory pressure and return a preemption signal if the
    /// pressure threshold is exceeded.
    ///
    /// The signal contains a list of sequence IDs recommended for
    /// eviction according to the configured preemption policy. The
    /// list is ordered so that evicting sequences from the front
    /// should relieve pressure progressively.
    pub fn check_memory_pressure(&self) -> Option<PreemptionSignal> {
        if !self.is_under_pressure() {
            return None;
        }

        let candidates = self.select_eviction_candidates();
        if candidates.is_empty() {
            return None;
        }

        Some(PreemptionSignal {
            sequence_ids: candidates,
            source_stage: self.config.stage_index,
            memory_usage_fraction: self.memory_usage_fraction(),
            reason: PreemptionReason::MemoryPressure,
        })
    }

    /// Select eviction candidates ordered by the current preemption policy.
    ///
    /// Returns sequence IDs in eviction priority order (first = evict first).
    fn select_eviction_candidates(&self) -> Vec<SequenceId> {
        let mut entries: Vec<_> = self.allocations.values().collect();

        match self.preemption_policy {
            PreemptionPolicy::LRU => {
                entries.sort_by_key(|a| a.last_accessed);
            }
            PreemptionPolicy::Shortest => {
                entries.sort_by_key(|a| a.current_offset);
            }
            PreemptionPolicy::Longest => {
                entries.sort_by_key(|a| std::cmp::Reverse(a.current_offset));
            }
        }

        entries.iter().map(|a| a.sequence_id).collect()
    }

    /// Generate a metadata sync message for a given sequence.
    ///
    /// Returns `None` if the sequence is not present on this stage.
    pub fn generate_metadata_sync(&self, sequence_id: SequenceId) -> Option<CacheMetadataSync> {
        self.allocations
            .get(&sequence_id)
            .map(|alloc| CacheMetadataSync {
                sequence_id,
                current_offset: alloc.current_offset,
                prompt_len: alloc.prompt_len,
                is_active: true,
                source_stage: self.config.stage_index,
            })
    }

    /// Get the allocation for a specific sequence, if present.
    pub fn get_allocation(&self, sequence_id: SequenceId) -> Option<&StageCacheAllocation> {
        self.allocations.get(&sequence_id)
    }

    /// Iterate over all active allocations.
    pub fn allocations(&self) -> impl Iterator<Item = &StageCacheAllocation> {
        self.allocations.values()
    }

    /// Reference to the cache configuration.
    pub fn config(&self) -> &PipelineCacheConfig {
        &self.config
    }
}

impl fmt::Display for PipelineCacheManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheManager[stage={} layers={}..{} seqs={}/{} mem={}/{}B ({:.1}%)]",
            self.config.stage_index,
            self.config.layer_range.start,
            self.config.layer_range.end,
            self.allocations.len(),
            self.config.max_sequences,
            self.used_memory_bytes,
            self.config.memory_budget_bytes,
            self.memory_usage_fraction() * 100.0,
        )
    }
}

/// Coordinate admission of a sequence across multiple stage cache managers.
///
/// Checks all stages and only finalizes admission if every stage accepts.
/// If any stage rejects, the sequence is not admitted on any stage.
///
/// # Arguments
///
/// * `managers` - Mutable references to all stage cache managers.
/// * `req` - The admission request.
///
/// # Returns
///
/// `Ok(AdmissionDecision::Admitted)` if all stages accepted.
/// `Ok(AdmissionDecision::Rejected(...))` if any stage rejected (with the
/// reason from the first rejecting stage).
pub fn coordinated_admission(
    managers: &mut [&mut PipelineCacheManager],
    req: &CacheAdmissionRequest,
) -> Result<AdmissionDecision> {
    ensure!(!managers.is_empty(), "at least one manager is required");

    // Phase 1: Check all stages without mutating (dry-run).
    // We check capacity and memory constraints for each stage.
    for mgr in managers.iter() {
        if mgr.allocations.contains_key(&req.sequence_id) {
            return Ok(AdmissionDecision::Rejected(RejectionReason::AlreadyCached));
        }
        if mgr.allocations.len() >= mgr.config.max_sequences {
            return Ok(AdmissionDecision::Rejected(
                RejectionReason::MaxSequencesReached,
            ));
        }
        let required = mgr.config.estimate_memory(req.effective_tokens());
        let available = mgr.available_memory();
        if required > available {
            return Ok(AdmissionDecision::Rejected(
                RejectionReason::InsufficientMemory {
                    required_bytes: required,
                    available_bytes: available,
                },
            ));
        }
    }

    // Phase 2: All stages can accept; commit the admission.
    for mgr in managers.iter_mut() {
        let decision = mgr.request_admission(req);
        debug_assert_eq!(decision, AdmissionDecision::Admitted);
    }

    Ok(AdmissionDecision::Admitted)
}

/// Broadcast an eviction across all stage cache managers.
///
/// Evicts the sequence from every stage that has it. Returns the
/// eviction event from the initiating stage.
///
/// # Arguments
///
/// * `managers` - Mutable references to all stage cache managers.
/// * `sequence_id` - Sequence to evict.
/// * `initiating_stage` - Index of the stage that initiated the eviction.
/// * `reason` - Reason for the eviction.
pub fn broadcast_eviction(
    managers: &mut [&mut PipelineCacheManager],
    sequence_id: SequenceId,
    initiating_stage: u32,
    reason: EvictionReason,
) -> Result<EvictionEvent> {
    let event = EvictionEvent {
        sequence_id,
        initiating_stage,
        reason,
    };

    for mgr in managers.iter_mut() {
        mgr.apply_eviction_broadcast(&event)?;
    }

    Ok(event)
}

/// Synchronize cache metadata from a source stage to all other stages.
///
/// # Arguments
///
/// * `managers` - Mutable references to all stage cache managers.
/// * `sync` - The metadata sync message to apply.
pub fn sync_metadata(
    managers: &mut [&mut PipelineCacheManager],
    sync: &CacheMetadataSync,
) -> Result<()> {
    for mgr in managers.iter_mut() {
        // Skip the source stage (it already has the correct state).
        if mgr.stage_index() == sync.source_stage {
            continue;
        }
        mgr.apply_metadata_sync(sync)?;
    }
    Ok(())
}

/// Check all stages for memory pressure and return the most urgent
/// preemption signal (if any).
///
/// When multiple stages are under pressure, the stage with the highest
/// memory usage fraction is selected as the source.
pub fn check_pipeline_pressure(managers: &[&PipelineCacheManager]) -> Option<PreemptionSignal> {
    managers
        .iter()
        .filter_map(|mgr| mgr.check_memory_pressure())
        .max_by(|a, b| {
            a.memory_usage_fraction
                .partial_cmp(&b.memory_usage_fraction)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

// ---------------------------------------------------------------------------
// 2D (PP x TP) cache coordination
// ---------------------------------------------------------------------------

/// Identifier for a 2D cache slot — one per `(pp_stage, tp_rank)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PpTpCoord {
    /// Pipeline stage index.
    pub stage: u32,
    /// Tensor-parallel rank within the stage.
    pub rank: u32,
}

impl PpTpCoord {
    /// Create a new 2D coordinate.
    pub fn new(stage: u32, rank: u32) -> Self {
        Self { stage, rank }
    }
}

impl fmt::Display for PpTpCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(stage={}, rank={})", self.stage, self.rank)
    }
}

/// Result of a 2D admission attempt across the full `(stage, rank)` grid.
///
/// When a sequence is submitted to the 2D mesh, every `(stage, rank)` cache
/// manager must accept. If any rank rejects, the entire submission is rolled
/// back — otherwise the stage would run an un-cached sharded forward pass and
/// silently diverge.
///
/// Used by: 2D PP × TP admission controller.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PpTpAdmissionOutcome {
    /// Every `(stage, rank)` slot admitted the sequence.
    Admitted,
    /// At least one slot rejected; nothing was committed anywhere.
    Rejected {
        /// Coordinate of the first slot that rejected.
        at: PpTpCoord,
        /// Reason reported by that slot.
        reason: RejectionReason,
    },
}

/// Check admission across a 2D `(pp_stage, tp_rank)` grid of cache managers.
///
/// `managers[(stage, rank)]` must reference the cache manager that owns the
/// KV shard for that intersection. The function performs a two-phase check:
///
/// 1. Dry-run admission on every slot (no mutation). If any slot rejects,
///    return the first rejection without touching any manager's state.
/// 2. If every slot accepts, finalize the admission on each manager.
///
/// This preserves the invariant that a 2D stage is coherent: either every
/// rank in the grid holds the KV shard for the sequence, or none does.
///
/// Used by: 2D PP × TP cache admission controller.
pub fn coordinated_2d_admission(
    managers: &mut [(PpTpCoord, &mut PipelineCacheManager)],
    req: &CacheAdmissionRequest,
) -> Result<PpTpAdmissionOutcome> {
    ensure!(!managers.is_empty(), "at least one 2D manager is required");

    // Phase 1: dry-run check (do not mutate state).
    for (coord, mgr) in managers.iter() {
        if mgr.allocations.contains_key(&req.sequence_id) {
            return Ok(PpTpAdmissionOutcome::Rejected {
                at: *coord,
                reason: RejectionReason::AlreadyCached,
            });
        }
        if mgr.allocations.len() >= mgr.config.max_sequences {
            return Ok(PpTpAdmissionOutcome::Rejected {
                at: *coord,
                reason: RejectionReason::MaxSequencesReached,
            });
        }
        let required = mgr.config.estimate_memory(req.effective_tokens());
        let available = mgr.available_memory();
        if required > available {
            return Ok(PpTpAdmissionOutcome::Rejected {
                at: *coord,
                reason: RejectionReason::InsufficientMemory {
                    required_bytes: required,
                    available_bytes: available,
                },
            });
        }
    }

    // Phase 2: commit on every slot.
    for (_, mgr) in managers.iter_mut() {
        let decision = mgr.request_admission(req);
        debug_assert_eq!(decision, AdmissionDecision::Admitted);
    }

    Ok(PpTpAdmissionOutcome::Admitted)
}

/// Broadcast an eviction across every `(stage, rank)` slot in a 2D grid.
///
/// Returns the eviction event produced by the initiating slot.
///
/// Used by: 2D PP × TP cache admission controller.
pub fn broadcast_2d_eviction(
    managers: &mut [(PpTpCoord, &mut PipelineCacheManager)],
    sequence_id: SequenceId,
    initiating: PpTpCoord,
    reason: EvictionReason,
) -> Result<EvictionEvent> {
    let event = EvictionEvent {
        sequence_id,
        initiating_stage: initiating.stage,
        reason,
    };

    for (_, mgr) in managers.iter_mut() {
        mgr.apply_eviction_broadcast(&event)?;
    }
    Ok(event)
}

/// Coordinated admission that returns an [`AdmissionDiagnostic`] on
/// rejection, instead of just the enum. Used by the server HTTP error
/// path so operators see the offending stage identity and KV occupancy.
///
/// On success the semantics are identical to [`coordinated_admission`] —
/// every manager's allocation is committed atomically. On rejection no
/// mutation is performed anywhere, mirroring the existing dry-run-then-
/// commit contract.
///
/// Used by: `/metrics` admission rejection pipeline.
pub fn coordinated_admission_with_attribution(
    managers: &mut [&mut PipelineCacheManager],
    req: &CacheAdmissionRequest,
) -> Result<std::result::Result<(), AdmissionDiagnostic>> {
    ensure!(!managers.is_empty(), "at least one manager is required");

    // Phase 1: dry-run check.
    for mgr in managers.iter() {
        let stage_index = mgr.config.stage_index;
        let budget = mgr.config.memory_budget_bytes;
        if mgr.allocations.contains_key(&req.sequence_id) {
            return Ok(Err(AdmissionDiagnostic {
                stage_index,
                reason: RejectionReason::AlreadyCached,
                used_memory_bytes: mgr.used_memory_bytes,
                budget_bytes: budget,
                active_sequences: mgr.allocations.len(),
            }));
        }
        if mgr.allocations.len() >= mgr.config.max_sequences {
            return Ok(Err(AdmissionDiagnostic {
                stage_index,
                reason: RejectionReason::MaxSequencesReached,
                used_memory_bytes: mgr.used_memory_bytes,
                budget_bytes: budget,
                active_sequences: mgr.allocations.len(),
            }));
        }
        let required = mgr.config.estimate_memory(req.effective_tokens());
        let available = mgr.available_memory();
        if required > available {
            return Ok(Err(AdmissionDiagnostic {
                stage_index,
                reason: RejectionReason::InsufficientMemory {
                    required_bytes: required,
                    available_bytes: available,
                },
                used_memory_bytes: mgr.used_memory_bytes,
                budget_bytes: budget,
                active_sequences: mgr.allocations.len(),
            }));
        }
    }

    // Phase 2: commit.
    for mgr in managers.iter_mut() {
        let decision = mgr.request_admission(req);
        debug_assert_eq!(decision, AdmissionDecision::Admitted);
    }

    Ok(Ok(()))
}

#[cfg(test)]
#[path = "cache_manager_tests.rs"]
mod tests;
