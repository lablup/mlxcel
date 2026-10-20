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

//! When checkpoint weights are read and uploaded (#1564).
//!
//! MLX's `load_safetensors` returns unevaluated `Load` arrays, so without a
//! policy a checkpoint is read off the page cache and copied to the device
//! inside the first `eval` that needs it: the first token. That made
//! `Model loaded in 0.98s (resident: 0.00 GB)` followed by a 15 s first token
//! on a 27B checkpoint, and charged a server's weight upload to its first
//! request. Whether this happened was a loader property: the Gemma 4 loader
//! evaluated its weights, every loader built on [`crate::weights`] did not.
//!
//! The decision is now made once, here, for every loader. The weights loaders
//! ([`crate::weights::load_weights_from_dir`] and the filtered and
//! single-file variants built on [`crate::weights::load_safetensors_filtered`])
//! register each tensor they hand out with the [`MaterializationScope`] open on
//! the loading thread. The model-load entry points open a scope around family
//! construction and [`MaterializationScope::finish`] it after the model is
//! built, so every family inherits the policy without deciding for itself.
//! Loads outside a scope (pipeline stages, embedding loaders, tests) register
//! nothing and stay lazy, exactly as before.
//!
//! Under the default [`MaterializationPolicy::Eager`], `finish` evaluates and
//! detaches, through [`crate::eval_all`] and [`crate::detach_all`], every
//! registered tensor except:
//!
//! - **Modality-tower tensors** ([`is_multimodal_tower_tensor`]): a vision or
//!   audio tower is only evaluated when a request carries that modality, so
//!   forcing it resident would charge a text-only run for memory it never
//!   uses. Those tensors stay lazy and load on the first image or clip.
//! - **Unreferenced tensors**: a tensor the loader dropped (an MTP head the
//!   runtime does not use, a tied `lm_head`, a sanitized-away duplicate) is
//!   held by nothing but the scope, so reading it would be wasted I/O.
//!
//! A tensor a loader consumed into a lazy repack (expert stacking, a norm
//! shift, a dtype conversion) is still referenced by the repack's input list,
//! so its source is materialized here and the repack runs at the first token,
//! where MLX frees each source as its repacked result is produced. The peak
//! therefore stays near one copy of the weights; the measurements are in
//! `docs/benchmark_results/weight-materialization-gb10-2026-10-20.md`.
//!
//! `MLXCEL_WEIGHT_MATERIALIZATION=lazy` restores the previous behaviour.

use std::cell::RefCell;
use std::time::{Duration, Instant};

use cxx::UniquePtr;

use crate::{MlxArray, ffi};

/// Environment variable that selects the [`MaterializationPolicy`].
pub const POLICY_ENV: &str = "MLXCEL_WEIGHT_MATERIALIZATION";

/// When checkpoint tensors registered with a scope are materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum MaterializationPolicy {
    /// At the end of model load, so `Model loaded` means resident and the
    /// first token pays no weight I/O.
    #[default]
    Eager,
    /// On first use, inside the first evaluation that reaches each tensor.
    Lazy,
}

impl MaterializationPolicy {
    /// Parse a policy value. `None` and the empty string select the default.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.map(str::trim) {
            None | Some("") => Ok(Self::default()),
            Some(v) if v.eq_ignore_ascii_case("eager") => Ok(Self::Eager),
            Some(v) if v.eq_ignore_ascii_case("lazy") => Ok(Self::Lazy),
            Some(v) => Err(format!(
                "{POLICY_ENV}={v} is not a weight materialization policy; expected eager or lazy"
            )),
        }
    }

    /// The policy from [`POLICY_ENV`]; an invalid value warns and keeps the
    /// default rather than failing the load.
    #[must_use]
    pub fn from_env() -> Self {
        let value = std::env::var(POLICY_ENV).ok();
        Self::parse(value.as_deref()).unwrap_or_else(|error| {
            tracing::warn!("{error}; using eager");
            Self::default()
        })
    }

    /// The lowercase name used in logs and in the `[TTFT]` line.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Lazy => "lazy",
        }
    }
}

/// Leading name components that mark a vision or audio tower, or the
/// projector that only feeds one into the language model.
const TOWER_COMPONENTS: &[&str] = &[
    "vision_tower",
    "visual",
    "vision_model",
    "vision_backbone",
    "vision_encoder",
    "vision_embedder",
    "embed_vision",
    "multi_modal_projector",
    "mm_projector",
    "vit_large_projector",
    "audio_tower",
    "audio",
    "audio_model",
    "audio_encoder",
    "audio_embed",
    "embed_audio",
    "sound_encoder",
    "sound_projection",
    "sound_feature_extractor",
];

/// Whether a raw checkpoint tensor name belongs to a modality tower that a
/// text-only run never evaluates.
///
/// Matches a known tower name in any of the first three dot-separated
/// components, which covers the bare (`vision_tower.blocks.0...`), wrapped
/// (`model.visual.blocks.0...`) and doubly wrapped
/// (`model.encoder.vision_tower...`) layouts the loaders accept. The rule is
/// by modality, not by family. A tower it misses is materialized eagerly,
/// which costs memory but never changes a result.
#[must_use]
pub fn is_multimodal_tower_tensor(name: &str) -> bool {
    name.split('.')
        .take(3)
        .any(|component| TOWER_COMPONENTS.contains(&component))
}

/// The CUDA weight-upload counters of `Load::eval_gpu`, or a difference of
/// two snapshots. All zero on backends without them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct WeightTransferStats {
    /// `Load` primitives evaluated on a CUDA stream.
    pub tensors: u64,
    /// Their bytes.
    pub bytes: u64,
    /// Uploads that went through the pinned staging slots.
    pub pinned_tensors: u64,
    /// Uploads that went through a pageable `malloc` buffer.
    pub pageable_tensors: u64,
    /// `cudaMemcpyAsync` calls issued for them.
    pub memcpy_calls: u64,
    /// Host staging leg: the read into the staging buffer (plus `malloc` on
    /// the pageable path). Invisible to a CUDA profiler.
    pub read_ns: u64,
    /// Host time blocked on a staging slot whose copy had not finished.
    pub wait_ns: u64,
    /// Wall clock inside `Load::eval_gpu`.
    pub total_ns: u64,
}

impl WeightTransferStats {
    /// The process-wide counters, or `None` on a build without CUDA.
    #[must_use]
    pub fn snapshot() -> Option<Self> {
        let mut raw = [0u64; 8];
        ffi::weight_load_transfer_stats(&mut raw).then_some(Self {
            tensors: raw[0],
            bytes: raw[1],
            pinned_tensors: raw[2],
            pageable_tensors: raw[3],
            memcpy_calls: raw[4],
            read_ns: raw[5],
            wait_ns: raw[6],
            total_ns: raw[7],
        })
    }

    /// What happened between `earlier` and `self`.
    #[must_use]
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            tensors: self.tensors.saturating_sub(earlier.tensors),
            bytes: self.bytes.saturating_sub(earlier.bytes),
            pinned_tensors: self.pinned_tensors.saturating_sub(earlier.pinned_tensors),
            pageable_tensors: self
                .pageable_tensors
                .saturating_sub(earlier.pageable_tensors),
            memcpy_calls: self.memcpy_calls.saturating_sub(earlier.memcpy_calls),
            read_ns: self.read_ns.saturating_sub(earlier.read_ns),
            wait_ns: self.wait_ns.saturating_sub(earlier.wait_ns),
            total_ns: self.total_ns.saturating_sub(earlier.total_ns),
        }
    }

    /// `weights=N/X.XXGB staging=..ms h2d_wait=..ms upload=..ms pinned=N pageable=N memcpy=N`.
    #[must_use]
    pub fn format_fields(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        format!(
            "weights={}/{:.2}GB staging={:.2}ms h2d_wait={:.2}ms upload={:.2}ms pinned={} pageable={} memcpy={}",
            self.tensors,
            self.bytes as f64 / 1e9,
            ms(self.read_ns),
            ms(self.wait_ns),
            ms(self.total_ns),
            self.pinned_tensors,
            self.pageable_tensors,
            self.memcpy_calls,
        )
    }
}

/// What one [`MaterializationScope::finish`] did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MaterializationReport {
    /// The policy that was applied.
    pub policy: MaterializationPolicy,
    /// Tensors the weights loaders handed out inside the scope.
    pub registered: usize,
    /// Tensors evaluated and detached.
    pub materialized: usize,
    /// Their bytes.
    pub materialized_bytes: u64,
    /// Tensors left lazy because they belong to a modality tower.
    pub skipped_multimodal: usize,
    /// Tensors left alone because nothing but the scope held them.
    pub skipped_unreferenced: usize,
    /// Wall clock of the evaluation.
    pub elapsed: Duration,
    /// Upload counters over the evaluation (`None` without CUDA).
    pub transfer: Option<WeightTransferStats>,
    /// MLX active and peak bytes right after the evaluation.
    pub active_bytes: u64,
    /// See `active_bytes`.
    pub peak_bytes: u64,
}

impl MaterializationReport {
    /// The `[TTFT] load` line `MLXCEL_PROFILE_TTFT` prints at model load.
    #[must_use]
    pub fn format_line(&self) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        let mut line = format!(
            "[TTFT] load policy={} materialized={}/{:.2}GB skipped_multimodal={} skipped_unreferenced={} eval={:.2}ms",
            self.policy.as_str(),
            self.materialized,
            gb(self.materialized_bytes),
            self.skipped_multimodal,
            self.skipped_unreferenced,
            self.elapsed.as_secs_f64() * 1e3,
        );
        if let Some(transfer) = &self.transfer {
            line.push(' ');
            line.push_str(&transfer.format_fields());
        }
        line.push_str(&format!(
            " | active={:.2}GB peak={:.2}GB",
            gb(self.active_bytes),
            gb(self.peak_bytes)
        ));
        line
    }

    /// Log the report, and print [`Self::format_line`] to stderr when
    /// `MLXCEL_PROFILE_TTFT` is set.
    pub fn emit(&self) {
        tracing::info!(
            policy = self.policy.as_str(),
            registered = self.registered,
            materialized = self.materialized,
            materialized_bytes = self.materialized_bytes,
            skipped_multimodal = self.skipped_multimodal,
            skipped_unreferenced = self.skipped_unreferenced,
            eval_seconds = self.elapsed.as_secs_f64(),
            active_bytes = self.active_bytes,
            peak_bytes = self.peak_bytes,
            "Weight materialization at model load",
        );
        if crate::generate::ttft_profile_enabled() {
            eprintln!("{}", self.format_line());
        }
    }
}

/// Tensors registered with the open scope, by checkpoint name.
type Registry = Vec<(String, UniquePtr<MlxArray>)>;

thread_local! {
    static ACTIVE: RefCell<Option<Registry>> = const { RefCell::new(None) };
}

/// Record a checkpoint tensor with the scope open on this thread, if any.
///
/// Holds a second handle to the same node (no copy, no evaluation).
pub(crate) fn register(name: &str, array: &MlxArray) {
    ACTIVE.with(|active| {
        if let Some(entries) = active.borrow_mut().as_mut() {
            entries.push((name.to_string(), ffi::array_handle_clone(array)));
        }
    });
}

/// A model load during which the weights loaders register their tensors.
///
/// Nested scopes are inert: the outermost one owns the registry, so a load
/// entry point that delegates to another does not materialize twice. Dropping
/// a scope without finishing it (a failed load) discards the registry without
/// evaluating anything.
#[must_use = "a scope materializes nothing until it is finished"]
#[derive(Debug)]
pub struct MaterializationScope {
    owner: bool,
    policy: MaterializationPolicy,
}

impl MaterializationScope {
    /// Open a scope with the policy from the environment.
    pub fn begin() -> Self {
        Self::begin_with(MaterializationPolicy::from_env())
    }

    /// Open a scope with an explicit policy.
    pub fn begin_with(policy: MaterializationPolicy) -> Self {
        let owner = ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if active.is_some() {
                false
            } else {
                *active = Some(Vec::new());
                true
            }
        });
        Self { owner, policy }
    }

    /// Apply the policy to every tensor registered since [`Self::begin`].
    ///
    /// Returns `None` for a nested scope. Call it after the model is built and
    /// the loader's own weight maps are gone, so the reference counts say what
    /// the model kept.
    pub fn finish(mut self) -> Option<MaterializationReport> {
        if !self.owner {
            return None;
        }
        self.owner = false;
        let entries = ACTIVE
            .with(|active| active.borrow_mut().take())
            .unwrap_or_default();
        Some(materialize(self.policy, entries))
    }
}

impl Drop for MaterializationScope {
    fn drop(&mut self) {
        if self.owner {
            ACTIVE.with(|active| active.borrow_mut().take());
        }
    }
}

fn materialize(policy: MaterializationPolicy, entries: Registry) -> MaterializationReport {
    let mut report = MaterializationReport {
        policy,
        registered: entries.len(),
        ..MaterializationReport::default()
    };
    if policy == MaterializationPolicy::Lazy {
        return report;
    }
    let mut targets: Vec<*const MlxArray> = Vec::with_capacity(entries.len());
    for (name, array) in &entries {
        let Some(array) = array.as_ref() else {
            continue;
        };
        if is_multimodal_tower_tensor(name) {
            report.skipped_multimodal += 1;
        } else if ffi::array_use_count(array) <= 1 {
            report.skipped_unreferenced += 1;
        } else {
            report.materialized_bytes += ffi::array_nbytes(array) as u64;
            targets.push(array as *const MlxArray);
        }
    }
    report.materialized = targets.len();
    let before = WeightTransferStats::snapshot();
    let start = Instant::now();
    if !targets.is_empty() {
        // SAFETY: every pointer comes from a live `UniquePtr` in `entries`,
        // which outlives both calls and is not mutated while they run.
        unsafe {
            crate::eval_all(&targets);
            crate::detach_all(&targets);
        }
    }
    report.elapsed = start.elapsed();
    report.transfer = before
        .zip(WeightTransferStats::snapshot())
        .map(|(before, after)| after.since(&before));
    let snapshot = crate::memory::snapshot();
    report.active_bytes = snapshot.active_bytes;
    report.peak_bytes = snapshot.peak_bytes;
    drop(entries);
    report
}
