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

//! SSD expert streaming for Qwen3.8-Flash-Next (`qwen4_exp`, issue #2882).
//!
//! Routed experts live in a resident arena of slots, one `[S, out, in]` tensor
//! per plane per layer, filled from the checkpoint as tokens route to them. A
//! slot holds the expert's checkpoint bytes unchanged, so the kernels read the
//! same numbers at another address and results stay exact.
//!
//! Each layer synchronises with the host once, between the router and the
//! expert matmuls (see `docs/supported-models.md` for what that costs).

#[path = "qwen4_exp_stream_arena.rs"]
pub(crate) mod arena;
#[path = "qwen4_exp_stream_slots.rs"]
pub(crate) mod slots;
#[path = "qwen4_exp_stream_source.rs"]
pub mod source;

#[cfg(test)]
#[path = "qwen4_exp_stream_tests.rs"]
mod tests;

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use mlxcel_core::host_memory::{AvailabilityGate, HostMemoryStatus, host_memory_status};

use source::ExpertSource;

/// `1` / `on`, `0` / `off`, or `auto` (the default).
pub const EXPERT_STREAM_ENV: &str = "MLXCEL_EXPERT_STREAM";
/// Arena size in bytes; forces a size instead of deriving it from free memory.
pub const EXPERT_ARENA_ENV: &str = "MLXCEL_EXPERT_ARENA_BYTES";

/// Streams automatically when routed experts exceed this share of Metal's
/// working set.
const AUTO_SHARE: f64 = 0.60;
/// Memory left free for the KV cache and transients when sizing the arena.
const ARENA_HEADROOM_BYTES: u64 = 6 << 30;
/// Below this many slots a layer could not hold one prefill sort group of
/// 64 routed experts, the size at which the sorted kernels take over.
pub(crate) const MIN_SLOTS: usize = 64;
/// Slot counts are multiples of this, so every plane's buffer is a whole
/// number of 16 KiB pages, which Metal requires to wrap a buffer without
/// copying it.
const SLOT_GRANULARITY: usize = 16;
/// Below this much memory a draft layer does more as expert cache.
const MTP_MEMORY_FLOOR_BYTES: u64 = 96 << 30;

/// What `MLXCEL_EXPERT_STREAM` asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamRequest {
    Auto,
    On,
    Off,
}

impl StreamRequest {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("auto") => Ok(Self::Auto),
            Some("1") | Some("on") | Some("true") => Ok(Self::On),
            Some("0") | Some("off") | Some("false") => Ok(Self::Off),
            Some(other) => Err(format!(
                "{EXPERT_STREAM_ENV}: unknown value {other:?}; expected 1, 0, or auto"
            )),
        }
    }
}

/// The decision rule: explicit requests win; `auto` streams only when the
/// experts would take more than [`AUTO_SHARE`] of Metal's working set.
pub fn should_stream(request: StreamRequest, expert_bytes: u64, working_set_bytes: u64) -> bool {
    match request {
        StreamRequest::On => true,
        StreamRequest::Off => false,
        StreamRequest::Auto => {
            working_set_bytes > 0 && expert_bytes as f64 > working_set_bytes as f64 * AUTO_SHARE
        }
    }
}

/// Slots per layer for an arena of `arena_bytes` over `layers` layers of
/// `layer_expert_bytes` per expert: rounded down to the granularity, and
/// clamped to `[MIN_SLOTS, num_experts]`.
pub fn slots_per_layer(
    arena_bytes: u64,
    layers: usize,
    layer_expert_bytes: u64,
    num_experts: usize,
) -> usize {
    let per_layer = arena_bytes / (layers.max(1) as u64);
    let slots = (per_layer / layer_expert_bytes.max(1)) as usize;
    let rounded = slots / SLOT_GRANULARITY * SLOT_GRANULARITY;
    rounded.clamp(MIN_SLOTS, num_experts)
}

/// Arena size from memory macOS can give now, not installed RAM: what is
/// usable minus the resident non-expert weights and a KV headroom, capped at
/// the experts themselves.
pub fn arena_bytes_for(
    status: Option<&HostMemoryStatus>,
    non_expert_bytes: u64,
    expert_bytes: u64,
) -> u64 {
    match status {
        Some(status) => status
            .usable_bytes()
            .saturating_sub(non_expert_bytes)
            .saturating_sub(ARENA_HEADROOM_BYTES)
            .min(expert_bytes),
        None => expert_bytes,
    }
}

/// Whether a drafter is likely to cost more than it returns: streaming on a
/// machine under 96 GB, where its memory would do more as expert cache.
pub fn mtp_discouraged(total_ram_bytes: u64, streaming: bool) -> bool {
    streaming && total_ram_bytes < MTP_MEMORY_FLOOR_BYTES
}

/// Arena counters.
#[derive(Debug, Default)]
pub struct StreamStats {
    pub(crate) windows: AtomicU64,
    pub(crate) hits: AtomicU64,
    pub(crate) misses: AtomicU64,
    pub(crate) bytes_read: AtomicU64,
    pub(crate) io_ns: AtomicU64,
    pub(crate) sync_ns: AtomicU64,
}

/// A point-in-time copy of [`StreamStats`], with the arena's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamStatsSnapshot {
    pub arena_bytes: u64,
    pub slots: u64,
    pub windows: u64,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub io_ns: u64,
    pub sync_ns: u64,
}

impl StreamStatsSnapshot {
    /// Share of routed experts found resident, in percent.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 * 100.0 / total as f64
        }
    }
}

/// The opened checkpoint plus the arena decision, shared by every layer.
#[derive(Debug)]
pub struct ExpertStream {
    pub(crate) source: ExpertSource,
    /// Slots of each layer.
    pub(crate) slots: Vec<usize>,
    pub(crate) stats: StreamStats,
}

/// The most recently loaded streaming model. Weak, so a model that is dropped
/// stops being reported and releases its checkpoint files.
static ACTIVE: Mutex<Weak<ExpertStream>> = Mutex::new(Weak::new());
/// Final counters of the last stream dropped in this process.
static LAST_SNAPSHOT: Mutex<Option<StreamStatsSnapshot>> = Mutex::new(None);

impl ExpertStream {
    /// Decide whether this checkpoint streams and, if it does, size the arena.
    /// `Ok(None)` is the resident path.
    pub fn resolve(
        model_dir: &Path,
        layers: usize,
        num_experts: usize,
        resident_bytes: u64,
    ) -> Result<Option<Arc<Self>>, String> {
        let request = StreamRequest::parse(std::env::var(EXPERT_STREAM_ENV).ok().as_deref())?;
        if request == StreamRequest::Off {
            return Ok(None);
        }
        if !cfg!(target_os = "macos") {
            return match request {
                StreamRequest::On => Err(format!(
                    "{EXPERT_STREAM_ENV}=1: expert streaming is implemented for Metal only"
                )),
                _ => Ok(None),
            };
        }
        if load_surgery_active() {
            // A slot is filled from the checkpoint file, so a transform of the
            // experts would apply to everything but them.
            return match request {
                StreamRequest::On => Err(format!(
                    "{EXPERT_STREAM_ENV}=1: expert streaming reads the checkpoint bytes and \
                     cannot be combined with --surgery"
                )),
                _ => Ok(None),
            };
        }
        let source = match (ExpertSource::open(model_dir, layers, num_experts), request) {
            (Ok(source), _) => source,
            (Err(error), StreamRequest::On) => return Err(error),
            // A layout auto cannot stream is simply loaded the resident way.
            (Err(_), _) => return Ok(None),
        };
        let expert_bytes = source.total_expert_bytes();
        let working_set = mlxcel_core::get_wired_limit() as u64;
        if !should_stream(request, expert_bytes, working_set) {
            return Ok(None);
        }

        let status = host_memory_status();
        let non_expert = resident_bytes.saturating_sub(expert_bytes);
        let arena = match std::env::var(EXPERT_ARENA_ENV) {
            Ok(raw) => raw.trim().parse::<u64>().map_err(|e| {
                format!("{EXPERT_ARENA_ENV}: expected a byte count, got {raw:?}: {e}")
            })?,
            Err(_) => arena_bytes_for(status.as_ref(), non_expert, expert_bytes),
        };
        let slots = Self::size_layers(&source, arena)?;
        let stream = Arc::new(Self {
            source,
            slots,
            stats: StreamStats::default(),
        });
        let snapshot = stream.snapshot();
        eprintln!(
            "Expert streaming on: arena {:.2} GB ({} slots across {layers} layers) for {:.2} GB of experts{}.",
            snapshot.arena_bytes as f64 / (1u64 << 30) as f64,
            snapshot.slots,
            expert_bytes as f64 / (1u64 << 30) as f64,
            if stream.source.uses_contiguous_file() {
                ", reading the contiguous file"
            } else {
                ""
            },
        );
        *ACTIVE.lock().map_err(|_| "expert stream state poisoned")? = Arc::downgrade(&stream);
        Ok(Some(stream))
    }

    /// Give each layer its slots, asking the availability gate before each
    /// layer so a load that macOS cannot back shrinks instead of swapping.
    fn size_layers(source: &ExpertSource, arena_bytes: u64) -> Result<Vec<usize>, String> {
        let layers = source.layers.len();
        let gate = AvailabilityGate::new();
        let mut slots = Vec::with_capacity(layers);
        for layer in 0..layers {
            let expert = source.expert_bytes(layer);
            let wanted = slots_per_layer(arena_bytes, layers, expert, source.num_experts);
            let granted = if gate.allows_step(wanted as u64 * expert) {
                wanted
            } else if gate.allows_step(MIN_SLOTS as u64 * expert) {
                eprintln!(
                    "Expert arena: layer {layer} reduced to {MIN_SLOTS} slots, memory is short."
                );
                MIN_SLOTS
            } else {
                return Err(format!(
                    "expert streaming: macOS cannot back even {MIN_SLOTS} slots of layer {layer}; \
                     free memory or set {EXPERT_ARENA_ENV} and retry"
                ));
            };
            slots.push(granted);
        }
        Ok(slots)
    }

    pub fn snapshot(&self) -> StreamStatsSnapshot {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        StreamStatsSnapshot {
            arena_bytes: self
                .slots
                .iter()
                .enumerate()
                .map(|(layer, slots)| *slots as u64 * self.source.expert_bytes(layer))
                .sum(),
            slots: self.slots.iter().map(|s| *s as u64).sum(),
            windows: load(&self.stats.windows),
            hits: load(&self.stats.hits),
            misses: load(&self.stats.misses),
            bytes_read: load(&self.stats.bytes_read),
            io_ns: load(&self.stats.io_ns),
            sync_ns: load(&self.stats.sync_ns),
        }
    }
}

/// Whether a `--surgery` pipeline will rewrite this load's weights.
fn load_surgery_active() -> bool {
    #[cfg(feature = "surgery")]
    {
        crate::surgery::snapshot_active_pipeline().is_some()
    }
    #[cfg(not(feature = "surgery"))]
    {
        false
    }
}

/// The streaming counters of the model loaded in this process, if it streams.
pub fn active_stream_stats() -> Option<StreamStatsSnapshot> {
    ACTIVE.lock().ok()?.upgrade().map(|s| s.snapshot())
}

impl Drop for ExpertStream {
    fn drop(&mut self) {
        if let Ok(mut last) = LAST_SNAPSHOT.lock() {
            *last = Some(self.snapshot());
        }
    }
}

/// One stats line for the CLI, or `None` when nothing streamed in this
/// process. The CLI prints it after the model is gone, so a dropped stream's
/// final counters are used when no stream is live.
pub fn expert_stream_stats_line() -> Option<String> {
    let s = active_stream_stats().or_else(|| *LAST_SNAPSHOT.lock().ok()?)?;
    Some(format!(
        "[Expert arena: {:.2} GB, {} slots, {:.1}% hit, {} misses, {:.1} MiB read, {:.1} ms I/O, {:.1} ms router sync, {} windows]",
        s.arena_bytes as f64 / (1u64 << 30) as f64,
        s.slots,
        s.hit_rate(),
        s.misses,
        s.bytes_read as f64 / (1024.0 * 1024.0),
        s.io_ns as f64 / 1e6,
        s.sync_ns as f64 / 1e6,
        s.windows,
    ))
}

/// Bytes of `model_dir`'s weights that would be resident: the shard sizes, or
/// the external-table footprint when the n-gram table is on disk (#2881).
pub fn resident_weight_bytes(model_dir: &Path) -> u64 {
    if let Some(external) = super::ple_footprint::external_ple_footprint(model_dir) {
        return external.resident_bytes();
    }
    std::fs::read_dir(model_dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                .filter_map(|e| e.metadata().ok())
                .map(|meta| meta.len())
                .sum()
        })
        .unwrap_or(0)
}

/// What the memory preflight should count as resident for `model_dir` when
/// it will stream: the non-expert weights plus the arena. `None` keeps the
/// usual whole-checkpoint figure.
pub fn preflight_resident_bytes(model_dir: &Path) -> Option<u64> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json")).ok()?).ok()?;
    if config["model_type"].as_str() != Some("qwen4_exp") {
        return None;
    }
    let text = config.get("text_config").unwrap_or(&config);
    let layers = text["num_hidden_layers"].as_u64()? as usize;
    let num_experts = text["num_experts"].as_u64()? as usize;
    let request = StreamRequest::parse(std::env::var(EXPERT_STREAM_ENV).ok().as_deref()).ok()?;
    if load_surgery_active() {
        return None;
    }
    let source = ExpertSource::open(model_dir, layers, num_experts).ok()?;
    let expert_bytes = source.total_expert_bytes();
    if !should_stream(request, expert_bytes, mlxcel_core::get_wired_limit() as u64) {
        return None;
    }
    let resident = resident_weight_bytes(model_dir);
    let non_expert = resident.saturating_sub(expert_bytes);
    let arena = match std::env::var(EXPERT_ARENA_ENV) {
        Ok(raw) => raw.trim().parse().ok()?,
        Err(_) => arena_bytes_for(host_memory_status().as_ref(), non_expert, expert_bytes),
    };
    // The same clamped slot counts the loader allocates: at least 64 per layer
    // however small the requested arena is.
    let allocated: u64 = (0..layers)
        .map(|layer| {
            let expert = source.expert_bytes(layer);
            slots_per_layer(arena, layers, expert, num_experts) as u64 * expert
        })
        .sum();
    Some(non_expert.saturating_add(allocated))
}
