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

//! The PLE n-gram table behind a row store: `Resident` (the checkpoint's shard
//! arrays, today's path) or `External` (rows read from disk through a manifest,
//! `text_config.ple_storage`).
//!
//! External lookups deduplicate the ids, serve hits from a bounded LRU of raw
//! row records, read the misses through [`RowSource`], dequantize once and
//! restore the caller's order. Output tensors are bit-identical to the resident
//! table's. Gather-ahead reads the next prefill piece's rows on a worker thread
//! while the GPU runs the current one; a hint can only change what is cached,
//! never what a lookup returns.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{PleStorageConfig, ResolvedQwen4ExpConfig};
use super::ple::ShardedEmbedding;
use super::ple_cache::{AlwaysGrow, RowCache, RowCacheGrowth};
use super::ple_io::{PleIoMode, RowSource};
use super::ple_manifest::{PleManifest, PleManifestError, PleQuantization};
use super::ple_stats::{LAST_TABLE, PleStats, PleStatsSnapshot};

/// Environment override for the row-cache budget, in bytes.
pub const PLE_CACHE_ENV: &str = "MLXCEL_PLE_CACHE_BYTES";
/// 0.5 GB, the budget Strata's Metal engine reserves for the same rows.
pub const DEFAULT_PLE_CACHE_BYTES: u64 = 512 * 1024 * 1024;
/// Upper bound on how long a lookup waits for an in-flight gather-ahead.
const GATHER_AHEAD_WAIT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PleStoreError {
    #[error("PLE n-gram id {id} is outside the {rows}-row table")]
    IdOutOfRange { id: i64, rows: u64 },
    #[error("PLE row read failed: {0}")]
    Read(String),
}

/// Row lookup shared by the two table kinds.
pub(crate) trait PleRowStore {
    /// Dequantized rows `[ids.len(), row_width]`, in `ids` order.
    fn lookup(&self, ids: &[i64]) -> Result<UniquePtr<MlxArray>, PleStoreError>;

    /// Whether [`Self::prefetch`] does anything. The resident table says no,
    /// so its path never pays for hashing ahead.
    fn wants_prefetch(&self) -> bool {
        false
    }

    /// Start loading `ids` so a following lookup finds them. Advisory.
    fn prefetch(&self, _ids: &[i64]) {}
}

impl PleRowStore for ShardedEmbedding {
    fn lookup(&self, ids: &[i64]) -> Result<UniquePtr<MlxArray>, PleStoreError> {
        let rows = self.row_count();
        if let Some(&id) = ids.iter().find(|&&id| id < 0 || id as u64 >= rows) {
            return Err(PleStoreError::IdOutOfRange { id, rows });
        }
        if ids.is_empty() {
            return Ok(mlxcel_core::zeros(
                &[0, self.row_width()],
                mlxcel_core::dtype::BFLOAT16,
            ));
        }
        Ok(ShardedEmbedding::lookup(self, ids))
    }
}

/// The row-cache budget: env, then a row-count hint from the manifest or
/// config (`cache_rows`, upstream's knob, 0 meaning none), then the default.
pub fn resolve_cache_budget(cache_rows: u64, stride: usize) -> Result<u64, String> {
    if let Ok(value) = std::env::var(PLE_CACHE_ENV)
        && !value.is_empty()
    {
        return value
            .parse::<u64>()
            .map_err(|_| format!("{PLE_CACHE_ENV}={value:?} must be a byte count"));
    }
    Ok(if cache_rows > 0 {
        cache_rows.saturating_mul(stride as u64)
    } else {
        DEFAULT_PLE_CACHE_BYTES
    })
}

struct Shared {
    source: RowSource,
    cache: Mutex<RowCache>,
    stride: usize,
    stats: Arc<PleStats>,
    /// `true` while a gather-ahead is running.
    in_flight: (Mutex<bool>, Condvar),
}

impl Shared {
    /// Put the records for sorted, deduplicated `ids` into the cache and, for
    /// a lookup, into `out`. Returns the miss count.
    fn fetch(&self, ids: &[u64], mut out: Option<&mut [u8]>) -> Result<usize, String> {
        let stride = self.stride;
        let mut miss_ids = Vec::new();
        let mut miss_slots = Vec::new();
        {
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| "PLE row cache lock poisoned")?;
            let mut probe = vec![0u8; stride];
            for (index, &id) in ids.iter().enumerate() {
                let hit = match out.as_deref_mut() {
                    Some(out) => cache.get(id, &mut out[index * stride..(index + 1) * stride]),
                    None => cache.get(id, &mut probe),
                };
                if !hit {
                    miss_ids.push(id);
                    miss_slots.push(index);
                }
            }
        }
        if miss_ids.is_empty() {
            return Ok(0);
        }
        let mut buffer = vec![0u8; miss_ids.len() * stride];
        let read = self.source.read(&miss_ids, &mut buffer)?;
        self.stats.bytes_read.fetch_add(read, Ordering::Relaxed);
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| "PLE row cache lock poisoned")?;
        for (position, (record, &id)) in buffer.chunks_exact(stride).zip(&miss_ids).enumerate() {
            if let Some(out) = out.as_deref_mut() {
                let slot = miss_slots[position];
                out[slot * stride..(slot + 1) * stride].copy_from_slice(record);
            }
            cache.insert(id, record);
        }
        Ok(miss_ids.len())
    }

    fn wait_for_gather_ahead(&self) {
        let (lock, condvar) = &self.in_flight;
        let Ok(guard) = lock.lock() else { return };
        // Bounded so a stuck read degrades into a duplicate read, not a hang.
        let _ = condvar.wait_timeout_while(guard, GATHER_AHEAD_WAIT, |running| *running);
    }

    fn finish_gather_ahead(&self) {
        let (lock, condvar) = &self.in_flight;
        if let Ok(mut running) = lock.lock() {
            *running = false;
            condvar.notify_all();
        }
    }
}

/// A row file table opened from a validated manifest.
pub struct ExternalPleTable {
    shared: Arc<Shared>,
    row_width: usize,
    row_count: u64,
    quantization: PleQuantization,
    capacity_rows: u32,
    sender: Option<SyncSender<Vec<u64>>>,
    worker: Option<JoinHandle<()>>,
}

impl ExternalPleTable {
    pub fn open(
        manifest: &PleManifest,
        mode: PleIoMode,
        cache_budget_bytes: u64,
        growth: Arc<dyn RowCacheGrowth>,
    ) -> Result<Self, String> {
        let source = RowSource::open(manifest, mode)?;
        let stride = manifest.row_stride();
        let budget = if source.wants_cache() {
            cache_budget_bytes
        } else {
            0
        };
        let cache = RowCache::new(budget, stride, growth);
        let capacity_rows = cache.capacity_rows();
        let stats = Arc::new(PleStats::default());
        if let Ok(mut slot) = LAST_TABLE.lock() {
            *slot = Some((mode, Arc::clone(&stats)));
        }
        let shared = Arc::new(Shared {
            source,
            cache: Mutex::new(cache),
            stride,
            stats,
            in_flight: (Mutex::new(false), Condvar::new()),
        });
        let (sender, receiver) = sync_channel::<Vec<u64>>(1);
        let worker_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("mlxcel-ple-gather".to_string())
            .spawn(move || {
                while let Ok(ids) = receiver.recv() {
                    // Read errors are not surfaced here: the lookup repeats
                    // the read and reports them with its own context.
                    let _ = worker_shared.fetch(&ids, None);
                    worker_shared
                        .stats
                        .prefetched_rows
                        .fetch_add(ids.len() as u64, Ordering::Relaxed);
                    worker_shared.finish_gather_ahead();
                }
            })
            .map_err(|e| format!("cannot start the PLE gather-ahead thread: {e}"))?;
        Ok(Self {
            shared,
            row_width: manifest.row_width,
            row_count: manifest.row_count,
            quantization: manifest.quantization,
            capacity_rows,
            sender: Some(sender),
            worker: Some(worker),
        })
    }

    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    pub fn row_width(&self) -> usize {
        self.row_width
    }

    pub fn stats(&self) -> PleStatsSnapshot {
        self.shared.stats.snapshot()
    }

    fn dequantize(&self, records: &[u8], count: usize) -> UniquePtr<MlxArray> {
        let row_width = self.row_width;
        let stride = self.shared.stride;
        let quantization = self.quantization;
        let mut offsets = Vec::new();
        let mut within = 0;
        for &(name, _, item) in quantization.tensors() {
            let bytes = quantization.tensor_row_bytes(name, item, row_width);
            offsets.push((within, bytes, item));
            within += bytes;
        }
        let column = |index: usize| -> Vec<u8> {
            let (start, bytes, _) = offsets[index];
            let mut staged = Vec::with_capacity(bytes * count);
            for record in records.chunks_exact(stride) {
                staged.extend_from_slice(&record[start..start + bytes]);
            }
            staged
        };
        let n = count as i32;
        let groups = (row_width / quantization.group_size()) as i32;
        let weight = mlxcel_core::from_bytes(
            &column(0),
            &[n, (row_width / 8) as i32],
            mlxcel_core::dtype::UINT32,
        );
        let (scales, biases) = match quantization {
            PleQuantization::Affine4x32 => (
                mlxcel_core::from_bytes_f16(&column(1), &[n, groups], true),
                Some(mlxcel_core::from_bytes_f16(&column(2), &[n, groups], true)),
            ),
            PleQuantization::Nvfp4x16 => (
                mlxcel_core::from_bytes(&column(1), &[n, groups], mlxcel_core::dtype::UINT8),
                None,
            ),
        };
        let biases_ptr: *const MlxArray = biases
            .as_ref()
            .map_or(std::ptr::null(), |array| &**array as *const MlxArray);
        // SAFETY: `weight`, `scales` and `biases` outlive the call, and
        // `biases_ptr` is null or points at the array held in `biases`.
        unsafe {
            mlxcel_core::dequantize(
                &weight,
                &scales,
                biases_ptr,
                quantization.group_size() as i32,
                PleQuantization::BITS,
                quantization.mode(),
            )
        }
    }
}

impl Drop for ExternalPleTable {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl PleRowStore for ExternalPleTable {
    fn lookup(&self, ids: &[i64]) -> Result<UniquePtr<MlxArray>, PleStoreError> {
        let started = Instant::now();
        if let Some(&id) = ids
            .iter()
            .find(|&&id| id < 0 || id as u64 >= self.row_count)
        {
            return Err(PleStoreError::IdOutOfRange {
                id,
                rows: self.row_count,
            });
        }
        if ids.is_empty() {
            return Ok(mlxcel_core::zeros(
                &[0, self.row_width as i32],
                mlxcel_core::dtype::BFLOAT16,
            ));
        }
        self.shared.wait_for_gather_ahead();

        let mut keyed: Vec<(u64, u32)> = ids
            .iter()
            .enumerate()
            .map(|(index, &id)| (id as u64, index as u32))
            .collect();
        keyed.sort_unstable();
        let mut unique = Vec::with_capacity(keyed.len());
        let mut inverse = vec![0i32; keyed.len()];
        for &(id, index) in &keyed {
            if unique.last() != Some(&id) {
                unique.push(id);
            }
            inverse[index as usize] = (unique.len() - 1) as i32;
        }
        let mut records = vec![0u8; unique.len() * self.shared.stride];
        let misses = self
            .shared
            .fetch(&unique, Some(&mut records))
            .map_err(PleStoreError::Read)?;

        let stats = &self.shared.stats;
        stats.lookups.fetch_add(1, Ordering::Relaxed);
        stats.rows.fetch_add(ids.len() as u64, Ordering::Relaxed);
        stats.misses.fetch_add(misses as u64, Ordering::Relaxed);
        stats
            .hits
            .fetch_add((ids.len() - misses) as u64, Ordering::Relaxed);

        let values = self.dequantize(&records, unique.len());
        let order = mlxcel_core::from_slice_i32(&inverse, &[inverse.len() as i32]);
        let values = mlxcel_core::take(&values, &order, 0);
        stats
            .elapsed_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(values)
    }

    fn wants_prefetch(&self) -> bool {
        self.capacity_rows > 0
    }

    fn prefetch(&self, ids: &[i64]) {
        let Some(sender) = &self.sender else { return };
        let mut unique: Vec<u64> = ids
            .iter()
            .filter(|&&id| id >= 0 && (id as u64) < self.row_count)
            .map(|&id| id as u64)
            .collect();
        unique.sort_unstable();
        unique.dedup();
        // More rows than the cache holds would evict their own earlier reads.
        unique.truncate(self.capacity_rows as usize);
        if unique.is_empty() {
            return;
        }
        self.shared.wait_for_gather_ahead();
        let (lock, _) = &self.shared.in_flight;
        match lock.lock() {
            Ok(mut running) => *running = true,
            Err(_) => return,
        }
        if sender.try_send(unique).is_err() {
            self.shared.finish_gather_ahead();
        }
    }
}

/// Resolve `ple_storage.manifest` against the model directory.
pub fn resolve_manifest_path(
    storage: &PleStorageConfig,
    model_dir: Option<&Path>,
) -> Result<PathBuf, PleManifestError> {
    let path = Path::new(&storage.manifest);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    model_dir
        .map(|dir| dir.join(path))
        .ok_or_else(|| PleManifestError {
            field: "ple_storage.manifest".to_string(),
            reason: format!(
                "relative path {:?} needs the model directory",
                storage.manifest
            ),
        })
}

/// Open the external table a config declares, from the environment's I/O and
/// cache settings. Called once per PLE layer at load.
pub fn open_external_table(
    storage: &PleStorageConfig,
    config: &ResolvedQwen4ExpConfig,
) -> Result<ExternalPleTable, String> {
    let path =
        resolve_manifest_path(storage, config.model_dir.as_deref()).map_err(|e| e.to_string())?;
    let manifest = PleManifest::load(&path, storage.cache_rows).map_err(|e| e.to_string())?;
    let mode = PleIoMode::from_env()?;
    let budget = resolve_cache_budget(manifest.cache_rows, manifest.row_stride())?;
    ExternalPleTable::open(&manifest, mode, budget, Arc::new(AlwaysGrow))
}

#[cfg(test)]
#[path = "qwen4_exp_ple_storage_tests.rs"]
mod tests;
