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

//! Layer-parallel concurrent KV cache transfer.
//!
//! Sends multiple layers simultaneously over the transport layer,
//! utilizing available network bandwidth more efficiently than
//! sequential transfer.
//!
//! # Concurrency Model
//!
//! Uses a Tokio semaphore to bound the number of in-flight layer
//! transfers. Each layer is sent as an independent task, and results
//! are collected after all complete.
//!
//! ```text
//! Concurrency = 4:
//!   Task 0: [Layer 0]──────────>
//!   Task 1: [Layer 1]──────>
//!   Task 2: [Layer 2]──────────────>
//!   Task 3: [Layer 3]──────────>
//!   Task 0: [Layer 4]──────>        (reused after Layer 0 completes)
//! ```
//!
//! Used by: disaggregated serving pipeline (prefill -> decode handoff)

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use bytes::Bytes;
use tokio::sync::Semaphore;

use super::streamed::prepare_layer_payload;
use super::{
    CacheQuantizationLevel, LayerTransferHeader, LayerTransferResult, TransferConfig,
    TransferResult, TransferStrategy,
};
use crate::distributed::kv_cache_serde::types::SerializableCacheEntry;
use crate::distributed::transport::{Transport, TransportMessage};

/// Parallel multi-layer KV cache transfer.
///
/// Sends cache entries for multiple layers concurrently, bounded by
/// a configurable concurrency limit via a Tokio semaphore.
pub struct ParallelLayerTransfer {
    config: TransferConfig,
    /// Target peer address.
    peer: String,
    /// Sequence ID for this transfer.
    sequence_id: u64,
}

impl ParallelLayerTransfer {
    /// Create a new parallel transfer.
    pub fn new(config: TransferConfig, peer: String, sequence_id: u64) -> Self {
        Self {
            config,
            peer,
            sequence_id,
        }
    }

    /// Transfer all layers concurrently with bounded parallelism.
    ///
    /// Spawns up to `config.concurrency` tasks, each sending one layer.
    /// Returns when all layers have been transferred.
    pub async fn transfer_all(
        &self,
        transport: Arc<dyn Transport>,
        entries: &[SerializableCacheEntry],
    ) -> Result<TransferResult> {
        let transfer_start = Instant::now();
        let total_layers = entries.len();
        let concurrency = self.config.concurrency.max(1);
        let semaphore = Arc::new(Semaphore::new(concurrency));

        // Prepare all payloads synchronously (quantization is CPU-bound).
        let prepared: Vec<PreparedLayer> = entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let (wire_data, original_bytes, num_elements) =
                    prepare_layer_payload(entry, self.config.quantization, self.config.compress)?;
                Ok(PreparedLayer {
                    layer_index: i,
                    wire_data,
                    original_bytes,
                    num_elements,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Spawn concurrent send tasks.
        let mut handles = Vec::with_capacity(total_layers);

        for prep in prepared {
            let sem = semaphore.clone();
            let transport = transport.clone();
            let peer = self.peer.clone();
            let sequence_id = self.sequence_id;
            let quantization = self.config.quantization;

            let handle = tokio::spawn(async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|_| anyhow::anyhow!("semaphore closed"))?;

                let layer_start = Instant::now();

                let header = LayerTransferHeader {
                    sequence_id,
                    layer_index: prep.layer_index,
                    total_layers,
                    quantized: quantization != CacheQuantizationLevel::None,
                    quantization_level: quantization,
                    original_num_elements: prep.num_elements,
                };

                let header_json =
                    serde_json::to_vec(&header).context("serializing layer header")?;

                // Send header.
                transport
                    .send(
                        &peer,
                        TransportMessage::Control {
                            operation: "kv_cache_layer_parallel".to_string(),
                            payload: Bytes::from(header_json),
                        },
                    )
                    .await
                    .with_context(|| format!("sending layer {} header", prep.layer_index))?;

                // Send data.
                let wire_bytes = prep.wire_data.len();
                transport
                    .send_stream(&peer, Bytes::from(prep.wire_data))
                    .await
                    .with_context(|| format!("sending layer {} data", prep.layer_index))?;

                Ok::<LayerTransferResult, anyhow::Error>(LayerTransferResult {
                    layer_index: prep.layer_index,
                    wire_bytes,
                    original_bytes: prep.original_bytes,
                    duration: layer_start.elapsed(),
                })
            });

            handles.push(handle);
        }

        // Collect results.
        let mut layer_results = Vec::with_capacity(total_layers);
        let mut total_wire_bytes = 0usize;
        let mut total_original_bytes = 0usize;

        for handle in handles {
            let result = handle
                .await
                .map_err(|e| anyhow::anyhow!("layer transfer task panicked: {e}"))??;
            total_wire_bytes += result.wire_bytes;
            total_original_bytes += result.original_bytes;
            layer_results.push(result);
        }

        // Sort results by layer index for deterministic ordering.
        layer_results.sort_by_key(|r| r.layer_index);

        Ok(TransferResult {
            strategy: TransferStrategy::LayerParallel,
            quantization: self.config.quantization,
            layer_results,
            total_duration: transfer_start.elapsed(),
            total_wire_bytes,
            total_original_bytes,
        })
    }

    /// Return the configured concurrency level.
    pub fn concurrency(&self) -> usize {
        self.config.concurrency
    }
}

/// Pre-prepared layer data ready for async sending.
struct PreparedLayer {
    layer_index: usize,
    wire_data: Vec<u8>,
    original_bytes: usize,
    num_elements: usize,
}

/// Estimate the optimal concurrency level based on cache size and bandwidth.
///
/// Heuristic: aim for each parallel task to transfer at least 1 MB to
/// amortize per-message overhead. Cap at the number of available layers.
pub fn estimate_concurrency(num_layers: usize, total_bytes: usize, bandwidth_bps: f64) -> usize {
    if num_layers == 0 || bandwidth_bps <= 0.0 {
        return 1;
    }

    let bytes_per_layer = total_bytes / num_layers;
    let min_bytes_per_task = 1024 * 1024; // 1 MB minimum per task

    // Don't use more concurrency than needed to saturate the link.
    let max_useful = if bytes_per_layer >= min_bytes_per_task {
        num_layers
    } else {
        // Group small layers together.
        (total_bytes / min_bytes_per_task).max(1)
    };

    // Reasonable upper bound to avoid overwhelming the transport.
    max_useful.min(16).min(num_layers)
}

#[cfg(test)]
#[path = "parallel_tests.rs"]
mod tests;
