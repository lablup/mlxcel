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

//! Streamed layer-by-layer KV cache transfer.
//!
//! Sends each layer's cache as soon as it finishes prefill computation,
//! overlapping transfer with the remaining layers' compute. This reduces
//! end-to-end TTFT by pipelining network I/O with GPU work.
//!
//! # Flow
//!
//! ```text
//! Prefill (GPU)   : [Layer 0][Layer 1][Layer 2][Layer 3]...
//! Transfer (Net)  :          [Send 0 ][Send 1 ][Send 2 ]...
//!                   ← overlap saves time →
//! ```
//!
//! Used by: disaggregated serving pipeline (prefill -> decode handoff)

use std::time::Instant;

use anyhow::{Context, Result};
use bytes::Bytes;
use tokio::sync::mpsc;

use super::{
    CacheQuantizationLevel, LayerTransferHeader, LayerTransferResult, TransferConfig,
    TransferResult, TransferStrategy,
};
use crate::distributed::kv_cache_serde::types::SerializableCacheEntry;
use crate::distributed::transport::{Transport, TransportMessage};

/// Notification channel for layer completion during prefill.
///
/// The prefill loop calls [`notify_layer_ready`] after each layer's
/// forward pass completes. The streamed transfer listens and sends
/// each layer's cache data as it becomes available.
#[derive(Debug)]
pub struct LayerReadyNotifier {
    /// Sender for layer-ready events.
    tx: mpsc::Sender<LayerReadyEvent>,
    /// Receiver for layer-ready events (consumed by the transfer loop).
    rx: Option<mpsc::Receiver<LayerReadyEvent>>,
}

/// Event emitted when a layer's cache is ready for transfer.
#[derive(Debug, Clone)]
pub struct LayerReadyEvent {
    /// Index of the layer that just completed.
    pub layer_index: usize,
    /// The cache entry for this layer (already extracted from the KV cache).
    pub entry: SerializableCacheEntry,
}

impl LayerReadyNotifier {
    /// Create a new notifier with the given channel capacity.
    ///
    /// `capacity` controls backpressure: if the transfer cannot keep up
    /// with prefill, the channel will buffer up to `capacity` layers
    /// before blocking the prefill loop.
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        Self { tx, rx: Some(rx) }
    }

    /// Notify that a layer's cache is ready for transfer.
    ///
    /// This is called from the prefill loop after each layer completes.
    /// If the channel is full, this will wait (providing backpressure).
    pub async fn notify_layer_ready(&self, event: LayerReadyEvent) -> Result<()> {
        self.tx
            .send(event)
            .await
            .map_err(|_| anyhow::anyhow!("layer notifier channel closed"))
    }

    /// Signal that all layers have completed (no more events will follow).
    ///
    /// This is achieved by dropping the sender, which causes the receiver
    /// to return `None` on the next recv.
    pub fn finish(self) {
        // tx is dropped here, closing the channel.
        drop(self.tx);
    }

    /// Take the receiver (can only be called once).
    pub fn take_receiver(&mut self) -> Option<mpsc::Receiver<LayerReadyEvent>> {
        self.rx.take()
    }
}

/// Streamed layer-by-layer cache transfer.
///
/// Consumes layer-ready events from a [`LayerReadyNotifier`] and sends
/// each layer's data to the decode node as it becomes available.
pub struct StreamedCacheTransfer {
    config: TransferConfig,
    /// Target peer address.
    peer: String,
    /// Sequence ID for this transfer.
    sequence_id: u64,
    /// Total layers expected.
    total_layers: usize,
}

impl StreamedCacheTransfer {
    /// Create a new streamed transfer.
    pub fn new(
        config: TransferConfig,
        peer: String,
        sequence_id: u64,
        total_layers: usize,
    ) -> Self {
        Self {
            config,
            peer,
            sequence_id,
            total_layers,
        }
    }

    /// Run the streamed transfer, consuming layer events from the receiver.
    ///
    /// Returns a [`TransferResult`] summarizing the entire transfer.
    /// The transfer completes when the receiver is closed (all layers sent)
    /// or an error occurs.
    pub async fn run(
        &self,
        transport: &dyn Transport,
        mut rx: mpsc::Receiver<LayerReadyEvent>,
    ) -> Result<TransferResult> {
        let transfer_start = Instant::now();
        let mut layer_results = Vec::with_capacity(self.total_layers);
        let mut total_wire_bytes = 0usize;
        let mut total_original_bytes = 0usize;

        while let Some(event) = rx.recv().await {
            let result = self
                .send_layer(transport, &event)
                .await
                .with_context(|| format!("sending layer {}", event.layer_index))?;

            total_wire_bytes += result.wire_bytes;
            total_original_bytes += result.original_bytes;
            layer_results.push(result);
        }

        Ok(TransferResult {
            strategy: TransferStrategy::Streamed,
            quantization: self.config.quantization,
            layer_results,
            total_duration: transfer_start.elapsed(),
            total_wire_bytes,
            total_original_bytes,
        })
    }

    /// Send the entire cache state as a streamed transfer (non-incremental).
    ///
    /// This is the fallback when layer-by-layer notification is not available.
    /// Iterates over all entries and sends them sequentially.
    pub async fn send_all(
        &self,
        transport: &dyn Transport,
        entries: &[SerializableCacheEntry],
    ) -> Result<TransferResult> {
        let transfer_start = Instant::now();
        let mut layer_results = Vec::with_capacity(entries.len());
        let mut total_wire_bytes = 0usize;
        let mut total_original_bytes = 0usize;

        for (i, entry) in entries.iter().enumerate() {
            let event = LayerReadyEvent {
                layer_index: i,
                entry: entry.clone(),
            };

            let result = self
                .send_layer(transport, &event)
                .await
                .with_context(|| format!("sending layer {i}"))?;

            total_wire_bytes += result.wire_bytes;
            total_original_bytes += result.original_bytes;
            layer_results.push(result);
        }

        Ok(TransferResult {
            strategy: TransferStrategy::Streamed,
            quantization: self.config.quantization,
            layer_results,
            total_duration: transfer_start.elapsed(),
            total_wire_bytes,
            total_original_bytes,
        })
    }

    /// Send a single layer's cache entry over the transport.
    async fn send_layer(
        &self,
        transport: &dyn Transport,
        event: &LayerReadyEvent,
    ) -> Result<LayerTransferResult> {
        let layer_start = Instant::now();

        let (wire_data, original_bytes, num_elements) =
            prepare_layer_payload(&event.entry, self.config.quantization, self.config.compress)?;

        let header = LayerTransferHeader {
            sequence_id: self.sequence_id,
            layer_index: event.layer_index,
            total_layers: self.total_layers,
            quantized: self.config.quantization != CacheQuantizationLevel::None,
            quantization_level: self.config.quantization,
            original_num_elements: num_elements,
        };

        let header_json = serde_json::to_vec(&header).context("serializing layer header")?;

        // Send header as control message.
        transport
            .send(
                &self.peer,
                TransportMessage::Control {
                    operation: "kv_cache_layer".to_string(),
                    payload: Bytes::from(header_json),
                },
            )
            .await
            .context("sending layer header")?;

        // Send tensor data as stream.
        let wire_bytes = wire_data.len();
        transport
            .send_stream(&self.peer, Bytes::from(wire_data))
            .await
            .context("sending layer data")?;

        Ok(LayerTransferResult {
            layer_index: event.layer_index,
            wire_bytes,
            original_bytes,
            duration: layer_start.elapsed(),
        })
    }
}

/// Prepare a layer's cache entry for wire transfer.
///
/// Applies optional quantization and compression, returning the wire
/// bytes, the original byte count, and the number of float16 elements
/// (for dequantization sizing on the receiver).
///
/// Used by: StreamedCacheTransfer, ParallelLayerTransfer
pub fn prepare_layer_payload(
    entry: &SerializableCacheEntry,
    quantization: CacheQuantizationLevel,
    compress: bool,
) -> Result<(Vec<u8>, usize, usize)> {
    // Concatenate key and value tensor data.
    let (raw_bytes, num_elements) = match (&entry.keys, &entry.values) {
        (Some(keys), Some(values)) => {
            let mut data = Vec::with_capacity(keys.data.len() + values.data.len());
            data.extend_from_slice(&keys.data);
            data.extend_from_slice(&values.data);
            let elems = keys.data.len() / 2 + values.data.len() / 2; // float16 = 2 bytes
            (data, elems)
        }
        _ => return Ok((Vec::new(), 0, 0)),
    };

    let original_bytes = raw_bytes.len();

    // Apply quantization.
    let quantized = match quantization {
        CacheQuantizationLevel::None => raw_bytes,
        CacheQuantizationLevel::Int8 => {
            crate::distributed::tensor_quantize::quantize_int8(&raw_bytes)
        }
        CacheQuantizationLevel::Int4 => {
            crate::distributed::tensor_quantize::quantize_int4(&raw_bytes)
        }
    };

    // Apply optional compression.
    let wire_data = if compress {
        match crate::distributed::tensor_compress::compress_if_beneficial(&quantized) {
            Some(compressed) => compressed,
            None => quantized,
        }
    } else {
        quantized
    };

    Ok((wire_data, original_bytes, num_elements))
}

/// Receive and reassemble a single layer from streamed transfer.
///
/// Applies dequantization and decompression as needed based on the
/// layer header.
///
/// Used by: decode node receiving streamed cache data
pub fn reassemble_layer_payload(
    header: &LayerTransferHeader,
    wire_data: &[u8],
    compressed: bool,
) -> Result<Vec<u8>> {
    // Decompress if needed.
    let decompressed = if compressed {
        crate::distributed::tensor_compress::decompress(wire_data)
            .context("decompressing layer data")?
    } else {
        wire_data.to_vec()
    };

    // Dequantize if needed.
    let raw_data = match header.quantization_level {
        CacheQuantizationLevel::None => decompressed,
        CacheQuantizationLevel::Int8 => crate::distributed::tensor_quantize::dequantize_int8(
            &decompressed,
            header.original_num_elements,
        ),
        CacheQuantizationLevel::Int4 => crate::distributed::tensor_quantize::dequantize_int4(
            &decompressed,
            header.original_num_elements,
        ),
    };

    Ok(raw_data)
}

#[cfg(test)]
#[path = "streamed_tests.rs"]
mod tests;
