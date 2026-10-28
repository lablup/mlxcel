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

use std::borrow::Cow;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use bytes::Bytes;
use tokio::sync::mpsc;

use super::{
    CacheQuantizationLevel, LayerTransferHeader, LayerTransferResult, TransferConfig,
    TransferResult, TransferStrategy,
};
use crate::distributed::kv_cache_serde::types::SerializableCacheEntry;
use crate::distributed::tensor_protocol::max_decoded_tensor_bytes;
use crate::distributed::transport::{Transport, TransportMessage};
use crate::distributed::{tensor_compress, tensor_quantize};

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
            original_bytes,
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
/// (half the original byte count, for dequantization sizing on the
/// receiver). Int8 and Int4 quantize the concatenated keys and values as
/// float16, so an odd byte count is an error for them.
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
            // Count over the concatenation, as the quantizers do: per-tensor
            // halves undercount when both tensors have an odd byte count.
            let elems = data.len() / 2; // float16 = 2 bytes
            (data, elems)
        }
        _ => return Ok((Vec::new(), 0, 0)),
    };
    if quantization != CacheQuantizationLevel::None && raw_bytes.len() % 2 != 0 {
        bail!(
            "{quantization:?} cache quantization needs float16 data, got an odd {}-byte layer",
            raw_bytes.len()
        );
    }

    let original_bytes = raw_bytes.len();

    // Apply quantization.
    let quantized = match quantization {
        CacheQuantizationLevel::None => raw_bytes,
        CacheQuantizationLevel::Int8 => tensor_quantize::quantize_int8(&raw_bytes),
        CacheQuantizationLevel::Int4 => tensor_quantize::quantize_int4(&raw_bytes),
    };

    // Apply optional compression.
    let wire_data = if compress {
        match tensor_compress::compress_if_beneficial(&quantized) {
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
/// layer header. The wire data comes from a peer, so the header's
/// `original_bytes` must not exceed `max_layer_bytes`, decompression is
/// capped at what the header allows, an unquantized layer must decode to
/// exactly `original_bytes`, and the quantized layout is validated before any
/// output is allocated.
///
/// `max_layer_bytes` is the receiver's own bound on one layer's raw key plus
/// value bytes, derived from the geometry it expects
/// (`ExpectedBlockGeometry`): `2 * n_kv_heads * head_dim * element_size *
/// token_budget`.
///
/// Used by: decode node receiving streamed cache data
pub fn reassemble_layer_payload(
    header: &LayerTransferHeader,
    wire_data: &[u8],
    compressed: bool,
    max_layer_bytes: usize,
) -> Result<Vec<u8>> {
    let original_bytes = header.original_bytes;
    if original_bytes > max_layer_bytes {
        bail!("layer declares {original_bytes} original bytes, above the limit {max_layer_bytes}");
    }
    let num_elements = header.original_num_elements;

    type MaxPayloadLen = fn(usize) -> Option<usize>;
    type Dequantize = fn(&[u8], usize) -> Result<Vec<u8>>;
    let (max_payload_len, dequantize, label): (MaxPayloadLen, Dequantize, &str) =
        match header.quantization_level {
            CacheQuantizationLevel::None => {
                return reassemble_unquantized(wire_data, compressed, original_bytes);
            }
            CacheQuantizationLevel::Int8 => (
                tensor_quantize::int8_payload_max_len,
                tensor_quantize::dequantize_int8,
                "int8",
            ),
            CacheQuantizationLevel::Int4 => (
                tensor_quantize::int4_payload_max_len,
                tensor_quantize::dequantize_int4,
                "int4",
            ),
        };

    // Quantized layers carry float16 data: two bytes per element.
    ensure!(
        num_elements.checked_mul(2) == Some(original_bytes),
        "quantized layer header declares {num_elements} elements but {original_bytes} bytes"
    );
    let decompressed = if compressed {
        let max_len = max_payload_len(num_elements)
            .ok_or_else(|| {
                anyhow!("layer element count {num_elements} overflows the decoded size")
            })?
            .min(max_decoded_tensor_bytes());
        Cow::Owned(
            tensor_compress::decompress(wire_data, max_len).context("decompressing layer data")?,
        )
    } else {
        Cow::Borrowed(wire_data)
    };
    dequantize(&decompressed, num_elements)
        .with_context(|| format!("dequantizing {label} layer data"))
}

/// Level None: the layer is the raw key plus value bytes, exactly
/// `original_bytes` of them whether or not they were compressed.
fn reassemble_unquantized(
    wire_data: &[u8],
    compressed: bool,
    original_bytes: usize,
) -> Result<Vec<u8>> {
    let raw_data = if compressed {
        tensor_compress::decompress(wire_data, original_bytes.min(max_decoded_tensor_bytes()))
            .context("decompressing layer data")?
    } else {
        ensure!(
            wire_data.len() == original_bytes,
            "uncompressed layer data is {} bytes, header declares {original_bytes}",
            wire_data.len()
        );
        wire_data.to_vec()
    };
    ensure!(
        raw_data.len() == original_bytes,
        "layer data decompressed to {} bytes, header declares {original_bytes}",
        raw_data.len()
    );
    Ok(raw_data)
}

#[cfg(test)]
#[path = "streamed_tests.rs"]
mod tests;
