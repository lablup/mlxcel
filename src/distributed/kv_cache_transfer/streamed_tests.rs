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

use super::*;
use crate::distributed::kv_cache_serde::types::RawTensorData;

/// Helper to create a synthetic cache entry with float16 data.
fn make_entry(num_elements: usize) -> SerializableCacheEntry {
    let bytes = num_elements * 2; // float16
    let data = vec![0x00u8; bytes]; // zero-filled float16
    SerializableCacheEntry {
        keys: Some(RawTensorData {
            data: data.clone(),
            shape: vec![1, 1, num_elements as i32, 1],
            dtype: 9, // FLOAT16
        }),
        values: Some(RawTensorData {
            data,
            shape: vec![1, 1, num_elements as i32, 1],
            dtype: 9,
        }),
    }
}

// --- LayerReadyNotifier ---

#[tokio::test]
async fn notifier_send_receive() {
    let mut notifier = LayerReadyNotifier::new(4);
    let rx = notifier.take_receiver().unwrap();

    let entry = make_entry(64);
    notifier
        .notify_layer_ready(LayerReadyEvent {
            layer_index: 0,
            entry: entry.clone(),
        })
        .await
        .unwrap();

    notifier
        .notify_layer_ready(LayerReadyEvent {
            layer_index: 1,
            entry: entry.clone(),
        })
        .await
        .unwrap();

    notifier.finish();

    // Receiver should get both events then None.
    let mut rx = rx;
    let event0 = rx.recv().await.unwrap();
    assert_eq!(event0.layer_index, 0);
    let event1 = rx.recv().await.unwrap();
    assert_eq!(event1.layer_index, 1);
    assert!(rx.recv().await.is_none());
}

#[tokio::test]
async fn notifier_take_receiver_once() {
    let mut notifier = LayerReadyNotifier::new(4);
    assert!(notifier.take_receiver().is_some());
    assert!(notifier.take_receiver().is_none());
}

// --- prepare_layer_payload ---

#[test]
fn prepare_no_quantization() {
    let entry = make_entry(128);
    let (data, orig, elems) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::None, false).unwrap();

    // Should be raw concatenation of keys + values.
    assert_eq!(orig, 128 * 2 * 2); // 128 elems * 2 bytes * 2 tensors
    assert_eq!(data.len(), orig);
    assert_eq!(elems, 128 * 2); // 128 elems * 2 tensors (each float16 = 2 bytes / 2)
}

#[test]
fn prepare_with_int8_quantization() {
    let entry = make_entry(256);
    let (data, orig, _elems) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::Int8, false).unwrap();

    // Int8 quantized output should be smaller than original.
    assert!(data.len() < orig);
    // Roughly 50% reduction.
    let ratio = data.len() as f64 / orig as f64;
    assert!(ratio < 0.7, "Expected <70% ratio, got {ratio:.2}");
}

#[test]
fn prepare_with_int4_quantization() {
    let entry = make_entry(256);
    let (data, orig, _elems) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::Int4, false).unwrap();

    // Int4 should be even smaller.
    assert!(data.len() < orig);
    let ratio = data.len() as f64 / orig as f64;
    assert!(ratio < 0.5, "Expected <50% ratio, got {ratio:.2}");
}

#[test]
fn prepare_empty_entry() {
    let entry = SerializableCacheEntry {
        keys: None,
        values: None,
    };
    let (data, orig, elems) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::Int8, true).unwrap();
    assert!(data.is_empty());
    assert_eq!(orig, 0);
    assert_eq!(elems, 0);
}

// --- reassemble_layer_payload ---

#[test]
fn reassemble_no_quantization() {
    let entry = make_entry(128);
    let (wire_data, _, _) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::None, false).unwrap();

    let header = LayerTransferHeader {
        sequence_id: 1,
        layer_index: 0,
        total_layers: 1,
        quantized: false,
        quantization_level: CacheQuantizationLevel::None,
        original_num_elements: 256, // 128 * 2 tensors
    };

    let result = reassemble_layer_payload(&header, &wire_data, false).unwrap();
    assert_eq!(result.len(), wire_data.len());
}

#[test]
fn reassemble_int8_roundtrip() {
    let entry = make_entry(128);
    let (wire_data, _, num_elements) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::Int8, false).unwrap();

    let header = LayerTransferHeader {
        sequence_id: 1,
        layer_index: 0,
        total_layers: 1,
        quantized: true,
        quantization_level: CacheQuantizationLevel::Int8,
        original_num_elements: num_elements,
    };

    let result = reassemble_layer_payload(&header, &wire_data, false).unwrap();
    // Dequantized should have float16 bytes for all original elements.
    assert_eq!(result.len(), num_elements * 2);
}
