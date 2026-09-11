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

/// Helper to create a cache entry with float16 data.
fn make_entry(num_elements: usize) -> SerializableCacheEntry {
    let bytes = num_elements * 2;
    // Fill with a recognizable pattern (small float16 values).
    let mut data = vec![0u8; bytes];
    for i in 0..num_elements {
        // float16 for ~0.5: sign=0, exp=14, mantissa=0 => 0x3800
        data[i * 2] = 0x00;
        data[i * 2 + 1] = 0x38;
    }
    SerializableCacheEntry {
        keys: Some(RawTensorData {
            data: data.clone(),
            shape: vec![1, 1, num_elements as i32, 1],
            dtype: 9,
        }),
        values: Some(RawTensorData {
            data,
            shape: vec![1, 1, num_elements as i32, 1],
            dtype: 9,
        }),
    }
}

// --- CacheQuantizationConfig ---

#[test]
fn config_uniform() {
    let config = CacheQuantizationConfig::uniform(CacheQuantizationLevel::Int8);
    assert_eq!(config.level_for_layer(0), CacheQuantizationLevel::Int8);
    assert_eq!(config.level_for_layer(31), CacheQuantizationLevel::Int8);
}

#[test]
fn config_protect_boundary() {
    let config =
        CacheQuantizationConfig::protect_boundary_layers(CacheQuantizationLevel::Int8, 32, 2);
    // First 2 layers should be unquantized.
    assert_eq!(config.level_for_layer(0), CacheQuantizationLevel::None);
    assert_eq!(config.level_for_layer(1), CacheQuantizationLevel::None);
    // Middle layers should be int8.
    assert_eq!(config.level_for_layer(2), CacheQuantizationLevel::Int8);
    assert_eq!(config.level_for_layer(15), CacheQuantizationLevel::Int8);
    // Last 2 layers should be unquantized.
    assert_eq!(config.level_for_layer(30), CacheQuantizationLevel::None);
    assert_eq!(config.level_for_layer(31), CacheQuantizationLevel::None);
}

#[test]
fn config_default_is_none() {
    let config = CacheQuantizationConfig::default();
    assert_eq!(config.default_level, CacheQuantizationLevel::None);
    assert!(config.layer_overrides.is_empty());
}

// --- QuantizedCacheTransfer ---

#[test]
fn quantize_none_is_identity() {
    let transfer = QuantizedCacheTransfer::new(CacheQuantizationConfig::uniform(
        CacheQuantizationLevel::None,
    ));
    let entry = make_entry(256);
    let result = transfer.quantize_entry(&entry, 0).unwrap();

    assert_eq!(result.level, CacheQuantizationLevel::None);
    assert_eq!(result.original_key_bytes, result.quantized_key_bytes);
    assert_eq!(result.original_value_bytes, result.quantized_value_bytes);
    assert_eq!(result.compression_ratio(), 1.0);
}

#[test]
fn quantize_int8_reduces_size() {
    let transfer = QuantizedCacheTransfer::new(CacheQuantizationConfig::uniform(
        CacheQuantizationLevel::Int8,
    ));
    let entry = make_entry(256);
    let result = transfer.quantize_entry(&entry, 0).unwrap();

    assert_eq!(result.level, CacheQuantizationLevel::Int8);
    assert!(result.quantized_key_bytes < result.original_key_bytes);
    assert!(result.quantized_value_bytes < result.original_value_bytes);
    assert!(result.compression_ratio() < 0.7);
}

#[test]
fn quantize_int4_reduces_more() {
    let transfer = QuantizedCacheTransfer::new(CacheQuantizationConfig::uniform(
        CacheQuantizationLevel::Int4,
    ));
    let entry = make_entry(256);
    let result_int4 = transfer.quantize_entry(&entry, 0).unwrap();

    let transfer_int8 = QuantizedCacheTransfer::new(CacheQuantizationConfig::uniform(
        CacheQuantizationLevel::Int8,
    ));
    let result_int8 = transfer_int8.quantize_entry(&entry, 0).unwrap();

    assert!(result_int4.quantized_total_bytes() < result_int8.quantized_total_bytes());
}

#[test]
fn quantize_dequantize_roundtrip() {
    let transfer = QuantizedCacheTransfer::new(CacheQuantizationConfig::uniform(
        CacheQuantizationLevel::Int8,
    ));
    let entry = make_entry(256);
    let quantized = transfer.quantize_entry(&entry, 0).unwrap();
    let dequantized = transfer.dequantize_entry(&quantized).unwrap();

    // Shape should be preserved.
    let orig_shape = &entry.keys.as_ref().unwrap().shape;
    let deq_shape = &dequantized.keys.as_ref().unwrap().shape;
    assert_eq!(orig_shape, deq_shape);

    // Data length should match original.
    let orig_len = entry.keys.as_ref().unwrap().data.len();
    let deq_len = dequantized.keys.as_ref().unwrap().data.len();
    assert_eq!(orig_len, deq_len);

    // Dtype should be restored to float16.
    assert_eq!(dequantized.keys.as_ref().unwrap().dtype, 9);
}

#[test]
fn quantize_all_layers() {
    let transfer = QuantizedCacheTransfer::new(CacheQuantizationConfig::uniform(
        CacheQuantizationLevel::Int8,
    ));
    let entries: Vec<_> = (0..4).map(|_| make_entry(128)).collect();
    let results = transfer.quantize_all(&entries).unwrap();

    assert_eq!(results.len(), 4);
    for result in &results {
        assert!(result.compression_ratio() < 0.7);
    }
}

#[test]
fn quantize_empty_entry() {
    let transfer = QuantizedCacheTransfer::new(CacheQuantizationConfig::uniform(
        CacheQuantizationLevel::Int8,
    ));
    let entry = SerializableCacheEntry {
        keys: None,
        values: None,
    };
    let result = transfer.quantize_entry(&entry, 0).unwrap();
    assert_eq!(result.original_total_bytes(), 0);
    assert_eq!(result.quantized_total_bytes(), 0);
}

// --- estimate_savings ---

#[test]
fn savings_none() {
    let savings = estimate_savings(1_000_000, CacheQuantizationLevel::None);
    assert_eq!(savings.saved_bytes, 0);
    assert_eq!(savings.ratio, 1.0);
}

#[test]
fn savings_int8() {
    let savings = estimate_savings(1_000_000, CacheQuantizationLevel::Int8);
    assert!(savings.saved_bytes > 400_000);
    assert!(savings.ratio < 0.55);
}

#[test]
fn savings_int4() {
    let savings = estimate_savings(1_000_000, CacheQuantizationLevel::Int4);
    assert!(savings.saved_bytes > 700_000);
    assert!(savings.ratio < 0.30);
}

// --- QuantizedEntry ---

#[test]
fn quantized_entry_totals() {
    let qe = QuantizedEntry {
        entry: SerializableCacheEntry {
            keys: None,
            values: None,
        },
        level: CacheQuantizationLevel::Int8,
        original_key_bytes: 1000,
        original_value_bytes: 1000,
        quantized_key_bytes: 500,
        quantized_value_bytes: 500,
    };
    assert_eq!(qe.original_total_bytes(), 2000);
    assert_eq!(qe.quantized_total_bytes(), 1000);
    assert!((qe.compression_ratio() - 0.5).abs() < 0.01);
}
