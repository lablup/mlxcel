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
        original_bytes: 512,
    };

    let result = reassemble_layer_payload(&header, &wire_data, false, NO_LIMIT).unwrap();
    assert_eq!(result.len(), wire_data.len());
}

#[test]
fn reassemble_int8_roundtrip() {
    let entry = make_entry(128);
    let (wire_data, original_bytes, num_elements) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::Int8, false).unwrap();

    let header = LayerTransferHeader {
        sequence_id: 1,
        layer_index: 0,
        total_layers: 1,
        quantized: true,
        quantization_level: CacheQuantizationLevel::Int8,
        original_num_elements: num_elements,
        original_bytes,
    };

    let result = reassemble_layer_payload(&header, &wire_data, false, NO_LIMIT).unwrap();
    // Dequantized should have float16 bytes for all original elements.
    assert_eq!(result.len(), num_elements * 2);
}

/// `max_layer_bytes` for tests that exercise another check.
const NO_LIMIT: usize = usize::MAX;

/// A header for an honest float16 layer of `original_num_elements`.
fn layer_header(
    level: CacheQuantizationLevel,
    original_num_elements: usize,
) -> LayerTransferHeader {
    LayerTransferHeader {
        sequence_id: 1,
        layer_index: 0,
        total_layers: 1,
        quantized: level != CacheQuantizationLevel::None,
        quantization_level: level,
        original_num_elements,
        original_bytes: original_num_elements * 2,
    }
}

/// Compressed layers still round-trip at every quantization level with the
/// decompression limit derived from `original_num_elements`.
#[test]
fn reassemble_compressed_roundtrip_at_every_level() {
    let entry = make_entry(1024);
    for level in [
        CacheQuantizationLevel::None,
        CacheQuantizationLevel::Int8,
        CacheQuantizationLevel::Int4,
    ] {
        let (wire_data, original_bytes, num_elements) =
            prepare_layer_payload(&entry, level, true).unwrap();
        assert!(
            wire_data.len() < original_bytes / 4,
            "zero-filled layer should compress at {level:?}"
        );
        let result = reassemble_layer_payload(
            &layer_header(level, num_elements),
            &wire_data,
            true,
            NO_LIMIT,
        )
        .unwrap();
        assert_eq!(result.len(), num_elements * 2, "{level:?}");
    }
}

/// A compressed layer whose LZ4 prefix declares more than the header's
/// element count allows is rejected before the buffer is allocated.
#[test]
fn reassemble_rejects_compressed_size_beyond_header() {
    let entry = make_entry(1024);
    let (wire_data, _, num_elements) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::None, true).unwrap();
    let header = layer_header(CacheQuantizationLevel::None, num_elements / 2);
    let err = match reassemble_layer_payload(&header, &wire_data, true, NO_LIMIT) {
        Ok(out) => panic!("must be rejected, got {} bytes", out.len()),
        Err(err) => format!("{err:#}"),
    };
    assert!(err.contains("exceeds limit"), "unexpected error: {err}");
}

/// A quantized layer whose header element count disagrees with the payload
/// layout is an error, not a zero-filled buffer.
#[test]
fn reassemble_rejects_quantized_layout_mismatch() {
    let entry = make_entry(128);
    let (wire_data, _, num_elements) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::Int8, false).unwrap();
    let header = layer_header(CacheQuantizationLevel::Int8, num_elements * 4);
    assert!(reassemble_layer_payload(&header, &wire_data, false, NO_LIMIT).is_err());
}

/// A cache entry of raw bytes in a 1-byte dtype (uint8), so the key and value
/// byte counts can be odd.
fn byte_entry(key_bytes: usize, value_bytes: usize) -> SerializableCacheEntry {
    let tensor = |len: usize| RawTensorData {
        data: vec![0u8; len],
        shape: vec![1, 1, len as i32, 1],
        dtype: 1, // UINT8
    };
    SerializableCacheEntry {
        keys: Some(tensor(key_bytes)),
        values: Some(tensor(value_bytes)),
    }
}

/// The header a sender builds from `prepare_layer_payload`'s output.
fn sent_header(
    level: CacheQuantizationLevel,
    original_bytes: usize,
    num_elements: usize,
) -> LayerTransferHeader {
    LayerTransferHeader {
        original_bytes,
        ..layer_header(level, num_elements)
    }
}

/// Issue #2716 item 1: a header whose `original_bytes` exceeds the
/// receiver's own per-layer bound is rejected before decompressing. The wire
/// bytes are not valid LZ4, so reaching `decompress` would fail differently.
#[test]
fn reassemble_rejects_original_bytes_above_max_layer_bytes() {
    let header = layer_header(CacheQuantizationLevel::None, 2048);
    let err = match reassemble_layer_payload(&header, &[0xFF; 16], true, 4095) {
        Ok(out) => panic!("must be rejected, got {} bytes", out.len()),
        Err(err) => format!("{err:#}"),
    };
    assert!(
        err.contains("4096 original bytes, above the limit 4095"),
        "unexpected error: {err}"
    );

    // The same bound admits an honest layer of exactly that size.
    let (wire_data, original_bytes, num_elements) =
        prepare_layer_payload(&make_entry(1024), CacheQuantizationLevel::None, true).unwrap();
    let header = sent_header(CacheQuantizationLevel::None, original_bytes, num_elements);
    let out = reassemble_layer_payload(&header, &wire_data, true, original_bytes).unwrap();
    assert_eq!(out.len(), original_bytes);
}

/// Issue #2716 item 2: a 1-byte layer with odd key and value byte counts
/// used to count `4095 / 2 + 4097 / 2 = 4095` elements, a decompression bound
/// two bytes short of the 8192-byte layer, so a valid compressed layer was
/// rejected with "exceeds limit".
#[test]
fn reassemble_round_trips_odd_byte_layer_compressed() {
    let entry = byte_entry(4095, 4097);
    let (wire_data, original_bytes, num_elements) =
        prepare_layer_payload(&entry, CacheQuantizationLevel::None, true).unwrap();
    assert_eq!(original_bytes, 8192);
    assert_eq!(num_elements, 4096);
    assert!(
        wire_data.len() < original_bytes,
        "the zero-filled layer must go out compressed"
    );
    let header = sent_header(CacheQuantizationLevel::None, original_bytes, num_elements);
    let out = reassemble_layer_payload(&header, &wire_data, true, NO_LIMIT).unwrap();
    let keys = &entry.keys.as_ref().unwrap().data;
    let values = &entry.values.as_ref().unwrap().data;
    assert_eq!(out, [keys.as_slice(), values.as_slice()].concat());
}

/// Issue #2716 item 2: Int8 and Int4 quantize float16 pairs and silently
/// dropped a trailing byte; an odd layer is now an error.
#[test]
fn prepare_rejects_odd_byte_layer_for_quantization() {
    for level in [CacheQuantizationLevel::Int8, CacheQuantizationLevel::Int4] {
        let err = match prepare_layer_payload(&byte_entry(3, 4), level, false) {
            Ok((wire, ..)) => panic!(
                "{level:?} must reject an odd layer, got {} bytes",
                wire.len()
            ),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("odd 7-byte layer"), "unexpected error: {err}");
    }
}

/// Issue #2716 item 2: a quantized header whose byte count disagrees with
/// its float16 element count is rejected.
#[test]
fn reassemble_rejects_quantized_header_bytes_mismatch() {
    let (wire_data, original_bytes, num_elements) =
        prepare_layer_payload(&make_entry(128), CacheQuantizationLevel::Int8, false).unwrap();
    let header = sent_header(
        CacheQuantizationLevel::Int8,
        original_bytes + 2,
        num_elements,
    );
    let err = match reassemble_layer_payload(&header, &wire_data, false, NO_LIMIT) {
        Ok(out) => panic!("must be rejected, got {} bytes", out.len()),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("elements but"), "unexpected error: {err}");
}

/// Issue #2716 item 3: uncompressed level-None wire data must be exactly the
/// header's `original_bytes`, neither shorter nor longer.
#[test]
fn reassemble_rejects_uncompressed_length_mismatch() {
    let (wire_data, original_bytes, num_elements) =
        prepare_layer_payload(&make_entry(128), CacheQuantizationLevel::None, false).unwrap();
    let header = sent_header(CacheQuantizationLevel::None, original_bytes, num_elements);
    let short = &wire_data[..wire_data.len() - 1];
    let mut long = wire_data.clone();
    long.push(0);
    for (what, wire) in [("short", short), ("long", long.as_slice())] {
        let err = match reassemble_layer_payload(&header, wire, false, NO_LIMIT) {
            Ok(out) => panic!("{what} wire data must be rejected, got {} bytes", out.len()),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("uncompressed layer data is"),
            "{what}: unexpected error: {err}"
        );
    }
    let out = reassemble_layer_payload(&header, &wire_data, false, NO_LIMIT).unwrap();
    assert_eq!(out, wire_data);
}
