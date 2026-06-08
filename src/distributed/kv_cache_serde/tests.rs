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

use super::*;
use mlxcel_core as ffi;

// ---------------------------------------------------------------------------
// Helper: create a small KVCache with known data
// ---------------------------------------------------------------------------

fn make_test_kv_cache(seq_len: i32, n_heads: i32, head_dim: i32) -> mlxcel_core::cache::KVCache {
    let mut cache = mlxcel_core::cache::KVCache::new();
    if seq_len > 0 {
        let n_elements = (n_heads * seq_len * head_dim) as usize;
        let key_data: Vec<f32> = (0..n_elements).map(|i| i as f32 * 0.1).collect();
        let val_data: Vec<f32> = (0..n_elements).map(|i| i as f32 * 0.2).collect();

        let keys = ffi::from_slice_f32(&key_data, &[1, n_heads, seq_len, head_dim]);
        let values = ffi::from_slice_f32(&val_data, &[1, n_heads, seq_len, head_dim]);
        cache.update(keys, values);
    }
    cache
}

// ---------------------------------------------------------------------------
// Types tests
// ---------------------------------------------------------------------------

#[test]
fn cache_type_round_trip() {
    for &ct in &[CacheType::Standard, CacheType::Rotating, CacheType::Chunked] {
        let byte = ct as u8;
        let recovered = CacheType::try_from(byte).unwrap();
        assert_eq!(ct, recovered);
    }
}

#[test]
fn cache_type_invalid_discriminant() {
    assert!(CacheType::try_from(255u8).is_err());
}

#[test]
fn raw_tensor_data_serde_round_trip() {
    let tensor = RawTensorData {
        data: vec![1, 2, 3, 4, 5, 6, 7, 8],
        shape: vec![1, 1, 2, 1],
        dtype: mlxcel_core::dtype::FLOAT32,
    };
    let json = serde_json::to_vec(&tensor).unwrap();
    let recovered: RawTensorData = serde_json::from_slice(&json).unwrap();
    assert_eq!(tensor.data, recovered.data);
    assert_eq!(tensor.shape, recovered.shape);
    assert_eq!(tensor.dtype, recovered.dtype);
}

// ---------------------------------------------------------------------------
// Extract + reconstruct round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn kv_cache_extract_reconstruct_round_trip() {
    let cache = make_test_kv_cache(4, 2, 8);

    // Extract
    let entry = extract_kv_cache_entry(&cache);
    assert!(entry.keys.is_some());
    assert!(entry.values.is_some());

    // Reconstruct
    let k = reconstruct_mlx_array(entry.keys.as_ref().unwrap()).unwrap();
    let v = reconstruct_mlx_array(entry.values.as_ref().unwrap()).unwrap();

    // Verify shape
    let k_shape = ffi::array_shape(&k);
    let v_shape = ffi::array_shape(&v);
    assert_eq!(k_shape, vec![1, 2, 4, 8]);
    assert_eq!(v_shape, vec![1, 2, 4, 8]);

    // Verify data is bit-exact by comparing the filled portion
    // The original buffer may be pre-allocated (e.g. 256), so we slice to offset
    let original_keys = cache.keys.as_ref().unwrap();
    let oks = ffi::array_shape(original_keys);
    let orig_k_sliced = ffi::slice(
        original_keys,
        &[0, 0, 0, 0],
        &[oks[0], oks[1], cache.offset, oks[3]],
    );
    let orig_k_flat = ffi::reshape(&orig_k_sliced, &[-1]);
    let new_k_flat = ffi::reshape(&k, &[-1]);

    let eq = ffi::array_equal(&orig_k_flat, &new_k_flat, false);
    ffi::eval(&eq);
    assert!(ffi::item_bool(&eq), "reconstructed keys must be bit-exact");

    let original_values = cache.values.as_ref().unwrap();
    let ovs = ffi::array_shape(original_values);
    let orig_v_sliced = ffi::slice(
        original_values,
        &[0, 0, 0, 0],
        &[ovs[0], ovs[1], cache.offset, ovs[3]],
    );
    let orig_v_flat = ffi::reshape(&orig_v_sliced, &[-1]);
    let new_v_flat = ffi::reshape(&v, &[-1]);

    let eq_v = ffi::array_equal(&orig_v_flat, &new_v_flat, false);
    ffi::eval(&eq_v);
    assert!(
        ffi::item_bool(&eq_v),
        "reconstructed values must be bit-exact"
    );
}

/// The serde restore path (`reconstruct_mlx_array` -> `from_bytes`) must
/// round-trip 16-bit float tensors bit-exactly. Real model KV is fp16/bf16, and
/// both the dense and paged (#125) KV serde carry it as `RawTensorData`; a
/// `from_bytes` regression on 16-bit dtypes silently corrupts every transferred
/// tensor. The other extract/reconstruct tests above use FP32 (which has an
/// explicit `from_bytes` case), so they are blind to the 16-bit path that bit
/// the distributed handoff.
#[test]
fn reconstruct_mlx_array_round_trips_16bit_floats() {
    let vals: Vec<f32> = vec![
        -2.5, -1.0, -0.25, 0.0, 0.5, 1.0, 2.0, 3.5, 42.0, -100.0, 0.125, 255.0,
    ];
    let shape = [3i32, 4];
    for &dt in &[mlxcel_core::dtype::FLOAT16, mlxcel_core::dtype::BFLOAT16] {
        let orig = ffi::astype(&ffi::from_slice_f32(&vals, &shape), dt);
        ffi::eval(&orig);
        // Build the RawTensorData exactly as the serialize path does.
        let raw = RawTensorData {
            data: ffi::array_to_raw_bytes(&orig),
            shape: ffi::array_shape(&orig),
            dtype: ffi::array_dtype(&orig),
        };
        let recon = reconstruct_mlx_array(&raw).unwrap();
        ffi::eval(&recon);
        assert_eq!(ffi::array_dtype(&recon), dt, "dtype {dt} must be preserved");
        assert_eq!(
            ffi::array_to_raw_bytes(&recon),
            raw.data,
            "reconstruct_mlx_array must reproduce the exact 16-bit bytes (dtype {dt})"
        );
    }
}

#[test]
fn empty_kv_cache_extract_reconstruct() {
    let cache = mlxcel_core::cache::KVCache::new();

    let entry = extract_kv_cache_entry(&cache);
    assert!(entry.keys.is_none());
    assert!(entry.values.is_none());
}

// ---------------------------------------------------------------------------
// Full state serialization round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn full_state_serialize_deserialize_round_trip() {
    let cache = make_test_kv_cache(3, 1, 4);

    let entry = extract_kv_cache_entry(&cache);

    let state = SerializableCacheState {
        cache_type: CacheType::Standard,
        entries: vec![entry],
        metadata: CacheMetadata {
            prompt_len: 3,
            current_offset: 3,
            num_layers: 1,
            layer_offsets: vec![3],
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: Some(SerializableSamplingState {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            min_p: 0.0,
            seed: Some(42),
            repetition_penalty: 1.1,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 0,
            dry_sequence_breakers: vec![],
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token_ids: vec![2],
        }),
        token_history: vec![100, 200, 300],
        sequence_id: 42,
        sequence_backend: SerializableSequenceBackend::DenseKvCache,
        paged_state: None,
        paged_blocks: Vec::new(),
    };

    // Serialize
    let bytes = serialize_cache_state(&state).unwrap();
    assert!(!bytes.is_empty());

    // Verify header
    assert_eq!(bytes[0], CACHE_FORMAT_VERSION);
    assert_eq!(bytes[1], CacheType::Standard as u8);

    // Deserialize
    let recovered = deserialize_cache_state(&bytes).unwrap();
    assert_eq!(recovered.cache_type, CacheType::Standard);
    assert_eq!(recovered.metadata.prompt_len, 3);
    assert_eq!(recovered.metadata.current_offset, 3);
    assert_eq!(recovered.metadata.num_layers, 1);
    assert_eq!(recovered.metadata.layer_offsets, vec![3]);
    assert_eq!(recovered.token_history, vec![100, 200, 300]);
    assert_eq!(recovered.sequence_id, 42);

    // Verify sampling state
    let sampling = recovered.sampling_state.as_ref().unwrap();
    assert!((sampling.temperature - 0.7).abs() < 1e-6);
    assert_eq!(sampling.top_k, 50);
    assert_eq!(sampling.seed, Some(42));

    // Verify tensor data round-trip
    assert_eq!(recovered.entries.len(), 1);
    let entry = &recovered.entries[0];
    let k = reconstruct_mlx_array(entry.keys.as_ref().unwrap()).unwrap();
    let v = reconstruct_mlx_array(entry.values.as_ref().unwrap()).unwrap();
    assert_eq!(ffi::array_shape(&k), vec![1, 1, 3, 4]);
    assert_eq!(ffi::array_shape(&v), vec![1, 1, 3, 4]);
}

#[test]
fn multi_layer_state_round_trip() {
    let num_layers = 4;
    let mut entries = Vec::with_capacity(num_layers);
    let mut layer_offsets = Vec::with_capacity(num_layers);

    for _ in 0..num_layers {
        let cache = make_test_kv_cache(2, 1, 4);
        entries.push(extract_kv_cache_entry(&cache));
        layer_offsets.push(cache.offset);
    }

    let state = SerializableCacheState {
        cache_type: CacheType::Standard,
        entries,
        metadata: CacheMetadata {
            prompt_len: 2,
            current_offset: 2,
            num_layers,
            layer_offsets,
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: None,
        token_history: vec![],
        sequence_id: 7,
        sequence_backend: SerializableSequenceBackend::DenseKvCache,
        paged_state: None,
        paged_blocks: Vec::new(),
    };

    let bytes = serialize_cache_state(&state).unwrap();
    let recovered = deserialize_cache_state(&bytes).unwrap();

    assert_eq!(recovered.entries.len(), num_layers);
    assert_eq!(recovered.metadata.num_layers, num_layers);
    assert_eq!(recovered.sequence_id, 7);

    // Verify all layers have data
    for entry in &recovered.entries {
        assert!(entry.keys.is_some());
        assert!(entry.values.is_some());
        let k = reconstruct_mlx_array(entry.keys.as_ref().unwrap()).unwrap();
        assert_eq!(ffi::array_shape(&k), vec![1, 1, 2, 4]);
    }
}

#[test]
fn empty_cache_state_round_trip() {
    let empty_cache = mlxcel_core::cache::KVCache::new();
    let entry = extract_kv_cache_entry(&empty_cache);

    let state = SerializableCacheState {
        cache_type: CacheType::Standard,
        entries: vec![entry],
        metadata: CacheMetadata {
            prompt_len: 0,
            current_offset: 0,
            num_layers: 1,
            layer_offsets: vec![0],
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: None,
        token_history: vec![],
        sequence_id: 0,
        sequence_backend: SerializableSequenceBackend::DenseKvCache,
        paged_state: None,
        paged_blocks: Vec::new(),
    };

    let bytes = serialize_cache_state(&state).unwrap();
    let recovered = deserialize_cache_state(&bytes).unwrap();

    assert_eq!(recovered.entries.len(), 1);
    assert!(recovered.entries[0].keys.is_none());
    assert!(recovered.entries[0].values.is_none());
}

// ---------------------------------------------------------------------------
// Restore into live cache tests
// ---------------------------------------------------------------------------

#[test]
fn restore_into_kv_caches_round_trip() {
    // Create source cache with data
    let source = make_test_kv_cache(5, 2, 8);

    // Extract and build state
    let entry = extract_kv_cache_entry(&source);
    let state = SerializableCacheState {
        cache_type: CacheType::Standard,
        entries: vec![entry],
        metadata: CacheMetadata {
            prompt_len: 5,
            current_offset: 5,
            num_layers: 1,
            layer_offsets: vec![source.offset],
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: None,
        token_history: vec![],
        sequence_id: 1,
        sequence_backend: SerializableSequenceBackend::DenseKvCache,
        paged_state: None,
        paged_blocks: Vec::new(),
    };

    // Serialize and deserialize
    let bytes = serialize_cache_state(&state).unwrap();
    let recovered = deserialize_cache_state(&bytes).unwrap();

    // Restore into fresh cache
    let mut target = vec![mlxcel_core::cache::KVCache::new()];
    restore_into_kv_caches(&recovered, &mut target).unwrap();

    // Verify restored cache has correct offset
    assert_eq!(target[0].offset, source.offset);
    assert!(!target[0].is_empty());

    // Verify data is bit-exact
    let orig_k = source.keys.as_ref().unwrap();
    let restored_k = target[0].keys.as_ref().unwrap();

    // Slicing to the actual data region for comparison (buffer may be larger)
    let orig_shape = ffi::array_shape(orig_k);
    let restored_shape = ffi::array_shape(restored_k);

    // Use the minimum sequence length for comparison
    let orig_seq = orig_shape[2];
    let restored_seq = restored_shape[2];
    let cmp_seq = orig_seq.min(restored_seq);

    let orig_slice = ffi::slice(
        orig_k,
        &[0, 0, 0, 0],
        &[orig_shape[0], orig_shape[1], cmp_seq, orig_shape[3]],
    );
    let restored_slice = ffi::slice(
        restored_k,
        &[0, 0, 0, 0],
        &[
            restored_shape[0],
            restored_shape[1],
            cmp_seq,
            restored_shape[3],
        ],
    );

    let eq = ffi::array_equal(&orig_slice, &restored_slice, false);
    ffi::eval(&eq);
    assert!(ffi::item_bool(&eq), "restored cache data must be bit-exact");
}

#[test]
fn restore_layer_count_mismatch_fails() {
    let state = SerializableCacheState {
        cache_type: CacheType::Standard,
        entries: vec![
            SerializableCacheEntry {
                keys: None,
                values: None,
            },
            SerializableCacheEntry {
                keys: None,
                values: None,
            },
        ],
        metadata: CacheMetadata {
            prompt_len: 0,
            current_offset: 0,
            num_layers: 2,
            layer_offsets: vec![0, 0],
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: None,
        token_history: vec![],
        sequence_id: 0,
        sequence_backend: SerializableSequenceBackend::DenseKvCache,
        paged_state: None,
        paged_blocks: Vec::new(),
    };

    let mut target = vec![mlxcel_core::cache::KVCache::new()]; // only 1 layer
    let result = restore_into_kv_caches(&state, &mut target);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("layer count mismatch")
    );
}

// ---------------------------------------------------------------------------
// Deserialization error tests
// ---------------------------------------------------------------------------

#[test]
fn deserialize_too_short_buffer() {
    let result = deserialize_cache_state(&[0u8; 5]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too short"));
}

#[test]
fn deserialize_wrong_version() {
    let mut buf = vec![0u8; 20];
    buf[0] = 99; // wrong version
    let result = deserialize_cache_state(&buf);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("version"));
}

// ---------------------------------------------------------------------------
// MLX dtype conversion tests
// ---------------------------------------------------------------------------

#[test]
fn mlx_dtype_to_tensor_dtype_valid() {
    use crate::distributed::tensor_protocol::TensorDtype;

    assert_eq!(
        types::mlx_dtype_to_tensor_dtype(mlxcel_core::dtype::FLOAT32).unwrap(),
        TensorDtype::Float32
    );
    assert_eq!(
        types::mlx_dtype_to_tensor_dtype(mlxcel_core::dtype::FLOAT16).unwrap(),
        TensorDtype::Float16
    );
    assert_eq!(
        types::mlx_dtype_to_tensor_dtype(mlxcel_core::dtype::BFLOAT16).unwrap(),
        TensorDtype::BFloat16
    );
}

#[test]
fn mlx_dtype_to_tensor_dtype_invalid() {
    assert!(types::mlx_dtype_to_tensor_dtype(99).is_err());
}

// ---------------------------------------------------------------------------
// Rotating cache extraction test
// ---------------------------------------------------------------------------

#[test]
fn rotating_cache_extract_round_trip() {
    let mut cache = mlxcel_core::cache::RotatingKVCache::new(4);

    // Add some tokens
    for i in 0..3 {
        let k = ffi::from_slice_f32(&[i as f32], &[1, 1, 1, 1]);
        let v = ffi::from_slice_f32(&[(i as f32) * 10.0], &[1, 1, 1, 1]);
        cache.update_and_fetch(k, v);
    }

    let entry = extract_rotating_cache_entry(&cache);
    assert!(entry.keys.is_some());
    assert!(entry.values.is_some());

    // Reconstruct and verify shape
    let k = reconstruct_mlx_array(entry.keys.as_ref().unwrap()).unwrap();
    let shape = ffi::array_shape(&k);
    assert_eq!(shape[0], 1); // batch
    assert_eq!(shape[1], 1); // heads
    // seq_len may be up to max_size (4)
    assert!(shape[2] >= 3);
    assert_eq!(shape[3], 1); // head_dim
}

// ---------------------------------------------------------------------------
// validate_raw_tensor tests
// ---------------------------------------------------------------------------

#[test]
fn validate_raw_tensor_valid_float32() {
    let tensor = RawTensorData {
        data: vec![0u8; 4 * 2 * 3], // 6 f32 elements
        shape: vec![2, 3],
        dtype: mlxcel_core::dtype::FLOAT32,
    };
    assert!(validate_raw_tensor(&tensor).is_ok());
}

#[test]
fn validate_raw_tensor_size_mismatch() {
    let tensor = RawTensorData {
        data: vec![0u8; 10], // wrong size for 2x3 f32
        shape: vec![2, 3],
        dtype: mlxcel_core::dtype::FLOAT32,
    };
    let err = validate_raw_tensor(&tensor).unwrap_err();
    assert!(err.to_string().contains("mismatch"));
}

#[test]
fn validate_raw_tensor_negative_shape() {
    let tensor = RawTensorData {
        data: vec![],
        shape: vec![2, -1],
        dtype: mlxcel_core::dtype::FLOAT32,
    };
    let err = validate_raw_tensor(&tensor).unwrap_err();
    assert!(err.to_string().contains("non-negative"));
}

#[test]
fn validate_raw_tensor_scalar() {
    // Scalar tensor: shape=[], 1 element of f32 = 4 bytes
    let tensor = RawTensorData {
        data: vec![0u8; 4],
        shape: vec![],
        dtype: mlxcel_core::dtype::FLOAT32,
    };
    assert!(validate_raw_tensor(&tensor).is_ok());
}

// ---------------------------------------------------------------------------
// mlx_dtype_element_size tests
// ---------------------------------------------------------------------------

#[test]
fn mlx_dtype_element_size_known_types() {
    use crate::distributed::kv_cache_serde::types::mlx_dtype_element_size;

    assert_eq!(
        mlx_dtype_element_size(mlxcel_core::dtype::FLOAT32).unwrap(),
        4
    );
    assert_eq!(
        mlx_dtype_element_size(mlxcel_core::dtype::FLOAT16).unwrap(),
        2
    );
    // BFLOAT16 is a 16-bit type: 2 bytes. (#125 fixed `mlx_dtype_element_size`
    // to return 2 here so `validate_raw_tensor` stops rejecting valid bf16
    // payloads; this assertion was left stale at the old, wrong value 4.)
    assert_eq!(
        mlx_dtype_element_size(mlxcel_core::dtype::BFLOAT16).unwrap(),
        2
    );
    assert_eq!(mlx_dtype_element_size(0).unwrap(), 1); // BOOL
}

#[test]
fn mlx_dtype_element_size_unknown() {
    use crate::distributed::kv_cache_serde::types::mlx_dtype_element_size;
    assert!(mlx_dtype_element_size(99).is_err());
}

// ---------------------------------------------------------------------------
// restore_into_sequence_cache_set test
// ---------------------------------------------------------------------------

#[test]
fn restore_into_sequence_cache_set_round_trip() {
    let source = make_test_kv_cache(4, 2, 8);
    let entry = extract_kv_cache_entry(&source);

    let state = SerializableCacheState {
        cache_type: CacheType::Standard,
        entries: vec![entry],
        metadata: CacheMetadata {
            prompt_len: 4,
            current_offset: 4,
            num_layers: 1,
            layer_offsets: vec![source.offset],
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: None,
        token_history: vec![],
        sequence_id: 99,
        sequence_backend: SerializableSequenceBackend::DenseKvCache,
        paged_state: None,
        paged_blocks: Vec::new(),
    };

    let bytes = serialize_cache_state(&state).unwrap();
    let recovered = deserialize_cache_state(&bytes).unwrap();

    // Build a SequenceCacheSet with one layer (direct construction for test use)
    let seq_id = mlxcel_core::cache::SequenceId::from_raw(99);
    let mut cache_set = mlxcel_core::cache::SequenceCacheSet::dense_external(
        seq_id,
        vec![mlxcel_core::cache::KVCache::new()],
    );

    restore_into_sequence_cache_set(&recovered, &mut cache_set).unwrap();

    assert_eq!(cache_set.prompt_len, 4);
    assert_eq!(cache_set.current_offset, 4);
    assert!(!cache_set.caches[0].is_empty());
    assert_eq!(cache_set.caches[0].offset, source.offset);
}

#[test]
fn deserialize_v1_payload_defaults_to_dense_backend() {
    let metadata_json = serde_json::json!({
        "cache_type": CacheType::Standard,
        "entries": [],
        "metadata": {
            "prompt_len": 0,
            "current_offset": 0,
            "num_layers": 0,
            "layer_offsets": [],
            "max_size": null,
            "layer_indices": null,
            "chunk_size": null,
            "start_positions": null
        },
        "sampling_state": null,
        "token_history": [],
        "sequence_id": 5
    });
    let metadata_bytes = serde_json::to_vec(&metadata_json).unwrap();

    let mut buf = Vec::new();
    buf.push(CACHE_FORMAT_VERSION_V1);
    buf.push(CacheType::Standard as u8);
    buf.extend_from_slice(&(0u32).to_le_bytes());
    buf.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&metadata_bytes);

    let recovered = deserialize_cache_state(&buf).unwrap();
    assert_eq!(
        recovered.sequence_backend,
        SerializableSequenceBackend::DenseKvCache
    );
    assert!(recovered.paged_state.is_none());
}

#[test]
fn restore_into_sequence_cache_set_restores_paged_state() {
    let layout = mlxcel_core::cache::PagedKvLayout::uniform(2, 4, 128).unwrap();
    let state = SerializableCacheState {
        cache_type: CacheType::Standard,
        entries: vec![],
        metadata: CacheMetadata {
            prompt_len: 6,
            current_offset: 6,
            num_layers: 0,
            layer_offsets: vec![],
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: None,
        token_history: vec![],
        sequence_id: 777,
        sequence_backend: SerializableSequenceBackend::PagedKvCache,
        paged_state: Some(SerializablePagedSequenceState {
            block_size: 4,
            bytes_per_block: vec![128, 128],
            layers: vec![
                SerializablePagedLayerState {
                    block_ids: vec![10, 11],
                    len: 6,
                    logical_start: 0,
                },
                SerializablePagedLayerState {
                    block_ids: vec![20],
                    len: 3,
                    logical_start: 1,
                },
            ],
        }),
        paged_blocks: Vec::new(),
    };

    let mut cache_set = mlxcel_core::cache::SequenceCacheSet::paged(
        mlxcel_core::cache::SequenceId::from_raw(777),
        layout,
    );
    restore_into_sequence_cache_set(&state, &mut cache_set).unwrap();

    let paged = cache_set.paged_state().unwrap();
    assert_eq!(paged.block_size, 4);
    assert_eq!(paged.layers[0].block_ids[0].as_u64(), 10);
    assert_eq!(paged.layers[0].block_ids[1].as_u64(), 11);
    assert_eq!(paged.layers[1].block_ids[0].as_u64(), 20);
    assert_eq!(paged.layers[1].len, 3);
    assert_eq!(paged.layers[1].logical_start, 1);
    assert_eq!(cache_set.prompt_len, 6);
    assert_eq!(cache_set.current_offset, 6);
}

// ---------------------------------------------------------------------------
// Chunked cache extraction test
// ---------------------------------------------------------------------------

#[test]
fn chunked_cache_extract_round_trip() {
    let mut cache = mlxcel_core::cache::ChunkedKVCache::new(8);

    let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 1, 3, 1]);
    let values = ffi::from_slice_f32(&[4.0, 5.0, 6.0], &[1, 1, 3, 1]);
    cache.update_and_fetch(keys, values);

    let entry = extract_chunked_cache_entry(&cache);
    assert!(entry.keys.is_some());
    assert!(entry.values.is_some());

    let k = reconstruct_mlx_array(entry.keys.as_ref().unwrap()).unwrap();
    let shape = ffi::array_shape(&k);
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 1);
    assert!(shape[2] >= 3);
    assert_eq!(shape[3], 1);
}

// ---------------------------------------------------------------------------
// Untrusted-network handoff hardening tests (#126): frame caps, model-geometry
// anchor, table <-> contents layer consistency, and empty-sequence routing.
// These exercise `validate_paged_payload` / `deserialize_cache_state_with_limits`
// over synthetic payloads, so they need no Metal device.
// ---------------------------------------------------------------------------

const HARDEN_BS: usize = 4;
const HARDEN_H: i32 = 2;
const HARDEN_D: i32 = 8;

fn harden_slab() -> RawTensorData {
    let dt = mlxcel_core::dtype::FLOAT16;
    let elements = HARDEN_BS * HARDEN_H as usize * HARDEN_D as usize;
    let bytes = elements * types::mlx_dtype_element_size(dt).unwrap();
    RawTensorData {
        data: vec![0u8; bytes],
        shape: vec![HARDEN_BS as i32, HARDEN_H, HARDEN_D],
        dtype: dt,
    }
}

fn harden_block(block_id: u64, layer_idx: usize) -> SerializablePagedBlock {
    SerializablePagedBlock {
        block_id,
        layer_idx,
        keys: harden_slab(),
        values: harden_slab(),
    }
}

fn harden_expected() -> ExpectedBlockGeometry {
    ExpectedBlockGeometry {
        num_layers: 2,
        block_size: HARDEN_BS,
        n_kv_heads: HARDEN_H,
        head_dim: HARDEN_D,
        dtype: mlxcel_core::dtype::FLOAT16,
    }
}

/// Coherent 2-layer pool-backed handoff: layer 0 references blocks 10 and 11
/// (len 6 over block_size 4 = 2 blocks), layer 1 references block 20 (len 3 = 1
/// block). Each referenced block has matching contents on its own layer.
fn coherent_paged_payload() -> SerializableCacheState {
    SerializableCacheState {
        cache_type: CacheType::Standard,
        entries: vec![],
        metadata: CacheMetadata {
            prompt_len: 6,
            current_offset: 6,
            num_layers: 2,
            layer_offsets: vec![],
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: None,
        token_history: vec![],
        sequence_id: 1,
        sequence_backend: SerializableSequenceBackend::PagedKvCache,
        paged_state: Some(SerializablePagedSequenceState {
            block_size: HARDEN_BS,
            bytes_per_block: vec![HARDEN_BS, HARDEN_BS],
            layers: vec![
                SerializablePagedLayerState {
                    block_ids: vec![10, 11],
                    len: 6,
                    logical_start: 0,
                },
                SerializablePagedLayerState {
                    block_ids: vec![20],
                    len: 3,
                    logical_start: 0,
                },
            ],
        }),
        paged_blocks: vec![
            harden_block(10, 0),
            harden_block(11, 0),
            harden_block(20, 1),
        ],
    }
}

#[test]
fn validate_paged_payload_accepts_coherent_handoff() {
    let payload = coherent_paged_payload();
    validate_paged_payload(&payload, &CacheIngestLimits::default(), None).unwrap();
    validate_paged_payload(
        &payload,
        &CacheIngestLimits::default(),
        Some(&harden_expected()),
    )
    .unwrap();
}

#[test]
fn validate_paged_payload_passes_dense_payload_unchanged() {
    let mut payload = coherent_paged_payload();
    payload.sequence_backend = SerializableSequenceBackend::DenseKvCache;
    payload.paged_state = None;
    payload.paged_blocks = Vec::new();
    validate_paged_payload(&payload, &CacheIngestLimits::default(), None).unwrap();
}

#[test]
fn validate_paged_payload_rejects_geometry_shape_mismatch() {
    let payload = coherent_paged_payload();
    let mut expected = harden_expected();
    expected.head_dim = HARDEN_D + 1;
    let err = validate_paged_payload(&payload, &CacheIngestLimits::default(), Some(&expected))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("does not match decode model geometry"),
        "{err}"
    );
}

#[test]
fn validate_paged_payload_rejects_geometry_dtype_mismatch() {
    let payload = coherent_paged_payload();
    let mut expected = harden_expected();
    expected.dtype = mlxcel_core::dtype::BFLOAT16;
    let err = validate_paged_payload(&payload, &CacheIngestLimits::default(), Some(&expected))
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not match decode model dtype"), "{err}");
}

#[test]
fn validate_paged_payload_rejects_block_count_over_limit() {
    let payload = coherent_paged_payload();
    let limits = CacheIngestLimits {
        max_paged_blocks: 2,
        ..CacheIngestLimits::default()
    };
    let err = validate_paged_payload(&payload, &limits, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds limit"), "{err}");
}

#[test]
fn validate_paged_payload_rejects_oversized_slab() {
    let payload = coherent_paged_payload();
    let limits = CacheIngestLimits {
        max_block_slab_bytes: 1,
        ..CacheIngestLimits::default()
    };
    let err = validate_paged_payload(&payload, &limits, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds limit"), "{err}");
}

#[test]
fn validate_paged_payload_rejects_layer_out_of_range() {
    let mut payload = coherent_paged_payload();
    // A block declaring a layer beyond the table's layer count.
    payload.paged_blocks.push(harden_block(99, 7));
    payload
        .paged_state
        .as_mut()
        .unwrap()
        .layers
        .push(SerializablePagedLayerState {
            block_ids: vec![99],
            len: 1,
            logical_start: 0,
        });
    let err = validate_paged_payload(&payload, &CacheIngestLimits::default(), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("out of range"), "{err}");
}

#[test]
fn validate_paged_payload_rejects_missing_block_contents() {
    let mut payload = coherent_paged_payload();
    // Drop block 20's contents while the layer-1 table still references it.
    payload.paged_blocks.retain(|b| b.block_id != 20);
    let err = validate_paged_payload(&payload, &CacheIngestLimits::default(), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no transferred contents"), "{err}");
}

#[test]
fn validate_paged_payload_rejects_cross_layer_block() {
    let mut payload = coherent_paged_payload();
    // Block 10 is referenced by layer 0, but its contents claim layer 1.
    for block in &mut payload.paged_blocks {
        if block.block_id == 10 {
            block.layer_idx = 1;
        }
    }
    let err = validate_paged_payload(&payload, &CacheIngestLimits::default(), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("declare layer"), "{err}");
}

#[test]
fn validate_paged_payload_rejects_orphan_contents() {
    let mut payload = coherent_paged_payload();
    // A block with contents that no layer table references (a pool-block leak).
    payload.paged_blocks.push(harden_block(404, 0));
    let err = validate_paged_payload(&payload, &CacheIngestLimits::default(), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not referenced by any layer table"), "{err}");
}

#[test]
fn validate_paged_payload_rejects_metadata_only_handoff() {
    let mut payload = coherent_paged_payload();
    // Table still references blocks, but no contents are carried.
    payload.paged_blocks = Vec::new();
    let err = validate_paged_payload(&payload, &CacheIngestLimits::default(), None)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("metadata-only paged restore is invalid"),
        "{err}"
    );
}

#[test]
fn validate_paged_payload_rejects_empty_block_table() {
    let mut payload = coherent_paged_payload();
    payload.paged_blocks = Vec::new();
    for layer in &mut payload.paged_state.as_mut().unwrap().layers {
        layer.block_ids.clear();
        layer.len = 0;
    }
    let err = validate_paged_payload(&payload, &CacheIngestLimits::default(), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty block table"), "{err}");
}

#[test]
fn validate_paged_payload_rejects_paged_backend_without_metadata() {
    let mut payload = coherent_paged_payload();
    payload.paged_state = None;
    payload.paged_blocks = Vec::new();
    let err = validate_paged_payload(&payload, &CacheIngestLimits::default(), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing paged_state metadata"), "{err}");
}

#[test]
fn deserialize_cache_state_rejects_oversized_frame() {
    // A well-formed dense frame, then a limit below its size.
    let state = SerializableCacheState {
        cache_type: CacheType::Standard,
        entries: vec![],
        metadata: CacheMetadata {
            prompt_len: 0,
            current_offset: 0,
            num_layers: 0,
            layer_offsets: vec![],
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: None,
        token_history: vec![],
        sequence_id: 1,
        sequence_backend: SerializableSequenceBackend::DenseKvCache,
        paged_state: None,
        paged_blocks: Vec::new(),
    };
    let bytes = serialize_cache_state(&state).unwrap();
    let limits = CacheIngestLimits {
        max_frame_bytes: bytes.len() - 1,
        ..CacheIngestLimits::default()
    };
    let err = deserialize_cache_state_with_limits(&bytes, &limits)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds limit"), "{err}");
    // The same bytes deserialize fine under the default cap.
    deserialize_cache_state(&bytes).unwrap();
}

#[test]
fn deserialize_cache_state_rejects_forged_metadata_length() {
    // Header claims a metadata length far beyond the frame cap.
    let mut buf = vec![0u8; 10];
    buf[0] = CACHE_FORMAT_VERSION;
    buf[1] = CacheType::Standard as u8;
    buf[2..6].copy_from_slice(&0u32.to_le_bytes());
    buf[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
    let limits = CacheIngestLimits {
        max_frame_bytes: 4096,
        ..CacheIngestLimits::default()
    };
    let err = deserialize_cache_state_with_limits(&buf, &limits)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds frame limit"), "{err}");
}
