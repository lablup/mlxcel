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

use super::super::tensor_protocol::{TensorDtype, TensorKind};
use super::*;

#[test]
fn test_split_into_chunks_basic() {
    let data = vec![0xABu8; 4096];
    let config = ChunkedTransferConfig {
        chunk_size: 1024,
        compress_chunks: false,
    };

    let chunked = split_into_chunks(
        TensorDtype::Float32,
        &[1024],
        &data,
        TensorKind::WeightShard,
        &config,
    )
    .unwrap();

    assert_eq!(chunked.num_chunks(), 4);
    assert_eq!(chunked.total_data_bytes(), 4096);
    assert!(chunked.header.flags.is_chunked());

    for (i, chunk) in chunked.chunks.iter().enumerate() {
        assert_eq!(chunk.chunk_index, i as u32);
        assert_eq!(chunk.total_chunks, 4);
        assert_eq!(chunk.data.len(), 1024);
    }
}

#[test]
fn test_split_into_chunks_non_aligned() {
    let data = vec![0xCDu8; 3000];
    let config = ChunkedTransferConfig {
        chunk_size: 1024,
        compress_chunks: false,
    };

    let chunked = split_into_chunks(
        TensorDtype::Float16,
        &[1500],
        &data,
        TensorKind::Activation,
        &config,
    )
    .unwrap();

    assert_eq!(chunked.num_chunks(), 3);
    assert_eq!(chunked.chunks[0].data.len(), 1024);
    assert_eq!(chunked.chunks[1].data.len(), 1024);
    assert_eq!(chunked.chunks[2].data.len(), 952);
    assert_eq!(chunked.total_data_bytes(), 3000);
}

#[test]
fn test_split_single_chunk() {
    let data = vec![0xEFu8; 512];
    let config = ChunkedTransferConfig {
        chunk_size: 1024,
        compress_chunks: false,
    };

    let chunked = split_into_chunks(
        TensorDtype::Int8,
        &[512],
        &data,
        TensorKind::KVCache,
        &config,
    )
    .unwrap();

    assert_eq!(chunked.num_chunks(), 1);
    assert_eq!(chunked.chunks[0].data.len(), 512);
}

#[test]
fn test_split_empty_data() {
    let config = ChunkedTransferConfig {
        chunk_size: 1024,
        compress_chunks: false,
    };

    let chunked = split_into_chunks(
        TensorDtype::Float32,
        &[0],
        &[],
        TensorKind::Activation,
        &config,
    )
    .unwrap();

    assert_eq!(chunked.num_chunks(), 1);
    assert_eq!(chunked.total_data_bytes(), 0);
}

#[test]
fn test_split_zero_chunk_size() {
    let config = ChunkedTransferConfig {
        chunk_size: 0,
        compress_chunks: false,
    };

    let result = split_into_chunks(
        TensorDtype::Float32,
        &[4],
        &[0; 16],
        TensorKind::Activation,
        &config,
    );
    assert!(result.is_err());
}

#[test]
fn test_assembler_basic() {
    let data = vec![0xABu8; 4096];
    let config = ChunkedTransferConfig {
        chunk_size: 1024,
        compress_chunks: false,
    };

    let chunked = split_into_chunks(
        TensorDtype::Float32,
        &[1024],
        &data,
        TensorKind::WeightShard,
        &config,
    )
    .unwrap();

    let mut assembler = ChunkAssembler::new(chunked.header);

    // Insert chunks in order.
    for (i, chunk) in chunked.chunks.into_iter().enumerate() {
        let complete = assembler.insert(chunk).unwrap();
        if i < 3 {
            assert!(!complete);
        } else {
            assert!(complete);
        }
    }

    assert!(assembler.is_complete());
    let (_header, assembled_data) = assembler.assemble().unwrap();
    assert_eq!(assembled_data, data);
}

#[test]
fn test_assembler_out_of_order() {
    let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let config = ChunkedTransferConfig {
        chunk_size: 1024,
        compress_chunks: false,
    };

    let chunked = split_into_chunks(
        TensorDtype::UInt8,
        &[4096],
        &data,
        TensorKind::Activation,
        &config,
    )
    .unwrap();

    let mut assembler = ChunkAssembler::new(chunked.header);
    let mut chunks = chunked.chunks;

    // Insert in reverse order.
    chunks.reverse();
    for chunk in chunks {
        let _ = assembler.insert(chunk).unwrap();
    }

    assert!(assembler.is_complete());
    let (_, assembled_data) = assembler.assemble().unwrap();
    assert_eq!(assembled_data, data);
}

#[test]
fn test_assembler_progress() {
    let data = vec![0u8; 4096];
    let config = ChunkedTransferConfig {
        chunk_size: 1024,
        compress_chunks: false,
    };

    let chunked = split_into_chunks(
        TensorDtype::Float32,
        &[1024],
        &data,
        TensorKind::Activation,
        &config,
    )
    .unwrap();

    let mut assembler = ChunkAssembler::new(chunked.header);
    assert_eq!(assembler.progress(), 0.0);

    assembler.insert(chunked.chunks[0].clone()).unwrap();
    assert!((assembler.progress() - 0.25).abs() < 0.01);

    assembler.insert(chunked.chunks[1].clone()).unwrap();
    assert!((assembler.progress() - 0.50).abs() < 0.01);
}

#[test]
fn test_assembler_duplicate_chunk() {
    let data = vec![0u8; 2048];
    let config = ChunkedTransferConfig {
        chunk_size: 1024,
        compress_chunks: false,
    };

    let chunked = split_into_chunks(
        TensorDtype::Float32,
        &[512],
        &data,
        TensorKind::Activation,
        &config,
    )
    .unwrap();

    let mut assembler = ChunkAssembler::new(chunked.header);
    assembler.insert(chunked.chunks[0].clone()).unwrap();
    assembler.insert(chunked.chunks[0].clone()).unwrap(); // Duplicate.
    assert_eq!(assembler.received_count(), 1); // Should not double-count.
}

#[test]
fn test_assembler_incomplete_assemble() {
    let data = vec![0u8; 2048];
    let config = ChunkedTransferConfig {
        chunk_size: 1024,
        compress_chunks: false,
    };

    let chunked = split_into_chunks(
        TensorDtype::Float32,
        &[512],
        &data,
        TensorKind::Activation,
        &config,
    )
    .unwrap();

    let mut assembler = ChunkAssembler::new(chunked.header);
    assembler.insert(chunked.chunks[0].clone()).unwrap();

    // Assemble without all chunks should fail.
    let result = assembler.assemble();
    assert!(result.is_err());
}

#[test]
fn test_should_chunk() {
    assert!(!should_chunk(512, 1024));
    assert!(!should_chunk(1024, 1024));
    assert!(should_chunk(1025, 1024));
    assert!(should_chunk(1_000_000, 1024));
}

#[test]
fn test_chunk_frame_encode_decode_roundtrip() {
    let frame = ChunkFrame {
        chunk_index: 7,
        total_chunks: 20,
        data: vec![0xDE, 0xAD, 0xBE, 0xEF],
    };
    let encoded = frame.encode();
    let decoded = ChunkFrame::decode(&encoded).unwrap();
    assert_eq!(decoded.chunk_index, 7);
    assert_eq!(decoded.total_chunks, 20);
    assert_eq!(decoded.data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
}
