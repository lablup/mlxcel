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

#[test]
fn test_tensor_flags_default() {
    let flags = TensorFlags::new();
    assert_eq!(flags.bits(), 0);
    assert!(!flags.is_compressed());
    assert!(!flags.is_quantized());
    assert!(!flags.is_chunked());
}

#[test]
fn test_tensor_flags_set_and_check() {
    let mut flags = TensorFlags::new();
    flags.set(TensorFlags::COMPRESSED);
    assert!(flags.is_compressed());
    assert!(!flags.is_quantized());

    flags.set(TensorFlags::QUANTIZED);
    assert!(flags.is_compressed());
    assert!(flags.is_quantized());
    assert!(!flags.is_chunked());

    flags.set(TensorFlags::CHUNKED);
    assert!(flags.is_chunked());
    assert_eq!(flags.bits(), 0b0000_0111);
}

#[test]
fn test_tensor_kind_roundtrip() {
    for (val, expected) in [
        (0u8, TensorKind::KVCache),
        (1, TensorKind::Activation),
        (2, TensorKind::WeightShard),
    ] {
        let kind = TensorKind::try_from(val).unwrap();
        assert_eq!(kind, expected);
        assert_eq!(kind as u8, val);
    }
}

#[test]
fn test_tensor_kind_invalid() {
    assert!(TensorKind::try_from(42u8).is_err());
}

#[test]
fn test_tensor_dtype_element_size() {
    assert_eq!(TensorDtype::Float32.element_size(), 4);
    assert_eq!(TensorDtype::Float16.element_size(), 2);
    assert_eq!(TensorDtype::BFloat16.element_size(), 2);
    assert_eq!(TensorDtype::Int8.element_size(), 1);
    assert_eq!(TensorDtype::Int4.element_size(), 0);
}

#[test]
fn test_tensor_dtype_roundtrip() {
    for val in 0..=8u8 {
        let dtype = TensorDtype::try_from(val).unwrap();
        assert_eq!(dtype as u8, val);
    }
}

#[test]
fn test_header_encode_decode_roundtrip() {
    let header = TensorHeader {
        version: PROTOCOL_VERSION,
        flags: TensorFlags::new(),
        kind: TensorKind::Activation,
        dtype: TensorDtype::Float16,
        shape: vec![2, 128, 64],
        metadata: b"{}".to_vec(),
        data_len: 32768,
    };

    let encoded = header.encode();
    let (decoded, consumed) = TensorHeader::decode(&encoded).unwrap();

    assert_eq!(consumed, encoded.len());
    assert_eq!(decoded, header);
}

#[test]
fn test_header_with_empty_metadata() {
    let header = TensorHeader {
        version: PROTOCOL_VERSION,
        flags: TensorFlags::new(),
        kind: TensorKind::KVCache,
        dtype: TensorDtype::Float32,
        shape: vec![1, 256],
        metadata: vec![],
        data_len: 1024,
    };

    let encoded = header.encode();
    let (decoded, _) = TensorHeader::decode(&encoded).unwrap();
    assert_eq!(decoded.metadata.len(), 0);
    assert_eq!(decoded, header);
}

#[test]
fn test_header_with_flags() {
    let mut flags = TensorFlags::new();
    flags.set(TensorFlags::COMPRESSED);
    flags.set(TensorFlags::QUANTIZED);

    let header = TensorHeader {
        version: PROTOCOL_VERSION,
        flags,
        kind: TensorKind::WeightShard,
        dtype: TensorDtype::Float16,
        shape: vec![4096, 4096],
        metadata: b"{\"group_size\":128}".to_vec(),
        data_len: 8_000_000,
    };

    let encoded = header.encode();
    let (decoded, _) = TensorHeader::decode(&encoded).unwrap();
    assert!(decoded.flags.is_compressed());
    assert!(decoded.flags.is_quantized());
    assert!(!decoded.flags.is_chunked());
    assert_eq!(decoded, header);
}

#[test]
fn test_header_decode_wrong_version() {
    let mut encoded = TensorHeader {
        version: PROTOCOL_VERSION,
        flags: TensorFlags::new(),
        kind: TensorKind::Activation,
        dtype: TensorDtype::Float32,
        shape: vec![1],
        metadata: vec![],
        data_len: 4,
    }
    .encode();

    // Corrupt version byte.
    encoded[0] = 99;
    assert!(TensorHeader::decode(&encoded).is_err());
}

#[test]
fn test_header_decode_truncated() {
    assert!(TensorHeader::decode(&[1, 0, 0]).is_err());
}

#[test]
fn test_header_num_elements() {
    let header = TensorHeader {
        version: PROTOCOL_VERSION,
        flags: TensorFlags::new(),
        kind: TensorKind::Activation,
        dtype: TensorDtype::Float16,
        shape: vec![2, 3, 4],
        metadata: vec![],
        data_len: 48,
    };
    assert_eq!(header.num_elements(), 24);
}

#[test]
fn test_chunk_frame_roundtrip() {
    let frame = ChunkFrame {
        chunk_index: 3,
        total_chunks: 10,
        data: vec![0xAB; 256],
    };
    let encoded = frame.encode();
    let decoded = ChunkFrame::decode(&encoded).unwrap();
    assert_eq!(decoded.chunk_index, 3);
    assert_eq!(decoded.total_chunks, 10);
    assert_eq!(decoded.data, vec![0xAB; 256]);
}

#[test]
fn test_chunk_frame_decode_truncated() {
    assert!(ChunkFrame::decode(&[0; 8]).is_err());
}

#[test]
fn test_tensor_kind_display() {
    assert_eq!(format!("{}", TensorKind::KVCache), "KVCache");
    assert_eq!(format!("{}", TensorKind::Activation), "Activation");
    assert_eq!(format!("{}", TensorKind::WeightShard), "WeightShard");
}

#[test]
fn test_tensor_dtype_display() {
    assert_eq!(format!("{}", TensorDtype::Float32), "float32");
    assert_eq!(format!("{}", TensorDtype::Float16), "float16");
    assert_eq!(format!("{}", TensorDtype::Int4), "int4");
}
