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

//! Tests for collective communication primitives.
//!
//! Uses in-process channels to simulate multi-rank communication without
//! network. Each rank runs on its own thread; paired send/recv is performed
//! through std mpsc channel pairs.

use std::sync::Arc;
use std::thread;

use super::*;

// ---------------------------------------------------------------------------
// Byte conversion helpers for tests
// ---------------------------------------------------------------------------

fn f32_vec_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i32_vec_to_bytes(vals: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_i32_vec(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// In-process multi-rank test harness
// ---------------------------------------------------------------------------

/// Build a set of `CollectiveGroup` instances connected by in-process channels.
///
/// Each group's exchange function sends to the target rank's inbox and receives
/// from its own inbox using std mpsc channels.
fn build_test_groups(world_size: usize) -> Vec<CollectiveGroup> {
    use std::sync::mpsc;

    // Create channel pairs: senders[i] delivers data INTO rank i's inbox.
    let mut senders: Vec<mpsc::Sender<Vec<u8>>> = Vec::with_capacity(world_size);
    let mut receivers: Vec<mpsc::Receiver<Vec<u8>>> = Vec::with_capacity(world_size);

    for _ in 0..world_size {
        let (tx, rx) = mpsc::channel();
        senders.push(tx);
        receivers.push(rx);
    }

    let senders: Vec<Arc<mpsc::Sender<Vec<u8>>>> = senders.into_iter().map(Arc::new).collect();

    // Pop receivers in forward order by removing from front via drain.
    let mut rx_vec: Vec<_> = receivers.into_iter().map(Some).collect();

    let mut groups = Vec::with_capacity(world_size);
    for (rank, rx_slot) in rx_vec.iter_mut().enumerate().take(world_size) {
        let config = CollectiveConfig {
            rank,
            world_size,
            chunk_size: 1024 * 1024,
        };

        let my_rx = Arc::new(std::sync::Mutex::new(rx_slot.take().unwrap()));
        let all_senders = senders.clone();

        let exchange_fn = Arc::new(
            move |target_rank: usize, data: Vec<u8>| -> Result<Vec<u8>> {
                all_senders[target_rank]
                    .send(data)
                    .map_err(|e| anyhow::anyhow!("send failed: {e}"))?;
                let received = my_rx
                    .lock()
                    .map_err(|e| anyhow::anyhow!("lock failed: {e}"))?
                    .recv()
                    .map_err(|e| anyhow::anyhow!("recv failed: {e}"))?;
                Ok(received)
            },
        );

        let group = CollectiveGroup::new(config, exchange_fn).expect("group creation failed");
        groups.push(group);
    }

    groups
}

/// Run a closure on each rank in parallel and collect results.
fn run_parallel<F, T>(groups: Vec<CollectiveGroup>, f: F) -> Vec<(usize, T)>
where
    F: Fn(CollectiveGroup) -> T + Send + Sync + Clone + 'static,
    T: Send + 'static,
{
    let n = groups.len();
    let handles: Vec<_> = groups
        .into_iter()
        .enumerate()
        .map(|(i, group)| {
            let f = f.clone();
            thread::spawn(move || (i, f(group)))
        })
        .collect();

    let mut results: Vec<Option<(usize, T)>> = (0..n).map(|_| None).collect();
    for h in handles {
        let (i, val) = h.join().expect("thread panicked");
        results[i] = Some((i, val));
    }
    results.into_iter().map(|r| r.unwrap()).collect()
}

// ---------------------------------------------------------------------------
// RingTopology tests
// ---------------------------------------------------------------------------

#[test]
fn test_ring_topology_neighbors() {
    let ring = RingTopology::new(0, 4);
    assert_eq!(ring.send_to(), 1);
    assert_eq!(ring.recv_from(), 3);

    let ring = RingTopology::new(3, 4);
    assert_eq!(ring.send_to(), 0);
    assert_eq!(ring.recv_from(), 2);

    let ring = RingTopology::new(0, 2);
    assert_eq!(ring.send_to(), 1);
    assert_eq!(ring.recv_from(), 1);
}

#[test]
fn test_ring_topology_single_rank() {
    let ring = RingTopology::new(0, 1);
    assert_eq!(ring.send_to(), 0);
    assert_eq!(ring.recv_from(), 0);
}

// ---------------------------------------------------------------------------
// CollectiveConfig validation
// ---------------------------------------------------------------------------

#[test]
fn test_config_validation_ok() {
    let cfg = CollectiveConfig {
        rank: 0,
        world_size: 4,
        chunk_size: 4096,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_validation_rank_out_of_range() {
    let cfg = CollectiveConfig {
        rank: 4,
        world_size: 4,
        chunk_size: 4096,
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_zero_world() {
    let cfg = CollectiveConfig {
        rank: 0,
        world_size: 0,
        chunk_size: 4096,
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_zero_chunk() {
    let cfg = CollectiveConfig {
        rank: 0,
        world_size: 2,
        chunk_size: 0,
    };
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// elementwise_add tests
// ---------------------------------------------------------------------------

#[test]
fn test_elementwise_add_f32() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let mut dst = f32_vec_to_bytes(&a);
    let src = f32_vec_to_bytes(&b);
    elementwise_add_inplace(&mut dst, &src, TensorDtype::Float32).unwrap();
    let result = bytes_to_f32_vec(&dst);
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_elementwise_add_f16() {
    // Use the module's own f16 conversion helpers.
    let a = f32_to_f16(1.0);
    let b = f32_to_f16(2.0);
    let mut dst = Vec::new();
    dst.extend_from_slice(&a.to_le_bytes());
    dst.extend_from_slice(&a.to_le_bytes());
    let mut src = Vec::new();
    src.extend_from_slice(&b.to_le_bytes());
    src.extend_from_slice(&b.to_le_bytes());

    elementwise_add_inplace(&mut dst, &src, TensorDtype::Float16).unwrap();

    let r0 = f16_to_f32(u16::from_le_bytes([dst[0], dst[1]]));
    let r1 = f16_to_f32(u16::from_le_bytes([dst[2], dst[3]]));
    assert!((r0 - 3.0).abs() < 0.01);
    assert!((r1 - 3.0).abs() < 0.01);
}

#[test]
fn test_elementwise_add_bf16() {
    let a = f32_to_bf16(1.5);
    let b = f32_to_bf16(2.5);
    let mut dst = a.to_le_bytes().to_vec();
    let src = b.to_le_bytes().to_vec();

    elementwise_add_inplace(&mut dst, &src, TensorDtype::BFloat16).unwrap();

    let result = bf16_to_f32(u16::from_le_bytes([dst[0], dst[1]]));
    assert!((result - 4.0).abs() < 0.1);
}

#[test]
fn test_elementwise_add_length_mismatch() {
    let mut dst = vec![0u8; 8];
    let src = vec![0u8; 12];
    assert!(elementwise_add_inplace(&mut dst, &src, TensorDtype::Float32).is_err());
}

// ---------------------------------------------------------------------------
// Chunk helpers
// ---------------------------------------------------------------------------

#[test]
fn test_split_and_reassemble_even() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let bytes = f32_vec_to_bytes(&data);
    let chunks = split_into_chunks(&bytes, 3, 4).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].len(), 16); // 4 floats * 4 bytes
    assert_eq!(chunks[1].len(), 16);
    assert_eq!(chunks[2].len(), 16);
    let reassembled = reassemble_chunks(&chunks);
    assert_eq!(reassembled, bytes);
}

#[test]
fn test_split_and_reassemble_uneven() {
    // 7 elements, 3 chunks: base=2, remainder=1 -> 3, 2, 2
    let data: Vec<f32> = (0..7).map(|i| i as f32).collect();
    let bytes = f32_vec_to_bytes(&data);
    let chunks = split_into_chunks(&bytes, 3, 4).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].len(), 12); // 3 * 4
    assert_eq!(chunks[1].len(), 8); // 2 * 4
    assert_eq!(chunks[2].len(), 8); // 2 * 4
    let reassembled = reassemble_chunks(&chunks);
    assert_eq!(reassembled, bytes);
}

// ---------------------------------------------------------------------------
// Single-rank collective tests (no-op)
// ---------------------------------------------------------------------------

#[test]
fn test_all_reduce_single_rank() {
    let group = CollectiveGroup::single_rank().unwrap();
    let original: Vec<f32> = vec![1.0, 2.0, 3.0];
    let mut data = f32_vec_to_bytes(&original);
    all_reduce_sum(&mut data, TensorDtype::Float32, &group).unwrap();
    let result = bytes_to_f32_vec(&data);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_all_gather_single_rank() {
    let group = CollectiveGroup::single_rank().unwrap();
    let shard: Vec<f32> = vec![1.0, 2.0];
    let bytes = f32_vec_to_bytes(&shard);
    let result = all_gather(&bytes, TensorDtype::Float32, &group).unwrap();
    assert_eq!(result, bytes);
}

#[test]
fn test_reduce_scatter_single_rank() {
    let group = CollectiveGroup::single_rank().unwrap();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes = f32_vec_to_bytes(&data);
    let result = reduce_scatter(&bytes, TensorDtype::Float32, &group).unwrap();
    assert_eq!(result, bytes);
}

// ---------------------------------------------------------------------------
// Two-rank collective tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_reduce_two_ranks_f32() {
    let groups = build_test_groups(2);
    let results = run_parallel(groups, |group| {
        let rank = group.rank();
        // Rank 0: [1, 2, 3, 4], Rank 1: [10, 20, 30, 40]
        let values: Vec<f32> = (1..=4)
            .map(|v| (v * if rank == 0 { 1 } else { 10 }) as f32)
            .collect();
        let mut data = f32_vec_to_bytes(&values);
        all_reduce_sum(&mut data, TensorDtype::Float32, &group).unwrap();
        bytes_to_f32_vec(&data)
    });

    let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0];
    for (rank, result) in &results {
        assert_eq!(result, &expected, "rank {rank} got wrong all-reduce result");
    }
}

#[test]
fn test_all_gather_two_ranks_f32() {
    let groups = build_test_groups(2);
    let results = run_parallel(groups, |group| {
        let rank = group.rank();
        let values: Vec<f32> = if rank == 0 {
            vec![1.0, 2.0]
        } else {
            vec![3.0, 4.0]
        };
        let bytes = f32_vec_to_bytes(&values);
        let result = all_gather(&bytes, TensorDtype::Float32, &group).unwrap();
        bytes_to_f32_vec(&result)
    });

    let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    for (rank, result) in &results {
        assert_eq!(result, &expected, "rank {rank} got wrong all-gather result");
    }
}

#[test]
fn test_reduce_scatter_two_ranks_f32() {
    let groups = build_test_groups(2);
    let results = run_parallel(groups, |group| {
        // Each rank has [1, 2, 3, 4]. After sum: [2, 4, 6, 8].
        // Rank 0 gets [2, 4], rank 1 gets [6, 8].
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = f32_vec_to_bytes(&values);
        let result = reduce_scatter(&bytes, TensorDtype::Float32, &group).unwrap();
        bytes_to_f32_vec(&result)
    });

    // After reduce-scatter with ring rotation, rank 0 gets chunk[1] and
    // rank 1 gets chunk[0]:
    //   chunk[0] sum = [1+1, 2+2] = [2, 4]
    //   chunk[1] sum = [3+3, 4+4] = [6, 8]
    for (rank, result) in &results {
        if *rank == 0 {
            assert_eq!(result, &[6.0, 8.0], "rank 0 reduce_scatter (chunk 1)");
        } else {
            assert_eq!(result, &[2.0, 4.0], "rank 1 reduce_scatter (chunk 0)");
        }
    }
}

// ---------------------------------------------------------------------------
// Four-rank collective tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_reduce_four_ranks_f32() {
    let groups = build_test_groups(4);
    let results = run_parallel(groups, |group| {
        let rank = group.rank();
        // Each rank contributes [rank+1, rank+1, rank+1, rank+1]
        // Sum = [1+2+3+4, ...] = [10, 10, 10, 10]
        let val = (rank + 1) as f32;
        let values: Vec<f32> = vec![val; 4];
        let mut data = f32_vec_to_bytes(&values);
        all_reduce_sum(&mut data, TensorDtype::Float32, &group).unwrap();
        bytes_to_f32_vec(&data)
    });

    let expected: Vec<f32> = vec![10.0; 4];
    for (rank, result) in &results {
        assert_eq!(result, &expected, "rank {rank} 4-rank all-reduce");
    }
}

#[test]
fn test_all_gather_four_ranks_f32() {
    let groups = build_test_groups(4);
    let results = run_parallel(groups, |group| {
        let rank = group.rank();
        let values: Vec<f32> = vec![(rank * 10 + 1) as f32, (rank * 10 + 2) as f32];
        let bytes = f32_vec_to_bytes(&values);
        let result = all_gather(&bytes, TensorDtype::Float32, &group).unwrap();
        bytes_to_f32_vec(&result)
    });

    let expected: Vec<f32> = vec![1.0, 2.0, 11.0, 12.0, 21.0, 22.0, 31.0, 32.0];
    for (rank, result) in &results {
        assert_eq!(result, &expected, "rank {rank} 4-rank all-gather");
    }
}

// ---------------------------------------------------------------------------
// Odd tensor sizes
// ---------------------------------------------------------------------------

#[test]
fn test_all_reduce_odd_element_count() {
    // 7 elements across 2 ranks: chunks will be 4 + 3.
    let groups = build_test_groups(2);
    let results = run_parallel(groups, |group| {
        let rank = group.rank();
        let values: Vec<f32> = (0..7).map(|i| (i + 1) as f32 * (rank + 1) as f32).collect();
        let mut data = f32_vec_to_bytes(&values);
        all_reduce_sum(&mut data, TensorDtype::Float32, &group).unwrap();
        bytes_to_f32_vec(&data)
    });

    // Rank 0: [1,2,3,4,5,6,7], Rank 1: [2,4,6,8,10,12,14]
    // Sum: [3,6,9,12,15,18,21]
    let expected: Vec<f32> = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0];
    for (rank, result) in &results {
        assert_eq!(result, &expected, "rank {rank} odd-count all-reduce");
    }
}

// ---------------------------------------------------------------------------
// i32 dtype test
// ---------------------------------------------------------------------------

#[test]
fn test_all_reduce_two_ranks_i32() {
    let groups = build_test_groups(2);
    let results = run_parallel(groups, |group| {
        let rank = group.rank();
        let values: Vec<i32> = vec![100 * (rank as i32 + 1); 8];
        let mut data = i32_vec_to_bytes(&values);
        all_reduce_sum(&mut data, TensorDtype::Int32, &group).unwrap();
        bytes_to_i32_vec(&data)
    });

    // Rank 0: [100; 8], Rank 1: [200; 8] -> Sum: [300; 8]
    let expected: Vec<i32> = vec![300; 8];
    for (rank, result) in &results {
        assert_eq!(result, &expected, "rank {rank} i32 all-reduce");
    }
}

// ---------------------------------------------------------------------------
// Benchmark helper tests
// ---------------------------------------------------------------------------

#[test]
fn test_ring_allreduce_data_volume() {
    // Single rank: 0
    assert_eq!(ring_allreduce_data_volume(1024, 1), 0.0);

    // 2 ranks: 2 * (1/2) * 1024 = 1024
    assert!((ring_allreduce_data_volume(1024, 2) - 1024.0).abs() < 1e-6);

    // 4 ranks: 2 * (3/4) * 1024 = 1536
    assert!((ring_allreduce_data_volume(1024, 4) - 1536.0).abs() < 1e-6);
}

#[test]
fn test_benchmark_result_display() {
    let r = BenchmarkResult {
        operation: "all_reduce_sum".to_string(),
        dtype: TensorDtype::Float32,
        tensor_bytes: 1024 * 1024,
        world_size: 4,
        elapsed_us: 500,
        bandwidth_bytes_per_sec: 3.0e9,
    };
    let s = r.to_string();
    assert!(s.contains("all_reduce_sum"));
    assert!(s.contains("1 MiB"));
    assert!(s.contains("4 ranks"));
    assert!(s.contains("GB/s"));
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_elementwise_add_unsupported_dtype() {
    let mut dst = vec![0u8; 4];
    let result = elementwise_add_inplace(&mut dst, &[0u8; 4], TensorDtype::UInt8);
    assert!(result.is_err());
}

#[test]
fn test_all_reduce_unaligned_buffer() {
    // 3 bytes is not aligned to f32 (4 bytes).
    let groups = build_test_groups(2);
    let results = run_parallel(groups, |group| {
        let mut data = vec![0u8; 3];
        all_reduce_sum(&mut data, TensorDtype::Float32, &group).is_err()
    });
    assert!(results.iter().all(|(_, e)| *e));
}

#[test]
fn test_all_reduce_int4_unsupported() {
    let groups = build_test_groups(2);
    let results = run_parallel(groups, |group| {
        let mut data = vec![0u8; 8];
        all_reduce_sum(&mut data, TensorDtype::Int4, &group).is_err()
    });
    assert!(results.iter().all(|(_, e)| *e));
}

// ---------------------------------------------------------------------------
// f16/bf16 conversion round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn test_f16_roundtrip() {
    let values = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -0.0];
    for &v in &values {
        let bits = f32_to_f16(v);
        let back = f16_to_f32(bits);
        assert!(
            (back - v).abs() < 0.01 || (v == 0.0 && back == 0.0),
            "f16 roundtrip failed for {v}: got {back}"
        );
    }
}

#[test]
fn test_bf16_roundtrip() {
    let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -0.0];
    for &v in &values {
        let bits = f32_to_bf16(v);
        let back = bf16_to_f32(bits);
        assert!(
            (back - v).abs() < 0.1 || (v == 0.0 && back == 0.0),
            "bf16 roundtrip failed for {v}: got {back}"
        );
    }
}
