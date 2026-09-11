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
fn split_even_division() {
    let specs = split_into_micro_batches(8, 2).unwrap();
    assert_eq!(specs.len(), 4);
    for (i, s) in specs.iter().enumerate() {
        assert_eq!(s.id, i as u32);
        assert_eq!(s.start_index, i * 2);
        assert_eq!(s.end_index, (i + 1) * 2);
        assert_eq!(s.size, 2);
    }
}

#[test]
fn split_uneven_division() {
    let specs = split_into_micro_batches(7, 3).unwrap();
    assert_eq!(specs.len(), 3);
    assert_eq!(specs[0].size, 3); // 0..3
    assert_eq!(specs[1].size, 3); // 3..6
    assert_eq!(specs[2].size, 1); // 6..7
}

#[test]
fn split_single_micro_batch() {
    let specs = split_into_micro_batches(4, 10).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].start_index, 0);
    assert_eq!(specs[0].end_index, 4);
    assert_eq!(specs[0].size, 4);
}

#[test]
fn split_one_per_micro_batch() {
    let specs = split_into_micro_batches(3, 1).unwrap();
    assert_eq!(specs.len(), 3);
    for (i, s) in specs.iter().enumerate() {
        assert_eq!(s.start_index, i);
        assert_eq!(s.end_index, i + 1);
        assert_eq!(s.size, 1);
    }
}

#[test]
fn split_zero_batch_size_errors() {
    assert!(split_into_micro_batches(0, 2).is_err());
}

#[test]
fn split_zero_micro_batch_size_errors() {
    assert!(split_into_micro_batches(4, 0).is_err());
}

#[test]
fn coverage_no_gaps() {
    // Verify that all micro-batches tile the original batch without gaps.
    for batch in 1..=20 {
        for mb in 1..=batch {
            let specs = split_into_micro_batches(batch, mb).unwrap();
            let total: usize = specs.iter().map(|s| s.size).sum();
            assert_eq!(total, batch, "batch={batch} mb={mb}");

            // Check contiguity.
            for pair in specs.windows(2) {
                assert_eq!(pair[0].end_index, pair[1].start_index);
            }
            assert_eq!(specs[0].start_index, 0);
            assert_eq!(specs.last().unwrap().end_index, batch);
        }
    }
}

#[test]
fn suggested_size_basic() {
    // 16 sequences, 4 stages -> target 8 micro-batches -> size 2
    assert_eq!(suggested_micro_batch_size(16, 4), 2);
}

#[test]
fn suggested_size_small_batch() {
    // 2 sequences, 4 stages -> target 8 micro-batches but batch is only 2
    assert_eq!(suggested_micro_batch_size(2, 4), 1);
}

#[test]
fn suggested_size_single_stage() {
    // 10 sequences, 1 stage -> target 2 micro-batches -> size 5
    assert_eq!(suggested_micro_batch_size(10, 1), 5);
}

#[test]
fn suggested_size_edge_cases() {
    assert_eq!(suggested_micro_batch_size(0, 4), 1);
    assert_eq!(suggested_micro_batch_size(4, 0), 1);
}

#[test]
fn micro_batch_advance_and_complete() {
    let spec = MicroBatchSpec {
        id: 0,
        start_index: 0,
        end_index: 4,
        size: 4,
    };
    let req_id = RequestId::from_string("test-req".to_string()).unwrap();
    let mut mb = MicroBatch::new(spec, vec![req_id]);

    assert_eq!(mb.current_stage, 0);
    assert!(!mb.completed);

    mb.advance_stage();
    assert_eq!(mb.current_stage, 1);

    mb.mark_completed();
    assert!(mb.completed);
}

#[test]
fn micro_batch_display() {
    let spec = MicroBatchSpec {
        id: 2,
        start_index: 4,
        end_index: 6,
        size: 2,
    };
    let display = format!("{spec}");
    assert!(display.contains("id=2"));
    assert!(display.contains("range=4..6"));
}
