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

// --- estimate_concurrency ---

#[test]
fn concurrency_zero_layers() {
    assert_eq!(estimate_concurrency(0, 1_000_000, 1_000_000_000.0), 1);
}

#[test]
fn concurrency_zero_bandwidth() {
    assert_eq!(estimate_concurrency(32, 1_000_000, 0.0), 1);
}

#[test]
fn concurrency_large_layers() {
    // 32 layers, 128 MB total, 1 GB/s bandwidth.
    // Each layer is 4 MB, well above 1 MB minimum.
    let c = estimate_concurrency(32, 128 * 1024 * 1024, 1_000_000_000.0);
    assert!(c > 1);
    assert!(c <= 16);
}

#[test]
fn concurrency_small_layers() {
    // 32 layers, 1 MB total (tiny per-layer).
    // Should group layers together.
    let c = estimate_concurrency(32, 1024 * 1024, 1_000_000_000.0);
    assert_eq!(c, 1); // Only 1 MB total, so 1 task suffices.
}

#[test]
fn concurrency_capped_at_16() {
    // Even with many layers, cap at 16.
    let c = estimate_concurrency(256, 256 * 1024 * 1024, 1_000_000_000.0);
    assert!(c <= 16);
}

#[test]
fn concurrency_capped_at_num_layers() {
    // 4 layers should produce at most 4 concurrency.
    let c = estimate_concurrency(4, 128 * 1024 * 1024, 1_000_000_000.0);
    assert!(c <= 4);
}

// --- ParallelLayerTransfer construction ---

#[test]
fn parallel_transfer_concurrency() {
    let config = TransferConfig {
        concurrency: 8,
        ..TransferConfig::default()
    };
    let transfer = ParallelLayerTransfer::new(config, "peer:8080".to_string(), 42);
    assert_eq!(transfer.concurrency(), 8);
}
