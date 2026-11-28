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

//! The decode-throughput rule shared by every generation path's stats.

use super::decode_tok_per_sec;

#[test]
fn rate_counts_only_window_tokens() {
    // 128 tokens, the first from prefill: 127 produced in a 1000 ms window.
    assert!((decode_tok_per_sec(127, 1000.0) - 127.0).abs() < 1e-9);
    assert!((decode_tok_per_sec(127, 500.0) - 254.0).abs() < 1e-9);
}

#[test]
fn rate_is_zero_without_a_window() {
    assert_eq!(decode_tok_per_sec(0, 12.5), 0.0);
    assert_eq!(decode_tok_per_sec(10, 0.0), 0.0);
    assert_eq!(decode_tok_per_sec(0, 0.0), 0.0);
}
