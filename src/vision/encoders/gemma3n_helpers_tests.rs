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

use super::helpers::{
    get_same_padding, get_static_padding, is_static_pad, make_divisible, num_groups,
    split_symmetric_padding,
};

#[test]
fn make_divisible_rounds_mobile_channel_counts() {
    assert_eq!(make_divisible(15.0, 8), 16);
    assert_eq!(make_divisible(8.0, 8), 8);
}

#[test]
fn num_groups_handles_depthwise_and_dense_paths() {
    assert_eq!(num_groups(0, 64), 1);
    assert_eq!(num_groups(16, 64), 4);
}

#[test]
fn same_padding_helpers_compute_static_and_dynamic_values() {
    assert!(is_static_pad(1));
    assert!(!is_static_pad(2));
    assert_eq!(get_static_padding(3, 1), 1);
    assert_eq!(get_same_padding(8, 3, 2, 1), 1);
}

#[test]
fn split_symmetric_padding_balances_before_and_after() {
    assert_eq!(split_symmetric_padding(4), (2, 2));
    assert_eq!(split_symmetric_padding(5), (2, 3));
}
