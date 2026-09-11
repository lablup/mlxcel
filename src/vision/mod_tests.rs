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

use super::require_array_ref;
use mlxcel_core::{self, UniquePtr, dtype};

#[test]
fn require_array_ref_rejects_null_unique_ptrs() {
    let array: UniquePtr<mlxcel_core::MlxArray> = UniquePtr::null();
    let err = match require_array_ref(&array, "test array") {
        Ok(_) => panic!("expected null pointer to fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("null test array"));
}

#[test]
fn require_array_ref_accepts_real_arrays() {
    let array = mlxcel_core::ones(&[1, 2], dtype::FLOAT32);
    let resolved = require_array_ref(&array, "test array").unwrap();
    assert_eq!(mlxcel_core::array_shape(resolved), vec![1, 2]);
}
