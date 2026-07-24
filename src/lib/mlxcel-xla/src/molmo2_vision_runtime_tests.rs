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

#[test]
fn output_decode_rejects_non_finite_values() {
    let mut bytes = 1.0f32.to_ne_bytes().to_vec();
    bytes.extend_from_slice(&f32::INFINITY.to_ne_bytes());
    assert!(decode_output(&bytes).unwrap_err().contains("flat index 1"));
}

#[test]
fn compiler_identity_changes_when_same_path_and_version_bytes_change() {
    let path = std::env::temp_dir().join(format!("mlxcel-molmo2-compiler-{}", std::process::id()));
    std::fs::write(&path, b"first").unwrap();
    let first = sha256_file(&path).unwrap();
    std::fs::write(&path, b"second").unwrap();
    let second = sha256_file(&path).unwrap();
    std::fs::remove_file(path).ok();
    assert_ne!(first, second);
}
