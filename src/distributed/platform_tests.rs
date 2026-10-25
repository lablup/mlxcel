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

use super::{distributed_refusal, ensure_distributed_supported};

#[test]
fn windows_refuses_and_names_the_requested_mode() {
    let message = distributed_refusal(true, "--tp-size 2").expect("Windows must refuse");
    assert!(message.starts_with("--tp-size 2 is not supported on native Windows"));
    assert!(message.contains("single-node only"));
}

#[test]
fn other_platforms_allow_distributed_requests() {
    assert_eq!(distributed_refusal(false, "--pp-layers"), None);
}

#[test]
fn the_build_platform_decides_the_runtime_gate() {
    assert_eq!(
        ensure_distributed_supported("--distributed-config").is_err(),
        cfg!(windows)
    );
}
