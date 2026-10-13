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

/// The fixture's event error is consumable, so this must call the newly
/// fallible bridge directly without probing availability or evaluating first.
#[test]
fn failed_event_readback_returns_an_error() {
    let array = array_with_failed_event_for_test();
    let err = array_evaluated_bytes(&array).expect_err("failed event must cross as Result::Err");
    assert!(
        err.to_string().contains("synthetic array read failure"),
        "unexpected bridge error: {err}"
    );
}
