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

//! Decider-2b parity against mlx-vlm on the published checkpoint.
//!
//! Loads `models/decider-2b` through the decision loader and answers the two
//! requests of `tests/fixtures/decision/decider2_parity.json` (a 3-option
//! choice, an isolated 4-level score and a described bool; then a 12-option
//! choice and a bool over an annotated JSON state). In the short ticket
//! request every probability must be within 2e-3 of upstream's (measured
//! 1.4e-3). The structured request runs about 400 tokens, where bf16
//! accumulation order alone moves the answers: against mlx-vlm run in f32,
//! upstream's own bf16 answers are off by up to 7.2e-3 and change by 8e-3
//! when its bool is scored alone instead of batched with the choice, and
//! this port's are off by up to 1.5e-2. That request therefore gets 2e-2,
//! and its argmax labels and bool values must still agree. Prompt ids and
//! usage are exact in both. Skips when the checkpoint is
//! absent. Run it alone under the GPU lock:
//!
//! ```sh
//! RUST_MIN_STACK=16777216 cargo test --profile test-fast --features metal,accelerate \
//!     --lib decision::decider2_parity_tests
//! ```

use serde_json::Value;

use super::decider2_tests::{checkpoint, fixture};
use super::{DecisionMedia, load_decision_model, predict};

/// Tolerance for the short ticket request (the acceptance bound).
const TOLERANCE: f64 = 2e-3;
/// Tolerance for the ~400-token structured request (see the module docs).
const LONG_TOLERANCE: f64 = 2e-2;

/// Assert `actual` matches `expected`; returns the largest number gap.
fn close(actual: &Value, expected: &Value, path: &str, tolerance: f64) -> f64 {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (
                a.as_f64().unwrap_or(f64::NAN),
                e.as_f64().unwrap_or(f64::NAN),
            );
            assert!((a - e).abs() <= tolerance, "{path}: {a} vs upstream {e}");
            (a - e).abs()
        }
        (Value::Object(a), Value::Object(e)) => {
            let keys =
                |map: &serde_json::Map<String, Value>| map.keys().cloned().collect::<Vec<_>>();
            assert_eq!(keys(a), keys(e), "{path}: keys");
            e.iter()
                .map(|(key, value)| close(&a[key], value, &format!("{path}.{key}"), tolerance))
                .fold(0.0, f64::max)
        }
        _ => {
            assert_eq!(actual, expected, "{path}");
            0.0
        }
    }
}

#[test]
fn decider2_matches_mlx_vlm_on_the_published_checkpoint() {
    let Some(dir) = checkpoint(true) else {
        return;
    };
    let mut loaded = load_decision_model(&dir).expect("decider-2b loads");
    assert_eq!(loaded.model_type, "decider2");
    for request in fixture()["requests"].as_array().expect("requests") {
        let name = request["name"].as_str().unwrap_or("request");
        let result = predict(
            loaded.model.as_mut(),
            Some(&request["state"]),
            &request["questions"],
            DecisionMedia::default(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let response = result.to_response_json("models/decider-2b");
        let expected = &request["expected"];
        // Token accounting is exact: the prompts are the same ids.
        assert_eq!(response["usage"], expected["usage"], "{name}: usage");
        eprintln!("{name}: {}", response["answers"]);
        let tolerance = if name == "ticket" {
            TOLERANCE
        } else {
            LONG_TOLERANCE
        };
        let gap = close(&response["answers"], &expected["answers"], name, tolerance);
        eprintln!("{name}: largest gap to mlx-vlm {gap:.4} (tolerance {tolerance})");
    }
}
