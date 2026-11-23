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

//! Clef-Flash parity against mlx-vlm on the published 8-bit checkpoint.
//!
//! Loads `models/clef-flash-mlx-8bit` through the decision loader and answers
//! the two requests of `tests/fixtures/decision/clef_parity.json`: a
//! text-only ticket (a choice, a described choice whose labels sort
//! differently from the request, a 4-level score and a described `noul`) and
//! a null-state request over `tests/fixtures/sam3/cats.png` (a choice and a
//! bool). The usage must be exact and every probability and metadata number
//! close to upstream's: within 2e-3 on the image request and 3e-3 on the
//! text one.
//!
//! The text bound is wider than the image one because the published bf16
//! weights leave upstream itself sensitive to rounding order: mlx-vlm bf16
//! and mlx-vlm f32 differ by up to 7.9e-3 on the urgency score, while the
//! two implementations differ by at most 2.3e-3 (probabilities) and 2.8e-3
//! (the score value) in bf16. Converted to f32, mlxcel and mlx-vlm give
//! identical answers to four decimal places on both requests (measured on a
//! local f32 copy of the checkpoint; not committed). Skips when the
//! checkpoint is absent. Run it alone under the GPU lock:
//!
//! ```sh
//! RUST_MIN_STACK=16777216 cargo test --profile test-fast --features metal,accelerate \
//!     --lib decision::clef_parity_tests
//! ```

use serde_json::Value;

use super::clef_tests::{cats, checkpoint, fixture};
use super::{DecisionMedia, load_decision_model, predict};

/// The bound for a request with media.
const IMAGE_TOLERANCE: f64 = 2e-3;
/// The bound for the text-only request (see the module docs).
const TEXT_TOLERANCE: f64 = 3e-3;

/// Assert `actual` matches `expected` within `tolerance`; returns the
/// largest number gap.
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
fn clef_flash_matches_mlx_vlm_on_the_published_checkpoint() {
    let Some(dir) = checkpoint(true) else {
        return;
    };
    let mut loaded = load_decision_model(&dir).expect("clef-flash loads");
    assert_eq!(loaded.model_type, "clef");
    for request in fixture()["requests"].as_array().expect("requests") {
        let name = request["name"].as_str().unwrap_or("request");
        let image = request["image"].as_bool() == Some(true);
        let tolerance = if image {
            IMAGE_TOLERANCE
        } else {
            TEXT_TOLERANCE
        };
        let media = DecisionMedia {
            images: if image { vec![cats()] } else { Vec::new() },
            ..DecisionMedia::default()
        };
        let result = predict(
            loaded.model.as_mut(),
            Some(&request["state"]),
            &request["questions"],
            media,
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let response = result.to_response_json("clef");
        let expected = &request["expected"];
        assert_eq!(response["usage"], expected["usage"], "{name}: usage");
        eprintln!("{name}: {}", response["answers"]);
        let gap = close(&response["answers"], &expected["answers"], name, tolerance);
        eprintln!("{name}: largest gap to mlx-vlm {gap:.4} (tolerance {tolerance})");
    }
}
