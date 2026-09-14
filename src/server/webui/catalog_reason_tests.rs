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

use super::sanitize_detection_error;
use std::path::Path;

#[test]
fn detection_reason_limit_counts_unicode_scalars_without_splitting_utf8() {
    for (glyph, width) in [("x", 1), ("한", 3), ("🦀", 4)] {
        for count in [511, 512, 513, 1024] {
            let input = glyph.repeat(count);
            let output = sanitize_detection_error(Path::new(""), &input);
            assert_eq!(output.chars().count(), count.min(512));
            if count <= 512 {
                assert_eq!(output, input);
                assert_eq!(output.len(), count * width);
            } else {
                assert_eq!(output, format!("{}…", glyph.repeat(511)));
            }
            let encoded = serde_json::to_string(&output).unwrap();
            assert_eq!(serde_json::from_str::<String>(&encoded).unwrap(), output);
        }
    }
}

#[test]
fn detection_reason_redacts_full_unicode_path_before_applying_limit() {
    let private_path = format!("/Users/private-catalog/{}", "비밀🦀".repeat(300));
    let message = format!(
        "{private_path} is not a standalone model. {}",
        "설명🦀".repeat(300)
    );
    let output = sanitize_detection_error(Path::new(&private_path), &message);
    assert!(output.starts_with("model directory is not a standalone model. "));
    assert_eq!(output.chars().count(), 512);
    assert!(output.ends_with('…'));
    assert!(!output.contains("private-catalog"));
    assert!(!output.contains("비밀"));
}
