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

use tempfile::TempDir;

use super::*;

#[test]
fn the_enhancer_rule_follows_upstream() {
    // Absent: on for generation when pe/ exists, never for an edit.
    assert!(should_enhance(None, true, false).unwrap());
    assert!(!should_enhance(None, true, true).unwrap());
    assert!(!should_enhance(None, false, false).unwrap());
    // An explicit choice wins, including enhancing an edit.
    assert!(!should_enhance(Some(false), true, false).unwrap());
    assert!(should_enhance(Some(true), true, true).unwrap());
    // Requested without pe/: upstream's message.
    let err = should_enhance(Some(true), false, false).unwrap_err();
    assert_eq!(err.to_string(), MISSING_ENHANCER);
}

#[test]
fn the_user_turn_is_python_json_dumps() {
    assert_eq!(
        enhancer_content("A \"red\" panda, 茶", 1024, 768).unwrap(),
        r#"{"prompt": "A \"red\" panda, 茶", "width": 1024, "height": 768}"#
    );
}

#[test]
fn availability_needs_pe_weights_and_the_pe_tokenizer() {
    let dir = TempDir::new().unwrap();
    assert!(!enhancer_available(dir.path()));
    std::fs::create_dir_all(dir.path().join("pe")).unwrap();
    std::fs::write(dir.path().join("pe/model.safetensors"), b"").unwrap();
    assert!(!enhancer_available(dir.path()));
    std::fs::create_dir_all(dir.path().join("pe_tokenizer")).unwrap();
    std::fs::write(dir.path().join("pe_tokenizer/tokenizer.json"), b"{}").unwrap();
    assert!(enhancer_available(dir.path()));
}
