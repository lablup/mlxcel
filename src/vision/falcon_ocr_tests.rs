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

//! Falcon-OCR runtime helper tests.

use super::*;

fn ids() -> FalconOcrTokenIds {
    FalconOcrTokenIds {
        img_id: 227,
        image_cls_token_id: 244,
        img_end_id: 230,
        image_reg_token_ids: [245, 246, 247, 248],
    }
}

#[test]
fn the_patch_and_block_opening_tokens_are_suppressed() {
    let suppressed = suppressed_token_ids(&ids());
    assert!(
        suppressed.contains(&227),
        "patch placeholder must be suppressed"
    );
    assert!(suppressed.contains(&244), "image CLS must be suppressed");
    for reg in [245, 246, 247, 248] {
        assert!(
            suppressed.contains(&reg),
            "register token {reg} must be suppressed"
        );
    }
}

/// `<|end_of_image|>` sits outside the bidirectional region and is a normal
/// structural token, so suppressing it would be wrong.
#[test]
fn the_image_closing_token_is_not_suppressed() {
    assert!(!suppressed_token_ids(&ids()).contains(&230));
}
