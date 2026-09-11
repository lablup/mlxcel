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

use super::{
    PHI4_SIGLIP_IMAGE_TOKEN_INDEX, count_phi4_siglip_image_tokens, ensure_phi4_siglip_image_tokens,
    prepare_phi4_siglip_prompt_tokens,
};

#[test]
fn ensure_phi4_siglip_image_tokens_prefixes_missing_placeholders() {
    let prompt = ensure_phi4_siglip_image_tokens("Describe the image.", 2);
    assert!(prompt.starts_with("<image>\n<image>\n"));
    assert_eq!(count_phi4_siglip_image_tokens(&prompt), 2);
}

#[test]
fn ensure_phi4_siglip_image_tokens_inserts_after_user_tag() {
    let prompt = ensure_phi4_siglip_image_tokens("<|user|>\nDescribe it.", 1);
    assert_eq!(prompt, "<|user|>\n<image>\nDescribe it.");
}

#[test]
fn prepare_phi4_siglip_prompt_tokens_inserts_negative_placeholder_ids() {
    let prepared = prepare_phi4_siglip_prompt_tokens("A <image> B", 1, |text, add_special| {
        let mut out = Vec::new();
        if add_special {
            out.push(101);
        }
        out.extend(text.bytes().map(|byte| byte as i32));
        out
    })
    .unwrap();

    assert_eq!(prepared.image_slots, 1);
    assert!(prepared.tokens.contains(&PHI4_SIGLIP_IMAGE_TOKEN_INDEX));
}

#[test]
fn prepare_phi4_siglip_prompt_tokens_rejects_placeholder_image_mismatch() {
    let err =
        prepare_phi4_siglip_prompt_tokens("<image>", 2, |_text, _add_special| vec![1]).unwrap_err();
    assert!(err.contains("2 image(s) were provided"));
}
