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

//! Prompt formatting against upstream's tokenizer output on the real
//! Z-Image-Turbo tokenizer (`models/mlx/z-image-turbo-bf16/tokenizer`, or
//! `MLXCEL_Z_IMAGE_TURBO`). Skips when the checkpoint is absent.

use std::path::PathBuf;

use super::*;

fn tokenizer_dir() -> Option<PathBuf> {
    let root = std::env::var_os("MLXCEL_Z_IMAGE_TURBO")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/mlx/z-image-turbo-bf16")
        });
    let dir = root.join("tokenizer");
    if dir.join("tokenizer.json").exists() {
        Some(dir)
    } else {
        eprintln!("skipping: {} is absent", dir.display());
        None
    }
}

#[test]
fn prompt_matches_upstream_chat_template_and_ids() {
    let Some(dir) = tokenizer_dir() else {
        return;
    };
    let formatter = PromptFormatter::load(&dir).unwrap();
    let prompt = "A red panda serving tea in a moonlit bamboo forest";
    // `tokenizer.apply_chat_template(..., enable_thinking=True)` upstream.
    assert_eq!(
        formatter.format(prompt).unwrap(),
        "<|im_start|>user\nA red panda serving tea in a moonlit bamboo forest<|im_end|>\n<|im_start|>assistant\n"
    );
    let (ids, mask, valid) = formatter.padded(prompt).unwrap();
    assert_eq!(valid, 19);
    assert_eq!(ids.len(), MAX_PROMPT_TOKENS);
    assert_eq!(
        &ids[..21],
        [
            151644, 872, 198, 32, 2518, 88222, 13480, 15243, 304, 264, 17788, 31635, 58882, 13638,
            151645, 198, 151644, 77091, 198, 151643, 151643
        ]
    );
    assert_eq!(mask.iter().sum::<i32>(), 19);
    assert_eq!(mask[18..20], [1, 0]);
    // The negative prompt of the base parity run.
    let (ids, _, valid) = formatter.padded("blurry, low quality").unwrap();
    assert_eq!(valid, 13);
    assert_eq!(&ids[3..8], [33542, 884, 11, 3347, 4271]);
}

#[test]
fn long_prompts_truncate_to_the_padded_length() {
    let Some(dir) = tokenizer_dir() else {
        return;
    };
    let formatter = PromptFormatter::load(&dir).unwrap();
    let long = "tea ".repeat(2 * MAX_PROMPT_TOKENS);
    let (ids, mask, valid) = formatter.padded(&long).unwrap();
    assert_eq!((ids.len(), valid), (MAX_PROMPT_TOKENS, MAX_PROMPT_TOKENS));
    assert!(mask.iter().all(|&m| m == 1));
    assert!(formatter.token_ids(&long).unwrap().len() > MAX_PROMPT_TOKENS);
}
