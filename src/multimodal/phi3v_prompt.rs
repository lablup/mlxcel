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

//! Phi3V-specific prompt normalization helpers.
//!
//! Phi3V can receive prompts with missing or partially numbered image tags.
//! These helpers normalize `<|image_N|>` placement before tokenization so both
//! CLI and server flows share the same prompt-to-image-slot mapping.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi3vImageTag {
    pub start: usize,
    pub end: usize,
    pub image_num: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi3vPromptTokens {
    pub tokens: Vec<i32>,
    pub image_slots: usize,
}

pub fn ensure_phi3v_image_tags(prompt: &str, num_images: usize) -> String {
    if num_images == 0 || (1..=num_images).any(|i| prompt.contains(&format!("<|image_{}|>", i))) {
        return prompt.to_string();
    }

    let image_tags: String = (1..=num_images)
        .map(|i| format!("<|image_{}|>\n", i))
        .collect();

    if let Some(pos) = prompt.find("<|user|>\n") {
        let mut text = prompt.to_string();
        text.insert_str(pos + "<|user|>\n".len(), &image_tags);
        text
    } else {
        format!("{}{}", image_tags, prompt)
    }
}

pub fn collect_phi3v_image_tags(text: &str, num_images: usize) -> Vec<Phi3vImageTag> {
    let mut tags = Vec::new();

    for image_num in 1..=num_images {
        let tag = format!("<|image_{}|>", image_num);
        let mut search_from = 0;
        while let Some(pos) = text[search_from..].find(&tag) {
            let start = search_from + pos;
            tags.push(Phi3vImageTag {
                start,
                end: start + tag.len(),
                image_num,
            });
            search_from = start + tag.len();
        }
    }

    tags.sort_by_key(|tag| tag.start);
    tags
}

pub fn prepare_phi3v_prompt_tokens<E, I>(
    prompt: &str,
    num_images: usize,
    mut encode: E,
    mut image_token_count: I,
) -> Option<Phi3vPromptTokens>
where
    E: FnMut(&str, bool) -> Vec<i32>,
    I: FnMut(usize) -> usize,
{
    let text = ensure_phi3v_image_tags(prompt, num_images);
    let tag_positions = collect_phi3v_image_tags(&text, num_images);

    if tag_positions.is_empty() {
        return None;
    }

    let mut tokens = Vec::new();
    let mut last_end = 0;

    for (chunk_idx, tag) in tag_positions.iter().enumerate() {
        let before = &text[last_end..tag.start];
        if !before.is_empty() {
            tokens.extend(encode(before, chunk_idx == 0 && last_end == 0));
        }

        let neg_id = -(tag.image_num as i32);
        for _ in 0..image_token_count(tag.image_num) {
            tokens.push(neg_id);
        }

        last_end = tag.end;
    }

    let after = &text[last_end..];
    if !after.is_empty() {
        tokens.extend(encode(after, false));
    }

    Some(Phi3vPromptTokens {
        tokens,
        image_slots: tag_positions.len(),
    })
}

#[cfg(test)]
#[path = "phi3v_prompt_tests.rs"]
mod tests;
