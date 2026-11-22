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

//! Prompt shaping for PLaMo 2.1 VL (`plamo2vl`, issue #1380).
//!
//! The checkpoint ships no chat template. Its processor always wraps the user
//! text in one fixed Japanese instruction template (`_apply_ja01_template`),
//! with one `<plamo:image>` per image in front of the text:
//!
//! ```text
//! 以下はタスクを説明する指示で、文脈を説明した入力とペアになっています。要求を適切に補完するよう応答を書いてください。
//!
//! ### 指示:
//! <plamo:image>{text}
//!
//! ### 応答:
//!
//! ```
//!
//! Each `<plamo:image>` then becomes `<siglip2:img>` + `<plamo:image>` x
//! `(729 * tiles_i)` + `</siglip2:img>`, and `<|plamo:bos|>` leads when the
//! checkpoint's tokenizer config sets `add_bos_token`. The server's builtin template
//! renders the same instruction text without the image markers, so this
//! builder recognizes the `### 指示:` line and only inserts the markers there:
//! the CLI's bare `-p` text and a server render converge on one prompt.
//!
//! Reference: `processing_plamo2_vl.py` in `pfnet/plamo-2.1-2b-vl`.

/// The instruction line the template opens with.
pub const PLAMO2_VL_INSTRUCTION_HEADER: &str = "以下はタスクを説明する指示で、文脈を説明した入力とペアになっています。要求を適切に補完するよう応答を書いてください。";

/// The line that opens the instruction body; image markers follow it.
pub const PLAMO2_VL_INSTRUCTION_MARKER: &str = "### 指示:\n";

/// The response opener the template ends with.
pub const PLAMO2_VL_RESPONSE_MARKER: &str = "### 応答:";

/// The per-image placeholder (`<plamo:image>`, id 100002).
pub const PLAMO2_VL_IMAGE_TAG: &str = "<plamo:image>";

/// What [`prepare_plamo2_vl_prompt_tokens`] assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plamo2VlPromptStats {
    /// `<siglip2:img> ... </siglip2:img>` blocks placed (one per image).
    pub image_blocks: usize,
    /// Tiles across all images (each contributes `num_image_token` rows).
    pub total_tiles: usize,
    /// `<plamo:image>` feature placeholders in the final token stream.
    pub total_image_tokens: usize,
    /// Whether the caller's text already carried the instruction template.
    pub pre_templated: bool,
}

/// `_apply_ja01_template(<plamo:image>, text, image_length)`.
pub fn format_plamo2_vl_prompt(text: &str, image_count: usize) -> String {
    [
        PLAMO2_VL_INSTRUCTION_HEADER.to_string(),
        String::new(),
        "### 指示:".to_string(),
        format!("{}{text}", PLAMO2_VL_IMAGE_TAG.repeat(image_count)),
        String::new(),
        PLAMO2_VL_RESPONSE_MARKER.to_string(),
        String::new(),
    ]
    .join("\n")
}

/// Whether `text` already carries the instruction template.
pub fn is_plamo2_vl_templated(text: &str) -> bool {
    text.contains(PLAMO2_VL_INSTRUCTION_MARKER)
}

/// Place one `<plamo:image>` per image.
///
/// Text that already carries `image_count` markers keeps them where they are.
/// Templated text (the server render) gets the markers right after
/// `### 指示:\n`, where upstream puts them; bare text is wrapped in the
/// template. Returns the prompt and whether the input was templated.
pub fn build_plamo2_vl_prompt(text: &str, image_count: usize) -> (String, bool) {
    let templated = is_plamo2_vl_templated(text);
    let placed = text.matches(PLAMO2_VL_IMAGE_TAG).count() == image_count && image_count > 0;
    let prompt = match (templated, placed) {
        (true, true) => text.to_string(),
        (true, false) => {
            let at = text.find(PLAMO2_VL_INSTRUCTION_MARKER).unwrap_or(0)
                + PLAMO2_VL_INSTRUCTION_MARKER.len();
            format!(
                "{}{}{}",
                &text[..at],
                PLAMO2_VL_IMAGE_TAG.repeat(image_count),
                &text[at..]
            )
        }
        (false, true) => format_plamo2_vl_prompt(text, 0),
        (false, false) => format_plamo2_vl_prompt(text, image_count),
    };
    (prompt, templated)
}

/// Token ids of the image framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plamo2VlImageTokenIds {
    /// Prepended when set (the tokenizer config's `add_bos_token`).
    pub bos: Option<i32>,
    pub img_start: i32,
    pub image: i32,
    pub img_end: i32,
}

/// Replace the `i`-th `<plamo:image>` token with `img_start` + `image` x
/// `num_image_token * tiles_per_image[i]` + `img_end`.
///
/// The placeholder count must equal the image count: a mismatch means the
/// tokenizer did not resolve `<plamo:image>` to its id (or the caller placed
/// extra markers), and the scatter would misalign.
pub fn expand_plamo2_vl_image_tokens(
    tokens: &[i32],
    tiles_per_image: &[usize],
    num_image_token: usize,
    ids: Plamo2VlImageTokenIds,
) -> Result<(Vec<i32>, usize), String> {
    let placeholders = tokens.iter().filter(|&&t| t == ids.image).count();
    if placeholders != tiles_per_image.len() {
        return Err(format!(
            "PLaMo 2.1 VL prompt has {placeholders} <plamo:image> placeholder(s) for {} image(s)",
            tiles_per_image.len()
        ));
    }
    let total_image_tokens: usize = tiles_per_image.iter().map(|t| t * num_image_token).sum();
    let mut out = Vec::with_capacity(tokens.len() + total_image_tokens + 2 * placeholders);
    let mut image_idx = 0;
    for &token in tokens {
        if token == ids.image {
            out.push(ids.img_start);
            out.extend(std::iter::repeat_n(
                ids.image,
                tiles_per_image[image_idx] * num_image_token,
            ));
            out.push(ids.img_end);
            image_idx += 1;
        } else {
            out.push(token);
        }
    }
    Ok((out, total_image_tokens))
}

/// Build, tokenize (with a leading BOS when `ids.bos` is set) and expand the
/// prompt.
pub fn prepare_plamo2_vl_prompt_tokens<E>(
    text: &str,
    tiles_per_image: &[usize],
    num_image_token: usize,
    ids: Plamo2VlImageTokenIds,
    encode: &mut E,
) -> Result<(Vec<i32>, Plamo2VlPromptStats), String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let (prompt, pre_templated) = build_plamo2_vl_prompt(text, tiles_per_image.len());
    let mut tokens = encode(&prompt, false);
    if let Some(bos) = ids.bos
        && tokens.first() != Some(&bos)
    {
        tokens.insert(0, bos);
    }
    let (tokens, total_image_tokens) =
        expand_plamo2_vl_image_tokens(&tokens, tiles_per_image, num_image_token, ids)?;
    Ok((
        tokens,
        Plamo2VlPromptStats {
            image_blocks: tiles_per_image.len(),
            total_tiles: tiles_per_image.iter().sum(),
            total_image_tokens,
            pre_templated,
        },
    ))
}

#[cfg(test)]
#[path = "plamo2_vl_prompt_tests.rs"]
mod tests;
