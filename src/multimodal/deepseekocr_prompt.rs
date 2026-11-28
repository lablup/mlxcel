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

//! DeepSeek-OCR prompt shaping and image-token expansion.
//!
//! Each `<image>` (id 128815) placeholder expands to a flat run whose length is
//! the per-image placeholder count decided by the processor (global mosaic +
//! view separator, plus the tile mosaic when the image is cropped). The run
//! tokens are all identical; the vision features fill them in feature order.
//!
//! DeepSeek-OCR and DeepSeek-OCR 2 build their prompt the way
//! `modeling_deepseekocr.py::infer` (`modeling_deepseekocr2.py` is the same
//! here) does, through [`prepare_deepseekocr_prompt_tokens`]:
//!
//! ```text
//! <bos> <image>xN \n Free OCR.
//! ```
//!
//! The reference takes the prompt with its `<image>` marker already written
//! (`<image>\nFree OCR.`, `<image>\n<|grounding|>Convert the document to
//! markdown.`), strips it (`format_messages`, `sft_format='plain'`, no role
//! text), splits it on `<image>`, tokenizes each piece without special tokens
//! and prepends id 0. So the model reads a newline after the image run and
//! nothing after the instruction.
//!
//! Unlimited-OCR does not go through that builder: its reference writes one
//! `<image>` for all pages with no newline (mlx-vlm renders `<image>Free
//! OCR.` for it), which is what [`insert_unlimited_ocr_image_tokens`]
//! produces.
//!
//! Reference: `modeling_deepseekocr.py` in
//! <https://huggingface.co/deepseek-ai/DeepSeek-OCR> and
//! `modeling_deepseekocr2.py` in
//! <https://huggingface.co/deepseek-ai/DeepSeek-OCR-2>.

/// The image placeholder as prompt text (id 128815 on the released
/// checkpoints).
pub const DEEPSEEKOCR_IMAGE_TOKEN: &str = "<image>";

/// The id the reference puts first by hand (`bos_id = 0`,
/// `<｜begin▁of▁sentence｜>`).
pub const DEEPSEEKOCR_BOS_TOKEN_ID: i32 = 0;

pub struct InsertedDeepSeekOcrTokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
}

/// The prompt text DeepSeek-OCR's reference tokenizes for `prompt` and
/// `num_images` pages.
///
/// `prompt` is what the front rendered: the instruction alone under
/// `--no-chat-template`, or the checkpoint's template around it. The MLX
/// conversions ship mlx-vlm's `{{message['content']}} ` template, so that is
/// `Free OCR. ` with a space the reference never tokenizes: its
/// `format_messages` strips both the message and the assembled prompt. This
/// strips the same way and, when the text names no `<image>` itself, puts one
/// `<image>\n` per page in front, the `IMAGE_TOKEN_NEWLINE` format mlx-vlm
/// gives `deepseekocr` and `deepseekocr_2` and the shape of every prompt the
/// model card lists. A prompt that already carries `<image>` is only
/// stripped, so nothing is doubled.
pub fn deepseekocr_prompt_text(prompt: &str, num_images: usize) -> String {
    let text = prompt.trim();
    if num_images == 0 || text.contains(DEEPSEEKOCR_IMAGE_TOKEN) {
        return text.to_string();
    }
    let mut out = format!("{DEEPSEEKOCR_IMAGE_TOKEN}\n").repeat(num_images);
    out.push_str(text);
    // An empty instruction leaves `<image>\n`, which the reference's final
    // strip reduces to `<image>`.
    out.trim_end().to_string()
}

/// Tokenize `prompt` the way DeepSeek-OCR's reference does, with each
/// `<image>` expanded to its page's placeholder run (`counts[i]`, in order).
///
/// The text is [`deepseekocr_prompt_text`]. It is split on `<image>`, each
/// piece is tokenized without special tokens, and
/// [`DEEPSEEKOCR_BOS_TOKEN_ID`] goes first, as `infer()` does. On success
/// `prompt_tokens` is replaced. Returns `None` and leaves `prompt_tokens`
/// alone when there are no pages or the text names a different number of
/// `<image>` markers than there are pages.
///
/// Used by: `multimodal::vlm_runtime` (the DeepSeek-OCR and DeepSeek-OCR 2
/// arms). Not used by Unlimited-OCR or HunyuanVL, which expand the tokens the
/// front produced ([`insert_unlimited_ocr_image_tokens`],
/// [`insert_deepseekocr_image_tokens`]).
pub fn prepare_deepseekocr_prompt_tokens<E>(
    prompt_tokens: &mut Vec<i32>,
    prompt: &str,
    counts: &[i32],
    image_token_id: i32,
    mut encode: E,
) -> Option<InsertedDeepSeekOcrTokens>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    if counts.is_empty() {
        return None;
    }
    let text = deepseekocr_prompt_text(prompt, counts.len());
    let pieces: Vec<&str> = text.split(DEEPSEEKOCR_IMAGE_TOKEN).collect();
    if pieces.len() != counts.len() + 1 {
        return None;
    }

    let total: i32 = counts.iter().sum();
    let mut tokens = Vec::with_capacity(1 + text.len() + total as usize);
    tokens.push(DEEPSEEKOCR_BOS_TOKEN_ID);
    for (idx, piece) in pieces.iter().enumerate() {
        if !piece.is_empty() {
            tokens.extend(encode(piece, false));
        }
        if let Some(&count) = counts.get(idx) {
            tokens.resize(tokens.len() + count as usize, image_token_id);
        }
    }
    *prompt_tokens = tokens;
    Some(InsertedDeepSeekOcrTokens {
        image_blocks: counts.len(),
        total_image_tokens: total,
    })
}

/// Expand each `<image>` placeholder in `prompt_tokens` to `counts[i]` copies
/// (in order). Returns `None` when there are no images or the placeholder count
/// does not match the image count.
pub fn insert_deepseekocr_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    counts: &[i32],
    image_token_id: i32,
) -> Option<InsertedDeepSeekOcrTokens> {
    if prompt_tokens.is_empty() || counts.is_empty() {
        return None;
    }
    let placeholders = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();

    // One placeholder per image: expand each in place.
    if placeholders == counts.len() {
        let total: i32 = counts.iter().sum();
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total as usize);
        let mut idx = 0usize;
        for &tok in prompt_tokens.iter() {
            if tok == image_token_id {
                for _ in 0..counts[idx] {
                    expanded.push(image_token_id);
                }
                idx += 1;
            } else {
                expanded.push(tok);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedDeepSeekOcrTokens {
            image_blocks: counts.len(),
            total_image_tokens: total,
        });
    }

    // No placeholder: splice one run per image after the first token (BOS).
    if placeholders == 0 {
        let total: i32 = counts.iter().sum();
        let bos = prompt_tokens[0];
        let rest = prompt_tokens[1..].to_vec();
        let mut out = Vec::with_capacity(prompt_tokens.len() + total as usize);
        out.push(bos);
        for &c in counts {
            for _ in 0..c {
                out.push(image_token_id);
            }
        }
        out.extend(rest);
        *prompt_tokens = out;
        return Some(InsertedDeepSeekOcrTokens {
            image_blocks: counts.len(),
            total_image_tokens: total,
        });
    }

    None
}

/// Unlimited-OCR variant: a single literal `<image>` placeholder covers *all*
/// pages, so one placeholder expands to the sum of every page's placeholder
/// count (in page order). Falls back to [`insert_deepseekocr_image_tokens`] for
/// the one-placeholder-per-image and no-placeholder layouts.
pub fn insert_unlimited_ocr_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    counts: &[i32],
    image_token_id: i32,
) -> Option<InsertedDeepSeekOcrTokens> {
    if prompt_tokens.is_empty() || counts.is_empty() {
        return None;
    }
    let placeholders = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();

    // Single `<image>` covering every page: expand it to the summed run. This
    // also matches the common single-page case (one placeholder, one count).
    if placeholders == 1 {
        let total: i32 = counts.iter().sum();
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total as usize);
        for &tok in prompt_tokens.iter() {
            if tok == image_token_id {
                for _ in 0..total {
                    expanded.push(image_token_id);
                }
            } else {
                expanded.push(tok);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedDeepSeekOcrTokens {
            image_blocks: counts.len(),
            total_image_tokens: total,
        });
    }

    insert_deepseekocr_image_tokens(prompt_tokens, counts, image_token_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: i32 = 128815;

    /// The pieces the DeepSeek-OCR prompt builder may tokenize, with the ids
    /// the checkpoints' `tokenizer.json` gives them without special tokens
    /// (the file is byte-identical in `deepseek-ai/DeepSeek-OCR`,
    /// `deepseek-ai/DeepSeek-OCR-2` and their `mlx-community` 4-bit
    /// conversions). Any other piece, a trailing space included, panics.
    fn encode(text: &str, add_special: bool) -> Vec<i32> {
        assert!(!add_special, "pieces are tokenized without special tokens");
        match text {
            "\nFree OCR." => vec![201, 21431, 126041, 16],
            "Free OCR." => vec![21431, 126041, 16],
            "\n" => vec![201],
            "\n<|grounding|>Convert the document to markdown." => {
                vec![201, 128820, 21842, 270, 4940, 304, 2121, 7919, 16]
            }
            other => panic!("unexpected piece {other:?}"),
        }
    }

    /// What the front tokenized before the builder runs: the shipped
    /// template's `Free OCR. ` with BOS, ending in the space token 223.
    fn rendered_free_ocr() -> Vec<i32> {
        vec![0, 21431, 126041, 16, 223]
    }

    /// `<image>\nFree OCR.` is the reference prompt. The front renders
    /// `Free OCR. ` (the shipped template prints a space after the message);
    /// the ids must be the reference's `[0, <image>.., 201, 21431, 126041, 16]`
    /// and not `[0, <image>.., 21431, 126041, 16, 223]`.
    #[test]
    fn deepseekocr_free_ocr_prompt_has_the_reference_ids() {
        let mut toks = rendered_free_ocr();
        let stats =
            prepare_deepseekocr_prompt_tokens(&mut toks, "Free OCR. ", &[3], IMG, encode).unwrap();
        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 3);
        assert_eq!(toks, vec![0, IMG, IMG, IMG, 201, 21431, 126041, 16]);

        // `--no-chat-template` hands over the instruction alone.
        let mut toks = vec![0, 21431, 126041, 16];
        prepare_deepseekocr_prompt_tokens(&mut toks, "Free OCR.", &[3], IMG, encode).unwrap();
        assert_eq!(toks, vec![0, IMG, IMG, IMG, 201, 21431, 126041, 16]);
    }

    /// The model card's document prompt,
    /// `<image>\n<|grounding|>Convert the document to markdown.`.
    #[test]
    fn deepseekocr_grounding_prompt_has_the_reference_ids() {
        let mut toks = vec![0, 128820, 21842, 270, 4940, 304, 2121, 7919, 16, 223];
        prepare_deepseekocr_prompt_tokens(
            &mut toks,
            "<|grounding|>Convert the document to markdown. ",
            &[2],
            IMG,
            encode,
        )
        .unwrap();
        assert_eq!(
            toks,
            vec![
                0, IMG, IMG, 201, 128820, 21842, 270, 4940, 304, 2121, 7919, 16
            ]
        );
    }

    /// A prompt that already names `<image>` keeps its own layout: the marker
    /// is not added a second time, and a caller who writes no newline after it
    /// gets none, as the reference would give them.
    #[test]
    fn deepseekocr_explicit_image_marker_is_not_doubled() {
        let mut toks = vec![0, IMG, 201, 21431, 126041, 16, 223];
        prepare_deepseekocr_prompt_tokens(&mut toks, "<image>\nFree OCR. ", &[2], IMG, encode)
            .unwrap();
        assert_eq!(toks, vec![0, IMG, IMG, 201, 21431, 126041, 16]);

        let mut toks = vec![0, IMG, 21431, 126041, 16, 223];
        prepare_deepseekocr_prompt_tokens(&mut toks, "<image>Free OCR. ", &[2], IMG, encode)
            .unwrap();
        assert_eq!(toks, vec![0, IMG, IMG, 21431, 126041, 16]);
    }

    /// The reference strips the message, so a newline or spaces the caller
    /// leaves around the instruction do not reach the model either.
    #[test]
    fn deepseekocr_prompt_is_stripped() {
        for prompt in [
            "Free OCR.\n ",
            "  Free OCR.  ",
            "\nFree OCR. ",
            "Free OCR.\n\n",
        ] {
            let mut toks = rendered_free_ocr();
            prepare_deepseekocr_prompt_tokens(&mut toks, prompt, &[1], IMG, encode).unwrap();
            assert_eq!(toks, vec![0, IMG, 201, 21431, 126041, 16], "{prompt:?}");
        }
    }

    /// Two pages: one `<image>\n` each, in page order, ahead of the
    /// instruction (`<image>\n<image>\nFree OCR.`).
    #[test]
    fn deepseekocr_each_page_gets_its_own_marker() {
        let mut toks = rendered_free_ocr();
        let stats =
            prepare_deepseekocr_prompt_tokens(&mut toks, "Free OCR. ", &[2, 3], IMG, encode)
                .unwrap();
        assert_eq!(stats.image_blocks, 2);
        assert_eq!(stats.total_image_tokens, 5);
        assert_eq!(
            toks,
            vec![0, IMG, IMG, 201, IMG, IMG, IMG, 201, 21431, 126041, 16]
        );
    }

    /// A prompt whose own `<image>` markers do not match the page count is
    /// left as the front tokenized it, as before.
    #[test]
    fn deepseekocr_marker_count_mismatch_leaves_the_tokens_alone() {
        let mut toks = vec![0, IMG, IMG, 21431, 126041, 16, 223];
        let before = toks.clone();
        assert!(
            prepare_deepseekocr_prompt_tokens(
                &mut toks,
                "<image><image>Free OCR. ",
                &[4],
                IMG,
                encode
            )
            .is_none()
        );
        assert_eq!(toks, before);

        let mut toks = rendered_free_ocr();
        assert!(
            prepare_deepseekocr_prompt_tokens(&mut toks, "Free OCR. ", &[], IMG, encode).is_none()
        );
        assert_eq!(toks, rendered_free_ocr());
    }

    #[test]
    fn deepseekocr_prompt_text_adds_markers_only_when_absent() {
        assert_eq!(
            deepseekocr_prompt_text("Free OCR. ", 1),
            "<image>\nFree OCR."
        );
        assert_eq!(
            deepseekocr_prompt_text("Free OCR. ", 2),
            "<image>\n<image>\nFree OCR."
        );
        assert_eq!(
            deepseekocr_prompt_text("<image>\nFree OCR. ", 1),
            "<image>\nFree OCR."
        );
        // No image: no marker. (A text-only request never reaches the
        // builder, so its prompt stays as the front rendered it.)
        assert_eq!(deepseekocr_prompt_text("Free OCR. ", 0), "Free OCR.");
        // No instruction at all: the final strip takes the newline too.
        assert_eq!(deepseekocr_prompt_text(" ", 1), "<image>");
    }

    /// Unlimited-OCR keeps its own layout: one run for all pages straight
    /// after BOS and no newline, the ids of mlx-vlm's `<image>Free OCR.`.
    #[test]
    fn unlimited_ocr_prompt_has_no_newline_after_the_image() {
        let mut toks = vec![0, 21431, 126041, 16];
        insert_unlimited_ocr_image_tokens(&mut toks, &[2, 1], IMG).unwrap();
        assert_eq!(toks, vec![0, IMG, IMG, IMG, 21431, 126041, 16]);
    }

    #[test]
    fn unlimited_single_image_expands_in_place() {
        let mut toks = vec![0, IMG, 5, 6];
        let stats = insert_unlimited_ocr_image_tokens(&mut toks, &[3], IMG).unwrap();
        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 3);
        assert_eq!(toks, vec![0, IMG, IMG, IMG, 5, 6]);
    }

    #[test]
    fn unlimited_single_placeholder_covers_all_pages() {
        // One literal <image> covers three pages: expand to the summed count.
        let mut toks = vec![0, IMG, 9];
        let stats = insert_unlimited_ocr_image_tokens(&mut toks, &[2, 3, 4], IMG).unwrap();
        assert_eq!(stats.image_blocks, 3);
        assert_eq!(stats.total_image_tokens, 9);
        assert_eq!(toks.iter().filter(|&&t| t == IMG).count(), 9);
        assert_eq!(toks.first().copied(), Some(0));
        assert_eq!(toks.last().copied(), Some(9));
    }

    #[test]
    fn unlimited_no_placeholder_falls_back_to_splice() {
        let mut toks = vec![0, 7, 8];
        let stats = insert_unlimited_ocr_image_tokens(&mut toks, &[2, 2], IMG).unwrap();
        assert_eq!(stats.total_image_tokens, 4);
        // Spliced after BOS.
        assert_eq!(toks, vec![0, IMG, IMG, IMG, IMG, 7, 8]);
    }

    #[test]
    fn unlimited_empty_inputs_return_none() {
        let mut empty: Vec<i32> = vec![];
        assert!(insert_unlimited_ocr_image_tokens(&mut empty, &[1], IMG).is_none());
        let mut toks = vec![0, IMG];
        assert!(insert_unlimited_ocr_image_tokens(&mut toks, &[], IMG).is_none());
    }
}
