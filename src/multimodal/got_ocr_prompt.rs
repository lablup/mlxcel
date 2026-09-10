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

//! Prompt shaping for GOT-OCR 2.0 (`model_type: "GOT"`).
//!
//! `stepfun-ai/GOT-OCR2_0` ships no chat template. Its `modeling_GOT.py`
//! `chat()` builds one fixed MPT-style conversation and every released
//! checkpoint was trained against exactly that framing:
//!
//! ```text
//! <|im_start|>system
//!         You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
//! <img><imgpad>x256</img>
//! OCR: <|im_end|><|im_start|>assistant
//! ```
//!
//! Three details are load-bearing and easy to lose. The system line carries
//! eight literal spaces of Python source indentation (the reference builds it
//! from a triple-quoted literal inside a method body). `<|im_end|>` closes the
//! system turn with no newline after it. And the image block is emitted as
//! *text*: `<img>`, `<imgpad>` and `</img>` are real vocabulary entries
//! (151857 / 151859 / 151858) once the tiktoken loader selects the QWen
//! special table, so the runtime only has to tokenize the assembled string and
//! the 256 `<imgpad>` positions land where [`crate::vision::merge::merge_llava`]
//! scatters the tower's 256 feature rows.
//!
//! Reference: `modeling_GOT.py::GOTQwenForCausalLM.chat` in
//! <https://huggingface.co/stepfun-ai/GOT-OCR2_0>.

/// Image-block opening tag (id 151857 on the released checkpoints).
pub const GOT_IM_START_TAG: &str = "<img>";

/// Image-block closing tag (id 151858).
pub const GOT_IM_END_TAG: &str = "</img>";

/// Image-feature placeholder (id 151859), repeated `image_token_len` times.
pub const GOT_IM_PATCH_TAG: &str = "<imgpad>";

/// Turn separator, and the stop string upstream's `KeywordsStoppingCriteria`
/// watches for (id 151645).
pub const GOT_SEP: &str = "<|im_end|>";

/// User-turn opener, including its trailing newline.
pub const GOT_USER_ROLE: &str = "<|im_start|>user\n";

/// Assistant-turn opener, including its trailing newline.
pub const GOT_ASSISTANT_ROLE: &str = "<|im_start|>assistant\n";

/// The fixed system turn, byte-for-byte from `modeling_GOT.py` including the
/// eight leading spaces the Python source indentation contributes.
pub const GOT_SYSTEM: &str = "<|im_start|>system\n        You should follow the instructions carefully and explain your answers in detail.";

/// The generic `<image>` marker other mlxcel prompt paths emit.
const GENERIC_IMAGE_MARKER: &str = "<image>";

/// Summary of what [`build_got_prompt`] assembled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GotPromptStats {
    /// Image blocks placed (0 or 1; the decoder was trained with one).
    pub image_blocks: usize,
    /// `<imgpad>` placeholders emitted in total.
    pub total_image_tokens: usize,
    /// Whether the caller's text already carried the GOT conversation framing
    /// and was therefore used as-is rather than wrapped.
    pub pre_templated: bool,
}

/// `<img>` + `<imgpad>` * `image_token_len` + `</img>`.
pub fn got_image_block(image_token_len: usize) -> String {
    let mut block = String::with_capacity(
        GOT_IM_START_TAG.len() + GOT_IM_PATCH_TAG.len() * image_token_len + GOT_IM_END_TAG.len(),
    );
    block.push_str(GOT_IM_START_TAG);
    for _ in 0..image_token_len {
        block.push_str(GOT_IM_PATCH_TAG);
    }
    block.push_str(GOT_IM_END_TAG);
    block
}

/// Whether `text` already carries the GOT conversation framing.
///
/// Either marker is enough: the builtin chat template renders both, and a
/// caller passing a hand-written conversation may end a prefill mid-turn with
/// only the separator present.
pub fn is_got_templated(text: &str) -> bool {
    text.contains(GOT_SEP) || text.contains(GOT_USER_ROLE)
}

/// Assemble the prompt GOT-OCR 2.0 was trained on.
///
/// `text` is whatever reached the runtime: the raw `-p` instruction on the CLI
/// (this checkpoint family ships no template of its own, so nothing rewrites
/// it), or the render of the builtin GOT template on the server. Placement
/// rules, in order:
///
/// 1. An explicit `<image>` marker is replaced in place by the block, so a
///    caller that positioned the image keeps that position.
/// 2. Otherwise, in already-framed text the block is spliced immediately after
///    the first `<|im_start|>user\n`, which is where upstream puts it. Blindly
///    prepending would place it ahead of the *system* turn.
/// 3. Otherwise the text is a bare instruction: the block, a newline, then the
///    instruction, all wrapped in the fixed conversation.
///
/// Already-framed text is never re-wrapped, so a server render and a CLI run of
/// the same instruction converge on one prompt instead of nesting two system
/// turns.
///
/// `image_count` above 1 is rejected: the decoder was trained with a single
/// 256-token block and upstream's `forward` asserts one `</img>` per `<img>`.
pub fn build_got_prompt(
    text: &str,
    image_count: usize,
    image_token_len: usize,
) -> Result<(String, GotPromptStats), String> {
    if image_count > 1 {
        return Err(format!(
            "GOT-OCR 2.0 accepts one image per request, got {image_count}"
        ));
    }
    let pre_templated = is_got_templated(text);
    let block = if image_count == 0 {
        String::new()
    } else {
        got_image_block(image_token_len)
    };

    let body = if text.contains(GENERIC_IMAGE_MARKER) {
        if image_count == 0 {
            text.replace(GENERIC_IMAGE_MARKER, "")
        } else {
            text.replacen(GENERIC_IMAGE_MARKER, &block, 1)
                .replace(GENERIC_IMAGE_MARKER, "")
        }
    } else if image_count == 0 {
        text.to_string()
    } else if pre_templated {
        match text.find(GOT_USER_ROLE) {
            Some(pos) => {
                let split = pos + GOT_USER_ROLE.len();
                format!("{}{block}\n{}", &text[..split], &text[split..])
            }
            // Framed only by a separator (a prefill continuation, say): the
            // block still has to lead the user content, so it goes in front.
            None => format!("{block}\n{text}"),
        }
    } else {
        format!("{block}\n{text}")
    };

    let prompt = if pre_templated {
        body
    } else {
        format!("{GOT_SYSTEM}{GOT_SEP}{GOT_USER_ROLE}{body}{GOT_SEP}{GOT_ASSISTANT_ROLE}")
    };

    let total_image_tokens = image_count * image_token_len;
    Ok((
        prompt,
        GotPromptStats {
            image_blocks: image_count,
            total_image_tokens,
            pre_templated,
        },
    ))
}

/// Build the prompt and tokenize it.
///
/// GOT adds no BOS: `tokenizer([prompt])` on the QWen tiktoken tokenizer has no
/// post-processor, and the framing opens with `<|im_start|>` instead.
pub fn prepare_got_prompt_tokens<E>(
    text: &str,
    image_count: usize,
    image_token_len: usize,
    encode: &mut E,
) -> Result<(Vec<i32>, GotPromptStats), String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let (prompt, stats) = build_got_prompt(text, image_count, image_token_len)?;
    Ok((encode(&prompt, false), stats))
}

#[cfg(test)]
#[path = "got_ocr_prompt_tests.rs"]
mod tests;
