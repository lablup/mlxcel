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

use super::{
    VlmPreparationSummary, expand_gemma4_audio_tokens_for_server,
    expand_gemma4_unified_video_tokens, expand_gemma4_video_tokens,
    format_molmo_v1_prompt_for_processor, prepared_embedding_refs,
    shift_molmo_v1_image_input_idx_for_bos, should_prepare_vlm_embeddings,
};
use crate::vision::merge::InputEmbeddings;
use crate::vlm_prompt::{ImageTokenBlockAction, ImageTokenBlockStats};
use mlxcel_core::{self, UniquePtr, dtype};

#[test]
fn should_prepare_vlm_embeddings_rejects_non_vlm_image_requests() {
    let err = should_prepare_vlm_embeddings(1, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Images provided but model is not a vision-language model"));
}

#[test]
fn should_prepare_vlm_embeddings_accepts_vlm_image_requests() {
    assert!(should_prepare_vlm_embeddings(2, true).unwrap());
    assert!(!should_prepare_vlm_embeddings(0, true).unwrap());
}

#[test]
fn image_block_summary_preserves_stats_shape() {
    let summary = VlmPreparationSummary::ImageBlocks(ImageTokenBlockStats {
        action: ImageTokenBlockAction::Inserted { image_blocks: 2 },
        tokens_per_image: 256,
    });

    assert_eq!(
        summary,
        VlmPreparationSummary::ImageBlocks(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Inserted { image_blocks: 2 },
            tokens_per_image: 256,
        })
    );
}

#[test]
fn prepared_embedding_refs_requires_input_embeddings() {
    let embeddings = InputEmbeddings {
        inputs_embeds: UniquePtr::null(),
        attention_mask_4d: None,
    };

    let err = match prepared_embedding_refs(&embeddings) {
        Ok(_) => panic!("expected missing input embeddings to fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("missing input embeddings"));
}

#[test]
fn prepared_embedding_refs_rejects_null_attention_masks() {
    let embeddings = InputEmbeddings {
        inputs_embeds: mlxcel_core::ones(&[1, 2], dtype::FLOAT32),
        attention_mask_4d: Some(UniquePtr::null()),
    };

    let err = match prepared_embedding_refs(&embeddings) {
        Ok(_) => panic!("expected null attention mask to fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("null 4D attention mask"));
}

#[test]
fn molmo_v1_prompt_matches_model_processor_role_format() {
    assert_eq!(
        format_molmo_v1_prompt_for_processor("What is in this image?"),
        " User: What is in this image? Assistant:"
    );
}

#[test]
fn molmo_v1_prompt_does_not_double_wrap_chat_template_output() {
    assert_eq!(
        format_molmo_v1_prompt_for_processor("User: What is in this image? Assistant:"),
        " User: What is in this image? Assistant:"
    );
}

#[test]
fn molmo_v1_prompt_strips_image_placeholder() {
    assert_eq!(
        format_molmo_v1_prompt_for_processor("<|image|> What is in this image?"),
        " User: What is in this image? Assistant:"
    );
}

#[test]
fn molmo_v1_image_input_idx_shifts_only_valid_positions_for_bos() {
    assert_eq!(
        shift_molmo_v1_image_input_idx_for_bos(&[-100, -1, 0, 12, 313]),
        vec![-100, -1, 1, 13, 314]
    );
}

// -- Gemma 4 video token expansion ----------------------

#[test]
fn expand_gemma4_video_tokens_replaces_explicit_placeholder() {
    // BOS + video_token + suffix
    let mut prompt = vec![1, 100, 7];
    let frames = vec![vec![3, 3]]; // one video, two frames, 3 soft tokens each
    expand_gemma4_video_tokens(&mut prompt, 100, 200, 201, 202, &frames).unwrap();
    // Expected:
    //   BOS=1
    //   <boi>=201, image=200,200,200, <eoi>=202   (frame 1)
    //   <boi>=201, image=200,200,200, <eoi>=202   (frame 2)
    //   suffix=7
    assert_eq!(
        prompt,
        vec![1, 201, 200, 200, 200, 202, 201, 200, 200, 200, 202, 7]
    );
}

#[test]
fn expand_gemma4_video_tokens_inserts_after_bos_when_no_placeholder() {
    let mut prompt = vec![1, 7, 8];
    let frames = vec![vec![2]]; // one video, one frame, 2 soft tokens
    expand_gemma4_video_tokens(&mut prompt, 100, 200, 201, 202, &frames).unwrap();
    // BOS=1, <boi>=201, image=200,200, <eoi>=202, suffix=7,8
    assert_eq!(prompt, vec![1, 201, 200, 200, 202, 7, 8]);
}

#[test]
fn expand_gemma4_video_tokens_handles_multiple_videos() {
    let mut prompt = vec![1, 100, 100, 7];
    // Two videos: 1 frame x 2 tokens, 2 frames x 1 token each.
    let frames = vec![vec![2], vec![1, 1]];
    expand_gemma4_video_tokens(&mut prompt, 100, 200, 201, 202, &frames).unwrap();
    assert_eq!(
        prompt,
        vec![
            1, // BOS
            201, 200, 200, 202, // video 1 frame 1
            201, 200, 202, // video 2 frame 1
            201, 200, 202, // video 2 frame 2
            7,
        ]
    );
}

#[test]
fn expand_gemma4_video_tokens_errors_on_count_mismatch() {
    let mut prompt = vec![1, 100, 100, 7];
    let frames = vec![vec![2]]; // only one video for two placeholders
    let err = expand_gemma4_video_tokens(&mut prompt, 100, 200, 201, 202, &frames).unwrap_err();
    assert!(err.to_string().contains("video placeholder"));
}

#[test]
fn expand_gemma4_video_tokens_no_op_when_videos_empty() {
    let mut prompt = vec![1, 7, 8];
    let original = prompt.clone();
    expand_gemma4_video_tokens(&mut prompt, 100, 200, 201, 202, &[]).unwrap();
    assert_eq!(prompt, original);
}

// ── Gemma 4 Unified video token expansion (issue #164) ────────────────────────
// Per-frame soft tokens are the VIDEO token id (not the image token id), framed
// BOI video* EOI per frame. video_token_id = 100, boi = 201, eoi = 202.

#[test]
fn expand_gemma4_unified_video_tokens_splices_after_bos_when_no_placeholder() {
    // CLI path: no <|video|> placeholder in the prompt, two frames of 2 soft
    // tokens each spliced after BOS.
    let mut prompt = vec![1, 7, 8];
    let frames = vec![vec![2, 2]]; // one video, two frames
    expand_gemma4_unified_video_tokens(&mut prompt, 100, 201, 202, &frames).unwrap();
    assert_eq!(
        prompt,
        vec![
            1, // BOS
            201, 100, 100, 202, // frame 1: BOI video video EOI
            201, 100, 100, 202, // frame 2: BOI video video EOI
            7, 8,
        ]
    );
}

#[test]
fn expand_gemma4_unified_video_tokens_replaces_placeholder() {
    // Server/chat-template path: a single <|video|> (video_token_id=100)
    // placeholder is replaced by its video's per-frame runs. The replacement
    // scan does not re-expand the video tokens it just inserted.
    let mut prompt = vec![1, 100, 7];
    let frames = vec![vec![3]]; // one video, one frame of 3 soft tokens
    expand_gemma4_unified_video_tokens(&mut prompt, 100, 201, 202, &frames).unwrap();
    assert_eq!(prompt, vec![1, 201, 100, 100, 100, 202, 7]);
}

#[test]
fn expand_gemma4_unified_video_tokens_errors_on_count_mismatch() {
    let mut prompt = vec![1, 100, 100, 7]; // two placeholders
    let frames = vec![vec![2]]; // only one video
    let err = expand_gemma4_unified_video_tokens(&mut prompt, 100, 201, 202, &frames).unwrap_err();
    assert!(err.to_string().contains("video placeholder"));
}

#[test]
fn expand_gemma4_unified_video_tokens_no_op_when_videos_empty() {
    let mut prompt = vec![1, 7, 8];
    let original = prompt.clone();
    expand_gemma4_unified_video_tokens(&mut prompt, 100, 201, 202, &[]).unwrap();
    assert_eq!(prompt, original);
}

// Audio token ids used across the audio-placement tests below.
const AUDIO: i32 = 50; // audio_token_id
const BOA: i32 = 51; // boa_token_id
const EOA: i32 = 52; // eoa_token_id
const EOT: i32 = 106; // <end_of_turn>
const SOT: i32 = 105; // <start_of_turn>

#[test]
fn server_audio_wraps_rendered_placeholder_in_place() {
    // A rendered `<|audio|>` (AUDIO) already sits in the user turn: wrap the
    // first occurrence as BOA + AUDIO*N + EOA in place, leaving the turn
    // structure intact.
    // [BOS, <sot>, user_text, AUDIO, <eot>, <sot>, model]
    let mut prompt = vec![2, SOT, 7, AUDIO, EOT, SOT, 8];
    expand_gemma4_audio_tokens_for_server(&mut prompt, AUDIO, BOA, EOA, 3, Some(EOT));
    assert_eq!(
        prompt,
        vec![2, SOT, 7, BOA, AUDIO, AUDIO, AUDIO, EOA, EOT, SOT, 8]
    );
}

#[test]
fn server_audio_inserts_inside_last_user_turn() {
    // Text-only server render: no AUDIO placeholder. The block must land
    // before the user turn's closing `<end_of_turn>` so it stays in the user
    // turn, not the model turn (issue #437).
    // [BOS, <sot>, user, text, <eot>, <sot>, model]
    let mut prompt = vec![2, SOT, 11, 7, EOT, SOT, 8];
    expand_gemma4_audio_tokens_for_server(&mut prompt, AUDIO, BOA, EOA, 3, Some(EOT));
    // Block spliced before the EOT at index 4, i.e. after the user text.
    assert_eq!(
        prompt,
        vec![2, SOT, 11, 7, BOA, AUDIO, AUDIO, AUDIO, EOA, EOT, SOT, 8]
    );
    // The audio block precedes the (only) `<end_of_turn>`, never the trailing
    // `<start_of_turn>model` generation prompt.
    let eot_pos = prompt.iter().position(|&t| t == EOT).unwrap();
    let eoa_pos = prompt.iter().position(|&t| t == EOA).unwrap();
    assert!(eoa_pos < eot_pos);
}

#[test]
fn server_audio_targets_the_last_user_turn_in_multi_turn() {
    // Multi-turn prompt: user / model / user, then the generation prompt. The
    // block must go before the LAST `<end_of_turn>` (closing the latest user
    // turn), not an earlier one.
    // idx: 0   1    2   3    4    5   6    7    8   9    10   11
    //     [BOS,<sot>,u1, EOT,<sot>,m1, EOT,<sot>,u2, EOT,<sot>,model]
    let mut prompt = vec![2, SOT, 21, EOT, SOT, 31, EOT, SOT, 22, EOT, SOT, 8];
    expand_gemma4_audio_tokens_for_server(&mut prompt, AUDIO, BOA, EOA, 2, Some(EOT));
    assert_eq!(
        prompt,
        vec![
            2, SOT, 21, EOT, SOT, 31, EOT, SOT, 22, BOA, AUDIO, AUDIO, EOA, EOT, SOT, 8
        ]
    );
    // The first two `<end_of_turn>` markers stay adjacent to their turn text;
    // only the final user turn gains the audio block.
    assert_eq!(prompt[3], EOT);
    assert_eq!(prompt[6], EOT);
}

#[test]
fn server_audio_falls_back_before_last_token_without_end_of_turn() {
    // No placeholder and no `<end_of_turn>` id (e.g. --no-chat-template): keep
    // the historical "before the final token" insertion as a last resort.
    let mut prompt = vec![2, 11, 8];
    expand_gemma4_audio_tokens_for_server(&mut prompt, AUDIO, BOA, EOA, 2, None);
    assert_eq!(prompt, vec![2, 11, BOA, AUDIO, AUDIO, EOA, 8]);
}

#[test]
fn server_audio_falls_back_when_end_of_turn_absent_from_prompt() {
    // `<end_of_turn>` id is known but the prompt does not contain it: the
    // last-resort fallback still fires rather than dropping the audio block.
    let mut prompt = vec![2, 11, 8];
    expand_gemma4_audio_tokens_for_server(&mut prompt, AUDIO, BOA, EOA, 2, Some(EOT));
    assert_eq!(prompt, vec![2, 11, BOA, AUDIO, AUDIO, EOA, 8]);
}
