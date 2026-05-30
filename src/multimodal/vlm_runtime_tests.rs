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
    VlmPreparationSummary, expand_gemma4_video_tokens, format_molmo_v1_prompt_for_processor,
    prepared_embedding_refs, shift_molmo_v1_image_input_idx_for_bos, should_prepare_vlm_embeddings,
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
