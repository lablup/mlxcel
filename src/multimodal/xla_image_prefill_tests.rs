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

//! Per-family image expansion and the OpenXLA shape it sizes (issue #3361).
//!
//! Each test runs the expansion the family's XLA preprocessor runs, on the
//! geometry of `tests/fixtures/test_image.png` (224x224) or a large image,
//! with the processor parameters of the checkpoint the issue was verified on,
//! and pins both the expanded prompt length and the request-sized shape. The
//! old sizing added the config-derived worst case on top, which is what
//! compiled a 32768 shape for a 64-token Qwen2-VL image and a 256 shape that
//! could not hold one Gemma3 image.

use image::{DynamicImage, RgbImage};

use super::*;
use crate::FakeHostMultimodalPreprocessor;
use crate::multimodal::gemma3_vlm_prompt::expand_image_blocks_with_newline_fusion;
use crate::multimodal::phi4mm_prompt::expand_phi4mm_placeholders;
use crate::multimodal::qwen_vl::insert_qwen_vl_image_tokens;
use crate::multimodal::vlm_prompt::{ImageTokenBlockInfo, apply_image_token_blocks};
use crate::multimodal::youtu_vl_prompt::insert_youtu_vl_image_tokens;
use crate::phi4_siglip_prompt::PHI4_SIGLIP_IMAGE_TOKEN_INDEX;
use crate::vision::processors::molmo::{MolmoImageTokens, MolmoProcessor};
use crate::vision::processors::phi4mm::Phi4MMProcessor;
use crate::vision::processors::qwen2_vl::{DEFAULT_MAX_PIXELS, Qwen2VLProcessor, max_image_tokens};
use crate::vision::processors::youtu_vl::YoutuVLProcessor;

/// `-n 16`, the budget the acceptance runs use.
const BUDGET: usize = 16;

fn image(width: u32, height: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::new(width, height))
}

/// The fixture's geometry.
fn fixture() -> DynamicImage {
    image(224, 224)
}

/// The shape `mlxcel_xla::resolve_context_capacity` compiles for an unpinned
/// request: the smallest ladder rung at or above the request, never below the
/// 256 floor. Kept free of the environment so a pinned capacity in the test
/// process cannot change the answer.
fn request_shape(effective_prompt_len: usize) -> usize {
    let required = xla_request_positions(effective_prompt_len, BUDGET);
    mlxcel_xla::CONTEXT_CAPACITY_LADDER
        .iter()
        .copied()
        .find(|&rung| rung >= required.max(mlxcel_xla::DEFAULT_CONTEXT_CAPACITY))
        .expect("request fits the ladder")
}

/// Qwen-VL prompt: `<bos> <|vision_start|> <|image_pad|> <|vision_end|> text`.
fn qwen_expanded_len(processor: &Qwen2VLProcessor, img: &DynamicImage) -> (usize, usize) {
    const START: i32 = 151_652;
    const PAD: i32 = 151_655;
    const END: i32 = 151_653;
    let mut tokens = vec![1, START, PAD, END, 2, 3];
    let grids = processor.compute_grid_thw(std::slice::from_ref(img));
    insert_qwen_vl_image_tokens(&mut tokens, &grids, 2, START, PAD).expect("one image block");
    let image_tokens = tokens.iter().filter(|&&token| token == PAD).count();
    (image_tokens, tokens.len())
}

#[test]
fn qwen2_vl_sizes_the_fixture_from_its_64_merged_tokens() {
    // Qwen2-VL-2B and Qwen2.5-VL-3B share the 14px patch, 2x2 merge geometry.
    let processor = Qwen2VLProcessor::new(14, 2, 2);
    let (image_tokens, len) = qwen_expanded_len(&processor, &fixture());
    assert_eq!(image_tokens, 64);
    assert_eq!(len, 6 - 1 + 64);
    assert_eq!(request_shape(len), 256);
    // A pinned 256 shape admits it, which the worst-case sizing refused.
    assert!(len + BUDGET <= 256);

    // The worst case the old sizing added, and the shape it compiled.
    let floor = max_image_tokens(14, 2, DEFAULT_MAX_PIXELS).expect("floor");
    assert_eq!(floor, 16_384);
    assert_eq!(request_shape(len - image_tokens + floor), 32_768);
}

#[test]
fn qwen2_vl_sizes_a_large_image_from_its_resized_grid() {
    let processor = Qwen2VLProcessor::new(14, 2, 2);
    let (image_tokens, len) = qwen_expanded_len(&processor, &image(2048, 1536));
    // smart_resize keeps the area under the cap and rounds to 28px.
    assert_eq!(image_tokens, (1540 / 28) * (2044 / 28));
    assert!(image_tokens < 16_384);
    assert_eq!(request_shape(len), 4096);
}

#[test]
fn qwen3_vl_sizes_the_fixture_and_documents_the_512_patch_bound() {
    // Qwen3-VL-2B: 16px patch, 2x2 merge, the checkpoint's size bounds.
    let processor = Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3])
        .with_pixel_bounds(65_536, 16_777_216);
    let grids = processor.compute_grid_thw(&[fixture()]);
    // The fixture is below `shortest_edge`, so it is upscaled to 256x256.
    assert_eq!(grids, vec![(1, 16, 16)]);
    let (image_tokens, len) = qwen_expanded_len(&processor, &fixture());
    assert_eq!(image_tokens, 64);
    assert_eq!(request_shape(len), 256);

    // A large image keeps its resolution, so its grid exceeds the largest
    // qualified IREE vision bucket (512 patches) and the preprocessor refuses
    // it before any context shape is compiled. That bound is the vision
    // tower's, not the context's.
    let grids = processor.compute_grid_thw(&[image(1024, 768)]);
    let patches = grids
        .iter()
        .map(|&(t, h, w)| (t * h * w) as usize)
        .sum::<usize>();
    assert_eq!(patches, 48 * 64);
    assert!(patches > 512);
}

#[test]
fn youtu_vl_sizes_the_fixture_and_shrinks_a_large_image_to_its_patch_cap() {
    const START: i32 = 10;
    const PAD: i32 = 11;
    const END: i32 = 12;
    // Youtu-VL-4B: 16px patch, 2x2 merge, 256-patch IREE vision cap.
    let processor = YoutuVLProcessor::new(16, 2).with_max_patches_per_image(256);
    for (img, expected_tokens) in [(fixture(), 49), (image(2048, 1536), 64)] {
        let shapes = processor.compute_spatial_shapes(std::slice::from_ref(&img));
        let mut tokens = vec![1, START, PAD, END, 2];
        insert_youtu_vl_image_tokens(&mut tokens, &shapes, 2, START, END, PAD)
            .expect("placeholders match")
            .expect("one image block");
        let image_tokens = tokens.iter().filter(|&&token| token == PAD).count();
        let patches = shapes.iter().map(|&(h, w)| (h * w) as usize).sum::<usize>();
        assert!(patches <= 256, "{patches} patches");
        assert!(image_tokens <= expected_tokens, "{image_tokens} tokens");
        assert_eq!(request_shape(tokens.len()), 256);
    }
    let shapes = processor.compute_spatial_shapes(&[fixture()]);
    assert_eq!(shapes, vec![(14, 14)]);
}

#[test]
fn gemma3_sizes_its_272_token_prompt_past_the_256_default() {
    // gemma-3-4b-it: 256 soft tokens per image framed by BOI/EOI, with the
    // wrapper newlines fused into the template's (the 272-id prompt of #2761).
    const BOI: i32 = 255_999;
    const EOI: i32 = 256_000;
    const SOFT: i32 = 262_144;
    let newline_runs: Vec<i32> = (107..=137).collect();
    let info = crate::multimodal::host_preprocessor::gemma3_iree_token_block_info(
        SOFT, BOI, EOI, 256, 108,
    );
    let mut tokens = vec![
        2, 105, 2364, 107, BOI, 82858, 506, 2471, 236_761, 106, 107, 105, 4368, 107,
    ];
    expand_image_blocks_with_newline_fusion(&mut tokens, &info, 1, &newline_runs)
        .expect("expansion")
        .expect("one image block");
    assert_eq!(tokens.len(), 272);
    // The old sizing had no Gemma3 floor and compiled 256, which refused it.
    assert!(tokens.len() + BUDGET > 256);
    assert_eq!(request_shape(tokens.len()), 512);
}

#[test]
fn molmo_v1_sizes_the_fixture_from_its_processor_tokens() {
    // Molmo-7B-D: 12 high-res crops, 336px base, 14px patches, 12x12 pooled.
    let processor = MolmoProcessor::new(
        12,
        Some((4, 4)),
        Some(14),
        Some((336, 336)),
        Some((12, 12)),
        None,
        None,
        MolmoImageTokens::default(),
    );
    let output = processor.preprocess_image(&fixture());
    let image_tokens = output.image_token_ids.len();
    // One global crop plus one high-res crop, 12 rows of 12 patches plus a
    // column token each, framed by start/end.
    assert_eq!(image_tokens, 2 * (12 * 13 + 2));
    // `<bos>` + image block + an 8-token formatted prompt.
    let len = 1 + image_tokens + 8;
    assert_eq!(request_shape(len), 512);
    assert!(len + BUDGET > 256, "the unsized 256 shape refused it");
}

#[test]
fn phi4mm_sizes_the_fixture_after_the_image_expands() {
    // Phi-4-multimodal: 448px crops, 14px patches, dynamic_hd = 36.
    let processor = Phi4MMProcessor::new(448, 14, 36);
    let processed = processor.preprocess(&[fixture()]);
    let sizes = processed
        .iter()
        .map(|image| image.num_img_tokens)
        .collect::<Vec<_>>();
    assert_eq!(sizes, vec![545]);
    // The 8 normalized text tokens, one of which is the image sentinel.
    let text = [200_022, 1, 2, PHI4_SIGLIP_IMAGE_TOKEN_INDEX, 3, 4, 5, 6];
    let expanded = expand_phi4mm_placeholders(&text, &sizes, &[]).expect("expansion");
    assert_eq!(expanded.len(), 552);
    // Sizing from the 8 text tokens compiled 256 and refused the request.
    assert_eq!(request_shape(text.len()), 256);
    assert_eq!(request_shape(expanded.len()), 1024);
}

#[test]
fn llava_sizes_the_fixture_from_its_projector_tokens() {
    // llava-1.5-7b: (336 / 14)^2 = 576 tokens and no frame or separator.
    let info = ImageTokenBlockInfo {
        use_boi_eoi: false,
        image_token_id: 32_000,
        mm_tokens_per_image: 576,
        boi_token_id: 0,
        eoi_token_id: 0,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };
    let mut tokens = vec![1, 3148, 32_000, 29871, 13];
    apply_image_token_blocks(&mut tokens, info, 1)
        .expect("expansion")
        .expect("one image block");
    assert_eq!(tokens.len(), 5 - 1 + 576);
    assert_eq!(request_shape(tokens.len()), 1024);
}

#[test]
fn the_shared_dispatch_sizes_from_the_prepared_length() {
    let preprocessor = FakeHostMultimodalPreprocessor {
        image_token_id: -200,
        tokens_per_image: 3,
        hidden_size: 4,
        max_sequence_len: 64,
    };
    let prepared =
        prepare_image_prefill(&preprocessor, &[1, -200, 2], &[fixture()]).expect("fake prepare");
    assert!(matches!(prepared, PreparedImagePrefill::Standard(_)));
    assert_eq!(prepared.sequence_len(), 2 + 3);
    assert_eq!(prepared.sequence_len(), prepared.prepared().sequence_len);
    assert_eq!(xla_request_positions(prepared.sequence_len(), 16), 21);
    assert_eq!(xla_request_positions(usize::MAX, 1), usize::MAX);
}
