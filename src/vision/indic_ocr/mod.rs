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

//! IndicOCR page pipeline logic (layout routing, cleanup, Markdown).
//!
//! The two model stages live in `vision::detection::pp_doclayout_v3` (layout)
//! and the Qwen3.5 VLM loader (OCR). This module holds the pure logic between
//! them so the routing rules are testable without weights.

pub mod blocks;
pub mod labels;
pub mod reconstruct;

pub use blocks::{
    Block, CONTAIN, NESTED, NestMode, WRAP, blocks_from_detections, clamp_to_page, clean_layout,
    densify, prepare_blocks, resolve_nested_equations,
};
pub use labels::{
    EQUATION_PROMPT, MAX_CROP_PIXELS, MIN_CROP_PIXELS, TABLE_PROMPT_HTML, TABLE_PROMPT_MARKDOWN,
    TEXT_PROMPT, TableFormat, area_clamp, crop_for_block, is_drop_type, is_transcribed, map_label,
    prompt_for,
};
pub use reconstruct::{dehyphenate, reconstruct, repair_math};

use image::DynamicImage;

/// One recognized block paired with the crop and prompt it was sent with.
pub struct OcrRequest {
    /// Index into the input block list.
    pub block: usize,
    pub crop: DynamicImage,
    pub prompt: &'static str,
}

/// Pair every transcribable block with its area-clamped crop and prompt.
pub fn build_ocr_requests(
    blocks: &[Block],
    page: &DynamicImage,
    table_format: TableFormat,
    pad_px: i32,
    min_px: usize,
    max_px: usize,
) -> Vec<OcrRequest> {
    let mut requests = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        if is_drop_type(&block.block_type) || !is_transcribed(&block.label) {
            continue;
        }
        let Some(crop) = crop_for_block(&block.bbox, page, pad_px) else {
            continue;
        };
        requests.push(OcrRequest {
            block: index,
            crop: area_clamp(crop, min_px, max_px),
            prompt: prompt_for(&block.block_type, table_format),
        });
    }
    requests
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            width,
            height,
            image::Rgb([9, 9, 9]),
        ))
    }

    fn block(order: usize, label: &str, bbox: [f32; 4]) -> Block {
        Block::new(order, label, bbox, 0.9)
    }

    #[test]
    fn ocr_requests_skip_figures_marginalia_and_collapsed_boxes() {
        let page = solid(400, 400);
        let blocks = vec![
            block(0, "Paragraph", [10.0, 10.0, 200.0, 200.0]),
            block(1, "Diagram", [210.0, 10.0, 300.0, 100.0]),
            block(2, "Header", [10.0, 210.0, 200.0, 240.0]),
            block(3, "Table", [210.0, 210.0, 210.0, 240.0]),
            block(4, "Equation", [210.0, 250.0, 300.0, 300.0]),
        ];
        let requests = build_ocr_requests(
            &blocks,
            &page,
            TableFormat::Html,
            0,
            MIN_CROP_PIXELS,
            MAX_CROP_PIXELS,
        );
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].block, 0);
        assert_eq!(requests[0].prompt, TEXT_PROMPT);
        assert_eq!(requests[1].block, 4);
        assert_eq!(requests[1].prompt, EQUATION_PROMPT);
        // The paragraph crop is upscaled into the minimum area band.
        assert!(requests[0].crop.width() * requests[0].crop.height() >= MIN_CROP_PIXELS as u32);
    }
}
