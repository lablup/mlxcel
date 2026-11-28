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

//! Layout cleanup for IndicOCR.
//!
//! Port of `blocks.py` (`idp_blocks.py`): duplicate suppression, one
//! header/footer, nested-equation resolution and dense renumbering. Pure
//! geometry, so the rules that decide what gets transcribed are testable
//! without a model.

use crate::vision::detection::Detection;

use super::labels::{is_drop_type, is_transcribed, map_label};

/// Duplicate containment threshold.
pub const CONTAIN: f32 = 0.90;
/// Header/footer wrap threshold.
pub const WRAP: f32 = 0.5;
/// Nested equation containment threshold.
pub const NESTED: f32 = 0.70;

const TEXTLIKE: [&str; 5] = ["Text", "Title", "SectionHeader", "Caption", "Footnote"];

/// One detected region. `text` is `None` before OCR and `Some("")` for a block
/// deliberately left untranscribed.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// Zero-based reading order.
    pub order: usize,
    /// Detector label, kept with its original spelling.
    pub label: String,
    /// Pipeline type produced by [`map_label`].
    pub block_type: String,
    /// `[left, top, right, bottom]` in page pixels.
    pub bbox: [f32; 4],
    pub conf: f32,
    pub text: Option<String>,
}

impl Block {
    pub fn new(order: usize, label: &str, bbox: [f32; 4], conf: f32) -> Self {
        Self {
            order,
            label: label.to_string(),
            block_type: map_label(label).to_string(),
            bbox,
            conf,
            text: None,
        }
    }
}

/// How nested equations are collapsed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NestMode {
    /// An equation inside text or another equation is dropped.
    #[default]
    Both,
    /// Only an equation nested in another equation is dropped.
    TextOnly,
    /// Only an equation nested in text is dropped.
    EqOnly,
}

pub fn area(bbox: &[f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

/// Fraction of `small` that lies inside `big`.
pub fn contained_frac(small: &[f32; 4], big: &[f32; 4]) -> f32 {
    let ix0 = small[0].max(big[0]);
    let iy0 = small[1].max(big[1]);
    let ix1 = small[2].min(big[2]);
    let iy1 = small[3].min(big[3]);
    let intersection = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
    let small_area = area(small);
    if small_area > 0.0 {
        intersection / small_area
    } else {
        0.0
    }
}

pub fn clamp_to_page(bbox: &[f32; 4], width: u32, height: u32) -> [f32; 4] {
    [
        bbox[0].max(0.0),
        bbox[1].max(0.0),
        bbox[2].min(width as f32),
        bbox[3].min(height as f32),
    ]
}

fn transcribed(block: &Block) -> bool {
    !is_drop_type(&block.block_type) && is_transcribed(&block.label)
}

/// Drop duplicate and spurious boxes. Survivors keep input order.
pub fn clean_layout(blocks: &[Block], contain: f32, wrap: f32) -> Vec<Block> {
    let n = blocks.len();
    let mut drop = vec![false; n];
    let marginalia = ["PageHeader", "PageFooter", "PageNumber"];
    let head_foot = ["PageHeader", "PageFooter"];

    let groups: [&dyn Fn(&Block) -> bool; 2] = [
        &|block: &Block| !marginalia.contains(&block.block_type.as_str()),
        &|block: &Block| {
            marginalia.contains(&block.block_type.as_str())
                && !head_foot.contains(&block.block_type.as_str())
        },
    ];
    for in_group in groups {
        let mut indices: Vec<usize> = (0..n).filter(|&index| in_group(&blocks[index])).collect();
        indices.sort_by(|&left, &right| {
            area(&blocks[right].bbox).total_cmp(&area(&blocks[left].bbox))
        });
        let mut kept: Vec<usize> = Vec::new();
        for index in indices {
            let has_text = transcribed(&blocks[index]);
            let inside_kept = kept.iter().any(|&other| {
                contained_frac(&blocks[index].bbox, &blocks[other].bbox) >= contain
                    && (transcribed(&blocks[other]) || !has_text)
            });
            if inside_kept {
                drop[index] = true;
            } else {
                kept.push(index);
            }
        }
    }

    for block_type in head_foot {
        let group: Vec<usize> = (0..n)
            .filter(|&index| blocks[index].block_type == block_type && !drop[index])
            .collect();
        if group.is_empty() {
            continue;
        }
        let biggest = *group
            .iter()
            .max_by(|&&left, &&right| {
                area(&blocks[left].bbox).total_cmp(&area(&blocks[right].bbox))
            })
            .expect("group is non-empty");
        for &index in &group {
            if index != biggest {
                drop[index] = true;
            }
        }
        let wraps = (0..n).any(|index| {
            blocks[index].block_type != block_type
                && contained_frac(&blocks[index].bbox, &blocks[biggest].bbox) >= wrap
        });
        if !wraps {
            drop[biggest] = true;
        }
    }

    blocks
        .iter()
        .enumerate()
        .filter(|(index, _)| !drop[*index])
        .map(|(_, block)| block.clone())
        .collect()
}

fn nested_in(inner: &[f32; 4], outer: &[f32; 4], threshold: f32) -> bool {
    let inner_area = area(inner);
    if inner_area <= 0.0 {
        return false;
    }
    contained_frac(inner, outer) >= threshold && inner_area < 0.95 * area(outer)
}

fn absorbs_equation(container: &Block, mode: NestMode) -> bool {
    if container.block_type == "Equation" {
        return mode != NestMode::TextOnly;
    }
    if TEXTLIKE.contains(&container.block_type.as_str()) {
        return mode != NestMode::EqOnly;
    }
    false
}

/// Drop equation boxes nested in a container that already covers them.
pub fn resolve_nested_equations(
    blocks: &[Block],
    nest: bool,
    mode: NestMode,
    nested: f32,
) -> Vec<Block> {
    if !nest {
        return blocks.to_vec();
    }
    let mut drop = vec![false; blocks.len()];
    for (index, inner) in blocks.iter().enumerate() {
        if inner.block_type != "Equation" {
            continue;
        }
        for (other, container) in blocks.iter().enumerate() {
            if index != other
                && nested_in(&inner.bbox, &container.bbox, nested)
                && absorbs_equation(container, mode)
            {
                drop[index] = true;
                break;
            }
        }
    }
    blocks
        .iter()
        .enumerate()
        .filter(|(index, _)| !drop[*index])
        .map(|(_, block)| block.clone())
        .collect()
}

/// Sort by detector reading order and renumber to a gap-free 0-based rank.
pub fn densify(blocks: &[Block]) -> Vec<Block> {
    let mut ordered = blocks.to_vec();
    ordered.sort_by_key(|block| block.order);
    for (rank, block) in ordered.iter_mut().enumerate() {
        block.order = rank;
    }
    ordered
}

/// Convert detector output to pixel-coordinate OCR blocks.
pub fn blocks_from_detections(detections: &[Detection], width: u32, height: u32) -> Vec<Block> {
    detections
        .iter()
        .enumerate()
        .map(|(index, detection)| {
            let order = detection
                .reading_order
                .map(|order| order.saturating_sub(1) as usize)
                .unwrap_or(index);
            let clamped = clamp_to_page(&detection.bbox, width, height);
            Block {
                order,
                label: detection.class_name.clone(),
                block_type: map_label(&detection.class_name).to_string(),
                bbox: [
                    round_decimal(clamped[0], 1),
                    round_decimal(clamped[1], 1),
                    round_decimal(clamped[2], 1),
                    round_decimal(clamped[3], 1),
                ],
                conf: round_decimal(detection.score, 3),
                text: None,
            }
        })
        .collect()
}

/// Clamp, clean and renumber detector blocks in one step.
pub fn prepare_blocks(blocks: Vec<Block>) -> Vec<Block> {
    densify(&clean_layout(&blocks, CONTAIN, WRAP))
}

fn round_decimal(value: f32, places: u32) -> f32 {
    let scale = 10f32.powi(places as i32) as f64;
    ((value as f64 * scale).round_ties_even() / scale) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(order: usize, label: &str, bbox: [f32; 4]) -> Block {
        Block::new(order, label, bbox, 0.9)
    }

    #[test]
    fn nested_duplicates_are_dropped_but_disjoint_boxes_survive() {
        let blocks = vec![
            block(0, "Paragraph", [0.0, 0.0, 100.0, 100.0]),
            block(1, "Paragraph", [10.0, 10.0, 90.0, 90.0]),
            block(2, "Paragraph", [200.0, 0.0, 300.0, 100.0]),
        ];
        let kept = clean_layout(&blocks, CONTAIN, WRAP);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].bbox, [0.0, 0.0, 100.0, 100.0]);
        assert_eq!(kept[1].bbox, [200.0, 0.0, 300.0, 100.0]);
    }

    #[test]
    fn only_the_largest_header_survives_and_an_empty_one_is_dropped() {
        let blocks = vec![
            block(0, "Header", [0.0, 0.0, 100.0, 20.0]),
            block(1, "Header", [0.0, 0.0, 100.0, 10.0]),
            // A page number wraps the header, so the largest header is kept.
            block(2, "Page-number", [80.0, 2.0, 95.0, 18.0]),
        ];
        let kept = clean_layout(&blocks, CONTAIN, WRAP);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].bbox, [0.0, 0.0, 100.0, 20.0]);
        assert_eq!(kept[1].label, "Page-number");
    }

    #[test]
    fn a_footer_wrapping_nothing_is_dropped() {
        let blocks = vec![block(0, "Footer", [0.0, 95.0, 100.0, 100.0])];
        assert!(clean_layout(&blocks, CONTAIN, WRAP).is_empty());
    }

    #[test]
    fn nested_equation_resolution_respects_the_mode() {
        let blocks = vec![
            block(0, "Paragraph", [0.0, 0.0, 100.0, 100.0]),
            block(1, "Equation", [20.0, 20.0, 80.0, 40.0]),
        ];
        let both = resolve_nested_equations(&blocks, true, NestMode::Both, NESTED);
        assert_eq!(both.len(), 1);
        let text_only = resolve_nested_equations(&blocks, true, NestMode::TextOnly, NESTED);
        assert_eq!(text_only.len(), 1);
        let eq_only = resolve_nested_equations(&blocks, true, NestMode::EqOnly, NESTED);
        assert_eq!(eq_only.len(), 2);
        let off = resolve_nested_equations(&blocks, false, NestMode::Both, NESTED);
        assert_eq!(off.len(), 2);
    }

    #[test]
    fn densify_sorts_and_renumbers_reading_order() {
        let blocks = vec![
            block(7, "Paragraph", [0.0, 0.0, 1.0, 1.0]),
            block(2, "Title", [0.0, 2.0, 1.0, 3.0]),
            block(5, "Paragraph", [0.0, 4.0, 1.0, 5.0]),
        ];
        let dense = densify(&blocks);
        assert_eq!(
            dense.iter().map(|block| block.order).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(dense[0].label, "Title");
    }

    #[test]
    fn detection_blocks_use_one_based_reading_order_and_clamp() {
        let detections = vec![
            Detection {
                bbox: [-5.0, 10.0, 50.0, 90.0],
                score: 0.91234,
                label: 4,
                class_name: "Title".into(),
                reading_order: Some(2),
                mask: None,
            },
            Detection {
                bbox: [10.0, 10.0, 90.0, 95.0],
                score: 0.8,
                label: 1,
                class_name: "Paragraph".into(),
                reading_order: Some(1),
                mask: None,
            },
        ];
        let blocks = blocks_from_detections(&detections, 80, 90);
        assert_eq!(blocks[0].order, 1);
        assert_eq!(blocks[0].block_type, "Title");
        assert_eq!(blocks[0].bbox, [0.0, 10.0, 50.0, 90.0]);
        assert!((blocks[0].conf - 0.912).abs() < 1e-6);
        let dense = densify(&blocks);
        assert_eq!(dense[0].label, "Paragraph");
        assert_eq!(dense[0].order, 0);
        assert_eq!(dense[1].order, 1);
    }
}
