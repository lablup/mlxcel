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

//! Layout-aware Falcon-OCR post-processing.
//!
//! Falcon-OCR's two-stage mode runs a layout detector over the page, then OCRs
//! each detected region separately with a category-specific instruction. This
//! module is the second half of that pipeline: category routing, nested-box
//! suppression, region cropping, and prompt selection, ported from the
//! checkpoint's `modeling_falcon_ocr.py::generate_with_layout` and mlx-vlm's
//! `falcon_ocr/layout.py`.
//!
//! The detector itself (PP-DocLayoutV3) is a separate object-detection
//! architecture and is **not** part of this port. The stage is expressed over
//! [`crate::vision::detection::Detection`], the shape mlxcel's existing
//! RT-DETR-family detector already produces, so a future PP-DocLayoutV3 loader
//! drops straight in. Callers that have no detector run the single-region
//! `plain` path, which is what the reference also does when detection returns
//! nothing usable.

use crate::vision::detection::Detection;

/// Minimum crop side, in pixels, that is worth sending to the OCR decoder.
pub const MIN_CROP_DIM: u32 = 16;

/// The OCR output format requested through the instruction text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrCategory {
    Plain,
    Text,
    Table,
    Formula,
    Caption,
    Footnote,
    ListItem,
    PageFooter,
    PageHeader,
    SectionHeader,
    Title,
}

impl OcrCategory {
    /// The instruction sentence Falcon-OCR was trained with.
    ///
    /// `CATEGORY_PROMPTS` in `modeling_falcon_ocr.py`. Everything except
    /// `plain` interpolates the category name into the same sentence, but the
    /// mapping is spelled out rather than derived so a rename upstream shows up
    /// as a diff rather than as silently different text.
    pub fn instruction(self) -> &'static str {
        match self {
            Self::Plain | Self::Text => "Extract the text content from this image.",
            Self::Table => "Extract the table content from this image.",
            Self::Formula => "Extract the formula content from this image.",
            Self::Caption => "Extract the caption content from this image.",
            Self::Footnote => "Extract the footnote content from this image.",
            Self::ListItem => "Extract the list-item content from this image.",
            Self::PageFooter => "Extract the page-footer content from this image.",
            Self::PageHeader => "Extract the page-header content from this image.",
            Self::SectionHeader => "Extract the section-header content from this image.",
            Self::Title => "Extract the title content from this image.",
        }
    }

    /// Parse a user-facing category name (the `--ocr-category` spelling).
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "plain" => Some(Self::Plain),
            "text" => Some(Self::Text),
            "table" => Some(Self::Table),
            "formula" => Some(Self::Formula),
            "caption" => Some(Self::Caption),
            "footnote" => Some(Self::Footnote),
            "list-item" | "list_item" => Some(Self::ListItem),
            "page-footer" | "page_footer" => Some(Self::PageFooter),
            "page-header" | "page_header" => Some(Self::PageHeader),
            "section-header" | "section_header" => Some(Self::SectionHeader),
            "title" => Some(Self::Title),
            _ => None,
        }
    }
}

/// Map a layout-detector class name onto an OCR category.
///
/// `LAYOUT_TO_OCR_CATEGORY` in the reference. `None` means the region carries
/// no text and must be skipped (figures, charts, seals).
///
/// The reference's keys are PP-DocLayoutV3's spelling, which hyphenates the
/// four two-word classes (`list-item`, `page-footer`, `page-header`,
/// `section-header`). DocLayNet-derived detectors (docling-layout, and
/// therefore anything routed through `mlxcel detect`) spell the same four with
/// an underscore. Both are accepted: an unmapped class is silently skipped, so
/// without the aliases a whole page's list items and running heads would
/// disappear from the output with nothing to explain it. The single-word
/// classes are unaffected, and `paragraph_title` / `doc_title` / `figure_title`
/// keep their reference (underscore) spelling because no detector hyphenates
/// them.
pub fn layout_to_ocr_category(layout_class: &str) -> Option<OcrCategory> {
    match layout_class.trim().to_ascii_lowercase().as_str() {
        "text" | "header" | "number" | "reference_content" | "reference" | "abstract"
        | "aside_text" | "content" | "formula_number" | "algorithm" => Some(OcrCategory::Text),
        "table" => Some(OcrCategory::Table),
        "formula" => Some(OcrCategory::Formula),
        "caption" | "figure_title" => Some(OcrCategory::Caption),
        "footnote" | "vision_footnote" => Some(OcrCategory::Footnote),
        "list-item" | "list_item" => Some(OcrCategory::ListItem),
        "title" | "doc_title" => Some(OcrCategory::Title),
        "footer" | "page-footer" | "page_footer" => Some(OcrCategory::PageFooter),
        "page-header" | "page_header" => Some(OcrCategory::PageHeader),
        "paragraph_title" | "section-header" | "section_header" => Some(OcrCategory::SectionHeader),
        "image" | "picture" | "figure" | "chart" | "seal" => None,
        _ => None,
    }
}

fn box_area(bbox: &[f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

fn intersection_area(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0)
}

/// Fraction of `small` that lies inside `large`.
fn containment_ratio(small: &[f32; 4], large: &[f32; 4]) -> f32 {
    let area = box_area(small);
    if area <= 0.0 {
        return 0.0;
    }
    intersection_area(small, large) / area
}

/// Drop every box that is mostly contained in a strictly larger box.
///
/// Without this, an inline formula detected inside a paragraph is OCRed twice
/// and the page text duplicates.
///
/// The container must be *strictly* larger, matching the reference: equal-area
/// boxes would otherwise delete each other and the region would vanish
/// entirely. Collapsing the same-box duplicates a DETR head emits is a separate
/// concern handled in [`plan_layout_region_boxes`], which can consult the OCR
/// category mapping that this pure-geometry filter has no business knowing
/// about.
pub fn filter_nested_detections(
    detections: &[Detection],
    containment_threshold: f32,
) -> Vec<Detection> {
    let areas: Vec<f32> = detections.iter().map(|d| box_area(&d.bbox)).collect();
    detections
        .iter()
        .enumerate()
        .filter(|(i, det)| {
            !detections.iter().enumerate().any(|(j, other)| {
                j != *i
                    && areas[j] > areas[*i]
                    && containment_ratio(&det.bbox, &other.bbox) > containment_threshold
            })
        })
        .map(|(_, det)| det.clone())
        .collect()
}

/// The clamped `(x, y, width, height)` a bbox crops to, or `None` when the
/// region is too small to be worth OCRing.
///
/// Mirrors `layout.py::crop_region`: clamp to the image, reject anything under
/// `min_dim` on a side, and also reject a region whose short side would fall
/// below `min_dim` once the OCR resize caps the long side at `max_dim`.
///
/// Split out of [`crop_region`] because deciding a region's viability must not
/// require allocating its pixels: `image::crop_imm` copies, so planning a page
/// eagerly would hold one full crop per detection before any OCR runs.
pub fn crop_rect(
    image_width: u32,
    image_height: u32,
    bbox: &[f32; 4],
    min_dim: u32,
    max_dim: u32,
) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = (image_width as f32, image_height as f32);
    let x1 = bbox[0].round().clamp(0.0, w) as u32;
    let y1 = bbox[1].round().clamp(0.0, h) as u32;
    let x2 = bbox[2].round().clamp(0.0, w) as u32;
    let y2 = bbox[3].round().clamp(0.0, h) as u32;
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let (cw, ch) = (x2 - x1, y2 - y1);
    if cw < min_dim || ch < min_dim {
        return None;
    }
    let (short, long) = if cw < ch { (cw, ch) } else { (ch, cw) };
    if long > max_dim && (short as f32 * (max_dim as f32 / long as f32)) < min_dim as f32 {
        return None;
    }
    Some((x1, y1, cw, ch))
}

/// Crop a detected region, or `None` when it is too small to be worth OCRing.
pub fn crop_region(
    image: &image::DynamicImage,
    bbox: &[f32; 4],
    min_dim: u32,
    max_dim: u32,
) -> Option<image::DynamicImage> {
    crop_rect(image.width(), image.height(), bbox, min_dim, max_dim)
        .map(|(x, y, w, h)| image.crop_imm(x, y, w, h))
}

/// One OCR unit produced by the layout stage.
#[derive(Debug, Clone)]
pub struct LayoutOcrRegion {
    /// The detector's class name, or `"plain"` for the whole-page fallback.
    pub category: String,
    /// `(left, top, right, bottom)` in original-image pixels.
    pub bbox: [f32; 4],
    pub score: f32,
    /// The instruction to prepend for this region.
    pub ocr_category: OcrCategory,
    pub image: image::DynamicImage,
}

/// One OCR unit, planned but not yet cropped.
#[derive(Debug, Clone)]
pub struct LayoutOcrPlan {
    /// The detector's class name, or `"plain"` for the whole-page fallback.
    pub category: String,
    /// `(left, top, right, bottom)` in original-image pixels.
    pub bbox: [f32; 4],
    pub score: f32,
    /// The instruction to prepend for this region.
    pub ocr_category: OcrCategory,
    /// The clamped `(x, y, width, height)` this region crops to.
    pub crop: (u32, u32, u32, u32),
}

/// Turn a page's dimensions plus its detections into the ordered list of
/// regions to OCR, without cropping any of them.
///
/// This is the planner proper; [`plan_layout_regions`] is it plus the crops.
/// A caller that OCRs one region at a time should use this and crop inside its
/// own loop, so peak image memory is the page plus one crop rather than the
/// page plus every crop. A page-sized box costs megabytes, and a detections
/// file listing twenty of them is a few kilobytes.
///
/// The detections are consumed in the order the caller supplies, which for the
/// reference detector is reading order. When nothing usable survives, the whole
/// page is returned as a single `plain` region, matching the reference's
/// "no detections, or one `image` detection" branch.
pub fn plan_layout_region_boxes(
    image_width: u32,
    image_height: u32,
    detections: &[Detection],
    containment_threshold: f32,
    min_dim: u32,
    max_dim: u32,
) -> Vec<LayoutOcrPlan> {
    let whole_page = || LayoutOcrPlan {
        category: "plain".to_string(),
        bbox: [0.0, 0.0, image_width as f32, image_height as f32],
        score: 1.0,
        ocr_category: OcrCategory::Plain,
        crop: (0, 0, image_width, image_height),
    };

    let only_figure =
        detections.len() == 1 && layout_to_ocr_category(&detections[0].class_name).is_none();
    if detections.is_empty() || only_figure {
        return vec![whole_page()];
    }

    let kept = filter_nested_detections(detections, containment_threshold);
    let mut plans: Vec<LayoutOcrPlan> = Vec::with_capacity(kept.len());
    for det in kept {
        let Some(ocr_category) = layout_to_ocr_category(&det.class_name) else {
            continue;
        };
        // One region, one OCR run. A DETR-style detection head takes no
        // per-query argmax, so a single query whose sigmoid clears the
        // threshold under several classes emits one detection per class, each
        // carrying that query's box verbatim (see
        // `rt_detr_v2::predictor::decode_detections`). Those entries are
        // alternative labels for one region, not several regions, so OCRing
        // each of them would repeat the same text once per label.
        //
        // The check runs after the category mapping, not before, so a box
        // labelled both `picture` (no OCR category) and `text` still gets
        // OCRed as text instead of being dropped on the un-OCRable label. The
        // surviving entry is the first that both maps to a category and crops,
        // which for `mlxcel detect` output is the highest-confidence such label
        // for that box.
        if plans.iter().any(|plan| plan.bbox == det.bbox) {
            continue;
        }
        let Some(crop) = crop_rect(image_width, image_height, &det.bbox, min_dim, max_dim) else {
            continue;
        };
        plans.push(LayoutOcrPlan {
            category: det.class_name.clone(),
            bbox: det.bbox,
            score: det.score,
            ocr_category,
            crop,
        });
    }

    if plans.is_empty() {
        return vec![whole_page()];
    }
    plans
}

/// Turn a page plus its detections into the ordered list of cropped regions.
///
/// Delegates to [`plan_layout_region_boxes`] so the two planners cannot drift
/// apart. Every crop is materialized up front, which is only appropriate when
/// the caller genuinely needs all of them at once.
pub fn plan_layout_regions(
    image: &image::DynamicImage,
    detections: &[Detection],
    containment_threshold: f32,
    min_dim: u32,
    max_dim: u32,
) -> Vec<LayoutOcrRegion> {
    plan_layout_region_boxes(
        image.width(),
        image.height(),
        detections,
        containment_threshold,
        min_dim,
        max_dim,
    )
    .into_iter()
    .map(|plan| {
        let (x, y, w, h) = plan.crop;
        LayoutOcrRegion {
            category: plan.category,
            bbox: plan.bbox,
            score: plan.score,
            ocr_category: plan.ocr_category,
            image: image.crop_imm(x, y, w, h),
        }
    })
    .collect()
}

#[cfg(test)]
#[path = "falcon_ocr_layout_tests.rs"]
mod tests;
