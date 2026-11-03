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

//! IndicOCR block labels, prompts and crop sizing.
//!
//! Port of `processing_indic_ocr.py` (`idp_contract.py` / `idp_crops.py` in
//! the upstream repository). Pure data and geometry so the routing rules stay
//! testable without a model.

use image::DynamicImage;

/// Verbatim from `idp_contract.TEXT_PROMPT`.
pub const TEXT_PROMPT: &str = "Transcribe the text in this image. Write any mathematical expressions in LaTeX, using $...$ for inline math and $$...$$ for display equations.";
/// Verbatim from `idp_contract.EQUATION_PROMPT`.
pub const EQUATION_PROMPT: &str = "Output only the LaTeX for this equation image.";
/// Verbatim from `idp_contract.TABLE_PROMPT_HTML`.
pub const TABLE_PROMPT_HTML: &str = "Convert this table image to HTML. Preserve the structure exactly, using colspan and rowspan for merged cells and <br/> for line breaks within a cell. Output only the HTML table.";
/// Verbatim from `idp_contract.TABLE_PROMPT_MARKDOWN`.
pub const TABLE_PROMPT_MARKDOWN: &str =
    "Convert this table image to a GitHub-flavored markdown table. Output only the table.";

/// Smallest crop area before upscaling (256 x 256).
pub const MIN_CROP_PIXELS: usize = 256 * 256;
/// Largest crop area before downscaling (1536 x 1536).
pub const MAX_CROP_PIXELS: usize = 1536 * 1536;

/// Requested table transcription format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TableFormat {
    #[default]
    Html,
    Markdown,
}

impl TableFormat {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "html" => Ok(Self::Html),
            "markdown" => Ok(Self::Markdown),
            other => Err(format!(
                "--table-format must be 'html' or 'markdown', got {other:?}"
            )),
        }
    }
}

/// Blocks never sent to the recognizer (`idp_contract.DROP_TYPES`).
pub fn is_drop_type(block_type: &str) -> bool {
    matches!(block_type, "Figure" | "Picture")
}

/// Labels never transcribed but kept in place with empty text.
const OCR_SKIP_LABELS: [&str; 6] = [
    "header",
    "footer",
    "diagram",
    "image",
    "chart",
    "advertisement",
];

/// Indicator that a label is transcribed. `header`/`footer`/`chart` are not.
pub fn is_transcribed(label: &str) -> bool {
    let lowered = label.trim().to_lowercase();
    !OCR_SKIP_LABELS.contains(&lowered.as_str())
}

/// IndicDocLayout label to pipeline type (`idp_contract.LABEL_TO_TYPE`).
fn indic_label_to_type(label: &str) -> Option<&'static str> {
    Some(match label {
        "table" => "Table",
        "table-caption" => "Caption",
        "equation" | "expression" => "Equation",
        "diagram" | "chart" => "Figure",
        "image" => "Picture",
        "image-caption" => "Caption",
        "title" | "chapter-title" => "Title",
        "section-title" | "sub-section-title" | "sub-sub-section-title" => "SectionHeader",
        "header" => "PageHeader",
        "footer" => "PageFooter",
        "page-number" | "folio" => "PageNumber",
        "footnote" => "Footnote",
        _ => return None,
    })
}

/// Stock PP-DocLayoutV3 label to pipeline type.
fn pp_doclayout_label_to_type(label: &str) -> Option<&'static str> {
    Some(match label {
        "abstract" | "algorithm" | "aside_text" | "content" | "formula_number" | "reference"
        | "reference_content" | "seal" | "text" => "Text",
        "chart" => "Figure",
        "formula" => "Equation",
        "doc_title" => "Title",
        "figure_title" => "Caption",
        "footer" => "PageFooter",
        "footnote" | "vision_footnote" => "Footnote",
        "header" => "PageHeader",
        "image" => "Picture",
        "number" => "PageNumber",
        "paragraph_title" => "SectionHeader",
        "table" => "Table",
        _ => return None,
    })
}

/// Route a detector label through both taxonomies, defaulting to `Text`.
pub fn map_label(label: &str) -> &'static str {
    let lowered = label.trim().to_lowercase();
    indic_label_to_type(&lowered)
        .or_else(|| pp_doclayout_label_to_type(&lowered))
        .unwrap_or("Text")
}

/// The instruction one block type is transcribed with.
pub fn prompt_for(block_type: &str, table_format: TableFormat) -> &'static str {
    if block_type == "Table" {
        return match table_format {
            TableFormat::Html => TABLE_PROMPT_HTML,
            TableFormat::Markdown => TABLE_PROMPT_MARKDOWN,
        };
    }
    if block_type == "Equation" {
        return EQUATION_PROMPT;
    }
    TEXT_PROMPT
}

/// Scale a crop so its area lands in `[min_px, max_px]`, preserving aspect.
pub fn area_clamp(image: DynamicImage, min_px: usize, max_px: usize) -> DynamicImage {
    let (width, height) = (image.width(), image.height());
    let pixels = width as f64 * height as f64;
    if pixels <= 0.0 {
        return image;
    }
    let scale = if pixels < min_px as f64 {
        (min_px as f64 / pixels).sqrt()
    } else if pixels > max_px as f64 {
        (max_px as f64 / pixels).sqrt()
    } else {
        return image;
    };
    let target_width = python_round(width as f64 * scale).max(1.0) as u32;
    let target_height = python_round(height as f64 * scale).max(1.0) as u32;
    image::DynamicImage::ImageRgb8(image::imageops::resize(
        &image.to_rgb8(),
        target_width,
        target_height,
        image::imageops::FilterType::Lanczos3,
    ))
}

/// Crop one layout box out of the page, or `None` when it collapses.
pub fn crop_for_block(bbox: &[f32; 4], page: &DynamicImage, pad_px: i32) -> Option<DynamicImage> {
    let (width, height) = (page.width() as i64, page.height() as i64);
    let mut x0 = python_round(bbox[0] as f64) as i64;
    let mut y0 = python_round(bbox[1] as f64) as i64;
    let mut x1 = python_round(bbox[2] as f64) as i64;
    let mut y1 = python_round(bbox[3] as f64) as i64;
    if pad_px != 0 {
        x0 -= pad_px as i64;
        y0 -= pad_px as i64;
        x1 += pad_px as i64;
        y1 += pad_px as i64;
    }
    x0 = x0.max(0);
    y0 = y0.max(0);
    x1 = x1.min(width);
    y1 = y1.min(height);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(page.crop_imm(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32))
}

/// Python's `round`: nearest integer, ties to even.
fn python_round(value: f64) -> f64 {
    value.round_ties_even()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            width,
            height,
            image::Rgb([1, 2, 3]),
        ))
    }

    #[test]
    fn label_routing_covers_both_taxonomies() {
        // IndicDocLayout spellings.
        assert_eq!(map_label("Paragraph"), "Text");
        assert_eq!(map_label("Table"), "Table");
        assert_eq!(map_label("Expression"), "Equation");
        assert_eq!(map_label("Diagram"), "Figure");
        assert_eq!(map_label("Image-caption"), "Caption");
        assert_eq!(map_label("Section-title"), "SectionHeader");
        assert_eq!(map_label("Sub-sub-section-title"), "SectionHeader");
        assert_eq!(map_label("Page-number"), "PageNumber");
        // Stock PP-DocLayoutV3 spellings.
        assert_eq!(map_label("doc_title"), "Title");
        assert_eq!(map_label("paragraph_title"), "SectionHeader");
        assert_eq!(map_label("figure_title"), "Caption");
        assert_eq!(map_label("formula"), "Equation");
        assert_eq!(map_label("vision_footnote"), "Footnote");
        // Unknown labels fall back to Text.
        assert_eq!(map_label("something_new"), "Text");
    }

    #[test]
    fn prompt_selection_is_verbatim() {
        assert_eq!(prompt_for("Text", TableFormat::Html), TEXT_PROMPT);
        assert_eq!(prompt_for("Title", TableFormat::Html), TEXT_PROMPT);
        assert_eq!(prompt_for("Equation", TableFormat::Html), EQUATION_PROMPT);
        assert_eq!(prompt_for("Table", TableFormat::Html), TABLE_PROMPT_HTML);
        assert_eq!(
            prompt_for("Table", TableFormat::Markdown),
            TABLE_PROMPT_MARKDOWN
        );
        assert!(TableFormat::parse("markdown").unwrap() == TableFormat::Markdown);
        assert!(TableFormat::parse("csv").unwrap_err().contains("html"));
    }

    #[test]
    fn transcription_skipping_matches_upstream() {
        assert!(!is_transcribed("Header"));
        assert!(!is_transcribed("footer"));
        assert!(!is_transcribed("Chart"));
        assert!(!is_transcribed("Diagram"));
        assert!(!is_transcribed("Advertisement"));
        assert!(is_transcribed("Paragraph"));
        assert!(is_transcribed("Table"));
        assert!(is_transcribed("Equation"));
        assert!(is_drop_type("Figure"));
        assert!(is_drop_type("Picture"));
        assert!(!is_drop_type("Caption"));
    }

    #[test]
    fn area_clamp_raises_and_lowers_into_the_band() {
        let small = area_clamp(solid(10, 10), MIN_CROP_PIXELS, MAX_CROP_PIXELS);
        assert!(small.width() * small.height() >= MIN_CROP_PIXELS as u32);
        assert_eq!(small.width(), small.height());

        let large = area_clamp(solid(4000, 2000), MIN_CROP_PIXELS, MAX_CROP_PIXELS);
        assert!((large.width() * large.height()) as usize <= MAX_CROP_PIXELS);

        let unchanged = area_clamp(solid(1000, 1000), MIN_CROP_PIXELS, MAX_CROP_PIXELS);
        assert_eq!((unchanged.width(), unchanged.height()), (1000, 1000));
    }

    #[test]
    fn crop_clamps_to_the_page_and_drops_collapsed_boxes() {
        let page = solid(100, 80);
        let crop = crop_for_block(&[-10.0, 10.0, 50.0, 200.0], &page, 0).unwrap();
        assert_eq!((crop.width(), crop.height()), (50, 70));
        assert!(crop_for_block(&[20.0, 20.0, 20.0, 40.0], &page, 0).is_none());
        assert!(crop_for_block(&[200.0, 20.0, 300.0, 40.0], &page, 0).is_none());
    }

    #[test]
    fn crop_padding_extends_the_box_before_clamping() {
        let page = solid(100, 100);
        let crop = crop_for_block(&[10.0, 10.0, 20.0, 20.0], &page, 5).unwrap();
        assert_eq!((crop.width(), crop.height()), (20, 20));
    }
}
