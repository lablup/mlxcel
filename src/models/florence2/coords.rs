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

//! Florence-2 location tokens and the bin-to-pixel dequantizers.
//!
//! Florence-2 encodes every spatial answer as `<loc_N>` tokens. The vocabulary
//! holds exactly 1000 of them, `<loc_0>` .. `<loc_999>`, occupying a
//! contiguous id range, and each one names a bin index on a 1000-bin axis
//! normalized to the *original* image extent. Turning a bin back into a pixel
//! coordinate therefore needs the pre-resize image size, which is why the
//! processor keeps it alongside the pixel tensor.
//!
//! Two dequantizers exist upstream and they are not interchangeable:
//! [`dequantize_box`] handles an `(xmin, ymin, xmax, ymax)` 4-tuple and
//! [`dequantize_coordinates`] handles an `(x, y)` point run. They happen to
//! carry the same 1000/1000 bin counts and `floor` mode in every shipped
//! config, so their arithmetic coincides, but the reachable tasks route to
//! them separately and a future config could split the bin counts.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py
//! (`BoxQuantizer`, `CoordinatesQuantizer`).

/// Number of bins on each axis. `NUM_BBOX_{WIDTH,HEIGHT}_BINS` and
/// `COORDINATES_{WIDTH,HEIGHT}_BINS` are all 1000 in the shipped default
/// config, and the tokenizer only carries 1000 `<loc_*>` tokens, so a
/// different value could not be expressed anyway.
pub(crate) const LOC_BINS: i32 = 1000;

/// Vocabulary id of `<loc_0>` in the Florence-2 tokenizer. The 1000 location
/// tokens are contiguous from here, so `<loc_N>` is `FLORENCE2_LOC_TOKEN_BASE
/// + N`, pinned against the checkpoint by
/// `florence2_loc_tokens_are_contiguous`.
///
/// Post-processing works on decoded *text* rather than ids and does not need
/// this; it is here for callers that want to inspect or constrain the id
/// range directly.
pub const FLORENCE2_LOC_TOKEN_BASE: u32 = 50269;

/// Vocabulary id of `<loc_n>`, or `None` when `n` is outside `0..1000`.
pub fn florence2_loc_token_id(n: u32) -> Option<u32> {
    (n < LOC_BINS as u32).then(|| FLORENCE2_LOC_TOKEN_BASE + n)
}

/// Size of the image a Florence-2 answer's coordinates are relative to, in
/// pixels.
///
/// Named fields on purpose. Upstream's `post_process_generation` docstring
/// says its `image_size` argument is "height x width" but every call site
/// unpacks it as `image_width, image_height = image_size`, so the docstring is
/// wrong and a bare tuple is a coin flip at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Florence2ImageSize {
    pub width: u32,
    pub height: u32,
}

impl Florence2ImageSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Pixels per bin on each axis, `(width, height)`.
    ///
    /// The division is done in f64 and the result narrowed to f32 because
    /// upstream computes `size_w / bins_w` in Python (f64) and then hands the
    /// result to torch as a scalar against an f32 tensor. Doing the whole
    /// chain in f32 would round the quotient twice.
    fn size_per_bin(self) -> (f32, f32) {
        (
            (f64::from(self.width) / f64::from(LOC_BINS)) as f32,
            (f64::from(self.height) / f64::from(LOC_BINS)) as f32,
        )
    }
}

/// An axis-aligned box in original-image pixels, left-top-right-bottom.
///
/// Deliberately not `crate::vision::detection::rt_detr_v2::Detection`: that
/// type carries a confidence score and a `label: usize` class index into a
/// fixed COCO-style label set, and Florence-2 produces neither. Its labels are
/// open-vocabulary strings decoded from the answer, and the model emits no
/// per-box score at all, so reusing `Detection` would mean two fabricated
/// fields on every box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Florence2BoundingBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
}

impl Florence2BoundingBox {
    /// Flat `[xmin, ymin, xmax, ymax]`, the layout upstream returns.
    pub fn to_array(self) -> [f32; 4] {
        [self.xmin, self.ymin, self.xmax, self.ymax]
    }

    /// Bin indices for this box, the inverse of [`dequantize_box`].
    ///
    /// `floor(pixel / size_per_bin)` clamped into `0..1000`, per upstream's
    /// `BoxQuantizer.quantize` in `floor` mode. Not needed to read an answer;
    /// it exists to *write* one, see [`Self::to_region_prompt`].
    pub fn quantize(self, size: Florence2ImageSize) -> [i32; 4] {
        let (spb_w, spb_h) = size.size_per_bin();
        let bin = |value: f32, per_bin: f32| -> i32 {
            if per_bin <= 0.0 {
                return 0;
            }
            (value / per_bin).floor().clamp(0.0, (LOC_BINS - 1) as f32) as i32
        };
        [
            bin(self.xmin, spb_w),
            bin(self.ymin, spb_h),
            bin(self.xmax, spb_w),
            bin(self.ymax, spb_h),
        ]
    }

    /// This box as the `<loc_a><loc_b><loc_c><loc_d>` string the
    /// region-input tasks expect, for example
    /// `"<loc_52><loc_332><loc_932><loc_774>"`.
    ///
    /// `<REGION_TO_CATEGORY>`, `<REGION_TO_DESCRIPTION>`, `<REGION_TO_OCR>`
    /// and `<REGION_TO_SEGMENTATION>` all interpolate a region in this form
    /// into their prompt, and the location tokens are real vocabulary entries,
    /// so the string tokenizes to four ids rather than being spelled out.
    pub fn to_region_prompt(self, size: Florence2ImageSize) -> String {
        self.quantize(size)
            .iter()
            .map(|bin| format!("<loc_{bin}>"))
            .collect()
    }
}

/// An OCR region: four corner points in original-image pixels, flattened
/// `[x0, y0, x1, y1, x2, y2, x3, y3]`.
///
/// Quad rather than a box because `<OCR_WITH_REGION>` predicts a rotated
/// quadrilateral around each text line, which a left-top-right-bottom box
/// cannot represent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Florence2QuadBox {
    pub points: [f32; 8],
}

/// A closed polygon outline in original-image pixels, flattened
/// `[x0, y0, x1, y1, ...]`.
///
/// Flat rather than `Vec<(f32, f32)>` to match the upstream shape exactly:
/// the segmentation tasks emit an odd-length location run often enough that
/// the parser has an explicit rule for it (drop the unpaired trailing bin),
/// and keeping the flat layout makes that rule the only place pairing is
/// decided.
#[derive(Debug, Clone, PartialEq)]
pub struct Florence2Polygon {
    pub points: Vec<f32>,
}

/// Bin 4-tuple `(xmin, ymin, xmax, ymax)` to pixels.
///
/// `(bin + 0.5) * size_per_bin` per axis: the `+ 0.5` takes the bin center
/// rather than its left edge, which is what makes dequantize-of-quantize a
/// half-bin-accurate round trip instead of a systematically biased one.
///
/// Arithmetic is f32 throughout, matching upstream: `torch.tensor(bins)` is an
/// int64 tensor, and adding a Python float to it promotes to torch's default
/// dtype, f32.
pub(crate) fn dequantize_box(bins: [i32; 4], size: Florence2ImageSize) -> Florence2BoundingBox {
    let (spb_w, spb_h) = size.size_per_bin();
    Florence2BoundingBox {
        xmin: (bins[0] as f32 + 0.5) * spb_w,
        ymin: (bins[1] as f32 + 0.5) * spb_h,
        xmax: (bins[2] as f32 + 0.5) * spb_w,
        ymax: (bins[3] as f32 + 0.5) * spb_h,
    }
}

/// Bin run to pixels, read as consecutive `(x, y)` pairs.
///
/// `bins` must have even length; the callers that can produce an odd run drop
/// the unpaired trailing element first, mirroring upstream's
/// `if len(_polygon) % 2 == 1: _polygon = _polygon[:-1]`.
pub(crate) fn dequantize_coordinates(bins: &[i32], size: Florence2ImageSize) -> Vec<f32> {
    let (spb_w, spb_h) = size.size_per_bin();
    bins.chunks_exact(2)
        .flat_map(|pair| {
            [
                (pair[0] as f32 + 0.5) * spb_w,
                (pair[1] as f32 + 0.5) * spb_h,
            ]
        })
        .collect()
}

#[cfg(test)]
#[path = "florence2_coords_tests.rs"]
mod florence2_coords_tests;
