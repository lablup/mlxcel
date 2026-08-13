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

//! Unit tests for RT-DETRv2 primitives that don't require a checkpoint.
//!
//! These exercise the host-side helpers (anchor generation, the composed
//! pooling / upsample / grid_sample ops) against hand-computed references so a
//! regression in the math surfaces without a multi-hundred-MB model download.

use mlxcel_core::MlxArray;

use super::layers::{grid_sample, max_pool_3x3_s2_p1, upsample_nearest_2x};
use super::transformer::{generate_anchors, inverse_sigmoid};

/// Read a small MLX array to a row-major `Vec<f32>`.
fn read_f32(arr: &MlxArray) -> Vec<f32> {
    let c = mlxcel_core::contiguous(arr, false);
    let c = c.as_ref().unwrap();
    mlxcel_core::eval(c);
    mlxcel_core::array_to_raw_bytes(c)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn upsample_nearest_2x_doubles_hw() {
    // (1, 2, 2, 1) with values [[1,2],[3,4]] -> (1, 4, 4, 1) nearest.
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let x = mlxcel_core::from_slice_f32(&data, &[1, 2, 2, 1]);
    let up = upsample_nearest_2x(&x);
    let shape = mlxcel_core::array_shape(&up);
    assert_eq!(shape, vec![1, 4, 4, 1]);
    let v = read_f32(&up);
    // Row 0: 1 1 2 2 ; Row 1: 1 1 2 2 ; Row 2: 3 3 4 4 ; Row 3: 3 3 4 4.
    let expected = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    for (a, b) in v.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "got {a}, want {b}");
    }
}

#[test]
fn max_pool_3x3_s2_p1_on_4x4() {
    // 4x4 ramp 0..16, single channel. With kernel 3, stride 2, pad 1 ->
    // out = floor((4 + 2 - 3) / 2) + 1 = 2x2.
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let x = mlxcel_core::from_slice_f32(&data, &[1, 4, 4, 1]);
    let pooled = max_pool_3x3_s2_p1(&x);
    let shape = mlxcel_core::array_shape(&pooled);
    assert_eq!(shape, vec![1, 2, 2, 1]);
    // Padded grid (pad=1) has the 4x4 ramp centered; window centers at output
    // (0,0) cover rows -1..1, cols -1..1 -> max of {0,1,4,5} = 5.
    // (0,1) cover cols 1..3 -> max of {1,2,3,5,6,7} = 7.
    // (1,0) -> max of {4,5,8,9,12,13} = 13.
    // (1,1) -> max of {5,6,7,9,10,11,13,14,15} = 15.
    let v = read_f32(&pooled);
    assert_eq!(v, vec![5.0, 7.0, 13.0, 15.0]);
}

#[test]
fn grid_sample_center_is_bilinear_mean() {
    // 2x2 single-channel image [[0,1],[2,3]]. Sample at normalized (0,0) which
    // (align_corners=False) maps to pixel (0.5, 0.5) -> bilinear average of all
    // four corners = (0+1+2+3)/4 = 1.5.
    let img = [0.0f32, 1.0, 2.0, 3.0];
    let x = mlxcel_core::from_slice_f32(&img, &[1, 2, 2, 1]);
    // grid (1, 1, 1, 2) = (gx=0, gy=0).
    let grid = mlxcel_core::from_slice_f32(&[0.0, 0.0], &[1, 1, 1, 2]);
    let out = grid_sample(&x, &grid);
    let shape = mlxcel_core::array_shape(&out);
    assert_eq!(shape, vec![1, 1, 1, 1]);
    let v = read_f32(&out);
    assert!((v[0] - 1.5).abs() < 1e-5, "got {}", v[0]);
}

#[test]
fn grid_sample_out_of_bounds_is_zero_padded() {
    let img = [5.0f32, 6.0, 7.0, 8.0];
    let x = mlxcel_core::from_slice_f32(&img, &[1, 2, 2, 1]);
    // Far outside [-1,1] -> all four corners out of bounds -> zero.
    let grid = mlxcel_core::from_slice_f32(&[5.0, 5.0], &[1, 1, 1, 2]);
    let out = grid_sample(&x, &grid);
    let v = read_f32(&out);
    assert!(v[0].abs() < 1e-6, "got {}", v[0]);
}

#[test]
fn generate_anchors_shapes_and_logit_consistency() {
    // Two tiny levels: 2x2 and 1x1.
    let shapes = [(2, 2), (1, 1)];
    let (anchors, mask) = generate_anchors(&shapes);
    let a_shape = mlxcel_core::array_shape(&anchors);
    let m_shape = mlxcel_core::array_shape(&mask);
    // total = 4 + 1 = 5 positions.
    assert_eq!(a_shape, vec![1, 5, 4]);
    assert_eq!(m_shape, vec![1, 5, 1]);

    let a = read_f32(&anchors);
    let m = read_f32(&mask);
    // For each valid position, sigmoid(logit) should reconstruct the anchor
    // center within [eps, 1-eps]; for masked positions the logit is f32::MAX.
    for (i, &valid) in m.iter().enumerate() {
        if valid > 0.5 {
            // first component (cx) sigmoid in (0,1).
            let s = 1.0 / (1.0 + (-a[i * 4]).exp());
            assert!(s > 0.0 && s < 1.0);
        } else {
            assert_eq!(a[i * 4], f32::MAX);
        }
    }
}

#[test]
fn inverse_sigmoid_is_logit() {
    // inverse_sigmoid(0.5) == log(0.5/0.5) == 0.
    let x = mlxcel_core::from_slice_f32(&[0.5], &[1]);
    let out = inverse_sigmoid(&x, 1e-5);
    let v = read_f32(&out);
    assert!(v[0].abs() < 1e-4, "got {}", v[0]);

    // inverse_sigmoid then sigmoid round-trips a mid-range value.
    let x = mlxcel_core::from_slice_f32(&[0.73], &[1]);
    let inv = inverse_sigmoid(&x, 1e-5);
    let back = mlxcel_core::sigmoid(&inv);
    let v = read_f32(&back);
    assert!((v[0] - 0.73).abs() < 1e-4, "got {}", v[0]);
}

// ---------------------------------------------------------------------------
// Output readback and box denormalization (issue #1089).
//
// The detector used to return boxes that matched no page content. The seam was
// the host readback in `predictor`: the forward graph inherits the checkpoint
// dtype (bf16 for the shipped checkpoints), but the readback parsed the raw
// buffer as 4-byte f32. bf16 is the high half of an f32, so a 4-byte parse over
// a bf16 buffer fuses each adjacent pair into one bogus f32 and returns half
// the elements, which silently misaligns every query/label/box association.
//
// These pin both halves of that seam without needing a checkpoint: the dtype
// round-trip, and the pixel mapping for tall / wide / square images.
// ---------------------------------------------------------------------------

/// A bf16 array must read back element-for-element, not pair-fused.
///
/// This is the exact failure that produced the wrong boxes: a length mismatch
/// here means the decode indexes `query * num_labels` into a buffer half the
/// size it expects.
#[test]
fn bf16_output_reads_back_elementwise() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.015_625 - 0.5).collect();
    let f32_arr = mlxcel_core::from_slice_f32(&values, &[1, 16, 4]);
    let bf16_arr = mlxcel_core::astype(&f32_arr, mlxcel_core::dtype::BFLOAT16);

    let got = super::predictor::read_output_f32(&bf16_arr);

    assert_eq!(
        got.len(),
        values.len(),
        "bf16 readback returned {} of {} elements; a fixed-width parse fuses \
         adjacent bf16 values into one f32",
        got.len(),
        values.len()
    );
    // The chosen values are exact in bf16 (multiples of 1/64 in [-0.5, 0.5)),
    // so the round-trip is lossless and can be compared exactly.
    for (i, (g, w)) in got.iter().zip(values.iter()).enumerate() {
        assert!((g - w).abs() < 1e-6, "element {i}: got {g}, want {w}");
    }
}

/// The same readback must stay correct for an f32 array, so the fix is a
/// dtype-normalizing conversion rather than a bf16 special case.
#[test]
fn f32_output_reads_back_elementwise() {
    let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let arr = mlxcel_core::from_slice_f32(&values, &[1, 8, 4]);
    let got = super::predictor::read_output_f32(&arr);
    assert_eq!(got.len(), values.len());
    for (g, w) in got.iter().zip(values.iter()) {
        assert!((g - w).abs() < 1e-6);
    }
}

/// The box mapping is an independent per-axis scale, so a box at the same
/// normalized position must land at the proportional pixel position on a tall,
/// a wide, and a square page alike.
///
/// The RT-DETRv2 preprocessor resizes straight to a square with `do_pad:
/// false`, so there is no letterbox offset to undo and no shared scale factor.
#[test]
fn denormalize_box_scales_each_axis_independently() {
    use super::predictor::denormalize_box;

    // A box centered at (0.25, 0.94) sized (0.4, 0.04): the footer band of a
    // page, left-of-center. Deliberately near the bottom edge, where a bad
    // inverse transform shows up worst.
    let norm = [0.25, 0.94, 0.4, 0.04];

    // Tall portrait page.
    let b = denormalize_box(norm, 1000.0, 1300.0);
    assert!((b[0] - 50.0).abs() < 1e-3, "left {}", b[0]);
    assert!((b[1] - 1196.0).abs() < 1e-3, "top {}", b[1]);
    assert!((b[2] - 450.0).abs() < 1e-3, "right {}", b[2]);
    assert!((b[3] - 1248.0).abs() < 1e-3, "bottom {}", b[3]);

    // Wide landscape page: same normalized box, axes swapped.
    let b = denormalize_box(norm, 1300.0, 1000.0);
    assert!((b[0] - 65.0).abs() < 1e-3, "left {}", b[0]);
    assert!((b[1] - 920.0).abs() < 1e-3, "top {}", b[1]);
    assert!((b[2] - 585.0).abs() < 1e-3, "right {}", b[2]);
    assert!((b[3] - 960.0).abs() < 1e-3, "bottom {}", b[3]);

    // Square page.
    let b = denormalize_box(norm, 1000.0, 1000.0);
    assert!((b[0] - 50.0).abs() < 1e-3, "left {}", b[0]);
    assert!((b[1] - 920.0).abs() < 1e-3, "top {}", b[1]);
    assert!((b[2] - 450.0).abs() < 1e-3, "right {}", b[2]);
    assert!((b[3] - 960.0).abs() < 1e-3, "bottom {}", b[3]);
}

/// A box running past an edge clamps to the page, and the clamp uses the
/// per-axis extent rather than a single dimension.
#[test]
fn denormalize_box_clamps_per_axis() {
    use super::predictor::denormalize_box;

    // Full-page box, oversized: clamps to exactly the page rectangle.
    let b = denormalize_box([0.5, 0.5, 2.0, 2.0], 1000.0, 1300.0);
    assert_eq!(b, [0.0, 0.0, 1000.0, 1300.0]);

    // A box hanging off the top-left corner clamps only the two low edges.
    let b = denormalize_box([0.05, 0.02, 0.4, 0.1], 1000.0, 1300.0);
    assert_eq!(b[0], 0.0);
    assert_eq!(b[1], 0.0);
    assert!((b[2] - 250.0).abs() < 1e-3, "right {}", b[2]);
    assert!((b[3] - 91.0).abs() < 1e-3, "bottom {}", b[3]);
}

/// A synthetic page with three known regions must decode to three boxes that
/// land on those regions, on a tall page.
///
/// This is the end-to-end shape of the reported failure: before the fix, every
/// detection collapsed onto one origin-anchored box that reached neither the
/// footer nor any drawn element. The logits/boxes here stand in for the model
/// output so the geometry is pinned without a checkpoint.
#[test]
fn decode_places_known_regions_on_a_tall_page() {
    use super::predictor::decode_detections;

    let (img_w, img_h) = (1000.0f32, 1300.0f32);
    let num_labels = 3;
    let labels: Vec<String> = vec!["title".into(), "text".into(), "page_footer".into()];

    // Three queries, each one region of a portrait page, in (cx, cy, w, h).
    //   q0 title  : x 80..680,   y 60..110   -> c(0.38, 0.0654) s(0.60, 0.0385)
    //   q1 body   : x 80..900,   y 300..330  -> c(0.49, 0.2423) s(0.82, 0.0231)
    //   q2 footer : x 80..300,   y 1220..1250 -> c(0.19, 0.95)  s(0.22, 0.0231)
    let boxes: Vec<f32> = vec![
        0.38,
        0.065_384_6,
        0.60,
        0.038_461_5, //
        0.49,
        0.242_307_7,
        0.82,
        0.023_076_9, //
        0.19,
        0.95,
        0.22,
        0.023_076_9,
    ];
    // Logit 2.0 -> sigmoid 0.881 (kept); -2.0 -> 0.119 (dropped by threshold).
    let (hi, lo) = (2.0f32, -2.0f32);
    let logits: Vec<f32> = vec![
        hi, lo, lo, // q0 -> title
        lo, hi, lo, // q1 -> text
        lo, lo, hi, // q2 -> page_footer
    ];

    let dets = decode_detections(
        &logits,
        &boxes,
        3,
        num_labels,
        img_w,
        img_h,
        0.3,
        Some(&labels),
    );

    assert_eq!(dets.len(), 3, "expected one detection per region: {dets:?}");

    let find = |name: &str| {
        dets.iter()
            .find(|d| d.class_name == name)
            .unwrap_or_else(|| panic!("no {name} detection in {dets:?}"))
    };

    let title = find("title");
    assert!((title.bbox[0] - 80.0).abs() < 1.0, "{:?}", title.bbox);
    assert!((title.bbox[1] - 60.0).abs() < 1.0, "{:?}", title.bbox);
    assert!((title.bbox[2] - 680.0).abs() < 1.0, "{:?}", title.bbox);
    assert!((title.bbox[3] - 110.0).abs() < 1.0, "{:?}", title.bbox);

    let body = find("text");
    assert!((body.bbox[0] - 80.0).abs() < 1.0, "{:?}", body.bbox);
    assert!((body.bbox[1] - 300.0).abs() < 1.0, "{:?}", body.bbox);
    assert!((body.bbox[2] - 900.0).abs() < 1.0, "{:?}", body.bbox);
    assert!((body.bbox[3] - 330.0).abs() < 1.0, "{:?}", body.bbox);

    // The regression that started this issue: the footer detection must reach
    // the bottom of a tall page, not stop short around the midpoint.
    let footer = find("page_footer");
    assert!((footer.bbox[0] - 80.0).abs() < 1.0, "{:?}", footer.bbox);
    assert!((footer.bbox[1] - 1220.0).abs() < 1.0, "{:?}", footer.bbox);
    assert!((footer.bbox[2] - 300.0).abs() < 1.0, "{:?}", footer.bbox);
    assert!((footer.bbox[3] - 1250.0).abs() < 1.0, "{:?}", footer.bbox);
    assert!(
        footer.bbox[3] > img_h * 0.9,
        "footer must reach the bottom of a {img_h}-tall page, got {:?}",
        footer.bbox
    );

    // Every region gets its own box: no collapse onto a single rectangle.
    assert_ne!(title.bbox, body.bbox);
    assert_ne!(body.bbox, footer.bbox);
}

/// One query clearing the threshold under several classes emits one detection
/// per class, all sharing that query's box.
///
/// This is `RTDetrImageProcessor.post_process_object_detection` semantics for
/// `use_focal_loss=True`: the top-K runs over the flattened
/// `(queries x labels)` grid with no per-query argmax. Repeated boxes are the
/// head's intended output, so this test states the contract rather than
/// treating the repeat as a defect. What was broken was the geometry, not the
/// repetition.
///
/// Two queries are needed to observe this: K is `num_queries`, so a
/// single-query input can only ever yield one entry no matter how many classes
/// clear the threshold.
#[test]
fn one_query_may_emit_several_labels_sharing_its_box() {
    use super::predictor::decode_detections;

    let labels: Vec<String> = vec!["title".into(), "section_header".into(), "text".into()];
    // q0 is a real region; q1 is an empty background query.
    let boxes: Vec<f32> = vec![
        0.5, 0.1, 0.6, 0.04, //
        0.5, 0.5, 0.1, 0.1,
    ];
    // Two of q0's classes clear 0.3; its third and all of q1 stay well below,
    // so the top-2 are both q0 entries.
    let logits: Vec<f32> = vec![
        1.5, 0.8, -3.0, //
        -4.0, -4.0, -4.0,
    ];

    let dets = decode_detections(&logits, &boxes, 2, 3, 1000.0, 1300.0, 0.3, Some(&labels));

    assert_eq!(dets.len(), 2, "{dets:?}");
    assert_eq!(dets[0].bbox, dets[1].bbox, "same query -> same box");
    assert!(dets[0].score > dets[1].score, "sorted by score descending");
    let names: Vec<&str> = dets.iter().map(|d| d.class_name.as_str()).collect();
    assert_eq!(names, vec!["title", "section_header"]);
    // Both entries describe q0's region, not q1's.
    assert!((dets[0].bbox[1] - 104.0).abs() < 1.0, "{:?}", dets[0].bbox);
}
