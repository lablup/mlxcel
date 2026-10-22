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

//! Host-side SAM 3 thresholding, geometry conversion, and NMS.

use anyhow::{Result, ensure};

pub(crate) fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub(crate) fn confidence_scores(logits: &[f32], presence_logit: f32) -> Vec<f32> {
    let presence = sigmoid(presence_logit);
    logits
        .iter()
        .map(|logit| sigmoid(*logit) * presence)
        .collect()
}

pub(crate) fn clip_xyxy(mut bbox: [f32; 4], width: u32, height: u32) -> [f32; 4] {
    let max_x = width as f32;
    let max_y = height as f32;
    bbox[0] = bbox[0].clamp(0.0, max_x);
    bbox[2] = bbox[2].clamp(0.0, max_x);
    bbox[1] = bbox[1].clamp(0.0, max_y);
    bbox[3] = bbox[3].clamp(0.0, max_y);
    bbox
}

pub(crate) fn normalized_cxcywh(boxes: &[[f32; 4]], width: u32, height: u32) -> Result<Vec<f32>> {
    ensure!(
        width > 0 && height > 0,
        "source dimensions must be positive"
    );
    let width = width as f32;
    let height = height as f32;
    let mut out = Vec::with_capacity(boxes.len() * 4);
    for &bbox in boxes {
        ensure!(
            bbox.iter().all(|value| value.is_finite()),
            "box coordinates must be finite"
        );
        let [x1, y1, x2, y2] = clip_xyxy(bbox, width as u32, height as u32);
        ensure!(x2 >= x1 && y2 >= y1, "box has negative width or height");
        out.extend_from_slice(&[
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]);
    }
    Ok(out)
}

fn intersection_over_union(left: [f32; 4], right: [f32; 4]) -> f32 {
    let x1 = left[0].max(right[0]);
    let y1 = left[1].max(right[1]);
    let x2 = left[2].min(right[2]);
    let y2 = left[3].min(right[3]);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    intersection / (left_area + right_area - intersection).max(1e-6)
}

/// Return score-descending indices after class-agnostic NMS.
pub(crate) fn nms_indices(
    boxes: &[[f32; 4]],
    scores: &[f32],
    iou_threshold: f32,
) -> Result<Vec<usize>> {
    ensure!(
        boxes.len() == scores.len(),
        "NMS boxes/scores length mismatch"
    );
    ensure!(
        iou_threshold.is_finite() && (0.0..=1.0).contains(&iou_threshold),
        "NMS IoU threshold must be finite and between 0 and 1"
    );
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| left.cmp(&right))
    });

    let mut keep = Vec::with_capacity(order.len());
    for index in order {
        ensure!(scores[index].is_finite(), "NMS scores must be finite");
        ensure!(
            boxes[index].iter().all(|value| value.is_finite()),
            "NMS boxes must be finite"
        );
        if keep
            .iter()
            .all(|&kept| intersection_over_union(boxes[index], boxes[kept]) <= iou_threshold)
        {
            keep.push(index);
        }
    }
    Ok(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_matches_upstream_and_suppresses_across_labels() {
        let boxes = [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
            [0.0, 0.0, 10.0, 10.0],
        ];
        let scores = [0.7, 0.9, 0.8, 0.6];
        assert_eq!(nms_indices(&boxes, &scores, 0.5).unwrap(), vec![1, 2]);
    }

    #[test]
    fn pixel_boxes_are_clipped_and_normalized() {
        let boxes = normalized_cxcywh(&[[-10.0, 50.0, 350.0, 500.0]], 640, 480).unwrap();
        assert_eq!(
            boxes,
            vec![175.0 / 640.0, 265.0 / 480.0, 350.0 / 640.0, 430.0 / 480.0]
        );
    }

    #[test]
    fn presence_and_query_logits_are_combined() {
        let scores = confidence_scores(&[0.0, f32::NEG_INFINITY], 0.0);
        assert!((scores[0] - 0.25).abs() < 1e-7);
        assert_eq!(scores[1], 0.0);
    }
}
