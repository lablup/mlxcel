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

//! Pure RF-DETR output decoding and per-class greedy NMS.

use std::collections::BTreeMap;

const MAX_QUERIES: usize = 1024;
const MAX_LOGIT_CLASSES: usize = 1025;
const MAX_SELECT: usize = 1024;

/// COCO's sparse 91-category table. Index zero and omitted category ids are
/// intentionally represented as `N/A`, matching the pinned processor.
pub const COCO_CLASSES: [&str; 91] = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// One decoded RF-DETR row. `query_index` selects the matching mask-logit row.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedDetection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub label: usize,
    pub class_name: String,
    pub query_index: usize,
}

#[derive(Debug, Clone)]
struct Candidate {
    bbox: [f32; 4],
    score: f32,
    label: usize,
    query_index: usize,
}

/// Decode first-batch RF-DETR logits and normalized `cxcywh` boxes.
#[allow(clippy::too_many_arguments)]
pub fn postprocess(
    pred_logits: &[f32],
    pred_boxes: &[f32],
    num_queries: usize,
    logits_classes: usize,
    original_width: u32,
    original_height: u32,
    score_threshold: f32,
    num_select: usize,
    nms_threshold: f32,
    class_names: Option<&[String]>,
) -> Result<Vec<DecodedDetection>, String> {
    validate_inputs(
        pred_logits,
        pred_boxes,
        num_queries,
        logits_classes,
        original_width,
        original_height,
        score_threshold,
        num_select,
        nms_threshold,
    )?;

    let mut best_by_query = Vec::with_capacity(num_queries);
    for query in 0..num_queries {
        let start = query * logits_classes;
        let logits = &pred_logits[start..start + logits_classes];
        let (label, logit) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .ok_or_else(|| format!("RF-DETR query {query} contains no class logits"))?;
        best_by_query.push((query, label, sigmoid(logit)));
    }
    best_by_query.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    best_by_query.truncate(num_select.min(num_queries));

    let width = original_width as f32;
    let height = original_height as f32;
    let mut candidates = Vec::with_capacity(best_by_query.len());
    for (query_index, label, score) in best_by_query {
        if score <= score_threshold {
            continue;
        }
        let offset = query_index * 4;
        let cx = pred_boxes[offset];
        let cy = pred_boxes[offset + 1];
        let box_width = pred_boxes[offset + 2];
        let box_height = pred_boxes[offset + 3];
        candidates.push(Candidate {
            bbox: [
                ((cx - box_width / 2.0) * width).clamp(0.0, width),
                ((cy - box_height / 2.0) * height).clamp(0.0, height),
                ((cx + box_width / 2.0) * width).clamp(0.0, width),
                ((cy + box_height / 2.0) * height).clamp(0.0, height),
            ],
            score,
            label,
            query_index,
        });
    }

    let keep = if nms_threshold < 1.0 && !candidates.is_empty() {
        nms_per_class(&candidates, nms_threshold)
    } else {
        (0..candidates.len()).collect()
    };
    Ok(keep
        .into_iter()
        .map(|index| {
            let candidate = &candidates[index];
            DecodedDetection {
                bbox: candidate.bbox,
                score: candidate.score,
                label: candidate.label,
                class_name: class_name(candidate.label, class_names),
                query_index: candidate.query_index,
            }
        })
        .collect())
}

fn validate_inputs(
    pred_logits: &[f32],
    pred_boxes: &[f32],
    num_queries: usize,
    logits_classes: usize,
    original_width: u32,
    original_height: u32,
    score_threshold: f32,
    num_select: usize,
    nms_threshold: f32,
) -> Result<(), String> {
    if num_queries == 0 || num_queries > MAX_QUERIES {
        return Err(format!(
            "RF-DETR query count must be in 1..={MAX_QUERIES}; got {num_queries}"
        ));
    }
    if logits_classes == 0 || logits_classes > MAX_LOGIT_CLASSES {
        return Err(format!(
            "RF-DETR logit class count must be in 1..={MAX_LOGIT_CLASSES}; got {logits_classes}"
        ));
    }
    if num_select == 0 || num_select > MAX_SELECT {
        return Err(format!(
            "RF-DETR num_select must be in 1..={MAX_SELECT}; got {num_select}"
        ));
    }
    if original_width == 0 || original_height == 0 {
        return Err("RF-DETR original image dimensions must be positive".into());
    }
    for (name, value) in [
        ("score_threshold", score_threshold),
        ("nms_threshold", nms_threshold),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!("RF-DETR {name} must be finite and in [0, 1]"));
        }
    }

    let expected_logits = num_queries
        .checked_mul(logits_classes)
        .ok_or("RF-DETR logits shape overflows usize")?;
    let expected_boxes = num_queries
        .checked_mul(4)
        .ok_or("RF-DETR box shape overflows usize")?;
    if pred_logits.len() != expected_logits || pred_boxes.len() != expected_boxes {
        return Err(format!(
            "RF-DETR output shape mismatch: logits={} expected={expected_logits}, boxes={} expected={expected_boxes}",
            pred_logits.len(),
            pred_boxes.len()
        ));
    }
    ensure_finite("pred_logits", pred_logits)?;
    ensure_finite("pred_boxes", pred_boxes)
}

fn ensure_finite(name: &str, values: &[f32]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("RF-DETR {name}[{index}] is not finite: {value}"));
    }
    Ok(())
}

fn class_name(label: usize, class_names: Option<&[String]>) -> String {
    match class_names {
        Some(names) => names
            .get(label)
            .cloned()
            .unwrap_or_else(|| format!("class_{label}")),
        None => COCO_CLASSES
            .get(label)
            .map(|name| (*name).to_string())
            .unwrap_or_else(|| format!("class_{label}")),
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn nms_per_class(candidates: &[Candidate], threshold: f32) -> Vec<usize> {
    let mut by_class: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        by_class.entry(candidate.label).or_default().push(index);
    }

    let mut keep = Vec::new();
    for mut indices in by_class.into_values() {
        indices.sort_by(|&left, &right| {
            candidates[right]
                .score
                .total_cmp(&candidates[left].score)
                .then_with(|| left.cmp(&right))
        });
        while let Some((&selected, remaining)) = indices.split_first() {
            keep.push(selected);
            indices = remaining
                .iter()
                .copied()
                .filter(|&index| {
                    box_iou(candidates[selected].bbox, candidates[index].bbox) < threshold
                })
                .collect();
        }
    }
    keep.sort_by(|&left, &right| {
        candidates[right]
            .score
            .total_cmp(&candidates[left].score)
            .then_with(|| left.cmp(&right))
    });
    keep
}

fn box_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let intersection_width = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0);
    let intersection_height = (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    let intersection = intersection_width * intersection_height;
    let left_area = (left[2] - left[0]) * (left[3] - left[1]);
    let right_area = (right[2] - right[0]) * (right[3] - right[1]);
    intersection / (left_area + right_area - intersection + 1.0e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_class_greedy_nms_preserves_overlapping_different_classes() {
        let detections = postprocess(
            &[
                4.0, -4.0, // query 0, class 0
                3.0, -4.0, // query 1, class 0, suppressed
                -4.0, 5.0, // query 2, class 1, retained despite overlap
            ],
            &[
                0.5, 0.5, 0.4, 0.4, // query 0
                0.5, 0.5, 0.4, 0.4, // query 1
                0.5, 0.5, 0.4, 0.4, // query 2
            ],
            3,
            2,
            200,
            100,
            0.5,
            3,
            0.5,
            Some(&["zero".into(), "one".into()]),
        )
        .unwrap();
        assert_eq!(
            detections
                .iter()
                .map(|detection| detection.query_index)
                .collect::<Vec<_>>(),
            [2, 0]
        );
        assert_eq!(detections[0].class_name, "one");
        for (actual, expected) in detections[1].bbox.iter().zip([60.0, 30.0, 140.0, 70.0]) {
            assert!((*actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn threshold_is_strict_and_query_index_survives_top_k() {
        let none = postprocess(
            &[0.0],
            &[0.5, 0.5, 0.2, 0.2],
            1,
            1,
            10,
            10,
            0.5,
            1,
            0.5,
            None,
        )
        .unwrap();
        assert!(none.is_empty());

        let selected = postprocess(
            &[1.0, 3.0],
            &[0.25, 0.5, 0.5, 1.0, 0.75, 0.5, 0.5, 1.0],
            2,
            1,
            200,
            100,
            0.0,
            1,
            1.0,
            None,
        )
        .unwrap();
        assert_eq!(selected[0].query_index, 1);
        assert_eq!(selected[0].bbox, [100.0, 0.0, 200.0, 100.0]);
        assert_eq!(selected[0].class_name, "N/A");
    }

    #[test]
    fn malformed_nonfinite_and_oversized_outputs_fail_before_decode() {
        assert!(
            postprocess(&[0.0], &[0.0; 4], 2, 1, 1, 1, 0.5, 1, 0.5, None)
                .unwrap_err()
                .contains("shape mismatch")
        );
        assert!(
            postprocess(&[f32::NAN], &[0.0; 4], 1, 1, 1, 1, 0.5, 1, 0.5, None)
                .unwrap_err()
                .contains("not finite")
        );
        assert!(
            postprocess(&[], &[], MAX_QUERIES + 1, 1, 1, 1, 0.5, 1, 0.5, None)
                .unwrap_err()
                .contains("query count")
        );
    }

    #[test]
    fn unknown_class_uses_upstream_fallback_name() {
        let detections = postprocess(
            &[0.0, 0.0, 0.0, 0.0, 2.0],
            &[0.5, 0.5, 0.2, 0.2],
            1,
            5,
            10,
            10,
            0.5,
            1,
            0.5,
            Some(&["only-zero".into()]),
        )
        .unwrap();
        assert_eq!(detections[0].label, 4);
        assert_eq!(detections[0].class_name, "class_4");
    }
}
