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

//! PP-DocLayoutV3 detection and reading-order decoding.

use crate::vision::detection::Detection;

/// Convert a normalized `(cx, cy, width, height)` box to pixel-space
/// `(left, top, right, bottom)` coordinates.
pub fn denormalize_box(box_cxcywh: [f32; 4], image_width: f32, image_height: f32) -> [f32; 4] {
    let half_width = box_cxcywh[2] * 0.5;
    let half_height = box_cxcywh[3] * 0.5;
    [
        (box_cxcywh[0] - half_width) * image_width,
        (box_cxcywh[1] - half_height) * image_height,
        (box_cxcywh[0] + half_width) * image_width,
        (box_cxcywh[1] + half_height) * image_height,
    ]
}

/// Decode the complete pairwise query-order matrix into query indices from
/// first to last.
///
/// This is the CPU equivalent of mlx-vlm's `decode_order`: sigmoid, upper
/// column votes plus complementary lower votes, then a stable ascending sort.
pub fn decode_reading_order(order_logits: &[f32], q: usize) -> Result<Vec<usize>, String> {
    validate_query_count(q)?;
    let expected = q
        .checked_mul(q)
        .ok_or_else(|| format!("query count {q} overflows the order matrix size"))?;
    if order_logits.len() != expected {
        return Err(format!(
            "order_logits has {} values; expected {q}x{q}={expected}",
            order_logits.len()
        ));
    }
    ensure_finite("order_logits", order_logits)?;

    let mut votes = vec![0.0_f32; q];
    for earlier_index in 0..q {
        for later_index in (earlier_index + 1)..q {
            let probability = sigmoid(order_logits[earlier_index * q + later_index]);
            votes[later_index] += probability;
            votes[earlier_index] += 1.0 - probability;
        }
    }
    let mut sequence: Vec<usize> = (0..q).collect();
    // `sort_by` is stable, preserving the query index for an exact vote tie.
    sequence.sort_by(|&left, &right| votes[left].total_cmp(&votes[right]));
    Ok(sequence)
}

#[derive(Debug)]
struct Candidate {
    query: usize,
    detection: Detection,
}

/// Decode flat query-by-class logits, boxes, and the pairwise reading-order
/// head into detections sorted in model-predicted page order.
#[allow(clippy::too_many_arguments)]
pub fn decode_detections(
    logits: &[f32],
    boxes: &[f32],
    order_logits: &[f32],
    q: usize,
    num_labels: usize,
    image_width: f32,
    image_height: f32,
    threshold: f32,
    labels: Option<&[String]>,
) -> Result<Vec<Detection>, String> {
    validate_query_count(q)?;
    if num_labels == 0 || num_labels > 512 {
        return Err(format!("num_labels must be in 1..=512; got {num_labels}"));
    }
    if !image_width.is_finite()
        || !image_height.is_finite()
        || image_width <= 0.0
        || image_height <= 0.0
    {
        return Err(format!(
            "image dimensions must be finite and positive; got {image_width}x{image_height}"
        ));
    }
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(format!(
            "confidence threshold must be finite and in [0, 1]; got {threshold}"
        ));
    }
    if let Some(names) = labels
        && names.len() != num_labels
    {
        return Err(format!(
            "label name count {} does not match num_labels {num_labels}",
            names.len()
        ));
    }

    let flat_count = q
        .checked_mul(num_labels)
        .ok_or_else(|| format!("{q} queries x {num_labels} labels overflows usize"))?;
    let box_count = q
        .checked_mul(4)
        .ok_or_else(|| format!("{q} queries x 4 box coordinates overflows usize"))?;
    if logits.len() != flat_count || boxes.len() != box_count {
        return Err(format!(
            "PP-DocLayoutV3 output shape mismatch: logits has {} values (expected \
             {q}x{num_labels}={flat_count}), boxes has {} values (expected {q}x4={box_count})",
            logits.len(),
            boxes.len()
        ));
    }
    ensure_finite("logits", logits)?;
    ensure_finite("boxes", boxes)?;

    // Decode all queries before confidence filtering. The global-pointer head
    // predicts one page-wide order; decoding only a retained submatrix changes
    // its vote totals and can invert surviving queries.
    let sequence = decode_reading_order(order_logits, q)?;
    let mut rank_by_query = vec![0_usize; q];
    for (rank, query) in sequence.into_iter().enumerate() {
        rank_by_query[query] = rank;
    }

    // Focal-loss postprocessing takes the top Q scores from the flattened
    // Q-by-C grid. Stable sorting retains flat-index order for exact ties and
    // keeps multiple labels from the same query when they are in the top Q.
    let mut scored: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(flat_index, &logit)| (flat_index, sigmoid(logit)))
        .collect();
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));
    scored.truncate(q);

    let mut candidates = Vec::with_capacity(scored.len());
    for (flat_index, score) in scored {
        if score < threshold {
            continue;
        }
        let query = flat_index / num_labels;
        let label = flat_index % num_labels;
        let box_start = query * 4;
        let bbox = denormalize_box(
            [
                boxes[box_start],
                boxes[box_start + 1],
                boxes[box_start + 2],
                boxes[box_start + 3],
            ],
            image_width,
            image_height,
        );
        let class_name = labels
            .and_then(|names| names.get(label))
            .cloned()
            .unwrap_or_else(|| label.to_string());
        candidates.push(Candidate {
            query,
            detection: Detection {
                bbox,
                score,
                label,
                class_name,
                reading_order: None,
            },
        });
    }

    // Stable sorting preserves descending score order among labels emitted by
    // the same query. Renumber after filtering so the public order is dense.
    candidates.sort_by_key(|candidate| rank_by_query[candidate.query]);
    let mut detections = Vec::with_capacity(candidates.len());
    for (index, mut candidate) in candidates.into_iter().enumerate() {
        candidate.detection.reading_order = Some((index + 1) as u32);
        detections.push(candidate.detection);
    }
    Ok(detections)
}

fn validate_query_count(q: usize) -> Result<(), String> {
    if q == 0 || q > 512 {
        return Err(format!("query count must be in 1..=512; got {q}"));
    }
    Ok(())
}

fn ensure_finite(name: &str, values: &[f32]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("{name}[{index}] is not finite: {value}"));
    }
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_decoder_matches_global_pointer_vote_equation() {
        let logits = [0.0, -10.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0];
        assert_eq!(decode_reading_order(&logits, 3).unwrap(), [1, 0, 2]);
    }

    #[test]
    fn flat_top_q_uses_full_order_then_stably_renumbers() {
        let logits = [5.0, 4.0, 3.0, -10.0, 2.0, -10.0];
        let boxes = [
            0.5, 0.5, 0.4, 0.2, // query 0
            0.2, 0.3, 0.2, 0.2, // query 1
            0.9, 0.9, 0.4, 0.4, // query 2, clipped if retained
        ];
        // Query order is 1, 0, 2 even though query 2 is not in flat top-Q.
        let order = [0.0, -10.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0];
        let labels = ["title".to_string(), "text".to_string()];
        let detections = decode_detections(
            &logits,
            &boxes,
            &order,
            3,
            2,
            100.0,
            200.0,
            0.5,
            Some(&labels),
        )
        .unwrap();

        assert_eq!(detections.len(), 3);
        assert_eq!(detections[0].label, 0); // query 1
        for (actual, expected) in detections[0].bbox.iter().zip([10.0, 40.0, 30.0, 80.0]) {
            assert!((*actual - expected).abs() < 1e-4);
        }
        assert_eq!(detections[1].class_name, "title"); // query 0, higher score
        assert_eq!(detections[2].class_name, "text"); // same query, stable
        assert_eq!(
            detections
                .iter()
                .map(|detection| detection.reading_order)
                .collect::<Vec<_>>(),
            [Some(1), Some(2), Some(3)]
        );
    }

    #[test]
    fn optional_labels_fall_back_to_numeric_names_without_clipping() {
        let detections = decode_detections(
            &[2.0],
            &[1.0, 0.0, 1.0, 1.0],
            &[0.0],
            1,
            1,
            100.0,
            200.0,
            0.5,
            None,
        )
        .unwrap();
        assert_eq!(detections[0].class_name, "0");
        assert_eq!(detections[0].bbox, [50.0, -100.0, 150.0, 100.0]);
    }

    #[test]
    fn malformed_or_nonfinite_outputs_fail() {
        assert!(
            decode_detections(&[0.0], &[0.0; 4], &[0.0], 2, 1, 1.0, 1.0, 0.5, None)
                .unwrap_err()
                .contains("shape mismatch")
        );
        assert!(
            decode_detections(&[f32::NAN], &[0.0; 4], &[0.0], 1, 1, 1.0, 1.0, 0.5, None,)
                .unwrap_err()
                .contains("not finite")
        );
        assert!(
            decode_reading_order(&[0.0, f32::INFINITY, 0.0, 0.0], 2)
                .unwrap_err()
                .contains("not finite")
        );
        assert!(
            decode_reading_order(&[], 513)
                .unwrap_err()
                .contains("1..=512")
        );
    }
}
