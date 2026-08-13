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

//! Unit tests for `mlxcel detect` output ordering.

use super::*;
use mlxcel::vision::detection::Detection;

fn det(class_name: &str, label: usize, bbox: [f32; 4], score: f32) -> Detection {
    Detection {
        bbox,
        score,
        label,
        class_name: class_name.to_string(),
    }
}

/// The emitted list is document reading order, not detector confidence order.
///
/// `mlxcel generate --layout-detections` preserves the order of the file it is
/// handed, so this ordering is what makes the two commands compose into
/// correctly ordered per-region OCR (issue #1089).
#[test]
fn detections_are_emitted_in_reading_order() {
    // A page whose confidence order and reading order disagree: the body
    // paragraphs score highest but sit below the headings that introduce them.
    let mut dets = vec![
        det("text", 9, [59.0, 213.0, 957.0, 427.0], 0.947),
        det("text", 9, [64.0, 541.0, 943.0, 718.0], 0.926),
        det("text", 9, [70.0, 825.0, 961.0, 962.0], 0.808),
        det("section_header", 7, [72.0, 153.0, 463.0, 185.0], 0.798),
        det("section_header", 7, [81.0, 482.0, 513.0, 513.0], 0.712),
        det("section_header", 7, [80.0, 60.0, 771.0, 109.0], 0.706),
        det("page_footer", 4, [80.0, 1200.0, 670.0, 1237.0], 0.510),
    ];
    sort_reading_order(&mut dets);

    let tops: Vec<f32> = dets.iter().map(|d| d.bbox[1]).collect();
    assert_eq!(
        tops,
        vec![60.0, 153.0, 213.0, 482.0, 541.0, 825.0, 1200.0],
        "expected top-to-bottom order, got {tops:?}"
    );
    // The footer is last, which is the whole point on a tall page.
    assert_eq!(dets.last().unwrap().class_name, "page_footer");
}

/// Boxes on the same line read left to right.
#[test]
fn boxes_sharing_a_top_edge_read_left_to_right() {
    let mut dets = vec![
        det("text", 9, [500.0, 100.0, 900.0, 150.0], 0.5),
        det("text", 9, [100.0, 100.0, 400.0, 150.0], 0.4),
    ];
    sort_reading_order(&mut dets);
    let lefts: Vec<f32> = dets.iter().map(|d| d.bbox[0]).collect();
    assert_eq!(lefts, vec![100.0, 500.0]);
}

/// The several labels one query can emit share a box, so they compare equal on
/// geometry. They must stay adjacent and most-confident-first, which is what
/// lets the layout planner keep the best label when it collapses them to one
/// region.
#[test]
fn labels_sharing_one_box_stay_adjacent_most_confident_first() {
    let heading = [80.0, 60.0, 771.0, 109.0];
    let mut dets = vec![
        det("page_header", 5, heading, 0.400),
        det("title", 10, heading, 0.415),
        det("text", 9, [59.0, 213.0, 957.0, 427.0], 0.947),
        det("section_header", 7, heading, 0.706),
    ];
    sort_reading_order(&mut dets);

    let names: Vec<&str> = dets.iter().map(|d| d.class_name.as_str()).collect();
    assert_eq!(
        names,
        vec!["section_header", "title", "page_header", "text"],
        "the heading's labels must group together, best first"
    );
}

/// The order is a strict total order, so equal geometry and equal confidence
/// still sort reproducibly rather than depending on input order.
#[test]
fn fully_tied_detections_order_by_class_id() {
    let same = [10.0, 10.0, 100.0, 50.0];
    let mut dets = vec![det("text", 9, same, 0.5), det("footnote", 1, same, 0.5)];
    sort_reading_order(&mut dets);
    assert_eq!(dets[0].label, 1);

    let mut reversed = vec![det("footnote", 1, same, 0.5), det("text", 9, same, 0.5)];
    sort_reading_order(&mut reversed);
    assert_eq!(reversed[0].label, 1, "order must not depend on input order");
}
