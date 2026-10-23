// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Bounded parser for Falcon-OCR layout-detection JSON.

use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value};

use mlxcel::vision::detection::Detection;

/// Largest `--layout-detections` document that is read at all.
///
/// A real detector dump for one page is kilobytes, so this is three orders of
/// magnitude of headroom. The ceiling exists because the alternative is reading
/// whatever the path happens to be into memory unbounded: a truncated or
/// hostile file, or a fifo that never ends.
pub(super) const MAX_LAYOUT_DETECTIONS_BYTES: usize = 32 * 1024 * 1024;

/// Largest number of detections one page may carry.
///
/// A dense page yields low hundreds of boxes. The cap matters because the
/// nested-box suppression is quadratic in the entry count, so an array that
/// parses in a moment can still cost minutes of comparisons.
pub(super) const MAX_LAYOUT_DETECTIONS: usize = 4096;

/// Longest layout class name accepted, in characters.
///
/// Real class names are under 20 characters. The bound is on the class name in
/// particular because it is the one attacker-controlled string this command
/// echoes back to the terminal, in the summary's `no text category:` list.
pub(super) const MAX_LAYOUT_CLASS_NAME_CHARS: usize = 64;

/// Read and validate a `--layout-detections` file.
///
/// Called before the model is loaded so a malformed file costs nothing.
///
/// The read is bounded rather than a plain `read_to_string`, and the bound is
/// enforced on the bytes actually read rather than on `metadata().len()`, which
/// reports zero for a fifo and for anything else that is not a regular file.
pub(crate) fn load_layout_detections(path: &Path) -> Result<Vec<Detection>> {
    let read_bounded = || -> std::io::Result<String> {
        let file = std::fs::File::open(path)?;
        let mut text = String::new();
        file.take(MAX_LAYOUT_DETECTIONS_BYTES as u64 + 1)
            .read_to_string(&mut text)?;
        Ok(text)
    };
    let text = read_bounded()
        .with_context(|| format!("--layout-detections: failed to read {}", path.display()))?;
    if text.len() > MAX_LAYOUT_DETECTIONS_BYTES {
        bail!(
            "--layout-detections {}: the file is larger than the {} MiB limit; a detector \
             dump for one page is kilobytes",
            path.display(),
            MAX_LAYOUT_DETECTIONS_BYTES / (1024 * 1024)
        );
    }
    parse_layout_detections(&text)
        .map_err(|error| anyhow!("--layout-detections {}: {error}", path.display()))
}

/// Parse the detections array out of a `--layout-detections` document.
///
/// Three spellings are accepted, all producing the same [`Detection`] list:
///
/// - `{"detections": [...]}`, what `mlxcel detect --format json` prints;
/// - a bare top-level `[...]` array, convenient for a hand-written file;
/// - per entry, either `{"label", "confidence", "box": {"l","t","r","b"}}` or
///   the `{"category", "score", "bbox": [l, t, r, b]}` spelling mlx-vlm's
///   `falcon_ocr/layout.py` emits.
///
/// Every rejection names the offending entry by index, because a detector
/// dump is long enough that "invalid input" alone is not actionable.
pub(super) fn parse_layout_detections(text: &str) -> Result<Vec<Detection>> {
    let root: Value =
        serde_json::from_str(text).map_err(|error| anyhow!("not valid JSON: {error}"))?;

    let items: &[Value] = match &root {
        Value::Array(items) => items,
        Value::Object(map) => match map.get("detections") {
            Some(Value::Array(items)) => items,
            Some(other) => bail!("`detections` must be an array, got {}", json_kind(other)),
            None => bail!(
                "expected an object with a `detections` array (the shape `mlxcel detect \
                 --format json` prints) or a bare array of detections"
            ),
        },
        other => bail!(
            "expected a JSON object or array at the top level, got {}",
            json_kind(other)
        ),
    };

    // Checked before the loop so an enormous array is rejected without first
    // building the `Detection` vector it describes.
    if items.len() > MAX_LAYOUT_DETECTIONS {
        bail!(
            "{} detections is more than the {MAX_LAYOUT_DETECTIONS} this command accepts \
             for one page",
            items.len()
        );
    }

    let mut detections = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let object = item.as_object().ok_or_else(|| {
            anyhow!(
                "detection #{index}: expected an object, got {}",
                json_kind(item)
            )
        })?;
        detections.push(parse_detection(object, index)?);
    }
    Ok(detections)
}

fn parse_detection(object: &Map<String, Value>, index: usize) -> Result<Detection> {
    const CLASS_KEYS: [&str; 4] = ["label", "category", "class_name", "class"];

    let class_value = CLASS_KEYS
        .iter()
        .filter_map(|key| object.get(*key))
        .find(|value| value.is_string());
    let class_name = match class_value {
        Some(Value::String(name)) => name.trim(),
        // A key is there but holds a number (some dumps put the class id under
        // `label`). Say which key was wrong instead of "missing".
        _ if CLASS_KEYS.iter().any(|key| object.contains_key(*key)) => bail!(
            "detection #{index}: the layout class name must be a string (found a \
             non-string under `label` / `category` / `class_name`)"
        ),
        _ => bail!(
            "detection #{index}: missing the layout class name (expected `label`, \
             `category`, or `class_name`)"
        ),
    };
    if class_name.is_empty() {
        bail!("detection #{index}: the layout class name is empty");
    }
    // The class name is the one piece of this file that is printed back to the
    // operator verbatim (the summary's `no text category:` list), so a control
    // character in it is an escape-sequence injection into their terminal.
    if class_name.chars().any(char::is_control) {
        bail!("detection #{index}: the layout class name contains a control character");
    }
    if class_name.chars().count() > MAX_LAYOUT_CLASS_NAME_CHARS {
        bail!(
            "detection #{index}: the layout class name is longer than \
             {MAX_LAYOUT_CLASS_NAME_CHARS} characters"
        );
    }

    let bbox = parse_bbox(object, index)?;

    let score = match ["confidence", "score"]
        .iter()
        .find_map(|key| object.get(*key))
    {
        None => 1.0,
        Some(value) => {
            let score = as_f32(value).ok_or_else(|| {
                anyhow!(
                    "detection #{index}: `confidence` must be a number, got {}",
                    json_kind(value)
                )
            })?;
            if !score.is_finite() {
                bail!("detection #{index}: `confidence` is not a finite number");
            }
            if !(0.0..=1.0).contains(&score) {
                bail!("detection #{index}: `confidence` must be in [0, 1]");
            }
            score
        }
    };

    let label = match object.get("label_id") {
        None => 0,
        Some(value) => {
            let label = value.as_u64().ok_or_else(|| {
                anyhow!("detection #{index}: `label_id` must be a non-negative integer")
            })?;
            usize::try_from(label)
                .map_err(|_| anyhow!("detection #{index}: `label_id` exceeds the usize range"))?
        }
    };
    let reading_order = match object.get("reading_order") {
        None => None,
        Some(value) => {
            let order = value.as_u64().ok_or_else(|| {
                anyhow!("detection #{index}: `reading_order` must be a positive integer")
            })?;
            let order = u32::try_from(order).map_err(|_| {
                anyhow!("detection #{index}: `reading_order` exceeds the u32 range")
            })?;
            if order == 0 {
                bail!("detection #{index}: `reading_order` must be one-based");
            }
            Some(order)
        }
    };

    Ok(Detection {
        bbox,
        score,
        label,
        class_name: class_name.to_string(),
        reading_order,
    })
}

fn parse_bbox(object: &Map<String, Value>, index: usize) -> Result<[f32; 4]> {
    let raw = ["box", "bbox", "bounding_box"]
        .iter()
        .find_map(|key| object.get(*key))
        .ok_or_else(|| {
            anyhow!(
                "detection #{index}: missing the bounding box (expected a `box` object \
                 with `l` / `t` / `r` / `b`, or a `bbox` array of four numbers)"
            )
        })?;

    let bbox = match raw {
        Value::Array(values) => {
            if values.len() != 4 {
                bail!(
                    "detection #{index}: the bounding-box array must hold exactly four \
                     numbers (left, top, right, bottom), got {}",
                    values.len()
                );
            }
            let mut bbox = [0.0f32; 4];
            for (axis, value) in values.iter().enumerate() {
                bbox[axis] = as_f32(value).ok_or_else(|| {
                    anyhow!(
                        "detection #{index}: bounding-box entry {axis} must be a number, got {}",
                        json_kind(value)
                    )
                })?;
            }
            bbox
        }
        Value::Object(map) => {
            // `l/t/r/b` is what `mlxcel detect` prints; the other two spellings
            // are what detector dumps in the wild use.
            const AXES: [[&str; 3]; 4] = [
                ["l", "x1", "left"],
                ["t", "y1", "top"],
                ["r", "x2", "right"],
                ["b", "y2", "bottom"],
            ];
            let mut bbox = [0.0f32; 4];
            for (axis, aliases) in AXES.iter().enumerate() {
                let value = aliases
                    .iter()
                    .find_map(|key| map.get(*key))
                    .ok_or_else(|| {
                        anyhow!(
                            "detection #{index}: the bounding box has no `{}` (or `{}` / `{}`)",
                            aliases[0],
                            aliases[1],
                            aliases[2]
                        )
                    })?;
                bbox[axis] = as_f32(value).ok_or_else(|| {
                    anyhow!(
                        "detection #{index}: bounding-box `{}` must be a number, got {}",
                        aliases[0],
                        json_kind(value)
                    )
                })?;
            }
            bbox
        }
        other => bail!(
            "detection #{index}: the bounding box must be an object or an array, got {}",
            json_kind(other)
        ),
    };

    if bbox.iter().any(|coord| !coord.is_finite()) {
        bail!("detection #{index}: the bounding box has a non-finite coordinate");
    }
    if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
        bail!(
            "detection #{index}: the bounding box is empty or inverted \
             (left={:.2}, top={:.2}, right={:.2}, bottom={:.2}); expected right > left \
             and bottom > top",
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3]
        );
    }
    Ok(bbox)
}

fn as_f32(value: &Value) -> Option<f32> {
    value.as_f64().map(|number| number as f32)
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}
