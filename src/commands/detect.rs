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

//! CLI handler for `mlxcel detect`.
//!
//! Runs the detector selected by a checkpoint's `model_type` and prints
//! bounding boxes (`l, t, r, b, label, confidence` in original-image pixels).
//! This is the output surface for detection models, kept separate from the
//! text/VLM `generate` loop because detection emits boxes rather than tokens.

use anyhow::{Result, anyhow};
use image::ImageEncoder;

use mlxcel::initialize_runtime_checked;
use mlxcel::vision::detection::{detector_info, load_detector};

use crate::DetectArgs;

/// Run the `mlxcel detect` subcommand.
pub(crate) fn run_detect(args: DetectArgs) -> Result<()> {
    if !args.model.exists() {
        return Err(anyhow!(
            "Model directory does not exist: {}",
            args.model.display()
        ));
    }
    if !args.image.exists() {
        return Err(anyhow!(
            "Image file does not exist: {}",
            args.image.display()
        ));
    }
    let info = detector_info(&args.model)
        .map_err(|error| anyhow!("Failed to inspect detector config: {error}"))?;
    let threshold = args.threshold.unwrap_or(info.default_threshold);
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(anyhow!("--threshold must be in [0, 1], got {}", threshold));
    }
    if args.masks_dir.is_some() && !info.supports_masks {
        return Err(anyhow!(
            "--masks-dir requires an RF-DETR segmentation checkpoint; model_type {:?} does not emit masks",
            info.model_type
        ));
    }
    if let Some(directory) = &args.masks_dir {
        prepare_masks_dir(directory)?;
    }

    // Initialize the MLX runtime (selects GPU/CPU) before any forward pass.
    // Refuses to start when this build's CUDA architectures do not cover the
    // host GPU (#1537).
    let _runtime = initialize_runtime_checked()?;

    let detector = load_detector(&args.model, threshold)
        .map_err(|error| anyhow!("Failed to load detector: {error}"))?;
    if detector.supports_masks() != info.supports_masks {
        return Err(anyhow!(
            "detector mask capability changed between config inspection and model loading"
        ));
    }
    let detector_name = detector.name();
    let result = detector
        .predict_path_with_masks(&args.image, args.masks_dir.is_some())
        .map_err(|error| anyhow!("{detector_name} detection failed: {error}"))?;

    let mut detections = result.detections;
    sort_reading_order(&mut detections);
    let mask_paths = match &args.masks_dir {
        Some(directory) => save_masks(&detections, directory)?,
        None => vec![None; detections.len()],
    };

    match args.format {
        OutputFormat::Text => {
            if detections.is_empty() {
                println!(
                    "No detections above threshold {:.2} in {}",
                    threshold,
                    args.image.display()
                );
                return Ok(());
            }
            println!(
                "{} detection(s) in {} (threshold {:.2}):",
                detections.len(),
                args.image.display(),
                threshold
            );
            let show_order = detections.iter().any(|d| d.reading_order.is_some());
            if show_order {
                println!(
                    "{:>5}  {:<22} {:>7}  {:>8} {:>8} {:>8} {:>8}",
                    "order", "label", "conf", "l", "t", "r", "b"
                );
            } else {
                println!(
                    "{:<22} {:>7}  {:>8} {:>8} {:>8} {:>8}",
                    "label", "conf", "l", "t", "r", "b"
                );
            }
            for d in &detections {
                if show_order {
                    let order = d
                        .reading_order
                        .map_or_else(|| "-".to_string(), |value| value.to_string());
                    println!(
                        "{:>5}  {:<22} {:>7.3}  {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                        order, d.class_name, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
                    );
                } else {
                    println!(
                        "{:<22} {:>7.3}  {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                        d.class_name, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
                    );
                }
            }
        }
        OutputFormat::Json => {
            let arr: Vec<serde_json::Value> = detections
                .iter()
                .zip(&mask_paths)
                .map(|(detection, mask)| detection_json(detection, mask.as_deref()))
                .collect();
            let out = serde_json::json!({
                "image": args.image.display().to_string(),
                "threshold": threshold,
                "detections": arr,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
    }

    Ok(())
}

fn detection_json(
    detection: &mlxcel::vision::detection::Detection,
    mask_path: Option<&str>,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "label": detection.class_name,
        "label_id": detection.label,
        "confidence": detection.score,
        "box": {
            "l": detection.bbox[0],
            "t": detection.bbox[1],
            "r": detection.bbox[2],
            "b": detection.bbox[3],
        }
    });
    if let Some(order) = detection.reading_order {
        value
            .as_object_mut()
            .expect("detection JSON is an object")
            .insert("reading_order".to_string(), serde_json::json!(order));
    }
    if let Some(mask_path) = mask_path {
        value
            .as_object_mut()
            .expect("detection JSON is an object")
            .insert("mask".to_string(), serde_json::json!(mask_path));
    }
    value
}

fn prepare_masks_dir(directory: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(directory).map_err(|error| {
        anyhow!(
            "Failed to create mask output directory {}: {error}",
            directory.display()
        )
    })?;
    if !directory
        .metadata()
        .map_err(|error| anyhow!("Failed to inspect {}: {error}", directory.display()))?
        .is_dir()
    {
        return Err(anyhow!(
            "Mask output path is not a directory: {}",
            directory.display()
        ));
    }
    let probe = directory.join(format!(".mlxcel-mask-write-probe-{}", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|error| {
            anyhow!(
                "Mask output directory {} is not writable: {error}",
                directory.display()
            )
        })?;
    drop(file);
    std::fs::remove_file(&probe).map_err(|error| {
        anyhow!(
            "Failed to remove mask output probe {}: {error}",
            probe.display()
        )
    })
}

fn save_masks(
    detections: &[mlxcel::vision::detection::Detection],
    directory: &std::path::Path,
) -> Result<Vec<Option<String>>> {
    detections
        .iter()
        .enumerate()
        .map(|(index, detection)| {
            let mask = detection.mask.as_ref().ok_or_else(|| {
                anyhow!("RF-DETR segmentation detection {index} does not contain a mask")
            })?;
            let path = directory.join(format!("mask-{index:03}.png"));
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| anyhow!("Failed to create {}: {error}", path.display()))?;
            let encoded = image::codecs::png::PngEncoder::new(file).write_image(
                mask.as_raw(),
                mask.width(),
                mask.height(),
                image::ExtendedColorType::L8,
            );
            if let Err(error) = encoded {
                let _ = std::fs::remove_file(&path);
                return Err(anyhow!("Failed to save {}: {error}", path.display()));
            }
            Ok(Some(path.display().to_string()))
        })
        .collect()
}

/// Order detections top-to-bottom then left-to-right, so the emitted list is
/// document reading order rather than detector confidence order.
///
/// This is what makes `mlxcel detect --format json | mlxcel generate
/// --layout-detections` correct end to end: the Falcon-OCR layout path
/// deliberately preserves the order of the file it is handed (the reference
/// detector, PP-DocLayoutV3, carries an `order_logits` head and emits reading
/// order directly), so whatever order this command prints becomes the order the
/// per-region OCR output appears in. RT-DETRv2 has no order head, so the order
/// is derived from geometry here instead.
///
/// The comparison is a strict total order so the output is reproducible:
/// top edge, then left edge, then descending confidence, then class id. The
/// confidence tiebreak keeps the several labels a single query can emit (see
/// `decode_detections`) adjacent and most-confident-first, since they share one
/// box and therefore compare equal on geometry.
///
/// Geometry-derived order is an approximation: it reads a multi-column page
/// row-wise rather than column-wise, because nothing in the box coordinates
/// distinguishes a two-column layout from a wide single-column one.
fn sort_reading_order(detections: &mut [mlxcel::vision::detection::Detection]) {
    if detections
        .iter()
        .all(|detection| detection.reading_order.is_some())
    {
        detections.sort_by_key(|detection| detection.reading_order);
        return;
    }
    detections.sort_by(|a, b| {
        let key = |d: &mlxcel::vision::detection::Detection| (d.bbox[1], d.bbox[0]);
        let (ka, kb) = (key(a), key(b));
        ka.0.partial_cmp(&kb.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.label.cmp(&b.label))
    });
}

/// Output format for the `detect` subcommand.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub(crate) enum OutputFormat {
    /// Human-readable aligned table.
    #[default]
    Text,
    /// Machine-readable JSON.
    Json,
}

#[cfg(test)]
#[path = "detect_tests.rs"]
mod tests;
