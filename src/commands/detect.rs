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

//! CLI handler for `mlxcel detect`.
//!
//! Runs object detection on an image with an RT-DETRv2 checkpoint and prints
//! bounding boxes (`l, t, r, b, label, confidence` in original-image pixels).
//! This is the output surface for detection models, kept separate from the
//! text/VLM `generate` loop because detection emits boxes rather than tokens.

use anyhow::{Result, anyhow};

use mlxcel::initialize_runtime;
use mlxcel::vision::detection::RtDetrV2Predictor;

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
    if !(0.0..=1.0).contains(&args.threshold) {
        return Err(anyhow!(
            "--threshold must be in [0, 1], got {}",
            args.threshold
        ));
    }

    // Initialize the MLX runtime (selects GPU/CPU) before any forward pass.
    let _runtime = initialize_runtime();

    let predictor = RtDetrV2Predictor::from_pretrained(&args.model, args.threshold)
        .map_err(|e| anyhow!("failed to load RT-DETRv2 model: {e}"))?;

    let result = predictor
        .predict_path(&args.image)
        .map_err(|e| anyhow!("detection failed: {e}"))?;

    // Sort detections by descending confidence for stable, readable output.
    let mut detections = result.detections;
    detections.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    match args.format {
        OutputFormat::Text => {
            if detections.is_empty() {
                println!(
                    "No detections above threshold {:.2} in {}",
                    args.threshold,
                    args.image.display()
                );
                return Ok(());
            }
            println!(
                "{} detection(s) in {} (threshold {:.2}):",
                detections.len(),
                args.image.display(),
                args.threshold
            );
            println!(
                "{:<22} {:>7}  {:>8} {:>8} {:>8} {:>8}",
                "label", "conf", "l", "t", "r", "b"
            );
            for d in &detections {
                println!(
                    "{:<22} {:>7.3}  {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                    d.class_name, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
                );
            }
        }
        OutputFormat::Json => {
            let arr: Vec<serde_json::Value> = detections
                .iter()
                .map(|d| {
                    serde_json::json!({
                        "label": d.class_name,
                        "label_id": d.label,
                        "confidence": d.score,
                        "box": {
                            "l": d.bbox[0],
                            "t": d.bbox[1],
                            "r": d.bbox[2],
                            "b": d.bbox[3],
                        }
                    })
                })
                .collect();
            let out = serde_json::json!({
                "image": args.image.display().to_string(),
                "threshold": args.threshold,
                "detections": arr,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
    }

    Ok(())
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
