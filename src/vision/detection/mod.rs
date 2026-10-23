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

//! Config-dispatched object-detection vision models.
//!
//! Detection models output bounding boxes rather than a token stream, so they
//! sit outside the text/VLM generation flow and are exposed through the
//! `detect` CLI subcommand.

use std::path::Path;

pub mod pp_doclayout_v3;
pub mod rt_detr_v2;

pub use pp_doclayout_v3::{PpDocLayoutV3Config, PpDocLayoutV3Model, PpDocLayoutV3Predictor};
pub use rt_detr_v2::{RtDetrV2Config, RtDetrV2Model, RtDetrV2Predictor};

/// One detected box.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// `(left, top, right, bottom)` in original-image pixel coordinates.
    pub bbox: [f32; 4],
    /// Confidence in `[0, 1]`.
    pub score: f32,
    /// Integer class id. Repeated class names still retain distinct ids.
    pub label: usize,
    /// Human-readable class name.
    pub class_name: String,
    /// One-based model reading order. Detectors without an order head use
    /// `None`; callers must then derive their own geometric order.
    pub reading_order: Option<u32>,
}

/// All detections for one image.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DetectionResult {
    pub detections: Vec<Detection>,
}

/// Config-dispatched detector used by the `detect` command.
pub trait Detector {
    /// Detect objects in one image file.
    fn predict_path(&self, path: &Path) -> Result<DetectionResult, String>;

    /// Stable family name used in diagnostics.
    fn name(&self) -> &'static str;
}

/// Load the detector declared by `config.json`'s exact `model_type`.
pub fn load_detector(dir: &Path, threshold: f32) -> Result<Box<dyn Detector>, String> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(format!(
            "detector confidence threshold must be finite and in [0, 1], got {threshold}"
        ));
    }
    let config_path = dir.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .map_err(|error| format!("failed to read {}: {error}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse {}: {error}", config_path.display()))?;
    let model_type = value
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            format!(
                "{} does not contain a string model_type",
                config_path.display()
            )
        })?;

    match model_type {
        "rt_detr_v2" => RtDetrV2Predictor::from_pretrained(dir, threshold)
            .map(|predictor| Box::new(predictor) as Box<dyn Detector>)
            .map_err(|error| format!("failed to load rt_detr_v2 detector: {error}")),
        "pp_doclayout_v3" => PpDocLayoutV3Predictor::from_pretrained(dir, threshold)
            .map(|predictor| Box::new(predictor) as Box<dyn Detector>)
            .map_err(|error| format!("failed to load pp_doclayout_v3 detector: {error}")),
        other => Err(format!("unsupported detector model_type '{other}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_loader_rejects_invalid_threshold_before_dispatch() {
        for threshold in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            let error = load_detector(Path::new("missing-detector"), threshold)
                .err()
                .expect("invalid threshold must fail before model dispatch");
            assert!(error.contains("threshold"), "{threshold}: {error}");
        }
    }
}
