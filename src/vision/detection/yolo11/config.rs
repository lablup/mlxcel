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

//! YOLO11 detector configuration.

use std::collections::BTreeMap;

use serde::Deserialize;

/// Backbone widths the OmniParser `icon_detect` layer table produces at P3 /
/// P4 / P5. A checkpoint that declares different widths is a different
/// topology and cannot use this graph.
pub const DETECT_CHANNELS: [usize; 3] = [256, 512, 512];
/// Strides of the three detection scales.
pub const DETECT_STRIDES: [usize; 3] = [8, 16, 32];

/// `config.json` of a converted YOLO11 OmniParser detector.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Yolo11Config {
    pub model_type: String,
    /// Number of classes the head predicts.
    pub nc: usize,
    /// Detection-head input channels per scale.
    pub ch: Vec<usize>,
    /// DFL bins per box edge.
    pub reg_max: usize,
    /// Class id (as a JSON key string) to class name.
    pub names: BTreeMap<String, String>,
    /// Detection strides per scale.
    pub stride: Vec<usize>,
}

impl Default for Yolo11Config {
    fn default() -> Self {
        Self {
            model_type: "yolo11".into(),
            nc: 1,
            ch: DETECT_CHANNELS.to_vec(),
            reg_max: 16,
            names: BTreeMap::new(),
            stride: DETECT_STRIDES.to_vec(),
        }
    }
}

impl Yolo11Config {
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(json)
            .map_err(|error| format!("YOLO11 config parse error: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Reject configurations the fixed OmniParser topology cannot execute.
    pub fn validate(&self) -> Result<(), String> {
        if self.model_type != "yolo11" {
            return Err(format!(
                "unsupported YOLO11 model_type {:?}; expected \"yolo11\"",
                self.model_type
            ));
        }
        if self.nc == 0 || self.nc > 4096 {
            return Err(format!("YOLO11 nc must be in 1..=4096, got {}", self.nc));
        }
        if self.reg_max == 0 || self.reg_max > 64 {
            return Err(format!(
                "YOLO11 reg_max must be in 1..=64, got {}",
                self.reg_max
            ));
        }
        if self.ch != DETECT_CHANNELS {
            return Err(format!(
                "YOLO11 detection channels must be {DETECT_CHANNELS:?}, got {:?}; \
                 the OmniParser layer table has a fixed depth and width",
                self.ch
            ));
        }
        if self.stride != DETECT_STRIDES {
            return Err(format!(
                "YOLO11 strides must be {DETECT_STRIDES:?}, got {:?}",
                self.stride
            ));
        }
        for key in self.names.keys() {
            let id: usize = key
                .parse()
                .map_err(|_| format!("YOLO11 names key {key:?} is not a nonnegative class id"))?;
            if id >= self.nc {
                return Err(format!(
                    "YOLO11 names key {key:?} is outside nc={}",
                    self.nc
                ));
            }
        }
        Ok(())
    }

    /// Class name for a head label, falling back to the numeric id.
    pub fn class_name(&self, label: usize) -> String {
        self.names
            .get(&label.to_string())
            .cloned()
            .unwrap_or_else(|| label.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_omniparser_header() {
        let config = Yolo11Config::from_json_str(
            r#"{"model_type":"yolo11","nc":1,"ch":[256,512,512],"reg_max":16,
                "names":{"0":"icon"},"stride":[8,16,32]}"#,
        )
        .unwrap();
        assert_eq!(config.class_name(0), "icon");
        assert_eq!(config.class_name(3), "3");
    }

    #[test]
    fn rejects_a_different_topology() {
        let error = Yolo11Config {
            ch: vec![128, 256, 256],
            ..Yolo11Config::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.contains("channels"), "{error}");

        let error = Yolo11Config {
            stride: vec![4, 8, 16],
            ..Yolo11Config::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.contains("strides"), "{error}");
    }

    #[test]
    fn rejects_out_of_range_names() {
        let mut config = Yolo11Config::default();
        config.names.insert("1".into(), "ghost".into());
        assert!(config.validate().unwrap_err().contains("outside nc"));
    }
}
