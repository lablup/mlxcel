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

use super::*;

#[test]
fn missing_processor_config_uses_pinned_defaults() {
    let directory = tempfile::tempdir().unwrap();
    let config = InklingProcessorConfig::from_model_path(directory.path()).unwrap();
    assert_eq!(config, InklingProcessorConfig::default());
    config.validate().unwrap();
}

#[test]
fn loaded_processor_config_accepts_exact_frontend_and_rejects_drift() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join("processor_config.json"),
        serde_json::json!({
            "feature_extractor": {
                "feature_size": 80,
                "sampling_rate": 16_000,
                "hop_length": 800,
                "n_fft": 1_600,
                "window_size": 1_600,
                "audio_token_duration_s": 0.05,
                "window_size_multiplier": 2.0,
                "return_attention_mask": true
            },
            "num_dmel_bins": 16,
            "dmel_min_value": -7.0,
            "dmel_max_value": 2.0
        })
        .to_string(),
    )
    .unwrap();
    InklingProcessorConfig::from_model_path(directory.path()).unwrap();

    std::fs::write(
        directory.path().join("processor_config.json"),
        serde_json::json!({
            "feature_extractor": {"sampling_rate": 24_000}
        })
        .to_string(),
    )
    .unwrap();
    assert!(InklingProcessorConfig::from_model_path(directory.path()).is_err());
}
