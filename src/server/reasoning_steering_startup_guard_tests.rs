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

use serde_json::json;

use super::{ServePaths, refuse_reasoning_effort_steering_on_unsteered_paths};
use crate::distributed::disaggregated::ServingMode;

/// A tiny config directory, with or without a steering block.
fn model_dir(steered: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = json!({"model_type": "raon1_mini", "num_hidden_layers": 4});
    if steered {
        config["reasoning_effort_steering"] = json!({
            "alpha": {"none": 0, "low": 0.3, "high": 0.05, "max": 0},
            "default": "high",
            "layers": [1, 2]
        });
    }
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("config");
    dir
}

const SINGLE_NODE: ServePaths = ServePaths {
    pipeline_parallel: false,
    serving_mode: ServingMode::Hybrid,
    xla: false,
};

fn refusal(paths: ServePaths) -> String {
    let dir = model_dir(true);
    let err = refuse_reasoning_effort_steering_on_unsteered_paths(dir.path(), paths)
        .expect_err("a steering model must be refused on this path");
    let message = err.to_string();
    assert!(
        message.contains(&dir.path().display().to_string()),
        "{message}"
    );
    assert!(message.contains("reasoning_effort_steering"), "{message}");
    message
}

#[test]
fn pipeline_parallel_startup_refuses_a_steering_model() {
    let message = refusal(ServePaths {
        pipeline_parallel: true,
        ..SINGLE_NODE
    });
    assert!(message.contains("pipeline-parallel"), "{message}");
}

#[test]
fn disaggregated_prefill_and_decode_startup_refuse_a_steering_model() {
    let prefill = refusal(ServePaths {
        serving_mode: ServingMode::PrefillOnly,
        ..SINGLE_NODE
    });
    assert!(prefill.contains("disaggregated prefill"), "{prefill}");
    let decode = refusal(ServePaths {
        serving_mode: ServingMode::DecodeOnly,
        ..SINGLE_NODE
    });
    assert!(decode.contains("disaggregated decode"), "{decode}");
}

#[test]
fn xla_startup_refuses_a_steering_model() {
    let message = refusal(ServePaths {
        xla: true,
        ..SINGLE_NODE
    });
    assert!(message.contains("XLA"), "{message}");
}

#[test]
fn single_node_steering_and_unsteered_models_pass() {
    let steered = model_dir(true);
    refuse_reasoning_effort_steering_on_unsteered_paths(steered.path(), SINGLE_NODE)
        .expect("the batch scheduler carries the level");
    let router = ServePaths {
        serving_mode: ServingMode::Router,
        ..SINGLE_NODE
    };
    refuse_reasoning_effort_steering_on_unsteered_paths(steered.path(), router)
        .expect("the router loads no model");
    let plain = model_dir(false);
    for paths in [
        ServePaths {
            pipeline_parallel: true,
            ..SINGLE_NODE
        },
        ServePaths {
            serving_mode: ServingMode::DecodeOnly,
            ..SINGLE_NODE
        },
        ServePaths {
            xla: true,
            ..SINGLE_NODE
        },
    ] {
        refuse_reasoning_effort_steering_on_unsteered_paths(plain.path(), paths)
            .expect("a model without steering is unaffected");
    }
    refuse_reasoning_effort_steering_on_unsteered_paths(std::path::Path::new(""), router)
        .expect("image-only serving has no checkpoint");
}
