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

//! LLM-jp-VL loader gates.
//!
//! The config surface is where the two released checkpoints disagree, so these
//! gates pin the three resolutions that differ between them: which key holds
//! the decoder config, which ids the image framing uses, and which ids stop
//! generation. The last gate reads both real `tokenizer_config.json` files
//! rather than trusting the defaults, because a synthetic fixture that agrees
//! with itself proves nothing about a checkpoint that ships different numbers.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::{Value, json};

use super::{
    added_token_ids, resolve_eos_token_ids, resolve_image_token_ids, select_text_config,
    validate_select_layer, validate_text_model_type,
};

/// Environment overrides for a checkout whose snapshots live outside the
/// store lookup (for example a shared `models/mlx/...` tree).
const JAGLE_DIR_ENV: &str = "MLXCEL_TEST_JAGLE_VL_DIR";
const LLMJP_9B_DIR_ENV: &str = "MLXCEL_TEST_LLMJP_4VL_9B_DIR";

fn tokenizer_config(entries: &[(i32, &str)]) -> Value {
    let decoder: serde_json::Map<String, Value> = entries
        .iter()
        .map(|(id, content)| (id.to_string(), json!({"content": content})))
        .collect();
    json!({ "added_tokens_decoder": decoder, "model_max_length": 4096 })
}

fn jagle_tokenizer_config() -> Value {
    tokenizer_config(&[
        (151645, "<|im_end|>"),
        (151655, "<|image_pad|>"),
        (151669, "<|image_start|>"),
        (151670, "<|image_end|>"),
        (151671, "<|start|>"),
        (151672, "<|end|>"),
        (151675, "<|return|>"),
    ])
}

fn llmjp_9b_tokenizer_config() -> Value {
    tokenizer_config(&[
        (2, "<|return|>"),
        (10, "<|start|>"),
        (11, "<|end|>"),
        (14, "<|image_pad|>"),
        (15, "<|image_start|>"),
        (16, "<|image_end|>"),
    ])
}

#[test]
fn llm_config_is_used_when_text_config_is_absent() {
    // Both released checkpoints carry `llm_config` and no `text_config`; the
    // InternVL loader's key would find nothing here.
    let config = json!({
        "model_type": "llmjpvl",
        "llm_config": {"model_type": "qwen3", "hidden_size": 2048},
    });
    let selected = select_text_config(&config).expect("llm_config is selected");
    assert_eq!(selected["model_type"], "qwen3");
    assert_eq!(selected["hidden_size"], 2048);

    // A converted checkpoint that renamed the key to the InternVL spelling
    // still loads.
    let converted = json!({
        "model_type": "llmjpvl",
        "text_config": {"model_type": "llama", "hidden_size": 4096},
    });
    assert_eq!(
        select_text_config(&converted).expect("text_config fallback")["model_type"],
        "llama"
    );

    // `llm_config` wins when a checkpoint somehow carries both.
    let both = json!({
        "llm_config": {"model_type": "qwen3"},
        "text_config": {"model_type": "llama"},
    });
    assert_eq!(
        select_text_config(&both).expect("llm_config wins")["model_type"],
        "qwen3"
    );

    // Neither key present is a load error, not a silent default.
    assert!(select_text_config(&json!({"model_type": "llmjpvl"})).is_err());
}

#[test]
fn top_level_quantization_is_inherited_by_the_decoder_config() {
    // A quantized conversion carries the block at the top level only; without
    // this the backbone loader would fall back to 64/4 defaults.
    let config = json!({
        "llm_config": {"model_type": "llama"},
        "quantization": {"group_size": 32, "bits": 8},
    });
    let selected = select_text_config(&config).expect("selects llm_config");
    assert_eq!(selected["quantization"]["group_size"], 32);
    assert_eq!(selected["quantization"]["bits"], 8);

    // A decoder config that already declares its own block keeps it.
    let own = json!({
        "llm_config": {"model_type": "llama", "quantization": {"group_size": 64, "bits": 4}},
        "quantization": {"group_size": 32, "bits": 8},
    });
    let selected = select_text_config(&own).expect("selects llm_config");
    assert_eq!(selected["quantization"]["group_size"], 64);
}

#[test]
fn image_token_ids_resolve_from_added_tokens_decoder() {
    // The two checkpoints disagree on every id, so a constant cannot serve
    // both.
    let jagle = added_token_ids(Some(&jagle_tokenizer_config()));
    let config = json!({"img_context_token_id": 151655});
    assert_eq!(
        resolve_image_token_ids(&config, &jagle),
        (151655, 151669, 151670)
    );

    let nine_b = added_token_ids(Some(&llmjp_9b_tokenizer_config()));
    let config = json!({"img_context_token_id": 14});
    assert_eq!(resolve_image_token_ids(&config, &nine_b), (14, 15, 16));

    // With no `img_context_token_id`, the tokenizer still resolves the pad id.
    assert_eq!(
        resolve_image_token_ids(&json!({}), &jagle),
        (151655, 151669, 151670)
    );

    // With neither, the 9B ids are the documented fallback.
    assert_eq!(
        resolve_image_token_ids(&json!({}), &HashMap::new()),
        (14, 15, 16)
    );
}

#[test]
fn stop_ids_combine_generation_config_and_the_harmony_terminators() {
    let jagle = added_token_ids(Some(&jagle_tokenizer_config()));
    let generation = json!({"eos_token_id": [151675, 151645]});
    let ids = resolve_eos_token_ids(Some(&generation), None, &jagle);
    for expected in [151675, 151645, 151672] {
        assert!(ids.contains(&expected), "{expected} missing from {ids:?}");
    }

    // The 9B's `generation_config.json` repeats one id; `<|end|>` comes from
    // the tokenizer.
    let nine_b = added_token_ids(Some(&llmjp_9b_tokenizer_config()));
    let generation = json!({"eos_token_id": [2, 2]});
    let ids = resolve_eos_token_ids(Some(&generation), None, &nine_b);
    assert_eq!(ids, vec![2, 11]);

    // A scalar `eos_token_id` on the decoder config is honored too.
    let ids = resolve_eos_token_ids(
        None,
        Some(&json!({"eos_token_id": 151645})),
        &HashMap::new(),
    );
    assert_eq!(ids, vec![151645]);

    // Nothing anywhere falls back to the 9B stop ids rather than to an empty
    // set that could never terminate a generation.
    assert_eq!(
        resolve_eos_token_ids(None, None, &HashMap::new()),
        vec![2, 11]
    );
}

#[test]
fn select_layer_other_than_minus_one_is_rejected_by_name() {
    assert!(validate_select_layer(&json!({"select_layer": -1})).is_ok());
    assert!(validate_select_layer(&json!({})).is_ok());
    let err = validate_select_layer(&json!({"select_layer": -2}))
        .expect_err("an intermediate-layer tap is out of scope");
    let message = format!("{err:#}");
    assert!(message.contains("-2"), "{message}");
    assert!(message.contains("select_layer"), "{message}");
}

#[test]
fn an_unsupported_decoder_backbone_is_named_in_the_error() {
    assert!(validate_text_model_type("llama").is_ok());
    assert!(validate_text_model_type("qwen3").is_ok());
    // A conversion may relabel the Llama-shaped decoder as qwen2; mlxcel
    // already serves that graph with the same backbone.
    assert!(validate_text_model_type("qwen2").is_ok());

    let err = validate_text_model_type("mistral").expect_err("unsupported backbone");
    let message = format!("{err:#}");
    assert!(message.contains("mistral"), "{message}");
}

/// Locate a released checkpoint, soft-skipping when it is not downloaded.
fn gate_checkpoint(env_key: &str, store_repo: &str, local_dirs: &[&str]) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var(env_key) {
        let dir = PathBuf::from(dir);
        if dir.join("config.json").is_file() {
            return Some(dir);
        }
        eprintln!("skipping real-checkpoint gate: {env_key} has no config.json");
        return None;
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut candidates: Vec<PathBuf> = crate::downloader::model_dir(store_repo)
        .into_iter()
        .collect();
    candidates.extend(local_dirs.iter().map(|dir| manifest.join(dir)));
    let found = candidates
        .into_iter()
        .find(|dir| dir.join("config.json").is_file());
    if found.is_none() {
        eprintln!("skipping real-checkpoint gate: {store_repo} not present");
    }
    found
}

fn read_json(dir: &std::path::Path, name: &str) -> Option<Value> {
    let text = std::fs::read_to_string(dir.join(name)).ok()?;
    serde_json::from_str(&text).ok()
}

/// The gate the synthetic fixtures above cannot stand in for: the ids are read
/// from the checkpoints' own `tokenizer_config.json` and `config.json`, so a
/// checkpoint that ships different numbers than the issue recorded fails here
/// instead of at inference time.
#[test]
fn image_and_stop_ids_match_both_released_tokenizers() {
    let cases: [(&str, &str, &[&str], (i32, i32, i32), &[i32]); 2] = [
        (
            JAGLE_DIR_ENV,
            "llm-jp/Jagle-VL-2.2B-Jagle-FineVision",
            &[
                "models/mlx/jagle-vl-2.2b-jagle-finevision",
                "models/Jagle-VL-2.2B-Jagle-FineVision",
            ],
            (151655, 151669, 151670),
            &[151675, 151645, 151672],
        ),
        (
            LLMJP_9B_DIR_ENV,
            "llm-jp/llm-jp-4-vl-9B-beta",
            &[
                "models/mlx/llm-jp-4-vl-9b-beta",
                "models/llm-jp-4-vl-9B-beta",
            ],
            (14, 15, 16),
            &[2, 11],
        ),
    ];

    let mut checked = 0usize;
    for (env_key, repo, dirs, expected_images, expected_stops) in cases {
        let Some(dir) = gate_checkpoint(env_key, repo, dirs) else {
            continue;
        };
        checked += 1;
        let config = read_json(&dir, "config.json").expect("config.json parses");
        assert_eq!(
            config.get("model_type").and_then(Value::as_str),
            Some("llmjpvl"),
            "{repo}: model_type"
        );
        assert!(
            config.get("llm_config").is_some(),
            "{repo}: the decoder config lives under llm_config"
        );
        assert_eq!(
            config.get("select_layer").and_then(Value::as_i64),
            Some(-1),
            "{repo}: only select_layer -1 is implemented"
        );
        validate_select_layer(&config).expect("released select_layer is accepted");

        let text_config = select_text_config(&config).expect("decoder config selected");
        let backbone = text_config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        validate_text_model_type(&backbone).expect("released backbone is supported");

        let tokenizer = read_json(&dir, "tokenizer_config.json");
        let added = added_token_ids(tokenizer.as_ref());
        assert_eq!(
            resolve_image_token_ids(&config, &added),
            expected_images,
            "{repo}: image framing ids"
        );

        let generation = read_json(&dir, "generation_config.json");
        let stops = resolve_eos_token_ids(generation.as_ref(), config.get("llm_config"), &added);
        for expected in expected_stops {
            assert!(
                stops.contains(expected),
                "{repo}: stop id {expected} missing from {stops:?}"
            );
        }

        // Neither checkpoint prepends BOS; a wrongly added one shifts every
        // position and silently degrades the answer.
        assert_eq!(
            tokenizer
                .as_ref()
                .and_then(|c| c.get("add_bos_token"))
                .and_then(Value::as_bool),
            Some(false),
            "{repo}: add_bos_token"
        );
    }

    if checked == 0 {
        eprintln!("skipping: neither LLM-jp-VL checkpoint is downloaded");
    }
}
