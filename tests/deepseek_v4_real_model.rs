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

//! DeepSeek-V4 real-checkpoint smoke: load parity plus greedy generation.
//!
//! Loading alone is half the validation: `DeepSeekV4Model::from_weights`
//! runs the strict weight-coverage check, so a successful load proves every
//! tensor in the checkpoint index mapped onto a module path and every module
//! parameter found its tensor, with no silent fallbacks. Generation then
//! exercises all three attention kinds (the real `compress_ratios` mixes
//! local, compressed, and sparse-compressed layers), the hash-routed and
//! bias-routed MoE gates, the HyperConnection Sinkhorn gates, and the
//! rotating-window plus pooling caches across prefill and decode.
//!
//! ## Invocation
//!
//! ```bash
//! cargo test --test deepseek_v4_real_model --release --features metal,accelerate -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`-gated as real-model heavy (~145 GB checkpoint; needs a
//! high-memory Apple Silicon host). Skips silently when the checkpoint
//! directory is absent.

mod common;

use common::repo_model_dir;

use mlxcel::{CxxGenerator, LanguageModel, SamplingConfig, initialize_runtime, load_model};

const MODEL_DIR: &str = "deepseek-v4-flash-4bit";

#[test]
#[ignore]
fn deepseek_v4_real_model_loads_and_generates_coherently() {
    let model_dir = repo_model_dir(MODEL_DIR);
    if !model_dir.join("config.json").exists() {
        eprintln!(
            "Skipping: DeepSeek-V4 checkpoint not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/DeepSeek-V4-Flash-4bit",
            model_dir.display()
        );
        return;
    }

    let _runtime = initialize_runtime();
    let (model, tokenizer) = load_model(&model_dir).expect(
        "DeepSeek-V4 checkpoint must load: a failure here is a weight-coverage or \
         quantization-layout mismatch, not an environment problem",
    );

    let prompt = "The capital of France is";
    let prompt_ids: Vec<i32> = tokenizer
        .encode(prompt, true)
        .expect("tokenize prompt")
        .iter()
        .map(|&id| id as i32)
        .collect();
    assert!(!prompt_ids.is_empty());

    let mut generator = CxxGenerator::new(model.num_layers());
    let tokens = generator.generate(&model, &prompt_ids, 24, &SamplingConfig::greedy());
    assert!(
        !tokens.is_empty(),
        "greedy decode must produce at least one token before EOS"
    );

    let gen_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, true).expect("decode generation");
    eprintln!("[deepseek-v4] prompt: {prompt:?}");
    eprintln!("[deepseek-v4] greedy continuation ({} tokens): {text:?}", tokens.len());

    assert!(
        text.chars().any(|c| c.is_ascii_alphabetic()),
        "greedy continuation should contain text, got {text:?}"
    );
    assert!(
        text.contains("Paris"),
        "greedy continuation of {prompt:?} should mention Paris; got {text:?}. \
         A fluent-but-wrong answer here usually means a silently-misweighted \
         component (Sinkhorn order, overlap compressor, selection-vs-weighting \
         gate contract), not a load failure"
    );
}

#[test]
#[ignore]
fn deepseek_v4_real_model_decode_crosses_pooling_windows() {
    // A longer decode than one compress window (ratio 4) plus one full
    // simple window boundary region, so decode-mode pooling emission and the
    // HiSA decode fast path both run against real weights.
    let model_dir = repo_model_dir(MODEL_DIR);
    if !model_dir.join("config.json").exists() {
        eprintln!("Skipping: DeepSeek-V4 checkpoint not found");
        return;
    }

    let _runtime = initialize_runtime();
    let (model, tokenizer) = load_model(&model_dir).expect("load DeepSeek-V4 checkpoint");

    let prompt = "Write one short sentence about the ocean.";
    let prompt_ids: Vec<i32> = tokenizer
        .encode(prompt, true)
        .expect("tokenize prompt")
        .iter()
        .map(|&id| id as i32)
        .collect();

    let mut generator = CxxGenerator::new(model.num_layers());
    let tokens = generator.generate(&model, &prompt_ids, 48, &SamplingConfig::greedy());
    let gen_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, true).expect("decode generation");
    eprintln!("[deepseek-v4] 48-token decode: {text:?}");
    assert!(
        text.split_whitespace().count() >= 3,
        "a 48-token greedy decode should produce multiple words, got {text:?}"
    );
}
