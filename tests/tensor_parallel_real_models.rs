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

mod common;

use std::path::Path;

use common::repo_model_dir;
use mlxcel::{
    CxxGenerator, LanguageModel, SamplingConfig, distributed::ShardConfig, initialize_runtime,
    load_model, load_model_with_tensor_parallel, tokenizer::MlxcelTokenizer,
};

fn prompt_tokens(tokenizer: &MlxcelTokenizer, prompt: &str) -> Vec<i32> {
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    tokenizer
        .encode(prompt, add_special)
        .unwrap()
        .into_iter()
        .map(|token| token as i32)
        .collect()
}

fn decode_tokens(tokenizer: &MlxcelTokenizer, tokens: &[i32]) -> String {
    let tokens: Vec<u32> = tokens.iter().map(|&token| token as u32).collect();
    tokenizer.decode(&tokens, true).unwrap_or_default()
}

fn assert_tp_matches_single_rank_stepwise(
    model_dir: &Path,
    prompt: &str,
    tp_size: usize,
    decode_token: i32,
) {
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let runtime = initialize_runtime();
    eprintln!(
        "Comparing TP stepwise parity on {} using {}",
        model_dir.display(),
        runtime.device
    );
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (full_model, tokenizer) = load_model(model_dir).unwrap();
    let prompt_tokens = prompt_tokens(&tokenizer, prompt);
    let prompt_ids = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let decode_ids = mlxcel_core::from_slice_i32(&[decode_token], &[1, 1]);
    let mut full_caches = full_model.make_caches();
    let full_prefill = full_model.forward(&prompt_ids, &mut full_caches, None);
    let full_decode = full_model.forward(&decode_ids, &mut full_caches, None);

    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (tp_model, _) =
        load_model_with_tensor_parallel(model_dir, None, &ShardConfig::with_tp_size(tp_size))
            .unwrap();
    let mut tp_caches = tp_model.make_caches();
    let tp_prefill = tp_model.forward(&prompt_ids, &mut tp_caches, None);
    let tp_decode = tp_model.forward(&decode_ids, &mut tp_caches, None);

    let prefill_shape = mlxcel_core::array_shape(&full_prefill);
    let seq_len = prefill_shape[1];
    let vocab_size = prefill_shape[2];
    let full_prefill_last = mlxcel_core::slice(
        &full_prefill,
        &[0, seq_len - 1, 0],
        &[1, seq_len, vocab_size],
    );
    let tp_prefill_last =
        mlxcel_core::slice(&tp_prefill, &[0, seq_len - 1, 0], &[1, seq_len, vocab_size]);

    let atol = 1e-4f32;
    let prefill_close = mlxcel_core::allclose(
        &full_prefill_last,
        &tp_prefill_last,
        atol as f64,
        atol as f64,
    );
    let decode_close = mlxcel_core::allclose(&full_decode, &tp_decode, atol as f64, atol as f64);
    let mut prefill_ok = mlxcel_core::item_bool(&prefill_close);
    if !prefill_ok {
        let diff = mlxcel_core::subtract(&full_prefill_last, &tp_prefill_last);
        let abs_diff = mlxcel_core::abs(&diff);
        let max_diff = mlxcel_core::max_all(&abs_diff);
        mlxcel_core::eval(&max_diff);
        let max_diff_val = mlxcel_core::item_f32(&max_diff);
        prefill_ok = max_diff_val <= atol;
        let full_finite = mlxcel_core::all_all(&mlxcel_core::isfinite(&full_prefill_last));
        let tp_finite = mlxcel_core::all_all(&mlxcel_core::isfinite(&tp_prefill_last));
        mlxcel_core::eval(&full_finite);
        mlxcel_core::eval(&tp_finite);
        let full_argmax = mlxcel_core::argmax_last_axis(&full_prefill_last);
        let tp_argmax = mlxcel_core::argmax_last_axis(&tp_prefill_last);
        mlxcel_core::eval(&full_argmax);
        mlxcel_core::eval(&tp_argmax);
        eprintln!(
            "prefill-last max_abs_diff={} full_argmax={} tp_argmax={} full_finite={} tp_finite={}",
            max_diff_val,
            mlxcel_core::item_i32(&full_argmax),
            mlxcel_core::item_i32(&tp_argmax),
            mlxcel_core::item_bool(&full_finite),
            mlxcel_core::item_bool(&tp_finite)
        );
    }
    let mut decode_ok = mlxcel_core::item_bool(&decode_close);
    if !decode_ok {
        let diff = mlxcel_core::subtract(&full_decode, &tp_decode);
        let abs_diff = mlxcel_core::abs(&diff);
        let max_diff = mlxcel_core::max_all(&abs_diff);
        mlxcel_core::eval(&max_diff);
        let max_diff_val = mlxcel_core::item_f32(&max_diff);
        decode_ok = max_diff_val <= atol;
        let full_finite = mlxcel_core::all_all(&mlxcel_core::isfinite(&full_decode));
        let tp_finite = mlxcel_core::all_all(&mlxcel_core::isfinite(&tp_decode));
        mlxcel_core::eval(&full_finite);
        mlxcel_core::eval(&tp_finite);
        let full_argmax = mlxcel_core::argmax_last_axis(&full_decode);
        let tp_argmax = mlxcel_core::argmax_last_axis(&tp_decode);
        mlxcel_core::eval(&full_argmax);
        mlxcel_core::eval(&tp_argmax);
        eprintln!(
            "decode max_abs_diff={} full_argmax={} tp_argmax={} full_finite={} tp_finite={}",
            max_diff_val,
            mlxcel_core::item_i32(&full_argmax),
            mlxcel_core::item_i32(&tp_argmax),
            mlxcel_core::item_bool(&full_finite),
            mlxcel_core::item_bool(&tp_finite)
        );
    }
    assert!(
        prefill_ok,
        "prefill logits mismatch for {} at tp={}",
        model_dir.display(),
        tp_size
    );
    assert!(
        decode_ok,
        "decode logits mismatch for {} at tp={}",
        model_dir.display(),
        tp_size
    );
}

fn assert_tp_matches_single_rank(
    model_dir: &Path,
    prompt: &str,
    max_tokens: usize,
    tp_size: usize,
) {
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let runtime = initialize_runtime();
    eprintln!(
        "Comparing TP parity on {} using {}",
        model_dir.display(),
        runtime.device
    );
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (tokenizer, prompt_tokens, single_rank_tokens) = {
        let (single_rank_model, tokenizer) = load_model(model_dir).unwrap();
        let prompt_tokens = prompt_tokens(&tokenizer, prompt);
        let sampling = SamplingConfig::greedy();
        let mut single_rank_generator = CxxGenerator::new(single_rank_model.num_layers());
        let single_rank_tokens = single_rank_generator.generate(
            &single_rank_model,
            &prompt_tokens,
            max_tokens,
            &sampling,
        );
        (tokenizer, prompt_tokens, single_rank_tokens)
    };
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let tensor_parallel_tokens = {
        let (tensor_parallel_model, _) =
            load_model_with_tensor_parallel(model_dir, None, &ShardConfig::with_tp_size(tp_size))
                .unwrap();
        let sampling = SamplingConfig::greedy();
        let mut tensor_parallel_generator = CxxGenerator::new(tensor_parallel_model.num_layers());
        tensor_parallel_generator.generate(
            &tensor_parallel_model,
            &prompt_tokens,
            max_tokens,
            &sampling,
        )
    };
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    assert!(
        single_rank_tokens.len() >= 8,
        "expected a longer generation for {}, got {} tokens",
        model_dir.display(),
        single_rank_tokens.len()
    );
    assert_eq!(
        single_rank_tokens,
        tensor_parallel_tokens,
        "generated token mismatch for {}\nsingle-rank: {:?}\ntp={}: {:?}\nsingle-rank text: {}\ntp={} text: {}",
        model_dir.display(),
        single_rank_tokens,
        tp_size,
        tensor_parallel_tokens,
        decode_tokens(&tokenizer, &single_rank_tokens),
        tp_size,
        decode_tokens(&tokenizer, &tensor_parallel_tokens)
    );
}

fn assert_tp_generates_tokens(
    model_dir: &Path,
    prompt: &str,
    max_tokens: usize,
    tp_size: usize,
    min_tokens: usize,
) {
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let runtime = initialize_runtime();
    eprintln!(
        "Running TP generation smoke on {} using {}",
        model_dir.display(),
        runtime.device
    );
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (tensor_parallel_model, tokenizer) =
        load_model_with_tensor_parallel(model_dir, None, &ShardConfig::with_tp_size(tp_size))
            .unwrap();
    let prompt_tokens = prompt_tokens(&tokenizer, prompt);
    let sampling = SamplingConfig::greedy();
    let mut tensor_parallel_generator = CxxGenerator::new(tensor_parallel_model.num_layers());
    let generated = tensor_parallel_generator.generate(
        &tensor_parallel_model,
        &prompt_tokens,
        max_tokens,
        &sampling,
    );
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    assert!(
        generated.len() >= min_tokens,
        "expected TP generation on {} to emit at least {} tokens, got {} ({})",
        model_dir.display(),
        min_tokens,
        generated.len(),
        decode_tokens(&tokenizer, &generated)
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn llama_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("llama-3.2-1b-4bit"),
        "Continue this sequence with more entries separated by commas: 1, 2, 3, 4, 5,",
        32,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn llama31_8b_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("llama-3.1-8b-4bit"),
        "Continue this sequence with more entries separated by commas: 1, 2, 3, 4, 5,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen2_5_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen2.5-0.5b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        24,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen2_5_7b_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen2.5-7b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen3-0.6b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        32,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen3-0.6b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_4b_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen3-4b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_5_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen3.5-0.8b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        24,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_5_4b_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen3.5-4b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_5_9b_tp2_matches_single_rank_stepwise() {
    assert_tp_matches_single_rank_stepwise(
        &repo_model_dir("qwen3.5-9b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        2,
        11,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_5_9b_tp4_matches_single_rank_stepwise() {
    assert_tp_matches_single_rank_stepwise(
        &repo_model_dir("qwen3.5-9b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        4,
        11,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_5_27b_tp4_matches_single_rank_stepwise() {
    assert_tp_matches_single_rank_stepwise(
        &repo_model_dir("qwen3.5-27b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        4,
        11,
    );
}

/// Qwen3.8-27B is architecturally identical to Qwen3.5-27B and rides the same
/// `qwen3_5` path (#1163). It is here so a change to that path is checked
/// against both generations, not only the one it was written for.
#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_8_27b_tp4_matches_single_rank_stepwise() {
    assert_tp_matches_single_rank_stepwise(
        &repo_model_dir("qwen3.8-27b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        4,
        11,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn ernie45_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("ernie-4.5-0.3b-4bit"),
        "Continue this sequence with more entries separated by commas: red, blue, green,",
        32,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn ernie45_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("ernie-4.5-0.3b-4bit"),
        "Continue this sequence with more entries separated by commas: red, blue, green,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn hunyuan_v1_dense_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("hunyuan-1.8b-4bit"),
        "Continue this sequence with more entries separated by commas: spring, summer, autumn,",
        32,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn hunyuan_v1_dense_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("hunyuan-1.8b-4bit"),
        "Continue this sequence with more entries separated by commas: spring, summer, autumn,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn gemma3_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("gemma3-1b-4bit"),
        "Continue this sequence with more entries separated by commas: north, south, east,",
        24,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn llama_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("llama-3.2-1b-4bit"),
        "Continue this sequence with more entries separated by commas: 1, 2, 3, 4, 5,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn gemma3_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("gemma3-1b-4bit"),
        "Continue this sequence with more entries separated by commas: north, south, east,",
        24,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn gemma4_31b_tp4_matches_single_rank_stepwise() {
    assert_tp_matches_single_rank(
        &repo_model_dir("gemma-4-31b-4bit"),
        "Continue this sequence with more entries separated by commas: north, south, east,",
        24,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn gemma4_e2b_tp2_matches_single_rank_stepwise() {
    assert_tp_generates_tokens(
        &repo_model_dir("gemma-4-e2b-it-4bit"),
        "Continue this sequence with more entries separated by commas: north, south, east,",
        12,
        2,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn gemma4_e4b_tp4_matches_single_rank_stepwise() {
    assert_tp_matches_single_rank_stepwise(
        &repo_model_dir("gemma-4-e4b-it-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        4,
        11,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn gemma4_26b_a4b_tp4_matches_single_rank_stepwise() {
    assert_tp_matches_single_rank_stepwise(
        &repo_model_dir("gemma-4-26b-a4b-it-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        4,
        11,
    );
}
