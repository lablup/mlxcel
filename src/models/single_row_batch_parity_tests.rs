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

//! A single row equals a batch of one (ADR 0007, #2172 item 7).
//!
//! `Engine::step` is the one decode entry for every row count, and at `B=1`
//! it runs the family's single-row forward (`forward_with_sequence_id`), as
//! the provided `forward_batched` does at `b == 1`. The families that
//! override the batched entry (Qwen3 and Llama 3 through `attend_batched`)
//! must agree with their single-row forward bitwise, or the engine's B=1
//! route and the server's batched route would differ for a lone request.
//! These tests pin that on a real checkpoint: the same prompt is prefilled
//! into two cache sets, then four decode tokens run through the single-row
//! forward on one and the one-row batched forward on the other, and every
//! step's logits must be byte-identical.
//!
//! Real-checkpoint tests: `#[ignore]`, run under
//! `MLXCEL_SDPA_DETERMINISTIC=1` on the GPU host with
//! `cargo test --profile test-fast --features cuda --lib single_row_batch_parity -- --ignored --test-threads=1`.
//! The checkpoint comes from `MLXCEL_QWEN3_MODEL` / `MLXCEL_LLAMA3_MODEL`,
//! else `models/mlx/qwen3-1.7b-4bit` / `models/mlx/llama-3.2-1b-instruct-4bit`.

use std::path::PathBuf;

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;

const DECODE_STEPS: usize = 4;

fn checkpoint(env: &str, default: &str) -> Option<PathBuf> {
    let path = std::env::var(env).map_or_else(|_| PathBuf::from(default), PathBuf::from);
    if path.join("config.json").exists() {
        Some(path)
    } else {
        eprintln!(
            "skipping: no checkpoint at {} (set {env} to point at one)",
            path.display()
        );
        None
    }
}

fn argmax_last(logits: &mlxcel_core::MlxArray) -> i32 {
    let last = mlxcel_core::slice_last_logits(logits);
    let token = mlxcel_core::fused_sample(&last, 0.0, 0, 1.0, 0.0);
    mlxcel_core::eval(&token);
    mlxcel_core::item_i32(&token)
}

fn evaluated_bytes(logits: &mlxcel_core::MlxArray) -> (Vec<i32>, Vec<u8>) {
    mlxcel_core::eval(logits);
    (
        mlxcel_core::array_shape(logits),
        mlxcel_core::array_evaluated_bytes(logits),
    )
}

fn run(env: &str, default: &str) {
    if std::env::var("MLXCEL_SDPA_DETERMINISTIC").ok().as_deref() != Some("1") {
        eprintln!(
            "skipping: run with MLXCEL_SDPA_DETERMINISTIC=1 so both routes reduce in one order"
        );
        return;
    }
    let Some(path) = checkpoint(env, default) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let (model, tokenizer) = crate::load_model(&path).expect("load checkpoint");
    assert!(
        model.supports_batching(),
        "this test pins a family that overrides the batched entry"
    );
    let prompt: Vec<i32> = tokenizer
        .encode("The quick brown fox jumps over the lazy dog because", true)
        .expect("tokenize")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    assert!(
        prompt.len() > 4,
        "prompt tokenized to {} tokens",
        prompt.len()
    );

    let mut single: Vec<KVCache> = model.make_caches();
    let mut batched: Vec<KVCache> = model.make_caches();
    let input = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);
    let logits_single = model.forward_with_sequence_id(&input, None, &mut single, None);
    let logits_batched = model.forward_with_sequence_id(&input, None, &mut batched, None);
    let mut token = argmax_last(&logits_single);
    assert_eq!(token, argmax_last(&logits_batched), "prefill diverged");

    for step in 0..DECODE_STEPS {
        let next = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        let one = model.forward_with_sequence_id(&next, None, &mut single, None);
        let row = model.forward_batched_with_ids(&next, None, &mut [batched.as_mut_slice()], None);
        let (shape_one, bytes_one) = evaluated_bytes(&one);
        let (shape_row, bytes_row) = evaluated_bytes(&row);
        assert_eq!(
            shape_one, shape_row,
            "logits shape differs at decode step {step}"
        );
        assert!(
            bytes_one == bytes_row,
            "single-row and one-row batched logits differ at decode step {step} ({} of {} bytes differ)",
            bytes_one
                .iter()
                .zip(&bytes_row)
                .filter(|(a, b)| a != b)
                .count(),
            bytes_one.len()
        );
        token = argmax_last(&one);
    }
}

#[test]
#[ignore = "needs a real Qwen3 checkpoint and the GPU"]
fn qwen3_single_row_matches_one_row_batched_bitwise() {
    run("MLXCEL_QWEN3_MODEL", "models/mlx/qwen3-1.7b-4bit");
}

#[test]
#[ignore = "needs a real Llama 3 checkpoint and the GPU"]
fn llama3_single_row_matches_one_row_batched_bitwise() {
    run(
        "MLXCEL_LLAMA3_MODEL",
        "models/mlx/llama-3.2-1b-instruct-4bit",
    );
}
