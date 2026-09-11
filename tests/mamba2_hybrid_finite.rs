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

//! The guard `granitemoehybrid.rs` and `falcon_h1.rs` name in the comment that
//! records why the M5 per-mixer `eval` was removed. It did not exist: both
//! comments cited `mamba2_hybrid_decode_is_finite` and nothing defined it, so
//! the claim that the regression was covered rested on a test that was never
//! written.
//!
//! Run enough forwards to see a sporadic fault. The reduction defect that
//! prompted this reproduced on roughly one call in six, so a single pass proves
//! nothing; sixty is chosen to make a fault of that rate practically certain
//! while still finishing in seconds. Raised to 250 after 60 caught it in only one
//! run of three: two test functions sharing the GPU lowers the observed rate.

mod common;
use common::repo_model_dir;

fn finite_over_repeated_forwards(model_name: &str) {
    let dir = repo_model_dir(model_name);
    if !dir.exists() {
        eprintln!("Skipping {model_name}: not in the store at {dir:?}");
        return;
    }
    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load model");
    let tokens: Vec<i32> = tokenizer
        .encode("Hello, world.", true)
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();

    for iteration in 0..250 {
        let input_ids = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);
        let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
        let logits =
            mlxcel_core::generate::LanguageModel::forward(&model, &input_ids, &mut caches, None);
        mlxcel_core::eval(&logits);

        let shape = mlxcel_core::array_shape(&logits);
        let vocab = *shape.last().unwrap();
        let last = shape[1] - 1;
        let row = mlxcel_core::slice(&logits, &[0, last, 0], &[1, last + 1, vocab]);
        let max = mlxcel_core::max_all(&row);
        mlxcel_core::eval(&max);
        let v = mlxcel_core::item_f32(&max);
        assert!(
            v.is_finite(),
            "{model_name}: iteration {iteration} produced {v} (nan={}, inf={})",
            v.is_nan(),
            v.is_infinite()
        );
    }
}

// One model per test function. Running another model first changed the outcome:
// with the tiny checkpoint exercised ahead of it the vision checkpoint stopped
// faulting entirely, so a combined test would have passed without the fix and
// guarded nothing.
#[test]
fn mamba2_hybrid_decode_is_finite() {
    finite_over_repeated_forwards("granite-4.0-3b-vision-4bit");
}

#[test]
fn mamba2_hybrid_decode_is_finite_tiny() {
    finite_over_repeated_forwards("granite-4.0-h-tiny-4bit");
}
