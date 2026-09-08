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

//! Mamba2 hybrid decode stays finite without a per-mixer `eval` boundary.
//!
//! `falcon_h1` and `granitemoehybrid` used to force `mlxcel_core::eval` at the
//! end of every mixer forward on M5 Max, because the lazy float32/float16 SSM
//! graph was observed fusing into NaN inside one Metal command buffer. That
//! boundary is a per-layer GPU sync and cost 3.31x and 2.24x of decode there.
//! It was removed after measuring byte-identical greedy output without it.
//!
//! These tests are what makes the removal safe to keep: they decode long
//! enough to build the graph the boundary used to split, and fail if any
//! output token is unusable. A silent NaN would otherwise reach a user as
//! garbage text rather than a failing build.
//!
//! The guarded failure mode is M5-only, so a green result off M5 says nothing
//! about it. These tests therefore report three distinct states rather than
//! collapsing them into "pass": verified, not-applicable on this host, and
//! applicable but unverified because the checkpoint is absent. The last one is
//! the dangerous state, because that is a machine where the risk exists and
//! nothing checked it, so it is printed loudly.
//!
//! **Both tests are `#[ignore]`d.** The `self-hosted-macos-26-arm64` pool is M1
//! and M4, so in CI they could only ever report NOT APPLICABLE, and a green run
//! that verified nothing is worse than no run at all. Select them explicitly on
//! an M5 host that has the two checkpoints:
//!
//! ```text
//! cargo test --release --features metal,accelerate \
//!     --test mamba2_hybrid_decode_finite -- --ignored
//! ```
//!
//! Treat that as a manual step before shipping a change to the Mamba2 hybrid
//! mixers. Adding an M5 machine to the runner pool is what would make this
//! automatic; until then a green CI badge is not evidence about this code.

mod common;

use common::repo_model_dir;
use mlxcel::{CxxGenerator, LanguageModel, SamplingConfig, initialize_runtime, load_model};

/// Decode `max_tokens` greedily and assert the text is usable.
///
/// A NaN in the SSM graph surfaces as an empty generation, a repeated
/// degenerate token, or replacement characters, so the assertions cover all
/// three rather than only checking for a literal "NaN".
fn assert_decode_is_finite(model_name: &str, max_tokens: usize) {
    // The eval boundary this test replaces only ever fired on M5 Max, so the NaN
    // it prevented cannot arise elsewhere. Passing on another chip would be a
    // vacuous green, so say plainly that nothing was verified here.
    if !mlxcel_core::hardware::is_m5_neural_accelerator() {
        eprintln!(
            "{model_name}: NOT APPLICABLE on this host. The removed eval boundary was \
             M5-only (has_neural_accelerator && macos_supports_na), so this host cannot \
             exhibit the failure and this result verifies nothing about it."
        );
        return;
    }

    let model_dir = repo_model_dir(model_name);
    if !model_dir.join("config.json").exists() {
        eprintln!(
            "{model_name}: NOT VERIFIED. This host is M5, so the failure this test guards \
             CAN occur here, but the checkpoint is missing at {}. Fetch it to make this \
             test meaningful.",
            model_dir.display()
        );
        return;
    }

    let _runtime = initialize_runtime();
    let (model, tokenizer) =
        load_model(&model_dir).unwrap_or_else(|e| panic!("{model_name} must load: {e}"));

    let prompt = "The capital of France is";
    let prompt_ids: Vec<i32> = tokenizer
        .encode(prompt, true)
        .expect("tokenize prompt")
        .iter()
        .map(|&id| id as i32)
        .collect();
    assert!(!prompt_ids.is_empty());

    let mut generator = CxxGenerator::new(model.num_layers());
    let tokens = generator.generate(&model, &prompt_ids, max_tokens, &SamplingConfig::greedy());
    assert!(
        !tokens.is_empty(),
        "{model_name}: greedy decode produced no tokens, which is how a NaN logit row surfaces"
    );

    let gen_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, true).expect("decode generation");
    eprintln!("[{model_name}] {} tokens: {text:?}", tokens.len());

    assert!(
        text.chars().any(|c| c.is_ascii_alphabetic()),
        "{model_name}: continuation has no letters, got {text:?}"
    );
    assert!(
        !text.contains('\u{FFFD}'),
        "{model_name}: continuation contains replacement characters, got {text:?}"
    );
    let distinct = tokens
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct > 1 || tokens.len() == 1,
        "{model_name}: every generated token is identical, which is what a collapsed \
         (NaN-poisoned) logit distribution looks like: {text:?}"
    );
}

#[test]
#[ignore = "M5-only: the guarded failure cannot occur on the M1/M4 CI runners"]
fn falcon_h1_decode_is_finite() {
    assert_decode_is_finite("falcon-h1-tiny-90m-instruct-4bit", 64);
}

#[test]
#[ignore = "M5-only: the guarded failure cannot occur on the M1/M4 CI runners"]
fn granitemoehybrid_decode_is_finite() {
    assert_decode_is_finite("granite-4.0-h-tiny-4bit", 64);
}
