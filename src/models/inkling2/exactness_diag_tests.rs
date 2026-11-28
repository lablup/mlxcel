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

//! Where a K-row verify block first parts from K single-token steps on a
//! real Inkling 2 checkpoint (issue #3280). Ignored; run under the GPU lock:
//!
//! ```text
//! cargo test --profile test-fast --features metal,accelerate --lib \
//!     models::inkling2::exactness_diag_tests -- --ignored --nocapture
//! ```
//!
//! `INKLING2_DIAG_MODEL` picks the checkpoint (default `models/inkling-2-mini`),
//! `INKLING2_DIAG_WIDTH` the block width (default 8), and
//! `INKLING2_DIAG_VERIFY=0` runs the block outside the speculative scope, and
//! `INKLING2_DIAG_PROMPT` sets the prompt length (default 32).

use super::Inkling2Model;
use crate::models::speculative_exactness_diag::{self as diag, Stage};
use mlxcel_core::sampling::TokenBiasMap;

#[test]
#[ignore = "needs a real Inkling 2 checkpoint and the GPU lock"]
fn verify_block_vs_chain_per_visit() {
    let path =
        std::env::var("INKLING2_DIAG_MODEL").unwrap_or_else(|_| "models/inkling-2-mini".into());
    if !std::path::Path::new(&path).join("config.json").exists() {
        eprintln!("skipping: {path} is not present");
        return;
    }
    let width: usize = std::env::var("INKLING2_DIAG_WIDTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let verify = std::env::var("INKLING2_DIAG_VERIFY").map_or(true, |v| v != "0");
    // The wired limit the CLI sets: without it a checkpoint the size of
    // Small (363 GiB) is not kept resident and a forward takes seconds.
    let _ = crate::execution::runtime::initialize_runtime();
    let (model, _) = Inkling2Model::load(&path).unwrap();
    report(&model, &path, width, verify);
    // The verdict the DSpark gate reads (`dflash_exactness_allows`).
    println!(
        "probe_block_chain_exactness({width}): {:?}",
        model.probe_block_chain_exactness(width)
    );
}

/// The same comparison on the synthetic model of `tiny_tests`.
#[test]
#[ignore = "diagnostic; run under the GPU lock"]
fn tiny_verify_block_vs_chain_per_visit() {
    for passes in [1, 3] {
        report(&super::tiny_tests::model(passes, false), "tiny", 2, true);
    }
}

fn report(model: &Inkling2Model, path: &str, width: usize, verify: bool) {
    let visits = model.layer_indices.len();
    let ids: Vec<usize> = (0..=visits).collect();
    let none = TokenBiasMap::new();
    let vocab = model.config.text_config.logits_width();
    let prompt_len: usize = std::env::var("INKLING2_DIAG_PROMPT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let prompt: Vec<i32> = (0..prompt_len)
        .map(|i| ((i * 7 + 2) % vocab) as i32)
        .collect();
    let block: Vec<i32> = (0..width).map(|i| ((i * 13 + 4) % vocab) as i32).collect();
    let input = |t: &[i32]| mlxcel_core::from_slice_i32(t, &[1, t.len() as i32]);

    // One stage per captured visit, then the logits, through the shared
    // per-layer diagnostic (issue #3409).
    let stages = |out: &super::speculative::Inkling2VerifyOutput| -> Vec<Stage> {
        let mut stages: Vec<Stage> = model
            .layer_indices
            .iter()
            .chain(std::iter::once(&usize::MAX))
            .enumerate()
            .map(|(v, &layer)| {
                let kind = if layer == usize::MAX {
                    "final".to_owned()
                } else {
                    format!("layer {layer}")
                };
                let hidden = out.hidden_states[v].as_ref().unwrap();
                Stage::of(v, &kind, "visit input", hidden)
            })
            .collect();
        stages.push(Stage::of(visits + 1, "final", "logits", &out.logits));
        stages
    };

    let mut chain = model.make_internal_caches();
    let _ = model.forward_speculative(&input(&prompt), &mut chain, &[], false, &none);
    let chain_trace: Vec<Vec<Stage>> = block
        .iter()
        .map(|&t| stages(&model.forward_speculative(&input(&[t]), &mut chain, &ids, false, &none)))
        .collect();

    let mut caches = model.make_internal_caches();
    let _ = model.forward_speculative(&input(&prompt), &mut caches, &[], false, &none);
    let block_trace =
        stages(&model.forward_speculative(&input(&block), &mut caches, &ids, verify, &none));

    println!("model {path} width {width} verify {verify} visits {visits}");
    for line in diag::report(&block_trace, &chain_trace) {
        println!("{line}");
    }
    match diag::locate(&block_trace, &chain_trace) {
        Ok(Some(found)) => println!("{found}"),
        Ok(None) => println!("every captured visit and the logits are byte-identical"),
        Err(why) => println!("no localization: {why}"),
    }
}
