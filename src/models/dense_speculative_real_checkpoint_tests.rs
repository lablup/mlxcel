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

//! Real-checkpoint gates for the two published DSpark pairings of issue
//! #3261: `Qwen/Qwen3-4B` with `RedHatAI/Qwen3-4B-speculator.dspark`, and
//! `openbmb/MiniCPM5-2B` with `openbmb/MiniCPM5-2B-DSpark`.
//!
//! Each test soft-skips when a checkpoint is absent, following the convention
//! of `bert_real_checkpoint_tests.rs`. Fetch with:
//!
//! ```sh
//! mlxcel download Qwen/Qwen3-4B
//! mlxcel download RedHatAI/Qwen3-4B-speculator.dspark
//! mlxcel download openbmb/MiniCPM5-2B
//! mlxcel download openbmb/MiniCPM5-2B-DSpark
//! ```
//!
//! ## What is compared, and why row by row
//!
//! The contract of a speculative burst is that its temperature-0 tokens are
//! the target's own greedy tokens. Whether a multi-row verify block
//! reproduces single-token decode bit for bit is a property of the host's
//! kernels, which the block-versus-chain probe measures and the burst is
//! gated on; where the probe declines (a bf16 target on CUDA takes another
//! matmul kernel and another attention kernel for one row than for eight),
//! drafted text can leave classic text at a near-tie.
//!
//! These tests separate the two questions. [`RowwiseVerify`] runs each row
//! of a verify block as its own single-token forward, so the verify is
//! classic decode's arithmetic by construction, on every backend. With it,
//! any difference between drafted and classic tokens would be a defect in
//! the parts this change adds (the config parse, the drafter's draft step,
//! the layer captures, the hidden slicing, the rollback, the round loop
//! wiring) and not kernel rounding. They also require that the drafter
//! actually had proposals accepted, so "equal" cannot be reached by a
//! drafter that proposes nothing useful.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use mlxcel_core::drafter::Drafter;
use mlxcel_core::drafter::dflash::{DFlashDrafter, DFlashGenerator, SpeculativeTarget};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::layers::KVCache;
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::{MlxArray, UniquePtr};

use super::*;

/// Locate a downloaded checkpoint that carries safetensors weights: the
/// mlxcel store, then the HuggingFace cache, then `<repo>/models/<name>`.
/// `None` skips the test.
fn local_checkpoint(repo_id: &str) -> Option<PathBuf> {
    let has_weights = |dir: &PathBuf| {
        dir.join("config.json").is_file()
            && (dir.join("model.safetensors").is_file()
                || dir.join("model.safetensors.index.json").is_file())
    };
    let candidates = [
        crate::downloader::model_dir(repo_id),
        crate::downloader::hf_cache_snapshot(repo_id, None),
        Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join(crate::downloader::repo_basename(repo_id)),
        ),
    ];
    let found = candidates.into_iter().flatten().find(has_weights);
    if found.is_none() {
        eprintln!(
            "skipping real-checkpoint gate: {repo_id} not present with safetensors weights \
             (mlxcel download {repo_id})"
        );
    }
    found
}

fn input(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

/// The greedy token of the last row of `logits` (`[1, L, vocab]`).
fn last_argmax(logits: &MlxArray) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let last = mlxcel_core::slice(logits, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
    let token = mlxcel_core::argmax_last_axis(&last);
    mlxcel_core::eval(&token);
    mlxcel_core::item_i32(&mlxcel_core::reshape(&token, &[]))
}

/// Classic greedy decode through `forward_last_logits`, the forward
/// `mlxcel generate` runs: one prefill, then one token at a time.
fn classic_greedy(model: &dyn LanguageModel, prompt: &[i32], count: usize) -> Vec<i32> {
    let mut caches = model.make_caches();
    let logits = model.forward_last_logits(&input(prompt), &mut caches, None, prompt.len() - 1);
    let mut next = last_argmax(&logits);
    let mut out = vec![next];
    while out.len() < count {
        let logits = model.forward_last_logits(&input(&[next]), &mut caches, None, 0);
        next = last_argmax(&logits);
        out.push(next);
    }
    out
}

/// A dense target whose verify block is run one row at a time: row `r` is
/// the single-token forward classic decode would run for it, so the block
/// equals the decode chain by construction. Everything else (captures,
/// rollback, the hidden concatenation) is the production target's.
struct RowwiseVerify<'a, T>(&'a T);

impl<T: DenseSpeculativeBackbone> SpeculativeTarget for RowwiseVerify<'_, T> {
    type Cache = KVCache;
    type VerifyOut = DenseVerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        &[]
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.verify_forward_with_capture_layers(verify_input, caches, &[])
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        let rows = mlxcel_core::array_shape(verify_input)[1];
        let steps: Vec<DenseVerifyOutput> = (0..rows)
            .map(|r| {
                let token = mlxcel_core::utils::slice_axis(verify_input, 1, r, r + 1);
                let out =
                    forward_speculative(self.0, &token, caches, capture_layer_ids, Rows::Last);
                mlxcel_core::eval(&out.logits);
                out
            })
            .collect();
        let logits: Vec<&MlxArray> = steps.iter().map(|s| &*s.logits).collect();
        let hidden_states = (0..capture_layer_ids.len())
            .map(|c| {
                let slabs: Vec<&MlxArray> = steps.iter().map(|s| &*s.hidden_states[c]).collect();
                mlxcel_core::concatenate_many(&slabs, 1)
            })
            .collect();
        DenseVerifyOutput {
            logits: mlxcel_core::concatenate_many(&logits, 1),
            hidden_states,
        }
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        _verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        rollback(caches, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        let slabs: Vec<&MlxArray> = verify_out.hidden_states.iter().map(|s| &**s).collect();
        mlxcel_core::concatenate_many(&slabs, -1)
    }

    fn verify_logits<'b>(&self, verify_out: &'b Self::VerifyOut) -> &'b MlxArray {
        &verify_out.logits
    }
}

/// Counters of one drafted run.
struct Drafted {
    tokens: Vec<i32>,
    proposed: usize,
    accepted: usize,
}

/// Load the drafter, run its pairing gates against `target`, prefill as the
/// offline driver does and run the round loop with a row-wise verify.
fn drafted_rowwise<T: DenseSpeculativeTarget + DenseSpeculativeBackbone>(
    target: &T,
    drafter_dir: &Path,
    prompt: &[i32],
    count: usize,
) -> Drafted {
    let drafter = DFlashDrafter::load(drafter_dir).expect("published DSpark drafter loads");
    assert!(drafter.is_dspark());
    drafter
        .validate_target_compat(target)
        .expect("the published pairing passes the drafter's gate");
    assert!(
        target.dflash_drafter_pairing_error(&drafter).is_none(),
        "the published pairing passes the target's gate"
    );
    let width = drafter
        .configured_block_size()
        .expect("a DSpark drafter declares its verify width");
    assert_eq!(
        width, 8,
        "both published drafters run at 7 proposals plus the anchor"
    );
    let taps = drafter.dflash_target_layer_ids().unwrap().to_vec();

    let mut caches = LanguageModel::make_caches(target);
    let out = target.dflash_prefill_planned(
        &input(prompt),
        &mut caches,
        &taps,
        mlxcel_core::prefill_plan::DEFAULT_PREFILL_CHUNK,
    );
    let first = last_argmax(&out.logits);
    let rowwise = RowwiseVerify(target);
    let first_hidden = rowwise.concat_hidden_for_drafter(&out);
    drop(out);

    let mut generator = DFlashGenerator::new(
        Box::new(drafter),
        SamplingConfig::greedy(),
        width as u32,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let cancel = AtomicBool::new(false);
    let run = generator
        .run(
            &rowwise,
            target as &dyn LanguageModel,
            &mut caches,
            first,
            first_hidden,
            &[],
            count,
            &cancel,
            &LogprobsConfig::default(),
        )
        .expect("round loop");
    let mut tokens = vec![first];
    tokens.extend(run.tokens);
    Drafted {
        tokens,
        proposed: run.diagnostics.proposed_tokens,
        accepted: run.diagnostics.accepted_tokens,
    }
}

/// The prompts: the two the issue names, as plain text (no chat template, so
/// the test needs nothing but the tokenizer), and a longer one that puts the
/// drafter's context past a few hundred rows.
fn prompts() -> Vec<String> {
    let long = "Speculative decoding lets a small draft model propose several tokens that a \
                large target model then checks in one forward pass. "
        .repeat(12);
    vec![
        "Explain speculative decoding.".to_string(),
        "Write a Python function that adds two integers.".to_string(),
        format!("{long}In one paragraph, explain why the output does not change."),
    ]
}

fn check_pairing<T: DenseSpeculativeTarget + DenseSpeculativeBackbone>(
    name: &str,
    target: &T,
    target_dir: &Path,
    drafter_dir: &Path,
) {
    let tokenizer = crate::tokenizer::load_tokenizer(target_dir).expect("target tokenizer loads");
    let count = 96;
    let mut total_proposed = 0;
    let mut total_accepted = 0;
    for prompt in prompts() {
        let ids: Vec<i32> = tokenizer
            .encode(&prompt, true)
            .expect("prompt encodes")
            .into_iter()
            .map(|id| id as i32)
            .collect();
        assert!(!ids.is_empty());
        let classic = classic_greedy(target, &ids, count);
        let drafted = drafted_rowwise(target, drafter_dir, &ids, count);
        eprintln!(
            "REAL {name}: prompt_tokens={} generated={} proposed={} accepted={} rate={:.3}",
            ids.len(),
            drafted.tokens.len(),
            drafted.proposed,
            drafted.accepted,
            drafted.accepted as f64 / drafted.proposed.max(1) as f64
        );
        assert_eq!(
            drafted.tokens,
            classic,
            "{name}: with a row-wise verify the drafted tokens must be classic decode's tokens \
             (prompt of {} tokens)",
            ids.len()
        );
        total_proposed += drafted.proposed;
        total_accepted += drafted.accepted;
    }
    assert!(total_proposed > 0, "{name}: the drafter proposed nothing");
    assert!(
        total_accepted > 0,
        "{name}: none of {total_proposed} proposals was accepted"
    );
}

#[test]
fn qwen3_4b_with_the_speculators_dspark_drafter_emits_classic_tokens() {
    let (Some(target_dir), Some(drafter_dir)) = (
        local_checkpoint("Qwen/Qwen3-4B"),
        local_checkpoint("RedHatAI/Qwen3-4B-speculator.dspark"),
    ) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let (target, _args) = Qwen3Model::load(&target_dir).expect("Qwen3-4B loads");
    check_pairing("qwen3-4b", &target, &target_dir, &drafter_dir);
}

#[test]
fn minicpm5_2b_with_the_openbmb_dspark_drafter_emits_classic_tokens() {
    let (Some(target_dir), Some(drafter_dir)) = (
        local_checkpoint("openbmb/MiniCPM5-2B"),
        local_checkpoint("openbmb/MiniCPM5-2B-DSpark"),
    ) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let (target, _args) = Llama3Model::load(&target_dir).expect("MiniCPM5-2B loads");
    check_pairing("minicpm5-2b", &target, &target_dir, &drafter_dir);
}
