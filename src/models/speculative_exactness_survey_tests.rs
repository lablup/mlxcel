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

//! Block-versus-chain survey of one real checkpoint (issue #3409). Ignored;
//! run under the GPU lock:
//!
//! ```text
//! EXACTNESS_SURVEY_MODEL=models/qwen3.5-4b-4bit \
//! cargo test --profile test-fast --features metal,accelerate --lib \
//!     models::speculative_exactness_survey_tests -- --ignored --nocapture
//! ```
//!
//! - `EXACTNESS_SURVEY_MODEL`: the checkpoint directory (required).
//! - `EXACTNESS_SURVEY_WIDTHS`: verify widths, default `2,4,8`.
//! - `EXACTNESS_SURVEY_PROMPTS`: extra prompt lengths probed one draw each,
//!   default `600,2100` (past the sliding windows shipped today except
//!   2048, and past the 2048-token prefill chunk).
//! - `EXACTNESS_SURVEY_PATH`: `family` (the probe the model's own gate
//!   runs), `engine` (the forward the token-only verify round runs, prompt
//!   lookup's), or `auto` (default: both where they exist).
//! - `EXACTNESS_SURVEY_DIAG=1`: on a divergence, rerun the draw under
//!   [`diag`] capture and print the per-stage report (the model must place
//!   `diag::capture` calls for this to print anything).
//!
//! Every verdict is printed as a tab-separated `SURVEY` line: checkpoint,
//! path, width, prompt length (`startup` is the gate's own plan), quantized
//! kernel (`wide`, or `narrow` for the gate's `qmv_wide` retry) and verdict.

use std::path::Path;

use mlxcel_core::cache::DecodeLookaheadAppendScope;
use mlxcel_core::generate::LanguageModel;

use super::speculative_exactness::{BlockChainExactness, compare_block_against_chain};
use super::speculative_exactness_diag as diag;
use super::speculative_exactness_draws as draws;
use crate::LoadedModel;

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

/// The family probe of `model`: `None` lengths run the gate's own plan.
fn family_probe(
    model: &LoadedModel,
    width: usize,
    lengths: Option<&[usize]>,
) -> Option<BlockChainExactness> {
    macro_rules! run {
        ($target:expr) => {
            Some(match lengths {
                Some(lengths) => $target.probe_block_chain_exactness_at(width, lengths),
                None => $target.probe_block_chain_exactness(width),
            })
        };
    }
    match model {
        LoadedModel::Qwen35(m) | LoadedModel::Qwen35Moe(m) => run!(m),
        LoadedModel::Qwen35VLM(v) | LoadedModel::Qwen35MoeVLM(v) => run!(v.text_model),
        LoadedModel::Qwen4ExpVLM(v) => run!(v.text_model),
        LoadedModel::Gemma4(m) => run!(m),
        LoadedModel::Gemma4VLM(v) => run!(v.text_model),
        LoadedModel::Gemma4Unified(v) => run!(v.text_model),
        LoadedModel::Glm4MoeLite(m) => run!(m),
        LoadedModel::Inkling(m) => run!(m),
        LoadedModel::InklingVLM(v) => run!(v.text),
        LoadedModel::Lfm2(m) => run!(m),
        LoadedModel::Lfm2VL(v) => run!(v.text_model),
        LoadedModel::Laguna(w) => run!(w.model),
        // Inkling 2 keeps its own plan (owned by #3410): startup verdict only.
        LoadedModel::Inkling2VLM(v) => Some(match lengths {
            None => v.text.probe_block_chain_exactness(width),
            Some(_) => BlockChainExactness::NotRun("Inkling 2 runs its own plan only"),
        }),
        _ => None,
    }
}

/// Block versus chain through the plain `LanguageModel` forward on external
/// KV caches, which is what the engine's token-only verify round runs (prompt
/// lookup, a full-model drafter). `NotRun` for a family that keeps sequence
/// state outside those caches, except first-generation Inkling, whose verify
/// rounds rewind its own state (issue #3422) and whose `make_caches` resets
/// the default state both arms run on.
fn engine_probe(
    model: &LoadedModel,
    width: usize,
    draw: usize,
    prompt_len: usize,
    trace: bool,
) -> (BlockChainExactness, Option<Vec<String>>) {
    let own_state_probed = matches!(model, LoadedModel::Inkling(_) | LoadedModel::InklingVLM(_));
    if (model
        .sequence_state_layout()
        .backend
        .keeps_state_outside_kv_slice()
        && !own_state_probed)
        || mlxcel_core::speculative::prompt_lookup::prompt_lookup_unsupported_reason(model)
            .is_some()
    {
        return (
            BlockChainExactness::NotRun("the token-only verify round does not take this model"),
            None,
        );
    }
    let vocab = {
        let mut caches = model.make_caches();
        let logits = model.forward(&draws::as_input(&[0]), &mut caches, None);
        *mlxcel_core::array_shape(&logits).last().expect("rank 3") as usize
    };
    let prompt = draws::prompt_tokens(draw, prompt_len, vocab);
    let block = draws::block_tokens(draw, width, vocab);
    let chunk = mlxcel_core::generate::prefill_chunk_len().max(1);
    let prefill = |caches: &mut Vec<mlxcel_core::layers::KVCache>| {
        for part in prompt.chunks(chunk) {
            let logits = model.forward(&draws::as_input(part), caches, None);
            mlxcel_core::eval(&logits);
        }
        model.after_prefill();
    };
    let traced = |run: &mut dyn FnMut() -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray>| {
        if trace {
            diag::with_capture(run)
        } else {
            (run(), Vec::new())
        }
    };

    let mut chain_caches = model.make_caches();
    prefill(&mut chain_caches);
    let mut chain_positions = Vec::with_capacity(width);
    let mut chain_trace = Vec::with_capacity(width);
    for &token in &block {
        let (logits, stages) =
            traced(&mut || model.forward(&draws::as_input(&[token]), &mut chain_caches, None));
        chain_positions.push(draws::position_bytes(&logits, 0));
        chain_trace.push(stages);
    }

    let mut block_caches = model.make_caches();
    prefill(&mut block_caches);
    let (logits, block_trace) = traced(&mut || {
        let _speculative = DecodeLookaheadAppendScope::enter();
        model
            .try_forward_with_sequence_id(&draws::as_input(&block), None, &mut block_caches, None)
            .expect("verify forward")
    });
    let block_positions: Vec<Vec<u8>> = (0..width)
        .map(|i| draws::position_bytes(&logits, i as i32))
        .collect();

    let verdict = compare_block_against_chain(&block_positions, &chain_positions);
    let report = trace.then(|| {
        let mut lines = diag::report(&block_trace, &chain_trace);
        lines.push(match diag::locate(&block_trace, &chain_trace) {
            Ok(Some(found)) => found.to_string(),
            Ok(None) => "no captured stage differs".to_owned(),
            Err(why) => format!("no localization: {why}"),
        });
        lines
    });
    (verdict, report)
}

fn verdict_text(verdict: &BlockChainExactness) -> String {
    match verdict {
        BlockChainExactness::Equal => "equal".to_owned(),
        other => other.reason(),
    }
}

#[test]
#[ignore = "needs a real checkpoint and the GPU lock"]
fn survey_block_vs_chain() {
    let Ok(path) = std::env::var("EXACTNESS_SURVEY_MODEL") else {
        eprintln!("skipping: EXACTNESS_SURVEY_MODEL is not set");
        return;
    };
    if !Path::new(&path).join("config.json").exists() {
        eprintln!("skipping: {path} is not present");
        return;
    }
    let widths = env_list("EXACTNESS_SURVEY_WIDTHS", &[2, 4, 8]);
    let prompts = env_list("EXACTNESS_SURVEY_PROMPTS", &[600, 2100]);
    let which = std::env::var("EXACTNESS_SURVEY_PATH").unwrap_or_else(|_| "auto".into());
    let want_diag = std::env::var("EXACTNESS_SURVEY_DIAG").is_ok_and(|v| v == "1");
    let (model, _tokenizer) = crate::load_model(Path::new(&path)).expect("load checkpoint");
    let name = Path::new(&path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or(path.clone());
    let line = |route: &str, width: usize, prompt: &str, kernel: &str, verdict: &str| {
        println!("SURVEY\t{name}\t{route}\t{width}\t{prompt}\t{kernel}\t{verdict}");
    };
    // A diverging draw is measured again on the narrow quantized kernel, the
    // gate's own retry; the switch is restored so every row starts wide.
    let both_kernels = |run: &dyn Fn() -> BlockChainExactness| {
        let wide = run();
        let retry = !wide.is_equal()
            && !matches!(wide, BlockChainExactness::NotRun(_))
            && mlxcel_core::qmv_wide_enabled();
        let narrow = retry.then(|| {
            mlxcel_core::set_qmv_wide(false);
            let narrow = run();
            mlxcel_core::set_qmv_wide(true);
            narrow
        });
        (wide, narrow)
    };

    for &width in &widths {
        if which != "engine" && family_probe(&model, width, Some(&[])).is_some() {
            let mut rows: Vec<(String, Option<Vec<usize>>)> = vec![("startup".to_owned(), None)];
            rows.extend(prompts.iter().map(|&n| (n.to_string(), Some(vec![n]))));
            for (label, lengths) in rows {
                let (wide, narrow) = both_kernels(&|| {
                    family_probe(&model, width, lengths.as_deref()).expect("family probe")
                });
                line("family", width, &label, "wide", &verdict_text(&wide));
                if let Some(narrow) = narrow {
                    line("family", width, &label, "narrow", &verdict_text(&narrow));
                }
            }
        }
        if which != "family" {
            let mut lengths = vec![8];
            lengths.extend(&prompts);
            for (draw, &prompt_len) in lengths.iter().enumerate() {
                let (wide, narrow) =
                    both_kernels(&|| engine_probe(&model, width, draw, prompt_len, false).0);
                let label = prompt_len.to_string();
                line("engine", width, &label, "wide", &verdict_text(&wide));
                if let Some(narrow) = &narrow {
                    line("engine", width, &label, "narrow", &verdict_text(narrow));
                }
                let declined = !narrow.as_ref().unwrap_or(&wide).is_equal();
                if want_diag && declined && !matches!(wide, BlockChainExactness::NotRun(_)) {
                    if narrow.is_some() {
                        mlxcel_core::set_qmv_wide(false);
                    }
                    let (_, report) = engine_probe(&model, width, draw, prompt_len, true);
                    mlxcel_core::set_qmv_wide(true);
                    for text in report.unwrap_or_default() {
                        println!("DIAG\t{name}\tengine\t{width}\t{label}\t{text}");
                    }
                }
            }
        }
    }
}
