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

//! Decode stays finite on the f16 families whose attention arithmetic is
//! widened to float32 (issue #1830, following #1710).
//!
//! `phi-2-4bit` ran its `q @ k^T` in f16 for months. The products pass 65504 at
//! layer 29 of 32, the softmax turns the resulting inf into NaN, and every token
//! after that is the argmax of a NaN row and decodes as `!`. Nothing in the tree
//! failed. Layers 0 through 29 tracked the reference to three decimal places
//! first (`src/models/phixtral.rs`), so a parity check that samples an early
//! layer or a short prompt passes right up to the layer that overflows. #1709
//! additionally masked it: `gelu_approx` returned f32 for a half-precision
//! input, which widened the residual stream from the first MLP onward and held
//! the scores in range by accident. Fixing that exposed it.
//!
//! So the guard these tests protect is the `astype(&q, FLOAT32)` line in
//! `src/models/phi.rs`, and its counterparts in `src/models/stablelm.rs` and
//! `src/models/phixtral.rs`. Delete one and the corresponding test here must go
//! red. That is the only thing that makes a green run mean anything; #1830's
//! acceptance criteria require the experiment to be run and reported.
//!
//! **Two failure shapes, and on this prompt it is the second one.** Deleting
//! phi's widen and running this file was measured on 2026-09-14: every step's
//! max logit stayed finite and argmax returned token 0 at all 33 steps. So the
//! finiteness assertion alone would have passed a model producing nothing but
//! `!`. `max_all` does propagate NaN (checked directly: a row containing one
//! reduces to NaN, and `is_finite` is false), so the finiteness check is sound
//! for the NaN shape #1710 described; it is simply not the shape this prompt
//! reaches. Both checks are therefore kept, and the collapse check runs inside
//! the decode loop so its failure names a step.
//!
//! **The threshold is a property of f16, not of attention.** bfloat16 carries
//! float32's exponent range, so 65504 is the wrong number for a bf16 checkpoint
//! and these tests assert finiteness rather than a 65504 bound. Do not extend
//! the bound to a bf16 family. `mlxcel_core::layers::max_abs_attention_score`
//! and `F16_MAX` exist for measuring the headroom itself; they are diagnostics
//! and nothing on a decode path may call them.
//!
//! Run with the checkpoints forced to resolve, so a local pass cannot come from
//! skipping everything:
//!
//! ```text
//! MLXCEL_REQUIRE_MODELS=1 cargo test --profile test-fast \
//!     --features metal,accelerate --test attention_f16_range -- \
//!     --test-threads=1 --nocapture
//! ```

mod common;
use common::repo_model_dir;

/// Decode steps taken after the prefill.
///
/// #1710's overflow appeared at a specific layer rather than a specific step, so
/// the prefill alone would surface it. The decode steps are here because the
/// per-step KV growth is what changes the score magnitudes as generation runs,
/// and because a poisoned row propagates: once one step is NaN every later one
/// is too, and the step index in the failure names which.
const DECODE_STEPS: usize = 32;

/// How many identical consecutive tokens count as a collapsed distribution.
///
/// Deleting phi's widen does not produce a NaN row on this prompt; it produces a
/// finite row that argmax resolves to the same token forever. Eight is short
/// enough to name the step where the collapse began and long enough that
/// ordinary repetition (a run of newlines, a repeated word in a list) does not
/// trip it: the three families here produce 24 to 28 distinct tokens in 33 steps
/// with their guards present.
const COLLAPSE_RUN: usize = 8;

/// The prompt is fixed so a failure is reproducible from the test name alone.
const PROMPT: &str = "The capital of France is";

/// Prefill `PROMPT`, decode greedily, and assert every step's sampled row is
/// finite.
///
/// The decode loop is written out rather than run through `CxxGenerator`
/// because the assertion has to happen after *every* step. A generator returns
/// the finished token list, by which point a NaN at step 3 is indistinguishable
/// from a model that simply produced odd text.
fn decode_stays_finite(model_name: &str) {
    let dir = repo_model_dir(model_name);
    if !dir.join("config.json").exists() {
        // Loud, with the resolved path: a silent skip is how the NaN guard from
        // #1718 came to cover nothing on either machine (`tests/common/mod.rs`).
        eprintln!(
            "SKIPPED {model_name}: no checkpoint at {}. This test verified nothing. \
             Set MLXCEL_REQUIRE_MODELS=1 to make an unresolvable name fail instead.",
            dir.display()
        );
        return;
    }

    let _runtime = mlxcel::initialize_runtime();
    let (model, tokenizer) =
        mlxcel::load_model(&dir).unwrap_or_else(|e| panic!("{model_name} must load: {e}"));

    let prompt_ids: Vec<i32> = tokenizer
        .encode(PROMPT, true)
        .expect("tokenize prompt")
        .iter()
        .map(|&t| t as i32)
        .collect();
    assert!(
        !prompt_ids.is_empty(),
        "{model_name}: the prompt tokenized to nothing"
    );

    let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
    let mut input = mlxcel_core::from_slice_i32(&prompt_ids, &[1, prompt_ids.len() as i32]);
    let mut sampled: Vec<i32> = Vec::with_capacity(DECODE_STEPS);
    let mut peaks: Vec<f32> = Vec::with_capacity(DECODE_STEPS);

    for step in 0..=DECODE_STEPS {
        let logits =
            mlxcel_core::generate::LanguageModel::forward(&model, &input, &mut caches, None);
        mlxcel_core::eval(&logits);

        let shape = mlxcel_core::array_shape(&logits);
        let vocab = *shape.last().expect("logits have a trailing vocab axis");
        let last = shape[1] - 1;
        let row = mlxcel_core::slice(&logits, &[0, last, 0], &[1, last + 1, vocab]);

        let max = mlxcel_core::max_all(&row);
        mlxcel_core::eval(&max);
        let peak = mlxcel_core::item_f32(&max);
        assert!(
            peak.is_finite(),
            "{model_name}: step {step} (prefill is step 0) produced a max logit of {peak} \
             (nan={}, inf={}). The f32 widen on this family's attention scores is the guard \
             for this; check that it is still present.",
            peak.is_nan(),
            peak.is_infinite()
        );

        let next = mlxcel_core::argmax(&row, -1, false);
        mlxcel_core::eval(&next);
        let token = mlxcel_core::item_i32(&next);
        sampled.push(token);
        peaks.push(peak);

        // The other shape the failure takes, and the one actually observed when
        // the phi widen is deleted on this prompt: the row stays finite and
        // collapses onto a single token. Checked inside the loop, not after it,
        // so the failure names the step where the run began rather than only
        // reporting that the whole generation was degenerate.
        if sampled.len() > COLLAPSE_RUN {
            let tail = &sampled[sampled.len() - COLLAPSE_RUN..];
            if tail.iter().all(|&t| t == token) {
                let first = step + 1 - COLLAPSE_RUN;
                let tail_peaks = &peaks[peaks.len() - COLLAPSE_RUN..];
                panic!(
                    "{model_name}: steps {first}..={step} all sampled token {token}, with max \
                     logits {tail_peaks:?}. The row is finite, so this is a collapsed logit \
                     distribution rather than a NaN one; the f32 widen on this family's \
                     attention scores is the guard for it."
                );
            }
        }

        input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
    }

    let distinct = sampled
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();

    let text = tokenizer
        .decode(&sampled.iter().map(|&t| t as u32).collect::<Vec<_>>(), true)
        .expect("decode generation");
    let steps = sampled.len();
    eprintln!("[{model_name}] {steps} steps, {distinct} distinct tokens: {text:?}");
}

// One test function per checkpoint. `tests/mamba2_hybrid_finite.rs` records why:
// running another checkpoint first changed the outcome there, and a combined
// test passed without the fix and guarded nothing.
//
// Not `#[ignore]`d, unlike the M5-only SSM tests next door. #1710's overflow
// reproduces on any Apple Silicon host and in bare MLX, so the only reason to
// skip is a host without the checkpoint, which the loud skip above covers.

#[test]
fn phi_attention_scores_stay_in_range() {
    decode_stays_finite("phi-2-hf-4bit-mlx");
}

#[test]
fn stablelm_attention_scores_stay_in_range() {
    decode_stays_finite("stablelm-2-1_6b-chat-4bit");
}

#[test]
fn phixtral_attention_scores_stay_in_range() {
    decode_stays_finite("phixtral-4x2_8-4bit");
}
