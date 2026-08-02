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

//! DeepSeek-V2 greedy-decode parity against the mlx-lm reference.
//!
//! Closes the coverage gap that let issue #991 ship: nothing in the tree ever
//! generated from a real `deepseek_v2` text checkpoint, so a fully
//! bidirectional prefill (the model handed the caller's `None` mask straight to
//! the unmasked SDPA instead of applying its own causal mask) passed every unit
//! test and every CI job while producing `}'_` for every generated token.
//!
//! Two independent gates:
//!
//! * `deepseek_v2_greedy_parity_matches_mlx_lm` pins the generated ids against
//!   `mlx_lm` 0.31.3 greedy decode on `models/deepseek-v2-lite-4bit`.
//! * `deepseek_v2_prefill_is_causal` pins the property the bug violated,
//!   without depending on any reference ids: the first prompt position's logits
//!   must not move when later prompt tokens are added. A bidirectional prefill
//!   fails it by a wide margin, so it stays meaningful for any checkpoint of
//!   this family.
//!
//! Both skip when the checkpoint is absent (CI has no Metal and no weights).

use mlxcel::models::DeepSeekV2Model;
use mlxcel_core::generate::LanguageModel;

const MODEL_DIR: &str = "models/deepseek-v2-lite-4bit";

// tokenizer.encode("The capital of France is") with the BOS the checkpoint's
// tokenizer adds, which is exactly what mlx-lm feeds prefill.
const INPUT_IDS: &[i32] = &[100000, 549, 6077, 280, 7239, 317];

// mlx-lm 0.31.3 greedy (temp 0) continuation:
// " Paris.\nThe official language of France is French.\nThe currency of France
// is the Euro ("
const REF_GREEDY_OUT: &[i32] = &[
    8913, 13, 185, 549, 6269, 4706, 280, 7239, 317, 6016, 13, 185, 549, 19305, 280, 7239, 317, 254,
    28071, 334,
];

fn argmax_last_token(logits: &mlxcel_core::MlxArray) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let last = mlxcel_core::slice(logits, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
    let argmax = mlxcel_core::argmax_last_axis(&last);
    mlxcel_core::eval(&argmax);
    mlxcel_core::item_i32(&argmax)
}

/// The `[1, vocab]` logit row at prompt position `pos`, as host floats.
fn logit_row(logits: &mlxcel_core::MlxArray, pos: i32) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(logits);
    let row = mlxcel_core::slice(logits, &[0, pos, 0], &[1, pos + 1, shape[2]]);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    mlxcel_core::array_to_raw_bytes(&row)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// RMS of `a - b` divided by the RMS of `b`.
fn relative_rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "logit row length mismatch");
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        diff_sq += (x - y) * (x - y);
        ref_sq += y * y;
    }
    if ref_sq == 0.0 {
        return diff_sq.sqrt();
    }
    (diff_sq / ref_sq).sqrt()
}

fn skip_if_absent() -> bool {
    if std::path::Path::new(MODEL_DIR).exists() {
        return false;
    }
    eprintln!("skipping deepseek_v2 parity: {MODEL_DIR} not present");
    true
}

#[test]
fn deepseek_v2_greedy_parity_matches_mlx_lm() {
    if skip_if_absent() {
        return;
    }

    let (model, _args) = DeepSeekV2Model::load(MODEL_DIR).expect("load deepseek_v2");
    let mut caches = LanguageModel::make_caches(&model);

    let prompt = mlxcel_core::from_slice_i32(INPUT_IDS, &[1, INPUT_IDS.len() as i32]);
    let mut logits = LanguageModel::forward(&model, &prompt, &mut caches, None);

    let mut out = Vec::with_capacity(REF_GREEDY_OUT.len());
    for _ in 0..REF_GREEDY_OUT.len() {
        let tok = argmax_last_token(&logits);
        out.push(tok);
        let next = mlxcel_core::from_slice_i32(&[tok], &[1, 1]);
        logits = LanguageModel::forward(&model, &next, &mut caches, None);
    }

    assert_eq!(
        out, REF_GREEDY_OUT,
        "DeepSeek-V2 greedy decode diverged from the mlx-lm reference"
    );
}

#[test]
fn deepseek_v2_prefill_is_causal() {
    if skip_if_absent() {
        return;
    }

    let (model, _args) = DeepSeekV2Model::load(MODEL_DIR).expect("load deepseek_v2");

    // Full prompt in one prefill, then the same first token on its own. Under a
    // causal prefill position 0 sees only itself in both runs, so the two logit
    // rows agree to f16 rounding. Under the bidirectional prefill that shipped
    // before issue #991, position 0 also attended to the five later tokens and
    // the rows diverged by tens of percent.
    let mut caches_full = LanguageModel::make_caches(&model);
    let prompt = mlxcel_core::from_slice_i32(INPUT_IDS, &[1, INPUT_IDS.len() as i32]);
    let full = LanguageModel::forward(&model, &prompt, &mut caches_full, None);
    let row_full = logit_row(&full, 0);

    let mut caches_single = LanguageModel::make_caches(&model);
    let first = mlxcel_core::from_slice_i32(&INPUT_IDS[..1], &[1, 1]);
    let single = LanguageModel::forward(&model, &first, &mut caches_single, None);
    let row_single = logit_row(&single, 0);

    let deviation = relative_rms(&row_full, &row_single);
    assert!(
        deviation < 2e-2,
        "DeepSeek-V2 prefill is not causal: position-0 logits moved by {deviation:.4e} \
         relative RMS when five later prompt tokens were added"
    );
}
