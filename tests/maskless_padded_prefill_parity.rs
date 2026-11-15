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

//! Gates for the families #3072 opted into
//! `LanguageModel::supports_maskless_padded_prefill`.
//!
//! A tile-aligned padded prefill appends zero tokens after the real prompt.
//! A family that opts in gets no padding mask for such a piece, so its own
//! causal forward has to hide the pad keys from every real row. Each gate
//! prefills a real prompt three ways and compares the logits at the last real
//! position:
//!
//! * unpadded, the reference;
//! * padded with no mask, the opted-in path;
//! * padded with `create_padded_prefill_mask`, the path the family took
//!   before opting in.
//!
//! A pad key leaking into a real row moves its logits by far more than the
//! bound, which only absorbs the kernel difference between an `n`-row and an
//! `n + pad`-row forward (a different quantized-matmul width).
//!
//! The checkpoints come from `MLXCEL_MODELS_DIR` (default `models`), and each
//! gate skips when its checkpoint is absent. The Mistral 4 gate also needs
//! `MLXCEL_TEST_MISTRAL4=1`: its only checkpoint is a 119B MoE.

use mlxcel::models::{Cohere2Model, Gemma3nModel, MiniCPM3Model, Mistral4Model, OLMo3Model};
use mlxcel_core::generate::LanguageModel;
use std::path::PathBuf;

/// Real prompt length and its tile-aligned padded length.
const REAL: usize = 40;
const PADDED: usize = 64;

/// Relative RMS bound on the last real row between the three prefills.
const BOUND: f64 = 2e-2;

fn checkpoint(name: &str) -> Option<PathBuf> {
    let root = std::env::var("MLXCEL_MODELS_DIR").unwrap_or_else(|_| "models".to_string());
    let dir = PathBuf::from(root).join(name);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!(
            "skipping maskless padded prefill gate: {} not present",
            dir.display()
        );
        None
    }
}

/// The `[vocab]` logit row at position `pos`, as host floats.
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
    let (mut diff, mut base) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        diff += (x - y) * (x - y);
        base += y * y;
    }
    if base == 0.0 {
        diff.sqrt()
    } else {
        (diff / base).sqrt()
    }
}

fn last_real_row<M: LanguageModel>(
    model: &M,
    tokens: &[i32],
    mask: Option<&mlxcel_core::MlxArray>,
) -> Vec<f32> {
    let mut caches = LanguageModel::make_caches(model);
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let logits = LanguageModel::forward(model, &input, &mut caches, mask);
    logit_row(&logits, REAL as i32 - 1)
}

fn assert_maskless_padded_prefill_matches<M: LanguageModel>(model: &M, who: &str) {
    assert!(
        model.supports_maskless_padded_prefill(),
        "{who} must opt into maskless padded prefill"
    );
    let real: Vec<i32> = (0..REAL as i32).map(|i| 300 + (i * 137) % 4000).collect();
    let mut padded = real.clone();
    padded.resize(PADDED, 0);

    let reference = last_real_row(model, &real, None);
    let maskless = last_real_row(model, &padded, None);
    let mask = mlxcel_core::utils::create_padded_prefill_mask(REAL as i32, PADDED as i32, 0);
    let masked = last_real_row(model, &padded, Some(&mask));

    let maskless_rms = relative_rms(&maskless, &reference);
    let masked_rms = relative_rms(&masked, &reference);
    eprintln!("{who}: maskless {maskless_rms:.3e}, padding mask {masked_rms:.3e}");
    assert!(
        maskless_rms < BOUND,
        "{who}: a maskless padded prefill moved the last real row by {maskless_rms:.3e}"
    );
    assert!(
        masked_rms < BOUND,
        "{who}: the padding-mask prefill moved the last real row by {masked_rms:.3e}"
    );
}

#[test]
fn cohere2_maskless_padded_prefill_matches_unpadded() {
    let Some(dir) = checkpoint("c4ai-command-r7b-12-2024-4bit") else {
        return;
    };
    let (model, _) = Cohere2Model::load(&dir).expect("load cohere2");
    assert_maskless_padded_prefill_matches(&model, "cohere2");
}

#[test]
fn olmo3_maskless_padded_prefill_matches_unpadded() {
    let Some(dir) = checkpoint("olmo-3.1-32b-think-4bit") else {
        return;
    };
    let (model, _) = OLMo3Model::load(&dir).expect("load olmo3");
    assert_maskless_padded_prefill_matches(&model, "olmo3");
}

#[test]
fn minicpm3_maskless_padded_prefill_matches_unpadded() {
    let Some(dir) = checkpoint("minicpm3-4b-4bit") else {
        return;
    };
    let (model, _) = MiniCPM3Model::load(&dir).expect("load minicpm3");
    assert_maskless_padded_prefill_matches(&model, "minicpm3");
}

#[test]
fn gemma3n_maskless_padded_prefill_matches_unpadded() {
    let Some(dir) = checkpoint("gemma-3n-e2b-it-4bit") else {
        return;
    };
    let model = Gemma3nModel::load(&dir).expect("load gemma3n");
    assert_maskless_padded_prefill_matches(&model, "gemma3n");
}

#[test]
fn mistral4_maskless_padded_prefill_matches_unpadded() {
    if std::env::var("MLXCEL_TEST_MISTRAL4").as_deref() != Ok("1") {
        eprintln!("skipping the Mistral 4 gate: set MLXCEL_TEST_MISTRAL4=1");
        return;
    }
    let Some(dir) = checkpoint("mistral-small-4-119b-2603-4bit") else {
        return;
    };
    // The local checkpoint is a VLM wrapper; its decoder is `text_config`.
    let (model, _) = Mistral4Model::load_from_text_config(&dir).expect("load mistral4");
    assert_maskless_padded_prefill_matches(&model, "mistral4");
}
