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

//! Prefill memory of the Qwen 3.5 family (issue #3045).
//!
//! The two unit tests guard the last-row LM head projection: the prefill
//! entry points return `[1, 1, vocab]` logits equal to the sampled row of the
//! full forward, and never materialize the `[1, seq_len, vocab]` tensor.
//!
//! The ignored probe attributes a long single-pass prefill's MLX peak to its
//! stages on a real checkpoint, evaluating after each stage:
//!
//! ```text
//! MLXCEL_PREFILL_PROBE_MODEL=models/qwen3.5-4b-4bit MLXCEL_PREFILL_PROBE_LEN=28000 \
//! cargo test --profile test-fast --features metal,accelerate -p mlxcel --lib -- --ignored \
//!   --nocapture models::qwen3_5::qwen3_5_prefill_memory_tests::prefill_stage_peak_memory_probe
//! ```
//!
//! Results: `docs/benchmark_results/qwen35-prefill-peak-m3ultra-2026-11-15.md`.

use super::Qwen35Model;
use crate::models::qwen3_5_tests::two_layer_model_with_vocab;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::memory::{active_memory, peak_memory, reset_peak_memory};
use mlxcel_core::{MlxArray, UniquePtr};

fn ids(len: usize, vocab: usize) -> UniquePtr<MlxArray> {
    let tokens: Vec<i32> = (0..len).map(|i| ((i * 7 + 3) % vocab) as i32).collect();
    mlxcel_core::from_slice_i32(&tokens, &[1, len as i32])
}

fn assert_close(actual: &MlxArray, expected: &MlxArray) {
    let close = mlxcel_core::allclose(actual, expected, 1e-4, 1e-4);
    mlxcel_core::eval(&close);
    assert!(
        mlxcel_core::item_bool(&close),
        "logits differ beyond tolerance"
    );
}

#[test]
fn prefill_last_logits_are_the_sampled_row_of_the_full_forward() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let vocab = 8;
    let input = ids(11, vocab);
    let full_model = two_layer_model_with_vocab(vocab);
    let full = LanguageModel::forward(&full_model, &input, &mut [], None);
    let expected = mlxcel_core::generate::logits_at_position(&full, 10);

    let model = two_layer_model_with_vocab(vocab);
    let last = model.forward_last_logits(&input, &mut [], None, 10);
    assert_eq!(mlxcel_core::array_shape(&last), vec![1, 1, vocab as i32]);
    assert_close(&last, &expected);

    // The scheduler's per-sequence entry point, mid-piece position included.
    let seq = SequenceId::from_raw(304_500);
    let model = two_layer_model_with_vocab(vocab);
    model.prepare_sequence_state(seq);
    let mid = model.forward_last_logits_with_sequence_id(&input, Some(seq), &mut [], None, 4);
    assert_eq!(mlxcel_core::array_shape(&mid), vec![1, 1, vocab as i32]);
    assert_close(&mid, &mlxcel_core::generate::logits_at_position(&full, 4));

    // A chunked prefill continues the same state: the second piece's last
    // row matches the single forward's.
    let model = two_layer_model_with_vocab(vocab);
    let head = mlxcel_core::slice(&input, &[0, 0], &[1, 6]);
    let tail = mlxcel_core::slice(&input, &[0, 6], &[1, 11]);
    let first = model.forward_last_logits(&head, &mut [], None, 5);
    mlxcel_core::eval(&first);
    let second = model.forward_last_logits(&tail, &mut [], None, 4);
    assert_close(&second, &expected);
}

/// The regression guard for #3045: a prefill's peak stays far below the
/// full-sequence logits it used to materialize. At 1024 positions and a
/// 131072-entry vocabulary those are 512 MiB in f32; the rest of this
/// model's prefill (mostly `L x L` attention and gated-delta intermediates)
/// stays near 32 MiB.
#[test]
fn prefill_last_logits_never_materialize_the_full_sequence_logits() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let (vocab, len) = (131_072_usize, 1024_usize);
    let full_bytes = (vocab * len * 4) as u64;
    let model = two_layer_model_with_vocab(vocab);
    let input = ids(len, vocab);
    mlxcel_core::eval(&input);
    mlxcel_core::clear_memory_cache();
    let before = active_memory();
    reset_peak_memory();
    let last = model.forward_last_logits(&input, &mut [], None, len - 1);
    mlxcel_core::eval(&last);
    let transient = peak_memory().saturating_sub(before);
    assert_eq!(mlxcel_core::array_shape(&last), vec![1, 1, vocab as i32]);
    assert!(
        transient < full_bytes / 4,
        "prefill peaked {transient} bytes above its start, the full logits are {full_bytes}"
    );
}

/// One stage's MLX memory: the peak above the active bytes before it, and
/// what it left resident.
struct Stage {
    name: String,
    transient: u64,
    retained: i64,
}

fn measure<T>(name: impl Into<String>, run: impl FnOnce() -> T, eval: impl Fn(&T)) -> (T, Stage) {
    let before = active_memory();
    reset_peak_memory();
    let out = run();
    eval(&out);
    let stage = Stage {
        name: name.into(),
        transient: peak_memory().saturating_sub(before),
        retained: active_memory() as i64 - before as i64,
    };
    (out, stage)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn text_model(model: &crate::LoadedModel) -> &Qwen35Model {
    match model {
        crate::LoadedModel::Qwen35(m) => m,
        crate::LoadedModel::Qwen35VLM(vlm) => &vlm.text_model,
        _ => panic!("MLXCEL_PREFILL_PROBE_MODEL is not a Qwen 3.5 checkpoint"),
    }
}

/// Per-stage attribution of a single-pass prefill (#3045), then the
/// end-to-end lazy prefill through the served entry point with and without
/// the last-row projection.
#[test]
#[ignore = "needs a GPU and a Qwen 3.5 checkpoint in MLXCEL_PREFILL_PROBE_MODEL"]
fn prefill_stage_peak_memory_probe() {
    let Ok(dir) = std::env::var("MLXCEL_PREFILL_PROBE_MODEL") else {
        eprintln!("[3045] skipping: MLXCEL_PREFILL_PROBE_MODEL is unset");
        return;
    };
    let len: usize = std::env::var("MLXCEL_PREFILL_PROBE_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(28_000);
    let (loaded, _tokenizer) = crate::backend::select_backend()
        .load_model(std::path::Path::new(&dir))
        .expect("probe model must load");
    let model = text_model(&loaded);
    let vocab = model.config.vocab_size;
    let input = ids(len, vocab);
    mlxcel_core::eval(&input);
    mlxcel_core::clear_memory_cache();
    let base = active_memory();
    println!(
        "[3045] model={dir} prompt={len} vocab={vocab} resident_before={:.2} GiB",
        gib(base)
    );

    let mut stages = Vec::new();
    let (mut h, stage) = measure(
        "embedding",
        || model.embed_tokens.forward(&input),
        |h| mlxcel_core::eval(h),
    );
    stages.push(stage);

    let mut caches = model.make_internal_caches();
    let (mask, stage) = measure(
        "causal mask",
        || mlxcel_core::utils::create_causal_mask(len as i32, 0),
        |m| mlxcel_core::eval(m),
    );
    stages.push(stage);
    let (mut linear_max, mut full_max) = (0u64, 0u64);
    let (mut linear_n, mut full_n) = (0usize, 0usize);
    for (idx, (layer, cache)) in model.layers.iter().zip(caches.iter_mut()).enumerate() {
        let layer_mask = if layer.is_linear { None } else { Some(&*mask) };
        let (out, stage) = measure(
            format!("layer {idx}"),
            || layer.forward(&h, layer_mask, cache, None),
            |o| mlxcel_core::eval(o),
        );
        if layer.is_linear {
            linear_max = linear_max.max(stage.transient);
            linear_n += 1;
        } else {
            full_max = full_max.max(stage.transient);
            full_n += 1;
        }
        h = out;
        mlxcel_core::clear_memory_cache();
    }
    stages.push(Stage {
        name: format!("{linear_n} linear-attention layers (max of one)"),
        transient: linear_max,
        retained: 0,
    });
    stages.push(Stage {
        name: format!("{full_n} full-attention layers (max of one)"),
        transient: full_max,
        retained: 0,
    });
    drop(mask);

    let (normed, stage) = measure(
        "final norm",
        || model.norm.forward(&h),
        |n| mlxcel_core::eval(n),
    );
    stages.push(stage);
    let project = |x: &MlxArray| match &model.lm_head {
        Some(head) => head.forward(x),
        None => model.embed_tokens.as_linear(x),
    };
    let (full_logits, stage) = measure(
        "lm_head, every position",
        || project(&normed),
        |l| mlxcel_core::eval(l),
    );
    stages.push(stage);
    let (token, stage) = measure(
        "sampling (argmax of the last row)",
        || {
            let row = mlxcel_core::generate::logits_at_position(&full_logits, len - 1);
            mlxcel_core::argmax(&mlxcel_core::reshape(&row, &[-1]), -1, false)
        },
        |t| mlxcel_core::eval(t),
    );
    stages.push(stage);
    let full_token = mlxcel_core::item_i32(&token);
    drop(full_logits);
    mlxcel_core::clear_memory_cache();
    let (last_logits, stage) = measure(
        "lm_head, last position only",
        || project(&mlxcel_core::generate::logits_at_position(&normed, len - 1)),
        |l| mlxcel_core::eval(l),
    );
    stages.push(stage);
    let last_token = mlxcel_core::item_i32(&mlxcel_core::argmax(
        &mlxcel_core::reshape(&last_logits, &[-1]),
        -1,
        false,
    ));
    assert_eq!(
        full_token, last_token,
        "the last-row projection picks the same token"
    );

    println!("| Stage | Peak above start (GiB) | Retained (GiB) |");
    println!("|---|---|---|");
    for s in &stages {
        println!(
            "| {} | {:.2} | {:.2} |",
            s.name,
            gib(s.transient),
            s.retained as f64 / (1u64 << 30) as f64
        );
    }
    drop((h, normed, last_logits, caches));
    mlxcel_core::clear_memory_cache();

    // End to end, lazily, through the wrapper the engine calls: the trait
    // default (full logits, then the slice) against the last-row override.
    for (label, last_row) in [
        ("full logits then slice", false),
        ("last-row projection", true),
    ] {
        LanguageModel::reset_runtime_state(&loaded);
        mlxcel_core::clear_memory_cache();
        let before = active_memory();
        reset_peak_memory();
        let logits = if last_row {
            LanguageModel::forward_last_logits_with_sequence_id(
                &loaded,
                &input,
                None,
                &mut [],
                None,
                len - 1,
            )
        } else {
            let full =
                LanguageModel::forward_with_sequence_id(&loaded, &input, None, &mut [], None);
            mlxcel_core::generate::logits_at_position(&full, len - 1)
        };
        mlxcel_core::eval(&logits);
        println!(
            "[3045] single-pass prefill, {label}: peak {:.2} GiB above the weights",
            gib(peak_memory().saturating_sub(before))
        );
    }
}
