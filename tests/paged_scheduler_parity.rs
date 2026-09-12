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

//! Real-model parity for the scheduler-driven paged KV cache (#121).
//!
//! #152 proved a hand-built pool-backed `KVCache` matches a dense cache through
//! `forward`. This test goes one layer up: it drives the **scheduler's** paged
//! allocation + cache path — [`CachePool::allocate_with_layout`] with the exact
//! paged layout the scheduler builds (`sequence_state_layout_override`) — and
//! confirms the resulting pool-backed caches generate the same greedy tokens as
//! the scheduler's dense path (`CachePool::allocate` with the model's natural
//! dense layout).
//!
//! # Scope: the pool-backed model families
//!
//! Pool-backing is gated to **Fp16 dense-natural-backend** sequences. The two
//! such families with small checkpoints are exercised here: **qwen3** and
//! **llama3**. Model-owned-state families (gemma3, llama4, qwen3.5) and
//! quantized-KV (Int8 / Turbo) paths keep the dense backend and are therefore
//! out of scope for this pool-parity test — their decode path is unchanged by
//! #121. Each pool-backed family runs both the single-sequence and the batched
//! parity case below.
//!
//! # Why `CachePool` and not `BatchScheduler`
//!
//! `BatchScheduler::new` is `pub(crate)`, so an integration test (a separate
//! crate) cannot construct one. `CachePool` is the scheduler's
//! cache-management core: `allocate_with_layout(model, paged_override)` is
//! exactly what `BatchScheduler::allocate_sequence_state` calls when
//! `decode_storage_backend == Paged` and `max_batch_size > 1`, and
//! `allocate(model)` is the dense path. Driving the model through the caches
//! `CachePool` hands back therefore faithfully reproduces the scheduler's
//! per-sequence paged path without standing up the async worker loop.
//!
//! # What each parity case covers
//!
//! * [`assert_single_sequence_parity`] — the scheduler's **single-sequence**
//!   path. A lone request decodes via `decode_single_step` (`scheduler.rs`: `if
//!   seq_ids.len() <= 1 { decode_single_step }`), i.e. single-sequence
//!   `model.forward`, whose pool intercept (`update_and_fetch` / `update`)
//!   writes to and gathers from the pool.
//! * [`assert_batched_decode_parity`] — the **batched** decode wiring (`B == 2`)
//!   via `forward_batched_with_context_and_ids` + a paged [`DecodeBatchContext`],
//!   exactly as `execute_batched_decode` dispatches it. For pool-backed caches
//!   this reaches the #899 whole-batch pooled decode helper, including its
//!   gather fallback below the fused dispatch floor; if that helper declines,
//!   the model's `is_paged_backed()` guard still keeps the dense-pointer native
//!   kernel away from placeholder dense buffers.
//!
//! # Running
//!
//! `#[ignore]` (loads a real checkpoint and runs a real GPU forward). Run with:
//!
//! ```text
//! cargo test --test paged_scheduler_parity --release \
//!     --features metal,accelerate -- --ignored --nocapture --test-threads=1
//! MLX_ENABLE_TF32=0 cargo test --test paged_scheduler_parity --release \
//!     --features cuda -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Each case soft-skips when its model directory is absent. Fetch with:
//!
//! ```text
//! ./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit
//! ./target/release/mlxcel download mlx-community/Llama-3.2-1B-Instruct-4bit
//! ```

mod common;
use common::repo_model_dir;

use mlxcel::{DecodeBatchContext, LanguageModel, initialize_runtime, load_model};
use mlxcel_core::cache::{CachePool, PagedKvLayout, SequenceStateLayout};

/// qwen3 checkpoint directory name (pool-backed family).
const QWEN3_DIR: &str = "qwen3-0.6b-4bit";
/// llama3 checkpoint directory name (pool-backed family).
const LLAMA3_DIR: &str = "llama-3.2-1b-instruct-4bit";

/// Paged block size (tokens per physical block) — matches the scheduler's
/// `DEFAULT_PAGED_BLOCK_SIZE`.
const BLOCK_SIZE: usize = 32;

/// Number of greedy decode steps to compare after prefill.
const DECODE_STEPS: usize = 16;

/// Paged-vs-dense single-sequence logit tolerance for the backend contract.
///
/// The #1773 server/CLI reproducer is a near-tie VLM continuation and must not
/// be turned into a Japanese-token fixture here. The backend contract this test
/// pins is narrower: qwen3 and llama3's scheduler-shaped paged path and dense
/// path must choose the same greedy tokens while each compared logit row stays
/// within this relative-RMS envelope. CUDA validation must run with TF32
/// disabled so TF32's 10-bit mantissa does not define the tolerance.
const PAGED_DENSE_MAX_RELATIVE_RMS: f64 = 5e-5;

#[derive(Debug)]
struct DecodeStepTrace {
    token: i32,
    selector_logits: Vec<f32>,
}

/// Fixed prompt token ids (deterministic; no tokenizer needed). Identical bytes
/// feed every run so the comparison is purely about the cache backend. All ids
/// are < 128k, valid for both the qwen3 (~151k) and llama3 (128k) vocabularies.
const PROMPT_TOKENS: &[i32] = &[
    9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911, 697, 7990, 13, 358,
];

/// Greedy argmax over the vocab at sequence position `pos` of a
/// `[batch, seq_len, vocab]` logits tensor, for batch row `row`.
fn greedy_token(logits: &mlxcel_core::MlxArray, row: i32, pos: i32) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let vocab = shape[2];
    let at_pos = mlxcel_core::slice(logits, &[row, pos, 0], &[row + 1, pos + 1, vocab]);
    let flat = mlxcel_core::reshape(&at_pos, &[vocab]);
    mlxcel_core::eval(&flat);
    mlxcel_core::item_i32(&mlxcel_core::argmax_last_axis(&flat))
}

/// The `[1, vocab]` logit row at sequence position `pos`, as host f32 values.
fn logit_row(logits: &mlxcel_core::MlxArray, row: i32, pos: i32) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(logits);
    let vocab = shape[2];
    let row = mlxcel_core::slice(logits, &[row, pos, 0], &[row + 1, pos + 1, vocab]);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    mlxcel_core::array_to_raw_bytes(&row)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// RMS of `actual - reference` divided by the RMS of `reference`.
fn relative_rms(actual: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(
        actual.len(),
        reference.len(),
        "logit rows must have matching widths"
    );
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference.iter()) {
        let actual = f64::from(actual);
        let reference = f64::from(reference);
        let diff = actual - reference;
        diff_sq += diff * diff;
        ref_sq += reference * reference;
    }
    if ref_sq == 0.0 {
        diff_sq.sqrt()
    } else {
        (diff_sq / ref_sq).sqrt()
    }
}

fn tokens(trace: &[DecodeStepTrace]) -> Vec<i32> {
    trace.iter().map(|step| step.token).collect()
}

fn assert_trace_within_relative_rms(
    actual: &[DecodeStepTrace],
    reference: &[DecodeStepTrace],
    max_relative_rms: f64,
    label: &str,
) {
    assert_eq!(
        tokens(actual),
        tokens(reference),
        "{label}: greedy tokens differ between paged and dense backends"
    );
    for (idx, (actual, reference)) in actual.iter().zip(reference.iter()).enumerate() {
        let rms = relative_rms(&actual.selector_logits, &reference.selector_logits);
        assert!(
            rms <= max_relative_rms,
            "{label}: decode step {idx} relative RMS {rms:e} exceeded {max_relative_rms:e}"
        );
    }
}

/// The paged sequence-state layout the scheduler builds for the default Fp16 KV
/// mode (`BatchScheduler::sequence_state_layout_override` → the non-Turbo
/// `PagedKvLayout::uniform` branch with `DEFAULT_PAGED_BLOCK_SIZE`).
fn scheduler_paged_layout(num_layers: usize) -> SequenceStateLayout {
    SequenceStateLayout::paged_kv_cache(
        PagedKvLayout::uniform(num_layers, BLOCK_SIZE, BLOCK_SIZE).expect("valid paged layout"),
    )
}

/// Run prefill + `DECODE_STEPS` greedy decode steps via single-sequence
/// `model.forward` (the scheduler's `execute_full_prefill` + `decode_single_step`
/// path), returning each emitted token with the logit row that selected it. The
/// first trace row therefore compares the prefill terminal logits, and later
/// rows compare the decode logits from the preceding step.
fn run_single_sequence_trace(
    model: &mlxcel::LoadedModel,
    caches: &mut [mlxcel_core::cache::KVCache],
) -> Vec<DecodeStepTrace> {
    let prompt_len = PROMPT_TOKENS.len() as i32;

    let prompt = mlxcel_core::from_slice_i32(PROMPT_TOKENS, &[1, prompt_len]);
    let mask = mlxcel_core::utils::create_causal_mask(prompt_len, 0);
    let prefill_logits = model.forward(&prompt, caches, Some(&mask));
    mlxcel_core::eval(&prefill_logits);

    let mut selector_logits = logit_row(&prefill_logits, 0, prompt_len - 1);
    let mut next = greedy_token(&prefill_logits, 0, prompt_len - 1);

    let mut decoded = Vec::with_capacity(DECODE_STEPS);
    for _ in 0..DECODE_STEPS {
        decoded.push(DecodeStepTrace {
            token: next,
            selector_logits,
        });
        let step_input = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        let logits = model.forward(&step_input, caches, None);
        mlxcel_core::eval(&logits);
        selector_logits = logit_row(&logits, 0, 0);
        next = greedy_token(&logits, 0, 0);
    }
    decoded
}

/// Load `model_dir_name` from the repo's `models/` directory, or return `None`
/// (and print a fetch hint) when it is absent so the case soft-skips.
fn load_or_skip(model_dir_name: &str, fetch_repo: &str) -> Option<mlxcel::LoadedModel> {
    let model_dir = repo_model_dir(model_dir_name);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {model_dir_name}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download {fetch_repo}",
            model_dir.display()
        );
        return None;
    }
    let (model, _tokenizer) =
        load_model(&model_dir).unwrap_or_else(|e| panic!("load {model_dir_name}: {e:?}"));
    Some(model)
}

/// The scheduler's single-sequence paged path must match its dense path: a lone
/// request allocates pool-backed caches (`decode_storage_backend == Paged`),
/// prefills + decodes through them, and produces the same greedy tokens as the
/// dense backend.
fn assert_single_sequence_parity(model: &mlxcel::LoadedModel, label: &str) {
    let num_layers = model.num_layers();
    eprintln!(
        "\n=== scheduler paged vs dense (single sequence): {label} ({num_layers} layers) ==="
    );

    // Dense backend: scheduler `decode_storage_backend == Dense` → no layout
    // override → `CachePool::allocate` with the model's natural dense layout.
    let mut dense_pool = CachePool::new(2);
    let dense_id = dense_pool.allocate(model).expect("dense allocate");
    assert!(
        !dense_pool.get_caches_mut(dense_id).unwrap()[0].is_paged_backed(),
        "dense backend must not pool-back caches"
    );
    let dense_trace =
        run_single_sequence_trace(model, dense_pool.get_caches_mut(dense_id).unwrap());
    let dense_tokens = tokens(&dense_trace);
    eprintln!("dense  decoded: {dense_tokens:?}");

    // Paged backend: scheduler `decode_storage_backend == Paged` +
    // `max_batch_size > 1` → paged layout override → `allocate_with_layout`,
    // which pool-backs the per-layer caches for this dense-natural Fp16 model.
    let mut paged_pool = CachePool::new(2);
    let paged_id = paged_pool
        .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
        .expect("paged allocate");
    assert!(
        paged_pool
            .get_caches_mut(paged_id)
            .unwrap()
            .iter()
            .all(|c| c.is_paged_backed()),
        "paged backend must pool-back every layer cache for a dense-natural Fp16 model"
    );
    let paged_trace =
        run_single_sequence_trace(model, paged_pool.get_caches_mut(paged_id).unwrap());
    let paged_tokens = tokens(&paged_trace);
    eprintln!("paged  decoded: {paged_tokens:?}");

    assert_trace_within_relative_rms(
        &paged_trace,
        &dense_trace,
        PAGED_DENSE_MAX_RELATIVE_RMS,
        label,
    );
    eprintln!(
        "OK: {DECODE_STEPS} single-sequence decode steps identical between paged and dense backends within relative RMS {PAGED_DENSE_MAX_RELATIVE_RMS:e}."
    );
}

/// The batched decode wiring (the #121 `is_paged_backed()` guard) must also
/// match dense. Two identical pool-backed sequences are decoded together via
/// `forward_batched_with_context_and_ids` + a paged `DecodeBatchContext` —
/// exactly the dispatch `execute_batched_decode` performs — and both rows must
/// reproduce the dense single-sequence reference.
fn assert_batched_decode_parity(model: &mlxcel::LoadedModel, label: &str) {
    if !model.supports_paged_decode_backend() {
        eprintln!("Skipping: {label} does not support the paged decode backend");
        return;
    }
    let num_layers = model.num_layers();
    let prompt_len = PROMPT_TOKENS.len() as i32;
    eprintln!(
        "\n=== scheduler paged batched decode (B=2) vs dense: {label} ({num_layers} layers) ==="
    );

    // Dense single-sequence reference.
    let mut dense_pool = CachePool::new(4);
    let dense_id = dense_pool.allocate(model).expect("dense allocate");
    let dense_trace =
        run_single_sequence_trace(model, dense_pool.get_caches_mut(dense_id).unwrap());
    let dense_tokens = tokens(&dense_trace);
    eprintln!("dense  decoded: {dense_tokens:?}");

    // Two pool-backed paged sequences fed the identical prompt.
    let mut paged_pool = CachePool::new(4);
    let id0 = paged_pool
        .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
        .expect("paged allocate 0");
    let id1 = paged_pool
        .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
        .expect("paged allocate 1");

    // Prefill each sequence with the single-sequence forward (as
    // `execute_full_prefill` does), seeding each sequence's pool blocks.
    let prompt = mlxcel_core::from_slice_i32(PROMPT_TOKENS, &[1, prompt_len]);
    let mask = mlxcel_core::utils::create_causal_mask(prompt_len, 0);
    let mut next = [0i32; 2];
    let mut selector_logits = [Vec::new(), Vec::new()];
    for (slot, id) in [id0, id1].into_iter().enumerate() {
        let caches = paged_pool.get_caches_mut(id).unwrap();
        let logits = model.forward(&prompt, caches, Some(&mask));
        mlxcel_core::eval(&logits);
        selector_logits[slot] = logit_row(&logits, 0, prompt_len - 1);
        next[slot] = greedy_token(&logits, 0, prompt_len - 1);
    }

    // Batched greedy decode through the paged batched path.
    let context = DecodeBatchContext::paged_with_native(BLOCK_SIZE as i32, true);
    let mut batched: [Vec<DecodeStepTrace>; 2] = [Vec::new(), Vec::new()];
    for _ in 0..DECODE_STEPS {
        let input = mlxcel_core::from_slice_i32(&[next[0], next[1]], &[2, 1]);
        let logits = {
            let mut batch_caches = paged_pool
                .get_batch_caches_mut(&[id0, id1])
                .expect("batch caches");
            model.forward_batched_with_context_and_ids(
                &input,
                Some(&[id0, id1]),
                &mut batch_caches,
                None,
                Some(&context),
            )
        };
        mlxcel_core::eval(&logits);
        batched[0].push(DecodeStepTrace {
            token: next[0],
            selector_logits: std::mem::take(&mut selector_logits[0]),
        });
        batched[1].push(DecodeStepTrace {
            token: next[1],
            selector_logits: std::mem::take(&mut selector_logits[1]),
        });
        selector_logits[0] = logit_row(&logits, 0, 0);
        selector_logits[1] = logit_row(&logits, 1, 0);
        next[0] = greedy_token(&logits, 0, 0);
        next[1] = greedy_token(&logits, 1, 0);
    }
    eprintln!("paged0 decoded: {:?}", tokens(&batched[0]));
    eprintln!("paged1 decoded: {:?}", tokens(&batched[1]));

    assert_eq!(
        tokens(&batched[0]),
        dense_tokens,
        "batched paged row 0 diverged from dense reference"
    );
    assert_eq!(
        tokens(&batched[1]),
        dense_tokens,
        "batched paged row 1 diverged from dense reference"
    );
    eprintln!(
        "OK: {DECODE_STEPS} batched (B=2) paged decode steps identical to the dense reference."
    );
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs a real GPU forward; run with --ignored"]
fn paged_scheduler_single_sequence_matches_dense_qwen3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(QWEN3_DIR, "mlx-community/Qwen3-0.6B-4bit") else {
        return;
    };
    assert_single_sequence_parity(&model, QWEN3_DIR);
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs a real GPU forward; run with --ignored"]
fn paged_scheduler_batched_decode_matches_dense_qwen3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(QWEN3_DIR, "mlx-community/Qwen3-0.6B-4bit") else {
        return;
    };
    assert_batched_decode_parity(&model, QWEN3_DIR);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs the CUDA paged/dense logit contract"]
fn cuda_paged_scheduler_qwen3_matches_dense_within_logit_contract() {
    let _runtime = initialize_runtime();
    if !mlxcel_core::cuda_is_available() {
        eprintln!("Skipping CUDA paged scheduler contract: CUDA backend is not available");
        return;
    }
    let Some(model) = load_or_skip(QWEN3_DIR, "mlx-community/Qwen3-0.6B-4bit") else {
        return;
    };

    let num_layers = model.num_layers();
    let mut dense_pool = CachePool::new(2);
    let dense_id = dense_pool.allocate(&model).expect("dense allocate");
    let dense_trace =
        run_single_sequence_trace(&model, dense_pool.get_caches_mut(dense_id).unwrap());

    let mut paged_pool = CachePool::new(2);
    let paged_id = paged_pool
        .allocate_with_layout(&model, Some(scheduler_paged_layout(num_layers)))
        .expect("paged allocate");
    let paged_trace =
        run_single_sequence_trace(&model, paged_pool.get_caches_mut(paged_id).unwrap());

    eprintln!("dense decoded: {:?}", tokens(&dense_trace));
    eprintln!("paged decoded: {:?}", tokens(&paged_trace));
    assert_trace_within_relative_rms(
        &paged_trace,
        &dense_trace,
        PAGED_DENSE_MAX_RELATIVE_RMS,
        "CUDA qwen3 paged-vs-dense scheduler contract",
    );
}

#[test]
#[ignore = "loads llama-3.2-1b-4bit and runs a real GPU forward; run with --ignored"]
fn paged_scheduler_single_sequence_matches_dense_llama3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(LLAMA3_DIR, "mlx-community/Llama-3.2-1B-Instruct-4bit") else {
        return;
    };
    assert_single_sequence_parity(&model, LLAMA3_DIR);
}

#[test]
#[ignore = "loads llama-3.2-1b-4bit and runs a real GPU forward; run with --ignored"]
fn paged_scheduler_batched_decode_matches_dense_llama3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(LLAMA3_DIR, "mlx-community/Llama-3.2-1B-Instruct-4bit") else {
        return;
    };
    assert_batched_decode_parity(&model, LLAMA3_DIR);
}
