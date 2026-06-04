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

//! Real-model end-to-end parity for the transparent pool-backed `KVCache`.
//!
//! Verifies that the paged KV cache primitives (#118 layout / #119 write /
//! #120 read) actually work through a real model's `forward`. This is the
//! single-stream (`B == 1`) groundwork for the scheduler-driven paged cache
//! (#121): one sequence, one [`PagedBlockPool`], every layer's `KVCache`
//! created with `KVCache::new_paged` so that — without touching the model
//! `forward` at all — each layer transparently writes new K/V into the shared
//! pool (`write_prefill`) and reads the visible window back
//! (`gather_visible`).
//!
//! The strongest "it actually works" signal is *identical generation*: the
//! greedy-decoded token sequence from a pool-backed run must exactly match a
//! dense-cache run of the same prompt. As a secondary check the prefill
//! last-position argmax is compared directly.
//!
//! # Running
//!
//! The test is `#[ignore]` so it never blocks `cargo test` on machines without
//! the model checkout. Run it explicitly (it executes a real GPU forward, so
//! prefer `--release`):
//!
//! ```text
//! cargo test --test paged_real_model_parity --release \
//!     --features metal,accelerate -- --ignored --nocapture
//! ```
//!
//! It soft-skips (prints a notice and returns) when
//! `models/qwen3-0.6b-4bit` is absent. Fetch it with:
//!
//! ```text
//! ./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit
//! ```

mod common;
use common::repo_model_dir;

use mlxcel::{LanguageModel, initialize_runtime, load_model};
use mlxcel_core::cache::{KVCache, PagedBlockPool, PagedKvLayout, PagedSequenceState};
use std::cell::RefCell;
use std::rc::Rc;

/// Model directory name (present at `models/qwen3-0.6b-4bit`).
const MODEL_DIR_NAME: &str = "qwen3-0.6b-4bit";

/// Paged block size (tokens per physical block). Any positive value works;
/// 32 keeps several blocks in play across the prompt + decode window so the
/// gather path exercises multi-block stitching rather than a single block.
const BLOCK_SIZE: usize = 32;

/// Number of greedy decode steps to compare after prefill.
const DECODE_STEPS: usize = 16;

/// Fixed prompt token ids (kept small and deterministic — no tokenizer needed,
/// so the test stays purely about cache parity). These are arbitrary valid ids
/// in the Qwen3 vocab; identical bytes feed both the dense and paged runs.
const PROMPT_TOKENS: &[i32] = &[
    9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911, 697, 7990, 13, 358,
];

/// Greedy argmax over the vocab at sequence position `pos` of a
/// `[1, seq_len, vocab]` logits tensor. Mirrors the private
/// `generate::logits_at_position` + `argmax_last_axis` idiom.
fn greedy_token(logits: &mlxcel_core::MlxArray, pos: usize) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let batch = shape[0];
    let vocab = shape[2];
    // [1, 1, vocab] at the requested position.
    let at_pos = mlxcel_core::slice(logits, &[0, pos as i32, 0], &[batch, pos as i32 + 1, vocab]);
    // [vocab] so argmax_last_axis collapses to a scalar.
    let flat = mlxcel_core::reshape(&at_pos, &[vocab]);
    mlxcel_core::eval(&flat);
    mlxcel_core::item_i32(&mlxcel_core::argmax_last_axis(&flat))
}

/// Run prefill + `DECODE_STEPS` greedy decode steps against `caches`,
/// returning the decoded token sequence (one entry per decode step).
///
/// `caches` is whatever the caller built — `model.make_caches()` for the dense
/// run or a `Vec<KVCache>` of `new_paged(...)` for the paged run. The forward
/// calls are byte-for-byte identical between the two; only the cache backing
/// differs.
fn run_generation(model: &mlxcel::LoadedModel, caches: &mut [KVCache]) -> Vec<i32> {
    let prompt_len = PROMPT_TOKENS.len();

    // ── Prefill ───────────────────────────────────────────────────────────
    let prompt = mlxcel_core::from_slice_i32(PROMPT_TOKENS, &[1, prompt_len as i32]);
    let mask = mlxcel_core::utils::create_causal_mask(prompt_len as i32, 0);
    let prefill_logits = model.forward(&prompt, caches, Some(&mask));
    mlxcel_core::eval(&prefill_logits);

    let mut next = greedy_token(&prefill_logits, prompt_len - 1);

    // ── Decode ────────────────────────────────────────────────────────────
    let mut decoded = Vec::with_capacity(DECODE_STEPS);
    for _ in 0..DECODE_STEPS {
        decoded.push(next);
        // Feed the just-sampled token back as a [1, 1] step (no mask on decode,
        // matching the production single-token path).
        let step_input = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        let logits = model.forward(&step_input, caches, None);
        mlxcel_core::eval(&logits);
        next = greedy_token(&logits, 0);
    }
    decoded
}

/// Build a shared paged pool + per-sequence state and one `new_paged`
/// `KVCache` per layer.
///
/// `bytes_per_block` only needs to be a positive multiple of `BLOCK_SIZE`: the
/// pool infers the real `(n_kv_heads, head_dim)` geometry lazily from the first
/// `write_prefill`, and `bytes_per_block` is used purely for the layout's
/// scheduling-byte accounting. We still size it from the qwen3-0.6b-4bit
/// geometry (8 KV heads × 128 head_dim × 2 bytes/fp16 = 2048 bytes/token) so
/// the accounting is realistic.
fn build_paged_caches(num_layers: usize) -> Vec<KVCache> {
    const FP16_BYTES_PER_TOKEN: usize = 8 /* n_kv_heads */ * 128 /* head_dim */ * 2 /* fp16 */;
    let bytes_per_block = BLOCK_SIZE * FP16_BYTES_PER_TOKEN;

    let layout = PagedKvLayout::uniform(num_layers, BLOCK_SIZE, bytes_per_block)
        .expect("uniform paged layout");
    let pool = Rc::new(RefCell::new(PagedBlockPool::new(layout.clone())));
    let state = Rc::new(RefCell::new(PagedSequenceState::new(&layout)));

    (0..num_layers)
        .map(|layer_idx| KVCache::new_paged(Rc::clone(&pool), Rc::clone(&state), layer_idx))
        .collect()
}

/// Pool-backed `KVCache` must produce identical greedy generation to a dense
/// cache on a real model — proving #118/#119/#120 work through `forward`.
#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs a real GPU forward; run with --ignored"]
fn paged_kvcache_matches_dense_qwen3_06b() {
    let model_dir = repo_model_dir(MODEL_DIR_NAME);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {MODEL_DIR_NAME}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit",
            model_dir.display()
        );
        return;
    }

    let _runtime = initialize_runtime();
    let (model, _tokenizer) = load_model(&model_dir).expect("load qwen3-0.6b-4bit");
    let num_layers = model.num_layers();
    eprintln!("\n=== paged vs dense KV cache parity: {MODEL_DIR_NAME} ({num_layers} layers) ===");

    // ── Dense reference run ───────────────────────────────────────────────
    let mut dense_caches = model.make_caches();
    let dense_tokens = run_generation(&model, &mut dense_caches);
    eprintln!("dense  decoded: {dense_tokens:?}");

    // ── Pool-backed run ───────────────────────────────────────────────────
    let mut paged_caches = build_paged_caches(num_layers);
    let paged_tokens = run_generation(&model, &mut paged_caches);
    eprintln!("paged  decoded: {paged_tokens:?}");

    // The decisive parity signal: identical greedy generation.
    assert_eq!(
        paged_tokens, dense_tokens,
        "pool-backed KVCache produced different greedy tokens than the dense cache\n\
         dense: {dense_tokens:?}\npaged: {paged_tokens:?}"
    );

    eprintln!("OK: {DECODE_STEPS} decode steps identical between dense and pool-backed KV cache.");
}
