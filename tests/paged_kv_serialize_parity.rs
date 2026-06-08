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

//! Real-model parity for the distributed paged KV block-content handoff (#125).
//!
//! Proves the headline behaviour of issue #125: a prefill node can serialize a
//! POOL-BACKED paged sequence's KV (block table + the referenced pool block
//! CONTENTS), ship it over the wire format, and a decode node reconstructs
//! identical KV so greedy decode continues bit-identically to a single-node run,
//! with block accounting (live block count + paged stats) after restore matching
//! the originating node.
//!
//! ## What it drives
//!
//! Like `paged_prefix_share_parity` / `paged_scheduler_parity`, it drives the
//! `CachePool` cache core directly (the live disaggregated scheduler is not
//! wired to the serde functions yet; that transport/scheduler wiring is the
//! epic capstone #126). For each pool-backed family (**qwen3**, **llama3**):
//!
//! 1. **Reference (single node)** — a pool-backed paged sequence prefills the
//!    prompt and greedily decodes `DECODE_STEPS` tokens.
//! 2. **Handoff** — an origin (prefill) `CachePool` prefills the same prompt,
//!    `serialize_cache_pool_sequence` captures its dense metadata + paged block
//!    table + pool block contents, the bytes round-trip through
//!    `serialize_cache_state` / `deserialize_cache_state`, a fresh decode
//!    `CachePool` allocates a paged sequence and `restore_into_cache_pool_sequence`
//!    rebuilds the KV on fresh pool blocks, then greedily decodes the same
//!    `DECODE_STEPS` from the handed-off first token.
//!
//! The handoff decode must equal the reference decode, the origin's first token
//! (its first-token logits' argmax) must match the cold run, and the decode
//! pool's block accounting must equal the origin's.
//!
//! ## Running
//!
//! `#[ignore]` (loads a real checkpoint and runs real GPU forwards). Run with:
//!
//! ```text
//! cargo test --test paged_kv_serialize_parity --release \
//!     --features metal,accelerate -- --ignored --nocapture --test-threads=1
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

use mlxcel::distributed::kv_cache_serde::{
    deserialize_cache_state, restore_into_cache_pool_sequence, serialize_cache_pool_sequence,
    serialize_cache_state,
};
use mlxcel::{LanguageModel, initialize_runtime, load_model};
use mlxcel_core::cache::{CachePool, PagedKvLayout, SequenceStateLayout};

/// qwen3 checkpoint directory name (pool-backed family).
const QWEN3_DIR: &str = "qwen3-0.6b-4bit";
/// llama3 checkpoint directory name (pool-backed family).
const LLAMA3_DIR: &str = "llama-3.2-1b-4bit";

/// Paged block size (tokens per physical block) — matches the scheduler's
/// `DEFAULT_PAGED_BLOCK_SIZE`.
const BLOCK_SIZE: usize = 32;

/// Number of greedy decode steps to compare across the boundary.
const DECODE_STEPS: usize = 16;

/// A fixed ~50-token prompt (> one 32-token block, so the sequence spans two
/// physical blocks and the block-content transfer is non-trivial). Deterministic
/// plausible ids; no tokenizer needed.
const PROMPT_TOKENS: &[i32] = &[
    9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911, 697, 7990, 13, 358, 2776, 264,
    10950, 17847, 13, 6771, 594, 1438, 419, 1495, 3019, 553, 3019, 11, 323, 1473, 697, 975, 13,
    5209, 387, 2797, 624, 14374, 14582, 25, 3555, 374, 220, 17, 488, 220, 17, 30,
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

/// The paged layout the scheduler builds for the default Fp16 KV mode.
fn scheduler_paged_layout(num_layers: usize) -> SequenceStateLayout {
    SequenceStateLayout::paged_kv_cache(
        PagedKvLayout::uniform(num_layers, BLOCK_SIZE, BLOCK_SIZE).expect("valid paged layout"),
    )
}

/// Prefill `prompt` at offset 0 through `caches` and return the first greedy
/// token (its argmax at the last prompt position).
fn prefill_first_token(
    model: &mlxcel::LoadedModel,
    caches: &mut [mlxcel_core::cache::KVCache],
    prompt: &[i32],
) -> i32 {
    let len = prompt.len() as i32;
    let input = mlxcel_core::from_slice_i32(prompt, &[1, len]);
    let mask = mlxcel_core::utils::create_causal_mask(len, 0);
    let logits = model.forward(&input, caches, Some(&mask));
    mlxcel_core::eval(&logits);
    greedy_token(&logits, 0, len - 1)
}

/// Greedily decode `steps` tokens through `caches`, starting from `first`.
fn decode_from(
    model: &mlxcel::LoadedModel,
    caches: &mut [mlxcel_core::cache::KVCache],
    first: i32,
    steps: usize,
) -> Vec<i32> {
    let mut next = first;
    let mut decoded = Vec::with_capacity(steps);
    for _ in 0..steps {
        decoded.push(next);
        let step_input = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        let logits = model.forward(&step_input, caches, None);
        mlxcel_core::eval(&logits);
        next = greedy_token(&logits, 0, 0);
    }
    decoded
}

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

/// Gather one layer's visible K window directly from a sequence's pool and
/// return its raw bytes, for cross-node KV-reconstruction parity checks.
fn gather_layer_k_bytes(
    cp: &CachePool,
    id: mlxcel_core::cache::SequenceId,
    layer: usize,
) -> Vec<u8> {
    let pool = cp.paged_pool_ref().expect("paged pool present");
    let seq = cp.get(id).expect("sequence present");
    let state = seq.paged_state().expect("paged state present");
    let (k, _v) = pool
        .gather_visible(&state, layer)
        .expect("gather_visible ok")
        .expect("gather_visible returned a window");
    mlxcel_core::array_to_raw_bytes(&k)
}

/// Single-node reference vs a serialize -> wire -> deserialize -> restore handoff
/// onto a fresh decode `CachePool`. The handoff must reproduce the reference
/// decode exactly and reconstruct identical block accounting.
fn assert_serialize_handoff_parity(model: &mlxcel::LoadedModel, label: &str) {
    let num_layers = model.num_layers();

    eprintln!(
        "\n=== paged serialize handoff parity: {label} ({num_layers} layers, \
         prompt={} tokens) ===",
        PROMPT_TOKENS.len()
    );

    // ---- REFERENCE: single-node cold prefill + decode. ----
    let (ref_first, ref_tokens) = {
        let mut pool = CachePool::new(2);
        let id = pool
            .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
            .expect("reference paged allocate");
        let first = {
            let caches = pool.get_caches_mut(id).unwrap();
            prefill_first_token(model, caches, PROMPT_TOKENS)
        };
        let caches = pool.get_caches_mut(id).unwrap();
        (first, decode_from(model, caches, first, DECODE_STEPS))
    };
    eprintln!("reference decoded: {ref_tokens:?}");

    // ---- HANDOFF: origin prefill node. ----
    let mut origin = CachePool::new(2);
    let id_o = origin
        .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
        .expect("origin paged allocate");
    let origin_first = {
        let caches = origin.get_caches_mut(id_o).unwrap();
        assert!(
            caches.iter().all(|c| c.is_paged_backed()),
            "origin must be pool-backed (Fp16 paged)"
        );
        prefill_first_token(model, caches, PROMPT_TOKENS)
    };
    assert_eq!(
        origin_first, ref_first,
        "origin first-token logits (argmax) must match the single-node run"
    );

    // Serialize the origin sequence (carries dense metadata + paged block table +
    // pool block CONTENTS) and round-trip through the binary wire format.
    let state = serialize_cache_pool_sequence(&origin, id_o, None, PROMPT_TOKENS.to_vec())
        .expect("serialize origin sequence");
    assert!(
        !state.paged_blocks.is_empty(),
        "a pool-backed sequence must serialize its block contents"
    );
    let wire = serialize_cache_state(&state).expect("serialize to wire bytes");
    let restored_state = deserialize_cache_state(&wire).expect("deserialize wire bytes");

    let origin_live = origin.paged_pool_ref().unwrap().live_block_count();
    let origin_stats = origin.paged_stats();

    // ---- Decode node: fresh pool, allocate, restore via the content path. ----
    let mut decode = CachePool::new(2);
    let id_d = decode
        .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
        .expect("decode paged allocate");
    restore_into_cache_pool_sequence(&restored_state, &mut decode, id_d)
        .expect("restore handed-off paged sequence");

    // KV reconstruction parity: every layer's gathered visible window on the
    // decode node must be byte-identical to the origin's. This is the headline
    // #125 guarantee and localizes any per-layer content/offset drift before the
    // (noisier) end-to-end decode comparison below.
    for layer in 0..num_layers {
        assert_eq!(
            gather_layer_k_bytes(&decode, id_d, layer),
            gather_layer_k_bytes(&origin, id_o, layer),
            "{label}: layer {layer} restored visible K window differs from the origin"
        );
    }

    // RoPE continuation: each restored cache's monotonic offset must equal the
    // prefilled prompt length so the first decode token rotates at the right
    // position (gather above does not exercise `cache.offset`).
    {
        let caches = decode.get_caches_mut(id_d).expect("decode caches");
        for (layer, cache) in caches.iter().enumerate() {
            assert_eq!(
                cache.offset,
                PROMPT_TOKENS.len() as i32,
                "{label}: layer {layer} restored cache offset"
            );
        }
    }

    // Block accounting after restore matches the origin (no leak, no double-free).
    assert_eq!(
        decode.paged_pool_ref().unwrap().live_block_count(),
        origin_live,
        "decode-node live block count must match the origin"
    );
    assert_eq!(
        decode.paged_stats(),
        origin_stats,
        "decode-node paged stats must match the origin"
    );

    // Continue decode on the decode node from the handed-off first token.
    let handoff_tokens = {
        let caches = decode.get_caches_mut(id_d).unwrap();
        assert!(
            caches.iter().all(|c| c.is_paged_backed()),
            "restored decode sequence must be pool-backed"
        );
        decode_from(model, caches, origin_first, DECODE_STEPS)
    };
    eprintln!("handoff   decoded: {handoff_tokens:?}");

    assert_eq!(
        handoff_tokens, ref_tokens,
        "serialize/restore handoff decode must equal the single-node run\n\
         reference: {ref_tokens:?}\nhandoff:   {handoff_tokens:?}"
    );
    eprintln!(
        "OK: {label} reconstructed {} pool blocks across the wire and decoded \
         {DECODE_STEPS} tokens identically to the single-node run.",
        state.paged_blocks.len()
    );
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs real GPU forwards; run with --ignored"]
fn paged_serialize_handoff_matches_single_node_qwen3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(QWEN3_DIR, "mlx-community/Qwen3-0.6B-4bit") else {
        return;
    };
    assert_serialize_handoff_parity(&model, QWEN3_DIR);
}

#[test]
#[ignore = "loads llama-3.2-1b-4bit and runs real GPU forwards; run with --ignored"]
fn paged_serialize_handoff_matches_single_node_llama3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(LLAMA3_DIR, "mlx-community/Llama-3.2-1B-Instruct-4bit") else {
        return;
    };
    assert_serialize_handoff_parity(&model, LLAMA3_DIR);
}
