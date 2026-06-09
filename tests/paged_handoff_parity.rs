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

//! Real-model parity for the disaggregated serving-role KV handoff mechanism
//! (#126 step B1).
//!
//! #125's `paged_kv_serialize_parity` proved the `CachePool`-level serde
//! round-trip (serialize -> wire -> restore). This test goes one layer up to the
//! serving-role HANDOFF mechanism that a disaggregated worker uses, and adds the
//! transport hop, exercising the full path end to end:
//!
//! ```text
//! prefill role:  prefill -> extract_sequence_handoff  (serialize to wire bytes)
//!                                    |
//!                          send_handoff_payload -> MockTransport -> recv_handoff_payload
//!                                    |
//! decode role:   probe_block_geometry + ingest_sequence_handoff (anchored restore)
//!                                    |
//!                          decode from the handed-off first token
//! ```
//!
//! The decode-role output must be byte-identical to a single-node run, the
//! decode model's geometry probe must anchor the restore, and block accounting
//! after restore must match the originating node.
//!
//! ## Why free functions and not a `BatchScheduler`
//!
//! `BatchScheduler` is `pub(crate)`, so an integration test cannot construct
//! one (same constraint as `paged_scheduler_parity`). The scheduler's
//! `extract_sequence_handoff` / `ingest_sequence_handoff` methods are thin
//! delegations to these `pub` `handoff_impl` functions, which carry all the
//! logic; driving them over a `CachePool` + model faithfully reproduces the
//! serving-role handoff without standing up the async worker loop.
//!
//! ## Running
//!
//! `#[ignore]` (loads a real checkpoint and runs real GPU forwards). Run with:
//!
//! ```text
//! cargo test --test paged_handoff_parity --release \
//!     --features metal,accelerate,test-utils -- --ignored --nocapture --test-threads=1
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

use mlxcel::distributed::disaggregated::{
    extract_sequence_handoff, ingest_sequence_handoff, probe_block_geometry, recv_handoff_payload,
    send_handoff_payload,
};
use mlxcel::distributed::kv_cache_serde::CacheIngestLimits;
use mlxcel::distributed::mock_transport::{MockRouter, MockTransport, MockTransportConfig};
use mlxcel::{LanguageModel, initialize_runtime, load_model};
use mlxcel_core::cache::{CachePool, PagedKvLayout, SequenceStateLayout};

/// qwen3 checkpoint directory name (pool-backed family).
const QWEN3_DIR: &str = "qwen3-0.6b-4bit";
/// llama3 checkpoint directory name (pool-backed family).
const LLAMA3_DIR: &str = "llama-3.2-1b-4bit";

/// Paged block size (tokens per physical block), matching the scheduler's
/// `DEFAULT_PAGED_BLOCK_SIZE`.
const BLOCK_SIZE: usize = 32;

/// Number of greedy decode steps to compare across the boundary.
const DECODE_STEPS: usize = 16;

/// A fixed ~50-token prompt (> one 32-token block, so the sequence spans two
/// physical blocks and the block-content handoff is non-trivial). Deterministic
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

/// Ship `bytes` from a prefill node to a decode node over an in-process
/// `MockTransport` pair and return the bytes the decode node received. Proves
/// the serde<->transport byte bridge moves the handoff frame unchanged.
fn ship_over_mock_transport(bytes: &[u8]) -> Vec<u8> {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let router = MockRouter::new();
        let prefill = MockTransport::new(
            "prefill".to_string(),
            router.clone(),
            MockTransportConfig::default(),
        )
        .await;
        let decode = MockTransport::new(
            "decode".to_string(),
            router.clone(),
            MockTransportConfig::default(),
        )
        .await;
        send_handoff_payload(&prefill, "decode", bytes)
            .await
            .expect("send handoff payload");
        let (from, received) = recv_handoff_payload(&decode)
            .await
            .expect("recv handoff payload");
        assert_eq!(from, "prefill", "handoff sender must be the prefill node");
        received
    })
}

/// Single-node reference vs a prefill-role extract -> transport -> decode-role
/// ingest handoff onto a fresh decode `CachePool`. The handoff must reproduce
/// the reference decode exactly and reconstruct identical block accounting.
fn assert_handoff_over_transport_parity(model: &mlxcel::LoadedModel, label: &str) {
    let num_layers = model.num_layers();

    eprintln!(
        "\n=== paged serving-role handoff parity: {label} ({num_layers} layers, \
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

    // ---- PREFILL ROLE: prefill the sequence, then extract a wire frame. ----
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

    // `extract_sequence_handoff` is the prefill-role scheduler hook's core:
    // serialize the pool-backed sequence (metadata + block table + block
    // CONTENTS) to a single wire frame.
    let wire = extract_sequence_handoff(&origin, id_o, None, PROMPT_TOKENS.to_vec())
        .expect("extract origin sequence handoff");

    let origin_live = origin.paged_pool_ref().unwrap().live_block_count();
    let origin_stats = origin.paged_stats();

    // ---- TRANSPORT: ship the frame prefill node -> decode node. ----
    let delivered = ship_over_mock_transport(&wire);
    assert_eq!(
        delivered, wire,
        "the transport must deliver the handoff frame unchanged"
    );

    // ---- DECODE ROLE: probe local geometry, ingest, decode. ----
    // `probe_block_geometry` derives the decode model's exact (n_kv_heads,
    // head_dim, dtype) by a one-token probe, the anchor source the ingest hook
    // caches on the scheduler.
    let geometry = probe_block_geometry(model, BLOCK_SIZE).expect("probe decode geometry");
    assert_eq!(
        geometry.num_layers, num_layers,
        "probed geometry layer count must match the model"
    );

    let mut decode = CachePool::new(2);
    let id_d = ingest_sequence_handoff(
        &mut decode,
        model,
        &delivered,
        &CacheIngestLimits::default(),
        &geometry,
        BLOCK_SIZE,
    )
    .expect("ingest handed-off sequence");

    // KV reconstruction parity: every layer's gathered visible window on the
    // decode node must be byte-identical to the origin's. This is the headline
    // handoff guarantee and localizes any per-layer content/offset drift before
    // the (noisier) end-to-end decode comparison below.
    for layer in 0..num_layers {
        assert_eq!(
            gather_layer_k_bytes(&decode, id_d, layer),
            gather_layer_k_bytes(&origin, id_o, layer),
            "{label}: layer {layer} restored visible K window differs from the origin"
        );
    }

    // RoPE continuation: each restored cache's monotonic offset must equal the
    // prefilled prompt length so the first decode token rotates at the right
    // position.
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
        "serving-role handoff decode must equal the single-node run\n\
         reference: {ref_tokens:?}\nhandoff:   {handoff_tokens:?}"
    );
    eprintln!(
        "OK: {label} extracted, shipped over transport, and reconstructed the sequence; \
         decoded {DECODE_STEPS} tokens identically to the single-node run."
    );
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs real GPU forwards; run with --ignored"]
fn paged_handoff_over_transport_matches_single_node_qwen3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(QWEN3_DIR, "mlx-community/Qwen3-0.6B-4bit") else {
        return;
    };
    assert_handoff_over_transport_parity(&model, QWEN3_DIR);
}

#[test]
#[ignore = "loads llama-3.2-1b-4bit and runs real GPU forwards; run with --ignored"]
fn paged_handoff_over_transport_matches_single_node_llama3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(LLAMA3_DIR, "mlx-community/Llama-3.2-1B-Instruct-4bit") else {
        return;
    };
    assert_handoff_over_transport_parity(&model, LLAMA3_DIR);
}
