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

//! Real-model parity for the serving-role KV handoff scheduler entries (#126
//! B2b/B2c).
//!
//! `tests/paged_handoff_parity.rs` proves the handoff at the `CachePool` + model
//! level. This goes one layer up to the BatchScheduler serving-role entries that
//! a disaggregated worker drives: it prefills a request through
//! [`BatchScheduler::prefill_request_for_handoff`], ships the frame over an
//! in-process [`ServingCoordinator`] pair (MockTransport), reconstructs it with
//! [`BatchScheduler::ingest_handoff_as_active`], and decodes the restored
//! sequence. The decode token ids must be byte-identical to a single-node run of
//! the same prompt through the same scheduler.
//!
//! This test is in-crate because `BatchScheduler` is `pub(crate)` and cannot be
//! constructed from an integration test (same constraint noted in
//! `tests/paged_scheduler_parity.rs`). It is `#[ignore]` (loads a real
//! checkpoint and runs real GPU forwards) and soft-skips when the model is
//! absent. Run with:
//!
//! ```text
//! cargo test -p mlxcel --lib --release --features metal,accelerate,test-utils \
//!     serving_handoff_parity -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Fetch the model with:
//! `./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit`.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::generate::SamplingConfig;

use super::BatchScheduler;
use crate::distributed::disaggregated::ServingCoordinator;
use crate::distributed::disaggregated::serving::ServingMode;
use crate::distributed::mock_transport::{MockRouter, MockTransport, MockTransportConfig};
use crate::server::batch::BatchObservability;
use crate::server::batch::sequence::{RequestPriority, SequenceInfo, SequenceState};
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy};
use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

/// qwen3 checkpoint directory name (a pool-backed Fp16 family, the handoff scope).
const QWEN3_DIR: &str = "qwen3-0.6b-4bit";

/// Number of greedy decode tokens to compare across the handoff.
const DECODE_STEPS: usize = 16;

/// A high per-request budget so neither run finishes before `DECODE_STEPS`.
const MAX_TOKENS: usize = 64;

/// A fixed ~50-token prompt (> one 32-token block, so the sequence spans two
/// physical blocks). Deterministic ids; matches `paged_handoff_parity`.
const PROMPT_TOKENS: &[i32] = &[
    9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911, 697, 7990, 13, 358, 2776, 264,
    10950, 17847, 13, 6771, 594, 1438, 419, 1495, 3019, 553, 3019, 11, 323, 1473, 697, 975, 13,
    5209, 387, 2797, 624, 14374, 14582, 25, 3555, 374, 220, 17, 488, 220, 17, 30,
];

/// `<CARGO_MANIFEST_DIR>/models/<name>` (the mlxcel crate root is the repo root).
fn repo_model_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

/// Build a paged-backed batch scheduler over a real model. A `max_batch_size`
/// above 1 with `DecodeStorageBackend::Paged` selects the pool-backed path for a
/// dense-natural Fp16 family (qwen3), which the handoff requires.
fn build_paged_scheduler(
    model: crate::LoadedModel,
    tokenizer: MlxcelTokenizer,
    config_eos: Vec<i32>,
) -> BatchScheduler {
    let (_req_tx, req_rx) = mpsc::channel();
    BatchScheduler::with_config(
        model,
        tokenizer,
        config_eos,
        req_rx,
        2,  // max_batch_size (> 1 so the paged backend is eligible)
        64, // max_queue_depth
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        512,   // prefill_chunk_size (> prompt len, so full prefill)
        false, // enable_preemption
        PreemptionPolicy::default(),
        1, // max_batch_prefill
        DecodeStorageBackend::Paged,
    )
}

/// Build a greedy decode request sequence bound to `seq_id` with the shared
/// prompt. The response channel is returned but unused (parity reads decoded
/// token ids directly off the active sequence).
fn make_request(
    seq_id: mlxcel_core::cache::SequenceId,
    tokenizer: &MlxcelTokenizer,
) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let prompt_tokens = PROMPT_TOKENS.to_vec();
    let decode_state = StreamingDecodeState::new(tokenizer, &prompt_tokens);
    let seq = SequenceInfo {
        seq_id,
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::greedy(),
        max_tokens: MAX_TOKENS,
        eos_token_ids: Vec::new(),
        priority: RequestPriority::Normal,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        prefill_start_offset: 0,
        already_cached_tokens: 0,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        merged_eos: Vec::new(),
        thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        structured: None,
    };
    (seq, rx)
}

/// Drive `execute_decode_step` for `seq_id` until it has accumulated
/// `DECODE_STEPS` tokens (or the sequence leaves the active batch), then return
/// its decoded token ids.
fn drive_decode(sched: &mut BatchScheduler, seq_id: mlxcel_core::cache::SequenceId) -> Vec<i32> {
    loop {
        let produced = sched
            .active_batch
            .get_mut(seq_id)
            .map(|s| s.generated_tokens.len());
        match produced {
            Some(n) if n >= DECODE_STEPS => break,
            Some(_) => sched.execute_decode_step(&[seq_id]),
            None => break,
        }
    }
    sched
        .active_batch
        .get_mut(seq_id)
        .map(|s| s.generated_tokens.clone())
        .unwrap_or_default()
}

/// Ship `bytes` from a PrefillOnly to a DecodeOnly [`ServingCoordinator`] over an
/// in-process MockTransport pair (exercising the B2 coordinator seam) and return
/// the delivered bytes.
fn ship_over_coordinators(bytes: &[u8]) -> Vec<u8> {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let router = MockRouter::new();
        let prefill = ServingCoordinator::new(
            ServingMode::PrefillOnly,
            Box::new(
                MockTransport::new(
                    "prefill".to_string(),
                    router.clone(),
                    MockTransportConfig::default(),
                )
                .await,
            ),
            "decode",
        );
        let decode = ServingCoordinator::new(
            ServingMode::DecodeOnly,
            Box::new(
                MockTransport::new(
                    "decode".to_string(),
                    router.clone(),
                    MockTransportConfig::default(),
                )
                .await,
            ),
            "prefill",
        );
        prefill.send_handoff(bytes).await.expect("send handoff");
        let (from, delivered) = decode.recv_handoff().await.expect("recv handoff");
        assert_eq!(from, "prefill", "handoff sender must be the prefill node");
        delivered
    })
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs real GPU forwards; run with --ignored"]
fn serving_handoff_parity_matches_single_node_qwen3() {
    let _runtime = crate::initialize_runtime();
    let dir = repo_model_dir(QWEN3_DIR);
    if !dir.exists() {
        eprintln!(
            "Skipping {QWEN3_DIR}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit",
            dir.display()
        );
        return;
    }
    let (model, sched_tokenizer) =
        crate::load_model(&dir).unwrap_or_else(|e| panic!("load {QWEN3_DIR}: {e:?}"));
    // A second tokenizer instance for building request sequences (the first is
    // moved into the scheduler).
    let seq_tokenizer = crate::tokenizer::load_tokenizer(&dir).expect("load tokenizer");
    let config_eos = crate::read_eos_token_ids(&dir);
    let mut sched = build_paged_scheduler(model, sched_tokenizer, config_eos);

    // ---- REFERENCE: single-node prefill + decode through the scheduler. ----
    let ref_id = sched
        .allocate_sequence_state()
        .expect("allocate reference sequence");
    let (mut ref_seq, _ref_rx) = make_request(ref_id, &seq_tokenizer);
    BatchScheduler::begin_prefill(&mut ref_seq).expect("begin reference prefill");
    sched.execute_full_prefill(ref_seq);
    let ref_tokens = drive_decode(&mut sched, ref_id);
    assert_eq!(
        ref_tokens.len(),
        DECODE_STEPS,
        "reference run should produce {DECODE_STEPS} tokens, got {}",
        ref_tokens.len()
    );
    // Release the reference sequence so the handoff run starts from a clean pool.
    sched.active_batch.remove(ref_id);
    sched.release_sequence_caches(ref_id);
    eprintln!("reference decoded: {ref_tokens:?}");

    // ---- HANDOFF: prefill -> extract -> coordinator wire -> ingest -> decode. ----
    let prefill_id = sched
        .allocate_sequence_state()
        .expect("allocate handoff prefill sequence");
    let (prefill_seq, _pfx_rx) = make_request(prefill_id, &seq_tokenizer);
    let wire = sched
        .prefill_request_for_handoff(prefill_seq)
        .expect("prefill-role extract")
        .expect("prefill produced a handoff frame (not an immediate EOS)");
    let delivered = ship_over_coordinators(&wire);
    assert_eq!(
        delivered, wire,
        "the coordinator transport must deliver the handoff frame unchanged"
    );
    let (resp_tx, _resp_rx) = mpsc::channel();
    let decode_id = sched
        .ingest_handoff_as_active(&delivered, MAX_TOKENS, SamplingConfig::greedy(), resp_tx)
        .expect("decode-role ingest");
    let handoff_tokens = drive_decode(&mut sched, decode_id);
    eprintln!("handoff   decoded: {handoff_tokens:?}");

    assert_eq!(
        handoff_tokens, ref_tokens,
        "serving-role scheduler handoff decode must equal the single-node run\n\
         reference: {ref_tokens:?}\nhandoff:   {handoff_tokens:?}"
    );
    eprintln!(
        "OK: prefill-role extracted, shipped over the coordinator transport, and the \
         decode-role reconstructed + decoded {DECODE_STEPS} tokens identically to the \
         single-node run."
    );
}
