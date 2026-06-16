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
//! B2b/B2c) and the disaggregated role loops over a real transport (#126 B3a).
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
//! [`serving_role_loop_parity_matches_single_node_qwen3`] goes one layer further:
//! two independent schedulers run the [`ServingCoordinator::run_prefill_role`] /
//! [`ServingCoordinator::run_decode_role`] loops over a real localhost
//! [`TcpTransport`] (not the in-process MockTransport), the decode loop as a
//! concurrent task. The prefill node streams the first token and the decode node
//! streams the continuation, the same split a router merges for a client, so the
//! concatenated text must match the single-node reference.
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
use crate::distributed::disaggregated::coordinator::{
    serve_decode_role_blocking, serve_prefill_role_blocking,
};
use crate::distributed::disaggregated::serving::ServingMode;
use crate::distributed::disaggregated::{
    DecodeRoleHandoff, PrefillRoleRequest, ServingCoordinator,
};
use crate::distributed::mock_transport::{MockRouter, MockTransport, MockTransportConfig};
use crate::distributed::tcp_transport::{TcpTransport, TcpTransportConfig};
use crate::distributed::transport::Transport;
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

/// Bounded token budget for the role-loop run (#126 B3a). Both the reference and
/// the two-node handoff generate exactly this many tokens (the prefill first
/// token plus the decode continuation), then stop on the length limit, so the
/// comparison is deterministic and quick.
const ROLE_LOOP_MAX_TOKENS: usize = 16;

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
    // 512-token chunk size > the test prompt, so handoff prefills run whole.
    build_paged_scheduler_with_chunk_size(model, tokenizer, config_eos, 512)
}

/// [`build_paged_scheduler`] with an explicit `prefill_chunk_size`, for the
/// chunked-prefill handoff case (issue #197).
fn build_paged_scheduler_with_chunk_size(
    model: crate::LoadedModel,
    tokenizer: MlxcelTokenizer,
    config_eos: Vec<i32>,
    prefill_chunk_size: usize,
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
        prefill_chunk_size,
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

/// Issue #197: a handoff prefill of a prompt LONGER than `--prefill-chunk-size`
/// must drive the standard chunked-prefill machinery to completion before
/// extracting, and the decode node must continue byte-identically to a
/// single-node run. The reference leg runs an UNCHUNKED full prefill through
/// the same scheduler, so this also re-proves chunked == full prefill across
/// the handoff boundary.
#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs real GPU forwards; run with --ignored"]
fn serving_handoff_parity_chunked_prefill_matches_single_node_qwen3() {
    let _runtime = crate::initialize_runtime();
    let dir = repo_model_dir(QWEN3_DIR);
    if !dir.exists() {
        eprintln!("Skipping {QWEN3_DIR}: model directory not found");
        return;
    }
    let (model, sched_tokenizer) =
        crate::load_model(&dir).unwrap_or_else(|e| panic!("load {QWEN3_DIR}: {e:?}"));
    let seq_tokenizer = crate::tokenizer::load_tokenizer(&dir).expect("load tokenizer");
    let config_eos = crate::read_eos_token_ids(&dir);
    // Chunk size 16 << the ~50-token prompt forces a 4-chunk handoff prefill.
    let mut sched = build_paged_scheduler_with_chunk_size(model, sched_tokenizer, config_eos, 16);
    assert!(
        PROMPT_TOKENS.len() > 16,
        "test prompt must exceed the chunk size for a chunked prefill"
    );

    // ---- REFERENCE: single-node UNCHUNKED prefill + decode. ----
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
        "reference run should produce {DECODE_STEPS} tokens"
    );
    sched.active_batch.remove(ref_id);
    sched.release_sequence_caches(ref_id);
    eprintln!("reference decoded: {ref_tokens:?}");

    // ---- HANDOFF: CHUNKED prefill -> extract -> wire -> ingest -> decode. ----
    let prefill_id = sched
        .allocate_sequence_state()
        .expect("allocate handoff prefill sequence");
    let (prefill_seq, _pfx_rx) = make_request(prefill_id, &seq_tokenizer);
    let wire = sched
        .prefill_request_for_handoff(prefill_seq)
        .expect("chunked prefill-role extract")
        .expect("chunked prefill produced a handoff frame (not an immediate EOS)");
    let delivered = ship_over_coordinators(&wire);
    let (resp_tx, _resp_rx) = mpsc::channel();
    let decode_id = sched
        .ingest_handoff_as_active(&delivered, MAX_TOKENS, SamplingConfig::greedy(), resp_tx)
        .expect("decode-role ingest");
    let handoff_tokens = drive_decode(&mut sched, decode_id);
    eprintln!("chunked handoff decoded: {handoff_tokens:?}");

    assert_eq!(
        handoff_tokens, ref_tokens,
        "chunked-prefill handoff decode must equal the single-node run"
    );
    eprintln!(
        "OK: a 4-chunk handoff prefill extracted, shipped, and decoded {DECODE_STEPS} \
         tokens identically to the unchunked single-node run."
    );
}

/// Drain a streamed generation channel into the concatenated token text,
/// ignoring the terminal `Done` event. Panics on an error event so a failed
/// generation surfaces loudly rather than as a silent text mismatch.
fn collect_text(rx: &mpsc::Receiver<GenerateEvent>) -> String {
    let mut text = String::new();
    while let Ok(event) = rx.try_recv() {
        match event {
            GenerateEvent::Token(t) | GenerateEvent::TokenWithLogprobs(t, _) => text.push_str(&t),
            GenerateEvent::Done(_) => {}
            GenerateEvent::Error(e) => panic!("unexpected generation error event: {e}"),
        }
    }
    text
}

/// A TCP transport config bound to an ephemeral localhost port. Two of these in
/// one process simulate a prefill node and a decode node over a real socket.
fn loopback_tcp_config() -> TcpTransportConfig {
    TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ..TcpTransportConfig::default()
    }
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit (3x) and runs real GPU forwards; run with --ignored"]
fn serving_role_loop_parity_matches_single_node_qwen3() {
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

    // ---- REFERENCE: single-node prefill + decode-to-idle, collecting the
    // streamed text. The scheduler is dropped before the two nodes load. ----
    let ref_text = {
        let (model, sched_tokenizer) =
            crate::load_model(&dir).unwrap_or_else(|e| panic!("load {QWEN3_DIR}: {e:?}"));
        let seq_tokenizer = crate::tokenizer::load_tokenizer(&dir).expect("load tokenizer");
        let config_eos = crate::read_eos_token_ids(&dir);
        let mut sched = build_paged_scheduler(model, sched_tokenizer, config_eos);
        let ref_id = sched
            .allocate_sequence_state()
            .expect("allocate reference sequence");
        let (mut ref_seq, ref_rx) = make_request(ref_id, &seq_tokenizer);
        ref_seq.max_tokens = ROLE_LOOP_MAX_TOKENS;
        BatchScheduler::begin_prefill(&mut ref_seq).expect("begin reference prefill");
        sched.execute_full_prefill(ref_seq);
        sched.decode_handoff_until_idle();
        let text = collect_text(&ref_rx);
        assert!(!text.is_empty(), "reference run produced no text");
        eprintln!("reference text: {text:?}");
        text
    };

    // ---- TWO NODES: a prefill node and a decode node, each with its own model
    // load and scheduler, connected over a real localhost TCP transport. ----
    let (prefill_model, prefill_tokenizer) =
        crate::load_model(&dir).unwrap_or_else(|e| panic!("load prefill node {QWEN3_DIR}: {e:?}"));
    let (decode_model, decode_tokenizer) =
        crate::load_model(&dir).unwrap_or_else(|e| panic!("load decode node {QWEN3_DIR}: {e:?}"));
    let mut prefill_sched = build_paged_scheduler(
        prefill_model,
        prefill_tokenizer,
        crate::read_eos_token_ids(&dir),
    );
    let mut decode_sched = build_paged_scheduler(
        decode_model,
        decode_tokenizer,
        crate::read_eos_token_ids(&dir),
    );

    // The prefill node's first-token channel and the decode node's continuation
    // channel; drained after the loops finish.
    let (prefill_resp_tx, prefill_resp_rx) = mpsc::channel();
    let (decode_resp_tx, decode_resp_rx) = mpsc::channel();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread tokio runtime");
    let local = tokio::task::LocalSet::new();
    rt.block_on(local.run_until(async move {
        // Bind both transports on an ephemeral localhost port and cross-wire the
        // peers from the actual bound addresses.
        let decode_transport = TcpTransport::bind(loopback_tcp_config())
            .await
            .expect("bind decode transport");
        let prefill_transport = TcpTransport::bind(loopback_tcp_config())
            .await
            .expect("bind prefill transport");
        let decode_addr = decode_transport.local_addr().expect("decode local addr");
        let prefill_addr = prefill_transport.local_addr().expect("prefill local addr");
        let prefill_coord = ServingCoordinator::new(
            ServingMode::PrefillOnly,
            Box::new(prefill_transport),
            decode_addr,
        );
        let decode_coord = ServingCoordinator::new(
            ServingMode::DecodeOnly,
            Box::new(decode_transport),
            prefill_addr,
        );

        let (prefill_req_tx, prefill_req_rx) = tokio::sync::mpsc::channel::<PrefillRoleRequest>(4);
        let (decode_meta_tx, decode_meta_rx) = tokio::sync::mpsc::channel::<DecodeRoleHandoff>(4);

        // The decode node runs its role loop as a concurrent local task: it owns
        // its coordinator + scheduler, blocks on the inbound frame, and decodes
        // it when the prefill node ships it. The future is `!Send` (the scheduler
        // holds the MLX pool), so it must be `spawn_local`.
        let decode_task = tokio::task::spawn_local(async move {
            decode_coord
                .run_decode_role(&mut decode_sched, decode_meta_rx)
                .await
        });

        // Hand the decode node its per-request coordination metadata, then feed
        // the prefill request. Closing each channel makes its loop return after
        // the single item.
        decode_meta_tx
            .send(DecodeRoleHandoff {
                max_tokens: ROLE_LOOP_MAX_TOKENS,
                sampling: SamplingConfig::greedy(),
                response_tx: decode_resp_tx,
            })
            .await
            .expect("send decode metadata");
        drop(decode_meta_tx);
        prefill_req_tx
            .send(PrefillRoleRequest {
                prompt_tokens: PROMPT_TOKENS.to_vec(),
                sampling: SamplingConfig::greedy(),
                max_tokens: ROLE_LOOP_MAX_TOKENS,
                response_tx: prefill_resp_tx,
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .await
            .expect("send prefill request");
        drop(prefill_req_tx);

        // Drive the prefill loop inline: it prefills the request and ships the
        // extracted frame over TCP, which the decode task receives and decodes.
        prefill_coord
            .run_prefill_role(&mut prefill_sched, prefill_req_rx)
            .await
            .expect("prefill role loop");
        decode_task
            .await
            .expect("join decode role task")
            .expect("decode role loop");
    }));

    let first_token_text = collect_text(&prefill_resp_rx);
    let continuation_text = collect_text(&decode_resp_rx);
    let handoff_text = format!("{first_token_text}{continuation_text}");
    eprintln!(
        "handoff text: {handoff_text:?} \
         (prefill first token: {first_token_text:?}, decode continuation: {continuation_text:?})"
    );

    assert_eq!(
        handoff_text, ref_text,
        "two-node role-loop handoff text must equal the single-node run\n\
         reference: {ref_text:?}\nhandoff:   {handoff_text:?}"
    );
    eprintln!(
        "OK: the prefill node prefilled and shipped the KV frame over localhost TCP, the decode \
         node reconstructed and decoded it, and the concatenated stream is byte-identical to the \
         single-node run."
    );
}

/// The prefill-role driver (#126 B3b1) builds its own current-thread runtime,
/// binds a real localhost TCP listener, and drives the role loop to a clean
/// return when its request intake closes. This exercises the worker-flip glue
/// (runtime + bind + graceful drive) the model worker uses to run a serving
/// role; the request-by-request handoff behaviour is covered by
/// `serving_role_loop_parity_matches_single_node_qwen3`, and the real
/// two-process path lands in B3b2. A closed intake makes the loop return before
/// any model forward, so this is fast, single-threaded, and needs no second node.
#[test]
#[ignore = "loads qwen3-0.6b-4bit to build a scheduler; run with --ignored"]
fn serve_prefill_role_blocking_binds_then_returns_on_closed_intake() {
    let _runtime = crate::initialize_runtime();
    let dir = repo_model_dir(QWEN3_DIR);
    if !dir.exists() {
        eprintln!(
            "Skipping {QWEN3_DIR}: model directory not found at {}.",
            dir.display()
        );
        return;
    }
    let (model, tokenizer) =
        crate::load_model(&dir).unwrap_or_else(|e| panic!("load {QWEN3_DIR}: {e:?}"));
    let mut sched = build_paged_scheduler(model, tokenizer, crate::read_eos_token_ids(&dir));

    // A closed intake makes the role loop drain nothing and return immediately,
    // so the driver only exercises the runtime + bind + graceful-return glue.
    let (requests_tx, requests_rx) = tokio::sync::mpsc::channel::<PrefillRoleRequest>(1);
    drop(requests_tx);
    let (ready_tx, ready_rx) = mpsc::channel();

    serve_prefill_role_blocking(
        loopback_tcp_config(),
        "127.0.0.1:1".to_string(),
        &mut sched,
        requests_rx,
        Some(ready_tx),
    )
    .expect("prefill role driver returns cleanly when the intake is closed");

    let addr = ready_rx
        .recv()
        .expect("the prefill role driver reports its bound listener address");
    assert!(
        addr.starts_with("127.0.0.1:"),
        "the driver bound a localhost listener, got {addr}"
    );
    eprintln!("OK: prefill-role driver bound {addr} and returned on a closed intake.");
}

/// The decode-role counterpart of
/// [`serve_prefill_role_blocking_binds_then_returns_on_closed_intake`]: the same
/// worker-flip glue, driving the decode role loop.
#[test]
#[ignore = "loads qwen3-0.6b-4bit to build a scheduler; run with --ignored"]
fn serve_decode_role_blocking_binds_then_returns_on_closed_intake() {
    let _runtime = crate::initialize_runtime();
    let dir = repo_model_dir(QWEN3_DIR);
    if !dir.exists() {
        eprintln!(
            "Skipping {QWEN3_DIR}: model directory not found at {}.",
            dir.display()
        );
        return;
    }
    let (model, tokenizer) =
        crate::load_model(&dir).unwrap_or_else(|e| panic!("load {QWEN3_DIR}: {e:?}"));
    let mut sched = build_paged_scheduler(model, tokenizer, crate::read_eos_token_ids(&dir));

    let (handoffs_tx, handoffs_rx) = tokio::sync::mpsc::channel::<DecodeRoleHandoff>(1);
    drop(handoffs_tx);
    let (ready_tx, ready_rx) = mpsc::channel();

    serve_decode_role_blocking(
        loopback_tcp_config(),
        "127.0.0.1:1".to_string(),
        &mut sched,
        handoffs_rx,
        Some(ready_tx),
    )
    .expect("decode role driver returns cleanly when the intake is closed");

    let addr = ready_rx
        .recv()
        .expect("the decode role driver reports its bound listener address");
    assert!(
        addr.starts_with("127.0.0.1:"),
        "the driver bound a localhost listener, got {addr}"
    );
    eprintln!("OK: decode-role driver bound {addr} and returned on a closed intake.");
}
