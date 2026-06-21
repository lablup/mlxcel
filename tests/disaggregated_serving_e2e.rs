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

//! Real two-process disaggregated serving handoff (#126 B3b2a).
//!
//! Spawns two real `mlxcel-server` processes, one `--node-role prefill` and one
//! `--node-role decode`, cross-wired over TCP, and drives a request through the
//! live worker-flip + networked wire protocol the way a router will in B3b2b:
//! the test stands in for the router. It sends a [`PrefillRequestFrame`] to the
//! prefill node, which prefills, returns its first token, and hands the KV off
//! to the decode node over TCP; the decode node reconstructs the sequence and
//! returns the continuation. The merged stream (prefill first token + decode
//! continuation) must be byte-for-byte the single-node greedy output.
//!
//! The in-crate `serving_role_loop_parity_matches_single_node_qwen3` test proves
//! the disaggregated role loops reproduce a single-node run for this prompt; this
//! test proves the *live two-process path* (CLI flags -> config chain -> worker
//! flip -> networked role loops across process boundaries) reproduces that same
//! output. [`EXPECTED_SINGLE_NODE_TEXT`] is that single-node reference.
//!
//! `#[ignore]` (loads a real checkpoint twice and runs real GPU forwards across
//! two processes) and soft-skips when the model or the built binary is absent.
//! Run with:
//!
//! ```text
//! cargo test --test disaggregated_serving_e2e --release \
//!     --features metal,accelerate -- --ignored --nocapture
//! ```
//!
//! Fetch the model with:
//! `./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit`.
//!
//! [`PrefillRequestFrame`]: mlxcel::distributed::disaggregated::PrefillRequestFrame

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

use mlxcel::SamplingConfig;
use mlxcel::distributed::disaggregated::{
    PrefillRequestFrame, ResultFrame, ResultPhase, control_parts, sampling_to_serializable,
};
use mlxcel::distributed::tcp_transport::{TcpTransport, TcpTransportConfig};
use mlxcel::distributed::transport::Transport;

/// qwen3 checkpoint directory name (a pool-backed Fp16 family, the handoff scope).
const QWEN3_DIR: &str = "qwen3-0.6b-4bit";

/// Tokens to generate (prefill first token + decode continuation). Matches the
/// in-crate role-loop parity test so they verify the same reference.
const MAX_TOKENS: u64 = 16;

/// The fixed ~50-token prompt (> one 32-token block), identical to the in-crate
/// serving parity tests so the greedy output is the same.
const PROMPT_TOKENS: &[i32] = &[
    9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911, 697, 7990, 13, 358, 2776, 264,
    10950, 17847, 13, 6771, 594, 1438, 419, 1495, 3019, 553, 3019, 11, 323, 1473, 697, 975, 13,
    5209, 387, 2797, 624, 14374, 14582, 25, 3555, 374, 220, 17, 488, 220, 17, 30,
];

/// The single-node greedy continuation qwen3-0.6b-4bit produces for
/// [`PROMPT_TOKENS`] over [`MAX_TOKENS`] tokens. This is the reference the
/// in-crate `serving_role_loop_parity_matches_single_node_qwen3` test verifies
/// the disaggregated role loops reproduce; the live two-process path must
/// reproduce it byte-for-byte. Update both together if the model changes.
const EXPECTED_SINGLE_NODE_TEXT: &str = " \n\n###Answer: 4\n\n###Step 1: 2 + ";

/// Kills its child process on drop so a panicking assertion never leaks a
/// spawned `mlxcel-server`.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Reserve `n` distinct ephemeral localhost ports by binding them all at once
/// (so two reservations cannot return the same port), then releasing them. A
/// small window remains before a child rebinds them, acceptable for a test.
fn reserve_ports(n: usize) -> Vec<u16> {
    let listeners: Vec<TcpListener> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port"))
        .collect();
    listeners
        .iter()
        .map(|l| l.local_addr().expect("local addr").port())
        .collect()
    // `listeners` drop here, freeing the ports for the spawned servers.
}

fn spawn_role_server(args: &[&str]) -> ChildGuard {
    ChildGuard(
        Command::new(repo_binary_path("mlxcel-server"))
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn mlxcel-server"),
    )
}

/// Poll-connect `addr` until it accepts (the node's role transport is bound,
/// which happens after its model loads) or the deadline passes.
async fn wait_for_listener(addr: &str, deadline: Instant) -> bool {
    while Instant::now() < deadline {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "spawns two real mlxcel-server processes loading qwen3-0.6b-4bit; run with --ignored"]
async fn disaggregated_two_process_handoff_matches_single_node() {
    let model_dir = repo_model_dir(QWEN3_DIR);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {QWEN3_DIR}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit",
            model_dir.display()
        );
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!(
            "Skipping: mlxcel-server binary not found at {}.\n\
             Build it with: cargo build --release --bin mlxcel-server --features metal,accelerate",
            binary.display()
        );
        return;
    }

    let ports = reserve_ports(4);
    let prefill_http = ports[0].to_string();
    let decode_http = ports[1].to_string();
    let prefill_serving_addr = format!("127.0.0.1:{}", ports[2]);
    let decode_serving_addr = format!("127.0.0.1:{}", ports[3]);
    let model_arg = model_dir.to_string_lossy().to_string();

    // Spawn the decode node first so it is listening before the prefill node
    // hands off to it. `--decode-storage-backend paged` + `--max-batch-size 2`
    // select the pool-backed path the handoff requires.
    let _decode = spawn_role_server(&[
        "-m",
        &model_arg,
        "--host",
        "127.0.0.1",
        "--port",
        &decode_http,
        "--parallel",
        "2",
        "--max-batch-size",
        "2",
        "--decode-storage-backend",
        "paged",
        "--node-role",
        "decode",
        "--serving-bind",
        &decode_serving_addr,
        "--no-warmup",
    ]);
    let _prefill = spawn_role_server(&[
        "-m",
        &model_arg,
        "--host",
        "127.0.0.1",
        "--port",
        &prefill_http,
        "--parallel",
        "2",
        "--max-batch-size",
        "2",
        "--decode-storage-backend",
        "paged",
        "--node-role",
        "prefill",
        "--serving-bind",
        &prefill_serving_addr,
        "--decode-peers",
        &decode_serving_addr,
        "--no-warmup",
    ]);

    // The test client stands in for the router: it binds its own transport so
    // the nodes can return results to it via the request's `reply_to`.
    let client = TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ..TcpTransportConfig::default()
    })
    .await
    .expect("bind client transport");
    let reply_to = client.local_addr().expect("client local addr");

    // Wait until both nodes' serving transports accept connections (model
    // loaded + role transport bound). Generous deadline for two model loads.
    let deadline = Instant::now() + Duration::from_secs(180);
    assert!(
        wait_for_listener(&decode_serving_addr, deadline).await,
        "decode node serving transport never came up at {decode_serving_addr}"
    );
    assert!(
        wait_for_listener(&prefill_serving_addr, deadline).await,
        "prefill node serving transport never came up at {prefill_serving_addr}"
    );

    // Send the prefill request to the prefill node.
    let request = PrefillRequestFrame {
        request_id: 1,
        prompt_tokens: PROMPT_TOKENS.to_vec(),
        sampling: sampling_to_serializable(&SamplingConfig::greedy()),
        max_tokens: MAX_TOKENS,
        reply_to: reply_to.clone(),
        // Single configured decode peer: let the prefill node use its
        // `--decode-peers` fallback rather than a router-chosen target.
        decode_target: None,
    };
    client
        .send(
            &prefill_serving_addr,
            request.encode().expect("encode prefill request"),
        )
        .await
        .expect("send prefill request to the prefill node");

    // Collect the prefill node's first-token result and the decode node's
    // continuation result (the split a router merges for the client).
    let mut first_token_text = String::new();
    let mut continuation_text = String::new();
    let mut seen_first = false;
    let mut continuation_done = false;
    let mut continuation_frames = 0usize;
    let mut next_sequence: u64 = 1;
    while !(seen_first && continuation_done) {
        let received = tokio::time::timeout(Duration::from_secs(120), client.recv()).await;
        let (_from, message) = match received {
            Ok(Ok(message)) => message,
            Ok(Err(e)) => panic!("client transport recv failed: {e}"),
            Err(_) => break,
        };
        let (_op, payload) = control_parts(message).expect("result is a control frame");
        let frame = ResultFrame::decode(&payload).expect("decode result frame");
        if let Some(err) = frame.error {
            panic!("a serving node returned a generation error: {err}");
        }
        match frame.phase {
            ResultPhase::FirstToken => {
                first_token_text = frame.tokens.concat();
                seen_first = true;
            }
            ResultPhase::Continuation => {
                // Issue #199: the decode node streams the continuation as
                // multiple per-tick frames; accumulate them and verify the
                // wire sequence tags are contiguous.
                if !frame.tokens.is_empty() {
                    assert_eq!(
                        frame.start_sequence, next_sequence,
                        "continuation frame sequence gap"
                    );
                    next_sequence += frame.tokens.len() as u64;
                    continuation_frames += 1;
                }
                continuation_text.push_str(&frame.tokens.concat());
                if frame.done {
                    continuation_done = true;
                }
            }
        }
    }

    assert!(
        seen_first,
        "did not receive the prefill node's first-token result"
    );
    assert!(
        continuation_done,
        "did not receive the decode node's terminal continuation frame"
    );
    // The incremental-streaming acceptance: a multi-token continuation must
    // arrive as MULTIPLE frames (per decode tick), not one buffered frame.
    assert!(
        continuation_frames > 1,
        "expected the continuation streamed across multiple frames, got \
         {continuation_frames}"
    );

    let merged = format!("{first_token_text}{continuation_text}");
    eprintln!(
        "two-process handoff text: {merged:?} \
         (prefill first token {first_token_text:?}, decode continuation {continuation_text:?})"
    );
    assert_eq!(
        merged, EXPECTED_SINGLE_NODE_TEXT,
        "the live two-process disaggregated handoff must reproduce the single-node greedy output\n\
         expected: {EXPECTED_SINGLE_NODE_TEXT:?}\n     got: {merged:?}"
    );
    eprintln!(
        "OK: two real mlxcel-server processes (prefill + decode) handed a sequence off over TCP \
         and reproduced the single-node output byte-for-byte."
    );
}
