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

//! The coordinator's failure path against a silent or hostile stage
//! (#2782): the availability gate, stage text made safe for the log, and
//! the budget on teardown acknowledgement warnings.
//!
//! Every test runs a real `RemotePipelineRuntime` over authenticated TCP
//! against a scripted stage on an ephemeral loopback port, with short
//! timeouts (`stage_timeout` 1 s, teardown 100 ms, probe interval 300 ms)
//! so the whole file runs in seconds.

use std::sync::Arc;
use std::time::{Duration, Instant};

use mlxcel_core::cache::SequenceId;

use super::super::stage_availability::PipelineUnavailable;
use super::super::stub_stage::{RunFault, STUB_FORCED_ERROR, StubRunBehavior, StubStageHandle};
use super::log_capture_tests::{capture_logs, gate_open_lines};
use super::*;
use crate::distributed::cluster_auth::{ClusterAuth, ClusterSecret};
use crate::distributed::handshake_warn::HANDSHAKE_WARN_PER_WINDOW;

const VOCAB: i32 = 8;
const TOKEN: i32 = 3;
const TEARDOWN_ACK: Duration = Duration::from_millis(100);
// Long enough that a loaded CI runner cannot pass it between B's probe and
// C's refusal (300 ms let C probe again on the hosted 2-host logical job).
const PROBE_INTERVAL: Duration = Duration::from_secs(2);

fn test_auth() -> Arc<ClusterAuth> {
    Arc::new(ClusterAuth::new(&ClusterSecret::random()))
}

/// A stage that serves every forward with logits, except the `fault_on`-th.
fn serving(fault_on: Option<u32>, fault: RunFault) -> StubRunBehavior {
    StubRunBehavior::Logits {
        vocab: VOCAB,
        token: TOKEN,
        fault_on,
        fault,
    }
}

fn runtime_over(
    stage: &StubStageHandle,
    auth: Arc<ClusterAuth>,
    stage_timeout: Duration,
) -> RemotePipelineRuntime {
    runtime_over_stages(&[stage], auth, stage_timeout)
}

fn config_over(
    stage_peers: Vec<String>,
    auth: Arc<ClusterAuth>,
    stage_timeout: Duration,
) -> RemotePipelineRuntimeConfig {
    RemotePipelineRuntimeConfig {
        stage_peers,
        transport_backend: TransportBackend::Tcp,
        bind_address: "127.0.0.1:0".to_string(),
        stage_timeout,
        auth,
        vocab_size: Some(VOCAB as usize),
    }
}

/// A runtime over `stages` in pipeline order.
fn runtime_over_stages(
    stages: &[&StubStageHandle],
    auth: Arc<ClusterAuth>,
    stage_timeout: Duration,
) -> RemotePipelineRuntime {
    let peers = stages
        .iter()
        .map(|stage| stage.addr().to_string())
        .collect();
    RemotePipelineRuntime::new(config_over(peers, auth, stage_timeout))
        .expect("create runtime")
        .with_teardown_ack_timeout(TEARDOWN_ACK)
        .with_probe_interval(PROBE_INTERVAL)
}

fn input_ids() -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(&[1], &[1, 1])
}

/// The typed refusal anywhere in the error chain.
fn unavailable(err: &anyhow::Error) -> Option<&PipelineUnavailable> {
    err.chain().find_map(|cause| cause.downcast_ref())
}

/// Item 1: while a stage stays silent, requests after the first fail fast
/// instead of each waiting a full `stage_timeout`. The first request after
/// the timeout probes (one short wait); requests inside the probe interval
/// are refused at once and send nothing; once the stage answers a probe the
/// gate closes and the request that probed is served.
#[test]
fn silent_pipeline_fails_new_prepares_fast_until_a_probe_succeeds() {
    let auth = test_auth();
    let stage = StubStageHandle::spawn(serving(None, RunFault::Error), Arc::clone(&auth))
        .expect("spawn stub stage");
    let runtime = runtime_over(&stage, auth, Duration::from_secs(1));
    let input = input_ids();

    // A is prepared while the stage answers, then the stage goes silent:
    // A's forward times out and opens the gate.
    let a = SequenceId::from_raw(1);
    runtime.prepare_sequence_state(a).expect("prepare A");
    stage.set_silent(true);
    let err = runtime
        .forward_sequence(a, &input, None)
        .err()
        .expect("forward A times out");
    assert!(err.to_string().contains("timed out"), "{err:#}");
    assert_eq!(
        stage.wait_for_commands(4, Duration::from_secs(2)),
        ["prepare", "run_entry", "cancel", "release"],
        "A is torn down without waiting on the silent stage"
    );

    // B is the first request after the timeout: it probes, the probe goes
    // unanswered, and B is refused after one teardown-length wait, not a
    // stage timeout.
    let started = Instant::now();
    let err = runtime
        .prepare_sequence_state(SequenceId::from_raw(2))
        .expect_err("prepare B is refused while the stage is silent");
    let elapsed = started.elapsed();
    let refused =
        unavailable(&err).unwrap_or_else(|| panic!("expected PipelineUnavailable: {err:#}"));
    assert_eq!(refused.stage.as_deref(), Some(stage.addr()));
    assert!(
        elapsed < Duration::from_millis(500),
        "prepare B took {elapsed:?}; the gate must refuse within one probe wait"
    );
    let after_b = stage.wait_for_commands(5, Duration::from_secs(2));
    assert_eq!(
        after_b,
        ["prepare", "run_entry", "cancel", "release", "probe_state"],
        "B sends one probe and no prepare"
    );

    // C arrives inside the probe interval: refused at once, nothing sent.
    let started = Instant::now();
    let err = runtime
        .prepare_sequence_state(SequenceId::from_raw(3))
        .expect_err("prepare C is refused while the stage is silent");
    let elapsed = started.elapsed();
    assert!(unavailable(&err).is_some(), "{err:#}");
    // The command log below is the precise check (nothing is sent). The time
    // bound only has to separate a refusal from the 1 s stage timeout, and
    // stays loose because a loaded CI runner took about 100 ms here (#2787).
    assert!(
        elapsed < Duration::from_millis(500),
        "prepare C took {elapsed:?}; a refusal inside the probe interval must not wait for the stage timeout"
    );
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(
        stage.commands(),
        after_b,
        "a refusal inside the probe interval adds no command"
    );

    // The stage answers again. After the probe interval the next request
    // probes, the probe succeeds, the gate closes and D is served.
    stage.set_silent(false);
    std::thread::sleep(PROBE_INTERVAL + Duration::from_millis(50));
    let d = SequenceId::from_raw(4);
    runtime
        .prepare_sequence_state(d)
        .expect("prepare D is served once the probe succeeds");
    let commands = stage.commands();
    assert_eq!(
        &commands[commands.len() - 2..],
        ["probe_state", "prepare"],
        "{commands:?}"
    );
    let logits = runtime
        .forward_sequence(d, &input, None)
        .expect("forward D is served with the gate closed");
    assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 1, VOCAB]);
    runtime.release_sequence_state_by_id(d);

    runtime.shutdown().expect("shutdown runtime");
    stage.shutdown().expect("shutdown stub stage");
}

/// Item 1, the reason the gate probes instead of cooling down: a single
/// dropped reply (the stage is healthy, one `RunResult` never arrived) must
/// not refuse the following window of requests. The first forward times
/// out, the next prepare probes, the stage answers, and that request is
/// served normally.
#[test]
fn a_single_dropped_reply_does_not_refuse_the_next_request() {
    let auth = test_auth();
    let stage = StubStageHandle::spawn(serving(Some(1), RunFault::Silent), Arc::clone(&auth))
        .expect("spawn stub stage");
    let runtime = runtime_over(&stage, auth, Duration::from_millis(300));
    let input = input_ids();

    let a = SequenceId::from_raw(1);
    runtime.prepare_sequence_state(a).expect("prepare A");
    let err = runtime
        .forward_sequence(a, &input, None)
        .err()
        .expect("the dropped reply times out");
    assert!(err.to_string().contains("timed out"), "{err:#}");

    let b = SequenceId::from_raw(2);
    runtime
        .prepare_sequence_state(b)
        .expect("the next prepare probes the stage, which answers, and is served");
    let logits = runtime
        .forward_sequence(b, &input, None)
        .expect("forward B returns logits");
    assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 1, VOCAB]);
    assert_eq!(
        stage.commands(),
        [
            "prepare",
            "run_entry",
            "cancel",
            "release",
            "probe_state",
            "prepare",
            "run_entry",
        ],
        "the request after a timeout probes once and is then served"
    );
    runtime.release_sequence_state_by_id(b);

    runtime.shutdown().expect("shutdown runtime");
    stage.shutdown().expect("shutdown stub stage");
}

/// Item 2: the text of a stage's `Error` reply reaches the coordinator log
/// through the failed forward's error. Newlines and escapes in it are
/// written as escape sequences, so an authenticated stage cannot forge
/// coordinator log lines.
#[test]
fn stage_error_text_is_escaped_before_logging() {
    let auth = test_auth();
    let stage = StubStageHandle::spawn(
        StubRunBehavior::Error {
            message: "boom\nINFO forged\u{1b}[31m",
        },
        Arc::clone(&auth),
    )
    .expect("spawn stub stage");
    let runtime = runtime_over(&stage, auth, Duration::from_secs(1));

    let seq = SequenceId::from_raw(1);
    runtime.prepare_sequence_state(seq).expect("prepare");
    let err = runtime
        .forward_sequence(seq, &input_ids(), None)
        .err()
        .expect("the stage fails the forward");
    let text = format!("{err:#}");
    assert!(
        !text.contains('\n') && !text.contains('\u{1b}'),
        "stage text reached the error unescaped: {text:?}"
    );
    assert!(text.contains("boom\\n"), "{text:?}");

    runtime.shutdown().expect("shutdown runtime");
    stage.shutdown().expect("shutdown stub stage");
}

/// Item 3: a stage that is slow to acknowledge a teardown (here: never
/// does) draws its "did not acknowledge" line from the `teardown-ack`
/// warning budget. Twenty failed forwards produce forty misses; every miss
/// is logged, at most two windows' worth at warn (the budget is process
/// wide and this test may straddle one rollover), the rest at debug, and
/// each warn line says the frame was sent. Before the budget, all forty
/// were warnings.
#[test]
fn teardown_ack_warnings_are_rate_limited() {
    const FORWARDS: u64 = 20;
    let (log, _guard) = capture_logs();

    let auth = test_auth();
    let stage = StubStageHandle::spawn_silent_on_teardown(
        StubRunBehavior::Error {
            message: STUB_FORCED_ERROR,
        },
        Arc::clone(&auth),
    )
    .expect("spawn stub stage");
    let runtime = runtime_over(&stage, auth, Duration::from_secs(1))
        .with_teardown_ack_timeout(Duration::from_millis(10));
    let input = input_ids();
    for n in 1..=FORWARDS {
        let seq = SequenceId::from_raw(n);
        runtime.prepare_sequence_state(seq).expect("prepare");
        runtime
            .forward_sequence(seq, &input, None)
            .err()
            .expect("the stage fails every forward with an Error reply");
    }

    let captured = String::from_utf8_lossy(&log.0.lock().unwrap()).into_owned();
    let needle = "did not acknowledge";
    let lines: Vec<&str> = captured.lines().filter(|l| l.contains(needle)).collect();
    let misses = 2 * FORWARDS as usize;
    assert_eq!(lines.len(), misses, "log:\n{captured}");
    let warns: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| l.contains(" WARN "))
        .collect();
    let debugs = lines.iter().filter(|l| l.contains("DEBUG ")).count();
    assert_eq!(warns.len() + debugs, misses, "log:\n{captured}");
    assert!(
        warns.len() <= 2 * HANDSHAKE_WARN_PER_WINDOW as usize,
        "{} of {misses} teardown misses were logged at warn, log:\n{captured}",
        warns.len()
    );
    for line in &warns {
        assert!(
            line.contains("the frame was sent"),
            "a teardown miss warning must say the frame was sent: {line}"
        );
    }

    runtime.shutdown().expect("shutdown runtime");
    stage.shutdown().expect("shutdown stub stage");
}

/// The stage the availability gate names, if any.
fn failed_peer(runtime: &RemotePipelineRuntime) -> Option<String> {
    let availability = runtime.availability.lock().expect("availability poisoned");
    availability.failed_peer().map(str::to_string)
}

/// #2790 item 5: a `RunEntry` reply comes from the last stage after every
/// stage ran, so its timeout cannot say which stage is silent. Here the
/// entry stage holds the run (a stub does not forward it) and stage 2 has
/// stopped answering. The gate opens without naming a stage, so the entry
/// stage is neither blamed in the log nor denied its teardown wait, and the
/// next request's probe names stage 2. Before, the gate named stage 0.
#[test]
fn run_entry_timeout_does_not_name_the_entry_stage() {
    let (log, _guard) = capture_logs();
    let auth = test_auth();
    let spawn = |behavior| StubStageHandle::spawn(behavior, Arc::clone(&auth)).expect("spawn");
    let entry = spawn(StubRunBehavior::Silent);
    let middle = spawn(serving(None, RunFault::Error));
    let last = spawn(serving(None, RunFault::Error));
    let runtime = runtime_over_stages(&[&entry, &middle, &last], auth, Duration::from_secs(1));

    let a = SequenceId::from_raw(1);
    runtime
        .prepare_sequence_state(a)
        .expect("prepare A on every stage");
    last.set_silent(true);
    let err = runtime
        .forward_sequence(a, &input_ids(), None)
        .err()
        .expect("the run times out");
    assert!(err.to_string().contains("timed out"), "{err:#}");
    assert_eq!(
        failed_peer(&runtime),
        None,
        "a RunEntry timeout must not name the entry stage"
    );
    let opened = gate_open_lines(&log);
    assert_eq!(opened.len(), 1, "one open line: {opened:?}");
    assert!(
        !opened[0].contains(entry.addr()),
        "blames the entry: {opened:?}"
    );
    assert!(opened[0].contains("did not complete within"), "{opened:?}");
    // The entry stage still got its teardown commands.
    assert_eq!(
        entry.wait_for_commands(4, Duration::from_secs(2)),
        ["prepare", "run_entry", "cancel", "release"]
    );

    // The next request probes; stage 2 does not answer and is named.
    let err = runtime
        .prepare_sequence_state(SequenceId::from_raw(2))
        .expect_err("prepare B is refused while stage 2 is silent");
    let refused =
        unavailable(&err).unwrap_or_else(|| panic!("expected PipelineUnavailable: {err:#}"));
    assert_eq!(refused.stage.as_deref(), Some(last.addr()));
    assert_eq!(failed_peer(&runtime).as_deref(), Some(last.addr()));

    last.set_silent(false);
    runtime.shutdown().expect("shutdown runtime");
    for stage in [entry, middle, last] {
        stage.shutdown().expect("shutdown stub stage");
    }
}

/// #2790 item 5: a command that cannot be sent opens the gate with the
/// send failure as its reason, not as a timeout. Stage 1's address is a
/// loopback port nothing listens on, so the runtime is built over a bound
/// transport without the startup connect `RemotePipelineRuntime::new` does.
/// Before, the open line said the stage "did not answer within" the timeout.
#[test]
fn send_failure_opens_the_gate_with_a_send_reason() {
    let (log, _guard) = capture_logs();
    let auth = test_auth();
    let stage = StubStageHandle::spawn(serving(None, RunFault::Error), Arc::clone(&auth))
        .expect("spawn stub stage");
    let closed = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve a port");
        listener.local_addr().expect("local addr").to_string()
    };
    // The coordinator's listener must keep accepting the stub's replies, so
    // the transport lives on a runtime whose worker keeps running.
    let listener_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("listener runtime");
    let transport = listener_rt
        .block_on(bind_transport(
            TransportBackend::Tcp,
            "127.0.0.1:0",
            Some(Arc::clone(&auth)),
        ))
        .expect("bind coordinator transport");
    let peers = vec![stage.addr().to_string(), closed.clone()];
    let config = config_over(peers, auth, Duration::from_secs(1));
    let runtime = RemotePipelineRuntime::with_transport(config, transport)
        .expect("create runtime")
        .with_teardown_ack_timeout(TEARDOWN_ACK)
        .with_probe_interval(PROBE_INTERVAL);

    runtime
        .prepare_sequence_state(SequenceId::from_raw(1))
        .expect_err("the prepare cannot reach stage 1");
    let opened = gate_open_lines(&log);
    assert_eq!(opened.len(), 1, "one open line: {opened:?}");
    let line = &opened[0];
    assert!(
        line.contains(&closed) && line.contains("could not be reached"),
        "{line}"
    );
    assert!(
        !line.contains("did not answer within"),
        "not a timeout: {line}"
    );
    assert_eq!(failed_peer(&runtime), Some(closed.clone()));

    drop(runtime);
    stage.shutdown().expect("shutdown stub stage");
    listener_rt.shutdown_background();
}
