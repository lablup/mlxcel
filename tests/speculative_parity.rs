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

//! Greedy-parity verification for speculative drafter pairings (sub-9).
//!
//! ## What this file pins
//!
//! At `temperature = 0.0` the speculative-decoding round loop emits a token
//! sequence that is byte-identical to the equivalent drafter-less greedy
//! pass through the target — this invariant is the load-bearing correctness
//! gate of every speculative path in mlxcel. The integration tests below
//! drive a real target + real drafter end-to-end and assert byte-equality
//! against a paired drafter-less baseline.
//!
//! ## Reachable pairings (this PR)
//!
//! - **Qwen 3.5 4B + DFlash** (`models/qwen3.5-4b-4bit` target, drafter
//!   `models/Qwen3.5-4B-DFlash`, `block_size = 16`). The DFlash drafter
//!   is loadable via `mlxcel_core::drafter::dflash::DFlashDrafter::load`
//!   and the Qwen 3.5 text / VLM wrappers implement
//!   `mlxcel_core::drafter::dflash::SpeculativeTarget` (wired the B>1 path too; enables the VLM-wrapped text-only server dispatch). The published upstream `z-lab/Qwen3.5-4B-DFlash`
//!   checkpoint omits `embed_tokens.weight`; ported upstream's
//!   lazy-bind shape so the drafter loads with a tombstone and resolves
//!   its embedding from the target during `Drafter::bind`. The
//!   `greedy_parity_dflash_qwen35_4b` test runs the structural load +
//!   bind pin *and* the end-to-end byte-equality phase
//!   (`mlxcel-server` with vs without `--draft-kind dflash` at temp=0),
//!   plus an log assertion that the speculative server emitted
//!   `Speculative burst completed` instead of silently falling back.
//!
//! - **Gemma 4 31B + MTP assistant** (`models/gemma-4-31b-it-4bit` target,
//!   drafter `models/gemma-4-31B-it-assistant-bf16`, `block_size = 4`). The
//!   MTP drafter is loadable via
//!   `mlxcel_core::drafter::gemma4_assistant::Gemma4AssistantDraftModel::from_path`,
//!   `Gemma4Wrapper` exposes the underlying primitives
//!   (`forward_with_speculative_sinks`, `rollback_speculative_cache`), and
//!   the server-side MTP burst dispatch + `MtpTarget` adapter were wired.
//!   The `greedy_parity_mtp_gemma4_31b`
//!   test runs the structural load pin *and* the end-to-end
//!   byte-equality phase (`mlxcel-server` with vs without `--draft-kind
//!   mtp` at temp=0).
//!
//! ## Deferred pairings (drafter checkpoint not on disk)
//!
//! - **Gemma 4 E2B / E4B** — drafter checkpoints (`gemma-4-E2B-it-assistant-bf16`,
//!   `gemma-4-E4B-it-assistant-bf16`) are not on local disk and the E-family
//!   pairings additionally depend on the centroid LM head integration tracked
//!   by follow-up.
//! - **Gemma 4 26B-A4B** — drafter checkpoint (`gemma-4-26B-A4B-it-assistant-bf16`)
//!   is not on local disk. Skipped pending checkpoint availability.
//!
//! ## End-to-end byte-equality
//!
//! `greedy_parity_dflash_qwen35_4b` and `greedy_parity_mtp_gemma4_31b` each
//! run a two-phase check:
//!
//! 1. **Structural phase** (in-process): load the target, assert the model
//!    variant, resolve the drafter kind, load the drafter, and — for
//!    DFlash — `bind()` the drafter to the target (the lazy-bind pin). The in-process models are then dropped and the MLX
//!    memory cache cleared before phase 2.
//! 2. **Byte-equality phase** (subprocess): spawn `mlxcel-server` twice
//!    against the same target — once with `--model-draft --draft-kind
//!    --draft-block-size` (speculative) and once without (drafter-less
//!    baseline) — submit the same fixed prompt to `/v1/chat/completions`
//!    at `temperature = 0`, and assert the two responses are
//!    byte-identical (same `message.content` *and* same
//!    `usage.completion_tokens`). The speculative server log must also
//!    contain `Speculative burst completed`, preventing a classic fallback
//!    from silently passing the parity assertion. The two servers run
//!    sequentially so only one target's worth of GPU memory is resident at a
//!    time.
//!
//! This is the load-bearing correctness gate: at `temperature = 0` the
//! speculative round loop MUST emit a token sequence byte-identical to
//! the drafter-less greedy pass through the target.
//!
//! ## Invocation
//!
//! ```bash
//! # Full set including ignored real-model tests:
//! cargo test --test speculative_parity --release -- --ignored --test-threads=1 --nocapture
//!
//! # Structural-only check (loads models, asserts trait wiring; no decode):
//! cargo test --test speculative_parity --release -- --test-threads=1 --nocapture
//! ```
//!
//! `--test-threads=1` is required because real-model tests share GPU memory
//! and concurrent loads will OOM on smaller (32-48 GB) Apple Silicon hosts.
//! The `#[ignore]`-gated real-model tests additionally spawn `mlxcel-server`
//! subprocesses; they are run by the CI hardware lane on a fixed cadence
//! and skipped by default on dev machines.

mod common;

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

/// Fixed prompt submitted to both the speculative and the drafter-less
/// server in the byte-equality phase. Deterministic content, no system
/// prompt, so the only variable between the two runs is whether the
/// drafter is attached.
const BYTE_EQUALITY_PROMPT: &str = "List the first five prime numbers.";

/// Decode budget for the byte-equality phase. Large enough that the
/// speculative round loop runs many draft/verify rounds (so a parity bug
/// has room to surface), small enough to keep the CI hardware lane fast.
const BYTE_EQUALITY_MAX_TOKENS: u32 = 96;

/// Reserve an ephemeral localhost port by binding and immediately
/// releasing it. There is a benign TOCTOU window before the server
/// rebinds; in practice the CI hardware lane runs these tests
/// `--test-threads=1` so nothing else races for the port.
fn reserve_port() -> u16 {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port for test server");
    let port = listener.local_addr().expect("read local addr").port();
    drop(listener);
    port
}

/// Spawn `mlxcel-server` with the given CLI args. stdout/stderr are captured
/// for diagnostics, while the structured server logs are written via
/// `--log-file` (added by [`server_round`]).
fn spawn_server(args: &[&str]) -> Child {
    Command::new(repo_binary_path("mlxcel-server"))
        .args(args)
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mlxcel-server binary")
}

/// Kill the server child and reap it so the next phase's server can bind
/// a fresh port without a lingering process holding GPU memory.
fn stop_server(child: &mut Child) -> String {
    let _ = child.kill();
    let mut captured = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = pipe.read_to_string(&mut captured);
    }
    if let Some(mut pipe) = child.stderr.take() {
        let _ = pipe.read_to_string(&mut captured);
    }
    let _ = child.wait();
    captured
}

/// Poll `/health` until the server is ready. Real-model loads (especially
/// the 31B Gemma 4 target) can take a while, so the deadline is generous.
async fn wait_for_health(client: &reqwest::Client, base_url: &str) {
    let deadline = Instant::now() + Duration::from_secs(300);
    while Instant::now() < deadline {
        if let Ok(response) = client.get(format!("{base_url}/health")).send().await
            && response.status().is_success()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("mlxcel-server did not become healthy at {base_url} within the deadline");
}

/// Submit [`BYTE_EQUALITY_PROMPT`] to `/v1/chat/completions` at
/// `temperature = 0` and return the `(message.content, usage.completion_tokens)`
/// pair. The completion-token count is part of the byte-equality
/// assertion: a parity bug that produces the same prefix but a different
/// length is still a regression.
async fn chat_content_and_token_count(client: &reqwest::Client, base_url: &str) -> (String, u64) {
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "speculative-parity",
            "messages": [{"role": "user", "content": BYTE_EQUALITY_PROMPT}],
            "max_tokens": BYTE_EQUALITY_MAX_TOKENS,
            "temperature": 0.0,
        }))
        .send()
        .await
        .expect("send /v1/chat/completions request");
    assert!(
        resp.status().is_success(),
        "/v1/chat/completions returned non-success status {}",
        resp.status(),
    );
    let body: serde_json::Value = resp.json().await.expect("parse chat completion response");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("chat response must carry choices[0].message.content")
        .to_string();
    let completion_tokens = body["usage"]["completion_tokens"]
        .as_u64()
        .expect("chat response must carry usage.completion_tokens");
    (content, completion_tokens)
}

struct ServerRoundOutput {
    content: String,
    completion_tokens: u64,
    logs: String,
}

/// Run one server-spawn round: bring `mlxcel-server` up with `extra_args`
/// appended to the shared base flags, wait for health, submit the fixed
/// prompt, tear the server down, and return the
/// [`ServerRoundOutput`] bundle.
///
/// The server is always stopped before this function returns (including
/// on the panic paths inside the helpers it calls — the `Child` is owned
/// locally and `stop_server` runs before the value is asserted on), so
/// the next round can reuse the GPU without two targets resident at once.
async fn server_round(
    pairing_name: &str,
    label: &str,
    target_path: &std::path::Path,
    extra_args: &[&str],
) -> ServerRoundOutput {
    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();
    let target_str = target_path.to_string_lossy().to_string();
    let log_file = tempfile::Builder::new()
        .prefix("mlxcel-speculative-parity-")
        .suffix(".log")
        .tempfile()
        .expect("create temporary server log file");
    let log_path = log_file.path().to_path_buf();
    let log_path_str = log_path.to_string_lossy().to_string();

    let mut args: Vec<&str> = vec![
        "--model",
        &target_str,
        "--host",
        "127.0.0.1",
        "--port",
        &port_str,
        "--no-warmup",
        "--log-file",
        &log_path_str,
    ];
    args.extend_from_slice(extra_args);

    eprintln!("[{pairing_name}] {label}: spawning mlxcel-server with args {args:?}");
    let mut child = spawn_server(&args);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;
    eprintln!("[{pairing_name}] {label}: server healthy, submitting fixed prompt");

    let result = chat_content_and_token_count(&client, &base_url).await;

    let process_logs = stop_server(&mut child);
    let file_logs = std::fs::read_to_string(&log_path).unwrap_or_default();
    let logs = if process_logs.is_empty() {
        file_logs
    } else {
        format!("{file_logs}\n--- process stdout/stderr ---\n{process_logs}")
    };
    eprintln!(
        "[{pairing_name}] {label}: completion_tokens={}, content.len()={}",
        result.1,
        result.0.len(),
    );
    ServerRoundOutput {
        content: result.0,
        completion_tokens: result.1,
        logs,
    }
}

/// byte-equality phase: spawn `mlxcel-server` twice against the
/// same `target_path` — once speculative (drafter attached via
/// `--model-draft --draft-kind --draft-block-size`), once drafter-less —
/// and assert the `/v1/chat/completions` responses are byte-identical.
///
/// The two servers run **sequentially** (the speculative one is fully
/// stopped before the baseline one starts) so a 32-48 GB Apple Silicon
/// host only ever holds one target's worth of weights at a time.
async fn assert_server_byte_equality(
    pairing: &Pairing,
    target_path: &std::path::Path,
    draft_path: &std::path::Path,
) {
    let draft_str = draft_path.to_string_lossy().to_string();
    let block_size_str = pairing.block_size.to_string();

    // Round 1: speculative — drafter attached.
    let spec = server_round(
        pairing.name,
        "speculative",
        target_path,
        &[
            "--model-draft",
            &draft_str,
            "--draft-kind",
            pairing.kind,
            "--draft-block-size",
            &block_size_str,
        ],
    )
    .await;

    assert!(
        spec.logs.contains("Speculative burst completed"),
        "[{}] speculative server logs did not contain the burst completion marker; \
         this usually means the request fell back to classic decode and the parity \
         assertion would be a false pass. Captured logs:\n{}",
        pairing.name,
        spec.logs,
    );
    if pairing.kind == "dflash" {
        assert!(
            !spec.logs.contains("DFlash speculative dispatch declined"),
            "[{}] DFlash server declined speculative dispatch during the parity run. \
             Captured logs:\n{}",
            pairing.name,
            spec.logs,
        );
    }

    // Round 2: drafter-less baseline — no `--draft-*` / `--model-draft`.
    let baseline = server_round(pairing.name, "baseline", target_path, &[]).await;

    // Byte-equality: at temperature = 0 the speculative round loop must
    // emit a token sequence byte-identical to the drafter-less greedy
    // pass. Compare both the decoded text and the completion-token count.
    assert_eq!(
        spec.completion_tokens, baseline.completion_tokens,
        "[{}] speculative completion_tokens ({}) != baseline ({}); \
         the speculative round loop diverged from the drafter-less greedy pass at temp=0",
        pairing.name, spec.completion_tokens, baseline.completion_tokens,
    );
    assert_eq!(
        spec.content, baseline.content,
        "[{}] speculative /v1/chat/completions content is NOT byte-identical to the \
         drafter-less baseline at temp=0 — this is a speculative-decoding parity bug.\n\
         speculative: {:?}\n\
         baseline:    {:?}",
        pairing.name, spec.content, baseline.content,
    );
    eprintln!(
        "[{}] byte-equality phase passed: speculative output is byte-identical to the \
         drafter-less baseline ({} completion tokens)",
        pairing.name, spec.completion_tokens,
    );
}

/// Reachable pairings whose drafter checkpoint is present on disk in the
/// canonical `models/` directory (resolved via [`repo_model_dir`]). The
/// test below short-circuits with an informative message when the
/// corresponding target / drafter directory is missing, mirroring the
/// pattern used by `tests/tensor_parallel_real_models.rs`.
struct Pairing {
    name: &'static str,
    target_dir: &'static str,
    draft_dir: &'static str,
    /// Drafter kind name as it appears in CLI flags (`--draft-kind`).
    kind: &'static str,
    /// Draft block length. MTP defaults to 4 (Gemma 4 assistant); DFlash
    /// defaults to 16 (Qwen 3.5 DFlash).
    block_size: u32,
}

const REACHABLE_PAIRINGS: &[Pairing] = &[
    Pairing {
        name: "Qwen 3.5 4B + DFlash (b=16)",
        target_dir: "qwen3.5-4b-4bit",
        draft_dir: "Qwen3.5-4B-DFlash",
        kind: "dflash",
        block_size: 16,
    },
    Pairing {
        name: "Gemma 4 31B + MTP assistant (b=4)",
        target_dir: "gemma-4-31b-it-4bit",
        draft_dir: "gemma-4-31B-it-assistant-bf16",
        kind: "mtp",
        block_size: 4,
    },
    Pairing {
        name: "Gemma 4 Unified 12B + MTP assistant (b=4)",
        target_dir: "gemma-4-12b-it-4bit",
        draft_dir: "gemma-4-12B-it-assistant-4bit",
        kind: "mtp",
        block_size: 4,
    },
];

/// Index of the Gemma 4 Unified 12B + MTP assistant pairing in
/// [`REACHABLE_PAIRINGS`]. Kept as a named constant so the unified parity test
/// stays correct if the pairing list is reordered.
const UNIFIED_12B_MTP_PAIRING: usize = 2;

/// Issue #203 jitter verifier: greedy byte-parity between a B > 1 batched
/// MTP run and its B = 1 references is only defined up to evaluation-path fp
/// jitter. Measured on M1 Ultra: B = 2 vs B = 1 forwards of IDENTICAL
/// content deviate ~1e-3 relative in hidden state (see
/// `divergent_hidden_probe_31b`'s LOCKSTEP control), and at repetition-loop
/// entropy cliffs even two B = 1 evaluation paths (incremental MTP cache vs
/// one-shot prefill) pick argmaxes ~1 logit apart. Structural defects (like
/// the pre-#203 position holes) shift the distribution by far more than this
/// margin and flip decisive positions, which the verifier rejects.
///
/// Margin calibration (M1 Ultra, gemma-4-31b-it-4bit): with the #203 fix the
/// observed jitter-flip top-gaps were 0.0–2.0; with the fix disabled
/// (`MLXCEL_MTP_DISABLE_DIVERGENT_FIX=1`) the structural break produced
/// first-mismatch top-gaps of 0.88–38 across rows, always exceeding the
/// margin on at least one row (row 3: 38 logits). The margin separates the
/// two regimes at gate level (every row must pass), not per row.
const EVAL_PATH_JITTER_MARGIN: f32 = 2.5;

/// Compute the one-shot B = 1 last-position logits for the prefix
/// `prompt + emitted[..i]` and return `(top - logit[want], top - logit[got],
/// argmax)`. Runs one fresh single-row prefill forward through the wrapper
/// (unique `seq_id` per call).
fn b1_top_gaps(
    wrapper: &mlxcel::models::Gemma4Wrapper,
    seq_raw: u64,
    prompt: &[i32],
    emitted_prefix: &[i32],
    want: i32,
    got: i32,
) -> (f32, f32, i32) {
    use mlxcel_core::generate::LanguageModel;
    let mut tokens: Vec<i32> = prompt.to_vec();
    tokens.extend_from_slice(emitted_prefix);
    let input = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);
    let seq_id = mlxcel_core::cache::SequenceId::from_raw(seq_raw);
    let logits = wrapper.forward_with_sequence_id(&input, Some(seq_id), &mut [], None);
    let shape = mlxcel_core::array_shape(&logits);
    let last = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last, 0], &[1, last + 1, shape[2]]);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    let vals: Vec<f32> = mlxcel_core::array_to_raw_bytes(&row)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (argmax, top) = vals
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i as i32, v))
        .unwrap_or((-1, f32::NAN));
    (top - vals[want as usize], top - vals[got as usize], argmax)
}

/// Apply the jitter-aware parity rule to one row's `(batched, reference)`
/// stream pair (issue #203). Byte-equal streams pass outright. Otherwise the
/// FIRST mismatch position is re-evaluated with a one-shot B = 1 forward over
/// the (shared) prefix; BOTH the batched and the reference token must sit
/// within [`EVAL_PATH_JITTER_MARGIN`] logits of that arbiter's top token,
/// proving the position is an entropy cliff where evaluation-path fp jitter
/// (incremental vs one-shot chunking, B = 1 vs B > 1 kernels, left-padding
/// RoPE shift; measured ~1 logit on M1 Ultra at repetition-loop boundaries,
/// where even two B = 1 paths disagree) legitimately flips the argmax. A
/// structural positional defect (like the pre-#203 hole) corrupts the
/// distribution by far more than jitter and fails this check; see the
/// kill-switch validation in the issue #203 PR. The tail beyond a legitimate
/// jitter flip follows a different but equally valid greedy chain and is not
/// compared.
#[allow(clippy::too_many_arguments)]
fn check_row_parity_with_near_tie(
    wrapper: &mlxcel::models::Gemma4Wrapper,
    label: &str,
    row: usize,
    seq_raw: u64,
    prompt: &[i32],
    batched: &[i32],
    reference: &[i32],
) -> Option<String> {
    let first_mismatch = batched
        .iter()
        .zip(reference.iter())
        .position(|(g, w)| g != w)
        .or_else(|| (batched.len() != reference.len()).then(|| batched.len().min(reference.len())));
    let Some(i) = first_mismatch else {
        eprintln!(
            "[{label}] row {row}: {} tokens byte-identical",
            batched.len()
        );
        return None;
    };
    let (Some(&got), Some(&want)) = (batched.get(i), reference.get(i)) else {
        return Some(format!(
            "row {row}: length mismatch without token mismatch (batched {} vs ref {}): \
             EOS handling drift",
            batched.len(),
            reference.len()
        ));
    };
    let (gap_want, gap_got, b1_argmax) =
        b1_top_gaps(wrapper, seq_raw, prompt, &reference[..i], want, got);
    eprintln!(
        "[{label}] row {row}: first mismatch at token {i} (batched {got} vs b1 {want}); \
         one-shot B=1 arbiter: argmax {b1_argmax}, top-gap(ref)={gap_want:e}, \
         top-gap(batched)={gap_got:e}"
    );
    if !(0.0..=EVAL_PATH_JITTER_MARGIN).contains(&gap_want)
        || !(0.0..=EVAL_PATH_JITTER_MARGIN).contains(&gap_got)
    {
        return Some(format!(
            "row {row}: mismatch at token {i} (batched {got} vs b1 {want}) is NOT \
             evaluation-path jitter (top-gaps ref {gap_want:e} / batched {gap_got:e} \
             exceed {EVAL_PATH_JITTER_MARGIN}): structural parity break"
        ));
    }
    None
}

/// Returns the target / drafter paths and whether both are present on disk.
fn pairing_present(pairing: &Pairing) -> (std::path::PathBuf, std::path::PathBuf, bool) {
    let target = repo_model_dir(pairing.target_dir);
    let draft = repo_model_dir(pairing.draft_dir);
    let present = target.exists() && draft.exists();
    (target, draft, present)
}

/// Structural pin: every reachable pairing's checkpoint layout matches what
/// the speculative-decoding loaders expect.
///
/// Specifically:
///   1. The target directory exists OR the test is cleanly skipped with a
///      log line so CI hosts without the models do not red-flag the build.
///   2. The drafter's `config.json` exists and the resolved drafter kind
///      via `mlxcel_core::drafter::resolve_drafter_kind` matches the
///      pairing's declared `kind` value.
///
/// This is the cheapest gate that catches "we silently picked the wrong
/// drafter shape" bugs at test time. The expensive byte-equality assertion
/// for the full round-loop pass is gated behind `#[ignore]` (see
/// `greedy_parity_dflash_qwen35_4b` / `greedy_parity_mtp_gemma4_31b`).
#[test]
fn pairing_kind_resolution_matches_declaration() {
    use mlxcel_core::drafter::{DrafterKind, resolve_drafter_kind};
    use std::str::FromStr as _;

    let mut checked = 0u32;
    for pairing in REACHABLE_PAIRINGS {
        let (_target, draft, present) = pairing_present(pairing);
        if !present {
            eprintln!(
                "Skipping pairing kind resolution for {}: drafter or target missing on disk",
                pairing.name,
            );
            continue;
        }

        let expected_kind = DrafterKind::from_str(pairing.kind)
            .expect("pairing kind string must parse to a known DrafterKind variant");
        let resolved_kind = resolve_drafter_kind(&draft, None)
            .expect("drafter config.json must be readable and parseable");

        assert_eq!(
            resolved_kind, expected_kind,
            "Pairing {} declared kind={:?} but auto-detect from {:?} resolved to {:?}; \
             check the drafter's config.json::model_type",
            pairing.name, expected_kind, draft, resolved_kind,
        );
        checked += 1;
    }

    // It is intentionally OK for `checked == 0` (no pairings on disk) — the
    // test logs that case and passes so CI hosts without the model
    // checkpoints don't fail the build. The reachable pairing discovery
    // test below documents that path explicitly.
    eprintln!(
        "pairing_kind_resolution_matches_declaration: checked {checked} \
         pairing(s) (out of {} reachable; skipped pairings are not on disk)",
        REACHABLE_PAIRINGS.len(),
    );
}

/// Greedy parity for the Qwen 3.5 4B + DFlash pairing.
///
/// **Status:** end-to-end byte-equality.
///
/// Two-phase test (see the module docstring):
///
/// 1. **Structural phase** (in-process): loads the target, asserts it is
///    a Qwen 3.5 family variant, resolves the drafter kind to
///    `DrafterKind::Dflash`, loads the DFlash drafter against the
///    published upstream `z-lab/Qwen3.5-4B-DFlash` checkpoint — which
///    omits `embed_tokens.weight` — and `bind()`s it to the target,
///    resolving `embed_tokens` lazily from the target (the pin). The in-process models are then dropped.
/// 2. **Byte-equality phase** (subprocess): spawns `mlxcel-server` with
///    `--model-draft --draft-kind dflash --draft-block-size 16` and again
///    without any `--draft-*` flag, submits the same fixed prompt to
///    `/v1/chat/completions` at `temperature = 0`, asserts the speculative
///    server logged `Speculative burst completed` (no silent fallback), and asserts the two responses are byte-identical.
///
/// `#[ignore]`-gated as real-model heavy: it loads the Qwen 3.5 4B target
/// + DFlash drafter and spawns `mlxcel-server` subprocesses. The CI
/// hardware lane runs it on a fixed cadence; dev runs skip it by default.
#[tokio::test]
#[ignore = "real-model heavy (loads Qwen3.5-4B target + drafter, spawns mlxcel-server); runs in CI hardware lane only"]
async fn greedy_parity_dflash_qwen35_4b() {
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, resolve_drafter_kind};

    let pairing = &REACHABLE_PAIRINGS[0];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {}: target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    // ---- Phase 1: structural check (in-process) ----
    {
        let _runtime = initialize_runtime();
        mlxcel_core::synchronize_default();
        mlxcel_core::clear_memory_cache();

        eprintln!("[{}] Loading target from {:?}", pairing.name, target_path);
        let (loaded_target, _target_tokenizer) =
            load_model(&target_path).expect("target model must load");

        // Qwen 3.5 4B checkpoints from `mlx-community` are published as
        // `Qwen3_5ForConditionalGeneration` (VLM variant) even for the
        // text-only 4B checkpoint, so the variant we expect is `Qwen35VLM`
        // / `Qwen35MoeVLM` for the VLM-wrapped variants and `Qwen35` /
        // `Qwen35Moe` for the pure text-only variants (less common in
        // `mlx-community`).
        let target_is_qwen35 = matches!(
            loaded_target,
            LoadedModel::Qwen35(_)
                | LoadedModel::Qwen35Moe(_)
                | LoadedModel::Qwen35VLM(_)
                | LoadedModel::Qwen35MoeVLM(_)
        );
        assert!(
            target_is_qwen35,
            "DFlash pairing requires a Qwen 3.5 target but load_model returned a different \
             variant; check the pairing target_dir matches a Qwen 3.5 4-bit checkpoint",
        );
        eprintln!(
            "[{}] Target loaded as Qwen 3.5 family (Qwen35Model / Qwen35VLModel)",
            pairing.name
        );

        eprintln!("[{}] Loading drafter from {:?}", pairing.name, draft_path);
        let resolved_kind =
            resolve_drafter_kind(&draft_path, None).expect("drafter config.json must be readable");
        assert_eq!(resolved_kind, DrafterKind::Dflash);

        // `load_drafter` constructs the full DFlashDrafter (weight loading
        // + sanitize). The upstream `z-lab/Qwen3.5-4B-DFlash` checkpoint
        // does NOT ship `embed_tokens.weight`; ported upstream's
        // lazy-bind shape so the loader builds an `embed_tokens = None`
        // tombstone instead of failing. A `LoadFailed` here is a
        // regression.
        let (mut drafter, drafter_kind) =
            mlxcel_core::drafter::load_drafter(&draft_path, Some(DrafterKind::Dflash)).expect(
                "DFlash drafter must load against the published z-lab/Qwen3.5-4B-DFlash \
                 checkpoint (embed_tokens.weight is absent and the loader \
                 builds a lazy-bind tombstone instead of failing)",
            );
        assert_eq!(drafter_kind, DrafterKind::Dflash);
        eprintln!(
            "[{}] Drafter loaded successfully; block_size={}",
            pairing.name, pairing.block_size,
        );

        // `bind` resolves the drafter's `embed_tokens` from the target's
        // embedding module (`LanguageModel::embed_tokens_module`, implemented by the Qwen 3.5 family —). A `BindFailed`
        // here means either the target does not expose
        // `embed_tokens_module` or the lazy-bind tombstone was not wired.
        drafter
            .bind(&loaded_target)
            .expect("DFlash drafter must bind to the Qwen 3.5 target (lazy-bind)");
        eprintln!(
            "[{}] Drafter bound to target; embed_tokens resolved via lazy-bind",
            pairing.name,
        );

        // Drop the in-process target + drafter and clear the MLX memory
        // cache before phase 2 so the spawned servers do not contend with
        // these weights for GPU memory.
        drop(drafter);
        drop(loaded_target);
        mlxcel_core::synchronize_default();
        mlxcel_core::clear_memory_cache();
    }

    // ---- Phase 2: end-to-end byte-equality (subprocess) ----
    assert_server_byte_equality(pairing, &target_path, &draft_path).await;
}

/// Greedy parity for the Gemma 4 31B + MTP assistant pairing.
///
/// **Status:** end-to-end byte-equality.
///
/// Two-phase test (see the module docstring):
///
/// 1. **Structural phase** (in-process): loads the target, asserts it is
///    a Gemma 4 family variant, resolves the drafter kind to
///    `DrafterKind::Mtp` (auto-detected from `model_type:
///    "gemma4_assistant"` in the drafter `config.json`), and loads the
///    MTP assistant drafter from its checkpoint. The in-process models
///    are then dropped.
/// 2. **Byte-equality phase** (subprocess): spawns `mlxcel-server` with
///    `--model-draft --draft-kind mtp --draft-block-size 4` and again
///    without any `--draft-*` flag, submits the same fixed prompt to
///    `/v1/chat/completions` at `temperature = 0`, and asserts the two
///    responses are byte-identical. The server-side MTP burst dispatch
///    and the `MtpTarget` adapter for `Gemma4Wrapper` were wired by
///
/// Note on dtype: this pairing uses a 4-bit quantized target with a bf16
/// drafter. The 4-bit target keeps bf16 scales/biases as-is per
/// `docs/apple-silicon-precision.md`; the drafter is dtype-agnostic as
/// long as the target's `forward_with_speculative_sinks` preserves the
/// captured hidden + shared K/V slab dtypes (which it does — see the
/// "preserves the model's native bf16/f16 dtype — no f32 promotion" note
/// on `Gemma4SpeculativeSinks`).
///
/// `#[ignore]`-gated as real-model heavy: it loads the Gemma 4 31B target
/// + MTP drafter and spawns `mlxcel-server` subprocesses. The CI hardware
/// lane runs it on a fixed cadence; dev runs skip it by default.
#[tokio::test]
#[ignore = "real-model heavy (loads Gemma-4-31B target + drafter, spawns mlxcel-server); runs in CI hardware lane only"]
async fn greedy_parity_mtp_gemma4_31b() {
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, resolve_drafter_kind};

    let pairing = &REACHABLE_PAIRINGS[1];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {}: target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    // ---- Phase 1: structural check (in-process) ----
    {
        let _runtime = initialize_runtime();
        mlxcel_core::synchronize_default();
        mlxcel_core::clear_memory_cache();

        eprintln!("[{}] Loading target from {:?}", pairing.name, target_path);
        let (loaded_target, _target_tokenizer) =
            load_model(&target_path).expect("target model must load");

        // Gemma 4 31B checkpoints are typically published as
        // `Gemma4ForConditionalGeneration` (VLM variant), so we accept
        // either the text-only `Gemma4` wrapper or the `Gemma4VLM` VLM
        // wrapper. The MTP path operates on the inner text model in both
        // cases (the vision tower is bypassed when no image tokens are
        // present in the prompt).
        let target_is_gemma4 = matches!(
            loaded_target,
            LoadedModel::Gemma4(_) | LoadedModel::Gemma4VLM(_)
        );
        assert!(
            target_is_gemma4,
            "MTP pairing requires a Gemma 4 target but load_model returned a different variant; \
             check the pairing target_dir matches a Gemma 4 4-bit checkpoint",
        );
        eprintln!(
            "[{}] Target loaded as Gemma 4 family (Gemma4Wrapper / Gemma4VLModel)",
            pairing.name
        );

        eprintln!("[{}] Loading drafter from {:?}", pairing.name, draft_path);
        let resolved_kind =
            resolve_drafter_kind(&draft_path, None).expect("drafter config.json must be readable");
        assert_eq!(resolved_kind, DrafterKind::Mtp);

        let (drafter, _kind) =
            mlxcel_core::drafter::load_drafter(&draft_path, Some(DrafterKind::Mtp))
                .expect("Gemma 4 MTP assistant drafter must load from checkpoint");
        eprintln!(
            "[{}] Drafter loaded; block_size={}",
            pairing.name, pairing.block_size
        );

        // Drop the in-process target + drafter and clear the MLX memory
        // cache before phase 2 so the spawned servers do not contend with
        // these weights for GPU memory.
        drop(drafter);
        drop(loaded_target);
        mlxcel_core::synchronize_default();
        mlxcel_core::clear_memory_cache();
    }

    // ---- Phase 2: end-to-end byte-equality (subprocess) ----
    assert_server_byte_equality(pairing, &target_path, &draft_path).await;
}

/// Greedy parity for the Gemma 4 Unified 12B + MTP assistant pairing
/// (issue #154).
///
/// **Status:** end-to-end byte-equality on real models.
///
/// Identical two-phase structure to [`greedy_parity_mtp_gemma4_31b`], but the
/// target is the `gemma4_unified` 12B checkpoint (loads as
/// [`mlxcel::LoadedModel::Gemma4Unified`]) and the drafter is the
/// `gemma4_unified_assistant` 12B assistant (`backbone_hidden_size = 3840`,
/// centroid LM head). The MTP burst dispatch routes `Gemma4Unified` through
/// `Gemma4UnifiedMtpTargetAdapter` → the inner `Gemma4MtpTargetAdapter` over
/// `unified.text_model`, exactly as the VLM path routes through the VL adapter.
///
/// 1. **Structural phase** (in-process): loads the target, asserts it is the
///    `Gemma4Unified` variant, resolves the drafter kind to `DrafterKind::Mtp`
///    (auto-detected from `model_type: "gemma4_unified_assistant"`), and loads
///    the assistant drafter from its checkpoint.
/// 2. **Byte-equality phase** (subprocess): spawns `mlxcel-server` with
///    `--model-draft --draft-kind mtp --draft-block-size 4` and again without
///    any `--draft-*` flag, submits the same fixed prompt at `temperature = 0`,
///    and asserts the two responses are byte-identical (MTP speculative decode
///    is exactness-preserving).
///
/// `#[ignore]`-gated as real-model heavy: loads the Gemma 4 Unified 12B target
/// + assistant drafter and spawns `mlxcel-server` subprocesses. Runs in the CI
/// hardware lane / the orchestrator's real-model measurement gate.
#[tokio::test]
#[ignore = "real-model heavy (loads Gemma-4-Unified-12B target + drafter, spawns mlxcel-server); runs in CI hardware lane only"]
async fn greedy_parity_mtp_gemma4_unified_12b() {
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, resolve_drafter_kind};

    let pairing = &REACHABLE_PAIRINGS[UNIFIED_12B_MTP_PAIRING];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {}: target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    // ---- Phase 1: structural check (in-process) ----
    {
        let _runtime = initialize_runtime();
        mlxcel_core::synchronize_default();
        mlxcel_core::clear_memory_cache();

        eprintln!("[{}] Loading target from {:?}", pairing.name, target_path);
        let (loaded_target, _target_tokenizer) =
            load_model(&target_path).expect("target model must load");

        // The 12B Unified checkpoint loads as the encoder-free
        // `Gemma4Unified` variant; the MTP path operates on the inner text
        // model (multimodal placeholders are absent for a text prompt).
        assert!(
            matches!(loaded_target, LoadedModel::Gemma4Unified(_)),
            "Unified MTP pairing requires a Gemma4Unified target but load_model returned a \
             different variant; check the pairing target_dir matches the gemma4_unified \
             12B 4-bit checkpoint",
        );
        eprintln!("[{}] Target loaded as Gemma4Unified", pairing.name);

        eprintln!("[{}] Loading drafter from {:?}", pairing.name, draft_path);
        let resolved_kind =
            resolve_drafter_kind(&draft_path, None).expect("drafter config.json must be readable");
        assert_eq!(
            resolved_kind,
            DrafterKind::Mtp,
            "gemma4_unified_assistant must auto-detect to the MTP round loop",
        );

        let (drafter, _kind) =
            mlxcel_core::drafter::load_drafter(&draft_path, Some(DrafterKind::Mtp))
                .expect("Gemma 4 Unified MTP assistant drafter must load from checkpoint");
        eprintln!(
            "[{}] Drafter loaded; block_size={}",
            pairing.name, pairing.block_size
        );

        drop(drafter);
        drop(loaded_target);
        mlxcel_core::synchronize_default();
        mlxcel_core::clear_memory_cache();
    }

    // ---- Phase 2: end-to-end byte-equality (subprocess) ----
    assert_server_byte_equality(pairing, &target_path, &draft_path).await;
}

/// acceptance item 1: a B = 4 batched MTP burst produces
/// per-row token streams that are **byte-identical** to running the
/// B = 1 MTP burst in isolation on each row's prompt.
///
/// This is the load-bearing correctness gate for the batched MTP
/// dispatch. The batched MTP target adapter
/// (`Gemma4MtpBatchedTargetAdapter`) forwards the `[B, L]` prompt batch
/// in one pass; with equal-length prompts the 2-D causal masks broadcast
/// cleanly across the batch, so every row's verify forward sees exactly
/// the logits it would see in an isolated B = 1 run. The test drives:
///
/// 1. Four B = 1 runs via `MtpGenerator` + `Gemma4MtpTargetAdapter`,
///    one per row's prompt — the per-row reference streams.
/// 2. One B = 4 run via `MtpBatchedGenerator` +
///    `Gemma4MtpBatchedTargetAdapter` over all four prompts.
///
/// and asserts each batched row equals its B = 1 reference token-for-token.
///
/// Gated `#[ignore]` (real-model heavy: loads the Gemma-4-31B target +
/// bf16 drafter, which exceeds the dev-machine stream-idle watchdog) —
/// runs in the CI hardware lane.
#[test]
#[ignore = "real-model heavy (Gemma-4-31B target + drafter, B=4 batched run); CI hardware lane only"]
fn greedy_parity_mtp_gemma4_batched_b4_matches_b1() {
    use mlxcel::models::gemma4_mtp_target::{
        Gemma4MtpBatchedTargetAdapter, Gemma4MtpTargetAdapter,
    };
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, load_drafter};
    use mlxcel_core::generate::SamplingConfig;
    use mlxcel_core::speculative::mtp::{MtpBatchedGenerator, MtpGenerator};

    let pairing = &REACHABLE_PAIRINGS[1];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {} (batched B=4): target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (loaded_target, _tok) = load_model(&target_path).expect("target model must load");
    // The batched MTP adapter binds against the text-only `Gemma4Wrapper`.
    // The 31B checkpoints can be published as either the text-only or VLM
    // variant; resolve the inner text wrapper for both.
    let wrapper: &mlxcel::models::Gemma4Wrapper = match &loaded_target {
        LoadedModel::Gemma4(w) => w,
        LoadedModel::Gemma4VLM(vlm) => &vlm.text_model,
        _ => panic!("MTP batched pairing requires a Gemma 4 family target"),
    };

    let block_size = pairing.block_size as usize;
    let sampling = SamplingConfig::greedy();
    let max_tokens = 24_usize;

    // Equal-length prompts (the batched MTP adapter requires a
    // rectangular [B, L] prefill — see its docstring). Four distinct
    // 6-token prompts.
    let prompts: Vec<Vec<i32>> = vec![
        vec![2, 105, 2364, 107, 9259, 108],
        vec![2, 105, 2364, 107, 1596, 108],
        vec![2, 105, 2364, 107, 6176, 108],
        vec![2, 105, 2364, 107, 3030, 108],
    ];

    // ---- B = 1 reference runs ----
    let mut reference: Vec<Vec<i32>> = Vec::with_capacity(prompts.len());
    for (row, prompt) in prompts.iter().enumerate() {
        // Each B=1 run gets a fresh per-sequence cache slot via a unique
        // SequenceId so the runs do not alias each other's KV cache.
        let seq_id = mlxcel_core::cache::SequenceId::from_raw(1000 + row as u64);
        let adapter = Gemma4MtpTargetAdapter::new(wrapper, Some(seq_id));
        let (mut drafter, kind) =
            load_drafter(&draft_path, Some(DrafterKind::Mtp)).expect("MTP drafter must load");
        assert_eq!(kind, DrafterKind::Mtp);
        drafter
            .bind(wrapper as &dyn mlxcel_core::generate::LanguageModel)
            .expect("drafter bind");
        let mut generator = MtpGenerator::new(adapter, drafter, block_size);
        // Offline test: greedy sampling needs no token history (issue
        //'s `token_history` parameter is `&[]` for a config where
        // `needs_token_history()` is false), no cooperative
        // cancellation (added the `cancel` parameter; the offline path passes a never-set flag, matching `src/commands/generate.rs`), and no logprobs (added the `logprobs_config` parameter; this byte-parity test compares only token streams, so a disabled-default `LogprobsConfig` keeps the zero-overhead path).
        let no_cancel = std::sync::atomic::AtomicBool::new(false);
        let logprobs_config = mlxcel_core::sampling::LogprobsConfig::default();
        let (tokens, _logprobs, _stats) = generator.generate(
            prompt,
            max_tokens,
            &sampling,
            &[],
            &no_cancel,
            &logprobs_config,
        );
        eprintln!(
            "[batched B=4] B=1 reference row {row}: {} tokens",
            tokens.len()
        );
        reference.push(tokens);
    }

    // ---- B = 4 batched run ----
    let batch_adapter = Gemma4MtpBatchedTargetAdapter::new(wrapper, prompts.len());
    let (mut batched_drafter, _kind) = load_drafter(&draft_path, Some(DrafterKind::Mtp))
        .expect("MTP drafter must load for the batched run");
    batched_drafter
        .bind(wrapper as &dyn mlxcel_core::generate::LanguageModel)
        .expect("batched drafter bind");
    let mut batched_generator =
        MtpBatchedGenerator::new(batch_adapter, batched_drafter, block_size);
    let run = batched_generator
        .run_batched(&prompts, &sampling, max_tokens)
        .expect("batched MTP run must succeed");

    // ---- Byte-equality assertion ----
    assert_eq!(
        run.tokens.len(),
        reference.len(),
        "batched run must produce one token stream per row"
    );
    // Parity rule (issue #203): byte-equality wherever the model decides
    // decisively; a first mismatch is acceptable only if the B = 1 logits
    // prove it a near-tie (batch-size-dependent kernel fp noise, measured in
    // `divergent_hidden_probe_31b`, flips only near-ties; a structural
    // positional defect flips decisive tokens and still fails this gate).
    let mut failures: Vec<String> = Vec::new();
    for (row, (batched_row, reference_row)) in run.tokens.iter().zip(reference.iter()).enumerate() {
        eprintln!(
            "[batched B=4] row {row}: accept_lens {:?}\n  batched ({}): {:?}\n  b1 ref  ({}): {:?}",
            run.accept_lens[row],
            batched_row.len(),
            batched_row,
            reference_row.len(),
            reference_row,
        );
        if let Some(err) = check_row_parity_with_near_tie(
            wrapper,
            "batched B=4",
            row,
            4000 + row as u64,
            &prompts[row],
            batched_row,
            reference_row,
        ) {
            failures.push(err);
        }
    }
    assert!(
        failures.is_empty(),
        "greedy-parity violation in the batched MTP dispatch:\n{}",
        failures.join("\n")
    );
    eprintln!(
        "[batched B=4] PASS: all {} rows match B=1 isolated bursts (byte-identical or \
         verified near-tie deviation)",
        run.tokens.len()
    );
}

/// Greedy-parity gate for a VARIABLE-length (ragged) B = 4 batched MTP burst:
/// each row is left-padded to the max prompt length, so the adapter
/// auto-routes to the ragged left-padding prefill (the
/// `prefill_and_seed_batched` length-uniformity check).
///
/// Asserts FULL-STREAM parity against each row's isolated B = 1 stream,
/// near-tie aware (see [`check_row_parity_with_near_tie`]). Issue #163 made
/// the lockstep prefix structurally exact via NaN-safe padding rows and
/// stale-tail exclusion; issue #203 extends that to the whole stream by
/// compacting each row's post-rollback position hole and rotating/masking
/// divergent verify rounds at per-row logical positions, so the
/// lockstep-prefix-only relaxation is gone. Residual batch-size/left-padding
/// kernel fp noise may still flip a NEAR-TIE argmax (hardware-dependent);
/// the gate verifies any first mismatch is such a near-tie via B = 1 logits.
///
/// Gated `#[ignore]` (real-model heavy: loads the Gemma-4-31B target + bf16
/// drafter); runs in the CI hardware lane.
#[test]
#[ignore = "real-model heavy (Gemma-4-31B target + drafter, ragged B=4 batched run); CI hardware lane only"]
fn greedy_parity_mtp_gemma4_batched_b4_ragged_matches_b1() {
    use mlxcel::models::gemma4_mtp_target::{
        Gemma4MtpBatchedTargetAdapter, Gemma4MtpTargetAdapter,
    };
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, load_drafter};
    use mlxcel_core::generate::SamplingConfig;
    use mlxcel_core::speculative::mtp::{MtpBatchedGenerator, MtpGenerator};

    let pairing = &REACHABLE_PAIRINGS[1];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {} (ragged B=4): target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (loaded_target, _tok) = load_model(&target_path).expect("target model must load");
    let wrapper: &mlxcel::models::Gemma4Wrapper = match &loaded_target {
        LoadedModel::Gemma4(w) => w,
        LoadedModel::Gemma4VLM(vlm) => &vlm.text_model,
        _ => panic!("MTP batched pairing requires a Gemma 4 family target"),
    };

    let block_size = pairing.block_size as usize;
    let sampling = SamplingConfig::greedy();
    let max_tokens = 24_usize;

    // DIFFERENT-length prompts (lengths 6/7/9/12) so the batched adapter
    // auto-routes to the ragged left-padding prefill. Same chat-template token
    // style as the equal-length B=4 test, extended with distinct content ids.
    let prompts: Vec<Vec<i32>> = vec![
        vec![2, 105, 2364, 107, 9259, 108],
        vec![2, 105, 2364, 107, 9259, 1596, 108],
        vec![2, 105, 2364, 107, 9259, 1596, 6176, 3030, 108],
        vec![
            2, 105, 2364, 107, 9259, 1596, 6176, 3030, 4711, 1234, 5678, 108,
        ],
    ];

    // ---- B = 1 reference runs (one per row, unique SequenceId) ----
    let mut reference: Vec<Vec<i32>> = Vec::with_capacity(prompts.len());
    for (row, prompt) in prompts.iter().enumerate() {
        let seq_id = mlxcel_core::cache::SequenceId::from_raw(2000 + row as u64);
        let adapter = Gemma4MtpTargetAdapter::new(wrapper, Some(seq_id));
        let (mut drafter, kind) =
            load_drafter(&draft_path, Some(DrafterKind::Mtp)).expect("MTP drafter must load");
        assert_eq!(kind, DrafterKind::Mtp);
        drafter
            .bind(wrapper as &dyn mlxcel_core::generate::LanguageModel)
            .expect("drafter bind");
        let mut generator = MtpGenerator::new(adapter, drafter, block_size);
        let no_cancel = std::sync::atomic::AtomicBool::new(false);
        let logprobs_config = mlxcel_core::sampling::LogprobsConfig::default();
        let (tokens, _logprobs, _stats) = generator.generate(
            prompt,
            max_tokens,
            &sampling,
            &[],
            &no_cancel,
            &logprobs_config,
        );
        eprintln!(
            "[ragged B=4] B=1 reference row {row} (prompt_len={}): {} tokens",
            prompt.len(),
            tokens.len()
        );
        reference.push(tokens);
    }

    // ---- ragged B = 4 batched run ----
    let batch_adapter = Gemma4MtpBatchedTargetAdapter::new(wrapper, prompts.len());
    let (mut batched_drafter, _kind) = load_drafter(&draft_path, Some(DrafterKind::Mtp))
        .expect("MTP drafter must load for the batched run");
    batched_drafter
        .bind(wrapper as &dyn mlxcel_core::generate::LanguageModel)
        .expect("batched drafter bind");
    let mut batched_generator =
        MtpBatchedGenerator::new(batch_adapter, batched_drafter, block_size);
    let run = batched_generator
        .run_batched(&prompts, &sampling, max_tokens)
        .expect("ragged batched MTP run must succeed");

    // ---- Full-stream byte-equality assertions (per row) ----
    //
    // Issue #203's divergent-round compaction + per-row logical RoPE/masks
    // make every row structurally identical to its standalone B = 1 run, so
    // the gate asserts the WHOLE stream (the former lockstep-prefix-only
    // relaxation is removed per the #203 acceptance criteria).
    assert_eq!(
        run.tokens.len(),
        reference.len(),
        "ragged batched run must produce one token stream per row"
    );
    let batch = run.tokens.len();
    // Same near-tie-aware parity rule as the equal-length gate. The ragged
    // path additionally carries the constant left-padding RoPE shift, which
    // is exact in real arithmetic but adds its own bf16 noise on top of the
    // batch-size noise; both only ever flip near-ties.
    let mut failures: Vec<String> = Vec::new();
    for row in 0..batch {
        let batched_row = &run.tokens[row];
        let reference_row = &reference[row];
        assert!(
            !batched_row.is_empty(),
            "row {row}: ragged batched run emitted no tokens"
        );
        // The prefill bonus comes from a uniform-phase forward and doubles as
        // the NaN canary: a NaN-poisoned ragged prefill (pre-#163 main on M1
        // Ultra) degenerates it to token id 0 instead of the B = 1 token.
        assert_eq!(
            batched_row[0], reference_row[0],
            "row {row} prefill bonus: ragged B=4 ({}) != B=1 isolated ({}); \
             NaN-safe ragged prefill regression",
            batched_row[0], reference_row[0],
        );
        eprintln!(
            "[ragged B=4] row {row}: accept_lens {:?}\n  batched ({}): {:?}\n  b1 ref  ({}): {:?}",
            run.accept_lens[row],
            batched_row.len(),
            batched_row,
            reference_row.len(),
            reference_row,
        );
        if let Some(err) = check_row_parity_with_near_tie(
            wrapper,
            "ragged B=4",
            row,
            5000 + row as u64,
            &prompts[row],
            batched_row,
            reference_row,
        ) {
            failures.push(err);
        }
    }
    assert!(
        failures.is_empty(),
        "greedy-parity violation after divergent accepts:\n{}",
        failures.join("\n")
    );
    eprintln!(
        "[ragged B=4] PASS: all {batch} variable-length rows match their B=1 isolated \
         bursts over the FULL stream (byte-identical or verified jitter; issue #203)"
    );
}

/// Throughput study plumbing for the issue #163 default-on evaluation (item 4).
/// Measures, on the real 31B, (a) serial B=1 over variable-length prompts, (b)
/// one ragged B=4 batched run over the same prompts, and (c) a classic
/// equal-length control (B=4 batched and serial B=1) for the same-length
/// speedup reference. Prints tok/s for each leg plus the ragged/serial and
/// classic/serial ratios; asserts only run success. The orchestrator runs this
/// on local hardware and records the study in the PR / issue. Changes no
/// defaults or env-var semantics.
#[test]
#[ignore = "real-model heavy throughput probe; run manually with --nocapture"]
fn mtp_gemma4_ragged_throughput_probe() {
    use mlxcel::models::gemma4_mtp_target::{
        Gemma4MtpBatchedTargetAdapter, Gemma4MtpTargetAdapter,
    };
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, load_drafter};
    use mlxcel_core::generate::{LanguageModel, SamplingConfig};
    use mlxcel_core::speculative::mtp::{MtpBatchedGenerator, MtpGenerator};
    use std::time::Instant;

    let pairing = &REACHABLE_PAIRINGS[1];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {} (throughput probe): target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (loaded_target, _tok) = load_model(&target_path).expect("target model must load");
    let wrapper: &mlxcel::models::Gemma4Wrapper = match &loaded_target {
        LoadedModel::Gemma4(w) => w,
        LoadedModel::Gemma4VLM(vlm) => &vlm.text_model,
        _ => panic!("MTP batched pairing requires a Gemma 4 family target"),
    };

    let block_size = pairing.block_size as usize;
    let sampling = SamplingConfig::greedy();
    let max_tokens = 48_usize;

    // Variable-length prompts (6/8/10/12) for the ragged-vs-serial comparison.
    let ragged_prompts: Vec<Vec<i32>> = vec![
        vec![2, 105, 2364, 107, 9259, 108],
        vec![2, 105, 2364, 107, 9259, 1596, 6176, 108],
        vec![2, 105, 2364, 107, 9259, 1596, 6176, 3030, 4711, 108],
        vec![
            2, 105, 2364, 107, 9259, 1596, 6176, 3030, 4711, 1234, 5678, 108,
        ],
    ];
    // Equal-length control prompts (all length 8) for the classic same-length
    // batched-vs-serial speedup reference.
    let classic_prompts: Vec<Vec<i32>> = vec![
        vec![2, 105, 2364, 107, 9259, 1596, 6176, 108],
        vec![2, 105, 2364, 107, 1596, 6176, 3030, 108],
        vec![2, 105, 2364, 107, 6176, 3030, 4711, 108],
        vec![2, 105, 2364, 107, 3030, 4711, 1234, 108],
    ];

    let no_cancel = std::sync::atomic::AtomicBool::new(false);
    let logprobs_config = mlxcel_core::sampling::LogprobsConfig::default();

    // Serial B=1 over a prompt set; returns (total_generated_tokens, seconds).
    let serial_b1 = |prompts: &[Vec<i32>], seq_base: u64| -> (usize, f64) {
        let start = Instant::now();
        let mut total = 0_usize;
        for (row, prompt) in prompts.iter().enumerate() {
            let seq_id = mlxcel_core::cache::SequenceId::from_raw(seq_base + row as u64);
            let adapter = Gemma4MtpTargetAdapter::new(wrapper, Some(seq_id));
            let (mut drafter, _kind) =
                load_drafter(&draft_path, Some(DrafterKind::Mtp)).expect("MTP drafter must load");
            drafter
                .bind(wrapper as &dyn LanguageModel)
                .expect("drafter bind");
            let mut generator = MtpGenerator::new(adapter, drafter, block_size);
            let (tokens, _lp, _stats) = generator.generate(
                prompt,
                max_tokens,
                &sampling,
                &[],
                &no_cancel,
                &logprobs_config,
            );
            total += tokens.len();
        }
        (total, start.elapsed().as_secs_f64())
    };

    // Batched B=4 over a prompt set; returns (total_generated_tokens, seconds).
    let batched_b4 = |prompts: &[Vec<i32>]| -> (usize, f64) {
        let batch_adapter = Gemma4MtpBatchedTargetAdapter::new(wrapper, prompts.len());
        let (mut batched_drafter, _kind) =
            load_drafter(&draft_path, Some(DrafterKind::Mtp)).expect("MTP drafter must load");
        batched_drafter
            .bind(wrapper as &dyn LanguageModel)
            .expect("batched drafter bind");
        let mut generator = MtpBatchedGenerator::new(batch_adapter, batched_drafter, block_size);
        let start = Instant::now();
        let run = generator
            .run_batched(prompts, &sampling, max_tokens)
            .expect("batched MTP run must succeed");
        let total: usize = run.tokens.iter().map(Vec::len).sum();
        (total, start.elapsed().as_secs_f64())
    };

    let eps = f64::EPSILON;
    // (a) serial B=1 baseline over the variable-length prompts.
    let (serial_tokens, serial_secs) = serial_b1(&ragged_prompts, 3000);
    let serial_tps = serial_tokens as f64 / serial_secs.max(eps);
    // (b) ragged B=4 over the same variable-length prompts.
    let (ragged_tokens, ragged_secs) = batched_b4(&ragged_prompts);
    let ragged_tps = ragged_tokens as f64 / ragged_secs.max(eps);
    // (c) classic equal-length control: B=4 batched and serial B=1.
    let (classic_b4_tokens, classic_b4_secs) = batched_b4(&classic_prompts);
    let classic_b4_tps = classic_b4_tokens as f64 / classic_b4_secs.max(eps);
    let (classic_serial_tokens, classic_serial_secs) = serial_b1(&classic_prompts, 4000);
    let classic_serial_tps = classic_serial_tokens as f64 / classic_serial_secs.max(eps);

    eprintln!("[throughput] === Gemma 4 31B MTP ragged throughput probe ===");
    eprintln!(
        "[throughput] (a) serial B=1 (variable len): {serial_tokens} toks / {serial_secs:.3}s = {serial_tps:.1} tok/s",
    );
    eprintln!(
        "[throughput] (b) ragged B=4 (variable len): {ragged_tokens} toks / {ragged_secs:.3}s = {ragged_tps:.1} tok/s",
    );
    eprintln!(
        "[throughput] (c) classic B=4 (equal len):   {classic_b4_tokens} toks / {classic_b4_secs:.3}s = {classic_b4_tps:.1} tok/s",
    );
    eprintln!(
        "[throughput] (c) classic serial B=1 (equal len): {classic_serial_tokens} toks / {classic_serial_secs:.3}s = {classic_serial_tps:.1} tok/s",
    );
    eprintln!(
        "[throughput] ratio ragged/serial = {:.3}x; classic B=4/serial = {:.3}x",
        ragged_tps / serial_tps.max(eps),
        classic_b4_tps / classic_serial_tps.max(eps),
    );
}

/// Sanity that the test discovery against `models/` finds at least one of
/// the reachable pairings on hosts that have downloaded the checkpoints,
/// and cleanly skips with a log line on hosts that have not.
#[test]
fn reachable_pairing_discovery_reports_status() {
    let mut any_present = false;
    for pairing in REACHABLE_PAIRINGS {
        let (target_path, draft_path, present) = pairing_present(pairing);
        eprintln!(
            "  - {}: target={:?} draft={:?} present={}",
            pairing.name, target_path, draft_path, present,
        );
        any_present |= present;
    }
    if any_present {
        eprintln!("At least one reachable speculative pairing is present on disk.");
    } else {
        eprintln!(
            "No reachable speculative pairings on disk — skipping perf benchmarks. \
             Run `mlxcel download mlx-community/Qwen3.5-4B-DFlash` (and similar) to populate.",
        );
    }
}

/// Static check: the reachable pairing list cannot accidentally be empty
/// (a future cleanup that drops every reachable pairing should fail loudly
/// at compile time rather than silently making the whole file a no-op).
const _: () = {
    assert!(
        !REACHABLE_PAIRINGS.is_empty(),
        "REACHABLE_PAIRINGS must declare at least one pairing; \
         see docs/model_tests.md::Speculative drafters",
    );
};

/// Issue #203 debugging probe: run each equal-length gate prompt through the
/// BATCHED adapter at batch size 1 (a configuration that can never diverge,
/// so the whole run stays on the uniform path) and compare against the B = 1
/// adapter reference. Any mismatch here implicates the batched adapter's
/// baseline mechanics rather than the divergent-round geometry.
#[test]
#[ignore = "real-model heavy diagnostic (Gemma-4-31B target + drafter)"]
fn b1_batched_baseline_probe() {
    use mlxcel::models::gemma4_mtp_target::{
        Gemma4MtpBatchedTargetAdapter, Gemma4MtpTargetAdapter,
    };
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, load_drafter};
    use mlxcel_core::generate::SamplingConfig;
    use mlxcel_core::speculative::mtp::{MtpBatchedGenerator, MtpGenerator};

    let pairing = &REACHABLE_PAIRINGS[1];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!("Skipping b1_batched_baseline_probe");
        return;
    }
    let _runtime = initialize_runtime();
    let (loaded_target, _tok) = load_model(&target_path).expect("target model must load");
    let wrapper: &mlxcel::models::Gemma4Wrapper = match &loaded_target {
        LoadedModel::Gemma4(w) => w,
        LoadedModel::Gemma4VLM(vlm) => &vlm.text_model,
        _ => panic!("requires a Gemma 4 family target"),
    };
    let block_size = pairing.block_size as usize;
    let sampling = SamplingConfig::greedy();
    let max_tokens = 24_usize;
    let prompts: Vec<Vec<i32>> = vec![
        vec![2, 105, 2364, 107, 9259, 108],
        vec![2, 105, 2364, 107, 1596, 108],
        vec![2, 105, 2364, 107, 6176, 108],
        vec![2, 105, 2364, 107, 3030, 108],
    ];
    let mut failures = Vec::new();
    for (row, prompt) in prompts.iter().enumerate() {
        let seq_id = mlxcel_core::cache::SequenceId::from_raw(3000 + row as u64);
        let adapter = Gemma4MtpTargetAdapter::new(wrapper, Some(seq_id));
        let (mut drafter, _) =
            load_drafter(&draft_path, Some(DrafterKind::Mtp)).expect("MTP drafter must load");
        drafter
            .bind(wrapper as &dyn mlxcel_core::generate::LanguageModel)
            .expect("drafter bind");
        let mut generator = MtpGenerator::new(adapter, drafter, block_size);
        let no_cancel = std::sync::atomic::AtomicBool::new(false);
        let logprobs_config = mlxcel_core::sampling::LogprobsConfig::default();
        let (reference, _, _) = generator.generate(
            prompt,
            max_tokens,
            &sampling,
            &[],
            &no_cancel,
            &logprobs_config,
        );

        let batch_adapter = Gemma4MtpBatchedTargetAdapter::new(wrapper, 1);
        let (mut batched_drafter, _) =
            load_drafter(&draft_path, Some(DrafterKind::Mtp)).expect("drafter must load");
        batched_drafter
            .bind(wrapper as &dyn mlxcel_core::generate::LanguageModel)
            .expect("batched drafter bind");
        let mut batched_generator =
            MtpBatchedGenerator::new(batch_adapter, batched_drafter, block_size);
        let run = batched_generator
            .run_batched(std::slice::from_ref(prompt), &sampling, max_tokens)
            .expect("B=1 batched MTP run must succeed");
        let got = &run.tokens[0];
        let first_mismatch = got
            .iter()
            .zip(reference.iter())
            .position(|(g, w)| g != w)
            .or_else(|| (got.len() != reference.len()).then(|| got.len().min(reference.len())));
        eprintln!(
            "[b1-batched probe] row {row}: accept_lens {:?}\n  b1-batched ({}): {:?}\n  b1 ref     ({}): {:?}\n  first_mismatch: {:?}",
            run.accept_lens[0],
            got.len(),
            got,
            reference.len(),
            reference,
            first_mismatch,
        );
        if let Some(i) = first_mismatch {
            failures.push(format!("row {row} mismatch at {i}"));
        }
    }
    assert!(
        failures.is_empty(),
        "B=1 batched baseline drift:\n{}",
        failures.join("\n")
    );
}

/// Issue #203 diagnostic v2: EQUAL-length forced-geometry probe. Row 0 runs
/// the same windows and forced accepts in three configurations:
///   (a) B=1 batched reference (uniform forever),
///   (b) B=2 with a lockstep filler row (control: no divergence),
///   (c) B=2 with a diverging filler row (row 0 carries a 3-slot hole).
/// Equal lengths mean no left-padding RoPE shift, so any deviation from (a)
/// is a real defect (or kernel batch instability for (b)). Reports per-round
/// argmax + hidden deviation; asserts the divergent case matches.
#[test]
#[ignore = "real-model heavy diagnostic (Gemma-4-31B target)"]
fn divergent_hidden_probe_31b() {
    use mlxcel::models::gemma4_mtp_target::Gemma4MtpBatchedTargetAdapter;
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::generate::SamplingConfig;
    use mlxcel_core::speculative::mtp::target::MtpTarget;

    let pairing = &REACHABLE_PAIRINGS[1];
    let (target_path, _draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!("Skipping divergent_hidden_probe_31b");
        return;
    }
    let _runtime = initialize_runtime();
    let (loaded_target, _tok) = load_model(&target_path).expect("target model must load");
    let wrapper: &mlxcel::models::Gemma4Wrapper = match &loaded_target {
        LoadedModel::Gemma4(w) => w,
        LoadedModel::Gemma4VLM(vlm) => &vlm.text_model,
        _ => panic!("requires a Gemma 4 family target"),
    };
    let sampling = SamplingConfig::greedy();

    // Equal-length prompts: probe row = the equal gate's failing row 3,
    // filler row = the equal gate's row 0.
    let probe_prompt: Vec<i32> = vec![2, 105, 2364, 107, 3030, 108];
    let filler_prompt: Vec<i32> = vec![2, 105, 2364, 107, 9259, 108];

    let to_f32 = |arr: &mlxcel_core::MlxArray| -> Vec<f32> {
        let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&arr);
        mlxcel_core::array_to_raw_bytes(&arr)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let row0_hidden = |captured: &mlxcel_core::speculative::mtp::target::VerifyCaptured| {
        let hidden = &captured.tensors[0];
        let shape = mlxcel_core::array_shape(hidden.as_ref().unwrap());
        let s = mlxcel_core::slice(
            hidden.as_ref().unwrap(),
            &[0, 0, 0],
            &[1, shape[1], shape[2]],
        );
        to_f32(s.as_ref().unwrap())
    };

    // ---- Discover the probe row's pure greedy chain on a throwaway adapter
    //      (width-1 walk: each round keeps exactly the bonus). ----
    let walk = Gemma4MtpBatchedTargetAdapter::new(wrapper, 1);
    let (b0, _) = walk
        .prefill_and_seed_batched(std::slice::from_ref(&probe_prompt), &sampling)
        .expect("walk prefill");
    let mut chain: Vec<i32> = vec![b0[0]];
    for _ in 0..12 {
        let f = walk
            .verify_forward_batched(&[vec![*chain.last().unwrap()]], &sampling)
            .expect("walk forward");
        let next = f.target_tokens_per_row[0][0];
        walk.verify_finalize_batched(&[0], 1, f.captured)
            .expect("walk finalize");
        chain.push(next);
    }
    eprintln!("[probe-v2] greedy chain: {chain:?}");
    // chain[0] = prefill bonus; chain[k] = k-th continuation token.
    // Round windows (width 4): r1 = [chain0..4) accepted 0; r2 = [chain1..5)
    // accepted 3; r3 = [chain5..9) forward-only comparison.
    let win_r1: Vec<i32> = chain[0..4].to_vec();
    let win_r2: Vec<i32> = chain[1..5].to_vec();
    let win_r3: Vec<i32> = chain[5..9].to_vec();

    // Filler-row windows (content arbitrary but FIXED across configs).
    let fill_r1 = vec![chain[0], 11, 12, 13];
    let fill_r2 = vec![14, 15, 16, 17];
    let fill_r3 = vec![18, 19, 20, 21];

    // One configuration runner: returns (argmax_r3, hidden_r1, hidden_r2, hidden_r3).
    let run_config =
        |filler: Option<(&[i32], [usize; 2])>| -> (Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let batch = if filler.is_some() { 2 } else { 1 };
            let adapter = Gemma4MtpBatchedTargetAdapter::new(wrapper, batch);
            let mut prompts = vec![probe_prompt.clone()];
            if let Some((fp, _)) = filler {
                prompts.push(fp.to_vec());
            }
            let (bons, _) = adapter
                .prefill_and_seed_batched(&prompts, &sampling)
                .expect("prefill");
            assert_eq!(
                bons[0], chain[0],
                "probe row prefill bonus must match the chain"
            );
            let mut w1 = vec![win_r1.clone()];
            let mut w2 = vec![win_r2.clone()];
            let mut w3 = vec![win_r3.clone()];
            let mut a1 = vec![0usize];
            let mut a2 = vec![3usize];
            if let Some((_, fa)) = filler {
                w1.push(fill_r1.clone());
                w2.push(fill_r2.clone());
                w3.push(fill_r3.clone());
                a1.push(fa[0]);
                a2.push(fa[1]);
            }
            let f1 = adapter.verify_forward_batched(&w1, &sampling).expect("r1");
            let h1 = row0_hidden(&f1.captured);
            adapter
                .verify_finalize_batched(&a1, 4, f1.captured)
                .expect("r1 fin");
            let f2 = adapter.verify_forward_batched(&w2, &sampling).expect("r2");
            let h2 = row0_hidden(&f2.captured);
            adapter
                .verify_finalize_batched(&a2, 4, f2.captured)
                .expect("r2 fin");
            let f3 = adapter.verify_forward_batched(&w3, &sampling).expect("r3");
            let h3 = row0_hidden(&f3.captured);
            (f3.target_tokens_per_row[0].clone(), h1, h2, h3)
        };

    let (am_ref, h1_ref, h2_ref, h3_ref) = run_config(None);
    let (am_lock, h1_lock, h2_lock, h3_lock) = run_config(Some((&filler_prompt, [0, 0])));
    let (am_div, h1_div, h2_div, h3_div) = run_config(Some((&filler_prompt, [3, 3])));

    let stats = |name: &str, a: &[f32], b: &[f32]| {
        let n_diff = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let max_abs = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[probe-v2] {name}: n_diff={n_diff}/{} max_abs={max_abs:e}",
            a.len()
        );
    };
    eprintln!("[probe-v2] r3 argmax: ref={am_ref:?} lockstep={am_lock:?} divergent={am_div:?}");
    stats("LOCKSTEP r1 hidden vs ref", &h1_lock, &h1_ref);
    stats("LOCKSTEP r2 hidden vs ref", &h2_lock, &h2_ref);
    stats("LOCKSTEP r3 hidden vs ref", &h3_lock, &h3_ref);
    stats("DIVERGENT r1 hidden vs ref", &h1_div, &h1_ref);
    stats("DIVERGENT r2 hidden vs ref", &h2_div, &h2_ref);
    stats("DIVERGENT r3 hidden vs ref", &h3_div, &h3_ref);
    assert_eq!(
        am_lock, am_ref,
        "lockstep control argmax must match the B=1 reference"
    );
    assert_eq!(
        am_div, am_ref,
        "divergent-geometry argmax must match the B=1 reference"
    );
}
