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
    for (row, (batched_row, reference_row)) in run.tokens.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            batched_row.len(),
            reference_row.len(),
            "row {row}: batched emitted {} tokens, B=1 reference emitted {}",
            batched_row.len(),
            reference_row.len(),
        );
        for (i, (got, want)) in batched_row.iter().zip(reference_row.iter()).enumerate() {
            assert_eq!(
                got, want,
                "row {row} token {i}: B=4 batched ({got}) != B=1 isolated ({want}) \
                 — greedy-parity violation in the batched MTP dispatch",
            );
        }
    }
    eprintln!(
        "[batched B=4] PASS: all {} rows byte-identical to B=1 isolated bursts",
        run.tokens.len()
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
