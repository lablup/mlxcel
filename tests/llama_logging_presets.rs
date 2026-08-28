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

//! End-to-end behavior of the b10621 logging, introspection, and preset
//! surface against the real server binaries (issue #1448, epic #1431).
//!
//! The unit tests beside each module cover the pure resolution rules. What
//! only a spawned process can answer lives here:
//!
//! - a log file written by a real run carries neither an API key nor a
//!   repository token, proved with canary values rather than by inspection;
//! - that log file is created `0600` rather than at whatever the umask allows;
//! - an unwritable or otherwise impossible `--log-file` fails at startup
//!   instead of falling back to the terminal;
//! - `--completion-bash` emits a script `bash -n` accepts, which offers the
//!   visible compatibility options and none of the hidden ones;
//! - `--cache-list` reports a store this test controls, in b10621's format;
//! - every preset and `--log-prompts-dir` are refused with a usable
//!   diagnostic, on both binaries, before any model is resolved.
//!
//! No model is needed: the refusals and the introspection actions all run
//! before model resolution, and the redaction case deliberately points `-m`
//! at a path that does not exist so the run installs logging, writes its
//! startup lines, and then fails.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

mod common;
use common::resolve_repo_binary;

/// Canary values that must never appear in any log sink.
const CANARY_API_KEY: &str = "canary-api-key-9f3c1d7e42b8";
const CANARY_HF_TOKEN: &str = "hf_canaryTOKEN5a1b2c3d4e5f6a7b8c";
const CANARY_FILE_KEY: &str = "canary-file-key-1122334455667788";

/// Every b10621 preset flag, from `common/arg.cpp` at the pinned commit.
const PRESET_FLAGS: [&str; 12] = [
    "--embd-gemma-default",
    "--fim-qwen-1.5b-default",
    "--fim-qwen-3b-default",
    "--fim-qwen-7b-default",
    "--fim-qwen-7b-spec",
    "--fim-qwen-14b-spec",
    "--fim-qwen-30b-default",
    "--gpt-oss-20b-default",
    "--gpt-oss-120b-default",
    "--vision-gemma-4b-default",
    "--vision-gemma-12b-default",
    "--spec-default",
];

/// The two server invocations, as (binary, leading arguments).
fn invocations() -> [(&'static str, Vec<&'static str>); 2] {
    [("mlxcel-server", Vec::new()), ("mlxcel", vec!["serve"])]
}

/// Run one server invocation with `extra` appended and a scrubbed
/// environment, returning its output.
///
/// `RUST_LOG` and every `LLAMA_ARG_*` variable this suite cares about are
/// cleared so an operator's shell cannot change what the test observes.
fn run(bin: &str, leading: &[&str], extra: &[&str], env: &[(&str, &str)]) -> Output {
    let (path, resolution) = resolve_repo_binary(bin);
    let mut command = Command::new(&path);
    command.args(leading).args(extra);
    for var in [
        "RUST_LOG",
        "LLAMA_ARG_LOG_FILE",
        "LLAMA_LOG_FILE",
        "LLAMA_ARG_LOG_COLORS",
        "LLAMA_ARG_LOG_PREFIX",
        "LLAMA_ARG_NO_LOG_PREFIX",
        "LLAMA_ARG_LOG_TIMESTAMPS",
        "LLAMA_ARG_NO_LOG_TIMESTAMPS",
        "LLAMA_ARG_LOG_VERBOSITY",
        "LLAMA_ARG_MODEL",
        "HF_TOKEN",
        "LLAMA_API_KEY",
        "MLXCEL_API_KEY",
    ] {
        command.env_remove(var);
    }
    for (key, value) in env {
        command.env(key, value);
    }
    command
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin} from {path:?}: {e}\n{resolution}"))
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

// ── secrets never reach a log file ──────────────────────────────────────

/// Run one binary with every credential this suite tracks set to a canary,
/// logging to `log_path`, against the model reference `model`.
fn run_with_canaries(
    bin: &str,
    leading: &[&str],
    model: &str,
    log_path: &Path,
    key_file: &Path,
) -> Output {
    run(
        bin,
        leading,
        &[
            "-m",
            model,
            "--log-file",
            &log_path.to_string_lossy(),
            "--api-key",
            CANARY_API_KEY,
            "--api-key-file",
            &key_file.to_string_lossy(),
            "--hf-token",
            CANARY_HF_TOKEN,
            "--verbosity",
            "5",
        ],
        &[("HF_TOKEN", CANARY_HF_TOKEN)],
    )
}

/// Assert no canary reached the log file, stdout, or stderr.
fn assert_no_canary(bin: &str, log_path: &Path, output: &Output) -> String {
    let logged = std::fs::read_to_string(log_path).unwrap_or_default();
    let combined = format!("{logged}{}{}", stdout_of(output), stderr_of(output));
    for canary in [CANARY_API_KEY, CANARY_HF_TOKEN, CANARY_FILE_KEY] {
        assert!(
            !combined.contains(canary),
            "{bin}: the canary {canary} reached a log sink\n--- log file ---\n{logged}\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            stdout_of(output),
            stderr_of(output)
        );
    }
    logged
}

#[test]
fn a_log_file_from_a_real_run_contains_no_api_key_or_repository_token() {
    let dir = tempfile::tempdir().expect("tempdir");
    let key_file = dir.path().join("keys.txt");
    std::fs::write(&key_file, format!("{CANARY_FILE_KEY}\n")).expect("write key file");

    // An existing but empty directory resolves as a local model path, so the
    // run reaches `start_server`, installs the subscriber, writes its startup
    // lines through the redacting writer, and only then fails on the load.
    // That is the window a configuration dump would leak in, and it is the
    // only arm where the log file has any content to inspect.
    let empty_model = dir.path().join("empty-model");
    std::fs::create_dir(&empty_model).expect("mkdir");

    for (bin, leading) in invocations() {
        let log_path = dir.path().join(format!("{bin}.log"));
        let output = run_with_canaries(
            bin,
            &leading,
            &empty_model.to_string_lossy(),
            &log_path,
            &key_file,
        );
        assert!(
            !output.status.success(),
            "{bin}: an empty model directory must fail the run"
        );
        let logged = assert_no_canary(bin, &log_path, &output);
        assert!(
            !logged.is_empty(),
            "{bin}: the log file is empty, so this test proved nothing about \
             redaction. The run must reach the subscriber before it fails.\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            stdout_of(&output),
            stderr_of(&output)
        );
    }
}

#[test]
fn no_canary_leaks_when_the_run_fails_before_the_subscriber_exists() {
    // The other half of the window: a model reference that cannot resolve at
    // all fails before any subscriber is installed, so the diagnostic goes
    // straight to the terminal. It must not echo the credentials either.
    let dir = tempfile::tempdir().expect("tempdir");
    let key_file = dir.path().join("keys.txt");
    std::fs::write(&key_file, format!("{CANARY_FILE_KEY}\n")).expect("write key file");

    for (bin, leading) in invocations() {
        let log_path = dir.path().join(format!("{bin}-early.log"));
        let output = run_with_canaries(
            bin,
            &leading,
            "/nonexistent/mlxcel/canary-model",
            &log_path,
            &key_file,
        );
        assert!(
            !output.status.success(),
            "{bin}: a nonexistent model must fail the run"
        );
        assert_no_canary(bin, &log_path, &output);
    }
}

#[cfg(unix)]
#[test]
fn a_log_file_from_a_real_run_is_created_owner_only() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().expect("tempdir");
    for (bin, leading) in invocations() {
        let log_path = dir.path().join(format!("{bin}-perms.log"));
        let _ = run(
            bin,
            &leading,
            &[
                "-m",
                "/nonexistent/mlxcel/canary-model",
                "--log-file",
                &log_path.to_string_lossy(),
            ],
            &[],
        );
        let metadata = std::fs::metadata(&log_path)
            .unwrap_or_else(|e| panic!("{bin}: the log file was never created: {e}"));
        let mode = metadata.permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "{bin}: log file mode is {mode:o}, expected 600"
        );
    }
}

// ── an impossible destination fails at startup ──────────────────────────

#[test]
fn an_unwritable_log_destination_fails_at_startup_rather_than_falling_back() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("no-such-dir").join("server.log");
    for (bin, leading) in invocations() {
        let output = run(
            bin,
            &leading,
            &[
                "-m",
                "/nonexistent/mlxcel/canary-model",
                "--log-file",
                &missing.to_string_lossy(),
            ],
            &[],
        );
        assert!(!output.status.success(), "{bin}: must not start");
        let text = format!("{}{}", stdout_of(&output), stderr_of(&output));
        assert!(
            text.contains("--log-file"),
            "{bin}: the diagnostic must name the option: {text}"
        );
        assert!(
            text.contains("no-such-dir"),
            "{bin}: the diagnostic must name the directory: {text}"
        );
    }
}

#[test]
fn a_directory_as_the_log_destination_fails_at_startup() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (bin, leading) in invocations() {
        let output = run(
            bin,
            &leading,
            &[
                "-m",
                "/nonexistent/mlxcel/canary-model",
                "--log-file",
                &dir.path().to_string_lossy(),
            ],
            &[],
        );
        assert!(!output.status.success(), "{bin}: must not start");
        let text = format!("{}{}", stdout_of(&output), stderr_of(&output));
        assert!(
            text.contains("is a directory"),
            "{bin}: unexpected diagnostic: {text}"
        );
    }
}

// ── --completion-bash ───────────────────────────────────────────────────

/// `bash -n` over `script`, or `None` when no bash is available.
fn bash_syntax_check(script: &str, dir: &Path) -> Option<Output> {
    let path = dir.join("completion.bash");
    std::fs::write(&path, script).expect("write script");
    Command::new("bash").arg("-n").arg(&path).output().ok()
}

#[test]
fn the_completion_script_is_syntactically_valid_bash() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (bin, leading) in invocations() {
        let output = run(bin, &leading, &["--completion-bash"], &[]);
        assert!(
            output.status.success(),
            "{bin} --completion-bash exited with {:?}: {}",
            output.status,
            stderr_of(&output)
        );
        let script = stdout_of(&output);
        let Some(checked) = bash_syntax_check(&script, dir.path()) else {
            eprintln!("skipping the bash -n check: no bash on PATH");
            continue;
        };
        assert!(
            checked.status.success(),
            "{bin}: bash -n rejected the completion script: {}\n{script}",
            String::from_utf8_lossy(&checked.stderr)
        );
    }
}

/// The spellings a completion script's `opts` list offers, as a set.
///
/// Read out of the script rather than asserted with `contains`, so `--model`
/// is not reported present merely because `--models-dir` is.
fn offered_spellings(script: &str) -> std::collections::BTreeSet<String> {
    let line = script
        .lines()
        .find(|line| line.trim_start().starts_with("opts=\""))
        .expect("the script must carry an opts list");
    let inner = line
        .trim()
        .trim_start_matches("opts=\"")
        .trim_end_matches('"');
    inner.split_whitespace().map(str::to_owned).collect()
}

#[test]
fn the_completion_script_offers_visible_compatibility_options() {
    for (bin, leading) in invocations() {
        let script = stdout_of(&run(bin, &leading, &["--completion-bash"], &[]));
        let offered = offered_spellings(&script);
        for spelling in [
            "--model",
            "--log-file",
            "--log-disable",
            "--log-colors",
            "--log-prefix",
            "--no-log-prefix",
            "--log-timestamps",
            "--verbose",
            "--log-verbose",
            "--verbosity",
            "--log-verbosity",
            "--cache-list",
            "--completion-bash",
            "--usage",
        ] {
            assert!(
                offered.contains(spelling),
                "{bin}: {spelling} is missing from the completion script's option \
                 list {offered:?}"
            );
        }
    }
}

#[test]
fn the_completion_script_hides_the_compatibility_surface_that_is_hidden() {
    for (bin, leading) in invocations() {
        let script = stdout_of(&run(bin, &leading, &["--completion-bash"], &[]));
        let mut forbidden = vec![
            "--dump-flag-surface",
            "--log-prompts-dir",
            "--n-gpu-layers",
            "--control-vector",
        ];
        forbidden.extend_from_slice(&PRESET_FLAGS);
        for spelling in forbidden {
            assert!(
                !script.contains(spelling),
                "{bin}: the hidden argument {spelling} leaked into the completion \
                 script:\n{script}"
            );
        }
    }
}

#[test]
fn the_completion_script_registers_against_the_invoked_command() {
    let server = stdout_of(&run("mlxcel-server", &[], &["--completion-bash"], &[]));
    assert!(
        server
            .trim_end()
            .ends_with("complete -F _mlxcel_server_completions mlxcel-server"),
        "{server}"
    );
    let serve = stdout_of(&run("mlxcel", &["serve"], &["--completion-bash"], &[]));
    assert!(
        serve
            .trim_end()
            .ends_with("complete -F _mlxcel_completions mlxcel"),
        "{serve}"
    );
}

// ── --cache-list ────────────────────────────────────────────────────────

/// Seed `<root>/<repo_id>/config.json`.
fn seed_checkpoint(root: &Path, repo_id: &str) {
    let dir = root.join(repo_id);
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("config.json"), "{}").expect("write");
}

#[test]
fn cache_list_reports_the_store_in_b10621s_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store: PathBuf = dir.path().join("models");
    seed_checkpoint(&store, "mlx-community/Qwen3-4B-4bit");
    seed_checkpoint(&store, "mlx-community/gemma-3-4b-it-4bit");

    for (bin, leading) in invocations() {
        // `--model-store-root` on the command line, not the environment: the
        // store `--cache-list` reports must be the one the same invocation
        // would load from. #1438 reserved `--models-dir` for b10621 router
        // discovery, so naming that spelling here would silently report the
        // developer's real cache instead of the seeded temporary one, and the
        // assertion below would depend on whatever happens to be cached.
        let output = run(
            bin,
            &leading,
            &[
                "--cache-list",
                "--model-store-root",
                &store.to_string_lossy(),
            ],
            &[],
        );
        assert!(
            output.status.success(),
            "{bin} --cache-list exited with {:?}: {}",
            output.status,
            stderr_of(&output)
        );
        let expected = concat!(
            "number of models in cache: 2\n",
            "   1. mlx-community/Qwen3-4B-4bit\n",
            "   2. mlx-community/gemma-3-4b-it-4bit\n",
        );
        assert_eq!(stdout_of(&output), expected, "{bin}");
    }
}

#[test]
fn cache_list_needs_no_model_argument() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (bin, leading) in invocations() {
        let output = run(
            bin,
            &leading,
            &["--cache-list"],
            &[("MLXCEL_MODELS_DIR", &dir.path().to_string_lossy())],
        );
        assert!(
            output.status.success(),
            "{bin}: --cache-list must not require -m: {}",
            stderr_of(&output)
        );
        assert_eq!(
            stdout_of(&output),
            "number of models in cache: 0\n",
            "{bin}"
        );
    }
}

#[test]
fn the_llama_server_short_spelling_of_cache_list_is_accepted() {
    // `-cl` is a llama.cpp multi-letter single-dash token; without the argv
    // pre-pass clap would read it as `-c -l` and start doing something else.
    let dir = tempfile::tempdir().expect("tempdir");
    for (bin, leading) in invocations() {
        let output = run(
            bin,
            &leading,
            &["-cl"],
            &[("MLXCEL_MODELS_DIR", &dir.path().to_string_lossy())],
        );
        assert!(
            output.status.success(),
            "{bin} -cl exited with {:?}: {}",
            output.status,
            stderr_of(&output)
        );
        assert_eq!(
            stdout_of(&output),
            "number of models in cache: 0\n",
            "{bin}"
        );
    }
}

// ── refusals ────────────────────────────────────────────────────────────

#[test]
fn every_preset_is_refused_on_both_binaries_with_a_download_example() {
    for (bin, leading) in invocations() {
        for flag in PRESET_FLAGS {
            let output = run(bin, &leading, &[flag], &[]);
            assert!(
                !output.status.success(),
                "{bin} {flag}: a GGUF preset must not start a server"
            );
            let text = format!("{}{}", stdout_of(&output), stderr_of(&output));
            assert!(
                text.contains(flag),
                "{bin} {flag}: the diagnostic must name the flag: {text}"
            );
            if flag == "--spec-default" {
                assert!(
                    text.contains("--draft-kind"),
                    "{bin} {flag}: must point at mlxcel's drafters: {text}"
                );
            } else {
                assert!(
                    text.contains("mlxcel download mlx-community/"),
                    "{bin} {flag}: must show the download command: {text}"
                );
            }
        }
    }
}

#[test]
fn log_prompts_dir_is_refused_and_never_creates_the_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let prompts = dir.path().join("prompts");
    for (bin, leading) in invocations() {
        let output = run(
            bin,
            &leading,
            &[
                "-m",
                "/nonexistent/mlxcel/canary-model",
                "--log-prompts-dir",
                &prompts.to_string_lossy(),
            ],
            &[],
        );
        assert!(!output.status.success(), "{bin}: must not start");
        let text = format!("{}{}", stdout_of(&output), stderr_of(&output));
        assert!(
            text.contains("--log-prompts-dir"),
            "{bin}: the diagnostic must name the option: {text}"
        );
        // b10621 creates the directory from inside its parser. mlxcel refuses
        // the option, so it must leave no trace on the filesystem either.
        assert!(
            !prompts.exists(),
            "{bin}: a refused --log-prompts-dir must not create {}",
            prompts.display()
        );
    }
}

#[test]
fn a_preset_is_refused_before_the_model_reference_is_resolved() {
    // The whole point of refusing at startup: a copied llama-server command
    // line fails immediately rather than after a multi-gigabyte download. The
    // model reference here is nonsense, and the preset diagnostic must still
    // be the one that comes back.
    for (bin, leading) in invocations() {
        let output = run(
            bin,
            &leading,
            &[
                "-m",
                "/nonexistent/mlxcel/canary-model",
                "--gpt-oss-20b-default",
            ],
            &[],
        );
        let text = format!("{}{}", stdout_of(&output), stderr_of(&output));
        assert!(
            text.contains("--gpt-oss-20b-default"),
            "{bin}: the preset refusal must win over the model error: {text}"
        );
    }
}

#[test]
fn an_unknown_log_colors_value_is_refused() {
    for (bin, leading) in invocations() {
        let output = run(
            bin,
            &leading,
            &[
                "-m",
                "/nonexistent/mlxcel/canary-model",
                "--log-colors",
                "sometimes",
            ],
            &[],
        );
        assert!(!output.status.success(), "{bin}: must not start");
        let text = format!("{}{}", stdout_of(&output), stderr_of(&output));
        assert!(
            text.contains("--log-colors") && text.contains("sometimes"),
            "{bin}: unexpected diagnostic: {text}"
        );
    }
}
