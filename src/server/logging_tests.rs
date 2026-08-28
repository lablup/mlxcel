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

//! Unit tests for the server logging policy (issue #1448): verbosity
//! precedence, log-file permissions and refusals, and secret redaction.

use super::*;

// ── verbosity mapping ───────────────────────────────────────────────────

#[test]
fn the_verbosity_scale_is_monotone_in_b10621s_direction() {
    // b10621 drops messages ABOVE the threshold, so a larger number is always
    // at least as verbose. Encoding that as an ordering check rather than a
    // literal table keeps the property, not the spelling, under test.
    let rank = |directive: &str| match directive {
        "error" => 0,
        "warn" => 1,
        "info" => 2,
        "debug" => 3,
        _ => 4,
    };
    let mut previous = rank(filter_directive_for_verbosity(i32::MIN));
    for threshold in -2..=7 {
        let current = rank(filter_directive_for_verbosity(threshold));
        assert!(
            current >= previous,
            "verbosity {threshold} is less verbose than {}",
            threshold - 1
        );
        previous = current;
    }
}

#[test]
fn the_default_threshold_is_info() {
    assert_eq!(DEFAULT_VERBOSITY, 3);
    assert_eq!(filter_directive_for_verbosity(DEFAULT_VERBOSITY), "info");
}

#[test]
fn the_top_tier_raises_only_mlxcels_own_targets_to_trace() {
    // A bare `trace` directive turns on hyper and tokio internals and buries
    // the mlxcel messages the operator asked for.
    let top = filter_directive_for_verbosity(5);
    assert!(top.contains("mlxcel=trace"), "{top}");
    assert!(top.contains("mlxcel_core=trace"), "{top}");
    assert!(top.starts_with("debug,"), "{top}");
}

#[test]
fn a_negative_or_zero_threshold_still_reports_errors() {
    for threshold in [i32::MIN, -1, 0, 1] {
        assert_eq!(filter_directive_for_verbosity(threshold), "error");
    }
}

// ── verbosity precedence ────────────────────────────────────────────────

#[test]
fn the_verbose_flag_beats_rust_log() {
    // The divergence #1448 was filed to close: before it,
    // `EnvFilter::try_from_default_env` ran first, so `RUST_LOG=warn -v`
    // silently ignored `-v` while b10621's `-v` is unconditional.
    let (directive, source) = resolve_log_filter(true, None, Some("warn"), None);
    assert_eq!(source, VerbositySource::VerboseFlag);
    assert_eq!(directive, filter_directive_for_verbosity(i32::MAX));
}

#[test]
fn a_command_line_verbosity_beats_rust_log() {
    let (directive, source) = resolve_log_filter(false, Some(2), Some("trace"), None);
    assert_eq!(source, VerbositySource::VerbosityFlag);
    assert_eq!(directive, "warn");
}

#[test]
fn rust_log_beats_the_llama_environment_binding() {
    let (directive, source) = resolve_log_filter(false, None, Some("mlxcel=debug"), Some(1));
    assert_eq!(source, VerbositySource::RustLog);
    assert_eq!(directive, "mlxcel=debug");
}

#[test]
fn the_llama_environment_binding_beats_the_compiled_default() {
    let (directive, source) = resolve_log_filter(false, None, None, Some(4));
    assert_eq!(source, VerbositySource::LlamaEnv);
    assert_eq!(directive, "debug");
}

#[test]
fn nothing_set_resolves_to_the_compiled_default() {
    let (directive, source) = resolve_log_filter(false, None, None, None);
    assert_eq!(source, VerbositySource::Default);
    assert_eq!(directive, "info");
}

#[test]
fn an_empty_rust_log_is_not_a_directive() {
    // `RUST_LOG=` (or all whitespace) is how a shell unsets a variable it
    // cannot remove; treating it as a filter would silence the server.
    for value in ["", "   "] {
        let (directive, source) = resolve_log_filter(false, None, Some(value), None);
        assert_eq!(source, VerbositySource::Default, "RUST_LOG={value:?}");
        assert_eq!(directive, "info");
    }
}

#[test]
fn the_verbose_flag_and_the_top_verbosity_tier_agree() {
    // `-v` must never be less verbose than `-lv 5`, which is what b10621's
    // `verbosity = INT_MAX` guarantees.
    let (verbose, _) = resolve_log_filter(true, None, None, None);
    let (top, _) = resolve_log_filter(false, Some(5), None, None);
    assert_eq!(verbose, top);
}

// ── colors ──────────────────────────────────────────────────────────────

#[test]
fn auto_colors_never_write_escapes_into_a_log_file() {
    assert!(!LogColors::Auto.resolve(true, true));
    assert!(!LogColors::Auto.resolve(true, false));
    assert!(LogColors::Auto.resolve(false, true));
    assert!(!LogColors::Auto.resolve(false, false));
}

#[test]
fn explicit_colors_ignore_the_sink() {
    assert!(LogColors::On.resolve(true, false));
    assert!(!LogColors::Off.resolve(false, true));
}

// ── log file ────────────────────────────────────────────────────────────

#[test]
fn a_new_log_file_is_created_with_owner_only_permissions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.log");
    let mut file = open_log_file(&path).expect("creates the log file");
    file.write_all(b"hello\n").expect("write");
    drop(file);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "log file mode is {mode:o}, expected 600");
    }
}

#[test]
fn an_existing_world_readable_log_file_is_tightened() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.log");
    std::fs::write(&path, "old\n").expect("seed");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");
    }

    let _file = open_log_file(&path).expect("reopens the log file");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "existing log file left at {mode:o}");
    }
}

#[test]
fn appending_never_truncates_an_existing_log() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.log");
    std::fs::write(&path, "first\n").expect("seed");
    let mut file = open_log_file(&path).expect("open");
    file.write_all(b"second\n").expect("write");
    drop(file);
    let contents = std::fs::read_to_string(&path).expect("read");
    assert_eq!(contents, "first\nsecond\n");
}

#[test]
fn a_directory_destination_fails_instead_of_falling_back() {
    let dir = tempfile::tempdir().expect("tempdir");
    let error = open_log_file(dir.path()).expect_err("a directory is not a log file");
    assert!(
        error.to_string().contains("is a directory"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn a_missing_parent_directory_fails_with_the_directory_named() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("no-such-dir").join("server.log");
    let error = open_log_file(&path).expect_err("a missing parent must fail");
    let text = error.to_string();
    assert!(text.contains("does not exist"), "unexpected: {text}");
    assert!(text.contains("no-such-dir"), "unexpected: {text}");
}

#[cfg(unix)]
#[test]
fn a_symlink_destination_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("target.log");
    std::fs::write(&target, "").expect("seed");
    let link = dir.path().join("server.log");
    std::os::unix::fs::symlink(&target, &link).expect("symlink");

    let error = open_log_file(&link).expect_err("a symlinked log destination must be refused");
    assert!(
        error.to_string().contains("symbolic link"),
        "unexpected diagnostic: {error}"
    );
}

#[cfg(unix)]
#[test]
fn an_unwritable_directory_fails_at_open_time() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let locked = dir.path().join("locked");
    std::fs::create_dir(&locked).expect("mkdir");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o500)).expect("chmod");

    let result = open_log_file(&locked.join("server.log"));

    // Restore before asserting so the temp dir can be cleaned up either way.
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o700)).expect("chmod");
    let error = result.expect_err("an unwritable directory must fail at startup");
    assert!(
        error.to_string().contains("cannot open"),
        "unexpected diagnostic: {error}"
    );
}

// ── redaction ───────────────────────────────────────────────────────────

#[test]
fn a_registered_secret_never_survives_a_line() {
    let secret = "logging-tests-secret-abcdefgh";
    register_log_secret(secret);
    let line = format!("api key {secret} accepted");
    let redacted = redact(&line);
    assert!(!redacted.contains(secret), "{redacted}");
    assert!(redacted.contains(REDACTED), "{redacted}");
}

#[test]
fn a_line_with_no_secret_is_returned_borrowed() {
    let line = "nothing sensitive here";
    assert!(matches!(redact(line), std::borrow::Cow::Borrowed(_)));
}

#[test]
fn short_values_are_never_registered_as_secrets() {
    let before = registered_secret_count();
    for value in ["", "   ", "a", "abc", "1234567"] {
        register_log_secret(value);
    }
    assert_eq!(
        registered_secret_count(),
        before,
        "a value shorter than the minimum would redact ordinary prose"
    );
}

#[test]
fn registering_the_same_secret_twice_is_idempotent() {
    let secret = "logging-tests-idempotent-secret";
    register_log_secret(secret);
    let after_first = registered_secret_count();
    register_log_secret(secret);
    assert_eq!(registered_secret_count(), after_first);
}

// ── line-buffered writer ────────────────────────────────────────────────

#[test]
fn a_secret_split_across_two_writes_is_still_redacted() {
    // The reason redaction is line-buffered: `tracing_subscriber`'s formatter
    // emits an event in several `write` calls, so a per-call scan would let a
    // secret straddling two of them through.
    let secret = "logging-tests-split-secret-xyz";
    register_log_secret(secret);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("split.log");
    let file = open_log_file(&path).expect("open");
    let writer = RedactingWriter {
        shared: Arc::new(Mutex::new(LineRedactor {
            sink: LogSink::File(file),
            pending: Vec::new(),
        })),
    };

    let (head, tail) = secret.split_at(7);
    let mut sink = writer.make_writer();
    sink.write_all(b"prefix ").expect("write");
    sink.write_all(head.as_bytes()).expect("write");
    sink.write_all(tail.as_bytes()).expect("write");
    sink.write_all(b" suffix\n").expect("write");
    sink.flush().expect("flush");

    let contents = std::fs::read_to_string(&path).expect("read");
    assert!(!contents.contains(secret), "{contents}");
    assert_eq!(contents, format!("prefix {REDACTED} suffix\n"));
}

#[test]
fn a_trailing_partial_line_is_flushed_rather_than_dropped() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("partial.log");
    let file = open_log_file(&path).expect("open");
    let writer = RedactingWriter {
        shared: Arc::new(Mutex::new(LineRedactor {
            sink: LogSink::File(file),
            pending: Vec::new(),
        })),
    };

    let mut sink = writer.make_writer();
    sink.write_all(b"no trailing newline").expect("write");
    sink.flush().expect("flush");

    let contents = std::fs::read_to_string(&path).expect("read");
    assert_eq!(contents, "no trailing newline\n");
}

#[test]
fn several_lines_in_one_write_are_each_emitted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("multi.log");
    let file = open_log_file(&path).expect("open");
    let writer = RedactingWriter {
        shared: Arc::new(Mutex::new(LineRedactor {
            sink: LogSink::File(file),
            pending: Vec::new(),
        })),
    };

    let mut sink = writer.make_writer();
    sink.write_all(b"one\ntwo\nthree\n").expect("write");
    sink.flush().expect("flush");

    let contents = std::fs::read_to_string(&path).expect("read");
    assert_eq!(contents, "one\ntwo\nthree\n");
}
