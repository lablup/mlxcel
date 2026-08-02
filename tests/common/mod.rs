use std::ffi::OsStr;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
pub fn repo_model_dir(name: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let primary = manifest_dir.join("models").join(name);
    if primary.exists() {
        return primary;
    }

    let shared_checkout = manifest_dir
        .parent()
        .map(|parent| parent.join("mlxcel-internal").join("models").join(name))
        .unwrap_or(primary.clone());
    if shared_checkout.exists() {
        return shared_checkout;
    }

    primary
}

/// Candidate locations for one of the crate's binaries, most authoritative
/// first, each labelled with where the path came from so a failure is
/// diagnosable from the panic message alone.
///
/// The list is deliberately short, and every entry is a signal cargo itself
/// produces. What used to be here, `<manifest>/target/<profile>/<name>`, is
/// gone: it is wrong whenever `CARGO_TARGET_DIR` is set, which is how every
/// test in `tests/cli_help_consistency.rs` came to fail on CI with `No such
/// file or directory` while passing locally (issue #962). `release.yml` and
/// `nightly-verify.yml` both set it to `$HOME/.cargo-target/mlxcel` so the
/// self-hosted runner keeps a warm build cache outside the checkout. It is
/// also silently wrong under `--target`, where the binaries live under an
/// extra triple segment it cannot know, and its `cfg!(debug_assertions)`
/// profile guess is only a proxy for the real profile name. A candidate that
/// can hand back a stale binary from an unrelated build is worse than no
/// candidate at all.
///
/// The runtime environment is deliberately not consulted either. Cargo never
/// sets `CARGO_BIN_EXE_*` at test runtime, so the previous `var_os` lookup was
/// dead code, and honouring it would let a stale exported value win over the
/// binary cargo just built, which is the same shape as the bug this ordering
/// exists to fix.
fn binary_path_candidates(name: &str) -> Vec<(&'static str, PathBuf)> {
    let mut candidates: Vec<(&'static str, PathBuf)> = Vec::new();

    // 1. The compile-time path cargo hands every integration test target. It
    //    already accounts for the profile, the target triple, and
    //    `CARGO_TARGET_DIR`, and cargo builds the binary before running the
    //    test, so for the binaries named here it is authoritative and needs no
    //    reconstruction. This is the same mechanism `tests/surgery_cli.rs` and
    //    `tests/lang_bias.rs` already use directly.
    let compile_time: Option<&'static str> = match name {
        "mlxcel" => Some(env!("CARGO_BIN_EXE_mlxcel")),
        "mlxcel-server" => Some(env!("CARGO_BIN_EXE_mlxcel-server")),
        "mlxcel-bench-decode" => Some(env!("CARGO_BIN_EXE_mlxcel-bench-decode")),
        _ => None,
    };
    if let Some(path) = compile_time {
        candidates.push(("CARGO_BIN_EXE_* (compile time)", PathBuf::from(path)));
    }

    // 2. Derived from this test binary's own location, covering any binary not
    //    named above. Integration tests link into
    //    `<target-dir>/[<triple>/]<profile>/deps/`, so the directory holding
    //    the package binaries is two levels up. The `deps` check enforces that
    //    layout rather than assuming it: a test binary running from a copied
    //    location (a nextest archive, an extracted CI artifact) must not spawn
    //    whatever sibling file happens to share the binary's name.
    let derived_profile_dir = std::env::current_exe().ok().and_then(|test_exe| {
        test_exe
            .parent()
            .filter(|deps| deps.file_name() == Some(OsStr::new("deps")))
            .and_then(Path::parent)
            .map(Path::to_path_buf)
    });
    if let Some(profile_dir) = derived_profile_dir {
        candidates.push(("derived from current_exe()", profile_dir.join(name)));
    }

    candidates
}

/// Resolve one of the crate's binaries, returning both the path and a report
/// of how it was chosen.
///
/// Both halves come from a single pass over the candidates, so the report can
/// never contradict the choice it is explaining, and the caller stats the
/// filesystem once rather than twice.
#[allow(dead_code)]
pub fn resolve_repo_binary(name: &str) -> (PathBuf, String) {
    let candidates = binary_path_candidates(name);
    let selected = candidates
        .iter()
        .position(|(_, path)| path.exists())
        .or(if candidates.is_empty() { None } else { Some(0) });

    let mut report = format!("binary resolution for {name:?}:\n");
    if candidates.is_empty() {
        report.push_str(
            "  (no candidate: this binary is not one the helper names, and the \
             running test binary is not inside a cargo `deps` directory)\n",
        );
    }
    for (index, (source, path)) in candidates.iter().enumerate() {
        let state = if path.exists() { "exists" } else { "missing" };
        let marker = if Some(index) == selected {
            " <- selected"
        } else {
            ""
        };
        report.push_str(&format!(
            "  [{state}] {source}: {}{marker}\n",
            path.display()
        ));
    }
    report.push_str(&match std::env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => format!(
            "  CARGO_TARGET_DIR is set to {}",
            PathBuf::from(dir).display()
        ),
        None => "  CARGO_TARGET_DIR is unset".to_string(),
    });

    let path = selected
        .map(|index| candidates[index].1.clone())
        // Nothing to go on. Hand back a path that deliberately does not exist,
        // so the caller's spawn fails with this report attached, rather than a
        // bare name, which would be resolved through `PATH`.
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join(name)
        });
    (path, report)
}

/// Path to one of the crate's binaries.
///
/// Returns the first candidate that exists, and otherwise the most
/// authoritative one. It still returns a path unconditionally, so the
/// `if !binary.exists() { skip }` guards in the real-model integration tests
/// keep compiling and behaving the same way. Callers that spawn without such a
/// guard should use [`resolve_repo_binary`] and put the report in their panic.
#[allow(dead_code)]
pub fn repo_binary_path(name: &str) -> PathBuf {
    resolve_repo_binary(name).0
}

#[allow(dead_code)]
pub fn extract_generated_body(stdout: &str) -> Option<&str> {
    let start = stdout.rfind("Generating...\n")?;
    let start = start + "Generating...\n".len();
    let rest = &stdout[start..];
    let end = rest.find("\n\n[")?;
    Some(&rest[..end])
}
