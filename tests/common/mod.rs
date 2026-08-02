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
/// first. Each entry carries a short label naming where the path came from so
/// a failure can be diagnosed from the panic message alone.
///
/// The list exists because `<manifest>/target/<profile>/<name>` is not where
/// the binaries live whenever `CARGO_TARGET_DIR` is set. Both `release.yml`
/// and `nightly-verify.yml` set it to `$HOME/.cargo-target/mlxcel` so the
/// self-hosted runner keeps a warm build cache outside the checkout, which is
/// how every test in `tests/cli_help_consistency.rs` came to fail on CI with
/// `No such file or directory` while passing locally (see issue #962).
fn binary_path_candidates(name: &str) -> Vec<(&'static str, PathBuf)> {
    let mut candidates: Vec<(&'static str, PathBuf)> = Vec::new();

    // 1. Runtime override. Cargo never sets this at test runtime, so it only
    //    fires when a harness or an operator points the tests at a
    //    pre-built binary on purpose.
    if let Some(path) = std::env::var_os(format!("CARGO_BIN_EXE_{name}")) {
        candidates.push(("CARGO_BIN_EXE_* (runtime env)", PathBuf::from(path)));
    }

    // 2. The compile-time path cargo hands every integration test target. It
    //    already accounts for the profile, the target triple, and
    //    `CARGO_TARGET_DIR`, so it needs no reconstruction and cannot drift.
    let compile_time: Option<&'static str> = match name {
        "mlxcel" => Some(env!("CARGO_BIN_EXE_mlxcel")),
        "mlxcel-server" => Some(env!("CARGO_BIN_EXE_mlxcel-server")),
        "mlxcel-bench-decode" => Some(env!("CARGO_BIN_EXE_mlxcel-bench-decode")),
        _ => None,
    };
    if let Some(path) = compile_time {
        candidates.push(("CARGO_BIN_EXE_* (compile time)", PathBuf::from(path)));
    }

    // 3. Derived from this test binary's own location. Integration tests are
    //    linked into `<target-dir>/[<triple>/]<profile>/deps/`, so the profile
    //    directory holding the package binaries is two levels up. This covers
    //    binaries not named above without hardcoding a target directory.
    let derived_profile_dir = std::env::current_exe().ok().and_then(|test_exe| {
        test_exe
            .parent()
            .and_then(|deps| deps.parent())
            .map(Path::to_path_buf)
    });
    if let Some(profile_dir) = derived_profile_dir {
        candidates.push(("derived from current_exe()", profile_dir.join(name)));
    }

    // 4. Last resort: reconstruct from `CARGO_TARGET_DIR` when set, else from
    //    the manifest directory. Kept so a stripped-down invocation still has
    //    something to try, not relied on by the paths above.
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    candidates.push((
        "reconstructed from CARGO_TARGET_DIR / CARGO_MANIFEST_DIR",
        target_dir.join(profile).join(name),
    ));

    candidates
}

/// Resolve the path to one of the crate's binaries for an integration test.
///
/// Returns the first candidate from [`binary_path_candidates`] that exists on
/// disk, falling back to the most authoritative candidate when none do. The
/// caller-visible behaviour of returning a path unconditionally is preserved
/// so the existing `if !binary.exists() { skip }` guards in the real-model
/// integration tests keep working; [`binary_resolution_report`] supplies the
/// diagnostics for the paths that spawn unconditionally.
#[allow(dead_code)]
pub fn repo_binary_path(name: &str) -> PathBuf {
    let candidates = binary_path_candidates(name);
    for (_, path) in &candidates {
        if path.exists() {
            return path.clone();
        }
    }
    candidates
        .into_iter()
        .next()
        .map(|(_, path)| path)
        .unwrap_or_else(|| PathBuf::from(name))
}

/// Human-readable account of how [`repo_binary_path`] resolved `name`, for
/// use in panic messages when spawning the binary fails. Names every
/// candidate, whether it exists, and the target directory the reconstruction
/// fallback derived, so a CI failure is diagnosable without reproducing the
/// environment.
#[allow(dead_code)]
pub fn binary_resolution_report(name: &str) -> String {
    let mut report = format!("binary resolution for {name:?}:\n");
    for (source, path) in binary_path_candidates(name) {
        let state = if path.exists() { "exists" } else { "missing" };
        report.push_str(&format!("  [{state}] {source}: {}\n", path.display()));
    }
    let configured_target_dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from);
    report.push_str(&match configured_target_dir {
        Some(dir) => format!("  CARGO_TARGET_DIR is set to {}", dir.display()),
        None => format!(
            "  CARGO_TARGET_DIR is unset; reconstruction assumed {}",
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .display()
        ),
    });
    report
}

#[allow(dead_code)]
pub fn extract_generated_body(stdout: &str) -> Option<&str> {
    let start = stdout.rfind("Generating...\n")?;
    let start = start + "Generating...\n".len();
    let rest = &stdout[start..];
    let end = rest.find("\n\n[")?;
    Some(&rest[..end])
}
