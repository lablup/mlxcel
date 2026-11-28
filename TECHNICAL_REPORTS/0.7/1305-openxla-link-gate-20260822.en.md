# Technical Report: PR #1305 - link an OpenXLA binary in CI so link-only regressions cannot reach main

**Date**: 2026-08-22
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: YAML (GitHub Actions)
**Risk Level**: Low (CI configuration only; no runtime code changed)

---

## Executive Summary

CI compiled the OpenXLA feature combinations but never linked them. `cargo check` does not invoke the linker, so a regression in the IREE link recipe in `build.rs` could reach `main` with every check green. That is not a hypothetical: issue #1274 was exactly such a failure, and `cargo check --features cuda,xla-iree --all-targets` passed on the same tree that could not link a single integration test.

This PR adds an `xla-link` job to `.github/workflows/ci.yml` that links a real target under `--features cuda,xla-iree` on the self-hosted GB10 runner, triggered by a narrow path filter over the paths that can actually move the link line. The gate was validated by breaking the recipe and confirming the split: `cargo check` still passed in 59 seconds while the link failed.

The review cycle corrected three factual errors in the change's own comments, including one this report treats as its main learning point: a claim about the IREE archive that came from running `nm` in a shell where `nm` is an alias for an unrelated command.

## 1. Problem Statement

### 1.1 Background

PR #1282 added the `xla-compile` job, which runs two commands on the GB10 runner:

```
cargo check --features cuda,xla-iree --all-targets
cargo check --no-default-features --features xla-diagnostics --all-targets
```

Nothing else in CI compiles any XLA feature: `pipeline-parallel-ci.yml` runs clippy on default features, `nightly-verify.yml` runs `make verify` with `metal,accelerate`, and `release.yml` does not build `xla-iree`. That job's own comment recorded the linking gap as deliberately open, pending this work.

### 1.2 Existing Issues

`cargo check` stops after type checking and never runs the linker, so the entire `cargo:rustc-link-arg` recipe that `build.rs` emits for the IREE runtime is untested by CI. The #1274 failure is the canonical example:

```
/usr/bin/ld: libiree_runtime_unified.a(call.c.o): undefined reference to symbol '__stack_chk_guard@@GLIBC_2.17'
/usr/bin/ld: /lib/ld-linux-aarch64.so.1: error adding symbols: DSO missing from command line
```

It reached `main` and was found by hand while validating an unrelated PR. The fix, PR #1275, appends a second `-lc` after the IREE archives, because rustc emits its own `-lc` before anything `cargo:rustc-link-arg` can append.

### 1.3 Risk Assessment

The exposure is bounded but real: the OpenXLA path is not on the default feature set, so a broken link does not break the shipped default build. It does break every developer working on that path, and it is discovered at the worst time, by hand, after the fact.

## 2. Change Summary

One file, `.github/workflows/ci.yml`.

| Change | Detail |
|---|---|
| New `changes` output | `xla_link`, from a sibling `dorny/paths-filter` filter |
| Filter paths | `build.rs`, `src/lib/mlxcel-xla/build.rs`, `src/lib/mlxcel-xla/csrc/**`, `scripts/iree/**`, `rust-toolchain.toml`, `.github/workflows/ci.yml` |
| New job | `xla-link`, after `xla-compile`, `runs-on: GB10`, `timeout-minutes: 120`, `permissions: contents: read` |
| Link command | `cargo test --release --features cuda,xla-iree --test xla_prepared_prefill --no-run` |
| Target dir | `$HOME/.cargo-target/mlxcel-xla-link-ci`, separate from `xla-compile`'s and the release job's |
| Concurrency | Job-scoped group keyed per PR, `cancel-in-progress: true` |
| Comment corrections | `clippy`'s fork-guard note; `xla-compile`'s "not covered" list now points at the new job |

## 3. Technical Decisions

### 3.1 The release profile is forced, not chosen

The debug profile cannot link these targets on this host at all. It fails with hundreds of `relocation truncated to fit: R_AARCH64_CALL26` errors against ordinary `libstd` and `compiler_builtins` symbols, because the unoptimized binary exceeds the AArch64 direct-branch range. The cheaper debug link was never an option, which is why the job costs minutes rather than seconds and why it is a separate job rather than a step appended to `xla-compile`.

### 3.2 Link the smallest target, not the shipped binary

`build.rs` emits the recipe through `cargo:rustc-link-arg`, which applies to every linked artifact of the crate. The recipe under test is therefore identical no matter which target is linked, and the integration test is the direct reproducer because #1274 was an integration-test link failure.

The review corrected the rationale here. Cargo builds the package's `[[bin]]` targets whenever an integration test is selected, so this command links `mlxcel`, `mlxcel-server`, `speculative_bench`, and `mlxcel-bench-decode` in addition to the test binary. The rejected `cargo build --release --bin mlxcel-server` alternative is a strict subset of the chosen command, not a costlier one, and the measured figures below already include the binaries. The target choice was right; the reason first written down for it was not.

### 3.3 Path-filtered trigger, not every Rust PR and not a schedule

A schedule decouples the failure from the PR that caused it, which is precisely how #1274 escaped. Running on every Rust PR spends minutes of a shared runner on the large majority of PRs that cannot affect the link line.

The filter covers the causal surface: the root `build.rs` holds the recipe, `scripts/iree/**` pins the distribution whose archive set the recipe names, `mlxcel-xla`'s build script and its `csrc/**` sources produce the shim object whose undefined symbols those archives resolve, and `rust-toolchain.toml` is on the path because #1274 was entirely about where rustc places its own `-lc` relative to appended arguments, which a toolchain bump can move.

The last two `mlxcel-xla` paths were added during review. The original filter named only `build.rs` and `scripts/iree/**`; as a picomatch pattern `build.rs` matches only the root build script, so a change to the C shim would have passed `cargo check` and not started this job either.

### 3.4 `RUSTFLAGS` is deliberately unset

`xla-compile` sets `RUSTFLAGS: "-D unused_imports"` and owns the lint policy. Leaving it unset here means a red run is unambiguously a link failure rather than a lint failure wearing a link job's name. Issue #1304 separately clears the dead-code backlog and widens `xla-compile` to `-D warnings`.

### 3.5 A concurrency group, because this is the first slow job on GB10

Every other GB10 job is seconds warm. This one is minutes, with `timeout-minutes: 120`. `ci.yml` carries no concurrency control at all, so three pushes to a `build.rs` PR would queue three link jobs on the single runner that also serves clippy for every Rust PR and the release build, with superseded runs occupying it to completion. The group is keyed per PR so PRs never cancel each other, matching the pattern `pipeline-parallel-ci.yml` already uses.

## 4. Review Findings

### 4.1 HIGH: a comment recorded a fact that a shell alias had invented

An earlier revision explained the failed `-lc` control by asserting that the pinned IREE runtime was built without the stack protector and that `libiree_runtime_unified.a` held no `__stack_chk_guard` reference, "`nm` reports zero".

It holds 176 such references, including in the `call.c.o` named in #1274. The zero came from this shell:

```
nm is an alias for mosh --ssh="ssh -i ~/.ssh/nubimaru.pem" ubuntu@<host>
```

`nm <archive> | grep -c __stack_chk_guard` therefore counted the output of a program that never examined the archive, and `2>/dev/null` hid the `command not found` that would have exposed it. With `/usr/bin/nm`, every precondition the `build.rs` comment records still holds: the symbol is undefined in 176 objects of the archive, undefined in `libc.so.6`, and defined only in `ld-linux-aarch64.so.1`.

### 4.2 HIGH: the path filter missed part of the causal surface

See 3.3. Fixed by adding `src/lib/mlxcel-xla/build.rs` and `src/lib/mlxcel-xla/csrc/**`.

### 4.3 MEDIUM: the "what it links" rationale was wrong about the binaries

See 3.2. The comment now records that the binaries are linked here too.

### 4.4 Security: the fork-guard comment was false and propagating

The note above the `clippy` job asserted that `if: github.repository == 'lablup/mlxcel'` means fork PRs never queue on the self-hosted runner. On a `pull_request` event opened from a fork against this repository, `github.repository` is the base repository, so the guard is true and the job runs, executing the PR's `build.rs` and `scripts/iree/**` as the runner's own user. The guard's real effect is to stop the job queueing forever in a *fork of* the repository, which has no GB10 runner. What gates the fork case is the repository's Actions fork-PR approval policy, currently `first_time_contributors`.

Issue #1303 repeated the incorrect reading when specifying this job, so the misconception was spreading. The comment now records what the guard actually does. This PR adds no new exposure: `clippy` and `xla-compile` gate on a strictly wider filter and already run fork PR code on that runner.

## 5. Validation

All runs on the GB10 host with the IREE runtime provisioned.

| Run | Result |
|---|---|
| Cold link, purged `CARGO_TARGET_DIR` | exit 0, 15m40s, ends at `Executable tests/xla_prepared_prefill.rs` |
| Warm link (two runs) | exit 0, 6m01s and 7m24s |
| `cargo check` on a link-broken tree | **exit 0, 59s** |
| Link on the same broken tree | **exit 101**, `error: linking with cc failed`, undefined references to `flatcc_verify_*` |
| Link after restoring the archive | exit 0 |
| The job on this PR itself | success, all four commits |

The break was dropping `-l:libflatcc_parsing.a` from the `IREE_CUDA_HOME` branch of `build.rs`. The middle two rows are the whole point of the change: the same tree, one command green and the other red, which is the shape of #1274. `build.rs` is unchanged by this PR; every control was run outside the branch and reverted.

Most of the cold figure is MLX's CUDA sources compiling from scratch rather than the link. Warm is the figure that matters, because the trigger fires on `build.rs`, which invalidates the `mlxcel` crate but not `mlxcel-core`'s MLX build.

## 6. What Remains Unverified

- **Why the `-lc` control does not reproduce.** It was tried first, per the issue's acceptance criteria, and the link succeeded with `rustc-link-arg=-lc` confirmed absent from the emitted build-script output and the IREE archives confirmed present on the link line. The stack-protector explanation was wrong (4.1). Two further hypotheses were tested and ruled out: `-lpthread` and `-ldl` resolve to stub archives on this glibc rather than to linker scripts that would pull libc in late, and `libm.so` groups only `libm.so.6`. The entry stays, and the comment says not to remove it on the strength of one control that failed to reproduce.
- **The `IREE_DIST` and macOS `IREE_MACOS_HOME` recipes** in `build.rs` remain unverified by any link, on any machine, because no runner holds either distribution.
- **A green run does not prove a link happened.** The target directory persists across PRs, so cargo can find nothing to redo; this job's first run finished in 12 seconds with `Finished release profile in 0.14s`. That is correct behavior, since cargo relinks exactly when the fingerprint moves and `build.rs` declares `rerun-if-env-changed` for all three IREE variables. The narrow hole is a `scripts/iree/**` edit that changes how the runtime is built without moving `IREE_VERSION`: the script is idempotent, so nothing rebuilds and nothing relinks.
- **`Cargo.toml` and `Cargo.lock` are outside the filter**, so a dependency-driven link change does not trigger the job. Widening would fire on every dependency bump.

## 7. Learning Points

1. **Verify that a diagnostic tool is the tool you think it is.** A shell alias turned `nm` into an SSH client, and suppressing stderr turned the resulting `command not found` into a confident empty result. A search that returns zero matches deserves one confirming positive control before a conclusion is built on it.
2. **`cargo check` is not a link gate, and `--all-targets` does not change that.** The flag widens what is type checked, not what is linked.
3. **`cargo:rustc-link-arg` applies to every linked artifact**, so any linked target exercises the same recipe. Choosing the smallest one is a cost decision, not a coverage decision.
4. **Selecting an integration test also builds the package's binaries.** `cargo test --test X --no-run` is not the minimal link it appears to be, which makes it a better gate than expected and makes `--bin` alternatives a subset rather than a saving.
5. **A path filter is a claim about causality.** Writing one forces the question of what can actually move the artifact under test, and the first draft here missed the C shim that produces the very symbols the archives resolve.

## 8. Follow-up Actions

- **Fork PRs do reach the GB10 runner** (4.4). Inherited from the existing jobs, not introduced here. Worth a hardening issue covering a head-repository check on all three GB10 jobs, a tighter approval policy, or a dedicated runner account.
- **Disk pressure.** This adds a fourth permanent target directory on a volume at 93 percent; `~/.cargo-target` is already 39 GB with nothing pruning it, on the same host that cuts releases.
- **The `-lc` root cause** is open (section 6). Whoever resolves it should also revisit the "Measured, not assumed" claim in the `build.rs` comment.
- **Issue #1304** clears the dead-code backlog and widens `xla-compile` to `-D warnings`.

## References

- Issue #1303 (this work), #1274 (the link failure), #1275 (the `-lc` fix), #1282 (the compile gate this extends), #1304 (the lint half of the same exclusion list)
- `.github/workflows/ci.yml`, `build.rs:143-176`, `src/lib/mlxcel-xla/build.rs`
