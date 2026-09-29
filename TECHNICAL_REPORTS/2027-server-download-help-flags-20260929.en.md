# Technical Report: PR #2027 - mlxcel-server download Accepts --help, -h and --usage

**Date**: 2026-09-29
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

PR #2027 fixes `mlxcel-server download --help` (and `-h`, `--usage`) exiting 2 with "unexpected argument" instead of printing help. The parent `Cli` command's hand-declared help argument, needed because llama-server b10621 spells help `-h`/`--help`/`--usage` and clap's generated help argument cannot carry an alias, was not marked `global`, so `disable_help_flag` left the `download` subcommand with no help flag of its own. Adding `global = true` fixes all three spellings on `download` while keeping `mlxcel-server --help`'s own output byte-identical, verified by diffing against a pre-fix capture.

## 1. Problem Statement

### 1.1 Background

Both `mlxcel` and `mlxcel-server` parse `DownloadArgs` (`src/downloader/cli.rs`) for the `download` subcommand. `mlxcel download --help` worked, and `mlxcel-server --help`/`help download` worked, but `mlxcel-server download --help` did not: clap reported "unexpected argument '--help' found" and exited 2. The parent `Cli` struct sets `disable_help_flag = true` and declares its own `-h`/`--help`/`--usage` argument by hand (`action = clap::ArgAction::Help`) because clap 4's derive cannot attach `visible_alias` to its auto-generated help flag, and b10621 (the llama-server release mlxcel-server targets for CLI compatibility) spells the flag all three ways.

### 1.2 Existing Issue

- **No help flag on `download`**: `disable_help_flag` propagates the *absence* of clap's generated help argument into subcommands under clap 4.6, but a hand-declared replacement argument only follows subcommands when explicitly marked `global = true`. Without that marker, `download` had no help argument at all: not the disabled default, and not the hand-declared substitute.
- **Silent asymmetry**: `mlxcel-server --help`, `mlxcel-server help download`, and `mlxcel download --help` all worked, which made the missing case easy to miss during manual testing; only the direct `mlxcel-server download --help` invocation failed.

### 1.3 Risk Assessment

Low. `global = true` only changes where clap looks for the existing help argument; it adds no new argument, and `ArgAction::Help` already short-circuits parsing before any of `download`'s own logic runs. The one behavior that must not change is the top-level `--help` rendering, since `global` arguments are documented once and can shift heading placement; this was verified directly rather than assumed.

## 2. Change Summary

| Item | Value |
|------|-------|
| Files changed | 1 (`src/bin/mlx_server.rs`) |
| Commits | 1 |
| Lines added / deleted | 29 / 0 |

- Added `global = true` to the `#[arg(...)]` attribute on `Cli::help` (the hand-declared `-h`/`--help`/`--usage` argument), making it reachable as `mlxcel-server download --help`/`-h`/`--usage` in addition to the top-level spellings.
- The rationale is recorded as a plain `//` comment immediately below the field's existing `///` doc comment, not appended to the doc comment itself. Clap renders `///` doc comments as the argument's own `--help` description text, so extending it would have changed `mlxcel-server --help`'s rendered output for the `-h, --help` line; a `//` comment carries the same information for future readers without being compiled into help text.
- Added `the_download_subcommand_accepts_all_three_help_spellings` beside the existing `the_download_subcommand_does_not_take_negative_numbers` test, looping over `["--help", "-h", "--usage"]` and asserting `Cli::try_parse_from(["mlxcel-server", "download", flag])` returns `ErrorKind::DisplayHelp` for each. Confirmed this test fails on pre-fix code with `ErrorKind::UnknownArgument` instead.

## 3. Technical Decisions

### 3.1 `global = true` over a per-subcommand `ServerDownloadArgs` wrapper

**Context**: The issue proposed two approaches: first, marking the existing hand-declared help argument `global`; fallback, introducing a server-local `ServerDownloadArgs` struct that flattens `DownloadArgs` and re-declares the same hand-built help argument, used only if the first approach changed `mlxcel-server --help`'s output.

**Rationale**: `global = true` was sufficient and the byte-identity check (`diff` against a pre-change capture of `mlxcel-server --help`) passed once the explanatory text was moved out of the doc comment. The fallback wrapper was not needed, avoiding a second struct and a second hand-maintained copy of the same four `#[arg(...)]` attributes that would need to stay in sync with the parent's.

### 3.2 Plain comment placement to protect rendered help text

**Context**: The field's existing `///` doc comment ("Print usage and exit. Declared by hand, with `disable_help_flag`...") is not just documentation; clap compiles it into the `-h, --help` line's description in every `--help` invocation.

**Rationale**: The first implementation attempt appended the `global = true` rationale to that doc comment, which passed compilation and both new-flag checks but failed the top-level `--help` byte-identity diff (`diff` reported a one-line change on the `-h, --help` description). Moving the same explanation to a `//` line directly above the `#[arg(...)]` attribute keeps the rationale in the source for future maintainers without it reaching the compiled help text, and the subsequent diff was clean.

## 4. Validation

- `mlxcel-server --help` output diffed byte-for-byte against a capture taken before the change: no difference.
- `mlxcel-server download --help`, `-h`, `--usage` and top-level `--help`, `-h`, `--usage`: all exit 0.
- `mlxcel-server download -1`: still `UnknownArgument`, unaffected by the fix (per-subcommand `allow_negative_numbers` is unrelated to `global` help propagation).
- `cargo test --release --features metal,accelerate --bin mlxcel-server the_download_subcommand`: 2/2 passed (the new test and the existing negative-number regression test).
- `cargo test --release --features metal,accelerate --test cli_help_consistency`: 27/27 passed.
- `cargo clippy --release --features metal,accelerate --bin mlxcel-server --tests -- -D warnings`: clean.
- `cargo fmt --check` and `scripts/ci/check_cross_repo_refs.py`: clean.
- Re-verified all of the above after rebasing onto `origin/main` post-merge of PR #2023 (which touched `src/main.rs` only, not `src/bin/mlx_server.rs`), confirming no interaction between the two changes.

## 5. Related Work

- Issue #2025: source issue for this change.
- Issue #1448: original rationale for the hand-declared help argument (b10621's `-h`/`--help`/`--usage` spelling).
- Issue #1459 / `allow_negative_numbers`: the per-command setting whose non-propagation to subcommands motivated the `download -1` regression test that this PR's fix does not disturb.
