# Technical Report: PR #2028 - Correct Stale Doc-Comment Examples Across mlxcel and mlxcel-server

**Date**: 2026-09-29
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

PR #2028 fixes two doc-comment defects in `--help` output, both found while implementing and reviewing PR #2023. `mlxcel inspect --help`'s Examples block showed invocations without the required `-m`/`--model` flag, which fail to parse when run verbatim. `mlxcel-server`'s `download` subcommand summary was still a bare, unpunctuated line predating the trailing-period-plus-description style PR #2023 established for `main.rs`'s sibling summaries. Both are doc-comment-only changes with no logic impact.

## 1. Problem Statement

### 1.1 Background

Issue #1657 / PR #2023 unified doc-comment style across five subcommands of the `Commands` enum in `src/main.rs`: bare one-line summaries became a trailing period plus a short extended paragraph, and several subcommands gained copy-pasteable `Examples:` blocks. Two defects fell outside that PR's scope but share the same doc-comment subsystem and the same verification surface (`tests/cli_help_consistency.rs` plus manual `--help` inspection), so this repo's PR-scoping convention groups them into one issue and one PR.

### 1.2 Existing Issues

- **Item 1, `src/main.rs`**: `InspectArgs::model` (`#[arg(short, long, value_name = "PATH_OR_REPO_ID")]`, a required, non-positional option, confirmed by `mlxcel inspect --help` printing `Usage: mlxcel inspect [OPTIONS] --model <PATH_OR_REPO_ID>`) requires `-m`/`--model`. The `Commands::Inspect` doc comment's `Examples:` block instead showed bare positional invocations (`mlxcel inspect models/llama-3.2-1b-instruct-4bit`), which fail with `error: unexpected argument 'models/llama-3.2-1b-instruct-4bit' found` when run verbatim.
- **Item 2, `src/bin/mlx_server.rs`**: `Commands::Download`'s doc comment was still `/// Download a HuggingFace model repository snapshot` (no trailing period, no extended description), rendered verbatim under the `mlxcel-server download:` heading in `mlxcel-server --help`'s flattened output (`flatten_help = true`). PR #2023 brought `main.rs`'s equivalent `Commands::Download` summary up to the new style but intentionally left `mlx_server.rs` untouched, since #1657 scoped the change to `main.rs` only.

### 1.3 Risk Assessment

Low. Both changes are confined to `///` doc comments on `Commands` enum variants; neither alters argument parsing, validation, or runtime behavior. The main risk was rendering the wrong text into the wrong help surface (for example, leaking a long paragraph into a flattened top-level block where it does not belong), which was checked directly rather than assumed.

## 2. Change Summary

| Item | Value |
|------|-------|
| Files changed | 2 (`src/main.rs`, `src/bin/mlx_server.rs`) |
| Commits | 1 |
| Lines added / deleted | 11 / 4 |

- `src/main.rs`: all three `Commands::Inspect` Examples block invocations gained `-m` (`mlxcel inspect -m models/llama-3.2-1b-instruct-4bit`, plus the `--max-tokens` and `--cache-type-k`/`--cache-type-v` variants).
- `src/bin/mlx_server.rs`: `Commands::Download`'s doc comment changed from the bare one-liner to a trailing-period summary plus a short extended paragraph, reusing wording already present on `main.rs`'s `Download` variant (`Fetches an owner/name repo-id into the global mlxcel store...`), and gained `#[command(verbatim_doc_comment)]` to match how `main.rs` renders its own `Download` variant's line breaks.
- No new `Examples:` block was added to `mlx_server.rs`'s `Download` variant: `DownloadArgs`'s shared `after_help` (`src/downloader/cli.rs`) already renders a full Examples section under `mlxcel-server download --help`, the same reasoning #2023 applied when it left `main.rs`'s `Download` variant without its own Examples block.

## 3. Technical Decisions

### 3.1 Verifying premises against post-#2023 line numbers before editing

**Context**: The issue was filed against pre-#2023 line numbers and explicitly noted that #2023 (open at filing time) would shift them, instructing implementers to match by variant name (`Commands::Inspect`, `Commands::Download`) instead.

**Rationale**: By the time this issue was picked up, #2023 had merged (bringing `Commands::Download`'s `main.rs` doc comment to the new style) as had #2025 (adding `global = true` to `mlx_server.rs`'s help argument, in an unrelated part of the same file). Both `Commands::Inspect` in `main.rs` and `Commands::Download` in `mlx_server.rs` were re-located by variant name and re-verified against current source before editing, confirming both defects still reproduced exactly as described (the `Download` doc comment in `mlx_server.rs` was untouched by #2023's `main.rs`-only scope, and `Commands::Inspect`'s Examples block content was unchanged by #2023's insertions elsewhere in the enum).

### 3.2 `#[command(verbatim_doc_comment)]` on `mlx_server.rs`'s `Download` variant

**Context**: Without `verbatim_doc_comment`, clap reflows a multi-line doc comment's paragraph text to the terminal width, which can produce different line breaks from a hand-formatted paragraph, especially one intentionally kept in sync with another file's wording.

**Rationale**: `main.rs`'s `Download` variant already carries this attribute for the same paragraph. Adding it here keeps the rendered `mlxcel-server download --help` paragraph byte-for-byte matched to how `mlxcel download --help` renders the equivalent text, rather than leaving the two to reflow independently and drift apart wordwise over time.

### 3.3 Confirming the extended paragraph does not leak into the flattened top-level help

**Context**: `mlx_server.rs`'s parent `Cli` sets `flatten_help = true`, which renders each subcommand's full help under its own heading inside `mlxcel-server --help`. A long about paragraph appended to a doc comment risks appearing somewhere it was not intended, unlike a short `about` (first line).

**Rationale**: Rather than assume clap's about/long_about split behaves as expected, the flattened block was inspected directly: only `Download a HuggingFace model repository snapshot.` (the first line) renders under the `mlxcel-server download:` heading in `mlxcel-server --help`; the full extended paragraph, including its reference to "the Examples section below," renders only under `mlxcel-server download --help`, where DownloadArgs's Examples block does immediately follow. This confirmed the cross-reference in the new paragraph is accurate in the surface where it actually appears.

## 4. Validation

- `mlxcel inspect -m models/llama-3.2-1b-instruct-4bit`: parses cleanly, no "unexpected argument" error; proceeds to model resolution and exits 1 with a download 404, expected since the example path in the doc comment is not a real local checkpoint or HuggingFace repository in this environment.
- `mlxcel inspect --help`: Examples block now shows `-m` in all three lines.
- `mlxcel-server --help`: `download` summary now ends with a trailing period (`Download a HuggingFace model repository snapshot.`).
- `mlxcel-server download --help`: full extended paragraph renders, followed by the existing shared Examples section.
- `cargo test --release --features metal,accelerate --test cli_help_consistency`: 27/27 passed.
- `cargo clippy --release --features metal,accelerate --bin mlxcel --bin mlxcel-server --tests -- -D warnings`: clean.
- `cargo fmt --check` and `scripts/ci/check_cross_repo_refs.py`: clean.

## 5. Related Work

- Issue #2024: source issue for this change.
- Issue #1657 / PR #2023: established the trailing-period-plus-description doc-comment convention this PR extends to `mlx_server.rs`.
- Issue #2025 / PR #2027: prior fix in the same file (`src/bin/mlx_server.rs`), landed immediately before this PR in the same chain; confirmed no overlap since it touched only the `Cli::help` argument, not the `Commands::Download` doc comment.
