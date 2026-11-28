# Technical Report: PR #2023, Unify mlxcel --help subcommand summaries

**Date**: 2026-09-29
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

The `Commands` enum in `src/main.rs` had two documentation styles living side by side: seven subcommand variants (`Run`, `Inspect`, `Detect`, `Embed`, `Rerank`, `Rm`, `Tune`) carried a full doc-comment template (trailing period, short extended paragraph, an `Examples:` block), while five (`Generate`, `Serve`, `List`, `Arch`, `Download`) were bare one-liners with no period and no examples. This PR brings the five bare variants up to the same template, so `mlxcel --help` reads as one consistent surface instead of two.

## 1. Problem Statement

### 1.1 Background

Every subcommand's help text comes from the doc comment on its `Commands` enum variant. As the CLI grew, some variants (the ones with non-obvious behavior, like `Inspect`'s memory estimator or `Tune`'s autotuning cache) picked up a fuller template documenting an extended description and worked examples, while the five most commonly used verbs never got the same treatment.

### 1.2 Existing Issues

- **Issue 1**: `mlxcel --help`'s top-level `Commands:` listing mixed summaries ending in a period with summaries that did not, which reads as inconsistent formatting to anyone scanning the list.
- **Issue 2**: `mlxcel generate --help`, `mlxcel serve --help`, `mlxcel list --help`, and `mlxcel arch --help` gave no worked example, unlike every other subcommand's long help.

## 2. Technical Review

### 2.1 Code Quality

- **Test Coverage**: Unchanged. `tests/cli_help_consistency.rs` (27 tests) covers shared flag-group parity (TurboQuant KV-cache, speculative decoding) and cross-binary flag surfaces, not subcommand summary wording, so this change is outside its scope; all 27 passed unchanged both before and after.
- **Verification**: Every new example invocation was checked against the actual clap `Args` struct for that subcommand (`GenerateArgs` and its flattened `ModelOptions`/`GenerationOptions`/`SamplingOptions`, `ServeArgs`, `ListArgs`, `ArchArgs`), then cross-checked against the built release binary's real `--help` output, rather than assumed from memory or copied from a sibling subcommand's examples.

### 2.2 Compatibility & Dependencies

- **Breaking Changes**: None. Doc comments only; no `#[arg(...)]` definition, default, or parsing behavior changed.
- **New Dependencies**: None.

## 3. Technical Decisions

### 3.1 Download: extend the summary without duplicating the Examples block

**Context**: Unlike the other four bare variants, `Download`'s `Args` struct (`DownloadArgs` in `src/downloader/cli.rs`) already carries its own `#[command(after_help = "...")]` block with seven worked examples (default destination, bare-name org expansion, `--local-dir`, `--revision`, `--token`, `--force`, `--include`). That block is shared with `mlxcel-server download`, so it was already satisfying the "at least one example invocation" bar before this PR.

**Alternatives Considered**:

| Option | Pros | Cons |
|--------|------|------|
| Add a second `Examples:` block on the enum variant, matching the other four | Consistent literal presence of an `Examples:` block on every touched variant | Duplicates content already rendered later in the same `--help` output; two near-identical example lists back to back reads worse, not better |
| **Chosen: trailing period + short extended paragraph only, no new Examples block** | No duplication; the existing `after_help` block still renders and is pointed at from the new paragraph | The variant doc comment itself carries no example, which is a minor asymmetry with the other four |

**Rationale**: The acceptance criterion is that each subcommand's long help carries at least one example invocation, not that every variant's doc comment literally contains an `Examples:` heading. `download --help` already met that bar; adding a second block would have made the help text worse, not more consistent.

### 3.2 `verbatim_doc_comment` on `Download` despite no literal example block

**Context**: While rendering the result, the top-level `Commands:` listing showed `download   Download a HuggingFace model repository snapshot` with no trailing period, even though the source doc comment ended the first line with a period.

**Root cause**: Without `#[command(verbatim_doc_comment)]`, clap_derive's non-verbatim mode reflows the doc comment into a short `about` and a longer `long_about`. Once a second paragraph exists (an extended description below the summary), the automatic `about` derivation silently strips the trailing period from the first line. Every other touched variant already carried `verbatim_doc_comment` for its literal `Examples:` block formatting, which incidentally also preserved the period; `Download` had no such block, so the bug was only visible on that one variant.

**Rationale**: Added `#[command(verbatim_doc_comment)]` to `Download` as well, purely to keep the summary line intact. This was discovered empirically by building the binary and reading the actual rendered `--help` output, not by reading the clap_derive source.

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/main.rs`** (one representative variant; the other three bare-to-full conversions follow the same shape)

```rust
// Before
/// List downloaded models in the local store
#[command(visible_alias = "ls")]
List(ListArgs),

// After
/// List downloaded models in the local store.
///
/// Enumerates models downloaded into the global store (or
/// `--models-dir`), mirroring `ollama list`. The default table shows
/// NAME / SIZE / MODIFIED; the supported model-architecture catalog
/// lives under the separate `mlxcel arch` verb.
///
/// Examples:
///
///     mlxcel list
///     mlxcel list -v
///     mlxcel list --json
///     mlxcel list --sort modified
#[command(verbatim_doc_comment, visible_alias = "ls")]
List(ListArgs),
```

**Reason for change**: Matches the template already used by `Inspect`, `Detect`, `Embed`, `Rerank`, `Rm`, and `Tune`: a period-terminated summary, a short paragraph explaining behavior not obvious from the summary alone, and a literal `Examples:` block whose indentation and line breaks are preserved with `verbatim_doc_comment`.

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 1 (`src/main.rs`) |
| Lines added | +57 |
| Lines deleted | -8 |
| Tests added | 0 (existing `tests/cli_help_consistency.rs` already covers the help surface it targets; re-run and passed 27/27) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Documentation | 5 variants (`Generate`, `Serve`, `List`, `Arch`, `Download`) | Brought to the full doc-comment template already used by 7 other variants |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `33c6df6` | docs | Unify Commands enum subcommand summaries with the full-style template |

## 8. Follow-up Actions

### Future Improvements

- `Inspect`'s pre-existing `Examples:` block (not touched by this PR) omits the required `-m`/`--model` flag; the shown invocations (`mlxcel inspect models/llama-3.2-1b-instruct-4bit`) would actually fail with clap's "unexpected argument" error, since `--model` is a required option, not a positional. Worth a small follow-up fix.
- `src/bin/mlx_server.rs` carries its own `download` subcommand doc comment (`/// Download a HuggingFace model repository snapshot`, no trailing period) for the `mlxcel-server` binary, which is now inconsistent with the newly updated wording in `src/main.rs`. Out of scope here since this issue was scoped to the `Commands` enum in `src/main.rs` only; a follow-up could extend the same template to `mlxcel-server`'s subcommands.
