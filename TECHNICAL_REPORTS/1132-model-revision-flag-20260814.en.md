# Technical Report: PR #1132 - feat(cli): add --revision to the -m/--model resolver

**Date**: 2026-08-14
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Low

---

## Executive Summary

PR #1132 closes issue #1113. `mlxcel download` accepted `--revision` while the `-m/--model` resolver did not, so a user could fetch a pinned revision and then not run it by repo-id. `--revision <REV>` is now available on `generate`, `run`, `serve`, `inspect` and `mlxcel-server`.

The interesting part is not the plumbing, which the issue correctly described as already threaded end to end. It is the scope decision the issue demanded be made explicitly: **the mlxcel store is not revision-namespaced**, so a revision cannot be honoured everywhere. This PR honours it only where it can be honoured correctly and fails loudly everywhere else, rather than silently returning a revision nobody asked for.

That decision also surfaced a pre-existing defect: `mlxcel download --revision` can already return the wrong revision today, for the same reason. It is not introduced here and is filed as follow-up work.

---

## 1. Problem Statement

### 1.1 Background

`resolve_repo_id` already accepted `revision: Option<&str>` and threaded it into `store::hf_cache_snapshot` and `DownloadOptions`. Only the two public entry points hardcoded `None`, and the resolver's own doc comment recorded the gap deliberately:

> `revision` selects the HF-cache snapshot revision (branch / tag / commit); `None` means `main`. The CLI subcommands do not currently expose a `--revision` flag, so they pass `None`, matching `mlxcel download`'s default.

### 1.2 Existing Issues

- **A pinned revision was fetchable but not runnable.** `mlxcel download owner/name --revision v2` worked; `mlxcel run owner/name` then resolved against `main`.
- **The store cannot hold two revisions.** `store::model_dir_under` composes `<models_root>/<owner>/<name>` with no revision component.
- **The downloader silently skips a fetch into an occupied directory.** `download_repo_blocking` checks `snapshot_complete(&local_dir, &wanted)` against the requested revision's file list and returns early with "all expected files already present ..., skipping" when every wanted filename is present and non-zero. Revisions of a repo normally share filenames, so this returns the wrong revision while reporting success.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| A revision-qualified request silently answered with a different revision | High | High, absent the guards added here |
| A user believes `--revision` pinned a local path when it was ignored | Medium | Medium |
| Adding a store-layout change under a "good first issue" banner | Medium | Avoided by scoping |

---

## 2. Technical Review

### 2.1 Where a Revision Can Be Honoured

Reuse locations differ in whether they record a revision, and that difference drives the entire design.

| Location | Revision-aware | Treatment |
|---|---|---|
| `store::hf_cache_snapshot` | Yes (`refs/<rev>`, or a commit-named snapshot dir) | Consulted normally |
| Legacy `./models/<basename>` | No | Skipped for a revision-qualified request |
| mlxcel store `<owner>/<name>` | No | Skipped for reuse; refused as a download destination when occupied |
| Network fetch | Yes (`DownloadOptions.revision`) | Fetches the requested revision |

Skipping is not a limitation being papered over. A hit in a location with no revision provenance is indistinguishable from a hit on the wrong revision, so returning it would produce exactly the failure `--revision` exists to prevent.

### 2.2 Why the Post-Download Probe Is a Separate Function

`locate_cached_snapshot` answers "may this location answer a request for `revision`?" For a revision-qualified request the answer is no for any location without provenance. After a download, the question is different: "where did the bytes we just fetched land?" That is knowable, because the store directory was verified `SnapshotState::Absent` before the fetch, so whatever is there now is the revision that was requested.

Reusing the first function for the second question would have been a latent bug: the post-download probe would skip the store, miss, fall into the "clean re-download" recovery, fetch again, miss again, and finally return the "still incomplete afterwards" error for a snapshot that was in fact complete. `locate_landed_snapshot` consults the HF cache at that revision and then the store destination, and deliberately does not consult the legacy directory, which is never a download destination.

### 2.3 Public API Shape

`resolve_model_source(value)` keeps its single-argument form and delegates with `None`. The issue suggested adding the argument to both public entry points, but that function is a convenience wrapper with no in-tree callers, and widening it would break external callers for no gain. Only `resolve_model_source_with_override`, which every in-tree call site already uses, gains the parameter.

### 2.4 Compatibility

Every command line that worked before behaves identically. `revision` defaults to `None` at every call site, and with `None` the resolver's probe order and results are unchanged, which the retained `no_revision_with_existing_local_path_is_unchanged` and the untouched pre-existing tests pin.

---

## 3. Technical Decisions

### 3.1 Refuse Rather Than Fetch Into an Occupied Store

| Option | Pros | Cons |
|---|---|---|
| Fetch anyway | Simple; "works" in the common case | The fetch is skipped as "already present" and the caller silently gets the other revision |
| Silently reuse the store snapshot | No error path | Same silent-wrong-revision result, guaranteed rather than likely |
| Revision-namespace the store | Fully general | Changes an on-disk layout shared with `list`, `rm` and `download`; out of scope per the issue |
| **Chosen: refuse with both workarounds named** | The flag's promise stays exact: if a path is returned, it is that revision | A revision-qualified run can fail where a plain run would not |

The error names `mlxcel rm <repo>` and `--models-dir <PATH>`, so the user has a way forward in one read.

### 3.2 Error on `--revision` With a Local Path

Step 1 returns an existing path verbatim and has nothing to resolve a revision against. Erroring names the mistake, where ignoring would leave the user believing they had pinned something. This also keeps the flag's contract uniform: it either honours the revision or refuses, and never silently ignores it.

### 3.3 Split the Store-Layout Work Out

The issue said point 2 decides how big this is, and that if the store layout has to change it should be split. It does, so it was. This PR contains no layout change, and the follow-up carries both the namespacing and the pre-existing `mlxcel download --revision` collision that shares its root cause.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 12 |
| New tests | 5 |
| Store layout changes | 0 |
| Behavior changes without `--revision` | 0 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| Resolver | `src/downloader/resolver.rs` | `revision` parameter; revision gating in `locate_cached_snapshot`; new `locate_landed_snapshot`; two new errors; module-doc Revisions section; the stale "no `--revision` flag" comment replaced |
| CLI | `src/main.rs`, `src/commands/run.rs`, `src/bin/mlx_server.rs` | `--revision` on `ModelOptions`, `InspectArgs`, `ServeArgs`, `RunArgs`, `ServerArgs` |
| Call sites | `src/commands/{generate,chat,serve,inspect}.rs`, `src/bin/mlx_server.rs` | Threaded through all five; `ChatOptions` carries it for the REPL path |
| Tests | `src/downloader/resolver_tests.rs`, `src/commands/{generate,serve}_tests.rs` | 5 new tests; fixtures initialize the new field |
| Documentation | `CHANGELOG.md` | Entry under `## [Unreleased]` / `### Added` |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `55033e37` | feat | feat(cli): add --revision to the -m/--model resolver |
| `89e8cc39` | test | test: initialize the new revision field in the arg-struct fixtures |

---

## 5. Validation and Follow-up

### Passed

- `cargo test --profile test-fast --features cuda --lib downloader::resolver`: 33 passed, 0 failed. The five new tests cover the local-path refusal, the legacy-directory skip, the store skip, the unchanged no-revision local-path behaviour, and the occupied-store refusal (which also asserts the message is not a download failure, so nothing was fetched).
- `cargo test --profile test-fast --features cuda --test cli_help_consistency`: 25 passed, including `the_two_server_binaries_accept_the_same_flag_surface`, which is what keeps `mlxcel serve` and `mlxcel-server` from drifting on the new flag.
- `cargo test --profile test-fast --features cuda --bin mlxcel serve_`: 13 passed. `... validate_pipeline_parallel`: 5 passed.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- Against the built binary: `--revision` advertised with the intended wording on `generate` and `run`; `-m /tmp --revision v2` refused with the local-path error; an occupied store plus `--revision v2` refused with both workarounds named and no network request.

### A Note on the Clippy Run

The first clippy invocation was piped through `grep | tail`, which reports the exit status of the last pipeline stage rather than cargo's. It appeared to pass while cargo had in fact failed with two `E0063`s in the bin crate's test target, where `sample_generate_args` and `sample_args` build `ModelOptions` and `ServeArgs` with explicit field initializers. The run was repeated with the exit code captured directly, the two fixtures were fixed, and the result is recorded above. Worth remembering for any verification command whose output is filtered.

### Follow-up Candidates

- **Revision-namespace the mlxcel store.** This lifts every restriction above and fixes the pre-existing `mlxcel download --revision` collision, where a second revision requested into an occupied store directory is skipped as "already present" and the first revision is returned. That collision exists on `main` today and is independent of this change.
- **`--revision` for the resolver is refused rather than served when the store is occupied.** If the follow-up lands, this PR's `revision_store_occupied_error` becomes dead and should be removed with it.
