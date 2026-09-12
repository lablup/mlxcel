# Technical Report: PR #1872 — Safe WebUI model library operations

**Date**: 2026-09-13
**Status**: Open PR; focused CPU/fake validation complete; root real-model and broad workspace gates pending
**Languages**: Rust, JSON/OpenAPI fixtures
**Risk Level**: High
**Implementation snapshot**: `466c6071` before report-only commit

## Executive Summary

[PR #1872](https://github.com/lablup/mlxcel/pull/1872) implements the issue #1841 backend for WebUI model-library mutations in epic #1834. It adds authenticated WebUI download and managed-cache removal adapters on top of the existing router pool and lifecycle coordinator, while keeping startup state shared with #1838 and preserving compatibility routes.

The highest-risk parts are disk mutation and Hub access. The implementation keeps WebUI downloads anonymous, bounded, and fd-relative until atomic publication, and confines deletion to managed cache snapshots through a descriptor-anchored private quarantine rather than deleting user model roots or preset/external paths.

## 1. Problem Statement

The WebUI needs to download and remove models without turning the browser layer into a second model registry or a privileged filesystem shell. Existing router compatibility endpoints already had download/remove behavior, but they were not expressed as common WebUI operations with idempotency, bounded admission, progress, cancellation, and terminal results.

Disk deletion is irreversible and high blast radius. A model unload is not the same action as deleting a checkpoint, and deletion must not follow symlinks, race active writers/providers, or apply to models-dir/preset/external artifacts. WebUI downloads also must not silently consume ambient HuggingFace credentials just because the server host has CLI tokens.

## 2. Technical Decisions

### Shared coordinator authority instead of new state

`webui::library::routes()` mounts under `/ui-api/v1` and uses `RouterServerState` directly. Download and removal actions enter the same `RouterPool` and `LifecycleCoordinator` as other WebUI operations, so idempotency, bounded active operations, cancellation state, SSE operation events, and catalog rescan behavior have one authority.

Compatibility `POST /models` now enqueues through the same download primitive, while preserving its wire schema. The route no longer performs an unbounded synchronous metadata request before admission; duplicate and queue decisions happen first.

### Anonymous immutable downloads

WebUI downloads accept a public HuggingFace repo id plus optional revision, not arbitrary URLs or local paths. Anonymous WebUI metadata uses direct reqwest requests with a response byte cap instead of `hf-hub` builders that read ambient token files; CLI download behavior still uses the environment token mode.

The worker resolves metadata once, pins transfers to the resolved SHA, bounds manifest siblings, selected files, and filename length, computes known byte totals from HEAD when available, and validates actual bytes written against known sizes before publication. The terminal operation result carries the resolved revision; the shared `RevisionRef` schema remains nullable for request compatibility.

### Descriptor-anchored staging and cancellation boundary

On Unix, router-managed downloads write into a unique private `.mlxcel-staging` directory opened by fd. Remote paths are created relative to directory descriptors with no symlink traversal, and publication uses atomic no-replace rename after the coordinator's `begin_publish` hook seals cancellation. Cancellation accepted before writer exit becomes terminal only after the worker observes it; a late cancel after publication linearization is refused and the completed snapshot can publish.

The completeness policy requires `config.json` plus non-empty selected safetensors and known-size byte agreement. This does not claim independent LFS checksum verification beyond metadata and transfer evidence available to the current downloader.

### Managed-cache deletion only

`POST /ui-api/v1/model-removals` keeps the frozen request shape of `model_id`, `expected_revision`, and `idempotency_key`; user-facing confirmation is owned by the WebUI page work, and the backend endpoint itself is the explicit deletion intent. The pool requires a cache-sourced, current, idle model entry and marks it operation-busy before spawning deletion, which prevents rescan from rebuilding a stale entry while removal owns the catalog slot.

Deletion opens the configured cache root, owner, target, and private quarantine with `O_NOFOLLOW`, verifies identities at the final pre-rename boundary and after quarantine rename, then recursively unlinks relative to held fds. Nested directory deletion carries expected dev/ino through open and final rmdir checks. If identity changes, deletion fails closed and leaves the quarantined snapshot for inspection; a same-UID process can still cause denial of service or a failed quarantine cleanup race, so this is not a claim of absolute hostile-same-user isolation.

## 3. Review Refinements

Parent pre-review found and the implementation addressed these high-risk items before publication:

- Cancellation after the last transfer chunk but before publish is now linearized by `begin_publish_operation`, with a deterministic regression that late cancellation is refused and the operation succeeds.
- Known HEAD sizes are checked against actual bytes before fd-relative rename, and metadata/file counts/names are bounded before selected downloads are admitted.
- Anonymous WebUI metadata no longer constructs `hf-hub` API builders that read saved token files; tests seed token files and env vars and assert no Authorization header on metadata, HEAD, or GET fake requests.
- Removal now reserves lifecycle state with `operation_busy`, so rescans preserve the original entry until the removal worker finishes.
- Parent-swap and nested stat/open deletion races have deterministic tests and fail without deleting the wrong tree.
- New hand-written modules are below the 500-line cap after extracting `anchored_delete.rs` and adjacent test modules.

## 4. Validation Record

| Gate | Result at report preparation |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate anchored_remove -- --nocapture` | Passed: managed-cache quarantine deletion and parent/final-unlink swap regressions |
| `cargo test --profile test-fast --features metal,accelerate anchored_publish -- --nocapture` | Passed: no-replace publish, owner/staging identity, symlink cleanup regressions |
| `cargo test --profile test-fast --features metal,accelerate anchored_delete -- --nocapture` | Passed: nested directory stat/open swap regression |
| Focused downloader/router/WebUI filters | Passed: anonymous transport, saved token negative, fd staging, known-size mismatch, manifest bounds, lifecycle fixtures, WebUI library routes, duplicate replay, queue saturation, publish cancel linearization, case-alias reject, absent-store no-mkdir, managed removal, and rescan guard |
| `cargo fmt --check` and `git diff --check` | Passed |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | Passed |
| `PATH=/tmp/mlxcel-webui-contract/bin:$PATH make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` | Passed: 43 WebUI contract fixtures, DTO drift, and schema strictness |
| `PATH=/tmp/mlxcel-webui-contract/bin:$PATH make verify-llama-compat verify-versions verify-kernel-dtype-keys` | Passed |
| Root-owned real SmolLM download/load/generate/delete acceptance | Not run by this unit |
| Full workspace serial `make verify-test`, workspace all-target clippy, CUDA/GB10 checks | Not run by this unit; GB10 runner was reported down and the maintainer approved skipping only that unavailable required runner gate |

The local evidence is deliberately CPU/fake-network scoped. It does not establish real HuggingFace transfer success, real model load/generation, CUDA behavior, or production browser UX; root owns those acceptance gates and merge decisions.

## 5. Change Summary

| Area | Change |
|---|---|
| WebUI API | Added authenticated `/ui-api/v1/downloads` and `/ui-api/v1/model-removals` route module using existing `RouterServerState` |
| Operation coordinator | Added bounded download admission, active download replay, cancellation registration, progress bytes, and publish sealing |
| Downloader | Added anonymous token mode, bounded anonymous metadata, fd-relative Unix download helper, immutable revision pinning, plan/progress hooks, and size validation |
| Cache source | Replaced router-managed removal with descriptor-anchored quarantine deletion for managed snapshots only |
| Router pool | Shared compatibility/WebUI download primitive, operation-busy removal reservation, rescan guard, and cache removal by model id |
| Contracts | Added full producer fixtures for running and succeeded download operations while preserving nullable `RevisionRef` compatibility |
| Tests | Added fake HTTP/token, filesystem race, lifecycle idempotency, cancellation, bounded queue, case alias, no-mkdir, and deletion race regressions |

Statistics at PR creation: 24 files changed, 3,576 insertions, 231 deletions. Implementation commit: `466c6071 feat: add safe WebUI model library operations`.

## 6. Learning Points and Follow-up

A WebUI library action is an authority boundary, not just a button over an existing helper. Admission must be cheap and bounded, network and disk work must occur outside pool locks, and the final write/delete step must recheck the filesystem identity it is about to mutate.

The remaining follow-up is acceptance, not hidden implementation work in this report: root should run the pinned small SmolLM real download/load/generate/delete driver, the CI-faithful full workspace gate, workspace all-target clippy, and any available CUDA/GB10 checks or document the runner outage waiver. The page layer should provide the visible deletion confirmation and distinguish unload from disk deletion without changing this backend wire contract.
