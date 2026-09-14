# Technical Report: PR #1872 — Safe WebUI model library operations

**Date**: 2026-09-13
**Status**: Open PR / status:review; source `84f1aeb2` passed full, strict real-model and hosted acceptance; pending central merge
**Languages**: Rust, JSON/OpenAPI fixtures
**Risk Level**: High
**Implementation snapshot**: integrated baseline `3cb4817d` plus the owned-drain revision correction described below; earlier repair checkpoints `d688ea4f` and `b7555005`

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

The completeness policy requires `config.json`, at least one selected safetensors weight file, known-size byte agreement where metadata/headers provide a size, and LFS SHA-256 metadata for every safetensors file on the anonymous WebUI path. The stream is hashed before fd-relative rename, so same-size corrupted weights fail before publication.

### Managed-cache deletion only

`POST /ui-api/v1/model-removals` keeps the frozen request shape of `model_id`, `expected_revision`, and `idempotency_key`; user-facing confirmation is owned by the WebUI page work, and the backend endpoint itself is the explicit deletion intent. The pool requires a cache-sourced, current, idle model entry and marks it operation-busy before spawning deletion, which prevents rescan from rebuilding a stale entry while removal owns the catalog slot.

Deletion opens the configured cache root, owner, target, and private quarantine with `O_NOFOLLOW`, verifies identities at the final pre-rename boundary and after quarantine rename, then recursively unlinks relative to held fds. Nested directory deletion carries expected dev/ino through open and final rmdir checks. If identity changes, deletion fails closed and leaves the quarantined snapshot for inspection; a same-UID process can still cause denial of service or a failed quarantine cleanup race, so this is not a claim of absolute hostile-same-user isolation.

## 3. Review Refinements

Parent pre-review found and the implementation addressed these high-risk items before publication:

- Cancellation after the last transfer chunk but before publish is now linearized by `begin_publish_operation`, with a deterministic regression that late cancellation is refused and the operation succeeds.
- Metadata now requests `?blobs=true`; safetensors without LFS SHA-256 are rejected, streams are hashed before fd-relative rename, known metadata/HEAD sizes are checked against actual bytes, and metadata/file counts/names are bounded before selected downloads are admitted.
- Anonymous WebUI metadata no longer constructs `hf-hub` API builders that read saved token files; tests seed token files and env vars and assert no Authorization header on metadata, HEAD, or GET fake requests.
- Removal now reserves lifecycle state with `operation_busy`, rechecks after acquiring the per-entry operation guard, and rejects equality plus ancestor/descendant overlaps from configured models-dir or preset/non-cache entries so cache deletion cannot remove a user/preset-owned nested snapshot.
- Parent-swap and nested stat/open deletion races have deterministic tests and fail without deleting the wrong tree.
- New hand-written modules are below the 500-line cap after extracting `anchored_delete.rs` and adjacent test modules.

## 4. Validation Record

| Gate | Result at report preparation |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate anchored_remove -- --nocapture` | Passed: managed-cache quarantine deletion and parent/final-unlink swap regressions |
| `cargo test --profile test-fast --features metal,accelerate anchored_publish -- --nocapture` | Passed: no-replace publish, owner/staging identity, symlink cleanup regressions |
| `cargo test --profile test-fast --features metal,accelerate anchored_delete -- --nocapture` | Passed: nested directory stat/open swap regression |
| `cargo test --lib --profile test-fast --features metal,accelerate downloader::tests:: -- --nocapture` | Passed: 71 passed, 1 ignored; fake anonymous metadata/HEAD/GET, saved-token negative, `?blobs=true`, metadata status, offline, loopback read timeout, simulated ENOSPC writer cleanup, disconnect truncation, checksum, no-weight completeness, path filtering, size mismatch, and token-mode tests |
| `cargo test --lib --profile test-fast --features metal,accelerate server::router_models::router_models_tests:: -- --nocapture` | Passed: 50 passed; shared download queue/idempotency/revision-alias/progress/cancellation, retry-after-failure, managed removal reservation, equality and descendant physical-overlap blocks, in-flight compatibility removal, and rescan guard |
| `cargo test --lib --profile test-fast --features metal,accelerate router_lifecycle_tests:: -- --nocapture` | Passed: 16 passed; whole producer comparisons for download operation/event fixtures plus lifecycle coordinator regressions |
| `cargo test --lib --profile test-fast --features metal,accelerate router_cache:: -- --nocapture` | Passed: 11 passed; descriptor-anchored publish/remove/delete race regressions |
| Focused route filters | Passed: `ui_download_route_replays_same_idempotency_key` and `ui_model_removal_route_matches_operation_accepted_fixture` |
| `cargo fmt --check` and `git diff --check` | Passed |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | Passed |
| `PATH=/tmp/mlxcel-webui-contract/bin:$PATH make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` | Passed: 43 WebUI contract fixtures, DTO drift, and schema strictness |
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

Statistics after repair checkpoint `d688ea4f`: the latest cycle-2 source commit changes 4 files with 437 insertions and 12 deletions on top of the prior repair/docs head. Original implementation commit: `466c6071 feat: add safe WebUI model library operations`; repair commits: `b7555005 fix: harden WebUI model library edge cases` and `d688ea4f fix: close WebUI library edge-case acceptance gaps`.


## 6. Fake Acceptance Matrix at Source Repair Checkpoint

| Required fake case | Evidence in this PR | Status |
|---|---|---|
| Public valid download | `a_download_emits_the_b10621_event_sequence_and_lands_in_the_cache`; `anonymous_fd_download_sends_no_ambient_credentials_on_metadata_head_or_get` | Covered with fake downloader and fake HTTP/fd path |
| Bad repo / invalid revision / gated or private / 404 | `anonymous_fd_download_maps_metadata_status_without_publishing` covers 403 guidance and 404 missing repo/revision behavior | Covered for metadata-status handling |
| Offline | `anonymous_fd_download_offline_rejects_before_metadata_request` | Covered before metadata request |
| Timeout | `fd_stream_timeout_keeps_partial_private_and_unpublished` uses a loopback server that sends headers then stalls under a short client read timeout and verifies no final file or partial debris is left | Covered with loopback fake transport |
| Disconnect / truncation | `anonymous_fd_download_rejects_disconnect_truncation_before_publish`; `stream_file_rejects_known_size_mismatch_before_publish` | Covered |
| Disk full | `fd_stream_enospc_writer_cleans_partial_without_publishing` injects `ENOSPC` through a test-only writer factory around the production fd stream cleanup path | Covered as simulated ENOSPC; physical disk exhaustion was not run |
| Checksum and completeness | `anonymous_fd_download_rejects_same_size_checksum_mismatch_before_publish`; `anonymous_fd_download_rejects_metadata_without_weight_files_before_transfer`; `selected_safetensors_requires_lfs_sha256` | Covered for the anonymous WebUI fd path |
| Cancel/retry | `cancel_after_publish_linearization_is_refused_and_download_completes`; `remove_cancels_an_in_flight_download`; duplicate replay tests; `failed_download_can_retry_same_repo_with_new_idempotency_key` verifies a failed first operation can be retried successfully with a new operation id and one catalog entry | Covered for cancel, replay, and retry-after-failure |
| Restart / staging reconciliation | `killed_writer_and_terminal_history_reconcile_across_processes` uses three owned Rust test processes, real RouterPool/coordinator and anchored publication, and a fake local file writer; it checks killed running/queued work, session-local operation reset, explicit retry and one published catalog entry after another restart | CPU test-process coverage; root observed production CLI restart at `3cb4817d`, while the corrected complete real flow awaits rerun |
| Duplicate action | `duplicate_download_with_same_idempotency_key_replays_active_operation`; `active_download_replays_resolved_revision_alias`; `duplicate_download_idempotency_key_rejects_different_payload_before_alias_replay`; `ui_download_route_replays_same_idempotency_key` | Covered |
| Bounded queue saturation | `download_admission_rejects_queue_saturation_before_worker_network` | Covered |

## 7. Learning Points and Follow-up

A WebUI library action is an authority boundary, not just a button over an existing helper. Admission must be cheap and bounded, network and disk work must occur outside pool locks, and the final write/delete step must recheck the filesystem identity it is about to mutate.

The remaining follow-up is acceptance, not hidden implementation work in this report: root should run the pinned small SmolLM real download/load/generate/delete driver, the CI-faithful full workspace gate, workspace all-target clippy, and any available CUDA/GB10 checks or document the runner outage waiver. The page layer should provide the visible deletion confirmation and distinguish unload from disk deletion without changing this backend wire contract.

## 7. Bounded restart preparation (2026-09-14)

The regression terminates only its owned child after observing a real running writer and queued operation. The interrupted private stage survives both restarts byte-for-byte and never becomes a catalog entry. A new server instance has no old operation history: authenticated operation GET returns the entire canonical 404 response before new admission, the same idempotency key is not replayed, and operation identity is compared as `(server_instance_id, operation_id)` rather than assuming globally unique ID strings. A new explicit retry uses fresh anchored staging, publishes once, and retains the same catalog model identity across the next process restart. Terminal history is likewise absent in that new session. No model is loaded; fake weight bytes are not a valid checkpoint and do not validate transport, checksum, production CLI startup or power-loss durability.

The independent root CPU gate at head 42186c6a passed workspace all-target clippy, 43 contract fixtures, structural checks, formatting and diff checks. This does not replace the pending post-integration full test and real-checkpoint gates. No automatic abandoned-stage cleanup or resume behavior was added.

Validation of this delta: the exact `server::router_cache::restart_tests::` filter passed 2 tests, and a second execution of the parent restart test passed. Scoped library/test clippy, formatting, diff checks and 43 strict contract fixtures passed. The first fixture attempts incorrectly assumed globally unique operation strings and null optional error fields; execution/review caught both assumptions and only test expectations changed. Independent bounded correctness and security reviews have no remaining HIGH/CRITICAL finding in this test/docs delta.

## 8. Integrated acceptance and unload correction (2026-09-14)

Integration checkpoint `3cb4817d` is based on centrally merged #1838/main `7e4577b1`. Library routes are mounted once inside the existing secured API-prefix composition. A shared pure policy drives bootstrap capabilities and mutation admission from actual cache authority, mode, offline policy and descriptor-platform support; single-model mode remains read-only and observation never creates cache directories. All 46 merged fixtures, including upstream Unicode/error cases, and 48 frontend tests passed. Independent integration reviews cleared the bounded source scope.

Root's complete gate at `3cb4817d` passed 11,299 unique top-level tests with 361 ignored, plus two successful nested restart-child runs. The raw 11,301 aggregate includes those subprocess summaries and must not be presented as unique top-level coverage. Workspace all-target clippy, 46 strict fixtures, structural/format checks, feature-disabled compilation and both binaries' relocated empty/TTY checks passed. This is evidence for the pre-correction checkpoint, not a substitute for rerunning after the fix below.

The first isolated real driver downloaded and hash-verified the pinned public SmolLM checkpoint, restarted the production CLI, verified changed server identity and old-operation 404, rediscovered the same complete unloaded catalog identity, loaded and generated nonempty chat. Unload then failed with `stale_revision` expected3/current4; removal was not attempted. Both owned server processes exited0, and the new temporary checkpoint was retained. This failure is preserved, not relabeled success.

The defect already existed in merged main: unload validates the caller's revision, then its own Ready-to-Draining transition advances the revision, and its post-drain check incorrectly compares against the original token. A generic harness retry would hide this by retrying an already-draining model. The repair returns the owned drain revision atomically under the lifecycle mutex and rechecks that token/current entry after waiting. Shared revision authority means +1 cannot be assumed. RequestLease completion does not change revision. Only callers supplying a revision precondition (WebUI and explicit eviction) use the new token; legacy/shutdown callers retain identity-only cleanup, including failed-worker cleanup. The real harness is unchanged.

The loaded fake-provider regression failed before the repair with the same expected3/current4 error as the real run. After the final compatibility refinement, five unload/token tests, 16 lifecycle tests and 50 router fake tests passed. Coverage includes a held request lease, rescan preserving the current loaded entry, observed worker exit, genuine external revision/registry replacement rejection, shared revision authority and legacy failed-worker cleanup. Independent correctness/security rechecks found no remaining HIGH/CRITICAL issue in this bounded repair.

Final scoped library/test clippy, formatting and diff checks passed after the None-preserving refinement. Corrected full workspace and real-checkpoint acceptance remain root-owned and pending.

## 9. UI rebase and unresolved full-gate boundary (2026-09-14)

The branch was rebased onto merged UI PR #1863/main `b8d10fb1`, retaining the exact `@lablup/ui-common` version `0.1.0-alpha.19`, shared security/startup behavior and all 46 contract fixtures. The only rebase conflict was the generated bundle manifest. It was resolved through the canonical builder, not by hand-merging assets. Range comparison preserves all ten library commits unchanged except the generated manifest source digest; library/router/lifecycle production files are byte-identical to `857937ca`. The merged package lock, components, styles and generated JavaScript/CSS remain unchanged from main.

Root's latest full gate at `857937ca` failed/incomplete after the core configured-depth DFlash test aborted during a confirmed Metal firmware progress-timeout recovery. The diagnostic does not establish either an external process or this WebUI repair as the cause. The historical full pass at `3cb4817d` does not replace this failed latest gate, and the GB10 outage waiver does not waive it. Root's later CPU-only workspace all-target clippy, 46 fixtures, structural/format checks and feature-disabled compilation passed at `857937ca`; the corrected actual library driver was not rerun.

Further GPU, browser and full-suite execution is paused until the user confirms a quiet window. This rebase uses only CPU/fake tests and frontend unit/contract/build checks. Actual Safari, VoiceOver and native 200% manual checks are deferred by the user until the complete implementation is ready; they remain pending, not passed and not a per-unit blocker.

CPU-only rebase verification passed: unload/token tests5, pure policy1, secured fake-router9, owned restart2 (plus two nested child runs), scoped clippy/format checks, all46 strict fixtures and76 frontend unit tests, type/lint checks and canonical deterministic bundle verification. No GPU, browser or full-suite execution occurred in this unit rebase.

## 10. Latest-main integration and hosted portability repair (2026-09-14)

The branch is now rebased onto main `e9ee5f044fa28c14db50009fffbe02283513c3ae`, including #1888 shared SwitchGLU migration, #1889 WebUI feature-disabled helper gates, #1882 f16 attention-range tests and #1890 Phixtral narrowing. All eleven pre-existing branch commits are range-diff equivalent; no merge conflict or manual asset change occurred. Upstream app/auth/router helper gates remain intact, alongside the single secured library mount, shared policy, exact ui-common alpha.19 and 46 fixtures. Historical full/GPU results do not validate this new core baseline.

The GB10 runner subsequently executed the `71552f8c` hosted jobs: clippy failed on a same-type device-ID cast on Linux, and feature-disabled compilation failed on helpers now gated by upstream #1889. The earlier outage waiver does not apply to these executed failures. The local portability repair centralizes the existing cast in one documented helper with only a function-local unnecessary-cast allowance: Darwin's signed device representation is preserved, Linux's u64 representation remains unchanged, and neither comparison narrows identity or adds unsafe code. Temporary-directory regressions check matching identity, independent device/inode mismatch and child replacement; Darwin-specific cases check signed device values. Root's bounded read-only review cleared this delta. Actual hosted Linux clippy and feature-disabled results remain required after publication.

The user has now confirmed GPU availability; root exclusively owns the next GPU/full and real-acceptance runs, while this unit remains CPU-only. The failed first real unload and latest Metal firmware timeout remain preserved; unchanged real download/restart/load/unload/delete acceptance must still be rerun by root. No existing user checkpoint is removed by this preparation.

CPU preparation passed: unload5, policy1, secured router9, owned restart2 plus two child runs, anchored filesystem13 (including both new identity regressions), normal scoped library/test clippy, feature-disabled library/binary clippy, format/diff checks, 46 strict contracts, 76 frontend tests, type/lint and deterministic bundle verification. No GPU, browser or full-suite run was executed by this unit.

## 11. Final acceptance at source 84f1aeb2 (2026-09-14)

This section supersedes historical pending statuses above. After the user confirmed GPU availability, root's serialized chain at `84f1aeb2b4f5706bb0d63866b20ca48d74bf340a` exited0: 11,313 unique top-level tests passed, zero failed and 361 ignored across 124 summaries, plus two nested restart-child passes counted separately. Workspace all-target clippy passed with warnings denied in both default and feature-disabled configurations. All46 contracts, structural/format checks and both binaries' test-fast production relocated empty/offline/TTY checks passed. Actual hosted run [34844128193](https://github.com/lablup/mlxcel/actions/runs/34844128193) passed Linux clippy, OpenXLA feature compilation, WebUI bundle/feature-off and selected static checks; no GB10 exception was needed. Workflow-skipped CUDA sm70, MLX pin extraction and OpenXLA link are not execution claims.

The unchanged strict fresh-cache driver downloaded public `mlx-community/SmolLM-135M-Instruct-4bit` at `642e06afe3fab57fd6cc518637c471af0a569e1e`, verifying 75,789,919 SafeTensors bytes with SHA256 `e91560ee24b13eee6ddeb14879d728a90780053d69057b15ff199ccadfcfe33b`. Production CLI restart changed server identity, returned canonical404 for the old operation and retained the complete unloaded model's stable catalog ID. Actual load, nonempty 32-token generation, unload with `worker_exit_observed=true`, API deletion of only the newly downloaded checkpoint and an empty final catalog passed. Repetitive output demonstrates execution, not generation quality. Both owned servers exited0 without forced termination. Tested server binary SHA256: `4a1d0d3c2805b9b81f5e20e9d925337776d850702e8947896cf0e7d9fe7d6fe0`.

The first `3cb4817d` unload failure and `857937ca` Metal firmware timeout remain failures in the historical record; this later pass establishes neither an external cause nor a reboot claim. This final report update changes no source or bundle. Safari, VoiceOver and native 200% checks remain user-deferred until the whole implementation is ready, not passed. PR status remains review pending central merge.
