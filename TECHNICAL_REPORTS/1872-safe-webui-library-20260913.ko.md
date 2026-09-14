# 기술 리포트: PR #1872 — 안전한 WebUI 모델 라이브러리 작업

**작성일**: 2026-09-13
**상태**: 열린 PR; unload 수정과 ui-common 통합 유지, 최신 전체 게이트가 GPU firmware timeout으로 실패하여 소유자와 조율한 quiet window에서 전체·실모델 재검증 대기
**언어**: Rust, JSON/OpenAPI fixture
**위험도**: 높음
**구현 스냅샷**: 통합 기준 `3cb4817d`와 아래 owned-drain revision 수정; 이전 보완 체크포인트 `d688ea4f` 및 `b7555005`

## 요약

[PR #1872](https://github.com/lablup/mlxcel/pull/1872)는 에픽 #1834의 이슈 #1841에 해당하는 WebUI 모델 라이브러리 변경 작업의 백엔드를 구현합니다. 인증된 WebUI 다운로드 및 관리 캐시 삭제 어댑터를 기존 router pool과 lifecycle coordinator 위에 추가하고, #1838 시작 경로와 공유할 상태 형태 및 compatibility route를 유지합니다.

가장 위험한 부분은 디스크 변경과 Hub 접근입니다. 구현은 WebUI 다운로드를 익명·제한·fd-relative 방식으로 수행한 뒤 원자적으로 게시하고, 삭제는 descriptor-anchored private quarantine을 거치는 관리 캐시 snapshot으로 한정하여 사용자 모델 루트나 preset/external 경로를 삭제하지 않습니다.

## 1. 문제 정의

WebUI는 모델을 다운로드하고 제거할 수 있어야 하지만, 브라우저 계층이 두 번째 모델 레지스트리나 권한 있는 파일시스템 shell이 되어서는 안 됩니다. 기존 router compatibility endpoint에는 다운로드·삭제 동작이 있었지만, 멱등성, 제한된 접수, 진행률, 취소, 종료 결과를 갖춘 공통 WebUI operation으로 표현되지 않았습니다.

디스크 삭제는 되돌릴 수 없고 영향 범위가 큽니다. 모델 unload는 checkpoint 삭제와 다른 작업이며, 삭제는 symlink를 따라가거나 활성 writer/provider와 경쟁하거나 models-dir/preset/external artifact에 적용되면 안 됩니다. 또한 WebUI 다운로드는 서버 호스트에 CLI 토큰이 있다는 이유로 HuggingFace credential을 조용히 사용하면 안 됩니다.

## 2. 기술적 선택과 그 이유

### 새 상태가 아니라 공유 coordinator 권한 사용

`webui::library::routes()`는 `/ui-api/v1` 아래에 mount되고 `RouterServerState`를 직접 사용합니다. 다운로드와 제거는 다른 WebUI operation과 같은 `RouterPool` 및 `LifecycleCoordinator`로 들어가므로 멱등성, 제한된 활성 작업 수, 취소 상태, SSE operation event, catalog rescan 동작의 권한이 하나로 유지됩니다.

Compatibility `POST /models`도 wire schema는 유지하되 같은 다운로드 primitive로 enqueue합니다. 이제 route가 접수 전에 제한 없는 동기 metadata 요청을 수행하지 않고, 중복·queue 판단이 먼저 일어납니다.

### 익명 immutable 다운로드

WebUI 다운로드는 임의 URL이나 local path가 아니라 공개 HuggingFace repo id와 선택적 revision만 받습니다. 익명 WebUI metadata는 ambient token file을 읽는 `hf-hub` builder 대신 응답 byte cap이 있는 직접 reqwest 요청을 사용합니다. CLI 다운로드 동작은 기존 environment token mode를 유지합니다.

Worker는 metadata를 한 번 resolve하고 resolved SHA에 pin하여 전송하며, manifest sibling 수, 선택 파일 수, filename 길이를 제한하고, 가능한 경우 HEAD로 알려진 byte total을 계산한 뒤 실제 기록 byte와 비교하고 나서 게시합니다. 종료 operation result는 resolved revision을 담고, 공유 `RevisionRef` schema는 요청 호환성을 위해 nullable 그대로 둡니다.

### Descriptor-anchored staging과 취소 경계

Unix에서 router 관리 다운로드는 fd로 열린 고유 private `.mlxcel-staging` 디렉터리에 기록합니다. 원격 경로는 directory descriptor 기준으로 생성되고 symlink traversal을 하지 않으며, coordinator의 `begin_publish` hook이 취소를 봉인한 뒤 atomic no-replace rename으로 게시합니다. Writer가 관찰하기 전 접수된 취소는 worker가 종료를 확인한 뒤 terminal 상태가 되고, 게시 linearization 이후의 늦은 취소는 거절되어 완성된 snapshot이 게시될 수 있습니다.

완전성 정책은 `config.json`, 하나 이상의 선택된 safetensors weight 파일, metadata/header가 제공하는 알려진 크기와 기록 byte의 일치, 그리고 익명 WebUI 경로의 모든 safetensors에 대한 LFS SHA-256 metadata를 요구합니다. Stream은 fd-relative rename 전에 hash되므로 같은 크기의 손상된 weight도 게시 전에 실패합니다.

### 관리 캐시 삭제만 허용

`POST /ui-api/v1/model-removals`는 고정된 request shape인 `model_id`, `expected_revision`, `idempotency_key`를 유지합니다. 사용자에게 보이는 확인 modal은 WebUI page 작업 범위이고, backend endpoint 자체가 명시적인 삭제 의도입니다. Pool은 cache-sourced이고 현재 revision이 맞으며 idle 상태인 model entry만 허용하고, 삭제를 spawn하기 전에 operation-busy로 표시하여 removal이 catalog slot을 소유하는 동안 rescan이 stale entry를 재생성하지 못하게 합니다.

삭제는 설정된 cache root, owner, target, private quarantine을 `O_NOFOLLOW`로 열고, quarantine rename 직전과 rename 후 identity를 확인한 다음 held fd 기준으로 재귀 unlink를 수행합니다. 중첩 디렉터리 삭제는 예상 dev/ino를 open과 최종 rmdir 검사까지 전달합니다. Identity가 바뀌면 삭제는 fail-closed하고 quarantined snapshot을 조사용으로 남깁니다. 같은 UID 프로세스는 여전히 denial of service나 quarantine cleanup 실패 race를 유발할 수 있으므로, 이것은 hostile same-user를 절대적으로 격리한다는 주장이 아닙니다.

## 3. 리뷰 보완 사항

Parent pre-review에서 지적된 다음 고위험 항목을 PR 게시 전 반영했습니다.

- 마지막 전송 chunk 이후 publish 전 취소 race를 `begin_publish_operation`으로 linearize했고, 늦은 취소가 거절되며 operation이 성공하는 결정적 회귀 테스트를 추가했습니다.
- Metadata 요청은 `?blobs=true`를 사용하고, LFS SHA-256이 없는 safetensors는 거절하며, stream을 fd-relative rename 전에 hash하고, 알려진 metadata/HEAD 크기를 실제 byte 수와 비교하며, metadata와 선택 파일 수·파일명 길이를 제한한 뒤 다운로드를 진행합니다.
- 익명 WebUI metadata는 더 이상 저장된 token file을 읽는 `hf-hub` API builder를 생성하지 않으며, 테스트는 token file과 env var를 심고 metadata·HEAD·GET fake 요청에 Authorization header가 없음을 확인합니다.
- 제거 작업은 `operation_busy`로 lifecycle을 예약하고, per-entry operation guard 획득 후 다시 검사하며, 설정된 models-dir 또는 preset/non-cache entry와의 동일 경로 및 ancestor/descendant overlap을 거절하여 cache 삭제가 user/preset 소유 중첩 snapshot을 지우지 못하게 했습니다.
- Parent swap 및 중첩 stat/open 삭제 race는 결정적 테스트를 갖고 잘못된 tree를 삭제하지 않고 실패합니다.
- `anchored_delete.rs`와 인접 테스트 모듈로 분리하여 새 handwritten module을 500줄 이하로 유지했습니다.

## 4. 검증 기록

| 게이트 | 리포트 작성 시점 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate anchored_remove -- --nocapture` | 통과: 관리 캐시 quarantine 삭제 및 parent/final-unlink swap 회귀 |
| `cargo test --profile test-fast --features metal,accelerate anchored_publish -- --nocapture` | 통과: no-replace publish, owner/staging identity, symlink cleanup 회귀 |
| `cargo test --profile test-fast --features metal,accelerate anchored_delete -- --nocapture` | 통과: 중첩 디렉터리 stat/open swap 회귀 |
| `cargo test --lib --profile test-fast --features metal,accelerate downloader::tests:: -- --nocapture` | 통과: 71 passed, 1 ignored; fake anonymous metadata/HEAD/GET, saved-token 음성, `?blobs=true`, metadata status, offline, loopback read timeout, simulated ENOSPC writer cleanup, disconnect truncation, checksum, weight 없음 완전성, path filtering, size mismatch, token-mode 테스트 |
| `cargo test --lib --profile test-fast --features metal,accelerate server::router_models::router_models_tests:: -- --nocapture` | 통과: 50 passed; 공유 download queue/idempotency/revision-alias/progress/cancellation, 실패 후 재시도, 관리 removal reservation, 동일 경로 및 descendant physical-overlap 차단, in-flight compatibility removal, rescan guard |
| `cargo test --lib --profile test-fast --features metal,accelerate router_lifecycle_tests:: -- --nocapture` | 통과: 16 passed; download operation/event fixture 전체 producer 비교 및 lifecycle coordinator 회귀 |
| `cargo test --lib --profile test-fast --features metal,accelerate router_cache:: -- --nocapture` | 통과: 11 passed; descriptor-anchored publish/remove/delete race 회귀 |
| 집중 route 필터 | 통과: `ui_download_route_replays_same_idempotency_key`, `ui_model_removal_route_matches_operation_accepted_fixture` |
| `cargo fmt --check` 및 `git diff --check` | 통과 |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | 통과 |
| `PATH=/tmp/mlxcel-webui-contract/bin:$PATH make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` | 통과: WebUI contract fixture 43개, DTO drift, schema strictness |
| 루트 소유 실제 SmolLM 다운로드·로드·생성·삭제 acceptance | 이 유닛은 실행하지 않음 |
| 전체 workspace 직렬 `make verify-test`, workspace all-target clippy, CUDA/GB10 검사 | 이 유닛은 실행하지 않음; GB10 runner는 down 상태였고 maintainer가 해당 unavailable required runner gate만 생략 승인 |

로컬 증거는 의도적으로 CPU·fake-network 범위입니다. 실제 HuggingFace 전송 성공, 실제 모델 load/generation, CUDA 동작, production browser UX를 입증하지 않습니다. 해당 acceptance gate와 merge 판단은 루트가 소유합니다.

## 5. 변경 요약

| 영역 | 변경 |
|---|---|
| WebUI API | 기존 `RouterServerState`를 사용하는 인증된 `/ui-api/v1/downloads` 및 `/ui-api/v1/model-removals` route 모듈 추가 |
| Operation coordinator | 제한된 download admission, active download replay, cancellation 등록, progress bytes, publish sealing 추가 |
| Downloader | anonymous token mode, 제한된 익명 metadata, fd-relative Unix download helper, immutable revision pinning, plan/progress hook, size validation 추가 |
| Cache source | Router 관리 removal을 관리 snapshot 전용 descriptor-anchored quarantine deletion으로 교체 |
| Router pool | Compatibility/WebUI 공유 download primitive, operation-busy removal reservation, rescan guard, model id 기반 cache removal 추가 |
| 계약 | nullable `RevisionRef` 호환성을 유지하면서 running/succeeded download operation 전체 producer fixture 추가 |
| 테스트 | fake HTTP/token, 파일시스템 race, lifecycle idempotency, cancellation, 제한 queue, case alias, no-mkdir, deletion race 회귀 추가 |

보완 체크포인트 `d688ea4f` 이후 통계는 이전 보완/docs head 위에 최신 cycle-2 source commit 4개 파일 변경, 437줄 추가, 12줄 삭제입니다. 최초 구현 커밋은 `466c6071 feat: add safe WebUI model library operations`이고, 보완 커밋은 `b7555005 fix: harden WebUI model library edge cases` 및 `d688ea4f fix: close WebUI library edge-case acceptance gaps`입니다.


## 6. 소스 보완 체크포인트의 Fake Acceptance Matrix

| 요구 fake case | 이 PR의 증거 | 상태 |
|---|---|---|
| 공개 valid download | `a_download_emits_the_b10621_event_sequence_and_lands_in_the_cache`; `anonymous_fd_download_sends_no_ambient_credentials_on_metadata_head_or_get` | fake downloader 및 fake HTTP/fd 경로로 커버 |
| Bad repo / invalid revision / gated·private / 404 | `anonymous_fd_download_maps_metadata_status_without_publishing`이 403 안내와 404 repo/revision 없음 동작을 커버 | metadata status 처리 커버 |
| Offline | `anonymous_fd_download_offline_rejects_before_metadata_request` | metadata 요청 전 커버 |
| Timeout | `fd_stream_timeout_keeps_partial_private_and_unpublished`가 header를 보낸 뒤 body를 지연하는 loopback 서버와 짧은 client read timeout으로 최종 파일 및 partial debris가 남지 않음을 확인합니다 | loopback fake transport로 커버 |
| Disconnect / truncation | `anonymous_fd_download_rejects_disconnect_truncation_before_publish`; `stream_file_rejects_known_size_mismatch_before_publish` | 커버 |
| Disk full | `fd_stream_enospc_writer_cleans_partial_without_publishing`가 production fd stream cleanup 경로 주위의 test-only writer factory로 `ENOSPC`를 주입합니다 | simulated ENOSPC로 커버; 실제 물리 disk exhaustion은 실행하지 않음 |
| Checksum 및 completeness | `anonymous_fd_download_rejects_same_size_checksum_mismatch_before_publish`; `anonymous_fd_download_rejects_metadata_without_weight_files_before_transfer`; `selected_safetensors_requires_lfs_sha256` | 익명 WebUI fd 경로 커버 |
| Cancel/retry | `cancel_after_publish_linearization_is_refused_and_download_completes`; `remove_cancels_an_in_flight_download`; duplicate replay 테스트; `failed_download_can_retry_same_repo_with_new_idempotency_key`가 첫 실패 후 새 operation id로 재시도 성공 및 catalog entry 1개를 확인합니다 | cancel, replay, retry-after-failure 커버 |
| Restart / staging reconciliation | `killed_writer_and_terminal_history_reconcile_across_processes`가 소유한 Rust 테스트 프로세스 3개, 실제 RouterPool/coordinator와 anchored publication, 로컬 fake writer로 실행·대기 작업 중단과 세션 이력 초기화, 명시적 재시도 및 다음 재시작의 catalog entry 1개를 검증합니다 | CPU 테스트 프로세스 검증 및 루트의 `3cb4817d` production CLI 재시작 확인; 수정본 전체 실모델 흐름 재검증 대기 |
| Duplicate action | `duplicate_download_with_same_idempotency_key_replays_active_operation`; `active_download_replays_resolved_revision_alias`; `duplicate_download_idempotency_key_rejects_different_payload_before_alias_replay`; `ui_download_route_replays_same_idempotency_key` | 커버 |
| Bounded queue saturation | `download_admission_rejects_queue_saturation_before_worker_network` | 커버 |

## 7. 학습 포인트와 후속 조치

WebUI library action은 기존 helper 위의 버튼이 아니라 권한 경계입니다. 접수는 싸고 제한되어야 하며, network와 disk 작업은 pool lock 밖에서 수행되어야 하고, 마지막 write/delete 단계는 지금 변경하려는 filesystem identity를 다시 확인해야 합니다.

남은 작업은 이 리포트 안의 숨은 구현이 아니라 acceptance입니다. 루트는 pinned small SmolLM 실제 download/load/generate/delete driver, CI-faithful 전체 workspace gate, workspace all-target clippy, 사용 가능한 CUDA/GB10 검사 또는 runner outage waiver 기록을 수행해야 합니다. Page 계층은 이 backend wire contract를 바꾸지 않으면서 눈에 보이는 삭제 확인을 제공하고 unload와 disk deletion을 명확히 구분해야 합니다.

## 7. 제한된 재시작 검증 준비 (2026-09-14)

회귀 테스트는 실제 writer 실행과 queued operation을 관찰한 뒤 자신이 소유한 자식 프로세스만 종료합니다. 중단된 private stage는 두 번의 재시작 동안 바이트가 유지되며 catalog에 노출되지 않습니다. 새 server instance에서는 이전 operation 조회가 신규 작업 접수 전에 전체 canonical 404 응답을 반환하며, 같은 idempotency key도 이전 작업으로 replay되지 않습니다. operation ID 문자열의 전역 유일성을 가정하지 않고 `(server_instance_id, operation_id)`를 비교합니다. 명시적 재시도는 새 anchored stage에서 한 번만 publish하고 다음 프로세스 재시작에도 동일한 catalog model identity를 유지합니다. 완료된 작업 이력도 새 세션에는 남지 않습니다. 모델은 로드하지 않으며 fake weight는 유효한 checkpoint가 아닙니다. 실제 전송·checksum·production CLI 시작·전원 장애 내구성 검증을 대신하지 않습니다.

루트의 head 42186c6a CPU 게이트는 workspace all-target clippy, 43 contract fixture, 구조 검사, 포맷 및 diff 검사를 통과했습니다. 통합 후 전체 테스트와 실모델 게이트는 별도로 남아 있습니다. 버려진 stage의 자동 삭제나 resume 동작은 추가하지 않았습니다.

이번 변경 검증: 정확한 `server::router_cache::restart_tests::` 필터의 테스트2개와 부모 재시작 테스트의 추가 실행을 통과했습니다. scoped library/test clippy, 포맷, diff 검사와43 strict contract fixture도 통과했습니다. 초기 fixture는 operation 문자열의 전역 유일성과 optional 오류 필드의 null 직렬화를 잘못 가정했고 실행·리뷰에서 발견하여 테스트 기대값만 수정했습니다. 이 테스트·문서 변경에 대한 독립 correctness/security 리뷰의 HIGH/CRITICAL 잔여 사항은 없습니다.

## 8. 통합 검증과 unload 수정 (2026-09-14)

통합 체크포인트 `3cb4817d`는 중앙에서 머지한 #1838/main `7e4577b1` 기반입니다. 기존 보안·API prefix 구성 안에 library route를 한 번만 연결하고, 순수한 공유 정책으로 실제 cache 권한·mode·offline·descriptor platform에 따른 bootstrap capability와 mutation admission을 일치시켰습니다. 단일 모델 모드는 read-only이며 관찰 중 cache 디렉터리를 생성하지 않습니다. 상위 Unicode/error 사례를 포함한 fixture 46개와 frontend 테스트 48개를 통과했고 독립 통합 리뷰도 제한된 범위에서 완료했습니다.

루트의 `3cb4817d` 전체 게이트는 고유 top-level 테스트 11,299개 통과, 361개 ignored 및 별도 nested restart-child 실행 2개 통과입니다. 원시 합계 11,301에는 subprocess 결과가 포함되므로 고유 top-level 수로 보고하지 않습니다. Workspace all-target clippy, strict fixture 46개, 구조·포맷 검사, feature-disabled 컴파일과 두 바이너리의 relocated empty/TTY 검증도 통과했습니다. 이는 아래 수정 전 체크포인트의 증거이며 수정 후 재검증을 대신하지 않습니다.

첫 격리 실모델 driver는 고정한 공개 SmolLM checkpoint의 실제 다운로드·hash 검증, production CLI 재시작, 새 server identity와 이전 operation 404, 동일한 complete/unloaded catalog identity, 실제 load 및 비어 있지 않은 chat 생성을 확인했습니다. 이후 unload가 `stale_revision` expected3/current4로 실패하여 삭제는 시도하지 않았습니다. 소유한 서버 두 프로세스는 모두 exit0으로 종료했고 새 임시 checkpoint는 보존했습니다. 이 실패를 성공으로 바꾸어 기록하지 않습니다.

결함은 이미 merged main에 있었습니다. unload가 caller revision을 검사한 뒤 자신의 Ready-to-Draining 전환으로 revision을 증가시키고, 대기 후에도 이전 token을 비교하여 스스로 거부했습니다. 일반적인 harness 재시도는 이미 draining인 모델을 재시도하면서 결함을 가릴 수 있습니다. 수정은 lifecycle mutex 안에서 자신의 drain revision을 원자적으로 반환하고 대기 후 그 token과 현재 entry를 검사합니다. 공유 revision authority 때문에 +1을 가정하지 않으며 RequestLease 종료도 revision을 변경하지 않습니다. Revision precondition이 있는 WebUI·명시적 eviction caller만 새 token을 사용하고, legacy·shutdown은 failed-worker 정리를 포함한 기존 identity-only 검사를 유지합니다. 실제 harness는 변경하지 않았습니다.

로드된 fake provider 회귀 테스트는 수정 전에 실제 실행과 동일한 expected3/current4 오류로 실패했습니다. 최종 호환성 보완 후 unload/token 테스트 5개, lifecycle 테스트 16개, router fake 테스트 50개를 통과했습니다. 활성 request lease, loaded entry를 유지하는 rescan, 실제 worker 종료 관찰, 외부 revision·registry 교체 거부, 공유 revision authority 및 legacy failed-worker 정리를 검증합니다. 독립 correctness/security 재검토에서 이 제한된 수정의 HIGH/CRITICAL 잔여 사항은 없습니다.

None caller의 기존 동작을 보존한 최종 수정 후 scoped library/test clippy, 포맷 및 diff 검사를 통과했습니다. 수정본 전체 workspace·실모델 검증은 루트 담당으로 남아 있습니다.

## 9. UI rebase와 미해결 전체 게이트 경계 (2026-09-14)

머지된 UI PR #1863/main `b8d10fb1` 위로 rebase하면서 정확한 `@lablup/ui-common` 버전 `0.1.0-alpha.19`, 공유 보안·시작 동작 및 contract fixture 46개를 유지했습니다. 충돌은 생성된 bundle manifest 하나뿐이었으며 asset 수동 병합 없이 canonical builder로 해결했습니다. Range 비교에서 library 커밋 10개는 생성 manifest의 source digest 외에 동일하고, library/router/lifecycle production 파일은 `857937ca`와 바이트 단위로 동일합니다. Package lock, component, style 및 생성 JavaScript/CSS도 merged main을 유지합니다.

루트의 최신 `857937ca` 전체 게이트는 core의 configured-depth DFlash 테스트가 확인된 Metal firmware progress-timeout recovery 중 abort되어 실패·미완료 상태입니다. 진단은 외부 프로세스나 WebUI 수정을 원인으로 확정하지 않습니다. `3cb4817d`의 과거 전체 통과는 최신 실패를 대신하지 않으며 GB10 장애 면제도 이 실패에 적용되지 않습니다. 이후 루트의 `857937ca` CPU-only workspace all-target clippy, fixture 46개, 구조·포맷 검사와 feature-disabled 컴파일은 통과했으나 수정본 실제 library driver는 재실행하지 않았습니다.

사용자가 quiet window를 알려줄 때까지 GPU·browser·전체 suite 실행은 중단합니다. 이번 rebase에서는 CPU/fake 테스트와 frontend unit·contract·build 검사만 사용합니다. 실제 Safari·VoiceOver·native 200% 수동 검사는 사용자가 전체 구현 완료 시점으로 미뤘으므로 pending이며 통과나 개별 unit blocker로 보고하지 않습니다.

CPU-only rebase 검증: unload/token5, 순수 policy1, 보안 fake-router9, owned restart2(별도 nested child 실행2개), scoped clippy·포맷, strict fixture46개, frontend unit76개, type·lint 및 canonical deterministic bundle 검증을 통과했습니다. 이 unit rebase에서는 GPU·browser·전체 suite를 실행하지 않았습니다.
