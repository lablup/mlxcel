# 기술 리포트: PR #1872 — 안전한 WebUI 모델 라이브러리 작업

**작성일**: 2026-09-13
**상태**: 열린 PR; 집중 CPU·fake 검증 완료, 루트 실모델·전체 workspace 게이트 대기
**언어**: Rust, JSON/OpenAPI fixture
**위험도**: 높음
**구현 스냅샷**: 리포트 전용 커밋 전 `466c6071`

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

완전성 정책은 `config.json`, 비어 있지 않은 선택 safetensors, 알려진 크기와 기록 byte의 일치를 요구합니다. 이 구현은 현재 downloader가 제공하는 metadata와 전송 증거를 넘어 독립적인 LFS checksum 검증을 수행한다고 주장하지 않습니다.

### 관리 캐시 삭제만 허용

`POST /ui-api/v1/model-removals`는 고정된 request shape인 `model_id`, `expected_revision`, `idempotency_key`를 유지합니다. 사용자에게 보이는 확인 modal은 WebUI page 작업 범위이고, backend endpoint 자체가 명시적인 삭제 의도입니다. Pool은 cache-sourced이고 현재 revision이 맞으며 idle 상태인 model entry만 허용하고, 삭제를 spawn하기 전에 operation-busy로 표시하여 removal이 catalog slot을 소유하는 동안 rescan이 stale entry를 재생성하지 못하게 합니다.

삭제는 설정된 cache root, owner, target, private quarantine을 `O_NOFOLLOW`로 열고, quarantine rename 직전과 rename 후 identity를 확인한 다음 held fd 기준으로 재귀 unlink를 수행합니다. 중첩 디렉터리 삭제는 예상 dev/ino를 open과 최종 rmdir 검사까지 전달합니다. Identity가 바뀌면 삭제는 fail-closed하고 quarantined snapshot을 조사용으로 남깁니다. 같은 UID 프로세스는 여전히 denial of service나 quarantine cleanup 실패 race를 유발할 수 있으므로, 이것은 hostile same-user를 절대적으로 격리한다는 주장이 아닙니다.

## 3. 리뷰 보완 사항

Parent pre-review에서 지적된 다음 고위험 항목을 PR 게시 전 반영했습니다.

- 마지막 전송 chunk 이후 publish 전 취소 race를 `begin_publish_operation`으로 linearize했고, 늦은 취소가 거절되며 operation이 성공하는 결정적 회귀 테스트를 추가했습니다.
- 알려진 HEAD 크기를 fd-relative rename 전 실제 byte 수와 비교하고, metadata와 선택 파일 수·파일명 길이를 제한한 뒤 다운로드를 진행합니다.
- 익명 WebUI metadata는 더 이상 저장된 token file을 읽는 `hf-hub` API builder를 생성하지 않으며, 테스트는 token file과 env var를 심고 metadata·HEAD·GET fake 요청에 Authorization header가 없음을 확인합니다.
- 제거 작업은 `operation_busy`로 lifecycle을 예약하여 removal worker가 끝날 때까지 rescan이 원래 entry를 보존하게 했습니다.
- Parent swap 및 중첩 stat/open 삭제 race는 결정적 테스트를 갖고 잘못된 tree를 삭제하지 않고 실패합니다.
- `anchored_delete.rs`와 인접 테스트 모듈로 분리하여 새 handwritten module을 500줄 이하로 유지했습니다.

## 4. 검증 기록

| 게이트 | 리포트 작성 시점 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate anchored_remove -- --nocapture` | 통과: 관리 캐시 quarantine 삭제 및 parent/final-unlink swap 회귀 |
| `cargo test --profile test-fast --features metal,accelerate anchored_publish -- --nocapture` | 통과: no-replace publish, owner/staging identity, symlink cleanup 회귀 |
| `cargo test --profile test-fast --features metal,accelerate anchored_delete -- --nocapture` | 통과: 중첩 디렉터리 stat/open swap 회귀 |
| 집중 downloader/router/WebUI 필터 | 통과: 익명 전송, 저장 token 음성, fd staging, 알려진 크기 불일치, manifest bound, lifecycle fixture, WebUI library route, duplicate replay, queue saturation, publish cancel linearization, case-alias reject, 없는 store no-mkdir, 관리 removal, rescan guard |
| `cargo fmt --check` 및 `git diff --check` | 통과 |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | 통과 |
| `PATH=/tmp/mlxcel-webui-contract/bin:$PATH make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` | 통과: WebUI contract fixture 43개, DTO drift, schema strictness |
| `PATH=/tmp/mlxcel-webui-contract/bin:$PATH make verify-llama-compat verify-versions verify-kernel-dtype-keys` | 통과 |
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

PR 생성 시 통계는 24개 파일 변경, 3,576줄 추가, 231줄 삭제입니다. 구현 커밋은 `466c6071 feat: add safe WebUI model library operations`입니다.

## 6. 학습 포인트와 후속 조치

WebUI library action은 기존 helper 위의 버튼이 아니라 권한 경계입니다. 접수는 싸고 제한되어야 하며, network와 disk 작업은 pool lock 밖에서 수행되어야 하고, 마지막 write/delete 단계는 지금 변경하려는 filesystem identity를 다시 확인해야 합니다.

남은 작업은 이 리포트 안의 숨은 구현이 아니라 acceptance입니다. 루트는 pinned small SmolLM 실제 download/load/generate/delete driver, CI-faithful 전체 workspace gate, workspace all-target clippy, 사용 가능한 CUDA/GB10 검사 또는 runner outage waiver 기록을 수행해야 합니다. Page 계층은 이 backend wire contract를 바꾸지 않으면서 눈에 보이는 삭제 확인을 제공하고 unload와 disk deletion을 명확히 구분해야 합니다.
