# PR #1860 / 이슈 #1839 — 라우터 라이프사이클 코디네이터와 워커 종료 게이트

## 요약

이슈 #1839는 두 번째 모델 레지스트리를 만들지 않고 WebUI 에픽에 필요한 백엔드 라이프사이클 조정 계층을 추가했다. 라우터 엔트리는 안정적인 WebUI 모델 ID, 단조 증가 라이프사이클 revision, 활성 요청 lease, 그리고 기존 라우터 전이와 새 WebUI model-action/operation/event 어댑터가 함께 쓰는 operation/event 코디네이터를 갖는다.

## 구현

라우터 풀은 `LifecycleCoordinator`를 소유하고 각 `RouterModelEntry`는 `ModelLifecycle`을 소유한다. 라이프사이클은 추론 준비 상태, 다운로드 상태, busy 축과 capacity reservation 축, revision/generation, admission stop, 응답 바디 lease, 관찰된 worker exit를 추적한다. 풀 전체 revision authority가 재생성된 엔트리와 모든 라이프사이클 전이에 새 스탬프를 찍으므로 remove/rescan/re-add ABA가 같은 canonical model ID에 revision 1을 재사용할 수 없다.

모델 ID는 WebUI 계약의 canonical identity vector에서 생성된다. 원본 source namespace는 서버 내부에만 남고, 클라이언트에는 SHA-256 source-key hash와 동일 configured source/entry에 대해 안정적인 opaque `mdl_...` ID만 노출된다. 코디네이터는 active operation cap, terminal operation 200개/1시간, SSE event 1024개/10분, retained operation과 함께 pruning되는 idempotency key를 관리한다. Event sequence 할당, ring 삽입, broadcast는 하나의 mutex 순서 권한으로 묶인다.

UI model action은 이 코디네이터에 idempotent operation을 제출하며, 중복 키 replay를 revision 검사보다 먼저 처리하고, stale revision은 typed field error로 거절하며, `202 Accepted`는 ready가 아니라 queued임을 보장한다. Queued action은 요청된 primary entry `Arc`, stable model ID, revision을 보존하고, explicit eviction target도 별도의 entry `Arc`, stable ID, revision을 보존한다. 백그라운드 executor는 상태를 바꾸기 전에 load/unload transition lock 아래에서 이 identity들을 재검증하므로 표시 이름이나 canonical model ID가 재사용되어도 rescan/remove/re-add 교체를 거절한다.

Unload는 새 admission을 멈추고, 활성 request lease drain을 기다리고, provider shutdown 명령을 보낸 뒤 실제 provider worker thread exit가 관찰된 후에만 라우터 capacity를 해제한다. `ModelProvider`는 fake/test constructor를 포함한 모든 worker constructor를 `WorkerExitObserver`로 감싸므로 registry `Arc` drop에 의존하지 않고 모델 파괴를 관찰한다. Streaming response는 HTTP body가 끝나거나 drop될 때까지 lifecycle lease를 유지하므로 unload가 아직 살아 있는 stream과 경합하지 않는다. 라우터 shutdown은 reserved/downloading entry를 bounded report window 안에서 drain하고, 남은 리소스가 있으면 in-process GPU worker를 강제 종료하지 않고 보고한다.

## 이번 리뷰 수정

Post-review 수정은 `begin_load`, rescan, remove, queued UI action 주변의 high-risk race window를 닫는다. `begin_load`는 registry를 읽기 전에 load lock을 잡고, entry transition guard 아래에서 캡처한 entry를 재검증하며, `loading`으로 표시하기 직전에 다시 확인한다. Rescan은 오래된 snapshot clone 이후 reserved가 된 현재 entry를 보존한다. Remove는 cancel 전, download stop 후, unload 전, registry 삭제 중에 current `Arc`를 재검증한다. 취소된 다운로드의 terminal rescan이 해당 entry를 먼저 제거한 경우에는 replacement entry가 나타나지 않았을 때만 removal을 계속한다. LRU eviction은 serving/draining entry를 거절하고, explicit eviction은 `not_needed`, `displaced`, `failed_after_displacement` typed outcome을 남긴다.

WebUI administrative adapter는 더 이상 production/base router에 mount되지 않는다. `create_router_app`은 기존 라우터만 유지하고, 테스트와 향후 startup integration은 `create_router_app_with_authenticated_ui`를 사용한다. 이 accessor는 API key가 설정되고 제시되지 않으면 모든 `/ui-api/*` 요청을 거절한다. 느린 SSE client는 더 이상 전역 ring에 reset을 publish하지 않고 client-local reset event를 받는다. Route와 operation history의 오류는 bounded typed message로 redaction되고, 원본 build/load 세부 정보는 서버 로그에만 남는다.

WebUI 계약에는 `target.eviction_target_id`와 typed `ModelEvictionReport`가 추가되었고, TypeScript DTO와 fixture가 갱신되었다. Route-level producer test는 실제 `/ui-api/v1/operations` 직렬화 응답을 Rust DTO로 round-trip하고, schema-critical token/model-id/timestamp/nullability field를 검사하며, operation/result discriminator와 eviction outcome을 schema-validated fixture와 대조한다.

## 호환성 및 검증

기존 `/models`, `/models/load`, `/models/unload`, `/models/sse`, 라우터 dispatch 동작은 b10621 호환 형태를 유지한다. 기존 cache removal은 여전히 라우터 deletion 경로이며, reserved entry를 삭제 전에 멈춰야 할 때 lifecycle 보호를 받는다. Production router는 보안 startup/auth 이슈가 의도적으로 mount하기 전까지 WebUI administrative route를 노출하지 않는다.

이슈 worktree에서 검증한 명령은 다음과 같다.

- `cargo check --profile test-fast --features metal,accelerate --bin mlxcel-server`
- `cargo fmt --check`
- `git diff --check`
- `python3 scripts/insert_apache_header.py --check`
- `/tmp/mlxcel-webui-contract/bin/python scripts/ci/check_webui_contract.py`
- `cargo test --profile test-fast --features metal,accelerate router_server_tests:: -- --nocapture` (23 passed)
- `cargo test --profile test-fast --features metal,accelerate router_lifecycle_tests:: -- --nocapture` (9 passed)
- `cargo test --profile test-fast --features metal,accelerate router_models_tests:: -- --nocapture` (32 passed)

## 실제 체크포인트 수용 테스트

Root가 조율한 첫 실제 체크포인트 게이트는 post-implementation build(`584249a4`)에서 통과했다. 모델 A `meta-llama-3.1-8b-instruct-4bit`는 572자 streaming 응답을 냈고, response body가 살아 있는 동안 unload를 요청하면 기대한 drain refusal HTTP 400이 발생했으며, 모델 A의 worker exit가 관찰되었고, `--models-max 1` 아래에서 모델 B `granite-4.0-h-tiny-4bit`가 비어 있지 않은 텍스트(`Affirmative.`)를 생성했다. Scoped RSS 스냅샷은 start 32656 KiB, after load A 351824 KiB, after unload A 4173872 KiB, after load B 4257760 KiB였다. RSS는 정보성 측정치이며 0 메모리 보장을 의미하지 않는다.

Root는 오래된 binary에 대한 negative control도 수행했다(`SHA256 ef3146d4a2722cce81b683bd48e67996fc9b9c4512c66ae4d51549e0844c6a78`, source commit unknown). 같은 real harness는 unload-before-stream-drop 단계에서 exit 4로 실패했다. 증거 경로는 `/tmp/epic-1834-run-5tpfu3lc/issue-1839-real-20260912-143914/summary.json` 및 `/tmp/epic-1834-run-5tpfu3lc/negative-control-ef3146d4.log`이다. 리뷰 수정 commit 이후의 최종 real rerun은 root GPU 조율을 기다린다.
