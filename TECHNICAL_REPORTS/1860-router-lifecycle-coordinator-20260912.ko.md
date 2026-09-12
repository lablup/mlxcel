# PR #1860 / 이슈 #1839 — 라우터 라이프사이클 코디네이터와 워커 종료 게이트

## 요약

이슈 #1839는 두 번째 모델 레지스트리를 만들지 않고 WebUI 에픽에 필요한 백엔드 라이프사이클 조정 계층을 추가했다. 라우터 엔트리는 안정적인 WebUI 모델 ID, revision 기반 라이프사이클 스냅샷, 활성 요청 lease, 그리고 기존 라우터 전이와 새 WebUI model-action/operation/event 어댑터가 함께 쓰는 operation/event 코디네이터를 갖는다.

## 구현

라우터 풀은 `LifecycleCoordinator`를 소유하고 각 `RouterModelEntry`는 `ModelLifecycle`을 소유한다. 라이프사이클은 추론 준비 상태, 다운로드 상태, busy 축과 capacity reservation 축, revision/generation, admission stop, 응답 바디 lease, 관찰된 worker exit를 추적한다. UI model action은 이 코디네이터에 idempotent operation을 제출하며, 중복 키 replay를 revision 검사보다 먼저 처리하고, stale revision은 typed field error로 거절하며, `202 Accepted`는 ready가 아니라 queued임을 보장한다.

모델 ID는 WebUI 계약의 canonical identity vector에서 생성된다. 원본 source namespace는 서버 내부에만 남고, 클라이언트에는 SHA-256 source-key hash와 동일 configured source/entry에 대해 안정적인 opaque `mdl_...` ID만 노출된다. 코디네이터는 active operation cap, terminal operation 200개/1시간, SSE event 1024개/10분, retained operation과 함께 pruning되는 idempotency key를 관리한다. Event sequence 할당, ring 삽입, broadcast는 하나의 mutex 순서 권한으로 묶였고, `/ui-api/v1/events`는 snapshot fence, reconnect replay, gap/server restart typed reset event를 제공한다.

Unload는 새 admission을 멈추고, 활성 request lease drain을 기다리고, provider shutdown 명령을 보낸 뒤 실제 provider worker thread exit가 관찰된 후에만 라우터 capacity를 해제한다. `ModelProvider`는 fake/test constructor를 포함한 모든 worker constructor를 `WorkerExitObserver`로 감싸므로 registry `Arc` drop에 의존하지 않고 모델 파괴를 관찰한다. Streaming response는 HTTP body가 끝나거나 drop될 때까지 lifecycle lease를 유지하므로 unload가 아직 살아 있는 stream과 경합하지 않는다. 라우터 shutdown은 reserved/downloading entry를 bounded report window 안에서 drain하고, 남은 리소스가 있으면 in-process GPU worker를 강제 종료하지 않고 보고한다.

## 호환성 및 검증

기존 `/models`, `/models/load`, `/models/unload`, `/models/sse`, 라우터 dispatch 동작은 b10621 호환 형태를 유지한다. 기존 cache removal은 여전히 라우터 deletion 경로이며, reserved entry를 삭제 전에 멈춰야 할 때 lifecycle 보호를 받는다. WebUI 전용 removal operation은 이 이슈의 claim이 아니며 model-action/operation/event adapter와 별도 범위로 남아 있다.

이슈 worktree에서 검증한 명령은 `cargo check --profile test-fast --features metal,accelerate --bin mlxcel-server`, `cargo test --profile test-fast --features metal,accelerate router_ -- --nocapture`, `python scripts/ci/check_webui_contract.py` (using the local validator environment)이다. 집중 테스트는 contract identity vector, DTO fixture round-trip, busy/capacity 축 분리, event broadcast 순서, ring gap replay, operations list/get/cancel, revision 변경 후 idempotency replay, UI load terminal failure, explicit eviction refusal, rescan/delete reservation barrier, response-body lease drop, bounded shutdown report, WebUI HTTP action/operation/SSE adapter를 포함한다.

## 실제 체크포인트 수용 테스트

Root가 조율하는 GPU 체크포인트 harness가 maintainer 실행용으로 준비되어 있다. Harness는 모델 A를 로드하고 긴 streaming request를 시작한 뒤 response body가 살아 있는 동안 A unload를 요청하고, draining 중 새 admission이 거절되는지 확인하고, stream을 drop하고, worker-exit observation 로그를 확인한 뒤 `--models-max 1` 아래에서 모델 B를 로드한다. 또한 load/unload/load 전후 scoped process RSS 스냅샷을 기록하지만, RSS는 정보성 측정치이며 0 메모리 보장을 의미하지 않는다.
