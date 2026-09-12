# 이슈 #1839 — 라우터 라이프사이클 코디네이터와 worker-exit 게이트

## 요약

이슈 #1839는 WebUI 에픽에 필요한 백엔드 라이프사이클 조정 계층을 추가하되, 두 번째 모델 레지스트리를 만들지 않도록 구현했다. 라우터 엔트리는 이제 안정적인 WebUI 모델 ID, revision 기반 라이프사이클 스냅샷, 공유 operation/event 코디네이터를 가지며, 향후 `/ui-api/v1` 라우트와 기존 b10621 라우터 라우트가 같은 상태 전이를 사용한다.

## 구현

라우터 풀은 `LifecycleCoordinator`를 소유하고 각 `RouterModelEntry`는 `ModelLifecycle`을 소유한다. 라이프사이클은 추론 상태, 다운로드 상태, 활성 응답 바디 lease, revision/generation, admission stop, worker-exit 관측 여부를 추적한다. load, unload, autoload dispatch, 캐시 다운로드 종료, removal, UI action 어댑터는 모두 동일한 엔트리별 operation guard를 통해 동기화되므로 WebUI 전용 병렬 경로가 생기지 않는다.

모델 ID는 WebUI 계약의 canonical identity vector로 생성된다. 원본 source namespace는 서버에만 남고, 클라이언트에는 SHA-256 source-key hash와 불투명한 `mdl_...` ID만 노출된다. 같은 설정 source와 entry는 rescan 이후에도 같은 안정 ID를 유지한다. 코디네이터는 bounded store로 동작한다. active operation은 상한을 두고, terminal operation은 200개/1시간, SSE event는 1024개/10분으로 제한하며, idempotency key도 보존 중인 operation과 함께 pruning한다.

Unload는 새 admission을 먼저 중단하고, 활성 요청 lease가 모두 drain될 때까지 기다린 다음 provider shutdown을 전송하며, 실제 provider worker thread exit가 관측된 뒤에만 라우터 capacity를 해제한다. `ModelProvider`의 모든 worker 생성 경로와 fake/test 생성자는 `WorkerExitObserver`로 감싸져, 라우터가 단순 registry `Arc` drop이 아니라 worker-local model 파괴 이후의 thread 종료를 관측한다. Streaming 응답은 HTTP body가 완료되거나 drop될 때까지 lifecycle lease를 보유하므로 unload가 아직 살아 있는 스트림과 경합하지 않는다.

## 호환성 및 검증

기존 `/models`, `/models/load`, `/models/unload`, `/models/sse`, router dispatch 동작은 b10621 호환성을 유지한다. 일시적 라이프사이클 전이는 새 coordinator로 발행하고, legacy status stream에는 오해를 부를 수 있는 중간 상태 이벤트를 주입하지 않는다. 테스트는 contract identity vector, busy와 capacity 축 분리, request lease drain, operation idempotency, bounded idempotency pruning, SSE replay gap/restart 분류, router pool 회귀, router server 회귀, model provider worker wrapping을 검증한다.

## 실제 체크포인트 검증

GPU 체크포인트 하네스는 `/tmp/epic-1834-run-5tpfu3lc/issue-1839-real-lifecycle-harness.sh`에 준비했다. 이 하네스는 모델 A를 load하고 긴 streaming request를 시작한 뒤, response body가 살아 있는 동안 A unload를 요청한다. Draining 중 새 admission 거부를 확인하고, stream을 drop한 뒤 worker-exit 관측 로그를 확인한 다음 `--models-max 1` 상태에서 모델 B를 load한다. 또한 load/unload/load 전후의 프로세스 RSS를 기록하지만, RSS는 정보성 지표이며 zero-memory 보장을 의미하지 않는다.
