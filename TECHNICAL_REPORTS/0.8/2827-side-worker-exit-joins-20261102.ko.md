# PR #2827: 사이드 모델 워커 조인과 남은 종료 경로 구멍 정리

**Date**: 2026-11-02
**Status**: Completed (PR open; closes #2745)

## 요약

PR #2827은 #2742가 #2738 이후 남긴 teardown 구멍 네 개를 닫는다. 이제 오디오, realtime, embedding, image, rerank 사이드 워커가 채팅 워커와 같은 `shutdown_and_wait` 계약을 가지며, `serve_with_model_provider`가 하나의 공유 데드라인 아래 블로킹 풀에서 사이드 워커를 먼저 조인한 뒤 채팅 워커를 조인한다. sleeping 워커는 `Shutdown`을 받으면 모델을 다시 로드하지 않고 종료하고, `--pp-peer` 스테이지 경로는 시그널 핸들러를 설치하지 못해도 스레드를 항상 드레인한다. 23개 파일, +836/-75.

## 1. 문제 정의

CPU 빌드에서 MLX `CommonAllocator`는 함수 지역 static이라, `main`이 반환된 뒤 MLX 배열을 해제하는 스레드가 있으면 `pure virtual method called`로 프로세스가 중단된다. #2742는 채팅 워커를 조인했지만 네 경로가 여전히 워커를 살린 채 `main`에서 반환할 수 있었다.

- 사이드 워커는 `Drop`에서만 조인했고, 마지막 `Arc`를 드롭하는 async 스레드에서 실행되며 `main` 반환 전에 끝난다는 보장이 없었다.
- sleeping 워커는 `ModelRequest::Shutdown`을 깨우기 요청으로 취급해 모델 전체를 다시 로드한 뒤에야 종료를 봤다.
- 모델 로드에는 취소 지점이 없어, `WORKER_EXIT_TIMEOUT`보다 긴 로드는 조인을 조용히 빠져나갔다.
- `--pp-peer` 스테이지는 `shutdown` 전에 시그널 핸들러 오류를 반환했고, 블로킹 조인을 tokio 워커 스레드에서 직접 실행했다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/server/model_provider.rs` | 타입 있는 `WorkerShutdownError`(`Timeout` 또는 `Panicked`), 공용 `observe_worker_exit`, 백오프가 있는 `enqueue_shutdown_command`, `wait_for_worker_exit` |
| `src/server/startup.rs` | `SideWorkerExit` 수집기와 `join_model_workers_before_exit`, 로드 시점에 사이드 워커 등록, `StageServiceHandle` 트레이트를 쓰는 `wait_for_stage_shutdown` |
| `src/server/model_worker.rs` | sleeping 분기에서 `Shutdown`을 먼저 매칭하고 `sleeping`을 건드리지 않은 채 break |
| `audio_worker`, `embedding_worker`, `image_worker`, `rerank_worker`, `realtime_engine` | `WorkerExitObserver` 기반 `shutdown_and_wait`, `Drop`의 조인은 유지 |
| `startup_tests.rs`, `tests/server_exit_joins_worker.rs` | 단위 및 통합 테스트 추가, `ModelProvider::slow_to_start_worker_for_tests` |
| `docs/server-features.md` | `--sleep-idle-seconds` 옆에 취소 불가 제약을 기록 |

## 3. 기술적 선택

### 3.1 사이드 워커 먼저, 데드라인은 하나

`join_model_workers_before_exit`는 `spawn_blocking` 클로저 하나에서 사이드 워커 종료를 순서대로 실행하고, 각 워커에는 `WORKER_EXIT_TIMEOUT`의 남은 시간만 준다. 그다음 남은 예산을 `join_model_worker_before_exit`에 넘긴다. 데드라인 하나가 teardown 전체를 덮고, 모든 블로킹 조인이 tokio 워커 스레드 밖에 머문다.

### 3.2 사이드 워커 실패가 서빙 오류를 가리지 않는다

사이드 모델은 선택 사항이므로 종료 실패를 `warn!`로 남기고 버린다. 서버가 멈춘 진짜 이유(bind 오류, 로드할 수 없는 `--image-model`) 대신 사이드 종료 실패를 보고하면 중요한 실패가 가려진다. 오류 우선순위는 채팅 워커 조인이 소유하고, `served` 오류를 그대로 반환한다.

### 3.3 타입 있는 종료 오류

`shutdown_and_wait`는 이전에 타입 없는 `anyhow` 오류를 반환했다. 종료 경로는 워커가 아직 실행 중인 경우와 패닉한 경우를 구분해야 한다. CPU 빌드 중단 경고는 타임아웃 경우에만 붙기 때문이다. `WorkerShutdownError::{Timeout, Panicked}`가 그 구분을 명시하고, 기존 메시지 텍스트는 그대로 유지해 출력만 하는 호출자를 깨지 않는다.

### 3.4 sleeping은 상태이지 깨우기 요청이 아니다

sleeping 분기는 이제 일반 요청 분기보다 먼저 `Shutdown`을 매칭하고, `Shutdown requested while sleeping, worker exiting`을 남긴 뒤 break한다. `sleeping`을 일부러 지우지 않는다. 프로세스가 종료 중이고 다시 로드할 것이 없으므로 관측 상태를 종료 시점까지 사실대로 둔다. 데드라인보다 느린 재로드는 #2738 경쟁을 다시 열었을 것이다.

### 3.5 로드에는 취소 지점이 없다

이슈는 취소 지점 대신 문서화를 택했고 이 PR도 그 결정을 따른다. 로드는 패밀리별 `backend.load_model*` 구현을 지나며 공유 루프가 없고, MLX가 지연 평가하므로 샤드 사이 플래그로도 시간을 제한할 수 없다. 이제 타임아웃 메시지가 워커가 아직 실행 중이며 CPU 빌드에서 종료 시 중단될 수 있음을 밝힌다.

### 3.6 스테이지 헬퍼가 시그널 future를 받는다

`wait_for_stage_shutdown`이 select와 shutdown을 이 순서로 소유하므로 시그널 핸들러가 실패해도 스테이지를 드레인한다. 핸들을 작은 `StageServiceHandle` 트레이트 뒤에 두어, 종료 호출을 기록하는 가짜 구현으로 실패 분기를 단위 테스트할 수 있다. item 4의 라우터 쪽은 #2742에서 이미 고쳐져 이 PR에서 코드 변경이 없었다.

## 4. 검증

| 검사 | 결과 |
|---|---|
| `cargo fmt --all --check` | 통과 |
| `cargo check -p mlxcel --lib --tests` | 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | 통과 |
| `cargo test -p mlxcel --lib server::startup::` | 99 passed |
| `cargo test -p mlxcel --lib server::model_provider::tests::shutdown_and_wait` | 4 passed |
| `cargo test -p mlxcel --test server_exit_joins_worker` | 5 passed, 0 failed |

비모델 테스트 두 개가 타임아웃 메시지와 스테이지 드레인 분기를 덮는다. 실제 모델 사례(Whisper, embeddinggemma, qwen3-0.6b sleeping)는 이 호스트에 해당 체크포인트가 없고 `MLXCEL_REQUIRE_MODELS=1`이면 패닉하므로 passing skip으로 보고됐다.

## 5. 미검증

`--features cuda` 실행과 #2738의 2호스트 논리 PP CI 작업은 이 환경에 CUDA 호스트가 없어 실행하지 않았다. 둘은 PR의 GPU 및 분산 게이트로 남는다. 실제 모델 종료 테스트 네 건도 이곳 체크포인트로는 실행되지 않았다.

## 6. 학습 포인트

종료 경쟁은 단일 버그가 아니라 하나의 형태다. #2742가 한 사례를 고쳤지만, 조인되지 않은 teardown이라는 같은 형태가 사이드 워커 다섯, sleeping 경로, 제한된 로드 경로, 스테이지 시그널 경로에 남아 있었다. 형태를 고치려면 다섯 개의 국소 패치가 아니라 공용 옵저버와 수집기 하나가 필요했다.

## 7. 위험

중간. 모든 사이드 모델의 teardown 순서가 바뀌었고, 앞선 워커가 느리면 공유 데드라인이 뒤 워커의 예산을 줄일 수 있다. 사이드 워커의 예산 축소는 로그로 남고 서빙 오류를 대체하지 않지만, 데드라인이 워커별이 아니라 공유라는 점은 이후 독자가 알아야 한다.
