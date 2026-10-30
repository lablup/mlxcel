# PR #2742: 프로세스 종료 전에 model worker를 join

**날짜**: 2026-10-30 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust
**위험도**: 낮음 (종료 경로만 바뀜. `mlxcel-server`가 오류로 종료할 때 프로세스가 끝나기 전에 model worker를 최대 120초, router 모드에서는 150초까지 기다릴 수 있음)

## 요약

hosted `Pipeline Parallel CI / 2-host logical` job은 테스트를 통과한 뒤 테스트 바이너리가 종료되는 도중 `pure virtual method called`로 abort했다(issue #2738). 원인은 아무도 join하지 않은 `ModelProvider` worker 스레드다. `main`이 반환될 때 이 스레드는 아직 모델의 MLX array를 해제하고 있었고, CPU backend에서 MLX allocator는 이미 소멸자가 실행된 함수 지역 C++ static이었다. 이 PR은 join primitive 하나(`ModelProvider::shutdown_and_wait`)를 추가해 두 real-model pipeline 테스트와 `InProcessServer::shutdown`에서 사용하고, `mlxcel-server`(single-model 모드와 router 모드)가 모델 로드를 시작한 뒤의 모든 종료 경로가 worker join을 거치도록 바꾼다. 로컬 재현은 50회 중 44회 abort에서 50회 중 0회로 바뀌었고, CI workflow는 원래 abort가 모두 발생했던 runner image에서 연속 10회 dispatch 실행을 통과했다.

## 1. 문제 정의

### CI 증상

2026-10-28부터 `Pipeline Parallel CI / 2-host logical (ubuntu-latest)` job이 간헐적으로 실패했다. `pipeline_multi_stage_two_host_logical_smoke`는 `ok`를 보고하고 libtest가 요약을 출력한 뒤, 약 40ms 후 `pure virtual method called` / `terminate called without an active exception`과 함께 SIGABRT로 죽었다. abort는 테스트 안이 아니라 프로세스 종료 중에 일어났다. pipeline-parallel 코드를 건드리는 모든 PR의 merge가 막혔다.

실패는 commit이 아니라 GitHub runner image와 상관관계를 보였다. abort 10건은 모두 `ubuntu24/20261025.353`에서, 성공 19건은 모두 `ubuntu24/20261012.334`에서 실행됐고, 성공한 실행과 abort한 실행이 같은 base commit을 공유했다. 의미 있는 image 차이는 kernel(6.17에서 7.0)이었고 rustc, OpenBLAS, LAPACK은 같았다. 그래서 issue는 이를 새 kernel의 scheduling이 드러낸 기존의 종료 시점 race로 보고, 수정 전에 메커니즘을 반드시 확인하도록 요구했다.

### 확인된 메커니즘

같은 kernel(7.0.0)을 쓰는 머신에서 CI 구성(debug profile, `cuda` feature 없음, `MLXCEL_CI_PP_MODEL=qwen3-0.6b-4bit`)으로 재현했다. `__cxa_pure_virtual`에 breakpoint를 건 gdb가 3번 시도 중 3번 모두 abort를 잡았다. abort한 스레드는 model worker(`src/server/model_worker.rs`의 `spawn_model_worker_with_batch_config`)였고, scheduler의 `PagedBlockPool`을 drop하는 중이었다. 이 drop은 `MlxArray::~MlxArray`를 거쳐 `allocator().free(buffer)`를 호출하는 `mlx::core::allocator::Data::~Data`에 도달했다. 같은 순간 main 스레드는 `exit` → `__run_exit_handlers` → `_dl_fini` 안에 있었다(stack에 OpenBLAS finalizer가 있었다).

세 가지 사실이 겹쳐 abort가 된다.

1. `impl Drop for ModelProvider`는 `ModelRequest::Shutdown`을 보내고 바로 반환한다. worker의 `JoinHandle`은 `observe_worker_exit`가 만든 watcher 스레드가 소유하며, gdb에서 이 스레드는 worker에 대해 `pthread_join`으로 대기 중이었다. main 스레드에서는 둘 중 어느 것도 기다리지 않았다.
2. Rust는 `main`이 반환될 때 spawn된 스레드를 join하지 않는다. C runtime은 `exit()`를 실행하고, 다른 스레드가 계속 도는 동안 static 소멸자를 실행한다.
3. CPU backend에서 `allocator()`는 `mlx/backend/no_gpu/allocator.cpp`의 `static CommonAllocator allocator_`, 즉 함수 지역 static을 반환한다. 소멸자가 실행된 뒤에는 객체의 vtable이 추상 클래스 `Allocator`의 것이 되고 `free`는 pure virtual이 된다. Metal, CUDA, ROCm allocator는 의도적으로 heap에 leak되어 소멸되지 않으므로, CPU CI 빌드에서만 abort가 났다.

`tests/pipeline_ci_multi_stage_real_models.rs`의 테스트 helper `assert_multi_stage_coordinator_matches_dense_baseline`은 `drop(remote_provider); drop(dense_provider);`로 끝난 뒤 stage service 스레드만 join했다. 두 provider의 worker는 분리된 채로 남았다.

### production의 같은 패턴

`InProcessServer::shutdown`(#2224 이후 `mlxcel run`과 `chat`이 사용)과 router의 모델 unload는 이미 worker의 exit observer를 기다렸다. 프로세스 종료를 소유하는 다른 경로는 그렇지 않았으므로, issue는 경로별 판정을 요구했다.

| 경로 | 이 PR 이전 판정 | 이후 |
|---|---|---|
| `mlxcel-server` single-model (`start_server`) | 영향 있음. chat worker가 로드를 시작한 뒤 `serve_http` 오류(포트 점유, TLS 실패)나 side-model 해석 오류 다섯 가지 중 하나가 나면 worker가 살아 있는 채로 `main`에서 반환했다. | 수정: provider 생성 이후의 모든 반환이 `join_model_worker_before_exit`를 거친다. |
| Router 모드 (`run_router_server`) | 일부 영향. Ctrl-C는 `shutdown_all`로 drain했지만 `serve_http` 오류 분기는 drain 없이 반환했으므로, `load-on-startup` 이후 bind 실패가 나면 로드 중인 worker를 남기고 종료했다. | 수정: 두 분기 모두 drain한다. Ctrl-C handler 설치 실패 시에도 drain한다. |
| `--pp-peer` stage (`serve_remote_pipeline_stage`) | 영향 없음. `RemoteStageServiceHandle::shutdown`이 stage 모델을 소유한 stage 스레드를 join한다. | 변경 없음. |
| `mlxcel run` / `chat` | 영향 없음(#2224). | 변경 없음. `InProcessServer::shutdown`은 이제 공용 primitive에 위임한다. |

single-model 모드의 SIGINT와 SIGTERM은 여전히 기본 signal 처리로 종료된다. 이 경우 exit handler가 실행되지 않으므로 race가 생기지 않고, PR은 이 동작을 바꾸지 않았다.

## 2. 기술적 선택과 그 이유

### `ModelProvider`의 join primitive 하나

`ModelProvider::shutdown_and_wait(timeout)`은 shutdown signal을 보내고 기존 `WorkerExitObserver`를 기다린 뒤 다음을 반환한다.

- worker가 종료했으면 `Ok(())`. 호출 전에 이미 종료한 경우와 request channel이 닫힌 경우도 포함한다(worker가 이미 request loop를 벗어났으므로 send 실패는 오류가 아니다).
- 시간 초과면 `Err("server model worker did not exit within {timeout:?}")`.
- watcher가 panic을 기록했으면 `Err("server model worker panicked: ...")`.

멱등적이다. observer가 observed 상태로 남으므로 두 번째 호출은 즉시 `Ok`를 반환한다. 본문은 `InProcessServer::shutdown`에 있던 루프를 옮긴 것이다. issue가 두 번째 대기 루프 작성을 명시적으로 금지했으므로, 모든 소유자가 구현 하나를 쓰도록 했다. `WORKER_EXIT_TIMEOUT`(120초)은 `src/server/in_process/mod.rs`에서 `model_provider.rs`로 옮겨 `pub`이 되고 `mlxcel::server`에서 re-export된다. integration test는 crate 밖에 있으면서 같은 상한을 써야 하기 때문이다.

timeout은 `as_secs()`가 아니라 `Debug`로 포맷한다. 120초에서는 둘 다 `120s`를 출력하므로 production 메시지는 그대로이고, 1초 미만의 테스트 timeout은 오해를 부르는 `0s` 대신 `500ms`로 출력된다.

### `Drop`은 계속 non-blocking

`Drop for ModelProvider`가 worker를 join하게 만들면 호출자를 건드리지 않고 테스트를 고칠 수 있었다. issue가 이를 범위 밖으로 둔 데에는 이유가 있다. `ModelProvider`는 router entry 해제와 `AppState` 해제 과정에서 tokio runtime 스레드 위에서 drop된다. 그곳에서 blocking join을 하면 runtime worker 스레드가 timeout만큼 멈출 수 있다. PR은 대신 프로세스 종료를 소유하는 몇 곳에서 대기를 명시적으로 수행한다.

### 종료 우회 수단을 쓰지 않음

`std::process::exit`, `libc::_exit`, `quick_exit`, provider의 `mem::forget`, CI 재시도는 모두 abort를 사라지게 할 수 있었다. issue는 backtrace가 exit handler 자체의 결함을 증명하지 않는 한 이 모두를 배제했다. backtrace가 보여준 것은 그 반대다. handler는 정상 동작했고, 아무도 join하지 않은 스레드가 소멸된 static을 건드렸다. exit handler를 건너뛰면 다른 모든 라이브러리의 정상적인 flush도 건너뛰게 되고, 재시도는 production 바이너리에 있는 같은 race를 감출 뿐이다.

### 모든 `start_server` 반환이 구조적으로 join에 도달하도록

single-model 경로에는 provider 생성과 `serve_http` 사이에 `?` 반환이 많다(warmup, side-model 해석, `AppState` 생성). 각각의 앞에 join을 넣는 방식은 다음에 누군가 추가할 `?`에서 놓치기 쉽다. PR은 `ModelProvider` 생성 이후의 모든 코드를 `serve_with_model_provider(startup, ChatServeInputs { .. })`로 분리한다. `start_server`는 provider의 `Arc` clone을 보관하고, 내부 함수를 await한 뒤 그 `Result`를 `join_model_worker_before_exit`에 넘긴다. 내부 함수의 어떤 반환도, `?`를 포함해, join을 거치며 이를 compiler가 보장한다.

blocking `Drop`을 가진 scope guard는 위와 같은 이유로 선택지가 아니었다. async context 안에서 block하게 된다. `ChatServeInputs`는 분리된 함수의 signature를 읽기 쉽게 하기 위해서만 존재한다.

`join_model_worker_before_exit`의 동작:

- serve 오류가 있으면 먼저 error level로 기록한다(`server stopped: ...; waiting for the model worker to exit`). 긴 모델 로드 중 bind나 TLS 실패가 나도 로그에서 조용한 대기가 아니라 실패로 읽힌다.
- `shutdown_and_wait(WORKER_EXIT_TIMEOUT)`을 `spawn_blocking`에서 실행해 HTTP runtime worker를 점유하지 않는다.
- 성공하면 `model worker exited before process exit`를, 실패하면 warning을 기록한다.
- serve 오류가 있으면 그대로 반환한다. shutdown 실패는 serving 자체가 성공했을 때만 반환하므로, 서버가 멈춘 이유를 덮어쓰지 않는다.

### Router: `select!`의 두 분기 모두 drain

이전의 `run_router_server`는 Ctrl-C 분기 안에서만 pool을 drain했다. 이제 `select!`는 어느 분기에서든 `Result`를 만들고, 그 뒤에 `shutdown_pool.shutdown_all(ROUTER_SHUTDOWN_TIMEOUT)`이 조건 없이 실행되며 기존의 완료 및 pending-resource 로그를 남긴다. listener 오류는 drain 시작 전에 기록한다. `tokio::signal::ctrl_c()`가 handler 설치에 실패하면 그 오류를 기록하고, pool을 drain한 뒤 오류를 반환한다. 이전에는 `signal?`가 `shutdown_all` 전에 반환했다. `shutdown_all`은 이미 각 worker의 exit observer를 기다리고, entry operation guard를 통해 진행 중인 `build_model_app`도 기다리므로 새 대기 로직이 필요하지 않았다.

### 테스트의 종료 순서

두 real-model pipeline 테스트(`pipeline_ci_multi_stage_real_models.rs`, `pipeline_server_remote_real_models.rs`)는 이제 remote coordinator provider, 그다음 dense provider 순서로 `shutdown_and_wait`를 호출하고, stage service를 내리기 전에 둘 다 `expect`한다. coordinator를 먼저 내리는 이유는 stage가 아직 서비스 중일 때 마지막 stage 왕복을 끝내기 위해서다. 두 번째 파일은 현재 allocator가 leak되는 CUDA나 Metal에서만 빌드되지만, 같은 분리된 worker 패턴이 있어 CPU runner에서는 abort했을 것이다.

smoke test의 doc comment도 더 이상 구성을 "two processes"로 설명하지 않는다. stage, coordinator, dense baseline은 loopback TCP로 통신하는 하나의 테스트 프로세스 안의 스레드다.

## 3. 변경 요약

9개 파일, +545/-48. commit 두 개(수정은 e4312b89, join 전에 종료 사유를 기록하는 변경은 03c1e365)와 origin/main merge(881231bd).

- `src/server/model_provider.rs`: `ModelProvider::shutdown_and_wait`, `pub const WORKER_EXIT_TIMEOUT`.
- `src/server/mod.rs`: `WORKER_EXIT_TIMEOUT` re-export.
- `src/server/in_process/mod.rs`: `InProcessServer::shutdown`이 `shutdown_and_wait`에 위임하고, 지역 상수와 대기 루프를 제거.
- `src/server/startup.rs`: `start_server`를 `serve_with_model_provider`와 `ChatServeInputs`로 분리, `join_model_worker_before_exit` 추가.
- `src/server/router_server.rs`: `run_router_server`가 Ctrl-C뿐 아니라 listener 오류와 signal 오류 분기에서도 종료 사유를 먼저 기록하고 drain.
- 테스트:
  - `src/server/model_provider_tests.rs`: 직접 만든 worker 스레드 위의 `provider_with_worker` helper와 테스트 네 개. 협조적인 worker(`Ok`, observed, 두 번째 호출은 대기 없이 `Ok`), 이미 종료한 worker(`Duration::ZERO`로 `Ok`), teardown 중 panic하는 worker(`teardown boom`을 담은 `Err`), shutdown을 무시하는 worker(최소 1초 후 정확히 `did not exit within 1s` 메시지의 `Err`, 풀어준 뒤에는 `Ok`).
  - `tests/pipeline_ci_multi_stage_real_models.rs`, `tests/pipeline_server_remote_real_models.rs`: 위에서 설명한 명시적 join.
  - `tests/server_exit_joins_worker.rs`(신규): 테스트가 포트를 점유한 상태에서 실제 `mlxcel-server`를 실행해, 모델 로드가 시작된 뒤 bind가 실패하게 한다. single-model 테스트는 exit code 1, signal 없음, `pure virtual method called` / `terminate called` 텍스트 없음, `model worker exited before process exit` 로그를 확인한다. router 테스트는 `load-on-startup = 1` preset을 쓰고 `router: lifecycle shutdown completed attempted=1 completed=1`을 확인한다. 실패한 `begin_load`도 `attempted=0`으로 같은 메시지를 남기므로 개수까지 맞춘다. 두 테스트 모두 `qwen3-0.6b-4bit`가 없으면 skip한다.

## 4. 트레이드오프와 잔여 항목

- **오류 종료가 이제 기다릴 수 있다.** 큰 모델이 아직 로드 중일 때 bind나 TLS가 실패하면 예전에는 즉시 종료했다(CPU에서는 abort할 수 있었다). 이제는 worker를 최대 120초(single-model) 또는 150초(router) 기다린다. 오류는 대기 전에 기록되므로 운영자는 원인을 바로 본다.
- **timeout을 넘긴 worker는 여전히 handler 실행 중에 남는다.** worker가 `WORKER_EXIT_TIMEOUT`이나 `ROUTER_SHUTDOWN_TIMEOUT` 안에 종료하지 않으면 프로세스는 worker가 살아 있는 채로 `main`에서 반환하고, CPU backend에서는 원래 위험이 그대로 적용된다. 대안은 무기한 대기다. 현재 모델 로드는 취소할 수 없으므로 timeout보다 긴 로드는 이 경우에 해당하며, #2745에서 추적한다.
- **side-model worker는 join하지 않는다.** audio, embedding, image, rerank worker는 exit observer가 없는 trait object다. CPU backend에서 같은 종류의 위험이 적용된다. #2745가 이들을 sleeping worker의 shutdown 시 reload, `--pp-peer` signal 경로와 함께 추적한다.
- **수정은 모든 backend에 적용되지만 노출된 것은 CPU뿐이었다.** Metal, CUDA, ROCm은 allocator가 leak되므로 abort가 날 수 없었다. 그래도 join은 이들에도 상한이 있는 대기를 추가한다.

## 5. 검증

**메커니즘.** CI 구성의 debug CPU 바이너리에 `break __cxa_pure_virtual`을 건 gdb: 3번 중 3번 모두, main 스레드가 `__run_exit_handlers` 안에 있을 때 worker의 `PagedBlockPool` drop에서 멈췄다(1절 참고).

**로컬 전후 비교.** smoke 바이너리를 CI와 똑같이 50회 반복 실행했다(`MLXCEL_CI_PP_MODEL=qwen3-0.6b-4bit "$BIN" pipeline_multi_stage_two_host_logical_smoke --ignored --nocapture --test-threads=1`).

| 빌드 | abort |
|---|---|
| origin/main 8385f40b | 44/50 (모든 abort 로그에 `pure virtual method called`) |
| 이 브랜치 | 0/50 (같은 머신, 반복 실행 시간대 겹침) |

**테스트 suite.** `server::model_provider::`와 `server::in_process::` lib 테스트(`--profile test-fast --features cuda`, `--test-threads=1`), 실제 checkpoint와 `MLXCEL_REQUIRE_MODELS=1`로 `server_exit_joins_worker` 두 테스트, `cuda,test-utils`로 `pipeline_server_remote_real_models`의 `cargo check`, clippy `-D warnings`와 `cargo fmt --check`.

**CI 안정성.** head `03c1e365`에서 `pipeline-parallel-ci.yml`을 `workflow_dispatch`로 연속 10회 실행했다. workflow의 concurrency group이 같은 ref의 겹치는 실행을 취소하므로 한 번에 하나씩 dispatch했다. 10회 모두 `ubuntu24/20261025.353`에서 통과했고, 어느 job 로그에도 `pure virtual method called`가 없다.

| # | Run | 결과 | Runner image |
|---|---|---|---|
| 1 | 41045436759 | success | `ubuntu24/20261025.353` |
| 2 | 41048002710 | success | `ubuntu24/20261025.353` |
| 3 | 41049748107 | success | `ubuntu24/20261025.353` |
| 4 | 41051499873 | success | `ubuntu24/20261025.353` |
| 5 | 41053271228 | success | `ubuntu24/20261025.353` |
| 6 | 41055089949 | success | `ubuntu24/20261025.353` |
| 7 | 41056817489 | success | `ubuntu24/20261025.353` |
| 8 | 41058570988 | success | `ubuntu24/20261025.353` |
| 9 | 41060493226 | success | `ubuntu24/20261025.353` |
| 10 | 41062232317 | success | `ubuntu24/20261025.353` |

이 PR의 같은 workflow `pull_request` 실행 두 건(`e4312b89`의 41024288311, `03c1e365`의 41045135070)도 같은 image에서 통과했다.

**검증하지 못한 항목.**

- `pipeline_server_remote_real_models`는 컴파일만 했고 실행하지 않았다. CUDA와 Metal 빌드 대상이다.
- GPU backend 빌드에서는 allocator가 leak되므로 `server_exit_joins_worker`의 abort 텍스트 검사는 실패할 수 없다. 그곳에서 수정을 고정하는 것은 이 PR 이전에는 존재하지 않던 join 로그 줄이다. abort 텍스트 검사는 CPU backend 빌드에서만 의미가 있다.
- router의 `ROUTER_SHUTDOWN_TIMEOUT` 만료 후 프로세스 종료, `ctrl_c` 설치 실패 분기, `start_server`에서 `WORKER_EXIT_TIMEOUT`을 넘기는 worker는 코드 리뷰와 unit 수준 timeout 테스트로만 확인했고 프로세스 수준 테스트는 없다.
- Metal과 ROCm에서는 실행하지 않았다.

## 6. 학습 포인트

- **`main`에서 반환해도 스레드를 기다리지 않는다.** Rust에서 아무도 join하지 않은 스레드는 `exit()`가 C++ static 소멸자와 `atexit` handler를 실행하는 동안에도 계속 돈다. 그 순간 native 라이브러리 상태를 건드릴 수 있는 스레드는 scheduling이 맞아떨어지기만을 기다리는 crash다. 프로세스 종료를 소유하는 코드는 그런 스레드를 명시적으로 join해야 한다.
- **`main` 반환 후의 `pure virtual method called`는 대개 "static 소멸 후 사용"을 뜻한다.** C++ 소멸자 실행 중과 실행 후에는 객체의 vtable이 base class를 가리키므로, 소멸된 함수 지역 static에 대한 virtual 호출은 `__cxa_pure_virtual`에 도달한다. gdb에서 그곳에 breakpoint를 걸고 `thread apply all bt`를 실행하면 늦게 남은 스레드와 종료 경로를 한 번에 볼 수 있다.
- **backend별 singleton이 backend별 crash를 설명한다.** MLX는 Metal, CUDA, ROCm allocator를 의도적으로 leak하고 CPU fallback에는 static을 둔다. 같은 Rust 코드가 GPU 빌드에서는 안전했고 CI가 쓰는 CPU 빌드에서는 race가 있었다.
- **간헐적 CI 실패는 commit뿐 아니라 환경과도 대조한다.** runner image별로 실행을 나누자 10건 대 10건의 상관관계가 우연히 겹친 commit 범위와 분리되었고, #2687/#2688이 아니라 timing을 가리켰다.
- **정리 작업은 구조로 강제한다.** `start_server`의 로드 이후 절반을 별도 함수로 분리하자, blocking `Drop` guard를 쓸 수 없는 상황에서도 각 `?`를 일일이 점검하지 않고 모든 반환이 join에 도달하게 되었다.
- **정리 작업도 실패할 수 있다면 어느 오류를 남길지 정한다.** `join_model_worker_before_exit`는 serving을 끝낸 오류를 유지하고, serving이 성공했을 때만 shutdown 실패를 반환하므로 운영자는 항상 근본 원인을 본다.
