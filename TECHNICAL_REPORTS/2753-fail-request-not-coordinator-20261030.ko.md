# PR #2753: 원격 파이프라인 stage 오류 시 코디네이터가 아닌 요청 하나만 실패

**날짜**: 2026-10-30 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: 8c1a9af3에서 구현 완료, c76b0ce0에서 origin/main(99b3e563)을 병합함. PR 본문의 로컬 게이트는 8c1a9af3에서 GB10(spark-102) 기준 통과. 작성 시점에 병합된 head에서 두 번째 로컬 게이트가 실행 중이었고 호스팅 CI는 일부 대기 중이었음(`2-host logical`, `cargo-clippy`, `CPU-only link`, `OpenXLA feature compile`, `WebUI installed artifact`)
**언어**: Rust, Markdown
**위험도**: 낮음 (wire 변경 없음. 두 가지 실패 유형의 오류 문구가 바뀜. `LanguageModel`에 provided method 세 개, `EngineError`에 variant 두 개가 추가되므로, workspace 밖에서 `EngineError`를 exhaustive `match`하는 코드는 arm을 추가해야 함)

## 요약

PR #2753(#2727 해결)은 원격 파이프라인에서 코디네이터 전체가 abort되는 경로를 없앤다. 원격 stage가 forward에 실패하면(`Error` 응답, 전송 오류, `stage_timeout`) `PipelineServerModel`이 model worker 스레드에서 panic했고, 이 스레드는 `run_core_thread_or_abort` 아래에서 돌기 때문에 코디네이터 프로세스가 진행 중인 모든 요청과 함께 종료되었다. timeout에는 클러스터 secret이 필요 없다. 프레임 하나를 버리거나 변조하는 경로상 공격자면 충분하다. 이 PR은 `LanguageModel`에 fallible forward hook을 추가하고, engine에서 이를 `EngineError::Forward`로 전달하며, 스케줄러가 영향받은 row만 고정 문구 `model pipeline forward failed; see the server log for the stage error`로 실패시키게 한다. 거부된 `PrepareSequence`는 예전에는 클라이언트에 `Server busy: open: ... remote stage <addr> failed request: ...`로 도달했지만, 이제 capacity 라벨과 stage 주소 없이 `model pipeline could not prepare this request; see the server log for the stage error`로 도달한다. 리뷰 과정에서 같은 실패 경로의 runtime 수정 두 가지가 추가되었다. cancel과 release가 응답하지 않는 stage를 지나 모든 stage에 도달하고, row마다 logits shape가 다른 batched decode는 MLX `concatenate` 안에서 abort하는 대신 step을 실패시킨다.

## 1. 문제 정의

#2727은 #2723(이슈 #2714)의 잔여 문제 두 가지를 main 59c07652에서 확인해 모았고, engine 오류 경로와 클라이언트용 문구를 공유하므로 하나의 PR에서 함께 고치도록 요구했다. 0.8.0 릴리스 이후 LOW 우선순위로 잡혔으며, 항목 1은 secret 없이 도달 가능하다는 점이 명시되어 있었다.

### 1.1 원격 forward 오류가 코디네이터를 abort시켰다 (보안, DoS)

`LanguageModel`의 forward API는 infallible이었다. `forward_with_sequence_id`와 `forward_batched_with_ids`는 `Result`가 아니라 `MlxArray`를 반환한다. `PipelineServerModel`은 이 메서드를 `Result`를 반환하는 `RemotePipelineRuntime::forward_sequence`와 `forward_batched` 위에 구현했고, 모든 `Err`를 `panic!("pipeline forward failed for sequence ...")`(`server_runtime.rs:169`)로 바꿨다. model worker 스레드는 `run_core_thread_or_abort` 아래에서 돌며, 이 wrapper의 목적은 panic을 "불변식이 깨졌으니 프로세스를 abort한다"는 의미로 만드는 것이다. 그래서 stage 하나의 실패가 코디네이터와 그 위의 모든 요청을 함께 무너뜨렸다.

runtime은 stage의 `Error` 응답, 전송 오류, `stage_timeout` 만료에 대해 `Err`를 반환한다. 이 문제가 견고성 문제를 넘어 보안 문제인 이유는 마지막 경우 때문이다. AEAD 인증에 실패한 프레임은 연결을 닫고, exchange는 `stage_timeout`(`startup.rs` 기준 30초)까지 기다린 뒤 실패하며, 코디네이터는 abort된다. secret이 없는 경로상 공격자가 원할 때마다 이를 일으킬 수 있었다.

### 1.2 거부된 prepare에 "Server busy" 라벨이 붙고 stage 주소가 노출되었다 (보안, 정보 노출)

#2714는 거부된 `PrepareSequence`가 abort 대신 open 하나만 실패시키도록 만들었다. 하지만 메시지는 지나가는 모든 계층에서 덧붙여졌다. `broadcast_ack`가 `remote stage {addr} failed request: {message}`를 만들고, `Engine::open`이 이를 `EngineError::Open("prepare sequence state for {id}: ...")`로 감싸고, `allocate_sequence_state`가 `String`으로 펴고, admission이 `Server busy: `를 앞에 붙였다. 클라이언트는 capacity와 무관한 원인에 대해 capacity 라벨 아래에서 내부 stage 주소와 원격 `pp-seq-` id를 보게 되었다.

### 1.3 같은 경로의 abort 두 가지 더 (리뷰 중 발견)

- **teardown이 첫 번째 응답 없는 stage에서 멈췄다.** `cleanup_request`는 `CancelRequest`와 `ReleaseSequence`를 첫 오류에서 반환하는 `broadcast_ack`로 보냈다. timeout된 stage가 있으면 그 뒤의 모든 stage가 두 명령을 받지 못했다. 원격 stage에는 reaper가 없으므로 이런 실패마다 뒤쪽 stage에 시퀀스, 그 KV 캐시, 4096개 prepared-sequence 슬롯(#2746) 중 하나가 영구히 남았다. 실패가 쌓이면 상한이 차서 이후 모든 요청이 거부될 수 있었다.
- **shape가 섞인 batched logits가 MLX 안에서 abort했다.** `forward_batched`는 row별 logits를 shape 비교 없이 concatenate했다. `validate_logits_shape`는 각 row가 자기 입력과만 맞으면(시퀀스 축이 입력 길이 또는 1) 통과시키고, vocabulary 크기가 설정되지 않았으면 어떤 폭이든 허용한다. 폭이 다른 row가 섞이면 MLX가 오류를 보고할 수 없는 FFI 호출 안에서 throw했고 프로세스가 abort했다. 1.1을 고치고 나면 이것이 stage가 forward를 통해 코디네이터를 죽일 수 있는 다음 경로였다.

## 2. 변경 요약

| 영역 | 파일 | 변경 |
|---|---|---|
| 모델 trait | `mlxcel-core/src/generate.rs`, `language_model_ref.rs`, `src/loaded_model.rs` | provided method 세 개: `try_forward_with_sequence_id`, `try_forward_last_logits_with_sequence_id`, `try_forward_batched_with_ids`. 기본 구현은 infallible forward를 `Ok`로 감싸므로 모델 파일은 바뀌지 않는다. `&M`과 `LoadedModel`은 명시적으로 위임한다 |
| 파이프라인 모델 | `pipeline/server_runtime.rs` | 상수 `PIPELINE_FORWARD_FAILED`, `PIPELINE_PREPARE_REFUSED`. `PipelineServerModel`이 try hook 세 개와 `try_prepare_sequence_state`를 override해, `Err`이면 `{err:#}`를 sequence id와 함께 `warn`으로 기록하고 고정 문구를 반환한다. infallible forward는 오류 경로가 없는 호출자를 위해 panic을 유지한다 |
| Engine | `engine/mod.rs`, `prefill.rs`, `speculative.rs` | `EngineError::Forward(String)`(`forward: {0}`)와 `EngineError::Prepare(String)`(`{0}`). `step`/`submit`(`forward_rows` 경유), `prefill`과 `score`의 텍스트 분기, `prefill_cohort`, `verify`가 try hook을 호출한다. `Engine::open`은 거부된 prepare에 `Prepare`를 반환하고 pool capacity에는 `Open`을 유지한다 |
| 스케줄러 | `decode_tick.rs`, `planned_prefill.rs`, `prefill.rs`, `prompt_lookup.rs`, `paged_layout.rs`, `admission.rs`, `queued_adoption.rs` | 실패한 decode step은 모든 row를 메시지 그대로 실패시키고 `Finished(Error)`를 강제한다(`fail_forward_rows`). 실패한 prefill piece는 `PieceFailure::Abort(msg)`가 된다(`prefill_piece_failure`). 실패한 cohort prefill은 row별 prefill로 fallback하지 않고 row들을 실패시킨다. 실패한 verify는 그 요청을 실패시킨다. `allocate_sequence_state`는 `EngineError`를 반환하고, `allocation_failure_message`는 `Prepare`를 그대로 전달하며 capacity에는 `Server busy: ...`를 유지한다 |
| Runtime | `pipeline/runtime.rs` | `broadcast_best_effort`가 cancel과 release를 모든 stage에 보내고, 응답하지 않는 stage는 로그로 남긴다. prepare rollback과 `release_sequence_state_by_id`도 이를 쓴다. `forward_batched`는 logits shape가 첫 row와 다른 row를 거부한다 |
| 테스트 | `stub_stage.rs`(신규), `server_runtime_tests.rs`(신규), `scheduler_pipeline_failure_tests.rs`(신규), `engine/tests.rs`, `runtime.rs`, `paged_layout.rs`, `language_model_ref.rs` | 스크립트 stage를 공유 `cfg(test)` 모듈로 옮기고 logits 응답과 지정한 forward 하나의 fault를 지원하게 함. 신규 테스트 11개, 이동 4개 |
| 문서 | `docs/distributed.md` | "A stage failure fails one request, not the coordinator": 두 문구, teardown, mixed-shape 거부 |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 22 |
| 라인 | +1841 / -489 |
| 신규 테스트 | 11개(스케줄러 4, runtime 2, 파이프라인 모델 2, allocation 메시지 2, engine 1), 이동 4개, 확장 1개 |
| 커밋 | 5개와 main 병합 1개 |

| 커밋 | 목적 |
|---|---|
| 26c21722 | `run_core_thread_or_abort` 안에서 실제 `RemotePipelineRuntime`으로 도는 failing-first 스케줄러 테스트 |
| ba98789a | 수정 본체: try hook, engine 오류, 스케줄러 처리, 클라이언트용 문구 |
| 7ca6a515 | 실패 테스트를 고정 포트 20300-20307에서 ephemeral `127.0.0.1:0`으로 옮김 |
| 53cd14bd | 모든 stage에 대한 best-effort teardown, mixed-shape batched logits 거부 |
| 8c1a9af3 | `allocation_failure_message` 단위 테스트, teardown과 mixed-shape 동작 문서화 |
| c76b0ce0 | origin/main 병합 |

## 3. 기술적 선택과 그 이유

### 3.1 시그니처 변경 대신 infallible API 옆에 fallible hook 추가

**컨텍스트.** forward 시그니처는 저장소의 모든 모델이 구현한다. 다른 곳에서 계산하고 그곳에서 실패할 수 있는 구현은 원격 파이프라인 코디네이터 하나뿐이다.

| 선택지 | 평가 |
|---|---|
| forward 주변에서 panic을 catch | 이슈에서 기각됨. fail-fast wrapper는 panic이 불변식 위반을 뜻하도록 존재하며, catch하면 실제 버그도 삼키고 MLX 상태도 알 수 없게 된다 |
| sentinel logits와 별도 채널의 오류 반환 | 타입이 없고 많은 호출 지점 중 하나에서 놓치기 쉽다 |
| 모든 forward를 `Result` 반환으로 변경 | 모델 하나만 낼 수 있는 실패 때문에 모든 모델 파일을 고쳐야 한다 |
| **선택: 기본 구현이 infallible 메서드를 감싸는 provided `try_*` 메서드** | 모델 변경 없음. engine은 오류 경로가 있는 모든 곳에서 try 형태를 호출하고, override는 `PipelineServerModel`만 한다 |

이는 #2714가 같은 이유로 추가한 `try_prepare_sequence_state`와 같은 형태다. `try_forward_with_sequence_id`에 적힌 계약은 `Err`가 시퀀스 상태가 진행되지 않았음(또는 요청과 함께 폐기되었음)을 뜻하므로 호출자가 되돌릴 것이 없다는 것이다. 그래서 `EngineError::Forward`는 `EngineError::Eval`이 필요로 하는 `unwind_appends` 경로를 건너뛸 수 있다.

**트레이드오프.** wrapper는 try hook을 명시적으로 위임해야 한다. `LoadedModel`은 enum wrapper라서, trait 기본 구현에 맡겼다면 기본 구현이 `LoadedModel` 자신의 infallible forward를 감쌌을 것이고 파이프라인 override는 실행되지 않았을 것이다. PR은 명시적 위임과 그 이유를 설명하는 주석을 추가했다. 앞으로 생길 wrapper 타입도 같은 의무를 지지만 이를 강제하는 장치는 없다.

### 3.2 고정된 클라이언트용 문구, 상세는 로그로

**컨텍스트.** stage의 오류는 운영자에게는 유용하고 클라이언트에게는 해롭다. stage 주소, 원격 `pp-seq-{n}-{hex}` id(#2746이 미리 점유할 수 없도록 접미사를 비밀로 만든 바로 그 값), stage 쪽 텍스트가 들어 있다.

`PipelineServerModel`은 이 둘을 모두 보는 마지막 지점이다. 여기서 `{err:#}`를 sequence id와 함께 `warn`으로 기록하고 상수를 반환한다. 그 위의 모든 계층은 상수를 바꾸지 않고 전달한다. `EngineError::Prepare`는 메시지만 표시하고, `allocation_failure_message`는 접두사 없이 반환하며, `fail_sequence_with_message`는 그대로 보낸다. 스케줄러 테스트는 상수와의 일치를 확인하고, `assert_client_safe`는 stage 주소, `127.0.0.1`, `pp-seq-`, `stub stage`, `Server busy`가 들어간 텍스트를 추가로 거부한다.

`Server busy: ...`는 실제 capacity 실패(pool 슬롯, paged budget)에만 남는다. 이를 재시도 가능한 오류로 다루는 클라이언트와 로드 밸런서는 재시도가 도움이 되는 경우에만 그렇게 하게 된다.

### 3.3 row를 실패시키고, 실패한 forward는 재시도하지 않음

스케줄러 수준에서 실패한 forward는 재시도하지 않는다. decode step은 모든 row를 실패시키고, cohort prefill은 기존 row별 fallback을 타지 않고 모든 row를 실패시키며, verify는 그 요청을 실패시킨다. 재시도는 같은 stage에 같은 원격 exchange를 반복할 뿐이고, timeout의 경우 시도마다 단일 model worker 스레드에서 `stage_timeout`이 한 번 더 든다. 또한 forward가 `Err`를 반환하기 전에 runtime이 이미 그 시퀀스에 cancel과 release를 보냈으므로, 재시도는 stage가 더 이상 갖고 있지 않은 시퀀스에 대해 실행된다.

실패한 row는 `Finished(Error)`로 강제되며, 같은 tick에서 먼저 기록된 종료 상태를 덮어쓴다. 이렇게 해서 실패한 forward가 쓰지 않은 상태가 prompt cache에 기증되지 않는다. `fail_desynchronized_sequence`가 이미 적용하던 규칙과 같다.

forward를 한 번 반복하는 경로가 하나 있다. decode lookahead prime(`submit`)이 `Forward`를 받으면 `None`을 반환하고, tick은 synchronous step으로 fallback하며, 그 step이 forward를 다시 실행한 뒤 row를 실패시킨다. prime의 기존 "어떤 실패든 fallback" 형태를 유지한 것이며, tick당 exchange 한 번으로 제한된다.

### 3.4 모든 stage에 대한 best-effort teardown

**컨텍스트.** teardown 명령의 결과에 따라 행동하는 쪽은 없지만, stage 쪽 상태를 해제하는 수단은 이 명령뿐이다.

`broadcast_ack`는 실패하면 호출자가 멈춰야 하는 명령(prepare, shutdown)에는 맞다. cancel과 release에서 첫 실패에 멈추는 것은 정반대로 틀렸다. 실패한 stage는 그 명령이 가장 필요 없을 stage이고, 그 뒤의 stage들이 누수되는 쪽이다. `broadcast_best_effort`는 각 stage에 순서대로 보내고 stage별로 기록한다(id를 준비한 적 없는 stage는 `Error`로 답하므로 이는 `debug`, 응답이 없으면 `warn`). `cleanup_request`, prepare rollback, `release_sequence_state_by_id`가 모두 이를 쓴다.

**트레이드오프.** 전송은 여전히 순차적이다. 응답을 멈춘 stage는 그 stage에 보내는 teardown 명령마다 model worker 스레드에서 `stage_timeout`을 한 번씩 소모한다(5절).

### 3.5 `concatenate` 전에 batched row shape 확인

logits 검증은 row별이고, vocabulary 크기가 설정되지 않으면 올바른 폭을 알 수 없다. row끼리 비교할 수 있는 첫 지점은 batch이며, graph 구성 시점의 MLX shape 오류는 이를 전달할 수 없는 FFI 경계를 넘어 빠져나간다. concatenate 전에 각 row의 shape를 첫 row와 비교하면 이것이 일반 `Err`가 되고, 3.3의 경로를 따라 batch의 모든 row를 `PIPELINE_FORWARD_FAILED`로 실패시킨다.

### 3.6 마지막 위치만 반환하는 stage에 대한 last-logits

이슈는 `try_forward_last_logits_with_sequence_id`를 전체 forward 뒤에 `logits_at_position(last_pos)`를 붙인 것으로 정의했다. stage는 마지막 위치만 반환할 수 있고(`[1, 1, vocab]`, runtime의 shape 검사가 허용함), 이를 `last_pos > 0`로 인덱싱하면 범위를 벗어난다. override는 반환된 시퀀스 축의 길이가 1이면 위치 0을 쓴다.

## 4. 검증

### 4.1 Failing-first 근거

26c21722는 수정되지 않은 main 위에 스케줄러 테스트를 추가한다. 이 테스트는 스크립트 stage를 상대로 실제 `RemotePipelineRuntime`(인증된 TCP, 실제 exchange, timeout, cleanup 코드) 위에 `PipelineServerModel`을 만들고, `LoadedModel`로 감싸 `BatchScheduler`로 구동하며, 테스트 전체를 `run_core_thread_or_abort` 안에서 실행한다. 따라서 테스트 프로세스의 동작이 곧 코디네이터 프로세스의 동작이다.

| 테스트 | main에서 | 현재 |
|---|---|---|
| `remote_forward_error_fails_one_request_and_the_coordinator_serves_the_next` | `server_runtime.rs:169:35`에서 panic, 바이너리가 134(SIGABRT)로 종료 | 첫 요청은 정확히 `PIPELINE_FORWARD_FAILED`, 두 번째 요청은 `[3, 3]`으로 완료 |
| `remote_forward_timeout_fails_one_request_and_the_coordinator_serves_the_next`(timeout 300 ms, decode forward에서 stage가 무응답) | 같은 abort | 위와 같음 |
| `remote_prefill_failure_fails_one_request_and_the_coordinator_serves_the_next` | 같은 abort | `PieceFailure::Abort(PIPELINE_FORWARD_FAILED)`, 토큰 0개, 다음 요청 완료 |
| `refused_prepare_reaches_the_request_as_the_client_safe_message` | `Server busy: open: ... pp-seq-0-...: remote stage 127.0.0.1:20304 failed request: ...` | 정확히 `PIPELINE_PREPARE_REFUSED`, stage는 `prepare, release`를 받음 |
| `remote_runtime_teardown_reaches_every_stage_past_a_silent_one`(53cd14bd 수정을 끈 상태) | 무응답 stage 뒤의 stage가 `prepare`만 받음 | `prepare, cancel, release, release`를 받음 |
| `remote_runtime_refuses_a_batch_whose_rows_disagree_on_logits_shape`(수정을 끈 상태) | `concatenate`에서 SIGABRT | `logits with shape`를 담은 `Err` |

forward 테스트는 stage의 전체 명령 로그도 고정한다. 예를 들어 `prepare, run_entry, run_entry, cancel, release, release, prepare, run_entry, run_entry, release`는 다음 요청이 시작되기 전에 runtime이 실패한 시퀀스를 정리한다는 것을 보여준다.

### 4.2 로컬 게이트 (head 8c1a9af3)

GB10(spark-102)에서 `gpu-lock run --tag issue-2727`, `--profile test-fast`, `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`로 실행했다. `cargo fmt --check`와 `-D warnings` clippy(mlxcel과 mlxcel-core, lib과 tests)는 깨끗했다.

| 스위트 | 통과 |
|---|---|
| mlxcel-core `engine::`, `language_model_ref::` | 73 |
| `distributed::` | 1557, ignored 4 |
| 변경된 스케줄러 모듈(pipeline failure, allocation message, model-owned cache와 lookahead, prompt lookup, prompt cache, VLM fallback, mrope) | 97 |
| `distributed_integration`(`cuda,test-utils`) | 20 |
| `pipeline_e2e`(`cuda,test-utils`) | 30 |
| `pipeline_remote_real_models`(`--include-ignored --skip glm_moe_dsa`) | 10 |
| `pipeline_server_remote_real_models`(동일) | 7 |
| `pipeline_ci_multi_stage_real_models`(동일) | 4 |
| `disaggregated_serving_e2e`(`--include-ignored`) | 2/2 |
| `disaggregated_router_e2e`(`--include-ignored`) | 7/7 |

### 4.3 수동 확인

`--pp-auto 2` 코디네이터와 `--pp-peer` 프로세스 두 개로 loopback에서 qwen3-0.6b를 서빙하고, 스트리밍 chat 요청 중에 마지막 stage를 `kill -9`로 종료했다. 스트림은 `finish_reason: "error"`와 `[DONE]`으로 끝났고, 코디네이터는 살아 있었으며 `/health`는 200을 반환했다. 다음 요청은 stage가 사라졌으므로 `Generation error: model pipeline could not prepare this request; see the server log for the stage error`를 받았다. 어떤 클라이언트 텍스트에도 stage 주소나 `pp-seq-` id가 없었다.

### 4.4 main 병합 이후

c76b0ce0은 origin/main(99b3e563, #2742와 #2746 포함)을 병합한다. 위 게이트는 그 이전 결과다. 작성 시점에 병합된 head에서 두 번째 로컬 게이트가 실행 중이었고 호스팅 CI도 일부 대기 중이었으므로, 그 결과는 여기에 기록하지 않는다.

## 5. 남은 공백과 위험

- **격리는 요청 단위이지 지연시간 단위가 아니다.** model worker는 단일 스레드이고 모든 원격 exchange는 동기식이다. 응답을 멈춘 stage는 실패한 forward에 `stage_timeout` 한 번, 그 stage에 보내는 cancel, release, 스케줄러의 이후 release에 각각 한 번씩을 소모한다(teardown 테스트의 무응답 stage는 `cancel, release, release`를 받는다). 기본값 30초라면 실패 한 번마다 코디네이터의 다른 모든 요청이 약 2분 멈출 수 있다. 코디네이터는 살아남지만 처리량은 그렇지 않다. teardown 명령에 더 짧은 timeout을 쓰거나 이미 도달 불가로 알려진 stage를 건너뛰면 이를 제한할 수 있다.
- **실패한 요청마다 release가 두 번 간다.** runtime은 실패한 forward에서 release하면서 id 매핑을 유지하고, 스케줄러가 시퀀스를 닫을 때 다시 release한다. 정상 stage에는 무해하지만 무응답 stage에는 위 timeout 중 하나가 된다.
- **lookahead fallback은 forward를 한 번 반복한다.** 3.3 참조. 파이프라인 실패 테스트는 `run_entry`가 반복되지 않는 명령 로그를 고정하므로 이 경로는 그 테스트에서 실행되지 않는다.
- **스트리밍 chat 클라이언트는 오류 문구를 보지 못한다.** chat SSE 경로는 `finish_reason: "error"`로 스트림을 끝내고 메시지는 싣지 않는다. main과 동일하며 이 PR의 범위 밖이다.
- **wrapper 타입은 try hook을 위임해야 한다.** trait 기본 구현에 의존하는 새 `LanguageModel` wrapper는 파이프라인 모델의 infallible 경로를 조용히 되살린다. `LoadedModel`과 `&M`은 처리되었고, `&M` 테스트는 override에 도달하는지 확인한다.
- **실행하지 않음.** `glm_moe_dsa`(`glm5-4bit`는 이 128 GB 장비에서 로드되지 않음), Thunderbolt와 RDMA 전송, macOS. 성능 측정은 하지 않았다. 성공 경로에 추가되는 것은 forward당 `Result` wrapper 하나와 batched row당 shape 비교 하나다.

## 6. 학습 포인트

### 6.1 기본 구현이 감싸는 fallible 쌍둥이 메서드를 추가

널리 구현된 trait 메서드의 구현 하나에 실패 모드가 생기면, 기본 구현이 infallible 메서드를 감싸는 provided `try_*` 메서드를 추가해 변경을 그 구현과 오류를 처리할 수 있는 호출자로 국한할 수 있다.

```rust
fn try_forward_with_sequence_id(
    &self,
    input_ids: &MlxArray,
    seq_id: Option<SequenceId>,
    caches: &mut [KVCache],
    mask: Option<&MlxArray>,
) -> Result<UniquePtr<MlxArray>, String> {
    Ok(self.forward_with_sequence_id(input_ids, seq_id, caches, mask))
}
```

함정은 위임이다. 모든 wrapper(`&M`, enum dispatcher)가 새 메서드를 명시적으로 전달해야 한다. 그렇지 않으면 기본 구현이 wrapper의 infallible 메서드를 감싸고 override에는 도달하지 않는다.

### 6.2 원인을 아는 가장 낮은 계층에서 클라이언트 문구를 정한다

계층마다 컨텍스트를 덧붙이는 오류 문구는 결국 가장 낮은 계층이 알던 모든 것, 즉 내부 주소와 식별자까지 담게 된다. 원인을 아는 곳에서 클라이언트에 보일 문구를 한 번 정하고, 상세는 그곳에서 로그로 남기며, 위의 계층은 접두사 없이 전달하게 한다. capacity 라벨은 capacity에만 써서 클라이언트의 재시도 정책이 올바르게 동작하게 한다.

### 6.3 teardown은 첫 실패에서 멈추면 안 된다

첫 오류에서 중단하는 broadcast는 setup에는 맞고 cleanup에는 틀리다. cleanup에서 실패하는 peer는 상태를 가지고 있을 가능성이 가장 낮고, 그 뒤의 peer들이 누수되는 쪽이다. 모두에게 보내고, 각 실패를 기록하고, 아무것도 반환하지 않는다.

### 6.4 반환값뿐 아니라 프로세스 결과를 테스트한다

스케줄러 테스트는 실제 model worker와 같은 fail-fast wrapper인 `run_core_thread_or_abort` 안에서 돈다. 그래서 "코디네이터가 abort한다"가 반환된 panic payload에서 추론하는 대상이 아니라 테스트 결과(종료 코드 134)가 되며, 어떤 `Result`로도 보고될 수 없었던 mixed-shape `concatenate` abort도 이 방식으로 잡혔다.

## 7. 참고 자료

- 이슈 #2727: fail one request, not the coordinator, on remote pipeline errors
- PR #2723(이슈 #2714): fallible `try_prepare_sequence_state`와 시퀀스별 소유권. 이 PR은 그 잔여 문제 두 가지를 해결함
- PR #2746(이슈 #2726): teardown 수정과 메시지 비공개 처리가 보호하는 4096개 prepared-sequence 상한과 CSPRNG `pp-seq-` 접미사
- `src/worker_failfast.rs`: `run_core_thread_or_abort`
- `docs/distributed.md`, "What the secret protects"
