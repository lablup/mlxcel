# PR #2769: 원격 파이프라인 실패 경로와 stage 상태에 상한 두기

**날짜**: 2026-10-31 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: base bebdd62c(동작은 main 5c8ef80a와 동일) 위에서 8993404e로 구현 완료. PR 본문의 로컬 게이트는 GB10(spark-102)에서 통과. 작성 시점에 worktree에서 두 번째 로컬 게이트가 실행 중이었고 호스팅 CI는 일부 대기 중이었음(`2-host logical`, `cargo-clippy`, `CPU-only link`, `OpenXLA feature compile`, `python tooling tests`, `WebUI bundle`, `WebUI installed artifact`)
**언어**: Rust, Markdown
**위험도**: 낮음 (wire 변경과 public API 변경 없음. 코디네이터는 teardown 응답을 최대 2초만 기다리고, prepared-sequence 상한에 도달한 stage는 10분 동안 유휴였던 시퀀스를 해제함)

## 요약

PR #2769(#2762 해결)는 응답하지 않는 원격 파이프라인 stage 하나가 코디네이터에 주는 비용에 상한을 둔다. #2753 이후 stage 실패는 프로세스 abort 대신 요청 하나의 실패로 끝났지만, 단일 model worker는 실패한 요청마다 여전히 `stage_timeout` 약 네 번(고정값 30초 기준 약 2분)을 소비했다. 실패를 감지하는 데 한 번, 그 뒤 cancel, release, 스케줄러의 close가 보내는 두 번째 release에 각각 timeout 전체를 기다렸다. 이 PR은 teardown 응답 대기를 2초로 제한하고, 실패한 forward가 코디네이터의 시퀀스 매핑을 버리게 해 release가 한 번만 나가도록 하여 비용을 `stage_timeout` 한 번과 짧은 대기 두 번(약 34초)으로 줄인다. stage 쪽에서는 4096개 상한이 찬 상태에서 들어온 `PrepareSequence`가 10분 동안 유휴였던 시퀀스를 회수하게 하고, 소유자 없는 activation에 대한 프레임별 경고를 새 `stage-drop` 클래스로 rate limit하며, 테스트 전용 서비스 설정값을 프로덕션 빌드에서 컴파일되지 않게 한다.

## 1. 문제 정의

#2762는 #2753(이슈 #2727)과 #2746(이슈 #2726)의 리뷰, 그리고 #2753 리포트 5절이 실패 경로에 남긴 가용성 및 정리 관련 잔여 문제를 모은 이슈다. 항목 1은 클러스터 secret 없이도 도달 가능하기 때문에 우선순위가 medium이었다. 프레임별 AEAD는 경로상 공격자가 프레임을 버리는 것을 막지 못하며, 버려진 프레임, 네트워크 분할, 멈춘 stage 프로세스는 코디네이터 입장에서 모두 같은 stage timeout으로 보인다.

### 1.1 응답 없는 stage 하나가 model worker를 약 2분간 멈췄다 (가용성)

`RemotePipelineRuntime::exchange`는 teardown을 포함한 모든 명령에 대해 `stage_timeout`을 기다렸다. 실패한 `forward_sequence`는 `cleanup_request`를 호출했고, 이는 `broadcast_best_effort`로 `CancelRequest`와 `ReleaseSequence`를 보냈다. 응답 없는 stage에 대해 각각 30초 전체를 기다렸다. `cleanup_request`는 `request_ids` 항목도 남겨 두었기 때문에, 이후 스케줄러가 실패한 시퀀스를 close할 때 `release_sequence_state_by_id`가 매핑을 찾아 세 번째로 release를 보냈고 다시 30초가 걸렸다. model worker는 동기 exchange를 하는 단일 스레드이므로, 코디네이터의 다른 모든 요청이 네 번의 timeout 뒤에서 기다렸다. `remote_runtime_teardown_reaches_every_stage_past_a_silent_one`은 이 이중 release를 `prepare, run_entry, cancel, release, release`로 고정하고 있었다.

### 1.2 stage의 prepared 시퀀스가 만료되지 않았다 (자원 누수)

stage의 `sequence_owners` 맵은 cancel, release, shutdown, 실패한 activation으로만 비워졌다. 요청 처리 중 crash한 코디네이터는 진행 중이던 시퀀스와 그 KV 캐시를 stage 프로세스가 재시작될 때까지 `MAX_STAGE_SEQUENCES = 4096`에 계속 포함시켰다. 상한에 도달하면 새 `PrepareSequence`는 모두 거부되므로, 코디네이터 crash가 충분히 쌓이면 stage가 모든 작업을 거부하게 된다.

### 1.3 소유자 없는 activation 경고에 제한이 없었다 (로그 범람)

stage에서 아무도 prepare하지 않은 시퀀스의 activation은 owner 검사에 실패하고 `report_activation_failure`에 도달하는데, 두 지점에서 프레임마다 `warn`을 하나씩 남겼다. 마지막 stage의 "reply nonce 없음" 분기도 마찬가지였다. activation은 인증된 upstream peer만 보낼 수 있으므로 secret이 필요하지만, 오동작하거나 버그가 있는 peer가 프레임 속도로 로그를 채울 수 있었다.

### 1.4 프로덕션에 남은 테스트 전용 설정값과 도달 불가능한 분기

`set_max_sequences`와 `spawn_service`의 `sequence_cap: Option<usize>` 파라미터가 release 빌드에 포함되어 있었고, 프로덕션은 항상 `None`을 넘겼다. `run_activation`에서 마지막 stage의 `reply_target(..) == None` 분기는 도달할 수 없었다. owner 검사가 executor보다 먼저 실행되고, 서비스는 한 번에 프레임 하나만 처리하기 때문이다. 그런데도 이 분기는 자체적인 경고 후 버리기 처리를 갖고 있어서 읽는 사람이 따로 추론해야 했다.

### 1.5 잔여 목록에서 제외한 항목

#2753 리포트는 "decode lookahead fallback이 forward를 한 번 반복한다"도 적어 두었다. #2762는 현재 이것이 일어날 수 없음을 확인했다. `PipelineServerModel`은 model-owned layout을 선언하고 `supports_decode_lookahead_rewind`를 override하지 않으므로 `Engine::can_unwind_lookahead`가 false이고, `EngineError::Forward`를 반환할 수 있는 유일한 모델에 대해 스케줄러는 lookahead를 준비하지 않는다. 이 항목에 대해서는 변경하지 않았다.

## 2. 변경 요약

| 영역 | 파일 | 변경 |
|---|---|---|
| 코디네이터 runtime | `src/distributed/pipeline/runtime.rs` | `TEARDOWN_ACK_TIMEOUT = 2 s`. `teardown_ack_timeout = stage_timeout.min(2 s)` 필드와 `#[cfg(test)]` builder `with_teardown_ack_timeout`. `exchange`는 `exchange_within(peer, request, timeout)`에 위임. `broadcast_best_effort`는 teardown timeout을 넘김. `cleanup_request`는 `fail_sequence(seq_id, request_id)`가 되어 teardown 전에 `request_ids` 항목을 제거하며, `forward_sequence`의 다섯 실패 출구 모두에서 호출됨 |
| Stage 서비스 | `src/distributed/pipeline/remote_service.rs` | `sequence_owners: HashMap<String, OwnedSequence { owner, last_active }>`. prepare, re-prepare, 수락된 `RunEntry`, owner 검사를 통과한 activation에서 `touch_sequence`. `STAGE_SEQUENCE_IDLE_TTL = 600 s`와 `reap_idle_sequences`, 새 id가 상한이 찬 것을 발견할 때만 호출. 세 가지 drop 경고에 `log_stage_drop`. `StageOverrides`가 `sequence_cap`을 대체. 도달 불가능한 마지막 stage 분기는 `ok_or_else(..)?`로 변경 |
| 경고 budget | `src/distributed/handshake_warn.rs` | `HandshakeWarnClass::StageDrop`, `CLASS_COUNT` 4에서 5로, `Display` 문구 `stage-drop` |
| 테스트 | `remote_service_tests.rs`, `runtime.rs`, `handshake_warn_tests.rs`, `scheduler_pipeline_failure_tests.rs` | 새 테스트 4개, 강화 1개, 스케줄러 명령 로그 3개를 단일 release로 갱신, `Display` assertion 1개. stage 테스트용 `tracing` 캡처 writer와 `Harness::spawn_capped_with_ttl` |
| 문서 | `docs/distributed.md` | 단일 release, 2초 teardown 대기와 34초 수치, 상한에서의 유휴 회수, 다섯 번째 `stage-drop` 클래스. 이 문서의 한국어판은 없음 |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 7 |
| 라인 | +581 / -95 |
| 새 테스트 | 4개 (`silent_stage_failure_is_bounded_by_one_stage_timeout`, `idle_sequences_are_reaped_when_the_cap_is_reached`, `recently_active_sequences_are_not_reaped`, `unowned_activation_warnings_are_rate_limited`) |
| 강화된 테스트 | `remote_runtime_teardown_reaches_every_stage_past_a_silent_one`(release 한 번), 스케줄러 파이프라인 실패 테스트 3개(명령 로그), `classifies_handshake_errors`(`stage-drop`) |
| 커밋 | 2 |

| 커밋 | 목적 |
|---|---|
| 968c011f | #2762의 네 항목 전부, 테스트, 문서 |
| 8993404e | 리뷰 LOW 항목: `fail_sequence` 문서가 상태를 버리는 출구를 정확히 명시. 회수 테스트를 200 ms TTL과 300 ms 대기에서 1초와 1.3초로 변경 |

## 3. 기술적 선택과 그 이유

### 3.1 teardown에만 짧은 응답 대기

**컨텍스트.** cancel과 release의 결과를 사용하는 쪽은 없다. 응답이 오든 말든 프레임은 transport에 쓰이고, 정상이지만 바쁜 stage는 순서대로 처리하며, 늦게 온 응답은 그때 실행 중인 exchange의 nonce matcher가 버린다. `stage_timeout` 전체를 기다려서 얻는 것은 stage가 응답하지 않았다는 경고 한 줄뿐이다.

| 선택지 | 평가 |
|---|---|
| `stage_timeout`을 전역으로 줄이기 | `RunEntry` 대기까지 줄어든다. `RunEntry`는 큰 prefill이나 첫 실행 CUDA 커널 컴파일 때문에 정당하게 오래 걸릴 수 있다. `stage_timeout`을 설정 가능하게 만드는 것은 #2762 범위 밖 |
| fire-and-forget teardown(전혀 기다리지 않음) | 어느 stage가 응답을 멈췄는지 운영자에게 알려 주는 stage별 로그를 잃음 |
| stage가 응답 없음으로 알려진 동안 새 prepare를 즉시 실패시키는 peer별 circuit breaker | 모든 요청이 모든 stage를 필요로 하므로 서비스를 복구하지 못하고, 복구 probe 상태가 추가됨. #2762에서 기각 |
| **선택: `broadcast_best_effort`에 `stage_timeout.min(2 s)`로 `exchange_within` 사용** | 기존 경로에 파라미터 하나. 모든 teardown 전송(실패한 forward, prepare rollback, 스케줄러의 정상 close)이 상한을 적용받고, teardown이 아닌 명령은 `stage_timeout`을 유지 |

`min`은 `stage_timeout`이 2초보다 짧은 테스트나 향후 설정에서 teardown을 forward보다 오래 기다리지 않게 한다.

**트레이드오프.** release 응답에 2초 넘게 걸리는 정상 stage(예를 들어 코디네이터가 포기한 `RunEntry`를 아직 처리 중인 경우)는 이제 release가 실제로 적용되는데도 코디네이터에 "did not acknowledge" 경고를 남긴다. 영향은 이 경고뿐이며, 프레임은 이미 전송되었다.

### 3.2 실패한 forward는 teardown 전에 매핑을 버린다

**컨텍스트.** #2753이 정한 `try_forward_with_sequence_id` 계약은 `Err`가 시퀀스 상태가 폐기되었음을 뜻한다는 것이다. runtime은 이를 지키지 않았다. stage들은 정리했지만 `request_ids`를 남겼고, 그래서 스케줄러의 close가 다시 정리했다.

`fail_sequence`는 먼저 `request_ids` 항목을 제거한 뒤 cancel과 release를 broadcast한다. 그러면 스케줄러의 close는 매핑을 찾지 못하고 아무것도 보내지 않은 채 반환한다. 각 stage는 cancel과 release를 정확히 한 번씩 받고, 응답 없는 stage의 비용은 timeout 전체 세 번에서 teardown 대기 두 번으로 줄어든다.

`RunEntry` exchange 전에 실행되는 두 `serialize_mlx_array` 출구는 의도적으로 `fail_sequence`를 호출하지 않는다. 그 시점에는 stage에 아무것도 도달하지 않았으므로 cancel이 필요 없고, 매핑을 남겨 두면 스케줄러의 close가 시퀀스를 정확히 한 번 release한다. 마무리 커밋은 처음에 모든 `Err`가 상태를 버린다고 적었던 doc 주석을 바로잡았다.

`forward_batched`에서는 각 row가 `forward_sequence`를 거치므로, 실패한 row는 자기 매핑만 버리고 `?`가 그 자리에서 batch를 멈춘다. 이미 성공한 row와 아직 보내지 않은 row는 매핑을 유지하며, 스케줄러가 batch를 실패시킬 때 한 번씩 release된다. #2753의 mixed-shape 거부도 같은 규칙을 따른다. 문제의 row는 forward 자체가 성공했으므로 매핑이 스케줄러의 close까지 남는다.

### 3.3 유휴 시퀀스는 상한에 도달했을 때만 회수

**컨텍스트.** 1.2의 누수는 admission을 막을 때만 문제가 된다. 타이머로 회수하는 stage는 상한에 여유가 있는데도 살아 있지만 느린 요청의 시퀀스를 아무 이득 없이 해제할 수 있다.

| 선택지 | 평가 |
|---|---|
| 타이머 기반 백그라운드 reaper | 추가 task와 스케줄링. 막힌 것이 없어도 회수함 |
| 모든 `PrepareSequence`에서 회수 | 상한 아래에서는 이득 없이 요청마다 O(n) 스캔 |
| **선택: 새 id의 `PrepareSequence`가 상한이 찬 것을 발견하면 600초 이상 유휴인 시퀀스를 모두 종료하고, 그래도 가득 차 있을 때만 거부** | 상한에 도달하기 전까지 비용 0. 상한에서의 4096개 스캔은 마이크로초 단위. 개수를 담은 `warn` 하나로 이벤트가 드러남 |

"활동"은 그 시퀀스에 대한 소유자의 모든 메시지를 말한다. prepare, re-prepare, 수락된 `RunEntry`, owner 검사를 통과한 activation이다. 이 검사들에서 단순 `contains_key`를 `touch_sequence`가 대체하므로, 소유권을 검사하는 곳에서 갱신을 빠뜨릴 수 없다. 다른 identity가 지목한 시퀀스는 갱신되지 않는다. 회수는 cancel이나 release와 같은 `finish_request` 경로를 거치므로 캐시도 같은 방식으로 해제된다.

**트레이드오프.** 소유자로부터 10분간 아무 메시지도 받지 못한 살아 있는 요청이 회수되면, 다음 `RunEntry`에서 "not prepared"로 실패한다. #2753 이후 이는 코디네이터가 아니라 요청 하나를 클라이언트용 forward 메시지로 실패시킨다. 실제로 스케줄러는 decode tick마다 활성 시퀀스 각각에 `RunEntry`를 보내므로, 10분간 조용하다는 것은 코디네이터가 사라졌거나 멈췄다는 뜻이다.

### 3.4 `diagnostic`을 재사용하지 않고 다섯 번째 경고 클래스

handshake 경고 budget(#2655, #2692; 클래스는 #2712에서 추가)은 이미 경고를 클래스로 나눠, 시끄러운 출처 하나가 다른 경고를 debug로 밀어내지 못하게 한다. 소유자 없는 activation은 인증된 peer에서 오므로 handshake noise도 handshake diagnostic도 아니다. `diagnostic`에 넣으면 오동작하는 upstream이 운영자가 의존하는 typed refusal(secret 불일치 추정, suite 불일치)을 굶길 수 있다. `StageDrop`은 분당 5개의 자체 window를 가지며, 나머지는 debug로 가고 기존 rollover 요약이 억제된 개수를 보고한다.

### 3.5 release 빌드에 존재하지 않는 테스트 설정값

`StageOverrides`는 모든 필드가 `#[cfg(test)]`인 `#[derive(Default)]` struct라서, release 빌드에서는 빈 struct이고 `apply`와 `log_guard`는 비어 있는 `#[cfg(not(test))]` 본문이다. 프로덕션은 `spawn_service(.., StageOverrides::default())`를, 테스트는 `spawn_with_overrides`를 호출한다. 이렇게 해서 `allow(dead_code)`를 추가하지 않고도 배포되는 시그니처에서 `Option<usize>` 파라미터가 사라지고, 유휴 TTL과 로그 캡처가 들어갈 자리가 생긴다. 확인 수단은 test가 아닌 빌드에 대한 `cargo clippy --features cuda -p mlxcel --lib -- -D warnings`이며, 통과했다.

`log_dispatch` 필드는 서비스 스레드가 살아 있는 동안 `tracing` dispatcher를 그 스레드의 기본값으로 설치한다. stage는 자기 스레드에서 돌기 때문에 테스트 로컬 subscriber로는 보이지 않으며, 이 필드 덕분에 `unowned_activation_warnings_are_rate_limited`가 stage가 warn과 debug로 남긴 내용을 읽을 수 있다.

### 3.6 도달 불가능한 분기를 일반 오류로 바꾸기

`reply_target(..) == None` 분기를 지우고 불변식에 의존하는 대신, PR은 조회를 유지하고 `None`을 `anyhow!("sequence {request_key} lost its owner during execution")`으로 바꾼다. 불변식이 깨지면(예를 들어 서비스가 프레임을 동시에 처리하기 시작하면) 실패는 손으로 쓴 두 번째 drop 경로가 아니라 rate limit과 정리가 적용되는 기존 `report_activation_failure` 경로를 탄다.

## 4. 검증

### 4.1 실패 우선 증거

테스트, 테스트에 필요한 `#[cfg(test)]` builder, 항목 4의 `StageOverrides` 리팩터링만 있는 상태에서 변경 전 동작으로 먼저 실행했다.

| 테스트 | 변경 전 | 변경 후 |
|---|---|---|
| `silent_stage_failure_is_bounded_by_one_stage_timeout` (`stage_timeout` 1초, teardown 대기 100 ms, 상한 1.7초) | 4.006초 | 1.7초 이내. stage 로그 `prepare, run_entry, cancel, release` |
| `remote_runtime_teardown_reaches_every_stage_past_a_silent_one` | `prepare, run_entry, cancel, release, release` | `prepare, run_entry, cancel, release`. 응답 없는 stage 뒤의 stage는 `prepare, cancel, release` |
| `idle_sequences_are_reaped_when_the_cap_is_reached` (상한 2) | 세 번째 prepare 거부: "pipeline stage already holds 2 prepared sequences" | Ack. 유휴 시퀀스 둘 다 release. 회수된 시퀀스의 `RunEntry`는 not prepared로 거부 |
| `recently_active_sequences_are_not_reaped` | 같은 거부 | 유휴 시퀀스만 회수 |
| `unowned_activation_warnings_are_rate_limited` (소유자 없는 activation 20개) | 20개 모두 warn | 20줄, warn은 최대 `2 * HANDSHAKE_WARN_PER_WINDOW`, 실행 0회 |
| `remote_forward_error_fails_one_request_and_the_coordinator_serves_the_next` | release 두 번이 담긴 명령 로그. 모듈이 `run_core_thread_or_abort` 아래에서 돌기 때문에 불일치가 바이너리를 abort시킴(exit 134) | release 한 번 |

rate limit 테스트가 window 두 개 분량의 warn을 허용하는 이유는, budget이 프로세스 전역이라 같은 테스트 바이너리의 다른 stage-drop 지점이 현재 window 일부를 이미 썼을 수 있고, 테스트 도중 rollover가 한 번 일어날 수 있기 때문이다.

### 4.2 로컬 게이트

GB10(spark-102), `--profile test-fast --features cuda,test-utils`, 모든 실행은 `gpu-lock run --tag issue-2762` 아래에서 `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`로 수행.

| Suite | 결과 |
|---|---|
| 위 다섯 테스트와 `prepare_beyond_the_sequence_cap_is_refused`, `final_stage_drops_result_of_unprepared_sequence`, `classifies_handshake_errors` (filter) | 23 passed |
| `server::batch::scheduler::scheduler_pipeline_failure_tests::` | 4 passed |
| `distributed::` | 1561 passed, 4 ignored |
| `distributed_integration` | 20 passed |
| `pipeline_e2e` | 30 passed |
| `server_exit_joins_worker` (실제 qwen3-0.6b-4bit) | 2 passed |
| `disaggregated_serving_e2e --include-ignored` | 2 passed |
| `disaggregated_router_e2e --include-ignored` | 7 passed |
| `pipeline_remote_real_models --include-ignored --skip glm_moe_dsa` | batch 실행에서 9 passed, 1 failed (`glm4`가 load average 43인 호스트에서 처음으로 CUDA 커널을 컴파일하는 동안 30초 `RunEntry` timeout에 걸림). 커널이 캐시된 상태로 단독 재실행 시 1 passed |
| `pipeline_server_remote_real_models --include-ignored --skip glm_moe_dsa` | 7 passed |
| `pipeline_ci_multi_stage_real_models --include-ignored --skip glm_moe_dsa` | 4 passed |
| `cargo fmt --check`. lib+tests(`cuda,test-utils`)와 non-test lib(`cuda`)에 대한 clippy `-D warnings` | clean |

8993404e 이후: `distributed::pipeline::` 396 passed, 스케줄러 실패 테스트 4 passed, 회수 테스트는 각각 5회 중 5회 통과, fmt와 clippy clean.

`glm4` 실패는 이 PR이 바꾸지 않은 `RunEntry` 대기에서 발생했다. 그래도 고정 30초 `stage_timeout`이 부하가 걸린 호스트에서 stage의 첫 forward에는 짧을 수 있다는 근거다.

### 4.3 실행하지 않은 것

#2762의 수동 검증(`--pp-auto 2` 코디네이터, 스트리밍 요청 중 마지막 stage에 `kill -STOP`, 다음 요청이 약 34초 안에 시작되는지 확인)은 실행하지 않았다. 이 검증이 다룰 상한 경로는 실제 TCP 위의 `silent_stage_failure_is_bounded_by_one_stage_timeout`과 스케줄러 timeout 테스트가 다루지만, 34초라는 수치 자체를 실제로 멈춘 프로세스로 측정하지는 않았다. `glm_moe_dsa`(이 장비에서 `glm5-4bit`가 로드되지 않음), Thunderbolt와 RDMA transport, macOS도 실행하지 않았다.

## 5. 남은 공백과 위험

- **stage가 계속 응답하지 않는 동안 새 요청마다 `stage_timeout` 한 번을 여전히 치른다.** 상한은 실패한 요청 하나 기준이다. `prepare_sequence_state`는 `stage_timeout` 전체로 `broadcast_ack`를 쓰므로, 같은 응답 없는 stage를 상대로 한 이후 요청은 prepare에서 30초(응답 없는 stage마다 rollback release 2초 추가)를 기다린 뒤 실패한다. 그동안 코디네이터는 다른 요청을 처리하지 못한다. 기각된 circuit breaker가 다룰 경우가 바로 이것이다. 모든 요청이 모든 stage를 필요로 하므로 서비스를 복구하지는 못한다는 이유로 기각되었지만, 요청당 30초를 즉시 거부로 바꿀 수는 있다.
- **느리지만 정상인 stage가 이제 teardown 경고를 남긴다.** 3.1 참조. 운영자는 "did not acknowledge ... release"를 release 유실로 해석하면 안 된다.
- **34초는 응답 없는 stage가 하나일 때의 수치다.** 응답 없는 stage마다, 실패 감지에 쓰는 `stage_timeout` 한 번 위에 최대 2초의 teardown 대기 두 번(cancel, release)이 더해진다.
- **회수가 살아 있는 요청을 실패시킬 수 있다.** stage가 4096개 상한에 도달한 상태에서 소유자로부터 10분간 아무 메시지도 받지 못한 요청에 한한다. 코디네이터가 아니라 요청 하나로 실패한다.
- **`stage-drop` budget은 프로세스 전역이다.** 한 프로세스의 모든 stage 서비스가 분당 5개 window를 공유하므로, 시끄러운 stage가 다른 stage의 drop 경고를 debug로 밀어낼 수 있다. rollover 요약은 여전히 개수를 보고한다.
- **낡은 테스트 주석.** `idle_sequences_are_reaped_when_the_cap_is_reached`의 doc 주석(`src/distributed/pipeline/remote_service_tests.rs:1408`)은 8993404e가 TTL을 1초로 바꿨는데도 여전히 "here 200 ms, cap 2"라고 적혀 있다. 동작과 assertion은 맞으며, 주석만 1초로 고치면 된다.

## 6. 학습 포인트

### 6.1 대기 시간은 호출자가 응답으로 무엇을 하는지에 맞춘다

request/response transport는 모든 것에 timeout 하나를 쓰기 쉽다. 적절한 timeout은 호출자가 응답으로 무엇을 하는지에 달려 있다. forward에서는 응답이 곧 결과이므로 정당한 forward가 걸릴 수 있는 만큼 기다려야 한다. teardown에서는 프레임 전달이 효과이고 응답은 로그 한 줄일 뿐이므로, 일찍 대기를 멈춰도 잃는 것이 없다. `exchange`를 `exchange_within`으로 나누면서 이 결정이 각 호출 지점의 인자 하나가 되었다.

### 6.2 오류 계약과 상태를 일치시킨다

#2753은 forward의 `Err`가 시퀀스 상태 폐기를 뜻한다고 적었지만, runtime은 매핑을 남겼고 스케줄러의 close가 그것을 근거로 동작했다. 실패 지점에서, 모든 출구가 쓰는 helper 하나로 매핑을 제거하면 계약이 구조적으로 참이 되고, 호출자의 close는 중복 동작이 아니라 no-op이 된다. 예외(아무것도 보내기 전의 출구)는 helper의 doc에 명시해 두어, 나중에 읽는 사람이 이를 "고치지" 않게 한다.

### 6.3 누수된 상태는 일정이 아니라 막히는 지점에서 회수한다

용량 상한과 상한에서의 지연 회수를 결합하면 백그라운드 장치 없이 누수에 상한을 둘 수 있다. 누수된 항목이 새 작업을 거부하게 될 때까지는 아무것도 바뀌지 않고, 그때 TTL을 넘긴 유휴 항목만 해제된다. 마지막 활동 시각은 owner와 같은 struct에 있고 소유권을 검사하는 같은 함수가 갱신하므로, 활동 갱신 없이 소유권만 검사하는 경로는 존재할 수 없다.

```rust
fn touch_sequence(&mut self, request_key: &str) -> bool {
    match self.sequence_owners.get_mut(request_key) {
        Some(sequence) => {
            sequence.last_active = Instant::now();
            true
        }
        None => false,
    }
}
```

### 6.4 release에서 비어 있는 struct로 테스트 설정값을 컴파일에서 제외한다

모든 필드가 `#[cfg(test)]`이고 `#[cfg(not(test))]` no-op 메서드를 가진 struct는 두 빌드 모두에서 호출 지점 하나(`overrides.apply(&mut service)`)를 유지하면서, 배포 바이너리에 설정값이 존재하지 않음을 보장한다. `-D warnings`를 건 non-test clippy 실행이 dead code가 생기지 않았음을 확인하는 수단이다.

## 7. 참고 자료

- Issue #2762: 파이프라인 실패 경로와 stage 상태에 상한 두기
- PR #2753 (issue #2727): stage 실패가 코디네이터가 아니라 요청 하나를 실패시킴. 그 리포트 5절이 여기서 고친 잔여 문제를 나열함
- PR #2746 (issue #2726): 4096 prepared-sequence 상한과 owner 검사. 이 PR이 유휴 회수로 확장함
- `src/distributed/handshake_warn.rs` (#2655, #2692, #2712): 클래스별 경고 budget
- `docs/distributed.md`, "What the secret protects"와 handshake 경고 절
