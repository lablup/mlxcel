# PR #2787: 응답 없는 파이프라인 stage에 즉시 실패하고 stage 로그 정제하기

**날짜**: 2026-10-31 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: main 9f06fef2 위에서 487634ea로 구현 완료. PR 본문의 로컬 게이트는 GB10(spark-102)에서 통과. 작성 시점에 worktree에서 두 번째 로컬 게이트가 실행 중이었고, 호스팅 `2-host logical` 잡은 487634ea에서 새 테스트 하나의 시간 상한 때문에 한 번 실패했음(4.3 참고). `CPU-only link`와 `WebUI installed artifact`는 대기 중
**언어**: Rust, Markdown
**위험도**: 중간 (wire 변경과 public API 변경은 없지만, 코디네이터가 stage가 응답하지 않는다고 판단하는 동안 스스로 요청을 거부하는 새로운 서빙 동작이 생김. 클라이언트에 보이는 문구는 그대로)

## 요약

PR #2787(#2782 해결)은 #2769 리포트가 원격 파이프라인 코디네이터의 실패 경로에 남긴 잔여 문제 세 가지를 고친다. 핵심은 가용성이다. #2769 이후 응답 없는 stage에 대한 실패 요청 하나의 비용은 약 34초로 줄었지만, 그 뒤에 오는 모든 요청은 여전히 prepare에서 30초 `stage_timeout` 전체를 기다린 뒤 실패했고, model worker는 단일 스레드다. 이 PR은 runtime 전체에 걸친 가용성 게이트를 추가한다. `stage_timeout` 전체를 쓰는 exchange가 실패하면 게이트가 열리고, 열려 있는 동안 prepare와 forward는 타입이 있는 `PipelineUnavailable`로 즉시 실패한다. 단, 10초에 최대 한 요청이 2초 teardown 대기로 모든 stage를 probe하고, 모두 응답하면 그 요청은 정상 처리된다. 또한 stage가 보낸 텍스트(error 응답, request id, serde 오류 사유)를 어느 쪽 로그에 남기기 전에 escape하고, "did not acknowledge" teardown 경고를 별도의 `teardown-ack` 경고 예산 아래에 둔다.

## 1. 문제 정의

#2782는 세 항목을 모은 이슈다. 항목 1은 클러스터 secret 없이도 도달 가능하므로 우선순위가 medium이었다. 프레임별 AEAD는 경로상 공격자가 프레임을 버리는 것을 막지 못하며, 버려진 프레임, 네트워크 분할, 멈춘 stage 프로세스는 코디네이터에게 모두 stage timeout으로 보인다.

### 1.1 stage가 응답하지 않는 동안 새 요청마다 stage timeout 전체를 기다렸다 (가용성)

`prepare_sequence_state`는 `broadcast_ack`를, 이는 stage마다 고정 30초 `stage_timeout`으로 `exchange`를 호출한다. 방금 stage가 timeout됐다는 사실을 기억하는 곳이 없어서, 새 요청마다 prepare에서 model worker를 30초 막은 뒤 rollback release 비용까지 냈다. stage가 멈추기 전에 prepare된 시퀀스는 각각 `RunEntry`에서 30초를 기다렸다. #2769는 실패한 요청 하나의 비용에 상한을 두었지만 그다음 백 개의 비용에는 상한을 두지 않았다. #2762는 모든 요청이 모든 stage를 필요로 하므로 circuit breaker가 서비스를 복구하지 못한다는 이유로 이를 기각했다. 그 판단은 여전히 맞지만, breaker는 요청당 30초의 worker 정지를 즉시 거부로 바꿔 주며 #2782가 요구한 것이 바로 그것이다.

### 1.2 stage가 보낸 텍스트가 코디네이터 로그에 그대로 기록됐다 (로그 위조, LOW)

`RemoteStageResponse::Error`의 `message`는 `runtime.rs`의 네 지점에서 그대로 포맷됐고, 그중 둘은 `server_runtime.rs`의 `{err:#}`를 거쳐 `warn!`에 도달했다. stage 쪽에서는 `report_activation_failure`가 upstream이 보낸 `request_key`를 그대로 로그에 남겼다. 인증된 stage는 여기에 개행이나 ANSI escape를 넣어 로그 줄을 위조할 수 있었다. `stage_assignment.rs`에는 `Abort { detail }`용으로 적절한 helper(`loggable_abort_detail`)가 이미 있었지만 그 모듈 안에서만 쓰였다.

### 1.3 느리지만 정상인 stage가 제한 없는 teardown 경고를 만들었다 (로그 소음)

#2769는 teardown 응답 대기를 2초로 제한했다. 긴 `RunEntry`를 아직 처리 중인 정상 stage는 그 대기를 넘기고 cancel이나 release를 나중에 적용하는데도, 코디네이터는 명령과 stage마다 `warn`을 하나씩 남겼고 문구는 프레임이 유실된 것처럼 읽혔다. #2769 리포트 3.1절에 이 trade-off가 기록돼 있다.

## 2. 변경 요약

| 영역 | 파일 | 변경 |
|---|---|---|
| 게이트 상태 기계 | `src/distributed/pipeline/stage_availability.rs` (신규) | `StageAvailability`(장애 시작 시각, 실패한 peer, 마지막 probe, generation), `GateDecision { Pass, Probe, Refuse }`, `ProbeClose`, `ProbeFailure`, 타입이 있는 `PipelineUnavailable` 오류, `STAGE_PROBE_INTERVAL = 10 s`. 단위 테스트 7개 |
| 코디네이터 runtime | `src/distributed/pipeline/runtime.rs` | `availability: Mutex<StageAvailability>`와 `probe_interval` 필드, `#[cfg(test)]` `with_probe_interval`. `exchange`가 `Err`일 때 게이트를 엶. `ensure_stages_available`은 prepare에서 draining 검사 직후, forward에서 `RunEntry` 전에 실행(거부된 forward는 `fail_sequence`를 거침). `broadcast_best_effort`는 응답 없는 peer에 대기 시간 0으로 보냄. `teardown-ack` 예산 아래의 `log_teardown_miss`. 네 개의 `Error` message 지점과 malformed 응답 사유가 `loggable_peer_text`를 거침. forward `Error`는 이제 `remote stage {i} failed the forward: ...`로 표시 |
| 서버 모델 | `src/distributed/pipeline/server_runtime.rs` | `is_unavailable`이 `anyhow` chain에서 `PipelineUnavailable`을 찾음. 이런 거부는 debug로, 나머지는 기존처럼 warn으로 기록. 클라이언트 문구는 변경 없음 |
| peer 텍스트 | `src/distributed/peer_text.rs` (신규) | `MAX_PEER_TEXT_BYTES = 1024`, `truncate_peer_text`, `loggable_peer_text`(문자 경계에서 자른 뒤 `escape_debug`). 단위 테스트 3개 |
| 공용 helper 사용처 | `stage_assignment.rs`, `remote_service.rs` | `abort_detail`과 `loggable_abort_detail`이 `peer_text`에 위임. `request_key`를 기록하는 두 지점을 escape |
| 경고 예산 | `src/distributed/handshake_warn.rs` | `HandshakeWarnClass::TeardownAck`, `CLASS_COUNT` 5에서 6, `Display` 문자열 `teardown-ack` |
| 테스트 stub | `src/distributed/pipeline/stub_stage.rs` | 공유 `AtomicBool` 기반 `set_silent(bool)`, `StubRunBehavior::Error { message }`, `STUB_FORCED_ERROR`, `wait_for_commands(count, timeout)`. `Shutdown`도 공통 경로로 응답하게 되어 silent stub은 shutdown에도 응답하지 않음 |
| 테스트 | `runtime_availability_tests.rs` (신규), `runtime.rs`, `handshake_warn_tests.rs`, `scheduler_pipeline_failure_tests.rs` | 실제 TCP 위의 runtime 테스트 4개, serde escape 테스트 1개, `Display` assertion 1개. 기존 테스트 두 개가 `wait_for_commands`로 stub 로그를 읽음. scheduler 명령 로그 하나에 `probe_state` 추가 |
| 문서 | `docs/distributed.md` | 실패 경로 문단에 게이트, probe 간격, escape, `teardown-ack` 예산을 기술. 이 문서의 한국어판은 없음 |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 14 (신규 모듈 2, 신규 테스트 파일 1) |
| 라인 | +1318 / -73 |
| 신규 테스트 | 15 (게이트 상태 기계 7, peer 텍스트 3, runtime 가용성 4, malformed 응답 사유 1) |
| 커밋 | 2 |

| 커밋 | 목적 |
|---|---|
| 7dade070 | #2782의 세 항목 전부, 테스트, 문서 |
| 487634ea | generation 카운터로 겹치는 게이트 probe 결과를 조정하고, dropped-reply 경고의 serde 오류 사유를 escape |

## 3. 기술적 선택과 그 이유

### 3.1 고정 cooldown 대신 첫 요청에서 probe

**배경.** timeout은 죽은 stage를 뜻할 수도 있고, 정상 stage의 응답 하나가 버려진 것일 수도 있다. 고정 cooldown은 둘을 같게 취급해서 프레임 하나를 잃은 뒤에도 창 전체 동안 모든 요청을 거부한다.

| 선택지 | 평가 |
|---|---|
| `stage_timeout`을 낮추거나 설정 가능하게 | 긴 `RunEntry`의 정당한 대기도 줄어듦. 설정 가능화는 #2782 범위 밖 |
| timeout 후 고정 cooldown | 응답 하나가 버려지면 이후 10초의 트래픽이 거부됨 |
| 백그라운드 health checker | 별도 스레드와 transport 사용이 추가되고, model worker와 경쟁함 |
| **선택: timeout에 열고, 그 뒤 첫 요청이 probe하고, probe 간격 안의 요청은 거부** | 응답 하나가 버려지면 다음 요청이 probe 왕복 한 번만 부담(`a_single_dropped_reply_does_not_refuse_the_next_request`). 죽은 stage는 요청당 30초 대신 10초당 최대 2초 probe 한 번의 비용 |

probe는 `RemoteStageCommand::ProbeState`이며 `teardown_ack_timeout`(2초, 또는 그보다 작은 `stage_timeout`)으로 모든 stage에 순서대로 보낸다. `State`로 응답하지 않는 첫 stage에서 멈추고, 그 stage가 `failed_peer`로 기록된다. probe한 요청은 probe 비용을 내고, 성공하면 같은 호출 안에서 정상 처리된다.

### 3.2 `stage_timeout` 전체를 쓰는 경로만 게이트를 연다

게이트를 여는 것은 `exchange`뿐이고, 짧은 대기(teardown, probe)를 쓰는 `exchange_within`은 열지 않는다. teardown 응답을 놓친 것은 이미 응답 없다고 알려진 stage에 대해 새로운 정보가 아니며, 느리지만 정상인 stage는 이를 흔히 놓친다(항목 3). `Error` 응답은 응답이므로 게이트를 열지 않는다. 실제로 "`stage_timeout` 전체 실패"는 `exchange`의 모든 `Err`를 뜻하므로, transport 전송 오류(예: 연결 거부)도 게이트를 연다. 도달 불가능한 stage라는 의도와는 맞는다.

### 3.3 `anyhow` 안의 타입 오류를 debug로 기록

`PipelineUnavailable { stage, since }`는 `thiserror` 타입이며 `anyhow::Error`로 변환된다. `PipelineServerModel`은 `err.chain().any(|c| c.downcast_ref::<PipelineUnavailable>().is_some())`로 이를 찾아 debug로 기록한다. 장애 중 거부된 요청마다 warn을 남기면 항목 3이 다른 곳에서 없앤 범람이 그대로 생기기 때문이다. 운영자에게 필요한 신호는 게이트 자신의 전이다. 열릴 때 `warn` 한 줄(stage, 대기 시간, probe 간격), 닫힐 때 `info` 한 줄(장애 길이). 클라이언트는 여전히 `PIPELINE_PREPARE_REFUSED`나 `PIPELINE_FORWARD_FAILED`만 받으므로 주소나 시간 정보가 새지 않는다.

### 3.4 거부된 forward에서도 #2753의 오류 계약 유지

게이트가 거부한 forward는 매핑 이후의 다른 실패와 똑같이 `fail_sequence`를 실행한다. 따라서 `Err`는 여전히 그 시퀀스에 대한 코디네이터 상태가 버려졌음을 뜻하고, 스케줄러의 이후 close는 아무것도 보내지 않는다. 응답 없는 peer에는 `Duration::ZERO`로 teardown을 보낸다. `exchange_within`은 프레임을 쓰고, deadline이 이미 지났음을 확인한 뒤 반환하며, 이 누락은 debug로 기록된다. 정상 stage는 2초 대기를 유지한다. 거부된 prepare는 매핑이 생기기 전에 일어나므로 정리할 것이 없다.

### 3.5 I/O 동안 게이트 mutex를 잡지 않고, 겹치는 probe를 조정

`decide`는 probe를 보내기 전에 lock 안에서 `last_probe`를 기록하므로, probe 중에 도착한 요청은 함께 probe하지 않고 거부된다. probe exchange는 lock을 놓은 채 실행되고(다른 exchange처럼 `io_runtime` mutex로 직렬화됨), 결과는 다시 lock 안에서 반영된다. 그래도 probe 하나가 probe 간격보다 오래 걸리면 probe가 겹칠 수 있다. 두 번째 커밋은 새 timeout과 실패한 probe마다 증가하고 close에서는 증가하지 않는 `generation` 카운터를 추가했다.

| 반영 시점의 probe 결과 | 결과 |
|---|---|
| 성공, 같은 generation, 게이트 열림 | 게이트를 닫고(`ProbeClose::Closed`) info 줄 기록 |
| 성공, 같은 generation, 형제 probe가 이미 닫음 | 요청 처리, 기록 없음(`AlreadyClosed`) |
| 성공, 이전 generation(새 timeout이 장애를 대체함) | 요청 거부, 게이트 그대로(`Superseded`) |
| 실패, 같은 generation, 게이트 열림 | stage를 기록하고 generation 증가 |
| 실패, 같은 generation, 형제 probe가 닫음 | 새 장애로 다시 열고 open 줄을 다시 기록(`reopened`). 로그가 닫힌 게이트를 보여 주면서 실제로는 요청을 거부하는 일이 없도록 함 |
| 실패, 이전 generation | 더 새로운 상태는 건드리지 않고 이 요청만 거부 |

상태 기계는 순수 함수(`Instant`를 인자로 받음)이므로, 이 경우들은 소켓이나 sleep 없이 단위 테스트가 된다. 프로덕션에서는 model worker가 단일 스레드이고 probe는 stage당 2초로 제한되므로, 겹치려면 여러 stage에 걸친 느린 probe가 필요하다. 이 로직은 그런 일이 생겨도 로그와 게이트 상태가 어긋나지 않게 하려는 것이다.

### 3.6 모든 peer 텍스트에 helper 하나

`loggable_peer_text`는 문자 경계에서 1024바이트로 자른 뒤 `escape_debug`를 적용한다. 먼저 자르므로 상한은 peer가 보낸 텍스트에 적용되고, escape된 결과는 더 길어질 수 있지만 한 줄이다. `stage_assignment.rs`는 `abort_detail`(보내는 `Abort`에는 상한만 적용된 원문이 필요함)과 `loggable_abort_detail`을 유지하되 둘 다 위임한다. 리뷰 과정에서 이슈에 없던 지점 하나가 발견됐다. serde_json은 payload의 알 수 없는 variant나 field 이름을 그대로 인용하므로, dropped-reply 경고는 이를 escape하는 `malformed_reply_reason`을 거치게 됐다.

### 3.7 `stage-drop` 재사용 대신 여섯 번째 경고 클래스

`stage-drop`은 인증된 upstream이 보낸 프레임에 대한 stage 쪽 예산이고, teardown 누락은 코디네이터 쪽이며 부하 상황에서 예상되는 일이다. 창을 공유하면 바쁜 코디네이터가 stage-drop 경고를 debug로 밀어낼 수 있다. `TeardownAck`는 분당 5개짜리 별도 창을 갖고, 바뀐 문구는 프레임이 전송됐다고 명시해서 운영자가 release 유실로 읽지 않게 한다.

## 4. 검증

### 4.1 수정 전 실패 확인

테스트 지원 코드(stub `set_silent`, `StubRunBehavior::Error { message }`, `with_probe_interval`)만 넣은 9f06fef2에서 35개 통과, 4개 실패.

| 테스트 | 이전 | 이후 |
|---|---|---|
| `silent_pipeline_fails_new_prepares_fast_until_a_probe_succeeds` (stage timeout 1초, teardown 100ms, probe 간격 300ms) | prepare B: `remote stage ... timed out after 1s`, `PipelineUnavailable` 아님 | B는 probe 한 번 뒤 500ms 안에 거부, C는 명령 전송 없이 50ms 안에 거부, stage가 다시 응답한 뒤 D 처리, 로그 끝이 `probe_state, prepare` |
| `a_single_dropped_reply_does_not_refuse_the_next_request` | 명령 로그 assertion 실패 | `prepare, run_entry, cancel, release, probe_state, prepare, run_entry`, B의 logits 반환 |
| `stage_error_text_is_escaped_before_logging` | `"boom\nINFO forged\u{1b}[31m"`가 오류에 그대로 포함 | `{err:#}`에 `\n`과 `\u{1b}` 없음, `boom\\n` 포함 |
| `teardown_ack_warnings_are_rate_limited` (실패 forward 20회, teardown 10ms) | 누락 40개 모두 warn | 40줄, warn은 최대 `2 * HANDSHAKE_WARN_PER_WINDOW`, 각 warn에 `the frame was sent` 포함 |

### 4.2 487634ea의 로컬 게이트

GB10(spark-102), `gpu-lock` 아래 `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1`, `MLXCEL_SDPA_DETERMINISTIC=1`. fmt와 clippy(`--lib --tests -D warnings`) 통과.

| 스위트 | 결과 |
|---|---|
| `distributed::` lib 테스트 | 1576 통과, 4 ignored |
| `scheduler_pipeline_failure_tests` | 4 통과 |
| `distributed_integration` | 20 통과 |
| `pipeline_e2e` | 30 통과 |
| `pipeline_remote_real_models` (`--skip glm_moe_dsa`) | 10 통과 |
| `pipeline_server_remote_real_models` (`--skip glm_moe_dsa`) | 7 통과 |
| `pipeline_ci_multi_stage_real_models` | 4 통과 |
| `disaggregated_serving_e2e` | 2 통과 |
| `disaggregated_router_e2e` | 8 통과 |
| `server_exit_joins_worker` | 2 통과 |

### 4.3 487634ea의 호스팅 `2-host logical` 실패

이 잡(run 41161926566)은 529개 중 한 테스트에서 실패했다. `silent_pipeline_fails_new_prepares_fast_until_a_probe_succeeds`의 `runtime_availability_tests.rs:139`, "prepare C took 101.71692ms; a refusal inside the probe interval sends nothing". 그 앞의 assertion(A 정리, B가 probe 한 번 뒤 500ms 안에 거부)은 통과했고, 그 뒤는 실행되지 않았다. 거부 경로는 게이트 mutex를 잡고 오류를 만드는 것뿐이므로 100ms는 게이트 자체의 비용이 아니다.

추정 원인(검증하지 않음): 이 workflow는 `RUST_BACKTRACE=short`를 설정하므로 `PipelineUnavailable`을 `anyhow`로 변환할 때 backtrace를 캡처하고, std의 backtrace 캡처는 프로세스 전역 lock을 잡는다. 약 100ms 전(03:05:07.1547 대 07.2562)에 동시에 실행 중이던 다른 테스트의 백그라운드 tokio 태스크가 `runtime_correlation_tests.rs:105`에서 panic했고(닫힌 채널로의 `MockRouter` 전송), symbol을 해석한 backtrace를 출력했다. 출력하는 동안 그 lock이 잡혀 있다. 원인이 무엇이든 50ms wall-clock 상한은 병렬로 도는 호스팅 실행에는 빠듯하다. #2782는 이 잡이 계속 green이어야 한다고 요구하므로, 머지 전에 상한(또는 해당 테스트의 backtrace 설정)을 손봐야 한다.

### 4.4 실행하지 않은 것

stage에 `kill -STOP`을 거는 `--pp-auto 2` 수동 확인은 실행하지 않았다. probe와 거부 동작은 실제 TCP 위의 stub 테스트로 확인했지만, 실제로 멈춘 프로세스를 상대로 지연 시간을 측정하지는 않았다. `glm_moe_dsa`(`glm5-4bit`가 이 장비에서 로드되지 않음), Thunderbolt와 RDMA transport, macOS도 실행하지 않았다.

## 5. 남은 한계와 위험

- **느리지만 살아 있는 stage가 이제 죽은 것으로 취급된다.** stage가 느려서 `RunEntry`가 timeout되면(#2769 리포트는 부하가 큰 호스트에서 CUDA kernel을 처음 컴파일하던 `glm4`의 timeout을 기록했다) 게이트가 열리고, 다음 probe는 그 stage에 2초만 준다. 버려진 forward를 아직 처리 중인 stage는 이를 놓치므로, 이후 probe가 통과할 때까지 요청이 거부된다. 이 PR 전에는 이 요청들이 기다렸다가 성공했을 수도 있다. 근본 문제는 고정 30초 `stage_timeout`이며, 설정 가능화는 범위 밖이다.
- **최대 probe 간격 한 번만큼의 복구 지연.** stage가 돌아온 뒤에도 다음 probe 시점까지, 즉 마지막 probe 후 최대 10초 동안 요청이 거부된다.
- **원인 stage 지목.** `RunEntry` timeout은 첫 probe가 실제 응답 없는 stage를 지목하기 전까지 entry stage를 이름으로 남기고, probe는 순서상 처음 실패한 stage만 지목한다. stage별 지목은 범위 밖이다.
- **호스팅 CI 시간 상한.** 4.3 참고.
- **`teardown-ack` 예산은 프로세스 전역이다.** 한 프로세스 안의 여러 코디네이터가 분당 5개 창을 공유한다. rollover 요약은 억제된 개수를 계속 보고한다.
- **게이트는 `exchange`의 모든 오류에 열린다.** 시간 초과뿐 아니라 전송 실패도 포함한다. 의도에는 맞지만 문서는 "stage timeout에 도달하면"으로 표현한다.

## 6. 학습 포인트

### 6.1 순수 상태 기계와 그 주변 I/O를 분리하기

`StageAvailability`는 `now`를 인자로 받아 결정을 반환하고, `RemotePipelineRuntime`이 lock과 exchange를 맡는다. 이 분리 덕분에 겹치는 probe 경우(3.5)를 단위 테스트로 나열하기 쉬웠고, 두 번째 커밋이 transport 테스트를 건드리지 않고 bookkeeping의 race를 고칠 수 있었다.

### 6.2 상태가 바뀐 뒤 도착하는 결과에는 generation 카운터

probe 결과는 probe를 시작한 시점의 장애를 설명한다. 결정에 generation을 붙이고 결과를 반영할 때 확인하는 것은 ABA에 안전한 compare-and-swap과 같은 패턴이다. 오래된 generation의 성공이 더 새로운 장애를 닫으면 안 되고, 오래된 generation의 실패가 더 새로운 close를 덮어쓰면 안 된다. close에서 generation을 올리지 않는 것이 형제 probe의 성공을 거부 대신 처리로 이끄는 세부 사항이다.

### 6.3 신뢰 경계에서 escape하기, 라이브러리 오류 메시지 포함

명백한 지점은 stage 자신의 `message` 필드였다. 덜 명백한 지점은 payload의 공격자 제어 키를 인용하는 `serde_json::Error`의 `Display`였다. 신뢰할 수 없는 입력으로 만들어진 라이브러리 오류도 peer 텍스트다.

### 6.4 panic하는 이웃 테스트 옆에서의 wall-clock assertion

"50ms 미만"을 단언하는 테스트는 테스트 대상 코드가 아니라 프로세스 전체를 측정한다. backtrace 캡처, panic 출력, 스케줄러 부하가 공유 runner에서는 모두 그 창 안에 들어온다. "아무것도 보내지 않음"과 "timeout까지 기다림"을 구분하는 상한은 10배 느슨하게 잡아도 여전히 구분되며, 아니면 명령 로그만으로 단언할 수도 있다.

## 7. 참고

- Issue #2782: 응답 없는 stage에 즉시 실패하고 stage 로그 정제
- PR #2769(이슈 #2762): 실패한 요청 하나를 `stage_timeout` 한 번과 2초 teardown 대기 두 번으로 제한. 리포트 5절에 항목 1과 항목 3이 기록됨
- PR #2753(이슈 #2727): stage 실패가 요청 하나만 실패시키며, forward의 `Err`는 코디네이터 상태가 버려졌음을 뜻함
- `src/distributed/handshake_warn.rs` (#2655, #2692, #2712, #2762): 클래스별 경고 예산
- `docs/distributed.md`, "What the secret protects"의 실패 경로 문단
