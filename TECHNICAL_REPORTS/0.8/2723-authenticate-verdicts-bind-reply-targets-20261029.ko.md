# PR #2723: 핸드셰이크 verdict 인증과 응답 대상 바인딩

**날짜**: 2026-10-29 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: 71a8c6e0에서 구현 완료. 구현 리뷰는 CRITICAL, HIGH 이슈를 찾지 못함. 보안 및 성능 리뷰가 HIGH 이슈 하나(두 번째 secret 보유자가 코디네이터의 다음 시퀀스 id에 KV 상태를 미리 심을 수 있음)를 찾았고 71a8c6e0에서 고침. 71a8c6e0에서 GB10(spark-102) 로컬 게이트 통과. 작성 시점에 사전 머지 테스트 실행이 진행 중이었고 호스팅 CI는 일부 대기 중이었음. 호스팅 `Pipeline Parallel CI / 2-host logical` job은 다른 브랜치에서도 나타나는 프로세스 종료 시 abort로 실패함(4.3절)
**언어**: Rust, Markdown
**위험도**: 중간 (wire 호환성 깨짐: `PROTOCOL_VERSION` 4에서 5로, 클러스터의 모든 노드를 함께 업그레이드해야 함. stage가 이전에는 받아들이던 identity의 시퀀스 명령을 거부함. `PipelineModelRuntime::prepare_sequence_state`의 시그니처가 바뀜)

## 요약

PR #2723(#2714 해결)은 #2705와 #2708이 인증된 클러스터 전송 계층에 남긴 LOW 잔여 문제 세 가지와, #2720과 #2721이 main에 들어온 뒤 오케스트레이터가 추가한 두 항목을 처리한다. 인증 후 핸드셰이크 verdict 바이트에 이제 전체 transcript와 verdict에 대한 HMAC(`tag_v`)이 붙는다. 그래서 경로상의 공격자가 정상 서버를 포화 상태로 보이게 하거나 거부된 연결을 수락된 것으로 보이게 할 수 없다. 이로써 핸드셰이크는 `PROTOCOL_VERSION` 5가 된다. 원격 파이프라인 stage는 모든 명령이 덮어쓰던 단일 `coordinator_identity`를 `sequence_owners`로 바꾼다. 이는 request id마다 그 시퀀스의 `PrepareSequence`를 보낸 인증된 identity를 기록하는 맵이다. 그 owner만 시퀀스를 실행, 취소, 해제, 재준비할 수 있고, 마지막 stage의 `RunResult`와 모든 activation `Error`는 owner에게만 간다. 라우터의 결과 송신자 포트 검사는 바인딩하지 않은 채로 두고, 그 이유를 문서화했다.

소유권이 생기면 secret 보유자가 순차적인 `pp-seq-N` id를 먼저 점유할 수 있으므로, 이 PR은 거부된 `PrepareSequence`가 코디네이터를 중단시키지 않고 요청 하나만 실패시키게 한다(항목 a). 오류는 `RemotePipelineRuntime::prepare_sequence_state`에서 새 `LanguageModel::try_prepare_sequence_state`를 거쳐 `Engine::open`과 스케줄러로 전달된다. 항목 b는 `handle_activation`의 모든 실패 출구를 `report_activation_failure`로 보내서, activation을 거부하거나 실패한 non-entry stage가 코디네이터에게 즉시 알리고 코디네이터가 `stage_timeout`을 다 기다리지 않게 한다. 리뷰에서 추가된 수정은 한 id의 첫 `PrepareSequence`가 소유되지 않은 명령이 그 id 아래 남긴 캐시를 버리게 한다.

다섯 항목 중 어느 것도 클러스터 secret이 없는 자가 프레임을 읽거나 주입하게 하지 않는다. 항목 1은 경로상의 공격자가, 항목 2, 3, a와 리뷰 수정은 두 번째 secret 보유자가 있어야 한다.

## 1. 문제 정의

#2714는 #2705 보고서(3.7절과 "Known limitations")와 #2708 보고서에 열린 채로 기록된 잔여 문제를 모았고, main 752db4f9에서 다시 확인했다.

### 1.1 핸드셰이크 verdict가 인증 없는 1바이트였다 (LOW, 경로상 공격자)

클라이언트의 authenticator가 검증된 뒤 서버는 인증되지 않은 1바이트를 썼다: `VERDICT_ACCEPT` 0x01, `VERDICT_AT_CAPACITY` 0x02, `VERDICT_REJECT` 0x00. 클라이언트는 이 값으로 바로 분기했다. 경로상의 공격자는 원래도 연결을 끊을 수 있지만, 이 바이트를 바꾸면 실패가 어떻게 보고될지 고를 수 있었다. 0x01을 0x02로 바꾸면 정상 서버가 포화 상태로 보이고(클라이언트는 `ServerAtCapacity`를 받고 backoff), 0x02나 0x00을 0x01로 바꾸면 클라이언트가 거부된 연결을 살아 있는 것으로 여기고 첫 송신에서 일반 I/O 오류로 실패했다. 서버가 거부한 연결에서는 키가 도출되지 않으므로 프레임 기밀성과 무결성은 영향을 받지 않았다.

### 1.2 `RunResult` 대상이 마지막 명령을 따라갔다 (LOW, secret 보유자)

`RemoteStageService::coordinator_identity`는 받아들인 모든 명령의 인증된 출처로 설정되었고, 마지막 stage는 `RunResult`를 그곳으로 보냈다. 코디네이터가 unspecified IP로 설정된 경우(직접 쓴 설정, 또는 `0.0.0.0:<port>`를 쓰는 `--pp-auto` TOML을 복사한 경우) `PeerMatcher::matches`는 그 포트의 어떤 인증된 identity든 받아들인다. 다른 호스트에서 그 포트로 listen하는 secret 보유자는 `ProbeState` 한 번만 보내면 그 뒤로 진행 중인 모든 시퀀스의 logits를 받을 수 있었다. zero-config peer는 `localize_cluster`에서 구체적인 주소를 받으므로 영향이 없었다.

### 1.3 라우터 결과 송신자 포트는 증명되지 않는다 (LOW, secret 보유자)

`router_front.rs`의 `accepts_result_from`은 송신자의 정규화된 IP와 포트를 검사하지만, 핸드셰이크는 IP만 증명한다(`bind_advertised_identity`). #2714는 포트 바인딩을 검토한 뒤 하지 않기로 했고(3.4절), 산출물은 문서이다.

### 1.4 거부된 prepare가 코디네이터를 중단시켰다 (항목 a, 구현 중 추가)

stage가 다른 identity가 소유한 id의 `PrepareSequence`를 거부하게 되면, `RemotePipelineRuntime::prepare_sequence_state`는 `.expect("failed to prepare remote pipeline sequence state")`에 걸려 모델 워커를 중단시킨다. draining 검사도 같은 효과의 `assert!`였다. #2714는 처음에 이를 가용성만의 트레이드오프로 받아들였지만, request id가 순차적(`pp-seq-N`)이므로 secret 보유자가 원할 때 일으킬 수 있다. main에서 `catch_unwind` 프로브는 `runtime.rs:605`에서 panic했다. `request_id_for`도 매핑이 없으면 panic했다.

### 1.5 non-entry activation 거부가 코디네이터 타임아웃으로 끝났다 (항목 b, 구현 중 추가)

#2720은 `handle_activation`의 실행 실패와 downstream 송신 실패를 `report_activation_failure`로 보냈지만, 다른 출구는 여전히 `?`로 빠져나가 서비스 루프의 `dropping frame` 경고로 끝났다. #2721이 추가한 텐서 한도 거부, downstream peer 없음, `sequence_length`, request id 파싱, 직렬화가 여기에 해당한다. 코디네이터는 stage가 이미 포기한 교환을 `stage_timeout` 동안 기다렸다. #2721 보고서는 이를 첫 번째 남은 공백으로 적었다.

## 2. 변경 요약

| 항목 | 영역 | 변경 |
|---|---|---|
| 1 | `cluster_handshake.rs` | `VERDICT_LABEL = "mlxcel-cluster-auth-v1/verdict"`, `VERDICT_RECORD_LEN = 1 + TAG_LEN`(33바이트). 하나의 `write_verdict` 헬퍼가 reject, at-capacity, accept 모두에 `verdict \|\| tag_v`를 보내며, `tag_v = sign(VERDICT_LABEL, client_hello \|\| server_hello \|\| tag_s \|\| tag_c \|\| verdict)`이다. 서버는 검증 성공 여부와 관계없이 transcript에 `tag_c`를 붙이므로 reject verdict도 다른 verdict와 같은 바이트 위에서 서명된다. 클라이언트는 정확히 33바이트를 읽고 `match` 전에 `tag_v`를 검증하며, 실패하면 키를 도출하지 않고 `HandshakeRejected("server's handshake verdict does not verify ...")`를 반환한다. `PROTOCOL_VERSION = 5`. 버전 이력, wire 다이어그램, 모듈 문서를 갱신했다. |
| 2 | `pipeline/remote_service.rs` | `coordinator_identity: Option<String>`이 `sequence_owners: HashMap<String, String>`이 된다. `handle_command`는 먼저 `sequence_of(&command)`(`PrepareSequence`, `RunEntry`, `CancelRequest`, `ReleaseSequence`가 지정한 id, `BeginDrain`, `Shutdown`, `ProbeState`는 `None`)와 `owned_by_other(id, from)`을 확인한다. 해당하면 `Error("sequence <id> was prepared by another coordinator identity")`로 답하고 아무것도 바꾸지 않는다. `PrepareSequence`가 owner를 기록한다. `finish_request`가 owner를 지우고, `Shutdown`이 맵을 비운다. `reply_target`은 `Option<String>`을 반환한다: owner 아니면 없음이며, 설정된 주소로의 fallback은 없다. |
| 2 (리뷰) | `pipeline/remote_service.rs` | 한 id의 첫 `PrepareSequence`(아직 owner가 기록되지 않음)에서, owner를 기록하기 전에 그 id 아래 이미 있는 캐시를 지우고 해제한다. owner의 재준비는 캐시를 유지한다. |
| 2 | `pipeline/remote_service.rs` | 마지막 stage: owner가 없으면 logits를 경고와 함께 버리고 요청을 종료한다. |
| 3 | `router_front.rs`, `docs/distributed.md` | 문서만: 송신자 포트가 주장으로 남는 이유. |
| a | `pipeline/runtime.rs` | `PipelineModelRuntime::prepare_sequence_state`가 `Result<()>`를 반환한다. 원격 runtime은 draining 중이면 `bail!`하고, `broadcast_ack(PrepareSequence)`가 실패하면 id 매핑을 지우고, best effort로 `broadcast_ack(ReleaseSequence)`를 보내고, 컨텍스트를 붙인 오류를 반환한다. `request_id_for`는 공용 `request_id_of`를 통해 `Result`를 반환하고 forward 경로가 이를 전파한다. `request_id_for_sequence`가 `pp-seq-N`을 한 곳에서 만든다. |
| a | `mlxcel-core/src/generate.rs`, `language_model_ref.rs`, `loaded_model.rs` | 새 `LanguageModel::try_prepare_sequence_state(&self, SequenceId) -> Result<(), String>`. 기본 구현은 실패하지 않는 hook을 호출하고 `Ok`를 반환한다. `&M`과 `LoadedModel`이 이를 전달한다. |
| a | `pipeline/server_runtime.rs` | `PipelineServerModel`이 실패 가능한 hook을 override해서 runtime의 오류를 반환한다. 실패하지 않는 `prepare_sequence_state`는 오류 경로가 없는 호출자를 위해 panic을 유지한다. |
| a | `mlxcel-core/src/engine/mod.rs` | `Engine::open`이 `try_prepare_sequence_state`를 호출한다. `Err`이면 방금 할당한 pool 엔트리를 닫고 `EngineError::Open("prepare sequence state for <id>: ...")`를 반환한다. 스케줄러의 `allocate_sequence_state` 호출자(admission, decode re-prefill, queued adoption, prompt cache, handoff)는 이미 `Err`에서 자기 요청만 실패시킨다. |
| b | `pipeline/remote_service.rs` | `handle_activation`은 얇은 래퍼가 된다. 본문은 모든 실패에 `?`를 쓰는 `run_activation`으로 옮겨지고, 어떤 `Err`이든 `report_activation_failure`로 간다. 이 함수는 `finish_request`가 owner를 지우기 전에 owner를 읽고, activation의 `reply_nonce`와 함께 `Error`를 owner에게 보낸다. nonce나 owner가 없으면 로그를 남기고 버린다. `send_run_error`는 여기에 합쳐졌다. |
| 문서 | `docs/distributed.md` | "What the secret protects"가 시퀀스 소유권, 첫 prepare의 상태 폐기, fallback 없음 규칙, 실패 보고를 설명한다. 핸드셰이크 절과 계층 표가 인증된 verdict와 버전 5를 설명한다. |
| 테스트 | 테스트 파일 6개 | 새 테스트 13개. 기존 테스트는 이제 33바이트 verdict 레코드를 읽고, activation을 보내기 전에 시퀀스를 준비하며, 예전에 침묵을 허용하던 곳에서 `Error` 응답을 요구한다(4.1절). 원격 stage 테스트 하네스에 설정 가능한 코디네이터 주소, 코디네이터 포트의 rogue transport, 캐시 해제 카운터가 생겼다. `tests/pipeline_remote_*.rs`와 `runtime_correlation_tests.rs`의 호출 지점은 새 `Result`를 unwrap하고, drain 테스트는 `catch_unwind` 대신 `is_err()`를 확인한다. |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 16 (소스 9, 테스트 6, 문서 1) |
| 추가 / 삭제 줄 | main 대비 +1324 / -244 |
| 추가 테스트 | 13 (`cluster_handshake_tests.rs` 5, `remote_service_tests.rs` 5, `runtime.rs` 1, `server_runtime.rs` 1, `engine/tests.rs` 1) |
| 커밋 | 666a851f (항목 1, 2, 3, a, b), 71a8c6e0 (보안 리뷰의 첫 prepare 상태 폐기) |

## 3. 기술적 선택과 그 이유

### 3.1 verdict MAC은 첫 보호 프레임이 아니라 핸드셰이크 안에서

**컨텍스트.** verdict에는 무결성이 필요하고, 핸드셰이크에는 이미 label 기반 도메인 분리를 하는 키 기반 primitive(`ClusterAuth::sign`/`verify`)가 있다.

| 선택지 | 평가 |
|---|---|
| verdict를 첫 AEAD 보호 프레임으로 전달 | 모든 호출자(`tcp_transport`, `connection_pool`, `stage_assignment`)가 바뀌고, `HmacSha256` suite는 AEAD가 아니어서 어차피 별도 경로가 필요함 |
| pre-hello `REJECT_MAGIC` 레코드도 인증 | 이 레코드는 클라이언트가 아무것도 증명하기 전에 보내지며, hello 바이트를 바꾸면 이미 똑같이 종결적인 실패가 남 |
| **선택: verdict 바이트 뒤에 `tag_v`를 붙임** | `cluster_handshake.rs` 안에서 끝나고, `sign`/`verify`와 `tcp_transport.rs`, `stage_assignment.rs`, `handshake_warn.rs`의 기존 `HandshakeRejected` 분류를 재사용함 |

서명되는 메시지는 `tag_c`까지의 전체 transcript 뒤에 verdict 바이트를 붙인 것이므로, 태그는 결정을 이 교환(양쪽 nonce)과 클라이언트 자신의 authenticator에 묶는다. 보안 리뷰는 명백한 재사용 공격을 확인했다. 인증에 실패한 클라이언트는 서명된 reject verdict를 받지만, 서명된 verdict 메시지는 항상 자기 verdict 바이트로 끝나고 label은 메시지와 분리되어 있으므로 reject 태그가 accept나 at-capacity로 검증될 수는 없다. 서버는 클라이언트가 보낸 `tag_c`가 무엇이든 그 위에서 reject verdict에 서명한다. 그래서 transcript를 검증 분기 전에 확장한다.

**트레이드오프.** 레코드가 1바이트에서 33바이트로 늘어서 버전 4와 버전 5 노드는 통신할 수 없다. 버전 필드 덕분에 혼란스러운 read 오류 대신 명시적인 `speaks cluster handshake version 4, this build expects 5` 메시지로 실패한다. 인증 핸드셰이크는 v0.8.0-beta.1 이후에 들어왔으므로 영향받는 릴리스는 없다.

### 3.2 소유권은 시퀀스 단위이고 `PrepareSequence`만 설정한다

**컨텍스트.** 이전 설계는 stage 전체에 응답 대상이 하나였고, 받아들인 어떤 명령이든 이를 옮길 수 있었다.

owner는 `PrepareSequence`의 인증된 `from`이다(payload의 `reply_addr`가 아니라 핸드셰이크가 TCP 출처 IP에 묶은 identity). 바인딩을 만드는 것은 `PrepareSequence`뿐이다. 소유되지 않은 id에 대한 `RunEntry`는 여전히 실행되고 바인딩을 만들지 않는다. 코디네이터의 모든 `RunEntry`는 `prepare_sequence_state` 뒤에 오기 때문이다. `RunEntry`가 바인딩을 만들게 하면 rogue가 아무 명령으로나 id를 점유할 수 있다. `ProbeState`와 `BeginDrain`은 소유권이 필요 없고 건드리지도 않는다. `Shutdown`은 시퀀스 단위가 아니어서 코디네이터 matcher를 통과하는 모든 identity에게 열려 있다. 이는 기존 신뢰 모델과 같다: 코디네이터 포트의 secret 보유자는 이미 stage를 종료시킬 수 있다.

`reply_target`에는 fallback이 없다. 설정된 코디네이터 주소는 어떤 호스트도 가리키지 않는 `0.0.0.0:<port>`일 수 있고, 그곳으로 보내면 원래 문제가 다시 생긴다. 그 stage에서 아무도 준비하지 않은 시퀀스의 결과나 오류에는 정당한 수신자가 없으므로 로그를 남기고 버린다.

### 3.3 첫 prepare는 소유되지 않은 상태를 버린다 (리뷰 수정)

보안 리뷰는 소유권이 이미 owner가 있는 id만 보호한다는 점을 찾았다. 코디네이터 matcher를 통과하는 rogue는 코디네이터가 `pp-seq-N+1`을 준비하기 전에 `RunEntry { pp-seq-N+1 }`을 보낼 수 있었다. stage는 그 id 아래 캐시를 만들고, 코디네이터의 이후 `PrepareSequence`는 `caches_by_request.entry(id).or_insert_with(..)`를 통해 그 캐시를 재사용했다. rogue의 토큰이 다른 클라이언트 요청의 숨겨진 KV prefix가 되므로, 가용성만이 아닌 무결성 문제다. 71a8c6e0은 첫 `PrepareSequence`가 owner를 기록하기 전에 그 id 아래 있는 캐시를 해제하게 한다. owner 자신의 재준비(이미 owner가 기록됨)는 캐시를 유지하므로 기존 재준비 의미는 바뀌지 않는다.

### 3.4 라우터 결과 송신자 포트는 주장으로 남는다

포트를 증명하려면 프로세스별 키 자료가 필요하다(예: 두 hello에 서명하고 라우터가 pinning하는 프로세스별 Ed25519 키). 모든 멤버가 같은 secret을 공유하므로 현재 핸드셰이크로는 같은 IP의 두 secret 보유자를 구별할 수 없기 때문이다. 라우터가 연 연결에서만 결과를 받으려면 pool에 있는 outbound 연결에서 읽어야 하는데, `TcpTransport`는 그렇게 하지 않는다. 얻는 것도 작다. demux는 설정된 어떤 peer든 어떤 request id에 대해서도 받아들이므로, 남는 상대는 설정된 peer와 IP를 공유하는 설정되지 않은 secret 보유자이고, 위협 모델은 이미 이를 멤버로 취급한다. PR은 이 세 가지를 `router_front.rs` 모듈 문서와 `docs/distributed.md`에 적었다.

### 3.5 실패하지 않는 hook 옆에 실패 가능한 hook을 둔다

**컨텍스트.** `LanguageModel::prepare_sequence_state`는 실패하지 않는 메서드이고, 많은 모델 래퍼가 구현하거나 전달한다(`src/vision/`의 VLM 래퍼는 텍스트 모델로 전달함).

| 선택지 | 평가 |
|---|---|
| `prepare_sequence_state`가 `Result`를 반환하게 함 | 한 모델만 낼 수 있는 실패 때문에 모든 구현과 전달자를 고쳐야 함 |
| panic 후 스케줄러에서 catch | 모델 워커를 가로지르는 unwinding은 지원되는 제어 경로가 아님 |
| **선택: 기본 구현이 있는 `try_prepare_sequence_state` 추가** | override 하나(`PipelineServerModel`), 전달자 둘(`&M`, `LoadedModel`), 호출자 변경 하나(`Engine::open`) |

`Engine::open`은 스케줄러가 시퀀스 상태를 할당하는 유일한 지점이므로, 실패 가능한 hook을 여기로 보내면 admission, preemption 후 decode re-prefill, queued adoption, prompt cache 경로, disaggregated handoff가 모두 덮인다. 이들은 모두 `allocate_sequence_state`의 `Err`를 이미 자기 요청만 실패시키는 방식으로 처리한다. 실패하면 `open`은 `close(id)`를 호출해 pool 엔트리를 해제하고 `release_sequence_state_by_id`를 부른다. 원격 runtime은 이미 매핑을 지웠으므로 이 호출은 아무 일도 하지 않는다.

**트레이드오프.** 실패하지 않는 `PipelineServerModel::prepare_sequence_state`는 여전히 panic한다. 남은 호출자는 `CachePool::adopt`와 `adopt_parked_paged`뿐이고, model-owned 파이프라인 레이아웃은 이 경로를 타지 않는다. 보안 리뷰는 이 panic이 peer 입력으로 도달할 수 없음을 확인했다.

### 3.6 거부된 prepare의 롤백

broadcast가 실패하면 runtime은 먼저 id 매핑을 지우고(이후 forward가 반쯤 준비된 id로 진행되지 않고 오류가 되도록), 그다음 stage에 `ReleaseSequence`를 보낸다. 첫 오류에서 멈추는 `broadcast_ack`를 재사용한다. 일반적인 경우에는 이것으로 충분하다. broadcast prepare도 처음 거부한 stage에서 멈췄으므로, 그 앞의 stage들이 id를 준비했고 release는 거부한 stage에 닿기 전에 그들에게 먼저 간다. 리뷰가 기록한 비용은 5절에 있다.

### 3.7 activation 실패의 출구는 하나

각 실패 지점마다 `report_activation_failure` 호출을 추가하는 대신, PR은 `handle_activation`을 래퍼와 모든 곳에서 `?`를 쓰는 `run_activation` 본문으로 나눴다. 나중에 추가되는 실패 지점도 자동으로 보고된다. `report_activation_failure`는 `finish_request`가 owner를 지우기 전에 owner를 읽는다. 이 분리가 아니었다면 생기기 쉬운 순서 버그다.

## 4. 검증

### 4.1 실패 우선 증거

새 테스트와 강화된 테스트를 main 86409e76에서 실행했다(로그는 실행 scratchpad의 `2714-failing-first/`).

| 테스트 | main 86409e76 | 브랜치 |
|---|---|---|
| `flipped_accept_to_at_capacity_verdict_is_rejected_not_retried` (테스트 내 TCP 프록시가 `HELLO_HEAD_LEN + server_identity.len() + TAG_LEN` 위치의 바이트를 뒤집음) | FAILED (`ServerAtCapacity`) | ok |
| `flipped_at_capacity_to_accept_verdict_is_rejected_not_accepted` | FAILED (클라이언트 `Ok`) | ok |
| `flipped_verdict_tag_byte_is_rejected` | FAILED | ok |
| `handshake_rejects_forged_client_tag` (이제 33바이트 레코드를 읽음) | FAILED (1바이트 후 EOF) | ok |
| `server_at_capacity_sends_authenticated_capacity_verdict_record` | FAILED (1바이트 후 EOF) | ok |
| `version_4_hello_is_refused_with_the_version_message` | FAILED (v4 수락) | ok |
| `second_authenticated_peer_cannot_redirect_run_result` | FAILED (rogue의 `PrepareSequence`에 ack) | ok |
| `non_owner_cannot_cancel_or_release_a_prepared_sequence` | FAILED (non-owner 명령에 ack) | ok |
| `final_stage_drops_result_of_unprepared_sequence` | FAILED (`RunResult` 전송) | ok |
| `activation_outside_stage_limits_never_executes` (이제 nonce 42의 `Error`를 요구) | FAILED (5초 안에 응답 없음) | ok |
| `post_execution_failure_reports_error_to_coordinator` | FAILED (5초 안에 응답 없음) | ok |
| `first_prepare_discards_state_left_by_unowned_commands` (666a851f 대상) | FAILED (해제 0회, 기대값 1) | ok |
| `remote_runtime_prepare_failure_is_an_error_not_a_panic`, `pipeline_server_model_reports_prepare_failure_through_try_prepare`, `open_fails_and_releases_the_entry_when_prepare_is_refused` | main에서 컴파일 불가(새 API). main에서 `catch_unwind` 프로브는 `runtime.rs:605`에서 panic | ok |

rogue 테스트는 dual-stack 토폴로지를 쓴다. stage는 `[::]`에서 listen하고, 코디네이터는 `[::1]:<port>`에 bind하지만 `[::]:<port>`로 설정되며, rogue는 `127.0.0.1:<port>`에 bind한다. `[::1]`의 rogue가 IPv4 stage에 접속하면 matcher가 돌기 전에 핸드셰이크에서 거부되므로, 이슈가 설명한 와일드카드 matcher 경로에 닿는 것은 IPv4 rogue다.

### 4.2 로컬 게이트

GB10 spark-102, 71a8c6e0, `--profile test-fast`, `gpu-lock` 아래 `--test-threads=1`, 실모델 대상은 `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`:

| 대상 | 결과 |
|---|---|
| `cargo fmt --all -- --check`, `cargo clippy --features cuda -p mlxcel --lib --tests -D warnings`, 변경된 실모델 대상 clippy | 깨끗함 |
| `--lib distributed::` | 1544 통과, 4 ignored |
| `--lib server::router_front` | 40 통과 |
| mlxcel-core `engine::` | 71 통과 |
| `zero_config_cluster_init --include-ignored` | 7 통과 |
| `pipeline_remote_real_models` (llama, gemma3, gemma4, qwen3, qwen3.5 parity, llama drain 및 shutdown) | 6 통과 |
| `pipeline_server_remote_real_models` (같은 다섯 모델) | 5 통과 |
| `pipeline_ci_multi_stage_real_models --include-ignored` (two-host smoke, 3-stage parity) | 4 통과 |
| `disaggregated_serving_e2e`, `disaggregated_router_e2e` (`--include-ignored`) | 2, 7 통과 |

실행하지 않음: `glm_moe_dsa`(418 GB 체크포인트가 이 장비에 들어가지 않음), 원격 실모델 스위트의 gpt-oss, glm4, Thunderbolt, RDMA 변형. 성능 측정은 하지 않았다. 토큰당 경로에는 명령 하나와 마지막 stage 결과 하나마다 `HashMap` 조회가 하나씩 늘고, verdict는 핸드셰이크마다 HMAC 하나를 더한다.

### 4.3 호스팅 CI

71a8c6e0(그리고 666a851f)에서 `Pipeline Parallel CI / 2-host logical (ubuntu-latest)`가 실패했다. 테스트 자체인 `pipeline_multi_stage_two_host_logical_smoke`는 `ok`를 보고했고, 그 뒤 테스트 바이너리가 프로세스 종료 시 `pure virtual method called`와 SIGABRT로 중단되었다. 같은 종료 시 abort가 같은 날 다른 브랜치 실행(#2716의 ed803065, #2712의 89f5c196)에도 나타났고 그 브랜치의 다른 실행은 통과했으므로, 이 PR 이전부터 있던 테스트 바이너리의 간헐적 teardown abort이지 프로토콜 실패가 아니다. 그래도 녹색 체크를 막으므로 재실행하거나 별도 이슈로 다뤄야 한다.

## 5. 남은 공백과 위험

- **미리 점유된 id는 여전히 요청을 실패시킨다 (LOW, 가용성).** 코디네이터 matcher를 통과하는 secret 보유자는 코디네이터보다 먼저 다음 `pp-seq-N` id들에 `PrepareSequence`를 보낼 수 있다. 이제 그런 요청은 코디네이터를 중단시키지 않고 오류와 함께 깔끔하게 실패하지만, 실패하는 것은 같다. 순차적이지 않은(무작위) request id로 막을 수 있으며, 이는 모든 stage와 코디네이터가 공유하는 id 체계를 바꾼다.
- **`sequence_owners`에 상한이 없다 (LOW).** 이 맵이 대응하는 `caches_by_request`와 마찬가지로, 코디네이터 matcher를 통과하는 peer가 준비하는 id마다 커지고 종료나 shutdown에서만 줄어든다. 소유되지 않은 `RunEntry` 명령은 여전히 prepare나 종료가 지울 때까지 임의의 id 아래 캐시를 만들 수 있다.
- **prepare 실패 정리는 best effort다 (LOW).** `broadcast_ack(ReleaseSequence)`는 첫 오류에서 멈추고, 응답하지 않는 stage마다 모델 워커 스레드에서 `stage_timeout`을 다 기다린다. prepare를 적용했지만 `Ack`를 잃어 타임아웃이 난 stage는 shutdown까지 시퀀스를 보유한다. `release_sequence_state_by_id`도 같은 의미를 가진다.
- **클라이언트가 오해하기 쉬운 오류 레이블을 본다 (LOW).** 거부된 prepare는 API 클라이언트에 `Server busy: open: prepare sequence state ...: remote stage <addr> failed request: ...`로 전달된다. 용량 문제가 아니며, 메시지에 내부 stage 주소가 들어 있다. 이 PR 전에는 같은 조건이 프로세스를 중단시켰다.
- **중간 stage는 소유되지 않은 activation을 여전히 실행한다 (LOW).** forward activation은 인증된 upstream만 보낼 수 있으므로, 중간 stage는 owner가 기록되지 않은 시퀀스의 hidden state를 계산해 전달한다. 마지막 stage가 logits를 버리고 요청을 종료한다.
- **실패하지 않는 API에서 forward 오류는 여전히 panic한다 (기존 문제).** `LanguageModel` forward API가 실패하지 않는 형태이므로 `PipelineServerModel::forward_with_sequence_id`와 `forward_batched_with_ids`는 원격 forward 오류를 panic으로 바꾼다.
- **owner identity는 시퀀스 동안 안정적이어야 한다.** 소유권은 핸드셰이크가 묶은 identity 문자열을 비교한다. identity가 출처 IP에서 도출되는 코디네이터(unspecified IP로 설정됨)의 stage 경로가 시퀀스 도중 바뀌면 자기 명령이 거부된다. 경로는 보통 연결이 유지되는 동안 안정적이며, 변경을 다루는 테스트는 없다.
- **실행하지 않은 것.** `glm_moe_dsa`와 gpt-oss, glm4, Thunderbolt, RDMA 변형. 호스팅 `2-host logical` job은 4.3절의 종료 시 abort로 빨간 상태다.

## 6. 학습 포인트

### 6.1 프로토콜이 분기하는 모든 결정을 인증한다

인증된 핸드셰이크가 인증되지 않은 1바이트로 끝나면, 경로상의 공격자가 연결이 어떻게 끝날지 고르게 된다. 이 PR이 적용한 규칙은, 키가 존재하는(또는 존재하게 될) 시점 이후 상대가 분기 기준으로 쓰는 필드는 transcript와 그 필드에 대한 MAC으로 덮는다는 것이다.

```rust
let (verdict, tag_v) = record.split_at(1);
if !auth.verify(VERDICT_LABEL, &verdict_message(&transcript, verdict[0]), tag_v) {
    return Err(HandshakeRejected("server's handshake verdict does not verify ...".into()).into());
}
match verdict[0] { /* accept, at capacity, reject */ }
```

### 6.2 응답 대상은 최근 요청이 아니라 자원에 묶는다

모든 명령이 옮길 수 있는 "응답이 가는 곳" 필드 하나는 리다이렉트 primitive다. 대상을 자원 생성 시점에 자원별로 기록하고 그 자원에 대한 다른 identity의 명령을 거부하면, 나머지 코드가 믿을 수 있는 소유권 검사가 된다.

### 6.3 소유권에도 깨끗한 시작이 필요하다

non-owner 거부는 자원에 owner가 생긴 뒤에만 자원을 보호한다. 소유되지 않은 명령이 어떤 id 아래 상태를 만들 수 있다면, 첫 owner는 처음부터 시작해야 한다. 그렇지 않으면 먼저 움직인 rogue가 owner의 초기 상태를 정한다. 예측 가능한 이름에 대한 TOCTOU와 같은 형태다.

### 6.4 널리 구현된 trait 메서드를 바꾸지 말고 기본 구현이 있는 실패 가능 변형을 추가한다

수십 개 래퍼가 구현하는 trait 메서드는 바꾸는 비용이 크다. 실패하지 않는 메서드를 호출하는 기본 구현을 가진 `try_*` 메서드를 옆에 두면 기존 구현이 모두 유효하게 남고, 실패할 수 있는 구현 하나만 override하며, 오류를 처리할 수 있는 호출자 하나만 이를 쓰도록 바꾼다.

### 6.5 핸들러당 오류 출구 하나

핸들러 본문을 `Result`를 반환하는 함수로 옮기고 하나의 래퍼에서 보고하면, 나중에 추가되는 것을 포함한 모든 실패가 같은 보고 경로를 탄다. 순서도 명시적이 된다: 여기서는 `finish_request`가 owner를 지우기 전에 owner를 읽는다.

## 7. 참고 자료

- 이슈 #2714: 핸드셰이크 verdict 인증과 응답 대상 바인딩
- PR #2705 (이슈 #2692): 클러스터 pre-auth admission 강화와 at-capacity verdict(`PROTOCOL_VERSION` 4). 그 보고서의 3.7절이 항목 1의 출처
- PR #2708 (이슈 #2694): 마지막 stage가 `RunResult`를 코디네이터에게 직접 보냄. 그 보고서가 항목 2와 3의 출처
- PR #2720 (이슈 #2712): 실행 및 forward 실패에 대한 `report_activation_failure`. 항목 b가 이를 모든 출구로 확장
- PR #2721 (이슈 #2716): 텐서 디코드 한도. 그 보고서의 첫 번째 남은 공백을 항목 b가 고침
- PR #2674, #2684 (인증 핸드셰이크, 프레임별 AEAD)와 #2650 (클러스터 secret): 여기 모든 항목이 secret 또는 경로상 위치를 필요로 하는 이유
