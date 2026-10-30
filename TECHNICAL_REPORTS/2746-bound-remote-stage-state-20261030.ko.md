# PR #2746: 원격 stage 상태, RPC 응답, 텐서 축의 상한 설정

**날짜**: 2026-10-30 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: 4a18642b에서 구현 완료, c46934f4에서 origin/main을 병합함. 보안 리뷰가 공백 하나(길이 제한 없이 준비된 request id를 저장하므로 개수 상한이 메모리를 제한하지 못함)를 찾았고 4a18642b에서 고침. PR 본문의 로컬 게이트는 4a18642b에서 GB10(spark-102) 기준 통과. 작성 시점에 병합된 head에서 두 번째 로컬 게이트가 실행 중이었고 호스팅 CI는 일부 대기 중이었음(`2-host logical`, `cargo-clippy`, `CPU-only link`, `OpenXLA feature compile`, `WebUI installed artifact`)
**언어**: Rust, Markdown
**위험도**: 낮음 (wire 버전 변경 없음. stage가 준비되지 않은 id의 `RunEntry`와 activation, 그리고 4096개를 넘는 새 prepare를 거부함. 공개 타입 `DimBound`와 `ShapeBound::Uniform`에 variant와 필드가 추가됨)

## 요약

PR #2746(#2726 해결)은 #2720, #2721, #2723이 원격 파이프라인 stage 서비스, 텐서 디코드 한도, TCP 전송 계층에 남긴 LOW 보안 잔여 문제 네 가지와 같은 모듈의 견고성 항목 하나를 처리한다. 원격 stage는 이제 어느 코디네이터도 자신에게 준비하지 않은 id의 `RunEntry`와 forward activation을 거부하고, 소유된 시퀀스를 최대 `MAX_STAGE_SEQUENCES`(4096)개까지만 보유한다. 이 두 규칙이 함께 모든 시퀀스별 맵을 제한한다. `PrepareSequence`는 `RequestId::from_string`이 거부하는 id도 거부하므로, 상한은 항목 수뿐 아니라 메모리도 제한한다. 원격 코디네이터의 request id에는 CSPRNG 접미사가 붙어(`pp-seq-{n}-{16자리 hex}`), 두 번째 secret 보유자가 다음 id를 미리 점유할 수 없다. `rpc_call`은 256 MiB 프레임 한도 대신 1 MiB 상한으로 응답을 읽고, 시퀀스 축과 mask 축은 길이 0을 거부하며, 재연결 backoff에는 jitter가 붙고 서버의 at-capacity 경고는 rate limit이 걸린 경고 예산으로 옮겨진다.

항목 1에서 4는 클러스터 secret을 가진 peer만 도달할 수 있다(항목 3은 경로상 공격자도 가능). 어느 항목도 secret이 없는 쪽이 프레임을 읽거나 주입할 수 있게 하지 않는다.

## 1. 문제 정의

#2726은 main 59c07652에서 확인한 잔여 문제를 모았다. 다섯 항목을 하나의 PR에서 구현하고 마지막에 검증을 한 번 실행하도록 요구했다.

### 1.1 stage별 시퀀스 상태에 상한이 없었다 (LOW, secret 보유자)

`RemoteStageService`는 시퀀스별 맵 세 개를 가진다: `caches_by_request`, `active_requests`, `sequence_owners`. 어느 것에도 개수 제한이 없었다. `PrepareSequence`는 새 id마다 owner와 캐시 집합을 넣었고, `RunEntry`와 forward activation은 아무도 준비하지 않은 id에 대해서도 캐시 집합을 만들었다. `owned_by_other`가 소유되지 않은 id를 통과시켰기 때문이다. 코디네이터 matcher를 통과하는 secret 보유자는 세 맵을 끝없이 키울 수 있었고, 중간 stage는 소유되지 않은 id의 activation을 실행했다(#2723 보고서의 "중간 stage가 여전히 소유되지 않은 activation을 실행한다" 공백). #2723은 첫 `PrepareSequence`가 그 id 아래 캐시를 버리게 해서 무결성 측면은 막았지만, 메모리 측면은 열려 있었다.

### 1.2 순차적인 원격 request id는 예측 가능했다 (LOW, secret 보유자)

`request_id_for_sequence`는 스케줄러의 순차 `SequenceId`로 `pp-seq-{N}`을 만들었다. #2723이 시퀀스별 소유권을 도입한 뒤로, `pp-seq-{N+1}`을 먼저 준비한 secret 보유자는 코디네이터 자신의 prepare를 "prepared by another coordinator identity"로 실패시킬 수 있었다. #2723은 이 실패를 깔끔하게 만들었지만(요청이 실패하고 코디네이터는 살아남음), 미리 점유된 id마다 요청 하나가 여전히 거부되었다. #2723 보고서는 이를 첫 번째 남은 공백으로 적었다.

### 1.3 `rpc_call` 응답을 전체 프레임 한도로 읽었다 (LOW, secret 보유자 또는 경로상 공격자)

`TcpTransport::rpc_call`은 응답을 `config.max_frame_size`(기본 256 MiB)로 읽었다. `read_frame`은 프레임이 인증되기 전에 `len + tag`를 할당하므로, peer나 길이 헤더를 바꾸는 경로상 공격자가 진행 중인 호출마다 256 MiB 할당을 강제할 수 있었다. 유일한 호출자는 64바이트를 주고받는 지연시간 벤치마크(`bench.rs`)다.

### 1.4 길이 0인 축이 텐서 디코드 한도를 통과했다 (LOW, secret 보유자)

`DimBound::AtMost(n)`은 0을 허용하므로, hidden state `[1, 0, width]`, id `[1, 0]`, logits `[1, 0, vocab]`이 `StageWireLimits`를 통과해 executor에 도달했다. `ShapeBound::Uniform`은 길이 0인 mask 축을 허용했다. 정상적인 코디네이터나 stage는 빈 시퀀스 축을 만들지 않는다. 그런 텐서는 그 경우를 고려하지 않고 작성된 모델 코드 경로에만 도달한다.

### 1.5 동시에 일어나는 재연결과 rate limit 없는 capacity 경고 (견고성)

`connect_with_retry`는 연결 거부, 타임아웃, at-capacity verdict 뒤에 고정된 두 배 증가 지연만큼 잤으므로, 함께 거부된 클라이언트는 함께 재시도했다. 포화된 peer에 대한 N개의 동시 acquire는 최대 6N번의 핸드셰이크(`PoolConfig::router()`에서는 3N)를 만들었고, 각각이 rate limit 없는 서버 경고("max connections (N) reached")를 남겼다.

## 2. 변경 요약

| 항목 | 영역 | 변경 |
|---|---|---|
| 1 | `pipeline/remote_service.rs` | `handle_run_entry`는 텐서를 디코드하기 전에 `sequence_owners` 항목이 없는 id를 거부한다: `send_request_error`로 `Error("sequence <id> was not prepared on this stage")`를 보내고 아무것도 만들지 않는다. `run_activation`은 먼저 owner를 `ensure!`하므로, 소유되지 않은 activation은 디코드나 실행 없이 기존 `report_activation_failure` 경로로 간다(보고할 owner가 없으므로 로그만 남기고 버림). |
| 1 | `pipeline/remote_service.rs` | `pub(crate) const MAX_STAGE_SEQUENCES: usize = 4096`을 `load`에서 설정하는 새 `max_sequences` 필드에 저장한다. `sequence_owners.len() >= max_sequences`일 때 새 id의 `PrepareSequence`는 `Error("pipeline stage already holds {max} prepared sequences")`를 받고 아무것도 바뀌지 않는다. owner가 이미 가진 id를 다시 준비하는 것은 새 항목이 아니므로 여전히 `Ack`를 받는다. `#[cfg(test)]` `spawn_with_executor_and_sequence_cap`이 상한을 낮추고, `spawn_with_executor`와 새 생성자는 private `spawn_service` 하나를 공유한다. |
| 1 (리뷰) | `pipeline/remote_service.rs` | `PrepareSequence`는 먼저 `RequestId::from_string`(1에서 256바이트)으로 id를 검증한다. 잘못된 id는 request id 없이 `Error("invalid request id (must be 1 to 256 bytes)")`로 응답하고, draining 검사와 상한 검사보다 먼저 처리하며, 아무것도 저장하지 않는다. |
| 1 | `pipeline/remote_service.rs` | #2723의 첫 prepare 오래된 캐시 폐기는 다중 방어로 남긴다. 모듈 문서와 `owned_by_other` 문서가 새 규칙을 설명한다. |
| 2 | `pipeline/runtime.rs` | `RemotePipelineRuntime::prepare_sequence_state`는 `pp-seq-{n}-{auth.random_u64():016x}`를 만들어 `request_ids`에 저장하고, 이후 모든 명령이 거기서 읽는다. `InProcessPipelineRuntime`은 `request_id_for_sequence`로 `pp-seq-{n}`을 유지한다. 테스트용 `StubStageHandle`은 이제 `PrepareSequence` id를 각각 기록한다. |
| 3 | `tcp_transport.rs` | `const MAX_RPC_RESPONSE_FRAME: usize = 1 << 20`. `rpc_call`은 `MAX_RPC_RESPONSE_FRAME.min(config.max_frame_size)`로 읽는다. 크기를 넘는 응답은 헤더에서 호출을 실패시키고, 기존 오류 분기가 스트림을 버린다. |
| 4 | `tensor_serialize.rs` | 새 `DimBound::Within { min, max }`(양끝 포함, "between {min} and {max}"로 표시). `ShapeBound::Uniform`에 `min_dim`이 추가된다. 검사는 `DimBound`를 만들고(`min_dim == 0`이면 `AtMost`, 아니면 `Within`) "axis {i} is {d}, expected {bound}"로 보고한다. `TensorDecodeLimits::unrestricted`는 `min_dim: 0`을 쓴다. `AtMost`는 유지한다. |
| 4 | `pipeline/wire_tensor.rs` | `sequence_axis(max)` 헬퍼가 `Within { min: 1, max }`를 반환하고, `hidden_states()`, `input_ids()`, `logits()`의 시퀀스 축에 쓰인다. `attention_mask()`는 `min_dim: 1`을 쓴다. `exact_or_unbounded(None)`은 `Within { min: 1, max: u64::MAX }`가 된다. |
| 5 | `connection_pool.rs` | `fn jittered(delay) -> Duration`은 `ring::rand::SystemRandom`으로 `[delay/2, delay]` 구간의 균등 값을 반환하고, RNG가 실패하면 `delay`로 돌아간다. `connect_with_retry`는 시도마다 한 번 값을 뽑아 at-capacity debug 줄에 기록하고 그만큼 잔다. `delay`의 두 배 증가는 그대로다. |
| 5 | `tcp_transport.rs` | accept 루프의 `ConnectionLimitReached` 분기는 `handshake_warn_allowed(HandshakeWarnClass::AtCapacity)`가 허용할 때만 warn으로, 그 외에는 debug로 기록한다. 이웃한 핸드셰이크 실패 분기와 같은 방식이다. |
| 문서 | `docs/distributed.md` | "What the secret protects"가 소유되지 않은 명령 거부, 4096 상한, id 검증, 무작위 id 접미사를 설명한다. 텐서 한도는 "between 1 and 2^20"으로 바뀐다. backoff 문단은 jitter와 rate limit이 걸린 capacity 줄을 언급한다. residual risk 목록에 1 MiB `rpc_call` 상한을 적는다. 이 문서의 한국어판은 없다. |
| 테스트 | 테스트 파일 6개와 `runtime.rs` | 새 테스트 8개, 다시 쓴 테스트 1개, 조정한 테스트 여러 개(4.1절). `tcp_transport_frame_tests.rs`의 `LogBuf`는 robustness 테스트가 로그를 수집할 수 있도록 `pub(super)`가 되었다. |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 13개 (소스 6, 테스트 6, 문서 1) |
| 추가 / 삭제 라인 | main 060b5051 대비 +725 / -95 |
| 추가된 테스트 | 새 테스트 8개 (`remote_service_tests.rs` 3, `runtime.rs` 1, `wire_tensor_tests.rs` 1, `tcp_transport_robustness_tests.rs` 2, `connection_pool_tests.rs` 1), 다시 쓴 테스트 1개 |
| 커밋 | e8539643 (항목 1에서 5), 4a18642b (보안 리뷰의 prepare id 검증, 실제로 자는 jitter 지연을 기록), c46934f4 (origin/main 병합) |

## 3. 기술적 선택과 그 이유

### 3.1 맵마다 상한을 두지 않고 소유되지 않은 명령을 거부한다

**컨텍스트.** 세 맵이 id마다 커졌고, 항목을 만드는 경로가 둘이었다: `PrepareSequence`(소유됨)와 `RunEntry` 또는 activation(소유되지 않음).

| 대안 | 평가 |
|---|---|
| 맵마다 따로 상한을 두고 오래된 항목을 축출 | 소유된 항목을 축출하면 살아 있는 시퀀스가 조용히 깨진다. 일관되게 유지할 상한이 세 개다 |
| 소유되지 않은 상태를 타임아웃 후 만료 | id마다 타이머가 필요하고, 정상 상태에서도 공격자가 맵을 가득 채워 둘 수 있다 |
| **선택: owner 없이는 상태를 만들지 않고, owner에 상한 하나** | `caches_by_request`와 `active_requests`의 모든 항목이 이제 소유된 id에만 존재하므로, `sequence_owners`의 상한 하나가 세 맵을 모두 제한한다 |

정상 트래픽은 영향을 받지 않는다. 코디네이터의 모든 `RunEntry`는 모든 stage에 보내는 `PrepareSequence` broadcast 뒤에 오고(`runtime.rs`), 시퀀스의 activation은 코디네이터가 그 stage에 시퀀스를 준비한 뒤에만 도달한다. `RunEntry` 거부는 텐서 디코드 앞에 있으므로, 거부된 명령의 비용은 디코드가 아니라 맵 조회 한 번과 짧은 오류 응답이다. activation 경로는 별도 검사 대신 `run_activation` 안의 `ensure!`를 써서, #2723이 도입한 단일 실패 출구를 지나간다. owner가 없으면 그 출구는 로그만 남기고 버린다.

상한은 축출 대신 새 id를 거부한다. 거부하면 받아들인 시퀀스는 모두 유효하게 남고, 코디네이터는 이미 거부된 prepare를 그 요청만 실패시키는 방식으로 처리한다(#2723 항목 a).

**트레이드오프.** 상한은 identity별이 아니라 stage 전체에 하나다. 코디네이터 matcher를 통과하는 secret 보유자는 4096개 슬롯을 모두 자기 prepare로 채우고 해제하지 않을 수 있고, 그 뒤로 그 stage의 새 정상 prepare는 그 항목들이 해제되거나 stage가 종료될 때까지 모두 실패한다. 이 PR은 제한 없는 메모리 증가를 제한되고 눈에 보이는 가용성 실패로 바꾼다. 가용성 공격 자체를 없애지는 않으며, 그러려면 identity별 계정이나 만료가 필요하다(5절).

### 3.2 준비된 id를 검증해 개수 상한을 메모리 상한으로 만든다 (리뷰 수정)

보안 리뷰는 stage가 준비된 id를 세 맵에 (`String` 키나 값으로) 저장하고, 그 id가 프레임만큼 길 수 있다는 점을 지적했다. 프레임 크기에 가까운 id 4096개는 개수는 제한하지만 메모리는 제한하지 못한다. 4a18642b는 `RequestId::from_string`이 거부하는 id(비어 있거나 256바이트 초과)를 모두 거부한다. 정상 코디네이터는 같은 함수로 id를 만들기 때문에, 이는 정확히 정상 코디네이터가 절대 만들 수 없는 id의 집합이다. 이 검사는 draining 검사와 상한 검사보다 먼저 실행되고, 오류 응답에 request id를 싣지 않으므로 큰 id가 wire로 되돌아가지 않는다. 이 검사로 저장되는 id는 대략 4096 x 3 x 256바이트로 제한된다.

### 3.3 무작위 id가 아니라 무작위 접미사

**컨텍스트.** 항목 2에는 두 번째 secret 보유자가 예측할 수 없는 id가 필요하다.

| 대안 | 평가 |
|---|---|
| 전부 무작위인 id | 시퀀스 번호를 잃는다. 로그와 테스트는 그 번호로 원격 요청과 스케줄러 시퀀스를 연결한다 |
| `pp-seq-{n}`을 유지하고 소유권에 의존 | #2723의 공백 목록에 있던 선점 거부가 열린 채로 남는다 |
| **선택: `pp-seq-{n}-{random_u64:016x}`** | 예측 불가(exchange nonce에 이미 쓰는 `ClusterAuth` CSPRNG의 64비트), 여전히 연결 가능, 256바이트 id 한도 안 |

id는 한 번 생성되어 `request_ids`에 저장되고, 이후 모든 명령이 거기서 읽으므로 `runtime.rs`의 다른 코드는 바뀌지 않는다. `InProcessPipelineRuntime`은 경쟁할 peer가 없으므로 단순한 형식을 유지한다. stage는 id를 불투명한 문자열로 다루므로, 이전 stage도 새 id를 받아들이고 wire 버전 변경이 필요 없다.

### 3.4 `rpc_call`에는 설정 필드 대신 응답 크기의 상한

`rpc_call`의 호출자는 64바이트 echo를 하는 지연시간 벤치마크 하나뿐이다. 설정 가능한 상한은 아무 호출자도 설정하지 않는 필드를 늘린다. 상수는 `min(config.max_frame_size)`와 결합되므로, 더 작은 프레임 한도로 설정된 전송 계층은 그 한도를 유지한다. 크기 초과 오류는 `read_frame`에서 할당 전에 헤더만으로 발생하고, 기존 `Ok(Err)` 분기가 스트림을 버리므로 읽지 않은 바이트가 남은 연결이 pool로 돌아가지 않는다.

### 3.5 `AtMost` 옆에 `Within`, `Uniform`에 `min_dim`

**컨텍스트.** 한도 타입은 공개 모듈 `mlxcel::distributed::tensor_serialize`에 있다.

`AtMost`의 의미를 "1에서 n"으로 바꾸면 모든 호출자의 의미가 조용히 바뀐다. 빈 텐서를 계속 받아들여야 하는 `TensorDecodeLimits::unrestricted`도 포함된다. 새 `Within { min, max }` variant는 각 사용 지점에서 의도를 드러낸다. stage 한도는 `sequence_axis` 헬퍼로 이를 써서 "최소 1"이라는 결정을 한곳에 둔다. `Uniform`에는 `min_dim` 필드를 두어 mask의 균일 rank 형태를 유지하고, unrestricted 한도에서는 `0`을 쓴다. `Uniform` 검사는 이제 `DimBound`를 만들어 `admits`와 `Display`를 재사용하므로, 두 shape 형태가 같은 오류 문구를 낸다.

**트레이드오프.** 두 enum 모두 `#[non_exhaustive]`가 아니므로, 새 variant와 필드는 `DimBound`를 빠짐없이 match하거나 `ShapeBound::Uniform`을 구조체 리터럴로 만드는 외부 crate에서 소스 호환성을 깬다. `src/distributed/` 밖에서 두 타입을 쓰는 코드는 없다.

### 3.6 sleep에 jitter를 주고 두 배 증가는 유지한다

equal jitter(`[delay/2, delay]`)는 함께 거부된 클라이언트를 흩어 놓으면서도 최소 대기를 보장한다. at-capacity 경로는 포화된 서버가 슬롯을 비울 시간을 이 최소 대기에 의존한다. `delay` 자체의 두 배 증가는 그대로이므로 전체 예산의 상한은 이전과 같다. jitter는 연결 전에 시도마다 한 번 뽑으므로 debug 줄이 실제로 자는 값을 출력한다(4a18642b의 리뷰 수정. 첫 버전은 jitter 전 지연을 기록했다). `SystemRandom`은 `ClusterAuth`를 통해 이미 의존성에 있다.

서버의 capacity 경고는 새 클래스 대신 기존 `HandshakeWarnClass::AtCapacity` 예산으로 옮긴다. 이 클래스는 연결하는 쪽의 at-capacity 줄도 쓰므로, 클라이언트이자 서버인 프로세스에서는 두 줄이 예산 하나를 공유한다. 운영자는 여전히 window마다 처음 몇 줄과 분당 억제된 경고 요약을 본다.

## 4. 검증

### 4.1 먼저 실패하는 테스트

PR 본문은 변경하지 않은 main 8385f40b에서 새 테스트를 실행한 결과를 기록한다:

| 테스트 | main 8385f40b | 브랜치 |
|---|---|---|
| `unprepared_run_entry_is_refused_and_creates_no_state` | 실패 (`Error` 대신 `RunResult`) | ok |
| `unowned_run_entry_cannot_seed_state_before_first_prepare` (#2723의 `first_prepare_discards_state_left_by_unowned_commands`를 다시 씀) | 실패 (`RunResult`) | ok |
| `final_stage_drops_result_of_unprepared_sequence` (실행 0회도 확인하도록 강화) | 실패 (실행 1회) | ok |
| `remote_runtime_request_ids_are_unpredictable` | 실패 ("expected pp-seq-21-<16 hex digits>, got pp-seq-21") | ok |
| `zero_length_axes_are_refused` | 실패 (`[1, 0, 8]` 수락) | ok |
| `rpc_call_refuses_oversize_response` | 실패 (2 MiB 응답 반환) | ok |
| `full_server_rate_limits_capacity_warnings` | 실패 (20개 중 20개가 warn) | ok |
| `prepare_with_invalid_request_id_is_refused` (리뷰) | 검사를 끄면 실패 (300바이트 id가 `Ack`, `in_flight_requests: 1`) | ok |
| `prepare_beyond_the_sequence_cap_is_refused`, `jittered_delay_stays_in_half_to_full_range` | main에서 컴파일되지 않음 (새 API) | ok |

조정한 기존 테스트:

- `run_entry_with_bad_tensor_*`와 `run_entry_with_out_of_bounds_ids_*`는 이제 먼저 prepare한다. 그러지 않으면 원래 확인하려던 텐서 검사 대신 새 소유권 거부에 걸린다.
- `server_at_capacity_exhausts_retry_budget`는 경과 시간 하한을 60 ms에서 30 ms로 낮춘다. 20 ms와 40 ms에 jitter를 적용한 두 sleep의 최솟값이다.
- `full_server_rate_limits_capacity_warnings`는 이슈의 "정확히 한 번" 대신 warn이 최대 `2 * HANDSHAKE_WARN_PER_WINDOW`개임을 확인한다. 예산은 window당 5개이고, 프로세스 전역이며 pool 자신의 at-capacity 줄과 공유되고, 테스트 중에 window 교체가 한 번 일어날 수 있다. 20줄은 모두 나와야 한다(warn과 debug의 합이 20).
- wire tensor 메시지 단언 두 개는 이제 "between 1 and N"을 읽는다.

### 4.2 로컬 게이트

GB10 spark-102, 4a18642b, `--profile test-fast`, `gpu-lock run --tag issue-2726`에서 `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, 바이너리는 lock 밖에서 먼저 빌드:

| 대상 | 결과 |
|---|---|
| `cargo fmt --all -- --check`; `cargo clippy --profile test-fast --features cuda,test-utils -p mlxcel --lib --tests -- -D warnings` | 깨끗함 |
| `--lib distributed::` (`cuda,test-utils`) | 1552 통과, 4 ignored |
| `distributed_integration`, `pipeline_e2e` | 20, 30 통과 |
| `pipeline_remote_real_models`, `pipeline_server_remote_real_models`, `pipeline_ci_multi_stage_real_models` (`--include-ignored --skip glm_moe_dsa`, 모델 skip 없음) | 10, 7, 4 통과 |
| `zero_config_cluster_init`, `disaggregated_serving_e2e` (`--include-ignored`) | 7, 2 통과 |
| `disaggregated_router_e2e` (`--include-ignored`) | 7 통과 |

항목 1과 항목 2가 정상 코디네이터를 깨지 않는다는 증거는 실제 모델 대상들이다. 이 테스트들은 이제 소유되지 않은 명령을 거부하는 stage를 상대로, 새 id 형식으로 시퀀스를 준비하고 실행하고 해제한다.

첫 `disaggregated_router_e2e` 실행에서 `disaggregated_router_rejects_model_owned_family_and_keeps_serving` 하나가 실패했다(첫 요청에서 503, load average 약 50). 로그를 켠 재현으로 원인을 라우터의 health monitor로 추적했다. monitor는 시작 3초 뒤에 probe하는데, prefill이 아직 listen하지 않아 다음 tick까지 down으로 표시되었고, 테스트는 prefill의 TCP 포트가 열리자마자 요청을 보낸다. 이 경로에 pool이나 재시도 코드는 없다. 같은 빌드는 부하가 낮을 때 이 테스트를 3번 중 3번 통과했고 전체 대상은 7/7 통과했다. 이 race는 이 PR 이전부터 있었고 여기서 고치지 않는다.

실행하지 않음: `glm_moe_dsa`(418 GB checkpoint가 이 장비에 들어가지 않음), Thunderbolt와 RDMA 변형. 성능 측정은 하지 않았다. hot path에는 `RunEntry`와 activation마다 `HashMap::contains_key` 하나가 추가되고, 명령 프레임마다 id가 17바이트 늘어난다.

### 4.3 main 병합 이후

c46934f4는 origin/main(060b5051)을 브랜치에 병합한다. 위 게이트는 그 이전 결과다. 작성 시점에 병합된 head에서 두 번째 로컬 게이트가 실행 중이었고 호스팅 CI는 일부 대기 중이었으므로, 두 결과 모두 여기에는 기록하지 않는다.

## 5. 남은 공백과 위험

- **secret 보유자가 상한을 채울 수 있다 (LOW, 가용성).** `MAX_STAGE_SEQUENCES`는 identity 사이에 공유되고, 소유된 항목은 해제, 완료, 종료 때만 빠진다. 코디네이터 matcher를 통과하는 peer는 id 4096개를 준비해 붙잡아 둘 수 있고, 그 뒤로 그 stage의 정상 prepare는 요청 단위로 깔끔하게 실패한다. identity별 할당량이나 소유된 시퀀스의 유휴 만료가 이를 막을 수 있다.
- **상한은 상수다.** 4096은 현재 코디네이터의 스케줄러 캐시 pool보다 훨씬 크지만, 여러 코디네이터가 stage 하나를 공유하거나 캐시 pool이 훨씬 커지면 올릴 방법 없이 상한에 걸린다. 이슈는 의도적으로 "설정 불가"를 선택했다.
- **`rpc_call`은 1 MiB를 넘는 응답을 실을 수 없다.** 지연시간 벤치마크에는 충분하다. 더 큰 응답이 필요한 미래의 호출자는 새 경로나 매개변수가 필요하다.
- **공유되는 경고 예산.** 서버의 capacity 줄과 클라이언트의 at-capacity 줄이 `HandshakeWarnClass::AtCapacity`를 공유하므로, 같은 프로세스에서 바쁜 클라이언트 쪽이 서버 줄을 debug로 밀어낼 수 있다. 분당 요약은 여전히 억제된 개수를 보고한다.
- **공개 타입 변경.** `DimBound::Within`과 `ShapeBound::Uniform::min_dim`은 외부 crate가 있다면 그곳의 빠짐없는 match와 구조체 리터럴을 깬다.
- **기존 라우터 health race.** 높은 부하에서의 `disaggregated_router_e2e` 503(4.2절)은 라우터 health monitor의 테스트 타이밍 race로, 이 변경과 무관하며 아직 별도 이슈가 없다.
- **실행하지 않음.** `glm_moe_dsa`, Thunderbolt와 RDMA 변형. 병합된 head의 게이트와 호스팅 CI는 작성 시점에 대기 중이었다.

## 6. 학습 포인트

### 6.1 owner 없는 항목을 거부하고, owner에 상한을 둔다

여러 맵이 키마다 커질 때 맵마다 상한을 두면 불일치와 축출 버그가 생기기 쉽다. 모든 항목이 먼저 owner 기록을 요구하게 만들 수 있다면, owner 맵의 상한 하나가 전부를 제한하고, 상한에서 새 키를 거부하면 받아들인 키는 모두 유효하게 남는다.

```rust
if !self.sequence_owners.contains_key(&request_id)
    && self.sequence_owners.len() >= self.max_sequences
{
    // 거부; 아무것도 바뀌지 않음
}
```

### 6.2 개수 상한은 항목 크기가 제한될 때만 메모리 상한이다

리뷰 수정은 일반적인 규칙이다. 각 항목의 키 크기를 공격자가 정할 수 있다면, 저장 항목 수의 제한은 메모리에 대해 아무것도 말해 주지 않는다. 정상 클라이언트가 쓰는 같은 생성자로 키를 검증하고, 무엇이든 저장하거나 되돌려 보내기 전에 검증한다.

### 6.3 예측할 수 없는 식별자가 선점 race를 막는다

먼저 주장한 쪽에 주는 소유권은 주장 대상 이름의 예측 불가능성만큼만 강하다. 순차 id와 선착순 소유권의 조합은 예측 가능한 이름에 대한 TOCTOU다. CSPRNG 접미사는 사람이 읽을 수 있는 접두사를 연결용으로 남기면서 race를 없앤다.

### 6.4 읽기 상한은 전송 계층 최대치가 아니라 예상 응답에 맞춘다

전송 계층 전체의 프레임 한도는 가장 큰 정상 프레임(여기서는 텐서)을 위해 존재한다. 응답이 작은 request-response 경로는 응답 크기의 상한으로 읽어야 한다. 인증 전에 일어나는 할당은 공격자가 조종하는 할당이기 때문이다.

### 6.5 backoff의 equal jitter

`[delay/2, delay]` 구간의 균등 jitter는 함께 거부된 클라이언트의 동기를 깨면서 최소 대기를 보존한다. 즉시 재시도할 수 있는 full jitter(`[0, delay]`)와 다른 점이다. 로그가 실제로 잔 값을 보고하도록 시도마다 값을 한 번만 뽑는다.

## 7. 참고 자료

- 이슈 #2726: bound remote pipeline stage state, RPC replies and tensor shapes
- PR #2723 (이슈 #2714): 시퀀스별 소유권과 실패 가능한 prepare. 그 보고서의 "미리 점유된 id"와 "sequence_owners에 상한 없음" 공백을 여기서 닫는다
- PR #2721 (이슈 #2716): stage 텐서 디코드 한도. 항목 4가 하한을 추가한다
- PR #2720 (이슈 #2712): 전송 계층 견고성 후속 작업(backoff를 둔 capacity 재시도, 같은 연결에서 받는 `rpc_call` 응답). 항목 5가 그 backoff에 jitter를 준다
- `docs/distributed.md`, "What the secret protects"와 "Residual risks"
