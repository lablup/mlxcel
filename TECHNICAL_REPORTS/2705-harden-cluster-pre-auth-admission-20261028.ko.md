# PR #2705: 클러스터 인증 전 수락 단계와 거부 처리 강화

**날짜**: 2026-10-28 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: 구현 완료, 구현 리뷰와 보안 리뷰에서 CRITICAL/HIGH 이슈 없음, GB10 (spark-102)에서 로컬 테스트 통과, 작성 시점에 hosted CI 진행 중
**언어**: Rust, Markdown
**위험도**: 중간 (핸드셰이크 wire 버전이 3에서 4로 올라가므로 이 변경 전후로 빌드된 노드는 서로 연결되지 않음, 피어 목록이 송신자의 바인딩된 identity와 맞지 않는 router는 이제 그 송신자의 result frame을 버림)

## 요약

PR #2705 (#2692 해결)는 #2674 보안 리뷰가 `src/distributed/`의 인증된 클러스터 transport와 disaggregated router의 result demux에 남긴 다섯 가지 잔여 문제를 해결합니다. 어느 것도 클러스터 secret이 없는 쪽이 frame을 읽거나 주입할 수 있게 하는 문제는 아닙니다. 이미 secret을 요구하는 표면에 대한 DoS, 로그 가시성, 멤버 간 격리 강화이며, 같은 서브시스템과 같은 검증 절차를 공유하므로 하나의 PR로 묶었습니다.

1. 소스별 pending handshake 한도가 IPv6 소스를 전체 주소가 아닌 /64 단위로 셉니다. /64를 가진 호스트가 주소를 바꿔 가며 한도를 우회할 수 없습니다.
2. connection task가 panic하면 누수되던, 아무도 읽지 않는 `accept_loop`의 `active_conns` 카운터를 삭제했습니다. 연결별 자원은 모두 drop으로 해제됩니다.
3. 프로세스 전체에서 하나였던 "분당 경고 5개" limiter를 새 `handshake_warn` 모듈의 독립된 두 예산(`Noise`, `Diagnostic`)으로 바꿨습니다. 쓰레기 연결이 secret 불일치 의심 경고를 debug로 밀어내지 못합니다.
4. router는 secret 없이 바인딩된 transport로는 build되지 않으며, result demux는 인증된 송신자가 `--prefill-peers` 또는 `--decode-peers`에 없으면 frame을 버립니다.
5. `max_connections` 검사가 핸드셰이크 안, 클라이언트 인증 이후로 옮겨졌습니다. 가득 찬 listener는 새 `VERDICT_AT_CAPACITY` (0x02) 바이트로 응답하고, 클라이언트는 "핸드셰이크 성공 후 소켓 종료" 대신 타입이 있는 `ServerAtCapacity` 에러를 받습니다. `PROTOCOL_VERSION`이 3에서 4로 올라갑니다.

PR은 17개 파일, +1104 / -156이며, 작성 커밋 두 개(`699d208d` 구현, `80974c3c` 리뷰 문구 정리)와 `main` 머지 커밋(`9900bb89`)으로 구성됩니다.

## 1. 문제 정의

#2692 작성 시점에 다섯 항목 모두 `main` @ 54d63b0a에서 확인되었습니다.

### 1.1 IPv6 주소 순환으로 소스별 pending 한도 우회

#2655에서 도입된 `PendingHandshakeLimiter::try_acquire`는 진행 중인 핸드셰이크를 정규화된 소스 주소별로 세고 `max_pending_handshakes_per_ip`(기본 8)로 제한했습니다. IPv6 호스트는 보통 /64 전체를 쓰며 연결마다 다른 주소를 소스로 쓸 수 있으므로, 혼자서 전역 pending pool(기본 64) 전체를 차지할 수 있었습니다. 모듈 문서에도 이 공백이 이미 적혀 있었습니다.

### 1.2 누수되면서 아무도 읽지 않던 카운터

`accept_loop`는 `active_conns: Arc<AtomicUsize>`를 accept 시 증가시키고 세 종료 경로에서 수동으로 감소시켰습니다. connection task 안에서, 예를 들어 등록된 RPC handler 안에서 panic이 나면 감소가 건너뛰어졌습니다. 워크스페이스는 `panic = "unwind"`로 빌드되므로 tokio가 panic한 task의 future를 drop하고, semaphore permit과 pending slot은 RAII라 이미 정상 해제되었습니다. 피해를 보는 것은 아무도 읽지 않는 `active_conns`뿐이었습니다. 다만 이 값 위에 metric이나 한도를 처음 만드는 순간 panic마다 값이 위로 어긋나는 잠재적 함정이었습니다.

### 1.3 쓰레기 연결이 운영자에게 필요한 한 줄을 가림

`tcp_transport.rs`의 `handshake_warn_allowed()`는 60초당 경고 5개라는 하나의 예산을 모든 거부가 나눠 썼습니다: pending pool 거부, 모든 핸드셰이크 실패, 인증에 실패한 frame. 소켓을 열 수 있는 누구나 잘못된 magic 바이트, 깨진 hello, 5초 timeout까지 버티기로 이 예산을 소진할 수 있습니다. 쓰레기 소켓 몇 개 뒤에 도착한 진짜 `ClientAbortedAfterServerProof`("피어가 다른 클러스터 secret을 가지고 있을 가능성이 높음")는 debug로만 기록되었습니다.

### 1.4 router demux가 secret을 가진 누구든 신뢰

`spawn_result_demux`는 `from`과 상관없이 `request_id`가 맞는 대기 요청에 `ResultFrame`을 전달했습니다. request id는 `next_id.fetch_add`로 만들어지므로 순차적이고 추측 가능합니다. secret을 가진 누구든(예를 들어 같은 secret 파일을 공유하는 pipeline stage) 다른 클라이언트의 스트리밍 응답에 토큰을 주입할 수 있었습니다. 별개로 `RouterState::build`는 아무 `TcpTransport`나 받았습니다. router transport가 인증된 상태임을 보장하는 것은 상위의 resolver와 `startup.rs`의 주석뿐이었고, router 테스트 fixture 두 개 모두 auth 없이 바인딩했습니다.

### 1.5 가득 찬 연결 한도가 성공한 핸드셰이크처럼 보임

`server_handshake_inner`가 `VERDICT_ACCEPT`를 쓰고 반환한 뒤에야 `accept_loop`가 `conn_semaphore.try_acquire_owned()`를 시도했습니다. 실패하면 소켓을 그냥 닫았습니다. 클라이언트는 핸드셰이크 성공을 보고, 첫 send에서 끊긴 연결을 만났으며, 에러에는 피어가 가득 찼다는 정보가 없었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|------|------|
| `src/distributed/pending_handshakes.rs` | 새 `pending_source_key(ip)`: 정규화(IPv4-mapped IPv6는 IPv4로), IPv4는 그대로, IPv6는 `u128::from(v6) & !((1u128 << 64) - 1)`로 /64 마스킹. `try_acquire`와 테스트 헬퍼 `pending_from`이 사용. `PendingSlotRefusal::PerIp`는 이 키를 담고 IPv6 키를 `2001:db8:1:2::/64`처럼 출력. |
| `src/distributed/tcp_transport.rs` | `active_conns`, 수동 감소 세 곳, `AtomicUsize` import 제거. `accept_loop`가 `server_handshake_admitting(..., FrameChannel::Transport, \|\| conn_semaphore.try_acquire_owned().ok())`를 호출하고 `ConnectionLimitReached`는 rate limit 없이 기록(인증된 피어만 도달 가능). 새 `TcpTransport::is_authenticated()`. 기존 limiter 삭제, 호출부는 class를 넘김. |
| `src/distributed/handshake_warn.rs` (신규, 202줄) | `HandshakeWarnClass { Noise, Diagnostic }`, 에러 체인에서 타입이 있는 거부 다섯 가지를 찾는 `handshake_warn_class(&anyhow::Error)`, class별 window 두 칸과 시간 주입이 가능한 `allow_at(class, now) -> WarnDecision`을 가진 `WarnBudget`, rollover 시 `N cluster handshake warnings (<class>) were logged at debug in the last 60s`를 기록하는 프로세스 전역 `handshake_warn_allowed(class)`. |
| `src/distributed/cluster_handshake.rs` | `VERDICT_AT_CAPACITY = 0x02`, `PROTOCOL_VERSION = 4`. 새 `pub(crate) server_handshake_admitting<P>(..., admit: impl FnOnce() -> Option<P>) -> Result<(String, SessionKeys, P)>`, `server_handshake_on`은 `\|\| Some(())`로 감싼 wrapper. 클라이언트는 verdict 0x02를 `ServerAtCapacity`로, 서버는 `ConnectionLimitReached { peer }`를 반환. wire 다이어그램과 모듈 문서 갱신. |
| `src/distributed/stage_assignment.rs` | 새 limiter를 import, pending 거부는 `Noise`, 핸드셰이크 실패는 `handshake_warn_class(&err)`, 거부된 frame은 `Diagnostic`. 핸드셰이크 수락 로직은 그대로(wrapper가 모두 수락). |
| `src/distributed/connection_pool.rs`, `mod.rs` | 재시도하지 않는 핸드셰이크 경로의 주석이 일시적인 `ServerAtCapacity`까지 설명. `ServerAtCapacity`, `ConnectionLimitReached` re-export, `handshake_warn`은 `pub(crate)`. |
| `src/server/router_front.rs` | `RouterState::build`가 `!transport.is_authenticated()`이면 실패. 새 `result_senders: HashSet<SocketAddr>`(IP를 정규화한 prefill, decode 피어)와 `accepts_result_from(&str)`. demux는 다른 송신자의 frame을 `Diagnostic` 예산을 통해 `router: dropping result frame` 경고와 함께 버리며, 문자열을 조립하지 않고 tracing field로 기록. |
| `src/bin/mlx_server.rs`, `src/server/startup.rs` | `--prefill-peers` / `--decode-peers` 도움말에 각 항목은 피어가 접속해 오는 주소여야 한다고 명시. startup 주석은 `RouterState::build`가 auth를 강제한다고 기술. |
| `docs/distributed.md`, `docs/CONTINUOUS_BATCHING.md` | /64 단위 집계와 같은 서브넷 공유의 결과, router 송신자 검사가 인증하는 것과 하지 않는 것, at-capacity 결과, 두 경고 class, 버전 4, secret 없는 쪽도 `Diagnostic` class에 도달할 수 있다는 잔여 위험. |
| 테스트 (`_tests.rs` 5개, +523) | `pending_handshakes`, `handshake_warn`, `cluster_handshake`, `tcp_transport::cluster_auth`, `server::router_front`에 새 테스트 13개. router fixture 두 개는 이제 임의의 `ClusterSecret`으로 바인딩. |

## 3. 기술적 선택과 그 이유

### 3.1 별도 prefix 단계 대신 /64를 키로 사용

이슈는 자체 설정 필드를 가진 별도의 prefix별 한도를 추가하는 안을 기각했습니다. 하나의 설정값을 더 거친 소스 단위로 세는 것으로 충분합니다. 대가는 문서에 명시되어 있습니다: 같은 IPv6 /64 안의 클러스터 노드들은 이제 진행 중 핸드셰이크 8개라는 소스별 예산을 공유하므로, 하나의 /64에 있는 대규모 IPv6 클러스터는 `max_pending_handshakes_per_ip`를 높여야 할 수 있습니다. pending pool은 핸드셰이크의 몇 초 동안만 쓰이므로 대량 재연결 시에만 의미가 있습니다. 마스킹 전에 정규화하는 것이 중요합니다. 그렇지 않으면 IPv4-mapped IPv6 주소(`::ffff:a.b.c.d`)는 상위 64비트가 0이라 `::/64`로 마스킹되고, dual-stack 소켓의 모든 IPv4 클라이언트가 하나의 키를 공유하게 됩니다.

### 3.2 카운터를 고치지 않고 삭제

감소를 panic에 안전하게 만드는 것(drop guard)은 아무도 읽지 않는 값을 유지할 뿐입니다. 이슈는 대신 향후 활성 연결 metric은 `max_connections - conn_semaphore.available_permits()`로 계산하라고 기록합니다. permit이 이미 RAII이므로 어긋날 수 없습니다. 회귀 방지 테스트 `panicking_rpc_handler_releases_its_connection_slot`은 설계상 변경 전후 모두 통과하며, 삭제 자체가 아니라 "panic한 handler가 slot을 누수하지 않는다"는 성질을 고정합니다.

이 테스트의 관측 방법은 이슈 문구에서 바뀌었습니다. `TcpTransport`는 `RpcResponse` frame을 accept된 연결에서 도는 `handle_connection`에서만 파싱하고, `rpc_call`이 쓰는 pool의 outbound 연결은 읽지 않습니다. 따라서 어떤 TCP 클라이언트도 RPC 응답을 받을 수 없고, 테스트는 대신 클라이언트 B의 `send`가 2초 안에 `server.recv()`에 도달하는지를 봅니다. 이것은 TCP 위 RPC의 기존 한계이며(저장소 안의 유일한 호출자 `bench.rs`는 결과를 무시함), PR과 이슈에 기록되었지만 여기서 고치지는 않았습니다.

### 3.3 소스별 첫 이벤트 경고 대신 두 경고 class

대안인 "소스마다 첫 이벤트는 항상 경고"는 소스별 map이 필요하고, 그 map은 무한히 커지거나 축출을 해야 하며 IPv6 주소 순환은 둘 다 무력화합니다. 고정된 window 두 개(`Mutex<[Option<Window>; 2]>`)는 소스별 상태가 필요 없습니다. `WarnBudget::allow_at`이 `now`를 받으므로 테스트는 sleep 없이 noise 이벤트 50개와 61초 rollover를 재현할 수 있고, `Instant::now()`를 호출하는 곳은 프로세스 전역 wrapper뿐입니다.

분류는 `anyhow` 체인을 `downcast_ref`로 훑어 `ClientAbortedAfterServerProof`, `HandshakeRejected`, `IdentityNotOwned`, `FrameSuiteMismatch`, `FrameRejected`를 찾습니다. 타입이 없는 나머지(잘못된 magic, 깨진 hello, timeout, I/O)는 `Noise`입니다. pending 거부는 항상 `Noise`, 핸드셰이크 이후 frame 거부와 router drop은 항상 `Diagnostic`입니다.

이 구분은 보안 경계가 아닙니다. `IdentityNotOwned`와 `FrameSuiteMismatch`는 client hello만으로 결정되고, `ClientAbortedAfterServerProof`는 공격자 자신의 IP를 담은 정상 hello를 보낸 뒤 연결을 닫기만 하면 됩니다. 따라서 secret이 없는 쪽도 `Diagnostic`을 포화시킬 수 있습니다. 리뷰 커밋에서 문서를 이에 맞게 고쳤고, 그 경우 남는 신호는 rollover 요약 한 줄입니다.

### 3.4 핸드셰이크 안의 callback으로 수락 결정

`server_handshake_admitting`은 `admit: impl FnOnce() -> Option<P>`를 받고 permit을 key와 함께 반환합니다. 이 형태에서 세 가지 성질이 나옵니다.

- `admit()`은 클라이언트 authenticator가 검증된 뒤에 호출됩니다(불일치면 먼저 `VERDICT_REJECT`를 반환). 따라서 secret을 가진 쪽만 용량 상태를 알 수 있습니다. `admission_is_not_consulted_before_the_client_authenticates`가 이를 고정합니다.
- permit은 결과와 함께 밖으로 나가는 지역 변수입니다. `admit()` 이후 `VERDICT_ACCEPT` 쓰기가 실패하거나 5초 timeout이 걸리면 permit은 future와 함께 drop됩니다. 수동으로 세는 것은 없습니다.
- 연결 수를 제한하지 않는 listener(stage assignment 채널)는 계속 `|| Some(())`을 넘기는 `server_handshake_on`을 호출하므로 동작이 바뀌지 않습니다.

permit 타입을 generic으로 두어 `cluster_handshake.rs`가 `tcp_transport.rs`의 semaphore에 의존하지 않습니다.

### 3.5 `ServerAtCapacity`는 치명적이지 않고, backoff 재시도도 하지 않음

`Transport::connect`는 재시도로 secret 불일치를 고칠 수 없으므로 `HandshakeRejected`를 치명적으로 처리합니다. `ServerAtCapacity`는 별도 타입이므로 `connect`는 경고를 기록하고 계속합니다. `connect_with_retry`에서는 다른 핸드셰이크 실패와 같이 backoff 없이 빠져나갑니다: 에러는 바로 호출자에게 가고, 다음 `acquire`가 새 연결을 엽니다. 처음에는 문서와 pool 주석이 이를 누가 재시도하는지 말하지 않고 "retryable"이라고만 했는데, 리뷰 커밋에서 바로잡았습니다.

### 3.6 reject 바이트 재사용 대신 wire 버전 상향

이 PR 전에는 0x01이 아닌 verdict는 모두 거부였습니다. 이전 클라이언트가 0x02를 받으면 `HandshakeRejected`("handshake was altered in transit")로 보고하는데, 이는 오해를 부르고 `connect`에서 치명적입니다. `PROTOCOL_VERSION`을 4로 올리면 버전이 섞인 쌍은 hello 단계에서 명시적인 버전 메시지로 실패합니다. #2650, #2674, #2684와 이 변경은 모두 v0.8.0-beta.1 이후에 들어갔으므로 버전 3을 쓰는 릴리스 빌드는 없습니다. 그 PR들과 이 PR 사이의 `main` 빌드는 이후 빌드와 연결되지 않으며, 이는 기존의 "버전 혼용 미지원" 규칙과 같습니다.

### 3.7 정규화된 IP와 포트로 router 송신자 검사

`result_senders`는 설정된 피어를 `SocketAddr::new(ip.to_canonical(), port)`로 저장하고, `accepts_result_from`도 `from`을 같은 방식으로 파싱합니다. 양쪽에서 IPv4-mapped 형태, scope id, flowinfo가 제거됩니다. 파싱되지 않는 `from`은 거부되고, 바인딩된 identity는 unspecified일 수 없으므로 설정된 `0.0.0.0` 항목은 아무것과도 맞지 않습니다. 인증되는 것은 소스 IP뿐입니다(핸드셰이크가 TCP 연결에 바인딩). 포트는 송신자가 광고한 listener 포트입니다. 따라서 같은 IP 뒤의 secret 보유자 둘은 서로의 포트를 사칭할 수 있습니다. 문서에 이 점이 명시되어 있으며, secret 보유자를 멤버로 보는 위협 모델 아래에서 수용했습니다.

기각한 안: 각 요청을 라우팅된 정확한 prefill 또는 decode 노드에 묶는 것. demux에 요청별 라우팅 상태가 필요하고, 프로토콜은 어느 role이 어느 단계를 보내는지 고정하지 않습니다. 이 검사는 router의 result 수신만 다루며, prefill의 요청 수신과 decode의 handoff 수신은 계속 인증된 멤버 누구든 신뢰합니다(#2692의 범위 밖).

운영자가 체감하는 변화: 바인딩된 identity가 설정 주소와 다른 피어(multi-homed 노드, 또는 `0.0.0.0`에 바인딩되어 다른 인터페이스로 router에 접속하는 노드)의 frame은 이제 버려지고, 요청은 기존 handoff timeout으로 실패합니다. `router: dropping result frame` 경고가 진단 수단입니다.

## 4. 검증

PR 본문 기준(GB10 spark-102, `--profile test-fast`, 빌드는 `gpu-lock` 밖에서, 실행은 `gpu-lock run --tag issue-2692` 아래에서):

- `cargo fmt --check`와 `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`: 깨끗함, 리뷰 커밋 후 재실행.
- 새로 추가되거나 수정된 테스트 13개 이름 지정 실행: 13개 통과. `--lib distributed::`: 1473 통과, 0 실패, 4 ignored. `--lib server::router_front`: 35 통과. `--test zero_config_cluster_init`: 4 통과, 3 ignored. `tests/disaggregated_serving_e2e.rs`: 2 통과. `tests/disaggregated_router_e2e.rs`: 7 통과.
- 실제 모델 parity(`MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, `--ignored`): `pipeline_remote_real_models` 10 통과 1 실패, `pipeline_server_remote_real_models` 7 통과 1 실패, `pipeline_ci_multi_stage_real_models` 2 통과. llama, gemma3, gemma4, qwen3, qwen3.5, glm-4.7 parity 테스트는 모두 통과.
- 두 실패는 회귀가 아닙니다. `glm5-4bit` 체크포인트가 없어서 `glm_moe_dsa` 테스트가 `repo_model_dir`에서 panic한 것입니다. `mlx-community/GLM-5-4bit`는 418.6 GB라 128 GB GB10 장비에서 로드할 수 없으며, 이 테스트들은 #2674보다 먼저 있었습니다.
- 수동 3-프로세스 loopback(router, prefill, decode가 하나의 `--cluster-secret-file`을 공유, `qwen3-0.6b-4bit`, 127.0.0.1): `POST /v1/chat/completions`가 HTTP 200과 `"2 + 2 = 4."`를 반환했고, router 로그에 `dropping result frame` 줄이 없었습니다. loopback 구성에서 송신자 검사가 정상 frame을 버리지 않음을 보여 줍니다.

실패 우선 증거: 브랜치에서 수정 hunk를 임시로 되돌리고 테스트를 이름으로 실행했습니다. `ipv6_sources_in_one_64_share_the_per_source_limit`, `ipv6_sources_in_different_64s_are_counted_separately`, `noise_flood_does_not_suppress_diagnostic_warnings`, `build_refuses_an_unauthenticated_transport`, `result_demux_drops_frames_from_unconfigured_senders`가 모두 실패했습니다. 수락 관련 테스트(`server_at_capacity_sends_capacity_verdict`, `admission_is_not_consulted_before_the_client_authenticates`, `full_connection_cap_reports_capacity_to_the_client`)와 분류 테스트는 새 API 없이는 컴파일되지 않습니다. `rollover_reports_suppressed_count`와 panic 회귀 테스트는 설계상 양쪽에서 통과합니다.

머지된 head `9900bb89`의 머지 전 게이트(오케스트레이터 로그, 작성 시점에 진행 중): `verify-fmt`, `verify-versions`, `verify-env-var-docs`, `verify-kernel-dtype-keys`, `verify-kernel-port-dispatch`, `verify-binary-assets`, `verify-windows-paths`가 모두 exit 0이었습니다. contract 테스트 빌드는 아직 컴파일 중이었으므로 그 실행의 테스트 결과는 인용하지 않습니다.

## 5. 남은 공백과 위험

- **버전 3과 4 노드는 연결되지 않습니다.** `main`에서 빌드한 클러스터의 롤링 업그레이드는 모든 노드를 함께 갱신해야 합니다. 릴리스된 버전은 영향이 없습니다.
- **verdict 바이트는 이전처럼 인증되지 않습니다.** 경로상의 공격자는 0x01과 0x02를 바꿀 수 있지만, 이미 연결을 끊을 수 있었으므로 새 능력은 아닙니다.
- **`Diagnostic` 예산은 secret 없이도 도달 가능합니다**(3.3절). 두 class는 값싼 쓰레기 연결을 막을 뿐, 정상 hello를 보내는 외부인은 막지 못합니다.
- **router drop은 프로세스 전역 `Diagnostic` 예산을 핸드셰이크 진단과 공유합니다.** 잘못 설정된 피어는 요청마다 result frame을 연달아 보내고 각각이 버려지며 집계되므로, 분당 5개 경고를 소진해 동시에 발생한 secret 불일치 경고를 debug로 밀어낼 수 있습니다. rollover 요약은 개수를 세지만 어떤 종류의 이벤트가 억제되었는지는 말하지 않습니다.
- **rollover 요약은 지연되어 기록됩니다.** 해당 class의 다음 window 첫 이벤트에서 기록됩니다. flood가 멈추면 억제된 개수는 그 class의 다음 이벤트가 올 때까지(훨씬 나중일 수 있음) 보고되지 않으며, 그때는 메시지의 "in the last 60s" 문구가 부정확합니다.
- **`ServerAtCapacity`에는 backoff가 없습니다.** 가득 찬 피어에 대해 `send`를 빠르게 반복 재시도하는 호출자는 시도마다 전체 핸드셰이크를 수행하고, `connect_with_retry`는 매번 rate limit 없이 `Cluster handshake with peer ... failed`를 warn으로 기록합니다. secret 보유자만 관련되고 저장소 안에 이렇게 반복하는 호출자는 알려져 있지 않지만, 막는 장치도 없습니다.
- **같은 위치 멤버 간 포트 사칭**(3.7절)은 설계상 수용했습니다.
- **TCP 위 RPC는 응답을 돌려받을 수 없습니다**(3.2절). 기존 한계이며 PR에 기록되었지만, 이 보고서가 확인한 범위에서는 전용 이슈로 추적되지 않습니다.
- **`glm_moe_dsa`의 실제 모델 parity**는 체크포인트가 맞지 않아 이 하드웨어에서 검증되지 않은 채로 남습니다.

## 6. 학습 포인트

- **수락 판단은 인증된 교환 안에 둡니다.** 핸드셰이크 이후에 용량을 검사하면 정보는 새지 않지만 클라이언트 경험이 나쁘고, 이전에 검사하면 외부인에게 용량 상태가 샙니다. "클라이언트 검증 완료"와 "verdict 쓰기" 사이에 호출되는 callback은 둘 다 해결하며, permit을 session key와 함께 반환하면 자원 수명이 RAII 안에 머뭅니다.
- **신뢰 수준이 다른 이벤트가 예산을 공유하는 rate limiter는 목적이 뒤집힙니다.** 원래 limiter는 외부인으로부터 로그를 지키려다 운영자를 침묵시켰습니다. 누가 보냈는지가 아니라 그 줄의 가치로 예산을 나누는 것은 class가 적고 고정되어 있으면 비용이 거의 없습니다.
- **마스킹 전에 정규화합니다.** IPv4-mapped IPv6 주소는 prefix 처리 전에 IPv4로 바꿔야 합니다. 그렇지 않으면 dual-stack listener의 모든 IPv4 클라이언트가 하나의 /64로 합쳐집니다.
- **읽지 않는 상태는 강화하지 말고 삭제합니다.** 아무도 읽지 않는 카운터도 이후의 모든 수정에서 정확하게 유지되어야 합니다. 필요할 때 RAII 자원에서 값을 유도합니다.
- **window 로직에는 시간을 주입합니다.** `allow_at(class, now)` 덕분에 rollover와 flood 테스트가 sleep 없이 결정적으로 동작합니다.

## 7. 참고

- PR #2705, 이슈 #2692
- 출발점: #2674 (#2655 인증 전 pending 한도와 타입이 있는 거부), #2684 (#2656 frame별 AEAD), #2650 (분산 role의 클러스터 secret)
- 주요 파일: `src/distributed/handshake_warn.rs`, `src/distributed/cluster_handshake.rs`, `src/distributed/pending_handshakes.rs`, `src/distributed/tcp_transport.rs`, `src/server/router_front.rs`, `docs/distributed.md`
