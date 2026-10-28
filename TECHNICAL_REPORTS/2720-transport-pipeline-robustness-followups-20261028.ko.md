# PR #2720: 전송 계층과 원격 파이프라인 견고성 후속 작업

**날짜**: 2026-10-28 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: 89f5c196에서 구현 완료. 구현 리뷰와 보안 및 성능 리뷰 모두 CRITICAL, HIGH, MEDIUM 이슈를 찾지 못함. GB10(spark-102)에서 로컬 게이트 통과. 1ecb25cc에서 main을 머지함. 작성 시점에 호스팅 CI 대기 중
**언어**: Rust, Markdown
**위험도**: 낮음에서 중간 (wire 포맷이나 프로토콜 버전 변경은 없지만, 코디네이터가 이전보다 많은 stage로부터 `Error`를 받아들이고, `ConnectionPool::acquire`의 반환 타입이 바뀌고, TCP RPC 경로가 다시 작성됨)

## 요약

PR #2720(#2712 해결)은 #2700, #2705, #2708이 인증된 클러스터 전송 계층, 원격 파이프라인, 분리형(disaggregated) 라우터에 남긴 잔여 문제 여섯 가지를 고친다. 어느 것도 클러스터 secret이 없는 쪽이 프레임을 읽거나 주입할 수 있게 하지는 않는다. 실패 지연, 로그 가시성, API 정확성 문제이며, `src/distributed/`와 그 테스트 하네스, 한 번의 검증 패스를 공유하므로 하나의 PR로 묶었다.

사용자가 체감하는 변화 두 가지는 모두 실패 경로에 있다. 중간 또는 마지막 파이프라인 stage에서 실행이 실패하거나 중간 stage가 하위 stage로 전달하지 못하면, 이제 그 실패가 진행 중인 exchange의 `Error`로 코디네이터에 도달한다. 그래서 요청은 `stage_timeout` 전체를 기다리지 않고 1초 이내에 실패한다. 그리고 이 PR 이전에는 응답을 읽는 쪽이 없어 매번 타임아웃되던 TCP `rpc_call`이, 이제 요청을 보낸 그 연결에서 응답을 읽는다.

나머지는 커넥션 풀과 경고 예산을 다듬는다. 용량 거부(capacity refusal)를 backoff와 함께 재시도하고, 실패하거나 취소된 send가 drop guard를 통해 active 슬롯을 돌려주고, 읽지 않은 바이트가 남은 유휴 연결은 재사용하지 않으며, 라우터는 빠른 failover용 풀 프로필을 쓴다. 라우터 drop과 용량 거부는 각자의 경고 클래스를 갖고, rate limit 요약은 억제가 일어난 윈도가 언제 끝났는지 말하며, 알 수 없는 kind의 인증된 프레임은 연결을 조용히 닫는 대신 로그를 남긴다.

## 1. 문제 정의

여섯 항목 모두 #2712 작성 시점에 main 752db4f9 기준으로 다시 확인했다.

### 1.1 entry가 아닌 stage의 오류가 코디네이터 타임아웃으로 끝남

#2694 이후 마지막 stage는 logits를 `RunEntry` exchange의 `RunResult`로 코디네이터에 직접 보낸다. 오류는 이 경로를 따르지 않았다. `RemoteStageService::handle_activation`에서 중간 또는 마지막 stage의 실행 오류는 `RemoteStageResponse::Error { nonce: 0, .. }`를 상위 이웃에게 보냈고, 상위 이웃은 command나 activation이 아닌 것은 모두 버린다. 하위로의 `send_activation`이 실패한 중간 stage는 서비스 루프에 `Err`를 돌려주었고, 루프는 "dropping frame"을 로그로 남겼다. 두 경우 모두 코디네이터에 아무것도 도달하지 않았고 `RemotePipelineRuntime::exchange`는 `stage_timeout`을 다 기다렸다. 설령 도달했더라도 코디네이터는 받아들일 수 없었다. `ExchangeMatch::reject_reason`은 `Ack`, `State`, `Error`가 command를 보낸 stage에서 와야 한다고 요구했고, `RunEntry`의 경우 그 stage는 stage 0이다.

### 1.2 `ServerAtCapacity`에 backoff가 없고 경고에 rate limit이 없음

`ConnectionPool::connect_with_retry`는 #2705가 도입한 일시적인 at-capacity 판정을 포함해 모든 handshake 실패를 즉시 반환했고, 시도마다 `Cluster handshake with peer {peer} failed`를 rate limit 없이 warn으로 남겼다. 100 ms 뒤에 슬롯을 비우는 peer라도 그 `acquire`에서는 닿을 수 없었다.

### 1.3 라우터 drop이 진단 예산을 공유하고, 요약 문구가 사실과 달랐음

`spawn_result_demux`는 버리는 result 프레임마다 `HandshakeWarnClass::Diagnostic`을 소비했다. 그래서 result 프레임을 쏟아내는 잘못 설정된 peer가 동시에 발생한 secret 불일치 경고를 debug로 밀어낼 수 있었다. rollover 요약은 해당 클래스의 다음 이벤트가 쓰는데, 그 이벤트는 몇 시간 뒤에 올 수도 있다. 그런데 문구는 `... were logged at debug in the last 60s`였다.

### 1.4 알 수 없는 kind의 인증된 프레임이 연결을 조용히 닫음

#2708은 kind 바이트를 알 수 없는 인증된 프레임을 별도 오류로 만들었지만, 타입 없는 `anyhow!`였다. `handle_connection`은 연결을 닫기 전에 `FrameRejected`만 로그로 남기므로, 버전이 어긋난 멤버는 전송 계층 쪽에서 보이지 않았다.

### 1.5 풀 재사용과 계수

`ProtectedStream::peer_closed`는 읽지 않은 바이트가 있는 연결을, 그 뒤에 FIN이 있더라도 살아 있다고 보고했고 `acquire`는 그런 연결을 재사용했다. `send`, `send_stream`, `rpc_call`은 쓰기 오류 시 `release` 없이 `?`로 반환했기 때문에 `PeerPool::active`가 증가한 채 남았다. 취소된 send도 같은 방식으로 누수되었다. 라우터는 `PoolConfig::default()`로 전송 계층을 바인드했다. 100 ms에서 두 배씩 늘어나는 5회 재연결, 즉 죽은 prefill 노드로의 connect 거부가 failover로 이어지기까지 약 3.1 s의 sleep이 있었다.

### 1.6 TCP `rpc_call`이 응답을 받지 못함

`rpc_call`은 풀에서 꺼낸 outbound 스트림에 요청을 쓰고, 스트림을 반환한 뒤, `connect_timeout` 동안 `rpc_pending` oneshot을 기다렸다. 서버는 `RpcResponse`를 accept한 소켓, 즉 클라이언트 입장에서는 바로 그 outbound 스트림으로 돌려보냈는데, 클라이언트에서 그것을 읽는 쪽이 없었다. 응답은 accept된 연결에서 도는 `handle_connection`에서만 파싱되었다. 모든 TCP RPC가 타임아웃되었다. Thunderbolt와 RDMA의 `rpc_call`은 TCP에 위임하고, `bench::measure_latency`도 이를 사용한다.

## 2. 변경 요약

| 항목 | 영역 | 변경 |
|---|---|---|
| 1 | `pipeline/remote_service.rs` | `handle_activation`의 두 실패 경로가 `report_activation_failure`를 호출한다. 이 함수는 `finish_request`를 실행하고, activation에 `reply_nonce`가 있으면 `send_run_error`로 `reply_target(request_key)`(코디네이터의 인증된 identity, 없으면 설정된 주소)에 `Error`를 보낸다. nonce가 없으면 warn 한 줄만 남기고 아무것도 보내지 않는다. 상위로의 `nonce: 0` 전송과 `upstream_addr`는 제거했다. `Logits` 분기도 `RunResult`의 주소를 `reply_target`으로 정한다. |
| 1 | `pipeline/runtime.rs` | `ExchangeMatch`에 `error_peers` 추가: `RunEntry`면 `&self.stage_peers[..]`, 그 외에는 `std::slice::from_ref(peer)`. `RunEntry`에서 `Error`는 `request_id == Some(<run id>)`도 만족해야 한다. `reject_reason`은 응답 타입에 대한 `match`로 바뀌었다. |
| 2 | `connection_pool.rs`, `cluster_handshake.rs` | `connect_with_retry`는 오류 체인에 `ServerAtCapacity`가 있으면 connect 거부처럼 다룬다. 현재 backoff만큼 sleep한 뒤 재시도하고, 예산을 다 쓴 경우에만 타입을 보존한 채 오류를 반환한다. 다른 handshake 실패는 그대로 terminal이다. 시도별 warn은 `handshake_warn_allowed`를 거친다. |
| 3 | `handshake_warn.rs`, `router_front.rs` | 새 클래스 `RouterDrop`(`router-drop`)과 `AtCapacity`(`at-capacity`), 클래스마다 윈도 슬롯 하나(`CLASS_COUNT = 4`). `WarnDecision::previous_window_ended_ago`가 요약 문구 `N cluster warnings ({class}) were logged at debug during the 60s window that ended Ms ago`에 들어간다. 라우터 demux는 `RouterDrop`을 쓴다. |
| 4 | `tcp_frame.rs`, `tcp_transport.rs`, `handshake_warn.rs` | 기존 `Display` 문구를 유지하는 `UnknownAuthenticatedKind { kind: u8 }`. `read_frame`이 반환하고, `Diagnostic`으로 분류되며, `handle_connection`이 예산을 거쳐 `Closing connection from {peer}: {err} (version skew?)`로 로그를 남긴다. |
| 5 | `tcp_frame.rs`, `connection_pool.rs`, `startup.rs` | `probe_idle() -> IdleProbe { Idle, PendingData, Closed }`, `peer_closed()`는 `== Closed`로 유지. `acquire`는 `Idle`만 재사용하고 `PooledStream` guard를 반환한다(`ProtectedStream`으로의 `Deref`/`DerefMut`, peer별 `Arc<AtomicUsize>`, `Drop`에서 감소). `release`는 guard를 받는다. `PoolConfig::router()`(재시도 2회, 기본 지연 50 ms, connect 2 s)를 라우터 전송 계층이 쓴다. |
| 6 | `tcp_transport.rs` | `rpc_call`은 하나의 `timeout(connect_timeout, ..)` 안에서 같은 풀 스트림에 요청을 쓰고, id가 맞는 `RpcResponse`가 올 때까지 프레임을 읽는다. 그 뒤에만 스트림을 반환하고, 타임아웃이나 오류 시에는 버린다. `rpc_pending`과 `handle_connection`의 `RpcResponse` 분기를 제거했고, accept된 연결로 들어온 `RpcResponse`는 debug로 남긴다. |
| 문서 | `docs/distributed.md`, `docs/CONTINUOUS_BATCHING.md` | 용량 backoff, 네 개의 경고 클래스, 새 요약 문구. #2695 failover 설명 옆에 라우터 풀 프로필. 변경된 여섯 모듈의 모듈 문서 갱신. 이 절들의 한국어판은 없다. |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 18 (소스 9, 테스트 7, 문서 2) |
| 추가 / 삭제 줄 | +1356 / -220 |
| 추가 테스트 | `#[test]`/`#[tokio::test]` 함수 19개. 기존 테스트 3개 조정(`full_connection_cap_reports_capacity_to_the_client`, `panicking_rpc_handler_releases_its_connection_slot`, `peer_closed_probe_is_non_consuming`) |
| 커밋 | d7fd9803(failing-first 테스트), 89f5c196(여섯 항목), 1ecb25cc(main 머지) |

main 머지(1ecb25cc)는 #2711과 #2718을 가져오며, 이 PR의 18개 파일 중 어느 것도 건드리지 않는다.

## 3. 기술적 선택과 그 이유

### 3.1 실패한 stage가 직접 보고하고, 코디네이터는 `Error` 규칙만 넓힌다

| 선택지 | 장점 | 단점 |
|---|---|---|
| `Error`를 상위로 entry stage까지 전달하고 entry stage가 exchange에 답함 | "응답은 command를 받은 peer가 보낸다"는 규칙 유지 | #2694가 막 없앤 요청별 응답 상태가 entry stage에 다시 필요. 실패마다 hop 추가 |
| `Error`를 `result_peer`(마지막 stage)에서만 받음 | matcher 변경이 가장 작음 | 3-stage 파이프라인의 중간 stage도 실패할 수 있고, 그 경우는 여전히 타임아웃 |
| **선택: 설정된 어느 stage든 `RunEntry`의 `Error`를 보낼 수 있음** | hop 하나, 새 stage 상태 없음, 모든 stage 위치를 포괄 | 코디네이터가 더 많은 identity로부터 응답을 받아들임 |

이 확장은 #2694의 `RunResult` 규칙보다 느슨하지 않다. `Error`는 여전히 exchange마다 새로 만드는 랜덤 nonce를 그대로 돌려줘야 하며, 그 nonce는 해당 요청의 activation에만 실린다. `RunEntry`에서는 run의 `request_id`도 같아야 한다. nonce를 볼 수 있는 모든 stage는 원래도 그 요청을 같은 최종 상태(forward 오류)로 몰아넣을 수 있었다. `Error`의 `stage_index`와 `state`는 매칭에 쓰이지 않는다. `Ack`와 `State`는 command를 받은 peer 규칙을 유지하고, `RunEntry`가 아닌 exchange는 `error_peers = [peer]`를 유지한다.

새 `request_id` 검사는 entry stage가 보내는 `Error`에도 적용된다. `handle_run_entry`의 모든 `RunEntry` 오류 경로가 `Some(request_id)`를 보내므로 안전하다.

stage 쪽에서는 `reply_nonce`가 없는 실패를 warn 한 줄과 함께 버린다. 그래도 보낸다면 코디네이터가 버릴 프레임만 생기며, 이는 #2708이 nonce 없는 logits에 적용한 것과 같은 논리다. 이전의 상위 `nonce: 0` 프레임은 어떤 코드도 소비하지 않았으므로, 제거해도 로그 외의 동작은 바뀌지 않는다.

### 3.2 #2714를 위한 접점으로서의 `reply_target`과 `send_run_error`

`reply_target`은 지금은 요청 키를 쓰지 않지만 두 helper 모두 요청 키를 받는다. #2714(시퀀스별 응답 대상 바인딩)는 요청 단위로 라우팅해야 하는데, 모든 호출 지점에 키가 이미 있으므로 그 변경은 `reply_target` 안에서 끝난다. `Logits` 분기도 이 PR에서 같은 helper로 옮겨, 두 응답 종류의 주소 지정 규칙이 하나가 되었다.

### 3.3 at-capacity는 일시적이므로 connect 거부와 같은 backoff를 쓴다

#2705는 handshake에 예산을 낭비하지 않도록 `ServerAtCapacity`를 backoff 없이 반환하도록 의도적으로 설계했다. 그래서 `acquire`는 빨랐지만, 100 ms 동안 가득 찬 peer는 실패한 send가 되었다. 이 PR은 타입이 있는 at-capacity 판정만 connect 거부와 같은 backoff와 예산으로 재시도하고, 나머지 handshake 실패는 terminal로 둔다. 틀린 secret, identity, suite는 재시도해도 맞게 되지 않는다. 예산을 다 쓰면 체인에 `ServerAtCapacity`를 유지한 채 오류를 반환한다. 호출자가 이를 downcast하기 때문이다(`Transport::connect`는 실패하지 않고 로그를 남긴 뒤 계속한다). `full_connection_cap_reports_capacity_to_the_client`는 즉시 판정을 계속 테스트하도록 `max_reconnect_attempts: 0`을 쓴다.

### 3.4 경고 출처마다 예산 클래스 하나, 그리고 경과 시간을 말하는 요약

예산이 클래스별로 나뉘는 이유는 시끄러운 출처 하나가 다른 출처를 굶기지 못하게 하려는 것이다. 라우터 drop과 용량 거부는 인증 진단이 아닌 운영상 경고이고, 둘 다 잘못 설정되었지만 인증된 peer 하나가 높은 빈도로 만들어낼 수 있으므로 각자 윈도를 갖는다. `windows`의 크기는 `CLASS_COUNT`로 정한다.

요약에 대해서는 rollover 시점에 flush하는 타이머를 기각했다. 예산은 `stage_assignment.rs`의 동기 코드에서 호출되는 프로세스 전역 `static`이고, task를 소유할 런타임이 없다. 요약은 여전히 해당 클래스의 다음 이벤트가 쓰며, 이제 억제가 일어난 윈도가 얼마나 전에 끝났는지(`start + window`, saturating)를 문구에 넣는다. 조용한 기간 뒤에는 이 값이 60 s를 훨씬 넘을 수 있다. 문구도 "cluster handshake warnings"에서 "cluster warnings"로 바뀌었다. 네 클래스 중 둘은 handshake 이벤트가 아니기 때문이다. 이전 문구에 매칭되던 로그 알림이 있다면 갱신해야 한다.

### 3.5 active 계수를 위한 drop guard와 세 상태의 idle probe

오류 경로마다 명시적인 `discard()`를 두는 방안은 취소를 다룰 수 없어서 기각했다. 호출자의 타임아웃으로 drop된 `send` future는 오류 분기에 도달하지 않는다. `PooledStream`은 checkout 시 peer별 `Arc<AtomicUsize>`를 증가시키고 `Drop`에서 감소시키므로, 성공, 오류, 취소 모두에서 계수가 맞다. `release`는 내부 스트림을 `idle`로 옮기고 guard가 drop되게 둔다. 카운터가 `Arc`이므로, `PeerPool`보다 오래 사는 guard(`ConnectionPool::shutdown`이 맵을 비운 뒤)는 새 풀의 카운터가 아니라 아무도 읽지 않는 카운터를 감소시킨다.

`probe_idle`은 이전의 "살아 있음" 답을 둘로 나눈다. 이전에는 `PendingData`도 살아 있음으로 셌다. 항목 6이 모든 RPC 응답을 요청한 연결에서 소비하게 되면, 유휴 outbound 연결에는 정당한 미수신 입력이 없다. 남은 바이트는 다음 읽기를 어긋나게 만들고, 그 뒤에 FIN이 있을 수도 있다. 그래서 풀은 그런 연결을 버리고 새 연결을 연다. probe는 여전히 `socket2`를 통한 소비하지 않는 `MSG_PEEK` 한 번이다.

### 3.6 RPC 응답을 요청한 연결에서 읽는다

프로토콜은 이미 응답을 accept된 소켓으로 돌려보내고 있었으므로, 수정은 클라이언트 쪽이다. 그곳에서 읽으면 된다. TCP에서 `rpc_call`을 없애는 대안은 Thunderbolt와 RDMA가 여기에 위임하고 `bench::run_benchmark`가 지연 측정값을 잃게 되므로 기각했다. 읽기 루프는 쓰기와 읽기를 함께 감싸는 하나의 `tokio::time::timeout(connect_timeout, ..)` 안에서 돈다. id가 다른 응답은 debug로 건너뛰고, 다른 kind의 프레임은 오류다. 스트림은 일치하는 응답을 받은 뒤에만 풀로 돌아가고, 타임아웃이나 오류 시 버려진다. 이 덕분에 항목 5가 미수신 데이터를 사용 불가로 취급할 수 있다.

`rpc_pending` 제거는 보안 리뷰가 지적한 구멍도 닫는다. 이전에는 인증된 어느 peer든 accept된 연결로 순차 id를 추측한 `RpcResponse`를 보내 다른 호출의 pending 항목을 완료시킬 수 있었다.

### 3.7 라우터 풀 프로필 값

`PoolConfig::router()`는 `max_reconnect_attempts: 2`, `reconnect_base_delay: 50ms`, `connect_timeout: 2s` 외에는 기본값을 유지한다. connect 거부의 경우 오류가 라우터의 failover 로직에 도달하기까지 sleep이 약 3.1 s에서 50 + 100 ms로 줄어든다. 2 s connect 타임아웃은 패킷을 흘려버리는 호스트에 대한 상한이며, connect 거부 경로와는 관계없다. 이 프로필은 `start_server`의 라우터 분기에만 적용되고, prefill과 decode 노드는 기본값을 유지한다.

## 4. 검증

### 4.1 failing-first 근거

동작 테스트를 먼저 커밋(d7fd9803)하고, 752db4f9에 테스트 파일만 더한 상태로 실행했다. 13개 중 11개가 예상한 방식으로 실패했다.

| 테스트 | 752db4f9에서의 실패 |
|---|---|
| `final_stage_execution_error_goes_to_coordinator` | 5 s 안에 코디네이터가 아무것도 받지 못함 |
| `final_stage_drops_execution_error_without_reply_nonce` | 상위 stage가 `nonce: 0` Error를 받음 |
| `error_from_last_stage_fails_forward_promptly` | exchange가 5 s 타임아웃을 다 기다림 |
| `server_at_capacity_is_retried_with_backoff` | handshake가 즉시 실패, 재시도 없음 |
| `server_at_capacity_exhausts_retry_budget` | 60 ms 미만에 반환, backoff 미사용 |
| `failed_send_releases_active_slot`, `cancelled_send_releases_active_slot` | `total_active`가 1로 남음 |
| `rpc_call_round_trips_over_authenticated_tcp` | `RPC call timed out` |
| `timed_out_rpc_call_drops_its_stream` | `total_idle`이 1 |
| `authenticated_unknown_kind_is_logged_before_close` | close 로그가 한 줄도 잡히지 않음 |
| `acquire_drops_idle_connection_with_pending_data` | 유휴 연결이 재사용됨 |

양쪽에서 모두 통과한 두 개(`error_from_last_stage_with_wrong_{nonce,request_id}_is_dropped`)는 버그를 보여주는 것이 아니라 새 수용 규칙을 지킨다. 새 API(`HandshakeWarnClass::RouterDrop`, `WarnDecision::previous_window_ended_ago`, `UnknownAuthenticatedKind`, `PoolConfig::router`, `IdleProbe`)를 쓰는 테스트는 752db4f9에서 컴파일되지 않으며, 그 심볼이 없다는 것이 근거다. `authenticated_unknown_kind_is_logged_before_close`는 범위가 한정된 subscriber로 debug 레벨에서 로그를 잡는다. 같은 바이너리의 앞선 테스트가 프로세스 전역 예산을 이미 써버렸을 수 있기 때문이다.

### 4.2 로컬 게이트

GB10 spark-102, `--profile test-fast --features cuda,test-utils`, GPU lock 밖에서 빌드하고 `gpu-lock` 아래에서 `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`로 89f5c196에서 실행:

| 대상 | 결과 |
|---|---|
| `cargo fmt --check`, `cargo clippy --lib --tests -D warnings` | 깨끗함 |
| `--lib distributed::` | 1513 통과, 4 무시 |
| `--lib server::router_front` | 35 통과 |
| `zero_config_cluster_init --include-ignored` | 7/7 |
| `pipeline_remote_real_models --include-ignored` | 10/10 (`glm_moe_dsa` 제외) |
| `pipeline_server_remote_real_models --include-ignored` | 7/7 (`glm_moe_dsa` 제외) |
| `pipeline_ci_multi_stage_real_models --include-ignored` | 4/4 |
| `disaggregated_serving_e2e` / `disaggregated_router_e2e` | 2/2, 7/7 |

실제 모델 대상의 greedy parity는 #2708 대비 변화 없다. 제외한 `glm_moe_dsa` 테스트는 128 GB GB10에 올릴 수 없는 418.6 GB 체크포인트가 필요하다. 성능 측정은 하지 않았다. 토큰당 경로는 바뀌지 않았고, 실패 경로, 풀 bookkeeping, RPC 경로만 바뀌었다.

## 5. 남은 공백과 위험

- **entry가 아닌 stage의 forward 이전 실패는 여전히 타임아웃된다 (LOW).** `handle_activation`에서 하위 peer 없음, `sequence_length`나 `RequestId` 파싱 실패, `serialize_mlx_array` 실패, 들어온 activation의 `deserialize_wire_tensor` 실패는 여전히 `?`로 전파된다. 이들은 코디네이터의 stage 타임아웃으로 끝나고, `mark_request_started` 이후에 일어나는 것들은 코디네이터의 cleanup 전까지 요청의 캐시를 남긴다. #2712는 실행 오류와 send 실패만 명시했다. 이들을 `report_activation_failure`로 보내는 것은 작은 후속 작업이다.
- **라우터의 500 ms failover 기준은 단위 테스트 수준에서만 확인된다.** `router_pool_config_fails_fast`는 라우터 프로필로 거부되는 포트에 대한 `acquire`가 500 ms 안에 실패함을 확인한다. `disaggregated_router_balances_and_survives_node_failure`는 통과하지만 시간을 단언하지 않으므로, 종단 간 failover 지연은 측정되지 않았다.
- **entry가 아닌 stage의 오류는 하네스 테스트로만 다뤄지고 실제 모델 실행으로는 다뤄지지 않는다.** stage 실패를 주입하는 실제 모델 대상이 없다. stage 쪽은 `StubOutput::Error`로, 코디네이터 쪽은 correlation 하네스로 테스트한다.
- **용량 재시도가 handshake를 증폭한다 (LOW).** at-capacity 재시도에는 jitter가 없고 매번 connect와 handshake 전체를 수행하므로, 가득 찬 peer에 대한 동시 `acquire` N개는 최대 6N번(라우터 프로필이면 3N번)의 handshake를 만들고, 서버 쪽에서 각각 rate limit 없는 "max connections reached" warn을 남긴다. 상한이 있고 인증된 peer만 가능하다. 그 warn을 `AtCapacity`로 rate limit하고 jitter를 더하면 해결된다.
- **`Transport::connect`가 "Failed to pre-connect"를 rate limiter 없이 남긴다 (LOW).** 시도마다가 아니라 예산을 다 쓴 뒤 `connect` 호출당 한 번이다.
- **`PooledStream`이 자신의 peer를 기록하지 않는다 (LOW).** `release(other_peer, guard)`는 연결을 엉뚱한 idle 목록에 넣는다. active 계수는 맞게 유지되고, 현재 모든 호출자는 acquire한 키로 release한다.
- **`rpc_call`은 256 MiB `max_frame_size`로 읽는다 (LOW).** 길이 헤더는 본문을 열기 전까지 인증되지 않으므로, 경로상의 공격자가 진행 중인 호출마다 큰 할당 하나를 강제할 수 있다. accept된 연결이 이미 가진 것과 같은 노출이다.
- **stage의 nonce 없는 실패 warn은 rate limit되지 않는다 (LOW).** 바로 옆의 logits drop warn과 같다. 인증된 상위 stage만 이를 일으킬 수 있다.
- **`rpc_call`의 기한은 `connect_timeout`이다.** 이 PR 이전부터 그랬지만(이전 oneshot 대기도 이 값을 썼다), 라우터 프로필의 2 s가 라우터가 하는 모든 RPC의 상한이 되고, `acquire` 자체의 재시도 예산은 그 기한이 시작되기 전에 소비된다.

## 6. 학습 포인트

### 6.1 슬롯 계수를 호출이 아니라 값으로 만든다

모든 종료 경로가 감소를 기억해야 하는 카운터는 아무도 작성하지 않은 경로, 즉 취소에서 실패한다. checkout된 값의 `Drop`에 감소를 묶으면 성공, `?`, drop된 future가 모두 같은 경우가 된다.

```rust
pub struct PooledStream { stream: Option<ProtectedStream>, active: Arc<AtomicUsize> }
impl Drop for PooledStream {
    fn drop(&mut self) { self.active.fetch_sub(1, Ordering::SeqCst); }
}
```

`cancelled_send_releases_active_slot`는 32 MiB send를 200 ms 타임아웃으로 drop해 취소 경우를 고정한다. 명시적인 `discard()`로는 다룰 수 없던 경우다.

### 6.2 요청/응답 프로토콜은 요청을 쓴 곳에서 응답을 읽는다

원래 TCP RPC는 서버는 올바르게 동작했지만, 클라이언트가 읽지 않는 소켓으로만 도착할 수 있는 응답을 pending 맵이 기다리고 있었다. 서버가 accept된 소켓으로 답한다면 클라이언트는 같은 연결을 읽고, 일치하는 응답까지 연결을 쥐고, 타임아웃 시 버려서 늦은 응답이 풀 연결의 다음 사용자를 망가뜨리지 않게 해야 한다. 그러면 풀 연결은 정확한 idle 신호가 된다. 대기 중인 바이트가 하나라도 있으면 무언가 잘못된 것이다.

### 6.3 오류는 결과와 같은 길로 보낸다

#2694가 `RunResult`를 "답을 가진 stage가 코디네이터에 직접 보낸다"로 바꿨을 때, 오류는 옛 경로에 남아 조용히 타임아웃이 되었다. 프로토콜이 누가 응답하는지를 바꿀 때는 실패 응답도 성공 응답과 함께 옮겨야 하며, matcher의 응답 타입별 출처 규칙은 하나의 `match`로 써서 각 규칙이 나란히 보이도록 하는 것이 좋다.

### 6.4 지연 작성되는 rate limit 요약은 시점을 말해야 한다

타이머가 아니라 다음 이벤트가 쓰는 요약은 알 수 없는 거리의 과거를 묘사한다. 윈도의 경과 시간을 함께 보고하면, 동기적인 프로세스 전역 static에 백그라운드 task를 들이지 않고도 문구가 정확해진다.

## 7. 참고 자료

- Issue #2712: 전송 계층과 원격 파이프라인 견고성 후속 작업
- PR #2700 (#2695 해결): 분리형 서빙과 라우터 e2e 수정, `ProtectedStream::peer_closed`
- PR #2705 (#2692 해결): 인증 전 admission 강화, `ServerAtCapacity`, handshake 경고 예산
- PR #2708 (#2694 해결): 마지막 stage가 코디네이터로 `RunResult` 전송, `reply_nonce`, `coordinator_identity`, 알 수 없는 kind의 인증된 프레임 오류
- Issue #2637: 클러스터 전송 인증과 응답 correlation(`ExchangeMatch` 규칙)
- Issue #2714: handshake 판정 인증과 응답 대상 바인딩(`reply_target`의 사용처)
- Issues #2715, #2716: 라우터 parity 잔여 문제와 텐서 디코드 한도. 이 PR은 해당 영역을 건드리지 않음
