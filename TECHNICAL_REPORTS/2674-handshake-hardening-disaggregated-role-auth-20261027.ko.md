# PR #2674: pre-auth handshake slot 제한, handshake 거부의 타입화, disaggregated role 인증

**날짜**: 2026-10-27
**상태**: 완료 (merge 전에 PR branch에서 작성한 보고서)
**언어**: Rust, Markdown
**위험도**: 높음 (disaggregated prefill, decode, router role 운영자에게 breaking change, handshake wire version 2로 상향)

## 요약

PR #2650은 모든 pipeline-parallel TCP 연결 앞에 pre-shared cluster secret을 쓰는 상호 HMAC-SHA256 handshake를 두었습니다. 그 리뷰에서 같은 subsystem에 세 가지 잔여 문제가 남았고, 이슈 #2655가 이를 항목 a, b, d로 묶었습니다 (항목 c는 #2658로 이동). 이 PR은 세 가지를 모두 해결합니다. (a) auth가 켜진 상태에서 handshake를 끝내지 않은 socket은 더 이상 512개의 `max_connections` permit 중 하나를 차지하지 않습니다. 대신 별도의 pending pool(전체 64, source IP당 8)에서 handshake 동안만 slot을 잡으므로, 아무것도 보내지 않는 socket을 대량으로 열어도 인증된 peer가 connection cap에서 밀려나지 않습니다. (b) advertised identity가 자신의 source IP와 다른 client는 이제 9바이트 거부 record를 받고 1초 안에 타입이 있는 `HandshakeRejected`로 실패합니다. 이전에는 맨 EOF를 보고 경고만 남겼습니다. server log는 "identity not owned", "secret 불일치 추정", "transcript 변조", "handshake 전 종료"를 구분합니다. wire version은 1에서 2로 올라갑니다. (d) disaggregated `--node-role prefill`, `decode`, `router` transport도 같은 handshake를 수행하며, 이 role들은 secret 없이 시작하지 않습니다. single-node serving(`hybrid` 또는 `--node-role` 없음)은 바뀌지 않았습니다.

## 1. 문제

이슈 #2655는 각 잔여 문제를 당시 코드에서 직접 확인했습니다 (아래 위치는 이슈가 인용한 PR 이전 기준입니다).

1. **pre-auth socket이 connection cap을 고갈시킴 (항목 a).** `accept_loop`는 spawn된 task가 `server_handshake`를 실행하기 전, accept 시점에 `try_acquire_owned`로 `max_connections` permit(기본 512)을 잡고 handshake 내내(최대 `HANDSHAKE_TIMEOUT`, 5초) 유지했습니다. 따라서 연결만 하고 아무것도 보내지 않는 socket 512개가 모든 permit을 점유할 수 있었고, 그 사이 접속한 실제 stage는 인증하기도 전에 끊겼습니다. 접속하는 쪽은 `PeerClosedBeforeHandshake`를 받았는데 `Transport::connect`는 이를 경고로만 처리하므로, coordinator는 시작한 뒤 요청에서 실패했습니다.
2. **identity binding 실패가 구별 가능한 타입 거부가 아니었음 (항목 b).** client 메시지 "server rejected the cluster handshake (secret mismatch or identity binding failed)"는 이름을 붙인 두 경우 어느 쪽에서도 도달하지 않았습니다. secret 불일치는 client가 먼저 발견하고(server tag가 검증되지 않음) 연결을 끊기 때문에, server는 "connection closed before the client authenticator arrived"만 남겼고 이는 네트워크 단절과 구별되지 않았습니다. identity binding 실패 시 server는 타입 없는 `anyhow` 오류를 반환하고 아무것도 쓰지 않았기 때문에, client는 server hello 전에 EOF(`PeerClosedBeforeHandshake`)를 받았고 `Transport::connect`는 실패 대신 경고만 남겼습니다. 기존 테스트 `handshake_rejects_client_advertising_foreign_ip`도 client가 5초 timeout에 걸려서야 끝났습니다. `VERDICT_REJECT` 경로는 server tag는 검증되고 client tag는 검증되지 않을 때, 즉 전송 중 transcript가 변조됐을 때만 도달할 수 있었습니다.
3. **disaggregated transport가 handshake를 하지 않았음 (항목 d).** prefill/decode worker(`model_worker.rs`의 `run_disaggregated_serving_role`)와 router(`startup.rs`의 `start_server`)는 `auth: None`으로 `TcpTransportConfig`를 만들었습니다. role port에 닿을 수 있는 host라면 누구나 role loop에 frame(예: `DecodeMetaFrame`)을 넣을 수 있었습니다. 문서는 trusted segment 모델을 설명하며 secret을 "opt-in으로" 연결하는 follow-up을 예고했습니다.

항목 c(`broadcast_beacon_loop`에 production sender가 없어 `--pp-auto --cluster-discovery=mdns`가 항상 timeout)는 이 PR 범위가 아니며, 이 PR에 의존하는 #2658에서 다룹니다.

## 2. 기술적 결정

### 항목 a: per-IP 제한만이 아니라 별도의 pending pool

`src/distributed/pending_handshakes.rs`에 `PendingHandshakeLimiter`(`pub(crate)`)를 추가하고, `TcpTransportConfig`에 `max_pending_handshakes`(기본 `DEFAULT_MAX_PENDING_HANDSHAKES = 64`)와 `max_pending_handshakes_per_ip`(기본 `DEFAULT_MAX_PENDING_HANDSHAKES_PER_IP = 8`)를 추가했습니다. auth가 켜졌을 때 accept 경로는 다음과 같습니다.

```text
accept -> pending.try_acquire(peer_ip)        거부: socket 종료, rate-limited warn
       -> task spawn: server_handshake (<= 5 s, HANDSHAKE_TIMEOUT으로 감쌈)
       -> pending slot 해제 (모든 결과에서)
       -> Ok(identity): conn_semaphore.try_acquire_owned()   가득 참: 종료, "max connections reached"
       -> handle_connection
```

이슈는 per-IP 제한 단독안을 두 가지 이유로 기각했습니다. 실제 stage와 공격자가 하나의 NAT 주소를 공유할 수 있고, 여러 source에서 오는 flood는 여전히 main pool을 고갈시킵니다. 512개의 인증된 slot을 인증되지 않은 peer가 건드리지 못하게 하는 것은 전역 pending pool이며, per-IP 하위 제한은 한 source가 pending pool 전체를 차지하지 못하게 할 뿐입니다.

유지보수자가 지켜야 할 구현 세부 사항:

- **획득 순서와 rollback.** `try_acquire`는 먼저 `std::sync::Mutex` 아래에서 per-IP count를 올리고 `PerIpGuard`로 감싼 뒤 전역 semaphore를 시도합니다. 전역 pool이 가득 차면 오류를 반환하면서 guard가 drop되어 count가 다시 내려갑니다. guard는 count가 0이 되면 key를 제거하므로 map에 지난 source가 남지 않습니다. lock은 `.await`를 넘어 유지되지 않습니다.
- **key 정규화.** per-IP key는 `peer_ip.to_canonical()`이므로 IPv4-mapped IPv6 source(`::ffff:a.b.c.d`)는 IPv4 주소로 집계됩니다. `bind_advertised_identity`가 쓰는 key와 같습니다.
- **slot 수명.** slot은 결과와 상관없이 `server_handshake`가 반환된 직후 해제되고, `server_handshake`는 전체 교환을 5초 timeout으로 감쌉니다. 일반적인 RAII이므로 timeout과 unwind에서도 해제된다는 점을 보안 리뷰가 확인했습니다.
- **0 bound.** bound가 0이면 accept 시점에 모든 inbound socket을 닫아, 아무도 받지 않는 listener가 조용히 만들어집니다. `PendingHandshakeLimiter::new`가 이를 거부하고, `TcpTransport::bind`는 listener를 bind하기 전에 limiter를 만들기 때문에 오류가 시작 시점에 드러납니다. `auth`가 설정된 경우에만 적용되며, auth가 없으면 두 field는 무시된다고 문서화되어 있습니다.
- **auth가 없으면 바뀌는 것이 없음.** `None` 분기는 이전과 똑같이 accept 시점에 `max_connections` permit을 잡고 limiter를 만들지 않으므로, `default_config_stays_unauthenticated`가 계속 의미를 가집니다.
- **재사용.** limiter는 메서드 두 개짜리 `pub(crate)` API이므로 #2658의 coordinator-to-peer listener가 같은 집계를 쓸 수 있습니다.

trade-off는 리뷰 후 문서에 명시되었습니다. 기본값 기준으로 8개 이상의 source IP에서 오는 flood는 64개 pending pool을 채울 수 있고, 가득 찬 동안에는 실제 peer의 재접속을 포함해 어떤 source의 새 연결도 slot이 비워질 때까지(slot당 최대 5초) accept 시점에 닫힙니다. flood는 이미 인증된 연결의 slot을 빼앗을 수 없고 frame을 주입할 수도 없습니다. `docs/distributed.md` 초안은 이런 flood가 "cannot starve authenticated ones"라고 적었는데, 구현 리뷰가 과장이라고 지적해 문구를 고쳤습니다.

### 항목 b: server proof 이전의 명시적 거부 record

server는 identity 불일치를 알리는 데 verdict byte를 재사용할 수 없습니다. verdict는 두 tag 뒤에 오고, #2650의 속성은 자신의 주소를 속이는 client가 server nonce나 authenticator를 절대 받지 않는다는 것입니다. 따라서 거부는 server hello를 대체해야 합니다. 이제 wire는 다음을 허용합니다.

```text
server -> client : REJECT_MAGIC "MLXCREJ\0"(8) | reason u8 (0x01 = REJECT_IDENTITY_NOT_OWNED)
```

- **`MAGIC`과 같은 길이.** `read_hello`는 이제 정확히 8바이트를 먼저 읽고(0바이트는 여전히 `PeerClosedBeforeHandshake`) 분기합니다. client role에서 `REJECT_MAGIC`이면 reason byte를 읽고 `HandshakeRejected("server refused this node's advertised identity <id>: the advertised IP must be the address the server sees this connection come from (check the bind or advertised address and any NAT or port forward between the nodes)")`를 반환합니다. server role에서 `REJECT_MAGIC`은 그냥 예상하지 못한 바이트입니다. 기존 `&str` role 인자 대신 `HelloFrom` enum(`Server { local_identity }` 또는 `Client`)이 role과 메시지에 들어갈 identity를 전달합니다.
- **고정된 내용.** record는 9바이트 상수이고 server가 관찰한 source IP를 담지 않습니다. client가 스스로 알 수 있는 것만 알려 주며, peer가 제어하는 데이터를 반사하지 않습니다.
- **관대한 해석.** reason byte가 없는 잘린 record도 identity 거부로 보고하고, 알 수 없는 reason code는 `HandshakeRejected("... reason code 0x..")`가 됩니다. 나중에 reason을 추가해도 client가 I/O 오류로 잘못 보고하지 않습니다.
- **타입이 정책을 결정.** `Transport::connect`는 `HandshakeRejected`를 치명적으로 처리합니다(잘못된 설정은 저절로 고쳐지지 않습니다). 그래서 advertised address가 잘못된 node는 이제 5초 후 경고하는 대신 connect 시점에 실패합니다.
- **version 상향.** `PROTOCOL_VERSION`이 1에서 2가 되었습니다(이제 `pub`). 서로 다른 version 쌍은 혼란스러운 바이트 불일치가 아니라 `read_hello`의 명시적 version 검사에서 실패합니다. 혼합 version은 원래 지원하지 않았습니다(discovery beacon이 `CARGO_PKG_VERSION`을 고정).

`accept_loop`의 기존 rate-limited log가 쓰는 server 측 분류:

| 분류 | 타입 | 발생 시점 | log |
|---|---|---|---|
| liveness probe | `PeerClosedBeforeHandshake` | client hello 전에 0바이트 | debug |
| identity not owned | `IdentityNotOwned { advertised, source_ip }` | 구체적인 advertised IP가 source IP와 다름, record 전송 | warn (rate-limited) |
| secret 불일치 추정 | `ClientAbortedAfterServerProof` | server tag를 보낸 뒤 client tag를 기다리는 중 EOF, reset, abort, broken pipe | warn (rate-limited) |
| server가 본 client secret 불일치 | `HandshakeRejected` | client tag 검증 실패 | warn (rate-limited) |

`VERDICT_REJECT`에 대한 client 메시지는 이제 "server rejected this node's authenticator although the server's own authenticator verified; the handshake was altered in transit"이며, 이 분기에 도달하는 유일한 경우를 설명합니다.

보안 리뷰에서 바뀐 점 (commit `865cbaaa`, 원래 `471320ca`):

- **parse할 수 없는 identity에는 record를 보내지 않음.** 첫 버전은 `bind_advertised_identity`의 모든 오류에 `REJECT_IDENTITY_NOT_OWNED`를 보냈기 때문에, 형식이 잘못된 identity도 client에게 IP 불일치로 보고되었습니다. 이제 `IdentityNotOwned`로 downcast되는 경우에만 record를 씁니다. `SocketAddr`로 parse되지 않는 identity는 어떤 mlxcel client도 보내지 않으므로 protocol 오류로 보고 응답 없이 연결을 닫습니다.
- **log escape.** parse 오류 메시지는 identity를 `{:?}`로 포맷합니다. 인증되지 않은 peer 입력이 operator log에 들어가므로, Debug 포맷으로 제어 문자를 escape해 peer가 log 줄을 위조하지 못하게 합니다.
- **비정상 종료.** `is_eof`가 `is_peer_abort`로 바뀌어 `anyhow` chain을 따라가며 `UnexpectedEof`, `ConnectionReset`, `ConnectionAborted`, `BrokenPipe`를 매칭합니다. 읽지 않은 데이터가 남은 채로 닫거나 `SO_LINGER` 0으로 닫는 client는 FIN 대신 reset을 만드는데, 둘 다 이제 `ClientAbortedAfterServerProof`로 분류됩니다.

record는 `TcpTransport`가 아니라 `server_handshake`가 직접 쓰므로 어떤 호출자든 같은 동작을 얻습니다. `client_handshake`, `server_handshake`, `HANDSHAKE_TIMEOUT`, 네 가지 오류 타입은 #2658을 위해 `mlxcel::distributed`에서 re-export됩니다.

### 항목 d: opt-in이 아니라 fail-closed

#2650 문서는 disaggregated role에 secret을 "opt-in으로" 추가하는 follow-up을 예고했습니다. 이 PR은 대신 pipeline role과 같이 필수로 만듭니다. opt-in은 기본 배포를 인증 없는 상태로 남기는데, 이것이 바로 이슈가 없애려던 상태이고, 두 종류의 distributed role이 서로 다른 보안 기본값을 갖게 됩니다.

- **resolver.** `resolve_disaggregated_cluster_auth(sources, mode)`는 `resolve_pipeline_cluster_auth` 옆에 있습니다. `Hybrid`는 어떤 source도 읽지 않고 `None`을 반환합니다. `PrefillOnly`, `DecodeOnly`, `Router`는 secret을 요구하며, 없으면 role flag와 `CLUSTER_SECRET_SOURCES_HELP`(모든 source)를 담은 오류로 실패합니다. source는 `ClusterSecretSources::from_env(--cluster-secret-file, [cluster] secret_file)`이므로 우선순위와 "두 env var 동시 설정" 오류는 #2650과 같습니다. TOML은 `Hybrid`가 아닐 때만 parse합니다.
- **위치.** `start_server`는 `resolve_distributed_startup`, peer probe, listener bind, weight load보다 먼저 resolver를 실행합니다. 이를 위해 serving mode 계산을 `build_server_config`에서 `resolve_serving_mode`로 분리했습니다. `build_server_config`는 `cluster_auth: None`을 두고 secret을 읽지 않으며, `start_server`가 값을 채웁니다.
- **전달 경로.** `ServerConfig::cluster_auth`가 `ModelProvider`(생성자 세 개에 인자 추가), `WorkerSchedulerConfig::cluster_auth`, `run_disaggregated_serving_role` 순으로 전달됩니다. worker는 optional이 아닌 auth를 받는 `disaggregated_role_transport_config(bind, Arc<ClusterAuth>)`로 transport를 만들고, coordinator 테스트도 같은 helper를 호출하므로 테스트가 production과 정확히 같은 설정을 bind합니다. router는 자신의 `TcpTransportConfig`에 `auth: config.cluster_auth.clone()`을 설정합니다. `TcpTransport`가 `auth`를 `ConnectionPool`까지 전달하므로 outbound frame(router에서 prefill, prefill에서 decode로 가는 handoff, 돌아오는 결과 frame)도 인증됩니다. `mlxcel-server`와 `mlxcel serve` 모두 `start_server`를 거칩니다.
- **worker의 두 번째 방어선.** `run_disaggregated_serving_role`이 `cluster_auth: None`을 받으면 인증 없는 role listener를 bind하는 대신 오류를 남기고 `false`를 반환합니다(잘못 설정된 role에 대한 기존의 single-node scheduler loop fallback). `start_server`를 거치면 도달할 수 없고, resolver를 건너뛰는 미래의 호출자를 잡기 위한 것입니다.
- **테스트 가능성.** decode loop의 입력부를 `ServingCoordinator::next_decode_meta`로 옮기고 `run_decode_role_networked`가 이를 호출합니다. `networked_decode_role_rejects_unauthenticated_frame`은 raw socket에서 올바르게 framing된 `DecodeMetaFrame`을 쓰고, socket이 닫히는 것과 `next_decode_meta`가 500ms 안에 아무것도 내놓지 않는 것을 `BatchScheduler` 없이 확인합니다.
- **좁게 유지한 범위.** 연결 인증은 외부자를 role port에서 막습니다. disaggregated protocol 내부의 frame 단위 출처나 correlation 검사(#2650 coordinator 검사에 해당하는 것)는 범위 밖입니다. 인증된 member의 frame은 계속 신뢰하며, `MLXCEL_DECODE_ALLOWLIST`(#389)는 바뀌지 않았습니다. continuation sequence number에 대한 `router_front.rs` 주석도 이제 이것이 liveness 보조이지 변조 방어가 아니라고 적습니다.
- **identity binding도 적용됨.** accept 쪽이 advertised identity를 TCP source IP에 묶기 때문에, 문서는 이제 `--serving-bind`가 peer에게 보이는 이 node의 접속 IP여야 하며 multi-homed host에서는 `0.0.0.0`이 아니어야 한다고 요구합니다. 불일치는 항목 b의 거부로 실패합니다.

router health monitor의 connect-and-close probe는 이미 `PeerClosedBeforeHandshake` debug 분기로 가며 그대로 유지됩니다.

## 3. 변경 요약

| 항목 | 값 |
|------|-------|
| 변경 파일 | 23 |
| 추가 줄 | +1,501 |
| 삭제 줄 | -186 |
| 추가 테스트 | 15 (수정 1) |

- **새 모듈**: `src/distributed/pending_handshakes.rs`(`PendingHandshakeLimiter`, `PendingHandshakeSlot`, `PendingSlotRefusal`, 기본값)와 `pending_handshakes_tests.rs`.
- **handshake**: `cluster_handshake.rs`에 `REJECT_MAGIC`, `REJECT_IDENTITY_NOT_OWNED`, `IdentityNotOwned`, `ClientAbortedAfterServerProof`, `HelloFrom`, `read_rejection_reason`, `is_peer_abort`, `PROTOCOL_VERSION = 2` 추가. `distributed/mod.rs`가 handshake API를 re-export.
- **transport**: `tcp_transport.rs`에 config field 두 개 추가, `bind`에서 검증, `accept_loop`를 인증 없는 경로(변경 없음)와 pending slot 경로로 분리, Security Notes 재작성.
- **disaggregated role**: `disaggregated/coordinator.rs`의 `disaggregated_role_transport_config`와 `next_decode_meta`. `ServerConfig`, `WorkerSchedulerConfig`, `ModelProvider` 생성자의 `cluster_auth`. `startup.rs`의 `resolve_serving_mode`, `resolve_disaggregated_cluster_auth`, `toml_cluster_secret_file`, router transport auth.
- **CLI와 문서**: 두 binary의 `--node-role`, `--cluster-secret-file` help. `docs/distributed.md`(fail-closed 범위, 실패 분류, pending pool 잔여 위험, disaggregated trust model과 실행 예시), `docs/environment-variables.md`, secret을 포함한 `docs/CONTINUOUS_BATCHING.md` 세 프로세스 예시.
- **test harness**: `tests/common::TestClusterSecretFile`(테스트마다 temp dir에 새 random secret). `disaggregated_router_e2e.rs`는 `--cluster-secret-file`을 덧붙이는 `ClusterRoles` helper로 모든 role을 띄우고, `disaggregated_serving_e2e.rs`는 테스트 내부 client transport에 같은 auth를 줍니다.

새 테스트: `client_reports_identity_rejection_as_typed_rejection`, `server_reports_client_abort_after_proof_as_probable_secret_mismatch`, `identity_rejection_sends_only_the_reject_record`, `unparseable_identity_closes_without_a_reject_record`, `server_reports_client_reset_after_proof_as_probable_secret_mismatch`, `peer_abort_classifies_eof_reset_and_broken_pipe` (handshake), `per_ip_limit_refuses_the_excess_and_releases_on_drop`, `global_limit_refuses_across_sources_and_does_not_leak_per_ip_counts`, `zero_bounds_are_refused_at_construction` (limiter), `pending_handshakes_do_not_consume_connection_slots`, `per_ip_pending_limit_closes_excess_sockets`, `pending_slot_is_released_after_handshake`, `bind_refuses_zero_pending_bounds_with_auth` (transport), `networked_decode_role_rejects_unauthenticated_frame` (coordinator), `disaggregated_roles_fail_closed_without_a_secret` (startup). `handshake_rejects_client_advertising_foreign_ip`는 이제 server에서 `IdentityNotOwned`, client에서 `HandshakeRejected`를 확인합니다.

| Hash (rebase 후) | 원래 | 메시지 |
|---|---|---|
| `c2a4c783` | `0bd014c4` | fix(distributed): cap pre-auth slots, type rejections, auth DI roles |
| `865cbaaa` | `471320ca` | fix(distributed): tighten handshake reject and pending-pool edges |
| `5d4df5b2` | `6e3da1da` | docs: document handshake edge cases from the final #2655 code |

## 4. Breaking change와 migration

`--node-role prefill`, `--node-role decode`, `--node-role router`는 cluster secret이 없으면 `the disaggregated <role> role (--node-role <role>) requires a cluster secret ...`와 모든 source 목록을 출력하고 시작 시점에 종료합니다. migration 방법:

```bash
openssl rand -hex 32 > /etc/mlxcel/cluster.secret
chmod 600 /etc/mlxcel/cluster.secret
# 같은 파일을 모든 prefill, decode, router host에 복사한 뒤 각 실행에 추가:
mlxcel-server ... --node-role decode --serving-bind 10.0.0.2:8304 \
    --cluster-secret-file /etc/mlxcel/cluster.secret
```

`MLXCEL_CLUSTER_SECRET_FILE`, `MLXCEL_CLUSTER_SECRET`, 또는 `--distributed-config`로 넘긴 TOML의 `[cluster] secret_file`도 pipeline role과 같은 우선순위로 동작합니다. `--serving-bind`는 peer가 보는 이 node의 접속 IP로 설정합니다. 모든 node를 함께 업그레이드해야 합니다. handshake wire version이 2가 되었으므로, 한쪽 node가 이전 release인 pipeline 또는 disaggregated 쌍은 version 메시지와 함께 실패합니다.

## 5. 남은 위험

- **pending pool flood가 새 연결을 지연시킴.** 설계상 8개 이상의 source IP에서 오는 flood는 전역 pending pool을 채울 수 있고, 그동안 실제 peer의 재접속을 포함한 모든 새 연결이 slot당 최대 5초씩 지연됩니다. 접속하는 쪽은 server hello 전에 socket이 닫히는 것을 보는데, 이는 `PeerClosedBeforeHandshake`이고 `Transport::connect`에서 타입 있는 실패가 아니라 경고일 뿐입니다.
- **IPv6 per-IP key가 전체 주소.** 하나의 /64가 source 주소를 바꿔 가며 per-IP 제한을 넘을 수 있습니다. 전체 합계는 전역 pool이 여전히 제한합니다. prefix 단위 key는 적용하지 않았습니다.
- **warn 예산 공유.** pending pool 거부와 handshake 실패가 `handshake_warn_allowed`(process 전체에서 분당 warn 5줄)를 공유합니다. flood 중에는 거부 log가 예산을 소진해 실제 secret 불일치나 identity 거부 줄이 debug로 밀릴 수 있습니다. flood를 조사할 때는 debug log를 봐야 합니다.
- **`ClientAbortedAfterServerProof`는 추론.** handshake의 바로 그 시점에 네트워크가 끊기거나 client가 crash하면 "most likely holds a different cluster secret"으로 보고됩니다. 그래서 메시지가 단정하지 않습니다.
- **인증 후 main cap이 가득 찬 경우.** 512개 `max_connections` permit이 모두 사용 중이면, 인증된 peer는 이미 `VERDICT_ACCEPT`를 받은 뒤 transport가 "max connections reached"로 연결을 닫으므로 client는 타입 있는 거부가 아니라 나중에 I/O 오류를 봅니다. 이전에는 handshake 전에 닫혔습니다. 둘 다 타입이 없으며, 인증된 동시 연결이 512개여야 발생합니다.
- **task panic 시 `active_conns`.** connection task가 panic하면 counter가 감소하지 않습니다. 이 PR 이전부터 있던 문제입니다.
- **router에는 두 번째 방어선이 없음.** worker는 auth 없는 role transport의 bind를 거부하지만, router는 `auth: config.cluster_auth.clone()`을 설정하고 `start_server` 앞부분에서 resolver가 fail-closed했다는 사실에 의존합니다. 미래의 refactor가 `start_server`를 거치지 않고 router transport를 만들면 인증 없이 bind됩니다.
- **frame이 아닌 연결 인증.** #2650과 같이 frame은 암호화되지도 MAC되지도 않고, secret을 가진 모든 member를 전적으로 신뢰하며, disaggregated protocol에는 frame 단위 출처 검사가 없습니다. role port에 대한 trusted segment 요구 사항은 문서에 남아 있습니다.
- **항목 c는 아직 열려 있음.** mDNS discovery에는 production beacon sender가 없습니다 (#2658).

## 6. 검증

아래 결과는 모두 PR 본문과 리뷰 코멘트에서 가져온 것으로, rebase 이전 head(`471320ca`, `6e3da1da`)에서 GB10 host spark-102, `--profile test-fast --features cuda`, GPU 작업은 `gpu-lock` 아래에서 기록되었습니다. 이 보고서를 위해 다시 실행한 것은 없습니다.

**failing first.** 수정 전 코드에서 `client_reports_identity_rejection_as_typed_rejection`은 "expected HandshakeRejected, got: peer closed the connection without starting the cluster handshake"로, `server_reports_client_abort_after_proof_as_probable_secret_mismatch`는 "cluster handshake timed out after 5s"로, `pending_handshakes_do_not_consume_connection_slots`는 "cluster handshake with 127.0.0.1:... failed"로 실패했습니다. 나머지 새 테스트는 새 field나 함수를 참조하므로 이전 코드에서는 compile되지 않습니다.

**lib suite와 lint.** `--test-threads=1`의 `distributed::`: 1390 passed, 0 failed, 4 ignored. `distributed::cluster_handshake::` 8회 반복: 매번 15/15 통과. `server::startup::`: 93 passed. `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check`, `make verify-env-var-docs` 모두 clean. `disaggregated_router_e2e`, `disaggregated_serving_e2e`, `pipeline_remote_real_models`, `pipeline_server_remote_real_models` target은 compile됩니다.

**수동 loopback (세 프로세스, qwen3-0.6b-4bit, `target/test-fast/mlxcel-server`).**

- router, prefill, decode에 같은 `--cluster-secret-file`: `POST /v1/chat/completions`가 HTTP 200 반환 (prompt 16, completion 16 token).
- decode node에만 다른 secret: HTTP 500 `decode handoff to 127.0.0.1:19465 failed`. prefill log는 client 쪽에서 본 secret 불일치를 보였고, decode log는 `client closed the connection after receiving this server's authenticator; it most likely holds a different cluster secret`을 보였습니다. 항목 b의 새 분류가 end to end로 동작한 것입니다.
- secret 없는 decode, router role: fail-closed 메시지와 함께 exit 1.

**실제 모델 pipeline 회귀.** `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`로 `pipeline_remote_runtime_qwen3_real_model_parity_and_cleanup`과 `pipeline_server_remote_coordinator_qwen3_matches_dense_baseline`이 `6e3da1da`에서 통과했습니다. version 2 handshake가 pipeline role을 여전히 문제없이 연결한다는 뜻입니다.

**rebase.** orchestrator가 branch를 main `f3b2efc0` 위로 rebase했습니다. `git range-diff`는 세 commit 모두 patch가 동일함을 보여 줍니다(`0bd014c4`에서 `c2a4c783`, `471320ca`에서 `865cbaaa`, `6e3da1da`에서 `5d4df5b2`). 그 범위의 upstream commit 중 #2668(CORS origin env var) 하나가 `src/server/startup.rs`, `src/main.rs`, `src/bin/mlx_server.rs`, `docs/environment-variables.md`를 건드리며, `src/distributed/`를 건드린 commit은 없습니다. 이 보고서를 작성할 때 rebase된 head에 대한 pre-merge check가 실행 중이었습니다.

**검증하지 않은 것.** `#[ignore]`된 disaggregated e2e harness 자체는 실행하지 않았습니다(compile만 확인). 수동 loopback 실행이 같은 세 프로세스 흐름입니다. 물리 host 두 대, node 사이의 NAT나 port forward, IPv6 source를 쓴 실행은 없습니다. throughput이나 latency는 측정하지 않았습니다. accept 경로에 추가된 연결당 작업은 짧은 mutex 구간 하나와 semaphore `try_acquire` 하나이며, frame 단위로 바뀐 것은 없습니다.

## 7. 학습 포인트

- **connection cap은 자격을 얻은 연결에만 계산한다.** 인증되지 않은 socket이 차지할 수 있는 공유 cap은 handshake timeout을 lockout 구간으로 바꿉니다. handshake가 반환되는 즉시 해제되는 작은 별도 pre-auth pool을 두면, 비싼 자원은 스스로를 증명한 peer에게만 돌아갑니다.
- **RAII로 부분 획득을 안전하게.** per-IP count를 먼저 잡고 guard로 감싸 두면, 전역 획득이 실패했을 때 반환과 함께 자동으로 되돌려지므로 잊어버릴 수 있는 별도의 undo 경로가 없습니다.
- **거부가 protocol이 숨기는 것을 흘려서는 안 된다.** identity 거부는 server hello 뒤가 아니라 그 자리를 대신하므로, 주소를 속이는 client는 여전히 nonce나 authenticator를 보지 못하고 record에는 상수 바이트만 들어갑니다.
- **첫 read로 메시지 종류를 결정한다.** 고정 길이 prefix를 먼저 읽으면 같은 stream 위치에 hello나 거부 중 하나가 와도 모호하지 않고, 같은 길이의 magic이 parser를 단순하게 유지합니다.
- **오류는 protocol의 어느 지점에서 일어났는지로 분류한다.** client tag를 기다리던 중의 EOF와 client hello 전의 EOF는 의미가 다릅니다. 이를 타입(`ClientAbortedAfterServerProof`와 `PeerClosedBeforeHandshake`)으로 나눈 것이, 구별되지 않던 log 줄을 조치 가능한 "secret 불일치 추정"으로 바꾸었습니다.
- **인증되지 않은 입력은 log에 쓰기 전에 escape한다.** peer가 보낸 문자열에 `{:?}`를 쓰는 한 글자 수정으로 log 줄 위조를 막을 수 있습니다.
- **보안 기본값은 opt-in보다 fail-closed.** opt-in secret은 기존 배포를 모두 그대로 둡니다. 필수로 만드는 비용은 명확한 시작 오류 메시지 하나와 문서화된 migration입니다.
