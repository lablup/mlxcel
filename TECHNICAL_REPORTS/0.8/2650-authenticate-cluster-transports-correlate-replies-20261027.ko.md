# PR #2650: cluster transport 인증과 stage 응답 correlation

**날짜**: 2026-10-27
**상태**: 완료 (merge 전에 PR branch에서 작성한 보고서)
**언어**: Rust, Markdown, Shell, YAML
**위험도**: 높음 (보안 수정, distributed pipeline role 운영자에게는 breaking change)

## 요약

`src/distributed`에는 인증 계층이 없었습니다. pipeline port에 닿을 수 있는 host라면 누구나 `pipeline_stage_response` frame을 보내 coordinator가 그것을 모델 출력으로 반환하게 만들 수 있었고, remote stage를 조종하거나 종료시킬 수 있었으며(malformed frame 하나로 service thread를 끝낼 수도 있었습니다), `--pp-auto`가 생성하는 cluster TOML에 UDP discovery beacon을 위조해 끼워 넣을 수 있었습니다. 이 PR은 cluster마다 하나의 pre-shared secret을 도입하고 이를 세 계층에서 사용합니다. pipeline role 사이의 모든 TCP 연결에 대한 상호 HMAC-SHA256 handshake, coordinator와 stage가 처리하는 모든 frame에 대한 exchange 단위 correlation과 출처 검사, 그리고 replay window가 있는 discovery beacon MAC입니다. 이제 모든 pipeline role은 secret 없이 시작하지 않습니다. single-node serving과 disaggregated prefill/decode/router role은 secret을 읽지 않으며 동작이 바뀌지 않았습니다. `--pp-auto`, `--pp-peer`, 또는 `pipeline_parallel_size >= 2`인 `--distributed-config`를 운영하는 경우 업그레이드 후 secret을 반드시 제공해야 합니다 (이슈 #2637).

## 1. 문제

이슈 #2637은 pipeline-parallel transport에 대해 세 가지 결함을 보고했고, 셋 모두 자격 증명 없이 네트워크에서 도달할 수 있었습니다.

1. **응답 주입.** `RemotePipelineRuntime::exchange`는 command를 보낸 뒤 `transport.recv()`가 처음 내놓는 `pipeline_stage_response` frame을 그대로 반환했습니다. 보낸 쪽을 무시했고(`let (_from, msg) = ...`), 대조할 request 식별자도 없었습니다. coordinator의 transport port에 TCP 연결을 열 수 있는 peer라면 임의의 logits를 담은 `RunResult`로 실제 stage와 경쟁할 수 있었고, 그 logits가 sampling이 쓰는 forward 출력이 되었습니다.
2. **stage 탈취와 crash.** `RemoteStageService`는 받은 모든 `pipeline_stage_command`를 출처와 상관없이 실행했습니다. `Shutdown`, `BeginDrain`, `CancelRequest`, `RunEntry`도 포함되며, 응답은 payload 안에 적힌 `reply_addr`로 보냈습니다. `serde_json::from_slice(&payload)?`가 parse 오류를 run loop 밖으로 전파했기 때문에 malformed frame 하나로 service thread가 끝났고, `serve_remote_pipeline_stage`는 살아 있는 service 없이 `ctrl_c`를 기다리며 남아 있었습니다.
3. **discovery 위조.** `--cluster-discovery=mdns` 모드의 `discover_peers`는 `cluster_name`이 일치하는(기본값은 잘 알려진 `mlxcel-cluster`) JSON datagram이면 무엇이든 받아들여 그 `peer_addr`를 생성된 TOML에 pipeline stage로 넣었습니다. `version` field는 호환성 검사라고 문서화되어 있었지만 실제로 검사하지 않았습니다.

transport 자체의 Security Notes도 인증을 하지 않으니 호출자가 신뢰할 수 있는 interface에 bind하거나 TLS를 위에 얹으라고 적고 있었습니다. 후자를 한 호출자는 없었습니다.

## 2. 기술적 결정

### 하나의 secret, 세 가지 용도

PR은 `src/distributed/cluster_auth.rs`에 세 타입을 추가합니다.

- `ClusterSecret`: 끝의 `\n` 또는 `\r\n` 하나를 정확히 잘라낸 뒤 16 byte 이상이어야 합니다(`MIN_CLUSTER_SECRET_BYTES`). `Debug`는 `ClusterSecret(<redacted>)`를 출력하고, `Display`는 없으며, `Drop`은 volatile write로 byte를 지웁니다(주석에 적힌 대로 best effort입니다). unix에서는 `from_file`이 group 또는 other가 읽을 수 있는 파일에 경고를 남기지만 거부하지는 않습니다.
- `ClusterSecretSources`: secret이 올 수 있는 모든 출처를 모아 한 함수에서 해석하므로, process 환경을 건드리지 않고 우선순위를 unit test할 수 있습니다. inline 값은 `Debug`에서 가려집니다(security review에서 수정, 이전에는 `Debug`를 derive했습니다).
- `ClusterAuth`: HMAC-SHA256 key(`ring::hmac`)와 `SystemRandom` CSPRNG입니다. `sign(label, message)`는 `HMAC(K, label || 0x00 || message)`를 계산하고, `verify`는 `ring::hmac::verify`로 constant-time 검증합니다. 용도마다 고유한 label(`mlxcel-cluster-auth-v1/server`, `mlxcel-cluster-auth-v1/client`, `mlxcel-discovery-beacon-v1`)을 쓰므로 한 용도로 만든 tag가 다른 용도에서 검증되는 일은 없습니다.

`ring`은 이미 rustls와 reqwest를 통해 모든 빌드에 들어 있었습니다. `Cargo.toml`은 기본 feature set의 `0.17`을 직접 의존성으로 추가해 기존 인스턴스와 통합되게 했고, 빌드에 새 crate가 늘지 않습니다.

node별 key나 TLS 대신 공유 secret을 고른 이유는 PKI 없이 세 결함을 모두 막는 가장 작은 변경이고, 한 cluster의 모든 role이 이미 하나의 신뢰 영역으로 운영되기 때문입니다. 대가는 role별 자격 증명이 없다는 점입니다(5절 참고).

### secret 출처와 우선순위

`ClusterSecretSources::resolve`는 설정된 첫 번째 출처를 다음 순서로 반환합니다.

1. `--cluster-secret-file <PATH>` (`mlxcel-server`와 `mlxcel serve` 모두).
2. `MLXCEL_CLUSTER_SECRET_FILE=<PATH>`.
3. `MLXCEL_CLUSTER_SECRET=<VALUE>`, secret 값 자체.
4. cluster TOML의 `[cluster]` 아래 `secret_file = "<PATH>"`. `ClusterConfig::from_file`은 상대 경로를 TOML 파일이 있는 디렉터리 기준으로 해석하므로 config와 secret을 함께 복사할 수 있습니다.

두 환경 변수를 모두 설정하면 조용히 하나를 고르지 않고 오류로 처리합니다. 둘이 서로 달라도 아무도 눈치채지 못할 수 있기 때문입니다. 빈 환경 변수 값은 설정되지 않은 것으로 봅니다. TOML에는 secret 값이 절대 들어가지 않습니다. `--pp-auto` coordinator가 생성된 TOML을 디스크에 쓰고, 운영자는 그 파일을 host 사이에 복사하기 때문입니다. secret이 파일에서 왔다면 `--pp-auto`는 그 파일을 생성된 TOML에 절대 경로로 기록합니다(`pp_auto_secret_file_reference`, `std::path::absolute` 사용). 입력된 경로를 그대로 기록하면 틀립니다. command line에 입력한 상대 경로는 작업 디렉터리 기준인데, TOML을 읽는 쪽은 TOML 디렉터리(기본값 `./.mlxcel/`) 기준으로 해석하기 때문입니다. implementation review가 이 문제를 찾았고 `pp_auto_records_secret_file_as_absolute_path_peers_resolve`가 이를 검증합니다.

### fail-closed 범위와 바뀌지 않는 것

세 진입점이 secret 없이 시작하지 않으며, 각 오류 메시지는 모든 출처를 나열합니다(`CLUSTER_SECRET_SOURCES_HELP`, `openssl rand -hex 32 > cluster.secret`도 안내합니다).

- `--pp-auto`: `resolve_pp_auto_cluster_auth`가 discovery socket을 bind하거나 TOML을 쓰기 전에 실행되므로, secret 없는 coordinator는 네트워크를 전혀 건드리지 않습니다(`pp_auto_bring_up_fails_closed_before_discovery_or_toml_write`).
- `pipeline_parallel_size >= 2`인 `--distributed-config`: `resolve_distributed_startup` 안의 `resolve_pipeline_cluster_auth`. `--pp-peer`도 coordinator가 생성한 TOML을 가리키는 `--distributed-config`를 요구하므로 이 경로를 지납니다.
- `Mdns` 모드의 `discover_peers`는 `ClusterAuth`가 주어지지 않으면 오류로 끝납니다.

타입 시스템도 이를 뒷받침합니다. `RemotePipelineRuntimeConfig.auth`와 `RemoteStageServiceConfig.auth`는 `Option`이 아닌 `Arc<ClusterAuth>`이므로 인증 없는 pipeline role은 만들 수 없습니다. `resolve_remote_pipeline_topology`는 `Option<&Arc<ClusterAuth>>`를 받고, pipeline 깊이가 1보다 큰데 `None`이면 internal error로 처리합니다.

바뀌지 않는 것: 기본 single-node serving, 한 process 안의 tensor parallelism, `pipeline_parallel_size <= 1`인 cluster config는 secret이 필요 없습니다(`resolve_pipeline_cluster_auth`가 `None` 반환). `TcpTransportConfig::auth`의 기본값은 `None`이고 `ConnectionPool::new`는 인증 없는 pool을 만들므로, disaggregated prefill/decode transport와 router는 기존의 인증 없는 동작을 유지합니다. static discovery는 secret을 건드리지 않습니다.

### 연결 시점 handshake

`src/distributed/cluster_handshake.rs`는 TCP 연결마다 한 번, transport frame을 읽거나 쓰기 전에 실행됩니다. 모든 정수는 big-endian입니다.

```text
client -> server : MAGIC "MLXCAUTH"(8) | version u16 = 1 | nonce_c [32] | id_len u16 | identity
server -> client : MAGIC(8) | version u16 | nonce_s [32] | id_len u16 | identity
                   | tag_s = HMAC(K, "mlxcel-cluster-auth-v1/server", transcript)
client -> server : tag_c = HMAC(K, "mlxcel-cluster-auth-v1/client", transcript || tag_s)
server -> client : verdict u8 (0x01 accept; 그 외 값이나 EOF는 거부)
```

`transcript`는 client hello와 server hello의 정확한 byte를 이어 붙인 것입니다. 두 nonce 모두 CSPRNG에서 나오므로 각 tag는 양쪽의 새 난수를 포함하고, 기록해 둔 tag를 새 연결에서 재사용할 수 없습니다. identity(각 쪽 listener의 `host:port`, 최대 255 byte)가 transcript 안에 있으므로 tag는 양쪽이 주장한 identity에 묶입니다. client tag는 `tag_s`도 덮고 두 방향이 서로 다른 label을 쓰므로, server tag를 client tag로 되돌려 쓸 수 없습니다. secret이 틀린 peer는 어느 쪽이 연결을 열었든 실패합니다. client는 secret에서 파생된 값을 보내기 전에 `tag_s`를 검사하고, server는 accept하기 전에 `tag_c`를 검사합니다.

server는 client가 아무것도 증명하기 전에 자신의 authenticator를 보냅니다. 이때 드러나는 것은 server 자신의 새 nonce를 포함한 transcript에 대한 MAC이며, label 분리와 client의 새 nonce 때문에 그 연결 밖에서는 쓸모가 없습니다.

**identity binding은 accept 쪽에서만.** accept 쪽은 frame을 어느 peer에게 귀속시킬지 정할 identity가 필요합니다. accept된 socket의 임시 `ip:port`는 설정된 어떤 peer도 가리키지 않습니다. `bind_advertised_identity`는 client가 알린 listener 주소를 TCP source IP에 묶습니다. 지정되지 않은 IP(`0.0.0.0`, 모든 interface에 bind한 listener가 알리는 값)는 source IP로 바뀌고, source IP와 다른 구체적인 IP는 거부됩니다. server는 응답하기 전에 이 검사를 하고, 불일치하면 아무 응답 없이 연결을 닫습니다. 그래서 자기 주소를 속이는 client는 server authenticator를 받지 못합니다. 묶인 identity는 그 연결에서 `recv()`가 반환하는 모든 frame의 `from`이 되고, runtime과 stage service는 이것을 기준으로 권한을 검사합니다.

connect 쪽은 server가 알린 identity를 socket IP에 일부러 묶지 않습니다. 어떤 주소로 연결했는지 이미 알고 있고, server tag가 secret 보유를 증명합니다. NAT나 port forward 뒤에 있는 stage는 client가 보는 주소와 다른 주소를 정당하게 알립니다. PR의 첫 버전은 client 쪽에서도 binding을 했지만, implementation review에서 그 검사를 사용하는 곳이 없고 forward된 stage를 깨뜨린다는 점이 드러나 제거되었습니다(`client_accepts_server_advertising_a_forwarded_address`).

**timeout.** 양쪽 모두 handshake 전체를 `HANDSHAKE_TIMEOUT`(5초)으로 감싸므로, 연결만 하고 멈춘 peer가 connection slot을 차지하는 시간은 최대 5초입니다. 이 PR 이전에는 인증되지 않은 유휴 socket이 slot을 무기한 차지했습니다.

**verdict byte.** 이것이 없으면 client는 첫 frame이 닫힌 socket에 부딪힌 뒤에야, 그것도 일반 I/O 오류로 거부를 알게 됩니다. 명시적인 verdict 덕분에 `client_handshake`는 연결 시점에 타입이 있는 `HandshakeRejected`를 반환할 수 있습니다. `TcpTransport::connect`(설정된 peer에 대한 pre-warm)는 `HandshakeRejected`를 치명적 오류로, 그 외 오류는 경고로 처리합니다. 아직 뜨지 않은 peer는 나중에 시작될 수 있지만 secret 불일치는 저절로 고쳐지지 않기 때문입니다. `RemotePipelineRuntime::new`가 `connect`를 호출하므로, coordinator는 첫 요청이 아니라 시작 시점에 불일치로 실패합니다.

**실행 위치.** accept 쪽은 connection별 task 안에서 `handle_connection`이 frame을 읽기 전에 `server_handshake`를 실행하므로, 인증되지 않은 연결의 frame은 `recv()`, RPC handler, `rpc_pending` 어디에도 도달하지 않습니다. connect 쪽은 유일한 outbound 연결 경로인 `ConnectionPool::connect_with_retry` 안에서 `client_handshake`를 실행하므로, pool된 연결과 재연결된 연결 모두 열릴 때 정확히 한 번 인증됩니다. Thunderbolt와 RDMA는 같은 TCP core를 감쌉니다. `ThunderboltTransportConfig::from_bind_address_with_auth`와 `RdmaTransportConfig.auth`가 auth를 core로 전달하고, `bind_transport`는 모든 backend에 대해 `auth`를 받습니다.

**로깅.** 한 byte도 보내지 않고 연결을 닫은 peer는 `PeerClosedBeforeHandshake`가 되어 debug로 기록됩니다. liveness probe(`probe_peers`, router health monitor)가 정확히 그렇게 동작하기 때문입니다. loopback E2E에서 이 probe가 거부된 handshake로 기록되는 것을 발견해 바꿨습니다. 다른 실패는 peer IP(secret 관련 값은 절대 아님)를 process 전체 기준 분당 처음 5건까지 warn으로, 그 이후는 debug로 기록합니다(`handshake_warn_allowed`).

### coordinator의 응답 correlation

`RemoteStageRequest`에 exchange마다 CSPRNG에서 새로 뽑는 `nonce: u64`(`ClusterAuth::random_u64`)가 추가되었고, `RemoteStageResponse`의 모든 variant(`Ack`, `RunResult`, `State`, `Error`)가 이를 그대로 돌려줍니다. 그래서 위조된 결과뿐 아니라 위조된 ack와 error도 거부됩니다. entry stage는 nonce를 `PendingEntryReply`에 저장해 두므로, reverse path 이후에 보내는 `RunResult`도 올바른 값을 담습니다.

`exchange`는 이제 `ExchangeMatch::reject_reason`이 `None`을 반환할 때만 frame을 받아들입니다.

- 인증된 `from`이 command를 보낸 stage와 일치하고(`PeerMatcher`),
- 돌려받은 nonce가 이 exchange의 nonce와 같고,
- `RunEntry`라면 `RunResult`의 `request_id`와 `micro_batch_id`가 같아야 합니다.

`PeerMatcher`(`src/distributed/peer_identity.rs`)는 설정된 `host:port`를 system resolver로 한 번 해석해 canonical socket address로 비교하므로, hostname으로 설정한 stage도 숫자 identity와 일치합니다. `--pp-auto`가 HTTP host가 `0.0.0.0`일 때 내보내는 `0.0.0.0:<port>`는 특정 host를 가리키지 않으므로, 해당 port를 가진 모든 인증된 identity와 일치합니다. 그 identity는 handshake에서 이미 실제 source IP에 묶여 있습니다.

거부된 frame은 기록되고(exchange마다 처음 세 건은 warn, 나머지는 debug) loop는 계속 기다립니다. 대기는 한 번 계산한 절대 deadline(`deadline = now + timeout`)을 쓰고, 각 `recv`에는 남은 시간만 줍니다. 이전에는 맞지 않는 frame이 올 때마다 `timeout`이 처음부터 다시 시작되어, 쓰레기 frame을 꾸준히 흘려보내면 exchange를 무한히 붙잡아 둘 수 있었습니다(`exchange_timeout_is_absolute_under_frame_flood`). exchange별 nonce 덕분에 앞서 timeout된 exchange에 대한 늦은 응답이 다음 exchange를 만족시키는 일도 없습니다.

`forward_sequence`는 logits를 반환하기 전에 검증합니다. rank 3, batch 1, sequence 길이는 입력과 같거나 1(마지막 위치만 내는 family), 그리고 모델의 vocabulary 크기입니다. `read_vocab_size`는 `config.json`의 `vocab_size`를 읽고, vision-language checkpoint를 위해 `text_config.vocab_size`로 fallback하며, 다른 config reader처럼 `sanitize_config_json`을 적용합니다. `startup.rs`와 `PipelineServerModel::load_remote`가 모두 `RemotePipelineRuntimeConfig.vocab_size`를 채웁니다. `vocab_size`가 없는 checkpoint도 rank, batch, sequence 검사는 받습니다. shape가 맞지 않으면 오류를 반환하기 전에 해당 request의 상태를 정리합니다.

### remote stage service 권한 검사

`RemoteStageServiceConfig`에 `coordinator_peer`가 추가되었고, cluster TOML에서 role이 pipeline stage가 아닌 첫 node로 해석됩니다(`pipeline_coordinator_address`). service는 이제 다음을 강제합니다.

- `pipeline_stage_command` frame은 coordinator에게서만 받습니다. 다른 peer의 command는 인증 여부와 관계없이 경고와 함께 버려지고 응답도 없습니다.
- forward activation은 설정된 upstream stage에게서만, reverse activation은 설정된 downstream stage에게서만 받습니다. 해당 이웃이 없는 stage는 activation을 버립니다.
- 응답은 `reply_addr`가 아니라 command의 인증된 출처(`from`)로 보냅니다. 이 field는 wire 호환성을 위해 envelope에 남아 있고, 불일치는 debug로 기록됩니다. coordinator의 command만 받으므로 출처는 구조상 coordinator입니다.

run loop는 이제 frame 단위 조건과 인프라 장애를 구분합니다. malformed JSON, `RunEntry`의 잘못된 wire tensor(`Error` 응답으로 답함), pending reply가 없는 activation은 기록만 하고 계속 서비스합니다. loop를 끝내는 것은 닫힌 transport channel, coordinator의 `Shutdown`, 로컬 shutdown signal뿐입니다. 이를 위해 `RunEntry` 처리를 `handle_run_entry`로 분리했습니다.

service thread는 종료될 때 신호를 보내고(`exited_rx`), `serve_remote_pipeline_stage`는 `ctrl_c`와 `handle.exited()`를 함께 select합니다. 예를 들어 이웃이 handshake를 거부해 service가 죽으면, stage는 이제 오류를 반환하고 process가 종료됩니다. supervisor가 재시작하지 않을 zombie로 남지 않습니다.

executor는 주입 가능해졌습니다(`spawn_with_executor`와 service thread 위에서 만드는 `ExecutorFactory`. `Send`가 아닌 MLX executor가 thread를 넘지 않습니다). 덕분에 control plane test를 모델 weight 없이 돌릴 수 있습니다.

쓰이지 않던 RPC control plane인 `install_stage_control_service`와 `StageLifecycleRequest`, `StageLifecycleResponse`는 re-export와 함께 삭제되었습니다. 호출자는 test뿐이었고, 권한 검사 없는 두 번째 control 경로를 library에 남겨 두면 나중에 누군가 연결할 여지가 생깁니다.

### discovery beacon MAC

`src/distributed/discovery_beacon.rs`는 `DiscoveryBeacon`에 `timestamp_ms`, `nonce`, hex `mac`를 추가합니다. MAC은 JSON을 다시 직렬화한 값이 아니라 나머지 모든 field의 canonical, length-prefixed 인코딩(`canonical_bytes`)을 덮으므로, key 순서나 공백이 MAC을 바꾸지 못합니다. `broadcast_beacon_loop`는 tick마다 새 timestamp와 nonce로 beacon에 다시 서명합니다. `BeaconVerifier::accept`는 다음 순서로 검사합니다.

1. datagram이 parse되고 이 cluster를 가리키는지(그렇지 않으면 무관한 traffic으로 보고 조용히 버림),
2. `version`이 이 빌드의 `CARGO_PKG_VERSION`과 정확히 같은지,
3. MAC이 constant-time으로 검증되는지,
4. timestamp가 수신 측 시계 기준 앞뒤 60초(`BEACON_MAX_SKEW`) 안인지,
5. 그 window 안에서 이미 받아들인 nonce가 아닌지,
6. 알린 `peer_addr` IP가 datagram source IP와 같은지, 또는 지정되지 않은 IP라서 source IP로 바뀌는지.

version 검사가 엄격한 이유는 stage wire protocol이 version 없는 JSON이고 버전이 섞인 cluster는 동작이 정의되지 않기 때문입니다. 예전에 best-effort라고 적었던 doc comment도 이제 강제한다고 적습니다. nonce는 MAC 검증을 통과한 뒤에만 replay 집합에 들어가므로 secret 보유자만 이 집합을 키울 수 있고, window를 벗어나면 정리됩니다. 이 cluster를 가리키는 거부는 discovery 실행 한 번에 20건까지 warn, 그 뒤로는 debug로 기록됩니다.

security review에서 처음 `unhex`가 공격자가 정한 `mac` 문자열을 byte offset으로 slice한다는 점이 드러났습니다. 홀수 offset에 multi-byte 문자가 있는 datagram 하나로 MAC 검사 전에 `--pp-auto` discovery가 panic했습니다. 이제 byte 단위로 처리하고 정확히 64자의 ASCII hex만 받습니다(`verifier_rejects_non_ascii_mac_without_panicking`).

`broadcast_beacon_loop`는 이 PR 전후 모두 tree 안에 production 호출자가 없습니다. `--pp-peer`는 coordinator가 생성한 TOML로 참여합니다. 서명 쪽은 test(`broadcast_beacon_loop_resigns_each_tick_and_resolves_against_coordinator`)가 다룹니다.

### listener 범위

handshake가 적용되는 곳: pipeline coordinator의 control transport, remote stage service transport, Thunderbolt와 RDMA wrapper(같은 TCP core).

범위 밖으로 문서화한 곳: disaggregated prefill/decode role transport(`disaggregated/coordinator.rs`)와 router front(`router_front.rs`). 별도의 opt-in 기능으로 자체 role matrix가 있고 지금은 secret을 전달하는 경로가 없습니다. `docs/distributed.md`는 이들이 trusted-segment 모델을 유지하며, 같은 secret을 opt-in으로 연결하는 것은 후속 작업이라고 명시합니다. `discovery.rs`와 router health probe는 연결만 합니다. heartbeat와 KV-cache 전송은 넘겨받은 transport를 그대로 씁니다. `bench_activation.rs`는 benchmark harness입니다.

## 3. 변경 요약

- 새 module: `cluster_auth.rs`(secret, 출처, HMAC과 CSPRNG), `cluster_handshake.rs`(wire protocol, identity binding, 타입이 있는 오류), `discovery_beacon.rs`(canonical 인코딩, 서명, `BeaconVerifier`), `peer_identity.rs`(`PeerMatcher`). 각각 test 파일이 있습니다.
- transport core: `tcp_transport.rs`의 `TcpTransportConfig.auth`, accept 쪽 handshake, rate-limit된 로깅. `connection_pool.rs`의 `ConnectionPool::with_auth`와 connect 쪽 handshake. `thunderbolt_transport.rs`, `rdma_transport.rs`, `transport_factory.rs`의 auth 전달.
- pipeline: `runtime.rs`(`StagePeer`, `ExchangeMatch`, 절대 deadline, `validate_logits_shape`, 필수 `auth`, `vocab_size`). `remote_service.rs`(coordinator, upstream, downstream matcher, frame 단위 오류 처리, 출처로 응답, nonce echo, `spawn_with_executor`, `exited`). `server_runtime.rs`(vocab 크기 채우기). `activation_transfer.rs`(쓰이지 않던 RPC plane 제거, -160줄).
- cluster config와 bring-up: `config.rs`의 `ClusterMeta.secret_file`과 TOML 기준 상대 경로 해석. `cluster_init.rs`의 `ClusterInitRequest.secret_file`, 인증하는 `discover_peers`, `broadcast_beacon_loop`의 재서명.
- server 시작과 CLI: `mlxcel-server`와 `mlxcel serve`의 `--cluster-secret-file`. `startup.rs`의 `resolve_pipeline_cluster_auth`, `pipeline_coordinator_address`, `pp_auto_secret_file_reference`, `resolve_pp_auto_cluster_auth`, `exited` select. `loading/mod.rs`의 `read_vocab_size`.
- 문서와 예제: `docs/distributed.md`에 residual risk를 포함한 "Cluster authentication" 절. `docs/environment-variables.md`에 두 환경 변수. 두 예제 TOML에 `secret_file = "cluster.secret"`, README 안내, `.gitignore` 항목. `scripts/benchmark_pipeline_remote_rollout.sh`는 secret이 없으면 `umask 077`로 난수 secret을 생성합니다.
- CI: pipeline-parallel workflow의 path filter와 `scripts/ci/run-pp-two-host-logical.sh`에 새 module을 추가했습니다. integration test는 `tests/common::test_cluster_auth`로 난수 secret을 공유하므로 commit된 secret은 없습니다.

diff는 57개 파일, 4,532줄 추가와 491줄 삭제이며 추가분 대부분은 test입니다.

## 4. Breaking change와 migration

모든 distributed pipeline role은 이제 secret 없이 시작 시점에 종료합니다. migration은 cluster마다 secret 하나를 만들고, 소유자만 읽을 수 있는 권한으로 모든 host에 복사한 뒤, 위 출처 중 하나로 제공하면 됩니다.

```bash
openssl rand -hex 32 > /etc/mlxcel/cluster.secret
chmod 600 /etc/mlxcel/cluster.secret
mlxcel-server ... --distributed-config cluster.toml --cluster-secret-file /etc/mlxcel/cluster.secret
```

버전이 섞인 cluster는 두 가지 이유로 동작하지 않습니다. 이전 빌드는 handshake를 하지 않으므로(새 쪽은 `MLXCAUTH` magic을 받지 못함) 연결이 실패하고, `--pp-auto` discovery는 다른 `CARGO_PKG_VERSION`의 beacon을 거부합니다. 모든 host를 함께 업그레이드해야 합니다.

## 5. 남은 위험

- **frame 단위가 아닌 연결 단위 인증.** handshake는 연결을 한 번 인증합니다. frame은 암호화되지도 MAC이 붙지도 않습니다. 이미 수립된 TCP stream을 가로챌 수 있는 on-path 공격자, 또는 peer의 연결을 받아 실제 server로 handshake를 중계할 수 있는 공격자는 범위 밖입니다. TLS나 frame 단위 MAC이 후속 작업입니다. cluster port는 신뢰할 수 있는 segment에 두어야 합니다.
- **인증 전 slot.** 인증되지 않은 socket은 handshake가 진행되는 동안 512개 연결 한도를 함께 쓰며, 5초 timeout으로 제한됩니다. flood는 정상 연결을 늦출 수 있지만 frame을 주입할 수는 없습니다. 이 PR 이전에는 그런 socket이 아무 제한 없이 slot을 차지했습니다. IP별 인증 전 한도가 후속 작업입니다.
- **일시적 handshake 실패에 대한 재시도 없음.** `connect_with_retry`는 handshake 오류가 나면 남은 backoff 시도를 쓰지 않고 바로 반환하므로, timeout이나 I/O 오류로 실패한 handshake는 그 자리에서 재시도되지 않습니다. secret 불일치를 재시도하지 않으려는 의도입니다. `Transport::connect`는 `HandshakeRejected`가 아닌 오류를 경고로 처리하고 이후 `send`가 새 연결을 열기 때문에 실패가 영구적이지는 않지만, backoff를 둔 재시도도 아닙니다.
- **환경 변수 상속.** `MLXCEL_CLUSTER_SECRET`은 process 환경에 남아 child process로 상속됩니다. 파일 출처를 권장하며 문서에도 그렇게 적었습니다.
- **인증 실패 후에도 coordinator가 살아 있음.** coordinator의 모델 load가 handshake 거부로 실패해도 HTTP server는 계속 실행되며 기존 no-model 경로로 응답합니다(loopback 실행에서 HTTP 501). completion을 내지는 않지만 process가 종료되는 것도 아니므로, process만 감시하는 supervisor는 알아채지 못합니다.
- **secret 하나라 role별 자격 증명이 없음.** secret 보유자는 누구나 handshake를 통과합니다. role 검사(command는 coordinator에게서만, activation은 이웃에게서만)는 source IP에 묶이고, identity 안의 listener port는 자기 신고입니다. 이 검사는 비보유자와 잘못 연결된 peer를 막지만, 침해된 cluster 구성원은 막지 못합니다. coordinator 주소가 `0.0.0.0:<port>`로 설정되어 있으면 그 port의 인증된 모든 identity와 일치합니다.
- **coordinator 선택.** `pipeline_coordinator_address`는 pipeline stage가 아닌 첫 node를 고릅니다. 한 TOML에 pipeline이 아닌 node가 여러 개 있으면, 실제 coordinator가 먼저 나오지 않는 한 stage가 그 command를 버립니다. implementation review가 LOW로 지적했고 이 PR에서는 다루지 않았습니다.
- **identity binding 실패는 client에서 타입이 없음.** server는 identity를 거부할 때 아무 응답도 하지 않으므로, client는 `HandshakeRejected`가 아니라 닫힌 연결을 보고 `Transport::connect`는 실패 대신 경고를 남깁니다. outbound source IP가 listener를 bind한 구체적인 IP와 다른 multi-homed host가 여기에 해당합니다.
- **disaggregated serving은 적용되지 않음** (2절 참고).

## 6. 검증

아래 결과는 모두 PR 본문과 review comment에 기록된 것으로, rebase 전 head `bbb11737`에서 측정되었습니다.

**lib suite.** `cargo test --profile test-fast --features cuda -p mlxcel --lib -- --test-threads=1 distributed::`가 1375개 test를 실패 0으로 통과했습니다. `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check`, `make verify-env-var-docs`도 깨끗했습니다.

**revert 증거.** script가 수정 하나를 끄고 rebuild한 뒤 해당 test를 이름으로 실행하고, 수정을 되돌려 rebuild하고 다시 실행했습니다. 수정을 끈 실행은 모두 실패했고(exit 101), 복원 후 다섯 개를 다시 돌린 실행은 통과했습니다(exit 0).

| 결함 | 끈 수정 | 실패한 test |
|---|---|---|
| 1, 응답 주입 | `ExchangeMatch::reject_reason`이 `None` 반환 | `runtime::correlation_tests::exchange_drops_run_result_from_non_stage_source` (공격자 node의 위조 logits가 forward 출력이 됨) |
| 2, 인증 없는 control | accept 쪽 handshake와 coordinator 출처 검사 생략 | `remote_service::tests::unauthenticated_control_frame_is_rejected_before_dispatch` (raw socket의 `Shutdown`이 stage를 멈춤) |
| 2, malformed frame | frame 단위 `Err` 분기가 다시 `Err` 반환 | `remote_service::tests::malformed_json_frame_does_not_end_service` |
| 3, beacon 위조 | MAC 검증 생략 | `cluster_init::tests::discover_peers_rejects_spoofed_beacon_without_valid_mac`, `discovery_beacon::tests::verifier_rejects_spoofed_beacon_without_mac` |

review 후속 수정도 같은 방식으로 확인했습니다. `verifier_rejects_non_ascii_mac_without_panicking`은 이전 decoder에서 panic하고, `4b6bd035`(rebase 후 `c077ecc5`)에서 추가된 각 test는 이전 production 코드에서 실패했습니다. 이 밖의 test는 잘못된 nonce, 잘못된 `request_id`, 잘못된 vocab 차원, frame flood 아래의 절대 deadline, 위조된 `State` frame, 인증된 rogue peer가 보낸 `CancelRequest`, `BeginDrain`, `RunEntry`, `Shutdown`(효과 없음, 응답 없음, executor 호출 없음), `reply_addr`가 아닌 출처로의 응답, 이웃이 아닌 쪽의 activation, 잘못된 entry tensor, 양방향의 틀린 secret, 일반 연결과 인증 연결의 불일치, beacon 변조, 오래되거나 미래인 timestamp, 재사용된 nonce, source IP 불일치, version 불일치, secret 우선순위와 가림 처리를 다룹니다.

**실제 모델.** `gpu-lock` 아래에서 qwen3-0.6b-4bit로 `pipeline_remote_runtime_qwen3_real_model_parity_and_cleanup`(인증된 remote stage 두 개)와 `pipeline_server_remote_coordinator_qwen3_matches_dense_baseline`이 통과했습니다. 두 번째 test는 vocab 크기 검사가 실제 모델 출력을 받아들인다는 것을 보여 줍니다.

**loopback E2E.** `target/test-fast` binary로 coordinator 127.0.0.1:19370, stage 19371과 19372, HTTP 19380에서 19382까지 사용했습니다.

- 같은 `secret_file`: `/v1/chat/completions`가 completion과 함께 HTTP 200을 반환했고, liveness probe에 대한 handshake 경고는 없었습니다.
- stage-1만 다른 secret: stage-1이 시작 시점에 `cluster authentication with peer 127.0.0.1:19371 failed: ... (secret mismatch)`로 종료했고(secret이 맞는 upstream의 거부를 감지), stage-0은 `Rejecting connection from 127.0.0.1: cluster handshake failed`를 기록했으며, coordinator는 종료된 stage에 닿지 못했습니다(HTTP 503). 두 robustness 변경(`exited` select와 liveness probe의 debug 기록) 이전의 첫 실행에서는 stage들이 양방향으로 불일치를 기록했고, coordinator는 `Failed to load model: cluster authentication with peer 127.0.0.1:19372 failed`를 기록한 뒤 no-model 경로로 HTTP 501을 반환했습니다.
- 어떤 출처에도 secret 없음: coordinator가 `--cluster-secret-file`, 두 환경 변수, `secret_file`을 나열하는 메시지와 함께 exit 1로 종료했습니다.

실제 모델 test와 E2E는 review 변경 후 rebuild한 binary로 다시 실행했고 결과는 같았습니다.

**rebase.** 이 보고서 작성 전에 branch를 main `57b54b0f` 위로 rebase했습니다. `git range-diff`에서 네 commit 모두 patch가 동일하고(`b5b43ffc`에서 `a0e175b9`, `9e78bf71`에서 `3dacaf7b`, `4b6bd035`에서 `c077ecc5`, `bbb11737`에서 `6a52a698`), 새로 들어온 upstream commit은 `src/distributed/`, `src/server/startup.rs`, `src/loading/`, `Cargo.toml`을 건드리지 않습니다. rebase 후 이 보고서 head와 소스가 같은 rebase된 트리를 GB10 호스트에서 `--profile test-fast --features cuda,test-utils`로 다시 검증했습니다: `cargo clippy --lib --tests -D warnings` exit 0; lib selector `distributed::`, `server::startup`, `loading::`, `server::cli_input`, `commands::serve` 1937 통과, 0 실패; `distributed_integration` 20 통과; `zero_config_cluster_init` 2 통과(1 ignored); `pipeline_e2e` 30 통과; `make verify-fmt`와 `make verify-env-var-docs` 통과. 실모델과 loopback 실행은 rebase된 head에서 다시 하지 않았습니다.

**검증하지 않은 것.** 물리 host 두 대, 또는 auth를 켠 Thunderbolt나 RDMA hardware로 실행한 적은 없습니다. 이 wrapper들은 auth가 공유 TCP core로 전달된다는 점으로만 다뤄집니다. mDNS discovery는 실제 broadcast domain이 아니라 로컬 UDP socket으로 test했습니다. throughput이나 latency는 측정하지 않았습니다. 구조상 요청당 비용은 exchange마다 8 byte CSPRNG 읽기 한 번과 matcher 비교이고, handshake는 pool이 연결을 열 때만 실행됩니다.

## 7. 학습 포인트

- 연결을 인증해도 애플리케이션이 여전히 payload를 믿는다면 충분하지 않습니다. handshake는 외부자를 막지만, coordinator에는 여전히 출처, nonce, request 검사가 필요했고, stage는 frame 안에 적힌 주소로 응답하는 일을 멈춰야 했습니다. 각 계층이 다른 계층이 남긴 틈을 막습니다.
- parse 실패 하나가 liveness 실패가 되지 않게 해야 합니다. service loop 안에서 신뢰할 수 없는 입력에 `?`를 쓰면 malformed frame이 원격 kill switch가 됩니다. frame 단위 오류(기록 후 계속)와 인프라 오류(loop 종료)를 나누고, loop가 끝나면 supervisor가 알아채게 해야 합니다.
- 무관한 메시지가 올 때마다 다시 시작되는 timeout은 timeout이 아닙니다. deadline은 한 번 계산하고 매 대기에는 남은 시간을 줍니다.
- identity는 그것을 사용하는 쪽에서 묶습니다. accept 쪽은 frame을 peer에게 귀속시키므로 알린 identity를 source IP에 묶습니다. connect 쪽은 이미 어디로 연결했는지 알고, 거기서 묶으면 NAT만 깨집니다.
- 타입이 있는 오류는 정책을 담습니다. `HandshakeRejected`와 일반 I/O 오류의 구분 덕분에 transport는 secret 불일치에서 빨리 실패하면서도 아직 시작하지 않은 peer는 기다릴 수 있습니다.
- 공격자가 정한 텍스트는 byte로 parse합니다. `&str`을 byte offset으로 slice하면 multi-byte 문자에서 panic하고, 여기서는 그 panic이 MAC 검사보다 먼저 일어났습니다.
- 생성된 config의 상대 경로는 작성자의 작업 디렉터리가 아니라 읽는 쪽이 해석하는 기준에 맞춰야 합니다.
