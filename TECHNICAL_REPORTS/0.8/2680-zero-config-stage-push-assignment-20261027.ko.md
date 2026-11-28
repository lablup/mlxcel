# 기술 리포트: PR #2680 - --pp-peer 프로세스로 stage assignment push

**작성일**: 2026-10-27  
**상태**: 완료 (머지 전 작성; PR은 draft 상태로 열려 있음)  
**언어**: Rust  
**위험도**: 중간

## 요약

PR #2680은 zero-config pipeline bring-up을 처음부터 끝까지 동작하게 만든다. 이제 `mlxcel-server --pp-auto N`이 인증된 cluster handshake 위에서 N개의 `mlxcel-server --pp-peer` 프로세스에 stage assignment를 push하며, static과 mdns discovery 모두에서 동작한다. peer는 더 이상 손으로 복사한 cluster TOML이나 추측한 `--node-id`가 필요 없다. assignment 프로토콜은 새 모듈 `src/distributed/stage_assignment.rs`의 2단계 Offer/Accepted/Commit/Committed 교환이고, 기존 stage 시작 경로는 그대로 재사용한다. GB10 호스트에서 qwen3-0.6b-4bit 3-프로세스 클러스터는 두 discovery 모드 모두에서 단일 노드 서버와 같은 greedy `/v1/chat/completions` 텍스트를 반환했다.

## 1. 문제 정의

이 PR 이전에 zero-config 플래그는 존재하지 않는 흐름을 설명하고 있었다 (이슈 #2658):

- **mdns discovery는 끝날 수 없었다.** `broadcast_beacon_loop`를 호출하는 프로덕션 코드가 없어 어떤 프로세스도 beacon을 보내지 않았고, coordinator는 항상 30 s discovery timeout 후 실패했다. sender가 있었더라도 coordinator는 UDP receiver를 unicast control IP에 bind했는데, Linux와 macOS에서 그런 소켓은 `255.255.255.255`로 보낸 datagram을 받지 못한다.
- **`--pp-peer`는 사실상 수동 경로였다.** `--distributed-config`(coordinator에서 복사한 TOML)와 `--node-id`가 필요했고, 후자는 `plan_cluster`가 peer 주소의 사전순으로 정하는 `stage-{i}`와 같아야 했다. 운영자가 이를 추측해야 했다.
- **help 텍스트가 구현되지 않은 동작을 약속했다**: coordinator에 등록하고 coordinator가 돌려주는 stage assignment를 사용한다는 설명.
- **coordinator가 생성한 TOML은 다른 호스트에서 쓸 수 없었다.** coordinator의 절대 경로 `secret_file`이 들어 있었고, `--host 0.0.0.0`이면 coordinator가 `0.0.0.0:19000`으로 기록되어 stage의 `PeerMatcher`가 해당 포트의 인증된 아무 멤버에게서나 명령을 받아들이게 되었다.

방향 자체가 의도적인 번복이었다. #2655는 mdns 모드 제거를 계획했으나, 그 제거를 철회하고 동작하게 만드는 쪽을 택했다. 이 PR은 #2655의 protocol v2 handshake(typed identity rejection, `PendingHandshakeLimiter`)를 확장하지 않고 그 위에 쌓는다.

## 2. 기술적 선택과 그 이유

### 2.1 peer가 pull하지 않고 coordinator가 push

static 모드에서는 coordinator만 peer 주소(`--cluster-peers`)를 알고, mdns 모드에서는 coordinator가 beacon으로 주소를 배운다. 두 경우 모두 계획은 coordinator가 갖고 있으므로 coordinator가 각 peer의 listener에 접속해 assignment를 push한다. peer는 자신의 listen 주소만 알면 된다. pull 방식이었다면 static 모드에서 peer가 coordinator 주소를 알아야 하는데, 이는 플래그가 배포되는 방향과 반대다.

### 2.2 기존 handshake 위의 2단계 commit

| 선택지 | 결과 |
|---|---|
| Offer/Ack 한 번 | peer 2에서 실패하면 peer 1은 종료될 coordinator를 위해 stage를 실행한 채 남음 |
| **선택: Offer 후 모든 peer가 수락했을 때만 Commit** | offer 단계가 실패하면 어떤 peer도 commit하지 않음; 실패는 peer별로 보고 |
| abort 메시지가 있는 완전한 2PC | commit 단계 중 실패도 처리하지만 프로토콜 상태가 늘어남 |

채널은 `client_handshake`/`server_handshake`를 재사용한다. coordinator는 control 주소를, peer는 listener 주소를 advertise하고, peer는 stage transport와 똑같이 coordinator의 identity를 TCP source IP에 bind한다. handshake 이후 frame은 `u32` big-endian 길이와 JSON이며 64 KiB 상한(할당 전에 검사)이 있다. frame이 작고 드물며 진화해야 하므로 바이너리 stage 프로토콜 대신 JSON을 택했고, `ASSIGNMENT_PROTOCOL_VERSION`이 handshake와 독립적으로 버전을 관리한다.

commit 단계에는 남은 틈이 있다. `Commit`과 `Committed` 사이에서 어떤 peer가 실패하면 이미 commit한 peer는 commit된 채로 남고 coordinator는 종료된다. abort frame으로 해결하지 않고 알려진 한계로 문서화했다.

### 2.3 peer는 전달받은 내용을 스스로 다시 계산한다

offer에는 `ModelIdentity`(`config.json`의 SHA-256, 정렬된 `*.safetensors` 이름과 바이트 크기, layer 수)와 coordinator가 해당 stage에 대해 계산한 layer range가 들어 있다. TOML에는 layer range가 없고 각 stage는 여전히 자기 checkpoint로 partition을 계산한다. offer의 layer range는 두 번째 identity 검사로만 쓰여서, 나중에 hash 검사가 약해지더라도 partition을 움직이는 config 변경을 잡아낸다. `ModelIdentity`는 weight 바이트를 읽지 않아 큰 checkpoint에서도 bring-up이 빠르지만, config가 같고 shard 크기가 같은데 내용이 다른 두 checkpoint는 일치한다고 판단된다.

검증은 고정된 순서로 진행된다: protocol, mlxcel 버전, cluster 이름, model, layer 수 이하의 stage 수, layer range, `node_id == "stage-{index}"`, TOML 구조(우리 stage가 우리 listener 포트에 있고 coordinator node가 정확히 하나), 그리고 offer의 `coordinator_control_addr`와 TOML의 coordinator node를 handshake로 bind된 identity와 비교. protocol, 버전, model, layer range 불일치는 같은 coordinator로 재시도해도 성공할 수 없으므로 치명적(`AssignmentRefused`)이다. 이름, 주소, TOML 문제는 peer를 listening 상태로 되돌려, 잘못 설정된 coordinator가 올바르게 설정된 peer를 종료시키지 않게 한다.

자기 node 검사는 IP가 아니라 포트만 비교한다. coordinator가 아는 peer 주소는 peer가 bind한 주소(예: peer 쪽의 `0.0.0.0`)와 다를 수 있고, `localize_cluster`가 이후 peer 자신의 주소를 listener 주소로 바꾼다.

### 2.4 node id는 파일 이름이다

peer는 받은 cluster를 `create_dir_all`과 함께 `.mlxcel/cluster-<node_id>.toml`에 쓴다. TOML validator는 id에 `/`와 `..`를 허용하므로, 검사하지 않은 id는 coordinator 역할을 하는 secret 보유자가 `.mlxcel/` 밖에 쓰게 만들 수 있었다. 보안 검토에서 `verify_offer`가 id를 정확히 `stage-<stage_index>`로 제한했고, startup은 경로를 만들기 전에 `[A-Za-z0-9_-]`를 다시 검사한다. 같은 검토에서 TOML의 coordinator node도 bind된 identity와 일치해야 하도록 했다. stage가 신뢰하고 응답하는 주소는 offer 필드가 아니라 그 node이기 때문이다.

### 2.5 timeout은 독립 설정이 아니라 하나의 예산

| 상한 | 값 | 비고 |
|---|---|---|
| Handshake | 5 s | #2655에서 정의 |
| Frame I/O | 10 s | commit 대기를 제외한 모든 frame 읽기 |
| `Accepted` 후 peer의 `Commit` 대기 | 60 s | 10 s frame 상한 없이 읽음 (c8d257cb에서 수정) |
| Coordinator push | 60 s | offer 단계는 10 s 일찍 종료 (`commit_reserve`) |
| peer의 assignment 대기 | 600 s | 이후 주소, 모드, cluster를 명시한 메시지로 종료 |

이 영역에서 리뷰 중 결함 두 개가 발견되었다. 첫째, 처음 버전은 `Commit`을 `read_frame`으로 읽었는데 그 10 s 상한이 60 s commit timeout보다 먼저 만료되어, coordinator가 나중 peer에 닿는 데 10 s 넘게 걸리면(늦게 시작한 static peer) 먼저 수락한 peer가 포기했다. 둘째, offer와 commit이 하나의 deadline을 공유해서 늦은 `Accepted`가 `Commit`에 쓸 시간을 남기지 않고 peer를 둘로 갈라놓을 수 있었다. 이제 `commit_reserve(timeout) = min(ASSIGNMENT_IO_TIMEOUT, timeout / 2)`가 전체 60 s 안에서 commit 단계에 frame 하나 분량의 시간을 보장한다.

### 2.6 connect와 handshake 재시도

coordinator는 offer deadline까지 500 ms마다 connect와 handshake를 재시도하고, `HandshakeRejected`(secret 또는 identity 불일치)에서만 일찍 멈춘다. 하나의 루프로 두 경우를 처리한다. coordinator보다 늦게 시작한 static peer, 그리고 인증 전 슬롯(전체 8개, IP당 4개)이 인증되지 않은 소켓에 점유되어 coordinator의 소켓을 accept 시점에 닫는 peer다. 재시도가 없으면 네트워크의 어떤 호스트든 그 슬롯을 점유해 bring-up을 실패시킬 수 있었다.

### 2.7 이슈에 기록된 기각된 대안

- bind된 assignment listener를 `RemoteStageServiceHandle::spawn`에 넘기는 방식: 그 함수는 자체 스레드와 런타임에서 bind하고 `RemoteStageServiceConfig`에 `Clone`이 아닌 필드가 필요해진다. 대신 listener를 drop하고 stage service가 포트를 다시 bind한다(tokio는 Unix에서 `SO_REUSEADDR`를 설정). coordinator의 첫 stage 연결이 이 재bind와 경쟁할 수 있지만 `Transport::connect`가 이미 lazy reconnect로 이를 허용한다.
- coordinator 쪽 broadcast target: coordinator는 수신만 하므로 `--cluster-discovery-target`은 peer 플래그다.

## 3. 변경 요약

| 영역 | 변경 |
|---|---|
| `distributed::stage_assignment` (신규, 1292줄) | frame, `ModelIdentity`, `verify_offer`, `localize_cluster`, `await_stage_assignment`(peer), `push_stage_assignments`(coordinator), typed `AssignmentRefused`와 `AssignmentPushFailed` |
| `distributed::cluster_init` | `broadcast_beacon_loop`가 하드코딩된 limited broadcast 대신 `target: IpAddr`를 받음; 문서에서 mdns를 "future zeroconf"라 부르지 않음 |
| `distributed::tcp_transport` | assignment listener가 같은 rate limit을 쓰도록 `handshake_warn_allowed`를 `pub(crate)`로 변경 |
| `server::startup` | Coordinator: receiver를 unspecified 주소에 bind, non-loopback peer와 loopback control 주소 조합 거부, `plan_cluster`와 TOML 쓰기 사이에서 push(`--dry-run`에서는 생략). Peer: fail-closed secret, listen 주소 결정, assignment 대기, localize된 TOML 쓰기, `distributed_config`/`node_id` 설정으로 `resolve_distributed_startup`은 그대로. `--pp-peer --dry-run`은 bind 전에 거부 |
| CLI (`mlxcel-server`, `mlxcel serve`) | 새 `--cluster-discovery-target`(기본값 `255.255.255.255`); `--pp-auto`, `--pp-peer`, `--cluster-*` help 재작성; `--pp-peer`에서 `--cluster-control-addr`는 peer의 listen 주소 |
| `docs/distributed.md` | "Zero-config bring-up" 안내: secret 배포, static과 mdns 3-호스트 예시, 방화벽 포트, timeout, `--dry-run` 제한 |
| 테스트 | `stage_assignment` 단위 테스트 14개, startup 테스트 3개, weightless loopback bring-up 2개, ignored 실제 바이너리 parity 테스트 2개 |

운영자가 보는 동작 변화: `--distributed-config` 없는 `--pp-peer`는 이제 실패하지 않고 assignment를 기다린다. static 모드는 구체적인 포트가 있는 `--cluster-control-addr`를 요구한다. 모든 `--pp-peer` 프로세스는 무엇이든 bind하기 전에 cluster secret이 있어야 한다. `--pp-peer --distributed-config --node-id`는 수동 경로를 유지한다. 새 환경 변수는 없고 handshake 프로토콜도 바뀌지 않았다.

통계: 16개 파일, +3107/-125.

| Commit | 메시지 |
|---|---|
| `cce9b212` | feat(distributed): push stage assignments to --pp-peer processes |
| `c8d257cb` | fix(distributed): wait the full commit timeout for Commit on a peer |
| `0cb9fd38` | fix(security): bind assignment node ids and coordinator to the handshake |
| `482f5a5c` | fix(distributed): harden zero-config bring-up edge cases |
| `0e3fd2d2` | Merge origin/main |

## 4. 학습 포인트

**broadcast 수신은 bind 주소에 달려 있다.** Linux와 macOS에서 unicast IP에 bind한 UDP 소켓은 `255.255.255.255`로 온 datagram을 보지 못하므로 receiver는 unspecified 주소에 bind해야 한다. limited broadcast는 default route 인터페이스로만 나가고 loopback을 건너지 않는다. 그래서 `--cluster-discovery-target`이 directed subnet broadcast나 unicast IP를 받고, loopback 테스트는 target으로 `127.0.0.1`을 쓴다.

**실제 broadcast 실행으로 두 측면을 모두 확인했다.** 기본 target으로 실행했을 때 `0.0.0.0`에 bind한 coordinator receiver는 peer의 limited-broadcast beacon을 실제로 받았다. 그러나 datagram이 LAN 인터페이스(source `10.100.64.122`)로 나가면서 loopback listener를 advertise했기 때문에, `BeaconVerifier`가 30 s discovery timeout까지 매초 거부했다("beacon advertises 127.0.0.1 but was sent from 10.100.64.122"). 이는 #2650의 source-IP binding이 설계대로 동작한 것이고, 완전한 broadcast bring-up에는 stage listener가 LAN 주소에 있어야 한다.

**안쪽 timeout은 바깥 timeout을 조용히 줄일 수 있다.** c8d257cb의 버그는 일반적인 패턴이다. 자체 호출별 timeout이 있는 helper를 더 긴 `tokio::time::timeout`으로 감싸면 긴 쪽이 아니라 짧은 쪽 상한이 적용된다. 수정은 다른 모든 곳에서 `read_frame`의 상한을 유지하고, 상한이 바깥에서 오는 한 호출에만 `read_frame_unbounded`를 추가했다.

**rollback이 있는 commit 점유.** peer는 `Commit`을 읽은 뒤 `AtomicBool`의 `compare_exchange`로 단 하나의 commit을 점유하고, `Committed` 쓰기가 실패하면 점유를 해제한다. `Committed`를 받지 못한 coordinator는 push를 실패 처리하고 그 stage를 구동하지 않기 때문이다.

## 5. 검증

최종 head, GB10 spark-102, `--profile test-fast --features cuda`, `--test-threads=1`, GPU 작업은 `gpu-lock run --tag issue-2658` 아래에서 실행:

- `distributed::stage_assignment` lib 테스트: 14 통과. `server::startup`: 97 통과.
- `zero_config_cluster_init`: 4 통과(static과 mdns weightless bring-up), 3 ignored.
- `cli_help_consistency`: 27 통과. Clippy `-D warnings`, `cargo fmt --check`, `make verify-env-var-docs`: 문제 없음.
- 이 브랜치의 이전 head: `distributed::` lib 1411 통과; ignored `zero_config_cli_{static,mdns}_matches_single_node`는 `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`로 통과.

최종 head의 실제 바이너리 E2E(qwen3-0.6b-4bit, `MLXCEL_SDPA_DETERMINISTIC=1`, loopback 포트 19500-19549, `temperature: 0`, `max_tokens: 24`): mdns(target `127.0.0.1`)와 static 모드 모두에서 단일 노드와 클러스터 서버가 HTTP 200과 동일한 텍스트 "Three primary colors are **red, blue, and yellow**. These are the basic colors in the color wheel, representing"를 반환했다. loopback에서 전체 push는 "Pushing stage assignments to 2 peer(s)"부터 두 "committed" 줄까지 약 2 ms였다. peer가 쓴 TOML에는 `127.0.0.1:19500`의 coordinator와 두 stage가 있고 `secret_file`은 없었다.

검증하지 않은 것: 실제 LAN에서의 다중 호스트 bring-up(모든 실행은 loopback), 실제 바이너리를 통한 mdns 기본 listen 주소 `0.0.0.0:0`, macOS, `--test distributed_integration`(변경된 함수를 참조하지 않음).

## 6. 남은 항목

설계상 열어 둔 항목(PR에 기록됨):

- assignment handshake와 stage-transport handshake에 channel label이 없어 secret 보유자에게는 서로 바꿔 쓸 수 있다. frame별 MAC도 없다.
- `0.0.0.0:port`를 advertise하는 beacon은 IP를 서명하지 않는다. UDP를 위조할 수 있는 LAN 공격자는 다른 source에서 재전송할 수 있지만, handshake가 여전히 identity를 bind하므로 영향은 bring-up 실패에 그친다.
- commit 단계 중 실패하면 coordinator가 종료된 상태에서 일부 peer가 commit된 채로 남을 수 있다.
- coordinator 역할을 하는 secret 보유자는 치명적 거부(예: model 불일치)로 peer를 종료시킬 수 있다. 치명적 사유는 재시도로 해결할 수 없는 불일치로 한정된다.

이번 리뷰에서 발견했으나 PR에서 고치지 않은 사소한 사항:

- `src/server/startup.rs`에서 `resolve_pp_peer_listen_addr`의 doc comment("The listen address of a `--pp-peer` process ...")가 `ensure_pp_peer_not_dry_run`의 doc 블록에 합쳐져 `resolve_pp_peer_listen_addr`에는 문서가 없다.
- peer가 쓴 cluster TOML은 peer가 localize했음에도 "Generated by `mlxcel-server --pp-auto`" 헤더를 유지한다.
- `tests/zero_config_cluster_init.rs`는 이 호스트의 19500-19549 포트 블록을 우선 쓰고, 다른 곳에서는 ephemeral 포트로 대체한다.

관련: #2658(이 PR로 닫힘), #2655, #2650, #2637, #2654(llama, gemma3, gemma4 remote parity; parity 테스트가 qwen3을 쓰는 이유).
