# PR #2772: 503 응답 전에 down 상태의 prefill peer를 다시 probe하기

**날짜**: 2026-10-31 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: base bebdd62c 위에서 7726ef31로 구현 완료, 이후 57b9f9e5에서 origin/main과 머지(ab7b8509의 #2769를 가져오며, 이 PR의 코드는 건드리지 않음). PR 본문의 로컬 게이트는 7726ef31 기준으로 GB10(spark-102)에서 통과. 작성 시점에 worktree에서 두 번째 로컬 게이트가 실행 중이었고 호스팅 CI는 일부 대기 중이었음(`cargo-fmt`, `cargo-clippy`, `cargo-deny`, `CPU-only link`, `OpenXLA feature compile`, `python tooling tests`, `WebUI bundle`, `WebUI installed artifact` 포함)
**언어**: Rust, Markdown
**위험도**: 낮음 (wire, API, 플래그 변경 없음. `/health`와 503 메시지는 그대로. prefill 호스트가 연결 시도를 응답 없이 버리는 경우 거절 응답이 re-probe를 최대 500 ms 기다릴 수 있음)

## 요약

PR #2772(#2763 해결)는 prefill peer가 이미 listen 중인데도 disaggregated router가 HTTP 503을 돌려주던 문제를 없앤다. 이전에는 health monitor가 첫 probe 전에 3초를 잤고, admission은 monitor가 down으로 표시한 노드를 다시 확인하지 않았다. 그래서 probe 직후에 포트를 연 peer는 최대 한 주기(3초) 동안 거절되었다. 이제 monitor는 시작 즉시 probe하고, `admission_reject`는 `Queue` 판정을 503으로 바꾸기 전에 `Unreachable`로 표시된 prefill 가능 노드를 다시 probe한다. 이 re-probe는 single-flight이며 router 전체에서 250 ms에 최대 한 번 시작되고, 노드마다 500 ms connect를 한 번 시도한다. 노드별 probe와 상태 전이(새 파일 `src/server/router_front_health.rs`의 `probe_node`)는 monitor와 공유한다.

## 1. 문제 정의

### 1.1 배경

#2746과 #2747 작업 중 `disaggregated_router_rejects_model_owned_family_and_keeps_serving`가 간헐적으로 실패했다. 모든 peer가 떠 있고 router의 `/health`가 200을 응답하는데도, 시작 직후의 요청이 "all serving nodes are at capacity or unreachable; retry shortly"를 받았다. #2747은 모든 router e2e 테스트가 `GET /router/stats`에서 모든 peer가 `online`이 될 때까지 기다리게 하는 방식(`wait_for_router_peers_online`)으로 테스트만 고쳤다. 실제 클라이언트에게는 그런 대기가 없다. #2747 리뷰와 리포트(4절 "트레이드오프와 잔여 항목": "router 시작 직후의 503 구간은 그대로다")는 router 동작을 후속 과제로 남겼고, 그것이 #2763이 되었다.

### 1.2 monitor의 첫 probe가 한 주기 늦었다

`spawn_health_monitor`는 루프 맨 앞에서 `sleep(HEALTH_PROBE_INTERVAL)`(3초)을 실행했으므로 첫 probe는 시작 3초 뒤였고, 각 probe의 connect timeout은 2초였다. 그 순간 아직 모델을 로딩 중인 peer는 `NodeStatus::Unreachable`로 표시되어 다음 주기까지 down으로 남았다. 이후 복구 시에도 같은 구간이 다시 생겼다. 한 주기 직후에 돌아온 peer는 다음 주기까지 최대 3초를 기다렸다.

### 1.3 admission이 오래된 판정을 그대로 믿었다

`admission_reject`는 `RequestRouter::apply_backpressure`를 호출하는데, 이 함수는 `LoadLevel::Critical` 미만으로 online인 prefill 노드가 없으면 `Queue`를 반환한다. router에는 async queue가 없으므로 `Queue`는 즉시 503이 된다. 이 경로 어디에서도 registry가 down으로 표시한 노드를 다시 확인하지 않았기 때문에, 요청은 최대 3초 전의 monitor 관측 결과에 근거해 거절되었다.

### 1.4 방치할 경우의 위험

| 위험 | 영향 | 가능성 |
|---|---|---|
| 막 시작했거나 복구 중인 router의 클라이언트가, 클러스터가 처리할 수 있는데도 최대 약 3초 동안 503을 받음 | 중간 (재시도하지 않는 클라이언트는 요청 실패를 보고, `/health`로 판단하는 오케스트레이터는 green을 보고 트래픽을 보냄) | router가 peer보다 먼저 또는 동시에 시작하는 rolling restart마다 높음 |
| 테스트 스위트가 계속 router 전용 readiness polling을 필요로 함 | 낮음 | 확실 |

## 2. 변경 요약

| 영역 | 파일 | 변경 |
|---|---|---|
| Probe 모듈 | `src/server/router_front_health.rs` (신규, 242줄) | `HEALTH_PROBE_INTERVAL`(3초)과 `HEALTH_PROBE_TIMEOUT`(2초)을 이곳으로 이동. 새 상수 `ON_DEMAND_PROBE_MIN_INTERVAL`(250 ms), `ON_DEMAND_PROBE_TIMEOUT`(500 ms). 로그 태깅용 `ProbeTrigger { Monitor, Admission }`. `PeerProbe { on_demand: Arc<tokio::sync::Mutex<Option<Instant>>>, connect_attempts: AtomicU64 }`. `probe_node`(connect 한 번과 상태 전이, 전이는 connect 이후 registry에서 다시 읽은 상태로 결정). `spawn_health_monitor`는 첫 sleep 전에 `run_probe_cycle` 실행. `has_unreachable_prefill`과 `reprobe_unreachable_prefill` |
| Router front | `src/server/router_front.rs` | `#[path = "router_front_health.rs"] mod health;`로 모듈을 불러오고 `spawn_health_monitor`를 re-export. `RouterState`에 `peer_probe: health::PeerProbe` 추가. `admission_reject`는 `async`가 되고 `&Arc<RouterState>`를 받으며, prefill 가능 노드가 `Unreachable`일 때 `Queue`에서 re-probe. `route_chat`과 `route_completion`이 이를 await. 모듈 문서 갱신 |
| Lib 테스트 | `src/server/router_front_tests.rs` | 신규 테스트 4개: `first_health_probe_runs_at_startup`, `admission_reprobes_a_peer_marked_down_before_refusing`(chat과 completions), `admission_reprobe_is_bounded`, `probe_decides_transition_from_current_registry_status` |
| E2E 테스트 | `tests/disaggregated_router_e2e.rs` | 신규 `disaggregated_router_started_before_prefill_serves_first_request`와 helper `wait_for_router_node_status`. `wait_for_router_peers_online`의 doc에 대기를 유지하는 이유 기재 |
| 문서 | `docs/distributed.md`, `docs/CONTINUOUS_BATCHING.md` | 시작 시 probe, re-probe의 발동 조건과 상한, 호스트가 연결 시도를 버릴 때의 비용, `/health`는 계속 liveness 확인임을 기술. 이 문서들의 한국어판은 없음 |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 6 |
| 라인 | +646 / -93 |
| 신규 테스트 | lib 4개, 실제 프로세스 e2e 1개 |
| 커밋 | 2개와 origin/main 머지 1개 |

| 커밋 | 목적 |
|---|---|
| 0022227a | 시작 시 probe, async admission re-probe, 공유 `probe_node`, lib 테스트 3개, e2e 케이스, 문서 |
| 7726ef31 | 리뷰 반영: `probe_node`가 connect 후 registry 상태를 다시 읽음. re-probe가 `nodes_with_role`로 `Hybrid`까지 포함. gate를 소유한 spawn된 task에서 실행. 250 ms 간격을 완료 시점에 기록. 문서에서 모든 prefill 노드가 down이어야 한다는 서술 제거. `admission_reprobe_is_bounded`가 간격까지 검사. 신규 `probe_decides_transition_from_current_registry_status` |
| 57b9f9e5 | origin/main 머지(ab7b8509, #2769). 그쪽의 `docs/distributed.md` 변경은 pipeline 절에 있어 이 PR과 충돌 없음 |

## 3. 기술적 선택과 그 이유

### 3.1 시작 시 probe하고, 각 주기 뒤에 sleep

monitor 루프를 `sleep; probe`에서 `probe; sleep`으로 바꿨다. peer가 이미 listen 중인 router는 그 peer를 down으로 표시하지 않고, peer가 아직 뜨지 않은 router는 3초 뒤가 아니라 즉시 그 사실을 안다(e2e 테스트는 prefill 노드를 시작하기 전에 `unreachable`을 관측하는 데 이 동작을 이용한다). 이것만으로는 구간이 사라지지 않는다. 시작 probe 직후에 포트를 연 peer는 여전히 한 주기를 기다리며, 그래서 3.2가 필요하다.

### 3.2 prefill 가능 노드가 down이고 판정이 `Queue`일 때만 re-probe

`admission_reject`는 `apply_backpressure`가 `Queue`를 반환하고 `has_unreachable_prefill`이 참일 때만 re-probe한 뒤 `apply_backpressure`를 다시 호출하고, 두 번째 판정도 `Queue`일 때만 503을 응답한다. 이 조건 덕분에 정상 경로와 과부하 경로는 바뀌지 않는다.

- `Accept`는 probe하지 않으므로 정상 클러스터는 비용이 없다.
- `Reject`(prefill queue가 가득 참)는 probe하지 않는다. 노드가 돌아와도 queue가 비워지지 않는다.
- down으로 표시된 노드 없이 나온 `Queue`(모든 prefill 노드가 `LoadLevel::Critical`)는 probe하지 않는다. 복구할 대상이 없다.

re-probe 대상은 `Prefill`과 `Hybrid`로, `apply_backpressure`가 `get_prefill_load_infos`로 세는 역할과 같다. 첫 커밋은 `Prefill`만 다뤘고, 하드닝 커밋이 `Hybrid`를 더해 발동 조건과 판정이 일치하게 했다. 현재 router front는 hybrid 노드를 등록하지 않으므로 방어적 처리다.

| 선택지 | 장점 | 단점 |
|---|---|---|
| 모든 peer를 한 번 probe할 때까지 `/health`가 503 반환 | 오케스트레이터가 peer 상태를 알 때까지 트래픽을 보류 | 문서화된 liveness endpoint의 의미가 바뀜. 첫 주기 이후 또는 나중의 장애 이후에 뜬 peer에 대한 3초 구간은 그대로 |
| `HEALTH_PROBE_INTERVAL` 단축 | 한 줄 변경 | 구간을 줄일 뿐 없애지 못함. decode 노드를 포함한 모든 peer에 대한 백그라운드 connect가 배로 늘어남 |
| **선택: 시작 시 probe와 거절 직전의 on-demand re-probe** | 클라이언트가 거절되려는 바로 그 시점에 구간을 닫음. `/health`와 주기는 그대로 | 실제 장애 중의 거절이 probe를 기다릴 수 있음(3.3과 5절) |

### 3.3 on-demand 경로의 상한

그대로 두면 거절되는 요청마다 down 노드 전부에 connect를 시도할 수 있으므로, re-probe는 router 전체 기준으로 제한된다.

- **Single-flight.** `PeerProbe::on_demand`는 `tokio::sync::Mutex<Option<Instant>>`이다. lock을 쥐고 있다는 것은 re-probe가 진행 중이라는 뜻이다. 그 사이에 온 요청은 lock을 기다렸다가 최근 기록을 보고 반환하며, 진행 중이던 probe가 남긴 registry 상태를 그대로 쓴다.
- **완료 시점 기준 250 ms에 최대 한 번.** 기록은 `join_all`이 반환된 뒤에 쓴다. 첫 커밋은 시작 시점에 기록했는데, 500 ms timeout과 결합하면 느린 probe가 끝나자마자 새 probe가 시작될 수 있었다. 완료 시점에 기록하면 probe 소요 시간과 무관하게 probe 사이에 250 ms 동안 캐시된 판정이 보장된다.
- **down 노드마다 500 ms connect 한 번, 병렬.** probe를 시작한 요청은 down 노드 수에 500 ms를 곱한 시간이 아니라 최대 약 500 ms를 기다린다. monitor는 클라이언트가 기다리지 않으므로 2초 timeout을 유지한다.
- **connect 동안 registry나 request map lock을 잡지 않음.** `unreachable_prefill_nodes`는 snapshot을 만든다(`nodes_with_role`이 read lock 아래에서 복제하고 lock을 놓는다). `probe_node`는 connect가 끝난 뒤에만 registry lock을 잡는다. `.await`을 가로질러 잡히는 lock은 tokio gate 하나뿐이다.

클라이언트 요청량으로 connect 빈도를 올릴 수 없다. 상한은 (probe 소요 시간 + 250 ms)마다 probe 한 번이며, connect 대상은 설정된 peer 주소뿐이다.

### 3.4 gate를 소유한 spawn된 task에서 re-probe 실행

`reprobe_unreachable_prefill`은 `lock_owned`로 gate를 얻고 probe, 기록, 해제를 하는 task를 spawn하고, handler는 그 `JoinHandle`을 await한다. 클라이언트가 연결을 끊으면 axum이 handler future를 drop하는데, `JoinHandle`을 drop하면 task는 취소되지 않고 분리되므로 probe는 끝까지 실행되어 결과를 기록한다. 이렇게 하지 않으면 connect 도중의 연결 끊김이 probe를 상태 전이 없이 버리고, 시작 시점 기록과 결합하면 이후 250 ms 동안의 요청이 새 정보 없이 거절된다. probe task가 panic하면 warn으로 로그를 남기고 "변화 없음"으로 처리하며, 판정은 두 번째 `apply_backpressure`가 내린다.

### 3.5 노드별 probe 하나, 현재 상태로 전이 결정

connect와 상태 전이(`Online`에서 `Unreachable`로 갈 때 `handle_node_failure` 호출과 warn 로그, `Unreachable`에서 `Online`으로 갈 때 info 로그)를 monitor 루프에서 꺼내 두 호출자가 함께 쓰는 `probe_node`로 옮겼다. 두 로그 모두 `trigger = "monitor"` 또는 `"admission"`을 남기므로 운영자는 어느 경로가 노드를 복구하거나 down시켰는지 구분할 수 있다.

첫 버전은 호출자 snapshot의 `node.status`로 match했다. monitor는 주기마다 snapshot을 한 번 만들고 connect는 2초가 걸릴 수 있으므로, admission 경로가 노드를 복구한 뒤 monitor가 그 노드에 도달하기 전에 노드가 다시 죽으면 monitor는 `(false, Unreachable)`을 보고 죽은 노드를 한 주기 더 online으로 두었다. 이제 `probe_node`는 snapshot을 id와 주소에만 쓰고 connect 후 `registry.get_node`로 상태를 다시 읽는다. 그 사이 registry에서 제거된 노드는 건드리지 않는다. `probe_decides_transition_from_current_registry_status`는 registry에서는 `Online`인 노드를 `Unreachable`이라고 적힌 snapshot으로 넘겨 이 동작을 고정한다.

### 3.6 `/health`와 e2e readiness 대기는 유지

`router_health`는 문서화된 대로 여전히 무조건 `"ok"`를 반환한다. 이는 클러스터가 처리 가능한지가 아니라 router 프로세스가 살아 있는지를 알린다. 503 메시지도 같으므로 메시지로 분기하는 클라이언트는 차이를 보지 않는다. e2e 스위트는 `wait_for_router_peers_online`을 유지한다. 503을 피하는 데는 더 이상 필요 없지만, listen을 시작하지 못한 peer가 관련 없는 요청 assertion이 아니라 이름이 붙은 메시지로 그 자리에서 실패하게 한다.

## 4. 검증

### 4.1 실패 우선 증거

main bebdd62c에 테스트만 추가하고 router 코드는 그대로 둔 상태:

| 테스트 | main | 변경 후 |
|---|---|---|
| `admission_reprobes_a_peer_marked_down_before_refusing` | `/v1/chat/completions: refused although the peer is listening: all serving nodes are at capacity or unreachable; retry shortly` | chat과 completions 모두 200, 이후 노드는 `Online` |
| `first_health_probe_runs_at_startup` | `the first health probe did not mark the closed peer down within 1 s of the monitor starting (status Some(Online))` | 1초 안에 `Unreachable`로 표시 |
| 나머지 `server::router_front::` | 43 통과, 2 실패 | 47 통과 |

`admission_reprobe_is_bounded`는 `connect_attempts` hook이, `probe_decides_transition_from_current_registry_status`는 `health::probe_node`가 필요한데 둘 다 main에 없어 main에서는 실행할 수 없다. 상한 테스트는 닫힌 peer에 동시 chat 요청 20개를 보내 모든 응답이 기존과 같은 503인지 확인하고, `1 <= attempts <= 1 + elapsed / 250 ms`를 검사한다. 이어서 구간 안의 요청이 시도를 늘리지 않는지(측정된 경과 시간이 구간이 아직 열려 있었음을 보여줄 때만 검사), 400 ms sleep 후의 요청이 정확히 한 번 늘리는지 확인한다.

### 4.2 로컬 게이트 (PR 본문 기준, 7726ef31)

GB10(spark-102), `--profile test-fast --features cuda`, `--test-threads=1`. e2e는 `gpu-lock` 아래에서 `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`로 실행.

| 스위트 | 결과 |
|---|---|
| `server::router_front::` (lib) | 47 통과. 신규 테스트 4개는 10회 반복 실행 모두 통과 |
| `disaggregated_router_e2e --include-ignored` | 8/8 (기존 7개와 신규 시작 순서 케이스) |
| `disaggregated_serving_e2e --include-ignored` | 2/2 |
| `cargo clippy ... -D warnings` (`cuda`로 lib와 테스트, `cuda,test-utils`로 `disaggregated_router_e2e`), `cargo fmt --all -- --check` | 깨끗함 |

신규 e2e 케이스는 #2763의 수동 확인 절차다. router와 decode 노드가 먼저 시작하고, `/router/stats`가 `prefill-0`을 `unreachable`로 보여주고, prefill 노드가 시작되며, `wait_for_tcp`가 serving 포트의 accept를 확인하자마자 첫 `/v1/completions`를 보낸다. 응답은 200이고, 이후 `/router/stats`는 노드를 `online`으로 보고한다.

### 4.3 실행하지 않은 것

- 신규 e2e 케이스는 main에서 실행하지 않았으므로 이 케이스의 실패 우선 증거는 없고, 어느 경로가 노드를 복구했는지도 증명하지 않는다. 포트 accept와 요청 사이 간격은 밀리초 단위라 그 사이에 monitor 주기가 끼어들 가능성은 낮지만, 테스트가 두 경우를 구분하지는 않는다. admission 경로를 고정하는 것은 monitor 없이 실행되는 lib 테스트 `admission_reprobes_a_peer_marked_down_before_refusing`이다.
- SYN을 응답 없이 버리는 호스트를 다루는 테스트가 없다(모든 테스트 peer는 즉시 거절하거나 즉시 accept한다). 따라서 3.3의 500 ms 대기는 timeout 값에서 추론한 것이지 측정한 값이 아니다.
- `Hybrid` 노드를 등록하는 테스트가 없다. re-probe의 `Hybrid` 분기는 리뷰로만 확인했다.
- Metal, ROCm, 멀티 호스트 배포에서는 실행하지 않았다.

## 5. 남은 공백과 위험

- **최대 250 ms의 잔여 구간.** re-probe가 끝난 뒤 250 ms 안에 온 요청은 그 판정을 재사용한다. 그 구간 안에 peer가 accept를 시작하면 해당 요청은 여전히 503을 받는다. 이전의 최대 3초 구간을 대체한다.
- **실제 장애 중의 거절이 느려질 수 있다.** 이 PR 이전에는 "사용 가능한 prefill 없음"에 대한 503이 즉시 나갔다. 이제 down된 prefill 호스트가 connect를 거절하면 re-probe가 즉시 끝나므로 실질적인 차이가 없다. 반면 패킷을 응답 없이 버리면, re-probe를 시작한 요청과 gate 뒤에 대기한 모든 요청이 최대 500 ms를 기다리고, 그 다음 250 ms의 요청은 즉시 거절된다. 그 상태에서 거절의 약 3분의 2가 최대 500 ms의 지연을 더 갖는다. 문서에 이 비용을 적어 두었다.
- **부하 차단이 관련 없는 down 노드를 기다릴 수 있다.** online prefill 노드들이 `LoadLevel::Critical`이고 다른 prefill 노드가 `Unreachable`이면 `Queue` 판정이 re-probe를 일으키므로, 과부하 503도 위의 지연을 이어받는다. 이슈의 명세와 일치하며 리뷰에서 MEDIUM으로 지적되었으나 변경하지 않았다.
- **gate가 대기자를 직렬화한다.** 노드가 down인 `Queue` 상태의 요청은 각각 task를 spawn하고 tokio mutex를 차례로 잡는다. 대기자는 기록만 읽고 반환하므로 비용은 작지만, 거절 트래픽이 몰릴 때 router 전체의 직렬화 지점이 된다.
- **거절되는 요청마다 registry 복제 두 번.** `has_unreachable_prefill`은 비어 있는지 확인하려고 down 노드를 모으고, spawn된 task가 다시 모은다. registry 크기는 요청 수가 아니라 peer 수이므로 LOW로 남겼다.
- **liveness 기준은 여전히 "포트가 accept함"이다.** re-probe는 monitor와 같은 TCP connect 검사를 쓴다. 처리 가능해지기 전에 serving 포트를 여는 peer는 두 경로 모두에서 일찍 복구된다. 이 PR은 그 의미를 바꾸지 않는다.
- **`closed_loopback_addr`는 테스트가 connect하기 전에 포트를 놓는다.** 그 사이 다른 프로세스가 포트를 잡을 수 있다. 구간은 마이크로초 단위이며 반복 실행은 통과했다.

## 6. 학습 포인트

### 6.1 캐시된 health 판정은 작업을 거절하려는 시점에 다시 확인한다

백그라운드 monitor가 보는 peer 상태는 주기를 TTL로 하는 캐시다. 주기를 줄이면 TTL만 줄어든다. 오래된 항목이 피해를 주는 유일한 순간, 여기서는 503 직전에 캐시를 재검증하면 백그라운드 부하를 늘리지 않고 구간을 닫을 수 있다. 단, 거절 경로가 connect 증폭기가 되지 않도록 재검증에 상한이 있어야 한다.

### 6.2 소요 시간이 일정하지 않은 작업의 rate limit은 완료 시점에 기록한다

시작 시점에 기록하는 "N ms에 한 번" gate는 느린 작업 직후에 다음 작업이 바로 시작되도록 허용한다. 완료 시점에 기록하면 N이 작업 사이의 보장된 휴지 간격이 되며, 이것이 대상에 가는 부하를 제한한다. `admission_reprobe_is_bounded`는 그 간격의 양쪽을 모두 검사한다.

### 6.3 공유 작업을 호출자의 취소와 분리한다

한 호출자가 여러 호출자를 대신해 작업하는 경우(single-flight), 그 작업은 해당 호출자와 함께 사라지면 안 된다. gate를 소유한 task로 spawn하고 호출자는 `JoinHandle`을 await하게 하면, 클라이언트 연결이 끊겨도 공유 상태가 반쯤 갱신된 채로 남지 않는다. tokio `JoinHandle`을 drop하면 task는 abort되지 않고 분리된다.

### 6.4 전이는 검사를 예약한 snapshot이 아니라 현재 상태로 결정한다

무엇을 probe할지 고르는 데는 snapshot으로 충분하다(id와 주소는 바뀌지 않는다). 하지만 전이는 적용하는 순간의 상태로 결정해야 한다. 사이에 있는 느린 작업 동안 다른 writer가 끼어들 수 있기 때문이다. `await` 뒤에 다시 읽는 비용은 map 조회 한 번이다.

## 7. 참고 자료

- Issue #2763: Stop 503s while a just-started peer is still marked down
- PR #2747과 그 리포트(`TECHNICAL_REPORTS/2747-router-reasoning-only-eos-parity-20261030.ko.md`, 4절 "트레이드오프와 잔여 항목"): 테스트 측 `wait_for_router_peers_online` 수정과 기록된 후속 과제
- Issue #201: Disaggregated serving: multi-node routing, load balancing, and failover (health monitor와 admission control 도입)
- `src/distributed/disaggregated/request_router.rs`: `RequestRouter::apply_backpressure`
- `src/distributed/registry.rs`: `NodeRegistry::nodes_with_role`, `get_node`, `set_node_status`
