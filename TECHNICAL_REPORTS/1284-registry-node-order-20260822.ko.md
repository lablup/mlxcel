# 기술 보고서: PR #1284 - 분산 레지스트리 노드 접근자에 정의된 순서 부여

## 요약

`RegistryInner.nodes`는 `HashMap`이고, 레지스트리의 목록 접근자 넷이 `.values()`를 정렬 없이 반환하는데 소비자들은 위치에 의존하고 있었다. 가장 큰 여파는 장애 경로가 아니라 **주 요청 경로**였다. `all_nodes`가 `select_prefill_node` / `select_decode_node`로 흘러가고, 거기서 `RoundRobin`은 위치로 인덱싱하며 `LeastLoaded`는 `min_by_key`(동점이면 첫 번째), `MemoryAware`는 `max_by_key`(동점이면 마지막)를 쓴다. 차갑거나 유휴 상태인 클러스터가 정확히 동점 상황이고, 출하 기본값이 prefill `LeastLoaded` / decode `MemoryAware`라, 유휴 클러스터에서 "least loaded" 노드는 `HashMap` 순회가 앞에 놓은 노드였다.

수정은 접근자 안에서 `config.id`로 정렬하는 것이다. 한 번의 리뷰 스윕에서 찾은 같은 근본 원인 계열의 **네 번째이자 마지막** 인스턴스다. 앞선 셋은 #1265(테스트 픽스처), #1267(`lang_bias.rs`), #1276(RT-DETRv2 레이아웃 판별).

## 1. 문제

같은 파일의 형제 접근자 `nodes_at_stage`와 `nodes_at_rank`는 호출자가 위치에 의존하기 때문에 이미 `sort_by_key`로 끝난다. `all_nodes`와 `nodes_with_role`도 위치 의존 호출자를 똑같이 가졌는데 그 처리를 받은 적이 없었다.

요청 **개수**는 균형이 유지된다. 라운드로빈이 주어진 순서가 무엇이든 균등하게 나누기 때문이다. 기존 테스트가 통과한 이유도 그것이다. `nodes_with_role_filter`는 원소가 하나인 리스트에 인덱스를 단언하므로 고정된 순서와 임의의 순서를 구별할 수 없다. 흔들리는 것은 배정이었고, 그래서 동일한 클러스터 상태에서 장애 사건을 재현할 수 없었으며, 이기종 클러스터에서는 선택의 품질이 운영자가 볼 수 없는 이유로 달라졌다.

## 2. 기술적 판단

### 2.1 접근자에서 정렬하되 `sort_by_key`가 아니라 `sort_by`

원천에서 정렬하면 소비자마다 정렬을 기억할 필요 없이 결정성을 물려받는다. `sort_by(|a, b| a.config.id.cmp(&b.config.id))`를 택한 이유는 키가 `String`이고 원소에서 빌려올 수 없어 `sort_by_key`는 키 평가마다 클론하기 때문이다. 12개짜리 뒤섞인 벡터에서 실측: 키 평가 78회(각각 클론) 대 비교 39회 및 할당 0.

`nodes`를 `BTreeMap`으로 바꾸는 안은 이슈에서 이미 기각됐다. 목록 접근자만 필요로 하는 성질을 위해 O(1) `get_node` 경로에 비용을 물리기 때문이다. 구현 중에 그 판단을 뒤집을 근거는 나오지 않았다.

### 2.2 후보 목록만 정렬해서는 부족했다

이슈 자체의 분석을 정정하는 부분이다. `handle_node_failure`는 영향받은 요청 id를 `self.requests`(역시 `HashMap`)에서 모으고, 재라우팅 루프는 후보를 **그 순서로** 라운드로빈 배분한다. 후보를 정렬해도 요청 목록이 정렬되지 않으면 요청→노드 짝은 계속 실행마다 움직이고, 그게 바로 운영자가 재현하려는 성질이다. 접근자 수정과 함께 `affected.sort()`를 추가했다.

### 2.3 이슈가 지목한 것보다 접근자가 둘 더 있었다

`peer_addresses`와 `topology_summary`도 `.values()`를 정렬 없이 반환하거나 렌더링하는데 이슈에 이름이 없었다. `topology_summary`는 디스커버리와 클러스터 초기화 때 출력되는 **운영자 대면** 출력이라, 변하지 않은 클러스터에 대해 다시 실행하면 노드 목록이 뒤섞였다. 둘 다 정렬했고, 이로써 정렬된 접근자는 넷이 됐다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `src/distributed/registry.rs` | `all_nodes`, `nodes_with_role`, `peer_addresses`, `topology_summary`를 `config.id`로 정렬 |
| `src/distributed/disaggregated/request_router.rs` | `handle_node_failure`가 라운드로빈 전에 영향 요청 id를 정렬 |
| `src/distributed/registry_tests.rs`, `.../request_router_tests.rs` | 신규 테스트 9건 |

## 4. 리뷰 지적사항

이슈 본문의 주장 둘이 틀렸고, 그대로 안고 가는 대신 여기서 정정한다.

`routing.rs`가 순서 민감 소비자 목록에 있었다. **아니다.** `RoundRobinRouter`는 인덱싱 전에 이미 `online.sort_by(|a, b| a.node_id.cmp(&b.node_id))`를 수행하며 그렇게 적은 주석도 있다. 누군가 이미 그 경로를 결정적으로 만들어 두었다.

`find_pp_tp_node`가 안전한 이유는 가정이 아니라 기록해 둘 가치가 있다. `ClusterConfig::validate`가 중복 `(stage, rank)` 쌍을 거부하므로 `.values().find()`의 매치가 순서와 무관하게 유일하다.

이슈가 "영향 없음"으로 정리한 목록은 그 외에는 정확했고, 빠뜨린 소비자 하나(`router_front.rs:1218`)는 `.len()`만 취해 순서에 둔감하다.

회귀 테스트는 확정 전에 수정 전 접근자에 대고 실패를 증명했다. 이 버그 계열은 수정 전후 모두 통과하는 테스트를 쉽게 만들어내기 때문이다. 아홉 건 전부 실패했고 대부분 0~1번째 반복에서 났다. 예:

```
---- all_nodes_ordered_by_id_across_registries ----
  left: ["yankee-prefill", "alpha-decode", "mike-prefill", "bravo-decode", "zulu-hybrid"]
 right: ["alpha-decode", "bravo-decode", "mike-prefill", "yankee-prefill", "zulu-hybrid"]
---- round_robin_selection_sequence_is_stable_across_routers ----
  left: ["prefill-0", "prefill-1", "prefill-0", "prefill-1"]
 right: ["prefill-1", "prefill-0", "prefill-1", "prefill-0"]
```

레지스트리 테스트는 매번 레지스트리 64개를, 라우터 테스트는 라우터 32개를 **새로** 만든다. 하나를 재사용하지 않는 이유는 `RandomState`가 프로세스가 아니라 맵 인스턴스마다 무작위화하기 때문이다. 이 성질은 #1267 발행 시 측정했다. 같은 다섯 키로 만든 맵 10개가 한 프로세스에서 고유 순서 9개를 냈다.

## 5. 검증

GB10(DGX Spark, CUDA sm_121, Linux aarch64)에서 실측.

- `make verify-test-cuda`: PR 스레드에 기록.
- 좁은 필터, 전부 exit 0: `distributed::registry` 18 통과, `distributed::disaggregated::request_router` 29, `distributed::scheduler` 16, `distributed::routing` 19, `distributed::heartbeat` 9, `distributed::discovery` 3, `distributed::cluster_init` 14, `server::router_front` 20.
- `cargo fmt --all -- --check`: exit 0.
- `cargo clippy --lib --tests --features cuda -- -D warnings -A clippy::err_expect`: exit 0. allow 없이는 `src/multimodal/host_preprocessor_tests.rs:416`의 선재 `clippy::err_expect`로 101이 되는데, 이 브랜치보다 앞선 결함이고 #1283으로 추적 중이며 이 PR의 파일은 출력에 등장하지 않는다.

관측으로 검증하지 않은 것: 이슈의 수동 다중 노드 확인. prefill 노드 3개 이상과 라우터 반복 재시작이 필요한데 이 기계는 단일 노드다. 접근자 순서와, 동점 상황에서 `select_prefill_node` / `select_decode_node`까지 도달하는 결정성은 테스트로 덮었다. 다만 "정의된 접근자 순서면 프로덕션 선택이 재현 가능하다"는 주장은 각 전략 갈래가 후보 시퀀스와 원자 카운터의 순수 함수라는 점에 기대며, 이는 코드 검토로 참이고 동점에 대해서는 테스트로 확인했지만, 실제로는 라이브 메트릭이 대부분의 동점을 순서가 개입하기 전에 깨뜨린다.

## 6. 관련 작업

- #1277: 이 PR이 닫는 이슈.
- #1265와 PR #1266, #1267과 PR #1269, #1276과 PR #1281: 같은 계열의 형제 인스턴스.
- #1283: 이 브랜치를 검증하며 마주친 선재 clippy 실패.

`src/distributed/` 아래 남은 `.values()` / `.iter()` 수집 지점을 훑어 같은 계열 셋을 더 찾았고, 라우팅이 아니라 **축출(eviction) 동작**을 바꾸므로 별도 이슈로 미뤘다. `tensor_parallel/cache_manager.rs:711`은 `check_pressure`가 메모리 목표를 채우면 조기 중단해서 동점 시퀀스 중 무엇이 축출될지를 해시 순서가 정한다. `pipeline/cache_manager.rs:697`은 축출 우선순위 순서로 문서화된 `PreemptionSignal.sequence_ids`를 발행한다. `request_tracker.rs:333`은 종료된 요청을 `created_at`으로 정렬해 접두사를 취하는데 같은 `Instant` 동점이 해시 순서로 갈린다.
