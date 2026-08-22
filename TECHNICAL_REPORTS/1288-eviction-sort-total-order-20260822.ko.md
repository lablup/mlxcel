# 기술 보고서: PR #1288 - 축출 후보 정렬에 전순서 부여

## 요약

축출 경로 셋이 `HashMap`에서 후보 목록을 만들고, 정렬한 뒤, 접두사를 소비하고 있었다. 정렬은 결과를 결정적으로 만드는 것처럼 보이지만 아니다. `slice::sort_by_key`는 **안정 정렬**이라 같다고 비교되는 원소가 입력 순서를 유지하고, 그 입력 순서가 곧 `HashMap` 순회 순서다. 정렬 키가 동점인 후보들 중 어느 것이 소비되는 접두사에 들어가는지가 실행마다 달라졌다.

#1265, #1267, #1276, #1277이 고친 결함 계열의 미묘한 쪽이다. 그 넷은 정렬이 아예 없어서 눈으로 보였다. 이 셋은 **정렬을 하고 있어서** 안전해 보였다.

## 1. 문제

- `src/distributed/tensor_parallel/cache_manager.rs`: `select_eviction_candidates`가 `allocations.values()`를 모아 `last_accessed`(LRU) 또는 `current_offset`(LeastTokens)로 정렬한다. 호출자 `check_pressure`는 `projected_used <= target_bytes`가 되는 즉시 `break`하므로 접두사 소비자다. 동점이 어느 시퀀스가 실제로 캐시를 잃는지를 정하고, LeastTokens에서는 토큰 수가 겹치는 것이 예외가 아니라 일상이다.
- `src/distributed/pipeline/cache_manager.rs`: `PreemptionPolicy` 세 갈래에 같은 형태. 결과가 `PreemptionSignal.sequence_ids`로 발행되고 축출 우선순위 순서로 문서화돼 있다. 트리 안에 접두사 소비자가 아직 없으므로 깨져 있던 것은 관측된 동작이 아니라 발행된 계약이다.
- `src/distributed/request_tracker.rs`: `evict_if_needed`가 종료된 요청을 모아 `created_at`으로 정렬한 뒤 `.take(to_remove)`로 명시적 접두사를 취한다.

손상되는 것도, 죽는 것도 없다. 축출은 해소하려던 압박을 해소하고 개수도 맞다. 잃은 것은 **어느 시퀀스가 희생됐는가의 재현성**이고, 그것이 바로 요청이 선점됐을 때 운영자가 이유를 따지려고 필요로 하는 정보다.

## 2. 기술적 판단

### 2.1 정렬 키를 전순서로

각 정렬에 고유 id를 타이브레이커로 넣어 두 원소가 같다고 비교되는 일이 없게 했고, 그러면 안정성 여부가 무의미해진다. 비교자 형태와 튜플 키 형태 둘 다 옳다. request tracker는 `String` 키가 비교마다 튜플로 클론되지 않도록 비교자 형태를 쓰고, 두 캐시 매니저는 `Copy` 필드라 튜플 형태를 쓴다.

`sort_unstable_by`는 수정이 아니라고 명시했다. 임의 순서를 다른 임의 순서로 바꿀 뿐 임의성을 없애지 않는다.

### 2.2 파이프라인 계약은 경계에서 고정

이슈는 종단 단언을 tensor-parallel 쪽에만 요구했다. 거기만 실제 접두사 소비자가 있기 때문이다. 파이프라인 쪽에도 하나 넣어, 비공개 헬퍼가 아니라 발행되는 `PreemptionSignal.sequence_ids`에 단언을 걸었다. 소비자가 생기는 날 그 경계에서 이미 순서가 고정돼 있다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `src/distributed/tensor_parallel/cache_manager.rs` | 두 정책 갈래를 `(키, sequence_id)`로 정렬 |
| `src/distributed/pipeline/cache_manager.rs` | 세 정책 갈래를 `(키, sequence_id)`로 정렬, 뒤엉킨 `PreemptionSignal.sequence_ids` 문서를 `first = evict first`로 정정 |
| `src/distributed/request_tracker.rs` | `created_at` 정렬이 요청 키로 타이브레이크(비교자 형태) |
| 대응 `*_tests.rs` 3개 | 신규 테스트 8건 |

공개 setter도 `#[cfg(test)]` 접근자도 추가하지 않았다. 세 테스트 파일 모두 대상 모듈 안에서 `#[path]`로 붙으므로 비공개 헬퍼와 비공개 상태에 이미 닿을 수 있었고, 테스트 편의를 위해 API를 넓히는 것은 잘못된 교환이었을 것이다.

## 4. 리뷰 지적사항

테스트를 먼저 쓰고 아직 수정되지 않은 소스에 돌렸다. 수정을 쓴 뒤 되돌려 확인하는 방식이 아니다. 여기서 공허한 테스트를 쓰는 방법 셋을 미리 식별했고 셋 다 실제로 중요했다.

1. 맵을 재사용하면 수정 없이도 통과한다. `RandomState`가 `HashMap` 인스턴스마다 무작위화하기 때문이다. 각 테스트는 반복마다 맵을 새로 만들고 32회 또는 64회 돈다.
2. 정렬 키가 서로 다른 데이터는 수정 없이도 통과한다. 서로 다른 키는 이미 전순서이기 때문이다. 모든 테스트가 실제 동점을 만든다.
3. `Instant::now()`는 이 플랫폼에서 나노초 해상도라 스스로 충돌하지 않는다. LRU와 `created_at` 테스트는 같은 `Instant` 값을 의도적으로 써 넣는다.

수정 전 출력. `left`가 안정 정렬이 그대로 통과시킨 생 해시 순서다.

```
---- eviction_candidates_lru_tie_break_is_deterministic ----
  left: [5, 3, 1, 4, 7, 2, 8, 6]
 right: [1, 2, 3, 4, 5, 6, 7, 8]
---- check_pressure_prefix_is_deterministic_under_ties ----
  left: [15, 11, 14]
 right: [11, 12, 13]
---- eviction_tie_break_is_deterministic ----
  left: ["req-0", "req-1", "req-4", "req-5"]
 right: ["req-2", "req-3", "req-4", "req-5"]
```

이슈에 없던 함정 하나를 기록해 둔다. 다음 사람이 걸릴 것이기 때문이다. `evict_if_needed`는 `submit_with_id` 안에서 insert **전에** 돌기 때문에, 제출과 동시에 완료 처리하는 테스트는 여섯 번째 제출에서 의도치 않은 축출을 일으킨다. 테스트는 여섯 개를 모두 제출하고, 모두 완료한 뒤, `evict_if_needed`를 직접 호출한다.

## 5. 검증

GB10(DGX Spark, CUDA sm_121, Linux aarch64)에서 실측. 머지될 트리를 검증하려고 게이트 전에 `main`(`8fcc01f2`) 위로 리베이스했다.

- `make verify-test-cuda`: **8246 통과, 0 실패, 311 무시**, 101 스위트, exit 0. `main`(8238) 대비 +8이고 diff가 `#[test]`를 정확히 8개 추가하고 하나도 제거하지 않는다.
- 세 모듈 필터: 각각 48, 47, 21 통과, exit 0. 별도 프로세스 5회 추가 실행에서도 green이며, 이는 실행마다 `RandomState`를 다시 시드한다.
- `cargo fmt --all -- --check`: exit 0. `cargo clippy --lib --tests --features cuda -- -D warnings`: exit 0 (전에 이걸 red로 만들던 `err_expect`는 #1283 / PR #1285가 수정).

게이트 로그를 실패 테스트뿐 아니라 **프로세스 수준 abort**까지 훑었다. 스위트별 집계는 teardown 크래시를 볼 수 없다. 같은 게이트의 앞선 실행이 0 실패를 보고하면서도 101로 종료했는데, 모든 테스트가 통과한 뒤 다른 cargo 프로세스가 GPU를 포화시킨 상태에서 `Destroy(handle_) failed: driver shutting down`으로 abort했기 때문이다. 단독 실행하면 이 게이트도 #1283의 게이트도 깨끗하다.

## 6. 관련 작업

- #1286: 이 PR이 닫는 이슈. #1277 작업 중 `src/distributed/` 스윕에서 나왔다.
- #1265와 PR #1266, #1267과 PR #1269, #1276과 PR #1281, #1277과 PR #1284: 형제 인스턴스들.
- #1287: 이 계열을 `docs/code-guidelines.md`에 기록하고 정적 검사 가능성을 판단하자는 제안.

의도적으로 손대지 않은 근접 사례 둘을 확인했다. `src/distributed/routing.rs`는 안정 정렬 후 `online[0]`을 취하고 유휴 클러스터는 모든 성분이 동점이지만, PR #1284가 `Registry::all_nodes`에 정의된 순서를 주면서 상류에서 이미 치유했다. 그 경로는 접근자 수정에 조용히 의존한다. 그리고 `nodes_at_stage` / `nodes_at_rank`는 `Option<u32>` 키를 `unwrap_or(u32::MAX)`로 정렬해 동점이 날 것처럼 보이지만, `ClusterConfig::validate`가 두 필드 중 하나라도 비어 있는 PPTP 노드를 레지스트리에 닿기 전에 거부하고, 두 접근자 모두 비테스트 소비자가 없다.
