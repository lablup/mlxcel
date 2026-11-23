# 기술 보고서: PR #3268 - Prometheus에서 prompt-cache 조회를 한 번씩만 세기

**날짜**: 2026-11-23
**상태**: 완료
**위험도**: 낮음 (메트릭 집계만 바뀌며, 요청 경로 동작, 캐시 내용, 메트릭 이름과 HELP 문구는 그대로)

## 요약

prompt-cache 저장소와 배치 스케줄러가 같은 `BatchMetrics` 카운터를 둘 다 써서, `/metrics`가 실제보다 약 두 배의 hit, miss, 재사용 토큰 수를 보고했다 (#3246). 이제 카운터마다 쓰는 곳이 정확히 하나다. 저장소의 `BatchMetricsCacheAdapter`는 eviction과 snapshot miss만 넘기고, 스케줄러가 dense hit, dense miss, snapshot hit, 두 재사용 토큰 합계를 쓴다. snapshot 복원 경로와 state checkpoint 경로에도 같은 이중 집계가 있었고, 같은 어댑터 변경으로 함께 고쳐졌다. 운영 환경과 같은 방식으로 연결한 회귀 테스트 세 개가 main에서는 실패하고(dense hit 2 대 1, snapshot hit 2 대 1, checkpoint 경로 snapshot hit 8 대 4) 이 브랜치에서는 통과한다. macOS 전체 게이트는 16227개 테스트, 실패 0으로 통과했다.

## 1. 문제 정의

`startup.rs`는 저장소를 `PromptCacheStore::with_metrics(.., BatchMetricsCacheAdapter::new(batch_metrics))`로 만들고, 같은 `Arc<BatchMetrics>`를 스케줄러에도 넘긴다. 저장소는 조회할 때마다 `record_lookup`을 호출했고, 어댑터가 이를 `record_prompt_cache_hit(matched_len)` 또는 `record_prompt_cache_miss()`로 바꿨다. 그런데 스케줄러도 같은 사건을 다시 기록했다. 채택에 성공하면 `finish_prompt_cache_adopt`가, 아무것도 채택하지 않으면 admission의 miss 지점이 기록했다. 이슈에서 측정한 값은 `/metrics` hit 18 대 `/v1/cache/stats` 9, 재사용 토큰 111840 대 `cached_tokens` 합계 약 55.9k였다.

두 기록 지점은 서로 다른 것을 재고 있었다. 저장소는 원시 일치 길이를 기록했고, 스케줄러는 `whole_prompt_reuse_cap`으로 상한을 두고 블록 크기로 내림한 뒤 `cached_tokens`로 반환하는 `adopted_len`을 기록한다. 저장소는 hit로 봤지만 스케줄러가 거절한 경우는 hit 한 번, miss 한 번으로 셌다. 비율은 우연히 맞았지만 절대 수치와 토큰 절감량은 두 배로 틀렸다.

기존 테스트로는 이를 볼 수 없었다. 모든 스케줄러 테스트가 메트릭 어댑터가 없는 `PromptCacheStore::with_config`로 저장소를 만들었기 때문이다.

## 2. 변경 요약

- `src/server/state.rs`: `BatchMetricsCacheAdapter::record_lookup`은 `record_insert`처럼 아무것도 하지 않는다. `record_snapshot_lookup`은 miss만 넘긴다. 타입 문서 주석에 기록 지점 하나 원칙을 적었다.
- `src/server/batch/scheduler/prompt_cache.rs`: "also increment BatchMetrics" 주석을 지웠다. 스케줄러 쪽 호출은 이미 보고되는 `cached_tokens` 값을 기록하므로 바꾸지 않았다.
- `src/server/startup.rs`: 연결 지점 주석이 어댑터가 eviction과 snapshot miss만 전달한다고 설명하도록 고쳤다.
- `docs/turbo-kv-cache.md`: snapshot 카운터가 `/v1/cache/stats`의 `snapshot_*` 필드와 보고된 `cached_tokens`와 같다는 문장 하나를 추가했다.
- 테스트:
  - `src/server/batch/scheduler_prompt_cache_metrics_tests.rs`(신규): dense Llama 실행(KV adopt)과 tiny Gemma 3 실행(snapshot 복원). 각각 저장소를 운영 환경과 같게 연결하고, admission을 거쳐 cold와 warm 두 번의 턴을 돌린다. `/metrics` 카운터가 저장소의 `hits`, `lookups - hits`와 같은지, 재사용 토큰이 `cached_tokens` 합계와 같은지 확인한다.
  - `src/server/batch/scheduler_state_checkpoint_tests.rs`: `checkpoint_rebuild_counts_once_in_prometheus_with_production_wiring`이 네 턴과 `edited(2)` checkpoint 재구성을 다룬다. 기존 `scheduler()` 헬퍼는 새 `scheduler_with_metrics(store, metrics)`에 위임한다.
  - `src/server/state_tests.rs`: 어댑터 테스트가 이제 어댑터가 dense 카운터를 건드리지 않는지 확인하고, snapshot miss만 넘기는지 보는 테스트를 새로 추가했다.
  - 재사용을 위해 헬퍼 두 개(`cycle_llama`, `EOS_ID`)를 `pub(super)`로 바꿨다.

## 3. 기술적 선택과 그 이유

**기록은 저장소가 아니라 스케줄러가 한다.** 반대로 어댑터를 남기고 스케줄러 호출을 지우는 방안은 기각했다. 그러면 거절된 채택이 hit로 세지고, `cached_tokens` 대신 저장소의 원시 일치 길이가 보고된다. 조회가 실제로 채택됐는지, 토큰이 몇 개 재사용됐는지 아는 것은 스케줄러뿐이고, `/metrics` HELP 문구도 이미 "Successful prompt-prefix cache adoptions"라고 쓰여 있다.

**snapshot miss는 어댑터에 남긴다.** snapshot 조회가 아무것도 찾지 못했다는 사실은 저장소만 안다. 스케줄러는 snapshot miss를 별도 사건으로 보지 못한다. 옮기려면 스케줄러에 새 신호가 필요한데 얻는 것이 없다. 그래서 원칙은 하위 시스템마다가 아니라 카운터마다 기록 지점 하나다.

**테스트는 저장소를 운영 환경과 같게 연결한다.** 결함이 연결 방식에 있었으므로, 기존 `with_config` 하네스로 만든 테스트는 main에서도 통과했을 것이다. 새 테스트는 모두 main의 `state.rs`로 실패하는 것을 확인한 뒤 받아들였다.

## 4. 측정

단위 테스트 수준, macOS Metal, `--profile test-fast`, `--test-threads=1`.

| 테스트 | main | 이 PR |
|---|---|---|
| dense 조회, `/metrics` hit 대 저장소 hit | 2 대 1 (실패) | 1 대 1 |
| snapshot 복원, snapshot hit 대 저장소 | 2 대 1 (실패) | 1 대 1 |
| state checkpoint 재구성, snapshot hit 대 저장소 | 8 대 4 (실패) | 4 대 4 |
| state checkpoint 재구성, 재사용 토큰 대 채택 길이 합계 | 192 대 96 (리뷰어 일회성 실행) | 96 대 96 |

게이트: `make verify-test`가 192개 테스트 바이너리에서 16227개 통과, 실패 0, 무시 499로 통과했다.

## 5. 측정하지 않은 것

- 실제 체크포인트로 서버를 띄워 `/metrics`, `/v1/cache/stats`, `cached_tokens` 합계를 비교하는 수동 확인은 하지 않았다. 검증 머신에 체크포인트가 없었다. 운영 환경과 같게 연결한 스케줄러 테스트가 작은 모델로 같은 비교를 수행한다.
- Linux/CUDA에서는 실행하지 않았다. 백엔드와 무관한 집계 변경이다.

## 6. 학습 포인트

- 두 구성 요소가 `Arc` 싱크를 공유할 때는 연결 방식도 동작의 일부다. 구성 요소를 `startup.rs`와 다르게 조립하는 하네스는 운영 연결에서만 생기는 결함을 가릴 수 있다.
- 기록 지점이 둘인 메트릭은 비율로는 맞아 보이면서 모든 절대값이 틀릴 수 있다. 그래서 검증은 독립된 출처(`/v1/cache/stats`, 응답의 `cached_tokens`)와 절대 수치를 비교해야 한다.

## 7. 후속 작업

- warmup snapshot miss: `run_next_prompt_cache_warmup`이 `lookup_snapshot_prefix`를 호출하므로, warmup miss는 여전히 `prompt_cache_snapshot_misses_total`을 올린다. warmup hit는 더 이상 올리지 않는다. warmup을 빼려면 `/v1/cache/stats`에도 영향을 주는 저장소 API 변경이 필요하다.
- snapshot 계열과 dense hit rate(기존 문제): `supports_snapshot_reuse()` 모델에서는 cold 요청마다 admission이 `prompt_cache_misses_total`을 올리지만, 복원은 `prompt_cache_snapshot_hits_total`에만 반영된다. 그래서 `/metrics`로 계산한 dense hit rate가 이 모델들에서는 0%로 나온다.
- 저장소는 snapshot hit로 봤지만 스케줄러가 거절한 경우(`ModeMismatch`, 복원 오류, 슬롯 부족)는 이제 어느 snapshot 카운터도 올리지 않는다. "successful restores"라는 HELP 문구와 맞는다. 따라서 Prometheus의 snapshot hit와 miss를 합친 값이 저장소의 `snapshot_lookups`보다 작을 수 있다.
