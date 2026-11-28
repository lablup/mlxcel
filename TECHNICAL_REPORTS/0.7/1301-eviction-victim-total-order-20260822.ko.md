# 기술 보고서: PR #1301 - 선점 victim 선택에 전순서 부여

## 요약

`BatchScheduler::select_eviction_victim`은 어느 진행 중 시퀀스를 선점할지를 `LongestFirst`에서는 `generated_tokens.len()`에 대한 `max_by_key`로, `LowestPriority`에서는 우선순위 다음 길이에 대한 `min_by`로 골랐다. 둘 다 `ActiveBatch::iter_sequences` 즉 `HashMap::values` 위에서 돌았으므로 동점이 해시 순서로 떨어졌다.

`docs/code-guidelines.md`에 기록된 계열의 여덟 번째 인스턴스이고, **동점이 키의 거칠어짐이 아니라 평범한 상태에서 도달 가능한 유일한 것**이다. 함께 승인된 시퀀스들은 나란히 디코딩하므로 토큰 수를 공유하고, `PreemptionPolicy::LongestFirst`와 `RequestPriority::Normal` 둘 다 출하 기본값이다. 동일하게 설정된 워커 둘이 동일한 부하에서 서로 다른 요청을 선점했고, 운영자는 배치 상태로부터 그 선택을 재현할 수 없었다.

## 1. 문제

손상된 것은 없고 정책의 명시된 의도도 위배되지 않았다. 동점인 가장 긴 시퀀스 중 무엇이든 "가장 긴 것을 축출한다"를 만족한다. 잃은 것은 **어느 사용자의 요청이 희생됐는가의 재현성**이고, 선점을 조사하는 사람이 정확히 필요로 하는 정보다.

도달 가능성이 #1291의 네 지점과 이것을 가른다. 그쪽 키는 호출당 한 번 찍히는 `Instant`라 사실상 충돌하지 않는다. 여기 키는 자주 충돌하는 작은 정수다.

## 2. 기술적 판단

### 2.1 최소 `seq_id`, 그리고 `created_at`을 기각한 이유

`seq_id`는 단조 증가하지만 `try_evict_for_preemption`이 victim을 **새로운, 더 큰** id로 재할당한다. 따라서 큰 id는 늦게 도착한 것이 아니라 최근에 선점된 것을 뜻한다. 최소 id 우선은 아직 안 맞은 시퀀스로 선점을 돌린다.

운영자에게 더 의미 있는 키로 `created_at`을 검토했다가 정반대 이유로 기각했다. 이 필드는 선점을 넘어 살아남으므로, 최소 `created_at` 우선은 같은 최고령 요청을 영원히 계속 고른다. 선점당한다고 필드가 움직이지 않기 때문이다. 게다가 `Instant`라 어차피 뒤에 `seq_id`를 놓아야 한다. 두 갈래가 같은 방향으로 수렴하므로 선택이 정책에 따라 달라지지 않는다.

### 2.2 같은 방향을 위해 서로 반대로 향하는 타이브레이크 성분

`max_by_key`는 마지막 최댓값을, `min_by`는 첫 번째 최솟값을 반환한다. 그래서 같은 결과를 내려면 두 갈래의 타이브레이크 성분이 서로 반대를 향해야 한다.

- `LongestFirst`는 `(generated_tokens.len(), std::cmp::Reverse(seq_id.as_u64()))`로 키잉한다. 그 키의 최대는 가장 긴 시퀀스이고, 같은 길이에서는 `Reverse(id)`가 최대인 것, 즉 **id가 최소**인 것이다.
- `LowestPriority`는 `.then_with(|| a.seq_id.as_u64().cmp(&b.seq_id.as_u64()))`를 오름차순으로 덧붙인다. 최소는 가장 낮은 우선순위, 다음 가장 긴 것, 다음 **id 최소**다.

두 키가 이제 전순서이므로 마지막-최댓값 규칙도 첫-최솟값 규칙도 발동할 여지가 없다. `SequenceId`는 `Ord`를 파생하지 않아 둘 다 `.as_u64()`를 거친다. #1291에서 `PromptCacheKeyDigest`가 놓았던 것과 같은 함정이다.

### 2.3 정책 사본은 하나, 필터는 그 안에

기존 테스트 둘은 비공개 메서드를 호출하지 않고 선택 표현식을 인라인으로 재구현했고 이미 드리프트했다. 프로덕션에 있는 `.filter(|seq| seq.structured.is_none())` 가드가 둘 다 없었다. 그 형태로 쓴 회귀 테스트는 사본을 고정했을 것이다.

정책은 이제 자유 함수 `select_eviction_victim_from(sequences, policy)`이고 메서드는 한 줄 호출로 줄었다. `ActiveBatch`의 메서드가 아니라 자유 함수인 이유는 `active.rs`가 현재 `PreemptionPolicy`를 전혀 모르기 때문이다. `structured.is_none()` 필터는 함수 **안으로** 옮겨서 어떤 호출자도 다시 잃을 수 없게 했고, 문서 주석에 정책은 여기서 확장하고 호출부에서는 절대 하지 말라고 적었다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `src/server/batch/scheduler.rs` | `select_eviction_victim_from` 추출, 두 갈래에 전순서 부여, 필터 내부 이동, `seq_id`를 전순서 성분으로 지목하는 주석 |
| `src/server/batch/scheduler_tests.rs` | 드리프트한 테스트 둘을 추출 함수 호출로 재작성, 신규 테스트 3건 |
| `docs/code-guidelines.md` | 인스턴스 목록을 여덟으로, "일곱"에 의존하던 산술도 함께 |

## 4. 리뷰 지적사항

테스트를 수정 전에 먼저 만들었고, 수정 후 되돌려 확인하는 방식이 아니었다. 그래서 `git stash`도 파일 수술도 필요 없었다. 미수정 표현식에 대고:

```
LongestFirst resolved a fully tied batch to something other than seq_id 2 on 58 of 64
freshly built batches; the tie is falling through to HashMap order

LowestPriority resolved a batch tied on priority and length to something other than seq_id 2
on 39 of 64 freshly built batches; the tie is falling through to HashMap order
```

테스트가 첫 실패에서 패닉하지 않고 64회 전체의 불일치를 누적한다. 그래서 "64 중 N"이라는 수치가 나오고 수정 전 실패율이 그대로 보인다.

기록할 것 셋:

가이드라인의 인스턴스 수를 여덟으로 올리자 다른 문장 셋의 산술이 틀리게 됐다. 이슈가 예상하지 못한 부분이다. 의존하는 수치를 고쳤고, 측정 문장은 "the first five fixes above (#1293 was found after the measurement ran)"로 좁혔다. #1287의 측정을 그것이 재지 않은 것까지 덮는 것처럼 조용히 재진술하지 않기 위해서다.

오독을 막으려고 강제 조항 기록에 단서를 하나 더했다. #1293은 **실제로 `max_by_key`를 쓴다.** 따라서 후보 검사의 "0 of 8"은 그 규칙이 해당 메서드를 무시해서가 아니다. 수신자가 리터럴 `.values()`가 아니라 모듈 간 접근자(`ActiveBatch::iter_sequences`)라서 놓친 것이고, 같은 절이 #1277에 대해 이미 서술한 한계다.

`structured.is_none()` 가드가 이제 실행 경로에 있지만 어떤 테스트도 `Some(..)`을 만들지 않는다. `StructuredOutputConstraint`에 테스트 생성자가 없고, 만들려면 실제 토크나이저와 컴파일된 문법이 필요하다. 암묵으로 두지 않고 PR 본문에 적었다.

## 5. 검증

GB10(DGX Spark, CUDA sm_121, Linux aarch64)에서 실측. 게이트 시점에 브랜치가 `main`과 동기였고, 박스에 다른 cargo 프로세스 없이 돌렸다.

- `make verify-test-cuda`: PR 스레드에 기록.
- `cargo test --profile test-fast --features cuda --lib server::batch::scheduler`: 111 통과, 7 무시, exit 0. `server::batch`: 373 통과, exit 0. 축출 필터만: 5 통과.
- `cargo fmt --all -- --check`, `cargo check --lib --tests --features cuda`, `cargo clippy --lib --tests --features cuda -- -D warnings`: 전부 exit 0.

기존 축출 테스트 둘은 예상 victim이 바뀌지 않았다. 둘 다 동점을 만들지 않기 때문이다.

## 6. 관련 작업

- #1293: 이 PR이 닫는 이슈.
- #1287과 PR #1290, 그리고 PR #1294의 정정: 이 인스턴스가 추가되는 가이드라인, 그리고 수정이 의존하는 `max_by_key` 방향 사실.
- #1291과 PR #1299: 잠재적 형제 넷. 동점이 도달 불가라 테스트에 대해 반대 판단을 했다.
- #1265와 PR #1266, #1267과 PR #1269, #1276과 PR #1281, #1277과 PR #1284, #1286과 PR #1288: 계열의 나머지.
