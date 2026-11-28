# PR #2471: HashMap 순서 절의 인용을 현재 트리에 맞게 갱신

**작성일**: 2026-10-20
**상태**: 구현 완료, PR 열림
**위험도**: Low (문서 전용, `docs/code-guidelines.md`의 한 절)

## 요약

#1706을 닫는다. `docs/code-guidelines.md`의 "HashMap Iteration Order" 절은 이 결함 부류에 CI 게이트를 두지 않기로 한 이유(#1287)를 기록하는데, 그 논증은 구체적 근거에 기대고 있다. LRU 축출(eviction) 지점 네 곳과 그 key 타입, 파일 수, 테스트와 Makefile을 가리키는 file:line 참조 몇 개가 그것이다. 이 가운데 여러 개가 트리와 어긋나 있었다. 네 지점 중 둘은 명시적 전순서(total order)로 바뀌었고(#1299) 나머지 둘은 `BTreeSet<LruKey>` 색인으로 대체되었으며(#1519), 파일 수는 측정 당시 트리의 크기인데 현재 값처럼 적혀 있었고, 줄 번호 참조 셋은 엉뚱한 줄을 가리켰다. 이제 이 절은 오늘의 트리에 대해 참인 것을 적고, 그 전환들이 결정에 어떤 의미인지도 말한다.

## 문제 정의

코드와 대조해 본 독자는 `src/server/prompt_cache/store.rs:315`와 `:336`이 관련 없는 줄을 가리키고, `responses_store.rs:243`과 `conversation_store.rs:151`은 `min_by_key` 호출이 아예 없는 파일을 가리키며, "네 key가 모두 `std::time::Instant`"라는 주장은 #1299가 `prompt_cache` key에 entry digest를 덧붙인 순간부터 거짓이 되었음을 발견했다. 결정을 정직하게 지키는 것이 임무인 절은 인용이 하나라도 틀리는 순간 그 자격을 잃는다. 이 절 스스로가 경고하는 실패 양상("누군가 확인하는 순간 규칙은 신뢰를 잃는다")이다.

## 기술적 결정

- 네 지점을 함수 이름(`evict_oldest`, `evict_oldest_snapshot`)과 그 뒤의 운명으로 설명하고, 살아남은 두 `min_by_key` 호출에는 현재 줄 번호를 적는다. 다시 어긋날 줄 번호 넷만 고쳐 가리키는 데서 그치지 않는다.
- 1,248이라는 파일 수는 #1287이 측정한 트리의 크기로 두고 그렇게 밝히며, 이후 2,000을 넘었다는 주석을 붙인다. 오늘의 수로 바꾸면 프로토타입을 오늘의 트리에서 다시 돌렸다는 주장이 되는데, 그런 일은 없었다.
- 오늘 다시 돌리면 무엇이 나올지 적는다. `store.rs`의 두 호출은 여전히 잡힌다. 정규식은 tuple key가 이제 전순서라는 사실을 볼 수 없기 때문이며, 두 번째 후보 검사가 #1286 지점들에서 부딪히는 한계와 같다. 두 store에서는 아무것도 잡히지 않는다. 그래서 전환은 결정을 뒷받침하는 근거가 된다. 실제 버그 0건이 잡힌 버그 0건 더하기 영구 오탐 2건이 되었고, 이는 게이트를 두지 않기로 한 결정이 예측한 궤적 그대로다.
- "Choosing BTreeMap instead"에 컨테이너를 바꾼 실례로 #1519를 추가해, `HashMap`을 유지하고 정렬한 실례인 #1277과 나란히 둔다.
- 모든 줄 번호는 스크립트가 작업 트리에서 계산했다(인용된 상수, 호출, 주석 문구를 검색). 같은 스크립트가 두 store에 대한 구조적 주장도 단언했으므로, diff의 숫자 중 손으로 친 것은 없다.

## 변경 사항

- `docs/code-guidelines.md`의 "HashMap Iteration Order" 절만: enforcement 단락과 첫 번째 bullet을 위와 같이 고쳐 쓰고, "Choosing BTreeMap instead"에 `#1519` 문장을 추가하고, `src/lang_bias.rs`(755에서 774), `src/distributed/registry_tests.rs`(257에서 271), 철회된 #1265 결론을 기록한 `Makefile` 범위(604-616에서 713-725)의 줄 참조를 갱신했다. `sanitize.rs:388`과 `request_router_tests.rs:865`는 여전히 맞아서 그대로 두었다.

## 검증

- 스크립트가 인용된 상수 각각, `(slot.entry.last_used(), *digest.as_bytes())`를 key로 쓰는 두 `min_by_key` 호출, `store_budget.rs`의 `LruKey` 정의 범위, Makefile 블록을 찾아 그 숫자를 본문에 넣었다.
- 같은 스크립트가 `responses_store.rs`와 `conversation_store.rs`에 `min_by_key` / `max_by_key` 호출이 없고, `BTreeSet<LruKey>`를 들고 있으며, `state.lru.iter().next()`로 축출한다는 것을 단언했다.
- 옛 인용 각각을 절 안에서 `grep`하면 아무것도 나오지 않는다.
- Rust 소스는 바뀌지 않았다.

## 후속 조치

- 소스 파일을 가리키는 줄 참조는 그 위쪽이 편집될 때마다 어긋난다. 이 절은 기호(상수 이름, 함수 이름)만 인용하고 줄 번호를 빼는 쪽으로 갈 수 있다. 이 PR은 절의 기존 양식을 유지했고 그 변경은 하지 않았다.

## 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 1 |
| 추가된 라인 | 5 |
| 삭제된 라인 | 5 |
| 테스트 추가 | 0 |

## 관련

- Issue #1706: docs: the code-guidelines HashMap-order section cites line numbers and counts that no longer exist.
- Issue #1287: 이 절이 기록하는 결정.
- PR #1299: LRU 축출 `min_by_key` 호출 네 곳에 전순서 부여.
- PR #1519: `BTreeSet<LruKey>` 색인을 갖춘 바이트 상한 response store.
