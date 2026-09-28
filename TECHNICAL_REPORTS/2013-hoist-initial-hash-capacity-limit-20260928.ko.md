# 기술 리포트: PR #2013 - refactor(server): hoist INITIAL_HASH_CAPACITY_LIMIT into store_budget

**작성일**: 2026-09-28
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: 낮음 (동작 변화 없음; 두 스토어가 동일한 상수를 참조하는지는 컴파일러가 검증한다)

---

## 요약

PR #2013은 이슈 #1665를 닫는다. 두 개의 바운디드 서버 스토어가 초기 `HashMap` 할당 크기를 제한하는 데 쓰는 `usize` 상수 `INITIAL_HASH_CAPACITY_LIMIT`는 `src/server/conversation_store.rs`와 `src/server/responses_store.rs`에 각각 동일하게 선언되어 있었다. 이제 두 스토어가 이미 가져다 쓰던 공유 헬퍼 모듈인 `src/server/store_budget.rs`에 단일 정의로 존재하며, 두 호출 지점은 상수를 재선언하는 대신 그곳에서 가져온다.

---

## 1. 문제 정의

### 1.1 배경

`ConversationStore`와 `ResponsesStore`는 `HashMap::with_capacity(max_entries.min(INITIAL_HASH_CAPACITY_LIMIT))`로 내부 `HashMap`의 초기 크기를 미리 잡아 두는 바운디드 인메모리 스토어다. 이렇게 해 두면 설정된 `max_entries`가 크더라도 실제 엔트리가 들어오기 전에 과도한 선(先)할당이 일어나지 않는다. `src/server/store_budget.rs`는 바로 이런 공유 관심사를 위해 이미 존재하는 모듈로, 모듈 독스트링에 "Shared helpers for bounded in-memory server stores"라고 명시되어 있고, 두 스토어는 이미 여기서 `LruKey`와 `serialized_json_len_saturating`를 가져다 쓰고 있었다.

### 1.2 기존 문제점

- **상수 중복.** `const INITIAL_HASH_CAPACITY_LIMIT: usize = 4096;`가 `conversation_store.rs:36`과 `responses_store.rs:50`에 각각 따로 선언되어 있었고, 둘 사이에 단일한 출처가 없었다.
- **사용처를 알려주는 기록이 없었다.** 두 선언 모두 이 값이 사실상 공유 개념이라는 점을 문서화하지 않았다. 한쪽 파일만 읽어서는 다른 파일에 동일한 상수가 있다는 사실을 알 방법이 없었다.

### 1.3 위험성 평가

| 위험 | 영향 | 발생 가능성 |
|------|------|------|
| 한쪽 스토어의 워크로드에 맞춰 값을 재조정하면서 다른 쪽은 그대로 두어, 두 바운디드 스토어의 용량 제한 동작이 조용히 달라짐 | 중 | 낮음 |
| 향후 세 번째 바운디드 스토어가 상수를 가져오지 않고 세 번째로 복제함 | 낮음 | 중 |

---

## 2. 기술적 선택과 그 이유

### 2.1 새 모듈이 아니라 store_budget.rs

**컨텍스트:** 이 상수는 두 스토어가 공유할 단 하나의 자리가 필요했다.

**선택 이유:** `store_budget.rs`는 두 스토어가 이미 가져다 쓰는 나머지 두 항목(`LruKey`, `serialized_json_len_saturating`)을 이미 담고 있고, 독스트링도 이미 "바운디드 인메모리 서버 스토어"로 범위를 잡고 있다. 상수를 여기에 추가하는 것은 기존 import를 확장하는 일일 뿐, `usize` 하나를 위해 네 번째 파일이나 새 공유 모듈을 만드는 것이 아니다.

**트레이드오프:** 실질적인 트레이드오프는 없다. 이 상수가 개념적으로 이미 속해 있던 모듈이기 때문이다.

### 2.2 `pub(crate)` 가시성과 "Used by" 독 코멘트

상수는 두 호출 지점과 동일한 크레이트 내부 가시성인 `pub(crate)`로 선언되며, 두 소비처를 경로로 명시하는 독 코멘트(`conversation_store::StoreState::with_capacity`, `responses_store::StoreState::with_capacity`)를 달고 있다. 이는 `docs/code-guidelines.md`의 공유 함수 컨벤션(`// Used by: ...` 형태의 발견 메커니즘)을 공유 함수가 아니라 공유 상수에 적용한 것이다: 다음에 이 값을 바꾸는 사람은 누가 이 값을 참조하는지 확인할 자리가 하나로 정해져 있다.

---

## 3. 변경 요약

### 통계

| 항목 | 값 |
|------|------|
| 변경된 파일 수 | 3 |
| 추가된 줄 | +14 |
| 삭제된 줄 | -6 |
| 동작 변화 | 0 |

### 파일별 변경 내용

| 파일 | 변경 내용 |
|------|------|
| `src/server/store_budget.rs` | `pub(crate) const INITIAL_HASH_CAPACITY_LIMIT: usize = 4096;`를 추가하고, 두 호출 지점을 명시하는 독 코멘트를 단다. |
| `src/server/conversation_store.rs` | 로컬 `INITIAL_HASH_CAPACITY_LIMIT` 선언을 제거하고, 기존 `LruKey`, `serialized_json_len_saturating` import에 더해 `store_budget`에서 상수를 가져온다. |
| `src/server/responses_store.rs` | `conversation_store.rs`와 동일한 변경. |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|------|
| `aa6faf5` | refactor | refactor(server): hoist INITIAL_HASH_CAPACITY_LIMIT into store_budget |

---

## 4. 검증

PR 설명에 따르면 작성자는 다음을 실행했다.

- `cargo check --lib --tests` (2코어 범위)
- `cargo clippy --lib --tests -- -D warnings` (2코어 범위)
- `cargo test --lib server::conversation_store` (13개 통과)
- `cargo test --lib server::responses_store` (16개 통과)
- `cargo fmt --check`
- `python3 scripts/ci/check_cross_repo_refs.py`

이 리포트는 위 검증을 독립적으로 다시 실행하지 않는다. 이 리포트 자체는 PR에 문서(리포트)만 추가하는 변경이며, 위 명령들은 PR 본문에 기록된 작성자 자신의 검증 결과다.

### 다루지 않은 것

- 새 테스트는 추가하지 않았고 필요하지도 않다: 값과 두 호출 지점에서의 사용 방식은 그대로이고 선언 위치만 옮겨졌기 때문이다. 두 스토어의 기존 통과 테스트(13개 + 16개)가 이 변경의 커버리지 역할을 한다.

---

## 5. 관련 자료

- 이슈 #1665: 이 변경의 출처가 된 이슈 ("`INITIAL_HASH_CAPACITY_LIMIT`가 두 바운디드 서버 스토어에 동일하게 선언되어 있다").
- `docs/code-guidelines.md`: 이 PR이 따르는 공유 함수/공유 항목 "Used by" 코멘트 컨벤션.
