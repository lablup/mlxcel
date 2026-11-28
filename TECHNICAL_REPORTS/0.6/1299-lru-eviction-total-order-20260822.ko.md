# 기술 보고서: PR #1299 - LRU 축출 victim 선택 네 곳에 전순서 부여

## 요약

LRU 축출 네 곳이 `HashMap` 위에서 `min_by_key`로 victim을 골랐다. `Iterator::min_by_key`는 첫 번째 최솟값을 반환하므로, 타임스탬프가 같은 엔트리는 `HashMap` 순회 순서로 갈렸고 그 순서는 `RandomState`가 맵 인스턴스마다 무작위화한다. 이제 네 곳 모두 맵 키를 덧붙여 전순서로 선택한다.

**이것은 live 결함이 아니었고 이 변경은 관측된 버그를 고치지 않는다.** 네 키가 모두 호출당 한 번 찍히는 `std::time::Instant`라 동점이 도달 불가였다. 이 변경이 없애는 것은 키가 호출당 `Instant`보다 거칠어지는 순간 실재하게 될 잠재적 취약성이다.

## 1. 문제

전부 정렬되지 않은 맵 위의 지점들이다.

| 지점 | 함수 | 맵 |
| --- | --- | --- |
| `src/server/prompt_cache/store.rs:315` | `evict_oldest` | `entries: HashMap<PromptCacheKeyDigest, EntrySlot>` |
| `src/server/prompt_cache/store.rs:336` | `evict_oldest_snapshot` | `snapshots` |
| `src/server/responses_store.rs:243` | LRU 축출 루프 | `HashMap<String, Entry>` |
| `src/server/conversation_store.rs:151` | `evict_to_capacity` | `HashMap<String, Entry>` |

지금 동점이 왜 도달 불가인지 적어 둔다. 나중에 심각도를 잘못 읽지 않도록. 각 writer가 호출당 `Instant::now()`를 한 번 계산해 엔트리 하나에만 찍고, `Instant`는 이 프로젝트가 대상으로 하는 플랫폼에서 나노초로 해석되므로 두 엔트리가 같은 값을 갖지 않는다. 이 패턴의 **도달 가능한** 짝은 #1293이며, 거기서는 키가 `generated_tokens.len()`이라는 작은 정수라 동점이 일상이다.

이 네 곳은 #1287에서 `min_by_key` 정적 검사 후보가 기각된 이유이기도 하다. 정확히 이것들만 플래그하고, live 인스턴스 일곱 중 하나도 못 잡으며, 게이트를 걸면 도입 당일 4건 중 4건을 억제해야 했다. 손으로 고치면 그 긴장이 남지 않는다.

## 2. 기술적 판단

### 2.1 키 타입에 따라 두 형태

`PromptCacheKeyDigest`는 `Clone, Copy, PartialEq, Eq, Hash`만 파생하므로(`src/server/prompt_cache/key.rs:54`) 명백해 보이는 튜플 키가 컴파일되지 않는다. 타입의 derive를 넓히는 대신 두 프롬프트 캐시 지점은 원시 바이트로 키잉한다: `min_by_key(|(digest, slot)| (slot.entry.last_used(), *digest.as_bytes()))`. `[u8; 32]`는 `Ord`이고 digest 타입은 원래 표면을 유지한다.

`String` 키를 쓰는 두 저장소는 비교자 형태를 쓴다. `min_by_key`라면 원소마다 `String`을 튜플에 클론해야 하기 때문이다.

두 형태 모두 `docs/code-guidelines.md`의 "HashMap Iteration Order"에 옳은 것으로 기록돼 있고, 그 선택은 명시적으로 취향 문제다. 각 지점에 타이브레이크 성분이 전순서를 만드는 것임을 지목하는 주석을 달았다. 같은 가이드라인의 요구이며, 없으면 잉여로 읽혀 다음 사람이 지운다.

### 2.2 회귀 테스트 없음, 의도적

이슈가 이것을 요구가 아니라 판단 사항으로 규정했고, 여기서의 판단은 넣지 않는 것이다. 동점을 만들려면 같은 `Instant`를 비공개 상태에 직접 써 넣어야 하고, 이는 실제 쓰기 경로(`touch`, `insert`, `get`, `append`)를 전부 우회한다. 그렇게 만든 테스트는 도달 가능한 프로덕션 상태가 아니라 비교자의 동작을 고정할 뿐이고, 애초에 존재하려면 비공개 필드 접근이나 테스트 전용 setter가 필요하다. 부재가 실수가 아니라 결정이 되도록 근거를 PR 본문에 적었다.

#1288과는 반대 판단이다. 거기서는 동점이 도달 가능했고 테스트 여덟 개가 가능하면서 필요했다. 구분 기준은 노력이 아니라 **도달 가능성**이다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `src/server/prompt_cache/store.rs` | 두 지점을 `(last_used(), *digest.as_bytes())`로 키잉, 주석 포함 |
| `src/server/responses_store.rs` | 맵 키로 타이브레이크하는 비교자 형태, 주석 포함 |
| `src/server/conversation_store.rs` | 동일 |

세 파일에 33줄 추가, 5줄 제거. 공개 API 변경 없음, 타입 derive 확장 없음.

## 4. 리뷰 지적사항

가정하지 않고 확인한 구현 세부 하나를 기록해 둔다. `min_by_key`는 클로저를 원소당 한 번 호출하므로 `slot.entry.last_used()`가 잡는 뮤텍스 횟수는 이전과 같다. digest를 키에 접는다고 락이 늘지 않는다.

이슈의 #1248 조정 메모는 구현 시점에 재확인했다. 여전히 `status:ready`, PR 없음, 진행 중 아님. 그래서 네 지점이 전부 이 PR에 남았고 둘이 그쪽으로 옮겨가지 않았다.

## 5. 검증

GB10(DGX Spark, CUDA sm_121, Linux aarch64)에서 실측. 게이트 시점에 브랜치가 `main`과 동기였다.

- `make verify-test-cuda`: PR 스레드에 기록.
- `cargo test --profile test-fast --features cuda --lib server::prompt_cache`: 168 통과, exit 0. `server::responses_store`: 8 통과. `server::conversation_store`: 5 통과. 뒤의 둘은 `cargo test`가 필터를 하나만 받으므로 미리 빌드한 바이너리로 실행.
- `cargo fmt --all -- --check`, `cargo check --lib --tests --features cuda`, `cargo clippy --lib --tests --features cuda -- -D warnings`: 전부 exit 0.

## 6. 관련 작업

- #1291: 이 PR이 닫는 이슈.
- #1287과 PR #1290: 계열과 허용 해법 형태를 기록한 가이드라인. 거기서 기각된 정적 검사가 정확히 이 네 지점을 플래그했다.
- #1293: `BatchScheduler::select_eviction_victim`의 도달 가능한 동점 형제. 평범한 배치 상태에서 동점이 나므로 테스트가 필요해 별도로 처리한다.
- #1265와 PR #1266, #1267과 PR #1269, #1276과 PR #1281, #1277과 PR #1284, #1286과 PR #1288: 같은 계열의 live 인스턴스 일곱.
