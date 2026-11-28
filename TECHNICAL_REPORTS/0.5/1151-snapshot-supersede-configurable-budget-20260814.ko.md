# 기술 보고서: PR #1151 - feat(server): supersede session snapshot chains and expose the budget

**날짜**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: 낮음 (스토어 규칙과 설정 노브의 추가 변경. 기본값은 그대로이고 dense-KV 계열은 건드리지 않음)

---

## 요약

exact-prefix 스냅샷 스토어는 512 MiB 고정 예산으로 돌았고, 같은 대화의 새 스냅샷이 이전 것을 대체한다는 개념이 없었다. 멀티턴 대화는 턴마다 직전 턴의 토큰 벡터를 확장한 스냅샷을 기증하므로, 스토어에는 체인 전체가 쌓이고 바이트 압력이 걸리면 LRU가 임의의 희생자를 골랐다. 31B급 스냅샷 하나가 실측 300-370 MB라서 둘째 턴의 삽입이 첫 턴의 엔트리를 밀어냈고, 동시 대화 두 개는 서로를 밀어냈다.

이번 변경은 삽입 시점에 세션 체인 supersede 규칙을 넣는다. 새 스냅샷이 들어오면 같은 세션에서 토큰 벡터가 새 벡터의 진부분 접두사(strict prefix)인 저장 엔트리를 전부 제거하되, 새 엔트리의 바이트를 계상하기 전에 수행한다. 대화의 정상 상태 점유는 턴 수와 무관하게 엔트리 하나가 된다. `PromptCacheConfig`에 이미 있던 스냅샷 필드 세 개(용량 바이트, 최대 엔트리, TTL)를 두 서버 바이너리의 CLI 플래그와 `MLXCEL_*` 환경변수로 연결했고, `/v1/cache/stats`에 `snapshot_supersedes` 카운터를 새로 실어 결정적 대체와 실제 예산 압력을 구분할 수 있게 했다.

이슈의 "모델 인지 기본 예산" 하위 요구는 어중간하게 만들지 않고 훅 지점을 문서화한 채 명시적으로 미뤘다. 기술적 선택 절 참조.

---

## 1. 문제 정의

### 1.1 배경

에픽 #1148은 snapshot-only 계열(Gemma 4, Qwen 3.5, SSM 하이브리드 등 `supports_snapshot_reuse()`를 보고하는 13개 계열)이 멀티턴 프롬프트 캐시에 전혀 적중하지 못함을 실측으로 확정했다. #1146은 그중 용량 위생을 맡는다. 매칭이 형제 이슈들로 고쳐져도 예산 위생이 없으면 스래시가 남는다. `DEFAULT_SNAPSHOT_CAPACITY_BYTES`는 `DEFAULT_CAPACITY_BYTES / 4` = 512 MiB인데 31B 대화 스냅샷 하나가 대략 300-370 MB(실측 307,693,200 및 369,133,200 바이트)이고, 운영자가 조정할 수단이 어느 바이너리에도 없었다.

### 1.2 기존 문제

- **체인이 쌓였다.** 턴마다 기증되는 벡터가 직전 것을 엄격히 확장하므로 대화 하나가 턴당 엔트리 하나를 차지했다. 최신 것을 제외하면 전부 죽은 무게였다. 조회는 가장 긴 것에만 적중할 수 있는데도 각각이 바이트 예산을 소비했다.
- **LRU가 엉뚱한 희생자를 골랐다.** 압력이 걸리면 확장 중인 엔트리가 무관한 엔트리만큼 쉽게 밀려났다. 31B 모델에서 관측된 `snapshot_inserts = 2, snapshot_entries = 1`이 바로 둘째 턴이 첫 턴을 밀어낸 흔적이다.
- **운영자 제어가 없었다.** 용량, 엔트리 수, TTL이 `PromptCacheConfig`에 있었지만 형제 `prompt_cache_*` 노브와 달리 CLI나 환경변수로 닿을 수 없었다.
- **통계로 진단이 안 됐다.** 모든 제거가 `snapshot_evictions_lru`로 잡혀서, 건강한 세션 내 대체와 실제 예산 압력을 구분할 수 없었다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|---|---|---|
| 동시 31B 대화 두 개가 서로의 스냅샷을 스래시 | 높음 | 높음 (변경 전) |
| 대화 중간 분기(이전 턴 수정·재생성) 시 supersede가 아직 유효한 조상을 제거 | 낮음 | 낮음 |
| 익명 세션 공유로 한 호출자가 다른 호출자의 스냅샷을 대체 | 낮음 | 낮음 |

---

## 2. 기술 검토

### 2.1 supersede 규칙은 의도적으로 좁다

`insert_snapshot`은 세 조건이 동시에 성립할 때만 저장 엔트리를 제거한다. 같은 sessionless 버킷(모델, LoRA, 템플릿, 멀티모달 정체성), 같은 non-`None` `session_key`, 그리고 저장된 토큰 벡터가 들어오는 벡터의 진부분 접두사일 것. `None` 세션은 체인을 걸 대화 정체성이 없으므로 규칙을 발동시키지 않는다. 서로 다른 세션은 절대 서로를 건드리지 않으며, 테스트가 교차 세션 격리로 이를 고정한다.

익명 세션 케이스는 가정으로 넘기지 않고 검토했다. `prompt_cache_key`도 `user`도 없는 요청은 `lookup_longest_prefix`에서 이미 그러듯 `ANONYMOUS_SESSION_SENTINEL` 하나를 공유한다. 그런 호출자가 다른 호출자의 엔트리를 대체하려면 저장된 벡터 전체(프롬프트와 생성 꼬리)가 상대 벡터의 진부분 접두사여야 하는데, 실제로는 같은 대화록의 진짜 이어쓰기에서만 성립한다.

### 2.2 용량 검사와의 순서

supersede는 새 엔트리의 바이트를 계상하기 전에 돈다. 그래서 `enforce_snapshot_caps`가 해제된 바이트를 보고, 자기 조상과 나란히는 못 들어갔을 확장을 수용할 수 있다. 이 순서가 512 MiB 예산을 "31B 스냅샷 하나는 들어가고 둘째 턴부터 스래시"에서 "31B 대화 하나가 정상 상태로 상주"로 바꾼다. 조상의 바이트가 먼저 해제되어야만 들어갈 수 있는 확장을 삽입하는 전용 테스트가 이 바이트 계상을 증명한다.

### 2.3 카운터 분리는 진단 계약이다

`snapshot_supersedes`는 스토어 내부, `PromptCacheStats`, `/v1/cache/stats`까지 끝단 전체에서 `snapshot_evictions_lru`와 분리되어 있다. supersede는 세션 내 결정적 대체이고 LRU 축출은 용량 압력이다. 둘을 합치면 새 플래그가 열어준 튜닝 워크플로에서 축출 카운터를 읽을 수 없게 된다. 귀속 대조도 라이브로 수행했다. 동일 요청 반복은 `snapshot_inserts`와 LRU 카운터를 올리면서 `snapshot_supersedes`는 0에 머물러, 새 카운터가 멱등 대체를 이중 계상하지 않음을 확인했다.

### 2.4 수용한 트레이드오프 하나

대화 중간 분기(이전 턴을 수정하거나 재생성하는 경우)는 제거된 조상을 잃고 re-prefill 한 번을 지불한다. 대화 점유를 엔트리 하나로 유지하는 값이며, 다음 독자가 재발견하도록 두지 않고 규칙 자리의 코드 주석에 기록했다.

---

## 3. 기술적 선택과 그 이유

### 3.1 더 똑똑한 축출 정책이 아니라 삽입 시점 supersede

대안은 축출 시점에 LRU에 체인 개념을 가르치는 것이었다. 삽입 시점 제거가 엄격히 단순하다. 필요한 정보(이 엔트리가 무엇을 대체하는가)가 삽입 시점에 온전히 있고, 해제된 바이트가 그 삽입 자신에게 바로 쓰이며, 축출은 순수한 용량 메커니즘으로 남는다.

### 3.2 모델 인지 기본값은 정직하게 미뤘다

제안 해법 (b)는 모델의 토큰당 상태 크기에서 기본 예산을 유도하라고 했다. PR은 쓸 만한 훅 지점을 문서화했고(`start_server`가 `model_path`를 쥔 채 스토어를 만들며, 바로 위 코드가 이미 hybrid-SSM APC 자동 비활성화를 위해 `config.json`을 읽는다), 빠진 것이 무엇인지도 명시했다. 13개 계열 각각의 검증된 상태 크기 공식이다. 이 공식이 틀리면 스토어가 조용히 잘못 크기 잡히므로 기본값은 512 MiB로 두고, 운영자 요구는 노브와 실측 기반 사이징 안내(모델 폭에 상수인 `snapshot_bytes`를 재서 상주 엔트리 수를 곱하라)로 채웠다. 초기 버전은 설정 해석 시점 제약을 근거로 들었는데 리뷰에서 거짓으로 드러나 PR 본문에 정정을 남겼다.

### 3.3 플래그는 기존 패턴을 그대로 따른다

플래그 세 개와 환경변수는 두 바이너리(`src/main.rs`, `src/bin/mlx_server.rs`)에서 형제 `prompt_cache_*` 선례를 그대로 따른다. CLI가 env를 이기는 우선순위와 파싱 불가 값 폴백까지 동일해서 표면이 배우기 쉽고 테스트도 기존 픽스처를 재사용했다.

### 3.4 리뷰 지적은 미루지 않고 반영했다

pr-reviewer와 pr-security-checker가 독립적으로 돌았고 CRITICAL/HIGH는 없었다. MEDIUM 둘은 문서의 같은 결함이었다. 예산이 실제로는 엔트리 단위로 소비되는데 사이징 안내가 대화 단위로 쓰여 있어, 이 이슈가 없애려는 바로 그 스래시를 유발할 안내였다. `ead5ac9f`, `8af676e4`, `3efebe30`에서 수정했고 `snapshot_supersedes` 와이어 포맷을 고정하는 LOW 하나도 함께 처리했다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 14 |
| 라인 | +609 / -4 |
| 새 CLI 플래그 / 환경변수 | 3 / 3 (두 바이너리) |
| 새 통계 필드 | 1 (`snapshot_supersedes`) |
| 새 스토어 테스트 | 5 |
| 새 cli_input 테스트 | 5 |
| 기본 동작 변화 | 0 (supersede는 세션 키와 진부분 접두사 체인이 있을 때만 발동) |

### 영역별 변경

**`src/server/prompt_cache/store.rs`**
- `insert_snapshot`의 세션 체인 supersede(바이트 계상 전 수행), `Inner`와 `stats()`를 관통하는 `snapshot_supersedes` 카운터.

**`src/server/prompt_cache/policy.rs`, `src/server/routes/cache.rs`**
- `PromptCacheStats`와 `/v1/cache/stats`의 `snapshot_supersedes`.

**`src/server/cli_input.rs`, `src/main.rs`, `src/bin/mlx_server.rs`, `src/commands/serve.rs`**
- `--prompt-cache-snapshot-capacity-bytes`, `--prompt-cache-snapshot-max-entries`, `--prompt-cache-snapshot-ttl`과 `MLXCEL_*` 폴백, `with_snapshot_limits(...)`를 거쳐 `build_prompt_cache_config`로 연결.

**`src/server/prompt_cache/entry.rs`**
- 테스트 전용 `ModelSnapshotEntry::new_for_test`. 예산 테스트가 MLX 텐서 할당 없이 계상 크기를 제어한다.

**`docs/environment-variables.md`**
- 새 변수 세 개와 사이징 안내(`snapshot_bytes` 실측 후 예상 상주 엔트리 수를 곱하기).

---

## 5. 검증과 후속

### 통과

- `cargo test --release --lib prompt_cache::store::tests`: 28 통과 (신규 5: 체인이 엔트리 하나로 붕괴, 교차 세션 격리, 세션 키 없으면 미발동, 진확장 요건, 용량 검사 전 바이트 계상).
- `cargo test --release --lib server::cli_input`: 98 통과 (신규 5).
- `cargo test --release --lib server::routes::cache`: 17 통과.
- `cargo clippy --lib --tests -- -D warnings`, `cargo fmt` 클린.
- 실물 체크포인트(`qwen3.5-0.8b-4bit`, 포트 18146): 플래그 경로가 끝단까지 동작(기동 로그와 `/v1/cache/stats`가 컴파일 기본 512 MiB 대신 설정된 40 MB를 보고). 스냅샷 크기는 턴 벡터 72, 98, 124, 151 토큰에 걸쳐 13,246,464 바이트로 일정해, 비용이 프롬프트 길이가 아니라 모델 폭을 따름을 확인. 보고된 스래시는 작은 예산 아래 HEAD에서 재현됨(`snapshot_entries`가 고정된 채 `snapshot_evictions_lru` 상승).

### 미검증

- HTTP 종단 supersede. 가정이 아니라 실측이다. 기증 벡터가 생성 프롬프트와 생성 내용으로 끝나는데 다음 턴의 재렌더링 프롬프트가 그 지점을 재현하지 않아, 형제 경계 스냅샷 작업(#1143/#1144)이 들어오기 전에는 같은 세션의 진확장이 스토어에 도달하지 않는다. 규칙 자체는 스토어 계층에서 검증됨.
- 같은 이유로 HTTP 종단의 두 대화 공존.
- 모델 인지 기본값 (미룸, 3.2 참조).

### 후속

- #1143/#1144가 들어오면 정상 상태 주장(대화당 상주 엔트리 하나, 턴마다 `snapshot_supersedes` 증가)이 HTTP에서 관측 가능해지며 에픽 수준 검증에 포함할 것.
- 실측 `snapshot_bytes`로 검증한 계열별 상태 크기 공식이 문서화된 `start_server` 훅에서 모델 인지 기본값을 완성한다.
