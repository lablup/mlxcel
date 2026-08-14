# 기술 보고서: PR #1149 - feat(server): classify snapshot-divergence rejects in prompt-cache stats

**날짜**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: 낮음 (관측성 전용 변경. 기존 조회 호출부는 시그니처와 의미를 그대로 유지)

---

## 요약

요청과 같은 세션 버킷에 있으면서도 엔트리 끝에 이르기 전에 갈라지는 스냅샷 후보를, `lookup_snapshot_prefix`는 그냥 `None`으로 돌려보냈다. 그래서 `/v1/cache/stats`에서는 에픽 #1148이 모든 snapshot-only 계열에서 확인한 구조적 멀티턴 미스가 빈 스토어와 구분되지 않았다. 진단하려면 저장된 토큰 벡터를 손으로 detokenize해서, 후보가 존재하고 긴 공통 접두사를 공유하며 템플릿 산출물에서 갈라진다는 사실을 직접 찾아야 했다.

이번 변경으로 스토어가 본 것을 그대로 보고한다. 새 `SnapshotLookupOutcome`이 `Hit`, `Diverged`(`common_prefix_len`과 `stored_len` 동반), `NoCandidate`를 구분하고, 스케줄러는 가운데 경우에만 분류된 `snapshot_diverged` 리젝트를 기록하며, 그 기하 정보가 `/v1/cache/stats`(`reject_snapshot_diverged`, `last_reject_context_len` / `last_reject_entry_len`)와 `/metrics`의 `reason="snapshot_diverged"` 시리즈로 드러난다. 서로 다른 하이브리드 경로의 두 계열에서 라이브로 검증했고, 진짜 히트가 카운터를 0에 붙잡아 두는 음성 대조까지 포함한다.

---

## 1. 문제 정의

### 1.1 배경

에픽 #1148의 범위 검증에서 조사한 모든 snapshot-only 계열(gemma-4-31b, 두 thinking 모드의 qwen3.5-4b, falcon-h1-tiny)이 멀티턴 프롬프트 캐시에 적중하지 못했고, 각 미스의 원인은 서로 다른 발산 클래스였다. 생성 프롬프트에만 붙는 스캐폴드 토큰, 히스토리에서 제거되는 thinking 블록, 재토크나이즈 드리프트. 어느 경우든 `snapshot_lookups`는 오르고 `snapshot_hits`는 0에 머물렀는데, 이는 차가운 스토어가 보이는 모습과 정확히 같다. 스토어는 후보를 순회하며 발산 지점을 이미 찾아 놓고도 그 정보를 버렸다.

### 1.2 기존 문제

- **가장 흔한 실패 모드에 신호가 없었다.** 13개 계열에서 "후보는 있는데 갈라진다"가 둘째 턴의 기본 결과인데, 통계 표면 어디에도 카운터도 사유도 기하 정보도 남지 않았다.
- **진단 비용이 과했다.** 에픽의 미스를 확정하려면 저장 벡터를 덤프하고 손으로 detokenize해서 발산 지점을 찾아야 했다. gemma-4-31b의 90/139 기하도 그렇게 찾았다.
- **후속 작업이 이 분류를 필요로 했다.** 부분 복원 이슈(#1145)는 "갈라졌지만 잘라낼 수 없음"을 "후보 자체가 없음"과 다른 결과로 보고해야 하는데, 결과 타입이 없으니 놓을 자리가 없었다.
- **오분류가 실제 위험이었다.** 순진한 구현은 차가운 스토어나 남의 세션 버킷까지 발산으로 셀 수 있고, 그러면 새로 넣는 신호 자체가 오염된다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|---|---|---|
| 고장난 캐시와 구분할 수 없는 캐시를 상대로 운영자가 튜닝하거나 버그를 제기 | 높음 | 높음 (변경 전) |
| 새 리젝트가 빈 스토어나 다른 세션에서 발화해 카운터를 못 읽게 됨 | 중간 | 낮음 (테스트로 고정) |
| 조회 경로 변경이 기존 호출자의 히트 동작을 바꿈 | 중간 | 낮음 (기존 진입점을 래퍼로 유지) |

---

## 2. 기술 검토

### 2.1 불리언 추가가 아니라 결과 타입

`lookup_snapshot_outcome`은 `SnapshotLookupOutcome::{Hit, Diverged(SnapshotDivergence), NoCandidate}`를 반환하고, `SnapshotDivergence`는 `common_prefix_len`과 `stored_len`을 담는다. 3분할 자체가 설계다. `NoCandidate`가 빈 스토어와 남의 세션 버킷을 의도적으로 함께 덮으므로, 후보가 애초에 없던 요청에 호출자가 발산을 기록할 방법이 없다. `lookup_snapshot_prefix`는 옛 시그니처 그대로의 얇은 래퍼로 남아, 히트만 원하는 호출자는 아무 영향도 받지 않는다.

### 2.2 기하 정보는 공짜로 나온다

공통 접두사 길이는 조회가 이미 수행하는 비교에서 그대로 떨어진다. 토큰 벡터를 다시 훑는 추가 패스는 없다. 버킷에 후보가 여럿이면 가장 긴 공통 접두사를 가진 후보의 수치를 보고한다. 운영자가 알고 싶은 것, 즉 가장 가까운 엔트리가 얼마나 가까웠는지가 바로 그 값이다.

### 2.3 기록 경로는 기존 리젝트 배관을 재사용한다

`PromptCacheRejectReason::SnapshotDiverged`는 기존 `PromptCacheRejectCounters` 메커니즘에 올라탄다. `PromptCacheLastReject`는 새 `record_detailed`를 통해 `entry_len`을 얻고, 기존 `record`는 `None`으로 위임하므로 이전 호출부의 동작은 하나도 바뀌지 않는다. 스케줄러에서는 `try_adopt_cached_prefix`가 결과를 소비해 `Diverged`일 때만 분류된 리젝트를 기록한다.

### 2.4 음성 대조가 핵심 증거다

클래스 (c) 재토크나이즈 드리프트는 데이터 의존적이다. 답변이 동일하게 재토크나이즈될 수도, 안 될 수도 있다. falcon-h1-tiny의 둘째 턴 답변은 동일하게 재토크나이즈됐고, 조회는 올바르게 히트를 돌려줬으며, `reject_snapshot_diverged`는 0에 머물렀다. 이 행이 증명하는 바는 분류기가 스냅샷 경로가 돌 때마다가 아니라, 저장 엔트리 중 어느 것도 요청의 접두사가 아닐 때만 움직인다는 것이다. 차가운 스토어 행은 빈 스토어가 아무것도 내보내지 않음을 증명한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 분류는 스토어에서, 기록은 스케줄러에서

후보를 보는 것은 스토어뿐이므로 결과 계산은 스토어가 맡고, 실제 요청을 처리 중임을 아는 것은 스케줄러뿐이므로 기록은 스케줄러가 맡는다. 이 분담 덕에 테스트나 내부 조회가 운영자용 카운터를 부풀리지 않는다.

### 3.2 카운터 하나가 아니라 기하 정보

이슈는 단순 카운트가 아니라 `common_prefix_len`과 `stored_len`을 함께 실으라고 요구했고, 라이브 수치가 그 이유를 보여준다. qwen3.5의 33/95와 falcon-h1의 23/139는 발산 지점이 히스토리 경계 직후임을 즉시 짚어 준다. 이전에는 detokenize를 해야 얻던 진단이다. `last_reject_context_len` / `last_reject_entry_len`이 로그를 뒤지지 않고도 최근 기하를 보여준다.

### 3.3 에픽의 검증 시나리오가 픽스처가 됐다

`snapshot_divergence_tests.rs`는 스토어가 비교하는 토큰 벡터 수준에서 세 발산 클래스를 고정하며, 에픽의 라이브 검증에서 나온 gemma-4-31b의 90/139 기하와 오분류 방지 케이스도 포함한다. 회귀 스위트가 합성 형태가 아니라 실측 형태를 담는다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 9 |
| 라인 | +702 / -21 |
| 새 통계 필드 | 3 (`reject_snapshot_diverged`, `last_reject_entry_len`, `/metrics`의 `reason="snapshot_diverged"`) |
| 기존 조회 동작 변화 | 0 (`lookup_snapshot_prefix`가 새 결과를 감쌈) |
| 새 테스트 파일 | 1 (`snapshot_divergence_tests.rs`, 343라인) |

### 영역별 변경

**`src/server/prompt_cache/store.rs`**
- `SnapshotDivergence`, `SnapshotLookupOutcome`, 같은 세션의 최선 발산 후보를 보고하는 `lookup_snapshot_outcome`. `lookup_snapshot_prefix`는 래퍼로 유지.

**`src/server/prompt_cache/metrics.rs`**
- `PromptCacheRejectReason::SnapshotDiverged`(라벨 `snapshot_diverged`), `PromptCacheLastReject.entry_len`과 `record_detailed`.

**`src/server/batch/scheduler.rs`, `src/server/batch/observability.rs`**
- `try_adopt_cached_prefix`의 결과 소비, `record_prompt_cache_reject_detailed`, `prompt_cache_reject_snapshot_diverged` 카운터, 최근 리젝트 스냅샷의 `entry_len`.

**`src/server/routes/cache.rs`, `src/server/routes/metrics.rs`**
- `/v1/cache/stats`의 `reject_snapshot_diverged`와 `last_reject_entry_len`, `/metrics`의 `snapshot_diverged` 시리즈.

**`src/server/prompt_cache/snapshot_divergence_tests.rs`** (신규)
- 검증된 세 발산 클래스와 오분류 방지 케이스.

---

## 5. 검증과 후속

### 통과

- `cargo test --release --lib server::prompt_cache`: 153 통과.
- `cargo test --release --lib server::routes::cache`: 19 통과 (새 필드의 라우트 수준 커버리지).
- `cargo test --release --lib server::batch::observability`: 15 통과.
- `cargo clippy --lib --tests -- -D warnings`, `cargo fmt --check`.
- 실물 체크포인트를 프로덕션 `/v1/chat/completions` 경로로 검증. qwen3.5-0.8b-4bit(Attention + GatedDeltaNet)는 차가운 스토어에서 아무것도 내보내지 않고, `cached_tokens = 0`인 둘째 턴에서 리젝트 1과 기하 33/95를 기록. falcon-h1-tiny-90m-instruct-4bit(Mamba 하이브리드)는 둘째 턴이 진짜 히트라 카운터가 0에 머물고(음성 대조), 이른 발산 턴에서 기하 23/139를 기록. 서로 다른 하이브리드 경로의 두 계열이 모두 분류하므로 리젝트는 특정 템플릿의 성질이 아니라 계열 불문이다.

### 미검증

- 이슈가 지목한 gemma-4-31b-it-4bit 시나리오는 이 브랜치에서 라이브로 돌리지 않았다. 오케스트레이터의 대형 모델 패스가 맡는다. 대신 그 90/139 기하를 유닛 테스트 픽스처로 고정했다.
- 벽시계 영향은 재지 않았다. 추가 작업은 이미 수행하던 비교와 카운터 증가뿐이다.

### 후속

- #1145의 부분 복원은 이 변경이 도입한 결과 타입을 통해 "갈라졌지만 wrap되어 잘라낼 수 없음"을 보고하며, PR #1152로 들어갔다.
- 리젝트 사유와 라벨은 이제 안정 계약이다. 에픽 검증과 향후 회귀 점검이 `snapshot_diverged`를 키로 삼으므로, 이름을 바꾸면 진단 표면의 파괴적 변경이 된다.
