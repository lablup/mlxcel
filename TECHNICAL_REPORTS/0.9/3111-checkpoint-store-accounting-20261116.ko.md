# 기술 리포트: PR #3111 - state checkpoint 집계와 orphan sweep 범위 제한

**작성일**: 2026-11-16
**상태**: PR 열림 (#3058 해결)
**검토자**: 독립 검토, CRITICAL/HIGH/MEDIUM 없음
**언어**: Rust
**위험도**: Low

---

## 요약

PR #3055는 하이브리드 모델용 턴 경계 state checkpoint를 도입했고, 네 가지 알려진 공백을 남겼다. 이 공백은 #3058로 함께 등록되었다. checkpoint 바이트가 스토어의 `len()`, `bytes()`, stats에 보이지 않았고 #2883 snapshot growth guardrail도 적용되지 않았다. orphan sweep은 매 insert마다 모든 checkpoint를 모든 snapshot과 비교했다. donor snapshot이 제거된 뒤에도 checkpoint가 남았다. rebuild에 실패하면 더 짧은 whole snapshot 히트 대신 cold prefill로 떨어졌다.

이 PR은 네 가지를 한 번에 고친다. checkpoint는 스토어 footprint에 포함되고 `snapshot_growth_allowed`를 거친다. orphan sweep은 변경된 conversation으로 한정된다. orphan은 donor와 함께 제거된다. rebuild가 실패하면 checkpoint를 버리고 checkpoint 없이 다시 조회하며, 히트는 rebuild 성공 후에만 센다. 변경은 6개 파일, +496/-63줄이고 새 테스트 6개가 있다.

---

## 1. 문제 정의

### 1.1 배경

checkpoint는 히스토리 경계의 recurrent 및 convolution 상태이다. conversation의 가장 최신 whole snapshot인 donor와 함께만 제공되며, donor가 prefix의 attention KV를 채운다. 조건에 맞는 donor가 없는 checkpoint는 쓸모가 없으므로 스토어가 이런 orphan을 제거한다.

### 1.2 기존 문제점

- checkpoint 바이트가 `store.bytes()`와 `store.len()` 밖에 있어 prompt-cache 게이지가 과소 보고했고, `insert_state_checkpoint`는 growth guardrail을 건너뛰었다.
- `drop_orphan_state_checkpoints`는 checkpoint insert 때만 실행되었고, write lock 아래에서 모든 checkpoint를 모든 snapshot과 비교했다.
- donor가 LRU, TTL, cap으로 제거되어도 해당 checkpoint는 자체 TTL이나 LRU까지 남았다.
- `adopt_state_checkpoint`가 실패하면 더 짧은 whole snapshot이 있어도 cold prefill을 했다. 또한 조회가 rebuild 전에 히트를 셌기 때문에 실패한 rebuild도 히트로 집계되었다.

### 1.3 위험성

| 위험 | 완화 |
|------|------|
| supersede 경로가 더 긴 후속 snapshot이 donor가 될 checkpoint를 orphan으로 만들 수 있음 | `insert_snapshot`은 `remove_snapshot_unswept`를 쓰고 새 항목 저장 후 한 번 sweep |
| sweep 범위를 좁히면 다른 conversation의 orphan을 놓칠 수 있음 | conversation의 donor는 해당 conversation의 변경으로만 바뀌므로 다른 bucket과 session은 영향받지 않음 |
| 계속 실패하는 checkpoint로 인한 재시도 반복 | 재시도 전에 실패한 checkpoint를 제거하고, 재시도 조회는 checkpoint를 제외 |
| 게이지에 checkpoint가 포함되어 운영자가 보는 값이 달라짐 | `docs/CONTINUOUS_BATCHING.md`에 문서화 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

격리 방식은 변하지 않는다. 범위 제한 sweep은 변경의 sessionless bucket과 session을 받아 일치하는 checkpoint만 `donor_for`에 넘기므로 기존 bucket 및 session 규칙이 유지된다. 다른 bucket과 session의 checkpoint는 sweep이 읽지 않으며, 테스트가 이를 고정한다.

### 2.2 성능 관점

- sweep 비용이 전체 checkpoint에서 한 conversation의 checkpoint로 줄어든다. 여전히 스토어 write lock 아래에서 실행된다.
- 재시도 경로는 rebuild 실패 후에만 조회를 한 번 더 수행한다.
- 히트 경로에 새 할당은 없다.

### 2.3 호환성/의존성 관점

새 의존성이나 환경 변수는 없다. `stats().entries`와 `bytes`가 checkpoint를 포함한다. 새 스토어 메서드: `state_checkpoint_growth_bytes`, `discard_state_checkpoint`, `record_state_checkpoint_hit`, `lookup_snapshot_outcome_without_checkpoints`, `remove_snapshot_unswept`.

### 2.4 코드 품질 관점

두 번째 gate를 만들지 않고 `snapshot_growth_allowed`를 재사용한다. 스케줄러는 게이지를 checkpoint insert 전이 아니라 후에 갱신한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 기존 합계에 checkpoint 포함

별도의 합계를 노출하지 않고 checkpoint를 `len()`, `bytes()`, stats의 entries와 bytes에 더한다. 그러면 growth guardrail과 게이지를 포함한 모든 footprint 소비자가 하나의 값을 보게 된다. `state_checkpoint_growth_bytes`는 `snapshot_growth_bytes`와 같은 방식이며, checkpoint 예산이 가득 차면 증가량 0을 보고한다.

### 3.2 변경된 conversation으로 sweep 한정

donor는 sessionless bucket과 session 안에서 찾으므로, 변경은 해당 bucket과 session의 checkpoint만 orphan으로 만들 수 있다. `drop_orphan_state_checkpoints`는 둘을 받아 해당 checkpoint만 확인한다.

### 3.3 `remove_snapshot`에서 sweep, supersede 도중에는 제외

`remove_snapshot`은 LRU, TTL, cap 제거를 포함해 범위 제한 sweep을 실행한다. `insert_snapshot`의 supersede 및 같은 키 교체 제거는 `remove_snapshot_unswept`를 쓴다. 후속 snapshot이 저장되기 전에 sweep하면 checkpoint가 먼저 사라지기 때문이다. 새 항목이 자리 잡은 뒤 sweep을 한 번 실행한다.

### 3.4 rebuild 실패 시 checkpoint 없이 재시도

restore에 실패하면 스케줄러가 `discard_state_checkpoint`를 호출하고 `lookup_snapshot_outcome_without_checkpoints`로 다시 조회해 더 짧은 최선의 whole snapshot을 채택한다. `CheckpointHit`은 더 이상 조회에서 세지 않는다. 스케줄러가 rebuild 성공 후 `record_state_checkpoint_hit`을 호출하므로, rebuild 실패와 재시도는 한 번의 조회로 집계된다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
insert_snapshot:
  supersede / replace -> remove_snapshot_unswept
  새 항목 저장
  범위 제한 sweep 1회 (bucket, session)

remove_snapshot (LRU, TTL, cap):
  항목 제거 -> 범위 제한 sweep (bucket, session)

scheduler, checkpoint 히트:
  adopt_state_checkpoint 성공 -> record_state_checkpoint_hit
  adopt_state_checkpoint 실패 -> discard_state_checkpoint
                              -> lookup_snapshot_outcome_without_checkpoints -> whole snapshot 채택
```

### 4.2 주요 코드 변경

**스토어** (`prompt_cache/store.rs`, `state_checkpoints.rs`): footprint 집계, `state_checkpoint_growth_bytes`, 범위 제한 `drop_orphan_state_checkpoints`, `remove_snapshot`과 `remove_snapshot_unswept`, `discard_state_checkpoint`, `record_state_checkpoint_hit`, `lookup_snapshot_outcome_without_checkpoints`.

**스케줄러** (`scheduler/prompt_cache.rs`): `insert_state_checkpoint` 전 growth 검사, 후 게이지 갱신, rebuild 실패 재시도.

**문서** (`docs/CONTINUOUS_BATCHING.md`): 새 sweep, 재시도, 집계 동작.

### 4.3 데이터 모델 변경

없음. 저장되는 상태도, 새 설정도 없다.

---

## 5. 학습 포인트

### 5.1 파생 항목도 원본과 같은 방식으로 집계해야 한다

메모리를 쓰는 항목은 메모리 증가를 제어하는 합계에 들어가야 한다. `bytes()`를 건너뛴 보조 맵은 guardrail도 건너뛰었다.

### 5.2 의존 관계가 바뀌는 지점에서 sweep한다

orphan은 donor가 사라질 때 생기므로 sweep은 donor 제거 경로에 있어야 한다. supersede는 제거 뒤 후속 항목 저장이므로, 중간이 아니라 후속 항목 저장 후에 sweep해야 한다.

### 5.3 이벤트는 성공했을 때 센다

조회에서 히트를 세면 rebuild 실패도 히트로 보인다. rebuild 후에 세면 카운터가 실제로 요청을 처리한 경로와 일치한다.

---

## 6. 추가 학습 리소스

### 핵심 키워드
| 키워드 | 설명 | 관련성 |
|--------|------|--------|
| Checkpoint | 히스토리 경계의 recurrent 상태 | 이 PR이 집계하는 항목 |
| Donor | conversation의 최신 whole snapshot | 제거되면 checkpoint가 orphan이 됨 |
| Orphan | 조건에 맞는 donor가 없는 checkpoint | 범위 제한 sweep이 제거 |
| Supersede | 세션 체인에서 더 긴 snapshot이 이전 것을 대체 | checkpoint를 orphan으로 만들면 안 됨 |

### 관련 PR/이슈
- #3058: 이 PR의 이슈 (이 PR로 해결)
- #3055, #3047: checkpoint 기능
- #2883: snapshot growth guardrail, 이제 checkpoint에도 적용

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|------|-----|
| 변경된 파일 | 6 |
| 추가된 줄 | +496 |
| 삭제된 줄 | -63 |
| 새 테스트 | 6 (prompt-cache 및 스케줄러 테스트 파일 5개, growth 테스트 1개) |

### 카테고리별 변경

| 카테고리 | 요약 |
|----------|------|
| 버그 수정 | footprint 집계, growth guardrail, 범위 제한 orphan sweep, donor와 함께 orphan 제거, rebuild 실패 재시도 |
| 테스트 | 집계, sweep 범위, donor 제거, 히트 집계와 폐기, rebuild 실패, growth gate |
| 문서 | `CONTINUOUS_BATCHING.md` |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|--------|
| `255ad02` | fix | count state checkpoints and bound the orphan sweep |

---

## 8. 후속 조치

### 완료 필요
없음. 독립 검토에서 CRITICAL, HIGH, MEDIUM 이슈는 발견되지 않았다.

### 모니터링 필요
- `/v1/cache/stats`의 `entries`와 `bytes`가 checkpoint를 포함하므로 배포 후 두 값이 한 단계 오른다.
- 긴 에이전트 conversation에서 `state_checkpoint_hits`와 `state_checkpoint_evictions`.
- 부록 A의 SIGSEGV 재발 여부.

### 향후 개선 사항
- (LOW) 관측 카운터가 rebuild 실패와 재시도를 두 개의 이벤트로 본다.
- (LOW) `discard_state_checkpoint`가 게이지를 직접 갱신하지 않는다.
- (LOW) `state_checkpoint_growth_bytes`가 reserve 근처에서 증가량을 과대 추정한다 (안전한 쪽).
- (LOW) `donor_for`가 checkpoint마다 여전히 모든 snapshot을 스캔한다.
- (LOW) 범위 제한 sweep 테스트가 테스트 전용 recorder에 의존한다.

---

## 부록

### A. 테스트 결과

- `cargo test --lib server::prompt_cache::`: 215 통과.
- `cargo test --lib server::batch::`: 612 통과, 10 무시.
- `state_checkpoint` 테스트: 21 통과.
- 변이 검사: 재시도와 `remove_snapshot` sweep을 비활성화하자 새 테스트 중 3개가 실패했다.
- `server::batch::` 실행 한 번이 모델이 소유한 cache 관련 무관한 테스트에서 SIGSEGV로 중단되었다. 이후 두 번의 실행에서는 재현되지 않았다. 원인은 조사하지 않았으며 모니터링 항목에 포함했다.
- 이 리포트를 작성하는 동안 테스트는 실행하지 않았다.

### B. 참고 자료
- 이슈 #3058
- `TECHNICAL_REPORTS/0.9/3055-turn-boundary-state-checkpoints-20261115.ko.md`
- `docs/CONTINUOUS_BATCHING.md`
