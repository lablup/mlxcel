# PR #2493: Qwen3-Next 다중 토큰 캐시 포워드 수정

**작성일**: 2026-10-20
**상태**: 완료
**위험도**: Medium
**언어**: Rust

---

## 요약

#2463 종결. `Qwen3NextModel::forward_for_sequence`가 시퀀스 id 없는 모든 다중 토큰 포워드에서 model-owned GatedDeltaNet과 KV 캐시를 리셋해, 캐시가 채워진 상태의 T=5 포워드 한 번은 빈 상태에서 실행되고 T=1 포워드 다섯 번은 상태를 이어가는 비대칭이 있었다. 리셋을 제거하고 #2295가 추가한 `supports_chunked_prefill()==false` 오버라이드를 삭제했다. Qwen3-Next의 청크 프리필이 이제 청크 간에 상태를 이어가는 per-sequence 방식으로 엔드투엔드 동작한다.

---

## 1. 문제 정의

### 1.1 배경

#2295는 `forward_for_sequence`에 no-ID 다중 토큰 리셋(#601에서 시작, 프롬프트 간 인스턴스 재사용을 보호)과 Qwen3-Next용 `supports_chunked_prefill()==false` 옵트아웃을 추가했다. 리셋은 `seq_id.is_none() && seq_len > 1`에서 발동하는데, 이는 스페큘러티브 버스트와 `LanguageModel::forward`를 통한 청크 프리필의 정확한 형태다.

### 1.2 기존 문제점

- **증상**: width-5 실제 체크포인트 trace가 ppl 987.4 (거의 무작위)를 기록한 반면 스텝와이즈 오라클은 high-teens를 기록했다. width 1과 256은 리셋이 T>1 전용이라 영향받지 않았다.
- **잠재 오류**: 캐시 뒤의 다중 토큰 포워드 한 번이 빈 상태에서 시작해, 첫 라운드의 프리필과 verify 연속이 둘 다 틀렸다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 폴백 슬롯을 쓰는 model-owned 패밀리의 T>1 생성 오염 | High | High (Qwen3-Next에서 결정론적) |
| Qwen3-Next 긴 프롬프트에서 청크 프리필 불가 | Medium | High (#2295가 옵트아웃) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

클린. 제거된 리셋은 `seq_id.is_none()` 게이트라 서버 경로에서 실행된 적이 없다. 모든 서버 진입점은 실제 SequenceId를 전달하고, 크로스 리퀘스트 격리는 구조적(`Engine::open`이 per-sequence 상태를 삽입, `Engine::close`가 해제, `with_or_create_sequence_state(Some(id), ...)`는 공유 internal 슬롯으로 폴백하지 않음). 요청 간 상태 유출 없음.

### 2.2 성능 관점

- 첫 full-attention 레이어 자체 오프셋으로 만든 단일 오프셋 프리필 마스크는 모든 FA 레이어에 유효하다. model-owned 레이아웃은 dense pool 캐시가 없어 FA 오프셋이 갈라질 수 없다. GDN 레이어는 마스크를 무시한다.
- GDN 상태는 고정 크기(conv_state, state_cache)로 시퀀스 길이에 따른 증가가 없고, per-sequence 슬롯은 첫 청크에서 한 번 생성 후 포워드마다 이동하므로 청크당 상태 재할당이 없다.
- 청크 프리필은 `[1, prompt_len, vocab]` 로지트 트랜지언트를 `[1, chunk_len, vocab]`로 바꿔, 총 lm-head 작업이 같으면 트랜지언트 메모리가 같거나 더 낫다.

### 2.3 호환성/의존성 관점

- **호환성 깨짐**: 없음. classic speculative walk의 Qwen3-Next 관찰 동작은 한 세대 안에서 개선되고, `reset_runtime_state` 시맨틱은 보존되며 이제 테스트로 고정된다.
- **새 의존성**: 없음.

### 2.4 코드 품질 관점

- 테스트: 재작성된 `qwen3_next_loaded_model_chunking_contract_is_safe`가 세 계약 다리를 모두 고정한다 (supports_chunked_prefill true; no-ID 연속이 2+3 후 선형 오프셋 5 (main에서는 실패); reset_runtime_state -> 0; 안정적 seq-id 유지). 새 8레이어 테스트 두 개가 `LanguageModel::forward` (오프라인 CLI 진입)와 명시적 마스크의 내재 포워드로 로지트 동치를 고정한다.
- 픽스처는 두 레이어 종류 (interval 4: 레이어 3, 7이 full attention), dense MLP, tied embeddings, 확립된 byte-decode readback 패턴을 다룬다.

---

## 3. 기술적 선택과 그 이유

### 3.1 리셋을 좁히는 대신 제거

**컨텍스트:** #601이 프롬프트 간 모델 재사용하는 캐시 없는 no-ID 호출자를 보호하려고 리셋을 추가했고, #2295가 그 때문에 Qwen3-Next를 청크 프리필에서 옵트아웃했다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|------|------|------|
| 리셋 유지, prompt-lookup 지원으로 게이트 | 더 작은 diff | 모든 다중 토큰 호출자에 대해 폴백 슬롯이 고장인 채로 유지됨; 저장소의 플로우는 폴백을 프롬프트 간 재사용하지 않음 |
| **선택: 리셋 제거, 옵트아웃 삭제** | 첫 라운드 프리필과 verify 연속 복원; 청크 프리필 엔드투엔드 활성화; qwen3_5 패밀리 패턴과 일치 | stale-state 보호를 per-sequence 라이프사이클이 대신 제공해야 함 |

**선택 이유:** 저장소의 실제 플로우에서 폴백 다중 토큰 경로의 프로덕션 호출자는 deprecated classic speculative walk 하나(프로세스당 한 세대)다. per-sequence 격리는 구조적이라 리셋의 보호는 중요한 곳에서는 죽은 코드였고 Qwen3-Next에서는 정확성 버그였다.

---

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/models/qwen3_next.rs`**
- `forward_for_sequence`의 no-ID 다중 토큰 리셋 블록 제거.
- `supports_chunked_prefill()==false` 오버라이드 삭제 (트레이트 기본값 true 적용).

**파일: `src/models/qwen3_next_tests.rs`**
- 청킹 계약 테스트 재작성; 수락 테스트 `qwen3_next_multi_token_forward_behind_cache_matches_stepwise`와 명시적 마스크의 내재 포워드를 통한 state-update 쌍둥이 추가.

---

## 5. 검증

| 검사 | 결과 |
|---|---|
| 수락 테스트 (main) | FAILED (divergence 0.025385 > 0.002) |
| 수락 테스트 (브랜치) | PASS |
| 실제 체크포인트 qwen3-next-80b-a3b-instruct-4bit, width 5, docs/architecture.md, args 5 60 8 512, MLX_ENABLE_TF32=0 | 300행, 평균 NLL 2.9325, ppl 18.77 (되돌린 arm 987.4) |
| width 1, 256 (양쪽 arm) | byte-identical |
| 스코프드: qwen3_next / gated_delta | 16/16, 11/11 |
| 풀 게이트 (2ff66014) | 161 테스트 바이너리 build 0, clippy 0, doc 0, contract 0 |
| 코멘트 수정 커밋 008b8ad8 | fmt 클린, clippy 클린, TF32=0으로 qwen3_next 16/16 |
| 보안 검토 | 클린; 요청 간 유출 없음; 성능 회귀 없음 |

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 2 (코멘트 수정도 같은 2파일) |
| 추가 | +251 (메인 커밋), +7 (코멘트 수정) |
| 삭제 | -23 (메인 커밋), -7 (코멘트 수정) |

### 관련 커밋

| 해시 | 타입 | 메시지 |
|------|------|--------|
| `2ff66014` | fix | fix(models): continue Qwen3-Next cache across multi-token forwards |
| `008b8ad8` | docs | docs(models): fix orphaned qwen3_next comment and in_proj_qkvz formula |

---

## 7. 후속 조치

### 완료 필요

- [ ] 이슈 #2495 (등록됨): deprecated classic speculative walk가 rejected round 후 model-owned 패밀리를 되감지 못함 (기존 결함, 이 PR 범위 밖).
- [ ] 기록만: `src/server/batch/scheduler/prefill.rs:755-761`, `planned_prefill.rs:295-298`의 기존 no-op `make_caches` 코멘트는 model-owned 패밀리에서 아무것도 리셋하지 않음.

### 모니터링 필요

- 머지 후 첫 릴리스에서 서버 planned prefill의 Qwen3-Next 청크 프리필 동작.

---

## 부록

### A. 리뷰 판정

pr-reviewer: PASS, CRITICAL/HIGH 0 (MEDIUM/LOW 3건, 모두 회귀 아님; MEDIUM은 #2495로 분리). pr-security-checker: 보안과 성능 모두 클린. pr-finalizer가 두 코멘트 수정을 `008b8ad8`으로 반영.
