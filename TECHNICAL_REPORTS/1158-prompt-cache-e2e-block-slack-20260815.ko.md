# 기술 보고서: PR #1158 - test: derive prompt_cache_e2e slack from the paged block size

**작성일**: 2026-08-15
**작성자**: AI Code Reviewer
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

수동 실행 multi-turn prompt-cache e2e 테스트(`tests/prompt_cache_e2e.rs`, `#[ignore]`, 로컬 qwen3-0.6b-4bit 사용)가 turn 3에서 결정적으로 실패했다. under-allowance 어서션의 slack이 4토큰뿐인데, APC lookup 경로는 캐시 크레딧을 16토큰 블록 단위로 내림(floor)하므로 최대 15토큰의 정당한 손실이 발생한다. 이 PR은 slack을 서버 자체의 `DEFAULT_APC_BLOCK_SIZE` 상수에서 유도하고(slack = block size - 1 = 15), flooring 메커니즘을 정확하게 문서화하며, 서버 spawn 시 `APC_BLOCK_SIZE`/`APC_ENABLED` env 폴백을 제거해 어서션의 전제가 호출 셸과 무관하게 성립하도록 한다. 테스트 전용 변경이며, 실물 fixture로 6회 실측 모두 turn별 값이 완전히 동일하다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1156은 에픽 #1148 통합 검증에서 발행됐다. 양팔 실측으로 귀속이 확정됐다: 에픽 이전 기준선 `9c154ff3`이 토큰 단위까지 동일한 값(5턴에 걸쳐 cached 0/48/64/96/112)으로 실패하므로, 에픽 회귀가 아니라 테스트의 선재 결함이다.

### 1.2 기존 문제점

- **문제 1**: turn 3에서 `cached_tokens`가 64인데 이전 프롬프트 길이가 73이고, 어서션 `cached + 4 >= prev_prompt_len`이 실패. 4토큰 slack은 블록 flooring을 감당하지 못한다.
- **문제 2**: 메커니즘. APC lookup은 검증된 전체 블록만 인정하고(`apc_consistent_prefix_len`이 `consistent_blocks * block_size` 반환, `src/server/prompt_cache/apc_lookup.rs`), dense adopt 경로는 정확히 그 값으로 잘라낸다. `max_tokens=16`이면 qwen3 thinking 모델의 assistant content가 비어(16토큰 전부 미종결 think 블록으로 소진), 히스토리 재렌더가 이전 프롬프트 경계 바로 그 지점에서 갈라져 flooring 손실이 그대로 노출된다: 최악 `block_size - 1 = 15`토큰으로 4를 초과.
- **문제 3** (리뷰 중 발견): spawn된 서버가 부모 환경을 상속하는데 `mlxcel-server`는 플래그 부재 시 `APC_BLOCK_SIZE`/`APC_ENABLED` env var로 폴백한다. 셸에 export된 `APC_BLOCK_SIZE`가 있으면 어서션은 컴파일 타임 상수를 쓰는데 서버 블록 크기는 조용히 달라진다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 결정적 거짓 실패가 항상 붉은 테스트 뒤로 실제 캐시 회귀를 숨김 | Medium | High (매 실행 turn 3 실패) |
| 하드코딩 slack이 블록 크기 상수 변경 시 조용히 어긋남 | Low | Low |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

보안 표면 없음(테스트 전용). 보안 검토에서 15가 tight bound임을 검증했고, 산술은 오버플로 안전(const 평가, `cached + 15` wrap은 `cached`가 `u64::MAX` 근방일 때뿐이며 그 경우 어서션은 어차피 통과)하며, env var 상속 발견(MEDIUM)은 오케스트레이터가 `spawn_server`의 `.env_remove`로 수정했다(선례: `tests/cli_help_consistency.rs`).

**발견된 이슈:**
| 이슈 | 심각도 | 상태 |
|-----|-------|-----|
| 거짓 정당화 주석: crate 상수를 import할 수 없다고 주장(루트 패키지에 "lib 타깃 없음") — 실제로는 있고, 형제 테스트가 이미 같은 경로를 import | High | Fixed (`203d1118`) |
| 주석의 메커니즘 오귀속: "dense-trie flooring"/"전체 블록 donation" — donation은 정확한 토큰을 저장하고 trie는 정확한 LCP를 반환하며, floor는 APC lookup에 있다 | Medium | Fixed (`203d1118`) |
| 상속된 `APC_BLOCK_SIZE`/`APC_ENABLED`가 어서션 전제를 양방향으로 깨뜨림(32 → 거짓 실패, 8 → 과잉 허용) | Medium | Fixed (`45d8757c`) |
| `apc_lookup.rs:46`의 선재 doc 결함("input matched_len은 보존된다" — 거짓, 반환은 항상 블록 내림) | Low | Open (프로덕션 doc, 스코프 밖, PR에 기록) |

### 2.2 성능 관점

없음. 어서션 산술만 변경. prefill-latency 어서션은 매 실측에서 통과(비율 0.37~0.50, 상한 1.3).

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음
- **새로운 의존성**: 없음. lib import는 빌드 표면을 추가하지 않는다(`tests/` 아래 18개 파일이 이미 `mlxcel` lib을 import)
- **호환성**: Dense 디코드 백엔드(`--batch-size 1`) 전제에서만 유효. paged 백엔드는 더 굵은 32토큰 floor(`DEFAULT_PAGED_BLOCK_SIZE`)를 적용하며, 상수 주석에 문서화됨

### 2.4 코드 품질 관점

- **테스트 커버리지**: 항상 붉던 수동 테스트가 사용 가능한 회귀 게이트로 복원됨
- **코드 복잡도**: 상수 import 1건, 어서션 변경 1건, env 제거 2건
- **기술 부채**: 감소 (매직 넘버를 source-of-truth 상수로 대체, env 의존 전제 봉쇄)

---

## 3. 기술적 선택과 그 이유

### 3.1 리터럴 확대나 fixture 변경 대신 상수에서 slack 유도

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| 리터럴을 15로 확대 | 최소 diff | 블록 크기 상수가 바뀌면 조용히 어긋남 |
| fixture 변경(non-thinking 모델 또는 더 큰 `max_tokens`) | empty-content 분기 회피 | 메커니즘을 수용하는 대신 숨김. 비정렬 경계에서 갈라지는 어떤 경우에도 flooring 손실은 여전히 존재 |
| **선택: `DEFAULT_APC_BLOCK_SIZE` import, slack = block size - 1** | 어서션이 source of truth를 추적. 15가 tight bound임을 증명 가능 | crate 상수의 (실제로 성립하는) import 가능성 필요 |

**선택 이유:** 수용 기준이 상수 유도를 선호했다. 리뷰 단계에서 15가 정확히 tight함을 증명했다: 블록 하나를 잃는 회귀는 `cached <= P - 16`이므로 `cached + 15 <= P - 1`이 되어 어서션이 여전히 발화하며, 전체 블록 하나가 dense APC 경로가 표현할 수 있는 최소 손실이다. 관측된 모든 값이 `floor(P/16)*16`과 정확히 일치(50→48, 73→64, 96→96, 119→112)해 flooring 모델이 실증적으로 확인됐다.

### 3.2 각 호출 지점에 `--apc-block-size`를 넘기는 대신 `spawn_server`에서 env 폴백 제거

**선택 이유:** `.env_remove("APC_BLOCK_SIZE").env_remove("APC_ENABLED")`는 파일 내 현재·미래의 모든 spawn을 한 지점에서 고치며, `tests/cli_help_consistency.rs`의 기존 선례와 일치한다. 플래그 전달 방식은 호출 지점마다 반복해야 하고 `APC_ENABLED`는 여전히 상속된다.

---

## 4. 구현 상세

### 4.2 주요 코드 변경

**파일: `tests/prompt_cache_e2e.rs`**
```rust
// 변경 전
assert!(
    cached + 4 >= prev_prompt_len,
    ...
);

// 변경 후
use mlxcel::server::prompt_cache::DEFAULT_APC_BLOCK_SIZE;
const APC_BLOCK_SIZE: u64 = DEFAULT_APC_BLOCK_SIZE as u64;
...
assert!(
    cached + (APC_BLOCK_SIZE - 1) >= prev_prompt_len,
    ...
);
```

**변경 이유:** APC lookup은 검증된 전체 블록만 인정하므로 이전 프롬프트 경계에서의 재렌더 분기는 최대 `block_size - 1`토큰을 잃을 수 있다. 어서션은 정확히 그만큼만 허용해야 한다. doc 주석에 두 전제조건을 기록했다: `--batch-size 1`을 통한 Dense 백엔드(paged는 32에서 floor), 그리고 `spawn_server`의 env 스크럽.

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 1 |
| 추가된 라인 | +44 |
| 삭제된 라인 | -2 |
| 테스트 추가 | 0 (기존 수동 테스트 1건 수리) |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| Code Quality | 3 | 상수 유도 slack, 정확한 메커니즘 문서화, env 폴백 스크럽 |

### 관련 커밋
| Hash | Type | Message |
|------|------|---------|
| `c3b33cd8` | test | derive prompt_cache_e2e slack from the paged block size |
| `203d1118` | test | import the APC block size instead of mirroring it |
| `45d8757c` | test | scrub APC env fallbacks when spawning the e2e server |

---

## 8. 후속 조치

### 완료 필요
- [ ] 없음. 세 수용 기준 모두 실물 모델 증거로 충족

### 향후 개선 사항
- `apc_lookup.rs:46` doc 주석 수정("input matched_len은 보존된다"는 거짓, 반환은 항상 `consistent_blocks * block_size`) — PR에 기록된 선재 프로덕션 doc 결함
- 테스트가 `--batch-size`를 올리게 되면 slack을 `DEFAULT_PAGED_BLOCK_SIZE - 1`로 넓혀야 함(상수 주석에 문서화됨)

---

## 부록

### A. 테스트 결과

로컬 qwen3-0.6b-4bit fixture로 `cargo test --release --features metal,accelerate --test prompt_cache_e2e multi_turn -- --ignored` 6회 실측(구현자 2회, 리뷰어 2회, 보안 검토 기준 증거 1회, finalizer의 env 스크럽 이후 1회), 전부 turn별 값 동일:

| Turn | prompt_tokens | cached_tokens | flooring 검산 |
|------|---------------|---------------|----------------|
| 1 | 50 | 0 | cold start |
| 2 | 73 | 48 | floor(50/16)*16 = 48 |
| 3 | 96 | 64 | floor(73/16)*16 = 64 (기존 실패 지점: 64 + 15 >= 73) |
| 4 | 119 | 96 | floor(96/16)*16 = 96 |
| 5 | 142 | 112 | floor(119/16)*16 = 112 |

턴별 slack 소비량: 허용 15 중 2/9/0/7. 엄격 어서션(turn 2부터 cached > 0, 단조 증가)은 불변·통과.

### C. 참고 자료
- 이슈 #1156 (사양), 에픽 #1148 (통합 검증, 요약 코멘트에 전체 턴 테이블)
- `src/server/prompt_cache/apc_lookup.rs` (flooring 지점), `src/server/prompt_cache/block_hash.rs` (`DEFAULT_APC_BLOCK_SIZE`), `src/server/batch/scheduler.rs` (dense adopt 절단, `DEFAULT_PAGED_BLOCK_SIZE`)
- PR #1158의 리뷰·보안 코멘트
