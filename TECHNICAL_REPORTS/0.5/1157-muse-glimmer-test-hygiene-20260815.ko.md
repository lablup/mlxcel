# 기술 보고서: PR #1157 - test: skip pinned muse_glimmer checkpoint tests and fix MLXCEL_BACKEND race

**작성일**: 2026-08-15
**작성자**: AI Code Reviewer
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

#1116에서 도입된 muse_glimmer 테스트 3건이 pinned 체크포인트(`models/mlx/muse-glimmer-30b`)가 없는 머신에서 전체 `cargo test --release --features metal,accelerate --lib` 실행을 깨뜨렸다. 2건은 인덱스 파일 부재로 panic했고, 1건은 형제 테스트가 `MLXCEL_BACKEND`를 프로세스 전역으로 변경해 간헐적으로 실패했다. 이 PR은 체크포인트 의존 테스트 2건을 우아하게 skip하도록 바꾸고, 경합하는 테스트를 crate 전역 `test_support::env_lock`으로 직렬화해 체크포인트 없는 머신에서 스위트를 green(5621 passed / 0 failed)으로 복원한다. 테스트 전용 변경이며 프로덕션 코드는 건드리지 않는다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1155는 에픽 #1148 통합 검증 중 발견되어 발행됐다. main `a206f089`과 에픽 이전 기준선 양팔 실측에서 세 실패가 동일하게 재현되고 에픽 커밋 어느 것도 muse 파일을 건드리지 않으므로, 에픽 회귀가 아니라 #1116의 선재 결함이다.

### 1.2 기존 문제점

- **문제 1**: `loading::vlm::muse_glimmer::tests::pinned_weight_index_classifies_each_source_weight_once`가 `models/mlx/muse-glimmer-30b/model.safetensors.index.json` 부재 시 "Failed to read ... No such file or directory"로 panic.
- **문제 2**: `vision::encoders::muse_glimmer_fusion::tests::pinned_post_tower_weight_roots_and_shapes_match_published_contract`가 같은 인덱스 부재에 assert로 실패.
- **문제 3**: `server::startup::muse_glimmer_startup_guard_tests::muse_glimmer_startup_allows_baseline_and_keeps_video_disabled`가 단독 실행은 통과하나 기본 병렬 실행에서 실패. 형제 테스트 `muse_glimmer_startup_rejects_xla_backend_selection`이 `MLXCEL_BACKEND=xla`를 프로세스 전역으로 설정(set_var/restore)하는데, 기준선 테스트의 validator가 그 env를 읽어 과도기 값을 관측할 수 있었다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 기여자 머신에서 붉은 스위트가 실제 회귀를 가림 | Medium | High (30B 체크포인트 없는 모든 머신) |
| 순서 의존 플레이크가 스위트 신뢰를 잠식 | Medium | Medium (스케줄러 의존) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

보안 표면 없음. 변경된 3개 파일 모두 `#[cfg(test)]` 테스트 모듈이며 프로덕션 바이너리에 포함되지 않는다. 보안 검토에서 인덱스 경로 TOCTOU는 무해(하드코딩된 리터럴 경로, 공격자 입력 없음, 최악의 경우가 PR 이전의 panic)로 확인됐고, 기존 모듈별 MLX 가드와의 잠금 순서 사이클도 없다.

**발견된 이슈:**
| 이슈 | 심각도 | 상태 |
|-----|-------|-----|
| opt-in `xla-backend`/`experimental-backend` 피처 하에서 `backend::tests`의 미잠금 `MLXCEL_BACKEND` 리더 | Medium | Open (#1116 선재, 기본·CI 피처셋에서는 도달 불가, PR 스코프 유지를 위해 후속으로 남김) |
| fusion 테스트 skip 가드가 인덱스만 검사해 부분 다운로드된 체크포인트는 `config.json`/shard `unwrap()`에서 여전히 panic | Low | Open (저장소 전반의 skip 가드 관례, 후속 후보) |

### 2.2 성능 관점

추가된 뮤텍스 획득은 한 모듈의 테스트 2건만 직렬화하며, 스위트 전체 시간에 미치는 영향은 측정 불가 수준(모듈 실행이 1초 미만).

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음
- **새로운 의존성**: 없음 (기존 crate 내부 `test_support::env_lock` 재사용)
- **호환성**: pinned 체크포인트가 있는 머신에서는 동작 불변, 두 contract 테스트 모두 전체 어서션을 그대로 수행

### 2.4 코드 품질 관점

- **테스트 커버리지**: 개수 불변. 체크포인트 없는 머신에서 스위트를 끝까지 돌릴 수 있게 되어 실효 커버리지는 개선
- **코드 복잡도**: early-return 가드 2개와 잠금 획득 1줄로 리뷰가 쉬움
- **기술 부채**: 감소 (순서 의존 제거, `tests/prompt_cache_e2e.rs`의 기존 skip 관례와 정렬)

---

## 3. 기술적 선택과 그 이유

### 3.1 `#[ignore]` 대신 우아한 skip(`eprintln` + early return)

**컨텍스트:** 체크포인트 의존 contract 테스트 2건은 체크포인트를 보유한 머신에서는 자동으로 계속 실행되어야 하고, 그 외에서는 실패하면 안 된다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| `#[ignore = "..."]` | 테스트 출력에 ignored로 표시됨 | 체크포인트가 있어도 자동 실행되지 않고 `--ignored` 필요 |
| **선택: 가드 + eprintln + return** | 체크포인트가 있으면 전부 실행, 없으면 사유를 출력하고 skip. `tests/prompt_cache_e2e.rs` 관례와 일치 | skip된 실행도 `ok`로 보고되어 커버리지 잠식이 조용함 (후속 아이디어로 opt-in `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` 기록) |

**선택 이유:** 이슈 #1155가 저장소의 기존 skip 관례를 명시적으로 선호했고, 체크포인트 보유 머신은 워크플로 변경 없이 contract 커버리지를 유지한다.

### 3.2 모듈별 뮤텍스·`serial_test`·파라미터 리팩터링 대신 crate 전역 `env_lock`

**컨텍스트:** 경합은 같은 모듈의 mutator 1개와 reader 1개 사이지만, env var는 프로세스 전역 상태다.

**선택 이유:** `crate::test_support::env_lock`(`src/lib.rs:52` 선언)은 crate 전체에 18개 획득 지점을 가진 확립된 패턴이고 poison 복구(`unwrap_or_else(|p| p.into_inner())`)까지 문서화되어 있다. 변경하는 형제 테스트가 이미 이 잠금을 잡고 있었으므로 reader 쪽에 획득 한 줄을 추가하는 것으로 경합이 닫힌다. `serial_test` 의존성 추가나 startup guard 시그니처 리팩터링은 같은 보장에 비해 순전히 더 큰 변경이다. 리뷰 단계에서 차등 검증으로 잠금이 load-bearing함을 증명했다: 잠금 제거 시 모듈 스트레스 실행 25/25 실패, 잠금 유지 시 10/10 통과.

---

## 4. 구현 상세

### 4.2 주요 코드 변경

**파일: `src/vision/encoders/muse_glimmer_fusion_tests.rs`**
```rust
// 변경 전
assert!(
    index_path.exists(),
    "Muse Glimmer pinned checkpoint index is required for this shape contract"
);

// 변경 후
if !index_path.exists() {
    eprintln!(
        "Skipping pinned_post_tower_weight_roots_and_shapes_match_published_contract: \
         pinned Muse Glimmer checkpoint index not present at {}",
        index_path.display()
    );
    return;
}
```

**변경 이유:** 하드 환경 의존을 우아한 skip으로 전환하되, 체크포인트가 있으면 shape contract 어서션을 그대로 유지한다. `src/loading/vlm_muse_glimmer_tests.rs`도 인덱스 읽기 전에 같은 형태의 가드를 받는다.

**파일: `src/server/muse_glimmer_startup_guard_tests.rs`**
```rust
// muse_glimmer_startup_allows_baseline_and_keeps_video_disabled 시작부에 추가
let _env_guard = crate::test_support::env_lock::env_lock();
```

**변경 이유:** `validate_muse_glimmer_unsupported_startup`이 `MLXCEL_BACKEND`를 읽고, 형제 XLA 거부 테스트가 같은 잠금 아래에서 이 값을 변경하므로, 양쪽이 직렬화되어 기준선 테스트가 과도기 `xla` 값을 관측할 수 없게 된다. 가드 선언 순서도 올바르다: `_env_guard`가 `TempDir`보다 먼저 선언되어 나중에 해제된다.

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 3 |
| 추가된 라인 | +25 |
| 삭제된 라인 | -4 |
| 테스트 추가 | 0 (기존 테스트 3건 수리) |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| Code Quality | 3 | 체크포인트 skip 가드 2건, env-lock 획득 1건 |

### 관련 커밋
| Hash | Type | Message |
|------|------|---------|
| `79bc61a4` | test | skip pinned muse_glimmer checkpoint tests and fix MLXCEL_BACKEND race |

---

## 8. 후속 조치

### 완료 필요
- [ ] 없음. 기본 피처셋 기준으로 이슈의 수용 기준을 모두 충족

### 향후 개선 사항
- opt-in `xla-backend`/`experimental-backend` 피처 하에서 `backend::tests::mlx_session_threads_the_token_bias_through`의 미잠금 `select_backend()` env 읽기 가드 (선재, CI 도달 불가)
- fusion 테스트 skip 가드를 부분 다운로드된 체크포인트까지 커버하도록 확장 (현재는 인덱스만 검사 후 `config.json`·shard를 `unwrap()`)
- 체크포인트 보유 머신에서 조용한 skip을 실패로 승격시키는 opt-in `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` 검토

---

## 부록

### A. 테스트 결과
- 체크포인트 없는 머신의 전체 스위트: `cargo test --release --features metal,accelerate --lib` → 5621 passed / 0 failed / 117 ignored
- 타깃 실행: `--lib muse_glimmer` → 80 passed / 0 failed; `--lib muse_glimmer_startup_guard` 반복(구현자 5회, 보안 검토 3회, 리뷰어 10회) → 플레이크 없음
- 경합 수정의 차등 증명: 잠금 제거 → 모듈 실행 25/25 실패, 잠금 유지 → 10/10 통과
- 두 skip 가드의 체크포인트-존재 팔은 로컬 체크포인트가 없어 코드 검토로만 확인했고, PR 본문에 정직하게 명시함

### C. 참고 자료
- 이슈 #1155 (사양), #1116 (해당 테스트 도입), 에픽 #1148 (실패를 드러낸 통합 검증)
- PR #1157의 구현 리뷰·보안 리뷰 코멘트
