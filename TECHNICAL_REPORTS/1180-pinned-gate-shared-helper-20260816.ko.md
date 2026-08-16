# 기술 보고서: PR #1180 - fix(test): gate the loading-side pinned Muse Glimmer guard on MLXCEL_REQUIRE_PINNED_CHECKPOINTS

**작성일**: 2026-08-16
**작성자**: AI Code Reviewer
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Low

---

## 요약

PR #1173은 `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1`을 도입했다. 고정 체크포인트 테스트의 우아한 skip을 하드 실패로 바꿔, 손상된 체크포인트가 그것을 보유한 장비에서 유일하게 존재하는 커버리지를 조용히 무력화하지 못하게 하는 장치다. 그런데 이 게이트가 고정 Muse Glimmer 가드 둘 중 하나에만 걸려 있었다. 이 PR은 나머지 하나에도 적용하고, 게이트를 공유 test-support 헬퍼로 옮겨 두 호출부가 한 구현을 쓰게 하며, 변수를 문서화한다.

여기서 문서화는 부수적인 일이 아니라 절반의 가치를 차지한다. 이 게이트는 트리 어디에도 설정돼 있지 않았다. `.github/` 워크플로에도, `scripts/` 아래에도, 셸 프로파일에도 없었다. 아무도 켜지 않는 opt-in 노브는 아무것도 지키지 못하고, 저장소 변경으로 남의 셸에 export를 넣을 수도 없다. 그래서 구현 도중 이슈에 네 번째 산출물이 추가됐다. 무엇을 하는 변수인지가 아니라, 언제 어떻게 실제로 켜야 하는지를 문서화하는 것이다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1177은 PR #1173 기술 보고서의 후속 항목에서 발행됐다. 지적하는 빈틈은 좁지만 실재한다. 로딩 쪽 가드가 무조건 skip하므로, 체크포인트 보유 장비에서 아무도 모르게 영구히 조용해질 수 있다.

### 1.2 기존 문제점

- **문제 1**: `pinned_weight_index_classifies_each_source_weight_once`(`src/loading/vlm_muse_glimmer_tests.rs`)가 인덱스 부재 시 원래의 무조건 skip을 썼다. 이 테스트는 1436개 가중치의 `MuseWeightInventory` 내역이라는 실제 계약을 단언하므로, 조용한 skip은 잡을 가치가 있는 것을 가린다.
- **문제 2**: 게이트가 vision 테스트 모듈 내부의 private 함수여서, 확장하려면 환경변수 읽기와 락 규율을 복제하거나 추출해야 했다.
- **문제 3**: 변수가 문서화되지도, 어디서 켜지지도 않아 메커니즘 전체가 실질적으로 무력했다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| 체크포인트 보유 장비에서 로딩 쪽 계약 검사가 조용히 무력화 | Low | Low |
| 게이트 사본 두 개가 서로 어긋남 | Low | Medium (복제로 확장할 경우) |
| 게이트가 아무도 켜지 않는 노브로 남아 위 항목들이 실제로는 예방되지 않음 | Medium | 문서화가 없으면 High |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

테스트 전용이자 문서 전용 변경이다. `test_support`는 크레이트 루트에서 `#[cfg(test)] pub(crate)`로 선언되고(`src/lib.rs:51-52`), 새 모듈은 그 안의 `pub(crate)`이며 재수출이 없다. 트리 전체에서 참조는 두 테스트 호출부뿐이라 프로덕션 코드, 바이너리, `tests/` 어디에서도 닿을 수 없다.

보안 검토가 환경변수 처리를 상세히 확인했다. 가드는 `let required = { let _env_guard = env_lock(); ... };`로 묶여 문장 끝이 아니라 블록 끝에서 해제되므로 어서션은 락 없이 실행된다. 교착은 불가능하다. `env_lock`은 비재진입 `std::sync::Mutex`이고, 호출부 어느 쪽도 락을 쥔 채 부르지 않으며, 크레이트 내 다른 보유자가 이 헬퍼에 닿지 못한다. poisoning은 조기 해제와 `env_lock()` 자체의 `unwrap_or_else(|p| p.into_inner())`로 이중 방어된다. 크레이트 내 환경변수 변경 지점을 모두 락 사용과 대조했고, 락 없이 변경하는 세 곳은 각각 무해함을 확인했다.

문서의 영구 export 권고도 액면 그대로 받아들이지 않고 안전성을 검증했다. 테스트 파일 밖에서 이 변수를 읽는 코드가 없고, `std::env::vars()`를 열거하는 코드도 없으며, 미지의 `MLXCEL_*`를 거부하는 로직도 없다. 따라서 export해도 구동 중인 `mlxcel`이나 `mlxcel-server`에 영향을 주지 못한다.

**발견된 이슈:**
| 이슈 | 심각도 | 상태 |
|------|--------|------|
| PR 본문이 문서 재구성으로 낡아진 표 형태를 서술, 이 본문이 squash 커밋 메시지가 됨 | Low | 수정 (리뷰 단계) |
| PR 본문이 vision 쪽 실패 메시지가 그대로라고 주장, 실제로는 테스트 이름이 추가됨 | Low | 수정 (리뷰 단계) |
| 문서가 #1173 이전에 availability pre-check가 없었다고 서술 | Low | 수정 (`d2b4f419`, 이후 `758fe64c`로 재수정) |

### 2.2 성능 관점

없다. 락은 skip이나 실패 경로에서만 잡히고, `std::env::var` 호출 한 번 동안만 유지되며, 어서션과 파일 I/O 이전에 해제된다. `assert_post_tower_contract`의 shard 헤더 읽기는 전부 락 바깥에서 일어난다. 체크포인트가 온전한 장비에서는 헬퍼가 아예 호출되지 않는다.

### 2.3 호환성/의존성 관점

- **호환성 파괴**: 없음
- **신규 의존성**: 없음
- **호환성**: 프로덕션 경로에 영향 없음

### 2.4 코드 품질 관점

- **테스트 커버리지**: 개수는 그대로. 로딩 쪽 가드가 게이트를 얻었고, 두 가드 모두 skip과 실패 메시지에서 자기 이름을 계속 밝힌다
- **코드 복잡도**: private 사본 하나가 공유 헬퍼 하나로 바뀌고 테스트 이름이 인자로 전달된다
- **기술 부채**: 감소. PR #1173 보고서의 후속 항목이 닫혔고 게이트 사본이 늘어나는 것을 막았다

---

## 3. 기술적 선택과 그 이유

### 3.1 `src/test_support/`로 추출하고 테스트 이름을 인자로 전달

**선택 이유:** `test_support`는 어느 깊이의 테스트 모듈에서도 이름으로 부를 수 있도록 크레이트 루트에 선언돼 있고, `src/vision/encoders/`와 `src/loading/` 사이에서 공유할 헬퍼에는 바로 그 성질이 필요하다. 원래 함수는 skip 메시지에 vision 테스트 이름을 하드코딩했으므로, 공유 버전은 `test_name: &str`을 받아 각 호출부가 자기를 밝히게 했다. 어느 테스트가 skip됐는지 말하지 않는 메시지는 퇴보이고, 호출자가 둘이면 적극적으로 혼란을 준다.

### 3.2 로딩 쪽 분류 오류의 `panic!` 유지

**선택 이유:** `pinned_weight_index_classifies_each_source_weight_once`는 `read_muse_weight_inventory_from_index`가 `Err`를 반환하면 패닉한다. 이것은 가용성 신호가 아니라 계약 신호이므로 패닉으로 남긴다. 게이트는 그 위의 `!index_path.exists()` 분기에만 삽입했다. 헬퍼는 언제나 skip을 실패로 바꿀 뿐 그 반대는 하지 않으며, 이 방향성을 모든 호출부에서 검증했다.

### 3.3 표를 새로 만들지 않고 기존 표에 한 행으로 접어 넣음

**고려한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 같은 섹션에 `Variable / Values / Default / Notes` 표를 추가 | 값 의미를 전용 열로 표현 | 한 섹션에 표가 둘이 되고, 형식 차이를 설명하는 문장이 붙는다. 변수가 아니라 문서에 대한 해설이다 |
| **채택: 기존 `Variable / Purpose` 표에 한 행** | 섹션의 확립된 형태와 일치 | 값과 기본값 의미를 셀 안에 서술해야 함 |

**선택 이유:** 섹션 자체의 선례가 답을 정한다. `MLXCEL_ALLOW_PARALLEL_CUDA_TESTS`와 `MLXCEL_ALLOW_CONCURRENT_GPU_TESTS` 행이 이미 이슈 번호까지 담은 여러 문장을 Purpose 셀에 넣고 있다. 긴 셀은 그곳에서 정상이고, 표 형식에 대한 메타 문장은 그렇지 않다.

### 3.4 무엇을 하는지가 아니라 언제 켜야 하는지를 문서화

**선택 이유:** 이 항목은 변수가 어디에도 설정돼 있지 않음을 확인한 뒤 구현 도중 이슈에 추가됐다. 그래서 문서는 이것이 테스트 전용이며 런타임 노브가 아니라는 점, 고정 체크포인트를 보유한 장비에 속하며 그곳에서의 skip이 왜 실제 손실인지, 한 번만 켜는 방법과 영구 export 방법, 그리고 이미 실현된 두 번째 용도를 다룬다. 게이트를 켠 상태에서 고정 테스트가 `ok`를 보고했다면 체크포인트를 읽고 계약을 단언했다는 뜻일 수밖에 없다. 사용 불가 상태였다면 실패했을 것이기 때문이다. 이슈 #1161의 수용 기준 2를 이 방법으로 증명했다.

---

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/test_support/pinned_checkpoint.rs`** (신규)
```rust
pub(crate) fn skip_or_fail_pinned_checkpoint(test_name: &str, reason: &str) {
    // The crate-wide env lock serializes this read against tests that mutate
    // the process environment with `unsafe set_var`; on Rust 2024 an
    // unsynchronized concurrent read of the env block is undefined behavior.
    // Hold the guard only for the read, and drop it before the assertion so
    // a failing assertion here cannot poison the mutex for later tests.
    let required = {
        let _env_guard = crate::test_support::env_lock::env_lock();
        std::env::var("MLXCEL_REQUIRE_PINNED_CHECKPOINTS").is_ok_and(|value| value == "1")
    };
    assert!(!required, "...{test_name}...{reason}");
    eprintln!("Skipping {test_name}: {reason}");
}
```

**변경 이유:** 두 가드가 한 구현을 쓰되, 호출자의 정체가 skip 메시지와 실패 메시지 양쪽에 보존된다.

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|------|-----|
| 변경 파일 | 5 |
| 추가 줄 | +88 |
| 삭제 줄 | -29 |
| 추가 테스트 | 0 (기존 가드 2개의 동작을 확장) |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|------|------|
| 테스트 정확성 | 1 | 로딩 쪽 가드가 게이트를 따름 |
| 리팩터링 | 1 | 게이트를 공유 test-support 헬퍼로 추출 |
| 문서화 | 1 | 변수 문서화, 켜는 시점과 방법 포함 |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|--------|
| `3a1c014e` | fix(test) | gate the loading-side pinned Muse Glimmer guard on MLXCEL_REQUIRE_PINNED_CHECKPOINTS |
| `d3993635` | docs | fold the pinned-checkpoint gate into the existing test-variable table |
| `d2b4f419` | docs | correct the pre-#1173 behavior described for the pinned gate |
| `758fe64c` | docs | name PR #1157 as the pre-#1173 pinned availability guard |

---

## 8. 후속 조치

### 완료 필요
- [ ] 없음. 수용 기준 다섯 가지 모두 충족

### 향후 개선 사항
- `skip_or_fail_pinned_checkpoint`는 이미 `env_lock()`을 쥔 테스트 안에서 호출되면 자기 교착에 빠진다. 현재 호출부는 그런 경우가 없고 헬퍼가 락을 잡는 이유는 설명돼 있지만, 호출자에게 락을 쥔 채 부르지 말라는 경고는 없다. 세 번째 고정 가드가 생기면 기억할 일이다.
- `src/lib.rs:47-49`는 아직 `test_support`를 "the single shared `ENV_LOCK`"을 제공하는 곳으로 서술한다. 이제 그곳의 모듈은 둘이다.
- `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=true`는 조용히 아무 일도 하지 않는다. `1` 엄격 일치는 이슈가 지정한 사항이고 문서도 명시하지만 날카로운 모서리다.

---

## 부록

### A. 테스트 결과

| 명령 | 결과 |
|------|------|
| `cargo test --lib -- vision::encoders::muse_glimmer_fusion loading::vlm::muse_glimmer` | 37 passed, 0 failed |
| 위와 동일 + `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` | 37 passed, 0 failed |
| `cargo clippy --lib --tests -- -D warnings` | clean |
| `cargo fmt --check` | clean |

skip 및 게이트 실패 경로는 `models/`가 없는 임시 디렉터리에서 컴파일된 테스트 바이너리를 실행해 확인했다. 실제 체크포인트는 건드리지 않았다.

| 조건 | 결과 |
|------|------|
| 게이트 미설정 | 두 가드 모두 skip 후 `ok`, 각자 이름을 밝힘: `Skipping pinned_weight_index_classifies_each_source_weight_once: ...`, `Skipping pinned_post_tower_weight_roots_and_shapes_match_published_contract: ...` |
| `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` | 두 가드 모두 FAIL, 각자 이름을 밝힘: `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1 but the pinned checkpoint <test_name> needs is not usable: ...` |

첫 표의 둘째 행이 "완전한 체크포인트에서 계약이 그대로 검증된다"는 기준의 핵심 증거다. 게이트가 모든 skip을 실패로 바꾸므로, 그 상태에서 두 가드가 통과했다는 것은 둘 다 실제 60 GB 체크포인트를 읽었다는 뜻이지 건너뛴 것이 아니다.

### B. 문서 정정 과정에 관한 메모

#1173 이전 동작을 설명한 문단은 맞기까지 두 번 틀렸다. 순서를 잘못 잡기 쉬운 대목이라 기록해 둔다. 첫 판본은 #1173 이전에 availability pre-check가 패닉했다고 적었는데, 그때는 일반 pre-check가 없었다. 이어진 수정은 pre-check가 아예 없었다고 적었는데 이번엔 과잉교정이었다. PR #1157이 이미 vision 쪽의 `assert!(index_path.exists(), ...)`를 조용한 skip으로 바꾸고 로딩 쪽에도 같은 인덱스 부재 skip을 추가해 뒀기 때문이다. 지금 문서에 담긴 정확한 순서는 이렇다. #1157이 인덱스 부재 경우를 조용하게 만들었고, #1173이 인덱스는 있으나 불완전한 경우를 조용하게 만들었으며, 게이트가 두 경우 모두에 대해 요란함을 되돌린다.

### C. 참고 자료
- 이슈 #1177과 범위 추가 코멘트, PR #1173과 그 기술 보고서(후속 항목으로 기록한 곳), 이슈 #1161, PR #1157
- `src/test_support/pinned_checkpoint.rs`(공유 게이트), `src/test_support/env_lock.rs`(잡는 락), `docs/environment-variables.md`(문서)
- PR #1180의 리뷰 및 보안 코멘트
