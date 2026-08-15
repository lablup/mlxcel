# 기술 보고서: PR #1171 - docs: correct apc_consistent_prefix_len doc claim that matched_len is preserved

**작성일**: 2026-08-16
**작성자**: AI Code Reviewer
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

`apc_consistent_prefix_len`(`src/server/prompt_cache/apc_lookup.rs`)의 doc 주석은 "커버된 블록이 모두 일치하면 입력 `matched_len`이 보존된다"고 적혀 있었다. 거짓이다. 이 함수는 항상 `consistent_blocks * block_size`를 반환하므로, 블록 정렬되지 않은 `matched_len`은 마지막 온전한 블록 경계 이후의 꼬리를 잃는다. 문서화된 동작을 고정하는 것처럼 보이던 단위 테스트는 차이를 감지할 수 없었다. `matched_len`이 64로 블록 크기 16에 이미 정렬돼 있었기 때문이다. 이 PR은 문서를 바로잡고, 오해를 부르는 테스트 이름을 바꾸고, 내림과 보존을 실제로 구분하는 비정렬 `matched_len` 테스트를 추가한다.

프로덕션 동작은 바뀌지 않았다. `apc_lookup.rs`에서 바뀐 모든 줄이 `///`로 시작함을 기계적으로 확인했다. `tests/prompt_cache_e2e.rs`가 바로 이 내림 동작에서 캐시 slack 어서션을 유도하므로 이 확인은 중요하다.

---

## 1. 문제 정의

### 1.1 배경

이 결함은 PR #1158(#1156 구현)의 기술 보고서에 후속 항목으로 기록돼 있었다. 그 PR은 결정적으로 실패하던 e2e 어서션을 `DEFAULT_APC_BLOCK_SIZE`에서 slack을 유도하는 방식으로 고쳤고, 그 과정에서 내림 동작을 정밀하게 따져야 했다. 거짓 doc 줄은 그때 발견되어 선재 프로덕션 문서 버그로 PR에 지적됐고 범위 밖으로 남겨졌다.

### 1.2 기존 문제점

- **문제 1**: 문서는 전체 일치 시 `matched_len`이 보존된다고 주장했다. 이 함수의 반환은 두 종류뿐이다. 리터럴 `0`을 내는 조기 반환 세 곳과 마지막의 `consistent_blocks * block_size`. `matched_len`을 반환하는 경로는 없다.
- **문제 2**: `matching_chains_preserve_matched_len`은 이름 자체가 그 거짓 주장을 담고 있었고, 입력(`matched_len = 64`, `BLOCK = 16`)이 이미 블록 정렬이라 두 계약 중 어느 쪽에서도 동일하게 통과해 불일치를 잡을 수 없었다.
- **문제 3** (리뷰 중 발견): 최초 대체 문구인 "never the raw input `matched_len`"도 부정확했다. `matched_len`이 이미 블록 정렬이고 전체가 일치하면 반환값은 수치상 `matched_len`과 같아진다. 통과시킨 것이 아니라 계산된 결과일 뿐인데, 바로 그 혼동을 없애는 것이 목적인 PR이라면 이 구분을 명시해야 했다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| 후속 구현자가 문서에 맞춰 내림을 "고쳐" APC 안전 속성과 e2e slack 유도가 깨짐 | Medium | Low |
| 캐시 크레딧을 따질 때 문서를 믿고 예상 `cached_tokens`를 잘못 계산 | Low | Medium |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

보안 표면이 추가되지 않는다. 변경분은 doc 주석과 테스트뿐이고 함수 본문은 그대로다. 보안 검토는 모든 심각도에서 지적 0건을 반환했고, 여기서 실제로 중요한 불변식인 "검증된 분기 지점을 넘어 후보를 채택하지 않는다"를 확인했다. 서술된 동작은 언제나 토큰을 덜어내기만 하며(결과는 `matched_len`을 상한으로 갖는다), 수정된 문구는 이를 통과 방식으로 바꾸지 말라고 명시한다. `try_adopt_cached_prefix`의 dense 분기는 APC가 내림한 `matched_len`으로 정확히 잘라내고 paged 분기는 풀 블록 크기로 한 번 더 내리므로, 검증 경계를 넘어 KV를 채택하는 경로는 없다.

무제한 작업 가능성은 실제 지적이 나올 만한 유일한 지점으로 보고 검토한 뒤 근거를 갖고 배제했다. `matched_len`은 프롬프트를 통해서만 공격자 영향을 받고, 두 선택 계층 모두에서 `tokens.len()`으로 상한이 걸린다(`lookup.rs`의 `match_depth.min(dl.token_len)`, `select_best_by_scan`의 `common_prefix_len`). 또 `cap_tokens`가 `request_tokens.len()`과 한 번 더 min되므로 슬라이스 패닉이 불가능하다. 작업량은 최대 `matched_len` 토큰에 대한 선형 해시 1회다. 선재 속성이며 이 PR이 건드리지 않는다.

**발견된 이슈:**
| 이슈 | 심각도 | 상태 |
|------|--------|------|
| 대체 문구 "never the raw input `matched_len`"이 블록 정렬 경우에 대해 부정확 | Low | 수정 (`daf4a69a`) |
| 새 문단이 기존 단축 반환 문장과 이어져 rustdoc이 한 블록으로 병합 | Low | 수정 (`daf4a69a`) |
| 새로 추가한 주석에 프로젝트 스타일이 금지하는 em dash 2개 | Low | 수정 (`599b0931`) |

### 2.2 성능 관점

없음. 실행되는 프로덕션 코드가 한 줄도 바뀌지 않았다.

### 2.3 호환성/의존성 관점

- **호환성 파괴**: 없음
- **신규 의존성**: 없음
- **호환성**: 변화 없음. 반환값이 비트 단위로 동일하다

### 2.4 코드 품질 관점

- **테스트 커버리지**: 실제 계약을 고정하는 테스트 1개 증가. 이전에는 두 후보 계약을 구분할 수 있는 테스트가 없었다
- **코드 복잡도**: 변화 없음
- **기술 부채**: 감소. 안전성과 직결되는 헬퍼의 거짓 주장이 제거됐고, PR #1158 보고서에 기록된 후속 항목이 닫혔다

---

## 3. 기술적 선택과 그 이유

### 3.1 동작이 아니라 문서를 고침

**선택 이유:** 내림은 의도된 설계다. APC는 Merkle-DAG 체인을 블록 단위로 검증하므로, 부분 블록을 인정하면 검증되지 않은 토큰을 인정하는 셈이 된다. 또 `tests/prompt_cache_e2e.rs`가 바로 이 동작에서 slack(`block size - 1`)을 유도하므로, 반환값을 바꾸면 방금 고쳐 놓은 테스트가 깨진다. 틀린 쪽은 문서였다.

### 3.2 새 테스트에 73토큰과 블록 크기 16을 선택

**고려한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 기존 정렬 테스트만 두고 문서만 수정 | 변경 최소 | 계약이 고정되지 않아 문서가 다시 어긋날 수 있음 |
| 길이만 다른 정렬 테스트 추가 | 기존 스타일과 일관 | 여전히 내림과 보존을 구분하지 못함 |
| **채택: 비정렬 `matched_len = 73`, `BLOCK = 16`, 기대값 64** | 통과 방식 구현에서도, ceil 기반 상한에서도 실패함 | 부분 말단 블록에 대한 추론이 필요 |

**선택 이유:** 73은 의도적으로 16의 배수가 아니다. `BlockHashChain::compute`가 `div_ceil`을 쓰므로 후보 체인은 해시 5개를 갖고 다섯 번째는 9토큰만 덮지만, `coverable_blocks = 73 / 16 = 4`가 비교를 4블록으로 제한해 그 부분 블록은 비교되지 않는다. 기대값은 `4 * 16 = 64`이고 보존 구현이라면 73을 반환하므로, 거짓 계약에서는 두 어서션이 모두 실패한다.

보안 검토는 이 테스트가 함께 잡아내는 두 번째 변형을 짚었고 그쪽이 더 위험하다. `coverable_blocks`를 floor 대신 `div_ceil`로 계산하면 다섯 번째 후보 해시가 온전한 블록처럼 비교되어 함수가 80을 반환하고, 검증된 9토큰에 대해 16토큰을 인정하게 된다. 지금의 floor가 그 인덱스를 구조적으로 도달 불가능하게 만들며, 이 테스트가 그것을 고정한다. 따라서 이 테스트는 동어반복이 아니다.

### 3.3 기존 테스트를 삭제하지 않고 이름 변경

**선택 이유:** `matching_chains_preserve_matched_len`은 이름에 거짓 주장을 담고 있었다. 본문은 정렬 경우에 대한 유효한 검사로 여전히 쓸모가 있어, 본문을 그대로 둔 채 `aligned_matching_chains_return_the_block_floored_len`으로 이름만 바꿨다. 주석에는 이 경우가 두 계약을 구분하지 못한다는 점과 새 테스트를 가리키는 안내를 명시했다. 옛 이름을 참조하는 파일은 없었다.

---

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/server/prompt_cache/apc_lookup.rs`**
```rust
// Before
/// If every covered block agrees, the input `matched_len` is preserved.

// After
/// The return value is always a multiple of `block_size`, never a
/// pass-through of the raw input `matched_len`, even when every covered
/// block agrees: the result is `consistent_blocks * block_size`, where
/// `consistent_blocks` is capped by the blocks covered by `matched_len`
/// (`floor(matched_len / block_size)`), by the candidate chain's length,
/// and by the request's own recomputed chain length. An already
/// block-aligned `matched_len` comes back numerically equal only because
/// the arithmetic lines up, not because it was preserved. Any tokens past
/// the last agreeing block boundary are dropped, including a
/// non-block-aligned tail of `matched_len`. This flooring is deliberate and
/// should not be "fixed" to pass `matched_len` through unchanged.
```

**파일: `src/server/prompt_cache/apc_lookup_tests.rs`**
```rust
#[test]
fn non_aligned_matched_len_floors_to_covered_blocks() {
    let tokens: Vec<i32> = (0..73).collect();
    let extra = empty_extra();
    let candidate_chain = BlockHashChain::compute(&tokens, BLOCK, ApcHashAlgo::Sha256, &extra);
    assert_eq!(candidate_chain.hashes.len(), 5);
    let consistent = apc_consistent_prefix_len(
        &tokens, &candidate_chain.hashes, BLOCK, ApcHashAlgo::Sha256, &extra, tokens.len(),
    );
    assert_eq!(consistent, 64);
    assert_ne!(consistent, tokens.len());
}
```

**변경 이유:** `hashes.len() == 5` 어서션은 부분 말단 블록이 존재함을 문서화하고, `assert_ne!`는 이 테스트의 요점인 "결과는 입력이 아니다"를 직접 진술한다.

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|------|-----|
| 변경 파일 | 2 |
| 추가 줄 | +44 |
| 삭제 줄 | -4 |
| 추가 테스트 | 1 (이름 변경 1건 별도) |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|------|------|
| 문서화 | 1 | 거짓 보존 주장을 정확한 내림 서술로 교체 |
| 테스트 정확성 | 2 | 오해를 부르는 테스트 이름 변경, 비정렬 테스트 추가 |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|--------|
| `58517b0a` | docs | correct apc_consistent_prefix_len doc claim that matched_len is preserved |
| `599b0931` | docs | replace em dashes in the new apc_consistent_prefix_len comments |
| `daf4a69a` | docs | tighten the apc_consistent_prefix_len flooring wording |

---

## 8. 후속 조치

### 완료 필요
- [ ] 없음. 수용 기준 세 가지 모두 충족

### 향후 개선 사항
- `apc_lookup.rs`와 `src/` 전반의 선재 em dash는 그대로 뒀다. 저장소 전체에 걸친 상태이므로 별도 정리 작업의 몫이지 이 이슈의 범위가 아니다.

---

## 부록

### A. 테스트 결과

| 명령 | 결과 |
|------|------|
| `cargo test --lib server::prompt_cache::apc_lookup` | 10 passed, 0 failed |
| `cargo test --lib server::prompt_cache` | 168 passed, 0 failed |
| `cargo check --lib --tests` | clean |
| `cargo clippy --lib --tests -- -D warnings` | clean |
| `cargo fmt --check` | clean |

`tests/prompt_cache_e2e.rs`는 의도적으로 실행하지 않았다. `#[ignore]` 상태이고 실물 모델과 구동 중인 서버가 필요하며, 여기서 영향을 받을 수도 없다. `apc_lookup.rs`에서 `///`로 시작하지 않는 변경 줄이 하나도 없음을 기계적으로 확인했으므로 그 테스트가 의존하는 함수는 바이트 단위로 동일하다.

### B. 무변경 주장의 검증

```
git diff origin/main...HEAD -- src/server/prompt_cache/apc_lookup.rs \
  | grep -E "^[+-]" | grep -vE "^[+-]{3}" | grep -vE "^[+-]///"
```

후속 커밋 두 개 전후 모두 출력이 비어 있다.

### C. 참고 자료
- 이슈 #1160(명세), PR #1158과 그 기술 보고서(후속 항목으로 기록한 곳), 이슈 #1156
- `src/server/prompt_cache/apc_lookup.rs`(수정된 문서), `src/server/prompt_cache/block_hash.rs`(`BlockHashChain::compute`, `div_ceil`), `tests/prompt_cache_e2e.rs`(내림에 의존하는 slack 유도)
- PR #1171의 리뷰 및 보안 코멘트
