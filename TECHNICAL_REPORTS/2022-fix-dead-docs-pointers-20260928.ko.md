# 기술 리포트: PR #2022, 존재하지 않는 docs/model_tests.md, docs/testing.md 참조 수정

**작성일**: 2026-09-28
**상태**: 완료
**언어**: Rust, TOML
**위험도**: 낮음

## 요약

소스 코드 여섯 곳(`Cargo.toml`의 주석 한 줄, `src/bin/speculative_bench.rs`의 문서 주석 두 곳, `tests/speculative_parity.rs`의 컴파일 타임 assert 메시지 한 곳, `tests/prompt_cache_prefill_bench.rs`의 문서 주석 한 곳, `src/tokenizer/mod.rs`의 주석 한 곳)이 `docs/model_tests.md` 또는 `docs/testing.md`를 가리키고 있었지만, 두 파일 모두 이 저장소에 존재한 적이 없다. 다섯 곳은 실제로 해당 내용을 담고 있는 `docs/benchmark_results/model_tests.md`로 재지정했고, 나머지 한 곳은 원래 의도했던 대상이 이 저장소 밖에 있는 것으로 확인되어 포인터 자체를 제거하고 인라인 설명으로 대체했다. 다섯 개 수정 파일에 두 문자열이 다시 나타나지 않도록 지키는 회귀 테스트 `tests/dead_doc_pointers.rs`를 새로 추가했다.

## 1. 문제 정의

### 1.1 배경

이슈 #1658은 이슈 #26과 같은 종류의 결함이다(존재하지 않는 `docs/model_implementations.md` 참조, `src/main_tests.rs`의 `supported_models_output_has_no_dead_doc_link`가 이를 방지한다). #26과 다른 점은, 이번 죽은 경로들이 CLI 렌더러 출력이 아니라 소스 주석과 assert 메시지 안에 있어서 기존 가드가 이를 볼 수 없다는 것이다.

### 1.2 기존 문제점

- **문제 1**: `docs/model_tests.md`는 실존한 적이 없는 경로다. `git log --all --diff-filter=A -- docs/model_tests.md`는 아무것도 반환하지 않는다. 원래 의도된 파일은 `docs/benchmark_results/model_tests.md`이며, 다섯 개 참조가 설명하는 `## Speculative drafters`(317행)와 `## Prompt cache benchmarks`(240행) 섹션을 정확히 담고 있다.
- **문제 2**: `docs/testing.md` 역시 실존한 적이 없다. 이를 도입한 커밋(`290eb645`)은 `docs/en/user-guide/server.md`도 함께 수정했고, `mkdocs.yml`의 nav에는 `development/testing.md`가 등록되어 있다. `docs/README.md`의 "The MkDocs manual" 절에 따르면 `docs/en/`과 `docs/ko/`는 별도로 유지되는 매뉴얼(`mlxcel.lablup.ai`에 게시)의 소스이며 이 저장소에 포함되지 않는다고 명시되어 있다. 그 경로로 재지정해도 이 저장소 기준으로는 여전히 죽은 링크가 된다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|--------|------------|
| 새로운 speculative 페어링을 포팅하거나 prompt-cache 벤치마크를 작성하는 기여자가 포인터를 따라갔다가 아무것도 찾지 못하거나, 실제 모델 테스트 설정 방법을 찾지 못하고 포기함 | 낮음 | 수정 전에는 확실했고, 이제 해소됨 |

## 3. 기술적 선택과 그 이유

### 3.1 재지정 vs. 제거, 참조별 개별 판단

**컨텍스트**: 이슈는 참조마다 개별 판단을 요구했다. 실제 현재 위치로 재지정할지, 대상이 애초에 실존한 적이 없다면 포인터 자체를 제거할지.

**`docs/testing.md`에 대해 검토한 대안**:

| 옵션 | 장점 | 단점 |
|--------|------|------|
| `docs/en/development/testing.md`로 재지정 | 원래 의도(TurboQuant 테스트 섹션)와 부합 | 이 경로는 별도 매뉴얼 트리(이 저장소 밖)에 있어, 여기서는 여전히 죽은 링크가 됨 |
| `CLAUDE.md`의 "Testing with real models" 섹션으로 재지정 | 주제상 근접(모델을 내려받아 테스트 실행) | `CLAUDE.md`는 이 저장소에서 `.gitignore` 대상이며 `git ls-files`에 없음: 공개되는 트리의 일부가 아님 |
| **선택: 포인터 제거, 필요한 사실 한 가지만 인라인화** | 주석이 그 자체로 완결되고 정확해짐; 향후 죽은 링크가 될 위험 없음 | 더 넓은 범위의 외부 매뉴얼을 찾고 싶은 독자를 위한 "참고" 링크를 잃음 |

**근거**: 두 대안 모두 이 이슈가 다루는 것과 동일한 문제, 즉 이 저장소 트리 밖을 가리킨다는 문제를 그대로 가진다. 주석이 없어진 문서에서 실제로 필요했던 정보는 "모델을 어떻게 받는가" 하나뿐이었으므로, `mlxcel download mlx-community/gemma-4-e4b-it-8bit`(`docs/model-catalog.tsv:63`과 대조해 확인)를 인라인으로 넣어 죽은 포인터 없이도 주석이 자체적으로 완결되도록 했다.

### 3.2 `src/main_tests.rs` 확장 대신 별도의 좁은 테스트 파일 추가

**컨텍스트**: 이슈는 선택 사항으로, 기존 #26 가드 테스트를 확장해 이 두 경로도 함께 검사하도록 제안했다.

**근거**: `supported_models_output_has_no_dead_doc_link`는 `write_supported_models()`의 렌더링 결과를 검사하는 테스트다. 이번에 고친 여섯 개의 죽은 포인터는 원본 소스 텍스트 안에 있고, 대응하는 렌더러가 없다. `src/main_tests.rs`는 이미 1535행으로 프로젝트의 500행 가이드라인을 크게 넘어선 상태라, 무관한 검사를 더 얹어 파일을 키우는 것도 피했다. 대신 `tests/dead_doc_pointers.rs`가 수정 대상 다섯 개 소스 파일을 `include_str!`로 읽어 두 죽은 문자열이 다시 나타나지 않는지 확인한다.

**트레이드오프**: 이 가드는 의도적으로 좁다(지정된 다섯 개 파일, 지정된 두 개 문자열). "소스에서 언급되는 모든 `docs/*.md` 경로가 실존해야 한다"는 전체 트리 스캐너가 아니다. 구현 중 저장소 전체를 grep한 결과, 이 이슈 범위 밖의 기존 죽은 참조들(예: `docs/USAGE.md`, `docs/bridge-overhead-microbench.md`)이 발견되었다. 범용 스캐너였다면 이런 참조들에서도 실패했을 것이며, 이는 별도의 정리 작업이다.

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/tokenizer/mod.rs`**
```rust
// 변경 전
// is missing so the test suite stays portable; run on demand with
// `cargo test -- --ignored` against a workspace that has the model
// downloaded (per `docs/testing.md`).

// 변경 후
// is missing so the test suite stays portable; run on demand with
// `cargo test -- --ignored` against a workspace that has the model
// downloaded (`mlxcel download mlx-community/gemma-4-e4b-it-8bit`).
```

나머지 다섯 곳은 동일한 재지정 패턴이다: `docs/model_tests.md`를 `docs/benchmark_results/model_tests.md`로 바꾸며, `tests/speculative_parity.rs`의 컴파일 타임 `assert!` 메시지도 포함된다. 이 메시지는 여전히 포맷 인자가 없는 단순 문자열 리터럴이므로, 실제로 발동하더라도 정상적으로 읽힌다.

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-------|
| 변경된 파일 | 6 |
| 추가된 줄 | +75 |
| 삭제된 줄 | -8 |
| 추가된 테스트 | 2 (`tests/dead_doc_pointers.rs`) |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|-------|---------|
| 문서화 | 참조 5곳 | `docs/model_tests.md`를 `docs/benchmark_results/model_tests.md`로 재지정 |
| 문서화 | 참조 1곳 | `docs/testing.md` 포인터 제거, 다운로드 명령을 인라인화 |
| 테스트 | 신규 파일 1개 | `tests/dead_doc_pointers.rs`: 두 문자열의 재발을 막는 회귀 가드 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|---------|
| `004b8b9` | docs | Fix dead docs/model_tests.md and docs/testing.md pointers |
| `a21d09e` | docs | Fix comment counting and stale line ref in dead_doc_pointers.rs |

### 관련 PR/이슈

- 이슈 #26: 같은 종류의 결함이 먼저 발생했던 사례이며, 이번에 그대로 재사용할 수 없었던 `src/main_tests.rs` 가드 패턴의 출처다.
- 이슈 #1667 / PR #2016: `src/bin/speculative_bench.rs`의 `encode_prompt`를 수정하는 형제 PR. 이 PR은 해당 파일에서 충돌을 피하기 위해 문서 주석 줄만 건드리도록 범위를 제한했다.

## 8. 후속 조치

### 완료 필요

- [ ] 없음.

### 향후 개선 사항

- 이 저장소에는 이번 이슈 범위 밖의 기존 죽은 `docs/*.md` 참조가 더 있다(예: `docs/USAGE.md`, `docs/bridge-overhead-microbench.md`, `docs/metal4-fused-attention-research.md`, `docs/papers/` 하위 파일 3개). 후속 이슈에서 동일한 참조별 판단 절차를 적용할 수 있다.
