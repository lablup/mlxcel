# 기술 보고서: PR #1120 - docs: fix stale cross-references left behind by earlier renames

**작성일**: 2026-08-14
**상태**: 완료
**언어**: Markdown, Rust (주석만)
**위험도**: Low

---

## 요약

PR #1120은 이슈 #1106을 해결한다. 과거 리네임 작업이 남긴 네 곳의 문서 참조를 고쳤다. 삭제된 `AGENTS.md`를 가리키던 PR 템플릿, 같은 죽은 참조가 남아 있던 `tests/surgery_cli.rs` 문서 주석, `docs/` 아래 두 곳의 `v0.0.27` 버전 스탬프, 그리고 `docs/README.md` 목차에서 빠져 있던 항목이다.

코드 변경과 동작 변경은 없다. diff에 포함된 `.rs` 파일 하나는 상수에 달린 rustdoc 주석이며, 상수 값 자체는 그대로다.

---

## 1. 문제 정의

### 1.1 배경

CI가 검증하지 않는 문서 영역에 서로 독립적인 staleness 버그 세 건이 쌓였다. 각각 한 줄짜리이고, 각각 독립적으로 검증 가능하며, 각각 다른 방식으로 독자를 오도한다.

### 1.2 기존 문제점

- **죽은 `AGENTS.md` 참조.** `.github/PULL_REQUEST_TEMPLATE.md:3`이 첫 기여자 전원을 체크아웃에 존재하지 않는 파일로 안내하고 있었다. `AGENTS.md`는 `.gitignore`에 다른 로컬 전용 작업 파일들과 함께 등재되어 있다. 즉 설계상 로컬 파일이며 어떤 clone에도 존재하지 않는다. `CHANGELOG.md`는 PR #1014가 `CONTRIBUTING.md`의 죽은 `AGENTS.md` 링크를 교체했다고 기록하지만, 그 작업은 세 표면 중 트래픽이 가장 많은 템플릿을 놓쳤다. 이번 작업에서 저장소 전체 grep으로 이슈가 열거하지 않은 네 번째 지점도 찾았다. `tests/surgery_cli.rs`의 `REFERENCE_MODEL` 문서 주석이다.
- **`v0.0.27` 버전 스탬프.** `docs/supported-models.md`는 "model-family support in the v0.0.27 source tree"로 시작했고, `docs/turbo-kv-cache.md`는 allowlist 앞에 "As of v0.0.27"을 달고 있었다. 워크스페이스 버전은 `0.5.0-beta.1`이다. `supported-models.md`는 `README.md`와 `CONTRIBUTING.md`가 모두 지원 매트릭스를 보라고 안내하는 페이지이므로, 이 스탬프는 페이지 전체를 여러 릴리스 뒤처진 문서처럼 보이게 만든다.
- **불완전한 목차.** `docs/README.md`는 18개 문서를 열거했지만 `docs/`에는 README를 제외한 `.md` 파일이 19개 있다. 빠진 항목은 `code-guidelines.md`로, `CONTRIBUTING.md`가 지목하는 세 개의 핵심 계약 문서 중 하나다. 나머지 둘인 `architecture.md`와 `adding-models.md`는 이미 목차에 있었다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| 기여자가 PR 템플릿에서 기여 계약 문서를 찾지 못함 | Medium | High |
| 스탬프 때문에 현재 유효한 지원 매트릭스를 낡은 문서로 판단해 무시함 | Medium | Medium |
| 기여자가 `code-guidelines.md`를 끝내 찾지 못하고 dtype이 캐시 키에 빠진 JIT 커널을 머지함 | Medium | Low |

---

## 2. 기술적 검토 사항

### 2.1 정확성

`v0.0.27` 수정은 스탬프를 제거하기 전에 이것이 내용 문제가 아니라 스탬프 문제임을 먼저 확인했다. `docs/turbo-kv-cache.md`가 열거한 세 개의 allowlist model-type prefix(`qwen3_5`, `qwen3_5_moe`, `qwen3_next`)를 `src/lib/mlxcel-core/src/cache/turbo/allowlist.rs`의 `ALLOWED_SYMMETRIC_TURBO_FAMILIES`와 대조한 결과 정확히 일치했다. 두 스탬프 뒤의 내용은 다시 쓸 필요가 없었다.

목차 수정은 눈으로가 아니라 산술로 확인했다. `README.md`를 제외한 `ls docs/*.md`의 basename 정렬 결과와 번호 목록에서 추출한 파일명 정렬 결과가 이제 양쪽 모두 19개로 완전히 동일하다.

### 2.2 범위 통제

이슈 #1106의 수용 기준 4는 "docs only; no code or behavior change"이고, 기준 1은 "`CHANGELOG.md` 밖에 `AGENTS.md` 참조가 남아 있지 않을 것"이다. `tests/surgery_cli.rs`의 참조는 두 기준 사이에 놓인다. 기준 1을 만족하려면 `.rs` 파일을 건드려야 한다. 수정은 `const`에 달린 rustdoc 주석 문장으로 한정했으므로 기준 4도 실질적으로 유지된다. `rustfmt`는 기본값에서 주석을 재정렬하지 않으므로(`wrap_comments`가 off) `cargo fmt --all -- --check`에 영향이 없고, 해당 파일은 어차피 `#![cfg(feature = "surgery")]` 뒤에 있다.

남아 있는 `AGENTS.md` 언급 두 종류는 모두 의도적이다. `CHANGELOG.md`의 세 항목은 이슈가 명시적으로 보존하라고 한 기록이고, `.gitignore` 한 줄은 참조가 아니라 ignore 패턴이다.

---

## 3. 기술적 선택과 그 이유

### 3.1 버전 스탬프를 올리지 않고 제거

| 옵션 | 장점 | 단점 |
|---|---|---|
| 양쪽을 `0.5.0-beta.1`로 갱신 | 오늘 기준으로는 정확 | 아무도 검증하지 않으므로 다음 릴리스에 다시 낡음 |
| **선택: 스탬프 제거 후 코드 포인터 강화** | 낡을 수 없고, 독자를 권위 있는 출처로 보냄 | 스냅샷 시점을 알고 싶던 독자는 "as of" 신호를 잃음 |

`docs/supported-models.md`에는 이미 소스 경로 네 개를 지목하는 "the runtime source of truth is the code" 블록이 있어서, 스탬프만 제거해도 페이지가 지속 가능한 대상을 가리키게 되고 추가 편집이 필요 없었다. `docs/turbo-kv-cache.md`는 `allowlist.rs`를 언급하되 그것이 권위 있는 출처라고는 말하지 않았으므로, 제거한 스탬프 자리에 그 주장을 넣었다. "which is the source of truth for this list."

### 3.2 PR 템플릿이 두 문서를 가리키게 함

`AGENTS.md`는 이미 focused reference docs로 분해되었다(`CHANGELOG.md`가 313줄에서 75줄로 줄인 분할을 기록하고 있다). 그것이 담고 있던 계약은 이제 `CONTRIBUTING.md`와 `docs/code-guidelines.md`에 있고, `CONTRIBUTING.md`가 후자를 링크한다. 템플릿의 체크리스트가 이미 `docs/code-guidelines.md`를 정확히 그 경로로 인용하고 있으므로, 헤더에서 둘을 함께 지목하면 본문과 일관된다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 5 |
| 추가된 라인 | +9 |
| 삭제된 라인 | -7 |
| 테스트 추가 | 0 |

### 영역별 변경

| 영역 | 파일 | 주요 내용 |
|---|---|---|
| 기여자 온보딩 | `.github/PULL_REQUEST_TEMPLATE.md` | 죽은 `AGENTS.md` 참조를 `CONTRIBUTING.md`와 `docs/code-guidelines.md`로 재지정 |
| 문서 정확성 | `docs/supported-models.md`, `docs/turbo-kv-cache.md` | `v0.0.27` 스탬프 제거, allowlist 절에 source-of-truth 포인터 추가 |
| 문서 목차 | `docs/README.md` | `code-guidelines.md`를 19번 항목으로 추가 |
| 주석 정리 | `tests/surgery_cli.rs` | `REFERENCE_MODEL` 문서 주석에서 죽은 `AGENTS.md` 참조 제거 |

### 관련 커밋

| Hash | Type | Message |
|---|---|---|
| `e251af08` | docs | docs: fix stale cross-references left behind by earlier renames |

---

## 5. 검증 및 후속 조치

### 통과

- `python3 scripts/ci/check_cross_repo_refs.py` (새로 추가된 3자리 이상 bare `#NNN` 없음).
- `cargo fmt --all -- --check`.
- 저장소 전체 `grep -rn "AGENTS\.md"`: `CHANGELOG.md`의 기록 세 건과 `.gitignore` 패턴만 남음.
- `grep -rn "v0\.0\.27" docs/`: 매치 없음.
- `ls docs/*.md` basename 정렬 결과와 `docs/README.md` 번호 목록의 정렬 diff: 동일, 양쪽 19개.
- PR #1120 CI: crate versions, kernel dtype keys, cross-repo refs, cargo-deny, cargo-fmt 전부 green.

### 관련 작업

- 이슈 #1110이 `docs/code-guidelines.md` 본문을 동시에 수정하고 있다. 이 PR은 그 파일을 지목하는 목차 항목만 추가하고 내용은 건드리지 않으므로 충돌하지 않는다.
- 이슈 #1111은 `docs/README.md`에서 새 19번 항목 몇 줄 아래의 "Expected future layout examples" 블록을 수정한다. 그 때문에 이 PR 다음 순서로 배치했다.

### 처리하지 않은 것

스탬프는 사라졌지만, 앞으로 누군가 새 스탬프를 다는 것을 막는 장치는 없다. 문서의 주장이 코드와 여전히 일치하는지 검사하는 CI는 없다. 이번 완화책은 구조적인 것으로, 코드를 권위 있는 출처로 지목해 산문이 낡을 여지 자체를 줄이는 방식이다.
