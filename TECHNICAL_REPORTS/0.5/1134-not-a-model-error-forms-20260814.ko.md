# 기술 보고서: PR #1134 - fix(cli): name all three accepted forms in the "not a model" error

**작성일**: 2026-08-14
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Low

---

## 요약

PR #1134는 이슈 #1114를 해결한다. 해석할 수 없는 `-m/--model` 값에 대한 최종 오류가 허용 형식을 두 가지로 설명했는데 해석기는 세 가지를 받는다. 빠진 것은 `$MLXCEL_DEFAULT_ORG`로 해석되는 접두사 없는 bare 이름이었다.

문장 하나 빠진 것치고 파장이 컸다. 이 오류 갈래의 위치 때문이다. bare 이름은 `is_repo_segment`를 통과하지 못할 때만, 즉 `[A-Za-z0-9._-]` 밖 문자가 들어 있을 때만 여기에 도달한다. 그러니 여기 도착하는 전형적인 사용자는 bare 이름에 엉뚱한 문자를 하나 넣은 사람이고, 메시지는 그들이 쓰지 않은 두 형식을 알려 주면서 그중 더 수고스러운 쪽을 권했다.

메시지 텍스트만 바뀐다. 해석 로직 변경은 없다.

---

## 1. 문제 정의

### 1.1 배경

해석기의 우선순위는 허용 형식 세 개와 오류 갈래 하나다.

1. 존재하는 로컬 경로, 그대로 사용
2. `owner/name` repo-id
3. 단일 bare 세그먼트, `<$MLXCEL_DEFAULT_ORG>/<segment>`로 확장(기본 `mlx-community`)
4. 그 외에는 `not_a_model_error`

3번은 일급 형식이다. 모듈 문서가 설명하고, `README.md`는 quick start에 `mlxcel run Qwen3.5-0.8B-4bit`와 "Bare name resolves to mlx-community/<name>" 주석을 넣어 두었다. 오류는 1번과 2번만 언급했다.

### 1.2 기존 문제점

- **메시지가 해석기와 모순됐다.** "neither an existing path nor a valid HuggingFace repo-id"는 전수를 주장하는 어법인데, 실제 답의 3분의 1이 빠진 집합에 대해 그렇게 말했다.
- **하필 발동하는 지점에서 틀렸다.** `is_repo_segment`는 `[A-Za-z0-9._-]`를 받으므로 공백이나 다른 문자가 섞이면 오류 갈래로 떨어진다. 이 메시지를 보는 집단은 bare 이름 사용자로 크게 기울어 있고, bare 이름이 바로 언급되지 않은 형식이었다.
- **권한 수정 방법이 더 비싼 쪽이었다.** 문자 하나만 고치면 되는데 "로컬 디렉터리나 repo-id를 넘기라"며 완전한 `owner/name`을 요구했다.
- **파일 안에서 일관되지 않았다.** 같은 경로의 `bad_default_org_error`는 이미 "the org must be a single path segment (`[A-Za-z0-9._-]`)"라고 문자 클래스를 적고 있었다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| README가 광고하는 bare 이름 사용을 사용자가 포기 | Low | High |
| 오타 난 bare 이름을 지원하지 않는 문법으로 오해 | Medium | Medium |
| 목록을 고정하는 장치가 없어 향후 수정에서 형식이 다시 누락 | Low | Medium |

---

## 2. 기술 검토

### 2.1 메시지

이슈가 든 예시 그대로 비교한다.

```
$ mlxcel generate -m "Qwen3 4B" -p x -n 1

# 이전
Error: model 'Qwen3 4B' is neither an existing path nor a valid HuggingFace repo-id (expected
`owner/name`, e.g. `mlx-community/Qwen3-4B-4bit`). Pass a local model directory or a repo-id to
auto-download.

# 이후
Error: model 'Qwen3 4B' is not a model mlxcel can resolve. Accepted forms: a local model directory;
a HuggingFace repo-id `owner/name` (e.g. `mlx-community/Qwen3-4B-4bit`); or a bare model name made
only of [A-Za-z0-9._-], which resolves against $MLXCEL_DEFAULT_ORG (default `mlx-community`). A
repo-id or bare name is auto-downloaded.
```

### 2.2 첫 절을 바꿔야 했던 이유

이슈는 메시지를 확장하라고 했다. 확장만으로는 안 됐다. "is neither an existing path nor a valid HuggingFace repo-id" 자체가 두 형식의 열거라서, 세 번째를 덧붙이면 문장이 자기 목록과 모순된다. 이제 첫 절은 열거하지 않는 서술("is not a model mlxcel can resolve")이고, 열거는 목록에서 한 번만 일어난다.

파급이 있는 유일한 변경이다. 옛 부분 문자열을 단언하던 테스트가 둘 있었고 모두 갱신했다. 이슈가 바로 그 확인을 요청했다.

### 2.3 범위

`is_repo_segment`, `expand_bare_name`, `resolve_model_source_with_override`와 모든 해석 분기는 손대지 않았다. 동작 표면은 `anyhow!` 하나의 텍스트뿐이다.

---

## 3. 기술적 결정

### 3.1 문자 클래스를 서술하지 않고 그대로 적는다

"made only of `[A-Za-z0-9._-]`"는 `is_repo_segment`가 강제하는 바로 그 클래스이고, `bad_default_org_error`가 쓰는 방식과 같다. 산문 풀이("문자, 숫자, 점, 밑줄, 하이픈")가 읽기는 매끄럽지만 더 쉽게 어긋난다. 리터럴 클래스는 자신이 설명하는 술어와 grep으로 대조된다.

### 3.2 목록을 테스트로 고정한다

옛 테스트는 각각 부분 문자열 하나만 단언했고, 그래서 목록이 낡아도 아무것도 실패하지 않았다. 갱신된 테스트는 세 형식 이름, 문자 클래스, `$MLXCEL_DEFAULT_ORG` 언급, 보존된 예시를 모두 고정한다. 이 결함 자체가 근거인 값싼 보험이다. 메시지를 해석기에 책임지게 만드는 장치가 없었다.

### 3.3 예시를 보존한다

`mlx-community/Qwen3-4B-4bit`는 이슈의 세 번째 수용 기준대로 그대로 유지했고, 이제 메시지 전체가 아니라 2번 형식의 예시로 정확히 붙는다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 3 |
| 해석 로직 변경 | 0 |
| 명시된 허용 형식, 이전 / 이후 | 2 / 3 |

### 영역별 변경

| 영역 | 파일 | 요약 |
|---|---|---|
| 해석기 | `src/downloader/resolver.rs` | `not_a_model_error`가 세 형식을 모두 명시하고 문자 클래스와 기본 org를 밝힘. 주석은 이 갈래에서 bare 이름 형식이 왜 지배적인지 설명 |
| 테스트 | `src/downloader/resolver_tests.rs` | 옛 부분 문자열 단언 두 곳 갱신. 첫 번째는 세 형식, 클래스, org 변수, 예시까지 고정 |
| 문서 | `CHANGELOG.md` | `## [Unreleased]` / `### Fixed` 항목 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `7c1744ad` | fix | fix(cli): name all three accepted forms in the "not a model" error |

---

## 5. 검증 및 후속 과제

### 통과

- `cargo test --profile test-fast --features cuda --lib downloader::resolver`: 33 passed, 0 failed.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`: 통과(파이프라인이 아니라 종료 코드를 직접 확인).
- `cargo fmt --all -- --check`: 통과.
- 빌드된 바이너리 기준: `mlxcel generate -m "Qwen3 4B"`가 세 형식 메시지를 출력한다. 이슈의 재현 예시 그대로다. `no/such/nested/path`도 같은 메시지에 도달해 다중 세그먼트 갈래가 그대로임을 확인했다.
- `grep -rn "neither an existing path" src/ tests/ docs/ README.md`: 시나리오를 설명하는 테스트 주석 하나 외에 옛 텍스트 참조 없음.

### 후속 후보

- **오류 메시지의 열거를 그것이 설명하는 코드에 묶어 두는 테스트가 없다.** 이 메시지는 3번 형식이 생긴 이래로 계속 해석기와 어긋나 있었고, 이번 수정도 손으로 쓴 단언 목록이다. 같은 종류의 drift를 플래그 표면에 대해서는 `tests/cli_help_consistency.rs`가 처리하는데, 오류 메시지 불변식에는 대응하는 거처가 없다.
