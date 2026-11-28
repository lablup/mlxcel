# 기술 보고서: PR #2014 - 레지스트리 스냅샷 생명주기를 설명하는 recipes/README 추가

**날짜**: 2026-09-28
**상태**: 완료
**언어**: Markdown
**위험도**: 낮음

## 요약

PR #2014는 `recipes/` 디렉터리에 처음으로 문서를 추가한다: `recipes/README.md`를 새로 작성하고, `docs/supported-models.md`에서 이 문서로 향하는 한 줄짜리 링크를 추가했다. `recipes/` 트리는 그동안 아키텍처 레지스트리의 커밋된 JSON 스냅샷만 담고 있었을 뿐, 이를 설명하는 글은 없었다. 그럼에도 `.github/ISSUE_TEMPLATE/recipe_request.yml`은 외부 기여자를 이 디렉터리로 적극적으로 유도하고 있었다.

## 1. 문제 정의

### 1.1 배경

`recipes/registry/`는 `make recipes-registry` Makefile 타깃이 생성하는 `mlxcel arch --json` 출력의 버전별 스냅샷과, 그 중 `CURRENT` 버전을 가리키는 포인터 파일을 담고 있다. 메커니즘 자체는 존재했고 정상 동작했지만, `CHANGELOG.md`의 한 줄짜리 언급과 Makefile 타깃 자체의 주석 외에는 이를 설명하는 글이 저장소 어디에도 없었다.

### 1.2 기존 문제점

- **안내 문서 부재**: 이슈 템플릿을 따라 `recipes/`로 온 기여자는 JSON 파일만 보고, 어느 수준에도 README가 없었다.
- **문서화되지 않은 생명주기**: 스냅샷 파일명의 의미, `make recipes-registry`를 실행해야 하는 시점, `CURRENT`의 용도, 스냅샷과 `mlxcel arch`의 관계는 모두 Makefile 타깃을 직접 읽어야만 추론할 수 있었다.
- **오래된 이슈 전제**: 이슈 #1677의 파일 목록(`0.6.0.json`, `0.7.0-beta.1.json`, `CURRENT`)과 Makefile 줄 번호(147-160)는 모두 현재 트리와 어긋나 있었다. 현재 트리는 `0.7.0.json`도 추가로 보유하고 있고, 타깃은 203-216번 줄에 있다. 이 PR은 이슈 본문이 아니라 검증된 현재 상태를 기준으로 작성되었다.

### 1.3 위험도 평가

낮음. 문서만 변경되었으며 빌드, 런타임, CI 동작에는 영향이 없다.

## 2. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경된 파일 | 2 |
| 추가된 줄 | 28 |
| 삭제된 줄 | 0 |

- `recipes/README.md` (신규, 27줄): `recipes/`가 레시피 자체가 있는 곳이 아니라는 점(레시피는 `mlxcel.ai/recipes`에 있다), `recipes/registry/<version>.json` 스냅샷이 담고 있는 내용, `make recipes-registry`가 이를 재생성하는 방식, `CURRENT`의 의미, 그리고 스냅샷이 `mlxcel arch`와 `mlxcel arch --json`에 대해 갖는 관계를 설명한다.
- `docs/supported-models.md` (+1줄): 기존의 `mlxcel arch --json` 문서에서 새 README로 향하는 링크 항목을 추가해, 이미 문서화된 레지스트리의 JSON 스키마와 그 디스크상 스냅샷 메커니즘이 양방향으로 교차 참조되도록 했다.

## 3. 기술적 선택과 그 이유

### 3.1 이슈의 파일 목록이 아니라 현재 트리를 기준으로 검증

**배경**: 이슈 본문은 `recipes/registry/`가 `0.6.0.json`, `0.7.0-beta.1.json`, `CURRENT`만 담고 있다고 적었고, Makefile 타깃을 147-160번 줄로 인용했다.

**근거**: 현재 트리는 `0.7.0.json`도 함께 보유하고 있고(`CURRENT`는 `0.7.0`을 가리킴), 타깃은 203-216번 줄에 있다. 오래된 목록을 그대로 옮겨 적었다면 병합 시점에 이미 틀린 문서가 되었을 것이다. README는 정확한 줄 번호나 파일 전체 목록을 나열하는 대신 메커니즘을 이름과 패턴으로 설명하므로, 같은 방식으로 금방 낡지 않는다.

### 3.2 백엔드 필드를 고정된 집합이 아니라 스냅샷별로 설명

**배경**: 초안은 스냅샷이 "Metal/CUDA 백엔드 상태"를 담는다고 서술했다.

**근거** (리뷰 과정에서 발견, 4절 참고): `mlxcel arch --json`은 이제 `rocm` 항목도 내보내지만(`src/models/registry.rs`), 커밋된 `0.7.0.json`을 비롯한 이전 스냅샷들은 이 필드가 도입되기 전에 생성되었다. 고정된 필드 집합으로 서술하면 커밋된 스냅샷에는 맞지만 다음 재생성 결과에는 틀리게 된다. 문서가 스키마 확장에도 계속 정확하도록, 백엔드 필드를 스냅샷이 생성된 시점의 속성으로 서술하도록 수정했다.

## 4. 검증

- `python3 "$HOME/.claude/skills/commit-conventions/scripts/validate_body.py" recipes/README.md`: 통과 (하드랩된 문단 없음).
- `python3 scripts/ci/check_cross_repo_refs.py`: 통과 (새로 추가된 저장소 간 참조 없음).
- README의 모든 사실 주장을 실제 `Makefile` 타깃, `recipes/registry/*.json`과 `CURRENT`의 실제 내용과 JSON 형태, `.github/ISSUE_TEMPLATE/recipe_request.yml`과 수동으로 대조 확인했다.
- 독립적인 `pr-reviewer` 검토에서 두 가지 정확성 문제(백엔드 필드 서술, 스냅샷 덮어쓰기 의미론)를 발견해 후속 커밋으로 수정했다. `pr-security-checker`와 `pr-finalizer` 검토에서는 추가 문제가 발견되지 않았다. 이 저장소에는 마크다운 린터가 구성되어 있지 않아 실행하지 않았다.
- 실행하지 않음: 문서만 변경하는 작업이므로 적용 가능한 빌드나 테스트 스위트가 없다.

## 5. 관련 작업

- 이슈 #1677: 이 변경의 원본 이슈.
- `docs/supported-models.md`: 레지스트리 스냅샷이 직렬화하는 `mlxcel arch --json` 스키마를 이미 문서화하고 있는 문서.
- `.github/ISSUE_TEMPLATE/recipe_request.yml`: 기여자를 `recipes/`로 유도하는 진입점.
