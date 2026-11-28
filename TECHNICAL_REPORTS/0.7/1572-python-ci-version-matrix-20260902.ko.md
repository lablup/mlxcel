# 기술 보고서: PR #1572 - Python 클라이언트 CI를 3.9-3.13 버전 매트릭스로 확장

**작성일**: 2026-09-02
**상태**: 완료
**언어**: YAML, TOML
**위험도**: Low

## 요약

PR #1572는 Python 클라이언트 CI의 커버리지 공백을 해소한다. `python/pyproject.toml`은 `requires-python = ">=3.9"`와 3.13까지의 classifier를 공개하지만, `.github/workflows/python.yml`은 모든 검사를 단일 고정 인터프리터(3.11)에서만 실행했다. 공개한 5개 버전 중 4개는 테스트 스위트를 한 번도 실행한 적이 없었다. 이번 변경은 워크플로를 단일 leg의 lint 잡과 3개 leg(3.9, 3.11, 3.13)의 pytest 매트릭스로 분리하고 `fail-fast: false`를 적용했으며, pyproject의 classifier를 CI가 실제로 검증하는 버전에 맞춰 좁혔다.

## 1. 문제 정의

### 1.1 배경

`.github/workflows/python.yml`은 하나의 `check` 잡에서 `ruff check`, `ruff format --check`, `mypy python/src`, `pytest python/tests -m "not e2e"`를 모두 실행했고, `actions/setup-python@v7`는 `python-version: '3.11'`로 고정되어 있었다. 파일 전체에 `strategy`/`matrix` 블록은 존재하지 않았다.

### 1.2 기존 문제점

- **검증되지 않은 버전 범위**: `requires-python = ">=3.9"`와 classifier 목록(3.9~3.13)은 5개 버전을 지원한다고 공개하지만, CI가 실제로 실행한 것은 그중 하나뿐이었다. 같은 파일에 선언된 의존성 하한(`openai>=1.40`, `httpx>=0.27`)도 그 하나의 인터프리터에서만 해석되었으므로, 다른 버전에서만 발생하는 의존성 해석 실패나 런타임 동작 차이는 눈에 띄지 않고 넘어갈 수 있었다.
- **정적 검사는 이미 floor에 고정, 런타임은 아니었음**: `ruff`(`target-version = "py39"`)와 `mypy`(`python_version = "3.9"`)는 이미 3.9 floor 기준으로 설정되어 있어 3.9 문법·타입 회귀는 이미 잡히고 있었다. 실제로 커버되지 않은 것은 테스트 스위트가 검증하는 인터프리터별 런타임 동작뿐이었다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 3.9 floor에서만 실패하는 의존성 해석이 감지되지 않고 배포됨 | Medium | Low |
| 3.12/3.13의 stdlib·런타임 동작 차이가 최신 인터프리터 사용자에게서만 클라이언트를 깨뜨림 | Medium | Low |
| classifier가 CI가 한 번도 검증하지 않은 지원 범위를 주장함 | Low | High (이미 사실이었음) |

## 2. 기술적 선택과 그 이유

### 2.1 단일 매트릭스 잡에 `if:` 가드를 추가하는 대신 `lint`/`test` 두 잡으로 분리

**컨텍스트**: 버전별로 실행할 필요가 있는 것은 `pytest`뿐이다. `ruff`와 `mypy`는 이미 `py39` 시맨틱스로 고정되어 있어 어떤 인터프리터에서 실행하든 결과가 같으므로, 세 leg 모두에서 돌리는 것은 순수한 낭비다.

**고려한 대안**:

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| 하나의 `check` 잡을 매트릭스화하고 lint/mypy 스텝에 `if: matrix.python-version == '3.11'` 가드를 추가 | 변경량이 최소 | 조건부 스텝이 잡을 지저분하게 만들고, lint/type-check 상태가 특정 매트릭스 leg의 리포팅에 종속되어 Actions UI에서 오해하기 쉬움 |
| **선택: `lint`(단일 leg)와 `test`(매트릭스) 두 잡으로 분리** | 각 잡의 목적과 성공/실패 상태가 명확함; `test` leg들은 `test (Python 3.9)`, `test (Python 3.11)`, `test (Python 3.13)`으로 독립적으로 리포트됨 | YAML이 다소 늘어남; `actions/checkout`/`actions/setup-python` 블록이 두 번 반복됨 |

**선택 이유**: 두 잡 분리는 이슈 자체의 제안("Optionally run `ruff` and `mypy` on one leg only... that keeps the matrix to pytest, which is the part that actually varies")과 일치하며, 조건부 스텝 없이 각 잡의 Actions UI 상태를 명확하게 유지한다.

### 2.2 매트릭스 구성: 전체 5개 classifier 버전이 아닌 floor·현재·ceiling(`3.9`, `3.11`, `3.13`)

**컨텍스트**: classifier 목록은 5개 버전(3.9~3.13)을 명시한다. 5개 모두 테스트하면 메타데이터/커버리지 공백을 완전히 해소하지만, 이 잡의 유일한 비용(`pip install`과 `ubuntu-latest`에서의 lint·단위 테스트, MLX나 컴파일된 바이너리 없음)을 감안하면 잡 수가 거의 두 배가 된다.

**선택 이유**: 이슈의 완료 기준은 모든 classifier 버전을 검증하거나, 두 목록이 서로 어긋나지 않는다는 조건 하에 classifier를 CI가 검증하는 범위로 좁히는 것 중 하나를 명시적으로 허용한다. 세 버전(floor·현재·ceiling)은 인터프리터 매트릭스의 표준적인 관행으로, 양 끝의 회귀를 잡아내면서 그 사이 버전은 두 끝 사이의 연속성으로 커버된다고 본다. 이번 PR은 좁히는 쪽을 택했다: `python/pyproject.toml`의 classifier 목록을 이제 3.9, 3.11, 3.13으로 매트릭스와 정확히 맞췄다. `requires-python = ">=3.9"`는 그대로 무제한으로 남겨두었는데, 이 필드는 커버리지 주장이 아니라 pip가 실제로 강제하는 설치 가능성 제약이기 때문이다.

**트레이드오프**: 3.10과 3.12는 실제로 클라이언트가 깨졌다는 근거가 없음에도 PyPI classifier 메타데이터에서 더 이상 나타나지 않는다. 3.10 또는 3.12에 특정된 회귀가 의심되면, 이는 매트릭스를 다시 넓힐 사유이지 이번 PR이 범위를 과소 커버했다는 증거는 아니다.

### 2.3 `test` 매트릭스에 `fail-fast: false` 적용

**선택 이유**: 이 설정이 없으면 GitHub Actions는 하나의 leg가 실패하는 즉시 나머지 매트릭스 leg를 취소하므로, floor 또는 ceiling에서만 나는 실패가 먼저 실패한 leg 뒤에 가려질 수 있다. 이는 이슈가 명시한 완료 기준 중 하나였다.

## 3. 구현 상세

### 3.1 워크플로 구조

**파일: `.github/workflows/python.yml`**

변경 전: `python-version: '3.11'`에서 네 스텝을 모두 실행하는 단일 `check` 잡.

변경 후:

```yaml
jobs:
  lint:
    name: lint, type-check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: '3.11'
      - run: ruff check python
      - run: ruff format --check python
      - run: mypy python/src

  test:
    name: test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.9', '3.11', '3.13']
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: ${{ matrix.python-version }}
      - run: pytest python/tests -m "not e2e" -q
```

두 잡 모두 기존 단일 잡과 동일한 방식으로 패키지를 설치하므로(`pip install -e "python[dev]"`), 실질적인 의존성 해석과 테스트 실행 방식은 그대로다. 인터프리터만 `test`의 각 leg에서 달라진다.

### 3.2 Classifier 축소

**파일: `python/pyproject.toml`**

```
- "Programming Language :: Python :: 3.9",
- "Programming Language :: Python :: 3.10",
- "Programming Language :: Python :: 3.11",
- "Programming Language :: Python :: 3.12",
- "Programming Language :: Python :: 3.13",
+ "Programming Language :: Python :: 3.9",
+ "Programming Language :: Python :: 3.11",
+ "Programming Language :: Python :: 3.13",
```

`requires-python`, 의존성, `python/src/` 하위 파일은 변경하지 않았다.

## 4. 학습 포인트

### 4.1 잡 이름 변경 후 GitHub Actions 필수 체크 명명

`check`를 `lint`/`test`로 이름을 바꾸면 GitHub이 리포트하는 체크 이름도 바뀐다(`Python / lint`, `Python / test (Python 3.9)` 등). 이 저장소의 `main` 브랜치 룰셋은 `gh api repos/lablup/mlxcel/rules/branches/main`으로 확인한 결과 `deletion`과 `non_fast_forward` 규칙만 가지고 있으며, 이전 잡 이름을 고정한 `required_status_checks` 규칙은 없었으므로 이번 이름 변경이 필수 체크를 고아로 만들지 않는다. 잡 이름으로 필수 체크를 고정해 둔 저장소라면 같은 변경 안에서 그 설정도 함께 갱신해야 한다.

## 5. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 2 |
| 추가된 라인 | +26 |
| 삭제된 라인 | -8 |
| 테스트 추가 | 0 (기존 스위트가 인터프리터 1개가 아닌 3개에서 실행됨) |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| CI | 1 | `.github/workflows/python.yml`을 `lint`(단일 leg)와 `test`(3.9/3.11/3.13 매트릭스, `fail-fast: false`)로 분리 |
| 메타데이터 | 1 | `python/pyproject.toml` classifier를 테스트 매트릭스에 맞춰 축소 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `8c6a94e` | test | test(ci): run Python client CI across a 3.9-3.13 version matrix |

## 6. 후속 조치

### 모니터링 필요

- 이 PR의 첫 CI 실행에서 3.9 또는 3.13에서만 발생하는 실패가 없는지 확인한다. 원본 이슈에 따르면 floor/ceiling에서 실제 비호환성이 발견되면 매트릭스에서 해당 버전을 빼는 대신 별도 버그로 등록해야 한다.

### 향후 개선 사항

- 3.10 또는 3.12에 특정된 회귀가 보고되면, 이번 PR의 3버전 선택을 영구적인 상한으로 취급하지 말고 매트릭스(그리고 classifier 목록)를 다시 넓힌다.

## 부록

### A. 테스트 결과

로컬 검증 결과(CI에 푸시하기 전, uv로 관리한 인터프리터 기준):

| 인터프리터 | `pip install -e "python[dev]"` | `pytest python/tests -m "not e2e" -q` |
|---|---|---|
| 3.9.6 (시스템 CPython) | 정상 해석 | 43 passed, 2 deselected |
| 3.11.10 (uv 관리) | 정상 해석 | 43 passed, 2 deselected |
| 3.13.5 (uv 관리) | 정상 해석 | 43 passed, 2 deselected |

3.11 환경에서 `ruff check`, `ruff format --check`, `mypy python/src`도 추가로 실행했으며 모두 이슈 없이 통과했다.
