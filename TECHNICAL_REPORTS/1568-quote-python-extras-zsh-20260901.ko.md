# 기술 보고서: PR #1568 - zsh용 Python extras 설치 명령 인용부호 추가

**작성일**: 2026-09-01
**상태**: 완료
**언어**: Markdown
**위험도**: Low

## 요약

PR #1568은 Python client 문서에 나온 `./python[dev]` extras 인자를 모두 인용부호로 감싸서, 복사-붙여넣기한 명령이 zsh에서도 그대로 동작하게 만든다. 변경 자체는 작지만, macOS 기본 셸인 zsh에서 대괄호를 glob 패턴으로 해석해 온보딩이 바로 깨지던 문제를 없앤다.

## 1. 문제 정의

저장소는 Python 개발용 설치 방법을 `python/README.md`와 `docs/python-client.md` 두 곳에 적고 있다. 이 중 세 개의 명령이 `pip install ./python[dev]` 또는 `pip install -e ./python[dev]`처럼 인용부호 없이 적혀 있었다. zsh에서는 `[dev]`가 package extras 문법의 일부가 아니라 glob 문자 클래스처럼 먼저 해석된다.

그 결과 `pip`가 실행되기 전에 셸 단계에서 명령이 실패한다. 문제는 패키지 메타데이터나 Python client 구현이 아니라, 사용자가 그대로 복사할 것으로 기대되는 문서 예시에만 있다.

## 2. 기술적 선택과 그 이유

### 2.1 예시를 다른 형식으로 바꾸지 않고 extras 인자만 인용한다

문서의 설치 형태는 그대로 두고, zsh에서 꼭 필요한 셸 인용만 추가했다. 이 저장소의 CI가 이미 `pip install -e "python[dev]"` 형태를 쓰고 있으므로, 더 긴 설명이나 다른 명령 형식을 도입하지 않고도 문제를 가장 직접적으로 고칠 수 있다.

### 2.2 중복된 세 예시를 한 PR에서 함께 수정한다

같은 실패 형태가 사용자 문서와 패키지 로컬 README 양쪽에 있었고, 테스트 섹션의 editable install까지 포함되어 있었다. 세 줄을 한 번에 고쳐야 한 문서만 낡은 상태로 남아 같은 혼란을 다시 만들지 않는다.

## 3. 변경 요약

| 영역 | 변경 내용 |
|---|---|
| `python/README.md` | 설치 섹션의 development install 명령과 테스트 섹션의 editable development install 명령에 인용부호를 추가. |
| `docs/python-client.md` | Python client 가이드의 대응하는 development install 명령에 인용부호를 추가. |

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 2 |
| 추가된 라인 | +3 |
| 삭제된 라인 | -3 |
| 테스트 추가 | 0 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `5fdc3eb` | docs | docs: quote Python extras installs |

## 4. 검증

이 PR은 문서만 바꾸므로 검증도 의도적으로 좁게 유지했다.

- `rg -n 'pip install (\./python\[dev\]|-e \./python\[dev\])' python/README.md docs/python-client.md`가 일치 항목을 내지 않아, 깨진 비인용 형식이 모두 제거됐음을 확인했다.
- `rg -n 'pip install ("\./python\[dev\]"|-e "\./python\[dev\]")' python/README.md docs/python-client.md`가 의도한 세 명령을 모두 찾아, 문서가 인용된 형태를 일관되게 보여 줌을 확인했다.

## 5. 후속 조치

- [ ] 앞으로 다른 문서에 Python extras 예시를 추가할 때도 셸별 회귀를 막기 위해 extras 인자를 인용된 형태로 유지한다.
- [ ] 설치 가이드를 나중에 통합한다면, 패키지 README와 메인 문서가 기계적으로 공유하거나 동기화할 수 있는 단일 출처를 고려한다.

## 6. 관련 작업

- Issue #1222: zsh globbing 실패와 영향받는 세 줄을 정리한 이슈.
- PR #1568: 실제 수정 PR이며 이 이슈를 닫는다.
