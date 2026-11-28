# 기술 보고서: PR #1404 - Python Repr 도구 호출 값 파싱

**작성일**: 2026-08-25
**작성자**: mlxcel 기여자
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1404는 서버의 Pythonic tool-call parser가 LFM2 계열 체크포인트가 출력하는 single-quoted string, Python boolean/null literal, 인용된 comma, 중첩 list 값을 처리하도록 수정한다. 또한 안전하지 않은 in-band comma sentinel을 직접 quote·bracket-aware scanner로 교체하고 parser 및 streaming 회귀 테스트를 추가했다.

---

## 1. 문제 정의

### 1.1 배경

Pythonic tool-call 형식은 호출을 `name(key=value, ...)`로 인코딩한다. 기존 parser는 JSON 형태의 double-quoted value를 가정했지만 일부 체크포인트는 `name(query='hello', enabled=True, extra=None)` 같은 Python `repr` 문법을 출력한다.

### 1.2 기존 문제점

- **Quote 불일치**: Single-quoted string의 바깥 quote가 제거되지 않고 값에 남았다.
- **Literal 불일치**: `True`, `False`, `None`이 JSON boolean 또는 null이 아니라 string으로 반환됐다.
- **잘못된 comma 경계**: 인용된 string이나 bracket list 안의 comma가 하나의 인자를 여러 조각으로 나눌 수 있었다.
- **안전하지 않은 중간 표현**: 초기 수정은 list comma를 private-use 문자로 치환했으며, 모델 출력에 해당 문자가 실제로 포함되면 데이터가 손상될 수 있었다.

### 1.3 위험성

| 위험 | 영향도 | 수정 전 가능성 |
|------|--------|----------------|
| Tool handler가 잘못된 JSON type의 인자를 수신 | High | 영향받는 체크포인트에서 High |
| 인용 text 또는 list 값이 잘리거나 거부됨 | High | Medium |
| 사용자 제어 private-use 데이터가 조용히 변조됨 | Medium | Low |

---

## 2. 변경 요약

### 2.1 Python Repr 값 변환

Value parser는 이제 짝이 맞는 single/double delimiter를 인식하고 프로토콜에 필요한 matching quote와 backslash만 제한적으로 unescape하며 Python boolean/null literal을 JSON 값으로 변환한다. 기존 JSON-compatible 입력 동작은 유지된다.

### 2.2 구조 기반 인자 분리

인자를 문자 단위로 scan하면서 현재 quote, escape 상태, bracket depth를 추적한다. Quote 밖이면서 bracket depth가 0일 때만 comma를 separator로 취급하므로 인용된 comma와 중첩 list 내용이 같은 `key=value` segment에 남는다.

### 2.3 Streaming 검증

Regression test는 chunk 사이에서 입력이 분리되는 경우를 포함해 marker-wrapped 호출을 stream filter 전체로 검증한다. Pythonic enter-only marker는 terminal tool-call framing이므로 파싱한 호출 뒤의 일반 text는 assistant content로 방출하지 않는다.

---

## 3. 기술적 선택과 그 이유

### 3.1 In-Band Sentinel 대신 직접 Scanner 사용

**결정:** 원본 입력 문자를 보존하고 separator를 구조적으로 탐지한다.

**근거:** Source text를 모델이 제어하므로 어떤 Unicode scalar도 안전한 내부 sentinel이 될 수 없다. 직접 scan은 collision에 의한 변조를 없애면서 list parsing에도 필요한 quote와 nesting 개념을 재사용한다.

**트레이드오프:** Scanner는 regex 치환보다 길지만 상태와 separator 규칙이 명시적이며 독립적으로 테스트할 수 있다.

### 3.2 지원하는 Python Subset 제한

**결정:** 범용 Python parser를 구현하지 않고 tool-call protocol에서 관찰된 repr 구성만 지원한다.

**근거:** Matching quote 제거, 제한된 escape, scalar literal, list nesting으로 체크포인트 출력을 처리하면서 비신뢰 입력 parsing을 결정적이고 dependency-free하게 유지한다.

**트레이드오프:** 전체 Python escape 의미론, 한 응답의 여러 호출, call-level parenthesis nesting, outer matcher 내부의 인용된 `)]`는 계속 지원하지 않는다.

---

## 4. 리뷰 및 품질 검토

### 4.1 구현 리뷰

Hardening 이후 correctness review에서 해결되지 않은 이슈는 없었다. 변환 우선순위, nesting·escape 상태, first-call-only 호환성, marker 처리, fragmented stream을 집중 검토했다.

### 4.2 보안 및 성능 리뷰

남은 CRITICAL 또는 HIGH finding은 없다. 리뷰에서 private-use sentinel collision을 MEDIUM data-integrity 이슈로 식별했으며, 최종 구현은 sentinel을 제거하고 정확한 보존 테스트를 추가했다. 기존 unmarked bare-Pythonic streaming 제한과 outer matcher의 인용된 `)]` 제한은 이 이슈 범위 밖에 남는다. Scanner는 입력 길이에 대해 선형이며 새 dependency나 무제한 secondary allocation을 추가하지 않는다.

### 4.3 호환성

- **Breaking change**: CLI 또는 HTTP request schema에는 없음.
- **새 의존성**: 없음.
- **동작 변경**: LFM2-style Python repr 값이 올바른 JSON type으로 tool handler에 전달된다.

---

## 5. 검증

- `cargo test --profile test-fast pythonic_ --lib`의 집중 테스트 31개가 통과했다.
- `cargo test --workspace --profile test-fast --features metal,accelerate` 전체 workspace gate가 통과했다.
- `cargo clippy --workspace --all-targets -- -D warnings`가 통과했다.
- `cargo fmt --all -- --check`와 `git diff --check`가 통과했다.
- 보고서 생성 전 CI formatting, lint, dependency policy, crate-version, cross-repository reference, OpenXLA compile, CLA check가 통과했다.
- `models/` 아래에 로컬 LFM2 체크포인트가 없어 실제 모델 검증은 차단됐다. 대신 marker-wrapped parser와 fragmented-stream 회귀 테스트로 영향받는 protocol boundary를 합성 검증했다.

---

## 6. 변경 통계

| 항목 | 값 |
|------|----|
| 변경 파일 | 3 |
| 추가 줄 | 243 |
| 삭제 줄 | 57 |
| 구현 커밋 | 2 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `fcf6dc1d5` | fix | Python repr tool-call 값 파싱 |
| `d0ef04b7b` | fix | Pythonic tool 인자 분리 hardening |

---

## 7. 후속 고려 사항

- LFM2 체크포인트를 로컬에서 사용할 수 있을 때 실제 `/v1/chat/completions` tool-call round trip을 검증한다.
- 실제 체크포인트가 인용된 `)]`, 중첩 call parenthesis, 또는 한 응답의 여러 Pythonic call을 출력하면 outer call recognizer를 확장한다.
- 일반 assistant text의 false positive를 늘리지 않으면서 stream filter가 unmarked bare Pythonic call을 탐지해야 하는지는 별도로 결정한다.

---

## 참고

- 이슈 #1306: Pythonic tool call의 Python repr 문법
- PR #1404: parser 변환, 구조 기반 분리, streaming 회귀 테스트
