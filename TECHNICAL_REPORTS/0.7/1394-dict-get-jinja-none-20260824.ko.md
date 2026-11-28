# 기술 보고서: PR #1394 - 누락된 템플릿 맵 키에 None 반환

**작성일**: 2026-08-24

**작성자**: 신정규

**상태**: 완료

**언어**: Rust, Jinja

**위험도**: Medium

## 요약

PR #1394는 mlxcel의 공용 Python 스타일 `dict.get()` 호환 shim을 CPython Jinja2 의미론에 맞췄다. 명시적 기본값 없이 누락된 키를 조회하면 이제 `Undefined`가 아니라 Minijinja `None`을 반환하며, Muse Glimmer 다중 턴 프롬프트의 완료된 assistant 메시지가 연속 생성 표식 `<|eom|>` 대신 올바른 `<|eot|>`로 끝난다. 명시적 기본값, falsy 값, `or` 체인 동작은 그대로 보존된다.

## 1. 문제 정의

Muse Glimmer 체크포인트 템플릿은 `message.get('end_turn') is none`으로 턴 종료 표식을 추론할지 결정한다. mlxcel은 누락된 키에 `Undefined`를 반환했기 때문에 `is none` 분기가 실행되지 않았고, 모든 일반 assistant 메시지가 끝나지 않은 턴으로 렌더링됐다. 프롬프트는 문법적으로 유효했지만 체크포인트 학습 및 검증에 사용된 바이트열과 달라졌다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/server/chat_template.rs` | 인자 하나인 `dict.get()`에서 키가 없으면 `Value::from(())`을 반환하고 Python 호환 의미론을 문서화 |
| 템플릿 호환성 테스트 | 누락된 키가 `none`이면서 `defined`임을 검증하고, 존재하는 JSON `null`, `false`, 0, 빈 문자열, 명시적 기본값, fallback 체인을 보존 |
| Muse Glimmer fixture 테스트 | `<|eot|>` 종료 표식을 이름으로 검증하고 영향받은 두 렌더 digest를 Jinja 기준값으로 갱신 |

## 3. 기술적 선택과 그 이유

### 3.1 공용 호환성 경계에서 수정

수정은 typed/raw `ChatTemplateProcessor` 렌더 경로가 모두 설치하는 단일 callback인 `configure_environment`에 적용됐다. 템플릿 전용 우회는 다른 체크포인트에 같은 CPython 호환성 결함을 남기고 chat completions, Responses, Anthropic 변환, router, offline CLI 사이에 서로 다른 동작을 만들 수 있다.

### 3.2 Falsy 의미론을 명시적으로 보존

Minijinja의 `None`과 `Undefined`는 모두 falsy이므로 기존 `m.get('a') or m.get('b')` 템플릿은 계속 동작한다. 리뷰에서는 존재하는 `null`, `false`, `0`, `""`를 raw JSON 경로로 직접 검증해 향후 리팩터링이 누락된 키와 존재하는 falsy 값을 혼동하지 못하게 했다.

### 3.3 Hash와 사용자-visible 동작을 함께 고정

Digest 검증은 바이트 동일성을 증명하지만 회귀 후 기계적으로 다시 고정될 수 있다. Muse 테스트는 이제 완료된 assistant 내용이 `<|eot|>`를 포함하고 `<|eom|>`를 포함하지 않는다는 의미론적 불변식도 직접 검증한다.

## 4. 검증

- `cargo fmt --all -- --check`: 통과.
- `cargo test -p mlxcel --profile test-fast --lib test_dict_get_method`: 6개 통과.
- `cargo test -p mlxcel --profile test-fast --lib muse_glimmer_template_`: 6개 통과.
- 로컬 Jinja2 기준 렌더: `multi_turn`은 325바이트, SHA-256 `433d37ff14caf2f2b177d904726b34ff09cb2aad4426a237a7b27772eab47007`; `tools_and_results`는 2340바이트, SHA-256 `dc451d3030d24f37ecc20fc0236c0b5fa7f70032d8c5331f8f6690689620d6ae`와 일치.
- Hosted CI: formatting, Clippy, cargo-deny, OpenXLA feature compile, metadata 검사, CLA 통과. 변경과 무관한 MLX pin extraction과 OpenXLA link job은 skip.
- 구현, 보안/성능, finalization 리뷰는 raw-value 커버리지 추가 후 미해결 정확성 또는 보안 결함을 발견하지 못했다.

## 5. 남은 검증 경계

이 Linux 호스트에는 사용 가능한 NVIDIA driver와 Metal backend가 없어 실제 Muse Glimmer 다중 턴 GPU smoke test를 실행하지 못했다. 결정적 렌더 테스트와 Jinja 기준 비교는 수정된 프롬프트 바이트를 증명하지만 real-checkpoint 생성 증거로 주장하지 않는다.

## 6. 관련 작업

- Issue #1383: Python 스타일 `dict.get()`이 누락된 키에 `Undefined`를 반환하던 결함.
- PR #1394: 이 보고서가 다루는 구현 및 리뷰 수정.
- PR #1382 / issue #1309: 기존 종료 표식 결함을 발견한 인접 `tojson` 호환성 작업이며 이 PR의 변경 범위에는 포함되지 않음.
