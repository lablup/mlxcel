# PR #2469: 설치 문서의 feature 표에 선언된 Cargo feature를 모두 싣기

**작성일**: 2026-10-20
**상태**: 구현 완료, PR 열림
**위험도**: Low (문서와 manifest 주석 하나만 수정, 코드나 feature 의미 변경 없음)

## 요약

#1705를 닫는다. `docs/installation.md`의 Cargo feature 표는 `Cargo.toml`이 선언한 16개 feature 가운데 9개만 적고 있었다. 빠진 7개는 모두 사용자가 닿을 수 있는 것들이다. `webui`는 기본으로 켜져 있고, `voicechat-mic`는 예제 하나를 가리며, `xla-*` 진단 feature 다섯 개는 oracle과 reference-check 예제들을 가린다. 이 PR은 각 feature의 기존 Cargo.toml 주석을 압축해 행을 하나씩 추가하고, 기본 feature 집합을 여전히 `surgery` 하나로 설명하던 문장을 고치며, manifest에서 유일하게 설명 주석이 없던 `[[example]]` 블록(`xla_aux_smoke`)에 다른 블록들과 같은 주석을 붙인다.

## 문제 정의

설치 문서만 읽어서는 `webui`, `voicechat-mic`, `xla-reference-diagnostics`, `xla-diagnostics`, `xla-diagnostics-cpu`, `xla-diagnostics-rocm`, `xla-micro-oracle`이 존재한다는 것, 무엇을 끌어오는지, 어떤 예제와 테스트 타깃이 이를 필요로 하는지 알 수 없었다. 표 아래 본문은 `default = ["surgery"]`라고 적고 있었는데, `webui`가 기본 집합에 들어간 뒤(#1836)로는 틀린 말이다. 맨 `cargo build`가 WebUI 번들을 임베드하고 `--no-default-features`가 그것을 뺀다는 사실을 문서 어디에도 적지 않았다. #1705는 feature가 10개이던 시점에 세 행이 빠졌다고 보고했지만, 그 뒤 manifest가 커져서 이 변경이 들어갈 때는 일곱 행이 비어 있었다.

## 기술적 결정

- 행 내용은 새로 쓰지 않고 Cargo.toml 주석을 압축했다. 두 설명이 의미에서 어긋날 수 없게 하기 위해서다. 각 행은 feature의 구성(`xla-iree` + `mlxcel-xla/diagnostics`, `cuda` + `xla-diagnostics-cpu` 등)과 manifest의 `required-features` 항목에서 가져온 `[[example]]` 타깃을 적는다.
- 행 순서는 알파벳순이 아니라 manifest의 묶음을 따른다. `webui`는 `surgery` 옆(기본 feature 둘), `xla-*` 진단 행들은 모두가 바탕으로 삼는 `xla-iree` 뒤, `voicechat-mic`는 `test-utils` 뒤(둘 다 예제와 테스트 scaffolding)에 둔다.
- 도입 문장 "Shipping builds enable only the platform backend flags"는 `surgery`가 기본으로 켜진 뒤로 이미 사실이 아니었으므로, 기본 feature와 backend flag를 함께 켠다고 바꾸었다.
- `xla_aux_smoke` 주석은 예제의 `main`과 `mlxcel_xla::aux_smoke`(generic auxiliary ABI의 실제 런타임 수용 probe)에서 끌어냈고, 예제가 무엇을 하는지와 왜 `xla-iree`에서만 빌드되는지만 적는다. 앞에 주석이 없는 다른 두 예제 블록(`voicechat_microphone`, `webui_static_harness`)은 이슈 범위 밖이라 그대로 두었다. 해당 feature 주석이 설명을 담고 있다.

## 변경 사항

- `docs/installation.md`: feature 표에 행 7개 추가, 도입 문장과 기본 feature 단락 수정, `--no-default-features` 예시의 주석에 WebUI 자산 언급 추가.
- `Cargo.toml`: `[[example]] name = "xla_aux_smoke"` 위에 다섯 줄 주석. manifest 의미 변경 없음.

## 검증

- `cargo metadata --no-deps --format-version 1`이 수정된 manifest를 파싱한다.
- 일회성 스크립트로 `Cargo.toml`의 `[features]` 블록과 표를 대조했다. 기본값을 제외한 feature 16개, 표 16행, 빠진 것도 선언되지 않은 것도 없다.
- 표에 추가한 상대 링크(`webui/bundling.md`, `nemotron-voicechat.md`)는 모두 `docs/` 아래 실제 파일을 가리킨다.
- Rust 소스는 바뀌지 않았으므로 fmt, clippy, 테스트 게이트에는 영향이 없다.

## 후속 조치

- 이 표는 손으로 유지한다. `scripts/ci/check_kernel_dtype_keys.py`와 같은 모양으로 `[features]` 블록과 표를 대조하는 검사를 두면 재발을 막을 수 있다. 이번 이슈는 문서 전용이고 저장소의 문서 검사에 아직 그런 훅이 없어서 여기서는 추가하지 않았다.

## 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 2 |
| 추가된 라인 | 21 |
| 삭제된 라인 | 8 |
| 테스트 추가 | 0 |

## 관련

- Issue #1705: docs(install): the Cargo feature table omits three declared features.
- Issue #1836: `webui`를 기본 feature 집합에 넣은 WebUI epic.
- Issue #1376: `voicechat-mic` 뒤의 Nemotron VoiceChat 마이크 예제.
