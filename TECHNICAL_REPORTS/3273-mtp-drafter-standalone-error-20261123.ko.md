# 기술 보고서: PR #3273 - `-m`에 전달된 MTP 드래프터를 "unsupported" 대신 정확히 안내

**작성일**: 2026-11-23
**상태**: 열림 (머지 전 작성)
**위험도**: 낮음 (이미 감지에 실패하던 config의 오류 메시지만 바뀜)

## 요약

`mlxcel-server -m mlx-community/gemma-4-12B-it-assistant`가 `Unsupported model type: gemma4_unified_assistant`로 종료되었다. 이 체크포인트는 포팅되지 않은 패밀리가 아니라 Gemma 4 12B용 MTP 투기적 디코딩 드래프터이며, 타깃 모델과 함께 `--draft-model`로 넘기면 이미 지원된다. 이 PR은 모델 감지가 `mlxcel-core` 드래프터 레지스트리에 등록된 MTP 드래프터 `model_type`을 알아보고, 해당 체크포인트가 드래프터임을 밝히며 `--draft-model`을 안내하는 오류를 반환하도록 한다. 새로 지원되거나 지원이 빠지는 모델은 없다.

## 1. 문제 정의

MTP 드래프터(`gemma4_assistant`, `gemma4_unified_assistant`, `qwen3_5_mtp`, `glm4_moe_lite_mtp`, `qwen4_exp_mtp`)는 타깃 모델의 은닉 상태로 초안 토큰을 예측하며 단독 생성 경로가 없다. 드래프터 로더는 `get_model_type`이 아니라 `peek_drafter_model_type`으로 `model_type`을 읽는다. 이런 체크포인트를 `-m`에 넘기면 일반 감지기로 들어가는데, 감지기에 해당 갈래가 없어서 mlxcel이 전혀 모르는 아키텍처와 같은 메시지가 나왔다. 그 메시지는 포팅 요청을 하라는 뜻으로 읽히지만, 실제 해결책은 다른 플래그를 쓰는 것이다.

DFlash 계열 드래프터는 평범한 `model_type: qwen3`을 선언해 잘못 라우팅될 수 있으므로 이미 전용 거부(`dflash_drafter_not_standalone_error`, #1168)가 있었다. MTP 드래프터에는 그에 해당하는 처리가 없었다.

## 2. 변경 요약

- `src/models/detection.rs`: `Unsupported model type` 기본 갈래 바로 앞에 `_ if is_mtp_drafter_model_type(&model_type)` 가드 갈래를 추가했다. `is_mtp_drafter_model_type`은 소문자로 정규화한 `model_type`을 `mlxcel_core::drafter::drafter_kind_by_model_type()`에서 찾아 `DrafterKind::Mtp`일 때만 참을 반환하고, `mtp_drafter_not_standalone_error`가 안내 메시지를 만든다.
- `src/models/detection_tests.rs`: `gemma4_unified_assistant`, `gemma4_assistant`, `qwen3_5_mtp`, 대소문자가 섞인 `Gemma4_Unified_Assistant`에 대해 새 문구, `--draft-model` 안내, 일반 메시지 부재를 확인하는 테스트 하나를 추가했다.

## 3. 기술적 선택과 그 이유

**앞단 검사가 아니라 기본 갈래 직전에 둔다.** DFlash 거부는 해당 config가 `qwen3` 갈래에 걸리기 때문에 분기 전에 실행된다. MTP `model_type`은 현재 어떤 갈래에도 걸리지 않으므로, 일반 오류가 나던 자리에서만 검사하면 지금 로드되는 config의 라우팅이 바뀌지 않음이 보장된다. 나중에 드래프터 `model_type`이 타깃 갈래와 겹치더라도 마찬가지다.

**별도 목록 대신 드래프터 레지스트리를 재사용한다.** `drafter_kind_by_model_type()`은 `--draft-model` 자동 감지가 이미 쓰는 표이므로, 여기에 새 MTP 드래프터를 등록하면 감지 코드를 고치지 않아도 올바른 오류가 나온다. `DrafterKind::Mtp`만 받으므로 `Dflash`로 등록된 Muse Glimmer assistant는 더 앞에서 구조적으로 잡는 기존 DFlash 거부가 계속 처리한다.

**자동 페어링 없이 오류만 낸다.** 드래프터 디렉터리에서 타깃을 자동으로 불러오려면 드래프터 config가 안정적으로 담고 있지 않은 타깃 경로가 필요하다. 대신 어떤 플래그에 어떤 체크포인트를 넘겨야 하는지 메시지로 알려 준다.

## 4. 검증

- 새 감지 테스트가 통과했고, main 리베이스 후 감지 테스트 278개, `-D warnings` 워크스페이스 clippy, fmt가 로컬에서 통과했다. PR CI는 19개 통과, 14개 건너뜀이다.
- 공개된 `mlx-community/gemma-4-12B-it-assistant-bf16`의 `config.json`(`model_type: gemma4_unified_assistant`, `Gemma4UnifiedAssistantForCausalLM`)으로 릴리스 `mlxcel-server -m`을 실행하면 새 메시지가 출력된다.
- 테스트 외 코드 중 기존 `Unsupported model type` 문자열에 의존하는 곳이 없으므로, WebUI 카탈로그 같은 호출자는 드래프터 디렉터리에서 오류 문구만 달라진다.

## 5. 한계

- 검증 호스트에 Gemma 4 가중치가 없어 `-m <gemma-4-12B-it> --draft-model <assistant>` 전체 페어링은 실행하지 않았다. 이 변경은 그 경로를 건드리지 않는다.
- 드래프터가 Qwen이나 GLM 헤드여도 메시지 예시는 Gemma 4를 든다. 문구가 조건부("for a Gemma 4 assistant")이긴 하지만 해당 유형에서는 덜 구체적이다.
