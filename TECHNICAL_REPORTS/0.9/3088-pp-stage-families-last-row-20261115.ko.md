# 기술 보고서: PR #3088 - 나머지 마지막 stage에서 head 앞에서 샘플링 행을 자름

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 낮음

## 요약

#3080에서 pipeline coordinator가 샘플링할 행의 위치를 마지막 stage에 보내도록 했고, `TransformerStageModel`이 LM head 앞에서 그 행을 자르게 했다. 그러나 마지막 stage를 이 타입 밖에서 돌리는 8개 계열은 여전히 모든 위치를 projection한 뒤에야 행을 잘랐다. Gemma3, Gemma4, Qwen3.5, Qwen3-Next, GptOss, GlmMoeDsa(DeepSeekV32), Jamba, NemotronH다. 이제 각 계열이 `execute_last_row`를 override해, final norm, head, softcap보다 먼저 hidden state를 샘플링할 행에서 자른다(#3081).

## 1. 문제 정의

`StageExecutor::execute_last_row`의 기본 구현은 forward 전체를 돌린 뒤 logits 행을 자른다. coordinator로 돌려보내는 frame은 줄지만 head의 `[B, L, vocab]` 임시 텐서는 그대로다. vocab이 248K~262K이면 이 텐서는 2048 토큰 piece마다 약 1 GB다.

## 3. 기술적 선택과 그 이유

- **Gemma3, Gemma4, Qwen3.5, Qwen3-Next, GptOss, DeepSeekV32:** 각 stage model이 선택적 `last_pos`를 받아 `logits_at_position`으로 자른다. 따라서 Gemma4의 final softcap도 행 하나에만 적용된다.
- **Jamba, NemotronH:** `forward_stage_last_row`가 새로 생기고 `forward_stage`는 이를 감싸는 한 줄짜리 함수가 된다. in-process forward와 `LanguageModel` hook은 건드리지 않는다.
- **마지막이 아닌 stage:** `last_pos`를 무시하고, `None`이면 모든 위치를 그대로 둔다. 따라서 구버전 peer와 역방향 경로의 동작은 바뀌지 않는다.

## 7. 변경 요약

| 항목 | 값 |
|------|-------|
| 변경 파일 | 27개 (+852 / -75) |
| 새 테스트 | 계열마다 stage-executor 테스트 1개 (합성 가중치로 세 위치를 전체 forward의 행과 비교하고, 마지막이 아닌 stage가 hidden state 전체를 유지하는지 확인) |
| 테스트 묶음 | `distributed::pipeline`과 Gemma stage-parity 테스트: 441개 통과 |
| 실모델 측정 | 없음 (다중 노드 환경 없음) |

## 8. 후속 조치

- override와 기본 구현이 같은 행을 내므로, 테스트는 메모리 경로가 아니라 출력 행만 확인한다. NemotronH fixture에는 Mamba 층이 없다. 실제 checkpoint에서의 메모리 절감은 측정하지 않았다.
