# 기술 보고서: PR #3080 - TP·PP prefill에서 샘플링할 행만 projection

**작성일**: 2026-11-15
**상태**: 완료 (자체 stage 로직을 가진 pipeline 계열은 남음, 후속 조치 참고)
**위험도**: 중간

## 요약

tensor-parallel runtime과 pipeline-parallel의 마지막 stage는 prefill의 모든 위치를 LM head에 통과시킨 뒤 마지막 행만 잘라 썼다. 그래서 TP Qwen 3.5와 TP Gemma 4는 단일 프로세스에서 얻은 #3054와 Gemma 4 override의 절감 효과를 누리지 못했다. 이제 7개 TP runtime 모두 final norm, head, softcap보다 먼저 hidden state를 샘플링할 행에서 자른다. pipeline coordinator는 그 행의 위치를 마지막 stage에 넘기고, 공용 transformer stage model은 그 행만 projection한다.

## 1. 문제 정의

engine prefill은 `forward_last_logits*`를 호출하는데, trait 기본 구현은 `[B, L, vocab]` logits 전체를 계산한 다음 자른다. vocab이 248K이면 head를 가진 rank에서 2048 토큰 piece마다 약 1 GB, 28K 단일 패스에서는 약 14 GB가 든다(#3074).

## 3. 기술적 선택과 그 이유

- **TP:** 각 runtime의 forward는 last position을 선택적으로 받는 하나의 공용 내부 함수를 거친다. 일반, embedding, last-row 진입점이 모두 이 함수로 들어가므로 서로 어긋날 수 없다. 이 runtime들은 head를 rank 0에 복제해 두기 때문에 vocab 샤딩 gather는 고칠 것이 없었다.
- **PP:** `last_pos`는 `RunEntry`와 `ActivationMessage`의 `#[serde(default)]` 필드다. 구버전 peer는 이 필드를 무시하고 모든 위치를 돌려주며, coordinator가 여전히 행을 자르므로 버전이 섞인 pipeline도 동작한다. `StageExecutor::execute_last_row`의 기본 구현은 forward 전체를 돌린 뒤 logits 행을 자른다. `TransformerStageModel`은 이를 override해 head 앞에서 자르며, Llama, Mistral, Mixtral, Qwen3, Llama4, DeepSeekV3, GLM4의 마지막 stage가 여기에 해당한다.

## 7. 변경 요약

| 항목 | 값 |
|------|-------|
| 변경 파일 | 23개 (+1380 / -370) |
| TP 테스트 | `distributed::tensor_parallel::llama_runtime`: 43개 통과 (TP 크기 1과 2에서 새 last-row 테스트 7개 포함) |
| PP 테스트 | `distributed::pipeline`: 422개 통과 (새 stage-executor, worker, server-model 테스트 포함) |
| 실모델 측정 | 없음 (GPU 1대, 다중 노드 환경 없음) |

## 8. 후속 조치

- 마지막 stage가 `TransformerStageModel` 밖에서 도는 pipeline 계열(Gemma3, Gemma4, Qwen3.5, Qwen3-Next, GptOss, GlmMoeDsa, Jamba, NemotronH)은 여전히 모든 위치를 projection한다. 기본 hook은 coordinator로 돌려보내는 frame 크기만 줄인다.
