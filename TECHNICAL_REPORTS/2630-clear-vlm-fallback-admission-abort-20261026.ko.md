# PR #2630: 승인 중단 시 VLM fallback 상태 정리

**Date**: 2026-10-26
**Status**: Qwen3-VL은 Metal(M1 Ultra)에서 완료. 나머지 계열은 단위 테스트로만 검증
**Languages**: Rust
**Risk Level**: Medium

## 요약

PR #2630(이슈 #2625)은 서버 승인 단계의 요청 간 상태 누수를 막습니다. 미디어 준비는 요청에 sequence id가 생기기 전에 그 요청의 미디어 상태를 모델 단위 fallback 슬롯에 씁니다. 승인 단계는 이후 `bind_*_to_sequence` 호출로 그 상태를 id 아래로 옮깁니다. 두 단계 사이에 승인이 반환되는 경로가 셋 있었고, 어느 경로도 슬롯을 비우지 않았습니다. 그래서 다음 요청의 bind가 실패한 요청의 상태를 가져갔습니다. 텍스트 전용 요청은 이 슬롯을 덮어쓰지 않으므로, 다른 요청의 MRoPE delta, DeepStack 특징, `per_layer_inputs`, cross-attention 이미지, Falcon-OCR rope delta로 decode했습니다. 실제 Qwen3-VL 서버에서는 이미지 요청이 중단된 뒤 온 텍스트 전용 요청이 단독 실행 결과 대신 잘못된 토큰 하나를 냈고, 수정 후에는 단독 실행과 같습니다.

## 1. 문제 정의

`bind_fallback_to_sequence`는 fallback에 있는 것을 무조건 가져가고, `prepare_request_vlm_embeddings`의 텍스트 전용 분기는 fallback을 건드리지 않고 반환합니다. 따라서 미디어를 준비한 승인은 반드시 bind까지 도달해야 올바르게 동작합니다. `src/server/batch/scheduler/admission.rs`의 세 반환 경로가 이를 깼습니다.

- `--enable-vlm-prefix-cache`가 sequence id 없이 미리 준비를 마친 뒤 `allocate_sequence_state()`가 `Server busy`로 실패하는 경로.
- 미리 하는 준비가 중간에 실패하는 경로(예: 한 계열이 슬롯을 쓴 뒤의 취소). 기존 주석은 여기서 정리할 것이 없다고 적고 있었습니다.
- 할당 이후의 일반 준비가 실패하는 경로. `release_sequence_caches`는 sequence별 항목만 지우고 fallback은 지우지 않습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `LoadedModel` | `clear_vlm_fallback_state()`가 승인 단계가 sequence별로 bind하는 fallback 슬롯을 모두 지웁니다. 대상은 Qwen VL 런타임(MRoPE, 있으면 DeepStack), Qwen3-Omni, Gemma 4(두 런타임), Gemma 3n, Llama 3.2 Vision, Falcon-OCR입니다. sequence별 항목은 건드리지 않습니다. |
| 승인 단계 | 세 중단 경로 모두에서 이 정리를 호출합니다. |
| `QwenVlRuntime` | `clear_pending_media_state()`는 기본 구현이 없는 필수 메서드라서, 새 Qwen 계열 런타임이 조용히 빠뜨릴 수 없습니다. |
| 상태 컨테이너 | Gemma 4와 Gemma 3n의 `per_layer_inputs` 컨테이너, Falcon-OCR 상태에 정리 메서드를 추가했습니다. GLM-4V-MoE, Qwen3.5, Qwen4-Exp 텍스트 모델에는 `clear_mrope_state()`를 추가했습니다. |

## 3. 기술적 선택과 그 이유

### 3.1 매 승인마다 슬롯이 비워지는 계열만 정리

이 정리는 요청이 실행되는 동안 fallback이 비어 있어야 안전합니다. 승인 단계가 매 요청마다 슬롯을 bind해서 비우는 계열이면 이 조건이 성립합니다. 첫 구현은 PaddleOCR-VL, ERNIE-4.5-MoE-VL, Hunyuan-VL도 정리했습니다. 그런데 pr-reviewer가 승인 단계가 이 세 계열을 sequence별로 bind하지 않는다는 것을 찾았습니다. `bind_qwen_vl_mrope_state_to_sequence`는 `Qwen`과 `Qwen3OmniMoe` 런타임만 처리합니다. 이 세 계열의 실행 중 요청은 fallback 자체로 decode하므로, 다른 요청이 중단될 때 fallback을 지우면 실행 중인 이미지 요청의 position이 사라집니다. 그래서 정리 대상에서 뺐습니다.

### 3.2 다음 bind가 아니라 중단 시점에 정리

대안은 bind할 때마다 fallback이 bind하는 요청의 것인지 검사하는 방식이었습니다. 이 방식은 fallback에 요청 식별 정보가 필요한데 준비 단계에는 그런 정보가 없습니다. 또 중단된 요청의 텐서가 다음 승인까지 메모리에 남습니다. 중단 시점에 정리하면 불변식이 그것을 깨는 코드 옆에 있고, 텐서도 바로 해제됩니다.

## 4. 검증

- **실패 먼저 확인한 단위 테스트**: `src/server/batch/scheduler/vlm_fallback_abort_tests.rs`는 sequence 2개짜리 pool에서 실제 `enqueue_request`를 실행합니다. Qwen3-VL에서 중단 경로마다 테스트가 하나씩 있고, Llama 3.2 Vision의 할당 실패 테스트가 하나 더 있습니다. 승인 단계의 정리 호출을 되돌리면 네 테스트 모두 실패하고, 정리 호출이 있으면 모두 통과합니다. Gemma 컨테이너는 상태 수준 테스트 두 개로 확인합니다. 작은 테스트 모델에는 vision tower가 없어서 테스트가 fallback을 직접 채웁니다.
- **실제 체크포인트**: `models/qwen3-vl-4b-instruct-4bit`를 `mlxcel-server --parallel 2 --max-queue-depth 1 --enable-vlm-prefix-cache`로 띄웠습니다.
  - 구성: 긴 텍스트 요청 둘이 실행 행을 채우고, 이미지 요청 Y가 대기합니다. 같은 순간에 보낸 이미지 요청 X는 미리 준비를 마친 뒤 `Server busy`로 거절됩니다.
  - 결과: `main`에서는 다음 텍스트 전용 요청 T가 단독 실행 결과 대신 `"`를 반환했습니다. 이 PR에서는 T가 단독 실행과 같습니다.
  - Y가 이미지 요청이어야 하는 이유: 대기 중인 텍스트 전용 prefill이 Qwen3-VL fallback을 우연히 비워서 누수를 가립니다.
- **회귀 확인**: PR #2610의 동시 요청 대 단독 요청 비교가 여전히 단독 실행과 같습니다. 최신 `main`을 병합한 뒤 workspace clippy, fmt, scheduler 테스트 221개가 통과합니다.

## 5. 학습 포인트

- fallback과 bind를 나누는 구조는 fallback을 채우는 모든 경로가 bind에 도달해야 올바릅니다. 깨지는 곳은 중단 경로입니다. "정리할 것이 없다"는 주석이 틀린 이유는 정리 대상이 스케줄러 상태가 아니라 모델 상태에 있었기 때문입니다.
- 실제 재현은 대기열 타이밍에 달려 있었습니다. 대기 중인 텍스트 prefill이 슬롯을 비우는 경우와 HTTP 대기열 게이트가 준비 전에 X를 거절하는 경우, 두 우연이 버그를 가렸습니다. 재현 도구는 둘 다 피하도록 구성해야 했습니다.

## 6. 후속 조치와 제한 사항

- PaddleOCR-VL, ERNIE-4.5-MoE-VL, Hunyuan-VL은 sequence별로 bind되지 않아서, 중단이 없어도 요청들이 MRoPE fallback 하나를 공유합니다. 원래 있던 문제이고, PR 본문에 범위 밖이라고 적었습니다.
- Llama 3.2 Vision, Gemma 4 / 3n, Falcon-OCR은 실제 체크포인트로 실행하지 않았습니다.

## 참고 자료

- 이슈 #2625, 관련 #2567, PR #2610
- `src/loaded_model_capabilities.rs`, `src/server/batch/scheduler/admission.rs`, `src/multimodal/qwen_vl.rs`
