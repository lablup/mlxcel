# PR #2449: QKV, VLM, DeepSeek의 NVFP4 global-scale 로드 누락 해소

**작성일**: 2026-10-19
**상태**: CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. PR은 열려 있음. "미해결 사항"의 항목은 계속 열려 있음

## 요약

#2444를 닫는다. #2441이 NVFP4 텐서 scale을 잘못 처리하는 로드 경로 3개를 남겼고, 이 PR이 세 가지를 모두 고친다. dense ModelOpt NVFP4의 fused QKV projection, VLM 로더와 Gemma 4 unified expert 분할, DeepSeek v1의 bias 없는 expert plane이다. `nvidia/Qwen3-30B-A3B-NVFP4`(ModelOpt)가 이제 로드되고 생성한다. `main`에서는 로드에 실패했다.

## 1. 문제 정의

- **fused QKV의 dense ModelOpt NVFP4.** `nvidia/Qwen3-30B-A3B-NVFP4`는 `main`에서 `1-bit weights are affine-only ... (prefix: model.layers.0.self_attn.q_proj)`로 로드에 실패했다. `FusedQKVLinear::from_weights_separate`가 `affine`을 하드코딩해서 bias 없는 nvfp4 plane이 1비트 affine 레이아웃으로 풀렸고, 이어 붙인 plane에는 각 plane의 `weight_scale_2`(repack된 `{plane}.global_scale`)를 담을 자리가 없었다. 또한 Qwen3와 Qwen3-MoE의 `group_size()`는 ModelOpt만 있는 설정에서 64로 폴백했다.
- **VLM 로더와 Gemma 4 unified expert 분할.** `apply_native_global_scale_flags`는 `load_text_weights`와 Gemma 4 sanitizer에서만 실행되었다. Gemma 4 unified 로더의 `split_fused_moe_key`는 fused `experts.gate_up_proj`의 `.weight` / `.scales` / `.biases`만 분할하므로 `.global_scale`이 고아가 된다.
- **DeepSeek v1의 bias 없는 expert plane.** `SwitchLinear::from_stacked_parts`가 `.biases` 없는 `.scales`를 `Regular`로 매핑해서, ModelOpt 또는 mlx-lm nvfp4 expert plane이 첫 forward의 `gather_mm`에서 중단되었다.

## 2. 기술적 선택과 그 이유

- **plane에 scale이 있으면 QKV를 분리한다.** `FusedQKVLinear::from_weights_separate`는 `UnifiedLinear::from_weights`처럼 `infer_quantization_mode`로 `q_proj`에서 모드를 추론한다. q/k/v 중 하나라도 `{plane}.global_scale` 사이드카가 있으면 레이어를 `QkvProjection::Split`로 로드한다. 각 plane은 `UnifiedLinear::from_weights_with_mode`가 정리된 레이아웃으로 만들고, matmul 뒤에 자신의 scalar를 적용하며(키 이름과 의미는 그대로, MLX의 native `global_scale`이 아니라 `inv_global_scale` 규약), 폭이 고정된 자신의 런타임 LoRA 항(#2438)을 직접 claim한다. 이슈는 fused 경로의 row별 scale 벡터를 배제한다. fused rope 커널이 scale을 받지 않기 때문이다.
- **레이어별 struct에 필드를 추가하지 않는다.** debug Pipeline CI 스모크 테스트가 2 MiB 스택 한계에 걸려 있으므로(#2434), 레이어별 struct에 필드를 추가하지 않는다. 같은 이유로 이슈는 `QuantizedWeight`, `UnifiedLinear`, `QkvProjection`, `FusedQKVLinear`에 필드를 추가하지 말라고 지정한다.
- **fused block-float plane은 group size를 정리한다.** fused 경로는 block-float plane의 group size를 shape에서 정리하고 저장하기 전에 백엔드를 확인한다. 이전에는 선언된 값을 그대로 저장했다(예: 32로 패킹된 mxfp4 plane에 64). (두 번째 커밋 `abd936f6`에서 추가.)
- **ModelOpt group size 폴백.** Qwen3와 Qwen3-MoE의 `group_size()`는 PhiMoE처럼 `modelopt_nvfp4_group_size`(16)로 폴백한다. 이로 인해 두 `ModelArgs`에 `#[serde(default)] quantization_config` 필드가 추가되고, struct 리터럴을 쓰는 테스트 fixture 3곳에 한 줄씩 추가된다.
- **VLM 로더의 flag 패스.** `finish_vlm_weights_common`과 `load_gemma4_unified`(sanitizer 이전의 raw 키)가 native global-scale flag 패스를 실행한다. 필터링된 VLM 로더도 `finish_vlm_weights_common`에서 끝난다. 이 경우 subset 변형은 weight와 scale이 모두 맵에 없는 flag된 plane을 건너뛰고, 로드된 flag plane에 scale이 없으면 여전히 실패한다.
- **fused expert 키 분할.** `split_fused_moe_key`가 `.native_global_scale`과 `.global_scale`을 처리한다. `[num_experts]` 벡터는 분할하지 않고 `gate_proj`와 `up_proj`에 복사하고 `down_proj` 아래로 이름을 바꾸므로, flag되지 않은 scale은 고아가 되는 대신 `switch_glu.gate_proj` prefix에서 실패한다.
- **DeepSeek v1의 반쪽 양자화 plane.** `SwitchLinear::from_stacked_parts`는 `.scales` / `.biases` 중 하나만 있는 plane을 `Regular`로 만들어 `gather_mm`에서 중단하는 대신, prefix를 밝히며 거부한다.

## 3. 변경 내용

- `src/lib/mlxcel-core/src/layers.rs`: 모드 추론, scale이 있는 plane의 `Split` 경로, fused 경로의 group size 정리, 테스트(223줄 변경).
- `src/models/qwen3.rs`, `src/models/qwen3_moe.rs`: `modelopt_nvfp4_group_size` 폴백과 새 `quantization_config` 필드. 이에 맞춰 `qwen3_tests.rs`, `qwen3_moe_tests.rs`, `qwen3_embedding_tests.rs`, `src/rerank/qwen3_generative_tests.rs`, tensor-parallel `llama_runtime_tests.rs`의 fixture를 갱신.
- `src/loading/vlm.rs`, `src/loading/vlm_gemma_unified.rs`, `src/models/switch_layers.rs`, `src/models/sanitize.rs`: VLM 로더의 flag 패스, subset 변형, `split_fused_moe_key` 확장.
- `src/models/deepseek.rs`: 반쪽 양자화 plane 거부. `src/models/jamba.rs`: 작은 시그니처 갱신.
- 테스트: `src/loading/vlm_gemma_unified_tests.rs`, `src/loading/vlm_tests.rs`, `src/models/deepseek_tests.rs`.

## 4. 검증

GB10, CUDA, 모든 GPU 실행은 `gpu-lock` 아래에서 수행.

- `MLXCEL_SDPA_DETERMINISTIC=1 mlxcel generate -m ~/models_archive/hf-src/Qwen3-30B-A3B-NVFP4 -p "The capital of France is" -n 80 --temp 0 --show-reasoning`가 로드되고(main은 로드 실패) 일관된 텍스트를 생성한다. "... Since the question is about the capital of France, the answer is Paris. ...". mlx-lm 0.32.2가 이 체크포인트를 거부하므로(예상치 못한 파라미터 55968개, stack되지 않은 ModelOpt 키) mlx-lm teacher-forced trace는 없다.
- `~/models_archive/mlx/laguna-xs-2.1-nvfp4`(`--show-reasoning` 포함), `olmoe-1b-7b-0924-nvfp4-gs`, `gemma-4-31b-it-nvfp4`(프롬프트 2개)의 greedy 출력이 main과 토큰 단위로 동일하다(`MLXCEL_SDPA_DETERMINISTIC=1`, 같은 프롬프트). 최종 커밋에서 다시 확인했다.
- 새 테스트 10개. 각각 수정을 되돌리면 실패하고 복원하면 통과한다.
  - layers: plane별 scale이 있는 nvfp4 분리 QKV와 `x @ dequantize(w).T * s`에 대한 forward 검증, group 32의 mxfp4 QKV fused.
  - qwen3 / qwen3_moe: ModelOpt만 있을 때 `group_size() == 16`.
  - Gemma 4 Unified: flag된 fused experts와 flag되지 않은 fused experts.
  - VLM: 체크포인트가 제공한 `.native_global_scale` 거부, flag된 scale 해석, 필터링된 로드가 필터 밖 plane을 건너뜀.
  - DeepSeek: 반쪽 양자화 plane.
- 다음 필터가 통과한다: `layers::`, `runtime_lora`, `one_bit`(mlxcel-core)와 `qwen3 vlm deepseek:: switch_layers sanitize`(mlxcel). 두 crate 모두 clippy `--lib --tests -D warnings`가 깨끗하다. fmt와 다섯 가지 contract 검사가 통과한다.

## 5. 미해결 사항

둘 다 기존부터 있던 문제이다.

- Kimi K3 VLM 로더는 `finish_vlm_weights_common`을 우회하므로, flag된 export는 `SwitchLinear`에서 깔끔하게 거부된다.
- dense `.global_scale`은 shape 검증을 하지 않으며, 이는 main의 `o_proj`와 MLP와 같다.

이슈는 O(planes x keys) 사이드카 스캔 비용도 범위 밖으로 명시한다.

## 6. 변경 요약

| 항목 | 값 |
|------|----|
| 변경된 파일 | 17 |
| 추가된 줄 | 672 |
| 삭제된 줄 | 24 |

| 커밋 | 메시지 |
|------|--------|
| `f1a74b25` | fix(nvfp4): close global-scale load gaps in QKV, VLM and DeepSeek |
| `abd936f6` | fix(nvfp4): guard fused block-float QKV and filtered VLM flag pass |
