# 기술 보고서: PR #3243 - IFM K2-Horizon(`k2_horizon`) dense 지원

**날짜**: 2026-11-23
**상태**: 열림 (draft PR, 머지 전에 작성)
**위험도**: 낮음 (공유 Llama decoder 위의 새 모델 타입. 공유 코드 변경은 다른 계열이 읽지 않는 `RopeScalingSpec`의 선택적 `attention_factor` 필드와 `yarn_get_mscale`의 `pub(crate)` 전환뿐이다)

## 요약

`mlxcel`은 `IFM/K2-Horizon-0.9B`에서 `Unsupported model type: k2_horizon`으로 종료했다(#2358). 0.9B는 YaRN RoPE를 쓰는 평범한 Llama 형태 decoder(GQA, SwiGLU, pre-norm RMSNorm, untied head, Llama weight key)이므로 이 PR은 새 decoder를 추가하지 않는다. `src/models/k2_horizon.rs`가 config를 변환하고 검사한 뒤 `llama3::ModelArgs`를 공유 `Llama3Model`에 넘기며, config-backed text 계열로 등록되어 CLI, `mlxcel-server`, adapter 경로, `mlxcel arch`가 모두 이 경로를 쓴다. 공유 decoder에 없는 K2-Horizon 기능(grouped RMSNorm, QK-norm, MoE와 MoVA, gated attention, sliding window, partial rotary, `silu`가 아닌 activation, 별도 YaRN magnitude)은 모두 로드 시 필드 이름을 적은 오류로 거부한다. 로더는 chat template의 turn terminator `<|ifm|im_end|>`도 EOS id에 추가한다. checkpoint가 `<|endoftext|>`만 선언하기 때문이다. 실제 bf16 0.9B에서 mlx-vlm `k2_horizon` 대비 24576 위치 teacher-forced trace는 decided 불일치 0개, perplexity는 기준보다 0.31% 낮다. `rope_scaling`을 빼면 decided 불일치 9.62%, perplexity +89.1%가 된다.

## 1. 문제

K2-Horizon은 하나의 형태가 아니라 계열이다. HF `modeling_k2_horizon.py`는 grouped RMSNorm(3.7B / 7B / 32B에서 `layernorm_num_groups` 2 / 4 / 4), head별 Q/K RMSNorm, MoE와 MoVA(mixture of value attention) expert, gated attention 출력, sliding window, `rope_head_dim`을 통한 partial rotary를 지원한다. 이 중 아무것도 쓰지 않는 것은 0.9B뿐이다. `k2_horizon`을 단순히 Llama 분기로 보냈다면 모든 변형이 로드되었을 것이다. 큰 dense checkpoint는 잘못된 group 수로 정규화하고, QK-norm checkpoint는 `q_norm` / `k_norm` weight를 조용히 무시한다. 둘 다 오류 없이 틀린 결과를 낸다. 그래서 작업의 핵심은 공유 decoder가 정확히 계산하는 부분집합만 받아들이고 나머지는 이름을 밝혀 거부하는 것이었다.

0.9B 자체에도 작은 문제가 두 가지 있었다. RoPE가 transformers 5의 `rope_parameters` 블록에 적혀 있어, Llama 경로가 읽는 flat `rope_theta` / `rope_scaling` 키에서는 보이지 않는다. 그리고 `config.json`, `generation_config.json`, `tokenizer_config.json`이 모두 `<|endoftext|>`만 EOS로 선언하는데, chat template이 렌더링하는 모든 assistant turn은 `<|ifm|im_end|>`로 끝난다. 그래서 서버는 모델이 EOS를 내기 전에 그 marker를 `content`에 디코딩했다.

## 2. 변경 내용

PR은 커밋 네 개로 이루어진다. 계열 추가(`e782a982`), turn terminator(`4adb83c6`), 생략된 검사 키에 HF 기본값 적용(`c1b62d05`), geometry와 activation 검사 강화(`c7a28def`).

- `src/models/k2_horizon.rs`(신규): `K2HorizonConfig`, `validate()`, `to_llama3_args()`, 디렉터리 `load`, adapter용 `from_weights`, turn terminator 조회. 구조체는 `#[serde(remote = "Self")]`와 `mlxcel_core::impl_rope_lifting_deserialize!`를 써서 필드 역직렬화 전에 `rope_parameters` 블록을 `rope_theta` / `rope_scaling`으로 복사한다. 0.9B는 base 1e6에서 YaRN factor 16(beta_fast 128, beta_slow 4, original context 8192), 최상위 `rope_theta`가 null인 7B는 scaling 없이 base 1e7이다.
- 검사(`validate`): `layernorm_num_groups != 1`, `query_key_norm`, 0보다 큰 `num_experts` 또는 `mova_num_experts`, null이 아닌 `attention_gate_func`, `use_sliding_window`(null은 false로 읽음), `silu`가 아닌 `hidden_act`, `rope_head_dim != head_dim`, `0.1 * ln(factor) + 1`과 다른 YaRN `attention_factor`, 유한하지 않거나 양수가 아닌 `rope_theta` / `rms_norm_eps`, 잘못된 quantization 파라미터. geometry 검사는 0 크기, KV head로 나누어지지 않는 head 수, 홀수이거나 너무 큰 `head_dim`, 고정 상한을 넘는 값, decoder의 i32 cast를 넘치게 하는 `heads * head_dim`과 `kv_heads * head_dim`을 거부한다.
- `src/models/rope_utils.rs`: `RopeScalingSpec`이 선택적 `attention_factor`를 읽지만 `yarn_rope_setup`은 이를 쓰지 않는다. K2-Horizon 검사가 YaRN 경로와 같은 함수를 호출하도록 `yarn_get_mscale`을 `pub(crate)`로 바꿨다.
- 등록: `models/mod.rs`(enum, `ALL_MODEL_TYPES`, 설명), `detection.rs`, `registry.rs`, `model_metadata.rs`(config-backed, `wrap: LoadedModel::Llama`), `execution/memory_estimate.rs`, `distributed/tensor_parallel/inference.rs`의 `ModelType::K2Horizon`. TP의 fallback architecture는 `"llama"`가 아니라 `"k2_horizon"`이다.
- 테스트: `src/models/k2_horizon_tests.rs`(14개: lift를 거친 두 Hub config, 모든 거부, 생략 키의 HF 기본값, attention factor 검사, terminator 조회와 병합, 있을 때 실제 checkpoint tokenizer), detection 테스트 하나, `rope_parameters_lift_tests.rs`의 registry 항목.
- 문서: `CHANGELOG.md`와 `docs/supported-models.md`. 거부되는 변형과 `<ifm|think>` 한계를 적었다.

## 3. 기술적 결정

**새 decoder가 아니라 `Llama3Model` 위의 config 변환.** 0.9B는 Llama layout과 key 단위로 일치하므로, 새 모델 파일은 동작 차이 없이 decoder를 복제할 뿐이다. Helium과 TeleChat3의 방식을 따른다. `to_llama3_args`는 전체 struct literal이라 `llama3::ModelArgs`에 필드가 추가되면 조용히 기본값이 들어가는 대신 이 파일이 컴파일에 실패한다. 0.9B의 `32 * 64 = 2048`이 `hidden_size` 1536과 다르므로 `head_dim`은 명시적으로 넘긴다.

**생략된 검사 키는 HF config class의 기본값을 따른다.** 첫 커밋은 없는 `query_key_norm`을 false로, 없는 `num_experts`를 0으로 두어 불완전한 config를 dense로 읽었다. `configuration_k2_horizon.py`의 기본값은 true와 128이므로, 그런 config는 실제로 QK-norm 또는 MoE 모델을 뜻한다. `c1b62d05`가 serde 기본값을 HF와 맞춰, 키를 빼도 선언한 것과 똑같이 거부된다. 공개된 0.9B는 두 키를 모두 선언하므로 영향이 없다. `head_dim`도 같은 규칙이다(`hidden_size / heads`가 아니라 HF 기본값 128). 명시적 `null`이면 여전히 나눗셈으로 계산한다.

**`attention_factor`는 적용하지 않고 검사한다.** 공유 YaRN 경로는 attention magnitude를 `yarn_get_mscale(factor, 1) / yarn_get_mscale(factor, 0)`, 즉 `0.1 * ln(factor) + 1`로 계산하며, factor 16에서 1.27726이다. 이는 0.9B가 선언한 값과 같고, floor/ceil 보정 범위도 이미 `truncate: true`와 일치한다. 그래서 YaRN 코드는 바꾸지 않았고, 다른 값이 선언되면(상대 허용 오차 1e-6) 절반만 반영하지 않고 거부한다. `attention_factor`를 `yarn_rope_setup`까지 전달하는 대안은, 공개 checkpoint 어디에도 쓰이지 않는 값을 위해 모든 YaRN 계열이 공유하는 경로를 바꾸는 일이다.

**TP fallback 이름을 `"llama"`로 두지 않는다.** tensor-parallel runtime은 `config.json`을 바로 `llama3::ModelArgs`로 파싱한다. `k2_horizon`을 거기로 보내면 `validate()`를 건너뛰고 grouped-norm이나 MoE checkpoint를 shard하게 된다. 고유 이름을 쓰면 `runtime_kind_for`에 해당 분기가 없어 TP가 거부된다.

**terminator는 id가 아니라 이름으로 찾는다.** `<|ifm|im_end|>`를 `tokenizer.json`의 `added_tokens`에서, 없으면 `tokenizer_config.json`의 `added_tokens_decoder`에서 찾고, 발견되었고 아직 없을 때만 선언된 EOS id 뒤에 붙인다. InternVL과 LLM-jp-VL 로더를 따르며 다른 계열의 공유 EOS 처리는 건드리지 않는다. id를 하드코딩하면 tokenizer를 다시 만든 checkpoint에서 깨진다.

## 4. 측정

`models/k2-horizon-0.9b`, bf16, M3 Ultra.

- `mlxcel generate -p "The capital of France is" -n 50 --temp 0`: 일관된 출력("... The capital of France is Paris. ..."), 182 tok/s. chat 질문과 `-n 200`으로는 68 토큰 뒤 terminator에서 멈춘다. 마지막 검사 커밋 뒤에도 로드와 생성이 된다.
- `mlxcel-server` `/v1/chat/completions`, streaming과 non-streaming: `finish_reason: stop`, 68 토큰, 출력에 `<|ifm|im_end|>` 없음.
- vendored mlx-vlm `k2_horizon`(d998a620, bf16) 대비 teacher-forced logit trace, 12288 토큰 chunk 2개, `compare_logit_traces.py --decided 2`:

| arm | 위치 수 | top-1 불일치 | decided (gap >= 2) | perplexity (기준 15.123 대비) |
|---|---|---|---|---|
| YaRN (이 PR) | 24576 | 2.73% | 0 / 7279 | -0.31% |
| YaRN, pos >= 8192 | 8192 | 2.94% | 0 / 2354 | -0.43% |
| `rope_scaling` 제거 | 24576 | 37.64% | 700 / 7279 (9.62%) | +89.1% |
| 제거, pos >= 8192 | 8192 | 43.21% | 352 / 2354 (14.95%) | +152.2% |

YaRN arm의 불일치는 모두 기준의 top-2 gap이 0.25 미만인 위치에 있으며, mlxcel이 weight를 f16으로 실행하는 것과 맞는다. 제거 arm은 대조군이다. trace가 RoPE scaling에 민감하다는 것을 보여 주고, 손상은 YaRN이 중요해지는 원래 8192 토큰 context 너머에서 더 커진다.

## 5. 측정하지 않은 것

- 실행되는 것은 0.9B뿐이다. 7B의 `rope_parameters` lift(base 1e7, scaling 없음)는 단위 테스트로 고정했지만, 7B 자체는 grouped RMSNorm 때문에 거부되므로 그 RoPE 경로는 실제 checkpoint를 디코딩한 적이 없다.
- 양자화된 K2-Horizon checkpoint는 trace하지 않았다. quantization은 공유 Llama 경로에 그대로 전달되며 파라미터 유효성만 검사한다.
- CUDA나 ROCm 실행은 없다. 코드는 backend와 무관하지만 Metal 밖의 수치는 없다.
- adapter 경로(`from_weights`)는 `validate()`를 실행하지만 turn terminator를 추가하지 않으므로, adapter로 로드한 모델은 여전히 `<|ifm|im_end|>`를 `content`에 흘릴 수 있다. config-backed `weight_builder` hook은 weight와 config만 받고, 모델 디렉터리나 tokenizer는 받지 않는다.
- `<ifm|think>` 블록(그리고 `think_fast` / `think_faster` 변형)은 `reasoning_content`로 분리되지 않는다. 추론 텍스트와 `</ifm|think>`가 `content`에 남는다.

## 6. 학습 포인트

**serde 기본값은 모델에 대한 주장이다.** 검사용 필드의 기본값을 "안전한" 값(false, 0)으로 두자, 생략된 키가 upstream config class의 의미와 정반대인 "dense"를 뜻하게 되었다. 거부가 목적인 필드라면 기본값은 upstream 기본값이어야 하며, 그렇지 않으면 키가 빠진 config가 검사를 통과한다.

**대조군이 있어야 parity trace가 의미를 가진다.** top-1 불일치 2.73%만으로는 알 수 있는 것이 적다. decided 불일치 700개를 낸 `rope_scaling` 제거 arm이 이 비교가 RoPE 실수를 잡아낸다는 것을 보여 주므로, YaRN arm의 0 / 7279는 신호가 없는 것이 아니라 증거가 된다.

**dispatch 표 하나하나가 진입점이다.** 모델이 Llama 위에서 돌기 때문에 TP fallback 이름을 `"llama"`로 두기 쉽다. 그러면 검사를 건너뛰는 로드 경로가 생긴다. 한 계열이 검사용 변환을 거쳐 decoder를 재사용한다면, 모델 타입을 매핑하는 모든 표가 그 변환을 가리켜야 한다.

## 7. 후속 작업

- 공유 decoder에 grouped RMSNorm(`layernorm_num_groups` > 1)을 넣으면 3.7B / 7B / 32B dense checkpoint가 열리고, base 1e7 RoPE 경로도 실제 trace를 받게 된다.
- `<ifm|think>` / `</ifm|think>` 쌍(과 effort 변형)을 공유 reasoning marker 표 다섯 곳에 등록해 서버가 `reasoning_content`를 채우게 한다.
- config-backed `weight_builder` hook이 tokenizer에 접근하게 하거나 이후 단계에서 terminator를 추가해, adapter 경로도 `<|ifm|im_end|>`에서 멈추게 한다.
- QK-norm, MoE, MoVA 지원은 이 변환의 확장이 아니라 새 decoder가 필요한 별도 작업이다.
