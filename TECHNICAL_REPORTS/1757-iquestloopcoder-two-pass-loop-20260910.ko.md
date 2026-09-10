# 기술 보고서: PR #1757 - feat(models): add the IQuest-Coder Loop two-pass decoder (iquestloopcoder)

**날짜**: 2026-09-10
**작성**: mlxcel 메인테이너
**검토**: 구현 및 검증 사이클
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: 중간 (새 모델 계열, 텐서 병렬·파이프라인 병렬 미지원, 형제 계열과 공유하며 의도적으로 그대로 둔 토크나이저 간극 1건, 이 체크포인트에서는 측정 자체가 불가능한 인수 기준 1건)

---

## 요약

PR #1757은 `model_type: "iquestloopcoder"`를 추가한다. `mlx-community/IQuest-Coder-V1-40B-Loop-Instruct-4bit`(21GB, 4비트 affine, group_size 64)로 배포된 두 번 도는 디코더다. 형태는 평범한 Llama다. 80층, hidden 5120, 어텐션 헤드 40개에 KV 헤드 8개, head_dim 128, intermediate 27648, vocab 76800, rms_norm_eps 1e-5, rope_theta 500000, `tie_word_embeddings` false, `eos_token_id` `[2, 75864, 75869]`. 새로운 것은 이 80층이 `loop_num` 2와 `loop_window_size` 64 아래에서 같은 토큰을 같은 가중치로 두 번 지나간다는 점이다.

1차 패스는 평범한 causal 어텐션이고 자신의 K/V를 저장한다. 2차 패스에서 각 층의 어텐션 출력은 두 갈래를 헤드별 sigmoid 게이트로 섞은 값이다. 전역 갈래는 같은 층의 1차 패스가 만든 K/V를 보고 지역 갈래는 2차 패스 자신의 K/V를 64토큰 슬라이딩 윈도우로 본다. 게이트는 `sigmoid(q2 . gate_w[h] + gate_b[h])`이고 RoPE를 거친 2차 패스 쿼리에서 계산한다. `model.gate_projections.{i}.weight` `[40,128]`과 `.bias` `[40]`이 2차 패스 전용 가중치의 전부이고 체크포인트에서 한 번도 양자화되지 않는 유일한 텐서이기도 하다. 그래서 층마다 캐시를 둘씩 쥔다. 1차 패스용 dense `KVCache`와 2차 패스용 `RotatingKVCache(64)`다.

검증에는 체크포인트 자신의 `modeling_iquestloopcoder.py`를 보고 새로 쓴 스트리밍 float32 NumPy 오라클을 썼고 벤더 클래스와 대조해 16개 형상 사례 전부에서 1.1e-06으로 일치했다. 두 테스트 프롬프트 모두 greedy argmax가 맞았고 12토큰 이어쓰기도 동일하다. 정작 윈도우 자체는 이 방식으로 검증하지 못했다. 이유는 6절의 측정에 있다. 모든 층의 게이트 bias가 정확히 +2.0이라 지역 갈래가 출력의 12퍼센트 정도만 차지하고 윈도우를 통째로 없애도 218토큰과 521토큰 양쪽에서 argmax도 top-10 순서도 바뀌지 않는다. 파일 14개, +2681줄, 새 유닛 테스트 17개와 테스트 4개짜리 실제 체크포인트 대조 하네스를 더했다.

---

## 1. 문제 정의

### 1.1 아키텍처의 실체

벤더는 가중치와 함께 `modeling_iquestloopcoder.py`를 배포한다. 그 안의 `_forward_loop`는 80층 스택을 한 번 돌려 층별 K/V를 남기고 같은 스택을 자신의 출력 위에서 한 번 더 돌린다. 1차 패스 어텐션은 causal이고 특별할 것이 없다. 2차 패스는 토큰마다 쿼리를 하나 만들어 두 번 어텐션하는데 한 번은 같은 층의 1차 K/V를 상대로 하고 한 번은 자신이 계산한 K/V를 상대로 하되 마지막 64자리로 마스킹한다. 두 결과는 2차 쿼리에서 읽어낸 스칼라 게이트로 헤드마다 섞는다.

### 1.2 각 지점이 위험한 이유

- **2차 패스가 잘못된 캐시를 읽는 경우.** 전역 갈래가 1차 대신 2차 K/V를 읽게 만드는 것은 토큰 하나짜리 수정이고 오류를 내지 않는다. 모델은 여전히 온전한 causal 이력을 상대로 어텐션하고 여전히 그럴듯한 코드를 낸다.
- **게이트가 잘못된 쿼리를 쓰는 경우.** RoPE 이전 쿼리로 게이트를 계산해도 형상도 값 범위도 그대로이고 헤드별 혼합 비율만 몇 퍼센트씩 어긋난다.
- **윈도우가 빠지는 경우.** 64토큰 제한이 없는 지역 갈래는 그냥 더 넓은 어텐션이라 넘치는 것도 없고 걸리는 단언도 없다.
- **캐시 표면 불일치.** 생성기의 `Vec<KVCache>`로는 층마다 캐시가 둘이고 그중 하나가 회전하는 구조를 표현하지 못한다. 트레이트 기본 구현은 무관한 기능 플래그에서 캐시 레이아웃을 추론하는데 이것이 서버가 디코더 레이어를 0개 돌렸던 Falcon-OCR 전례(PR #1075)다.

---

## 2. 기술적 결정

### 2.1 `Attention`을 세 메서드로 나눈 이유

공유 `llama3::Attention::forward`는 캐시 하나를 중심으로 투영, 캐시 갱신, SDPA를 한데 묶는다. 2차 패스는 쿼리 하나에 K/V 두 벌이 서로 다른 캐시에 들어 있으므로 `src/models/iquestloopcoder.rs`는 `get_qkv(x, offset)`, `attend(q, k, v, window)`, `project_out(attn)`을 따로 노출한다. `llama3::MLP`는 필드가 공개돼 있어 직접 생성해 재사용했다. 튜닝된 SwiGLU 경로를 복사하지 않고 공유하는 셈이다.

`attend`는 마스크를 직접 만들지 않고 `mlxcel_core::causal_attention(q, k, v, scale, 0.0, window)`에 위임한다. 이 헬퍼는 causality를 오른쪽 아래로 정렬하며(`offset = k_len - q_len`) 두 갈래 모두 그 정렬을 원한다. `RotatingKVCache`와 짝이 맞는 헬퍼도 이쪽이다. 윈도우보다 긴 프리필에서 K/V를 잘라내는 대신 키를 전부 유지한 채 전폭 마스크로 윈도우를 강제하는데 잘라냈다면 가장 이른 쿼리 행들이 softmax 행 전체가 `-inf`인 상태로 남는다. 디코드(`q_len == 1`)는 마스크 없는 경로를 탄다. 회전 캐시가 이미 키를 최대 64개만 쥐고 있기 때문이다. 흔히 지나가는 경로에서는 마스크를 물리적으로 만들지 않는다.

### 2.2 모델이 소유하는 시퀀스 상태, 그리고 그 선언이 정리하는 것

상태는 `SequenceId`로 키를 잡는 `ModelOwnedSequenceState<LayerCaches>`에 들어간다. Gemma 3, Llama 4, Qwen3-Next가 쓰는 것과 같은 장치다. `make_caches()`는 빈 벡터를 돌려주고 `sequence_state_layout()`은 `SequenceStateLayout::model_owned(num_layers)`를 돌려준다.

이 선언이 프리픽스 캐싱까지 정리한다. `BatchScheduler`의 프롬프트 캐시 기증·채택 경로는 둘 다 `SequenceStateBackend::ModelOwned`에서 일찍 반환하고 `PromptCacheRejectReason::ModelOwnedState`를 기록한다. 계열이 구조적으로 제외되고 요청에는 오해를 부르는 캐시 미스 대신 건너뜀이 남는다. 스냅숏 재사용은 꺼져 있고(트레이트 기본값) `supports_batching`은 false다. `supports_padded_prefill`도 false인데 이미 한 바퀴 돈 회전 캐시에 붙인 패드 토큰은 dense 캐시와 달리 다시 잘라낼 수 없기 때문이다. 텐서 병렬은 근사하지 않고 거부한다. `fallback_architecture`는 `"llama"`가 아니라 `"iquestloopcoder"`를 돌려준다. `"llama"`를 돌려주면 이 디코더를 감당하지 못하는 TP Llama 런타임이 켜진다.

이슈의 계획에서 벗어난 부분이라 그대로 기록해 둔다. 이슈 #1360은 `make_caches`가 `[pass1_0..., pass2_0...]` 순서로 평탄한 `KVCache` `2 * num_hidden_layers`개를 돌려주는 안을 제시했다. 구현하지 않았다. 평탄한 dense 목록으로는 회전하는 절반을 표현할 수 없고 스케줄러에 실제로는 소유하지 않은 시퀀스별 상태를 쥐여 주게 된다. 모델 소유 경로는 대신 진짜 시퀀스별 격리를 주고 이는 구동 중인 서버에서 확인했다(8절).

### 2.3 탐지 가드

`"iquestloopcoder"`는 `iquest_loop_coder_model_type`을 거친다. 여기서 가중치를 한 바이트라도 읽기 전에 `loop_num != 2`를 거부한다. 가중치를 읽는다는 것은 21GB를 읽는다는 뜻이다. 정수로 파싱되지 않는 `loop_num`은 벤더 기본값 2로 되돌리지 않고 그대로 실패시킨다.

`"model_type": "llama"`로 이름표만 바꿔 달고 `architectures: ["IQuestLoopCoderForCausalLM"]`은 그대로 선언한 설정도 루프 디코더로 보내고 이 분기는 평범한 Llama 분기보다 앞에 둔다. `llama`로 이름표를 바꾸는 것은 이 계열에서 실제로 쓰이는 방식이다. `auto_map` 코드를 실행하지 않는 스택에서도 체크포인트가 로드되게 만드는 방법이 그것이다. 그냥 흘려보내면 스택을 두 번이 아니라 한 번만 돌리고 `gate_projections` 텐서는 하나도 읽지 않은 채 모델의 절반으로 그럴듯한 출력을 낸다. 아키텍처 문자열은 부분 문자열이 아니라 전체 일치로 비교한다. `IQuestLoopCoderForCausalLM`과 형제 계열의 `IQuestCoderForCausalLM`이 중간 한 조각만 다르기 때문이다.

### 2.4 벤더 디코드 경로와 어긋나는 세 지점

`_forward_with_cache`는 벤더의 단일 토큰 디코드 경로이고 `_forward_loop`와 두 가지가 어긋나는데 이 포트는 둘 다 재현하지 않았다. 재현하면 디코드가 모델이 학습된 의미와 어긋나기 때문이다. 첫째, 이쪽 프리필은 2차 패스 지역 캐시를 층이 이미 실행된 뒤의 은닉 상태에서 다시 계산한 K/V로 채운다. 그 층의 2차 패스 어텐션이 실제로 쓴 K/V가 아니다. 둘째, 프롬프트가 윈도우보다 길어도 지역 캐시가 `loop_window_size`로 줄어들지 않는다. `update_local`이 프롬프트 토큰 전부로 캐시를 채운 뒤 스텝마다 하나씩 밀어내므로 윈도우 폭이 프롬프트 폭 그대로 남는다. 세 번째 특이점은 `forward_decode_loop2`에 있다. 윈도우 마스크의 적용 여부를 `q2.shape[2]`로 판정하는데 디코드 중에는 이 값이 1이므로 거기서도 윈도우가 적용되지 않는다. 프리필 로짓은 세 가지 모두의 영향을 받지 않는다.

`_forward_loop` 자체에 대해서도 하나 더 확인했다. `use_cache=False`면 공유 캐시가 아예 할당되지 않고 2차 패스 은닉 상태에서 K1/V1을 다시 계산하는 폴백이 켜지며 그러면 어텐션 A와 B가 마스크만 다른 같은 K/V 위에서 돈다. `use_cache`의 기본값은 `config.use_cache = true`이므로 실제 `generate()` 프리필은 전부 1차 KV 경로를 타고 이 포트가 구현한 것도 그 경로다.

---

## 3. 변경 요약

| 영역 | 변경 |
| --- | --- |
| `src/models/iquestloopcoder.rs` | 설정, `LoopGate`, `get_qkv` / `attend` / `project_out`으로 나뉜 `Attention`, `TransformerBlock`, `LayerCaches` |
| `src/models/iquestloopcoder_model.rs` | `IQuestLoopCoderModel`, `IQuestLoopCoderWrapper`, `sanitize_weights` |
| `src/models/detection.rs`, `detection_tests.rs` | `iquest_loop_coder_model_type`, `loop_num` 가드, Llama보다 앞에 놓인 이름표 교체 설정 분기 |
| `src/models/mod.rs`, `registry.rs`, `src/loaded_model.rs`, `src/model_metadata.rs`, `src/execution/memory_estimate.rs` | 등록, 메타데이터, 메모리 추정 |
| `src/distributed/tensor_parallel/inference.rs` | `fallback_architecture`가 이 계열에 대해 TP Llama 런타임을 거부 |
| `src/tokenizer/mod.rs` | 7절의 non-special 추가 토큰 동작을 고정 |
| `src/models/iquestloopcoder_tests.rs`, `tests/iquestloopcoder_parity.rs`, `docs/supported-models.md` | 유닛 테스트 17개, 실제 체크포인트 하네스, 문서 |

---

## 4. 테스트를 차등 방식으로 짠 이유

1.2의 세 실패 모드는 틀려도 그럴듯한 출력을 낸다. 그래서 테스트는 틀린 변종을 명시적으로 만들어 두고 구현이 그것과 일치하지 않는지를 확인한다.

`pass2_matches_only_the_correct_reference`는 구현의 2차 패스 어텐션 출력을 테스트 안에서 조립한 레퍼런스와 비교하며 변종 넷을 둔다. 올바른 것, `GateFromPreRope`, `GlobalReadsPass2Kv`, `LocalUnwindowed`이다. 윈도우 4에 12토큰 실행에서 올바른 쪽과는 최대 절대 차 1e-5 미만을, 틀린 쪽 각각과는 1e-3 초과를 요구한다.

`window_limits_local_branch`는 윈도우 4짜리 12토큰 실행에서 키 0..7을 덮어쓴 뒤 11번 자리가 그대로인지(1e-5 미만) 그리고 7번 자리가 움직이는지(1e-3 초과)를 확인한다. 윈도우 없는 갈래였다면 11번 자리도 움직였을 것이라는 점을 따로 단언해서 테스트가 공허해지지 않게 했다. 유닛 테스트는 모두 17개이고 전부 통과한다.

---

## 5. 실제 체크포인트 검증

레퍼런스는 두 번 도는 루프를 스트리밍 float32 NumPy로 구현한 것이다. 체크포인트 자신의 `modeling_iquestloopcoder.py`를 보고 썼고 4비트 affine 가중치를 층 단위로 직접 역양자화하므로 fp32로 펴면 160GB가 될 모델을 상대로 피크 RSS가 1.5GiB에 그친다. mlxcel과 코드를 전혀 공유하지 않는다. 벤더 클래스로 만든 작은 무작위 가중치 fp32 torch 모델과 대조하면 `L <= window`, `window < L <= 2*window`, `L > 2*window`를 아우르는 16개 형상 사례 전부에서 최대 절대 차 1.1e-06으로 일치한다.

쓰인 MLX affine 역양자화 식은 `w[o,i] = q[o,i] * scales[o, i//64] + biases[o, i//64]`이고 `q`는 낮은 니블부터 푼다. 독립적인 방법 넷으로 확인했는데 결정적이었던 것은 이것이다. MLX는 크기가 큰 쪽 그룹 끝점을 코드 0에 대응시키므로 scale의 절반가량이 음수이고 `bias`는 `q=0`에서의 값이다. 코드 0과 코드 15를 모두 쓰는 모든 그룹에서 복원한 그룹 최솟값·최댓값이 실제 그룹 최솟값·최댓값과 차이 0.0으로 정확히 맞았고 이것이 `(q-8)*scale` 해석을 배제한다. 니블 순서는 `q_proj`의 입력 열별 RMS를 `|input_layernorm.weight|`와 상관 지어 확정했다. 낮은 니블 우선이 +0.7564, 높은 니블 우선이 -0.0096이다.

CUDA sm_121에서 mlxcel과 대조한 마지막 자리 로짓 결과다.

| 프롬프트 | 토큰 | argmax 일치 | KL(oracle \|\| mlxcel) | 로짓 상관 |
| --- | --- | --- | --- | --- |
| short chat | 38 | 예 (66644) | 0.029 | 0.9895 |
| long chat | 218 | 예 (5123) | 0.0047 | 0.9980 |

38토큰 프롬프트에서 이어 쓴 greedy 출력은 12토큰 전부 오라클과 일치하고 디코딩하면 `# Python Function to Reverse a String\n\nHere are several`이다.

---

## 6. 부정적 결과: 오라클로는 윈도우를 검증할 수 없다

이슈 #1360의 인수 기준은 128토큰보다 긴 프롬프트에서 오라클과 대조할 것을 요구했다. 근거로는 `loop_window_size`를 넘어서면 지역 갈래가 키를 버리므로 윈도우 없는 구현은 어긋날 수밖에 없다고 적었다. 이 체크포인트에서는 그 전제가 성립하지 않는다.

모든 층의 `gate_projections.{i}.bias`가 정확히 +2.0이고 게이트 가중치도 작아 80개 층 전부에서 게이트 값이 `sigmoid(2.0) = 0.8808`에 놓인다. 218토큰 프롬프트에서 층별로 재면 최소 0.874602(층 64), 최대 0.884268(층 77), 평균 0.879620, 표준편차 0.001995다. 그래서 섞인 출력에서 지역 갈래가 차지하는 몫은 12퍼센트 정도뿐이다.

윈도우를 빼고 오라클을 다시 돌리면 이렇다.

| | 218 토큰 | 521 토큰 |
| --- | --- | --- |
| argmax 변화 | 없음 | 없음 |
| top-10 순서 변화 | 없음 | 없음 |
| 평균 절대 로짓 차 | 5.01e-02 | 8.94e-03 |
| 최대 절대 로짓 차 | 3.37e-01 | 5.34e-02 |

길이가 늘수록 효과는 커지지 않고 오히려 줄어든다. mlxcel 쪽은 f16 활성값을 양자화된 `lm_head`에 통과시키고 오라클은 f32로 도는데 적합된 관계식 `mlxcel = 1.09477 * oracle - 0.15529`가 기술하는 상수 오프셋을 걷어낸 뒤 mlxcel 자신의 편차는 평균 2.674e-01이다. 윈도우 효과 전체의 5.3배다. 218토큰 프롬프트에서 직접 재면 mlxcel은 윈도우 있는 오라클에서 0.2674, 없는 오라클에서 0.2678 떨어져 있다. 분리되지 않는다.

분명히 적어 둔다. 이 체크포인트에서는 실제 체크포인트 오라클과 greedy id나 로짓을 비교하는 방식으로 윈도우 있는 구현과 없는 구현을 구별할 수 없고 프롬프트를 아무리 길게 잡아도 마찬가지다. 윈도우는 대신 4절의 유닛 테스트로 검증한다. 거기서는 지역 갈래를 따로 떼어 보므로 기여분이 게이트에 희석되지 않는다. 인수 기준은 잴 수 있는 부분(구조, 게이트, 1차 KV 배선, RoPE, 가중치 로드, 디코드 인계)에서 충족으로, 잴 수 없는 부분(오라클을 통한 윈도우 구별)에서 미충족으로 명시해 기록한다.

대조 하네스에도 결과가 하나 따라온다. `tests/iquestloopcoder_parity.rs`의 top-5 순서 단언은 오라클이 mlxcel의 분해능보다 크게 벌려 놓은 항목 사이에만 적용한다. mlxcel의 양자화된 `lm_head`는 크기 16에서 32 구간의 인접 로짓을 0.125 정도까지 분해하는데 218토큰 프롬프트에서 오라클의 2위와 3위는 0.0395 차이다. 둘의 상대 순서는 잡음이라 단언하지 않는다.

---

## 7. 토크나이저 발견

체크포인트는 SentencePiece `tokenizer.model`과 `added_tokens.json`, `tokenizer_config.json`을 담고 있고 `tokenizer.json`은 없으며 `add_bos_token`과 `add_prefix_space`가 모두 false다. `added_tokens_decoder`에는 항목이 30개 있는데 그중 27개가 추가 토큰이고 나머지 3개는 SentencePiece 자신의 `<unk>`, `<s>`, `</s>`다.

27개 중 19개는 `"special": true`로 표시돼 있고 양방향으로 왕복한다. `token_to_id`가 해석하고 정확히 id 하나로 인코딩되며 다시 자기 자신으로 디코딩된다. 나머지 8개는 `"special": false`다(`<think>`, `</think>`, `<tools>`, `</tools>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`). 이 여덟은 디코딩은 제대로 되지만 `token_to_id`가 `None`을 돌려주고 인코딩하면 여러 조각으로 쪼개진다. `<think>`는 `[66580, 30272, 66604]`가 되고 `</tool_call>`은 `[469, 7005, 66561, 2998, 66604]`가 된다.

공유 로더가 의도한 동작이다. `src/tokenizer/mod.rs`의 `parse_special_tokens`는 special이 아닌 추가 토큰을 디코딩 전용 `added_token_contents` 맵에 넣는데 HuggingFace는 `special` 값과 무관하게 `added_tokens_decoder` 항목 전부를 인코딩에서 매칭하므로 둘이 갈린다. 이 포트가 만든 문제가 아니고 형제인 `iquestcoder`에도 똑같이 나타난다. 드러나는 자리는 도구 호출 프롬프트뿐이다. `chat_template.jinja`는 도구 태그 여섯 개를 리터럴로 내보내고 `<think>`나 `</think>`는 내보내지 않는다. 이 동작은 문서 없이 두는 대신 테스트로 고정했고 후속 이슈를 등록했다. 바꾸려면 저장소의 SentencePiece 체크포인트 전부를 건드려야 해서 여기서는 하지 않았다.

---

## 8. 런타임 근거

`mlxcel inspect`는 8192토큰 컨텍스트 기준 총 28.08GiB를 추정한다. 가중치 20.85GiB, KV 2.50GiB, 할당기 오버헤드 4.67GiB다. 디코드는 GB10에서 48~64토큰 구간 4.3~4.5 tok/s로 측정됐고 80층을 두 번 지나가는 것과 앞뒤가 맞는다.

`mlxcel-server`는 CLI와 따로 검증했다. 모델이 CLI 테스트를 전부 통과하면서도 서버에서는 다른 경로를 탈 수 있기 때문이다. 서로 다른 프롬프트 세 개를 동시에 보낸 요청 세 건의 출력이 각각의 직렬 기준선과 바이트 단위로 같았고 이것이 `ModelOwnedSequenceState`가 제공하려는 시퀀스별 격리다.

---

## 9. 실행하지 않은 것

- 이슈 #1360이 인수 기준으로 적은 `cargo test --workspace --profile test-fast --features metal,accelerate`는 이 호스트에서 돌릴 수 없다(Linux aarch64, CUDA, Metal 없음, Accelerate 없음). 돌리지 않았다. 이 계열의 Apple Silicon 동작은 미검증이다.
- CUDA 쪽 대응물은 범위를 좁혀 돌렸다. `cargo clippy --lib --tests --features cuda -- -D warnings`와 `cargo fmt --all -- --check`가 모두 깨끗하고 위에 적은 테스트 스위트도 전부 통과한다.
- 이 계열에서 텐서 병렬과 파이프라인 병렬은 켜지 않았다.
- Turbo와 INT8 KV 캐시 모드는 회전하는 2차 패스 캐시에 배선되지 않아 계열 전체가 FP16 전용으로 남는다.
- `--max-kv-size` 트리밍은 모델 소유 캐시에 닿지 않는다. 다른 모델 소유 계열과 같다.

---

## 10. 교훈

- **인수 기준을 그 기준이 지목한 체크포인트가 반증하기도 한다.** #1360의 128토큰 기준선은 윈도우가 출력에서 유의미하다는 전제 위에 서 있었다. 80개 층 전부의 게이트 bias가 +2.0이면 지역 갈래는 12퍼센트에 머물고 이는 mlxcel 자신의 양자화된 `lm_head`가 만드는 잡음보다 작다. 기준이 미충족이었던 게 아니라 애초에 측정 불가였다.
- **벤더의 디코드 경로가 곧 명세는 아니다.** `_forward_with_cache`는 지역 캐시를 어느 K/V로 채우는지, 윈도우가 64로 좁아지기는 하는지를 두고 `_forward_loop`와 어긋난다. 그대로 베꼈다면 디코드가 가중치를 학습시킨 의미와 어긋났을 것이다.
- **토크나이저 왕복에는 방향이 있다.** special이 아닌 추가 토큰 8개는 디코딩은 모두 제대로 되고 인코딩에서만 실패하므로 디코딩 쪽만 확인했다면 이 계열은 깨끗하다고 보고됐을 것이다.
