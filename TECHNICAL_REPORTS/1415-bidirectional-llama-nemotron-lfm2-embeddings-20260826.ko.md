# 기술 보고서: PR #1415 - 양방향 Llama, Nemotron-3-Embed, LFM2.5-Embedding

**작성일**: 2026-08-26
**작성자**: mlxcel maintainers
**상태**: 완료. 검증 중 정확성 버그 한 건을 발견해 수정
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1415는 이슈 #1325를 구현한다. #1353이 만든 임베딩 기반 위에 세 개 계열의 forward pass를 더 올린다. 셋 다 mlxcel이 이미 생성용으로 돌리는 디코더 백본을 패딩 전용 마스크와 풀링 단계만 붙여 양방향으로 재사용한 것이다. 양방향 Llama(LLM2Vec 레시피, `model_type: llama_bidirec`), Ministral 3 백본 위의 Nemotron-3-Embed(`is_causal: false`), 그리고 LFM2의 short-conv + attention 하이브리드 위의 LFM2.5-Embedding이다.

셋 중 둘은 백본을 전혀 건드리지 않았다. 나머지 하나는 두 군데를 바꿨고, 둘 다 short convolution 안에 갇혀 있다. 패딩을 전부 왼쪽에 붙이는 대신 양쪽으로 나눠 넣는 `conv_causal` 플래그, 그리고 믹서 입력의 패딩 위치를 0으로 만드는 패딩 곱셈자다.

두 번째가 실질적인 발견이다. convolution에는 attention mask가 작용할 key 축이 없어서, 모델의 나머지를 덮는 마스크가 여기까지 닿지 않는다. 명시적으로 0을 넣지 않으면 패드 토큰 임베딩이 경계 옆 실제 위치로 섞여 들어가고, 그 위의 attention 레이어 여섯 개가 그것을 행 전체로 퍼뜨린다. `LiquidAI/LFM2.5-Embedding-350M`에서 측정한 결과, 마스크 아래 토큰 id만 바꿔도 풀링된 벡터가 코사인 0.94만큼 움직였다. 수정은 레퍼런스의 `apply_mask_to_padding_states`를 그대로 따르며, 이 PR의 모든 계열이 이제 이 성질을 허용오차 0으로 검증한다.

실제 체크포인트 네 개가 `mlxcel embed`와 `POST /v1/embeddings` 양쪽으로 로드되고 서빙된다. 검색 마진은 이슈가 요구한 0.15에 대해 0.42에서 0.50이고, 무관한 쌍은 모두 0.06 미만이다. Nemotron-3-Embed의 `mlx-community` 8비트 변환본은 bf16 원본과 코사인 0.9998로 일치한다.

---

## 1. 문제 정의

### 1.1 배경

에픽 #1348은 `src/embeddings/`의 기반 위로 임베딩 계열을 하나씩 포팅한다. 이 PR 이전까지 `ModelType::LlamaBidirec`, `ModelType::Ministral3Embedding`, `ModelType::Lfm2Embedding`은 전부 탐지는 정확히 되면서 `<family> is detected as an embedding checkpoint, but this embedding family is not yet supported by /v1/embeddings`만 답했다.

세 계열은 형태를 공유한다. 생성기와의 차이가 마스크와 없는 head뿐인 causal 디코더다. 원칙적으로 싸다는 뜻이고, 그래서 묶였다. 동시에 조용히 틀리기도 쉽다. 재사용한 causal 백본은 로드되고, 돌고, 유한한 단위 벡터를 내놓으며, 오직 품질에서만 달라지기 때문이다.

### 1.2 기존 장애물

| 장애물 | 위치 |
|---|---|
| `Llama3Model.lm_head`가 `Option`이 아니라 head 없는 모델을 만들 수 없음 | `src/models/llama3.rs` |
| `get_llama4_attn_scale`과 `Cache::as_interface`가 Ministral 3 모듈 내부에 갇혀 있음 | `src/models/ministral3.rs` |
| `ShortConv::forward`가 "depthwise conv를 causal로 유지하려고" 항상 `L_cache - 1`만큼 왼쪽 패딩 | `src/models/lfm2.rs` |
| `Lfm2Model` 필드가 private이고 hidden state 출력이 없음 | `src/models/lfm2.rs` |
| 세 체크포인트 모두 모든 로더가 요구하는 `model.` 접두사 없이 백본 루트를 저장 | 공개된 export |

### 1.3 위험 평가

| 위험 | 영향 | 완화 |
|---|---|---|
| 재사용한 causal 백본이 causal로 남음 | 조용히 열화된 벡터, 오류 없음 | 계열별 게이트: 마지막 토큰을 바꾸면 위치 0이 움직여야 함 |
| Llama 4 attention scale을 잘못된 offset에서 계산 | 생성기와 점수가 어긋나고 위치가 커질수록 벌어짐 | 문서화된 공식을 독립적으로 다시 쓴 값과 비교 |
| short conv가 패딩을 읽음 | 배치 요청마다 오염된 임베딩 | 배치 형상을 고정하고 마스크 아래 내용만 바꾸는 허용오차 0 게이트 |
| `1_Dense` late-interaction 체크포인트가 단일 벡터 임베더로 로드 | ColBERT 모델이 조용히 풀링된 벡터 하나만 반환 | 로드 시점 거부 |

---

## 2. 기술적 선택과 그 이유

### 2.1 `llama3.rs`를 건드리는 대신 weight map에서 레이어를 직접 만든다

이슈의 계획은 `src/models/llama3.rs`에 `forward_hidden` 분리와 head 없는 생성자를 추가하자고 했다. 먼저 머지된 EmbeddingGemma 계열이 이미 대안을 정립해 두었다. `UnifiedEmbedding`, `Vec<TransformerBlock>`, `RMSNorm`을 계열 구조체가 직접 들고 루프를 그 안에서 돌리는 방식이다.

여기서 그 경로를 택한 이유는 `llama3::TransformerBlock::from_weights_with_rope`, `Attention`, `MLP`가 이미 전부 `pub`이라 계열이 백본에서 새로 필요한 게 없기 때문이다. 묶인 head를 만들지 않는 이점도 있다. `Llama3Model::from_weights`는 `tie_word_embeddings`가 켜져 있으면 `model.embed_tokens`에서 `lm_head`를 만드는데, 이 체크포인트에서는 임베더가 절대 적용하지 않을 `128256 x 2048` `UnifiedLinear` 뷰다. 비용은 네 줄짜리 레이어 루프 하나가 중복되는 것이고, 이득은 형제 유닛 셋이 동시에 편집 중인 1523줄 파일에 대한 0줄 diff다.

### 2.2 Ministral 3는 모델 타입을 통째로 재사용한다

`Ministral3Model`은 이미 `lm_head: Option<UnifiedLinear>`와 `pub` 필드를 갖고 있어서, 임베더는 모델을 다시 조립하지 않고 감싼다. 필요한 것은 가시성 확대 두 건뿐이고 둘 다 `pub(crate)`이며 동작 변경은 없다. 임베더가 생성기와 같은 위치별 스케줄을 계산하도록 `get_llama4_attn_scale`, 임베더가 모델 자신의 혼합 캐시 벡터로 `TransformerBlock::forward`를 구동하도록 `Cache::as_interface`.

손으로 만든 `Vec<KVCache>` 대신 `make_caches()`를 쓴다는 것은, 앞으로 `sliding_attention` 레이어를 선언하는 체크포인트가 나오면 그 레이어들이 자동으로 `RotatingKVCache`를 받는다는 뜻이다. forward pass는 이미 그런 레이어에 `create_bidirectional_window_mask`를 고른다. 공개된 두 체크포인트는 이 경로를 밟지 않으므로(`sliding_window`가 `null`이고 `layer_types`가 없다) 작성됐지만 검증되지 않았다.

### 2.3 attention scale은 offset 0에서 계산하며, 이유를 밝혀 둘 값어치가 있다

`Ministral3Model::forward_with_caches`는 Llama 4 스케줄을 자신의 full-attention 캐시 offset에 고정한다. 새 prefill에서 그 offset은 0이고, 임베더가 넘기는 값도 정확히 그것이라, 두 경로 모두 위치 `i`를 토큰 `i`로 색인한다. 스케줄은 `1 + beta * ln(1 + floor(pos / original_max_position_embeddings))`이고 `beta`는 0.1, 윈도우는 16384라, 8192로 잘린 입력에서는 모든 스케일이 정확히 1.0이며 스케일링은 무효다. 그래도 계산은 한다. 윈도우가 더 작거나 상한을 올린 체크포인트에는 필요하고, 조용히 빠진 곱셈자는 품질 회귀로만 드러나는 종류의 문제이기 때문이다.

### 2.4 `conv_causal`의 기본값은 true라 생성 경로는 바이트 단위로 동일하다

유일한 백본 동작 변경은 serde 기본값 뒤에 놓았다. 공개된 어떤 `config.json`도 `conv_L_cache`와 방향성 키를 함께 선언하지 않으므로, LFM2와 LFM2-MoE 생성은 순전히 기본값만으로 causal 분기를 유지한다. 그 분기는 변경 전 코드 경로 그대로다. `L_cache - 1`개의 0을 전부 왼쪽에, conv state는 패딩된 꼬리에서 기록한다.

비causal 분기는 같은 총량을 나눈다. `left = L_cache / 2`, `right = L_cache - 1 - left`라서 어떤 `L_cache`에서도 출력 길이는 `L`이고, 공개 체크포인트의 홀수 `L_cache = 3`에서는 윈도우가 정확히 대칭이다. conv state도 더 이상 쓰지 않는다. 양방향 윈도우의 꼬리를 담은 "state"는 decode 단계가 이어받을 수 있는 물건이 아니기 때문이다.

### 2.5 short conv는 자기만의 패딩 메커니즘이 필요하다

이 PR의 발견이다. 임베딩 기반은 계열에 `create_bidirectional_padding_mask`를 준다. attention key 축에서 패딩을 막는 마스크다. convolution에는 key 축이 없다. 레이어마다 잔차 스트림의 패딩 위치를 0으로 만드는 방식도 답이 아니다. 레퍼런스는 잔차를 그대로 두고 믹서마다 다시 마스킹하기 때문이다. 0으로 만들어야 하는 것은 conv 입력이고, 모든 conv 레이어에서, 잔차가 무엇을 담고 있든 상관없이 그래야 한다.

그래서 `ShortConv::forward`는 활성값 dtype의 `[B, L, 1]` 곱셈자를 선택적으로 받아 입력에 곱한다. 생성 경로는 `None`을 넘기고 변경되지 않는다. 임베더는 호출마다 곱셈자를 하나 만들어 모든 레이어에 넘긴다. attention 레이어는 마스크가 이미 패딩을 덮으므로 무시한다.

수정 전에는, 폭과 마스크를 고정한 `B = 1` 실행에서 마스크 아래 토큰 id만 바꿨는데 풀링된 벡터가 코사인 0.94만큼 움직였고, 배치 대 단독 일치도는 코사인 0.9968이었다. 수정 후에는 앞의 값이 정확히 0, 뒤의 값이 0.99992다.

### 2.6 LFM2의 최종 norm 루트는 공유 sanitizer가 아니라 계열 안에서 접두사를 붙인다

`sanitize_decoder_embedding_weights`는 `embed_tokens.`, `layers.`, `norm.`에 `model.`을 붙인다. LFM2는 최종 norm을 `embedding_norm.weight`로 쓰는데 셋 중 어디에도 맞지 않는다. 공유 상수에 네 번째 루트를 더하는 것도 의미상으로는 맞지만, 그 공유 모듈은 동시 진행 중인 형제 유닛들이 편집하고 있었다. 그래서 계열이 같은 멱등 계약을 가진 다섯 줄짜리 `prefix_embedding_norm`을 직접 든다. 작은 중복을 더 작은 머지 표면과 맞바꾼 의도적 선택이며, 에픽의 웨이브가 착지한 뒤 공유 목록으로 접는 것이 합리적인 후속이다.

### 2.7 적용되지 않는 `Dense` 모듈은 조용한 누락이 아니라 로드 오류다

공개된 세 체크포인트 중 어느 것도 풀링 후 projection을 싣지 않는다. 그래도 세 계열 모두 그런 것이 있으면 로드를 거부한다. `sanitize_decoder_embedding_weights`가 접은 `Dense` 폴더 개수를 반환하고, 0이 아니면 그 개수를 밝히며 로드를 실패시킨다. 로드는 되지만 적용되지 않는 `Dense` 모듈은 하류에서 아무도 감지할 수 없는 방식으로 틀린 벡터를 만든다.

LFM2에서는 같은 검사가 두 몫을 한다. `1_Dense` 폴더가 바로 ColBERT late-interaction 레이아웃이고, 거부 메시지가 그렇게 말한다. 그 레이아웃은 이번 이슈의 범위 밖이며 #1337이 착지한 뒤의 후속 후보다.

### 2.8 프롬프트 접두사는 호출자 쪽에 남는다

세 체크포인트 모두 `config_sentence_transformers.json`에 접두사를 선언한다(NVIDIA 두 모델은 `query: `와 `passage: `, LFM2는 `query: `와 `document: `). 셋 다 `format_text`를 재정의하지 않으므로 입력은 보낸 그대로 임베딩된다. 호출자가 준 task 문자열을 포맷에 넣어야 해서 명시적인 `instruction` 훅을 가진 Qwen3-Embedding을 빼면, 머지된 모든 계열과 같은 방식이다.

---

## 3. 구현 세부

### 3.1 양방향 Llama forward

```
h    = embed_tokens[input_ids]
mask = create_bidirectional_padding_mask(attention_mask)      # [B, 1, 1, L]
for layer in layers:                                          # 매번 새 KVCache::new()
    h = layer.forward(h, cache, Some(mask))
h    = norm(h)
out  = mean_pool(h, attention_mask)                           # L2 정규화는 엔진이
```

weight sanitize 순서: `rotary_emb.inv_freq` 또는 `position_ids`로 끝나는 키 제거, `language_model.` 래퍼 접두사 제거, 그 다음 공유 패스가 `Dense` 폴더를 접고 `lm_head.*` / `head.*`를 버리고 `embed_tokens.` / `layers.` / `norm.`에 `model.`을 붙인다. 순서가 중요하다. 래퍼 제거가 접두사 판단보다 앞서야 하며, 그렇지 않으면 `language_model.layers.0.…`는 맨 백본 루트로 인식되지 못한다.

### 3.2 Nemotron-3-Embed forward

```
h          = model.embed_tokens[input_ids]
attn_scale = get_llama4_attn_scale(L, 0, beta, original_max_position_embeddings)
full_mask  = create_bidirectional_padding_mask(attention_mask)
window     = create_bidirectional_window_mask(attention_mask, sliding_window)  # 슬라이딩 레이어가 있을 때만
for i, layer in model.layers:
    h = layer.forward(h, attn_scale, caches[i].as_interface(),
                      Some(window if layer.use_sliding else full_mask))
h          = model.norm(h)
out        = mean_pool(h, attention_mask)
```

`Ministral3Model::from_weights`가 돌기 전에 `tie_word_embeddings`를 true로 강제한다. 그래야 sanitize 패스가 방금 버린 head를 다시 찾지 않고 `lm_head`를 `None`으로 둔다.

### 3.3 LFM2.5-Embedding forward

```
h        = embed_tokens[input_ids]
mask     = create_bidirectional_padding_mask(attention_mask)
pad_mult = padding_multiplier(attention_mask, dtype_of(h))    # [B, L, 1], 실제 1 / 패딩 0
for layer in layers:                                          # sequence_state가 아닌 새 캐시
    h = layer.forward(h, cache,
                      Some(mask) if layer.is_attention() else None,
                      Some(pad_mult))
h        = embedding_norm(h)
out      = cls_pool(h, attention_mask)
```

`Lfm2Model::from_weights`는 소유권을 넘겨받은 weight를 자기 sanitize에 태운다(`w1`/`w2`/`w3` feed-forward 이름 변경과 `[hidden, 1, L_cache]` -> `[hidden, L_cache, 1]` conv 전치). 그래서 계열이 먼저 `model.` 접두사를 붙인 뒤 맵을 넘긴다.

CLS 풀링은 토크나이저 post-processor가 앞에 붙이는 `<|startoftext|>`를 읽는다. 오른쪽 패딩에서는 모든 행의 인덱스 0이지만, `pool`은 그렇게 가정하지 않고 첫 실제 토큰 argmax로 찾으므로 왼쪽 패딩 배치에서도 올바르게 풀링한다.

### 3.4 등록

`src/embeddings/loader.rs`에 arm 세 개가 늘었다. 다른 계열의 `not yet supported` arm은 손대지 않았다. 탐지, `src/model_metadata.rs`, `mlxcel arch` / `mlxcel list` 출력은 기반에서 이미 맞게 되어 있어 이 PR은 거기를 바꿀 필요가 없었다. `src/embeddings/real_checkpoint_tests.rs`에서는 LFM2.5-Embedding이 미포팅 계열 목록을 떠난다. EmbeddingGemma, Qwen3-Embedding, BERT가 앞서 그랬던 것과 같다.

---

## 4. 테스트 전략

### 4.1 어떤 형상 검사로도 잡을 수 없는 두 성질

**양방향성.** 재사용한 causal 백본은 형상도 dtype도 맞고 유한한 단위 벡터를 낸다. 게이트는 96개 토큰 중 마지막을 바꾸면 위치 0이 움직여야 한다는 것이다. attention만 있는 두 계열은 기준이 크기지만, LFM2의 기준은 "움직였는가"다. all-conv 대조군이 판별력을 증명하기 때문이다. attention 레이어를 conv로 바꾸면 토큰 95는 스택의 레이어당 `L_cache / 2` 도달 범위를 한참 벗어나 위치 0이 비트 단위로 같게 남는다. 그러니 0보다 크면 그것은 마스크가 일한 것이다.

**패딩 불가시성.** 게이트는 배치 형상, 슬롯, 마스크를 전부 고정하고 마스크 아래 토큰 id만 바꾼 뒤 비트 단위 동일 출력을 요구한다. 커널 기하가 전혀 움직이지 않으므로 허용오차는 정확히 0이고, 차이가 있다면 진짜 누출이다. LFM2 short-conv 버그를 잡은 것이 이 테스트다.

### 4.2 방향성 convolution은 모델을 통해서가 아니라 믹서에서 직접 테스트한다

`lfm2_tests`가 one-hot 임펄스로 구동할 수 있도록 `ShortConv`를 `pub(crate)`로 열었다. fixture는 두 채널이다. 채널 0이 임펄스를 싣고 채널 1은 상수 1이며, `in_proj` weight는 게이트 `C`와 값 `x`가 어디서나 1이고 `B`만 임펄스가 되도록 골랐다. 그러면 출력이 임펄스의 순수 convolution이 되고, 탭 세 개를 서로 다르게 두어 각각이 어디에 떨어지는지 모호하지 않다. causal은 `t`의 임펄스를 `t, t+1, t+2`로 퍼뜨리고 conv state를 쓴다. 양방향은 `t-1, t, t+1`로 퍼뜨리고 쓰지 않는다. 모델 전체의 16개 레이어를 통과시키면 이 한 칸 차이는 보이지 않는다.

### 4.3 배치 기하 바닥, 그리고 이슈의 수치 형태 중 하나를 쓰지 않은 이유

이슈는 패딩된 배치가 단일 입력 벡터와 "1e-3 이내"로 일치할 것을 요구한다. 풀링된 벡터는 코사인에서 그 조건을 만족하고, 보고서에 수치를 싣는다. 하지만 가장 큰 단일 성분에서는 1e-3으로 일치하지 않으며, 이 하드웨어에서는 그럴 수 없다.

같은 텍스트를 **패딩 없이** 2, 3, 4, 5, 8개 복사본으로 한 배치에 넣으면 `nvidia/Nemotron-3-Embed-1B-BF16`의 최대 성분이 최대 3.7e-3까지 움직이는 동안 코사인은 0.99997 위에 머문다. 이미 머지된 `Qwen/Qwen3-Embedding-0.6B`도 똑같이 행동하고, 관련된 핵심 프리미티브 네 개(`conv1d`, `matmul`, `[B,1,1,L]` 마스크를 받는 `attention_from_ptr`, 실제 head 형상의 GQA attention)는 각각 따로 보면 배치 슬롯에 대해 결정적이다. 이 효과는 MLX CUDA 백엔드가 bf16에서 배치 기하로부터 누적 형상을 고르는 것이지 포트가 통제하는 것이 아니므로, 성분 단위 한계는 코드가 아니라 백엔드를 검증하게 된다.

코사인 형태는 1e-3으로 게이트하고, 성분 편차는 매 실행 출력하며, 포트가 실제로 책임지는 성질은 4.1의 패딩 내용 테스트가 허용오차 0으로 검증한다.

### 4.4 MLX 테스트 직렬화

`EmbeddingModel`은 단일 스레드로 문서화되어 있고 제품은 임베딩 워커로 그것을 지키지만, `cargo test`는 논리 CPU당 스레드 하나를 돌린다. 새 모듈 세 개의 모든 MLX 평가 테스트가 공유 `embedding_test_support::mlx_test_guard`를 잡고, `lfm2_tests.rs`에 원래 있던 MLX 평가 테스트 둘도 소급 적용했다. 이 보고서의 모든 게이트 수치는 `--test-threads=1`에서 기록했다.

---

## 5. 실제 체크포인트 결과

연속 세 번의 `--test-threads=1` 실행이 바이트 단위로 같은 출력을 냈다. 산포가 0이라 단일 값을 인용한다.

| 체크포인트 | dim | max_length | 관련 | 무관 | 마진 | 중복 행 | 패딩 내용 누출 | 배치 코사인 |
|---|---|---|---|---|---|---|---|---|
| `nvidia/llama-nemotron-embed-1b-v2` | 2048 | 8192 | 0.432512 | 0.012978 | 0.419534 | 1.000000119 | 0 | 0.999944985 |
| `nvidia/Nemotron-3-Embed-1B-BF16` | 2048 | 8192 | 0.552243 | 0.057058 | 0.495185 | 1.000000119 | 0 | 0.999971032 |
| `mlx-community/Nemotron-3-Embed-1B-BF16-8bit` | 2048 | 8192 | 0.551046 | 0.056344 | 0.494702 | 1.000000119 | 0 | 1.000000000 |
| `LiquidAI/LFM2.5-Embedding-350M` | 1024 | 512 | 0.400611 | -0.026907 | 0.427518 | 1.000000119 | 0 | 0.999917984 |

"관련"은 태양광 질의 대 광전 효과 문단, "무관"은 같은 질의 대 요리법 문단이다. 이슈의 기준은 마진 0.15와 모든 무관 쌍 0.5 미만이다.

**양자화.** 8비트 변환본은 입력별로 bf16 원본과 코사인 0.999839, 0.999839, 0.999857, 0.999839로 일치한다. 이슈 기준은 0.99다.

**실제 weight에서의 양방향 prefill.** 64토큰 이상 프롬프트가 자기 자신의 3분의 1 접두부에 대해 0.999 미만을 기록한다. 양방향 Llama 0.977499, LFM2 0.961427. 같은 풀링의 causal prefill이라면 공유 접두부 위치는 그대로 남았을 것이다.

**엔드포인트 일치.** `mlxcel embed --json`과 `mlxcel-server`의 `POST /v1/embeddings`를 네 체크포인트에 모두 돌려 위 표를 재현했다. 폭이 맞는 단위 노름 벡터, `model_type`이 각각 `LlamaBidirec`, `Ministral3Embedding`, `Lfm2Embedding`으로 보고되고, HTTP 요청 하나 안의 중복 행이 코사인 1.000000000이다.

---

## 6. 검증 요약

| 게이트 | 명령 | 결과 |
|---|---|---|
| 계열 단위 및 실제 체크포인트 테스트 | `cargo test --profile test-fast --features cuda --lib -- models::lfm2_embedding models::llama_bidirec models::ministral3_embedding models::lfm2_tests --test-threads=1` | 36 통과, 0 실패, 연속 3회 |
| 임베딩 서브시스템 | `cargo test --profile test-fast --features cuda --lib embeddings:: -- --test-threads=1` | 63 통과, 0 실패 |
| 린트 | `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | 클린 |
| 타입 검사 | `cargo check --profile test-fast --features cuda --all-targets` | 클린 |
| 포맷 | `cargo fmt --all -- --check` | 클린 |
| CLI | 체크포인트 4개에 `mlxcel embed --json` | 단위 벡터, 표와 같은 마진 |
| HTTP | 체크포인트 4개에 `mlxcel-server` + `POST /v1/embeddings` | 단위 벡터, 표와 같은 마진 |

검증 호스트는 CUDA를 쓰는 Linux(GB10)다. 이슈의 `--features metal,accelerate` 툴체인 게이트는 `--features cuda` 등가물로 돌렸다. `metal`과 `accelerate`는 macOS 전용이라 여기서 빌드할 수 없다.

---

## 7. 변경 요약

| 파일 | 줄 수 | 역할 |
|---|---|---|
| `src/models/llama_bidirec.rs` | +268 | `LlamaBidirecModel`, weight sanitize, 어댑터 전용 디렉터리 거부 |
| `src/models/llama_bidirec_tests.rs` | +522 | sanitize, 어댑터 거부, 양방향성, 패딩, 실제 체크포인트 |
| `src/models/ministral3_embedding.rs` | +195 | `Ministral3EmbeddingModel`, offset 0의 attention scale, 슬라이딩 오버레이 |
| `src/models/ministral3_embedding_tests.rs` | +515 | attention scale 공식, 양방향성, 패딩, 두 변환본 |
| `src/models/lfm2_embedding.rs` | +222 | `Lfm2EmbeddingModel`, `embedding_norm` 접두사, late-interaction 거부 |
| `src/models/lfm2_embedding_tests.rs` | +589 | CLS 풀링, all-conv 대조군, 패딩, 실제 체크포인트 |
| `src/models/lfm2.rs` | +182 / -20 | `conv_causal`, 방향성 패딩, 패딩 곱셈자, `forward_hidden_bidirectional` |
| `src/models/lfm2_tests.rs` | +170 / -1 | 양방향 conv 임펄스 응답 테스트, MLX 가드 소급 적용 |
| `src/models/ministral3.rs` | +8 / -2 | `pub(crate)` 확대 두 건, 동작 변경 없음 |
| `src/embeddings/loader.rs` | +12 / -3 | 계열 arm 세 개 |
| `src/embeddings/real_checkpoint_tests.rs` | +12 / -5 | LFM2가 미포팅 목록을 떠나고, 목록에는 멀티모달 임베더 둘만 남음 |
| `src/models/mod.rs` | +3 | 모듈 선언 세 개 |
| `docs/embeddings.md` | +58 | 계열 절 세 개, 소스 맵, bf16 배치 기하 설명 |
| `docs/supported-models.md` | +3 | Embedding 행 세 개 |

---

## 8. 검증되지 않은 부분

- **슬라이딩 윈도우 Ministral 3 임베더.** `create_bidirectional_window_mask` 분기와 `RotatingKVCache` 선택은 구현되어 있고 도달 가능하지만, 공개된 두 체크포인트 어느 쪽도 `sliding_attention` 레이어를 선언하지 않아 실제 체크포인트 커버리지가 없다.
- **레퍼런스 구현과의 수치 비교.** 검증 호스트에 PyTorch나 `transformers` 설치가 없어 레퍼런스 프레임워크와 벡터를 대조하지 않았다. 게이트는 자기 일관성, 검색 의미론, 공개된 양자화 허용오차, 모델 카드가 선언한 접두사다.
- **Nemotron-3-Embed-8B.** 같은 코드 경로이며 내려받지도 돌리지도 않았다.
- **머지된 LLM2Vec 어댑터.** 어댑터 전용 거부는 합성 디렉터리로 검증했다. 실제 PEFT LLM2Vec 체크포인트는 머지된 것이든 아니든 다루지 않았다.
- **짝수 `L_cache`의 비causal 경로.** `conv_padding`이 처리하지만(`L_cache / 2`와 나머지로 나눈다) 공개된 모든 체크포인트가 `L_cache = 3`이라 홀수, 정확히 대칭인 경우만 테스트가 덮는다.

---

## 9. 후속 과제

1. 에픽의 병렬 웨이브가 머지된 뒤 `src/models/embedding_sanitize.rs`의 공유 `BACKBONE_ROOTS`에 `embedding_norm.`을 접어 넣어 지역 중복을 제거한다.
2. LFM2 / LFM2.5 ColBERT late-interaction 지원. 현재는 로드 시점 거부가 미지원이라고 보고한다. #1337이 다중 벡터 경로를 착지시키면 후보가 된다.
3. 생성 경로에서 패딩된 prefill이 도달 가능해질 때 short conv도 마스킹해야 하는지 검토한다. 지금 `None`인 이유는 LFM2 생성이 패딩 없는 시퀀스 하나를 돌리기 때문이지만, 이는 가정하기보다 단언해 둘 만한 불변조건이다.

---

## 참고

- 이슈 #1325, 에픽 #1348, 기반 PR #1408 / 이슈 #1353
- 머지된 형제 계열: #1410(SigLIP text), #1411(BERT, XLM-RoBERTa), #1412(ModernBERT), #1413(EmbeddingGemma, Qwen3-Embedding), #1414(ColIdefics3, ColQwen2.5)
- `docs/embeddings.md`, `docs/supported-models.md`
