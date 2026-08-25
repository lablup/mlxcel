# 기술 보고서: PR #1413 - EmbeddingGemma와 Qwen3-Embedding forward pass

**작성일**: 2026-08-26
**작성자**: mlxcel contributors
**상태**: 완료. 한 가지 레이아웃은 공개 체크포인트 대신 합성 체크포인트로 검증
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1413은 이슈 #1329가 요구한 두 decoder-backbone 임베딩 계열을 #1408에서 착지한 `/v1/embeddings` 기반 위에 구현한다. EmbeddingGemma는 기존 Gemma 3 레이어를 양방향 마스크로 돌리고 mean pooling 후 두 개의 `Dense` 투영을 적용한다. Qwen3-Embedding은 기존 Qwen3 레이어를 causal로 돌리고 마지막 실제 토큰을 pooling한다. 디코더를 새로 추가하지 않는다. 변경의 실체는 마스크, pooling, weight key 정규화, 그리고 `Qwen3Model::forward_impl`의 최소 분리뿐이다. 두 계열 모두 공개 체크포인트에서 로드되며 공개된 기준 수치를 재현한다.

---

## 1. 문제 정의

### 1.1 배경

EmbeddingGemma(`model_type: gemma3_text`, `architectures: ["Gemma3TextModel"]`, `use_bidirectional_attention: true`)와 Qwen3-Embedding(`model_type: qwen3`, `architectures: ["Qwen3ForCausalLM"]` + sentence-transformers `1_Pooling` 모듈)은 다운로드가 가장 많은 소형 임베더 두 종이고, 둘 다 mlxcel이 이미 구현한 디코더 백본 위에 올라간다. #1408은 공통 기반을 만들었다. `EmbeddingModel` 트레이트, pooling, 토크나이즈, 마이크로 배치, 길이 제한, `mlxcel_core::utils`의 마스크 빌더, `ModelType::Gemma3Embedding` / `ModelType::Qwen3Embedding` 탐지, 그리고 모든 arm이 "not yet supported"를 반환하던 계열 디스패처다. 이 PR은 그중 두 arm을 채운다.

### 1.2 기존 장애물

- **Gemma3Model은 head를 요구한다.** `Gemma3Model::from_weights`는 `lm_head`를 무조건 로드하고 `forward_with_caches_and_embeddings`는 `self.lm_head.forward(&h)`로 끝난다. EmbeddingGemma 체크포인트에는 `lm_head`가 아예 없으므로 생성용 생성자를 그대로 쓸 수 없다.
- **Gemma3Model은 항상 causal 마스크를 만든다.** 생성 경로의 prefill은 `create_causal_mask`와 `create_sliding_window_prefill_mask`를 구성한다. 양방향 임베더는 둘 다의 반대가 필요하고, 각 레이어는 추가로 `Attention.window_size`를 들고 있는데 Metal 4 어텐션 경로는 명시적 마스크가 주어져도 이 값을 causal window로 적용한다.
- **Qwen3Model은 logits만 노출한다.** `forward_impl`이 head로 끝나므로 임베더는 `[B, L, 151669]` 텐서를 만들었다가 버려야 했다.
- **두 체크포인트 모두 생성용과 키 체계가 다르다.** sentence-transformers export는 내부 `...Model`을 저장하므로 `embed_tokens.weight`, `layers.{i}....`, `norm.weight`가 생성자가 기대하는 `model.` 접두사 없이 도착한다. EmbeddingGemma는 여기에 더해 배포처에 따라 투영을 서로 다른 위치에 저장한다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 양방향 레이어에 causal 마스크가 들어감 | High. 임베딩이 조용히 틀리고 수치 게이트를 제외한 모든 게이트는 통과 | Medium |
| sliding 주기가 1만큼 어긋남 | High. 잘못된 레이어에 잘못된 마스크가 들어가지만 출력은 그럴듯하게 유지 | Medium |
| `dense.0`과 `dense.1`이 뒤바뀜 | High. 한 방향으로는 shape가 맞아떨어지고 값만 틀림 | Low |
| 동시 MLX 실행으로 테스트가 서로 간섭 | Medium. 게이트 수치가 증거로서 신뢰할 수 없게 됨 | 이 호스트에서는 High |

---

## 2. 기술적 선택과 그 이유

### 2.1 백본을 재사용하고 디코더를 새로 포팅하지 않는다

두 계열은 마스크가 다르고 head가 없는 기존 디코더다. 따라서 구현은 `gemma3::TransformerBlock`과 `Qwen3Model`을 직접 만들고 다른 부분만 더한다. 이 선택의 비용은 1.2의 장애물 두 가지를 새 모듈 내부가 아니라 이음새에서 처리해야 한다는 점이고, 이득은 Gemma 3나 Qwen3의 어텐션, 양자화, RoPE에 가해지는 모든 후속 수정이 임베더에 자동으로 도달한다는 점이다.

`src/models/gemma3.rs`는 전혀 수정하지 않았다. `Attention.window_size`가 이미 `pub`이라 임베딩 로더가 생성 후에 값을 지운다.

### 2.2 임베딩 경로가 모든 마스크를 소유한다

`Gemma3EmbeddingModel::forward_hidden`이 두 마스크를 직접 만들어 모든 레이어에 `Some(mask)`로 전달한다.

- full-attention 레이어: `create_bidirectional_padding_mask(attention_mask)`, shape `[B, 1, 1, L]`, 패딩 key만 차단
- sliding 레이어: `create_bidirectional_window_mask(attention_mask, sliding_window)`, shape `[B, 1, L, L]`, key가 패딩이거나 `|q - k| >= window`이면 차단

두 번째 형태는 `transformers`가 Gemma 3용으로 합성하는 양방향 sliding overlay(`kv_idx > q_idx - w` 그리고 `kv_idx < q_idx + w`)와 정확히 같다. 입력이 `sliding_window` 토큰 이하이면 두 마스크는 동등하므로 window는 512 토큰을 넘어서야 의미를 갖기 시작한다. 실제 체크포인트 게이트가 일부러 1749 토큰 문서를 임베딩하는 이유가 이것이다.

각 레이어의 `self_attn.window_size`는 로드 시 `0`으로 설정된다. 이는 마스크와 중복이 아니다. Metal 4 어텐션 경로에서는 `window_size`를 커널이 적용하므로, 양방향 마스크와 causal window를 함께 받은 레이어는 여전히 causal이 되고 그것도 macOS에서만 그렇게 된다. 기각한 대안은 `Attention::forward`에 호출별 플래그를 흘려보내는 것이었는데, 차가운 경로를 위해 생성의 뜨거운 경로를 건드리게 된다.

호출별 캐시는 offset 0의 `gemma3::Cache::Standard(KVCache::new())`이고 forward 안에서 만들어져 소멸한다. 여기서 rotating 캐시는 무의미하다. 호출 간 재사용이 없고, offset 0이 RoPE 위치와 마스크 key 축을 입력 길이에 맞춰 준다.

### 2.3 sliding 주기를 스칼라가 아니라 `layer_types`에서 도출한다

`gemma3::ModelArgs`는 `sliding_window_pattern`을 파싱한다. transformers 4.57은 그 키를 `_sliding_window_pattern`으로 이름을 바꾸고 `layer_types`를 권위 있는 값으로 만들었으므로, 최신 EmbeddingGemma config에서 파싱되는 값은 체크포인트에서 읽은 값이 아니라 계열 기본값 6이다. `mlx-community/embeddinggemma-300m-4bit`에서는 기본값이 우연히 맞는데, 바로 그 점이 이 값을 못박아야 하는 근거다. 1만큼 어긋난 주기도 로드되고 실행되며 조용히 틀린 벡터로만 드러난다.

`resolve_sliding_window_pattern`은 `layer_types`가 있으면 그것을 읽고 첫 `full_attention` 항목의 인덱스에서 주기를 도출한 뒤, 표본 검사가 아니라 목록 전체를 그 주기에 대해 검증한다. 불규칙한 목록은 문제 레이어를 이름으로 지목하는 로드 오류다. `full_attention` 항목이 하나도 없는 config는 마지막 레이어를 넘어서는 주기를 얻으므로 어떤 레이어도 full attention으로 취급되지 않는다. 폴백은 `sliding_window_pattern`, `_sliding_window_pattern`, 호출자 기본값 순이다.

이 해석은 레이어 생성 전에 일어난다. `gemma3::layer_rope_params`가 같은 주기로 `rope_theta`와 `rope_local_base_freq` 중 하나를 고르기 때문이다. 생성 후에 주기를 바로잡았다면 RoPE base가 틀린 채로 남았을 것이다.

### 2.4 내부를 노출하는 대신 `Qwen3Model::forward_impl`을 분리한다

`forward_hidden`은 임베딩 조회, 레이어 루프, 최종 norm까지를 담당하고 `forward_impl`은 그것을 호출한 뒤 head를 적용한다. 생성 출력은 구조적으로 불변이며, 합성 2레이어 모델에서 `forward_hidden` 위에 tied head를 다시 얹어 토큰 단위로 정확히 일치하는지 요구하는 테스트가 이를 증명한다. 대안이던 `layers`와 `norm`의 공개, 또는 임베더에서 루프 복제는 두 경로가 갈라지도록 방치했을 것이다.

같은 진입점을 Qwen3 생성형 리랭커(#1356)가 필요로 하므로 doc comment가 두 소비자를 함께 명시한다.

### 2.5 공유 weight key 정규화 하나

디코더 백본의 sentence-transformers export는 생성용 체크포인트와 기계적으로 세 가지가 다르고, 생성용 백본을 재사용하는 모든 계열이 그 셋을 모두 되돌려야 한다. `src/models/embedding_sanitize.rs`가 고정된 순서로 처리한다. `{N}_Dense.linear.*`를 폴더 순위에 따라 `dense.{k}.*`로 접고, `lm_head.*`와 `head.*`를 버리고, 벌거벗은 `embed_tokens.` / `layers.` / `norm.`에 `model.`을 붙인다. 순서가 중요하다. `Dense` 키는 백본 루트가 아니므로 접두사 단계 전에 이름을 바꿔야 하며, 그렇지 않으면 `layers.0...`이 옮겨가는 동안 `2_Dense.linear.weight`는 그대로 남는다.

폴더 번호는 투영 인덱스가 아니라 sentence-transformers 모듈 위치(`2_Dense`는 `1_Pooling` 다음)이므로 직접 쓰지 않고 순위로 환산한다. 공개된 두 EmbeddingGemma 레이아웃이 동일한 `dense.0` / `dense.1` 키로 수렴하는 이유가 이것이다. 이 함수는 mlx 변환본에서는 no-op이고 멱등이므로 계열은 조건 없이 호출해도 된다.

### 2.6 로드 시점에 `Dense` 사슬을 검증한다

`load_dense_stack`은 `dense.0`, `dense.1`, ...를 순회하며 각 투영의 입력 폭이 이전 단계의 출력과 같은지를 백본 hidden size에서 출발해 요구한다. 뒤바뀐 쌍은 투영을 이름으로 지목하는 로드 오류다. 이 검사가 없으면 `768 -> 3072 -> 768`과 그 역방향이 모두 로드되고 forward도 돌아가며 값만 틀리는데, 이는 알아차리기 가장 어려운 부류의 실패다.

양자화 체크포인트에서 입력 폭을 읽는 데는 주의가 필요하다. weight는 입력 축을 따라 패킹되므로(4비트에서 `dense.0.weight`는 `[3072, 96]`), `linear_features`는 폭을 `scales` 그룹화(group size 64에서 `[3072, 12]`, 따라서 768)에서 읽고 dense 텐서일 때만 weight의 두 번째 차원으로 폴백한다.

스택은 고정 쌍이 아니라 `Vec`이므로 `Dense` 모듈이 전혀 없는 양방향 Gemma 3 export도 로드되고 `embedding_dim == hidden_size`를 보고한다.

### 2.7 Qwen3 임베딩 경로에서 `tie_word_embeddings`를 강제한다

임베더는 최종 norm에서 멈추므로 head는 결코 적용되지 않는다. `Qwen3Model::from_weights` 전에 플래그를 세우면 untied `lm_head`가 메모리에서 빠지고(0.6B 체크포인트에서 그 텐서는 151669 x 1024이다) 이 경로가 읽지도 않을 head 때문에 생성자가 실패하는 일도 막는다.

### 2.8 투영할 대상이 있을 때만 체크포인트 dtype으로 되돌린다

`pool`은 의도적으로 f32를 반환한다. 활성화 dtype이 무엇이든 축약을 전정밀도로 수행하기 위해서다. `Dense` 투영은 체크포인트 dtype을 들고 있으므로 그 matmul 전에 pooling 결과를 되돌린다. 투영이 없으면 f32 벡터가 f16을 불필요하게 왕복하지 않고 엔진으로 바로 간다.

### 2.9 prompt 접두사는 호출자 쪽에 두되 opt-in 훅 하나를 제공한다

서버 쪽에서 주입하는 것은 없다. 입력은 보낸 그대로 임베딩된다. `Qwen3EmbeddingModel::format_text`는 요청이 `instruction`을 줄 때만 질의를 `Instruct: {task}\nQuery: {query}`로 감싸고 그 외에는 항등이며, 이는 `EmbeddingModel` 트레이트의 doc comment가 이 계열에 대해 이미 약속한 동작이다. `instruction`은 요청의 모든 입력에 적용되므로 질의와 문서가 섞인 배치는 요청을 둘로 나누거나 질의를 본문에서 직접 포맷해야 하고, `docs/embeddings.md`가 그렇게 적는다. EmbeddingGemma는 항등을 유지하고 `config_sentence_transformers.json`의 일곱 가지 접두사를 표로 문서화한다.

---

## 3. 구현 세부

### 3.1 EmbeddingGemma forward

```
h = embed_tokens[input_ids] * sqrt(hidden_size)
full    = create_bidirectional_padding_mask(attention_mask)      # [B, 1, 1, L]
sliding = create_bidirectional_window_mask(attention_mask, 512)  # [B, 1, L, L]
for i, layer in layers:                                          # window_size는 0으로 지움
    h = layer.forward(h, offset 0의 새 KVCache,
                      if (i + 1) % pattern == 0 { full } else { sliding })
h      = norm(h)                                                 # GemmaRMSNorm
pooled = pool(h, attention_mask, Mean)                           # f32
pooled = dense.1(dense.0(astype(pooled, 체크포인트 dtype)))       # 768 -> 3072 -> 768
```

L2 정규화와 `dimensions` 절단은 이후 엔진이 수행한다.

### 3.2 Qwen3-Embedding forward

```
mask   = create_causal_padding_mask(attention_mask, 0)           # [B, 1, L, L]
h      = model.forward_hidden(input_ids, None, 새 캐시, Some(mask))
pooled = pool(h, attention_mask, LastToken)
```

두 번째 코드 경로 없이 이것이 옳게 동작하는 이유는 오른쪽 패딩이다. pooling 위치가 마지막 실제 토큰이고 패딩은 그 뒤에 놓이며 패딩 key는 모든 실제 query에 대해 차단되므로, 패딩된 행이 단독 실행을 그대로 재현한다.

### 3.3 weight key 매핑

| 공개된 형태 | 정규화 후 |
|-------------|-----------|
| `embed_tokens.weight` (sentence-transformers) | `model.embed_tokens.weight` |
| `model.embed_tokens.weight` (mlx) | 그대로 |
| `2_Dense/model.safetensors`의 `linear.weight` | `dense.0.weight` |
| `3_Dense/model.safetensors`의 `linear.weight` | `dense.1.weight` |
| `dense.0.*`, `dense.1.*` (mlx) | 그대로 |
| `lm_head.*`, `head.*` | 삭제 |

### 3.4 등록

`build_family_model`에서 `Gemma3Embedding`과 `Qwen3Embedding` arm만 바뀌었고 다른 계열의 "not yet supported" arm은 손대지 않았다. 형제 포팅이 병렬로 착지하던 중이었기 때문에 이 점이 중요하다. 탐지, `ModelType` variant, `mlxcel arch` 항목은 #1408에서 이미 존재하므로 추가 변경 없이 `mlxcel arch`가 두 계열을 `Embedding` 아래에 나열하고 `mlxcel list`가 두 체크포인트를 보여준다.

---

## 4. 테스트 전략

### 4.1 shape 검사로는 잡을 수 없는 두 성질

합성 테스트는 결정론적 가중치로 16폭 Gemma 3와 16폭 Qwen3를 만들므로 체크포인트도 특정 장치도 필요 없다.

- `bidirectional_prefill_is_not_causal`: 96 토큰에서 마지막 토큰을 바꾸면 첫 토큰의 hidden state가 움직여야 한다. causal 마스크에서는 그 차이가 정확히 0이 된다.
- `causal_prefill_is_causal`: Qwen3용 거울상. 마지막 토큰을 바꿔도 이전 hidden state는 1e-6 이내로 변하지 않아야 하고, 바뀐 토큰 자신은 움직여야 한다.
- `global_layers_use_padding_mask_and_sliding_layers_use_window`: 동일한 가중치로 만든 1레이어 모델 두 개, 하나는 full-attention, 다른 하나는 window 4의 sliding. 6칸 떨어진 토큰이 첫 모델에서는 첫 토큰 상태를 움직이고 둘째 모델에서는 움직이지 않는다.
- `forward_hidden_then_head_matches_forward_impl`: 리팩터링 가드, 토큰 단위 일치.
- `padding_invariance`와 `last_token_pool_uses_appended_eos`: 오른쪽 패딩된 2행 배치가 패딩 없는 단일 행 결과를 재현한다.

### 4.2 gated 체크포인트 없이 레이아웃 동등성 확인

EmbeddingGemma는 gated인 `google/embeddinggemma-300m` 약관에 동의하지 않으면 mlx 변환본만 내려받을 수 있으므로, `sentence_transformers_subfolder_layout_loads_from_disk_and_matches_the_mlx_layout`가 체크포인트를 직접 만들어낸다. 같은 합성 가중치를 sentence-transformers 표기(벌거벗은 백본 루트, 쓰이지 않는 `lm_head`, `2_Dense/`와 `3_Dense/` 모듈 폴더의 투영)로 기록하고 실제 `Gemma3EmbeddingModel::load(dir)` 경로로 로드한 뒤, mlx 방식 인메모리 빌드와 비트 단위로 같은 임베딩이 나오는지 확인한다. 이는 `load_weights_from_dir_with_subfolders`, 정규화기, 생성자를 함께 구동하며 키 매핑 단위 테스트만으로는 닿지 않는 영역이다.

### 4.3 테스트 쪽 MLX 동시성

`EmbeddingModel`은 단일 스레드로 문서화되어 있고 제품은 임베딩 워커를 통해 그것을 지키므로 이 위험은 테스트 쪽에만 있다. 한 프로세스 안의 동시 MLX forward pass는 이 트리에서 두 가지로 관찰되는 간섭을 일으킨다. CUDA graph capture가 프로세스를 중단시키고(`cudaStreamEndCapture ... operation failed due to a previous error during capture`, 넓은 필터에서 대략 4회 중 3회 재현), 결과가 흔들려 형제 유닛이 한 배치 안의 바이트 단위로 동일한 두 행에서 cosine 1.0 대신 0.999912를 측정했다.

여기서 모델을 만들거나 forward pass를 돌리는 모든 테스트는 프로세스 전역 `mlx_test_guard()`를 잡는다. 기존 `llama4_helpers_tests::test_guard` 패턴을 따르고 poisoned lock을 복구해 패닉한 테스트 하나가 단독으로 실패하게 한다. 이 가드는 다른 모듈의 MLX 작업까지 직렬화할 수는 없으며, 그래서 이 저장소가 정의하는 모든 게이트(`make verify-test`, `make verify-test-cuda`, `make test-fast`)가 이미 `--test-threads=1`을 전달한다. Makefile은 그것을 정돈이 아니라 필수 조건으로 기록하고 있다. 아래 수치는 모두 그 구성에서 나왔고, 실제 체크포인트 게이트는 값을 출력하므로 반복 실행이 통과 여부 한 비트가 아니라 산포를 보여준다.

---

## 5. 실제 체크포인트 결과

Linux, GB10, CUDA. 단일 스레드 연속 3회 실행이 비트 단위로 동일한 수치를 냈다.

### `mlx-community/embeddinggemma-300m-4bit` (4-bit, 768폭, `max_length` 2048)

| 게이트 | 관측값 | 요구 조건 |
|--------|--------|-----------|
| 벡터 폭과 norm | 768, norm 1.0 | 768차원 단위 벡터 |
| 질의 vs Mars | 0.642725 | 최소 0.1 차이로 최고 |
| 질의 vs Venus | 0.323360 | 매칭 문서보다 낮음 |
| 질의 vs Jupiter | 0.329467 | 매칭 문서보다 낮음 |
| 매칭 마진 | 0.313 | 최소 0.1 |
| `dimensions: 256` | 폭 256, norm 1.0, Mars 0.693574 > Venus 0.422158, Jupiter 0.412816 | 단위 벡터, 순위 불변 |
| 배치 내 동일 입력 | cosine 1.000000000 | 1e-6 이내 1.0 |
| 1749토큰 문서, 단독 vs 패딩 배치 | `max_abs_diff` 0.0 | 1e-3 이내 |
| 무관한 문장 | 0.0271 | 0.5 미만 |

### `Qwen/Qwen3-Embedding-0.6B` (bf16, 1024폭, `max_length` 8192)

| 게이트 | 관측값 | 요구 조건 |
|--------|--------|-----------|
| 질의-문서 행렬 | `[[0.766633, 0.143439], [0.136450, 0.600714]]` | model card `[[0.7646, 0.1414], [0.1355, 0.6000]]` 대비 2e-2 이내 |
| 최대 편차 | 0.00204 | 2e-2 미만 |
| 배치 내 동일 입력 | cosine 1.000000119 | 1e-6 이내 1.0 |
| 단독 vs 패딩 배치 | `max_abs_diff` 0.0 | 1e-3 이내 |
| 비유한 값 | 없음 | 없음 |
| 무관한 문장 | 0.1568 | 0.5 미만 |
| `dimensions: 256`, `encoding_format: base64` | 256폭 단위 벡터로 디코드 | 유효 |

검증 호스트에는 PyTorch나 `transformers` 설치가 없으므로, 이슈가 지시한 대로 Qwen3-Embedding의 기준은 체크포인트의 공개 model card다. 이 보고서의 어떤 수치도 로컬 참조 구현에서 계산되지 않았고 추정된 값도 없다.

두 계열 모두 `mlxcel embed`와 `mlxcel-server` + `POST /v1/embeddings`에서 같은 값을 반환하며, 각각 `/v1/models`가 서빙 중인 임베딩 모델을 나열한다.

---

## 6. 검증 요약

| 명령 | 결과 |
|------|------|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo check --profile test-fast --features cuda --all-targets` | exit 0 |
| `cargo test ... --lib -- --test-threads=1 models::gemma3_embedding models::qwen3_embedding models::embedding_sanitize` | 22 passed, 0 failed, 3회 반복, 동일 게이트 수치 |
| `cargo test ... --lib embeddings:: -- --test-threads=1` | 62 passed, 0 failed |
| `cargo test ... --lib models::gemma3 -- --test-threads=1` | 23 passed, 5 ignored, 0 failed |
| `cargo test ... --lib models::qwen3 -- --test-threads=1` | 59 passed, 11 ignored, 0 failed |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |

이슈의 수용 기준은 macOS `metal,accelerate` feature set을 명시한다. 이 호스트에서는 CUDA 게이트가 그 등가물이며 실제로 실행된 것도 그것이다. macOS 게이트는 CI의 몫이다.

성능 수치는 보고하지 않는다. 에픽이 조용한 머신에서 별도로 성능 패스를 돌린다.

---

## 7. 변경 요약

| 파일 | 변경 |
|------|------|
| `src/models/gemma3_embedding.rs` (신규, 327) | `Gemma3EmbeddingModel`: 양방향 마스크, mean pooling, `Dense` 스택, 주기 해석 |
| `src/models/gemma3_embedding_tests.rs` (신규, 635) | 합성 마스크 및 pooling 게이트, 디스크 subfolder 레이아웃 테스트, 실제 체크포인트 게이트 |
| `src/models/qwen3_embedding.rs` (신규, 155) | `Qwen3EmbeddingModel`: causal + padding 마스크, last-token pooling, instruction 포맷 |
| `src/models/qwen3_embedding_tests.rs` (신규, 381) | 리팩터링 가드, causality 게이트, 패딩 게이트, 실제 체크포인트 게이트 |
| `src/models/embedding_sanitize.rs` (신규, 160) | 공유 weight key 정규화와 `linear_features` |
| `src/models/embedding_sanitize_tests.rs` (신규, 140) | 두 레이아웃, 멱등성, 알 수 없는 서브모듈 처리 |
| `src/models/embedding_test_support.rs` (신규, 229) | 결정론적 가중치, shaped safetensors writer, 체크포인트 탐색, MLX 가드 |
| `src/models/qwen3.rs` (+24/-4) | `forward_impl`을 `forward_hidden` + head로 분리 |
| `src/embeddings/loader.rs` (+6/-2) | 두 계열 arm 생성 |
| `src/embeddings/real_checkpoint_tests.rs` (+5/-3) | Qwen3-Embedding이 미포팅 목록에서 빠짐 |
| `src/models/mod.rs` (+5) | 모듈 선언 3개 |
| `docs/embeddings.md` (+64/-21) | 계열 노트, prompt 접두사, 소스 맵. 삭제분은 rebase 중 SigLIP과 ModernBERT의 계열 절을 하나의 `Family notes` 제목 아래로 합친 결과이며 내용이 사라진 것은 아니다 |
| `docs/supported-models.md` (+2) | Embedding 행 2개 |

13개 파일, 2133 추가, 30 삭제. 이 작업이 진행되는 동안 머지된 SigLIP(#1410)과 ModernBERT(#1412) 포팅 위에 올려져 있다.

---

## 8. 검증되지 않은 부분

- **공개된 sentence-transformers EmbeddingGemma 체크포인트.** `google/embeddinggemma-300m`은 gated다. subfolder 레이아웃은 그 표기로 기록한 합성 체크포인트로 증명했을 뿐 공개 산출물로 증명하지 않았다. 따라서 비양자화 EmbeddingGemma도 여기서 end-to-end로 로드된 적이 없다. 다만 같은 코드 경로가 dense인 Qwen3-Embedding 체크포인트를 서빙한다.
- **macOS와 Metal.** 여기 모든 실행은 Linux와 CUDA에서 이루어졌다. `window_size = 0` 처리는 Metal 4 어텐션 경로를 위해 존재하며 이 호스트에서는 간접적으로만 구동된다. CI의 macOS 게이트가 확인해 줄 부분이다.
- **더 큰 Qwen3-Embedding 변형.** 4B와 8B는 크기만 다른 같은 코드이고 이슈의 범위 밖이며 로드하지 않았다.
- **256 외의 Matryoshka 폭.** 512와 128도 학습된 폭이고 같은 `truncate_dimensions` 경로를 지나지만, 순위 요구 조건에 대해 측정한 것은 256뿐이다.

---

## 9. 후속 과제

- #1356(Qwen3 생성형 리랭커)은 `Qwen3Model::forward_hidden`을 그대로 사용할 수 있다.
- 검증 호스트에서 `google/embeddinggemma-300m`을 사용할 수 있게 되면 `local_embedding_checkpoints_detect_to_their_families`와 EmbeddingGemma 게이트에 추가해 공개 subfolder 레이아웃을 직접 덮는다.
- 에픽의 성능 패스에는 긴 입력 EmbeddingGemma 사례를 포함해야 한다. sliding window는 512 토큰을 넘어야 관여하고 두 마스크의 shape가 다르기 때문이다(`[B, 1, 1, L]` 대 `[B, 1, L, L]`).

---

## 참고

- 이슈 #1329, 에픽 #1348
- PR #1408(임베딩 기반), 이슈 #1353
- `docs/embeddings.md`, `docs/supported-models.md`
- `mlx-community/embeddinggemma-300m-4bit`, `Qwen/Qwen3-Embedding-0.6B`
