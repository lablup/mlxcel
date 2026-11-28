# 기술 보고서: PR #1735 - feat(vlm): Cohere Compass (North-Micro-Vision) VLM 추가

**날짜**: 2026-09-10
**작성**: mlxcel maintainers
**검토**: 구현 리뷰 사이클
**상태**: 완료 (GB10 / CUDA에서 transformers 오라클과 대조 검증. 인수 조건에 적힌 metal+accelerate 명령은 이 호스트에서 실행할 수 없어 실행하지 않음)
**언어**: Rust
**위험도**: 중간 (신규 VLM 계열 하나. 기존 계열의 동작을 바꾸는 공용 경로 변경 없음)

---

## 요약

`model_type: "cohere_compass"`(CohereLabs North-Micro-Vision-Instruct, 2.5B)는 Qwen3-VL deepstack 비전 타워 뒤에 Command 계열의 병렬 블록 텍스트 디코더를 붙인 모델이다. 비전 쪽은 거의 전부 재사용이다. 인코더, 이미지 전처리기, `<|IMAGE_PAD|>` 확장, MRoPE 위치 인덱스 규칙, 가중치 프리픽스 리맵 모두 Qwen3-VL의 것을 그대로 쓴다.

새로 쓴 것은 디코더이고, 그 안의 한 가지가 이 보고서를 쓰는 이유다. 이슈 #1354는 sliding 레이어에 Qwen3-VL의 `InterleavedMRoPE`를 쓰라고 명시했다. 틀린 명세다. 체크포인트에 `mrope_interleaved: true`가 들어 있기는 하지만, 상류 transformers는 그 키를 `ignore_keys_at_rope_validation`에 올려두고 읽지 않는다. `CohereCompassRotaryEmbedding`은 `inv_freq`를 먼저 재배열한 뒤 축을 연속 구간 `[H, W, T]`로 배정한다. transformers 클래스 자체와 대조하면, 이슈가 지정한 interleave 방식은 `cos`에서 최대 1.96까지 어긋나고 실제 구현한 테이블은 2.2e-7 안에 든다. 이슈대로 만들었다면 로드되고 실행되며 그럴듯한 쓰레기를 뱉는 모델이 나왔을 것이다.

---

## 1. 이 계열이 무엇인가

### 1.1 텍스트 디코더 (`cohere_compass_text`)

Command 계열 디코더. 28층, hidden 2048, Q 16 / KV 8 헤드에 `head_dim` 128, SwiGLU `intermediate_size` 6144, vocab 262144, 임베딩 타이.

```
h = embed_tokens(ids)
for i in 0..28:
    n = LayerNorm(h)                        # weight만, eps 1e-5
    h = h + Attn_i(n) + SwiGLU_i(n)         # 병렬 블록, 같은 norm 출력을 둘 다 읽는다
    if i < len(deepstack) and 이미지가 있으면:
        h[비주얼 위치] += deepstack[i]
h = LayerNorm(h)
logits = embed_tokens.as_linear(h) * 0.25   # logit_scale
```

체크포인트 어디에도 `post_attention_layernorm`이 없다. `layer_types`는 `sliding_attention` 세 층마다 `full_attention` 한 층이 오는 패턴이라 full 레이어는 인덱스 3, 7, 11, 15, 19, 23, 27에 있다.

### 1.2 위치 인코딩은 레이어 타입별이고, 한쪽은 아예 없다

`rope_parameters`는 레이어 타입별 맵이다.

```json
{
  "sliding_attention": {"mrope_interleaved": true, "mrope_section": [24, 20, 20],
                        "rope_type": "default", "rope_theta": 50000},
  "full_attention": null,
  "rope_theta": 10000.0, "rope_type": "default"
}
```

여기서 JSON `null`이 의미를 갖는다. full_attention 일곱 층에는 **위치 인코딩이 전혀 없다**는 뜻이지 "블록 레벨 `rope_theta`로 폴백하라"는 뜻이 아니다. 두 번째로 읽으면 회전 없이 학습된 NoPE 일곱 층을 회전시키게 되고, 그 결과 모델은 어디서도 시끄럽게 실패하지 않는다. `CompassTextConfig::rope_for_layer_type`은 항목이 `null`이거나, 다른 레이어 타입 항목이 있는데 자기 항목만 없는 경우 `Ok(None)`을 돌려주고, `CompassAttention::rope`는 `Option<CompassMRoPE>`라서 NoPE 레이어는 비활성화된 테이블이 아니라 테이블 자체를 갖지 않는다.

### 1.3 비전 타워

`cohere_compass_vision`은 `Qwen3VLVisionConfig`와 키가 하나씩 그대로 대응한다. depth 27, hidden 1152, 헤드 16, patch 16, temporal patch 2, spatial merge 2, `out_hidden_size` 2048, `num_position_embeddings` 2304(이미지마다 보간되는 48x48 격자), `deepstack_visual_indexes` `[8, 16, 24]`. `Qwen3VLVisionEncoder`가 수정 없이 파싱하고 실행하며, `forward_with_grid`는 디코더가 필요로 하는 `(features, deepstack_features)` 쌍을 이미 그대로 돌려준다.

---

## 2. 이슈가 틀린 MRoPE

### 2.1 상류가 실제로 계산하는 것

`CohereCompassRotaryEmbedding.compute_default_rope_parameters`는 Qwen3-VL이 하지 않는 일을 한다. 축을 고르기 전에 주파수 목록을 **미리 재배열**한다.

```python
inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))     # 자연 순서, 64개
hw_dim   = mrope_section[0] + mrope_section[1]           # 24 + 20 = 44
t_dim    = mrope_section[2]                              # 20
inv_freq_3d[:hw_dim] = cat([inv_freq[:-t_dim][0::2], inv_freq[:-t_dim][1::2]])
inv_freq_3d[-t_dim:] = inv_freq[-t_dim:]
```

이어서 `recomposition_frequencies`가 `[3, B, L, 64]` 주파수 텐서를 `mrope_section`으로 쪼갠 뒤 `i`번째 조각을 축 `(i + 1) % 3`, 즉 H, W, T 순서로 가져온다. 구간은 **연속이고 순서는 `[H, W, T]`**이며, `cos`/`sin`은 `rotate_half`가 기대하는 절반 복제(split) 배치다.

공개된 `[24, 20, 20]`에 대해 채널별로 풀어 쓰면 이렇다.

| 채널 | 축 | 주파수 |
|---|---|---|
| 0..23 | H | `inv_freq[0, 2, ..., 42]` 다음에 `inv_freq[1]`, `inv_freq[3]` |
| 24..43 | W | `inv_freq[5, 7, ..., 43]` |
| 44..63 | T | `inv_freq[44..63]` |

공개 구간이 `[24, 20, 20]`인데 코드의 기본값은 `[22, 22, 20]`이라서 H/W 경계가 재배열 중간에 떨어진다. `[22, 22, 20]`이면 짝수/홀수로 깔끔하게 갈린다. 이 비대칭은 상류의 것이고, 정리하지 않고 그대로 재현했다.

### 2.2 이슈가 지정한 것

같은 `[24, 20, 20]`에 Qwen3-VL의 interleave를 적용하면 자연 순서 주파수에서 T가 채널 `{0, 3, ..., 57}`과 `{60..63}`(24개), H가 `{1, 4, ..., 58}`, W가 `{2, 5, ..., 59}`를 갖는다. 분할도 다르고 주파수도 다르며 넓은 구간을 가져가는 축도 다르다. Compass 테이블의 라벨만 바꾼 것이 아니다.

### 2.3 측정

`rope_check.py`는 실제 `config.json`으로 `CohereCompassRotaryEmbedding`을 만들고 `(t, h, w) = (7, 11, 13)`에서의 `cos`/`sin`을 두 방식과 비교한다.

```
mlxcel formulation  max|cos diff|: 2.2290754853049322e-07
mlxcel formulation  max|sin diff|: 3.8017985570792945e-07
qwen3-vl interleave max|cos diff|: 1.9554328814065203
```

2.2e-7은 f32 반올림이다. 1.96은 다른 함수다.

### 2.4 텍스트 전용 경로에 미치는 영향

`qwen3_vl.rs`에는 이미지가 없는 시퀀스에서 MRoPE 테이블 대신 `fast_rope`를 쓰는 빠른 경로가 있다. 세 축이 같은 위치를 갖게 되므로 interleave 테이블이 평범한 1-D RoPE로 무너진다는 논리다. 이 논리는 **여기에 옮겨오지 않는다**. Compass의 `inv_freq_3d`는 자연 순서의 치환이라, 위치가 같아도 채널 `j`는 여전히 `inv_freq_3d[j]`를 쓰는데 `fast_rope`는 `inv_freq[j]`를 쓴다. 그래서 이 계열에는 빠른 경로를 일부러 두지 않았고 모듈 헤더에 그 이유를 적어두었다. 텍스트 전용 시퀀스는 세 행이 모두 같은 상태로 같은 테이블에 들어갈 뿐이다.

---

## 3. 슬라이딩 윈도우: 링 버퍼가 아니라 마스크

이슈는 sliding 레이어에 `RotatingKVCache(max_size = 4096)`를 쓰라고 했다. 이 저장소는 슬라이딩 윈도우를 그렇게 처리하지 않는다. `cohere2.rs`, `cohere2_moe.rs`, `olmo3.rs`, `gemma3n.rs` 모두 조밀한 `KVCache`를 유지하고 프리필 마스크 두 개와 융합 SDPA의 `window_size` 인자로 창을 강제한다. 애초에 `LanguageModel::make_caches`가 `Vec<KVCache>`를 돌려주므로 회전 캐시는 그 자리에 표현할 수도 없다.

Compass도 이미 굳어진 계약을 따른다. `prefill_masks`가 full 레이어에는 `create_causal_mask(L, live_len)`, sliding 레이어에는 `create_sliding_window_prefill_mask_dense(L, live_len, 4096)`을 만들고, 단조 증가하는 `offset`이 아니라 `live_len()`으로 크기를 잡는다. `--max-kv-size` 트림이 마스크를 반환된 K/V보다 넓게 만들 수 없게 하려는 것이다(이슈 #419). `CompassAttention::attend`는 창이 마스크를 잘랐을 때 K/V를 마스크의 키 축까지 슬라이스하는데, `cohere2.rs`와 같은 처리다(이슈 #408 / #419). 디코드 폭에서는 두 마스크가 모두 `None`이고 창은 `causal_attention`의 `window_size`에서 나온다.

호출자가 넘긴 `mask`는 일부러 무시한다. 이 모델은 한 스텝에 모양이 다른 마스크 두 개가 필요해서 항상 자기 캐시에서 직접 만들고, 이는 `cohere2.rs`와 `olmo3.rs`가 하는 방식과 같다.

---

## 4. 서버 경로

이 저장소에는 VLM 특유의 함정이 있다. CLI는 자기 캐시를 직접 만들기 때문에, `sequence_state_layout()`을 재정의하지 않고 `supports_batching()`이 false이면 스케줄러가 빈 캐시 벡터를 받아 디코더 레이어가 하나도 돌지 않는데도 CLI 테스트는 전부 통과할 수 있다. `CohereCompassTextModel`은 기본 `supports_batching()`(true)을 그대로 두므로 기본 `dense_kv_cache` 레이아웃을 쓰고, `CohereCompassModel`은 `forward_batched_with_context_and_ids`를 공용 `forward_batched_with_seq_ids_dispatch`로 재정의해 VL과 텍스트가 섞인 배치에서도 각 행이 자기 MRoPE 항목을 찾게 한다. `QwenVlRuntime` 구현이 DeepStack 형태의 비전 캐시와 시퀀스별 MRoPE bind / take / install 훅을 들고 있다.

이건 추론이 아니라 검증이다. 5절에 `image_url` 파트를 포함한 `/v1/chat/completions` 실행 결과를 기록했다.

---

## 5. 검증

### 5.1 오라클 구성

이 호스트에 공개 bf16 체크포인트가 없고 mlx-vlm에도 이 계열 구현이 없다. 그래서 오라클은 `cohere_compass`를 담고 있는 transformers `5.18.0.dev0`이며, mlxcel이 읽는 것과 **같은 가중치**를 읽게 했다. `dequantize.py`가 저장소 자신의 공식(`w = q * scale + bias`, u32마다 낮은 비트부터 채워진 값, 그룹 64)으로 MLX 4bit affine 텐서를 풀어 CohereLabs 키 배치의 float32 체크포인트 하나를 만든다. mlxcel의 `remap_qwen3_vl_weights`가 그 배치를 자기 키 공간으로 옮기므로 양쪽이 같은 파일 하나를 읽고, 차이가 나면 그건 양자화가 아니라 순전파의 차이다.

### 5.2 결과

아래 수치는 모두 GB10(sm_121)에서 `--profile test-fast --features cuda`로, `mlx-community/North-Micro-Vision-Instruct-4bit`와 그것을 역양자화한 float32 체크포인트를 대상으로 측정했다.

**공개된 두 가중치 배치가 모두 로드된다.** 4bit 변환본은 `language_model.model.*` / `vision_tower.*`이고, 역양자화 파일은 CohereLabs의 `model.language_model.*` / `model.visual.*` 배치로 썼다. 아래 실행은 전부 양쪽에서 돌렸으므로 두 프리픽스는 단위 픽스처가 아니라 실제 실행으로 덮인다.

**로터리 테이블은 상류 클래스와 일치하고, 이슈가 지정한 방식은 일치하지 않는다.** `rope_check.py`가 실제 `config.json`으로 `CohereCompassRotaryEmbedding`을 만들고 `(t, h, w) = (7, 11, 13)`에서 평가한 결과다.

```
mlxcel formulation  max|cos diff|: 2.2290754853049322e-07
mlxcel formulation  max|sin diff|: 3.8017985570792945e-07
qwen3-vl interleave max|cos diff|: 1.9554328814065203
```

**MRoPE 위치 인덱스는 `get_rope_index`와 비트 단위로 같다.** `rope_index_check.py`가 `qwen_vl_mrope_positions_from_tokens`를 파이썬으로 옮겨 실제 이미지 프롬프트에서 모델 자신의 메서드와 비교한다.

| 프롬프트 | 격자 | 길이 | 동일 | 최대 | rope_delta |
|---|---|---|---|---|---|
| 이미지 1장 | `[1, 28, 28]` | 210 | 예 | 27 | -182 (양쪽) |
| 이미지 2장 | `[1, 28, 28]`, `[1, 14, 14]` | 261 | 예 | 36 | -224 (양쪽) |

**텍스트 전용은 토큰 단위로 일치한다.** 서버 `/v1/chat/completions`에 `logprobs: true`를 주면 실제로 방출된 토큰이 나오므로, 디코딩된 문자열 비교가 아니라 토큰 비교다.

| 실행 | 토큰 수 | 오라클과 동일 |
|---|---|---|
| f32 서버, "Summarize the history of the Apollo program..." | 32 | 예, 32/32 |
| 4bit 서버, 같은 프롬프트 | 32 | 예, 32/32 |

CLI도 세 프롬프트(그중 하나는 79토큰 프롬프트)에서 양쪽 체크포인트 모두 같은 문자열을 내고, 서버 텍스트는 CLI와 일치한다.

**서버가 실제로 디코더를 돌린다.** 텍스트 요청의 `prompt_tokens`는 18, 이미지 요청은 210으로 둘 다 오라클과 정확히 같고, `completion_tokens`는 0이 아니며, 이미지 요청은 25토큰 뒤 EOS로 멈춘다. `sequence_state_layout()`을 재정의하지 않아 스케줄러가 빈 캐시 벡터를 받고 모델이 조용히 아무것도 내지 않는, 이 계열의 알려진 실패 모드가 아니라는 뜻이다.

**이미지 경로는 올바르지만 토큰 단위로 일치하지는 않는다.** 디코더 입력까지는 오라클과 전부 일치한다. 격자 `[1, 28, 28]`에서 이미지 토큰 196개, 프롬프트 210토큰, MRoPE 위치 비트 동일. 생성 토큰은 두 스텝 일치한 뒤 스텝 2에서 갈린다.

```
f32 server image1:  mlxcel=25 oracle=32 identical=False common_prefix=2
   first divergence: ' features' vs ' displays'
4bit server image1: 결과 동일, 바이트 단위로
```

두 경우 모두 mlxcel은 무관한 토큰이 아니라 오라클의 2순위 토큰을 고르지만, 격차가 무시할 수준은 아니다. 이미지 1장 스텝 2에서는 `' displays'` -1.5065 대 `' features'` -1.7449로 0.238 nat, 이미지 2장("How many shapes are in each image?") 스텝 0에서는 `'3'` -0.511 대 `"Let's"` -1.3771로 0.866 nat이다. 비전 조건부 로짓이 첫 생성 토큰에서 최대 0.9 nat 가량 이동한다는 뜻이고, 이건 반올림이 아니라 실재하는 패리티 격차다. 두 문장 모두 같은 그림을 설명하며, 이미지 1장 쪽은 오히려 mlxcel이 더 구체적이다. `tests/fixtures/test_image_shapes.png`의 빨간 사각형, 파란 원, 초록 삼각형을 색까지 짚는데 오라클 쪽은 색을 말하지 않는다.

이미지 2장 프롬프트는 아래 로더 수정 뒤로는 그 외에 정확히 일치한다. 이미지 토큰 245개(196 + 49), 프롬프트 263토큰으로 둘 다 오라클과 같고 MRoPE 위치도 비트 동일하다.

원인에서 배제한 것들: `MLX_ENABLE_TF32=0`은 출력을 바꾸지 않는다(바이트 동일). 비전 활성화 함수 두 개 모두 레퍼런스와 일치한다(블록 MLP는 tanh-GELU로 `ACT2FN["gelu_pytorch_tanh"]`와, 패치 머저는 정확한 GELU로 `nn.GELU()`와). DeepStack 분기 순서와 주입 깊이도 일치한다. 양자화도 아니다. 시험한 모든 프롬프트에서 4bit와 float32 체크포인트가 바이트 단위로 같은 출력을 낸다. 남는 것은 27블록 타워에 누적된 수치 차이이고, 이는 이 PR이 건드리지 않는 공용 Qwen3-VL 코드다. 덮지 않고 #1738로 남겼다.

**오라클이 잡아낸 실제 버그 하나.** 로더 초판은 이슈가 지정한 대로 `preprocessor_config.json`에서 리사이즈 경계를 읽었다(`size.shortest_edge` 65536). HF `AutoProcessor`는 대신 `processor_config.json`의 중첩 `image_processor` 블록을 읽는다(`min_pixels` 16384). 224x224 입력에서 낡은 경계는 16x16 병합 격자로 확대해 이미지 토큰 64개를 만드는데 레퍼런스는 49개다. 그래서 이미지 2장 프롬프트가 오라클의 245개 대신 260개를 실었고, 두 실행은 같은 프롬프트를 비교하고 있지 않았다. 아무것도 실패하지 않고 개수만 조용히 어긋났다. 수정했고 우선순위는 단위 테스트로 고정했다.

**계열 노출면.** `mlxcel arch`는 `Cohere Compass / North-Micro-Vision`을 보여주고, `mlxcel list`는 체크포인트를 보여주며, `mlxcel inspect`는 설정에서 유도한 레이어 분할을 `sliding-window: 21 layer(s) capped at 4096 tokens, 7 global`로 보고한다. 28층 3대1 패턴 그대로다.

**게이트.**

| 게이트 | 결과 |
|---|---|
| `cargo test --profile test-fast --features cuda --lib cohere_compass` | `test result: ok. 23 passed; 0 failed` |
| `cargo clippy --workspace --all-targets -- -D warnings` (CI) | 통과 |
| `cargo fmt --all -- --check` (CI) | 통과 |
| `cargo test --workspace --profile test-fast --features metal,accelerate` | **이 호스트에서 실행 불가** (Linux + CUDA, Metal도 Accelerate도 없음). 미실행 |


---

## 6. 검증하지 못한 것

- 이슈 인수 조건의 `cargo test --workspace --profile test-fast --features metal,accelerate`는 **이 호스트에서 실행할 수 없다**. Linux + CUDA 환경이고 Metal도 Accelerate도 없다. 대신 해당 범위에 대해 CUDA 등가 명령을 실행했다. metal+accelerate 조건은 미실행이다.
- 비디오 입력(`<|VIDEO_PAD|>`, `video_token_id` 255032)은 이슈 범위 밖이고 일부러 연결하지 않았다. 이 계열은 `qwen_video_runtime`과 서버의 비디오 지원 목록 양쪽에 없으므로 비디오 요청은 잘못 처리되는 대신 거부된다.
- `CohereCompassTextForSequenceClassification` 리랭커 헤드(`score_shift_a/b`, `pooling`)는 범위 밖이다. 이를 쓰는 공개 체크포인트가 없다.
- 텐서 병렬 샤딩은 범위 밖이다. `fallback_architecture`가 이 계열에 `"cohere_compass"`를 돌려주므로 범용 트랜스포머 계획이 이 모델을 가져가지 않는다.
- `norm_type: "rms_norm"`과 `transformer_block_type`은 설정 표면으로만 구현·검증했다. 상류 `CohereCompassTextConfig`에도 공개 체크포인트에도 없는 키라서, 실행되는 대신 받아서 검사만 한다. 모르는 `norm_type`과 `sequential` 블록 타입은 둘 다 다른 무언가로 돌아가는 대신 이름이 붙은 오류로 로드 단계에서 실패한다.

---

## 7. 파일

| 파일 | 역할 |
|---|---|
| `src/models/cohere_compass.rs` | 디코더: 상태, 마스크, DeepStack 주입, `logit_scale`, `LanguageModel` 구현 |
| `src/models/cohere_compass_config.rs` | `text_config` 파싱과 레이어 타입별 RoPE 해석 |
| `src/models/cohere_compass_layers.rs` | norm 분기, sliding/full 분할 어텐션, SwiGLU, 병렬 블록 |
| `src/models/cohere_compass_rope.rs` | Compass MRoPE 테이블(재배열된 `inv_freq`, `[H, W, T]` 구간) |
| `src/vision/cohere_compass.rs` | VLM 래퍼: 비전 특징, 비주얼 위치 마스크, MRoPE 상태 |
| `src/loading/vlm_cohere_compass.rs` | 로더. 체크포인트 자신의 리사이즈 경계 포함 |
| `src/multimodal/qwen_vl.rs` | `QwenVlRuntime` 구현(DeepStack 캐시 경로) |
| `docs/supported-models.md` | 계열 항목 |
