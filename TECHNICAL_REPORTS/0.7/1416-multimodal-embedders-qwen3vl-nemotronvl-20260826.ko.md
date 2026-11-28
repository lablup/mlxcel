# 기술 보고서: PR #1416 - Qwen3-VL-Embedding과 Llama-Nemotron-VL-Embed

**작성일**: 2026-08-26
**작성자**: mlxcel maintainers
**상태**: 완료. 검증 호스트에 레퍼런스 구현이 없어 수치 parity는 주장하지 않음
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1416은 이슈 #1345가 요구한 멀티모달 임베더 두 종을 구현한다. `/v1/embeddings`에서 `image_url` 항목을 받는 첫 계열이므로, 기반(#1408 / #1353)이 만들어 두었지만 소비자가 없던 이미지 경로를 처음으로 끝까지 통과시키는 작업이기도 하다.

두 계열 모두 디코더나 비전 타워를 새로 추가하지 않는다. Qwen3-VL-Embedding은 생성용 로더가 그대로 올린 생성용 Qwen3-VL 스택이고, head 자리에 last-token pooling이 들어간다. Llama-Nemotron-VL-Embed는 mlxcel이 이미 돌리는 세 조각을 조립한다. SigLIP-400M 타워, InternVL의 pixel-shuffle `mlp1` 커넥터, 그리고 양방향으로 돌린 뒤 mean pooling하는 Llama 3.2 1B 디코더다. 새로 작성한 알고리즘 코드는 체크포인트 자체의 동적 타일링뿐인데, 이것이 InternVL의 것과 다르기 때문에 기존 프로세서를 공유하는 선택은 옳지 않았다.

두 계열 바깥에서 일어난 유일한 구조 변경은 `Qwen3VLModel::forward_for_sequence`를 head 없는 `forward_hidden`과 `lm_head`로 쪼갠 것이고, 토큰 단위로 정확히 일치하는 테스트가 생성 경로의 불변을 증명한다.

---

## 1. 문제 정의

### 1.1 배경

`Qwen/Qwen3-VL-Embedding-2B`는 평범한 `Qwen3VLForConditionalGeneration` export(`model_type: qwen3_vl`)에 `pooling_mode: lasttoken`을 선언한 sentence-transformers `1_Pooling` 모듈이 붙은 형태다. 모든 가중치가 생성용 그대로다. `model.visual`의 24층 ViT, 인덱스 5, 11, 17의 DeepStack merger, 그리고 interleaved M-RoPE를 쓰는 28층 Qwen3 텍스트 디코더(`mrope_section: [24, 20, 20]`, `rope_theta: 5e6`, tied embedding)다.

`nvidia/llama-nemotron-embed-vl-1b-v2`는 `model_type: llama_nemotron_vl`에 `architectures: ["LlamaNemotronVLModel"]`을 선언하고, `llm_config`는 `LlamaBidirectionalModel`(`model_type: llama_bidirec`)을 선언한다. 업스트림은 이를 모든 어텐션 모듈에 `is_causal = False`를 설정한 `LlamaModel`로 정의한다. 비전 쪽은 `vision_model.vision_model`의 27층 SigLIP 타워(hidden 1152, `image_size` 512, `patch_size` 16)이고, 커넥터는 InternVL의 `mlp1`이다.

기반은 이미 두 변형을 탐지에 등록했고 `mlxcel arch`에 노출했으며 요청 쪽도 연결해 두었다. `EmbeddingModel::supports_images()`가 참이면 `image_url` 항목이 `EmbeddingEngine::embed_image`로 흐르고, `mlxcel embed --image`도 오프라인에서 같은 일을 한다. 디스패처의 두 arm만 "not yet supported"를 반환하고 있었다. 이 PR이 그 둘을 채운다.

### 1.2 기존 장애물

- **`Qwen3VLModel`은 항상 head를 적용한다.** `forward_for_sequence`가 `self.lm_head.forward(&h)`로 끝나고 text-only 빠른 경로도 마찬가지다. 그대로 두면 임베더가 `[B, L, 151936]` 텐서를 만들었다가 버려야 한다.
- **엔진이 계열보다 먼저 토크나이즈한다.** `EmbeddingEngine::embed_image`는 `format_text`를 먼저 부르고 `encode_row`를 거친 다음에야 배치와 이미지를 `EmbeddingModel::embed`에 넘긴다. 시각 placeholder의 개수는 이미지의 종횡비에 달려 있으므로 어느 계열도 포맷 시점에 그 수를 알 수 없다.
- **`EmbeddingModel` 트레이트에 "이 행에 이미지가 있다"는 신호가 없다.** 계열이 받는 것은 `format_text(&self, text, instruction)`뿐이다.
- **DeepStack 주입은 단일 행 전용이다.** `Qwen3VLModel::deepstack_process`는 `batch == 1`만 처리하고 그 외에는 `h`를 그대로 복사하며, M-RoPE 델타도 시퀀스마다 다르다.
- **Nemotron 체크포인트는 어떤 생성용과도 키 체계가 다르다.** 텍스트 가중치는 `language_model.` 아래, SigLIP 타워는 `vision_model.vision_model.` 아래에 있고, 타워는 어떤 임베더도 읽지 않는 attention-pooling `head.*`를 싣고 있다.
- **기존 InternVL 프로세서는 타일링 규칙이 다르다.** `InternVLProcessor::find_closest_aspect_ratio`는 종횡비 차이를 최소화하고 동률은 면적으로 가른다. 이 체크포인트의 `processing_llama_nemotron_vl.py`는 `min(area_ratio, 0.6) * min(target / actual, actual / target)`을 최대화하는 다른 목적 함수를 쓰고, 정규화도 ImageNet이 아니라 SigLIP 상수를 쓴다.
- **같은 wave에서 형제 계열 포팅 세 건이 동시에 진행 중이었다.** 그중 #1325는 이 계열이 의존할 수도 있었던 양방향 Llama 백본을 도입한다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| `forward_hidden` 분리가 생성 결과를 바꿈 | High. 모든 Qwen3-VL 생성 경로가 조용히 퇴행 | 토큰 단위 게이트가 있으면 Low |
| placeholder 수와 비전 피처 수가 어긋남 | High. `merge_llava`가 잘못된 개수를 흩뿌리고 임베딩이 조용히 틀림 | Medium |
| 타일링 규칙이 레퍼런스와 갈라짐 | Medium. 타일 수가 다르면 벡터가 달라지고, 겉보기엔 멀쩡하지만 레퍼런스 벡터와 비교할 수 없음 | Medium |
| Nemotron 스택에 causal 마스크가 들어감 | High. 검색 품질이 떨어지지만 겉으로 드러나는 증상이 없음 | Medium |
| `InternVLProcessor` 재사용이 InternVL VLM 동작을 바꿈 | High. 무관한 계열이 퇴행 | 공유했다면 Medium |
| 형제 포팅과 공유 파일에서 충돌 | Medium. 정확성이 아니라 rebase 비용 | High |

---

## 2. 기술적 선택과 그 이유

### 2.1 Qwen3-VL-Embedding을 생성용 로더로 올린다

`Qwen3VLEmbeddingModel::load`는 `crate::loading::load_qwen3_vl(model_dir)`를 호출하고 `LoadedModel::Qwen3VL`을 매칭한다. 편의가 아니라 의도적인 선택이다. 대안인 임베딩 모듈 내 스택 재구성은 `read_sanitized_vlm_config`, `parse_required_vlm_subconfig`, 양자화 상속 헬퍼 두 개, `remap_qwen3_vl_weights`(`model.language_model.`을 `model.`로, `model.visual.`을 `vision_tower.`로 다시 씀), `sanitize_tied_embeddings`, `qwen_vl_token_ids`를 모두 복제해야 했다. 전부 `loading::vlm` 안의 `pub(crate)`이고 계열 전용이 아니므로, 복제는 Qwen3-VL 레이아웃 계약의 두 번째 사본을 만들어 첫 번째와 갈라질 여지를 남긴다.

비용은 `src/loading/mod.rs`에서 `load_qwen3_vl`을 크레이트 내부로 재노출하는 한 줄이다. 이득은 Qwen3-VL 가중치 재매핑, 양자화 상속, 토큰 ID 해석에 가해지는 모든 후속 수정이 임베더에 그대로 도달한다는 것, 그리고 임베더와 생성기가 같은 텐서를 본다는 사실이 증명된다는 것이다.

### 2.2 head만 떼어내고 내부는 노출하지 않는다

`forward_hidden_for_sequence`는 최종 norm까지의 본문 전체이고, M-RoPE 위치 해석, DeepStack 주입, 상태 정리 로직은 손대지 않았다. `forward_for_sequence`는 이제 그것에 `lm_head`를 얹은 것이다. `forward_text_only`도 같은 이유로 `forward_text_only_hidden`이 되었고, 호출자가 각 분기가 아니라 최상위에서 head를 한 번 적용한다.

`src/models/qwen3_vl_tests.rs`는 결정론적 가중치로 합성 2층 Qwen3-VL을 만들고, `forward_impl`을 돌린 다음 `forward_hidden`에 `lm_head`를 손으로 얹어 `max_abs_diff == 0.0`을 단언한다. "가깝다"가 아니라 정확히 0이다. 이것이 이 분리를 동등성 주장이 아니라 증명된 리팩터링으로 만든다.

기각한 대안은 `layers`와 `norm`을 공개하는 것(두 경로가 갈라질 수 있다), 그리고 임베더에서 루프를 복제하는 것(같은 문제에 더해 DeepStack과 M-RoPE 부기까지 복제해야 한다)이었다. doc 주석은 세 번째 소비자인 #1356의 Qwen3-VL 리랭커를 명시해 두었으므로 다음 포팅이 같은 판단을 다시 하지 않아도 된다.

### 2.3 시각 placeholder는 `EmbeddingModel::expand_image_tokens`로 확장한다

두 계열 모두 `format_text`에서는 placeholder를 정확히 하나만 내보내고, 엔진이 패딩 전에 인코딩된 행에 대해 호출하는 `expand_image_tokens`에서 실제 개수로 키운다. `expand_image_placeholders`는 id 행을 훑어 각 placeholder를 `counts[i]`개로 치환하고, 패딩 행이 플래그를 유지하도록 어텐션 마스크를 같은 보폭으로 확장한다.

이 훅은 이 작업이 진행되는 동안 late-interaction 포팅(#1414)과 함께 기반에 착지했다. 이 포팅의 이전 리비전은 `embed` 안에서 확장했는데, 동작은 했지만 `usage.prompt_tokens`가 시각 블록이 아니라 래퍼 토큰을 세게 두었다. 훅을 도입해 그 문제가 사라졌다. 사진 두 장을 쓴 CLI 실행에서 이미지 행이 Qwen3-VL은 1332, Nemotron은 3624 토큰을 보고한다. 이전에는 각각 134와 55였다. `embed`도 id를 다시 쓰지 않고 엔진이 만든 배치를 그대로 읽는다.

개수의 출처는 계열마다 다르다. Qwen3-VL은 Qwen2-VL 프로세서 자신의 그리드에서 이미지당 `t * (h / merge) * (w / merge)`를 얻는데, 생성 경로가 `<|image_pad|>` 구간의 길이를 정할 때 쓰는 것과 같은 산술이다. Nemotron은 `num_image_token * tiles`, 즉 `512x512` 타일당 256이다.

id 수준에서 확장하는 것은 토크나이즈 전에 문자열을 확장하는 것과 정확히 같다. `<|image_pad|>`와 `<IMG_CONTEXT>`는 둘 다 추가된 특수 토큰이라 항상 id 하나로 토크나이즈되기 때문이다. 함수는 두 계열이 하나의 구현을 공유하고, 공유 오류 경로(placeholder 수와 이미지 수의 불일치, 시각 토큰이 0개인 이미지)에는 단위 테스트가 붙어 있다.

개수는 forward pass가 쓰는 것과 같은 프로세서에서 호출당 한 번 유도하므로, 엔진이 확장에 쓴 수와 `embed`가 병합하는 수가 어긋날 수 없다.

### 2.4 빈 문자열을 "이 행에 이미지가 있다"는 신호로 쓴다

`format_text`는 이미지 블록을 내보낼지 알아야 하는데 트레이트는 텍스트만 준다. 엔진의 이미지 경로는 `format_text("", instruction)`을 호출하고, 모든 텍스트 경로는 빈 문자열이 계열에 닿기 전에 두 곳에서 거절한다. `EmbeddingEngine::embed_texts`는 `input[i] is an empty string`을 반환하고, 라우트의 `validate_items`는 같은 내용을 `400`으로 반환한다. 따라서 이 코드베이스 안에서 빈 문자열은 모호하지 않다.

실재하는 결합이고 두 계열의 doc 주석이 그렇게 적으면서 `embed_image`와 두 거절 지점을 지목한다. wave 병렬 포팅이 쓸 수 있는 가장 덜 침습적인 선택지다. 트레이트에 명시적 플래그를 더하는 후속 작업이 두 주석과 분기를 함께 지울 수 있다.

### 2.5 Nemotron 타일링에 `InternVLProcessor`를 재사용하지 않는다

`src/models/llama_nemotron_vl_tiling.rs`는 공유 InternVL 프로세서의 생성자 변형이 아니라 별도의 187줄 모듈이다. 두 가지 차이가 이를 강제했다.

1. `find_closest_aspect_ratio`가 다른 함수다. InternVL 포팅은 `|aspect - target|`을 최소화하고 동률을 면적 근접도로 가른다. 이 체크포인트는 `min(area_ratio, 0.6) * min(target / actual, actual / target)`을 최대화한다. 두 규칙은 대부분의 이미지에서 일치하고 일부에서 갈라지므로, 차이가 조용하다는 점이 무시할 이유가 아니라 제대로 맞출 이유가 된다.
2. 정규화가 SigLIP 방식이고(`processor_config.json`의 `norm_type: "siglip"`에 따라 `mean = std = 0.5`), 출력이 channels-last여야 한다. `SigLipVisionModel`은 `[B, H, W, C]`를 합성곱하는데 `InternVLProcessor`는 `[tiles, 3, H, W]`를 내놓는다.

공유 프로세서를 양쪽을 다 덮을 만큼 설정 가능하게 만드는 선택은 호출자 하나를 위해 InternVL VLM의 타일링 동작을 플래그 뒤에서 바꾸는 일이었다. 새 모듈은 공유 쪽을 손대지 않는다. 대신 재사용하는 것은 실제 모델 코드인 `InternVLConnector`다. `ps_version: v2`의 permute 두 번을 모두 포함한 `pixel_shuffle(0.5)` 다음에 `LayerNorm(4608) -> Linear(4608 -> 2048) -> GELU -> Linear(2048 -> 2048)`가 온다.

후보 그리드 열거는 타일 수만이 아니라 `(cols * rows, cols, rows)`로 정렬한다. 레퍼런스는 파이썬 set을 `cols * rows`로 정렬하므로 그룹 내부 순서가 set 순회에 맡겨진다. 이를 고정하면 strict greater-than 동률 처리가 실행과 플랫폼에 걸쳐 결정론적이 된다.

### 2.6 타일 예산은 `config.json`이 아니라 `processor_config.json`에서 읽는다

체크포인트는 `max_input_tiles`를 서로 다른 값으로 두 번 선언한다. `config.json` 최상위에서는 `2`, `processor_config.json`에서는 `6`이다. 레퍼런스는 `AutoProcessor`를 통해 `processor_config.json`으로 프로세서를 만들기 때문에 실제로 도는 값은 `6`이고, mlxcel도 그것을 읽는다. `image_size`, `use_thumbnail`, `num_image_token`, `passage_prefix`도 같은 파일에서 오며, 키나 파일 자체가 없으면 공개된 값이 기본값으로 쓰인다.

### 2.7 접두사를 먼저 벗겨 `embedding_sanitize`를 재사용한다

Nemotron의 텍스트 가중치는 `language_model.embed_tokens.weight`, `language_model.layers.{i}....`, `language_model.norm.weight`로 도착하는데 `Llama3Model::from_weights`는 `model.` 접두사 키를 읽는다. 곧바로 `model.`로 바꾸는 대신 `sanitize_nemotron_vl_weights`는 `language_model.`을 벗기고 공유 `sanitize_decoder_embedding_weights`를 부른다. 이 함수가 백본 루트 세 개에만 `model.`을 다시 붙이고, 생성 head를 버리고, `{N}_Dense` 모듈 폴더를 접는다.

공유 헬퍼를 경유하는 것이 요점이다. 이 체크포인트에는 오늘 `Dense` 모듈도 `lm_head`도 없지만 재배포본이나 향후 리비전에는 있을 수 있고, 공유 경로를 지나면 두 번째 구현 없이 처리된다. Nemotron 전용 제거 두 가지, 즉 SigLIP attention-pooling `head.*`와 파라미터가 아닌 버퍼(`rotary_emb.inv_freq`, `*position_ids`)는 공유 패스보다 먼저 일어나므로 백본 루트로 오인될 수 없다.

### 2.8 형제 포팅에 의존하지 않고 Llama 레이어를 직접 돌린다

이슈 #1345는 #1325의 `Llama3Backbone::from_weights_without_head` "또는 로컬 동등물"을 제안했다. #1325는 같은 wave에서 같은 공유 파일을 대상으로 돌고 있었으므로, `LlamaNemotronVLEmbeddingModel`은 `tie_word_embeddings`를 참으로 강제한 평범한 `Llama3Model`을 만들고(이러면 생성자가 결코 적용하지 않을 head 자리에 `embed_tokens`를 재사용하므로, 이 체크포인트에 없는 `lm_head`를 찾다 실패하지 않는다) `embed_tokens`, `layers`, `norm`을 직접 돌린다. 셋 다 이미 `pub` 필드다.

양방향 동작은 전적으로 마스크에서 나온다. `create_bidirectional_padding_mask`가 패딩 key만 차단하는 `[B, 1, 1, L]`을 만들고, `Attention::forward`가 causal 빠른 경로 대신 주어진 마스크를 쓴다. offset 0의 새 `KVCache`들이 RoPE 위치를 입력 길이에 맞춘다.

wave 1의 머지 충돌과 맞바꾼 열다섯 줄 남짓의 의도적 중복이다. 둘 다 머지된 뒤 #1325의 백본 타입이 더 나은 자리라면 루프를 그쪽으로 접는 것은 기계적인 후속 작업이다.

### 2.9 픽셀 범위를 체크포인트에서 가져온다

`load_qwen3_vl`은 Qwen2-VL 기본값으로 프로세서를 만드는데 그 상한은 `16384 * 28 * 28` 픽셀이다. `patch_size` 16, `spatial_merge_size` 2에서 이미지 하나가 최대 12544 시각 토큰이 되고, 이는 임베더의 8192 토큰 예산을 한참 넘는다. `Qwen/Qwen3-VL-Embedding-2B`는 `preprocessor_config.json`에 `max_pixels: 1310720`을 선언하며, 이 값이 이미지 하나를 1280 토큰으로 묶는다. `apply_pixel_bounds`가 로드 후 그 파일에서 `min_pixels`와 `max_pixels`를 덮어쓰므로, 범위가 다른 재배포본도 하드코딩된 숫자를 물려받지 않고 올바르게 동작한다.

### 2.10 오른쪽 패딩, 그리고 그것이 업스트림보다 여기서 더 중요한 이유

레퍼런스 Nemotron 프로세서는 `tokenizer.padding_side = "left"`로 두면서도 `position_ids = arange(0, L)`을 그대로 계산한다. 그래서 패딩된 행의 실제 토큰은 `pad_count..L-1`에, 단독 행의 실제 토큰은 `0..n-1`에 놓이고 둘은 수치적으로 다르다. mlxcel 엔진은 오른쪽으로 패딩하므로 두 경우 모두 실제 토큰이 `0..n-1`에 놓이고, 패딩된 배치가 허용 오차가 아니라 구성상 단독 결과를 재현한다. Qwen3-VL도 마찬가지다. causal + padding 마스크 아래의 last-token pooling에서는 패딩이 pooling 위치 뒤에 놓이고 모든 실제 쿼리에 대해 key로서 차단된다.

### 2.11 교차 모달 테스트 이미지는 픽스처 대신 그린다

저장소의 유일한 이미지 픽스처는 단색 주황 정사각형이라 두 캡션을 가를 수 없다. `src/models/vl_embedding_test_images.rs`는 요청한 어떤 종횡비로든 두 장면을 결정론적으로 그린다. 흰 바탕의 막대 그래프, 그리고 하늘과 해와 바다와 모래가 있는 해변이다. 덕분에 게이트가 바이트 단위로 재현 가능하고, 바이너리 픽스처가 늘지 않으며, 타일링 게이트가 6:1 띠 이미지를 요구할 수 있다. 실제 체크포인트 수동 검증에는 내려받은 사진 세 장을 썼고, 그쪽이 더 강한 증거라 5절에 보고한다.

---

## 3. 구현 상세

### 3.1 Qwen3-VL-Embedding forward

텍스트 행:

```
mask   = create_causal_padding_mask(attention_mask, 0)          # [B, 1, L, L]
hidden = text_model.forward_hidden(input_ids, None, fresh caches, Some(mask))
pooled = pool(hidden, attention_mask, LastToken)
```

M-RoPE와 DeepStack 슬롯을 먼저 비우므로 `forward_hidden`은 text-only 빠른 경로를 탄다. 그 경로는 `fast_rope`를 쓰는데, 비전 토큰이 없는 시퀀스에서는 이것이 멀티모달 경로와 수치적으로 같다. 모든 위치에서 `T == H == W`이면 M-RoPE 세 구간이 같은 주파수를 나르고 interleaving이 항등이 되기 때문이다.

이미지 행(항상 `B == 1`. 엔진이 placeholder를 이미 확장해 두었다):

```
pixels, grid = processor.preprocess_with_grid([image])
merged       = vlm.get_input_embeddings(input_ids, pixels, grid)  # M-RoPE + DeepStack 상태 설정
hidden       = text_model.forward_hidden(input_ids, Some(merged), fresh caches,
                                         Some(create_causal_padding_mask(attention_mask, 0)))
pooled       = pool(hidden, attention_mask, LastToken)
```

나가는 길에 상태를 다시 비우므로 뒤따르는 텍스트 호출이 그것을 물려받지 못한다. `input_embeddings`가 `Some`이므로 `forward_hidden`은 상태 정리 분기와 빠른 경로를 모두 건너뛰고 일반 경로를 타는데, DeepStack 주입이 있는 곳이 바로 거기다.

### 3.2 렌더링되는 프롬프트

`format_text`는 체크포인트 자신의 `chat_template.jinja`를 `ChatTemplateProcessor::apply_raw`로 두 메시지 목록에 적용하며, `add_generation_prompt`는 기본값 `true`다. 텍스트 행은 이렇게 된다.

```
<|im_start|>system
Represent the user's input.<|im_end|>
<|im_start|>user
a photo of a dog<|im_end|>
<|im_start|>assistant
```

이미지 행은 user 턴이 `<|vision_start|><|image_pad|><|vision_end|>`로 바뀐다. 두 문자열 모두 실제 템플릿 파일을 상대로 테스트에서 정확히 단언되며, 이 테스트는 체크포인트 디렉터리를 필요로 하되 가중치는 필요로 하지 않는다.

instruction 기본값은 `config_sentence_transformers.json`의 `prompts[default_prompt_name]`이고, 공개된 2B에서는 `Represent the user's input.`이다. 호출자가 준 instruction은 trim된 뒤 마지막 문자가 영숫자이면 `.`을 붙인다. 그래서 `Represent the user's input`은 문장이 되고 `画像を表す。`나 `Find the matching image?`는 그대로 남는다.

pooling 위치는 assistant 헤더의 마지막 개행이며, 이 프롬프트에서 `pooling_mode: lasttoken`이 고르는 자리가 그곳이다.

### 3.3 Llama-Nemotron-VL-Embed forward

```
h = embed_tokens[input_ids]                                       # [B, L, 2048]
if image:
    v = SigLipVisionModel.forward(pixels).hidden_states            # post_layernorm, [tiles, 1024, 1152]
    v = mlp1(pixel_shuffle(v, 0.5))                                # [tiles, 256, 2048]
    h = merge_llava(img_context_token_id, v, h, input_ids)
mask = create_bidirectional_padding_mask(attention_mask)           # [B, 1, 1, L]
for i, layer in layers: h = layer.forward(h, fresh KVCache, Some(mask))
h = norm(h); pooled = pool(h, attention_mask, Mean)
```

`select_layer: -1`은 레퍼런스가 타워의 `last_hidden_state`를 읽는다는 뜻이고, `SiglipVisionModel`에서 그것은 `post_layernorm` 출력이며, 이는 feature-layer 선택이 설정되지 않았을 때 mlxcel의 `SigLipVisionModel::forward`가 반환하는 값과 같다. attention-pooling head를 로드 시에 버리는 이유가 바로 이 경로가 그것에 닿지 않기 때문이다. 레퍼런스도 SigLIP 타워에서는 `vit_embeds[:, 1:, :]` CLS 제거를 건너뛰고, mlxcel의 SigLIP 임베딩에는 CLS 토큰이 없으므로 별도 분기 없이 둘이 일치한다.

`mlp1`의 LayerNorm eps는 PyTorch `nn.LayerNorm` 기본값인 `1e-5`를 쓴다. 체크포인트가 이 모듈에 대한 eps를 선언하지 않기 때문이다.

### 3.4 문서 프롬프트

`format_text`는 텍스트에 대해 항등이다. `query: `와 `passage: ` 접두사는 텍스트 전용 형제인 `nvidia/llama-nemotron-embed-1b-v2`와 똑같이 호출자 쪽 책임이다. 이미지 행에는 호출자 텍스트가 없으므로 계열이 레퍼런스의 문서 형식을 직접 내보낸다.

```
passage: <img><IMG_CONTEXT></img><공백>
```

끝의 공백은 우연이 아니다. 레퍼런스는 `content = "<image>" + " " + text`를 만들고 다시 `content = passage_prefix + " " + content`를 만들므로 텍스트가 비면 문자열이 공백으로 끝난다. 이를 재현해야 토큰 열이 레퍼런스와 같아진다. `<IMG_CONTEXT>` 하나는 `embed`에서 `256 * tiles`로 확장되고, 실제 체크포인트 테스트가 내보낸 프롬프트를 체크포인트 자신의 토크나이저로 토크나이즈해 placeholder가 정확히 하나이고 앞에 `<|begin_of_text|>`가 붙는지 단언한 뒤 1타일과 7타일 이미지 각각의 확장 길이를 확인한다.

### 3.5 가중치 키 매핑

| 공개된 형태 | 정규화 후 |
|-------------|-----------|
| `language_model.embed_tokens.weight` | `model.embed_tokens.weight` |
| `language_model.layers.{i}.*` | `model.layers.{i}.*` |
| `language_model.norm.weight` | `model.norm.weight` |
| `vision_model.vision_model.*` | 그대로 |
| `mlp1.0.*`, `mlp1.1.*`, `mlp1.3.*` | 그대로 |
| `vision_model.vision_model.head.*` | 제거 |
| `lm_head.*`, `*rotary_emb.inv_freq`, `*position_ids` | 제거 |

`patch_embedding.weight`는 여기서 다룰 필요가 없다. `VisionEmbeddings::from_weights`가 이미 PyTorch의 `[1152, 3, 16, 16]` 레이아웃을 감지해 channels-last로 전치한다.

### 3.6 등록과 요청 쪽

`build_family_model`의 두 arm만 바뀌었고 다른 계열의 "not yet supported" arm은 손대지 않았다. 형제 포팅 세 건이 병렬로 착지 중이었고 그중 #1411(BERT와 XLM-RoBERTa)이 작업 도중 이 파일에 머지되었기 때문에 이것이 중요했다.

탐지, `src/model_metadata.rs`, `mlxcel arch` 항목은 이미 있었으므로 `mlxcel arch`는 둘을 `Embedding` 아래에 나열하고 `mlxcel list`는 두 체크포인트를 수정 없이 보여 준다. 라우트도 손댈 것이 없었다. `validate_items`가 이미 `image_url`을 `supports_images()`로 막고, `fetch_images`가 공유 이미지 제한 아래에서 디코딩하며, `embed_items`가 이미지를 한 번에 하나씩 돌리고 결과를 요청 순서로 되돌려 쓰고, `instruction`이 이미 `format_text`에 닿는다. `mlxcel embed --image`도 오프라인에서 같은 일을 이미 하고 있었다.

---

## 4. 테스트 전략

### 4.1 체크포인트 없이

- `forward_hidden_then_head_matches_forward_impl`: 합성 2층 Qwen3-VL에서 토큰 단위로 정확한 리팩터링 가드
- `expand_image_placeholders_*`: placeholder 여러 개에 걸친 순서, 패딩 플래그 보존, 양쪽 오류 경로
- `instruction_gets_trailing_period_only_when_it_ends_mid_sentence`: 비 ASCII 종결 문자 포함
- `sanitize_drops_the_vision_head_and_maps_the_language_model_prefix`: 살아남아야 할 모든 키와 사라져야 할 모든 부류
- `a_square_image_uses_one_tile_and_a_wide_image_uses_the_budget`, `image_block_expands_to_num_image_token_per_tile`: 정사각형, 6:1 띠, 2:3 페이지의 타일 예산
- `preprocess_emits_channels_last_tiles_in_the_siglip_range`: `[tiles, H, W, 3]` shape와 `[-1, 1]` 범위
- `pixel_shuffle_then_mlp1_maps_one_tile_to_256_language_tokens`: 무작위 가중치에서 `[1, 1024, 1152]`가 `[1, 256, 2048]`로

### 4.2 체크포인트가 있을 때만, 없으면 soft-skip

- `format_text` 게이트 둘은 실제 `chat_template.jinja`로 렌더링해 프롬프트 문자열을 정확히 단언한다.
- `bidirectional_prefill_lets_an_early_token_see_a_later_one`: 96 토큰에서 마지막 토큰만 바꾸면 pooling 벡터가 움직여야 한다. causal 스택은 그럴 수 없다.
- `the_document_prompt_expands_to_256_tokens_per_tile`: 내보낸 프롬프트를 체크포인트 토크나이저로 토크나이즈한 뒤 1타일과 7타일에 대해 확장
- 두 계열의 텍스트 게이트: 동일 행, 패러프레이즈 대 무관 문장, 무관 상한, 그리고 패딩 배치 대 단독 입력 벡터
- 두 계열의 이미지 게이트: 단위 벡터, 유한한 성분, 양방향 교차 모달 마진, 같은 이미지 재임베딩

모델을 만들거나 MLX 연산을 평가하는 모든 테스트가 프로세스 전역 `mlx_test_guard()`를 잡고, 모든 게이트 수치는 `--test-threads=1`에서 기록했다.

### 4.3 생성 경로 회귀 확인

`forward_hidden` 분리는 토큰 단위 단위 테스트가 지키고, 별도로 `Qwen/Qwen3-VL-Reranker-2B`(여기에 내려받혀 있는 생성용 Qwen3-VL 체크포인트)로 `mlxcel generate`를 텍스트 한 번, 이미지 한 번 돌려 확인했다. 텍스트 실행은 질문에 조리 있게 답했고, 이미지 실행은 줄무늬 고양이를 호랑이로 묘사했다. 이는 이 체크포인트가 캡셔너가 아니라 리랭커라는 사실 때문이며, DeepStack이나 M-RoPE 경로가 깨졌을 때 나오는 쓰레기 출력과는 거리가 멀다.

---

## 5. 실제 체크포인트 결과

Linux, GB10, CUDA, bf16(Apple Silicon의 bf16 to f16 규칙은 여기 적용되지 않는다). 모든 명령을 연속 세 번 실행해 CLI와 서버 양쪽에서 유사도 행렬이 비트 단위로 같았고, 여덟 건의 측정 모두 실행 간 최대 절대 편차가 `0.000e+00`이었다.

### `Qwen/Qwen3-VL-Embedding-2B` (2048 차원, `max_length` 8192)

| 게이트 | 관측값 | 요구 조건 |
|--------|--------|-----------|
| 벡터 폭과 norm | 2048, norm 1.000000 | 2048차원 단위 벡터 |
| 배치 내 동일 입력 | cosine 1.000000000 | 1e-6 이내로 1.0 |
| 패러프레이즈 대 무관 문장 | 0.7382 대 0.1245 | 마진 0.15 이상 |
| 무관 상한 | 0.1245 | 0.5 미만 |
| 패딩 배치 대 단독 입력 | 형태별 최대 편차 9e-4 | 1e-3 이내 |
| 자유의 여신상 캡션, 일치 사진 대 무관 사진 | 0.5575 대 0.1352 | 마진 0.1 이상 |
| 고양이 캡션, 일치 사진 대 무관 사진 | 0.5342 대 0.1676 | 마진 0.1 이상 |
| 비유한 성분 | 없음 | 없음 |

`mlxcel embed`와, 텍스트와 `image_url` 항목을 섞은 `POST /v1/embeddings` 요청 하나를 받은 `mlxcel-server` 양쪽에서 측정했고 두 표면이 같은 값을 돌려주었다.

세 번째 쌍은 마진이 약했고, 실제로 관측된 값이므로 그대로 보고한다. 슈거파우더를 뿌린 페이스트리가 놓인 어수선한 실내 사진에 대해 그 캡션은 자기 이미지를 0.016 차이로만 선호했다(0.1471 대 무관 사진 두 장의 0.1291, 0.1312). 방향은 맞으므로 에픽의 순서 게이트는 성립하지만, 이슈가 명시한 0.1 마진이 이 모델의 모든 쌍에서 성립하지는 않는다. 그 세트의 두 이미지는 서로 0.6187을 기록하기도 하는데, 이 계열의 이미지 대 이미지 유사도 바닥이 높다는 뜻이고 SigLIP 텍스트 타워가 보이는 것과 같은 이방성이다.

### `nvidia/llama-nemotron-embed-vl-1b-v2` (2048 차원, `max_length` 8192)

| 게이트 | 관측값 | 요구 조건 |
|--------|--------|-----------|
| 벡터 폭과 norm | 2048, norm 1.000000 | 2048차원 단위 벡터 |
| 배치 내 동일 입력 | CLI 배치에서 cosine 1.000000000 | 1e-6 이내로 1.0 |
| 관련 passage 대 무관 passage | 0.4284 대 -0.0219 | 마진 0.15 이상 |
| 무관 상한 | -0.0219 | 0.5 미만 |
| 패딩 배치 대 단독 입력 | 1e-3 이내 | 1e-3 이내 |
| 자유의 여신상 질의, 일치 사진 대 무관 사진 | 0.4041 대 0.0963 | 마진 0.1 이상 |
| 고양이 질의, 일치 사진 대 무관 사진 | 0.4836 대 0.0440 | 마진 0.1 이상 |
| 페이스트리 질의, 일치 사진 대 무관 사진 둘 | 0.4836 대 0.0792, 0.0896 | 마진 0.1 이상 |
| 6타일 가로 이미지 + 썸네일 | `7 * 256` 시각 토큰, 오류 없이 임베딩 | 잘림 없음 |

Qwen3-VL이 순서만 맞혔던 페이스트리 사진을 포함해, 이 계열에서는 교차 모달 세 쌍이 모두 마진을 넘겼다.

### 동일 행 관측

서버 측정 하나가 cosine 1.0에 도달하지 않았다. 텍스트 항목이 동일한 자유의 여신상 캡션 둘과 다른 캡션 하나였던 요청에서, Nemotron의 동일 쌍이 0.99995714를 반환했다. 레이어별 프로브가 갈라지는 지점을 정확히 찾아냈다. 동일한 세 행 중 0번과 1번은 레이어 4까지 비트 단위로 일치하고, 레이어 5에서 2번 행이 `9.766e-4`만큼 벌어진다. 그 크기에서 정확히 bf16 1 ulp다. 차이는 남은 열한 레이어를 지나며 pooling 직전 hidden state에서 `5.0e-1`까지 누적되고, 정규화된 벡터에서는 `4.3e-5`가 된다.

세 가지 대조가 이 현상을 이 포팅 바깥에 둔다.

- 배치 크기 2에서 8, 토큰 길이 아홉 가지에 걸친 스윕이 이미 머지된 `Qwen3Embedding` 계열에서 같은 부류의 편차를 재현한다(한 길이에서 6.0e-5). 그 계열은 이 포팅의 코드를 하나도 공유하지 않는다.
- 관련된 정확한 shape에서, 동일 행을 타일링해 마스크가 있는 경우와 없는 경우 모두로 `matmul`과 `scaled_dot_product_attention`을 따로 재현하면 행 위치에 대해 정확히 불변이다.
- 편차는 완전히 결정론적이다. 같은 입력이 매 실행 같은 값을 낸다. 경쟁 상태도 아니고 MLX 스레딩 위험도 아니다.

`MLXCEL_FUSED_ADD_RMSNORM`과 `MLXCEL_FUSED_ROPE_APPEND`를 꺼도 값이 달라지지 않는다. 테스트 상수에 기록한 결론은, 이것이 어느 계열의 forward pass가 아니라 일부 shape에서 공유 bf16 배치 디코드 경로가 갖는 성질이라는 것, 그리고 1e-6 단언이 앞으로 실패하면 forward pass를 의심하기 전에 스윕을 먼저 돌려야 한다는 것이다.

---

## 6. 검증 요약

| 명령 | 결과 |
|------|------|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo check --profile test-fast --features cuda --all-targets` | exit 0 |
| `cargo test ... --lib -- --test-threads=1 models::qwen3_vl models::llama_nemotron_vl` | 23 passed, 0 failed |
| `cargo test ... --lib -- --test-threads=1 embeddings::` | 69 passed, 0 failed |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |
| 두 체크포인트에서 `mlxcel embed`, 텍스트와 이미지, 각 3회 | exit 0, 비트 단위 동일 |
| 두 체크포인트에서 `mlxcel-server` + `POST /v1/embeddings`, 텍스트와 이미지 혼합, 각 3회 | HTTP 200, 비트 단위 동일 |
| `Qwen/Qwen3-VL-Reranker-2B`에서 `mlxcel generate`, 텍스트와 이미지 | exit 0, 조리 있는 출력 |

이슈의 수용 기준은 macOS `metal,accelerate` 피처 세트를 명시한다. 이 호스트에서 그에 대응하는 것은 CUDA 게이트이고 실제로 돈 것도 그쪽이다. macOS 게이트는 CI의 몫이다.

임베딩 전용 서버는 채팅 모델이 로드되지 않으므로 `/health`에서 `503`을 반환한다. 이는 기반의 기존 동작이고 이번 변경이 아니다. 준비 상태는 `/v1/embeddings`로 확인했다.

성능 수치는 보고하지 않는다. 에픽이 성능 측정을 별도로 돌린다.

---

## 7. 변경 요약

| 파일 | 변경 |
|------|------|
| `src/models/qwen3_vl_embedding.rs` (신규, 402) | `Qwen3VLEmbeddingModel`: 로더 재사용, chat template 포맷, placeholder 확장, last-token pooling |
| `src/models/qwen3_vl_embedding_tests.rs` (신규, 300) | instruction 규칙, 확장, 정확한 프롬프트 렌더링, 실제 체크포인트 텍스트와 이미지 게이트 |
| `src/models/llama_nemotron_vl_embedding.rs` (신규, 387) | `LlamaNemotronVLEmbeddingModel`: SigLIP + `mlp1` + 양방향 Llama, mean pooling, 가중치 정규화 |
| `src/models/llama_nemotron_vl_embedding_tests.rs` (신규, 495) | 정규화기, 타일링, 커넥터 shape, 양방향성, 실제 체크포인트 게이트 |
| `src/models/llama_nemotron_vl_tiling.rs` (신규, 187) | 체크포인트의 면적 인식 타일링과 SigLIP 정규화, channels-last |
| `src/models/qwen3_vl_tests.rs` (신규, 173) | 토큰 단위 `forward_hidden` 리팩터링 가드 |
| `src/models/vl_embedding_test_images.rs` (신규, 100, 테스트 전용) | 교차 모달 게이트용 결정론적 합성 장면 둘 |
| `src/models/qwen3_vl.rs` (+40/-6) | `forward_for_sequence`를 `forward_hidden_for_sequence` + head로 분리, text-only 경로도 동일 |
| `src/embeddings/loader.rs` (+15/-7) | 계열 arm 두 개 구성. 에픽이 열거한 모든 변형에 forward pass가 생겼으므로 `not yet supported` arm 제거 |
| `src/embeddings/real_checkpoint_tests.rs` (+27/-14) | 미포팅 계열 게이트를 뒤집음. 어떤 임베딩 변형도 `not yet supported`를 보고해서는 안 된다 |
| `src/loading/mod.rs` (+5) | `load_qwen3_vl` 크레이트 내부 재노출 |
| `src/models/mod.rs` (+5) | 모듈 선언 네 개 |
| `docs/embeddings.md` (+47) | 계열 노트 두 절 |
| `docs/supported-models.md` (+4) | Embedding 행 두 개와, 이미지를 받는 네 계열을 모두 명시한 문단 |

14개 파일, 2202 추가, 30 삭제. wave 동안 세 번 rebase했다. 작업 도중 머지된 BERT와 XLM-RoBERTa 포팅(#1411), late-interaction 포팅(#1414), 양방향 디코더 포팅(#1415) 위로 각각 올렸다. 에픽 #1348의 마지막 계열이므로 `build_family_model`에는 더 이상 미포팅 arm이 없다.

---

## 8. 검증되지 않은 부분

- **레퍼런스 구현 대비 수치 parity.** 검증 호스트에 PyTorch나 `transformers`가 없으므로, 업스트림에서 도는 `LlamaNemotronVLModel`이나 Qwen3-VL-Embedding 래퍼와 비교한 것은 아무것도 없다. 두 모델 카드 어느 쪽도 기준 유사도 행렬을 공개하지 않으므로, Qwen3-Embedding 포팅과 달리 대조할 공개 수치도 없었다. 5절의 모든 값은 이슈가 정한 임계치를 끝까지 측정한 결과이지 parity 주장이 아니다.
- **macOS와 Metal.** 전부 Linux와 CUDA에서 돌았다. Apple Silicon에서만 작동하는 bf16 to f16 변환 규칙은 두 계열 모두 미검증이다.
- **한 요청에 이미지 여러 장.** `expand_image_placeholders`는 개수 목록을 받고 placeholder 두 개로 단위 테스트되지만, 엔진은 `embed` 호출당 이미지 하나를 보내므로 다중 placeholder 경로에는 종단 간 커버리지가 없다.
- **비디오 입력.** 이슈상 범위 밖이다. `<|video_pad|>`는 확장되지 않으며, 비디오 항목은 조용히 틀린 벡터를 내는 대신 placeholder 개수 검사에서 실패한다.
- **`Qwen/Qwen3-VL-Embedding-8B`.** 크기만 다른 같은 코드이고 범위 밖이며 로드하지 않았다.
- **양자화 export.** 두 계열 모두 아직 공개된 양자화 체크포인트가 없다. 코드는 `quantization_params`를 SigLIP 타워와 커넥터로 흘려보내고 `UnifiedLinear`는 `.scales`가 없으면 dense linear로 물러서지만, 양자화 산출물을 로드해 보지는 않았다.
- **Nemotron의 레퍼런스 절단 길이.** `max_length`가 레퍼런스의 `p_max_length` 4096, `q_max_length` 512가 아니라 공유 상한 8192로 결정된다. 해당 키가 `processor_config.json`에 있고 공유 길이 유도 로직이 그 파일을 읽지 않기 때문이다. `--embedding-max-length`로 재현할 수 있으나 기본값으로는 되지 않는다.

---

## 9. 후속 작업

- `EmbeddingModel::format_text`에 "이 행에 이미지가 있다"는 입력을 명시적으로 주거나 포맷을 전처리 뒤로 옮겨서, 2.4의 빈 문자열 관행을 지울 것. 기반 표면이므로 에픽의 병렬 wave 이후에 착지해야 한다.
- #1325가 머지된 뒤, Nemotron 레이어 루프를 그쪽 양방향 Llama 백본으로 접는 것을 검토할 것.
- `transformers`가 있는 호스트가 확보되면 `LlamaNemotronVLModel.encode_queries`와 `encode_documents` 대비 parity 검사를 추가할 것. 이 보고서가 제시할 수 없는 유일한 부류의 증거다.
- 에픽의 성능 측정에 두 계열의 이미지 행을 포함할 것. 그 경로는 비전 타워와 타일 예산이 지배하는데 여기서 측정한 것이 없다.

---

## 참고

- 이슈 #1345, 에픽 #1348
- PR #1408(임베딩 기반), 이슈 #1353
- PR #1411(BERT와 XLM-RoBERTa), PR #1413(EmbeddingGemma와 Qwen3-Embedding)
- `docs/embeddings.md`, `docs/supported-models.md`
- `Qwen/Qwen3-VL-Embedding-2B`, `nvidia/llama-nemotron-embed-vl-1b-v2`
