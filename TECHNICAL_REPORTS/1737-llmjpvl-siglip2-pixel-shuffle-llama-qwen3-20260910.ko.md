# 기술 보고서: PR #1737 - feat(llmjpvl): port LLM-jp-4 VL and Jagle-VL

**날짜**: 2026-09-10
**작성**: mlxcel 메인테이너
**검토**: 구현 및 검증 사이클
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: 중간 (기존 런타임을 재사용한 새 모델 계열, 그 런타임과 조용히 갈리는 지점 4곳, 배포된 두 체크포인트와 그 레퍼런스 구현으로 검증)

---

## 요약

PR #1737은 LLM-jp 연구소의 일본어 VLM 계열인 `model_type: "llmjpvl"`을 추가한다. 아키텍처는 하나이고, 배포된 두 체크포인트의 실질적 차이는 `llm_config.model_type`이 지목하는 디코더뿐이다(`llm-jp-4-vl-9B-beta`는 Llama, `Jagle-VL-2.2B-Jagle-FineVision`은 Qwen3). SigLIP2-so400m 타워가 InternVL식 동적 타일링 뒤에서 돌고, `pixel_shuffle(0.5)`가 512px 타일마다 폭 4608의 벡터 256개를 만들며, `mlp1` 커넥터가 그것을 디코더 폭으로 투영해 `<|image_pad|>` 자리를 대체한다. 이슈가 요청한 대로 InternVL 런타임을 재사용했고, 그 주변에서 조용히 실패하는 방식으로 갈리는 지점이 넷이다. 디코더 설정이 `text_config`가 아니라 `llm_config`에 있고, SigLIP 타워에는 CLS 토큰이 없어 InternViT의 `[:, 1:, :]` 슬라이스가 진짜 패치 한 줄을 지우며, `mlp1`의 LayerNorm은 비전 설정의 1e-6이 아니라 torch 기본값 1e-5를 쓰고, 런타임이 두 디코더 중 하나를 실어야 한다. 합성 테스트로는 잡히지 않았을 프롬프트 결함 둘도 찾아 고쳤다. InternVL의 "첫 토큰 뒤에 끼우기" 대체 경로는 이 템플릿에서 이미지 블록을 **시스템** 턴 안에 넣고, 서버는 타입 있는 미디어를 보지 않는 템플릿에 `{{ message['content'] }}`용 content **리스트**를 넘기고 있었다. 공유 InternVL 프로세서에서 물려받은 타일링 타이브레이크도 두 계열이 함께 읽는 업스트림 규칙과 어긋나, 레퍼런스가 5타일을 내는 768x768 이미지에 1타일을 냈다. 검증은 체크포인트 자신의 `modeling_llmjpvl.py`를 기준으로 했다. CLI에서 두 백본 모두 레퍼런스의 프롬프트 토큰 수와 greedy 출력을 정확히 재현하고, 서버에서는 Jagle-VL이 바이트 단위로 같으며 9B의 토큰 하나 차이는 배치 디코드 경로로 귀속된다(`--max-batch-size 1`에서 일치). 새 테스트 32개, 그리고 건드린 공유 파일의 회귀 범위에서 269개 통과.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1362는 LLM-jp 연구소의 일본어 VLM 계열인 `model_type: "llmjpvl"` 지원을 요청했다. 이 값을 선언한 배포 체크포인트는 둘이고, 아키텍처는 하나이며, 차이는 `llm_config.model_type`이 지목하는 디코더뿐이다. `llm-jp/llm-jp-4-vl-9B-beta`는 `llama`(hidden 4096, 32층, 어휘 196608, 비결속 LM 헤드), `llm-jp/Jagle-VL-2.2B-Jagle-FineVision`은 `qwen3`(hidden 2048, 28층, 결속 임베딩, 헤드별 q/k RMSNorm)이다. 둘 다 SigLIP2-so400m 타워(512px, 패치 16, 타일당 패치 1024개, 폭 1152의 27층, `gelu_pytorch_tanh`)를 InternVL식 동적 타일링 뒤에 두고, `pixel_shuffle(0.5)`로 2x2 패치를 접어 `[tiles, 256, 4608]`을 만든 다음 `LayerNorm / Linear / GELU / Linear` 구조의 `mlp1`으로 디코더 폭에 투영한다.

이슈는 InternVL 런타임을 재사용 대상으로 지목했고, 이 포트는 그대로 따른다. `InternVLConnector`, `InternVLProcessor`, `insert_internvl_image_tokens`, `SigLipVisionModel` 타워를 손대지 않고 쓴다. 새로운 것은 그것들을 둘러싼 배치다.

### 1.2 InternVL과 실제로 다른 지점

넷이고, 넷 다 틀렸을 때 조용하다.

- **디코더 설정 키.** 이 체크포인트들은 `llm_config`를 싣고 `text_config`는 없다. InternVL 로더는 `text_config`를 읽으므로 곧바로 실패하는데, 그건 오히려 안전한 실패다. 키를 바꾼 변환 체크포인트를 위해 `text_config`는 대체 경로로 받아들인다.
- **CLS 토큰 없음.** `InternVLChatVLM::get_input_embeddings`는 InternViT가 CLS 토큰을 앞에 붙이기 때문에 타워 출력에서 `[:, 1:, :]`을 잘라낸다. SigLIP에는 그 토큰이 없다. 그 슬라이스를 그대로 옮기면 진짜 패치 한 줄이 사라지고 나머지가 한 칸씩 밀리며, 오류 대신 근거 없는 유창한 출력이 나온다.
- **커넥터 엡실론.** 업스트림은 `mlp1`의 LayerNorm을 `nn.LayerNorm(vit_hidden_size * 4)`로만 만들고 `eps`를 넘기지 않으므로 torch 기본값 1e-5로 정규화한다. `vision_config.layer_norm_eps`는 1e-6이고, InternVL 로더는 같은 자리에서 그 값을 넘긴다.
- **`model_type` 하나에 디코더 둘.** 런타임이 Llama 그래프와 Qwen3 그래프 중 하나를, 로드 시점의 설정 문자열로 골라 실어야 한다.

### 1.3 이 저장소가 이미 한 번씩 당한 함정 둘

- **합성 대조는 체크포인트 증거가 아니다.** 두 체크포인트는 이미지 id와 정지 id가 전부 다르다(9B는 14 / 15 / 16에 정지 2, 11; Jagle-VL은 151655 / 151669 / 151670에 정지 151675, 151645, 151672). 상수를 박으면 잘해야 한쪽만 맞는다.
- **VLM은 서버에서 디코더 층을 하나도 돌리지 않으면서 CLI 테스트를 전부 통과할 수 있다.** CLI는 자기 캐시를 직접 만들기 때문에, `sequence_state_layout()`을 재정의하지 않고 `supports_batching() == false`이면 스케줄러가 빈 캐시 벡터를 넘겨받고 모델은 조용히 아무 일도 하지 않는다(Falcon-OCR 선례, PR #1075).

---

## 2. 기술 검토

### 2.1 라우팅과 디코더 둘을 싣는 런타임

`src/models/detection.rs`가 `"llmjpvl"`을 `ModelType::LlmJpVLM`으로 보낸다. 배포된 두 형태(Llama 9B, Qwen3 2.2B)의 `config.json`을 임시로 써서 둘 다 같은 모델 타입에 닿는지 확인하는 테스트로 고정했다. 검출기는 `llm_config`를 보지 않는다. `model_type` 하나가 두 체크포인트를 덮고, 디코더 선택은 라우팅이 아니라 로더의 일이다.

`src/vision/llmjp_vl_text.rs`에 `LlmJpTextModel`이 있다. `Llama3Model`과 `Qwen3Model` 위의 2분기 열거형으로, `LanguageModel`을 위임으로 구현하고 트레이트에 없으면서 병합에 필요한 `get_embed_tokens`를 더한다. 두 백본이 `LanguageModel` 말고는 공통 상위 트레이트가 없는 별개 타입이라 존재하는 타입이다. `qwen2`도 `llama`와 함께 받는데, mlxcel이 이미 그 그래프를 같은 Llama 백본으로 서비스하므로 디코더 이름만 바꾼 변환도 로드된다. 그 밖의 값은 값을 적시한 로드 오류이며, 가중치 로드 **전에** 낸다. 9B 체크포인트는 18GB이고, 지원하지 않는 백본을 가리킨 운영자는 1분이 아니라 1초 만에 알아야 한다.

### 2.2 타워, 그리고 여기 없는 CLS 슬라이스

`src/vision/llmjp_vl.rs`는 `SigLipVisionModel::from_weights_with_quant_and_gelu(&weights, &vision_cfg, "vision_backbone.vision_model", ...)`로 타워를 만들고 `post_layernorm` 출력을 통째로 쓴다. InternVL 런타임은 같은 자리에서 `[:, 1:, :]`을 자르는데, 여기서 그러면 패치 0번이 사라지고 나머지 1023개가 밀린다. 모델은 실패하지 않고 그대로 답한다.

conv 레이아웃은 따로 손댈 것이 없다. `VisionEmbeddings::from_weights`가 이미 형태 검사로 `[O, I, kH, kW]`를 `[O, kH, kW, I]`로 정규화하고, 배포된 bf16 원본(`[1152, 3, 16, 16]`)과 이미 변환된 체크포인트가 모두 그 검사를 만족한다. 어텐션 풀링 `head.*` 텐서는 로드 시점에 가중치 맵에서 버린다. 읽는 곳이 없고 bf16으로 20MB 정도다.

디코더 텐서는 복사가 아니라 **이동**으로 가중치 맵에서 꺼낸다. 9B 디코더가 18GB이고, 타워도 커넥터도 그것을 읽지 않는다.

### 2.3 커넥터 엡실론

`InternVLConnector::from_weights`는 비전 설정의 1e-6이 아니라 이름 붙인 상수 `LLMJP_MLP1_LAYER_NORM_EPS`(1e-5)로 호출한다. 단위 테스트가 상수를 `vision_config.layer_norm_eps`와 대조해 고정하고, 같은 가중치에서 두 값이 다른 임베딩을 낸다는 것까지 보이므로 이 선택은 우연이 아니다.

### 2.4 프롬프트 구성, 배치 위험이 사는 곳

업스트림 `LLMjpVLProcessor`는 템플릿이 렌더한 텍스트를 다시 쓰면서 `<image>`마다 `<|image_start|> + <|image_pad|> * (256 * tiles) + <|image_end|>`로 치환한다. 그 자리 표시자는 Harmony 사용자 턴 오프너 바로 뒤에 있다. `apply_chat_template`이 구조화된 content 조각들을 먼저 이어 붙인 뒤 템플릿이 `<|start|>user<|message|>{content}<|end|>`를 렌더하기 때문이다.

mlxcel은 템플릿을 먼저 렌더하고 토큰화하므로, `src/multimodal/llmjp_vl_prompt.rs`가 그 배치를 토큰 스트림 위에서 재현한다. InternVL의 대체 경로(프롬프트 첫 토큰 뒤에 삽입)는 여기서 틀린다. 렌더된 LLM-jp-VL 프롬프트의 첫 토큰은 **시스템** 턴의 `<|start|>`라서, 이미지 블록이 시스템 메시지 안으로 들어간다. 대신 이 모듈은 렌더된 텍스트에서 마지막 `<|start|>user<|message|>`를 찾고, 그 앞부분만 다시 인코딩해 호출자가 이미 만든 토큰 벡터의 진짜 접두인지 확인한 뒤 그 위치에 끼워 넣는다. 어긋나면 잘못된 위치에 끼우는 대신 맨 앞에 붙이는 쪽으로 물러나고, 어느 쪽이든 요청은 원래 토큰화를 그대로 유지한다. 이미 맨 `<|image_pad|>` 자리 표시자가 있는 프롬프트는 InternVL과 공유하는 제자리 확장 경로를 탄다.

요청별 타일 예산은 업스트림의 것을 옮겼고, 긴 프롬프트가 만드는 음수 중간값이 파이썬 `//`처럼 동작하도록 내림 나눗셈을 쓴다. `max_num = ((model_max_length - text_tokens) // images - 2) // image_seq_length - 1`을 `[1, max_dynamic_patch]`로 자른다.

`ensure_image_token_feature_cardinality`가 scatter를 지킨다. `merge_llava`는 `<|image_pad|>` 자리마다 특징 행을 하나씩 쓰므로, 블록을 잃거나 겹쳐 넣는 배치 결함은 뭉개진 텍스트가 아니라 오류로 드러난다.

### 2.5 생성 프롬프트, 양쪽 프런트엔드에서

배포된 `chat_template.jinja`는 `add_generation_prompt` 렌더를 `<|start|>assistant`에서 끝낸다. 체크포인트 자신의 프로세서가 그 뒤에 `<|channel|>final<|message|>`를 붙이므로, 템플릿 렌더만으로는 프롬프트가 미완성이고 모델은 Harmony 채널을 스스로 고르게 된다.

두 프런트엔드 모두 `ChatTemplateProcessor::from_model_path`로 프롬프트를 만들기 때문에, 이 마무리를 `ModelType::LlmJpVLM`에 키를 둔 `GenerationPromptSuffix` 규칙으로 그 안에 넣었다. 끝 문자열이 아니라 모델 계열에 키를 두는 것이 중요하다. 다른 Harmony 템플릿(gpt-oss)도 생성 렌더를 `<|start|>assistant`에서 끝내지만 final 채널로 몰아서는 안 된다. 규칙은 `add_generation_prompt`가 참이고 렌더가 실제로 그 오프너에서 끝날 때만 작동하므로, 프롬프트 캐시의 히스토리 경계 렌더는 건드리지 않는다.

프런트엔드 사이의 차이 하나를 더 닫아야 했다. CLI는 템플릿에 문자열 `content`를 넘기지만, OpenAI 요청은 `image_url` 조각이 든 content *리스트*를 싣고, `prepare_chat_request`는 타입 있는 미디어 조각이 있는 요청을 전부 raw JSON 렌더로 보냈다. 이 템플릿은 타입 있는 미디어 항목을 들여다보지 않으므로(`image`라는 문자열조차 없다) minijinja가 `{{ message['content'] }}`에 리스트를 받아 리스트 자체를 프롬프트에 찍었을 것이다. 이제 raw JSON 경로는 템플릿이 실제로 타입 있는 미디어를 읽을 때(또는 요청에 오디오가 있어 순서 sentinel이 필요할 때)만 탄다. 그 밖에는 content가 텍스트로 평탄화되며, 그것이 템플릿이 기대하는 형태이자 토큰 단위 확장이 가정하는 형태다. 테스트는 프롬프트 토큰 수가 아니라 서버 렌더와 CLI 렌더의 바이트 동일성을 확인한다. 서로 다른 프롬프트가 같은 토큰 수를 가질 수 있고, 이 저장소의 앞선 VLM 포트가 정확히 그렇게 숫자가 맞는 잘못된 프롬프트를 내보낸 적이 있다.

### 2.6 서버 캐시 레이아웃

`sequence_state_layout()`, `supports_batching()`, `supports_batched_prefill()`, `supports_maskless_padded_prefill()`, `supports_paged_decode_backend()`가 모두 트레이트 기본값으로 떨어지지 않고 텍스트 백본에 위임한다. 테스트는 두 백본 모두에 대해 레이아웃이 디코더 층마다 항목 하나를 가지는지와 `make_caches()`가 빈 벡터를 돌려주지 않는지를 확인한다. 여기서는 트레이트 기본값도 마침 맞지만, Falcon-OCR 선례가 남긴 교훈은 "우연히 맞는 것"이 바로 VLM이 CLI 테스트를 다 통과하면서 서버에서 디코더 층을 하나도 돌리지 않는 방식이라는 것이다.

세 개의 프레이밍 id는 `output_suppressed_token_ids()`가 돌려주므로 CLI도 스케줄러도 그중 하나를 텍스트 스트림에 뽑을 수 없다. 정지 id는 여전히 방출 가능한지도 테스트로 확인한다.

---

## 3. 기술적 결정

### 3.1 분할 지점을 믿지 말고 검증한다

이미지 블록은 업스트림의 `<image>`가 있던 자리, 즉 *텍스트* 위치에 들어가야 하는데 mlxcel은 호출자가 이미 만든 토큰 벡터 위에서 일한다. 삽입 지점 주변에서 프롬프트 전체를 다시 토큰화하면 경계가 달라질 위험이 있고, 토큰 인덱스를 추정해 끼우면 엉뚱한 턴에 들어갈 위험이 있다. 이 모듈은 둘 다 하지 않는다. `<|start|>user<|message|>`까지의 접두만 인코딩하고, 그 id들이 정말 호출자 토큰 벡터의 머리인지 확인한 뒤 그 길이 위치에 끼운다. 원래 토큰화는 그대로 보존되고, 토크나이저가 어긋나면 조용히 밀린 블록 대신 문서화된 대체 경로가 나온다. 호출자가 어느 `add_special_tokens`를 썼는지는 여기서 보이지 않으므로 둘 다 시도한다.

### 3.2 생성 프롬프트 마무리는 프런트엔드마다가 아니라 프로세서에서

접미사를 CLI 프롬프트 빌더와 서버 쪽에 각각 붙일 수도 있었다. 그러면 동기화할 곳이 둘이 되고, 앞선 포트가 프롬프트를 갈라지게 만든 바로 그 모양이 된다. `ChatTemplateProcessor::from_model_path`에 넣으면 두 프런트엔드가 이미 공유하는 하나의 이음매에 놓이고, "둘이 같다"가 두 번의 편집이 맞아떨어진 결과가 아니라 생성 시점의 성질이 된다. 규칙에 모델 계열로 키를 둔 이유는 끝 문자열(`<|start|>assistant`)만으로는 이렇게 마무리하면 안 되는 Harmony 계열과 구분되지 않기 때문이다.

### 3.3 이 계열만 특수 처리하지 않고 타입 있는 content 경로를 고친다

content 리스트를 문자열 템플릿에 넘기는 불일치는 LLM-jp-VL만의 문제가 아니다. 템플릿 텍스트가 아니라 토큰 id로 이미지를 나르는 모든 계열에 해당한다. 템플릿이 실제로 타입 있는 미디어를 읽는지로 raw JSON 경로를 가르는 쪽이, 애초에 받지 말았어야 할 렌더를 로더 하나가 우회하도록 가르치는 것보다 작고 정직한 변경이다. 타입 있는 미디어를 읽는 템플릿은 영향이 없고, 오디오 경로는 순서 sentinel을 그대로 유지한다.

### 3.4 `select_layer != -1`은 로드에서 거부한다

업스트림은 그 밖의 값에 대해 `hidden_states[select_layer]`를 읽는다. 대신 `last_hidden_state`를 주면 체크포인트가 학습하지 않은 특징이 되고, 모델은 그걸 근거로 답해 버린다. 로더는 값을 적시하며 실패한다.

---

## 4. 검증

호스트: Linux aarch64, NVIDIA GB10, CUDA sm_121. 바이너리는 `--profile test-fast --features cuda`로 빌드했다. 두 체크포인트 모두 MLX 변환본이 아니라 배포된 bf16 원본이므로, HF conv 레이아웃 분기(`[1152, 3, 16, 16]`)를 실제 가중치로 지난다.

### 4.1 레퍼런스 오라클

체크포인트 자신의 `modeling_llmjpvl.py`와 `processing_llmjpvl.py`를 `torch 2.14.0+cpu`(bf16, CPU)와 `transformers 4.57.1`에서 greedy, `max_new_tokens=32`로 실행했다.

| | Jagle-VL-2.2B | llm-jp-4-vl-9B-beta |
|---|---|---|
| 프롬프트 토큰 | 301 | 297 |
| 타일 / 이미지 pad | 1 / 256 | 1 / 256 |
| 출력 | `赤色です。` | `この画像は、単色の赤色で塗りつぶされた正方形の図形です。` |

레퍼런스 타일 수(두 체크포인트 동일): 224x224와 512x512는 1타일, 768x768과 1000x1000은 5타일(2x2 + 썸네일), 1024x768은 13타일(4x3 + 썸네일, 이미지 토큰 3328개).

### 4.2 CLI, 백본 둘 다

```
$ ./target/test-fast/mlxcel generate -m models/mlx/jagle-vl-2.2b-jagle-finevision \
    --image solid_red.png -p "この画像の色は何色ですか。" -n 32 --temp 0 --profile
LLM-jp-VL: inserted 1 image block(s), 1 tile(s) under a budget of 12 (256 total image tokens)
赤色です。
  Prompt tokens:    301
  Generated tokens: 4
```

```
$ ./target/test-fast/mlxcel generate -m models/mlx/llm-jp-4-vl-9b-beta \
    --image solid_red.png -p "この画像について説明してください。" -n 32 --temp 0 --profile
LLM-jp-VL: inserted 1 image block(s), 1 tile(s) under a budget of 12 (256 total image tokens)
この画像は、単色の赤色で塗りつぶされた正方形の図形です。
  Prompt tokens:    297
  Generated tokens: 15
```

프롬프트 토큰 수와 생성 텍스트가 양쪽 다 레퍼런스와 같다. 9B 답변은 연속 3회 실행에서 동일하게 재현됐다. Jagle-VL의 타일 수는 224x224가 1타일 256 이미지 토큰, 1024x768이 13타일 3328 이미지 토큰으로 레퍼런스와 일치한다.

### 4.3 서버, 백본 둘 다

같은 이미지를 `data:image/png;base64` content 조각으로 넣고 `temperature: 0`으로 `mlxcel-server`에 요청했다.

| | prompt_tokens | completion_tokens | content |
|---|---|---|---|
| Jagle-VL | 301 | 4 | `赤色です。` |
| llm-jp-4-vl-9B | 297 | 15 | `この画像は、単色の赤色で塗りつぶされた正方形のキャンバスです。` |

Jagle-VL은 CLI 및 레퍼런스와 바이트 단위로 같다. 9B는 토큰 하나가 다르고(CLI와 레퍼런스의 `図形` 자리에 `キャンバス`), 요청 3회에서 그 결과가 안정적으로 재현되므로 실행 간 잡음이 아니라 경로 차이다. 귀속도 가능하다. 같은 서버를 `--max-batch-size 1`(배치 스케줄러 없이 순차 처리)로 다시 띄우면 `図形`이 나와 CLI 및 레퍼런스와 정확히 일치한다. 프롬프트는 변수가 아니며, 이는 주장이 아니라 고정돼 있다. `one_image_prompts_are_token_exact_against_the_reference_processor`가 두 체크포인트 모두에 대해 조립된 프롬프트의 모든 id를 레퍼런스 프로세서의 `input_ids`와 비교하고, `the_server_image_request_renders_the_same_prompt_as_the_cli`가 두 프런트엔드 렌더의 바이트 동일성을 단언한다. 남는 것은 배치 디코드 경로의 리덕션 순서가 근소차 토큰 하나를 뒤집는 것이고, 이는 이 저장소가 게이트가 아니라 기록으로 다루는 드리프트 부류다(이슈 #932).

### 4.4 게이트

- `cargo clippy --lib --tests --features cuda -- -D warnings`: 통과.
- `cargo fmt --all`: 적용, 잔여 diff 없음.
- `cargo check --lib --tests --features cuda`: 통과.
- 새 단위 테스트(`--profile test-fast --features cuda`): `vision::llmjp_vl` 7/7, `multimodal::llmjp_vl_prompt`(대조 게이트 포함) 8/8, `loading::vlm::llmjp_vl` 7/7, `server::llmjp_chat_template_tests` 5/5, 검출 테스트 1/1, `models::metadata_tests` 4/4.
- 이 PR이 건드린 공유 파일의 회귀 범위: `vision::processors::internvl` 5/5(새 타이브레이크 게이트 포함), `server::chat_request` 104/104, `server::chat_template` 154/154, `vision::internvl` 2/2, `multimodal::internvl_prompt` 4/4.

---

## 5. 검증하지 못한 것

- **`cargo test --workspace --profile test-fast --features metal,accelerate`는 실행하지 않았다.** 이슈가 수용 기준으로 적었지만 이 호스트는 CUDA를 쓰는 Linux aarch64이고 Metal도 Accelerate도 없다. 위의 CUDA 등가물이 실제로 실행한 것이며 이 변경이 건드린 모듈로 범위를 좁혔다. 전체 워크스페이스 스위트는 머지 게이트에 맡긴다.
- **4.3의 배치 디코드 차이는 귀속했을 뿐 고치지 않았다.** 배치 스케줄러의 디코드 경로가 단일 시퀀스 경로와 비트 단위로 일치해야 하는가는 LLM-jp-VL이 아니라 저장소 전체의 질문이고, 여기서 그 경로를 바꾸지 않는다.
- **실제 체크포인트에서는 단일 이미지 요청만 돌렸다.** 다중 이미지 경로는 구현돼 있고 단위 테스트로 덮여 있지만(이미지별 타일 수가 블록 크기를 정하고, 블록이 이미지 순서대로 이어지며, 특징 개수 가드가 scatter를 지킨다), 실제 체크포인트에 이미지 둘을 넣어 보지는 않았다.
- **사진 이미지는 오라클과 대조하지 않았다.** mlxcel의 bicubic 리사이즈는 `image` 크레이트의 Catmull-Rom이고 업스트림은 PIL의 것이라, 사진에서는 타일 픽셀이 미세하게 다르고 greedy 토큰 동일성을 기대하지 않는다. 위에서 쓴 단색 이미지는 필터에 무관하며, 그래서 토큰 동일성 비교의 대상으로 삼았다.
- **양자화된 `llmjpvl` 변환본은 로드하지 않았다.** 로더는 최상위 `quantization` 블록을 디코더 설정으로 물려주고 group size와 bit 폭을 타워와 커넥터에 넘기지만(단위 테스트 있음), 이 계열의 4비트 체크포인트가 존재하지 않아 실행하지 못했다.
- **`select_layer != -1`은 구현이 아니라 거부이며**, 비디오 입력은 범위 밖이다.

---

## 6. 교훈

- **레퍼런스는 체크포인트 자신의 프로세서이고, 프롬프트만 뽑는 데는 비용이 거의 들지 않는다.** `transformers`로 `processing_llmjpvl.py`를 올려 `input_ids`만 물어보는 데는 몇 초와 GPU 0이 들고, 이 포트가 지금 고정 대조하는 정확한 id 열이 거기서 나왔다. 자기 자신과만 맞는 합성 픽스처는 어느 체크포인트에 대해서도 아무것도 증명하지 못했을 것이고, 두 체크포인트는 이미지 id와 정지 id가 전부 다르다.
- **프롬프트 토큰 수가 같다는 것은 두 프런트엔드가 같은 프롬프트를 렌더했다는 증거가 아니다.** 서버와 CLI는 서로 다른 content 형태로 템플릿에 닿고, 잘못된 렌더에서도 토큰 수는 살아남는다. 써 둘 가치가 있는 단언은 바이트 동일성이다.
- **재사용한 런타임은 이웃의 가정까지 함께 가져온다.** InternVL의 CLS 슬라이스와 "첫 토큰 뒤에 끼우는" 대체 경로는 InternVL에서는 둘 다 옳고 여기서는 둘 다 틀리며, 어느 쪽도 시끄럽게 실패하지 않는다. 재사용은 그럴 값어치가 있지만, 빌려 온 줄마다 새 체크포인트 자신의 코드에 대고 다시 유도해야 한다.
- **트레이트 기본값이 맞는 것과 레이아웃을 명시한 것은 다르다.** 여기서는 `sequence_state_layout()`이 위임으로 올바르게 풀렸겠지만, Falcon-OCR의 CRITICAL은 한 래퍼 앞에서 바로 그런 우연한 정확성에서 나왔다.
