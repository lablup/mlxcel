# 기술 보고서: PR #1414 - ColIdefics3와 ColQwen2.5 late-interaction 임베더

**작성일**: 2026-08-26
**작성자**: mlxcel maintainers
**상태**: 완료. 체크포인트 레이아웃 두 가지는 공개 아티팩트 대신 단위 테스트로 검증
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1414는 이슈 #1337이 요구한 late-interaction(ColBERT 계열) 시각 문서 검색기 두 종을 #1408의 `/v1/embeddings` 기반 위에 구현한다. ColIdefics3는 기존 SmolVLM / Idefics3 스택을, ColQwen2.5는 기존 Qwen2.5-VL 스택을 그대로 돌린다. 둘 다 디코더의 마지막 norm에서 멈추고, `Linear` 하나로 모든 토큰의 hidden state를 128차원으로 투영하고, 토큰 벡터마다 L2 정규화하고, 패딩 행을 0으로 만들고, 코사인 대신 MaxSim으로 순위를 매긴다. 디코더도 비전 타워도 새로 만들지 않는다. 실제 변경은 head 없는 조립, 투영 하나, 프롬프트 형식 두 개, 가중치 키 레이아웃 세 개, 그리고 확장된 이미지 프롬프트가 실제로 소비한 토큰 수를 보고하게 하는 트레이트 훅 하나다.

이슈 명세 중 두 가지가 실제 체크포인트와 맞지 않았고, 그것을 찾아낸 과정이 이 PR의 절반이다. ColQwen2.5 문서 프롬프트는 assistant 헤더가 아니라 `<|endoftext|>`로 닫히며, `vidore/colqwen2.5-base`는 비전 패치 필터를 `Conv3d`의 PyTorch 원본 레이아웃으로 저장하는데 이 트리의 인코더는 mlx 변환이 만드는 channels-last 레이아웃만 받는다. 둘 다 틀린 상태에서는 검색기가 무관한 페이지를 정답 페이지보다 높게 매겼고, 둘 다 고친 뒤에는 50.7퍼센트 마진으로 올바르게 매긴다.

---

## 1. 문제 정의

### 1.1 배경

ColIdefics3와 ColQwen2.5는 시각 문서 검색기다. 페이지를 이미지로 렌더링해 토큰 벡터 집합으로 임베딩하고, 질의의 토큰 벡터와 MaxSim(`sum_i max_j dot(q_i, d_j)`)으로 점수를 낸다. mlxcel은 두 백본을 이미 생성용으로 돌리고 있었으므로 빠진 조각은 좁았지만, 하나하나가 시끄럽게 실패하지 않고 조용히 틀리는 종류였다.

- 두 로더 모두 `lm_head`를 요구한다. `SmolVLMModel`은 `Llama3Model`을 만들고, 그 생성자는 tied든 untied든 항상 head를 읽는다. ColIdefics3 체크포인트는 `tie_word_embeddings: false`를 선언하면서 head 텐서를 아예 담지 않으므로 그 생성자로는 열리지 않는다.
- `Qwen2VLModel::forward_for_sequence`는 `lm_head`로 끝난다. 임베더가 이 경로로 hidden state에 도달하면 마이크로배치마다 `[B, L, 151936]` 로짓 텐서를 만들고 곧바로 버리게 된다.
- `/v1/embeddings` 엔진은 반환 행 수와 `usage.prompt_tokens`를 패딩 전 토큰화 행에서 유도한다. VLM 프롬프트는 프로세서가 수백 개의 이미지 토큰으로 확장하는 플레이스홀더 하나를 담으므로, 훅이 없으면 forward pass가 800토큰으로 돌린 시퀀스를 응답은 열두 행이라고 주장하게 된다.
- `mlxcel embed`는 코사인 행렬을 출력했다. multi-vector 계열에는 잘못된 점수 규칙이며, 실패하는 대신 조용히 나쁘게 순위를 매긴다.

### 1.2 에픽이 요구한 것

합성이 아니라 실제 체크포인트다. `vidore/colSmol-256M`과 `vidore/colqwen2.5-v0.2`가 학습된 검색기인데, 둘 다 `-base` 저장소 위의 PEFT LoRA 어댑터이고 학습된 투영은 sentence-transformers `1_Dense/` 폴더에만 들어 있다. 어댑터 병합은 mlxcel의 범위 밖으로 명시되어 있어 검증 문제가 남는다. base 저장소는 로드되지만 투영이 무작위 초기화 상태라 가중치 레이아웃만 증명할 뿐 검색 품질은 아무것도 말해 주지 않는다.

---

## 2. 기술적 선택과 그 이유

### 2.1 `llama3.rs`를 건드리는 대신 head 없는 Llama를 조립

이슈 #1337은 `src/models/llama3.rs`에 `forward_hidden`과 `from_weights_without_head`를 추가하자고 제안하면서, #1325도 같은 함수를 원하니 먼저 머지되는 쪽이 이긴다고 적었다. 이번에는 둘이 병렬로 돌았고, `llama3.rs`는 그 쌍에서 가장 경합이 심한 파일이다.

그래서 건드리지 않기로 했다. `Llama3Model`의 필드는 모두 public이고 `TransformerBlock`, `UnifiedEmbedding`, `RMSNorm`도 public 생성자를 가진 public 타입이므로, head 없는 백본은 아키텍처의 두 번째 구현이 아니라 조립 방식의 차이다. `src/models/headless_llama.rs`는 98줄로 같은 블록을 같은 rope 테이블로 만들고 같은 최종 norm에서 멈춘다. 대가는 의도적 누락 하나다. `Llama3Model::forward`는 레이어 사이에서 private `pipeline_hint`를 호출하는데 이 경로는 그것을 건너뛴다. 그 힌트는 파이프라인 병렬 스케줄링 주석으로 수치에 영향이 없고, 임베딩 forward는 단일 장치에서의 한 번의 패스다.

이슈 본문에서 벗어난 선택이므로, 머지 충돌을 빼고 보아도 이쪽이 나은 이유를 적어 둔다. `Llama3Model`에 `from_weights_without_head`를 두면 `lm_head` 필드를 `None`으로 만들 수 없으니 타입을 바꾸지 않는 한 더미 head, `Option`, 또는 별도 구조체 중 하나가 된다. 이것이 그 별도 구조체다.

### 2.2 `Qwen2VLModel::forward_for_sequence`에서 `forward_hidden` 분리

`Qwen2VLModel`의 필드는 private이므로 ColQwen2.5는 분리가 실제로 필요하다. 기계적인 분리다. `norm`까지가 `forward_hidden`으로 옮겨지고, `forward_for_sequence`는 그 호출 뒤에 `self.lm_head.forward(&h)`가 된다. `forward_hidden_then_head_matches_forward`가 2레이어 합성 모델에서 두 경로의 로짓이 비트 단위로 같음(`max_abs_diff == 0.0`)을 단언하므로, "생성 경로는 그대로다"가 다짐이 아니라 검사된 진술이 된다.

`qwen2_vl.rs`는 병렬 유닛 중 어느 쪽도 건드리지 않으므로 이 분리에는 조율 비용이 없다.

### 2.3 이미지 프롬프트 확장은 `embed` 안이 아니라 엔진에서

엔진의 `postprocess`는 `token_counts`로 `[B, L, D]`를 항목별 행렬로 자르는데, 이 값은 `EncodedBatch`에서 오고 그것은 계열이 아무것도 보기 전에 만들어진다. 계열이 `embed` 안에서 `<image>`를 832 토큰으로 확장하면 출력은 870행이 되고 엔진은 그중 앞 12행만 돌려주며 프롬프트 토큰도 12라고 보고한다.

그래서 `EmbeddingModel::expand_image_tokens(ids, images) -> Result<Vec<u32>>`를 `EmbeddingEngine::embed_image`가 `encode_row`와 `EncodedBatch::from_rows` 사이에서 호출한다. 기본 구현이 항등이므로 기존 계열은 바뀌지 않고 다른 계열이 이 문제를 고민할 필요도 없다. 두 새 계열은 전처리의 값싼 절반(ColIdefics3는 타일 레이아웃, ColQwen2.5는 `compute_grid_thw`)만 계산하고 실제 토큰 삽입은 생성 경로의 `insert_smolvlm_image_tokens`와 `insert_qwen_vl_image_tokens`에 위임한다. 비싼 절반인 픽셀 추출은 여전히 `embed`에서 한 번만 일어난다.

대안으로 `EmbeddingOutput`이 자체 행 수를 들고 오게 하는 방법을 검토했다. 계열 하나를 위한 계약 변경치고 크고, `usage.prompt_tokens`의 권한을 다른 모든 계열이 두고 있는 토크나이저 밖으로 옮기게 된다.

### 2.4 `1_Dense` 폴더가 루트 투영보다 우선

병합된 export는 투영을 두 개 갖는다. base 저장소의 학습되지 않은 `linear.*` 또는 `custom_text_proj.*`가 메인 샤드에, 학습된 쪽이 `1_Dense/model.safetensors`에 있고 후자는 `load_weights_from_dir_with_subfolders`가 `1_Dense.linear.*`로 노출한다. sentence-transformers는 모듈 폴더를 적용하므로 폴더가 이겨야 하고, `apply_dense_projection_override`가 그것을 루트 키 위로 옮긴다.

한 가지 미묘한 점을 명시적으로 처리한다. 루트 투영이 양자화되어 있고 폴더가 dense이면, 루트의 `scales`와 `biases`를 남겨 둘 경우 `UnifiedLinear::from_weights`가 폴더의 dense 가중치를 packed로 취급한다. 그래서 override는 폴더가 직접 제공하지 않은 양자화 텐서를 제거한다.

`embedding_sanitize::fold_dense_modules`는 의도적으로 재사용하지 않았다. 그것은 `{N}_Dense.linear.*`를 `dense.{k}.*`로 바꾸는 EmbeddingGemma의 pooling 후 체인용이고, 여기서는 폴더가 목록에 덧붙는 것이 아니라 이름 붙은 투영을 대체한다.

### 2.5 LoRA만 있는 저장소는 해결책을 담아 거부

`reject_lora_only_checkpoint`는 `adapter_model.safetensors`만 있고 다른 최상위 샤드가 없는 디렉터리를 병합을 해결책으로 지목하며 거부한다. base를 로드하고 어댑터를 무시하는 대안은 형태도 맞고 정규화도 맞지만 의미가 전혀 없는 벡터를 내는 학습되지 않은 검색기를 서빙하게 된다. 순위를 측정하지 않는 게이트는 정확히 이 실패를 잡지 못하므로 로드 시점에 거부한다.

### 2.6 MaxSim은 평균이 아니라 원값으로 보고

`crate::embeddings::maxsim`은 ColBERT 합을 돌려주므로 질의를 자기 자신과 비교하면 자기 행 수가 나온다. `mlxcel embed`는 이전에 multi-vector 점수를 질의 길이로 나눴는데, 그러면 코사인처럼 읽히고 이 숫자를 검증 가능하게 만드는 성질이 가려진다. 대각선의 정확한 `24.0000`은 모든 질의 행이 단위 벡터이며 문서에 대한 최대값이 동일 행에서 달성된다는 직접 증거다. 두 실제 체크포인트 실행 모두 그 대각선을 보인다.

### 2.7 텍스트 배치 전에 M-RoPE 슬롯 비우기

`Qwen2VLModel::forward_hidden`은 저장된 `[3, B, L]` 위치 그리드의 배치가 맞고 요청 범위를 덮으면 그것을 재사용한다. 이미지 요청은 정확히 그런 그리드를 fallback 슬롯에 남긴다. 배치 1이고 폭이 같거나 짧은 텍스트 마이크로배치는 그래서 이미지의 공간 위치를 받게 된다. `text_input_embeddings`가 먼저 `clear_mrope_state()`를 호출하고, `stale_image_positions_do_not_leak_into_a_text_batch`가 반복되는 가짜 그리드를 심어 놓고 텍스트 결과가 변하지 않음을 단언한다.

---

## 3. 명세 수정 두 가지

### 3.1 ColQwen2.5 문서 프롬프트

이슈는 `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|im_start|>assistant\n`을 명시한다. 참조 구현 `ColQwen2_5_Processor.visual_prompt_prefix`는 대신 `<|endoftext|>`로 끝나며, 이는 현재 `colpali-engine` main과 `v0.2` 체크포인트가 학습된 0.3.x 계열 양쪽에서 동일하고, `vidore/colqwen2.5-v0.2`가 담고 있는 `additional_chat_templates/sentence_transformers.jinja`도 같은 문자열을 낸다. 검색된 페이지 뒤에는 아무것도 생성되지 않으므로 assistant 헤더는 단지 다른 것이 아니라 모델이 학습 중 본 적 없는 턴 열기다.

### 3.2 원본 HuggingFace 패치 임베딩 레이아웃

`Qwen25VLVisionEncoder::PatchEmbed`는 5차원 `visual.patch_embed.proj.weight`를 읽어 `[0, 2, 3, 4, 1]`로 치환해 `[out, T, C, H, W]`에 도달한다. 그 치환은 mlx-vlm 변환의 `[out, kT, kH, kW, in]`에 대해 옳고, 생성 로더 `load_qwen2_5_vl`은 `mlx-community` 변환에만 쓰이므로 그것이 유일하게 보는 레이아웃이다. `vidore/colqwen2.5-base`는 원본 HuggingFace export이고 `Conv3d`의 네이티브 `[out, in, kT, kH, kW]`, 즉 `[1280, 3, 2, 14, 14]`를 저장한다. 이것을 치환하면 `[1280, 3, 14, 2, 14]`가 되고, 원소 수는 맞으므로 어떤 shape 단언도 실패하지 않은 채 뒤섞인 필터가 된다.

`normalize_patch_embed_layout`은 채널 축으로 레이아웃을 판별하고(`in_channels`는 3, 패치 크기는 14라 마지막 축이 3인 것은 변환된 레이아웃뿐이다) 인코더가 보기 전에 mlx-vlm의 변환 단계 `[0, 2, 3, 4, 1]`을 적용한다.

### 3.3 두 수정이 만든 차이

둘 다 읽어서가 아니라 측정해서 찾았다. 병합된 체크포인트에 대한 첫 ColQwen2.5 실행은 무관한 페이지를 정답보다 높게 매겼고, 두 페이지가 서로에 대해 자기 유사도의 86퍼센트를 기록했다. 이는 프롬프트 형식 문제가 아니라 비전 타워가 거의 상수인 특징을 내고 있다는 신호다.

| 측정 | 두 수정 전 | 두 수정 후 |
| --- | --- | --- |
| MaxSim, 질의 대 매출 표 | 7.8338 | 18.3363 |
| MaxSim, 질의 대 무관 페이지 | 8.0446 | 9.0444 |
| 순위 | 틀림 | 맞음 |
| MaxSim, 표 대 무관 페이지 | 781 중 671.61 (86퍼센트) | 779 중 247.58 (32퍼센트) |

ColIdefics3 프롬프트는 이슈 명세 그대로 두었다. 이 체크포인트도 같은 조각을 다르게 배열한 두 번째 렌더링(`<|im_start|>User: Describe the image.<image><end_of_utterance>`)을 담고 있어 병합 체크포인트에서 둘 다 측정했다. 프로세서 형식이 53.64퍼센트, 템플릿 형식이 53.77퍼센트의 관련성 마진을 냈다. 실질적으로 동등하므로 이슈와 참조 프로세서가 일치하는 형식을 유지하고, 측정값을 해당 상수 옆에 기록해 두었다.

---

## 4. 구현 상세

### 4.1 모듈 구성

| 파일 | 역할 |
| --- | --- |
| `src/models/col_late_interaction.rs` | `embedding_dim`, `reject_lora_only_checkpoint`, `apply_dense_projection_override`, `project_and_normalize`, `format_query` |
| `src/models/headless_llama.rs` | `HeadlessLlama`: 임베딩 테이블, 블록, 최종 norm, `forward_hidden` |
| `src/models/colidefics3.rs` | `ColIdefics3Model`, SmolVLM 가중치 재매핑, 타일 프로세서, 마커 인코딩 |
| `src/models/colqwen2_5.rs` | `ColQwen25Model`, `rewrite_colqwen25_key`, `normalize_patch_embed_layout`, `text_input_embeddings`, `token_vectors` |
| `src/embeddings/maxsim.rs` | 행 기반 `maxsim`, 디바이스 배열 기반 `maxsim_mlx` |

### 4.2 공유되는 forward 꼬리

`project_and_normalize(hidden, projection, attention_mask)`는 투영 결과를 f32로 캐스팅하고 각 토큰 행을 `max(||row||, 1e-9)`로 나눈 뒤 `[B, L, 1]`로 브로드캐스트한 마스크를 곱한다. 활성 dtype이 아니라 f32에서 정규화하는 것이 측정된 단위노름 오차를 bf16 ulp가 아니라 1e-7로 만들고, epsilon이 전부 0인 행을 NaN이 아니라 정확히 0으로 유지한다. 엔진은 한 번 더 정규화하는데 단위 행에는 항등이고 0은 0으로 남으므로 `EmbeddingModel::normalize()`는 `true`로 두고 `dimensions` 절단 후 재정규화도 그대로 동작한다.

### 4.3 ColQwen2.5 키 정리

로더에는 세 가지 레이아웃이 들어온다. `rewrite_colqwen25_key`는 앞의 `vlm.`을 먼저 떼어 나머지 규칙을 두 번 쓰지 않게 하고, 이어서 `embedding_proj_layer.`를 `custom_text_proj.`로, `model.language_model.`과 맨 `language_model.`을 `model.`로, `model.visual.`과 `visual.`을 `vision_tower.`로 매핑한다. tied `lm_head`는 버리고 `tie_word_embeddings`는 로드 시 true로 강제하는데, 이것이 151936 곱하기 2048짜리 head를 메모리와 생성자 양쪽에서 치운다. `sanitize_tied_embeddings`는 일부러 호출하지 않는다. 이 경로가 읽지도 않는 head를 위해 임베딩 테이블을 `lm_head.*`로 복사하기 때문이다.

### 4.4 이미지 배치는 한 행이다

`compute_rope_index`는 `input_ids`의 0번 행을 읽어 그리드 하나를 유도한다. 엔진은 이미지를 한 번에 하나씩 임베딩하므로 이것은 제약이 아니라 불변식이고, `ColQwen25Model::forward`는 이제 그것이 깨지면 1번 이후 행에 0번 행의 위치를 주는 대신 시끄럽게 실패한다.

---

## 5. 테스트 전략

### 5.1 합성 체크포인트가 실제 로드 경로를 돈다

`colidefics3_tests.rs`는 완전한 합성 체크포인트를 디스크에 만든다. `config.json`, `preprocessor_config.json`, 타일 마커 문자열을 어휘에 담은 `WordLevel` `tokenizer.json`, 그리고 텍스트 백본과 SigLIP 타워와 커넥터와 투영을 모두 담은 f32 safetensors 샤드 하나다. 8픽셀 이미지, 4픽셀 패치, pixel-shuffle 2로 잡아 타일 하나가 정확히 이미지 토큰 하나로 압축된다. 모든 테스트는 `1_Dense` override와 이미지 플레이스홀더 확장을 포함해 `ColIdefics3Model::load(dir, config)`를 통과한다. `sanitize_prefers_1_dense_projection`은 가중치가 상수인 폴더를 써서, override가 올바르면 모든 정규화 행이 같은 상수 단위 벡터가 되게 한다. 무작위 루트 투영으로는 나올 수 없는 결과다.

`colqwen2_5_tests.rs`는 텍스트 경로를 맨 `Qwen2VLModel` 위에서, 모델이 쓰는 것과 같은 `text_input_embeddings`와 `token_vectors`를 통해 돌린다. 마스크와 head 없는 forward와 정규화가 다시 쓰인 코드가 아니라 제품 코드로 검증된다. 비전 절반은 32블록 windowed ViT를 손으로 만드는 대신 아래의 실제 체크포인트 게이트가 덮는다. 손으로 만들면 인코더를 테스트하는 것이 아니라 다시 쓰는 셈이 된다.

### 5.2 MLX 테스트 가드

이 모듈들에서 모델을 만들거나 MLX 연산을 평가하는 모든 테스트는 병합된 임베딩 계열들이 이미 공유하는 프로세스 전역 잠금 `crate::models::embedding_test_support::mlx_test_guard`를 잡고, 게이트 수치는 `--test-threads=1`로 기록했다. `EmbeddingModel`은 계약상 단일 스레드이고 제품은 그것을 지킨다. libtest는 지키지 않으며, 이 트리에서 동시 MLX 작업이 결과를 조용히 오염시키고 `cudaStreamEndCapture`에서 프로세스를 죽이는 것이 둘 다 관측되었다.

### 5.3 실제 체크포인트 게이트는 부드럽게 건너뛴다

`real_colsmolvlm_base_loads_and_projects_to_128`과 `real_colqwen25_base_loads_and_projects_to_128`은 공개 base 저장소가 있으면 로드해 유도된 기하(ColIdefics3는 타일당 64개 특징 행, ColQwen2.5는 768 곱하기 28 곱하기 28 픽셀 상한)와 토큰 id와 프롬프트 형식을 단언한다. 체크포인트가 없으면 메시지와 함께 조기 반환하는데, `src/embeddings/real_checkpoint_tests.rs`가 따르는 관행이다.

---

## 6. 실제 체크포인트 결과

### 6.1 학습된 체크포인트 만들기

공개된 검색기 둘 다 LoRA 어댑터다. 각각을 저장소 밖의 독립 스크립트로 base에 병합했다(`W' = W + (B @ A) * alpha / r`, `alpha / r = 32 / 32 = 1.0`). ColIdefics3는 210개, ColQwen2.5는 253개 텐서가 병합되었다. 병합을 그냥 믿지는 않았다. `vidore/colqwen2.5-v0.2`는 `custom_text_proj`에 LoRA를 갖는 동시에 완전히 학습된 투영을 `1_Dense/`에도 담고 있으므로 둘이 일치해야 한다. 실제로 최대 0.104 크기의 가중치에서 2.44e-4까지 일치했고(그 크기에서의 bf16 반올림이다) bias는 정확히 같았다. 이 비교 하나가 스케일과 `B @ A` 방향과 대상 키 매핑을 검증하며, 독립 기준이 없는 252개 백본 델타까지 함께 뒷받침한다.

### 6.2 게이트

질의 `What was the total revenue in 2023?`, 렌더링한 매출 표 페이지와 무관한 페이지, 각각 3회 반복, GB10 CUDA.

| 게이트 | ColIdefics3 (colSmol-256M 병합) | ColQwen2.5 (colqwen2.5-v0.2 병합) |
| --- | --- | --- |
| 형태 | `[24, 128]`, `[876, 128]`, `[876, 128]` | `[24, 128]`, `[779, 128]`, `[779, 128]` |
| 행 수 대 `usage.prompt_tokens` | 1776 = 1776 | 1582 = 1582 |
| 최악 단위노름 오차 (상한 1e-5) | 1.03e-7 | 0.0 |
| MaxSim, 질의 대 정답 페이지 | 18.7396, 3회 편차 0 | 18.3363, 3회 편차 0 |
| MaxSim, 질의 대 무관 페이지 | 8.6879, 3회 편차 0 | 9.0444, 3회 편차 0 |
| 관련성 마진 (상한 10퍼센트) | 53.6퍼센트 | 50.7퍼센트 |
| 같은 입력, 다른 프로세스 | 0.0 | 0.0 |
| 질의 대 자기 자신의 MaxSim | 정확히 24.0000 | 정확히 24.0000 |

`POST /v1/embeddings`도 모든 수치를 그대로 재현한다. 질의는 텍스트 항목 하나, 두 페이지는 `image_url` data URI로 보냈다. `base64` 모드에서는 페이로드가 같은 float으로 비트 단위 동일하게 디코딩되고 형제 필드 `shape`를 함께 담는다.

### 6.3 충족하지 못한 게이트 하나

에픽은 패딩된 배치가 패딩 없는 단일 입력 결과와 1e-3 이내로 일치할 것을 요구한다. f32에서는 성립하며, 합성 단위 테스트가 그 허용오차로 통과한다. 이 CUDA 박스의 bf16에서는 성립하지 않는다. 같은 행을 배치 1과 배치 2에서 임베딩했을 때 ColIdefics3는 최대 1.5e-2, ColQwen2.5는 7.5e-3 차이가 난다.

네 가지 관찰이 이것을 이 변경 밖에 둔다.

- 패딩 행은 정확히 0이므로 마스크와 0 처리는 연루되지 않는다.
- 패딩이 없는 동일한 두 행이 한 배치 안에서 이미 각각 8.6e-3, 6.6e-3 차이가 나므로 패딩도 연루되지 않는다.
- 모든 측정이 결정적이다. 3회 반복이 비트 단위로 같고 별개 프로세스 둘이 0.0으로 일치하므로 경쟁 조건은 배제된다.
- #1413에서 병합되고 이번에 손대지 않은 `Qwen/Qwen3-Embedding-0.6B`도 같은 박스에서 배치 1과 배치 2 사이에 3.8e-3 차이를 낸다.

여기 기록하는 결론은 이 하드웨어의 배치 bf16 prefill이 1e-3 수준에서 배치 크기 불변이 아니며, multi-vector 계열은 pooling 평균 대신 토큰별 벡터를 보고하기 때문에 그것이 눈에 띌 뿐이라는 것이다. 순위는 어느 쪽도 바뀌지 않는다. 위 마진이 50퍼센트대이고 드리프트는 성분 하나의 2퍼센트 미만이다.

---

## 7. 검증 요약

| 검사 | 결과 |
| --- | --- |
| `cargo fmt --all -- --check` | 통과 |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | 통과 |
| `cargo check --profile test-fast --features cuda --all-targets` | 통과 |
| `cargo test ... --lib models::col -- --test-threads=1` | 21 통과, 0 실패 |
| `cargo test ... --lib embeddings:: -- --test-threads=1` | 69 통과, 0 실패 |
| 병합된 두 체크포인트에 대한 `mlxcel embed` | 통과, 수치는 6절 |
| `mlxcel-server` 및 `POST /v1/embeddings`, float과 base64 | 통과, 수치는 6절 |

---

## 8. 변경 요약

19개 파일, 2758줄 추가, 35줄 삭제.

신규: `src/models/col_late_interaction.rs`와 테스트, `src/models/headless_llama.rs`, `src/models/colidefics3.rs`와 테스트, `src/models/colqwen2_5.rs`와 테스트, `src/embeddings/maxsim.rs`와 테스트.

수정: `src/models/qwen2_vl.rs`(`forward_hidden` 분리), `src/embeddings/model.rs`(`expand_image_tokens`), `src/embeddings/engine.rs`(그 호출), `src/embeddings/loader.rs`(두 arm), `src/embeddings/mod.rs`, `src/models/mod.rs`, `src/embeddings/real_checkpoint_tests.rs`, `src/commands/embed.rs`(MaxSim 행렬), `docs/supported-models.md`, `docs/embeddings.md`.

---

## 9. 검증되지 않은 부분

- **네이티브 `colqwen2` / `ColQwen2ForRetrieval` 레이아웃.** 해당 체크포인트를 구할 수 없었다. 다섯 개 재매핑 규칙은 키 문자열 단위 테스트로만 덮여 있고 실제 로드로는 덮이지 않았다.
- **참조 구현 대조 실행.** 이 검증 호스트에는 `colpali-engine`, `transformers`, PyTorch가 설치되어 있지 않아 참조 forward pass와의 나란한 비교가 불가능했다. 대신 각 체크포인트가 스스로 담고 있는 것을 기준으로 삼았다. 학습된 `1_Dense/model.safetensors`(병합을 대조한 대상)와 `additional_chat_templates/sentence_transformers.jinja`(프롬프트를 대조한 대상), 그리고 공개된 프로세서 소스다.
- **macOS와 Metal.** 여기 모든 실행은 Linux CUDA였다. Apple Silicon에서 적용되는 bf16에서 f16으로의 변환 규칙도, Metal 어텐션 경로도 전혀 실행되지 않았다.
- **양자화 변환본.** 공개된 ColIdefics3 / ColQwen2.5 양자화 체크포인트가 없다. 양자화 배관(`quantization_params`, 폴더가 제공하지 않은 `scales`와 `biases`를 버리는 `1_Dense` override)은 작성되고 단위 테스트되었으나 실제 양자화 아티팩트로 로드된 적은 없다.
- **입력당 이미지 두 장 이상.** 엔진은 이미지를 하나씩 임베딩하고 ColQwen2.5는 이제 다중 행 이미지 배치를 명시적으로 거부한다. 다중 이미지 입력은 구현도 테스트도 되지 않았다.
- **유도된 `max_length`보다 긴 문서.** 이미지 확장은 절단 이후에 일어나므로 아주 큰 페이지는 원리상 상한을 넘을 수 있다. 공개된 기하에서는 불가능하다. ColIdefics3는 약 1100 이미지 토큰, ColQwen2.5는 768에서 멈추고 상한은 8192다.

---

## 10. 후속 조치

- bf16 prefill의 배치 크기 민감성은 에픽의 수치 게이트 문구에 대한 별도 이슈로 다룰 가치가 있다. 이 쌍이 아니라 CUDA의 모든 임베딩 계열에 적용되는 성질이기 때문이다.
- `crate::embeddings::maxsim_mlx`는 아직 제품 호출자가 없다. #1337이 범위 밖으로 둔, 저장된 multi-vector 문서에 대한 검색 또는 스코어링 엔드포인트가 그 자리다.
- 네이티브 `ColQwen2ForRetrieval` export를 구할 수 있게 되면 `local_embedding_checkpoints_detect_to_their_families`와 ColQwen2.5 게이트에 추가해 `vlm.` 래핑 레이아웃을 문자열 테스트가 아니라 로드로 덮어야 한다.
- `normalize_patch_embed_layout`은 이 계열에 한해 원본 HuggingFace Conv3d 레이아웃을 고친다. 생성 로더 `load_qwen2_5_vl`도 같은 사각지대를 갖고 있어 원본 Qwen2.5-VL export를 같은 방식으로 잘못 로드할 것이다. 이는 별개의 기존 이슈로 등록할 만하다.

---

## 참고

- 이슈 #1337, 에픽 #1348
- PR #1408 (임베딩 기반), 이슈 #1353
- PR #1411 (BERT와 XLM-RoBERTa), PR #1413 (EmbeddingGemma와 Qwen3-Embedding)
- `docs/embeddings.md`, `docs/supported-models.md`
- `vidore/ColSmolVLM-Instruct-256M-base`, `vidore/colSmol-256M`, `vidore/colqwen2.5-base`, `vidore/colqwen2.5-v0.2`
- `colpali-engine` 프로세서: `ColIdefics3Processor`, `ColQwen2_5_Processor`
