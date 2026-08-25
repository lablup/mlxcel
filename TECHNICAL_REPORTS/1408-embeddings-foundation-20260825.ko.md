# 기술 보고서: PR #1408 - 임베딩 기반 구조: 풀링, 마스크, Embedding 종류와 /v1/embeddings

**날짜**: 2026-08-25
**작성자**: mlxcel maintainers
**상태**: 완료. 패밀리별 forward pass는 epic #1348의 하위 이슈로 이관
**언어**: Rust, Markdown
**위험 수준**: 중간

---

## 요약

PR #1408은 이슈 #1353(epic #1348)이 요구한 공통 임베딩 기반 구조를 추가합니다. 이전의 mlxcel에는 임베딩 경로가 전혀 없었습니다. 모든 체크포인트는 텍스트 생성기 아니면 VLM이었고, 감지는 `model_type`만 읽었으며, attention 마스크는 causal 계열뿐이었고, sentence-transformers 모듈 하위 폴더는 읽히지 않았으며, `/v1/embeddings` 라우트도 없었습니다. 이 PR은 `src/embeddings/` 서브시스템(`EmbeddingModel` 트레이트, `1_Pooling/config.json` 리더를 갖춘 네 가지 풀링 모드, L2 정규화, `dimensions` 절단, `max_length` 유도, 오른쪽 패딩 배치 토큰화, 패밀리 디스패처, 길이 정렬 마이크로배치 엔진), `mlxcel-core`의 패딩 인식 마스크 빌더 세 개와 하위 폴더 인식 safetensors 로더, 열세 개 `ModelType` 변형과 레이아웃 기반 감지를 갖춘 `ModelKind::Embedding`, 전용 임베딩 워커 스레드, OpenAI 호환 `POST /v1/embeddings` 라우트, `--embedding-*` 서버 플래그 다섯 개, `mlxcel embed` 명령, `docs/embeddings.md`를 추가합니다.

패밀리 forward pass는 포함하지 않습니다. 감지된 모든 패밀리는 자체 포트가 머지되기 전까지 로더에서 `not yet supported`를 보고하며, 라우트, 워커, 배치, 풀링, 인코딩은 테스트 전용 스텁 모델로 끝까지 검증했습니다. 감지, 풀링 설정 파싱, 길이 유도, 토큰화는 실제 `all-MiniLM-L6-v2`와 `Qwen3-Embedding-0.6B` 체크포인트로 확인했습니다.

---

## 1. 문제 정의

### 1.1 배경

Epic #1348은 여러 임베딩 패밀리(BERT와 MiniLM, XLM-RoBERTa, ModernBERT, SigLIP text, EmbeddingGemma, Qwen3-Embedding, Qwen3-VL-Embedding, LFM2, Ministral 3, Llama 양방향 임베더, Llama-Nemotron-VL, ColIdefics3, ColQwen2.5)를 포팅합니다. 각 패밀리에는 같은 기반이 필요합니다. 생성기가 아닌 임베더로 인식되는 방법, f16에서 유한하게 유지되는 패딩 인식 attention 마스크, 풀링과 정규화, 배치 엔진, 워커 스레드, HTTP 표면입니다. 이슈 #1353은 이 기반을 분리하여 패밀리 포트가 forward pass만 추가하도록 합니다.

### 1.2 기존 한계

- `ModelKind`에는 `Text`와 `Vlm`만 있었고 `load_model`은 항상 `LanguageModel`을 반환했으므로, 임베딩 체크포인트는 명확한 메시지 대신 `lm_head` 누락 같은 텐서 부재 증상으로 실패했습니다.
- `get_model_type`은 `model_type`만 읽었습니다. Qwen3-Embedding(`model_type: qwen3`)과 EmbeddingGemma(`gemma3_text`)는 causal 생성기로 라우팅됐습니다.
- `mlxcel_core::utils`에는 causal 마스크(`create_causal_mask`, window 변형, left-padding 변형)만 있었습니다. 인코더 패밀리는 양방향으로 패딩 키를 차단하는 마스크가 필요합니다.
- `load_weights_from_dir`는 shard 인덱스나 최상위 `*.safetensors`만 나열하여 sentence-transformers 내보내기의 `2_Dense/model.safetensors` 투영을 조용히 버렸습니다.
- 서버에는 Whisper와 Kokoro용 `audio_model` 슬롯과 워커는 있었지만 임베딩에 해당하는 것은 없었고 `/v1/embeddings` 라우트도 없었습니다.
- `MlxcelTokenizer::encode`는 id만 반환했습니다. 배치 패딩도, `token_type_ids`도, 일부 `tokenizer.json`이 고정해 둔 패딩을 제어하는 수단도 없었습니다.

### 1.3 위험 평가

| 위험 | 영향 | 수정 전 가능성 |
|------|------|----------------|
| 임베딩 체크포인트가 causal 생성으로 잘못 라우팅되어 쓰레기 출력이나 로더 패닉 발생 | 높음 | Qwen3-Embedding과 EmbeddingGemma에서 높음 |
| 각 패밀리 포트가 풀링, 마스크, 배치를 제각기 다른 의미로 재구현 | 높음 | 공통 모듈 없이는 높음 |
| `(1 - m) * C` 형태의 f16 마스크가 모든 attend 위치에 NaN을 발생 | 높음 | 중간 |
| 지원하지 않는 풀링 모드에서 조용히 폴백하여 오류 없이 잘못된 벡터 반환 | 중간 | 중간 |
| 임베딩 추론이 채팅 워커의 MLX 스트림을 공유하여 생성을 지연 | 중간 | 중간 |

---

## 2. 기술적 선택과 그 이유

### 2.1 `model_type` 디스패치 전에 레이아웃으로 임베딩 체크포인트 인식

`is_embedding_checkpoint`는 `get_model_type`의 Kokoro 검사 직후, `model_type` match 이전에 실행됩니다. 다음 중 하나라도 해당하면 임베더입니다. `model_type`이 인코더 전용 패밀리(`bert`, `xlm-roberta`, `modernbert`, `siglip`)인 경우, `architectures[0]`이 임베딩 아키텍처(`BertModel`, `XLMRobertaModel`, `ModernBertModel`, `SiglipModel`, `SiglipTextModel`, `LlamaBidirectionalModel`, `LlamaNemotronVLModel`, `Lfm2BidirectionalModel`, `ColIdefics3`, `ColQwen2_5`, `ColQwen2ForRetrieval`, 또는 플래그로 판별하는 `use_bidirectional_attention: true`의 `Gemma3TextModel`과 `is_causal: false`의 `Ministral3Model`)인 경우, `modules.json`에 `type`이 `.Pooling`으로 끝나는 모듈이 있는 경우, `1_Pooling/config.json`이 존재하는 경우입니다.

두 가지 부정 규칙이 리랭커 이슈(#1356)와 생성기를 보호합니다. `architectures[0]`이 `ForSequenceClassification`으로 끝나면 절대 임베더가 아니며, 추가 모듈이 `1_LogitScore`뿐인 `modules.json`(Qwen3-VL-Reranker)은 해당하지 않습니다. 결정된 변형은 `model_type`으로 키가 정해지며, 임베딩 패밀리가 없는 `model_type`에 풀링 레이아웃이 있으면 causal 생성기로 흘러가지 않고 `model_type`을 명시한 오류로 반환됩니다. 풀링 레이아웃이 없는 `Qwen3ForCausalLM`은 여전히 `Qwen3`으로 감지됩니다.

검토한 대안: `architectures[0]`만으로 키를 잡으면 기본 아키텍처 이름을 유지하는 sentence-transformers 내보내기(`Qwen3-Embedding`은 `Qwen3ForCausalLM`에 `1_Pooling`을 더한 형태)를 놓치고, `1_Pooling`만으로는 단순 `BertModel` 내보내기를 놓칩니다. 규칙의 합집합이 5절의 감지 스윕에 포함된 모든 체크포인트를 처리합니다.

### 2.2 불리언 비교로 만드는 additive f32 `0 / -inf` 마스크

세 빌더(`create_bidirectional_padding_mask -> [B, 1, 1, L]`, `create_causal_padding_mask(mask, offset) -> [B, 1, L, L + offset]`, `create_bidirectional_window_mask(mask, window) -> [B, 1, L, L]`)는 하나의 꼬리 `additive_from_allowed`를 공유하며, `create_causal_mask`와 같은 방식으로 `where_cond`를 써서 불리언 "attend" 배열을 f32 `0.0 / -inf`로 바꿉니다. 출력이 f32인 것은 의도적입니다. `fast_scaled_dot_product_attention`이 부동소수 마스크를 쿼리 dtype으로 캐스팅하는데 `-inf`는 그 캐스팅을 견디므로, 같은 마스크를 f16과 bf16 활성화에 안전하게 쓸 수 있습니다(4비트 체크포인트는 f16으로 역양자화됩니다). 활성화 dtype에서 유한한 `C`로 `(1 - m) * C`를 만드는 방식은 명시적으로 배제했습니다. f16에서는 `-1e9`가 `-inf`로 오버플로되고 `0 * -inf`가 NaN이 되기 때문입니다.

causal과 window 빌더는 `create_causal_mask_with_left_padding`과 같은 구제 규칙을 적용합니다. 허용된 키 집합이 비어 있는 쿼리 행(왼쪽 패딩 배치의 선행 패딩 행)은 대각선 열을 유지하므로 모든 softmax 행이 유한합니다. 패딩 행의 출력은 유한하지만 의미 없는 값이며, 그 키 위치는 모든 실제 쿼리에 대해 차단되므로 소비되지 않습니다. 양방향 패딩 마스크는 패딩 쿼리 행도 모든 실제 키를 보므로 구제가 필요 없습니다. `utils_embedding_mask_tests.rs`는 각 규칙을 검사하고 마스크를 f16과 bf16으로 캐스팅해 softmax가 유한한지 확인합니다.

### 2.3 forward pass와 풀링은 패밀리가, 나머지는 엔진이 소유

`EmbeddingModel`은 이미 풀링된 `[B, D]` 벡터(다중 벡터 패밀리는 패딩 행을 0으로 채운 `[B, L, D]` 토큰 행렬)를 반환합니다. `src/embeddings/engine.rs`의 엔진은 패밀리별 텍스트 포매팅(`format_text`), 토큰화, 길이 정렬 마이크로배치, f32 캐스팅, L2 정규화, 재정규화를 동반한 `dimensions` 절단, 형태 검증, 읽기를 소유합니다. 트레이트는 의도적으로 `Send`도 `Sync`도 요구하지 않습니다. 구현체는 MLX 배열 핸들을 직접 보유하며 정확히 하나의 스레드(워커 스레드 또는 `mlxcel embed` 메인 스레드)에서만 사용됩니다.

공유 헬퍼가 패밀리 포트를 작게 만듭니다. `pool(hidden, mask, mode)`는 `cls`(마스크의 argmax, 왼쪽 패딩 인식), `mean`(`max(count, 1e-9)`로 나눈 마스크 가중 합), `max`(패딩 아래 `-inf` 채움), `lasttoken`(가장 큰 실제 인덱스, 전부 패딩인 행은 `L - 1`)을 구현합니다. `resolve_pooling_mode`는 `1_Pooling/config.json`, 패밀리 기본값, `MLXCEL_EMBEDDING_POOLING` 오버라이드를 순서대로 적용합니다. `load_embedding_weights`는 하위 폴더 로더와 텍스트 bf16-to-f16 규칙(Apple Silicon에서 비양자화 bf16은 변환, 양자화 scale과 bias는 bf16 유지)을 적용합니다. `quantization_params`는 `UnifiedLinear::from_weights`를 위해 `config.quantization`을 읽습니다.

### 2.4 조용한 폴백 대신 명시적 오류

`PoolingConfig::parse`는 새 방식의 `"pooling_mode"` 문자열과 레거시 `pooling_mode_*` 불리언을 모두 받지만, `weightedmean`, `mean_sqrt_len_tokens`, `include_prompt: false`, 둘 이상의 레거시 플래그 설정은 모드 이름을 명시한 로드 오류입니다. `build_family_model`은 감지된 모든 변형에 대해 `<family> is detected as an embedding checkpoint, but this embedding family is not yet supported by /v1/embeddings`를 반환하고, 생성 로더는 `ModelKind::Embedding`에 대해 `/v1/embeddings`와 `mlxcel embed`를 가리키는 메시지로 중단합니다. 생성기 체크포인트에 `load_embedding_model`을 호출하면 `not an embedding checkpoint`를 보고합니다. 지원하지 않는 풀링 모드를 쓰는 체크포인트는 아예 로드할 수 없다는 것이 트레이드오프이지만, 잘못된 풀링으로 계산한 벡터를 제공하는 것보다 낫습니다.

### 2.5 오디오 워커의 admission 설계를 따르는 전용 워커 스레드

`EmbeddingWorker`는 스레드 로컬 기본 스트림을 설치하고 `LoadedEmbeddingModel`을 자기 자신에 로드한 뒤 bounded `mpsc::SyncSender`의 명령을 처리하는 스레드 하나를 소유합니다. 큐가 가득 찼을 때의 `try_send`는 `EmbeddingError::QueueFull`(HTTP 503)을, `recv_timeout`은 `EmbeddingError::Timeout`(HTTP 504, 진행 중인 MLX 작업은 취소되지 않음)을 반환하며, 모든 엔진 호출은 `catch_unwind` 아래에서 실행된 뒤 스트림을 동기화하므로 패닉을 일으킨 요청 하나가 워커를 죽이지 않습니다. 모델, 토크나이저, 모든 배열은 워커 스레드에 있고 HTTP 쪽은 `Vec<f32>`만 봅니다. 큐 깊이와 타임아웃은 오디오 값(8과 120초)을 기본값으로 씁니다. 채팅 스케줄러에 임베딩 작업을 얹는 대신 `AudioWorker`를 그대로 따랐으므로 채팅과 임베딩은 스트림을 공유하지 않습니다.

### 2.6 체크포인트 한 개 또는 두 개에 대한 시작 의미론

`resolve_embedding_source`는 `--embedding-model`과 `-m` 감지 결과를 결합합니다. `-m <임베딩 체크포인트>`만 주면 라우트를 서비스하고 채팅 워커의 `load_model`은 중단하며 로그를 남깁니다(Whisper 패턴). `-m <채팅> --embedding-model <임베딩>`은 별도 워커에서 둘 다 서비스합니다. 둘 다 임베딩(`-m`에 임베딩 체크포인트와 `--embedding-model` 동시 지정)이면 리스너가 바인딩되기 전에 시작 오류입니다. 명시적 `--embedding-model`의 로드 실패는 치명적이고, `-m` 임베딩 체크포인트의 로드 실패는 로그를 남기고 슬롯을 비워 두어 라우트가 구조화된 501을 응답합니다. `/v1/models`는 두 id가 다를 때만 채팅 모델 옆에 임베딩 모델을 나열합니다. `MLXCEL_EMBEDDING_MODEL`은 clap의 `LLAMA_ARG_EMBEDDING_MODEL` 위에 기존 우선순위 규칙(플래그나 `LLAMA_ARG_*`가 우선, 충돌하는 별칭은 로그 후 무시)으로 덧씌워집니다.

### 2.7 패밀리가 없는 상태에서 라우트를 먼저 통합

`src/embeddings/stub.rs`는 mean 풀링한 one-hot 토큰 임베딩을 반환하는 `#[cfg(test)]` 모델로, 프로덕션과 같은 `finish_loaded_model` 꼬리를 거쳐 만들어집니다. 라우트 테스트, 엔진 테스트, 워커 테스트가 이 스텁으로 실행되므로 문자열, 리스트, 토큰 id, base64, `dimensions`, 빈 입력, 501, 모델 불일치, usage 집계, 이미지 거부, 다중 벡터 출력이 첫 패밀리 포트 이전에 모두 검증됐습니다. 패밀리 하위 이슈를 `mlxcel embed`만으로 검증할 수 있는 이유입니다.

### 2.8 길이 제한과 토크나이저 정리

`max_length`는 `sentence_bert_config.json`의 `max_seq_length`, `0 < value < 1_000_000`일 때의 `tokenizer_config.json` `model_max_length`(HuggingFace의 미설정 센티널은 무시), 절대 위치 패밀리(BERT, XLM-RoBERTa, SigLIP)에 한한 `config.json`의 `max_position_embeddings`, 하드 캡 8192, `--embedding-max-length` 중 최솟값입니다. `strip_padding_and_truncation`은 일부 `tokenizer.json`이 고정해 둔 패딩과 절단(MiniLM은 128으로 패딩)을 제거합니다. 엔진이 마이크로배치 단위로 패딩하고 체크포인트 제한으로 절단하기 때문입니다. 절단은 토크나이저가 붙인 후행 특수 토큰을 유지하므로 BERT 입력은 `[SEP]`을, Qwen3-Embedding 입력은 `<|endoftext|>`를 보존합니다. 토큰 id를 그대로 넘긴 입력은 특수 토큰 처리 없이 오른쪽에서 자릅니다.

---

## 3. 구현 상세

### 3.1 요청 흐름

```
POST /v1/embeddings
  -> routes/embeddings.rs: 본문 파싱, provider 없으면 501, 모델 id 검사,
     encoding_format, dimensions, 항목별 검증, 이미지 fetch + decode
  -> spawn_blocking(embed_items)
       -> EmbeddingModelProvider (EmbeddingWorkerProvider)
            -> bounded SyncSender -> 임베딩 워커 스레드
                 -> EmbeddingEngine: format_text, encode_row, 길이 정렬,
                    --embedding-batch-size 크기의 마이크로배치, EncodedBatch
                    -> EmbeddingModel::embed (패밀리 forward pass)
                    -> astype f32, normalize_l2, truncate_dimensions,
                       normalize_l2 재적용, array_to_vec_f32
       <- EmbedReply { vectors, prompt_tokens } 요청 순서로 되돌려 기록
  <- EmbeddingsResponse { data, model, usage }
```

### 3.2 모듈별 변경

| 영역 | 변경 |
|------|------|
| `src/lib/mlxcel-core/src/utils.rs` | `create_bidirectional_padding_mask`, `create_causal_padding_mask`, `create_bidirectional_window_mask`, 공유 꼬리 `additive_from_allowed`, 읽기 헬퍼 `array_to_vec_f32`. |
| `src/lib/mlxcel-core/src/weights.rs` | `load_weights_from_dir_with_subfolders`: 최상위 shard는 기존과 동일하게, 이어서 숨김이 아닌 직계 하위 디렉터리를 정렬 순서로 순회하며 텐서 이름에 `<folder>.`를 접두. `consolidated.safetensors`와 `adapter_model.safetensors`는 두 수준 모두에서 건너뜀. |
| `src/embeddings/model.rs` | `EmbeddingModel` 트레이트, `EmbeddingBatch`(`input_ids`, `attention_mask`, 선택적 `token_type_ids`, 선택적 이미지), `EmbeddingOutput`, `ImageInput`. |
| `src/embeddings/pooling.rs` | `PoolingMode`, `PoolingConfig::read`와 `parse`, `resolve_pooling_mode`, `pool`, `normalize_l2`, `truncate_dimensions`. |
| `src/embeddings/limits.rs` | `EmbeddingLimits::derive`, `derive_max_length`, `resolve_pad_token_id`(`pad_token`, 다음 `eos_token`, 다음 0), `resolve_vocab_size`(`vocab_size`, 다음 `text_config.vocab_size`, 다음 토크나이저), `config_normalize_flag`. |
| `src/embeddings/tokenize.rs` | `EncodedRow`, `EncodedBatch::from_rows`(오른쪽 패딩, 선택적 고정 폭), HuggingFace 특수 토큰 마스크 또는 다른 토크나이저용 추론 마스크를 쓰는 `encode_row`, 리랭커 이슈용 `encode_pair_row`와 `encode_pairs`, `strip_padding_and_truncation`. |
| `src/embeddings/loader.rs` | `load_embedding_model[_with_options]`, 패밀리 디스패처 `build_family_model`, `load_embedding_weights`, `quantization_params`, `finish_loaded_model`. |
| `src/embeddings/engine.rs` | `embed_texts`, `embed_tokens`, `embed_image`, `run_rows`, `forward_batch`, `postprocess`를 갖춘 `EmbeddingEngine`; `EmbedOptions`, `EmbedReply`, `EmbeddingVector`, `EmbeddingEngineError`. |
| `src/model_metadata.rs`, `src/models/mod.rs` | `ModelKind::Embedding`, `is_embedding_model_type`, `mlxcel arch`의 `Embedding` 패밀리 아래 열세 개 `ModelType` 변형, `weight: None`과 어댑터 거부 메시지를 갖춘 등록. |
| `src/models/detection.rs` | `is_embedding_checkpoint`와 2.1절의 규칙, `model_type` match 이전 호출. |
| `src/loading/mod.rs` | `load_model`이 `ModelKind::Embedding`에 대해 라우트를 가리키며 중단. |
| `src/distributed/tensor_parallel/inference.rs` | fallback 아키텍처 테이블의 완전성을 유지하는 자리표시 arm. |
| `src/server/embedding_model.rs` | `EmbeddingModelProvider` 트레이트와 `EmbeddingError { QueueFull, Timeout, InvalidInput, Internal }`. |
| `src/server/embedding_worker.rs` | `EmbeddingWorker`, `worker_loop`, `run_guarded`, `EmbeddingWorkerProvider`. |
| `src/server/routes/embeddings.rs`, `src/server/types/embeddings.rs` | 핸들러, 검증 순서, 오류 매핑, untagged `EmbeddingInput` enum, `EmbeddingEncoding`, base64 헬퍼, `EmbeddingData::from_vector`. |
| `src/server/app.rs`, `routes/models.rs`, `state.rs`, `startup.rs`, `config.rs`, `cli_input.rs` | `/v1/embeddings`와 `/embeddings` 마운트, `/v1/models` 항목, `embedding_model` 슬롯, `resolve_embedding_source`와 `resolve_embedding_provider`, 설정 필드 다섯 개와 기본값, `env_fallback_embedding_model`. |
| `src/bin/mlx_server.rs`, `src/main.rs`, `src/commands/serve.rs` | 두 바이너리 모두의 `--embedding-model`, `--embedding-batch-size`, `--embedding-max-length`, `--embedding-queue-depth`, `--embedding-request-timeout-secs`; `--embedding-model`은 `-m`과 같은 스토어 조회와 자동 다운로드로 해석. |
| `src/commands/embed.rs` | `mlxcel embed -m <path> -p ... [--image ...] [--instruction] [--dimensions] [--max-length] [--batch-size] [--json]`, 입력당 한 줄의 벡터와 코사인 행렬(다중 벡터 출력은 쿼리 행 평균 MaxSim). |
| `docs/embeddings.md`, `docs/supported-models.md`, `docs/environment-variables.md`, `docs/README.md`, `README.md`, `CONTRIBUTING.md` | 엔드포인트, 감지 규칙, 풀링, 제한, 마스크, 플래그, CLI, "패밀리 추가" 체크리스트; 패밀리 포트가 채울 빈 Embedding models 표. |

### 3.3 HTTP 계약

`input`은 문자열, 문자열 리스트, 토큰 id 배열, 토큰 id 배열의 리스트, 타입이 지정된 part 리스트(`{type: text}`와 `{type: image_url}`)를 받습니다. 토큰 id 입력은 그대로 사용되며(특수 토큰 추가 없음) 모든 id는 `vocab_size` 미만이어야 합니다. `encoding_format: base64`는 little-endian f32 바이트를 표준 패딩으로 인코딩합니다. 다중 벡터 모델의 float 형식은 행 리스트이고, base64 형식은 `shape: [num_real_tokens, D]`를 형제 필드로 동반합니다. `usage.prompt_tokens`는 특수 토큰을 포함한 실제 토큰 수이며 `total_tokens`는 같은 값입니다. `image_url` 항목은 공유 미디어 제한(`try_read_image_url_with_limits`, `decode_request_images_with_limits`)을 거치며 한 번에 하나씩 임베딩됩니다.

| 상태 | 유형 | 조건 |
|------|------|------|
| 400 | `invalid_request_error` | 잘못된 본문, 빈 `input`, 빈 문자열 또는 토큰 리스트, `vocab_size` 이상의 토큰 id, 텍스트 전용 모델에 이미지, `1..=D` 밖의 `dimensions`, 지원하지 않는 `encoding_format`, 서비스 중인 id와 다른 `model` |
| 501 | `not_implemented` | 임베딩 모델이 로드되지 않음 |
| 503 | `server_busy` | 워커 큐가 가득 참 |
| 504 | `server_timeout` | 워커 응답 타임아웃 |
| 500 | `server_error` | forward pass 실패 또는 복구된 워커 패닉 |

---

## 4. 보안, 성능, 품질 검토

### 4.1 입력 처리

모든 요청 필드는 워커에 닿기 전에 HTTP 쪽에서 고정된 순서로 검증되고, 엔진 안에서 다시 검증됩니다(빈 문자열, 어휘 밖 id, `dimensions`, 텍스트 전용 패밀리에 대한 이미지). 따라서 CLI와 서버가 같은 거부 규칙을 공유합니다. 이미지 페이로드는 채팅 이미지 경로의 기존 크기와 해상도 제한을 재사용합니다. 임베딩 요청에서 MLX 배열이 생성되는 곳은 워커뿐이며, bounded 큐와 타임아웃이 오디오 라우트와 같은 부하 차단 동작을 제공합니다. `-m`이 쓰는 기존 모델 다운로드 외에 자격 증명, 파일 쓰기, 새 네트워크 경로는 도입되지 않았습니다.

### 4.2 성능 특성

텍스트 항목은 토큰 길이로 정렬한 뒤 마이크로배치로 잘라 패딩 낭비가 요청 전체가 아닌 한 배치 안의 길이 편차로 제한되며, 되돌려 기록하는 단계가 요청 순서를 복원합니다. forward pass 출력은 한 번만 f32로 캐스팅되고 디바이스에서 정규화와 절단을 거친 뒤 마이크로배치당 한 번 읽힙니다. 이미지는 배치하지 않습니다. `mean`과 `lasttoken` 풀링 경로는 `argmax`, `where_cond`, `take_along_axis`를 써서 데이터 의존적인 호스트 루프를 피합니다. 감지는 이제 모든 체크포인트에 대해 `modules.json`을 읽고 `1_Pooling/config.json`을 확인하는데, 둘 다 로드 시점의 작은 파일 연산입니다.

### 4.3 호환성

- 호환성 파괴: 기존 생성과 오디오 경로에는 없음. 이전에 생성기로 잘못 라우팅되어 실패하던 임베딩 체크포인트는 이제 임베더로 감지되고, 포트가 도착할 때까지 `not yet supported` 메시지로 실패합니다.
- 새 의존성: 없음. `base64`, `image`, `thiserror`, `safetensors`는 이미 워크스페이스에 있었습니다.
- 새 `ModelType` 변형은 `ALL_MODEL_TYPES`, `mlxcel arch` 패밀리 순서, tensor-parallel fallback 테이블에 추가되어 모든 완전 매칭이 유지됩니다.

### 4.4 테스트 커버리지

이 PR은 테스트 함수 84개(동기 70개, 비동기 14개)를 추가합니다. 영역별로는 f16과 bf16 유한성 검사를 포함한 `mlxcel-core` 마스크 테스트 5개, 왼쪽과 오른쪽 패딩 아래 모든 모드와 두 설정 키 방식, 지원하지 않는 각 모드, 해석 순서, env 오버라이드를 다루는 풀링 테스트, 패딩, 후행 특수 토큰 절단, `token_type_ids`, 제거된 내장 패딩을 다루는 토크나이저 테스트, 하위 폴더 순회, 양자화 파라미터, `max_length` 유도, pad와 vocab 해석, 세 가지 거부 경로를 다루는 로더 테스트, 마이크로배치 간 순서 보존, `dimensions`, 다중 벡터 행, 이미지 거부를 다루는 엔진 테스트, 준비 완료, 로더 실패, 패닉 복구, 타임아웃, 큐 포화를 다루는 워커 테스트, 모든 입력 형태와 오류 코드를 다루는 라우트 테스트, 각 규칙과 각 부정 규칙을 다루는 감지 테스트, 설정 배선을 다루는 시작과 CLI 테스트, 체크포인트가 없으면 건너뛰는 실제 체크포인트 게이트 8개입니다.

---

## 5. 검증 증거

PR의 테스트 계획과 실제 체크포인트 절에서 가져온 결과입니다(CUDA 빌드, `--profile test-fast`로 실행).

| 검증 | 결과 |
|------|------|
| `server::embedding_worker`, `server::routes::embeddings`, `models::detection_tests`, `model_metadata_tests`, `models::tests`, `embeddings::`, `server::cli_input_tests`, `server::startup_tests`에 대한 `cargo test --lib` | 115개 통과 |
| `cargo test --lib -- embeddings::real_checkpoint_tests` | 두 체크포인트가 있는 상태에서 8개 통과 |
| `cargo test -p mlxcel-core --lib embedding_mask_tests` | 5개 통과 |
| `cargo test --bin mlxcel -- embed serve_tests` | 3개 통과 |
| `cargo check --all-targets`, `cargo clippy --lib --bins --tests -- -D warnings`, `cargo fmt --all -- --check` | 통과 |
| Metal CI 러너의 워크스페이스 테스트 | 머지 시점에 미실행 |

PR에 기록된 실제 체크포인트 관찰:

- `sentence-transformers/all-MiniLM-L6-v2`는 `model_type: bert`와 `1_Pooling/config.json`으로 `Bert`로 감지되고, 풀링은 `mean`으로 파싱되며, `max_length`는 `sentence_bert_config.json`에서 256으로 유도됩니다. 하위 폴더 로더는 `embeddings.word_embeddings.weight` `[30522, 384]`를 읽고 합성 `2_Dense/model.safetensors`를 `2_Dense.linear.weight`로 접두합니다. 배치 토크나이저는 `[PAD]`(id 0)로 오른쪽 패딩하고, `tokenizer.json`에 고정된 128으로 더 이상 패딩하지 않으며, 절단 시 `[SEP]`을 유지합니다. `load_model`은 `/v1/embeddings` 메시지로 중단합니다.
- `Qwen/Qwen3-Embedding-0.6B`는 `modules.json`과 `pooling_mode_lasttoken`의 `1_Pooling`으로 `Qwen3Embedding`으로 감지되고, `max_length`는 8192 캡으로 유도되며, pad 토큰은 `<|endoftext|>`(151643)로 결정되고 절단 시 유지됩니다.
- 로컬에 있는 임베딩 체크포인트 열두 개(`multilingual-e5-small`, `bge-m3`, `modernbert-embed-base`, `siglip-base-patch16-224`, `embeddinggemma-300m-4bit`, `Qwen3-VL-Embedding-2B`, `LFM2.5-Embedding-350M`, Nemotron 임베더들, ColSmolVLM과 ColQwen2.5 리트리버 포함)에 대한 감지 스윕이 각각 자기 패밀리로 결정되고, 리랭커 다섯 개는 임베딩 변형으로 라우팅되지 않습니다.
- `mlxcel-server -m all-MiniLM-L6-v2` 종단 간: 채팅은 로드되지 않고, `/v1/models`가 체크포인트를 나열하며, `/v1/embeddings`와 `/embeddings`는 구조화된 501을, 잘못된 본문은 400을 응답합니다. 두 임베딩 모델 오류와 미포팅 패밀리 시작 오류는 설계대로 발생합니다.

squash 머지 전 PR에 기록된 리뷰 스레드는 없습니다.

---

## 6. 학습 포인트

- 임베딩 내보내기는 생성기의 `model_type`을 그대로 쓰는 경우가 많아 `model_type`만 읽는 감지기는 Qwen3-Embedding과 Qwen3을 구분할 수 없습니다. 레이아웃 신호(`1_Pooling`, `modules.json`, `architectures[0]`, 양방향 플래그)를 먼저 확인하고 리랭커 제외 규칙을 명시해야 합니다.
- additive attention 마스크는 불리언에서 `where_cond`로 f32 `0 / -inf`를 만들어야 합니다. 산술 형태 `(1 - m) * C`는 큰 상수가 오버플로되고 `0 * -inf`가 NaN이 되어 f16에서 깨집니다.
- 쿼리 행을 완전히 차단할 수 있는 마스크에는 구제(대각선 유지)가 필요하며, 그러면 패딩 행은 소비되지 않는 유한한 값을 냅니다.
- sentence-transformers 토크나이저는 `tokenizer.json`에 고정 패딩과 절단을 넣을 수 있으므로 배치 엔진은 이를 제거해야 합니다. 그러지 않으면 모든 입력이 고정 폭으로 패딩됩니다.
- 임베딩 입력을 절단할 때는 모델이 풀링하도록 학습된 후행 특수 토큰(`[SEP]`, `<|endoftext|>`)을 보존해야 합니다. 단순한 오른쪽 절단은 이를 버립니다.
- 프로덕션 트레이트를 구현한 테스트 전용 모델은 첫 실제 패밀리 이전에 전체 요청 경로를 통합하고 검증하게 하며, 이후 패밀리 포트의 diff를 forward pass로 한정합니다.

---

## 7. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 52 |
| 추가 라인 | 6,689 |
| 삭제 라인 | 23 |
| 새 모듈 | `src/embeddings/`(모듈 6개, 스텁, 테스트 형제 5개), `src/server/embedding_model.rs`, `src/server/embedding_worker.rs`, `src/server/routes/embeddings.rs`, `src/server/types/embeddings.rs`, `src/commands/embed.rs`, `docs/embeddings.md` |
| 추가된 테스트 함수 | 84 |
| 새 의존성 | 0 |

| 범주 | 요약 |
|------|------|
| Core(`mlxcel-core`) | 패딩 인식 마스크 빌더 세 개, `array_to_vec_f32`, 하위 폴더 safetensors 로딩 |
| 모델 레지스트리 | `ModelKind::Embedding`, `ModelType` 변형 13개, 레이아웃 기반 감지, 생성 로더 중단 |
| 임베딩 서브시스템 | 트레이트, 풀링, 제한, 토큰화, 로더 디스패처, 마이크로배치 엔진 |
| 서버 | provider 트레이트, 워커 스레드, `/v1/embeddings`와 `/embeddings`, `/v1/models` 항목, 플래그 다섯 개와 env 별칭, 시작 소스 결정 |
| CLI | `mlxcel embed` |
| 문서 | `docs/embeddings.md`, supported-models와 environment-variables 절, README와 CONTRIBUTING 링크 |

| 커밋 | 목적 |
|------|------|
| `3634b9a60`(PR head), `b6f77857`로 squash 머지 | feat(embeddings): pooling, masks, Embedding kind and /v1/embeddings |

---

## 8. 후속 조치

### 필수

- Epic #1348 아래에서 패밀리 forward pass를 구현하여 `build_family_model`의 각 `not yet supported` arm을 교체하고 `docs/supported-models.md`의 Embedding models 표를 채웁니다.
- Metal CI 러너에서 워크스페이스 테스트 게이트를 실행합니다. PR의 검사는 CUDA 빌드에서 수행됐습니다.
- 여기서 출하됐지만 아직 소비자가 없는 `encode_pairs` 위에 리랭커 경로(#1356)를 구현합니다.

### 미검증 항목

- 마스크 빌더는 형태, 차단 규칙, f16 유한성을 단독으로 검증했을 뿐 실제 attention 레이어 안에서는 아직 검증하지 않았습니다.
- `pool`, `normalize_l2`, 엔진은 스텁과 합성 배열로 검증됐습니다. 패밀리별 Python 레퍼런스와의 수치 일치는 패밀리 포트의 책임입니다.
- 양자화 임베딩 체크포인트는 `quantization_params`만 거칩니다. `UnifiedLinear::from_weights` 통합은 패밀리로 이관됐습니다.

### 모니터링

- 패밀리가 서비스되기 시작하면 `/v1/embeddings`의 503과 504 비율을 관찰합니다. 기본값(`--embedding-queue-depth 8`, `--embedding-request-timeout-secs 120`)은 오디오 워커에서 물려받은 것으로 임베딩 워크로드에 맞춰 조정되지 않았습니다.
- 응답 타임아웃은 HTTP 스레드를 해제하지만 진행 중인 MLX 작업을 취소하지 않습니다. 오디오 워커와 같은 제약입니다.

### 향후 개선

- 이미지 입력을 한 번에 하나씩 임베딩하는 대신 배치 처리합니다.
- 대상 체크포인트가 요구한다면 `weightedmean` 풀링을 검토합니다. 현재는 로드 시 거부됩니다.
- `MLXCEL_EMBEDDING_POOLING` 오버라이드는 서버와 `mlxcel embed`에 적용되는 디버깅 보조 수단이며 프로덕션 설정에서 의존해서는 안 됩니다.

## 참고

- Issue #1353: 풀링, 양방향 마스크, Embedding 모델 종류와 /v1/embeddings.
- Epic #1348: 임베딩 모델 패밀리 포트.
- Issue #1356: 리랭커 지원.
- PR #1408: 구현.
- `docs/embeddings.md`: 엔드포인트, 감지, 풀링, 제한, 플래그, 패밀리 체크리스트.
