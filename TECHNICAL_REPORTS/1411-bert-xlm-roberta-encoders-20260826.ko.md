# 기술 보고서: PR #1411 - BERT / XLM-RoBERTa 인코더

**작성일**: 2026-08-26
**작성자**: mlxcel contributors
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1411은 이슈 #1321을 구현한다. #1353이 만든 임베딩 하부구조 위에 올라가는 첫 번째 family forward pass다. BERT(`model_type: bert`)와 XLM-RoBERTa(`model_type: xlm-roberta`)는 절대 위치 임베딩 위의 동일한 post-LayerNorm 인코더이므로, 두 개의 포트가 아니라 `BertVariant` 스위치 하나로 갈라지는 단일 모듈로 구현했다.

`/v1/rerank`(#1356)가 필요로 하는 `BertForSequenceClassification` / `XLMRobertaForSequenceClassification` 헤드도 함께 제공한다. 이 헤드는 `pooler.` 텐서를 남긴 동일 trunk를 재사용하며 디렉터리 경로로 직접 로드한다. `ForSequenceClassification` export는 reranker이고, detection이 이를 embedder로 분류하지 않도록 의도적으로 막아 두었기 때문이다.

실제 체크포인트 네 개가 `mlxcel embed`와 `POST /v1/embeddings` 양쪽에서 로드되고 실행되며, CLI와 HTTP 벡터는 비트 단위로 동일하다. `all-MiniLM-L6-v2`는 sentence-transformers quickstart가 공개한 코사인 값을 소수 넷째 자리까지 재현하고, `BAAI/bge-m3`는 모델 카드의 dense 행렬을 6e-4 이내로 재현하며, `BAAI/bge-reranker-v2-m3`는 모델 카드의 부호 분리를 재현한다.

---

## 1. 문제 정의

`src/embeddings/`에는 trait, pooling 모드, mask builder, tokenizer 연결, worker thread, HTTP route, CLI가 모두 있었지만 family가 하나도 없었다. 인식된 임베딩 체크포인트는 전부 `<family> is detected as an embedding checkpoint, but this embedding family is not yet supported by /v1/embeddings`로 응답했다. 엔드포인트는 존재했지만 쓸모 있는 값을 반환하지 못했다.

BERT와 XLM-RoBERTa는 첫 포트로서 가치가 가장 크다. `all-MiniLM-L6-v2`, `multilingual-e5` 계열, `bge-m3`를 한꺼번에 덮고, 이 셋이 실제로 배포되는 문장 임베딩 체크포인트의 대부분을 차지하며, reranker 엔드포인트가 필요로 하는 classification head까지 함께 가지고 있다.

일반적인 포팅 작업과 달리 이 family에만 있는 위험이 둘 있었다.

| 위험 | 영향 | 발생 지점 |
|------|------|-----------|
| XLM-RoBERTa 위치 id가 `pad_token_id + 1`만큼 밀려 있음 | 긴 입력에서 범위를 벗어난 gather | 파생 `max_length`가 `max_position_embeddings`에서 오는 모든 XLM-RoBERTa 체크포인트 |
| XLM-RoBERTa 위치 id는 attention mask가 아니라 token id를 기준으로 함 | 패딩된 배치에서 조용히 잘못된 임베딩 | 배치 요청 전부 |

---

## 2. 기술적 선택

### 2.1 두 개의 포트가 아니라 variant 스위치 하나

두 family가 다른 곳은 정확히 다섯 군데다. 위치 id 구성, `type_vocab_size` 기본값(2 대 1), `layer_norm_eps` 기본값(1e-12 대 1e-5), `pad_token_id` 기본값(0 대 1), 가중치 키 접두사(`bert.` 대 `roberta.`). 임베딩 합, LayerNorm 위치, attention 형태, GELU feed-forward, residual 순서는 완전히 같다.

이를 분리했다면 다섯 개의 상수를 표현하려고 블록 코드 300여 줄을 복제해야 했다. `BertVariant`는 trunk를 하나로 유지하면서 차이를 열거 가능하게 만든다. classification head가 접두사 쌍만 바꿔 하나의 `ClassifierHead`로 두 방언을 모두 처리할 수 있었던 것도 같은 이유다.

### 2.2 방언은 오직 `model_type`으로 결정

`intfloat/multilingual-e5-small`은 `model_type: bert`와 `architectures: [BertModel]`을 선언하면서 `tokenizer_class: XLMRobertaTokenizer`와 sentencepiece 어휘를 함께 배포한다. 가중치는 BERT 레이아웃이고 upstream도 위치 id를 BERT 방식으로 만들기 때문에, tokenizer나 어휘로 방언을 판단했다면 널리 쓰이는 체크포인트에서 조용히 잘못된 위치가 나왔을 것이다. 스위치는 `model_type`만 읽는다.

### 2.3 토큰 상한은 family가 직접 보고

XLM-RoBERTa는 위치 테이블을 `pad_token_id + 1`부터 인덱싱하므로, `max_position_embeddings` 행짜리 테이블이 실제로 담는 토큰은 `max_position_embeddings - pad_token_id - 1`개다. `bge-m3`는 8194행을 공개하고 8192 토큰을 담는다.

`EmbeddingLimits::derive`는 이를 알 방법이 없다. 불리언 `is_absolute_position`을 받아 `max_position_embeddings`를 그대로 읽을 뿐이다. `bge-m3`는 자체 `sentence_bert_config.json`과 공유 상한 8192이 우연히 문제를 가리지만, `sentence_bert_config.json`이 없는 514행짜리 표준 `xlm-roberta-base` 레이아웃이라면 514를 파생시켜 테이블을 두 행 넘겨 gather하게 된다.

해법은 기본 구현이 있는 새 trait 메서드 `EmbeddingModel::max_sequence_length() -> Option<usize>`이고, 기존 `pad_to_max_length` clamp 옆에서 `max_length`에 접힌다. 규칙을 `limits.rs`가 아니라 family에 두면 공유 파생 로직이 family별 산술에서 자유로워지고, 이후 모든 family가 같은 훅을 쓸 수 있다.

인코더는 그 상한보다 넓은 배치를 이름 있는 오류로 거부하기도 한다. 상한 계산보다 앞단에 버그가 생기면 MLX gather 오류가 아니라 메시지로 드러난다.

### 2.4 token-type 임베딩은 항상 적용

`XLMRobertaModel`의 `type_vocab_size`는 1이라 segment 테이블이 없는 것처럼 다루기 쉽다. 실제로는 없지 않다. `bge-m3`는 upstream이 모든 토큰에 더하는 진짜 학습된 `[1, 1024]` 벡터를 담고 있다. 이를 빼면 모든 임베딩이 어긋난다.

그래서 인코더는 항상 테이블을 읽는다. 배치에 `token_type_ids`가 없으면 `[1, 1]` 0 배열로 인덱싱하고, 그 `[1, 1, D]` 결과가 `[B, L, D]`에 브로드캐스트된다. `[B, L]` 0 행렬을 만드는 대신 한 행만 조회하면 된다. `needs_token_type_ids()`는 BERT 방언에서만 `true`로 남아, XLM-RoBERTa가 절대 바꾸지 않는 segment 축의 비용을 엔진이 치르지 않게 한다.

### 2.5 classification head는 detection이 아니라 경로로 접근

`is_embedding_checkpoint`는 `architectures[0]`이 `ForSequenceClassification`으로 끝나면 `Ok(None)`을 반환하므로, `get_model_type`은 `bge-reranker-v2-m3`에 대해 `Unsupported model type`을 보고한다. 이는 옳다. cross-encoder reranker는 embedder가 아니고, `/v1/embeddings`로 보내면 관련도 점수 대신 pooled hidden state가 나온다.

따라서 `BertSequenceClassifier::load(dir)`가 `config.json`을 직접 읽고 `BertVariant::from_config`로 방언을 고른다. 이 PR은 여기에 HTTP 표면을 붙이지 않는다. 그것은 #1356의 몫이다.

### 2.6 sanitize는 멱등이며 접두사에 무관

task-head 체크포인트는 인코더를 `bert.` 또는 `roberta.` 밑에 넣고, 순수 `BertModel` export는 그러지 않는다. `sanitize`는 존재하는 접두사를 벗겨 두 레이아웃이 같은 키 집합에 도달하게 한 뒤, `position_ids` 버퍼(구버전 transformers export가 텐서로 등록한다), `cls.`와 `lm_head.` masked-LM 헤드를 버리고, classifier head를 만드는 경우가 아니면 `pooler.`도 버린다. 두 번 실행해도 결과가 같아 `src/models/sanitize.rs`의 관례를 따른다.

---

## 3. 구현 상세

| 영역 | 변경 |
|------|------|
| `src/models/bert_config.rs` | `BertVariant`, variant별 기본값을 채우는 `BertArgs`, `num_labels` 또는 `id2label`에서 오는 라벨 수, `max_sequence_length()`. `bert`를 통해 재수출해 호출자는 공개 경로 하나만 쓴다. |
| `src/models/bert.rs` | `sanitize`, `xlm_roberta_position_ids`, `BertEmbeddings`, `BertLayer`, `BertEncoder`, 그리고 `gelu_new`를 자체 구현한 `Activation`. |
| `src/models/bert_heads.rs` | `EmbeddingModel`을 구현하는 `BertEmbeddingModel`, `BertSequenceClassifier`, 둘이 공유하는 `load_encoder`. |
| `src/embeddings/loader.rs` | `Bert` / `XlmRoberta` dispatcher arm. `finish_loaded_model`이 `max_sequence_length()`를 파생 상한에 접는다. |
| `src/embeddings/model.rs` | 기본값 `None`인 `EmbeddingModel::max_sequence_length()`. |
| `docs/embeddings.md`, `docs/supported-models.md` | family 표 행, prompt 접두사 표, 위치 기반 길이 규칙, classification head 위치. |

forward pass는 다음과 같다.

```
positions = 0..L                                  # BERT
positions = cumsum(ids != pad) * (ids != pad) + pad   # XLM-RoBERTa
h = LayerNorm(word[ids] + position[positions] + type[segments])
mask = create_bidirectional_padding_mask(attention_mask)   # [B, 1, 1, L]
for layer:
    a = LayerNorm(attn_out(sdpa(q, k, v, head_dim^-0.5, mask)) + h)
    h = LayerNorm(output(gelu(intermediate(a))) + a)
```

`sdpa`는 명시적 additive mask를 받는 `mlxcel_core::layers::attention`이다. 모든 projection에 bias가 있어서 bias 없는 경로가 아니라 `UnifiedLinear`를 일관되게 사용한다.

### 3.1 테스트 직렬화

`cargo test`는 테스트 함수를 병렬 스레드에서 실행하는데, 한 프로세스 안에서 동시에 도는 MLX forward pass는 서로를 교란한다. 형제 유닛 #1332는 다른 real-checkpoint 테스트가 함께 돌 때에만 한 배치 안의 바이트 단위로 동일한 두 행이 코사인 1.0이 아니라 0.999912로 돌아오고 classifier logit이 0.05만큼 움직이는 것을 측정했다. `EmbeddingModel`은 단일 스레드 사용으로 문서화되어 있고 서버는 embedding worker로 이를 지키므로 위험은 테스트 쪽에만 있지만, 그 상태에서 측정한 gate 값은 의미가 없다.

MLX를 건드리는 모든 BERT 테스트는 모듈 수준 `OnceLock<Mutex<()>>` 가드를 잡고, poisoned lock은 전파하지 않고 복구한다. 가드를 적용한 모듈을 연속 세 번 실행한 결과 출력된 gate 값이 모든 자릿수에서 동일했다.

---

## 4. 실제 체크포인트 결과

네 체크포인트 모두 f32이며 f32로 실행된다. 공유 bf16-to-f16 규칙은 이들에게 적용되지 않는데, `all-MiniLM-L6-v2`가 `layer_norm_eps: 1e-12`를 쓰고 이 값은 f16에서 underflow하므로 중요한 지점이다.

| 체크포인트 | 기준값 | 측정값 |
|------------|--------|--------|
| `sentence-transformers/all-MiniLM-L6-v2` | quickstart 코사인 0.666, 0.105 | 0.6660, 0.1046 |
| `BAAI/bge-m3` (safetensors 미러) | 모델 카드 `[[0.6265, 0.3477], [0.3499, 0.678]]` | `[[0.6259, 0.3475], [0.3499, 0.6782]]` |
| `BAAI/bge-reranker-v2-m3` | 모델 카드 부호 분리, 약 -8과 +5 | -8.1838, 5.2650 |
| `intfloat/multilingual-e5-small` | 일치하는 passage가 1순위 | 0.9252 대 0.7632, 한국어 질의에서도 같은 순서 |

에픽 self-consistency gate: 동일 입력의 코사인은 1.0000000에서 1.0000001, 패딩된 배치와 패딩 없는 단일 입력의 일치도는 0.9999999에서 1.0000001, 무관한 문장 점수는 MiniLM 0.1046, bge-m3 0.2519다. 2613 토큰짜리 bge-m3 입력이 단위 벡터로 임베딩되어 512를 한참 넘긴 지점에서 밀린 위치 id를 검증한다.

gate에서 벗어난 항목은 `multilingual-e5-small` 하나다. 무관한 쌍이 0.7390으로 에픽 기준 0.5를 넘는다. 이는 포트가 아니라 체크포인트의 압축된 코사인 범위 때문이다. 테스트는 이 family에 대해 문서화된 0.80 경계를 두고 있고, 실제 판별 gate는 ranking 테스트다.

---

## 5. 검증 요약

GB10(Linux, CUDA)에서 모두 `--profile test-fast --features cuda`로 실행했다.

| 검사 | 결과 |
|------|------|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo check --profile test-fast --features cuda --all-targets` | exit 0 |
| `cargo test --profile test-fast --features cuda --lib models::bert` | 27 passed, 0 failed. 세 번 반복, gate 값 동일 |
| `cargo test --profile test-fast --features cuda --lib embeddings::` | 63 passed, 0 failed |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |
| `mlxcel arch`, `mlxcel list`, `/v1/models` | 두 variant가 `Embedding` 아래 표시되고 서빙 모델이 보고됨 |
| `dimensions: 8` + `encoding_format: base64` | 임베더 셋 모두 8성분 단위 벡터 |

---

## 6. 변경 요약

| 파일 | 줄 수 | 목적 |
|------|-------|------|
| `src/models/bert.rs` | 신규 387 | 인코더 trunk |
| `src/models/bert_config.rs` | 신규 199 | config 해석 |
| `src/models/bert_heads.rs` | 신규 258 | 임베딩 모델과 classification head |
| `src/models/bert_tests.rs` | 신규 | sanitize, config, 위치 id, forward 형태 |
| `src/models/bert_heads_tests.rs` | 신규 | 결정적 fixture 위의 head 동작 |
| `src/models/bert_real_checkpoint_tests.rs` | 신규 | 실제 체크포인트 gate, 없으면 soft-skip |
| `src/embeddings/loader.rs` | +15/-5 | dispatcher arm과 위치 상한 |
| `src/embeddings/model.rs` | +11 | `max_sequence_length()` |
| `src/embeddings/loader_tests.rs` | +27/-6 | unported-family 테스트를 family 무관하게 변경 |
| `src/embeddings/real_checkpoint_tests.rs` | +4/-3 | BERT를 unported 목록에서 제외 |
| `docs/embeddings.md`, `docs/supported-models.md` | +48/-21 | family notes, 표 행, rebase 과정에서 형제 `Family notes` 세 절을 하나로 통합 |

이 브랜치는 형제 포트가 머지되는 동안 두 번 rebase했다(#1410 SigLIP, #1412 ModernBERT). 둘 다 `docs/embeddings.md`에 자체 `## Family notes` 제목을 추가했기 때문에, rebase 과정에서 세 절을 family 표 순서대로 하나의 제목 아래로 모으고 형제들의 문장은 그대로 두었다. `build_family_model`에는 이제 실제 arm이 셋이고 `not yet supported` 목록이 그만큼 줄었다.

---

## 7. 알려진 제약과 후속 과제

- `BAAI/bge-m3`는 `pytorch_model.bin`만 배포하고 mlxcel 다운로더는 safetensors만 받으므로, XLM-RoBERTa 임베더 gate는 `seansitter/bge-m3-safetensors`로 실행했다. `config.json`과 텐서 레이아웃은 BAAI 저장소와 동일하다(391 텐서, `XLMRobertaModel`, 8194 위치 행, 어휘 250002). BAAI 저장소에 safetensors 변환본이 올라오면 미러가 필요 없어진다.
- 검증 머신에 PyTorch, transformers, numpy가 설치되어 있지 않아 위의 기준값은 모두 체크포인트 자체의 모델 카드나 quickstart가 공개한 값이며 로컬에서 재계산한 값이 아니다. transformers가 설치된 머신에서 parity를 돌리면 허용 오차를 1e-2에서 부동소수점 잡음 수준까지 좁힐 수 있다.
- 양자화된 BERT / XLM-RoBERTa 체크포인트는 `UnifiedLinear`와 `UnifiedEmbedding`을 거치므로 로드되어야 하지만, 두 family의 양자화 체크포인트를 구하지 못해 실행하지 못했다.
- 이 family의 공개 체크포인트에는 `gelu` 활성화만 나타난다. `gelu_new`와 `relu`는 구현되어 `hidden_act`로 도달할 수 있지만 실제 체크포인트로 검증하지 않았다.
- `BertSequenceClassifier`의 `/v1/rerank` 연결, 그리고 `bge-m3`의 sparse / ColBERT 출력은 범위 밖이다.
- 성능은 의도적으로 측정하지 않았다. 에픽은 마지막에 조용한 머신에서 성능 측정을 한 번 수행한다.

---

## 참고

- 이슈 #1321, 에픽 #1348
- PR #1408(이슈 #1353), 이 포트가 올라가는 임베딩 하부구조
- 이슈 #1356, classification head의 `/v1/rerank` 소비자
- `docs/embeddings.md`의 `Family notes` 절
