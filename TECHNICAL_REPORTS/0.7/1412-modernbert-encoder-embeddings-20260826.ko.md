# 기술 보고서: PR #1412 - /v1/embeddings용 ModernBERT 인코더

**작성일**: 2026-08-26
**작성자**: mlxcel 기여자
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1412는 ModernBERT 인코더(local/global 어텐션 교대, RoPE, GeGLU)를 이식해 `nomic-ai/modernbert-embed-base`가 `POST /v1/embeddings`와 `mlxcel embed`로 서빙되게 한다. 아울러 #1356의 `/v1/rerank`가 사용할 `ModernBertForSequenceClassification` 헤드를 추가한다. #1353이 만든 임베딩 기반 위에 올라가는 첫 번째 패밀리 forward pass이므로, 남은 패밀리 이식이 따를 패턴(`src/models/` 아래 패밀리 모듈, 로더 디스패처의 arm 하나, 체크포인트가 없으면 조용히 건너뛰는 실제 체크포인트 게이트)도 함께 정한다.

---

## 1. 문제 정의

### 1.1 배경

#1353 이후 mlxcel은 ModernBERT 체크포인트를 탐지해 `/v1/embeddings`로 라우팅했지만 forward pass가 없었다. `nomic-ai/modernbert-embed-base`를 로드하면 `ModernBERT encoder is detected as an embedding checkpoint, but this embedding family is not yet supported`가 나왔다. 탐지 테이블, `ModelType::ModernBert` variant, `mlxcel arch` 항목, `ModelKind::Embedding` 등록은 모두 이미 있었고 모델만 없었다.

### 1.2 아키텍처가 요구하는 것

ModernBERT는 하이퍼파라미터만 다른 BERT 변형이 아니다. 기존 인코더 코드에 전례가 없는 성질이 다섯 가지다.

- **어텐션 교대.** 세 층 중 두 층은 양방향 sliding window를, 나머지 한 층은 전체 시퀀스를 본다. 층 인덱스만으로 결정된다.
- **RoPE base가 둘.** local 층은 `local_rope_theta`(10000), global 층은 `global_rope_theta`(160000)로 회전한다. 16배 차이다.
- **위치 테이블 없음.** RoPE가 절대 위치를 대체하므로 BERT, XLM-RoBERTa, SigLIP과 달리 `max_position_embeddings`가 입력 길이를 제한하지 않는다.
- **융합 projection.** `Wqkv`는 `[3 * hidden, hidden]` 텐서 하나, `Wi`는 `[2 * intermediate, hidden]` 텐서 하나다.
- **0번 층의 norm 부재.** 상위 구현이 `layers.0.attn_norm`을 `nn.Identity()`로 두므로 체크포인트에 해당 텐서가 아예 없다.

### 1.3 위험성

| 위험 | 영향 | 변경 전 발생 가능성 |
|------|------|---------------------|
| local/global 패리티나 RoPE base를 잘못 적용해 그럴듯하지만 품질이 떨어진 임베딩 생성 | High | High. shape assertion으로는 잡히지 않음 |
| 융합 projection을 weight 슬라이싱으로 분리해 양자화 export가 조용히 깨짐 | High | Medium |
| `layers.0.attn_norm` 부재를 로드 오류로 처리해 모든 정상 체크포인트를 거부 | High | High |
| reranker 헤드가 자기 logits와 어긋나는 label 수를 광고 | Medium | Medium. #1356이 사용하기 전까지 드러나지 않음 |

---

## 2. 기술적 선택과 그 이유

### 2.1 마스크만이 아니라 RoPE base도 함께 교대

`ModernBertArgs::is_local_layer`와 `ModernBertArgs::rope_base`를 층 생성 시점에 참조하는 단일 진실 공급원으로 두었다. 마스크만 이식한 사람은 로드되고 실행되며 유한한 단위 노름 벡터를 반환하지만 검색 품질이 눈에 띄게 나쁜 모델을 만들게 된다. `layer_parity_selects_local_and_global_rope_base`가 아홉 개 층 인덱스에서 두 측면을 함께 고정하고, 상위 구현의 `if config.local_rope_theta is not None`에 해당하는 `local_rope_theta: null` 시 global base 폴백도 함께 검증한다.

### 2.2 융합 projection은 matmul 이후에 분리하고 weight는 건드리지 않음

`Wqkv`와 `Wi`는 `UnifiedLinear`로 로드하고 projection 출력에 `slice_axis`를 적용해 나눈다. weight 텐서를 직접 자르는 편이 읽기 쉽고 여기 있는 dense f32 체크포인트에서는 동작하지만, 양자화 export는 각 융합 텐서를 자체 scales/biases와 함께 하나의 단위로 패킹하므로 패킹된 평면을 자르면 쓰레기 값이 나온다. dense 경로에서 비용이 없고, 향후 양자화 ModernBERT가 수정 없이 로드되는 유일한 근거다.

### 2.3 serde alias 대신 epsilon 필드 두 개

이슈는 `norm_eps`에 `#[serde(alias = "layer_norm_eps")]`를 지정했다. 그러나 공개된 두 체크포인트 모두 같은 `config.json`에 **두 키를 동시에** 담고 있고, serde는 alias가 있는 필드가 두 이름으로 동시에 제공되면 duplicate-field 오류를 낸다. 즉 alias 방식은 모든 실제 체크포인트에서 파싱에 실패했을 것이다. `norm_eps`와 `layer_norm_eps`를 독립적인 `Option<f32>`로 두고 `norm_eps()` 접근자로 해석한다. `norm_eps_accepts_either_spelling_and_both_together`가 네 가지 조합을 모두 고정하므로 나중에 alias로 되돌리면 즉시 실패한다.

### 2.4 `num_labels`를 config가 아닌 텐서에서 유도

최초 구현은 `num_labels`를 `config.json`에서 읽었다(`num_labels`, 없으면 `len(id2label)`, 없으면 1). 합성 테스트가 그 결과를 드러냈다. label 수를 실제보다 적게 적은 config는 `num_labels()`가 `logits`의 실제 폭과 어긋나게 만든다. #1356이 그 접근자로 출력 크기를 잡을 것이므로, 불일치는 원인에서 멀리 떨어진 하위 shape 버그로 나타났을 것이다. 이제 `classifier.weight`의 행 수가 기준이다. 행 수는 양자화로 패킹되지 않으므로 두 경로 모두에서 옳고, 열 수는 dense 경로에서만 `hidden_size`와 대조하며 config와 텐서가 어긋나면 조용히 받아들이지 않고 경고를 남긴다.

### 2.5 reranker 헤드는 탐지가 아니라 디렉터리로 접근

`is_embedding_checkpoint`는 `architectures[0]`이 `ForSequenceClassification`으로 끝나면 의도적으로 `Ok(None)`을 반환한다. reranker는 embedder가 아니며, #1353의 테스트가 `Alibaba-NLP/gte-reranker-modernbert-base`는 임베딩 variant로 탐지되지 않음을 단언한다. 이 규칙을 약화하는 대신 `ModernBertSequenceClassifier::load`가 `config.json`을 직접 읽고 검증하며 디렉터리로 접근된다. 따라서 #1356의 `/v1/rerank` 배선은 탐지를 건드리지 않고 헤드를 채택할 수 있다.

---

## 3. 구현 세부

### 3.1 Forward Pass

```
h = LayerNorm(tok_embeddings[input_ids])                    # embeddings.norm, bias 없음
global_mask  = create_bidirectional_padding_mask(mask)      # [B, 1, 1, L]
sliding_mask = create_bidirectional_window_mask(mask, local_attention / 2 + 1)   # [B, 1, L, L]
for i in 0..num_hidden_layers:
    local = i % global_attn_every_n_layers != 0
    x = h if i == 0 else LayerNorm(h)                       # layers.0에는 attn_norm 없음
    q, k, v = split(Wqkv(x))                                # 각각 [B, heads, L, head_dim]
    q, k = fast_rope(.., head_dim, traditional=false, base = local ? local : global)
    h = h + Wo(sdpa(q, k, v, head_dim^-0.5, local ? sliding_mask : global_mask))
    inp, gate = chunk(Wi(LayerNorm(h)))
    h = h + Wo_mlp(gelu(inp) * gate)
last_hidden_state = LayerNorm(h)                            # final_norm
```

window 인자가 `local_attention / 2 + 1`인 이유는 `create_bidirectional_window_mask`가 `|q - k| >= window`에서 차단하는 반면 ModernBERT는 `|q - k| <= local_attention / 2`를 어텐션하기 때문이다. 공개된 `local_attention: 128`에서 경계는 65이며, `sliding_mask_attends_within_64_and_blocks_beyond`가 additive 마스크 값을 직접 읽어 검증한다.

### 3.2 가중치 레이아웃과 sanitize

`sanitize_modernbert_weights`는 선택적 선행 `model.`(MLM/classifier export에는 있고 `ModernBertModel`에는 없음)을 제거하고, `decoder.*`와 `pooler.*`는 항상 버리며, 분류 헤드를 만드는 경우가 아니면 `head.*`와 `classifier.*`도 버린다. 따라서 `ModernBertForMaskedLM`은 MLM 헤드를 버린 순수 embedder로 로드된다.

`Wqkv` 블록 순서가 Q, K, V이고 각 블록이 모든 head를 담는다는 사실은 상위 구현의 `qkv.view(bs, -1, 3, num_heads, head_dim)`에서 따라온다. 2304 폭 축이 head 분할 이전에 768 폭 블록 세 개로 나뉜다. `Wi`의 절반은 (input, gate) 순서이므로 활성화는 앞쪽 절반에, gate 곱은 뒤쪽 절반에 적용된다.

---

## 4. 테스트 하네스 정확성 문제

검증 중 발견한 세 가지는 모델이 아니라 테스트의 문제였고, 각각 그대로 두었다면 오해를 부르는 게이트가 머지될 뻔했다.

### 4.1 여러 스레드에서 동시에 구동된 MLX

`EmbeddingModel`은 `src/embeddings/model.rs`에서 정확히 한 스레드에서만 쓰인다고 문서화돼 있고, 서버는 모델당 전용 MLX 소유 워커로 그 계약을 뒷받침한다. `cargo test`는 기본적으로 테스트를 병렬 실행하므로, 계약을 어긴 유일한 구성 요소가 테스트 스위트였다. 이 CUDA 호스트에서 두 가지 증상이 나타났다.

- **조용히 틀린 값.** 병렬 실행 세 번 중 한 번에서, 한 배치 안의 바이트 단위로 동일한 두 행이 cosine 1.0이 아닌 0.99991을 기록했고 reranker logit이 0.05 움직였다. 반면 모든 배치의 첫 행은 3.0652723으로 비트 단위 동일했다. 부동소수점 reduction 순서로는 한 배치 내 동일한 두 행이 9e-5만큼 어긋날 수 없다.
- **abort.** MLX의 CUDA graph capture 안에서 `cudaStreamEndCapture ... operation failed due to a previous error during capture`, SIGABRT.

이제 두 모듈의 MLX를 건드리는 모든 테스트가 공유 모듈 락을 잡는다. 최초 수정은 실제 체크포인트 게이트 다섯 개만 보호하고 인코더를 만드는 합성 테스트 아홉 개를 놓쳤는데, 이후 abort가 정확히 그 지점을 때렸다. 최종 감사 결과 MLX를 건드리는 14개 테스트가 모두 락을 잡고, 잡지 않는 5개는 순수 config 파싱 또는 파일시스템 테스트다.

### 4.2 아티팩트에 맞춘 허용 오차

padding 불변성 게이트는 처음에 성분별 최대 편차 1.2e-3으로 실패했고, 이를 f32 reduction 순서 잡음으로 해석해 5e-3 경계를 부여했다. 그 해석은 틀렸다. 편차의 원인은 위의 교차 간섭이었다. 직렬화 이후 측정값은 1.3e-7이고 경계는 1e-4다. 아티팩트에 맞춰 잡은 허용 오차는 없느니만 못하다. 엄밀해 보이면서 게이트를 조용히 넓히기 때문이다.

### 4.3 자기 전제를 가정한 게이트

긴 문서 게이트는 이름이 4096 토큰이었지만 실제로는 문장을 220번 반복할 뿐 결과를 확인하지 않았다. 실제 길이는 4187 토큰이다. 이제 테스트가 입력이 4096 토큰을 넘는다는 것과 `max_length` 아래에 있어 절단되지 않는다는 것을 함께 단언하므로, 문장을 수정해도 통과하면서 짧은 시퀀스 테스트로 조용히 축소되는 일이 없다.

---

## 5. 실제 체크포인트 결과

직렬화 이후 15회 연속 실행에서 게이트 값이 바이트 단위로 동일했다.

| 게이트 | 관측값 | 요구 조건 |
|--------|--------|-----------|
| 동일 입력 cosine | 0.999999583 | 1.0에서 1e-6 이내 |
| 질의 vs 관련 문서 | 0.625270 | 무관 문서를 0.15 이상 앞서야 함 |
| 질의 vs 무관 문서 | 0.144859 | 0.5 미만 |
| 패딩 배치 vs 단일 입력 | cosine 0.999999642, 최대 성분 편차 1.3e-7 | 1e-3 이내 |
| 4187 토큰 문서, 배치 vs 단독 | cosine 0.999999821 | 1e-3 이내 |
| 벡터 L2 노름 | 1.000000004 / 1.000000025 | 1.0에서 1e-5 이내 |
| gte-reranker logits (관련, 무관) | 3.0652723, -1.1471016 | 유한한 `[B, 1]` |

종단 확인으로 `mlxcel embed`는 대각선 1.0000, 질의 vs t-SNE 문서 0.6253, 질의 vs 에펠탑 문서 0.1448의 cosine 행렬을 dim 768, max_length 8192로 반환한다. `mlxcel-server`는 `GET /v1/models`에서 `modernbert-embed-base`를, `POST /v1/embeddings`에서 cosine 0.62521인 단위 노름 768차원 벡터 두 개를 반환하며, `dimensions: 256`은 재정규화된 256개 성분을 돌려준다.

---

## 6. 검증

| 명령 | 결과 |
|------|------|
| `cargo test --profile test-fast --features cuda --lib models::modernbert` | 19 passed, 0 failed |
| `cargo test --profile test-fast --features cuda --lib embeddings::` | 62 passed, 0 failed |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check` | exit 0 |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |
| `mlxcel embed` / `mlxcel-server` + `POST /v1/embeddings` | 실제 체크포인트로 확인 |
| `mlxcel arch` / `mlxcel list` | Embedding 아래 ModernBERT encoder 표시, 두 체크포인트 모두 표시 |

clippy가 잡은 다섯 건은 모두 기계적 수정이었다. `manual_is_multiple_of` 두 건(`is_multiple_of(0)`은 패닉 없이 반환하므로 `%`가 가진 잠재적 0 나눗셈 형태도 함께 제거된다), `neg_multiply` 한 건, f32가 표현할 수 없는 erf 참조 상수의 `excessive_precision` 두 건이다.

---

## 7. 변경 요약

| 파일 | 변경 |
|------|------|
| `src/models/modernbert.rs` | 신규. config 파싱과 검증, 가중치 sanitize, GeGLU 분리, layer와 encoder. |
| `src/models/modernbert_heads.rs` | 신규. `EmbeddingModel` 구현과 sequence-classification 헤드. |
| `src/models/modernbert_tests.rs` | 신규. 패리티, 마스크, GeGLU, sanitize, config, 탐지, classifier를 다루는 단위 테스트 14개. |
| `src/models/modernbert_real_checkpoint_tests.rs` | 신규. 조용히 건너뛰는 실제 체크포인트 게이트 5개. |
| `src/embeddings/loader.rs` | `not yet supported` match에서 `ModelType::ModernBert` arm만 분리. |
| `src/models/mod.rs` | 모듈 및 테스트 모듈 등록. |
| `docs/supported-models.md` | Embedding 표 행 추가. |
| `docs/embeddings.md` | ModernBERT 패밀리 절과 상태 갱신. |

합계: 8개 파일, +1866 / -2.

---

## 8. 후속 조치와 미검증 영역

향후 유지보수자가 이미 다뤄졌다고 가정하지 않도록 기록한다.

- **양자화 ModernBERT 체크포인트가 없어 테스트하지 못했다.** 2.2의 projection 이후 분리 설계가 양자화 경로를 성립시키지만, 양자화 export가 없어 실제로 실행하지는 못하고 논증에 머물렀다.
- **transformers와의 성분 단위 대조가 없다.** 이 호스트에 PyTorch와 transformers가 설치돼 있지 않아 참조 구현과의 수치 일치를 직접 확인하지 못했다. 검증은 이슈가 제시한 수용 기준(상대 순위, 0.15 마진, 단위 노름, 패딩 불변성)에 근거하며 텐서 단위 parity는 아니다. parity 하네스가 남은 가장 강력한 게이트다.
- **MLM 경로는 합성 테스트만 있다.** `sanitize_modernbert_weights`가 `head.*`와 `decoder.*`를 버리고 합성 테스트도 있지만, 실제 `ModernBertForMaskedLM` 체크포인트를 로드하지는 않았다.
- **길이 커버리지는 4187 토큰까지**이며 체크포인트가 허용하는 8192 전체는 아니다.
- **reranker 순서 검증은 한 쌍뿐이다.** 헤드가 관련 문서를 무관 문서보다 위에 두는 것은 확인했으나 전체 순서 검증은 #1356의 몫이다.
- **기존부터 있던 teardown race, 이 PR 범위 밖.** 통합 `models::modernbert` 필터가 `test result: ok` 이후 프로세스 종료 시점에 `Destroy(handle_) failed: driver shutting down`으로 abort하며 통과한 실행을 exit 101로 만드는 경우가 있다. 수정 이후 17회 중 2회 관측됐고, 가장 최근 10회 연속 실행과 단일 모듈 9회 실행에서는 0회였으며, 두 번 모두 형제 유닛이 같은 GPU를 점유하던 시점과 겹쳤다. 수정되지 않은 main에서 이미 기록됐고 손대지 않은 `embeddings::` 스위트에서도 재현된, 부하 의존적인 MLX CUDA 컨텍스트 teardown race다. 이를 감추기 위해 테스트 명령을 변형하지는 않았다.

---

## 참고

- 이슈 #1332(본 이식), 에픽 #1348(임베딩 패밀리)
- PR #1408 / 이슈 #1353(임베딩 기반: pooling, 마스크, `ModelKind::Embedding`, `/v1/embeddings`)
- 이슈 #1356(`/v1/rerank`, `ModernBertSequenceClassifier` 사용)
- `docs/embeddings.md`, `docs/supported-models.md`
