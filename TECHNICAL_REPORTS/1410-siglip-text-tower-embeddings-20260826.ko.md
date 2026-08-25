# 기술 보고서: PR #1410 - SigLIP text tower의 /v1/embeddings 서빙

**작성일**: 2026-08-26
**작성자**: mlxcel contributors
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Low

---

## 요약

PR #1410(이슈 #1341, 에픽 #1348)은 SigLIP text tower를 `POST /v1/embeddings`와 `mlxcel embed`로 서빙한다. `google/siglip-base-patch16-224` 체크포인트는 지금까지 embedding dispatcher까지 도달한 뒤 `not yet supported`에서 멈췄고, 이제 로드되어 768차원 단위 노름 벡터를 반환한다.

이 포팅에서 나중에 코드를 만질 사람이 알아야 할 지점은 두 가지이며, 둘 다 forward pass 자체가 아니다.

첫째, SigLIP의 pooling은 별도 단계가 아니라 padding의 결과다. pad token과 EOS token이 같은 id이므로 모든 입력을 64개 학습 position으로 오른쪽 패딩하면 position 63에는 항상 `</s>`가 놓이고, 63을 고정 slice하는 것만으로 5토큰 캡션과 64토큰 캡션 양쪽에서 레퍼런스의 pooling이 재현된다. `PoolingMode::LastToken`과 attention mask를 쓰는 구현은 같은 토큰을 다른 index에서 뽑게 되고, tower가 mask 없이 동작하므로 결과 벡터가 달라진다.

둘째, 에픽의 공통 acceptance gate가 이 family에 맞지 않는다. 무관한 문장의 cosine이 0.5 미만이기를 기대하지만 SigLIP은 0.707을 낸다. 이를 조정하지 않고 조사했으며, 그 조사 과정이 이 보고서의 핵심이다. 독립적으로 작성한 NumPy 레퍼런스가 동일한 0.707을 재현했고, 이 family의 text 공간 전체가 약 0.65의 바닥값 위에 있다는 사실이 드러났다. 절대 임계값은 올바른 구현으로는 도달할 수 없으므로, gate는 이슈가 실제로 명시한 margin을 검증하도록 바뀌었다.

## 1. 문제 정의

`src/models/detection.rs`는 이미 `model_type: siglip`을 `ModelType::SiglipText`로 라우팅했고 `src/model_metadata.rs`도 이미 `ModelKind::Embedding`으로 등록해 두었다. 둘 다 #1353의 embedding foundation과 함께 들어왔다. 없던 것은 text tower 자체다. `src/vision/encoders/siglip.rs`는 vision 쪽만 구현했고 encoder block은 private이었으며, `VisionEmbeddings`는 patch convolution이라 token/position lookup도 projection head도 없다.

단순 이식으로 끝나지 않게 만든 제약이 셋 있었다.

**Encoder block에는 복사하면 갈라질 결정이 이미 들어 있었다.** 그 파일의 `gelu_pytorch_tanh`는 bf16 `x^3` overflow를 피하려고 일부러 f32에서 계산하고, `select_mlp_activation`은 SigLIP/CLIP/Idefics2 호출자에 걸친 3방향 호환 매트릭스를 담고 있다. text 모듈에 block을 한 벌 더 두면 처음엔 동일하다가 한쪽에 수정이 들어가는 순간 갈라진다.

**Config가 대부분 비어 있다.** base 체크포인트의 `text_config`가 선언하는 키는 `hidden_size`, `intermediate_size`, `num_attention_heads`, `vocab_size`, `model_type` 다섯 개다. tower가 필요로 하는 나머지 네 값은 전부 default에서 오며, default가 틀려도 시끄럽게 실패하지 않는다. 모델은 그대로 로드되고 그럴듯한 단위 벡터를 계속 내놓는다.

**레퍼런스 구현이 설치되어 있지 않았다.** 빌드 호스트에 PyTorch도 transformers도, 심지어 NumPy도 없었다. 절대값을 대조할 oracle이 없고 self-consistency만 가능한 상태였다.

## 2. 기술적 선택

### 2.1 Encoder block을 공유하되, 공유가 vision 경로를 바꾸지 않았음을 증명

대안은 복사였고 위의 drift 이유로 기각했다. 공유의 비용은 하나다. `EncoderLayer::forward`가 `Option<&MlxArray>` mask 인자를 갖게 됐지만 현재 어떤 호출자도 `Some`을 넘기지 않는다. vision tower도 text tower도 mask 없이 동작하기 때문이다.

수치 커널에 인자를 추가하는 변경은 안전하다고 주장하기는 쉽고 보이기는 어렵다. 그래서 주장 대신 보였다. 결정론적 1-block fixture(LCG 시드 weight map, `hidden = 8`, `intermediate = 16`, head 2개)를 **변경 전** `EncoderLayer::forward(x)`에 통과시켜 32개 출력을 먼저 캡처했다. 그 다음 mask 인자를 추가하고, 같은 fixture를 `forward(x, None)`으로 돌려 캡처값과 대조한다. 순서가 이것을 동어반복이 아니라 증거로 만든다. 숫자가 그것을 지키는 코드보다 먼저 존재했다.

golden만으로는 mask를 받고 무시하는 `forward`를 잡지 못하므로, 두 번째 테스트가 배선의 양쪽을 모두 검증한다. 전부 0인 additive mask는 mask 없는 경로를 1e-6 이내로 재현해야 하고, 마지막 key 열을 막는 mask는 출력을 1e-4 넘게 움직여야 한다. 앞쪽만 있으면 인자를 버리는 구현도 통과하고, 뒤쪽이 그것을 떨어뜨린다.

`EncoderLayer::from_weights_parts`는 `&VisionConfig` 대신 `EncoderBlockShape`(hidden size, head 수, LayerNorm epsilon)를 받는다. text tower가 아무 의견도 없는 `patch_size`와 `image_size`를 지어내지 않아도 되게 하기 위해서다.

### 2.2 Padding이 pooling을 하게 둔다

레퍼런스는 `last_hidden_state[:, -1, :]`를 pooling하며 EOS가 "sticky"하다고 주석을 단다. 세 가지 사실이 그것을 고정 slice로 만든다.

| 사실 | 출처 | 결과 |
|---|---|---|
| `pad_token`과 `eos_token`이 모두 `</s>` | `tokenizer_config.json` | 입력이 아무리 짧아도 position 63에는 `</s>`가 온다 |
| `model_input_names: ["input_ids"]` | `tokenizer_config.json` | 레퍼런스 processor가 attention mask를 만들지 않으므로 tower는 mask 없이 동작 |
| `model_max_length: 64`, `max_position_embeddings` 64 | `tokenizer_config.json`, `SiglipTextConfig` default | 입력은 63토큰 + 뒤따르는 `</s>`로 잘린다 |

그래서 `pad_to_max_length()`는 `Some(64)`를 반환하고, 엔진은 모든 micro-batch를 정확히 그 폭으로 패딩하며, `embed`는 index 63을 slice한다. `default_pooling()`은 startup 로그용으로만 `LastToken`을 보고한다. `1_Pooling/config.json`은 참조하지 않고 `MLXCEL_EMBEDDING_POOLING`도 적용되지 않으며, 둘 다 의도한 것이다.

앞으로의 변경이 피해야 할 함정은 이것이다. 공용 `pool()` 헬퍼에 `PoolingMode::LastToken`을 넣으면 index 63이 아니라 마지막 *실제* 토큰이 선택된다. 짧은 캡션에서는 같은 토큰 id가 서로 다른 position에 있는 상황이고, attention이 mask 없이 동작하므로 두 position의 hidden state는 다르다. 결과는 그럴듯해 보이면서 틀린다.

### 2.3 Vision 쪽 activation default를 상속하지 않는다

`VisionHiddenActivation`의 `Default`는 의도적으로 `ExactGelu`다. `hidden_act`를 선언하지 않는 구형 vision 체크포인트가 exact-erf GELU를 유지하도록 하기 위해서다. 반면 `SiglipTextConfig`의 default는 `gelu_pytorch_tanh`다. enum의 `Default`를 그대로 쓰면 하필 이 이슈가 대상으로 삼는, `hidden_act`를 선언하지 않는 그 체크포인트에서 조용히 잘못된 GELU가 선택된다.

`SigLipTextArgs::hidden_act`는 `Option<VisionHiddenActivation>`이며, `None`(키 없음 또는 명시적 null)은 `gelu_pytorch_tanh`를 의미하고, 문자열이 선언되면 공용 `select_mlp_activation`으로 해석된다. 세 경로 모두 테스트한다.

### 2.4 절대 검증을 포기하는 대신 oracle을 만든다

레퍼런스 프레임워크가 없는 상황에서 정직한 선택지는 self-consistency만 내보내거나, 독립 구현을 만드는 것이었다. NumPy와 `tokenizers`가 scratch virtualenv에 깨끗이 설치되어, `siglip_text_reference.py`가 체크포인트의 safetensors와 `tokenizer.json`을 직접 읽어 레퍼런스 forward pass를 재구현한다. tokenize, 63토큰 + `</s>`로 절단, 64로 패딩, token + position embedding, mask 없는 pre-norm block 12개, final LayerNorm, `head`, L2 정규화.

| 경로 | NumPy 레퍼런스 대비 |
|---|---|
| 엔진 경로, 테스트 내, 고정된 12개 성분 | 2.7e-8 |
| `mlxcel embed` 프로세스, 768차원 전체 | 4.5e-5 |
| `mlxcel embed` 두 번째 프로세스 반복 | 비트 동일 |

이것이 증명하는 것: MLX 연산 조합, weight key 매핑, tokenizer 경로, 절단/패딩 규칙, f32 누적이 모두 동일 아키텍처에 대한 두 번째 독립 구현과 일치한다.

증명하지 못하는 것: 두 구현 모두 같은 사람이 레퍼런스를 같은 방식으로 읽고 작성했으므로, 아키텍처를 공통으로 잘못 읽었다면 서로 일치할 것이다. 그에 대한 독립 증거는 수치가 아니라 의미론적이며, 5절의 순서가 그것이다. pooling index가 틀렸거나 projection이 전치된 tower라면 `cat`/`kitten`을 `cat`/`car engine`보다 0.26 위에 놓지 못한다.

두 mlxcel 경로는 각자 비트 안정적이면서 서로는 약 4e-5 차이가 난다. MLX가 프로세스마다 커널을 선택하기 때문이다. 고정 허용오차가 테스트 측정만 보면 정당화되는 1e-7이 아니라 2e-4인 이유가 이 편차다.

### 2.5 실패한 유사도 임계값을 결함이 아니라 family 불일치로 처리

에픽 공통 gate는 무관한 문장이 cosine 0.5 미만이기를 기대한다. 첫 실제 체크포인트 실행은 `a photo of a cat`과 `a diagram of a car engine`에 대해 0.707366을 반환했다.

NumPy oracle이 같은 쌍에 0.707365를 반환하므로 구현은 용의선상에서 제외된다. 한 쌍이 아니라 공간 전체를 측정하니 설명이 된다.

| 문장 6개(동물, 기계, 금융, 음식, 물리) 통계 | 값 |
|---|---|
| 측정한 무관한 쌍 | 14 |
| 최소 | 0.5187 |
| 최대 | 0.7250 |
| 평균 | 0.6531 |
| 유일한 관련 쌍, `cat`과 `kitten` | 0.9664 |

SigLIP text tower는 이미지에 대해 대조 학습되며 다른 텍스트와는 결코 학습되지 않는다. 무관한 캡션 둘을 떼어놓는 힘이 목적함수에 없으므로 text 공간은 비등방적이고 cosine 바닥값이 0.65 근처에 놓인다. 절대 0.5 임계값은 이 family의 올바른 구현으로는 도달할 수 없고, 통과했다면 오히려 버그의 증거였을 것이다.

gate는 이제 이슈가 실제로 명시한 것, 즉 0.1 이상의 margin(측정값 0.259)과, 정말로 붕괴한 tower라면 여전히 넘어설 느슨한 상한, 그리고 2.4의 절대 NumPy 고정값을 검증한다. 이 기하 구조는 `docs/embeddings.md`에 기록해, 운영자가 sentence-transformers encoder처럼 점수에 임계값을 걸지 않고 margin으로 순위를 매기도록 했다.

### 2.6 MLX 테스트를 직렬화하고, 락 없이 얻은 수치는 폐기

이 작업이 리뷰 중일 때 형제 유닛이 `cargo test`의 동시 MLX forward pass가 서로를 오염시키는 것을 측정했다. 한 배치 안의 바이트 동일한 두 행이 cosine 1.0이 아니라 0.999912를 기록했다. `EmbeddingModel`은 단일 스레드로 문서화되어 있고 제품은 그것을 지킨다(서버는 embedding worker 스레드 하나를 소유하고 `mlxcel embed`는 메인 스레드에서 실행). 따라서 위험은 테스트 하네스에 국한되며, 하네스는 테스트 함수를 스레드 풀에서 돌린다.

이 gate들의 허용오차는 전부 1e-4보다 빡빡하므로 락 없이는 모두 무의미했다. 이제 `siglip_text::test_guard`의 `OnceLock<Mutex<()>>` 하나가 새 테스트 모듈 둘을 함께 덮는다. 모듈마다 하나씩 두지 않은 것은 의도적이다. 두 모듈이 같은 encoder block을 구동하므로 모듈별 락으로는 서로 간의 경합이 남는다. poisoned 락은 전파하지 않고 복구하므로, 실패 하나가 이후 모든 테스트로 번져 어느 것이 실제로 깨졌는지 가리는 일이 없다.

락 아래에서 다시 측정한 결과, 연속 3회 실행이 출력된 모든 자리까지 동일했다.

| 항목 | 실행 1, 2, 3 |
|---|---|
| 동일 입력 두 개의 cosine | 1.000000000 |
| `cat`과 `kitten` | 0.966430 |
| `cat`과 `car engine` | 0.707366 |
| margin | 0.259065 |
| 패딩된 batch와 단일 입력 | 5.96e-8 |
| NumPy 대조, 고정된 12개 성분 | 2.7e-8 |

### 2.7 모듈 안의 가드 없는 테스트 하나가 락을 무력화했다

가드를 넣자 오염된 cosine 보고에 없던 두 번째 실패가 드러났다. `cargo test --lib vision::encoders::siglip`을 반복하면 네 번에 한 번꼴로 프로세스가 실행 도중 abort했다.

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  cudaStreamEndCapture(stream, &handle_) failed: operation failed due to a previous error during capture
```

테스트 4개 중 2개만 보고되고 나머지 2개는 아예 실행되지 않았으므로 teardown 현상이 아니다. 테스트가 아직 돌고 있는 동안 CUDA graph capture가 실패한 것이다. 원인은 가드가 불완전했다는 것이다. `pytorch_tanh_gelu_matches_hugging_face_f32_golden`은 이 PR 이전부터 있었고 같은 모듈에 있으며 MLX 연산을 평가한다. 그래서 가드를 잡은 block 테스트와 계속 동시에 실행되며 graph capture와 교차할 수 있었다. 락은 그것을 잡는 테스트만 직렬화한다.

그 기존 테스트 하나에 가드를 넣자 이 유형은 완전히 사라졌다. 이후 10회 반복에서 10회 모두 테스트 4개가 전부 `ok`를 보고했고 실행 도중 abort는 0회였다.

그 10회 중 3회는 여전히 non-zero로 종료했지만, 원인은 다르고 전적으로 결과 출력 이후에 발생한다.

```
  what():  Destroy(handle_) failed: driver shutting down
```

이것은 모든 테스트 결과가 출력된 뒤에 발생하므로 결과에 영향을 줄 수 없으며, 이 PR의 것도 아니다. 이 PR의 코드가 전혀 없는 대조군 필터 `cargo test --lib embeddings::`가 10회 중 1회에서 동일한 메시지를 재현했다. 이 호스트에서 CUDA graph handle 소멸자가 driver 종료와 경합하는 현상이며, 형제 유닛 셋이 GPU를 공유하면서 악화된 것이고, 이 장비에서 이미 알려진 기존 full-suite abort와 일치한다.

| 필터 | 실행 | 테스트 전부 ok | 실행 중 capture abort | 결과 출력 후 teardown abort |
|---|---|---|---|---|
| `vision::encoders::siglip`, 기존 테스트에 가드 넣기 전 | 4 | 3 | 1 | 0 |
| `vision::encoders::siglip`, 넣은 후 | 10 | 10 | 0 | 3 |
| `embeddings::` 대조군, 이 PR 코드 없음 | 10 | 10 | 0 | 1 |

## 3. 구현 상세

| 파일 | 변경 |
|---|---|
| `src/models/siglip_text.rs` | 신규. `SigLipTextArgs`, `sanitize_siglip_text_weights`, `encode`와 `EmbeddingModel` 구현을 가진 `SigLipTextModel`, `load_siglip_text_model`, 공용 `test_guard`. 309줄. |
| `src/models/siglip_text_tests.rs` | 신규. 합성 tower/config 테스트 8개와, 체크포인트가 없으면 soft-skip하는 실제 체크포인트 gate 2개. 635줄. |
| `src/vision/encoders/siglip.rs` | encoder block을 `pub(crate)`로; `EncoderLayer::forward`와 `VisionAttention::forward_impl`에 선택적 mask 전달; `EncoderBlockShape`와 `from_weights_parts` 추가; 내부 호출부 6곳 모두 `None` 전달. |
| `src/vision/encoders/siglip_block_tests.rs` | 신규. 리팩터 이전 golden과 mask no-op / mask-effective 쌍. 237줄. |
| `src/embeddings/loader.rs` | `ModelType::SiglipText` arm이 tower를 생성; dispatcher 인자 두 개의 underscore 접두 제거. |
| `src/models/mod.rs` | 모듈 선언과 re-export. |
| `docs/supported-models.md` | Embedding models 표 행과 고정 폭 설명. |
| `docs/embeddings.md` | Family notes 절, 그리고 "아직 어떤 family도 랜딩하지 않았다"는 현재 상태 문단 수정. |

Weight sanitize는 `vision_model.*`, `logit_scale`, `logit_bias`, 그리고 모든 `position_ids` 버퍼를 버린다. 나머지는 `text_model.` 접두 아래에서 그대로 사용하며, `SiglipTextModel` export도 같은 접두를 유지한다. 두 레이아웃 모두 export되는 모듈이 `text_model` 속성을 소유하기 때문이다.

이슈 본문과 한 가지 다르다. 이슈는 `position_embedding: UniquePtr<MlxArray>`라는 raw table을 지정했지만 여기서는 `UnifiedEmbedding`을 쓴다. dense 체크포인트에서는 동일한 lookup이고, 추가로 양자화 변환본도 로드하며, `VisionEmbeddings`가 vision 쪽 position table을 읽는 방식과 일치한다.

## 4. 테스트 커버리지

| 테스트 | 고정하는 것 |
|---|---|
| `text_config_defaults_match_the_reference_config` | 선언되지 않은 default 4개, 그리고 activation default가 vision enum의 exact-erf가 아니라 tanh라는 점 |
| `text_config_overrides_are_read_including_projection_and_activation` | 선언된 override, 명시적 null, `text_config` 없는 레이아웃 |
| `sanitize_drops_vision_and_logit_keys` | 버려지는 키 형태 4종 |
| `pooling_takes_the_last_position_and_not_cls_or_mean` | pooling 결과가 `head(h[:, L-1, :])`이며 `head(h[:, 0, :])`, `head(mean)` 둘 다와 불일치 |
| `every_position_reaches_the_pooled_slot` | 양방향 도달성과 반복 forward의 재현성 |
| `trait_surface_reports_fixed_width_padding_and_last_token_pooling` | `EmbeddingModel` 표면 전체 |
| `encode_rejects_more_tokens_than_learned_positions` | position table 초과 오류와 짧은 batch의 허용 |
| `embed_rejects_image_inputs` | 이미지 입력 거부 |
| `siglip_base_detects_pads_to_64_and_keeps_trailing_eos` | 탐지, pad id, 64폭 패딩, 절단 시 `</s>` 유지 |
| `siglip_base_text_tower_passes_the_embedding_gate` | limits, 단위 노름, 실제 토큰 집계, margin, batch 대 단일 일치, NumPy 고정값 |
| `encoder_block_shared_with_vision_is_unchanged` | 리팩터 이전 golden |
| `an_all_attend_mask_is_a_no_op_and_a_blocking_mask_is_not` | mask 배선의 양쪽 |

실제 체크포인트 gate는 `src/embeddings/real_checkpoint_tests.rs`의 관례를 따라 체크포인트가 없으면 soft-skip하므로, 체크포인트가 없는 머신에서도 나머지 10개는 실행된다.

## 5. 실제 체크포인트 결과

`google/siglip-base-patch16-224`, `mlxcel embed`, 커널 캐시된 상태, 서로 다른 두 프로세스에서 반복했고 프로세스 간 비트 동일:

| 항목 | 값 |
|---|---|
| 탐지된 family | `SiglipText` |
| 벡터 폭 | 768 |
| `max_length` | 64 |
| 프롬프트 4개의 `prompt_tokens` | 33, 패딩 슬롯 256이 아님 |
| 동일 프롬프트 두 개의 cosine | 1.000000000 |
| `cat`과 `kitten`의 cosine | 0.966439 |
| `cat`과 `car engine`의 cosine | 0.707386 |
| NumPy 레퍼런스 대비 최대 성분별 차이 | 4.461e-5 |

`mlxcel-server -m google/siglip-base-patch16-224` 후 같은 입력 4개로 `POST /v1/embeddings`를 호출하면 동일한 벡터, 단위 노름, `usage.prompt_tokens` 33을 반환한다.

## 6. 검증 요약

| 명령 | 결과 |
|---|---|
| `cargo test --profile test-fast --features cuda --lib models::siglip_text` | 10 passed, 0 failed, 연속 3회 동일 |
| `cargo test --profile test-fast --features cuda --lib vision::encoders::siglip` | 4 passed, 0 failed, 10회 반복 모두 |
| `cargo test --profile test-fast --features cuda --lib embeddings::` | 62 passed, 0 failed |
| `cargo test --profile test-fast --features cuda --lib models::detection` | 40 passed, 0 failed |
| `cargo check --profile test-fast --features cuda --all-targets` | exit 0 |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check` | exit 0 |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |

`vision::encoders::siglip` 10회 반복 중 3회는 모든 테스트가 `ok`를 보고한 뒤 non-zero로 종료했다. 2.7절에서 설명한 기존 `Destroy(handle_) failed: driver shutting down` teardown 경합이다. 이 PR의 코드가 없는 `embeddings::` 대조군도 10회 중 1회 재현했으므로, 이 테스트들의 성질이 아니라 이 호스트의 환경 문제다.

성능 측정은 하지 않았다. 에픽이 마지막에 조용한 머신에서 성능 패스를 한 번 돌리며, 이 호스트에서는 형제 유닛 셋이 동시에 실행 중이었다.

## 7. 변경 요약

| 지표 | 값 |
|---|---|
| 변경 파일 | 8 |
| 추가 줄 | 1,323 |
| 삭제 줄 | 32 |
| 추가 테스트 | 12 |
| 락 아래로 들어온 기존 테스트 | 1 |

## 8. 후속 조치

- SigLIP vision tower와 attention-pooling head를 통한 이미지 임베딩은 이 이슈의 범위 밖이며 여전히 서빙되지 않는다. `supports_images()`는 false이고 오류 메시지가 그렇게 말한다.
- SigLIP 2(`siglip2`) text tower는 탐지되지 않으며 이 포팅의 대상이 아니다.
- 모듈별 테스트 락은 모듈 간을 직렬화하지 않는다. 에픽 #1348의 모든 family가 각자 락을 추가하고 있으므로, 전체 lib 스위트에서는 한 family의 실제 체크포인트 forward가 다른 family의 것과 여전히 함께 돈다. 그 대가는 2.7절이 보여준다. 같은 모듈 안의 가드 없는 MLX 테스트 하나가 네 번에 한 번 프로세스를 abort시켰다. 크레이트 전역 MLX 테스트 guard가 실제 해법이며, 에픽의 family들이 랜딩한 뒤 이슈로 제기할 가치가 있다.
- 프로세스 종료 시의 `Destroy(handle_) failed: driver shutting down` abort는 기존 현상이며 무관한 필터에서도 재현되고, GPU 동시 부하 아래에서 통과한 실행을 10회 중 1회 정도 non-zero로 만든다. 별도 이슈로 다룰 가치가 있다. 결과는 영향받지 않지만, 초록색 스위트를 무작위로 빨갛게 만들고 flaky 테스트로 오독될 것이다.
- NumPy oracle은 저장소가 아니라 scratch 디렉터리에 있다. 실제 체크포인트 gate에 고정된 12개 성분이 남는 것이며, tower의 수치 경로를 다시 손볼 일이 생기면 oracle을 재생성해야 한다.
- 같은 입력에 대해 `mlxcel embed` 프로세스와 테스트 바이너리가 약 4e-5 차이를 보이는 것은 결함이 아니라 프로세스별 커널 선택이다. 다만 이 family에 대한 이후 gate가 주장할 수 있는 허용오차의 하한을 정한다.

## 참고

- 이슈 #1341: SigLIP text tower의 `/v1/embeddings` 서빙.
- 에픽 #1348: embedding family 포팅.
- 이슈 #1353, PR #1408: 이 작업이 올라선 embedding foundation.
- `docs/embeddings.md`: 탐지, pooling, 길이 제한, SigLIP family notes.
- `docs/supported-models.md`: Embedding models 표.
