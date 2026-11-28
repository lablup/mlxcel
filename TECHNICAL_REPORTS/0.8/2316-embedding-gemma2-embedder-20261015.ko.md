# PR #2316: EmbeddingGemma 2 (embedding_gemma2) 멀티모달 임베더 지원

**작성일**: 2026-10-15

**상태**: 구현 및 독립 리뷰 완료; 단위 및 통합 테스트 통과

**위험도**: 중간 (Gemma 4 양방향 스택의 어텐션 마스킹, 레이어별 지오메트리 오버라이드, 프로젝션 전용 PLE)

---

## 요약

PR #2316은 Google의 Gemma 4 백본 기반 0.6B 멀티모달 양방향 임베딩 모델인 EmbeddingGemma 2(`google/embeddinggemma-2` 및 `mlx-community/embeddinggemma-2-*`)에 대한 전체 지원을 구현합니다. 하이브리드 로컬/글로벌 양방향 어텐션 마스크, 스케일 1.0 SDPA, 가중치 없는 `v_norm`(`RMSNormNoScale`), 게이팅된 레이어별 잔차 블록이 포함된 프로젝션 전용 PLE(Per-Layer Embeddings), 모델 내장 `512 -> 768` 임베딩 프로젝션, 마스크 적용 평균 풀링, 마트료시카(Matryoshka) 차원 절단, 선택적 Gemma 4 비전 및 오디오 타워 로딩을 지원합니다. 리뷰 과정에서 이미지 프롬프트 포맷팅의 플레이스홀더 문자열 손상 버그를 수정하고, 논슬라이딩 레이어의 대체 지오메트리 계산을 정정하였으며, 공식 금성/화성 벤치마크 점수(`[0.6775, 0.8348]`)에 맞춘 패리티 테스트 보강 및 멀티모달 토큰 확장 단위 테스트를 추가했습니다.

---

## 1. 문제 정의

### 1.1 배경

Google DeepMind는 2026년 10월 6일 EmbeddingGemma 2(`embedding_gemma2`, `embedding_gemma2_text`)를 출시했습니다. 연속적인 외부 Dense 프로젝션 모듈을 사용하던 1세대 텍스트 전용 EmbeddingGemma(`gemma3_text`)와 달리, EmbeddingGemma 2는 텍스트, 이미지, 오디오에 걸친 네이티브 멀티모달 모델이며 최종 레이어 정규화 직후 모델 내장 선형 프로젝션(`512 -> 768`)을 수행합니다. 이를 mlxcel에서 서비스하기 위해서는 인코더 전용 양방향 실행, 레이어별 GQA/MQA 헤드 차원 오버라이드, 토큰 룩업 테이블 없는 프로젝션 전용 PLE 구현이 필요합니다.

### 1.2 핵심 요구사항 및 해결된 격차

- **레이어별 어텐션 오버라이드**: 20개 로컬 레이어(`sliding_attention`)는 `num_attention_heads: 4`, `num_key_value_heads: 2`, `head_dim: 256`, `rope_theta: 10000.0`을 사용하며, 4개 글로벌 레이어(`full_attention`, 인덱스 `05`, `11`, `17`, `23`)는 `head_dim: 512`, `num_key_value_heads: 1`, `rope_theta: 1000000.0`으로 오버라이드.
- **양방향 어텐션 마스크**: 로컬 레이어는 대칭 슬라이딩 윈도우 마스크(`|q - k| <= sliding_window`)와 양방향 패딩 마스크의 교집합을 적용하고, 글로벌 레이어는 양방향 패딩 마스크를 적용.
- **프로젝션 전용 PLE**: 산란(scatter) 후 입력 프로젝션(`512 -> 24 * 512`, 계수 `hidden_size^-0.5`)을 `per_layer_projection_norm`을 거쳐 각 레이어의 `ple_block`(`per_layer_input_gate` -> GeLU -> `* ple_slice` -> `per_layer_projection` -> `post_per_layer_input_norm`) 및 `layer_scalar`에 전달.
- **Dtype 정책**: 깊은 비스케일 잔차 연결에서 `float16`이 활성화 동적 범위를 초과하므로 `bf16_to_f16_at_load` 변환에서 명시적으로 제외.
- **멀티모달 모듈식 로딩**: 선택적 `vision_tower`(`Gemma4VisionModel`) 및 `audio_tower`(`AudioEncoder`)를 지원하여 임베딩을 `<|image|>` 및 `<|audio|>` 플레이스홀더 위치에 산란.

---

## 2. 기술적 검토 사항 및 발견 사항

### 2.1 보안 및 안정성

- 가중치 키 정규화 과정에서 신뢰할 수 없거나 불필요한 가중치(`lm_head.*`, 로터리 테이블, 포지션 ID)를 제거하고 Hugging Face 및 MLX 레이아웃을 표준 이름으로 정규화.
- 길이가 0인 이미지 입력이나 누락된 모달리티 타워 설정 시 패닉 없이 안전하게 처리.

### 2.2 발견 및 조치된 문제점

1. **`format_text`의 잘못된 플레이스홀더 문자열 (High)**:
   - *문제*: 빈 입력 시 `format_text`가 손상된 문자열 `"<|image><|image|><image|>"`을 반환했습니다. `embed_image`를 통한 이미지 임베딩 시 비정상 서브워드가 생성되어 `expand_image_tokens`의 `else` 분기를 우회하고 입력 시퀀스를 오염시켰습니다.
   - *조치*: `format_text`를 정상적인 항등 포맷팅(`text.to_string()`)으로 수정하고, `expand_image_tokens`의 `else` 분기에서 BOS 등 기존 선두 토큰을 유지한 후 `boi_token_id + image_tokens + eoi_token_id`가 추가되도록 개선했습니다.
2. **`per_layer_config` 대체 지오메트리 버그 (High)**:
   - *문제*: `EmbeddingGemma2Config::from_json`에서 논슬라이딩 레이어는 `per_layer_map.is_none()`일 때만 `fallback_global_kv_heads` 및 `fallback_global_head_dim`을 사용했습니다. `per_layer_config`가 존재하지만 특정 레이어 항목에 해당 필드가 생략된 경우 로컬 레이어 지오메트리(`head_dim: 256`, `kv_heads: 2`)로 잘못 폴백되었습니다.
   - *조치*: 논슬라이딩 레이어에 대해 무조건 `fallback_global_kv_heads`(1) 및 `fallback_global_head_dim`(512)을 사용하도록 수정하여 `rope_theta` 동작과 일치시켰습니다.
3. **참조 검색 벤치마크 패리티 테스트 (Medium)**:
   - *문제*: `real_checkpoint_embeddinggemma2_parity_and_ranking`이 Venus/Mars 없이 임의의 쿼리를 사용하고 공식 검증 점수 비교를 생략했습니다.
   - *조치*: 공식 Venus/Mars 쿼리-문서 쌍을 사용하고 `retrieval_scores_venus_mars: [0.6775, 0.8348]`에 대한 코사인 유사도 검증을 추가했습니다.
4. **멀티모달 토큰 확장 단위 테스트 부재 (Medium)**:
   - *문제*: `expand_image_tokens`에 대한 직접적인 단위 테스트가 없었습니다.
   - *조치*: 텍스트 전용 패스스루 및 이미지 토큰 처리를 검증하는 `expand_image_tokens_text_only_and_vision_paths` 테스트를 추가했습니다.
5. **멀티모달 스캐터 빈 리스트/Null DoS 및 오디오 0길이 0나누기 방지 (High)**:
   - *문제*: `forward_multimodal`에서 `concat_many(&feat_list, 1)`가 `!arrays.is_empty()`를 단언하여, 멀티모달 전처리에서 0개 아이템 또는 null 픽셀 값이 발생할 경우 서버 패닉이 발생했습니다. 또한 빈 오디오 시퀀스 처리 시 `masked_scatter`의 나머지 연산에서 0나누기 패닉이 발생했습니다.
   - *조치*: `concat_many` 호출 전 `!feat_list.is_empty()` 검사를 추가하고, 스캐터 전 `audio_size > 0` 가드를 적용했으며, `expand_image_tokens`에서 소프트 토큰 수를 검증하도록 보강했습니다.
6. **혼합 양자화 boolean `false` 오버라이드 무시 문제 (High)**:
   - *문제*: `quant_params_for_prefix`가 `override_val.as_object()`만 확인하여 boolean `false` 오버라이드(예: `"language_model.ple": false`)를 무시했습니다. 그 결과 기본 group-64 4-bit 양자화 `(64, 4)`로 폴스루되어 비양자화 모듈을 양자화된 모듈로 취급해 로드 시 가중치가 손상되었습니다.
   - *조치*: `override_val.as_bool() == Some(false)`인 경우 `(0, 0)`(비양자화)을 반환하도록 수정하고, `text_config` 및 루트 `config` 모두를 확인하며, `"ple"` 및 `"language_model.ple"` 접두사가 `.ple_block`에 매핑되도록 처리했습니다.
7. **짧은 시퀀스에 대한 슬라이딩 윈도우 마스크 2차 메모리 할당 최적화 (High)**:
   - *문제*: `sliding_window = 512`일 때 시퀀스 길이 `L <= 513`인 모든 시퀀스는 토큰 쌍 간 거리가 항상 512 이하를 만족합니다. 짧은 시퀀스에 대해 `[B, 1, L, L]` float32 텐서(최대 268MB)를 매번 생성하여 불필요한 VRAM 사용과 그래프 생성 오버헤드가 발생했습니다.
   - *조치*: `has_sliding && l > sliding_window + 1` 조건을 적용하여 `L <= sliding_window + 1`일 때는 `[B, 1, 1, L]` 전체 양방향 마스크 브로드캐스트로 폴백하여 메모리 사용량과 피크 VRAM을 절감했습니다.
8. **루프 내 호스트 스칼라 전송 및 불필요한 복사 제거 (Medium)**:
   - *문제*: `layer_scalar`에서 `from_slice_f32` 및 `astype`을 통해 불필요한 호스트 임시 벡터 할당과 타입 변환이 발생했습니다.
   - *조치*: 호스트 변환을 `mlxcel_core::multiply_scalar`로 대체하고, `inputs_embeds`의 불필요한 복제를 제거했으며, 스칼라 dtype이 활성화 dtype과 이미 일치하는 경우 `astype`을 건너뛰도록 최적화했습니다.
9. **비정상 설정 지오메트리 하한 및 아키텍처 탐지 보강 (Low)**:
   - *문제*: 0 또는 음수 설정 치수가 들어올 경우 0나누기 패닉이 발생할 수 있었습니다. 또한 중첩된 `text_config.architectures`가 탐지 및 가중치 정규화에서 확인되지 않았습니다.
   - *조치*: 모든 지오메트리 파라미터에 `.max(1)` 하한을 적용하고, `detection.rs` 및 `sanitize.rs`에서 `text_config.architectures`를 확인하도록 지원했으며, `lm_head.*` 가중치 제거 시 `model.` 및 `language_model.` 접두사를 선행 제거하도록 개선했습니다.

---

## 3. 기술적 선택과 그 이유

- **`mlxcel_core::compiled_geglu_approx_activation` 재사용**: EmbeddingGemma 2의 MLP와 PLE 게이팅 블록 모두 `gelu_pytorch_tanh`(GeGLU approx)를 사용합니다. 컴파일된 C++ / HIP 커널을 재사용하여 그래프 컴파일 오버헤드를 방지하고 Gemma 4 생성 추론과 동작을 통일했습니다.
- **포함적(inclusive) 슬라이딩 윈도우 마스크**: `create_bidirectional_window_mask(attention_mask, sliding_window + 1)`를 사용하여 EmbeddingGemma 2의 포함적 `|q - k| <= sliding_window` 정의와 정확하게 매핑했습니다.
- **모델 내장 프로젝션 vs 외부 모듈**: EmbeddingGemma 2는 풀링 전 순방향 패스 내부에서 `language_model.embedding_projection`을 실행하여 외부 `Dense` 모듈 없이 표준 `pool(&hidden, mask, PoolingMode::Mean)` 및 마트료시카 절단을 지원합니다.

---

## 4. 검증 결과

- `cargo test --lib embedding_gemma2`: 8개 통과, 0개 실패, 1개 무시(체크포인트 필요 통합 테스트).
- `cargo test --lib embeddings::`: 85개 통과, 0개 실패.
- `cargo clippy --lib --tests -- -D warnings`: 경고 0건.
- `cargo fmt --check`: 통과.
