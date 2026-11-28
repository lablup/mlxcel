# 기술 보고서: PR #3245 - LFM2.5 Encoder와 ColBERT를 임베딩 모델로 서빙

**날짜**: 2026-11-23
**상태**: 완료
**위험도**: 낮음 (새 체크포인트에만 해당한다. LFM2.5-Embedding은 이전과 같은 경로를 타고, 새 레이아웃은 모두 로드 시점에 검증한다)

## 요약

남아 있던 LFM2.5 양방향 체크포인트 두 종이 이제 `ModelType::Lfm2Embedding`을 통해 기존 임베딩 경로로 로드된다(#2366, #1325의 후속). `LiquidAI/LFM2.5-Encoder-350M`(`Lfm2BidirectionalForMaskedLM`)은 CLS 풀링 단일 벡터 임베딩 모델이 된다. 이 PR 전에는 `lfm2` 텍스트 생성 모델로 빠졌다. LFM2.5-ColBERT는 HF(`1_Dense/`), mlx-community(`dense.*`, bf16 / 4-bit / 8-bit), mlx-vlm 재내보내기(`projection.*`) 레이아웃 모두에서 `[num_tokens, 128]` 단위 벡터를 돌려주는 다중 벡터 임베딩 모델이 된다. 이 PR 전에는 HF 빌드가 거부되었고, mlx-community 빌드는 projection을 쓰지 않은 채 단일 벡터 CLS 임베딩 모델로 조용히 로드되었다. mlx-vlm d998a620 대비 인코더 hidden state는 토큰별 코사인 1.000000, 두 full-precision ColBERT 빌드는 최소 0.999938로 일치한다. 4-bit 빌드는 bf16 기준 대비 0.98 이상을 유지하고, 모든 빌드에서 MaxSim이 관련 문서를 먼저 고른다.

## 1. 문제 정의

#1325는 LFM2.5-Embedding을 서빙하고 ColBERT는 미뤘다. 그 가드인 `reject_late_interaction`은 `{N}_Dense` 폴더나 접힌 `Dense.linear.*` 텐서가 있을 때만 체크포인트를 거부했다. mlx-community ColBERT 변환본은 projection을 최상위 `dense.weight`로 저장하므로 가드를 통과했고, 1024 폭의 풀링 벡터로 서빙되었다. 오류 없이 틀린 답을 내는 상태였다. 인코더 내보내기는 임베딩 아키텍처로 인식되지 않았고, 키가 공용 sanitize가 벗기지 않는 `lfm2.` 래퍼 아래에 있었다.

## 2. 변경 요약

- 감지(`src/models/detection.rs`): `Lfm2BidirectionalForMaskedLM`을 `EMBEDDING_ARCHITECTURES`에 추가해, `model_type: lfm2`에 이 아키텍처면 `Lfm2Embedding`으로 결정된다.
- 키 처리(`src/models/lfm2_embedding.rs`): `sanitize_family_weights`가 앞의 `lfm2.`를 벗긴 뒤 기존처럼 `sanitize_decoder_embedding_weights`를 실행한다. MLM head는 `embed_tokens`에 묶여 있어 따로 버릴 것이 없다.
- head 선택(`src/models/lfm2_embedding_head.rs`, 신규): `find_projection`은 `dense.0`, `dense`, `projection`, `model.projection` 중 정확히 하나의 linear를 찾고, 둘 이상이면 거부한다. `validate_projection`은 `1_Dense/config.json`이 Identity이고 residual이 없으며 폭이 weight와 맞는지 확인한다. `mlx.proj_dim`도 대조하고, `modules.json`에 Pooling 모듈이 있으면 거부한다. `projection_widths`는 만들어진 `UnifiedLinear`에서 폭을 읽고 `bias`, `scales`, `biases`를 그 폭과 대조한다.
- forward: projection이 있으면 `embed`는 `project_and_normalize(hidden, projection, mask)`(ColIdefics3 / ColQwen2.5와 공용)를 돌려주고, `multi_vector()`는 true, `embedding_dim()`은 projection 폭이다. 이 경로에는 풀링과 `--pooling`이 적용되지 않는다.
- 문서: `docs/embeddings.md`와 `docs/supported-models.md`에서 "ColBERT는 거부된다" 문장을 바꾸고, 인코더가 파인튜닝용 백본임을 밝히고, PyLate와 알려진 차이를 적었다.
- 테스트: 모든 레이아웃, 양자화 head, 패딩, 각 거부 경우에 대한 합성 테스트. 실제 체크포인트 게이트(체크포인트가 없으면 soft-skip)는 `tests/fixtures/lfm2_bidirectional/`에 커밋한 mlx-vlm fixture와 비교한다.

## 3. 기술적 선택과 그 이유

**`ModelType` 하나, head는 weight로 결정.** Encoder, Embedding, ColBERT는 `model_type`, 아키텍처 문자열(인코더 제외), 백본을 공유한다. `ModelType`을 따로 두면 감지 규칙이 중복되는데, 그 규칙으로는 mlx-community ColBERT 빌드와 LFM2.5-Embedding을 config만 보고 구별할 수 없다. 둘을 가르는 것은 projection 텐서의 존재이므로 로더가 그것으로 판단한다.

**ColBERT 경로에서도 패딩을 0으로 만든 conv 입력을 유지.** 업스트림은 ColBERT를 `mask_convolutions=False`로 돌린다. 패딩 없는 행은 어느 쪽이든 같다. 마스크를 유지하면 패딩된 행이 같은 텍스트를 단독으로 돌린 결과와 같아지고, 배치 처리하는 서버에는 이것이 필요하다.

**projection 텐서를 로드 시점에 검증.** MLX는 `Result` 없이 FFI를 넘는 op 안에서 피연산자 shape을 확인한다. 길이가 틀린 `bias`나 packed weight와 행 수가 다른 `scales`가 있으면 첫 요청에서 서버가 abort된다. 보안 리뷰가 이를 찾아 모든 확인을 로드 시점으로 옮겼고, 이제는 해당 텐서 이름을 담은 오류가 된다. 폭이 0인 projection도 같은 이유로 거부한다.

**Dense 옆의 Pooling 모듈을 거부.** Pooling 뒤에 Dense가 오는 sentence-transformers 모델은 풀링된 단일 벡터 head다. 이 확인이 없으면 다중 벡터로 로드된다. 그런 LFM2 체크포인트는 공개된 적이 없으므로, 앞으로 생길 수 있는 조용한 오로드를 막는 가드다.

**PyLate 동작은 호출자 몫으로 둔다.** `[Q] ` / `[D] ` 접두어, mask 토큰 32개로의 쿼리 확장, 구두점 skiplist는 mlx-vlm 레퍼런스에도 없다. 구현하지 않고 알려진 점수 차이로 문서화했다.

## 4. 측정

M3 Ultra, Metal, 고정 토큰 id에 대한 mlx-vlm d998a620 forward 덤프와 비교.

| 체크포인트 | 비교 | 결과 |
|---|---|---|
| lfm2.5-encoder-350m | 마지막 hidden state, 토큰별 코사인 | 최소 1.000000 (서빙 CLS 벡터 1.000001) |
| lfm2.5-colbert-350m (HF, `1_Dense`) | 토큰 벡터, 토큰별 코사인 | 최소 0.999938 |
| lfm2.5-colbert-350m-bf16 (`dense.weight`) | 같음 | 최소 0.999938 |
| lfm2.5-colbert-350m-4bit | bf16 기준 대비 (하한 0.95) | 최소 0.982725, 평균 0.993~0.998 |

관련 문서 대 무관 문서 MaxSim: HF와 bf16에서 6.4645 대 5.6824(mlx-vlm 6.4688 대 5.6871), 4-bit에서 6.4184 대 5.6941. `mlxcel-server -m`은 `/v1/embeddings`에서 인코더에 `(1024,)` 벡터를, ColBERT에 `(5, 128)` / `(9, 128)` 행렬을 돌려준다.

## 5. 측정하지 않은 것

- `-8bit` ColBERT 변환본은 이 호스트에 없었다. 4-bit 빌드와 같은 양자화 `dense.*` 경로를 탄다.
- mlx-vlm `projection.*` 레이아웃을 쓰는 체크포인트는 없고, 합성 테스트로만 확인했다.
- PyLate의 엔드투엔드 검색 점수는 비교하지 않았다. 위의 접두어, 확장, skiplist 차이 때문에 원래 다르게 나온다.

## 6. 학습 포인트

- MLX가 op 안에서 잡는 shape 불일치는 FFI 너머에서 복구할 수 없다. 신뢰할 수 없는 체크포인트에서 읽은 텐서는 op에 닿기 전에 shape을 확인해야 한다.
- late-interaction 체크포인트는 변환 중에 sentence-transformers 폴더를 잃을 수 있으므로, 감지는 폴더 구조가 아니라 텐서를 기준으로 해야 한다.

## 7. 후속 작업

- #1426은 `multi_vector()` 임베딩 모델을 MaxSim 점수로 `/v1/rerank`에 노출하며, 거기에 LFM2.5-ColBERT도 포함해야 한다.
- `docs/model-catalog.tsv`의 인코더와 ColBERT 행은 다른 벤치마크 호스트에서 호스트별 인벤토리 열을 채워야 한다.
