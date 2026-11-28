# PR #2461: DeBERTa-v2 인코더 기반 GLiNER2.5 span 추출

**작성일**: 2026-10-20
**상태**: `gliner2.5-small-v1`로 CUDA (GB10)에서 구현 및 검증 완료, PR 열림

## 요약

에픽 #2391의 마지막 하위 이슈인 #2390을 닫는다. Fastino의 GLiNER2.5를 `mlxcel extract`의 첫 family로 추가한다. GLiNER2.5는 DeBERTa-v3 인코더에 boundary head를 붙인 모델이다. #2383(PR #2456)의 `SpanExtractorModel`을 구현하며, 이후 DeBERTa 기반 reranker나 embedder가 함께 쓸 수 있는 재사용 가능한 DeBERTa-v2 인코더를 추가한다.

## 문제 정의

span 추출 family가 하나도 없어서 `mlxcel extract`는 모든 체크포인트를 거부했다. GLiNER2.5에는 mlxcel에 없던 DeBERTa-v2 인코더가 필요하다. 이 인코더는 log bucketing된 상대 위치를 쓰는 disentangled content-to-position, position-to-content attention을 갖는다. 그 밖에 GLiNER schema 프롬프트, 단어 분할, candidate pool, boundary scorer, overlap 해소도 필요하다. 실제 `fastino/gliner2.5-small-v1` config는 `model_type: gliner2`와 `Gliner2ForSchemaExtraction`을 선언하는데, mlx-vlm 로더는 이를 매핑하지 않는다.

## 기술적 결정

- **재사용 가능한 인코더.** `src/models/deberta_v2.rs`와 `deberta_v2_config.rs`는 GLiNER 코드가 없는 DeBERTa-v2 인코더다. 상대 위치는 Hugging Face의 log bucketing을 따르며, `max_relative_positions < 1`이면 `max_position_embeddings`를 쓴다. mlx-vlm은 bucketing 없이 clip만 하므로 128 토큰을 넘으면 hidden state가 `transformers`와 최대 7e-2까지 달라진다. mlxcel은 `transformers`를 따르며, 306 토큰에서 최대 차이는 5.7e-6이다. 포팅하지 않은 옵션은 config 단계에서 거부하고, `share_att_key` 기본값은 Hugging Face와 같은 `false`이며, 예전 형식인 `"p2c|c2p"` 문자열도 받는다.
- **GLiNER2.5 포팅.** `src/token_classification/gliner2_5*.rs`는 mlx-vlm `gliner2_5`(`gliner2_5.py`, `boundary.py`)를 포팅한다. `[P]`, `[E]`, `[L]`, `[DESCRIPTION]`, `[SEP_TEXT]`를 쓰는 schema 프롬프트, upstream 단어 정규식, 공유 candidate pool, boundary scorer, weighted-interval overlap DP(`_resolve_flat_overlaps`)를 포함하며 DP는 f64로 계산한다. 탐지는 `gliner2`와 `gliner2_5`, 두 architecture 이름을 모두 받고, `boundary`가 아닌 head는 거부한다.
- **계약 유지.** `TokenModelKind::Gliner2_5` variant 하나, detector 하나, `load_span_extractor` arm 하나만 추가했다. #2383 trait은 바뀌지 않았다. 재사용을 위해 `bert::Activation::{from_name, apply}`를 `pub(crate)`로 바꿨다.
- **저장된 dtype 유지.** `gliner2.5-small-v1`은 `encoder_config`에 `dtype: float16`이라고 적혀 있지만 실제 tensor는 float32다. mlx-vlm도 float32로 실행하므로 mlxcel도 저장된 dtype을 그대로 쓴다.
- **레이어마다 CUDA graph 분리.** 12개 레이어 인코더를 CUDA graph 하나로 캡처하면, 새로 캡처한 graph의 첫 실행이 GB10에서 간헐적으로 잘못된 hidden state를 냈다. 306 토큰 입력에서 프로세스의 절반가량이 1e-2에서 9e-1의 오차를 보였고, replay는 정확했다. `MLX_USE_CUDA_GRAPHS=0`에서는 0/80이었다. disentangled position gather가 원인이며, 레이어마다 eval하면 0/192였다. 그래서 `DebertaV2Encoder`는 CUDA에서 레이어마다 graph를 끝낸다. Metal은 바뀌지 않았다. encode와 replay가 비트 단위로 같은지 테스트로 확인한다.
- **리뷰 반영.** 단어 문자는 Python의 `\w`를 따른다(`[\p{L}\p{N}_]`). upstream의 e-mail 분기는 입력 길이에 제곱으로 느려지므로 RFC 5321 길이로 제한했다. config 크기는 `i32`로 검증하고 상한을 둔다(`pool_boundary_top_k` <= 1024, `max_len` <= 32768). pool은 `min(pool_size, top_k^2)`로 padding한다. bos/eos와 상대 위치 테이블 폭은 로드할 때 확인한다. mlx-vlm으로 변환한 export는 `sanitize`의 역방향 이름 변환으로 로드된다.

## 변경 사항

- `src/models/deberta_v2.rs`, `deberta_v2_config.rs`, `deberta_v2_tests.rs` (신규), `bert.rs`, `detection.rs`, `mod.rs`.
- `src/token_classification/gliner2_5.rs`, `_prepare.rs`, `_head.rs`, `_layers.rs`, `_pool.rs`, `_scorer.rs`와 각 테스트 (신규), `loader.rs`, `mod.rs`.
- `src/main.rs` (`mlxcel extract --help` 예시).
- `docs/token-classification.md`는 두 shape 모두에 대해 HTTP route를 명시적으로 두지 않는다고 적는다(upstream도 정의하지 않는다). 이로써 에픽 #2391의 마지막 수용 기준이 충족된다.
- `tests/fixtures/token_classification/`: mlx-vlm GLiNER2.5 fixture, DeBERTa-v2 레이어 fixture, `transformers` 긴 입력 fixture, 생성 스크립트 (mlx-vlm 0.7.8 git `73d366ec`, mlx 0.32.4, transformers 5.20.1, torch 2.14.1+cpu).
- `docs/token-classification.md` (GLiNER2.5 절과 README 예시), `docs/supported-models.md`, `CHANGELOG.md`.

## 검증

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0` 환경에서 테스트 바이너리를 `gpu-lock` 아래 `--test-threads=1`로 실행했고, 오케스트레이터가 최종 head에서 다시 돌렸다. `deberta_v2` 7, `gliner` 22, `token_classification::` 86, `rerank::` 51, `models::detection` 83, `models::bert` 31, `privacy_filter` 20개가 모두 통과했다. bin 테스트 `extract` 5개와 `tag` 4개도 통과했다. clippy, fmt, 저장소 계약 검사도 문제없다.

| 검사 | 결과 |
|---|---|
| `extract --labels company,person,location --include-confidence --include-spans "Apple hired Sam in New York."` | Apple 0.9993 [0,5], Sam 0.9905 [12,15], New York 0.9995 [19,27] |
| `--classify '{"sentiment": ["positive","neutral","negative"]}' "The launch was excellent."` | `positive` |
| `--labels-json '{"person": "a human name", "company": null}' "Ada Lovelace worked with Charles Babbage."` | mlx-vlm과 바이트 단위로 동일 |
| mlx-vlm fixture 7개 케이스 (설명, e-mail/URL, CJK `char` splitter, 분류) | id, 위치, span, offset 일치, confidence 차이 1e-6 이내 |
| DeBERTa-v2 인코더 대 `transformers`, 306 토큰 | 최대 절대 오차 5.7e-6, encode와 replay 비트 단위 일치 |
| 259 토큰 GLiNER 입력, 4회 반복 | 결정적 |
| `mlxcel generate -m <gliner>` | `GLiNER2.5 is a span extractor; use \`mlxcel extract\`` |

`MLX_ENABLE_TF32=1`에서도 GLiNER 출력은 일치한다(confidence가 1e-5 정도 움직인다). DeBERTa 행 비교는 허용 오차 문제가 된다(2e-3 기준에 2.7e-3). 이 모델은 decode 경로가 없는 encoder-only span 추출기이므로 logit trace는 해당하지 않는다.

## 후속 작업

- CUDA graph 첫 실행 오류는 별도 이슈로 다룰 만하다. 큰 graph 하나 안에서 position embedding을 gather하는 다른 인코더도 같은 문제를 겪을 수 있다. 레이어별 graph 분리는 우회책이지 근본 원인 해결이 아니다.
- `gliner2.5-base-v1`과 `-multi-v1`은 이 호스트에 없다.
- `max_len`과 `char` 단어 분할기에는 CLI 플래그가 없다. 둘 다 `Gliner2_5Model`에는 있다.
- 관계 추출, record 추출, DeBERTa reranker와 embedder, HTTP route는 이슈 범위 밖이다.

## 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 25 |
| 추가 줄 | 19175 |
| 삭제 줄 | 28 |

추가된 줄의 대부분(약 14900줄)은 JSON fixture다.

| 커밋 | 메시지 |
|--------|---------|
| `2690e4f1` | feat(models): support GLiNER2.5 span extraction on a DeBERTa-v2 encoder |
| `e92a73ab` | fix(models): harden GLiNER2.5 and DeBERTa-v2 after review |
| `2e4f125b` | test(models): cover GLiNER2.5 prepare limits, document load ceilings |
| `f9c4ec71` | docs: decline an HTTP route for token classification and span extraction |

## 관련 항목

- 에픽 #2391. #2383(PR #2456) 위에 구현했고, 형제 이슈는 #2387(PR #2460, OpenAI Privacy Filter)이다.
