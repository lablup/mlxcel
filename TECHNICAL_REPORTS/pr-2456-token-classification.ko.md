# PR #2456: 토큰 분류와 span 추출 (mlxcel tag, mlxcel extract)

**작성일**: 2026-10-19
**상태**: Rampart 체크포인트로 CUDA (GB10)에서 구현 및 검증 완료, PR 열림

## 요약

에픽 #2391의 기반 이슈인 #2383을 닫는다. 기존 generate, embed, rerank 경로 옆에 토큰별 출력 경로를 추가하고, BERT / XLM-RoBERTa 토큰 분류 head로 처음부터 끝까지 연결한다. `mlxcel tag`는 Rampart로 엔티티를 태깅하고 가리며, mlx-vlm과 span 단위로 일치한다. `mlxcel extract`는 GLiNER2.5(#2390)가 구현할 span 추출 계약을 담는다. OpenAI Privacy Filter(#2387)는 같은 파이프라인 위에서 `TokenClassifierModel`을 구현한다.

## 문제 정의

mlxcel에는 입력 토큰마다 예측을 하나씩 돌려주는 경로가 없었다. 모든 모델 head는 어휘 projection(generate), pooled 벡터(embed), 쌍 점수(rerank) 중 하나로 끝났다. 그래서 Rampart 같은 토큰 분류기나 GLiNER 같은 span 추출기를 붙일 자리가 없었고, mlx-vlm은 같은 체크포인트용으로 `mlx_vlm.token_classification`을 이미 제공한다.

## 기술적 결정

- **trait 두 개, 파이프라인 하나.** `TokenClassifierModel`은 감싼 window 하나에 대해 위치별 log-probability를 돌려준다. window 방식의 `TokenClassifier<M>` 파이프라인이 토큰화, windowing, special 행 제거, f32 log-softmax, 디코딩, span 조립을 맡으므로 모델은 logit만 내면 된다. `SpanExtractorModel`은 엔티티를 직접 돌려주는 모델(GLiNER)을 위한 별도 trait이고, `ExtractOptions`, `EntityResult`, `ClassifyTask`는 upstream의 JSON 형태를 따른다.
- **디코드 모드는 모델이 선언.** `decode_modes()[0]`이 기본값이고, `--argmax`는 `Argmax`를 강제하며, 모델이 나열하지 않은 모드는 거부한다. trait을 바꾸지 않고 #2387의 Viterbi 디코더를 넣을 여지를 남긴다.
- **upstream span helper를 그대로 이식.** `spans.rs`는 upstream의 label 파싱(BIO와 BIOES), span 구성, 단어 묶기, 구두점 병합, trimming, placeholder, 가리기 helper를 이식한다. offset은 Unicode scalar 단위라서 멀티바이트 텍스트도 upstream과 맞는다.
- **config만 보는 탐지와 확장 표.** `loader.rs`는 `config.json`에서 계열을 `TokenModelKind`로 탐지한다. 계열마다 variant 하나와 탐지 arm 하나이므로 #2387과 #2390은 각각 한 행만 추가하면 된다. 탐지는 config만 보므로 `id2label` 너비와 classifier 텐서의 일치 여부는 로드 시점에 검사한다.
- **기존 라우팅은 그대로.** `get_model_type`은 여전히 Rampart를 BERT embedder로 탐지하므로 `mlxcel embed`와 rerank는 이전과 똑같이 동작한다. `tag`와 `extract`는 자체 탐지를 쓴다. 형태가 맞지 않는 체크포인트는 올바른 하위 명령 이름을 알려주는 오류로 실패한다.
- **XLM-RoBERTa context.** 파이프라인은 주소 지정 가능한 위치 수(`max_position_embeddings - pad - 1`)를 window 크기로 쓴다. upstream처럼 `max_position_embeddings`를 그대로 쓰면 위치 테이블 밖을 참조한다.
- **리뷰 후 보강.** `redact`는 `decode_spans`가 만드는 정렬되고 겹치지 않는 span을 한 번에 순회한다. 이전처럼 텍스트 길이와 span 수의 곱에 비례하는 비용이 들지 않는다. 그 밖의 span 목록에는 upstream의 오른쪽에서 왼쪽으로의 splice를 유지한다. `BertTokenClassifier`는 `max_position_embeddings`보다 짧은 위치 테이블을 거부하고, tagger는 word-embedding 테이블 밖의 토큰 id를 거부한다. 문서에는 JSON 출력의 `text`와 `spans[].text`가 가려지지 않는다는 점을 적었다.

## 변경 사항

- `src/token_classification/`: `mod.rs`(trait, `DecodeMode`, window 파이프라인), `spans.rs`, `extract.rs`, `loader.rs`, `bert.rs`, 테스트(`spans_tests.rs`, `mod_tests.rs`, `loader_tests.rs`, `extract_tests.rs`, `real_checkpoint_tests.rs`).
- `src/models/bert_heads.rs`: encoder에 위치별 `classifier` projection을 붙인 `BertTokenClassifier`, 그리고 `bert_heads_tests.rs`, `bert_tests.rs`.
- `src/commands/tag.rs`, `src/commands/extract.rs`, `src/commands/mod.rs`, `src/main.rs`, `src/lib.rs`: 두 하위 명령을 `mlxcel` 바이너리에 연결.
- `tests/fixtures/token_classification/`: mlx-vlm parity fixture, 생성 스크립트, Rampart config.
- 문서: 새 `docs/token-classification.md`, `docs/supported-models.md`의 섹션, README quick tour, 문서 색인, CHANGELOG.

## 검증

GB10, `--profile test-fast --features cuda`, 테스트 바이너리는 `gpu-lock` 아래에서 `--test-threads=1`과 `MLX_ENABLE_TF32=0`으로 실행했다.

| 검사 | 결과 |
|---|---|
| lib 테스트, 필터 `token_classification` | 43개 통과 |
| lib 테스트, 필터 `bert` (새 head와 변경 없는 embed, rerank 테스트) | 56개 통과 |
| bin 테스트, 필터 `extract`와 `tag` | 6개 통과 |
| `mlxcel tag -m models/mlx/rampart-mlx-fp16 --keep CITY,STATE,ZIP_CODE ...` | 모델 카드 출력과 같은 `I'm <GIVEN_NAME> <SURNAME>, <BUILDING_NUMBER> <STREET_NAME>, Los Angeles CA 90012, <EMAIL>` |
| `rampart-mlx-4bit`로 같은 실행 | 같은 span 8개, 같은 offset |
| mlx-vlm parity fixture (mlx CPU 위 mlx-vlm 0.7.8) | label, offset, text, 가리기 결과 모두 정확히 일치: 23, 12, 42 토큰(span 8, 3, 6개)과 window 경계가 단어 안에 걸리는 706 토큰 2-window 사례(span 81개) |
| clippy (`-p mlxcel --lib --tests --no-deps -D warnings`), fmt, 계약 스크립트 | 통과 |

fixture 생성기는 top-1과 top-2의 차이가 0.05 미만인 사례를 거부한다. 처음에 반복 템플릿으로 만든 긴 사례에서 한 토큰이 정확히 동점이었고, CUDA가 mlx CPU와 반대로 동점을 깼다. 그래서 동점이 없도록 긴 사례를 다시 만들었다. 새 경로이고 기존 출력이 바뀌지 않으므로 logit trace는 해당하지 않는다.

## 후속 작업

- 그대로 둔 낮은 등급 리뷰 지적: `--keep` trimming, `-`로 stdin 읽기, `--classify`가 알 수 없는 필드를 받는 점, 입력의 special-token 텍스트가 special token으로 토큰화되는 점(upstream도 같음), stderr에 escape되지 않은 `model_type`.
- #2387(OpenAI Privacy Filter)과 #2390(GLiNER2.5)이 이 계약 위에 구현된다.

## 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 27 |
| 추가 줄 | 4776 |
| 삭제 줄 | 18 |

| 커밋 | 메시지 |
|--------|---------|
| `0f3b5185` | feat(token-classification): add per-token output module and BERT head |
| `8293d1d1` | feat(cli): wire mlxcel tag and mlxcel extract, add tests and docs |
| `5e71679d` | docs: add mlxcel tag to the README quick tour |
| `11b3829c` | fix(token-classification): annotate the grouped path slice type |
| `c62a6448` | test(token-classification): stop shadowing the path helper |
| `999cadea` | test(token-classification): make the multi-window parity case tie-free |
| `6d308b18` | style(token-classification): apply rustfmt |
| `be088772` | fix(token-classification): bound redaction cost and embedding-table reads |

## 관련 항목

- 에픽 #2391 (토큰 분류와 span 추출).
- #2387(OpenAI Privacy Filter)과 #2390(DeBERTa-v2 기반 GLiNER2.5)이 이 PR 위에 구현된다.
