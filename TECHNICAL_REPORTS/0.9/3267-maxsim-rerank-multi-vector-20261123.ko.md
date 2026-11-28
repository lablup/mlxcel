# 기술 보고서: PR #3267 - 다중 벡터 임베딩 모델로 `/v1/rerank`를 MaxSim 점수화

**날짜**: 2026-11-23
**상태**: 완료
**위험도**: 낮음 (서빙 중인 임베딩 모델이 다중 벡터이고 reranker가 설정되지 않았을 때만 동작한다. 확률을 내는 reranker 종류는 기존 검사를 그대로 유지한다)

## 요약

서빙 중인 임베딩 모델이 다중 벡터(LFM2.5-ColBERT, ColIdefics3, ColQwen2.5)이고 `--reranker-model`이 설정되지 않았으면, `POST /v1/rerank`와 그 별칭(`/rerank`, `/v1/reranking`, `/reranking`)이 이제 late-interaction MaxSim으로 문서를 점수화한다(#1426, #1414와 에픽 #1348의 후속). 새 라우트나 새 응답 스키마는 없다. 새 `RerankerKind::LateInteraction`(`/props`에서 `late_interaction`)은 확률이 아닌 MaxSim 합 원값을 보고한다. M3 Ultra에서 LFM2.5-ColBERT는 관련 대 무관 문서에 6.4645 대 5.6824를 내며, bf16과 HF 레이아웃 모두 `mlxcel embed`와 같다. adapter를 병합한 colSmol-256M은 렌더링한 두 페이지에서 19.1788 대 8.5367을 낸다.

## 1. 문제 정의

#1414가 `crate::embeddings::maxsim`을 추가했고 `mlxcel embed`는 col* 임베딩 모델에 대해 MaxSim 행렬을 출력했다. HTTP에는 점수화 엔드포인트가 없었다. "어느 문서가 이 쿼리에 답하는가"를 알고 싶은 클라이언트는 `/v1/embeddings`에서 모든 문서의 `[num_tokens, 128]` 행렬(두 페이지 ColIdefics3 예시에서 1776행)을 받아 MaxSim을 직접 계산해야 했다. #3245가 reranker로 쓸 만큼 가벼운 텍스트 전용 다중 벡터 임베딩 모델인 LFM2.5-ColBERT를 추가하면서 이 빈틈이 더 드러났다.

## 2. 변경 요약

- `src/server/rerank_maxsim.rs`(신규): `MaxSimRerankProvider`가 기존 `EmbeddingModelProvider` 위에서 `RerankModelProvider`를 구현한다. 쿼리와 문서를 임베딩한 뒤 문서마다 `maxsim(query_rows, doc_rows)`로 점수를 매긴다. `usage.prompt_tokens`는 임베딩 응답의 합이고, 서버의 `--embd-normalize`를 적용해 점수화하는 행이 `/v1/embeddings`가 돌려주는 행과 같다. 오류는 큐 가득 참이 503, 타임아웃이 504, 잘못된 입력이 400, 그 밖의 오류가 500으로 매핑된다.
- 시작 단계(`resolve_late_interaction_reranker`, `startup.rs`에서 `resolve_rerank_provider` 다음에 호출): reranker가 설정되지 않은 다중 벡터 임베딩 모델에만 provider를 설치한다. 설정된 reranker는 로드에 실패해도 우선하고, 그 경우 로그가 `/v1/rerank`가 501을 돌려준다고 경고한다. 임베딩 워커를 공유하므로 두 번째 `SideWorkerExit`를 등록하지 않는다. `check_serving_mode`가 이를 세므로 다중 벡터 `-m`에서도 `--reranking`이 동작한다.
- `src/rerank/mod.rs`: `LateInteraction` 종류. `accepts_instruction()`은 false이고, 새 `scores_are_probabilities()`는 이 종류에서만 false다. 이 종류는 reranker 체크포인트에서 나오지 않으므로 reranker 로더가 거부한다.
- `src/server/routes/rerank.rs`: 점수 검사가 `score_is_valid(score, probabilities)`로 바뀌었다. 모든 종류는 유한한 점수를 내야 하고, `[0, 1]` 범위는 확률 종류에만 요구한다. `instruction` 400 메시지는 로드된 종류를 밝히고, 501 메시지는 다중 벡터 임베딩 모델을 언급한다.
- 문서: `docs/embeddings.md`의 "Late-interaction scoring" 절이 원점수 규모, LFM2.5-ColBERT의 호출자 쪽 `[Q] ` / `[D] ` 접두어, colSmol의 adapter 병합을 다룬다. rerank 오류 표를 갱신했고, `docs/supported-models.md`의 reranker 표에 행을 추가했다.
- 테스트: stub 임베딩 모델을 쓰는 라우트 테스트가 알려진 행의 `maxsim`과의 일치, `top_n`, `return_documents`, TEI `texts` envelope, instruction 400, 오류 매핑, 종류별 점수 범위, 문서 500개의 단일 호출, 개수가 틀린 응답, NaN 행을 확인한다. 실제 체크포인트 게이트는 LFM2.5-ColBERT(bf16, HF)와 병합한 colSmol-256M을 다룬다. `tests/fixtures/generate_colsmol_rerank_fixture.py`가 LoRA adapter를 병합하고(`W + (alpha/r) B A`와 학습된 `1_Dense`) 두 페이지를 렌더링한다.

## 3. 기술적 선택과 그 이유

**`/v1/score`를 따로 두지 않고 `/v1/rerank`를 확장.** 요청 검증, 이미지 가져오기, `top_n`, `return_documents`, 두 응답 envelope가 이미 `RerankModelProvider` 뒤에 있다. provider와 종류 하나를 추가하면 이를 모두 재사용하고, 라우트에는 종류를 아는 메시지와 범위 검사만 필요했다.

**정규화하지 않은 MaxSim 합 원값.** 점수는 쿼리 토큰 수에 따라 커진다. 정규화하면 차이가 가려지고 CLI 수치와도 맞지 않게 된다. `RerankResult.relevance_score` 문서 주석과 문서에 이를 적었다. 라우트의 `[0, 1]` 검사를 종류별로 바꿔야 했던 이유도 이것이다. 바꾸기 전에는 모든 MaxSim 점수(예: 19.18)가 500을 돌려주었을 것이다.

**요청당 배치 텍스트 호출 하나(보안 리뷰).** 첫 리비전은 문서마다 `embed_texts`를 한 번씩 호출했다. 워커의 응답 타임아웃은 호출마다 적용되고, `/v1/rerank`는 문서 수를 본문 크기로만 제한한다. 그래서 짧은 문서 수천 개를 담은 요청 하나가 공유 임베딩 워커를 타임아웃 수천 번 동안 연달아 붙잡을 수 있었고, 앞쪽 forward를 이미 돌린 뒤 중간에 `QueueFull`을 만날 수도 있었다. 이제 모든 텍스트 항목을 `/v1/embeddings`처럼 한 번에 보낸다. 큐 슬롯 하나, 타임아웃 하나, 길이순 마이크로 배치다. 이미지 항목은 기존 요청당 이미지 상한 안에서 하나씩 호출한다. 부수 효과로 LFM2.5 점수가 이제 CLI와 정확히 같다. 항목별 리비전은 bf16 배치 prefill의 누적 방식 차이로 8e-4만큼 달랐다.

**설정된 reranker가 항상 우선.** 명시한 reranker가 로드에 실패했을 때 MaxSim이 조용히 대신하면, 운영자는 요청한 것과 다른 점수 모델을 받게 된다. 그래서 실패한 경우에는 501을 돌려주고 시작 로그가 이유를 밝힌다.

## 4. 측정

M3 Ultra, Metal, release `mlxcel-server`와 체크포인트 게이트.

| 체크포인트 | 관련 | 무관 | `mlxcel embed` MaxSim | prompt_tokens |
|---|---|---|---|---|
| lfm2.5-colbert-350m-bf16 | 6.4645 | 5.6824 | 6.4645 / 5.6824 | 33 |
| lfm2.5-colbert-350m (HF 레이아웃) | 6.4645 | 5.6824 | 6.4645 / 5.6824 | 33 |
| 병합한 colSmol-256M, PR #1414 쿼리 | 19.1788 | 8.5367 | 19.1788 (이미지별 경로) | 1776 |

## 5. 측정하지 않은 것

- PR #1414의 페이지 이미지는 커밋된 적이 없어 18.7396 대 8.6879를 재현할 수 없다. 게이트는 대신 렌더링한 페이지로 순서, 10% 차이, embed 경로와의 일치를 확인한다. prompt_tokens 1776이 #1414의 행 합계와 같으므로 페이지 기하는 같다.
- ColQwen2.5는 `/v1/rerank`에서 돌리지 않았다. ColIdefics3와 같은 provider 경로를 탄다.
- rerank와 embeddings 동시 부하에서의 처리량은 측정하지 않았다.

## 6. 학습 포인트

- 공유 워커 위의 항목별 루프는 호출당 제한된 타임아웃을 요청당 무제한 점유로 바꾼다. 요청 경계에서 배치하거나 항목 수에 상한을 둬야 한다.
- 한 점수 도메인(확률)을 위해 쓴 유효성 검사는 새 종류가 같은 라우트에 합류하면 틀리게 된다. 그 검사는 라우트가 아니라 종류에 속해야 한다.

## 7. 후속 작업

- `/v1/rerank`에는 본문 크기 외에 문서 수나 총 토큰 상한이 어느 reranker 종류에도 없다. 공통 상한이 있으면 모든 종류의 최악 요청 비용이 제한된다.
- `maxsim`은 NaN 행을 건너뛰는 `f32::max`를 쓴다. 일부 행만 NaN인 문서는 유한한 점수를 받고, 모든 행이 NaN인 문서만 `-inf`로 거부된다.
- 드러났지만 범위 밖: 다중 벡터 임베딩 모델을 받는 `mlxcel rerank`, 보조 임베딩 모델의 `/v1/models` 항목에 `rerank` capability 추가.
