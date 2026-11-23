# PR #3309: `mlxcel rerank`에서 multi-vector 임베더로 리랭킹

**날짜**: 2026-11-23
**상태**: PR 열림 (#3300 해결). 단위 테스트와 실제 체크포인트 테스트 통과. M3 Ultra에서 LFM2.5-ColBERT-350M과 병합된 colSmol-256M 모두 CLI 점수와 `/v1/rerank` 점수가 비트 단위로 일치

## 요약

#3267 이후 `mlxcel-server -m <multi-vector 임베더>`는 리랭커가 따로 설정되지 않으면 `/v1/rerank`를 MaxSim으로 제공한다. 그런데 같은 체크포인트에 `mlxcel rerank -m`을 쓰면 "is not a reranker checkpoint"로 실패했다. CLI 로더가 cross-encoder와 Qwen3 생성형 두 종류만 알았기 때문이다. 결과적으로 서버와 오프라인 검증 도구가 리랭킹 가능한 체크포인트를 서로 다르게 판단했다.

이 PR은 서버 `MaxSimRerankProvider`의 점수 계산 부분을 공용 라이브러리 함수 `rerank::score_late_interaction`으로 옮기고 양쪽에서 이를 호출한다. 서버는 임베딩 워커 위에서, CLI는 호출 스레드의 `EmbeddingEngine` 위에서 실행한다. 이제 `mlxcel rerank`는 리랭커가 아닌 체크포인트를 임베더로 로드하고, multi-vector이면 `kind: "late_interaction"`으로 MaxSim 점수를 낸다. 그 밖의 체크포인트는 기존 오류를 그대로 낸다.

## 1. 문제

`src/commands/rerank.rs`는 `load_reranker_with_options`를 호출했고, 그 안의 `detect_reranker_kind`는 BERT, XLM-RoBERTa, ModernBERT의 `ForSequenceClassification` 헤드와 `qwen3`, `qwen3_vl`만 받는다. LFM2.5-ColBERT(`model_type: lfm2`), ColIdefics3(`idefics3`), ColQwen2.5는 `Unsupported reranker model type`으로 떨어졌다. 서버는 별도의 시작 분기(`resolve_late_interaction_reranker`)로 이 체크포인트를 처리했기 때문에, "이 체크포인트로 리랭킹할 수 있다"는 판단이 한 곳에 있지 않았다. 이 종류의 `/v1/rerank` 결과를 CLI로 오프라인 확인할 수도 없었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/rerank/late_interaction.rs` (신규) | `MultiVectorEmbedder` 트레이트(`supports_images`, `embed_texts`, `embed_image`), 서버 provider에서 옮겨 온 `score_late_interaction`, `LateInteractionError`, 그리고 `EmbeddingEngine` 위의 `Reranker`인 `LateInteractionReranker`. `EmbeddingEngine`이 트레이트를 구현한다. |
| `src/server/rerank_maxsim.rs` | `MaxSimRerankProvider::rerank`가 `EmbeddingModelProvider`를 감싼 `WorkerEmbedder` 어댑터를 통해 `score_late_interaction`을 호출한다. `embed_items`와 `single_vector`는 복제하지 않고 옮겼다. 오류 매핑과 메시지는 그대로다. |
| `src/rerank/loader.rs` | `load_reranker_or_late_interaction`, 그리고 테스트하기 쉽게 분리한 `late_interaction_from_embedder`. |
| `src/commands/rerank.rs` | 새 로더를 쓴다. `--instruction` 거부 메시지가 종류를 밝힌다. `-m`, `--batch-size`, `--max-length` 도움말이 임베더 경우를 설명한다. |
| 테스트 | `rerank/late_interaction_tests.rs`(9개)와 `server/rerank_maxsim_checkpoint_tests.rs`의 실제 체크포인트 게이트 2개. |
| 문서 | `docs/embeddings.md`(소스 맵, Late-interaction scoring, `mlxcel rerank`), `docs/supported-models.md`(late-interaction 행). |

## 3. 기술적 결정

### 3.1 작은 트레이트 뒤의 단일 점수 함수

서버 provider가 규칙 전체를 갖고 있었다. 모든 텍스트 항목을 `embed_texts` 한 번으로 보내고, 이미지는 하나씩 따로 임베딩하고, 응답 개수를 확인한 뒤 문서마다 `embeddings::maxsim`을 실행한다. 이 규칙이 필요로 하는 것은 임베더 연산 세 가지뿐이므로, 이제 연관 오류 타입을 가진 `MultiVectorEmbedder`를 받는다. 서버 어댑터는 `EmbeddingError`를 돌려주고 이전과 똑같이 `RerankError`로 매핑한다(queue full은 503, timeout은 504, invalid input은 400, internal은 500). CLI 엔진은 `EmbeddingEngineError`를 돌려준다. CLI 쪽에 루프를 복사하는 대안은 두 경로의 배칭이 어긋날 여지를 남기고, 배칭은 점수를 바꾼다(3.3 참고).

### 3.2 감지 순서와 비용

`load_reranker_or_late_interaction`은 먼저 리랭커 감지를 시도하므로 리랭커 체크포인트는 전과 똑같이 로드된다. 감지가 실패하면 임베딩 로더와 같은 조합인 `get_model_type`과 `is_embedding_model_type`으로 확인한다. 채팅 체크포인트는 `config.json`만으로 이 확인에서 떨어지고, 가중치를 읽지 않은 채 원래 감지 오류를 돌려준다. 임베딩 체크포인트는 로드한 뒤 limits의 multi-vector 플래그를 읽고, single-vector 임베더면 원래 오류로 거부한다. 오류로 끝나는 경로에서 전체 로드 비용이 들지만, 오프라인 도구로서는 감수할 만하고 로드된 모델로 판단하는 서버 동작과도 같다. 임베더 로드 자체가 실패하면, 리랭커가 아니라는 점과 multi-vector 임베더로 로드하는 것도 실패했다는 점을 함께 알려서 실제 원인을 감지 메시지 뒤에 숨기지 않는다.

### 3.3 서버 배칭과 맞추기

길이로 정렬한 마이크로 배칭은 배치 안의 행을 패딩하므로, 배치 구성에 따라 점수가 조금 달라질 수 있다. CLI는 임베딩 배치 폭의 기본값을 리랭커 종류의 기본값 8이 아니라 서버 `--embedding-batch-size` 기본값인 `DEFAULT_EMBEDDING_BATCH_SIZE`(16)로 둔다. 배치 폭과 `embed_texts` 단일 호출이 같으므로 두 경로는 같은 forward를 실행하고, 측정한 점수는 마지막 비트까지 같았다. `--batch-size`로 폭을 바꿀 수 있다. 임베딩 옵션은 기본값이며, `--embd-normalize` 없이 시작한 서버와 같다.

## 4. 검증

| 항목 | 결과 |
|---|---|
| `cargo clippy --lib --tests --bins --features metal,accelerate -- -D warnings` | 경고 없음 |
| `rerank::`, `server::rerank_maxsim` 라이브러리 테스트 (GPU 잠금 아래 직렬 실행) | 84개 통과 |
| 실제 체크포인트 게이트, LFM2.5-ColBERT-350M, CLI 로더 대 서버 provider | 양쪽 모두 `[5.6823516, 6.4644585]`, prompt token 33 |
| 실제 체크포인트 게이트, 병합된 colSmol-256M, 텍스트 쿼리 대 텍스트 문서 하나와 페이지 이미지 두 개 | 양쪽 모두 `[8.537709, 22.494436, 19.177414]`, prompt token 1804 |
| 릴리스 바이너리: `mlxcel rerank --json` 대 `mlxcel-server -m`에 대한 `curl /v1/rerank` | LFM2.5-ColBERT: 문서 세 개 점수 동일(5.6823516, 6.4644585, 6.31933), 41 token. 병합 colSmol: 텍스트 하나와 이미지 두 개 점수 동일(22.494436, 19.177414, 8.537709), 1804 token. `kind`는 `late_interaction` |
| 거부 동작 | colSmol에 `--instruction`: MaxSim을 밝히며 거부. LFM2.5-ColBERT에 `--image`: "text-only". `sentence-transformers/all-MiniLM-L6-v2`(single-vector)와 `k2-horizon-0.9b`(채팅): 둘 다 기존 "is not a reranker checkpoint ... Unsupported reranker model type" 메시지로 실패 |

colSmol 어댑터(`vidore/colSmol-256M`)는 LoRA 전용이라 단독으로 로드할 수 없다. 게이트는 서버 문서가 이미 요구하는 대로 `tests/fixtures/generate_colsmol_rerank_fixture.py`가 만든 병합 디렉터리를 쓴다.

## 5. 한계와 후속 작업

- CLI에는 `--embd-normalize`와 `--pooling`이 없고, 체크포인트 자체의 정규화가 적용된다. pooling은 multi-vector 계열에 영향이 없다.
- single-vector 임베더는 거부되기 전에 전체가 로드된다(3.2).
- 새 게이트는 체크포인트가 없으면 같은 파일의 기존 게이트처럼 건너뛴다.
