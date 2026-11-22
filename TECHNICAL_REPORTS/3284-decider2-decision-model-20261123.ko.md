# PR #3284: Decider-2b (decider2) 결정 모델 추가

**작성일**: 2026-11-23
**상태**: PR 열림 (draft, #2359 종료 예정, #2375의 일부). 단위, 로더, 실제 체크포인트 parity, 요청 한도 테스트 통과. `models/decider-2b`로 end-to-end 확인
**언어**: Rust, Python (fixture 생성기)
**위험도**: 낮음 (새 결정 모델 계열 추가. Qwen3.5 hidden-state 경로에 공유 변경 하나가 있으나 생성 경로는 건드리지 않음)

---

## 요약

Decider-2b(`model_type: decider2`, `nativ-community/decider-2b`)는 Qwen3.5-2B 파인튜닝 모델이다. 상태(state)에 대한 타입 있는 질문(`choice`, `score`, `bool`/`noul`)에 답할 때 토큰을 생성하지 않고, 열린 답 슬롯 위치의 라벨 토큰 logit만 읽는다. 이 PR은 mlx-vlm `decider2`의 기본 independent 모드를 포팅해 #2343 결정 API 뒤에 붙는 실사용 계열로 만든다(이 PR이 열려 있는 동안 LiquidAI d1-3B, #3282가 먼저 머지되어 또 하나의 계열이 되었다). 이제 `mlxcel decide -m models/decider-2b`와 `mlxcel-server --decision-model models/decider-2b`(`POST /v1/decisions`)로 서빙된다. 실제 체크포인트 기준으로 짧은 요청은 mlx-vlm과 1.4e-3, 긴 JSON 상태는 1.48e-2 이내로 일치한다. 양쪽을 f32로 돌리면 소수점 4자리까지 0.0000으로 같으므로, 남은 bf16 차이는 포팅이 아니라 backbone 연산에서 생긴다. 리뷰 과정에서 요청 하나가 단일 결정 worker에 prefill을 무제한으로 쌓지 못하도록 요청별 prompt 토큰 예산을 추가했다.

## 1. 문제 정의

### 1.1 배경

#2343은 결정 계층(`src/decision/`, `DecisionModel` trait, `mlxcel decide`, `POST /v1/decisions`, `--decision-model`)을 추가했지만 그 뒤에는 `#[cfg(test)]` stub만 있었다. 프로덕션 빌드에서는 `DecisionKind`에 variant가 없어서 모든 로드가 "<path> is not a decision checkpoint"로 실패했다. #2343에는 실제 체크포인트로 end-to-end를 확인하는 항목이 남아 있었고, Decider-2b(#2359, 결정 에픽 #2375 소속)가 그 첫 대상이었다.

### 1.2 기존 문제점

- **감지 불가**: `src/models/detection.rs`에 `decider2` arm이 없어서 `get_model_type`이 `Unsupported model type: decider2`로 실패했고, 결정 로더도 dispatch할 대상이 없었다.
- **다음 계열과 공유할 hidden-state 경로 부재**: `Qwen35Model::prefill_hidden`(#3107 MOSS-Image에서 추가)은 토큰 임베딩과 layer loop를 한 함수에서 처리했다. #2365 Clef는 같은 loop를 직접 만든 `inputs_embeds`와 명시적인 MRoPE position으로 시작해야 한다. 분리하지 않으면 `forward_prefill_with_last_hidden` 옆에 세 번째 loop 복사본이 생긴다.
- **로드 시 f16 캐스트**: Apple Silicon에서 `load_text_weights`는 bf16 체크포인트를 f16으로 바꾼다. 출력이 보정된 확률인 모델에서는 이 캐스트가 속도 옵션이 아니라 수치를 바꾸는 변경이다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| prompt 렌더링이 upstream과 토큰 하나만 달라도 모든 확률이 이동 | 높음 | 토큰 단위 fixture가 없으면 중간 |
| 요청 증폭: 질문마다, isolated score level마다 상태가 반복되므로 작은 본문으로 수십억 prefill 토큰을 쌓을 수 있음 | 높음 (worker 정체, 메모리) | 공개 엔드포인트에서 중간 |
| Qwen3.5 리팩터가 생성 출력을 바꿈 | 높음 | 낮음 (기계적 분리, position은 기존처럼 `None`) |

## 2. 기술적 검토 사항

### 2.1 보안 관점

보안 리뷰(PR 코멘트, `e6ee24b0`에서 수정)의 발견 사항:

| 문제 | 심각도 | 상태 |
|------|--------|------|
| 요청이 제어하는 prefill 증폭. 상태만 `max_state_tokens`로 제한되었고 instruction과 option 텍스트는 제한이 없었다. 32k 토큰 상태에 짧은 질문을 2 MB 채우면 수십억 prefill 토큰과 수 GB의 prompt id가 쌓일 수 있었다 | 높음 | 수정: prompt 하나가 `max_position_embeddings`(262144, `config.json`에 없을 때의 기본값도 같음)를 넘거나 전체 합이 `MAX_REQUEST_TOKENS = 1 << 20`(1048576)을 넘으면 400. 어느 한도든 넘는 첫 row에서 인코딩을 멈춤 |
| row마다 상태를 다시 토큰화 | 중간 | 수정: `build_prompts`가 요청당 한 번 인코딩하고 id를 공유 |
| `decision_config` temperature가 0, 음수, non-finite이면 매 요청이 non-finite 답(500)을 냄 | 낮음 | 수정: 로드 시 거부 |
| group은 패딩 전 길이로 묶이므로 64 패딩 후 8192 토큰을 약간 넘을 수 있음 | 낮음 | 유지 (upstream과 동일) |
| 504 timeout이 worker에서 실행 중인 예측을 취소하지 않음 | 낮음 | 열림, 공유 #2343 worker 소관 |

`const _: () = assert!(MAX_REQUEST_TOKENS >= 32 * 32768);`로 예산이 32768 토큰 상태 위의 질문 32개를 항상 수용하도록 고정했다. 중첩 깊이는 serde_json의 128단계 제한으로 막히고, option, level, label 개수는 인덱싱 전에 검증된다.

### 2.2 성능 관점

row는 배치로 처리된다. 연속된 row를 `rows_in_group * longest <= 8192`(`GROUP_TOKEN_BUDGET`)인 동안 한 group으로 묶고, group마다 64의 배수(`PAD_MULTIPLE`)로 오른쪽 패딩한다. group당 forward 한 번으로 `[rows, length, hidden]`을 얻고, 각 row의 마지막 실제 위치 hidden에 `[hidden, 255]` 라벨 행렬을 곱한다. 전체 vocabulary에 대한 LM head는 계산하지 않는다. 처리량은 측정하지 않았다.

### 2.3 호환성/의존성 관점

- **Breaking change**: 없음. `mlxcel generate -m models/decider-2b`는 이제 "Unsupported model type" 대신 "decider2 is a decision model; use `mlxcel decide` or `mlxcel-server --decision-model`"로 실패한다.
- **새 의존성**: 없음.
- **Qwen3.5 회귀**: `mlxcel generate -m models/qwen3.5-4b-4bit -n 64 --temp 0 --show-reasoning`의 출력 텍스트가 변경 전후 동일했다.

## 3. 기술적 선택과 그 이유

### 3.1 tied embedding row에서 라벨 logit 계산

**컨텍스트:** 체크포인트는 `tie_word_embeddings: true`이고, 답은 라벨 토큰 logit에만 softmax를 적용한 값이다.

**선택:** 로드 시 `A`..`Z`, 이어서 `AA`..`ZZ` 중 특수 토큰 없이 인코딩했을 때 정확히 토큰 하나가 되는 이름을 순서대로 골라 앞의 255개(`LABEL_COUNT`)를 쓴다. 255개가 안 되면 "The decider tokenizer does not provide 255 distinct label tokens"로 실패한다. 해당 `embed_tokens` row를 한 번 모아 `[hidden, 255]`로 transpose하고 eval해 둔다. group마다 logit은 bf16 `matmul(slots, label_weights_t)` 후 f32로 읽는데, upstream과 같은 순서(bf16 곱셈 후 캐스트)다.

**이유:** 라벨로 제한한 tied head는 전체 head와 같은 logit을 주면서 열 수가 vocabulary 전체가 아니라 255다. `embed_tokens.forward`를 거치므로 임베딩이 양자화되어 있어도 row가 dequantize된다.

### 3.2 independent 모드 row와 isolated score level

**컨텍스트:** upstream에는 independent 모드(질문당 prompt 하나)와 packed 모드가 있고, Python keyword 인자로 주는 `isolated` override가 있다.

**선택:** independent 모드만 포팅했다. `plan_rows`는 질문당 row 하나를 만들되, `isolated_levels: true`(이 체크포인트의 설정)이면 `score` 질문을 level마다 `no`/`yes` row 하나로 펼친다. instruction은 `"{instruction}\nProposed answer: {level}\nDoes the proposed answer fit?"`(level 앞의 `"<int>:"`는 제거)이고 temperature는 score용을 쓴다. 답은 level별 `yes` 확률을 그 합으로 정규화하고(합이 0이면 `1e-9`로 나눔) metadata에 `level_fit`, `fit_mass`를 넣는다.

**트레이드오프:** packed 모드와 호출별 override는 API로 도달할 수 없어서 이슈에 적힌 대로 범위 밖이다. isolated level은 row 수를 level 수(최대 10)만큼 곱한다. 2.1의 예산이 질문 수가 아니라 row의 토큰 합을 세는 이유 중 하나다.

### 3.3 `prefill_hidden`을 `hidden_from_embeddings`로 분리

**컨텍스트:** scorer는 새 cache에서 head 없이 post-norm hidden을 받아야 하고, 이는 `forward_prefill_hidden`이 이미 제공한다. #2365 Clef는 같은 loop를 `inputs_embeds`와 MRoPE position으로 시작해야 한다.

| 대안 | 장점 | 단점 |
|------|------|------|
| A: `forward_prefill_hidden` 그대로 사용 | Qwen3.5 변경 없음 | Clef가 layer loop를 세 번째로 복사해야 함 |
| B: Decider 전용 forward 추가 | 격리됨 | lockstep으로 관리할 복사본이 또 생김 |
| **선택: 임베딩 lookup과 `hidden_from_embeddings(h, caches, position_ids)`로 분리** | 모든 hidden-state entry point가 loop 하나를 공유하고, Clef는 entry point만 추가 | 공유 모델 파일 수정 |

`prefill_hidden`은 `embed_tokens.forward(input_ids)` 후 `hidden_from_embeddings(h, caches, None)`을 호출한다. 동작에 영향을 주는 줄 변경은 `layer.forward(&h, mask, cache, position_ids)` 하나이고, 여기에 기존과 똑같이 `None`이 들어간다. `forward_internal*`과 모든 생성 경로는 그대로다. doc comment에는 scorer가 의존하는 성질을 적었다. `[B, L]` batch의 row는 서로 독립이고 causal이므로, 오른쪽 패딩된 row의 마지막 실제 위치 hidden은 한 row만 넣은 호출과 같다.

### 3.4 로드 시 bf16 유지

**컨텍스트:** Apple Silicon에서 `load_text_weights`는 bf16 weight를 f16으로 캐스트한다. Decider-2b의 타입별 temperature(`choice 1.164`, `noul 1.624`, `score 1.124`)는 레퍼런스가 실행하는 bf16 weight에서 맞춘 값이다.

**선택:** `limite`와 같은 제외 목록에 `is_decider2`를 추가해 bf16을 유지한다. f16 캐스트 상태에서는 isolated score level 하나가 mlx-vlm 대비 2.1e-3 움직여 parity 기준 2e-3을 넘었다. `bd0ac7cb`의 sanitize 테스트가 이를 고정한다.

**트레이드오프:** 일부 Apple GPU에서 bf16 matmul이 f16보다 느릴 수 있지만, 보정된 출력의 정확도를 우선했다. 속도 차이는 측정하지 않았다.

### 3.5 parity 기준과 그 의미

fixture `tests/fixtures/decision/decider2_parity.json`은 mlx-vlm d998a620으로 `tests/fixtures/generate_decider2_fixture.py`가 만들었다. prompt token id는 3-option choice, 12-option choice(10개 초과 시 label id splice 경로), score option, isolated score row, `_index`가 붙는 8항목 리스트 상태에서 upstream과 정확히 같다.

- ticket 요청(choice, isolated 4-level score, bool): 최대 확률 차이 1.4e-3, 기준 `TOLERANCE = 2e-3`.
- 약 400토큰 JSON 상태(12-option choice, bool): 최대 차이 1.48e-2, `LONG_TOLERANCE = 2e-2`로 검사하며 argmax 라벨과 bool 값은 일치해야 한다.

2e-2는 느슨해 보여서 따로 확인했다. 체크포인트를 f32로 캐스트하면 mlx-vlm과 이 포팅은 두 요청의 모든 확률과 metadata 값에서, batch로든 질문 하나씩이든 0.0000으로 일치한다. 따라서 렌더링, 패딩, position, 라벨 weight, temperature, softmax는 정확하고 bf16 차이는 backbone 연산에서 온다. upstream 자신도 bf16에서 자기 f32 결과와 최대 7.2e-3 차이 나고, bool 값이 batch일 때와 단독일 때 0.3001과 0.2921로 달라진다. usage 수치는 정확히 같다.

## 4. 구현 상세

| 파일 | 역할 |
|------|------|
| `src/decision/decider2.rs` | `Decider2Model`: 로드(config, `decision_config`, prompter, pad id, backbone, 라벨 행렬), 두 한도를 적용하는 `build_prompts`, `score_rows`/`group_logits`, `DecisionModel::predict` |
| `src/decision/decider2_prompt.rs` | 라벨 선택, `render_question`, `annotate`(8개 이상 리스트에 `_index`), float 표기까지 맞춘 Python `json.dumps(ensure_ascii=False)` 직렬화, 10개 초과 option splice를 포함한 prompt 조립 |
| `src/decision/decider2_rows.rs` | row plan, grouping, 패딩, 보정 softmax, upstream 반올림과 metadata(`confidence`, `certainty`, `legend`, `level_fit`, `fit_mass`) |
| `src/decision/loader.rs` | `DecisionKind::Decider2`와 constructor arm |
| `src/models/qwen3_5.rs` | `hidden_from_embeddings` 분리 |
| `src/models/sanitize.rs` | `decider2`의 bf16 유지 |
| `src/models/detection.rs` | `generate`를 결정 entry point로 안내하는 `decider2` arm |
| `docs/decisions.md`, `docs/supported-models.md`, README, CHANGELOG | 계열 표, prompt 형식, 한도 |

요청 흐름: `predict`가 상태와 질문을 렌더링하고, `plan_rows`가 isolated score를 펼치고, `build_prompts`가 인코딩하며 예산을 적용하고, `score_rows`가 group마다 backbone forward를 한 번 돌리고, `answer`가 각 질문의 row 구간에서 타입별 답을 만든다.

## 5. 학습 포인트

### 5.1 제한된 head로 분류하기

causal LM은 생성 없이 보정된 분류기로 쓸 수 있다. option을 단일 토큰 라벨 뒤에 두고, prompt를 열린 슬롯(`"\nAnswer: ("`)으로 끝내고, 그 위치에서 라벨 logit에만 맞춘 temperature로 softmax를 적용한다. tied embedding이면 제한된 head는 임베딩 row를 모은 것이고 로드 시 한 번만 계산하면 된다.

### 5.2 parity는 tokenizer 경계에서 갈린다

option이 10개를 넘으면 `K` 같은 라벨이 뒤의 `)`나 앞 텍스트와 BPE로 합쳐질 수 있다. upstream은 `"\n("`을 인코딩하고 라벨 id를 직접 붙인 뒤 `") {option}"`을 인코딩한다. 이 방식과 상태용 Python `json.dumps` 출력의 바이트 일치를 fixture에서 token id 단위로 먼저 확인한 뒤에 확률을 비교했다.

### 5.3 포팅 오류와 연산 잡음을 분리하기

bf16 parity 차이가 크면 양쪽을 f32로 돌린다. f32에서 0.0000이면 포팅은 정확하고 bf16 차이는 backbone에서 온다는 증거가 된다. upstream bf16과 upstream f32의 차이가 잡음 하한이고, 기준은 그에 맞춰 잡는다.

### 5.4 입력이 아니라 펼쳐진 작업량에 예산을 건다

증폭은 잘못된 양을 제한해서 생겼다. 상태는 제한되어 있었지만 row가 그것을 곱했다. 이제 한도는 펼친 뒤의 prompt 토큰 합에 걸리고, 예산을 넘는 순간 인코딩을 멈추므로 거부되는 요청도 메모리에 예산 이상을 들고 있지 않는다.

## 6. 추가 학습 리소스

### 관련 PR/이슈

- Issue #2359: Decider-2b 지원 (이 PR로 종료)
- Issue #2343: 결정 API. 이 PR이 남아 있던 실제 체크포인트 end-to-end 확인을 마무리
- Issue #2375: 결정 모델 에픽
- Issue #2365: Clef. `inputs_embeds`와 MRoPE position으로 `hidden_from_embeddings`를 쓸 예정
- PR #3107: `forward_prefill_hidden`을 도입한 MOSS-Image

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 20 |
| 추가 줄 | +3213 (그중 1178은 parity fixture JSON) |
| 삭제 줄 | -16 |
| 추가 테스트 | `#[test]` 함수 19개 (decider2 단위와 parity, loader, sanitize, detection) |

### 카테고리별 변경

| 카테고리 | 요약 |
|----------|------|
| 기능 | decider2 scorer, prompt 렌더링, row plan, loader와 detection arm |
| 공유 backbone | Qwen3.5 `hidden_from_embeddings` 분리 |
| 수치 | `decider2` 로드 시 bf16 유지 |
| 보안 | 요청별, prompt별 토큰 한도, 상태 1회 인코딩, 로드 시 temperature 검증 |
| 문서 | `docs/decisions.md`, `docs/supported-models.md`, README, CHANGELOG |

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|--------|
| `a93e2d17` | feat | feat(decision): add Decider-2b (decider2) decision model |
| `be6f25eb` | merge | Merge origin/main into feature/issue-2359-decider2-decision-model |
| `8bc6b0b5` | docs | docs(decision): add Decider-2b changelog entry and f32 parity evidence |
| `e6ee24b0` | fix | fix(decision): bound Decider-2b prompt tokens per request |
| `bd0ac7cb` | test | test(decision): pin Decider-2b bf16 load and sync docs with the final code |

## 8. 검증 결과와 미검증 항목

### 검증됨

- `mlxcel decide -m models/decider-2b`에 상태 "Please refund my duplicate charge"와 라벨 3개짜리 department 질문을 주면 `billing`, 확률 `billing 0.9198, technical 0.0506, sales 0.0296`, `confidence 0.9198`, `certainty 0.6977`, `input_tokens 37`을 출력했다.
- `mlxcel-server -m models/qwen3.5-4b-4bit --decision-model models/decider-2b`: 같은 요청을 `POST /v1/decisions`로 보내면 HTTP 200과 동일한 답을 받았다. `/v1/models`는 `decider-2b`를 capability `decision`으로 나열하고, 11-level score는 400 `score needs 2 to 10 levels`를 반환했다.
- `mlxcel generate -m models/decider-2b`는 결정 모델 안내 메시지와 함께 exit 1로 끝난다.
- 테스트: `decision::`(단위, 로더, parity, 요청 한도), `models::sanitize_tests::decider2`, `models::detection_tests`, `models::qwen3_5`, `imagegen::moss_image`, `coverage_tests`. `-D warnings` clippy, `cargo fmt --check`, `scripts/ci` contract 검사, 라이선스 헤더.

### 미검증 또는 남은 항목

- 긴 prompt의 bf16 drift(1.48e-2)는 줄이지 않았다. 줄이려면 생성과 공유하는 Qwen3.5 kernel을 바꿔야 하므로 이 PR 범위 밖이다.
- 504 timeout이 실행 중인 예측을 취소하지 않는다. 공유 #2343 worker가 계속 계산한다.
- packed 모드, 호출별 `isolated` override, 31B 형제 체크포인트는 범위 밖이다.
- Qwen3.5 회귀 확인의 "변경 전" 바이너리는 origin/main을 새로 빌드한 것이 아니라 11월 19일 release 빌드였다. 따라서 이 비교는 이 PR과 함께 11월 19일부터 merge base 사이에 main에 들어간 변경까지 포함한다.
- scorer의 처리량과 지연, 3.4의 f16 대비 bf16 속도 비용은 측정하지 않았다.
