# PR #3247: 타입이 있는 decision 모델 API (/v1/decisions, --decision-model, mlxcel decide)

**날짜**: 2026-11-23
**상태**: Draft PR, 머지 전 (#2343 해결, #2375의 일부). 단위 테스트와 M3 Ultra에서 검증했으며 실제 decision 체크포인트로는 검증하지 않음

## 요약

mlx-vlm은 생성된 텍스트 대신 보정된 확률로 상태(state)에 대한 이름 붙은 질문에 답하는 decision 모델 계열(Decider-2b, Laya, Clef, LiquidAI d1-3B, d1-omni-600M)을 추가했습니다. 이 모델들은 CLI, HTTP 라우트, 서버 플래그, 그리고 모델별로 읽는 질문 타입과 미디어를 선언하는 방식을 하나의 API로 공유합니다. mlxcel에는 구조화된 타입 있는 답을 돌려주는 경로가 없어서, 가중치를 로드할 수 있게 되더라도 이 체크포인트들을 서빙할 수 없었습니다.

이 PR은 그 경로를 한 번에 만듭니다. `src/decision`에는 질문과 답 타입, `DecisionModel` trait, 공통 검증(upstream 메시지를 그대로 쓴 규칙 1~8), 답 생성자, kind 디스패처, CLI 입력 준비가 들어 있습니다. 서버에는 전용 워커 위의 `POST /v1/decisions`와 `/decisions`, `mlxcel-server`와 `mlxcel serve`의 사이드 모델 플래그 `--decision-model` / `MLXCEL_DECISION_MODEL`이 추가되고, `mlxcel decide`는 같은 JSON을 오프라인으로 출력합니다. 현재 `DecisionKind`는 `#[cfg(test)]` stub 하나뿐이므로, 첫 모델 포팅(#2359, #2365, #2367)이 variant, 감지 분기, scorer를 추가하기 전까지 실제 로드는 모두 `<path> is not a decision checkpoint`로 실패합니다.

## 1. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/decision/mod.rs` | `QuestionKind`(`noul`은 `Bool`로 역직렬화), `MediaKind`, `Question`, `Criteria`, `DecisionMedia`, `DecisionResult`, `DecisionError`, `DecisionModel` trait, `predict`, `check_result`. |
| `src/decision/validate.rs` | `validate_questions`(규칙 1~8)와 `validate_question_structure`(규칙 4만, CLI가 모델 로드 전에 사용). |
| `src/decision/answer.rs` | `Answer::choice`, `score`, `bool`, `multi_label`. 모든 모델에서 `confidence`, `legend`, bool `value`를 같은 방식으로 채우고 답 JSON을 만듭니다. |
| `src/decision/loader.rs` | `DecisionKind`, `detect_decision_kind`, `rerank`와 같은 방식으로 HF repo id를 해석하는 `load_decision_model`. |
| `src/decision/stub.rs` | `choice`, `score`, `bool`과 이미지만 선언하는 `#[cfg(test)]` 결정적 모델. |
| `src/decision/cli.rs`, `media.rs` | `mlxcel decide` 입력 준비, CLI 미디어 로딩, 300초 길이 제한이 있는 16 kHz mono 오디오 디코드. |
| `src/server/decision_model.rs`, `decision_worker.rs` | `DecisionModelProvider`와 `DecisionWorkerProvider`: 전용 MLX 스레드, 크기 제한 큐(503), 응답 타임아웃(504), 일반 500을 돌려주는 panic 경계. |
| `src/server/routes/decisions.rs` | `/v1/decisions`와 `/decisions`의 `create_decisions`. 모델이 없으면 `--decision-model`을 명시한 501. |
| 서버 연결 | `cli_input.rs`, `startup.rs`, `state.rs`, `app.rs`, `routes/models.rs`(`decision` capability), `chat_unavailable_message`, `bin/mlx_server.rs`, `commands/serve.rs`. 로드 실패는 시작 시 치명적 오류입니다. |
| `src/commands/decide.rs`, `src/main.rs` | `mlxcel decide` 서브커맨드. |
| 문서 | 새 `docs/decisions.md`(계약, 오류, 플래그, CLI, 소스 맵, "Adding a decision model"), 그리고 `docs/README.md`, `README.md`, `docs/environment-variables.md`. |

`origin/main` 대비 36개 파일, +4175 / -3 줄.

## 2. 기술적 선택과 그 이유

### 2.1 rerank 확장 대신 별도 서브시스템

가장 가까운 기존 경로는 reranking이었습니다. 플래그로 로드되어 전용 워커에서 돌고 자체 라우트로 서빙되는 사이드 모델입니다. 하지만 reranker는 문서마다 `f32` 하나를 돌려주고, decision 모델은 이름 붙은 타입 있는 답의 맵을 돌려주며 답마다 모양이 다릅니다(라벨별 확률이 붙은 argmax 라벨, legend가 붙은 기대 레벨 인덱스, true 확률, 독립 점수가 붙은 threshold 적용 라벨 집합). `RerankerKind` variant를 추가하면 이런 답을 표현할 수 없는 점수 벡터를 쓰거나 rerank 라우트 안에 두 번째 응답 타입을 넣어야 했고, 고정 메시지를 가진 순서 있는 8개 규칙과 모델별 미디어 게이트라는 검증은 rerank에 대응되는 부분이 없습니다.

그래서 이 PR은 새 `decision` 서브시스템에 rerank의 계층 구조(core 모듈, loader, provider trait, worker, route, CLI)를 그대로 따르고, rerank의 타입이 아니라 메커니즘을 재사용합니다: 워커 구조, 501과 오류 envelope 헬퍼, 사이드 모델 시작 경로. 미디어 획득은 `src/server/media.rs`의 chat fetcher를 재사용하므로 URL, 사설 주소, `--media-path`, payload 제한이 chat 요청과 같습니다.

### 2.2 답을 순서 있는 쌍의 `Vec`으로

`DecisionResult::answers`는 `Vec<(String, Answer)>`입니다. upstream은 답을 질문 순서로, `choice` 확률을 라벨 순서로 돌려줍니다. `serde_json`에는 이미 `preserve_order`가 켜져 있어 들어오는 맵은 요청 순서를 유지하고, 쌍의 `Vec`은 `indexmap` 의존성 없이 나가는 쪽에서도 그 순서를 유지합니다.

### 2.3 모델이 미디어를 선언하고 공통 게이트가 강제

각 모델은 읽는 미디어를 `decision_media`에 나열합니다. upstream은 `predict` 시그니처를 검사하던 게이트가 `**kwargs`로 미디어를 넘기는 Clef의 이미지를 거부한 뒤 커밋 be0dbae9에서 이 방식으로 바꿨습니다. 선언 목록은 명시적이므로 공통 검증이 바이트 하나 가져오기 전에 선언되지 않은 미디어를 거부할 수 있습니다(규칙 3). d1-omni의 images-or-audio 배타 규칙이나 Decider의 2~255개 choice 옵션 같은 모델 고유 규칙은 scorer에 남아 `DecisionError::InvalidInput`, 즉 400으로 나갑니다.

### 2.4 `videos`는 `PathBuf`가 아닌 `VideoSource`

이슈는 `DecisionMedia::videos: Vec<PathBuf>`를 지정했습니다. PR은 `VideoSource`를 써서, 비디오 allowlist resolver가 `O_NOFOLLOW`로 연 fd 기반 핸들을 서버가 그대로 넘기게 합니다. 경로를 넘기면 검사와 읽기 사이에 다시 열리거나 바꿔치기될 수 있습니다. `videos` 필드 자체는 mlxcel이 wire 계약에 추가한 것입니다. upstream 라우트는 `images`와 `audio`만 받지만 Python `predict`는 `videos`도 받고, Clef가 이를 선언합니다.

### 2.5 프로덕션 빌드에는 아직 decision 계열이 없음

`DecisionKind`에는 `#[cfg(test)]` `Stub` variant만 있습니다. 테스트가 아닌 빌드는 도달 가능한 분기가 없는 디스패처를 dead-code나 unreachable 경고 없이 컴파일하고, 실제 체크포인트에 `--decision-model`을 주면 시작 시 `<path> is not a decision checkpoint`로 실패합니다. API 경로만 따로 리뷰할 수 있게 하면서, 모델이 없다는 사실을 조용히 넘기지 않고 드러냅니다.

## 3. Wire 계약과 검증 순서

요청(`POST /v1/decisions`, `mlxcel decide`도 같은 값을 만듭니다):

```json
{"model": "optional id", "state": "string | any JSON value | null", "questions": {"<name>": {"type": "choice", "instructions": "...", "criteria": ["a", "b"]}}, "images": ["url, path or data URI"], "videos": ["url or path"], "audio": "url, path or base64 data URI"}
```

규칙은 아래 순서로 실행되며, 실패하면 라우트에서는 `400 invalid_request_error`, CLI에서는 같은 메시지와 0이 아닌 종료 코드입니다.

1. 미디어가 있을 때만 `state`가 `null`이거나 없을 수 있습니다: `state is required unless images or audio are given`.
2. 모델이 질문 타입을 하나 이상 선언합니다: `This model does not support decision prediction`.
3. 들어온 미디어는 모두 선언되어 있어야 합니다: `This model does not support image input`(또는 `video`, `audio`).
4. `questions`는 이름에서 객체로 가는 비어 있지 않은 맵입니다.
5. `type`의 기본값은 `choice`이고, `choice`, `score`, `bool`, `multi_label`(`noul`은 `bool`의 별칭) 중 하나이며 모델이 선언한 것이어야 합니다.
6. criteria 개수(`choice`와 `score`는 2, `multi_label`은 1), 비어 있지 않고 중복 없는 라벨, `score` criteria는 순서 있는 리스트.
7. `multi_label` `threshold`는 [0, 1] 범위의 숫자이며 boolean은 거부.
8. `bool` criteria는 없거나 `false` / `true` 설명 맵.

라우트에서는 요청을 파싱하고, provider 존재 여부(501)와 `model` id를 확인한 뒤, 각 미디어의 존재 여부만 아는 상태에서 워커가 선언한 타입과 미디어로 `validate_questions`를 실행합니다. 미디어는 검증을 통과한 뒤에만 가져옵니다: `validate_image_count`와 chat 이미지 제한 아래의 이미지, 그다음 ffmpeg 확인, allowlist를 거친 비디오 해석, 마지막으로 blocking pool에서의 오디오 fetch와 디코드. 워커가 `predict`를 실행하고, 응답을 만들기 전에 `check_result`가 답의 이름, 순서, kind, 유한성을 확인합니다. 여기서 실패하면 로그에 남는 내부 오류이며 `500 Decision prediction failed`로 돌려줍니다.

응답: `{"model", "answers": {name: answer}, "usage": {"input_tokens", "output_tokens": 0}}`. kind별 답 모양은 `docs/decisions.md`에 있습니다.

## 4. 모델 포팅을 위한 확장 지점

Decider-2b(#2359), Clef(#2365), LiquidAI d1 모델(#2367)은 각각 세 가지만 추가하고 나머지는 건드리지 않습니다.

1. `src/decision/loader.rs`의 `DecisionKind` variant와 `as_str` 분기.
2. `config.json`으로 체크포인트를 알아보는 `detect_decision_kind`의 감지 분기와 `construct`의 생성자 분기.
3. `DecisionModel`을 구현하는 scorer: `decision_types`, `decision_media`, 그리고 검증을 통과한 질문만 요청 순서로 받아 자체 제한을 확인하고 `Answer` 생성자로 답을 만들며 모델 고유 지표는 `metadata`에 넣는 `predict`.

```rust
pub trait DecisionModel: Send {
    fn decision_types(&self) -> &[QuestionKind];
    fn decision_media(&self) -> &[MediaKind];
    fn predict(
        &mut self,
        state: Option<&Value>,
        questions: &[(String, Question)],
        media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionError>;
}
```

워커, 라우트, CLI, `/v1/models` 목록은 새 variant를 수정 없이 받아들입니다. `check_result`는 이 포팅들이 기대는 안전망입니다. 질문을 빼먹거나, 답 순서를 바꾸거나, `choice` 질문에 `bool` 답을 내거나, NaN을 만드는 scorer는 잘못된 응답 대신 500이 됩니다.

## 5. 리뷰 중 추가된 보강

구현 리뷰와 보안 및 성능 리뷰에서 CRITICAL이나 HIGH 항목은 없었습니다. 후속 커밋 두 개가 MEDIUM과 일부 LOW 항목을 처리했습니다.

| 커밋 | 발견 사항 | 수정 |
|---|---|---|
| `0ce2c1c6` | MEDIUM: `mlxcel decide`가 `predict` 안의 검증보다 먼저 모든 `--image`, `--video`, `--audio`를 읽고 디코드했습니다. 그래서 선언되지 않은 미디어를 읽거나 가져왔고, 디코드되지 않는 클립은 `This model does not support audio input` 대신 디코드 오류를 냈습니다. | CLI가 `load_cli_media` 전에, 인자 존재 여부로 로드된 모델의 선언에 대해 `validate_questions`를 실행합니다. 라우트와, 경로를 읽지 않고 `predict`에 넘기는 upstream `decide.py`와 같은 순서입니다. |
| `0ce2c1c6` | LOW: `side_model_id_for_path` doc comment에 `--decision-model`이 없었습니다. | doc comment 수정, CLI 순서를 `docs/decisions.md`에 명시. |
| `3c62601e` | MEDIUM: 라우트 오디오의 제한이 인코딩 크기 500 MiB뿐이라, 긴 저비트레이트 클립이 크기 제한 큐에 닿기 전에 몇 시간 분량의 샘플로 늘어날 수 있었습니다. | `MAX_DECISION_AUDIO_SECONDS = 300`(chat 오디오 계열이 선언한 가장 긴 클립 제한, `AudioFamilyPolicy::inkling`). 리샘플 전에 원래 샘플로 확인합니다. 400 메시지는 `audio: audio is <n> seconds long; decision audio is limited to 300 seconds`. 고정 상수이며 새 환경 변수는 없습니다. |
| `3c62601e` | LOW: `ffmpeg` 사용 가능 여부를 확인하기 전에 비디오를 내려받았습니다. | 어떤 클립도 해석하기 전에 chat과 같은 메시지로 `check_video_support`를 실행합니다. chat에 요청별 비디오 개수 제한이 없으므로 추가하지 않았습니다. |
| `3c62601e` | LOW: `check_result`가 답 이름과 유한성은 확인했지만 답 kind는 확인하지 않았습니다. | 각 답의 kind를 질문의 kind와 비교합니다(`noul`과 `bool`은 `bool` 등). |

## 6. 검증

단위 테스트(`cargo test --profile test-fast --features metal,accelerate`):

- `--lib decision`: 리뷰 시점 45개 통과. 규칙마다 정확한 메시지를 확인하는 테스트, 답 생성, decision 체크포인트가 아닌 경우의 loader 거부, 워커 큐(503)와 타임아웃(504)을 다룹니다. `3c62601e`는 오디오 길이 제한(경계값과 초과), ffmpeg 게이트, 답 kind 확인 테스트를 추가했습니다.
- stub provider를 쓰는 라우트 테스트: `model`, `answers`, `usage`가 있는 200, 모든 400, 플래그를 명시한 501, 503, 504, 500, `choice` 확률의 라벨 순서, `bool`로 돌아오는 `noul`, 미디어 게이트(오디오 거부, 이미지는 `predict`까지 도달), 너무 긴 오디오.
- `--bin mlxcel decide`: 7개 통과(잘못된 질문, state와 미디어 모두 없음, 플래그 충돌).
- `coverage_tests`, `server::cli_input`, `server::routes::models`, `server::routes::availability`.
- `-D warnings`를 건 `cargo clippy --lib --tests`와 `--bins`, `cargo fmt --check`, `scripts/ci` 계약 검사.

실제 하드웨어, M3 Ultra, chat 모델 `models/qwen3.5-4b-4bit`:

| 확인 항목 | 결과 |
|---|---|
| `--decision-model` 없는 서버에 `POST /v1/decisions`, `POST /decisions` | 501, 메시지에 `--decision-model` 명시 |
| `--decision-model models/qwen3.5-4b-4bit`(chat 체크포인트) | 시작 시 `is not a decision checkpoint`와 함께 exit 1 |
| 잘못된 `--questions`로 `mlxcel decide` | 모델 해석이나 로드 전에 실패 |
| state와 미디어 모두 없이 `mlxcel decide` | 모델 해석이나 로드 전에 실패 |
| 올바른 입력으로 `mlxcel decide -m models/qwen3.5-4b-4bit` | `is not a decision checkpoint` 보고 |

포팅된 decision 계열이 아직 없어서 실제 체크포인트의 200 응답은 확인하지 못했습니다.

## 7. 검증되지 않은 항목과 남은 과제

- **실제 decision 체크포인트 없음.** 성공 경로는 stub으로만 테스트했습니다. 처음 머지되는 모델 하위 이슈(#2359, #2365, #2367)가 실제 가중치로 `POST /v1/decisions`와 `mlxcel decide`를 끝까지 검증해야 하며, 이는 #2343의 마지막 미완료 수용 기준이기도 합니다.
- **오디오 디코드는 여전히 큐 앞에서 일어남.** 300초 제한은 긴 클립을 거부하지만 원래 샘플레이트로 디코드한 뒤에 확인하므로, 너무 긴 클립의 원래 샘플 버퍼는 한 번은 할당되고, 동시 요청은 blocking pool에서 동시에 디코드됩니다. 첫 오디오 decision 모델(#2367 / #2368)은 decision용 `AudioFamilyPolicy` 아래에서 디코드하거나 워커 스레드에서 디코드해야 합니다.
- **보안 리뷰에서 남은 LOW 항목.** 현재 오디오나 비디오를 선언한 모델이 없어 어느 것도 도달할 수 없습니다.
  - 미디어 획득(이미지 fetch와 디코드, 개당 최대 1 GiB 비디오 다운로드, 오디오)이 큐 확인 전에 일어나므로 큐가 가득 차도 획득 작업은 제한되지 않습니다. `/v1/rerank`, chat과 같습니다.
  - `videos`에 요청별 개수 제한이 없습니다. chat과 같습니다.
  - `fetch_labeled_image`의 이미지 디코드는 rerank, embeddings와 마찬가지로 Tokio 워커에서 실행됩니다.
  - 질문과 라벨 개수는 JSON body 제한으로만 제한됩니다. 검증은 선형이고, rerank의 문서 목록과 같습니다.
- **구현 리뷰에서 남은 LOW 항목.** 이미지를 지원하지 않는 모델에 `images: []`를 보내면 통과하지만, upstream `predict`는 null이 아닌 리스트가 있으면 거부합니다. upstream 라우트도 빈 리스트를 버리므로 wire 동작은 같습니다.
- **갱신하지 않은 문서.** `CLAUDE.md`는 gitignore되어 저장소에 없고 mkdocs `docs/en` / `docs/ko` 트리도 없어서, 추가할 참조 행, nav 항목, 한국어 미러가 없었습니다.

## 8. 학습 포인트

- 새 출력 모양이 기존 서브시스템과 다르면, 계층 구조를 따라 하고 메커니즘(워커, envelope, fetcher)을 재사용하되 타입은 재사용하지 않습니다. 운영을 일관되게 유지하는 것은 계층 구조이고, 새 모양을 옛 타입에 밀어 넣으면 특수 처리가 곳곳에 퍼집니다.
- 입력을 가져오기 전에 선언에 대해 검증합니다. 어떤 미디어가 들어왔는지만 알아도 선언되지 않은 미디어를 거부할 수 있으므로, 거부된 요청은 다운로드나 디코드 비용이 없습니다. `0ce2c1c6`에서 고친 CLI 버그는 같은 검증을 입력 로딩 뒤에 실행해서 생긴 것입니다.
- 인코딩된 바이트 크기 제한은 디코드된 크기를 제한하지 못합니다. 압축 오디오에는 디코드된 샘플에 대한 길이 제한이 필요합니다.
- 모델 코드와 응답 코드 사이 경계의 공통 결과 검사(`check_result`)는 scorer 버그를 잘못된 응답이 아니라 로그에 남는 500으로 바꿉니다. 같은 trait에 대해 여러 모델 포팅이 작성될 때 특히 중요합니다.

## 관련 이슈

- #2343: 이 이슈. #2375: 상위 epic.
- #2359(Decider-2b), #2365(Clef), #2367 / #2368(LiquidAI d1): 이 API 위에 올라가는 모델 포팅. #2363과 #3218도 #2343에 막혀 있습니다.
- Upstream: Blaizzy/mlx-vlm#2396(공통 API와 Decider), Blaizzy/mlx-vlm#2402(CLI와 HTTP 라우트), 커밋 be0dbae9(미디어 선언).
