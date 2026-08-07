# 기술 보고서: PR #1083 - feat(server): serve Florence-2 through a seq2seq worker loop

**작성일**: 2026-08-08
**작성자**: mlxcel maintainers
**리뷰어**: 구현 및 보안 리뷰 사이클
**상태**: 완료 (요청 경계 문제 3건은 리뷰에서 수정, 단일 스트림 워커 공통 제약 4건은 기록만)
**언어**: Rust, Markdown
**위험도**: Medium (서버가 아예 거부하던 계열이 HTTP로 열렸다. 다른 계열의 응답 형태와 워커 경로는 그대로다)

---

## 요약

Florence-2는 CLI에서만 동작했다. `mlxcel-server`는 #856이 넣은 명시적 오류로 시작 단계에서 체크포인트를 거부했다. mlxcel의 생성 엔진은 Whisper ASR 파이프라인을 빼면 전부 디코더 전용인데 Florence-2는 BART 계열 seq2seq이기 때문이다. 비전과 프롬프트를 융합한 시퀀스에 인코더를 한 번 태우고, 캐시된 인코더 출력에 크로스 어텐션하면서 자기회귀 디코딩한다. 이걸 디코더 전용 워커에 넘기면 trait 완결성용 forward부터 쓰레기를 내보내게 되므로, 그 거부는 버그가 아니라 올바른 잠정 동작이었다.

이번에 서빙 경로를 넣었다. 전용 단일 스트림 워커(`src/server/florence2_worker.rs`)가 배치 워커와 같은 `mpsc` 요청 채널에서 분기하며, 두 spawn 경로 모두에서 스케줄러가 뜨기 전에 갈라진다. CLI의 답변 렌더러를 라이브러리로 옮겨 HTTP `message.content`가 CLI 출력과 바이트 단위로 같아진다. 인수 조건을 애초에 검증 가능하게 만드는 것이 이 이동의 목적이다. 파싱된 좌표는 `message.florence2_result`로 나란히 실리며, 서버가 이미 쓰던 `reasoning_content` 선택 필드 관례를 따른다.

흥미로운 부분은 짧은 워커 루프가 아니다. 요청별 격리가 강제된 것이 아니라 구조적이라는 점이다. 인코더 출력과 seq2seq 디코드 캐시는 `run_task_with_cancel` 안에서 만들어지고 반환과 함께 소멸하므로 샐 공유 상태 자체가 없다. 이를 고정하는 테스트는 순차 요청 두 개가 새 실행과 일치한다고만 주장하지 않는다. 두 요청의 인코더 출력으로 디코딩하면 로짓이 측정 가능하게 달라진다는 것까지 증명해서, 캐시가 새면 우연히 일치하는 게 아니라 답이 실제로 바뀌도록 만들어 두었다.

---

## 1. 문제 정의

### 1.1 배경

에픽 #850이 #852(BART seq2seq 스택), #853(DaViT 타워), #854(융합), #855(프로세서와 후처리), #856(CLI 파이프라인)으로 Florence-2를 올렸고 #1082가 양자화 로드 경로를 더했다. 전부 `mlxcel generate`에서만 닿을 수 있었다. `start_server`는 이렇게 중단했다.

```
Florence-2 is an encoder-decoder (seq2seq) VLM that mlxcel-server cannot serve
yet. Run it through the CLI instead: mlxcel generate -m <model> --image <image>
-p '<CAPTION>' (or another task marker such as <OCR> or <OD>).
```

### 1.2 기존 문제점

- **HTTP로 닿을 수 없었다.** OpenAI 호환 chat completions를 쓰는 모든 배포에서 Florence-2를 전혀 쓸 수 없었고, #1082가 막 풀어준 양자화 변환본도 마찬가지였다.
- **응답 형태가 미결 설계 항목이었다.** OpenAI chat 스키마에는 바운딩 박스, 쿼드 박스, 폴리곤, OCR 영역을 담을 자리가 없다. 이슈는 이걸 즉흥으로 정할 세부가 아니라 작업이 결론지어야 할 결정으로 명시했다.
- **#855에서 넘어온 보안 요구 두 건을 어떤 서버 표면이든 지켜야 했다.** `preprocess_with_sizes`는 이미 디코딩된 `DynamicImage`를 받으므로 압축 폭탄 방어를 프로세서 안에 둘 수 없다. 그리고 `Florence2Task::expand`는 15개 태스크 중 7개에서 호출자 텍스트를 인코더 프롬프트에 끼워 넣는다. CLI는 운영자 자신의 플래그만 받아 이 문제를 비껴가지만 서버는 그럴 수 없다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 거부를 없앤 뒤 Florence-2 체크포인트가 디코더 전용 워커에 닿음 | High (조용히 쓰레기를 서빙) | spawn 경로를 하나라도 빠뜨리면 확실 |
| 하나의 로드된 모델에서 순차 요청 간 인코더 상태 공유 또는 누수 | High (답변 교차 오염, 무증상) | 낮음. 다만 구별 가능성 테스트 없이는 탐지 불가 |
| 검증되지 않은 태스크 프롬프트 텍스트가 `Florence2Task::expand`에 도달 | High (마커 밀반입) | 경계 검사가 없으면 확실 |
| 승인 한도보다 먼저 이미지 디코딩이 실행됨 | High (압축 폭탄) | 워커가 `decode_request_images`를 건너뛰면 확실 |
| 렌더러가 복제되면서 HTTP 답변이 CLI 답변과 어긋남 | Medium (인수 조건이 조용히 깨짐) | 사본이 둘이면 시간 문제 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

#855 인계 두 건 모두 지켜졌고, PR 설명을 믿는 대신 리뷰에서 다시 확인했다.

**검토 항목:**

- [x] 입력 검증: `validate_task_input`이 입력을 받는 모든 모드를 요청 경계에서 길이와 형태 양쪽으로 검사한다.
- [x] 자원 한계: 이미지 바이트는 `decode_request_images`를 거치며 `current_image_input_limits()`(페이로드 크기, 해상도 상한, 디코드 할당 상한)를 픽셀 작업 이전에 적용한다. HTTP 경계의 `try_collect_image_data_with_limits`가 페이로드 한도를 한 번 더 건다.
- [x] 새로 생긴 인증/인가 표면 없음.
- [x] 로그에 요청 텍스트나 이미지 바이트가 남지 않음.

**태스크 입력 경계.** `parse_task_prompt`는 15개 중 인식된 마커만 받는다. 그 위에 입력을 받는 7개 모드에 대해 최대 2048바이트, 제어 문자 금지, 그리고 분류별 규칙이 붙는다. 영역 태스크 4개는 정확히 `<loc_a><loc_b><loc_c><loc_d>` 형태에 각 값이 `0..=999`여야 하며, 토큰 사이/앞/뒤의 잡문자, 숫자가 아니거나 자릿수가 넘치는 값, 범위 밖 값을 전부 거르는 엄격한 스캐너가 판정한다. 자유 텍스트 태스크 3개는 `<`와 `>`를 통째로 거부해서 위치 토큰, 시퀀스 토큰, 태스크 마커 어느 것도 인코더 프롬프트에 밀어 넣을 수 없게 한다.

드리프트 가드가 이 분할을 고정한다. `input_taking_task_set_matches_takes_input`이 `Florence2Task::ALL`을 훑으며 검증기의 두 태스크 집합이 `takes_input()`과 일치하는지 확인하므로, 입력을 받는 16번째 태스크가 추가되어도 경계를 조용히 우회할 수 없다.

**발견된 이슈:**

| 이슈 | 심각도 | 상태 |
|-----|-------|-----|
| 취소된 요청이 첫 취소 폴링 전에 인코더 패스를 전부 치름 | Medium | `cf4faa86`에서 수정 |
| 미디어, 이미지 개수, finish_reason 매핑에 단위 테스트 없음 | Medium | `cf4faa86`에서 수정 |
| `MAX_TASK_INPUT_BYTES` 주석이 서버 경로에 없는 "토큰화 이전" 성질을 주장 | Low | `cf4faa86`에서 수정 |
| `response_format`을 받아 놓고 무시 (다른 단일 스트림 워커와 공통) | Medium | 문서화, 미수정 (8절 참조) |
| 어떤 단일 스트림 워커도 `--max-queue-depth`를 지키지 않음 | Medium | 기록, 범위 밖 (8절 참조) |
| 선언된 이미지 개수와 해석된 개수를 대조하지 않음 | Low | 기록, 문서화된 MLX 워커 관례와 일치 (8절 참조) |
| `builtin_chat_template`이 `model_type` 원문 문자열을 대조하는 반면 `get_model_type`은 JSON 정제와 소문자 정규화를 먼저 거치므로, 정규화 경로로 Florence-2에 라우팅되는 체크포인트가 범용 템플릿으로 떨어져 모든 요청이 태스크 마커 파싱에서 거부될 수 있음 | Medium | `94d5774a`에서 수정: 템플릿 선택이 `get_model_type` 자체를 타며, 대소문자 혼용 회귀 테스트 추가 |
| 대화형 REPL(`mlxcel chat`)이 DiffusionGemma와 LLaDA-2는 거부하면서 Florence-2는 자기회귀 루프에 받아들임 (trait 완결성용 forward는 매 스텝 재인코딩) | Medium | `94d5774a`에서 수정: REPL이 태스크 파이프라인 안내와 함께 거부 |

초기 리뷰 한 번은 chat 라우트가 생성 오류를 `server_error`로 표시하므로 잘못된 클라이언트 요청이 HTTP 500을 받는다고 보고했다. 재현되지 않았다. `ErrorResponse::new`는 `StatusCode::BAD_REQUEST`를 상수로 넣는다. `error.type` 문자열만 `server_error`이며 이는 서버 전체에 걸친 기존 표기 부정확이지 이 경로만의 문제가 아니다.

보안 검토는 수정 없이 다음도 확인했다. `parse_region_bins`는 악의적 입력에 패닉 경로가 없고(워커 스레드가 `run_core_thread_or_abort` 아래에서 돌므로 중요한 성질이다), 2048바이트 입력 상한은 `encode_fused`가 테이블 조회 전에 과길이 시퀀스를 거부하므로 위치 테이블 범위 밖 gather로 이어질 수 없으며, 디코드 연산량은 `max_tokens`와 무관하게 `max_position_embeddings`로 유계이고, 꺾쇠 거부는 릴리스된 토크나이저의 모든 특수 토큰(전부 꺾쇠 형태)을 덮는다.

### 2.2 성능 관점

서빙은 한 번에 한 요청이다. 이슈는 문서화를 조건으로 첫 랜딩에서 이를 허용했고, `docs/supported-models.md`와 워커 모듈 문서 양쪽에 적혀 있다. 단순한 편법이 아닌 이유는 이렇다. 인코더 패스는 디코드 루프와 비용 프로파일이 다르므로 연속 배칭 승인 로직에 그냥 끼워 넣는 건 추측이 되고, 이 계열을 위한 배치 승인 정책은 설계도 측정도 되지 않았다. 동시 처리량에 대한 주장은 하지 않는다.

직렬 서빙이야말로 취소 누락이 문제였던 이유다. `generate_greedy_with_cancel`은 디코드 스텝마다 플래그를 폴링하는데, 그 폴링이 인코더 *이후에야* 시작한다는 점을 보기 전까지는 충분해 보인다. 이미 끊긴 클라이언트의 요청이 앞 요청 뒤에 줄 서 있었다면 DaViT 타워와 융합된 577+프롬프트 시퀀스에 대한 양방향 BART 패스를 그대로 치렀고, 그쪽이 비싼 절반이다. 수정은 핸들러 첫머리의 플래그 검사다.

2단계 생성 타임아웃과의 상호작용도 확인했고 안전하다. 1단계는 긴 prefill이 중단되지 않도록 설계상 무한 대기하며, Florence-2 워커는 생성이 끝날 때까지 아무것도 내보내지 않으므로 실행 전체가 1단계에 머문다. 상한이 걸린 디코드 행 감지 창은 적용되지 않는다.

### 2.3 호환성/의존성 관점

- **Breaking changes**: 없음. `florence2_result`와 `GenerationResult.structured_output`은 모두 `Option`이고 응답 필드에 `skip_serializing_if = "Option::is_none"`이 붙어, Florence-2가 아닌 응답은 전과 똑같이 직렬화된다. 키가 null이 아니라 아예 없음을 테스트가 확인한다.
- **새로운 의존성**: 없음.
- **호환성**: `mlxcel-server`가 체크포인트를 받아 서빙 상태에 도달한다. 시작 거부는 사라졌고, 같은 `get_model_type` 검사는 이제 텍스트 전용 워밍업만 막는다. 이미지 태스크 모델에 워밍업을 돌리면 무의미한 실패 로그만 남는다.

### 2.4 코드 품질 관점

렌더러 이동이 구조적 핵심이다. `render_task_result`가 `src/commands/generate_florence2.rs`에서 `src/models/florence2/render.rs`로 옮겨져 CLI 출력과 서버 응답이 같은 함수의 결과가 된다. 그러면 두 표면의 바이트 동일성은 매번 재확인할 주장이 아니라 코드의 성질이 된다. 이동하면서 기존의 `_ => String::new()` 포괄 분기도 없앴다. `Florence2TaskResult`는 `#[non_exhaustive]`라 크레이트 밖 CLI 사본에서는 와일드카드가 강제됐지만 크레이트 안에서는 아니므로, 이제 결과 variant가 추가되면 조용히 빈 문자열로 렌더되는 대신 컴파일이 깨진다.

추가된 테스트는 4개 파일 516줄이다. `florence2_render_tests.rs`가 5개 결과 variant 전부의 텍스트/JSON 형태를, `florence2_worker_tests.rs`가 영역 파서와 15개 태스크 전체에 대한 검증기, 그리고 리뷰 이후 매핑 가드 3종을 덮는다. `florence2_tests.rs`에는 순차 격리 테스트가 들어갔다.

---

## 3. 기술적 선택과 그 이유

### 3.1 구조화된 좌표를 어디에 실을 것인가

**컨텍스트:** OpenAI chat 스키마에는 박스, 폴리곤, OCR 영역을 담을 필드가 없고, 이슈는 이를 즉흥이 아니라 결론으로 요구했다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| `content`에 JSON 직렬화 | 필드 하나, 확장 없음 | 표준 클라이언트에게 흔한 경우가 읽히지 않게 됨 |
| 텍스트만 | 응답 형태 변화 0 | 모든 소비자가 `postprocess.rs`가 이미 하는 일을 다시 구현해야 함 |
| **선택: 둘을 나란히** | 표준 클라이언트는 정상 텍스트를, 좌표가 필요한 클라이언트는 JSON을 읽음 | 문서화할 비표준 필드 하나 |

**선택 이유:** 이슈 #1073에 메인테이너가 기록한 결정이다. 구현은 새 관례를 만들지 않고 서버가 이미 쓰던 확장 필드 관례(`reasoning_content`, 이 자체가 vLLM을 따른 것)를 따랐고, `docs/responses-api.md`에 "mlxcel extension fields" 절을 새로 만들어 문서화했다.

**트레이드오프:** 이 필드는 비스트리밍 chat completions 표면에만 실린다. 스트리밍과 `/v1/responses`는 렌더된 텍스트만 반환하며, 둘 다 문서에 적혀 있다.

### 3.2 인코더 캐시는 누가 소유하는가

**컨텍스트:** 이슈는 "요청당 인코더 패스를 한 번 돌리고, 인코더 출력을 캐시하고, 크로스 어텐션 디코드 루프를 구동하는 seq2seq 워커 변종"을 적었다.

**대신 한 일:** 그 파이프라인은 #856의 `Florence2Model::generate_greedy`로 이미 있었다. 워커는 서버 안에 인코더 캐시 계층을 새로 만드는 대신, 취소 훅만 추가한 모델 소유 파이프라인을 재사용한다.

**선택 이유:** 이것이 요청별 격리를 구조적으로 만든다. 인코더 출력, seq2seq 캐시, 전처리된 픽셀 텐서가 모두 `run_task_with_cancel` 호출 하나에 지역적이고 반환과 함께 소멸한다. 나중에 누군가 수명을 잘못 다룰 서버 측 캐시 객체가 없다. 서버에 캐시를 다시 구현했다면 이슈의 "Technical Considerations"가 경고한 바로 그 공유 상태 위험을 만들었을 것이다.

### 3.3 chat 모델이 아닌 모델을 위한 내장 chat 템플릿

**컨텍스트:** Florence-2 체크포인트는 chat 템플릿 없는 평범한 BART `tokenizer_config.json`만 싣고 있고 `chat_template.jinja`도 `chat_template.json`도 없다. 로컬 변환본 5개 전부에서 확인했다.

**문제:** 일반 폴백인 `User:\n\nAssistant: `는 태스크 프롬프트 앞에 `User:`를 붙이는데, `parse_task_prompt`는 문자열이 태스크 마커로 시작할 것을 요구한다. 모든 요청이 거부됐을 것이다.

**선택:** `model_type == "florence2"`를 키로 하는 내장 템플릿. Jina VLM이 만든 선례에 합류한다. 메시지 텍스트를 그대로 내보내며, 문자열 content는 직접, 타입 있는 content 목록은 `text` 파트만 취하고, 역할 접두사도 생성 프롬프트도 붙이지 않는다. 이미지 파트는 픽셀로 대역 외 전달되고 Florence-2에는 렌더할 이미지 자리표시 토큰이 없다.

**트레이드오프:** 다중 메시지 요청은 연결되므로, 태스크를 담은 사용자 메시지 하나를 넘어서는 대화는 유효 마커를 나열한 메시지와 함께 태스크 파서가 거부한다. 대화 의미론이 없는 모델에는 수용 가능하고 템플릿 주석에 적혀 있다. 운영자의 `--chat-template` 재정의는 여전히 우선하며 이는 기존 해석 순서 그대로다.

### 3.4 스트리밍은 델타 청크 하나

후처리에는 완성된 디코드가 필요하다. `<loc_*>` 토큰은 답변 전체를 원본 이미지 크기에 맞춰 파싱해야 픽셀 좌표가 된다. 원시 토큰 시퀀스를 스트리밍하면 같은 요청의 비스트리밍 `content`와 일치하지 않는 텍스트가 나간다. 그래서 워커는 렌더된 답변 전체를 `delta.content` 청크 하나로 보내고 `finish_reason` 청크를 잇는다. 문서에도 그렇게 적었다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[변경 전]
start_server --> get_model_type == Florence2VLM --> anyhow::bail!

[변경 후]
start_server --> is_florence2 플래그 (텍스트 전용 워밍업만 차단)
     |
     v
spawn_model_worker_with_batch_config / spawn_legacy_model_worker
     |
     +-- LoadedModel::DiffusionGemma --> diffusion 워커 루프 --> return
     +-- LoadedModel::Llada2Moe      --> llada2 워커 루프    --> return
     +-- LoadedModel::Florence2VLM   --> florence2 워커 루프 --> return
     |
     v
BatchScheduler (디코더 전용 계열만)
```

분기는 `LoadedModel` 구성 이후, 스케줄러 설정 이전에 놓이며 MLX spawn 경로 두 곳 모두에 있다. 텐서 병렬과 파이프라인 병렬 요청도 같은 `spawn_model_worker_with_batch_config`를 지나므로 같은 분기에 걸린다. XLA 워커는 `LoadedModel`을 아예 만들지 않고 `XlaBatchEngine`으로 로드하는데, 이 아키텍처를 로드할 수 없어 무언가를 서빙하기 전에 명시적 오류로 실패한다.

### 4.2 주요 코드 변경

**`src/server/florence2_worker.rs` (신규, 리뷰 후속 포함 419줄)**

요청 하나의 처음부터 끝까지: 취소 검사, 미디어 거부, 이미지 개수 검사, `parse_task_prompt`, `validate_task_input`, 한도가 걸린 이미지 디코딩, `run_task_with_cancel`, 그리고 `render_task_result`를 `content`로 `structured_task_json`을 `florence2_result`로. 모든 실패 경로는 `GenerateEvent::Error` 하나를 보내고 반환하므로 잘못된 요청 하나가 워커를 무너뜨리지 않는다.

**`src/models/florence2/render.rs` (신규)**

`render_task_result`(CLI에서 이동, 동작 불변)와 `structured_task_json`. JSON 키 이름은 업스트림 `Florence2Processor.post_process_generation`(`bboxes`, `quad_boxes`, `polygons`, `labels`, `bboxes_labels`, `polygons_labels`)을 따르므로 HuggingFace나 mlx-vlm dict 형태로 작성된 코드가 그대로 이식된다.

**`src/models/florence2/{model,runtime}.rs`**

`generate_greedy_with_cancel`과 `run_task_with_cancel`이 디코드 스텝마다 폴링되는 `Option<&AtomicBool>`을 받는다. 기존 진입점은 `None`으로 위임하므로 CLI 경로는 그대로다. `Florence2RunOutput`에는 usage 블록용 `prompt_tokens`가 붙었다.

**리뷰 후속, `cf4faa86`**

```rust
// Serving is serial, so a request can sit in the channel while its client
// goes away. The per-step poll inside `generate_greedy_with_cancel` only
// starts after the encoder pass, which is the expensive half (DaViT tower
// plus the bidirectional BART encoder over the fused sequence), so an
// already-abandoned request would still pay for it. Drop it here instead.
if cancelled.load(Ordering::Relaxed) {
    let _ = response_tx.send(GenerateEvent::Error(
        FLORENCE2_CANCELLED_BEFORE_START_MSG.to_string(),
    ));
    return;
}
```

같은 커밋에서 `reject_media`, `reject_image_count`, `florence2_finish_reason`을 테스트가 붙은 순수 함수로 추출했다. 옆 단일 스트림 워커가 이미 쓰는 `reject_audio_video` / `diffusion_finish_reason_str` 모양을 따른 것이다. 그 전까지 거부 메시지 상수 둘은 `pub(crate)`인데 참조하는 테스트가 없었고, "length"와 "stop" 선택도 테스트되지 않았다.

---

## 5. 학습 포인트

### 5.1 디코더 전용 엔진 위에서 seq2seq 모델 서빙하기

**개념:** 디코더 전용 서빙 엔진은 생성 상태가 토큰 시퀀스 하나에 대해 자라는 KV 캐시라고 가정한다. seq2seq 모델의 상태는 수명이 다른 두 종류다. 한 번 계산되고 요청 내내 고정인 인코더 출력, 그리고 스텝마다 자라는 디코더 self-attention 캐시. 크로스 어텐션 K/V는 인코더 출력에서 한 번 투영된 뒤 고정된다.

**이 PR에서의 적용:** 스케줄러에 두 번째 캐시 종류를 가르치는 대신, 이 계열은 `supports_batching() == false`를 선언하고 전용 루프를 받는다. DiffusionGemma와 LLaDA-2가 다른 이유로 했던 것과 같은 수순이라, 이제 트리에는 채널 프로토콜 하나와 관례 한 벌을 공유하는 단일 스트림 워커가 셋 있다.

**일반화되는 지점:** 생성이 스텝 소유가 아니라 모델 소유인 모든 모델. 채널 프로토콜(`ModelRequest` in, `Token` / `Done` / `Error` out)이 통합 표면의 전부다.

### 5.2 실패할 수 있는 격리 테스트 쓰기

**개념:** "A 다음의 B가 B 단독과 같다"는 그 자체로는 약한 주장이다. 두 요청이 우연히 같은 답을 내거나 모델이 해당 입력에 둔감하면 그냥 통과한다.

**이 PR에서의 적용:** `sequential_requests_reuse_no_encoder_state`는 후반부를 덧붙인다. 요청 A의 인코더 출력과 B의 인코더 출력으로 각각 한 스텝을 디코딩하고 로짓 차이가 1e-6을 넘는지 확인한다. 그러면 앞의 주장이 "둘이 같았다"에서 "누수가 있었다면 달랐을 텐데도 둘이 같았다"로 바뀐다. 테스트는 이 사실을 자기 실패 메시지에 적어 두어, 나중에 구별 가능성 절반을 깨뜨린 사람에게 격리 주장이 왜 무의미해졌는지 알려준다.

### 5.3 구조적 성질로서의 바이트 동일성

**개념:** "HTTP 답변이 CLI 답변과 같다" 형태의 인수 조건은 답변 구현이 하나일 때만 검증 가능하다.

**이 PR에서의 적용:** 렌더러를 라이브러리로 옮기면서 이 조건이 검사로 참인 게 아니라 구성상 참이 됐다. 규율로 동기화되는 렌더러 두 벌이라는 대안은 어느 한쪽이 처음 바뀌는 순간 무너진다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 관련성 |
|-------|-----|-------|
| `seq2seq` | 고정된 인코더 출력에 크로스 어텐션하는 인코더-디코더 생성 | 서버가 두 릴리스 동안 이 계열을 거부한 아키텍처적 이유 |
| `Florence2SeqCache` | 일회성 크로스 어텐션 K/V와 자라는 디코더 self-attention 캐시를 함께 든 이중 캐시 | 수명 자체가 격리 보장인 요청별 상태 |
| `ImageInputLimits` | 이미지 디코딩 전에 적용되는 페이로드/해상도/디코드 할당 상한 | #855가 넘긴 압축 폭탄 경계 |
| `skip_serializing_if` | `None` 필드를 통째로 생략하는 serde 속성 | `florence2_result`가 다른 계열의 응답 형태를 건드리지 않게 하는 장치 |
| `<loc_N>` | `tokenizer.json`에서 special로 표시된 Florence-2의 1000빈 좌표 토큰 | 파서가 소비하는 출력 형식이자 검증기가 막는 주입 벡터 |

### 관련 기술

- **Florence-2** (Microsoft): 태스크 프롬프트 기반 비전 파운데이션 모델. https://huggingface.co/microsoft/Florence-2-base-ft
- **mlx-vlm** `processing_florence2.py`: `structured_task_json`이 따르는 후처리 dict 형태. https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py

### 관련 PR/이슈

- Issue #1073: 이 작업.
- Issue #856 / PR #1071: 이 PR이 없애는 시작 거부를 넣었고, 워커가 재사용하는 CLI 파이프라인을 만들었다.
- Issue #855: 보안 요구 두 건을 넘겼다.
- PR #1082: 양자화 Florence-2 로드 경로. 그 체크포인트들이 이 경로로 그대로 서빙된다.
- Issue #217 phase 3 / #546: 이 워커가 따르는 DiffusionGemma, LLaDA-2 단일 스트림 워커.
- Issue #633: dispatch 스레드 사전 토큰화. `MAX_TASK_INPUT_BYTES`와의 상호작용을 리뷰에서 문서로 바로잡았다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 22 |
| 추가된 라인 | +1483 |
| 삭제된 라인 | -219 |
| 추가된 테스트 라인 | 516 |
| 신규 단위 테스트 | 30개 (렌더러/JSON 12, 워커 경계 및 매핑 15, 모델 격리 1, chat 템플릿 2) |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| Feature | 1 | OpenAI 호환 HTTP로 Florence-2 서빙 |
| Security | 2 | 요청 경계에서 이미지 디코딩 한도와 태스크 입력 검증 |
| Code Quality | 3 | 렌더러 라이브러리 이동, 매핑 가드 추출 및 테스트, 완전 매치 복원 |
| Performance | 1 | 취소된 요청을 인코더 패스 전에 폐기 |
| Documentation | 4 | `supported-models.md`, `responses-api.md` 확장 필드 절, `MAX_TASK_INPUT_BYTES` 정정, 오래된 텐서 병렬 주석 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `fbdc37db` | feat | serve Florence-2 through a seq2seq worker loop |
| `cf4faa86` | fix | harden the Florence-2 seq2seq worker request boundary |

---

## 8. 후속 조치

### 기록만 하고 수정하지 않은 것

아래 네 가지는 이 PR의 성질이 아니라 단일 스트림 워커 부류의 성질이다. DiffusionGemma와 LLaDA-2도 1, 2, 4번을 똑같이 갖고 있어서 한 계열만 고치면 트리가 어긋난다. 워커 셋을 함께 다루는 이슈 하나로 가야 한다.

- **`--max-queue-depth`가 지켜지지 않는다.** `AppState::can_accept_request()`는 `BatchScheduler`만 갱신하는 `batch_metrics.queue_depth()`를 읽는다. 요청은 디코딩된 이미지 페이로드를 실은 채 무제한 `mpsc` 채널에 쌓이므로 승인 제어가 적용되지 않는다. 레거시 워커는 이 값에 `usize::MAX`를 명시적으로 넘기므로 적어도 그쪽에서는 의도된 동작이다.
- **`response_format`을 받고 무시한다.** `options.structured`는 `batch/scheduler.rs`에서만 소비된다. 이번 리뷰에서 Florence-2에 한해 `supported-models.md`에 명시했고, diffusion 워커 쪽에는 같은 문장이 아직 없다.
- **usage `prompt_tokens`가 이미지 특징 토큰을 제외한다.** `Florence2RunOutput.prompt_tokens`는 인코더의 텍스트 토큰 수다. 실제 융합 인코더 시퀀스는 `base-ft` 기준 그 앞에 투영된 이미지 토큰 577개를 싣고 있으므로 보고되는 prompt usage가 그만큼 실제 prefill을 낮게 잡는다. 필드에 문서화돼 있다. 융합 길이를 보고하려면 `Florence2Model::encode`에서 값을 꺼내 와야 한다.
- **선언된 이미지 개수와 해석된 개수를 대조하지 않는다.** 이미지 해석기는 풀 수 없는 `image_url`을 관대하게 버리므로, 이미지 둘을 선언했는데 하나가 실패한 요청이 이미지 하나짜리 요청으로 받아들여진다. `MediaRequestMetadata`는 두 개수를 모두 보관하며, MLX와 diffusion 워커는 이를 무시하고 XLA는 검증한다는 사실이 그 필드 문서에 적혀 있다. Florence-2는 문서화된 관례를 따른 것이다.

### 모니터링 필요

- `Florence-2 seq2seq worker ready` 로그로 분기가 탔는지 확인한다. Florence-2 체크포인트가 로드된 상태에서 이 줄이 없다면 모델이 스케줄러에 닿았다는 뜻이다.
- `Florence-2 task prompt:`와 `Florence-2 requires exactly one image` 오류 비율은 클라이언트가 태스크 마커 대신 대화형 요청을 보내고 있음을 알려준다.
- 요청당 벽시계 시간은 인코더 패스가 지배하므로, 동시 요청 시 큐 지연이 깊이에 선형으로 늘어나는 것이 설계된 동작이다.

### 향후 개선 사항

- seq2seq 경로의 배치 승인. 저장소의 성능 이슈 완료 기준에 따라 인코더 대 디코드 비용 분할 측정이 먼저다.
- 점진적 스트리밍은 `<loc_*>` 접두사를 파싱할 수 있는 후처리기를 요구하는데, 이 정도 길이의 답변에 그만한 가치가 있는지는 분명하지 않다.
- 소비자 요구가 있으면 `/v1/responses`에도 `florence2_result`를 실을 수 있다.

---

## 부록

### A. 테스트 결과

```
cargo clippy --profile test-fast --features metal,accelerate --lib --tests   clean
cargo fmt --check                                                             clean
cargo test --profile test-fast --features metal,accelerate --lib florence2    186 passed
cargo test --profile test-fast --features metal,accelerate --lib server::     1655 passed, 8 ignored
```

### B. 실제 체크포인트 검증

`Florence-2-base-ft-bf16`과 `Florence-2-base-ft-4bit`에 COCO `val2017/000000039769.jpg`(640x480)로 실행했다. `<CAPTION>`, `<OD>`, `<CAPTION_TO_PHRASE_GROUNDING>`의 HTTP 답변이 같은 바이너리의 CLI 답변과 바이트 단위로 일치한다. 거부 경로 5종이 명시적 오류와 함께 HTTP 400을 반환했고 이후에도 워커가 계속 서빙했다. 전송 크기 380 KB짜리 20000x20000 PNG를 디코딩 전에 거부, 잘못된 영역 입력, 2799바이트 초과 입력, 꺾쇠 밀반입 시도, 알 수 없는 태스크 마커. 같은 세션에서 스트리밍과 반복 요청 격리도 확인했다. 전체 기록은 PR 본문에 있다.

### C. 참고 자료

- `docs/supported-models.md` Florence-2 항목: 서빙 의미론과 문서화된 제약.
- `docs/responses-api.md` "mlxcel extension fields": `florence2_result` 응답 계약.
