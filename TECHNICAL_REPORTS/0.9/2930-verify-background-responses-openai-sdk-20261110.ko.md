# PR #2930: OpenAI SDK로 백그라운드 Responses 검증

**날짜**: 2026-11-10
**상태**: 완료 (PR은 draft로 열려 있음; #2902를 닫으며 에픽 #2903의 일부)
**언어**: Python, Rust, Markdown
**위험도**: 낮음

## 요약

PR #2930은 백그라운드 Responses 에픽의 마지막 단위입니다. 제품 동작은 추가하지 않습니다. 대신 세 가지를 추가합니다. 공식 OpenAI Python SDK로 실행 중인 서버를 끝까지 검증하는 적합성 스크립트, 2,000개 작업을 돌리는 스크립트 기반 누수 테스트, 그리고 `docs/responses-api.md` 정합성 정리입니다. 검증은 `qwen3-0.6b-4bit`(GB10, CUDA 릴리스 빌드)에서 했습니다. 단일 모델 서버에서 일곱 검사가 모두 통과했습니다. 검사 7(429 작업 한도)은 `--responses-background-max-jobs 2`로 띄운 서버에서 통과했습니다. 다중 모델 라우터에서는 retrieve와 cancel에 `model`을 주지 않고 검사 1과 3이 통과했습니다. 200개 연속 작업에서 측정 구간의 RSS 증가는 0.2 MiB였습니다(한도 64 MiB). 백그라운드 모드를 끈 서버에서는 검사 1에서 실패하므로, 기능이 빠졌을 때 스크립트가 이를 잡아낸다는 것도 확인했습니다. 제품 버그는 발견되지 않았습니다.

## 1. 문제 정의

하위 이슈 #2896~#2901이 백그라운드 모드를 만들었습니다. 작업 저장소, 생성과 폴링, 취소, 스트리밍과 재개, 라우터와 disaggregated 역할이 여기에 해당하며, 모두 스크립트 provider 위의 라우트 테스트로 검증했습니다. 그런데 라우트 테스트는 서버 쪽 요청 형태를 직접 만들어 씁니다. 그래서 공식 SDK의 `responses.create(background=True)`, `retrieve(id, stream=True, starting_after=n)`, `cancel(id)`가 서버가 기대하는 요청을 보내고 응답을 제대로 해석하는지는 보여주지 못했습니다. 실제 모델에서 기능을 돌려 본 적도 없었습니다. 결과가 섞인 작업 다수가 아무것도 남기지 않는지 확인하는 테스트도 없었고, 머지된 동작에 맞춰 문서를 마지막으로 점검할 필요도 있었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `scripts/verify_responses_background.py` (신규) | 일곱 검사와 RSS 누수 한도. 표준 라이브러리와 `openai`만 사용하고 `uv run --with openai`로 실행. 옵션은 `--checks`, `--expect-max-jobs`, `--jobs`/`--pid`, `--timeout` |
| `src/server/routes/responses_background_leak_tests.rs` (신규) | `background_jobs_do_not_leak`: 일곱 종류에 걸친 2,000개 스크립트 작업 |
| `src/server/routes/responses_background_route_tests.rs` | `leak` 자식 모듈 등록 |
| `docs/responses-api.md` | `?stream=true&starting_after=N` 엔드포인트 행 분리, cancel 행 재작성, 백그라운드의 `store` 보존 규칙, 미디어 실패가 `failed`/`invalid_prompt`로 끝남, 스크립트 안내 |

### 일곱 검사

1. `background=True`로 생성하면 5초 안에 `background: true`와 함께 `queued`/`in_progress`를 받고, 폴링하면 비어 있지 않은 `output_text`로 `completed`에 도달합니다.
2. `store=False` 작업은 끝난 뒤에도 조회할 수 있습니다.
3. 8192 토큰 작업이 `in_progress`일 때 취소합니다. 취소, 반복 취소, 조회의 원시 HTTP 본문이 바이트 단위로 같고 `cancelled`입니다.
4. 스트리밍 백그라운드 생성은 `background: true`를 담은 `response.created`로 시작합니다. `sequence_number`는 엄격히 증가하고, 종료 이벤트는 정확히 하나이며 마지막에 옵니다.
5. 5번째 delta 뒤에 스트림을 끊고 `starting_after`로 재개합니다. 재개된 스트림은 `cursor + 1`에서 시작해 빈틈없이 이어져야 하고, 끊기 전후의 delta를 이으면 최종 `output_text`와 같아야 합니다.
6. 끝난 작업에 `previous_response_id`로 이어 붙인 후속 요청이 완료됩니다.
7. `max_retries=0`에서 한도를 넘는 작업은 `openai.RateLimitError`를 냅니다. 검사가 만든 작업은 `finally`에서 취소합니다.

## 3. 기술적 선택과 그 이유

### 3.1 이슈의 줄 번호 대신 문서 정합성 정리

이슈는 `8d3e16ad` 기준으로 작성되어 `:89`와 `:475`의 "no background job mode" 문구 교체를 요구했습니다. 이 단위가 실행될 때는 #2898~#2901이 이미 Background mode, Streaming and resuming, Serving roles 섹션과 저장소 표의 세 플래그를 추가한 뒤였습니다. 그래서 이 PR은 기존 문서를 이슈 체크리스트와 코드에 대조하고 빠진 부분만 채웁니다. 가장 중요한 수정은 cancel 행입니다. 기존 행은 포그라운드 스트림 경우만 설명하고 백그라운드 작업은 "대신 응답한다"고만 적었습니다. 그래서 포그라운드 스트림 취소가 여전히 최소 `{id, object, status}` 봉투를 돌려준다는 점이 가려져 있었습니다. 새 행은 두 경우를 모두 적습니다.

### 3.2 취소 멱등성을 바이트 단위로 비교

검사 3은 SDK 객체를 파싱해 비교하지 않고, 취소, 반복 취소, 조회의 원시 HTTP 본문을 비교합니다. 파싱해서 비교하면 필드 순서와 알 수 없는 필드를 무시하게 됩니다. 바이트 비교는 반복 취소가 기록된 객체를 그대로 돌려주지 않고 새로 만드는 경우(예: 새 타임스탬프)까지 잡아냅니다. #2899가 약속한 것이 바로 기록된 객체를 그대로 돌려주는 동작입니다.

### 3.3 모든 요청에 `reasoning: {effort: "none"}`

qwen3-0.6b는 단일 턴에서는 `/no_think`를 따르지만, `previous_response_id`로 이어진 턴에서는 이를 무시했습니다. 그 결과 출력 예산을 추론에 다 써서 빈 `output_text`를 냈고 검사 6이 실패했습니다. 서버는 `reasoning.effort: "none"`을 템플릿의 `enable_thinking: false`로 매핑하므로, 이 방식은 모든 턴에서 안정적으로 동작합니다. 이는 모델 동작이며 서버 버그가 아닙니다.

### 3.4 빠르고 결정적인 누수 테스트

누수 테스트는 스크립트 provider를 써서 2,000개 작업을 약 2초에 끝냅니다. 설정은 256 KiB 저장소 예산, 500 ms TTL, `store: false`용 2초 TTL입니다. 작은 예산 때문에 저장된 사본이 이미 축출되었을 수 있으므로, 각 종료 객체는 저장소가 아니라 `wait_background_terminal`로 읽습니다. `store: false` TTL은 처음에 250 ms였습니다. 그런데 보안 리뷰에서, 작업 종료와 테스트의 조회 사이에 그보다 긴 스케줄링 지연이 생기면 항목이 만료되어 테스트가 panic할 수 있다는 점이 드러났습니다. 그래서 2초로 늘렸습니다. 마지막 빈 저장소 확인은 고정 시간 sleep 대신 폴링으로 합니다(각 `get`이 sweep을 수행합니다). 테스트는 작업마다 바이트 예산을 확인합니다. 마지막에는 in-flight 취소 레지스트리가 비었는지, `active_background_jobs()`가 0인지, 작업당 생성이 정확히 한 번 실행되었는지, TTL 이후 저장소가 비고 0바이트인지 확인합니다.

### 3.5 클라이언트 호출에 상한 두기

SDK 기본 읽기 타임아웃은 600초입니다. 그래서 `--timeout` 값과 상관없이 `retrieve`나 스트림 읽기 하나가 멈추면 스크립트 전체가 멈출 수 있었습니다. 클라이언트는 `timeout=args.timeout`, `max_retries=0`으로 만들고, 스트리밍 검사에는 전체 기한도 둡니다.

## 4. 검증

- `background_jobs_do_not_leak`: 통과. 2,000개 작업에 약 1.9초가 걸렸고, CPU를 모두 점유한 상태에서도 테스트 전체가 최대 3.6초였습니다.
- `server::routes::responses_background`: 36개 통과 (`--test-threads=1`, `gpu-lock` 하에서 실행).
- clippy(`--lib --tests`, cuda, `-D warnings`), `cargo fmt --check`, `check_env_var_docs.py`, `insert_apache_header.py --check`, 스크립트 대상 `ruff check` 모두 깨끗합니다.
- 실제 모델(`qwen3-0.6b-4bit`, CUDA 릴리스 바이너리):
  - 단일 모델, `--jobs 200`: 검사 1~6 통과. 측정 구간 RSS는 1843.7 MiB에서 1843.9 MiB로 변했습니다.
  - `--responses-background-max-jobs 2`: 일곱 검사 모두 통과했고 세 번째 작업은 429로 거부되었습니다. 리뷰 수정 후 다시 실행해도 통과했습니다.
  - `--responses-background-max-jobs 0`: 백그라운드 비활성 400으로 검사 1에서 FAIL.
  - 라우터(`--models-dir`, `--models-max 2`, gemma-3-1b를 먼저 로드): retrieve와 cancel에 `model` 없이 검사 1과 3 통과.

## 5. 학습 포인트

- 라우터의 모델 탐색은 실제 디렉터리일 뿐 아니라 `config.json`이 일반 파일이어야 인식합니다(#1438의 경로 제한). 라우터 테스트용 임시 모델 디렉터리는 작은 파일은 복사하고 가중치만 심볼릭 링크로 걸어야 합니다.
- SDK 기반 스크립트는 라우트 테스트가 볼 수 없는 wire 형식 변화를 적은 비용으로 막아 줍니다. 라우트 테스트는 요청을 스스로 만들기 때문에 이런 변화를 볼 수 없습니다.

## 6. 후속 조치

- 막는 항목은 없습니다. 스크립트는 실제 서버와 모델이 필요해서 CI에서 돌지 않습니다. 릴리스 전에 GB10 한 대와 Apple Silicon 한 대에서 실행하면 SDK 계약을 계속 확인할 수 있습니다.
- 스크립트에는 `ruff format`이 적용되어 있지 않습니다. 저장소는 `python/` 아래에만 ruff format을 강제하며, 다른 `scripts/*.py` 39개도 같은 상태입니다.
