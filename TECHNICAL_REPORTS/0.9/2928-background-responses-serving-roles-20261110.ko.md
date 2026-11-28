# PR #2928: 서빙 역할 전반의 background Responses 지원

**작성일**: 2026-11-10
**상태**: 완료 (draft PR로 열려 있음; #2901을 닫으며 epic #2903의 일부)
**언어**: Rust, Markdown
**위험도**: Medium

## 요약

PR #2928은 멀티 모델 라우터(`--models-dir`, `--models-preset`, 모델 캐시)에서 background Responses를 쓸 수 있게 하고, disaggregated front에는 명확한 거절 응답을 추가합니다. 라우터가 response id를 가진 로드된 모델을 찾아 주므로 `client.responses.retrieve(id)`, `.cancel(id)`, `.delete(id)`가 모델 이름 없이 동작합니다. eviction은 실행 중인 작업이 있는 모델을 건드리지 않고, 명시적 unload는 그 작업을 `failed`로 끝냅니다. disaggregated front는 `POST /v1/responses`에 serving node로 보내라는 400을 돌려줍니다. 단일 모델과 pipeline head 동작은 바뀌지 않습니다. 14개 파일, +1203/-50이며 약 810줄이 테스트입니다. 구현 리뷰에서 CRITICAL이나 HIGH 문제는 나오지 않았습니다.

## 1. 문제 정의

이 PR 전에는 라우터의 모델별 sub-app에 response store가 아예 없어서 background 모드는 store-disabled 400으로 거절되었고 `previous_response_id` 체이닝도 되지 않았습니다. store가 있더라도 id로 지정하는 요청(`GET`, `DELETE /v1/responses/{id}`, `POST .../cancel`)에는 모델이 없어서 라우터 dispatcher가 `model name is missing from the request`로 답했습니다. eviction은 `active_requests`만 봤는데, background 작업은 create 호출이 반환된 뒤에는 여기에 잡히지 않으므로 세 번째 모델을 로드하면 작업 도중인 모델이 조용히 unload될 수 있었습니다. disaggregated front는 `/v1/responses`에 빈 404를 돌려줬습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/server/startup.rs` | 단일 모델 startup에서 `build_responses_store`를 분리해 라우터의 로드마다 재사용 |
| `src/server/router_models.rs` | 로드마다 store 생성; `active_background_jobs`, `is_busy_for_eviction`, `holds_response`; `entry_holding_response`; `BackgroundJobsOnUnload::{Fail, Refuse}`; drain 시작 시와 `shutdown_worker` 직전에 작업을 실패 처리 |
| `src/server/router_server.rs` | `response_affinity_id` 경로 파서와 `dispatch`의 probe; cancel에서 `?model=` 허용 |
| `src/server/responses_store.rs`, `responses_store_cancel.rs` | 읽기 전용 `contains_response`; `fail_active_background` |
| `src/server/routes/responses.rs` | 공용 `response_not_found` 404 |
| `src/server/router_front.rs` | `POST /v1/responses`, `/responses`를 `invalid_request_error`로 거절 |
| 문서 | `responses-api.md`의 "Serving roles" 절; `distributed.md`, `server-features.md`의 안내 |

## 3. 기술적 선택과 그 이유

### 3.1 라우터 수준 id 맵 대신 프로세스 내 probe

라우터는 로드된 각 엔트리의 store에 `contains_response(id)`를 묻고, 이름 순서(`entries`는 `BTreeMap`)로 처음 일치한 엔트리로 요청을 넘깁니다. 라우터 전체 id 맵은 각 store와 함께 상한을 두고 만료시켜야 하는 두 번째 구조라서 채택하지 않았습니다. probe는 `--models-max`로 상한이 정해지고, autoload하지 않으며, LRU를 갱신하지 않고, 다른 dispatch 요청과 마찬가지로 라우터 수준 API-key 미들웨어 뒤에서 실행됩니다. 호출자가 각 모델에 `?model=`을 붙여 알아낼 수 없는 정보는 드러내지 않습니다.

`contains_response`는 jobs 뮤텍스, entries 읽기 락, in-flight 읽기 락을 중첩 없이 하나씩 잡습니다. 종료 기록(`record_terminal_locked`)은 jobs 뮤텍스를 쥔 채로 저장 엔트리를 넣으므로, probe는 끝나는 작업을 두 곳 중 하나에서 반드시 보고 일시적인 404를 내지 않습니다.

### 3.2 probe가 다루는 경로

`/v1` 유무와 관계없이 `GET`, `DELETE /v1/responses/{id}`와 `POST /v1/responses/{id}/cancel`입니다. id는 비어 있지 않은 한 세그먼트이고, sub-app의 `Path` extractor와 같은 방식으로 percent-decode합니다. `compact`와 `input_tokens`는 제외합니다. query는 경로에 포함되지 않으므로 `?stream=true&starting_after=N` 재개 요청도 같은 방식으로 라우팅됩니다. `GET .../input_items`는 다루지 않습니다. sub-app에 그 라우트가 없으므로 dispatcher의 missing-name 400이 유지됩니다.

### 3.3 eviction은 거절하고 unload는 실패시킨다

두 eviction 지점(명시적 대상과 LRU 필터)이 모두 `is_busy_for_eviction`을 쓰고, eviction unload는 `BackgroundJobsOnUnload::Refuse`로 실행됩니다. busy 검사와 drain 사이에 작업이 수용되었다면 작업을 죽이지 않고 `mark_ready`와 capacity 오류로 unload를 되돌립니다. 명시적 unload(`POST /models/unload`, WebUI, 제거, 종료)는 `Fail`을 씁니다. 작업은 drain 시작 시 `server_error` / `The model was unloaded before the response finished.`로 `failed`가 됩니다. 작업을 스트리밍하는 클라이언트가 request lease를 쥐고 있어서, 그렇게 하지 않으면 drain이 작업을 기다리기 때문입니다. drain 시작 시점에 진행 중이던 create를 잡기 위해 `shutdown_worker` 직전에 한 번 더 실패 처리합니다. cancel이 이미 요청되었지만 아직 기록되지 않은 작업은 `JobRecorder::fail`과 같이 `cancelled`로 끝나므로, cancel과 겹친 unload도 작업 자체 경로와 같은 상태를 냅니다. 라우터에는 idle 또는 TTL 자동 unload가 없고, `entry_idle_for_webui_removal`은 이미 unload된 엔트리를 요구하므로 다른 idle 검사에는 새 helper가 필요하지 않았습니다.

### 3.4 기존 라우터 사용자에게 보이는 동작 변화

로드된 모델마다 라우터의 `--responses-store-*`, `--responses-background-*` 플래그로 자기 store를 갖습니다. 기본값 `store: true`인 포그라운드 `/v1/responses`가 이제 라우터 모드에서도 저장되고, 메모리 상한은 로드된 모델 단위입니다. 최악의 경우는 `--models-max` 곱하기 `--responses-store-max-bytes`(기본 256 MiB)입니다. store는 unload 시 모델과 함께 버려지므로 그 뒤에는 모든 id가 404를 돌려주며, 실제 체크포인트에서 확인했습니다. 문서에는 모델 단위라는 점과 곱해진 최악값이 적혀 있습니다: `--models-max` x (`--responses-store-max-bytes` + `--responses-background-max-jobs` x `--responses-background-event-log-bytes`), 기본값에서 2 GiB.

## 4. 검증

- GB10에서 `gpu-lock`, `--test-threads=1`로 lib 테스트: `router_` 265개 통과(3개 ignored), `server::routes::responses` 103개, `responses_store` 37개, `router_front` 56개, 새 테스트 12개 포함 모두 통과.
- `cargo clippy -p mlxcel --lib --tests --profile test-fast --features cuda -- -D warnings`와 `cargo fmt --all -- --check` 통과.
- `~/models-102/mlx` 위의 실제 라우터에 `qwen3-0.6b-4bit`와 `qwen2.5-0.5b-instruct-4bit`를 로드(`--models-max 2`): qwen2.5의 background 작업을 모델 없이 poll하고 cancel했고 전체 `cancelled` Response를 받았습니다. qwen3 작업은 정상적인 문장으로 완료되었고 모델 없이 조회되었습니다. 모르는 id는 `not_found` 404였습니다. `in_progress` 작업 중 qwen2.5를 unload하자 `unload ended background responses as failed ... jobs=1`이 기록되었고 그 id는 이후 404였습니다.

## 5. 후속 조치

- 해결됨: 최악 메모리 공식은 `docs/responses-api.md`에 있고, `fail_active_background`는 cancel이 요청된 작업을 `cancelled`로 기록합니다(커밋 a8fc8cae).
- LOW: 명시적 unload가 drain 시작 시 작업을 실패시키므로, 이후 drain 타임아웃이나 stale revision으로 unload가 실패하면 모델은 로드된 채 작업만 끝나 있습니다. 이 내용은 이제 `docs/responses-api.md`에 문서화되어 있습니다.
- LOW: 테스트는 test factory로 store를 주입하므로 실제 `build_model_app(..., build_responses_store(...))` 연결은 수동 실행으로만 확인되었습니다.
