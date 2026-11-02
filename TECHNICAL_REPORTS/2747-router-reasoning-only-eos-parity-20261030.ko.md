# PR #2747: Router reasoning_only 및 응답 parity, LLaVA와 TP Llama의 EOS id

**날짜**: 2026-10-30 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust
**위험도**: 낮음 (router 응답에 필드 하나를 추가하는 변경이며, LLaVA의 Llama backbone checkpoint와 tensor-parallel Llama runtime의 모델 수준 EOS 목록이 늘어남)

## 요약

PR #2719 (issue #2715)는 parity 잔여 항목 세 개를 남겼고, issue #2729가 이를 모았으며 이 PR이 모두 닫는다. LLaVA text model과 tensor-parallel Llama runtime은 새 `loading::merged_eos_token_ids`를 통해 checkpoint sidecar가 선언한 EOS id를 보고하게 되어, `llava-1.5-7b-4bit`는 `[]` 대신 `[2]`를 보고한다. disaggregated router는 single node와 정확히 같은 조건에서 non-stream chat 응답에 `message.reasoning_only: true`를, streaming finish chunk에 `delta.reasoning_only: true`를 넣는다. 규칙을 복사하지 않고 single node의 판정 함수(`log_if_reasoning_only`, `stream_reasoning_only`)를 그대로 호출한다. 실제 프로세스로 띄우는 thinking e2e 테스트에는 thinking 블록이 닫히는 1024-token turn이 추가되어, reasoning과 answer의 분리가 처음으로 실제 모델에서 router와 single node 사이에 비교된다.

## 1. 문제 정의

### 두 loader가 checkpoint 선언보다 짧은 EOS 목록을 보고함

#2719 이후 `Llama3Model::eos_token_ids()`는 `ModelArgs`가 선언한 id를 반환하고, 어휘 크기 128256인 Llama 3 vocabulary일 때에만 128001/128009를 덧붙인다. 두 loader는 sidecar id가 빠진 config로 이 args를 만든다. `build_llava_text_model`은 `text_config`만 파싱하는데, `llava-1.5-7b-4bit`는 그곳에 `eos_token_id`를 선언하지 않으므로(vocab 32064) `read_eos_token_ids`가 `tokenizer_config.json`에서 `[2]`(`</s>`)를 찾아내는데도 모델은 `[]`를 보고했다. `TensorParallelLlamaModel::from_model_dir`은 최상위 `config.json`만 읽고 `generation_config.json`이나 `tokenizer_config.json`은 읽지 않는다.

server와 CLI가 `read_eos_token_ids`를 sampling stop id에 병합하므로 생성 자체는 제대로 멈췄다. 문제는 `LanguageModel::eos_token_ids()`만 읽는 소비자였다. `src/server/engine_probe/cli_engine.rs`의 `suppress_eos_for_cli`와 `admission.rs`의 XTC allowlist가 그렇다. `src/loading/special.rs`의 loader들은 이미 `set_eos_token_ids(read_eos_token_ids(..))`를 호출하고 있었으므로 이 두 경로만 예외였다. #2719 리포트도 이를 알려진 잔여 항목("빈 목록이 가능하다")으로 남겼다.

### router가 `reasoning_only`를 내보내지 않음

single node는 토큰을 생성했지만 모두 `reasoning_content`로 갔을 때, 보통 `max_tokens`에서 thinking 블록이 잘렸을 때 응답에 `reasoning_only: true`를 표시한다. 이 플래그가 없으면 client는 `content: ""`만 받고 더 올 내용이 없다는 신호를 받지 못한다. router의 `chat_completion_json`과 `chat_chunk_finish`에는 이 필드가 없었기 때문에, 같은 요청이 disaggregated 배포를 거치면 신호가 사라졌다. main 8385f40b에서 먼저 실패시킨 결과: `reasoning_content: "still thinking"`, `finish_reason: "length"`인 본문에서 `message.reasoning_only`가 `Null`이었고, stream finish chunk는 `delta: {}`였다.

### reasoning과 answer가 함께 있는 응답을 실제 프로세스로 비교한 테스트가 없음

`disaggregated_router_filters_thinking_output`은 router와 single node의 chat 응답을 바이트 단위로 비교했지만, `max_tokens: 32`에서 qwen3-0.6b는 `</think>`를 닫지 못하므로 양쪽 모두 `content: ""`를 반환했다. 닫힌 thinking 블록과 비어 있지 않은 answer를 나누는 코드 경로는 두 실제 프로세스 사이에서 한 번도 비교되지 않았다. #2719 리포트는 이를 "검증하지 못한 항목"에 기록했다.

## 2. 기술적 선택과 그 이유

### 선언된 id 뒤에 sidecar id를 병합하는 공용 helper

`merged_eos_token_ids(declared, model_dir)`는 선언된 id를 원래 순서대로 반환하고, 그 뒤에 `read_eos_token_ids(model_dir)`가 찾은 id 중 아직 없는 것을 덧붙인다. 위치는 `src/loading/mod.rs`의 `read_eos_token_ids` 옆이다.

선언된 id를 앞에 두면 모델이 기존에 보고하던 순서가 유지되므로, config에 모든 id를 선언한 checkpoint에는 아무 변화가 없다. `loading::special`처럼 목록을 `read_eos_token_ids` 결과로 통째로 바꾸면, sidecar가 더 적은 id를 나열할 때 `ModelArgs::eos_token_ids()`가 128256 vocabulary에 덧붙이는 Llama 3 기본값이 사라진다. 합집합은 두 출처를 모두 지킨다.

LLaVA의 `Llama` 분기는 `from_weights` 뒤에 `set_eos_token_ids(merged_eos_token_ids(text_args.eos_token_ids(), model_path))`를 호출한다. `TensorParallelLlamaModel`에는 모든 rank에 같은 목록을 설정하는 `set_eos_token_ids`가 추가되고, `from_model_dir`은 모델을 만든 뒤 이를 호출한다. `eos_token_ids()`가 읽는 `ranks[0]`만이 아니라 모든 rank에 설정하므로, 나중에 다른 rank를 읽는 호출자가 생겨도 rank 사이의 값이 일치한다. LLaVA의 `Qwen2` 분기는 바꾸지 않았다. `Qwen2Model`에는 setter가 없고, 이를 추가하는 일은 범위 밖이었다.

### 규칙을 다시 쓰지 않고 single node의 판정 함수를 재사용

`src/server/routes/chat.rs`의 `log_if_reasoning_only`와 `stream_reasoning_only`가 private에서 `pub(crate)`로 바뀌고 router가 이를 호출한다. 두 함수 모두 자명하지 않은 조건을 담고 있다. non-stream 판정은 shaping된 `reasoning_content`가 비어 있지 않을 때만 참이 되므로, 정리 후 `""`가 되는 Gemma 4의 marker 전용 출력이 reasoning을 주장하지 않는다. stream 판정은 `finish_reason == "tool_calls"`를 제외한다. router 쪽에 복사본을 두면 어느 규칙이든 처음 바뀌는 순간 어긋난다. 응답 shaping에 대해 #2719가 내린 결정(`shape_plain_chat_reply`)과 같은 방식으로, 양쪽이 함수 하나를 호출한다.

non-stream: router는 #2719 이후 이미 handoff의 raw 텍스트와 shaping된 응답을 가지고 있으므로, `log_if_reasoning_only(&raw, &shaped.content, shaped.reasoning_content.as_deref(), completion_tokens, finish_reason)`를 호출해 결과를 `chat_completion_json`에 넘긴다. `--skip-chat-parsing`에서는 shaping된 응답에 reasoning이 없으므로 플래그는 false로 남는다.

stream: single node는 raw 텍스트만이 아니라 client가 실제로 받은 내용으로 판단한다. router도 이를 따른다. `emit_filtered`는 비어 있지 않은 reasoning delta를 보낼 때 `saw_reasoning_content`를, 비어 있지 않은 content를 보낼 때 `saw_content`를 기록한다. `--reasoning-format none`과 `deepseek-legacy`가 쓰는 thoughts-in-content 형태도 content로 센다. handoff callback은 raw 텍스트를 모은다. 종료 시점에 router는 `completed && stream_reasoning_only(&raw, saw_content, saw_reasoning_content, finish_reason)`를 계산한다. `completed` 조건은 생성 오류 시 `false`를 넘기는 single node와 맞추기 위한 것이다.

두 플래그는 `Cell<bool>`이 아니라 `Relaxed` ordering의 `AtomicBool`이다. 두 closure 모두 spawn된 task 하나 안에서 실행되므로 스레드 간 순서를 강제할 필요는 없다. 하지만 `.await`를 가로질러 `&Cell`을 들고 있으면 future가 `!Send`가 되어 `tokio::spawn`이 받아주지 않는다.

두 builder 모두 값이 true일 때만 필드를 추가하고(`message["reasoning_only"] = true`, `delta["reasoning_only"] = true`) 그렇지 않으면 생략한다. single node의 `with_reasoning_only`가 만드는 형태와 같으므로, 두 본문을 비교하는 client는 한쪽에만 `"reasoning_only": false`가 있는 상황을 보지 않는다.

### fake prefill peer가 실제 노드처럼 토큰을 보내도록 수정

router는 `first_token` frame에서 첫 토큰만 읽는다. #2719에서 만든 `router_front_tests`의 fake peer는 모든 토큰을 terminal `first_token` frame 하나에 담아 보냈는데, 기존 테스트가 모두 토큰 하나만 보냈기 때문에 문제가 드러나지 않았다. 새 `reasoning_only` 테스트는 여러 토큰(`"<think>"`, `"still"`, `" thinking"`)이 필요하므로, fake는 이제 decode 노드처럼 `tokens[0]`을 `first_token` frame에, 나머지를 `start_sequence: 1`인 terminal `continuation` frame 하나에 담아 보낸다. 토큰이 하나인 테스트는 `rest`가 비어 첫 frame이 계속 `done: true`이므로 영향이 없다.

### 첫 요청 전에 router가 모든 peer를 online으로 볼 때까지 대기

router e2e 테스트 일곱 개 모두 `wait_for_router_peers_online`을 호출한다. 이 함수는 모든 node가 `online`이 될 때까지 `GET /router/stats`를 polling한다. router의 health monitor는 시작 시 peer를 한 번 probe하고 이후 3초마다 probe한다. 첫 probe 때 아직 모델을 로드 중인 peer는 down으로 표시되고, 다음 probe 전에 보낸 요청은 `/health`와 peer 포트가 모두 살아 있어도 503을 받는다. 부하가 걸린 GB10 호스트에서 이 때문에 `disaggregated_router_rejects_model_owned_family_and_keeps_serving`이 연속 두 번 실패했다. 로그를 켠 수동 재현에서는 probe가 `prefill-0`을 bind 1초 전에 down으로 표시했다. polling 요청마다 최대 5초, 남은 테스트 deadline을 넘지 않는 timeout을 두어, 멈춘 연결이 deadline을 넘기지 못하게 했다(commit 17a05ed3).

이는 테스트 쪽 수정이다. router의 시작 구간 동작 자체는 바뀌지 않았다. 4절을 참고한다.

### e2e 비교가 공허하게 통과하지 않도록 사전 조건 추가

새 1024-token turn(`LONG_MAX_TOKENS`)은 먼저 single node reference의 non-stream 본문과 이어 붙인 stream delta 양쪽에서 `content`와 `reasoning_content`가 비어 있지 않은지 확인하고, 실패 메시지에 `LONG_MAX_TOKENS`를 늘리라고 적는다. 그다음에야 router를 reference와 비교한다. 기존 32-token turn도 router의 플래그를 비교하기 전에 single node가 두 형태 모두에서 `reasoning_only: true`를 보고하는지 확인한다. 이런 사전 조건이 없으면, thinking을 더 일찍 닫는(또는 끝내 닫지 않는) 모델이나 template 변경이 생겼을 때 양쪽이 같아져 parity 단언이 아무것도 검사하지 않고 통과한다.

## 3. 변경 요약

10개 파일, +517/-23, commit 두 개(수정은 2e2ab809, 32-token reference 확인과 polling timeout은 17a05ed3)와 origin/main merge 하나(0a64688e, #2737과 #2740을 가져왔고 이 PR의 파일은 건드리지 않음).

- `src/loading/mod.rs`: `merged_eos_token_ids`.
- `src/loading/vlm_llava.rs`: `Llama` text 분기가 병합된 EOS 목록을 설정.
- `src/distributed/tensor_parallel/llama_runtime.rs`: `TensorParallelLlamaModel::set_eos_token_ids`(모든 rank), `from_model_dir`이 이를 호출.
- `src/server/routes/chat.rs`: `log_if_reasoning_only`와 `stream_reasoning_only`가 `pub(crate)`가 됨.
- `src/server/router_front.rs`: streaming chat 경로가 `saw_content`, `saw_reasoning_content`, raw 텍스트를 기록하고 `reasoning_only`를 `chat_chunk_finish`에 넘김. non-stream 경로는 `chat_completion_json`에 넘김. 두 builder 모두 `reasoning_only: bool`을 받음.
- 테스트:
  - `generation_config_tests::merged_eos_token_ids_appends_sidecar_ids` (선언 id 우선, sidecar id 추가, sidecar가 빈 경우).
  - `llama_runtime_tests::tensor_parallel_llama_set_eos_token_ids_updates_every_rank`.
  - `vlm_llava_tests::llava_text_model_reports_tokenizer_eos` (ignored, 실제 `llava-1.5-7b-4bit`, `MLXCEL_REQUIRE_MODELS=1`이면 모델이 없을 때 panic).
  - `router_front_tests`: `router_non_stream_chat_marks_reasoning_only`, `router_stream_finish_chunk_marks_reasoning_only`, `router_reasoning_only_absent_when_an_answer_follows`. fake prefill peer가 first frame과 continuation frame을 나눠 보냄. 기존 `chat_completion_json` 호출 네 곳은 `false`를 넘김.
  - `tests/disaggregated_router_e2e.rs`: 일곱 테스트 모두 `wait_for_router_peers_online` 사용. thinking 테스트에 1024-token non-stream/stream 비교, 32-token `reasoning_only` reference 확인, 두 형태 모두에 대한 router와 reference의 `reasoning_only` 일치 확인이 추가됨.

문서는 바뀌지 않았다. router 응답 parity나 `reasoning_only`를 설명하는 문서가 없다.

## 4. 트레이드오프와 잔여 항목

- **router가 reasoning-only 경고를 로그로 남긴다.** `log_if_reasoning_only`는 true를 반환할 때 target `mlxcel::thinking`으로 `tracing::warn!`을 남긴다. 이 함수를 재사용하므로, router를 거친 thinking 응답이 잘리면 single node에서처럼 router에도 경고가 남는다. 의도한 parity지만, `max_tokens`가 작은 workload에서는 router 로그 양이 늘어난다.
- **router 시작 직후의 503 구간은 그대로다.** e2e 수정은 router를 바꾸지 않고 `/router/stats`를 기다린다. 실제 운영에서는 router의 `/health`가 정상이 된 뒤, health monitor가 느리게 로드되는 peer를 다시 probe하기 전에 도착한 요청이 최대 약 3초 동안 503을 받을 수 있다. router가 필요할 때 다시 probe해야 하는지, readiness를 늦춰야 하는지는 이 PR이 다루지 않는 별도 문제다.
- **Qwen2 backbone의 LLaVA는 여전히 `text_config`의 id만 보고한다.** `Qwen2Model`에는 `set_eos_token_ids`가 없다. stop id는 다른 곳에서 병합되므로 생성에는 영향이 없지만, 이런 checkpoint에서 `eos_token_ids()`만 읽으면 여전히 목록이 짧다.
- **1024-token 비교는 프로세스 간 결정성에 의존한다.** qwen3-0.6b의 1024-token 생성이 두 프로세스에서 바이트 단위로 같다는 것은 GB10에서 `MLXCEL_SDPA_DETERMINISTIC=1`로 확인했다. Metal이나 ROCm에서는 돌려 보지 않았다. 그쪽에서 불안정하게 실패한다면 원인은 router shaping보다 수치적 비결정성일 가능성이 높다.

## 5. 검증

**main 8385f40b에서 먼저 실패** (수정 전에 새 테스트를 작성): `llava_text_model_reports_tokenizer_eos`는 `left: [] right: [2]`로 실패했다. `router_non_stream_chat_marks_reasoning_only`는 `message.reasoning_only`가 `Null`이어서, `router_stream_finish_chunk_marks_reasoning_only`는 finish chunk가 `delta: {}`여서 실패했다. e2e thinking 테스트는 `router stream finish chunk reasoning_only does not match the single-node chat route`, `left: Null right: Bool(true)`로 실패했다. `merged_eos_token_ids`와 `set_eos_token_ids` 단위 테스트는 새 API를 쓰므로 main에서 컴파일되지 않는다.

**커버리지 공백에 대한 mutation 확인.** 2번 항목은 의도대로 main에서도 통과한다. router의 non-stream 분기가 `shape_router_chat_reply`를 호출하지 않고 handoff의 raw 텍스트를 그대로 반환하게 바꾸면 기존 32-token 단언이 먼저 실패한다. 그 실행에서 해당 단언을 우회하면 새 단언이 `router 1024-token non-stream content does not match the single-node chat route`로 실패하며, left는 raw `<think>\nOkay, so the question is asking...`, right는 `2 + 2 equals 4. ...`였다. mutation은 되돌렸고 스위트를 다시 빌드해 재실행했다.

**스위트** (GB10, `--profile test-fast --features cuda`, `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, `gpu-lock` 아래): `server::router_front::`, `server::routes::chat::tests`의 판정 함수 두 그룹, `models::llama3`, `loading::generation_config_tests`, `loading::tests`, `distributed::tensor_parallel::llama_runtime`에서 236개 통과(ignored 4개는 무관한 qwen3.5 대형 가중치 디버그 테스트). `llava_text_model_reports_tokenizer_eos --include-ignored`는 실제 checkpoint에서 통과했다. `disaggregated_router_e2e --include-ignored`는 모델 skip 없이 7/7 통과했고 최종 head에서 다시 돌렸다. `cargo fmt --all -- --check`와 `cargo clippy --profile test-fast --features cuda,test-utils -p mlxcel --lib --tests -- -D warnings`는 깨끗했다.

**검증하지 못한 항목.** `TensorParallelLlamaModel::from_model_dir`은 실제 checkpoint로 실행하지 않았다. 단위 테스트는 `from_full_weights`로 만든 모델에서 setter를 검사하므로, `from_model_dir` 안의 호출은 리뷰와 컴파일로만 확인됐다. `--reasoning-format none` / `deepseek-legacy`(thoughts가 content에 있으므로 플래그가 없어야 함)와 `--skip-chat-parsing`에서 router의 `reasoning_only` 동작은 코드로 추론한 것이며 router 테스트로 고정하지 않았다. thinking e2e 테스트에 추가된 1024-token 생성 쌍이 CI 시간에 미치는 영향은 측정하지 않았다.

## 6. 학습 포인트

- front end가 backend route와 같은 확장 필드를 내보내야 할 때는 backend의 판정 함수를 export하고 동등한 입력으로 호출한다. 이번에는 single node의 stream 판정이 "client가 받은 내용"을 기준으로 정의되어 있으므로, router 자신의 emit 지점에서 그 값(`saw_content`, `saw_reasoning_content`)을 기록해야 했다.
- 실제 프로토콜보다 단순한 테스트 fake는 테스트가 단순한 경우를 넘어서는 순간 버그를 숨긴다. fake prefill peer는 모든 토큰을 `first_token` frame 하나에 넣었고 router는 거기서 토큰 하나만 읽는다. 실제 frame 순서(first token 다음 continuation)를 맞추는 데는 몇 줄이면 되고, 이후 테스트에서 조용한 잘림이 사라진다.
- 두 실제 프로세스 사이의 parity 테스트에는 비교하는 필드가 실제로 의미 있는 값을 담는다는 사전 조건이 필요하다. 동등성을 단언하기 전에 reference가 비어 있지 않은지(또는 `reasoning_only: true`인지) 단언하면, 공허한 통과가 해결 방법이 적힌 실패로 바뀐다.
- health monitor가 일정 간격으로 peer를 polling할 때 "health가 정상"과 "router가 라우팅할 수 있음"은 서로 다른 상태다. 실제 프로세스 테스트는 router의 `/health`나 peer의 열린 포트가 아니라 router가 보는 peer 상태(`/router/stats`)를 기다려야 한다.
