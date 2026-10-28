# PR #2719: #2700에서 남은 disaggregated router parity 잔여 항목 해소

**날짜**: 2026-10-28 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust, Markdown
**위험도**: 낮음에서 중간 (wire 필드 하나를 추가하는 변경이며, `LoadedModel::Llama`로 로드되는 모든 checkpoint의 모델 수준 EOS 목록이 바뀜)

## 요약

PR #2700 (issue #2695)은 실제 프로세스로 띄우는 disaggregated e2e 스위트를 통과시켰고, router가 single node와 여전히 다른 세 곳을 잔여 항목으로 남겼다. 이 PR (issue #2715)은 세 항목을 모두 닫는다. router의 non-stream chat 응답은 이제 single node의 no-tools 응답과 같은 helper(`routes::chat::shape_plain_chat_reply`)로 만들어지므로, MiniCPM의 `" Café"`는 양쪽 모두 `"Café"`가 된다. `Llama3Model::eos_token_ids`는 이제 checkpoint의 `eos_token_id`에서 나오고, Llama 3 id인 128001과 128009는 어휘 크기가 128256인 Llama 3 vocabulary일 때만 덧붙는다. 그래서 llama 형식의 MiniCPM-2B는 어휘 밖의 id 두 개 대신 `[2]`를 보고한다. `ResultFrame`에는 선택 필드 `prompt_cache_enabled`가 추가되고 prefill 노드가 성공한 first-token frame에 이 값을 채운다. 그 결과 router의 `usage.prompt_tokens_details`는 router 자신의 플래그가 아니라 worker의 prompt cache 설정을 따른다.

## 1. 문제 정의

### non-stream chat content에 앞 공백이 남음

MiniCPM "café" 케이스에서 router는 `" Café"`, single node는 `"Café"`를 반환했다. 원인은 detokenization이 아니었다. completions 본문은 이미 바이트 단위로 일치했고, scheduler 텍스트는 양쪽 모두 `" Café"`였다. 두 경로는 응답을 만드는 단계에서 갈라졌다. single node는 raw 텍스트를 `clean_structural_tokens`(마지막에 `clean_content_markers`의 `.trim()`을 거침), `<answer>` wrapper 제거, `strip_unclosed_primed_thinking`, `shape_response`에 차례로 통과시켰다. router의 non-stream 분기는 `StreamFilter` 출력을 이어 붙이기만 하고 이 단계를 거치지 않았다. `disaggregated_router_chat_usage_matches_single_node_byte_fallback`은 `usage`와 `finish_reason`만 비교했기 때문에 이 차이를 잡지 못했다.

### `Llama3Model`의 EOS id가 Llama 3 값으로 하드코딩됨

`eos_token_ids()`는 `llama3`로 로드되는 모든 checkpoint에 `[128001, 128009]`를 반환했다. llama 형식의 MiniCPM-2B(`vocab_size` 122753, `eos_token_id` 2)도 그중 하나이며, 두 id 모두 이 모델의 어휘 밖이다. 트리 안의 모든 생성 경로(scheduler admission, handoff의 prefill과 decode 경로, CLI, `bench_decode`, pipeline runtime)는 `read_eos_token_ids`로 sidecar EOS를 병합하므로, 실제로 잘못 멈추는 경로는 없었다. 그래도 병합하지 않는 소비자, 즉 `src/vision/mod.rs`의 llava 계열 wrapper와 `llama_runtime.rs`의 tensor-parallel runtime에게는 모델 자체의 답이 틀린 상태였다.

### `prompt_tokens_details`가 router의 플래그를 따름

single node는 자신의 prompt cache store가 있을 때에만 `usage.prompt_tokens_details`를 넣는다. disaggregated 배포에서 그 store는 worker에 있고 router는 store를 만들지 않는데도, router의 usage builder 네 개(chat과 completions, stream과 non-stream)는 모두 router의 `state.config.prompt_cache.is_enabled()`로 판단했다. 기본 설정의 router 뒤에서 worker를 `--no-prompt-cache`로 띄운 클러스터는 single node라면 아무것도 넣지 않을 자리에 `{"cached_tokens": 0}`을 보고했다.

## 2. 기술적 선택과 그 이유

### router에서 trim하지 않고 shaping 함수를 공유

single node의 no-tools shaping은 `plain_chat_reply(raw, primed_open_thinking, answer_wrapper, reasoning_format) -> (String, ShapedResponse)`로 옮겨졌다. `shape_plain_chat_reply`는 `ShapedResponse`만 돌려주는 `pub(crate)` wrapper다. tuple 형태가 필요한 이유는 single node route가 `--reasoning-format` 배치 전의 정리된 answer를 다음 turn의 prompt cache warm-up(#1144)에도 넘기기 때문이다. `extract_reasoning_content`도 함께 옮겨졌다. no-tools 경로는 helper 안에서 reasoning을 계산하고 tool-call 분기는 따로 계산하므로, 어느 경로도 두 번 계산하지 않는다.

issue는 router가 조립한 content만 trim하는 좁은 수정을 기각했다. 그렇게 하면 MiniCPM 케이스는 맞지만 `--reasoning-format none`, primed open thinking 블록, `<answer>` wrapper에서는 다시 갈라진다. 각각이 trim 대상 자체를 바꾸기 때문이다. 양쪽이 같은 함수 하나를 호출하면 drift는 관례가 아니라 컴파일 수준의 결합으로 막힌다.

router의 non-stream 분기는 이제 handoff의 raw 텍스트를 그대로 모은 뒤 `shape_router_chat_reply`를 호출한다. 이 함수는 `--skip-chat-parsing`이면 raw 텍스트를 그대로 반환하고(single node도 shaping 전에 `result.text`를 반환한다), 아니면 `shape_plain_chat_reply`를 호출한다. 이 분기에 있던 `StreamFilter`와 `ThinkingDelimiterEcho` 코드는 제거됐다. single node의 streaming도 trim하지 않으므로 streaming 분기는 바꾸지 않았다.

### EOS id는 config에서, Llama 3 기본값은 vocabulary로 제한

`llama3::ModelArgs`에 `#[serde(default)] eos_token_id: Option<EosTokenId>`가 추가됐다. `helium::TokenIdField`를 본뜬 로컬 untagged int-or-list enum이다. serde는 한 필드의 형태가 맞지 않으면 config 전체를 실패시키므로 두 형태를 모두 받아야 한다. `ModelArgs::eos_token_ids()`는 선언된 id를 순서대로 중복 없이 반환하고, `vocab_size == 128256`일 때에만 128001과 128009 중 빠진 것을 덧붙인다. `from_weights`는 그 결과를 새 private 필드 `Llama3Model.eos_token_ids`에 저장하고, `set_eos_token_ids`로 호출자가 값을 바꿀 수 있다. `loading::special`의 기존 패턴을 따른 것이다.

vocabulary 조건 덕분에 Llama 3 checkpoint는 기존 동작을 유지하고, 다른 모델에는 id를 지어내지 않는다. 128001은 더 큰 non-Llama vocabulary에서는 평범한 토큰이므로 무조건 덧붙이면 일반 토큰에서 생성이 끝난다. `nemotron_nas::NemotronNASConfig`에 문서화된 바로 그 위험이다. 아무것도 선언하지 않은 non-Llama-3 checkpoint는 빈 목록을 받는다. 이 PR 이전에는 샘플링될 수 없는 id 두 개(MiniCPM)나 일반 텍스트로 샘플링될 수 있는 id 두 개(더 큰 vocabulary)를 받았으므로, 빈 목록이 어느 경우에도 더 나쁘지 않다.

#2695의 decode-intake 회귀 테스트(`serving_handoff_parity_stops_on_config_eos_minicpm`)는 예전의 틀린 답을 전제로 했다. 모델이 `[128001, 128009]`를 보고하고 `2`는 config 병합으로만 들어온다는 전제였다. 이제 테스트는 먼저 모델이 `[2]`를 보고하는지 확인하고, `set_eos_token_ids(vec![128001, 128009])`로 전제를 복원한 다음, decode 경로가 config EOS를 병합하는지를 여전히 검증한다.

### worker의 prompt cache 설정을 wire로 보고

`ResultFrame.prompt_cache_enabled: Option<bool>`은 `generated_tokens`의 mixed-version 선례를 따라 `#[serde(default)]`다. frame은 JSON(`serde_json`)이고 `deny_unknown_fields`가 없으므로, 이전 router는 새 키를 무시하고 새 router는 키가 없으면 `None`으로 읽는다. prefill coordinator는 `handoff_supported`처럼 시작 시 `BatchScheduler::has_prompt_cache_store()`를 한 번 읽고, 성공한 first-token frame에만 필드를 채운다(`drained.error.is_none().then_some(..)`). error frame과 decode continuation frame은 `None`이다. `drive_handoff_result`는 첫 frame에서 이 값을 받아 `HandoffOutcome`에 담고, `HandoffOutcome::reports_prompt_cache(router_flag)`가 `prompt_cache_enabled.unwrap_or(router_flag)`로 결정한다. usage builder 네 개가 모두 이 메서드를 쓰므로 이전 버전의 prefill 노드에서는 기존 동작이 유지된다.

기각된 대안은 router 플래그를 계약으로 문서화하는 것이었다. 운영자는 prompt cache를 worker 단위로 설정하고, wire 필드는 선택적 bool 하나의 비용뿐이다.

## 3. 변경 요약

파일 16개, +821/-132. 커밋 두 개(수정은 bdeb91b4, 잘못 놓인 `helium.rs` 주석 이동은 670034c1)와 origin/main merge(164eaeed, #2718의 media 정책 파일만 들어옴)로 구성된다.

- `src/server/routes/chat.rs`: `plain_chat_reply`와 `shape_plain_chat_reply` 추가. no-tools 분기가 이를 호출하고, reasoning 추출은 tool-call 분기와 helper 안으로 이동.
- `src/server/router_front.rs`: non-stream chat 분기가 raw 텍스트를 모아 `shape_router_chat_reply`를 호출. `HandoffOutcome.prompt_cache_enabled`와 `reports_prompt_cache` 추가. chat과 completions의 usage 경로(stream usage chunk 두 곳, non-stream 본문 두 곳)가 이를 사용. `router_usage_json`과 `completion_response`의 doc comment 갱신.
- `src/distributed/disaggregated/serving_protocol.rs`, `coordinator.rs`: 새 frame 필드. 성공한 first-token frame에 채우고 나머지 frame 생성 여섯 곳은 `None`.
- `src/server/batch/scheduler/handoff.rs`: `has_prompt_cache_store()`.
- `src/models/llama3.rs`: `EosTokenId`, `LLAMA3_VOCAB_SIZE`, `LLAMA3_EOS_TOKEN_IDS`, `ModelArgs::eos_token_ids`, 저장 필드와 setter. `helium.rs`와 테스트 struct literal 세 곳에 `eos_token_id: None` 추가.
- `docs/CONTINUOUS_BATCHING.md`: `prompt_tokens_details` 설명을 worker 기준 규칙과 fallback으로 바꾸고, non-stream shaping parity 항목을 추가. 이 문서의 한국어판은 없다.
- 테스트: `llama3_tests`의 EOS 케이스 다섯 개, 키가 없는 frame을 `None`으로 읽는 `serving_protocol_tests` 케이스, `router_front_tests`의 fake prefill peer(`spawn_fake_prefill`, `router_with_fake_prefill`)와 그것을 쓰는 테스트(shaping된 non-stream 응답, reasoning format별 shaping helper, `prompt_tokens_details` 세 경우: worker off에 router on, worker on에 router off, worker `None`)를 chat, completions, 두 usage chunk에 걸쳐 검증. MiniCPM e2e 테스트는 케이스마다 non-stream `content`와 stream 연결 결과를 비교하고, qwen3 thinking e2e 테스트에는 `content`와 `reasoning_content`의 non-stream 비교가 추가됨.

## 4. 트레이드오프와 잔여 항목

- **모델 수준 EOS 목록이 바뀜.** Llama 3.1 Instruct checkpoint는 `[128001, 128008, 128009]`를 선언하므로 `Llama3Model`은 이제 128008(`<|eom_id|>`)도 보고한다. 트리 안의 생성 경로는 이미 sidecar 파일에서 이 값을 병합했으므로 차이가 보이는 곳은 병합하지 않는 소비자(llava 계열 backbone, TP runtime)뿐이고, 그쪽도 checkpoint와 맞춰진다. vocabulary 크기를 바꾼 Llama 3 파생 모델(128256이 아닌 경우)에는 더 이상 128001/128009가 덧붙지 않고 config에 선언된 값만 보고된다.
- **빈 목록이 나올 수 있음.** `config.json`에 `eos_token_id`가 없고 vocabulary가 Llama 3가 아닌 llama 형식 checkpoint는 이제 `[]`를 보고한다. vision wrapper와 TP runtime은 여전히 `generation_config.json`이나 tokenizer EOS id를 병합하지 않으므로, 그런 checkpoint에서는 사실상 이전과 마찬가지로 `max_tokens`에서만 멈춘다. 이 소비자들이 sidecar id를 병합하게 만드는 일은 이 PR의 범위 밖이다.
- **prefill 노드가 쌍 전체를 결정.** 설정을 보고하는 것은 prefill 노드뿐이다. prefill 노드와 decode 노드의 `--no-prompt-cache` 설정이 다르면 prefill 노드의 설정이 `prompt_tokens_details`를 결정한다. single node에는 store가 하나뿐이므로 이 분리 상황에서 맞춰야 할 single node 동작은 없다.
- **router non-stream의 `--reasoning-format none`, `deepseek-legacy`가 이제 single node의 non-stream 형태를 따름.** 이전 분기는 router의 두 응답 형태가 일치하도록 `ThinkingDelimiterEcho`를 썼다(#2464). 새 분기는 single node non-stream처럼 `content_with_thinking_block`으로 thought를 다시 구성한다. 이제 router의 stream과 non-stream 형태는 single node에서 일치하는 만큼만 일치한다.
- **`reasoning_only`를 내보내지 않음.** single node의 non-stream 응답은 응답 전체가 reasoning일 때 확장 필드 `reasoning_only: true`를 담지만 router는 담지 않는다. PR 본문에 후속 작업으로 기록되어 있다.

## 5. 검증

**752db4f9에서 먼저 실패 확인** (수정 전에 작성한 새 테스트): MiniCPM e2e chat 테스트는 `left: String(" Café")`, `right: String("Café")`로 실패했고, fake prefill router 테스트도 같은 방식으로 실패했다. `prompt_tokens_details_omitted_when_the_worker_cache_is_off`는 `left: Some({"cached_tokens":0})`, `right: None`으로, "present" 케이스는 `left: None`으로 실패했다. MiniCPM handoff parity 테스트는 `left: [128001, 128009]`, `right: [2]`로 실패했다. `None` fallback 테스트와 qwen3 non-stream 비교는 설계상 main에서도 통과한다. `llama3_tests`의 EOS 케이스, shaping helper 테스트, `serving_protocol`의 키 누락 케이스는 새 API를 쓰므로 main에서는 컴파일되지 않는다.

**스위트** (GB10, `--profile test-fast --features cuda`, `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, `gpu-lock` 아래): `server::router_front` 40/40, `server::routes::chat` 73/73, `models::llama3` 34/34, `models::headless_llama` 3/3, `distributed::disaggregated::serving_protocol` 19/19, `serving_handoff_parity --include-ignored` 6/6, `models::llama_bidirec` 9/9, `models::helium` 28/28, `distributed::tensor_parallel::llama_runtime` 35 통과(4 ignored). `disaggregated_router_e2e --include-ignored` 7/7(49초, soft-skip 없음), `disaggregated_serving_e2e` 2/2. `cargo fmt --check`와 `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`는 깨끗했다. helium 주석 커밋 뒤에는 fmt, clippy, `models::helium`을 다시 돌렸다.

**실제 서버**: `minicpm-2b-sft-4bit-llama-format-mlx`를 올린 single node `mlxcel-server`(127.0.0.1:19701), greedy, `max_tokens` 16. "café" 프롬프트는 `finish_reason: stop`, `completion_tokens: 1`로 `"Café"`를 반환했고, "Чай" 프롬프트는 `stop`, 4 토큰으로 `"Чай"`를 반환했다. 둘 다 `</s>`에서 멈췄다.

**검증하지 못한 것.** qwen3 e2e 비교는 main에서도 통과한다. 32 토큰에서는 양쪽 모두 `content: ""`와 같은 reasoning을 반환하기 때문이다. reasoning과 answer가 모두 있는 응답을 router를 거쳐 비교하는 실제 모델 실행은 없다. `shape_router_chat_reply_matches_the_single_node_shaping`은 router helper를 그것이 위임하는 함수와 비교하므로, `--reasoning-format none`과 `deepseek-legacy` 형태는 helper 수준에서만 고정되어 있고 실제 프로세스 router 실행으로는 확인되지 않았다. mixed-version 클러스터(이전 router와 새 prefill, 또는 그 반대)는 돌려보지 않았고, 호환성은 JSON 인코딩과 `serde(default)` 단위 테스트에 기대고 있다. 스위트는 origin/main merge(164eaeed) 전인 bdeb91b4/670034c1에서 돌았고, 이 보고서를 쓰는 시점에는 merge 커밋에 대한 pre-merge 테스트가 진행 중이었다.

## 6. 학습 포인트

- 두 route가 같은 바이트를 반환해야 한다면 shaping을 양쪽이 호출하는 함수 하나로 뽑아내고, parity는 버그 보고서가 언급한 필드가 아니라 응답 본문 전체로 테스트한다. #2700의 e2e 테스트는 `usage`와 `finish_reason`만 비교해서 눈에 보이는 `content` 차이를 놓쳤다.
- 모델 수준 fallback 상수에는 그 상수가 어느 checkpoint의 것인지 말해주는 조건이 필요하다. Llama 3 id를 128256 vocabulary로 제한하면 Llama 3의 기존 동작은 유지되고, 같은 id가 평범한 토큰인 다른 vocabulary로 상수가 새어 나가지 않는다.
- 예전의 틀린 답을 전제로 한 회귀 테스트는 먼저 바로잡힌 값을 확인하고 전제를 명시적으로 복원(`set_eos_token_ids`)해야, 원래 테스트하려던 경로를 계속 테스트한다.
- front end가 보고하는 worker 단위 설정은 `serde(default)`가 붙은 `Option<T>`로 wire에 싣고 `None`은 기존 동작으로 해석한다. 이전 송신자도 계속 동작하고, fallback은 builder 네 곳에 흩어지지 않고 메서드 하나(`reports_prompt_cache`)에 모인다.
