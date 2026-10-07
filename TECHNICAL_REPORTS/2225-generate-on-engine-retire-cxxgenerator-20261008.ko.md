# 기술 보고서: PR #2225 - 엔진 위의 `mlxcel generate`, CxxGenerator 제거

**작성일**: 2026-10-08

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Rust, Markdown(ADR, 문서, CHANGELOG), shell

**위험도**: 높음. CLI의 모든 decode(`generate`, `--profile`, `--prompt-lookup`, inference session, decode 벤치마크)가 새 클라이언트로 옮겨 가고, decode 루프 네 개를 담은 3,000줄짜리 `CxxGenerator`가 사라진다. 확인한 체크포인트에서 greedy 출력은 그대로이고, 여섯 모델 계열에서 pipelined 실행과 동기 실행 결과가 같으며, 패리티 arm도 모두 일치한다. 다만 처리량은 epic 마지막에야 측정한다.

## 요약

Phase 5 이후에도 `mlxcel generate`와 CLI speculative 기능은 `CxxGenerator`로 decode했다. 그래서 decode를 고칠 때마다 CLI 전용 수정이 따로 필요할 수 있었고, `bench_decode`는 서버가 쓰지 않는 루프를 쟀다. epic #2166의 마지막 단계(#2176)인 이 PR은 패리티 하네스의 `DirectEngine`을 `mlxcel_core::engine::DirectEngine<M>`로 올린다. 이것이 엔진의 단일 시퀀스 raw completion 클라이언트이고, ADR 0007에서 말한 llama-completion에 해당한다. `generate`, `MlxInferenceSession`, `bench_decode`, `bench_engine`의 CLI arm, `speculative_bench`가 모두 이 클라이언트를 쓴다. 클라이언트는 엔진의 분할 step으로 B=1 decode를 pipelining한다. prompt lookup은 `Drafter`가 되고, classic draft-model 생성기는 폐기 예고 상태가 되며, `CxxGenerator`는 제거한다.

## 1. 문제 정의

- `CxxGenerator`에는 decode 루프 네 개와 log-likelihood 계산이 있었고, 서버와 `run`이 쓰는 엔진과는 별개였다.
- `bench_decode`와 패리티 하네스의 CLI arm은 그 별도 루프를 쟀다.
- prompt-lookup decoding은 CLI 전용 생성기여서 서버가 제공할 수 없었다.
- 패리티 하네스와 `bench_decode`는 chat 프롬프트를 자체 코드로 렌더했고, 이것이 서버와 달랐다(#2224에서 `enable_thinking` 기본값이 빠진 것을 찾았다).

## 2. 변경 요약

- **`DirectEngine<M>`**(`engine/direct.rs`): open, `PrefillPlan` 조각 prefill, `complete_prefill`, finish step이 나올 때까지 `step`, close 순으로 돈다. 요청의 KV 모드(Boundary-V 표, 그리고 첫 시퀀스의 캐시가 그 모드로 만들어지도록 `open` 전에 주입하는 모델 소유 모드)와 token bias를 갖는다. `impl LanguageModel for &M` 덕분에 호출자가 가진 모델을 빌려 쓸 수 있고, trait 메서드 41개를 모두 그대로 전달한다.
- **B=1 pipeline**(`engine/direct_decode.rs`, `engine/lookahead.rs`): 첫 토큰 이후에는 n번째 토큰을 읽기 전에 n+1번째 step의 forward를 `submit`, `finish_rows`, `unwind_appends`로 먼저 보낸다. 끝날 때는 남는 append 하나를 되돌린다(`Teardown::Unwind`). 되돌리기를 지원하지 않는 모델 소유 계열도 pipelining하되, 남는 step은 시퀀스를 닫을 때 버린다(`Teardown::Discard`). history penalty, DRY, mask, override, logprobs, feedback 샘플러를 쓰는 요청은 동기로 돈다. `MLXCEL_FORCE_SYNC`를 주면 동기 경로를 강제한다. 적격성 helper는 scheduler와 공유한다.
- **점수 계산**: `Engine::score`가 `evaluate_loglikelihoods`를 대신한다(같은 context slice, fp32 log-softmax, gather).
- **prompt lookup**: `PromptLookupDrafter`가 서버의 `Drafter` trait을 구현하고, 루프는 새로 추가한 `Engine::verify`와 `Engine::commit_appends` 위에서 도는 `DirectEngine::generate_with_drafter`다. plain round는 예전 루프처럼 pipelining한다. `PromptLookupGenerator`는 지웠다.
- **chat template 진입점 하나**(`server::chat_front`)를 서버, `generate`, 패리티 하네스, `bench_decode`가 함께 쓴다.
- **폐기 예고**: `SpeculativeGenerator`에 `#[deprecated]`를 달고, 실행하면 안내를 출력하며, CHANGELOG Unreleased에 v0.8.0 제거 예정으로 기록했다.
- **벤치마크**: `bench_decode`는 CSV 스키마를 유지하고 마지막에 `decode_path` 열을 더한다.
- **제거**: `CxxGenerator`와 그 helper, 예전 루프만 읽던 진단용 환경 변수 다섯 개, 쓰이지 않던 `plan_single_sequence_prefill`.

## 3. 기술적 선택과 그 이유

- **네 번째 루프를 새로 쓰지 않고 probe의 클라이언트를 올린다.** `DirectEngine`은 이미 패리티 하네스의 `d:engine` arm이었다. 이것을 CLI 클라이언트로 쓰면 하네스의 CLI arm과 실제 CLI가 같은 코드가 된다.
- **B=1 decode를 pipelining한다.** 이 PR의 첫 버전은 동기로 decode했다. 예전 루프와 scheduler는 다음 forward를 호스트 읽기와 겹쳐 돌리므로, 동기 클라이언트로는 ADR 0007의 1.0% 허용치를 넘길 가능성이 컸다. 클라이언트는 별도 상태 기계를 만들지 않고 엔진의 분할 step을 재사용한다. scheduler의 상태 기계는 admission, preemption, prompt-cache 기부까지 다뤄야 해서 따로 둔다.
- **거부하지 않고 버린다.** 보안 검토에서, 정확한 되돌리기를 요구하면 Gemma 4, Llama 4, Qwen 3.5, SSM hybrid, 대부분의 VLM wrapper가 동기 경로로 빠진다는 것을 찾았다. 예전 루프는 이들도 pipelining했다. `DirectEngine`을 쓰는 곳은 모두 decode가 끝나면 시퀀스를 바로 닫으므로, 종료 지점을 넘어 쓴 step 하나는 닫을 때 버리면 된다. 서버는 정확한 되돌리기 규칙을 그대로 쓴다.
- **KV 모드는 `open` 전에 넣는다.** `Engine::open`이 부르는 `prepare_sequence_state`는 모델 소유 계열의 캐시를 그 순간 모델이 가진 모드로 만든다. 모드를 그 뒤에 넣으면, 새로 로드한 모델의 첫 시퀀스는 `--kv-cache-mode int8|turbo4`를 줘도 FP16으로 돌았다.
- **prompt lookup을 `Drafter`로 만든다.** 서버의 MTP, DFlash drafter와 같은 인터페이스를 쓴다. 서버 옵션으로 제공하려면 burst arm이 따로 필요해서 후속 과제로 남겼다.
- **classic draft-model 경로는 지우지 않고 폐기 예고만 한다.** 서버는 이 경로를 쓴 적이 없지만, 바로 지우면 사용자가 예고 없이 깨진다.

## 4. 검증

- clippy `-D warnings`(루트는 examples까지, mlxcel-core는 테스트까지), fmt, 라이선스 헤더, `vlm_wrapper_capability_delegation`, `dead_doc_pointers`, `model_loader_stdout_guard`, `cli_help_consistency`, `llama_model_source_cli`.
- `--test-threads=1` 모듈 테스트: core는 `engine`(모든 종료 경우에서 pipelined와 동기 실행을 비교하는 `direct_decode_tests`, `speculative_plain_tests` 포함), `session`, `speculative::`, `drafter::`, `generate::`, sampling, prefill, decode_finish, `cache::attend`, `language_model_ref`. 루트는 `engine_probe`, `backend::`, `server::batch::scheduler`, `lookahead`, `server::in_process`, chat 라우트, `cli_input`, `lang_analyzer`, gemma3, gemma4, nanochat, `prefill_span_coverage`, `multimodal::`, 그리고 바이너리의 `commands::`.
- 실제 체크포인트, release 빌드, `MLXCEL_SDPA_DETERMINISTIC=1`:
  - Qwen3-1.7B, Llama-3.2-1B, Gemma-4-E2B, Granite-4.0-H-350M, Qwen3.5-0.8B, Gemma-3-1B에서 `generate --temp 0 -n 96`의 출력이 pipelined일 때와 `MLXCEL_FORCE_SYNC=1`일 때 같았다.
  - Llama-3.2-1B에서 `--prompt-lookup` 출력이 plain decode와 같았다.
  - `--profile`은 `[TTFT]` 줄을 하나 출력했다. `--no-chat-template`, Qwen2.5-VL 이미지 프롬프트, Gemma 3 int8 KV `--profile`, `bench_decode` 시험 실행도 모두 정상 종료했다.
  - Gemma 3를 int8과 turbo4로 `--profile` 실행하면 이제 각 모드의 일반 경로와 같은 출력이 나온다. 수정 전에는 FP16 출력이 나왔다.
  - `mlxcel-engine-parity`: CLI, dense, paged, engine, `run -p`, 서버 arm이 모두 같았고, 문서화된 prompt-cache 행만 갈라졌다.
  - `single_row_batch_parity`와 `scheduler_real_batch_parity`도 여전히 통과한다.

## 5. 남은 위험

- **처리량은 아직 재지 않았다.** epic 마지막 측정에서 epic 이전의 `CxxGenerator` 빌드와 `generate`를 1.0% 허용치 기준으로 비교한다.
- **penalty나 DRY를 쓰는 요청은 동기로 돈다.** 예전 루프는 행별 샘플러로 이 경우도 겹쳐 돌렸다. `generation_config.json`에 penalty가 설정된 체크포인트는 행별 `submit`이 생길 때까지 이 겹침을 잃는다.
- **seed를 준 샘플링 결과는 경로에 따라 다르다.** pipelined fused draw는 행별 체인과 RNG를 다르게 소비한다. 서버의 동기 tick과 lookahead tick도 이미 그렇다.
- **실제 체크포인트로 돌려 보지 못한 것**: classic draft-model 폐기 안내(로컬에 draft 모델이 없다), Metal capture 스크립트, Llama 4, Inkling.
- **`Teardown::Discard`에서 submit이 실패하면** 동기 경로로 넘어가지 않고 오류로 끝난다. 이 경로는 리뷰로만 확인했고 stub 테스트는 없다.

## 6. 학습 포인트

- **상태를 만드는 시점을 바꾸는 리팩터링은, 그 시점에 누가 어떤 입력을 읽는지 확인해야 한다.** KV 모드 버그는 캐시 생성을 `open` 안으로 옮기면서 모드 주입은 그 뒤에 남겨서 생겼다. `prepare_sequence_state`가 본 모드를 기록하는 테스트로 잡았다.
- **'정확히 되돌릴 수 있어야 한다'는 조건은 한 번 호출하고 끝나는 클라이언트에는 과하다.** scheduler는 시퀀스가 계속 살아 있으므로 되돌려야 하지만, decode 후 시퀀스를 닫는 클라이언트는 남는 step을 버리고 pipeline을 유지하면 된다.

## 7. 관련 항목

- epic #2166, 이슈 #2176, ADR 0007, ADR 0009.
- #2217(엔진 step API), #2224(프로세스 내 서버 클라이언트), #2182(decode-undo 로그), ggml-org/llama.cpp#17824(llama-cli가 서버로 옮겨 간 뒤의 llama-completion).
