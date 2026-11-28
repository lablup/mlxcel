# 기술 보고서: PR #2256 - 엔진 B=1 클라이언트에서 penalty와 DRY 요청 pipelining

**작성일**: 2026-10-10

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Rust, Markdown(ADR, CHANGELOG)

**위험도**: 중간. history penalty, DRY, mirostat, adaptive-p, extended chain을 쓰는 `mlxcel generate` 요청이 모두 동기 루프에서 새 pipelined 루프로 옮겨 간다. 단위 테스트와 여섯 모델 계열의 실제 체크포인트에서 출력이 동기 루프와 토큰 단위로 같고, `MLXCEL_FORCE_SYNC=1`은 여전히 이전 루프를 고른다. 처리량은 #2255와 함께 실행 마지막에 측정한다.

## 요약

#2176 이후 `DirectEngine`(`mlxcel generate`, `MlxInferenceSession`, decode 벤치마크가 쓰는 클라이언트)은 sampler가 fused device draw에 맞는 요청만 pipelining했다. repetition, frequency, presence penalty, DRY, mirostat, adaptive-p, extended chain을 쓰는 요청은 토큰마다 기다린 뒤에야 다음 forward를 인코딩하는 동기 루프로 떨어졌고, 은퇴한 CLI 루프 기준으로 이 대기는 decode 시간의 약 12퍼센트였다. 이 PR은 이런 요청에 per-row lookahead pipeline을 추가한다. 아직 읽지 않은 device 토큰으로 다음 forward를 제출하고, 그 forward가 도는 동안 토큰을 읽어 commit한 다음, commit된 history로 다음 draw를 만든다. 원래 이슈의 Part 2(서버 prompt lookup)는 #2255로 옮겼다.

## 1. 문제 정의

- `DirectEngine::decode`는 fused pipeline(`decode_pipelined`)과 `decode_sync` 중 하나를 골랐다. fused 경로 적격 여부는 `client_fused_params`가 정하며, 이전 토큰을 호스트에서 알아야 하는 sampler(history penalty, DRY)나 자기 상태로 되먹임하는 sampler(mirostat, adaptive-p)는 거부한다.
- fused 경로를 그대로 넓힐 수는 없다. 토큰 t+1의 penalty draw는 history에 토큰 t가 있어야 하므로, fused draw처럼 device에서 forward에 얹어 보낼 수 없다.
- forward는 draw의 호스트 값이 아니라 device 토큰에만 의존한다. 은퇴한 `CxxGenerator::sample_next_step`은 이 점을 이용해 penalty 요청에서도 forward를 앞서 보냈는데, #2176이 `generate`를 엔진으로 옮기면서 이 overlap이 빠졌다.
- 이슈는 `mlxcel generate`가 체크포인트의 `generation_config.json`에서 penalty를 이어받는다고 가정했지만 그렇지 않다. `GenerationConfigDefaults`(`src/loading/mod.rs`)는 EOS id, temperature, top_p, top_k만 읽는다. 새 경로로 들어오는 요청은 플래그로 직접 지정한 것뿐이다. 이어받기를 추가하면 출력이 바뀌므로 넣지 않았다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `engine/mod.rs` | `Engine::submit_forward`(draw 없는 `submit`의 speculative forward, `async_eval`로 스케줄하고 lazy logits를 반환)와 `Engine::draw_row`(그 logits에서 한 row의 per-row chain draw, token bias는 원래 위치, lazy 스케줄). 두 submit은 `speculative_forward`와 `schedule`을 공유하므로 `submit` 동작은 바뀌지 않는다. `draw_row`는 logit mask, token override, logprobs가 있는 row를 `EngineError::Batch`로 거부한다. 이들은 draw를 확정하기 전에 호스트 토큰이 필요하기 때문이다. |
| `engine/direct_decode_rows.rs`(신규) | `DirectEngine::decode_pipelined_rows`: 대기 중인 draw를 입력으로 forward n+1을 제출하고, `try_eval` 후 대기 토큰을 읽고, `finish_rows`로 commit(penalty history에 추가)한 다음, n+1의 `draw_row`를 만든다. speculative append는 최대 두 개다. |
| `engine/direct_decode.rs` | `lookahead_teardown`이 `decode_lookahead`에서 unwind/discard 규칙을 분리한다. `decode`는 fused pipeline, per-row pipeline(`decode_row_lookahead`: fused 비적격, `lang_bias_counters` 꺼짐), `decode_sync` 순으로 시도한다. 두 pipeline 모두 `DecodeCommandBufferBudget`을 잡는다. |
| `src/bin/bench_engine.rs` | `--repetition-penalty` 플래그. `main`과 테스트가 함께 쓰는 `bench_sampling_config`로 연결한다. |
| 문서 | ADR 0009(시그니처 표, raw-completion 단락), 모듈 문서, CHANGELOG Unreleased. |
| 테스트 | `direct_decode_rows_tests.rs`(376줄): repetition, frequency+presence, DRY, DRY와 windowed penalty, seed 지정 mirostat, seed 지정 adaptive-p에 대해 EOS, `max_tokens`(추가 forward 없음), callback 중단에서 스트림과 최종 상태가 `decode_sync`와 같음. penalty가 실제로 logits를 바꾸는 `NoisyModel` stub. forward 횟수로 경로를 확인하는 테스트(5회, 동기 루프는 4회)로 main에서는 통과할 수 없다. 라우팅, model-owned unwind/discard 테스트, `draw_row` 거부 테스트. |

파일 11개, +755 / -67.

## 3. 기술적 선택과 그 이유

**forward는 앞서고 draw는 뒤따른다.** pipeline은 forward를 호스트 읽기 및 finish 단계와 겹친다. draw와는 겹치지 않는다. t+1의 draw는 동기 step과 같은 logits, history, sampler 상태로 만들어지고 forward는 난수를 쓰지 않으므로, greedy와 seed 지정 스트림이 `decode_sync`와 같다. 대안인 device 쪽 penalty 커널(history를 device에서 갱신)은 draw까지 forward에 얹을 수 있지만 모든 sampler를 device에 중복 구현해야 하고, overlap을 되찾는 데는 필요하지 않았다.

**overlap 정도는 sampler에 따라 다르다.** penalty, DRY, adaptive-p 없는 extended chain은 draw를 lazy로 유지하므로 draw와 finish가 모두 다음 forward와 겹친다. mirostat와 adaptive-p는 sampler 안에서(`sampling.rs`) 균등 난수 draw를 호스트로 읽으므로 finish 단계만 겹친다. 문서에 이 내용을 적었고, 예정된 측정은 `--repetition-penalty`만 다룬다.

**teardown은 fused pipeline을 따른다.** 종료 토큰이나 callback 중단 뒤에는 commit되지 않은 append 하나, `max_tokens`를 소진한 토큰 뒤에는 없음, `Teardown::Unwind`에서 submit이나 draw가 실패하면 마지막 commit 토큰부터 동기 루프로 이어 간다. `Teardown::Discard`(되감기 못 하는 model-owned 계열)에서는 추가 append를 되돌릴 수 없으므로 draw 실패 시 오류로 실행을 끝낸다.

**호스트 읽기 앞의 `try_eval`(리뷰 수정 ea1a45e4).** 처음 버전은 실패를 보고할 수 없는 `ffi::item_i32`로 대기 토큰을 읽었다. 그래서 그 대기 지점에서 비동기 백엔드 오류가 드러나면 실행이 실패하지 않고 프로세스가 종료됐다(#822). 이제 `try_eval(&pending)`을 먼저 실행한다. 이 호출은 그 배열 하나만 기다리므로 다음 forward와의 overlap은 유지된다.

**`lang_bias_counters`가 켜지면 동기로 남는다.** B9 pre-bias 카운터는 draw마다 호스트에서 bias 적용 전 argmax를 읽으므로 어차피 pipeline을 직렬화한다.

## 4. 검증

- 최종 브랜치의 로컬 게이트(GB10, `--profile test-fast --features cuda`, `gpu-lock`, `--test-threads=1`): fmt, clippy `-D warnings`(루트 lib/tests/bins/examples와 mlxcel-core), mlxcel-core `engine::`(46), `session`, `sampling`, `generate::`, `drafter::`, `speculative::`, `decode_finish`, 루트 `engine_probe`, `backend::`, `server::batch::scheduler`, `server::in_process`, `cli_input`, `commands::`, `mlxcel-bench-engine` 단위 테스트, `dead_doc_pointers`, `model_loader_stdout_guard`, `cli_help_consistency`, Apache 헤더 검사. 모두 통과했다(mlxcel-core `session` 12, `sampling` 202, `generate::` 38, `drafter::` 223, `speculative::` 161, 루트 `server::batch::scheduler` 187, `commands::` 202, `mlxcel-bench-engine` 12).
- 실제 체크포인트, release 빌드, `MLXCEL_SDPA_DETERMINISTIC=1`: per-row pipeline의 `mlxcel generate` 출력이 `MLXCEL_FORCE_SYNC=1` 출력과 12개 경우 모두 같다. Qwen3-1.7B(repetition penalty, DRY, `--repeat-last-n 32`를 준 penalty와 DRY, greedy, 모두 `--show-reasoning`), Llama-3.2-1B(penalty, DRY, temperature 0.8 seed 지정 penalty), Gemma-3-1B, Gemma-4-E2B(model-owned 상태), Granite-4.0-H-350M과 Qwen3.5-0.8B(되감기 못 하는 hybrid 상태, `Teardown::Discard` 경로), Hunyuan-1.8B(seed 지정 penalty), 각 160 토큰.
- 패리티 하네스(`mlxcel-engine-parity -n 400`), Qwen3-1.7B, Llama-3.2-1B, Gemma-3-1B 4-bit: CLI arm(`a:cli`, seeded-penalties-DRY 행에서 이제 이 pipeline을 탄다)이 모든 행에서 server dense, server paged, direct engine과 같다. 달라진 쌍은 이 PR이 건드리지 않은 서버 prompt-cache 행뿐이다. Qwen3와 Gemma 3는 #2225 실행과 정확히 같고, Llama에서는 prompt-cache hit 행의 불일치가 seeded 행(토큰 25)에서 greedy 행(토큰 213)으로 옮겨 갔다.
- 리뷰: MEDIUM 1건(실패를 보고할 수 없는 호스트 읽기, ea1a45e4에서 수정)과 LOW 5건. 보안 리뷰에서 새 문제는 없었다. LOW 2건은 finalizer가 고쳤다(문서의 overlap 표현, 실제 연결 경로를 타는 벤치 플래그 테스트).

## 5. 남은 위험

- draw를 만든 뒤 commit하기 전에 백엔드 오류가 나면 `decode_sync`가 그 토큰을 다시 draw하므로 mirostat, adaptive-p, seed RNG 상태가 두 번 진행될 수 있다. 실행은 계속되지만 동기 스트림과 달라진다. 백엔드 오류 뒤에만 생기며 #2258에서 다룬다.
- 기존 경로에 실패를 보고할 수 없는 호스트 읽기가 남아 있다. fused B=1 pipeline, 서버 decode tick, prompt-lookup plain round, mirostat/adaptive-p sampler 읽기다(#2258). 실패 경로 테스트도 아직 없다(#2258).
- 전체 history에 대한 DRY는 step마다 위치 맵을 다시 만들고 반복 출력에서 제곱 시간 스캔을 하며, DRY와 frequency/presence는 step마다 vocab 크기 버퍼를 할당한다(#2259). 이 PR로 이 비용이 늘지는 않고, 다음 forward 뒤에 가려진다.
- 처리량은 아직 측정하지 않았다. 실행 마지막 측정에서 이 브랜치와 플래그를 손으로 넣은 pre-epic `4c44e317`을 `--path cli --repetition-penalty 1.1`, Qwen3-1.7B와 Llama-3.2-1B, 256과 8192 토큰으로 비교한다.

## 6. 학습 포인트

- pipelined decode 루프에 필요한 것은 다음 draw의 호스트 출력이 아니라 다음 forward의 device 입력이다. "forward 제출"과 "draw"를 나누면 호스트 쪽 상태를 가진 sampler도 overlap 대부분을 유지한다.
- 실패를 보고할 수 없는 readback(`item_*`, `tokens_to_host`)은 비동기 백엔드 오류가 드러나는 곳이다. 새 배열을 기다릴 때는 먼저 실패를 보고할 수 있는 `try_eval`을 거쳐야 한다.
- 숨은 reasoning 출력을 비교하는 실제 체크포인트 diff는 의미 없이 통과할 수 있다. `--show-reasoning` 없는 Qwen3는 모든 토큰이 숨은 채널로 가서 양쪽 출력이 모두 비어 있다. 검증 스크립트는 이 때문에 출력이 세 줄 미만인 경우를 실패로 처리한다.

## 7. 관련 자료

- 이슈 #2229(Part 1), epic #2166, 분리된 #2255(서버 prompt lookup), 후속 #2258과 #2259.
- PR #2225(`DirectEngine`, fused B=1 pipeline), PR #2217(엔진 step API), ADR 0009.
