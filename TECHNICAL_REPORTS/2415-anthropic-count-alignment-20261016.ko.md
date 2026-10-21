# PR #2415: count 라우트와 message_start를 usage에 맞춤

**작성일**: 2026-10-16
**상태**: CUDA(GB10)에서 gemma-3-4b-it-4bit 실제 체크포인트로 구현 및 검증 완료. `xla-iree` 서빙 루프와 disaggregated coordinator에서는 미디어 본문의 `message_start`가 여전히 placeholder 기준 토큰 수를 담음 (「미검증 항목과 후속 조치」 참고)
**위험도**: 중간 (미디어 본문의 Anthropic 스트리밍 이벤트 순서와, 네 라우트가 공유하는 프롬프트 렌더링을 변경)

## 요약

이슈 #2337은 count 라우트가 `count_prompt_tokens`를 공유하도록 만든 #2321(#1748)과 #2335(#1788)의 후속이다. `/v1/messages`나 그 형제 라우트가 실제로 처리하는 값과 토큰 수가 어긋나는 오래된 세 가지 틈이 남아 있었다. 스트리밍 `message_start.usage.input_tokens`가 미디어 확장을 무시했고, count 라우트가 #2110 reasoning re-echo를 건너뛰었으며, 서버가 생성할 수 없는 상태에서도 count 라우트가 200을 반환했다. 이 PR은 세 가지를 한 번에 해결한다.

## 문제 정의

- `stream_messages`는 `message_start`를 `tokenizer.encode(&prepared.prompt, true)`로 채웠다. placeholder 텍스트를 대상으로 `add_special`을 강제로 켜고, 네이티브 `prompt_token_ids`는 무시했다. gemma-3-4b-it-4bit에서 이미지 본문은 `count_tokens`와 비스트리밍 `usage`가 275인데 `message_start`는 17이었다. 텍스트 전용 본문은 gemma-3 템플릿이 이미 BOS를 넣기 때문에 15 대신 16이 나왔다.
- `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/v1/messages/count_tokens`, `/apply-template`이 공유하는 `render_admitted_chat_prompt`는 요청을 받은 그대로 렌더링했다. 생성 라우트는 `reasoning_echo::chat_scope`와 `render_request`도 실행하므로, echo 저장소가 켜진 상태에서 content만 있는 assistant 턴의 count가 생성 결과보다 작았다 (새 테스트에서 52 대 78).
- `chat_input_tokens`, `responses_input_tokens`, `anthropic_count_tokens`는 형제 라우트가 가장 먼저 실행하는 chat 사용 불가 검사를 하지 않았다. `--embeddings`이거나 chat 모델이 없을 때 `/v1/chat/completions/input_tokens`가 200 `{"input_tokens":0}`을 반환했다.

## 기술적 선택과 그 이유

- `count_prompt_tokens`의 텍스트 전용 규칙을 `prompt_inspection::count_text_prompt_tokens`로 분리했다. 네이티브 id가 있으면 그것을 쓰고, 없으면 `add_special = !prompt_carries_bos`로 인코딩한다. 텍스트 전용 스트리밍 본문은 이 함수를 쓰며 `message_start`는 기존처럼 즉시 나간다. BOS 과다 계산도 함께 사라진다.
- 미디어가 있는 본문(`PreparedChatRequest::carries_media()`)은 worker의 첫 `PrefillStats`가 올 때까지 `message_start`를 보류하고 그 `prompt_tokens`를 쓴다. 이 값은 `usage`와 `admission.rs`의 count 전용 요청이 보고하는 확장 후 길이와 같다. `streaming_anthropic::AnthropicMessageStart`가 이벤트를 보관하고 정확히 한 번만 보낸다. 다른 이벤트를 내보낼 수 있는 모든 경로는 먼저 `send`를 호출하며, 이때 쓰는 값은 prefill 보고, 없으면 토큰이나 에러 직전의 로컬 count, 그것도 없으면 완료 시 결과의 `prompt_tokens`다.
- 기각한 대안: 디스패치 전에 `count_prompt_tokens`를 호출하는 방식. worker 슬롯을 한 번 더 점유하고 스트리밍 미디어 요청마다 미디어 준비를 반복한다. prefill 보고는 추가 비용이 없고 미디어를 두 번 가져오지도 않는다.
- 이 순서 보장은 MLX 비전 계열을 포함한 모든 일반 모델을 서빙하는 batch scheduler에서 성립한다. prefill 완료 단계가 첫 토큰이 `stream_decoded_text`에 도달하기 전에 `GenerateEvent::Prefill`을 보낸다. speculative burst와 chunked prefill 경로도 토큰보다 먼저 보고한다. Florence-2와 Nemotron-Parse 이미지 루프는 `Prefill`을 보내지 않았으므로, 단일 답변 토큰 직전에 첫 토큰 스냅샷을 보내도록 바꿨다.
- count 라우트는 reasoning echo 저장소를 읽기만 하고 `record_reply`는 호출하지 않는다. 적중하면 생성과 마찬가지로 항목의 LRU 최근성이 갱신된다. `/apply-template`도 같은 렌더링을 쓰므로, 이제 생성 라우트가 prefill할 재주입된 reasoning이 그대로 보인다.
- 사용 가능 여부 검사는 각 count 라우트의 첫 문장이며 형제 라우트와 같은 501 상태와 본문을 반환한다. `anthropic::chat_unavailable_response`를 새로 만들어 `anthropic_messages`와 공유한다.

트레이드오프: 긴 미디어 큐 대기와 prefill 동안 클라이언트는 `message_start` 전에 SSE keepalive 주석만 받으며, `--sse-ping-interval -1`이면 아무것도 받지 못한다. `message_start`보다 앞서는 이벤트는 없고, 이는 Anthropic 스트림이 요구하는 순서다. `docs/llama-server-compat.md`에 이 내용을 적었다.

## 변경 요약

| 영역 | 파일 |
|------|------|
| 공유 텍스트 count, 미디어 감지 | `src/server/routes/prompt_inspection.rs`, `src/server/chat_request.rs` |
| `message_start` 보류 | `src/server/streaming_anthropic.rs`, `src/server/routes/anthropic.rs` |
| count 라우트의 reasoning re-echo, 사용 가능 여부 검사 | `src/server/routes/prompt_inspection.rs`, `src/server/routes/anthropic.rs` |
| 이미지 루프의 prefill 보고 | `src/server/florence2_worker.rs`, `src/server/nemotron_parse_worker.rs` |
| 테스트와 테스트 지원 코드 | `src/server/routes/prompt_count_route_tests.rs`, `src/server/routes/availability_tests.rs`, `src/server/model_provider_test_support.rs` |
| 문서 | `docs/llama-server-compat.md` |

origin/main 대비 10개 파일, +666 -64. 커밋: `89f875e6` fix(server): align count routes and message_start with usage, `043063e5` fix(server): report prefill on Florence-2 and Nemotron-Parse loops.

## 검증

새 테스트는 모두 수정을 되돌렸을 때 실패하는 것을 확인했다.

- `anthropic_streaming_image_input_tokens_equal_count_and_usage`: `message_start`가 첫 이벤트이고 한 번만 나가며 `count_tokens`, `usage`와 같다. 되돌리면 left 33, right 1033.
- `{chat,responses,anthropic}_count_includes_the_reasoning_echo`: count가 후속 생성이 받는 프롬프트 길이와 같다. 되돌리면 left 52, right 78.
- `count_routes_refuse_like_their_generating_siblings`: chat 모델이 없을 때와 `--embeddings`일 때 여섯 개 count 경로 전부.
- `anthropic_streaming_media_without_a_prefill_report_still_opens_with_message_start`: prefill 보고 없이 토큰을 스트리밍하는 backend도 `message_start` 하나로 시작한다.
- `message_start_goes_out_once_with_the_first_figure`와 텍스트 전용 일치 테스트 (후자는 stub tokenizer가 BOS를 넣지 않아 되돌린 코드에서도 통과한다).

로컬 게이트(`--features cuda`, `--profile test-fast`, `--test-threads=1`): `prompt_inspection` 28 통과(1 무시, ffmpeg 필요), `availability_tests` 6, `routes::anthropic` 21, `streaming_anthropic` 4, `reasoning_echo` 30, `florence2_worker` 15, `nemotron_parse_worker` 7. `-D warnings` clippy, `cargo fmt --check`, 저장소 검사 스크립트 모두 통과했다.

실제 체크포인트 `~/models/mlx/gemma-3-4b-it-4bit`에서 같은 본문을 `count_tokens`, 비스트리밍 `/v1/messages`, 스트리밍 `/v1/messages`로 보냈다.

| 본문 | 빌드 | count_tokens | usage.input_tokens | message_start.usage.input_tokens |
|---|---|---|---|---|
| 이미지 | main 93e5ef75 | 275 | 275 | 17 |
| 이미지 | 이 PR | 275 | 275 | 275 |
| 텍스트 전용 | main 93e5ef75 | 15 | 15 | 16 |
| 텍스트 전용 | 이 PR | 15 | 15 | 15 |

## 미검증 항목과 후속 조치

- `xla-iree` 서빙 루프(`src/server/batch/xla_worker.rs`)와 disaggregated coordinator는 prefill 보고를 보내지 않는다. 미디어 본문의 `message_start`는 이전과 같이 로컬 placeholder 기준 값을 담는다. XLA 루프는 로컬 게이트에서 컴파일되지 않고, coordinator 변경은 해당 서브시스템에서 다루는 것이 맞다.
- BOS 과다 계산 수정은 실제 체크포인트에서만 확인했다. route 테스트의 stub tokenizer는 BOS를 넣지 않는다.
- 전체 `server::routes` 테스트 필터는 공유 장비에서 너무 커서 로컬로 실행하지 않았다. CI가 다룬다.
- gemma-3 체크포인트에는 reasoning 채널이 없으므로 re-echo는 route 테스트로만 검증했다.
