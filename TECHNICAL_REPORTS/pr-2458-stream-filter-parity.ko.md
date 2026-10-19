# PR #2458: 스트리밍 tool-call과 marker 정리를 비스트리밍 출력과 일치시킴

**작성일**: 2026-10-19
**상태**: `mlxcel-server`에서 gpt-oss-20b, Aya Vision, GLM-4.1V-Thinking, Kimi-VL로 CUDA (GB10) 구현 및 검증 완료, PR 열림

## 요약

#2394를 닫는다. 서버의 스트리밍 `StreamFilter`와 비스트리밍 정리 경로(`clean_structural_tokens`, `clean_content_markers`, `parse_tool_calls`)는 같은 모델 출력을 다섯 군데에서 다르게 처리했다. 이 PR은 두 경로가 같은 결과를 내도록 맞춘다. 검증에는 parity 테스트 harness를 쓴다. 이 harness는 모든 사례를 통째로, 한 글자씩, 여러 chunk 폭으로, 그리고 모든 경계에서 나눠 넣어 본다.

## 문제 정의

1. 뒤에 보이는 텍스트가 없어도 tool call 뒤의 공백이 content로 스트리밍되었다.
2. thinking 닫기 뒤의 줄바꿈은 비스트리밍에서는 잘렸지만, 다음 chunk로 도착하면 스트리밍되었다.
3. 닫히지 않은 tool call과, 요청에 tool이 없을 때의 tool markup을 두 경로가 다르게 처리했다. 비스트리밍 `clean_content_markers`는 Hermes `<tool_call>` 블록도 흘렸다.
4. 출력 marker(Aya Vision의 `<|START_RESPONSE|>` / `<|END_RESPONSE|>`, GLM의 `<|begin_of_box|>` / `<|end_of_box|>`)가 클라이언트에 그대로 전달되었다. Kimi-VL은 `<|im_assistant|>`에서 멈추지 않았다.
5. gpt-oss Harmony 출력은 `analysis` 텍스트와 channel markup을 `delta.content`로 스트리밍했고, tool 없는 비스트리밍 응답도 이를 흘렸다.

## 기술적 결정

- **공백 보류.** tool call 뒤의 공백은 보이는 텍스트가 올 때까지 보류한다. 다른 call이나 thinking 블록이 열리거나 스트림이 끝나면 버린다.
- **post-thinking gap을 필드로 전달.** thinking 닫기 뒤의 줄바꿈은 여러 `feed()` 호출에 걸쳐서도 잘라 내고, 새 필드 `FilterOutput::post_thinking_gap`으로 보고한다. `--reasoning-format none` / `deepseek-legacy`에서는 `routes/chat.rs`와 `router_front.rs`가 닫기 marker 뒤에 이를 보내므로, #1470의 바이트 일치 테스트는 계속 통과한다.
- **Hermes 블록.** 비스트리밍 no-tools 경로에서는 `strip_hermes_tool_blocks`가 Hermes tool 블록을 제거한다. 모듈 문서에 #1743 규칙을 적었다.
- **marker 제거.** `<|...|>` marker 네 개는 `CHAT_DELIMITERS`와 `clean_content_markers` 양쪽에서 제거한다. Kimi-VL은 `<|im_assistant|>` id를 `eos_token_ids`에 추가한다. 이 id는 `tokenizer_config.json`의 `added_tokens_decoder`에서 먼저 찾고 없으면 `tokenizer.json`의 `added_tokens`에서 찾는다. 중복은 건너뛰고, 토큰이 없으면 아무것도 하지 않는다.
- **GLM answer wrapper는 tokenizer 기준으로 제한.** 대부분의 모델에서 `<answer>` / `</answer>`는 일반 텍스트다. 그래서 tokenizer가 둘 다 added token으로 정의할 때(`answer_wrapper_enabled`)만 제거하며, GLM-4.1V-Thinking이 여기에 해당한다. 스트리밍 경로는 `StreamFilter::with_answer_wrapper`를 설정하고, 비스트리밍 route는 content에 `strip_answer_wrapper`를 호출한다. reasoning에서는 태그를 유지한다.
- **Harmony는 공유 classifier로 처리.** filter는 `formats::HarmonyRoute`로 `<|start|>assistant ... <|channel|>...<|message|>` 헤더를 찾으며, 이 classifier는 이제 `try_harmony`와 공유한다. `<|end|>`, `<|return|>`, `<|call|>`은 Harmony 출력을 본 뒤에만 동작하므로 Phi-3의 `<|end|>`에는 영향이 없다. `clean_structural_tokens`가 `try_harmony`를 실행하므로 비스트리밍 no-tools 경로도 고쳐진다.
- **리뷰 후 보강.** Muse 또는 Harmony 헤더가 `<|message|>`에 닿기를 기다리는 보류는 256바이트(`MAX_HEADER_HOLD`)로 제한된다. 그래서 문자 그대로의 `<|channel|>`이 스트림을 멈추거나, `feed()`마다 커지는 버퍼를 다시 훑게 만들 수 없다. Harmony 출력을 본 뒤에는 다른 계열의 marker가 상태를 바꾸지 않는다. 따라서 call 인자 안의 `</tool_call>`이나 `analysis` 안의 `</think>`가 나머지를 content로 밀어내지 못한다. Harmony가 아닌 응답에 들어 있는 문자 그대로의 `<|channel|>`은 `try_harmony`와 마찬가지로 스트림 끝에서 텍스트로 유지한다.

## 변경 사항

- `src/server/tool_calls/stream_filter.rs`, `parser.rs`, `formats.rs`, `mod.rs`: filter 상태, 공유 Harmony classifier, marker 목록, answer-wrapper 조건.
- `src/server/tool_calls/stream_parity_tests.rs`(신규): parity harness.
- `src/server/routes/chat.rs`, `src/server/router_front.rs`, `src/server/in_process/chat.rs`, responses와 anthropic route: `post_thinking_gap`, answer-wrapper 조건.
- `src/loading/vlm_kimi_vl.rs`: `<|im_assistant|>` eos.
- `src/tokenizer/`: answer-wrapper 감지 테스트.
- `docs/supported-models.md`.

## 검증

GB10, `--profile test-fast --features cuda`. 테스트 바이너리는 `gpu-lock` 아래에서 `--test-threads=1`로, 최종 head에서 실행했다. `server::tool_calls`는 433개 통과했다. `server::`, `tokenizer::`, `reasoning_display`를 합치면 3674개 통과, 0개 실패, 21개 무시다. clippy(`--lib --tests -D warnings`)와 fmt도 깨끗하다.

원래의 parity 테스트 11개 중 10개는 main에서 실패한다. Phi-3 `<|end|>` guard는 설계상 양쪽에서 통과한다. 리뷰 중 추가한 테스트는 각자의 부모 커밋에서 실패한다. `answer_tags_are_ordinary_text_for_other_models`는 모든 모델에서 `<answer>`를 제거하던 이전 head에서 실행 중에 실패한다.

`mlxcel-server`, temperature 0:

| 체크포인트 | main | 이 PR |
|---|---|---|
| gpt-oss-20b-mxfp4, 4 arm (stream / non-stream, tools / no tools) | analysis 텍스트와 channel markup이 `delta.content`로 스트리밍됨. 비스트리밍 no-tools 응답에도 섞여 나옴 | 네 arm 모두 일치: no tools는 content `51`과 analysis를 `reasoning_content`로, tools는 빈 content, 같은 reasoning, `get_weather {"city":"Paris"}`, `finish_reason: tool_calls` |
| aya-vision-8b | `<|START_RESPONSE|>...<|END_RESPONSE|>` | 스트리밍과 비스트리밍 모두 일반 문장 |
| glm-4.1v-9b-thinking-4bit | `<answer>There are <|begin_of_box|>3<|end_of_box|> shapes...` | 두 경로 모두 `There are 3 shapes in the image.` |
| kimi-vl-a3b-thinking-4bit | 멈춤(`config.json`에 이미 163586이 있음) | 멈춤. 새 eos 경로는 fixture 테스트로 검증 |

## 후속 작업

- `--reasoning-format none`에서 gpt-oss는 analysis와 답을 함께 content로 스트리밍하지만, 비스트리밍 응답은 답만 돌려준다. Harmony에는 표준 thinking marker 쌍이 없으므로 `none`에서 어떻게 보여줄지 결정이 필요하다.
- `try_harmony`는 role 헤더에 붙은 `to=` 수신자(`<|start|>assistant to=functions.X<|channel|>...`)를 무시한다. 이 PR 이전부터 있던 동작이다.
- call 앞에 텍스트가 없는 `<tool_call>..</tool_call>\nB`는 스트리밍에서 `\nB`, 비스트리밍에서 `B`가 된다. 항목 1의 문구가 구분자 유지를 요구하기 때문이다.
- no-tools 경로의 MiniMax M3 namespace 잔여물. 여러 message로 나뉜 Harmony content가 비스트리밍에서는 `\n`으로 이어지고 스트리밍에서는 그대로 붙는 점. Kimi-VL의 `◁think▷` marker가 두 경로 모두에서 content에 남는 점.

## 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 13 |
| 추가 줄 | 1497 |
| 삭제 줄 | 116 |

| 커밋 | 메시지 |
|--------|---------|
| `e517fdfb` | fix: align streamed tool-call and marker cleanup with non-streamed output |
| `3efec1fa` | fix: strip GLM-4.1V-Thinking answer wrapper on both paths |
| `f8d3a32c` | fix: keep a literal <\|channel\|> outside Harmony output when streaming |
| `e013673e` | fix(security): bound the stream header hold and isolate Harmony mode |
| `4f8eb72a` | test: cover Harmony route classifier and stray Hermes close tags |
| `176626dc` | fix: strip the <answer> wrapper only for GLM-4.1V-Thinking tokenizers |

## 관련 항목

- #1470(바이트 단위 reasoning 그대로 전달), #1743(tool 없는 요청의 tool markup).
