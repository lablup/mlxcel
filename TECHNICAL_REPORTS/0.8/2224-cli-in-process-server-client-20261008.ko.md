# 기술 보고서: PR #2224 - 프로세스 내 서버 클라이언트가 된 `mlxcel run`과 chat

**작성일**: 2026-10-08

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Rust, Markdown(문서)

**위험도**: 중간. CLI의 대화형 경로가 엔진, 기본값(ADR 0007 표 기준), 일부 출력 방식까지 바뀌고, 운영 중인 HTTP chat 핸들러를 공유 함수로 나눴다. HTTP 검증 순서와 메시지는 그대로이고, 실제 체크포인트 두 개에서 `run -p`와 서버의 greedy 출력이 같았으며, 텍스트 모델과 VLM으로 REPL을 처음부터 끝까지 돌려 확인했다.

## 요약

`mlxcel run`과 chat REPL은 `CxxGenerator`를 썼고, 서버에 이미 있는 기능을 따로 갖고 있었다. reasoning 분리(`ReasoningFilter`), VLM 임베딩 분기(`compute_vlm_embeddings`), Laguna만 지원하는 오프라인 DFlash 드라이버가 그것이다. epic #2166의 Phase 5(#2173)인 이 PR은 `llama-cli`가 `llama-server`의 클라이언트인 llama.cpp 구조를 따라, 두 명령을 서버 엔진의 프로세스 내 클라이언트로 바꾼다. 모델 worker를 슬롯 하나로 띄우고, 모든 턴을 `/v1/chat/completions` 핸들러와 같은 코드로 보내며, HTTP listener는 열지 않는다. CLI에는 터미널 입출력과 REPL 명령만 남고, 템플릿, reasoning 분리, tool call, stop 문자열, 미디어 준비, speculative decoding은 서버 코드를 쓴다.

## 1. 문제 정의

- `run -p`는 `generate`와 코드를 공유하니 출력이 바이트 단위로 같다고 약속했지만, 둘 다 서버와는 달랐다. 기본값(penalty 창, DRY 구분자, 루프 감지, prefill 청크)이 다르고, reasoning 분리와 미디어 준비 구현도 따로였다.
- 오프라인 드라이버가 별개라서 CLI에서는 Qwen 3.5용 서버 DFlash target이나 MTP burst를 쓸 수 없었다.
- 서버 chat 경로를 고칠 때마다 CLI에도 같은 수정을 따로 해야 했다.

## 2. 변경 요약

- **`server::in_process::InProcessServer`**: `start_server`가 만드는 것을 listener와 부가 모델만 빼고 그대로 만든다. 설정, prompt-cache 저장소, `ModelProvider` worker, 새로 공유한 `startup::load_chat_front`로 읽는 tokenizer와 템플릿, `AppState`, 공유 함수 `run_startup_warmup`이다. 엔진 probe의 `ServerEngine`도 이제 이것으로 시작하므로 서버 구성 코드는 하나뿐이다.
- **`routes::chat_generation`**: chat 핸들러의 요청 검증(`admit_chat_request`)과 렌더/옵션 구성(`prepare_chat_generation`)을 `routes/chat.rs`에서 분리했다. 두 HTTP 핸들러와 `InProcessServer::chat`이 이를 호출한다. `chat`은 worker 스트림을 `StreamFilter`로 읽고, tool call을 파싱하고, reasoning echo를 기록하고, 다음 턴 warm-up을 보낸다. `--no-chat-template`은 `/v1/completions` 경로를 쓴다.
- **CLI**
  - `cli/in_process_client.rs`가 플래그를 서버 옵션으로 옮긴다.
  - `cli_turn.rs`가 터미널 출력과 Ctrl-C를 맡는다. 첫 Ctrl-C는 worker의 cancel 플래그로 턴을 취소하고, 두 번째 Ctrl-C나 턴 밖의 Ctrl-C는 SIGINT로 프로세스를 끝낸다.
  - `chat_transcript.rs`는 이미지를 한 번만 읽어 `data:` URI로 넣어 메시지를 만들고, 서버의 요청당 이미지 한도를 넘으면 거절한다.
  - REPL은 대화마다 `prompt_cache_key`를 두고 prompt cache를 켜 둔다. `/clear`를 하면 새 key로 바뀐다.
- **제거**: `src/reasoning_stream.rs`(출력은 `StreamFilter`로)와 `compute_vlm_embeddings`(`mlxcel generate`는 `prepare_request_vlm_embeddings` 위의 `server::local_media`로 미디어를 준비).
- **`run`의 새 플래그**: `--draft-model`, `--draft-kind`, `--draft-block-size`, 서버의 샘플링 플래그(frequency/presence, XTC, mirostat, dynatemp, DRY 구분자), `--stop`.

## 3. 기술적 선택과 그 이유

- **loopback HTTP가 아니라 프로세스 내 요청 채널을 쓴다.** llama-cli는 llama-server 스레드에 HTTP로 요청한다. 한 프로세스 안에서는 소켓과 SSE 직렬화만 늘고, 공유되는 코드는 요청 파싱 이상 늘지 않는다. mlxcel의 `ModelProvider` 채널은 모든 HTTP 라우트가 HTTP 계층 아래에서 이미 쓰는 경로다(ADR 0007).
- **핸들러 로직을 복사하지 않고 분리한다.** CLI가 HTTP 라우트와 같은 검증, 렌더 함수를 호출하므로 이후의 수정이 양쪽에 함께 들어간다. 리뷰에서 HTTP 검증 순서(빈 입력, logprobs, XTC, top-n-sigma, typical-p, 읽기 전 미디어 지원 여부, 비디오 확장, tool 입력, thinking 예산, 문법)와 오류 메시지가 바뀌지 않았음을 확인했다.
- **기본값은 ADR 0007 표를 따르고 greedy는 유지한다.** 지정하지 않은 플래그는 서버 값을 쓴다. penalty와 DRY 창 64, b10621 DRY 구분자, 서버의 루프 감지 규칙, prefill 청크 2048, 시퀀스별 seed다. 표의 예외 하나는 그대로 둔다. `generation_config.json`에 값이 없으면 CLI는 greedy를 유지해서 모델 테스트와 벤치마크의 재현성을 지킨다.
- **`generate`의 Inkling 레이아웃은 유지한다.** 서버의 ordered 오디오 레이아웃은 chat 라우트만 렌더하는 미디어 마커가 있어야 동작한다. `generate`의 Inkling 미디어를 그쪽으로 보내면 템플릿을 쓰는 프롬프트와 raw 프롬프트가 모두 깨진다. 그래서 `server::local_media`는 `generate`의 `Structured`, `Plain` 레이아웃을 유지한다.
- **REPL에서 classic `--draft-model`은 안내 문구와 함께 무시한다.** `generate`에서 이 플래그는 classic draft 모델을 뜻하는데 서버는 이를 실행하지 않는다. REPL은 `--draft-kind dflash|mtp`일 때만 drafter를 쓴다.

## 4. 검증

- clippy `-D warnings`(lib, tests, bins, examples), fmt, 라이선스 헤더, `vlm_wrapper_capability_delegation`, `dead_doc_pointers`, `model_loader_stdout_guard`, `cli_help_consistency`, `llama_model_source_cli`.
- `--test-threads=1` 모듈 테스트: `server::in_process`, `server::local_media`, `server::routes::`, `server::tool_calls`, `server::request_options`, `startup`, `cli_input`, `engine_probe`, `reasoning_display`, `server::chat_template`, model worker와 provider, `server::batch::scheduler`, `multimodal::`, 바이너리의 `commands::`.
- 실제 체크포인트, release 빌드:
  - `MLXCEL_SDPA_DETERMINISTIC=1`에서 `mlxcel-engine-parity`: `e:run`(`run -p`가 만드는 설정)과 `f:server`(같은 chat 요청을 `mlxcel-server` 기본값, 슬롯 4개, paged로 실행)가 Qwen3-1.7B 4-bit와 Llama-3.2-1B 4-bit 모두에서 같았다. 기존 CLI, dense, paged, 엔진 arm 결과도 그대로다.
  - Llama-3.2-1B에서 `run -p`를 돌렸다. Qwen3-1.7B REPL에서는 여러 턴에 걸친 기억, 답변 중 Ctrl-C, `/clear` 후 이름을 잊는 것까지 확인했다. Qwen2.5-VL-3B REPL에서는 `/image` 턴과 후속 턴을 확인했다.
  - `run -p --draft-model`: Qwen3.5-4B DFlash(초안 토큰 510개 중 429개 채택), Gemma-4-12B와 assistant drafter의 MTP.
  - `generate` 이미지(Qwen2.5-VL)와 오디오(Gemma-4-E2B): 변경 전 바이너리와 prompt id, greedy 토큰이 같았다.

## 5. 남은 위험

- **CLI 사용자가 보는 출력 변화**(PR에 정리): 템플릿이 없는 체크포인트에는 서버 기본 템플릿(안내 문구 포함), `run -p`의 프롬프트 echo 제거, 플래그를 주지 않았을 때 서버 샘플링 기본값, `generate` 출력의 markup 제거.
- **패리티 하네스의 CLI식 렌더와 `mlxcel-bench-decode`는 여전히 chat 프롬프트를 서버와 다르게 렌더한다**(`enable_thinking=true`를 빼먹는다). 그래서 Qwen3에서는 하네스의 arm a~d와 `e:run`이 다른 프롬프트를 비교한다. 문서에 적어 두었고, Phase 6(#2176)이 모든 렌더를 함수 하나로 모은다.
- **Inkling 경로는 단위 테스트로만 확인했다.** 로컬에 체크포인트가 없다.
- **CLI decode 처리량과 시작 시간은 아직 재지 않았다.** epic 마지막 측정에서 다룬다.
- **오디오, 비디오, 오디오 출력, `--profile`, `--estimate-memory`, diffusion과 OCR 계열을 쓰는 `run -p`는** Phase 6 전까지 `generate`를 거친다.

## 6. 학습 포인트

- **운영 라우트를 리팩터링할 때는 분리한 함수를 기존 두 복사본과 필드 단위로 대조해야 한다.** 두 HTTP 핸들러는 조금씩 달라져 있었고, 분리한 함수는 이상적인 버전이 아니라 각각의 실제 동작을 재현해야 했다.
- **'같은 코드'보다 '같은 요청'이 더 강한 보장이다.** 예전 `run -p`의 약속은 `generate`와 코드를 공유한다는 데 기댔는데, 그 `generate`부터 서버와 달랐다. 새 보장은 요청 경로에 대한 패리티 테스트다.

## 7. 관련 항목

- epic #2166, 이슈 #2173, ADR 0007, ADR 0009.
- #2217(엔진), #2176(Phase 6), ggml-org/llama.cpp#17824와 #24948(서버 클라이언트가 된 llama-cli).
