# 기술 보고서: PR #1740 - feat(gemma4_unified): 한 프롬프트에 비디오와 오디오를 함께 받는다

**날짜**: 2026-09-10
**작성**: mlxcel maintainers
**검토**: 구현 및 리뷰 후속 사이클
**상태**: 완료 (Apple Silicon / Metal에서 `models/gemma-4-12b-it-4bit`로 검증. `--workspace` 게이트는 머신을 공유 중이라 로컬에서 돌리지 않았고, 좁은 범위 테스트와 CI가 대신 덮는다)
**언어**: Rust, Markdown
**위험도**: 중간 (한 계열이 기능을 얻고, 모든 생성 라우트가 공유하던 거부가 HTTP 경계로 옮겨가며, 생성하지 않는 두 프런트가 없던 미디어 게이트를 얻는다)

---

## 요약

`Gemma4UnifiedModel::merge_multimodal`은 처음부터 이미지, 비디오 프레임, 오디오 특징, 오디오 마스크를 한 번의 호출로 받는다. 정작 네 가지를 다 넘겨주는 진입점이 없었다. CLI는 `--video`와 `--audio`를 같이 주면 에러를 냈고, 서버는 두 번 거부했다. `prepare_chat_request_with_cache`에서 한 번, 워커에서 또 한 번이다. 사운드트랙이 붙은 클립은 인코더 없는 통합 체크포인트에 가장 자연스러운 입력이므로, `gemma4_unified`에 대해서만 거부를 풀고 나머지 계열에는 문구 그대로 유지한다.

흥미로운 부분은 거부를 푼 것이 아니라 검사가 어디로 갔는지다. 요청 준비 단계는 템플릿을 렌더링할 뿐 로드된 모델을 보지 못하므로, 두 모달리티를 한 토큰 스트림으로 섞는 체크포인트와 각각 따로만 받는 계열을 구분할 수 없다. 검사를 HTTP 경계의 `media_capability_rejection`으로 옮기자 검사가 fetch보다 앞에 서게 되었고, 그 순간 경계를 거치지 않고 채팅 바디를 렌더링하는 두 프런트가 드러났다. 분산 라우터와 프롬프트 검사용 라우트 세 개다. 세 번째 커밋이 둘 다 막는다. 그 과정에서 오디오 디코드가 비디오 디코드보다 앞으로 옮겨졌다. 몇 킬로바이트짜리 깨진 오디오가 전 클립 ffmpeg 디코드와 샘플된 모든 프레임의 패치 프로젝션을 사들이게 둘 이유가 없다.

두 확장이 서로를 덮어쓰지 않는다는 가장 깨끗한 증거는 산술이다. 결합 프롬프트는 637토큰이고, 이는 정확히 537(비디오 단독) + 오디오 런 95 + 길어진 프롬프트가 더한 텍스트 5토큰이다.

---

## 1. 거부가 옮겨가야 했던 자리

### 1.1 `prepare_chat_request_with_cache`는 이 판단을 할 수 없다

기존 서버 측 거부는 `src/server/chat_request.rs`에 있었고, `declared_audio > 0 && declared_videos > 0`이 되는 순간 조건 없이 `anyhow::bail!("Combined video and audio inputs are not supported")`를 냈다. 이 함수는 채팅 템플릿을 렌더링하고 미디어 파트를 해석한다. 로드된 모델을 전혀 보지 못하므로, `merge_multimodal`이 두 모달리티를 한 토큰 스트림에 흩뿌리는 인코더 없는 통합 체크포인트와, 비디오만 따로 오디오만 따로 받고 둘을 합칠 경로는 없는 계열을 구분할 방법이 없다. 이 함수에 구분법을 가르치려면 로드된 모델을 요청 준비 단계까지 끌고 들어와야 하는데 방향이 틀렸다. 답은 체크포인트의 속성이고, 서버는 시작 시점에 그것을 이미 한 번 계산한다.

검사는 이제 `media_capability_rejection`(`src/server/media.rs`)에 있다. 이 함수는 `detect_model_media_support`가 검출한 `ModelMediaSupport` 플래그를 이미 읽고 있으며, 참조된 이미지·오디오·비디오 페이로드의 어느 바이트도 가져오기 전인 HTTP 경계에서 돈다. `prepare_chat_request_with_cache`에는 bail이 있던 자리에 새 주인을 지목하는 주석만 남는다.

### 1.2 `ModelMediaSupport::video_with_audio`는 논리곱이 아니라 별도 플래그다

`src/server/state.rs`에 필드 하나가 늘었다.

```rust
pub struct ModelMediaSupport {
    pub image: bool,
    pub audio: bool,
    pub video: bool,
    pub video_with_audio: bool,
}
```

`video_with_audio`는 일부러 `audio && video`가 아니다. 그 논리곱은 Gemma 4 VL, Kimi-VL, Inkling, Qwen-VL에 대해 틀린다. 넷 다 각 모달리티를 단독으로는 소비하지만 둘을 합칠 경로는 없다. `detect_model_media_support`는 `matches!(model_type, ModelType::Gemma4Unified)`로 플래그를 세우고, `src/loaded_model_capabilities.rs`의 `LoadedModel::supports_video_with_audio`가 같은 단일 arm을 가진 워커 측 짝이다. `server::startup`의 검출 테스트 다섯 개가 Gemma 4 VL, Kimi-VL 2.5, Inkling, Qwen3.5 VL, config 누락 폴백에 대해 음성을, `gemma4_unified`에 대해서만 양성을 단언한다.

### 1.3 경계 검사 내부의 순서

결합 arm은 모달리티별 arm 세 개 뒤, 맨 마지막에 놓인다.

```rust
if !support.image && !request.image_urls().is_empty() { return refuse("image"); }
if !support.audio && !request.audio_inputs().is_empty() { return refuse("audio"); }
if !support.video && !request.video_urls().is_empty() { return refuse("video"); }
if !support.video_with_audio && !request.video_urls().is_empty() && !request.audio_inputs().is_empty() { /* 400 invalid_request_error */ }
```

이 순서가 의미를 갖는다. 비디오와 오디오를 함께 받은 텍스트 전용 체크포인트는 영영 갖지 못할 능력을 지목하는 `audio input is not supported`를 들어야지, 어차피 조치할 수 없는 결합 거부를 들어서는 안 된다. `media_capability_rejection_reports_the_missing_modality_before_the_combination`이 이를 고정한다.

거부는 문구를 그대로 유지하며(이제 공유 상수 `COMBINED_VIDEO_AUDIO_REFUSAL`), `invalid_request_error` 타입과 400 상태도 그대로다. `/v1/chat/completions`와 `/v1/responses`는 이 이동으로 달라지지 않는다.

### 1.4 Anthropic 봉투: 능력 부재가 아니라 클라이언트 오류

`routes::anthropic`은 모든 미디어 능력 거부를 `501 not_supported_error`로 렌더링했다. 모달리티별 거부에는 맞고 결합 거부에는 틀리다. 그런 체크포인트에서 각 모달리티는 단독으로 받아들여지고 오직 그 쌍만 합칠 경로가 없으므로, 클라이언트는 파트 하나를 빼면 고칠 수 있다. `media_rejection_response`는 `is_combined_video_audio_rejection`(상태와 공유 상수를 본다. 그래서 라우트가 문자열 사본을 들지 않는다)으로 갈라 이 한 경우에만 경계의 400을 유지한다. 오늘 `/v1/messages`로는 이 arm에 닿을 수 없다. `AnthropicContentBlock`에 video나 audio 변형이 없기 때문이다. 그래도 미리 쓴 이유는 그 스키마가 video 블록을 갖게 되는 날 상태 코드가 이미 맞아 있게 하기 위해서고, 새 테스트 두 개가 스위치를 직접 호출하는 이유도 같다.

---

## 2. 단일 모달리티 헬퍼로 결합 프롬프트를 짓는다

### 2.1 양쪽 표면의 모달리티별 헬퍼

워커와 CLI 모두 결합 빌더를 쓰기 전에 헬퍼로 쪼개졌고, 결합 빌더는 정확히 그 헬퍼들을 호출한다.

| 단계 | 서버 (`src/server/model_worker.rs`) | CLI (`src/commands/generate_vlm.rs`) |
|---|---|---|
| 동반 이미지 | `gemma4_unified_server_images` | `gemma4_unified_cli_images` |
| 비디오 디코드 | `gemma4_unified_server_decode_videos` | `gemma4_unified_cli_decode_videos` |
| 프레임 패치화와 플레이스홀더 확장 | `gemma4_unified_server_video_frames` | `gemma4_unified_cli_video_frames` |
| 오디오 디코드와 청킹 | `gemma4_unified_server_audio_features` | `gemma4_unified_cli_audio` |
| 오디오 런 확장 | `gemma4_unified_server_expand_audio_run` | (`gemma4_unified_cli_audio` 안) |

복제가 아니라 재사용이라는 점이 단일 모달리티 경로의 바이트 동일성을 보장하는 근거 전부다. `prepare_gemma4_unified_audio_embeddings`, `prepare_gemma4_unified_video_embeddings`, `prepare_gemma4_unified_video_and_audio_embeddings`가 같은 입력에 같은 코드를 돌리므로, 비디오 단독이나 오디오 단독 프롬프트가 이 경로 이전과 달라질 수 없다. 4절의 검증이 이를 실측으로 확인하지만, 그것을 우연이 아니라 참으로 만드는 것은 구조다.

`src/vision/gemma4_unified.rs`의 `get_input_embeddings_with_video_and_audio`는 네 번째이자 가장 넓은 래퍼이며, `merge_multimodal`의 모든 인자에 닿는 유일한 래퍼다. 자체 로직은 없다.

### 2.2 이미지는 비디오 프레임보다 먼저 확장되어야 한다

순서는 장식이 아니다. `expand_gemma4_image_tokens`는 `image_token_id` **또는** `boi_token_id`를 이미지 플레이스홀더로 세고, `expand_gemma4_unified_video_tokens`는 방출하는 모든 프레임을 각자의 `boi_token_id`로 감싼다. 따라서 이미지를 나중에 확장하면 방출된 비디오 프레임 하나하나가 이미지 플레이스홀더로 세어지고, 프롬프트는 호출자가 설명할 수 없는 개수로 이미지 카디널리티 검사에 실패하거나(`Gemma4 prompt has N image placeholder(s) but M image(s) were provided`), 개수가 우연히 맞아떨어지면 엉뚱한 런에 대해 확장된다. 두 결합 빌더는 순서를 명시하는 데 그치지 않고 이 이유를 문서 주석에 담는다.

### 2.3 플레이스홀더 순서 제약과 오디오 분할

오디오 런은 이미지·비디오 런 **뒤에** 확장되어야 한다. 그래야 세 플레이스홀더 스트림이 `merge_multimodal`이 흩뿌리는 순서로 놓인다. `image_token_id`에 대한 `merge_llava`, 이어서 진행 중인 임베딩에 대해 `video_token_id`에 대한 `merge_llava`, 마지막으로 `audio_token_id`에 대한 `masked_scatter`다. 비디오 런은 BOS 뒤에 끼어들고 오디오 런은 마지막 `<end_of_turn>` 앞에 놓이므로(이슈 #437), 둘은 서로 다른 id를 서로 다른 삽입 지점에서 다룬다.

이 제약은 더 값싼 다른 제약과 충돌한다. 검증은 먼저 돌아야 한다. 그래서 `gemma4_unified_server_audio`가 둘로 쪼개졌다. `gemma4_unified_server_audio_features`는 클립을 디코드하고 `require_single_server_audio_clip`을 강제하며 파형을 `audio_samples_per_token` 프레임으로 청킹한다. 프롬프트는 건드리지 않는다. `gemma4_unified_server_expand_audio_run`은 프롬프트 변형만 한다. 결합 빌더는 둘을 양 끝에 둔다.

```rust
let audio_input = gemma4_unified_server_audio_features(unified, audio_data)?;   // 검증, 맨 앞
let decoded_videos = gemma4_unified_server_decode_videos(videos)?;
let processed_images = gemma4_unified_server_images(unified, prompt_tokens, images, image_soft_tokens)?;
let video_frames = gemma4_unified_server_video_frames(unified, prompt_tokens, &decoded_videos)?;
gemma4_unified_server_expand_audio_run(unified, prompt_tokens, audio_input.num_frames, end_of_turn_token_id);  // 맨 뒤
```

디코드가 낼 수 있는 두 실패, 즉 클립 개수가 1이 아닌 경우와 WAV 리더가 거부하는 파형은 모두 클라이언트가 이미 보낸 바이트로 결정된다. 디코드를 마지막에 두면 몇 킬로바이트짜리 깨진 오디오가 거부되기 전에 전 클립 ffmpeg 디코드와 샘플된 모든 프레임의 패치 프로젝션을 사들인다. 오디오 단독 경로는 이미 먼저 검증하고 있었다. 이 변경은 결합 경로가 그 작업을 더 싸게 사는 통로가 되는 것을 막는다.

의도한 비대칭이 하나 있다. 오디오 단독 경로는 `embed_audio`가 `None`이면 경고를 내고 오디오를 버리지만, 결합 경로는 `MISSING_AUDIO_EMBEDDER_REFUSAL`로 거부한다. 그 검사가 발동할 시점이면 비디오 런은 이미 프롬프트에 들어가 있으므로, 비디오만으로 답하는 것은 호출자가 하지 않은 질문에 답하는 것이고, 200 응답 어디에도 입력 절반이 사라졌다는 표시가 없다. `ModelMediaSupport`도 `LoadedModel::supports_video_with_audio`도 이를 잡을 수 없다. 둘 다 실제로 어떤 가중치가 로드되었는지가 아니라 모델 타입을 보기 때문에, 모델을 쥔 코드가 이 사실을 처음 아는 자리다.

---

## 3. fetch 이전 미디어 게이트

거부를 경계로 옮기자 잠복해 있던 문제가 드러났다. `prepare_chat_request_with_cache`는 렌더링의 일부로 모든 `image_url`과 `input_audio` 페이로드를 내려받고 모든 `video_url`을 연다. 그 뒤에 거부한다는 것은 클라이언트가 지목한 URL을 먼저 가져온다는 뜻이다. 두 프런트가 정확히 그렇게 하고 있었고, 둘 다 모델 워커에 아예 닿지 않으므로 워커 백스톱은 도움이 되지 않는다.

**`router_front`.** `route_chat`은 준비 함수가 반환한 뒤에야, 해석기 출력에 대해 `has_declared_media`를 물어서 미디어를 거부했다. 분산 라우터는 뒤에 있는 모든 체크포인트에 대해 텍스트 전용이므로 그 다운로드는 어디에도 쓰일 수 없다. 이제 `request_declares_media`가 렌더링 전에 요청 자체에 묻는다.

```rust
fn request_declares_media(request: &ChatCompletionRequest) -> bool {
    !request.image_urls().is_empty()
        || !request.audio_inputs().is_empty()
        || !request.video_urls().is_empty()
}
```

해석 후의 `has_declared_media` 검사는 백스톱으로 남긴다. 요청 검사 이후에 미디어 파트를 합성하는 변환 단계가 훗날 생기더라도 조용히 버려지는 대신 거부되도록 하기 위해서다. 새 테스트는 `http://169.254.169.254/`를 가리키는 `image_url`, `input_audio` 파트, `video_url`로 술어를 구동하고 텍스트 전용 음성 사례도 함께 확인한다.

**`prompt_inspection`.** `/apply-template`와 `chat/completions/input_tokens` 두 경로는 아예 거부하지 않았다. 체크포인트에 타워가 없는 미디어를 먼저 가져온 뒤 토큰 수만 보고했다. 이제 `render_chat_prompt`가 `validate_chat_tool_inputs`보다 먼저 `media_capability_rejection`을 돌린다. `/v1/chat/completions`가 쓰는 순서 그대로이므로, 검사 전용 라우트 세 개가 생성 라우트와 같은 방식으로 답한다. 테스트 셋이 덮는다. 텍스트 전용 스텁에 이미지 파트를 보내면 세 경로 모두 `501 not_supported_error`로 거부하고 바디에 `prompt`도 `input_tokens`도 없다는 것, 비디오와 오디오 결합도 여기서 여전히 거부된다는 것, 텍스트 전용 바디는 영향을 받지 않는다는 것이다.

---

## 4. 검증

모든 실행은 Apple GPU(Metal)에서 `models/gemma-4-12b-it-4bit`에 대해 그리디(`--temp 0`)로 수행했고, 변경 전 바이너리의 동일 명령과 대조했다. 두 번째 커밋에서 그 트리로 다시 빌드한 바이너리로 전체 세트를 재실행해 모든 수치를 재현했다.

### 4.1 토큰 회계는 정확히 가법적이다

| 실행 | 확장 라인 | 총 프롬프트 토큰 |
|---|---|---|
| `--video clip.mp4` | `expanded 1 video(s) into 8 frame slot(s)` | 537 |
| `--audio speech.wav` | `expanded audio into 93 soft tokens` | 114 |
| 둘 다 | `expanded 1 video(s) into 8 frame slot(s) and audio into 93 soft tokens` | 637 |

637 = 537 + 95 + 5이다. 비디오 단독 프롬프트에, 오디오 런 95(소프트 토큰 93개를 BOA와 EOA가 감싼다), 길어진 프롬프트(`Describe the video and transcribe what is said.`)가 더하는 텍스트 5토큰이다. 잃은 것도 이중으로 센 것도 없고, 이것이 두 확장이 서로를 덮어쓰지 않는다는 가장 깨끗한 증거다.

단일 모달리티 실행은 변경 전 바이너리와 확장 라인·생성 텍스트 모두 바이트 동일하다. 비디오 단독은 `The video shows a black screen with a white text "The video is black" appearing in the center.`를, 오디오 단독은 `The red square moves from the left side of the frame to the right side.`를 생성한다.

결합 실행을 `-n 160`으로 다시 돌리면 75토큰에서 스스로 멈추며 `The video shows a person in a black shirt and pants standing in front of a white wall. They are holding a white object in their hands and are moving it from left to right. The person is speaking in a clear and articulate voice.`에 이어 `The transcript of the video is as follows: "The frame moves from the left side of the frame to the right side."`를 내놓아 사운드트랙을 거의 그대로 재현한다. 이 변경 전에는 같은 명령이 `Error: Combined --video and --audio inputs are not supported yet`로 1을 반환하며 끝났다.

### 4.2 서버

`mlxcel-server --port 19349`, `MLXCEL_VIDEO_DIR_ALLOWLIST`는 픽스처 디렉터리를 가리킨다. `video_url` 파트(`file://` 절대 경로), `input_audio` 파트(base64 wav), 동일 텍스트 프롬프트를 담고 `max_tokens` 64, `temperature` 0으로 보낸 `POST /v1/chat/completions`는 HTTP 200과 `prompt_tokens` 637을 반환했다. CLI와 정확히 일치한다. 본문은 `The video shows a person's hand moving from the left side of the frame to the right side.`에 이어 전사 시도였다. 같은 서버의 단일 모달리티 대조군은 200과 함께 114(오디오 단독, 전사 정확)와 537(비디오 단독) 프롬프트 토큰을 반환해 각자의 CLI 대응물과 토큰 단위로 일치했다.

### 4.3 음성 대조군

`generate -m models/gemma-4-e4b-it-4bit ... --video clip.mp4 --audio speech.wav`는 여전히 `Combined --video and --audio inputs are not supported yet`로 1을 반환하며 끝난다. 이 체크포인트는 `model_type: gemma4`이고 12층 Conformer 오디오 인코더를 로드하므로 비디오도 오디오도 각각은 진짜로 소비한다. 논리곱이 참이 되었을 실제 체크포인트라는 점에서, 플래그가 `audio && video`보다 좁다는 사실에 대한 가장 강한 대조군이다.

### 4.4 2차 패스: 브랜치 헤드에서 전체 재실행

위의 패스는 `06141567`이 들어오기 전 릴리스 바이너리에서 취한 것이다. 그 커밋(3절에서 설명한 fetch 이전 미디어 게이트와 비디오보다 앞선 오디오 디코드) 이후, 전체 세트를 `target/test-fast/mlxcel`과 `target/test-fast/mlxcel-server`로 다시 돌렸다. 릴리스를 상속하고 `opt-level = 3`을 유지하는 `[profile.test-fast]` 빌드다. 체크포인트도 Apple GPU(Metal)도 그리디 디코드도 동일하다.

픽스처는 다시 만들어야 했다. 머신이 재부팅되면서 `/tmp`가 비워졌기 때문이다. 빨간 사각형이 왼쪽에서 오른쪽으로 움직이는 4.25초 320x240 클립과, macOS `say` wav를 16 kHz 모노로 리샘플한 55550 샘플 3.5초 음성이며, 후자는 소프트 토큰 87개로 청킹된다. 1차 패스의 wav는 93개였다. 그래서 2차 패스는 같은 총합의 반복이 아니라 산술의 검사가 된다.

| 실행 | 확장 라인 | CLI 총 프롬프트 토큰 | 서버 `prompt_tokens` |
|---|---|---|---|
| `--video clip.mp4` | `expanded 1 video(s) into 8 frame slot(s)` | 537 | 537 |
| `--audio speech.wav` | `expanded audio into 87 soft tokens` | 108 | 108 |
| 둘 다 | `expanded 1 video(s) into 8 frame slot(s) and audio into 87 soft tokens` | 631 | 631 |

631 = 537 + 89 + 5이다. 같은 비디오 단독 프롬프트에, 소프트 토큰 87개를 BOA와 EOA가 감싼 오디오 런, 그리고 같은 텍스트 5토큰이다. 1차 패스의 637 = 537 + 95 + 5와 대응한다. 오디오 길이가 바뀐 픽스처에서도 가법적 회계가 살아남았고, 애초에 보이려던 성질이 바로 그것이다.

단일 모달리티 생성 텍스트는 세 커밋 뒤의 바이너리에서도 4.1에 기록된 것과 바이트 동일하다. 비디오 단독은 여전히 `The video shows a black screen with a white text "The video is black" appearing in the center.`를, 오디오 단독은 여전히 `The red square moves from the left side of the frame to the right side.`를 낸다. 결합 CLI 실행은 `-n 160` 안에서 스스로 멈췄고, 음성 대조군도 그대로다. `models/gemma-4-e4b-it-4bit`는 `Error: Combined --video and --audio inputs are not supported yet`로 1을 반환하며 끝난다.

포트 19349의 서버에 `MLXCEL_VIDEO_DIR_ALLOWLIST`를 픽스처 디렉터리로 걸고 보낸 세 요청 모두 표의 프롬프트 토큰 수 그대로 HTTP 200을 반환했고, 각각 자기 CLI 값과 토큰 단위로 일치했다. `max_tokens` 200에서 결합 요청은 완료 토큰 70개 뒤 `finish_reason` `stop`으로 끝났으며, 본문은 `The video shows a person's hand moving a small, white, rectangular object across a dark surface. The object is being moved in a repetitive, back-and-forth motion. The background is dark and out of focus.`에 이어 `The transcript of the video is as follows:`, 그리고 `"I'm from the left side to the right side."`였다.

같은 결합 바디를 `/apply-template`와 `/v1/chat/completions/input_tokens`에 보내면 200이 돌아온다. `06141567`이 추가한 게이트의 양성 쪽이다. 이 라우트들은 이제 생성 라우트가 거부하는 바디를 거부하면서도, 결합을 소비할 수 있는 유일한 계열은 나머지와 싸잡히지 않고 그대로 받아들인다.

**감추지 않고 적어 두는 불일치 하나.** 토큰이 동일한 프롬프트에 그리디 디코딩을 걸어도, 결합 자유 응답의 문면은 CLI와 서버 사이에서, 그리고 두 패스 사이에서 달라진다. 4.1과 4.2의 1차 패스 기록에도 이 프롬프트에서 같은 CLI 대 서버 불일치가 이미 남아 있다. 따라서 두 확장이 합성된다는 증거는 토큰 회계와 바이트 동일한 단일 모달리티 응답이지, 결합 응답의 문면이 아니다.

### 4.5 추가된 테스트 커버리지

| 모듈 | 고정하는 것 |
|---|---|
| `vision::merge` | 1건. `merge_multimodal`이 도는 2연산 조합(`video_token_id`에 대한 `merge_llava` 다음 `audio_token_id`에 대한 `masked_scatter`)을 모달리티별로 다른 상수로 돌려, 각 런이 자기 특징을 받고 텍스트 행이 살아남는 것을 단언 |
| `multimodal::vlm_runtime` | 3건. 비디오 스플라이스(BOS 뒤)와 오디오 스플라이스(마지막 `<end_of_turn>` 앞)가 어느 순서로도 합성되는 것, 템플릿이 렌더링한 프롬프트가 각 플레이스홀더 id를 독립적으로 확장하는 것 |
| `server::media` | 3건. 각 모달리티를 단독으로만 받는 계열에 대한 거부, `video_with_audio` 아래에서의 허용, 단일 모달리티 요청 무간섭, 텍스트 전용 체크포인트에서 모달리티별 거부가 결합 거부를 이기는 것 |
| `server::startup` | 검출 테스트 5건에 `video_with_audio` 단언 추가. `gemma4_unified`에 양성, Gemma 4 VL·Kimi-VL 2.5·Inkling·Qwen3.5 VL·config 누락 폴백에 음성 |
| `server::chat_request` | 기존 거부 테스트를 다시 써서, 준비 단계가 더 이상 결합 거부를 내지 않음을 단언 |
| `server::router_front` | 1건. fetch 이전 술어가 요청 자체에서 이미지·오디오·비디오 파트를 거부하고 텍스트는 통과시키는 것 |
| `server::routes::prompt_inspection` | 3건. 세 경로 모두에서 미디어 파트가 가져와 세어지는 대신 거부되는 것, 결합이 여기서도 거부되는 것, 텍스트 전용 바디 무영향 |
| `server::routes::anthropic` | 2건. 결합은 `400 invalid_request_error`로, 모달리티별 거부는 `501 not_supported_error`로 렌더링 |

### 4.6 게이트

`cargo fmt --all -- --check`가 통과하고 `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`가 진단 없이 0으로 종료한다. `--profile test-fast --features metal,accelerate` 아래 좁은 범위는 모두 초록이다. `server::chat_request` 104 통과, `server::media` 75, `server::startup` 74, `server::router_front` 28, `server::routes::prompt_inspection` 16, `server::routes::anthropic` 17, `multimodal::vlm_runtime` 53, `vision::merge` 5, `vision::gemma4_unified` 19.

---

## 5. 검증하지 못한 것

- **전체 `--workspace` 게이트를 로컬에서 돌리지 않았다.** 머신을 다른 빌드 에이전트 일곱과 공유 중이고 타깃 디렉터리가 사용 중이다. 4.6의 범위가 이 변경이 건드리는 모든 모듈을 덮고, 나머지는 머지 게이트가 돌린다.
- **오디오 픽스처는 합성이다.** `tests/fixtures` 아래에 음성 샘플이 없어서, macOS `say` TTS가 16 kHz로 `The red square moves from the left side of the frame to the right side.`를 읽은 것을 썼고, 빨간 사각형이 왼쪽에서 오른쪽으로 움직이는 4.25초짜리 생성 클립(기본 2.0 fps에서 8프레임)과 짝지었다. 모델은 변경 전 기준선을 포함해 모든 비디오 실행에서 이 클립의 프레임을 검은 화면으로 보고하므로, 위의 시각 묘사는 시각 정확성에 대해 사실상 정보를 담지 않는다. 그것들이 확립하는 것은 프롬프트가 조립되고 토큰 수가 가법적이며 오디오가 백본에 도달한다는 사실이다.
- **어텐션 희석은 측정했을 뿐 고치지 않았다.** 기본 2.0 fps에서 비디오 토큰 528개가 짧은 프롬프트를 압도하며, `What sentence is spoken in the audio? Reply with the sentence only.`로 물었을 때 `The audio is not provided.`라고 답했다. 같은 질문을 `fps` 0.25(프롬프트 382토큰)로 던지면 `The red square moves from the left side of the frame to the right side.`, 즉 전사 그대로를 답한다. 가법적 토큰 회계 및 바이트 동일한 단일 모달리티 실행과 함께 읽으면, 이는 프레임에 정보가 없는 픽스처에 대한 어텐션 희석이지 배관 결함이 아니다. 측정한 그대로 보고하며, 혼합 프롬프트에 기본 fps가 적절하다는 주장은 하지 않는다.
- **결합 응답의 문면은 재현되지 않으며, 증거도 아니다.** 토큰이 동일한 프롬프트에 그리디 디코딩을 걸어도 결합 자유 응답의 문면은 CLI와 서버 사이에서, 그리고 두 검증 패스 사이에서 달라진다(4.2, 4.4). 이 보고서는 그 원인을 조사하지 않고, 어떤 주장도 거기에 기대지 않는다. 두 패스가 고정하는 것은 토큰 회계와 바이트 동일한 단일 모달리티 응답이다.
- **요청당 오디오 클립 하나 제한은 그대로다.** `require_single_server_audio_clip`이 다른 개수를 여전히 거부하고, 결합 경로는 이를 가장 먼저 검증한다.
- **Anthropic 결합 arm은 오늘 도달 불가다.** `AnthropicContentBlock`에 video나 audio 변형이 없어 단위 테스트로만 구동된다.
- **다른 계열은 넓히지 않았다.** `video_with_audio`의 `true` arm은 서로를 미러링하는 두 자리에 정확히 하나씩뿐이고, 음성 대조군은 합성 config가 아니라 실제 체크포인트다.

---

## 6. 파일

| 파일 | 역할 |
|---|---|
| `src/vision/gemma4_unified.rs` | `merge_multimodal`의 최광폭 래퍼 `get_input_embeddings_with_video_and_audio`, `MISSING_AUDIO_EMBEDDER_REFUSAL` |
| `src/server/state.rs` | `ModelMediaSupport::video_with_audio` |
| `src/server/startup.rs` | `detect_model_media_support`가 `ModelType::Gemma4Unified`에만 플래그를 세운다 |
| `src/loaded_model_capabilities.rs` | 워커 측 짝 `LoadedModel::supports_video_with_audio` |
| `src/server/media.rs` | `COMBINED_VIDEO_AUDIO_REFUSAL`, `media_capability_rejection`의 결합 arm, `is_combined_video_audio_rejection` |
| `src/server/chat_request.rs` | 준비 단계 bail 제거, 그 자리에 새 주인을 지목 |
| `src/server/model_worker.rs` | 모달리티별 헬퍼, 오디오 디코드/확장 분할, `prepare_gemma4_unified_video_and_audio_embeddings`, 백스톱 arm |
| `src/commands/generate_vlm.rs` | CLI 헬퍼와 `compute_gemma4_unified_video_and_audio_embeddings` |
| `src/server/router_front.rs` | `request_declares_media`. 렌더링이 fetch하기 전에 거부 |
| `src/server/routes/prompt_inspection.rs` | `render_chat_prompt`에서 툴 가드보다 앞선 `media_capability_rejection` |
| `src/server/routes/anthropic.rs` | 400 / 501 분기 `media_rejection_response` |
| `src/multimodal/vlm_runtime.rs` | `VlmPreparationSummary::Gemma4VideoAudio` |
| `docs/supported-models.md`, `src/main.rs` | 계열 항목과 `--video` / `--audio` 도움말 |
