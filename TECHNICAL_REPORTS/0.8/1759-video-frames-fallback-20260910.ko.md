# 기술 보고서: PR #1759 - feat(server): 네이티브 비디오 경로가 없는 모델에 샘플 프레임을 순서대로 보낸다

**날짜**: 2026-09-10
**작성**: mlxcel maintainers
**검토**: 구현 및 리뷰, 보안 후속 사이클
**상태**: 완료 (Apple Silicon / Metal에서 `models/gemma-3-4b-it-4bit`, `models/lfm2-vl-450m-4bit`, `models/qwen2.5-vl-3b-instruct-4bit`, `models/gemma-4-e4b-it-4bit`로 검증. `--workspace` 게이트는 머신을 공유 중이라 로컬에서 돌리지 않았고, 좁은 범위 테스트와 CI가 대신 덮는다)
**언어**: Rust, Markdown
**위험도**: 중간 (소수 계열을 뺀 전 계열에 걸려 있던 거부가 풀리고, 미디어 지원 플래그 하나가 둘로 갈라지며, 렌더링 전에 요청 자체를 고쳐 쓴다)

---

## 요약

비디오 입력은 시간축 경로를 직접 만든 계열에만 닿았다. 나머지, 그러니까 비전 언어 모델 목록의 대부분은 클립을 받으면 곧바로 거부했다. CLI는 `--video input is currently only supported by ...`, 서버는 HTTP 400에 `video_url content blocks are not supported by model '...'`이다. 이상한 대목은 그 체크포인트들이 정지 이미지 여러 장은 아무 불평 없이 읽는다는 것이다. 능력은 있었고 배관만 없다고 했다.

이번 변경은 그 관찰을 쓴다. 비전 타워가 있고 네이티브 비디오 경로가 없는 체크포인트는 이제 요청한 fps로 클립을 디코드하고, 첫 프레임과 마지막 프레임을 반드시 남기면서 `--video-max-frames` 장 이하로 균등 샘플링하고, PNG로 인코딩해서, 이것이 한 편의 비디오라고 알리는 문장 한 줄 뒤에 순서대로 `image_url` 파트로 끼워 넣는다. 하류에는 새로 가르치는 것이 없다. 템플릿은 프레임당 이미지 자리 하나를 내고, 요청당 이미지 예산도 소프트 토큰 회계도 프레임을 센다. 프롬프트 캐시의 멀티모달 다이제스트는 프레임 바이트를 해시한다. 그 시점에는 프레임이 이미 이미지이지 특수 사례가 아니기 때문이다.

설계의 핵심은 고쳐 쓰는 위치다. 자리는 하나다. HTTP 경계의 `media_capability_rejection` 뒤, `prepare_chat_request_with_cache`가 렌더링하기 전이다. 이보다 앞이면 체크포인트가 클립을 볼 자격이 있는지 판정도 하기 전에 돌고, 뒤면 모델에 자리표가 없는 `video_url` 파트를 두고 템플릿이 이미 렌더링을 마친 뒤다.

치환이 원본에 충실하다는 증거는 서술이 아니라 대조군이다. 프레임 4장 조건에서, 같은 질문을 `video_url`로 물었을 때(프롬프트 1112토큰)와 같은 프레임 4장을 호출자가 직접 `image_url` 파트 4개로 넘겨 물었을 때(1097토큰)의 답이 완전히 같다. 15토큰 차이는 앞머리 문장 한 줄이고 그것뿐이다.

---

## 1. '네이티브'의 정의가 있어야 할 자리

### 1.1 목록 하나를 두 프런트가 읽는다

이전에는 비디오를 받는 계열 집합이 두 군데에 적혀 있었다. `server::startup::detect_model_media_support`는 `matches!` 11개 팔로 `ModelMediaSupport { video }`를 정했고, `commands/generate_vlm::compute_vlm_embeddings`는 로드된 모델로 분기하면서 자기 팔을 따로 두고, 마지막에 계열 이름을 산문으로 나열한 에러로 떨어졌다. 반드시 일치해야 하는 목록 둘이 다른 파일에 있고 둘을 묶어 두는 장치는 없었다. 고장 나는 방식은 정해져 있다. 서버는 받는데 CLI는 거부하는 계열이 생기거나, 그 반대가 생긴다.

이제 술어는 `src/models/detection.rs`의 `get_model_type` 옆에 한 번만 있다.

```rust
pub fn model_type_has_native_video(model_type: ModelType) -> bool
```

`detect_model_media_support`는 이것을 불러 `video_native`를 계산하고, `commands::generate`는 `--video`에 폴백이 필요한지 판단할 때 같은 것을 부른다. 시간축 경로가 생긴 계열은 한 곳에만 추가하면 두 프런트가 따라온다. 함수 위 주석은 함께 고쳐야 할 분기 지점 두 곳을 지목하는데, 이 저장소가 공유 함수에 쓰는 발견 장치가 그것이다.

`model_type_is_vision_capable`을 옆에 함께 내보내는 이유도 같다. 바이너리 크레이트의 `--video` 처리에도 모델 레지스트리가 이미 계산해 둔 `config.json` 기반 VLM 술어가 필요하고, CLI에서 다시 유도하면 목록이 셋이 된다.

### 1.2 `ModelMediaSupport::video`는 둘로 갈라지되 뜻은 그대로다

`video: bool`은 "이 요청이 `video_url` 파트를 실어도 되는가"라는 질문 하나에 답했고, 그 질문의 답은 지금도 하나다. 다만 답하는 두 방식은 요청을 받아 주느냐가 아니라 모델에 실제로 무엇이 닿느냐에서 갈리므로 플래그를 둘로 나눈다.

```rust
pub video_native: bool,
pub video_frames_fallback: bool,

pub const fn video(&self) -> bool { self.video_native || self.video_frames_fallback }
```

`media_capability_rejection`은 `support.video()`를 읽고 전과 똑같이 동작한다. `expand_video_parts_to_frames`는 `video_frames_fallback`을 읽고 네이티브 계열에는 손대지 않은 채 `Ok(0)`을 돌려주며, 그래서 그 체크포인트들의 `prepared.videos`가 그대로 채워진 채 남는다. `/props`는 `video()`를 보고한다. "이 서버가 비디오를 받는가"라는 와이어 수준의 답은 달라지지 않았기 때문이고, 둘을 구분해 주는 것은 로그 줄이다.

폴백은 계열 목록이 아니라 `multimodal && !video_native`로 건다. 내일 들어오는 VLM은 들어오는 날 폴백을 얻고, 나중에 진짜 시간축 경로가 생긴 계열은 둘 다 하는 대신 조용히 대체 경로를 쓰지 않게 된다.

### 1.3 예외 하나

Muse Glimmer는 이름으로 제외한다. CLI는 이미 `validate_muse_glimmer_cli_unsupported_options`에서 이 계열의 `--video`를 거부하고 있고, HTTP 경계에서만 클립을 받아 주면 한 체크포인트를 두고 두 프런트가 어긋난다. 1.1이 막으려는 실패가 정확히 그것이다. 다중 이미지 프롬프트에 이 계열이 검증되면 두 가드를 함께 푼다.

---

## 2. 고쳐 쓰기

### 2.1 위치를 먼저 모으고 디코드는 그다음

`expand_video_parts_to_frames_with_allowlist`는 디코드 전에 모든 `video_url` 파트의 `(메시지 인덱스, 파트 인덱스, VideoUrl)`을 먼저 모은다. 디코드가 비동기이고 삽입이 그 뒤의 모든 인덱스를 밀기 때문이다. 한 번에 처리하면 걷는 도중 밑에서 움직이는 목록을 걷게 된다.

디코드 자체는 이미지 경로와 같이 `tokio::task::spawn_blocking`에서 돈다. 긴 클립이면 ffmpeg와 PNG 인코딩이 CPU를 수 초 쓰고, 그것을 Tokio 워커에 둘 이유가 없다.

### 2.2 살아남을 프레임만 디코드한다

처음 판은 요청한 fps로 클립 전체를 디코드하고 그다음에 샘플링했다. 이슈가 적어 둔 모양이 그것이다. 그런데 이는 최대 메모리를 48배로 부풀리고, 보안 검토가 이를 물린 것은 옳다. `load_video_source`는 fps를 받으면 최대 `FPS_MAX_FRAMES`(768)장의 원해상도 프레임을 한꺼번에 들고 있고, 폴백은 그중 16장을 쓴다. 배포 기본 한도(`MLXCEL_VIDEO_MAX_PIXELS` 4096x4096, `MLXCEL_VIDEO_MAX_DURATION_SEC` 600)에서는 사진 열여섯 장을 보내려고 RGB 약 38GB를 만드는 셈이다. 특별한 `video_url.fps`가 필요하지도 않다. 기본값 2.0fps로도 6분쯤 넘는 클립이면 768 천장에 닿는다. 게다가 폴백은 네이티브 다섯 계열이 아니라 이미지를 받는 사실상 전 VLM에서 켜지므로, 노출 범위가 구석이 아니라 목록 전체다.

`load_video_source_frames_fallback`은 컨테이너를 한 번 프로브하고, 로그가 "몇 장 중 몇 장"을 그대로 말할 수 있도록 `target_fps` 단독으로 샘플했을 개수를 계산한 다음, `min(sampled, max_frames)`장만 디코드한다. 남는 프레임은 클립 전체에 걸친 같은 균등 간격이다. 균등 표본의 균등 표본은 균등 표본이기 때문인데, 논증이 아니라 그것을 실측한 것이 5.4다. `subsample_evenly`는 뒤에 그대로 돌고 이제 거의 항상 항등이다. `max_frames`가 약속이 아니라 상한이고, CLI와 테스트가 이 함수를 직접 쓰기 때문에 남겨 둔다.

`subsample_evenly`는 원소 타입에 대해 제네릭이라 디코드된 프레임에도 인코딩된 버퍼에도 같은 간격을 적용할 수 있지만, 모든 호출자가 싼 쪽에 쓴다.

간격은 `i in 0..max`에 대해 `round(i * (len - 1) / (max - 1))`이고, `evenly_spaced_indices`로 따로 노출한다. `uniform_indices`는 이웃한 질문(어느 프레임을 디코드할 것인가)에 답하지 이 질문(디코드된 것 중 무엇을 남길 것인가)에 답하지 않기 때문이다. 첫 장과 마지막 장은 항상 남고, `max_frames <= len`이면 간격이 1 이상이므로 결과는 항상 증가한다.

JPEG이 아니라 PNG인 이유는, 잠시 뒤 이미지 경로가 이 프레임을 다시 디코드하는데 손실 왕복을 거치면 원본 클립에 없던 아티팩트가 비전 타워 앞에 놓이기 때문이다.

### 2.3 삽입은 뒤에서 앞으로

`apply_video_frame_expansion`은 확장을 역순으로 적용해서 앞쪽 삽입이 뒤쪽 파트의 인덱스를 밀지 못하게 한다. `video_url` 파트 하나는 `Here is a video as a sequence of N frames in chronological order.`를 담은 텍스트 파트 하나와 `data:image/png;base64,...`를 담은 `image_url` 파트 N개로, 시간 순서대로 바뀐다. 클립을 둘러싸고 있던 파트들은 클립에 대한 상대 위치를 지키고, 그래서 사진과 클립을 섞어 보낸 요청이 호출자가 쓴 순서 그대로 나온다.

이 함수를 I/O 쪽에서 떼어 놓은 것은 의도적이다. 위쪽은 전부 allowlist 해석과 ffmpeg와 인코딩이고, 여기는 전부 순서 계약이다. 테스트는 합성 바이트로 이쪽 절반만 직접 돌리므로 ffmpeg가 없는 호스트에서도 계약이 덮인다.

### 2.4 예산 거부는 프레임을 지목하고, 청구서보다 먼저 온다

프레임은 평범한 이미지가 되고 요청당 이미지 예산을 평범하게 쓴다. 예산을 넘긴 요청은 잠시 뒤 `validate_image_count`가 거부하지만, 그 메시지는 호출자가 보낸 적 없는 이미지 개수를 말하므로 배포에 `--video-max-frames`가 너무 높다는 뜻이 아니라 서버 버그처럼 읽힌다. `video_frame_budget_rejection`이 먼저 거부하면서 주입된 프레임 수, 호출자 자신의 이미지 수, 한도를 함께 적고 이를 푸는 두 플래그를 지목한다.

무엇을 말하느냐만큼 어디서 검사하느냐도 중요하다. 처음 판은 바디의 모든 클립을 해석하고 디코드하고 PNG로 인코딩한 뒤에 한 번 검사했다. `video_url` 파트 개수에는 상한이 없고 JSON 바디 한도는 약 1.4GB이므로, 작은 비디오 참조를 잔뜩 실은 바디는 어차피 받을 거부를 받기 전에 클립마다 프로브와 디코드를 다 사들였다. 이제 그 작업 앞에 검사 둘이 선다. 첫 ffprobe 전의 클립 개수 가드는 클립 하나가 최소 프레임 이미지 하나를 내므로 `max_images_per_request`보다 많은 클립은 애초에 감당할 수 없다는 사실에서 나오고, 프레임 검사는 루프 안으로 들어가 예산을 깬 클립에서 멈춘다. `video_expansion_refuses_more_clips_than_the_image_budget_before_decoding`이 앞의 것을 고정한다. URL이 아무것도 가리키지 않게 두었으므로 해석기까지 갔다면 다른 에러가 났을 것이다.

클립을 보냈다고 알리는 `info` 줄은 클립별 거부 뒤에 놓인다. 거부된 요청이 예산을 깬 그 클립을 보냈다는 로그를 남기지 않게 하기 위해서다.

이 모두가 강제가 아니라 친절한 선점이다. 미디어 경로가 볼 때쯤 프레임은 진짜 `image_url` 파트이므로 `validate_image_count`와 `validate_resolved_image_count`가 이미지마다 그대로 돌고, 예산을 빠져나갈 수는 없다.

---

## 3. 연결 지점

| 프런트 | 연결 | 이유 |
|---|---|---|
| `routes/chat.rs` | 함 | `media_capability_rejection` 뒤, `validate_chat_tool_inputs`와 렌더링 앞. `chat_completions`가 스트리밍과 비스트리밍 경로를 함께 먹이는 유일한 입구라 한 번 호출로 둘 다 덮는다. |
| `routes/responses.rs` | 함 | 번역된 `ChatCompletionRequest`에, 같은 자리에서, 두 핸들러가 읽기 전에. |
| `routes/prompt_inspection.rs` | 함 | 이 라우트들은 "이 바디로 생성 라우트가 만들 프롬프트가 무엇인가"에 답하려고 있으므로 같은 치환을 돌려야 한다. 핸들러가 요청을 빌려 쓰고 생성은 하지 않으므로 복사본에 적용한다. |
| `routes/anthropic.rs` | 안 함 | `AnthropicContentBlock`의 변형은 text, image, document, tool_use, tool_result뿐이고 video는 없다. `anthropic_request_to_chat`이 `VideoUrl` 파트를 만들 수 없으므로 연결하면 닿지 않는 호출이 하나 는다. |
| `router_front.rs` | 안 함 | `route_chat`은 `request_declares_media`에 걸리는 것을 전부, `video_urls()` 포함해서, 렌더링 전에 거부한다. 분산 경로가 풀 기반 계열에 대해 텍스트 전용이기 때문이다. 확장이 돌 대상이 아예 없다. |

이슈의 구현 계획은 마지막 둘을 지목했다. 둘 다 이유와 함께 PR 본문에 적어 두었으므로, 리뷰어가 빠뜨린 것으로 읽지 않는다.

`model_worker::prepare_request_video_embeddings`의 워커 가드는 남기고 문구만 고쳤다. 이제 이것은 호출자가 평소에 만나는 거부가 아니라 뒤를 받치는 장치다. 네이티브 경로가 없는 이미지 지원 체크포인트는 HTTP 경계에서 클립이 고쳐 쓰이므로 요청이 여기 도착할 때는 비디오가 아예 없다. 이 팔에 닿았다는 것은 어떤 라우트가 확장을 건너뛰었다는 뜻이고, 메시지가 그렇게 말한다.

---

## 4. CLI

`src/commands/generate.rs`의 `expand_cli_videos_to_frames`는 표현만 다른 같은 변환이다. 프레임을 시스템 임시 디렉터리에 PNG로 쓰고, 그 경로를 `args.generation.image`에 붙이고, 앞머리 문장을 `user_prompt` 앞에 놓고, `args.generation.video`를 비워서 이 경로에서는 `compute_vlm_embeddings`가 클립을 보지 않게 한다.

여기서도 `VideoSource::from_path`를 거쳐 같은 `load_video_source_frames_fallback`을 부른다. 서비스가 아니라 로컬 프로세스 하나이므로 메모리 논거는 약하지만 정합성 논거는 그렇지 않다. 디코더가 둘이면 같은 `--fps`와 `--video-max-frames`에서 같은 클립을 두고 CLI와 서버가 서로 다른 프레임을 모델에 보여 줄 수 있고, 프런트마다 달라지는 답은 아무도 재현하지 못하는 버그 보고가 된다.

배치 제약이 셋이다.

- `--video`를 읽는 모든 검증기(파이프라인 병렬 검사, `--output-audio`, `--layout-detections`, Muse Glimmer 가드) 뒤에 돈다. 그래야 어느 것도 뜻이 달라지지 않는다.
- 프롬프트 렌더링 앞에 돈다. 그래야 `load_cli_prompt`가 프레임당 이미지 파트 하나를 센다.
- `TempFile` RAII 가드는 `run_generate_once`가 반환할 때까지 사는 변수에 묶인다. 반환 시점은 비전 타워가 PNG를 읽은 뒤다. 더 일찍 떨어뜨리면 읽는 쪽 밑에서 파일이 사라진다.

네이티브 계열, 비전 타워가 없는 체크포인트, Muse Glimmer에 대해서는 `Ok(None)`으로 물러난다. `generate_vlm.rs`의 거부는 아직 여기 닿는 유일한 경우, 즉 네이티브 경로도 비전 타워도 없는 체크포인트에 맞게 문구를 고쳤다.

---

## 5. 검증

이 트리에서 빌드한 릴리스 바이너리(`cargo build --release --features metal,accelerate`, exit 0), M5 Max / macOS 27.0. 픽스처는 흰 배경을 128x128 빨간 사각형이 왼쪽에서 오른쪽으로 가로지르는 8초 448x448 25fps H.264 클립이고, `ffmpeg -f lavfi -i "color=c=white:s=448x448:d=8:r=25" -f lavfi -i "color=c=red:s=128x128:d=8:r=25" -filter_complex "[0:v][1:v]overlay=x='(W-w)*t/8':y=(H-h)/2"`로 만들었으며 첫 프레임과 마지막 프레임을 눈으로 확인했다. 스크래치 파일이고 커밋하지 않았다.

픽스처 첫 시도는 `drawbox`에 `t` 의존 `x`를 준 것이었고 8초 내내 흰 화면이 나왔다. 모델들은 성실하게 백지라고 답했다. 이것을 적어 두는 이유는 그 실행이 성공한 검증처럼 보였기 때문이다. 로그 줄도 프레임 수도 토큰 증가도 다 맞았고, 어긋난 것은 답뿐이었다. 아래 수치를 믿기 전에 프레임을 눈으로 확인했다.

### 5.1 CLI, 네 번 모두 exit 0

| 사례 | 체크포인트 | 실행 결과 |
|---|---|---|
| 폴백 | `gemma-3-4b-it-4bit`, `--video-max-frames 8` | `model_type=Gemma3VLM has no native video path; sending 8 of 16 sampled frames from <clip> as ordered images` 뒤에 `Loaded 8 image(s).`, `Expanded 8 <image> token(s) to 256 tokens each`. 프롬프트가 앞머리 문장으로 시작한다 |
| 폴백 | `lfm2-vl-450m-4bit`, 기본 상한 | `model_type=Lfm2VL ... sending 16 of 16 sampled frames`, `Loaded 16 image(s).`, `LFM2-VL: inserted 16 image block(s) (3136 total image tokens)` |
| 네이티브 대조 | `qwen2.5-vl-3b-instruct-4bit` | 폴백 줄 없음. `Loaded 1 Qwen-VL video(s) (16 total frames after sampling)`, `1 video block(s) (2048 video tokens)` |
| 네이티브 대조 | `gemma-4-e4b-it-4bit` | 폴백 줄 없음. `Loaded 1 video(s) (16 total frames after sampling)`, `Gemma4: expanded 1 video(s) into 16 frame slot(s) (1079 total tokens)` |

### 5.2 서버, 토큰 회계

`mlxcel-server -m models/gemma-3-4b-it-4bit --port 19322 --video-max-frames 8`, `MLXCEL_VIDEO_DIR_ALLOWLIST`를 클립 디렉터리로 지정, `/v1/chat/completions`, 전 구간 HTTP 200.

| 요청 | `usage.prompt_tokens` |
|---|---|
| 텍스트만 | 19 |
| 텍스트 + 정지 PNG 1장 | 279 |
| 텍스트 + `video_url` | 2114 |

2114 - 19 = 2095이고, 프레임 8 x 260 = 2080에 앞머리 문장 15를 더한 값이다. 이슈가 요구한 증가가 토큰 단위로 맞는다.

로그 두 줄이 모두 나왔다. 기동 시 `model_type=Gemma3VLM: no native video path; video_url content blocks will be served as ordered sampled frames`, 요청마다 `model gemma-3-4b-it-4bit has no native video path; sending 8 of 16 sampled frames from <clip> as ordered images`.

`models/gemma-4-e4b-it-4bit`는 같은 바디에 네이티브 경로로 답하며 프롬프트 1080토큰을 썼고(CLI 실행의 프레임 슬롯 1079와 맞는다), 폴백 줄은 0번 찍었다. `GET /props`는 둘 다 `modalities.video: true`를 보고한다.

### 5.3 충실도 대조군

`gemma-3-4b-it-4bit`에 `--video-max-frames 4`, 그리디 디코딩으로 같은 질문을 두 방식으로 물었다.

- `video_url`로 물어 폴백이 일하게 한 경우: 프롬프트 1112토큰
- 같은 프레임 4장을 호출자가 `image_url` 파트 4개로 넘긴 경우: 1097토큰

두 번 다 답이 같다. 15토큰 차이는 앞머리 문장이다. 고쳐 쓴 요청은 호출자가 직접 프레임을 뽑아 보냈을 때 모델이 받았을 평범한 다중 이미지 요청이고, 이 변경이 기대는 주장이 그것이다.

### 5.4 경계 디코드는 모델이 보는 것을 바꾸지 않는다

5.1부터 5.3까지 모든 실행을 두 번 했다. 한 번은 원래의 디코드 후 샘플링으로, 한 번은 `load_video_source_frames_fallback`으로. 수치는 하나도 움직이지 않았다. 같은 `sending 8 of 16 sampled frames`, 같은 19 / 279 / 2114 프롬프트 토큰, 네이티브 대조군의 같은 1080과 여전히 없는 폴백 줄, 그리고 같은 1112 대 1097에 두 번 다 같은 답이다.

보안 수정에서 흥미로운 대목이 이것이다. "균등 표본의 균등 표본은 균등 표본"은 정확 산술의 이야기인데 두 경로는 반올림을 각각 두 번과 한 번 하므로, 남는 프레임이 한 칸 밀렸는지 묻는 것은 정당했다. 이 클립과 이 한도에서는 밀리지 않았고, 체크포인트 넷에 걸친 토큰 일치가 그렇게 말한다.

### 5.5 추가한 테스트

- `src/multimodal/video_tests.rs`: `subsample_evenly_keeps_first_and_last`, `subsample_evenly_identity_when_under_cap`, `subsample_evenly_indices_match_formula`(샘플 37, 상한 16, 인덱스 `[0, 2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29, 31, 34, 36]`), `subsample_evenly_indices_are_strictly_increasing`, `frames_to_png_round_trips_every_frame_in_order`.
- `src/server/chat_request_tests.rs`: `video_part_expands_to_ordered_image_parts_with_lead_text`, `video_expansion_preserves_surrounding_images`, `video_expansion_skipped_for_native_video_model`, `video_expansion_is_a_no_op_without_video_parts`, `video_expansion_changes_mm_digest_when_frames_change`, `video_frame_budget_refusal_names_the_frames`, `video_frames_lead_text_names_the_frame_count`, `video_expansion_refuses_more_clips_than_the_image_budget_before_decoding`, ffmpeg를 쓰는 `video_part_expands_through_a_real_clip`.
- `src/server/startup_tests.rs`: `media_support_marks_image_only_vlms_as_frames_fallback`, `media_support_keeps_native_video_families_native`, `media_support_gives_the_vit_gemma4_vlm_no_frames_fallback`, `media_support_denies_the_frames_fallback_to_text_only_and_muse_glimmer`.
- `src/commands/generate_tests.rs`: `cli_video_fallback_declines_native_and_text_only_checkpoints`, `cli_video_fallback_appends_frame_images_and_clears_video`.

ffmpeg를 쓰는 두 테스트는 저장소 관례(#1172)대로 `#[ignore]`이고, `MLXCEL_TEST_VIDEO=1 ... -- --include-ignored`로 실제로 돌려 둘 다 통과했다.

### 5.6 게이트

`cargo test --profile test-fast --features metal,accelerate`를 `--lib multimodal::video`(40 통과), `--lib server::chat_request`(112), `--lib server::startup`(78), `--lib server::media`(75), `--lib server::routes`(394), `--lib server::cli_input`(144), `--bin mlxcel commands::generate`(85)에 돌렸다. `commands::generate_tests`는 lib이 아니라 바이너리 크레이트에 있어서 `--lib commands::generate_tests`는 0개를 잡는다. `--bin` 범위가 맞는 쪽이다.

`cargo clippy --lib --tests`와 `--bins`를 `-D warnings`로, `cargo fmt --all -- --check`, `cargo check --lib --tests` 모두 깨끗하다. 리베이스한 헤드에 대한 GitHub CI도 `cargo-clippy`와 `cargo-fmt`를 포함해 전부 통과했다.

---

## 6. 검증하지 않은 것

- **`--workspace` 게이트 전체는 로컬에서 돌리지 않았다.** 머신을 빌드 에이전트 일곱과 공유 중이다. 5.5의 범위가 이번 변경이 건드리는 모든 모듈을 덮고, 나머지는 CI가 돈다.
- **움직임을 말로 풀어낸 체크포인트는 없었다.** `gemma-3-4b-it-4bit`는 흰 배경의 빨간 사각형을 묘사하고 그것이 계속 그대로라고 답한다. `lfm2-vl-450m-4bit`는 프레임을 하나씩 훑으며 각 장의 빨간 사각형을 묘사한다. 같은 클립에서 네이티브 경로도 나을 것이 없다. Qwen2.5-VL은 "a static image ... no changes or movements"라고 답한다. 이것은 체크포인트의 다중 이미지 시간 추론 능력이지 치환의 문제가 아니고, 둘을 갈라 주는 것이 5.3이다. 폴백이 작은 모델에 움직임을 추론하게 해 준다는 주장은 여기에 없고 문서도 같은 말을 한다.
- **여기서 Qwen2.5-VL은 폴백 사례가 아니라 네이티브 대조군이다.** 이슈는 폴백 검증 대상으로 이 체크포인트를 지목했고 제기 시점에는 맞는 말이었다. Qwen-VL은 #1166에서 네이티브 비디오 경로를 얻었으므로 이 트리에서는 네이티브가 그대로인지 확인하는 쪽을 맡고, 두 번째 폴백 계열 자리는 `lfm2-vl-450m-4bit`가 대신했다.
- **Anthropic 프런트와 분산 라우터 프런트는 비디오에 대해 닿지 않는 경로이고**, 요청을 실제로 보내 본 것이 아니라 타입과 가드에 대한 추론으로 덮었다.
- **서버의 `--video-fps`는 요청별 오버라이드가 아니다.** 요청 자신의 `video_url.fps`가 여전히 이기고, 플래그는 요청이 생략했을 때의 기본값을 준다.

---

## 7. 파일

| 파일 | 역할 |
|---|---|
| `src/multimodal/video.rs` | `DEFAULT_FALLBACK_MAX_FRAMES`, `MIN_FALLBACK_MAX_FRAMES`, `evenly_spaced_indices`, `subsample_evenly`, `frames_to_png`, `load_video_source_frames_fallback` |
| `src/models/detection.rs`, `src/models/mod.rs` | `model_type_has_native_video`, `model_type_is_vision_capable`, 두 프런트가 함께 읽는 단일 목록 |
| `src/server/state.rs` | `ModelMediaSupport::video_native` / `video_frames_fallback`과 `video()` 접근자 |
| `src/server/startup.rs` | `detect_model_media_support`가 두 플래그를 세팅. `ServerStartupConfig`의 `video_max_frames` / `video_fps`와 `build_server_config`의 클램프 |
| `src/server/chat_request.rs` | `VideoFramesFallback`, `expand_video_parts_to_frames(_with_allowlist)`, 클립 개수 검사와 클립별 예산 검사, `video_frame_budget_rejection`, `apply_video_frame_expansion` |
| `src/server/routes/chat.rs`, `routes/responses.rs`, `routes/prompt_inspection.rs` | 연결 지점 셋 |
| `src/server/media.rs` | `resolve_video_url`을 `pub(crate)`로 넓힘. `media_capability_rejection`이 `support.video()`를 읽음 |
| `src/server/model_worker.rs` | 네이티브 비디오 가드를 뒤를 받치는 장치로 남기고 문구 수정 |
| `src/commands/generate.rs` | `CliVideoFrames`, `expand_cli_videos_to_frames`, `run_generate_once`의 호출 지점 |
| `src/commands/generate_vlm.rs` | 아직 닿는 유일한 경우에 맞게 거부 문구 수정 |
| `src/main.rs`, `src/bin/mlx_server.rs`, `src/commands/serve.rs`, `src/server/cli_input.rs`, `src/server/config.rs` | 세 프런트에 걸친 `--video-max-frames`와 `--video-fps` |
| `docs/supported-models.md`, `docs/llama-server-compat.md`, `docs/environment-variables.md` | 네이티브와 폴백 구분, `video_url` 절, 새 환경 변수 둘 |
