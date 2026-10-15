# PR #2327: 이미지 지원이 없는 템플릿에서도 CLI가 비디오 콘텐츠 항목을 렌더링하도록 수정

**날짜**: 2026-10-16
**상태**: 구현 완료, CLI 테스트의 되돌림 검사와 실제 체크포인트(`qwen3-vl-2b-instruct-4bit`)는 머지 전 검증, video 전용 템플릿 경우는 합성 템플릿으로만 검증 ("미검증 및 후속" 참고)
**위험도**: 낮음 (프롬프트 구성만 바뀜, image 분기가 있는 템플릿은 기존과 같고, 서버 변경은 템플릿이 렌더링할 수 없던 항목만 제거)

## 요약

이슈 #1789는 `video` 분기만 있고 `image` 분기가 없는 채팅 템플릿에서 CLI와 서버의 프롬프트가 달라지는 문제를 고친다.

1. `apply_vlm_chat_template`(`src/commands/generate.rs`)는 템플릿이 image, video, audio 중 하나라도 처리하면 타입이 있는 콘텐츠 리스트를 만들고, 항목은 모달리티별로 결정한다. 클립의 선행 문장은 유지한다.
2. 서버(`src/server/chat_request.rs`)는 raw 렌더 게이트를 유지하고, 템플릿에 분기가 없는 타입 지정 `image`, `video` 항목을 제거하는 `drop_unsupported_media_items`를 추가한다.
3. 테스트 두 개가 양쪽에서 같은 프롬프트를 고정한다: `cli_video_only_template_renders_the_server_prompt`와 `video_only_template_renders_the_cli_prompt`.

## 문제 정의

CLI는 템플릿에 `image` 분기가 없으면 항상 턴을 텍스트로 평탄화했다. 그래서 `video` 분기만 있고 `image` 분기가 없는 템플릿은 서버의 raw 렌더에서는 비디오 항목을 받지만 CLI에서는 아무것도 받지 못했다. 비디오 하나와 질문 "What moves?"가 있는 요청에서 서버는 `<|user|><VIDEO>What moves?<|end|><|assistant|>`를, CLI는 `<|user|>What moves?<|end|><|assistant|>`를 렌더링했다. 모델이 보는 프롬프트가 진입 경로에 따라 달랐다.

## 기술적 선택과 그 이유

- 리스트 형태 게이트는 "image, video, audio 중 하나라도"로 유지한다. 서버의 raw 렌더 게이트와 같은 조건이다. 서버 게이트를 모달리티별 검사로 좁히면 `vlm_chat_template_omits_video_when_template_lacks_video_support`가 깨졌으므로, 게이트는 넓게 두고 리스트 안에서 항목을 걸러낸다.
- CLI에서는 항목을 모달리티별로 결정한다. image 항목과 폴백 프레임은 image 분기가 있을 때만, video 항목은 video 분기가 있을 때만, audio는 audio 분기가 있을 때만 넣는다. 프레임이 빠져도 폴백 프레임 그룹의 선행 문장은 그대로 내보내므로 두 경로의 텍스트가 같다.
- CLI가 모든 항목을 내보내게 바꾸는 대신 서버에 `drop_unsupported_media_items`를 추가해, 한 모달리티만 렌더링하는 템플릿에서 두 경로가 같은 항목을 받게 한다. audio 항목은 건드리지 않는다. 서버에서 audio는 계속 평탄화된다.
- image 분기 없이 video 분기만 가진 체크포인트가 알려진 것이 없어 합성 템플릿으로 테스트한다. 같은 템플릿 문자열을 양쪽에 고정해 두 프롬프트를 하나의 리터럴과 비교한다.

## 변경 사항

- `src/commands/generate.rs`: `emit_image`, `emit_video`, `emit_audio`를 먼저 계산하고, 평탄화 게이트를 "셋 다 아님"으로 바꾸고, image 항목과 폴백 프레임은 `emit_image`로 제어한다.
- `src/server/chat_request.rs`: 렌더링 전에 raw JSON 메시지에 `drop_unsupported_media_items`를 호출한다.
- `src/commands/generate_tests.rs`(새 테스트, 59줄), `src/server/chat_request_tests.rs`(새 테스트, 46줄).

## 동작 변화

`video` 분기만 있고 `image` 분기가 없는 템플릿에서 CLI가 이제 비디오 항목을 포함하며, CLI와 서버 모두 템플릿이 렌더링할 수 없는 image 항목을 생략한다. image 분기가 있는 템플릿과 세 분기가 모두 없는 템플릿은 기존과 같다.

## 검증

- 되돌림 검사: 수정을 되돌리면 `cli_video_only_template_renders_the_server_prompt`가 실패하며 `<|user|><VIDEO>What moves?<|end|><|assistant|>` 대신 `<|user|>What moves?<|end|><|assistant|>`를 렌더링한다. 서버 쌍둥이 테스트는 서버가 원래 비디오를 raw 렌더했으므로 변경 유무와 관계없이 통과하며, 되돌림 증거는 CLI 테스트가 담당한다.
- `--features cuda --profile test-fast -- --test-threads=1`로 `commands::generate`(87개 통과)와 `server::chat_request`(133개 통과). `-D warnings` clippy, `cargo fmt --check`, 계약 스크립트 모두 통과.
- 실제 체크포인트 `qwen3-vl-2b-instruct-4bit`와 ffmpeg `testsrc` 클립: CLI가 렌더링한 프롬프트와 서버 `POST /apply-template` 출력이 바이트 단위로 같고, `mlxcel generate --video`가 정상 동작한다.

## 미검증 및 후속

- image 분기 없이 video 분기만 가진 체크포인트가 로컬이나 알려진 범위에 없다(Qwen-VL과 Gemma 4 템플릿은 둘 다 가진다). 실제 체크포인트 실행은 기존 image+video 경로만 다루며, video 전용 경우는 합성 템플릿 테스트에 의존한다.
- 서버에서 audio는 여전히 평탄화되고 CLI는 audio 분기가 있을 때만 audio 항목을 내보내므로, audio 분기만 있는 템플릿은 일치 테스트로 고정되어 있지 않다.
