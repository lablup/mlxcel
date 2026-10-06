# 기술 보고서: 이슈 #1683 - Granite Vision 설명형 프롬프트가 거절 응답을 받던 문제

**날짜**: 2026-10-06

**상태**: GB10(CUDA)에서 구현 및 검증 완료, 머지 대기. Apple Silicon 실행은 이 호스트에서 수행하지 않음.

**언어**: Rust (토크나이저 로더, Granite Vision 프롬프트 확장, CLI 요약, 테스트)

**위험도**: 낮음~중간 (`tokenizer_config.json`이 추가 토큰을 더 선언한 `tokenizer.json` 체크포인트의 토큰화가 바뀜. 로컬 체크포인트 172개 중 3개가 해당하며 실제로 바뀌는 것은 2개)

## 요약

`granite-vision-3.2-2b-4bit`는 색상 질문에는 정확히 답했지만 설명형 프롬프트에는 "이미지를 볼 수 없다"며 거절했습니다. 갈라지는 단계는 렌더링된 프롬프트의 토큰화입니다. 채팅 템플릿은 mlx-vlm 0.6.17과 같은 텍스트를 만들었지만, `<image>`(id 49155)가 체크포인트의 `tokenizer_config.json` `added_tokens_decoder`에만 있고 `tokenizer.json`에는 없습니다. 그래서 `tokenizers` 크레이트가 이를 `<`, `image`, `>`(`[46, 893, 48]`)로 쪼갰습니다. `insert_granite_vision_image_tokens`는 플레이스홀더를 찾지 못해 1485개 이미지 토큰을 첫 토큰, 즉 `<|system|>`의 `<` 뒤에 끼워 넣는 폴백을 탔습니다. 이미지 임베딩은 들어갔으므로 색상 질문은 통했지만, 깨진 시스템 마커 안쪽, 사용자 턴 바깥에 놓였고 사용자 턴에는 `<image>` 문자열이 그대로 남았습니다.

## 수정

- `src/tokenizer/added_tokens.rs`(신규): `tokenizer.json` 로드 후 빠진 `added_tokens_decoder` 항목을 `transformers`의 `PreTrainedTokenizerFast.__init__`와 같은 방식으로 등록합니다. 내용이나 id가 이미 있는 항목은 건너뛰고, 선언된 id가 크레이트가 다음에 부여할 id(`AddedVocabulary::add_tokens` 규칙)와 정확히 같을 때만 추가합니다. `special` 플래그가 같은 연속 구간 단위로 묶어 등록하고 id를 검증합니다. id에 공백이 있으면 로드를 실패시키지 않고 경고 후 중단합니다.
- `InsertedGraniteVisionTokens`와 Granite Vision / Granite 4 Vision 준비 요약에 `spliced`를 추가했습니다. CLI는 이제 제자리 확장으로 보고하는 대신 "no <image> placeholder in the prompt; spliced ..."를 출력합니다. 이번 실패가 정상처럼 보였던 이유가 이 출력이었습니다.

## mlx-vlm 0.6.17과의 단계별 비교 (CUDA, 같은 체크포인트와 이미지)

- 렌더링 텍스트: 동일.
- 프롬프트 id: 플레이스홀더에서 달랐음(`[46, 893, 48]` 대 `49155`). 수정 후 설명형/색상 질문 모두 1545개 id 전부 일치(나머지 두 프롬프트도 1538 / 1539).
- 픽셀 값: `(2 타일, 384, 384)` 레이아웃과 정규화 값 동일.
- SigLIP 탭과 프로젝터: mlx-vlm의 `nn.GELU(approx="fast")`를 exact GELU로 바꾸면 bf16 오차 범위에서 일치(프로젝터 타일 평균 -0.00171 대 -0.00172, 표준편차 0.1617 대 0.1611). 기본 fast GELU에서는 mlx-vlm 쪽이 더 벗어남.
- 특징 패킹과 병합: mlx-vlm 0.6.17의 `granite_vision` 모델은 LLaVA-Next 패킹을 구현하지 않습니다. `[2, 729, D]`로 브로드캐스트한 `image_newline`을 축 0으로 이어 붙이고 729행 블록 4개를 1485개 플레이스홀더와 zip하므로 언어 모델은 약 2976개 위치를 봅니다. mlxcel은 HF `LlavaNext.pack_image_features`(베이스 타일, permute, unpad, 줄바꿈 열, flatten)를 따르며, 두 프로세서가 만드는 1485 토큰 수와 일치합니다. 따라서 이 단계 이후의 생성 결과는 비교 대상이 아니고, 첫 토큰 logprob 차이도 동등성 지표가 아닙니다.

## 검증

- 재조정 호출을 끈 로더로 `tokenizer::added_tokens::tests::config_only_added_token_encodes_to_its_declared_id`와 `multimodal::granite_vision_prompt_parity_tests`를 실행하면 둘 다 실패했고(`[0, 2, 3, 4, 5, 6, 1]` 대 `[8]`, 패리티 테스트는 splice 폴백 단언에서 실패), 호출을 켜면 통과합니다. `tokenizer::`, `granite_vision` 선택자: 123개 통과.
- GB10에서 CLI와 `mlxcel-server`, `--temp 0`: "What is in this image? Describe it briefly."는 "In this image we can see a red color.", "Describe this image."는 "The image provided is a solid, uniform orange color. ...", "What do you see?"는 "orange", 색상 질문은 "Orange". 거절 없음. 서버 `prompt_tokens` 1545 / 1538 / 1539 / 1545로 mlx-vlm과 일치.

## 여기서 다루지 않은 발견 사항

- 이 체크포인트에서 텍스트 전용 요청은 사용자 메시지를 잃습니다(CLI와 서버 모두 `<|user|>\n<|assistant|>` 렌더링). 타입형 `chat_template.jinja`가 `type == 'text'` 항목만 고르므로 문자열(또는 평탄화된) content는 아무것도 렌더링되지 않습니다. mlx-vlm은 텍스트를 타입형 리스트로 감쌉니다. 별도 결함입니다.
- Granite Vision의 `vision_config`에는 `hidden_act`가 없습니다. HF SigLIP 기본값은 `gelu_pytorch_tanh`이고 mlxcel의 키 누락 기본값은 exact GELU입니다. 수치 차이는 작고(위 참조) 이 기본값은 저장소 전체 정책입니다.
