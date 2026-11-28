# 기술 보고서: PR #2034 - Nemotron-Parse를 seq2seq 워커에 포팅

**날짜**: 2026-09-29

**상태**: 구현 완료, 실제 체크포인트로 검증 완료; 머지 대기.

**언어**: Rust (모델, 로더, CLI, 서버), Python (오라클과 픽스처 생성기, 배포 대상 아님)

**위험도**: 중간 (한 가지 변경이 모든 `tokenizer.json` 모델에 영향)

## 요약

Nemotron-Parse (`nvidia/NVIDIA-Nemotron-Parse-2.0`, `model_type: nemotron_parse`)는 페이지 정규화 좌표인 `<x_..><y_..>` 박스 토큰과 `<class_..>` 태그가 붙은 마크다운을 출력하는 문서 OCR / 레이아웃 모델입니다. 인코더-디코더 구조로, C-RADIOv2-H ViT-H/16 타워와 압축 넥이 흰색으로 패딩된 2048x1664 페이지를 3329개의 인코더 상태로 바꾸고, 토큰화된 태스크 프롬프트로 시드된 10층 pre-norm mBART 디코더가 이를 대상으로 greedy 디코딩합니다. mlxcel에는 이미 seq2seq 계열(Florence-2)이 하나 있고, 이번 포팅은 그 어텐션 서브레이어, self/cross 이중 KV 캐시, causal mask 헬퍼, CLI 조기 분기 패턴, 단일 스트림 서버 워커 패턴을 재사용합니다.

모델 밖으로 번진 발견이 두 가지 있습니다. 체크포인트의 `tokenizer.json`이 `padding: Fixed(9000)`을 직렬화하고 있고 `tokenizers` 크레이트는 이를 매 encode마다 적용하므로, mlxcel은 이제 `transformers`처럼 로드 시점에 직렬화된 padding과 truncation을 제거합니다. 또한 8비트 export에서 bf16 디코더 활성값이 근소한 동률 두 곳을 뒤집기 때문에 디코더 활성값을 f32로 실행합니다.

결과는 체크포인트 자체의 `transformers` 코드를 CPU fp32로 실행한 결과와, 모델이 직접 선택한 79개 greedy 토큰 전부가 일치합니다. 반복 패널티 1.0과 1.1 모두에서, Hub 체크포인트와 8비트 export 모두에서 확인했습니다.

## 1. 문제 정의

mlxcel에는 `nemotron_parse` 분기가 없었습니다. 이 모델은 디코더 전용 생성 루프에서 실행할 수 없고(디코더가 페이지별 인코더 패스에 대한 cross-attention K/V를 필요로 함), 두 부분은 트리에 대응물이 없었습니다. CPE 위치 그리드와 teacher CLS 토큰을 가진 C-RADIO 타워, 그리고 pre-norm이면서 위치 테이블이 없고 임베딩을 `sqrt(d_model)`로 스케일하며 단일 시작 토큰이 아닌 여러 토큰의 프롬프트로 시드되는 디코더입니다.

## 2. 변경 요약

- `src/models/nemotron_parse/`: `config.rs` (`encoder` / `decoder` 하위 설정; ViT 형상은 `args.model`에서, CLS와 register 개수와 summary 인덱스는 teacher 목록에서 유도), `checkpoint.rs` (Hub에서 MLX로의 키 정규화와 shape 기반 conv 변환), `encoder.rs` (patchify, CPE 위치 crop 또는 align-corners 리샘플, fused SDPA 커널을 쓰는 32개의 pre-norm 블록), `neck.rs` (1x1 conv를 Linear로, `(1, 4)` stride-4 conv를 reshape와 matmul 한 번으로, summary 행), `decoder.rs` (Florence-2 어텐션 위의 pre-norm mBART), `model.rs` (greedy 루프, 시드 검사, 반복 패널티), `processor.rs` (리사이즈, 흰색 패딩, CLIP 정규화, 시드 구성), `runtime.rs` (`LoadedModel` 단위).
- 통합: detection, `ModelType`, 레지스트리, `model_metadata`, 로더, `LoadedModel`, `mlxcel generate` 조기 분기, `mlxcel run` 거절, `mlxcel arch` 카탈로그 항목, 두 워커 선택 지점의 서버 워커 선택, 단일 스트림 큐 admission, warmup 생략, 내장 채팅 템플릿.
- `src/server/nemotron_parse_worker.rs`: 요청 경계 검증을 포함한 batch-1 루프.
- `src/tokenizer/mod.rs`: `tokenizer.json` 로드 경로의 `clear_serialized_padding_and_truncation`.
- 테스트: 모델 단위 테스트 22개, 워커 테스트 7개, detection, 채팅 템플릿, 토크나이저 테스트, 그리고 커밋된 픽스처 페이지와 생성 스크립트를 쓰는 `tests/nemotron_parse_real_model.rs`.
- `docs/supported-models.md`: 태스크 프롬프트 토큰을 포함한 계열 항목.

## 3. 기술적 결정

### 공유 트레이트 대신 Florence-2 seq2seq 부품 재사용

`Florence2Attention`에는 mBART 디코더가 필요로 하는 bias 포함 `q/k/v/out` 프로젝션과 일회성 cross K/V 캐시가 이미 있고, `Florence2LayerCache`, `additive_causal_mask`, `layer_norm`(eps 1e-5, 디코더 값)은 `pub(crate)`입니다. 서버 워커는 `florence2_worker.rs`를 일반화하지 않고 형제 파일로 두었습니다. 두 워커는 프롬프트 파싱, 검증, 출력 형태가 다르기 때문에(Florence-2는 태스크 마커를 파싱해 구조화된 좌표를 반환하고, Nemotron-Parse는 프롬프트를 그대로 넘겨 텍스트를 반환) 공유 트레이트는 공통점이 적은 두 호출 지점을 억지로 묶는 추상화가 됩니다.

### 시드는 이슈의 규칙이 아니라 `transformers` generate를 따름

이슈는 special token을 포함해 토큰화한 뒤 래퍼를 제거하는 방식을 제안했습니다. 레퍼런스 프로세서 호출은 `add_special_tokens=False`를 쓰고, `generate`는 첫 id가 다르면 `decoder_start_token_id`를 앞에 붙입니다. 오라클로 확인했습니다: `<predict_bbox><predict_classes><output_markdown>`은 `[2, 50004, 50008, 50001]`이 됩니다. 포팅은 이를 구현했고, 기본 프롬프트는 어느 쪽이든 `[2, 0, 50004, 50008, 50001, 50010]`이 됩니다.

### 반복 패널티는 시드까지 포함

`RepetitionPenaltyLogitsProcessor`는 시드의 `</s>`와 제어 토큰을 포함한 디코더 시퀀스 전체를 봅니다. 포팅도 같은 집합에 패널티를 적용하므로 1.1 패널티가 레퍼런스와 토큰 단위로 일치합니다.

### 넥의 conv는 matmul

`(1, 4)` 커널에 stride `(1, 4)`는 서로 겹치지 않는 4열 묶음을 읽으므로, conv는 정확히 `reshape [B, hp, wp/4, 4*C] @ W_flat`이며 커널은 MLX `[out, 1, kw, in]` 레이아웃에서 `(kw, c_in)` 순서로 펼칩니다. 직접 컨볼루션 단위 테스트가 인덱스 순서를 고정하고, 로드 시 전치되는 torch 레이아웃의 Hub 체크포인트가 레퍼런스와 일치하는 것이 end-to-end 검증입니다.

### 디코더 활성값은 f32

첫 8비트 실행은 생성 2번째 스텝(`**Hello` 대신 `# Hello`)과 60번째 스텝(좌표 bin 하나)에서 갈라졌습니다. 8비트 export의 가중치를 fp32로 역양자화해 레퍼런스를 돌리면 비양자화 시퀀스와 같았으므로 원인은 양자화가 아니었습니다. 디코더 활성값이나 타워 활성값 중 하나를 f32로 넓히면 일치가 회복되었고, 디코더는 짧은 시퀀스 위의 폭 1024, 10층이므로 저렴한 쪽입니다. 양자화된 프로젝션은 bf16 scale을 그대로 유지합니다. 해당 스텝에서 레퍼런스의 top-2 마진은 0.04에서 0.07입니다.

### 직렬화된 padding과 truncation은 모든 모델에서 제거

`transformers` fast 토크나이저는 일반 encode에서 `tokenizer.json`의 padding이나 truncation을 적용하지 않지만, `tokenizers` 크레이트는 매 encode마다 적용합니다. 이 계열에만 적용했다면 같은 체크포인트에 대해 `generate`의 일반 프롬프트 토큰화와 서버 디스패치 스레드가 여전히 9000개의 id를 만들었을 것이고, `transformers`와 같게 동작하는 것이 모든 모델에 대해 올바른 동작입니다. 토크나이저, 로딩, 채팅 템플릿 테스트 모듈은 이 변경 후에도 통과합니다.

### 내장 템플릿이 체크포인트의 템플릿을 대체

Hub 체크포인트는 `{% for message in messages %}{{ message['content'] }}{% endfor %}`를 제공합니다. 이 계열이 요구하는 이미지 요청에서 `content`는 타입이 지정된 리스트이므로, 이 템플릿은 base64 페이지를 포함한 리스트 자체를 디코더 시드로 렌더링합니다. mlxcel의 규칙은 내장 템플릿이 제공된 템플릿을 가리지 않는 것인데, 이 계열이 유일한 예외이며 텍스트 전용 내장 템플릿은 문자열 content에 대해서는 제공된 템플릿과 동일합니다.

## 4. 검증

- 오라클: `nvidia/C-RADIOv2-H` 타워를 포함한 체크포인트 자체의 `transformers` 코드, CPU fp32 (torch 2.14.0, transformers 5.17.0), 기본 프롬프트, greedy, 새 토큰 80개, 리샘플이 필요 없는 1240x1754 페이지.
- Hub 체크포인트(f32)와 8비트 export: 모델이 직접 고른 79개 토큰이 패널티 1.0과 1.1에서 모두 일치. 레퍼런스의 80번째 토큰은 예산 도달 시 `forced_eos_token_id`로 강제된 `</s>`이며(로짓 마진이 무한대), mlxcel은 이를 흉내 내지 않습니다.
- 4비트 export: 페이지를 올바르게 읽음(같은 박스의 `**Hello Nemotron**`).
- 서버: Hub 체크포인트에서 `/v1/chat/completions`가 레퍼런스 텍스트를 반환하고, 빈 프롬프트, 이미지 두 장, 이미지 없음, 제어 문자, 스트리밍이 명세대로 동작.
- 게이트: fmt, `clippy --release -p mlxcel --lib --tests -D warnings`, 세 개의 계약 테스트, PR에 나열한 단위 테스트 모듈.

## 5. 한계와 후속 과제

- 1664x2048보다 큰 페이지는 `image`의 bilinear 필터로 축소되며 PIL의 antialias BILINEAR 탭과 약간 다르므로, 이런 페이지의 출력은 레퍼런스와 가깝지만 비트 단위로 같지는 않습니다.
- v1.x의 untied `lm_head` 경로는 구현되어 합성 모델로 단위 테스트했지만 실제 v1.x 체크포인트로는 실행하지 않았습니다.
- 모델 카드의 표 삽입 및 반복 중단 logits processor, 후처리 스크립트, 배치 크기 2 이상은 범위 밖입니다.
- 9000 위치 한계에서 멈춘 디코딩은 `finish_reason: "stop"`을 보고하며, `max_tokens` 소진만 `"length"`를 보고합니다.
