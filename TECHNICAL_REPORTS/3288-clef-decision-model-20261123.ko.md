# PR #3288: Clef 및 Clef-Flash (clef) decision 모델 추가

**작성일**: 2026-11-23
**상태**: PR 열림 (draft, #2365 해결, #2375의 일부). 단위, 시퀀스 fixture, 실제 체크포인트 parity 테스트 통과. `models/clef-flash-mlx-8bit`에서 CLI와 서버 확인, 보안 수정 후 재실행
**언어**: Rust, Python (fixture 생성기)
**위험도**: 낮음 (새 decision 계열. Qwen3.5 hidden-state 진입점 하나와 비디오 헬퍼 하나를 추가만 했고 생성 경로는 변경 없음)

---

## 요약

Clef와 Clef-Flash (`model_type: clef`, `nativ-community/clef-*`)는 Qwen3.5 vision-language backbone 위에 joint schema head를 올린 모델로, 텍스트, 이미지, 비디오를 읽고 한 번의 forward로 모든 질문의 모든 옵션을 점수화한다. 이 PR은 mlx-vlm의 `models/clef`를 #2343 decision API 뒤에 포팅하여 `mlxcel decide -m <path>` (`--image` / `--video` 포함)와 `mlxcel-server --decision-model <path>` (`POST /v1/decisions`)에서 사용할 수 있게 한다. Clef-Flash 8-bit에서 token id와 span은 upstream과 정확히 일치하고, 확률은 이미지 요청에서 mlx-vlm 대비 1.1e-3, 텍스트 요청에서 2.3e-3 (score 값 2.8e-3) 이내이며, 양쪽을 f32로 돌리면 소수 넷째 자리까지 같다. 리뷰에서 정확성 결함은 발견되지 않았다. 이후 보안 검토에서 토큰 상한 이전에 수행되는 작업을 제한하고, 리터럴 pad 토큰을 거부하며, 비디오 오류에서 서버 경로를 제거했다 (2.5절). 수정 후 실제 체크포인트 출력을 다시 돌렸고 결과는 그대로였다.

## 1. 문제 정의

### 1.1 배경

#2343이 decision 레이어를, #3282 / #3284가 d1-3B와 Decider-2b 계열을 추가했다. Clef는 여전히 `Unsupported model type: clef`로 거부되고 있었다 (#2365).

### 1.2 기존 문제점

- **로더 arm 없음**: `model_type: clef`를 인식하는 `DecisionKind`가 없었다.
- **병합된 embedding 위의 post-norm hidden 없음**: Decider-2b의 `hidden_from_embeddings`는 token id만 받았다. Clef는 미디어가 병합된 `inputs_embeds`와 MRoPE position이 필요하다.
- **임의 인덱스 비디오 샘플링 없음**: Clef는 모든 원본 프레임에 대해 `linspace(0, total - 1, n)`으로 샘플링하는데, 기존 `FrameSamplingPolicy`로는 표현할 수 없다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 같은 인스턴스의 다음 요청으로 MRoPE 상태 누출 | 중간 | 낮음 (매 forward 전후에 비움) |
| Qwen3.5 hook으로 인한 생성 출력 변화 | 높음 | 낮음 (함수 추가만, 공유 코드 수정 없음) |
| 요청 하나가 단일 decision worker에서 비디오를 무제한 디코딩 | 중간 | 낮음 (원본 해상도로 계산하는 4 GiB 디코딩 예산, 시퀀스가 상한 안에 들어갈 때만 비디오를 하나씩 디코딩, 768 프레임 상한, 서버 미디어 개수 제한) |
| state의 리터럴 pad 토큰이 MRoPE position 빌더를 panic시킴 (500) | 중간 | 낮음 (미디어가 있으면 400으로 거부) |

## 2. 기술적 검토 사항

### 2.1 Head 충실도

`JointSchemaHead`, `EvidenceRoutingLayer`, `FieldDecoderLayer`의 모든 연산을 `clef.py`와 MLX `nn` 모듈에 대조했다: `fast::layer_norm`을 통한 LayerNorm eps 1e-5. 네 projection 모두 bias가 있는 MHA, `reshape(B, L, heads, -1)` head 분할과 `sqrt(1 / head_dim)` scale. `addmm(bias, x, W.T)`로서의 `nn.Linear`. activation dtype의 erf 형태 GELU. f32 matmul 후 hidden dtype으로 되돌리는 span 평균. option summary의 precise softmax. routing layer마다 정규화되지만 field decoder에는 원본으로 들어가는 memory. 두 `exp(min(scale, ln 100))` clamp가 있는 prior와 residual 수식. type id `noul 0, choice 1, score 2`. 역양자화한 `lm_head` 행을 f32로 평균하고 hidden dtype으로 변환한 lexical 벡터.

### 2.2 시퀀스와 옵션

`bool`은 `true`, `false` 순서이며 기본 설명을 criteria가 덮어쓴다 (null 설명은 upstream처럼 생략). `choice` 라벨은 schema에서는 code point 순으로 정렬하고, 응답에서는 요청 순서로 보고하며 argmax는 Python처럼 첫 최대값을 고른다. 압축 정렬 JSON은 `json.dumps(..., ensure_ascii=False, separators=(",", ":"), sort_keys=True)`와 같다. 16384 상한은 고정 부분이 넘치면 upstream 메시지로 거부하고, 그렇지 않으면 state를 자른다. 미디어 블록 (이미지 run, 바깥 vision token 안의 timestamp가 붙은 비디오 group)과 비디오 샘플링은 커밋된 fixture로 고정되어 있다.

### 2.3 Backbone hook과 상태 격리

`Qwen35Model::forward_final_hidden_with_embeddings`는 fallback MRoPE 슬롯에서 cache offset 0의 prefill과 똑같이 position을 결정하고, 텍스트 전용 입력에는 세 축 모두 `0..L`을 쓰며 (upstream의 `get_rope_index`), 새 cache 위에서 공유 `hidden_from_embeddings`를 돌린 뒤 반환 전에 fallback 슬롯을 비운다. `predict`도 진입 시 비운다. `ClefModel`은 생성 모델과 별개인 자체 `Qwen35VLModel`을 가지고 한 스레드에서만 구동되므로 생성 요청으로 상태가 넘어가지 않는다. `forward_internal`, `forward_internal_at`, token 경로는 변경되지 않았다.

### 2.4 Parity 차이

텍스트 요청은 이슈의 2e-3 기준을 3e-4 넘는다. head에서 cast 위치 차이는 발견되지 않았고, 개발자의 f32 비교 (양쪽이 넷째 자리까지 동일)가 구조적 포팅 오류를 배제한다. mlx-vlm 자체의 bf16과 f32 실행이 같은 요청에서 7.9e-3 차이 나므로, 남은 차이는 생성과 공유하는 backbone kernel의 bf16 반올림 순서다. 테스트는 텍스트 3e-3, 이미지 2e-3을 기준으로 커밋했다.

### 2.5 보안 수정 (커밋 `6a0cb27a`)

수정 전에는 Clef가 16384 토큰 상한을 확인하기 전에 요청된 모든 비디오를 디코딩했고, 공유 4 GiB 예산도 리사이즈된 프레임 기준으로 계산했다. 그래서 비디오가 많은 요청은 단일 decision worker에서 비디오당 최대 4 GiB를 디코딩한 뒤에야 거부될 수 있었다. 이제 비디오는 하나씩, 지금까지의 시퀀스 (텍스트, 이미지, 앞선 비디오)가 상한 안에 들어갈 때만 디코딩하며, 예산은 원본 해상도 바이트로 계산한다. schema도 상한까지만 encode한다. 일찍 거부하면 `schema requires at least N tokens`라고 하고, 전체 개수를 센 경우에는 upstream의 `schema requires N tokens before state; maximum is 16384`를 유지한다.

state나 questions에 들어간 리터럴 `<|image_pad|>` 또는 `<|video_pad|>`가 placeholder로 파싱되어, 미디어가 있을 때 MRoPE position 빌더가 추가된 run에서 panic했다. 이제 이런 요청은 400이다. 비디오 오류는 더 이상 허용 목록 파일의 서버 경로나 다운로드 임시 파일 경로를 노출하지 않는다. `decision::clef_limits_tests`가 schema encode 상한, 상한을 넘는 비디오의 디코딩 생략, pad 토큰 거부, 경로 제거를 검증한다.

## 3. 기술적 선택과 그 이유

### 3.1 layer loop 복사 대신 `hidden_from_embeddings` 확장

**컨텍스트:** Decider-2b가 이 용도로 `position_ids` 인자를 추가해 두었다. **선택:** position을 결정하고 공유 loop를 호출하는 얇은 진입점을 두어 layer loop가 한 곳에만 있게 했다. **트레이드오프:** fallback 슬롯을 비우는 계약을 진입점이 책임진다.

### 3.2 텍스트 전용 입력에도 명시적 MRoPE position

cache offset rope 경로는 수학적으로 같지만 bf16에서 반올림이 다르다. upstream이 `get_rope_index` position을 layer에 넘기므로 포팅도 그렇게 한다.

### 3.3 `video.rs`의 sampler callback

`load_video_source_frames_at`은 probe한 프레임 수에 대한 sampler를 받고, 기존 probe 상한과 디코딩 바이트 예산을 그대로 유지한다. 호출자 하나를 위해 policy variant를 추가하지 않았다.

## 4. 구현 상세

- `src/decision/clef.rs`: 로딩, 미디어 준비, lexical 벡터, readout.
- `src/decision/clef_head.rs`, `clef_layers.rs`: head 포팅.
- `src/decision/clef_sequence.rs`, `clef_media.rs`: 시퀀스 구성과 비디오 geometry (상한까지만 하는 schema encode 포함).
- `src/decision/clef_limits_tests.rs`: 위 보안 테스트.
- `src/multimodal/video_tests.rs`: `load_video_source_frames_at` 테스트 (ffmpeg 필요).
- `src/decision/loader.rs`, `src/models/detection.rs`: `Clef` arm과 `generate` 거부.

## 5. 학습 포인트

### 5.1 포팅 오류와 연산 노이즈 구분

bf16 차이가 기준을 넘으면 먼저 양쪽을 f32로 돌린다. 거기서 넷째 자리까지 일치하고 upstream 자체의 bf16/f32 차이가 크다면, 포팅 오류가 아니라 반올림 순서 문제다.

### 5.2 span은 개별 encode에서 나온다

upstream은 tokenizer 호출 결과를 이어 붙이므로 span 위치가 조각 경계에 따라 정해진다. 포팅에서도 조각마다 따로 encode해야 span이 정확히 일치한다.

## 6. 추가 학습 리소스

### 관련 PR/이슈

- Issue #2365: Clef / Clef-Flash decision 모델.
- Issue #2375: decision 모델 epic.
- PR #3284: `hidden_from_embeddings`를 도입한 Decider-2b.
- PR #3282: LiquidAI d1-3B decisions.

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경된 파일 | 25 (Decider-2b 기준) |
| 추가된 줄 | +5500 |
| 삭제된 줄 | -13 |
| 추가된 테스트 | 20 (Clef 단위 13, 상한 4, parity 1, detection 1, 비디오 sampler 1) |

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|--------|
| `661634e7` | feat | add Clef and Clef-Flash decision models (clef) |
| `c15c8d68` | fix | mark the Clef fixture generator executable |
| `6a0cb27a` | fix | bound Clef work before the token cap and refuse pad tokens |
| (최종) | test/docs | 비디오 sampler 테스트, 보안 변경 문서화, 재검증 |

## 8. 후속 조치

### 검증됨 (보안 커밋 이후 최종 실행)

- 릴리스 빌드로 `models/clef-flash-mlx-8bit`에서 `mlxcel decide`를 실행했고, 출력은 보안 수정 전과 동일하다: 텍스트 요청 `billing` 0.9757, usage 135. 이미지 요청 `two` 0.9686 (remote true 0.9515), usage 521. 비디오 요청 pattern true 0.9586, kind `color bars test pattern` 0.9859, usage 578. 오디오는 거부되고 (`This model does not support audio input`), `mlxcel generate`는 `mlxcel decide`를 안내하며 거부되고, 이미지와 함께 `<|image_pad|>`가 든 state는 400 계열 메시지로 거부된다.
- `decision::` 95개 통과 (실제 체크포인트 parity 테스트와 `clef_limits_tests` 포함), `models::detection_tests` 53개, `coverage_tests` 18개, `multimodal::video` 86개 통과. `--include-ignored`로 `source_frames_at_decodes_the_sampler_picks_and_rejects_bad_ones`도 통과 (ffmpeg 필요).
- lib과 tests에 대한 clippy `-D warnings`, fmt, `scripts/ci` contract check 통과.
- `forward_final_hidden_with_embeddings`는 텍스트, 이미지, fixture 경로를 도는 실제 체크포인트 parity 테스트로 검증된다.

### 미검증 또는 남은 항목

- `multimodal::video::tests::bench_single_pass_768_frames` (ignored, ffmpeg)는 이 호스트에서 ffmpeg가 768개 항의 `select` 표현식을 거부해 (`Cannot allocate memory`) 실패한다. 이 PR이 추가한 테스트가 아니며 공유 single-pass 추출기의 문제이므로, 수백 프레임을 샘플링하는 긴 Clef 클립도 같은 문제를 만날 수 있다. 여기서는 고치지 않았다.

- NVFP4 / MXFP4 빌드는 실행하지 않았다. `clef-MLX-8bit`은 로드와 간단한 요청 하나만 확인했다.
- 비디오는 커밋된 parity 테스트가 없다 (수동 확인 7e-4).
- 양자화하지 않은 Clef 체크포인트 (공개된 것은 없음)는 Apple Silicon에서 backbone이 f16으로 로드되고 head는 bf16으로 남는다.
