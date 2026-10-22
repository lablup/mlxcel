# PR #2551: SAM 3 prompt 기반 segmentation과 tracking 추가

**작성일**: 2026-10-23
**리뷰어**: -
**상태**: 부분 완료 (runtime parity 완료; 중앙 통합과 CI 대기 중)
**언어**: Rust, C++, Python, Markdown
**위험도**: 높음

## 요약

PR #2551은 `mlxcel predict`에 native SAM 3 image detection, prompt 기반 instance segmentation, memory 기반 video tracking을 추가한다. 공용 dense-prediction 계약에 반복 가능한 text·pixel-box prompt, source-pixel instance, source resolution u8 mask, 안정적인 PNG 이름, video object identity를 추가하고, `sam3`와 `sam3_video` checkpoint 실행에 필요한 ViT-L, CLIP text, DETR, geometry, segmentation, SAM mask decoder, tracker component를 port한다.

고정된 gfx1151 ROCm stack에서 BF16 reference parity와 실제 affine 4-bit inference의 runtime 검증을 완료했다. 최종 BF16 실행은 24개 CLIP text layer 전체에서 exact-native reference와 error 0을 달성했고, image와 box-prompt mask IoU는 최소 0.9989319였으며, 고정된 3-frame tracker 동작을 재현했다. End-to-end CLI artifact 검증, 중앙 동기화 후 architecture/redirect 검사, 새 PR CI 실행은 대기 중이다.

## 1. 문제 정의

mlxcel의 dense-prediction 기반은 map과 keypoint를 serialize할 수 있었지만, 선택적 mask가 있는 가변 개수의 scored object를 위한 model-neutral 표현이 없었다. Prompt flag와 family capability 선언도 없어서 prompt 기반 open-vocabulary segmenter가 text prompt, pixel box, instance별 PNG, 안정적인 object ID를 기존 CLI로 제공할 수 없었다.

SAM 3은 image detector와 segmenter를 stateful memory tracker와 결합한다. 충실한 port에는 model loader뿐 아니라 source-pixel geometry, task별 threshold, prompt semantics, source resolution mask 복원, video frame/object identity를 보존하면서 BF16과 quantized MLX checkpoint를 모두 불러오는 경로가 필요하다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 공용 instance 계약 | `Instance { bbox, score, label, mask_index, object_id, frame }`와 `DenseOutput.instances`를 추가했다. Bounding box는 source image pixel 좌표를 쓰며 instance별 mask row는 선택 사항이다. |
| 출력 artifact | `instances.json`, u8 `masks` tensor, `masks/000.png` 형태의 image 이름, `masks/f000012_o003.png` 형태의 video 이름, `boxes`/`scores` safetensor를 추가했다. Writer는 누락된 box/score tensor를 생성하고 제공된 tensor가 instance 값과 일치하는지 검증한다. |
| Prompt surface | 반복 가능한 `--prompt`와 `--boxes` parsing, finite하고 순서가 올바른 pixel box 검증, family별 prompt capability 검사, 구조화된 `prompts`/`boxes` predictor option을 추가했다. |
| Task 선택 | Input을 고려하는 predictor default를 추가했다. SAM 3 image는 `segment`를 기본으로 쓰고 video는 명시적인 `track`을 요구한다. |
| Native model | ViT-L vision encoder와 FPN, 24-layer CLIP text encoder, DETR encoder/decoder, geometry prompt, segmentation head, SAM prompt/mask decoder, memory 기반 tracker를 port했다. |
| Checkpoint loading | Idempotent convolution layout sanitize와 dense·affine-quantized weight용 `UnifiedLinear` 지원을 추가했다. |
| Registry와 문서 | `sam3`와 `sam3_video`를 등록하고 `masks` output kind를 추가했으며 prompt, output, tracking 동작을 문서화했다. |
| Acceptance fixture | Two-cat image, box prompt, 3-frame DAVIS clip, exact-native text feature, sampled processor pixel, JSON box/score, binary mask PNG, hash와 provenance를 고정했다. |

## 3. 기술적 선택과 그 이유

### 3.1 Model-neutral instance와 mask surface

Instance 수와 mask row 수는 서로 독립적이다. Instance는 mask가 없을 수 있으며, 생성된 각 mask row는 `mask_index`로 정확히 한 번 참조되어야 한다. 이 구조는 SAM 전용 file format을 만들지 않고 detection-only output과 후속 segmenter를 지원한다.

`result.safetensors`는 source-pixel `boxes`와 `scores`, shape `(M, H, W)`와 0 또는 1 값을 갖는 u8 `masks`를 저장한다. `instances.json`은 label과 선택적 video identity field를 추가하고, PNG는 mask 값을 0 또는 255로 변환한다. 안정적인 이름을 사용하므로 downstream consumer는 directory 순서에 의존하지 않고 JSON, tensor row, PNG를 연결할 수 있다.

Prompt 지원 여부는 registry metadata로 명시한다. 이 capability를 선언하지 않은 family에 `--prompt`나 `--boxes`가 들어오면 model을 load하기 전에 CLI가 거부한다.

### 3.2 Native image와 video 실행

Image `detect`와 `segment`는 prompt 사이에서 cached vision backbone을 공유하고, query logit과 presence logit을 결합하며, source coordinate에서 box를 clip하고, label 전체에 NMS를 적용한다. Video `track`은 threshold를 통과한 각 text detection 또는 pixel box에서 object를 초기화한 뒤 안정적인 `object_id`를 유지하며 tracker memory를 전파한다.

Image threshold 기본값은 0.3이고 NMS IoU 기본값은 0.5다. Video의 0.15 threshold는 text-prompt object 초기화에만 적용된다. 이후 tracker score는 upstream raw value를 유지하며 clamp하거나 probability로 재해석하지 않는다.

### 3.3 Linear layer와 GELU를 위한 범위가 제한된 BF16 연산

Upstream MLX는 bias가 있는 dense projection을 fused `addmm` 경로로 계산한다. BF16 matrix multiplication과 bias addition을 분리하면 rounding 결과가 달라진다. 따라서 SAM 3은 `UnifiedLinear`를 local wrapper로 감싸 일반적인 biased dense projection에는 fused 연산을 사용하고, quantized·biasless·runtime LoRA 활성 경로는 공용 구현을 그대로 사용한다. 이 방식은 공용 layer 계약과 다른 model family의 동작을 유지한다.

고정된 mlx-vlm runtime은 특정 BF16 연산 순서로 GELU를 `x * (1 + erf(x / sqrt(2))) / 2`로 계산한다. SAM 3은 text, vision, encoder, mask decoder, tracker 경로에 local activation helper를 사용한다. 이 범위가 제한된 구현은 다른 model의 activation 동작을 유지하면서 같은 native MLX backend에 대한 SAM 3 text pipeline을 bit-exact하게 만든다.

실제 layer-zero GELU oracle이 통과했고, 최종 text diagnostic은 embedding, 기록된 모든 layer-zero substage, layer 0부터 23까지의 residual output, final normalization, detector projection, 32개 token row 전체에서 maximum error와 mean error 0을 보고했다.

### 3.4 재현 가능한 exact-native fixture

Reference set은 source asset, 변환 규칙, runtime, SHA-256 값을 기록한다. 큰 numeric reference는 JSON metadata, one-bit PNG mask, sampled processor pixel, stage diagnostic에 필요한 text safetensor로 줄였다. Exact-native reference는 Rust acceptance test와 동일한 pinned MLX ROCm backend를 사용하므로 backend 차이가 parity 결과에 섞이지 않는다.

## 4. Provenance

| 구성 요소 | 고정 값 |
|---|---|
| Upstream 구현 | `Blaizzy/mlx-vlm` commit `abd3b2d7b728cb5372d4e092260da325e5ded209` |
| BF16 checkpoint | `mlx-community/sam3-bf16` revision `dfe573c3171dbcfda8399c650d9135afa7e94592` |
| Affine 4-bit checkpoint | `mlx-community/sam3-4bit` revision `53c009b4ef4cfa5b1d9fa4549e7b153e881b5636` |
| Native MLX pin | `bff10220c742907f0c68e55ff174aaea56f74e3d` |
| Reference runtime | mlxcel exact native archive의 MLX 0.32.5, BF16 weight를 cast하지 않음 |
| 검증 장치 | ROCm gfx1151, AMD Radeon 8060S Graphics, 20 compute unit, warp size 32 |

COCO two-cat image는 lossless 방식으로 다시 압축했다. 3-frame video는 고정된 MP4 decode 경로를 유지한다. `tests/fixtures/sam3/provenance.json`은 upstream asset hash, commit된 fixture hash, preprocessing SHA-256, reference 변환 규칙을 기록한다.

이 ROCm 장치에서는 `MLXCEL_KEEP_BF16` 설정 여부와 관계없이 공용 dtype policy가 BF16을 유지했다. 이 보고서는 강제로 F16으로 변환한 parity 실행을 주장하지 않는다.

## 5. Runtime 검증

### 5.1 BF16 text, image, box parity

| Case | 결과 |
|---|---|
| 24-layer CLIP text 경로 | 범위가 제한된 GELU 수정 후 기록된 모든 stage에서 maximum error 0, mean error 0 |
| Processor | 전체 resized RGB8 SHA-256이 Pillow와 일치하며 sampled normalized pixel이 maximum error `1e-6`, mean error `1e-7` 한계를 통과 |
| Text-only cat segmentation | Detection 2개, 최대 box coordinate error 0.0021 px 미만, 최대 score error `6.7e-6` 미만, mask IoU 1.0과 0.99998 |
| Text와 box prompt | Detection 2개, 최대 box coordinate error 0.0019 px 미만, 최대 score error `1.2e-5` 미만, mask IoU 0.9999825와 0.9989319 |

최종 통합 BF16 실행은 ignored real-checkpoint test 4개를 27.74초에 완료했고 4 passed, 0 failed를 기록했다.

### 5.2 Video parity와 upstream limitation

고정된 3-frame reference는 안정적인 `object_id = 1`을 전파한다. Raw tracker score는 `0.9603320`, `-3.0101275`, `-3.0060225`다. Frame 0은 source-mask IoU threshold 0.95를 통과한다. 고정된 upstream model은 object를 유지하면서 frame 1과 2에서 native mask와 source resolution mask를 비우며, Rust test는 해당 mask가 정확히 비어 있는지 확인하고 음수 raw score를 보존한다.

이 fixture는 tracker memory 실행, identity 안정성, source resolution output, 정확한 empty propagation을 검증한다. Frame 0 이후에는 의미 있는 non-empty-mask IoU evidence를 제공하지 않는다.

두 번째 실제 upstream 실행은 cat image 3장을 반복하고 box prompt를 사용했다. 이 경우에도 object ID 1은 유지되었지만 frame 1과 2의 mask는 비어 있었다. 따라서 이 제한은 움직이는 video와 정지된 반복 image 입력 모두에서 재현되었다.

### 5.3 Affine 4-bit inference

실제 `mlx-community/sam3-4bit` checkpoint가 `UnifiedLinear`를 통해 load되고 prompt 기반 image segmentation을 완료한다. Smoke test는 비어 있지 않은 instance, finite score와 box, 반환된 instance당 하나의 mask row, 480x640 source resolution mask, 비어 있지 않은 foreground를 검증한다. gfx1151에서 6.50초에 통과했다.

### 5.4 검증 log

| Log | 증거 |
|---|---|
| `issue-2429-gelu-build.log` | 최종 test-fast unit-test executable build 성공 |
| `issue-2429-gelu-oracle.log` | 실제 BF16 GELU oracle 통과 |
| `issue-2429-gelu-text-bf16.log` | 24-layer text와 projection의 zero-error parity |
| `issue-2429-gelu-full-bf16.log` | 최종 text, image, box-prompt, video parity, 4 passed |
| `issue-2429-4bit-inference-smoke.log` | 실제 affine 4-bit segmentation inference 통과 |
| `issue-2429-focused-tests.log` | SAM 3 option, NMS, clipping, tokenizer, sanitizer, threshold, preprocessing test 통과 |

이전 diagnostic log는 문제 범위를 좁힌 과정을 기록하며 acceptance evidence로 사용하지 않는다. 위의 최종 log가 이를 대체한다.

### 5.5 대기 중인 integration evidence

- **중앙 동기화 후 대기:** image와 video에 대한 end-to-end `mlxcel predict` CLI artifact 실행.
- **중앙 동기화 후 대기:** `mlxcel arch --json`에서 SAM 3을 `predict` runtime과 `masks` output으로 표시하는지 확인.
- **중앙 동기화 후 대기:** `mlxcel generate`가 두 SAM 3 model type을 `mlxcel predict`로 안내하는지 확인.
- **중앙 동기화 후 대기:** format, clippy, repository asset/provenance gate, 필수 platform 검사를 포함한 PR #2551의 새 CI 결과.

## 6. 위험과 호환성

이 port는 server endpoint를 추가하지 않고 token-generation 또는 closed-vocabulary detection output 계약을 변경하지 않는다. 새 CLI flag는 prompt 지원을 선언하지 않은 dense family에서 거부된다. Point-prompt CLI input, webcam streaming, overlay rendering, background replacement는 이 issue의 범위 밖이다.

주요 numerical risk는 큰 model 전체에서 backend에 민감한 BF16 연산 순서다. Exact-native fixture와 stage diagnostic이 gfx1151에서 이 위험을 제한한다. 이 Linux host에서는 full-checkpoint Metal runtime parity를 실행하지 않았다. Platform CI는 build와 test compatibility를 확인할 수 있지만 checkpoint acceptance test를 실행하지 않는 한 full model parity 증거가 아니다. Video fixture는 upstream tracker limitation도 보여준다. Identity가 음수 raw score와 빈 mask를 가진 상태로 유지될 수 있다.

## 7. 후속 조치

- 중앙 동기화 후 대기 중인 CLI, architecture catalog, generate redirect 검사를 완료한다.
- 새 PR CI를 실행하고 최종 integration evidence로 이 보고서를 갱신한다.
- Issue #2430의 SAM 3.1에서 공용 prompt, instance, output, model component를 재사용한다.
- 향후 작업에서 고정된 empty propagation 재현보다 여러 frame의 mask 품질 측정이 필요하면 이후 mask가 비어 있지 않은 tracking acceptance clip을 사용한다.

## 참고 자료

- PR #2551: `feat: add SAM 3 promptable segmentation and tracking`
- Issue #2429: SAM 3 promptable segmentation and video tracking
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: 공용 CLI와 output 계약
- `tests/fixtures/sam3/provenance.json`: source와 fixture provenance
