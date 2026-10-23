# PR #2575: SAM 3.1 multiplex tracking 추가

**작성일**: 2026-10-23
**리뷰어**: -
**상태**: 부분 완료 (라이브러리 구현, ROCm parity, CLI 검증, clippy, repository contract 완료; 중앙 동기화, 최종 CI 대기)
**언어**: Rust, Python, Markdown
**위험도**: 높음

## 요약

PR #2575는 mlxcel에 native SAM 3.1 open-vocabulary detection, prompt 기반 image segmentation, Object Multiplex video tracking을 추가한다. SAM 3 ViT backbone, CLIP text path, DETR detector, geometry와 mask component, 공용 FPN 구현, prompt 처리, instance-mask output contract를 재사용하면서 SAM 3.1 three-branch neck, multiplex mask decoder, decoupled memory attention, 제한된 tracker state를 추가한다.

고정된 Linux ROCm gfx1151 환경에서 라이브러리와 실제 checkpoint 검증을 완료했다. 배포된 `mlx-community/sam3.1-bf16` checkpoint에는 BF16 tensor가 아니라 F32 tensor 1,961개가 들어 있으며 mlxcel은 이 F32 parameter와 F32 input을 유지한다. 실제 3-frame traffic test는 자동차 9대를 비어 있지 않은 mask 27개로 tracking하고, 안정적인 ID를 유지하며, 모든 object와 frame에서 mask IoU 0.95 이상을 달성하고, object 수와 무관하게 `track_step`을 정확히 두 번 호출한다. Current-head CLI image/video artifact, architecture dispatch, generation redirect, warning을 error로 처리한 clippy, repository contract 8개, formatting, diff check가 passed했다. VDA 및 MoGe 작업과의 동기화와 최종 PR CI는 아직 대기 중이다.

## 1. 문제 정의

SAM 3.1은 SAM 3의 detector와 image-segmentation stack을 유지하지만 tracking path를 Object Multiplex로 교체한다. 충실한 port는 큰 detector 구현을 복사하지 않고 공유해야 하며, detection, interaction, propagation용 FPN branch 세 개를 로드하고, 임의 개수의 object를 16-slot tracker bucket에 매핑하며, 공용 source-pixel instance 및 mask output contract를 유지해야 한다.

공개 checkpoint에는 precision policy 함정도 있다. Repository 이름에는 `bf16`이 들어 있지만 고정된 safetensor에는 F32 parameter만 들어 있다. Repository label을 tensor dtype으로 취급하면 inference가 달라진다. Video tracking에는 두 번째 위험이 있다. Source-resolution mask, 보존된 spatial memory, object pointer, conditioning concatenation, attention projection이 output serialization 전에 큰 working set을 만들 수 있기 때문이다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 공용 SAM 3 재사용 | 재사용 가능한 detector, prompt, mask-decoder, memory, FPN hook을 공개했다. SAM 3.1 model은 이를 복제하지 않고 기존 ViT, CLIP, DETR, geometry, segmentation, preprocessing 구현을 조합한다. |
| Three-branch neck | SAM 3.1 scale에 detection, interactive, propagation FPN head를 추가하고 FPN resize와 projection 계산은 공용 SAM 3 module에 유지했다. |
| Object Multiplex | Gather 기반 multiplex 및 demultiplex state, valid-slot padding, 16-object bucket size, multiplex mask decoding, decoupled memory attention, temporal position encoding, 제한된 conditioning history를 추가했다. |
| Video tracking | Frame 0에서 object를 detect하고, 선택적 pixel box를 region filter로 적용하고, multiplex session 하나를 초기화하며, 이후 각 frame에서 모든 object에 대해 `track_step` 한 번을 수행한다. |
| Precision 및 loading | Checkpoint의 실제 F32 tensor와 F32 model input을 유지하고, 멱등적인 SAM 3 checkpoint sanitizer를 재사용하며, 지원되는 quantization metadata 위치 두 곳을 모두 검증한다. |
| 안전성과 resource | 잘못된 shape-critical configuration과 non-finite raw tensor를 거부하고, detector/tracker tensor shape을 검사하고, object 및 prompt expansion을 제한하며, source-mask resize 전에 checked retained-plus-transient tracker memory accounting을 적용한다. |
| Registry 및 CLI 통합 | `sam3.1`과 `sam3.1_video`를 등록하고, `detect`, `segment`, `track`을 표시하며, target FPS가 없는 명시적 tracking에서 모든 video frame을 유지하고, 공용 prompt/output surface를 문서화한다. |
| 재현 가능한 근거 | 고정된 two-cat 및 three-frame traffic reference, binary mask PNG 27개, 이식 가능한 reference script, source와 output hash, 정확한 runtime provenance, binary-asset budget 검증을 추가한다. |

## 3. 기술적 선택과 그 이유

### 3.1 SAM 3 detector와 FPN 구현 재사용

SAM 3.1은 비용이 큰 image 및 text encoder와 detector stack을 SAM 3와 공유한다. 구현은 재사용 가능한 detector hook을 추출하고 FPN 구성을 공용 module로 이동한다. `TriFpnNeck`은 checkpoint root 세 개를 로드하지만 각 branch를 동일한 FPN layer code에 위임한다. Frame 0은 backbone을 한 번 계산해 detection과 tracker 초기화에 재사용하고, 이후 각 frame은 backbone 한 번과 tracker feature path 한 번을 계산한다.

이 접근은 기존 parity suite에서 SAM 3 checkpoint 동작을 계속 검증하며, preprocessing, dtype, box conversion, NMS, mask interpolation에서 서로 달라질 수 있는 병렬 detector 구현을 피한다.

### 3.2 Multiplex bucket 사이에서 object 순서 보존

`MultiplexState`는 object를 고정된 16-slot bucket에 할당하고, 사용하지 않는 slot을 padding하며, object-to-slot 역매핑을 기록한다. Decoder와 memory 연산은 bucket 공간에서 실행한 뒤 frame-0 object 순서로 demultiplex한다. 따라서 1부터 시작하는 `object_id`는 frame 사이와 bucket 하나에서 두 개로 전환할 때도 안정적으로 유지된다.

Unit 검증은 padding이 있는 object 3개, 가득 찬 16-object bucket, 두 bucket으로 넘어가는 object 17개를 실행한다. Video acceptance test는 object 9개에도 이후 frame마다 `track_step` 호출이 한 번만 필요함을 검증한다.

### 3.3 Checkpoint 이름보다 tensor metadata 신뢰

고정된 `mlx-community/sam3.1-bf16` revision에는 F32 tensor 1,961개가 들어 있다. Runtime probe는 로드된 parameter 1,961개가 모두 F32로 유지되고, preprocessed input이 F32이며, tracker logit과 score도 F32임을 확인한다. 따라서 loader는 repository 이름에 따라 강제 변환하지 않고 checkpoint에 실제 저장된 data를 유지한다.

이는 독립적으로 계속 검증되는 SAM 3 BF16 acceptance path와 다르다. PR은 공용 loader와 detector refactor가 quantized weight를 손상하지 않았는지 확인하기 위해 실제 affine 4-bit SAM 3 inference도 다시 실행한다.

### 3.4 Allocation-heavy transform 전에 failure bound 적용

Configuration validation은 model construction 전에 실행된다. 공용 detector head와 dimension, quantization block 두 개, 공개된 정확한 SAM 3.1 decoder, memory-attention, downsampler, fuser, RoPE, multiplex, history constant를 검증한다. Raw detector와 tracker logit, box, presence value, 선택된 mask는 sigmoid, interpolation, thresholding 전에 finite여야 한다. Indexing 전에 tensor rank와 query/object cardinality를 검사한다.

Video 초기화는 선택된 query-instance pair를 region box로 filter하고, source-resolution resize 전에 예상 object count와 working-set budget을 검사한다. 512 MiB guard는 retained output, tracker 및 resize work, retained frame feature, pointer, conditioning concatenation 및 projection, current-frame attention activation을 checked arithmetic으로 계산한다. Test는 overflow를 증폭하는 dimension과 bound를 넘는 12-object, 32-frame case를 다룬다.

### 3.5 공용 prompt와 output contract 유지

Image `detect`와 `segment`는 공용 repeatable text 및 pixel-box prompt surface를 사용한다. Video `track`에는 text prompt가 필요하다. 제공된 box는 IoU가 upstream threshold를 넘는 text detection을 유지하는 source-pixel region으로 작동한다. Frame-0 instance가 안정적인 label과 object ID를 정의한다. 이후 score는 `sigmoid(object_score_logits)`이고, box는 nonzero mask extent에서 만들며, mask는 nearest-neighbor source resize와 threshold `> 0`을 사용하고, row는 공용 `Instance` 및 `DenseOutput` contract를 사용한다.

Frame-0 detection을 통과한 결과가 없는 video는 frame metadata와 함께 빈 결과를 기록하고, single-frame video는 `track_step`을 호출하지 않고 frame 0을 기록한다.

## 4. Provenance

| 구성 요소 | 고정 값 |
|---|---|
| Upstream 구현 | `Blaizzy/mlx-vlm` commit `abd3b2d7b728cb5372d4e092260da325e5ded209` |
| SAM 3.1 checkpoint | `mlx-community/sam3.1-bf16` revision `a992e302ea9b0f03f41dfd93414a4fd0e818f65b` |
| Checkpoint safetensor | SHA-256 `a1b1c19dcc9bdd68438bcd74433fadc90740e73c37a1f386872672d134879c42`; F32 tensor 1,961개 |
| Processor asset | `mlx-community/sam3-bf16` revision `dfe573c3171dbcfda8399c650d9135afa7e94592` |
| Reference runtime | 정확한 mlxcel ROCm archive의 MLX 0.32.5 |
| Native library | SHA-256 `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094` |
| Python binding | SHA-256 `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc` |
| 검증 device | ROCm gfx1151, AMD Radeon 8060S Graphics, compute unit 20개, warp size 32 |
| Traffic source | `opengeos/datasets` cars video, source SHA-256 `fdfe751efd5423cb70be1440bb2e9850d63715bf3ccb67fd2743f1bb514b687f` |

`tests/fixtures/sam3_1/provenance.json`은 source checkout command, checkpoint 및 processor hash, 정확한 native path 및 hash, device 선택, TF32 control statement, clip 및 decode command, 이식 가능한 reference script, committed fixture hash, 캡처한 output hash를 기록한다. Commit된 three-frame MP4의 SHA-256은 `115950dbc5d9572e6103d2e9649ab4dbc7e73952696195cb0e1380ecd4ea9e7a`이다.

## 5. Runtime 검증

### 5.1 Unit, registry, bucket 동작

| Case | 결과 |
|---|---|
| SAM 3.1 focused unit suite | 14 passed, 0 failed, 실제 checkpoint test 2개 ignored |
| Registry focus | Model alias 두 개와 mask capability를 포함해 3 passed |
| Multiplex size | Object 3, 16, 17개의 round-trip 및 padding 검사 passed |
| Conditioning | Closest-frame 선택과 고정된 temporal sine-position literal passed |
| Edge case | Empty detection, single-frame tracking, non-finite output 거부, Pillow mode-F mask resize, memory-bound 검사 passed |

### 5.2 실제 F32 cat segmentation

실제 고정 checkpoint는 F32 parameter와 F32 input으로 로드되어 8.14초에 prompt 기반 segmentation을 완료했다. Upstream reference와 같은 detection 두 개를 반환했다. Score 차이는 0.01 이내이고, 모든 box coordinate 차이는 2 pixel 이내이며, source-resolution mask 두 개 모두 요구된 IoU 0.98 이상을 충족한다. Output score는 F32를 유지하고 mask는 u8이다.

### 5.3 9-object video tracking

3-frame 640x360 traffic fixture는 실제 자동차 9대를 초기화하고 instance row 27개와 비어 있지 않은 mask 27개를 생성한다. 모든 frame에 object 9개가 모두 있으며, 안정적인 `object_id` 값, 0.01 이내의 score, 2 pixel 이내의 box, mask별 IoU 0.95 이상이 고정된 upstream reference와 일치한다. 모든 reference foreground count는 0보다 크고, 실제 mask는 각 reference와 양의 overlap을 가지므로 비어 있지 않다.

Tracker는 frame 0 이후 각 frame에 한 번씩, object별 호출이 아니라 정확히 두 번의 `track_step` 호출을 기록한다. 같은 실제 test는 single-frame contract와 empty-threshold contract도 검증한다. 이 test는 29.44초에 passed했다.

### 5.4 기존 SAM 3 regression

최종 binary는 focused run에서 accelerator 기반 case 6개를 ignored한 채 focused SAM 3 unit test 15개를 passed했다. 이어서 기존 실제 BF16 reference case 네 개를 모두 33.50초에 passed했다. 대상은 24-layer text parity, text-only cat segmentation, box-prompt segmentation, three-frame video tracking이다. Cat mask IoU는 1.0과 1.0을 유지하고, box-prompt mask IoU는 0.9999825와 1.0을 유지하며, text projection은 BF16 동작을 유지한다.

실제 affine 4-bit checkpoint도 로드되어 6.15초에 prompt 기반 segmentation을 완료했다. 이는 공용 detector, FPN, validation, loader 변경이 BF16과 quantized SAM 3 path를 모두 보존함을 확인한다.

### 5.5 검증 log

| Log | 근거 |
|---|---|
| `issue-2430-compile-v3.log` | 최종 V6 test-fast library test binary가 10분 만에 성공적으로 build됨 |
| `issue-2430-sam31-units.log` | SAM 3.1 unit 및 edge-case 검증; 14 passed |
| `issue-2430-sam31-registry.log` | SAM 3.1 alias 및 architecture capability; 3 passed |
| `issue-2430-sam3-units.log` | 기존 SAM 3 focused regression; 15 passed |
| `issue-2430-sam31-real-cat.log` | 실제 F32 two-cat segmentation parity; 1 passed |
| `issue-2430-sam31-real-video.log` | Nine-car, three-frame multiplex parity 및 edge contract; 1 passed |
| `issue-2430-sam3-real-bf16.log` | 기존 실제 SAM 3 BF16 case 네 개 전체; 4 passed |
| `issue-2430-sam3-real-4bit.log` | 실제 affine 4-bit SAM 3 inference; 1 passed |
| `issue-2430-gpu-batch.log` | 배타적 ROCm guard가 계속 유지됨; 모든 target이 0으로 종료 |
| `issue-2430-validation-summary.md` | Frozen-source, binary hash, 검증 결과, review, 대기 gate |
| `issue-2430-clippy-v2.log` | Current-head `cargo clippy --lib --tests -- -D warnings`; 21분 4초에 passed |
| `issue-2430-cli-build-v2.log` | Current-head ROCm CLI binary; 9분 19초에 built |
| `issue-2430-fast-contracts-final.log` | Repository contract check 8개; 모두 passed |
| `issue-2430-cli-gpu-batch.log` | Guarded image 및 video CLI prediction; 둘 다 exit 0 |
| `issue-2430-cli-metrics.json` | JSON, PNG, tensor dtype, shape, instance, numerical artifact 검증; exit 0 |
| `issue-2430-cli-arch.json` | Architecture family, alias, runtime, output-kind dispatch 검증 |
| `issue-2430-cli-redirect-alias.log`, `issue-2430-cli-redirect-video.log` | 두 generation command 모두 예상된 `use mlxcel predict` message와 exit 1 반환 |

Compile된 V6 test binary는 `target/test-fast/deps/mlxcel-d5e762440845331a`이고 SHA-256은 `a7af693e828828b5f7b5d32b2949f9881c061aec9729d9711b077cec2ce7718c`이다. 이 binary의 library runtime 및 실제 checkpoint parity run은 이후의 기계적인 CI fix보다 먼저 실행됐다. 해당 fix는 사용하지 않는 private tracker-ID storage 제거, divisibility expression의 `is_multiple_of` 치환, binary-asset budget expression을 parser가 읽을 수 있는 한 줄로 유지하는 변경이다. 이 변경들은 model 계산이나 public output contract를 바꾸지 않는다. Current-head clippy와 실제 CLI run은 fix 이후 source를 검증했다. 해당 CLI binary는 `target/test-fast/mlxcel`이고 SHA-256은 `c8dd6bb4d56a7fc447895ebfbe93790476c10e7d4a0ba4ae6be815b1a81a4bfb`이다. `cargo fmt --all -- --check`, `git diff --check`, fast repository contract 8개가 모두 passed했다. 독립적인 구현 review와 독립적인 security/performance review는 최종 review snapshot에서 finding 없이 종료됐다.

### 5.6 CLI artifact 및 dispatch 검증

Current-head CLI binary는 ROCm guard 아래에서 실제 image 및 video prediction을 완료했다. Image prediction은 의도적으로 `--task`를 생략해 default `segment` path를 실행했다. `result.safetensors`, 안정적인 PNG mask 두 개, `instances.json`, `meta.json`을 기록했다. Mask IoU 두 값은 모두 1.0이고, 최대 절대 box 오차는 `0.000064765625` pixel이며 최대 절대 score 오차는 `0.0000000205734253`이다.

Video prediction은 `result.safetensors`, 안정적인 `fNNNNNN_oNNN.png` file 27개, `instances.json`, `meta.json`을 기록했다. Validator는 3개 frame에서 안정적인 ID를 가진 비어 있지 않은 mask 27개와 `0.999113475177305`의 최소 mask IoU를 확인했다. 또한 tensor mask가 u8이고 box와 score가 F32인지 검사하고 JSON row, tensor row, PNG artifact를 서로 대조했다.

`mlxcel arch --json`은 family `sam3_1`, model type 및 alias `sam3.1`과 `sam3.1_video`, runtime `predict`, output kind `masks`를 보고한다. 어느 model type으로든 `mlxcel generate`를 호출하면 예상된 exit code 1을 반환하고 caller를 `mlxcel predict`로 안내한다.

### 5.7 대기 중인 통합 근거

다음 항목은 아직 acceptance evidence가 아니며 중앙 동기화 후 갱신해야 한다.

- 선행 Video Depth Anything 및 MoGe 작업의 중앙 통합 후 focused SAM 3.1 및 공용 SAM 3 regression 재실행.
- Repository asset/provenance 및 필요한 platform gate를 포함한 통합 head의 최종 PR #2575 CI.

## 6. 위험과 호환성

이 PR은 server endpoint를 추가하지 않으며 SAM 3에서 도입한 공용 prompt, instance, mask, artifact contract를 유지한다. Checkpoint sanitizer는 멱등성을 유지하고 혼합 convolution layout을 거부한다. Region box는 text prompt를 대체하지 않고 text detection을 filter하며, video default는 명시적으로 유지된다. Caller는 `track`을 요청해야 한다.

Runtime parity 근거는 고정된 native MLX library를 사용하는 Linux ROCm gfx1151로 제한된다. 이 보고서는 full-checkpoint Metal 또는 CUDA parity를 주장하지 않는다. Platform CI는 compilation 및 일반 test 호환성을 입증할 수 있지만, 고정된 실제 checkpoint test를 실행하지 않으면 model parity 근거가 아니다.

Realtime webcam caching, video 중간 object 추가 및 reconditioning, interactive point refinement, 제한된 Meta `.pt` checkpoint에서의 conversion, annotator는 issue #2430 범위 밖이다.

## 7. 후속 조치

- VDA와 MoGe 작업을 중앙 통합 branch에 동기화하고 focused SAM 3.1 및 SAM 3 regression을 다시 실행한다.
- 통합 head에서 최종 PR CI를 실행하고 결과 근거로 이 보고서를 갱신한다.
- 대기 중인 draft gate가 통과한 뒤에만 PR #2575를 ready로 표시한다.

## 참고 자료

- PR #2575: `feat: add SAM 3.1 multiplex tracking`
- Issue #2430: Object Multiplex video tracking을 포함한 SAM 3.1 port
- PR #2551 / issue #2429: 공용 SAM 3 detector, prompt, instance, mask, output artifact
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: 공용 CLI 및 output contract
- `tests/fixtures/sam3_1/provenance.json`: source, checkpoint, runtime, fixture provenance
