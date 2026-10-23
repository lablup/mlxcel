# 기술 보고서: PR #2581 — 네이티브 RF-DETR 탐지와 인스턴스 마스크

> RF 구현과 모든 로컬 통합 source, runtime, CLI, unit, architecture, contract, format, fixture, lint gate가 통과했습니다. 보고서 포함 head의 CI와 merge 신원이 기록될 때까지 이 보고서의 상태는 Partial입니다.

| 항목 | 값 |
|---|---|
| 작성일 | 2026-10-24 |
| PR | #2581 |
| 이슈 | #2431 |
| 통합 소스 head | `383d77a2a22231f9cad108b147ff7562f0e5b800` |
| 병합된 Body base | `68d7cc36d6e7fe4f61f126a5490467a7fe90f5d3` |
| 기존 runtime 소스 | `f673b9d134f61b0d43e22d1f627036e53a002f6f` |
| 기존 lint/metadata head | `d36af6213788ba50969d926787333f9e2d7b3c5e` |
| 상태 | 부분 완료; 모든 로컬 통합 gate 통과, 보고서 포함 head의 CI와 merge lifecycle 대기 |
| 언어 | Rust, Python, Markdown |
| 위험도 | 보고서 포함 head의 CI 완료 전까지 중간 |
| 검증한 backend | 통합 `383d...` ROCm/gfx1151 Base/Seg 동등성, RF CLI, 기존 PP/RT CLI, RT-DETRv2 회귀, 9개 계열 architecture |
| 검증하지 않은 항목 | 최종 contract check, Metal 및 CUDA 추론 |
| 성능 주장 | 없음; benchmark를 실행하지 않음 |

## 요약

PR #2581은 RF-DETR Base 탐지와 RF-DETR Seg Small 인스턴스 분할을 공용 `mlxcel detect` 표면에 추가합니다. 구현은 업스트림 windowed DINOv2 backbone, C2f projector, two-stage deformable decoder, bounding-box reparameterization, class별 NMS, 선택적 segmentation head를 포팅합니다. `rf-detr`와 `rfdetr`를 모두 등록하고, 업스트림 기본 confidence/NMS threshold 0.5를 사용하며, 원본 이미지 pixel box와 COCO label을 출력합니다.

Segmentation mask는 `--masks-dir`로 명시할 때만 요청합니다. 명령은 출력 탐지 순서대로 원본 해상도 binary PNG를 기록하고 JSON에 각 경로를 넣습니다. Box-only 실행은 mask-head 출력과 mask resize/readback을 건너뛰도록 설계했습니다. Detection-only checkpoint는 weight를 읽기 전에 `--masks-dir`를 거부하고, mask 출력은 기존 파일, symlink, 쓰기 불가능한 경로를 거부합니다.

기존 runtime 소스 `f673b9d1...`은 Rust type check, 집중/detect binary-unit, 실제 Base/Seg 수치 동등성, RF CLI 수락, 기존 PP/RT CLI 호환성, RT-DETRv2 회귀를 ROCm/gfx1151에서 통과했습니다. `d36af621...`의 좁은 lint/backend metadata 변경은 모델 수학을 바꾸지 않으며 warning-denying clippy와 독립 동등성 검토를 통과했습니다.

Clean 통합 head `383d77a2...`는 병합된 Body `68d7cc36...` 위의 추가 commit 3개입니다. 독립 integration/문서 검토에 finding이 없습니다. 필수 문서 두 개는 동일한 9개 계열 표를 포함하고 checkpoint `model_type`과 `rfdetr` registry/loader alias를 올바르게 구분합니다. Source-bound library build, 집중/binary unit, Base/Seg 동등성, RT-DETRv2 회귀, RF와 기존 계열 CLI 수락, 9개 계열 `arch --json`, contract, format, fixture 검증, warning-denying clippy가 모두 통과했습니다. 이 보고서를 포함한 commit의 전체 CI 결과와 merge 신원만 대기 중입니다.

## 1. 문제와 범위

이 PR 이전에는 `mlxcel detect`가 RF-DETR checkpoint를 읽을 수 없었습니다. RF-DETR은 window별 CLS token, local/global backbone attention 교대, multi-scale projector, two-stage query 선택, 선택적 mask head를 사용하므로 표준 DINOv2 계열과 다릅니다. `mlxcel predict`로 라우팅하면 한 detector 계열이 두 공개 명령으로 나뉘는 문제도 있습니다.

선택한 설계는 RF-DETR을 공용 `Detector` API와 `detect` CLI에 둡니다. 공개된 F32 Base와 Seg Small checkpoint layout을 지원합니다. PyTorch 변환, video/realtime 모드, annotator, server 노출은 이 PR 범위가 아닙니다.

## 2. 구현

### 네이티브 graph와 전처리

새 `src/vision/detection/rfdetr/` 모듈은 다음을 구현합니다.

- checkpoint resolution의 정사각 RGB 전처리와 ImageNet normalization
- RF 전용 windowed DINOv2 backbone과 position interpolation
- C2f multi-scale projector
- two-stage query 선택과 deformable decoder
- bounding-box reparameterization과 class별 greedy NMS
- 선택적 segmentation head와 원본 해상도 bicubic mask resize

Base checkpoint는 resolution 560, 추론 query 300개, decoder layer 3개를 사용합니다. Seg Small은 resolution 384, 추론 query 100개, decoder layer 4개, 96 x 96 raw mask logit을 사용합니다.

### 공용 CLI 동작

`load_detector`는 기존 detector 경계를 통해 RF model type을 dispatch합니다. CLI는 detector별 기본 threshold를 유지하고 안정적인 출력을 위해 탐지를 기하학적으로 정렬하며, `--masks-dir`가 있을 때만 mask 경로를 추가합니다. 기존 RT-DETRv2와 PP-DocLayoutV3 JSON field는 바뀌지 않습니다.

의도한 공개 흐름은 다음과 같습니다.

```bash
mlxcel detect -m models/rfdetr-base-fp32 -i image.png --format json
mlxcel detect -m models/rfdetr-seg-small-fp32 -i image.png --format json --masks-dir masks
```

### Checkpoint와 자원 수락

Config parsing, checkpoint index, safetensors header, tensor shape/dtype, dimension, allocation 총량, 이미지 payload, 출력 mask workset을 비싼 작업 전에 제한합니다. Sanitization은 업스트림 namespace remap, 추론에 사용하지 않는 mask token 제거, fused decoder QKV 분할, 필요한 경우에만 convolution weight transpose를 수행합니다.

검토한 보안 모델은 선택한 checkpoint directory가 load 동안 안정적이라고 신뢰합니다. 소비하는 config, processor, index, weight 파일을 검증하고 신뢰하지 않는 dimension과 allocation을 검사합니다. 권한 있는 프로세스가 이미 선택한 directory를 동시에 교체하는 상황까지 보호한다고 주장하지 않습니다.

## 3. 인증된 업스트림 레퍼런스 근거

Tracked reference package는 변경하지 않은 mlx-vlm revision `4319ef2c837b7226b758c3dd69edda51b7079785`, 수정된 ROCm LayerNorm runtime, 커밋된 비정사각형 cats 이미지, 정확한 공개 checkpoint를 사용합니다.

| Variant | Checkpoint tensor | Sanitized loaded parameter | 공개 탐지 | Mask |
|---|---:|---:|---:|---:|
| Base | 487 F32 | 498 F32 | 5 | 없음 |
| Seg Small | 544 F32 | 559 F32 | 4 | 비어 있지 않은 원본 크기 mask 4개 |

변경 불가능한 Base와 Seg receipt hash는 각각 `cd322b62ae9902a1ef05f2f62c02ff0935e6ff92514d1255efab205bbde53cd9`, `6cd1e05437fcbf975ccee2917f44094751c4442840e94250e9cd0b4e3897c49c`입니다. Fixture package에는 tracked entry 46개가 있고 inventory identity는 `91ff3b3d3a6136c33155bf69c7c1b46f7387a0f70600e8346819f8c7d9e8b063`입니다.

독립 audit `fc53893e3bc3d1235140b47f155708ff68db231933d159dddbb905a1e405879b`는 어느 모델도 실행하지 않고 보존한 정확한 공개 postprocessor를 다시 실행했습니다. Base와 Seg의 box, score, label, query index가 정확히 일치했고 PNG mask 4개도 공개 cubic resize와 strict-positive 결과에 정확히 일치했습니다. Fail-closed verifier는 두 receipt를 고정하고 `python -O`에서 detection, receipt map, inventory를 함께 변조한 경우를 거부합니다.

이 결과는 예상 입력과 출력을 인증합니다. 업스트림 결과만으로는 Rust 동등성을 확립할 수 없으며, 아래의 별도 Rust gate가 이를 검증합니다.

## 4. Rebase 전 기존 Rust 수치 종료

집중 build는 검토한 source `f673b9d134f61b0d43e22d1f627036e53a002f6f`, source snapshot SHA-256 `330f4cae5ce7584df6da3e36040fad92ae855cb30603c88d4adb39195eea61d3`에서 10분 52초 만에 성공했습니다. Test binary `target/test-fast/deps/mlxcel-d5e762440845331a`의 SHA-256은 `48bfd9cfda13d34724c08e4494e60fc370bda248edc28febad44d0333952fb4e`입니다.

집중 unit 선택은 23개 통과, 실패 0개, 실제 모델 test 2개 ignore를 기록했습니다. Sanitization, configuration, allocation limit, preprocessing admission, window partition, bbox reparameterization, NMS, nonfinite/oversized output 거부, custom label, `unrequested_masks_skip_the_mask_computation_entirely`를 포함합니다.

별도 `src/main.rs` unit binary는 선택한 `commands::detect` test 8개를 모두 통과했고 관련 없는 test 303개를 필터링했습니다. 안정적인 기하/신뢰도 정렬, 선택적 reading order와 mask JSON field, detection-only model의 runtime 초기화 전 mask 요청 거부, 기존 mask symlink를 통한 덮어쓰기 거부를 검증합니다. Test log SHA-256은 `5b1fef991fc9d50335dd36d5c2d4ad14c4ab2456de2dbdcd627f39f72b5da8a4`이고 build/test status는 모두 0입니다.

첫 clippy 실행은 test-only initializer lint 4개를 찾은 뒤 status 101로 보존했습니다. 수정 diff `654e1c03f4bd53fc4a94560edd98d88cec17d62946d2a332d092e52ecea439b7`은 해당 test initializer 4개를 바꾸고 RF registry backend metadata를 ROCm supported, Metal/CUDA partial로 바로잡습니다. 모델 수학은 바꾸지 않습니다. 두 번째 warning-denying clippy는 2분 만에 status 0으로 완료했고 log와 launcher SHA-256은 각각 `9e1074615024b5fbdd1776d4cddb0accb68119065f90f907c4097eb1fbaa2c47`, `31038483c326c6ba582e085d81198af733bc57ad3c7c32ceda26d559de8f85bc`입니다. 독립 repository-map 검토는 동등성 문제가 없음을 확인했고, 변경은 `d36af6213788ba50969d926787333f9e2d7b3c5e`로 commit/push되었습니다. 실제 runtime 근거는 변경하지 않은 source `f673b9d1...`에 계속 결합됩니다.

변경 불가능한 116-entry 기존 packet은 `proof/pre-rebase-f673-d36-v1`에 있습니다. `SHA256SUMS`, receipt, payload manifest의 SHA-256은 각각 `4013b8c3af201ed57a56460acd506425b5a8f2311c6b9697e44e37ab91da578b`, `1b4281dc9a64b3aa651fc2dd96317aff73bda311b9ebf4492710066922ed362c`, `4a946243dbf1762d9213bf138254af8e54525eaa7900dfb30375da8d8c32ff57`입니다. Packet은 `f673...` runtime 소스와 `d36...` metadata-only 소스를 명시적으로 분리합니다.

Ignore된 실제 test 2개는 GPU guard 아래에서 별도로 실행했습니다.

| Variant | Processor max abs | Logits max abs | Boxes max abs | Raw masks max abs | 최종 결과 |
|---|---:|---:|---:|---:|---|
| Base | `0` | `3.342628479e-4` | `1.327395439e-4` | 해당 없음 | 최종 NMS 동등성 통과 |
| Seg Small | `0` | `0` | `0` | `5.340576172e-5` | Mask 4개 모두 IoU `1.0` |

모든 값은 tolerance 완화 없이 이슈의 raw `1e-3`와 mask IoU `0.98` 요구를 만족합니다. 실제 공용 detector RT-DETRv2 회귀도 gfx1151에서 통과했습니다. 1개 통과, 실패 0개이며 child/guard status가 모두 0입니다. Child log SHA-256은 `ab6ffe805083127aa94cb6d275fa5a911eec07c2a8cf1fd71585ff365b5387a3`입니다.

### Rebase 전 CLI와 기존 계열 검증 종료

실제 CLI gate는 binary SHA-256 `0ca46ccc86f56df111c1094aedd717d8ccdee0ade6543aaf1ee13cb81f7d2334`를 사용했습니다. Source receipt `512d41b447156df51ed2a6b935767d4a5631b28e3a524c6bd734d2182388890e`는 이 binary를 clean source `f673b9d134f61b0d43e22d1f627036e53a002f6f` 및 source snapshot `330f4cae5ce7584df6da3e36040fad92ae855cb30603c88d4adb39195eea61d3`에 결합합니다.

RF 수락 receipt `43dd5559d560daebdc713658ab3bea1fe2ddb1b83c33068aca320ecf9d7c1414`는 계열 기본 threshold 0.5에서 Base 탐지 5개와 Seg 탐지 4개를 기록합니다. 원본 해상도 binary mask 4개는 모두 인증된 fixture와 IoU `1.0`이고, 출력 순서에 맞는 파일명을 사용합니다. Seg box-only 경로는 mask field 없이 label, box, confidence, count를 유지합니다. Detection-only mask 요청과 쓸 수 없는 mask 경로는 GPU 실행 전에 실패하고, threshold 1은 빈 목록을 반환하며 PNG를 쓰지 않습니다. Child와 guard status는 모두 0입니다.

기존 계열 receipt `e152c7f9583d2fe8123bbc16c657e1808770a9518b0d4e2e1508aa3cc2832bf4`는 PP-DocLayoutV3 checkpoint 각각에서 순서가 있는 탐지 16개, RT-DETRv2에서 탐지 28개를 기록합니다. PP 출력 두 개는 동일하고, 세 실행 모두 기본 threshold 0.3과 mask가 없는 정확한 JSON schema를 유지하며, child와 guard status는 0입니다. Gate는 Falcon OCR model/processor 소스가 merged PP baseline과 byte-identical임을 검증하고, 관련 소스 차이가 새 detection field를 `mask: None`으로 초기화하는 한 곳뿐임을 검증합니다. 새 OCR 실행을 주장하지 않고 기존의 인증된 전체 OCR 실행을 고정합니다.

### 최종 추가 integration 소스 신원

Clean integration chain은 병합된 Body `68d7cc36...` 위의 `383d77a2...`(문서 통합), `acd22bdf...`(lint/backend metadata), `c0e219b5...`(RF 구현)입니다. 통합 tree SHA는 `f2d5682707a1ed204827d6ad2c91a31f2e59b942`입니다.

`docs/dense-prediction.md`와 `docs/supported-models.md`는 table SHA-256 `4f7c113fcb497711097c2b19c1ba1178845ad11e0b8b1912166b35b00917dd79`인 byte-identical 9개 계열 contract 표를 포함합니다. 두 page 모두 RF-DETR은 `detect`를 사용하고 segmentation checkpoint와 `--masks-dir`를 함께 사용할 때만 mask를 쓰며, 나머지 `predict` 계열은 `instances.json`을 포함한 shared writer를 사용함을 명시합니다. Registry와 loader는 `rf-detr`를 checkpoint spelling, `rfdetr`를 허용 alias로 동일하게 처리합니다. 독립 integration/문서 검토에 남은 소스/문서 finding은 없습니다.

### 통합 library와 runtime 검증 종료

최종 integration packet은 source archive SHA-256 `9295e4d7952454622353c157f77e36a446ffa3b315eb3da27b2e01c77eefe3ec`와 source-envelope SHA-256 `3e510f289bcf34b7f52bd0cc92283f4d66eec1ce3aba2f70db664452fb1804a2`를 clean head `383d77a2...`에 결합합니다. 통합 library test binary SHA-256은 `c53a6fdeda249f74916a0ef764d910069aac8f32556ed98dba1d727c1b71170e`입니다. 9분 43초 build는 status 0으로 끝났고 log SHA-256은 `29cef0210219cc0def2b7ad77425435e9160e40e91e5b5a0fec6612d9d09a895`입니다.

이 binary의 집중 선택은 test 23개 통과와 별도 실행할 인증된 실제 모델 test 2개 ignore를 기록했습니다(log SHA-256 `482f8691f678c114a35889a4d2ecff6935f46946b8294e5d2bcbf3eff122f3be`). 별도 guard에서 실행한 Base 동등성, Seg 동등성, RT-DETRv2 회귀는 모두 child/guard status 0으로 완료했습니다. 각 log SHA-256은 `e0ec1281b73dd3995c4b0c5679f7f0869f1c4ae8f2d9bb0814a67ed0bb349c40`, `b24156c4985fd12e4002b993bdac4ee6edb1591a9e00ac502e7cd536c494430c`, `d792499cfbe247a31264c627b3d213fa48f808b6e19dc9c5276566248e560c7d`이고 guard log SHA-256은 `7486977be14120ab93ca31e915eb64a24aa99de91b6c3272e5384993a41cffa4`입니다. 수치 결과는 위 표의 기존 수락 값과 정확히 동일합니다.

### 통합 CLI와 architecture 검증 종료

정식 통합 CLI SHA-256은 `328d10b9fffe842e981eeca317a9a00eb18c8d91cc0f5c04e5d11be102b9a685`이고 source/build receipt SHA-256 `bf12b39ec441c17463603e190e6b1d65b58fc4e252107077bcbcc9894444b831`에 결합됩니다. RF 수락 receipt SHA-256 `cd63ca1e4aeae89c598efdcb6ef74e2857837b12df7c572857b8e5606b095c4a`는 Base 5개, Seg 4개, mask IoU 4개 `1.0`, Seg boxes-only 동등성, runtime 전 mask error, threshold-1 빈 출력을 기록합니다. Child/guard log SHA-256은 `4bddd0f2ca3ec90e23b728b0d2e0adee27bf63179163456270a2e8c12f88de98`, `7bb5acf0166eae50e0db80172b8c60906157267b70939aeeb983c3e08fc44cb9`이고 status는 모두 0입니다.

기존 계열 receipt SHA-256 `553f670b3e55ef0e944e9f9ff6d16ea0198f29cf5ed4f16699b6077bf83dd777`는 PP 두 layout의 동일한 순서 출력 16개와 RT-DETRv2 출력 28개를 기록합니다. 기존에 인증한 전체 OCR proof를 고정하고 새 OCR 실행을 주장하지 않으면서 현재 Falcon source 동등성을 검증합니다. Child/guard log SHA-256은 `b8e6d5d7890b18d491a4568a277f5fd1e78f1d351f0d951422d4202c5b78ef86`, `f3e5df928966fe516e3b279157f7760e6558ec798ede031aa4a174fac6c99e8b`이고 status는 모두 0입니다.

SHA-256 `df9b686c2d0399d6b7acac48b8a8e0fdb5fd40b6d25d6090dbecf8dc39f4db19`인 최종 `arch --json` validator는 문서화한 9개 계열 모두 `PASS`를 보고합니다. 결과 SHA-256은 `433721d43ce978779e48971a2385cc5627a96ca695e3d0d908021ef2694bf1e8`이고 status는 0입니다. RF-DETR은 runtime `detect`, `model_types: ["rf-detr"]`, alias `rfdetr`, output `boxes`, ROCm supported, Metal/CUDA partial로 보고되어 검토한 문서 표 두 개와 일치합니다.

통합 binary-test executable SHA-256은 `851bd5103af9399ca0ff803c05dca5ed09c6ca69b596eee78bf1702935f1a93a`입니다. Build log SHA-256은 `e847c4bf11774b9053042744e412c322a302ddfb968175c4e9ebb4f0d286fd00`, 선택 실행 log SHA-256은 `413804bd719d1d8e4f37b442ab677636e5fa7e5eb1c8017b5cb1e55d153bf453`이고 status는 모두 0입니다. Detect 정렬, JSON, mask opt-out, symlink/runtime 전 error test 8개를 포함한 command test 20개가 실패/ignore 없이 모두 통과했습니다.

### 통합 품질과 contract 검증 종료

Warning-denying clippy(`--lib --tests --bin mlxcel -- -D warnings`)는 1분 48초 만에 status 0으로 완료했습니다. Wrapper와 log SHA-256은 `fec136558b2459bf3a46e126fdb4a59eb834bffff204449631765bd1dbb2fd34`, `c30358c1535a2cfe96d5767ad4efb1a34409bb8d4302c389e93fc103c6ff70c9`입니다.

최종 fast-contract suite는 crate version, kernel dtype key, 환경 변수 문서, kernel-port dispatch, llama compatibility, ROCm overlay/ownership, tracked binary asset, Windows reserved path의 8개 group을 모두 통과했습니다. `cargo fmt --all -- --check`와 `git diff --check`도 통과했습니다. Launcher, core script, log SHA-256은 `759a6d1203d592579e4c3b81a61e5264c13261be4a263fc2125483c0462e86d2`, `d8eb7481ece1ba762239805cecfa4e22a8164cecab812f7fa15bfdce6d3db996`, `ab2293b74ba215e53a8c5ac09e9cf5cc3f8451c4f2c008eee79d1654c0294d61`이고 status는 0입니다.

수정한 optimized fixture verifier 두 번째 실행은 `PASS: 46 authentic RF-DETR fixture files`, status 0을 보고합니다. Wrapper와 log SHA-256은 `cd0492d8bc46cb8e5367b9f474c52e12557d2374c51c07129eec8c4c5ea5f9d2`, `2134b53eef7f2e7e9f9fa368cf67f07c862676e7b513bb317de2d3e8f8514c57`입니다. 첫 번째 missing-Pillow 환경 실패는 보존하고 근거에서 제외합니다. 최종 CLI/architecture evidence index SHA-256은 `06a0db219c59af1ec7d4fc168cbfed41afebcabfe2327cddd9255cdb014483f3`입니다.

## 5. 검증 상태

| Gate | 상태 | 근거 또는 요구 사항 |
|---|---|---|
| 기존 Rust type check | 통과 | Rebase 전 소스 `f673b9d1...` |
| 독립 구현/보안 소스 검토 | 통과 | 기존 구현에 남은 source finding 없음; stable-directory trust scope 문서화 |
| 통합 소스/문서 검토 | 통과 | Clean `383d77a2...`; 추가 integration과 9개 계열 표에 finding 없음 |
| 인증된 업스트림 Base 레퍼런스 | 통과 | Loaded F32 parameter 498개, 공개 탐지 5개 |
| 인증된 업스트림 Seg Small 레퍼런스 | 통과 | Loaded F32 parameter 559개, 탐지 4개와 mask 4개 |
| Fixture verifier와 변조 거부 | 통과 | 수정된 독립 verifier가 인증된 파일 46개를 보고; immutable receipt gate; `python -O` 음성 검증 거부 |
| 집중 Rust unit suite | Rebase 전 head에서 통과 | 23개 통과, 실패 0개, 별도 실행한 실제 test 2개 ignore |
| Detect binary-unit suite | Rebase 전 head에서 통과 | 8개 통과, 실패 0개, 관련 없는 test 303개 필터링 |
| Rust Base raw/postprocess 동등성 | 통과 | Processor `0`, logits `3.342628479e-4`, boxes `1.327395439e-4`, 최종 NMS 통과 |
| Rust Seg raw/mask 동등성 | 통과 | Processor/logits/boxes `0`, masks `5.340576172e-5`, IoU 4개 `1.0` |
| 직접 mask opt-out 회귀 | 통과 | `unrequested_masks_skip_the_mask_computation_entirely` |
| 실제 RT-DETRv2 회귀 | 통과 | gfx1151에서 1개 통과, child/guard status 0 |
| 실제 Rust CLI 수락 | Rebase 전 head에서 통과 | Base 5개, Seg 4개와 IoU-1.0 mask 4개, box-only, runtime 전 error, threshold-1 빈 출력 |
| 기존 PP/Falcon 호환성 | Rebase 전 head에서 통과 | PP 두 layout의 동일한 순서 출력 16개, RT-DETRv2 28개, mask 없는 정확한 schema, 기존 전체 OCR 근거 고정 |
| Warning-denying clippy | Rebase 전 lint 수정 소스에서 통과 | 두 번째 실행 2분, status 0; 첫 status-101 lint finding 보존 |
| 좁은 lint/metadata diff 동등성 | 통과 | 독립 repository-map 검토에 finding 없음; `d36af621...`로 commit; 모델 수학 변경이나 runtime 재실행을 주장하지 않음 |
| 통합 library build와 집중 unit | 통과 | Source-bound binary `c53a6fd...`, build status 0, 23개 통과와 별도 실행한 인증 test 2개 |
| 통합 Base/Seg와 RT runtime | 통과 | Base/Seg가 수락 metric 재현, Seg IoU 4개 `1.0`, RT 회귀 통과, child/guard status 0 |
| 통합 RF와 기존 계열 CLI | 통과 | Binary `328d10b9...`, RF Base 5/Seg 4와 IoU-1 mask, PP 16/16 동일, RT 28, 모든 child/guard status 0 |
| 통합 binary-unit suite | 통과 | Source-bound binary `851bd510...`, 20개 통과, 실패/ignore 0, build/run status 0 |
| 통합 9개 계열 architecture | 통과 | Validator `df9b686c...`, 9개 계열 모두 검토한 runtime/model-type/alias/backend contract와 일치 |
| 통합 contract, format, diff | 통과 | Fast-contract group 8개, `cargo fmt --check`, `git diff --check`, status 0 |
| 통합 clippy | 통과 | `--lib --tests --bin mlxcel -- -D warnings`, 1분 48초, status 0 |
| CI | 대기 | 이 보고서를 포함한 commit의 전체 필수 check 결과가 아직 기록되지 않음 |
| 9개 계열 문서와 architecture | 통과 | 필수 표 두 개가 동일하고 검토되었으며 최종 `arch --json`으로 교차 검증됨 |
| Body integration | 통과 | RF chain은 병합된 Body `68d7cc36...` 기반 |
| 보고서 포함 head CI | 대기 | 완료된 EN/KO 보고서를 포함한 commit에서 CI 완료 필요 |
| Merge | 대기 | Draft PR이며 merge 신원을 주장하지 않음 |

이전 combined fixture-verifier wrapper가 잘못 보고한 false zero는 근거에서 제외합니다. 수정한 독립 `PASS: 46 authentic RF-DETR fixture files` 결과만 사용합니다.

## 6. 호환성과 제한

- 구현은 검토한 공개 F32 checkpoint 두 개를 대상으로 합니다. 더 넓은 checkpoint 호환성은 주장하지 않습니다.
- RF-DETR은 `mlxcel detect`를 사용하고 JSON을 stdout에 출력하며, Seg checkpoint와 `--masks-dir`를 함께 사용할 때만 mask 파일을 만듭니다.
- 이 detector 명령은 result directory, `result.safetensors`, `meta.json`을 만들지 않습니다.
- Tracked reference path는 원래 신뢰한 local environment와 변경 불가능한 checkpoint 내용을 식별합니다. Portable installation path가 아니라 provenance입니다.
- 이 보고서에서는 Metal과 CUDA 추론을 검증하지 않았습니다.
- Throughput, latency, memory, cross-device benchmark를 실행하지 않았습니다.
- 최종 문서 소스는 실제 runtime, `model_type` alias, task, 구체적 파일을 포함해 수락된 9개 계열을 모두 나열합니다. 게시 전에는 위에 기록한 통합 runtime과 `arch --json` 교차 검증이 필요합니다.

## 7. 변경 요약

| 항목 | 현재 PR 값 |
|---|---:|
| 통합 소스 head | `383d77a2a22231f9cad108b147ff7562f0e5b800` |
| 병합된 Body 대비 변경 파일 | 82 |
| 추가 줄 | 28,029 |
| 삭제 줄 | 64 |
| Body 위 RF commit | 3 |

추가 줄의 대부분은 네이티브 RF 구현과 인증된 업스트림 소스, safetensors metadata, receipt, fixture입니다. Production 변경은 RF module, 공용 detector 경계, detect CLI, registry, test, resource policy, 현재 문서에 집중됩니다.

## 8. 필수 후속 조치

- [x] 통합 library를 build하고 집중 unit, 제한된 Base/Seg 및 RT-DETRv2 runtime gate를 실행했습니다.
- [x] 통합 RF CLI, 기존 PP/RT CLI, 최종 9개 계열 `arch --json` gate를 완료했습니다.
- [x] 통합 command binary-unit 선택 test 20개를 실행했습니다.
- [x] 최종 contract, format/diff check, fixture 검증, warning-denying clippy를 실행했습니다.
- [ ] 이 EN/KO 보고서를 포함한 commit에서 모든 필수 CI check를 완료합니다.
- [ ] 최종 merge commit을 기록합니다.

## 참고 자료

- PR #2581: `feat: add RF-DETR detection and instance masks`
- Issue #2431: 네이티브 RF-DETR 탐지와 분할
- 독립 레퍼런스 audit: `/tmp/epic-2382-rfdetr-reference-review.md`
- 최종 문서 audit: `/tmp/epic-2382-final-docs-audit.md`
