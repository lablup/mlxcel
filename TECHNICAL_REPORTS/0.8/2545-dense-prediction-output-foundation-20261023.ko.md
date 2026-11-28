# PR #2545: Dense prediction 출력 기반 추가

**작성일**: 2026-10-23
**상태**: 부분 완료 (PR 열림; 전체 ROCm workspace 게이트와 CI 재실행 대기 중)
**언어**: Rust, C++, Python, Markdown
**위험도**: 중간

## 요약

PR #2545는 텍스트나 bounding box 대신 map과 keypoint를 반환하는 dense vision 모델의 공용 기반을 만든다. `mlxcel predict` CLI, object-safe predictor와 family registry, 제한이 있는 이미지·비디오 입력, 업스트림과 호환되는 interpolation·preprocessing, f32 safetensors·JSON metadata·선택적 keypoint·PNG preview로 구성된 파일 계약을 추가한다. 이 기반 PR의 family registry는 의도적으로 비어 있으며, epic #2382의 후속 이슈가 모델 network와 post-processing을 추가한다.

구현에는 pinned MLX ROCm overlay의 선행 수정도 포함된다. Linux에서 Metal 경로로 들어가던 `AsStrided`가 HIP device header와 encoder를 선택하도록 고쳤다. gfx1151 집중 테스트, 실제 ffmpeg MP4 경로, package 범위 clippy, formatting, 저장소 contract 게이트는 통과했다. 이 pre-merge 보고서를 작성할 때 Metal 검증, 새 CI 실행, 전체 ROCm workspace 회귀 게이트는 완료되지 않았다.

## 1. 문제 정의

기존 mlxcel에는 token generation과 object detection CLI가 있었지만, pixel 단위 depth, normal, mask, segmentation, point map, human keypoint를 위한 공용 출력 surface가 없었다. 이 기반이 없으면 Video Depth Anything, MoGe-3, Sapiens2, SAM 3D Objects와 공용 DINOv2 backbone의 각 port가 입력 검증, interpolation, serialization, preview, CLI 동작을 따로 만들어야 한다. 그 결과 format이 달라지고 resource safety 로직이 중복된다.

또한 프로젝트의 ROCm 개발 경로에서 이 기반이 compile되어야 했다. pinned MLX overlay의 `AsStrided`에는 업스트림 CUDA-or-Metal dispatch가 남아 있어 Linux ROCm build가 Metal header를 include했고, 새 Rust 경로를 gfx1151에서 compile하고 테스트할 수 없었다.

이 변경이 없으면 model port마다 operator와 downstream tooling을 위한 계약이 달라지고, 이미지·비디오 입력 또는 반복 출력이 무제한 resource를 쓰거나 관계없는 파일을 덮어쓸 수 있다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| CLI와 입력 | `--image` 또는 `--video` 중 정확히 하나를 받는 `mlxcel predict`, model task 선택, JSON object option, frame/FPS 제어, payload/decode/allocation 제한, 단계별 error context를 추가했다. |
| Predictor 구조 | `DenseInput`, `DenseOutput`, `PersonKeypoints`, object-safe `DensePredictor` trait, `model_type` constructor registry를 추가했다. 실제 family table은 model PR이 들어올 때까지 비어 있다. |
| Tensor 처리 | NHWC bilinear, bicubic, nearest interpolation을 port했고, ImageNet normalization, integer scaling, patch 배수 resize, 보호된 OIHW-to-OHWI convolution 변환을 추가했다. |
| 출력 계약 | contiguous f32 `result.safetensors`, `meta.json`, 선택적 `keypoints.json`, depth/normal/mask/segmentation preview를 쓴다. 비디오 preview는 번호가 붙은 directory를 사용한다. |
| 안전한 재실행 | host에는 한 번에 tensor 하나만 materialize하고 writer가 소유한 artifact만 교체한다. 오래된 preview 경로와 directory symlink는 외부 target을 따라가지 않고 처리한다. |
| Registry 통합 | `Runtime::Predict`, `OutputKind::DenseMap`, `OutputKind::Keypoints`를 추가했다. 알려진 dense checkpoint를 `generate`로 실행하면 `mlxcel predict` 안내를 반환한다. |
| ROCm 선행 수정 | pinned MLX `AsStrided` overlay가 HIP encoder를 사용하도록 고치고 overlay provenance를 갱신했다. |
| 문서 | dense CLI, 출력 format, preprocessing API, registration guide, 범위 제한, README index를 추가했다. |

이 보고서를 제외한 구현 diff는 29개 파일, +2,823/-8이다. 관련 commit은 `65cb1fdd`(ROCm 선행 수정), `aee9d056`(기반 구현), `41775825`(문서 표현 수정)이다.

## 3. 기술적 선택과 그 이유

### 3.1 별도의 `predict` surface

Dense 결과는 `generate`의 token stream이나 `detect`의 box row 계약에 맞지 않는다. 별도 command를 두어 기존 interface를 유지하면서 dense 모델에 output directory, task 선택, file 중심 lifecycle을 제공한다. Map, intrinsics, keypoint는 다른 serialization이 필요하고 비디오 sequence가 될 수 있으므로 `detect` 확장은 채택하지 않았다.

### 3.2 Model port보다 먼저 빈 registry 제공

이 PR은 interface를 정의하고 test-only stub family로 검증하지만 dense checkpoint가 이미 지원되는 것처럼 보이게 하지 않는다. 알려졌지만 아직 port되지 않은 `model_type`은 구체적인 unsupported-family error를 내고, `generate`를 잘못 사용하면 `predict`로 안내한다. 이후 model PR은 검증된 계약에 predictor와 registry row 하나만 추가하면 된다.

### 3.3 Safetensors를 무손실 원본으로, PNG를 preview로 사용

모든 tensor 출력은 upstream key와 shape를 유지한 contiguous f32로 저장한다. JSON에는 family, task, source size, elapsed time, finite range, model metadata, 평탄화한 intrinsics를 기록한다. PNG는 파생 preview다. Depth는 16-bit normalization, normal은 `[-1, 1]` mapping, mask는 0.5 threshold, segmentation은 8-bit class ID를 쓴다. 수치 정확도가 필요한 consumer는 visualization을 역변환하지 않고 safetensors를 읽는다.

### 3.4 입력 제한과 보수적인 출력 교체

이미지 loading은 payload, decoder, 예상 RGB-f32 allocation 제한을 적용한다. 비디오 decoding은 공용 ffmpeg loader를 재사용하고 기본 32개 sampled frame과 512 MiB decoded-frame budget을 적용한다. 재실행할 때 writer가 만든 알려진 artifact만 지우고 관계없는 파일은 유지한다. 따라서 기존 directory를 재사용하면서도 recursive deletion 범위로 만들지 않는다.

### 3.5 공용 interpolation은 업스트림 수치 계약을 따름

Resize helper는 NHWC tensor를 사용하고, 업스트림 PyTorch가 생성한 고정 literal로 bilinear·bicubic antialiasing, size·scale-factor mode, nearest sampling을 upsample과 downsample 양방향에서 비교한다. 이후 model port의 parity가 이 세부 동작에 의존하므로 same-size와 dtype 보존 동작도 문서화했다.

## 4. 구조와 통합 계약

```text
image / video
      |
제한이 있는 loader와 sampler
      |
model_type registry -> DensePredictor::predict(task, opts)
      |
DenseOutput { tensors, keypoints, meta }
      |
result.safetensors + meta.json + keypoints.json + PNG previews
```

Model 구현은 object-safe predictor를 노출하고, upstream tensor name을 유지하고, 지원 task를 선언하고, 내부 preprocessing error에 context를 붙이고, lowercase top-level `model_type`을 등록해야 한다. CLI는 바깥쪽 load, predict, write context를 제공한다. Mesh/Gaussian export, streaming/webcam 입력, server route는 이 계약의 범위 밖이다.

## 5. 검증과 제한 사항

| 확인 항목 | 보고서 작성 시점의 결과 |
|---|---|
| gfx1151 dense module 테스트 | 17개 통과. Interpolation, serialization/preview, registry 동작, 보호된 layout 변환, preprocessing을 포함한다. |
| `predict` command 테스트 | 집중 테스트 5개 통과. ffmpeg가 있는 상태에서 실제 MP4 테스트를 다시 실행해 통과했다. |
| Registry와 정확한 contract filter | Registry 테스트 12개 통과. Parser, architecture JSON, generate redirect, catalog projection, video-budget filter가 통과했다. |
| 출력 안전성 | Synthetic round-trip, 번호가 붙은 video preview, 오래된 writer-owned artifact 정리, directory-symlink 교체 테스트가 통과했다. |
| Lint와 format | Package library와 `mlxcel` binary clippy가 `-D warnings`로 통과했다. `cargo fmt --all -- --check`, `git diff --check`도 통과했다. |
| 저장소 fast contract | 보고서 생성 전에 orchestrated fast contract make 게이트가 통과했다. |
| 전체 ROCm workspace 회귀 | 보고서 작성 시 실행 중이며 여기서는 통과했다고 기록하지 않는다. |
| 보고서 commit 이후 CI | 대기 중이다. 보고서 commit이 새 PR 실행을 시작한다. |
| Metal | ROCm Linux host에서는 실행하지 않았고 CI에 맡긴다. |

실제 production family table이 의도적으로 비어 있으므로 이 PR은 실제 dense model inference를 지원한다고 주장하지 않는다. Throughput 또는 memory benchmark도 수행하지 않았다. 여기의 resource 작업은 측정된 성능 최적화가 아니라 hard limit와 한 번에 host tensor 하나만 쓰는 serialization이다.

## 6. 학습 포인트

- Model이 들어오기 전에도 test-only predictor를 등록하면 실제 CLI, input decoding, task 선택, writer를 통과하는 infrastructure를 검증할 수 있다.
- Preview는 수치 교환 format이 아니다. 무손실 f32 tensor와 정확한 normalization range를 함께 저장하면 visualization 선택이 암묵적인 API가 되는 것을 막는다.
- Resource budget은 representation 확장을 포함해야 한다. 압축된 이미지가 payload 제한을 통과해도 decode와 RGB f32 변환 뒤에는 memory 제한을 넘을 수 있다.
- Output directory를 안전하게 재사용하려면 명시적인 소유 목록이 필요하다. 표준 artifact만 제거하면 사용자 파일의 recursive deletion을 피하고 symlink 동작도 테스트할 수 있다.
- 수치 port에는 reference implementation이 만든 fixture가 필요하다. 고정 PyTorch probe는 Rust test 환경에 Python이나 PyTorch를 요구하지 않으면서 interpolation drift를 드러낸다.

## 7. 후속 조치

- Merge 전에 전체 ROCm workspace 회귀 게이트와 새 PR CI 실행을 완료한다.
- CI에서 Metal compile과 테스트를 확인한다.
- Epic #2382의 dependent issue를 통해 DINOv2와 dense model family를 추가한다.
- SAM 3D Objects 구현에서 mesh/Gaussian export를 추가한다. Server wire contract는 client 요구가 생길 때 정의한다.

## 참고 자료

- PR #2545: `feat: add dense prediction output foundation`
- Issue #2349: dense-prediction output surface
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: operator 및 model integration 계약
