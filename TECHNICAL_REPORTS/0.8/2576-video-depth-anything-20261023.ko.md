# PR #2576: Video Depth Anything Dense Prediction 추가

**작성일**: 2026-10-23

**상태**: 최종 integration head `45e895ab`에서 구현 및 local 검증 완료, 현재 CI 대기 중

**언어**: Rust, Python, Markdown

**위험도**: 중간

## 요약

PR #2576은 dense-prediction API와 CLI에 정지 image 및 video용 offline Video Depth Anything inference를 추가한다. 공용 DINOv2 기반 temporal DPT graph, OpenCV 호환 preprocessing, 32-frame overlap window inference, relative scale/shift alignment, metric-scale bypass, source-resolution F32 depth output, frame별 16-bit preview, registry metadata와 공개된 Small, Base, Large relative/metric checkpoint 6종에 대한 문서를 구현한다.

최종 integration head `45e895ab`에서 구현의 local 검증 gate를 완료했다. 최종 library/test build, 집중 unit test 14개, 집중 clippy, formatting, 실제 checkpoint 6종의 load-only 검사, consecutive-prefix regression, Small relative 및 metric의 ROCm/CPU 40-frame parity와 실제 ROCm 64-frame CLI inference가 통과했다. 실제 fixture와 provenance 전체가 commit되었고 정확한 binary inventory와 repository contract group 8개가 `f69dd301` rebase 후에도 모두 통과했다. 독립 implementation 및 security/performance review에는 남은 finding이 없다. Rebase 전에는 모든 non-skipped CI check가 통과했으며 rebased final head의 CI는 대기 중이다.

Runtime 결과는 최초 검증 구현 commit `3f01fb6a`와 해당 source 및 binary의 정확한 hash에 연결된다. 이후 commit은 독립 검토를 마친 동등한 lint rewrite, test extraction, comment-only license header와 fixture/provenance file을 포함한다. 이 보고서는 이전 실행을 이후 source에서 실행한 것처럼 label을 바꾸지 않는다. Range-diff는 exact integration base `f69dd301`에 대한 conflict-free rebase 후 issue commit 6개가 모두 patch-equivalent임을 확인했다. Rebased implementation commit은 `bcf6e2a7`이다.

## 1. 문제 정의

mlxcel에는 video decoding과 공용 DINOv2 backbone이 있었지만 `video_depth_anything` checkpoint를 dispatch하거나 시간적으로 일관된 depth를 생성할 수 없었다. 공개 checkpoint를 지원하려면 frame별 depth head만으로는 부족하다. Preprocessing, temporal motion module, overlap window conditioning, relative-window alignment, metric bypass, padding, interpolation, source-resolution 복원과 output metadata가 모두 upstream offline predictor와 일치해야 한다.

주요 correctness 위험은 window 경계의 temporal drift, 잘못된 preprocessing geometry, checkpoint layout 또는 dtype 변환, 그리고 `--max-frames`가 consecutive prefix를 유지해야 할 때 sparse subset을 sampling하는 동작이다. 긴 clip과 잘못된 configuration에도 checked allocation bound가 필요하다. 각 32-frame window가 backbone, decoder, resized depth와 host alignment state를 유지하기 때문이다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| Model graph | 공용 DINOv2 VDA wrapper, 4-scale DPT decoder, AnimateDiff 스타일 temporal attention과 GEGLU block, feature fusion, depth head를 추가했다. |
| Checkpoint loading | 공개 Small, Base, Large preset과 relative/metric mode, convolution layout sanitization, model 전체에 적용되는 하나의 floating-dtype policy, uniform dtype 검사, linear, normalization, convolution, temporal-head의 정확한 shape 검사를 추가했다. |
| Preprocessing | Upstream의 518-pixel lower-bound 규칙, 14-pixel multiple, wide-aspect adjustment, replicated border를 쓰는 OpenCV cubic sampling, ImageNet normalization, source-size validation을 재현했다. |
| Temporal inference | 정확한 32-frame conditioning, 10-frame overlap, upstream keyframe, 마지막 frame padding과 trim, relative least-squares scale/shift alignment, metric bypass, 8-frame overlap blending을 추가했다. |
| Dense 및 CLI integration | `video_depth_anything`을 image/video `depth` task로 등록하고 `(T,H,W)` shape의 F32 `depth`를 source resolution으로 반환하며, 16-bit frame preview와 metric mode 및 units를 기록한다. |
| Video sampling | 기존 decoded-byte 및 frame limit을 유지하면서 dense temporal predictor가 keep-all/consecutive-prefix policy를 사용할 수 있게 했다. 기본 multimodal sampling 동작은 유지된다. |
| 안전성과 provenance | Source, preprocessing, output, model workset, token, window, dtype, layout, finite-value 검사를 추가했다. 실제 reference 생성은 source, checkpoint, input, decoded frame, native runtime, Python binding, TF32 상태를 pin한다. |
| 문서와 테스트 | 운영 문서, architecture capability, 환경 변수, 집중 unit coverage, 실제 40-frame reference harness, checkpoint 6종 load test, CLI output validation과 인증된 recovery record를 추가했다. |

최종 integration base `f69dd301`부터 현재 head `45e895ab`까지 branch는 tracked file 71개를 변경하며 9,259 line을 추가하고 7 line을 삭제한다. Rebased VDA commit 6개는 다음과 같다.

- `bcf6e2a7` — 구현 및 integration, runtime 검증 `3f01fb6a`와 patch-equivalent
- `a00a9c0e` — 동등한 lint-only rewrite 3개
- `0fa70ed6` — byte-equivalent predictor test 8개를 별도 file로 추출
- `dc670ca3` — comment-only test license header
- `e52a3ee2` — 실제 fixture, provenance와 binary policy update
- `45e895ab` — CPU CLI parity evidence 및 checksum archive

## 3. 기술적 선택과 그 이유

### 3.1 Checkpoint precision을 바꾸지 않고 DINOv2 재사용

VDA는 공용 DINOv2 구현과 선택된 normalized intermediate layer를 사용하지만, backbone 또는 temporal head를 만들기 전에 checkpoint 전체에 하나의 backend dtype policy를 적용한다. Scoped preserve-dtype constructor는 기존 standalone DINO load path가 backbone만 변환하고 VDA head는 다른 dtype으로 남기는 일을 막는다. 이후 모든 loaded tensor는 같은 supported floating dtype을 사용해야 하며 소비되는 모든 layer는 configured shape와 정확히 일치해야 한다.

### 3.2 Upstream offline window 계약 보존

각 model 호출은 32 frame을 받는다. 새로운 window는 22 frame씩 이동하고 upstream keyframe set으로 conditioning하며, 필요한 경우 마지막 source frame을 반복하고 stitching 후 padding을 제거한다. Relative checkpoint는 blending 전에 overlap에 대한 scale과 shift를 fitting하고, metric checkpoint는 metre 단위를 보존하기 위해 alignment를 생략한다. 각 window가 끝날 때 alignment를 수행하므로 predictor는 전체 video의 raw 32-frame output을 모두 보관하지 않는다.

### 3.3 수치 비교 전에 preprocessing 일치

Processor는 upstream lower-bound 및 wide-aspect 규칙으로 patch-aligned target을 계산한다. Replicated edge index를 사용하는 OpenCV 스타일 cubic coefficient를 F32 RGB 값에 적용한 뒤 ImageNet normalization을 수행한다. 따라서 processor 차이가 transformer나 temporal head의 차이로 잘못 해석되지 않는다.

### 3.4 Host 및 model 작업 전체의 live memory 제한

Predictor는 RGB 변환 전에 source dimension을 검사하고 frame 및 processed pixel 수를 제한하며 host output, overlap reference, conditioned frame과 model intermediate에 checked arithmetic을 사용한다. DPT high-resolution refinenet과 output head는 frame 4개 단위로 실행한다. 통합 preflight는 inference 시작 전에 retained DINO feature, projected pyramid level, temporal path, resized head tensor, source-resolution output과 host alignment buffer를 함께 계산한다.

### 3.5 실제 reference evidence 인증

Reference tooling은 MLX를 import하기 전에 revision-pinned source file과 checkpoint를 검증하고 `MLX_ENABLE_TF32`가 `0`이 아니면 거부하며 실제 mapped native library와 Python binding의 정확한 hash를 기록한다. Fixture는 sparse probe 대신 40-frame output 전체를 저장한다. Rust 비교는 정확한 shape, F32 dtype, finite value, checkpoint identity, decoded input identity, processor identity와 device별 RMS threshold를 요구한다.

## 4. 구조와 데이터 흐름

```text
image 또는 consecutive video frame
    -> source/frame/workset validation
    -> OpenCV 호환 resize + ImageNet normalization
    -> 32-frame conditioned window
    -> 공용 DINOv2 selected intermediate feature
    -> 4-scale DPT projection과 temporal motion module
    -> chunked high-resolution depth head
    -> source-resolution bilinear depth
    -> relative overlap alignment 또는 metric bypass
    -> 8-frame blend + padding trim
    -> F32 depth [T,H,W] + metadata + 16-bit preview
```

정지 image는 single frame을 conditioned window에 반복하고 trim된 output frame 하나를 반환하는 동일한 temporal model path를 사용한다. CLI에 `--target-fps` 없이 `--max-frames`를 지정하면 공용 video loader가 요청한 첫 consecutive frame을 유지해 temporal adjacency를 보존한다.

## 5. 보안 및 성능 검토

독립 review finding은 현재 source에서 모두 닫혔다. Loader는 잘못된 model type, 지원하지 않는 encoder preset, 호환되지 않는 temporal dimension, 혼합 또는 non-floating checkpoint dtype, 양수가 아닌 tensor shape, 예상하지 않은 bias와 정확한 layer shape mismatch를 거부한다. Inference는 일치하지 않는 frame dimension, 과도한 source 또는 processed size, arithmetic overflow, 과도한 combined workset, 잘못된 model output 길이와 non-finite model 또는 alignment 값을 거부한다.

이 구현은 작업량을 제한하며 throughput 향상을 주장하지 않는다. 전체 32-frame temporal attention은 의도한 upstream algorithm을 유지하고 high-resolution decoder 작업은 검증된 budget 안에 peak allocation을 유지하도록 chunking한다. 성능 benchmark는 수행하지 않았다.

## 6. 검증

### 6.1 Source와 binary identity

Runtime 검증은 과거 구현 commit `3f01fb6a3d2d20c366a4673c0da3cb3d130961a7`과 source-manifest SHA-256 `fc330220267a3968ee6aacb72a6c8c0716e0109789bd060fb7bdab7f869f389a`에 연결된다. Rebased implementation commit `bcf6e2a7`은 patch-equivalent이다. 최종 library/test binary SHA-256은 `19a3f527055d41e45063c47f493d59e7831879cda6b9b7658ac53ee8995b6d1b`이고 CPU CLI capture는 binary SHA-256 `dcc303d1e3a3c62ec3aa004ce71bd68f2f1c2160983620c8175cfa31ee41809b`를 사용한다.

이후 lint commit의 source-manifest SHA-256은 `7273c98d90b33290d47d75012cb3c2bc0783b01306cbe0918589e39d0de43f51`이다. 정확한 lint diff SHA-256은 `de2c434000682db93b26ebc22b3b4deb91f382fe348d6ce4272d264c73f5aadd`이다. 두 integer `max(1).min(16)` 표현을 `clamp(1,16)`으로 바꾸고 불필요한 shared-slice borrow 하나를 제거했다. 독립 review는 math, control flow, output 변화가 없음을 확인했다. Test extraction은 de-indent 후에도 test body 8개를 byte-equivalent하게 보존했고 license 추가는 comment-only이며 fixture commit은 production Rust model code를 변경하지 않는다. 이전 runtime artifact는 원래 source 및 binary label을 유지한다.

### 6.2 Build, unit, load, lint 및 repository gate

- 최종 library/test build: exit `0`.
- 집중 unit test: target sizing, 40/100-frame window plan, alignment, interpolation, processor 동작, allocation guard와 video prefix sampling을 포함해 14개 통과.
- 실제 checkpoint matrix: Small/Base/Large relative/metric checkpoint 6종 모두 예상 configuration, layout와 F32 backbone, head, normalization tensor를 사용해 순차 load-only 검증 통과.
- 집중 `cargo clippy --lib --tests -- -D warnings`: 21분 43초에 exit `0`; formatting 통과.
- 실제 fixture inventory: decoded 40-frame input, CPU output와 provenance 2종, recovery audit 자료, CPU CLI acceptance receipt와 checksum, generator/downloader, native-runtime provenance, checkpoint matrix와 64-frame clip을 포함한 tracked file 49개.
- Post-rebase contract: formatting, diff, license, binary policy, CPU evidence checksum과 최종 repository contract group 8개가 모두 exit `0`. Contract log SHA-256은 `d287974b158243a1a4a9b6820fd0b002cbc144326699d3b4c07af77e79d2ef59`이다. 정확한 binary-assets 검사는 declared binary 40개가 10,505 KiB budget 중 10,295.8 KiB를 사용한다고 확인했다.
- CI: rebase 전 completed non-skipped check가 모두 통과했다. Final head `45e895ab`의 check는 대기 중이다.

Base 및 Large 증거는 load-only이다. Base 또는 Large forward inference는 주장하지 않는다.

### 6.3 실제 ROCm 40-frame parity

고정된 40-frame DAVIS clip은 하나의 window 경계를 넘어 conditioning, overlap 처리, alignment, blending과 trimming을 실행한다. 두 Small checkpoint는 gfx1151에서 `1e-2` ROCm threshold를 통과했다.

| Checkpoint | Frame | 상대 RMS | Threshold | 결과 |
|---|---:|---:|---:|---|
| Small relative | 40 | `4.912878453e-6` | `1e-2` | 통과 |
| Small metric | 40 | `7.973837467e-7` | `1e-2` | 통과 |

### 6.4 실제 CPU 40-frame parity

두 Small CPU 결과는 독립 검토된 fail-closed comparator SHA-256 `37c3616b4d6ce4087ba14020839fa7fa6f8db2255203d4d84e68f40e21a30386`를 `python -O`에서 실행해 finite F32 `[40,96,128]` tensor 전체를 비교했다. 둘 다 엄격한 `1e-4` threshold를 통과했다.

| Checkpoint | 상대 RMS | Threshold | 결과 |
|---|---:|---:|---|
| Small relative | `5.142513694620207e-6` | `1e-4` | 통과 |
| Small metric | `7.445840126949833e-7` | `1e-4` | 통과 |

Metric upstream oracle은 clean run이다. Python과 wrapper가 exit `0`으로 끝났고 complete two-window log가 `metric_cpu_run/`에 commit되었으며 output 및 provenance hash가 fixture inventory와 일치한다.

Relative upstream oracle에는 문서화된 recovery 경계가 있다. Shell command-not-found 오류 후 wrapper가 `127`로 끝났고 Python exit code는 별도로 capture되지 않았으며 최종 상세 generator log가 손실되었다. 완전한 output과 legacy provenance는 독립 full audit가 hash와 invariant를 인증하고 recovered CPU tensor를 pinned GPU output과 비교한 후에만 수용되었다. CPU와 GPU output은 서로 다르며 CPU-to-GPU 상대 RMS는 `3.4021821538643326e-7`, 최대 절대 차이는 `5.793571472167969e-5`이다. 정확한 review packet은 `relative_cpu_recovery/`에 commit되어 있다. 존재하지 않는 clean completion line 또는 Python exit를 재구성하지 않는다.

두 Rust CPU capture 모두 실제 CLI는 exit `0`, 독립 saved-output validator도 exit `0`이다. 외부 wrapper는 이후 device 표시의 대소문자만 비교하는 assertion 때문에 exit `1`이었다. 이 wrapper는 `finished_unix`를 저장하지 않았으므로 보고서가 해당 시각을 만들어 내지 않는다. Run이 활성 상태일 때 capture script가 변경되어 실행된 pre-edit byte와 SHA를 보존하지 못했다. 인증된 archived script는 실행 증거가 아닌 post-edit snapshot으로 명시되어 있다. 이러한 제한은 parity receipt에 기록되어 있으며 inference를 다시 실행하거나 label을 바꾸지 않았다.

### 6.5 실제 ROCm 64-frame CLI output

두 Small checkpoint 모두 실제 CLI inference를 완료해 `[64,96,128]` shape의 F32 depth tensor, 유효한 16-bit preview PNG 64개와 family, task, input size, frame 수, metric mode, units를 식별하는 metadata를 기록했다. Relative output의 units는 `relative_disparity`이고 metric output은 `metres`이다.

Relative CLI command는 exit `0`이었다. 기존 external wrapper는 이후 runtime device 표시의 대소문자를 잘못 비교한 이유만으로 exit `1`이었다. 독립 수정 validator는 inference를 다시 실행하지 않고 저장된 output을 검증했다. Metric wrapper는 complete log와 함께 exit `0`으로 끝났다.

## 7. 제한 사항

- Base 및 Large relative/metric checkpoint는 실제 load-only coverage를 갖는다. 이 보고서는 Base 또는 Large forward, parity, memory 결과를 주장하지 않는다.
- Metal runtime validation은 수행하지 않았다.
- Streaming, webcam input과 server route는 scope 밖이다.
- Throughput, latency 또는 memory benchmark를 수행하지 않았다.
- Runtime parity는 명시한 provenance 경계 안에서 Small checkpoint의 CPU와 ROCm gfx1151에 대해 입증되었다.
- Final rebased head `45e895ab`의 CI가 아직 대기 중이다.

## 8. 학습 포인트

- Temporal model parity는 window 경계를 넘어야 한다. 32 frame보다 짧은 clip은 conditioning, alignment와 overlap blending을 검증할 수 없다.
- Relative 및 metric checkpoint는 graph를 공유하지만 scale/shift correction이 metric 단위를 손상시키므로 서로 다른 stitching semantics가 필요하다.
- Consecutive prefix 유지는 video-loader preference뿐 아니라 temporal predictor의 model correctness에 해당한다.
- 수치 parity를 감사 가능하게 만들려면 checkpoint provenance가 실제 tensor, preprocessing input, runtime library, device control과 전체 output artifact를 연결해야 한다.
- Recovery evidence를 acceptance에 사용하려면 누락된 사실을 명시적으로 남기고, 독립 검사가 사용할 수 없는 run state를 재구성하지 않은 채 전체 artifact를 인증해야 한다.

## 9. 남은 작업

- [ ] Final rebased head `45e895ab`의 repository CI 완료를 확인한다.

## 참고 자료

- PR #2576: Video Depth Anything dense prediction
- Issue #2352: Video Depth Anything temporal monocular depth
- Epic #2382: dense-prediction model delivery
- Upstream `Blaizzy/mlx-vlm` source revision `abd3b2d7b728cb5372d4e092260da325e5ded209`
- `docs/dense-prediction.md`: VDA 동작 및 output 계약
- `tests/fixtures/video_depth_anything/README.md`: 실제 fixture provenance 및 재생성 절차
