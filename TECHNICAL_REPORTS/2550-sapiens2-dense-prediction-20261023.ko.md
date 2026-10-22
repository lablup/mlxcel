# PR #2550: Sapiens2 pose 및 dense prediction 포팅

**작성일**: 2026-10-23
**상태**: 로컬 완료 (새 PR CI 대기 중)
**언어**: Rust, Python, Markdown
**위험도**: 높음

## 요약

PR #2550은 Sapiens2 human-centric vision family를 mlxcel의 dense-prediction runtime에 포팅한다. 공개된 flat schema와 legacy nested schema checkpoint를 모두 읽고, config 기반 ViT backbone과 다섯 task head를 실행하며, pose, semantic segmentation, surface normal, pointmap, matting, backbone feature를 `mlxcel predict`로 제공한다. Pose는 업스트림의 affine crop, 선택적 flip test, DARK refinement, UDP source-coordinate transform을 따르고, dense task는 공개된 resize와 padding 규칙에 따라 source resolution 출력을 반환한다.

구현에는 config와 pose 입력 제한, checkpoint layout 정규화, 실제 업스트림 fixture, Sapiens2 전용 BF16 유지 정책도 포함된다. 정적 구현 리뷰와 보안·성능 리뷰에서 발견한 문제를 수정한 뒤 열린 major finding은 없었다. 집중 ROCm library 및 real-model integration target compile과 Sapiens2/grid-sample unit filter는 통과했다. 첫 실제 checkpoint 실행에서는 gfx1151 pose parity 차단 문제가 드러났다. Stage 격리로 JPEG decoder 차이와 biased linear, convolution, normalization, kernel-4 deconvolution, activation, packed RoPE의 BF16 연산 그래프 차이를 찾았다. 이 경로들은 이제 고정된 upstream MLX 동작을 따르고 fixture는 canonical lossless PNG를 사용한다. Compiled packed-RoPE, deconvolution-layout, compiled-SiLU 수정 후 capture한 모든 backbone 및 pose-head stage가 bit-exact다. Gfx1151에서 pose flip 비활성·활성, segmentation, normal, pointmap, matting의 실제 checkpoint acceptance test 6개가 모두 통과했다. Rebase 후 CLI build와 end-to-end GPU pose 실행도 통과해 문서화된 artifact를 출력했다. 필요한 clippy target은 warning을 거부하는 설정으로 통과했으며 이 보고서 snapshot에서는 새 PR CI만 대기 중이다.

## 1. 문제 정의

Dense-output 기반은 `sapiens2`를 인식했지만 checkpoint를 load하거나 실행할 수 없었다. Sapiens2는 일반 vision backbone과 단일 decoder의 조합으로도 표현할 수 없다. 공개 family는 1024x768 ViT, register token, checkpoint 기반 2D RoPE period, layer별 MHSA/GQA schedule, QK RMSNorm, LayerScale, SwiGLU, task별 deconvolution 또는 pixel-shuffle head를 사용한다.

공개 checkpoint에는 두 가지 통합 문제가 있다. 초기 segmentation export는 nested backbone/head schema를 쓰고 이후 checkpoint는 flat Transformers-style schema를 쓴다. Convolution과 attention weight도 여러 공개 layout으로 제공될 수 있다. Pose에는 top-down crop과 수치 decode pipeline이 추가되며, coordinate, blur, flip-pair, half-pixel 세부 동작이 업스트림과 일치해야 1.5-pixel 허용 범위를 만족할 수 있다.

이 port가 없으면 사용자는 다섯 가지 공개 human dense-prediction task를 공용 CLI에서 실행하거나 표준 tensor·file 출력을 사용할 수 없다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| Runtime 통합 | Dense family table과 architecture catalog에 `sapiens2`를 등록하고 인식된 checkpoint를 `mlxcel predict`로 연결했다. |
| Configuration | Flat 및 legacy nested config를 canonicalize하고 checkpoint metadata에서 task를 선택하며, 표준 0.4B, 0.8B, 1B, 5B dimension을 지원하고 allocation을 결정하는 모든 field에 제한을 두었다. |
| Backbone | Patch embedding, register token, checkpoint 기반 2D RoPE, layer별 GQA/MHSA, QK RMSNorm, LayerScale, SwiGLU block, final normalization, backbone feature 출력을 추가했다. |
| Head | Deconvolution과 pixel-shuffle decoder, segmentation, normal, pointmap, matting, pose projection, pointmap scale branch를 추가했다. |
| Pose | 제한된 pixel-space box, 업스트림 aspect-ratio crop, affine grid sampling, flip-pair remapping, device-side Gaussian blur, DARK/UDP decode, source-image coordinate 변환을 추가했다. |
| Dense 출력 | Segmentation, normal, pointmap, matting 결과를 source resolution으로 복원하고 issue #2371의 task별 tensor key를 출력한다. |
| Weight loading | Q/K/V 및 convolution layout을 정규화하고 dtype과 shape를 검증하며 checkpoint RoPE period를 읽고 모든 backend에서 공개 Sapiens2 BF16 weight를 유지한다. |
| 검증 asset | 고정 이미지, 실제 업스트림 pose·segmentation reference, checkpoint/source hash, 재현 가능한 reference script, 비교 도구, 집중 unit test, ignore된 실제 checkpoint test를 추가했다. |
| 문서 | CLI option, preprocessing 규칙, tensor shape, task 범위, config 호환성, BF16 예외, 지원 architecture를 문서화했다. |

수치 parity 수정, 집중 regression, 검증된 CLI 경로는 rebase된 구현에 통합됐다. 필요한 local gate는 모두 통과했고 이 보고서 revision에서는 새 PR CI만 남아 있다.

## 3. 기술적 선택과 그 이유

### 3.1 Checkpoint의 BF16 dtype 유지

Sapiens2는 mlxcel의 일반적인 Apple Silicon BF16-to-F16 load 최적화에 대한 명시적 예외다. 공개 pose checkpoint로 실행한 실제 업스트림 비교에서 BF16과 F16 사이의 decoded-keypoint 최대 편차는 1671.8956 pixel이었고, 308개 중 15개가 이슈의 1.5-pixel 범위를 넘었다. 따라서 공용 load policy는 modern 및 legacy Sapiens2 config를 모두 감지해 BF16을 유지한다. Attention 외 floating weight는 load된 attention dtype으로 정규화하고, floating이 아니거나 quantized된 attention weight는 거부한다.

### 3.2 Checkpoint에서 RoPE period 읽기

Backbone은 `backbone.rope_embed.periods`를 읽어 f32로 변환하고, 정확히 `head_dim / 4`개의 유한한 양수인지 검사한다. `rope_theta`에서 fallback을 합성하지 않는다. 이 방식은 공개 weight를 그대로 따르고 malformed 또는 호환되지 않는 checkpoint를 load 단계에서 실패시킨다.

### 3.3 수치 경로를 바꾸지 않고 pose memory 제한

Pose는 최대 16개 box를 받고 순차 처리한다. Flip test를 사용해도 crop batch는 2까지만 늘어난다. Gaussian blur는 MLX device에서 두 번의 matrix pass로 처리하고, DARK는 point마다 maximum과 주변 7개 값만 host로 전송한다. Affine warp는 host coordinate grid를 바로 사용해 불필요한 device round trip을 피하면서 업스트림 half-pixel과 padding 규칙을 유지한다.

### 3.4 공개 schema와 layout 유지

Loader는 초기 nested segmentation config와 현재 flat config, 분리 또는 fused Q/K/V projection, PyTorch 또는 MLX convolution layout을 모두 처리한다. Model size별 network를 hard-code하지 않고 공개 Sapiens2 범위 안에서 config 기반 architecture를 사용한다.

## 4. 구조와 출력 계약

```text
checkpoint config + safetensors
          |
schema canonicalization 및 제한 검증
          |
weight/layout 정규화 -> Sapiens2 ViT backbone
          |
  task head 또는 backbone feature
          |
pose decode / dense resize 및 unpadding
          |
DenseOutput -> result.safetensors + metadata + task preview
```

Predictor는 checkpoint마다 하나의 task를 노출한다. Pose는 `keypoints`, `scores`, `boxes`를 반환하고 `keypoints.json`을 위한 구조화된 entry도 제공한다. Segmentation은 `segmentation`, normal estimation은 `normals`, pointmap estimation은 `pointmaps`와 `scales`, matting은 `alphas`와 `foregrounds`, backbone checkpoint는 `last_hidden_state`와 `pooler_output`을 반환한다. 자동 person detection, video, streaming, server route는 이 이슈의 범위 밖이다.

## 5. 검증과 제한 사항

| 확인 항목 | 보고서 작성 시점의 결과 |
|---|---|
| 정적 구현 리뷰 | 완료. BF16 load 정규화, f32 flip averaging, checkpoint RoPE 검증 문제를 수정했고 열린 major correctness finding은 없다. |
| 보안·성능 리뷰 | 완료. Allocation을 결정하는 config field와 pose box에 제한을 두었고, pose blur와 warp에서 기존의 대형 host copy 및 scalar loop 경로를 제거했다. 열린 major finding은 없다. |
| Format과 가벼운 저장소 확인 | `cargo fmt --all`, commit된 parity script의 Python AST parse, `git diff --check`가 성공한 것으로 보고되었다. 이 finalizer도 `git diff --check`를 독립적으로 확인했다. |
| Fixture 무결성 | 원본 JPEG의 SHA-256은 `16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb`이다. Pillow로 decode한 RGB byte를 canonical PNG에 lossless로 보존했으며 SHA-256은 `3342393fccc820a687ec932d9e1b799e74146c0d9319397775169ae5b5f98a69`이다. 이 PNG는 업스트림 reference RGB와 pose crop을 정확히 재현한다. Source provenance는 mlx-vlm revision `abd3b2d7b728cb5372d4e092260da325e5ded209`를 고정한다. CPU baseline은 MLX 0.32.4를 사용한다. Primary same-backend reference는 gfx1151의 MLX 0.32.5를 사용하며 Rust native archive와 새로 build한 Python binding 모두 `bff10220c742907f0c68e55ff174aaea56f74e3d`에 고정한다. Native library SHA-256은 `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094`, binding SHA-256은 `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc`이다. Rust archive copy의 차이는 ELF visibility에 한정되며 executable section hash는 동일하다. 이전 MLX `0.32.3.dev20260912+a7d4c85a` ROCm 출력과 CPU 출력은 진단 이력으로 보존한다. 다섯 validation model의 checkpoint hash도 기록한다. |
| 실제 업스트림 reference 생성 | CPU와 exact-pin target ROCm backend에서 다섯 task 모두 완료됐다. 고정된 CPU baseline은 pose flip 비활성 537.7초, flip 활성 1118.7초, segmentation 528.7초, normal 781.8초, pointmap 1524.649930초, matting 1916.786644초가 걸렸다. Primary exact-pin gfx1151 reference는 각각 3.913447초, 3.648052초, 1.324742초, 2.306536초, 4.680237초, 4.635040초에 exit 0으로 완료되어 유한한 source-resolution 출력을 반환했다. Primary SHA-256은 pose flip 비활성 `4b6cf998528f0fa1e03ed87e63ae384d76b0fa31c55da0ca6feb7dc7e4416a0d`, flip 활성 `1633beba837f8bccc070b28c33792a34a6696326729acaf7b2a55566d1846ccb`, segmentation `cd6f17a6182b6d5ab738c0e0dc413c98a6e883187faf719191ea37575f932f18`, normal `ab9fdd20f1d0ebcd92b54f8486fc00756b18405497a1a9015402eb7d1559ce45`, pointmap `f39deabf0e6141d133de8cbbaf45436bc9f4b067fa20d34058734287f9a7ed14`, matting `e6677044316294248f768102ed976091b57ad6fdad48744f2aab8616cb21cb7c`이다. 모두 업스트림 fixture 실행이며 Rust parity pass가 아니다. 시간은 공유 GPU guard 아래의 진단 정보이며 benchmark가 아니다. |
| 집중 Rust compile | Canonical input과 BF16 연산 그래프 수정이 포함된 대체 source가 8분35초 만에 compile을 통과했다. 이어서 device-side RoPE graph와 stage dump가 포함된 focused no-run rebuild가 8분46초, exact-pin first-block 진단 build가 8분48초 만에 exit 0으로 완료됐다. Compiled packed-RoPE bridge를 포함한 library build는 9분21초, comprehensive all-stage diagnostic build는 8분34초 만에 exit 0으로 완료됐다. Deconvolution-layout 대체 build는 8분29초 만에 exit 0으로 완료됐다. Compiled-SiLU 대체 build는 8분22초 만에 exit 0으로 완료되어 `target/test-fast/deps/mlxcel-d5e762440845331a`를 생성했다. `c6ca6c48`에 rebase한 뒤 latest real-model integration target은 7분35초 만에 exit 0으로 compile되어 `target/test-fast/deps/sapiens2_real_model-a115aa67e5945075`를 생성했다. |
| 집중 unit test | 현재 library binary는 공유 guard 아래 gfx1151에서 통과했다. `sapiens2` filter는 26 passed, 0 failed, 1 ignored를 기록했으며 exact-pin compiled-RoPE oracle와 compact gfx1151 deconvolution-layout 및 compiled-SiLU-layout oracle를 포함한다. 이전 `grid_sample` filter는 4 passed, 0 failed를 기록했다. Config와 head 동작, pose 수학, 공용 grid sampling, BF16 sanitizer, fused linear bias, affine-free InstanceNorm, kernel-4 deconvolution, packed RoPE, graph layout의 실제 crafted BF16 regression과 exact-pin staged 진단을 포함한다. |
| 실제 Rust checkpoint parity | **현재 exact-pin same-backend acceptance test 6개가 모두 통과한다.** 보존한 Rust stage dump는 exact-pin upstream input을 bit 단위로 재현한다. 24개 backbone block 각각의 16개 captured stage, final normalization, head feature, 모든 deconvolution 및 convolution raw/normalized/activated stage, 최종 heatmap이 bit-exact다. 명시적으로 실행한 staged test는 19.59초 만에 통과했고 stage dump SHA-256은 `52c1d87e971d188b60d0ae7716972d11211d62c073d0b0885403a4001d7ec7b2`이다. Rebase 후 real-model suite는 serial test 6개를 32.64초 만에 모두 통과했다. Filter하지 않은 pose point 308개의 최대 오차는 flip 비활성 `0.001930` pixel, flip 활성 `0.010028` pixel이고 score는 정확히 일치했다. Segmentation label agreement는 `0.999723307`, normal relative RMS는 `0.000000046`, pointmap relative RMS는 `0.000000037`이었다. 1B matting checkpoint는 유한한 alpha `[720,1280]` 및 foreground `[720,1280,3]` tensor를 반환했다. Matting은 numerical parity가 아닌 유한성·shape smoke test다. 이전 MLX `a7d4c85a` 비교는 Python runtime이 Rust native pin과 달랐으므로 진단 결과다. 0.8B와 5B architecture는 아직 실행하지 않았다. |
| CLI end to end | Rebase 후 `mlxcel` CLI build가 exit 0으로 완료됐다. 공유 guard 아래 gfx1151 pose 실행도 성공했고 `result.safetensors`, `keypoints.json`, `meta.json`을 출력했다. 구조화된 출력에는 person 1명, 값 4개의 box 1개, `[x,y,score]` point 308개, name 308개가 포함됐다. Exact-pin reference와 독립적으로 비교한 최대 keypoint 오차는 `0.001930` pixel이었다. Build 및 smoke 시간은 진단 정보이며 benchmark가 아니다. |
| Clippy, CI와 platform | Library, `mlxcel` binary, real-model integration test를 대상으로 한 clippy는 warning을 거부하는 설정으로 15분33초 만에 통과했다. 이어서 이전에 실패했던 cfg(test) module을 포함하는 필수 `--lib --tests` 범위도 warm cache에서 1분37초 만에 warning 없이 통과했다. 이전 `f809d137` CI 실패는 이 수정 전의 결과이며 새 PR CI는 대기 중이다. Metal 검증은 명시적인 platform 제한 사항이다. |

Tracked repository는 이제 exact-pin primary 출력을 stable pose flip 비활성·활성 및 segmentation fixture name으로 사용하며, 이전 `a7d4c85a` ROCm 출력은 진단 이력을 위해 명시적인 `a7d` name으로 보존한다. Output hash manifest는 크기 때문에 외부에 유지하는 complete exact-pin normal, pointmap, matting artifact도 기록한다. 공개 0.4B checkpoint와 1B matting checkpoint는 runtime 검증에 사용할 수 있다. 0.8B와 5B config URL은 작업 중 접근할 수 없었으므로 제한된 config 기반 architecture로 지원하지만 여기서는 runtime 검증을 주장하지 않는다.

`.artifacts/sapiens2` 아래의 다섯 실제 checkpoint는 약 0.8 GB에서 3.1 GB 크기의 local validation 입력이다. 의도적으로 untracked 상태이며 stage하거나 commit하면 안 된다.

Dtype audit에서는 의도하지 않은 F16 경로를 찾지 못했다. 공개 pose safetensors header의 328개 tensor는 모두 BF16이고, modern config는 family retention gate에 맞으며, 정규화는 attention dtype을 유지하고, backbone은 pixel을 BF16 patch weight dtype으로 변환하므로 Q/K/V도 BF16으로 SDPA에 들어간다. Canonical PNG는 JPEG decoder 변수를 제거하며 대체 Rust input은 CPU 및 ROCm upstream input과 bit 단위로 일치한다. CPU stage 비교는 여전히 relative RMS `0.018312981`, 최대 절대 오차 `0.029296875`를 보이고, upstream CPU와 이전 ROCm runtime 자체도 decoded pixel 기준 최대 391.849487만큼 달라 backend 의존성이 큼을 확인했다. 따라서 acceptance는 원래 threshold를 유지하면서 동일 gfx1151 backend의 exact-pin upstream 구현을 기준으로 한다. Exact-pin upstream stage capture SHA-256은 `4b33a455525ee3ef3e791b15cfeeb755a71302319a45ea61e3b3c12c8ec05735`이고 first-block capture SHA-256은 `ee3ea395e49d24dc40d3135731100e1389bf6e12d50de89a215bb54b132bf348`이다. Shape-specialized compiled bridge는 업스트림 packed rotation graph를 재현하고 strict two-value BF16 oracle가 통과했으며 capture한 backbone tensor 전체가 bit-exact다. 정확한 take/stack/transpose/reshape 준비로 deconvolution stage 전체도 bit-exact가 되었고 deterministic gfx1151 oracle는 수정된 `21.375`와 기존 `21.5`를 구분한다. Direct exact-native 실험에서는 upstream compiled SiLU와 eager SiLU의 BF16 값이 동일하지만 뒤따르는 addmm 결과가 다르며 eager layout이 기존 Rust stage 오차를 정확히 재현했다. 기존 upstream-equivalent shapeless compiled SiLU는 이제 capture한 모든 pose-head tensor를 bit-exact하게 만들고 엄격한 flip 비활성 pose output도 통과한다. Stage 근거 없는 attention 대체나 tolerance 완화는 수용하지 않았다.

## 6. 해결한 리뷰 항목

- 공용 BF16 sanitizer가 원래 Sapiens2 weight를 F16으로 변환했다. 이제 두 config schema에서 family를 감지하고 end-to-end BF16을 유지한다.
- Flip-test averaging이 혼합된 저정밀 dtype을 물려받을 수 있었다. 이제 두 heatmap branch를 각각 f32로 변환한 뒤 평균한다.
- RoPE 생성이 합성 fallback을 허용했다. 이제 checkpoint period를 필수로 읽고 검증해 사용한다.
- 초기 DARK 구현은 전체 heatmap을 host로 복사하고 수억 번의 scalar 연산을 수행했다. 이제 Gaussian blur를 MLX matrix pass로 실행하고 local decode 값만 전송한다.
- Pose box와 allocation을 결정하는 config dimension에 공개 family 기준의 명시적 제한을 추가했고, legacy segmentation schema에는 원본 그대로의 regression fixture를 추가했다.
- Parity 이미지는 Pillow의 정확한 decoded RGB byte로 만든 lossless PNG를 사용해 reference pixel을 바꾸지 않으면서 JPEG decoder별 rounding 차이를 제거한다.
- Bias가 있는 Sapiens2 linear layer는 이제 MLX `addmm`를 사용한다. 따라서 matmul을 먼저 반올림하고 별도로 bias를 더하지 않고 `nn.Linear`와 같은 단일 BF16 bias epilogue를 실행한다.
- Head 1x1 projection은 flatten한 `addmm`를 사용하고, 조건에 맞는 3x3 convolution은 업스트림의 image당 96 Mi-element im2col 기준과 `addmm`를 사용한다. Affine-free InstanceNorm은 `fast.layer_norm`을 사용한다.
- Kernel-4 transposed convolution은 이제 업스트림의 고정 phase/tap 순서와 단일 GEMM을 사용한다. Kernel-2와 kernel-3 checkpoint는 native transposed-convolution 경로를 유지한다.
- RoPE coordinate, angle, sine/cosine table, packed sign, paired-component rotation은 host-generated table 대신 업스트림의 device-side MLX 연산 그래프를 사용한다.
- BF16 packed RoPE는 검증된 shape-specialized compiled bridge로 실행되어 업스트림 compiled negative-stride graph와 일치하고 block 0 전체를 bit-exact로 만든다.
- Kernel-4 deconvolution weight 준비는 이제 업스트림 take/stack/transpose/reshape graph layout을 유지한다. Weight 값은 이미 동일했지만 기존 concatenation layout은 대형 ROCm GEMM에서 다른 BF16 누산 경로를 선택했다.
- Sapiens2 activation은 이제 업스트림 `nn.SiLU`와 같은 기존 shapeless compiled SiLU를 사용한다. 기존 eager graph는 동일한 activation 값을 반환했지만 다음 대형 BF16 addmm가 보는 layout을 바꿨다.

## 7. 후속 조치

- Rebase source에서 새 PR CI를 완료한다.
- 통합된 exact-pin pose·segmentation fixture, hash, provenance를 stable primary fixture name으로 유지한다.
- 0.8B, 5B, Metal 실행 공백은 일치하는 checkpoint와 hardware를 사용할 수 있을 때까지 명시적인 검증 제한으로 유지한다.
- `.artifacts/sapiens2`를 version control 밖에 유지한다.

## 참고 자료

- PR #2550: `feat: port Sapiens2 dense prediction`
- Issue #2371: Sapiens2 human pose 및 dense human prediction
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: CLI, preprocessing 및 output 계약
- `tests/fixtures/sapiens2/README.md`: fixture 재현 및 비교 절차
