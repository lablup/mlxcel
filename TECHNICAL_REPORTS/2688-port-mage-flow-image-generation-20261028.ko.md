# PR #2688: Mage-Flow 이미지 생성 및 편집 포팅

**날짜**: 2026-10-28
**상태**: CUDA(GB10)에서 완료, Metal은 Partial 표기(실행 안 함), ROCm 실행 안 함
**언어**: Rust, Python(픽스처 생성기), Markdown
**위험도**: 낮음(새 패밀리이며 공유 코드 변경은 추가형)

## 요약

PR #2688(이슈 #2426, 에픽 #2348)은 mlx-vlm `models/mage_flow`(git 73d366ec)를 `mage_flow` 이미지 패밀리로 포팅합니다. 여섯 변형 모두(텍스트-이미지용 Base, aligned, Turbo와 다중 참조 편집용 Edit-Base, Edit, Edit-Turbo), 기존 트리 내 VLM 로더를 통한 Qwen3-VL 컨디셔너, affine, mxfp4, mxfp8, nvfp4 양자화 프로젝션을 지원하는 12 블록 dual-stream 트랜스포머, 그리고 새로운 Mage-VAE 인코더와 디코더를 포함합니다. 패밀리는 `mlxcel generate-image`, `/v1/images/generations`, `/v1/images/edits`(모든 참조 이미지를 한 번의 컨디셔닝 패스로 처리), probe, 모델 레지스트리(여섯 변형 이름 모두 별칭으로 등록), `mlxcel arch --json`에 연결됩니다.

512x512, 4 스텝, seed 42에서 업스트림 대비 Turbo는 PNG PSNR 38.4 dB입니다. 참조 이미지 두 장의 Edit-Turbo는 우리 컨디셔닝으로 21.8 dB, 업스트림 컨디셔닝으로 39.9 dB입니다. 사용자는 Edit-Turbo를 에픽의 30 dB 기준에 대한 예외로 승인했고, 이슈의 컨디셔너 기준을 다른 기준으로 교체했습니다. 포팅 과정에서 MLX CUDA LayerNorm 커널의 레이스도 발견했으며 이슈 #2696으로 등록되었습니다. Mage-Flow는 LayerNorm을 f32 연산으로 조합해 이를 피합니다.

## 1. 문제 정의

Mage-Flow(Microsoft, 4B, `mage-flow-community/` 아래 MIT 체크포인트)는 네이티브 해상도 텍스트-이미지 생성과 다중 참조 편집 패밀리입니다. 구성은 Qwen3-VL 컨디셔너, NR-MMDiT 트랜스포머(double-stream 블록 12개, hidden 3072, 24 heads, `axes_dim [16, 56, 56]`의 multi-scale RoPE), 자체 Mage-VAE(`DConvEncoder`, 32 픽셀 local attention을 쓰는 `CoDDecoder` 기반 denoiser, latent 채널 128, downsample 16), 그리고 static-shift flow-match 샘플러(기본 shift 6.0)입니다. #524는 이를 범위 밖으로 기록했었습니다. #2348이 이미지 생성을 범위에 넣었고, #2341 런타임이 머지된 뒤에도 Mage-Flow 체크포인트는 아무것도 인식하지 못해 로드 단계에서 실패했습니다.

트리 내 대응물이 있던 것은 컨디셔너뿐입니다. `Qwen3VLModel::forward_hidden`이 final norm을 거친 hidden state를 반환하며, Mage-Flow가 쓰는 것이 바로 이 값입니다. 이슈는 디코더를 복사하지 말고 이것을 재사용할 것, 인코딩 후 컨디셔너를 해제할 것(8B 컨디셔너와 4B 트랜스포머는 bf16에서 24 GB를 넘음), 그리고 mlxcel에 대응물이 없는 트랜스포머와 VAE를 포팅할 것을 요구했습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 패밀리 모듈 | 새 `src/imagegen/mage_flow/`(`config`, `conditioner`, `transformer`, `layers`, `pipeline`, `weights`, `mod`, `vae/{blocks,encoder,decoder,mod}`)가 `ImageGenerationModel`을 구현합니다. 생성 변형은 Generate만, 편집 변형은 Edit만 지원합니다. |
| 컨디셔너 | `conditioner.rs`는 기존 로더로 `<model>/text_encoder/`에서 Qwen3-VL을 로드하고, 업스트림의 `GENERATION_TEMPLATE` / `EDIT_TEMPLATE`을 적용한 뒤 앞쪽 34개(생성) 또는 64개(편집) 토큰을 버립니다. 편집은 참조마다 `Image N: <|vision_start|><|image_pad|><|vision_end|>`를 넣고 기존 Qwen3-VL 프로세서와 vision tower를 실행합니다. `qwen3_vl.rs`는 건드리지 않았습니다. |
| 트랜스포머 | `transformer.rs`: dual-stream 블록, msrope position id, `qwen_proj` 시간 임베딩. `.scales`가 있으면 양자화 linear 경로로 프로젝션합니다. |
| VAE | `vae/`: 인코더와 디코더 포팅. FLUX.2 VAE의 `Conv2d`(depthwise conv용 `groups` 인자 추가), `GroupNorm`(활성화 dtype으로 계산하는 `forward_native` 추가), conv layout probe를 재사용합니다. |
| 샘플링 | 이슈의 스케줄러 계약대로 공유 `FlowMatchEulerDiscrete::static_shift`를 씁니다. CFG는 guidance > 1이고 negative prompt가 있을 때만 적용하며, `renormalization`은 선택 사항입니다. |
| 공유 코드(추가형) | 작업 오류 메시지에 변형 이름을 넣기 위한 `ImageGenerationModel::variant`(기본값 `None`). 양자화 텐서 검증을 Z-Image에서 `src/imagegen/quant.rs`로 옮기면서 오류 문구는 유지하고 `QuantSpec`은 `z_image::weights`에서 재export합니다. `imagegen/vae`의 가시성 변경. 공유 VAE 가중치 오류 문구가 `FLUX.2 VAE weight not found`에서 `VAE weight not found`로 바뀌었습니다. |
| 등록과 문서 | `imagegen::FAMILIES`, `probe.rs`(`MageFlowPipeline`), `models/registry.rs`, 서버 이미지 모델과 워커, `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md`. `MLXCEL_MAGE_FLOW_ORACLE_DIR`를 `scripts/ci/check_env_var_docs.py`에 허용 목록으로 추가하고 픽스처를 `scripts/ci/check_binary_assets.py`에 선언했습니다. |
| 픽스처 | `tests/fixtures/imagegen/mage_flow_parity.safetensors`, `mage_flow_tiny_refs.safetensors`, 생성 스크립트 `tests/fixtures/generate_mage_flow_fixtures.py`. |

`c4468af0` 기준 통계: 파일 40개, +6107 / -125. 커밋: `721e2a06`(포팅), `6453e542`(문서와 픽스처 선언), `c19bad02`(결정적인 f32 컨디셔너 기준값), `2fe3da6a`(컨디셔너 입력 상한), `98653b52`(vision marker), `91e2a2a6`(DiT 길이 상한과 명시적 크기), `dd5b2cb9`와 `e1efc517`(`origin/main` 머지), `c4468af0`(vision model scope 문서).

## 3. 기술적 선택과 그 이유

### 3.1 parity 기준과 Edit-Turbo 예외(사용자 결정, 에픽 #2348)

이슈는 처음에 최종 latent의 상대 RMS 1e-2 미만을 요구했습니다. 에픽 #2348 진행 중 사용자는 이를 에픽 전체 기준으로 바꿨습니다. 잘못된 포팅을 걸러내는 케이스별 latent 상한, CUDA 오라클 대비 PNG PSNR 30 dB 이상, 비트 단위로 동일한 초기 노이즈입니다. 측정한 rel-RMS는 drift로 기록할 뿐 머지 조건이 아닙니다. 결과는 이슈 #2426의 2026-10-28 acceptance revision에 기록되어 있습니다.

오라클: mlx 0.32.4 위의 mlx-vlm 0.7.8(git 73d366ec), CUDA 백엔드, NVIDIA GB10, `MLX_ENABLE_TF32=0`, 512x512, 4 스텝, seed 42. 두 케이스 모두 초기 노이즈가 비트 단위로 동일합니다.

| 케이스 | 최종 latent rel-RMS(drift) | PNG PSNR | 업스트림 컨디셔닝 사용 시 |
|---|---|---|---|
| Turbo | 6.6e-2 | 38.4 dB | 3.4e-2, 42.3 dB |
| Edit-Turbo, 참조 두 장 | 2.5e-1 | 21.8 dB | 5.1e-2, 39.9 dB |

커밋된 상한:

| 케이스 | pooled latent 상한(측정값) | PNG 상한 |
|---|---|---|
| Turbo | 8e-2(4.75e-2) | 30 dB |
| Edit-Turbo, 자체 컨디셔닝 | 3.5e-1(2.13e-1) | 18 dB |
| Edit-Turbo, 업스트림 컨디셔닝 | | 30 dB(`ORACLE_COND_PNG_PSNR`) |

Turbo는 에픽 기준을 통과합니다. 자체 컨디셔닝의 Edit-Turbo는 통과하지 못하며, 사용자는 이를 자체 컨디셔닝 18 dB, 업스트림 컨디셔닝 30 dB 상한의 예외로 승인했습니다. 근거는 다음과 같습니다.

- 업스트림 컨디셔닝을 우리 트랜스포머와 VAE에 넣으면 편집이 39.9 dB에 도달합니다. 따라서 트랜스포머, VAE, 샘플러는 정상이고 차이는 bf16 Qwen3-VL 컨디셔너에 있습니다.
- 업스트림도 자기 자신에 대해 그만큼 움직입니다. 컨디셔너만 f32로 돌리면 업스트림은 자신의 bf16 PNG 대비 22.9 dB입니다. 21.8 dB는 참조 두 장 편집에서 이 컨디셔너의 bf16 정밀도 하한이지 포팅 오류가 아닙니다.

18 dB 상한은 크게 망가진 파이프라인만 걸러냅니다. 컨디셔너의 잘못된 포팅을 잡는 장치가 아닙니다. 그 역할은 3.2의 컨디셔너 기준과, 컨디셔너 이후 단계의 오류를 잡는 업스트림 컨디셔닝 30 dB 기준이 맡습니다.

### 3.2 컨디셔너 기준(사용자 결정, 에픽 #2348)

이슈는 컨디셔너 hidden state가 업스트림 대비 상대 RMS 1e-3 이내일 것을 요구했습니다. 이 값은 bf16 커널 하한보다 낮습니다. 같은 입력에서 Qwen3-VL 한 레이어만으로도 mlx 0.32.4와 5.7e-3 차이가 납니다(업스트림 RoPE를 흉내 내면 5.6e-3). 업스트림 자신의 bf16 hidden state도 같은 모델을 f32로 넓힌 값과 1.8e-2(생성), 2.6e-1(편집) 떨어져 있습니다.

사용자는 기준을 완화하지 않고 교체했습니다. 새 기준은 다음을 요구합니다.

- 토큰 id, 이미지 grid, M-RoPE 위치가 업스트림과 정확히 일치;
- 우리 bf16 hidden state와 f32 기준값 사이 거리가, 업스트림 bf16 state와 같은 기준값 사이 거리의 1.2배 이하(`conditioner_tests.rs`의 `F32_DISTANCE_RATIO`).

측정 비율은 0.96(생성, 우리 1.7e-2 대 업스트림 1.8e-2)과 1.04(편집, 2.7e-1 대 2.6e-1)입니다. 같은 픽셀에서 나온 vision 행은 6.9e-2 차이로, 업스트림의 6.4e-2와 비슷합니다. 커밋 `c19bad02`가 f32 기준값을 결정적으로 만들었습니다(3.3 참고).

구현 리뷰가 HIGH로 표시하고 메인테이너 결정으로 남긴 단서가 있습니다. 이 기준은 64 채널 pooled 평균(`pool64`)을 비교하므로, 한 채널 그룹 안에 갇힌 오류는 드러나지 않습니다. 18 dB 종단 간 편집 상한도 이를 잡지 못합니다. 우리 pooled 편집 state는 업스트림 bf16 실행과 2.7e-1 떨어져 있습니다. 업스트림도 컨디셔너를 f32로 돌리면 스스로 2.8e-1 움직이므로 정밀도 혼돈과 일치하지만, 같은 크기의 구조적 오류도 똑같이 보일 것입니다. 이를 확정하려면 f32 대 f32 비교가 필요하고, 그러려면 mlxcel에 없는 f32 Qwen3-VL 로드 경로가 필요합니다. 사용자는 이 공백을 기록한 채로 기준을 승인했습니다.

### 3.3 MLX CUDA LayerNorm 레이스(이슈 #2696)

동일한 Edit-Turbo 실행 두 번의 PNG가 서로 29.8에서 35.5 dB밖에 일치하지 않았고, 이는 업스트림 mlx-vlm과 mlxcel 모두에서 같았습니다. 원인은 MLX의 `backend/cuda/layer_norm.cu`에 있습니다. 커널은 공유 `temp` 버퍼 하나를 선언하고 `BlockBroadcastReduce::Sum`을 평균과 분산에 한 번씩, 두 번 호출합니다. `Reduce`는 `temp` 쓰기와 읽기 사이에 `block.sync()`가 있지만 읽은 뒤에는 없습니다. 빠른 warp가 느린 warp가 아직 평균 부분합을 읽지 않은 `temp[warp]` 슬롯에 분산 부분합을 써버릴 수 있습니다. 384 폭 bf16 행(Mage-VAE 인코더의 `norm_out`, 입력 `[2, 32, 32, 384]`)에서는 호출마다 몇몇 행이 틀리며, mlx 0.32.4에서 30번 호출 중 30번이 서로 달랐습니다. 편집 경로의 참조 latent는 동일한 호출 사이에 약 1e-1 움직였습니다.

이 PR은 패밀리 안에서만 우회합니다.

- `layers::LayerNorm::forward`는 LayerNorm을 개별 연산으로 조합합니다. f32로 캐스트, 평균, 중심화 분산, rsqrt, f32 affine, 마지막에 한 번 캐스트. 결정적이며 fused 커널의 정상 행과 bf16 ulp 하나 이내로 일치합니다. ulp 차이는 fused 커널이 `w * norm + b` 전에 정규화 값을 활성화 dtype으로 반올림하기 때문이며, 리뷰 중 doc comment도 이렇게 바로잡았습니다.
- 회귀 테스트는 `vae_tests.rs::layer_norm_is_deterministic_on_384_wide_bf16_rows`입니다.
- 픽스처 생성기도 업스트림 트랜스포머와 VAE에 같은 조합을 적용해 오라클을 재현 가능하게 만들었습니다. 두 번 재생성해 바이트 단위로 동일했습니다.

여전히 노출된 부분: CoD 디코더의 GroupNorm(`GroupNorm::forward_native`)은 업스트림과 마찬가지로 fused `fast_layer_norm`을 호출하며, 생성기도 업스트림 GroupNorm을 fused로 둡니다. 이 경로가 결정적인지는 테스트하지 않았습니다. mlxcel의 다른 모든 CUDA `fast_layer_norm` 사용처도 노출되어 있고(`mlxcel_core::layers::LayerNorm`, Cohere, StableLM, Nemotron, 오디오 인코더, `dit_blocks`, FLUX.2 임베딩, FLUX.2 VAE GroupNorm, dense vision tower), `layer_norm_vjp`에도 같은 위험이 있습니다. #2696은 `temp`를 읽은 뒤 `block.sync()`를 하나 더 넣는 `layer_norm.cu` CUDA overlay를 제안하며, 이것으로 forward와 VJP가 모두 고쳐집니다. 이 파일은 `ml-explore/mlx` main과 바이트 단위로 같으므로 pin을 올려도 해결되지 않습니다.

### 3.4 리뷰 중 추가한 요청 상한

이 상한들은 mlxcel 자체의 것이며 업스트림에는 없습니다. 출발점은 공유 `Qwen3VLModel::get_input_embeddings`가 vision tower를 실패를 반환하지 않는 `mlxcel_core::eval`로 실행한다는 점입니다. 그래서 그 안에서 MLX가 실패하면 500을 반환하지 않고 서버가 중단됩니다. 수정 전에는 `vl_cond_long_edge`가 `null`, 0 이하, 8192까지의 값을 받았고, 체크포인트의 4096x4096 프로세서 상한과 참조 16장을 합치면 Qwen3-VL 토큰이 약 262k까지 가능했습니다.

- `2fe3da6a`: `vl_cond_long_edge` 최대 2048(`MAX_VL_COND_LONG_EDGE`). 참조별 프로세서 픽셀 상한 2048x2048(`MAX_VL_COND_PIXELS`, 참조당 visual 토큰 최대 4096). 2048 토큰을 넘는 편집 지시문은 400. 200:1 종횡비를 넘는 참조는 500 대신 400. 기본값은 바뀌지 않았습니다.
- `98653b52`: Qwen3-VL vision marker(`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, `<|video_pad|>`)를 문자 그대로 포함한 편집 프롬프트는 400. 이전에는 남는 `<|image_pad|>`가 placeholder 확장에서 500으로 실패했고, 남는 `<|video_pad|>`는 그대로 통과해 공유 merge에서 이미지 특징을 위치와 어긋나게 밀어냈습니다. `masked_scatter`가 인덱스를 wrap하므로 오류 없이 잘못된 출력이 나왔습니다.
- `91e2a2a6`: 전체 DiT 이미지 토큰 `(1 + 참조 수) * (H/16) * (W/16)`을 65536으로 제한(`MAX_DIT_IMAGE_TOKENS`, 참조 세 장의 2048x2048 편집까지 가능). 넘으면 개수와 상한을 명시한 400입니다. 이전에는 2048x2048에 참조 16장이면 forward당 joint 토큰이 약 278k였고 CFG에서 두 배였습니다. chunked attention과 스텝별 `try_eval`로 메모리는 제한되었지만 요청 하나가 이미지 워커를 몇 시간 붙잡을 수 있었습니다. 같은 커밋에서 명시적 편집 크기가 `max_size`보다 우선하도록 바꿨고, `max_size`는 크기가 첫 참조에서 유도된 기본값일 때만 적용됩니다.

기타 요청 규칙: 변당 512..=2048, 16의 배수, 종횡비 4:1까지. 생성 변형에 편집을 요청하거나 그 반대면 변형 이름을 담은 400입니다. 알려진 차이: 공유 16 MP 상한을 넘는 참조는 `max_size`가 출력을 줄이기 전에 공유 `ImageRequest::validate`에서 거부됩니다. 결정된 크기는 항상 `validate_dimensions`를 거치므로 안전한 쪽으로 실패하며, 문서에 기록되어 있습니다.

### 3.5 작은 결정들

- **변형 판별.** 업스트림의 전체 경로 매칭 대신 `mage`를 포함한 가장 가까운 경로 구성 요소에서 변형을 정합니다. 그래서 `/data/database/` 아래 모델이 `-base`로 잡히지 않습니다.
- **`sample_posterior`** 기본값은 업스트림 파이프라인과 같은 true입니다(이슈 본문은 false라고 했지만 코드는 파이프라인을 따릅니다).
- **VAE 기준.** 단계별로 업스트림 입력에서 5e-3을 적용하고, 마지막 denoiser 블록만 2e-2(측정 9.8e-3)입니다. 오라클의 CUDA `mx.mean`이 bf16을 bf16으로 누적하기 때문이며, #932의 선례를 따릅니다. 종단 간 VAE 상한은 5e-2(측정 decode 1.0e-2, encode 2.4e-2)입니다.
- **정밀도.** 가중치는 업스트림처럼 bf16으로 유지합니다. 정상 상태 메모리는 bf16 가중치 약 16.5 GB입니다.
- **메모리.** 첫 요청에서는 이슈 요구대로 인코딩 후 컨디셔너를 해제합니다. 이후 요청에서는 트랜스포머와 VAE가 상주한 상태로 컨디셔너를 디스크에서 다시 로드하므로, 정상 상태 최대치는 세 구성 요소 전부입니다. 리뷰 중 모듈 문서를 이에 맞게 고쳤습니다.

### 3.6 ERNIE-Image(#2687)와의 머지

이 PR이 열려 있는 동안 ERNIE-Image가 `90d6409a`로 main에 머지되었고, `e1efc517`이 이를 가져왔습니다. 양쪽에서 모두 바뀐 파일은 아홉 개입니다. `docs/image-generation.md`, `docs/supported-models.md`, CI 스크립트 두 개, `src/imagegen/mod.rs`, `src/imagegen/registry.rs`, `src/imagegen/vae/model.rs`, `src/imagegen/z_image/weights.rs`, `src/models/registry.rs`. 레지스트리 표와 `FAMILIES`, `IMAGE_FAMILIES`에는 이제 `ernie_image`와 `mage_flow`가 함께 있습니다.

교차 의존은 하나입니다. `ernie_image/weights.rs`는 `z_image::weights::index_metadata`를 import하는데, 이 PR은 양자화 텐서 검증을 `z_image::weights`에서 `imagegen::quant`로 옮겼습니다. `index_metadata`는 `z_image::weights`에 남았고 `QuantSpec`도 그곳에서 재export되므로 ERNIE의 import는 수정 없이 컴파일됩니다. `index_metadata`와 `Flux2Vae::bn_normalize`에 ERNIE가 추가한 `// Used by:` 주석도 그대로 유지되었습니다.

머지 후 `docs/vision-model-scope.md`는 ERNIE가 같은 절을 다시 썼기 때문에 `mage_flow`가 `IMAGE_FAMILIES`에 없다고 적힌 상태로 남았습니다. `c4468af0`이 `FAMILIES`와 `IMAGE_FAMILIES` 목록에 `mage_flow`를 넣고 해당 행을 등록됨으로 표시했습니다. 4절의 검증 수치는 `dd5b2cb9`에서 측정했고, 그 뒤 두 커밋은 ERNIE 머지와 이 문서 수정입니다. 이 리포트 작성 시점에 `c4468af0`의 CI는 아직 실행 중이었습니다.

## 4. 검증

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0`, head `dd5b2cb9`:

- `imagegen::mage_flow --include-ignored` 46/46, 무시된 실제 체크포인트 테스트 4개 포함(Turbo와 Edit-Turbo parity, 두 컨디셔너 기준). `imagegen::` 263 통과, 19 무시. `server::routes::images` 18, `server::image` 14, `models::registry` 17, `arch` 55. lib, bins, tests에 대한 clippy `-D warnings`와 make verify 대상이 통과합니다.
- 구현 리뷰가 `c19bad02`에서 바이너리를 다시 빌드해 테스트를 돌렸고 모든 수치를 재현했습니다. Turbo 38.39 dB, 편집 21.80 dB, 업스트림 컨디셔닝 편집 39.87 dB, 컨디셔너 비율 0.960과 1.042.
- 픽스처는 GB10의 mlx 0.32.4 CUDA와 73d366ec의 mlx-vlm 0.7.8에서 TF32를 끄고, 트랜스포머와 VAE에만 조합 LayerNorm을 적용해 만들었습니다. 두 번 재생성해 바이트 단위로 동일했습니다(Turbo 17.4 s / 16.6 GB, 편집 19.1 s / 17.8 GB). 프로세서 픽셀은 PIL bicubic과 rel-RMS 1.95e-3, 8비트 기준 최대 두 단계 차이입니다.
- CLI, 1024x1024 Turbo, seed 42, 기본값(4 스텝, guidance 1): 빨간 테이블 위의 파란 유약 찻주전자. 참조 두 장의 Edit-Turbo: 이미지 1의 헤드폰이 이미지 2의 대리석 판 위에 놓인 결과. 두 번 실행이 바이트 단위로 동일했습니다.
- 서버: Turbo 생성 200(오라클 대비 38.4 dB). 참조 두 장 Edit-Turbo 200, CLI PNG와 바이트 단위로 동일하며 크기를 주지 않으면 1248x832. 잘못된 작업, 16의 배수가 아닌 크기, 512..2048 밖의 변, 잘못된 `static_shift`는 400을 반환합니다.
- 단위 테스트는 모듈이 없는 main에서는 컴파일되지 않습니다. 여섯 디렉터리 이름의 변형 판별, 업스트림 대비 msrope position id, VAE 단계, 템플릿과 프로세서 resize 규칙, 요청 상한, LayerNorm 결정성 테스트를 다룹니다.

## 5. 학습 포인트

- PSNR 미달의 위치를 찾으려면 한 단계의 입력을 업스트림 것으로 바꿔 넣어 봅니다. 업스트림 컨디셔닝을 우리 트랜스포머와 VAE에 넣자 Edit-Turbo가 21.8에서 39.9 dB가 되었고, 한 번의 실행으로 컨디셔너 이후 단계 전체가 혐의에서 벗어났습니다.
- 차이를 결함으로 판단하기 전에 업스트림을 자기 자신과 비교합니다. f32 컨디셔너를 쓴 업스트림의 자기 대비 22.9 dB가 편집 케이스의 하한이 되었고, 업스트림 자신의 f32 거리가 컨디셔너 기준의 척도가 되었습니다.
- 정밀도 하한보다 낮은 절대 상한은 공유 f32 기준값에 대한 비율 기준으로 대체할 수 있습니다. 다만 비교 대상을 pooling한 해상도를 그대로 물려받으므로, 그 해상도를 한계로 기록해야 합니다.
- 동일한 두 실행이 다르면 parity보다 결정성을 먼저 확인합니다. 동일한 편집 실행 사이의 29.8에서 35.5 dB 편차는 포팅 오류가 아니라 커널 레이스를 가리켰고, 업스트림에서도 재현되었습니다.
- 실패를 반환하지 않는 eval이 있는 공유 경로에서는 큰 입력이 오류가 아니라 크래시가 됩니다. 공유 경로가 실패를 반환할 수 있게 될 때까지 패밀리의 요청 계층에서 입력을 제한합니다.

## 6. 후속 조치와 한계

리뷰 중 수정: 컨디셔너 입력 상한, vision marker 거부, DiT 토큰 상한, `max_size`보다 우선하는 명시적 편집 크기, `MageFlowPipeline` probe 단위 테스트, 레지스트리의 여섯 변형 별칭 전부(기존에는 세 개), LayerNorm doc comment, 정상 상태 메모리에 대한 모듈 문서.

남은 MEDIUM 지적:

- 공유 `get_input_embeddings` vision 경로는 실패를 반환하지 않는 eval을 쓰며 chat 경로도 같습니다. 3.4의 요청 상한은 입력을 제한할 뿐 경로를 실패 반환 가능하게 만들지는 않습니다.
- CFG를 쓰는 편집은 요청마다 vision tower를 두 번 실행합니다. `get_input_embeddings_with_cache`로 결과를 재사용할 수 있습니다.
- 첫 요청 이후 매 요청마다 컨디셔너를 디스크에서 다시 로드합니다.
- `GroupNorm::forward_native`(CoD 디코더)는 레이스가 있는 fused 커널을 여전히 쓰며 결정성은 테스트하지 않았습니다.
- `layers.rs`가 `dit_blocks.rs`와 `flux2/embeddings.rs`에 이미 있는 `scalar_like`, `split_last`, `modulate`, `rms_norm`을 다시 구현합니다. 다른 패밀리도 비공개 사본을 갖고 있습니다.

남은 LOW 지적: n > 1 편집은 이미지마다 참조를 다시 인코딩합니다. 참조 픽셀을 f32 호스트 버퍼 하나에 모읍니다(2048x2048 참조 16장에서 약 805 MB). 양자화되지 않은 트랜스포머와 VAE 텐서는 로드 시 config와 shape을 대조하지 않습니다(체크포인트는 운영자가 신뢰하는 입력).

외부 후속 작업:

- #2696: CUDA `layer_norm.cu` overlay. 머지되면 Mage-Flow는 fused 커널로 되돌아갈 수 있고, 위 GroupNorm 문제도 해결됩니다.
- f32 Qwen3-VL 로드 경로가 있으면 3.2의 pooled 평균 단서를 해소하는 f32 대 f32 컨디셔너 비교가 가능합니다. PR은 이 경로를 추가할지 메인테이너에게 묻고 있습니다.

Metal은 레지스트리에서 Partial로 표기되어 있고 실행하지 않았습니다. ROCm도 실행하지 않았습니다.

## 참고 자료

- 이슈 #2426, 에픽 #2348, 런타임 이슈 #2341(PR #2497), ERNIE-Image PR #2687, LayerNorm 레이스 #2696, 정밀도 선례 #932, 이전 범위 기록 #524
- 업스트림: Blaizzy/mlx-vlm `mlx_vlm/models/mage_flow`와 PR #1708, #1979, #2373
- `docs/image-generation.md`, `docs/vision-model-scope.md`
