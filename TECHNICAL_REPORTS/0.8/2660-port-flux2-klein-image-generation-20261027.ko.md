# PR #2660: FLUX.2 Klein 텍스트-이미지 생성 및 편집 포팅

**날짜**: 2026-10-27
**상태**: CUDA(GB10)에서 완료, Metal 및 ROCm 실행 기록 없음
**언어**: Rust, Python(픽스처 생성기), Markdown
**위험도**: 중간

## 요약

PR #2660(이슈 #2424, 에픽 #2348)은 mlxcel에서 실제로 실행 가능한 첫 이미지 생성 패밀리인 FLUX.2 Klein(`flux2`)을 mlx-vlm `models/flux2`(git 73d366ec)에서 포팅합니다. Klein 4B와 9B 체크포인트의 증류판(4 스텝, guidance 1.0)과 base판(50 스텝, guidance 4.0), `9b-kv` 참조 K/V 캐시 변형, bf16 Diffusers 트리, MLX affine 4비트 익스포트에 대해 텍스트-이미지 생성과 참조 이미지 편집을 지원합니다. 패밀리는 `imagegen::FAMILIES`와 모델 레지스트리에 등록되어 `mlxcel generate-image`, `mlxcel-server --image-model`(`/v1/images/generations`, `/v1/images/edits`), `mlxcel arch`에서 모두 인식됩니다.

같은 GB10 호스트의 CUDA 업스트림 mlx-vlm과 비교했을 때 4B bf16 수용 케이스(512x512, 4 스텝, seed 9909)는 초기 노이즈가 비트 단위로 동일하고 디코딩 PSNR은 32.27 dB이며, 4B 편집 케이스는 40.54 dB입니다. 이슈의 수용 수치 두 개는 어떤 구현으로도 달성할 수 없음을 보이고 더 강한 검사로 대체했습니다(3.3절). 리뷰 과정에서 업스트림의 1 MP 편집 크기 규칙이 연결되었고, 호스트 메모리를 소진할 수 있는 무제한 배치 VAE 디코드가 수정되었습니다.

## 1. 문제 정의

#2341은 공유 이미지 생성 런타임(`src/imagegen/`: `ImageGenerationModel` 트레이트, 체크포인트 프로브, `Flux2Vae`, 잠재 패킹, 공유 Qwen3 hidden-state 컨디셔너, CLI 및 서버 라우트)을 비어 있는 패밀리 테이블과 함께 도입했습니다. FLUX.2 디렉터리는 프로브가 인식한 뒤 `image generation family flux2 is not supported by this build`로 거부되었습니다. FLUX.2 Klein은 #2341이 스케줄러를 지원하던 Qwen-Image 계열과도 다릅니다. 이미지 시퀀스 길이에 따른 경험적 `mu` 시간 시프트, Qwen3 9/18/27번 레이어 연결 컨디셔닝, 4축 RoPE를 쓰는 dual-stream 및 single-stream DiT, 큰 이미지를 위한 타일 VAE 디코드, 참조 이미지 잠재를 토큰 시퀀스에 이어 붙이는 편집 경로가 필요합니다. Prism Bonsai(#2428)는 3값(ternary) FLUX.2 변형이므로 여기서 들어가는 코드는 비공개가 아니라 재사용 가능해야 했습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 패밀리 모듈 | 새 `src/imagegen/flux2/`(config, embeddings, attention, blocks, transformer, weights, scheduler, prompt, tiling, kv_cache, edit, pipeline, mod)가 Generate와 Edit용 `ImageGenerationModel`을 구현합니다. |
| 변형 판별 | `config.rs`는 업스트림 `_defaults_variant`를 따릅니다. 디렉터리 이름이나 단일 파일 가중치로 `9b-kv`를 판별하고, 그 외에는 `model_index.json`의 `is_distilled`로 증류판 또는 base판 기본값을 고릅니다. 트랜스포머 형상은 `transformer/config.json`에서, 그것이 없는 양자화 익스포트는 인덱스 메타데이터에서 읽습니다. |
| 트랜스포머 | 토큰별 모듈레이션 블렌드, timestep 및 guidance 임베딩, 4축 RoPE, SwiGLU를 갖춘 dual-stream 및 single-stream 블록. 선형 레이어는 `LinearSource` 트레이트(`embeddings.rs`)로 생성하며 기본값은 affine 또는 MLX 양자화 `UnifiedLinear`입니다. |
| 스케줄러와 프롬프트 | `flux2/scheduler.rs`는 경험적 `mu` 스케줄러를 f32로 포팅합니다(공유 Qwen-Image 스케줄러 파일과 분리). `prompt.rs`는 채팅 템플릿, 시퀀스 길이 버킷(32..512), 공유 컨디셔너의 9/18/27 레이어 스택, `(0, 0, 0, i)` 텍스트 id를 적용합니다. |
| 잠재 헬퍼(공유) | `latent::prepare_packed_latents`와 `prepare_grid_ids`. 업스트림과 같은 RNG 호출과 추출 순서를 사용해 초기 노이즈가 비트 단위로 같습니다. |
| VAE(공유) | `Flux2Vae::bn_normalize`와 `denormalize_packed_latents` 추가. `tiling.rs`는 mlx-vlm#1381 블렌딩 수정을 포함한 업스트림 타일 디코드를 포팅하며 `max(h, w) >= 256`에서 자동으로 켜집니다. |
| 편집과 9b-kv | 참조 이미지를 리사이즈해 VAE로 인코딩하고 참조 잠재를 `t = 10 + 10 * i`에 배치합니다. `9b-kv`는 0번 스텝에서 참조 K/V를 저장해 재사용하며 참조 토큰에는 timestep 0 모듈레이션을 씁니다. |
| 트레이트 훅(공유) | `ImageGenerationModel::check_request`(기본은 허용)로 패밀리가 작업 전에 요청을 거부할 수 있고, 이미지 워커는 실패를 HTTP 400으로 매핑합니다. `default_edit_size`(기본은 참조 크기 유지)로 FLUX.2가 CLI와 `/v1/images/edits`에서 업스트림 1 MP 크기 규칙을 적용합니다. |
| 등록과 문서 | `imagegen::FAMILIES`와 `src/models/registry.rs`에 행 추가(`image_generate` 런타임, `image` 출력). `docs/image-generation.md`, `docs/supported-models.md` 갱신. `MLXCEL_TEST_FLUX2_DIR`를 테스트 전용 변수로 허용 목록에 추가. |
| 픽스처 | `tests/fixtures/flux2/`(약 2.6 MB): 생성 및 편집 잠재와 PNG, f32 트랜스포머 0번 스텝 텐서, 참조 이미지, 생성 스크립트. |

## 3. 기술적 선택과 그 이유

### 3.1 패밀리 전용 스케줄러와 선형 레이어 이음매

FLUX.2 스케줄러는 공유 `src/imagegen/scheduler.rs`가 아니라 `flux2/scheduler.rs`에 둡니다. `mu` 계산(4300 토큰에서 분기)이 FLUX.2와 Bonsai 전용이고, 공유 파일은 Z-Image 유닛이 병렬로 수정하기 때문입니다. 트랜스포머는 모든 선형 레이어를 `LinearSource`로 읽으므로 #2428은 3값 소스만 제공하면 블록, 임베딩, 스케줄러, 프롬프트 경로, 잠재 헬퍼를 포크 없이 재사용할 수 있습니다. `dit_blocks`의 RoPE와 timestep 임베딩은 레이아웃과 정밀도가 FLUX.2와 달라 재사용하지 않았습니다.

### 3.2 구성 요소별 정밀도

업스트림은 전부 bf16으로 실행합니다. f16 정책으로 측정한 결과 트랜스포머는 f16에서 유지되지 않아(NaN 잠재, 10.2 dB) 모든 백엔드에서 bf16을 씁니다. VAE는 유지되어(타일 54.76 dB, 비타일 52.37 dB) `Flux2Vae::default_dtype`을 따르고, Qwen3 텍스트 인코더도 유지되어 Qwen3 로더 정책을 따릅니다. 편집은 업스트림과 마찬가지로 f32 참조 잠재에 의한 승격으로 트랜스포머를 f32로 실행합니다.

### 3.3 달성 불가능한 수용 수치 두 개의 대체

- **최종 잠재 상대 RMS 1e-2 미만.** 측정값은 1.03e-1(생성), 4.11e-2(편집)입니다. 처음 갈라지는 텐서는 context embedder 출력으로, 같은 입력과 가중치에 대한 7680폭 bf16 matmul 하나입니다. 128폭 image embedder는 비트 단위로 같습니다. 원인은 이 빌드의 MLX 핀과 mlx 0.32.4 사이의 bf16 GEMM 축약 순서 차이이며, 4번의 Euler 스텝이 이를 증폭합니다. 업스트림 자체의 f32 실행과 bf16 실행도 같은 케이스에서 7.1e-2 차이가 납니다. 테스트는 대신 다음을 단언합니다: f32 트랜스포머가 업스트림 f32와 비트 단위로 동일(16384개 중 16384개), VAE가 업스트림 잠재를 57.59 / 58.45 dB로 디코딩, 종단 간 PSNR 30 dB 이상, 그리고 잘못된 포트(약 1.4)는 통과할 수 없는 0.2 잠재 상한.
- **9b-kv 편집이 9B 편집과 35 dB 이내.** 9b-kv 체크포인트는 자체 가중치를 가지며, 업스트림 자체의 9b-kv 편집과 9B 편집도 25.15 dB 차이입니다(이 포트는 25.09 dB). 의미 있는 검사는 9b-kv 대 업스트림 9b-kv이며 42.94 dB입니다.

### 3.4 배치 시드와 배치 디코드

mlx-vlm#2355를 따라 `n`개의 생성 이미지는 컨디셔닝을 공유하고 배치 `n`인 노이즈 텐서 하나에서 나오므로, 모든 이미지가 `seed`를 보고하며 이미지 `i`는 `seed + i` 단독 실행으로 재현되지 않습니다. 이 예외는 트레이트 문서와 `docs/image-generation.md`에 기록했습니다. 업스트림은 배치를 타일 없이 디코딩하는데, 2048x2048에서 이미지당 약 15 GiB이므로 최대 `n = 10`이면 121 GB 호스트를 넘습니다. 이 포트는 이미지마다 단일 이미지 타일링으로 디코딩합니다(2048x2048, `n = 2`에서 최대 0.55 GiB). 체크포인트 기반 테스트가 이것이 각 이미지를 따로 디코딩한 결과와 바이트 단위로 같음을 단언하며, 업스트림의 비타일 배치 디코드와는 타일/비타일 차이만큼 다릅니다.

### 3.5 업스트림 VAE 인코더 레이아웃 버그

업스트림 `_match_conv_layout`은 `[128, 3, 3, 3]` 형상에서 `encoder.conv_in.weight`의 PyTorch 레이아웃과 MLX 레이아웃을 구분하지 못해 PyTorch 순서로 남깁니다. 이 포트는 올바른 레이아웃을 계산하며, 편집 픽스처는 그 텐서를 전치해 생성했습니다(`--fix-conv-in`). 수정하지 않은 업스트림과 비교하면 알려진 버그와 비교하는 셈이 됩니다.

## 4. 검증과 측정 결과

오라클: mlx-vlm 0.7.8(git 73d366ec), mlx 0.32.4 CUDA 백엔드, 같은 GB10 호스트, `MLX_ENABLE_TF32=0`.

| 검사 | 결과 |
|---|---|
| 초기 노이즈, 생성 seed 9909 및 편집 seed 123 | 비트 단위 동일 |
| 4B bf16 생성, 512x512, 4 스텝 | 32.27 dB, CLI와 `/v1/images/generations` 픽셀 동일 |
| 4B 편집 "Add black sunglasses", seed 123 | 40.54 dB, CLI와 `/v1/images/edits` 픽셀 동일 |
| 4B 4비트 생성 | 업스트림 4비트 대비 35.47 dB |
| base-4B 12 스텝(기본값 50 / 4.0으로 판별) | 32.80 dB |
| 9B bf16 / 9B 4비트 생성 | 25.78 / 20.70 dB(구도 동일, 세부 묘사만 차이) |
| 9B / 9b-kv 편집 | 44.39 / 42.94 dB |
| 512에서 타일 대 비타일 디코드 | 25.05 dB, 업스트림 자체 값 25.05 dB와 동일 |

단위 테스트는 시퀀스 길이 1024와 5000에서 4 및 50 스텝 스케줄러(오라클 비트 패턴 대비), 프롬프트 길이 1, 33, 600의 버킷, 텍스트 id 레이아웃, 타일 디코드, kv 추출 대 캐시 동등성, 설정과 변형 처리, `--gen-kwargs` 파싱, 편집 크기, `check_request`의 400 매핑을 다룹니다. `FAMILIES` 행을 제거하면 등록 테스트 네 개가 실패합니다. 최종 lib 필터: `imagegen::` 167 통과, `flux2` 35, `server::image` 13, `server::routes::images` 18, `models::registry` 16, bin `generate_image` 15, 체크포인트 기반 패리티 6개 중 6개.

## 5. 학습 포인트

- 두 쪽이 서로 다른 MLX 핀으로 실행되면 올바른 포트라도 bf16 종단 간 허용 오차를 달성하지 못할 수 있습니다. 결정적 증거는 같은 입력에서 같은 그래프를 f32로 실행하는 것이며, 이는 연산 순서 효과와 포팅 오류를 분리합니다.
- 체크포인트를 살펴보기 전에 작성한 수용 기준은 존재하지 않는 동등성(9b-kv와 9B 가중치)을 가정할 수 있습니다. 업스트림을 업스트림 자신과 비교하는 것이 임계값이 의미 있는지 가장 빨리 확인하는 방법입니다.
- 단일 Apple 장치에서는 문제없는 업스트림 동작(비타일 배치 디코드)도 `n`이 최대 10인 HTTP 엔드포인트 뒤에서는 서비스 거부 경로가 될 수 있습니다.

## 6. 후속 작업과 한계

- Metal과 ROCm 실행은 기록되지 않았으며, `arch` 행은 Metal을 partial로 보고합니다.
- 4B 생성 케이스는 30 dB 기준을 약 2 dB 차이로 통과하므로 향후 MLX 핀 변경으로 바뀔 수 있습니다.
- 보안 리뷰의 미해결 MEDIUM 항목: 로드 실패 시 500 응답에 체크포인트 절대 경로 노출(여러 라우트에 공통된 기존 동작), 공유 상한 외의 요청별 연산 예산 부재, 새 프롬프트마다 텍스트 인코더 재로드(업스트림 축출 방식과 동일).
- LOW: `n > 1` 편집은 같은 참조를 이미지마다 다시 인코딩합니다. `9b-kv` 0번 스텝은 캐시 K/V 배열을 하나씩 평가합니다. `generate_flux2_fixtures.py`는 4B 픽스처만 재현합니다.
- `250x512`는 공유 검증기가 패밀리 검사보다 먼저 실행되어 256..=2048 범위가 아니라 16의 배수 규칙을 보고합니다.

## 참고 자료

- 이슈 #2424, 에픽 #2348, 런타임 이슈 #2341(PR #2497), 후속 #2428(Prism Bonsai)
- 업스트림: Blaizzy/mlx-vlm `mlx_vlm/models/flux2` 및 PR #1232, #1236, #1381, #1729, #1730, #2355, #2373
- `docs/image-generation.md`, `docs/supported-models.md`
