# PR #2662: Z-Image 텍스트-이미지 생성 및 img2img 포팅

**날짜**: 2026-10-27
**상태**: CUDA(GB10)에서 완료, Metal 및 ROCm 실행 기록 없음
**언어**: Rust, Python(픽스처 생성기), Markdown
**위험도**: 중간

## 요약

PR #2662(이슈 #2425, 에픽 #2348)는 mlx-vlm `models/z_image`(git 73d366ec)를 이미지 생성 패밀리로 포팅합니다. Tongyi-MAI Z-Image-Turbo(8 스텝, guidance 0)와 Z-Image base(50 스텝, guidance 4)에 대해 텍스트-이미지 생성과 img2img를 지원합니다. 변환 단계 없이 원본 Diffusers 트리를 바로 읽고, mlx-vlm 변환본(affine, mxfp4, mxfp8, nvfp4, 로드 시 dequantize)도 받습니다. 패밀리는 FLUX.2(#2660) 옆에 `imagegen::FAMILIES`로, 그리고 모델 레지스트리에 등록되어 `mlxcel generate-image`, `mlxcel-server --image-model`, `mlxcel arch --json`에서 모두 `z_image`로 보입니다.

같은 GB10 호스트의 업스트림과 비교하면 초기 노이즈는 모든 케이스에서 비트 단위로 동일하고, PNG PSNR은 32.9 dB(Turbo), 34.7 dB(base), 46.1 dB(img2img)입니다. 이슈의 원래 기준인 최종 latent 상대 RMS 1e-2 미만은 충족하지 못했고(8.6e-2, 6.2e-2, 1.7e-2), 이 기준이 업스트림 자체의 재현성보다 낮다는 것을 보였습니다. 에픽 #2348 진행 중 사용자가 이 기준을 케이스별 latent 상한, PSNR >= 30 dB, 비트 동일 노이즈로 바꿨습니다. 리뷰 과정에서 f16 가중치 변환 검사, 서버를 종료시키던 할당을 없앤 query 청크 attention, `try_eval` 복구, 양자화 텐서의 Rust 측 검증이 추가되었습니다.

## 1. 문제 정의

#2341(PR #2497)은 Z-Image DiT 블록(`src/imagegen/dit_blocks.rs`)을 LLaDA-Image와 Ming-Image용 공유 코드로 포팅했지만 Z-Image 자체는 로드되지 않았습니다. Diffusers 디렉터리에 최상위 `model_type`이 없어서, #2341 이후 프로브는 이를 `z_image`로 분류한 다음 `image generation family z_image is not supported by this build`로 실패했습니다. 빠져 있던 것은 최상위 transformer(patchify, position id, noise/context refiner, final layer), FLUX.1 `AutoencoderKL` config로 만드는 VAE(latent 채널 16, 일부 익스포트에는 `quant_conv` 없음, batch-norm 통계 없음), Qwen3의 끝에서 두 번째 hidden state 컨디셔닝, `steps - 1` 스케줄러 스텝을 쓰는 static-shift 스케줄러, `cfg_truncation`이 있는 CFG, img2img 시작 인덱스 규칙입니다.

업스트림은 먼저 `convert.py` 실행을 요구하지만, 로드 경로가 이미 원본 Diffusers 가중치를 즉석에서 정리합니다. 그래서 mlxcel은 프로젝트의 직접 체크포인트 규칙에 따라 원본 트리를 바로 로드합니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 패밀리 모듈 | 새 `src/imagegen/z_image/`(`config`, `prompt`, `transformer`, `weights`, `pipeline`, `mod`)가 Generate와 Edit에 대해 `ImageGenerationModel`을 구현합니다. |
| Config와 변형 | `config.rs`는 스케줄러 `shift <= 3.0`이면 Turbo, 아니면 base를 고르고, `--gen-kwargs` 또는 추가 JSON 필드에서 `cfg_truncation`(기본 1.0), `strength`(img2img, 기본 0.6), `negative_prompt`를 읽습니다. `img2img_start_index`는 업스트림의 `int(max(steps - min(steps * strength, steps), 0))`를 옮긴 것입니다. |
| Transformer | `dit_blocks`로 조립한 `ZImageTransformer`(블록은 복사하지 않음)와 업스트림 키 재매핑(`all_final_layer.2-1.`, `all_x_embedder.2-1.`, `t_embedder.mlp.0/2`). |
| 프롬프트 | `enable_thinking=True` chat template, 512로 패딩, 공유 Qwen3 컨디셔너의 `hidden_states(...)[num_layers - 1]`, 유효 길이로 잘라냄. |
| 가중치 | `weights.rs`: index 메타데이터, mlx-vlm 포맷 감지, 양자화 표, 양자화 텐서 검증, dequantize, 검사가 붙은 f16 변환, VAE 키 이름 변경. |
| 공유 런타임 | `Flux2Vae`의 `quant_conv`를 선택 사항으로 변경, `apply_rotary`가 batch-1 테이블을 broadcast, DiT와 VAE mid-block attention이 `mlxcel_core::layers::attention`(query 청크)을 거치도록 변경, 이미지 워커가 400으로 매핑하는 `InvalidImageRequest` 오류 타입 추가. |
| 등록과 문서 | `imagegen::FAMILIES`와 `STANDALONE_ARCHITECTURE_FAMILIES`에 행 추가(별칭 `z-image`, `z-image-turbo`), `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md` 갱신, 테스트 전용 변수를 `scripts/ci/check_env_var_docs.py` 허용 목록에 추가. |
| 픽스처 | `tests/fixtures/imagegen/z_image_tiny_refs.safetensors`(1.6 KB), `z_image_parity.safetensors`(198 KB, 4x4 풀링한 latent와 이미지), 생성기 `tests/fixtures/generate_z_image_fixtures.py`. |

통계: 파일 35개, +3933 / -51. 커밋: `e540035b`(포팅), `7aceac10`(f16 변환 검사), `ab732046`(attention 메모리 상한과 MLX 실패 복구), `dde3116c`(양자화 텐서 검증), `94f6553d`(FLUX.2가 들어간 `origin/main` 병합).

## 3. 기술적 선택과 그 이유

### 3.1 1e-2 최종 latent 기준의 수정

이슈는 512x512, seed 42에서 최종 latent 상대 RMS 1e-2 미만을 요구했습니다. PR은 Turbo 케이스에서 업스트림을 업스트림 자신과 비교했습니다. caption embedder linear만 bf16 대신 f32로 계산하면 업스트림 latent가 2.7e-2(41.3 dB) 움직이고, 텍스트 인코더 attention만 f32 대신 bf16으로 돌리면 1.0e-1(32.1 dB), TF32를 켜면 9.5e-2(31.6 dB) 움직입니다. 30개 joint layer를 지나는 8번의 Euler 스텝이 caption stream의 bf16 수준 차이를 증폭하므로, 1e-2는 어떤 bf16 구현에도 바닥값보다 낮습니다.

이 포팅의 drift는 대부분 알려진 한 가지 차이에서 옵니다. 공유 Qwen3 컨디셔너는 attention을 bf16으로 돌리지만 업스트림은 q, k, v를 f32로 캐스팅합니다. caption feature는 9.3e-3 차이가 나고, 업스트림의 caption feature를 이 denoiser에 넣으면 Turbo latent가 8.6e-2에서 3.4e-2로 떨어집니다. `Qwen3Model`은 FLUX.2, Bonsai와 공유하는 실제 텍스트 모델이고 바꾸려면 logit-trace 회귀 실행이 필요하므로 여기서는 수정하지 않았습니다.

에픽 #2348 진행 중 사용자는 기준을 잘못된 포팅을 걸러내는 케이스별 latent 상한, PSNR >= 30 dB, 비트 동일 초기 노이즈로 바꿨습니다. 측정된 rel-RMS는 drift로 기록하고 병합 조건으로 쓰지 않습니다. `src/imagegen/z_image/mod_tests.rs`의 ignored 실제 체크포인트 테스트가 이를 반영합니다: `TURBO_LATENTS = 1.2e-1`, `BASE_LATENTS = 1.0e-1`, `EDIT_LATENTS = 5.0e-2`, `MIN_PSNR_DB = 30.0`. 컨디셔너 수정은 후속 이슈 #2666이며, caption feature는 업스트림 자체의 1.9e-3 근처로, Turbo latent는 1e-2 아래가 아니라 3.4e-2 쪽으로 움직일 것으로 예상합니다.

### 3.2 측정된 round-trip 검사를 붙인 f16 저장

mlxcel MLX의 CUDA 빌드는 bf16과 f32를 bf16으로 승격하므로, bf16 가중치에서는 업스트림의 f32 latent stream이 bf16으로 돌아갑니다(forward 1회 후 2.1e-2, 최종 1.1e-1). 그래서 transformer와 VAE 가중치를 f32와 만나면 f32로 승격되는 f16으로 저장합니다. DiT 블록은 rotation, SDPA, residual add에서 업스트림 승격 규칙에 맞게 캐스팅합니다(Metal에서는 no-op). caption embedder와 첫 context-refiner의 attention 앞 경로(`reads_caption_stream`)는 업스트림이 bf16으로 계산하고 joint layer가 그 반올림을 약 25배 증폭하므로 bf16을 유지합니다.

첫 버전은 bf16에서 f16으로의 변환을 정확하다고 설명했습니다. 리뷰에서 공개 체크포인트의 transformer 텐서 521개 중 355개, VAE 텐서 244개 중 76개가 바뀐다는 것이 드러났습니다. 범위는 안전하지만(최대 크기 47.75) subnormal bf16 값의 하위 비트가 사라져 비트 단위로는 같지 않습니다. 커밋 `7aceac10`은 로드 시 텐서마다 f16 round trip을 측정해 `F16_MAX_REL_RMS = 1e-5`를 넘거나 overflow하는 텐서를 f32로 두고, VAE 텐서 하나라도 실패하면 VAE 전체를 f32로 둡니다. 공개 체크포인트의 최악값은 2.0e-7이라 수치는 바뀌지 않았습니다. 이 검사는 로드 시 텐서마다 호스트 동기화를 두 번 합니다.

### 3.3 raw SDPA 호출 대신 query 청크 attention

DiT attention은 f32 q, k로 raw SDPA FFI를 호출했습니다. MLX CUDA는 f16과 bf16만 cuDNN으로 보내므로, fallback이 전체 `[1, 30, L, L]` f32 score 행렬을 만들었습니다. 레이어당 2048x2048에서 약 34 GB, 공유 요청 상한인 4096x4096에서 약 500 GB입니다. VAE mid-block attention(헤드 1개, head_dim 512)도 같은 형태 문제가 있었습니다. 할당 실패는 non-try `eval`의 C++ 예외로 나타나 cxx 경계를 넘어 `mlxcel-server`를 종료시켰고, `run_guarded`는 Rust panic만 잡습니다.

커밋 `ab732046`은 둘 다 `mlxcel_core::layers::attention`을 거치게 합니다. 이 함수는 `MLXCEL_ATTENTION_CHUNK_BUDGET_MB`(기본 1024)를 넘으면 score를 query 청크로 나누고, 그 아래에서는 같은 단일 SDPA 호출입니다. denoise, 참조 인코딩, VAE 디코드, 프롬프트 인코딩은 이제 `try_eval`과 `try_array_to_raw_bytes`를 써서 MLX 실패 시 프로세스를 죽이지 않고 500을 돌려줍니다. 청크를 강제한 테스트(budget을 프로세스당 한 번 읽으므로 1 MiB budget의 자식 프로세스에서 실행)는 청크 없는 출력과 비교해 DiT 블록 최대 절대 오차 0, VAE attention 3.6e-7이고, `MLXCEL_ATTENTION_CHUNK_BUDGET_MB=64`에서 Turbo parity PNG가 바이트 단위로 동일합니다. half-precision SDPA는 unfused f32 경로를 피하지만 업스트림 parity와 충돌하므로 택하지 않았습니다. OOM 자체는 드라이버 wedge 위험 때문에 GB10에서 재현하지 않았고, 근거는 계산입니다.

### 3.4 MLX에 넘기기 전 양자화 텐서 검증

`dequantize_weights`는 체크포인트 메타데이터를 non-try MLX `dequantize`에 넘겼고, 이 함수는 지원하지 않는 bits나 group size, shape 불일치, uint32가 아닌 packed weight, affine bias 누락에서 예외를 던집니다. 커밋 `dde3116c`는 업스트림 `_apply_weights`에 맞춰 Rust에서 먼저 각 triple을 검사합니다: affine bits {2, 3, 4, 5, 6, 8}, group size {32, 64, 128}; mxfp4 32, mxfp8 32, nvfp4 16; 2차원 이상의 uint32 packed weight와 bit 폭으로 나누어떨어지는 packed 폭; `[..., in / group_size]` 형태의 scale; affine은 float scale과 짝이 맞는 bias, fp 포맷은 uint8 scale과 bias 없음. 각 오류는 텐서 키를 밝힙니다.

### 3.5 모델이 올린 요청 오류는 400

일부 요청 오류는 패밀리 안에서만 판별됩니다(범위를 벗어난 `strength`, 참조 이미지 개수 불일치, Turbo에서의 guidance). Z-Image의 `validate_request`는 GPU 작업 전에 이를 검사해 `src/imagegen/pipeline.rs`의 `InvalidImageRequest`로 표시하고, 이미지 워커는 `anyhow` context가 덧붙어도 `downcast_ref`로 이를 찾아 500 대신 400으로 응답합니다.

### 3.6 FLUX.2(#2660)와의 병합

FLUX.2가 먼저 병합되어 `94f6553d`가 `origin/main`을 이 브랜치에 병합했습니다. 두 PR 모두 `src/server/image_worker.rs`에 실행 전 요청 검사를 넣었습니다. 병합된 워커는 먼저 FLUX.2의 `validate_request`(`ImageGenerationModel::check_request` 호출)를 실행해 실패를 400으로 매핑한 다음, `classify_model_error`와 함께 `run_request`를 호출해 `InvalidImageRequest`는 400, 그 밖은 500으로 매핑합니다. `FAMILIES`는 두 행(`flux2`, `z_image`) 테이블이 되었고, 레지스트리 테스트는 두 행을 모두 확인하고 중복을 거부합니다. Z-Image 레지스트리 행은 `category: "image_generation"`을 썼는데, 병합에서 FLUX.2가 쓰는 표기인 `"image generation"`으로 바꾸고 runtimes를 공유 상수 `IMAGE_GENERATE`로 바꿨습니다. 문서 표도 하나의 패밀리 표로 합쳤습니다.

## 4. 검증과 측정 결과

오라클: git 73d366ec의 업스트림 mlx-vlm, mlx 0.32.4, CUDA 백엔드, 같은 GB10 호스트, 양쪽 모두 `MLX_ENABLE_TF32=0`, 512x512, seed 42.

| 케이스 | 최종 latent rel-RMS | PNG PSNR | 테스트 상한 |
|---|---|---|---|
| Turbo, 8 스텝 | 8.6e-2 | 32.9 dB | 1.2e-1 |
| Base, 50 스텝, guidance 4, negative "blurry, low quality", `cfg_truncation` 1.0 | 6.2e-2 | 34.7 dB | 1.0e-1 |
| img2img, strength 0.6 | 1.7e-2 | 46.1 dB | 5e-2 |

세 케이스 모두 초기 노이즈가 비트 단위로 동일합니다. 표의 전체 해상도 수치는 임시 오라클 텐서에서 나왔고, 커밋된 픽스처는 4x4 풀링 요약을 저장합니다(Turbo: 풀링 latent 3.15e-2, 풀링 PSNR 36.25 dB).

- CLI, Turbo 1024x1024, seed 42, 기본값: 36.9초, 깨끗한 PNG.
- 서버(`mlxcel-server -m qwen3-0.6b --image-model z-image-turbo-bf16`): 생성 200(10.7초, 512), data-URL 이미지와 strength 0.6 편집 200(5.9초). strength 0, strength 1.5, 이미지 2장, Turbo에 guidance 4는 각각 400. TF32를 켠 서버 기본값에서 PNG는 오라클 대비 32.4 dB와 46.6 dB.
- main에서 실패하는 단위 테스트(모듈 없음): shift 3과 6, 스텝 8과 49의 `static_shift` sigma, `_img2img_start_index` 표, transformer 키 정리, 변환된 디렉터리에서 VAE 정리 no-op, 양자화 표와 검증, 워커 400 매핑.
- 최종 실행(CUDA, TF32 끔): imagegen lib 173 통과, 4 ignored, `server::image` 12, `server::routes::images` 17, `models::registry` 16. clippy `-D warnings`와 fmt 통과.
- `quant_conv`를 선택 사항으로 바꾼 뒤 ERNIE VAE ignored 테스트: f32 수치 변화 없음(3.3e-6 / 1.2e-6 / 3.0e-6). default-dtype assert는 CUDA(bf16 기본)에서 main에서도 똑같이 실패하므로 기존 문제입니다.

업스트림 Turbo 생성 기본값은 README상 9 스텝이지만, 이 포팅은 업스트림 코드와 이슈가 지정한 8 스텝을 씁니다.

## 5. 학습 포인트

- 수치 허용치를 받아들이기 전에 업스트림을 업스트림 자신과 비교합니다. 참조 쪽에서 정밀도 설정 하나만 바꿔 1e-2가 달성 불가능함을 보였고, 업스트림의 중간값(caption feature)을 포팅에 넣어 남은 drift를 구성 요소 하나로 좁혔습니다.
- 이 MLX 빌드에서는 업스트림 MLX와 달리 bf16과 f32가 bf16으로 승격됩니다. 가중치를 f16으로 저장하면 업스트림의 f32 activation stream이 돌아오지만, 변환이 정확하다고 가정하지 말고 텐서마다 측정해야 합니다.
- non-try `eval`의 C++ 예외는 Rust panic 가드와 상관없이 서버 프로세스를 종료시킵니다. HTTP 라우트 뒤에서 요청이 구동하는 MLX 작업에는 `try_eval`과 상한이 있는 할당 형태가 필요합니다.
- 같은 워커와 레지스트리를 건드리는 두 패밀리 PR은 나중 정리가 아니라 병합 시점에 하나의 검증 경로와 공유 문자열의 단일 표기로 맞춰야 합니다.

## 6. 후속 작업과 한계

- #2666: 공유 Qwen3 컨디셔너 attention을 업스트림처럼 f32로 실행. `Qwen3Model` attention에 opt-in 플래그를 두고 `Qwen3TextEncoder`만 켭니다. FLUX.2, Z-Image, Bonsai, Ideogram 4에 영향을 주며, 실제 텍스트 모델은 바이트 단위로 그대로여야 합니다.
- 보고되었지만 수정하지 않은 리뷰 항목: 로드 시 transformer 텐서 shape를 config와 대조하지 않음(크기가 틀린 체크포인트는 첫 요청에서 중단), 공유 한도(1000 스텝, n=10, 16M 픽셀, CFG 2배) 안의 요청 하나가 단일 이미지 워커를 오래 점유할 수 있음, timestep과 무관한 작업(position id, RoPE 테이블, caption embed, context refiner)을 매 스텝 다시 계산(약 1~2%), n>1 img2img가 이미지마다 참조를 다시 인코딩, 텍스트 인코더가 denoiser 전에 해제되지 않고 상주(약 8 GB).
- CUDA에서 DiT attention은 항상 unfused f32 경로를 탑니다(1024x1024에서 레이어당 약 2.5 GB score 트래픽).
- 레지스트리 행은 Metal을 supported로 표시하지만 PR에 Metal과 ROCm 실행 기록은 없습니다.
- `server::image_worker::tests::a_full_queue_sheds_load`는 main에서도 있는 타이밍 flake입니다(약 10회 중 1회).

## 참고 자료

- 이슈 #2425, 에픽 #2348, 런타임 이슈 #2341(PR #2497), FLUX.2 Klein PR #2660, 후속 이슈 #2666
- 업스트림: Blaizzy/mlx-vlm `mlx_vlm/models/z_image`와 PR #1952, #2354, #2373
- `docs/image-generation.md`, `docs/supported-models.md`
