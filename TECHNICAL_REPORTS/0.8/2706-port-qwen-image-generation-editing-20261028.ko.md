# PR #2706: Qwen-Image-2.1 생성 및 편집 포팅 (qwen_image)

**날짜**: 2026-10-28
**상태**: CUDA(GB10)에서 완료, Metal과 ROCm은 미실행
**언어**: Rust, Python (픽스처 생성기), Markdown
**위험도**: 낮음에서 중간 (신규 패밀리, 라이브 Qwen3-VL 모델에 대한 추가형 변경 하나, 호출자가 하나뿐인 스케줄러 모드의 수치 변경 하나)

## 요약

PR #2706(이슈 #2342, 에픽 #2348)은 mlx-vlm `models/qwen_image`(git 73d366ec)를 `qwen_image` 이미지 패밀리로 포팅합니다. 원본 `Qwen/Qwen-Image-2.1` Diffusers 트리를 변환 없이 읽어 텍스트-이미지 생성과 지시 기반 편집을 제공하며, 편집은 하나 이상의 참조 이미지를 받습니다. RGBA 참조의 알파는 VAE를 거쳐 PNG에 그대로 돌아옵니다. `mlxcel generate-image`, `/v1/images/generations`, `/v1/images/edits`, 프로브, 모델 레지스트리, `mlxcel arch --json`에 연결되어 있고, VAE는 Ming-Image-0.1-Design(#2347)이 재사용하도록 설계했습니다.

CUDA 오라클(mlx 0.32.4, TF32 끔, 1024x1024, seed 7, 4 steps) 대비 모든 케이스가 에픽의 30 dB PSNR 기준을 크게 넘습니다. 생성 52.22 dB, RGBA 편집 52.61 dB(알파 53.58 dB)이고, 참조 KV 캐시를 끈 편집은 캐시를 켠 결과와 비트 단위로 같습니다. 초기 노이즈도 업스트림과 비트 단위로 같습니다. 최종 latent rel-RMS 드리프트는 7.55e-3(생성), 6.38e-3(편집)으로 이슈의 원래 기준 1e-2보다도 작습니다.

## 1. 문제 정의

Qwen-Image-2.1은 Qwen3-VL 8B 텍스트/비전 컨디셔너(36층 x 4096), 32블록 단일 스트림 트랜스포머(32 heads x 128, 3축 RoPE `[16, 56, 56]`, `causal_condition`), 프레임별 2D 합성곱으로 실행되는 3D causal residual VAE(`base_dim` 96, `decoder_base_dim` 144, `z_dim` 64, 공간 배율 16)로 구성됩니다. 이 PR 이전에는 Qwen-Image 디렉터리를 인식하는 코드가 없었습니다.

트리 안에 대응물이 있던 것은 컨디셔너뿐인데 그것도 정확히 맞지 않았습니다. `Qwen3VLModel::forward_hidden`은 항상 최종 RMSNorm을 적용하지만 Qwen-Image는 그 norm 이전의 마지막 디코더 층 출력을 씁니다. 이슈는 Qwen3-VL 디코더를 복사하지 말고 재사용할 것, 라이브 Qwen3-VL 동작을 바이트 단위로 유지할 것, Ming-Image의 4채널 non-residual 설정이 재사용할 수 있게 VAE를 구성할 것을 요구했습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 패밀리 모듈 | 신규 `src/imagegen/qwen_image/`(`config`, `text_encoder`, `transformer`, `blocks`, `kv_cache`, `vae`, `reference`, `pipeline`, `mod`, `*_tests.rs`, `parity_tests.rs`). Generate와 Edit를 지원하는 `ImageGenerationModel` 구현. |
| 컨디셔너 | `text_encoder.rs`가 `<model>/text_encoder/`의 Qwen3-VL을 기존 로더로, `<model>/processor/`의 프로세서와 함께 읽습니다. 토크나이저로 `drop_idx`(14)를 계산하고 편집 참조에는 비전 타워를 돌립니다. 인코딩 후 텍스트 인코더는 해제합니다. |
| Qwen3-VL (라이브 모델) | `src/models/qwen3_vl.rs`에 공유 내부 forward의 두 번째 const generic `FINAL_NORM`을 통해 `forward_hidden_pre_norm`을 추가. 기존 호출자는 모두 `true`를 넘깁니다. `// Used by:`에 MageFlow와 QwenImage 기재. |
| 트랜스포머 | zero-centered RMSNorm, SwiGLU, 3축 RoPE, causal-condition 변조(변조 행 2개)를 쓰는 단일 스트림 블록. block-causal 어텐션은 `mlxcel_core::layers::attention`(큰 해상도에서 query chunking)을 거칩니다. |
| 편집 KV 캐시 | `kv_cache.rs`: step 0에서 프롬프트와 참조의 K/V를 추출하고 이후 step은 타깃 토큰만 실행. `use_kv_cache: false`로 끌 수 있습니다. |
| VAE | `QwenImageVae::new` / `load`. 설정에 이슈가 나열한 모든 필드와 `decoder_base_dim`을 담고, residual과 plain 레이아웃, Diffusers와 MLX conv 레이아웃(이중 permute 방지)을 모두 처리합니다. `// Used by: QwenImage, MingImage`. |
| 참조 이미지 | `reference.rs`: Pillow와 정확히 같은 Lanczos 리사이즈와 흰 배경 합성. 전처리가 업스트림과 바이트 단위로 일치합니다. |
| 스케줄러 | `FlowMatchEulerDiscrete`의 `AccumulateF32` step이 덧셈 전에 velocity delta를 f32로 넓힙니다. 이 모드를 만드는 것은 `FlowMatchEulerDiscrete::dynamic`뿐이고 호출자는 Qwen-Image뿐입니다. |
| 등록과 문서 | `imagegen::FAMILIES`, `models/registry.rs`(image generation 카테고리와 테스트), `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md`. `MLXCEL_QWEN_IMAGE_ORACLE_DIR`, `MLXCEL_QWEN_IMAGE_OUT` 허용 목록 등록, `check_binary_assets.py`에 픽스처 선언. |
| 픽스처 | `tests/fixtures/imagegen/qwen_image_tiny_refs.safetensors`(304 KB), `qwen_image_parity.safetensors`(96 KB), 생성기 `tests/fixtures/generate_qwen_image_fixtures.py`. |

통계: 29개 파일, +5684 / -14. 커밋: `397e84ed`(포팅), `45839010`과 `076f371d`(정밀도와 CLI 수치 문서화), `1c5519f6`(vision model scope 문서), `f0fb5440`(편집 메모리 상한), `22a68ed3`(요청 조기 거부), `origin/main` 병합(`8887f10b`, `6fb5c1f0`, `b5b6ebe9`).

## 3. 기술적 선택과 그 이유

### 3.1 패리티 기준 (에픽 #2348 사용자 결정)

이슈는 최종 latent rel-RMS 1e-2 미만, 디코딩 이미지 rel-RMS 2e-2 미만을 요구했습니다. 에픽 #2348은 이를 잘못된 포팅을 거부하는 케이스별 latent 상한, CUDA 오라클 대비 PNG PSNR 30 dB 이상, 비트 단위로 같은 초기 노이즈로 바꾸었고 rel-RMS는 드리프트로 기록만 합니다.

| 케이스 | 최종 latent rel-RMS (드리프트) | 8x8 풀링 latent (상한 2e-2) | PNG PSNR |
|---|---|---|---|
| 생성 | 7.55e-3 | 5.57e-3 | 52.22 dB |
| 편집, RGBA 참조, KV 캐시 켬 | 6.38e-3 | 4.45e-3 | 52.61 dB (알파 53.58 dB) |
| 편집, KV 캐시 끔 | 6.38e-3 | 4.45e-3 | 52.61 dB, 캐시 켠 결과 대비 inf |

풀링 latent 상한이 잘못된 포팅을 거부함도 확인했습니다. RoPE 높이/너비 축을 바꾸면 0.770과 13.22 dB, 모든 토큰을 실제 timestep 변조 행에 두면 0.234와 18.34 dB입니다. 동일 입력 컴포넌트 검사: 첫 step velocity 6.3e-3 / 6.9e-3, 참조 latent 1.66e-3, 업스트림 최종 latent의 VAE 디코딩 97.79 dB, 실제 VAE 디코딩 2.07e-4와 인코딩 5.7e-7, 컨디셔너 텍스트 위치 3.65e-2 / 3.71e-2(`<|image_pad|>` 행은 0.59로 더 크게 다르지만 트랜스포머가 그 행을 VAE 토큰으로 대체합니다).

Mage-Flow(#2688)와 달리 편집 케이스에 예외가 필요하지 않았습니다. 이곳의 컨디셔너 드리프트는 Mage-Flow 생성 케이스 수준이고, bf16 비전 타워가 가장 크게 벗어나는 image-pad 행은 트랜스포머가 버립니다.

### 3.2 const generic을 통한 pre-norm hidden state

pre-norm 경로는 런타임 플래그나 복사본이 아니라 기존 내부 forward의 두 번째 const generic입니다. 기존 호출자는 `<_, true>`로 인스턴스화되어 같은 연산으로 컴파일되며, teacher-forced logit trace(수정 vs 되돌린 `qwen3_vl.rs`)에서 텍스트 512개 위치와 이미지 128개 위치 모두 불일치 0, TSV 출력은 바이트 단위로 동일했습니다. 런타임 `bool`은 라이브 디코드 경로에 분기를 더하고, 복사는 이슈의 재사용 규칙을 어깁니다.

### 3.3 f32 Euler step에서 velocity delta 확장

업스트림은 f32 sample에 bf16 delta를 더하고 MLX가 합을 f32로 승격합니다. mlxcel의 CUDA MLX 빌드에서는 그 덧셈이 bf16으로 반올림되어 f32 누적의 의미가 사라졌습니다. 수정은 delta를 먼저 f32로 바꿉니다. 공유 코드이지만 `AccumulateF32` 모드만 건드리며, 이 모드는 `dynamic`만 만들고 Qwen-Image만 호출하므로 Z-Image, Mage-Flow 등 다른 패밀리에는 영향이 없습니다.

### 3.4 편집 메모리 상한 (보안 리뷰)

첫 버전은 Mage-Flow의 이미지 토큰 상한 65536을 그대로 가져왔습니다. Qwen-Image는 토큰당 비용이 더 큽니다. `S x S` bool block-causal 마스크가 65536에서 약 4.4 GB에 중간값이 더해지고, 편집 KV 캐시는 CFG 분기당 `S * 512 KiB`(약 34 GB, CFG 사용 시 68 GB)를 40 GB 트랜스포머 옆에 둡니다. 상한은 이제 32768입니다(2048 편집에 2048 참조 하나, 또는 1024 편집에 참조 최대 7개). step 0 추출도 층마다 압축된 prefix를 저장하면서 평가하므로, 모든 층의 전체 시퀀스 K/V가 첫 step 그래프 내내 살아 있지 않습니다. 값은 변하지 않았습니다(편집 패리티 수치 그대로).

토크나이저가 필요 없는 요청 검사(프롬프트나 CFG negative의 비전 마커, 프로세서가 다시 리사이즈할 참조, 토큰 상한)는 `resolve`로 옮겨, 잘못된 편집은 36층 텍스트 인코더를 읽기 전에 400이 됩니다.

### 3.5 정밀도

트랜스포머는 `bf16_to_f16_at_load`를 따릅니다(Metal은 f16, CUDA는 `MLXCEL_CUDA_F16_NORMALIZE=1`이 아니면 bf16). CUDA에서 f16을 강제해도 유한값을 유지하며 생성 48.32 dB, 편집 47.92 dB입니다. VAE는 저장된 대로 f32이고 업스트림도 f32로 실행합니다.

## 4. 검증

- `imagegen::` 373 통과 / 30 무시, `server::routes::images` 18, `image_worker` 14, `models::registry` 18, `qwen3_vl` 51.
- 실제 체크포인트와 오라클 출력으로 `imagegen::qwen_image --ignored`: 최종 head에서 6개 통과.
- `-D warnings` 범위 clippy, `cargo fmt --check`, 계약 `make verify-*` 타깃, `check_binary_assets.py`, `check_env_var_docs.py`: 통과.
- 실제 체크포인트: CLI 생성(30 steps, 119.5초)은 일관된 빨간 찻주전자를 그렸고, CLI 편집은 이를 파랗게 바꾸었으며, RGBA 편집은 타원형 알파 마스크를 유지했습니다(투명 픽셀 39%, 입력은 41%). 서버는 두 라우트에서 200, 1000x1000 크기, 생성 요청의 `use_kv_cache`, 편집 프롬프트의 `<|image_pad|>`에서 400을 반환했습니다.

## 5. 남은 과제와 위험

- 첫 참조의 기본 크기와 같은 명시적 편집 크기를 "크기 미지정"으로 취급합니다(Mage-Flow와 같은 휴리스틱). 고치려면 공유 `ImageRequest`에 "크기 지정됨" 플래그가 필요합니다.
- 공유 `src/vision/qwen3_vl.rs`의 `get_input_embeddings`가 실패 불가 eval을 써서, 비전 타워의 MLX 실패가 500 대신 서버 중단으로 이어집니다. Mage-Flow와 Qwen3-VL 채팅이 공유하는 코드입니다.
- block-causal 마스크를 캐시되지 않은 step마다 다시 만듭니다. 토큰 상한이 크기를 제한합니다.
- `MLXCEL_CUDA_F16_NORMALIZE=1`에서 ERNIE-Image 패리티 테스트가 VAE 디코딩 오류로 실패합니다. 이 PR과 무관하지만 아직 이슈가 없습니다.
- Metal과 ROCm은 실행하지 않았습니다. 패리티 기준은 TF32 끔으로, CLI와 서버 실행은 기본값으로 돌렸습니다.

## 6. 학습 포인트

- 컨디셔너를 판단하기 전에 bf16 드리프트가 어디에 떨어지는지 확인하십시오. 편집 컨디셔너의 image-pad 행은 0.59나 벗어나 보이지만, 트랜스포머가 그 행을 VAE 토큰으로 덮어쓰므로 편집 전체는 52 dB입니다.
- dtype 승격은 백엔드의 성질입니다. 업스트림 MLX에서 f32로 승격되는 f32 + bf16 덧셈이 다른 빌드에서는 bf16으로 반올림될 수 있으니, 누적 정밀도가 중요한 곳에서는 명시적으로 넓히십시오.
- 토큰당 메모리가 다른데 형제 패밀리의 요청 상한을 복사하는 것은 안전하지 않습니다. 패밀리 자신의 마스크와 캐시 크기로 계산하십시오.
