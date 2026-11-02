# PR #2710: Ming-Image-0.1-Design RGBA 텍스트-이미지 포팅 (ming_image)

**날짜**: 2026-10-29
**상태**: CUDA(GB10)에서 완료, gen 케이스는 메인테이너가 승인한 패리티 예외. Metal과 ROCm은 미실행
**언어**: Rust, Python (픽스처 생성기), Markdown
**위험도**: 낮음에서 중간 (신규 패밀리, 라이브 Bailing MoE 모델과 공유 DiT 블록에 대한 추가형 변경)

## 요약

PR #2710(이슈 #2347, 에픽 #2348)은 mlx-vlm `models/ming_image`(git 73d366ec)를 `ming_image` 이미지 패밀리로 포팅합니다. Ming-Image-0.1-Design은 텍스트가 많은 디자인용으로 RGBA를 직접 출력하는 inclusionAI의 6B 텍스트-이미지 모델입니다. 원본 bf16 트리와 MLX 4-bit, 8-bit 트리를 읽고, `mlxcel generate-image`, `/v1/images/generations`(RGBA 유지), 프로브, 모델 레지스트리, `mlxcel arch --json`에 연결됩니다. 업스트림이 지원하지 않는 편집과 1.0 이외의 guidance는 400으로 거부합니다.

RGBA 케이스는 에픽의 30 dB 기준을 끝단 간에 통과합니다(30.67 dB, 알파 72.16 dB). gen 포스터 케이스는 끝단 간 24.98 dB입니다. 메인테이너가 이를 컨디셔너 정밀도 예외로 승인했습니다. 업스트림 자신의 컨디셔닝을 넣으면 포트는 32.45 dB에 도달하고, 같은 크기의 드리프트로 교란한 업스트림은 자기 자신 대비 16.37 dB에 그칩니다. 이슈의 RGBA 기준("픽셀 절반 이상에서 알파 16 미만")은 업스트림 자체가 seed 0에서 완전히 불투명한 출력을 내므로, 오라클 대비 알파 PSNR 30 dB 이상으로 바뀌었습니다.

## 1. 문제 정의

Ming-Image는 BailingMoeV2 멀티모달 LLM으로 Z-Image NextDiT를 컨디셔닝합니다. LLM은 20층 x 2048, 전문가 256개 중 top-8이고, MoE 층마다 라우터가 둘(텍스트 `gate`와 `image_gate`) 있어 이미지 패치 마스크로 토큰별로 고릅니다. 학습된 쿼리 토큰 256개가 이미지 패치 구간을 대체합니다. 쿼리 구간의 마지막 hidden state는 `proj_in`, 28층 양방향 Qwen2 계열 커넥터, `proj_out`을 거쳐 `cap_feats`가 되고, 텍스트 위치에서 선택된 층들의 hidden state를 이어 붙인 것은 `directvlm_norm`과 `directvlm_proj`를 거쳐 `cap_feats_2`가 됩니다. 디코딩은 Qwen-Image VAE의 4채널 non-residual 변형이 맡아 알파를 직접 디코딩합니다.

mlxcel의 `src/models/bailing_moe.rs`는 층마다 라우터가 하나였고, 토큰별 라우터 선택도, 층별 hidden state 반환도 없었습니다. Qwen-Image VAE는 Ming의 재사용을 염두에 두고 #2706에서 막 들어왔습니다. 이슈는 Bailing MoE 층, Qwen-Image VAE, Z-Image 블록, 공유 스케줄러를 재사용하면서 라이브 Bailing MoE 텍스트 모델을 바이트 단위로 유지하라고 요구했습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 패밀리 모듈 | 신규 `src/imagegen/ming_image/`(`config`, `weights`, `text_encoder`, `transformer`, `pipeline`, `mod`, 테스트와 `parity_tests.rs`). Generate 전용. 배치의 `i`번째 이미지는 seed `seed + i`. |
| 이미지 라우터 | 신규 `src/models/bailing_moe_image_router.rs`: 불리언 `[B, L]` 마스크로 `gate`와 `image_gate`를 토큰별 선택, 모든 층의 hidden state와 최종 정규화 상태를 돌려주는 `inputs_embeds` 진입점. `bailing_moe.rs`는 `router_prefix`와 `BailingMoeGate::from_weights`가 `"gate"`로 새 `_named` 변형을 호출하도록 위임하는 것만 바뀝니다. |
| 컨디셔너 | 학습된 쿼리 토큰, 양방향 커넥터, `cap_feats`와 `cap_feats_2`. 커넥터와 브리지는 f32이고 bf16 상태는 명시적으로 넓힙니다. 프롬프트 템플릿과 토큰화는 33 GB 인코더를 읽기 전에 수행됩니다. |
| 트랜스포머 | `src/imagegen/z_image`와 `dit_blocks.rs`의 Z-Image NextDiT에 두 번째 캡션 스트림과 마스크된 zero padding. `dit_blocks::multiply_promoted`는 변조 곱에 업스트림의 dtype 승격을 적용하며, Z-Image처럼 dtype이 같으면 일반 곱셈입니다. |
| VAE | Ming의 4채널 non-residual 설정으로 만든 `qwen_image::vae::QwenImageVae`, 원본 트리의 5차원 커널용 sanitizer. |
| 가중치 | 원본, MLX 4-bit, MLX 8-bit 트리. 로드 시 역양자화 전에 양자화 packing(scale, bias의 shape와 dtype) 검증, 전문가 stacking의 shape 검사. |
| 요청 제한 | guidance != 1.0, 편집, 1..=10 밖의 `n`, 1..=1000 밖의 steps, 256..=2048 밖이거나 16의 배수가 아닌 크기, 256 KiB 또는 8192 토큰을 넘는 프롬프트, 블록 토큰을 직접 쓴 프롬프트는 모두 `InvalidImageRequest`(400). |
| 등록과 문서 | `imagegen::FAMILIES`, `models/registry.rs`, `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md`. `MLXCEL_MING_IMAGE_*` 환경 변수 허용 목록 등록, `check_binary_assets.py`에 픽스처 선언. |
| 픽스처 | `tests/fixtures/imagegen/ming_image_parity.safetensors`(103 KB), `ming_image_tiny_refs.safetensors`(2 KB), `tests/fixtures/generate_ming_image_fixtures.py`(모드 `tiny`, `real`, `layers`, `perturb`). |

통계: 29개 파일, +4698 / -19. 커밋: `0531b6bb`(포팅), `259397d9`(업스트림 컨디셔닝과 최종 norm 검사), `c006b576`(gen을 컨디셔너에서 판정, CLI, 서버, 문서), `e7245d1d`(포맷과 `Used by:`), `8c00c416`(프롬프트 상한과 라우터 마스크 검증), `origin/main` 병합들.

## 3. 기술적 선택과 그 이유

### 3.1 gen 예외와 그 판정 방식 (메인테이너 결정, 2026-10-29)

1024x1024, 12 steps, seed 0, TF32 끔, mlx 0.32.4 CUDA(GB10)의 mlx-vlm 73d366ec 대비, 초기 노이즈는 비트 단위로 동일:

| 케이스 | 풀링 latent (상한) | 최종 latent 드리프트 | PNG PSNR | 알파 PSNR | 판정 |
|---|---|---|---|---|---|
| gen, bf16 | 3.37e-1 (0.6) | 3.87e-1 | 24.98 dB | 65.81 dB | 기록만 |
| rgba, bf16 | 3.03e-2 (6e-2) | 9.82e-2 | 30.67 dB | 72.16 dB | 판정, >= 30 |
| 업스트림 컨디셔닝의 gen | 3.90e-2 (8e-2) | 1.82e-1 | 32.45 dB | 67.19 dB | 판정, >= 30 |
| 업스트림 컨디셔닝의 rgba | 1.38e-3 (1e-2) | 7.08e-3 | 52.69 dB | 82.02 dB | 판정, >= 30 |
| 4-bit 트리에서 q4 대 업스트림 | 4.44e-2 | 1.70e-1 | 41.91 dB | 75.97 dB | 기록만 |

진단은 gen의 차이를 컨디셔너의 누적 bf16 반올림으로 좁힙니다.

- 업스트림 입력 상태에서 백본 층을 하나씩 돌리면 1e-4에서 7.5e-4 차이로 반올림 수준입니다. 최종 norm만 돌리면 비트 단위로 같습니다.
- 20층을 지나며 쿼리 구간 드리프트는 최종 norm 전 8.8e-2, 후 2.16e-1에 이르고, `cap_feats`는 2.05e-1입니다.
- 컨디셔너 이후는 정상입니다. 업스트림 입력에서 첫 step velocity는 4.2e-3이고, VAE는 업스트림 최종 latent를 97.7 dB로 디코딩합니다.
- 백본 층마다 1e-3으로 교란해 같은 드리프트를 만든 업스트림은 교란 없는 자기 PNG 대비 16.37 dB(gen), 29.98 dB(rgba)입니다. 포스터 프롬프트는 이 드리프트 수준에서 업스트림 자신에게도 혼돈적입니다.

끝단 간 풀링 latent로는 잘못된 컨디셔너를 잡지 못합니다. 쿼리 토큰을 텍스트 라우터로 보낸 실행은 1.07e-1로, 올바른 실행보다 업스트림에 더 가깝습니다. 그래서 gen은 컨디셔너에서 판정하고(업스트림 id에서 `cap_feats` <= 0.3, `cap_feats_2` <= 0.12, 텍스트 상태 <= 3e-2), 잘못된 라우터 경로가 4.59e-1로 실패함을 단언합니다. 트랜스포머, 스케줄러, VAE는 업스트림 컨디셔닝 경로로 판정하며, 거기서 `cap_feats_2`를 빼면 실패함(풀링 latent 7.73e-1)도 단언합니다.

### 3.2 RGBA 기준 수정 (메인테이너 결정)

seed 0에서 업스트림의 출력은 시험한 RGBA 프롬프트 둘(사과와 포스터) 모두, bf16과 4-bit 모두 불투명합니다. 알파 16 미만 픽셀 0.0%, 최소 알파 227에서 252입니다. 기준은 이제 출력이 실제 알파 채널을 가지고 오라클 대비 알파 PSNR이 30 dB 이상인 것입니다(측정 72.16 dB). seed에 따라 달라지는 모델 동작 대신 포트의 충실도를 잽니다.

### 3.3 추가형 라우터와 라이브 모델 안전성

이미지 라우터는 별도 모듈에 있고, `bailing_moe.rs`에는 원래 함수가 `"gate"`로 호출하는 `_named` 변형만 생겨 키와 오류 문구가 그대로입니다. logit trace 회귀는 작은 시드 고정 랜덤 `bailing_moe` 체크포인트 두 개(4층, hidden 64, 전문가 8, f32, 하나는 expert bias가 있는 sigmoid 라우터, 하나는 softmax)로 `examples/logit_trace`의 수정판과 되돌린 판을 빌드해 돌렸고, trace가 바이트 단위로 같았습니다(512개 위치 중 불일치 0, 최대 차이 0). 이 호스트에는 Ling이나 Bailing 텍스트 체크포인트가 없고, Ming의 LLM을 텍스트 전용으로 보려면 키 이름을 바꿔야 해서 가중치 복사가 필요합니다.

### 3.4 명시적 dtype 승격

mlxcel의 CUDA MLX 빌드는 bf16 x f32를 bf16으로 반올림하지만 업스트림은 f32로 승격합니다. 두 곳에 명시적 확장이 필요했습니다. 커넥터와 브리지 입력, 그리고 공유 Z-Image 블록의 변조 곱(`multiply_promoted`)입니다. 공유 헬퍼는 dtype이 같으면 아무것도 하지 않으므로 Z-Image와 LLaDA-Image는 이전과 같은 연산을 수행하며, LLaDA의 업스트림 입력 denoiser 단계 테스트도 그대로입니다.

### 3.5 프롬프트 상한 (보안 리뷰)

공유 요청 검사는 빈 프롬프트만 거부했고 JSON 본문 한도는 매우 큰 프롬프트를 허용합니다. 프롬프트 길이는 20층 MoE prefill, 보관되는 21개 hidden state, 매 step의 DiT 캡션 스트림을 키우므로 요청 하나가 통합 메모리를 소진할 수 있었습니다. 이제 무엇이든 읽기 전에 256 KiB, 템플릿 적용 후 8192 토큰으로 제한합니다. 모델이 자체 토크나이저를 보유해 33 GB 인코더를 읽기 전에 검사합니다.

## 4. 검증

- 최종 head의 좁은 테스트: `imagegen::` 440 통과 / 40 무시, `models::bailing_moe` 85, `models::registry` 19, `server::routes::images` 18, `image_worker` 14.
- 오라클 디렉터리로 최종 head에서 실제 체크포인트 무시 패리티 테스트: 4개 모두 통과, 수치 변화 없음.
- 범위 clippy, `cargo fmt --check`, 계약 `make verify-*` 타깃, `check_binary_assets.py`, `check_env_var_docs.py`: 통과.
- CLI: 4-bit 포스터는 기본 12 steps로 파란 배경에 읽을 수 있는 "HELLO"를 그리고, RGBA 프롬프트는 4채널 PNG를 쓰며, `--guidance 3`은 guidance 1.0 메시지로 실패하고, `-n 2`는 seed 0과 1을 단일 실행과 픽셀 단위로 같게 씁니다. bf16과 8-bit도 실행됩니다.
- 서버: `/v1/images/generations` 200, RGBA 유지(CLI 실행과 픽셀 단위 동일), `/v1/images/edits`는 task를 명시한 400, 잘못된 guidance는 400.

## 5. 남은 과제와 위험

- gen 포스터 케이스는 메인테이너 결정으로 끝단 간 30 dB 미만입니다. 이를 메우려면 메모리를 더 쓰는 f32 컨디셔너 경로가 필요합니다.
- 첫 요청 이후 텍스트 인코더를 요청마다 디스크에서 다시 읽습니다(Qwen-Image와 같은 설계). 그래서 최대 메모리에 모든 구성 요소가 포함됩니다.
- 알 수 없는 `--gen-kwargs` 키는 거부하지 않고 무시합니다.
- 공유 이미지 워커는 요청이 대기 중일 때만 응답 기한을 확인합니다.
- `ming_image::weights::dequantize`가 Z-Image의 역양자화 루프를 반복합니다. 양자화 트랜스포머는 로드 시 역양자화됩니다.
- 무시 패리티 단언은 `MLXCEL_MING_IMAGE_ORACLE_DIR`이 없으면 건너뜁니다.
- 공유 Z-Image 블록의 `FinalLayer`는 여전히 fused `fast_layer_norm`을 써서 MLX CUDA LayerNorm 경쟁 조건(#2696)에 노출됩니다. 여기서 반복 실행은 안정적이었습니다.

## 6. 학습 포인트

- 느슨한 끝단 간 지표는 고장 난 구성 요소를 올바른 것보다 낫게 평가할 수 있습니다. 결함이 드러나는 단계(여기서는 컨디셔너)에서 판정하고, 고장 경로가 실패함을 단언하십시오.
- 실패를 버그라 부르기 전에 업스트림의 노이즈 바닥을 재십시오. 측정한 드리프트만큼 업스트림을 교란해 자기 자신과 비교하면 30 dB가 애초에 도달 가능한지 알 수 있습니다.
- 한 seed에서의 모델 동작(여기서는 투명도)을 기술하는 수용 기준은 먼저 업스트림으로 확인해야 합니다. 이식 가능한 형태는 충실도 지표(알파 PSNR)입니다.
