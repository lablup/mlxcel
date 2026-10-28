# PR #2687: ERNIE-Image 텍스트-이미지 생성 및 SDEdit 포팅

**날짜**: 2026-10-28
**상태**: CUDA(GB10)에서 완료, Metal 및 ROCm 실행 기록 없음
**언어**: Rust, Python(픽스처 생성기), Markdown
**위험도**: 중간(동작 중인 Ministral 3 텍스트 모델의 출력이 바뀜)

## 요약

PR #2687(이슈 #2345, 에픽 #2348)은 mlx-vlm `models/ernie_image`(git 73d366ec)를 `ernie_image` 이미지 생성 패밀리로 포팅합니다. `baidu/ERNIE-Image-Turbo`(8 스텝, guidance 1, CFG 없음)와 `baidu/ERNIE-Image`(50 스텝, guidance 4, CFG)에 대해 텍스트-이미지 생성과 SDEdit img2img를 지원합니다. 디코딩은 공유 FLUX.2 VAE로 하고, 원본 Diffusers 트리를 바로 로드합니다. `imagegen::FAMILIES`와 모델 레지스트리에 등록되어 `mlxcel generate-image`, `mlxcel-server --image-model`(두 이미지 라우트 모두), `mlxcel arch --json`에서 보입니다. 텍스트 컨디셔너와 선택 사항인 `pe/` 프롬프트 인핸서는 복사본 대신 새 진입점 `Ministral3Model::forward_hidden_states`를 통해 트리 내 Ministral 3 디코더를 실행합니다.

이 PR은 공유 코드의 버그도 고칩니다. 모든 Ministral 3 체크포인트가 YaRN rope scaling을 선언하지만 트리 내 모델은 이를 무시했습니다. 이 수정으로 Ministral 3, Mistral 3 VLM 텍스트 타워, Nemotron-3-Embed의 출력이 바뀝니다. 업스트림 대비 Turbo 케이스는 비트 단위로 동일하고(57.00 dB), Turbo edit과 base 8 스텝 대조군은 에픽의 30 dB 기준을 통과하며, base 50 스텝 케이스는 24.24 dB입니다. 사용자는 이 케이스를 에픽 기준의 예외로 승인했습니다.

## 1. 문제 정의

#2341 이후 이미지 런타임, FLUX.2 VAE, latent packing은 있었지만 ERNIE-Image 체크포인트에는 패밀리가 없어 로드 단계에서 실패했습니다. 빠져 있던 것은 36층 single-stream DiT(hidden 4096, FFN 12288, 32 head, AdaLN modulation, 텍스트 패딩을 가리는 key mask가 있는 3축 RoPE), Ministral 3의 마지막 층을 빼고 실행하며 final norm을 건너뛰는 컨디셔너, f32 Euler 스텝을 쓰는 static-shift(4.0) flow-match 스케줄러, `image_strength`가 (0, 1] 범위(기본 0.6)인 SDEdit, 업스트림이 다시 쓴 프롬프트를 `revised_prompt`로 돌려주는 프롬프트 인핸서입니다.

`src/models/ministral3.rs`에는 마지막 층 전에 멈추는 출력이 없어 컨디셔너로 쓸 수 없었습니다. 포팅 중에 두 번째 결함도 드러났습니다. `Attention`은 `rope_theta`로 일반 RoPE 회전만 했고 체크포인트의 `rope_parameters.rope_type: yarn`을 무시했지만, mlx-lm과 업스트림 텍스트 인코더는 `YarnRoPE`를 적용합니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 패밀리 모듈 | 새 `src/imagegen/ernie_image/`(`config`, `blocks`, `transformer`, `text_encoder`, `prompt_enhancer`, `scheduler`, `pipeline`, `weights`, `mod`)가 Generate와 Edit에 대해 `ImageGenerationModel`을 구현합니다. 요청 오류는 `InvalidImageRequest`(400)입니다. |
| DiT 블록 | Z-Image `dit_blocks`가 아닌 별도 `blocks.rs`: ERNIE는 회전 각도를 인접 원소에 반복해 두면서 half-split `rotate_half`로 회전하고, modulation과 gate를 f32 구간에서 계산한 뒤 스트림 dtype으로 되돌립니다. |
| 컨디셔너 | `text_encoder.rs`는 `Mistral3Model` 체크포인트에서 `language_model.*` 타워만 로드하고 `forward_hidden_states(num_layers = layers - 1, final_norm = false)`를 호출합니다. 프롬프트는 `MAX_TEXT_TOKENS` = 2048에서 잘립니다. |
| 프롬프트 인핸서 | `prompt_enhancer.rs`: JSON 사용자 턴 `{"prompt", "width", "height"}`을 `pe_tokenizer`의 chat template으로 렌더링하고, temperature 0.6, top-p 0.95, 최대 2048 새 토큰, 이미지마다 seed를 줍니다. 처음 쓸 때 로드되고(약 7 GB) 이후 상주합니다. |
| 스케줄러 | 별도 `scheduler.rs`: static shift 4.0 아래 `linspace(1, 0, n + 1)[:-1]`. 공유 `FlowMatchEulerDiscrete::static_shift`는 `linspace(1, 1/n, n)`을 만들어 50 스텝에서 f32 마지막 비트가 다르고, 스텝을 샘플 dtype으로 계산합니다. |
| Ministral 3 | 층 수와 캐시 수를 검사하는 `forward_hidden_states(input_ids, caches, num_layers, final_norm)`를 추가했고, `forward_with_caches`는 이 본체 뒤에 LM head를 붙인 형태가 되었습니다. `forward`와 `forward_batched`에서 YaRN 주파수와 mscale을 적용하며, 기존 `gpt_oss_common::yarn_rope_freqs`를 씁니다(Ministral 3의 기본 `truncate`는 true). |
| 등록과 문서 | `imagegen::FAMILIES`, 레지스트리 행, `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md`, `docs/embeddings.md`(Nemotron-3-Embed 안내). 테스트 전용 `MLXCEL_ERNIE_IMAGE_*` 변수를 `scripts/ci/check_env_var_docs.py` 허용 목록에 추가했습니다. |
| 픽스처 | `tests/fixtures/imagegen/ernie_image_parity.safetensors`(노이즈 체크섬과 앞부분, 8x8 풀링한 latent와 이미지)와 생성기 `tests/fixtures/generate_ernie_image_fixtures.py`. |

머지 전 헤드 `154f618a` 기준 통계: 41개 파일, +4241 / -30. 주요 커밋: `ea90089f`(hidden-state 진입점), `5072fc06`(`forward`의 YaRN), `a8a70102`(포팅), `a7352a95`(배치 디코드의 YaRN), `d25471d8`(문서), `cf1ed59f`(8 스텝 CFG 대조군), `148cef1d`(Nemotron-3-Embed 안내), `eda88684`(인핸서 입력 상한), `7f1d8ef6`(`// Used by:` 주석), 그리고 `origin/main` 머지 커밋들.

## 3. 기술적 선택과 그 이유

### 3.1 parity 기준과 base 예외 승인(사용자 결정, 에픽 #2348)

이슈는 처음에 최종 latent 상대 RMS 1e-2 미만을 요구했습니다. 에픽 #2348 진행 중 사용자가 이를 에픽 전체 기준으로 바꿨습니다. 그 기준은 잘못된 포팅을 걸러내는 케이스별 latent 상한, CUDA oracle 대비 PNG PSNR >= 30 dB, 비트 동일한 초기 노이즈입니다. 측정한 rel-RMS는 drift로 기록할 뿐 머지를 막지 않습니다.

Oracle: mlx-vlm 0.7.8(git 73d366ec), mlx 0.32.4, CUDA 백엔드, NVIDIA GB10, 양쪽 모두 `MLX_ENABLE_TF32=0`, 1024x1024, seed 42, 프롬프트 "A red panda serving tea in a moonlit bamboo forest", 인핸서 끔.

| 케이스 | 첫 forward rel-RMS | 최종 latent rel-RMS(drift) | PNG PSNR | 테스트 상한(latent / PSNR) |
|---|---|---|---|---|
| Turbo, 8 스텝, guidance 1 | 0 | 0 | 57.00 dB | 1e-2 / 30 dB |
| Turbo edit, guidance 3, strength 0.7 | 4.90e-2 | 1.35e-1 | 32.23 dB | 2.5e-1 / 30 dB |
| Base, 50 스텝, guidance 4 | 5.02e-3 | 3.03e-1 | 24.24 dB | 4.5e-1 / 20 dB |
| Base 대조군, 8 스텝, guidance 4 | 5.02e-3 | 1.64e-1 | 31.43 dB | 2.5e-1 / 30 dB |

초기 노이즈는 모든 케이스에서 비트 단위로 같고 컨디셔닝도 정확히 일치합니다. 57 dB는 동일한 latent를 PNG로 왕복한 값입니다.

base 50 스텝 케이스는 30 dB 기준에 미달합니다. 사용자는 이 케이스를 20 dB 테스트 상한 아래 24.24 dB로 예외 승인했고, 이슈 #2345의 acceptance revision에 기록되어 있습니다. 원인은 GEMM 하나입니다. CFG에서는 transformer가 batch 2로 돌고, 작은 M의 bf16 `text_proj`(`[2, 13, 3072] x [3072, 4096]`)가 mlxcel이 고정한 MLX와 mlx 0.32.4 wheel에서 서로 다른 커널을 골라 마지막 비트가 다르게 반올림됩니다. 36층을 50 스텝 거치면서 그 차이가 증폭됩니다. 근거는 다음과 같습니다.

- 8 스텝 대조군은 같은 체크포인트, 프롬프트, seed, CFG batch-2 forward로 31.43 dB에 도달합니다. 따라서 배치 forward 자체는 정상이고, 미달은 50 스텝 증폭에서 옵니다.
- 업스트림도 자기 자신과 비교하면 비슷하게 움직입니다. 해당 projection만 행 단위로 실행하면 업스트림 base 케이스는 2.89e-1 / 23.70 dB, edit 케이스는 1.39e-1 / 32.14 dB가 됩니다. 업스트림의 batch-2 row 1도 batch 1의 같은 행과 5.3e-2만큼 다릅니다.
- 50 스텝 이미지는 구도가 같습니다(같은 판다, 찻잔, 찻주전자, 달이고 세부 묘사만 다름).

20 dB 상한은 크게 망가진 파이프라인만 걸러내며, 그것만으로는 잘못된 포팅을 검출하지 못합니다. 잘못된 포팅 검출은 첫 forward 상한(`MAX_FIRST_FORWARD` = 1e-1)과 나머지 세 케이스의 30 dB 기준이 맡습니다. GELU를 f32로 계산하고 bias를 matmul 뒤에 더하던 수정 전 포팅은 batch-1 Turbo에서 3.3e-1 / 24.8 dB(첫 forward 2.0e-1)로 벗어났습니다. 이 포팅은 base 상한은 통과하지만 Turbo 기준에서 걸러집니다.

### 3.2 batch 1에서 MLX `nn` 커널에 맞추기

batch 1에서 비트 단위로 일치시키려면 mlxcel 헬퍼 대신 MLX `nn` 레이어를 따르는 두 가지 선택이 필요했습니다. GELU는 입력 dtype으로 계산하고(`nn.gelu`, `mlxcel_core::gelu`는 f32로 계산), bias가 있는 linear는 `addmm` 하나로 계산합니다(`nn.Linear`, `Linear::forward`는 matmul 뒤에 bias를 더함). 이 두 수정으로 Turbo는 24.8 dB에서 완전 일치가 되었습니다.

### 3.3 #2341 규칙에 따른 정밀도

transformer는 업스트림처럼 모든 백엔드에서 bf16(`weights::TRANSFORMER_DTYPE`)을 유지합니다. f16 A/B에서 Turbo 케이스는 NaN latent와 9.54 dB가 나왔습니다. 텍스트 인코더와 인핸서는 텍스트 모델 정책인 `crate::models::load_text_weights`로 로드합니다. VAE는 `Flux2Vae::default_dtype`을 씁니다.

### 3.4 Ministral 3 재사용과 YaRN 동작 변경

업스트림은 컨디셔너용으로 Ministral 3 디코더를 다시 구현합니다. 이 포팅은 이슈 요구대로 `forward_hidden_states`를 통해 `Ministral3Model`을 재사용합니다. 리팩터 자체는 동작을 바꾸지 않습니다. `pe/`의 logit trace는 두 arm 모두 불일치 0/256, perplexity 40.1104입니다. 업스트림 인코더에는 트리 내 레이어가 적용하는 Llama 4 attention scale이 없습니다. 하지만 이 값은 `original_max_position_embeddings`(16384) 미만에서 정확히 1이고 프롬프트는 2048 토큰에서 잘리므로 두 구현은 같은 값을 계산합니다.

디코더를 공유하면서 YaRN 누락이 이미지 경로까지 이어졌습니다. PR은 컨디셔너 전용 경로를 두지 않고 공유 attention에서 고쳤으므로 동작 중인 텍스트 모델도 바뀝니다.

- **Ministral 3 텍스트 모델.** 컨텍스트 4096~8704 토큰에서 mlx-lm 대비 logit trace를 측정했습니다. 수정 arm은 argmax 불일치 0/1024, perplexity 46.5053(mlx-lm 46.5054)이고, 되돌린 arm은 불일치 12/1024, perplexity +0.173%입니다.
- **배치 디코드**(`a7352a95`). B1 대 B4는 불일치 1/128, 결정된 위치에서 0/20으로 반올림 수준입니다. 배치 수정을 되돌리면 `ministral3_yarn_native_isolation`이 실패합니다.
- **Mistral 3 VLM 텍스트 타워.** `text_config`의 `ministral3`을 통해 같은 attention을 쓰며, mlx-vlm의 `initialize_rope`와 일치합니다.
- **Nemotron-3-Embed.** 두 체크포인트 모두 YaRN(`factor` 16)을 NeMo 키 `apply_yarn_scaling: false`와 함께 선언하는데, Hugging Face Transformers 5.20은 이 키를 무시합니다. 임베딩은 이제 HF 레퍼런스를 따르며 이전 mlxcel 버전과 다릅니다. 이전 버전으로 만든 인덱스는 다시 임베딩해야 합니다(`docs/embeddings.md`). 호스트에 Nemotron-3-Embed 체크포인트가 없어서 이 변경은 측정하지 않았습니다.

`RopeParameters::yarn_freqs`는 mlx-lm `initialize_rope`를 따릅니다. 레거시 `type` 키가 `rope_type`보다 우선하고, 기본값은 mlx-lm 값(`beta_fast` 32, `beta_slow` 1, `mscale` 1, `mscale_all_dim` 0, `truncate` true)입니다. YaRN이 아닌 체크포인트는 기존 `fast_rope` 그래프를 그대로 씁니다.

### 3.5 프롬프트 인핸서 규칙과 4096 토큰 입력 상한

`use_prompt_enhancer`는 업스트림 규칙을 따릅니다. 명시한 값이 우선합니다. `pe/`가 없는 체크포인트에 `true`를 주면 업스트림 메시지와 함께 400입니다. 옵션이 없으면 `pe/`가 있을 때 텍스트-이미지는 항상 인핸서를 거치고, edit은 인핸서가 원본 이미지를 볼 수 없으므로 거치지 않습니다. 인핸서가 만든 텍스트가 각 이미지의 `revised_prompt`입니다. 이 필드는 항상 직렬화되며 프롬프트를 다시 쓰지 않았으면 null입니다. CLI는 `revised prompt:`를 출력합니다. 공개 체크포인트는 업스트림처럼 중국어로 다시 씁니다.

보안 리뷰에서 인핸서가 렌더링된 프롬프트 전체를 prefill한다는 점이 드러났습니다. `pe/`는 262144 토큰 컨텍스트를 선언하고, JSON 본문 상한은 메가바이트 단위를 허용하며, 인핸서는 기본으로 켜져 이미지마다 한 번 돕니다. 그래서 요청 하나로 7 GB 모델에 이미지마다 전체 컨텍스트 prefill을 강제할 수 있었습니다. 커밋 `eda88684`는 렌더링된 입력을 컨디셔너 상한 2048의 두 배인 `MAX_ENHANCER_INPUT_TOKENS` = 4096으로 제한합니다. 더 긴 프롬프트는 400이고, 메시지는 컨디셔너가 대신 잘라 주는 `use_prompt_enhancer: false`를 안내합니다. JSON 사용자 턴이 문자열 중간에서 잘리지 않도록 자르지 않고 거부합니다. 테스트 `an_overlong_enhancer_input_is_the_callers_error`가 이를 검증합니다.

## 4. 검증

- CLI Turbo, 인핸서 자동: 40.8초, 중국어 revised prompt, 정상 이미지. CLI base 기본값: 99.4초. CLI edit(strength 0.7): 30.3초, 같은 구도에 눈이 추가됨. 원본 이미지 두 장: exit 1.
- 서버 생성: 인핸서 자동은 39.8초에 200과 `revised_prompt`, 인핸서 끔은 13.3초에 200, `revised_prompt` null, 이미지는 parity 출력과 같음(57.00 dB). 서버 edit: 19.8초에 200, 이미지 두 장은 400, `image_strength` 1.5는 400.
- 메모리: 인코더, transformer, VAE가 약 24 GB 상주하고, 인핸서가 로드되면 7 GB가 더해집니다.
- 테스트: 새 ERNIE 단위 테스트 36개(모듈과 진입점이 없는 main에서는 컴파일되지 않음)와 무시 표시된 실제 체크포인트 테스트 4개(8 스텝 대조군을 포함한 parity 테스트 3개, 인코더 테스트 1개). 최종 헤드: `imagegen::` 257 통과, 19 무시, `models::ministral3` 9, `models::registry` 16. make verify 타깃, `check_binary_assets.py`, fmt, clippy `-D warnings`가 통과합니다. 실제 체크포인트 수치는 리뷰 커밋 전에 측정했고, 리뷰 커밋은 수치를 바꾸지 않았습니다.

## 5. 학습 포인트

- 패밀리가 공유 텍스트 모델을 재사용하면 포팅의 parity 작업에서 그 모델의 버그가 드러날 수 있습니다. 공유 코드에서 고치고, 이미지 경로뿐 아니라 동작 중인 각 사용처를 `docs/benchmarks.md`에 나온 teacher-forced logit trace로 측정해야 합니다.
- batch 1에서 비트 일치하는 케이스와 짧은 루프 대조군이 있으면 정상 forward와 루프 증폭을 구분할 수 있습니다. 50 스텝 미달이 포팅 오류가 아니라 반올림 누적이라는 것은 8 스텝 대조군으로 확인했습니다.
- PSNR 미달을 결함으로 보기 전에 업스트림을 자기 자신과 비교해야 합니다(행 단위 `text_proj`, B1 대 B2). 업스트림이 자기 자신과 비교해 얻은 23.70 dB가 이 케이스의 하한입니다.
- 서버가 기본으로 실행하는 선택적 생성 단계에는 후단의 잘라내기와 별개로 자체 입력 상한이 필요합니다.

## 6. 후속 조치와 한계

구현 리뷰와 보안 리뷰에서 남은 MEDIUM 항목:

- transformer 가중치는 존재 여부만 검사하고 `config.json` 대비 shape은 검사하지 않습니다. matmul 브리지는 그냥 `UniquePtr`를 돌려주므로, 형식이 잘못된 체크포인트는 500을 돌려주지 못하고 첫 요청에서 프로세스를 중단시킵니다. 다른 패밀리도 같은 방식입니다.
- ERNIE는 공유 16 MP 상한 외에 한 변의 길이 상한이 없습니다(FLUX.2는 2048). CFG를 쓰는 4096x4096에서는 스텝마다 약 135k 토큰입니다. OOM은 `try_eval`을 거쳐 500으로 나타납니다.
- 이미지 워커는 요청 시작 전에만 deadline을 검사하므로, 클라이언트가 떠난 `n=10` 인핸서 사용 base 실행도 계속 돕니다. 공유 워커 동작입니다.
- 커밋된 픽스처만으로는 edit 케이스, 첫 forward 검사, 8 스텝 대조군에 `MLXCEL_ERNIE_IMAGE_ORACLE_DIR`가 필요하므로, CI에서 재현 가능한 잘못된 포팅 검출기는 Turbo뿐입니다.
- Nemotron-3-Embed의 YaRN 변경은 측정하지 않았습니다.

그 외 낮은 등급으로 남은 항목: `random_seed(seed)`가 인핸서 호출마다 MLX 전역 RNG를 다시 seed합니다(업스트림과 `diffusion_worker`도 같음). EOS 집합에 `additional_special_tokens_ids`가 빠져 있습니다(이 체크포인트에는 없음). 잘라내기 전에 프롬프트 전체를 토크나이즈합니다. `imagegen/testing.rs`의 `rel_rms`와 `psnr`이 FLUX.2와 Z-Image 테스트의 사본과 중복됩니다.

#2666은 공유 Qwen3 컨디셔너에 대한 에픽의 미해결 정밀도 후속 이슈로, FLUX.2, Z-Image, Bonsai, Ideogram 4용 opt-in f32 attention을 추가합니다. ERNIE는 Ministral 3로 컨디셔닝하고 그 컨디셔닝은 이미 업스트림과 정확히 일치하므로, #2666은 ERNIE 수치를 바꾸지 않습니다. base 케이스의 차이는 batch-2 `text_proj` GEMM에서 오며, #2666의 범위가 아닙니다.

Metal과 ROCm 실행은 PR에 기록되어 있지 않습니다.

## 참고 자료

- 이슈 #2345, 에픽 #2348, 런타임 이슈 #2341(PR #2497), Z-Image PR #2662, 후속 이슈 #2666
- 업스트림: Blaizzy/mlx-vlm `mlx_vlm/models/ernie_image`, PR #1954, #2373
- `docs/image-generation.md`, `docs/embeddings.md`, `docs/benchmarks.md`
