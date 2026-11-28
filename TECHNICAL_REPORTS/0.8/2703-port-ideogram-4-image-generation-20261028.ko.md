# PR #2703: Ideogram 4 텍스트-이미지 포팅 (행 단위 FP8 가중치)

**날짜**: 2026-10-28
**상태**: CUDA(GB10)에서 완료, Metal은 Partial 표기(실행 안 함), ROCm 실행 안 함
**언어**: Rust, Python(픽스처 생성기), Markdown
**위험도**: 낮음(새 패밀리이며 공유 코드 변경은 추가형)

## 요약

PR #2703(이슈 #2427, 에픽 #2348)은 mlx-vlm `models/ideogram4`(git 73d366ec)를 라이선스 게이트가 있는 `ideogram-ai/ideogram-4-fp8` 스냅샷용 `ideogram4` 이미지 패밀리로 포팅합니다. FP8 트랜스포머 두 개(conditional, unconditional), 행 단위 FP8 역양자화, interleaved M-RoPE, 세 가지 샘플러 프리셋을 가진 logit-normal 스케줄러, JSON 캡션 래핑, 기존 Qwen3 텍스트 인코더를 통한 13개 레이어 스택 텍스트 피처 경로, FLUX.2 VAE 디코드를 포함합니다. `mlxcel generate-image`, `/v1/images/generations`, probe, 모델 레지스트리, `mlxcel arch --json`에 연결됩니다. 생성만 지원하며 `/v1/images/edits`는 400입니다.

업스트림 대비 512x512, `V4_TURBO_12`, seed 0에서 토큰과 초기 노이즈는 비트 단위로 같고, 풀링한 최종 latent는 rel-RMS 3.16e-2, PNG는 37.13 dB입니다. 남은 드리프트의 대부분은 공유 컨디셔너의 bf16 텍스트 타워에서 나옵니다(이슈 #2666). 업스트림의 텍스트 피처를 쓰면 PNG가 45.95 dB입니다. 실제 체크포인트는 GB10에서 CLI와 서버 모두 실행되며 업스트림과 일치하는 이미지를 만듭니다(이슈의 명령 기준 47.88 dB).

## 1. 문제 정의

Ideogram 4는 Qwen3 텍스트 타워의 hidden state 13개 레이어를 스택해 DiT를 컨디셔닝하고, 구조화된 JSON 캡션을 프롬프트로 사용하는 텍스트-이미지 모델입니다. 체크포인트는 두 트랜스포머와 텍스트 타워를 모두 FP8(`F8_E4M3` 바이트와 행 단위 f32 `weight_scale`)로 저장합니다. #2348로 이미지 생성이 범위에 들어왔고, #2341 런타임이 머지된 뒤에도 `ideogram-4-fp8` 스냅샷은 어떤 코드도 인식하지 못해 로드에서 실패했습니다.

트리에 대응물이 있던 것은 Qwen3 텍스트 인코더(`Qwen3TextEncoder::from_weights`), FLUX.2 VAE, FP8 바이트 디코더 `f8_e4m3_to_f32`였습니다. DiT 블록, M-RoPE 레이아웃, 스케줄러, 프롬프트 정규화, 행 단위 역양자화는 대응물이 없었습니다. 이슈는 업스트림 구현 대비 reference parity 기준과 실제 체크포인트 CLI 실행도 요구했습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 패밀리 모듈 | 새 `src/imagegen/ideogram4/`: `config`, `scheduler`, `prompting`, `latent_norm`, `weights`, `blocks`, `transformer`, `pipeline`, `mod`, 각각 `*_tests.rs`를 두고 `parity_tests.rs` 추가. generate만 지원하는 `ImageGenerationModel` 구현. |
| 스케줄러 | 세 프리셋(`V4_DEFAULT_20`, `V4_TURBO_12`, `V4_QUALITY_48`)을 가진 logit-normal 스케줄러. 업스트림 리터럴과 일치하도록 CPython의 AS241을 따른 역 정규 CDF. 기본 프리셋은 guidance 7, 마지막 두 스텝은 3. |
| 프롬프트 | JSON 캡션 패스스루, `auto_json_caption`이 켜진 경우 최소 캡션 래핑, 이스케이프, 캡션 경고. 래핑된 캡션은 `revised_prompt`로 반환. |
| 가중치 | `weights.rs`가 행 단위 FP8을 업스트림과 같은 반올림 순서로 bf16으로 역양자화(트랜스포머와 텍스트 타워 모두). 로드 시 shape과 dtype 검사. |
| 트랜스포머 | `blocks.rs`, `transformer.rs`: interleaved M-RoPE를 쓰는 DiT 블록, conditional과 unconditional 변형. unconditional은 `llm_cond_*`를 건너뜀. |
| 파이프라인 | 스냅샷 chat template 기반 토크나이저, 13개 레이어 스택 텍스트 피처, 전역 시드 노이즈, guidance 루프, FLUX.2 VAE 디코드. |
| 공유 코드(추가형) | `f8_e4m3_to_f32`의 `pub(crate)` 재노출, `fp8_block.rs`의 포인터 주석, `imagegen/vae/mod.rs`의 `Used by` 갱신. 공유 Qwen3 컨디셔너는 수정하지 않음. |
| 등록과 문서 | probe `IMAGE_FAMILIES`, 이미지 `FAMILIES`, `STANDALONE_ARCHITECTURE_FAMILIES`(별칭 `ideogram-4`, `ideogram-4-fp8`). `docs/image-generation.md`에 새 섹션, 정밀도 행, parity 표 추가. `docs/supported-models.md`, `docs/vision-model-scope.md`에 행 추가. |
| CI와 픽스처 | `ideogram4_parity.safetensors`(46 KB), `ideogram4_tiny_refs.safetensors`(61 KB)와 `tests/fixtures/generate_ideogram4_fixtures.py`. `scripts/ci/check_binary_assets.py`의 자산 예산, `scripts/ci/check_env_var_docs.py`의 `MLXCEL_IDEOGRAM4_ORACLE_DIR`, `MLXCEL_IDEOGRAM4_OUT` 허용 목록. |

규모: 34개 파일, +4818 / -8.

## 3. 기술적 선택과 그 이유

### 3.1 parity 기준 설계

**컨텍스트:** 이 PR 밖의 원인(bf16 텍스트 타워, #2666)으로 업스트림과 드리프트가 생기므로, 단일 end-to-end PSNR 임계값으로는 올바른 포팅과 잘못된 포팅을 구분할 수 없습니다.

**기준(CUDA, 오라클 mlx 0.32.4, mlx-vlm 0.7.8, 양쪽 `MLX_ENABLE_TF32=0`):**

| 측정 | 값 | 기준 |
|---|---|---|
| 토큰 id (45개) | 동일 | 정확 일치 |
| 초기 노이즈 | 비트 동일, FNV-1a `0xa671abe7` | 정확 일치 |
| 4x4 풀링 최종 latent rel-RMS | 3.16e-2 | 4.5e-2 이하 |
| 8x8 풀링 이미지 PSNR | 43.72 dB | 38 dB 이상 |
| 전체 PNG PSNR | 37.13 dB | 30 dB 이상 |
| 전체 최종 latent rel-RMS | 6.18e-2 | 기록만, 게이트 아님 |
| 텍스트 피처 rel-RMS | 6.40e-2 | 기록만, 게이트 아님 |
| 첫 스텝 velocity rel-RMS | cond 6.17e-3, uncond 3.17e-3 | 기록만, 게이트 아님 |
| 업스트림 텍스트 피처를 포트에 입력 | latent 2.87e-2, PNG 45.95 dB | 진단용 |

메인테이너의 #932 계약을 따릅니다. 연산별 동일 입력 parity와 greedy 정확성이 게이트이고, 누적된 중간 드리프트는 기록합니다.

**대조 실험:** 변형한 업스트림 실행 두 가지로 게이트가 실패할 수 있음을 확인했습니다. 잘못된 노이즈(seed 1)는 풀링 latent 4.2e-1, 12.5 dB, 12.3 dB로 모두 거부됩니다. 프리셋 스케줄을 무시(마지막 스텝 guidance를 3 대신 7)하면 풀링 latent 5.2e-2, 51.7 dB, 48.1 dB로 이미지 기준은 모두 통과합니다.

**결정:** 풀링 latent 상한을 리뷰에서 6e-2에서 4.5e-2로 조였습니다. 스케줄 대조 실험은 6e-2와 모든 이미지 기준을 통과했는데, 텍스트 타워 드리프트가 한 스텝의 guidance 오류보다 PSNR을 더 많이 깎기 때문입니다. 따라서 이런 종류의 버그는 latent 상한만이 구분합니다. 변경 후 대조(5.2e-2)는 거부되고 포트(3.16e-2)는 통과합니다. 양쪽 여유가 좁아서 수치를 PR과 문서에 남겼습니다.

### 3.2 요청 오류는 `InvalidImageRequest`

`size`와 종횡비, 빈 프롬프트, 1 MiB 초과 프롬프트, 잘못된 JSON 캡션, 알 수 없는 `sampler_preset`, `guidance_schedule` 길이, 2048 토큰 초과, `prompt_expansion_model`, 비어 있지 않은 `negative_prompt`, edit 작업은 모두 `InvalidImageRequest`로 매핑되어 서버가 400을 반환합니다. 잘못된 요청이 500으로 나타나지 않습니다.

비어 있지 않은 `negative_prompt` 거부는 의도적인 이탈입니다. 업스트림의 `generate_array`는 `**kwargs`를 받아 조용히 무시합니다. `prompt_expansion_model`은 포팅하지 않고 거부합니다.

### 3.3 리뷰 중 추가한 입력 보강

- JSON 파싱과 토큰화 전에 검사하는 1 MiB 프롬프트 바이트 상한. 어휘에서 가장 긴 토큰이 128바이트이므로 2048 토큰에 들어가는 프롬프트를 거부하지 않습니다.
- 캡션 경고는 선형 시간으로 수집하고 64개로 제한하며, 8개만 로그에 남기고 제어 문자는 이스케이프합니다.
- 오버플로 검사를 한 config 산술: `heads * head_dim`, `head_dim / 2`로 제한한 `mrope_section` 항목, 1..=4096 범위의 `num_layers`, 수집 전에 길이를 제한한 `guidance_schedule`.
- 로드 시 모든 트랜스포머 텐서의 shape과 dtype을 config와 대조. 불일치하거나 스케일이 없는 FP8 가중치는 첫 요청의 브리지 op 안 C++ 예외가 아니라 로드 오류가 됩니다.

### 3.4 메모리 수명주기는 업스트림을 따름

토크나이저는 `load()`에서 로드합니다. 텍스트 인코더(bf16 약 16 GB)는 요청마다 로드하고 트랜스포머 실행 전에 `clear_memory_cache`로 해제합니다. 두 트랜스포머와 VAE는 지연 로드되어 상주합니다(약 38 GB). unconditional 트랜스포머는 더 이상 `llm_cond_*`(여기서는 쓰이지 않는 약 0.5 GB)를 읽거나 역양자화하지 않으며, conditional은 여전히 필요로 합니다.

**시도 후 되돌림:** LLM 피처의 텍스트 행만 프로젝션하면 2048x2048에서 약 4 GB를 줄일 수 있습니다. 하지만 업스트림과 GEMM 행 수가 달라져 업스트림 피처 진단이 45.95 dB에서 31.8 dB로 떨어졌으므로 모든 행에 대해 프로젝션합니다. 후속 과제로 남겼습니다.

### 3.5 공유 컨디셔너는 건드리지 않음

텍스트 타워는 기존 `Qwen3TextEncoder::from_weights`를 거치며 그 안의 attention 정밀도는 수정하지 않았습니다. 이는 #2666이 담당하며, 여기서 바꾸면 패밀리 포팅과 컨디셔너를 쓰는 모든 패밀리의 변경이 섞입니다.

### 3.6 작은 결정들

- `num_images`는 공유 규칙대로 이미지마다 `seed + i`를 씁니다. 업스트림은 한 시드에서 배치 노이즈 텐서 하나를 뽑습니다. 문서화했습니다.
- 노이즈 단위 테스트는 CUDA와 CPU의 `erfinv`가 다른 510번 원소에서 1 ulp를 허용합니다. 전체 텐서의 비트 동일성은 ignore된 CUDA parity 테스트에서만 게이트합니다.
- Metal 실행 없이 레지스트리에 `Partial`로 표기했습니다.

## 4. 검증

모두 `MLX_CUDA_ARCHITECTURES=121`, `--profile test-fast --features cuda`, NVIDIA GB10, PR 본문 작성 시점의 최종 head 기준입니다.

- `gpu-lock` 아래 `imagegen:: models::registry server::image_worker --test-threads=1`: 377 통과, 0 실패, 24 무시(실제 체크포인트 필요).
- 새 테스트는 ignore되지 않은 36개와 ignore된 실제 체크포인트 parity 테스트 1개입니다. 세 프리셋의 두 해상도 스케줄러 리터럴, 프롬프트 래핑 리터럴과 캡션 경고, 업스트림 4x8 리터럴 대비 행 단위 FP8 역양자화, seed 0과 7의 노이즈 리터럴, 랜덤 bf16 가중치의 소형 2레이어 트랜스포머 대 업스트림(conditional과 unconditional 출력 rel-RMS 0, M-RoPE 테이블 동일)을 다룹니다.
- 회귀 확인: `IMAGE_FAMILIES`에서 `"ideogram4"`를 빼면 probe 테스트가 `left: Some("flux2") right: Some("ideogram4")`로 실패하고 라우팅 테스트도 실패합니다.
- `-D warnings` clippy는 깨끗하고, `make verify-*` 타깃과 `check_binary_assets.py`가 통과합니다.
- `MLX_ENABLE_TF32=0`으로 돌린 parity 테스트는 표의 모든 수치를 정확히 재현하며, 리뷰 커밋 이후에도 같습니다.

실제 체크포인트(`models/mlx/ideogram-4-fp8`):

- CLI, 이슈의 명령(1024x1024, seed 0, `V4_DEFAULT_20`): 종료 코드 0, 첫 실행 커널 컴파일 포함 3분 01초(업스트림 173초). 출력은 체크포인트 자체의 "Image blocked by safety filter" 플레이스홀더이며 업스트림과 47.88 dB로 일치합니다(최대 픽셀 절대 차이 27). 이 프롬프트와 시드에서 체크포인트가 그리는 결과이지 mlxcel의 필터가 아닙니다.
- 서버, 512x512, `V4_TURBO_12`, seed 1: 일반 텍스트 프롬프트는 43.4초에 200을 반환하며 `revised_prompt`가 업스트림의 래핑 캡션과 바이트 단위로 같습니다(이미지는 다시 플레이스홀더, 업스트림 대비 51.26 dB). `auto_json_caption: false`에서는 17.6초에 200, `revised_prompt: null`, 붉은 여우 두 마리의 수채화로 업스트림 대비 26.0 dB입니다(디테일 수준 차이이며, 텍스트 타워 탓이라는 판단은 turbo_512 진단에서 추정한 것이고 이 케이스에서 재실행하지 않았습니다). `/v1/images/edits`는 지원하지 않는 `edit` 작업을 밝히는 400을, `sampler_preset: "V4_FAST"`는 유효한 프리셋을 나열하는 400을 반환합니다.

## 5. 학습 포인트

- 행 단위 FP8은 저장 형식이지 연산 형식이 아닙니다. 체크포인트는 FP8 행렬의 각 행에 자기 f32 스케일을 곱하고, 결과를 고정된 순서로 bf16으로 반올림합니다. 업스트림의 반올림 순서를 맞춘 것이 소형 트랜스포머 테스트를 비트 동일하게 만들었습니다.
- parity 게이트는 실패할 수 있음을 보여야 합니다. 스케줄 대조 실험은 다른 구성요소가 더 큰 드리프트를 만들면 한 스텝의 guidance 오류가 모든 이미지 공간 지표를 통과함을 보였고, latent 상한만 이를 잡았습니다.
- 샘플러의 RNG를 재현하려면 같은 역 CDF가 필요합니다. CPython처럼 AS241을 구현하자 같은 시드에서 비트 동일한 노이즈가 나왔습니다.
- 실제 체크포인트 실행이 통과해도 좋은 데모라는 뜻은 아닙니다. 이슈가 요구한 프롬프트는 체크포인트의 안전 플레이스홀더를 유발하며, 업스트림이 같은 이미지를 그린다는 점은 포팅의 정확성도 확인해 줍니다.

## 6. 후속 조치와 한계

- #2666: 공유 Qwen3 컨디셔너의 bf16 attention 정밀도가 37 dB 대 46 dB 격차의 주된 원인입니다.
- 가능한 후속: 업스트림 GEMM 형태를 유지할 방법을 찾으면 LLM 피처의 텍스트 행만 프로젝션(2048x2048에서 약 4 GB 절감).
- 실행하지 않음: Metal, ROCm. `V4_QUALITY_48`은 end to end로 실행하지 않았고 스케줄만 업스트림 리터럴과 대조했습니다.
- 문서의 메모리 수치(상주 약 38 GB, 인코딩 중 약 54 GB)는 텐서 크기로 계산한 값이며 측정한 피크가 아닙니다. 2048x2048 피크는 그래프로 검토했을 뿐 측정하지 않았습니다.
- 텍스트 인코더는 업스트림처럼 요청마다 다시 로드합니다. 요청 간 재사용하면 약 16 GB 상주 메모리와 지연 시간을 맞바꾸게 됩니다.
- 이슈의 예시 프롬프트는 안전 플레이스홀더에 걸리므로 문서용 예시로는 적절하지 않습니다.

## 참고 자료

- 이슈 #2427, 에픽 #2348, 이슈 #2666(컨디셔너 attention 정밀도), 메인테이너 결정 #932(parity 계약).
- 관련 포팅: PR #2688(Mage-Flow), PR #2687(ERNIE-Image).
- 업스트림: mlx-vlm `models/ideogram4`, `73d366ecf9233b3fc6549531a7f2386726d09ff6`.
- 문서: `docs/image-generation.md`(Ideogram 4 섹션과 parity 표).
