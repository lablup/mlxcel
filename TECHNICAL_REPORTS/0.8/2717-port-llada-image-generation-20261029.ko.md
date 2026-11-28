# PR #2717: LLaDA-Image 텍스트, VQ, 편집 생성 포팅

**날짜**: 2026-10-29
**상태**: CUDA(GB10)에서 완료, 패리티 예외 승인됨. Metal과 ROCm은 실행 안 함
**언어**: Rust, Python(오라클 생성기), Markdown
**위험도**: 중간(새 패밀리이며 공유 코드 변경은 opt-in이지만, 자체 피처 end-to-end 패리티 목표는 미달)

## 요약

PR #2717(이슈 #2346 종료, 에픽 #2348)은 mlx-vlm `llada_image`(git 73d366ec)를 `inclusionAI/LLaDA-Image-Turbo`와 `inclusionAI/LLaDA-Image`용으로 포팅합니다. 텍스트-이미지, VQ(LLaDA-2 prior가 이산 이미지 토큰을 생성하고 SigVQ가 임베딩), 단일 이미지 편집의 세 모드를 추가하며 `mlxcel generate-image`, `/v1/images/generations`, `/v1/images/edits`, probe, 레지스트리에 연결됩니다.

mlxcel 자체 피처 기준 end-to-end 30 dB PSNR 기준과 VQ 토큰 id 완전 일치는 충족하지 못했고, 메인테이너가 이 차이를 문서화된 예외로 승인했습니다. 자체 피처 PSNR은 16.10 dB(Turbo 텍스트), 18.02 dB(base 텍스트), 26.71 dB(Turbo 편집), 22.24 dB(base 편집), 14.34 dB(Turbo VQ)이며 VQ id는 1024개 중 254개가 일치합니다. 대신 적용되는 게이트는 다음과 같습니다. 업스트림의 캡션/시맨틱 피처를 입력으로 한 end-to-end 테스트의 40~50 dB 경계(측정 44.79~58.45 dB), 텍스트 인코더 레이어별 정확도, 비트 단위로 같은 노이즈 추출, 반드시 실패해야 하는 오포팅 대조군입니다.

## 1. 문제 정의

LLaDA-Image는 30개 블록 DiT를 LLaDA-2 MoE 텍스트 인코더의 피처(QueryFormer와 텍스트 프로젝션 경유)로 조건화하고, 선택적으로 블록 확산 샘플링이 만든 SigVQ 토큰으로, 편집에서는 원본 이미지의 VAE latent로 조건화합니다. #2348이 이미지 생성을 범위에 넣었고, 두 체크포인트는 mlxcel에서 인식되지 않았습니다.

재사용 가능한 기존 요소는 `llada2_moe` 모델과 샘플러 헬퍼, 공유 `dit_blocks`, FLUX.2 VAE입니다. LLaDA-Image 설정, QueryFormer, SigVQ, VQ 샘플러, 스케줄러, 트랜스포머 연결, 파이프라인은 없었습니다.

어려운 부분은 구조가 아니라 수치입니다. 이 파이프라인은 카오스적이어서 MoE 라우팅 전환과 샘플러가 ulp 미만의 차이를 증폭하므로, 올바른 포팅도 업스트림 PNG와 크게 달라질 수 있습니다. 그래서 포팅 결함과 증폭된 반올림 드리프트를 분리하는 패리티 설계가 필요했습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 패밀리 모듈 | 새 `src/imagegen/llada_image/`: `config`(로드 시 shape/dtype 검사), `conditioning`(QueryFormer, 텍스트 프로젝션), `sigvq`(ViT 인코더, prior 프로젝터), `vq_sampling`(`llada2_moe::generate` 헬퍼를 재사용하고 argsort 폴백 top-count 헬퍼를 추가한 블록 확산 샘플러), `transformer`(`dit_blocks` 기반), `scheduler`, `weights`, `pipeline`, `mod`. 각각 `*_tests.rs`가 있고 `parity_tests.rs`, `transformer_stage_tests.rs`가 추가됨. |
| 스케줄러 | Turbo: 균일 sigma, shift 3, 확률적 재노이즈, 4 스텝, guidance 1. Base: f64 비선형 스케줄, shift 1, 50 Euler 스텝, guidance 5. 업스트림의 전역 RNG 추출 순서 유지. |
| 텍스트 수치 | `Llada2MoeModel`에 opt-in `Llada2Numerics::LladaImage` 경로 추가(`image_numerics.rs`, `forward_embeds`): gain 이전 bf16 반올림 RMSNorm, 행별 위치를 쓰는 bf16 반올림 RoPE 테이블, 점수 내림차순 expert 정렬, down projection 이전 라우팅 점수 적용, f32 expert 합. 기본 텍스트 경로는 변경 없음. |
| GELU | QueryFormer, 텍스트 프로젝션, SigVQ는 MLX가 컴파일하는 `nn.gelu_approx`, `nn.gelu` 그래프를 복제한 mlxcel-core의 컴파일된 구현을 사용. eager 연산 체인은 bf16 출력의 0.8%를 다르게 반올림했음. |
| `dit_blocks`(추가형) | 편집용 weightless 블록 norm과 토큰별 modulation, opt-in `BiasAdd::Fused`(업스트림 `nn.Linear`처럼 `addmm` 한 번), opt-in `MRoPE::compute_freqs_on_device`. Z-Image, Mage-Flow, Ming-Image는 기존 경로 유지. |
| 요청 처리 | 옵션 `generation_mode`, `stochastic_sampling`, `negative_prompt`. 잘못된 요청은 `InvalidImageRequest`(HTTP 400). 편집은 `MAX_EDIT_IMAGE_TOKENS` = 65536 denoiser 이미지 토큰, VQ 프롬프트는 `MAX_VQ_PROMPT_TOKENS` = 2048로 제한. 둘 다 가중치 로드 전에 검사. |
| 등록과 문서 | 이미지 레지스트리, probe(`LLaDAImagePipeline`)와 probe 테스트, 아키텍처 레지스트리, 패리티 테스트용 환경 변수 허용 목록. `docs/image-generation.md`에 LLaDA-Image 섹션, `docs/supported-models.md`에 행 추가. |
| 오라클 도구 | `tests/fixtures/generate_llada_image_oracle.py`, `generate_llada_image_denoiser_stages.py`. 케이스가 전체 해상도라 픽스처는 커밋하지 않음. |

규모: 파일 40개, +6589 / -47.

## 3. 기술적 선택과 그 이유

### 3.1 패리티: 승인된 예외와 대체 게이트

**배경:** 같은 시드에서 mlxcel 전체 파이프라인은 이슈가 요구한 자체 피처 30 dB PSNR에 도달하지 못하며 VQ id도 완전히 일치하지 않습니다. 메인테이너가 이를 예외로 승인했습니다(에픽 #2348). 아래 내용은 그 근거이며 수치가 좋다는 주장이 아닙니다.

오라클: 업스트림 mlx-vlm `73d366ec`, mlx 0.32.4 CUDA(GB10), 양쪽 `MLX_ENABLE_TF32=0`, seed 42, 프롬프트 "A red panda serving tea in a moonlit bamboo forest". 편집은 체크포인트의 `assets/edit_demo.jpg`(1440x704)와 "make it blue"를 사용합니다. "자체(own)"는 mlxcel 전체 파이프라인, "업스트림 피처"는 업스트림의 캡션/시맨틱 피처를 mlxcel의 denoiser와 VAE에 넣은 경우입니다.

| 케이스 | 첫 예측 rel-RMS | 자체: latent / PSNR | 업스트림 피처: latent / PSNR | 경계(업스트림 피처) |
|---|---|---|---|---|
| Turbo 텍스트 1024 | 0 | 0.882 / 16.10 dB | 0 / 57.97 dB | 5e-3 / 50.0 dB |
| Turbo VQ 512 | 3.00e-3 | 1.087 / 14.34 dB | 0.0454 / 44.79 dB | 0.08 / 40.0 dB |
| Turbo 편집 1440x704 | 0 | 0.154 / 26.71 dB | 0 / 56.21 dB | 5e-3 / 50.0 dB |
| Base 텍스트 1024 | 0 | 0.623 / 18.02 dB | 0 / 57.94 dB | 5e-3 / 50.0 dB |
| Base 편집 1440x704 | 0 | 0.547 / 22.24 dB | 0 / 58.45 dB | 5e-3 / 50.0 dB |
| Base VQ 512 | 오라클 없음 | | | |

Base VQ는 업스트림이 mlx 0.32.4의 CUDA 그래프 코드에서 두 번 실패(`cudaGraphAddKernelNode ... invalid argument`)해서 오라클이 없습니다. mlxcel 자체 base VQ 실행은 완료됩니다.

**`parity_tests.rs`에서 단언하는 게이트:**

| 게이트 | 요구 | 측정 |
|---|---|---|
| 모든 프롬프트의 토큰 id | 정확히 일치 | 일치 |
| 초기 노이즈와 확률적 재노이즈 추출 | 비트 단위 동일 | 동일 |
| 업스트림 hidden state에 대한 텍스트 프로젝션 | 1e-4 미만 | 0 |
| 텍스트 백본 레이어 0, 1 (각각 업스트림 입력 사용) | 2e-5 미만 | 0, 7.7e-6 |
| 업스트림 VQ id에 대한 SigVQ 임베딩 | 1e-2 미만 | 0 |
| 편집 원본 latent | 2e-2 미만 | 8.8e-3 |
| 편집 SigVQ 코드북 id 일치율 | 97% 이상 | 973/990 |
| 첫 예측 | 1e-3 미만(텍스트, 편집), 1e-2 미만(VQ) | 0(텍스트, 편집), 3.0e-3(VQ) |
| VQ prior 윈도 위치 | 업스트림과 동일 | 동일 |
| 업스트림 피처 end-to-end | 위 표의 경계 | 오라클 다섯 케이스 모두 충족 |

게이트 없이 보고만 하는 값: 캡션 피처 rel-RMS 1.0e-2~1.4e-2, 첫 VQ 윈도 hidden state 0.171.

**자체 피처 격차가 포팅 결함이 아닌 이유:**

- 티처 포싱(각 백본 레이어에 업스트림 자신의 입력을 넣음)에서 attention과 dense 레이어는 20개 레이어 전부 비트 단위로 같고, 모든 MoE 레이어가 모든 토큰을 같은 expert로 같은 순서로 라우팅합니다. MoE 출력의 0.6%가 한 번의 반올림만큼 다른 이유는 mlxcel의 expert 정렬 `gather_mm`이 오라클의 비정렬 호출과 다른 순서로 누적하기 때문입니다.
- 비정렬 호출을 쓰면 20개 레이어가 전부 비트 단위로 같지만, 프롬프트는 여전히 28행 QueryFormer `addmm`을 거치며 드리프트합니다. 이 빌드의 MLX와 mlx 0.32.4의 GEMM 커널이 다르기 때문입니다(57344개 값 중 40개). 비정렬 호출로 측정한 값(RoPE 수정 전이라 기존 6.0e-2 denoiser 잔차 포함): Turbo 텍스트 18.5 dB, Turbo 편집 36.0 dB, VQ id는 처음 48개 토큰 일치(377/1024), VQ prior 시간은 101 s 대신 206 s.
- 업스트림도 자기 섭동을 같은 방식으로 증폭합니다. x_embedder의 bias add를 분리하면 첫 예측이 8.6e-2 움직이고 PNG는 자기 자신 대비 26.5 dB가 됩니다. QueryFormer query에 2^-9 섭동을 주면 자기 대비 17.9 dB입니다. 텍스트 백본 레이어 2의 bf16 1 ulp는 MoE 라우팅이 뒤집혀 최종 hidden state를 0.167 rel-RMS 움직이며, mlxcel과 업스트림 사이는 0.172입니다.
- VQ id는 seed 42, 512x512에서 1024개 중 254개가 같고 처음 6개가 동일합니다. 카오스는 첫 VQ 윈도에서 들어옵니다. 업스트림 id를 쓰면 SigVQ와 denoiser가 게이트 안에서 업스트림을 재현합니다(44.79 dB).

**결정:** 업스트림이 호출하는 정렬 `gather_mm`을 유지하고(3.2 참고) 자체 피처 드리프트를 수용하며, 위 게이트를 승인 기준으로 삼습니다.

### 3.2 업스트림 MLX에서는 틀린 정렬 `gather_mm`을 유지

업스트림 `TextMoE`는 `gather_mm(sorted_indices=True)`를 호출합니다. mlx 0.32.4 CUDA에서 이 호출은 잘못된 행을 반환합니다(비정렬 호출 및 행별 matmul 대비 rel-RMS 0.80, 랜덤 가중치에서도 동일, CPU f32는 정상). 따라서 그 MLX에서 업스트림 텍스트 인코더는 첫 MoE 레이어부터 틀립니다. mlxcel의 MLX pin에는 이 버그가 없습니다(정렬 경로가 f32 CPU와 6e-3 이내). 오라클은 올바른 기준을 얻기 위해 `--unsorted-gather`로 생성하고, 포팅은 업스트림이 실제로 호출하는 정렬 호출을 유지합니다. 비정렬 호출은 자체 피처 PSNR을 개선하지만(3.1 참고) VQ prior 시간이 두 배가 되고 업스트림과 달라집니다.

### 3.3 두 번째 오라클 보정: VAE `conv_in`

업스트림 `_match_conv_layout`은 VAE 인코더 `conv_in`의 shape이 두 레이아웃 모두 [128,3,3,3]이라 전치하지 않습니다. 그 결과 업스트림 VAE는 편집 원본을 17.98 dB로 왕복하며, 가중치를 전치하면 33.79 dB입니다. mlxcel은 올바르게 로드합니다. 오라클 플래그: `--fix-conv-in`. 결과적으로 base 편집 출력은 보정된 오라클과 mlxcel 모두 파란 물결 빛무늬 텍스처이고, 보정 안 된 업스트림만 파란 콜라주를 렌더링합니다.

### 3.4 세 개의 opt-in 스위치로 denoiser 정확도 확보

티처 포싱 스테이지 테스트(`transformer_stage_tests.rs`)는 denoiser의 각 스테이지에 업스트림 입력을 넣고 출력을 비교합니다. 수정 전 유일하게 비정확한 스테이지는 이미지 RoPE 테이블이었고(항목의 2.8%가 호스트 `powf` 경로 때문에 f32 반올림 한 번씩 어긋남), 이것만으로 첫 예측 6.03e-2가 생겼습니다. 업스트림처럼 역주파수를 디바이스에서 계산하자 모든 스테이지와 Turbo 텍스트 전체 패스가 비트 단위로 같아졌고, 업스트림 피처 PSNR이 29.58 dB에서 57.97 dB로 올랐습니다. `BiasAdd::Fused`는 첫 예측을 0.18로 움직이던 분리 add 경로를 고칩니다. 세 가지 모두 opt-in이므로 Z-Image와 Mage-Flow의 동작은 유지됩니다(dit_blocks 18, z_image 33, mage_flow 42 단위 테스트 통과).

남은 잔차는 VQ 512의 최종 프로젝션(`[1024, 3840] x [3840, 128]` bf16 `addmm`)이며, 입력이 같아도 이 MLX가 mlx 0.32.4와 다른 커널을 고릅니다(3.0e-3 rel-RMS, 출력의 38.6%). 같은 프로젝션을 4096행으로 하면 정확합니다. 코드 변경으로는 닿지 않습니다.

### 3.5 컴파일된 GELU 복제 구현

eager GELU 체인은 MLX의 컴파일된 그래프와 bf16 출력의 0.8%를 다르게 반올림했습니다. mlxcel-core의 컴파일된 복제 구현을 쓰자 텍스트 프로젝션이 정확해졌습니다(eager는 1.9e-3).

### 3.6 자원 경계와 패닉 없는 로딩

- 편집은 65536 denoiser 이미지 토큰으로 제한합니다(원본과 타깃 그리드, `2 * (H / 16) * (W / 16)`, Mage-Flow와 같은 값). 업스트림에는 제한이 없습니다. 검사는 어떤 컴포넌트도 로드하기 전에 `validate_dimensions`에서 수행하며 HTTP 400이 토큰 수와 한도를 알려줍니다. 4096x4096 편집(131072 토큰)은 거부, 2880x2880은 통과, 2912x2912는 거부, 4096x4096 생성은 영향 없습니다.
- 템플릿 포함 2048 토큰을 넘는 VQ 프롬프트는 가중치 로드 전에 거부합니다. VQ prior가 매 스텝 전체 윈도를 다시 계산하기 때문입니다.
- 텍스트와 편집 프롬프트는 업스트림처럼 2048 토큰에서 자릅니다.
- QueryFormer와 텍스트 프로젝션은 둘 다 로드된 뒤에 저장하므로 로드 실패가 반쯤 만들어진 상태를 남기지 않으며, 요청 경로는 `expect` 대신 에러를 반환합니다.
- 오류는 `InvalidImageRequest`(HTTP 400)와 평가 실패(HTTP 500)로 나뉩니다.

### 3.7 그 밖의 결정

- `n > 1`이면 이미지마다 트랜스포머를 다시 로드합니다. 업스트림의 순서이며 그대로 두었습니다.
- 편집 원본은 PIL LANCZOS가 아니라 `image` 크레이트의 Lanczos3(공유 `z_image::pipeline::reference_rgb`)로 리사이즈합니다.
- 오포팅 대조군은 텍스트 인코더 드리프트에 가려지지 않도록 업스트림 피처 위에서 실행합니다.

## 4. 검증

하드웨어: NVIDIA GB10, CUDA, `gpu-lock` 하에서 `--profile test-fast`.

단위 테스트(`--test-threads=1`): `imagegen::llada_image` 35 통과(6 ignored: 패리티, 스테이지, 체크포인트 토크나이저 테스트), `imagegen::dit_blocks` 18, `imagegen::z_image` 33, `imagegen::probe` 26, `models::llada2_moe` 23, `imagegen::mage_flow` 42.

실제 체크포인트 패리티 테스트(`--ignored`, `MLX_ENABLE_TF32=0`):

- `turbo_matches_upstream`: 통과, 255 s.
- `base_matches_upstream`: `7f8d27d5`에서 통과, 929 s. Base VQ는 오라클이 없어 건너뜀.
- `turbo_controls_are_rejected`: 통과, 140 s. 모든 대조군이 모든 케이스에서 경계를 벗어남:

| 대조군 | 텍스트 | VQ | 편집 |
|---|---|---|---|
| Seed 43 | 1.256 / 12.78 dB | 0.978 / 17.71 dB | 0.505 / 20.46 dB |
| Shift 3 대신 1 | 0.791 / 18.02 dB | 0.669 / 22.20 dB | 0.230 / 25.37 dB |
| RoPE 이후 FP32 q, k (괄호는 첫 예측) | 0.250 / 28.62 dB (8.43e-2) | 0.117 / 35.28 dB (6.69e-2) | 0.0152 / 48.85 dB (1.28e-2) |

- `denoiser_stages_on_upstream_inputs`: 통과, 40 s.
- `edits_past_the_image_token_bound_are_rejected`: 통과하며, 경계를 끄면 실패.
- `an_overlong_vq_prompt_is_an_invalid_request`, `prompt_ids_carry_the_grid_and_count`: 통과.

실제 체크포인트 CLI(seed 42, 전부 rc 0): Turbo 텍스트 1024 52 s(재실행 14 s, PNG 바이트 동일), Turbo VQ 512 84 s, Turbo 편집 1440x704 40 s, base 텍스트 1024 166 s, base VQ 512 131 s, base 편집 327 s. 서버(`--image-model`): Turbo와 base의 텍스트, VQ, 편집이 모두 200을 반환하고, 1000x1000 크기, generations 요청의 `generation_mode: "editing"`, 32의 배수가 아닌 편집은 400을 반환합니다. 헤드 `7c3becea`에서 서버 텍스트 이미지는 CLI PNG와 픽셀 단위로 같습니다.

기본 `llada2_moe` 텍스트 경로 회귀(`llada2.0-mini-preview-4bit`, `examples/logit_trace`, 256 토큰 x 4 청크, 이 브랜치 대 diff를 되돌린 같은 트리): 1024개 위치 중 불일치 0, 결정된 674개 위치 중 불일치 0, TSV 바이트 동일, perplexity 양쪽 7.6092.

그 밖의 확인: `make verify-*` 타깃, `check_binary_assets.py`, `cargo clippy ... -D warnings` 통과. 같은 CLI Turbo 실행 두 번은 PNG가 바이트 단위로 같습니다(#2696의 fused LayerNorm 비결정성은 이 파이프라인에 영향 없음).

## 5. 학습 포인트

- 라우팅이 카오스적인 파이프라인에서 포팅 자체 피처의 end-to-end PSNR은 정확성이 아니라 민감도를 측정합니다. 각 스테이지에 레퍼런스 입력을 넣는 티처 포싱과, 레퍼런스 피처 위에서 하류 스테이지를 돌리는 방식이 둘을 분리합니다.
- 패리티 게이트는 실패할 수 있음을 보여야 합니다. 대조군(시드, shift, FP32 RoPE)은 모두 경계를 벗어나며, FP32 RoPE는 이제 첫 예측 경계도 벗어납니다.
- 호스트와 디바이스의 부동소수점은 다릅니다. 호스트 `powf`로 RoPE 역주파수를 계산하면 테이블 항목 2.8%가 f32 반올림 한 번씩 바뀌고, 이것만으로 첫 예측이 6.0e-2 움직였습니다.
- 레퍼런스 구현도 자기 플랫폼에서 틀릴 수 있습니다(정렬 `gather_mm`, `conv_in` 레이아웃). 오라클을 정답으로 쓰기 전에 독립 계산으로 확인해야 합니다.
- bf16에서 bias add를 합쳤는지(`addmm`) 분리했는지(matmul 후 add)는 눈에 띄는 수치 차이를 만듭니다. 레퍼런스의 연산 단위를 그대로 따라야 합니다.
- eager 형태의 같은 GELU 수식이 bf16에서 MLX의 컴파일된 그래프와 같지는 않습니다.

## 6. 후속 조치와 한계

- 자체 피처 30 dB PSNR과 VQ id 완전 일치는 미충족입니다(승인된 예외). 격차를 줄이려면 28행 QueryFormer `addmm`의 GEMM 커널 선택과 정렬/비정렬 MoE 누적 순서를 비트 호환으로 맞춰야 하는데, 이 PR에서는 바꿀 수 없습니다.
- Base VQ는 mlx 0.32.4의 업스트림 CUDA 그래프 실패가 해결될 때까지 오라클이 없습니다.
- VQ 512 최종 프로젝션 GEMM 잔차(3.0e-3)에는 코드 수정 방법이 없습니다.
- EXIF 방향: 업스트림은 편집 원본에 `ImageOps.exif_transpose`를 적용하지만 mlxcel의 공유 로더는 어떤 패밀리에도 적용하지 않습니다.
- 편집 원본 리사이즈는 PIL LANCZOS 대신 `image` 크레이트의 Lanczos3를 씁니다(서버 편집은 PIL 픽셀로 돌린 패리티 실행 대비 43.8 dB).
- 텍스트 백본 레이어 게이트는 백본 레이어 텐서로 오라클을 재생성한 `turbo_text_1024`에서만 실행되며, 다른 케이스는 재생성 전까지 건너뜁니다.
- 실행 안 함: Metal, ROCm.

## 참고 자료

- 이슈 #2346, 에픽 #2348, #2696(fused LayerNorm 비결정성), 메인테이너 결정 #932(패리티 계약).
- 관련 포팅: PR #2703(Ideogram 4), PR #2688(Mage-Flow).
- 업스트림: mlx-vlm `llada_image`, `73d366ecf9233b3fc6549531a7f2386726d09ff6`.
- 문서: `docs/image-generation.md`(LLaDA-Image 섹션).
