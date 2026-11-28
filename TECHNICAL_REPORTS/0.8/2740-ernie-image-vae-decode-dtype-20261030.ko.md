# PR #2740: ERNIE-Image 잠재 변수를 VAE dtype으로 디코딩하고 인코더를 bf16으로 고정

**날짜**: 2026-10-30 (머지 전에 작성한 보고서이며, 작성 시점에 PR은 열린 초안 상태)
**상태**: 열린 PR. CUDA(GB10)에서만 검증했고 Metal, ROCm, Ampere 이전 CUDA는 실행하지 않음
**언어**: Rust, Markdown
**위험도**: 낮음에서 중간 (VAE 캐스트는 추가적 변경이며, 인코더 로더 변경은 모든 백엔드에서 Ministral 3 가중치를 읽는 방식을 바꾸고 양자화 체크포인트 경로를 제거함)

## 요약

PR #2740(#2734 해결, 에픽 #2348)은 `MLXCEL_CUDA_F16_NORMALIZE=1`과 `MLX_ENABLE_TF32=0` 조건에서 ERNIE-Image를 1024x1024로 생성할 때 발생하는 메모리 부족 오류를 고친다. 이 스위치를 켜면 VAE 가중치는 f16이지만 bf16 트랜스포머는 bf16 잠재 변수를 반환한다. bf16 x f16 연산은 VAE 디코딩 전체를 f32로 승격시키고, TF32가 꺼진 f32 1024 디코딩은 `cudaMallocAsync`에서 실패한다. 수정은 `ernie_image::pipeline::decode_latents`를 추가해 디코딩 전에 새 접근자 `Flux2Vae::dtype()`의 dtype으로 잠재 변수를 캐스트하는 것이다.

이 수정을 위한 패리티 실행에서 두 번째 문제가 드러났다. 환경 변수를 켜면 Ministral 3 텍스트 인코더도 `load_text_weights`를 거쳐 f16으로 로드되어, 컨디셔닝이 rel-RMS 3.509e-3만큼 이동하고 Turbo가 비트 단위 일치에서 최종 잠재 변수 rel-RMS 2.93e-1(26.04 dB)로 악화되었다. 이제 인코더와 프롬프트 인핸서는 원시 로더 `load_weights_from_dir`로 읽으므로 어떤 로드 정책에서도 bf16을 유지한다. 원시 로더는 NVFP4/ModelOpt 재패킹을 건너뛰므로 `load_ministral3`는 양자화된 설정을 명확한 오류로 거부한다.

두 경우(환경 변수 유무) 모두 실제 체크포인트 패리티 테스트 42개가 기존 임계값 그대로 통과한다. 비용은 f16 정책에서 f16 VAE 디코딩 때문에 Turbo PNG PSNR이 57.00 dB에서 53.56 dB로 낮아진다는 점이다.

## 1. 문제 정의

`MLXCEL_CUDA_F16_NORMALIZE`는 로드 시 bf16 가중치를 f16으로 변환한다. sm_80 이상에서는 선택 사항이고 sm_80 미만에서는 기본으로 켜진다. ERNIE-Image 트랜스포머는 모든 백엔드에서 bf16이며(f16 가중치는 NaN 잠재 변수를 만든다), 따라서 `sample_latents`는 bf16을 반환한다. 스위치를 켜면 `Flux2Vae::default_dtype()`은 f16을 반환한다. `decode_to_pixels`는 bf16 잠재 변수를 그대로 `decode_packed_latents`에 넘겼고, 그 결과 모든 conv와 attention 연산이 f32로 실행되었다.

이슈 #2734는 `ernie-image-turbo`에서 VAE 단독 프로브로 이를 측정했다. f16 VAE에 bf16 잠재 변수를 넣고 1024에서 TF32를 끄면 f32로 디코딩되어 OOM이 나며, attention 청크 예산을 줄여도 마찬가지이므로 mid-block attention은 원인이 아니다. 같은 VAE에 f16 잠재 변수를 넣거나 bf16 VAE에 bf16 잠재 변수를 넣으면 3.84 GiB로 디코딩된다. 순수 f32 VAE에 f32 잠재 변수를 넣은 경우도 TF32 없이 1024에서 OOM이 나는데, 이는 별개의 CUDA 문제로 이슈에서 범위 밖으로 두었다. FLUX.2, Ideogram 4, LLaDA-Image는 이미 VAE dtype으로 먼저 캐스트하지만 ERNIE-Image만 하지 않았다.

두 번째 문제는 첫 수정으로 환경 변수 아래에서 패리티 테스트를 실행할 수 있게 된 뒤에야 발견되었다. 그 전에는 1024 케이스가 VAE에서 크래시해 인코더 이후 단계를 비교할 수 없었다.

## 2. 변경 요약

커밋 4개, 파일 6개(+103 / -6).

| 파일 | 변경 |
|---|---|
| `src/imagegen/vae/model.rs` | `Flux2Vae`가 생성 시의 `compute_dtype`을 저장하고 `dtype()`으로 노출. 순수 추가이며 ERNIE 디렉터리 밖의 파일. |
| `src/imagegen/ernie_image/pipeline.rs` | 새 `decode_latents(vae, latents)`가 `vae.dtype()`으로 캐스트한 뒤 `decode_packed_latents`를 호출. `decode_to_pixels`가 이를 사용. |
| `src/imagegen/ernie_image/text_encoder.rs` | `load_ministral3`가 `crate::models::load_text_weights` 대신 `mlxcel_core::weights::load_weights_from_dir`로 가중치를 읽음. 가중치를 읽기 전에 실행되는 `reject_quantized_config` 추가. |
| `src/imagegen/ernie_image/pipeline_tests.rs` | 새 `decode_latents_runs_in_the_vae_dtype`. |
| `src/imagegen/ernie_image/text_encoder_tests.rs` | 새 `quantized_configs_are_refused_before_any_weight_loads`. |
| `docs/image-generation.md` | ERNIE-Image 정밀도 설명과 정밀도 표의 텍스트 인코더, VAE 행. |

커밋 순서: `a07f2683`(VAE 캐스트, f16 정책용으로 패리티 임계값을 임시 완화), `9d52ba0a`(인코더를 bf16으로 고정, 완화한 임계값 제거), `b4804580`(문서와 주석 정렬, 동작 변경 없음), `3be12dd2`(양자화 설정 가드).

## 3. 기술적 선택과 그 이유

### 3.1 ERNIE 쪽에서 `vae.dtype()`으로 캐스트

**배경:** 캐스트는 `Flux2Vae::decode_packed_latents` 안에 둘 수도 있고 ERNIE 호출부에 둘 수도 있었다.

**결정:** ERNIE 쪽에 둔다. Z-Image는 공유 VAE에 f32 잠재 변수 스트림을 의도적으로 넘기므로 VAE가 dtype을 소유하면 안 된다.

**이슈 본문과의 차이:** 이슈는 캐스트 대상으로 `Flux2Vae::default_dtype()`을 제안했다. PR은 실제로 가중치가 로드된 dtype인 `vae.dtype()`을 쓴다. ERNIE의 실제 VAE에서는 둘이 같다. 환경 변수가 없으면 테스트 호스트에서 `default_dtype()`이 bf16이어서, 문구 그대로 f16 VAE 단위 테스트를 작성하면 실패한다. 로드된 dtype으로 캐스트하는 쪽이 명시적 dtype으로 만든 VAE에도 맞다. 비용은 `vae/model.rs`의 공개 접근자 하나다.

### 3.2 원시 로드로 인코더를 bf16에 고정

**근거:** VAE 수정만 적용했을 때 환경 변수를 켠 실제 체크포인트 실행에서 컨디셔닝은 rel-RMS 3.509e-3, Turbo는 2.93e-1(26.04 dB)이었다. 트랜스포머는 모든 백엔드에서 bf16이고, 업스트림 입력에 대한 첫 순전파가 두 경우 모두 0.000e0이었으므로 움직인 것은 인코더였다.

**결정:** f16이 시드 고정 임계값을 통과하지 못하면 해당 컴포넌트는 bf16을 유지한다는 프로젝트 규칙을 따른다. `load_ministral3`가 원시 로더를 호출하므로 f16 왕복이 없고 `sanitize.rs`와 공유 로더도 바뀌지 않는다. 로드 정책이 f16으로 변환하던 Metal에서도 인코더와 인핸서가 bf16을 유지한다.

**기각한 대안:**

- f16 정책용으로 패리티 임계값 완화(`a07f2683`에서 시도, `9d52ba0a`에서 되돌림): 실제 26 dB 회귀를 느슨한 게이트 뒤에 숨기게 된다.
- 공유 텍스트 가중치 로더에 bf16 예외 옵션 추가: 한 패밀리를 위해 영향 범위가 커진다.

### 3.3 양자화 설정 거부

원시 로더는 `load_text_weights`가 수행하는 NVFP4/ModelOpt 재패킹을 하지 않는다. 양자화된 체크포인트는 나중에 가중치 누락 오류로 실패하게 된다. `reject_quantized_config`는 최상위와 `text_config` 아래의 `quantization`, `quantization_config`를 확인하고(null 값은 무시) 일찍 실패시킨다. 양자화된 ERNIE 인코더는 존재하지 않으며, 양자화 변환은 이미 문서에서 범위 밖이다.

## 4. 검증

모두 GB10, `MLX_ENABLE_TF32=0`, `--test-threads=1`.

- 새 `decode_latents_runs_in_the_vae_dtype`: `vae_tiny.safetensors`의 작은 f16 VAE에 bf16 packed 잠재 변수를 넣어 출력이 f16인지, 캐스트 없는 `decode_packed_latents`는 f32인지 확인한다. main에는 `decode_latents`가 없으므로 main 실패 증명은 브랜치 안에서 캐스트를 제거해 수행했고, 단언이 `left: 10`(f32), `right: 9`(f16)로 실패했다. 수정 후에는 환경 변수 유무와 관계없이 통과한다.
- 최종 head의 비무시 `imagegen::ernie_image`: 39 통과, 4 무시.
- 실제 체크포인트 `imagegen::ernie_image --include-ignored`: 두 경우 모두 기존 임계값으로 42 통과, 건너뛴 케이스 없음. 전체 두 번의 실행은 `9d52ba0a`에서 했다. 이후 커밋은 문서, 주석, 양자화 가드이며 최종 head에서 `the_real_encoder_matches_upstream`이 환경 변수를 켠 상태로 컨디셔닝 rel-RMS 0.000e0을 여전히 출력한다.
- 오라클 디렉터리는 mlx-vlm 0.7.8 git `73d366ec`, mlx 0.32.4 CUDA에서 `tests/fixtures/generate_ernie_image_fixtures.py --control`로 재생성했고, 요약이 커밋된 `ernie_image_parity.safetensors`와 바이트 단위로 같다.

1024x1024 최종 PNG PSNR과 최종 잠재 변수 rel-RMS:

| 케이스 | 환경 변수 없음 | `MLXCEL_CUDA_F16_NORMALIZE=1` |
|---|---|---|
| turbo_1024 | 57.00 dB, 0 | 53.56 dB, 0 |
| turbo_edit_1024 | 32.23 dB, 1.35e-1 | 32.02 dB, 1.45e-1 |
| base_1024 | 24.24 dB, 3.03e-1 | 24.24 dB, 3.03e-1 |
| base_8step_1024 | 31.43 dB, 1.64e-1 | 31.42 dB, 1.64e-1 |

컨디셔닝 rel-RMS는 모든 케이스에서 0이다. f16 정책에서는 VAE 디코딩이 유일한 f16 컴포넌트이며, Turbo가 57.00에서 53.56 dB로 떨어지는 것은 이 때문이고 CFG 케이스는 거의 움직이지 않는다. Base 50스텝은 두 경우 모두 에픽의 30 dB 기준에 못 미치며, 이는 main에 이미 문서화된 내용이다(CFG 배치 2 GEMM 반올림, 이번 변경과 무관).

`9d52ba0a`(양자화 가드 이전)에서 빌드한 바이너리의 CLI 스모크: `MLXCEL_CUDA_F16_NORMALIZE=1 MLX_ENABLE_TF32=0 mlxcel generate-image -m models/mlx/ernie-image-turbo ... --size 1024x1024 --seed 42`가 73초에 끝나고 일관된 이미지를 생성했다. 변경 전에는 OOM이었다. 프롬프트 인핸서도 새 원시 로드 경로로 실행되었다.

다른 `decode_packed_latents` 호출부도 점검했다. LLaDA-Image, FLUX.2, Ideogram 4는 이미 캐스트하고, Z-Image는 의도적으로 f32를 넘기며, 나머지는 테스트다. 후속 작업은 필요 없다. 린트와 검증 게이트(`cargo fmt`, clippy `-D warnings`, `make verify-*`, `check_binary_assets.py`)는 통과한다.

## 5. 학습 포인트

- MLX에서 bf16과 f16 피연산자를 섞으면 조용히 f32로 승격된다. 원인과 먼 곳에서 나는 할당 실패는 dtype 승격 버그일 수 있으므로, 청크 예산을 조정하기 전에 dtype부터 출력해 본다.
- 테스트 경로를 열어 주는 수정이 그 경로가 가리고 있던 두 번째 회귀를 드러낼 수 있다. 여기서는 크래시가 26 dB 수준의 인코더 드리프트를 가리고 있었다. 임계값을 완화하지 않고 수정 후 패리티 경로를 다시 돌렸기 때문에 발견되었다.
- 공유 텍스트 가중치 로더를 쓰면 컴포넌트가 전역 dtype 정책에 묶인다. bf16 기준 구현과 패리티 계약이 있는 컴포넌트는 원시 로드를 하고 그 사실을 명시해야 한다.
- 캐스트 대상은 전역 기본값이 아니라 모듈이 실제로 만들어진 dtype(`vae.dtype()`)이어야 한다.

## 6. 한계와 미검증 사항

- GB10 CUDA에서만 측정했다. Ampere 이전 CUDA와 Apple Silicon에서는 정책이 기존에 인코더와 인핸서를 f16으로 변환했으나 이제 bf16으로 실행된다. 이 부분은 측정하지 않았다. 인코더는 프롬프트당 순전파 한 번이지만, 인핸서는 토큰 단위로 디코딩하므로 bf16을 기본 지원하지 않는 장치에서는 느려질 수 있다.
- 텍스트 생성의 `--surgery` 가중치 훅은 `load_text_weights` 안에 있으므로 더 이상 ERNIE 인코더나 인핸서에 적용되지 않는다.
- 양자화된 ERNIE 인코더는 이제 로드되지 않고 거부된다. 현재 그런 체크포인트는 없다.
- CLI 스모크는 양자화 설정 가드 이전에 실행되었다. 가드는 단위 테스트와 최종 head의 실제 인코더 테스트로 확인한다.
- f32 비TF32 1024x1024 VAE OOM 자체는 건드리지 않았다.
- f16 정책에서 Turbo PSNR이 57.00에서 53.56 dB로 낮아지는 것은 f16 디코딩의 비용으로 받아들인 것이다.

## 참고

- 이슈 #2734(해결), 에픽 #2348, PR #2706(발견 경위), 메인테이너 결정 #932(패리티 계약).
- 관련: PR #2730(Qwen3 컨디셔너 부정적 결과), PR #2717(LLaDA-Image).
- 업스트림: mlx-vlm `73d366ec`.
- 문서: `docs/image-generation.md`(ERNIE-Image 정밀도, 정밀도 표), `docs/environment-variables.md`(`MLXCEL_CUDA_F16_NORMALIZE`).
