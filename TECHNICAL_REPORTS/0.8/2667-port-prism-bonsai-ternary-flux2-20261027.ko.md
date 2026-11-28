# PR #2667: Prism Bonsai 3값 2비트 FLUX.2 포팅

**날짜**: 2026-10-27
**상태**: CUDA(GB10)에서 완료, Metal 실행과 Metal 메가커널 비교는 이 호스트에서 검증 불가
**언어**: Rust, Python(픽스처 생성기), Markdown
**위험도**: 낮음

## 요약

PR #2667(이슈 #2428, 에픽 #2348)은 `prism-ml/bonsai-image-ternary-4B-mlx-2bit`용 `bonsai` 이미지 생성 패밀리를 추가합니다. 이 모델은 3값(ternary) 선형 레이어를 2비트 MLX affine 가중치(group 128)로 패킹한 FLUX.2 Klein 4B 트랜스포머와 4비트 MLX Qwen3 텍스트 인코더로 구성됩니다. 복사본이 아니라 PR #2660에서 병합된 FLUX.2 포트 위에 구축되었습니다. 새 모듈은 패킹 가중치용 `LinearSource`, 양자화 텍스트 인코더, 프롬프트 패딩만 제공하고 FLUX.2 파이프라인, 스케줄러, 프롬프트 경로, 잠재 헬퍼, VAE를 재사용합니다. 패밀리는 `mlxcel generate-image`, `mlxcel-server --image-model`(`/v1/images/generations`), `mlxcel arch`에서 동작합니다.

Bonsai의 Metal 전용 퓨즈드 커널을 해당 연산으로 대체한 업스트림 mlx-vlm CUDA 빌드와 비교했을 때, 512x512, 4 스텝, seed 9909 케이스는 초기 노이즈가 비트 단위로 동일하고 디코딩 PSNR이 31.18 dB입니다. f32 트랜스포머는 업스트림 f32와 상대 RMS 1.19e-6로 일치합니다. FLUX.2 자체의 패리티 수치는 변하지 않았습니다.

## 1. 문제 정의

Bonsai의 `model_index.json`은 `Flux2KleinPipeline`을 가리키지만 체크포인트 구조는 FLUX.2 Diffusers 트리와 다릅니다. `quantization_config.json`과 패킹된 `uint32` 선형 레이어가 있는 `transformer-packed-mflux/`, `text_encoder-mlx-4bit/`, `manifest.json`이 있습니다. 클래스 이름 스캔보다 먼저 `bonsai`로 분류하는 프로브 규칙은 #2341에서 이미 들어갔지만 로더가 없어 지원되지 않는 패밀리로 보고되었습니다. 업스트림의 고속 경로(`klein_fast`)는 CUDA나 CPU 대체 경로가 없는 `mx.fast.metal_kernel`을 사용하므로, 구현도 패리티 기준도 이 호스트에서 업스트림 런타임 코드를 그대로 따를 수 없었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 패밀리 모듈 | 새 `src/imagegen/bonsai/`(`config.rs`, `loader.rs`, `mod.rs`, 테스트, 체크포인트 기반 `parity_tests.rs`). 생성만 지원, 기본 4 스텝과 guidance 1.0, 프롬프트는 512 토큰으로 패딩(업스트림 `bucketed_seq_len=False`), `n`개 이미지는 업스트림 서버처럼 seed `seed + i`로 하나씩 실행. |
| 패킹 로더 | `quantization_config.json`을 읽고 `mlx-packed-affine` 외의 `format`, mlxcel-core affine 집합 밖의 `bits`나 `group_size`를 거부합니다(1비트 바이너리 변형은 `bits (1)`에서 실패). `quantized_modules`의 모듈은 affine `UnifiedLinear`로, 나머지는 dense로 로드합니다. 목록에 있는데 `.scales`가 없거나, 목록에 없는데 `.scales`가 있으면 오류입니다. 각 패킹 레이어는 `validate_quantized_packing`을 통과해야 합니다. |
| FLUX.2 이음매 | `Flux2Pipeline::with_linear_source`(`new`는 기본 소스와 버킷팅 활성 상태로 이를 위임), `Flux2Transformer::load`가 호출하는 기본 허용 `LinearSource::check_weights` 훅, 몇몇 헬퍼의 가시성 확대(각각 `Used by:` 주석 포함). |
| 등록 | FLUX.2와 Z-Image 뒤에 `imagegen::FAMILIES`와 `STANDALONE_ARCHITECTURE_FAMILIES` 행 추가, 매니페스트가 없는 경우의 프로브 테스트. |
| 문서와 CI | `docs/image-generation.md` Bonsai 절(정밀도 행 포함), `docs/supported-models.md` 행, 테스트 전용 환경 변수 두 개 허용 목록 추가. |
| 픽스처 | `tests/fixtures/bonsai/`(약 1.45 MB)와 `generate_bonsai_fixtures.py`. |

## 3. 기술적 선택과 그 이유

### 3.1 `quantized_modules`를 기준으로 삼기

업스트림은 `.scales` 형제 텐서가 있는 선형 레이어를 모두 패킹으로 간주하고 `quantized_modules`나 `skip_patterns`를 읽지 않습니다. 이 포트는 목록을 사용하고, 목록과 파일이 어느 방향으로든 어긋나면 오류를 냅니다. 실제 체크포인트에서 `.scales`가 있는 100개 모듈은 목록의 100개와 정확히 일치하므로 정상 체크포인트에서는 동작이 같고, 손상되었거나 어긋난 익스포트는 모듈 이름과 함께 로드 시점에 실패합니다.

### 3.2 로드 시점의 패킹 형상 검증

보안 리뷰에서 scales와 형상이 다른 biases, weight와 scales의 선행 축 불일치, 1차원 scales가 모두 오류 없이 로드된 뒤 첫 forward에서 MLX 자체 형상 검사에 걸린다는 점이 발견되었습니다. 이 실패는 C++ 브리지를 넘어 잡을 수 없어 `mlxcel-server`가 첫 요청에서 중단됩니다. 모든 패킹 레이어에 기존 코어 검사 `validate_quantized_packing`을 실행해 이를 로드 오류로 바꿨습니다. 텐서끼리는 일관되지만 `config.json`과 다른 `hidden_size`로 패킹된 체크포인트는 여전히 늦게 실패합니다. 이를 고치려면 기대 입력 폭을 `LinearSource::load`로 전달해야 하고 FLUX.2에도 영향을 주므로 후속 작업으로 남겼습니다.

### 3.3 업스트림이 내려받는 디코더 대신 체크포인트의 `vae/` 사용

업스트림 `bonsai/weights.py::load_vae`는 체크포인트 `vae/`를 무시하고 로드 시점에 `black-forest-labs/FLUX.2-small-decoder`(채널 96/192/384/384)를 내려받습니다. 이슈는 `vae/`를 명시하고, 서버에서 모델 로드 중 네트워크 다운로드는 허용할 수 없으므로 이 포트는 `vae/`로 디코딩합니다. 두 디코더는 서로 34.96 dB 차이이며, 이 포트는 `vae/`를 쓰는 오라클 대비 31.18 dB, 업스트림 기본 동작 대비 29.82 dB입니다. 선택적 로컬 small decoder 경로는 후속 작업 후보입니다.

### 3.4 Metal 커널 없이 CUDA 기준 만들기

픽스처 생성기는 퓨즈드 norm-RoPE 커널을 해당 연산(f32 RMSNorm에 gain 곱, 인터리브 쌍 RoPE, bf16 캐스트)으로 대체한 업스트림 자체 `BonsaiImage` 파이프라인을 실행합니다. `--flux2-transformer` 모드는 f32 검사를 위해 Bonsai 가중치를 업스트림 FLUX.2 트랜스포머에 로드하며, mlx 0.32.4 CUDA가 2행 이상의 f32 2비트 `quantized_matmul`에서 실패하므로 가중치를 역양자화합니다. 메가커널 자체는 여기서 `quantized_matmul`과 비교할 수 없으며, 코드상으로 `QuantizedLinearKernel`은 `mx.quantized_matmul`을 호출합니다.

### 3.5 개정된 에픽 기준에 따른 잠재 허용 오차

에픽 전체에 적용된 메인테이너 결정에 따라 테스트는 PSNR 30 dB 이상, 비트 단위로 동일한 초기 노이즈, 0.2 최종 잠재 상한(잘못된 포트는 약 1.4), 프롬프트 임베딩 0.1 상한을 기준으로 합니다. 측정된 최종 잠재 상대 RMS 1.0e-1과 프롬프트 임베딩 2.3e-2는 드리프트로 기록합니다. 이것이 포팅 오류가 아닌 bf16 반올림이라는 근거는 다음과 같습니다. 업스트림의 bf16 4비트 컨디셔너는 자체 f32 실행과 1.95e-2 차이이고 이 포트는 같은 f32 실행과 1.74e-2 차이입니다. 또한 업스트림은 같은 가중치에서 FLUX.2 트랜스포머로 `klein_fast`를 대체하면 최종 잠재가 자기 자신과 3.5e-2 차이가 납니다. 공유 Qwen3 컨디셔너의 정밀도는 의도적으로 바꾸지 않았고, f32 q/k/v 어텐션은 실제 모델 logit-trace 기준을 갖춘 별도 후속 작업에서 다룹니다.

## 4. 검증과 측정 결과

오라클: mlx-vlm 0.7.8(git 73d366ec), GB10의 mlx 0.32.4 CUDA, `MLX_ENABLE_TF32=0`.

| 검사 | 결과 |
|---|---|
| 초기 노이즈 | 비트 단위 동일 |
| f32 트랜스포머 대 업스트림 f32 | 상대 RMS 1.19e-6 |
| 업스트림 0번 스텝 입력에서 bf16 한 스텝 | 상대 RMS 8.1e-3 |
| 업스트림 최종 잠재를 VAE로 디코딩 | 57.61 dB |
| 종단 간 PNG(CLI 수용 명령) | 31.18 dB, 서버 출력은 픽셀 동일 |
| `MLXCEL_CUDA_F16_NORMALIZE=1`에서 같은 케이스 | 31.21 dB |
| CLI 실행 | 512x512, 로드 포함 11.5 s, RSS 3.6 GB |

최종 트리의 lib 필터: `imagegen::` 220 통과, `models::registry` 16, `bonsai::parity --ignored` 5개 중 5개, `flux2::parity --ignored`는 PR #2660과 정확히 같은 수치로 6개 중 6개. `FAMILIES` 행을 제거하면 등록 테스트 두 개가 실패하며, 로더와 설정 테스트는 main에 존재하지 않습니다.

## 5. 학습 포인트

- 기존 패밀리의 양자화 변형은 가중치 소스 이음매로 포팅하는 것이 가장 저렴합니다. 선형 레이어 생성용 메서드 하나와 가중치 맵 검증용 메서드 하나로 FLUX.2 그래프를 건드리지 않고 패리티를 비트 단위로 유지했습니다.
- forward 중 MLX가 내는 형상 오류는 FFI 경계를 넘어 치명적이므로, 체크포인트 검증은 오류를 반환할 수 있는 로드 시점에 두어야 합니다.
- Metal 전용 커널에 의존하는 업스트림 참조 코드도 커널을 문서화된 연산으로 대체하면 CUDA 오라클로 쓸 수 있습니다. 단, 무엇을 대체했는지 PR에 정확히 밝혀야 합니다.

## 6. 후속 작업과 한계

- Metal 실행과 메가커널 대 `quantized_matmul` 비교는 이 호스트에서 검증할 수 없습니다.
- 공통 MEDIUM: config와 다른 `hidden_size`로 패킹된 체크포인트는 여전히 첫 forward에서 실패합니다(FLUX.2와 Bonsai 공통).
- 선택 사항: 업스트림 기본 동작과의 일치가 목표가 되면 업스트림 small decoder용 로컬 경로.
- 공유 Qwen3 컨디셔너 f32 어텐션 후속 작업(에픽 오케스트레이터가 등록).

## 참고 자료

- 이슈 #2428, 에픽 #2348, FLUX.2 포트 #2424(PR #2660), 런타임 #2341(PR #2497), Z-Image #2662
- 업스트림: Blaizzy/mlx-vlm `mlx_vlm/models/bonsai` 및 PR #1226, #1234, #2373
- `docs/image-generation.md`, `docs/supported-models.md`
