# 기술 보고서: PR #2030 - 백엔드 양자화 역량 테이블과 NVFP4 로드 정책

**날짜**: 2026-09-29

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: Rust

**위험도**: 중간 (모든 백엔드에서 NVFP4 로드 경로 선택이 바뀝니다. Metal과 CUDA가 이전 경로를 유지한다는 근거는 해당 백엔드에서의 실행이 아니라 단위 테스트입니다)

## 요약

이슈 #1806(에픽 #1801의 1단계)은 "이 백엔드가 양자화 모드 M을 네이티브로 실행할 수 있는가?"에 답하는 단일 지점과, 그 답을 바탕으로 로드 시점에 변환하거나 거부하는 정책을 요구했습니다. 이 PR은 그 테이블을 `mlxcel_core::hardware`에 추가하되 빌드 피처가 아니라 런타임 `GpuBackendKind`를 키로 삼고, ModelOpt NVFP4 재패킹 경로가 이 테이블을 거치도록 바꿉니다. 이제 ROCm에서 ModelOpt NVFP4 체크포인트는 환경 변수 없이 affine 4비트로 변환되어 로드됩니다. 이전에는 ROCm 디스패치가 실행할 수 없는 group-16 네이티브 레이아웃을 만들었습니다.

같은 테이블이 모든 양자화 레이어 로더의 레이어별 검사도 뒷받침하므로, ROCm에서 MLX 네이티브 NVFP4 익스포트는 첫 forward에서 프로세스를 중단시키는 대신 로드 시점에 레이어 이름과 함께 실패합니다. 네이티브 NVFP4 FFI 테스트 두 개는 테이블이 비네이티브로 표시한 백엔드에서 건너뜁니다. 이로써 ROCm에서 `mlxcel-core` lib 테스트 바이너리를 끝내고 그 뒤에 정렬되는 모든 테스트를 가리던 `terminate called` 중단이 사라졌습니다.

## 1. 문제 정의

세 문제 모두 로더가 자신이 만든 모드를 백엔드가 실행할 수 있다고 가정한 데서 나왔습니다.

**NVFP4 경로를 빌드 플래그가 골랐습니다.** `current_nvfp4_repack_strategy()`는 `cfg!(feature = "cuda")`를 읽었습니다. ROCm 빌드는 CUDA 빌드가 아니므로 기본값인 `DirectTranscode` 경로를 탔고, 이 경로는 MLX 네이티브 NVFP4(group 16, E4M3 블록 스케일)를 만듭니다. ROCm qmv 디스패치(`patches-rocm/mlx/backend/rocm/quantized/qmm.hip`)는 group 크기 32, 64, 128만 구현하고 나머지는 예외를 던집니다. 동작하는 `DenseAffine` 경로는 이미 있었지만 `MLXCEL_NVFP4_DENSE_REPACK=1`을 설정해야만 쓰였습니다.

**그 예외는 오류가 아니라 중단이었습니다.** `quantized_matmul`, `gather_qmm`, 컴파일된 MLP 헬퍼는 cxx 브리지를 넘어 `Result`를 돌려주지 않으므로, ROCm 백엔드의 C++ 예외는 프로세스를 종료시킵니다. 실제 사용에서는 레이어나 모드에 대한 언급 없이 첫 forward에서 크래시가 났습니다. 테스트 게이트에서는 `ffi_tests::compiled_qgelu_mlp_global_scale_native_nvfp4_prefill_matches_reference`가 단일 스레드 `-p mlxcel-core --lib` 실행을 중단시켰습니다. libtest는 이름 순으로 테스트를 실행하므로, 그 뒤에 정렬되는 테스트(`ffi_tests`의 나머지, 그리고 `hardware`, `layers`, `mla`, `paged_v2`, `sampling*` 등)는 이 중단이 생긴 뒤로 ROCm에서 한 번도 실행되지 않았습니다. 이슈는 이를 `make verify-rocm`의 마지막 빨간 항목이자 #1807, #1808, #1809의 블로커로 기록했습니다.

**변환할 수 없는 레이어는 조용히 건너뛰었습니다.** ModelOpt 레이어를 재패킹할 수 없을 때(스칼라가 아닌 `weight_scale_2`, 누락된 스케일, 맞는 group이 없는 in_dim), 로더는 "Skipping NVFP4 repack"을 출력하고 넘어갔습니다. NVFP4가 네이티브인 백엔드에서는 괜찮습니다. 손대지 않은 텐서가 없는 커널로 실행될 일이 없기 때문입니다. 변환해야 하는 백엔드에서는 건너뛴 레이어가 실행할 수 없는 레이아웃으로 남거나 조용히 틀린 값이 됩니다.

## 2. 변경 요약

- `src/lib/mlxcel-core/src/hardware.rs`: `QuantMode`(MLX가 파싱하는 네 모드, `as_str`와 정확 일치 `from_mlx_name` 포함), `QuantModeSupport`(`Native`, `ConvertTo(mode)`, `Unsupported`), `GpuBackendKind::display_name`, 테이블 본체인 `GpuBackendKind::quant_mode_support`, 실행 중인 백엔드를 읽는 자유 함수 `quant_mode_support`. Metal, CUDA, `None`(CPU)은 모든 모드를 네이티브로 보고합니다. ROCm은 affine, mxfp4, mxfp8을 네이티브로, NVFP4를 `ConvertTo(Affine)`으로 보고합니다.
- `src/models/sanitize.rs`: `nvfp4_repack_strategy`가 `cuda_build` 대신 `GpuBackendKind`를 받고 `Result<Nvfp4LoadRoute, String>`을 돌려줍니다. 이 구조체는 백엔드, 전략, 사람이 읽을 이유, `conversion_required` 플래그를 담습니다. ROCm에서는 항상 `conversion_required = true`인 `DenseAffine`이며, `MLXCEL_NVFP4_NATIVE_REPACK`은 무시되고 이유 문자열에 그렇게 적힙니다. `repack_nvfp4_weights_to_quantized`는 실패할 수 있게 바뀌었고, 테스트가 경로를 주입할 수 있도록 `repack_nvfp4_weights_with_route`로 분리되었습니다. 기존 "Skipping NVFP4 repack" 지점은 모두 `nvfp4_layer_not_repacked`를 거칩니다. 변환이 필요하면 레이어 이름과 이유로 로드를 실패시키고, 아니면 기존대로 건너뜁니다. ROCm 전용 사전 검사가 2차원이 아닌 가중치, U8이 아닌 가중치 dtype, `[out_dim, num_groups]`가 아닌 스케일 shape, `i32`에 맞지 않는 in_dim을 거부합니다. affine group 검사는 dense 재구성보다 앞으로 옮겨졌습니다. 로드 로그 한 줄이 원본 모드, 대상, 이유를 모두 적습니다.
- `src/models/sanitize.rs`: FP4 역양자화 루프가 `dequantize_modelopt_nvfp4_rows`가 되어 행을 최대 16개의 scoped 스레드에 나누고, 호스트 바이트 버퍼와 f32 행렬은 다음 단계에서 더 필요 없어지는 즉시 해제됩니다.
- `src/lib/mlxcel-core/src/layers.rs`: `validate_quantization_mode_runnable(mode, backend)`는 테이블이 `Native`로 표시하지 않은 모드를 모두 거부하며, 메시지에 모드, 백엔드, 해결책(`mlx_lm.convert -q`로 affine 재양자화, 또는 네이티브 커널이 있는 백엔드 사용)을 적습니다. `reconcile_quantization_layout_logged`(공용 dense 및 임베딩 로더)와 `QuantizedMultiLinear::from_weights`에서 호출되고, 바이너리 크레이트에서는 `SwitchLinear`, gpt-oss `ExpertLinear`, kimi_linear의 `MultiLinear`에서 호출됩니다.
- `src/models/gemma4.rs`, `src/loading/vlm_gemma.rs`: 실패할 수 있게 된 `sanitize_gemma4_nvfp4_weights`의 오류를 전파합니다.
- `src/lib/mlxcel-core/src/ffi_tests.rs`: `backend_runs_native_nvfp4`가 네이티브 NVFP4 테스트 두 개를 테이블로 게이트하고, 건너뛸 때 이유를 출력합니다.
- 문서: `docs/installation.md`의 NVFP4 행, `docs/environment-variables.md`의 NVFP4 변수 두 개, 그리고 더 이상 #1806 중단을 언급하지 않는 `Makefile`의 `verify-rocm` 순서 주석.

## 3. 기술적 선택과 그 이유

### 테이블의 키는 빌드 피처가 아니라 런타임 백엔드입니다

이슈의 원래 계획은 이 부분을 열어 두었고("#1803의 백엔드 종류 또는 빌드 피처"), 2026-09-29 갱신에서 결정했으며 PR은 그 결정을 따릅니다. `cfg!` 체인은 다중 백엔드 빌드를 표현하지 못하고, WebUI 카탈로그가 이미 한 번 그 대가를 치렀습니다(#1886, 모든 ROCm 빌드가 unsupported로 보고됨). `gpu_backend_kind()`는 이 프로세스에서 MLX가 실제로 선택한 백엔드를 알려주며, 모델이 실행되는 순간 커널이 있는지를 정하는 것은 바로 그것입니다.

이 선택에는 PR 본문이 밝힌 눈에 보이는 부작용이 하나 있습니다. 장치를 보지 못한 CUDA 빌드는 이제 `None`으로 실행됩니다. `MLXCEL_NVFP4_DENSE_REPACK=1`일 때 그곳의 dense 경로는 예전에는 (빌드가 CUDA였으므로) 네이티브 NVFP4를 대상으로 했지만 이제는 (실행 중인 백엔드가 CPU이므로) affine을 대상으로 합니다. 둘 다 CPU 백엔드에서 실행되므로 출력 형식만 바뀌고 동작이 깨지지는 않습니다.

### 불리언이 아니라 세 가지 판정입니다

`ConvertTo(mode)`는 에픽의 방향이 "가능하면 변환하고, 변환이 불가능할 때만 거부"이기 때문에 존재합니다. 불리언이었다면 호출하는 쪽마다 변환 대상을 다시 계산해야 합니다. `Unsupported`는 현재 어떤 행에도 쓰이지 않지만, 로더와 NVFP4 경로 모두 별도 메시지로 처리하므로 향후 백엔드 행이 다른 코드 변경 없이 쓸 수 있습니다. `quant_table_conversions_land_on_native_modes`는 `GpuBackendKind::ALL`을 순회하며 두 구조 규칙을 검사합니다. affine은 모든 백엔드에서 네이티브여야 하고(최후의 변환 대상), 변환 대상은 같은 백엔드에서 그 자체로 네이티브여야 하므로 변환이 연쇄되지 않습니다.

ROCm의 mxfp4 항목은 이슈가 처음 제안한 `ConvertTo(affine)`이 아니라 `Native`입니다. 갱신에서 그 제안을 낡은 것으로 표시했습니다. PR #1818이 mxfp4 `quantized_matmul`, `gather_qmm`, GPU `quantize`를 역양자화한 f32 기준과 비교해 측정했고 gpt-oss-20b-MXFP4-Q4를 정상적으로 실행했기 때문입니다. 원래의 멈춤이 재현되지 않는지 확인하는 일은 여전히 #1808의 몫입니다.

### 변환과 거부는 의도적으로 두 곳에 있습니다

`sanitize.rs`의 NVFP4 재패킹이 변환입니다. 모델을 만들기 전에, 알아보는 체크포인트(`weight_scale_2`가 있는 ModelOpt 삼중항)에 대해 실행됩니다. `layers.rs`의 레이어별 검사가 거부입니다. 백엔드가 실행할 수 없는 모드로 레이어 로더에 도달한 것은 구조상 어떤 변환 경로도 거치지 않은 것입니다. 실제 주요 사례는 로드 시점 변환이 없는 MLX 네이티브 NVFP4 익스포트(`config.json`의 `"mode": "nvfp4"`)입니다. 모델 단위 검사 하나가 아니라 레이어 로더에 검사를 두었기 때문에 MoE 전문가(`SwitchLinear`, gpt-oss, kimi_linear)와 MLA(`QuantizedMultiLinear`)가 같은 함수로 보호되고, 오류가 정확한 레이어를 지목합니다.

이 검사는 순수 함수인 `reconcile_quantization_layout` 바깥에 두어 그 shape 테스트가 호스트 백엔드에 의존하지 않게 했습니다. 파싱할 수 없는 모드는 `validate_quantization_mode`에 맡겨 두 검사가 같은 문자열을 두 번 보고하지 않습니다.

### 변환이 필요한 곳에서만 실패시킵니다

`nvfp4_layer_not_repacked`는 NVFP4가 네이티브인 백엔드에서 기존의 건너뛰기를 유지합니다. 그곳에서 건너뛴 레이어는 백엔드가 실행할 수 있는 레이아웃으로 남으며, 이를 바꾸면 이슈가 명시적으로 배제한 Metal과 CUDA의 동작 변화가 되기 때문입니다. ROCm에서는 같은 조건이 로드를 실패시킵니다. ROCm 전용 사전 검사는 예전 코드가 검사하지 않던 조건(dtype, 스케일 shape, 3차원 가중치)을 추가합니다. 변환 경로에서는 이런 입력이 건너뛰기가 아니라 틀린 affine 가중치를 만들었을 것이기 때문입니다.

### 테스트가 이전 경로를 붙잡고, 변환 경로는 모든 호스트에서 실행됩니다

`nvfp4_repack_strategy_keeps_pre_1806_routes_where_native`는 이전 결정 함수를 테스트 안에 그대로 두고, 테이블이 NVFP4를 네이티브로 표시하는 모든 백엔드와 두 환경 변수의 모든 조합에 대해 새 함수와 비교합니다. 이 호스트에 Metal과 CUDA가 없는 상황에서 "Metal과 CUDA는 이전과 같은 경로를 고른다"는 인수 조건의 근거가 바로 이것입니다. `repack_nvfp4_weights_with_route`를 분리했기 때문에 테스트가 Metal이나 CUDA 호스트에서도 ROCm 경로를 주입할 수 있어, 변환과 그 로드 오류가 gfx1151뿐 아니라 모든 CI 러너에서 실행됩니다.

### 비트 단위로 동일한 스레드 재구성

큰 ModelOpt 체크포인트의 dense f32 재구성은 단일 스레드였고, ROCm에서는 이제 환경 변수가 있을 때만이 아니라 매 로드마다 실행됩니다. 행 범위로 나누어 scoped 스레드에 분배하고 각 스레드는 겹치지 않는 슬라이스에 쓰므로 원소별 연산은 그대로이며, 테스트가 출력이 예전 직렬 루프와 비트 단위로 같음을 확인합니다. Gemma-4-E2B-it-NVFP4에서 측정한 로드 시간은 이전 225초, 이후 210초였고 `--temp 0` 출력 텍스트는 같았습니다.

## 4. 검증

작성자의 실행 결과, PR 본문 기준 (gfx1151, `--features rocm`):

- `cargo test -p mlxcel-core --lib --profile test-fast --features rocm -- --test-threads=1`이 `terminate called` 없이 끝까지 실행됨: 1770 통과, 35 실패, 1 무시. 35개 목록은 PR 본문에 있습니다.
- `-p mlxcel --lib`의 sanitize, nvfp4, gpt_oss, switch_layers, kimi_linear: 132 통과. nvfp4를 언급하는 나머지 모델 테스트 모듈: 621 통과.
- `--features rocm --lib --tests`(루트 크레이트는 `--examples` 추가)로 `clippy -D warnings` 깨끗함. `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt` 통과.
- `bg-digitalservices/Gemma-4-E2B-it-NVFP4`(ModelOpt), 환경 변수 없음: 로드 로그가 affine 경로와 이유를 적고, `--temp 0`에서 25 tok/s로 "The capital of France is Paris."라고 답함.
- `mlx-community/gemma-4-e2b-it-nvfp4`(MLX 네이티브): `language_model.model.embed_tokens: quantization mode nvfp4 has no native kernel on the ROCm backend ...`로 로드 시점에 실패함.

오케스트레이터 검증 (gfx1151, 브랜치를 origin/main `bfc2bfd9` 위로 리베이스):

- `make verify-rocm`이 모든 단계를 실행했습니다. 버전, 커널 dtype 키, 커널 포트 디스패치, llama-compat, fmt, `--features rocm` 워크스페이스 clippy, ROCm 스모크(32 토큰)가 모두 통과했습니다.
- `verify-test-rocm`은 정확히 두 타깃에서 실패했습니다.
  - `-p mlxcel-core --lib`: 1779 통과, 35 실패, 1 무시. 35개는 PR 본문의 목록과 정확히 같습니다. ROCm 포트가 없는 fused paged-attention 테스트 34개(#1814에서 추적)와 bf16 `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible` 하나이며, 모두 이전에는 NVFP4 중단 뒤에 가려져 있었습니다. 통과 수가 작성자의 1770보다 많은 것은 `bfc2bfd9`로 리베이스하면서 테스트가 더 들어왔기 때문입니다.
  - `-p mlxcel --lib`: 8555 통과, 3 실패. 3개는 main에서 #2031로 깨진 VLM 감지 테스트이며 이 PR과 무관합니다.
- 나머지 테스트 바이너리는 모두 통과했습니다.

이슈의 게이트 인수 조건(`-p mlxcel-core --lib`가 `terminate called` 없이 끝나고 전체 통과/실패 목록이 기록됨)은 충족되었습니다. 타깃은 여전히 빨간색이지만, 이제 실패가 보이고 그 실패는 이 변경이 아니라 #1814와 bf16 바이트 일치 사례 하나에 속합니다.

## 5. 학습 포인트

- **중단은 테스트를 실패시키지 않고 숨깁니다.** `--test-threads=1`과 이름 순 실행에서는 잡을 수 없는 예외 하나가 그 뒤의 모든 테스트를 보고서에서 지웁니다. "실패 목록에 없음"은 "실행되지 않음"이었습니다. 중단을 없애자 한동안 존재해 온 실패 35개가 드러났습니다. 게이트 타깃이 중단되면 실패 목록을 완전한 것으로 보기 전에 러너에 도달한 테스트 수부터 세어야 합니다.
- **cxx 브리지는 백엔드 예외를 프로세스 종료로 바꿉니다.** `Result`를 돌려주지 않는 FFI 호출은 호출 전에 Rust 쪽에서 막아야 합니다. 양자화 모드에 대해서는 역량 테이블이 그 방어선이고, 레이어별 검사가 실패 시점을 첫 forward에서 레이어 이름을 아직 아는 로드 시점으로 옮깁니다.
- **런타임 백엔드로 결정하십시오.** 빌드 피처는 무엇이 컴파일되었는지를 말할 뿐 무엇이 실행 중인지를 말하지 않습니다. 같은 교훈이 WebUI에 대해 #1886에 기록되었고, 이 PR은 이를 로드 경로에 적용합니다.
- **이전 함수를 테스트 안에 남기십시오.** 변경 전 결정 함수를 테스트에 넣고 전체 입력 공간에서 비교하는 것은, 작성자가 실행할 수 없는 백엔드에서 "동작 변화 없음"을 증명하는 저렴하고 강한 방법입니다.

## 6. 검증하지 않은 것

- **Metal과 CUDA는 실행하지 않았습니다.** 두 백엔드의 NVFP4 경로 선택이 빌드 플래그에서 런타임 백엔드로 옮겨졌고, 레이어 로더에 모드 검사가 추가되었습니다. 테이블은 두 백엔드에서 모든 모드를 네이티브로 보고하고 단위 테스트가 이전 함수와 같은 경로를 확인하지만, 이 PR을 위해 두 백엔드에서 모델을 로드하지는 않았습니다.
- **변환 손실 logit trace(인수 조건 4)는 측정하지 않았습니다.** 이 방법에는 네이티브 NVFP4 기준이 필요하고, 이는 Metal이나 CUDA를 뜻합니다. CPU 장치로 대신한 실행은 40분 안에 첫 32 토큰 청크를 끝내지 못했고, Linux에서는 CPU 전용 빌드가 링크되지 않습니다(`copy_gpu_inplace` 미정의). ROCm dense-affine trace는 기록해 두었으므로 같은 체크포인트의 Metal trace가 생기면 비교할 수 있습니다. 그때까지 변환 품질의 유일한 근거는 프롬프트 하나에서의 greedy 디코딩 일치입니다.
- **bf16 바이트 일치 실패**(`prefill_dense_gemm_matches_qmm_bytes_where_eligible`)는 드러났을 뿐 조사하지 않았습니다. f16은 통과합니다. ROCm bf16 GEMM의 반올림 차이인지 실제 불일치인지는 열려 있습니다.

## 7. 남은 작업

- 이슈의 5단계대로 Metal 호스트에서 변환 손실 수치를 기록합니다(Metal에서 네이티브 NVFP4 대 `DenseAffine`, 그다음 ROCm에서 `DenseAffine`).
- #1807(mxfp8)과 #1808(mxfp4)은 이 테이블 위에 각자의 변환을 만듭니다. #1808은 ROCm의 mxfp4 `Native` 항목을 확인해야 합니다.
- #1814: 남은 `mlxcel-core` 실패 35개 중 34개를 차지하는 fused paged-attention 커널의 ROCm 포트.
- ROCm에서의 bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` 실패 분류.
