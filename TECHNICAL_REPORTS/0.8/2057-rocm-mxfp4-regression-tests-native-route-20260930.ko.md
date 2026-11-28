# 기술 보고서: PR #2057 - ROCm mxfp4 회귀 테스트 커밋과 네이티브 경로 로드 로그

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: Rust (테스트, 로더 헬퍼), Markdown

**위험도**: 낮음 (커널, overlay, 빌드 변경은 없습니다. `feature = "rocm"`으로 제한된 통합 테스트 하나, ROCm에서만 stderr에 한 줄을 더하는 로드 시점 헬퍼 하나, 문서 수정이 전부입니다. 로더 호출 지점은 Metal, CUDA와 공유하지만 그 환경에서는 실행하지 않았습니다)

## 요약

이슈 #1808(에픽 #1801의 Phase 2)은 gfx1151에서 발생한 mxfp4 실패 세 가지를 보고했습니다. 256x512, M = 1에서 `quantized_matmul`이 멈추는 문제, 4096x4096에서 GPU `quantize`가 `hipLaunchKernel(...) failed: invalid configuration argument`로 실패하는 문제, `qmv_warp_shared_kernel`의 GPU 메모리 fault입니다. 이슈는 커널 수정을, 그것이 안 되면 로드 시점에 mxfp4를 affine 4-bit로 변환하는 방안을 제안했습니다.

PR의 감사 결과, main(`dfc59867`)에서는 이 실패가 하나도 재현되지 않습니다. 이슈가 등록되고 몇 시간 뒤 머지된 PR #1818의 overlay 항목 8, 10, 11(`src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`)이 고쳤지만, 이를 확인하는 커밋된 테스트는 없었습니다. 그래서 이 PR은 커널을 바꾸지 않습니다. 대신 `tests/rocm_mxfp4_quant.rs`를 커밋합니다. 이 테스트는 mxfp4 GPU `quantize`, `quantized_matmul`, `gather_qmm`을 CPU 기준값과 비교하고, capability table의 `Native` 항목을 고정합니다. 각 수치 테스트는 해당 overlay 수정을 되돌리면 실패하는 것을 확인했습니다. 항목 8을 되돌리면 NaN이, 항목 10을 되돌리면 GPU 메모리 fault가 나오고(#1804 이후 hang이 아니라 에러로 보고됩니다), 항목 11을 되돌리면 원래의 "invalid configuration argument"가 나옵니다. 이슈가 예비로 남겨 두었던 affine 4-bit fallback은 필요하지 않습니다.

PR은 이슈의 로드 로그 기준도 충족합니다. ROCm에서 네이티브로 실행되는 block-float 모드마다 첫 양자화 레이어가 `Quantization mode mxfp4: running on native ROCm kernels, no load-time conversion ...`을 출력합니다. NVFP4 repack 로그에 대응하는 네이티브 쪽 로그입니다. Metal, CUDA, CPU에서는 새로 출력되는 것이 없습니다.

## 1. 문제 정의

### 이미 고쳐진 실패를 기술한 이슈

이 이슈는 PR #1818이 overlay를 vendoring하기 전의 상태를 기준으로 등록되었습니다. 작업을 시작할 무렵 이슈의 refresh log(2026-09-29)는 PR #1818이 mxfp4가 정확하다고 보고한 사실을 기록했지만, `LOCAL_FIXES.md`의 어떤 항목도 hang 수정을 명시하지 않아 어느 변경이 hang을 없앴는지는 기록되어 있지 않았습니다. 그래서 계획의 첫 단계가 "먼저 재현"으로 바뀌었습니다. 정확히 그 hang 케이스를 timeout 아래에서 실행하고, 여전히 멈출 때만 근본 원인을 찾기로 했습니다.

gfx1151, `dfc59867`에서 각 케이스를 `timeout` 아래에서 실행한 감사 결과는 다음과 같습니다.

- mxfp4 `quantized_matmul` 256x512, M = 1: 정상 종료하고, CPU에서 f32로 계산한 `x @ dequantize(w).T`와 일치합니다(상대 오차 f32 1.8e-7, bf16 2.8e-3).
- GPU mxfp4 `quantize` 4096x4096: 약 80 ms에 실행되고 CPU quantizer와 비트 단위로 일치합니다.
- mxfp4 `gather_qmm`: sorted와 unsorted 모두 expert별 dequantize 기준값과 일치합니다(f32 1.6e-6, bf16 2.6e-3).

overlay와의 대응 관계:

| 이슈 증상 | overlay 수정 | 근본 원인 |
|---|---|---|
| 256x512, M = 1 qmm hang, `qmv_warp_shared_kernel` 메모리 fault | 항목 8 | qmv dispatch가 activation dtype을 scale 타입으로 썼지만 mxfp4는 그룹당 E8M0 한 바이트를 저장하므로, 커널이 실제 stride의 2에서 4배로 scale을 읽었습니다(NaN과 범위 밖 읽기) |
| `gather_qmm`을 거치는 MoE expert | 항목 10 | non-affine 모드가 `AFFINE=true`와 T 타입 scale로 instantiate된 `gather_qmv_kernel`에 도달했습니다 |
| 4096x4096 `quantize`의 "invalid configuration argument" | 항목 11 | 실패한 launch는 `fp_quantize.hip`이 아니라 `arg_reduce`였습니다. 1-D grid에 출력당 1024-thread 블록 하나를 띄워서, 16.7M 원소 argmin에서 AMD의 grid 차원당 2^32 - 1 thread 제한을 넘었습니다 |

### 수정을 지키는 커밋된 테스트가 없었음

PR #1818은 op 검사를 일회성으로 실행했습니다. `ffi_tests`에 mxfp4 group-32 케이스가 네 개 있었지만 폭이 128과 256이라 256x512 hang 형태보다 작았습니다. 어느 것도 `gather_qmm`을 수치로 검사하지 않았고, #1806 전까지는 그중 두 개가 NVFP4 abort 뒤에 정렬되어 게이트에서 실행되지 않았습니다. fork를 동기화하다 항목 8, 10, 11 중 하나가 빠져도 사용자가 gpt-oss를 로드하기 전까지는 알 수 없었을 것입니다.

### 로드 로그가 어떤 경로로 실행되는지 말하지 않았음

#1806 이후에는 backend capability table이 모드별로 네이티브 실행, 변환, 거부를 결정합니다. 변환되는 NVFP4 체크포인트는 repack 경로를 로그로 남기지만, ROCm에서 네이티브로 로드되는 mxfp4는 아무것도 남기지 않았습니다. 그래서 로드 로그만으로는 gpt-oss가 네이티브로 실행되었는지 변환을 거쳤는지 알 수 없었습니다. 이슈의 첫 번째 인수 기준이 바로 이 로그를 요구합니다.

## 2. 변경 요약

- **`tests/rocm_mxfp4_quant.rs`** (신규, 421줄, `#![cfg(feature = "rocm")]`, 각 테스트는 ROCm이 아니면 건너뛰고 본문 동안 `streams::lock_default_device`를 잡습니다).
  - `capability_table_reports_mxfp4_native_on_rocm`: `quant_mode_support(Mxfp4) == Native`. 문서 주석에 수치 테스트가 계속 실패하면 이 항목을 `ConvertTo(Affine)`로 바꿔야 한다고 적었습니다.
  - `gpu_quantize_matches_cpu_bitwise`: 256x512와 4096x4096, f32와 bf16. packed와 scale의 shape를 확인한 뒤, 행 구간별로 GPU 바이트를 CPU quantizer와 비교합니다(256x512는 전체 행, 4096x4096은 처음과 마지막 64행. launch 실패와 끝부분에 못 미치는 grid를 모두 잡습니다).
  - `quantized_matmul_matches_dequantized_reference`: 256x512(hang 케이스), 512x256, gpt-oss-20b의 2880x2880. f32, f16, bf16. M = 1(qmv)과 8, 64(qmm). 기준값은 CPU에서 f32로 계산한 `x @ dequantize(w).T`입니다.
  - `gather_qmm_matches_per_expert_reference`: expert 16개, N = 256, K = 2880(gpt-oss의 reduction 폭, 32짜리 그룹 90개), top 4. T = 1과 8. f32와 bf16. `SwitchLinear::forward`가 넘기는 unsorted(`[T,1,1,K]`, 인덱스 `[T, top_k]`)와 sorted(`[T*top_k,1,K]`, expert 순으로 정렬된 인덱스) 두 형태를 모두 다룹니다.
  - 모든 결과는 `try_eval`을 거치므로 GPU 실패는 HIP 상태와 함께 테스트 실패가 됩니다. 허용 오차는 `max |want|` 대비 상대값으로 f32 1e-4, f16 5e-3, bf16 2e-2입니다.
- **`src/lib/mlxcel-core/src/layers.rs`** (+114/-).
  - `native_quantization_route_line(mode, backend) -> Option<String>`: 파싱된 block-float 모드가 `Native`이고, 같은 backend에 네이티브가 아닌 모드가 하나라도 있을 때만 한 줄을 돌려줍니다. 현재는 ROCm의 mxfp4와 mxfp8이 해당합니다.
  - `validate_quantization_mode_for_running_backend(mode)`: `gpu_backend_kind()`에 대해 `validate_quantization_mode_runnable`을 실행한 뒤, 정적 `[AtomicBool; QuantMode::ALL.len()]`로 프로세스당 모드별 한 번 경로 로그를 stderr에 출력합니다.
  - dense와 embedding 로더(`reconcile_quantization_layout_logged`)와 `QuantizedMultiLinear::from_weights`가 새 함수를 호출합니다.
  - 단위 테스트 `native_route_line_only_where_a_route_was_chosen`은 모든 backend와 모드를 돌면서 Metal, CUDA, CPU에서는 로그가 없음을 확인합니다.
- **`src/models/{gpt_oss,kimi_linear,switch_layers}.rs`**: 모델 crate의 로더 세 곳(`ExpertLinear`, `MultiLinear`, `SwitchLinear`)도 같은 함수를 씁니다.
- **`src/lib/mlxcel-core/src/hardware.rs`**: `quant_mode_support` 문서 주석이 ROCm mxfp4 항목의 근거로 커밋된 테스트를 가리킵니다. 테이블 자체는 바뀌지 않았습니다.
- **`docs/installation.md`**: ROCm 상태 표에 로그 문구와 테스트를 적었습니다. gpt-oss decode 수치를 3.6 tok/s(PR #1818)에서 약 8 tok/s(#2056 기준선에서 8.07)로 고쳤습니다.

커밋: `c273284c`는 테스트, 경로 로그, 문서를 추가합니다. `4785419d`는 리뷰 후속 조치로, 모드별 한 번 출력용 slot 조회에서 `expect`를 없애고 `hardware.rs` 주석을 revert 검사가 #1808 작업 중에 수행되었다는 뜻으로 고쳤습니다. 저장소가 그 검사를 재현한다고 읽히지 않게 하려는 것입니다.

## 3. 기술적 선택과 그 이유

### 고치기 전에 감사

이슈의 두 해결책(hang 근본 원인 추적, affine fallback 추가)은 모두 실패가 아직 있다는 가정에 기대고 있었습니다. 정확한 케이스를 먼저 다시 실행하자, 커널 조사와 메모리 비용이 드는 로드 시점 repack이 테스트 하나와 로그 한 줄로 줄었습니다. 계획 4단계의 affine fallback은 만들지 않았고, capability table은 `Native`를 유지합니다.

### 각 테스트가 이름 붙인 결함을 실제로 잡는지 확인

고쳐진 코드에서 통과하는 테스트는 망가진 코드에서 실패하지 않으면 아무것도 증명하지 못합니다. 항목 8, 10, 11을 하나씩 되돌렸고, 그때마다 대응하는 테스트가 실패했습니다. 항목 8은 NaN, 항목 10은 GPU 메모리 fault, 항목 11은 launch 에러였습니다. 항목 10의 결과는 `LOCAL_FIXES.md`에 기록된 실패(상대 오차 최대 1e34의 잘못된 수치)와 다릅니다. 이 shape에서는 잘못 읽은 T 타입 scale stride가 쓰레기 값을 만들기 전에 범위를 벗어나기 때문으로 보이지만, 더 추적하지는 않았습니다. #1804 이후 queue fault는 대기 중인 event의 에러가 되므로, 프로세스가 멈추지 않고 테스트 실패로 나타납니다. 이 revert 검사는 개발 중에 수행했고 저장소에 스크립트로 남아 있지 않습니다. 리뷰 후속 커밋이 `hardware.rs` 문구를 그에 맞게 고쳤습니다.

### CPU 기준값 비용을 제한

느린 쪽은 CPU입니다. 이 호스트에서 MLX CPU quantizer는 4096x4096에 약 90초, CPU 난수 생성기는 16M 샘플에 약 16초가 걸립니다. 테스트는 입력을 GPU에서 생성하고 평가된 같은 바이트를 두 장치에서 비교합니다. 큰 행렬은 CPU에서 행 구간만 양자화하는데, 양자화가 행 그룹마다 독립이라 가능합니다. `gather_qmm`의 출력 폭과 expert 수는 256과 16으로 줄였고 K는 gpt-oss의 2880을 유지했습니다. gpt-oss의 전체 shape는 실제 체크포인트로 end to end 확인합니다.

### 경로를 선택한 곳에서만 로그

모든 backend에서 로그를 찍으면 모든 모드를 네이티브로 실행하는 Metal과 CUDA의 로드 출력이 정보 없이 바뀝니다. 조건은 `backend == Rocm` 검사가 아니라 구조적입니다. 네이티브가 아닌 모드가 하나라도 있는 backend에서 `Native`인 파싱된 non-affine 모드를 로그로 남깁니다. 나중에 다른 backend에 변환 항목이 생기면 코드 변경 없이 그 backend의 네이티브 block-float 로드도 로그를 남깁니다. affine은 모든 backend의 기본이라 로그를 남기지 않습니다. 프로세스당 모드별 한 번이므로 레이어가 많은 MoE를 로드해도 한 줄입니다.

### 모든 양자화 로더의 진입점을 하나로

두 crate의 호출 지점 다섯 곳이 각각 `validate_quantization_mode_runnable(mode, gpu_backend_kind())`를 호출했습니다. 이제는 검증과 로그를 함께 하는 함수 하나를 호출합니다. runnable 검사는 여전히 backend를 명시적으로 받으므로 단위 테스트가 호스트에 의존하지 않습니다.

## 4. 검증

PR 작성자 실행(gfx1151):

- `cargo test --profile test-fast --features rocm --test rocm_mxfp4_quant -- --test-threads=1`: 4개 통과, 26초.
- `mlxcel generate -m models/mlx/gpt-oss-20b-MXFP4-Q4 --temp 0`: 일관된 답("The capital of France is Paris.")을 냈고 네이티브 경로 로그가 한 번 나왔습니다. Qwen3-0.6B-4bit(affine)는 로그를 출력하지 않았습니다.
- mlxcel-core `ffi_tests::compiled_qgelu_mlp_global_scale` 5개 통과(예전에 NVFP4 abort에 가려졌던 mxfp4 케이스 두 개 포함). `layers::tests`: 새 테스트 통과. 실패는 알려진 bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` 하나뿐입니다. 루트 lib `models::{switch_layers,gpt_oss,kimi_linear}`: 40개 통과. `dead_doc_pointers`: 통과.
- mlxcel-core와 mlxcel에 대한 clippy `-D warnings`, `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: 모두 통과.

오케스트레이터 검증(gfx1151, 브랜치를 origin/main `dfc59867` 위에서 실행):

- `make verify-rocm`의 모든 단계를 실행했습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` workspace clippy, ROCm smoke(32 토큰)가 통과했습니다.
- `verify-test-rocm`은 세 타깃에서 실패했으며, 실패는 알려진 기준 실패 37개와 정확히 같고 그 밖에는 없습니다. 35개는 `-p mlxcel-core --lib`에 있습니다. 그중 34개는 ROCm 포트가 아직 없는 fused paged-attention 테스트(#1814)이고, 나머지 하나는 bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`입니다. 남은 두 개는 #2037에서 온 `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit`와 `tests::family_order_is_exhaustive`입니다.
- `tests/rocm_mxfp4_quant.rs`는 전체 suite 안에서 실행되어 4개가 26.0초에 통과했습니다.
- mlxcel-core lib은 1797개가 통과했습니다. 이전보다 하나 많으며, 새 경로 로그 테스트입니다.

## 5. 학습 포인트

- **오래된 이슈는 작업 전에 먼저 재현해야 합니다.** 이슈에는 실패 세 개, hang bisect 계획, 변환 fallback이 있었지만 모두 이미 쓸모가 없었고, 그 이유는 이슈 등록 몇 시간 뒤 머지된 PR에만 기록되어 있었습니다. 이슈 refresh에서 추가한 "먼저 재현" 단계는 몇 분이 걸렸고 커널 조사를 없앴습니다. 실제로 남아 있던 빈틈은 이슈가 말한 것과 달랐습니다. 수정은 진짜였지만 이를 지키는 테스트가 없었습니다.
- **회귀 테스트는 수정을 되돌려서 증명합니다.** overlay 항목을 되돌린 상태로 각 테스트를 실행해야 그 테스트가 해당 항목을 지킨다는 것을 알 수 있습니다. 결함을 우연히 피해 가는 shape나 dtype(폭 128과 256인 기존 `ffi_tests` 케이스)은 어느 쪽에서도 통과합니다.
- **증상은 보고 경로에 따라 달라질 수 있습니다.** 항목 10을 되돌리면 `LOCAL_FIXES.md`에 기록된 잘못된 수치가 아니라 메모리 fault가 나왔고, #1804 이후 그 fault는 hang이 아니라 에러로 드러납니다. revert 검사가 기록과 다른 실패를 보이면, 테스트가 틀렸다고 결론 내리기 전에 어느 계층이 보고하는지 확인해야 합니다.
- **느린 CPU 기준값이 GPU hang처럼 보일 수 있습니다.** 감사 도중 debug 프로파일로 빌드한 테스트가 멈춘 것처럼 보였습니다. GPU 작업은 이미 끝났고, 시간은 CPU 기준값 코드(CPU 난수 생성, 양자화, dequantize, f32 기준 matmul)에서 쓰이고 있었습니다. 최적화하지 않은 빌드에서는 이 코드가 훨씬 느립니다. 제목이 GPU hang인 이슈에서는 쉽게 빠지는 오탐입니다. 테스트는 `--profile test-fast`(release 기반, `opt-level = 3`)로 실행하고, 입력을 GPU에서 생성하며, CPU 기준값은 행 구간으로 제한합니다. 진짜 hang은 테스트를 멈추게 하므로, 모듈 문서는 bisect할 때 timeout 아래에서 실행하라고 여전히 권합니다.
- **결정이 일어난 곳에서만 로그를 남깁니다.** 선택지가 없는 backend의 경로 로그는 잡음입니다. 조건을 capability table에서 끌어내면 로그가 테이블과 함께 움직입니다.

## 6. 주의 사항과 검증되지 않은 부분

- **Metal과 CUDA**는 실행하지 않았습니다. 로더 호출 지점 다섯 곳은 공유되지만, 그 backend에서는 모든 모드에 대해 `native_quantization_route_line`이 `None`을 돌려줍니다. 모든 backend에 대해 단위 테스트로 확인하므로 구조상 동작이 바뀌지 않습니다.
- **revert 증거는 저장소에 없습니다.** 세 가지 revert 검사는 작업 중 수동으로 했습니다. 이후 overlay 동기화에서 항목이 빠지면 커밋된 테스트가 잡지만, 각 테스트가 자기 항목을 잡는다는 주장은 PR 기록에 기대고 있습니다.
- **검사 범위.** group size 32만 테스트합니다. ROCm `gather_qmm`이 받는 유일한 fp group size이고, 다른 값은 에러를 던집니다. `gather_qmm`은 N = 256, expert 16개로만 테스트하며 gpt-oss의 전체 expert shape는 테스트하지 않습니다. expert-batched gather 경로(항목 9, `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=1`로 opt-in)는 실행하지 않습니다.
- **gfx1151에서만** 실행했습니다.
- **속도는 범위 밖입니다.** fused MoE launcher에 ROCm 포트가 없어서 gpt-oss는 generic `gather_qmm` 커널로 약 8 tok/s로 decode합니다. 이는 #1814의 몫입니다.

## 7. 남은 작업

- #1814: ROCm fused MoE와 paged-attention 포트. gpt-oss decode 속도와 fused paged-attention 기준 실패 34개가 여기에 속합니다.
- #1813: 항목 8, 10, 11을 fork에 upstream합니다. 그 전까지는 커밋된 테스트가 fork 동기화에서 이 항목들이 조용히 빠지는 것을 막습니다.

참고: #1808 (이 PR로 종료), #1801, #1804, #1806, #1813, #1814, #2037, PR #1818, PR #2056.
