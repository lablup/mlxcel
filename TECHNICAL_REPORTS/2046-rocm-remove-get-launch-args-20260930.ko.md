# 기술 보고서: PR #2046 - 그리드를 조용히 잘라내던 ROCm get_launch_args 제거

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: C++ (HIP 헤더), Markdown

**위험도**: 낮음 (호출자가 없는 함수를 지우고 deleted 선언을 추가합니다. overlay 안에서만 바뀌고 어떤 커널의 실행 geometry도 바뀌지 않으며, Metal과 CUDA 빌드는 `patches-rocm/`을 복사하지 않습니다)

## 요약

이슈 #1874 (에픽 #1801의 일부)는 ROCm overlay의 `mlx/backend/rocm/kernel_utils.hpp`에 있는 `get_launch_args`가 그리드를 256 스레드 블록 65535개로 제한하면서, 커널이 grid-stride여야 한다는 사실을 어디에도 밝히지 않는다는 점을 지적했습니다. 스레드마다 인덱스 하나를 맡는 커널을 이 함수로 실행하면 65535 x 256을 넘는 원소는 전부 쓰이지 않고, 출력에는 할당 당시의 내용이 그대로 남습니다. 호출하는 곳이 없어서 아직 깨진 것은 없었습니다. 위험은 upstream CUDA에 같은 이름의 함수가 있고, 그 함수는 x를 제한하지 않으며 `mlx/backend/cuda` 전반에서 호출된다는 데 있었습니다. 그쪽에서 이식한 커널은 fork의 버전으로 그대로 컴파일되고, 아무 경고 없이 잘림을 물려받게 됩니다.

이 PR은 두 overload를 모두 지우고 그 자리에 `template <typename... Args> void get_launch_args(Args&&...) = delete;`와 이유를 적은 주석을 둡니다. 이제 upstream 형태든 fork 형태든 어떤 호출도 그 주석 위치에서 컴파일에 실패합니다. 이 결정은 `LOCAL_FIXES.md` 항목 19로 기록되었고, #1813의 upstream 반영 후보입니다. 리뷰 중에 헬퍼 밖에서 같은 결함이 실제로 쓰이는 곳 하나를 찾았습니다. `indexing.hip`에 있는 `SliceUpdate::eval_gpu`의 reduce 연산 경로입니다. 이 PR은 커널 geometry를 바꾸지 않으므로 이 부분은 기록만 하고 고치지 않았습니다.

## 1. 문제 정의

fork의 헬퍼는 `num_blocks = ceil(ceil(size / work_per_thread) / 256)`을 계산한 뒤 `num_blocks = std::min(num_blocks, 65535)`를 적용했습니다. `shape`, `strides`, `large`는 무시했고 블록 크기는 256으로 고정했습니다. 제한된 그리드는 커널이 `gridDim.x * blockDim.x` 단위로 반복할 때만 올바릅니다. 헬퍼의 이름, 시그니처, 주석 어디에도 이 계약이 드러나지 않았습니다.

이 백엔드의 커널은 이 헬퍼를 쓰지 않습니다. `binary.hip`와 `unary.hip`는 각자 grid-stride 루프를 작성합니다. #1856에서 추가된 두 커널 `hadamard.hip`와 `sort.hip`는 제한된 geometry를 직접 계산하고, 그 제한과 루프를 연결하는 주석을 달고 있습니다. 이슈가 걱정한 것은 그럴듯해 보이는 이 헬퍼를 다음에 집어 들 사람, 특히 이미 같은 이름의 함수를 호출하는 CUDA 커널을 이식하는 사람이었습니다.

upstream 정의(고정된 버전의 `mlx/backend/cuda/kernel_utils.cu:33-50`)는 다른 함수입니다. x를 제한하지 않고(CUDA의 x 한도는 2^31 - 1), `max_block_dim`을 받으며, `large`를 `get_2d_grid_dims`로 처리합니다. upstream에서 65535라는 상수는 `get_launch_args_general`의 `max_grid_yz_dim`에만 나오고, 그 함수는 y의 초과분을 버리지 않고 z로 넘깁니다. 즉 fork의 헬퍼는 upstream과 같아 보였지만, 조용히 틀린 출력을 내는 바로 그 방식으로 다르게 동작했습니다. 이는 #1823(`LOCAL_FIXES.md` 항목 13)과 같은 실패 형태로, 그때는 `quantized_matmul`이 새로 할당된 버퍼의 내용을 그대로 돌려주었습니다.

## 2. 변경 요약

- `src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/kernel_utils.hpp` (13줄 추가, 19줄 삭제): `get_launch_args`의 두 overload(크기 버전과 `const array&` 버전)를 지웠습니다. 그 자리에 `template <typename... Args> void get_launch_args(Args&&...) = delete;`와 주석을 두었습니다. 주석은 `LOCAL_FIXES.md` 항목 19를 가리키고, 예전 헬퍼가 왜 위험했는지, 이식된 호출이 왜 문제를 물려받는지, 그리고 호출 지점에서 제한한 그리드는 그 커널이 grid-stride일 때만 올바르다는 점을 적고 `hadamard.hip`와 `sort.hip`를 예로 듭니다.
- `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`: 새 항목 19에 결함, 이슈의 세 선택지와 삭제를 고른 이유, 리뷰에서 찾은 `indexing.hip` 사례, 그리고 fork 동기화로 정의가 되살아나면 다시 지워야 한다는 점을 기록했습니다. 표준 문구 "Applies to the fork; to be proposed there"로 끝납니다.

두 번째 커밋(`979f114c`)은 문서만 고칩니다. 첫 초안은 백엔드에서 65535 블록으로 제한하는 모든 곳이 grid-stride 커널을 실행한다고 적었습니다. 리뷰에서 `indexing.hip`는 그렇지 않다는 것이 드러나 헤더 주석과 항목 19를 고쳐 썼고, `get_launch_args_general` 비교도 upstream이 y를 z로 넘긴다고 바로잡았습니다.

## 3. 기술적 선택과 그 이유

### 이름을 바꾸거나 확장하지 않고 지웁니다

이슈는 세 가지를 제시했습니다. 두 overload를 지우기, grid-stride 요구를 드러내는 이름(예: `get_grid_stride_launch_args`)으로 바꾸기, upstream의 `get_launch_args_general`처럼 초과분을 다른 그리드 차원으로 넘기기입니다. PR은 삭제를 택했습니다. 호출자가 없으니 무엇도 퇴행시킬 수 없고, 나머지 두 방식은 upstream과 닮기만 한 헬퍼를 남깁니다. 이름을 바꿔도 모든 호출자가 계약을 읽고 지켜야 합니다. y로 넘기는 방식은 커널이 `blockIdx.y`까지 읽지 않는 한 스레드당 인덱스 하나인 커널을 고치지 못하므로, 버그를 없애는 것이 아니라 옮기게 됩니다. upstream geometry를 충실히 옮기는 것도 그대로 끼워 넣을 수 있는 해법이 아닙니다. AMD는 그리드 차원마다 2^32 - 1 스레드로 제한하고(항목 11), 이는 CUDA의 한도와 다릅니다.

### 선언을 없애지 않고 deleted 선언을 둡니다

이름을 완전히 없애면 이식된 호출은 "use of undeclared identifier"로 실패합니다. 이를 보고 누군가 history에서 예전 헬퍼를 되살리면 버그도 함께 돌아옵니다. deleted variadic template은 어떤 인자 목록에도 맞으므로, fork 형태와 upstream 형태의 호출이 모두 "call to deleted function"으로 실패하고, 컴파일러는 이유를 설명하는 주석이 있는 선언을 가리킵니다. 기록된 결정이 필요한 순간에 컴파일러가 직접 강제하는 형태가 됩니다.

### 실제 사례는 기록만 하고 여기서 고치지 않습니다

`SliceUpdate::eval_gpu`의 reduce 연산 경로(Sum, Prod, Max, Min)는 `num_blocks = min(ceil(ceil(update_size / nwork) / 256), 65535)`를 계산하고 `slice_update_op_kernel`을 실행합니다. 이 커널은 스레드마다 시작 인덱스 하나(`(blockIdx.x * blockDim.x + threadIdx.x) * NWORK`)를 계산해 최대 `NWORK`개 원소를 처리하고, `gridDim`을 읽지 않습니다. `nwork`는 가장 안쪽의 합쳐진 차원이 4 또는 2로 나누어지는지에 따라 4, 2, 1이므로, 16,776,960 x `NWORK`개(약 16.8M, 33.6M, 67.1M개)를 넘는 update는 뒷부분이 적용되지 않습니다. #1874가 설명한 결함이 실제 호출 지점에 있는 것입니다. 이슈가 커널 geometry를 범위 밖으로 명시했으므로, PR은 이 지점을 항목 19와 PR 본문에 기록하고 별도 이슈로 남깁니다. 덕분에 이 변경은 런타임에서 순수한 no-op으로 유지됩니다.

### fork 쪽 upstream 반영 후보로 기록합니다

결함은 upstream MLX가 아니라 fork(`75915908`에서 확인)에 있으므로, 항목 19는 upstream MLX 보고가 아니라 #1813의 fork upstream 반영 목록에 올라갑니다. fork 동기화로 정의가 다시 들어오면 지워야 한다는 메모는 overlay를 갱신할 때 이 수정이 사라지지 않도록 막습니다.

## 4. 검증

작성자 실행 결과, PR 본문 기준(gfx1151, Radeon 8060S):

- `cargo build --release --features rocm`: 처음부터 빌드했을 때와 두 번째 커밋 뒤 증분 빌드 모두 통과. 빌드 트리의 헤더 사본이 overlay와 일치합니다.
- `hipcc -fsyntax-only --offload-arch=gfx1151`로 `get_launch_args(arr, false)`와 `get_launch_args(n, shape, strides, false, 4)`를 호출하는 translation unit을 검사한 컴파일 실패 probe: 패치된 헤더에서는 둘 다 "call to deleted function"으로 거부되었습니다. 같은 파일이 `origin/main`의 헤더로는 컴파일되므로, probe가 두 상태를 실제로 구분합니다.
- `grep -rn get_launch_args src/lib/mlx-cpp/patches-rocm/`: deleted 선언과 그 주석만 남았습니다.
- Qwen3-0.6B-4bit로 `make verify-rocm-smoke`: OK.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: 통과.

오케스트레이터 검증(gfx1151, origin/main `fcf5f4c3`에 이 브랜치를 더한 상태):

- `make verify-rocm`이 모든 단계를 실행했습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` 워크스페이스 clippy, ROCm smoke(32 토큰)가 통과했습니다.
- `verify-test-rocm`은 세 타깃에서 실패했고, 실패는 알려진 기준선 37건과 정확히 같으며 그 밖에는 없습니다.
  - `-p mlxcel-core --lib`: 35건. 34건은 ROCm 포트가 없는 fused paged-attention 테스트(#1814에서 추적)이고, 나머지 하나는 bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`입니다.
  - `-p mlxcel --lib`: #2037에서 온 `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit`.
  - `-p mlxcel --bin mlxcel`: #2037에서 온 `tests::family_order_is_exhaustive`.

수용 기준은 기준선이 바뀌지 않을 것을 요구했습니다. 이슈에 적힌 기준선(실패 타깃 하나, #1806의 NVFP4 abort)은 이제 맞지 않습니다. #1806은 그 뒤 고쳐졌고, 현재 목록은 위의 37건이며 #2033 실행과 같습니다. 이 변경은 아무도 호출하지 않던 코드를 지우는 것이라 런타임 차이는 예상되지 않았고, 실제로도 나타나지 않았습니다.

## 5. 학습 포인트

- **upstream 이름을 가진 헬퍼는 읽는 사람의 머릿속에 upstream의 계약을 함께 가져옵니다.** fork의 `get_launch_args`는 CUDA 헬퍼와 이름은 같았지만 동작은 달랐습니다. `mlx/backend/cuda`에서 이식하는 사람은 이름을 믿게 됩니다. 로컬 함수가 upstream 계약을 지킬 수 없다면 upstream 이름도 쓰지 않아야 합니다.
- **`= delete`는 결정을 적어 두는 방법입니다.** 주석을 단 deleted 함수는 이름을 예약해 두고, 모든 호출 형태를 거부하며, 이유를 컴파일러 오류에 드러냅니다. 코드를 지우고 아무도 되살리지 않기를 바라는 것보다 강합니다.
- **그리드 제한과 grid-stride 루프는 한 몸입니다.** 그리드를 제한하는 것은 커널이 `gridDim`을 읽을 때만 올바릅니다. `hadamard.hip`와 `sort.hip`는 둘을 나란히 두고 주석으로 묶었고, `indexing.hip`의 `SliceUpdate`는 제한만 있을 때 무슨 일이 생기는지 보여 줍니다.
- **전체를 아우르는 주장은 모든 지점과 대조해 검토합니다.** 항목 19의 첫 초안은 백엔드의 모든 제한이 grid-stride 커널과 짝을 이룬다고 적었습니다. `65535`를 한 번 grep하고 실행되는 커널을 하나씩 읽어 보자 반례가 나왔습니다. 주장은 바로잡혔고, 그 과정에서 실제 버그 하나가 드러났습니다.

## 6. 검증하지 않은 것

- **#1826 gfx1151 정확도 매트릭스는 실행하지 않았습니다.** 이슈의 검증 절차는 이를 요구했습니다. PR은 호출자가 없는 코드를 지우고 커널을 바꾸지 않으므로 차이는 예상되지 않지만, 매트릭스 결과 자체는 기록되지 않았습니다.
- **Metal과 CUDA는 실행하지 않았습니다.** 두 빌드 모두 `patches-rocm/`을 복사하지 않으므로 구조상 영향이 없습니다.
- **`indexing.hip`의 잘림은 재현하지 않았습니다.** 임계값(reduce 타입 `SliceUpdate`에서 16.8M x `NWORK`개 초과)은 실행 코드와 커널을 읽어 얻은 것입니다. 그만큼 큰 update를 만드는 테스트는 없고, mlxcel의 모델 경로가 여기에 도달하는지는 확인하지 않았습니다.

## 7. 남은 작업

- `indexing.hip`의 `SliceUpdate::eval_gpu` reduce 연산 경로에 대한 별도 이슈: `slice_update_op_kernel`을 grid-stride로 만들거나, 제한을 없애고 AMD의 차원별 한도 안에서 그리드 크기를 정한 뒤, 임계값을 넘는 테스트를 추가합니다.
- #1813의 일부로 항목 19를 ROCm fork에 반영합니다.
- #1814: fused paged-attention 커널의 ROCm 포트(`mlxcel-core` 실패 35건 중 34건).
- ROCm에서 bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` 실패를 분류하고, #2037에서 생긴 두 실패(`gelu_approx_matches_mlx_nn_bit_for_bit`, `family_order_is_exhaustive`)를 고칩니다.
