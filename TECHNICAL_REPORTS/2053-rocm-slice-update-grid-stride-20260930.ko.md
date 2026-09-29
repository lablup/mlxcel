# 기술 보고서: PR #2053 - ROCm `slice_update_op_kernel`을 grid-stride로 변경

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: HIP C++ (ROCm overlay 커널), C++ 및 Rust (cxx bridge), Rust (통합 테스트), Markdown

**위험도**: 낮음 (ROCm 전용 overlay의 커널 하나를 기존 본문을 그대로 둔 채 루프로 감쌌습니다. 테스트 전용 bridge 함수 하나는 모든 백엔드에 있는 upstream MLX 함수를 호출합니다. 현재 reduce 커널에 도달하는 모델 경로는 없습니다)

## 요약

이슈 #2050 (에픽 #1801의 일부)은 ROCm 백엔드의 reduce 계열 `SliceUpdate`(Sum, Prod, Max, Min)에서 조용히 일어나는 잘림을 기록했습니다. `SliceUpdate::eval_gpu`는 grid를 256 스레드짜리 블록 65535개로 제한하는데, `slice_update_op_kernel`은 `gridDim`을 읽지 않고 스레드당 `NWORK`개 원소 묶음 하나만 처리했습니다. 따라서 65535 x 256 x `NWORK`개(`NWORK`가 1, 2, 4일 때 각각 16,776,960, 33,553,920, 67,107,840)보다 큰 update는 뒷부분이 입력값 그대로 남았고 오류도 없었습니다. #1874, #1823과 같은 형태의 결함이며, PR #2046이 fork의 `get_launch_args`를 삭제할 때 발견했지만 고치지 않고 남겨 둔 것입니다. 이 launch 지점은 grid를 인라인으로 제한하기 때문에 그 helper를 거치지 않았습니다.

이 PR은 커널 본문을 `NWORK`개 묶음 단위의 grid-stride 루프로 감싸고, 다른 제한된 ROCm 커널들과 마찬가지로 제한 자체는 유지합니다. mlxcel의 모델 코드는 None reduce만 쓰므로, 테스트 전용 bridge 함수 `slice_update_reduce`와 ROCm 통합 테스트를 추가했습니다. 테스트는 세 한계값 바로 위에서, 그리고 strided와 transposed 배치로 GPU 결과를 CPU 스트림과 int32 데이터에서 정확히 비교합니다. 커널 변경을 되돌리면 한계를 넘는 모든 케이스가 실패하고, 불일치 개수는 grid가 닿지 못한 원소 수와 정확히 같습니다. 변경을 적용하면 테스트 8개가 모두 통과합니다.

테스트를 작성하면서 별개의 결함이 하나 더 드러났습니다. ROCm의 `SliceUpdate::eval_gpu`는 소스 배열이 아직 참조되고 있는데도 소스 버퍼를 출력에 donate합니다. 이 결함은 #2052로 등록했고 이 PR에서는 고치지 않으며, 테스트가 이를 우회합니다.

## 1. 문제 정의

### 결함

`patches-rocm/mlx/backend/rocm/indexing.hip`의 `SliceUpdate::eval_gpu`에는 두 경로가 있습니다. None reduce(단순 덮어쓰기, mlxcel의 KV 캐시가 쓰는 경로)는 `copy_gpu_inplace`를 거치므로 영향이 없습니다. reduce 경로는 접힌(collapsed) update의 가장 안쪽 차원이 4 또는 2로 나누어지는지에 따라 `nwork`를 4, 2, 1 중에서 고르고, `num_blocks = min(ceil(ceil(update_size / nwork) / 256), 65535)`를 계산한 뒤 그 크기의 1차원 grid로 `slice_update_op_kernel`을 실행합니다.

커널은 스레드마다 시작 인덱스 `(blockIdx.x * blockDim.x + threadIdx.x) * NWORK` 하나를 계산하고, 거기서부터 최대 `NWORK`개 원소를 처리한 뒤 끝났습니다. `gridDim`을 보는 코드가 없었으므로, 제한은 16,776,960번째 이후의 묶음을 그냥 버렸습니다. 그 이후의 출력은 소스 값을 유지했는데, 이는 일부 입력에서 올바른 Max나 Min이 내는 값과 같기도 합니다. 그래서 update가 실제로 뒷부분을 바꾸지 않는 한 결함이 보이지 않습니다.

### 누가 도달하는가

reduce 변형은 MLX의 `slice_update_add`, `slice_update_prod`, `slice_update_max`, `slice_update_min`과 Prod에 대한 `SliceUpdate::vjp`가 만듭니다. 이 PR 이전의 mlxcel Rust bridge는 None reduce `slice_update`만 노출했으므로, 현재 이 커널에 도달하는 모델이나 서버 경로는 알려져 있지 않습니다. 결함은 vendored 백엔드 안에 잠재해 있고, ROCm fork(#1813)에 올려 보낼 후보입니다. 그래도 중요한 이유는, 앞으로 추가될 op이나 학습 및 gradient 경로, 또는 `slice_update_*`를 거치게 만드는 fork 동기화가 큰 KV 캐시나 임베딩에서는 흔한 크기에서 데이터를 조용히 잃을 수 있기 때문입니다.

### upstream은 어떻게 피하는가

MLX pin 81ba1c6a의 upstream CUDA는 같은 op을 `get_launch_args(upd, large, nwork)`로 실행합니다. 이 함수는 x를 제한하지 않고 `large`일 때 2차원 grid로 넘기며, 커널은 `cg::this_grid().thread_rank()`로 인덱싱합니다. ROCm 이식은 커널의 "스레드당 묶음 하나" 형태는 그대로 두고, overlay 다른 곳에서 쓰는 grid-stride 관례에 속하는 65535 블록 제한만 붙였습니다. 이 조합이 문제를 만들었습니다.

## 2. 변경 요약

- **`indexing.hip`, `slice_update_op_kernel`.** 본문이 이제 `for (base = tid * NWORK; base < update_size; base += gridDim.x * blockDim.x * NWORK)` 안에서 실행됩니다. 각 반복의 시작에서 `base`로부터 `out_idx`와 `update_idx`를 다시 계산합니다. 행 연속(row-contiguous)이면 그대로, 아니면 `elem_to_loc`으로, scalar update면 0입니다. 원소별 stride 증가를 포함한 안쪽 `NWORK` 루프는 바뀌지 않았습니다. 옮겨진 코드는 인덱스 prologue뿐이며, 루프 앞에서 루프 안으로 들어갔습니다.
- **`indexing.hip`, launch 지점.** 65535 제한은 유지합니다. 제한이 안전한 것은 커널이 grid-stride이기 때문이라는 주석을 달고 `LOCAL_FIXES.md` 항목 21을 가리킵니다.
- **Bridge: `slice_update_reduce`** (`mlx_cxx_bridge.{h,cpp}`, `mlxcel-core/src/lib.rs`의 `ffi` 블록). `reduce` 0, 1, 2, 3은 각각 `slice_update_add`, `slice_update_prod`, `slice_update_max`, `slice_update_min`을 호출합니다. 다른 값이면 `std::invalid_argument`를 던지고, cxx가 이를 유효 범위를 명시한 Rust `Err`로 바꿉니다. 반환형은 `Result<UniquePtr<MlxArray>>`이고, 테스트 전용이라고 문서화했습니다. 네 MLX 함수는 ROCm overlay뿐 아니라 upstream pin에도 있으므로 Metal과 CUDA에서도 bridge가 빌드됩니다.
- **`tests/rocm_slice_update_reduce.rs` (신규, 364줄, `#![cfg(feature = "rocm")]`).** `gpu_backend_kind()`가 ROCm이 아니면 건너뜁니다. 각 케이스는 CPU 스트림에서 먼저 op을 실행하고, 이어서 소스의 사본을 받은 GPU에서 실행한 뒤 `array_equal`로 비교합니다. 불일치가 있으면 개수와 첫 번째로 다른 flat 인덱스, 양쪽 값을 보고합니다. 케이스는 다음과 같습니다.
  - 원소 16,777,217개(NWORK=1), 33,554,434개(NWORK=2), 67,108,868개(NWORK=4)의 연속 Sum. 조금 더 큰 소스의 오프셋 3에 씁니다(전체 크기 slice는 MLX의 elementwise 지름길을 타서 `SliceUpdate`에 도달하지 않습니다).
  - `[4099, 4099]` 출력의 1..4098 열에 `[4099, 4097]` update로 Max (strided 출력, NWORK=1, 16,793,603개).
  - 같은 형태의 transposed update로 Sum (출력과 update 모두 strided이므로 매 묶음마다 두 인덱스를 `elem_to_loc`으로 계산).
  - 16,777,217개에 대한 scalar update Sum.
  - 한계 아래에서 strided 2차원 slice에 대해 모든 reduce op, 폭 8, 6, 5 (NWORK 4, 2, 1). Prod가 작게 유지되고 Max와 Min이 소스 값과 update 값을 섞어서 남기도록 값을 골랐습니다.
  - 범위를 벗어난 `reduce` 코드. 범위를 명시한 오류여야 합니다.
  각 테스트는 CPU 기준값을 만들 때 프로세스 전역 기본 장치를 바꾸므로, 본문 전체 동안 `streams::lock_default_device`를 잡습니다.
- **`LOCAL_FIXES.md`.** 새 항목 21에 결함, 수정, 측정, #2052 우회 방법을 기록하고 "Applies to the fork; to be proposed there"로 표시했습니다. 항목 19(`get_launch_args` 제거)는 더 이상 이 지점을 미수정으로 설명하지 않고 항목 21을 가리킵니다.

커밋 이력: `49c839fd`가 수정, bridge, 테스트, `LOCAL_FIXES.md` 항목입니다. `edb1ee47`(리뷰 반영)은 장치 lock 범위를 비교 구간에서 테스트 본문 전체로 넓히고 transposed-update 케이스를 추가합니다.

## 3. 기술적 선택과 그 이유

### grid-stride 루프, 제한은 유지

이슈는 제한을 없애는 방안도 검토했습니다. 그것만으로는 절벽이 옮겨갈 뿐입니다. `num_blocks`는 `int`이고, AMD는 grid 차원마다 스레드를 2^32 - 1개로 제한하며(`LOCAL_FIXES.md` 항목 11), y로 넘기려면 커널이 `blockIdx.y`도 읽어야 합니다. grid-stride 루프는 변경 하나로 모든 크기에서 올바르고, overlay가 이미 제한된 launch에 쓰는 관례입니다(`sort.hip`, `hadamard.hip`, `binary.hip`, `unary.hip`). 따라서 이 overlay에서 65535 제한을 본 사람은 그 뒤에 grid-stride 커널이 있다고 기대할 수 있습니다.

### 인덱스를 이어 가지 않고 묶음마다 다시 계산

비연속 경로는 차원마다 나눗셈이 들어가는 루프인 `elem_to_loc`이 필요합니다. `gridDim.x * blockDim.x * NWORK`개 원소만큼 건너뛰면서 인덱스를 이어 가려면 다차원 증가가 필요한데, 그것이 바로 `elem_to_loc`이 더 단순하게 계산하는 값입니다. 한계 아래에서는 스레드마다 반복이 한 번뿐이므로, 이미 동작하던 크기의 비용은 이전과 같은 prologue입니다.

### 묶음이 행을 넘지 않는 이유

안쪽 루프는 `out_idx`와 `update_idx`를 가장 안쪽 stride만큼 늘리는데, 이는 한 행 안에 머무는 동안에만 유효합니다. 각 묶음은 `NWORK`의 배수에서 시작하고, stride도 `NWORK`의 배수이며, launch 지점은 `NWORK`가 가장 안쪽 차원을 나눌 때만 그 값을 고릅니다. 따라서 모든 묶음은 행 안의 `NWORK` 경계에서 시작해 행이 끝나기 전에 끝납니다. 이전 커널은 묶음 하나에 대해 같은 논리에 기댔고, 루프는 이를 모든 묶음에 대해 유지합니다.

### 템플릿 인스턴스를 추가하지 않음

기존 모든 인스턴스에서 `IdxT`는 `int64_t`이므로, allocator가 담을 수 있는 어떤 크기에서도 `base`와 stride가 넘치지 않고 더 넓은 인덱스 타입이 필요 없습니다. `LOCAL_FIXES.md` 항목 17은 이 번역 단위의 컴파일 비용이 얼마나 큰지 기록하고 있으므로, 수정은 커널 본문만 바꿉니다.

### 운영 경로가 아닌 테스트 전용 bridge

Rust에서 reduce 커널을 테스트하려면 거기에 도달할 수 있어야 합니다. 숫자 `reduce` 코드를 받는 `slice_update_reduce`를 bridge에 추가하는 것이 그렇게 할 수 있는 가장 작은 표면이고, `Result`를 반환하므로 잘못된 코드가 다른 op으로 조용히 빠지지 않고 오류가 됩니다. 지원되는 API로 오해하지 않도록 테스트 전용이라고 문서화했습니다. 운영 코드에서 쓴다면 enum이 필요합니다.

### int32에서 CPU 스트림과 정확히 비교

int32에서는 Sum, Prod, Max, Min이 두 장치 모두에서 정확하므로 `array_equal`이 맞는 검사이고, 불일치는 반올림이 아니라 실제 결함입니다. 불일치 개수는 측정값 역할도 합니다. 이전 커널에서는 `65535 x 256 x NWORK`를 넘는 원소 수와 같으므로, 각 실패가 다른 버그가 아니라 제한 때문임을 보여 줍니다.

## 4. 검증

작성자 실행 (gfx1151, Radeon 8060S):

- `indexing.hip`만 `main`으로 되돌린 상태에서 `cargo test --features rocm --test rocm_slice_update_reduce -- --test-threads=1`: 한계를 넘는 여섯 케이스가 실패하고, 한계 아래 케이스와 오류 케이스는 통과했습니다(리뷰 중 추가된 transposed-update 케이스는 별도로 되돌린 빌드에서 같은 방식으로 확인했습니다). 불일치 개수는 grid가 닿지 못한 원소 수와 정확히 같습니다. 세 연속 Sum은 257, 514, 1028, scalar Sum은 257, strided Max와 transposed-update Sum은 16,643입니다. NWORK=1 연속 Sum의 첫 불일치는 flat 인덱스 16,776,963으로, 한계에 slice 오프셋 3을 더한 값입니다.
- 변경 적용 후: 8개 통과, 0개 실패 (debug 빌드에서 약 155초).
- `cargo clippy --features rocm --test rocm_slice_update_reduce -- -D warnings`, `cargo clippy -p mlxcel-core --features rocm --lib -- -D warnings`: 경고 없음.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`, `cargo test --features rocm --test dead_doc_pointers`: 통과.

오케스트레이터 검증 (gfx1151, origin/main `d913fc7d` 위의 브랜치):

- `make verify-rocm`의 모든 단계를 실행했습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` workspace clippy, ROCm smoke(32 토큰)가 통과했습니다.
- `verify-test-rocm`은 세 타깃에서 이미 알려진 기준 실패 37개와 정확히 같은 실패만 냈습니다.
  - `-p mlxcel-core --lib`: 35개 실패. 34개는 ROCm 이식이 없는 fused paged-attention 테스트(#1814)이고, 나머지 하나는 bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`입니다.
  - `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit`, `tests::family_order_is_exhaustive`, 둘 다 #2037에서 온 실패입니다.
- `tests/rocm_slice_update_reduce.rs`는 test-fast 프로필의 전체 스위트 안에서 실행되어 63.4초에 8개 모두 통과했습니다.

## 5. 학습 포인트

- **제한(clamp)은 커널과의 계약입니다.** 65535 블록 제한은 grid-stride 커널 앞에서만 올바르고, 타입 시스템은 이를 알려 주지 않습니다. #2046은 이 계약을 숨기던 helper를 없앴지만, 인라인 제한도 똑같이 계약을 깰 수 있습니다. 이번에 launch 지점에 단 주석은 다음 사람이 읽을 자리에 계약을 적어 둡니다. overlay에 새로 추가되는 제한된 launch도 같은 방식으로 확인해야 합니다.
- **불일치 개수를 증거로 삼습니다.** 다르다는 사실만이 아니라 몇 개가 다른지 보고하면서, "수정 없이 테스트가 실패한다"가 "제한이 버리는 원소 수만큼 정확히 실패한다"가 되었습니다. 우연한 실패를 배제하고 각 케이스가 어떤 geometry를 검사하는지 보여 줍니다.
- **크기만이 아니라 배치(layout)를 테스트합니다.** 연속 케이스는 행 연속 인덱스 경로만 검사합니다. strided Max는 출력 쪽 `elem_to_loc`을, 리뷰에서 추가된 transposed-update Sum은 update 쪽 `elem_to_loc` 재계산을 검사하는 유일한 케이스입니다. 뒤쪽 묶음에서 update 인덱스를 잘못 계산하는 grid-stride 재작성은 다른 모든 케이스를 통과했을 것입니다.
- **첫 테스트 버전은 다른 이유로 실패했습니다: 소스 donation (#2052).** 첫 초안은 같은 `src` 배열을 GPU op과 이어지는 CPU 기준값에 함께 썼습니다. 한계 아래 케이스를 포함해 모든 케이스가 실패했습니다. 예를 들어 `src[3] = 3`, update 1에서 GPU는 4, CPU는 5였고, 한계 아래 Prod는 곱을 두 번 적용했습니다. GPU op이 결과를 `src`의 버퍼에 써 버린 것입니다. ROCm의 `SliceUpdate::eval_gpu`는 `in.data_shared_ptr().use_count() == 1`이고 소스가 연속이면 소스를 donate합니다(`out.copy_shared_buffer(in)`). 이 검사는 데이터 버퍼의 소유자만 세고, 호출자가 `array` 자체를 아직 참조하는지는 보지 않습니다. upstream은 `array_desc_.use_count() == 1`과 데이터 소유자 하나를 모두 요구하는 `array::is_donatable()`로 판단하고, upstream CUDA의 `SliceUpdate::eval_gpu`는 `copy_gpu`를 거치므로 살아 있는 소스를 donate하지 않습니다. `backend/gpu/primitives.cpp`의 `DynamicSliceUpdate::eval_gpu`에도 같은 데이터 전용 검사가 있고, HIP graph capture 중에는 의도적으로 강제 donation을 합니다. 테스트 측면의 교훈은 ROCm overlay에서 GPU 대 CPU 비교를 할 때 donate할 수 있는 op과 입력을 공유하면 안 된다는 것입니다. 테스트는 이제 GPU op에 사본을 주고 CPU 기준값을 먼저 계산하며, #2052를 가리키는 주석을 답니다. 백엔드 측면의 교훈은 "버퍼 소유자가 하나"가 "이 배열을 아무도 보지 않는다"와 같지 않다는 것입니다.
- **기준값은 변형될 수 없는 쪽에서 먼저 계산합니다.** 사본에 더해 CPU 기준값을 GPU op보다 먼저 실행하므로, 앞으로 GPU op이 다른 방식으로 입력을 바꾸게 되더라도 기준값은 이미 깨끗한 데이터에서 계산되어 있습니다.

## 6. 검증되지 않은 부분

- **Metal과 CUDA.** 이 호스트에서는 사용할 수 없습니다. 이 경로에서 바뀐 것은 pin에 있는 upstream MLX 함수를 호출하는 `slice_update_reduce` bridge 함수뿐입니다. 커널 변경은 ROCm 전용이고 테스트는 `#![cfg(feature = "rocm")]`입니다.
- **다른 AMD GPU와 ROCm 버전.** gfx1151만 측정했습니다. 수정은 geometry에 관한 것이라 아키텍처에 의존하지 않지만, 다른 장치에서 실행하지는 않았습니다.
- **성능.** 벤치마크는 실행하지 않았습니다. 한계 아래에서는 스레드마다 이전과 같은 prologue로 한 번만 반복하므로 변화가 없을 것으로 봅니다. 한계 위에서는 이전 커널이 틀렸으므로 비교할 기준이 없습니다.
- **float dtype과 `SliceUpdate::vjp`의 Prod 경로.** 테스트는 정확한 비교를 위해 int32를 쓰고 `slice_update_*`를 직접 호출합니다. 커널은 원소 타입으로 템플릿화되어 있고 루프는 원소 타입에 의존하지 않지만, float dtype과 vjp 호출자는 실행하지 않았습니다.
- **별도 release 빌드.** `cargo build --release --features rocm`를 따로 실행하지는 않았고, `make verify-rocm`이 release 빌드를 포함합니다.

## 7. 남은 작업

- **#2052, `SliceUpdate`와 `DynamicSliceUpdate`의 소스 donation.** 두 지점의 데이터 전용 use-count 검사를 `in.is_donatable()`로 바꾸고, `DynamicSliceUpdate`의 `graph_active()` 예외는 여전히 필요하면 유지합니다. None reduce 경로는 mlxcel의 KV 캐시가 쓰는 경로이므로, 먼저 `slice_update` 이후에 갱신 전 배열을 읽는 mlxcel 경로(KV 캐시 스냅샷, prefix 캐시, speculative decoding 롤백)가 있는지 확인해야 합니다. 있다면 #2052는 잠재 결함이 아니라 실제 정확성 버그입니다. 이 지름길은 KV 캐시 append 속도 때문에 추가되었을 가능성이 높으므로, gfx1151에서 수정 전후의 decode 처리량을 측정해야 합니다. 인수 테스트는 `src`를 잡고 GPU `slice_update`(None과 reduce 하나)를 실행한 뒤 `src`가 바뀌지 않았는지 확인해야 합니다. 이것이 머지되면 `tests/rocm_slice_update_reduce.rs`의 사본은 없앨 수 있습니다.
- `LOCAL_FIXES.md` 항목 21을 #1813의 일부로 ROCm fork에 제안합니다.
- 이 PR과 무관한 기준 테스트 실패: #1814(fused paged-attention 이식 34개), bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` 실패, #2037의 두 실패.

참고: #2050 (이 PR로 종료), #1801, #1813, #1823, #1874, #2046, #2052.
