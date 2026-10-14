# 기술 보고서: PR #2237 - ROCm에서 2^32개 이상 원소의 strided 복사

**날짜**: 2026-10-08

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 코드 head `e5f4419c`, main `2aa5211f` 기준, PR 열림, 머지 대기. #2184 종료.

**언어**: C++/HIP(ROCm 오버레이 `copy/*.hip`, `copy/copy.hpp`), Rust(신규 `tests/rocm_strided_copy_grid.rs`), Markdown(`patches-rocm/LOCAL_FIXES.md`)

**위험도**: 낮음. 256스레드 블록 65535개(16.7M 원소) 아래에서는 각 복사 커널 인스턴스가 기존의 원소당 스레드 하나 본문 그대로이고, 4M 원소 전치는 main과 잡음 범위 안에서 같았다. 동작이 바뀌는 곳은 상한을 넘어 루프를 도는 복사와, 목적지 전체 대신 복사 영역으로 launch 크기를 정하게 된 dynamic 복사다. `copy_contiguous`는 `INT32_MAX` 위에서 인덱스 폭이 바뀐다. 일반 복사 커널 4개의 인스턴스 수는 두 배가 된다. Metal과 CUDA 코드 경로는 건드리지 않는다.

## 요약

ROCm strided 복사 launcher 3개가 `int` 블록 수와 루프 없이 원소당 스레드 하나로 그리드를 잡았다. HIP은 전체 스레드 수가 2^32에 이르는 launch를 거부하므로(`hipErrorInvalidConfiguration`, "invalid configuration argument") 2^32개 이상 원소의 strided 복사는 모두 실패했다. `copy_contiguous.hip`은 이미 그리드를 65535블록으로 제한하고 루프를 돌았다. #2184는 #2153의 paged 슬랩 테스트를 쓰다가 이를 발견했고, 그 테스트는 2^31 원소짜리 절반 둘을 concatenate하는 방식으로 우회하고 있었다.

`copy_general_dynamic`에는 두 번째 버그가 있었다. launch를 `out.size()`로 정한 것이다. `DynamicSliceUpdate`는 목적지 전체를 `out`으로 넘기므로 2^32 원소 슬랩에 블록 하나를 쓰는 갱신도 2^32 스레드를 launch해 실패했고, 그보다 작은 크기에서는 갱신 범위를 넘어선 스레드가 모두 갱신의 wrap된 복사본을 다시 썼다.

수정은 `rocm::copy_grid_blocks`(64비트 블록 수를 [1, 65535]로 제한), `copy_grid_loops`, 디바이스 헬퍼 `for_each_copy_index<kLoop>`를 추가한다. 일반 복사 커널 4개는 `kLoop`을 템플릿 인자로 받는다. 상한 아래에서는 기존 본문을 유지하고, 상한을 넘으면 스레드가 64비트 카운터로 건너뛰며 순회한다. dynamic launch는 복사 shape의 곱으로 정한다. `copy_contiguous`도 같은 헬퍼를 쓰고 `INT32_MAX` 위에서 64비트 인덱스로 바꾼다.

gfx1151에서 `#[ignore]` 테스트 4개(각 8 GiB)는 main에서 invalid configuration으로 실패하고 수정 후 통과한다. 42.6M 원소 빠른 테스트 3개는 통과하며, 루프를 한 번만 돌게 줄이면 실패한다. f16 전치 `contiguous`와 `concatenate` 마이크로벤치는 main과 0.4% 이내다. 오케스트레이터의 `e5f4419c` `make verify-rocm`은 통과했다(스위트 163개, 12,194개 통과, 0개 실패, 403개 무시, 스모크 OK). Metal과 CUDA는 검증하지 못했다.

## 1. 문제 정의

### 1.1 제한 없는 그리드

launcher 3개가 블록 크기 256으로 `int num_blocks = (size + block_size - 1) / block_size`를 계산하고 원소당 스레드 하나를 launch했다.

- `copy_general_input.hip`(`copy_g_byval`, strided 입력에서 연속 출력으로),
- `copy_general.hip`(`copy_gg_byval`, strided에서 strided로),
- `copy_general_dynamic.hip`(`copy_gg_dynamic_nd`와 `copy_gg_dynamic`, `DynamicSliceUpdate` 경로).

HIP은 `gridDim.x * blockDim.x`가 2^32에 이르는 launch를 거부한다. 2^32 원소 복사는 `{16777216, 1, 1}` 그리드에 `{256, 1, 1}` 블록을 launch해 실패했다(main `b7116d1d`에서, f16 `[131072, 32, 8, 128]` 배열에 블록 하나를 더한 `concatenate`를 `AMD_LOG_LEVEL=3`으로 재현). `int` 블록 수는 2^39 원소를 넘으면 오버플로하기도 한다. `copy_contiguous.hip`은 그리드를 65535블록으로 제한하고 루프로 데이터를 순회하므로 그리드 때문에 한계에 걸린 적이 없다.

### 1.2 목적지 크기로 정한 dynamic 복사

`copy_general_dynamic`은 `size = out.size()`로 두었다. `DynamicSliceUpdate`에서 `out`은 목적지 전체이고 `shape`는 갱신의 shape다. 커널은 갱신의 shape로 스레드 인덱스를 분해하므로, 갱신의 원소 수를 넘어선 스레드는 그 shape에 대한 나머지로 인덱스를 다시 계산해 갱신의 wrap된 복사본을 덮어썼다. 어떤 크기에서도 낭비였다. 목적지가 2^32 원소이면 2^32 스레드를 launch해 실패했고, 갱신이 블록 하나(paged KV 캐시 쓰기 패턴)여도 마찬가지였다. 업스트림 CUDA의 pin `81ba1c6a`도 `copy_general_dynamic`을 `out.size()`로 launch한다. 거기서 launch 한계에 닿는지는 알려져 있지 않다.

### 1.3 영향

슬랩이 면당 2^32 원소를 넘는 paged KV(#2153 이후 지원)는 슬랩 전체를 strided 복사하는 어떤 연산에서든 이 문제를 만날 수 있었다. 이슈는 현재 그런 프로덕션 경로를 찾지 못했으며, 철저히 검증한 것은 아니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `copy/copy.hpp` | 신규 `kMaxCopyBlocks`(65535), `copy_grid_blocks`, `copy_grid_loops`, `for_each_copy_index<kLoop>`. |
| `copy/copy_general_input.hip`, `copy_general.hip` | `copy_g_byval`과 `copy_gg_byval`이 `bool kLoop`을 받고 본문을 `for_each_copy_index`로 실행한다. launch는 제한된 그리드를 쓰고 `copy_grid_loops`로 인스턴스를 고른다. |
| `copy/copy_general_dynamic.hip` | `copy_gg_dynamic_nd`와 `copy_gg_dynamic`도 같다. launch 크기는 `shape`의 곱이고, 빈 복사는 일찍 반환한다. |
| `copy/copy_contiguous.hip` | 블록 수를 `copy_grid_blocks`로 계산. `UINT32_MAX` 대신 `INT32_MAX` 위에서 64비트 인덱스. |
| `tests/rocm_strided_copy_grid.rs` | 신규: 빠른 테스트 3개와 `#[ignore]` 테스트 4개(5절). |
| `patches-rocm/LOCAL_FIXES.md` | 항목 42(신규). |

커밋 1개, 파일 7개, 추가 483줄, 삭제 82줄.

## 3. 설계

### 3.1 헬퍼

```cpp
inline constexpr size_t kMaxCopyBlocks = 65535;

inline uint32_t copy_grid_blocks(size_t size, size_t per_block) {
  size_t blocks = (size + per_block - 1) / per_block;
  return static_cast<uint32_t>(std::min(std::max(blocks, size_t{1}), kMaxCopyBlocks));
}

inline bool copy_grid_loops(size_t size, size_t per_block) {
  return size > kMaxCopyBlocks * per_block;
}
```

블록 수는 `size_t`로 계산해 제한하므로 오버플로하지 않고 2^32 스레드 한계(65535 * 256) 아래에 머문다. `copy_grid_loops`는 제한된 그리드의 한 번 순회로 `size`를 덮지 못할 때만 참이다.

### 3.2 kLoop 분리

```cpp
template <bool kLoop, typename F>
__device__ __forceinline__ void for_each_copy_index(int64_t size, F&& body) {
  if constexpr (!kLoop) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) body(int64_t(index));
  } else {
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += stride)
      body(i);
  }
}
```

`kLoop`이 거짓이면 스레드 인덱스가 32비트에 들어가고(최대 65535 * 256 스레드) 인스턴스에 루프가 없으며, 기존 본문과 같다. 참이면 카운터가 `int64_t`라서 32비트 `IdxT`를 wrap할 수 없다. 각 커널 안의 인덱스 산술은 커널 고유의 `IdxT`로 유지한다. 호스트는 launch 지점에서 `loop ? <..., true> : <..., false>`로 인스턴스를 고른다. 이 분리로 커널 4개의 인스턴스 수가 두 배가 된다.

### 3.3 채택하지 않은 안

둘 다 gfx1151에서 4M 원소 전치로 기존 커널과 비교해 측정했다.

- **모든 크기에 루프.** 1.3% 느림.
- **런타임 분기로 한 커널에 두 경로.** 1.2~1.7% 느림.

일반적인 복사는 상한 아래이므로, 템플릿 분리는 컴파일 시간과 코드 크기를 대가로 이들을 그대로 유지한다.

### 3.4 dynamic launch 크기

`copy_general_dynamic`은 이제 `shape`(복사 영역)의 곱을 구해 그 크기로 launch한다. 크기 0인 복사는 launch 전에 반환하며, 기존 코드는 이를 따로 처리하지 않았다. 2^32 이상뿐 아니라 모든 크기에서 wrap된 재쓰기가 사라진다.

### 3.5 copy_contiguous

`copy_contiguous`는 이미 제한과 루프가 있었다. 코드를 읽다가 별개의 위험을 발견했다. 루프 카운터가 `uint32_t`이고 패스마다 `stride * N_READS`(최대 2^26)를 더하므로, 2^32 바로 아래 크기에서는 카운터가 `size`에 닿기 전에 wrap될 수 있다. 이제 `UINT32_MAX` 대신 `INT32_MAX` 위에서 64비트 인덱스 인스턴스를 쓰고, 블록 수는 `copy_grid_blocks`로 계산한다. 코드를 읽어서 찾은 것이며 재현하지는 않았다.

## 4. 변경 고유의 위험

- 상한을 넘으면 커널이 이제 루프를 돈다. 16.7M 원소 이상의 복사가 해당한다. 33.5M 원소 사례는 잡음 범위 안에서 같게 측정됐다(6.2절).
- 커널 4개의 인스턴스가 두 배가 되어 컴파일 시간과 바이너리 크기가 늘어난다.
- `INT32_MAX`와 `UINT32_MAX` 사이 크기의 `copy_contiguous`는 이제 64비트 인덱스 산술을 쓰며 약간 더 비싸다. 벤치마크하지 않았다.
- `DynamicSliceUpdate`의 dynamic launch가 이전보다 작다. 추가 스레드에 의존하는 커널이 있으면 깨지지만, 추가 스레드는 wrap된 복사본만 다시 썼으므로 그런 커널은 없다.

## 5. 검증

### 5.1 테스트

`tests/rocm_strided_copy_grid.rs`는 입력을 모든 stride가 1인 작은 버퍼의 `as_strided` 뷰로 만들어, 원소 `[r, j, k, l]`이 `buf[r + j + k + l]`이 되게 한다. 값이 위치마다 다르고, 호스트가 입력을 만들지 않고도 임의의 행을 다시 계산할 수 있으며, `collapse_contiguous_dims`가 차원을 합칠 수 없다. 각 연산은 `copy_gpu_inplace`를 통해 커널을 고른다. 뷰의 `contiguous`는 `copy_g_byval`, `concatenate`는 `copy_gg_byval`, `slice_update_dynamic`은 `copy_gg_dynamic`(4차원 갱신) 또는 `copy_gg_dynamic_nd`(1차원으로 합쳐지는 갱신)다.

| 그룹 | 테스트 | 크기 | 결과 |
|---|---|---|---|
| 빠른 테스트 | `strided_input_copy_past_the_grid_cap_matches_host`, `concatenate_past_the_grid_cap_matches_host`, `dynamic_update_past_the_grid_cap_matches_host` | `[32, 8, 128]` f16 1300행, 42.6M 원소, 제한된 그리드 약 2.5패스, 모든 원소를 호스트 참조와 비교 | 3개 통과, 루프를 한 번만 돌게 줄이면 3개 모두 실패 |
| `#[ignore]` | `strided_input_copy_past_u32_elements_keeps_the_last_rows`, `concatenate_past_u32_elements_keeps_the_last_rows`, `dynamic_update_past_u32_elements_keeps_the_last_rows`, `dynamic_one_block_update_into_a_slab_past_u32_elements` | 131,073행, 2^32 + 32,768 f16 원소, 각 약 8 GiB, 하나씩 실행, 2^31과 2^32 양쪽과 끝의 행을 확인 | main에서 4개 모두 "invalid configuration argument"로 실패, 수정 후 4개 모두 통과 |

블록 하나 테스트는 1.2절의 paged 캐시 패턴, 즉 2^32 원소를 넘는 슬랩에 블록 하나를 쓰는 경우다. `#[ignore]` 실행은 모두 `scripts/rocm_gpu_guard.sh`를 거쳤다. `AMD_LOG_LEVEL=3`으로 각 연산이 의도한 커널에 도달함을 확인했다. 모든 테스트는 본문 내내 `streams::lock_default_device`를 잡고, 다른 백엔드에서는 건너뛴다.

### 5.2 게이트

- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`: 통과. 신규 테스트 타깃의 clippy: 통과.
- 오케스트레이터 게이트(`e5f4419c`, `2aa5211f`로 리베이스한 이 브랜치에서 `make verify-rocm`): OK, 스위트 163개, 12,194개 통과, 0개 실패, 403개 무시, 스모크 포함.

검증하지 못한 것: 이 호스트에는 Metal과 CUDA가 없다. 변경은 `patches-rocm/`와 `feature = "rocm"`으로 제한된 테스트만 건드리므로 해당 경로는 이 변경으로 빌드되지 않는다.

## 6. 결과

### 6.1 환경

Ryzen AI MAX+ 395와 Radeon 8060S(`gfx1151`). 모든 GPU 실행은 `scripts/rocm_gpu_guard.sh`를 거쳤다.

### 6.2 마이크로벤치마크

전치된 뷰의 f16 `contiguous`를 eval당 측정, 변경 전(main)과 후, 번갈아 6라운드, 라운드별 중앙값의 중앙값.

| 케이스 | 변경 전 | 변경 후 |
|---|---|---|
| `[1, 8, 4096, 128]` | 147.82 us | 147.99 us |
| `[1, 2048, 32, 128]` | 388.85 us | 387.40 us |
| `[1, 32, 2048, 128]` 뷰 둘의 `concatenate` | 836.75 us | 833.55 us |

차이는 +0.1%, -0.4%, -0.4%다. `[1, 1, 32, 128]`(12~14 us, launch 병목)과 `[1, 8192, 32, 128]`(33.5M 원소, 상한 초과, 최선 실행 변경 전 5455 us, 후 5423 us)는 빌드 간보다 라운드 간 편차가 커서 주장하지 않는다.

## 7. LOCAL_FIXES 항목 42

항목 42는 실패한 launch와 그 한계, dynamic 크기 버그, 헬퍼와 `kLoop` 분리, 채택하지 않은 안과 그 비용, `copy_contiguous` 인덱스 변경(읽어서 찾았다고 표시), 테스트와 마이크로벤치 수치, 업스트림 CUDA 관찰을 기록한다. 포크 정책 문구로 끝난다: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there (lablup/mlxcel#2184)."

## 8. 기술적 선택과 그 이유

- **런타임 분기 대신 템플릿 분리.** 일반적인 복사는 상한 아래의 핫 경로이므로, 인스턴스가 두 배가 되더라도 기존 인스턴스를 유지한다.
- **64비트 카운터는 루프에서만.** 루프 없는 본문은 상한 아래에서 정확한 32비트 스레드 인덱스를 유지한다.
- **인덱스 산술은 `IdxT` 유지.** 커널은 이미 배열의 데이터 크기로 `IdxT`를 골랐고, 넓어지는 것은 루프 카운터뿐이다.
- **dynamic launch는 복사 shape로.** 커널이 분해하는 원소 수이므로 올바른 상한이다.
- **`copy_contiguous`의 상한 재사용.** 65535블록은 이미 관례였고 이제 헬퍼 하나로 공유한다.

## 9. 남은 위험

- **다른 launcher에도 같은 버그가 있다**(11절).
- **`copy_contiguous`의 wrap은 재현하지 못했다.** 변경은 카운터 산술을 읽은 결과에 근거한다.
- **상한 초과 성능은 대략만 측정했다.** 33.5M 원소 사례 하나는 실행 간 편차 안에 있다.
- **Metal과 CUDA 미검증.** `81ba1c6a`의 CUDA `copy_general_dynamic`도 같은 `out.size()` launch다.

## 10. 배운 점

- **launcher 하나의 상한이 이웃을 지켜 주지 않는다.** `copy_contiguous`는 루프를 돌았지만 이웃 셋은 그러지 않았고, 차이는 2^32 원소에서야 드러났다.
- **launch 크기는 버퍼가 아니라 작업량에서 나와야 한다.** 제자리 연산에서 `out`은 목적지이므로 `out.size()`는 복사한 원소 수가 아니다.
- **빠른 경로는 별도 인스턴스로 둔다.** 단일 커널 대안 둘 다 고치려는 대상이 아닌 케이스에서 1.2~1.7%의 비용이 들었다.
- **모든 폭 경계의 양쪽에서 인덱싱을 테스트한다.** 큰 테스트는 마지막 행뿐 아니라 2^31과 2^32 주변의 행도 확인한다.

## 11. 후속 작업

#2234가 제한 없는 그리드에 grid-stride 루프가 없는 나머지 ROCm launcher를 추적한다(경로는 `patches-rocm/mlx/backend/rocm/` 아래).

- `binary.hip`(`binary_g`)
- `unary.hip`, `ternary.hip`(`unary_g`, `ternary_g`: y 그리드가 제한 없음)
- `reduce/init_reduce.hip`(`init_reduce_kernel`), `reduce/col_reduce.hip`(`col_reduce_small`)
- `arange.hip`(`arange_kernel`)
- `indexing.hip`(`gather_general_kernel`, `scatter_general_kernel`, `gather_axis_kernel`, `scatter_axis_kernel`)
- `quantized/convert_fp8.hip`, `quantized/affine_quantize.hip`, `quantized/fp_quantize.hip`
- `sort.hip`(`iota_kernel`, `int` 인덱스)
- `copy/copy_general_input.hip`의 `copy_col_row`(제한 없는 2차원 타일 그리드, `int` 행과 열 산술)

`unary.hip:164`, `binary.hip:237`, `ternary.hip:132`는 이미 제한되어 있어 범위 밖이다. 이슈는 launcher 계열마다 PR 하나를 제안하며, 각각 65535블록을 넘기는 작은 크기 테스트와 2^32 원소를 넘기는 `#[ignore]` 테스트를 갖는다.
