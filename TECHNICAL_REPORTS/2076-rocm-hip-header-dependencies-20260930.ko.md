# 기술 보고서: PR #2076 - include한 헤더가 바뀌면 ROCm HIP object를 다시 빌드

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: CMake (ROCm overlay), Markdown

**위험도**: 낮음 (빌드 glue만 바뀌며 커널이나 런타임 코드는 바뀌지 않습니다. 변경 후 첫 빌드는 모든 HIP object를 한 번 다시 컴파일합니다. Metal과 CUDA 빌드는 `patches-rocm/`을 복사하지 않습니다)

## 요약

이슈 #2075(에픽 #1801의 일부)는 ROCm overlay가 각 `.hip` 파일을 별도의 `add_custom_command`로 컴파일하면서, 그 의존성으로 `.hip` 파일 자신만 등록한다는 점을 발견했습니다. CMake는 object가 어떤 헤더를 include하는지 알 수 없었고, 그래서 헤더만 바뀐 warm 빌드에서는 `hip_objs/*.o`가 하나도 다시 컴파일되지 않았습니다. fresh 빌드는 올바랐습니다. 재사용된 빌드 트리는 그렇지 않았고, 이 에픽에서는 바로 그 재사용된 빌드 트리가 머지 게이트입니다. hosted CI가 멈춘 동안에는 오래 쓰는 worktree에서 돌리는 `make verify-rocm`이 PR이 머지 전에 거치는 유일한 검사였습니다. 따라서 헤더만 고친 overlay 수정이 이전 헤더로 컴파일된 커널을 상대로 "검증"될 수 있었습니다.

이 PR은 `hipcc`가 object마다 depfile을 쓰게 하고(`-MD -MF <obj>.d`) 이를 `DEPFILE`로 CMake에 넘겨, 빌드 도구가 object가 직접 또는 간접으로 읽은 모든 헤더를 추적하게 합니다. gfx1151에서 이제 `kernel_utils.hpp`를 고치면 이를 include하는 39개 object가 다시 컴파일되고(이전에는 41개 중 0개), `reduce/reduce.hpp`를 고치면 정확히 그 7개 includer만 다시 컴파일되며, 변경 없는 재빌드는 아무것도 다시 컴파일하지 않습니다. 변경 내용은 `LOCAL_FIXES.md` 항목 26과 `docs/installation.md`에 기록되어 있습니다.

## 1. 에픽 #1801에 왜 중요한가

이 결함은 clean 빌드에서 잘못된 커널을 만들지는 않았습니다. 망가뜨린 것은 에픽이 의존하는 로컬 게이트의 신뢰성이었습니다. 이 에픽에서 세 번 부딪혔습니다.

- **#2059 (`kernel_utils.hpp`).** PR #2059(d1128266)는 divisor를 받는 `get_2d_grid_dims`를 고쳤습니다(`LOCAL_FIXES.md` 항목 22). 48개와 64개 head를 가진 Mamba2 layer의 `strided_scan`에서 `HSA_STATUS_ERROR_MEMORY_FAULT`를 낸 grid 크기 버그입니다. overlay 파일 중에서는 `kernel_utils.hpp`만 바뀌었습니다. 그 커밋 이전부터 `target/`을 가진 worktree에서는 `scan.hip`이 다시 컴파일되지 않았으므로, 테스트는 수정이 머지된 상태라고 보고하면서도 트리는 이전 grid 코드를 계속 실행했습니다.
- **#2051 (`lru_cache.h`).** `MLX_ROCM_FFT_CACHE_SIZE` 수정(PR #2073으로 머지, 항목 25)은 `lru_cache.h`에 있습니다. 첫 실행은 여전히 이전 코드처럼 동작했습니다. `fft.hip`도 함께 수정했기 때문에(파싱 규칙을 적은 주석) 그 object 하나가 자기 의존성으로 stale 판정을 받아서야 수정이 반영되었습니다. 항목 25는 원래 이를 명시하고 있었고, 이 PR은 그 문장을 항목 26을 가리키도록 고칩니다.
- **#2072 (bisect).** 간헐적인 `rocm_mxfp4_quant` 실패를 warm 트리에서 bisect한 결과는, 헤더만 다른 커밋 사이에서는 믿을 수 없었습니다. 그 사이를 오가도 HIP object가 하나도 다시 빌드되지 않았기 때문입니다.

공통점은, 재사용된 트리에서 도는 모든 머지 게이트가 stale 커널을 테스트할 수 있었고 빌드 출력 어디에도 그런 표시가 없었다는 것입니다. 이 PR 이후에는 헤더만 바꾼 overlay 변경도 다른 변경과 똑같이 HIP object에 반영되고, 이후 warm 트리에서 하는 bisect도 각 커밋의 실제 내용을 컴파일합니다.

## 2. 문제 정의

`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/CMakeLists.txt`는 HIP 소스를 CMake의 HIP 언어 지원 밖에서 컴파일합니다. `foreach(hip_src ${HIP_SOURCES})` 루프가 파일마다 `hipcc -c`를 실행하는 `add_custom_command`를 하나씩 만듭니다. custom command는 CMake에게 불투명합니다. CMake가 아는 의존성은 `DEPENDS`에 적힌 것뿐이었고, 그것은 `${hip_src}`였습니다. `target_sources(mlx ...)`로 추가된 `.cpp` 소스는 CMake가 직접 include를 스캔하므로 영향을 받지 않았습니다.

object는 `target/<profile>/build/mlxcel-core-*/out/build/_deps/mlx-build/mlx/backend/rocm/hip_objs/` 아래에 있습니다. `kernel_utils.hpp`, `device.h`, `lru_cache.h`, `quantized/*.hpp` 같은 헤더는 여러 `.hip` 파일이 공유하므로(`kernel_utils.hpp`만 39개가 include), 헤더 수정은 가장 많은 커널에 영향을 주면서도 가장 적은 재빌드를 일으키는 종류의 변경이었습니다.

## 3. 변경 요약

- **`patches-rocm/mlx/backend/rocm/CMakeLists.txt`**: `if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)` 아래에서 각 command가 `hipcc` 명령줄에 `-MD -MF ${hip_obj}.d`를 더하고 `DEPFILE ${hip_obj}.d`를 넘기며, 정책 `CMP0116`이 있으면 `NEW`로 설정합니다. 3.20 미만에서는 대신 모든 object가 `mlx/` 아래 모든 `*.h`, `*.hpp`, `*.cuh`의 `file(GLOB_RECURSE ... CONFIGURE_DEPENDS)` 목록에 의존합니다. 주석 블록이 동작 방식과 fallback을 설명합니다.
- **`patches-rocm/LOCAL_FIXES.md`**: 새 절 "Fixes to the fork's build"와 항목 26(동작 방식, 의존성 체인, 측정값, 한계, #1813 upstream 후보). 항목 25의 소스별 의존성 문장은 과거형으로 바뀌고 항목 26을 가리킵니다.
- **`docs/installation.md`** (ROCm, Build): incremental 빌드가 헤더를 추적한다는 점, overlay 헤더를 `touch`만 하면 아무것도 다시 빌드되지 않는다는 점, ROCm과 시스템 헤더는 추적되지만 device 쪽에서만 include되는 헤더와 컴파일러 자체는 추적되지 않는다는 점, 그리고 HIP를 강제로 clean 재빌드하는 방법(해당 profile의 `hip_objs/`를 지우고 overlay 파일을 touch해 Cargo가 build script를 다시 돌리게 하거나 `cargo clean -p mlxcel-core`).

커밋: `9d17bb1e`는 수정, 문서, 항목 26이고, `b8d8b437`은 fallback glob을 backend 디렉터리에서 `mlx/` 전체로 넓히고 depfile의 한계를 명시한 리뷰 후속 커밋입니다.

## 4. 기술적 선택과 그 이유

### 헤더 glob이 아니라 컴파일러가 만든 depfile

이슈는 두 방안의 우선순위를 정했습니다. `DEPFILE`이 먼저이고, 헤더 glob은 `DEPFILE`을 쓸 수 없을 때만입니다. glob은 모든 object를 모든 헤더에 의존하게 만들므로, `reduce/reduce.hpp` 한 번 수정에 7개가 아니라 41개 object가 모두 다시 컴파일됩니다. 목록을 관리하지 않고도 간접 include와 backend 디렉터리 밖의 헤더까지 따라가는 방법은 depfile뿐입니다. glob은 MLX 헤더에 대해 stale해지지는 않고 느릴 뿐이므로 fallback으로 남겨 두었습니다.

### 3.20과 `CMP0116`인 이유

`build.rs`는 generator를 지정하지 않고 `cmake::Config`로 빌드하므로, `CMAKE_GENERATOR`를 설정하지 않으면 Unix Makefiles를 씁니다. `add_custom_command`의 `DEPFILE`은 Makefile generator에서 CMake 3.20부터 동작합니다. `CMP0116` NEW는 Ninja가 상대 depfile 경로를 Makefile generator처럼 binary 디렉터리 기준으로 해석하게 합니다. overlay의 최상위 `CMakeLists.txt`가 3.25를 요구하므로 `CMP0116` NEW는 이미 적용되어 있고 fallback 분기는 현재 도달할 수 없습니다. 명시적인 guard와 정책 설정은 그 최소 버전이 낮아지더라도 파일이 올바르게 동작하도록 남겨 둔 것입니다.

### fallback glob은 `mlx/` 전체를 덮는다

이슈는 rocm backend 디렉터리를 glob하자고 제안했습니다. 리뷰 후속 커밋은 이를 `mlx/` 아래 모든 헤더로 넓혔습니다. HIP 소스가 backend 밖의 MLX 헤더(예: `mlx/primitives.h`)도 include하기 때문입니다. backend만 glob했다면 바로 그 헤더들에 대해 stale했을 것입니다.

### overlay 원본이 아니라 빌드 트리의 복사본을 추적

overlay는 `src/lib/mlx-cpp/CMakeLists.txt`의 `configure_file(... COPYONLY)`를 통해 빌드 트리에 들어가고, `hipcc`는 `_deps/mlx-src` 아래의 복사본을 include합니다(`.hip` 파일 자신의 디렉터리와 `-I${PROJECT_SOURCE_DIR}`). 그래서 depfile은 복사본을 절대 경로로 가리키며, 41개 depfile 중 `patches-rocm`을 언급하는 것은 0개입니다. 전체 체인은 다음과 같습니다.

1. overlay를 수정하면 `patches-rocm/` 아래 파일이 바뀌고, `build.rs`가 `rerun-if-changed`로 이를 감시하므로 Cargo가 build script를 다시 실행합니다.
2. CMake가 재구성되고, `configure_file(COPYONLY)`는 내용이 다를 때만 복사본을 다시 씁니다.
3. make가 depfile을 통해 복사본의 mtime과 object의 mtime을 비교합니다.

컴파일러가 읽는 것이 복사본이므로 이것이 올바른 추적 대상입니다. PR은 depfile이 원본을 가리키게 하려고 하지 않습니다.

## 5. 측정한 재빌드

gfx1151, CMake 3.31.6, Unix Makefiles, warm `test-fast` 트리. 각 행은 `cargo test --profile test-fast --features rocm --test rocm_strided_scan --no-run`으로 다시 빌드하고 marker 파일보다 새로운 `hip_objs/*.o`를 셌습니다.

| 변경 | 이전 | 이후 |
|---|---|---|
| 변경 없음 | 0 | 0 |
| `kernel_utils.hpp`에 주석 추가 | 41개 중 0개 (빌드 트리 복사본은 다시 쓰였음) | 39개: depfile이 이 헤더를 가리키는 모든 object (`event.o`와 `quantized/qmv_tiled_kernel.o` 제외) |
| 그 수정 되돌리기 | 0 | 같은 39개 |
| `reduce/reduce.hpp`에 주석 추가 후 되돌리기 | 실행 안 함 | 매번 7개: `reduce.o`, `reduce/{all,col,init,row}_reduce.o`, `layer_norm.o`, `rms_norm.o` |

두 includer 집합은 빌드 트리 소스의 `#include` 그래프를 depfile과 독립적으로 따라간 결과와도 일치하므로, depfile이 스스로와만 일관된 것이 아닙니다.

이 수치에 따르는 주의 사항:

- **`touch`만으로는 아무 일도 일어나지 않습니다.** `configure_file(COPYONLY)`는 내용이 바뀐 복사본만 다시 쓰므로, overlay 헤더를 touch해도 복사본의 mtime은 그대로이고 object는 하나도 다시 빌드되지 않습니다. 재현하려면 내용을 바꿔야 합니다. 의도된 동작이며 문서에 적혀 있습니다.
- **device 쪽에서만 include되는 헤더는 추적되지 않습니다.** depfile은 host 컴파일에서 나옵니다. ROCm과 시스템 헤더는 포함하므로(ROCm을 업데이트하면 모든 HIP object가 다시 빌드됨) device 전용 include는 포함하지 않습니다. 현재 해당하는 것은 `flash_attention_wmma.hip`의 `rocwmma/rocwmma.hpp`이며, `quantized/qmm.hip`은 host 쪽에서도 include하므로 그 depfile에는 나타납니다. 컴파일러 바이너리 자체도 추적되지 않습니다. `hipcc`나 rocWMMA를 업그레이드한 뒤에는 `docs/installation.md`에 적힌 대로 HIP clean 재빌드를 강제해야 합니다.
- **이 변경을 받은 뒤 첫 빌드는 41개 object를 모두 한 번 다시 컴파일합니다.** `hipcc` 명령줄이 바뀌었으므로 모든 object가 한 번은 out of date가 됩니다. 이것이 기존 warm 트리를, 아직 #2059 이전의 `scan.hip` object를 가진 트리까지 포함해 스스로 고쳐 주지만, 재사용된 worktree마다 첫 게이트 실행에서 HIP 전체 컴파일 비용이 듭니다.

## 6. 검증

PR 작성자 (gfx1151):

- 위의 재빌드 표와 `#include` 그래프 교차 확인.
- 전체 mlxcel 빌드에는 Ninja를 쓸 수 없었습니다. 같은 패턴(`configure_file` 복사본, depfile, `CMP0116` NEW)을 두 파일짜리 probe 프로젝트에서 Ninja 1.13과 Unix Makefiles로 확인했습니다. 수정하면 includer만 다시 컴파일되고, touch나 변경 없음은 아무것도 다시 빌드하지 않으며, depfile은 빌드 트리 복사본을 가리킵니다.
- 문서에 적은 강제 clean 재빌드: touch 없이 `hip_objs/`만 지우면 아무것도 다시 빌드되지 않았고(Cargo가 build script를 건너뜀), overlay `CMakeLists.txt`를 touch한 뒤에는 41개 object와 41개 depfile이 다시 만들어졌습니다. 그 fresh HIP 빌드에서 `rocm_strided_scan`(3개 통과)과 `rocm_slice_update_reduce`(8개 통과)를 둘 다 `--test-threads=1`로 실행해, 헤더만 바꾼 #2059의 `get_2d_grid_dims` 수정이 유지됨을 확인했습니다.
- 리뷰 후속 커밋(주석과 fallback glob만 변경) 뒤의 재빌드는 재구성만 하고 HIP object는 하나도 다시 컴파일하지 않았습니다.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`와 `cargo test --test dead_doc_pointers`가 통과했습니다.

오케스트레이터 검증 (gfx1151):

- base `0d17303d` 위의 브랜치에서 `make verify-rocm`이 모든 단계를 실행했습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` workspace clippy, ROCm smoke 실행(32 토큰)이 통과했습니다.
- `verify-test-rocm`은 네 target에서 실패했으며 모두 이 PR과 무관합니다. 알려진 baseline 실패 세 개(`layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, #2037의 `gelu_approx_matches_mlx_nn_bit_for_bit`와 `tests::family_order_is_exhaustive`)와, #2072에서 추적 중인 기존의 간헐적 실패 `rocm_mxfp4_quant::quantized_matmul_matches_dequantized_reference`입니다.
- 브랜치를 `a6e07542` 위로 rebase한 뒤, 오케스트레이터는 `LOCAL_FIXES.md` 충돌을 #2070의 항목 24, 이어서 이 PR이 수정한 항목 25와 새 항목 26 순서로 유지해 해결했습니다. 그 뒤 fast script 게이트와 `rocm_slice_update_source`, `rocm_strided_scan` 테스트를 다시 실행했습니다.

## 7. 학습 포인트

- **custom command는 알려 준 것만 압니다.** `add_custom_command`로 컴파일하면 컴파일러 명령줄을 통제하는 대신 CMake의 내장 include 스캔을 잃습니다. 이렇게 컴파일하는 빌드는 `DEPFILE`(또는 명시적인 헤더 목록)이 있어야 하며, 없으면 헤더 수정이 조용히 전파되지 않습니다.
- **머지 게이트는 그 빌드의 staleness만큼만 믿을 수 있습니다.** 이 게이트는 검증 대상 변경을 컴파일하지 않은 재사용 트리에서 초록불을 냈습니다. CI를 오래 쓰는 트리의 로컬 실행으로 대신할 때는 incremental 빌드의 정확성이 편의 기능이 아니라 검증의 일부가 됩니다.
- **include하는 쪽 파일을 고친 것이 버그를 가렸습니다.** #2051의 수정은 `fft.hip`도 함께 고쳤기 때문에 동작했습니다. "호출하는 쪽을 건드린 뒤에야 동작하는" 수정은 코드만이 아니라 빌드 그래프를 확인하라는 신호입니다.
- **내용이 다를 때만 복사하는 계층은 "바뀌었다"의 의미를 바꿉니다.** 소스와 컴파일러 사이에 `configure_file(COPYONLY)`가 있으면 mtime 기반 추적은 내용 변경만 봅니다. 재현과 강제 재빌드는 이를 고려해야 하며, 문서의 clean 재빌드가 `hip_objs/`를 지우고 overlay 파일을 touch해 Cargo가 build script를 다시 돌리게 하는 이유가 이것입니다.

## 8. 주의 사항과 검증하지 않은 것

- **전체 빌드에서의 Ninja**는 실행하지 않았고, probe 프로젝트만 확인했습니다.
- **Metal과 CUDA**는 실행하지 않았습니다(이 호스트에 없음). 변경은 ROCm overlay에 한정되며 그 빌드들은 overlay를 복사하지 않습니다.
- **device 전용 include와 컴파일러 업그레이드**는 여전히 추적되지 않으며, 문서의 clean 재빌드가 그 대응책입니다.
- **fallback 분기**는 3.25 최소 버전에서는 도달할 수 없고, 더 오래된 CMake에서 실행해 보지 않았습니다.
- **rebase 후 재실행**: 오케스트레이터가 `a6e07542` 위로 rebase한 뒤 fast script 게이트와 두 ROCm 테스트를 다시 실행했습니다. 이 보고서는 그 결과를 다시 적지 않습니다.

## 9. 남은 작업

- #1813: 다른 upstream 후보와 함께 항목 26을 fork에 제안.
- #2072: 간헐적인 `rocm_mxfp4_quant` 실패. 이제 헤더만 다른 커밋 사이의 bisect도 warm 트리에서 다시 할 수 있습니다.
- baseline `verify-test-rocm` 실패 세 개(`prefill_dense_gemm_matches_qmm_bytes_where_eligible`, #2037의 두 개)는 이 PR 밖에서 추적합니다.

참조: #2075 (이 PR로 닫힘), #1801, #2059, #2051, PR #2073, #2072, #2070, #1813, #2037.
