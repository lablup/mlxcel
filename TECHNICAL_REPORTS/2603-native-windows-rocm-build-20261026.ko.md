# PR #2603: Windows에서 ROCm으로 mlxcel 네이티브 빌드 및 실행

**날짜**: 2026-10-26
**상태**: Windows에서 완료. Linux ROCm 빌드는 검증되지 않음(CI ROCm 잡 중단 상태)
**언어**: Rust, C++, HIP, CMake, Markdown
**위험도**: 중간

## 요약

PR #2603(이슈 #2601)으로 Windows x86_64의 MSVC 환경에서 ROCm 10 SDK를 써서 `cargo build --release --features rocm`을 빌드할 수 있고, `mlxcel generate`가 GPU에서 올바른 텍스트를 생성합니다. AMD Ryzen AI MAX+ 395(Radeon 8060S, `gfx1151`)와 ROCm 10.1.0 TheRock tarball로 검증했습니다. greedy 디코딩에서 `mlx-community/Qwen3-0.6B-4bit`는 약 113 tok/s, `mlx-community/Qwen3-1.7B-bf16`은 약 28.7 tok/s로 올바른 답을 냈습니다.

빌드 쪽 작업은 #2599의 Windows CUDA 경로를 거의 그대로 따랐습니다. 새로 다뤄야 했던 부분은 메모리 모델입니다. Windows는 APU라도 device 메모리를 CPU 주소 공간에 매핑하지 않습니다. 그래서 "integrated GPU의 할당은 CPU에서 바로 접근할 수 있다"는 백엔드의 Linux 가정이 성립하지 않습니다. 이 PR은 Windows의 모든 CPU 접근을 백엔드에 이미 있던 host shadow 경로로 보내고, 그 과정에서 Linux discrete GPU에도 해당하는 기존 결함 두 개를 고쳤습니다.

실행할 때는 아직 ROCm `bin` 디렉터리가 `PATH`에 있어야 하고 `ROCM_PATH`(또는 `HIP_PATH`)가 설정되어 있어야 합니다. 이 두 조건은 #2602에서 없앱니다.

## 1. 문제 정의

`rocm` feature(#1801/#1802)는 Linux 전용이었습니다. 다른 타깃에서는 빌드 스크립트가 panic했습니다. 벤더링한 ROCm 백엔드 CMake는 GCC를 가정했고(libstdc++ include 탐색, `-fPIC`, `ar rcs libmlx_rocm_kernels.a`), 백엔드 소스 몇 곳은 POSIX 호출을 썼습니다. MSVC 툴체인 자체는 #2599에서 CUDA로 이미 검증됐습니다. 그래서 남은 질문은 AMD의 Windows 스택도 같은 틀에 맞는지, 그리고 빌드된 뒤 실행 단계에서 무엇이 깨지는지였습니다.

툴체인 문제는 코드를 고치기 전에 두 가지 사실로 정리됐습니다. 첫째, ROCm 10.1의 clang은 기본 타깃이 `x86_64-pc-windows-msvc`입니다. 둘째, ROCm 라이브러리(`rocblas.dll`, `hipblas.dll`, `libhipblaslt.dll`)는 MSVCP140, VCRUNTIME140, UCRT를 import하고 MinGW 런타임은 쓰지 않습니다. GNU식 이름을 가진 `libhipblaslt.dll.a`도 실제로는 COFF import library입니다. 따라서 빌드 전체를 MSVC ABI로 유지할 수 있습니다. MLX host C++은 `cl`로, HIP 커널은 ROCm clang으로 빌드하고, 런타임은 전부 `/MD`로 맞춥니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| Feature gate | Linux와 Windows x86_64 MSVC에서 `rocm` 허용. `cuda`, `metal`과는 여전히 동시에 켤 수 없음 |
| SDK 레이아웃 | 새 `build_support/windows_rocm.rs`: `ROCM_PATH` 다음 `HIP_PATH`로 루트 결정(빈 값은 unset), `RocmLayout`, `parse_amdgpu_arch`. `windows_cuda::cmake_path`를 재사용하므로 `windows_cuda`를 `any(cuda, rocm)`에서 컴파일. 테스트는 `mlxcel-mlx-pin`에 있음 |
| CMake | Ninja generator, 슬래시 경로의 `ROCM_PATH`/`CMAKE_PREFIX_PATH`/`CMAKE_HIP_COMPILER`, 모든 MSVC 빌드에서 MLX C++에 `/EHsc` |
| 링크 | `mlx_rocm_kernels.lib`, ROCm import library, hipBLASLt의 `.dll.a`를 `OUT_DIR/hipblaslt.lib`로 복사. rpath 없음 |
| Overlay CMake | WIN32 분기: libstdc++ 탐색 제외, `-fPIC` 제외, `--rocm-path`/`--rocm-device-lib-path`, `-fms-runtime-lib=dll`, `.obj`, `lib.exe` 아카이브. Linux 분기는 바이트 단위로 동일 |
| Overlay 소스 | `sysconf`를 `GlobalMemoryStatusEx`로 대체, weak symbol, `_dup2`/`NUL`, `<iterator>`, `__builtin_clz` shim, LLP64에서도 맞는 hiprtc `size_t`, `rocm_home()`의 `HIP_PATH` fallback과 빈 값 처리 |
| 메모리 모델 | CPU에서 직접 접근할 수 없는 메모리에 host shadow 사용, `raw_ptr()`이 dirty 플래그 설정, shadow 수명을 버퍼에 묶음 |
| 중복 심볼 | `binary_two.hip` 제거, `DivMod::eval_gpu`는 `binary.hip`에 남김 |
| 문서 | `docs/installation.md`의 "Windows with AMD ROCm (experimental)", 환경변수 항목, feature 주석 |

## 3. 기술적 선택과 그 이유

### 3.1 host-visible 메모리를 할당하지 않고 device 메모리에 shadow를 붙임

처음 링크까지 성공한 빌드는 `mlx::core::array::init`의 `memmove`에서 access violation으로 죽었습니다. allocator가 fine-grained device 메모리를 돌려줬고 CPU가 거기에 바로 썼기 때문입니다. Linux에서는 KFD가 APU의 이 메모리를 프로세스에 매핑해 주지만 Windows는 그렇지 않습니다. 이 머신에서 할당 방식별로 측정한 결과입니다.

| 할당 | CPU 쓰기 | GPU 읽기 | 최대 단일 할당 |
|---|---|---|---|
| `hipMalloc`, fine-grained, uncached | fault | 약 233 GB/s | 92 GiB (`hipMalloc`) |
| `hipHostMalloc` | 정상 | 약 233 GB/s | 약 28 GiB |
| `hipMallocManaged` | 정상 | OS RAM 이내에서 약 233 GB/s | OS RAM을 넘으면 CPU 2.9 GB/s, GPU 읽기가 6분 안에 끝나지 않음 |

이 머신은 BIOS UMA로 약 96GB를 GPU에 떼어 두었고, 이 영역은 device 메모리입니다. OS에는 약 32GB만 보입니다. host-visible 할당을 쓰면 모델 크기가 OS에 보이는 RAM으로 제한됩니다. 그래서 `hipMalloc`을 유지하고, 백엔드에 discrete GPU용으로 이미 있던 shadow 경로를 쓰기로 했습니다. `raw_ptr()`은 pinned host 사본을 돌려주고, `gpu_ptr`은 커널이 버퍼를 읽기 전에 그 사본을 device로 되돌립니다. 경로는 `device_memory_host_visible()`(constexpr이고 `_WIN32`에서만 false)로 고르므로, Linux에서는 컴파일 결과가 이전과 같습니다.

### 3.2 shadow를 내줄 때마다 dirty로 표시

shadow 경로는 실제로 쓰인 적이 없었습니다. 백엔드 어디에서도 `host_dirty = true`를 설정하지 않아서, `raw_ptr()`로 CPU가 쓴 내용이 device로 돌아가지 않았습니다. Windows뿐 아니라 Linux discrete GPU에서 async pool을 켰을 때도 마찬가지입니다. MLX의 `Buffer::raw_ptr()`은 호출자가 읽을지 쓸지 알려주지 않으므로, 이 PR은 shadow를 내줄 때마다 dirty로 표시합니다. 이 방식은 정확하지만 보수적입니다. CPU가 읽기만 한 뒤 커널이 실행되어도 버퍼 전체를 device로 한 번 복사합니다. 읽기와 쓰기를 구분하는 최적화는 후속 과제입니다.

구현 리뷰에서는 shadow가 원래 버퍼보다 오래 살아남는 문제가 추가로 나왔습니다. scalar pool로 돌아간 슬롯이 pinned shadow를 누수했고, 버퍼 캐시를 재사용할 때 이전 소유자의 dirty shadow가 새 소유자에게 넘어갔으며, shadow를 해제한 뒤에도 dirty 플래그가 남았습니다. 그래서 shadow를 버퍼와 함께 해제하고, 해제와 재사용 시점에 플래그를 지우도록 고쳤습니다.

### 3.3 hipBLASLt import library를 link argument가 아니라 복사로 연결

의존 크레이트 build script의 `cargo:rustc-link-arg`는 최종 바이너리까지 전달되지 않고, `cargo:rustc-link-lib`는 전달됩니다. 그래서 build script가 `libhipblaslt.dll.a`를 `OUT_DIR/hipblaslt.lib`로 복사하고 이름으로 링크합니다. 대가로, 같은 경로에서 SDK를 덮어써 업그레이드했는데 build script가 다시 실행되지 않으면 이전 사본이 남습니다.

### 3.4 중복된 `DivMod::eval_gpu`는 Linux가 쓰던 정의를 남김

`binary.hip`(FloorDivide와 Remainder로 두 번 계산)과 `binary_two.hip`(fused 커널 하나)이 둘 다 이 함수를 정의하고 있었습니다. GNU ld는 심볼을 정의한 첫 번째 아카이브 멤버를 가져옵니다. 아카이브 순서는 `HIP_SOURCES`를 따르고, 거기서 `binary.hip`이 먼저 나옵니다. 만약 두 오브젝트가 모두 링크됐다면 Linux에서 multiple-definition 오류가 났을 것입니다. MSVC `lib.exe`는 LNK4006 경고를 내고 첫 번째 정의를 유지했습니다. `binary_two.hip`에서 템플릿이 아닌 정의는 `DivMod::eval_gpu` 하나뿐이었고, 다른 곳에서 그 템플릿을 참조하지 않으므로 파일을 제거했습니다. 이 판단은 코드 분석에 근거한 것이고, Linux 링크로 추적하지는 못했습니다.

## 4. 작업 중 발견한 기존 결함

- **MSVC에서 MLX에 `/EHsc` 누락.** cmake-rs가 cc 크레이트의 플래그를 `CMAKE_CXX_FLAGS`로 넘기면서 CMake의 MSVC 기본값을 덮어썼습니다. 그 결과 MLX가 unwind 없이 컴파일됐고(C4530), 예외가 전파될 때 소멸자가 실행되지 않았습니다. #2599의 Windows CUDA 빌드에도 해당합니다.
- **hiprtc 프리앰블의 LP64 가정.** `using size_t = unsigned long;`은 Windows에서 32비트라 hiprtc 자체의 typedef와 충돌하고, 모든 fused 커널 JIT를 실패시켰습니다. `decltype(sizeof(0))`는 두 ABI 모두에서 컴파일러의 정의와 일치합니다.
- **빈 환경변수 값.** `rocm_home()`이 빈 `ROCM_PATH`를 그대로 받아 `-I/include`를 만들었습니다. Windows에서는 현재 드라이브의 루트이고, 일반 사용자도 거기에 헤더를 심어 JIT 커널에 포함시킬 수 있었습니다. 이제 빈 값은 build script와 마찬가지로 unset으로 처리합니다.
- **재배치 가능한 SDK의 device bitcode.** `hipcc.exe`는 tarball SDK의 device library를 스스로 찾지 못합니다. `--rocm-path`와 `--rocm-device-lib-path`를 넘기기 전까지 모든 `.hip`이 실패했습니다.

## 5. 검증

Windows 11, ROCm 10.1.0, VS 18 환경에서 확인한 결과입니다. 전체 `cargo build --release --features rocm`으로 두 바이너리가 빌드됐고 로그에 C4530이나 LNK4006은 없었습니다. `amdgpu-arch` 감지 결과는 `gfx1151`이었습니다. 최종 head에서 두 모델 모두 올바른 텍스트를 생성했습니다(약 113, 28.7 tok/s). `ROCM_HOME`과 `ROCM_PATH`를 비우고 `HIP_PATH`만 설정한 JIT 실행도 성공했습니다. `cargo test -p mlxcel-mlx-pin --lib`(44개, 신규 11개), fmt, `--features rocm`으로 `mlxcel-core` clippy, overlay `drift`/`records` 검사를 모두 통과했습니다.

## 6. 알려진 한계

- **Linux ROCm 빌드 미검증.** CI ROCm 잡이 중단 상태입니다(`rocm-gfx1151` runner 없음, #1811). 두 리뷰에서 Linux에 닿는 변경이 이전과 같거나 의도한 수정(dirty 플래그, shadow 수명, DivMod)뿐임을 확인했지만, Linux ROCm 링크를 실제로 컴파일하지는 못했습니다.
- **`/EHsc`를 넣은 Windows CUDA 빌드**는 CUDA 호스트에서 확인해야 합니다.
- **실행 환경**: ROCm `bin`을 `PATH`에 두고 `ROCM_PATH`/`HIP_PATH`를 설정해야 합니다. 또 바이너리가 ROCm DLL을 이름으로 import하므로, 작업 디렉터리에 심은 DLL이 먼저 로드될 수 있습니다. #2602에서 기록된 디렉터리 기반의 delay-load와 기록된 include 경로를 추가합니다.
- **Shadow 비용과 메모리 집계**: shadow는 `active_memory_`에 집계되지 않는 pinned 메모리이고, OS에 보이는 RAM 근처에서 한계에 걸립니다. CPU가 읽을 때마다 flush 비용이 듭니다. 가중치 로딩은 GPU를 거쳐 staging하므로 영향이 없습니다.
- **기존 문제**: `naive_gemm.hip`의 MoE workspace deleter가 shadow를 해제하지 않고 `RocmBuffer`를 delete합니다.

## 7. 학습 포인트

- Windows에서 "integrated GPU"라고 해서 device 메모리가 CPU에서 보이는 것은 아닙니다. 이는 WDDM이 결정하고, UMA carve-out 머신에서는 큰 메모리 풀이 device 전용입니다.
- 아무도 테스트하지 않는 하드웨어용으로 짠 경로(여기서는 discrete GPU shadow)는, 새 플랫폼이 그 경로에 의존하기 시작해야 비로소 드러나는 방식으로 틀려 있을 수 있습니다.
- 빌드가 ABI 경계를 넘어야 할 때는, 벤더 DLL의 import table을 확인하는 편(`llvm-readobj --coff-imports`)이 빌드를 여러 번 시도하는 것보다 툴체인 문제를 빨리 해결합니다.
