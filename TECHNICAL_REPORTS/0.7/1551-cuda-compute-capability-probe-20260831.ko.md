# 기술 보고서: PR #1551 - CUDA compute capability 프로브

**작성일**: 2026-08-31

**작성자**: mlxcel maintainers

**상태**: 구현과 sm_70 검증 완료. sm_80 이상 검증은 GB10 호스트로 이월

---

## 요약

PR #1551(이슈 #1537)은 mlxcel에 CUDA compute capability 개념을 처음으로 들여온다. 그 전까지 `grep -rn "compute_capability" src/ --include=*.rs`는 아무것도 찾지 못했다. 아키텍처 판단을 전부 MLX의 C++ 게이트에 위임한 탓에, mlxcel은 자신이 어떤 아키텍처 위에서 돌고 있는지 말할 수 없었고, 어떤 아키텍처로 컴파일됐는지 기록하지도 않았으며, 둘이 어긋났다는 사실을 사용자에게 알릴 수단도 없었다.

이 변경이 없애는 구체적 실패는 이렇다. 배포되는 x86_64 아카이브는 `80;86;89;90a;100;120`으로 빌드되므로 sm_70 코드 오브젝트를 아예 담고 있지 않다. 이걸 Volta 카드에서 실행하면 첫 커널 실행 시점에 불투명한 CUDA 로드 오류로 죽는데, 오류 메시지는 바이너리가 가진 아키텍처도 발견된 장치도 알려주지 않는다. 이제는 장치 초기화 단계에서 양쪽을 모두 이름 붙여 밝히고 거부한다.

배관과 진단만 다루며 커널 선택이나 디스패치 동작은 바꾸지 않는다. 에픽 #1536의 기반 항목이고, 이후 서브이슈 다섯 건이 여기 얹힌다.

## 1. 문제 정의

같은 결핍에서 파생된 세 가지 공백이 있었다.

**"지금 무엇 위에서 도는가"에 대한 런타임 답이 없었다.** MLX는 장치별로 `compute_capability_major` / `compute_capability_minor`를 캐시해 두지만, 그 값이 Rust 쪽으로 드러나지 않았다. 어떤 아키텍처 의존 커널 경로를 탔는지 기록할 수도, 호스트에 따라 자체 기본값을 조정할 수도 없었다.

**"무엇으로 컴파일됐는가"에 대한 빌드타임 기록이 없었다.** `build.rs`는 이미 `MLX_CUDA_ARCHITECTURES`나 `nvidia-smi --query-gpu=compute_cap` 파싱으로 아키텍처 목록을 만들고 있었지만, 그 문자열은 CMake까지만 가고 끝났다. 바이너리는 자기 타깃 집합을 몰랐다.

**둘의 불일치를 알아챌 방법이 없었다.** 가상의 상황이 아니다. 릴리스 워크플로는 x86_64를 `80;86;89;90a;100;120`으로, aarch64를 `90a;100;121`로 내보내며 둘 다 Ampere에서 시작한다. `nvidia-smi`가 없는 컨테이너에서 소스 빌드를 하면 폴백이 `90a`라, 빌드한 바로 그 호스트에서 돌지 못하는 바이너리가 조용히 만들어진다. 두 경우 모두 예전에는 첫 forward 패스 깊숙한 곳의 CUDA 로드 실패로만 드러났다.

## 2. 변경 요약

파일 19개, +1155 / -18.

**런타임 프로브.** `mlxcel_core::hardware::cuda_compute_capability() -> Option<(u32, u32)>`를 새 파일 `src/lib/mlxcel-core/src/cuda_arch.rs`에 구현하고 `hardware.rs`에서 재수출한다. compute capability는 프로세스 수명 동안 바뀔 수 없으므로 `OnceLock`에 캐시한다. MLX의 `device_info` 맵을 읽는 새 브리지 함수 `gpu_compute_capability(index)`가 뒤를 받친다.

**빌드타임 기록.** `build.rs`가 `resolve_cuda_architectures()`에서 아키텍처 목록을 한 번 확정해 CMake로 넘기고, 동시에 `cargo:rustc-env=MLXCEL_CUDA_ARCHITECTURES`로 기록한다.

**불일치 검사.** `arch_list_coverage()`가 컴파일된 목록을 CUDA 호환 규칙에 따라 현재 장치와 대조하고, `cuda_arch_mismatch()`가 미커버 판정을 `CudaArchMismatch`로 바꾸며, `enforce_cuda_arch_compatibility()`가 이를 올린다. 새 `initialize_runtime_checked()`가 장치 초기화 시점에 이를 호출하므로 `generate`, `chat`, `embed`, `rerank`, `detect`와 서버가 모두 첫 커널 이전에 거부한다.

**진단.** 시작 시 장치 보고에 `Detected N GPU(s)` 옆으로 아키텍처 정보가 붙는다. CLI와 서버 양쪽이다. `MLXCEL_TRACE_ARCH`는 capability, 컴파일된 목록, 커버리지 판정을 프로세스당 한 번 출력한다. CUDA `QuantizedMatmul` 오버레이는 디스패처가 첫 호출에서 고른 양자화 matmul 경로를 함께 보고한다.

**문서.** `docs/environment-variables.md`에 `MLXCEL_TRACE_ARCH` 항목을 추가하고, `docs/installation.md`의 CUDA 아키텍처 선택 절에서 거부를 유발하는 두 경우를 짚어 상호 참조한다.

## 3. 기술적 선택과 이유

### 3.1 프로브는 CUDA를 직접 부르지 않고 MLX의 `device_info` 캐시를 읽는다

대안은 Rust 쪽에서 `cudaDeviceGetAttribute`를 한 번 더 호출하는 것이었다. 그러면 크레이트에 CUDA 헤더 의존이 생기고 MLX가 이미 들고 있는 상태를 중복 보유하게 된다.

결정적인 근거는 중복 제거보다 미묘한 곳에 있다. **장치 속성 읽기는 cubin을 로드하지 않는다.** 바로 이 성질 덕분에, 불일치 검사가 진단해야 할 그 바이너리 위에서도 프로브가 여전히 동작한다. 만약 프로브가 컴파일된 장치 코드를 건드리는 무언가를 필요로 했다면, 아키텍처가 어긋난 바이너리에서 실패했을 것이고 검사는 유일하게 의미 있는 경우에 무용지물이 됐을 것이다. 같은 이유로 Metal 빌드와 CPU 전용 빌드에서도 동일 심볼이 링크되며, 그 경우 맵에 capability 키가 없으므로 프로브는 `None`을 보고한다.

### 3.2 확정 지점 하나가 CMake와 크레이트 양쪽을 먹인다

`resolve_cuda_architectures()`는 `build.rs::main`에서 한 번만 호출된다. 같은 문자열이 CMake의 `MLX_CUDA_ARCHITECTURES`로도, `cargo:rustc-env`로도 간다. 빌드가 컴파일하는 대상과 바이너리가 보고하는 대상이 **구조적으로** 같은 문자열이지, 오늘 우연히 일치하는 두 절차가 아니다. 누군가 한쪽 경로의 탐지 로직만 바꿔 두 값이 어긋나는 부류의 결함을 원천 차단한다. 그런 어긋남이 생기면 불일치 검사는 확신에 차서 틀린 답을 내놓게 된다.

### 3.3 커버리지는 동등 비교가 아니라 CUDA의 실제 규칙을 담는다

순진한 검사는 `device in list`다. 그런데 이건 문제가 되는 방식으로 틀린다. sm_80 cubin은 sm_86에서 돌고, 배포되는 릴리스 매트릭스가 바로 그 성질에 기대고 있다. 동등 비교였다면 멀쩡히 동작하는 구성에 불일치를 보고하고 시작을 거부했을 것이다.

그래서 `entry_coverage()`는 엔트리 변종 셋과 코드 오브젝트 두 종류를 모델링한다.

- **Generic**(`80`, `86`): cubin은 같은 major에서 minor가 같거나 높은 쪽을 커버한다(`major ==`, `minor <=`). PTX는 major를 넘어서까지 전방 JIT되므로 규칙은 튜플 비교 `(entry) <= (device)`다.
- **아키텍처 특화**(`90a`): 정확히 그 타깃만 커버한다. MLX의 Hopper 양자화 커널에서 `90a`가 load-bearing인 이유와 같은 의미론이다.
- **패밀리 특화**(`f`): 자기 major 안에서 전방으로 이어진다.

`-real`과 `-virtual` 한정자는 엔트리가 어떤 코드 오브젝트를 내보내는지 고른다. `arch_list_coverage()`는 엔트리별 판정에 `.max()`를 취해 PTX보다 cubin 일치를 우선한다. 둘 다 돌긴 하지만 cubin 일치는 첫 실행에 JIT이 없다는 뜻이라 구분해 보고할 값어치가 있다.

### 3.4 의도적으로 둔 안전판 둘

**파싱되지 않는 목록은 "커버 안 함"이 아니라 "알 수 없음"이다.** `MLX_CUDA_ARCHITECTURES`는 이 파서가 모델링하지 않은 표기도 받는다. `native`나 `all-major` 같은 것들이다. 파싱 실패를 미커버로 취급하면 파서의 공백이 멀쩡한 바이너리에 시작 실패를 만들어낼 수 있다. 대신 검사가 no-op으로 물러난다. 진단이 존재 이유의 전부인 가드로서는 옳은 교환이지만 유지보수상 대가가 따른다. 새 CUDA 아키텍처 표기를 추가할 때 검사는 요란하게 깨지는 대신 조용히 약해지므로, 파서 케이스와 단위 테스트를 함께 넣어야 한다.

**`MLXCEL_DEVICE=cpu`는 거부를 우회한다.** 아키텍처가 맞지 않는 아카이브를 쥔 사람에게 남은 우회로는 정확히 하나뿐이고, 가드가 그것마저 빼앗아서는 안 된다.

### 3.5 검사는 장치 초기화 시점에 걸린다

`initialize_runtime_checked()`를 각 커맨드가 아니라 장치 초기화에 두어, 여섯 진입점이 모두 이를 물려받고 거부가 커널 실행 도중이 아니라 그 이전에 일어난다.

## 4. 검증

호스트: Tesla V100-PCIE-32GB(compute capability 7.0, sm_70), CUDA 12.9.41, 드라이버 575.51.03, x86_64. 이 머신의 유일한 GPU다.

- `cuda_arch` 단위 테스트 26개가 `--features cuda`에서, 그리고 해당 기능 없이 빌드했을 때도 통과한다. 정확 일치, PTX 전방 일치, `90a`, `f`, `-real` / `-virtual`, PTX보다 cubin 우선, 미일치, 오류 메시지를 덮는다. GPU가 필요한 것은 없다.
- `execution::runtime` 테스트 13개 통과. 거부가 `cuda_arch_mismatch()`를 정확히 따라가는지, GPU를 실제로 요청했을 때만 적용되는지를 포함한다.
- `cargo clippy --lib --tests -- -D warnings`, `cargo fmt --check`, `git diff --check` 모두 깨끗하고 CI 전체가 녹색이다.
- `MLX_CUDA_ARCHITECTURES=70` 빌드에서 프로브는 실제 장치를 보고한다. `compute capability 7.0 (sm_70); compiled for [70]; coverage: cubin`이 디코드 8스텝에 걸쳐 정확히 한 번 출력된다. `MLXCEL_TRACE_ARCH`를 끄면 두 줄 모두 사라지고 생성 결과는 그대로다.
- 부정 경로는 논증이 아니라 빌드해서 실행했다. `MLX_CUDA_ARCHITECTURES="80;86;89;90a;100;120"` 빌드의 `libmlx.a`는 249 MB로, 단일 아키텍처 sm_70 빌드의 155 MB와 대비된다. 여섯 아키텍처가 실제로 들어갔다는 독립적 증거다. 그 바이너리를 V100에서 실행하면 커널을 건드리기 전에 0이 아닌 코드로 종료하며, 오류가 컴파일된 목록과 발견된 장치를 모두 이름 붙여 알린다. 같은 바이너리도 `MLXCEL_DEVICE=cpu`에서는 CPU로 정상 시작한다.

## 5. 검증 한계와 후속 작업

이 머신에는 Ampere 이상 장치가 전혀 없다. 다음 항목들은 가정하지 않고 체크하지 않은 채 남겼으며, 에픽 #1536의 `## GB10 (sm_121) continuation` 절에 속한다.

- GB10에서 `cuda_compute_capability()`가 `Some((12, 1))`을 반환하는 것. 하드웨어에서 돈 것은 `Some((7, 0))` 쪽뿐이다. 브리지의 `major * 1000 + minor` 언패킹과 `121` 엔트리는 단위 테스트가 덮지만, sm_121 장치가 이 코드를 실행한 적은 없다.
- Metal 빌드에서 `cuda_compute_capability()`가 `None`을 반환하는 것. macOS 호스트가 없었다. Linux CPU 전용 빌드가 동일 분기, 즉 capability 키가 없는 `device_info`를 그대로 지나가며, 이를 Metal 실행이라고 주장하지 않고 도달 가능한 가장 가까운 증거로 명시했다.
- 기존 플랫폼에서 동작 무변화 및 GB10 베이스라인 불변. sm_70에서만 확인했다. GB10이나 Apple Silicon 베이스라인은 측정하지 않았고 양쪽 모두에 대해 처리량 주장을 하지 않는다.

`cargo test --features cuda`와 `cargo test` 전체 스위트는 돌리지 않았다. 이 호스트에서 MLX / CUTLASS / cuDNN 콜드 구성 하나가 30분에서 47분씩 걸려, 검증을 이 변경이 건드리는 모듈과 위의 종단 실행으로 좁히고 나머지는 CI에 맡겼다.

## 참고

- 이슈 #1537, 에픽 #1536(GB10 이월 절 포함)
- 릴리스 아키텍처 매트릭스: `.github/workflows/release.yml`(aarch64 `90a;100;121`, x86_64 `80;86;89;90a;100;120`)
- 아키텍처 확정과 자동 탐지: `src/lib/mlxcel-core/build.rs`
- 커버리지 규칙: `src/lib/mlxcel-core/src/cuda_arch.rs`
