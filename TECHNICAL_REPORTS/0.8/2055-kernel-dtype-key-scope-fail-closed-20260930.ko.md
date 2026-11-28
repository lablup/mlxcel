# 기술 보고서: PR #2055 - 커널 dtype 키 검사기의 범위를 fail-closed로 변경

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: Python (CI 검사기), Bash (동반 테스트), Makefile, GitHub Actions YAML, Markdown

**위험도**: 낮음 (CI 스크립트와 문서만 바뀌었고 소스, 커널, 빌드 경로는 바뀌지 않았습니다. 새로 생긴 실패 조건은 모두 검사기 안에 있으며 현재 트리는 통과합니다)

## 요약

`scripts/ci/check_kernel_dtype_keys.py`는 #1053, #1054 계열의 버그를 막습니다. CUDA에서 JIT 커널의 캐시 이름은 `name + template_arguments_hash(template_args)`뿐이므로, `template_args`에 입력 dtype이 없는 launch는 처음 컴파일된 dtype의 모듈을 이후 모든 dtype에 조용히 재사용합니다. 검사기는 무엇을 검사할지를 검사 대상 파일에서는 보이지 않는 두 규칙으로 정했습니다. 두 디렉터리의 `*.cpp`만 glob했고, `cuda_kernel(` 문자열이 없는 파일은 건너뛰었습니다. 그래서 launch를 헤더, 공용 helper, HIP 전용 파일로 옮기는 리팩터링은 범위를 줄이면서도 검사는 계속 OK를 출력했습니다. 이슈 #1875(에픽 #1801의 일부)가 이를 기록했고, 이미 한 번 일어난 일이었습니다. `mlx_cxx_bridge.cpp`의 #1804 ROCm fault probe는 `fast::hip_kernel`을 호출하지만 한 번도 범위 안에 들어온 적이 없습니다.

이 PR은 범위를 fail-closed로 만듭니다. 검사기는 이제 git이 추적하거나 추적할 모든 C, C++, CUDA, HIP, Objective-C++ 소스와 헤더를 스캔하고, 주석 밖에서 `cuda_kernel(` 또는 `hip_kernel(`을 호출하는 파일을 범위에 넣으며, 스캔한 파일 수와 범위 안 파일 수를 함께 보고하고, 범위 집합을 `EXPECTED_IN_SCOPE`에 고정해 어떤 변화든 실패로 만듭니다. 새 동반 테스트 `check_kernel_dtype_keys_test.sh`는 트리의 임시 복사본을 16가지 방식으로 변형하며, `make verify-kernel-dtype-keys`와 `kernel dtype keys` CI job 양쪽에서 실행됩니다. 현재 트리는 스캔한 188개 중 9개가 범위 안인 상태로 통과합니다.

## 1. 문제 정의

### 결함

이 PR 이전의 `main()`은 `SEARCH_DIRS = ("src/lib/mlx-cpp/turbo", "src/lib/mlxcel-core/cpp")`를 `glob("*.cpp")`로 돌았고, `check_file`은 `"cuda_kernel(" not in src`이면 바로 반환했습니다. 결과는 세 가지였습니다.

- 헤더(`.h`, `.hpp`, `.cuh`)로 옮겨진 launch는 토큰 유무와 관계없이 읽히지 않았습니다.
- 파일 밖의 공용 helper로 옮겨진 launch는 그 파일 전체를 범위에서 빼냈고, 남겨진 `template_args` 초기화도 함께 빠졌습니다.
- `fast::hip_kernel`로만 launch하는 파일은 범위 밖이었습니다. ROCm도 같은 위험을 갖는데도 그렇습니다(아래 참고).

성공 메시지가 이를 가렸습니다. 출력되는 `scanned`는 토큰 필터 이전에 모든 `*.cpp`를 센 값이었습니다. 이슈 갱신 시점에는 실제로 8개를 검사하면서 `18 source files scanned`라고 출력했습니다. 관련 없는 `.cpp`를 추가하면 숫자가 올라갔고, launcher를 모두 없애도 숫자는 내려가지 않았습니다.

### HIP이 같은 위험을 갖는 이유

ROCm overlay는 CUDA와 똑같이 커널 이름을 만듭니다. `src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/custom_kernel.cpp:230-231`의 `"custom_kernel_" + name + template_arguments_hash(template_args)`입니다. `rocm::get_jit_module`(`patches-rocm/mlx/backend/rocm/jit_module.cpp:502`)은 컴파일된 모듈을 `std::to_string(mlx_device.index) + ":" + name`을 키로 하는 프로세스 전역 map에 memoise하고, miss일 때만 builder를 호출합니다. 추가된 것은 device index뿐이고 dtype은 들어가지 않으므로, ROCm의 캐시 키는 CUDA와 똑같이 dtype을 구분하지 못합니다. 입력 dtype을 키에 붙이는 백엔드는 Metal뿐입니다.

### 이미 일어난 일

#1804 ROCm fault probe(`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`의 `rocm_fault_probe_array`)는 CUDA launch가 없는 파일 안의 `fast::hip_kernel` launch이므로, 들어온 날부터 기존 범위 밖에 있었습니다. `template_args`를 inline `{}`로, 고정된 float32로 넘기므로 지금 dtype 키 버그는 없지만, 버그가 있었더라도 검사기는 잡을 수 없었습니다. 나머지 HIP launch 하나인 `mlx_cxx_kernels.cpp`의 BitNet은 같은 파일에 `cuda_kernel(`이 있어서 우연히 검사되었습니다. 에픽 #1814는 `.rocm` 포트를 더 추가할 예정이고, 그중 일부는 별도 파일에 들어갈 수 있습니다.

## 2. 변경 요약

- **`scripts/ci/check_kernel_dtype_keys.py`** (297줄 변경).
  - *스캔 집합.* `--root`가 work tree의 최상위이면 `source_files()`는 `git ls-files -z --cached --others --exclude-standard`를 사용합니다. 추적 파일과 무시되지 않은 미추적 파일은 보이고, 무시된 빌드 산출물과 `references/` 체크아웃(MLX 자체 launcher가 들어 있습니다)은 제외됩니다. work tree가 아니면 트리를 순회하되 최상위의 숨김 디렉터리와 `PRUNED_DIRS`(`target`, `node_modules`, `build`, `site`, `__pycache__`)만 건너뜁니다. 파일은 `SOURCE_SUFFIXES`(`.c .cc .cpp .cxx .cu .h .hh .hpp .hxx .cuh .inc .ipp .hip .m .mm`)로 거릅니다.
  - *범위 토큰.* `launches_jit_kernel()`은 `without_comments()`로 주석을 지운 소스에서 `\b(cuda|hip)_kernel\s*\(`을 찾습니다. lexer `LEXEME_RE`는 raw string(커널 소스는 `//`를 담은 raw string입니다), 일반 문자열, 문자 리터럴(`1'000` 같은 digit separator가 문자 리터럴을 열지 않도록 lookbehind 사용), 주석을 인식합니다. 지우는 것은 주석뿐이고 줄바꿈은 남겨 줄 번호를 유지합니다. 같은 줄에서 앞에 `CustomKernelFunction`이 오는 match는 vendored MLX 트리의 정의나 선언이므로 세지 않습니다. `\b`는 `precompiled_cuda_kernel(`을 제외합니다. 이것은 해시된 이름으로 JIT 컴파일하지 않고 미리 빌드된 바이너리를 불러옵니다.
  - *고정 목록.* `EXPECTED_IN_SCOPE`는 launcher 파일 9개를 나열합니다. turbo 파일 7개, `mlx_cxx_kernels.cpp`, 그리고 새로 들어간 `mlx_cxx_bridge.cpp`(#1804 probe를 명시하는 주석 포함)입니다. `scope_failures()`는 범위가 비었을 때, 고정된 파일이 사라졌을 때, 고정된 파일이 남아 있지만 더 이상 launch하지 않을 때(그 `template_args`가 이제 검사되지 않는다는 설명 포함), 고정되지 않은 새 launcher가 생겼을 때 실패합니다.
  - *출력.* 성공 시 `kernel-dtype-keys: OK, 9 in scope (launching cuda_kernel or hip_kernel) of 188 source files scanned.`를 출력합니다. 실패 시에도 같은 수를 출력한 뒤 규칙 실패와 범위 실패를 별도 블록으로 보여 줍니다.
  - *CLI.* `--root DIR` 인자로 다른 트리를 검사할 수 있고, 동반 테스트가 이를 사용합니다. 이슈 범위에 따라 규칙 자체(`TEMPLATE_ARGS_RE`, `DTYPE_BINDING_RE`, 초기화별 검사)는 바뀌지 않았습니다.
- **`scripts/ci/check_kernel_dtype_keys_test.sh`** (신규, 271줄). 고정된 launcher가 있는 모든 디렉터리(`CustomKernelFunction hip_kernel(` 정의가 존재하도록 vendored MLX 트리 포함)를 임시 트리로 복사하고, 변형한 뒤 `--root`로 검사기를 실행해 종료 코드와 메시지 일부를 확인합니다. 변형은 대상 문자열이 하나도 맞지 않으면 실패하는 Python helper를 거치므로, 변형이 더 이상 적용되지 않아 케이스가 통과하는 일은 없습니다. 케이스는 손대지 않은 대조군, dtype 키 제거, 같은 디렉터리와 새 디렉터리의 헤더로 옮긴 launch, helper로 옮긴 launch, 주석 처리된 launch, 삭제된 launcher, HIP 전용 launcher, int만 있는 초기화와 dtype 키가 있는 초기화를 넣은 실제 bridge probe, 올바르지만 고정되지 않은 새 launcher, `"`와 `/*`를 담은 raw string이 있는 주석 전용 헤더, 중첩된 `build/` 디렉터리 안의 `.hip` 파일, 무시된 `references/mlx`와 미추적 launcher가 있는 git work tree, 빈 범위를 포함합니다.
- **`Makefile`.** `verify-kernel-dtype-keys`가 동반 테스트도 실행하므로 `make verify`와 `make verify-rocm` 모두 이를 실행합니다. help 문구는 CUDA와 HIP을 명시하고 #1875를 인용합니다.
- **`.github/workflows/ci.yml`.** `kernel dtype keys` job의 첫 step 이름을 CUDA와 HIP으로 바꾸고, 두 번째 step에서 동반 테스트를 실행하며, job 주석이 두 가지를 설명합니다.
- **`docs/code-guidelines.md`.** 적용 단락이 HIP 규칙, 스캔 집합, 고정 목록, 동반 테스트, 남은 두 한계(inline `template_args`, 직접 호출 없는 launch)를 설명합니다.
- **`CONTRIBUTING.md`.** `make verify` 선행 목록이 6개라고 적고 `verify-kernel-port-dispatch`를 빠뜨리고 있었습니다. 이제 7개를 나열하고, dtype 키 게이트가 CUDA와 HIP을 다루며 범위 검사를 포함한다고 설명합니다.

커밋 이력: `58559aa3`은 fail-closed 범위, 고정 목록, 동반 테스트, 연결 작업입니다. 리뷰에서 나온 `e3d01509`은 파일 목록을 git으로 얻고, `.hip`을 스캔하고, 디렉터리를 최상위에서만 건너뛰며, 테스트의 GNU 전용 `sed -i`를 Python helper로 바꿉니다.

## 3. 기술적 선택과 그 이유

### 범위를 넓히기만 하지 않고 고정한 이유

이슈는 glob 확장, 두 수 출력, 기대 집합 고정의 세 방안을 비교했습니다. 확장은 헤더로 옮긴 launch를 잡고, 두 수를 출력하면 로그를 읽는 사람에게 감소가 보이지만, 이슈의 핵심 사례는 고정만 잡습니다. launch가 공용 helper로 옮겨지면 원래 파일의 `template_args`가 검사되지 않은 채 남고, helper 파일에는 초기화가 없을 수 있습니다. PR은 세 가지를 모두 적용합니다. 비용은 launcher를 정당하게 추가하거나 제거할 때 한 줄을 고쳐야 한다는 것이고, 실패 메시지가 그 줄을 정확히 알려 줍니다.

### "삭제됨"과 "launch 중단"을 구분

삭제된 launcher와 호출이 다른 곳으로 옮겨진 launcher는 대응이 다릅니다. 앞의 것은 고정 목록만 고치면 되고, 뒤의 것은 어떤 `template_args`가 더 이상 검사되지 않는다는 뜻입니다. `scope_failures()`는 고정된 경로가 아직 있는지 확인해 각각 다른 메시지를 출력하므로, 리뷰어가 둘 중 무엇인지 추론할 필요가 없습니다.

### 고정되지 않은 새 launcher도 실패

이것이 없으면 고정 목록은 점점 낡습니다. 새 launcher는 규칙으로 검사되지만 나중에 범위 밖으로 옮겨지는 것은 막지 못합니다. 범위 안의 모든 파일을 고정하도록 요구하면 고정 목록이 항상 범위와 같아지고, 그래야 나중의 감소를 감지할 수 있습니다.

### 스캔 대상은 git이 결정

첫 커밋은 어느 깊이에서든 이름으로 디렉터리를 건너뛰었습니다. 리뷰에서 이것이 과하게 제외하고(실제 소스를 담은 중첩 `build` 디렉터리) 동시에 덜 제외한다는 것(로컬 `references/` 체크아웃에는 MLX 자체의 `cuda_kernel(` launcher가 있어, 고정되지 않은 파일로 로컬 실행이 실패)이 드러났습니다. `git ls-files --cached --others --exclude-standard`는 저장소가 이미 관리하는 `.gitignore`에 판단을 맡기면서도 미추적 파일을 포함하므로, 새 launcher는 커밋 전에도 보입니다. 순회 fallback은 work tree가 아닌 동반 테스트의 임시 복사본을 위한 것이며 최상위 디렉터리만 건너뜁니다.

### 부분 문자열 검사 대신 lexer

기존 필터 `"cuda_kernel(" in src`는 토큰을 언급하는 주석도 launch로 취급했습니다. 주석을 지우려면 문자열 위치를 알아야 하고, 이 트리의 커널 소스는 `//`와 `/*`를 담은 raw string입니다. `LEXEME_RE`는 raw string, 문자열, 문자 리터럴을 단위로 건너뛰고 주석만 지웁니다. digit separator lookbehind가 필요했던 이유는 `0xFF'FF`가 그렇지 않으면 문자 리터럴을 열어 뒤의 코드를 가리기 때문입니다.

### HIP이 규칙을 그대로 따르는 이유

규칙의 전제는 캐시 키에 입력 dtype이 없다는 것입니다. ROCm overlay의 `custom_kernel.cpp:230-231`과 `jit_module.cpp:502`는 같은 이름 구성과, device index와 이름을 키로 하는 프로세스 전역 memo를 보여 주므로 HIP에도 전제가 그대로 성립합니다. HIP 전용 규칙은 필요 없고, 토큰에 `hip_kernel(`만 포함하면 됩니다.

## 4. 검증

작성자 실행(gfx1151): `make verify-kernel-dtype-keys verify-kernel-port-dispatch verify-versions verify-llama-compat verify-fmt` 통과, `cargo test --features rocm --test dead_doc_pointers` 통과(테스트 2개), 주 체크아웃에서 검사기 통과(188개 스캔, 9개 범위 안). 동반 테스트의 negative 케이스 11개는 모두 이전 검사기에 대해 실행하면 실패하므로, 각 케이스가 이전 검사기가 놓친 것을 잡는다는 것이 확인됩니다.

오케스트레이터 검증(gfx1151 호스트, origin/main `2c8045fe` 기준 브랜치). 변경이 CI 스크립트와 문서뿐이므로 전체 테스트 스위트 대신 영향을 받는 게이트를 실행했습니다.

- `make verify-kernel-dtype-keys`: exit 0. 검사기가 188개 소스 파일 중 9개가 범위 안이라고 보고했고, 동반 테스트의 negative 케이스가 모두 통과했습니다.
- `make verify-versions verify-kernel-port-dispatch verify-llama-compat verify-fmt`: exit 0.
- `.github/workflows/ci.yml`이 YAML로 파싱되며, 새 동반 테스트 step은 371번째 줄에 있습니다.

## 5. 학습 포인트

- **규칙을 만족하는 트리에만 실행되는 범위 규칙은 실패할 수 없습니다.** 이전 검사기는 읽은 파일에 대해서는 틀린 적이 없고, 읽는 파일이 줄었다는 것을 알 방법이 없었을 뿐입니다. 동반 테스트의 방식, 즉 리팩터링처럼 복사본을 변형하고 실패를 확인하는 것은 범위가 목록이 아니라 유도되는 모든 CI 검사에 적용할 수 있는 해법입니다.
- **본 것이 아니라 검사한 것을 보고합니다.** `18 source files scanned`는 커버리지처럼 읽혔지만 실제로는 8개를 검사했습니다. 범위 안 수를 따로 출력하면 조용한 감소가 보이는 숫자가 됩니다.
- **테스트 자신의 변형을 검증합니다.** `sed`로 파일을 고치는 negative 케이스는 관련 없는 수정 후 패턴이 맞지 않게 되면 헛되이 통과합니다. 테스트의 Python 치환 helper는 match가 0개면 실패하므로, 각 케이스는 변형이 적용되었음을 증명합니다.
- **vendored 백엔드의 공통 가정을 확인합니다.** HIP 공백은 ROCm overlay의 `custom_kernel.cpp`와 `jit_module.cpp`를 읽고 CUDA 키 구조가 반복되는 것을 보고 찾았습니다. 한 백엔드의 내부 구현으로 정당화되는 규칙이라면, 다른 백엔드가 다르다고 가정하기 전에 대응 코드를 읽어야 합니다.
- **로컬 게이트에는 이식 가능한 셸이 중요합니다.** 접미사 없는 `sed -i`는 GNU에서 동작하고 BSD sed에서 실패하는데, 이 테스트는 macOS 개발자 게이트인 `make verify`에서 실행됩니다. 리뷰가 머지 전에 이를 잡았습니다.

## 6. 검증되지 않은 부분

- **Metal 및 CUDA 호스트.** 이 호스트에서는 사용할 수 없습니다. 변경은 Metal이나 CUDA 코드 경로를 건드리지 않지만, `ubuntu-latest`의 `kernel dtype keys` CI job이 gfx1151 밖에서의 첫 실행입니다.
- **macOS 셸.** 동반 테스트를 BSD sed에 안전하게 만들었지만, `make verify`가 이를 실행할 macOS에서는 돌려 보지 않았습니다.
- **검사기가 여전히 보지 못하는 launch 형태.** `template_args`를 inline으로 넘기는 launch(#1804 probe가 `{}`로 그렇게 합니다)는 범위 안이지만 검사할 것이 없고, 함수 포인터나 다른 곳에 정의된 매크로로 도달하는 launch는 그 파일을 범위에 넣지 않습니다. 고정 목록은 고정된 파일이 그런 형태로 바뀌는 것은 잡지만, 처음부터 그런 형태로 시작하는 새 파일은 잡지 못합니다. 두 한계는 `docs/code-guidelines.md`와 스크립트 docstring에 적혀 있습니다.
- **전체 테스트 스위트.** Rust나 C++ 소스가 바뀌지 않았으므로 오케스트레이터는 실행하지 않았습니다.

## 7. 남은 작업

- #1814가 `.rocm` 포트를 추가할 때, 새 launcher 파일은 같은 PR에서 `EXPECTED_IN_SCOPE`에 추가해야 합니다. 추가하기 전까지 검사기는 실패하며, 이는 의도된 동작입니다.
- 앞으로 inline `template_args`가 필요한 launcher가 생기면, 규칙이 이를 읽도록 확장해 문서화된 첫 번째 한계를 해소할 수 있습니다.

참고: #1875(이 PR로 종료), #1801, #1053, #1054, #1803, #1804, #1814, PR #1877, PR #2026, PR #2029.
