# PR #2530: fused imag()를 0으로 만들던 Metal compiled.cpp overlay 제거

**Date**: 2026-10-22
**Status**: Completed (PR open, #2333 close 예정)

## 요약

PR #2530은 Metal `compiled.cpp` source overlay를 제거한다. 이 overlay는 fused op 입력 중 타입이 op의 출력 타입과 다른 것을 모두 출력 타입으로 cast했다. `Imag`(입력 complex64, 출력 float32)에서는 `Imag()(static_cast<float>(tmp_z))`가 생성됐다. Metal의 `complex_t` 변환은 `.real`만 남기므로, compile된 complex64 값의 `imag()`가 0을 반환했다. `Real`은 우연히 맞게 동작했다. 현재 MLX pin에서는 MLX op가 스스로 입력을 promote하므로 이 cast가 필요 없다. 그래서 overlay를 없애고 upstream 생성 코드를 쓴다. 또한 캐시된 MLX checkout에서 제거된 overlay 파일을 원복하는 단계를 추가했고, compile probe graph 4개와 테스트 2개를 더했다. Metal은 self-hosted runner에서 두 가지로 검증했다. fix를 적용한 run은 전부 green이고, 옛 overlay를 되돌린 run은 imag 테스트만 실패했다. 변경 규모는 7개 파일, +178/-544이며, 삭제분 중 509줄이 overlay 파일이다.

## 1. 문제 정의

`mlx/backend/metal/compiled.cpp` overlay는 1923da32에서 추가됐다. 피연산자 타입이 섞인 fused kernel이 compile되게 하려는 목적이었다. 입력 타입이 출력 타입과 다르면 출력 타입으로 cast했다. 출력 타입이 입력보다 좁고 변환이 값을 보존하지 않는 경우에는 이 cast가 틀린다. `Imag`는 complex64를 받아 float32를 반환하는데, `float`로 cast하면 허수부가 사라진다. 같은 cast가 비교 연산도 깨뜨렸다(#1792, 앞선 테스트가 다룬다). 이슈 #2333은 overlay가 아직 필요한지(옵션 b), 아니면 제자리에서 고쳐야 하는지(옵션 a)를 물었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `overlays/metal/mlx/backend/metal/compiled.cpp` | 삭제(509줄) |
| `src/lib/mlxcelverse/CMakeLists.txt` | overlay 목록에서 `compiled.cpp` 제거, 제거된 overlay를 원복하는 단계 추가 |
| `overlays/metal/README.md` | overlay 목록 갱신 |
| `cpp/mlx_cxx_compile_probe.cpp`, `cpp/mlx_cxx_bridge.h`, `src/lib.rs` | probe graph 7-10 |
| `compiled_dtype_cast_tests.rs` | `compiled_real_of_complex64_matches_eager`, `compiled_imag_of_complex64_matches_eager` |

## 3. 기술적 선택과 그 이유

### 3.1 cast를 고치지 않고 overlay를 제거한다(옵션 b)

pin(`bff10220`)에서 upstream `compiled.cpp`와 overlay의 차이는 헤더 주석과 이 cast 블록뿐이다. cast는 섞인 피연산자 타입을 위한 것이었는데, `mlx/ops.cpp`는 이제 그런 입력을 만들지 않는다.
- 이항 연산과 비교 연산은 두 입력을 promote된 dtype으로 astype한다.
- `logical_and`, `logical_or`는 bool로 cast한다.
- `where`는 조건을 bool로, 두 분기를 출력 dtype으로 cast한다.
- `Real`, `Imag`는 `complex64_t`를 받는다.
- fused `AsType`은 upstream에서 `cast_to<T>`로 생성된다(ml-explore/mlx#4351).

따라서 fused op는 섞인 피연산자 타입을 보지 않는다. CUDA에는 이런 overlay가 없고, 같은 graph를 upstream 생성 코드로 이미 실행한다. 불필요하면서 틀리기까지 한 cast 하나 때문에 upstream 파일 509줄 사본을 유지하면, pin을 올릴 때마다 갱신 부담도 생긴다.

### 3.2 캐시된 checkout에서 제거된 overlay를 원복한다

overlay는 fetch된 `_deps/mlx-src`에 복사된다. `build.rs`는 pin이 같으면 그 tree를 재사용한다. 그래서 목록에서 파일만 빼면 warm build마다 패치된 `compiled.cpp`가 그대로 남는다. `mlx_apply_source_overlays`는 이제 fetch된 tree에서 `git diff --name-only -- mlx/backend/metal`을 실행하고, 수정됐지만 overlay 목록에 더는 없는 파일마다 `git checkout`을 실행한다. 로그는 `Restored upstream <path> (no longer overlaid)`로 남는다. git checkout이 아니거나 `git diff`가 실패하면 경고하고 건너뛴다. 제거된 파일의 `git checkout`이 실패하면 configure를 중단하므로, 옛 overlay로 조용히 빌드되는 일은 없다.

### 3.3 실패를 재현하는 probe graph

graph 7-10은 허수부가 0이 아닌 `z = a + i*b`의 `real`과 `imag`를 계산한다. 한 쌍은 compile된 graph 안에서 `z`를 만들고, 다른 한 쌍은 eager로 만든 complex64 입력을 받는다. 각 결과는 곱셈과 덧셈에 넘겨 op가 fuse되게 한다. 각 테스트는 compiled와 compiled-shapeless 출력을 eager와 비교하고, eager는 host 산술과 비교한다. #1792 테스트는 주석 외에는 그대로다.

## 4. 검증

### 4.1 로컬(Linux, 기본 feature, CPU backend)

| 점검 | 결과 |
|---|---|
| warm cache 원복, 변경 전 | 캐시된 `_deps/mlx-src`에서 `git status --short mlx/backend/metal`이 `compiled.cpp`, `device.cpp`, `kernels/utils.h`, `quantized.cpp`를 수정됨으로 표시 |
| warm cache 원복, 같은 cache에서 `cargo test ... --no-run` 후 | `device.cpp`, `kernels/utils.h`, `quantized.cpp`만 수정됨. 로그에 `Restored upstream mlx/backend/metal/compiled.cpp (no longer overlaid)`. 파일은 pin의 upstream과 byte 단위로 동일 |
| `cargo test --profile test-fast -p mlxcel-core --lib compiled_dtype_cast_tests` | 7 passed |
| `cargo fmt --all -- --check`, `cargo clippy ... -D warnings` | 문제 없음 |

CPU 실행은 probe, eager oracle과 host 산술의 일치, CPU compile을 검증한다. Metal fix 자체는 검증하지 못한다.

### 4.2 Metal(self-hosted macOS runner, nightly-verify dispatch)

| 구분 | Run | 결과 |
|---|---|---|
| fix 적용 | [39703647781](https://github.com/lablup/mlxcel/actions/runs/39703647781) | 전체 `make verify` green. 새 테스트 2개와 #1792 테스트 5개를 포함해 `compiled_dtype_cast_tests` 7개 모두 통과 |
| 되돌림(새 테스트와 옛 overlay를 둔 임시 branch) | [39707309473](https://github.com/lablup/mlxcel/actions/runs/39707309473) | `compiled_imag_of_complex64_matches_eager`만 실패: `compiled [0.2, -0.5, 0.7, 0.0, 3.0] != eager [1.6, 0.100000024, 1.1, 2.0, 9.0]`. 예측대로 `compiled_real_of_complex64_matches_eager`는 통과 |

되돌린 run은 새 테스트가 명시한 이유로 실패하고, fix가 이를 green으로 바꾼다는 점을 확인해 준다. 임시 branch는 삭제했고, 자동 생성된 추적 이슈 #2536은 닫았다.

## 5. 미검증 항목과 남은 일

- **실행하지 않음:** 실제 모델의 Metal build에서는 확인하지 않았고, probe 테스트로만 확인했다.
- **warm build 원복은 Linux에서만 확인했다.** 원복은 Linux cache에서 실행됐다. macOS도 같은 CMake 코드를 쓰지만, 이 PR에서 macOS의 warm cache를 시험하지는 않았다.
- **pin 이동:** overlay README는 더는 `compiled.cpp`를 나열하지 않는다. 이후 pin에서 타입이 섞인 fused 입력이 다시 생기면, #1792 테스트와 complex64 probe 테스트가 이를 감지한다.

## 6. 배운 점

- overlay는 특정 pin 시점의 upstream 코드를 얼려 둔 사본이다. 우회하던 동작을 upstream이 고치거나 구조를 바꿨다면, 갱신하기 전에 pin과 다시 diff해야 한다. 이번에는 차이가 헤더와 cast뿐이어서 삭제가 더 작은 변경이었다.
- fetch된 source tree가 캐시되는 동안에는 overlay 목록에서 파일을 빼는 것만으로 부족하다. overlay를 폐기할 때는 파일을 원복하는 단계를 함께 넣어야 한다.
- backend 전용 버그의 fix에는 fix 없이 실패하는 테스트를 그 backend에서 돌려 봐야 한다. CPU suite는 전후 모두 통과했으므로, 테스트가 의미 있다는 것은 되돌린 Metal run만 보여 줬다.
