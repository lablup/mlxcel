# 기술 보고서: PR #1058 - fix(paged): key the paged decode v2 JIT cache on input dtypes

**날짜**: 2026-08-07
**작성**: mlxcel 메인테이너
**검토**: 구현 및 보안 리뷰 사이클
**상태**: 부분 완료 (근본 원인 규명과 수정은 끝났으나, 이슈가 요구한 CUDA 런타임 단언은 구현 호스트에서 실행할 수 없었다)
**언어**: C++, Rust
**위험도**: 높음 (실제 출하 dtype에서 조용히 틀린 답, CUDA 한정)

---

## 요약

`paged_v2::sparse::sparse_tests::an_f16_cache_matches_the_reference_within_its_own_precision`가 GB10에서 상대오차 약 1.0으로 실패했다. 정밀도 손실이 아니라 결과가 입력과 무관하다는 신호다. 융합 sparse 경로가 틀린 계산을 한 게 아니라, 다른 dtype으로 컴파일된 커널을 실행하고 있었다. MLX는 커스텀 커널의 버퍼 파라미터 타입을 입력의 런타임 dtype에서 생성하는데, 그 dtype을 JIT 캐시 키에 넣는 건 Metal 백엔드뿐이다. 이 PR은 paged decode v2 커널 두 개의 `template_args`에 입력 dtype을 명명해 `template_arguments_hash`가 변종을 구분하게 하고, 한 프로세스에서 하나의 geometry를 두 pool dtype으로 돌리는 가드를 추가한다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1053이 GB10(DGX Spark, sm_121), release 프로파일, `--features cuda`, MLX 핀 `2c46b953db88965c4270cc7306eda6887a3247f2`에서 실패를 보고했다. 단언 지점의 허용오차는 `3e-2`이고 f16 저장 반올림을 이미 수용하도록 잡힌 값이라, 상대오차 1.0은 정밀도 문제가 아니다. 이슈는 허용오차 완화를 해법에서 명시적으로 배제했다.

이 실패가 `main`까지 눈에 띄지 않고 도달한 이유는 그 플랫폼에서 스위트가 판정을 내린 적이 없기 때문이다. `--test-threads=1` 없이는 libtest가 여러 호스트 스레드에서 MLX를 동시에 몰아 SIGABRT로 중간에 죽는다(#1048).

### 1.2 기존 문제

- **출하되는 구성에서 조용히 틀린 답이 나온다.** 테스트 지점 주석이 프로덕션 할당은 f16이라고 적어 두었으니, 커널이 읽는 pool도 f16이다. 실제 출하 형태에서 틀린 결과가 나오고 아무것도 throw하지 않는다.
- **테스트 자신의 디스패치 단언은 먼저 통과한다.** 수치 비교보다 앞서 `outcome.is_fused()`를 단언하고 그게 통과한다. 폴백을 정확히 배제해 주는 신호지만, 그 때문에 이 실패가 빌드 시스템이 아니라 커널 본문의 수치 버그처럼 보였다.
- **같은 약점이 모양이 다른 두 번째 실패 밑에도 깔려 있었다.** #1054는 `paged_v2::launch::launch_tests::chunk_size_does_not_change_the_answer`가 전체 스위트에서만 약 55896배 오차로 실패한다고 보고했는데, 그 이슈 본문이 후보로 지목한 "한 번 컴파일·캡처된 뒤 테스트를 가로질러 충돌할 만큼 느슨하게 키잡힌 커널·그래프 상태"가 바로 이것이다.

### 1.3 위험 평가

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| 한 v2 geometry를 두 pool dtype으로 돌리는 CUDA 프로세스가 예외 없이 입력과 무관한 수치를 반환 | 치명 | 조건 충족 시 확실 |
| 모든 macOS 러너에서 보이지 않아 평소 개발 루프로는 잡을 수 없음 | 높음 | 확실 |
| 새 `cuda_kernel` 호출부에서 같은 약점이 재발 | 높음 | 높음 (#1054 이전에는 가드 부재) |

---

## 2. 기술 검토

### 2.1 근본 원인

MLX 소스 두 곳이 JIT 캐시 키에 무엇이 들어가야 하는지를 다르게 본다.

`mlx/backend/common/metal_kernel.cpp`는 커널 이름에 입력 dtype을 덧붙이고 이유까지 적어 놓았다.

```cpp
// The generated source depends on the dtypes of the inputs and outputs
// and on how each input is passed (see `write_signature`). Include them
// in the kernel name so that a given name always maps to the same source.
```

`mlx/backend/cuda/custom_kernel.cpp`는 그러지 않는다.

```cpp
std::string kernel_name =
    "custom_kernel_" + name + template_arguments_hash(template_args);
```

반면 같은 파일의 `build_kernel`은 버퍼 파라미터 타입을 `dtype_to_cuda_type(arr.dtype())`에서 생성한다. 그리고 `cu::get_jit_module`(`mlx/backend/cuda/jit_module.cpp`)이 컴파일된 모듈을 정확히 그 이름으로 프로세스 전역 `std::unordered_map`에 메모이즈하고, 캐시 미스에서만 소스 빌더를 호출한다.

`paged_attention_decode_v2_partial`은 정수 템플릿 인자(`Dim`, `PageSize`, `NRep`, `QHeads`, `QGroups`, `DimsPerThread`, `NumWarps`)만 넘겼고 전부 geometry에서 파생된다. 그래서 같은 geometry의 f32 pool과 f16 pool이 한 이름으로 해시된다. 먼저 컴파일된 dtype이 프로세스 수명 내내 이기고, 나머지는 자기 버퍼를 잘못된 포인터 타입으로 읽는다.

`sparse_tests`에서는 `dtype::FLOAT32` 케이스들이 `an_f16_cache_matches_the_reference_within_its_own_precision`보다 먼저 돌아 f32 모듈이 먼저 캐시되고, f16 테스트가 f16 저장을 `const float*`로 읽는다. 상대오차 1.0은 비트를 재해석했을 때 나오는 값이다.

### 2.2 CUDA 한정인가

이슈가 명시적으로 물은 항목이고 탐색 범위를 가른다. Metal(M1 Ultra, macOS 26.5.2, `--features metal,accelerate`)에서 판정했다.

| 명령 | 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate -p mlxcel-core --lib paged_v2 -- --test-threads=1` | 92 통과, 0 실패, f16 테스트 포함 |
| 변경 전 `main`에서 전체 `mlxcel-core` lib 스위트 직렬 실행 | 1415 통과, 0 실패, 92.23초 |

즉 **CUDA 한정**이고 이유도 분명하다. Metal 캐시 키는 이미 dtype을 담고 있었다. 이 결과는 #1054의 플랫폼 질문도 동시에 답한다.

### 2.3 성능

커널 본문은 건드리지 않았으니 주어진 dtype에 대해 생성되는 코드는 그대로다. CUDA에서 모듈 수가 geometry당 하나에서 (geometry, dtype)당 하나로 늘어나는데, 이건 Metal이 늘 갖던 개수다. 한 프로세스는 KV 캐시 dtype 하나를 쓰므로 정상 상태 개수는 변하지 않고, 달라지는 건 두 번째 dtype이 첫 번째 것을 조용히 재사용하는 대신 자기 모듈을 컴파일한다는 점이다.

### 2.4 호환성과 의존성

- **파괴적 변경**: 없음. 공개 API도, 설정도, 환경 변수도 건드리지 않는다.
- **새 의존성**: 없음. `TemplateArg`는 이미 `Dtype`을 받고, 양쪽 백엔드가 이미 렌더링한다(Metal은 `typename` 템플릿 파라미터로, CUDA는 `using` 별칭으로).
- **트리 내 선례**: 융합 decode-MoE 커널은 처음부터 `{"T", T}`를 넘겨 왔고, 그래서 이 결함이 없었다.

### 2.5 코드 품질

Metal·CUDA 소스 문자열에서 `QType`, `KVType`, `VType`, `LseType`과 식별자가 충돌하는지 손으로 확인했다. 각 이름은 파일당 정확히 두 번, 주석 하나와 `template_args` 하나로만 나온다. CUDA 본문은 Metal 호스트에서 컴파일되지 않으므로 거기서 충돌이 나도 빌드 시점에 드러나지 않는다. 그래서 이 확인이 필요했다.

---

## 3. 기술적 결정

### 3.1 키를 어디서 고칠 것인가

**맥락:** 결함은 CMake로 가져오는 핀 고정 의존성인 MLX 안에 있고 `src/lib/mlx-cpp/patches-cuda/`가 이미 존재하므로, 상류 패치도 선택지였다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| Metal처럼 `mlx/backend/cuda/custom_kernel.cpp`가 dtype을 덧붙이도록 패치 | MLX 자체 것을 포함해 현재·미래 호출부를 한 번에 고침 | MLX 핀을 올릴 때마다 재적용할 패치가 늘고, 독자가 예상하지 못할 방향으로 상류와 갈라짐 |
| **선택: mlxcel 호출부의 `template_args`에 dtype을 명명** | 전부 트리 안에서 해결되고, MLX가 제공하는 기능(`template_arguments_hash`는 이미 `Dtype`을 해시)을 씀. MoE 커널이 쓰는 `{"T", T}`와 같은 형태 | 호출부마다 반복해야 해서 새 호출부가 빠뜨릴 수 있음 |

**근거:** `template_arguments_hash`가 이미 `Dtype`으로 구분하므로, MLX를 바꾸는 대신 MLX가 주는 메커니즘을 쓴다. 핀을 올려도 손댈 게 없고, 트리에 이미 자리 잡은 패턴과 똑같이 읽힌다.

**절충:** 호출부마다 반복해야 한다는 건 실제 약점이고, 그래서 #1054가 관례를 믿는 대신 `scripts/ci/check_kernel_dtype_keys.py`를 추가했다.

### 3.2 `QType`을 넣을 것인가

**맥락:** sparse 경로는 `sparse.rs:580`에서 query를 f32로 캐스팅하므로 거기서는 `QType`이 상수처럼 보인다.

**그래도 넣은 이유:** 다른 진입점에서는 상수가 아니다. `launch_tests::run_with_chunk`는 `q_array(dtype::FLOAT32)`로 query를 만들고 `f16_pools_match_the_gather_reference`는 `q_array(dtype::FLOAT16)`을 쓰는데, 둘 다 같은 커널에 도달한다. pool dtype만 키로 잡으면 이 쌍이 계속 충돌한다.

---

## 4. 구현 상세

`turbo/paged_attention_v2.cpp`에 `QType`, `KVType`, `VType`을 추가했다. `turbo/paged_attention_v2_merge.cpp`에는 `VType`, `LseType`을 추가했는데, 이쪽은 이전 키가 `Dim` 하나뿐이었다. head dim은 계열을 가로질러 반복되므로 `Dim` 단독은 특히 약한 키다.

두 추가 모두 커널 본문이 참조하지 않아도 반드시 필요하다는 주석을 달았다. 나중에 정리 작업이 죽은 템플릿 파라미터로 오해해 지우지 않도록 하려는 것이다.

### 4.1 회귀 커버리지

`two_pool_dtypes_at_one_geometry_do_not_share_a_compiled_kernel`은 geometry를 고정하고(정수 템플릿 인자가 전부 여기서 파생된다) pool dtype만 움직인다. 한 프로세스에서 f32, f16, 다시 f32 순으로 돌리고 각각을 자기 호스트 레퍼런스와 대조한다. 마지막에 f32를 한 번 더 돌리는 덕분에, 프로세스가 어느 dtype을 먼저 컴파일했든 가드가 성립한다.

이 테스트에는 Metal에서는 실패할 수 없다는 주석을 명시했다. 개발자 본인의 장비에서 통과해도 아무 정보를 주지 않는 가드라면 그 사실을 스스로 밝혀야지, 나중에 읽는 사람이 검증됐다고 믿게 두어서는 안 된다.

---

## 5. 검증 공백

이슈의 인수 기준인 `cargo test --release --features cuda -p mlxcel-core --lib paged_v2 -- --test-threads=1` 통과는 **이 PR에서 충족되지 않았다**. 구현 호스트에 `nvcc`가 없고 도달 가능한 CUDA 노드도 없다. 2026-08-07 기준 `rexy.office.lablup`과 `indominus.office.lablup` 둘 다 bastion 이름 해석에 실패했다.

대신 확인한 것은 다음과 같다.

- 변경이 컴파일되고 Metal 스위트가 초록을 유지한다. paged_v2 93개 통과, 0 실패(가드 추가로 92에서 증가).
- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings`, `scripts/ci/check_cross_repo_refs.py` 모두 통과.
- `cuda_kernel`에 `Dtype` 템플릿 인자를 넘기는 건 `turbo/fused_norm.cpp`와 `turbo/fused_rope_append.cpp`가 이미 CUDA에서 쓰고 있으므로 새 경로가 아니다.

GB10 세션이 위 명령을 돌려 새 가드와 `sparse_tests::an_f16_cache_matches_the_reference_within_its_own_precision`를 함께 확인해야 한다.

---

## 6. 교훈

- **상대오차 1.0 근처는 크기가 아니라 범주 신호다.** 출력이 입력과 아무 관계가 없다는 뜻이므로 산술이 아니라 주소 계산·타이핑·디스패치를 가리킨다. 이걸 "아주 심한 반올림"으로 읽었다면 조사가 커널 본문으로 빠졌을 것이다.
- **통과한 단언이 실패에서 가장 많은 정보를 줄 때가 있다.** `outcome.is_fused()`가 통과했다는 사실이 폴백 경로를 지우고 결함을 커널로 좁혔고, 그래서 빌드 시스템 쪽 설명에 닿을 수 있었다.
- **의존성의 백엔드 비대칭은 옳은 쪽에서는 보이지 않는다.** mlxcel 트리만 놓고 읽으면 잘못된 곳이 없다. 결함은 CUDA가 Metal과 다르게 하는 부분에 상대적으로만 존재한다. API만이 아니라 의존성의 소스를 읽어야 하는 이유다.
