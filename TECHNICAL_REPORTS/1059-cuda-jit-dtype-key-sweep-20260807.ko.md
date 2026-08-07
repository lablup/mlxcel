# 기술 보고서: PR #1059 - fix(kernels): key every CUDA JIT launch on its input dtypes

**날짜**: 2026-08-07
**작성**: mlxcel 메인테이너
**검토**: 구현 및 보안 리뷰 사이클
**상태**: 부분 완료 (결함 부류 전체를 쓸고 가드를 세웠으나, 이슈 #1054가 요구한 최소 순서쌍은 제공하지 못했고 추측으로 채우지 않았다)
**언어**: C++, Python, YAML, Markdown
**위험도**: 높음 (쓸어낸 세 곳 중 하나가 프로덕션 샘플링에서 도달 가능, CUDA 한정)

---

## 요약

PR #1058이 JIT 캐시 키에서 입력 dtype이 빠진 paged decode v2 커널 두 개를 고쳤다. 이 PR은 같은 부류의 나머지를 쓸어낸다. `cuda_kernel` 런치 세 곳이 같은 결함을 갖고 있었고, 그중 하나가 샘플러다. `gumbel_max_sample_accepts`가 단일 `NumSplits`에서 f32·f16·bf16을 모두 받는다. 아울러 `scripts/ci/check_kernel_dtype_keys.py`를 추가해 `make verify`와 신규 CI 잡에 연결했다. 새 호출부가 같은 누락을 조용히 반복할 수 없게 하려는 것이다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1054는 `paged_v2::launch::launch_tests::chunk_size_does_not_change_the_answer`가 GB10의 전체 `mlxcel-core` lib 스위트에서만 약 55896배 오차로 실패하고, 단독 실행과 92개짜리 `paged_v2` 범위 실행에서는 통과한다고 보고했다. 본문이 누출된 프로세스 상태 후보로 세 가지를 지목했다. autotune 메모, "테스트를 가로질러 충돌할 만큼 느슨하게 키잡힌" 커널·그래프 캐시, 그리고 환경 변수다.

두 번째가 맞았고 PR #1058이 기전을 확정했다. `mlx/backend/cuda/custom_kernel.cpp`는 커널 이름을 `"custom_kernel_" + name + template_arguments_hash(template_args)`로 짓는데 버퍼 파라미터 타입은 런타임 입력 dtype에서 생성하고, `cu::get_jit_module`이 컴파일된 모듈을 그 이름으로 프로세스 전역 맵에 메모이즈한다. Metal은 키에 dtype을 넣고 CUDA는 넣지 않는다.

### 1.2 기존 문제

- **결함 부류가 보고된 증상보다 넓었다.** `cuda_kernel(` 호출을 포함한 파일의 모든 `template_args` 초기화를 감사하니 정수만으로 키를 잡은 곳이 세 군데 더 있었다.
- **그중 하나는 테스트 산물이 아니라 프로덕션 샘플링이다.** `gumbel_max_sample_accepts`(`turbo/sampling.cpp`)는 `float32`·`float16`·`bfloat16`을 명시적으로 받고 `NumSplits`는 `(batch, vocab)`에만 의존한다. 어휘 크기가 같고 로짓 dtype이 다른 두 모델이 CUDA에서 컴파일된 커널 하나를 공유했고, 나중 것은 잘못된 포인터 타입으로 읽은 버퍼에서 샘플링했다.
- **`rejection_sample_accepts`는 노출이 더 넓다.** dtype 제한이 아예 없어서 어떤 부동소수점 타입이든 커널에 도달한다.
- **다음 발생을 막을 장치가 없었다.** 이 수정은 호출부마다 반복되는 관례이고, 검사 없는 관례는 언젠가 낡는다.

### 1.3 위험 평가

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| 어휘 크기가 같고 로짓 dtype이 다른 모델 사이를 오가는 CUDA 서버가 쓰레기에서 샘플링 | 치명 | 조건 충족 시 확실 |
| 향후 `cuda_kernel` 호출부가 같은 누락을 반복 | 높음 | 검사가 없으면 높음 |
| Metal 전용 런처에 CUDA 포팅이 붙으면서 결함을 물려받음 | 중간 | 중간이고, 갱신되지 않은 allowlist에는 보이지 않음 |

---

## 2. 기술 검토

### 2.1 감사 결과

`src/lib/mlx-cpp/turbo/`와 `src/lib/mlxcel-core/cpp/` 아래의 모든 `std::vector<std::pair<std::string, TemplateArg>>` 초기화를 열거해 분류했다.

| 지점 | 기존 키 | 판정 |
|---|---|---|
| `turbo/paged_attention.cpp` (v1) | `Dim`, `NRep`, `DimsPerThread`, `NumSplits` | **여기서 수정**: `QType`, `KVType`, `VType` 추가 |
| `turbo/sampling.cpp` | `TgSize`, `NumSplits` | **여기서 수정**: `LogitsType` 추가 |
| `turbo/sampling_rejection.cpp` | `TgSize`, `MaxRounds` | **여기서 수정**: `ProbsType`, `ParamsType` 추가 |
| `turbo/fused_norm.cpp` | `T`, `TW`, `Dim`, `Threads` | 이미 정상 |
| `turbo/fused_rope_append.cpp` | `T`, `HeadDim`, … | 이미 정상 |
| `cpp/mlx_cxx_kernels.cpp` (7곳) | 각각 `T` 또는 `InT` 포함 | 이미 정상 |
| `turbo/sparse_v_sdpa.cpp` | `Dim`, `RepeatCount`, `NRep` | 범위 밖: Metal 전용, `cuda_kernel(` 없음 |
| `turbo/turbo4_delegated_sdpa.cpp` (3곳) | `Dim` (+ `RepeatCount`, `NRep`) | 범위 밖: Metal 전용 |

범위 밖 네 곳은 오늘 기준으로 실제로 안전하다. Metal 캐시 키가 이미 dtype을 담기 때문이다. 다만 구조적으로 안전한 건 아니고, 그 점이 검사 규칙의 모양을 결정했다.

### 2.2 보안

샘플러 발견이 보안 관점에서 의미가 있다. 다만 기밀성이 아니라 정확성 문제다. 로짓 버퍼를 잘못된 타입으로 읽으면 재해석된 비트 위에서 argmax가 계산되므로, 상태 코드와 지연 지표는 정상인 채로 서빙되는 완성문만 잡음에서 뽑힌다. 메모리 안전 경계는 넘지 않는다. 버퍼는 바이트 크기가 맞고 커널도 그 안에서 읽으므로 범위 밖 접근이 아니라 조용한 품질 실패다.

### 2.3 성능

커널 본문은 바뀌지 않았다. CUDA에서 해당 런치들이 geometry당 하나 대신 (geometry, dtype)당 하나의 모듈을 컴파일하는데, Metal과 같은 방식이다. dtype 하나를 서빙하는 프로세스라면 정상 상태는 그대로다.

### 2.4 호환성과 의존성

- **파괴적 변경**: 없음.
- **신규 CI 잡**: `kernel dtype keys`. `crate-versions`와 마찬가지로 `changes` 경로 필터 뒤에 두지 않았다. 툴체인이 필요 없고 몇 초면 끝나며, 건너뛸 수 있는 게이트는 그것을 깨뜨리는 바로 그 PR에서 건너뛰어진다.
- **신규 make 타깃**: `verify-kernel-dtype-keys`, `verify`에 포함.

---

## 3. 기술적 결정

### 3.1 검사 범위를 어떻게 정할 것인가

**맥락:** 검사는 CUDA 런치에 적용되고 Metal 전용 런처에는 적용되지 않아야 한다. 그러지 않으면 즉시 오탐 네 건이 난다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `check_crate_versions.py`가 독립 크레이트에 하듯, Metal 전용 네 곳을 이유와 함께 allowlist | 명시적이고 면제 사유가 옆에 기록됨 | 정확히 중요한 순간에 낡는다. allowlist된 파일에 CUDA 포팅이 붙어도 면제가 유지되어 결함이 조용히 되살아난다 |
| **선택: 파일에 `cuda_kernel(`이 있는지로 범위 결정** | 스스로 유지된다. Metal 전용 런처는 누군가 CUDA 포팅을 붙이는 순간부터 자동으로 검사 대상이 된다 | 다소 간접적이라 독자가 왜 파일 단위 술어가 옳은지 알아야 하므로, 스크립트가 그 이유를 길게 적어 둔다 |

**근거:** 막으려는 실패 형태는 *새* 호출부가 누락을 반복하는 것이고, 그 가장 유력한 모양이 기존 Metal 런처에 CUDA 포팅이 붙는 경우다. allowlist는 바로 그 경우에 눈이 먼다. `CLAUDE.md`는 크레이트 버전 규칙의 산문 판본이 같은 이유로 이미 한 번 실패했다고 기록하고 있고, 그래서 `check_crate_versions.py`가 규칙을 역전시켰다. 이번에는 목록 자체를 없애 역전을 한 걸음 더 밀었다.

**절충:** 술어가 호출 단위가 아니라 파일 단위다. CUDA 런치와 진짜 dtype 무관 런치가 한 파일에 섞이면 후자도 dtype 인자를 달아야 한다. 그런 파일은 현재 없고, 생기더라도 비용은 잉여 템플릿 인자 하나다.

---

## 4. 구현 상세

`scripts/ci/check_kernel_dtype_keys.py`는 `src/lib/mlx-cpp/turbo`와 `src/lib/mlxcel-core/cpp`를 훑는다. `cuda_kernel(`을 포함한 `.cpp`마다 모든 `template_args` 초기화를 뽑아, 값에 `.dtype()`이 직접 들어 있거나 파일 앞쪽에서 `.dtype()`으로 묶인 지역 변수를 가리키는 항목이 최소 하나 있을 것을 요구한다. 두 번째 형태가 필요한 이유는 MoE 커널이 `auto T = x.inner.dtype();`을 쓰고 나서 `{"T", T}`로 넘기기 때문이다.

실패하면 파일·행·변수명과 실제로 발견한 키를 함께 찍고, 기전을 설명한 뒤 이슈 #1053·#1054를 가리킨다.

### 4.1 음성 대조

실패시킬 수 없는 검사는 검사가 아니다. `turbo/sampling.cpp`에서 `{"LogitsType", logits.dtype()}`을 빼고 스크립트를 다시 돌렸다.

```
kernel-dtype-keys: FAIL
  src/lib/mlx-cpp/turbo/sampling.cpp:430: `template_args` names no input dtype; keys are ['TgSize', 'NumSplits']
```

줄을 복원하니 `OK — 13 source files scanned`로 돌아왔다. 이 절차를 기록해 두는 이유는, 빨간불이 드는 걸 한 번도 본 적 없는 초록 검사를 출하하는 것과 정규식이 깨진 검사를 출하하는 것이 겉으로 구분되지 않기 때문이다.

---

## 5. 이 PR이 제공하지 못한 것

이슈 #1054의 첫 인수 기준은 실패를 재현하는 **최소 순서쌍**을 요구한다. 그건 여기 없다.

- 이분 탐색은 CUDA에서 스위트를 돌려야 하는데, 구현 호스트에 `nvcc`가 없고 도달 가능한 CUDA 노드도 없다(2026-08-07 재확인).
- 분석적으로 좁히는 것도 수렴하지 않았다. `mlxcel-core` 안에서 v2 partial 커널에 도달하는 건 `paged_v2` 테스트뿐이고, 그 템플릿 인자 조합은 dtype을 가로질러 겹치지 않는다. `launch_tests`는 `Dim` 64·128에서 `PageSize=32`, `sparse_tests`는 pool을 reshape해 `PageSize=1`, `cascade_launch_tests`는 `Dim=16, PageSize=8`이다.
- merge 커널이 유망한 실마리였다. `pages_per_chunk 1`이 루프의 첫 값이자 merge가 도는 유일한 값이기 때문이다. 그런데 `paged_v2` 밖의 유일한 호출자인 MLA `split_kv`는 테스트가 `DIM=5`를 쓰고 프로덕션 경로는 `split_kv.rs:278`에서 f32로 캐스팅한 뒤 merge한다. **확인이 아니라 기각이다.**

Metal에서는 이 질문 자체가 보이지 않는다. 전체 스위트가 1416 통과 / 0 실패이고 `chunk_size_does_not_change_the_answer`도 포함된다.

GB10 세션이 `main`에서 `cargo test --release --features cuda -p mlxcel-core --lib -- --test-threads=1`을 돌려 실패가 사라졌는지 보고해야 한다. #1058이 partial과 merge 커널의 키를 모두 다시 잡았으니 없어졌을 법하지만, "법하다"가 정직한 표현이다.

---

## 6. 부수 발견

직렬 스위트 한 번의 실행에서 autotune 테스트 2건이 실패했는데, 동일 조건 재실행은 1416 / 0으로 통과했다. 회귀가 아니라 호스트 부하 플레이크다.

- 해당 모듈이 스스로 "Every test here is CPU-only: the `FakeOp` double implements `TunableOp` by sleeping for a per-tactic duration and overrides `sync` to a no-op"이라고 적어 두었으므로, C++ 커널 템플릿 인자가 닿을 수 없다.
- 단독 실행에서는 55 / 0으로 통과한다.
- sleep이 `Duration::from_micros`로 800 대 400인데 호스트 부하 평균이 40이었다.

`autotune_tests.rs:153`은 이미 이 테스트들이 "depend on how accurately the host honors `thread::sleep`"라고 경고하고 있다. 별도 이슈로 올릴 만하다. 마이크로초 단위 sleep 정확도에 기대는 단언은 CI를 포함한 어떤 부하 상태의 러너에서도 뒤집힐 수 있다.

---

## 7. 교훈

- **인스턴스가 우연히 발견됐다면 부류를 고쳐야 한다.** #1053이 드러난 건 한 테스트가 마침 f32를 f16보다 먼저 돌렸기 때문이다. 샘플러 노출을 드러내 줄 장치는 아무것도 없었고, 둘 중에서는 그쪽이 더 심각하다.
- **목록이 낡는 경우가 곧 위험한 경우라면 목록 대신 술어를 쓴다.** allowlist는 작성된 날에는 옳고 누군가 Metal 커널을 CUDA로 포팅한 날에 틀렸을 것이다. 그날이 바로 이 검사가 존재하는 이유다.
- **가장 그럴듯한 자기 가설을 글로 기각해 본다.** MLA merge 실마리는 사실처럼 단언해도 될 만큼 그럴듯했다. `DIM=5`와 f32 캐스팅을 확인하는 데 2분이 걸렸고, 확신에 찬 오답이 정직한 미해결 질문으로 바뀌었다.
