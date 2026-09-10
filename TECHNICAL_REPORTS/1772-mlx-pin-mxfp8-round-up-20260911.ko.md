# 기술 리포트: PR #1772 - fix(mlx): bump MLX pin to 81ba1c6a so mxfp8 block scales round up

**날짜**: 2026-09-11
**작성자**: mlxcel maintainers
**리뷰어**: 구현 리뷰 사이클(구현 리뷰, 보안·성능 리뷰, 마무리)
**상태**: Metal(M1 Ultra)에서 완료. CUDA는 CI에서 컴파일·링크 확인(GB10, sm_121); CUDA 테스트나 추론은 돌리지 않음. M5 Max 미측정.
**언어**: C++, Rust, CMake, YAML
**위험도**: 중상(MLX가 소유한 모든 커널과 호스트 경로가 업스트림 99커밋만큼 움직임; 한 GPU 세대에서만 측정)

---

## 요약

이슈 #1769는 `fp8_block_requantize_matches_direct_path`가 macOS 27로 올린 M5 Max에서 결정적으로 실패한다고 보고하며, 툴체인이나 GPU 세대를 의심했다. 이슈가 스스로 정한 판정 기준을 역시 macOS 27인 M1 Ultra에서 돌리자 실패 메시지가 바이트 단위로 같았다. 이 테스트는 애초에 Metal에서 한 번도 돌아간 적이 없었다. #1742는 Linux/CUDA에서만 검증됐다.

원인은 mlxcel의 재양자화 경로도, 테스트도 아닌 MLX에 있다. 고정 커밋 `9a795735`에서 Metal과 CPU 백엔드는 mxfp8 E8M0 블록 지수를 `round(log2(amax / 448))`로 고른다. log2 공간에서 최근접으로 반올림하면 블록 절반가량에서 스케일이 `amax / 448`보다 작게 잡히고, 그 블록의 큰 값들은 E4M3 상한 448을 넘어 포화된다. 블록 최대값의 최대 `1 - 2^-1/2`(29.3%)를 잃는다. CUDA는 지수를 올림하므로 테스트를 작성한 곳에서는 통과했다. 업스트림은 핀 엿새 뒤 ml-explore/mlx#4353에서 Metal과 CPU를 고쳤다.

즉 테스트의 경계는 옳았고 양자화기가 틀렸다. 같은 양자화기가 프로덕션에서도 돈다. `requantize_block_fp8_weights`는 트리 안의 유일한 E8M0 양자화 호출자이고, Metal에서 로드한 모든 벤더 FP8 체크포인트가 이 경로를 거쳤다.

메인테이너는 이를 우회 패치하지 않고 핀을 업스트림 main `81ba1c6a`로 옮기기로 했다. 그러면서 98개 커밋이 더 들어왔고, 그중 셋은 브리지 아래에서 동작을 바꾼다. 컴파일되거나 테스트를 통과하면서도 틀리는 방식이다. 시그니처 중간에 끼어든 인자, 새 링크 의존성, throw하기 시작한 접근자다. 셋 다 이 PR에서 처리했다. 첫째는 컴파일 단계에서 드러났고, 나머지 둘은 리뷰가 찾았다.

---

## 1. 실패, 그리고 이슈의 가설이 맞지 않은 이유

이슈는 어떤 변경보다 먼저 실험 하나를 요구했다. 같은 시드 테스트를 13세대 하드웨어에서 돌려 보라는 것이다. M1 Ultra의 결과는 다음과 같다.

```
element 4: mxfp8 error 0.50390625 exceeded 0.3002931 (group max 4.8046875)
```

M5 Max의 메시지와 한 글자도 다르지 않다. 두 호스트 모두 macOS 27과 Xcode 27이었으므로, 이 실험으로 세대 차이는 배제됐지만 툴체인 차이까지 배제되지는 않는다. 툴체인을 배제한 것은 이력이다. #1742의 검증 절은 CUDA 실행만 나열하므로, 되돌아갈 macOS 26 통과 기록이 애초에 없었다.

메커니즘은 고정된 커널(`mlx/backend/metal/kernels/fp8.h`, `struct fp8_e8m0`)에 그대로 적혀 있다.

```cpp
float le = metal::log2(x);
int n = int(metal::round(le));
```

픽스처의 블록 0에서 `log2(4.8046875 / 448) = -6.54`는 `-7`로 반올림된다. 그러면 블록 최대값은 `4.8046875 * 128 = 615`로 스케일되고, E4M3 변환은 448 이상을 모두 포화시킨다. 원소 4인 `4.00390625`는 512.5로 스케일되어 포화되고 `448 / 128 = 3.5`로 복원된다. 오차는 0.50390625다. 시드 픽스처 전체에 고정 커널을 호스트에서 에뮬레이션하면 이 첫 실패를 정확히 재현하고 전체 그림도 보여 준다. 650블록 중 301블록이 포화되고, 최악 손실은 그룹 최대값의 0.2928이다. log2 공간 최근접 반올림의 이론 상한은 `1 - 2^-1/2 = 0.2929`다.

이슈는 두 호스트가 모두 실패하면 경계를 E4M3 반 스텝(`group_max / 16`)에서 한 스텝(`group_max / 8`)으로 넓히자고 제안했다. 양자화기가 0 방향으로 반올림한다는 가정이었다. 실제로는 0 방향 반올림이 아니라 포화다. 한 스텝 경계도 포화된 최대값에서 여전히 실패하고, 통과할 만큼 느슨한 경계라면 결함을 인증하는 셈이 된다. 그 선택지는 기각했다.

---

## 2. 왜 핀 범프인가

원인이 드러난 뒤 선택지는 셋이었다.

| 선택지 | 비용 | 채택 여부와 이유 |
|---|---|---|
| 테스트 경계 확대 | 없음 | 프로덕션 정확도 결함을 인증한다. 기각. |
| ml-explore/mlx#4353 백포트 오버레이 | `fp8.h`, 2,000줄짜리 `fp_quantized.h`, CPU `quantized.cpp`, `ops.cpp`(이미 CUDA 전용 오버레이)의 파일 전체 오버레이 | 줄 수로는 작지만, 들고 다니다 나중에 걷어낼 새 오버레이가 넷이고 그중 하나는 기존 오버레이 대상과 겹친다. |
| 핀을 업스트림 main으로 이동 | 업스트림 99커밋을 흡수하고 검증 | 메인테이너가 선택. 트리는 이미 업스트림 main을 추적하고, 범위 안에 이 트리가 원할 다른 수정도 있다. |

선택한 방식의 비용은 MLX가 커널을 소유한 모든 곳에서 수치가 움직인다는 것이다. 그래서 6절의 검증은 fp8 경로에 한정하지 않고 넓게 잡았다.

---

## 3. 오버레이 재조정

이 트리는 파일 전체 오버레이 28개를 `src/lib/mlx-cpp/patches/`(모든 빌드에 적용)와 `src/lib/mlx-cpp/patches-cuda/`(CUDA 빌드 전용)에 둔다. 두 핀 사이에 업스트림이 그 대상 중 일곱을 건드렸고, 일곱 개 모두 `git merge-file overlay old-upstream new-upstream`으로 3-way 병합했다. 병합이 양쪽을 보존했는지는, 각 오버레이의 새 베이스 대비 델타가 옛 베이스 대비와 같은 `+/-` 수를 갖는지로 확인한다. 업스트림의 변경은 들어왔고 우리 변경은 남았다는 뜻이다. 나머지 21개 대상은 범위 전체에서 바이트 동일이라 그대로 뒀다. 그중 둘은 업스트림에 대응 파일이 없는 mlxcel 전용이다.

| 오버레이 대상 | 업스트림 커밋 | 델타, 옛 베이스 / 새 베이스 |
|---|---:|---|
| `metal/quantized.cpp` (`MLXCEL_QMV_WIDE`) | 5 | +59/-1, +59/-1 |
| `metal/kernels/utils.h` | 2 | +5/-4, +5/-4 |
| `metal/compiled.cpp` | 1 | 혼합 dtype 캐스트만 남도록 축소(아래) |
| `cuda/device/binary_ops.cuh` | 1 | +53/-0, +53/-0 |
| `cuda/quantized/quantized.cpp` | 1 | +358/-82, +361/-82 (동기화 주석만) |
| `patches-cuda/fast.cpp` | 2 | +33/-6, +33/-6 |
| `patches-cuda/ops.cpp` | 8 | +46/-11, +46/-11 |

유일한 충돌은 외형상의 것이었다. 업스트림 ml-explore/mlx#4353이 저작권 연도를 올린 헤더 줄에서 `ops.cpp` 오버레이가 자기 주석 블록을 시작한다.

병합 중 둘은 줄 수 이상의 의미가 있다. `metal/compiled.cpp`는 이제 퓨즈드 `AsType`에 `cast_to<>`를 내보낸다(ml-explore/mlx#4351). 이 함수는 새 `kernels/utils.h`에 정의돼 있다. `utils.h` 오버레이를 옛 베이스에 두고 `compiled.cpp`만 새 것을 받았다면, 캐스트를 포함한 모든 JIT 퓨즈드 커널이 빌드 때가 아니라 런타임에 컴파일에 실패했을 것이다. `metal/quantized.cpp`는 #1187의 `qmv_wide` 끄기 스위치를 담는다. 스위치가 거는 조건 `mode != "affine" || gen >= 15`는 업스트림에서 바뀌지 않았고, 유일한 호출자도 그대로라 스위치는 정확히 이전과 같은 것을 건다.

리뷰는 이 재조정 방법으로는 볼 수 없는 것을 하나 찾았다. 옛 베이스에서 이미 틀려 있던 오버레이다. `metal/compiled.cpp`는 1-D 입력에 `elem_to_loc_1<uint>`를 내보내고 있었다. 예전 동기화가 ml-explore/mlx#3720 일부를 손으로 적용하면서 이 줄을 빠뜨린 탓이다. 음수 stride가 선택하는 large-index 커널에서 `uint(stride)`는 stride -1을 약 2^32로 감싸, 버퍼 밖을 읽는다. 오늘은 트리 어디서도 닿지 않는다. 이제 이 줄은 업스트림과 같고, 오버레이의 델타는 존재 이유인 혼합 dtype 캐스트만 남았다. 헤더에 그 사실을 적고, 범프 때마다 업스트림과 비교하라고 경고해 뒀다. `binary_ops.cuh`의 CUDA 혼합 타입 `FloorDivide` 오버로드도 같은 방식으로 맞췄다. ml-explore/mlx#4108 이후 업스트림 float 분기처럼 floor를 쓴다.

---

## 4. 새 핀이 브리지 아래에서 바꾸는 것

### 4.1 시그니처 중간에 끼어든 인자

ml-explore/mlx#4458은 `mlx::core::gather_qmm`의 `mode`와 `sorted_indices` 사이에 `const std::optional<array>& global_scale`을 넣었다. 브리지 호출 13곳 모두 `sorted_indices`를 위치 인자로 넘긴다. 이 계열의 위험은 조용한 재바인딩이다. bool이 새 인자에 바인딩되고 그대로 컴파일되는 경우다. 여기서는 그런 일이 없다. `array`의 스칼라 생성자가 `explicit`이라 `bool`이 `std::optional<array>`로 암묵 변환될 길이 없고, 옛 호출은 컴파일에 실패한다. 작은 파서로 각 호출에 `/* global_scale = */ std::nullopt`를 넣었고, 인자가 정확히 11개가 아닌 호출은 건드리지 않게 했다. `fast::scaled_dot_product_attention`도 스트림 앞에 `force_fused` 인자를 얻었지만(ml-explore/mlx#4185), 트리의 어떤 호출도 스트림을 넘기지 않으므로 재바인딩이 없다.

### 4.2 새 링크 의존성

ml-explore/mlx#4208은 Cholesky를 cuSOLVER로 옮겼고, 새 핀의 `gpu::init()`은 CUDA를 시작할 때마다 cuSOLVER 핸들 캐시를 만든다. MLX의 CMake는 `CUDA::cusolver`를 PRIVATE로 링크해 cargo가 볼 수 없는데, `mlxcel-core/build.rs`의 `link_cuda()`에는 이것이 없었다. 따라서 모든 `--features cuda` 링크가 `cusolverDnCreate`에서 실패했을 것이다. 이제 `link_cuda()`가 이것을 명시한다. cuSOLVER가 빠진 최소 CUDA 런타임 설치에서는 이제 실행 시점에 실패하므로, `docs/installation.md`에 사전 빌드 CUDA 바이너리가 링크하는 공유 라이브러리 목록을 적었다.

이것이 CI가 아니라 리뷰에서 발견된 것은 CI 경로 필터의 구멍 때문이다. CUDA 바이너리를 실제로 링크하는 유일한 PR 잡 `xla-link`는 링크 레시피의 IREE 쪽 변경에만 반응했고, `mlxcel-core/build.rs`나 MLX 핀에는 반응하지 않았다. 이 PR에서 돈 CUDA 잡은 `cargo check` 하나였고, 이 명령은 링크를 하지 않는다. 이제 `ci.yml`이 두 경로 모두에서 `xla-link`를 트리거하며, 이 변경을 검증한 것도 그 잡이다(7절).

### 4.3 throw하기 시작한 접근자

ml-explore/mlx#3742는 `array::detach_event()`가 `Event::check_error()`를 부르게 바꿨다. 그래서 명령 버퍼가 실패한 실행에서 `array::is_available()`이 이제 throw하고, throw하면서 오류를 지운다. 실패한 명령 버퍼가 신호하는 모든 이벤트는 같은 인코더 `Error`를 가리킨다. 옛 핀에서는 신호된 이벤트에 대한 `is_available()`이 true를 반환하고 오류를 조용히 버렸다.

rejection 샘플러는 옛 동작에 기대고 있었다. `drain_pending_verification()`은 이전 실행들의 수렴 플래그를 들여다보는데, 거기엔 다른 요청이 남긴 실행도 포함된다. 이때 `is_available()`을 비차단 상태 조회로 썼다. 이 함수는 `Result` 브리지가 아닌 `fused_sample` 안에서 돈다. GPU 장애 뒤라면 어느 요청이든 다음 샘플링 호출에서 cxx를 통해 throw해 서버를 종료시켰을 것이다. 게다가 그 실행을 소유한 요청이 스스로 보고해야 할 오류까지 먼저 소비했을 것이다.

이제 드레인은 `stashed_launch_state()`에 묻는다. 이 함수는 배열 상태, 이벤트 신호 여부, 이벤트 오류 포인터에 살아 있는 메시지가 있는지를 읽기만 하고, 확인하면서 지우는 어떤 것도 부르지 않는다. 실패한 실행은 읽지 않고 버린다. Metal 완료 핸들러와 CPU 스케줄러는 모두 신호 전에 이벤트 오류를 저장하고 CUDA는 오류를 달지 않으므로, 신호가 보일 때쯤이면 오류도 보인다.

회귀 테스트 `a_failed_stashed_launch_is_dropped_without_throwing_or_consuming_its_error`는 MLX 공개 `Event`/`Error` API로 그 상태를 만든다. 실제 이벤트를 신호한 뒤 합성 오류를 붙이고, 그것을 단 실행을 넣고, 드레인한 다음, 아무것도 throw하지 않았는지, 오류가 살아 있는지, 슬롯이 비었는지를 단언한다. 드레인을 `is_available()`로 되돌리면 테스트 바이너리가 `terminating due to uncaught exception of type std::runtime_error`로 중단된다. 막으려던 실패 그 자체다.

### 4.4 확인했지만 트리에 닿지 않는 것

`StreamContext`는 이제 다른 스레드에서 파괴되면 throw한다(ml-explore/mlx#4462). 트리에서 이를 쓰는 곳은 없다. GGUF 경계 검사와 지연 소스 `save` 수정(ml-explore/mlx#4212, ml-explore/mlx#4378, ml-explore/mlx#4434)은 mlxcel이 부르지 않는 로더에 해당한다. `get_array_buffer_size`와 `fast::cross_entropy`는 추가 기능이다.

---

## 5. 테스트

바이트 동일성 테스트 `fp8_block_requantize_matches_direct_path`는 이제 이름과 문서 주석이 주장하는 바이트 동일성만 단언한다. 정확도 검사는 `fp8_block_requantize_round_trip_stays_within_half_an_e4m3_step`으로 옮겼다. 시드(`0x51D3_9E11`)와 130x160 형상은 그대로다. 패딩된 꼬리 블록도 경계로 묶이는 대상이기 때문이다. 문서 주석에 유도를 적었다. 스케일을 올림하면 블록 최대값은 `(224, 448]`로 스케일되고, 모든 원소가 최근접으로 반올림된다. 최상위 binade의 원소는 256 이상 단위 중 최대 16, 즉 최대값의 `2^-4`만큼 움직이고, 아래 binade는 그보다 덜 움직인다.

경계가 기대는 성질도 직접 검사하게 했다. 블록마다 E8M0 바이트를 복원해 `group_max <= 448 * scale`을 단언한다. 옛 핀에서는 이것이 먼저 실패하며 원인을 이름으로 말한다.

```
block 0: E8M0 scale 2^-7 is below amax / 448 = 0.0107247485, so its maximum 4.8046875 scales to 615 and saturates at 448 (the scale was rounded down, see ml-explore/mlx#4353)
```

이 비교는 이 픽스처에서 근사가 아니라 정확하다. 모든 픽스처 값은 E4M3 복원값에 bf16 스케일을 곱한 것이라 유효 비트가 12개 이하다. 따라서 `amax / 448`은 2의 거듭제곱과 같지 않은 한 f32 반올림 거리 안으로 들어올 수 없다.

---

## 6. 검증

모든 실행은 M1 Ultra(13세대), macOS 27.0, Xcode 27.0에서 했다. 옛 핀 arm은 `ca00c467`에서 빌드해 자기 `mlx.metallib`와 함께 격리했다. 바이너리는 metallib를 빌드 디렉터리 절대경로로 찾기 때문에, 그 사본이 없었다면 재빌드가 덮어쓴 뒤 옛 바이너리가 새 커널을 읽었을 것이다.

### 6.1 fp8 왕복

| | 옛 핀 | 새 핀 |
|---|---|---|
| 새 테스트 | 블록 0에서 실패(위) | 통과 |
| 포화 블록(650 중) | 301(호스트 에뮬레이션) | 0(장치) |
| 최악 오차 / 그룹 최대값 | 0.2928(호스트 에뮬레이션) | 0.0489, 곧 19.796875에서 0.96875(장치) |

새 핀의 장치 수치는 호스트 에뮬레이션의 올림 규칙 수치와 출력된 모든 자릿수에서 같다.

### 6.2 Turbo 런처

`sparse_v_kernel_threshold_zero_matches_graph` 통과. `delegated_fused_kernel_matches_reference_over_200_steps`는 최대 RMS 1.7263e-4, `delegated_steel_envelope_matches_cold_only_fused_over_200_steps`는 1.5259e-4로, 계약 5e-3 안이다.

### 6.3 teacher-forced logit trace

wikitext-2에서 `examples/logit_trace`를 옛 핀과 새 핀으로 떠 `scripts/compare_logit_traces.py`로 비교했다. 폭 1(128위치), 512토큰 문맥 뒤의 폭 8(640위치), 폭 256(512위치)에서 잰 결과다.

| 체크포인트 | top-1 불일치 (w1 / w8 / w256) | decided 위치 불일치 |
|---|---|---|
| qwen2.5-7b-instruct-4bit | 0 / 3 / 1 | 모든 폭에서 0 |
| gemma-3-4b-it-4bit | 0 / 0 / 0, 효과상 바이트 동일 | 0 |
| qwen3-30b-a3b-4bit | 4 / 14 / 15 | 모든 폭에서 0 |
| nvidia-nemotron-3-nano-30b-a3b-4bit | 4 / 21 / 8 | 모든 폭에서 0 |
| gemma-4-26b-a4b-it-4bit | 0 / 0 / 0, 효과상 바이트 동일 | 0 |

Gemma 계열은 GELU를 써서 업스트림이 `Sigmoid`를 `precise::exp`로 바꾼 변경(ml-explore/mlx#4461)을 타지 않는다. SiLU 계열은 이를 타고 반올림 계층 안에서 움직인다. qwen3-30b-a3b가 가장 많이 움직인 것은 MoE 라우팅이 라우터 로짓의 마지막 ulp 변화를 다른 전문가 선택으로 바꾸기 때문이다. 폭 256의 4,096위치에서 decided 위치 1,720개 중 1개가 불일치했고, 그것도 참조의 2순위였으며, perplexity는 -0.20% 움직였다.

### 6.4 짧은 실행이 닿지 못한 분기

575토큰 프롬프트는 디스패치 변경 셋에 닿지 않아, 각각을 직접 몰았다. greedy로 arm을 교대했고, 조용한 머신에서 3회 중앙값이다.

| 분기 | 체크포인트, 문맥 | 결과 |
|---|---|---|
| 키 1,024개 이상의 head-dim-512 벡터 디코드 | gemma-4-26b-a4b, 2,396토큰 | 옛 핀, 새 핀, `MLX_SDPA_D512_MIN_KL`로 커널을 끈 새 핀의 출력 텍스트 동일; 71.7 / 71.6 / 71.4 tok/s |
| 키 8,192개 이상의 GQA-8 2-pass 디코드(ml-explore/mlx#4077) | qwen3-30b-a3b, 8,819토큰 | 텍스트 동일; 디코드 55.6에서 60.8 tok/s(+9.4%), prefill 불변 |
| 비전 타워의 head-dim-72 fused 어텐션(ml-explore/mlx#4330) | 이미지 입력 gemma-3-4b, gemma-4-26b | gemma-3-4b 이미지 prefill 239에서 279 tok/s; 설명은 20~50토큰 뒤 똑같이 충실한 다른 문장으로 갈라짐; 답이 정해진 질문(픽스처 사각형의 색, gemma-4-26b의 막대 6개)의 답은 불변 |

### 6.5 짧은 문맥 처리량

`mlxcel generate --profile`, 575토큰 프롬프트, 128토큰, 3회 중앙값. qwen2.5-7b prefill 681에서 677, decode 100.3에서 100.4 tok/s; qwen3-30b-a3b 582에서 586, 76.1에서 75.8; gemma-4-26b-a4b 713에서 709, 70.9에서 70.8. 모두 0.6% 이내다.

### 6.6 게이트

최종 헤드에서 워크스페이스 게이트(`cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1`)는 122개 바이너리, 10,986 통과, 0 실패, 359 무시다. 이슈의 10,981 통과에 #1770이 고친 둘, 이 이슈의 하나, 새 테스트 둘을 더한 값이다. `-D warnings` 워크스페이스 clippy, `cargo fmt --check`, 두 핀 파서, 교차 저장소 참조 검사 모두 깨끗하다.

---

## 7. 검증하지 못한 것

- **CUDA 실행.** 이 PR의 필터 수정으로 트리거된 CI `OpenXLA feature link` 잡이 GB10(sm_121)에서 새 핀으로 `mlxcel-core`를 빌드하고 `--features cuda,xla-iree` 릴리즈 테스트 바이너리를 12분 19초에 링크했다. 병합된 CUDA 오버레이가 컴파일되고 cuSOLVER 링크가 풀린다는 뜻이다. CUDA 테스트나 추론은 돌리지 않았다. 초록인 `CUDA sm_70 compile` 체크는 스킵이다. 그 러너의 CUDA 13.0이 sm_70을 대상으로 삼을 수 없어 아무것도 컴파일하지 않는다.
- **M5 Max.** quantize 커널에는 세대별 분기가 없고, 이슈의 M5 실패 메시지가 M1 Ultra와 같다. 그러나 이 범위에서 바뀐 NAX 경로는 17세대에서만 돌고 측정하지 않았다(ml-explore/mlx#4171, ml-explore/mlx#4352, ml-explore/mlx#4392).
- **Metal에서의 실제 FP8 체크포인트.** 이 호스트에는 없다. 측정 수단은 픽스처이며, 그 위에서 양자화기의 동작은 이제 정확히 올림 규칙이다.
- **VLM 범위.** 체크포인트 둘, 합성 이미지 하나, 단색 픽스처 하나.

---

## 8. 후속 과제

- **`array_evaluated_bytes`도 대기하는 비 `Result` 브리지 함수다.** `array::eval()`을 거치므로 새 핀에서는 실패한 실행에 대해 이벤트가 도착한 뒤에도 throw한다. 옛 핀은 도착 전에만 throw했다. 서버의 lookahead 토큰 읽기가 이를 쓰므로, 거기서 GPU 장애가 나면 여전히 프로세스가 종료된다. `Result`로 선언하고 오류를 스케줄러의 스텝 실패 처리로 넘겨야 하는데, 그 자체로 별도의 변경이다.
- **`metal/compiled.cpp`의 혼합 dtype 캐스트는 비교 연산의 입력도 캐스트한다.** 비교의 출력 dtype은 `bool`이므로, float 입력을 비교 전에 `bool`로 캐스트하게 된다. 오늘 트리의 컴파일 함수 중 비교를 담은 것은 없어 잠재 결함이다. 그런 함수가 생기기 전에 막아야 한다.
- **NemotronH 로더가 진행 메시지를 stdout에 찍는다.** stdout으로 쓰는 `examples/logit_trace`의 TSV를 오염시키고, `[NemotronH]` 줄을 걸러내기 전까지 `compare_logit_traces.py`가 파일을 거부한다.

---

## 9. 학습 포인트

- **이슈의 판정 기준을 먼저 돌리고, 그 틀을 받아들이는 건 그다음이다.** 이슈는 M1 Ultra가 통과하거나(백엔드 발산), 애초에 성립하지 않던 경계로 실패하리라 봤다. 결과는 똑같은 실패였고, 이유는 둘 다 아니었다. 올림을 쓰는 단 하나의 백엔드에서만 돌던 테스트 뒤에, 핀 이후 업스트림에서 고쳐진 양자화기가 있었다.
- **초록 체크는 그것이 실제로 돌린 것만 증명한다.** 이 PR에는 CUDA를 다룬 것처럼 보이지만 실제로는 아니었던 체크가 셋 있었다. `cargo check`는 링크하지 않고, sm_70 잡은 스킵했고, 링크 잡은 트리거되지 않았다. 빠진 라이브러리는 업스트림 diff를 읽어서 찾았다.
- **핀 범프는 시그니처만이 아니라 의미를 깬다.** 시그니처 변경은 컴파일에 실패했다. 링크 변경과 접근자 변경은 컴파일도 되고 Metal 게이트도 통과한 뒤 프로덕션에서 실패했을 것이다. 헤더만이 아니라 업스트림의 접근자와 오류 경로 변경까지 읽어야 한다.
- **핀 A/B의 각 arm은 런타임 산출물까지 격리한다.** metallib를 절대경로로 찾는 바이너리는 arm의 절반일 뿐이다.
