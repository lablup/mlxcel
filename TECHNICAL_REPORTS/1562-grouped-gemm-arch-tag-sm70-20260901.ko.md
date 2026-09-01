# 기술 보고서: PR #1562 - Ampere 이전 그룹 GEMM 갈래에 Sm75 대신 Sm70 태그 달기

**작성일**: 2026-09-01

**작성자**: mlxcel maintainers

**상태**: 완료. 이슈가 죽은 갈래로 의심하던 분기는 에픽이 벤치마크하는 바로 그 MoE 체크포인트에서 살아 있었고, 거기 달린 태그는 틀렸으며, 태그를 고쳐도 디바이스 코드는 한 바이트도 움직이지 않는다. 에픽 #1536이 GB10으로 미뤄 둔 수용 기준 하나를 여기서 닫았고, 이 호스트에 없는 하드웨어가 필요한 둘은 체크하지 않은 채 남겼으며, 기존 테스트 실패 하나를 드러내고 이 변경 탓이 아님을 분리해 확인했다.

---

## 요약

`patches/mlx/backend/cuda/gemms/grouped_gemm_unaligned.cu`의 `dispatch_cutlass_arch`는 compute capability 8.0 미만의 모든 디바이스를 `cutlass::arch::Sm75`로 매핑했다. 이 태그는 Turing이 도입한 `m16n8k8` MMA를 가리킨다. Tesla V100은 compute capability 7.0, 한 세대 앞이고 `8x8x4` HMMA 모양만 갖는다. 따라서 이 디스패치는 여기 도달하는 모든 Volta 부품에서 디바이스에 없는 하드웨어를 서술하고 있었고, `get_grouped_mm_funcion`도 같은 `Sm75` 자리표시자로 시작했다.

이슈 #1544는 검증 우선으로 작성됐다. 이것이 실제 결함인지, 죽은 분기인지, 무해한지 묻고, 부정적 결과도 완결된 답이라고 명시했다. 답은 셋 중 어느 것도 아니다.

**분기는 살아 있고, 에픽 #1536이 벤치마크하는 그 체크포인트에서 그렇다.** 이슈 #629의 정렬 MoE 프리필 빠른 경로는 배치가 `B >= min_rows * num_experts`를 넘기면 양자화된 `GatherQMM`을 `cutlass_grouped_gemm_unaligned`로 보낸다. `gemma-4-26b-a4b-it-4bit`에서 이 조건은 128토큰 프롬프트다. 573토큰 프롬프트의 nsys 프로파일은 `cutlass::Kernel<GemmGrouped>` 실행 180회, GPU 시간의 3.8%를 보여 준다. #1538 베이스라인은 게이트 아래인 46토큰 프롬프트로 MoE를 프로파일했고, 그래서 그 커널 표에 이 커널이 없으며 이슈가 분기를 죽은 것으로 예상한 이유도 그것이다.

**계산이 틀린 적은 없었고, 이는 추론이 아니라 측정으로 확인했다.** Ampere 이전 갈래는 `GemmConfiguration`의 기본 템플릿으로 풀리고, 그것은 `InstructionShape<1, 1, 1>`인 `OpClassSimt`다. 두 텐서 코어 특수화는 모두 `Arch::kMinComputeCapability >= 80`으로 제약되므로 Ampere 이전 태그는 어떤 모양의 MMA 원자도 고를 수 없다. 그래서 CUTLASS는 태그를 지워 버린다. 실제로 실행된 커널은 `MmaSimt` / `OpMultiplyAdd` / `MmaPipelined` 인스턴스화이고, 맹글링된 이름에 아키텍처 토큰이 아예 없다.

**태그 교체는 어느 아키텍처에서도 디바이스 코드를 움직이지 않는다.** 변경 전후로 번역 단위를 `compute_70`, `compute_80`, `compute_121`에서 컴파일하면 세 경우 모두 동일한 51개 디바이스 심볼이 나오고 본문은 바이트 단위로 같다. 심볼별로 비교한 덤프가 144 MB다. 이것이 Turing에 별도 갈래를 주지 않기로 한 근거이기도 하다.

이 변경이 내놓는 것은 정확한 서술, GPU 없이 테스트할 수 있는 결정, 단일 Ampere 이전 갈래를 정당화하는 전제에 대한 `static_assert` 둘, 그리고 그룹 GEMM이 한 번도 가진 적 없던 수치 게이트다.

## 1. 문제 정의

변경 전 디스패치:

```cpp
template <typename F>
void dispatch_cutlass_arch(cu::Device& device, F&& f) {
  if (device.compute_capability_major() < 8) {
    f(type_identity<cutlass::arch::Sm75>{});
  } else if (device.compute_capability_major() == 8) {
    f(type_identity<cutlass::arch::Sm80>{});
  } else {
    f(type_identity<cutlass::arch::Sm90>{});
  }
}
```

`< 8` 갈래는 Ampere 이전이 곧 Turing인 것처럼 쓰여 있다. 아니다. Volta는 Turing보다 한 세대 아래다. CUTLASS는 Turing MMA를 `CUTLASS_ARCH_MMA_SM75_SUPPORTED` 뒤에 두므로, sm_70 타깃으로 `Sm75` 구성을 컴파일해도 빌드가 반드시 깨지지는 않는다. 런타임에 트랩하거나 퇴화하는 경로로 컴파일될 수 있고, 그 실패 양상은 그리디 텍스트 생성으로는 드러나지 않는다.

태그를 손대기 전에 두 가지를 순서대로 확정해야 했다. 이 부품에서 분기가 실행되기는 하는지, 실행된다면 출력이 맞는지. 같은 에픽의 #1541이 선례다. `gemms/gather_gemm.cu`는 `libmlx.a`에 컴파일돼 들어가지만 아카이브 어디에도 미정의 참조가 없다. mlxcel 자체 `matmul.cpp` 오버레이가 `GatherMM::eval_gpu`를 다른 곳으로 보내기 때문이다. 번역 단위가 컴파일된다는 사실은 그것이 호출된다는 증거가 되지 않는다.

## 2. 변경 요약

파일 10개. 기능 변경은 그중 둘에 걸친 약 40줄이고, 나머지는 증거와 테스트다.

`gemms/grouped_gemm_arch.h`(신규)가 아키텍처 결정을 compute capability 주 버전의 `constexpr` 함수로 담는다. 갈래 하나가 Volta와 Turing을 함께 덮는 근거 측정치와, 그것이 더는 안전하지 않게 되는 조건도 같이 적었다. `grouped_gemm_unaligned.cu`는 그 함수로 분기하고, `get_grouped_mm_funcion`의 `Sm75` 자리표시자를 `nullptr`로 초기화된 이름 있는 `GroupedGemmFn`과 명시적 가드로 바꾸며, `static_assert` 둘을 얻는다.

`cpp/grouped_gemm_arch_probe.cpp`(신규)와 `build.rs` 한 줄이 출하되는 그 함수를 C 심을 통해 Rust에 노출한다. #1541이 세운 패턴을 따라 `cuda` 피처 뒤가 아니라 무조건 컴파일된다. `grouped_gemm_arch_tests.rs`(신규)는 매핑을 모든 아키텍처에 대해 열거하고, `grouped_gemm_numeric_tests.rs`(신규)는 디바이스에서 `gather_mm`을 `f64` 조밀 전문가별 기준과 비교한다.

`docs/benchmark_results/grouped-gemm-arch-v100-2026-08-31.md`(신규)가 전체 기록이다. #1538 베이스라인은 예약돼 있던 사후 비교 행과, 그 커널 표에 그룹 GEMM이 없는 이유를 설명하는 MoE 절 주석을 얻는다.

## 3. 기술적 선택과 이유

### 3.1 디스패치를 건드리기 전에 도달 가능성부터 확정했다

`--cuda-graph-trace=node`를 건 nsys 실행 셋. 바이너리와 체크포인트는 같고 프롬프트 길이와 킬 스위치만 다르다.

| 실행 | 프롬프트 | `B` | `cutlass::Kernel<GemmGrouped>` | `prepare_grouped_mm_data` |
|---|---|---|---|---|
| 기본 | 573토큰 | 4584 | 180회, 853.8 ms, 3.8% | 180회 |
| `MLXCEL_GATHER_QMM_GROUPED=0` | 573토큰 | 4584 | 없음 | 없음 |
| 기본 | 46토큰 | 368 | 없음 | 없음 |

세 번째 실행은 #1538 베이스라인의 MoE 프로파일을 `qmm_naive` 326회로 정확히 재현한다. 두 프로파일이 서로 다른 측정이 아니라 프롬프트 길이만 다른 같은 측정임을 보이는 교차 확인이다.

정적 확인은 #1541의 기법을 재현하면서 결과는 뒤집는다. 빌드된 `libmlx.a`에 `nm -C`를 걸면 `cutlass_gather_mm`, `cutlass_grouped_gemm_unaligned`, `cutlass_segmented_mm` 셋 모두 `matmul.cpp.o`에서 오는 미정의 참조를 갖는 반면, `gather_gemm.cu`의 `mlx::core::gather_mm(bool, bool, ...)`은 여전히 하나도 없다.

### 3.2 Ampere 이전 갈래를 하나로 두되, 취향이 아니라 측정으로 결정했다

이슈는 `< 8` 갈래를 다시 태그하는 안과 Turing을 분리하는 안을 함께 제시했다. 대조군이 결론을 낸다. `Sm75` 옆에 `Sm70` 갈래를 추가해 컴파일하면 동일한 51개 디바이스 심볼과 493,538줄짜리 바이트 동일 SASS 덤프가 나오고, 대가로 호스트 측 텍스트 심볼 26개와 오브젝트 194,704바이트가 늘어난다. Turing 전용 갈래는 같은 디바이스 코드를 내는 인스턴스화의 호스트 측 사본 하나를 더 살 뿐이다.

그래서 갈래는 하나로 남고, 태그는 그 갈래가 덮는 범위의 하한을 가리킨다. 이는 갈래 아래 구성이 SIMT인 동안에만 정당하고, 그것은 CUTLASS의 약속이 아니라 구성의 성질이므로 이제 파일이 직접 단언한다.

```cpp
static_assert(
    std::is_same_v<
        GemmConfiguration<float, cutlass::arch::Sm70, 8, true>::OpClass,
        cutlass::arch::OpClassSimt>, ...);
```

`kEnableTF32 = true` 인스턴스화가 핵심이다. `MLX_ENABLE_TF32`가 닿는 갈래이고, 이것을 단언해야 Ampere 이전 태그가 텐서 코어 특수화를 고르는 일이 배제된다. 옆에서 #1543이 `qmm_naive`에 하고 있는 작업처럼 Ampere 이전 갈래가 텐서 코어 연산자를 갖게 되는 날, 빌드가 깨지고 태그를 다시 보게 된다. Turing이 조용히 강등되는 대신에.

### 3.3 결정이 순수 함수이므로, 비회귀 주장이 GPU가 아니라 단위 테스트가 된다

태그 매핑은 정수 하나에만 의존한다. 이것을 헤더로 옮기고 C 심으로 노출하면 "`== 8`과 `> 8` 갈래는 그대로다"라는 명제를 NVIDIA 하드웨어가 전혀 없는 호스트를 포함해 어디서나 열거로 답할 수 있다. `only_the_pre_ampere_arm_changed`는 compute capability 0부터 32까지 훑으며 출하 함수를 #1544 이전 매핑의 재기술과 비교하고, 8 이상에서는 아무것도 움직이지 않음을 단언한다. 그러지 않았다면 에픽 #1536은 이 항목을 GB10 호스트로 미뤘을 것이다.

`Sm75`는 열거형에서 의도적으로 빠졌다. 여기 열거자 하나는 GEMM 전체가 인스턴스화되는 템플릿 인자이므로, 함수가 절대 반환하지 않는 태그를 적어 두면 그 갈래의 모든 인스턴스화가 죽은 사본으로 한 벌 더 생성된다.

### 3.4 기본 초기화자는 죽은 코드였고, 아키텍처 이름을 달고 있었다

`get_grouped_mm_funcion`은 `grouped_gemm_v2<GemmConfiguration<float, cutlass::arch::Sm75>>`로 시작했다. 반환되는 값이 된 적은 없다. `dispatch_float_types`는 디스패치하지 않는 모든 dtype에서 예외를 던지고, 디스패치하는 모든 dtype은 `fun`을 대입하기 때문이다. 다만 초기화자 노릇만 하려고 템플릿 인스턴스화 하나를 바이너리에 밀어 넣었고, Ampere 이전 기본값처럼 읽혔다. 지금은 `nullptr`로 초기화된 이름 있는 함수 포인터 타입에 선택이 빠져나갔을 때 던지는 가드를 붙였으므로, 그 자리에 아키텍처 이름이 아예 없다.

### 3.5 `kStages`는 Ampere 이전 태그로 구조적으로 도달 불가능하다

이슈는 `cp.async`가 없는 곳에 `static const int kStages = 3; // use SM80_CP_ASYNC`가 적용될 수 있는지 물었다. 불가능하다. 그 멤버는 `GemmConfiguration<float, cutlass::arch::Sm80, kAlignmentC, true>`, 곧 `Sm80`에 대한 명시적 완전 특수화에 속하고 다른 태그는 그것을 지명할 수 없다. Ampere 이전 갈래는 기본 템플릿의 `kStages = 2`를 받고, 프로파일된 커널은 2단 메인루프인 `MmaPipelined` 인스턴스화다. 추론으로 남기지 않고 단언한 이유는, `cp.async` 없는 3단 파이프라인이 빌드 실패이거나 조용한 직렬화인데 둘 다 스스로 알리지 않기 때문이다.

## 4. 검증

**도달 가능성**: nsys 프로파일 셋, 3.1의 표.

**종단 정확성**: `-t 0.0` 그리디, 64토큰, 게이트를 넘는 285토큰 프롬프트로 그룹 경로와 레거시 `qmm_naive` 경로를 비교. {변경 전, 변경 후} x {그룹, 레거시} 네 조합 모두 1,793자 연속 출력이 바이트 단위로 같다. 같은 짝이 이 부품에서 #629가 산 것도 측정해 준다. 프리필이 그룹 12,642.88 ms 대 레거시 28,016.23 ms로 2.22배 빠르고 출력은 동일하다.

**단위 정확성**: `grouped_gemm_numeric_tests.rs`가 `gather_mm`을 업로드된 바로 그 호스트 바이트로부터 `f64`로 누산한 조밀 전문가별 기준과 비교한다. `gemma-4-26b-a4b-it`의 실제 전문가 차원(`k = 2816, n = 704`과 전치 방향), 두 진입점, 두 `kAlignmentC` 갈래, 두 피연산자 레이아웃, f32/bf16/f16 전부를 덮는다. 전문가마다 상수 행렬을 주는 사례는 인덱스를 잘못 모으면 반올림 오차가 아니라 슬래브 전체가 통째로 움직이게 만든다. 기존 `test_gather_mm`은 출력 모양만 단언하고 값은 한 번도 보지 않았다.

**디바이스 코드 차이**: 세 타깃에서 심볼별 `cuobjdump --dump-sass` 비교.

| 타깃 | 변경 전 심볼 | 변경 후 심볼 | 한쪽에만 | 본문이 다른 것 | 비교한 SASS |
|---|---|---|---|---|---|
| `compute_70` | 51 | 51 | 0 | 0 | 58,211,476바이트 |
| `compute_80` | 51 | 51 | 0 | 0 | 35,077,812바이트 |
| `compute_121` | 51 | 51 | 0 | 0 | 50,952,400바이트 |

**처리량**: 셀당 5회 반복, 따뜻한 PTX 캐시, 매 실행 전 `nvidia-smi --query-compute-apps` 비어 있음 확인, 디코드는 `-n 40`에서 `-n 120`까지의 기울기. 4비트 디코드 29.66 → 29.73 ms/토큰, 8비트 31.68 → 32.71, 4비트 프리필 12,720 → 12,644 ms, 8비트 프리필 15,212 → 15,606 ms. 모든 차이가 변경 전 팔의 반복 산포보다 작고, 이는 바이트 동일 디바이스 코드가 허용하는 유일한 결과다.

**스위트**: `cargo test -p mlxcel-core --release --features cuda --lib -- --test-threads=1`이 1672 통과, 1 실패, 1 무시. `cargo clippy -p mlxcel-core --release --features cuda --lib --tests -- -D warnings`와 `cargo fmt -p mlxcel-core -- --check` 깨끗. `cuobjdump --list-elf libmlx.a`는 큐빈 96개, 전부 sm_70.

## 5. 검증 한계와 후속 작업

### 5.1 기존 테스트 실패 하나, 가정이 아니라 분리로 확인

`sampling::tests::temperature_one_support_unchanged`가 실패한다. `T = 1.0`에서 `fused_sample_probs`의 비트 일치를 단언하는데 64개 항목 중 5개가 1 ULP씩 어긋난다. 이 브랜치는 샘플링 코드를 건드리지 않지만 그것은 증거가 아니라 주장이므로 변경을 분리해 확인했다. `grouped_gemm_unaligned.cu`를 `c2e54939` 버전으로 되돌리고 다시 빌드해 테스트를 돌리면 바이트 단위로 같은 실패가 재현된다. `main`에 이미 있던 문제이고 이 변경 탓이 아니다. #1557이 `tests/cuda_qmm_determinism.rs`에 대해 기록한 sm_70 부동소수점 축약 비결정성과 같은 부류다. 이번이 이 에픽에서 `mlxcel-core` 라이브러리 스위트를 통째로 돌린 첫 사례이기도 하다. #1559는 좁힌 필터만 돌렸고, 그래서 드러나지 않았다. 별도 이슈가 필요하다.

### 5.2 베이스라인 열은 인용이 아니라 재측정이다

이 이슈가 예약해 둔 #1538 행은 38.61과 34.29 ms/토큰이다. 그 기록은 MoE 디코드를 움직인 #1539보다 앞서므로, 여기 변경 전 열은 이 호스트에서 이 워크트리로 다시 측정했다. 29.66 대 31.68 ms/토큰으로, MoE 짝은 베이스라인의 4비트 대 8비트 결론도 뒤집는다. #1539 자체 기록이 조밀 짝에 대해 보고한 것과 같은 방향이다.

### 5.3 #1541에서 넘겨받은 `cuFuncSetAttribute` 공백: 기록하되 고치지 않음

#1541은 `gather_gemm.cu`의 누락된 동적 공유 메모리 옵트인을 이 이슈로 넘겼다. 여기서 고치지 않기로 한 근거는 둘이다. 공백은 개별 실행 지점이 아니라 공용 인코더에 있다. `CommandEncoder::add_kernel_node_raw`는 `sharedMemBytes`를 설정하고 `cudaGraphAddKernelNode`를 호출할 뿐 `cudaFuncSetAttribute`를 어디서도 부르지 않으므로, 모든 실행 지점이 스스로 옵트인해야 한다. 다만 도달 가능한 Volta 구성은 천장 근처에도 가지 않는다. 계산이 아니라 프로파일에서 측정한 값으로, 그룹 GEMM은 49,152바이트 비옵트인 한계 대비 10,320바이트, 21%를 요구한다. `gather_gemm.cu`가 mlxcel 빌드에서 도달 불가능하다는 사실도 여기서 재현된다.

진짜로 열려 있고 새로 드러난 것은 따로 있다. 같은 파일의 sm_80 텐서 코어 구성은 훨씬 큰 타일을 쓰고 tf32 갈래에서는 3단이다. 이 경로가 천장을 건드릴 법한 자리가 거기이고, 답하려면 Ampere 이후 부품이 필요하다. 눈감고 고치는 대신 후속 과제로 남긴다.

### 5.4 SASS 디프 기법은 #1541과 달리 여기서는 통한다

#1539는 sm_80과 sm_121로 컴파일해 `cuobjdump --dump-sass`를 비교하는 방식으로 sm_80+ 기준을 회수했다. #1541은 번역 단위가 호스트 디스패치뿐이고 오브젝트에 디바이스 코드가 없으면 이 기법이 옮겨 가지 않는다는 것을 확인하고, 무의미한 동일 디프를 주장하는 대신 그렇다고 적었다. 어느 쪽 경우인지는 디프가 의미를 갖기 전에 확인해야 한다. 이 파일은 진짜 디바이스 코드를 담는다. `grouped_gemm_unaligned.cu.o`는 함수 51개짜리 sm_70 큐빈 하나를 갖고 있으므로 4절의 동일 덤프는 결과다. 방법론 주의사항 하나. 두 빌드의 전체 파일 덤프는 다르다. 큐빈이 같은 함수를 다른 순서로 내보내기 때문이고, 그래서 비교는 심볼별로 해야 한다.

### 5.5 GB10 이월, 그리고 회수한 기준 하나

이 호스트에 sm_80 이후 부품은 없다. 에픽 #1536의 `## GB10 (sm_121) continuation`에 따라:

- **GB10 MoE 출력 바이트 동일**과 **GB10 MoE 처리량 불변**: 실행하지 않았고 체크하지 않은 채 남겼다. 로컬에서 회수한 것은 측정이 아니라 기전이다. 태그 매핑은 8 이상 모든 compute capability에서 불변임이 증명돼 있고, 그 매핑이 고르는 디바이스 코드는 `compute_80`과 `compute_121`에서 변경 전후로 심볼별 바이트 동일이다. GB10 출력이나 처리량이 움직일 경로가 남지 않지만, 둘 다 측정이고 이것은 논증이다.
- **`== 8`과 `> 8` 갈래 불변**: 이월하지 않고 여기서 닫았다. 출하 함수에 대한 열거와 아키텍처 교차 SASS 비교로.
- **sm_121에서 `cargo test --features cuda` 통과**: GB10 호스트가 필요하다.

`CUDA sm_70 compile` CI 체크는 위 어느 것에 대해서도 증거가 아니다. CUDA 13은 Volta 지원을 제거해 `compute_70`을 컴파일할 수 없고, 그래서 그 잡은 건너뛰며 11초쯤에 통과한다. 로컬 빌드가 유일한 실제 검증이다.

### 5.6 #1543 이후 재확인

#1543은 양자화 GEMM용 Volta 텐서 코어 MMA 원자를 준다. 그 작업이 그룹 GEMM의 Ampere 이전 `GemmConfiguration`에까지 닿으면 3.2절의 `static_assert`가 발화하고, Turing이 `m16n8k8`을 유지하도록 단일 Ampere 이전 갈래를 쪼개야 한다. 그것이 의도한 트리거이고, 그래서 단언문이 `grouped_gemm_arch.h`를 지목한다.

## 참고

- 이슈 #1544, 그리고 Volta 디코드 프로그램인 에픽 #1536.
- 전체 측정 기록: `docs/benchmark_results/grouped-gemm-arch-v100-2026-08-31.md`.
- 이 문서가 델타로 삼는 베이스라인: `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md` (#1538).
- 양자화 체크포인트에서 이 분기를 도달 가능하게 만드는 정렬 MoE 프리필 빠른 경로: #629, `docs/benchmark_results/moe-prefill-grouped-gemm-gb10-2026-07-10.md`.
- #1541에서 넘겨받은 `cuFuncSetAttribute` 공백과 그 `gather_gemm.cu` 도달 가능성 결론: `docs/benchmark_results/qmm-naive-tile-v100-2026-08-31.md`.
- SASS 디프 기법: 통하는 경우는 #1539, 통하지 않는 경우는 #1541.
- 기존 sm_70 비트 일치 부류: `tests/cuda_qmm_determinism.rs`에 대한 #1557.
