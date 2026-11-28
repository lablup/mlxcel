# 기술 보고서: PR #2867 - fix(metal): bound quantized loader reads to the tile extent

**작성일**: 2026-11-06
**상태**: 리뷰 중 (Open PR, #2852 종료)
**언어**: Metal/C++, Rust, CMake
**위험도**: Medium

---

## 요약

PR #2867은 이슈 #2852를 계속 다시 열리게 만든 간헐적 NaN을 수정한다. 원인은 프로세스 전역 상태 누수나 할당자에 남은 오래된 내용이 아니라, MLX의 Metal `QuantizedBlockLoader::load_safe`가 타일의 행 범위(`.y`)만 확인하고 비행 범위(`.x`)를 무시한 것이다. 그래서 전치된 양자화 곱의 partial-K 타일이 다음 출력 행의 packed 바이트와 group scale을 디양자화했다. Nemotron-H nvfp4 픽스처에서 `FF = 48`이 마지막 K 타일을 32폭 중 16폭으로 만들고, 이 읽기가 마지막 expert의 packed weight 평면을 8바이트, E4M3 scale 평면을 1바이트 넘어간다. 오래된 `0x7F` scale 바이트는 NaN으로 디코드되고, 대응하는 x 타일은 0으로 채워져 있으므로 `0 * NaN`이 그 expert 20개 pair의 마지막 출력 열을 NaN으로 만들었다. 이것이 40토큰에서 보고된 `non-finite 20 of 2560`과 정확히 일치한다. 수정은 세 loader 복사본(steel fp, affine, NAX) 모두에서 각 스레드의 읽기를 `src_tile_dim.x`로 제한하고 나머지를 0으로 채운다. 새 failing-first 회귀 테스트는 수정 전 트리에서 정확히 20개의 non-finite를 재현하고 수정 후 통과하며, 전체 직렬 lib 스위트는 11368 통과, 0 실패, 239 무시다. NAX 복사본은 이 M3 Ultra에 NAX 디바이스가 없어 컴파일 검증만 되었다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #2852는 머지 후 워크스페이스 게이트에서 `models::nemotron_h::tests::nemotron_h_runs_nvfp4_experts_sorted_and_unsorted`가 한 번 실패한 사건을 추적했다. 메시지는 "nvfp4 routed experts differ from the dequantized reference at 40 tokens"였다. 같은 테스트는 단독 실행에서 10회 중 10회 통과했고 최대 절대 오차는 1e-3 경계 대비 7.6e-6이었으며, 40회 이상의 전체 스위트 실행에서도 재현되지 않았다. 실패가 순서 의존적이어서, 안정적으로 빨간 테스트보다 더 나쁘다. 무관한 PR에 불시에 걸리기 때문이다.

이전 작업이 뻔한 프로세스 전역들을 배제했다. PR #2866은 MLX 기본 디바이스 누수 가설을 CPU 강제 측정으로 반증했고(40토큰 5.7e-6, 여전히 경계 안), 그 측정을 가드 테스트로 남겼다. `fused_moe_enabled`의 `OnceLock`, `SSD_PREFILL_DISPATCH_LOGGED`, rope override 싱글턴, 스레드에 남은 기본 스트림, 컴파일된 함수 캐시는 각각 프로브로 배제되었다. 남은 후보는 MLX 할당자와 Metal 커맨드 버퍼 상태였고, 이는 값이 잘못 쓰인 것이 아니라 커널이 오래된 메모리를 읽고 있다는 방향을 가리켰다.

실패하는 테스트는 결정론적 합성 가중치로 1계층 nvfp4 Nemotron-H MoE를 만들고, 같은 packed 평면을 f32 참조로 디양자화한 뒤 5토큰(unsorted gather)과 40토큰(sorted gather)에서 `SwitchMLP::forward`를 비교한다. 40토큰 팔만 실패했으므로 결함은 sorted gather 경로에 있고, 공유 디양자화나 unsorted 커널에는 없다.

### 1.2 기존 문제점

- **`load_safe`가 계약의 절반만 검사했다.** `load_safe(short2 src_tile_dim)`는 `src_tile_dim`을 타일의 유효 범위로 문서화하지만, loader는 `.y`(BROWS)만 검사했다. `reduction_dim == 1` 레이아웃에서 reduction(K) 축인 `.x` 범위는 무시되었다.
- **partial K 타일이 다음 출력 행을 디양자화했다.** 픽스처의 `FF = 48`은 fc2의 K를 32폭 K 타일에 대해 48로 만들므로 마지막 타일은 16폭이다. loader는 그대로 `n_reads`개 pack을 모두 디양자화해, 16폭 마지막 타일의 위쪽 절반을 다음 출력 행의 packed 바이트와 group scale에서 가져왔다.
- **마지막 expert의 마지막 출력 행이 평면을 벗어났다.** 그 읽기는 packed weight 평면을 8바이트(오프셋 3072 초과), E4M3 scale 평면을 1바이트(오프셋 384 초과) 넘어간다.
- **오래된 scale 바이트가 NaN 출력이 되었다.** E4M3 `0x7F`는 NaN으로 디코드된다. 대응하는 x 타일은 0으로 채워져 있으므로 곱은 `0 * NaN`, 즉 NaN이고, 조용히 버려지지 않고 유효한 출력 열에 도달한다.
- **증상이 모든 관측을 설명한다.** 마지막 expert로 라우팅되는 pair가 정확히 20개라서 40토큰에서 `non-finite 20 of 2560`이 되고, 모든 문제 인덱스는 32로 나눈 나머지가 31이다. 즉 그 20개 행 각각의 마지막 출력 열이다. 5토큰 팔은 다른 커널을 타서 실패한 적이 없다.
- **순서는 poison만 결정했다.** 384바이트 scale 평면이 재사용하는 크기 클래스에 어느 선행 테스트가 마지막으로 `0x7F`를 남겼는지가 NaN에 닿는지를 결정했고, 그래서 실패가 순서 의존적으로 보였다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|---|---|---|
| 범위 밖 읽기가 서빙되는 양자화 matmul을 오염시킨다 | High (잘못된 또는 NaN logit) | 이 픽스처에서는 Low이나, 같은 loader가 모든 전치 양자화 곱을 서빙하므로 K가 배수가 아닌 모든 모델이 걸릴 수 있다 |
| 오래된 NaN 바이트를 수치 허용 오차 문제로 오진한다 | Medium (1e-3 경계를 넓혀 결함을 숨긴다) | 유도 과정을 기록하지 않으면 High이며, 이 보고서가 그 기록이다 |
| NAX 복사본이 M5급 기기에서만 드러나는 방식으로 틀렸다 | Medium (NAX 하드웨어의 런타임 오염) | 미상: 여기서는 컴파일 검증만 되었다 |
| 나머지 0 채우기가 범위 안 수치를 바꾼다 | Medium (조용한 정확도 변화) | Low: x loader가 이미 그 자리를 0으로 채우므로 곱해지는 값이 같다 |
| 회귀 테스트가 결정론적이지 않고 흔들린다 | Medium (신뢰할 수 없는 카나리아) | Low: 팔을 돌리기 전에 풀을 비우고 명시적으로 seed한다 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

네트워크, 인증, 입력 파싱, 직렬화 표면의 변화는 없다. 수정은 디바이스 측 loader 안의 범위 검사다. 범위 밖 읽기를 제거하므로 커널이 만지는 메모리 표면을 엄격히 줄인다. 사용자 데이터 로깅은 없다.

### 2.2 성능 관점

변경은 `load_safe` 호출마다 정수 나눗셈 1회, 비교 1회, 0 채우기 루프 1회를 추가한다. `load_safe`는 출력 원소마다가 아니라 K 타일마다 thread group당 한 번 실행되므로, 비용은 협력 로드의 잡음 수준이다. 범위 안 타일은 같은 pack을 같은 scale로 디양자화하고, 0으로 채우는 나머지는 x loader가 이미 넣던 값이므로, 실제 답을 만드는 타일의 산술과 메모리 트래픽은 변하지 않는다. 변경이 처리량 경로가 아니라 범위 가드이므로 벤치마크는 실행하지 않았다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. 가득 찬 타일의 커널 동작은 그대로다.
- **New Dependencies**: 없음. 세 overlay 헤더는 upstream MLX `7ca40033132dae444d45f7fc3daae57e24c13f6c`에 고정된 그대로의 복사본에 표시된 `load_safe` 본문만 더한 것이다.
- **호환성**: nvfp4가 쓰는 steel fp loader, float 및 half scale이 쓰는 affine loader, NAX loader를 모두 고친다. NAX 경로는 이 호스트에서 실행되지 않는다.
- **Overlay 유지보수**: `src/lib/mlxcelverse/CMakeLists.txt`가 세 파일을 `_metal_patch_files`에 등록하고 refresh 규칙을 기록한다. 새 upstream 파일을 가져와 표시된 세 `load_safe` 본문을 다시 적용한다.

### 2.4 코드 품질 관점

새 회귀 테스트 `nemotron_h_nvfp4_partial_k_tile_does_not_read_past_the_expert_planes`는 결정론적이다. `clear_cache()`를 호출하고 `388..=768`의 모든 4바이트 크기를 `0x7F`로 seed한 뒤 두 팔을 실행하고 모든 출력이 유한인지 단언한다. 388에서 768 구간은 할당자의 `reuse_from_cache(384)` 허용 범위 `[384, 768)`이므로, 384바이트 scale 평면은 반드시 seed된 버퍼를 재사용한다. parity 진단은 이제 `got`과 `reference` 양쪽의 non-finite 개수를 보고하므로, 커널이 만든 NaN과 참조가 만든 NaN을 구분한다. `cargo fmt --all -- --check`와 `cargo clippy --profile test-fast --features metal,accelerate -p mlxcel --lib --tests -- -D warnings`는 깨끗하다.

---

## 3. 기술적 선택과 그 이유

### 3.1 읽기를 `src_tile_dim.x`로 제한하고 나머지를 0으로 채운다

**컨텍스트:**
`load_safe`는 K 또는 N 범위가 커널 타일을 채우지 못할 때 타일의 유효 범위와 함께 호출된다. loader는 `.y`(행)만 검사하고 `.x`는 검사하지 않아, K 범위와 무관하게 `n_reads`개 pack을 디양자화했다. 같은 헤더의 `Fp4HalfLoader::load_safe`는 이미 열 범위를 제한하고 있었고, 수정은 그 패턴을 따른다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|---|---|---|
| 대안 A: 각 스레드의 읽기를 `src_tile_dim.x`로 제한하고 나머지를 0으로 채운다 | 기존 `Fp4HalfLoader` 패턴과 같고, 범위 안 수치가 동일하며, 호스트 측 변경이 없다 | loader 호출마다 분기와 채우기 루프가 늘어난다 |
| 대안 B: 커널 이후 출력을 마스킹한다 | loader를 건드리지 않는다 | NaN이 이미 accumulator에 들어갔고 `0 * NaN`은 사후 마스킹이 불가능하며, scale 읽기 자체가 여전히 범위 밖이다 |
| 대안 C: 호스트에서 weight와 scale 평면을 패딩한다 | 커널을 그대로 둔다 | expert마다 할당과 복사가 필요하고 평면 stride가 바뀌며, 계약 위반을 고치지 않고 숨긴다 |
| **선택: 대안 A** | 계약을 근원에서 고치고 같은 파일의 기존 검토된 패턴과 일치한다 | 호출당 작은 분기 |

**선택 이유:**
읽기는 사후 복구가 아니라 예방해야 한다. NaN scale이 0으로 채워진 x 타일과 곱해지면 결과는 NaN이고, 어떤 하위 마스크도 의도된 0을 되살릴 수 없다. 읽기를 제한하면 스레드의 첫 pack이 이미 타일을 지났을 때 scale 로드를 통째로 건너뛰게 되어, 마지막 출력 행이 384바이트 scale 평면 바로 뒤 바이트를 건드리지 않는다.

**트레이드오프:**
loader가 partial 타일용 채우기 루프를 갖게 된다. x loader가 이미 하던 채우기와 같아서 scratch 레이아웃 변경도 추가 할당도 없다.

### 3.2 세 loader 복사본 모두에 수정을 적용한다

**컨텍스트:**
overlay는 loader를 세 벌 갖는다. steel fp loader(`fp_quantized.h`), affine loader(`quantized.h`), NAX 복사본(`fp_quantized_nax.h`)이다. upstream이 커널 계열별로 코드를 복제한다.

**선택 이유:**
셋 모두 같은 `.y` 전용 검사를 갖는다. 하나만 고치면 affine 경로(float 및 half scale)와 `fp_gather_qmm_rhs_nax`를 통해 같은 결함이 남는다. NAX는 K가 64폭 NAX K 타일의 배수가 아닐 때 `tile_w.x = k_remain`으로 `loader_w.load_safe(tile_w)`에 도달한다. CMake overlay 목록이 셋을 모두 등록하므로, 향후 upstream refresh가 하나를 조용히 빠뜨릴 수 없다.

**트레이드오프:**
upstream refresh마다 세 복사본을 동기화해야 한다. 그래서 refresh 규칙을 파일 목록 옆에 기록했다.

### 3.3 확률적 재현 대신 seed된 풀을 쓰는 회귀 테스트

**컨텍스트:**
원래 실패는 오염된 버퍼 풀당 한 번 나타났으므로, 단순 재실행으로 수정을 증명할 수 없다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|---|---|---|
| 대안 A: 재사용 구간을 명시적으로 seed하고 유한성 단언 | 매 실행 결정론적이고 정확한 메커니즘을 코드에 담는다 | 테스트가 문서화한 할당자 재사용 정책에 의존한다 |
| 대안 B: 실패할 때까지 픽스처를 재시도 | 할당자 지식이 필요 없다 | 비결정론적이고 느리며 게이트로 쓸 수 없다 |
| 대안 C: 커널에서 읽기 횟수를 단언 | 직접적이다 | 커널 계측과 별도 빌드 경로가 필요하다 |
| **선택: 대안 A** | 수정 전 트리에서 매번 실패하고 수정 후 통과한다 | 현재 재사용 범위에 묶인다 |

**선택 이유:**
`clear_cache()`로 풀을 비운 뒤 388에서 768까지 모든 4바이트 크기를 `0x7F7F7F7F` 내용으로 만들고 버린다. 따라서 384바이트 scale 평면이 재사용할 수 있는 버퍼가 모두 오염되어, 범위 밖 바이트가 매 실행 NaN이 된다. 테스트는 두 팔에서 유한성을 단언하고 기존 allclose 검사도 유지하므로, 올바른 이유로 실패하며 허용 오차가 밀려서 통과하지 않는다.

**트레이드오프:**
할당자 재사용 규칙이 바뀌면 seed 범위도 따라가야 한다. 주석에 범위가 `[384, 768)`인 이유를 기록했다.

### 3.4 반증된 대안들

**컨텍스트:**
증상은 출력 행이 쓰이지 않은 것처럼 보였고, 세 차례 조사가 그 가설과 전역 누수를 추적했다.

**선택 이유:**
stale-pool under-write 가설은 직접 검증했다. 같은 그래프를 all-NaN 풀과 all-7.0 풀로 두 번 실행한 결과가 비트 단위로 동일했으므로, 유효한 출력 원소가 쓰이지 않은 경우는 없다. 커널이 저장하지 않는 출력 행이라는 설명은 배제된다. 이전 프로브가 실제 읽기를 놓친 데는 기계적인 이유도 있다. all-NaN `f32` poison은 4로 나눈 나머지가 3인 오프셋에만 `0x7F`를 쓰는데, 범위 밖 바이트는 4정렬이어서 poison이 닿지 않았다. 정렬된 섹션별 seed가 두 교란 요인을 모두 없앤다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[Before]
quantized matmul (steel / affine / NAX)
  └── QuantizedBlockLoader::load_safe(src_tile_dim)
        src_tile_dim.y만 검사
        스레드마다 n_reads개 pack 디양자화
        └── partial K 타일이 다음 출력 행의 바이트와 group scale을 읽음

[After]
quantized matmul (steel / affine / NAX)
  └── QuantizedBlockLoader::load_safe(src_tile_dim)
        src_tile_dim.y와 src_tile_dim.x 검사
        bi < src_tile_dim.y일 때 reads = min(n_reads, valid_packs - bj)
        남은 pack을 0으로 채움
        └── partial K 타일이 자기 행과 평면 안에 머묾
```

### 4.2 주요 코드 변경

**파일: `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/fp_quantized.h`**

수정 전에는 범위 검사가 `.y`만 쓰고 모든 스레드가 자기 pack을 전부 읽었다:

```cpp
void load_safe(short2 src_tile_dim) const thread {
  if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
    return;
  }

  float scale =
      float(dequantize_scale<T, group_size>(*scales)) * inv_scale_enc;
  for (int i = 0; i < n_reads; i++) {
    dequantize<T, bits>(
        src[i * bytes_per_pack], scale, dst + i * pack_factor);
  }
}
```

수정 후에는 각 스레드가 타일 안의 pack만 읽고 나머지를 0으로 채운다:

```cpp
const int valid_packs =
    (int(src_tile_dim.x) + pack_factor - 1) / pack_factor;
const int reads = int(bi) < int(src_tile_dim.y)
    ? min(int(n_reads), valid_packs - int(bj))
    : 0;

if (reads <= 0) {
  for (int i = 0; i < n_reads * pack_factor; i++) {
    dst[i] = T(0);
  }
  return;
}

float scale =
    float(dequantize_scale<T, group_size>(*scales)) * inv_scale_enc;
for (int i = 0; i < reads; i++) {
  dequantize<T, bits>(src[i * bytes_per_pack], scale, dst + i * pack_factor);
}
for (int i = reads; i < n_reads; i++) {
  for (int j = 0; j < pack_factor; j++) {
    dst[i * pack_factor + j] = T(0);
  }
}
```

**변경 이유:** `valid_packs - bj`는 이 스레드가 타일 안에서 소유할 수 있는 pack 수다. 그 값이 0 이하일 때의 조기 반환은 `*scales` 로드도 막는다. 마지막 expert의 마지막 출력 행에서 384바이트 평면을 1바이트 넘어가던 읽기가 바로 이것이다.

**파일: `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/quantized.h`**

affine loader도 같은 제한을 받고, scale과 bias 로드가 조기 반환 아래로 내려간다:

```cpp
const int valid_packs =
    (int(src_tile_dim.x) + pack_factor - 1) / pack_factor;
const int reads = int(bi) < int(src_tile_dim.y)
    ? min(int(n_reads), valid_packs - int(bj))
    : 0;

if (reads <= 0) {
  for (int i = 0; i < n_reads * pack_factor; i++) {
    dst[i] = T(0);
  }
  return;
}

T scale = *scales;
T bias = *biases;
```

**변경 이유:** float 및 half scale이 쓰는 쌍둥이 loader다. 같은 `.y` 전용 검사와 같은 평면 밖 scale 및 bias 읽기를 갖고 있었다.

**파일: `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/fp_quantized_nax.h`**

NAX 복사본은 유효 읽기 수를 같은 방식으로 계산한 뒤 group 루프를 제한하고 나머지를 0으로 채운다:

```cpp
const int valid_groups =
    (valid_reads + n_reads_per_scale - 1) / n_reads_per_scale;
int k = 0;
for (int i = 0; i < valid_groups; i++) {
  float scale =
      float(dequantize_scale<T, group_size>(scales[i])) * inv_scale_enc;
  for (int j = 0; j < n_reads_per_scale; j++) {
    if (k < valid_reads) {
      dequantize<T, bits>(src[k * bytes_per_pack], scale, dst + k * pack_factor);
    } else {
      for (int p = 0; p < pack_factor; p++) {
        dst[k * pack_factor + p] = T(0);
      }
    }
    k++;
  }
}
for (; k < n_reads; k++) {
  for (int p = 0; p < pack_factor; p++) {
    dst[k * pack_factor + p] = T(0);
  }
}
```

**변경 이유:** NAX는 자체 복사본을 유지하고 `fp_gather_qmm_rhs_nax`를 통해 같은 partial-K finalize에 도달하지만, K 타일이 64폭이다. 대칭을 위해 고쳤고 이 호스트에서는 실행할 수 없다.

**파일: `src/models/nemotron_h_tests.rs`**

새 테스트는 풀을 비우고 재사용 구간을 poison한 뒤 두 팔의 유한성을 단언한다:

```rust
mlxcel_core::memory::clear_cache();
let seeds: Vec<_> = (97..=192u32)
    .map(|words| {
        mlxcel_core::from_slice_u32(&vec![0x7F7F_7F7Fu32; words as usize], &[words as i32])
    })
    .collect();
drop(seeds);

run_nvfp4_expert_arms(|tokens, got, want| {
    let got_values = mlxcel_core::utils::array_to_vec_f32(got);
    let non_finite: Vec<usize> = /* non-finite 값의 인덱스 */;
    assert!(non_finite.is_empty(), ...);
    /* 기존 allclose와 parity 검사 유지 */
});
```

**변경 이유:** 실패는 커널이 소유하지 않은 메모리를 읽는 것이므로 유한성이 올바른 단언이다. 허용 오차 검사는 밀렸지만 유한한 결과를 통과시킨다.

**파일: `src/lib/mlxcelverse/CMakeLists.txt`**

세 overlay 파일을 `_metal_patch_files`에 등록하고 refresh 규칙을 기록한다. 새 upstream 파일을 가져와 표시된 세 `load_safe` 본문을 다시 적용한다.

### 4.3 데이터 모델 변경

없음.

---

## 5. 학습 포인트

### 5.1 타일 범위는 2차원 계약이다

**개념:**
strided 평면에서 레지스터 타일을 채우는 협력 loader는 `short2` 범위를 받는다. 한 성분만 검사하면 다른 성분은 컴파일 타임 타일 크기와 같다고 조용히 가정하게 된다. 그 가정은 모든 가득 찬 타일에서 참이고, 배수가 아닌 범위의 경계 타일에서만 깨진다. `_safe` 변형이 존재하는 이유가 바로 그 경계다.

**이 PR에서의 적용:**
loader는 `.y`(BROWS)를 검사하고 `.x`(전치 레이아웃의 K 범위)를 무시했다. 경계 타일은 유효한 16개 대신 32개 pack을 전부 읽어 위쪽 절반을 다음 출력 행에서 가져왔다.

**흔한 사용 사례:**
- 런타임 범위를 받는 모든 `load_safe` 스타일 협력 타일 loader.
- K, N, group 수가 타일 배수가 아닌 양자화 matmul 커널.
- `_safe` 변형 코드 리뷰: 범위 인자의 모든 성분이 쓰이는지 확인한다.

### 5.2 `0 * NaN`은 버려질 lane을 오답으로 만든다

**개념:**
partial 타일은 보통 무해하다. 패딩된 lane이 0 activation과 곱해지기 때문이다. 그 추론은 패딩된 weight lane이 NaN일 때 깨진다. IEEE 754는 `0 * NaN`을 NaN으로 정의하므로, "버려질" lane이 accumulator를 오염시킨다.

**이 PR에서의 적용:**
x 타일은 partial K 위치에서 올바르게 0으로 채워져 있었지만, weight 쪽이 오래된 `0x7F` E4M3 scale 바이트를 읽어 NaN으로 디양자화했다. 곱이 NaN이 되어 실제 출력 열에 들어갔다.

**흔한 사용 사례:**
- block scale을 쓰는 양자화 커널. 오래된 scale 바이트는 어떤 비트 패턴이든 될 수 있다.
- 마스킹되거나 패딩된 reduction. 한 피연산자를 마스킹해도 다른 피연산자가 non-finite이면 안전하지 않다.

### 5.3 Poison 패턴은 읽기 폭과 정렬에 맞아야 한다

**개념:**
특정 바이트 패턴을 쓰는 할당자 poison은 실제로 쓰는 오프셋에서만 유효하다. `f32` all-NaN poison은 각 워드의 최상위 바이트에 `0x7F`를 쓰고 나머지 세 바이트는 이전 값으로 남긴다.

**이 PR에서의 적용:**
범위 밖 읽기는 4정렬이고 E4M3 scale 바이트를 읽는다. 따라서 all-NaN `f32` 풀은 4로 나눈 나머지가 3인 오프셋에 `0x7F`를 썼고 읽기에 닿지 않았다. 워드 단위로 `0x7F7F7F7F`를 seed하면 정렬 간극이 사라진다.

**흔한 사용 사례:**
- 범위 밖 읽기의 결정론적 재현.
- 테스트에서 할당자 또는 캐시 poison: 버그가 읽는 정확한 크기 클래스와 바이트 오프셋을 seed한다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 이 PR에서의 관련성 |
|---|---|---|
| `QuantizedBlockLoader` | packed weight 타일을 레지스터 타일로 디양자화하는 MLX Metal loader | 결함이 있던 컴포넌트 |
| `src_tile_dim` | 타일의 유효 행 및 열 범위를 담는 `short2` | `.x`가 무시되었고 그것이 버그다 |
| `reduction_dim == 1` | reduction 축이 BCOLS인 레이아웃. 모든 전치 양자화 곱이 쓴다 | `.x`를 K 범위로 만든다 |
| partial-K 타일 | K가 타일 폭의 배수가 아닐 때의 마지막 K 타일(여기서는 48 대 32) | 범위 밖 읽기가 일어나는 곳 |
| E4M3 | nvfp4 block scale에 쓰는 8비트 float 형식 | `0x7F`가 NaN으로 디코드된다 |
| nvfp4 | group size 16의 block scale을 쓰는 4비트 weight 형식 | 픽스처의 양자화 |
| `load_safe` | 경계 타일에서 호출되는 범위 검사 loader 변형 | 수정된 함수 |
| NAX | 자체 커널 복사본을 가진 Apple GPU 세대 | 여기서 컴파일 검증만 된 복사본 |

### 관련 기술/프레임워크

- **MLX**: Apple의 배열 프레임워크. Metal 백엔드와 양자화 matmul 커널이 `mlx/backend/metal/kernels/` 아래에 있다.
- **Metal**: Apple의 GPU API이자 셰이딩 언어. loader는 Metal로 컴파일되는 C++ 템플릿이다.
- **mlxcel Metal overlay**: 빌드 시점에 upstream MLX 소스를 패치하는 프로젝트 메커니즘. `src/lib/mlxcelverse/CMakeLists.txt`에 등록된다.

### 관련 PR/이슈

- 이슈 #2852: 이 PR이 종료하는 간헐적 실패.
- PR #2866: 기본 디바이스 가설을 반증하고 영구 가드 테스트를 추가해, 탐색 범위를 할당자와 커맨드 버퍼 상태로 좁혔다.
- PR #2853, PR #2865: 같은 배치의 인접 speculative decoding 작업. `TECHNICAL_REPORTS` 기록을 공유해서만 언급한다.
- 이슈 #2515, 이슈 #2409: 같은 테스트 바이너리와 테스트 순서를 바꾼 선행 변경으로, 실패가 드러나는 시점을 결정했다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 5 |
| 추가 라인 | +6988 |
| 삭제 라인 | -4 |
| 추가 테스트 | 회귀 테스트 1개, parity 진단의 양측 non-finite 카운터 |
| 의미 있는 커널 변경 | 세 `load_safe` 본문에 걸쳐 약 20줄 |

라인 수가 큰 것은 세 overlay 헤더 때문이다. 각각 upstream 그대로의 복사본에 표시된 수정을 더한 것이라, 대부분의 줄은 새 로직이 아니라 upstream 파일이다.

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|---|---|---|
| 정확성 | 헤더 3개 | steel fp, affine, NAX loader에서 `load_safe` 읽기를 `src_tile_dim.x`로 제한하고 나머지를 0으로 채움 |
| 테스트 | 1개 파일 | 결정론적 seed 풀 회귀 테스트, 양측 non-finite 보고 |
| 빌드 | 1개 파일 | 세 overlay 등록 및 refresh 규칙 기록 |
| 문서 | 0개 파일 | 유도 과정은 헤더 주석과 이 보고서에 있다 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `e2c2546` | fix | fix(metal): bound quantized loader reads to the tile extent |

---

## 8. 후속 조치

### 완료 필요

- [ ] M5급 기기에서 NAX loader를 실행한다. 이 M3 Ultra에 NAX 디바이스가 없어 NAX 복사본은 컴파일 검증만 되었고 런타임 동작은 미검증이다.
- [ ] upstream MLX refresh 후 세 `load_safe` 본문을 다시 적용한다. CMake 파일 목록이 규칙을 기록하지만 기계적으로 강제하지는 않는다.

### 미검증 표면

- NAX 복사본(`fp_quantized_nax.h`)은 실행되지 않았다. steel 수정과 같은 유도와 패턴을 공유하지만, 64폭 K 타일 경로와 group 루프는 여기서 한 번도 실행되지 않았다.
- affine loader는 전체 직렬 lib 스위트가 덮지만, partial-K float 또는 half scale 사례를 겨냥한 테스트는 없다. 그 커버리지는 우연적이다.
- 회귀 테스트는 Metal 할당자의 `reuse_from_cache(384)` 허용 범위가 `[384, 768)`로 유지된다고 가정한다. 그 정책이 바뀌면 seed 범위를 다시 유도해야 한다.
- 서빙 모델 스모크 테스트는 실행하지 않았다. 수정은 nvfp4 픽스처와 lib 스위트로 검증되었고 실제 nvfp4 체크포인트로는 검증되지 않았다.

### 모니터링 필요

- CI 하드웨어 레인에서 `nemotron_h_nvfp4_experts_sorted_and_unsorted`와 새 partial-K 테스트를 주시한다. 특히 MLX나 양자화 변경 이후에 그렇다.
- parity 진단의 양측 non-finite 카운터가 카나리아다. `got`만이면 커널이 만든 값이고, 양쪽이면 공유한 무언가가 만든 값이다.

### 향후 개선 사항

- 세 복사본이 refresh에서 갈라지지 않도록 `load_safe` 범위 제한을 중앙 헬퍼로 모은다.
- 개발 빌드에서 평면 밖 읽기를 감지하는 호스트 측 단언이나 디버그 커널 모드를 고려한다. 그래야 향후 계약 위반이 오염된 할당자에 의존하지 않고 크게 실패한다.

---

## 부록

### A. 테스트 결과

| 검사 | 결과 |
|---|---|
| 수정 전 트리의 `nemotron_h_nvfp4_partial_k_tile_does_not_read_past_the_expert_planes` | 40토큰에서 정확히 20개 non-finite로 실패 |
| 수정 후 같은 테스트 | 통과 |
| `cargo test --profile test-fast --features metal,accelerate -p mlxcel --lib models::nemotron_h::tests:: -- --test-threads=1` | 통과 |
| steel 및 affine loader 수정을 포함한 전체 직렬 lib 스위트 | 11368 통과, 0 실패, 239 무시 |
| `cargo fmt --all -- --check` | 통과 |
| `cargo clippy --profile test-fast --features metal,accelerate -p mlxcel --lib --tests -- -D warnings` | 통과 |

### B. 측정 환경

| 항목 | 값 |
|---|---|
| 호스트 | M3 Ultra, 512GB 통합 메모리, macOS, Metal |
| NAX 디바이스 | 없음(NAX loader는 컴파일 검증만) |
| 픽스처 | 1계층 nvfp4 Nemotron-H MoE, 결정론적 합성 가중치, `FF = 48` |
| 팔 | 5토큰(unsorted gather), 40토큰(sorted gather) |
| 참조 | 같은 packed 평면의 f32 디양자화 |
| 경계 | `1e-3 + 1e-3 * abs(ref)` (변경 없음) |

### C. 증거 기록

| 관측 | 값 |
|---|---|
| Non-finite 위치 | `got`만 (`got 20`, `reference 0`) |
| Non-finite 개수와 전체 | 40토큰에서 2560개 중 20개 |
| Non-finite 인덱스 | `[63, 95, 319, 351, ... 2399]`, 모두 32로 나눈 나머지가 31 |
| Expert와 pair | 마지막 expert이며, 그 20개 pair가 각각 마지막 출력 열을 기여 |
| Weight 평면 초과 | 3072를 8바이트 초과 |
| Scale 평면 초과 | 384를 1바이트 초과 |
| 오래된 scale 바이트 | E4M3 `0x7F`, NaN으로 디코드 |
| 대응 x 타일 | 0으로 채워져 곱이 `0 * NaN` |
| 반증된 대안 | stale-pool under-write: all-NaN 풀과 all-7.0 풀이 비트 단위로 동일한 결과 |
| 이전 프로브가 놓친 이유 | all-NaN `f32`는 4로 나눈 나머지가 3인 오프셋에만 `0x7F`를 쓰고, 읽기는 4정렬이다 |
| 5토큰 팔 | 다른 커널을 쓰며 실패한 적 없음 |
| Seed 범위 | `388..=768`의 모든 4바이트 크기, `reuse_from_cache(384)`의 `[384, 768)` 구간에 대응 |

### D. 참고 자료

- `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/fp_quantized.h`
- `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/quantized.h`
- `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/fp_quantized_nax.h`
- `src/lib/mlxcelverse/CMakeLists.txt`
- `src/models/nemotron_h_tests.rs`
- 이슈 #2852, PR #2866, PR #2867
- upstream MLX `7ca40033132dae444d45f7fc3daae57e24c13f6c`
