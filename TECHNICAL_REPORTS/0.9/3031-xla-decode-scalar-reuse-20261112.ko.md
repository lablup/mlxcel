# 기술 보고서: PR #3031 - perf(xla): decode step 스칼라와 토큰 readback 재사용

**작성일**: 2026-11-12
**상태**: 완료
**언어**: C, Rust
**위험도**: 낮음

---

## 요약

단일 시퀀스 OpenXLA decode step은 매 step마다 host-visible 버퍼 네 개(스칼라 입력 3개와 토큰 readback staging 1개)를 할당했고 각각에 pinned `hipHostMalloc`/`hipHostFree` 쌍을 지불했다. gfx1151에서 이 쌍은 약 129 us이며 매 step 경계의 GPU 유휴 구간으로 관측됐다. 이제 네 버퍼는 컨텍스트당 한 번 할당되어 영구 매핑되고, step 값은 그 자리에 직접 쓰이며, 4바이트 토큰 readback은 미리 할당된 슬롯으로 device-to-device 복사된다. Decode 처리량은 두 차례 교차 A/B 합산 중앙값 140.7에서 150.1 tok/s로 올랐고(step당 7.11에서 6.66 ms) greedy 출력은 바이트 단위로 동일하며, 강제 fallback device 테스트가 해당 heap을 제공할 수 없는 장치에서도 기존 경로가 동작함을 입증한다.

---

## 1. 문제 정의

### 1.1 배경

OpenXLA/IREE 백엔드는 같은 ROCm 호스트에서 MLX 백엔드에 크게 못 미쳤다. llama-3.2-1b-instruct-4bit에서 143 tok/s 대 239 tok/s다. decode 루프의 `rocprofv3 --kernel-trace`는 6.97 ms step 중 4.31 ms를 GPU 커널로 귀속시켰고, 약 2.2 ms의 GPU 유휴와 host 시간이 남았다. `rocprofv3 --hip-trace` 한 번이 이를 특정했다. 정상 상태 step마다 pinned `hipHostMalloc`/`hipHostFree` 쌍이 2.06회였고 free의 p50이 147.9 us였으며, kernel trace는 step 경계에서 148.3, 21.1, 147.0 us의 유휴 구간을 보였다.

### 1.2 기존 문제점

- 세 스칼라 입력(토큰 id, positions, cache 길이)은 각각 `xla_alloc_bv`로 할당됐다. 이 함수는 device 버퍼를 할당하고 바이트를 복사하며, IREE는 그 복사의 host 쪽을 자체 staging 할당으로 뒷받침한다.
- 4바이트 토큰 readback은 `iree_hal_device_transfer_d2h`를 썼고, 호출마다 host staging 버퍼를 할당하고 해제했다.
- `xla_ctx`에 step 상태를 둘 자리가 없어 구조적으로 모든 것이 호출 단위였다.

호스트에서 HIP 마이크로벤치마크로 기본 비용을 직접 쟀다. `hipHostMalloc(32 B)` 11.7 us, `hipHostFree` 122.4 us, 쌍 128.9 us이며 `hipMalloc`+`hipFree`는 0.9 us다. 전체 비용이 4에서 16바이트 payload에 대한 allocator churn이었다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|-------------|
| 기기가 쓰기 전에 영구 매핑을 읽음(동기화 누락) | 잘못된 토큰 | 낮음: d2d 전송이 동기이며 기존 d2h readback이 의존하던 순서와 같다 |
| 기기가 host-local device-visible heap을 제공하지 못함 | decode 즉시 실패 | 중간: 상태 머신과 fallback 테스트로 완화 |
| write-combined upload 메모리가 제때 device에 보이지 않음 | 잘못된 토큰 | 낮음: 모든 IREE h2d staging 복사가 이미 이 heap을 쓴다 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

입력 검증, 인증, 데이터 처리 표면의 변화는 없다. 새 버퍼는 모델 상태 int32 네 개를 담는다. fallback 경로는 IREE 상태 코드와 메시지를 stderr에 한 번 남기며 사용자 데이터는 포함하지 않는다.

### 2.2 성능 관점

| 영역 | 이전 | 이후 | 개선 |
|------|------|------|------|
| Decode step (gfx1151, n=256) | 7.11 ms | 6.66 ms | -6.3% |
| Decode 처리량 | 140.7 tok/s | 150.1 tok/s | +6.7% |
| step당 pinned malloc/free 쌍 | 2.06 | 0 | -100% |
| step당 할당 | 4 | 0 | -100% |

GPU 커널 수와 방출된 그래프는 그대로이며 절감분은 전부 host 쪽이다. 이 호스트의 실행 간 변동이 효과보다 커서 old와 new를 매 라운드 교차하는 4라운드 A/B를 두 번(팔당 8회) 사용했고, 새 경로가 8쌍 중 7쌍에서 앞섰다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. C ABI와 공개 Rust API는 그대로다.
- **New Dependencies**: 없음.
- **호환성**: `HOST_LOCAL | DEVICE_VISIBLE` 메모리를 제공할 수 있는 IREE HAL 기기는 빠른 경로를 타며, 그 외는 `step_scalars_state = -1`을 통해 이전 동작을 유지한다.

### 2.4 코드 품질 관점

device 테스트 세 개(재사용 카운터, 동일 토큰 스트림을 내는 강제 fallback, prefill만 하는 컨텍스트)를 추가했고 기존 prepared-prefill 4개와 ragged-hardening 6개 gate가 모두 통과한다. `cargo clippy`는 새 발견을 보고하지 않는다.

---

## 3. 기술적 선택과 그 이유

### 3.1 subspan view를 붙인 단일 블록 대신 영구 버퍼 집합

**컨텍스트:** 이슈는 subspan view 네 개를 붙인 32바이트 단일 할당을 제안했다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 32바이트 단일 블록 + subspan | 할당과 매핑이 각각 하나 | readback 워드와 입력이 서로 다른 heap을 필요로 하므로 한쪽에 잘못된 heap을 강제한다 |
| **선택: 네 개의 개별 할당** | 각 슬롯이 접근 패턴에 맞는 heap을 받는다 | 컨텍스트 설정 시 할당 호출이 하나에서 넷으로 늘어난다(첫 decode에서 한 번) |

**선택 이유:** HIP allocator는 요청된 타입 비트로부터 할당마다 memory heap을 고른다. 세 입력은 host가 쓰고 device가 읽으므로 write-combined upload heap이 맞고, readback 워드는 device가 쓰고 host가 읽으므로 write-combined 메모리에서는 CPU 캐시를 우회하게 된다. 할당을 나누는 비용은 컨텍스트당 `hipHostMalloc` 세 번 추가뿐이며 각 슬롯이 올바른 heap을 갖는다.

**트레이드오프:** 설정 코드가 조금 늘고 기준 포인터 하나 대신 포인터 네 개를 유지한다.

### 3.2 캐시된 d2h 대신 device-to-device readback

**컨텍스트:** readback 대상은 device가 쓸 수 있고 host가 읽을 수 있어야 한다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| `transfer_d2h` 유지 | 새 코드 없음 | 호출마다 staging 버퍼를 할당·해제하며 이것이 제거하려는 비용이다 |
| **선택: 매핑된 host-local 슬롯으로 `transfer_d2d`** | 할당 없음, 4바이트 복사 한 번 | 매핑된 pinned 포인터가 유효한 d2d 대상이어야 하며 ROCm의 단일 주소 공간에서는 성립한다 |

**선택 이유:** 전송 함수는 동기적으로 복사 완료를 기다리므로 직후 매핑 포인터를 읽는 것은 기존 d2h readback이 의존하던 순서와 같다. `HOST_COHERENT` 덕분에 명시적 invalidate가 필요 없다.

### 3.3 강제 요구가 아닌 3상태 필드

`step_scalars_state`는 첫 decode 전에 `0`, 재사용이 활성이면 `1`, 할당이나 매핑이 실패하면 `-1`이다. `-1` 컨텍스트는 손대지 않은 per-step 코드를 계속 쓰므로 드라이버나 allocator의 한계가 decode 실패가 아니라 성능 저하로 남는다. `xla_test_force_step_scalars_unsupported`가 이 상태를 직접 설정해 fallback을 테스트 가능하게 만들고, 테스트는 fallback이 같은 토큰 스트림을 낸다고 단언한다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[이전]
decode step:
  xla_alloc_bv(token)     -> device 버퍼 + h2d staging (pinned malloc/free)
  xla_alloc_bv(positions) -> device 버퍼 + h2d staging (pinned malloc/free)
  xla_alloc_bv(cache_len) -> device 버퍼 + h2d staging (pinned malloc/free)
  invoke, 출력 pop
  transfer_d2h(token_out) -> staging (pinned malloc/free)

[이후]
첫 decode:
  xla_step_scalars_ensure -> host-local device-visible 버퍼 4개, 한 번 매핑
decode step:
  매핑에 token/positions/cache_len 기록
  invoke, 출력 pop
  transfer_d2d(token_out -> 영구 슬롯), 매핑 포인터 읽기
```

### 4.2 주요 코드 변경

**파일: `src/lib/mlxcel-xla/csrc/xla_iree.c`**

```c
static const iree_hal_buffer_params_t kStepScalarParams = {
    .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
            IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
    .access = IREE_HAL_MEMORY_ACCESS_ALL,
    .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING |
             IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
};

if (xla_step_scalars_ensure(c, position_count) == 1) {
    owns_scalars = false;
    *(int32_t*)c->step_token_map.contents.data = token;
    *(int32_t*)c->step_len_map.contents.data = cache_len;
    memcpy(c->step_pos_map.contents.data, positions,
           (size_t)position_count * sizeof(int32_t));
    tok_bv = c->step_token; pos_bv = c->step_pos; len_bv = c->step_len;
} else {
    /* 이전 per-step 할당 경로 그대로 */
}
```

**변경 이유:** decode step이 할당을 멈추고 이미 소유한 메모리에 값을 쓴다.

readback은 decode 전용 helper로 옮겨 prefill과 ragged decode가 기존 readback 경로를 그대로 쓰게 했다.

```c
static iree_status_t xla_read_token_decode(xla_ctx* c,
                                           iree_hal_buffer_view_t* out,
                                           int32_t* out_token) {
  if (c->step_scalars_state != 1 || !c->decode_result) {
    return xla_read_token(c, out, out_token);
  }
  if (iree_hal_buffer_view_element_count(out) != 1 ||
      iree_hal_buffer_view_element_type(out) != IREE_HAL_ELEMENT_TYPE_INT_32) {
    return xla_read_token(c, out, out_token);
  }
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2d(
      c->device, iree_hal_buffer_view_buffer(out), 0,
      iree_hal_buffer_view_buffer(c->decode_result), 0, sizeof(int32_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  *out_token = *(const int32_t*)c->decode_result_map.contents.data;
  return iree_ok_status();
}
```

`xla_llama_free`는 device가 사라지기 전에 네 view를 unmap하고 release하며, 소유 플래그(`owns_scalars`)가 오류 경로에서 소유하지 않은 view를 release하지 않게 한다.

### 4.3 데이터 모델 변경

없음.

---

## 5. 학습 포인트

### 5.1 APU에서 pinned 할당 비용

**개념:** `hipHostMalloc`은 DMA나 매핑된 device 접근을 위한 page-locked 메모리를 만든다. 페이지 잠금과 해제는 요청 크기가 아니라 시스템 상태에 비례하는 커널 작업이다.

**이 PR에서의 적용:** 4바이트 readback staging 버퍼가 수 MB 버퍼와 같은 비용을 냈다. 쌍당 128.9 us는 6.97 ms step의 1.8%이며 추적된 두 쌍이면 약 3.7%다.

**예시:** 기본 연산만 따로 할당·해제하는 30줄 HIP 마이크로벤치마크가 end-to-end 수치만으로는 분리할 수 없던 GPU 작업과 이 비용을 갈라냈다.

### 5.2 IREE memory heap과 영구 매핑

`IREE_HAL_MEMORY_TYPE_HOST_LOCAL`은 `HOST_VISIBLE`과 `HOST_COHERENT`를 이미 포함하는 복합 타입이다. `HOST_CACHED`를 더하면 캐시된 page-locked heap이 선택되고, 빼면 write-combined upload heap이 선택된다. 영구 매핑(`IREE_HAL_MAPPING_MODE_PERSISTENT`)은 버퍼가 `MAPPING_PERSISTENT` usage로 할당됐어야 하고 버퍼 해제 책임이 호출자에게 남으므로, 네 view를 `xla_ctx`에 보관한다.

### 5.3 중앙값이 아니라 쌍을 재기

같은 old 바이너리로 두 번의 A/B 세션을 돌렸더니 old 중앙값이 4% 벌어졌고 이는 측정하려는 효과보다 컸다. 라운드마다 old와 new를 교차하고 라운드별 쌍을 보고하면 host 변동이 회귀로 오인되지 않는다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 이 PR에서의 의의 |
|--------|------|------------------|
| `hipHostMalloc` | page-locked host 할당 | 이 PR이 제거하는 per-step 비용 |
| `HOST_LOCAL` | IREE 복합 memory type(host visible, coherent) | 영구 스칼라에 요청한 타입 |
| `IREE_HAL_MAPPING_MODE_PERSISTENT` | 명시적으로 unmap할 때까지 유지되는 매핑 | 컨텍스트 수명 동안 쓰기 가능한 포인터 유지 |
| `transfer_d2d` | 전송 큐를 통한 device-to-device 복사 | staging d2h readback을 대체 |

### 관련 PR/이슈

- 이슈 #3028: 이 변경이 다루는 측정값을 담은 성능 이슈.
- PR #2976: ragged load 시간을 분리하고 decode 전용 처리량을 복원해 baseline을 비교 가능하게 만들었다.
- #2886, #2873: 남은 decode 경로 격차(좁은 weight projection, small-M vecmat, 직렬 argmax)로 별도 추적된다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 6 |
| 추가 라인 | +544 |
| 삭제 라인 | -17 |
| 추가 테스트 | device 테스트 3개 |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|------|------|
| 성능 | 1 | 영구 decode 스칼라와 d2d readback |
| 테스트 | 1 | 새 `xla_decode_scalar_reuse` gate와 `[[test]]` 항목 |
| 문서 | 1 | CHANGELOG 항목 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `5bea811d` | perf | perf(xla): reuse decode step scalars and token readback (#3028) |

---

## 8. 후속 조치

### 완료 필요

- [x] device 테스트, 생성 텍스트 동일성, A/B 벤치마크.
- [ ] 남은 항목 없음.

### 모니터링 필요

- merge 이후 decode 처리량. 이 변경은 커널 수나 GPU busy 시간을 바꾸지 않아야 하므로 그쪽이 움직이면 무관한 회귀를 뜻한다.

### 향후 개선 사항

- MLX와의 남은 격차(150 대 239 tok/s)는 GPU 쪽이다. 1B 4-bit 가중치에 커널 시간 4.31 ms가 든다. 좁은 weight projection 전문화(#2886)와 small-M vecmat, 병렬 argmax(#2873)가 다음 후보다.
- step pipelining이 들어오면 입력 슬롯 세 개는 double buffering이 필요하다. readback 슬롯은 뒤따르는 값 하나라 그대로 유효하다.

---

## 부록

### A. 테스트 결과

```
cargo test --features "xla-iree test-utils" --test xla_decode_scalar_reuse -- --ignored --test-threads=1   # 3 통과
cargo test --features "xla-iree test-utils" --test xla_ragged_hardening -- --ignored --test-threads=1      # 6 통과
cargo test --features "xla-iree test-utils" --test xla_prepared_prefill -- --ignored --test-threads=1      # 4 통과
cargo check --features xla-iree                                                                            # 통과
cargo fmt --all -- --check                                                                                 # 통과
```

### B. 성능 벤치마크

교차 4라운드 A/B 두 번, llama-3.2-1b-instruct-4bit, native quant, context capacity 512, n=256, greedy:

| 라운드 | old tok/s | old ms | new tok/s | new ms |
|--------|-----------|--------|-----------|--------|
| 1 | 138.72 | 1845.50 | 145.80 | 1755.82 |
| 2 | 142.66 | 1794.51 | 145.87 | 1755.02 |
| 3 | 137.69 | 1859.25 | 150.76 | 1698.03 |
| 4 | 127.83 | 2002.66 | 155.15 | 1650.06 |
| 5 | 144.98 | 1765.80 | 142.78 | 1792.96 |
| 6 | 136.68 | 1873.01 | 151.72 | 1687.33 |
| 7 | 144.73 | 1768.77 | 153.15 | 1671.58 |
| 8 | 143.60 | 1782.73 | 149.48 | 1712.58 |

합산 중앙값: 150.1 대 140.7 tok/s, step당 6.66 대 7.11 ms.

### C. 참고 자료

- IREE HAL memory type과 매핑: 고정된 ROCm 배포판의 `iree/hal/buffer.h`.
- HIP pinned 할당 동작: 같은 배포판의 `hip_allocator.c`.
