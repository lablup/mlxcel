# 기술 리포트: PR #3065 - decode 스칼라를 ABI i32 인자로 전달

**작성일**: 2026-11-16
**상태**: 완료
**언어**: Rust, C
**위험도**: 보통

---

## 요약

단일 시퀀스 OpenXLA decode 단계는 token, position, cache_len을 rank-0 `tensor<i32>` 버퍼 인자로 받았다. 그래프가 그 값들을 호스트 측 슬라이스 오프셋과 마스크 경계로 쓰기 때문에, IREE는 호출마다 호스트 가시 192바이트 `!stream.resource<staging>`와 `stream.resource.load` 세 개를 만들었고, `gfx1151`에서는 이것이 decode 단계마다 고정(pinned) `hipHostMalloc`/`hipHostFree` 쌍의 비용이었다. 세 값을 ABI 수준 `i32` 인자로 넘기면 컴파일된 그래프에서 staging 리소스 자체가 사라지고, decode는 195.56에서 200.25 tok/s로 올랐다(+2.40%, 짝지은 6회 모두 개선, greedy 출력 동일).

---

## 1. 문제 정의

### 1.1 배경

#3048이 decode KV 캐시를 모듈 전역 두 개로 옮긴 뒤, `gfx1151`의 decode 단계는 5.10 ms로 MLX ROCm의 약 4.2 ms와 비교됐다. 그 이전 단계의 커널 트레이스는 4.32 ms의 GPU 커널 시간과 1.51 ms의 유휴로 나뉘었고, 유휴는 다시 309~312회 디스패치의 0.79 ms 실행 간격, 163 us `hipHostFree` 옆의 0.35 ms 단계 시작 정지, 0.23 ms의 KV 복사, 0.14 ms의 기타로 측정됐다. #3048이 복사를 없애고 디스패치를 277로 줄였으며, staging churn은 남은 항목으로 문서화됐다.

### 1.2 기존 문제점

- **문제 1**: decode 호출마다 192바이트 고정 staging 버퍼를 할당하고 해제했다. 그래프가 슬라이스 오프셋을 계산하려면 세 값이 호스트에 있어야 하기 때문이다.
- **문제 2**: 같은 값을 먼저 rank-0 텐서 인자로 디바이스에 올린 뒤 다시 읽어 왔다. 인자 타입을 맞추기 위한 왕복이었다.
- **문제 3**: 할당이 단계 경계에 있어, 해제 비용이 커널 시간이 아니라 GPU 유휴로 나타났다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|-------------|
| 이미터와 C 셰임 사이의 인자 개수/타입 불일치 | 높음 | 낮음 (순서 단일, 양쪽 리뷰) |
| 새 ABI에서 기존 컴파일 번들이 깨진다 | 높음 | 없음 (capability 마커가 버퍼 경로 유지) |
| M-RoPE의 3축 position이 스칼라로 전달된다 | 높음 | 없음 (해당 경로는 마커를 내보내지 않음) |
| 수치가 바뀐다 | 높음 | 낮음 (프롤로그 이후 op 순서 불변, greedy 출력 동일) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

변경 범위는 세 정수가 그래프에 도달하는 방식뿐이다. 새 호스트 입력도, 파일·네트워크 표면도 없다. 스칼라 값은 기존 경계 검사(`position`과 `cache_len`을 `[0, context_capacity)`로 제한)를 그대로 유지하며, 제거된 buffer-view 단정은 값이 아니라 모양과 타입만 검사했다.

**체크리스트:**
- [x] 입력 검증 (position, cache_len 경계 검사 유지)
- [x] 인증/인가 (해당 없음)
- [x] 데이터 암호화 (해당 없음)
- [x] 로깅 (불가능한 ABI 불일치용 오류 경로 하나, 모델 데이터를 출력하지 않음)

**발견 사항:** 없음.

### 2.2 성능 관점

| 항목 | 이전 | 이후 | 개선 |
|------|------|------|------|
| Decode 속도 (n=128, 용량 512, native 양자화) | 195.56 tok/s | 200.25 tok/s | +2.40% |
| Decode 단계 | 5.09 ms | 5.01 ms | 약 -0.09 ms |
| 컴파일된 decode 그래프의 `!stream.resource<staging>` | 1 (IR 언급 12회) | 0 | 제거 |
| 호출당 `stream.resource.load` | 3 | 0 | 제거 |

단계 이득은 #3048이 staging 쌍에 배정한 0.35 ms의 단계 시작 유휴보다 작다. 그 유휴 중 일부는 staging 쌍 자체가 아니라 단계 경계의 호스트 작업이었음을 뜻한다. 라운드 기록은 깨끗하다. 짝지은 6회 모두 새 arm이 앞섰다.

### 2.3 호환성/의존성 관점

- **호환성 깨짐**: Rust/C API 수준에서는 없다. 컴파일된 아티팩트의 ABI가 단일 시퀀스 decode에서만 바뀌므로, 셰임은 가정하지 않고 `decode_step.scalar_inputs`를 프로브한다.
- **새 의존성**: 없음.
- **호환 범위**: prefill, 임베딩 입력 prefill, ragged/lockstep 배치 decode, Gemma3n decode, `phi4`의 `adapter_mode` 인자는 기존 전체 인자 ABI를 유지한다. M-RoPE는 `tensor<3xi32>` position과 버퍼 인자 ABI를 그대로 쓰며 staging이 남는다.

### 2.4 코드 품질 관점

- **테스트 범위**: 이미터 단위 테스트(시그니처, 프롤로그 재구성, 마커, 다른 모든 그래프의 시그니처 불변)와, 스칼라 ABI 아티팩트와 레거시 buffer-view 아티팩트를 한 프로세스에서 실행해 토큰 스트림을 비교하는 디바이스 게이트.
- **코드 복잡도**: `ArgDecl` 플래그 하나, 시그니처 렌더링 분기 하나, `Builder` 헬퍼 하나, 셰임의 프로브와 푸시 분기 하나씩.
- **기술 부채**: 스칼라 경로가 읽기 슬롯을 얻으려고 `xla_step_scalars_ensure`를 호출해, 더 이상 쓰지 않는 입력 맵 세 개를 함께 할당한다. 단계당이 아니라 컨텍스트당 한 번이다.

---

## 3. 기술적 선택과 그 이유

### 3.1 인덱스가 정적인 그래프 대신 ABI 수준 스칼라

**컨텍스트:** staging 리소스는 그래프가 자기 인자를 호스트에서 읽기 때문에 존재했다. 멈추는 방법은 두 가지였다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 읽기 인덱스를 정적으로 만들기 | ABI 무변경 | decode의 position과 마스크 경계는 구조상 동적이라 일반화 불가 |
| 토큰 스캔을 디바이스로 이동 | readback 완전 제거 | 단계당 디스패치 추가와 새 샘플링 경로 |
| `--iree-stream-resource-memory-model=unified`, `--iree-stream-resource-alias-mutable-bindings` | 코드 무변경 | #3048에서 IR이 동일하거나 불변임을 측정 |
| **선택: 세 값을 ABI `i32`로 전달** | staging 제거, 추가 디스패치 없음, 수치 불변 | 아티팩트 ABI 변경, 셰임에 두 경로 |

**선택 이유:** VM 스칼라 인자는 경계에서 이미 호스트 데이터이므로 materialize할 것도, 되읽을 것도 없다. IREE는 `util.func public @main(..., i32, i32, i32)`를 staging 리소스 없이 낮춘다는 점을 고정 컴파일러에서 확인한 뒤 실제 그래프로 다시 확인했다.

**트레이드오프:** 한 decode 함수 안에 두 ABI가 산다. 셰임이 세션마다 마커로 판단하고, 버퍼 경로는 그대로 남는다.

### 3.2 번들 버전 올리기 대신 프로브 마커

**컨텍스트:** 예전 시그니처로 컴파일된 셰임이 버퍼 뷰를 기대하는 그래프에 스칼라를 밀어 넣으면 안 된다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 번들이나 캐시 키 버전 올리기 | 명시적 | 기존 컴파일 캐시를 모두 무효화 |
| **선택: no-op `decode_step.scalar_inputs` 엔트리를 프로브** | 기존 아티팩트 유지, 셰임이 런타임에 판단 | 세션당 프로브 한 번 |

**선택 이유:** 엔트리의 존재 자체가 셰임에 필요한 capability이고, 번들 로드 뒤 한 번 호출하면 충분하다. 같은 패턴이 `decode_step.kv_import`에 이미 있었다.

**트레이드오프:** 마커가 대표하는 시그니처와 동기화를 유지해야 한다. 이미터 단위 테스트가 둘을 함께 고정하고, 디바이스 게이트가 양쪽을 실행한다.

### 3.3 M-RoPE는 버퍼 ABI 유지

**컨텍스트:** M-RoPE decode 경로는 rank-0 스칼라가 아니라 `tensor<3xi32>` position을 쓴다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 축마다 스칼라 인자 세 개 | M-RoPE staging도 제거 | 두 번째 스칼라 모양과 양방향 capability 신호 |
| **선택: M-RoPE는 버퍼 ABI 유지** | 마커가 on/off 하나, 변경 범위 작음 | 그 경로에는 staging이 남음 |

**선택 이유:** 측정된 이득은 plain-RoPE decode에 있다. 현재 단일 시퀀스 경로에서 M-RoPE 모델만 예외이고, 측정되지 않은 경로를 위해 양방향 마커를 도입할 이유가 없다.

**트레이드오프:** M-RoPE는 단계당 staging을 유지한다. 셰임은 M-RoPE 호출이 마커 있는 그래프를 만나면 잘못된 타입을 밀어 넣는 대신 거부하므로, 조용한 손상은 생기지 않는다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[이전]
호스트: token, pos, cache_len -> 디바이스 rank-0 텐서 -> 그래프
그래프: stream.resource.load x3 (192 B staging) -> 슬라이스 오프셋, 마스크 경계

[이후]
호스트: token, pos, cache_len -> VM i32 인자
그래프: tensor.from_elements x3 -> 슬라이스 오프셋, 마스크 경계  (staging 없음)
```

### 4.2 주요 코드 변경

**파일: `src/lib/mlxcel-xla/src/emitter/model.rs`**
```rust
// 이전: decode의 모든 인자가 버퍼 뷰
let token = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "token".into());

// 이후: 1-D decode만 세 스칼라를 표시하고, 프롤로그가 텐서를 재구성
let scalar_abi = !c.uses_mrope() && decode_scalar_abi_enabled();
let token = if scalar_abi {
    take_abi_scalar_arg(b, &mut decls, &mut idx, "token".into())
} else { ... };
```

**변경 이유:** IREE가 값을 staging할지 결정하는 것은 시그니처이므로, 인자 타입 자체가 바뀌어야 한다.

**파일: `src/lib/mlxcel-xla/csrc/xla_iree.c`**
```c
// 번들 로드 뒤 프로브, decode_step.kv_import와 같은 형태
iree_status_t s = iree_runtime_call_initialize_by_name(
    c->session, iree_make_cstring_view("decode_step.scalar_inputs"), &probe);
if (iree_status_is_ok(s)) { c->scalar_inputs = 1; iree_runtime_call_deinitialize(&probe); }
else { iree_status_ignore(s); }

// 호출 지점에서 스칼라 경로는 뷰 대신 값을 민다
XLA_CHECK(iree_vm_list_push_value(iree_runtime_call_inputs(&call), &token_value));
```

**변경 이유:** 셰임은 컴파일 시점에 아티팩트 ABI를 알 수 없고, 이 변경 이전에 컴파일된 아티팩트를 위해 버퍼 뷰 경로가 계속 동작해야 한다.

### 4.3 데이터 모델 변경

해당 없음(영속 상태 없음).

---

## 5. 학습 포인트

### 5.1 인자의 타입은 lowering 결정이다

**개념:** `tensor<i32>`와 `i32`는 같은 값을 담지만, 스트림 지향 런타임에서는 텐서 형태만 메모리로 존재해야 한다. 그래프가 그 값을 호스트에서 필요로 하면 컴파일러가 되가져와야 하고, 그 왕복은 실제 할당이다.

**이 PR에서의 적용:** staging 리소스는 그래프 산술의 결함이 아니라 인자 타입의 비용이었다. 타입을 바꾸자 프롤로그 이후 어떤 op도 건드리지 않고 사라졌다.

**일반적인 사용 사례:**
- IREE, CUDA 그래프, 명시적 버퍼 수명을 갖는 런타임 위의 모든 decode·샘플링 그래프
- 인덱스, 모양, 경계로만 쓰이는 그래프 파라미터

**예시 코드:**
```mlir
// staging 없는 프롤로그
func.func public @main(%weights: tensor<...>, %arg146: i32, %arg147: i32, %arg148: i32) {
  %token = tensor.from_elements %arg146 : tensor<i32>
  %slice = stablehlo.dynamic_slice %embed, %token, ...  // 호스트 값, readback 없음
```

### 5.2 capability 마커는 컴파일 번들 형식을 하나로 유지한다

**개념:** 그래프 ABI가 바뀔 때 선택지는 캐시된 아티팩트를 모두 무효화하거나, 호출자가 어떤 ABI를 들고 있는지 물을 수 있게 하는 것이다.

**이 PR에서의 적용:** 이름으로 찾는 no-op 엔트리 하나가 값싼 답이고, 호환되지 않는 아티팩트의 크래시를 런타임 분기로 바꾼다. #3048의 KV 전역 변경도 같은 패턴을 썼다.

**일반적인 사용 사례:**
- IREE 모듈 번들, ONNX 그래프 캐시, 새 호스트가 소비하는 모든 AOT 컴파일 아티팩트

---

## 6. 추가 학습 리소스

### 핵심 키워드
| 키워드 | 설명 | 이 PR에서의 의미 |
|--------|------|------------------|
| `!stream.resource<staging>` | 호스트가 읽어야 하는 값을 담는 IREE의 호스트 가시 버퍼 클래스 | 이 PR이 제거한 할당 |
| `stream.resource.load` | 그 버퍼를 읽는 op | 단계당 3회에서 0회로 |
| `tensor.from_elements` | 스칼라 값으로 텐서를 만드는 op | 프롤로그에서 rank-0 피연산자 재구성 |
| `iree_vm_list_push_value` | 스칼라 VM 값을 호출에 넣는 API | 셰임의 새 입력 경로 |

### 관련 기술/프레임워크
- **IREE**: XLA 백엔드가 대상으로 하는 컴파일러와 런타임. 공식 문서: https://iree.dev/

### 관련 PR/이슈
- 이슈 #3060: 이 PR이 닫는 성능 이슈.
- PR #3048: 이 PR이 잇는 모듈 전역 KV 캐시이며 프로브 패턴을 공유한다.
- PR #3031: 앞선 호스트 churn을 없앤 영속 매핑 decode 스칼라.
- PR #3078: #3048 ABI에 맞춰 decode 골든을 재생성했다. 이 PR의 골든이 충돌한 이유이며 rebase로 해결했다.

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|------|-----|
| 변경 파일 | 22 |
| 추가 줄 | +5357 |
| 삭제 줄 | -4915 |
| 추가 테스트 | 2 |

줄 수는 재생성된 decode 골든이 대부분을 차지한다.

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|----------|-----|------|
| 성능 | 3 | 이미터 ABI 스칼라, 셰임 프로브와 푸시 경로, M-RoPE 가드 |
| 코드 품질 | 2 | 이미터 단위 테스트, 두 ABI를 모두 실행하는 디바이스 게이트 |
| 문서 | 2 | `src/lib/mlxcel-xla/README.md` 절, 이 리포트 |
| 테스트 | 15 | `assets/*/decode.mlir` 골든 재생성 |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|--------|
| `3f9279b3` | perf | pass the decode scalars as ABI i32 arguments |
| `9246d11a` | perf | (main 기준 rebase) pass the decode scalars as ABI i32 arguments |
| `385ad4a3` | docs | document the decode scalar ABI and marker |

---

## 8. 후속 조치

### 완료 필요
- [ ] 없음.

### 모니터링 필요
- 다음 전체 `gfx1151` 스윕에서 decode 행이 `6e2e6810` 대비 측정된 약 2.4%만큼 이동해야 한다.

### 향후 개선 사항
- 스칼라 경로에서 `xla_step_scalars_ensure`를 부수 효과로 부르는 대신 `decode_result` 슬롯만 할당하기.
- M-RoPE 모델이 이 백엔드에서 성능상 중요해지면, 마커가 position 모양을 함께 전달하도록 확장해 그 경로의 staging도 제거하기.
- decode 단계의 HIP 그래프 캡처는 남은 실행 간격의 후보로 남는다.

---

## 부록

### A. 테스트 결과

- `cargo test -p mlxcel-xla --lib`: 417개 통과, 0개 실패, 19개 무시.
- `tests/xla_decode_scalar_reuse.rs -- --ignored`: 4개 통과.
- `tests/xla_greedy_timing.rs -- --ignored`: 3개 통과.
- `cargo fmt --all -- --check`, `cargo clippy -p mlxcel --features xla-iree --lib --tests -- -D warnings`, `cargo clippy -p mlxcel-xla --all-targets -- -D warnings`: 통과.
- `gcc -fsyntax-only -std=gnu11 -DMLXCEL_XLA_TEST_SUPPORT=1 csrc/xla_iree.c`: 통과.

### B. 성능 벤치마크

짝지은 교차 A/B, `gfx1151`, llama-3.2-1b-instruct-4bit, native 양자화, 용량 512, n=128, 6회:

| 회차 | 이전 (tok/s) | 이후 (tok/s) |
|------|--------------|--------------|
| 1 | 198.55 | 200.64 |
| 2 | 195.51 | 199.87 |
| 3 | 196.46 | 197.89 |
| 4 | 195.61 | 198.39 |
| 5 | 190.17 | 205.10 |
| 6 | 192.84 | 212.58 |
| 중앙값 | 195.56 | 200.25 |

### C. 참고 자료
- 이슈 #3060
- IREE stream dialect 문서: https://iree.dev/developers/design-docs/streaming/
