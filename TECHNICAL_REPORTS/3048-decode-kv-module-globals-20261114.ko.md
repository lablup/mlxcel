# 기술 리포트: PR #3048 - 모듈 전역으로 decode KV 캐시를 제자리 갱신

**작성일**: 2026-11-14
**상태**: 완료
**언어**: Rust, C
**위험도**: 보통

---

## 요약

단일 시퀀스 OpenXLA decode 단계가 KV 캐시를 함수의 입력이자 출력으로 다뤘기 때문에, IREE는 매 단계 16 MiB 캐시 두 개를 복사해야 했다. 캐시를 모듈 전역 두 개(`util.global private mutable`)로 옮기면 단계가 제자리에서 갱신하므로 복사가 사라지고, 디스패치 수와 수치는 그대로이며, prefill 결과를 전역에 한 번 넣어 주는 `kv_import` 엔트리가 새로 생긴다. `gfx1151` 측정에서 decode 속도는 179.78에서 196.06 tok/s로 올랐고(+9.05%, 짝지은 6회 모두 개선), greedy 출력은 동일하다.

---

## 1. 문제 정의

### 1.1 배경

XLA 백엔드는 MLX 백엔드와 칩별 벤치마크 프로토콜을 동일하게 유지해 두 백엔드를 직접 비교한다. #3028의 호스트 측 작업(영속 매핑 스칼라, 디바이스 간 토큰 readback) 이후 `gfx1151`에서 MLX와의 차이는 약 160 대 239 tok/s, 단계 시간은 5.9 ms 대 MLX의 약 4.2 ms였다. 커널 트레이스로 보면 이 단계는 GPU 커널 시간 4.32 ms와 유휴 1.51 ms로 나뉘고, 유휴 쪽은 다시 측정된 구성이 있었다. 309에서 312회의 디스패치에서 오는 0.79 ms의 실행 간격, 163 µs `hipHostFree` 옆의 0.35 ms 단계 시작 정지, 0.23 ms의 KV 복사, 그 밖의 0.14 ms다.

### 1.2 기존 문제점

- **문제 1**: decode 모듈이 캐시 텐서 두 개를 결과로 반환했고, 그래서 매 단계 16,777,216 바이트 `stream.cmd.copy`가 두 번 발생했다. 4비트 1B decode가 그 밖에는 이동하지 않는 32 MiB의 디바이스 트래픽이다.
- **문제 2**: 셰임은 매 단계 새 캐시 버퍼 뷰 쌍을 채택하고 이전 쌍을 해제해야 했는데, 이는 복사를 위해 존재하는 부기일 뿐이다.
- **문제 3**: 작은 elementwise와 reduction 사슬에 디스패치 309개가 붙어 있었고, 각각 약 2.5 µs의 실행 지연이 든다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|-------------|
| 제자리 갱신이 수치나 토큰 출력을 바꾼다 | 높음 | 낮음 (비트 동일성 확인) |
| 세션을 재사용하는 요청 사이로 모듈 상태가 새어 나간다 | 높음 | 보통 (import 플래그로 차단) |
| 전역을 쓰지 않는 그래프가 새 ABI에서 깨진다 | 높음 | 낮음 (capability 프로브가 기존 경로 유지) |
| aggressive fusion 플래그가 prefill 동작을 바꾼다 | 보통 | 없음 (decode 컴파일 전용) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

새 입력도, 호스트에 노출되는 표면도 없고, 모델 로딩이나 요청 파싱의 변경도 없다. 모듈 전역은 모듈 내부에만 있고 호스트에서 접근할 수 없다.

**체크리스트:**
- [x] 입력 검증 (변경 없음. 그래프 자체의 인덱스 계산도 변경 없음)
- [x] 인증/인가 (해당 없음)
- [x] 데이터 암호화 (해당 없음)
- [x] 로깅 (오류 경로 하나 추가, 모델 데이터를 출력하지 않음)

**발견 사항:** 없음.

### 2.2 성능 관점

| 항목 | 이전 | 이후 | 개선 |
|------|------|------|------|
| Decode 속도 (n=128, 용량 512) | 179.78 tok/s | 196.06 tok/s | +9.05% |
| Decode 단계 | 5.56 ms | 5.10 ms | -0.46 ms |
| Decode flow 디스패치 | 309 | 277 | -32 |
| 단계당 KV 캐시 복사 | 2 x 16 MiB | 0 | -32 MiB |

복사 제거와 fusion이 각각 기여한다. 이 PR은 두 요소를 분리한 A/B를 하지 않았으므로 9.05%는 합산 수치다.

### 2.3 호환성/의존성 관점

- **호환성 깨짐**: Rust API 수준에서는 없다. 컴파일된 아티팩트의 ABI가 바뀌므로 셰임이 가정하지 않고 `decode_step.kv_import`를 프로브한다.
- **새 의존성**: 없음.
- **호환 범위**: ragged 배치 엔진, prefill, 임베딩 입력 prefill, deepstack prefill, Gemma3n의 고정 스케줄 CUDA decode는 모두 기존 전체 캐시 인자 ABI를 유지한다.

### 2.4 코드 품질 관점

- **테스트 범위**: 이미터 단위 테스트 하나와 무시 표시된 디바이스 게이트 하나(새 세션, 두 번째 세션, 재-prefill).
- **코드 복잡도**: 이미터 헬퍼 두 개(`global_load`, `global_store`)와 셰임 헬퍼 하나. decode 호출 경로에 `c->kv_globals` 분기 두 개가 늘었다.
- **기술 부채**: `src/server/batch/xla_worker_tests.rs`의 기존 clippy 오류를 함께 고쳤다. 그 파일은 `xla-iree`에서만 컴파일되므로 기본 기능 clippy 작업은 보지 못한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 행 반환 ABI 대신 변경 가능한 모듈 전역

**컨텍스트:** 캐시가 ABI를 통해 오가는 것을 멈추되, 셰임은 prefill 결과가 decode 그래프에 도달해야 한다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 기록된 행만 반환하고 셰임에서 scatter | 전송량 작음, 모듈 상태 없음 | 레이어당 약 3 디스패치(약 120 µs)가 복사 약 130 µs와 맞먹어 이득이 사실상 없고, rank-5 ABI와 scatter 커널이 필요 |
| `--iree-stream-resource-alias-mutable-bindings` | 코드 변경 없음 | IR이 그대로임(측정) |
| **선택: `util.global private mutable` 두 개** | 복사 0, 추가 디스패치 0, 같은 연산 | 모듈 상태, 시드 엔트리 필요 |

**선택 이유:** IREE는 전역 읽기-갱신-저장을 하나의 영속 `!stream.resource<variable>` 버퍼에 대한 읽기-쓰기 바인딩으로 낮춘다. 갱신이 제자리이므로 복사할 것도, 더할 것도 없다.

**트레이드오프:** 그래프가 더 이상 인자만의 순수 함수가 아니므로, 같은 세션을 쓰는 두 번째 소비자가 다른 시퀀스를 끼워 넣으면 안 된다. 단일 시퀀스 decode 경로는 이미 셰임에 컨텍스트당 캐시 쌍 하나를 두고 있었으므로, 이는 기존 제약을 다시 적은 것이다.

### 3.2 프로브한 엔트리로 시드하기

**컨텍스트:** private 전역은 호스트에서 보이지 않으므로 prefill 결과를 넣을 엔트리가 필요하다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 번들 버전이나 캐시 키 올리기 | 명시적 | 기존 컴파일 캐시를 모두 무효화 |
| **선택: `@kv_import`와 프로브** | 기존 아티팩트가 계속 동작, 셰임이 런타임에 판단 | 세션당 프로브 한 번 |

**선택 이유:** `decode_step.kv_import`의 존재 자체가 셰임에 필요한 capability이고, 번들을 붙인 뒤 한 번 검사하면 충분하다. 그것이 없는 그래프는 기존 인자 ABI를 쓰는데, 그 경로는 원래 있던 코드다.

**트레이드오프:** 한 함수 안에 두 ABI가 산다. 비용은 decode 호출의 `if (!c->kv_globals)` 분기 두 개이고, 위험은 미래의 변경이 한쪽만 고치는 것이다. 디바이스 게이트가 전역 쪽을, ragged/prefill 단정이 다른 쪽을 덮는다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[이전]
prefill  ->  kcache, vcache (결과)
decode   ->  (kcache, vcache) -> (token, kcache', vcache')   단계당 복사 2회

[이후]
prefill   -> kcache, vcache (결과)
kv_import -> globals @kcache, @vcache                        prefill당 한 번
decode    -> () -> token                                     전역을 제자리 갱신
```

### 4.2 주요 코드 변경

**파일: `src/lib/mlxcel-xla/src/emitter/model.rs`**
```rust
// 이전
let (decls, a) = build_arg_schema(&mut b, c);
...
"{alias}module @decode_step {{\n{native}  func.func public @main({sig}) -> ({out_ty}, {cache_ty}, {cache_ty}) {{ ... }}"
```

```rust
// 이후
let (mut decls, mut a) = build_arg_schema(&mut b, c);
decls.truncate(decls.len() - 2);
a.kcache = b.global_load("kcache", cache_ty.clone());
a.vcache = b.global_load("vcache", cache_ty.clone());
...
"util.global private mutable @kcache : {cache_ty_s}" ...
"func.func public @kv_import(%k: {cache_ty_s}, %v: {cache_ty_s}) { util.global.store %k, @kcache ... }"
```

**변경 이유:** 캐시가 인자 목록과 결과 목록에서 빠지고, 단계는 모듈 전역을 읽고 쓴다.

**파일: `src/lib/mlxcel-xla/csrc/xla_iree.c`**
```c
// 이후: 번들 로드 뒤 capability 프로브
iree_status_t s = iree_runtime_call_initialize_by_name(
    c->session, iree_make_cstring_view("decode_step.kv_import"), &probe);
if (iree_status_is_ok(s)) { c->kv_globals = 1; iree_runtime_call_deinitialize(&probe); }
```

**변경 이유:** 셰임이 세션마다 그래프가 캐시를 전역에 두는지 판단하고, 한 번짜리 import는 prefill 경계에서만 실행된다.

### 4.3 데이터 모델 변경

해당 없음(영속 상태 없음).

---

## 5. 학습 포인트

### 5.1 함수형 캐시 시맨틱은 그래프의 요구가 아니라 lowering의 결정이다

**개념:** 텐서를 입력으로 받아 갱신된 복사본을 반환하는 그래프는 순수 함수지만, 스트리밍 런타임에서는 그 순수성이 실제 복사로 낮춰진다.

**이 PR에서의 적용:** 그래프는 이미 정확히 한 행만 썼다. 복사는 계산이 아니라 시그니처 때문에 존재했다. 상태 모델을 바꾸자 연산을 하나도 건드리지 않고 단계당 32 MiB 트래픽이 사라졌다.

**일반적인 사용 사례:**
- 스트림 지향 런타임 위의 모든 decode/KV 캐시 그래프
- 명시적 버퍼 수명을 갖는 런타임(IREE/HAL, CUDA 그래프)에서 컴파일러가 증명하지 못하면 aliasing은 곧 복사가 되는 경우

**예시 코드:**
```mlir
// 복사 없는 형태
%k = util.global.load @kcache : tensor<16x512x8x64xf32>
%k2 = ...현재 위치에서 갱신...
util.global.store %k2, @kcache : tensor<16x512x8x64xf32>
```

---

## 6. 추가 학습 리소스

### 핵심 키워드
| 키워드 | 설명 | 이 PR에서의 의미 |
|--------|------|------------------|
| `util.global private mutable` | MLIR 모듈 수준 변경 가능 상태, 호스트 비노출 | 복사를 없애는 기제 |
| `!stream.resource<variable>` | 모듈 전역용 영속 버퍼 클래스 | 갱신이 제자리인 이유 |
| `--iree-dispatch-creation-enable-aggressive-fusion` | 작은 elementwise 디스패치를 합치는 패스 | 309에서 277 디스패치로 |

### 관련 기술/프레임워크
- **IREE**: XLA 백엔드가 대상으로 하는 컴파일러와 런타임. 공식 문서: https://iree.dev/

### 관련 PR/이슈
- 이슈 #3043: 이 PR이 닫는 성능 이슈.
- PR #3042: decode 수치를 비교 가능하게 만든 단계별 `--profile` 집계.
- PR #3031: 이 PR이 딛고 선 호스트 측 스칼라 작업.

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|------|-----|
| 변경 파일 | 8 |
| 추가 줄 | +275 |
| 삭제 줄 | -27 |
| 추가 테스트 | 2 |

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|----------|-----|------|
| 성능 | 4 | 이미터 전역, 셰임 import 경로, decode 전용 fusion 플래그, 체인지로그 항목 |
| 코드 품질 | 2 | 이미터 단위 테스트, 디바이스 게이트 |
| 문서 | 1 | CHANGELOG `Changed` 항목 |
| 코드 품질 | 1 | `xla_worker_tests.rs`의 기존 clippy 수정 |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|--------|
| `cd0c026` | perf | update the decode KV cache in place through module globals (#3043) |

---

## 8. 후속 조치

### 완료 필요
- [ ] 없음.

### 모니터링 필요
- 다음 전체 스윕 뒤 ROCm 벤치마크 페이지의 `gfx1151` 행. 측정된 약 9%만큼 이동해야 한다.

### 향후 개선 사항
- 호출당 192바이트 고정 staging 할당이 남아 있다(약 0.35 ms의 GPU 유휴). `--iree-stream-resource-memory-model=unified`는 IR을 바이트 단위로 동일하게 만들고 `--iree-stream-resource-alias-mutable-bindings`는 IR을 바꾸지 않으므로, 이를 없애려면 읽기 인덱스가 정적인 그래프나 디바이스 측 토큰 스캔이 필요하다.
- decode 단계의 HIP 그래프 캡처는 남은 실행 간격의 후보로 남는다.

---

## 부록

### A. 테스트 결과

- `cargo test -p mlxcel --features xla-iree,test-utils --test xla_greedy_timing -- --ignored`: 3개 통과.
- 이미터 단위 테스트 `single_decode_threads_kv_through_module_globals`: 통과.
- `cargo fmt --all -- --check`: 통과.
- `cargo clippy -p mlxcel --features xla-iree --lib --tests -- -D warnings`: 통과.
- `cargo test -p mlxcel-core --profile test-fast --lib --no-run`: 링크 성공.

### B. 성능 벤치마크

짝지은 교차 A/B, `gfx1151`, llama-3.2-1b-instruct-4bit, native quant, 용량 512, n=128, 6회:

| 회차 | 이전 (tok/s) | 이후 (tok/s) |
|------|--------------|--------------|
| 1 | 184.97 | 193.49 |
| 2 | 173.26 | 198.17 |
| 3 | 179.49 | 193.94 |
| 4 | 182.33 | 198.23 |
| 5 | 180.07 | 198.61 |
| 6 | 177.21 | 193.41 |
| 중앙값 | 179.78 | 196.06 |

### C. 참고 자료
- 이슈 #3043
- IREE stream dialect 문서: https://iree.dev/developers/design-docs/streaming/
