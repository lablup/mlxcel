# PR #2936: ragged 배치 엔진 잔여 항목 강화

**작성일**: 2026-11-10
**상태**: 완료 (#2923 종료)

## 요약

이 PR은 #469 감사에서 나온 여섯 개의 ragged continuous-batching 잔여 항목을 묶은 #2923을 종료한다. 지연 할당되던 rank-5 배치 KV는 이제 `calloc` 호스트 미러와 업로드 대신 device fill로 0으로 채운다. M-RoPE decode 좌표는 IREE 호출 전에 컴파일된 context capacity로 상한 검사를 하고, Rust 래퍼와 C 진입점 양쪽에서 검사한다. ragged reset은 로드된 그래프가 컴파일되지 않은 배치 크기를 거부한다. prefill KV 슬롯은 K와 V 전송이 모두 끝난 뒤에만 valid로 표시되고, decode는 invalid 슬롯을 가진 활성 row를 거부한다. `examples/xla_batch_bench.rs`에는 bucket, seeded sampling, cancellation 모드가 추가되어 각자 PASS 라인을 출력한다. 시작 시에는 로드 후 살아남은 bucket 집합이 image-capable 형상을 모두 잃으면 경고한다. 실제 device에서 `#[ignore]` suite가 4/4 통과했고, bench는 새 모드 세 가지를 포함해 REFERENCE-EXACT를 출력하며, FFI 변경은 추가만 이루어졌다. 15개 파일, +1421/-88.

## 1. 문제 정의

#469 강화 작업은 이 PR을 막지 않을 만큼 작지만 실제 결함인 여섯 항목을 남겼다. 모두 튜닝 기회가 아니라 결함 부류다.

- **호스트 KV 미러**: `xla_ensure_batch_kv_inner`가 `calloc`으로 0을 채운 캐시를 만들어 업로드했고, 첫 admit마다 버킷당 `4*B*L*S*nkv*d` 호스트 바이트와 전체 H2D 전송을 소비했다. 이 버퍼는 호스트에서 읽히지 않는다.
- **M-RoPE 좌표 상한**: 두 decode 경로가 `cache_len + rope_delta`를 계산해 `[context_capacity, rot]` 테이블을 인덱싱하면서 상한 검사가 없었다. capacity와 같은 좌표는 그래프가 보고할 수 있는 오류가 아니라 device out-of-bounds read다. 음수 쪽은 검사되었지만 양수 쪽은 아니었다.
- **reset과 컴파일된 B**: `xla_llama_ragged_reset`은 어떤 `bsz`든 상태를 초기화해서, 불일치가 나중에 어느 값이 틀렸는지 알 수 없는 IREE shape 실패로 드러났다.
- **비원자적 슬롯 저장**: prefill 슬롯 저장이 K를 옮기고 V를 옮겼다. V 전송이 실패하면 K가 남아 있고 decode는 반쯤 쓰인 슬롯을 valid로 읽었다.
- **bench 커버리지**: `xla_batch_bench`는 greedy sampling의 `XlaBatchEngine`만 실행해서 bucket 집합 경로, seeded sampling, cancellation이 device에서 검증되지 않았다.
- **시작 시 image guard 범위**: 로드 전 image guard는 요청한 capacity를 판단했다. 유일한 image-capable bucket이 로드에 실패하면 경고 없이 image 요청이 admission에서 거부되었다.

## 2. 변경 사항

### 2.1 device에서 배치 KV 0 채우기 (`d9bbabdb`)

`xla_alloc_zeroed_bv`는 device-local 버퍼를 할당하고 4바이트 0 패턴으로 `iree_hal_device_queue_fill`을 수행한 뒤, fill의 세마포어를 기다린 다음에야 buffer view를 공개한다. `xla_ensure_batch_kv_inner`는 이를 K와 V 캐시 모두에 사용하고, 둘 중 하나라도 할당에 실패하면 두 view를 모두 해제한다. 테스트 전용 카운터가 지연 경로가 소비한 호스트 바이트와 H2D 바이트를 기록한다.

세마포어 대기가 핵심이다. fill은 동기 호출이 아니라 큐에 들어가고, 캐시를 바인딩하는 첫 decode 전에 fill 결과가 보여야 한다. 헬퍼가 반환 전에 기다리므로 대기되지 않은 fill이 새어 나갈 수 없다.

### 2.2 M-RoPE 좌표 상한 (`4b7cd7ca`, device 게이트 `cf6d3a0d`, C 가드 `7ba1d6a2`)

`mrope_decode_coordinate`와 `mrope_slot_coordinates`는 IREE 호출 전에 typed `MropeCoordinateError::OutOfRange { coordinate, context_capacity }`(그리고 `Negative`)를 반환한다. context capacity는 `prepared.rs`와 `iree.rs`를 통해 전달되어, 상수가 아닌 컴파일된 capacity로 검사한다.

C shim은 ragged decode validator와 단일 시퀀스 `xla_llama_decode`/`xla_llama_decode_mrope` 진입점에서 상한 검사를 반복하고 포인터 인자도 NULL 검사한다. Rust 래퍼만이 유일한 가드일 수 없는 이유는 C 호출자가 그 진입점에 직접 닿기 때문이며, 단일 시퀀스 추가는 ragged 전용 가드가 남겼을 우회로를 닫는다.

### 2.3 컴파일된 B에 묶인 ragged reset (`7cb73243`)

로드는 이제 reset 전에 추가된 `xla_llama_ragged_bind(ctx, compiled_b)`를 호출한다. `xla_llama_ragged_reset`은 바인딩되지 않은 context와 `bsz != rg_compiled_b`를 두 값을 명시해 거부한다. 기존 FFI 시그니처는 바뀌지 않아 spike와 `iree-ffi` 호출자는 영향을 받지 않는다.

### 2.4 상태로서의 슬롯 유효성 (`42d807b5`, 단언 수정 `9d907220`)

슬롯 저장은 K 전송 전에 `rg_valid[slot]`을 0으로 지우고, 두 전송이 모두 성공한 뒤에만 설정한다. `xla_llama_decode_ragged_impl`은 invalid 슬롯을 가진 활성 row를 `row=%d slot=%d token=%d cache_len=%d is active but its KV slot is not valid`로 거부한다. 실패 시 K 바이트는 롤백하지 않는다. 엔진이 슬롯을 다시 admit하기 전에 항상 re-prefill하기 때문이고, 반쯤 쓰인 상태를 관측 불가능하게 만드는 것은 플래그다.

### 2.5 bench 모드 (`0e0dadc8`)

- `--buckets c1,c2,...`는 capacity마다 라우팅 커버리지 요청 하나로 `XlaBucketSet`을 구동하고 `BUCKETS REFERENCE-EXACT PASS`를 출력한다.
- `--seed N`에 `--temperature`, `--top-p`를 주면 독립된 seeded 엔진 두 개를 실행해 결정성을 확인하고 `top_k=1`이 greedy 참조와 같은지 확인해 `SEEDED SAMPLING PASS`를 출력한다.
- `--cancel N`은 첫 토큰 뒤에 admit된 요청 N개를 취소하고 이후 이벤트가 없어야 한다는 조건으로 `CANCEL PASS`를 출력한다.

bucket 집합이 bucket마다 id 네임스페이스를 나누므로 제출 인덱스는 id가 아니다. 요청 id는 `HashMap<u64, usize>`로 매핑한다. bench는 불일치 시 0이 아닌 값으로 종료하므로 그 자체가 게이트다.

### 2.6 잃어버린 image bucket에 대한 시작 경고 (`26fb5055`, 문서 `31787aee`)

`xla_image_surviving_bucket_warning(&engine.capacities(), &capacities, image_floor)`는 요청 capacity와 생존 capacity 목록에 대한 순수 함수다. `XlaBucketSet::load` 뒤에 요청에 image-capable bucket이 있었고, floor 이상으로 살아남은 bucket이 없고, 실제로 탈락이 있었을 때 경고한다. 메시지는 탈락한 capacity, 가장 큰 생존 capacity, floor, `MLXCEL_XLA_CONTEXT_CAPACITY`를 담는다. `docs/environment-variables.md`가 탈락 동작을 문서화한다.

### 2.7 리뷰 커밋

- `a93ca23a`는 `tests/xla_ragged_hardening.rs`를 `xla-iree,test-utils`로 게이트해서, CI가 실행하는 일반 `--all-targets` 컴파일이 test-support 전용 import를 해석하지 않게 한다.
- `a4b112f1`은 빌드 내부 `MLXCEL_XLA_TEST_SUPPORT` define을 `scripts/ci/check_env_var_docs.py` allowlist에 넣는다. 운영자 환경 변수가 아니라 C shim용 `build.define`이기 때문이다.
- `7ba1d6a2`는 2.2에 설명한 단일 시퀀스 C 가드다.
- `31787aee`는 bucket 탈락 경고를 문서화한다.

## 3. 기술적 선택과 그 이유

### 3.1 호스트 미러 대신 device fill

대안은 `calloc`과 업로드를 유지하는 것, device 버퍼를 호스트에 매핑해 `memset`하는 것, 그래프 안에서 캐시를 0으로 채우는 것이었다. fill은 호스트 메모리를 늘리지 않고 전체 업로드 대신 버킷당 fill 한 번을 큐에 넣는다. 매핑은 이 변경이 없애려는 호스트 경로를 다시 만지고, 그래프 내 0 채우기는 캐시를 바인딩하는 모든 그래프에 dispatch를 추가한다. 대가는 세마포어와 명시적 대기이며, 런타임이 별도 진입점 호출 사이의 작업 순서를 보장하지 않으므로 여기서는 이것이 올바른 순서 기본 요소다.

### 3.2 Rust typed 오류와 C 반복 검사

C에서 상한을 반복하는 것은 안전 경계에서 의도된 중복이다. Rust 래퍼에만 의존하면 직접 C API가 무방비가 되고, 좌표를 clamp하면 실패 대신 조용히 틀린 위치로 decode하게 된다. typed 오류는 좌표와 capacity를 함께 담아 호출자가 무엇이 넘쳤는지 정확히 보고할 수 있게 한다.

### 3.3 reset 시그니처 변경 대신 추가 bind

`xla_llama_ragged_reset(bsz)`가 컴파일된 B도 받도록 바꾸면 안전 이득 없이 기존 호출자를 모두 깨뜨린다. reset은 비교 대상인 요청 `bsz`가 여전히 필요하다. 별도 bind가 로드 시 context 상태를 한 번 기록해 ABI를 안정적으로 유지하고, 미바인딩 상태를 검출 가능하게 만든다.

### 3.4 먼저 무효화하고 마지막에 공개

K 전송 전에 유효성 플래그를 지우면 모든 실패 구간이 invalid로 관측된다. K와 V 사이에 유효성을 설정하면 K만 있는 상태를 읽을 수 있게 남긴다. 실패 후에 지우는 방법은 실패 자체가 검출 대상이므로 성립하지 않는다. 대가는 실패한 저장이 슬롯을 re-prefill할 때까지 K 바이트를 남기는 것이고, 엔진은 이미 이를 보장한다.

### 3.5 bench 자체를 게이트로

세 모드는 실제 엔진, 작은 가중치, 참조 비교가 필요하다. `#[ignore]` 통합 테스트 세 개를 더하면 example이 이미 가진 fixture를 중복하게 된다. 명시적 PASS 라인과 불일치 시 비영 종료가 CI에 쓸 만한 신호를 주면서, CI가 어차피 실행하지 않을 ignored 테스트를 늘리지 않는다.

## 4. 검증

환경: AMD Strix Halo(`gfx1151`), ROCm, 고정 IREE ROCm 배포 `iree-rocm-3.12.0rc20260721`, device `hip`. 모든 cargo와 GPU 명령은 `scripts/rocm_gpu_guard.sh --hold` 아래에서 `OPENSSL_INCLUDE_DIR`, `OPENSSL_LIB_DIR`, `IREE_ROCM_HOME`을 설정하고 `eval "$(scripts/iree/setup-rocm.sh --env)"`를 적용한 뒤 `CARGO_BUILD_JOBS=8`로 실행했다.

실제 device 강화 suite(`#[ignore]` 게이트):

```
cargo test --release --features xla-iree,test-utils --test xla_ragged_hardening -- --ignored --test-threads=1
```

4 통과, 0 실패. stderr에 슬롯 validator가 `row=0 slot=0 token=7 cache_len=3`을 거부하고 reset이 `bsz=4`를 `compiled_b=8`과 비교해 거부한 줄이 남는다. 게이트는 `mlxcel-xla` unit-test 바이너리가 IREE 런타임을 링크할 수 없어 루트 크레이트의 `tests/xla_prepared_prefill.rs` 관례를 따라 `#[ignore]`로 둔다.

컴파일, lint, 포맷:

```
cargo fmt --all -- --check
cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings
cargo check --features xla-iree --all-targets
```

모두 종료 코드 0. clippy는 vendored IREE C 헤더 경고만 출력한다.

unit 및 tooling 게이트: `decode_coordinate` 3 통과, `mrope_slot_coordinates` 1 통과, `host_preprocessor` 29 통과(1 ignored), `check_env_var_docs.py` 종료 코드 0, python tooling 214 통과.

bench 모드, 생성한 작은 llama 체크포인트 사용(이 호스트에 실제 체크포인트가 없으므로 이것은 처리량이 아니라 참조 일치성을 증명한다):

```
cargo run --release --features xla-iree,test-utils --example xla_batch_bench -- \
  --model /tmp/xla2923-fixture/model --prompts /tmp/xla2923-fixture/prompt_ids.json \
  --device hip --batch 8 --requests 16 --maxcap 48 --buckets 256 --seed 7 --cancel 2
```

`RESULT: REFERENCE-EXACT PASS`, `BUCKETS REFERENCE-EXACT PASS`, `SEEDED SAMPLING PASS (deterministic=true, top_k=1==greedy=true)`, `CANCEL PASS (2/2 cancelled, non-cancelled requests reference-exact=true)`. 같은 플래그를 `--batch 4`로 실행해도 통과했고, `--buckets 256,2048`은 두 capacity를 모두 라우팅했다.

## 5. 학습 포인트

- **HAL fill은 비동기다.** `iree_hal_device_queue_fill`은 작업을 큐에 넣으면 반환한다. 정확성은 buffer view를 공개하기 전에 signal 세마포어를 기다리는 데 달려 있다. fill 패턴 길이는 1, 2, 4, 8바이트여야 하므로 f32 KV를 4바이트 0으로 채우는 것이 자연스럽고 합법적인 선택이다.
- **유효성 플래그는 작은 상태 기계다.** 첫 전송 전에 지우고 마지막 전송 뒤에 설정하는 것이 반쯤 쓰인 버퍼를 검출 가능한 상태로 만든다. 순서를 바꾸면 읽을 수 있는 반쪽 상태가 남는다.
- **제거한 코드 경로를 테스트하려면 계측이 필요하다.** 호스트 미러 테스트는 호스트와 H2D 바이트가 0임을 단언할 수 있는데, 카운터가 제거와 함께 들어왔기 때문이다. 대체된 경로는 항상 그 바이트를 소비했으므로 0이라는 판독은 의미가 있다.
- **순수 술어는 엔진 없이 테스트할 수 있다.** bucket 탈락 경고는 두 capacity 목록에 대한 함수이므로 네 경우 모두 device 테스트가 아니라 단위 테스트다.

## 6. 후속 조치

- 기존 문제로 범위 밖: `xla_llama_prefill`과 `xla_llama_prefill_slot_logits`는 토큰 수는 제한하지만 위치 값을 제한하지 않아, 직접 C 호출자가 범위 밖 prefill 위치를 IREE에 넘길 수 있다. prefill 진입점 네 개를 바꾸면 이 PR이 넓어지므로 후속 이슈로 분리해야 한다.
- #2922에서: B=4 ragged decode는 native multi-row GEMV에서 f16 경로보다 약 5퍼센트 느린데, 커널이 B와 무관하게 8개 고정 accumulator를 쓰기 때문이다.
- #2921에서: KV read convert와 context matmul이 step당 각각 약 103 us이고, tuning matcher가 측정된 batch 8에 고정되어 다른 capacity는 아직 커버되지 않는다.

## 7. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 15 |
| 추가 라인 | +1421 |
| 삭제 라인 | -88 |
| 커밋 | 13 |
| 실제 device 게이트 | 4 통과 |

| 영역 | 파일 | 요약 |
|------|------|------|
| C shim | `csrc/xla_iree.c` | device fill, 좌표 상한, 컴파일 B bind/reset, 슬롯 유효성 |
| Rust 래퍼 | `prepared.rs`, `iree.rs`, `batch.rs` | typed M-RoPE 오류, capacity 전달, 컴파일 B 로드 경로 |
| bench | `examples/xla_batch_bench.rs` | bucket, seeded sampling, cancel 모드 |
| 가드 | `src/server/model_worker.rs`, `src/multimodal/host_preprocessor.rs` | 생존 bucket 경고 |
| 테스트 | `tests/xla_ragged_hardening.rs`, `host_preprocessor_tests.rs` | device 게이트 4개, 경고 단위 테스트 4개 |
| tooling/문서 | `check_env_var_docs.py`, `build.rs`, `docs/environment-variables.md` | allowlist, test-support define, bucket 문서 |
