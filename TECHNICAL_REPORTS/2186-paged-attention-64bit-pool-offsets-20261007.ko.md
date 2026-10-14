# 기술 보고서: PR #2186 - 퓨즈드 Paged-Attention 커널에서 64비트 풀 오프셋 사용

**날짜**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 헤드 `c3fef346`(origin/main `9c0ae2e9` 기준 최신), PR 열림, 머지 대기. #2153을 닫음.

**언어**: C++(`turbo/paged_attention*.cpp`와 `paged_attention_hip.h`에 내장된 Metal, CUDA, HIP 커널 소스, merge 런처), Rust(mlxcel-core `paged_v2/plan.rs`, 새 `cache/paged_pool_offset_tests.rs`, `plan_tests.rs`), Markdown(`docs/environment-variables.md`)

**위험도**: 낮음. 각 커널 본문에서 선언 하나만 바뀝니다. 풀 베이스가 64비트가 되고 `row`가 곱셈 전에 넓혀집니다. 템플릿 인자, 그리드 형태, JIT 이름, dtype 키는 그대로입니다. 새 거부 조건 두 개(merge 런처와 plan 검증)는 partial 워크스페이스가 실제로 도달하는 크기보다 훨씬 큰 경우에만 적용되며, 둘 다 배치를 기존 gather 경로로 보냅니다.

## 요약

퓨즈드 paged-attention 커널은 토큰별 K/V 풀 오프셋 `(row * page + slot) * stride_kv + kv_head * dim`을 여섯 본문 모두에서 32비트 부호 없는 산술로 계산했습니다. 여섯 본문은 v1 decode와 v2 partial이며, 각각 Metal, CUDA, HIP용이 있습니다. 이 커널이 주소를 매기는 풀은 한 레이어의 side별 슬랩 하나이고, 서버는 그 슬랩을 `ceil(ctx / 32) * batch` 블록으로 잡습니다. 슬랩이 2^32개를 넘는 원소를 담으면 오프셋이 감싸져(wrap) 커널이 같은 버퍼의 다른 행을 읽습니다. 폴트도 오류도 없고, 어텐션 출력만 틀립니다. Llama-3.1-8B(KV 헤드 8, 헤드 차원 128)에서 한계는 `[32, 8, 128]` 블록 131,072개이며, 128K 컨텍스트에 `--parallel 32`가 정확히 이 값에 도달하고 그보다 크면 감싸집니다.

PR #2186은 각 본문의 베이스를 `ulong`(Metal) 또는 `uint64_t`(CUDA, HIP)로 넓히고, 곱셈 전에 `row`를 캐스트합니다. Sparse decode는 v2 partial 위에서 실행되므로 별도 변경 없이 같은 수정을 받습니다. 역시 32비트로 인덱싱하는 merge 커널은 넓히지 않았습니다. 대신 런처가 `UINT32_MAX`개를 넘는 입력이나 출력을 거부하고, `PagedDecodePlan::validate`가 그런 plan을 실행 전에 거절하므로 배치 경로가 패닉하지 않고 gather로 폴백합니다.

GPU 테스트는 블록 131,073개(2^32 + 32,768 원소)의 실제 f16 풀을 만들고, 마지막 블록에만 K/V를 쓴 뒤 v1과 v2를 호스트 어텐션과 비교합니다. main의 본문에서는 두 커널 모두 행 0(0으로 채워짐)을 읽어 최대 오차 0.456으로 실패하고, 이 PR에서는 통과합니다. 소스 고정 테스트는 본문 하나만 되돌려도 실패합니다. gfx1151의 서버 decode 속도는 양쪽 모두 중앙값 14.0 tok/s로 변화가 없습니다. Metal과 CUDA 줄은 리뷰만 했고 컴파일하거나 실행하지는 않았습니다. 테스트를 작성하면서 별개의 결함 두 개를 발견했고, #2184와 #2187로 등록했습니다.

## 1. 문제 정의

### 1.1 오프셋이 감싸지던 곳

모든 퓨즈드 본문은 풀 안에서 한 토큰의 K/V 벡터 시작 위치를 계산합니다.

```
uint base = (row * block_size + slot) * stride_kv + kv_head * dim;   // v1, Metal
uint32_t base = (row * page_size + entry) * stride_kv + kv_head * dim; // v2, CUDA/HIP
```

`row`는 풀 행(블록) 인덱스, `stride_kv`는 `Hkv * D`이며, 읽기는 `k_pool[base + d]`와 `v_pool[base + d]`입니다. 모든 항이 32비트였으므로 곱이 인덱싱 전에 2^32를 법으로 감싸졌습니다. 영향받은 여섯 줄은 다음과 같습니다.

| 백엔드 | v1 decode | v2 partial |
|---|---|---|
| Metal | `paged_attention.cpp`(`PAGED_ATTENTION_DECODE_SOURCE`) | `paged_attention_v2.cpp`(`PAGED_ATTENTION_V2_PARTIAL_SOURCE`) |
| CUDA | `paged_attention.cpp`(`..._CUDA_SOURCE`) | `paged_attention_v2.cpp`(`..._CUDA_SOURCE`) |
| HIP | `paged_attention_hip.h`(`PAGED_ATTENTION_DECODE_HIP_SOURCE`) | `paged_attention_hip.h`(`PAGED_ATTENTION_V2_PARTIAL_HIP_SOURCE`) |

HIP 본문은 CUDA 산술을 그대로 옮긴 #2103에서 들어왔고, 이 결함은 그 리뷰 중에 발견되었습니다.

### 1.2 프로덕션 서버가 여기에 도달하는 이유

한계는 모델 단위가 아니라 레이어 슬랩 단위입니다. `resolve_paged_slab_blocks`는 슬랩을 `ceil(per_slot_ctx / block_size) * batch`로 잡고, 블록 예산이 설정된 경우에만 레이어별 몫으로 상한을 둡니다. `MLXCEL_PAGED_SLAB_BLOCKS`는 그대로 사용됩니다. 감싸지는 지점은 `slab_blocks * block_size * Hkv * D > 2^32`입니다. Llama-3.1-8B에서는 32토큰 블록 131,072개로, f16 기준 레이어당 side별 약 8 GiB입니다. 메모리가 큰 호스트가 긴 컨텍스트를 높은 `--parallel`로 서빙하면 이를 넘으며, 원소 수를 보는 런처 검사는 없었습니다. 호스트 검사는 rank와 축 1~3만 확인했습니다.

감싸지면 행 `r`의 토큰은 (이 기하 구조에서) 행 `r - 131072`에서 읽힙니다. 출력은 다른 요청의 키와 값, 또는 0에 대한 그럴듯한 어텐션 결과입니다. 로그로는 구분할 수 없습니다.

### 1.3 Sparse decode와 merge

- **Sparse decode**는 dense `[B, H, Cap, D]` 할당을 `[B*H*Cap, 1, 1, D]` 풀로 reshape하고 페이지 크기 1로 v2 partial을 실행하므로, `B * H * Cap * D > 2^32`에서 감싸집니다. 별도 커널 변경이 필요 없고, v2 partial을 고치면 함께 고쳐집니다.
- **Merge**(`paged_attention_v2_merge.cpp`와 HIP merge 본문)는 `v_in[(i * heads + h) * dim + d]`를 32비트로 인덱싱합니다. 입력은 f32 partial 워크스페이스 `[num_chunks, Hq, D]`로 풀보다 훨씬 작지만, 보호 장치가 없었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 퓨즈드 본문 6개(Metal, CUDA, HIP; v1과 v2 partial) | `T base = ((T)row * page + slot) * (T)stride_kv + kv_head * dim;`, `T`는 `ulong` 또는 `uint64_t`. #2153을 언급하는 한 줄 주석 추가 |
| `paged_attention_v2_merge.cpp` | `v_in.size()` 또는 `num_outputs * heads * dim`이 `UINT32_MAX`를 넘으면 런처가 `paged_attention_merge_states`를 명시한 `std::invalid_argument`를 던짐. `<cstdint>` 추가 |
| `paged_v2/plan.rs` | `workspace_partial_v_elems()`가 `u32::MAX`를 넘으면 `PagedDecodePlan::validate`가 `Err` 반환 |
| `paged_v2/plan_tests.rs` | `validate_rejects_a_merge_input_past_the_u32_index_range`: 1페이지 청크 65,535개에서 `Hq * D = 64 * 1024`는 통과, `128 * 1024`는 거절 |
| `cache/paged_pool_offset_tests.rs`(신규) | 소스 고정 테스트, merge 거부 테스트, ignored 2^32 풀 테스트 |
| `cache/paged_batch_decode.rs` | 새 테스트 모듈 등록 |
| `docs/environment-variables.md` | `MLXCEL_PAGED_SLAB_BLOCKS` 행에 side별 2^32 원소를 넘는 슬랩이 지원된다고 명시(#2153) |

커밋 2개: 수정과 테스트(`e1184cf2`), 그리고 주석과 문서만 다듬은 리뷰 후속 커밋(`c3fef346`). 파일 9개, 388줄 추가, 7줄 삭제.

## 3. 설계

### 3.1 모든 인덱스가 아니라 주소만 넓히기

이슈는 풀 베이스만 넓히기로 정했습니다. 루프 카운터, `block_idx`, `slot`, `row`, 그리고 `rows`/`indices` 입력은 32비트로 남습니다. 행 수는 2^31보다 훨씬 작고, 32비트를 넘는 것은 최종 원소 오프셋뿐입니다. 캐스트는 곱셈 전에 `row`에 적용됩니다.

```
ulong base = ((ulong)row * block_size + slot) * (ulong)stride_kv + kv_head * dim;
```

`(ulong)(row * block_size)`는 곱셈이 먼저 32비트로 일어나므로 여전히 감싸집니다. `stride_kv`도 캐스트하면 컴파일러의 승격 순서와 관계없이 두 번째 곱셈이 64비트로 유지됩니다. `k_pool[base + d]`와 `v_pool[base + d]` 읽기는 그대로이며 이제 64비트 값을 씁니다.

같은 풀에 쓰는 `fused_norm.cpp`와 `fused_rope_append.cpp`는 이미 같은 산술에 `ulong`과 `uint64_t`를 쓰고 있었으므로, paged 본문이 이들과 맞춰졌습니다.

### 3.2 템플릿 인자와 dtype 키는 그대로

소스 텍스트는 바뀌지만 템플릿 인자, 그리드 형태, JIT 이름, dtype 키는 바뀌지 않습니다. `make verify-kernel-dtype-keys`는 여전히 범위 내 9개를 보고합니다. 소스 기반 JIT 캐시(업스트림 CUDA, #2181 이후 ROCm)에서는 새 소스가 자동으로 새 모듈로 컴파일되고, Metal에서는 소스가 라이브러리 빌드에 포함됩니다. 이슈가 요구한 대로 세 백엔드는 타입 표기만 다르고 산술은 동일합니다.

### 3.3 merge는 넓히지 않고 거부

merge 입력은 f32 원소 `num_chunks * Hq * D`개입니다. 2^32에 도달하려면 예컨대 청크 65,535개에 `Hq * D`가 65,536을 넘어야 하는데, 현재 어떤 모델과 plan도 이를 만들지 않습니다. merge 본문을 넓히면 일어나지 않는 크기를 위해 커널 세 개를 더 고쳐야 하므로, PR은 `paged_attention_merge_states`의 기존 검사 옆에 호스트 측 거부를 추가했습니다. `std::invalid_argument`를 던지며, 이는 기존 `Result` 브리지를 통해 Rust에서 `Err`가 됩니다. 검사는 shape만 읽으므로 실행 전에 이루어집니다.

### 3.4 plan 검사, 그리고 이슈와 다른 이유

이슈는 호출자가 거부된 실행을 "gather로 폴백"으로 처리한다고 가정했습니다. 거부를 outcome으로 보고하는 sparse decode에는 맞지만 배치 경로에는 맞지 않습니다. `cache/paged_batch_decode.rs`의 `paged_batch_decode_attention`은 `PagedBlockPool::paged_decode_batched`의 `Err`에서 패닉하는데, 그곳의 오류는 장부가 불일치한다는 뜻이라는 판단 때문입니다. 따라서 merge 거부가 `Err`로 올라오면 서버가 성능 저하가 아니라 크래시를 겪습니다.

PR은 이 틈을 한 단계 위에서 막습니다. `PagedDecodePlan::validate`는 이제 partial 워크스페이스가 `u32::MAX` 원소를 넘으면 `Err`를 반환합니다. `cache/paged.rs`의 풀 decode 경로는 실행 전에 `validate`를 호출하고, 그 `Err`를 실행 없음(`Ok(None)`)과 함께 `PlanRejected`로 기록하며, 배치 호출자는 이를 `gather_fallback`으로 처리합니다. 그래서 과대 배치는 어떤 커널도 실행되기 전에 gather로 보내지고, sparse 경로도 자체 `validate` 호출로 같은 방식으로 거절합니다. merge 런처 검사는 plan을 거치지 않는 호출자를 위한 마지막 방어선으로 남습니다.

### 3.5 기각안: 넓히지 않고 큰 풀을 거부

풀 커널에 호스트 측 거부를 두는 편이 더 작은 변경이었겠지만, 퓨즈드 경로가 존재하는 이유인 긴 컨텍스트와 큰 배치 형태를 gather로 되돌리게 됩니다. 메모리 대역폭에 묶인 커널에서 레인별 토큰당 64비트 곱셈 하나는 측정 가능한 비용이 없습니다(4.3절).

## 4. 검증

### 4.1 실제 2^32 풀 테스트

`paged_pool_past_u32_elements_matches_gather`(`#[ignore]`, 약 17 GiB)는 f16 풀 `[131073, 32, 8, 128]`을 만듭니다. 원소는 2^32 + 32,768개이므로 마지막 블록의 모든 원소가 감싸지는 지점 뒤에 있습니다. 마지막 블록만 데이터를 갖고 나머지는 0입니다. 테스트는 먼저 풀 자체가 올바른지 확인합니다. 마지막 블록은 쓴 값이 그대로 읽히고 행 0은 모두 0입니다. 그다음 마지막 행 하나만을 블록으로 갖는 시퀀스에 대해 v1 decode와 v2 partial(청크 1개, merge 없음)을 실행하고, 같은 f16 반올림 값으로 호스트에서 f64로 계산한 어텐션과 허용 오차 1e-3으로 비교합니다.

- **main의 본문**: 두 커널 모두 최대 오차 0.456으로 실패합니다. 행 131,072의 베이스는 `131072 * 32 * 1024 = 2^32`이고 0으로 감싸지므로, 커널이 행 0을 읽어 0을 반환합니다. 0.456은 참조값의 최댓값입니다.
- **이 PR**: 둘 다 통과합니다. 실행은 `rocm_gpu_guard.sh` 구간 안에서 했습니다.

풀은 0 텐서 하나가 아니라 0으로 채운 절반 두 개와 블록을 이어 붙여 만듭니다. ROCm에서 2^32개 이상 원소의 strided copy가 실행되지 않기 때문입니다(#2184, 아래 참조). 테스트는 이 한계를 피해 대상 커널만 측정합니다.

### 4.2 빠른 테스트

- `every_fused_body_computes_the_pool_base_in_64_bits`는 세 소스 파일을 `include_str!`로 읽어 모든 `<type> base = ...;` 선언을 찾고, 파일마다 정확히 두 개를 기대하며, 곱셈 전 `row` 캐스트를 포함한 타입과 식 전체를 확인합니다. 어떤 호스트에서도 실행되는데, 세 백엔드를 모두 실행할 수 있는 호스트가 없으므로 중요합니다. HIP v2 본문 하나만 되돌리면 실패하며 해당 파일을 지목합니다.
- `merge_refuses_inputs_or_outputs_past_u32_elements`는 2^32 + 1,024 원소의 평가되지 않은 broadcast(입력 케이스)와 과대 출력 수를 넘기므로 큰 메모리를 할당하지 않으며, 두 `Err` 메시지를 확인합니다.
- `validate_rejects_a_merge_input_past_the_u32_index_range`는 plan 경계의 양쪽을 확인합니다.
- `cargo test -p mlxcel-core --features rocm --lib -- paged`: 264개 통과. lib과 테스트의 clippy 깨끗함. 스크립트 게이트와 `verify-rocm-overlay` 통과.

### 4.3 Decode 성능

Llama-3.1-8B-Instruct-4bit, `--parallel 4 --ctx-size 131072` 서버, 약 16K 프롬프트 토큰의 클라이언트 4개, 출력 128토큰, 양쪽 모두 fused v2(로그에 `fused v2 launch` 표시). main의 커널 본문을 쓴 빌드(이 브랜치에서 커널 파일 세 개를 되돌린 것)와 이 PR을 하나의 `rocm_gpu_guard.sh` 구간 안에서 짝으로 세 번 실행했고, 모두 깨끗했습니다. 요청별 decode tok/s는 이전 14.0, 13.9, 14.0, 이후 14.1, 14.0, 14.0입니다. 중앙값은 둘 다 14.0이므로 회귀가 없습니다.

이슈의 수용 기준은 `examples/paged_attention_kernel_bench`를 지목했습니다. 이 벤치는 32블록 기본 슬랩을 유지하므로, 배치 4와 16K 토큰에서는 퓨즈드 커널이 아니라 gather 경로를 측정합니다. 그래서 사용하지 않았습니다. 서버 실행은 실제 슬랩에서 변경된 커널을 거칩니다.

### 4.4 오케스트레이터 게이트

오케스트레이터가 헤드 `c3fef346`(origin/main `9c0ae2e9` 기준 최신)에서 실행한 `make verify-rocm`은 `verify-test-rocm`을 제외한 모든 단계를 통과했습니다. 이 단계는 11,984개 통과, 5개 실패, 383개 ignored였습니다. 실패 5개는 #2187로 추적 중인 #2182의 알려진 테스트와 정확히 일치합니다.

- `scheduler_completion_snapshot_tests::model_owned_sequence_primes_the_decode_lookahead_only_with_a_rewind`
- `scheduler_model_owned_lookahead_tests` 4개

이 테스트들은 이 PR의 파일을 main으로 되돌려도 똑같이 실패하고 단독으로 실행하면 통과하므로, 이번 변경과 무관합니다.

### 4.5 검증하지 못한 것

호스트에 Metal과 CUDA가 없습니다. 변경된 네 줄(Metal v1과 v2 partial, CUDA v1과 v2 partial)은 리뷰만 했고 컴파일하거나 실행하지 않았습니다. 소스 고정 테스트는 텍스트를 확인할 뿐, 컴파일되거나 올바른 결과를 내는지는 확인하지 않습니다. 이 규모에서 해당 백엔드에 의존하기 전에, 통합 메모리 20 GiB 이상의 Metal 호스트와 CUDA 장치에서 2^32 풀 테스트를 실행해야 합니다.

## 5. 기술적 선택과 그 이유

- **본문마다 식 하나만 넓히기.** 결함은 주소 계산에만 있습니다. 모든 것을 넓히면 2^31 근처에도 가지 않는 인덱스에 비용과 변경만 늘어납니다.
- **곱셈 전에 캐스트.** 곱셈 뒤에 적용한 캐스트는 맞아 보이지만 여전히 감싸지므로, 고정 테스트가 `((T)row * ...)` 형태를 정확히 확인합니다.
- **JIT 키와 템플릿 인자 유지.** 디스패치, 오토튜닝, `verify-kernel-dtype-keys`에서 변경이 보이지 않습니다.
- **merge는 넓히지 않고 거부.** merge 워크스페이스는 현실적으로 한계에 도달할 수 없습니다. 가드는 비교 한 번이고 커널 세 개를 더 고치지 않아도 됩니다.
- **plan 검증에서 거절.** 배치 경로는 실행 오류에서 패닉하므로, 런처 안에서만 거부하면 과대 배치가 크래시가 됩니다. `validate`에서 거절하면 결과가 gather 폴백으로 유지됩니다.
- **실제 과대 풀로 테스트.** 합성이나 모의 검사로는 64비트 인덱스가 올바른 메모리에 닿는지 보여줄 수 없습니다. 테스트는 `#[ignore]` 뒤에서 17 GiB를 한 번 써서 주소를 증명합니다.
- **모든 호스트에서 소스 고정.** 이 호스트가 실행할 수 없는 네 개를 포함해 여섯 본문을 모두 덮는 검사는 고정 테스트뿐입니다.
- **커널 벤치가 아니라 서버로 측정.** 벤치의 기본 슬랩은 gather를 측정했을 것입니다.

## 6. 남은 위험과 후속 작업

- **Metal과 CUDA 미실행.** 4.5절 참조. 고정 패턴과 일치하면서 틀린 오타는 가능성이 낮지만, 대규모에서의 컴파일과 실행 동작은 확인되지 않았습니다.
- **2^32 원소를 넘는 ROCm strided copy(#2184).** 테스트 풀을 만드는 중, ROCm에서 2^32개 이상 원소의 strided MLX copy가 2^32 스레드 그리드를 요청했고 HIP가 `hipErrorInvalidConfiguration`으로 거부했습니다. 이는 paged 커널이 아니라 MLX의 ROCm copy 커널의 한계이지만, ROCm에서 과대 슬랩 전체를 복사하는 다른 경로는 여전히 실패할 수 있다는 뜻입니다. #2184로 등록했습니다.
- **Gemma 3 lookahead 테스트의 순서 의존성(#2187).** 전체 ROCm 테스트 실행의 `server::batch::scheduler` 실패 5개는 #2182에서 왔고, main에서도 똑같이 실패하며 단독으로는 통과합니다. #2187로 등록했습니다.
- **merge는 32비트로 남음.** 향후 plan이나 모델이 `num_chunks * Hq * D`를 2^32 넘게 밀어 올리면 그 배치는 실패하지 않고 gather로 갑니다. 정확성이 아니라 성능 절벽이며, merge 본문을 넓히면 사라집니다.
- **paged 계열의 다른 32비트 인덱싱.** 이 PR은 이슈가 지목한 여섯 퓨즈드 본문과 merge를 다룹니다. 그 밖의 커널은 여기서 감사하지 않았습니다.

## 7. 학습 포인트

- **조용한 감싸기는 인덱스의 가장 나쁜 실패 형태입니다.** 커널은 잘못된 행에서 그럴듯한 숫자를 돌려주었습니다. 감싸지는 지점 뒤에 데이터를 두고 독립적인 참조와 비교하는 테스트만이 이를 볼 수 있습니다.
- **한계는 모델이 아니라 커널이 주소를 매기는 버퍼 단위입니다.** 슬랩 크기는 `--ctx-size`와 `--parallel`에 따라 정해지므로, 파라미터 기준으로 작은 모델도 한 레이어의 풀에서 2^32 원소를 넘을 수 있습니다.
- **가드가 기대는 폴백이 실제로 있는지 확인하세요.** 이슈는 모든 호출자가 실행 `Err`를 폴백으로 처리한다고 가정했지만, 배치 경로는 패닉합니다. 거절을 plan 검증으로 옮겨 가드가 크래시를 만들지 않게 했습니다.
- **커널을 탓하기 전에 픽스처를 확인하세요.** 풀 테스트는 커널을 실행하기 전에 마지막 블록과 행 0이 기대한 값인지 확인하며, 그 덕분에 #2184를 커널 실패로 오인하지 않고 찾아냈습니다.
- **변경된 경로를 실제로 실행하는 벤치마크를 쓰세요.** 지목된 커널 벤치는 깨끗하지만 무관한 숫자를 냈을 것입니다.
