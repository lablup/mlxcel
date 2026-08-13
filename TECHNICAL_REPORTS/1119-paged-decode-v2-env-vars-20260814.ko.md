# 기술 보고서: PR #1119 - docs: add the six paged-decode v2 variables and correct the NATIVE row

**작성일**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Markdown
**위험도**: Low

---

## 요약

PR #1119는 이슈 #1104를 해결한다. `docs/environment-variables.md`는 `MLXCEL_*` 변수의 색인을 자처하지만, Rust에서 실제로 읽는 paged-decode v2 변수 여섯 개가 빠져 있었다. 게다가 유일하게 실려 있던 paged-attention 행은 `README.md`와 `docs/CONTINUOUS_BATCHING.md`가 말하는 것과 정반대를 운영자에게 알려주고 있었다.

여섯 변수를 이슈 본문이 아니라 정의 지점의 코드에서 읽은 기본값과 함께 추가했고, `MLXCEL_PAGED_ATTENTION_NATIVE` 행은 두 소비자를 모두 설명하도록 다시 썼다. 문서만 변경, 파일 하나, 동작 변경 없음.

---

## 1. 문제 정의

### 1.1 배경

이슈 #899가 fused paged-decode v2 커널을 pool 기반 continuous batching의 프로덕션 decode 경로로 승격시켰다. 이 승격으로 기존 변수 하나에 두 번째 소비자가 생기고 새 변수 다섯 개가 도입되었지만, 레퍼런스 페이지는 함께 갱신되지 않았다. 그 결과 레퍼런스 문서가 가질 수 있는 최악의 실패 형태가 남았다. 빈칸이 아니라 확신에 찬 오답이다.

### 1.2 기존 문제점

- **Rust에서 읽는 변수 여섯 개가 전부 미등재.** `MLXCEL_PAGED_ATTENTION_V2`, `MLXCEL_PAGED_DECODE_V2_CHUNK`, `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS`, `MLXCEL_PAGED_SLAB_BLOCKS`, `MLXCEL_PAGED_V2_MIN_KV_TOKENS`, `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST`. 이 중 뒤의 셋은 이미 `docs/CONTINUOUS_BATCHING.md`에 운영자용 knob으로 문서화되어 있었고, 같은 기능 영역의 형제 변수인 `MLXCEL_CASCADE_ATTENTION`은 이미 이 페이지에 실려 있었다. 이 패턴은 내부 전용 변수를 의도적으로 감춘 것이 아니라 단순 drift임을 보여준다.
- **다른 문서 두 개와 모순되는 행.** `MLXCEL_PAGED_ATTENTION_NATIVE` 행은 이 변수가 "a control for external mlxcel-core consumers and `examples/paged_attention_kernel_bench.rs`, not a server knob"이라는 문장으로 끝났다. #710이 라이브러리 진입점을 정리하던 시점에는 참이었고 #899 이후에는 거짓이다. `README.md`와 `docs/CONTINUOUS_BATCHING.md`는 둘 다 `MLXCEL_PAGED_ATTENTION_NATIVE=0`을 서버 측 kill switch로 설명한다. 바로 그 kill switch를 찾으려고 레퍼런스 페이지를 연 운영자가 "이 변수는 당신과 무관하다"는 답을 받은 셈이다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| 장애 상황에서 운영자가 v2 decode 경로를 끄지 못함. 레퍼런스 페이지가 서버 knob이 아니라고 말하기 때문 | High | Medium |
| `MLXCEL_PAGED_SLAB_BLOCKS`가 문서화되지 않아 운영자가 pool 크기를 추측으로 잡고, fused 경로가 조용히 도달 불가 상태로 남음 | Medium | Medium |
| diagnostic 표시가 없어 기여자가 진단용 변수 셋을 지원되는 튜닝 knob으로 오해함 | Low | Medium |

---

## 2. 기술적 검토 사항

### 2.1 기본값을 정의 지점에서 읽음

이슈는 일부 기본값을 산문으로 제시했다. 그중 어느 것도 그대로 옮기지 않고 전부 정의 지점에서 다시 읽었다. 아래 3.2의 불일치를 발견한 것이 이 절차 덕분이다.

| 변수 | 문서화된 기본값 | 출처 |
|---|---|---|
| `MLXCEL_PAGED_ATTENTION_V2` | off | `paged_v2/mod.rs` (`V2_ENV`, `parse_v2_enabled`) |
| `MLXCEL_PAGED_DECODE_V2_CHUNK` | unset (계획되었거나 autotune된 chunk 크기) | `autotune/ops/paged_decode_v2_chunk.rs` (`CHUNK_ENV`) |
| `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS` | 유도값: Apple은 `gpu_core_count * 8`에 하한 64, CUDA를 포함한 그 외 호스트는 `512` | `paged_v2/plan.rs` (`device_target_ctas`) |
| `MLXCEL_PAGED_SLAB_BLOCKS` | 유도값 `ceil(per_slot_ctx / block_size) * batch`, 32블록 pool 기본값을 하한으로, 레이어당 예산 몫을 상한으로 | `execution/memory_estimate.rs` (`resolve_paged_slab_blocks`), `cache/paged.rs` (`POOL_SLAB_BLOCKS = 32`) |
| `MLXCEL_PAGED_V2_MIN_KV_TOKENS` | `4096` | `paged_v2/dispatch.rs` (`MIN_SINGLE_REQUEST_KV_TOKENS`) |
| `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST` | `512` | `paged_v2/dispatch.rs` (`MIN_BATCHED_KV_TOKENS_PER_REQUEST`) |

### 2.2 다시 쓴 NATIVE 행

교체된 행은 `src/lib/mlxcel-core/src/layers.rs`의 `resolve_dispatch_decision`과 `resolve_paged_v2_dispatch`를 근거로 작성했다. 두 소비자 구조가 실제로 존재하는 곳이 거기다. 새 행은 소비자 둘을 모두 명시하고, #710의 정리를 현재 진술이 아니라 이력으로 남기며, 두 소비자 모두에서 override가 selector보다 먼저 검사되지만 강제 dispatch도 커널의 구조적 거부(단일 slab 레이어, 서비스 가능한 geometry, 비어 있지 않은 batch)는 그대로 통과해야 한다는 점을 적었다. Default 셀은 이제 한 칸이 두 개의 dispatch 정책을 설명하므로, 라이브러리 selector의 영역뿐 아니라 서버 쪽 token floor까지 포함하도록 넓혔다.

### 2.3 범위 통제

파일 하나, +26/-1. `README.md`와 `docs/CONTINUOUS_BATCHING.md`는 이미 올바른 동작을 서술하고 있었으므로 의도적으로 건드리지 않았다. 둘과 어긋나 있던 쪽이 레퍼런스 페이지였고, 나머지 둘을 고쳤다면 불일치를 해소하는 대신 옮기기만 했을 것이다.

---

## 3. 기술적 선택과 그 이유

### 3.1 여섯 개를 한 절이 아니라 두 절로 분리

| 옵션 | 장점 | 단점 |
|---|---|---|
| 여섯 개를 새 절 하나에 모음 | 기능 영역이 한곳에 모임 | 커널 개발용 스위치 셋을 운영자 앞에 튜닝 knob처럼 내놓게 됨 |
| **선택: 셋은 새 운영자용 절, 셋은 기존 diagnostic 절에 Diagnostic 표시** | 절의 독자와 대상이 일치하고, 행만 따로 읽혀도 diagnostic 표시가 남음 | 한 페이지 안에서 기능 영역이 두 곳에 서술됨 |

새 `## Paged decode v2 variables` 절은 `MLXCEL_PAGED_SLAB_BLOCKS`, `MLXCEL_PAGED_V2_MIN_KV_TOKENS`, `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST`를 담고 KV cache 절 뒤에 배치했다. `MLXCEL_PAGED_ATTENTION_V2`, `MLXCEL_PAGED_DECODE_V2_CHUNK`, `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS`는 `## Hardware and kernel diagnostic variables`로 보내고 각각 **Diagnostic**을 앞에 붙였다. 분리된 두 절은 `MLXCEL_PAGED_ATTENTION_NATIVE`를 kill switch로 지목하고 diagnostic 절을 anchor로 링크하는 도입부로 다시 연결했다.

특히 `MLXCEL_PAGED_ATTENTION_V2`에는 이 표시가 필요했다. 이 변수는 이슈 #898이 라이브러리 전용 v1 진입점을 비교하려고 둔 gate이며, 서버의 프로덕션 v2 decode는 전혀 gate하지 않는다. 그 사실을 명시하지 않으면 이름만 보고 기능 전체의 마스터 스위치로 읽히기 쉽다.

### 3.2 경고 문구가 아니라 실제 동작을 문서화

`src/execution/memory_estimate.rs`의 `resolve_paged_slab_blocks`는 파싱 결과로 분기한다. `Ok(0)`은 `None`을 반환하고, `Ok(n)`은 `Some(n)`을, `Err(_)`은 `"... is not a non-negative integer; using the derived slab size"`를 경고로 남긴 뒤 역시 `None`을 반환한다. `None`은 override 없음을 뜻하고, pool은 `POOL_SLAB_BLOCKS = 32`로 되돌아간다. 즉 파싱 불가 값은 `0`과 같은 경로를 타고 과거의 32블록 기본값에 안착하며, 경고가 말하는 유도값(derived size)에는 도달하지 않는다.

해당 행은 경고가 뭐라고 말하는지가 아니라 실제로 무슨 일이 일어나는지를 적었다("a value that is not a non-negative integer warns and leaves that same 32-block default in place"). 경고 문자열을 고치는 것은 코드 변경이라 문서 이슈의 범위 밖이었다. 5절 참조.

### 3.3 중복 서술 대신 상호 링크

dispatch 정책, 결과별 startup 로그 라인, fused 커널이 거부하는 형태 목록은 모두 `docs/CONTINUOUS_BATCHING.md`에 이미 쓰여 있다. 새 절은 이를 다시 쓰는 대신 `CONTINUOUS_BATCHING.md#seeing-which-path-ran`을 링크했다. 이슈가 지시한 방식이기도 하다. 가이드를 복제한 레퍼런스 페이지는 자기 몫의 drift를 따로 갖게 되는데, 그것이 바로 이 PR이 고치고 있던 실패 양상이다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 1 |
| 추가된 라인 | +26 |
| 삭제된 라인 | -1 |
| 새로 문서화한 변수 | 6 |
| 다시 쓴 행 | 1 |

### 영역별 변경

| 영역 | 파일 | 주요 내용 |
|---|---|---|
| 운영자 레퍼런스 | `docs/environment-variables.md` | 새 `## Paged decode v2 variables` 절에 운영자용 knob 셋, kill switch를 지목하는 도입부, continuous batching 가이드 상호 링크 |
| 진단 레퍼런스 | `docs/environment-variables.md` | `## Hardware and kernel diagnostic variables`에 행 셋 추가, 각각 **Diagnostic** 표시 |
| 정확성 | `docs/environment-variables.md` | `MLXCEL_PAGED_ATTENTION_NATIVE` 행을 두 소비자 서술로 재작성, #710은 이력으로 유지, Default 셀을 서버 floor까지 포함하도록 확장 |

### 관련 커밋

| Hash | Type | Message |
|---|---|---|
| `cf4e22cd` | docs | docs: add the six paged-decode v2 variables and correct the NATIVE row |

---

## 5. 검증 및 후속 조치

### 통과

- 모든 기본값을 이슈 본문이 아니라 정의 지점에서 확인(2.1 표).
- `python3 scripts/ci/check_cross_repo_refs.py` 통과.
- 추가한 행이 편입된 절의 4열 구조를 유지하는지 확인.
- `git diff --stat` 결과가 `docs/environment-variables.md` 하나뿐이므로 빌드나 테스트 표면과 무관.

### 발견했으나 의도적으로 고치지 않은 것

- **`docs/turbo-kv-cache.md`에 같은 낡은 서술이 남아 있다.** 298행에서 312행 부근에 여전히 #710이 pooled 진입점을 정리했으므로 "`MLXCEL_PAGED_ATTENTION_NATIVE` is a control for external mlxcel-core consumers and the kernel bench, not a server knob"이라고 적혀 있다. 이 PR이 레퍼런스 페이지에서 바로잡은 pre-#899 주장과 동일하다. 이슈 #1104는 파일 하나로 범위를 잡았으므로 두 번째 지점은 별도 후속 작업이다.
- **`MLXCEL_PAGED_SLAB_BLOCKS`의 경고 문구 자체가 코드에서 틀렸다.** 3.2에서 추적했듯 `Err(_)` 갈래는 "using the derived slab size"라고 경고한 뒤 `None`을 반환하며, 실제로는 32블록 pool 기본값이 유지된다. 메시지 수정은 Rust 변경이라 이번 범위 밖이다.

### 후속 후보

- 이슈가 제안한 CI 검사: `src/**/*.rs`의 `MLXCEL_*` 등장과 `docs/environment-variables.md` 표를 대조하는 스크립트로, `scripts/ci/check_kernel_dtype_keys.py`를 본뜬 형태. 이 PR은 오늘의 간극을 수작업으로 메웠을 뿐이고, 다음 기능이 같은 간극을 다시 여는 것을 막는 장치는 없다.
- `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS`는 CUDA에서 미검증이라고 문서에 명시했다. CUDA의 고정값 `512`는 SM 수에서 유도하는 편이 옳을 가능성이 높다. 해당 행은 이 상수를 보정된 값처럼 제시하지 않고 있는 그대로 적었다.
